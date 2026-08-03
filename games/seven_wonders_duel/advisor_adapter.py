"""7 Wonders Duel implementation of the shared :class:`AdvisorAdapter`.

First consumer of ``games.advisor``.  It proves the seam: the host drives the
resumable Rust/Python Gumbel tree through :meth:`open_search` + ``advance``,
ranks by visits, and never learns a card, a wonder, or whose turn it is.

Wire representation
-------------------
A public position is ``{seed, first_player, prefix}`` -- the move history from a
known deal.  Replaying ``new_game(seed, first_player)`` then the ``prefix``
action indices reproduces the exact state, hidden information included, with no
RNG crossing the boundary (the seed fixes every reveal).  This is the honest
MVP wire: exact, trivially serializable, and leak-free.  A future BGA-scrape
adapter that reconstructs a position from *public* observation alone (no seed)
is a separate, genuinely game-specific effort; it implements this same
Protocol and swaps only the codec.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from games.advisor import ActionStats, ActionView, EngineSpec, SearchSnapshot

from .codec import decode_action, legal_action_indices
from .engine import Action, ActionUse, apply_action
from .game import GameState, Phase, new_game
from .search import GumbelMCTS, SearchConfig, state_actor


@dataclass(slots=True)
class _Position:
    """Engine state the host treats as opaque: a materialized :class:`GameState`
    plus a stable identity.  ``seed``/``prefix`` are set on the replay wire;
    ``key`` is set on the scrape wire (which has no seed)."""

    game: GameState
    seed: int | None = None
    first_player: int = 0
    prefix: tuple[int, ...] = ()
    key: str | None = None


def _replay(seed: int, first_player: int, prefix: tuple[int, ...]) -> GameState:
    game = new_game(int(seed), first_player=int(first_player))
    for index in prefix:
        apply_action(game, decode_action(game, int(index)))
    return game


def _card_name_at(game, slot_id) -> str | None:
    """Name of the face-up card in ``slot_id``, or None.

    A slot id is geometry; a human picks a *card*. Every label and field that
    names a slot resolves it here so the advisor talks about "Chamber of
    Commerce" rather than "(6, 5)" -- the extension also needs the name to
    highlight the right card on the board.
    """
    if slot_id is None:
        return None
    card = game.tableau.cards.get(tuple(slot_id))
    if card is None or not card.present or not card.revealed:
        return None
    return card.card_name


def _label(action: Action, game=None) -> str:
    card = _card_name_at(game, action.slot_id) if game is not None else None
    where = card or (f"slot {action.slot_id}" if action.slot_id is not None else "")
    if action.use is ActionUse.DRAFT_WONDER:
        return f"Draft wonder: {action.wonder_name}"
    if action.use is ActionUse.CONSTRUCT_WONDER:
        # Which card a wonder consumes is a real decision in 7WD (it denies the
        # card to the opponent), so name both.
        return f"Wonder: {action.wonder_name} (using {where})"
    if action.use is ActionUse.CONSTRUCT_BUILDING:
        return f"Build: {where}"
    if action.use is ActionUse.DISCARD_FOR_COINS:
        return f"Discard for coins: {where}"
    if action.use is ActionUse.RESOLVE_PENDING_CHOICE:
        return f"Resolve choice: {action.choice}"
    if action.use is ActionUse.CHOOSE_NEXT_START_PLAYER:
        return f"Start next age: player {action.starting_player}"
    return action.use.name


class _ClosedHandle:
    """SearchHandle over a closed-mode Gumbel tree, driven one PUCT sim at a
    time.  Values are converted p0 -> actor frame via ``sign`` here, so the
    host only ever sees the asking player's edge."""

    def __init__(self, mcts: GumbelMCTS, root, actor: int, target: int):
        self._mcts = mcts
        self._root = root
        self._sign = 1.0 if actor == 0 else -1.0
        self._target = int(target)
        self._done = 0

    def advance(self, chunk_sims: int, stop_event) -> SearchSnapshot:
        for _ in range(int(chunk_sims)):
            if stop_event.is_set():
                break
            self._mcts.descend(self._root)
            self._done += 1
        entries = {
            str(edge.action_index): ActionStats(
                visits=int(edge.visits),
                q_value=self._sign * edge.q_p0,
                prior=float(edge.prior),
            )
            for edge in self._root.edges
        }
        return SearchSnapshot(
            sims_done=self._done,
            sims_target=self._target,
            root_value=self._sign * self._root.value_p0,
            entries=entries,
            partial=stop_event.is_set(),
        )

    def close(self) -> None:  # tree is GC'd with the handle
        pass


class _RustClosedHandle:
    """SearchHandle over the Rust resumable PUCT tree.

    Same contract as :class:`_ClosedHandle` -- ``advance`` deepens ONE tree
    across calls -- but the tree, the encoder and the selection all live in
    Rust, which is what the training path uses. Only the leaf evaluation crosses
    back into Python.

    ``leaf_batch > 1`` collects a wave of leaves and evaluates them in one
    forward pass. It also makes the root select under virtual loss, since root
    visits are the advisor's output -- measured at 32, the top action's visit
    share moved 0.941 -> 0.930 with no ranking change, for 3.9x throughput. It
    only pays with the batched adapter: on the scalar bridge every leaf still
    crosses into Python alone and leaf_batch buys nothing (1.00x-1.07x).
    """

    def __init__(self, search, actor: int, target: int):
        self._search = search
        self._sign = 1.0 if actor == 0 else -1.0
        self._target = int(target)

    def advance(self, chunk_sims: int, stop_event) -> SearchSnapshot:
        # Rust only checks the chunk boundary between evaluation requests, so a
        # position whose simulations mostly terminate can overshoot. Harmless
        # for streaming; it means "at least this many", not "exactly".
        if not stop_event.is_set():
            self._search.advance(int(chunk_sims))
        sims_done, root_visits, root_value_sum, _actor, edges = self._search.snapshot()
        entries = {
            str(action_index): ActionStats(
                visits=int(visits),
                q_value=self._sign * (value_sum / visits if visits else 0.0),
                prior=float(prior),
            )
            for action_index, visits, value_sum, prior in edges
        }
        return SearchSnapshot(
            sims_done=int(sims_done),
            sims_target=self._target,
            root_value=self._sign
            * (root_value_sum / root_visits if root_visits else 0.0),
            entries=entries,
            partial=stop_event.is_set(),
        )

    def close(self) -> None:  # the Rust arena is freed with the handle
        self._search = None


class SevenWondersAdvisor:
    """AdvisorAdapter for 7WD.  Pass ``evaluator=`` to inject a preloaded
    evaluator (tests); otherwise checkpoints load lazily from the request."""

    game_id = "seven_wonders_duel"

    def __init__(
        self,
        *,
        evaluator: Any = None,
        default_checkpoint: str | None = None,
        device: str = "cpu",
    ):
        self._injected = evaluator
        self._default_checkpoint = default_checkpoint
        self._device = device
        self._eval_cache: dict[tuple[str | None, str], Any] = {}

    # -- evaluator ----------------------------------------------------------

    def _evaluator(self, req):
        if self._injected is not None:
            return self._injected
        checkpoint = req.checkpoint_path or self._default_checkpoint
        if checkpoint is None:
            raise ValueError("no checkpoint_path supplied and no default set")
        device = req.device or self._device
        key = (checkpoint, device)
        cached = self._eval_cache.get(key)
        if cached is None:
            from .phase_e import load_evaluator

            cached = load_evaluator(checkpoint, device)
            self._eval_cache[key] = cached
        return cached

    # -- state codec --------------------------------------------------------

    def state_from_wire(self, payload: dict[str, Any]) -> _Position:
        if "bga" in payload:
            # Raw BGA capture from the browser extension:
            #   {"bga": <gamedatas>, "args": <gamestate.args>, "dom": {...}}
            # `dom` re-reads the fields BGA never refreshes so no page reload is
            # needed; the mapping stays in Python. Yields the scrape-wire shape
            # handled just below.
            from .bga_extract import wire_from_bga_payload

            payload = wire_from_bga_payload(payload)
        if "observation" in payload:
            from .advisor_scrape import determinize_observation, observation_from_wire

            obs = observation_from_wire(payload["observation"])
            rng = random.Random(int(payload.get("resample_seed", 0)))
            game = determinize_observation(
                obs,
                rng,
                unknown_burial_ages=tuple(
                    int(a) for a in payload.get("unknown_burial_ages", ())
                ),
            )
            digest = hashlib.sha256(
                json.dumps(payload["observation"], sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            return _Position(game=game, first_player=game.first_player, key=f"obs:{digest}")
        seed = int(payload["seed"])
        first_player = int(payload.get("first_player", 0))
        prefix = tuple(int(i) for i in payload.get("prefix", []))
        return _Position(
            game=_replay(seed, first_player, prefix),
            seed=seed,
            first_player=first_player,
            prefix=prefix,
        )

    def state_to_public(self, state: _Position) -> dict[str, Any]:
        game = state.game
        terminal = game.phase is Phase.COMPLETE
        actor = None if terminal else int(state_actor(game))
        observation = game.observation(0)
        cities = [
            {
                "player": player,
                "to_move": actor == player,
                "coins": int(city.coins),
                "wonders": len(city.wonders) + len(city.built_wonders),
                "built_wonders": len(city.built_wonders),
                "buildings": len(city.buildings),
                "science_pairs": len(city.claimed_science_pairs),
            }
            for player, city in enumerate(observation.cities)
        ]
        return {
            "game": self.game_id,
            "origin": "replay" if state.seed is not None else "observation",
            "seed": state.seed,
            "first_player": state.first_player,
            "prefix": list(state.prefix),
            "phase": game.phase.name,
            "age": int(game.age),
            "active_player": int(game.active_player),
            "actor": actor,
            "terminal": terminal,
            "winner": None if game.winner is None else int(game.winner),
            "conflict_position": int(game.conflict_position),
            "cities": cities,
            "legal_actions": [
                {"action_id": v.action_id, "label": v.label, "kind": v.kind}
                for v in self.action_views(state)
            ],
        }

    def state_key(self, state: _Position) -> str:
        if state.key is not None:
            return state.key
        return f"{state.seed}:{state.first_player}:{','.join(map(str, state.prefix))}"

    def action_views(self, state: _Position) -> list[ActionView]:
        game = state.game
        if game.phase is Phase.COMPLETE:
            return []
        views: list[ActionView] = []
        for index in legal_action_indices(game):
            action = decode_action(game, index)
            views.append(
                ActionView(
                    action_id=str(index),  # identity-indexed policy: index IS the id
                    label=_label(action, game),
                    kind=action.use.name,
                    fields={
                        "action_index": int(index),
                        "slot_id": action.slot_id,
                        "card_name": _card_name_at(game, action.slot_id),
                        "wonder_name": action.wonder_name,
                        "choice": action.choice,
                    },
                )
            )
        return views

    # -- search -------------------------------------------------------------

    def _open_rust_search(self, state: _Position, req):
        """A Rust-backed handle, or None if the crate is unavailable.

        Returns None rather than raising so ``search_impl='auto'`` degrades to
        the Python searcher on a machine without the built extension.
        """
        try:
            import seven_wonders_rust

            from .rust_bridge import (
                rust_batched_net_adapter,
                rust_game_from_state,
                rust_scalar_net_adapter,
            )
        except ImportError:
            return None
        if not hasattr(seven_wonders_rust, "RustPuctSearch"):
            return None  # crate predates the resumable handle

        # leaf_batch > 1 collects a wave of leaves and evaluates them in one
        # forward pass. It also makes the root select under virtual loss, which
        # perturbs the visit distribution the advisor reports -- measured at 32,
        # the top action's visit share moved 0.941 -> 0.930 and the ranking did
        # not change, against a 3.9x throughput gain. A batched wave needs the
        # batched adapter: on the scalar bridge, leaf_batch buys nothing.
        leaf_batch = max(1, int(req.options.get("leaf_batch", 16)))
        evaluator = self._evaluator(req)
        adapter = (
            rust_batched_net_adapter(evaluator)
            if leaf_batch > 1
            else rust_scalar_net_adapter(evaluator)
        )
        search = seven_wonders_rust.RustPuctSearch.open(
            rust_game_from_state(state.game),
            adapter,
            int(req.max_sims),
            int(req.seed),
            float(req.options.get("c_puct", 1.5)),
            50.0,
            0.1,
            16,
            leaf_batch,
        )
        return _RustClosedHandle(search, state_actor(state.game), req.max_sims)

    def open_search(self, state: _Position, req):
        engine = "nn" if req.engine in ("auto", "nn") else req.engine
        if engine != "nn":
            raise ValueError(f"unknown engine {req.engine!r}")
        force_expand = bool(req.options.get("force_expand_root_chance", True))

        # search_impl: "rust" (default) | "python" | "auto"
        #
        # The Rust searcher is the same one self-play uses, gated against the
        # Python tree by test_puct_root; measured ~2.5x on a representative
        # position. It cannot force-expand the root chance layer (that needs the
        # F4.5 forced-child cache), so an explicit force_expand_root_chance
        # request falls back to Python rather than silently dropping it -- the
        # two searches would otherwise differ in how root chance edges compute Q.
        impl = str(req.options.get("search_impl", "rust")).lower()
        explicit_force = "force_expand_root_chance" in req.options
        if impl in ("rust", "auto") and not (explicit_force and force_expand):
            handle = self._open_rust_search(state, req)
            if handle is not None:
                return handle
            if impl == "rust":
                raise RuntimeError(
                    "search_impl='rust' but the Rust searcher is unavailable; "
                    "build it with `maturin develop --release`, or pass "
                    "options={'search_impl': 'python'}"
                )

        config = SearchConfig(
            mode="closed",
            seed=int(req.seed),
            c_puct=float(req.options.get("c_puct", 1.5)),
            force_expand_root_chance=force_expand,
        )
        mcts = GumbelMCTS(self._evaluator(req), config)
        root = mcts.make_root(state.game)  # expands root; runs NO sims
        return _ClosedHandle(mcts, root, state_actor(state.game), req.max_sims)

    # -- discovery ----------------------------------------------------------

    def engines(self):
        return {
            "nn": EngineSpec(
                key="nn",
                label="Neural MCTS (Gumbel tree)",
                description="Closed-mode PUCT search on the 7WD net; visits rank moves.",
                needs_checkpoint=True,
                default_sims=800,
                streaming=True,
            )
        }

    def annotators(self):
        from .advisor_endgame import ExactEndgameAnnotator

        return [ExactEndgameAnnotator()]

    def contract(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "engines": list(self.engines()),
            "default_checkpoint": self._default_checkpoint,
            "wire": "seed+first_player+prefix (exact replay)",
        }
