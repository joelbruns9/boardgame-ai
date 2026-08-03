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

from .codec import decode_action, legal_action_indices, pending_choice_name
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
        # Actor-framed like every other number the host renders. The Age is
        # already dealt by the time this is asked, so name the one being started
        # rather than "the next age" -- it is the pyramid on screen.
        if game is None:
            return f"Start next age: player {action.starting_player}"
        if action.starting_player == state_actor(game):
            return f"You start Age {game.age}"
        return f"Opponent starts Age {game.age}"
    return action.use.name


FOLLOW_UP_CHOICES_SHOWN = 3
"""How many options a contingent follow-up names.

Exactly enough to always be usable. The Great Library offers 3 of the 5
off-board tokens, so the draw removes exactly two -- name three and at least one
survives, every time. Naming one is right only 60% of the time (C(4,2)/C(5,3)),
which is what this replaced; two would be 90%.
"""


def _follow_up_label(action_indices, contingent: bool) -> str | None:
    """Render the remainder of a move for the panel.

    Identities come out of the action indices alone (`pending_choice_name`),
    because the child nodes they came from live inside the searcher and are
    never reconstructed here.

    ``contingent`` means the option set is itself random -- the Great Library
    draws its three tokens from five -- so the honest rendering is a preference
    order over the pool rather than a single forced move. Everything else
    (Mausoleum, Zeus, Circus, the science-pair token) chooses from a set that is
    already on the table, where one name is exact.
    """

    names = [
        name
        for name in (pending_choice_name(int(index)) for index in action_indices)
        if name is not None
    ]
    if not names:
        return None
    if not contingent:
        return f"then {names[0]}"
    shown = names[:FOLLOW_UP_CHOICES_SHOWN]
    if len(shown) == 1:
        return f"then {shown[0]}"
    return "then best offered: " + " > ".join(shown)


def _python_follow_up(edge) -> tuple[list[int], bool] | None:
    """The Python searcher's half of the PV walk (`RustPuctSearch.follow_ups`
    is the Rust half), aggregated the same way.

    Pools each option's visits and value across EVERY chance child rather than
    reading the most-sampled one, then ranks by mean value in the chooser's
    frame. Only while a pending choice is open -- an extra turn is a fresh
    decision, not the rest of this move.
    """

    from .search import ChanceKind

    totals: dict[int, list] = {}
    actor = 0
    for child in getattr(edge, "children", {}).values():
        node = child.node
        if node.state.pending_choice is None:
            continue
        actor = node.actor
        for inner in node.edges:
            if not inner.visits:
                continue
            slot = totals.setdefault(inner.action_index, [0, 0.0])
            slot[0] += inner.visits
            slot[1] += inner.value_sum_p0
    if not totals:
        return None
    sign = 1.0 if actor == 0 else -1.0
    ranked = sorted(
        totals.items(), key=lambda item: (-sign * item[1][1] / item[1][0], item[0])
    )
    contingent = any(
        spec.kind is ChanceKind.GREAT_LIBRARY_DRAW for spec in getattr(edge, "specs", ())
    )
    return [index for index, _ in ranked[:5]], contingent


DEFAULT_ARENA_BUDGET_MB = 512
"""How much memory one advisor search may grow its tree to.

The panel asks for an effectively unbounded search -- "keep thinking until the
board changes" -- and on a wide root the closed tree allocates a node per
simulation, each owning a cloned ``GameState``. Measured on a 12-action Age II
turn: node count tracks simulations 1:1 at ~4.4 KB apiece, so 400k sims cost
1.7 GB and a long think froze the machine that reported it.

512 MB is ~120k simulations on such a root, about two minutes of thinking at the
~930 sims/s measured there. That costs nothing in advice: the same position's
root value had converged by ~15k sims and the ranking was stable at every depth.
Narrow roots never come close -- a three-action pending choice reached only
2,589 nodes in 41k sims.
"""

_MEASURED_BYTES_PER_NODE = 4_444
"""Resident bytes per closed-tree node, measured as an RSS delta (so allocator
overhead is included) over 400k simulations. Only the Python searcher needs it:
Rust reports its arena size exactly via ``arena_deep_bytes``."""


def _budget_bytes(req) -> int:
    mb = req.options.get("arena_budget_mb", DEFAULT_ARENA_BUDGET_MB)
    return max(0, int(mb)) * 1024 * 1024


def _budget_reached(used_bytes: int, budget_bytes: int) -> str | None:
    if not budget_bytes or used_bytes < budget_bytes:
        return None
    return (
        f"stopped growing the tree at {used_bytes / 2**20:.0f} MB "
        f"(budget {budget_bytes // 2**20} MB); the numbers shown are final"
    )


class _ArenaBudget:
    """Latching, amortised check that a search's tree has stopped growing.

    Weighing the arena is O(nodes + edges), and the tree gains a node per
    simulation, so measuring on every chunk made the accounting O(N^2) in total
    -- measured at 2.03 ms per scan around 20k nodes, costing ~16% throughput
    over a 30k-simulation run.

    Two things fix that. Node count is O(1) (a `Vec` length), so it gates the
    expensive weighing; and the gate grows with the tree, which makes the scans
    geometric and their total cost O(N) rather than O(N^2). Once tripped the
    answer latches, because a tree that has stopped growing does not shrink.

    A zero budget short-circuits before any of it, so the training and analysis
    paths pay nothing at all.
    """

    __slots__ = ("_budget", "_reason", "_next_scan")

    def __init__(self, budget_bytes: int):
        self._budget = int(budget_bytes)
        self._reason: str | None = None
        self._next_scan = 0

    def reason(self, node_count, weigh) -> str | None:
        if not self._budget or self._reason is not None:
            return self._reason
        nodes = node_count()
        if nodes < self._next_scan:
            return None
        # Re-check after another eighth of the current tree. The overshoot that
        # buys is bounded and reported honestly, since the message carries the
        # size actually measured.
        self._next_scan = nodes + max(2_000, nodes // 8)
        self._reason = _budget_reached(weigh(), self._budget)
        return self._reason


class _ClosedHandle:
    """SearchHandle over a closed-mode Gumbel tree, driven one PUCT sim at a
    time.  Values are converted p0 -> actor frame via ``sign`` here, so the
    host only ever sees the asking player's edge."""

    def __init__(self, mcts: GumbelMCTS, root, actor: int, target: int, budget: int = 0):
        self._mcts = mcts
        self._root = root
        self._sign = 1.0 if actor == 0 else -1.0
        self._target = int(target)
        self._done = 0
        self._budget = _ArenaBudget(budget)

    def _stop_reason(self) -> str | None:
        # Estimated from the node count, since a Python tree cannot be weighed
        # directly; Rust measures its arena instead. Both sides are O(1) here,
        # so the amortisation in _ArenaBudget costs nothing extra.
        nodes = lambda: self._mcts.closed_nodes_created
        return self._budget.reason(nodes, lambda: nodes() * _MEASURED_BYTES_PER_NODE)

    def advance(self, chunk_sims: int, stop_event) -> SearchSnapshot:
        for _ in range(int(chunk_sims)):
            if stop_event.is_set() or self._stop_reason():
                break
            self._mcts.descend(self._root)
            self._done += 1
        entries = {
            str(edge.action_index): ActionStats(
                visits=int(edge.visits),
                q_value=self._sign * edge.q_p0,
                prior=float(edge.prior),
                follow_up=(
                    _follow_up_label(*aggregated)
                    if (aggregated := _python_follow_up(edge)) is not None
                    else None
                ),
            )
            for edge in self._root.edges
        }
        return SearchSnapshot(
            sims_done=self._done,
            sims_target=self._target,
            root_value=self._sign * self._root.value_p0,
            entries=entries,
            partial=stop_event.is_set(),
            stop_reason=self._stop_reason(),
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

    def __init__(self, search, actor: int, target: int, budget: int = 0):
        self._search = search
        self._sign = 1.0 if actor == 0 else -1.0
        self._target = int(target)
        self._budget = _ArenaBudget(budget)

    def _stop_reason(self) -> str | None:
        # arena_nodes is a Vec length; arena_deep_bytes walks every node and its
        # owned vectors, so it runs only when the gate opens.
        return self._budget.reason(
            self._search.arena_nodes, self._search.arena_deep_bytes
        )

    def advance(self, chunk_sims: int, stop_event) -> SearchSnapshot:
        # Rust only checks the chunk boundary between evaluation requests, so a
        # position whose simulations mostly terminate can overshoot. Harmless
        # for streaming; it means "at least this many", not "exactly".
        if not stop_event.is_set() and not self._stop_reason():
            self._search.advance(int(chunk_sims))
        sims_done, root_visits, root_value_sum, _actor, edges = self._search.snapshot()
        follow_ups = {
            int(root_action): (ranked, bool(contingent))
            for root_action, ranked, contingent in self._search.follow_ups()
        }
        entries = {
            str(action_index): ActionStats(
                visits=int(visits),
                q_value=self._sign * (value_sum / visits if visits else 0.0),
                prior=float(prior),
                follow_up=(
                    _follow_up_label(*follow_ups[action_index])
                    if action_index in follow_ups
                    else None
                ),
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
            stop_reason=self._stop_reason(),
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
            "victory_outlook": self._victory_outlook(state),
        }

    # Class order matches `dataset._joint7_class`: my civ/sci/mil, then the
    # opponent's, then draw. Actor-framed, like every other value the host sees.
    _JOINT7_LABELS = (
        "you_civilian",
        "you_scientific",
        "you_military",
        "opponent_civilian",
        "opponent_scientific",
        "opponent_military",
        "draw",
    )

    def _victory_outlook(self, state: _Position) -> dict[str, Any] | None:
        """The net's read of *how* the game ends, not just who wins.

        One root evaluation, no search: these are properties of the position, so
        they are reported here rather than through an annotator (annotators run
        only when a search settles, and a streaming search at max_sims=1,000,000
        never does).

        For 7WD this is often more actionable than the win probability. On the
        captured Age III position the net gave a -1.03 VP margin -- losing on
        points -- alongside a 93% *scientific* win: the game ends before scoring.
        A single number cannot say that.

        Returns None when no evaluator is configured; the caller renders what it
        gets.

        **Also None at the start-player choice**, which is a calibration hole
        rather than a missing feature. Until 2026-08-03 the engine dealt the
        next Age *after* asking who begins it, so every checkpoint trained
        before that -- including the one being served -- only ever saw
        ``CHOOSE_NEXT_START_PLAYER`` with an exhausted tableau. Item F now hands
        it a full pyramid at that phase, which is an input the net has never
        seen.

        The ranked moves survive that, because search plays forward into
        ordinary positions and evaluates those. This does not: it is a single
        raw read of the root, and the aux heads behind it carry 0.2 loss weight
        and no search correction, so its win/type split here would be
        confident-looking noise. Showing nothing beats showing that. A
        checkpoint trained under the corrected ordering can drop this branch.
        """
        game = state.game
        if game.phase is Phase.COMPLETE:
            return None
        if game.phase is Phase.CHOOSE_NEXT_START_PLAYER:
            return None
        evaluator = self._injected
        if evaluator is None:
            if self._default_checkpoint is None:
                return None
            try:
                from .phase_e import load_evaluator

                key = (self._default_checkpoint, self._device)
                evaluator = self._eval_cache.get(key)
                if evaluator is None:
                    evaluator = load_evaluator(self._default_checkpoint, self._device)
                    self._eval_cache[key] = evaluator
            except Exception:
                return None
        try:
            row = evaluator.evaluate_states([game])[0]
        except Exception:
            return None
        joint = [float(p) for p in row.joint7]
        return {
            "victory_type": dict(zip(self._JOINT7_LABELS, joint)),
            "you_win": sum(joint[0:3]),
            "opponent_wins": sum(joint[3:6]),
            "draw": joint[6],
            "wdl": [float(x) for x in row.wdl],
            "vp_margin": float(row.margin),
            "final_military": float(row.military),
            # Trained as distinct symbols / 6, so 1.0 means the sixth symbol --
            # a scientific win. See dataset.py "sci_final my/opp: ... /6".
            "final_science": [float(x) for x in row.science],
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
        return _RustClosedHandle(
            search, state_actor(state.game), req.max_sims, _budget_bytes(req)
        )

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
        return _ClosedHandle(
            mcts, root, state_actor(state.game), req.max_sims, _budget_bytes(req)
        )

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
