"""Welcome To implementation of the shared :class:`~games.advisor.contract.AdvisorAdapter`.

Third consumer of ``games.advisor``, after 7 Wonders Duel.  Nothing about the
transport, the job lifecycle, ranking or the wire envelope is repeated here: the
host owns all of it, and this file supplies only the four things it cannot know
-- how to read a position, how to name a move, how to run the search, and what
the net thinks the game looks like.

WHY THIS ADVISOR EXISTS
-----------------------
Not to play the game for anyone.  ``welcome_to_plan_symptom_diagnosis`` records
that S2 reaches 32.1 points against GreedyBot's 50.8, and a bulk number cannot
say *which* decisions are wrong.  This puts the ranked move list and the net's
own forecasts side by side on a real board, which is the cheapest instrument for
finding that out.  Two consequences for the design:

* the **forecast panel is not decoration**.  The auxiliary heads (final score
  per seat, score by component, plans completed, turns to each plan, which end
  trigger fires) are what the value head is built on, and a policy that looks
  sane on top of a forecast that is plainly wrong localises the fault
  immediately.  They are read once per position, off the raw root evaluation,
  and never change as search deepens.
* every recommendation carries its **prior** as well as its visits, because
  "the net wanted this and search talked it out of it" and "the net never
  considered it" are different diseases.

THE SEARCH ROOT IS A MACRO
--------------------------
``macro_codec`` folds ``CHOOSE_CARDS -> WRITE_NUMBER`` into one action: you pick
a combination *for* a placement.  So at BGA's ``chooseCards`` the advisor ranks
whole (stack, temp modifier, box) moves rather than a stack first and a box
after, which is both the game's real decision and the only representation the
trained policy has.  ``action_id`` is the macro index, and it is stable across
``legal_actions`` calls because it is an index into a fixed vocabulary.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from games.advisor import ActionStats, ActionView, EngineSpec, SearchSnapshot

from games.welcome_to import action_codec as codec
from games.welcome_to import encoder as enc
from games.welcome_to import macro_codec as mc
from games.welcome_to import mcts
from games.welcome_to import network as nw
from games.welcome_to import training
from games.welcome_to.constants import Effect, TEMP_DELTAS
from games.welcome_to.game import GameState
from games.welcome_to.plans import PLANS
from games.welcome_to.snapshot import from_snapshot


@dataclass(slots=True)
class _Position:
    """Engine state plus a stable identity.  Opaque to the host by design."""

    game: GameState
    key: str
    warnings: tuple[str, ...] = ()
    observation: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
_EFFECT_WORDS = {
    Effect.SURVEYOR: "surveyor",
    Effect.ESTATE: "estate agent",
    Effect.PARK: "landscaper",
    Effect.POOL: "pool",
    Effect.TEMP: "temp agency",
    Effect.BIS: "bis",
}

_STREETS = ("1st", "2nd", "3rd")


def _box(x: int, y: int) -> str:
    return "%s street box %d" % (_STREETS[x], y + 1)


def _macro_label(state: GameState, index: int) -> str:
    """A move named the way a player would say it out loud.

    ``macro_codec.describe`` is the debugging form (``write(delta=+1)@2,7``);
    this is the human one, and it has to name the *combination* as well as the
    box because taking a combination is choosing which effect you get.
    """

    if mc.M_WRITE <= index < mc.M_REFUSE:
        slot, delta_slot, x, y = mc.decode_macro_write(index)
        number, effect = state.combination(slot)
        written = number + TEMP_DELTAS[delta_slot]
        modifier = ""
        if TEMP_DELTAS[delta_slot]:
            modifier = " (temp %+d from %d)" % (TEMP_DELTAS[delta_slot], number)
        return "Write %d in %s -- stack %d, %s%s" % (
            written,
            _box(x, y),
            slot + 1,
            _EFFECT_WORDS.get(effect, effect.name.lower()),
            modifier,
        )

    if mc.M_REFUSE <= index < mc.M_DIRECT_REFUSE:
        slot = index - mc.M_REFUSE
        number, effect = state.combination(slot)
        return "Refuse the permit, burning stack %d (%d / %s)" % (
            slot + 1,
            number,
            _EFFECT_WORDS.get(effect, effect.name.lower()),
        )

    if index == mc.M_DIRECT_REFUSE:
        return "Take a permit refusal (nothing is playable)"
    if index == mc.M_ROUNDABOUT_OPEN:
        return "Build a roundabout"

    return _primitive_label(state, mc.to_primitive(index))


def _primitive_label(state: GameState, action: int) -> str:
    if codec.A_ROUNDABOUT_POS <= action < codec.A_WRITE:
        return "Roundabout in %s" % (_box(*codec.decode_roundabout_pos(action)),)
    if codec.A_WRITE <= action < codec.A_SURVEYOR_FENCE:
        delta_slot, x, y = codec.decode_write(action)
        number = (state.ctx.number or 0) + TEMP_DELTAS[delta_slot]
        return "Write %d in %s" % (number, _box(x, y))
    if codec.A_SURVEYOR_FENCE <= action < codec.A_ESTATE_ROW:
        x, j = codec.decode_surveyor_fence(action)
        return "Fence in %s, right of box %d" % (_STREETS[x], j + 1)
    if codec.A_ESTATE_ROW <= action < codec.A_PARK_STREET:
        row = codec.decode_estate_row(action)
        return "Estate value: cross a box in the size-%d row" % (row + 1,)
    if codec.A_PARK_STREET <= action < codec.A_POOL_BUILD:
        return "Build a park in the %s street" % (
            _STREETS[codec.decode_park_street(action)],
        )
    if action == codec.A_POOL_BUILD:
        return "Build the pool"
    if codec.A_BIS <= action < codec.A_CHOOSE_PLAN:
        x, y, side = codec.decode_bis(action)
        return "Bis in %s, copying the %s neighbour" % (
            _box(x, y),
            "left" if side == 0 else "right",
        )
    if codec.A_CHOOSE_PLAN <= action < codec.A_VALIDATE_ESTATE:
        slot = codec.decode_plan(action)
        return "Validate City Plan %d (%s)" % (slot + 1, _plan_name(state, slot))
    if codec.A_VALIDATE_ESTATE <= action < codec.A_PASS_ROUNDABOUT:
        x, y = codec.decode_validate_estate(action)
        return "Hand over the estate starting at %s" % (_box(x, y),)
    if action == codec.A_PERMIT_REFUSAL:
        return "Take a permit refusal"
    if action == codec.A_RESHUFFLE_YES:
        return "Reshuffle the discard back into the deck"
    if action == codec.A_RESHUFFLE_NO:
        return "Leave the deck as it is"
    if action in (
        codec.A_PASS_ROUNDABOUT,
        codec.A_PASS_SURVEYOR,
        codec.A_PASS_ESTATE,
        codec.A_PASS_PARK,
        codec.A_PASS_POOL,
        codec.A_PASS_BIS,
        codec.A_PASS_PLAN,
    ):
        return "Pass"
    return codec.describe(action)


def _plan_name(state: GameState, slot: int) -> str:
    plan = PLANS[state.plan_ids[slot]]
    kind = plan.kind.name.replace("_", " ").lower()
    if plan.params:
        return "%s %s" % (kind, list(plan.params))
    return kind


def _macro_fields(state: GameState, index: int) -> dict[str, Any]:
    """The structured payload the panel highlights on the board.

    Presentation only: a missed field costs a highlight, never a wrong move, so
    it is safe for this to live at the wire's edge.
    """
    fields: dict[str, Any] = {"macro_index": int(index)}
    if mc.M_WRITE <= index < mc.M_REFUSE:
        slot, delta_slot, x, y = mc.decode_macro_write(index)
        number, effect = state.combination(slot)
        fields.update(
            {
                "kind": "write",
                "stack": slot,
                "box": [x, y],
                "number": number + TEMP_DELTAS[delta_slot],
                "temp_delta": TEMP_DELTAS[delta_slot],
                "effect": effect.name,
            }
        )
        return fields
    if mc.M_REFUSE <= index < mc.M_DIRECT_REFUSE:
        fields.update({"kind": "refuse", "stack": index - mc.M_REFUSE})
        return fields
    if index == mc.M_DIRECT_REFUSE:
        fields["kind"] = "refuse"
        return fields
    if index == mc.M_ROUNDABOUT_OPEN:
        fields["kind"] = "roundabout"
        return fields

    action = mc.to_primitive(index)
    fields["primitive"] = int(action)
    if codec.A_ROUNDABOUT_POS <= action < codec.A_WRITE:
        fields.update(
            {"kind": "roundabout", "box": list(codec.decode_roundabout_pos(action))}
        )
    elif codec.A_WRITE <= action < codec.A_SURVEYOR_FENCE:
        delta_slot, x, y = codec.decode_write(action)
        fields.update(
            {
                "kind": "write",
                "box": [x, y],
                "number": (state.ctx.number or 0) + TEMP_DELTAS[delta_slot],
                "temp_delta": TEMP_DELTAS[delta_slot],
            }
        )
    elif codec.A_SURVEYOR_FENCE <= action < codec.A_ESTATE_ROW:
        fields.update(
            {"kind": "fence", "fence": list(codec.decode_surveyor_fence(action))}
        )
    elif codec.A_BIS <= action < codec.A_CHOOSE_PLAN:
        x, y, side = codec.decode_bis(action)
        fields.update({"kind": "bis", "box": [x, y], "side": side})
    elif codec.A_CHOOSE_PLAN <= action < codec.A_VALIDATE_ESTATE:
        fields.update({"kind": "plan", "slot": codec.decode_plan(action)})
    elif codec.A_VALIDATE_ESTATE <= action < codec.A_PASS_ROUNDABOUT:
        fields.update(
            {"kind": "estate", "box": list(codec.decode_validate_estate(action))}
        )
    else:
        fields["kind"] = "other"
    return fields


# ---------------------------------------------------------------------------
# The resumable search handle
# ---------------------------------------------------------------------------
class _MctsHandle:
    """SearchHandle over one Welcome To tree.

    ``MCTS.search`` already treats ``SearchConfig.simulations`` as a *target
    total* and accepts the root node back, so deepening is a matter of raising
    the target and handing the same node in again -- the tree persists and each
    advance adds work rather than restarting it.  That is exactly the resumable
    contract, so no second search implementation is needed here.

    Cancellation is checked between advances rather than between simulations:
    one simulation is a leaf evaluation and a rollout to the next boundary, so a
    chunk is the natural cancellation grain and the host's ``chunk_sims`` sets
    it.
    """

    def __init__(
        self,
        search: mcts.MCTS,
        base_config: mcts.SearchConfig,
        state: GameState,
        rng: random.Random,
        target: int,
    ) -> None:
        self._search = search
        self._base = base_config
        self._state = state
        self._rng = rng
        self._node: Optional[mcts.Node] = None
        self._done = 0
        self._target = int(target)

    def advance(self, chunk_sims: int, stop_event: threading.Event) -> SearchSnapshot:
        if not stop_event.is_set():
            self._done = min(self._target, self._done + int(chunk_sims))
            self._search.config = replace(self._base, simulations=self._done)
            _, _, self._node = self._search.search(
                self._state, root=0, rng=self._rng, node=self._node
            )
        return self._snapshot(partial=stop_event.is_set())

    def _snapshot(self, *, partial: bool) -> SearchSnapshot:
        node = self._node
        if node is None:
            return SearchSnapshot(
                sims_done=0,
                sims_target=self._target,
                root_value=0.0,
                entries={},
                partial=partial,
            )
        visits = node.visits
        total = node.total
        entries = {
            str(int(action)): ActionStats(
                visits=int(visits[i]),
                # Root is seat 0, which is the viewer, so the tree's own frame is
                # already the actor frame the host renders.
                q_value=float(total[i] / visits[i]) if visits[i] else 0.0,
                prior=float(node.prior[i]),
            )
            for i, action in enumerate(node.actions)
        }
        seen = float(visits.sum())
        return SearchSnapshot(
            sims_done=int(seen),
            sims_target=self._target,
            root_value=float(total.sum() / seen) if seen else 0.0,
            entries=entries,
            partial=partial,
        )

    def close(self) -> None:  # the tree is GC'd with the handle
        self._node = None
        self._search.reset()


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------
class WelcomeToAdvisor:
    """AdvisorAdapter for Welcome To.

    Pass ``net=`` to inject a preloaded network (tests); otherwise checkpoints
    load lazily and are cached per ``(path, device)``.
    """

    game_id = "welcome_to"

    def __init__(
        self,
        *,
        net: Any = None,
        default_checkpoint: Optional[str] = None,
        device: str = "cpu",
        prune_roundabout_pass: bool = True,
    ) -> None:
        self._injected = (
            None
            if net is None
            else LoadedNet(net=net, path="<injected>", legacy_heads=False, source="injected")
        )
        self._default_checkpoint = default_checkpoint
        self._device = device
        # SEARCH_SPEC §5.1a. Training prunes PASS_ROUNDABOUT, so ROUNDABOUT_OPEN
        # means "commit to a roundabout for -3 or -8 points". The served S2
        # checkpoint puts ~0.94 of its prior there and burns both roundabouts by
        # turn 3 of every game.
        #
        # ⚠ **That is a strategy, not the §5.1a artifact**, and the measurement
        # says so: masking the move costs 18.4 points over 25 paired seeds (43.6
        # -> 25.2) and halves completed housing estates (5.96 -> 3.04 a game),
        # because breaking the ascending chain is what closes a street segment
        # into an estate. GreedyBot builds 1.16 a game for the same reason.
        #
        # Parity with training is therefore the default -- the advisor's job is
        # to show what the trained agent would actually do. Turning it off is how
        # you ask "should I pass on THIS roundabout?", which the pruned search
        # cannot answer because the pass is not in its action set.
        self._prune_roundabout_pass = bool(prune_roundabout_pass)
        self._net_cache: dict[tuple[str, str], "LoadedNet"] = {}

    # -- network -----------------------------------------------------------
    def _load(self, checkpoint: Optional[str], device: str) -> "LoadedNet":
        if self._injected is not None:
            return self._injected
        if checkpoint is None:
            raise ValueError("no checkpoint_path supplied and no default set")
        key = (checkpoint, device)
        cached = self._net_cache.get(key)
        if cached is None:
            cached = load_net(checkpoint, device)
            self._net_cache[key] = cached
        return cached

    def _net(self, req) -> "LoadedNet":
        return self._load(
            req.checkpoint_path or self._default_checkpoint, req.device or self._device
        )

    def warnings(self) -> list[str]:
        """Advisor-level warnings the host surfaces with every answer."""
        try:
            model = self._load(self._default_checkpoint, self._device)
        except Exception:
            return []
        if not model.legacy_heads:
            return []
        return [
            "this checkpoint predates the plan-outcome and end-trigger heads; "
            "they are served as a neutral 0.5 and mean nothing"
        ]

    # -- state codec -------------------------------------------------------
    def state_from_wire(self, payload: dict[str, Any]) -> _Position:
        """Three wires, one position.

        ``bga`` is the live browser capture, ``observation`` is its normalized
        form (what the game log stores, so a logged position reloads with no
        second codec), and ``snapshot`` is the engine's own exact serialisation,
        which is what tests and the lab UI use.
        """

        if "bga" in payload:
            from .bga_extract import state_from_bga_payload

            state, obs, warnings = state_from_bga_payload(
                payload, rng=random.Random(int(payload.get("resample_seed", 0)))
            )
            return _Position(
                game=state,
                key=_digest("bga", obs),
                warnings=tuple(warnings),
                observation=obs,
            )

        if "observation" in payload:
            from .bga_extract import state_from_observation

            obs = payload["observation"]
            state, warnings = state_from_observation(
                obs, rng=random.Random(int(payload.get("resample_seed", 0)))
            )
            return _Position(
                game=state,
                key=_digest("obs", obs),
                warnings=tuple(warnings),
                observation=obs,
            )

        if "snapshot" in payload:
            state = from_snapshot(payload["snapshot"])
            return _Position(game=state, key=_digest("snap", payload["snapshot"]))

        raise ValueError(
            "state must carry one of `bga` (browser capture), `observation` "
            "(normalized capture) or `snapshot` (engine serialisation)"
        )

    def state_to_public(self, state: _Position) -> dict[str, Any]:
        game = state.game
        terminal = game.is_terminal
        # A BGA capture knows who the other players are; a snapshot does not.
        # Names are for the panel only -- everything the search reads is keyed by
        # seat -- so falling back to the seat number costs nothing.
        obs_seats = (state.observation or {}).get("seats") or []
        return {
            "game": self.game_id,
            "turn": int(game.turn),
            "phase": game.phase.name,
            "actor": int(game.actor),
            "terminal": terminal,
            "players": int(game.config.players),
            "advanced": bool(game.config.advanced),
            "deck_remaining": int(game.deck_remaining),
            "discard": len(game.discard),
            "plans": [
                {
                    "slot": slot,
                    "plan_id": int(plan_id),
                    "name": _plan_name(game, slot),
                    "scores": list(PLANS[plan_id].scores),
                    "completed_by": {
                        str(seat): int(turn)
                        for seat, turn in game.plan_turns[slot].items()
                    },
                }
                for slot, plan_id in enumerate(game.plan_ids)
            ],
            "stacks": [
                {
                    "slot": slot,
                    "number": faces[0],
                    "effect": None if faces[1] is None else Effect(faces[1]).name,
                    "next_effect": (
                        None if nxt is None else Effect(nxt).name
                    ),
                    "playable": slot in game.playable_slots(),
                }
                for slot, (faces, nxt) in enumerate(
                    zip(game.visible_cards(0), game.next_effects(0))
                )
            ],
            "seats": [
                {
                    "seat": seat,
                    "you": seat == 0,
                    "name": (
                        obs_seats[seat]["name"]
                        if seat < len(obs_seats)
                        else "seat %d" % (seat + 1,)
                    ),
                    "score": int(game.scores(viewer=0)[seat]),
                    "houses": sum(
                        1
                        for row in game.sheet_for(0, seat).numbers
                        for value in row
                        if value is not None
                    ),
                    "temps": game.sheet_for(0, seat).temps,
                    "permits": game.sheet_for(0, seat).permits,
                }
                for seat in range(game.config.players)
            ],
            "legal_actions": [
                {"action_id": v.action_id, "label": v.label, "kind": v.kind}
                for v in self.action_views(state)
            ],
            "forecast": self._forecast(state),
            "warnings": list(state.warnings),
        }

    def state_key(self, state: _Position) -> str:
        return state.key

    def action_views(self, state: _Position) -> list[ActionView]:
        game = state.game
        if game.is_terminal:
            return []
        views: list[ActionView] = []
        # The FULL legal set, not the search's pruned one. Views only supply
        # labels -- the host ranks the search's own entries -- so a superset is
        # free, while a subset would print a raw macro index at the exact moment
        # the pruning config and this list disagreed.
        for index in mc.legal_macros(game):
            fields = _macro_fields(game, index)
            views.append(
                ActionView(
                    action_id=str(index),
                    label=_macro_label(game, index),
                    kind=str(fields.get("kind", "")),
                    fields=fields,
                )
            )
        return views

    # -- search ------------------------------------------------------------
    def open_search(self, state: _Position, req):
        engine = "nn" if req.engine in ("auto", "nn") else req.engine
        if engine != "nn":
            raise ValueError("unknown engine %r" % (req.engine,))

        config = mcts.SearchConfig(
            simulations=int(req.max_sims),
            c_puct=float(req.options.get("c_puct", 1.5)),
            alpha=float(req.options.get("alpha", 0.5)),
            chance_widening=req.options.get("chance_widening", 1.0),
            max_particles=int(req.options.get("max_particles", 4)),
            prune_roundabout_pass=bool(
                req.options.get("prune_roundabout_pass", self._prune_roundabout_pass)
            ),
            # Dirichlet noise exists to diversify SELF-PLAY data. On an advisor
            # it would perturb the very ranking the human is reading, so it stays
            # off no matter what the training config used.
            dirichlet_alpha=None,
            dirichlet_concentration=None,
        )
        import torch

        model = self._net(req)
        evaluator = mcts.NetEvaluator(
            model.net, torch.device(req.device or self._device), config
        )
        search = mcts.MCTS(evaluator, config)
        return _MctsHandle(
            search,
            config,
            state.game,
            random.Random(int(req.seed)),
            int(req.max_sims),
        )

    # -- the net's own read of the position --------------------------------
    #: Per-seat forecasts worth surfacing, and how to turn a head into points.
    _SCORE_HEADS: tuple[str, ...] = (
        "score",
        "score_plans",
        "score_parks",
        "score_pools",
        "score_estates",
        "score_temp",
        "score_bis",
        "score_permits",
        "score_roundabouts",
    )

    def _forecast(self, state: _Position) -> Optional[dict[str, Any]]:
        """One raw root evaluation, rendered as what the net expects to happen.

        No search: these are properties of the position, so they are reported
        here rather than through an annotator (annotators run when a search
        settles, and a streaming advisor search never does).

        For Welcome To this is the more diagnostic half of the panel. A single
        win probability cannot say whether the net is mis-ranking moves or
        mis-reading the board; a predicted final score of 32 against an
        opponent's 51, or a plan it thinks it will never finish, says exactly
        where to look.

        ``None`` when no checkpoint is configured or the forward fails; the
        caller renders what it gets.
        """

        game = state.game
        if game.is_terminal:
            return None
        try:
            model = self._load(self._default_checkpoint, self._device)
        except Exception:
            return None
        net = model.net

        try:
            import torch

            device = next(net.parameters()).device
            columns = enc.encode_state(game, 0)
            tensors = [
                torch.as_tensor(np.asarray(column)[None, ...]).float().to(device)
                for column in columns
            ]
            net.eval()
            with torch.no_grad():
                out = net(*tensors)
                seats = min(game.config.players, enc.MAX_SEATS)
                mask = torch.zeros((1, training.MAX_RANKS), device=device)
                mask[0, :seats] = 1.0
                rank_probs = (
                    nw.rank_probabilities(out["rank_logits"], mask)[0].cpu().numpy()
                )
                heads = {
                    name: out[name][0].cpu().numpy()
                    for name in nw.PER_SEAT_HEAD_TARGETS
                }
                turns_left = float(out["turns_left"][0].cpu().numpy())
        except Exception:
            return None

        scale = training.SCORE_SCALE

        def _seat(seat: int) -> dict[str, Any]:
            row: dict[str, Any] = {
                "seat": seat,
                "you": seat == 0,
                "final_score": float(heads["score"][seat]) * scale,
                "components": {
                    name[len("score_") :]: float(heads[name][seat]) * scale
                    for name in self._SCORE_HEADS[1:]
                },
                "plans_completed": float(heads["plans_completed"][seat])
                * training.PLAN_SCALE,
                "will_complete_plan": [
                    _sigmoid(float(heads["will_complete_plan_%d" % slot][seat]))
                    for slot in range(3)
                ],
                "turns_to_plan": [
                    float(heads["turns_to_plan_%d" % slot][seat]) * training.TURN_SCALE
                    for slot in range(3)
                ],
            }
            return row

        return {
            "rank_probs": [float(p) for p in rank_probs[:seats]],
            "turns_left": turns_left * training.TURN_SCALE,
            "end_trigger": {
                name[len("end_trigger_") :]: _sigmoid(float(heads[name][0]))
                for name in (
                    "end_trigger_full_sheet",
                    "end_trigger_all_plans",
                    "end_trigger_max_permit",
                )
            },
            "seats": [_seat(seat) for seat in range(seats)],
            # Heads this checkpoint never trained. They come back as an exact
            # 0.5, which is indistinguishable from a real coin flip unless it is
            # said out loud, so the panel greys them instead of reading them.
            "untrained_heads": list(UNTRAINED_IN_LEGACY) if model.legacy_heads else [],
            "checkpoint": {"path": model.path, "format": model.source},
        }

    # -- discovery ---------------------------------------------------------
    def engines(self) -> Mapping[str, EngineSpec]:
        return {
            "nn": EngineSpec(
                key="nn",
                label="Neural MCTS (macro tree)",
                description=(
                    "PUCT over the 684-slot macro vocabulary, with the chance "
                    "edge progressively widened; visits rank moves."
                ),
                needs_checkpoint=True,
                default_sims=256,
                streaming=True,
            )
        }

    def annotators(self) -> Sequence[Any]:
        return ()

    def contract(self) -> dict[str, Any]:
        try:
            model = self._load(self._default_checkpoint, self._device)
            checkpoint = {
                "path": model.path,
                "format": model.source,
                "legacy_heads": model.legacy_heads,
            }
        except Exception as exc:  # /health must answer even with no checkpoint
            checkpoint = {"error": str(exc)}
        return {
            "game_id": self.game_id,
            "engines": list(self.engines()),
            "default_checkpoint": self._default_checkpoint,
            "checkpoint": checkpoint,
            "device": self._device,
            "prune_roundabout_pass": self._prune_roundabout_pass,
            "wire": "bga | observation | snapshot",
            "action_space": mc.NUM_MACRO_ACTIONS,
            "encoder_abi": enc.ENCODER_ABI_VERSION,
            "supported": "standard rules, 2-4 seats, base board (advanced ok)",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


def _digest(prefix: str, payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return "%s:%s" % (prefix, hashlib.sha256(blob).hexdigest()[:16])


#: Per-seat heads that exist in the current network but not in checkpoints
#: written before the dense plan-outcome heads landed.  ``load_state_dict_compatible``
#: fills them with a **neutral zero logit**, which reads as a confident-looking
#: 0.5 -- so they have to be labelled rather than shown.
UNTRAINED_IN_LEGACY: tuple[str, ...] = tuple(
    name
    for name in nw.PER_SEAT_HEAD_TARGETS
    if name not in nw.LEGACY_PER_SEAT_HEAD_TARGETS
)


@dataclass(frozen=True, slots=True)
class LoadedNet:
    """A served checkpoint and what is known about it.

    ``legacy_heads`` is not a detail: the newer binary heads come back as an
    exact 0.5 on such a checkpoint, and an untagged 0.5 is indistinguishable
    from a genuine coin flip.
    """

    net: Any
    path: str
    legacy_heads: bool
    source: str


def load_net(path: str, device: str = "cpu") -> LoadedNet:
    """Load either checkpoint format the project writes.

    S2 checkpoints carry a ``format``/``version`` header; the S0 bootstrap
    writes a bare ``{net_config, state_dict}`` blob.  Trying the strict reader
    first means an S2 file with a *wrong* version is reported as such rather
    than silently falling through to the loose one.
    """

    import torch

    from games.welcome_to import s2_train

    blob = torch.load(path, map_location=device, weights_only=False)
    source = "s0"
    if blob.get("format") == s2_train.CHECKPOINT_FORMAT:
        version = int(blob.get("version", -1))
        if version not in (
            s2_train.LEGACY_CHECKPOINT_VERSION,
            s2_train.CHECKPOINT_VERSION,
        ):
            raise ValueError("unsupported S2 checkpoint version %r" % (version,))
        source = "s2 v%d" % (version,)
    net = nw.WelcomeToNet(nw.NetConfig(**blob["net_config"]))
    legacy = nw.load_state_dict_compatible(net, blob["state_dict"])
    return LoadedNet(
        net=net.to(device), path=str(path), legacy_heads=bool(legacy), source=source
    )
