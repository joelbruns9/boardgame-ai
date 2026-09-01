"""Exact endgame annotator for the 7WD advisor.

Fills the shared ``Annotator`` slot (games.advisor.contract).  When a position
is close enough to the end to solve to terminal within a budget, it replaces
the net's fuzzy Q with the game-theoretic truth and attaches it to each
recommendation.

Two honest regimes (the removed-3-cards subtlety):

* ``exact`` -- the solve encountered **no chance** (every remaining tableau
  card already face-up, no Great Library draw): perfect information, so the
  value is a deterministic win / loss / draw.
* ``exact_expectimax`` -- chance remained (face-down cards, whose identity is
  drawn from a pool that *includes* the 3 unused removed cards, or a Great
  Library draw).  The value is then the exact *expectation* -- true win
  probability, not a deterministic outcome.  The removed cards are handled
  correctly because ``enumerate_chains`` draws from the full unseen pool.

Regime is decided by whether the solve *actually hit a chance edge*, not by a
static guess, so a late Great Library draw is classified honestly.

The production annotator uses the Rust solver.  The Python implementation in
this module remains the deliberately simple reference used by the equivalence
tests.

Safety: the model gates attempts by predicted node count, but an admitted Rust
solve is bounded only by wall time.  Its call cannot be interrupted in flight,
so cancellation is checked before entry and the remaining host deadline is
converted to ``max_secs``.  A refusal or timeout returns ``None`` and the net
estimate stands.  A solve runs to terminal, so no evaluator/checkpoint is
needed.
"""

from __future__ import annotations

import math
import time as _time

from games.advisor import AnnotationResult

from .codec import decode_action, legal_action_indices
from .engine import apply_action
from .game import ChanceKind, Phase
from .search import chance_signature, enumerate_chains, state_actor

_EPS = 1e-9

# Cost prediction replaces the old card-count cap: it can accept a cheap
# 11-card board and reject a chance-heavy 8-card one.  This is an attempt
# threshold, not a runtime node budget: once admitted, wall time is the only
# solve limit.
_DEFAULT_MAX_PREDICTED_NODES = 10_000_000
_NO_NODE_LIMIT = (1 << 64) - 1
# Rust cannot observe stop_event while inside solve_endgame, so its own clock is
# the hard bound on stale work after the board changes.
_DEFAULT_MAX_SECS = 30.0


class _Unsolvable(Exception):
    """A sample-only chance event (AGE_DEAL) was reached: not enumerable."""


class _BudgetExceeded(Exception):
    """Node budget, deadline, or cancellation reached before completion."""


class _Ctx:
    __slots__ = ("max_nodes", "deadline", "stop", "nodes", "saw_chance")

    def __init__(self, max_nodes: int, deadline: float, stop):
        self.max_nodes = max_nodes
        self.deadline = deadline
        self.stop = stop
        self.nodes = 0
        self.saw_chance = False

    def tick(self) -> None:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise _BudgetExceeded("node budget")
        if self.stop is not None and self.stop.is_set():
            raise _BudgetExceeded("cancelled")
        if _time.perf_counter() > self.deadline:
            raise _BudgetExceeded("deadline")


def _terminal_p0(state) -> float:
    if state.winner is None:
        return 0.0
    return 1.0 if state.winner == 0 else -1.0


def _children(state, action):
    """(child, probability) list for one action, integrating enumerable chance.

    Barred clones + explicit outcomes: hidden identities are never read.  Raises
    :class:`_Unsolvable` on a sample-only AGE_DEAL edge.
    """

    specs = chance_signature(state, action)
    if any(spec.kind is ChanceKind.AGE_DEAL for spec in specs):
        raise _Unsolvable("AGE_DEAL")
    if specs:
        out = []
        mass = 0.0
        for outcomes, probability, _key in enumerate_chains(state, specs):
            child = state.clone()
            child.search_barrier = True
            apply_action(child, action, chance_outcomes=outcomes or None)
            out.append((child, probability))
            mass += probability
        if abs(mass - 1.0) > 1e-6:
            raise _Unsolvable(f"chance mass {mass:.6f} != 1")
        return out, True
    child = state.clone()
    child.search_barrier = True
    apply_action(child, action)
    return [(child, 1.0)], False


def _solve_p0(state, ctx: _Ctx) -> float:
    """Exact minimax / expectimax value in player-0 terms."""

    ctx.tick()
    if state.phase is Phase.COMPLETE:
        return _terminal_p0(state)
    actor = state_actor(state)
    sign = 1.0 if actor == 0 else -1.0
    best = None
    for index in legal_action_indices(state):
        action = decode_action(state, index)
        children, chanced = _children(state, action)
        if chanced:
            ctx.saw_chance = True
        value = 0.0
        for child, probability in children:
            value += probability * _solve_p0(child, ctx)
        actor_value = sign * value
        if best is None or actor_value > best:
            best = actor_value
    if best is None:  # no legal actions but not COMPLETE -- treat as terminal
        return _terminal_p0(state)
    return sign * best


def _outcome(actor_value: float) -> str:
    if actor_value > _EPS:
        return "win"
    if actor_value < -_EPS:
        return "loss"
    return "draw"


def solve_position(game, *, deadline: float, max_nodes: int, stop=None) -> dict | None:
    """Exact per-action solve of ``game`` (root-actor frame).

    Returns ``{regime, per_action_value, root_value, best_index}`` or ``None``
    when the position is not enumerable or exceeds the budget/deadline.
    ``per_action_value`` maps ``action_index -> actor_value`` in [-1, 1].
    """

    ctx = _Ctx(max_nodes, deadline, stop)
    actor = state_actor(game)
    sign = 1.0 if actor == 0 else -1.0
    per_action: dict[int, float] = {}
    try:
        for index in legal_action_indices(game):
            action = decode_action(game, index)
            children, chanced = _children(game, action)
            if chanced:
                ctx.saw_chance = True
            value = 0.0
            for child, probability in children:
                value += probability * _solve_p0(child, ctx)
            # True game value is in [-1, 1]; clamp probability-sum fp noise.
            per_action[index] = max(-1.0, min(1.0, sign * value))
    except (_Unsolvable, _BudgetExceeded):
        return None
    if not per_action:
        return None
    best_index = max(per_action, key=per_action.__getitem__)
    return {
        "regime": "exact_expectimax" if ctx.saw_chance else "exact",
        "per_action_value": per_action,
        "root_value": per_action[best_index],
        "best_index": best_index,
        "nodes": ctx.nodes,
    }


class ExactEndgameAnnotator:
    """Exact solve of near-terminal 7WD positions, attached per recommendation."""

    name = "exact_endgame"
    concurrent = True
    default_budget_secs = _DEFAULT_MAX_SECS

    def __init__(
        self,
        *,
        max_predicted_nodes: int = _DEFAULT_MAX_PREDICTED_NODES,
        max_secs: float = _DEFAULT_MAX_SECS,
    ):
        self._max_predicted_nodes = max_predicted_nodes
        self._max_secs = max_secs

    @staticmethod
    def _cost_prediction(rust_game) -> float:
        """Return predicted ``log10(nodes)`` for this exact solve.

        Features are computed in Rust at the exact state boundary the solver
        consumes.  The parity suite checks them feature-by-feature against the
        Python definitions used to fit the model.
        """

        import seven_wonders_rust as swr

        from .validate_cost_trigger import load_cost_model

        coefficients, features, _margin = load_cost_model()
        rust_names = tuple(swr.endgame_cost_model_features())
        if rust_names != features:
            raise RuntimeError(
                "endgame cost-model feature order differs from the Rust solver"
            )
        values = swr.endgame_cost_features(rust_game)
        predicted = coefficients[0] + sum(
            weight * value for weight, value in zip(coefficients[1:], values)
        )
        return predicted

    @staticmethod
    def _guaranteed(value: float, regime: str) -> bool:
        """Whether ``value`` supports a categorical W/D/L statement.

        A boundary expectation (+1 or -1) is necessarily the same terminal
        outcome under every positive-probability chance result, even if a
        conservative traversal happened to report ``exact_expectimax``.
        Zero is different: it can be a forced draw or cancelling win/loss
        probabilities, so only a chance-free proof may call it a draw.
        """

        return regime == "exact" or abs(abs(value) - 1.0) <= _EPS

    @staticmethod
    def _status(status: str, reason: str, **details) -> AnnotationResult:
        """A visible non-proof result, so the panel never has to infer silence."""

        return AnnotationResult(
            name="exact_endgame",
            summary={"status": status, "reason": reason, **details},
            partial=status in ("running", "timed_out"),
        )

    def annotate(self, state, recommendations, req, *, deadline, stop_event):
        game = getattr(state, "game", state)
        if game.phase is not Phase.PLAY_AGE or game.age != 3:
            return self._status(
                "ready",
                "age_3_only",
                age=int(game.age),
            )
        if stop_event is not None and stop_event.is_set():
            return None

        max_predicted_nodes = int(
            req.options.get(
                "endgame_max_predicted_nodes", self._max_predicted_nodes
            )
        )
        if max_predicted_nodes <= 0:
            return self._status("disabled", "invalid_attempt_threshold")

        try:
            from .rust_bridge import rust_game_from_state

            rust_game = rust_game_from_state(game)
            predicted_log_nodes = self._cost_prediction(rust_game)
        except ImportError:
            return self._status("unavailable", "rust_extension_missing")

        if not math.isfinite(predicted_log_nodes):
            return self._status("error", "invalid_cost_prediction")
        predicted_nodes = int(round(10.0**predicted_log_nodes))
        if predicted_log_nodes > math.log10(max_predicted_nodes):
            return self._status(
                "skipped",
                "predicted_too_large",
                predicted_nodes=predicted_nodes,
                max_predicted_nodes=max_predicted_nodes,
            )

        remaining = deadline - _time.perf_counter()
        configured_secs = float(
            req.options.get("endgame_max_secs", self._max_secs)
        )
        if not math.isfinite(configured_secs) or configured_secs <= 0.0:
            return self._status("disabled", "invalid_time_limit")
        max_secs = min(configured_secs, remaining)
        if max_secs <= 0.0:
            return self._status("timed_out", "no_time_remaining", nodes=0)

        solve_started = _time.perf_counter()
        answer = rust_game.solve_endgame(
            _NO_NODE_LIMIT, max_secs, "exact", "star1"
        )
        solve_ms = int(round((_time.perf_counter() - solve_started) * 1000.0))
        if answer["regime"] is None:
            stop = str(answer.get("stop") or "declined")
            return self._status(
                "timed_out" if stop == "deadline" else "declined",
                stop,
                nodes=int(answer.get("nodes", 0)),
                predicted_nodes=predicted_nodes,
                max_predicted_nodes=max_predicted_nodes,
                solve_ms=solve_ms,
            )

        solved = {
            "regime": str(answer["regime"]),
            "root_value": float(answer["root_value"]),
            "best_index": int(answer["best_index"]),
            "per_action_value": {
                int(index): float(value)
                for index, value in answer["per_action_value"].items()
            },
            "nodes": int(answer["nodes"]),
        }

        regime = solved["regime"]
        per_action_value = {
            str(index): value for index, value in solved["per_action_value"].items()
        }
        root_value = solved["root_value"]
        best_ids = {
            action_id
            for action_id, value in per_action_value.items()
            if value >= root_value - _EPS
        }
        # Rust deliberately names a useful representative among ties. Preserve
        # it for compatibility, while exposing the full proven-optimal set.
        best_id = str(solved["best_index"])

        per_action = {
            action_id: {
                "regime": regime,
                "exact_value": value,
                "win_pct": (value + 1.0) / 2.0 * 100.0,
                "outcome": (
                    _outcome(value) if self._guaranteed(value, regime) else None
                ),
                "is_best": action_id in best_ids,
            }
            for action_id, value in per_action_value.items()
        }
        summary = {
            "status": "solved",
            "regime": regime,
            "root_exact_value": root_value,
            "root_win_pct": (root_value + 1.0) / 2.0 * 100.0,
            "outcome": (
                _outcome(root_value)
                if self._guaranteed(root_value, regime)
                else None
            ),
            "best_action_id": best_id,
            "best_action_ids": sorted(best_ids, key=int),
            "nodes": solved["nodes"],
            "predicted_nodes": predicted_nodes,
            "max_predicted_nodes": max_predicted_nodes,
            "solve_ms": solve_ms,
        }
        return AnnotationResult(
            name=self.name, per_action=per_action, summary=summary, partial=False
        )
