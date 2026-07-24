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

Safety: the solver is hard-bounded by a node budget, the annotator deadline,
and ``stop_event`` -- it can never hang the host.  Too-large or non-enumerable
(``AGE_DEAL``) positions return ``None`` and the net estimate stands.  A solve
runs to terminal, so no evaluator/checkpoint is needed.
"""

from __future__ import annotations

import time as _time

from games.advisor import AnnotationResult

from .codec import decode_action, legal_action_indices
from .engine import apply_action
from .game import ChanceKind, Phase
from .search import chance_signature, enumerate_chains, state_actor

_EPS = 1e-9


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

    def __init__(self, *, max_present: int = 9, max_nodes: int = 300_000):
        self._max_present = max_present
        self._max_nodes = max_nodes

    def annotate(self, state, recommendations, req, *, deadline, stop_event):
        game = getattr(state, "game", state)
        if game.phase is not Phase.PLAY_AGE:
            return None  # COMPLETE has no recs; other phases aren't endgames
        present = sum(1 for card in game.tableau.cards.values() if card.present)
        max_present = int(req.options.get("endgame_max_present", self._max_present))
        if present > max_present:
            return None
        max_nodes = int(req.options.get("endgame_max_nodes", self._max_nodes))

        solved = solve_position(
            game, deadline=deadline, max_nodes=max_nodes, stop=stop_event
        )
        if solved is None:
            return None  # net estimate stands

        regime = solved["regime"]
        deterministic = regime == "exact"
        per_action_value = {
            str(index): value for index, value in solved["per_action_value"].items()
        }
        best_id = str(solved["best_index"])
        root_value = solved["root_value"]

        per_action = {
            action_id: {
                "regime": regime,
                "exact_value": value,
                "win_pct": (value + 1.0) / 2.0 * 100.0,
                "outcome": _outcome(value) if deterministic else None,
                "is_best": action_id == best_id,
            }
            for action_id, value in per_action_value.items()
        }
        summary = {
            "regime": regime,
            "root_exact_value": root_value,
            "root_win_pct": (root_value + 1.0) / 2.0 * 100.0,
            "outcome": _outcome(root_value) if deterministic else None,
            "best_action_id": best_id,
            "nodes": solved["nodes"],
        }
        return AnnotationResult(
            name=self.name, per_action=per_action, summary=summary, partial=False
        )
