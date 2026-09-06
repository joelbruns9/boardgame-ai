"""Certify a forced sudden-death defeat, against the real rules engine.

Workstream 3, step 2 of the plan in ``BOARD_CONTROL_RESEARCH_REQUEST.md`` §9.
This is the "verified tactical consequence" half of W3: ``tableau_control``
answers *who reaches a slot first* under a cost-free abstraction, and this module
answers the question that abstraction is not allowed to answer -- **can the
opponent actually force a science or military victory from here, paying real
costs, against every legal defence and every enumerable reveal?**

Why this is cheaper than ``advisor_endgame``
--------------------------------------------
The endgame solver prices every action to a terminal score, which is far more
than a yes/no needs. This one proves a *proposition*, so:

* it stops the instant a sudden-death victory fires -- it never plays out the
  rest of the Age, never scores civilian points, never resolves the final
  tableau;
* an OR node ends at the first winning move found; an AND node ends at the first
  escape found.

Cost therefore scales with the size of the forcing sequence, not with the number
of cards left. That is the whole reason it can reach positions the 8-11 card
endgame solver cannot. It is *not* a general solver and must never be described
as one.

The three-valued result, and what each value does NOT mean
----------------------------------------------------------
``PROVEN``    the loser cannot avoid a sudden-death defeat within the horizon,
              against every legal defence and every enumerable chance outcome.
``REFUTED``   a defence exists that avoids sudden death within the horizon.
              **This does not mean the loser wins, or is even better off.** The
              game may still be lost on civilian points, or lost to a forced
              sequence one ply beyond the horizon.
``UNKNOWN``   the node budget, the horizon, or a non-enumerable deal stopped the
              proof. Callers must leave the network's estimate alone.

Keeping those apart is the standing requirement from the review: "no forced win
within this horizon" must never collapse into "no forced win", and neither may
collapse into "losing".

Soundness notes
---------------
* Every continuation is executed by ``engine.apply_action`` -- the real rules,
  including costs, chains, discounts, Wonder retirement and the seventh-Wonder
  rule. Nothing here re-implements the economy, which is exactly the failure
  mode ``tableau_control`` is constrained to avoid.
* Chance is enumerated, never sampled. A ``PROVEN`` verdict requires *every*
  outcome to be proven; one escaping reveal refutes it. An ``AGE_DEAL`` edge is
  not enumerable and yields ``UNKNOWN``.
* Clones carry ``search_barrier``, so a hidden identity read would raise rather
  than silently leak the determinization.
* ``UNKNOWN`` is absorbing in the direction that matters: it can never be
  upgraded to ``PROVEN`` by an unexamined branch.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from enum import Enum

from .codec import decode_action, legal_action_indices
from .engine import apply_action
from .fast_clone import fast_clone
from .game import ChanceKind, Phase, VictoryType
from .search import chance_signature, enumerate_chains, state_actor

#: Victory types that end the game immediately, mid-Age. Civilian and shared
#: civilian are scored at the end and are NOT sudden death, so a certificate
#: never claims them.
SUDDEN_DEATH = (VictoryType.SCIENTIFIC, VictoryType.MILITARY)

DEFAULT_MAX_NODES = 400_000
DEFAULT_MAX_PLIES = 12
DEFAULT_MAX_SECS = 5.0


class Verdict(str, Enum):
    PROVEN = "proven"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


@dataclass
class Certificate:
    """The result of one certification attempt."""

    verdict: Verdict
    loser: int
    nodes: int
    seconds: float
    stopped_by: str | None = None
    """The limit that ENDED the search: 'nodes', 'deadline', 'cancelled', or
    None when the horizon alone truncated it."""
    limits_hit: tuple[str, ...] = ()
    """Every limit that fired anywhere in the tree -- 'plies', 'age_deal',
    'nodes', 'deadline'. A run can hit the horizon in thousands of branches and
    still die on the clock; reporting only one of those conflates two different
    fixes."""
    victory_type: VictoryType | None = None
    """Which sudden death was proven. Only set when verdict is PROVEN."""
    principal_line: list[str] = field(default_factory=list)
    """Labels of the winner's forcing moves, root first. Illustrative only --
    a certificate is the whole tree, never this one line."""

    @property
    def proven(self) -> bool:
        return self.verdict is Verdict.PROVEN


class _Budget:
    """Node/time budget, recording EVERY limit that fired.

    Recording only the first is a reporting trap: the ply horizon fires early
    and often on any deep search, so a first-reason field reads "plies" even
    when the run actually died on the clock -- and "the horizon was too short"
    and "the machine was too slow" call for opposite fixes. Both are kept, and
    `binding` names the one that ended the search.
    """

    __slots__ = ("max_nodes", "deadline", "stop", "nodes", "reasons", "binding")

    def __init__(self, max_nodes: int, deadline: float, stop):
        self.max_nodes = max_nodes
        self.deadline = deadline
        self.stop = stop
        self.nodes = 0
        self.reasons: set[str] = set()
        self.binding: str | None = None

    def note(self, reason: str) -> None:
        self.reasons.add(reason)

    def tick(self) -> bool:
        """False once the budget is spent."""

        self.nodes += 1
        if self.nodes > self.max_nodes:
            self.reasons.add("nodes")
            self.binding = self.binding or "nodes"
            return False
        if self.stop is not None and self.stop.is_set():
            self.reasons.add("cancelled")
            self.binding = self.binding or "cancelled"
            return False
        if _time.perf_counter() > self.deadline:
            self.reasons.add("deadline")
            self.binding = self.binding or "deadline"
            return False
        return True


# ---------------------------------------------------------------------------
# Threat detection -- the cheap gate that decides whether to attempt at all
# ---------------------------------------------------------------------------


def science_symbols(game, player: int) -> set:
    """Distinct science symbols ``player`` holds, buildings and tokens."""

    from .engine import _science_symbols

    return _science_symbols(game, player)


def sudden_death_threat(game, loser: int, *, science_gap: int = 1,
                        military_gap: int = 2) -> dict | None:
    """Is ``loser``'s opponent close enough to sudden death to be worth a proof?

    Deliberately loose: a cheap over-trigger costs one budgeted attempt, while a
    missed trigger costs the whole point of the module. Returns a dict
    describing the threat, or None.

    ``military_gap`` is in pawn steps toward ``loser``'s capital, so it counts
    shields the opponent still needs, not their total.
    """

    if game.phase is Phase.COMPLETE:
        return None
    winner = 1 - loser
    threats = {}

    missing = 6 - len(science_symbols(game, winner))
    if 0 < missing <= science_gap:
        threats["science_missing"] = missing

    # conflict_position is signed toward player 1's capital for player 0.
    toward = game.conflict_position if winner == 0 else -game.conflict_position
    steps_left = 9 - toward
    if 0 < steps_left <= military_gap:
        threats["military_steps"] = steps_left

    return threats or None


# ---------------------------------------------------------------------------
# The proof
# ---------------------------------------------------------------------------


def _terminal_verdict(state, loser: int) -> Verdict:
    """A finished game: did ``loser`` lose to sudden death?"""

    if (
        state.winner is not None
        and state.winner == 1 - loser
        and state.victory_type in SUDDEN_DEATH
    ):
        return Verdict.PROVEN
    # Any other ending -- civilian, shared civilian, or the loser winning -- is
    # not the proposition. REFUTED here means "no sudden death", nothing more.
    return Verdict.REFUTED


def _children(state, action):
    """``([(child, probability), ...], saw_chance)``, chance enumerated.

    Mirrors ``advisor_endgame._children``: barred clones and explicit outcomes,
    so no hidden identity is ever read. Returns ``None`` on a non-enumerable
    ``AGE_DEAL`` edge, which the caller turns into UNKNOWN.
    """

    specs = chance_signature(state, action)
    if any(spec.kind is ChanceKind.AGE_DEAL for spec in specs):
        return None
    if specs:
        out = []
        mass = 0.0
        for outcomes, probability, _key in enumerate_chains(state, specs):
            child = fast_clone(state)
            child.search_barrier = True
            apply_action(child, action, chance_outcomes=outcomes or None)
            out.append((child, probability))
            mass += probability
        if abs(mass - 1.0) > 1e-6:
            return None
        return out, True
    child = fast_clone(state)
    child.search_barrier = True
    apply_action(child, action)
    return [(child, 1.0)], False


def _order(state, indices, loser: int, winner_to_move: bool):
    """Move ordering. Correctness never depends on this; cost does.

    Both node types want the *decisive* move first: the winner wants the move
    that completes the victory, the loser wants the move that escapes it. Those
    are largely the same cards, so one ordering serves both -- science and red
    cards first, then Wonders (an extra turn is the tempo that decides these
    races), then everything else.

    This is where ``tableau_control`` belongs as a refinement: it can name the
    slot the race is about, so the moves touching it sort first. Left as a hook
    rather than guessed at, because a wrong ordering is silently expensive and a
    missing one is merely slow.
    """

    from .advisor_adapter import _card_name_at
    from .data import CARDS_BY_NAME
    from .data import CardColor

    def key(index: int) -> tuple:
        try:
            action = decode_action(state, index)
            name = _card_name_at(state, action.slot_id)
        except Exception:  # pragma: no cover - ordering must never raise
            return (3, index)
        card = CARDS_BY_NAME.get(name) if name else None
        if card is not None and (card.science is not None
                                 or card.color is CardColor.RED):
            return (0, index)
        if action.wonder_name:
            return (1, index)
        return (2, index)

    return sorted(indices, key=key)


def _prove(state, loser: int, plies: int, budget: _Budget, line: list) -> Verdict:
    """Three-valued proof that ``loser`` cannot escape sudden death.

    AND node (loser to move): every defence must be PROVEN.
    OR node (winner to move): one move that is PROVEN suffices.
    Chance: every enumerable outcome must be PROVEN.

    UNKNOWN never becomes PROVEN, so a truncated branch can only ever weaken the
    claim -- which is what makes an incomplete search safe to act on.
    """

    if state.phase is Phase.COMPLETE:
        return _terminal_verdict(state, loser)
    if plies <= 0:
        budget.note("plies")
        return Verdict.UNKNOWN
    if not budget.tick():
        return Verdict.UNKNOWN

    actor = state_actor(state)
    winner_to_move = actor != loser
    indices = list(legal_action_indices(state))
    if not indices:
        return _terminal_verdict(state, loser)

    saw_unknown = False
    for index in _order(state, indices, loser, winner_to_move):
        action = decode_action(state, index)
        got = _children(state, action)
        if got is None:
            budget.note("age_deal")
            saw_unknown = True
            continue
        children, _chanced = got

        # Every chance outcome must hold for the branch to count as proven.
        branch = Verdict.PROVEN
        sub_line: list = []
        for child, _probability in children:
            got_child = _prove(child, loser, plies - 1, budget, sub_line)
            if got_child is Verdict.REFUTED:
                branch = Verdict.REFUTED
                break
            if got_child is Verdict.UNKNOWN:
                branch = Verdict.UNKNOWN

        if winner_to_move:
            if branch is Verdict.PROVEN:
                line[:] = [_label(state, index)] + sub_line
                return Verdict.PROVEN
            if branch is Verdict.UNKNOWN:
                saw_unknown = True
        else:
            if branch is Verdict.REFUTED:
                return Verdict.REFUTED
            if branch is Verdict.UNKNOWN:
                saw_unknown = True

    if saw_unknown:
        return Verdict.UNKNOWN
    # Winner tried everything and none forced it; loser tried everything and all
    # lost.
    return Verdict.REFUTED if winner_to_move else Verdict.PROVEN


def _label(state, index: int) -> str:
    try:
        from .advisor_adapter import _label as label_of

        return label_of(decode_action(state, index), state)
    except Exception:  # pragma: no cover - labels are cosmetic
        return f"action:{index}"


def certify(
    game,
    *,
    loser: int | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_plies: int = DEFAULT_MAX_PLIES,
    max_secs: float = DEFAULT_MAX_SECS,
    stop=None,
    require_threat: bool = True,
) -> Certificate:
    """Attempt to prove that ``loser`` cannot escape a sudden-death defeat.

    ``loser`` defaults to the side to move. ``require_threat=False`` forces an
    attempt even when the cheap gate sees nothing, which is what the tests and
    the coverage study need.

    The returned certificate is only actionable when ``verdict is PROVEN``.
    Anything else means *leave the network's estimate alone*.
    """

    if loser is None:
        loser = state_actor(game)
    started = _time.perf_counter()

    if require_threat and sudden_death_threat(game, loser) is None:
        return Certificate(
            verdict=Verdict.UNKNOWN,
            loser=loser,
            nodes=0,
            seconds=0.0,
            stopped_by="no_threat",
            limits_hit=("no_threat",),
        )

    budget = _Budget(max_nodes, started + max_secs, stop)
    line: list = []
    root = fast_clone(game)
    root.search_barrier = True
    verdict = _prove(root, loser, max_plies, budget, line)
    elapsed = _time.perf_counter() - started

    victory = None
    if verdict is Verdict.PROVEN:
        threat = sudden_death_threat(game, loser) or {}
        if "science_missing" in threat and "military_steps" not in threat:
            victory = VictoryType.SCIENTIFIC
        elif "military_steps" in threat and "science_missing" not in threat:
            victory = VictoryType.MILITARY
        # Both threats live: the proof covers either, so name neither.

    return Certificate(
        verdict=verdict,
        loser=loser,
        nodes=budget.nodes,
        seconds=round(elapsed, 3),
        stopped_by=None if verdict is not Verdict.UNKNOWN else budget.binding,
        limits_hit=tuple(sorted(budget.reasons)),
        victory_type=victory,
        principal_line=line,
    )
