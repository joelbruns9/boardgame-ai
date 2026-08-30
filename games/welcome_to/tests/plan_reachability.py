"""Test-only oracle: can *any* continuation of this sheet satisfy this plan?

This is the reference `ENCODER_V3_SPEC.md` §10.1 checks `plans.feasible` against,
and it is deliberately written before `feasible` itself.  Five review rounds of
that spec produced four unsound death tests, and every one of them would have
been caught here — but only if this file is right.  A wrong oracle does not fail
loudly; it silently blesses the bug it was built to catch.  So the design
choices are spelled out rather than assumed.

WHAT IT ANSWERS
───────────────
Given a sheet and a plan, is there **any** sequence of legal sheet mutations
after which :func:`plans.can_be_scored` is true?

DELIBERATE OVER-APPROXIMATION — and why that is the safe direction
──────────────────────────────────────────────────────────────────
The oracle assumes an **unlimited deck**: any number, any effect, any number of
turns, in any order.  It therefore explores a strict superset of what a real
game can reach.

That asymmetry is the whole point:

* if the oracle finds **no** completion, none exists, so `feasible` returning
  ``0.0`` is justified;
* if the oracle finds a completion the real game could never reach, the test
  fails **loudly** on a correct `feasible`, and a human investigates.

The dangerous direction is the other one — an oracle that *misses* a completion
would silently confirm an unsound death test.  Nothing here may under-explore.

WHAT IS STILL ENFORCED
──────────────────────
Everything that binds regardless of what the deck offers:

* the strictly-ascending rule for ordinary writes (``available_locations``);
* ``surveyor_zones`` legality — no fence may split a bis pair or two houses
  already spent on the same City Plan;
* top fences already on the sheet, which no action removes;
* ``ROUNDABOUT_BOXES`` and ``TEMP_BOXES``, which really do gate their actions.

⚠ ``BIS_BOXES`` does **not** gate anything.  ``legal_actions`` offers
``bis_candidates()`` at ``ACTION_BIS`` without consulting ``bis_marks``, and the
apply path saturates the counter rather than refusing.  A sheet at nine marks
keeps placing bis houses — free of further penalty — so a bis cap here would
under-explore, which is the one direction this oracle may not take.

THE SIX WAYS A SHEET CHANGES
────────────────────────────
Enumerated exhaustively, because the recurring defect in the spec was reasoning
about *numbers* and forgetting the actions that write a house without one:

1. an ordinary write — any legal number at any legal box;
2. a write at a pool position that also builds the pool (``Actions/Pool`` binds
   the pool to the street of the house just written);
3. a **roundabout** — ``available_locations(None)`` ignores numeric fit entirely,
   and the sentinel counts as a built house;
4. a **bis** — ``bis_candidates`` has no ascending-order check at all;
5. a **fence** — creates estates of new sizes while consuming no box;
6. a park or temp track mark.

Estate-row marks are omitted: no plan kind requires the ESTATE *effect*, only
estate sizes, which come from fences.

TERMINATION
───────────
Every move strictly increases a monotone quantity — filled boxes, fences, or a
track mark — and none is ever undone, so the state graph is a DAG and the
visited set terminates the search.

⚠ THE CAP RAISES.  If the search exceeds ``max_states`` it raises
:class:`OracleExhausted` rather than returning ``False``.  A truncated search
that answered "unreachable" would confirm exactly the bugs this exists to find.
"""
from __future__ import annotations

from typing import Iterator, Optional

from games.welcome_to.constants import (
    BIS_BOXES,
    MAX_NUMBER,
    MIN_NUMBER,
    POOL_BOXES,
    POOL_POSITION_SET,
    ROUNDABOUT,
    ROUNDABOUT_BOXES,
    TEMP_BOXES,
)
from games.welcome_to.plans import Plan, PlanKind, can_be_scored
from games.welcome_to.sheet import Sheet


class OracleExhausted(RuntimeError):
    """The search cap bound before the space was covered.  Never a verdict."""


def _key(sheet: Sheet) -> tuple:
    return (
        tuple(tuple(row) for row in sheet.numbers),
        # `is_bis` is plan-relevant: can_be_scored(FIVE_BIS) reads it through
        # bis_count_per_street.  Two states with identical numbers but different
        # bis attribution are NOT the same state, and merging them here would
        # skip a scoring state before it was ever tested.
        tuple(tuple(row) for row in sheet.is_bis),
        tuple(tuple(row) for row in sheet.fences),
        tuple(tuple(row) for row in sheet.top_fences),
        tuple(sheet.parks),
        tuple(sheet.pools),
        sheet.temps,
        sheet.bis_marks,
        sheet.roundabouts,
    )


#: Which sheet fields each plan kind's ``can_be_scored`` predicate actually reads.
#:
#: ``GEOMETRY`` covers ``numbers``, ``is_bis``, ``fences`` and ``top_fences``.
#: They are one cluster because they are mutually load-bearing: a write changes
#: what ``bis_candidates`` offers, a fence changes what ``estates`` partitions
#: into, and every geometry move's own legality is a function of geometry.
#:
#: ``PARKS``, ``POOLS`` and ``TEMPS`` are **leaves**: nothing else on the sheet
#: reads them, and no action's legality depends on them.  (A pool is *built* by a
#: geometry move -- a write at a pool box -- but the counter it increments is
#: read by nothing else.)
GEOMETRY, FENCES, PARKS, POOLS, TEMPS = (
    "geometry",
    "fences",
    "parks",
    "pools",
    "temps",
)

#: Read off each ``can_be_scored`` branch in `plans.py`, field by field:
#:
#: * ``ESTATE``          -- ``free_estates`` -> ``estates`` -> numbers, FENCES, top fences
#: * ``FULL_STREET``     -- ``numbers[x]`` and ``top_fences[x]``
#: * ``EXTREMITIES``     -- ``numbers`` and ``top_fences`` at six named boxes
#: * ``FIVE_BIS``        -- ``bis_count_per_street`` -> ``is_bis``
#: * ``SEVEN_TEMP``      -- ``temps``
#: * ``DECORATIVE``      -- ``street_parks_complete`` / ``street_pools_complete``
#: * ``COMPLETE_STREET`` -- those two plus ``has_roundabout_in_street`` -> numbers
#:
#: ``GEOMETRY`` gates the three house-writing moves (write, bis, roundabout);
#: ``FENCES`` gates the standalone surveyor move on its own, because **only
#: ESTATE reads the fence grid**.
_READS: dict[PlanKind, frozenset[str]] = {
    PlanKind.ESTATE: frozenset({GEOMETRY, FENCES}),
    PlanKind.FULL_STREET: frozenset({GEOMETRY}),
    PlanKind.FIVE_BIS: frozenset({GEOMETRY}),
    PlanKind.EXTREMITIES: frozenset({GEOMETRY}),
    PlanKind.SEVEN_TEMP: frozenset({TEMPS}),
    PlanKind.DECORATIVE: frozenset({PARKS, POOLS}),
    PlanKind.COMPLETE_STREET: frozenset({PARKS, POOLS, GEOMETRY}),
}


def _reads(plan: Plan) -> frozenset[str]:
    """What ``can_be_scored(plan, ...)`` can observe.

    ⚠ This is a **mechanical** claim -- read the predicate and list the fields it
    touches -- deliberately NOT a strategic one like "a fence can only remove a
    bis option".  A move that cannot change any field the predicate reads cannot
    change its answer, and that argument holds without knowing anything about how
    the game is played.  Strategic pruning is how an oracle silently starts
    under-exploring, which is the one thing this file may not do.

    ⚠ **The ``FENCES`` axis rests on one step past field-disjointness, accepted
    deliberately on 2026-08-30.** A fence is *not* field-disjoint from the rest of
    geometry: ``bis_candidates`` tests ``not fence[y - 1]`` and ``surveyor_zones``
    tests ``not fences[x][j]``. But in both places the fence appears **only** as a
    conjunct requiring it to be ``False``, so setting one to ``True`` can only
    *remove* entries from those two sets and can never add one. It appears in no
    other move's enabling condition -- not ``available_locations`` (writes and
    roundabouts), not the park or temp marks.

    Therefore fencing never enables a completion that was otherwise unreachable,
    and a plan whose predicate does not read the fence grid may skip the move
    entirely. Only ``ESTATE`` reads it.

    This is a monotonicity argument, one notch stronger than the mechanical
    disjointness the rest of this table rests on. It is checkable by reading two
    functions, and it is what makes ``COMPLETE_STREET`` decidable at all: without
    it the search drags 2**n fence subsets behind 120 park combinations.

    (The incidental fencing done by ``build_roundabout`` is untouched -- that is
    part of a move which is kept, not pruned.)
    """
    try:
        return _READS[plan.kind]
    except KeyError:
        raise NotImplementedError(
            f"plan kind {plan.kind} has no declared read-set; add one rather "
            f"than defaulting, or the oracle will prune something it needs"
        ) from None


def _successors(sheet: Sheet, reads: frozenset[str]) -> Iterator[Sheet]:
    """Every sheet one legal action away, restricted to moves ``reads`` can see.

    ``reads`` never removes a *reachable* state from the answer -- only states
    that differ from an explored one solely in fields the predicate cannot
    observe.  Without it the search drags 120 park combinations and 12 temp
    values behind every geometry state: a 1440x multiplier on a query that, for
    four of the seven plan kinds, cannot read either.
    """
    geometry = GEOMETRY in reads
    # 1 + 2. ordinary writes, and the pool that a write at a pool box unlocks
    if geometry or POOLS in reads:
        for number in range(MIN_NUMBER, MAX_NUMBER + 1):
            for pos in sheet.available_locations(number):
                if geometry:
                    nxt = sheet.copy()
                    nxt.write(number, pos, turn=0)
                    yield nxt
                if (
                    POOLS in reads
                    and pos in POOL_POSITION_SET
                    and sheet.pool_count < POOL_BOXES
                ):
                    pooled = sheet.copy()
                    pooled.write(number, pos, turn=0)
                    pooled.pools[pos[0]] += 1
                    yield pooled

    # 3. a roundabout ignores numeric fit and still counts as a built house
    if geometry and sheet.roundabouts < ROUNDABOUT_BOXES:
        for pos in sheet.available_locations(None):
            nxt = sheet.copy()
            nxt.build_roundabout(pos, turn=0)
            yield nxt

    # 4. a bis copies a neighbour -- no ascending-order check, and NO CAP.
    #    `legal_actions` at ACTION_BIS offers `bis_candidates()` without
    #    consulting `bis_marks`, and the apply path saturates with
    #    `min(bis_marks + 1, BIS_BOXES)`.  So past nine marks a bis is free of
    #    its own penalty and still writes a house -- the same shape as the temp
    #    track.  Only the supply of empty boxes bounds these successors.
    for x, y, number, _side in sheet.bis_candidates() if geometry else ():
        nxt = sheet.copy()
        nxt.write(number, (x, y), turn=0, is_bis=True)
        nxt.bis_marks = min(nxt.bis_marks + 1, BIS_BOXES)
        yield nxt

    # 5. a fence re-partitions built runs into estates of new sizes,
    #    consuming no box at all
    for x, j in sheet.surveyor_zones() if FENCES in reads else ():
        nxt = sheet.copy()
        nxt.fences[x][j] = True
        yield nxt

    # 6. track marks
    if PARKS in reads:
        for x in sheet.park_streets():
            nxt = sheet.copy()
            nxt.parks[x] += 1
            yield nxt
    if TEMPS in reads and sheet.temps < TEMP_BOXES:
        nxt = sheet.copy()
        nxt.temps += 1
        yield nxt


def can_ever_be_scored(
    sheet: Sheet, plan: Plan, *, max_states: int = 200_000
) -> bool:
    """Does some continuation of ``sheet`` satisfy ``plan``?

    Raises :class:`OracleExhausted` if ``max_states`` binds — see the module
    docstring.  Callers must not catch it and treat it as ``False``.
    """
    if can_be_scored(plan, sheet):
        return True

    reads = _reads(plan)
    seen: set[tuple] = {_key(sheet)}
    stack: list[Sheet] = [sheet]
    while stack:
        current = stack.pop()
        for nxt in _successors(current, reads):
            key = _key(nxt)
            if key in seen:
                continue
            if len(seen) >= max_states:
                raise OracleExhausted(
                    f"exceeded {max_states} states proving reachability of "
                    f"plan {plan.id}; shrink the fuzz sheet or raise the cap"
                )
            seen.add(key)
            if can_be_scored(plan, nxt):
                return True
            stack.append(nxt)
    return False


def free_boxes(sheet: Sheet) -> int:
    """How much search a sheet implies — the fuzz uses this to stay tractable."""
    return sum(1 for row in sheet.numbers for n in row if n is None)


def open_fence_slots(sheet: Sheet) -> int:
    return len(sheet.surveyor_zones())
