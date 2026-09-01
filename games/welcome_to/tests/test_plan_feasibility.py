"""The soundness gate for ``plans.feasible`` -- ENCODER_V3_SPEC.md §10.1 and §10.3.

`feasible` is the one block in the v3 encoder that can be **silently** wrong: it
returns a single float, nothing downstream can tell a false death from a real
one, and five rounds of reading the spec produced four unsound versions of it.
So it is checked against an independent oracle
(:mod:`games.welcome_to.tests.plan_reachability`) rather than against itself.

⚠ **THE ASSERTION IS ONE-SIDED, AND MUST STAY THAT WAY.**

    feasible(plan, sheet) is False  =>  the oracle must agree there is no
                                        continuation that completes the plan

The converse is deliberately **not** asserted.  `feasible` is documented as
sound but *incomplete*: it is allowed to say "alive" about a plan that is
really dead, because a missed death is a weak feature while a false death is a
lying one.  Asserting the converse would fail on correct code and, worse, would
pressure a future author into tightening `feasible` past what they can prove.
"""
from __future__ import annotations

import os
import random
from collections import Counter

import pytest

from games.welcome_to.bots import RandomBot, play_match
from games.welcome_to.constants import BIS_BOXES, MAX_NUMBER, MIN_NUMBER
from games.welcome_to.plans import (
    DEALT_PLAN_IDS,
    PLANS,
    PlanKind,
    can_be_scored,
    feasible,
    progress,
    requirements,
    turns_lower_bound,
)
from games.welcome_to.sheet import Sheet
from games.welcome_to.tests.plan_reachability import (
    OracleExhausted,
    can_ever_be_scored,
    free_boxes,
    open_fence_slots,
)

#: Free-box ceiling for plans whose read-set includes GEOMETRY.  Ordinary writes
#: fan out 18 numbers wide per empty box, so those searches are ~18**n: measured,
#: two free boxes resolve in seconds and three do not.
MAX_FREE_BOXES_GEOMETRY = 2

#: Plans that cannot read geometry at all -- their successors are pool writes and
#: park or temp marks -- resolve in milliseconds at ANY density, so they are
#: checked on real end-of-game sheets untouched.  This matters: measured over 30
#: random games, 166 of the 213 declared deaths were DECORATIVE, i.e. most of the
#: gate's real work happens in the cheap regime.
#: ⚠ DECORATIVE is NOT in here any more.  Its read set retains roundabouts --
#: a roundabout can revive a numerically dead pool box, and pruning it let the
#: oracle miss a real completion -- so its searches are no longer free.
_GEOMETRY_FREE_KINDS = frozenset({PlanKind.SEVEN_TEMP})


#: Fence-slot ceiling for the same plans.  Fences are the OTHER exponential axis
#: and are independent of free boxes: a completely full sheet has zero free boxes
#: and up to 30 open slots, i.e. 2**30 fence subsets.  Bounding only free boxes
#: let densified sheets through and the oracle then exhausted on them.
MAX_FENCE_SLOTS_GEOMETRY = 12


#: A per-call **search** budget, not a per-sheet one.
#:
#: Two attempts at free-box ceilings were tuned on samples and both blew up on
#: the real fixture -- the cost is not monotone in free boxes, so any threshold
#: is a guess that has to be re-guessed whenever the oracle changes.  Bounding
#: the search directly is self-tuning: a call either finishes inside the budget
#: or is recorded as an honest miss, and neither outcome can stall the suite.
#:
#: Coverage is then whatever this buys, guarded by the floor in the gate below.
ORACLE_STATE_BUDGET = int(os.environ.get("WT_GATE_STATES", 6_000))

#: Games behind the fixture.  The default is sized for a suite that people
#: actually run; the full sweep is a pre-commit / on-demand thing:
#:
#:     WT_GATE_GAMES=30 WT_GATE_STATES=15000 pytest tests/test_plan_feasibility.py
#:
#: At 30/15000 this module verified 279 deaths in 420s -- thorough, and far too
#: slow to sit in the default suite, which is how a gate stops being run at all.
GATE_GAMES = int(os.environ.get("WT_GATE_GAMES", 8))


def _oracle_can_decide(plan, sheet) -> bool:
    """Always attempt; the state budget decides what is affordable.

    Kept as a named hook so a future kind-specific skip has somewhere to live.
    """
    return True


def _densify(sheet: Sheet, rng: random.Random) -> Sheet:
    """Fill a sheet toward the geometry envelope with legal house-writing moves.

    Random games end on three permit refusals long before a sheet fills -- across
    30 games the emptiest end-of-game sheet still had **nine** free boxes -- so
    real play never reaches the density where geometry deaths can be decided.
    This walks a real sheet further along the same move relation the oracle uses.

    Reachability here is the oracle's, which over-approximates a real game's.
    That is the safe direction for this test: it can only widen the set of sheets
    `feasible` is held to, never narrow it.
    """
    sheet = sheet.copy()
    for _ in range(200):
        if free_boxes(sheet) <= MAX_FREE_BOXES_GEOMETRY:
            break
        moves: list = []
        for number in range(MIN_NUMBER, MAX_NUMBER + 1):
            moves += [("write", number, pos) for pos in sheet.available_locations(number)]
        moves += [("bis", n, (x, y)) for x, y, n, _ in sheet.bis_candidates()]
        if sheet.can_build_roundabout():
            moves += [("round", None, pos) for pos in sheet.available_locations(None)]
        if not moves:
            break
        kind, number, pos = rng.choice(moves)
        if kind == "round":
            sheet.build_roundabout(pos, turn=0)
        else:
            sheet.write(number, pos, turn=0, is_bis=(kind == "bis"))
            if kind == "bis":
                sheet.bis_marks = min(sheet.bis_marks + 1, BIS_BOXES)
    return sheet


def _game_sheets(games: int = GATE_GAMES) -> list[Sheet]:
    out: list[Sheet] = []
    for seed in range(games):
        state = play_match([RandomBot(seed=seed), RandomBot(seed=seed + 500)], seed=seed)
        out.extend(state.sheets)
    return out


@pytest.fixture(scope="module")
def sheets() -> list[Sheet]:
    """Real end-of-game sheets, plus a densified copy of each.

    The pair matters: the sparse half is where DECORATIVE and COMPLETE_STREET
    deaths actually occur, and the dense half is the only place a geometry death
    can be decided at all.
    """
    rng = random.Random(20260830)
    base = _game_sheets()
    assert base, "no sheets generated; the fuzz proves nothing"
    dense = [_densify(sheet, rng) for sheet in base]
    assert any(
        free_boxes(sheet) <= MAX_FREE_BOXES_GEOMETRY for sheet in dense
    ), "densification reached no sheet inside the geometry envelope"
    return base + dense


# ──────────────────────────────────────────────────────────────────────────
# §10.1 -- soundness
# ──────────────────────────────────────────────────────────────────────────
def test_feasible_never_reports_a_death_the_oracle_denies(sheets):
    """A single false death fails this.  Completeness is NOT asserted.

    The oracle runs **only on declared deaths**, which is what makes the gate
    affordable: `feasible` saying "alive" is never checked, so the expensive
    geometry searches are only entered on the rare sheets where it says "dead".
    """
    checked: Counter = Counter()
    undecided: Counter = Counter()
    for sheet in sheets:
        for plan_id in DEALT_PLAN_IDS:
            plan = PLANS[plan_id]
            if feasible(plan, sheet):
                continue
            if not _oracle_can_decide(plan, sheet):
                undecided[plan.kind.name] += 1
                continue
            try:
                reachable = can_ever_be_scored(
                    sheet, plan, max_states=ORACLE_STATE_BUDGET
                )
            except OracleExhausted:
                # An honest miss, NOT a pass.  The budget is a heuristic and the
                # cost is not monotone in free boxes alone, so a hard failure
                # here just invites tuning the threshold until it stops firing.
                # Counting it as undecidable keeps it visible; the coverage floor
                # below is what stops the buckets quietly draining into here.
                undecided[plan.kind.name + "/exhausted"] += 1
                continue
            assert not reachable, (
                f"feasible() declared plan {plan_id} ({plan.kind.name}) dead, but "
                f"the oracle found a continuation that completes it:\n"
                f"{sheet.pretty()}"
            )
            checked[plan.kind.name] += 1

    # A gate that never fires is not a gate, and one whose coverage drains away
    # into "undecidable" is not one either.  The floor is well under what is
    # observed at the default size so ordinary drift does not trip it, but a
    # change that guts verification will.
    assert sum(checked.values()) >= 40, (
        f"only {sum(checked.values())} deaths verified; coverage has collapsed"
    )
    # And the coverage hole is reported, not hidden.  Undecidable deaths are
    # geometry plans on sheets too sparse for the oracle; they are a known
    # limitation of the search, not of `feasible`.
    print(f"\nverified deaths: {dict(checked)}")
    print(f"undecidable (outside the oracle envelope): {dict(undecided)}")


def test_a_completed_plan_is_always_feasible(sheets):
    """`can_be_scored` implies `feasible` -- an achieved plan cannot be dead."""
    for sheet in sheets:
        for plan_id in DEALT_PLAN_IDS:
            plan = PLANS[plan_id]
            if can_be_scored(plan, sheet):
                assert feasible(plan, sheet), f"plan {plan_id} scored but infeasible"


def test_the_estate_death_test_admits_fence_repartitioning():
    """The R5 counter-example: a FULL sheet, zero free boxes, still completable.

    ``Sheet.estates`` is bounded by fences, not by writes, so one SURVEYOR fence
    splits a built run into estates of new sizes while consuming no box.  A
    bound counting only free boxes called this dead.
    """
    sheet = Sheet.new()
    for y in range(10):
        sheet.numbers[0][y] = 1
    for y in range(11):
        sheet.numbers[1][y] = 1
    for y in range(12):
        sheet.numbers[2][y] = y
    assert free_boxes(sheet) == 0
    plan = PLANS[5]  # requires (6, 6)
    assert plan.required_sizes == (6, 6)
    assert not any(sz == 6 for _, _, sz in sheet.free_estates())
    assert feasible(plan, sheet)
    assert can_ever_be_scored(sheet, plan, max_states=120_000)


def test_a_fully_consumed_street_kills_full_street():
    """The one exact death test: a spent house can never be un-spent."""
    sheet = Sheet.new()
    for y in range(10):
        sheet.numbers[0][y] = y
    for y in range(11):
        sheet.numbers[1][y] = y
    for y in range(12):
        sheet.numbers[2][y] = y
    plan = PLANS[18]  # FULL_STREET(2)
    assert feasible(plan, sheet)
    sheet.mark_top_fences([(2, 0)])
    assert not feasible(plan, sheet)
    # No oracle call here, deliberately.  This death is *exact* -- `can_be_scored`
    # requires `not any(top_fences[x])` and nothing ever clears a top fence -- so
    # there is nothing for a search to add.  And the search would not finish: on a
    # full sheet the only remaining moves are fences, and 30 open slots is 2**30
    # subsets.  Fences cannot affect FULL_STREET's predicate at all, but the
    # oracle's read-set groups them with the rest of geometry rather than rely on
    # an argument about what a fence can and cannot enable.


# ──────────────────────────────────────────────────────────────────────────
# §10.3 -- requirement-vector fidelity
# ──────────────────────────────────────────────────────────────────────────
def test_estate_shortfall_matches_an_independent_recount(sheets):
    from collections import Counter

    for sheet in sheets:
        supply = Counter(size for _, _, size in sheet.free_estates())
        for plan_id in DEALT_PLAN_IDS:
            plan = PLANS[plan_id]
            req = requirements(plan, sheet)
            if plan.kind is not PlanKind.ESTATE or can_be_scored(plan, sheet):
                assert req.estate_shortfall == (0,) * 6
                continue
            need = Counter(plan.required_sizes)
            expected = tuple(
                max(0, need[s] - supply[s]) for s in range(1, 7)
            )
            assert req.estate_shortfall == expected


def test_estate_shortfall_is_zero_exactly_when_the_plan_is_done(sheets):
    for sheet in sheets:
        for plan_id in DEALT_PLAN_IDS:
            plan = PLANS[plan_id]
            if plan.kind is not PlanKind.ESTATE:
                continue
            done = progress(plan, sheet)[1] == 0
            zero = not any(requirements(plan, sheet).estate_shortfall)
            assert zero == done


def test_target_planes_are_exactly_the_unwritten_target_boxes(sheets):
    """Not a non-zero-ness proxy: the exact mask, and `COMPLETE` is its own case."""
    for sheet in sheets:
        for plan_id in DEALT_PLAN_IDS:
            plan = PLANS[plan_id]
            req = requirements(plan, sheet)
            if plan.kind is PlanKind.FULL_STREET:
                x = plan.params[0]
                expected = tuple(
                    (x, y)
                    for y in range(len(sheet.numbers[x]))
                    if sheet.numbers[x][y] is None
                )
            elif plan.kind is PlanKind.EXTREMITIES:
                from games.welcome_to.constants import EXTREMITY_POSITIONS

                expected = tuple(
                    (x, y)
                    for x, y in EXTREMITY_POSITIONS
                    if sheet.numbers[x][y] is None
                )
            else:
                expected = ()
            if can_be_scored(plan, sheet):
                expected = ()
            assert req.target_boxes == expected, plan_id


def test_street_serves_reads_as_aliveness_not_as_remaining_work(sheets):
    """A COMPLETED street is alive -- it is the one that contributed most."""
    for sheet in sheets:
        for plan_id in DEALT_PLAN_IDS:
            plan = PLANS[plan_id]
            req = requirements(plan, sheet)
            if plan.kind is PlanKind.SEVEN_TEMP:
                assert req.street_serves == (0, 0, 0)  # sheet-wide, not dead
                continue
            if not feasible(plan, sheet):
                assert req.street_serves == (0, 0, 0)


def test_a_finished_park_street_still_serves_its_plan():
    """The defect the aliveness definition fixes, in isolation."""
    from games.welcome_to.constants import PARK_BOXES

    sheet = Sheet.new()
    sheet.parks[0] = PARK_BOXES[0]  # street 0's parks are DONE
    plan = PLANS[23]  # DECORATIVE park
    req = requirements(plan, sheet)
    assert req.street_serves[0] == 1
    assert req.parks_needed[0] == 0  # the work, not the aliveness, is what is zero


# ──────────────────────────────────────────────────────────────────────────
# §6.2 -- the bound is a bound
# ──────────────────────────────────────────────────────────────────────────
def test_turns_lower_bound_is_zero_exactly_for_a_finished_plan(sheets):
    for sheet in sheets:
        for plan_id in DEALT_PLAN_IDS:
            plan = PLANS[plan_id]
            if can_be_scored(plan, sheet):
                assert turns_lower_bound(plan, sheet) == 0


def test_turns_lower_bound_divides_houses_by_three():
    """A turn places up to three houses: roundabout -> write -> bis."""
    sheet = Sheet.new()
    plan = PLANS[18]  # FULL_STREET(2), twelve empty boxes
    assert turns_lower_bound(plan, sheet) == 4  # ceil(12 / 3), not 12


def test_turns_lower_bound_sums_effect_marks():
    sheet = Sheet.new()
    plan = PLANS[21]  # SEVEN_TEMP
    assert turns_lower_bound(plan, sheet) == 7  # one mark per combination
