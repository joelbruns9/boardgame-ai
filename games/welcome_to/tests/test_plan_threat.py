"""Threat predicates -- ENCODER_V3_SPEC.md §10.2 and §10.2a.

Two things are checked here, and the first is the one that matters:

**The one-turn ceiling is tested over EVERY `steps_left`, not just the range the
predicate chooses to enumerate.** Restricting the test to the range the code
already explores makes it structurally incapable of catching a ceiling that is
too low -- which is precisely the defect review found in the first draft, where
`ESTATE` was given a ceiling of 3 and a single SURVEYOR fence can close two
steps at once.

**The joint draw is checked against brute force**, because a marginal cannot
answer "does any of three stacks supply it": the three cards are drawn without
replacement and are correlated.
"""
from __future__ import annotations

import itertools
import random

import pytest

from games.welcome_to.bots import RandomBot, play_match
from games.welcome_to.constants import Effect
from games.welcome_to.game import (
    GameConfig,
    GameState,
    TurnEnumerationExhausted,
    _estate_houses_short,
    _one_turn_hopeless,
    _p_any_stack_supplies,
    bis_usable,
    can_complete_this_turn,
    max_houses_this_turn,
    one_turn_ceiling,
    one_turn_sheets,
    p_complete_next_turn,
)
from games.welcome_to.plans import (
    DEALT_PLAN_IDS,
    PLANS,
    PlanKind,
    can_be_scored,
    progress,
)


@pytest.fixture(scope="module")
def late_states() -> list[GameState]:
    """Finished random games -- dense sheets, where the enumeration is affordable."""
    config = GameConfig(players=2, advanced=True)
    return [
        play_match([RandomBot(seed=s), RandomBot(seed=s + 900)], seed=s, config=config)
        for s in range(12)
    ]


# ──────────────────────────────────────────────────────────────────────────
# §10.2 -- the ceiling, over every steps_left
# ──────────────────────────────────────────────────────────────────────────
def test_no_single_turn_beats_the_ceiling(late_states):
    """The assertion that would have failed on the draft's ESTATE ceiling of 3."""
    checked = 0
    for state in late_states:
        for seat, sheet in enumerate(state.sheets):
            offers = state.visible_cards(seat)
            for plan_id in DEALT_PLAN_IDS:
                plan = PLANS[plan_id]
                before = progress(plan, sheet)[1]
                ceiling = one_turn_ceiling(plan)
                for candidate in one_turn_sheets(
                    sheet, offers, advanced=True, cap=40_000
                ):
                    gained = before - progress(plan, candidate)[1]
                    assert gained <= ceiling, (
                        f"plan {plan_id} ({plan.kind.name}) advanced {gained} steps "
                        f"in one turn, above its ceiling of {ceiling}"
                    )
                    checked += 1
    assert checked > 0


def test_a_single_fence_can_close_two_estate_steps():
    """Why ESTATE gets no early exit: one SURVEYOR fence, two steps."""
    # A run of six must be FENCED at its right end to be an estate at all --
    # `estates()` needs every box of a fence-delimited region written, and boxes
    # 6..9 of street 0 are empty.
    sheet = _sheet([[1, 2, 3, 4, 5, 6]])
    sheet.fences[0][5] = True
    before = progress(PLANS[2], sheet)[1]
    plan = PLANS[2]  # requires (3, 3, 3)
    assert [sz for _, _, sz in sheet.free_estates()] == [6]
    sheet.fences[0][2] = True  # split 6 into 3 + 3
    after = progress(plan, sheet)[1]
    assert before - after == 2
    assert one_turn_ceiling(plan) == len(plan.required_sizes)


# ──────────────────────────────────────────────────────────────────────────
# §10.2 -- the predicate, both directions
# ──────────────────────────────────────────────────────────────────────────
def test_can_complete_this_turn_is_bidirectional(late_states):
    """True **iff** some legal one-turn sequence ends with the plan scoreable."""
    for state in late_states:
        for slot in range(3):
            plan = PLANS[state.plan_ids[slot]]
            for seat in range(state.config.players):
                if seat in state.plan_turns_for(0, slot):
                    continue  # banked; covered separately
                sheet = state.sheet_for(0, seat)
                offers = state.visible_cards(0)
                truth = can_be_scored(plan, sheet) or any(
                    can_be_scored(plan, s)
                    for s in one_turn_sheets(
                        sheet, offers, advanced=state.config.advanced, cap=40_000
                    )
                )
                assert can_complete_this_turn(state, 0, seat, slot) is truth


def test_the_enumeration_cap_never_binds_on_a_reachable_state(late_states):
    """A capped search that answered "no threat" is worse than no feature."""
    for state in late_states:
        for seat in range(state.config.players):
            sheet = state.sheet_for(0, seat)
            offers = state.visible_cards(0)
            try:
                for _ in one_turn_sheets(sheet, offers, advanced=True):
                    pass
            except TurnEnumerationExhausted:  # pragma: no cover
                pytest.fail("one-turn enumeration cap bound on a reachable state")


def test_a_banked_plan_is_never_a_threat():
    """`can_be_scored` reads only the sheet, so SEVEN_TEMP stays true forever."""
    for seed in range(200):
        state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
        slot = next(
            (
                i
                for i, pid in enumerate(state.plan_ids)
                if PLANS[pid].kind is PlanKind.SEVEN_TEMP
            ),
            None,
        )
        if slot is not None:
            break
    assert slot is not None, "no seed in 0..199 dealt a SEVEN_TEMP plan"
    state.sheets[0].temps = 7
    assert can_be_scored(PLANS[state.plan_ids[slot]], state.sheets[0])
    assert can_complete_this_turn(state, 0, 0, slot)
    state.plan_turns[slot] = {0: 1}  # now banked
    assert can_complete_this_turn(state, 0, 0, slot) is False
    assert p_complete_next_turn(state, 0, 0, slot) == 0.0


# ──────────────────────────────────────────────────────────────────────────
# §10.2 -- the cheap pre-filter is sound
# ──────────────────────────────────────────────────────────────────────────
def test_the_hopeless_filter_never_hides_a_real_threat(late_states):
    for state in late_states:
        for slot in range(3):
            plan = PLANS[state.plan_ids[slot]]
            for seat in range(state.config.players):
                sheet = state.sheet_for(0, seat)
                if not _one_turn_hopeless(plan, sheet):
                    continue
                offers = state.visible_cards(0)
                assert not any(
                    can_be_scored(plan, s)
                    for s in one_turn_sheets(sheet, offers, advanced=True, cap=40_000)
                ), f"plan {state.plan_ids[slot]} filtered as hopeless but completable"


def test_estate_houses_short_is_zero_for_non_estate_plans():
    sheet = _sheet([[1, 2, 3]])
    for plan_id in DEALT_PLAN_IDS:
        plan = PLANS[plan_id]
        if plan.kind is not PlanKind.ESTATE:
            assert _estate_houses_short(plan, sheet) == 0


def test_an_empty_sheet_cannot_build_four_estates_in_one_turn():
    sheet = _sheet([[]])
    plan = PLANS[1]  # requires (2, 2, 2, 2) -- eight houses
    assert _estate_houses_short(plan, sheet) == 8
    assert _one_turn_hopeless(plan, sheet)


# ──────────────────────────────────────────────────────────────────────────
# §10.2a -- the joint draw
# ──────────────────────────────────────────────────────────────────────────
def _brute_force_any_supplies(counts: dict[int, int], wanted: list[set[int]]) -> float:
    """P(some stack gets a wanted number) by enumerating ordered card triples."""
    cards = [n for n, k in counts.items() for _ in range(k)]
    total = hits = 0
    for triple in itertools.permutations(range(len(cards)), 3):
        total += 1
        if any(cards[triple[i]] in wanted[i] for i in range(3)):
            hits += 1
    return hits / total


def test_the_joint_draw_matches_brute_force():
    """Falling factorials, not a product of marginals."""
    import numpy as np

    from games.welcome_to.constants import NUMBER_INDEX

    rng = random.Random(11)
    for _ in range(25):
        counts = {n: rng.randint(0, 3) for n in range(1, 7)}
        if sum(counts.values()) < 3:
            continue
        deck = np.zeros(15)
        for n, k in counts.items():
            deck[NUMBER_INDEX[n]] = k
        wanted = [set(rng.sample(range(1, 7), rng.randint(0, 3))) for _ in range(3)]
        got = _p_any_stack_supplies(deck, np.zeros(15), wanted)
        expected = _brute_force_any_supplies(counts, wanted)
        assert abs(got - expected) < 1e-9, (counts, wanted, got, expected)


def test_the_joint_draw_is_not_the_independent_approximation():
    import numpy as np

    from games.welcome_to.constants import NUMBER_INDEX

    deck = np.zeros(15)
    deck[NUMBER_INDEX[5]] = 4
    deck[NUMBER_INDEX[6]] = 4
    wanted = [{5}, {5}, {5}]
    exact = _p_any_stack_supplies(deck, np.zeros(15), wanted)
    independent = 1.0 - (1.0 - 4 / 8) ** 3
    assert abs(exact - independent) > 1e-3


# ──────────────────────────────────────────────────────────────────────────
# §6.4 scope, and §8's house ceiling
# ──────────────────────────────────────────────────────────────────────────
def test_the_threat_predicates_are_scoped_to_standard_mode():
    """Expert stacks are private; solo has no opponent and no known effects."""
    solo = GameState.new(seed=6, config=GameConfig(players=1))
    assert not solo.config.standard
    for slot in range(3):
        assert can_complete_this_turn(solo, 0, 0, slot) is False
        assert p_complete_next_turn(solo, 0, 0, slot) == 0.0


def test_a_turn_can_place_three_houses():
    """roundabout -> choose + write -> bis, which an earlier draft capped at two.

    ⚠ A bis needs a BIS combination to be **offered**, not merely a candidate on
    the sheet -- so this searches for a state that actually offers one rather
    than asserting on a seed that happened to.
    """
    for seed in range(300):
        state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
        if not any(e is Effect.BIS for _, e in state.visible_cards(0)):
            continue
        state.sheets[0].numbers[0][3] = 8  # a neighbour for the bis to copy
        if max_houses_this_turn(state, 0, 0) == 3:
            return
    pytest.fail("no seed in 0..299 offered a BIS with room for three houses")


def test_three_houses_needs_a_bis_OFFER_not_just_a_candidate():
    """The defect review found: independent predicates summed to an illegal 3."""
    for seed in range(300):
        state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
        if any(e is Effect.BIS for _, e in state.visible_cards(0)):
            continue
        state.sheets[0].numbers[0][3] = 8  # a bis candidate exists on the sheet...
        assert bis_usable(state, 0, 0)
        # ...but no BIS is offered, so no single turn reaches three houses.
        assert max_houses_this_turn(state, 0, 0) == 2
        return
    pytest.fail("every seed offered a BIS")


def _sheet(rows):
    from games.welcome_to.sheet import Sheet

    sheet = Sheet.new()
    for x, row in enumerate(rows):
        for y, n in enumerate(row):
            sheet.numbers[x][y] = n
    return sheet


# ──────────────────────────────────────────────────────────────────────────
# The differential oracle -- the only test here that is NOT self-referential
#
# Every other assertion in this file compares `can_complete_this_turn` against
# `one_turn_sheets`, i.e. the model against itself.  That proves consistency and
# nothing about legality: while those tests were green, the model let a PARK mark
# cross streets, ended a turn on a bare roundabout, and made the automatic TEMP
# mark optional.  All three were found here, by walking the real engine.
# ──────────────────────────────────────────────────────────────────────────
def _sheet_key(sheet):
    return (
        tuple(tuple(r) for r in sheet.numbers),
        tuple(tuple(r) for r in sheet.is_bis),
        tuple(tuple(r) for r in sheet.fences),
        tuple(sheet.parks),
        tuple(sheet.pools),
        sheet.temps,
        sheet.bis_marks,
        sheet.roundabouts,
    )


@pytest.mark.parametrize("seed", [3, 11, 29, 47, 58, 101])
def test_the_turn_model_matches_the_engine_exactly(seed):
    """`one_turn_sheets` must reach every engine outcome, and invent none."""
    from games.welcome_to.tests.engine_turn_oracle import engine_one_turn_sheets

    state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
    engine = {_sheet_key(s) for s in engine_one_turn_sheets(state, 0, cap=400_000)}
    model = {
        _sheet_key(s)
        for s in one_turn_sheets(
            state.sheets[0], state.visible_cards(0), advanced=True, cap=400_000
        )
    }
    assert not (engine - model), (
        f"{len(engine - model)} engine outcomes the model MISSES -- a threat the "
        f"predicate cannot see"
    )
    assert not (model - engine), (
        f"{len(model - engine)} sheets the model INVENTS -- a threat the engine "
        f"cannot produce"
    )


def test_the_temp_mark_is_mandatory():
    """`stActionTemp` crosses a box off with no decision; there is no pass."""
    from games.welcome_to.game import _resolve_effect

    sheet = _sheet([[]])
    results = list(_resolve_effect(sheet, Effect.TEMP, (0, 0)))
    assert all(s.temps == 1 for s in results), "a TEMP turn left the track untouched"
    optional = list(_resolve_effect(sheet, Effect.PARK, (0, 0)))
    assert any(s.parks == [0, 0, 0] for s in optional), "PARK should be passable"


def test_a_park_mark_cannot_cross_streets():
    """`Actions/Park` binds the mark to the street of the house just written."""
    from games.welcome_to.game import _resolve_effect

    sheet = _sheet([[]])
    for street in range(3):
        results = list(_resolve_effect(sheet, Effect.PARK, (street, 0)))
        touched = {x for s in results for x in range(3) if s.parks[x] > 0}
        assert touched <= {street}, f"park from street {street} reached {touched}"
