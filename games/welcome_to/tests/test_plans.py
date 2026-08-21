"""City Plan completion predicates."""
from __future__ import annotations

import pytest

from games.welcome_to.constants import EXTREMITY_POSITIONS, ROUNDABOUT, STREET_SIZES
from games.welcome_to.plans import (
    NUM_PLANS,
    PLANS,
    PlanKind,
    Variant,
    available_plan_ids,
    can_be_scored,
    estates_matching_size,
    validation_cells,
)
from games.welcome_to.sheet import Sheet


def _plan(kind: PlanKind, params=None):
    for plan in PLANS:
        if plan.kind is kind and (params is None or plan.params == params):
            return plan
    raise AssertionError(f"no plan {kind} {params}")


def _fill_street(sheet: Sheet, x: int) -> None:
    for y in range(STREET_SIZES[x]):
        sheet.numbers[x][y] = y


# ──────────────────────────────────────────────────────────────────────────
# The table itself
# ──────────────────────────────────────────────────────────────────────────
def test_plan_ids_are_their_own_index():
    assert NUM_PLANS == 37
    for i, plan in enumerate(PLANS):
        assert plan.id == i


def test_basic_game_deals_only_basic_plans():
    for stack in (1, 2, 3):
        ids = available_plan_ids(stack, advanced=False)
        assert len(ids) == 6
        assert all(PLANS[i].variant is Variant.BASIC for i in ids)


def test_advanced_variant_enlarges_the_first_two_stacks():
    assert len(available_plan_ids(1, advanced=True)) == 11
    assert len(available_plan_ids(2, advanced=True)) == 11
    assert len(available_plan_ids(3, advanced=True)) == 6


def test_seasonal_plans_are_never_dealt():
    dealt = {i for stack in (1, 2, 3) for i in available_plan_ids(stack, True)}
    seasonal = {
        p.id
        for p in PLANS
        if p.variant in (Variant.ICE_CREAM, Variant.CHRISTMAS, Variant.EASTER)
    }
    assert dealt & seasonal == set()


def test_only_estate_plans_ask_the_player_anything():
    for plan in PLANS:
        if plan.variant not in (Variant.BASIC, Variant.ADVANCED):
            continue
        assert plan.is_automatic == (plan.kind is not PlanKind.ESTATE)


# ──────────────────────────────────────────────────────────────────────────
# Estate plans
# ──────────────────────────────────────────────────────────────────────────
def test_estate_plan_needs_every_required_size():
    plan = PLANS[3]  # sizes (4, 4)
    assert plan.required_sizes == (4, 4)

    sheet = Sheet.new()
    _fill_street(sheet, 0)
    sheet.fences[0][3] = True  # sizes 4 and 6
    assert not can_be_scored(plan, sheet)

    sheet.fences[0][7] = True  # sizes 4, 4, 2
    assert can_be_scored(plan, sheet)


def test_estate_plan_cannot_reuse_a_consumed_estate():
    plan = PLANS[3]
    sheet = Sheet.new()
    _fill_street(sheet, 0)
    sheet.fences[0][3] = True
    sheet.fences[0][7] = True
    assert can_be_scored(plan, sheet)

    sheet.top_fences[0][0] = True  # spends the first size-4 estate
    assert not can_be_scored(plan, sheet)


def test_estates_matching_size_skips_already_chosen():
    sheet = Sheet.new()
    _fill_street(sheet, 0)
    sheet.fences[0][3] = True
    sheet.fences[0][7] = True
    fours = estates_matching_size(sheet, 4)
    assert fours == [(0, 0, 4), (0, 4, 4)]
    assert estates_matching_size(sheet, 4, ((0, 0, 4),)) == [(0, 4, 4)]


def test_validating_an_estate_plan_consumes_its_houses():
    plan = PLANS[3]
    sheet = Sheet.new()
    _fill_street(sheet, 0)
    cells = validation_cells(plan, sheet, ((0, 0, 4), (0, 4, 4)))
    assert cells == [(0, y) for y in range(8)]


# ──────────────────────────────────────────────────────────────────────────
# Advanced plans
# ──────────────────────────────────────────────────────────────────────────
def test_full_street_plan_wants_the_whole_street_unspent():
    plan = _plan(PlanKind.FULL_STREET, (0,))
    sheet = Sheet.new()
    assert not can_be_scored(plan, sheet)

    _fill_street(sheet, 0)
    assert can_be_scored(plan, sheet)

    sheet.top_fences[0][5] = True
    assert not can_be_scored(plan, sheet)


def test_full_street_plan_accepts_a_roundabout_as_a_built_house():
    plan = _plan(PlanKind.FULL_STREET, (0,))
    sheet = Sheet.new()
    _fill_street(sheet, 0)
    sheet.numbers[0][4] = ROUNDABOUT
    assert can_be_scored(plan, sheet)


def test_five_bis_plan_counts_one_street_only():
    plan = _plan(PlanKind.FIVE_BIS)
    sheet = Sheet.new()
    for y in range(3):
        sheet.is_bis[0][y] = True
    for y in range(3):
        sheet.is_bis[1][y] = True
    assert not can_be_scored(plan, sheet)

    sheet.is_bis[0][3] = True
    sheet.is_bis[0][4] = True
    assert can_be_scored(plan, sheet)


def test_seven_temp_plan():
    plan = _plan(PlanKind.SEVEN_TEMP)
    sheet = Sheet.new()
    sheet.temps = 6
    assert not can_be_scored(plan, sheet)
    sheet.temps = 7
    assert can_be_scored(plan, sheet)


def test_extremities_plan_wants_both_ends_of_every_street():
    plan = _plan(PlanKind.EXTREMITIES)
    sheet = Sheet.new()
    for x, y in EXTREMITY_POSITIONS:
        sheet.numbers[x][y] = 1
    assert can_be_scored(plan, sheet)

    sheet.top_fences[2][11] = True
    assert not can_be_scored(plan, sheet)

    sheet.top_fences[2][11] = False
    sheet.numbers[1][0] = None
    assert not can_be_scored(plan, sheet)


def test_decorative_park_plan_wants_two_finished_streets():
    plan = _plan(PlanKind.DECORATIVE, ("park",))
    sheet = Sheet.new()
    sheet.parks = [3, 4, 0]
    assert can_be_scored(plan, sheet)
    sheet.parks = [3, 3, 0]
    assert not can_be_scored(plan, sheet)


def test_decorative_pool_plan_wants_two_finished_streets():
    plan = _plan(PlanKind.DECORATIVE, ("pool",))
    sheet = Sheet.new()
    sheet.pools = [3, 3, 1]
    assert can_be_scored(plan, sheet)
    sheet.pools = [3, 2, 1]
    assert not can_be_scored(plan, sheet)


def test_decorative_combined_plan_names_one_street():
    plan = _plan(PlanKind.DECORATIVE, ("pool&park", 1))
    sheet = Sheet.new()
    sheet.parks = [3, 4, 5]
    sheet.pools = [3, 0, 3]
    assert not can_be_scored(plan, sheet)
    sheet.pools = [3, 3, 3]
    assert can_be_scored(plan, sheet)


def test_complete_street_plan_needs_parks_pools_and_a_roundabout():
    plan = _plan(PlanKind.COMPLETE_STREET)
    sheet = Sheet.new()
    sheet.parks = [3, 0, 0]
    sheet.pools = [3, 0, 0]
    assert not can_be_scored(plan, sheet)

    sheet.numbers[0][5] = ROUNDABOUT
    assert can_be_scored(plan, sheet)


@pytest.mark.parametrize("plan_id", [28, 31, 34])
def test_seasonal_plans_refuse_to_be_scored(plan_id):
    with pytest.raises(NotImplementedError):
        can_be_scored(PLANS[plan_id], Sheet.new())
