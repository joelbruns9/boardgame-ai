import pytest

from games.seven_wonders_duel.rules import (
    Resource,
    discard_income,
    normal_trade_unit_cost,
    trade_cost,
    treasury_victory_points,
)


def test_trade_unit_cost_is_two_plus_opponents_matching_production():
    assert normal_trade_unit_cost(0) == 2
    assert normal_trade_unit_cost(2) == 4


def test_rulebook_trade_example_costs_five_coins():
    missing = {Resource.CLAY: 1, Resource.PAPYRUS: 1}
    opponent = {Resource.CLAY: 1, Resource.PAPYRUS: 0}
    assert trade_cost(missing, opponent) == 5


def test_commercial_discount_fixes_selected_resource_at_one_coin():
    missing = {Resource.STONE: 2, Resource.GLASS: 1}
    opponent = {Resource.STONE: 3, Resource.GLASS: 1}
    assert trade_cost(missing, opponent, frozenset({Resource.STONE})) == 5


@pytest.mark.parametrize(("yellow_cards", "income"), [(0, 2), (2, 4)])
def test_discard_income(yellow_cards, income):
    assert discard_income(yellow_cards) == income


@pytest.mark.parametrize(("coins", "points"), [(0, 0), (2, 0), (3, 1), (8, 2)])
def test_treasury_scoring(coins, points):
    assert treasury_victory_points(coins) == points


def test_negative_counts_are_rejected():
    with pytest.raises(ValueError):
        normal_trade_unit_cost(-1)
    with pytest.raises(ValueError):
        discard_income(-1)
    with pytest.raises(ValueError):
        treasury_victory_points(-1)

