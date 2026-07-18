from collections import Counter

from games.seven_wonders_duel.data import (
    AGE_I_CARDS,
    AGE_II_CARDS,
    AGE_III_CARDS,
    ALL_BUILDING_CARDS,
    CARDS_BY_NAME,
    GUILD_CARDS,
    PROGRESS_TOKENS,
    TABLEAU_LAYOUTS,
    WONDERS,
    CardColor,
    covering_slots,
)


def test_component_counts_match_rulebook():
    assert len(AGE_I_CARDS) == 23
    assert len(AGE_II_CARDS) == 23
    assert len(AGE_III_CARDS) == 20
    assert len(GUILD_CARDS) == 7
    assert len(PROGRESS_TOKENS) == 10
    assert len(WONDERS) == 12


def test_all_component_names_are_unique():
    assert len(CARDS_BY_NAME) == len(ALL_BUILDING_CARDS)
    assert len({token.name for token in PROGRESS_TOKENS}) == 10
    assert len({wonder.name for wonder in WONDERS}) == 12


def test_age_three_has_no_brown_or_grey_cards():
    assert not {card.color for card in AGE_III_CARDS} & {CardColor.BROWN, CardColor.GREY}


def test_each_science_symbol_printed_on_buildings_appears_twice():
    counts = Counter(card.science for card in ALL_BUILDING_CARDS if card.science)
    assert set(counts.values()) == {2}
    assert len(counts) == 6


def test_every_chain_requirement_has_exactly_one_provider():
    providers = Counter(card.chain_to for card in ALL_BUILDING_CARDS if card.chain_to)
    for card in ALL_BUILDING_CARDS:
        if card.chain_from:
            assert providers[card.chain_from] == 1, card.name


def test_tableau_layouts_have_twenty_cards_and_correct_initial_accessibility():
    expected_accessible = {1: 6, 2: 2, 3: 2}
    for age, layout in TABLEAU_LAYOUTS.items():
        assert len(layout) == 20
        accessible = [slot for slot in layout if not covering_slots(layout, slot)]
        assert len(accessible) == expected_accessible[age]
        assert all(slot.face_up for slot in accessible)


def test_all_wonder_costs_are_sourced_and_nonempty():
    assert all(wonder.cost is not None for wonder in WONDERS)
    assert all(wonder.cost.total_resources > 0 for wonder in WONDERS if wonder.cost)
    assert all(wonder.cost_source for wonder in WONDERS)

