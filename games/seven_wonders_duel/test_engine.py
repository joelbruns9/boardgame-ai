import pytest

from games.seven_wonders_duel.data import CARDS_BY_NAME, Cost
from games.seven_wonders_duel.engine import (
    Action,
    ActionUse,
    apply_action,
    legal_actions,
    minimum_payment,
    resolve_pending_choice,
)
from games.seven_wonders_duel.game import PendingChoiceKind, Phase, new_game
from games.seven_wonders_duel.rules import Resource


def _playing_game(seed=1):
    game = new_game(seed)
    while game.phase is Phase.WONDER_DRAFT:
        game.pick_wonder(game.legal_wonder_choices()[0])
    return game


def _put_in_accessible_slot(game, card_name):
    target = game.tableau.accessible_slot_ids()[0]
    source = next(
        (slot_id for slot_id, card in game.tableau.cards.items() if card.card_name == card_name),
        None,
    )
    if source is None:
        game.tableau.cards[target].card_name = card_name
        return target
    game.tableau.cards[target].card_name, game.tableau.cards[source].card_name = (
        game.tableau.cards[source].card_name,
        game.tableau.cards[target].card_name,
    )
    return target


def _give_wonder(game, player, wonder_name):
    for city in game.cities:
        if wonder_name in city.wonders:
            city.wonders.remove(wonder_name)
    game.cities[player].wonders[0:0] = [wonder_name]


def test_payment_uses_opponent_brown_grey_production_and_trade_discount():
    game = _playing_game()
    game.cities[0].buildings = ["Quarry"]
    game.cities[1].buildings = ["Shelf Quarry"]
    payment = minimum_payment(game, 0, Cost(stone=3))
    assert payment.trade_coins == 8
    assert payment.purchased == ((Resource.STONE, 2),)

    game.cities[0].buildings.append("Stone Reserve")
    discounted = minimum_payment(game, 0, Cost(stone=3))
    assert discounted.trade_coins == 2


def test_flexible_production_is_assigned_to_the_most_expensive_need():
    game = _playing_game()
    game.cities[0].buildings = ["Forum"]
    game.cities[1].buildings = ["Glassworks"]
    payment = minimum_payment(game, 0, Cost(glass=1, papyrus=1))
    assert payment.total_coins == 2
    assert payment.purchased == ((Resource.PAPYRUS, 1),)


def test_architecture_and_masonry_choose_the_best_two_resource_reductions():
    game = _playing_game()
    game.cities[0].buildings = ["Quarry"]
    game.cities[0].progress_tokens = ["Architecture"]
    wonder_payment = minimum_payment(game, 0, Cost(stone=1, glass=2), is_wonder=True)
    assert wonder_payment.total_coins == 0

    game.cities[0].progress_tokens = ["Masonry"]
    blue = CARDS_BY_NAME["Aqueduct"]
    building_payment = minimum_payment(game, 0, blue.cost, card=blue)
    assert building_payment.total_coins == 0


def test_chain_construction_is_free_and_urbanism_pays_four():
    game = _playing_game()
    game.cities[0].coins = 0
    game.cities[0].buildings = ["Baths"]
    game.cities[0].progress_tokens = ["Urbanism"]
    slot = _put_in_accessible_slot(game, "Aqueduct")
    payment = minimum_payment(game, 0, CARDS_BY_NAME["Aqueduct"].cost, card=CARDS_BY_NAME["Aqueduct"])
    assert payment.used_chain and payment.total_coins == 0

    apply_action(game, Action(slot, ActionUse.CONSTRUCT_BUILDING))
    assert "Aqueduct" in game.cities[0].buildings
    assert game.cities[0].coins == 4


def test_discard_income_counts_existing_yellow_buildings():
    game = _playing_game()
    game.cities[0].buildings = ["Tavern", "Stone Reserve"]
    slot = game.tableau.accessible_slot_ids()[0]
    card_name = game.tableau.cards[slot].card_name
    apply_action(game, Action(slot, ActionUse.DISCARD_FOR_COINS))
    assert game.cities[0].coins == 11
    assert game.discard_pile == [card_name]
    assert game.active_player == 1


def test_construct_building_pays_cost_and_applies_immediate_coins():
    game = _playing_game()
    slot = _put_in_accessible_slot(game, "Tavern")
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_BUILDING))
    assert game.cities[0].coins == 11
    assert game.cities[0].buildings == ["Tavern"]


def test_economy_receives_trade_spend_but_not_printed_coin_cost():
    game = _playing_game()
    game.cities[0].coins = 20
    game.cities[1].buildings = ["Clay Pool"]
    game.cities[1].progress_tokens = ["Economy"]
    slot = _put_in_accessible_slot(game, "Forum")
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_BUILDING))
    assert game.cities[0].coins == 14  # 3 printed + clay at 2 + opponent's clay symbol
    assert game.cities[1].coins == 10  # starts at 7 and receives only the 3 trade coins


def test_temple_of_artemis_build_buries_card_gains_coins_and_replays():
    game = _playing_game()
    _give_wonder(game, 0, "The Temple of Artemis")
    game.cities[0].coins = 100
    slot = game.tableau.accessible_slot_ids()[0]
    buried = game.tableau.cards[slot].card_name
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_WONDER, "The Temple of Artemis"))
    assert game.cities[0].coins == 104  # 100 - four resources at 2 each + 12
    assert game.cities[0].built_wonders == ["The Temple of Artemis"]
    assert game.buried_cards == [buried]
    assert game.active_player == 0


def test_destructive_wonder_creates_public_pending_choice():
    game = _playing_game()
    _give_wonder(game, 0, "Circus Maximus")
    game.cities[0].coins = 100
    game.cities[1].buildings = ["Glassworks", "Press"]
    slot = game.tableau.accessible_slot_ids()[0]
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_WONDER, "Circus Maximus"))
    assert game.pending_choice is not None
    assert game.pending_choice.kind is PendingChoiceKind.DESTROY_OPPONENT_GREY
    assert set(game.pending_choice.options) == {"Glassworks", "Press"}
    assert game.observation(0).pending_choice == game.pending_choice
    assert {
        action.choice for action in legal_actions(game)
    } == {"Glassworks", "Press"}

    choice_action = next(
        action for action in legal_actions(game) if action.choice == "Glassworks"
    )
    apply_action(game, choice_action)
    assert game.cities[1].buildings == ["Press"]
    assert game.discard_pile == ["Glassworks"]
    assert game.active_player == 1


def test_great_library_draw_is_seeded_and_consumes_all_three_options():
    game = _playing_game(400)
    _give_wonder(game, 0, "The Great Library")
    game.cities[0].coins = 100
    slot = game.tableau.accessible_slot_ids()[0]
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_WONDER, "The Great Library"))
    pending = game.pending_choice
    assert pending is not None
    assert pending.kind is PendingChoiceKind.CHOOSE_UNUSED_PROGRESS
    assert len(pending.options) == 3
    chosen = pending.options[0]
    resolve_pending_choice(game, chosen)
    assert chosen in game.cities[0].progress_tokens
    assert len(game.unused_progress_tokens) == 2


def test_seventh_built_wonder_retires_the_only_remaining_wonder():
    game = _playing_game(700)
    game.cities[0].built_wonders = list(game.cities[0].wonders[:3])
    game.cities[1].built_wonders = list(game.cities[1].wonders[:3])
    target = game.cities[0].wonders[3]
    should_retire = game.cities[1].wonders[3]
    game.cities[0].coins = 100
    slot = game.tableau.accessible_slot_ids()[0]
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_WONDER, target))
    assert should_retire in game.retired_wonders
    assert target in game.cities[0].built_wonders


def test_illegal_or_malformed_action_does_not_mutate_tableau():
    game = _playing_game()
    before = tuple(card.present for card in game.tableau.cards.values())
    with pytest.raises(ValueError):
        apply_action(game, Action((0, 5), ActionUse.DISCARD_FOR_COINS))
    assert tuple(card.present for card in game.tableau.cards.values()) == before
