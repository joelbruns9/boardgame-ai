import pytest

from games.seven_wonders_duel.data import CARDS_BY_NAME, Cost
from games.seven_wonders_duel.engine import (
    Action,
    ActionUse,
    apply_action,
    legal_actions,
    minimum_payment,
    resolve_pending_choice,
    score_player,
)
from games.seven_wonders_duel.game import PendingChoiceKind, Phase, new_game
from games.seven_wonders_duel.rules import Resource


def _playing_game(seed=1):
    game = new_game(seed)
    while game.phase is Phase.WONDER_DRAFT:
        apply_action(game, legal_actions(game)[0])
    return game


def _put_in_accessible_slot(game, card_name):
    target = game.tableau.accessible_slot_ids()[0]
    source = next(
        (slot_id for slot_id, card in game.tableau.cards.items() if card.card_name == card_name),
        None,
    )
    if source is None:
        game.tableau.cards[target].card_name = card_name
    else:
        game.tableau.cards[target].card_name, game.tableau.cards[source].card_name = (
            game.tableau.cards[source].card_name,
            game.tableau.cards[target].card_name,
        )
    return target


def _give_wonder(game, player, wonder_name):
    for city in game.cities:
        if wonder_name in city.wonders:
            city.wonders.remove(wonder_name)
    game.cities[player].wonders.insert(0, wonder_name)


def _build_wonder(game, wonder_name):
    _give_wonder(game, game.active_player, wonder_name)
    game.cities[game.active_player].coins = 100
    slot = game.tableau.accessible_slot_ids()[0]
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_WONDER, wonder_name))


@pytest.mark.parametrize(
    ("card_name", "existing_buildings", "built_wonders", "expected_award"),
    [
        ("Tavern", [], [], 4),
        ("Brewery", [], [], 6),
        ("Chamber of Commerce", ["Glassworks", "Press"], [], 6),
        ("Port", ["Lumber Yard", "Clay Pool"], [], 4),
        ("Armory", ["Guard Tower", "Stable"], [], 2),
        ("Lighthouse", ["Tavern", "Stone Reserve"], [], 3),
        ("Arena", [], ["The Pyramids", "The Sphinx", "Piraeus"], 6),
    ],
)
def test_commercial_card_coin_awards(
    card_name, existing_buildings, built_wonders, expected_award
):
    game = _playing_game()
    city = game.cities[0]
    city.coins = 100
    city.buildings = list(existing_buildings)
    city.built_wonders = list(built_wonders)
    card = CARDS_BY_NAME[card_name]
    payment = minimum_payment(game, 0, card.cost, card=card)
    slot = _put_in_accessible_slot(game, card_name)
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_BUILDING))
    assert city.coins == 100 - payment.total_coins + expected_award


@pytest.mark.parametrize(
    ("guild_name", "player_zero_cards", "player_one_cards", "expected_award"),
    [
        ("Merchants Guild", [], ["Tavern", "Stone Reserve", "Forum"], 3),
        ("Shipowners Guild", [], ["Lumber Yard", "Clay Pool", "Glassworks", "Press"], 4),
        ("Magistrates Guild", [], ["Theater", "Altar", "Baths"], 3),
        ("Scientists Guild", [], ["Workshop", "Apothecary", "Scriptorium"], 3),
        ("Tacticians Guild", [], ["Guard Tower", "Stable", "Garrison"], 3),
    ],
)
def test_guild_construction_coin_awards(
    guild_name, player_zero_cards, player_one_cards, expected_award
):
    game = _playing_game()
    game.cities[0].coins = 100
    game.cities[0].buildings = list(player_zero_cards)
    game.cities[1].buildings = list(player_one_cards)
    guild = CARDS_BY_NAME[guild_name]
    payment = minimum_payment(game, 0, guild.cost, card=guild)
    slot = _put_in_accessible_slot(game, guild_name)
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_BUILDING))
    assert game.cities[0].coins == 100 - payment.total_coins + expected_award


@pytest.mark.parametrize(
    ("guild_name", "player_zero_cards", "player_one_cards", "wonders", "coins", "expected_vp"),
    [
        ("Merchants Guild", [], ["Tavern", "Stone Reserve", "Forum"], ([], []), (0, 0), 3),
        ("Shipowners Guild", [], ["Lumber Yard", "Clay Pool", "Glassworks", "Press"], ([], []), (0, 0), 4),
        ("Builders Guild", [], [], (["The Pyramids", "The Sphinx"], ["Piraeus"]), (0, 0), 4),
        ("Magistrates Guild", [], ["Theater", "Altar", "Baths"], ([], []), (0, 0), 3),
        ("Scientists Guild", [], ["Workshop", "Apothecary", "Scriptorium"], ([], []), (0, 0), 3),
        ("Moneylenders Guild", [], [], ([], []), (2, 11), 3),
        ("Tacticians Guild", [], ["Guard Tower", "Stable", "Garrison"], ([], []), (0, 0), 3),
    ],
)
def test_each_guild_final_scoring_formula(
    guild_name,
    player_zero_cards,
    player_one_cards,
    wonders,
    coins,
    expected_vp,
):
    game = _playing_game()
    game.cities[0].buildings = [*player_zero_cards, guild_name]
    game.cities[1].buildings = list(player_one_cards)
    game.cities[0].built_wonders = list(wonders[0])
    game.cities[1].built_wonders = list(wonders[1])
    game.cities[0].coins, game.cities[1].coins = coins
    with_guild = score_player(game, 0).buildings
    game.cities[0].buildings.remove(guild_name)
    without_guild = score_player(game, 0).buildings
    assert with_guild - without_guild == expected_vp


def test_appian_way_resolves_coins_loss_and_replay():
    game = _playing_game()
    game.cities[1].coins = 2
    _build_wonder(game, "The Appian Way")
    assert game.cities[0].coins == 93  # 100 - 10 cost + 3 coins
    assert game.cities[1].coins == 0
    assert game.active_player == 0


@pytest.mark.parametrize(
    ("wonder_name", "expected_coins"),
    [("The Hanging Gardens", 98), ("The Temple of Artemis", 104)],
)
def test_coin_and_replay_wonders(wonder_name, expected_coins):
    game = _playing_game()
    _build_wonder(game, wonder_name)
    assert game.cities[0].coins == expected_coins
    assert game.active_player == 0


@pytest.mark.parametrize("wonder_name", ["The Sphinx", "Piraeus"])
def test_other_printed_replay_wonders(wonder_name):
    game = _playing_game()
    _build_wonder(game, wonder_name)
    assert game.active_player == 0


@pytest.mark.parametrize(
    ("wonder_name", "cost", "expected_purchase"),
    [
        ("Piraeus", {Resource.GLASS: 1, Resource.PAPYRUS: 1}, 2),
        ("The Great Lighthouse", {Resource.WOOD: 1, Resource.CLAY: 1}, 2),
    ],
)
def test_resource_choice_wonders_produce_one_flexible_unit(
    wonder_name, cost, expected_purchase
):
    game = _playing_game()
    _build_wonder(game, wonder_name)
    payment = minimum_payment(
        game,
        0,
        Cost(
            wood=cost.get(Resource.WOOD, 0),
            clay=cost.get(Resource.CLAY, 0),
            glass=cost.get(Resource.GLASS, 0),
            papyrus=cost.get(Resource.PAPYRUS, 0),
        ),
    )
    assert payment.trade_coins == expected_purchase


def test_colossus_moves_military_two_spaces():
    game = _playing_game()
    _build_wonder(game, "The Colossus")
    assert game.conflict_position == 2


def test_statue_of_zeus_destroys_brown_then_moves_military():
    game = _playing_game()
    game.cities[1].buildings = ["Lumber Yard"]
    _build_wonder(game, "The Statue of Zeus")
    assert game.pending_choice is not None
    assert game.pending_choice.kind is PendingChoiceKind.DESTROY_OPPONENT_BROWN
    assert game.conflict_position == 0
    resolve_pending_choice(game, "Lumber Yard")
    assert game.cities[1].buildings == []
    assert game.conflict_position == 1


def test_mausoleum_builds_discarded_card_for_free_and_applies_effect():
    game = _playing_game()
    game.discard_pile = ["Tavern"]
    _build_wonder(game, "The Mausoleum")
    assert game.pending_choice is not None
    assert game.pending_choice.kind is PendingChoiceKind.BUILD_FROM_DISCARD_FREE
    coins_before = game.cities[0].coins
    resolve_pending_choice(game, "Tavern")
    assert "Tavern" in game.cities[0].buildings
    assert "Tavern" not in game.discard_pile
    assert game.cities[0].coins == coins_before + 4


def test_urbanism_grants_six_when_acquired_then_four_on_future_chain():
    game = _playing_game()
    game.available_progress_tokens = (
        "Urbanism",
        "Agriculture",
        "Architecture",
        "Economy",
        "Masonry",
    )
    game.cities[0].buildings = ["Workshop"]
    slot = _put_in_accessible_slot(game, "Laboratory")
    coins_before = game.cities[0].coins
    laboratory = CARDS_BY_NAME["Laboratory"]
    payment = minimum_payment(game, 0, laboratory.cost, card=laboratory)
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_BUILDING))
    resolve_pending_choice(game, "Urbanism")
    assert game.cities[0].coins == coins_before - payment.total_coins + 6

    game.active_player = 0
    game.cities[0].buildings.append("Baths")
    game.cities[0].coins = 0
    slot = _put_in_accessible_slot(game, "Aqueduct")
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_BUILDING))
    assert game.cities[0].coins == 4


def test_philosophy_and_mathematics_progress_scoring():
    game = _playing_game()
    game.cities[0].progress_tokens = ["Philosophy", "Mathematics", "Agriculture"]
    assert score_player(game, 0).progress == 20  # 7 + 4 + Mathematics 3 x 3
