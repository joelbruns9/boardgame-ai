import pytest

from games.seven_wonders_duel.engine import (
    Action,
    ActionUse,
    apply_action,
    resolve_pending_choice,
    score_player,
    start_next_age,
)
from games.seven_wonders_duel.game import PendingChoiceKind, Phase, VictoryType, new_game


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


def _leave_only_slot(game, slot_id):
    for candidate_id, card in game.tableau.cards.items():
        card.present = candidate_id == slot_id
        if card.present:
            card.revealed = True


def test_military_enters_both_penalty_zones_one_space_at_a_time():
    game = _playing_game()
    game.conflict_position = 3
    game.cities[0].coins = 100
    game.cities[0].progress_tokens = ["Strategy"]
    game.cities[1].coins = 10
    slot = _put_in_accessible_slot(game, "Arsenal")
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_BUILDING))
    assert game.conflict_position == 7  # three printed Shields + Strategy
    assert game.cities[1].coins == 3
    assert 4 not in game.military_tokens_remaining
    assert 7 not in game.military_tokens_remaining


def test_military_penalty_cannot_reduce_coins_below_zero():
    game = _playing_game()
    game.conflict_position = 3
    game.cities[1].coins = 1
    slot = _put_in_accessible_slot(game, "Guard Tower")
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_BUILDING))
    assert game.cities[1].coins == 0


def test_reaching_capital_declares_immediate_military_victory():
    game = _playing_game()
    game.conflict_position = 8
    slot = _put_in_accessible_slot(game, "Guard Tower")
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_BUILDING))
    assert game.phase is Phase.COMPLETE
    assert game.winner == 0
    assert game.victory_type is VictoryType.MILITARY


def test_circus_destroys_target_before_its_shield_resolves():
    game = _playing_game()
    _give_wonder(game, 0, "Circus Maximus")
    game.cities[0].coins = 100
    game.cities[1].buildings = ["Glassworks"]
    game.conflict_position = 8
    slot = game.tableau.accessible_slot_ids()[0]
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_WONDER, "Circus Maximus"))
    assert game.phase is Phase.PLAY_AGE
    assert game.pending_choice is not None
    assert game.conflict_position == 8
    resolve_pending_choice(game, "Glassworks")
    assert game.conflict_position == 9
    assert game.victory_type is VictoryType.MILITARY


def test_second_identical_science_symbol_requires_progress_choice():
    game = _playing_game()
    game.cities[0].buildings = ["Workshop"]
    game.available_progress_tokens = (
        "Agriculture",
        "Architecture",
        "Economy",
        "Masonry",
        "Mathematics",
    )
    slot = _put_in_accessible_slot(game, "Laboratory")
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_BUILDING))
    assert game.pending_choice is not None
    assert game.pending_choice.kind is PendingChoiceKind.CHOOSE_AVAILABLE_PROGRESS
    assert game.pending_choice.options == game.available_progress_tokens
    coins_before = game.cities[0].coins
    resolve_pending_choice(game, "Agriculture")
    assert game.cities[0].coins == coins_before + 6
    assert "Agriculture" not in game.available_progress_tokens
    assert game.active_player == 1


def test_sixth_distinct_symbol_wins_before_turn_passes():
    game = _playing_game()
    game.cities[0].buildings = [
        "Workshop",
        "Apothecary",
        "Scriptorium",
        "Pharmacist",
        "Academy",
    ]
    game.cities[0].coins = 100
    slot = _put_in_accessible_slot(game, "University")
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_BUILDING))
    assert game.phase is Phase.COMPLETE
    assert game.winner == 0
    assert game.victory_type is VictoryType.SCIENTIFIC


def test_law_from_progress_can_supply_the_sixth_distinct_symbol():
    game = _playing_game()
    game.cities[0].buildings = [
        "Workshop",
        "Apothecary",
        "Scriptorium",
        "Pharmacist",
        "Academy",
    ]
    game.available_progress_tokens = (
        "Law",
        "Agriculture",
        "Architecture",
        "Economy",
        "Masonry",
    )
    slot = _put_in_accessible_slot(game, "Laboratory")
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_BUILDING))
    resolve_pending_choice(game, "Law")
    assert game.winner == 0
    assert game.victory_type is VictoryType.SCIENTIFIC


def test_theology_grants_replay_to_future_wonder():
    game = _playing_game()
    _give_wonder(game, 0, "The Pyramids")
    game.cities[0].progress_tokens = ["Theology"]
    game.cities[0].coins = 100
    slot = game.tableau.accessible_slot_ids()[0]
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_WONDER, "The Pyramids"))
    assert game.active_player == 0


@pytest.mark.parametrize(
    ("position", "player_zero", "player_one"),
    [
        (0, 0, 0),
        (1, 2, 0),
        (3, 2, 0),
        (4, 5, 0),
        (6, 5, 0),
        (7, 10, 0),
        (8, 10, 0),
        (-4, 0, 5),
    ],
)
def test_military_civilian_scoring_bands(position, player_zero, player_one):
    game = _playing_game()
    game.conflict_position = position
    assert score_player(game, 0).military == player_zero
    assert score_player(game, 1).military == player_one


def test_score_breakdown_includes_guild_wonders_progress_and_treasury():
    game = _playing_game()
    game.conflict_position = 5
    game.cities[0].buildings = ["Palace", "Workshop", "Builders Guild"]
    game.cities[0].built_wonders = ["The Pyramids", "The Sphinx"]
    game.cities[0].progress_tokens = ["Agriculture", "Mathematics"]
    game.cities[0].coins = 8
    score = score_player(game, 0)
    assert score.military == 5
    assert score.buildings == 12  # Palace 7 + Workshop 1 + Builders Guild 4
    assert score.wonders == 15
    assert score.progress == 10  # Agriculture 4 + Mathematics 3 x 2 tokens
    assert score.treasury == 2
    assert score.total == 44
    assert score.blue_buildings == 7


def test_end_of_age_uses_weaker_military_player_as_next_age_chooser():
    game = _playing_game()
    game.conflict_position = 2  # player 1 is weaker
    slot = game.tableau.accessible_slot_ids()[0]
    _leave_only_slot(game, slot)
    apply_action(game, Action(slot, ActionUse.DISCARD_FOR_COINS))
    assert game.phase is Phase.CHOOSE_NEXT_START_PLAYER
    assert game.active_player == 1
    start_next_age(game, 0)
    assert game.age == 2
    assert game.active_player == 0
    assert len(game.tableau.accessible_slot_ids()) == 2


def test_civilian_tie_is_broken_by_blue_building_points():
    game = _playing_game()
    game.age = 3
    game.cities[0].coins = 0
    game.cities[0].buildings = ["Theater"]
    game.cities[1].coins = 3
    game.cities[1].buildings = ["Workshop", "Apothecary"]
    slot = game.tableau.accessible_slot_ids()[0]
    _leave_only_slot(game, slot)
    apply_action(game, Action(slot, ActionUse.DISCARD_FOR_COINS))
    assert game.final_scores == (3, 3)
    assert game.winner == 0
    assert game.victory_type is VictoryType.CIVILIAN


def test_second_civilian_tie_is_shared():
    game = _playing_game()
    game.age = 3
    game.cities[0].coins = 0
    game.cities[1].coins = 0
    slot = game.tableau.accessible_slot_ids()[0]
    _leave_only_slot(game, slot)
    apply_action(game, Action(slot, ActionUse.DISCARD_FOR_COINS))
    assert game.final_scores == (0, 0)
    assert game.winner is None
    assert game.victory_type is VictoryType.SHARED_CIVILIAN
