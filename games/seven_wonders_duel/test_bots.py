import random

import pytest

from games.seven_wonders_duel.bots import (
    GreedyBot,
    MilitaryAggressiveBot,
    MilitaryEconomyBot,
    RandomBot,
    ScienceAggressiveBot,
    ScienceEconomyBot,
    play_game,
    play_series,
)
from games.seven_wonders_duel.codec import encode_action
from games.seven_wonders_duel.engine import ActionUse, apply_action, legal_actions
from games.seven_wonders_duel.game import (
    PendingChoice,
    PendingChoiceKind,
    Phase,
    new_game,
)
from games.seven_wonders_duel.pool import resample_hidden
from games.seven_wonders_duel.rust_bridge import rust_game_for_self_play


RUSH_BOTS = (
    ScienceAggressiveBot,
    ScienceEconomyBot,
    MilitaryAggressiveBot,
    MilitaryEconomyBot,
)

RUST_BOTS = (
    (GreedyBot, "greedy"),
    (ScienceAggressiveBot, "science_aggressive/v1"),
    (ScienceEconomyBot, "science_economy/v1"),
    (MilitaryAggressiveBot, "military_aggressive/v1"),
    (MilitaryEconomyBot, "military_economy/v1"),
)


def _playing_game(seed: int = 1):
    game = new_game(seed)
    while game.phase is Phase.WONDER_DRAFT:
        game.pick_wonder(game.legal_wonder_choices()[0])
    return game


def _only_accessible_cards(game, *card_names: str) -> None:
    slots = game.tableau.accessible_slot_ids()
    assert len(slots) >= len(card_names)
    kept = set(slots[: len(card_names)])
    for slot_id, tableau_card in game.tableau.cards.items():
        tableau_card.present = slot_id in kept
    for slot_id, card_name in zip(slots[: len(card_names)], card_names, strict=True):
        game.tableau.cards[slot_id].card_name = card_name
        game.tableau.cards[slot_id].revealed = True


def _chosen_card_name(game, action):
    assert action.slot_id is not None
    return game.tableau.cards[action.slot_id].card_name


def test_clone_is_independent_and_preserves_rng_stream():
    game = new_game(123)
    clone = game.clone()
    clone.cities[0].coins += 10
    assert game.cities[0].coins == 7
    assert clone.cities[0].coins == 17
    assert game.rng.random() == clone.rng.random()


def test_seeded_random_bots_choose_same_action_on_same_state():
    game = new_game(5)
    first = RandomBot(88)
    second = RandomBot(88)
    assert first.select_action(game) == second.select_action(game)


def test_greedy_selection_is_legal_deterministic_and_nonmutating():
    game = new_game(9)
    bot = GreedyBot()
    before = repr(game)
    first = bot.select_action(game)
    second = bot.select_action(game)
    assert first == second
    assert first in legal_actions(game)
    assert repr(game) == before


@pytest.mark.parametrize("bot_type", RUSH_BOTS)
def test_rush_selection_is_legal_deterministic_and_nonmutating(bot_type):
    game = _playing_game(19)
    bot = bot_type()
    before = repr(game)
    first = bot.select_action(game)
    second = bot.select_action(game)
    assert first == second
    assert first in legal_actions(game)
    assert repr(game) == before


@pytest.mark.parametrize("bot_type", (ScienceAggressiveBot, ScienceEconomyBot))
def test_science_bots_take_a_missing_age_one_symbol_over_economy(bot_type):
    game = _playing_game(20)
    _only_accessible_cards(game, "Workshop", "Press")
    action = bot_type().select_action(game)
    assert action.use is ActionUse.CONSTRUCT_BUILDING
    assert _chosen_card_name(game, action) == "Workshop"


@pytest.mark.parametrize("bot_type", (MilitaryAggressiveBot, MilitaryEconomyBot))
def test_military_bots_take_an_age_one_shield_over_economy(bot_type):
    game = _playing_game(21)
    _only_accessible_cards(game, "Guard Tower", "Lumber Yard")
    action = bot_type().select_action(game)
    assert action.use is ActionUse.CONSTRUCT_BUILDING
    assert _chosen_card_name(game, action) == "Guard Tower"


@pytest.mark.parametrize(
    ("bot_type", "focus_token"),
    ((ScienceAggressiveBot, "Law"), (MilitaryAggressiveBot, "Strategy")),
)
def test_aggressive_bots_choose_their_route_progress_token(bot_type, focus_token):
    game = _playing_game(22)
    game.pending_choice = PendingChoice(
        PendingChoiceKind.CHOOSE_AVAILABLE_PROGRESS,
        game.active_player,
        (focus_token, "Agriculture"),
    )
    action = bot_type().select_action(game)
    assert action.use is ActionUse.RESOLVE_PENDING_CHOICE
    assert action.choice == focus_token


@pytest.mark.parametrize(
    ("bot_type", "producer"),
    ((ScienceEconomyBot, "Press"), (MilitaryEconomyBot, "Lumber Yard")),
)
def test_economy_bots_build_route_support_between_focus_cards(bot_type, producer):
    game = _playing_game(23)
    _only_accessible_cards(game, producer, "Theater")
    action = bot_type().select_action(game)
    assert action.use is ActionUse.CONSTRUCT_BUILDING
    assert _chosen_card_name(game, action) == producer


@pytest.mark.parametrize("bot_type", RUSH_BOTS)
def test_rush_choice_is_invariant_to_hidden_resampling(bot_type):
    game = _playing_game(24)
    first_world = game.clone()
    second_world = game.clone()
    resample_hidden(first_world, random.Random(100))
    resample_hidden(second_world, random.Random(200))
    assert first_world.observation(game.active_player) == second_world.observation(
        game.active_player
    )
    first = bot_type(seed=7, exploration=1.0).select_action(first_world)
    second = bot_type(seed=7, exploration=1.0).select_action(second_world)
    assert first == second


@pytest.mark.parametrize("bot_type", RUSH_BOTS)
def test_each_rush_bot_can_complete_a_game(bot_type):
    result = play_game(bot_type(), RandomBot(31), seed=25)
    assert result.actions <= 256
    assert result.victory_type is not None


def test_random_match_reaches_a_valid_terminal_state():
    result = play_game(RandomBot(1), RandomBot(2), seed=44)
    assert result.actions <= 256
    assert result.victory_type is not None
    if result.final_scores is None:
        assert result.winner in (0, 1)


def test_greedy_match_handles_every_decision_phase():
    result = play_game(GreedyBot(), RandomBot(3), seed=45)
    assert result.actions <= 256
    assert result.victory_type is not None


def test_series_alternates_bot_seats_and_accounts_for_every_game():
    result = play_series(RandomBot(10), RandomBot(20), games=6, seed=100)
    assert result.games == 6
    assert result.bot_a_wins + result.bot_b_wins + result.draws == 6
    assert result.military + result.scientific + result.civilian == 6
    assert result.average_actions > 0


def test_terminal_state_has_no_bot_action():
    game = new_game(1)
    game.phase = Phase.COMPLETE
    assert legal_actions(game) == ()


@pytest.mark.parametrize(("bot_type", "rust_name"), RUST_BOTS)
def test_rust_bot_matches_python_for_a_complete_deterministic_game(
    bot_type, rust_name
):
    seed = 2026072300
    game = new_game(seed, first_player=1)
    rust = rust_game_for_self_play(seed, 1)
    bot = bot_type()
    moves = 0
    while game.phase is not Phase.COMPLETE:
        expected = encode_action(game, bot.select_action(game))
        assert rust.bot_action(rust_name) == expected
        apply_action(game, bot.select_action(game))
        rust.apply_index(expected)
        moves += 1
        assert moves <= 256
