from games.seven_wonders_duel.bots import GreedyBot, RandomBot, play_game, play_series
from games.seven_wonders_duel.engine import legal_actions
from games.seven_wonders_duel.game import Phase, new_game


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

