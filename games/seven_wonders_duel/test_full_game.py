from collections import Counter

import pytest

from games.seven_wonders_duel.engine import ActionUse, apply_action, legal_actions
from games.seven_wonders_duel.game import Phase, VictoryType, new_game


def _scripted_discard_game(seed):
    game = new_game(seed)
    uses = Counter()
    while game.phase is not Phase.COMPLETE:
        actions = legal_actions(game)
        assert actions
        if game.phase is Phase.PLAY_AGE:
            action = next(action for action in actions if action.use is ActionUse.DISCARD_FOR_COINS)
        elif game.phase is Phase.CHOOSE_NEXT_START_PLAYER:
            action = next(action for action in actions if action.starting_player == 0)
        else:
            action = actions[0]
        uses[action.use] += 1
        apply_action(game, action)
    return game, uses


@pytest.mark.parametrize("seed", [0, 1, 999_983])
def test_complete_three_age_game_through_only_public_action_api(seed):
    game, uses = _scripted_discard_game(seed)
    assert uses[ActionUse.DRAFT_WONDER] == 8
    assert uses[ActionUse.DISCARD_FOR_COINS] == 60
    assert uses[ActionUse.CHOOSE_NEXT_START_PLAYER] == 2
    assert sum(uses.values()) == 70
    assert game.age == 3
    assert not any(card.present for card in game.tableau.cards.values())
    assert [city.coins for city in game.cities] == [67, 67]
    assert game.final_scores == (22, 22)
    assert game.winner is None
    assert game.victory_type is VictoryType.SHARED_CIVILIAN
    assert legal_actions(game) == ()


def test_full_game_is_reproducible_including_wonder_draft():
    first, first_uses = _scripted_discard_game(2026)
    second, second_uses = _scripted_discard_game(2026)
    assert first.setup_fingerprint() == second.setup_fingerprint()
    assert [city.wonders for city in first.cities] == [city.wonders for city in second.cities]
    assert first.final_scores == second.final_scores
    assert first_uses == second_uses

