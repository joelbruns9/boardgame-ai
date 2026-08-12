from __future__ import annotations

import random

import pytest

from games.kingdomino_classic import ClassicGameConfig, ClassicGameState
from games.kingdomino_classic.baselines import (
    DenialAwareBot,
    FlexibilityBot,
    ImmediateScoreBot,
    RandomBot,
)


@pytest.mark.parametrize(
    "bot",
    [RandomBot(), ImmediateScoreBot(), FlexibilityBot(), DenialAwareBot()],
)
@pytest.mark.parametrize("players", [3, 4])
def test_baseline_completes_a_legal_game(bot, players: int) -> None:
    state = ClassicGameState.new(
        config=ClassicGameConfig(players=players), seed=players
    )
    rng = random.Random(100 + players)

    while not state.is_terminal:
        actions = state.legal_actions()
        action = bot.choose_action(state, actions, rng=rng)
        assert action in actions
        state = state.step(action)

    state.assert_invariants()


@pytest.mark.parametrize(
    "bot",
    [RandomBot(), ImmediateScoreBot(), FlexibilityBot(), DenialAwareBot()],
)
def test_baseline_choice_is_reproducible_with_a_seeded_rng(bot) -> None:
    state = ClassicGameState.new(
        config=ClassicGameConfig(players=3), seed=19, start_player=0
    )
    actions = state.legal_actions()

    first = bot.choose_action(state, actions, rng=random.Random(55))
    second = bot.choose_action(state, actions, rng=random.Random(55))

    assert first == second


def test_immediate_bot_maximizes_its_declared_action_score() -> None:
    state = ClassicGameState.new(
        config=ClassicGameConfig(players=4), seed=29, start_player=0
    )
    bot = ImmediateScoreBot()
    actions = state.legal_actions()

    chosen = bot.choose_action(state, actions, rng=random.Random(1))
    scores = {action: bot.evaluate_action(state, action) for action in actions}

    assert scores[chosen] == max(scores.values())


@pytest.mark.parametrize(
    "bot",
    [RandomBot(), ImmediateScoreBot(), FlexibilityBot(), DenialAwareBot()],
)
def test_forced_choice_is_returned_unchanged(bot) -> None:
    state = ClassicGameState.new(
        config=ClassicGameConfig(players=4), seed=31, start_player=0
    )
    for _ in range(3):
        state = state.step(state.legal_actions()[0])
    actions = state.legal_actions()

    assert len(actions) == 1
    assert bot.choose_action(state, actions, rng=random.Random(3)) == actions[0]
