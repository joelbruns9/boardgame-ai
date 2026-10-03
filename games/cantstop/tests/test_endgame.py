"""Exact end-game solver: plays out to its own numbers, respects the board's
column symmetry, refuses positions over budget."""

import random

import numpy as np
import pytest

from games.cantstop.advisor_adapter import wins_on_stop
from games.cantstop.endgame import ExactEndgame, TooLarge
from games.cantstop.engine import (GameState, Phase, RuleSet, apply_move,
                                   can_stop, random_dice, reflect_state, roll,
                                   stop)


def board(rules, progress, claimed, active=0):
    s = GameState(rules)
    s.active_player = active
    for seat, row in enumerate(progress):
        s.progress[seat].update(row)
    for seat, cols in enumerate(claimed):
        for c in cols:
            s.claimed_by[c] = seat
    return s


# The board at turn 23 of BGA table 925127708: each player needs one column.
LATE = board(RuleSet(2, 5, False), [{2: 2, 3: 2, 11: 2}, {2: 2, 11: 1}],
             [[5, 7, 8, 12], [4, 6, 9, 10]])


def test_probabilities_and_symmetry():
    eg = ExactEndgame()
    for active in (0, 1):
        s = LATE.clone()
        s.active_player = active
        v = eg.board_value(s)
        assert v.sum() == pytest.approx(1.0, abs=1e-12) and (v >= 0).all()
        # columns c <-> 14 - c with dice d <-> 7 - d is an exact symmetry
        np.testing.assert_allclose(ExactEndgame().board_value(reflect_state(s)), v, atol=1e-9)
    assert eg.levels == 180


def test_the_turn_table_reproduces_the_board_value():
    eg = ExactEndgame()
    v = eg.board_value(LATE)
    np.testing.assert_allclose(eg.turn_solver(LATE).value(LATE), v, atol=1e-12)


def test_exact_play_wins_as_often_as_the_solver_says():
    """Both seats play the exact policy from LATE; the win rate must match
    the solved value (4 standard errors)."""
    eg = ExactEndgame()
    expected = eg.board_value(LATE)[0]
    rng, games, wins = random.Random(7), 3000, 0
    for _ in range(games):
        s = LATE.clone()
        while not s.game_over:
            solver = eg.turn_solver(s)
            while True:
                if not roll(s, random_dice(rng)):
                    break
                apply_move(s, solver.choose_move(s))
                if wins_on_stop(s) or (can_stop(s) and solver.should_stop(s)):
                    stop(s)
                    break
        wins += s.winner == 0
    se = np.sqrt(expected * (1 - expected) / games)
    assert abs(wins / games - expected) < 4 * se


def test_budget():
    early = board(RuleSet(2, 5, False), [{}, {}], [[], []])
    with pytest.raises(TooLarge):
        ExactEndgame(budget=500).board_value(early)
