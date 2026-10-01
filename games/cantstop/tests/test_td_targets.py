"""TD value targets: the lambda-return over a game's turns.

Run: python -m pytest games/cantstop/tests/test_td_targets.py -q
"""

import numpy as np
import pytest
import torch

from games.cantstop.encoder import MAX_SEATS, seat_mask
from games.cantstop.engine import RuleSet
from games.cantstop.model import masked_cross_entropy, masked_soft_cross_entropy
from games.cantstop.portable_rng import PortableRng
from games.cantstop.self_play import (GameResult, play_game, stack_rows,
                                      stack_training, td_targets)
from games.cantstop.solver import ProgressHeuristic


@pytest.fixture(scope="module")
def games():
    return [play_game(r, ProgressHeuristic(), PortableRng(s))
            for s, r in enumerate([RuleSet.make(2), RuleSet.make(3, True),
                                   RuleSet.make(4, blocking=True)])]


def _rotate(v, active, n):
    return [v[(active + k) % n] for k in range(n)]


def test_one_value_per_turn(games):
    for g in games:
        assert len(g.turn_values) == g.turns == len(g) + 1
        # An opening-roll bust (None) needs every pairing on a closed
        # column, so it is late-game only; the hand-worked test covers it.


def test_lambda_one_is_the_old_winner_label(games):
    x1, y1 = stack_rows(games)
    x2, y2 = stack_training(games, 1.0)
    assert np.array_equal(x1, x2)
    assert np.array_equal(np.argmax(y2, axis=1), y1)
    assert np.array_equal(y2.max(axis=1), np.ones(len(y2), np.float32))


def test_lambda_zero_is_the_next_turns_solver_value(games):
    for g in games:
        t = td_targets(g, 0.0)
        n = g.rules.num_players
        passthrough = np.zeros(n)
        passthrough[g.winner] = 1.0
        # Walk backwards exactly as the definition does, independently.
        for i in range(len(g) - 1, -1, -1):
            v = g.turn_values[i + 1]
            if v is not None:
                passthrough = np.asarray(v)
            active = (g.winner - int(g.winner_slots[i])) % n
            want = np.float32(_rotate(passthrough, active, n))
            assert np.array_equal(t[i, :n], want), i


def test_a_hand_worked_game():
    """Three rows, 2 players, seat 1 wins; turn values chosen by hand, one
    opening bust. lam = 0.5."""
    r = GameResult(rules=RuleSet.make(2), winner=1,
                   features=np.zeros((3, 98), np.float32),
                   winner_slots=np.array([1, 0, 1]),   # actives 0, 1, 0
                   turns=4, solves=3, evaluator_rows=0,
                   turn_values=[[.5, .5], [.8, .2], None, [.3, .7]])
    t = td_targets(r, 0.5)
    g3 = np.array([0., 1.])                          # the outcome
    g2 = 0.5 * np.array([.3, .7]) + 0.5 * g3         # row 2 <- turn 3
    g1 = g2                                          # turn 2 opened in a bust
    g0 = 0.5 * np.array([.8, .2]) + 0.5 * g1         # row 0 <- turn 1
    # Row 0: seat 0 to move -> slots (0, 1). Row 1: seat 1 -> (1, 0).
    assert np.allclose(t[2, :2], g2[[0, 1]])
    assert np.allclose(t[1, :2], g1[[1, 0]])
    assert np.allclose(t[0, :2], g0[[0, 1]])
    assert np.allclose(t.sum(axis=1), 1.0)
    assert np.all(t[:, 2:] == 0)


def test_targets_are_distributions_over_live_seats(games):
    for g in games:
        t = td_targets(g, 0.7)
        live = seat_mask(g.features)
        assert np.all(t[~live] == 0)
        assert np.allclose(t.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(t >= 0)


def test_turn_values_must_line_up_with_rows():
    r = GameResult(rules=RuleSet.make(2), winner=0,
                   features=np.zeros((2, 98), np.float32),
                   winner_slots=np.array([0, 1]), turns=3, solves=0,
                   evaluator_rows=0, turn_values=[None, None])
    with pytest.raises(ValueError, match="turn values"):
        td_targets(r, 0.5)


def test_soft_loss_equals_hard_loss_on_one_hot_targets():
    torch.manual_seed(0)
    logits = torch.randn(64, MAX_SEATS)
    mask = torch.zeros(64, MAX_SEATS, dtype=torch.bool)
    mask[:, :3] = True
    slots = torch.randint(0, 3, (64,))
    onehot = torch.nn.functional.one_hot(slots, MAX_SEATS).float()
    assert torch.allclose(masked_soft_cross_entropy(logits, onehot, mask),
                          masked_cross_entropy(logits, slots, mask))
