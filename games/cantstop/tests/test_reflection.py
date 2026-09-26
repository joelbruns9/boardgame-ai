"""Column reflection (c <-> 14 - c, dice d -> 7 - d): the symmetry itself,
its feature form, training augmentation and mirrored averaging.

Run: python -m pytest games/cantstop/tests/test_reflection.py -q
"""

import random
from itertools import combinations_with_replacement

import numpy as np
import pytest
import torch

from games.cantstop.encoder import (FEATURE_SIZE, REFLECTION, encode_batch,
                                    reflect_features)
from games.cantstop.engine import (ALL_RULESETS, COLUMN_HEIGHTS, GameState,
                                   Phase, apply_move, can_stop, legal_moves,
                                   random_dice, reflect_dice, reflect_move,
                                   reflect_state, roll, stop, stop_blocked)
from games.cantstop.model import CantStopNet, NetEvaluator
from games.cantstop.portable_rng import PortableRng
from games.cantstop.solver import ProgressHeuristic, TurnSolver
from games.cantstop.train import reflect_half

DICE = list(combinations_with_replacement(range(1, 7), 4))


def positions(count=120, phase=None):
    """Positions sampled uniformly from whole random games in every rule
    set (every phase, or only ``phase``)."""
    pool = []
    for i, rules in enumerate(ALL_RULESETS):
        rng = PortableRng(50 + i)
        state = GameState(rules)
        while not state.game_over:
            if state.phase == Phase.AWAIT_ROLL:
                pool.append(state.clone())
            moves = roll(state, random_dice(rng))
            if moves:
                pool.append(state.clone())                      # AWAIT_MOVE
                apply_move(state, moves[rng.randrange(len(moves))])
                if not state.game_over:
                    pool.append(state.clone())                  # AWAIT_DECISION
                if can_stop(state) and rng.next_float() < 0.4:
                    stop(state)
    if phase is not None:
        pool = [p for p in pool if p.phase == phase]
    order = list(range(len(pool)))
    PortableRng(7).shuffle(order)
    return [pool[i] for i in sorted(order[:count])]


# ---- the symmetry is exact ----

def test_heights_are_symmetric():
    assert all(COLUMN_HEIGHTS[c] == COLUMN_HEIGHTS[14 - c] for c in range(2, 13))


def test_mirror_is_an_involution():
    for s in positions(40):
        back = reflect_state(reflect_state(s))
        assert back.progress == s.progress and back.claimed_by == s.claimed_by
        assert back.runners == s.runners


def test_legal_moves_and_blocking_commute_with_the_mirror():
    for s in positions(60):
        m = reflect_state(s)
        if s.phase != Phase.AWAIT_MOVE:
            assert stop_blocked(s) == stop_blocked(m)
        for dice in DICE:
            want = sorted(reflect_move(mv) for mv in legal_moves(s, dice))
            assert legal_moves(m, reflect_dice(dice)) == want, (s, dice)


def test_solver_values_are_mirror_invariant_under_a_symmetric_evaluator():
    """The heuristic only uses per-column fractions, so it is symmetric;
    one exact turn of search on top of it must be too."""
    h = ProgressHeuristic()
    for s in positions(200, Phase.AWAIT_ROLL)[-8:]:        # late, small tables
        a = TurnSolver(s, h).value(s)
        b = TurnSolver(reflect_state(s), h).value(reflect_state(s))
        assert np.allclose(a, b, atol=1e-12)


# ---- the feature form ----

def test_feature_reflection_matches_the_board_mirror():
    boards = positions(150, Phase.AWAIT_ROLL)
    assert len(boards) == 150
    x = encode_batch(boards)
    y = encode_batch([reflect_state(b) for b in boards])
    assert np.array_equal(reflect_features(x), y)


def test_reflection_moves_only_the_column_blocks():
    moved = np.flatnonzero(REFLECTION != np.arange(FEATURE_SIZE))
    assert len(moved) == 4 * 2 * 10          # 4 slots x 2 blocks x 10 (7 fixed)
    assert np.array_equal(REFLECTION[REFLECTION], np.arange(FEATURE_SIZE))


# ---- training augmentation ----

def test_reflect_half_mirrors_about_half_the_rows_and_nothing_else():
    x = torch.from_numpy(encode_batch(positions(150, Phase.AWAIT_ROLL)))
    perm = torch.as_tensor(REFLECTION)
    out, flipped = reflect_half(x, random.Random(3), perm)
    rows_mirrored = sum(
        torch.equal(out[i], x[i, perm]) and not torch.equal(out[i], x[i])
        for i in range(len(x)))
    unchanged = sum(torch.equal(out[i], x[i]) for i in range(len(x)))
    assert unchanged + rows_mirrored == len(x)
    assert 0.25 * len(x) < flipped < 0.75 * len(x)


# ---- mirrored averaging ----

def test_mirror_average_is_exactly_symmetric():
    torch.manual_seed(0)
    ev = NetEvaluator(CantStopNet(), device="cpu", mirror_average=True)
    x = encode_batch(positions(150, Phase.AWAIT_ROLL))
    a = ev.relative_probs(x)
    b = ev.relative_probs(np.ascontiguousarray(reflect_features(x)))
    assert np.allclose(a, b, atol=1e-7)
    plain = NetEvaluator(ev.net, device="cpu")
    assert not np.allclose(plain.relative_probs(x),
                           plain.relative_probs(np.ascontiguousarray(
                               reflect_features(x))), atol=1e-4)
