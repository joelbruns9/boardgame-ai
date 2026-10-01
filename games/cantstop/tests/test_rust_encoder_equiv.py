"""The Rust leaf encoder is bit-identical to ``encoder.encode_batch``.

Compared as raw float32 bytes, not with a tolerance: the net must see the
same input whichever side encoded it. Skipped, not failed, when the
extension is not built.

Run: python -m pytest games/cantstop/tests/test_rust_encoder_equiv.py -q
"""

import numpy as np
import pytest
import torch

from games.cantstop.encoder import FEATURE_SIZE, encode_batch
from games.cantstop.engine import ALL_RULESETS, GameState, Phase, RuleSet
from games.cantstop.model import CantStopNet, NetEvaluator
from games.cantstop.rust_solver import (
    RustTurnSolver, leaf_features, sample_positions,
)
from games.cantstop.snapshot import from_snapshot, snapshot
from games.cantstop.solver import TurnSolver

rust = pytest.importorskip("cantstop_rust", reason="run maturin develop first")


def _leaf_boards(rs):
    return [from_snapshot(s) for s in rs.leaf_snapshots()]


def _positions(rules, count=4):
    return [p for p in sample_positions(rules, 21, 12)
            if rust.TurnSolver(snapshot(p)).num_positions < 8000][:count]


def test_feature_size_agrees():
    assert rust.FEATURE_SIZE == FEATURE_SIZE


@pytest.mark.parametrize("rules", ALL_RULESETS, ids=str)
def test_solver_leaf_features_are_bit_identical(rules):
    for state in _positions(rules):
        rs = rust.TurnSolver(snapshot(state))
        boards = _leaf_boards(rs)
        ours = leaf_features(rs)
        assert ours.shape == (len(boards), FEATURE_SIZE)
        assert ours.tobytes() == encode_batch(boards).tobytes()
        assert {b.active_player for b in boards} == {rs.leaf_active_player}


def test_encode_snapshots_on_boards_from_whole_games():
    """Boards from every stage of real games, not just one turn's leaves:
    claimed columns for several seats, late-game progress, all rule sets."""
    from games.cantstop.engine import (apply_move, can_stop, random_dice,
                                       roll, stop)
    from games.cantstop.portable_rng import PortableRng
    boards = []
    for i, rules in enumerate(ALL_RULESETS * 3):
        rng = PortableRng(i)
        state = GameState(rules)
        while not state.game_over:
            moves = roll(state, random_dice(rng))
            if moves:
                apply_move(state, moves[rng.randrange(len(moves))])
                if can_stop(state) and rng.next_float() < 0.35:
                    stop(state)
            if state.phase == Phase.AWAIT_ROLL:
                boards.append(state.clone())
        boards.append(state.clone())          # the final, winning board
    got = np.frombuffer(rust.encode_snapshots([snapshot(b) for b in boards]),
                        dtype="<f4").reshape(len(boards), FEATURE_SIZE)
    assert got.tobytes() == encode_batch(boards).tobytes()
    assert len(boards) > 1000


def test_encode_refuses_a_mid_turn_board():
    s = GameState(RuleSet.make(2))
    s.runners = {7: 1}
    s.phase = Phase.AWAIT_DECISION
    with pytest.raises(ValueError, match="end-of-turn"):
        rust.encode_snapshots([snapshot(s)])


@pytest.fixture(scope="module")
def net_evaluator():
    torch.manual_seed(0)
    return NetEvaluator(CantStopNet(), device="cpu")


def test_evaluate_features_is_the_same_evaluator(net_evaluator):
    state = _positions(RuleSet.make(3, blocking=True), 1)[0]
    boards = _leaf_boards(rust.TurnSolver(snapshot(state)))
    a = net_evaluator(boards)
    b = net_evaluator.evaluate_features(encode_batch(boards), boards[0])
    assert np.array_equal(a, b)


@pytest.mark.parametrize("rules", [RuleSet.make(2), RuleSet.make(4, True)],
                         ids=str)
def test_net_driven_solves_agree(rules, net_evaluator):
    """End to end with a real (untrained) net. Leaf ORDER differs between
    the two solvers, so torch may round a row differently in a different
    batch position; values are compared to 1e-6, not bit-exactly -- the
    bit-exact claims live in the mock-evaluator gate and the feature bytes
    above."""
    for state in _positions(rules, 2):
        ps = TurnSolver(state, net_evaluator)
        rs = RustTurnSolver(state, net_evaluator)
        assert rs.evaluator_calls == ps.evaluator_calls
        np.testing.assert_allclose(rs.value(state), ps.value(state),
                                   rtol=0, atol=1e-6)
        if state.phase == Phase.AWAIT_MOVE:
            assert rs.choose_move(state) == ps.choose_move(state)
