from dataclasses import replace

import numpy as np
import pytest
import cantstop_rust

from games.cantstop.decision_search import actions
from games.cantstop.engine import ALL_RULESETS, GameState, RuleSet, Phase, COLUMN_HEIGHTS, roll, apply_move, legal_moves
from games.cantstop.pool_rollout import PoolRolloutBackend
from games.cantstop.rollout_search import RolloutBackend, RolloutConfig, RolloutLimitExceeded, SharedDice
from games.cantstop.rust_solver import hashed_evaluator, flat_evaluator
from games.cantstop.rust_pool_equiv import HashedMock, MoverWinsMock
from games.cantstop.snapshot import snapshot


def race(n=4, phase=Phase.AWAIT_DECISION, blocking=True, rules=None):
    state = GameState(rules or RuleSet(n, 3, blocking))
    claims_per_seat = state.rules.columns_to_win - 1
    state.active_player = n - 1
    for c in range(2, 2 + claims_per_seat*n):
        state.claimed_by[c] = (c - 2) // claims_per_seat
    for c in range(2 + claims_per_seat*n, 13):
        for p in range(n):
            state.progress[p][c] = COLUMN_HEIGHTS[c] - 2
    if phase != Phase.AWAIT_ROLL:
        assert roll(state, (5, 5, 6, 6))
        if phase == Phase.AWAIT_DECISION:
            apply_move(state, legal_moves(state, state.dice)[0])
    return state


@pytest.mark.parametrize('seed', [0, 17, 2**64 - 1])
def test_native_shared_dice_matches_v1(seed):
    stream = SharedDice(seed)
    for turn in (0, 1, 7, 1000):
        for index in (0, 1, 25, 10000):
            assert tuple(cantstop_rust.audit_dice(seed, turn, index)) == stream.at(turn, index)


@pytest.mark.parametrize('n', [2, 3, 4])
@pytest.mark.parametrize('phase', [Phase.AWAIT_ROLL, Phase.AWAIT_MOVE, Phase.AWAIT_DECISION])
@pytest.mark.parametrize('dice_luck', [False, True])
@pytest.mark.parametrize('mock', ['board_hashed', 'feature_hashed', 'mover_wins'])
def test_pool_is_bit_identical_to_serial_on_mocks(n, phase, dice_luck, mock):
    state = race(n, phase, blocking=n != 3)
    before = snapshot(state)
    config = RolloutConfig(samples=5, horizon=None, seed=1729, sample_offset=7,
                           common_random_numbers=True, dice_luck=dice_luck)
    evaluator = {'board_hashed': hashed_evaluator, 'feature_hashed': HashedMock(), 'mover_wins': MoverWinsMock()}[mock]
    serial = RolloutBackend(evaluator, config, retain_samples=True)
    pool = PoolRolloutBackend(evaluator, config, retain_samples=True,
                             threads=2, in_flight=3, max_rows=1000)
    assert serial.evaluate(state) == pool.evaluate(state)
    assert serial.last_stats['trajectory_sha256'] == pool.last_stats['trajectory_sha256']
    for key in serial.last_samples:
        for field in ('raw', 'adjusted'):
            np.testing.assert_array_equal(serial.last_samples[key][field], pool.last_samples[key][field])
    for a, b in zip(serial.last_stats['actions'], pool.last_stats['actions'], strict=True):
        for key in ('samples', 'terminal_samples', 'completed_turns', 'dice_rolls'):
            assert a[key] == b[key]
    assert snapshot(state) == before


def test_chunking_threads_and_subset_do_not_change_samples():
    state = race(3, Phase.AWAIT_MOVE, False)
    config = RolloutConfig(samples=8, horizon=None, seed=2026100209,
                           common_random_numbers=True, dice_luck=True)
    whole = PoolRolloutBackend(hashed_evaluator, config, retain_samples=True, threads=1, in_flight=1)
    whole.evaluate(state)
    parts = []
    for offset in (0, 4):
        part = PoolRolloutBackend(hashed_evaluator, replace(config, samples=4, sample_offset=offset),
                                 retain_samples=True, threads=3, in_flight=7)
        part.evaluate(state)
        parts.append(part.last_samples)
    for key in whole.last_samples:
        for field in ('raw', 'adjusted'):
            np.testing.assert_array_equal(whole.last_samples[key][field],
                                          np.concatenate([p[key][field] for p in parts]))
    action = actions(state)[-1]
    subset = PoolRolloutBackend(hashed_evaluator, config, retain_samples=True, threads=2)
    subset.evaluate(state, candidates=(action,))
    for field in ('raw', 'adjusted'):
        np.testing.assert_array_equal(subset.last_samples[action.key][field], whole.last_samples[action.key][field])


@pytest.mark.parametrize('rules', ALL_RULESETS)
def test_all_ten_variants_match_serial(rules):
    state = race(rules.num_players, Phase.AWAIT_ROLL, rules=rules)
    config = RolloutConfig(samples=4, horizon=None, seed=301, common_random_numbers=True, dice_luck=True)
    evaluator = HashedMock()
    serial = RolloutBackend(evaluator, config, retain_samples=True)
    pool = PoolRolloutBackend(evaluator, config, retain_samples=True, threads=2, in_flight=2)
    assert serial.evaluate(state) == pool.evaluate(state)
    assert serial.last_stats['trajectory_sha256'] == pool.last_stats['trajectory_sha256']
    for field in ('raw', 'adjusted'):
        np.testing.assert_array_equal(serial.last_samples['roll'][field], pool.last_samples['roll'][field])


def test_one_stalled_native_game_fails_entire_batch():
    state = race(4, Phase.AWAIT_ROLL)
    specs = [(snapshot(state), 0, [0]*4)] * 2
    pool = cantstop_rust.SelfPlayPool(specs, 100, 1, 2)
    pool.configure_audit([(17, 1, 0, [0.]*4), (17, 1, 1, [0.]*4)], 1, True)
    pool.advance()
    assert pool.failure is not None
    with pytest.raises(RuntimeError, match='maximum dice rolls'):
        pool.audit_results()


@pytest.mark.parametrize('limits', [{'max_turns': 1}, {'max_rolls': 1}])
def test_failure_never_returns_partial_samples(limits):
    state = race(2, Phase.AWAIT_ROLL)
    cfg = RolloutConfig(samples=10, horizon=None, common_random_numbers=True, dice_luck=True, **limits)
    backend = PoolRolloutBackend(flat_evaluator, cfg, retain_samples=True)
    with pytest.raises(RolloutLimitExceeded):
        backend.evaluate(state)
    assert backend.last_samples is None
    assert backend.last_stats['status'] == 'incomplete'


def test_configuration_rejects_misalignment_and_nonbaseline_policy():
    state = race(4, Phase.AWAIT_ROLL)
    pool = cantstop_rust.SelfPlayPool([(snapshot(state), 0, [0]*4)], 100)
    with pytest.raises(ValueError):
        pool.configure_audit([], 100, True)
    with pytest.raises(ValueError):
        pool.configure_audit([(17, 0, 0, [0.]*4)], 100, True)
    pool.configure_audit([(17, 1, 0, [0.]*4)], 100, True)
    with pytest.raises(ValueError):
        pool.configure_audit([(17, 1, 0, [0.]*4)], 100, True)
    with pytest.raises(ValueError):
        PoolRolloutBackend(flat_evaluator, RolloutConfig(horizon=1, common_random_numbers=True))
    with pytest.raises(ValueError):
        PoolRolloutBackend(flat_evaluator, policy_factory=lambda *args: None)


def test_immediate_terminal_root_needs_no_future_pool():
    state = race(4, Phase.AWAIT_MOVE)
    apply_move(state, legal_moves(state, state.dice)[0])
    # Put an active runner at its top; banking wins this player's third column.
    column = next(iter(state.runners))
    state.runners[column] = COLUMN_HEIGHTS[column]
    calls = []
    def forbidden(states):
        calls.append(states)
        raise AssertionError('terminal root called NN')
    backend = PoolRolloutBackend(forbidden, RolloutConfig(samples=4, horizon=None,
                                common_random_numbers=True, dice_luck=True), retain_samples=True)
    backend.evaluate(state)
    assert not calls
    assert backend.last_stats['pool']['rounds'] == 0
    assert backend.last_stats['actions'][0]['terminal_samples'] == 4
