import numpy as np

from games.kingdomino.encoder import CANVAS_SIZE, FLAT_LAYOUT, FLAT_SIZE, NUM_BOARD_CHANNELS
from games.kingdomino.self_play import (
    Example,
    ReplayBuffer,
    SelfPlayConfig,
    _active_config_for_iteration,
    _apply_buffer_capacity,
    _compiled_schedules,
    _choose_playout_profile,
    _parse_schedule,
    _prune_examples_policy_targets,
    _prune_policy_target,
)


def _example(policy_vals, *, bag_count=12):
    flat = np.zeros(FLAT_SIZE, dtype=np.float16)
    bag = FLAT_LAYOUT["bag"]
    flat[bag.start:bag.start + bag_count] = 1.0
    policy_vals = np.asarray(policy_vals, dtype=np.float32)
    return Example(
        my_board=np.zeros((NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE), dtype=np.float16),
        opp_board=np.zeros((NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE), dtype=np.float16),
        flat=flat,
        policy_idx=np.arange(len(policy_vals), dtype=np.int32),
        policy_val=policy_vals / policy_vals.sum(),
        legal_idx=np.arange(len(policy_vals), dtype=np.int32),
        z=0.0,
        own_score=0.0,
        opp_score=0.0,
        win_target=0.5,
    )


def test_piecewise_schedule_uses_zero_based_iteration_steps():
    cfg = SelfPlayConfig(
        lr=1e-3,
        alpha=0.8,
        n_simulations=100,
        lr_schedule="0:0.01,2:0.001",
        alpha_schedule="1:0.2",
        sims_schedule="0:32,3:64",
        fast_game_fraction_schedule="0:0.1,2:0.25",
    )
    schedules = _compiled_schedules(cfg)

    assert _parse_schedule("2:3,0:1", cast=int) == [(0, 1), (2, 3)]
    it1 = _active_config_for_iteration(cfg, schedules, 1)
    it2 = _active_config_for_iteration(cfg, schedules, 2)
    it3 = _active_config_for_iteration(cfg, schedules, 3)
    it4 = _active_config_for_iteration(cfg, schedules, 4)

    assert it1.lr == 0.01
    assert it1.alpha == 0.8
    assert it1.n_simulations == 32
    assert it2.alpha == 0.2
    assert it3.lr == 0.001
    assert it3.fast_game_fraction == 0.25
    assert it4.n_simulations == 64


def test_policy_target_pruning_removes_one_visit_noise_and_renormalizes():
    idx = np.array([7, 8, 9], dtype=np.int32)
    vals = np.array([0.80, 0.01, 0.19], dtype=np.float32)

    new_idx, new_vals, removed, removed_mass = _prune_policy_target(
        idx, vals, total_visits=100)

    assert removed == 1
    assert np.array_equal(new_idx, np.array([7, 9], dtype=np.int32))
    assert np.isclose(float(new_vals.sum()), 1.0)
    assert np.isclose(removed_mass, 0.01, atol=1e-6)


def test_policy_target_pruning_can_skip_exact_endgame_examples():
    exact = _example([0.80, 0.01, 0.19], bag_count=4)
    midgame = _example([0.80, 0.01, 0.19], bag_count=12)

    stats = _prune_examples_policy_targets(
        [[exact, midgame]], total_visits=100, skip_exact=True)

    assert stats["policy_pruned_actions"] == 1
    assert len(exact.policy_idx) == 3
    assert len(midgame.policy_idx) == 2
    assert np.isclose(float(midgame.policy_val.sum()), 1.0)


def test_buffer_capacity_schedule_truncates_oldest_examples():
    buffer = ReplayBuffer(capacity=5)
    try:
        buffer.add([_example([1.0], bag_count=12) for _ in range(5)])
        for i, ex in enumerate(buffer.data):
            ex.iteration = i

        _apply_buffer_capacity(buffer, 3)

        assert buffer.capacity == 3
        assert [ex.iteration for ex in buffer.data] == [2, 3, 4]
        assert buffer._pos == 0
    finally:
        buffer.close()


def test_playout_profile_disabled_uses_full_search_settings():
    cfg = SelfPlayConfig(
        n_simulations=1600,
        dirichlet_epsilon=0.25,
        temp_moves=20,
        playout_cap_randomization=False,
    )

    is_full, sims, noise_eps, temp_moves, record = _choose_playout_profile(
        cfg, np.random.default_rng(0))

    assert is_full is True
    assert sims == 1600
    assert noise_eps == 0.25
    assert temp_moves == 20
    assert record is True


def test_playout_profile_fast_defaults_are_greedy_noiseless_and_unrecorded():
    cfg = SelfPlayConfig(
        n_simulations=1600,
        playout_cap_randomization=True,
        full_search_fraction=0.0,
        fast_move_sims=100,
    )

    is_full, sims, noise_eps, temp_moves, record = _choose_playout_profile(
        cfg, np.random.default_rng(0))

    assert is_full is False
    assert sims == 100
    assert noise_eps == 0.0
    assert temp_moves == 0
    assert record is False


def test_playout_profile_full_fraction_one_uses_full_settings():
    cfg = SelfPlayConfig(
        n_simulations=1600,
        dirichlet_epsilon=0.25,
        temp_moves=20,
        playout_cap_randomization=True,
        full_search_fraction=1.0,
        fast_move_sims=100,
    )

    is_full, sims, noise_eps, temp_moves, record = _choose_playout_profile(
        cfg, np.random.default_rng(0))

    assert is_full is True
    assert sims == 1600
    assert noise_eps == 0.25
    assert temp_moves == 20
    assert record is True
