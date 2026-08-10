"""Logic gates for the A1c cycle planner and initialization budget."""

from collections import Counter

import pytest

from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.game import GameState


def _zero_evaluator(my_board, opp_board, flat, legal_indices):
    import numpy as np

    batch = int(np.asarray(my_board).shape[0])
    return np.zeros(batch, dtype=np.float32), [
        np.zeros(len(indices), dtype=np.float32) for indices in legal_indices
    ]


def _deck8_boundary_state():
    state = GameState.new(seed=17)
    while not state.pending_claims:
        state = state.step(state.legal_actions()[0])
    state.deck = state.deck[:8]
    while state.actor_index < len(state.pending_claims) - 1:
        state = state.step(state.legal_actions()[0])
    return state


def test_balanced_cycles_expose_every_tile_once_per_cycle():
    import kingdomino_rust as kr

    deck = list(range(1, 29))
    cycles = kr.debug_a1c_panel_cycles(
        deck, max_cycles=4, sampling="balanced", seed=20260809
    )
    assert len(cycles) == 4
    for cycle in cycles:
        assert len(cycle) == 7
        assert Counter(tile for row in cycle for tile in row) == Counter(deck)
        assert all(row == sorted(row) and len(set(row)) == 4 for row in cycle)


def test_iid_cycles_match_width_without_claiming_tile_balance():
    import kingdomino_rust as kr

    deck = list(range(1, 29))
    cycles = kr.debug_a1c_panel_cycles(
        deck, max_cycles=4, sampling="iid", seed=20260809
    )
    assert [len(cycle) for cycle in cycles] == [7, 7, 7, 7]
    assert all(
        row == sorted(row) and len(set(row)) == 4 and set(row) <= set(deck)
        for cycle in cycles
        for row in cycle
    )
    assert Counter(tile for row in cycles[0] for tile in row) != Counter(deck)


def test_panel_plans_are_seed_stable_and_sampling_modes_are_distinct():
    import kingdomino_rust as kr

    deck = list(range(1, 13))
    balanced = kr.debug_a1c_panel_cycles(deck, 3, "balanced", 17)
    assert balanced == kr.debug_a1c_panel_cycles(deck, 3, "balanced", 17)
    assert balanced != kr.debug_a1c_panel_cycles(deck, 3, "balanced", 18)
    assert balanced != kr.debug_a1c_panel_cycles(deck, 3, "iid", 17)


def test_width_and_guard_charge_a_whole_proposed_batch():
    import kingdomino_rust as kr

    target, admitted = kr.debug_a1c_admission(
        visits=16,
        n_init=16,
        max_cycles=8,
        widening_c=0.25,
        initialization_nn_evals=0,
        total_nn_evals=33,
        additional_nn_evals=11,
        max_initialization_fraction=0.25,
    )
    assert target == 1
    assert admitted is True

    target, admitted = kr.debug_a1c_admission(
        visits=15,
        n_init=16,
        max_cycles=8,
        widening_c=0.25,
        initialization_nn_evals=0,
        total_nn_evals=32,
        additional_nn_evals=11,
        max_initialization_fraction=0.25,
    )
    assert target == 0
    assert admitted is False


@pytest.mark.parametrize("sampling", ["", "stratified", "IID"])
def test_invalid_panel_sampling_is_rejected(sampling):
    import kingdomino_rust as kr

    with pytest.raises(ValueError, match="balanced.*iid"):
        kr.debug_a1c_panel_cycles(list(range(1, 9)), 1, sampling, 0)


def test_invalid_admission_parameters_are_rejected():
    import kingdomino_rust as kr

    with pytest.raises(ValueError, match="widening_c"):
        kr.debug_a1c_admission(16, 16, 4, 0.0, 0, 33, 11)
    with pytest.raises(ValueError, match="max_initialization_fraction"):
        kr.debug_a1c_admission(16, 16, 4, 0.25, 0, 33, 11, 1.1)
    with pytest.raises(ValueError, match="cannot exceed"):
        kr.debug_a1c_admission(16, 16, 4, 0.25, 34, 33, 11)


@pytest.mark.parametrize("sampling", ["balanced", "iid"])
def test_a1c_search_initializes_complete_panels_without_fake_visits(sampling):
    import kingdomino_rust as kr
    from games.kingdomino.deck8_oracle_compare import a1c_node_diagnostics

    state = _rust_state_from_python(GameState.new(seed=17))
    children, _value, diagnostics = kr.advisor_one_reveal_search(
        state,
        _zero_evaluator,
        96,
        chance_exposure=4,
        chance_enum_max_rows=1,
        seed=20260809,
        leaf_batch=8,
        virtual_loss=1,
        chance_panel_mode="a1c",
        chance_panel_sampling=sampling,
        chance_init_visits=1,
        chance_widening_c=0.1,
        chance_init_max_fraction=1.0,
    )
    assert sum(row[1] for row in children) == 96
    assert diagnostics["a1c_initialized_chance_nodes"] > 0
    assert diagnostics["a1c_enabled"] == 1.0
    assert diagnostics["a1c_max_cycles"] == 4.0
    assert diagnostics["a1c_raw_planned_rows"] == 44.0
    assert 0 < diagnostics["a1c_unique_planned_rows"] <= 44.0
    assert diagnostics["a1c_initialized_cycles"] >= diagnostics[
        "a1c_initialized_chance_nodes"
    ]
    assert diagnostics["initialization_nn_evaluations"] > 0
    assert diagnostics["initialization_nn_evaluations"] < diagnostics["nn_evaluations"]
    assert diagnostics["a1c_wave_safe_admission"] == 1.0
    assert diagnostics["a1c_visit_prioritized_admission"] == 1.0
    assert diagnostics["a1c_admission_requested_paths"] >= diagnostics[
        "a1c_admission_unique_nodes"
    ]
    assert diagnostics["a1c_admission_committed_cycles"] == diagnostics[
        "a1c_initialized_cycles"
    ]
    assert 0 < diagnostics["a1c_admission_waves"] <= diagnostics[
        "a1c_admission_committed_cycles"
    ]
    # Bootstrap rows influence the panel mean but are not search visits.
    assert diagnostics["chance_node_visits"] == diagnostics["observation_visits"]
    node_rows = a1c_node_diagnostics(diagnostics)
    assert len(node_rows) == diagnostics["a1c_reached_chance_nodes"]
    assert sum(row["initialized_cycles"] for row in node_rows) == diagnostics[
        "a1c_initialized_cycles"
    ]
    assert sum(row["initialized_cycles"] == 0 for row in node_rows) == diagnostics[
        "a1c_uninitialized_chance_nodes"
    ]
    assert sum(
        row["visits"] for row in node_rows if row["initialized_cycles"] == 0
    ) == diagnostics["a1c_uninitialized_node_visits"]


def test_a1c_search_never_exceeds_initialization_fraction_guard():
    import kingdomino_rust as kr

    state = _rust_state_from_python(GameState.new(seed=17))
    _children, _value, diagnostics = kr.advisor_one_reveal_search(
        state,
        _zero_evaluator,
        128,
        chance_exposure=4,
        chance_enum_max_rows=1,
        seed=20260810,
        leaf_batch=8,
        chance_panel_mode="a1c",
        chance_panel_sampling="balanced",
        chance_init_visits=1,
        chance_widening_c=0.25,
        chance_init_max_fraction=0.25,
    )
    assert diagnostics["initialization_nn_fraction"] <= 0.25 + 1e-12
    assert diagnostics["initialization_blocked_cycles"] > 0


def test_a1c_reports_sampled_visits_that_predate_panel_admission():
    import kingdomino_rust as kr

    state = _rust_state_from_python(_deck8_boundary_state())
    _children, _value, diagnostics = kr.advisor_one_reveal_search(
        state,
        _zero_evaluator,
        256,
        chance_exposure=4,
        chance_enum_max_rows=1,
        seed=20260811,
        leaf_batch=8,
        chance_panel_mode="a1c",
        chance_panel_sampling="balanced",
        chance_init_visits=4,
        chance_widening_c=0.1,
        chance_init_max_fraction=1.0,
    )
    assert diagnostics["a1c_initialized_chance_nodes"] > 0
    assert diagnostics["a1c_preinit_visits"] > 0
    assert diagnostics["a1c_initialized_node_visits"] > diagnostics[
        "a1c_preinit_visits"
    ]
    assert 0.0 < diagnostics["a1c_preinit_visit_fraction"] < 1.0


def test_a1c_search_admits_panels_only_at_parallel_wave_boundaries():
    import kingdomino_rust as kr
    import numpy as np

    state = _rust_state_from_python(_deck8_boundary_state())
    evaluator_batch_sizes = []

    def recording_evaluator(my_board, opp_board, flat, legal_indices):
        batch = int(np.asarray(my_board).shape[0])
        evaluator_batch_sizes.append(batch)
        return np.zeros(batch, dtype=np.float32), [
            np.zeros(len(indices), dtype=np.float32) for indices in legal_indices
        ]

    children, _value, diagnostics = kr.advisor_one_reveal_search(
        state,
        recording_evaluator,
        64,
        chance_exposure=2,
        chance_enum_max_rows=1,
        seed=20260812,
        leaf_batch=8,
        virtual_loss=1,
        chance_panel_mode="a1c",
        chance_panel_sampling="balanced",
        chance_init_visits=1,
        chance_widening_c=0.1,
        chance_init_max_fraction=1.0,
    )
    assert sum(row[1] for row in children) == 64
    assert diagnostics["a1c_wave_safe_admission"] == 1.0
    assert diagnostics["a1c_admission_requested_paths"] >= diagnostics[
        "a1c_admission_unique_nodes"
    ]
    assert diagnostics["a1c_admission_committed_cycles"] > 0
    assert 0 < diagnostics["a1c_admission_waves"] <= 64 / 8
    # Root evaluation, then the entire first search wave, then any two-row
    # deck=8 cycle bootstrap. Admission during descent would put a size-2 call
    # before the size-8 wave evaluation.
    assert evaluator_batch_sizes[:2] == [1, 8]
    assert 2 in evaluator_batch_sizes[2:]
    # Initialization changes the estimator only after the whole wave has
    # completed, so all observations still correspond to backed-up chance visits.
    assert diagnostics["chance_node_visits"] == diagnostics["observation_visits"]


def test_hard_nn_budget_stops_incumbent_without_overshoot():
    import kingdomino_rust as kr

    state = _rust_state_from_python(GameState.new(seed=17))
    children, _value, diagnostics = kr.advisor_one_reveal_search(
        state,
        _zero_evaluator,
        128,
        chance_exposure=0,
        seed=20260813,
        leaf_batch=8,
        nn_eval_budget=33,
    )
    assert diagnostics["nn_evaluations"] == 33
    assert diagnostics["ordinary_nn_evaluations"] == 33
    assert diagnostics["initialization_nn_evaluations"] == 0
    assert diagnostics["nn_eval_budget_hit"] == 1.0
    assert diagnostics["nn_eval_budget_unused"] == 0.0
    assert diagnostics["simulations_completed"] == 32
    assert diagnostics["simulation_limit_hit"] == 0.0
    assert diagnostics["nn_evaluator_calls"] == 5
    assert diagnostics["nn_max_batch_size"] == 8
    assert sum(row[1] for row in children) == 32


def test_hard_nn_budget_charges_a1c_bootstraps_and_then_ordinary_work():
    import kingdomino_rust as kr

    state = _rust_state_from_python(_deck8_boundary_state())
    _children, _value, diagnostics = kr.advisor_one_reveal_search(
        state,
        _zero_evaluator,
        128,
        chance_exposure=4,
        chance_enum_max_rows=1,
        seed=20260814,
        leaf_batch=8,
        chance_panel_mode="a1c",
        chance_panel_sampling="balanced",
        chance_init_visits=1,
        chance_widening_c=0.1,
        chance_init_max_fraction=1.0,
        nn_eval_budget=33,
    )
    assert diagnostics["nn_evaluations"] == 33
    assert diagnostics["nn_eval_budget_hit"] == 1.0
    assert diagnostics["initialization_nn_evaluations"] > 0
    assert diagnostics["ordinary_nn_evaluations"] > 1
    assert diagnostics["ordinary_nn_evaluations"] + diagnostics[
        "initialization_nn_evaluations"
    ] == diagnostics["nn_evaluations"]
    assert diagnostics["initialization_nn_budget_blocked_cycles"] > 0
    assert diagnostics["initialization_nn_budget_blocked_rows"] > 0
    assert diagnostics["a1c_admission_committed_cycles"] == diagnostics[
        "a1c_initialized_cycles"
    ]


def test_hard_nn_budget_never_partially_initializes_a_cycle_that_does_not_fit():
    import kingdomino_rust as kr
    from games.kingdomino.deck8_oracle_compare import a1c_node_diagnostics

    state = _rust_state_from_python(_deck8_boundary_state())
    _children, _value, diagnostics = kr.advisor_one_reveal_search(
        state,
        _zero_evaluator,
        128,
        chance_exposure=4,
        chance_enum_max_rows=1,
        seed=20260815,
        leaf_batch=8,
        chance_panel_mode="a1c",
        chance_panel_sampling="balanced",
        chance_init_visits=1,
        chance_widening_c=0.1,
        chance_init_max_fraction=1.0,
        nn_eval_budget=10,
    )
    assert diagnostics["nn_evaluations"] == 10
    assert diagnostics["initialization_nn_evaluations"] == 0
    assert diagnostics["a1c_initialized_cycles"] == 0
    assert diagnostics["a1c_admission_committed_cycles"] == 0
    assert diagnostics["initialization_nn_budget_blocked_cycles"] > 0
    assert diagnostics["initialization_nn_budget_blocked_rows"] >= 11
    node_rows = a1c_node_diagnostics(diagnostics)
    assert node_rows
    assert all(row["initialized_cycles"] == 0 for row in node_rows)
    assert diagnostics["a1c_uninitialized_chance_nodes"] == len(node_rows)
    assert diagnostics["a1c_uninitialized_node_visits"] == sum(
        row["visits"] for row in node_rows
    )


def test_zero_nn_budget_preserves_the_unbudgeted_search_path():
    import kingdomino_rust as kr

    state = _rust_state_from_python(GameState.new(seed=17))
    common = dict(
        chance_exposure=0,
        seed=20260816,
        leaf_batch=8,
        virtual_loss=1,
    )
    unbudgeted = kr.advisor_one_reveal_search(
        state, _zero_evaluator, 64, **common
    )
    explicit_zero = kr.advisor_one_reveal_search(
        state, _zero_evaluator, 64, nn_eval_budget=0, **common
    )
    assert explicit_zero == unbudgeted


def test_simulation_ceiling_reports_unspent_nn_budget():
    import kingdomino_rust as kr

    state = _rust_state_from_python(GameState.new(seed=17))
    _children, _value, diagnostics = kr.advisor_one_reveal_search(
        state,
        _zero_evaluator,
        8,
        chance_exposure=0,
        seed=20260817,
        leaf_batch=8,
        nn_eval_budget=100,
    )
    assert diagnostics["nn_evaluations"] == 9
    assert diagnostics["nn_eval_budget_hit"] == 0.0
    assert diagnostics["nn_eval_budget_unused"] == 91
    assert diagnostics["simulations_completed"] == 8
    assert diagnostics["simulation_limit_hit"] == 1.0


def test_probe_arm_independently_confirms_rust_nn_accounting():
    from games.kingdomino.chance_correct_search_probe import _arm

    arm = _arm(
        _deck8_boundary_state(),
        _zero_evaluator,
        sims=128,
        exposure=4,
        enum_max_rows=1,
        seed=20260818,
        margin_gain=2.0,
        alpha=0.5,
        fpu=-0.2,
        cpuct=1.5,
        panel_mode="a1c",
        panel_sampling="balanced",
        init_visits=1,
        widening_c=0.1,
        init_max_fraction=1.0,
        leaf_batch=8,
        nn_eval_budget=33,
    )
    assert arm["nn_evaluations"] == 33
    assert arm["chance_diagnostics"]["nn_eval_budget_hit"] == 1.0
    assert arm["evaluator_calls"] == arm["chance_diagnostics"][
        "nn_evaluator_calls"
    ]
    assert arm["evaluator_max_batch_size"] == arm["chance_diagnostics"][
        "nn_max_batch_size"
    ]
