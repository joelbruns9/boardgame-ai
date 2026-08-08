"""Focused gates for the opt-in A1 one-reveal search topology."""

import numpy as np

from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.game import GameState
from games.kingdomino.chance_correct_search_probe import _mode_summary


def _zero_evaluator(my_board, opp_board, flat, legal_indices):
    batch = int(np.asarray(my_board).shape[0])
    values = np.zeros(batch, dtype=np.float32)
    logits = [
        np.zeros(len(np.asarray(legal_indices[row])), dtype=np.float32)
        for row in range(batch)
    ]
    return values, logits


def _search(state, **overrides):
    import kingdomino_rust as kr

    kwargs = dict(
        n_sims=96,
        dirichlet_eps=0.0,
        fpu=0.0,
        cpuct=1.5,
        seed=20260808,
        leaf_batch=8,
        virtual_loss=1,
        alpha=0.5,
    )
    kwargs.update(overrides)
    return kr.advisor_open_loop_search(
        _rust_state_from_python(state), _zero_evaluator, **kwargs
    )


def _diagnostic_search(state, **overrides):
    import kingdomino_rust as kr

    kwargs = dict(
        n_sims=96,
        fpu=0.0,
        cpuct=1.5,
        seed=20260808,
        leaf_batch=8,
        virtual_loss=1,
        alpha=0.5,
    )
    kwargs.update(overrides)
    return kr.advisor_one_reveal_search(
        _rust_state_from_python(state), _zero_evaluator, **kwargs
    )


def test_a1_disabled_is_exactly_the_incumbent_search():
    state = GameState.new(seed=17)
    implicit = _search(state)
    explicit = _search(
        state,
        chance_exposure=0,
        chance_enum_max_rows=70,
    )
    assert implicit == explicit


def test_a1_one_reveal_path_runs_at_equal_simulation_budget():
    state = GameState.new(seed=17)
    incumbent_children, incumbent_value = _search(state)
    hybrid_children, hybrid_value = _search(
        state,
        chance_exposure=1,
        chance_enum_max_rows=70,
    )

    assert [row[0] for row in hybrid_children] == [row[0] for row in incumbent_children]
    assert sum(row[1] for row in hybrid_children) == 96
    assert sum(row[1] for row in incumbent_children) == 96
    assert np.isfinite(hybrid_value)
    assert np.isfinite(incumbent_value)


def test_a1_diagnostics_measure_realized_support_coverage():
    state = GameState.new(seed=17)
    _children, _value, disabled = _diagnostic_search(state, chance_exposure=0)
    _children, _value, enabled = _diagnostic_search(
        state, chance_exposure=1, chance_enum_max_rows=70
    )

    assert disabled["chance_nodes"] == 0.0
    assert disabled["support_outcomes"] == 0.0
    assert enabled["chance_nodes"] > 0.0
    assert enabled["support_outcomes"] >= enabled["visited_outcomes"] > 0.0
    assert 0.0 < enabled["mean_visited_probability_mass"] <= 1.0
    assert 0.0 <= enabled["mean_unvisited_probability_mass"] < 1.0
    assert enabled["chance_node_visits"] == enabled["observation_visits"]
    assert 0.0 < enabled["visit_weighted_visited_probability_mass"] <= 1.0
    assert 0.0 <= enabled["visit_weighted_unvisited_probability_mass"] < 1.0


def test_multiseed_mode_summary_keeps_seed_repeats_clustered():
    summary = _mode_summary([17, 17, 9, 17, 9])
    assert summary == {
        "counts": {"9": 2, "17": 3},
        "mode": 17,
        "mode_count": 3,
        "mode_fraction": 0.6,
        "unanimous": False,
    }
