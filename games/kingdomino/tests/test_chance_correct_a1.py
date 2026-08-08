"""Focused gates for the opt-in A1 one-reveal search topology."""

import hashlib

import numpy as np

from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.game import GameState
from games.kingdomino.chance_correct_search_probe import (
    REFERENCE_SEED_XOR,
    _arm,
    _mode_summary,
    _parse_args,
    _position_consensus,
)
from games.kingdomino.denial_search import DenialSearch, SearchConfig


def _zero_evaluator(my_board, opp_board, flat, legal_indices):
    batch = int(np.asarray(my_board).shape[0])
    values = np.zeros(batch, dtype=np.float32)
    logits = [
        np.zeros(len(np.asarray(legal_indices[row])), dtype=np.float32)
        for row in range(batch)
    ]
    return values, logits


def _golden_evaluator(my_board, opp_board, flat, legal_indices):
    values = np.tanh(
        np.asarray(flat, dtype=np.float64).sum(axis=1) * 0.01
    ).astype(np.float32)
    logits = [
        np.asarray(legal_indices[row], dtype=np.float32) * 1e-4
        for row in range(len(legal_indices))
    ]
    return values, logits


def _midgame_state() -> GameState:
    state = GameState.new(seed=29)
    while len(state.deck) > 12:
        state = state.step(state.legal_actions()[0])
    return state


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


def test_a1_disabled_stable_opening_regression_vector():
    """Regression lock only; this is not proof of pre-A1 equivalence."""
    import kingdomino_rust as kr

    state = GameState.new(seed=17)
    children, root_value = kr.advisor_open_loop_search(
        _rust_state_from_python(state),
        _golden_evaluator,
        32,
        dirichlet_eps=0.0,
        fpu=0.0,
        cpuct=1.5,
        seed=20260808,
        leaf_batch=8,
        virtual_loss=1,
        alpha=0.5,
        chance_exposure=0,
        chance_enum_max_rows=70,
    )
    expected = [
        (3385, 8, -1.2280635237693787, 0.2499625024778055),
        (3386, 8, -1.2280635237693787, 0.24998749667597522),
        (3387, 8, -1.23147052526474, 0.25001250082431264),
        (3388, 8, -1.2302011847496033, 0.2500375000219066),
    ]
    assert [row[:2] for row in children] == [row[:2] for row in expected]
    np.testing.assert_allclose(
        [row[2:] for row in children],
        [row[2:] for row in expected],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        root_value,
        -0.13125014666355017,
        rtol=0.0,
        atol=1e-12,
    )


def test_a1_disabled_stable_midgame_regression_vector():
    """Lock the incumbent path through a deeper deck-12, 512-sim search."""
    import kingdomino_rust as kr

    children, root_value = kr.advisor_open_loop_search(
        _rust_state_from_python(_midgame_state()),
        _golden_evaluator,
        512,
        dirichlet_eps=0.0,
        fpu=0.0,
        cpuct=1.5,
        seed=20260809,
        leaf_batch=8,
        virtual_loss=1,
        alpha=0.5,
        chance_exposure=0,
        chance_enum_max_rows=70,
    )
    child_bytes = np.asarray(children, dtype="<f8").tobytes()
    assert len(children) == 20
    assert sum(row[1] for row in children) == 512
    assert hashlib.sha256(child_bytes).hexdigest() == (
        "318fceed1df48a73da969923a5acfa1107b131283ebeb7e10fea78ec6e3a9d9d"
    )
    np.testing.assert_allclose(
        root_value,
        -0.12609254337890805,
        rtol=0.0,
        atol=1e-12,
    )


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
    assert enabled["chance_panel_rows"] > 0.0
    assert enabled["chance_panel_exhaustive"] in (0.0, 1.0)
    assert "root_value_running_mean_player0" in enabled
    assert "root_value_current_children_player0" in enabled


def test_probe_records_exhaustive_panel_mode_and_rows():
    state = GameState.new(seed=17)
    # Initial deck is sampled at the default cap; force a deck=8 public bag so
    # the same probe seam exercises the exhaustive branch explicitly.
    state.deck = state.deck[:8]
    result = _arm(
        state,
        _zero_evaluator,
        sims=16,
        exposure=1,
        enum_max_rows=70,
        seed=20260808,
        margin_gain=2.0,
        alpha=0.5,
        fpu=-0.2,
        cpuct=1.5,
    )
    assert result["chance_panel_mode"] == "exhaustive"
    assert result["chance_panel_rows"] == 70
    assert "root_value_player0" not in result
    assert np.isfinite(result["root_value_running_mean_player0"])
    assert np.isfinite(result["root_value_current_children_player0"])


def test_denial_node_cache_key_includes_root_chance_configuration():
    search = object.__new__(DenialSearch)
    search.config = SearchConfig(root_chance_exposure=0, root_chance_enum_max_rows=70)
    state = GameState.new(seed=17)
    incumbent = search._node_key(state, depth=2, crossings=1, root_actor=0)
    search.config.root_chance_exposure = 4
    treatment = search._node_key(state, depth=2, crossings=1, root_actor=0)
    search.config.root_chance_enum_max_rows = 495
    wider_enum = search._node_key(state, depth=2, crossings=1, root_actor=0)
    assert incumbent != treatment != wider_enum


def test_multiseed_mode_summary_keeps_seed_repeats_clustered():
    summary = _mode_summary([17, 17, 9, 17, 9])
    assert summary == {
        "counts": {"9": 2, "17": 3},
        "mode": 17,
        "mode_count": 3,
        "mode_fraction": 0.6,
        "unanimous": False,
    }


def test_single_seed_does_not_serialize_trivial_consensus():
    consensus = _position_consensus([{"unused": True}], ["x0"])
    assert consensus["seed_count"] == 1
    assert consensus["selectors"] is None
    assert consensus["reference_mode_agreement"] is None


def test_probe_parses_targeted_subset_selection_reason():
    args = _parse_args([
        "--position-indices", "2,7",
        "--selection-reason", "previous reference disagreements",
    ])
    assert args.position_indices == "2,7"
    assert args.selection_reason == "previous reference disagreements"


def test_reference_seed_stream_is_disjoint_from_candidate_stream():
    candidate_seed = 20260808
    reference_seed = candidate_seed ^ REFERENCE_SEED_XOR
    assert reference_seed != candidate_seed
    assert (reference_seed ^ REFERENCE_SEED_XOR) == candidate_seed
