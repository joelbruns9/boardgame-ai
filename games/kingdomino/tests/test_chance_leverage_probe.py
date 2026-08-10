"""Gates for the deck=8 first-reveal causal leverage probe."""

import numpy as np
import pytest

from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.game import GameState, Phase


def _zero_evaluator(my_board, opp_board, flat, legal_indices):
    batch = int(np.asarray(my_board).shape[0])
    return np.zeros(batch, dtype=np.float32), [
        np.zeros(len(indices), dtype=np.float32) for indices in legal_indices
    ]


def _deck8_first_selection_state(seed=17):
    state = GameState.new(seed=seed)
    while not (
        state.phase == Phase.PLACE_AND_SELECT
        and state.actor_index == 0
        and len(state.deck) == 8
    ):
        assert state.phase != Phase.GAME_OVER
        state = state.step(state.legal_actions()[0])
    return _rust_state_from_python(state)


def test_control_is_the_existing_open_loop_path_with_added_diagnostics():
    import kingdomino_rust as kr

    state = _deck8_first_selection_state()
    common = dict(fpu=-0.2, cpuct=1.5, seed=20260820, leaf_batch=8)
    old_children, old_value, old_diagnostics = kr.advisor_one_reveal_search(
        state, _zero_evaluator, 800, chance_exposure=0, **common
    )
    children, value, diagnostics = kr.advisor_chance_leverage_probe(
        state, _zero_evaluator, 800, mode="control", **common
    )

    assert children == old_children
    assert value == old_value
    assert diagnostics["nn_evaluations"] == old_diagnostics["nn_evaluations"]
    assert diagnostics["simulations_completed"] == 800
    assert diagnostics["probe_enabled"] == 0.0
    assert diagnostics["probe_first_reveal_reached"] == 1.0
    assert diagnostics["probe_first_reveal_child_depth"] == 4.0
    assert 0 < diagnostics["probe_first_reveal_simulation"] <= 800


@pytest.mark.parametrize(
    ("mode", "pulse"),
    [("pulse_positive", 1.0), ("pulse_negative", -1.0)],
)
def test_free_pulses_add_no_nn_work_and_leave_many_post_admission_visits(mode, pulse):
    import kingdomino_rust as kr

    state = _deck8_first_selection_state()
    _children, _value, control = kr.advisor_chance_leverage_probe(
        state, _zero_evaluator, 800, mode="control", fpu=-0.2,
        seed=20260820, leaf_batch=8
    )
    children, _value, diagnostics = kr.advisor_chance_leverage_probe(
        state, _zero_evaluator, 800, mode=mode, fpu=-0.2,
        seed=20260820, leaf_batch=8
    )

    assert sum(row[1] for row in children) == 800
    assert diagnostics["nn_evaluations"] == control["nn_evaluations"]
    assert diagnostics["initialization_nn_evaluations"] == 0.0
    assert diagnostics["probe_pulse_value_actor"] == pulse
    assert diagnostics["probe_simulations_after_admission"] > 0
    if pulse > 0:
        assert diagnostics["probe_target_post_admission_visits"] > 0
    assert diagnostics["probe_full_panel_committed"] == 0.0


def test_real_panel_commits_all_70_rows_once_without_fake_visits():
    import kingdomino_rust as kr

    state = _deck8_first_selection_state()
    children, _value, diagnostics = kr.advisor_chance_leverage_probe(
        state, _zero_evaluator, 800, mode="full_panel", fpu=-0.2,
        seed=20260820, leaf_batch=8
    )

    assert sum(row[1] for row in children) == 800
    assert diagnostics["probe_first_reveal_child_depth"] == 4.0
    assert diagnostics["probe_full_panel_requested"] == 1.0
    assert diagnostics["probe_full_panel_committed"] == 1.0
    assert diagnostics["probe_full_panel_bootstrap_rows"] == 70.0
    assert diagnostics["initialization_nn_evaluations"] == 70.0
    assert diagnostics["nn_evaluations"] == (
        diagnostics["ordinary_nn_evaluations"] + 70.0
    )
    assert diagnostics["chance_node_visits"] == diagnostics["observation_visits"]
    assert diagnostics["probe_target_post_admission_visits"] > 0


def test_charged_panel_reserves_rows_only_after_it_is_admitted():
    import kingdomino_rust as kr

    state = _deck8_first_selection_state()
    children, _value, diagnostics = kr.advisor_chance_leverage_probe(
        state, _zero_evaluator, 800, mode="full_panel_charged", fpu=-0.2,
        seed=20260820, leaf_batch=8
    )

    assert diagnostics["probe_full_panel_committed"] == 1.0
    assert diagnostics["probe_full_panel_charged"] == 1.0
    assert diagnostics["probe_full_panel_bootstrap_rows"] == 70.0
    assert diagnostics["simulations_completed"] == 730.0
    assert sum(row[1] for row in children) == 730
    assert diagnostics["probe_realized_work_units"] == 800.0


def test_probe_rejects_non_target_roots_and_unknown_modes():
    import kingdomino_rust as kr

    opening = _rust_state_from_python(GameState.new(seed=17))
    with pytest.raises(ValueError, match="deck=8 actor_index=0"):
        kr.advisor_chance_leverage_probe(opening, _zero_evaluator, 8)

    state = _deck8_first_selection_state()
    with pytest.raises(ValueError, match="mode must be"):
        kr.advisor_chance_leverage_probe(
            state, _zero_evaluator, 8, mode="not-a-mode"
        )
