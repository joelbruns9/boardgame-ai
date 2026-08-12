"""G3 advisor plumbing, frozen arms, and fail-closed evidence checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from games.kingdomino.deck8_oracle_compare import progressive_node_diagnostics
from games.kingdomino.deck8_oracle_suite_compare import (
    G3_ARM_NAMES,
    G3_CONFIG_SCHEMA,
    G3_POSITION_SCHEMA,
    classify_g3_result,
    validate_config,
    validate_position_artifact,
)
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.game import GameState


G3_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "deck8_oracle_progressive_g3_v1.json"
)


def _zero_evaluator(my_board, opp_board, flat, legal_indices):
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
    return _rust_state_from_python(state)


def test_g3_config_freezes_five_arms_and_matched_budget():
    config = json.loads(G3_CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    assert config["schema_version"] == G3_CONFIG_SCHEMA
    assert config["search"]["nn_eval_budget"] == 4801
    assert [arm["name"] for arm in config["search"]["arms"]] == G3_ARM_NAMES
    assert config["search"]["arms"][2]["progressive_d_min"] == 4
    assert config["search"]["arms"][3]["progressive_d_min"] == 8
    assert config["search"]["arms"][4]["progressive_max_width"] == 16


def test_progressive_advisor_charges_rows_without_fake_visits_and_honors_cap():
    import kingdomino_rust as kr

    children, _value, diagnostics = kr.advisor_one_reveal_search(
        _deck8_boundary_state(),
        _zero_evaluator,
        9600,
        chance_exposure=1,
        chance_enum_max_rows=70,
        seed=20260811,
        leaf_batch=8,
        chance_backup="sampled",
        chance_traversal="progressive",
        chance_panel_mode="progressive",
        chance_init_visits=2,
        chance_init_max_fraction=0.25,
        chance_progressive_width_schedule="4,8,16,32,64,70",
        chance_progressive_d_min=4,
        chance_progressive_max_width=16,
        nn_eval_budget=4801,
    )
    assert diagnostics["nn_evaluations"] == 4801
    assert diagnostics["nn_eval_budget_hit"] == 1.0
    assert diagnostics["progressive_admission_count"] > 0
    assert diagnostics["progressive_widening_count"] > 0
    assert diagnostics["progressive_real_observation_subtrees"] >= 2
    nodes = progressive_node_diagnostics(diagnostics)
    assert nodes
    assert max(node["active_width"] for node in nodes) <= 16
    assert any(node["active_width"] == 16 for node in nodes)
    # Bootstrap rows affect chance Q but never policy/search visit counts.
    assert diagnostics["chance_node_visits"] == diagnostics["observation_visits"]
    assert sum(row[1] for row in children) == diagnostics["simulations_completed"]


def test_g3_classifier_defaults_to_d4_and_cap16_on_ties():
    arms = {name: {} for name in G3_ARM_NAMES}
    neutral = {
        "mean_regret_delta": 0.0,
        "mean_pairwise_accuracy_delta": 0.0,
        "position_win_tie_loss": {"wins": 0, "ties": 8, "losses": 0},
    }
    paired = {
        "pilot_sampled_split": {"x0": neutral},
        "progressive_full_d4": {
            "x0": neutral,
            "progressive_cap16_d4": neutral,
        },
        "progressive_full_d8": {
            "x0": neutral,
            "progressive_full_d4": neutral,
        },
        "progressive_cap16_d4": {"x0": neutral},
    }
    rules = json.loads(G3_CONFIG.read_text(encoding="utf-8"))["classification_rules"]
    result = classify_g3_result(arms, paired, [], rules)
    assert result["classification"] == "gate-pass"
    assert result["parameter_decisions"]["d_min"] == 4
    assert result["parameter_decisions"]["deck8_cap"] == 16


def test_g3_position_artifact_fails_closed_when_a_widening_is_missing(tmp_path):
    config = json.loads(G3_CONFIG.read_text(encoding="utf-8"))
    search = {**config["search"], "repeat_count": 1}
    provenance = {
        "search_configuration": search,
        "seeds": [17],
        "arm_execution_orders": [G3_ARM_NAMES],
    }
    arms = {}
    for name in G3_ARM_NAMES:
        progressive = name.startswith("progressive_")
        arms[name] = {
            "nn_evaluations": 4801,
            "nn_eval_budget_hit": True,
            "simulation_limit_hit": False,
            "python_rust_nn_accounting_match": True,
            "a1c_nodes": [],
            "progressive_admission_count": 1 if progressive else 0,
            "progressive_widening_count": 1 if progressive else 0,
            "progressive_real_observation_subtrees": 4 if progressive else 0,
            "progressive_nodes": (
                [{"node_id": 7, "active_width": 8}] if progressive else []
            ),
        }
    artifact = {
        "schema_version": G3_POSITION_SCHEMA,
        "position_index": 5,
        "provenance": provenance,
        "records": [
            {
                "seed": 17,
                "arm_execution_order": G3_ARM_NAMES,
                "arms": arms,
            }
        ],
    }
    path = tmp_path / "position.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    validate_position_artifact(path, provenance, position_index=5)
    artifact["records"][0]["arms"]["progressive_cap16_d4"][
        "progressive_widening_count"
    ] = 0
    path.write_text(json.dumps(artifact), encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="mechanism inactive"):
        validate_position_artifact(path, provenance, position_index=5)
