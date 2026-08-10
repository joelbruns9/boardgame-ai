from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from games.kingdomino.deck8_oracle_compare import build_parser as compare_parser
from games.kingdomino.deck8_oracle_suite_compare import (
    CONFIG_SCHEMA,
    CONTROL_NAMES,
    DEFAULT_CONFIG,
    EXPECTED_ARM_NAMES,
    POSITION_SCHEMA,
    aggregate_suite,
    clustered_bootstrap_interval,
    derive_seed,
    global_arm_order,
    validate_config,
    validate_position_artifact,
    validate_suite_inputs,
)


def _config() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _score(*, regret: float = 0.0, initialized: bool = False) -> dict:
    nodes = (
        [
            {
                "node_id": 7,
                "visits": 40,
                "preinit_visits": 8,
                "preinit_visit_fraction": 0.2,
                "initialized_cycles": 1,
            },
            {
                "node_id": 9,
                "visits": 12,
                "preinit_visits": 12,
                "preinit_visit_fraction": 1.0,
                "initialized_cycles": 0,
            },
        ]
        if initialized
        else []
    )
    return {
        "selected_action_idx": 1,
        "top1_exact": regret == 0.0,
        "selected_exact_rank": 1 if regret == 0.0 else 2,
        "exact_regret_actor": regret,
        "q_pairwise_correct": 1,
        "q_pairwise_pairs": 1,
        "q_pairwise_accuracy": 1.0,
        "nn_evaluations": 4801,
        "rust_nn_evaluations": 4801,
        "python_rust_nn_accounting_match": True,
        "simulations_completed": 6000,
        "initialization_nn_evaluations": 2 if initialized else 0,
        "initialization_nn_fraction": 2 / 4801 if initialized else 0.0,
        "nn_evaluator_calls": 601,
        "nn_max_batch_size": 8,
        "nn_mean_batch_size": 4801 / 601,
        "initialization_evaluator_calls": 1 if initialized else 0,
        "a1c_initialized_cycles": 1 if initialized else 0,
        "a1c_reached_chance_nodes": 2 if initialized else 0,
        "a1c_initialized_chance_nodes": 1 if initialized else 0,
        "a1c_uninitialized_chance_nodes": 1 if initialized else 0,
        "a1c_nodes": nodes,
        "nn_eval_budget_hit": True,
        "nn_eval_budget_unused": 0,
        "simulation_limit_hit": False,
        "elapsed_seconds": 0.5,
    }


def _provenance(repeats: int = 2) -> dict:
    search = _config()["search"]
    search = {**search, "repeat_count": repeats}
    return {
        "suite_config_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "positions_sha256": "c" * 64,
        "oracle_sha256": "d" * 64,
        "oracle_id": "e" * 64,
        "position_sha256": "f" * 64,
        "search_configuration": search,
        "seeds": list(range(10, 10 + repeats)),
        "arm_execution_orders": [EXPECTED_ARM_NAMES for _ in range(repeats)],
    }


def _artifact(path: Path, position: int, regrets: dict[str, list[float]]) -> dict:
    repeats = len(next(iter(regrets.values())))
    provenance = _provenance(repeats)
    records = []
    for repeat in range(repeats):
        arms = {}
        for name in EXPECTED_ARM_NAMES:
            arms[name] = _score(
                regret=regrets[name][repeat], initialized=name.startswith("a1c_")
            )
        records.append(
            {
                "seed_index": repeat,
                "seed": provenance["seeds"][repeat],
                "arm_execution_order": provenance["arm_execution_orders"][repeat],
                "arms": arms,
            }
        )
    payload = {
        "schema_version": POSITION_SCHEMA,
        "position_index": position,
        "legal_actions": 3,
        "provenance": provenance,
        "exact": {"top_gap_actor": 0.1},
        "records": records,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    payload["artifact_path"] = str(path)
    return payload


def _resolved() -> dict:
    config = _config()
    config["search"] = {**config["search"], "repeat_count": 2}
    config["bootstrap"] = {**config["bootstrap"], "resamples": 200}
    return {
        "suite_config": config,
        "suite_config_sha256": "a" * 64,
        "checkpoint": {"sha256": "b" * 64},
        "positions": {"sha256": "c" * 64},
        "git_commit": "1" * 40,
        "sources": {},
        "selection_limitation": "biased calibration fixture",
    }


def test_manifest_validation_accepts_frozen_eight_oracle_configuration():
    resolved = validate_suite_inputs(DEFAULT_CONFIG)
    assert resolved["schema_version"].endswith("resolved-v1")
    assert [row["position_index"] for row in resolved["oracles"]] == [
        5,
        11,
        17,
        23,
        29,
        35,
        41,
        47,
    ]
    assert len(resolved["cases"]) == 64


def test_duplicate_position_or_oracle_ids_are_rejected():
    duplicate_position = _config()
    duplicate_position["oracles"][1]["position_index"] = 5
    with pytest.raises(ValueError, match="duplicate position"):
        validate_config(duplicate_position)
    duplicate_oracle = _config()
    duplicate_oracle["oracles"][1]["expected_oracle_id"] = duplicate_oracle[
        "oracles"
    ][0]["expected_oracle_id"]
    with pytest.raises(ValueError, match="duplicate oracle"):
        validate_config(duplicate_oracle)


def test_missing_incomplete_and_hash_mismatched_oracles_are_rejected(
    tmp_path, monkeypatch
):
    root = DEFAULT_CONFIG.resolve().parents[3]
    monkeypatch.setattr(
        "games.kingdomino.deck8_oracle_suite_compare._repo_root", lambda _path: root
    )
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    missing = _config()
    missing["oracles"][0]["summary_path"] = "missing-summary.json"
    missing_path = config_dir / "missing.json"
    missing_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises((FileNotFoundError, OSError)):
        validate_suite_inputs(missing_path)

    drift = _config()
    drift["oracles"][0]["expected_sha256"] = "0" * 64
    drift_path = config_dir / "drift.json"
    drift_path.write_text(json.dumps(drift), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_suite_inputs(drift_path)

    original_read = __import__(
        "games.kingdomino.deck8_oracle_suite_compare", fromlist=["_read_json"]
    )._read_json
    incomplete = _config()
    incomplete_path = config_dir / "incomplete.json"
    incomplete_path.write_text(json.dumps(incomplete), encoding="utf-8")
    first_summary = (root / incomplete["oracles"][0]["summary_path"]).resolve()

    def read_with_incomplete(path):
        payload = original_read(Path(path))
        if Path(path).resolve() == first_summary:
            payload = {**payload, "status": "incomplete"}
        return payload

    monkeypatch.setattr(
        "games.kingdomino.deck8_oracle_suite_compare._read_json",
        read_with_incomplete,
    )
    with pytest.raises(ValueError, match="incomplete oracle"):
        validate_suite_inputs(incomplete_path)


def test_global_rotation_is_balanced_and_deterministic():
    orders = [
        global_arm_order(EXPECTED_ARM_NAMES, ordinal, repeat, 8)
        for ordinal in range(8)
        for repeat in range(8)
    ]
    assert orders == [
        global_arm_order(EXPECTED_ARM_NAMES, ordinal, repeat, 8)
        for ordinal in range(8)
        for repeat in range(8)
    ]
    counts = [sum(order[0] == name for order in orders) for name in EXPECTED_ARM_NAMES]
    assert max(counts) - min(counts) <= 1


def test_common_seeds_are_arm_shared_and_position_blocks_are_disjoint():
    blocks = [
        {derive_seed(2026084000, ordinal, repeat) for repeat in range(8)}
        for ordinal in range(8)
    ]
    assert all(len(block) == 8 for block in blocks)
    assert all(left.isdisjoint(right) for left, right in zip(blocks, blocks[1:]))
    # A case has one seed and one order containing every arm: no arm-specific seed.
    assert set(global_arm_order(EXPECTED_ARM_NAMES, 3, 2, 8)) == set(
        EXPECTED_ARM_NAMES
    )


def test_resume_accepts_matching_artifact(tmp_path):
    provenance = _provenance()
    path = tmp_path / "position.json"
    payload = _artifact(
        path, 5, {name: [0.0, 0.0] for name in EXPECTED_ARM_NAMES}
    )
    assert validate_position_artifact(path, provenance, position_index=5)[
        "records"
    ] == payload["records"]


@pytest.mark.parametrize("field", ["suite_config_sha256", "oracle_sha256", "search_configuration"])
def test_resume_rejects_schema_hash_or_search_drift(tmp_path, field):
    path = tmp_path / "position.json"
    _artifact(path, 5, {name: [0.0, 0.0] for name in EXPECTED_ARM_NAMES})
    expected = _provenance()
    if field == "search_configuration":
        expected[field] = {**expected[field], "cpuct": 9.0}
    else:
        expected[field] = "0" * 64
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_position_artifact(path, expected, position_index=5)


def test_partial_json_is_rejected_without_overwrite(tmp_path):
    path = tmp_path / "position.json"
    partial = '{"schema_version":'
    path.write_text(partial, encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        validate_position_artifact(path, _provenance(), position_index=5)
    assert path.read_text(encoding="utf-8") == partial


def test_aggregation_pairs_repeats_then_positions(tmp_path):
    base = {name: [0.0, 0.0] for name in EXPECTED_ARM_NAMES}
    first = copy.deepcopy(base)
    second = copy.deepcopy(base)
    first["a1c_x4_balanced"] = [1.0, -1.0]
    second["a1c_x4_balanced"] = [2.0, 2.0]
    artifacts = [
        _artifact(tmp_path / "p5.json", 5, first),
        _artifact(tmp_path / "p11.json", 11, second),
    ]
    summary = aggregate_suite(artifacts, _resolved())
    comparison = summary["paired_deltas"]["a1c_x4_balanced"]["x0"]
    assert [row["mean_regret_delta"] for row in comparison["position_deltas"]] == [
        0.0,
        2.0,
    ]
    assert comparison["mean_regret_delta"] == 1.0


def test_bootstrap_resamples_position_values_not_repeat_cells():
    interval = clustered_bootstrap_interval([0.0, 10.0], seed=7, resamples=1000)
    assert interval["positions"] == 2
    assert interval["unit"].startswith("position cluster")
    assert interval["lower_95"] == 0.0
    assert interval["upper_95"] == 10.0


def test_unmet_nn_budget_invalidates_resume(tmp_path):
    path = tmp_path / "position.json"
    _artifact(path, 5, {name: [0.0, 0.0] for name in EXPECTED_ARM_NAMES})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["arms"]["x0"]["nn_evaluations"] = 4800
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="budget/accounting"):
        validate_position_artifact(path, _provenance(), position_index=5)


def test_controls_require_empty_a1c_node_arrays(tmp_path):
    path = tmp_path / "position.json"
    _artifact(path, 5, {name: [0.0, 0.0] for name in EXPECTED_ARM_NAMES})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["arms"][CONTROL_NAMES[0]]["a1c_nodes"] = [
        {"node_id": 1}
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="control unexpectedly"):
        validate_position_artifact(path, _provenance(), position_index=5)


def test_late_and_never_initialized_nodes_survive_aggregation(tmp_path):
    artifacts = [
        _artifact(
            tmp_path / "p5.json",
            5,
            {name: [0.0, 0.0] for name in EXPECTED_ARM_NAMES},
        )
    ]
    summary = aggregate_suite(artifacts, _resolved())
    nodes = summary["arms"]["a1c_x4_balanced"][
        "late_and_never_initialized_nodes"
    ]
    assert {node["initialized_cycles"] for node in nodes} == {0, 1}
    assert any(node["preinit_visit_fraction"] == 1.0 for node in nodes)


def test_single_position_cli_defaults_remain_unchanged():
    parser = compare_parser()
    args = parser.parse_args(
        [
            "--oracle-summary",
            "oracle.json",
            "--output",
            "result.json",
            "--selection-reason",
            "compatibility test",
        ]
    )
    assert args.arm_order_rotation_offset == 0
    assert args.seed_count == 8
    assert args.include_a1c is False
    assert args.nn_eval_budget == 0
    assert args.sims == 4800
