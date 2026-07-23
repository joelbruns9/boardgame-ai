from __future__ import annotations

import copy
import json

from .f4_quality import CONTRACT_PATH, REQUIRED_PHASE_STRATA, summarize


def _synthetic_gate_inputs():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["fast_search_quality_gate"]["confidence_method"]["minimum_resamples"] = 100
    positions = []
    for stratum in REQUIRED_PHASE_STRATA:
        for _ in range(40):
            index = len(positions)
            positions.append({"id": f"p{index}", "stratum": stratum})
    truths = {
        position["id"]: {"trap_gap": 0.25 if index < 50 else 0.0}
        for index, position in enumerate(positions)
    }
    rows = []
    for position_index, position in enumerate(positions):
        for seed in range(32):
            rows.append(
                {
                    "position_id": position["id"],
                    "leaf_batch": 4,
                    "search_seed": seed,
                    "sequential_action": 1,
                    "fast_action": 1,
                    "action_agreement": 1.0,
                    "sequential_gap": 0.3 if position_index % 2 else 0.1,
                    "sequential_tree_regret": 0.0,
                    "policy_js": 0.0,
                    "root_value_abs_error": 0.0,
                    "sequential_policy": [0.6, 0.4],
                    "sequential_root_value": 0.1,
                    "sequential_trap": False,
                    "fast_trap": False,
                    "trap_delta": 0.0,
                    "metrics": {"requested": 64, "collisions": 2},
                }
            )
    return contract, positions, truths, rows


def test_f4_quality_summary_accepts_synthetic_clean_candidate():
    contract, positions, truths, rows = _synthetic_gate_inputs()
    summary = summarize(rows, positions, truths, contract)
    assert summary["corpus"]["minimums_met"]
    assert summary["natural_variance_control"]["minimum_met"]
    assert summary["leaf_batches"]["4"]["position_gate_eligible"]
    assert summary["largest_position_eligible_leaf_batch"] == 4
    assert not summary["quality_lock_written"]


def test_f4_quality_summary_rejects_new_clean_fixture_blunder():
    contract, positions, truths, rows = _synthetic_gate_inputs()
    mutated = copy.deepcopy(rows)
    mutated[0]["fast_trap"] = True
    mutated[0]["trap_delta"] = 1.0
    summary = summarize(mutated, positions, truths, contract)
    candidate = summary["leaf_batches"]["4"]
    assert not candidate["checks"]["zero_new_clean_blunders"]
    assert not candidate["position_gate_eligible"]

