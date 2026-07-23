"""Pure contract gates for the F4.2/F4.6 completion tooling."""

from __future__ import annotations

import hashlib
import json

import pytest

from .f4_quality import CONTRACT_PATH
from .f4_cloud_finalize import finalize
from .f4_cloud_select import select
from .f4_age_deal import summarize as summarize_age_deal
from .f4_frontier import summarize as summarize_frontier
from .f4_quality_lock import build_lock
from .f4_strength import summarize
from .f4_throughput_bench import native_inference_assessment, speedup_summary


def test_f4_strength_pair_bootstrap_and_sample_gate():
    rows = [
        {"pair_index": pair, "fast_score": score}
        for pair in range(6)
        for score in (1.0, 0.5)
    ]
    result = summarize(rows, required_pairs=6, confidence=0.95, seed=4040)
    assert result["complete_pairs"] == 6
    assert result["games"] == 12
    assert result["sample_size_met"]
    assert result["elo_one_sided_lower"] > 0.0


def test_f4_speedup_uses_paired_repetitions():
    python = [{"games_per_second": value} for value in (1.0, 1.1, 0.9, 1.0, 1.0)]
    rust = [{"games_per_second": value * 25.0} for value in (1.0, 1.1, 0.9, 1.0, 1.0)]
    result = speedup_summary(python, rust, 4040)
    assert result["mean_speedup"] == pytest.approx(25.0)
    assert result["speedup_one_sided_95_lower"] == pytest.approx(25.0)


def test_f4_7_is_triggered_only_by_material_boundary_share():
    base = {
        "seconds": 10.0,
        "pyo3_tensor_seconds": 0.5,
        "h2d_seconds": 1.0,
        "gpu_forward_seconds": 4.0,
        "gather_d2h_seconds": 1.0,
        "pyo3_call_seconds": 7.0,
    }
    assert not native_inference_assessment([base])["f4_7_required"]
    material = {**base, "pyo3_tensor_seconds": 1.5, "pyo3_call_seconds": 8.5}
    assert native_inference_assessment([material])["f4_7_required"]


def test_f4_quality_lock_requires_both_green_gates():
    contract_hash = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    position = {
        "largest_position_eligible_leaf_batch": 4,
        "manifest": {
            "contract_schema_version": "f4-contract-2",
            "contract_sha256": contract_hash,
            "checkpoint_sha256": "checkpoint",
            "positions_sha256": "corpus",
            "force_expand_root_chance": True,
            "sims": 128,
            "top_k": 16,
            "age_deal_sample_count": 8,
        },
    }
    strength = {
        "eligible": True,
        "manifest": {
            "contract_schema_version": "f4-contract-2",
            "contract_sha256": contract_hash,
            "checkpoint_sha256": "checkpoint",
            "leaf_batch": 4,
            "force_expand_root_chance": True,
            "sims": 128,
            "top_k": 16,
            "age_deal_sample_count": 8,
        },
    }
    age_deal = {
        "eligible": True,
        "selected_sample_count": 8,
        "manifest": {
            "contract_schema_version": "f4-contract-2",
            "contract_sha256": contract_hash,
            "checkpoint_sha256": "checkpoint",
            "diagnostic_reference_count": 32,
        },
    }
    lock = build_lock(position, strength, age_deal)
    required = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))[
        "quality_lock_format"
    ]["required_fields"]
    assert all(field in lock for field in required)
    assert lock["leaf_batch"] == 4
    assert lock["pending_policy"] == "wu_incomplete_visits"
    assert lock["age_deal_sampler"]["sample_count"] == 8

    strength["eligible"] = False
    with pytest.raises(ValueError, match="did not pass"):
        build_lock(position, strength, age_deal)

    position["manifest"]["contract_schema_version"] = "f4-contract-1"
    with pytest.raises(ValueError, match="incompatible"):
        build_lock(position, strength, age_deal)


def test_f4_cloud_selection_and_confirmation_are_config_locked(tmp_path):
    manifest = {
        "slots": 64,
        "global_batch_cap": 512,
        "max_inflight_batches": 2,
        "scheduler_workers": 1,
        "pinned_memory": True,
        "torch_compile": "none",
        "diagnostic_sync": False,
        "quality_lock_sha256": "lock",
        "checkpoint_sha256": "checkpoint",
        "contract_schema_version": "f4-contract-2",
        "contract_sha256": "contract",
        "git_commit": "commit",
        "dirty_worktree": False,
        "device": "cuda",
        "inference_precision": "float32",
        "torch_version": "test",
        "cuda_version": "test",
        "cpu_model": "test-cpu",
        "gpu_model": "test-gpu",
    }
    for name, speed in (("slow", 10.0), ("fast", 12.0)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "mode": "rust",
                    "eligible": True,
                    "rust_games_per_second_mean": speed,
                    "manifest": manifest,
                }
            ),
            encoding="utf-8",
        )
    selected_path = tmp_path / "selected.json"
    selected = select(tmp_path, selected_path)
    assert selected["winner"]["games_per_second"] == 12.0

    confirmation = {
        "eligible": True,
        "sample_minimums_met": True,
        "rust_games_per_second_mean": 11.5,
        "manifest": manifest,
    }
    diagnostic = {
        "eligible": True,
        "manifest": {**manifest, "diagnostic_sync": True},
        "conditional_f4_7": {"f4_7_required": False},
    }
    confirmation_path = tmp_path / "confirm.json"
    diagnostic_path = tmp_path / "diagnostic.json"
    confirmation_path.write_text(json.dumps(confirmation), encoding="utf-8")
    diagnostic_path.write_text(json.dumps(diagnostic), encoding="utf-8")
    result = finalize(
        selected_path,
        confirmation_path,
        diagnostic_path,
        tmp_path / "production.json",
    )
    assert result["confirmed"]
    assert result["confirmed_games_per_second"] == 11.5


def test_f4_r_age_deal_selects_smallest_registered_eligible_count():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    rows = []
    strength = {}
    for count, agreement, error, elo in (
        (4, 0.85, 0.02, -10.0),
        (8, 0.95, 0.02, -10.0),
        (16, 0.99, 0.01, 5.0),
    ):
        rows.extend(
            {
                "sample_count": count,
                "action_agreement": agreement,
                "root_value_abs_error": error,
            }
            for _ in range(20)
        )
        strength[str(count)] = {
            "sample_size_met": True,
            "elo_one_sided_lower": elo,
        }
    result = summarize_age_deal(rows, strength, contract)
    assert result["eligible"]
    assert result["selected_sample_count"] == 8


def test_f4_r_frontier_requires_both_gpu_knee_metrics():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    rows = []
    for repetition in range(5):
        rows.append(
            {
                "leaf_batch": 1,
                "slots": 128,
                "scheduler_workers": 1,
                "force_expand_root_chance": True,
                "sims": 128,
                "games_per_second": 10.0,
                "policy_eligible_targets_per_second": 100.0,
                "gpu_busy_fraction": 0.91,
                "isolated_forward_rows_ratio": 0.96,
                "oom_count": 0,
                "repetition": repetition,
            }
        )
    result = summarize_frontier(rows, contract)
    assert result["leaf1_reaches_gpu_knee"]
