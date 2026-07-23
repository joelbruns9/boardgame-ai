"""Validate forced/128 F4-R quality results and write the production lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .f4_quality import CONTRACT_PATH


SCHEMA = "f4-quality-lock-2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_manifest(name: str, manifest: dict, contract: dict, contract_hash: str) -> None:
    if manifest.get("contract_schema_version") != contract["schema_version"]:
        raise ValueError(f"{name} summary mixes an incompatible F4 contract version")
    if manifest.get("contract_sha256") != contract_hash:
        raise ValueError(f"{name} summary was produced under a different F4 contract")


def build_lock(
    position_summary: dict,
    strength_summary: dict | None,
    age_deal_summary: dict,
    frontier_summary: dict | None = None,
) -> dict:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract_hash = _sha256(CONTRACT_PATH)
    position_manifest = position_summary["manifest"]
    age_manifest = age_deal_summary["manifest"]
    _validate_manifest("position", position_manifest, contract, contract_hash)
    _validate_manifest("AgeDeal", age_manifest, contract, contract_hash)

    production = contract["production_semantics"]
    for field, expected in (
        ("sims", production["full_sims"]),
        ("top_k", production["top_k"]),
        ("force_expand_root_chance", production["force_expand_root_chance"]),
    ):
        if position_manifest.get(field) != expected:
            raise ValueError(f"position summary {field} differs from f4-contract-2")

    if not age_deal_summary.get("eligible"):
        raise ValueError("paired AgeDeal calibration gate did not pass")
    age_samples = int(age_deal_summary["selected_sample_count"])
    if age_samples not in production["age_deal"]["candidate_counts"]:
        raise ValueError("AgeDeal sample count is not a registered candidate")
    if int(age_manifest.get("diagnostic_reference_count", -1)) != int(
        production["age_deal"]["diagnostic_reference_count"]
    ):
        raise ValueError("AgeDeal calibration did not use the paired-32 reference")
    if age_manifest.get("checkpoint_sha256") != position_manifest.get("checkpoint_sha256"):
        raise ValueError("position and AgeDeal gates used different checkpoints")

    reaches_knee = bool(frontier_summary and frontier_summary.get("leaf1_reaches_gpu_knee"))
    if reaches_knee:
        leaf_batch = 1
        if frontier_summary.get("force_expand_root_chance") is not True:
            raise ValueError("leaf-1 frontier did not use forced root chance")
        if int(frontier_summary.get("sims", -1)) != production["full_sims"]:
            raise ValueError("leaf-1 frontier did not use 128 simulations")
    else:
        leaf_batch = position_summary.get("largest_position_eligible_leaf_batch")
        if leaf_batch is None:
            raise ValueError("no leaf batch passed the preregistered forced/128 position gate")
        if strength_summary is None:
            raise ValueError("candidate leaf batch requires playing-strength results")
        strength_manifest = strength_summary["manifest"]
        _validate_manifest("strength", strength_manifest, contract, contract_hash)
        if int(strength_manifest["leaf_batch"]) != int(leaf_batch):
            raise ValueError("strength leaf batch does not match the position-eligible batch")
        if not strength_summary.get("eligible"):
            raise ValueError("playing-strength non-inferiority gate did not pass")
        for field in (
            "checkpoint_sha256",
            "force_expand_root_chance",
            "sims",
            "top_k",
            "age_deal_sample_count",
        ):
            if strength_manifest.get(field) != position_manifest.get(field):
                raise ValueError(f"position and strength gates used different {field}")

    production_search = contract["laptop_comparative_benchmark"]["search"]
    return {
        "schema_version": SCHEMA,
        "contract_schema_version": contract["schema_version"],
        "contract_sha256": contract_hash,
        "pending_policy": "wu_incomplete_visits",
        "leaf_batch": int(leaf_batch),
        "pending_edge_cap": None,
        "force_expand_root_chance": True,
        "calibration_sims": int(position_manifest["sims"]),
        "top_k": int(position_manifest["top_k"]),
        "inference_precision": "float32",
        "production_search": production_search,
        "age_deal_sampler": {
            "method": "paired_common_outcome_sampling",
            "sample_count": age_samples,
            "diagnostic_reference_count": production["age_deal"][
                "diagnostic_reference_count"
            ],
        },
        "nn_work_metrics": {
            "forced": "forced_outcome_rows",
            "ordinary": "unique_nn_leaves",
            "total": "root_rows + forced_outcome_rows + unique_nn_leaves",
        },
        "checkpoint_sha256": position_manifest["checkpoint_sha256"],
        "corpus_sha256": position_manifest["positions_sha256"],
        "gate_results": {
            "position_summary": position_summary,
            "playing_strength_summary": strength_summary,
            "age_deal_summary": age_deal_summary,
            "concurrency_frontier_summary": frontier_summary,
        },
        "approved_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-summary", type=Path, required=True)
    parser.add_argument("--strength-summary", type=Path)
    parser.add_argument("--age-deal-summary", type=Path, required=True)
    parser.add_argument("--frontier-summary", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=CONTRACT_PATH.with_name("f4_quality_lock_v2.json"),
    )
    args = parser.parse_args()
    position = json.loads(args.position_summary.read_text(encoding="utf-8"))
    strength = (
        json.loads(args.strength_summary.read_text(encoding="utf-8"))
        if args.strength_summary
        else None
    )
    age_deal = json.loads(args.age_deal_summary.read_text(encoding="utf-8"))
    frontier = (
        json.loads(args.frontier_summary.read_text(encoding="utf-8"))
        if args.frontier_summary
        else None
    )
    lock = build_lock(position, strength, age_deal, frontier)
    args.output.write_text(
        json.dumps(lock, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(lock, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
