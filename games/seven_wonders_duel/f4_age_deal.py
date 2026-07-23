"""Select the smallest paired AgeDeal sample count that clears the F4-R gate.

The input rows are produced by paired root-position runs against the locked
32-sample diagnostic reference. Strength summaries are the corresponding
seat-swapped results. This module intentionally only certifies registered
counts and contract-aligned artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from .f4_quality import CONTRACT_PATH, _percentile


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict], strength: dict[str, dict], contract: dict) -> dict:
    registered = contract["production_semantics"]["age_deal"]
    bounds = contract["fast_search_quality_gate"]["eligibility"]
    strength_gate = contract["fast_search_quality_gate"]["playing_strength"]
    results = {}
    for sample_count in registered["candidate_counts"]:
        selected = [row for row in rows if int(row["sample_count"]) == sample_count]
        agreements = [float(row["action_agreement"]) for row in selected]
        errors = [float(row["root_value_abs_error"]) for row in selected]
        strength_row = strength.get(str(sample_count), {})
        checks = {
            "has_paired_positions": bool(selected),
            "action_agreement": bool(agreements)
            and mean(agreements) >= bounds["medium_gap_chosen_action_agreement_min"],
            "mean_root_value": bool(errors)
            and mean(errors) <= bounds["mean_absolute_root_value_error_max"],
            "p95_root_value": bool(errors)
            and _percentile(errors, 0.95) <= bounds["p95_absolute_root_value_error_max"],
            "playing_strength": bool(strength_row.get("sample_size_met"))
            and float(strength_row.get("elo_one_sided_lower", float("-inf")))
            >= strength_gate["non_inferiority_margin_elo"],
        }
        results[str(sample_count)] = {
            "positions": len(selected),
            "mean_action_agreement": mean(agreements) if agreements else None,
            "mean_root_value_abs_error": mean(errors) if errors else None,
            "p95_root_value_abs_error": _percentile(errors, 0.95) if errors else None,
            "strength": strength_row,
            "checks": checks,
            "eligible": all(checks.values()),
        }
    eligible = [
        count
        for count in registered["candidate_counts"]
        if results[str(count)]["eligible"]
    ]
    return {
        "schema": "f4-age-deal-calibration-1",
        "eligible": bool(eligible),
        "selected_sample_count": min(eligible) if eligible else None,
        "candidates": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-rows", type=Path, required=True)
    parser.add_argument("--strength-summaries", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in args.position_rows.read_text(encoding="utf-8").splitlines()
        if line
    ]
    strength = json.loads(args.strength_summaries.read_text(encoding="utf-8"))
    result = summarize(rows, strength, contract)
    result["manifest"] = {
        "contract_schema_version": contract["schema_version"],
        "contract_sha256": _sha256(CONTRACT_PATH),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "position_rows_sha256": _sha256(args.position_rows),
        "diagnostic_reference_count": contract["production_semantics"]["age_deal"][
            "diagnostic_reference_count"
        ],
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
