"""Validate selected/confirmation/diagnostic cloud rows into one production result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def finalize(
    selected_path: Path,
    confirmation_path: Path,
    diagnostic_path: Path,
    output: Path,
) -> dict:
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["winner"]
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if not confirmation.get("eligible") or not diagnostic.get("eligible"):
        raise SystemExit("cloud confirmation or diagnostic run is not eligible")
    if not confirmation.get("sample_minimums_met"):
        raise SystemExit("cloud confirmation is below the registered long-run sample minimums")
    if confirmation["manifest"].get("diagnostic_sync"):
        raise SystemExit("cloud throughput confirmation must not enable diagnostic synchronization")
    if not diagnostic["manifest"].get("diagnostic_sync"):
        raise SystemExit("F4.7 boundary diagnostic must enable synchronization")
    tuned = selected["manifest"]
    for summary, name in ((confirmation, "confirmation"), (diagnostic, "diagnostic")):
        manifest = summary["manifest"]
        for field in (
            "slots",
            "global_batch_cap",
            "max_inflight_batches",
            "scheduler_workers",
            "pinned_memory",
            "torch_compile",
            "quality_lock_sha256",
            "checkpoint_sha256",
        ):
            if manifest[field] != tuned[field]:
                raise SystemExit(f"{name} changed selected field {field}")
        for field in (
            "contract_schema_version",
            "contract_sha256",
            "git_commit",
            "dirty_worktree",
            "device",
            "inference_precision",
            "torch_version",
            "cuda_version",
            "cpu_model",
            "gpu_model",
        ):
            if manifest[field] != tuned[field]:
                raise SystemExit(f"{name} changed target environment field {field}")
    result = {
        "schema": "f4-cloud-production-1",
        "confirmed": True,
        "selected_sweep_games_per_second": selected["games_per_second"],
        "confirmed_games_per_second": confirmation["rust_games_per_second_mean"],
        "production_manifest": confirmation["manifest"],
        "conditional_f4_7": diagnostic["conditional_f4_7"],
        "selection": selected,
        "confirmation_summary": str(confirmation_path.resolve()),
        "diagnostic_summary": str(diagnostic_path.resolve()),
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.selected, args.confirmation, args.diagnostic, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
