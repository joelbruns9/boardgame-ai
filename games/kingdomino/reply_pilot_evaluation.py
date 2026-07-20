"""Frozen-reference behavior evaluation for the opponent-reply pilot.

This reruns only the inexpensive root-search ladder for a control or treatment
checkpoint.  The eight-ply searched references remain the immutable baseline
teacher from the overnight study, preventing a moving-target evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from games.kingdomino.denial_search import (
    AZBatchEvaluator, DenialSearch, load_checkpoint_network, public_state_key,
)
from games.kingdomino.denial_signal_sweep import file_sha256, load_frozen_positions
from games.kingdomino.promotion import sha256_file
from games.kingdomino.secondary_pick_seed_test import (
    ROOT_SEEDS, TREE_SEEDS, _competition_ranks, _config, _load_tree_rows,
    _population_sd, _stable_reference, distribution, root_q_by_pick,
    tie_guarded_flip,
)


SCHEMA_VERSION = 1
DEFAULT_POSITIONS = Path("runs/kingdomino/denial_search/signal_positions.jsonl")
DEFAULT_REFERENCE_DIR = Path("runs/kingdomino/denial_search/secondary_seed")
DEFAULT_OUTPUT_DIR = Path("runs/kingdomino/reply_pilot/evaluation")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def load_fixed_references(
    positions_path: str | Path, reference_dir: str | Path,
) -> tuple[dict[int, dict[Optional[int], float]], str, str]:
    positions_sha = file_sha256(positions_path)
    first = _read_jsonl(Path(reference_dir) / f"tree_seed{TREE_SEEDS[0]}.jsonl")
    if not first:
        raise ValueError("baseline searched-reference artifacts are missing")
    reference_checkpoint_sha = str(first[0]["checkpoint_sha256"])
    records = load_frozen_positions(positions_path)
    tree_rows = _load_tree_rows(reference_dir, positions_sha, reference_checkpoint_sha)
    references, _stability = _stable_reference(tree_rows, len(records))
    return references, positions_sha, reference_checkpoint_sha


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    sims_values = tuple(int(value) for value in args.sims.split(",") if value.strip())
    if sims_values != (3200, 10000):
        raise ValueError("the pre-registered pilot ladder is exactly 3200,10000")
    records = load_frozen_positions(args.positions_path)
    references, positions_sha, reference_checkpoint_sha = load_fixed_references(
        args.positions_path, args.reference_dir)
    if len(references) != len(records):
        raise ValueError("fixed reference count does not match frozen positions")
    checkpoint_sha = sha256_file(args.checkpoint)
    output = Path(args.output_dir) / f"{args.arm}_root_ladder.jsonl"
    existing = _read_jsonl(output)
    for row in existing:
        if (int(row.get("schema_version", -1)) != SCHEMA_VERSION
                or row.get("arm") != args.arm
                or row.get("checkpoint_sha256") != checkpoint_sha
                or row.get("positions_sha256") != positions_sha
                or row.get("reference_checkpoint_sha256") != reference_checkpoint_sha):
            raise ValueError(f"existing ladder provenance mismatch: {output}")
    completed = {(int(row["position_index"]), int(row["sims"]), int(row["seed"]))
                 for row in existing}
    if len(completed) != len(existing):
        raise ValueError(f"duplicate cells in {output}")

    net, checkpoint_cfg = load_checkpoint_network(args.checkpoint, args.device)
    evaluator = AZBatchEvaluator(
        net, device=args.device, batch_size=args.leaf_batch_size,
        margin_gain=float(checkpoint_cfg.get("margin_gain", 2.0)),
        alpha=float(checkpoint_cfg.get("alpha", 0.5)))
    search = DenialSearch(
        evaluator, checkpoint_path=args.checkpoint,
        config=_config(seed=ROOT_SEEDS[0], sims=3200))
    started = time.perf_counter()
    for sims in sims_values:
        search.config = _config(seed=ROOT_SEEDS[0], sims=sims)
        for seed in ROOT_SEEDS:
            for index, (state, _source) in enumerate(records):
                cell = (index, sims, seed)
                if cell in completed:
                    continue
                cell_started = time.perf_counter()
                root = search._root_search(
                    state, seed_override=seed,
                    cache_namespace=f"pilot-{args.arm}-s{sims}-seed{seed}")
                picks = root_q_by_pick(search, state, root)
                _append_jsonl(output, {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "kingdomino_reply_pilot_root_ladder",
                    "arm": args.arm,
                    "position_index": index,
                    "state_key": public_state_key(state),
                    "sims": sims,
                    "seed": seed,
                    "checkpoint_sha256": checkpoint_sha,
                    "positions_sha256": positions_sha,
                    "reference_checkpoint_sha256": reference_checkpoint_sha,
                    "elapsed_seconds": time.perf_counter() - cell_started,
                    "per_pick": list(picks.values()),
                })
                completed.add(cell)
                print(f"{args.arm}: sims={sims} seed={seed} "
                      f"position={index + 1}/{len(records)}", flush=True)
    rows = _read_jsonl(output)
    expected = len(records) * len(sims_values) * len(ROOT_SEEDS)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "kingdomino_reply_pilot_root_ladder_manifest",
        "arm": args.arm,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "positions": len(records),
        "positions_sha256": positions_sha,
        "reference_checkpoint_sha256": reference_checkpoint_sha,
        "sims": list(sims_values),
        "seeds": list(ROOT_SEEDS),
        "cells": len(rows),
        "expected_cells": expected,
        "complete": len(rows) == expected,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "output": str(output),
        "output_sha256": file_sha256(output),
    }
    _atomic_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def _index_ladder(path: Path, positions_sha: str) -> tuple[
    dict[tuple[int, int, int], dict[Optional[int], Optional[float]]], dict[str, Any]
]:
    rows = _read_jsonl(path)
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise ValueError(f"ladder manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("positions_sha256") != positions_sha:
        raise ValueError(f"ladder is incomplete or uses different positions: {path}")
    if file_sha256(path) != manifest.get("output_sha256"):
        raise ValueError(f"ladder hash mismatch: {path}")
    indexed = {}
    for row in rows:
        key = (int(row["position_index"]), int(row["sims"]), int(row["seed"]))
        if key in indexed:
            raise ValueError(f"duplicate ladder cell: {key}")
        indexed[key] = {pick["pick_domino_id"]: pick.get("root_q")
                        for pick in row["per_pick"]}
    return indexed, manifest


def arm_metrics(
    ladder: dict[tuple[int, int, int], dict[Optional[int], Optional[float]]],
    references: dict[int, dict[Optional[int], float]],
) -> dict[str, Any]:
    rank1, secondary = [], []
    for index, values in references.items():
        ranks = _competition_ranks(values)
        for pick, rank in ranks.items():
            (rank1 if rank == 1 else secondary).append((index, pick))

    def cohort(keys, sims):
        fragility, root_q, seed_sds = [], [], []
        missing = 0
        for index, pick in keys:
            values = [ladder[(index, sims, seed)].get(pick) for seed in ROOT_SEEDS]
            finite = [float(value) for value in values if value is not None]
            missing += len(ROOT_SEEDS) - len(finite)
            root_q.extend(finite)
            fragility.extend(value - references[index][pick] for value in finite)
            seed_sds.append(_population_sd(finite))
        return {
            "fragility": distribution(fragility),
            "root_q": distribution(root_q),
            "root_q_seed_sd": distribution(seed_sds),
            "expected_cells": len(keys) * len(ROOT_SEEDS),
            "missing_cells": missing,
        }

    by_sims = {}
    stable_sets = {}
    tie_killed = {}
    for sims in (3200, 10000):
        r1, sec = cohort(rank1, sims), cohort(secondary, sims)
        by_sims[str(sims)] = {
            "rank1": r1,
            "secondary": sec,
            "secondary_minus_rank1_median_fragility": (
                float(sec["fragility"]["median"] - r1["fragility"]["median"])),
            "secondary_minus_rank1_p90_fragility": (
                float(sec["fragility"]["p90"] - r1["fragility"]["p90"])),
        }
        stable = set()
        killed = 0
        for index in references:
            events = [tie_guarded_flip(
                ladder[(index, sims, seed)], references[index], tie_tolerance=1e-6)
                for seed in ROOT_SEEDS]
            killed += sum(event["tie_guard_killed"] for event in events)
            if sum(event["flip"] for event in events) >= 4:
                stable.add(index)
        stable_sets[sims] = stable
        tie_killed[sims] = killed
        by_sims[str(sims)]["stable_flips_ge_4_of_5"] = len(stable)
        by_sims[str(sims)]["tie_guard_killed"] = killed
    return {
        "by_sims": by_sims,
        "persistent_stable_flips_3200_to_10000": len(stable_sets[3200] & stable_sets[10000]),
        "all_common_root_q_mean": float(np.mean([
            float(value) for cell in ladder.values() for value in cell.values()
            if value is not None
        ])),
    }


def compare_arms(args: argparse.Namespace) -> dict[str, Any]:
    references, positions_sha, reference_checkpoint_sha = load_fixed_references(
        args.positions_path, args.reference_dir)
    control, control_manifest = _index_ladder(Path(args.control_ladder), positions_sha)
    treatment, treatment_manifest = _index_ladder(Path(args.treatment_ladder), positions_sha)
    if set(control) != set(treatment):
        raise ValueError("control and treatment ladder cells differ")
    control_metrics = arm_metrics(control, references)
    treatment_metrics = arm_metrics(treatment, references)

    c3200 = control_metrics["by_sims"]["3200"]
    t3200 = treatment_metrics["by_sims"]["3200"]
    common_rank1 = []
    common_all = []
    rank1_keys = {(index, pick) for index, values in references.items()
                  for pick, rank in _competition_ranks(values).items() if rank == 1}
    for cell in sorted(control):
        index = cell[0]
        if cell[1] != 3200:
            continue
        for pick in set(control[cell]) & set(treatment[cell]):
            left, right = control[cell][pick], treatment[cell][pick]
            if left is None or right is None:
                continue
            delta = float(right) - float(left)
            common_all.append(delta)
            if (index, pick) in rank1_keys:
                common_rank1.append(delta)

    control_median_excess = c3200["secondary_minus_rank1_median_fragility"]
    control_p90_excess = c3200["secondary_minus_rank1_p90_fragility"]
    gates = {
        "median_excess_reduction_at_least_20pct": bool(
            control_median_excess > 0.0
            and t3200["secondary_minus_rank1_median_fragility"]
                <= 0.80 * control_median_excess),
        "p90_excess_reduction_at_least_10pct": bool(
            control_p90_excess > 0.0
            and t3200["secondary_minus_rank1_p90_fragility"]
                <= 0.90 * control_p90_excess),
        "stable_flips_3200_at_most_14": bool(
            t3200["stable_flips_ge_4_of_5"] <= 14),
        "persistent_stable_flips_at_most_8": bool(
            treatment_metrics["persistent_stable_flips_3200_to_10000"] <= 8),
        "rank1_median_fragility_shift_within_0_02": bool(abs(
            t3200["rank1"]["fragility"]["median"]
            - c3200["rank1"]["fragility"]["median"]) <= 0.02),
        "mean_rank1_root_q_shift_within_0_02": bool(
            common_rank1 and abs(float(np.mean(common_rank1))) <= 0.02),
        "missing_q_not_increased": bool(
            sum(treatment_metrics["by_sims"][str(s)][cohort]["missing_cells"]
                for s in (3200, 10000) for cohort in ("rank1", "secondary"))
            <= sum(control_metrics["by_sims"][str(s)][cohort]["missing_cells"]
                   for s in (3200, 10000) for cohort in ("rank1", "secondary"))),
        "tie_guard_dependence_not_increased": bool(
            sum(treatment_metrics["by_sims"][str(s)]["tie_guard_killed"]
                for s in (3200, 10000))
            <= sum(control_metrics["by_sims"][str(s)]["tie_guard_killed"]
                   for s in (3200, 10000))),
        "median_root_q_seed_sd_not_increased_by_0_005": bool(
            t3200["secondary"]["root_q_seed_sd"]["median"]
            <= c3200["secondary"]["root_q_seed_sd"]["median"] + 0.005),
    }
    guard_names = (
        "rank1_median_fragility_shift_within_0_02",
        "mean_rank1_root_q_shift_within_0_02",
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "kingdomino_reply_pilot_frozen_reference_comparison",
        "positions_sha256": positions_sha,
        "reference_checkpoint_sha256": reference_checkpoint_sha,
        "control_checkpoint_sha256": control_manifest["checkpoint_sha256"],
        "treatment_checkpoint_sha256": treatment_manifest["checkpoint_sha256"],
        "control": control_metrics,
        "treatment": treatment_metrics,
        "common_cell_diagnostics": {
            "mean_rank1_root_q_treatment_minus_control": (
                float(np.mean(common_rank1)) if common_rank1 else None),
            "mean_all_root_q_treatment_minus_control": (
                float(np.mean(common_all)) if common_all else None),
            "rank1_cells": len(common_rank1),
            "all_cells": len(common_all),
        },
        "gates": gates,
        "anti_deflation_pass": all(gates[name] for name in guard_names),
        "behavior_pass": all(gates.values()),
        "route": ("proceed_to_bga_and_strength" if all(gates.values())
                  else "stop_before_expensive_evaluation"),
    }
    _atomic_json(Path(args.output), result)
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "report"), required=True)
    parser.add_argument("--arm", choices=("control", "treatment"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--positions-path", default=str(DEFAULT_POSITIONS))
    parser.add_argument("--reference-dir", default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--leaf-batch-size", type=int, default=512)
    parser.add_argument("--sims", default="3200,10000")
    parser.add_argument("--control-ladder", default=str(DEFAULT_OUTPUT_DIR / "control_root_ladder.jsonl"))
    parser.add_argument("--treatment-ladder", default=str(DEFAULT_OUTPUT_DIR / "treatment_root_ladder.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "behavior_report.json"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.mode == "run":
        if not args.arm or not args.checkpoint:
            raise ValueError("run mode requires --arm and --checkpoint")
        result = run_arm(args)
    else:
        result = compare_arms(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
