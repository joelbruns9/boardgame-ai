"""Frozen-position de-noising sweep for the Kingdomino denial signal.

This is a diagnosis harness, not a training path.  It varies only root-search
simulations and chance samples, persists each expensive cell independently, and
builds the requested migration/stability report from those resumable artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from games.kingdomino.denial_search import (
    ACTOR_FRAME,
    CASCADE_VERSION,
    AZBatchEvaluator,
    DenialSearch,
    SearchConfig,
    generate_az_midgame_positions,
    load_checkpoint_network,
    public_state_key,
)
from games.kingdomino.game import GameState
from games.kingdomino.promotion import DEFAULT_CURRENT_BEST, sha256_file


DEFAULT_DIR = Path("runs/kingdomino/denial_search")
DEFAULT_POSITIONS = DEFAULT_DIR / "signal_positions.jsonl"
DEFAULT_CELLS = DEFAULT_DIR / "signal_cells"
DEFAULT_REPORT = DEFAULT_DIR / "signal_sweep.json"


@dataclass(frozen=True)
class CellSpec:
    name: str
    search_sims: int
    chance_k: int
    seed_offset: int = 0
    role: str = "grid"


CELL_SPECS = {
    spec.name: spec for spec in (
        CellSpec("reference_s32_k4_seed0", 32, 4, 0, "reference"),
        CellSpec("chance_s128_k4_seed0", 128, 4),
        CellSpec("center_s128_k16_seed0", 128, 16),
        CellSpec("chance_s128_k64_seed0", 128, 64),
        CellSpec("sims_s32_k16_seed0", 32, 16),
        CellSpec("sims_s400_k16_seed0", 400, 16),
        CellSpec("sims_s400_k32_seed0", 400, 32, 0, "user_requested_followup"),
        CellSpec("reference_s32_k4_seed1", 32, 4, 1, "stability_reference"),
        CellSpec("reference_s32_k4_seed2", 32, 4, 2, "stability_reference"),
        CellSpec("sims_s400_k16_seed1", 400, 16, 1, "stability_denoised"),
        CellSpec("sims_s400_k16_seed2", 400, 16, 2, "stability_denoised"),
        # Retained only so pre-existing partial artifacts remain readable.  These
        # are not part of the updated stability study and are permission-gated.
        CellSpec("denoised_s400_k64_seed0", 400, 64, 0, "legacy_permission_gated"),
        CellSpec("denoised_s400_k64_seed1", 400, 64, 1, "legacy_permission_gated"),
        CellSpec("denoised_s400_k64_seed2", 400, 64, 2, "legacy_permission_gated"),
    )
}

REFERENCE_NAME = "reference_s32_k4_seed0"
GRID_NAMES = (
    REFERENCE_NAME,
    "chance_s128_k4_seed0",
    "center_s128_k16_seed0",
    "sims_s32_k16_seed0",
    "sims_s400_k16_seed0",
)
STAGE_B_NAMES = ("chance_s128_k64_seed0",)
STABILITY_NAMES = (
    REFERENCE_NAME,
    "reference_s32_k4_seed1",
    "reference_s32_k4_seed2",
    "sims_s400_k16_seed0",
    "sims_s400_k16_seed1",
    "sims_s400_k16_seed2",
)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_frozen_positions(
    records: Iterable[tuple[GameState, dict[str, Any]]],
    path: str | Path,
) -> dict[str, Any]:
    from games.kingdomino.web_app import state_to_debug_json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    keys = []
    for index, (state, source) in enumerate(records):
        key = public_state_key(state)
        keys.append(key)
        lines.append(json.dumps({
            "position_index": index,
            "state_key": key,
            "source": source,
            "discards": [int(x) for x in state.discards],
            "state": state_to_debug_json(state),
        }, sort_keys=True, separators=(",", ":")))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(output), "sha256": file_sha256(output),
            "positions": len(lines), "state_keys": keys}


def load_frozen_positions(path: str | Path) -> list[tuple[GameState, dict[str, Any]]]:
    from games.kingdomino.web_app import state_from_debug_json

    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        state = state_from_debug_json(payload["state"])
        state.discards = [int(x) for x in payload.get("discards", [0, 0])]
        actual = public_state_key(state)
        if actual != payload["state_key"]:
            raise ValueError(f"frozen position {line_number} key mismatch: {actual} != {payload['state_key']}")
        records.append((state, dict(payload.get("source", {}))))
    return records


def freeze_position_set(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.positions_path)
    if output.exists() and not args.force:
        records = load_frozen_positions(output)
        return {"path": str(output), "sha256": file_sha256(output),
                "positions": len(records),
                "state_keys": [public_state_key(state) for state, _source in records]}
    net, checkpoint_cfg = load_checkpoint_network(args.checkpoint, args.device)
    evaluator = AZBatchEvaluator(
        net, device=args.device, batch_size=args.leaf_batch_size,
        margin_gain=float(checkpoint_cfg.get("margin_gain", 2.0)),
        alpha=float(checkpoint_cfg.get("alpha", 0.5)))
    search = DenialSearch(
        evaluator, checkpoint_path=args.checkpoint,
        config=SearchConfig(
            pick_plies=8, chance_k=4, seed=args.seed,
            placement_top_k=2, root_search_sims=args.trajectory_sims,
            policy_temperature=0.10))
    positions = generate_az_midgame_positions(
        search, count=args.positions, seed=args.seed,
        min_deck=args.min_deck, max_deck=args.max_deck)
    return write_frozen_positions(positions, output)


def _percentiles(values: Sequence[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"min": None, "median": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(array.min()), "median": float(np.percentile(array, 50)),
            "p90": float(np.percentile(array, 90)), "max": float(array.max())}


def _position_summary(index: int, label: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    corrected = next(row for row in label["per_pick"]
                     if row["pick_domino_id"] == label["corrected_best_pick"])
    delta = float(corrected["policy_target"] - corrected["raw_prior"])
    fragility = None if label.get("fragility") is None else float(label["fragility"])
    headline = next(row for row in label["per_pick"]
                    if row["pick_domino_id"] == label["headline_pick"])
    frame_residual = (None if headline["headline_edge"] is None or fragility is None else
                      float(fragility - (headline["headline_edge"]
                                         - headline["searched_value_actor"])))
    return {
        "position_index": int(index), "state_key": label["state_key"], "source": source,
        "headline_pick": label["headline_pick"],
        "corrected_best_pick": label["corrected_best_pick"],
        "fragility": fragility,
        "high_fragility": bool(fragility is not None and fragility >= 0.20),
        "negative_fragility": bool(fragility is not None and fragility < 0.0),
        "raw_prior": float(corrected["raw_prior"]),
        "policy_target": float(corrected["policy_target"]),
        "policy_minus_prior": delta,
        "starved_upweight": bool(corrected["raw_prior"] <= 0.10 and delta > 0.0),
        "mc_standard_error": float(corrected["mc_standard_error"]),
        "headline_edge_actor": headline["headline_edge"],
        "searched_value_actor": float(headline["searched_value_actor"]),
        "actor_frame_residual": frame_residual,
        "material_correction": bool(
            label["corrected_best_pick"] != label["headline_pick"]
            and label["correction_margin"] >= 0.03),
    }


def _aggregate_metrics(summaries: Sequence[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    fragilities = [row["fragility"] for row in summaries if row["fragility"] is not None]
    stderrs = [row["mc_standard_error"] for row in summaries]
    deltas = [row["policy_minus_prior"] for row in summaries]
    negative = [abs(row["fragility"]) for row in summaries if row["negative_fragility"]]
    frame_residuals = [abs(row["actor_frame_residual"]) for row in summaries
                       if row["actor_frame_residual"] is not None]
    return {
        "high_fragility_positions": sum(row["high_fragility"] for row in summaries),
        "starved_picks_upweighted": sum(row["starved_upweight"] for row in summaries),
        "high_fragility_starved_picks_upweighted": sum(
            row["high_fragility"] and row["starved_upweight"] for row in summaries),
        "material_corrections": sum(row["material_correction"] for row in summaries),
        "fragility": _percentiles(fragilities),
        "mean_stderr": statistics.fmean(stderrs) if stderrs else 0.0,
        "mean_corrected_best_policy_minus_prior": statistics.fmean(deltas) if deltas else 0.0,
        "negative_fragility_count": len(negative),
        "mean_abs_negative_fragility": statistics.fmean(negative) if negative else 0.0,
        "actor_frame_max_abs_residual": max(frame_residuals, default=0.0),
        "elapsed_seconds": elapsed,
        "positions_per_hour": len(summaries) * 3600.0 / max(elapsed, 1e-9),
        "projection_10000_positions_hours": 10000.0 * elapsed / max(1, len(summaries)) / 3600.0,
    }


def run_cell(args: argparse.Namespace, spec: CellSpec) -> dict[str, Any]:
    if spec.chance_k >= 64 and not getattr(args, "allow_k64", False):
        raise PermissionError(
            f"{spec.name} uses chance_k={spec.chance_k}; explicit --allow-k64 is required")
    records = load_frozen_positions(args.positions_path)
    offset = max(0, int(args.offset))
    records = records[offset:]
    if args.limit:
        records = records[:args.limit]
    output = (Path(args.cells_dir) / "shards" / f"{spec.name}_{args.shard_name}.json"
              if args.shard_name else Path(args.cells_dir) / f"{spec.name}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] | None = None
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("cell") != asdict(spec):
            raise ValueError(f"existing output has a different cell spec: {output}")
        if existing.get("provenance", {}).get("positions_sha256") != file_sha256(args.positions_path):
            raise ValueError(f"existing output has a different frozen set: {output}")
    target_indices = list(range(offset, offset + len(records)))
    summaries = list(existing.get("positions", [])) if existing else []
    completed = {int(row["position_index"]) for row in summaries}
    if not completed.issubset(target_indices):
        raise ValueError(f"existing output covers indices outside this shard: {output}")
    prior_elapsed = (float(existing.get("metrics", {}).get("elapsed_seconds", 0.0))
                     if existing else 0.0)
    if completed == set(target_indices):
        print(f"reusing complete {output}", flush=True)
        return existing

    net, checkpoint_cfg = load_checkpoint_network(args.checkpoint, args.device)
    evaluator = AZBatchEvaluator(
        net, device=args.device, batch_size=args.leaf_batch_size,
        margin_gain=float(checkpoint_cfg.get("margin_gain", 2.0)),
        alpha=float(checkpoint_cfg.get("alpha", 0.5)))
    cell_seed = int(args.seed) + 1_000_003 * int(spec.seed_offset)
    search = DenialSearch(
        evaluator, checkpoint_path=args.checkpoint,
        config=SearchConfig(
            pick_plies=8, chance_k=spec.chance_k, seed=cell_seed,
            placement_top_k=2, root_search_sims=spec.search_sims,
            policy_temperature=0.10))
    started = time.perf_counter()

    def checkpoint() -> dict[str, Any]:
        elapsed = prior_elapsed + time.perf_counter() - started
        ordered = sorted(summaries, key=lambda row: row["position_index"])
        result = {
            "schema_version": 1,
            "cell": asdict(spec),
            "seeds": {"base": args.seed, "cell": cell_seed,
                      "root_formula": "cell_seed + 104729*(2*position_index+1)",
                      "chance_crn_seed": cell_seed},
            "provenance": {
                "checkpoint_path": args.checkpoint,
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "positions_path": args.positions_path,
                "positions_sha256": file_sha256(args.positions_path),
                "positions": len(records),
                "actor_frame": ACTOR_FRAME,
                "official_cascade_version": CASCADE_VERSION,
                "policy_temperature": 0.10, "material_margin": 0.03,
                "starved_prior": 0.10, "placement_top_k": 2,
            },
            "metrics": _aggregate_metrics(ordered, elapsed),
            "positions": ordered,
            "complete": len(ordered) == len(records),
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(output)
        return result

    for index, (state, source) in enumerate(records, start=offset):
        if index in completed:
            continue
        root_seed = cell_seed + 104729 * (2 * index + 1)
        root = search._root_search(
            state, cache_namespace=spec.name, seed_override=root_seed)
        label = search.search_position(state, root_result=root)
        summaries.append(_position_summary(index, label, source))
        result = checkpoint()
        print(f"signal {spec.name} {index + 1}/{len(records)} "
              f"fragility={label.get('fragility')}", flush=True)
    result = checkpoint()
    print(f"wrote {output}", flush=True)
    return result


def merge_cell_shards(args: argparse.Namespace, spec: CellSpec) -> dict[str, Any]:
    paths = sorted((Path(args.cells_dir) / "shards").glob(f"{spec.name}_*.json"))
    if not paths:
        raise FileNotFoundError(f"no shards found for {spec.name}")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    positions = sorted((row for shard in shards for row in shard["positions"]),
                       key=lambda row: row["position_index"])
    expected = list(range(args.positions))
    actual = [row["position_index"] for row in positions]
    if actual != expected:
        raise ValueError(f"{spec.name} shards cover {actual}, expected {expected}")
    wall = max(float(shard["metrics"]["elapsed_seconds"]) for shard in shards)
    result = {
        "schema_version": 1, "cell": asdict(spec),
        "seeds": shards[0]["seeds"], "provenance": shards[0]["provenance"],
        "metrics": _aggregate_metrics(positions, wall),
        "sharding": {"shards": [str(path) for path in paths],
                     "parallel_wall_seconds": wall,
                     "sum_process_seconds": sum(float(s["metrics"]["elapsed_seconds"])
                                                for s in shards)},
        "positions": positions,
    }
    output = Path(args.cells_dir) / f"{spec.name}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"merged {len(paths)} shards -> {output}", flush=True)
    return result


def _load_cell(cells_dir: str | Path, name: str) -> dict[str, Any]:
    path = Path(cells_dir) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing sweep cell: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _stability(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_index = []
    for index in range(len(cells[0]["positions"])):
        rows = [cell["positions"][index] for cell in cells]
        fragilities = [float(row["fragility"]) for row in rows]
        memberships = [1.0 if row["high_fragility"] else 0.0 for row in rows]
        deltas = [float(row["policy_minus_prior"]) for row in rows]
        by_index.append({
            "position_index": index,
            "fragility_variance": statistics.pvariance(fragilities),
            "high_fragility_membership_variance": statistics.pvariance(memberships),
            "high_fragility_membership_rate": statistics.fmean(memberships),
            "policy_minus_prior_variance": statistics.pvariance(deltas),
            "fragilities": fragilities,
            "policy_minus_prior": deltas,
        })
    return {
        "seeds": [cell["cell"]["seed_offset"] for cell in cells],
        "mean_fragility_variance": statistics.fmean(
            row["fragility_variance"] for row in by_index),
        "mean_high_fragility_membership_variance": statistics.fmean(
            row["high_fragility_membership_variance"] for row in by_index),
        "mean_policy_minus_prior_variance": statistics.fmean(
            row["policy_minus_prior_variance"] for row in by_index),
        "high_fragility_set_sizes": [
            cell["metrics"]["high_fragility_positions"] for cell in cells],
        "per_position": by_index,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    cells = {name: _load_cell(args.cells_dir, name)
             for name in set(GRID_NAMES) | set(STABILITY_NAMES)}
    stage_b_name = STAGE_B_NAMES[0]
    stage_b_path = Path(args.cells_dir) / f"{stage_b_name}.json"
    if stage_b_path.exists():
        candidate = _load_cell(args.cells_dir, stage_b_name)
        if len(candidate.get("positions", ())) == len(cells[REFERENCE_NAME]["positions"]):
            cells[stage_b_name] = candidate
    reference = cells[REFERENCE_NAME]
    anchor = {
        "expected": {"high_fragility_starved_picks_upweighted": 0,
                     "starved_picks_upweighted": 8,
                     "high_fragility_positions": 13,
                     "fragility_median": 0.03666265920868941},
        "observed": {
            "high_fragility_starved_picks_upweighted": reference["metrics"]["high_fragility_starved_picks_upweighted"],
            "starved_picks_upweighted": reference["metrics"]["starved_picks_upweighted"],
            "high_fragility_positions": reference["metrics"]["high_fragility_positions"],
            "fragility_median": reference["metrics"]["fragility"]["median"],
        },
    }
    anchor["passed"] = (
        anchor["observed"]["high_fragility_starved_picks_upweighted"] == 0
        and anchor["observed"]["starved_picks_upweighted"] == 8
        and anchor["observed"]["high_fragility_positions"] == 13
        and abs(anchor["observed"]["fragility_median"]
                - anchor["expected"]["fragility_median"]) <= 1e-12)
    if not anchor["passed"]:
        raise RuntimeError(f"reference-cell anchor failed: {anchor}")

    baseline_indices = [row["position_index"] for row in reference["positions"]
                        if row["starved_upweight"]]
    migration = []
    migration_names = list(GRID_NAMES)
    if stage_b_name in cells:
        migration_names.append(stage_b_name)
    for index in baseline_indices:
        row = {"position_index": index, "cells": {}}
        for name in migration_names:
            position = cells[name]["positions"][index]
            row["cells"][name] = {
                "fragility": position["fragility"],
                "clears_0_2_gate": position["high_fragility"],
                "policy_minus_prior": position["policy_minus_prior"],
                "still_starved_upweight": position["starved_upweight"],
            }
        migration.append(row)

    reference_stability = _stability([
        cells["reference_s32_k4_seed0"], cells["reference_s32_k4_seed1"],
        cells["reference_s32_k4_seed2"]])
    denoised_stability = _stability([
        cells["sims_s400_k16_seed0"], cells["sims_s400_k16_seed1"],
        cells["sims_s400_k16_seed2"]])
    fragility_variance_ratio = (
        denoised_stability["mean_fragility_variance"]
        / max(reference_stability["mean_fragility_variance"], 1e-15))
    chance_names = ["chance_s128_k4_seed0", "center_s128_k16_seed0"]
    if stage_b_name in cells:
        chance_names.append(stage_b_name)
    chance_sharpening = [
        {"cell": name, "chance_k": cells[name]["cell"]["chance_k"],
         "mean_corrected_best_policy_minus_prior":
             cells[name]["metrics"]["mean_corrected_best_policy_minus_prior"]}
        for name in chance_names
    ]
    chance_delta_values = [
        row["mean_corrected_best_policy_minus_prior"] for row in chance_sharpening]
    negative_names = ("sims_s32_k16_seed0", "center_s128_k16_seed0",
                      "sims_s400_k16_seed0")
    negative_by_sims = {
        name: {"search_sims": cells[name]["cell"]["search_sims"],
               "count": cells[name]["metrics"]["negative_fragility_count"],
               "mean_abs": cells[name]["metrics"]["mean_abs_negative_fragility"]}
        for name in negative_names
    }
    negative_low = negative_by_sims[negative_names[0]]
    negative_high = negative_by_sims[negative_names[-1]]
    negative_count_shrank = negative_high["count"] < negative_low["count"]
    negative_magnitude_shrank = negative_high["mean_abs"] < negative_low["mean_abs"]
    if negative_count_shrank and negative_magnitude_shrank:
        negative_characterization = "shrinks_with_search_sims"
    elif not negative_count_shrank and not negative_magnitude_shrank:
        negative_characterization = "persists_or_grows_with_search_sims"
    else:
        negative_characterization = "mixed_count_and_magnitude"
    most_name = (stage_b_name if stage_b_name in cells
                 else "sims_s400_k16_seed0")
    most = cells[most_name]
    migrated = sum(most["positions"][index]["high_fragility"] for index in baseline_indices)
    primary_passed = (migrated >= 4
                      and most["metrics"]["high_fragility_starved_picks_upweighted"] > 0)
    if primary_passed:
        verdict = (
            "PRIMARY PASSED: de-noising migrated at least four baseline "
            "starved-upweight positions above the fragility gate; Stage B is "
            "confirmatory and no retraining starts automatically.")
        route = "mechanic_confirmed_hold_for_retrain_decision"
    elif stage_b_name not in cells:
        verdict = (
            "STAGE A DID NOT PASS PRIMARY: chance_k=64 is the permission-gated "
            "decisive test. Do not infer NULL and do not run Stage B without "
            "explicit approval.")
        route = "await_k64_permission"
    else:
        verdict = (
            "NULL: the permission-gated Stage-B cell did not migrate enough "
            "baseline starved-upweight positions above the fragility gate. Do "
            "not retrain; investigate fragility definition first, then "
            "robust-softmax up-weight math, then placement_top_k.")
        route = "block_retrain_investigate_measurement"
    report = {
        "schema_version": 1,
        "scope": "signal diagnosis only; no retraining, mixture, or value-head re-verification",
        "provenance": {
            "checkpoint_path": args.checkpoint,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "frozen_positions_path": args.positions_path,
            "frozen_positions_sha256": file_sha256(args.positions_path),
            "positions": len(reference["positions"]),
            "trajectory_sims": args.trajectory_sims,
            "seed": args.seed,
            "reserved_test_split_opened": False,
        },
        "reference_anchor": anchor,
        "cells": {name: {"cell": cells[name]["cell"],
                           "seeds": cells[name]["seeds"],
                           "metrics": cells[name]["metrics"]}
                  for name in cells},
        "baseline_starved_upweight_positions": baseline_indices,
        "migration_table": migration,
        "stability": {"reference": reference_stability,
                      "most_denoised": denoised_stability,
                      "fragility_variance_ratio_denoised_over_reference":
                          fragility_variance_ratio},
        "secondary_evidence": {
            "fragility_variance_fell": fragility_variance_ratio < 1.0,
            "chance_k_policy_sharpening": chance_sharpening,
            "policy_minus_prior_monotonic_nondecreasing": all(
                right >= left for left, right in
                zip(chance_delta_values, chance_delta_values[1:])),
        },
        "negative_fragility": {
            "actor_frames_verified": all(
                cell["metrics"]["actor_frame_max_abs_residual"] <= 1e-12
                for cell in cells.values()),
            "by_search_sims_at_k16": negative_by_sims,
            "count_shrank_32_to_400": negative_count_shrank,
            "mean_abs_shrank_32_to_400": negative_magnitude_shrank,
            "characterization": negative_characterization,
        },
        "pre_registered_result": {
            "most_denoised_completed_cell": most_name,
            "stage_b_completed": stage_b_name in cells,
            "baseline_positions_migrated_above_0_2": migrated,
            "required": 4,
            "high_fragility_starved_picks_upweighted": most["metrics"]["high_fragility_starved_picks_upweighted"],
            "primary_passed": primary_passed,
            "route": route,
            "verdict": verdict,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "cell", "merge", "grid", "stability", "all", "report"),
                        default="all")
    parser.add_argument("--cell", choices=sorted(CELL_SPECS), default=REFERENCE_NAME)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CURRENT_BEST))
    parser.add_argument("--positions-path", default=str(DEFAULT_POSITIONS))
    parser.add_argument("--cells-dir", default=str(DEFAULT_CELLS))
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    parser.add_argument("--positions", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shard-name", default="")
    parser.add_argument("--trajectory-sims", type=int, default=32)
    parser.add_argument("--leaf-batch-size", type=int, default=1024)
    parser.add_argument("--min-deck", type=int, default=8)
    parser.add_argument("--max-deck", type=int, default=28)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-k64", action="store_true",
        help="required acknowledgement before running any chance_k >= 64 cell")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.mode in ("freeze", "all"):
        frozen = freeze_position_set(args)
        print(f"frozen {frozen['positions']} positions -> {frozen['path']} sha={frozen['sha256']}",
              flush=True)
        if args.mode == "freeze":
            return 0
    if args.mode == "cell":
        run_cell(args, CELL_SPECS[args.cell]); return 0
    if args.mode == "merge":
        merge_cell_shards(args, CELL_SPECS[args.cell]); return 0
    names = []
    if args.mode in ("grid", "all"):
        names.extend(GRID_NAMES)
    if args.mode in ("stability", "all"):
        names.extend(name for name in STABILITY_NAMES if name not in names)
    for name in names:
        path = Path(args.cells_dir) / f"{name}.json"
        if path.exists() and not args.force:
            print(f"reuse completed cell {path}", flush=True)
            continue
        result = run_cell(args, CELL_SPECS[name])
        if name == REFERENCE_NAME:
            metrics = result["metrics"]
            anchor_ok = (metrics["high_fragility_starved_picks_upweighted"] == 0
                         and metrics["starved_picks_upweighted"] == 8
                         and metrics["high_fragility_positions"] == 13
                         and abs(metrics["fragility"]["median"]
                                 - 0.03666265920868941) <= 1e-12)
            if not anchor_ok:
                raise RuntimeError(f"reference anchor failed; refusing sweep: {metrics}")
    if args.mode in ("report", "all"):
        report = build_report(args)
        print(report["pre_registered_result"]["verdict"], flush=True)
        print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
