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
    AZBatchEvaluator, DenialSearch, load_checkpoint_network, pick_floor_signature,
    public_state_key,
)
from games.kingdomino.denial_signal_sweep import file_sha256, load_frozen_positions
from games.kingdomino.promotion import sha256_file
from games.kingdomino.secondary_pick_seed_test import (
    ROOT_SEEDS, TREE_SEEDS, _competition_ranks, _config, _load_tree_rows,
    _population_sd, _stable_reference, distribution, root_q_by_pick,
    tie_guarded_flip,
)


SCHEMA_VERSION = 3

# Position-clustered paired bootstrap.  Fixed seed so a report is reproducible
# from its inputs; picks/seeds within a root are correlated, so positions are
# the resampling unit.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_805

# Safety tolerance for the paired rank-1 guard: the 95% upper bound on the
# median increase in |rank-1 fragility| (treatment minus control).  Rank-1
# fragility is the method-offset control and sat at ~0.016-0.023 in the
# overnight baseline, while the secondary-specific effect under test is ~0.136;
# 0.05 therefore allows ordinary estimator movement but rejects a treatment
# that buys secondary accuracy by degrading the pick actually played.
RANK1_ABS_FRAGILITY_TOLERANCE = 0.05

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
    floor = dict(
        pick_floor_frac=float(args.pick_floor_frac),
        pick_floor_min_depth=int(args.pick_floor_min_depth),
        pick_floor_max_depth=int(args.pick_floor_max_depth),
    )
    floor_signature = pick_floor_signature(**floor)
    existing = _read_jsonl(output)
    for row in existing:
        # pick_floor_signature is part of provenance: resuming a baseline
        # ladder with floor flags set (or vice versa) would silently produce a
        # file containing cells from two different allocators.
        if (int(row.get("schema_version", -1)) != SCHEMA_VERSION
                or row.get("arm") != args.arm
                or row.get("checkpoint_sha256") != checkpoint_sha
                or row.get("positions_sha256") != positions_sha
                or row.get("reference_checkpoint_sha256") != reference_checkpoint_sha
                or row.get("pick_floor_signature") != floor_signature):
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
        config=_config(seed=ROOT_SEEDS[0], sims=3200, **floor))
    assert search._pick_floor_signature() == floor_signature
    started = time.perf_counter()
    for sims in sims_values:
        search.config = _config(seed=ROOT_SEEDS[0], sims=sims, **floor)
        for seed in ROOT_SEEDS:
            for index, (state, _source) in enumerate(records):
                cell = (index, sims, seed)
                if cell in completed:
                    continue
                cell_started = time.perf_counter()
                root = search._root_search(
                    state, seed_override=seed,
                    cache_namespace=(
                        f"pilot-{args.arm}-{floor_signature}-s{sims}-seed{seed}"))
                picks = root_q_by_pick(search, state, root)
                _append_jsonl(output, {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "kingdomino_reply_pilot_root_ladder",
                    "arm": args.arm,
                    "pick_floor_signature": floor_signature,
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
        # Phase A arms share a checkpoint and differ only here.  Recorded so a
        # comparison of two ladders can prove which allocator produced each.
        "pick_floor": floor,
        "pick_floor_signature": floor_signature,
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


def _index_ladder(path: Path, positions_sha: str, q_field: str = "root_q") -> tuple[
    dict[tuple[int, int, int], dict[Optional[int], Optional[float]]], dict[str, Any]
]:
    """Index a ladder, reading one of the per-pick Q estimators.

    ``root_q`` is the Q of the most-visited placement in the pick group -- the
    shipped estimator, and the one every earlier result used.
    ``group_max_root_q`` is the max Q over scored placements in the group,
    which matches how the forced reference selects within a group.  Comparing
    reports across the two isolates estimator asymmetry from real fragility.
    """
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
        for pick in row["per_pick"]:
            if q_field not in pick:
                raise ValueError(
                    f"ladder {path} has no '{q_field}' field; it predates the "
                    f"placement-estimator diagnostics and must be regenerated")
        indexed[key] = {pick["pick_domino_id"]: pick.get(q_field)
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


def _cluster_by_position(
    ladder: dict[tuple[int, int, int], dict[Optional[int], Optional[float]]],
    references: dict[int, dict[Optional[int], float]],
    sims: int,
) -> dict[int, list[tuple[bool, float]]]:
    """Per-position (is_rank1, fragility) cells for one arm at one sim rung.

    Picks and seeds drawn from the same root are correlated, so the position is
    the resampling unit for every interval reported here.
    """
    clusters: dict[int, list[tuple[bool, float]]] = {}
    for index, values in references.items():
        ranks = _competition_ranks(values)
        rows: list[tuple[bool, float]] = []
        for seed in ROOT_SEEDS:
            cell = ladder.get((index, sims, seed), {})
            for pick, rank in ranks.items():
                root_q = cell.get(pick)
                if root_q is None:
                    continue
                rows.append((rank == 1, float(root_q) - values[pick]))
        clusters[index] = rows
    return clusters


def _bootstrap_positions(
    statistic,
    positions: Sequence[int],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Optional[float]]:
    """Position-clustered bootstrap of `statistic(resampled_positions)`.

    Resamples whole positions with replacement and pools their cells, so the
    interval reflects between-position variance rather than treating correlated
    picks/seeds as independent observations.
    """
    point = statistic(positions)
    if point is None or len(positions) < 2:
        return {"point": point, "ci_low": None, "ci_high": None, "resamples": 0}
    rng = np.random.default_rng(seed)
    draws = []
    order = np.asarray(positions)
    for _ in range(resamples):
        sample = order[rng.integers(0, len(order), size=len(order))]
        value = statistic(sample.tolist())
        if value is not None:
            draws.append(value)
    if not draws:
        return {"point": point, "ci_low": None, "ci_high": None, "resamples": 0}
    array = np.asarray(draws, dtype=np.float64)
    return {
        "point": float(point),
        "ci_low": float(np.percentile(array, 2.5)),
        "ci_high": float(np.percentile(array, 97.5)),
        "resamples": int(array.size),
    }


def _excess_statistic(clusters: dict[int, list[tuple[bool, float]]]):
    """median(secondary fragility) - median(rank-1 fragility) over a position set."""
    def statistic(positions: Sequence[int]) -> Optional[float]:
        rank1, secondary = [], []
        for index in positions:
            for is_rank1, fragility in clusters.get(index, ()):
                (rank1 if is_rank1 else secondary).append(fragility)
        if not rank1 or not secondary:
            return None
        return float(np.median(secondary) - np.median(rank1))
    return statistic


def _paired_excess_statistic(
    control_clusters: dict[int, list[tuple[bool, float]]],
    treatment_clusters: dict[int, list[tuple[bool, float]]],
):
    """Treatment minus control median excess, on ONE resampled position set.

    This is the primary endpoint.  It must be computed as a paired difference
    over shared positions: the arms are evaluated on the same roots, so
    comparing two independent marginal intervals would badly overstate the
    uncertainty of the difference.
    """
    control_stat = _excess_statistic(control_clusters)
    treatment_stat = _excess_statistic(treatment_clusters)

    def statistic(positions: Sequence[int]) -> Optional[float]:
        left, right = control_stat(positions), treatment_stat(positions)
        if left is None or right is None:
            return None
        return float(right - left)
    return statistic


def _paired_median_statistic(paired: dict[int, list[float]]):
    """Median of pooled paired differences over a position set."""
    def statistic(positions: Sequence[int]) -> Optional[float]:
        pooled = [value for index in positions for value in paired.get(index, ())]
        return float(np.median(pooled)) if pooled else None
    return statistic


def compare_arms(args: argparse.Namespace) -> dict[str, Any]:
    references, positions_sha, reference_checkpoint_sha = load_fixed_references(
        args.positions_path, args.reference_dir)
    q_field = getattr(args, "q_field", "root_q")
    control, control_manifest = _index_ladder(
        Path(args.control_ladder), positions_sha, q_field)
    treatment, treatment_manifest = _index_ladder(
        Path(args.treatment_ladder), positions_sha, q_field)
    if set(control) != set(treatment):
        raise ValueError("control and treatment ladder cells differ")
    # An arm must differ from its control in SOMETHING, or the comparison is a
    # no-op dressed up as a result.  Reply pilot: checkpoints differ, floors
    # match.  Phase A: checkpoints match, floors differ.  Neither differing
    # means a ladder path was passed twice or an arm was generated with the
    # wrong flags.
    if (control_manifest["checkpoint_sha256"] == treatment_manifest["checkpoint_sha256"]
            and (control_manifest.get("pick_floor_signature")
                 == treatment_manifest.get("pick_floor_signature"))):
        raise ValueError(
            "control and treatment are the same configuration (identical "
            "checkpoint and pick-floor); nothing to compare")
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

    # Paired rank-1 safety analysis.  Because the searched references are
    # FROZEN and shared by both arms, signed fragility differences equal root-Q
    # differences cell-for-cell — so a signed check is the old mean-root-Q guard
    # under another name, and a search-allocation arm fails it by construction
    # (flooring moves visits off the favourite, which legitimately moves its Q).
    # What must NOT happen is rank-1 evaluation getting WORSE, so the guard is
    # on |fragility|: rank-1 fragility is the method-offset control and should
    # stay near zero whatever the allocator does.  The signed median is retained
    # as a directional diagnostic.
    control_clusters = _cluster_by_position(control, references, 3200)
    treatment_clusters = _cluster_by_position(treatment, references, 3200)
    paired_rank1_abs: dict[int, list[float]] = {}
    paired_rank1_signed: dict[int, list[float]] = {}
    for index, values in references.items():
        ranks = _competition_ranks(values)
        abs_rows, signed_rows = [], []
        for seed in ROOT_SEEDS:
            cell = (index, 3200, seed)
            for pick, rank in ranks.items():
                if rank != 1:
                    continue
                left, right = control.get(cell, {}).get(pick), treatment.get(cell, {}).get(pick)
                if left is None or right is None:
                    continue
                frag_c = float(left) - values[pick]
                frag_t = float(right) - values[pick]
                abs_rows.append(abs(frag_t) - abs(frag_c))
                signed_rows.append(frag_t - frag_c)
        paired_rank1_abs[index] = abs_rows
        paired_rank1_signed[index] = signed_rows

    positions = sorted(references)
    rank1_abs_paired = _bootstrap_positions(
        _paired_median_statistic(paired_rank1_abs), positions,
        resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    rank1_signed_paired = _bootstrap_positions(
        _paired_median_statistic(paired_rank1_signed), positions,
        resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    control_excess_ci = _bootstrap_positions(
        _excess_statistic(control_clusters), positions,
        resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    treatment_excess_ci = _bootstrap_positions(
        _excess_statistic(treatment_clusters), positions,
        resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    paired_excess_ci = _bootstrap_positions(
        _paired_excess_statistic(control_clusters, treatment_clusters), positions,
        resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)

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
        # Replaces the two +/-0.02 absolute shift guards.  Passes when the
        # position-clustered upper bound on the median paired increase in
        # |rank-1 fragility| stays under tolerance: the treatment may move
        # rank-1 Q, but it may not make rank-1 evaluation materially worse.
        "rank1_abs_fragility_not_degraded": bool(
            rank1_abs_paired["ci_high"] is not None
            and rank1_abs_paired["ci_high"] <= RANK1_ABS_FRAGILITY_TOLERANCE),
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
    guard_names = ("rank1_abs_fragility_not_degraded",)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "kingdomino_reply_pilot_frozen_reference_comparison",
        # Which per-pick Q estimator every number below was computed from.
        "q_field": q_field,
        "positions_sha256": positions_sha,
        "reference_checkpoint_sha256": reference_checkpoint_sha,
        "control_checkpoint_sha256": control_manifest["checkpoint_sha256"],
        "treatment_checkpoint_sha256": treatment_manifest["checkpoint_sha256"],
        # Phase A arms differ by allocator, not checkpoint; the reply pilot was
        # the reverse.  Recording both makes the report self-describing.
        "control_pick_floor": control_manifest.get("pick_floor"),
        "treatment_pick_floor": treatment_manifest.get("pick_floor"),
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
        "paired_bootstrap": {
            "resampling_unit": "position",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "positions": len(positions),
            "rank1_abs_fragility_tolerance": RANK1_ABS_FRAGILITY_TOLERANCE,
            # The guard endpoint.
            "rank1_abs_fragility_median_paired_delta": rank1_abs_paired,
            # Directional diagnostic only; equals the root-Q shift by
            # construction because the references are frozen and shared.
            "rank1_signed_fragility_median_paired_delta": rank1_signed_paired,
            # PRIMARY endpoint with clustered uncertainty, per the findings
            # doc.  Negative means the treatment reduced secondary-specific
            # excess fragility.  Read this, not the two marginal intervals
            # below: the arms share positions, so the marginals overlap even
            # when the paired difference is well resolved.
            "paired_median_excess_delta": paired_excess_ci,
            # Marginal, for context only.
            "control_median_excess": control_excess_ci,
            "treatment_median_excess": treatment_excess_ci,
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
    # Phase A allocation arms.  Defaults are off, so an unflagged run reproduces
    # the pre-Phase-A pure-PUCT ladder exactly.
    parser.add_argument("--pick-floor-frac", type=float, default=0.0,
                        help="minimum per-pick-group visit share in the root "
                             "search; 0 disables floors (default)")
    parser.add_argument("--pick-floor-min-depth", type=int, default=0,
                        help="shallowest floored depth; 0 includes the root "
                             "(parent floor), 1 floors only reply nodes and below")
    parser.add_argument("--pick-floor-max-depth", type=int, default=0,
                        help="deepest floored depth; 0 with frac>0 is rejected")
    parser.add_argument("--control-ladder", default=str(DEFAULT_OUTPUT_DIR / "control_root_ladder.jsonl"))
    parser.add_argument("--treatment-ladder", default=str(DEFAULT_OUTPUT_DIR / "treatment_root_ladder.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "behavior_report.json"))
    parser.add_argument("--q-field", choices=("root_q", "group_max_root_q"),
                        default="root_q",
                        help="per-pick Q estimator to score: root_q is the "
                             "most-visited placement (shipped); "
                             "group_max_root_q is the max over scored "
                             "placements, matching how the forced reference "
                             "selects within a pick group")
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
