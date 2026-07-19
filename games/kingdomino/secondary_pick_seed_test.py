"""High-sim, multi-seed diagnosis of Kingdomino secondary-pick fragility.

The expensive eight-ply searched values are computed for three chance seeds.
Independent open-loop root searches are then swept over a simulation ladder and
five seeds.  Artifacts are append-only JSONL so an overnight run can resume
without repeating completed positions.  This module only diagnoses existing
search output; it does not train or alter :mod:`denial_search`.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from games.kingdomino.action_codec import encode_action
from games.kingdomino.denial_search import (
    ACTOR_FRAME,
    AZBatchEvaluator,
    DenialSearch,
    SearchConfig,
    _pick_key,
    load_checkpoint_network,
    public_state_key,
)
from games.kingdomino.denial_signal_sweep import file_sha256, load_frozen_positions
from games.kingdomino.game import GameState
from games.kingdomino.promotion import DEFAULT_CURRENT_BEST, sha256_file


DEFAULT_RUN_DIR = Path("runs/kingdomino/denial_search/secondary_seed")
DEFAULT_POSITIONS = Path("runs/kingdomino/denial_search/signal_positions.jsonl")
DEFAULT_REPORT = Path("runs/kingdomino/denial_search/secondary_seed_test.json")

# These are deliberately spaced rather than adjacent PRNG seeds.  S0 is the
# frozen-set study date; S1..S4 use the same prime stride as the earlier sweep.
S0 = 20_260_717
ROOT_SEEDS = tuple(S0 + 1_000_003 * offset for offset in range(5))
TREE_SEEDS = ROOT_SEEDS[:3]
SIMS = (800, 3200, 10_000)
TIE_POSITION_INDICES = (0, 3, 7, 10, 14, 18, 21, 25, 28, 32, 35, 39, 42, 46, 49)

TREE_CONFIG = {
    "pick_plies": 8,
    "chance_k": 16,
    "placement_top_k": 2,
    "root_search_sims": 3200,
    "policy_temperature": 0.10,
    "tie_tolerance": 1e-6,
    "uncertainty_z": 1.0,
    "starved_prior": 0.10,
}


def distribution(values: Iterable[float]) -> dict[str, Optional[float] | int]:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))],
                        dtype=np.float64)
    if not finite.size:
        return {"count": 0, "min": None, "mean": None, "median": None,
                "p90": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "median": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(finite.max()),
    }


def root_q_by_pick(
    search: DenialSearch,
    state: GameState,
    root_result: Any,
) -> dict[Optional[int], dict[str, Any]]:
    """Reproduce ``_root_candidates`` representative selection at pick level.

    This is intentionally independent of ``DenialSearch._root_candidates`` so
    Phase 0 can fail if the factoring logic drifts from the validated search.
    """
    policy = search.evaluator.policy(state)
    visits, _value0, info = root_result
    groups: dict[int, list[Any]] = {}
    for action in state.legal_actions():
        pick = _pick_key(action)
        groups.setdefault(-1 if pick is None else int(pick), []).append(action)
    total_visits = sum(visits.values()) or 1.0
    rows: dict[Optional[int], dict[str, Any]] = {}
    for pick, actions in groups.items():
        def rank(action: Any) -> tuple[float, float, int]:
            idx = int(encode_action(action, state))
            return (
                float(visits.get(idx, 0.0)),
                float(info.get(idx, (policy.get(idx, 0.0), None))[0]),
                -idx,
            )

        representative = max(actions, key=rank)
        rep_idx = int(encode_action(representative, state))
        group_idxs = [int(encode_action(action, state)) for action in actions]
        domino_id = None if pick == -1 else int(pick)
        rows[domino_id] = {
            "pick_domino_id": domino_id,
            "representative_action_idx": rep_idx,
            "group_visits": float(sum(visits.get(idx, 0.0) for idx in group_idxs)),
            "group_visit_fraction": float(
                sum(visits.get(idx, 0.0) for idx in group_idxs) / total_visits),
            "raw_prior": float(sum(policy.get(idx, 0.0) for idx in group_idxs)),
            "root_q": info.get(rep_idx, (0.0, None))[1],
        }
    return rows


def phase0_equivalence(
    search: DenialSearch,
    state: GameState,
    *,
    seed: int,
) -> dict[str, Any]:
    """Assert byte-identical factored and in-label root Q values."""
    namespace = f"phase0_s{search.config.root_search_sims}_seed{seed}"
    root = search._root_search(state, seed_override=seed, cache_namespace=namespace)
    factored = root_q_by_pick(search, state, root)
    label = search.search_position(state, root_result=root)
    embedded = {row["pick_domino_id"]: row["headline_edge"] for row in label["per_pick"]}
    if set(factored) != set(embedded):
        raise AssertionError(
            f"Phase-0 pick sets differ: factored={sorted(factored, key=str)} "
            f"embedded={sorted(embedded, key=str)}")
    comparisons = []
    for pick in factored:
        left = factored[pick]["root_q"]
        right = embedded[pick]
        equal = ((left is None and right is None)
                 or (left is not None and right is not None
                     and struct.pack("!d", float(left)) == struct.pack("!d", float(right))))
        comparisons.append({"pick_domino_id": pick, "factored": left,
                            "embedded": right, "byte_identical": equal})
        if not equal:
            raise AssertionError(
                f"Phase-0 root Q mismatch for pick {pick}: {left!r} != {right!r}")
    return {"passed": True, "state_key": public_state_key(state), "seed": int(seed),
            "root_search_sims": int(search.config.root_search_sims),
            "comparisons": comparisons}


def tie_guarded_flip(
    root_q: dict[Optional[int], float],
    searched_ref: dict[Optional[int], float],
    *,
    tie_tolerance: float,
) -> dict[str, Any]:
    """Classify one root decision, retaining tie-killed would-be flips."""
    usable = {pick: float(value) for pick, value in root_q.items() if value is not None}
    if not usable or not searched_ref:
        return {"flip": False, "would_be_flip": False, "tie_guard_killed": False,
                "root_top": None, "search_best": None, "value_at_risk": None}
    root_top = max(usable, key=lambda pick: (usable[pick], -(pick if pick is not None else -1)))
    search_best = max(searched_ref, key=lambda pick: (
        searched_ref[pick], -(pick if pick is not None else -1)))
    difference = float(searched_ref[search_best] - searched_ref[root_top])
    would_be = root_top != search_best
    flip = bool(would_be and difference > float(tie_tolerance))
    return {
        "flip": flip,
        "would_be_flip": bool(would_be),
        "tie_guard_killed": bool(would_be and not flip),
        "root_top": root_top,
        "search_best": search_best,
        "value_at_risk": difference,
    }


def _config(*, seed: int, sims: int = 3200, chance_k: int = 16) -> SearchConfig:
    return SearchConfig(
        pick_plies=8,
        chance_k=int(chance_k),
        seed=int(seed),
        placement_top_k=2,
        root_search_sims=int(sims),
        policy_temperature=0.10,
        tie_tolerance=1e-6,
        uncertainty_z=1.0,
    )


class Runtime:
    """One loaded network/evaluator shared across all resumable phases."""

    def __init__(self, args: argparse.Namespace):
        net, checkpoint_cfg = load_checkpoint_network(args.checkpoint, args.device)
        self.evaluator = AZBatchEvaluator(
            net,
            device=args.device,
            batch_size=args.leaf_batch_size,
            margin_gain=float(checkpoint_cfg.get("margin_gain", 2.0)),
            alpha=float(checkpoint_cfg.get("alpha", 0.5)),
        )
        self.search = DenialSearch(
            self.evaluator, checkpoint_path=args.checkpoint,
            config=_config(seed=TREE_SEEDS[0]),
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()


def _validate_artifact_rows(
    rows: Sequence[dict[str, Any]], *, positions_sha: str, checkpoint_sha: str,
) -> None:
    for row in rows:
        if row.get("positions_sha256") != positions_sha:
            raise ValueError("existing artifact uses a different frozen position set")
        if row.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("existing artifact uses a different checkpoint")


def run_phase0(
    args: argparse.Namespace,
    runtime: Runtime,
    records: Sequence[tuple[GameState, dict[str, Any]]],
    positions_sha: str,
) -> dict[str, Any]:
    output = Path(args.run_dir) / "phase0_gate.json"
    checkpoint_sha = runtime.search.checkpoint_sha256
    started = time.perf_counter()
    runtime.search.config = _config(seed=TREE_SEEDS[0])
    result = phase0_equivalence(runtime.search, records[0][0], seed=TREE_SEEDS[0])
    result.update({
        "checkpoint_sha256": checkpoint_sha,
        "positions_sha256": positions_sha,
        "elapsed_seconds": time.perf_counter() - started,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Phase 0 passed: {output}", flush=True)
    return result


def _tree_pick_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = ("pick_domino_id", "raw_prior", "group_visits", "headline_edge",
              "searched_value_actor", "policy_target", "fragility")
    return {field: row.get(field) for field in fields}


def run_phase1(
    args: argparse.Namespace,
    runtime: Runtime,
    records: Sequence[tuple[GameState, dict[str, Any]]],
    positions_sha: str,
) -> None:
    checkpoint_sha = runtime.search.checkpoint_sha256
    for seed in TREE_SEEDS:
        output = Path(args.run_dir) / f"tree_seed{seed}.jsonl"
        existing = _read_jsonl(output)
        _validate_artifact_rows(existing, positions_sha=positions_sha,
                                checkpoint_sha=checkpoint_sha)
        completed = {int(row["position_index"]) for row in existing}
        if len(completed) != len(existing):
            raise ValueError(f"duplicate Phase-1 rows in {output}")
        runtime.search.config = _config(seed=seed)
        for index, (state, source) in enumerate(records):
            if index in completed:
                continue
            started = time.perf_counter()
            # Explicit root seed makes the 3200 cell byte-identical to Phase 2;
            # config.seed independently controls chance-node CRN sampling.
            root = runtime.search._root_search(
                state, seed_override=seed,
                cache_namespace=f"tree_s3200_seed{seed}")
            label = runtime.search.search_position(state, root_result=root)
            artifact = {
                "phase": 1,
                "position_index": index,
                "state_key": public_state_key(state),
                "source": source,
                "seed": seed,
                "root_search_sims": 3200,
                "chance_k": 16,
                "checkpoint_sha256": checkpoint_sha,
                "positions_sha256": positions_sha,
                "elapsed_seconds": time.perf_counter() - started,
                "per_pick": [_tree_pick_row(row) for row in label["per_pick"]],
            }
            _append_jsonl(output, [artifact])
            print(f"Phase 1 seed={seed} position={index + 1}/{len(records)}", flush=True)


def run_phase2(
    args: argparse.Namespace,
    runtime: Runtime,
    records: Sequence[tuple[GameState, dict[str, Any]]],
    positions_sha: str,
) -> None:
    output = Path(args.run_dir) / "root_ladder.jsonl"
    checkpoint_sha = runtime.search.checkpoint_sha256
    existing = _read_jsonl(output)
    _validate_artifact_rows(existing, positions_sha=positions_sha,
                            checkpoint_sha=checkpoint_sha)
    completed = {(int(row["position_index"]), int(row["sims"]), int(row["seed"]))
                 for row in existing}
    if len(completed) != len(existing):
        raise ValueError(f"duplicate Phase-2 rows in {output}")
    for sims in args.sims:
        runtime.search.config = _config(seed=ROOT_SEEDS[0], sims=sims)
        for seed in ROOT_SEEDS:
            for index, (state, _source) in enumerate(records):
                cell = (index, int(sims), seed)
                if cell in completed:
                    continue
                started = time.perf_counter()
                root = runtime.search._root_search(
                    state, seed_override=seed,
                    cache_namespace=f"s{sims}_seed{seed}")
                picks = root_q_by_pick(runtime.search, state, root)
                artifact = {
                    "phase": 2,
                    "position_index": index,
                    "state_key": public_state_key(state),
                    "sims": int(sims),
                    "seed": seed,
                    "checkpoint_sha256": checkpoint_sha,
                    "positions_sha256": positions_sha,
                    "elapsed_seconds": time.perf_counter() - started,
                    "per_pick": list(picks.values()),
                }
                _append_jsonl(output, [artifact])
                print(f"Phase 2 sims={sims} seed={seed} "
                      f"position={index + 1}/{len(records)}", flush=True)


def _float_bits(value: float) -> bytes:
    return struct.pack("!d", float(value))


def _tied_pairs(values: dict[Optional[int], float]) -> set[tuple[Optional[int], Optional[int]]]:
    pairs = set()
    for left, right in combinations(sorted(values, key=lambda x: -1 if x is None else x), 2):
        if _float_bits(values[left]) == _float_bits(values[right]):
            pairs.add((left, right))
    return pairs


def run_phase3(
    args: argparse.Namespace,
    runtime: Runtime,
    records: Sequence[tuple[GameState, dict[str, Any]]],
    positions_sha: str,
) -> None:
    output = Path(args.run_dir) / "tie_probe.jsonl"
    checkpoint_sha = runtime.search.checkpoint_sha256
    k16_rows = _load_tree_rows(args.run_dir, positions_sha, checkpoint_sha)[TREE_SEEDS[0]]
    existing = _read_jsonl(output)
    _validate_artifact_rows(existing, positions_sha=positions_sha,
                            checkpoint_sha=checkpoint_sha)
    completed = {int(row["position_index"]) for row in existing}
    indices = [index for index in TIE_POSITION_INDICES if index < len(records)]
    runtime.search.config = _config(seed=TREE_SEEDS[0], chance_k=32)
    for index in indices:
        if index in completed:
            continue
        state = records[index][0]
        started = time.perf_counter()
        root = runtime.search._root_search(
            state, seed_override=TREE_SEEDS[0],
            cache_namespace=f"tie_k32_seed{TREE_SEEDS[0]}")
        label32 = runtime.search.search_position(state, root_result=root)
        values16 = {row["pick_domino_id"]: float(row["searched_value_actor"])
                    for row in k16_rows[index]["per_pick"]}
        values32 = {row["pick_domino_id"]: float(row["searched_value_actor"])
                    for row in label32["per_pick"]}
        ties16, ties32 = _tied_pairs(values16), _tied_pairs(values32)
        artifact = {
            "phase": 3,
            "position_index": index,
            "state_key": public_state_key(state),
            "seed": TREE_SEEDS[0],
            "checkpoint_sha256": checkpoint_sha,
            "positions_sha256": positions_sha,
            "elapsed_seconds": time.perf_counter() - started,
            "searched_k16": values16,
            "searched_k32": values32,
            "tie_pairs_k16": [list(pair) for pair in sorted(ties16, key=str)],
            "tie_pairs_k32": [list(pair) for pair in sorted(ties32, key=str)],
            "dissolved_ties": [list(pair) for pair in sorted(ties16 - ties32, key=str)],
            "new_ties": [list(pair) for pair in sorted(ties32 - ties16, key=str)],
        }
        _append_jsonl(output, [artifact])
        print(f"Phase 3 position={index + 1}/{len(records)}", flush=True)


def _load_tree_rows(
    run_dir: str | Path, positions_sha: str, checkpoint_sha: str,
) -> dict[int, dict[int, dict[str, Any]]]:
    out = {}
    for seed in TREE_SEEDS:
        rows = _read_jsonl(Path(run_dir) / f"tree_seed{seed}.jsonl")
        _validate_artifact_rows(rows, positions_sha=positions_sha,
                                checkpoint_sha=checkpoint_sha)
        indexed = {int(row["position_index"]): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"duplicate Phase-1 rows for seed {seed}")
        out[seed] = indexed
    return out


def _population_sd(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _stable_reference(
    tree_rows: dict[int, dict[int, dict[str, Any]]], positions: int,
) -> tuple[dict[int, dict[Optional[int], float]], list[dict[str, Any]]]:
    references: dict[int, dict[Optional[int], float]] = {}
    stability = []
    for index in range(positions):
        by_seed = {}
        for seed in TREE_SEEDS:
            if index not in tree_rows[seed]:
                raise ValueError(f"Phase 1 is incomplete: seed={seed}, position={index}")
            by_seed[seed] = {
                row["pick_domino_id"]: float(row["searched_value_actor"])
                for row in tree_rows[seed][index]["per_pick"]
            }
        picks = set.intersection(*(set(rows) for rows in by_seed.values()))
        if any(set(rows) != picks for rows in by_seed.values()):
            raise ValueError(f"Phase-1 pick IDs changed across seeds at position {index}")
        references[index] = {}
        for pick in picks:
            values = [by_seed[seed][pick] for seed in TREE_SEEDS]
            reference = float(statistics.median(values))
            references[index][pick] = reference
            stability.append({
                "position_index": index,
                "pick_domino_id": pick,
                "searched_by_seed": dict(zip(TREE_SEEDS, values)),
                "searched_ref": reference,
                "searched_seed_sd": _population_sd(values),
            })
    return references, stability


def _competition_ranks(values: dict[Optional[int], float]) -> dict[Optional[int], int]:
    return {pick: 1 + sum(other > value for other in values.values())
            for pick, value in values.items()}


def _root_ladder_index(
    rows: Sequence[dict[str, Any]], positions_sha: str, checkpoint_sha: str,
) -> dict[tuple[int, int, int], dict[Optional[int], float]]:
    _validate_artifact_rows(rows, positions_sha=positions_sha, checkpoint_sha=checkpoint_sha)
    indexed = {}
    for row in rows:
        key = (int(row["position_index"]), int(row["sims"]), int(row["seed"]))
        if key in indexed:
            raise ValueError(f"duplicate Phase-2 cell: {key}")
        indexed[key] = {pick["pick_domino_id"]: pick.get("root_q")
                        for pick in row["per_pick"]}
    return indexed


def _phase_elapsed(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    seconds = float(sum(float(row.get("elapsed_seconds", 0.0)) for row in rows))
    return {"elapsed_seconds": seconds, "gpu_hours": seconds / 3600.0}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    positions_sha = file_sha256(args.positions_path)
    checkpoint_sha = sha256_file(args.checkpoint)
    records = load_frozen_positions(args.positions_path)
    tree_rows = _load_tree_rows(args.run_dir, positions_sha, checkpoint_sha)
    references, searched_stability = _stable_reference(tree_rows, len(records))
    ladder_rows = _read_jsonl(Path(args.run_dir) / "root_ladder.jsonl")
    ladder = _root_ladder_index(ladder_rows, positions_sha, checkpoint_sha)
    expected = {(index, int(sims), seed) for index in range(len(records))
                for sims in args.sims for seed in ROOT_SEEDS}
    if set(ladder) != expected:
        missing = sorted(expected - set(ladder))
        raise ValueError(f"Phase 2 is incomplete; first missing cells: {missing[:5]}")

    secondary = []
    secondary_keys = set()
    for index, values in references.items():
        ranks = _competition_ranks(values)
        for pick, rank in ranks.items():
            if rank >= 2:
                secondary_keys.add((index, pick))
                secondary.append({"position_index": index, "pick_domino_id": pick,
                                  "searched_rank": rank, "searched_ref": values[pick]})

    fragility_by_sims: dict[int, list[float]] = {int(sims): [] for sims in args.sims}
    root_sd_by_sims: dict[int, list[float]] = {int(sims): [] for sims in args.sims}
    per_secondary = []
    for item in secondary:
        index, pick = item["position_index"], item["pick_domino_id"]
        row = dict(item)
        row["by_sims"] = {}
        for sims in args.sims:
            root_values = [ladder[(index, int(sims), seed)].get(pick) for seed in ROOT_SEEDS]
            finite = [float(value) for value in root_values if value is not None]
            fragilities = [value - references[index][pick] for value in finite]
            fragility_by_sims[int(sims)].extend(fragilities)
            sd = _population_sd(finite)
            root_sd_by_sims[int(sims)].append(sd)
            row["by_sims"][str(sims)] = {
                "root_q_by_seed": dict(zip(ROOT_SEEDS, root_values)),
                "root_q_seed_sd": sd,
                "fragility_seed_median": (float(statistics.median(fragilities))
                                            if fragilities else None),
            }
        low = row["by_sims"].get("3200", {}).get("fragility_seed_median")
        high = row["by_sims"].get(str(args.sims[-1]), {}).get("fragility_seed_median")
        row["slope_high_minus_3200"] = (None if low is None or high is None
                                         else float(high - low))
        per_secondary.append(row)

    stability_by_key = {(row["position_index"], row["pick_domino_id"]): row
                        for row in searched_stability}
    all_pick_stability = []
    for key, source_row in sorted(stability_by_key.items(), key=lambda item: str(item[0])):
        row = dict(source_row)
        root_values = [ladder[(key[0], 3200, seed)].get(key[1]) for seed in ROOT_SEEDS]
        finite = [float(value) for value in root_values if value is not None]
        root_sd = _population_sd(finite)
        row["root_q_3200_seed_sd"] = root_sd
        row["root_q_3200_observed_seeds"] = len(finite)
        row["searched_sd_over_root_q_3200_sd"] = (
            row["searched_seed_sd"] / root_sd if root_sd > 0.0 else None)
        all_pick_stability.append(row)
    secondary_stability = [row for row in all_pick_stability
                           if (row["position_index"], row["pick_domino_id"]) in secondary_keys]

    flip_tables = {}
    stable_sets = {}
    for sims in args.sims:
        per_position = []
        buckets = {str(count): 0 for count in range(6)}
        stable_positions = []
        stable_set = set()
        for index in range(len(records)):
            events = []
            for seed in ROOT_SEEDS:
                event = tie_guarded_flip(
                    ladder[(index, int(sims), seed)], references[index],
                    tie_tolerance=TREE_CONFIG["tie_tolerance"])
                event["seed"] = seed
                events.append(event)
            count = sum(event["flip"] for event in events)
            buckets[str(count)] += 1
            summary = {
                "position_index": index,
                "flip_seeds": count,
                "would_be_flip_seeds": sum(event["would_be_flip"] for event in events),
                "tie_guard_killed_seeds": sum(event["tie_guard_killed"] for event in events),
                "events": events,
            }
            per_position.append(summary)
            if count >= 4:
                stable_set.add(index)
                flips = [event for event in events if event["flip"]]
                root_top = Counter(event["root_top"] for event in flips).most_common(1)[0][0]
                stable_positions.append({
                    "position_index": index,
                    "root_top": root_top,
                    "root_top_by_seed": {event["seed"]: event["root_top"] for event in events},
                    "search_best": flips[0]["search_best"],
                    "value_at_risk": float(statistics.median(
                        event["value_at_risk"] for event in flips)),
                    "flip_seeds": count,
                })
        stable_sets[int(sims)] = stable_set
        flip_tables[str(sims)] = {
            "stability_buckets_0_to_5": buckets,
            "stable_flip_count_ge_4_of_5": len(stable_positions),
            "stable_flip_positions": stable_positions,
            "would_be_flips_killed_by_tie_guard": sum(
                row["tie_guard_killed_seeds"] for row in per_position),
            "per_position": per_position,
        }

    tie_rows = _read_jsonl(Path(args.run_dir) / "tie_probe.jsonl")
    _validate_artifact_rows(tie_rows, positions_sha=positions_sha,
                            checkpoint_sha=checkpoint_sha)
    expected_tie_indices = {index for index in TIE_POSITION_INDICES if index < len(records)}
    actual_tie_indices = {int(row["position_index"]) for row in tie_rows}
    if len(actual_tie_indices) != len(tie_rows) or actual_tie_indices != expected_tie_indices:
        raise ValueError(
            f"Phase 3 is incomplete or duplicated: {sorted(actual_tie_indices)} != "
            f"{sorted(expected_tie_indices)}")
    tie_metrics = {
        "positions": len(tie_rows),
        "positions_with_ties_k16": sum(bool(row["tie_pairs_k16"]) for row in tie_rows),
        "positions_with_ties_k32": sum(bool(row["tie_pairs_k32"]) for row in tie_rows),
        "tied_pairs_k16": sum(len(row["tie_pairs_k16"]) for row in tie_rows),
        "tied_pairs_k32": sum(len(row["tie_pairs_k32"]) for row in tie_rows),
        "dissolved_ties": sum(len(row["dissolved_ties"]) for row in tie_rows),
        "new_ties": sum(len(row["new_ties"]) for row in tie_rows),
        "per_position": tie_rows,
    }

    fragility_table = {str(sims): distribution(fragility_by_sims[int(sims)])
                       for sims in args.sims}
    slope_values = [row["slope_high_minus_3200"] for row in per_secondary
                    if row["slope_high_minus_3200"] is not None]
    searched_sds = [row["searched_seed_sd"] for row in all_pick_stability]
    root_3200_sds = [row["root_q_3200_seed_sd"] for row in all_pick_stability]
    searched_sd_dist = distribution(searched_sds)
    root_3200_sd_dist = distribution(root_3200_sds)
    secondary_searched_sd_dist = distribution(
        row["searched_seed_sd"] for row in secondary_stability)
    secondary_root_3200_sd_dist = distribution(root_sd_by_sims[3200])
    factoring_suspect = bool(
        searched_sd_dist["median"] is not None
        and root_3200_sd_dist["median"] is not None
        and float(searched_sd_dist["median"]) >= float(root_3200_sd_dist["median"]))

    high_sims = int(args.sims[-1])
    persistent = stable_sets[3200] & stable_sets[high_sims]
    persistent_risks = [
        row["value_at_risk"]
        for row in flip_tables["3200"]["stable_flip_positions"]
        if row["position_index"] in persistent
    ]
    median_high = fragility_table[str(high_sims)]["median"]
    mean_3200 = fragility_table["3200"]["mean"]
    median_slope = distribution(slope_values)["median"]
    median_root_sd = secondary_root_3200_sd_dist["median"]
    systematic_gates = {
        "fragility_high_sims_at_least_0_08": bool(median_high is not None and median_high >= 0.08),
        "median_abs_slope_at_most_0_02": bool(median_slope is not None
                                                and abs(float(median_slope)) <= 0.02),
        "median_root_q_3200_seed_sd_below_0_05": bool(
            median_root_sd is not None and median_root_sd < 0.05),
        "persistent_stable_flip_positions_at_least_8": len(persistent) >= 8,
        "persistent_median_value_at_risk_at_least_0_10": bool(
            persistent_risks and statistics.median(persistent_risks) >= 0.10),
    }
    systematic = all(systematic_gates.values())
    medians = [fragility_table[str(sims)]["median"] for sims in args.sims]
    monotone_to_zero = bool(
        all(value is not None for value in medians)
        and all(float(left) >= float(right) for left, right in zip(medians, medians[1:]))
        and abs(float(medians[-1])) <= 0.05)
    count_3200 = len(stable_sets[3200])
    count_high = len(stable_sets[high_sims])
    flips_collapsed = bool(count_3200 > 0 and count_high <= count_3200 / 2.0)
    sd_comparable = bool(
        median_root_sd is not None and mean_3200 is not None
        and float(median_root_sd) >= 0.5 * abs(float(mean_3200)))
    noise_signals = {
        "fragility_monotone_and_abs_high_sims_at_most_0_05": monotone_to_zero,
        "stable_flip_count_halved_by_high_sims": flips_collapsed,
        "median_seed_sd_at_least_half_abs_mean_fragility_3200": sd_comparable,
    }
    if systematic:
        classification = "SYSTEMATIC"
        route = "build_ply1_opponent_reply_label_channel_then_small_pilot_retrain"
        verdict = (
            "Secondary-pick overvaluation meets every pre-registered persistence, stability, "
            "move-flip, and value-at-risk gate. Route to the ply-1 opponent-reply label channel "
            "and a small held-out pilot; downweight overrated secondary picks only.")
    elif any(noise_signals.values()):
        classification = "NOISE"
        route = "close_curriculum_lever_raise_advisor_sims_if_flips_die"
        dead_at = next((sims for sims in args.sims if not stable_sets[int(sims)]), None)
        verdict = (
            "At least one pre-registered sampling-noise condition fired. Do not build a "
            f"curriculum; the first rung with zero >=4/5 flips is {dead_at!r} simulations.")
    else:
        classification = "BETWEEN"
        route = "default_to_noise_close_curriculum_lever"
        verdict = (
            "The effect is marginal or partly stable but misses the systematic routing gates. "
            "Per registration, default to NOISE and close the curriculum lever.")

    gate_path = Path(args.run_dir) / "phase0_gate.json"
    if not gate_path.exists():
        raise ValueError("Phase 0 artifact is missing")
    phase0 = json.loads(gate_path.read_text(encoding="utf-8"))
    if not phase0.get("passed"):
        raise ValueError("Phase 0 did not pass")
    if (phase0.get("positions_sha256") != positions_sha
            or phase0.get("checkpoint_sha256") != checkpoint_sha):
        raise ValueError("Phase 0 artifact uses different inputs")
    tree_flat = [row for seed in TREE_SEEDS for row in tree_rows[seed].values()]
    report = {
        "schema_version": 1,
        "scope": "diagnosis only; no retraining, label change, value-semantic change, or throughput change",
        "provenance": {
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_expected_prefix": "4bf07b0c",
            "checkpoint_matches_expected_prefix": checkpoint_sha.startswith("4bf07b0c"),
            "frozen_positions_path": str(args.positions_path),
            "frozen_positions_sha256": positions_sha,
            "positions": len(records),
            "tree_seeds": list(TREE_SEEDS),
            "root_seeds": list(ROOT_SEEDS),
            "sims": list(args.sims),
            "actor_frame": ACTOR_FRAME,
            "tree_config": TREE_CONFIG,
            "phase0": {"root_search_sims": 3200, "chance_k": 16,
                       "position_index": 0, "seed": TREE_SEEDS[0]},
            "phase2": {"root_search_only": True, "chance_k": 16,
                       "cache_namespace": "s{sims}_seed{seed}"},
            "phase3": {"chance_k": [16, 32], "seed": TREE_SEEDS[0],
                       "position_indices": list(TIE_POSITION_INDICES)},
            "phase_elapsed": {
                "phase0": {"elapsed_seconds": float(phase0.get("elapsed_seconds", 0.0)),
                           "gpu_hours": float(phase0.get("elapsed_seconds", 0.0)) / 3600.0},
                "phase1": _phase_elapsed(tree_flat),
                "phase2": _phase_elapsed(ladder_rows),
                "phase3": _phase_elapsed(tie_rows),
            },
            "reserved_test_split_opened": False,
        },
        "phase0_equivalence": phase0,
        "secondary_pick_definition": "competition rank >= 2 by median searched value; tied best picks remain rank 1",
        "secondary_pick_count": len(secondary),
        "sim_ladder_fragility": fragility_table,
        "secondary_pick_slopes": {
            "definition": f"median_seed_fragility({high_sims}) - median_seed_fragility(3200)",
            "distribution": distribution(slope_values),
            "per_pick": per_secondary,
        },
        "root_q_seed_sd": {str(sims): distribution(root_sd_by_sims[int(sims)])
                           for sims in args.sims},
        "searched_reference_stability": {
            "searched_seed_sd": searched_sd_dist,
            "root_q_3200_seed_sd": root_3200_sd_dist,
            "secondary_searched_seed_sd": secondary_searched_sd_dist,
            "secondary_root_q_3200_seed_sd": secondary_root_3200_sd_dist,
            "factoring_suspect": factoring_suspect,
            "picks_where_searched_sd_is_not_lower_than_root_sd": sum(
                row["searched_seed_sd"] >= row["root_q_3200_seed_sd"]
                for row in all_pick_stability),
            "per_pick": all_pick_stability,
        },
        "flips": flip_tables,
        "stable_flip_count_trend": {str(sims): len(stable_sets[int(sims)])
                                    for sims in args.sims},
        "persistent_stable_flip_positions_3200_to_high_sims": sorted(persistent),
        "tie_probe": tie_metrics,
        "routing": {
            "classification": classification,
            "systematic_gates": systematic_gates,
            "noise_signals": noise_signals,
            "persistent_stable_flip_count": len(persistent),
            "persistent_median_value_at_risk": (
                float(statistics.median(persistent_risks)) if persistent_risks else None),
            "route": route,
            "verdict": verdict,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(f"{classification}: {verdict}", flush=True)
    print(f"wrote {output}", flush=True)
    return report


def _parse_sims(value: str) -> tuple[int, ...]:
    sims = tuple(int(part) for part in value.split(",") if part.strip())
    if (len(sims) != 3 or sims[1] != 3200 or sims[0] >= 3200
            or sims[2] <= 3200 or any(item <= 0 for item in sims)):
        raise argparse.ArgumentTypeError(
            "--sims requires ordered low,3200,high positive rungs")
    return sims


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("gate", "trees", "ladder", "ties", "report", "all"),
                        default="all")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CURRENT_BEST))
    parser.add_argument("--positions-path", default=str(DEFAULT_POSITIONS))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    parser.add_argument("--sims", type=_parse_sims, default=SIMS,
                        help="three comma-separated rungs including 3200; use 800,3200,6400 if needed")
    parser.add_argument("--leaf-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    records = load_frozen_positions(args.positions_path)
    if len(records) != 50:
        raise ValueError(f"secondary-pick test requires the frozen 50, found {len(records)}")
    positions_sha = file_sha256(args.positions_path)
    if args.phase == "report":
        build_report(args)
        return 0
    runtime = Runtime(args)
    # The gate is mandatory at the start of every compute invocation, even when
    # resuming a later phase.  It is cheap relative to the overnight study and
    # prevents a stale artifact from masking factoring drift.
    run_phase0(args, runtime, records, positions_sha)
    if args.phase == "gate":
        return 0
    if args.phase in ("trees", "all"):
        run_phase1(args, runtime, records, positions_sha)
    if args.phase in ("ladder", "all"):
        run_phase2(args, runtime, records, positions_sha)
    if args.phase in ("ties", "all"):
        run_phase3(args, runtime, records, positions_sha)
    if args.phase == "all":
        build_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
