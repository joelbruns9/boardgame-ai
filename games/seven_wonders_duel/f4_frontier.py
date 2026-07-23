"""Analyze the forced/128 leaf-1 concurrency frontier and leaf-X override."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .f4_quality import CONTRACT_PATH


def _ratio_lower(candidate: list[float], baseline: list[float], seed: int) -> float:
    if not candidate or len(candidate) != len(baseline):
        return float("-inf")
    ratios = [c / b for c, b in zip(candidate, baseline) if b > 0]
    if len(ratios) != len(candidate):
        return float("-inf")
    rng = random.Random(seed)
    samples = [
        mean(ratios[rng.randrange(len(ratios))] for _ in ratios)
        for _ in range(10_000)
    ]
    samples.sort()
    return samples[int(0.05 * (len(samples) - 1))]


def summarize(rows: list[dict], contract: dict) -> dict:
    frontier = contract["concurrency_frontier"]
    production = contract["production_semantics"]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("force_expand_root_chance") is not True:
            raise ValueError("frontier row is not force=true")
        if int(row.get("sims", -1)) != production["full_sims"]:
            raise ValueError("frontier row is not sims=128")
        key = (
            int(row["leaf_batch"]),
            int(row["slots"]),
            int(row.get("scheduler_workers", 1)),
        )
        grouped[key].append(row)

    configs = []
    for (leaf_batch, slots, workers), group in sorted(grouped.items()):
        eligible = not any(row.get("oom_count", 0) or row.get("failed_games", 0) for row in group)
        configs.append(
            {
                "leaf_batch": leaf_batch,
                "slots": slots,
                "scheduler_workers": workers,
                "games_per_second": mean(row["games_per_second"] for row in group),
                "policy_eligible_targets_per_second": mean(
                    row["policy_eligible_targets_per_second"] for row in group
                ),
                "gpu_busy_fraction": mean(row["gpu_busy_fraction"] for row in group),
                "isolated_forward_rows_ratio": mean(
                    row["isolated_forward_rows_ratio"] for row in group
                ),
                "eligible": eligible,
                "raw": group,
            }
        )
    leaf1 = [row for row in configs if row["leaf_batch"] == 1 and row["eligible"]]
    if not leaf1:
        raise ValueError("frontier has no eligible leaf_batch=1 row")
    best = max(leaf1, key=lambda row: row["games_per_second"])
    knee_rows = [
        row
        for row in leaf1
        if row["gpu_busy_fraction"] >= frontier["gpu_knee"]["minimum_busy_fraction"]
        and row["isolated_forward_rows_ratio"]
        >= frontier["gpu_knee"]["minimum_isolated_forward_rows_ratio"]
    ]
    overrides = []
    for candidate in configs:
        if candidate["leaf_batch"] == 1 or not candidate["eligible"]:
            continue
        base_games = [row["games_per_second"] for row in best["raw"]]
        fast_games = [row["games_per_second"] for row in candidate["raw"]]
        base_targets = [row["policy_eligible_targets_per_second"] for row in best["raw"]]
        fast_targets = [
            row["policy_eligible_targets_per_second"] for row in candidate["raw"]
        ]
        games_lower = _ratio_lower(fast_games, base_games, 4040 + candidate["leaf_batch"])
        targets_lower = _ratio_lower(fast_targets, base_targets, 5040 + candidate["leaf_batch"])
        candidate["games_ratio_one_sided_95_lower"] = games_lower
        candidate["targets_ratio_one_sided_95_lower"] = targets_lower
        if (
            games_lower
            >= frontier["leaf_batch_override"][
                "minimum_games_per_second_lower_bound_ratio"
            ]
            and targets_lower
            >= frontier["leaf_batch_override"][
                "minimum_policy_targets_per_second_lower_bound_ratio"
            ]
        ):
            overrides.append(candidate)
    return {
        "schema": "f4-concurrency-frontier-1",
        "force_expand_root_chance": True,
        "sims": production["full_sims"],
        "leaf1_reaches_gpu_knee": bool(knee_rows),
        "best_leaf1": {key: value for key, value in best.items() if key != "raw"},
        "eligible_leaf_x_overrides": [
            {key: value for key, value in row.items() if key != "raw"}
            for row in overrides
        ],
        "configs": [{key: value for key, value in row.items() if key != "raw"} for row in configs],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line]
    result = summarize(rows, contract)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
