"""Test B: incumbent prior-to-4,800-search top-1 absorption gap.

The Stage-1 artifact already stores the incumbent's raw forward policy on each
root.  Stage 2 stores two independent 4,800-simulation searches on the selected
subset.  This diagnostic compares prior/search agreement with search/search
self-agreement for both full joint actions and aggregated tile choices.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any

from games.kingdomino.deep_target_screen import _read_jsonl


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _clustered_gap(
    rows: list[dict[str, Any]], metric: str, samples: int, seed: int
) -> dict[str, float | int]:
    by_game: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_game[str(row["table_id"])].append(float(row[metric]))
    game_means = [statistics.fmean(values) for values in by_game.values()]
    estimate = statistics.fmean(game_means)
    rng = random.Random(seed)
    draws = [
        statistics.fmean(rng.choice(game_means) for _ in game_means)
        for _ in range(samples)
    ]
    return {
        "source_games": len(game_means),
        "game_weighted_mean": estimate,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "ci95_lower": _percentile(draws, 0.025),
        "ci95_upper": _percentile(draws, 0.975),
    }


def analyze(
    stage1_rows: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260813,
) -> dict[str, Any]:
    stage1_by_id = {str(row["position_id"]): row for row in stage1_rows}
    per_root: list[dict[str, Any]] = []
    for stage2 in stage2_rows:
        position_id = str(stage2["position_id"])
        stage1 = stage1_by_id[position_id]
        searches = stage2["stage2_searches"]
        if len(searches) != 2:
            raise ValueError(f"{position_id} does not have two Stage-2 searches")
        prior_action = int(stage1["raw_policy"]["selected_action"]["action_idx"])
        prior_pick = stage1["raw_policy"]["selected_pick_domino_id"]
        search_actions = [int(row["selected_action"]["action_idx"]) for row in searches]
        search_picks = [row["selected_pick_domino_id"] for row in searches]
        prior_action_agreement = statistics.fmean(
            float(prior_action == value) for value in search_actions
        )
        prior_pick_agreement = statistics.fmean(
            float(prior_pick == value) for value in search_picks
        )
        action_self_agreement = float(search_actions[0] == search_actions[1])
        pick_self_agreement = float(search_picks[0] == search_picks[1])
        per_root.append(
            {
                "position_id": position_id,
                "table_id": stage2["table_id"],
                "prior_action_top1": prior_action,
                "search_action_top1": search_actions,
                "prior_pick_top1": prior_pick,
                "search_pick_top1": search_picks,
                "prior_vs_search_action_agreement": prior_action_agreement,
                "search_action_self_agreement": action_self_agreement,
                "action_absorption_gap": action_self_agreement
                - prior_action_agreement,
                "prior_vs_search_pick_agreement": prior_pick_agreement,
                "search_pick_self_agreement": pick_self_agreement,
                "pick_absorption_gap": pick_self_agreement - prior_pick_agreement,
            }
        )

    def mean(field: str) -> float:
        return statistics.fmean(float(row[field]) for row in per_root)

    action_consensus = [row for row in per_root if row["search_action_self_agreement"]]
    pick_consensus = [row for row in per_root if row["search_pick_self_agreement"]]
    return {
        "schema": "kingdomino-absorption-gap/v1",
        "positions": len(per_root),
        "source_games": len({str(row["table_id"]) for row in per_root}),
        "raw_forward_source": "cached Stage-1 incumbent raw_policy",
        "joint_action": {
            "prior_vs_4800_agreement": mean("prior_vs_search_action_agreement"),
            "4800_cross_seed_self_agreement": mean("search_action_self_agreement"),
            "absorption_gap_self_minus_prior": mean("action_absorption_gap"),
            "game_clustered_gap": _clustered_gap(
                per_root, "action_absorption_gap", bootstrap_samples, bootstrap_seed
            ),
            "prior_matches_4800_consensus": (
                statistics.fmean(
                    row["prior_vs_search_action_agreement"] for row in action_consensus
                )
                if action_consensus
                else None
            ),
            "consensus_positions": len(action_consensus),
        },
        "tile_group": {
            "prior_vs_4800_agreement": mean("prior_vs_search_pick_agreement"),
            "4800_cross_seed_self_agreement": mean("search_pick_self_agreement"),
            "absorption_gap_self_minus_prior": mean("pick_absorption_gap"),
            "game_clustered_gap": _clustered_gap(
                per_root,
                "pick_absorption_gap",
                bootstrap_samples,
                bootstrap_seed + 1,
            ),
            "prior_matches_4800_consensus": (
                statistics.fmean(
                    row["prior_vs_search_pick_agreement"] for row in pick_consensus
                )
                if pick_consensus
                else None
            ),
            "consensus_positions": len(pick_consensus),
        },
        "per_root": per_root,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1",
        type=Path,
        default=Path(
            "runs/kingdomino/placement_audit/"
            "deep_target_screen_development_s800_r2.jsonl"
        ),
    )
    parser.add_argument(
        "--stage2",
        type=Path,
        default=Path(
            "runs/kingdomino/placement_audit/"
            "deep_target_stage2_development_s4800_r2.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/kingdomino/placement_audit/absorption_gap_v1.json"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    args = parser.parse_args()
    summary = analyze(
        _read_jsonl(args.stage1),
        _read_jsonl(args.stage2),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary.update(
        {
            "stage1": str(args.stage1),
            "stage1_sha256": _sha256(args.stage1),
            "stage2": str(args.stage2),
            "stage2_sha256": _sha256(args.stage2),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: summary[key] for key in ("positions", "source_games", "joint_action", "tile_group")},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
