"""Matched forced-pick probes for Stage-2 roots with zero-visit pick groups.

For every affected root, this runs every available pick group with the same two
seeds and the same restricted-search budget.  That makes a formerly unvisited
group comparable to the groups ordinary MCTS did visit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import kingdomino_rust as kr

from games.kingdomino.action_codec import encode_action
from games.kingdomino.bga_reanalysis_corpus import DEFAULT_CORPUS
from games.kingdomino.deep_target_screen import DEFAULT_CHECKPOINT, _pick_key, _read_jsonl
from games.kingdomino.deep_target_stage2 import (
    DEFAULT_OUTPUT as DEFAULT_STAGE2,
    _forced_seed,
    _search_kwargs,
    aggregate_restricted_search,
)
from games.kingdomino.denial_search import load_checkpoint_network
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.late_model_policy_audit import prepare_bga_state
from games.kingdomino.self_play import make_rust_evaluator


SCHEMA = "kingdomino-deep-target-matched-forced-pick/v1"
SUMMARY_SCHEMA = "kingdomino-deep-target-matched-forced-pick-summary/v1"
DEFAULT_OUTPUT = Path(
    "runs/kingdomino/placement_audit/deep_target_stage2_matched_forced_pick_v1.jsonl"
)
DEFAULT_SUMMARY = Path(
    "runs/kingdomino/placement_audit/deep_target_stage2_matched_forced_pick_summary_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_probe(
    *,
    corpus_path: Path,
    stage2_path: Path,
    checkpoint: Path,
    output_path: Path,
    summary_path: Path,
    device: str,
    sims: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage2_rows = _read_jsonl(stage2_path)
    targets = [row for row in stage2_rows if row.get("forced_pick_probes")]
    corpus = {str(row["position_id"]): row for row in _read_jsonl(corpus_path)}
    net, config = load_checkpoint_network(checkpoint, device)
    margin_gain = float(config.get("margin_gain", 2.0))
    alpha = float(config.get("alpha", 0.5))
    evaluator = make_rust_evaluator(
        net,
        device=device,
        amp=bool(config.get("inference_amp", False)),
        margin_gain=margin_gain,
        alpha=alpha,
    )
    kwargs = _search_kwargs(config, margin_gain, alpha)

    rows: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, stage2 in enumerate(targets):
            position_id = str(stage2["position_id"])
            position = corpus[position_id]
            rules = position["state"].get("rules", {})
            state = prepare_bga_state(
                position["state"],
                harmony=bool(rules.get("harmony", True)),
                middle=bool(rules.get("middle_kingdom", True)),
            )
            rust_state = _rust_state_from_python(state)
            starved = {
                int(group["pick_domino_id"])
                for search in stage2["stage2_searches"]
                for group in search["pick_groups"]
                if group["pick_domino_id"] is not None and int(group["visits"]) == 0
            }
            all_pick_ids = sorted(
                {
                    int(group["pick_domino_id"])
                    for search in stage2["stage2_searches"]
                    for group in search["pick_groups"]
                    if group["pick_domino_id"] is not None
                }
            )
            pick_results: list[dict[str, Any]] = []
            for pick_id in all_pick_ids:
                allowed = sorted(
                    {
                        int(encode_action(action, state))
                        for action in state.legal_actions()
                        if _pick_key(state, int(encode_action(action, state))) == pick_id
                    }
                )
                searches: list[dict[str, Any]] = []
                for repeat in range(2):
                    seed = _forced_seed(position_id, repeat)
                    search_started = time.perf_counter()
                    children, root_value_p0 = kr.advisor_open_loop_search(
                        rust_state,
                        evaluator,
                        int(sims),
                        seed=seed,
                        root_allowed_actions=allowed,
                        **kwargs,
                    )
                    searches.append(
                        aggregate_restricted_search(
                            state,
                            children,
                            allowed_actions=allowed,
                            root_value_p0=float(root_value_p0),
                            elapsed_seconds=time.perf_counter() - search_started,
                            seed=seed,
                            expected_visits=int(sims),
                        )
                    )
                pick_results.append(
                    {
                        "pick_domino_id": pick_id,
                        "was_starved_at_4800": pick_id in starved,
                        "mean_root_value_actor": statistics.fmean(
                            search["root_value_actor"] for search in searches
                        ),
                        "searches": searches,
                    }
                )
            ranked = sorted(
                pick_results,
                key=lambda item: (-item["mean_root_value_actor"], item["pick_domino_id"]),
            )
            best_value = float(ranked[0]["mean_root_value_actor"])
            for result in pick_results:
                result["regret_vs_best_forced_q"] = best_value - float(
                    result["mean_root_value_actor"]
                )
            row = {
                "schema": SCHEMA,
                "position_id": position_id,
                "table_id": stage2["table_id"],
                "source_decision_index": stage2["source_decision_index"],
                "deck_count": stage2["deck_count"],
                "cohort_reasons": stage2["cohort_reasons"],
                "starved_pick_domino_ids": sorted(starved),
                "best_forced_pick_domino_id": ranked[0]["pick_domino_id"],
                "best_forced_root_value_actor": best_value,
                "pick_groups": pick_results,
            }
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            rows.append(row)
            print(f"  {index + 1}/{len(targets)} matched roots", flush=True)

    starved_groups = [
        group
        for row in rows
        for group in row["pick_groups"]
        if group["was_starved_at_4800"]
    ]
    starved_best = sum(
        group["regret_vs_best_forced_q"] <= 1e-12 for group in starved_groups
    )
    regret_buckets = Counter()
    for group in starved_groups:
        regret = float(group["regret_vs_best_forced_q"])
        if regret <= 0.01:
            regret_buckets["le_0.01"] += 1
        if regret <= 0.03:
            regret_buckets["le_0.03"] += 1
        if regret <= 0.05:
            regret_buckets["le_0.05"] += 1
    summary = {
        "schema": SUMMARY_SCHEMA,
        "positions": len(rows),
        "pick_groups_searched": sum(len(row["pick_groups"]) for row in rows),
        "starved_pick_groups": len(starved_groups),
        "starved_group_is_best": starved_best,
        "starved_group_regret_counts": dict(sorted(regret_buckets.items())),
        "sims_per_group": int(sims),
        "repeats": 2,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "corpus": str(corpus_path),
        "corpus_sha256": _sha256(corpus_path),
        "stage2": str(stage2_path),
        "stage2_sha256": _sha256(stage2_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sims", type=int, default=800)
    args = parser.parse_args()
    _rows, summary = run_probe(
        corpus_path=args.corpus,
        stage2_path=args.stage2,
        checkpoint=args.checkpoint,
        output_path=args.output,
        summary_path=args.summary,
        device=args.device,
        sims=args.sims,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
