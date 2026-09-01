"""Stage-3 30k qualification on the frozen Kingdomino deep-target cohort.

Each root receives two ordinary 30,000-simulation searches and, for a fair
tile-value comparison, two 10,000-simulation restricted searches for every
available pick group.  All searches in a repeat use the same seed.  The latter
is an equal-compute conditional teacher: ordinary root starvation cannot make a
tile look bad merely because it was not explored.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any

import kingdomino_rust as kr

from games.kingdomino.action_codec import encode_action
from games.kingdomino.bga_reanalysis_corpus import DEFAULT_CORPUS
from games.kingdomino.deep_target_screen import DEFAULT_CHECKPOINT, _pick_key, _read_jsonl, aggregate_search
from games.kingdomino.deep_target_stage2 import (
    DEFAULT_OUTPUT as DEFAULT_STAGE2,
    _search_kwargs,
    aggregate_restricted_search,
)
from games.kingdomino.deep_target_stage3_cohort import DEFAULT_OUTPUT as DEFAULT_COHORT
from games.kingdomino.denial_search import load_checkpoint_network
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.late_model_policy_audit import prepare_bga_state
from games.kingdomino.self_play import make_rust_evaluator


SCHEMA = "kingdomino-deep-target-stage3/v1"
SUMMARY_SCHEMA = "kingdomino-deep-target-stage3-summary/v1"
DEFAULT_OUTPUT = Path(
    "runs/kingdomino/placement_audit/deep_target_stage3_development_s30000_r2.jsonl"
)
DEFAULT_SUMMARY = Path(
    "runs/kingdomino/placement_audit/deep_target_stage3_summary_development_s30000_r2.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile of empty values")
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _clustered_interval(
    by_game: dict[str, list[float]],
    *,
    samples: int = 10_000,
    seed: int = 20260813,
) -> dict[str, float | int]:
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
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "ci95_lower": _percentile(draws, 0.025),
        "ci95_upper": _percentile(draws, 0.975),
    }


def game_clustered_interval(
    rows: list[dict[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 20260813,
) -> dict[str, float | int]:
    """Whole-source-game bootstrap for per-root mean matched pick regret."""
    by_game: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_game[str(row["table_id"])].append(
            statistics.fmean(float(value) for value in row["matched_pick_regret_by_repeat"])
        )
    return _clustered_interval(by_game, samples=samples, seed=seed)


def cross_seed_uplifts(
    row: dict[str, Any],
    *,
    chooser: str,
) -> list[float]:
    """Choose on one seed and score on the other to avoid winner's bias."""
    if chooser not in {"matched", "ordinary"}:
        raise ValueError(f"unknown chooser: {chooser}")
    uplifts: list[float] = []
    for repeat in range(2):
        validation_repeat = 1 - repeat
        candidate_pick = row["stage2_4800_searches"][repeat]["selected_pick_domino_id"]
        if chooser == "matched":
            selected_pick = row["matched_best_pick_by_repeat"][repeat]
        else:
            selected_pick = row["ordinary_30000_searches"][repeat][
                "selected_pick_domino_id"
            ]
        values = {
            group["pick_domino_id"]: float(group["root_value_actor"])
            for group in row["matched_10000_pick_groups_by_repeat"][validation_repeat]
        }
        uplifts.append(values[selected_pick] - values[candidate_pick])
    return uplifts


def _cross_seed_summary(
    rows: list[dict[str, Any]],
    *,
    chooser: str,
) -> dict[str, Any]:
    values = [value for row in rows for value in cross_seed_uplifts(row, chooser=chooser)]
    by_game: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_game[str(row["table_id"])].extend(cross_seed_uplifts(row, chooser=chooser))
    return {
        "decision_weighted_mean": statistics.fmean(values),
        "median": statistics.median(values),
        "positive": sum(value > 0 for value in values),
        "negative": sum(value < 0 for value in values),
        "greater_than_001": sum(value > 0.01 for value in values),
        "greater_than_003": sum(value > 0.03 for value in values),
        "less_than_minus_001": sum(value < -0.01 for value in values),
        "less_than_minus_003": sum(value < -0.03 for value in values),
        "game_clustered": _clustered_interval(by_game),
    }


def summarize_stage3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired_regrets = [
        float(regret)
        for row in rows
        for regret in row["matched_pick_regret_by_repeat"]
    ]
    ordinary_changes = sum(
        comparison["pick_changed_4800_to_30000"]
        for row in rows
        for comparison in row["paired_comparisons"]
    )
    ordinary_agree = sum(
        len({search["selected_pick_domino_id"] for search in row["ordinary_30000_searches"]})
        == 1
        for row in rows
    )
    matched_agree = sum(
        len(set(row["matched_best_pick_by_repeat"])) == 1 for row in rows
    )
    reasons = Counter(reason for row in rows for reason in row["cohort_reasons"])
    paired = len(paired_regrets)
    return {
        "positions": len(rows),
        "reason_counts": dict(sorted(reasons.items())),
        "paired_repeats": paired,
        "ordinary_pick_changes_4800_to_30000": ordinary_changes,
        "ordinary_pick_change_fraction": ordinary_changes / paired,
        "ordinary_30000_two_seed_pick_agreement": ordinary_agree / len(rows),
        "matched_teacher_two_seed_pick_agreement": matched_agree / len(rows),
        "matched_pick_regret_decision_weighted_mean": statistics.fmean(paired_regrets),
        "matched_pick_regret_median": statistics.median(paired_regrets),
        "matched_pick_regret_max": max(paired_regrets),
        "matched_pick_regret_le_001": sum(value <= 0.01 for value in paired_regrets),
        "matched_pick_regret_le_003": sum(value <= 0.03 for value in paired_regrets),
        "matched_pick_regret_le_005": sum(value <= 0.05 for value in paired_regrets),
        "matched_pick_regret_gt_003": sum(value > 0.03 for value in paired_regrets),
        "matched_pick_regret_gt_005": sum(value > 0.05 for value in paired_regrets),
        "game_clustered_regret": game_clustered_interval(rows),
        "cross_seed_matched_teacher_uplift": _cross_seed_summary(
            rows, chooser="matched"
        ),
        "cross_seed_ordinary_30000_uplift": _cross_seed_summary(
            rows, chooser="ordinary"
        ),
        "confirmation_positions_searched": sum(row["split"] == "confirmation" for row in rows),
    }


def run_stage3(
    *,
    corpus_path: Path,
    stage2_path: Path,
    cohort_path: Path,
    checkpoint: Path,
    output_path: Path,
    summary_path: Path,
    device: str,
    ordinary_sims: int = 30_000,
    matched_sims: int = 10_000,
    limit: int | None = None,
    reuse_ordinary_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(cohort_path.read_text(encoding="utf-8"))
    split = manifest.get("split")
    if split not in {"development", "confirmation"} or not manifest.get(
        "selection_frozen_before_stage3"
    ):
        raise ValueError(
            "Stage-3 requires a frozen development or confirmation cohort"
        )
    if _sha256(stage2_path) != manifest["stage2_sha256"]:
        raise ValueError("Stage-2 hash does not match frozen Stage-3 cohort")
    entries = list(manifest["entries"])
    if limit is not None:
        entries = entries[: max(0, int(limit))]
    if not entries:
        raise ValueError("empty Stage-3 cohort")

    corpus = {str(row["position_id"]): row for row in _read_jsonl(corpus_path)}
    stage2 = {str(row["position_id"]): row for row in _read_jsonl(stage2_path)}
    reusable_ordinary = (
        {str(row["position_id"]): row for row in _read_jsonl(reuse_ordinary_path)}
        if reuse_ordinary_path is not None
        else {}
    )
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
    reused_ordinary_positions = 0
    rerun_ordinary_positions = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, entry in enumerate(entries):
            position_id = str(entry["position_id"])
            source = corpus[position_id]
            previous = stage2[position_id]
            if source["state_sha256"] != entry["state_sha256"]:
                raise ValueError(f"state hash mismatch for {position_id}")
            rules = source["state"].get("rules", {})
            state = prepare_bga_state(
                source["state"],
                harmony=bool(rules.get("harmony", True)),
                middle=bool(rules.get("middle_kingdom", True)),
            )
            rust_state = _rust_state_from_python(state)
            action_indices = [int(encode_action(action, state)) for action in state.legal_actions()]
            pick_ids = sorted(
                {_pick_key(state, action_idx) for action_idx in action_indices},
                key=lambda value: (-1 if value is None else int(value)),
            )
            if len(pick_ids) < 2:
                raise ValueError(f"Stage-3 root {position_id} has fewer than two pick groups")

            reused = reusable_ordinary.get(position_id)
            reused_searches: list[dict[str, Any]] | None = None
            if reused is not None:
                if reused.get("state_sha256") != entry["state_sha256"]:
                    raise ValueError(f"reused ordinary state hash mismatch for {position_id}")
                reused_searches = list(reused.get("ordinary_30000_searches", []))
                expected_seeds = [
                    int(search["seed"]) for search in previous["stage2_searches"]
                ]
                if len(reused_searches) != len(expected_seeds) or [
                    int(search["seed"]) for search in reused_searches
                ] != expected_seeds:
                    raise ValueError(f"reused ordinary search seeds mismatch for {position_id}")
                reused_ordinary_positions += 1
            else:
                rerun_ordinary_positions += 1

            ordinary: list[dict[str, Any]] = []
            matched_by_repeat: list[list[dict[str, Any]]] = []
            comparisons: list[dict[str, Any]] = []
            regrets: list[float] = []
            best_picks: list[int | None] = []
            for repeat, stage2_search in enumerate(previous["stage2_searches"]):
                seed = int(stage2_search["seed"])
                if reused_searches is not None:
                    deep_search = reused_searches[repeat]
                else:
                    search_started = time.perf_counter()
                    children, root_value_p0 = kr.advisor_open_loop_search(
                        rust_state,
                        evaluator,
                        int(ordinary_sims),
                        seed=seed,
                        **kwargs,
                    )
                    deep_search = aggregate_search(
                        state,
                        children,
                        root_value_p0=float(root_value_p0),
                        elapsed_seconds=time.perf_counter() - search_started,
                        seed=seed,
                    )
                ordinary.append(deep_search)

                forced_groups: list[dict[str, Any]] = []
                for pick_id in pick_ids:
                    allowed = [
                        action_idx
                        for action_idx in action_indices
                        if _pick_key(state, action_idx) == pick_id
                    ]
                    forced_started = time.perf_counter()
                    forced_children, forced_root_p0 = kr.advisor_open_loop_search(
                        rust_state,
                        evaluator,
                        int(matched_sims),
                        seed=seed,
                        root_allowed_actions=allowed,
                        **kwargs,
                    )
                    forced = aggregate_restricted_search(
                        state,
                        forced_children,
                        allowed_actions=allowed,
                        root_value_p0=float(forced_root_p0),
                        elapsed_seconds=time.perf_counter() - forced_started,
                        seed=seed,
                        expected_visits=int(matched_sims),
                    )
                    forced_groups.append({"pick_domino_id": pick_id, **forced})
                forced_groups.sort(
                    key=lambda group: (
                        -float(group["root_value_actor"]),
                        -1 if group["pick_domino_id"] is None else int(group["pick_domino_id"]),
                    )
                )
                matched_by_repeat.append(forced_groups)
                best_pick = forced_groups[0]["pick_domino_id"]
                best_value = float(forced_groups[0]["root_value_actor"])
                prior_pick = stage2_search["selected_pick_domino_id"]
                prior_group = next(
                    group for group in forced_groups if group["pick_domino_id"] == prior_pick
                )
                regret = max(0.0, best_value - float(prior_group["root_value_actor"]))
                best_picks.append(best_pick)
                regrets.append(regret)
                comparisons.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "stage2_4800_pick_domino_id": prior_pick,
                        "ordinary_30000_pick_domino_id": deep_search["selected_pick_domino_id"],
                        "matched_teacher_pick_domino_id": best_pick,
                        "pick_changed_4800_to_30000": prior_pick
                        != deep_search["selected_pick_domino_id"],
                        "pick_changed_4800_to_matched_teacher": prior_pick != best_pick,
                        "matched_pick_regret": regret,
                    }
                )

            row = {
                "schema": SCHEMA,
                "position_id": position_id,
                "table_id": entry["table_id"],
                "source_decision_index": entry["source_decision_index"],
                "split": split,
                "deck_count": entry["deck_count"],
                "phase": entry["phase"],
                "state_sha256": entry["state_sha256"],
                "cohort_reasons": entry["reasons"],
                "stage2_4800_searches": previous["stage2_searches"],
                "ordinary_30000_searches": ordinary,
                "matched_10000_pick_groups_by_repeat": matched_by_repeat,
                "matched_best_pick_by_repeat": best_picks,
                "matched_pick_regret_by_repeat": regrets,
                "paired_comparisons": comparisons,
            }
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            rows.append(row)
            elapsed = time.perf_counter() - started
            print(
                f"  {index + 1}/{len(entries)} roots ({elapsed:.1f}s, "
                f"{(index + 1) / elapsed:.3f} roots/s)",
                flush=True,
            )

    summary = summarize_stage3(rows)
    summary.update(
        {
            "schema": SUMMARY_SCHEMA,
            "ordinary_sims": int(ordinary_sims),
            "ordinary_repeats": 2,
            "matched_sims_per_pick_group": int(matched_sims),
            "matched_repeats": 2,
            "ordinary_reuse_source": (
                str(reuse_ordinary_path) if reuse_ordinary_path is not None else None
            ),
            "ordinary_reuse_source_sha256": (
                _sha256(reuse_ordinary_path) if reuse_ordinary_path is not None else None
            ),
            "ordinary_reused_positions": reused_ordinary_positions,
            "ordinary_rerun_positions": rerun_ordinary_positions,
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_config": config,
            "corpus": str(corpus_path),
            "corpus_sha256": _sha256(corpus_path),
            "stage2": str(stage2_path),
            "stage2_sha256": _sha256(stage2_path),
            "cohort": str(cohort_path),
            "cohort_sha256": _sha256(cohort_path),
            "output": str(output_path),
            "output_sha256": _sha256(output_path),
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ordinary-sims", type=int, default=30_000)
    parser.add_argument("--matched-sims", type=int, default=10_000)
    parser.add_argument("--reuse-ordinary-from", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()
    if args.summarize_existing:
        rows = _read_jsonl(args.output)
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        summary.update(summarize_stage3(rows))
        summary["output_sha256"] = _sha256(args.output)
        args.summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    _rows, summary = run_stage3(
        corpus_path=args.corpus,
        stage2_path=args.stage2,
        cohort_path=args.cohort,
        checkpoint=args.checkpoint,
        output_path=args.output,
        summary_path=args.summary,
        device=args.device,
        ordinary_sims=args.ordinary_sims,
        matched_sims=args.matched_sims,
        limit=args.limit,
        reuse_ordinary_path=args.reuse_ordinary_from,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
