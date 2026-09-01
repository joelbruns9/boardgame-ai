"""Frozen Stage-2 rerun for the Kingdomino BGA deep-target audit.

Stage 2 compares each selected root's two 800-simulation searches with two
4,800-simulation searches using the same seeds.  The confirmation split is
excluded.  Pick groups that still receive zero visits at 4,800 get separate,
restricted searches so absence of visits is not treated as evidence of low Q.

Freeze the cohort before running it::

    python -m games.kingdomino.deep_target_stage2 --freeze-only

Then run the immutable manifest::

    python -m games.kingdomino.deep_target_stage2 --device cuda
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import kingdomino_rust as kr

from games.kingdomino.action_codec import encode_action
from games.kingdomino.bga_reanalysis_corpus import DEFAULT_CORPUS
from games.kingdomino.deep_target_screen import (
    DEFAULT_CHECKPOINT,
    _action_json,
    _pick_key,
    _read_jsonl,
    aggregate_search,
)
from games.kingdomino.denial_search import load_checkpoint_network
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.game import GameState
from games.kingdomino.late_model_policy_audit import prepare_bga_state
from games.kingdomino.self_play import make_rust_evaluator


COHORT_SCHEMA = "kingdomino-deep-target-stage2-cohort/v1"
RESULT_SCHEMA = "kingdomino-deep-target-stage2/v1"
SUMMARY_SCHEMA = "kingdomino-deep-target-stage2-summary/v1"
DEFAULT_STAGE1 = Path(
    "runs/kingdomino/placement_audit/deep_target_screen_development_s800_r2.jsonl"
)
DEFAULT_STAGE1_SUMMARY = Path(
    "runs/kingdomino/placement_audit/deep_target_screen_summary_development_s800_r2.json"
)
DEFAULT_COHORT = Path(
    "runs/kingdomino/placement_audit/deep_target_stage2_cohort_v1.json"
)
DEFAULT_OUTPUT = Path(
    "runs/kingdomino/placement_audit/deep_target_stage2_development_s4800_r2.jsonl"
)
DEFAULT_SUMMARY = Path(
    "runs/kingdomino/placement_audit/deep_target_stage2_summary_development_s4800_r2.json"
)


@dataclass(frozen=True)
class CohortRules:
    root_abs_max: float = 0.4
    top2_q_gap_max: float = 0.03
    starvation_per_deck_count: int = 1
    starvation_salt: str = "kingdomino-stage2-starvation-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean_root(row: dict[str, Any]) -> float:
    return statistics.fmean(float(search["root_value_actor"]) for search in row["searches"])


def _close_in_both(row: dict[str, Any], threshold: float) -> bool:
    gaps = [search.get("top2_pick_q_gap") for search in row["searches"]]
    return len(gaps) >= 2 and all(gap is not None and float(gap) <= threshold for gap in gaps)


def _sample_rank(position_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{position_id}".encode("utf-8")).hexdigest()


def select_cohort(
    stage1_rows: list[dict[str, Any]],
    rules: CohortRules = CohortRules(),
    *,
    split: str = "development",
) -> list[dict[str, Any]]:
    """Select the frozen Stage-2 cohort without consulting Stage-2 outcomes."""
    selected: dict[str, set[str]] = defaultdict(set)
    by_id = {str(row["position_id"]): row for row in stage1_rows}
    for row in stage1_rows:
        if row.get("split") != split:
            continue
        position_id = str(row["position_id"])
        live = abs(_mean_root(row)) <= rules.root_abs_max
        if live and _close_in_both(row, rules.top2_q_gap_max):
            selected[position_id].add("live_close_in_both")
        if bool(row["screen_flags"].get("seed_pick_disagreement")):
            selected[position_id].add("stage1_pick_unstable")
        if bool(row["screen_flags"].get("random_control")):
            selected[position_id].add("stage1_easy_control")

    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in stage1_rows:
        position_id = str(row["position_id"])
        if row.get("split") != split or position_id in selected:
            continue
        if not bool(row["screen_flags"].get("pick_group_starved")):
            continue
        if abs(_mean_root(row)) > rules.root_abs_max:
            continue
        candidates[int(row["deck_count"])].append(row)
    for deck_count, rows in candidates.items():
        ranked = sorted(
            rows,
            key=lambda row: (
                _sample_rank(str(row["position_id"]), rules.starvation_salt),
                str(row["position_id"]),
            ),
        )
        for row in ranked[: rules.starvation_per_deck_count]:
            selected[str(row["position_id"])].add("starvation_probe")

    entries: list[dict[str, Any]] = []
    for position_id in sorted(selected):
        row = by_id[position_id]
        entries.append(
            {
                "position_id": position_id,
                "state_sha256": row["state_sha256"],
                "table_id": row["table_id"],
                "source_decision_index": int(row["source_decision_index"]),
                "deck_count": int(row["deck_count"]),
                "phase": row["phase"],
                "stage1_mean_root_q_actor": _mean_root(row),
                "stage1_top2_pick_q_gaps": [
                    search.get("top2_pick_q_gap") for search in row["searches"]
                ],
                "reasons": sorted(selected[position_id]),
            }
        )
    return entries


def freeze_cohort(
    *,
    stage1_path: Path,
    stage1_summary_path: Path,
    corpus_path: Path,
    output_path: Path,
    rules: CohortRules = CohortRules(),
    split: str = "development",
) -> dict[str, Any]:
    rows = _read_jsonl(stage1_path)
    entries = select_cohort(rows, rules, split=split)
    reasons: Counter[str] = Counter(
        reason for entry in entries for reason in entry["reasons"]
    )
    stage1_summary = json.loads(stage1_summary_path.read_text(encoding="utf-8"))
    if stage1_summary.get("split") != split:
        raise ValueError(
            f"Stage-1 summary split {stage1_summary.get('split')!r} does not match {split!r}"
        )
    manifest = {
        "schema": COHORT_SCHEMA,
        "split": split,
        "selection_frozen_before_stage2": True,
        "rules": {
            "primary": {
                "mean_root_q_actor_abs_max": rules.root_abs_max,
                "top2_pick_q_gap_max_in_each_repeat": rules.top2_q_gap_max,
            },
            "include_all_stage1_pick_disagreements": True,
            "include_all_stage1_easy_controls": True,
            "starvation_probe": {
                "eligible_mean_root_q_actor_abs_max": rules.root_abs_max,
                "sample_per_deck_count": rules.starvation_per_deck_count,
                "deterministic_salt": rules.starvation_salt,
                "only_outside_core_union": True,
            },
            "stage2_sims": 4800,
            "stage2_repeats": 2,
            "reuse_stage1_seeds": True,
            "forced_probe_sims": 800,
            "forced_probe_repeats": 2,
            "forced_probe_trigger": "pick group has zero visits in either 4800 repeat",
        },
        "positions": len(entries),
        "reason_counts": dict(sorted(reasons.items())),
        "confirmation_positions": len(entries) if split == "confirmation" else 0,
        "stage1": str(stage1_path),
        "stage1_sha256": _sha256(stage1_path),
        "stage1_summary": str(stage1_summary_path),
        "stage1_summary_sha256": _sha256(stage1_summary_path),
        "corpus": str(corpus_path),
        "corpus_sha256": _sha256(corpus_path),
        "checkpoint": stage1_summary["checkpoint"],
        "checkpoint_sha256": stage1_summary["checkpoint_sha256"],
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _forced_seed(position_id: str, repeat: int) -> int:
    digest = hashlib.blake2b(
        f"{position_id}:stage2-forced:{repeat}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def aggregate_restricted_search(
    state: GameState,
    children: list[tuple[int, int, float, float]],
    *,
    allowed_actions: list[int],
    root_value_p0: float,
    elapsed_seconds: float,
    seed: int,
    expected_visits: int | None = None,
) -> dict[str, Any]:
    """Aggregate one forced-pick search, validating the restricted action set."""
    allowed = set(map(int, allowed_actions))
    returned = {int(child[0]) for child in children}
    if returned != allowed:
        raise ValueError(
            f"restricted action mismatch: missing={sorted(allowed - returned)}, "
            f"extra={sorted(returned - allowed)}"
        )
    restricted = children
    actor_sign = 1.0 if int(state.current_actor) == 0 else -1.0
    total_visits = sum(int(child[1]) for child in restricted)
    if expected_visits is not None and total_visits != expected_visits:
        raise ValueError(
            f"restricted visit mismatch: expected={expected_visits}, "
            f"actual={total_visits}"
        )

    def rank(child: tuple[int, int, float, float]) -> tuple[float, float, float, int]:
        action_idx, visits, value_sum_p0, prior = child
        q_actor = (
            actor_sign * float(value_sum_p0) / int(visits)
            if int(visits)
            else float("-inf")
        )
        return float(visits), q_actor, float(prior), -int(action_idx)

    chosen = max(restricted, key=rank)
    action_idx, visits, value_sum_p0, prior = chosen
    return {
        "seed": int(seed),
        "elapsed_seconds": float(elapsed_seconds),
        "root_value_actor": actor_sign * float(root_value_p0),
        "root_total_visits": total_visits,
        "allowed_action_count": len(allowed),
        "selected_action": {
            **_action_json(state, int(action_idx)),
            "visits": int(visits),
            "q_actor": (
                actor_sign * float(value_sum_p0) / int(visits) if int(visits) else None
            ),
            "prior": float(prior),
        },
    }


def _search_kwargs(config: dict[str, Any], margin_gain: float, alpha: float) -> dict[str, Any]:
    return {
        "dirichlet_eps": 0.0,
        "fpu": float(config.get("fpu", -0.2)),
        "cpuct": float(config.get("c_puct", 1.5)),
        "leaf_batch": max(1, int(config.get("leaf_batch", 6))),
        "virtual_loss": int(config.get("virtual_loss", 1)),
        "score_scale": float(config.get("score_scale", 160.0)),
        "margin_gain": margin_gain,
        "alpha": alpha,
    }


def _paired_comparison(stage1: dict[str, Any], stage2: dict[str, Any]) -> dict[str, bool]:
    return {
        "action_changed": int(stage1["selected_action"]["action_idx"])
        != int(stage2["selected_action"]["action_idx"]),
        "pick_changed": stage1["selected_pick_domino_id"]
        != stage2["selected_pick_domino_id"],
    }


def run_stage2(
    *,
    corpus_path: Path,
    stage1_path: Path,
    cohort_path: Path,
    checkpoint: Path,
    output_path: Path,
    summary_path: Path,
    device: str,
    sims: int = 4800,
    forced_sims: int = 800,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(cohort_path.read_text(encoding="utf-8"))
    split = manifest.get("split")
    if manifest.get("schema") != COHORT_SCHEMA or split not in {
        "development", "confirmation"
    }:
        raise ValueError(
            "Stage-2 cohort must be a frozen development or confirmation manifest"
        )
    if _sha256(stage1_path) != manifest["stage1_sha256"]:
        raise ValueError("Stage-1 artifact hash does not match frozen cohort")
    if _sha256(corpus_path) != manifest["corpus_sha256"]:
        raise ValueError("corpus hash does not match frozen cohort")
    if _sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint hash does not match frozen cohort")

    stage1_by_id = {str(row["position_id"]): row for row in _read_jsonl(stage1_path)}
    corpus_by_id = {str(row["position_id"]): row for row in _read_jsonl(corpus_path)}
    entries = list(manifest["entries"])
    if limit is not None:
        entries = entries[: max(0, int(limit))]
    if not entries:
        raise ValueError("empty Stage-2 cohort")

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, entry in enumerate(entries):
            position_id = str(entry["position_id"])
            position = corpus_by_id[position_id]
            stage1 = stage1_by_id[position_id]
            if position["state_sha256"] != entry["state_sha256"]:
                raise ValueError(f"state hash mismatch for {position_id}")
            rules = position["state"].get("rules", {})
            state = prepare_bga_state(
                position["state"],
                harmony=bool(rules.get("harmony", True)),
                middle=bool(rules.get("middle_kingdom", True)),
            )
            rust_state = _rust_state_from_python(state)
            stage2_searches: list[dict[str, Any]] = []
            comparisons: list[dict[str, bool]] = []
            for repeat, stage1_search in enumerate(stage1["searches"]):
                seed = int(stage1_search["seed"])
                search_started = time.perf_counter()
                children, root_value_p0 = kr.advisor_open_loop_search(
                    rust_state,
                    evaluator,
                    int(sims),
                    seed=seed,
                    **kwargs,
                )
                search = aggregate_search(
                    state,
                    children,
                    root_value_p0=float(root_value_p0),
                    elapsed_seconds=time.perf_counter() - search_started,
                    seed=seed,
                )
                stage2_searches.append(search)
                comparisons.append(_paired_comparison(stage1_search, search))

            starved_pick_ids = sorted(
                {
                    int(group["pick_domino_id"])
                    for search in stage2_searches
                    for group in search["pick_groups"]
                    if group["pick_domino_id"] is not None and int(group["visits"]) == 0
                }
            )
            forced_probes: list[dict[str, Any]] = []
            for pick_domino_id in starved_pick_ids:
                allowed = sorted(
                    {
                        int(encode_action(action, state))
                        for action in state.legal_actions()
                        if _pick_key(state, int(encode_action(action, state))) == pick_domino_id
                    }
                )
                probe_searches: list[dict[str, Any]] = []
                for repeat in range(2):
                    seed = _forced_seed(position_id, repeat)
                    search_started = time.perf_counter()
                    children, root_value_p0 = kr.advisor_open_loop_search(
                        rust_state,
                        evaluator,
                        int(forced_sims),
                        seed=seed,
                        root_allowed_actions=allowed,
                        **kwargs,
                    )
                    probe_searches.append(
                        aggregate_restricted_search(
                            state,
                            children,
                            allowed_actions=allowed,
                            root_value_p0=float(root_value_p0),
                            elapsed_seconds=time.perf_counter() - search_started,
                            seed=seed,
                            expected_visits=int(forced_sims),
                        )
                    )
                forced_probes.append(
                    {"pick_domino_id": pick_domino_id, "searches": probe_searches}
                )

            row = {
                "schema": RESULT_SCHEMA,
                "position_id": position_id,
                "table_id": entry["table_id"],
                "source_decision_index": entry["source_decision_index"],
                "split": split,
                "deck_count": entry["deck_count"],
                "phase": entry["phase"],
                "state_sha256": entry["state_sha256"],
                "cohort_reasons": entry["reasons"],
                "stage1_searches": stage1["searches"],
                "stage2_searches": stage2_searches,
                "paired_800_to_4800": comparisons,
                "forced_pick_probes": forced_probes,
            }
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            rows.append(row)
            if (index + 1) % 20 == 0 or index + 1 == len(entries):
                elapsed = time.perf_counter() - started
                print(
                    f"  {index + 1}/{len(entries)} positions "
                    f"({elapsed:.1f}s, {(index + 1) / elapsed:.2f} positions/s)",
                    flush=True,
                )

    summary = summarize_stage2(rows)
    summary.update(
        {
            "schema": SUMMARY_SCHEMA,
            "sims": int(sims),
            "repeats": 2,
            "forced_probe_sims": int(forced_sims),
            "forced_probe_repeats": 2,
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_config": config,
            "corpus": str(corpus_path),
            "corpus_sha256": _sha256(corpus_path),
            "stage1": str(stage1_path),
            "stage1_sha256": _sha256(stage1_path),
            "cohort": str(cohort_path),
            "cohort_sha256": _sha256(cohort_path),
            "output": str(output_path),
            "output_sha256": _sha256(output_path),
            "confirmation_positions_searched": (
                len(rows) if split == "confirmation" else 0
            ),
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows, summary


def summarize_stage2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: Counter[str] = Counter(
        reason for row in rows for reason in row["cohort_reasons"]
    )
    stage1_action_agree = stage1_pick_agree = 0
    stage2_action_agree = stage2_pick_agree = 0
    action_changes = pick_changes = positions_with_pick_change = 0
    stage1_unstable_stage2_resolved = consensus_pick_changed = 0
    close_both_4800 = forced_positions = forced_groups = 0
    for row in rows:
        s1 = row["stage1_searches"]
        s2 = row["stage2_searches"]
        s1_actions = {search["selected_action"]["action_idx"] for search in s1}
        s2_actions = {search["selected_action"]["action_idx"] for search in s2}
        s1_picks = {search["selected_pick_domino_id"] for search in s1}
        s2_picks = {search["selected_pick_domino_id"] for search in s2}
        stage1_action_agree += len(s1_actions) == 1
        stage2_action_agree += len(s2_actions) == 1
        stage1_pick_agree += len(s1_picks) == 1
        stage2_pick_agree += len(s2_picks) == 1
        row_action_changes = sum(c["action_changed"] for c in row["paired_800_to_4800"])
        row_pick_changes = sum(c["pick_changed"] for c in row["paired_800_to_4800"])
        action_changes += row_action_changes
        pick_changes += row_pick_changes
        positions_with_pick_change += row_pick_changes > 0
        stage1_unstable_stage2_resolved += len(s1_picks) > 1 and len(s2_picks) == 1
        consensus_pick_changed += (
            len(s1_picks) == 1 and len(s2_picks) == 1 and s1_picks != s2_picks
        )
        close_both_4800 += _close_in_both({"searches": s2}, 0.03)
        probes = row["forced_pick_probes"]
        forced_positions += bool(probes)
        forced_groups += len(probes)
    positions = len(rows)
    paired = sum(len(row["paired_800_to_4800"]) for row in rows)
    return {
        "positions": positions,
        "reason_counts": dict(sorted(reasons.items())),
        "stage1_two_seed_action_agreement": stage1_action_agree / positions,
        "stage2_two_seed_action_agreement": stage2_action_agree / positions,
        "stage1_two_seed_pick_agreement": stage1_pick_agree / positions,
        "stage2_two_seed_pick_agreement": stage2_pick_agree / positions,
        "paired_searches": paired,
        "paired_action_changes_800_to_4800": action_changes,
        "paired_action_change_fraction": action_changes / paired,
        "paired_pick_changes_800_to_4800": pick_changes,
        "paired_pick_change_fraction": pick_changes / paired,
        "positions_with_any_paired_pick_change": positions_with_pick_change,
        "stage1_pick_unstable_stage2_resolved": stage1_unstable_stage2_resolved,
        "consensus_pick_changed_800_to_4800": consensus_pick_changed,
        "top2_pick_q_gap_le_003_in_both_4800": close_both_4800,
        "forced_probe_positions": forced_positions,
        "forced_pick_groups": forced_groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--stage1", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--stage1-summary", type=Path, default=DEFAULT_STAGE1_SUMMARY)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sims", type=int, default=4800)
    parser.add_argument("--forced-sims", type=int, default=800)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument(
        "--split", choices=("development", "confirmation"), default="development"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.freeze_only or not args.cohort.exists():
        manifest = freeze_cohort(
            stage1_path=args.stage1,
            stage1_summary_path=args.stage1_summary,
            corpus_path=args.corpus,
            output_path=args.cohort,
            split=args.split,
        )
        print(json.dumps({key: manifest[key] for key in ("schema", "positions", "reason_counts")}, indent=2))
        if args.freeze_only:
            return 0
    _rows, summary = run_stage2(
        corpus_path=args.corpus,
        stage1_path=args.stage1,
        cohort_path=args.cohort,
        checkpoint=args.checkpoint,
        output_path=args.output,
        summary_path=args.summary,
        device=args.device,
        sims=args.sims,
        forced_sims=args.forced_sims,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
