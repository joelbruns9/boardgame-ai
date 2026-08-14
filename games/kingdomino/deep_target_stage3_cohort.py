"""Freeze the 30k Stage-3 cohort from completed Stage-2 artifacts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from games.kingdomino.deep_target_forced_pick_probe import (
    DEFAULT_OUTPUT as DEFAULT_FORCED,
)
from games.kingdomino.deep_target_screen import _read_jsonl
from games.kingdomino.deep_target_stage2 import DEFAULT_OUTPUT as DEFAULT_STAGE2


SCHEMA = "kingdomino-deep-target-stage3-cohort/v1"
DEFAULT_OUTPUT = Path(
    "runs/kingdomino/placement_audit/deep_target_stage3_cohort_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_stage3(
    stage2_rows: list[dict[str, Any]],
    forced_rows: list[dict[str, Any]],
    *,
    forced_q_gap_max: float = 0.03,
) -> list[dict[str, Any]]:
    reasons: dict[str, set[str]] = defaultdict(set)
    stage2_by_id = {str(row["position_id"]): row for row in stage2_rows}
    for row in stage2_rows:
        if row.get("split") != "development":
            continue
        position_id = str(row["position_id"])
        stage1_picks = {
            search["selected_pick_domino_id"] for search in row["stage1_searches"]
        }
        stage2_picks = {
            search["selected_pick_domino_id"] for search in row["stage2_searches"]
        }
        if len(stage1_picks) == 1 and len(stage2_picks) == 1 and stage1_picks != stage2_picks:
            reasons[position_id].add("stable_consensus_pick_changed")
        if len(stage2_picks) > 1:
            reasons[position_id].add("stage2_pick_unstable")
    for row in forced_rows:
        position_id = str(row["position_id"])
        if stage2_by_id[position_id].get("split") != "development":
            continue
        if any(
            bool(group["was_starved_at_4800"])
            and float(group["regret_vs_best_forced_q"]) <= forced_q_gap_max
            for group in row["pick_groups"]
        ):
            reasons[position_id].add("starved_forced_q_within_003")

    entries = []
    for position_id in sorted(reasons):
        row = stage2_by_id[position_id]
        entries.append(
            {
                "position_id": position_id,
                "table_id": row["table_id"],
                "source_decision_index": int(row["source_decision_index"]),
                "deck_count": int(row["deck_count"]),
                "phase": row["phase"],
                "state_sha256": row["state_sha256"],
                "reasons": sorted(reasons[position_id]),
            }
        )
    return entries


def freeze_stage3(
    *,
    stage2_path: Path,
    forced_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    entries = select_stage3(_read_jsonl(stage2_path), _read_jsonl(forced_path))
    reason_counts = Counter(reason for entry in entries for reason in entry["reasons"])
    manifest = {
        "schema": SCHEMA,
        "split": "development",
        "selection_frozen_before_stage3": True,
        "positions": len(entries),
        "reason_counts": dict(sorted(reason_counts.items())),
        "confirmation_positions": 0,
        "rules": {
            "include_stable_consensus_pick_changes_800_to_4800": True,
            "include_all_two_seed_pick_disagreements_at_4800": True,
            "include_starved_groups_with_matched_forced_q_regret_max": 0.03,
            "stage3_ordinary_sims": 30000,
            "stage3_ordinary_repeats": 2,
            "reuse_stage2_seeds": True,
            "matched_restricted_sims_per_pick_group": 10000,
            "matched_restricted_repeats": 2,
            "matched_restricted_scope": "every pick group at every Stage-3 root",
            "matched_restricted_common_seeds": True,
        },
        "stage2": str(stage2_path),
        "stage2_sha256": _sha256(stage2_path),
        "matched_forced_pick": str(forced_path),
        "matched_forced_pick_sha256": _sha256(forced_path),
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--forced", type=Path, default=DEFAULT_FORCED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = freeze_stage3(
        stage2_path=args.stage2,
        forced_path=args.forced,
        output_path=args.output,
    )
    print(json.dumps({key: manifest[key] for key in ("schema", "positions", "reason_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
