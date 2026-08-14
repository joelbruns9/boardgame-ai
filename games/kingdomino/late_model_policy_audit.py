"""Pair current-best raw-policy placement regret with exact late-game values."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any

from games.kingdomino.action_codec import encode_action
from games.kingdomino.board import Placement
from games.kingdomino.denial_search import AZBatchEvaluator, load_checkpoint_network
from games.kingdomino.dominoes import DOMINOES
from games.kingdomino.game import GameConfig, TurnAction
from games.kingdomino.placement_headroom_audit import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    _load_manifest,
    _read_jsonl,
    _resolve_game_path,
    _placement_layer,
    audit_game_completeness,
    reconstruct_game,
)
from games.kingdomino.web_app import state_from_debug_json


DEFAULT_CHECKPOINT = Path("runs/kingdomino/best_checkpoint/current_best.pt")


def prepare_bga_state(state_json: dict[str, Any], *, harmony: bool, middle: bool):
    """Import a BGA state and restore inferred rules/discard bookkeeping."""
    state = state_from_debug_json(state_json)
    state.config = replace(state.config, harmony=harmony, middle_kingdom=middle)
    layer = _placement_layer(state_json)
    if layer is not None:
        prefix = state.pending_claims[: state.actor_index]
        for owner in (0, 1):
            completed = 2 * layer + sum(claim.player == owner for claim in prefix)
            placed = (len(state.boards[owner].occupied_cells()) - 1) // 2
            state.discards[owner] = max(0, completed - placed)
    return state


def _board_key_after(state, player: int, domino_id: int, placement) -> tuple:
    board = state.boards[player].copy()
    if placement is not None:
        board.place(DOMINOES[domino_id], placement)
    return tuple(
        (x, y, int(board.terrain[y, x]), int(board.crowns[y, x]))
        for x, y in board.occupied_cells()
    )


def _placement_from_json(value: dict[str, Any] | None):
    if value is None:
        return None
    return Placement(
        int(value["x1"]),
        int(value["y1"]),
        int(value["x2"]),
        int(value["y2"]),
        bool(value["flipped"]),
    )


def _paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_placement: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_game[row["table_id"]].append(row)
        by_placement[int(row["placement_number"])].append(row)

    game_values = []
    for table_id, values in sorted(by_game.items()):
        human = statistics.fmean(value["human_regret"] for value in values)
        model = statistics.fmean(value["raw_policy_regret"] for value in values)
        game_values.append((table_id, human, model, model - human))
    rng = random.Random(0x4B44524157504F4C)
    boot = sorted(
        statistics.fmean(rng.choice(game_values)[3] for _ in game_values)
        for _ in range(10_000)
    ) if game_values else []

    def percentile(probability: float) -> float | None:
        return None if not boot else boot[round(probability * (len(boot) - 1))]

    return {
        "paired_decisions": len(rows),
        "paired_games": len(by_game),
        "human_game_weighted_mean_regret": (
            statistics.fmean(value[1] for value in game_values) if game_values else None
        ),
        "raw_policy_game_weighted_mean_regret": (
            statistics.fmean(value[2] for value in game_values) if game_values else None
        ),
        "raw_minus_human_game_weighted_mean_regret": (
            statistics.fmean(value[3] for value in game_values) if game_values else None
        ),
        "raw_minus_human_game_clustered_bootstrap_95": [
            percentile(0.025),
            percentile(0.975),
        ],
        "human_zero_regret_fraction": (
            sum(row["human_regret"] == 0 for row in rows) / len(rows) if rows else None
        ),
        "raw_policy_zero_regret_fraction": (
            sum(row["raw_policy_regret"] == 0 for row in rows) / len(rows) if rows else None
        ),
        "by_placement": {
            str(number): {
                "decisions": len(values),
                "human_mean_regret": statistics.fmean(v["human_regret"] for v in values),
                "raw_policy_mean_regret": statistics.fmean(
                    v["raw_policy_regret"] for v in values
                ),
                "raw_minus_human": statistics.fmean(
                    v["raw_policy_regret"] - v["human_regret"] for v in values
                ),
            }
            for number, values in sorted(by_placement.items())
        },
        "by_game": {
            table_id: {
                "decisions": len(by_game[table_id]),
                "human_mean_regret": human,
                "raw_policy_mean_regret": model,
                "raw_minus_human": difference,
            }
            for table_id, human, model, difference in game_values
        },
    }


def run_audit(
    *,
    manifest_path: Path,
    exact_rows_path: Path,
    output_dir: Path,
    checkpoint: Path,
    split: str,
    player: int,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    net, config = load_checkpoint_network(checkpoint, device)
    evaluator = AZBatchEvaluator(net, device=device, batch_size=256)
    exact_rows = [json.loads(line) for line in exact_rows_path.read_text(encoding="utf-8").splitlines()]
    exact_by_key = {
        (row["table_id"], int(row["source_decision_index"]), int(row["player"])): row
        for row in exact_rows
    }
    manifest = _load_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    skipped: defaultdict[str, int] = defaultdict(int)

    for game in manifest["games"]:
        if game["split"] != split:
            continue
        table_id = str(game["table_id"])
        path = _resolve_game_path(manifest_path, game["path"])
        completeness = next(
            row
            for row in audit_game_completeness(path, table_id, split)
            if row.player == player
        )
        if not completeness.whole_game_score_eligible:
            continue
        rule = completeness.scoring_rule_candidates[0]
        harmony = "harmony=True" in rule
        middle = "middle_kingdom=True" in rule
        records = _read_jsonl(path)
        decisions = [record for record in records if record.get("kind") == "decision"]
        reconstruction = {
            row.source_decision_index: row
            for row in reconstruct_game(path, table_id, split)
            if row.player == player
        }
        for key, exact in exact_by_key.items():
            if key[0] != table_id or key[2] != player:
                continue
            source_index = key[1]
            reconstructed = reconstruction.get(source_index)
            if reconstructed is None:
                skipped["missing_reconstruction"] += 1
                continue
            state = prepare_bga_state(
                decisions[source_index]["state"],
                harmony=harmony,
                middle=middle,
            )
            fixed_pick = reconstructed.next_pick_domino_id
            candidates = [
                action
                for action in state.legal_actions()
                if isinstance(action, TurnAction) and action.pick_domino_id == fixed_pick
            ]
            if not candidates:
                skipped["no_actions_with_fixed_pick"] += 1
                continue
            policy = evaluator.policy(state)
            selected = max(
                candidates,
                key=lambda action: (
                    policy[int(encode_action(action, state))],
                    -int(encode_action(action, state)),
                ),
            )

            action_scores: dict[tuple, int] = {}
            for action_value in exact["action_values"]:
                placement = _placement_from_json(action_value["placement"])
                board_key = _board_key_after(
                    state, player, int(exact["domino_id"]), placement
                )
                score = int(action_value["best_final_score"])
                if board_key in action_scores and action_scores[board_key] != score:
                    raise AssertionError("identical action board has conflicting exact values")
                action_scores[board_key] = score
            selected_key = _board_key_after(
                state, player, int(exact["domino_id"]), selected.placement
            )
            if selected_key not in action_scores:
                skipped["selected_action_missing_exact_value"] += 1
                continue
            selected_score = action_scores[selected_key]
            paired = dict(exact)
            paired.pop("action_values", None)
            paired.update(
                {
                    "fixed_next_pick_domino_id": fixed_pick,
                    "raw_policy_action_idx": int(encode_action(selected, state)),
                    "raw_policy_probability": float(
                        policy[int(encode_action(selected, state))]
                    ),
                    "raw_policy_placement": (
                        None
                        if selected.placement is None
                        else {
                            "x1": selected.placement.x1,
                            "y1": selected.placement.y1,
                            "x2": selected.placement.x2,
                            "y2": selected.placement.y2,
                            "flipped": selected.placement.flipped,
                        }
                    ),
                    "raw_policy_continuation_score": selected_score,
                    "raw_policy_regret": int(exact["best_continuation_score"]) - selected_score,
                    "raw_minus_human_regret": (
                        int(exact["best_continuation_score"]) - selected_score
                        - int(exact["human_regret"])
                    ),
                }
            )
            rows.append(paired)

    summary = _paired_summary(rows)
    summary.update(
        {
            "schema": "kingdomino-late-placement-raw-policy/v1",
            "manifest": str(manifest_path),
            "exact_rows": str(exact_rows_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_config": config,
            "split": split,
            "player": player,
            "device": device,
            "skipped": dict(sorted(skipped.items())),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"late_raw_policy_{split}_p{player}.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / f"late_raw_policy_summary_{split}_p{player}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--exact-rows", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("development", "confirmation"), default="development")
    parser.add_argument("--player", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    exact_rows = args.exact_rows or (
        args.output_dir / f"late_human_regret_{args.split}_p{args.player}.jsonl"
    )
    _rows, summary = run_audit(
        manifest_path=args.manifest,
        exact_rows_path=exact_rows,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        split=args.split,
        player=args.player,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
