"""Paired fixed-pick open-loop search audit on exact late placement states."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any

import kingdomino_rust as kr

from games.kingdomino.action_codec import decode_action, encode_action
from games.kingdomino.denial_search import load_checkpoint_network
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.game import TurnAction
from games.kingdomino.late_model_policy_audit import (
    DEFAULT_CHECKPOINT,
    _board_key_after,
    _placement_from_json,
    prepare_bga_state,
)
from games.kingdomino.placement_headroom_audit import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    _load_manifest,
    _read_jsonl,
    _resolve_game_path,
    audit_game_completeness,
    reconstruct_game,
)
from games.kingdomino.self_play import make_rust_evaluator


def _seed(table_id: str, source_index: int, sims: int) -> int:
    digest = hashlib.blake2b(
        f"{table_id}:{source_index}:{sims}:fixed-pick".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_placement: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_game[row["table_id"]].append(row)
        by_placement[int(row["placement_number"])].append(row)
    game_values = []
    for table_id, values in sorted(by_game.items()):
        human = statistics.fmean(value["human_regret"] for value in values)
        raw = statistics.fmean(value["raw_policy_regret"] for value in values)
        search = statistics.fmean(value["search_regret"] for value in values)
        game_values.append((table_id, human, raw, search))

    rng = random.Random(0x4B44534541524348)
    human_diffs = []
    raw_diffs = []
    if game_values:
        for _ in range(10_000):
            sample = [rng.choice(game_values) for _ in game_values]
            human_diffs.append(statistics.fmean(value[3] - value[1] for value in sample))
            raw_diffs.append(statistics.fmean(value[3] - value[2] for value in sample))
        human_diffs.sort()
        raw_diffs.sort()

    def interval(values: list[float]) -> list[float | None]:
        if not values:
            return [None, None]
        return [values[round(0.025 * (len(values) - 1))], values[round(0.975 * (len(values) - 1))]]

    return {
        "paired_decisions": len(rows),
        "paired_games": len(by_game),
        "human_game_weighted_mean_regret": (
            statistics.fmean(value[1] for value in game_values) if game_values else None
        ),
        "raw_policy_game_weighted_mean_regret": (
            statistics.fmean(value[2] for value in game_values) if game_values else None
        ),
        "search_game_weighted_mean_regret": (
            statistics.fmean(value[3] for value in game_values) if game_values else None
        ),
        "search_minus_human": (
            statistics.fmean(value[3] - value[1] for value in game_values)
            if game_values else None
        ),
        "search_minus_human_game_clustered_bootstrap_95": interval(human_diffs),
        "search_minus_raw": (
            statistics.fmean(value[3] - value[2] for value in game_values)
            if game_values else None
        ),
        "search_minus_raw_game_clustered_bootstrap_95": interval(raw_diffs),
        "zero_regret_fraction": {
            name: sum(row[field] == 0 for row in rows) / len(rows) if rows else None
            for name, field in (
                ("human", "human_regret"),
                ("raw_policy", "raw_policy_regret"),
                ("search", "search_regret"),
            )
        },
        "by_placement": {
            str(number): {
                "decisions": len(values),
                "human_mean_regret": statistics.fmean(v["human_regret"] for v in values),
                "raw_policy_mean_regret": statistics.fmean(v["raw_policy_regret"] for v in values),
                "search_mean_regret": statistics.fmean(v["search_regret"] for v in values),
            }
            for number, values in sorted(by_placement.items())
        },
    }


def run_audit(
    *,
    manifest_path: Path,
    exact_rows_path: Path,
    raw_rows_path: Path,
    output_dir: Path,
    checkpoint: Path,
    split: str,
    player: int,
    device: str,
    sims: int,
    table_ids: set[str] | None = None,
    output_tag: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    net, config = load_checkpoint_network(checkpoint, device)
    evaluator = make_rust_evaluator(
        net,
        device=device,
        amp=bool(config.get("inference_amp", True)),
        margin_gain=float(config.get("margin_gain", 2.0)),
        alpha=float(config.get("alpha", 0.5)),
    )
    exact_rows = {
        (row["table_id"], int(row["source_decision_index"]), int(row["player"])): row
        for row in (
            json.loads(line)
            for line in exact_rows_path.read_text(encoding="utf-8").splitlines()
        )
    }
    raw_rows = {
        (row["table_id"], int(row["source_decision_index"]), int(row["player"])): row
        for row in (
            json.loads(line)
            for line in raw_rows_path.read_text(encoding="utf-8").splitlines()
        )
    }
    manifest = _load_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    skipped: defaultdict[str, int] = defaultdict(int)

    for game in manifest["games"]:
        if game["split"] != split:
            continue
        table_id = str(game["table_id"])
        if table_ids is not None and table_id not in table_ids:
            continue
        path = _resolve_game_path(manifest_path, game["path"])
        complete = next(
            row for row in audit_game_completeness(path, table_id, split) if row.player == player
        )
        if not complete.whole_game_score_eligible:
            continue
        rule = complete.scoring_rule_candidates[0]
        harmony = "harmony=True" in rule
        middle = "middle_kingdom=True" in rule
        records = _read_jsonl(path)
        decisions = [record for record in records if record.get("kind") == "decision"]
        reconstruction = {
            row.source_decision_index: row
            for row in reconstruct_game(path, table_id, split)
            if row.player == player
        }
        for key, raw in raw_rows.items():
            if key[0] != table_id or key[2] != player:
                continue
            exact = exact_rows[key]
            source_index = key[1]
            reconstructed = reconstruction[source_index]
            state = prepare_bga_state(
                decisions[source_index]["state"], harmony=harmony, middle=middle
            )
            fixed_pick = reconstructed.next_pick_domino_id
            candidates = [
                action
                for action in state.legal_actions()
                if isinstance(action, TurnAction) and action.pick_domino_id == fixed_pick
            ]
            allowed = sorted({int(encode_action(action, state)) for action in candidates})
            if not allowed:
                skipped["no_fixed_pick_actions"] += 1
                continue
            rust_state = _rust_state_from_python(state)
            if rust_state is None:
                skipped["rust_state_conversion_failed"] += 1
                continue
            children, _root_value0 = kr.advisor_open_loop_search(
                rust_state,
                evaluator,
                sims,
                dirichlet_eps=0.0,
                fpu=float(config.get("fpu", -0.2)),
                cpuct=float(config.get("c_puct", 1.5)),
                seed=_seed(table_id, source_index, sims),
                leaf_batch=max(1, int(config.get("leaf_batch", 6))),
                virtual_loss=int(config.get("virtual_loss", 1)),
                score_scale=float(config.get("score_scale", 160.0)),
                margin_gain=float(config.get("margin_gain", 2.0)),
                alpha=float(config.get("alpha", 0.5)),
                root_allowed_actions=allowed,
            )
            returned = {int(child[0]) for child in children}
            if returned != set(allowed):
                skipped["search_root_action_mismatch"] += 1
                continue
            if sum(int(child[1]) for child in children) != int(sims):
                skipped["search_root_visit_mismatch"] += 1
                continue
            allowed_children = children
            actor = int(state.current_actor)

            def rank(child):
                idx, visits, value_sum0, prior = child
                q0 = float(value_sum0) / int(visits) if int(visits) else float("-inf")
                q_actor = q0 if actor == 0 else -q0
                return int(visits), q_actor, float(prior), -int(idx)

            chosen = max(allowed_children, key=rank)
            chosen_idx, visits, value_sum0, prior = chosen
            action = decode_action(int(chosen_idx), state)
            action_scores: dict[tuple, int] = {}
            for value in exact["action_values"]:
                placement = _placement_from_json(value["placement"])
                board_key = _board_key_after(state, player, int(exact["domino_id"]), placement)
                action_scores[board_key] = int(value["best_final_score"])
            selected_key = _board_key_after(state, player, int(exact["domino_id"]), action.placement)
            if selected_key not in action_scores:
                skipped["search_action_missing_exact_value"] += 1
                continue
            score = action_scores[selected_key]
            q0 = float(value_sum0) / int(visits) if int(visits) else None
            out = dict(raw)
            out.update(
                {
                    "search_sims": sims,
                    "search_action_idx": int(chosen_idx),
                    "search_visits": int(visits),
                    "search_prior": float(prior),
                    "search_q_actor": None if q0 is None else (q0 if actor == 0 else -q0),
                    "search_continuation_score": score,
                    "search_regret": int(exact["best_continuation_score"]) - score,
                    "search_minus_human_regret": int(exact["best_continuation_score"]) - score - int(exact["human_regret"]),
                    "search_minus_raw_regret": int(exact["best_continuation_score"]) - score - int(raw["raw_policy_regret"]),
                }
            )
            rows.append(out)

    summary = _summary(rows)
    summary.update(
        {
            "schema": "kingdomino-late-placement-fixed-pick-search/v1",
            "manifest": str(manifest_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "split": split,
            "player": player,
            "device": device,
            "sims": sims,
            "root_constraint": "only actions with the human's fixed next pick",
            "skipped": dict(sorted(skipped.items())),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_tag}" if output_tag else ""
    (output_dir / f"late_search_{split}_p{player}_s{sims}{suffix}.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (output_dir / f"late_search_summary_{split}_p{player}_s{sims}{suffix}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("development", "confirmation"), default="development")
    parser.add_argument("--player", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sims", type=int, default=4800)
    parser.add_argument("--table-ids", nargs="*")
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    exact = args.output_dir / f"late_human_regret_{args.split}_p{args.player}.jsonl"
    raw = args.output_dir / f"late_raw_policy_{args.split}_p{args.player}.jsonl"
    _rows, summary = run_audit(
        manifest_path=args.manifest,
        exact_rows_path=exact,
        raw_rows_path=raw,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        split=args.split,
        player=args.player,
        device=args.device,
        sims=args.sims,
        table_ids=None if not args.table_ids else set(args.table_ids),
        output_tag=args.output_tag,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
