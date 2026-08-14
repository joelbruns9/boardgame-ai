"""Exact human placement regret on late BGA states with fixed claimed tiles."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
from typing import Any

import kingdomino_rust

from games.kingdomino.placement_headroom_audit import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    _load_manifest,
    _read_jsonl,
    _resolve_game_path,
    audit_game_completeness,
    reconstruct_game,
)
from games.kingdomino.web_app import state_from_debug_json


@dataclass(frozen=True)
class LateRegretRow:
    table_id: str
    split: str
    player: int
    player_name: str | None
    source_decision_index: int
    placement_number: int
    remaining_tiles: int
    domino_id: int
    actual_placement: dict[str, Any] | None
    legal_placement_count: int
    forced_discard: bool
    human_continuation_score: int
    best_continuation_score: int
    human_regret: int
    optimal_action_count: int
    human_action_optimal: bool
    action_values: list[dict[str, Any]]


def _placement_dict(move: tuple[int, int, int, int, bool] | None) -> dict[str, Any] | None:
    if move is None:
        return None
    return {
        "x1": int(move[0]),
        "y1": int(move[1]),
        "x2": int(move[2]),
        "y2": int(move[3]),
        "flipped": bool(move[4]),
    }


def _move_tuple(placement: dict[str, Any]) -> tuple[int, int, int, int, bool]:
    return (
        int(placement["x1"]),
        int(placement["y1"]),
        int(placement["x2"]),
        int(placement["y2"]),
        bool(placement["flipped"]),
    )


def _rust_board(state_json: dict[str, Any], player: int):
    state = state_from_debug_json(state_json)
    board = state.boards[player]
    castle_x, castle_y = board.castle_pos
    return kingdomino_rust.RustBoard.from_flat_arrays(
        board.terrain.ravel().tolist(),
        board.crowns.ravel().tolist(),
        castle_x,
        castle_y,
    )


def _exact_score(
    board,
    suffix: list[int],
    *,
    harmony: bool,
    middle_kingdom: bool,
    state_limit: int,
) -> tuple[int, int, float]:
    result = kingdomino_rust.fixed_sequence_exact_score(
        board, suffix, harmony, middle_kingdom, state_limit
    )
    peak_states = max((int(item[4]) for item in result[6]), default=1)
    return int(result[0]), peak_states, float(result[7])


def audit_decision(
    *,
    table_id: str,
    split: str,
    player: int,
    player_name: str | None,
    source_decision_index: int,
    state_json: dict[str, Any],
    sequence: list[int],
    sequence_index: int,
    actual_placement: dict[str, Any] | None,
    harmony: bool,
    middle_kingdom: bool,
    state_limit: int,
) -> LateRegretRow:
    domino_id = sequence[sequence_index]
    suffix = sequence[sequence_index + 1 :]
    board = _rust_board(state_json, player)
    halves = tuple(int(value) for value in kingdomino_rust.domino_halves(domino_id))
    legal = [tuple(move) for move in board.legal_placements(*halves)]
    action_values: list[dict[str, Any]] = []

    if legal:
        if actual_placement is None:
            raise ValueError("missing_actual_placement")
        for move in legal:
            child = board.copy()
            child.place(*halves, *move)
            score, peak, elapsed = _exact_score(
                child,
                suffix,
                harmony=harmony,
                middle_kingdom=middle_kingdom,
                state_limit=state_limit,
            )
            action_values.append(
                {
                    "placement": _placement_dict(move),
                    "best_final_score": score,
                    "peak_suffix_states": peak,
                    "elapsed_seconds": elapsed,
                }
            )
        actual_move = _move_tuple(actual_placement)
        actual_child = board.copy()
        actual_child.place(*halves, *actual_move)
        human_score, human_peak, human_elapsed = _exact_score(
            actual_child,
            suffix,
            harmony=harmony,
            middle_kingdom=middle_kingdom,
            state_limit=state_limit,
        )
        action_values.append(
            {
                "placement": actual_placement,
                "best_final_score": human_score,
                "peak_suffix_states": human_peak,
                "elapsed_seconds": human_elapsed,
                "human_action_recheck": True,
            }
        )
    else:
        if actual_placement is not None:
            raise ValueError("placement_recorded_for_forced_discard")
        human_score, peak, elapsed = _exact_score(
            board,
            suffix,
            harmony=harmony,
            middle_kingdom=middle_kingdom,
            state_limit=state_limit,
        )
        action_values.append(
            {
                "placement": None,
                "best_final_score": human_score,
                "peak_suffix_states": peak,
                "elapsed_seconds": elapsed,
            }
        )

    best_score = max(int(action["best_final_score"]) for action in action_values)
    if human_score > best_score:
        raise AssertionError("forced human continuation exceeds enumerated best action")
    optimal_count = sum(int(action["best_final_score"]) == best_score for action in action_values)
    # The explicit human recheck can duplicate an enumerated physical action.
    if legal and human_score == best_score:
        optimal_count -= 1
    return LateRegretRow(
        table_id=table_id,
        split=split,
        player=player,
        player_name=player_name,
        source_decision_index=source_decision_index,
        placement_number=sequence_index + 1,
        remaining_tiles=len(sequence) - sequence_index,
        domino_id=domino_id,
        actual_placement=actual_placement,
        legal_placement_count=len(legal),
        forced_discard=not legal,
        human_continuation_score=human_score,
        best_continuation_score=best_score,
        human_regret=best_score - human_score,
        optimal_action_count=optimal_count,
        human_action_optimal=human_score == best_score,
        action_values=action_values,
    )


def _summary(rows: list[LateRegretRow], skipped: Counter[str]) -> dict[str, Any]:
    regrets = [row.human_regret for row in rows]
    by_game: dict[str, list[int]] = defaultdict(list)
    by_placement: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        by_game[row.table_id].append(row.human_regret)
        by_placement[row.placement_number].append(row.human_regret)
    game_means = [statistics.fmean(values) for values in by_game.values()]
    bootstrap_rng = random.Random(0x4B444C415445)
    bootstrap = sorted(
        statistics.fmean(bootstrap_rng.choice(game_means) for _ in game_means)
        for _ in range(10_000)
    ) if game_means else []

    def percentile(values: list[float], probability: float) -> float | None:
        if not values:
            return None
        return values[round(probability * (len(values) - 1))]

    return {
        "audited_decisions": len(rows),
        "audited_games": len(by_game),
        "skipped": dict(sorted(skipped.items())),
        "decision_mean_regret": statistics.fmean(regrets) if regrets else None,
        "game_weighted_mean_regret": statistics.fmean(game_means) if game_means else None,
        "game_clustered_bootstrap_95": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
        "bootstrap_games": len(game_means),
        "bootstrap_replicates": len(bootstrap),
        "median_regret": statistics.median(regrets) if regrets else None,
        "zero_regret_fraction": (
            sum(value == 0 for value in regrets) / len(regrets) if regrets else None
        ),
        "forced_discards": sum(row.forced_discard for row in rows),
        "by_placement": {
            str(number): {
                "decisions": len(values),
                "mean_regret": statistics.fmean(values),
                "median_regret": statistics.median(values),
                "zero_regret_fraction": sum(value == 0 for value in values) / len(values),
                "max_regret": max(values),
                "mean_legal_placements": statistics.fmean(
                    row.legal_placement_count
                    for row in rows
                    if row.placement_number == number
                ),
            }
            for number, values in sorted(by_placement.items())
        },
        "by_game": {
            table_id: {
                "decisions": len(values),
                "mean_regret": statistics.fmean(values),
                "total_regret": sum(values),
                "max_regret": max(values),
            }
            for table_id, values in sorted(by_game.items())
        },
    }


def run_audit(
    *,
    manifest_path: Path,
    output_dir: Path,
    split: str,
    player: int,
    min_placement: int,
    state_limit: int,
    table_ids: set[str] | None = None,
) -> tuple[list[LateRegretRow], dict[str, Any]]:
    manifest = _load_manifest(manifest_path)
    games = [
        game
        for game in manifest["games"]
        if (split == "all" or game["split"] == split)
        and (table_ids is None or str(game["table_id"]) in table_ids)
    ]
    rows: list[LateRegretRow] = []
    skipped: Counter[str] = Counter()
    for game in games:
        table_id = str(game["table_id"])
        path = _resolve_game_path(manifest_path, game["path"])
        complete = next(
            row
            for row in audit_game_completeness(path, table_id, game["split"])
            if row.player == player
        )
        if not complete.whole_game_score_eligible:
            skipped["game_not_whole_score_eligible"] += 1
            continue
        rule = complete.scoring_rule_candidates[0]
        harmony = "harmony=True" in rule
        middle = "middle_kingdom=True" in rule
        records = _read_jsonl(path)
        decisions = [record for record in records if record.get("kind") == "decision"]
        reconstruction = reconstruct_game(path, table_id, game["split"])
        for reconstructed in reconstruction:
            if reconstructed.player != player or not reconstructed.kept:
                continue
            if reconstructed.domino_id not in complete.ordered_domino_ids:
                skipped["current_tile_not_in_sequence"] += 1
                continue
            sequence_index = complete.ordered_domino_ids.index(reconstructed.domino_id)
            if sequence_index + 1 < min_placement:
                continue
            if reconstructed.placement is None and reconstructed.status != "forced_discard":
                source = decisions[reconstructed.source_decision_index]["state"]
                board = _rust_board(source, player)
                halves = kingdomino_rust.domino_halves(reconstructed.domino_id)
                if board.legal_placements(*halves):
                    skipped["actual_placement_ambiguous"] += 1
                    continue
            try:
                row = audit_decision(
                    table_id=table_id,
                    split=game["split"],
                    player=player,
                    player_name=complete.player_name,
                    source_decision_index=reconstructed.source_decision_index,
                    state_json=decisions[reconstructed.source_decision_index]["state"],
                    sequence=complete.ordered_domino_ids,
                    sequence_index=sequence_index,
                    actual_placement=reconstructed.placement,
                    harmony=harmony,
                    middle_kingdom=middle,
                    state_limit=state_limit,
                )
            except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                skipped[f"audit_error:{exc}"] += 1
                continue
            rows.append(row)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"late_human_regret_{split}_p{player}.jsonl").write_text(
            "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    summary = _summary(rows, skipped)
    summary.update(
        {
            "schema": "kingdomino-late-placement-regret/v1",
            "manifest": str(manifest_path),
            "split": split,
            "player": player,
            "min_placement": min_placement,
            "state_limit": state_limit,
        }
    )
    (output_dir / f"late_human_regret_summary_{split}_p{player}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("development", "confirmation", "all"), default="development")
    parser.add_argument("--player", type=int, choices=(0, 1), default=1)
    parser.add_argument("--min-placement", type=int, default=17)
    parser.add_argument("--state-limit", type=int, default=1_000_000)
    parser.add_argument("--table-ids", nargs="*")
    args = parser.parse_args()
    _rows, summary = run_audit(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        split=args.split,
        player=args.player,
        min_placement=args.min_placement,
        state_limit=args.state_limit,
        table_ids=None if not args.table_ids else set(args.table_ids),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
