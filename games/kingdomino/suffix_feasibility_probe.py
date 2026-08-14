"""Probe exact fixed-sequence suffix size from actual BGA board prefixes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import kingdomino_rust

from games.kingdomino.placement_headroom_audit import (
    DEFAULT_MANIFEST,
    FEASIBILITY_PROBE_V1,
    _board_is_reliable,
    _claim_domino,
    _load_manifest,
    _read_jsonl,
    _resolve_game_path,
    audit_game_completeness,
)
from games.kingdomino.web_app import state_from_debug_json


def run_probe(
    manifest_path: Path,
    output_path: Path,
    table_ids: list[str],
    player: int,
    start_placements: list[int],
    state_limit: int,
) -> dict:
    manifest = _load_manifest(manifest_path)
    games = {str(game["table_id"]): game for game in manifest["games"]}
    cases = []
    for table_id in table_ids:
        game = games[table_id]
        path = _resolve_game_path(manifest_path, game["path"])
        row = next(
            value
            for value in audit_game_completeness(path, table_id, game["split"])
            if value.player == player
        )
        sequence = row.ordered_domino_ids
        records = _read_jsonl(path)
        states_by_start = {}
        for record in records:
            if record.get("kind") != "decision":
                continue
            state_json = record.get("state", {})
            if state_json.get("current_actor") != player or not _board_is_reliable(
                state_json, player
            ):
                continue
            try:
                current = _claim_domino(state_json, player)
                start = sequence.index(current)
            except ValueError:
                continue
            states_by_start.setdefault(start, state_json)

        probes = []
        for requested_start in start_placements:
            available = [value for value in states_by_start if value >= requested_start]
            if not available:
                probes.append({"requested_start": requested_start, "status": "unavailable"})
                continue
            start = min(available)
            state = state_from_debug_json(states_by_start[start])
            board = state.boards[player]
            castle_x, castle_y = board.castle_pos
            rust_board = kingdomino_rust.RustBoard.from_flat_arrays(
                board.terrain.ravel().tolist(),
                board.crowns.ravel().tolist(),
                castle_x,
                castle_y,
            )
            suffix = sequence[start:]
            completed, raw_stats, elapsed = kingdomino_rust.fixed_sequence_exact_state_counts(
                rust_board, suffix, state_limit
            )
            stats = [
                {
                    "suffix_layer": int(item[0]),
                    "domino_id": int(item[1]),
                    "incoming_states": int(item[2]),
                    "expanded_actions": int(item[3]),
                    "unique_states": int(item[4]),
                }
                for item in raw_stats
            ]
            probes.append(
                {
                    "requested_start": requested_start,
                    "actual_start": start,
                    "remaining_tiles": len(suffix),
                    "completed_exactly": bool(completed),
                    "peak_unique_states": max(item["unique_states"] for item in stats),
                    "final_unique_states": stats[-1]["unique_states"],
                    "elapsed_seconds": float(elapsed),
                    "layers": stats,
                }
            )
        cases.append({"table_id": table_id, "player": player, "probes": probes})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"state_limit": state_limit, "cases": cases}, indent=2) + "\n",
            encoding="utf-8",
        )
    report = {
        "schema": "kingdomino-exact-suffix-feasibility/v1",
        "manifest": str(manifest_path),
        "state_limit": state_limit,
        "requested_starts": start_placements,
        "cases": cases,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/kingdomino/placement_audit/exact_suffix_feasibility_v1.json"),
    )
    parser.add_argument("--table-ids", nargs="*", default=list(FEASIBILITY_PROBE_V1))
    parser.add_argument("--player", type=int, choices=(0, 1), default=1)
    parser.add_argument("--starts", default="4,6,8,10,12,14,16,18,20")
    parser.add_argument("--state-limit", type=int, default=1_000_000)
    args = parser.parse_args()
    starts = [int(value) for value in args.starts.split(",") if value.strip()]
    report = run_probe(
        args.manifest,
        args.output,
        args.table_ids,
        args.player,
        starts,
        args.state_limit,
    )
    compact = {
        case["table_id"]: [
            {
                "start": probe.get("actual_start"),
                "remaining": probe.get("remaining_tiles"),
                "exact": probe.get("completed_exactly"),
                "peak": probe.get("peak_unique_states"),
                "seconds": probe.get("elapsed_seconds"),
            }
            for probe in case["probes"]
        ]
        for case in report["cases"]
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
