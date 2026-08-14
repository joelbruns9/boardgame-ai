"""Build a reconstructable BGA position corpus for deep-target qualification.

The AlphaZero replay buffer intentionally stores encoded examples rather than
``GameState`` objects, so it cannot be searched again.  This builder joins the
screened placement-reconstruction rows back to their original BGA decision
snapshots and emits compact, canonical, reconstructable public states.

The hidden deck is stored as a sorted *unordered bag*.  Sorting is only for
stable hashing; consumers must use information-set-safe determinization and
must never treat that order as the future reveal sequence.

Example::

    python -m games.kingdomino.bga_reanalysis_corpus build --split all
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from games.kingdomino.action_codec import encode_action
from games.kingdomino.board import Placement
from games.kingdomino.game import GameState, Phase, PickAction, TurnAction
from games.kingdomino.placement_headroom_audit import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    _final_scores_by_engine,
    _load_manifest,
    _player_metadata,
    _read_jsonl,
    _resolve_game_path,
    _verify_sha256,
)
from games.kingdomino.web_app import state_from_debug_json


SCHEMA = "kingdomino-bga-reanalysis-position/v1"
SUMMARY_SCHEMA = "kingdomino-bga-reanalysis-corpus-summary/v1"
DEFAULT_RECONSTRUCTION = DEFAULT_OUTPUT_DIR / "reconstruction_decisions_all.jsonl"
DEFAULT_CORPUS = DEFAULT_OUTPUT_DIR / "bga_reanalysis_positions_v1.jsonl"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "bga_reanalysis_positions_summary_v1.json"
KEPT_STATUSES = {"reconstructed", "forced_discard", "score_verified_terminal"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _claim_json(claim: Any) -> dict[str, int]:
    return {"player": int(claim.player), "domino_id": int(claim.domino_id)}


def _canonical_state(source: dict[str, Any]) -> tuple[GameState, dict[str, Any]]:
    """Return a searchable state plus its compact canonical serialization."""
    state = state_from_debug_json(source)
    state.deck = sorted(int(domino_id) for domino_id in state.deck)
    state.history = []
    cfg = state.config
    payload: dict[str, Any] = {
        "game": "kingdomino",
        "rules": {
            "players": int(cfg.players),
            "board_size": int(cfg.board_size),
            "canvas_size": int(cfg.canvas_size),
            "harmony": bool(cfg.harmony),
            "middle_kingdom": bool(cfg.middle_kingdom),
            "mighty_duel": bool(cfg.mighty_duel),
        },
        "phase": state.phase.name,
        "current_actor": int(state.current_actor),
        "actor_index": int(state.actor_index),
        "initial_pick_count": int(state.initial_pick_count),
        "start_player": int(state.start_player),
        "current_row": [int(value) for value in state.current_row],
        "pending_claims": [_claim_json(claim) for claim in state.pending_claims],
        "next_claims": [_claim_json(claim) for claim in state.next_claims],
        "deck_count": len(state.deck),
        "boards": [],
        "debug": {
            # This is deliberately a canonical bag, not a reveal order.
            "deck": list(state.deck),
            "history": [],
        },
    }
    for board in state.boards:
        cells = [
            {
                "x": int(x),
                "y": int(y),
                "terrain_id": int(board.terrain[y, x]),
                "crowns": int(board.crowns[y, x]),
                "domino_id": int(board.domino_id[y, x]),
            }
            for x, y in sorted(board.occupied_cells(), key=lambda item: (item[1], item[0]))
        ]
        payload["boards"].append(
            {
                "canvas_size": int(board.canvas_size),
                "castle_pos": [int(board.castle_pos[0]), int(board.castle_pos[1])],
                "cells": cells,
            }
        )

    # Ensure the emitted representation is sufficient on its own.
    roundtrip = state_from_debug_json(payload)
    if _state_signature(roundtrip) != _state_signature(state):
        raise ValueError("canonical_state_roundtrip_mismatch")
    return state, payload


def _state_signature(state: GameState) -> tuple[Any, ...]:
    boards = tuple(
        tuple(
            sorted(
                (
                    int(x),
                    int(y),
                    int(board.terrain[y, x]),
                    int(board.crowns[y, x]),
                    int(board.domino_id[y, x]),
                )
                for x, y in board.occupied_cells()
            )
        )
        for board in state.boards
    )
    return (
        state.phase.name,
        int(state.current_actor),
        int(state.actor_index),
        int(state.initial_pick_count),
        int(state.start_player),
        tuple(sorted(int(value) for value in state.deck)),
        tuple(int(value) for value in state.current_row),
        tuple((int(claim.player), int(claim.domino_id)) for claim in state.pending_claims),
        tuple((int(claim.player), int(claim.domino_id)) for claim in state.next_claims),
        boards,
    )


def _placement_from_json(value: dict[str, Any] | None) -> Placement | None:
    if value is None:
        return None
    return Placement(
        x1=int(value["x1"]),
        y1=int(value["y1"]),
        x2=int(value["x2"]),
        y2=int(value["y2"]),
        flipped=bool(value["flipped"]),
    )


def _played_action(row: dict[str, Any], state: GameState) -> PickAction | TurnAction | None:
    status = str(row["status"])
    if state.phase == Phase.INITIAL_SELECTION:
        picked = row.get("next_pick_domino_id")
        return None if picked is None else PickAction(int(picked))
    if status == "score_verified_terminal" and row.get("first_action_unique") is not True:
        return None
    if status not in KEPT_STATUSES:
        return None
    return TurnAction(
        _placement_from_json(row.get("placement")),
        None if row.get("next_pick_domino_id") is None else int(row["next_pick_domino_id"]),
    )


def _placement_json(placement: Placement | None) -> dict[str, Any] | None:
    return None if placement is None else asdict(placement)


def _action_json(action: PickAction | TurnAction, state: GameState) -> dict[str, Any]:
    if isinstance(action, PickAction):
        return {
            "kind": "pick",
            "domino_id": int(action.domino_id),
            "action_idx": int(encode_action(action, state)),
        }
    return {
        "kind": "turn",
        "placement": _placement_json(action.placement),
        "pick_domino_id": (
            None if action.pick_domino_id is None else int(action.pick_domino_id)
        ),
        "action_idx": int(encode_action(action, state)),
    }


def _legal_metadata(state: GameState) -> tuple[list[int], list[int], int]:
    actions = state.legal_actions()
    indices = [int(encode_action(action, state)) for action in actions]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate_legal_action_index")
    pick_ids = sorted(
        {
            int(action.domino_id)
            if isinstance(action, PickAction)
            else int(action.pick_domino_id)
            for action in actions
            if isinstance(action, PickAction) or action.pick_domino_id is not None
        }
    )
    placements = {
        None
        if isinstance(action, PickAction) or action.placement is None
        else (
            int(action.placement.x1),
            int(action.placement.y1),
            int(action.placement.x2),
            int(action.placement.y2),
            bool(action.placement.flipped),
        )
        for action in actions
    }
    return indices, pick_ids, len(placements)


def _counter_json(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=lambda item: str(item))}


def _selected_games(manifest: dict[str, Any], split: str) -> Iterable[dict[str, Any]]:
    for game in manifest["games"]:
        if split == "all" or game["split"] == split:
            yield game


def build_corpus(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    reconstruction_path: Path = DEFAULT_RECONSTRUCTION,
    output_path: Path = DEFAULT_CORPUS,
    summary_path: Path = DEFAULT_SUMMARY,
    split: str = "all",
) -> dict[str, Any]:
    """Build the JSONL corpus and return its versioned summary."""
    if split not in {"all", "development", "confirmation"}:
        raise ValueError(f"unsupported split: {split}")
    manifest_path = Path(manifest_path)
    reconstruction_path = Path(reconstruction_path)
    output_path = Path(output_path)
    summary_path = Path(summary_path)
    manifest = _load_manifest(manifest_path)

    reconstruction_rows = _read_jsonl(reconstruction_path)
    reconstruction_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in reconstruction_rows:
        key = (str(row["table_id"]), int(row["source_decision_index"]))
        if key in reconstruction_by_key:
            raise ValueError(f"duplicate reconstruction key: {key}")
        reconstruction_by_key[key] = row

    positions: list[dict[str, Any]] = []
    counts_split: Counter[str] = Counter()
    counts_phase: Counter[str] = Counter()
    counts_deck: Counter[int] = Counter()
    counts_role: Counter[str] = Counter()
    counts_status: Counter[str] = Counter()
    played_unknown = 0
    exact_candidates = 0
    games_with_positions: set[str] = set()

    for game in _selected_games(manifest, split):
        table_id = str(game["table_id"])
        game_split = str(game["split"])
        game_path = _resolve_game_path(manifest_path, str(game["path"]))
        _verify_sha256(game_path, str(game["sha256"]))
        records = _read_jsonl(game_path)
        decisions = [record for record in records if record.get("kind") == "decision"]
        metadata = _player_metadata(records, decisions)
        final_scores = _final_scores_by_engine(records, metadata)

        for source_index, source_record in enumerate(decisions):
            key = (table_id, source_index)
            reconstruction = reconstruction_by_key.get(key)
            if reconstruction is None or reconstruction.get("status") not in KEPT_STATUSES:
                continue
            if reconstruction.get("split") != game_split:
                raise ValueError(f"split mismatch for reconstruction {key}")
            source = source_record.get("state")
            if not isinstance(source, dict):
                raise ValueError(f"missing source state for {key}")
            state, state_json = _canonical_state(source)
            actor = int(state.current_actor)
            if actor != int(reconstruction["player"]):
                raise ValueError(f"actor mismatch for {key}")
            if int(source.get("deck_count", len(state.deck))) != len(state.deck):
                raise ValueError(f"deck count mismatch for {key}")

            legal_indices, legal_pick_ids, legal_placement_count = _legal_metadata(state)
            played = _played_action(reconstruction, state)
            played_json = None
            if played is not None:
                played_json = _action_json(played, state)
                if played_json["action_idx"] not in legal_indices:
                    raise ValueError(f"reconstructed action is illegal for {key}")
            else:
                played_unknown += 1

            state_sha = _sha256_json(state_json)
            position_id = f"bga-{table_id}-d{source_index:03d}-{state_sha[:12]}"
            player_id, player_name = metadata.get(actor, (None, None))
            viewer_id = (
                None if source_record.get("viewer_id") is None
                else str(source_record.get("viewer_id"))
            )
            role = "viewer" if player_id is not None and player_id == viewer_id else "opponent"
            is_exact_candidate = state.phase == Phase.FINAL_PLACEMENT or len(state.deck) <= 4
            if is_exact_candidate:
                exact_candidates += 1

            position = {
                "schema": SCHEMA,
                "position_id": position_id,
                "split": game_split,
                "table_id": table_id,
                "source_decision_index": source_index,
                "captured_at": source_record.get("captured_at"),
                "actor": actor,
                "actor_player_id": player_id,
                "actor_name": player_name,
                "actor_role": role,
                "phase": state.phase.name,
                "deck_count": len(state.deck),
                "state_sha256": state_sha,
                "state": state_json,
                "legal": {
                    "action_count": len(legal_indices),
                    "action_indices": legal_indices,
                    "pick_domino_ids": legal_pick_ids,
                    "placement_count": legal_placement_count,
                },
                "human_action": played_json,
                "human_action_known": played_json is not None,
                "reconstruction": {
                    "status": reconstruction["status"],
                    "target_decision_index": reconstruction.get("target_decision_index"),
                    "domino_id": reconstruction.get("domino_id"),
                    "next_pick_domino_id": reconstruction.get("next_pick_domino_id"),
                    "first_action_unique": reconstruction.get("first_action_unique"),
                },
                "audit_tags": {
                    "exact_candidate": is_exact_candidate,
                    "forced_discard": reconstruction["status"] == "forced_discard",
                    "terminal_score_verified": (
                        reconstruction["status"] == "score_verified_terminal"
                    ),
                },
                "diagnostics_only": {
                    "final_scores": {
                        str(player): int(score)
                        for player, score in sorted(final_scores.items())
                    },
                    "actor_final_score": final_scores.get(actor),
                },
                "provenance": {
                    "source_path": str(game["path"]),
                    "source_sha256": str(game["sha256"]),
                    "source_schema": manifest.get("source_schema"),
                    "source_corpus_id": manifest.get("corpus_id"),
                },
            }
            positions.append(position)
            games_with_positions.add(table_id)
            counts_split[game_split] += 1
            counts_phase[state.phase.name] += 1
            counts_deck[len(state.deck)] += 1
            counts_role[role] += 1
            counts_status[str(reconstruction["status"])] += 1

    positions.sort(key=lambda row: (row["table_id"], row["source_decision_index"]))
    ids = [row["position_id"] for row in positions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate position_id")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for position in positions:
            handle.write(_canonical_json(position) + "\n")

    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "corpus_schema": SCHEMA,
        "corpus_id": "kingdomino-bga-reanalysis-positions-v1",
        "selected_split": split,
        "positions": len(positions),
        "games": len(games_with_positions),
        "counts_by_split": _counter_json(counts_split),
        "counts_by_phase": _counter_json(counts_phase),
        "counts_by_deck_count": _counter_json(counts_deck),
        "counts_by_actor_role": _counter_json(counts_role),
        "counts_by_reconstruction_status": _counter_json(counts_status),
        "human_action_known": len(positions) - played_unknown,
        "human_action_unknown": played_unknown,
        "exact_candidates": exact_candidates,
        "information_policy": {
            "hidden_deck": "unordered bag sorted only for canonicalization",
            "future_reveal_order_present": False,
            "consumer_requirement": (
                "Use information-set-safe determinization. Never consume "
                "diagnostics_only as model/search input."
            ),
        },
        "source": {
            "manifest": str(manifest_path),
            "manifest_sha256": _file_sha256(manifest_path),
            "reconstruction": str(reconstruction_path),
            "reconstruction_sha256": _file_sha256(reconstruction_path),
        },
        "output": {
            "path": str(output_path),
            "sha256": _file_sha256(output_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build the reconstructable JSONL corpus")
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--reconstruction", type=Path, default=DEFAULT_RECONSTRUCTION)
    build.add_argument("--output", type=Path, default=DEFAULT_CORPUS)
    build.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    build.add_argument(
        "--split",
        choices=("all", "development", "confirmation"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "build":
        summary = build_corpus(
            manifest_path=args.manifest,
            reconstruction_path=args.reconstruction,
            output_path=args.output,
            summary_path=args.summary,
            split=args.split,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
