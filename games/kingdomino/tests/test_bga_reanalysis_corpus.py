from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from games.kingdomino.bga_reanalysis_corpus import SCHEMA, build_corpus
from games.kingdomino.game import GameState, Phase, PickAction
from games.kingdomino.web_app import state_from_debug_json, state_to_debug_json


def _first_placement_transition(seed: int = 71):
    state = GameState.new(seed=seed, start_player=0)
    while state.phase == Phase.INITIAL_SELECTION:
        state = state.step(PickAction(state.current_row[0]))
    action = state.legal_actions()[0]
    return state, action, state.step(action)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_builds_reconstructable_information_safe_position(tmp_path):
    source, action, target = _first_placement_transition()
    game_path = tmp_path / "table_synthetic.jsonl"
    records = [
        {
            "kind": "decision",
            "active_player": "p0",
            "viewer_id": "p0",
            "captured_at": "t0",
            "state": state_to_debug_json(source),
        },
        {
            "kind": "decision",
            "active_player": f"p{target.current_actor}",
            "viewer_id": "p0",
            "captured_at": "t1",
            "state": state_to_debug_json(target),
        },
        {
            "kind": "final",
            "final": {
                "players": {
                    "p0": {"name": "Hero", "score": 42},
                    "p1": {"name": "Opponent", "score": 39},
                }
            },
        },
    ]
    _write_jsonl(game_path, records)
    digest = hashlib.sha256(game_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "kingdomino-placement-corpus/v1",
                "corpus_id": "synthetic",
                "source_schema": "kingdomino-bga-gamelog/v1",
                "games": [
                    {
                        "table_id": "synthetic",
                        "path": str(game_path),
                        "sha256": digest,
                        "split": "development",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    reconstruction_path = tmp_path / "reconstruction.jsonl"
    _write_jsonl(
        reconstruction_path,
        [
            {
                "table_id": "synthetic",
                "split": "development",
                "source_decision_index": 0,
                "target_decision_index": 1,
                "player": source.current_actor,
                "domino_id": source.pending_claims[source.actor_index].domino_id,
                "next_pick_domino_id": action.pick_domino_id,
                "placement": asdict(action.placement),
                "status": "reconstructed",
            }
        ],
    )
    output_path = tmp_path / "positions.jsonl"
    summary_path = tmp_path / "summary.json"

    summary = build_corpus(
        manifest_path=manifest_path,
        reconstruction_path=reconstruction_path,
        output_path=output_path,
        summary_path=summary_path,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    rebuilt = state_from_debug_json(row["state"])
    assert row["schema"] == SCHEMA
    assert row["split"] == "development"
    assert row["actor_role"] == "viewer"
    assert row["human_action_known"]
    assert row["human_action"]["action_idx"] in row["legal"]["action_indices"]
    assert row["state"]["debug"]["deck"] == sorted(row["state"]["debug"]["deck"])
    assert len(row["state"]["debug"]["deck"]) == row["deck_count"]
    assert rebuilt.current_actor == source.current_actor
    assert len(rebuilt.legal_actions()) == row["legal"]["action_count"]
    assert summary["positions"] == 1
    assert summary["human_action_known"] == 1
    assert summary["information_policy"]["future_reveal_order_present"] is False
    assert summary_path.exists()


def test_split_filter_does_not_leak_confirmation_positions(tmp_path):
    source, action, target = _first_placement_transition(seed=73)
    reconstruction_rows = []
    games = []
    for table_id, split in (("dev", "development"), ("confirm", "confirmation")):
        game_path = tmp_path / f"table_{table_id}.jsonl"
        _write_jsonl(
            game_path,
            [
                {
                    "kind": "decision",
                    "active_player": "p0",
                    "viewer_id": "p0",
                    "state": state_to_debug_json(source),
                },
                {
                    "kind": "decision",
                    "active_player": f"p{target.current_actor}",
                    "viewer_id": "p0",
                    "state": state_to_debug_json(target),
                },
            ],
        )
        games.append(
            {
                "table_id": table_id,
                "path": str(game_path),
                "sha256": hashlib.sha256(game_path.read_bytes()).hexdigest(),
                "split": split,
            }
        )
        reconstruction_rows.append(
            {
                "table_id": table_id,
                "split": split,
                "source_decision_index": 0,
                "target_decision_index": 1,
                "player": source.current_actor,
                "domino_id": source.pending_claims[source.actor_index].domino_id,
                "next_pick_domino_id": action.pick_domino_id,
                "placement": asdict(action.placement),
                "status": "reconstructed",
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "kingdomino-placement-corpus/v1",
                "corpus_id": "synthetic",
                "source_schema": "kingdomino-bga-gamelog/v1",
                "games": games,
            }
        ),
        encoding="utf-8",
    )
    reconstruction_path = tmp_path / "reconstruction.jsonl"
    _write_jsonl(reconstruction_path, reconstruction_rows)
    output_path = tmp_path / "positions.jsonl"

    summary = build_corpus(
        manifest_path=manifest_path,
        reconstruction_path=reconstruction_path,
        output_path=output_path,
        summary_path=tmp_path / "summary.json",
        split="development",
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["table_id"] for row in rows] == ["dev"]
    assert summary["counts_by_split"] == {"development": 1}
