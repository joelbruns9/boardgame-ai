from __future__ import annotations

import json

import pytest

from games.kingdomino.game import GameState, Phase, PickAction
from games.kingdomino.placement_headroom_audit import (
    DEFAULT_MANIFEST,
    _board_fingerprint,
    _load_manifest,
    audit_game_completeness,
    reconstruct_decision,
    solve_fixed_tile_sequence,
)
from games.kingdomino.web_app import state_to_debug_json


def _first_placement_transition(seed: int = 17):
    state = GameState.new(seed=seed, start_player=0)
    while state.phase == Phase.INITIAL_SELECTION:
        state = state.step(PickAction(state.current_row[0]))
    action = state.legal_actions()[0]
    target = state.step(action)
    return state, action, target


def _play_to_final_phase(seed: int = 31):
    state = GameState.new(seed=seed, start_player=0)
    while state.phase != Phase.FINAL_PLACEMENT:
        state = state.step(state.legal_actions()[0])
    return state


def test_reconstructs_engine_verified_placement_and_pick():
    source, action, target = _first_placement_transition()
    decisions = [
        {"kind": "decision", "captured_at": "t0", "state": state_to_debug_json(source)},
        {"kind": "decision", "captured_at": "t1", "state": state_to_debug_json(target)},
    ]

    row = reconstruct_decision(
        table_id="synthetic",
        split="development",
        decisions=decisions,
        source_index=0,
    )

    assert row.status == "reconstructed"
    assert row.drop_reason is None
    assert row.next_pick_domino_id == action.pick_domino_id
    assert row.placement == {
        "x1": action.placement.x1,
        "y1": action.placement.y1,
        "x2": action.placement.x2,
        "y2": action.placement.y2,
        "flipped": action.placement.flipped,
    }
    assert _board_fingerprint(target.boards[source.current_actor])


def test_authoritative_bga_unknown_domino_ids_are_observable_exact():
    source, action, target = _first_placement_transition(seed=23)
    actor = source.current_actor
    source_json = state_to_debug_json(source)
    target_json = state_to_debug_json(target)
    for state_json in (source_json, target_json):
        state_json["debug"]["notes"] = [
            f"Player {actor} board built from authoritative BGA kingdom grid."
        ]
        for cell in state_json["boards"][actor]["cells"]:
            if cell["domino_id"] != -1:
                cell["domino_id"] = -2
    decisions = [
        {"kind": "decision", "state": source_json},
        {"kind": "decision", "state": target_json},
    ]

    row = reconstruct_decision(
        table_id="synthetic",
        split="development",
        decisions=decisions,
        source_index=0,
    )

    assert row.status == "reconstructed"
    assert row.next_pick_domino_id == action.pick_domino_id


def test_drops_warning_scoped_to_actor_board():
    source, _action, target = _first_placement_transition()
    source_json = state_to_debug_json(source)
    source_json["board_reconstruction_warning"] = {"player": source.current_actor}
    decisions = [
        {"kind": "decision", "state": source_json},
        {"kind": "decision", "state": state_to_debug_json(target)},
    ]

    row = reconstruct_decision(
        table_id="synthetic",
        split="development",
        decisions=decisions,
        source_index=0,
    )

    assert row.status == "dropped"
    assert row.drop_reason == "source_board_unreliable"


def test_terminal_regret_is_score_verified_without_a_later_snapshot():
    source = _play_to_final_phase()
    actor = source.current_actor
    terminal = source
    while terminal.phase != Phase.GAME_OVER:
        terminal = terminal.step(terminal.legal_actions()[0])
    decisions = [{"kind": "decision", "captured_at": "t0", "state": state_to_debug_json(source)}]

    row = reconstruct_decision(
        table_id="synthetic",
        split="development",
        decisions=decisions,
        source_index=0,
        final_scores={actor: terminal.scores()[actor]},
        scoring_rules={(source.config.harmony, source.config.middle_kingdom)},
    )

    assert row.status == "score_verified_terminal"
    assert row.actual_final_score == terminal.scores()[actor]
    assert row.best_final_score >= row.actual_final_score
    assert row.final_regret == row.best_final_score - row.actual_final_score
    assert row.matching_final_sequences >= 1
    assert isinstance(row.first_action_unique, bool)


def test_manifest_freezes_disjoint_whole_game_split():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    games = manifest["games"]
    assert len(games) == 36
    assert sum(game["split"] == "development" for game in games) == 24
    assert sum(game["split"] == "confirmation" for game in games) == 12
    assert len({game["table_id"] for game in games}) == 36
    assert all(len(game["sha256"]) == 64 for game in games)
    # The file itself remains ordinary, portable JSON rather than generated code.
    json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))


def test_complete_claim_sequences_cover_24_tiles_per_player(tmp_path):
    state = GameState.new(seed=47, start_player=0)
    records = []
    while state.phase != Phase.GAME_OVER:
        actor = state.current_actor
        records.append(
            {
                "kind": "decision",
                "active_player": f"player-{actor}",
                "state": state_to_debug_json(state),
            }
        )
        state = state.step(state.legal_actions()[0])
    records.append(
        {
            "kind": "final",
            "final": {
                "playerorder": ["player-0", "player-1"],
                "players": {
                    "player-0": {"name": "P0", "score": state.scores()[0]},
                    "player-1": {"name": "P1", "score": state.scores()[1]},
                },
            },
        }
    )
    path = tmp_path / "complete.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    rows = audit_game_completeness(path, "synthetic", "development")

    assert len(rows) == 2
    assert all(row.sequence_complete for row in rows)
    assert all(row.tile_count == 24 for row in rows)
    assert all(row.complete_layers == 12 for row in rows)
    assert all(all(len(layer) == 2 for layer in row.dominoes_by_layer) for row in rows)
    assert set(rows[0].ordered_domino_ids).isdisjoint(rows[1].ordered_domino_ids)
    assert set(rows[0].ordered_domino_ids) | set(rows[1].ordered_domino_ids) == set(range(1, 49))


def test_fixed_sequence_exact_dp_matches_unpruned_wide_beam():
    sequence = [1, 13, 19]
    exact = solve_fixed_tile_sequence(
        sequence,
        harmony=False,
        middle_kingdom=False,
        beam_width=None,
        exact_state_limit=100_000,
    )
    wide = solve_fixed_tile_sequence(
        sequence,
        harmony=False,
        middle_kingdom=False,
        beam_width=100_000,
    )

    assert exact.best_score == wide.best_score
    assert exact.discards == wide.discards
    assert len(exact.placements) == len(sequence)
    assert len(exact.layer_stats) == len(sequence)
    assert all(stat.kept_states <= stat.unique_states for stat in exact.layer_stats)
    assert all(stat.unique_states <= stat.expanded_actions for stat in exact.layer_stats)


def test_fixed_sequence_python_and_rust_backends_match_when_unpruned():
    rust = pytest.importorskip("kingdomino_rust")
    if not hasattr(rust, "fixed_sequence_beam"):
        pytest.skip("kingdomino_rust predates the placement-audit beam entry point")
    sequence = [1, 13, 19]
    python_result = solve_fixed_tile_sequence(
        sequence,
        harmony=True,
        middle_kingdom=True,
        beam_width=100_000,
        backend="python",
    )
    rust_result = solve_fixed_tile_sequence(
        sequence,
        harmony=True,
        middle_kingdom=True,
        beam_width=100_000,
        backend="rust",
    )

    assert python_result.best_score == rust_result.best_score
    assert python_result.discards == rust_result.discards
    assert [stat.unique_states for stat in python_result.layer_stats] == [
        stat.unique_states for stat in rust_result.layer_stats
    ]

    board = rust.RustBoard(7, 7)
    completed, state_counts, _elapsed = rust.fixed_sequence_exact_state_counts(
        board, sequence, 100_000
    )
    assert completed
    assert [int(item[4]) for item in state_counts] == [
        stat.unique_states for stat in python_result.layer_stats
    ]
    exact_score = rust.fixed_sequence_exact_score(
        board, sequence, True, True, 100_000
    )
    assert int(exact_score[0]) == python_result.best_score
