from __future__ import annotations

import argparse
import json

import pytest

from games.kingdomino.action_codec import encode_action
from games.kingdomino.deck8_oracle import (
    EXPECTED_ROWS,
    apply_prefix_actions,
    conditioned_tails,
    evaluate_boundary_in_memory,
    run_oracle,
    validate_boundary_state,
)
from games.kingdomino.deck8_oracle_compare import score_arm_against_oracle
from games.kingdomino.deck8_boundary_screen import (
    disagreement_verdict,
    selection_summary,
)
from games.kingdomino.deck8_oracle_corpus import balanced_panel, iid_panel
from games.kingdomino.denial_signal_sweep import write_frozen_positions
from games.kingdomino.game import GameState


def _boundary_state(seed: int = 17):
    state = GameState.new(seed=seed)
    while not state.pending_claims:
        state = state.step(state.legal_actions()[0])
    state.deck = state.deck[:8]
    while state.actor_index < len(state.pending_claims) - 1:
        state = state.step(state.legal_actions()[0])
    return state


def test_boundary_enumerates_all_uniform_deck8_rows():
    state = _boundary_state()
    action = state.legal_actions()[0]
    futures = conditioned_tails(state, action)

    assert len(futures) == EXPECTED_ROWS == 70
    assert len({tuple(child.current_row) for child, _ in futures}) == 70
    assert {len(child.deck) for child, _ in futures} == {4}
    assert sum(probability for _, probability in futures) == pytest.approx(1.0)
    assert all(probability == pytest.approx(1 / 70) for _, probability in futures)


def test_boundary_rejects_an_earlier_actor():
    state = GameState.new(seed=17)
    while not state.pending_claims:
        state = state.step(state.legal_actions()[0])
    state.deck = state.deck[:8]
    with pytest.raises(ValueError, match="final pre-reveal actor"):
        validate_boundary_state(state)


def test_prefix_actions_are_explicit_and_legality_checked():
    state = GameState.new(seed=17)
    state.deck = state.deck[:8]
    indices = []
    expected = state
    while expected.actor_index < len(expected.pending_claims) - 1:
        action = expected.legal_actions()[0]
        indices.append(encode_action(action, expected))
        expected = expected.step(action)

    actual = apply_prefix_actions(state, indices)
    assert actual.actor_index == expected.actor_index
    assert actual.current_row == expected.current_row
    with pytest.raises(ValueError, match="not legal"):
        apply_prefix_actions(state, [999_999])


def test_in_memory_oracle_uses_all_rows_and_actor_frame():
    state = _boundary_state()
    calls = []

    def solve_tail(child):
        calls.append(tuple(child.current_row))
        return sum(child.current_row) / 100.0, True

    rows = evaluate_boundary_in_memory(state, solve_tail)
    assert len(calls) == 70 * len(state.legal_actions())
    assert all(row["rows_solved"] == 70 for row in rows)
    assert rows == sorted(
        rows, key=lambda row: (-row["expected_value_actor"], row["action_idx"])
    )


def test_run_oracle_persists_and_resumes_solved_cells(tmp_path, monkeypatch):
    state = _boundary_state()
    corpus = tmp_path / "positions.jsonl"
    write_frozen_positions([(state, {"fixture": True})], corpus)
    output = tmp_path / "oracle"
    calls = 0

    def fake_exact(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return 0.25, True

    monkeypatch.setattr("games.kingdomino.deck8_oracle.exact_endgame_value", fake_exact)
    common = dict(
        positions_path=str(corpus), position_index=0, prefix_actions="",
        output_dir=str(output), selection_reason="unit test", score_scale=160.0,
        margin_gain=2.0, alpha=0.5, max_secs_per_tail=1.0,
        max_total_secs=0.0, max_cells=1, seed=7, stop_on_timeout=True,
    )
    first = run_oracle(argparse.Namespace(**common))
    assert first["status"] == "incomplete"
    assert first["solved_cells_this_run"] == 1
    assert calls == 1

    second = run_oracle(argparse.Namespace(**common))
    assert second["solved_cells_this_run"] == 1
    assert calls == 2
    assert len(list((output / "cells").glob("*.json"))) == 2
    assert json.loads((output / "manifest.json").read_text())["identity"]["position_index"] == 0


def test_compare_scores_top1_regret_and_pairwise_ordering():
    exact = {10: 0.3, 20: 0.2, 30: -0.1}
    arm = {
        "top_action_idx": 20,
        "nn_evaluations": 101,
        "elapsed_seconds": 0.5,
        "children": [
            {"action_idx": 10, "q_actor": 0.1, "visits": 20},
            {"action_idx": 20, "q_actor": 0.4, "visits": 30},
            {"action_idx": 30, "q_actor": -0.2, "visits": 10},
        ],
    }
    scored = score_arm_against_oracle(arm, exact)
    assert scored["top1_exact"] is False
    assert scored["selected_exact_rank"] == 2
    assert scored["exact_regret_actor"] == pytest.approx(0.1)
    assert scored["q_pairwise_accuracy"] == pytest.approx(2 / 3)
    assert scored["visit_pairwise_accuracy"] == pytest.approx(2 / 3)


def test_boundary_screen_requires_stable_disagreement():
    assert selection_summary([20, 20, 20])["unanimous"] is True
    stable = {
        "x0": {"selection": selection_summary([10, 10, 10])},
        "x1_hajek_balanced": {"selection": selection_summary([20, 20, 20])},
    }
    noisy = {
        "x0": {"selection": selection_summary([10, 10, 20])},
        "x1_hajek_balanced": {"selection": selection_summary([20, 20, 20])},
    }
    same = {
        "x0": {"selection": selection_summary([10, 10, 10])},
        "x1_hajek_balanced": {"selection": selection_summary([10, 10, 10])},
    }
    assert disagreement_verdict(stable)["stable_disagreement"] is True
    assert disagreement_verdict(noisy)["stable_disagreement"] is False
    assert disagreement_verdict(same)["stable_disagreement"] is False


def test_corpus_panels_match_width_and_balanced_exposure():
    bag = list(range(1, 9))
    balanced = balanced_panel(bag, 4, __import__("random").Random(7))
    iid = iid_panel(bag, 8, __import__("random").Random(7))
    assert len(balanced) == len(iid) == 8
    counts = {tile: sum(tile in row for row in balanced) for tile in bag}
    assert counts == {tile: 4 for tile in bag}
    assert all(len(row) == 4 and tuple(sorted(row)) == row for row in balanced + iid)
