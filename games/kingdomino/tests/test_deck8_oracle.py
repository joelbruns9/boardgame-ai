from __future__ import annotations

import argparse
import json
import random

import pytest

from games.kingdomino.action_codec import encode_action
from games.kingdomino.deck8_oracle import (
    EXPECTED_ROWS,
    RUST_SOLVER_BACKEND,
    _solve_rust_tail,
    apply_prefix_actions,
    conditioned_tails,
    evaluate_boundary_in_memory,
    run_oracle,
    validate_boundary_state,
)
from games.kingdomino.deck8_oracle_compare import (
    exact_separation,
    run as run_oracle_compare,
    score_arm_against_oracle,
)
from games.kingdomino.deck8_boundary_screen import (
    disagreement_verdict,
    selection_summary,
)
from games.kingdomino.deck8_oracle_corpus import (
    _choose,
    balanced_panel,
    exhaustive_panel,
    iid_panel,
)
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
        return 0.25, True, RUST_SOLVER_BACKEND

    monkeypatch.setattr("games.kingdomino.deck8_oracle._solve_rust_tail", fake_exact)
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


def test_compare_treats_all_exact_ties_as_rank_one():
    exact = {10: 0.3, 20: 0.3, 30: -0.1}
    arm = {
        "top_action_idx": 20,
        "nn_evaluations": 10,
        "elapsed_seconds": 0.1,
        "children": [
            {"action_idx": 10, "q_actor": 0.2, "visits": 5},
            {"action_idx": 20, "q_actor": 0.2, "visits": 6},
            {"action_idx": 30, "q_actor": -0.2, "visits": 1},
        ],
    }
    scored = score_arm_against_oracle(arm, exact)
    assert scored["exact_best_action_indices"] == [10, 20]
    assert scored["top1_exact"] is True
    assert scored["selected_exact_rank"] == 1
    assert scored["exact_regret_actor"] == 0.0


def test_compare_gap_skips_all_tied_best_actions():
    summary = exact_separation({10: 0.3, 20: 0.3, 30: 0.1, 40: -0.2})
    assert summary["best_action_indices"] == [10, 20]
    assert summary["runner_up_action_idx"] == 30
    assert summary["top_gap_actor"] == pytest.approx(0.2)
    assert summary["all_actions_tied"] is False

    all_tied = exact_separation({10: 0.3, 20: 0.3})
    assert all_tied["runner_up_action_idx"] is None
    assert all_tied["top_gap_actor"] is None
    assert all_tied["all_actions_tied"] is True


def test_a1c_oracle_comparison_is_disabled_until_leaf_batch_is_restored():
    with pytest.raises(RuntimeError, match="disabled until panel admission is wave-safe"):
        run_oracle_compare(argparse.Namespace(include_a1c=True))


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


def test_corpus_ties_use_absolute_tolerance_only():
    values = {10: 0.5, 20: 0.5 + 3e-10}
    assert _choose(values, random.Random(1)) == 20


def test_exposure_35_is_sampled_width_not_exhaustive_support():
    bag = list(range(1, 9))
    sampled = balanced_panel(bag, 35, random.Random(7))
    exhaustive = exhaustive_panel(bag)
    assert len(sampled) == len(exhaustive) == 70
    assert len(set(sampled)) < 70
    assert len(set(exhaustive)) == 70


def test_timeout_is_excluded_from_action_expectation(tmp_path, monkeypatch):
    state = _boundary_state()
    corpus = tmp_path / "positions.jsonl"
    write_frozen_positions([(state, {"fixture": True})], corpus)
    calls = 0

    def fail_then_solve(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0.0, False, RUST_SOLVER_BACKEND
        return 0.25, True, RUST_SOLVER_BACKEND

    monkeypatch.setattr(
        "games.kingdomino.deck8_oracle._solve_rust_tail", fail_then_solve
    )
    summary = run_oracle(
        argparse.Namespace(
            positions_path=str(corpus),
            position_index=0,
            prefix_actions="",
            output_dir=str(tmp_path / "oracle"),
            selection_reason="timeout exclusion test",
            score_scale=160.0,
            margin_gain=2.0,
            alpha=0.5,
            max_secs_per_tail=1.0,
            max_total_secs=0.0,
            max_cells=1,
            seed=7,
            stop_on_timeout=False,
        )
    )
    first_action = summary["actions"][0]
    assert first_action["complete"] is False
    assert first_action["rows_solved"] == 1
    assert first_action["expected_value_player0"] is None
    assert all(row["action_idx"] != first_action["action_idx"] for row in summary["ranking"])
    assert summary["timeouts_this_run"][0]["solver_backend"] == RUST_SOLVER_BACKEND


@pytest.mark.parametrize("stale_field", ["action_idx", "tail_state_key", "oracle_id"])
def test_resume_rejects_stale_cell_payload_identity(
    tmp_path, monkeypatch, stale_field
):
    state = _boundary_state()
    corpus = tmp_path / "positions.jsonl"
    write_frozen_positions([(state, {"fixture": True})], corpus)

    monkeypatch.setattr(
        "games.kingdomino.deck8_oracle._solve_rust_tail",
        lambda *_args, **_kwargs: (0.25, True, RUST_SOLVER_BACKEND),
    )
    args = argparse.Namespace(
        positions_path=str(corpus),
        position_index=0,
        prefix_actions="",
        output_dir=str(tmp_path / "oracle"),
        selection_reason="stale cell test",
        score_scale=160.0,
        margin_gain=2.0,
        alpha=0.5,
        max_secs_per_tail=1.0,
        max_total_secs=0.0,
        max_cells=1,
        seed=7,
        stop_on_timeout=True,
    )
    run_oracle(args)
    cell_path = next((tmp_path / "oracle" / "cells").glob("*.json"))
    payload = json.loads(cell_path.read_text(encoding="utf-8"))
    if stale_field == "action_idx":
        payload[stale_field] += 1
    else:
        payload[stale_field] = "stale-identity"
    cell_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cell identity/solve mismatch"):
        run_oracle(args)


def test_oracle_refuses_silent_python_fallback(monkeypatch):
    monkeypatch.setattr(
        "games.kingdomino.deck8_oracle._rust_state_from_python", lambda _state: None
    )
    with pytest.raises(RuntimeError, match="requires a working kingdomino_rust"):
        _solve_rust_tail(
            _boundary_state().step(_boundary_state().legal_actions()[0]),
            max_secs=1.0,
            score_scale=160.0,
            margin_gain=2.0,
            alpha=0.5,
        )
