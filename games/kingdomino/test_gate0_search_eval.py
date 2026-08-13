from __future__ import annotations

import pytest

from games.kingdomino import elo_rating, gate0_search_eval
from games.kingdomino.elo_rating import EloConfig
from games.kingdomino.gate0_search_eval import (
    build_parser,
    gate0_diagnostic_errors,
    gate0_verdict,
    paired_score_summary,
)
from games.kingdomino.round_robin_eval import GameResult, PairResult


def _diagnostics(search_count: int, seat: int) -> dict:
    return {
        "progressive_chance_search_count_deck8": search_count,
        "progressive_chance_search_count_deck12": search_count,
        "progressive_chance_path_count_deck8": 10,
        "progressive_chance_path_count_deck12": 10,
        "progressive_chance_admission_count": 10,
        "progressive_chance_widening_count": 4,
        "progressive_chance_bootstrap_rows": 40,
        "progressive_chance_budget_blocked_count": 0,
        "progressive_chance_guardrail_blocked_count": 0,
        "progressive_chance_width_sample_count": 8,
        "progressive_chance_mean_active_width": 8.0,
        "progressive_chance_mean_mature_width": 4.0,
        "progressive_chance_seat": seat,
    }


def test_search_ab_uses_one_evaluator_and_rotates_progressive_seat(monkeypatch):
    evaluator = object()
    monkeypatch.setattr(
        elo_rating, "make_rust_evaluator", lambda *args, **kwargs: evaluator)
    calls = []

    def fake_orientation(eval0, eval1, n_games, seed_start, cfg):
        calls.append((eval0, eval1, n_games, seed_start,
                      cfg.progressive_chance_seat))
        outcome = 1 if cfg.progressive_chance_seat == 0 else -1
        return [(seed_start, 40, 30, outcome)], _diagnostics(2, -99)

    monkeypatch.setattr(
        elo_rating, "_run_batched_orientation_with_diagnostics",
        fake_orientation)
    pair, games, diagnostics = (
        elo_rating.play_search_ab_games_with_diagnostics(
            object(), "chance_progressive", "open_loop", 1, 20360000,
            EloConfig(progressive_chance_decks=(8, 12))))

    assert calls == [
        (evaluator, evaluator, 1, 20360000, 0),
        (evaluator, evaluator, 1, 20360000, 1),
    ]
    assert games[0].p0 == "chance_progressive"
    assert games[1].p1 == "chance_progressive"
    assert pair.a_wins == 2
    assert diagnostics["orientations"][0]["progressive_chance_seat"] == 0
    assert diagnostics["orientations"][1]["progressive_chance_seat"] == 1


def test_gate0_diagnostics_require_exactly_one_treated_seat():
    rows = [_diagnostics(4, 0), _diagnostics(4, 1)]
    merged = dict(_diagnostics(8, -1), orientations=rows)
    assert gate0_diagnostic_errors(merged, (8, 12), paired_seeds=2) == []

    rows[1]["progressive_chance_search_count_deck12"] = 8
    errors = gate0_diagnostic_errors(merged, (8, 12), paired_seeds=2)
    assert any("exactly one treated seat" in error for error in errors)


def test_pair_aware_interval_and_gate_verdicts():
    proceed = paired_score_summary([1.0] * 16)
    fail = paired_score_summary([0.5] * 16)
    inconclusive = paired_score_summary([0.0, 1.0] * 8)

    assert gate0_verdict(proceed) == "proceed"
    assert gate0_verdict(fail) == "fail"
    assert gate0_verdict(inconclusive) == "inconclusive"
    assert inconclusive["pairs"] == 16
    assert inconclusive["mean"] == 0.5
    assert inconclusive["standard_error"] > 0


def test_gate0_cli_defaults_are_frozen_advisor_profile():
    args = build_parser().parse_args([
        "--checkpoint", "current_best.pt",
        "--output-dir", "out",
    ])
    assert args.games == 2048
    assert args.sims == 800
    assert args.seed == 20360000
    assert args.chance_progressive_decks == (8, 12)
    assert args.chance_deck8_cap == 16
    assert args.chance_deck12_cap == 16


def test_gate0_cli_loads_checkpoint_once_and_writes_artifacts(
        monkeypatch, tmp_path):
    checkpoint = tmp_path / "current_best.pt"
    checkpoint.write_bytes(b"same-network")
    output = tmp_path / "gate0"
    args = build_parser().parse_args([
        "--checkpoint", str(checkpoint),
        "--output-dir", str(output),
        "--games", "4",
        "--allow-smoke",
    ])
    loads = []
    monkeypatch.setattr(
        gate0_search_eval, "_net_from_checkpoint",
        lambda path, device: loads.append((path, device)) or object())

    games = [
        GameResult(20360000, "chance_progressive", "open_loop", 40, 30,
                   "chance_progressive", 0),
        GameResult(20360001, "chance_progressive", "open_loop", 30, 40,
                   "open_loop", 0),
        GameResult(20360000, "open_loop", "chance_progressive", 30, 40,
                   "chance_progressive", 0),
        GameResult(20360001, "open_loop", "chance_progressive", 40, 30,
                   "open_loop", 0),
    ]
    pair = PairResult(
        a="chance_progressive", b="open_loop", games=4,
        a_wins=2, b_wins=2, a_score_sum=140, b_score_sum=140)
    rows = [_diagnostics(4, 0), _diagnostics(4, 1)]
    diagnostics = dict(_diagnostics(8, -1), orientations=rows)
    monkeypatch.setattr(
        gate0_search_eval, "play_search_ab_games_with_diagnostics",
        lambda *args, **kwargs: (pair, games, diagnostics))

    result = gate0_search_eval.run_gate0_evaluation(args)

    assert len(loads) == 1
    assert result["network_is_identical"] is True
    assert result["search_is_asymmetric"] is True
    assert result["valid"] is True
    assert result["verdict"] == "smoke_only"
    assert result["parameters"]["sims_nn_rows_per_move"] == 800
    assert (output / "manifest.json").is_file()
    assert (output / "games.jsonl").read_text(encoding="utf-8").count("\n") == 4
    assert (output / "result.json").is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        gate0_search_eval.run_gate0_evaluation(args)


def test_gate0_cli_rejects_underpowered_run_without_explicit_smoke(
        tmp_path):
    checkpoint = tmp_path / "current_best.pt"
    checkpoint.write_bytes(b"same-network")
    args = build_parser().parse_args([
        "--checkpoint", str(checkpoint),
        "--output-dir", str(tmp_path / "gate0"),
        "--games", "2046",
    ])
    with pytest.raises(ValueError, match="requires at least 2048 games"):
        gate0_search_eval.run_gate0_evaluation(args)
