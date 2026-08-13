from __future__ import annotations

import pytest

from games.kingdomino import elo_rating, symmetric_checkpoint_eval
from games.kingdomino.elo_rating import (
    EloConfig,
    _make_batched,
    play_rating_games_with_diagnostics,
    progressive_chance_deck_mask,
)
from games.kingdomino.symmetric_checkpoint_eval import (
    build_parser,
    progressive_diagnostic_errors,
)
from games.kingdomino.round_robin_eval import GameResult, PairResult


def test_open_loop_remains_default_and_constructor_mask_is_zero(monkeypatch):
    captured = {}

    def fake_batched(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(elo_rating.kingdomino_rust, "BatchedMCTS", fake_batched)
    _make_batched(4, 100, EloConfig())

    assert captured["kwargs"]["open_loop"] is True
    assert captured["kwargs"]["progressive_chance_deck_mask"] == 0


def test_progressive_constructor_matches_frozen_training_treatment(monkeypatch):
    captured = {}

    def fake_batched(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(elo_rating.kingdomino_rust, "BatchedMCTS", fake_batched)
    cfg = EloConfig(progressive_chance_decks=(8, 12))
    _make_batched(4, 100, cfg)

    assert captured["progressive_chance_deck_mask"] == (1 << 8) | (1 << 12)
    assert captured["progressive_chance_width_schedule"] == "4,8,16,32,64,70"
    assert captured["progressive_chance_n_init"] == 2
    assert captured["progressive_chance_d_min"] == 4
    assert captured["progressive_chance_deck8_cap"] == 16
    assert captured["progressive_chance_deck12_cap"] == 16
    assert captured["progressive_chance_max_init_fraction"] == 0.25


def test_progressive_mask_rejects_all_below_twelve_interpretation():
    with pytest.raises(ValueError, match="only deck counts 8 and 12"):
        progressive_chance_deck_mask(
            EloConfig(progressive_chance_decks=(4, 8, 12)))


def test_symmetric_match_applies_same_config_to_both_orientations(monkeypatch):
    monkeypatch.setattr(
        elo_rating, "make_rust_evaluator", lambda net, **kwargs: f"eval-{net}")
    calls = []

    def fake_run(eval0, eval1, n_games, seed_start, cfg):
        calls.append((eval0, eval1, n_games, seed_start,
                      cfg.progressive_chance_decks))
        outcome = 1 if eval0 == "eval-a" else -1
        row = [(seed_start, 40, 30, outcome)]
        diagnostics = {
            "progressive_chance_search_count_deck8": 2,
            "progressive_chance_search_count_deck12": 3,
            "progressive_chance_path_count_deck8": 5,
            "progressive_chance_path_count_deck12": 7,
            "progressive_chance_admission_count": 11,
            "progressive_chance_widening_count": 13,
            "progressive_chance_bootstrap_rows": 17,
            "progressive_chance_budget_blocked_count": 0,
            "progressive_chance_guardrail_blocked_count": 0,
            "progressive_chance_width_sample_count": 10,
            "progressive_chance_mean_active_width": 8.0,
            "progressive_chance_mean_mature_width": 4.0,
        }
        return row, diagnostics

    monkeypatch.setattr(
        elo_rating, "_run_batched_orientation_with_diagnostics", fake_run)
    cfg = EloConfig(progressive_chance_decks=(8, 12))
    pair, games, diagnostics = play_rating_games_with_diagnostics(
        "a", "b", "candidate", "baseline", 1, 123, cfg)

    assert calls == [
        ("eval-a", "eval-b", 1, 123, (8, 12)),
        ("eval-b", "eval-a", 1, 123, (8, 12)),
    ]
    assert pair.games == 2
    assert len(games) == 2
    assert diagnostics["progressive_chance_admission_count"] == 22
    assert diagnostics["progressive_chance_width_sample_count"] == 20
    assert diagnostics["progressive_chance_mean_active_width"] == 8.0
    assert len(diagnostics["orientations"]) == 2


def test_progressive_diagnostics_fail_closed():
    diagnostics = {
        "progressive_chance_search_count_deck8": 1,
        "progressive_chance_path_count_deck8": 1,
        "progressive_chance_search_count_deck12": 0,
        "progressive_chance_path_count_deck12": 0,
        "progressive_chance_admission_count": 1,
        "progressive_chance_bootstrap_rows": 4,
        "progressive_chance_width_sample_count": 1,
    }
    errors = progressive_diagnostic_errors(diagnostics, (8, 12))
    assert any("searches recorded at deck 12" in error for error in errors)
    assert any("paths crossed at deck 12" in error for error in errors)

    diagnostics["progressive_chance_search_count_deck12"] = 1
    diagnostics["progressive_chance_path_count_deck12"] = 1
    assert progressive_diagnostic_errors(diagnostics, (8, 12)) == []


def test_cli_defaults_to_frozen_symmetric_progressive_profile():
    args = build_parser().parse_args([
        "--candidate", "candidate.pt",
        "--baseline", "baseline.pt",
        "--output-dir", "out",
    ])
    assert args.search_mode == "chance_progressive"
    assert args.chance_progressive_decks == (8, 12)
    assert args.chance_deck8_cap == 16
    assert args.chance_deck12_cap == 16
    assert args.games == 2048
    assert args.sims == 400


def test_cli_writes_auditable_artifacts_and_refuses_overwrite(
        monkeypatch, tmp_path):
    candidate = tmp_path / "candidate.pt"
    baseline = tmp_path / "baseline.pt"
    candidate.write_bytes(b"candidate")
    baseline.write_bytes(b"baseline")
    output = tmp_path / "evaluation"
    args = build_parser().parse_args([
        "--candidate", str(candidate),
        "--baseline", str(baseline),
        "--output-dir", str(output),
        "--games", "2",
    ])
    monkeypatch.setattr(
        symmetric_checkpoint_eval, "_net_from_checkpoint",
        lambda path, device: path.name)
    diagnostics = {
        "progressive_chance_search_count_deck8": 2,
        "progressive_chance_search_count_deck12": 2,
        "progressive_chance_path_count_deck8": 4,
        "progressive_chance_path_count_deck12": 4,
        "progressive_chance_admission_count": 4,
        "progressive_chance_widening_count": 2,
        "progressive_chance_bootstrap_rows": 16,
        "progressive_chance_budget_blocked_count": 0,
        "progressive_chance_guardrail_blocked_count": 0,
        "progressive_chance_width_sample_count": 4,
        "progressive_chance_mean_active_width": 8.0,
        "progressive_chance_mean_mature_width": 4.0,
        "orientations": [],
    }

    def fake_match(*unused_args, **unused_kwargs):
        pair = PairResult(
            a="candidate", b="baseline", games=2, a_wins=1, b_wins=1,
            a_score_sum=70, b_score_sum=70)
        games = [
            GameResult(20330000, "candidate", "baseline", 40, 30,
                       "candidate", 0),
            GameResult(20330000, "baseline", "candidate", 40, 30,
                       "baseline", 0),
        ]
        return pair, games, diagnostics

    monkeypatch.setattr(
        symmetric_checkpoint_eval,
        "play_rating_games_with_diagnostics",
        fake_match,
    )
    result = symmetric_checkpoint_eval.run_symmetric_evaluation(args)

    assert result["valid"] is True
    assert result["search_is_symmetric"] is True
    assert result["parameters"]["progressive_chance_decks"] == [8, 12]
    assert (output / "manifest.json").is_file()
    assert (output / "games.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (output / "result.json").is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        symmetric_checkpoint_eval.run_symmetric_evaluation(args)
