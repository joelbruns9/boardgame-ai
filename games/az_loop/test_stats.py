from __future__ import annotations

import json

import pytest

from .stats import (
    GateStats,
    GenerationStats,
    IterationStats,
    ModelStats,
    OutcomeStats,
    stats_from_row,
    wilson_interval,
)


def test_schema_v2_round_trip_for_synthetic_second_game():
    original = IterationStats(
        generation=GenerationStats(games=12, moves=144),
        outcomes=OutcomeStats(terminal_reason={"score": 12}),
        gates=[
            GateStats(
                opponent="baseline",
                score_rate=0.6,
                wilson_lcb=0.35,
                wilson_ucb=0.82,
                games=10,
                pairs=5,
                decision="continue",
            )
        ],
        model=ModelStats(
            d_model=64, layers=2, heads=4, parameters=1234, precision="fp32"
        ),
        game_specific={"synthetic_game": {"tiles": 42}},
    )
    payload = json.loads(json.dumps(original.to_dict()))
    restored = stats_from_row({"log_schema_version": 2, "stats": payload})
    assert restored == original


def test_schema_v1_rows_are_tolerated():
    assert stats_from_row({"log_schema_version": 1, "generated_games": 10}) is None


def test_stats_validation_rejects_invalid_gate_bounds():
    stats = IterationStats(
        gates=[GateStats(score_rate=0.5, wilson_lcb=0.8, wilson_ucb=0.2)]
    )
    with pytest.raises(ValueError, match="Wilson"):
        stats.to_dict()


def test_wilson_interval_contains_even_match():
    lower, upper = wilson_interval(50.0, 100)
    assert lower < 0.5 < upper


def test_report_reads_schema_v1_and_v2_and_exposes_decay_diagnostics():
    from tools.az_report import build_report

    typed = IterationStats(
        generation=GenerationStats(
            games=10,
            moves=100,
            games_per_second=0.8,
            mean_batch_size=12.0,
            forced_row_share=0.2,
        ),
        outcomes=OutcomeStats(terminal_reason={"score": 10}),
    )
    rows = [
        {
            "iteration": 0,
            "log_schema_version": 1,
            "generated_games": 10,
            "generation_performance": {
                "performance": {"games_per_second": 1.0},
                "summary": {"games": 10, "victory_types": {"score": 10}},
            },
        },
        {
            "iteration": 1,
            "log_schema_version": 2,
            "stats": typed.to_dict(),
        },
    ]
    report = build_report(rows, block_size=1)
    assert report["schema_versions"] == [1, 2]
    assert report["throughput_diagnostic"]["relative_change"] == pytest.approx(-0.2)
    assert report["throughput_diagnostic"]["diagnostic_fields_present"][
        "mean_batch_size"
    ]
