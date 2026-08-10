"""Pure analysis tests for the paired production chance-panel A/B."""

from __future__ import annotations

import pytest

from games.kingdomino.deck8_chance_ab import (
    bootstrap_mean_interval,
    summarize_match,
)


def _row(seed, score0, score1, outcome0):
    return {
        "seed": seed,
        "score0": score0,
        "score1": score1,
        "outcome0": outcome0,
    }


def test_summary_uses_treatment_frame_and_pairs_seats_by_seed():
    seat0 = [
        _row(10, 30, 20, 1),
        _row(11, 15, 25, -1),
    ]
    seat1 = [
        _row(10, 20, 30, -1),
        _row(11, 25, 25, 0),
    ]
    result = summarize_match(seat0, seat1, bootstrap_seed=7)

    assert result["treatment_wins"] == 2
    assert result["control_wins"] == 1
    assert result["draws"] == 1
    assert result["treatment_points_rate"] == pytest.approx(0.625)
    assert result["paired_points"]["mean"] == pytest.approx(0.625)
    assert result["paired_score_margin"]["mean"] == pytest.approx(2.5)
    assert [pair["seed"] for pair in result["pairs"]] == [10, 11]


def test_summary_rejects_unpaired_seed_sets():
    with pytest.raises(ValueError, match="identical seed sets"):
        summarize_match(
            [_row(10, 1, 0, 1)],
            [_row(11, 0, 1, -1)],
            bootstrap_seed=1,
        )


def test_bootstrap_is_deterministic_and_resamples_pair_units():
    first = bootstrap_mean_interval([0.0, 1.0], seed=13, resamples=2_000)
    second = bootstrap_mean_interval([0.0, 1.0], seed=13, resamples=2_000)
    assert first == second
    assert first["mean"] == pytest.approx(0.5)
    assert first["ci95_low"] == pytest.approx(0.0)
    assert first["ci95_high"] == pytest.approx(1.0)
