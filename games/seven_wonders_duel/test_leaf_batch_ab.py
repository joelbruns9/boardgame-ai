"""The A/B must be able to say "no" and must not invent a difference.

Kingdomino settled leaf batching with head-to-head play and no solver: run a
match, read the win rate. The risk in copying that is a harness that reports a
win rate which reflects the seat, the seeds or the sampling rather than the
setting -- all of which look like a plausible percentage.
"""

from __future__ import annotations

import pytest

from .leaf_batch_ab import summarise, wilson_interval


def _rows(a_wins: int, b_wins: int, draws: int = 0):
    rows = [{"a_won": 1} for _ in range(a_wins)]
    rows += [{"a_won": 0} for _ in range(b_wins)]
    rows += [{"a_won": None} for _ in range(draws)]
    return rows


def test_an_even_split_cannot_separate_from_even():
    summary = summarise(_rows(100, 100), 6, 1)
    assert summary["a_win_rate"] == pytest.approx(0.5)
    assert not summary["separates_from_even"]


def test_a_clear_loss_separates():
    """The verdict this exists to deliver: the batched arm is weaker."""

    summary = summarise(_rows(70, 130), 8, 1)
    assert summary["a_win_rate"] == pytest.approx(0.35)
    assert summary["separates_from_even"]
    assert summary["ci95"][1] < 0.5


def test_draws_leave_the_denominator_rather_than_counting_as_losses():
    """A draw is not evidence against the arm. Counting it as a loss would bias
    every result downward by the draw rate."""

    summary = summarise(_rows(50, 50, draws=20), 6, 1)
    assert summary["decided"] == 100
    assert summary["draws"] == 20
    assert summary["a_win_rate"] == pytest.approx(0.5)


def test_small_samples_report_uncertainty_rather_than_a_verdict():
    """8 games at 6-2 looks like a 75% win rate and means nothing. The interval
    has to say so, because the point estimate will not."""

    summary = summarise(_rows(6, 2), 8, 1)
    assert summary["a_win_rate"] == pytest.approx(0.75)
    assert not summary["separates_from_even"], "8 games must not produce a verdict"


def test_the_interval_stays_inside_zero_and_one():
    """Wilson, not normal approximation: at 20/20 the normal interval runs past
    1.0 and reads as false precision."""

    low, high = wilson_interval(20, 20)
    assert 0.0 <= low <= high <= 1.0
    assert low > 0.5

    low, high = wilson_interval(0, 20)
    assert 0.0 <= low <= high <= 1.0
    assert high < 0.5


def test_no_games_is_maximum_uncertainty_not_a_zero_win_rate():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    summary = summarise([], 6, 1)
    assert not summary["separates_from_even"]
