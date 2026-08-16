"""The oracle probe's arithmetic, which is what the conclusion rests on.

Generation is exercised by running the probe; these pin the summary, because a
mis-signed comparison or an off-by-one bucket would produce a plausible number
rather than an error -- and the number is meant to decide where effort goes.
"""

from __future__ import annotations

from .solver_oracle_probe import _sign, _summarise, report


def _row(*, truth: float, net: float, search: float, regime: str = "exact", **kw):
    base = {
        "regime": regime,
        "legal": 4,
        "optimal": 3,
        "sims": 64,
        "nodes": 100,
        "moves_from_end": 1,
        "truth": truth,
        "net": net,
        "search": search,
    }
    return base | kw


def test_a_computed_zero_counts_as_a_draw_not_a_win():
    """Expectimax sums in floating point, so a true zero arrives as -1.4e-17.
    Reading its sign naively would score a proven draw as a proven loss."""

    assert _sign(-1.4e-17) == 0
    assert _sign(1.0) == 1
    assert _sign(-1.0) == -1


def test_sign_agreement_counts_who_is_winning_not_how_much():
    rows = [
        _row(truth=1.0, net=0.05, search=0.9),  # both right about the winner
        _row(truth=1.0, net=-0.8, search=-0.7),  # both wrong
        _row(truth=-1.0, net=0.6, search=-0.2),  # search rescues it
    ]
    overall = report(rows)["overall"]
    assert overall["positions"] == 3
    assert overall["net_sign_agrees"] == 1
    assert overall["search_sign_agrees"] == 2


def test_search_improvement_is_measured_against_the_net_row_by_row():
    """The number that separates "bad value head, search compensates" from
    "bad value head, search inherits it"."""

    rows = [
        _row(truth=1.0, net=0.0, search=0.9),  # closer
        _row(truth=1.0, net=0.9, search=0.2),  # further
        _row(truth=1.0, net=0.5, search=0.5),  # unchanged is not an improvement
    ]
    assert report(rows)["overall"]["search_improves"] == 1


def test_absolute_error_is_summarised_over_the_right_axis():
    rows = [_row(truth=1.0, net=0.0, search=0.5), _row(truth=-1.0, net=0.0, search=-0.5)]
    overall = report(rows)["overall"]
    assert overall["net_abs_error"]["mean"] == 1.0
    assert overall["search_abs_error"]["mean"] == 0.5


def test_regimes_are_split_because_only_one_of_them_labels_a_value():
    rows = [
        _row(truth=1.0, net=0.0, search=0.0, regime="exact"),
        _row(truth=0.2, net=0.0, search=0.0, regime="exact_expectimax"),
        _row(truth=0.2, net=0.0, search=0.0, regime="exact_expectimax"),
    ]
    by_regime = report(rows)["by_regime"]
    assert by_regime["exact"]["positions"] == 1
    assert by_regime["exact_expectimax"]["positions"] == 2


def test_distance_buckets_saturate_rather_than_growing_without_bound():
    """A long tail of one-position buckets would read as noise; 9+ is one bucket."""

    rows = [_row(truth=1.0, net=1.0, search=1.0, moves_from_end=n) for n in (0, 9, 40)]
    buckets = report(rows)["by_moves_from_end"]
    assert set(buckets) == {"0", "9"}
    assert buckets["9"]["positions"] == 2


def test_an_empty_sample_reports_nothing_rather_than_dividing_by_zero():
    summary = report([])
    assert summary["overall"]["positions"] == 0
    assert summary["overall"]["net_abs_error"] == _summarise([])
    assert summary["tie_structure"]["mean_legal"] is None
