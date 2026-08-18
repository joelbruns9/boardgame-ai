"""Semantics of the cost-trigger validation, especially around censoring.

The failure this guards is a quiet one: if censored solves leak into the fit or
the R^2, the model learns how busy the machine was rather than how hard the
position is, and reports a good number for it.
"""

from __future__ import annotations

import math

import pytest

from .validate_cost_trigger import (
    TRIGGER_FEATURES,
    UNAVAILABLE_AT_DECISION_TIME,
    compare_triggers,
    fit,
    predict,
    score,
)

FEATURES = ("cards_left", "unrevealed")


def _row(cards: int, unrevealed: int, nodes: int, censored: bool = False) -> dict:
    return {
        "cards_left": cards,
        "unrevealed": unrevealed,
        "nodes": nodes,
        "censored": censored,
    }


def test_a_trigger_cannot_use_a_feature_only_known_after_the_game_ends():
    assert "moves_from_end" in UNAVAILABLE_AT_DECISION_TIME
    assert "moves_from_end" not in TRIGGER_FEATURES
    assert "cards_left" in TRIGGER_FEATURES


def test_the_fit_recovers_an_exact_log_linear_cost():
    # nodes = 10 ** (4 + 0.5 * cards + 0.25 * unrevealed), noise-free.
    #
    # The intercept is 4, not 0, because node counts are integers: at 10**1.5 the
    # truncation to 31 is a 0.9% error in the target and visibly biases the fit.
    # Real solves run 10**3 to 10**6 (measured), where truncation is ~1e-5.
    rows = [
        _row(c, u, int(10 ** (4 + 0.5 * c + 0.25 * u)))
        for c in range(1, 11)
        for u in range(0, 5)
    ]
    coefficients = fit(rows, FEATURES)
    assert coefficients[0] == pytest.approx(4.0, abs=1e-3)
    assert coefficients[1] == pytest.approx(0.5, abs=1e-3)
    assert coefficients[2] == pytest.approx(0.25, abs=1e-3)


def test_censored_rows_never_enter_the_r2():
    """A censored row's node count is where the clock cut it off, not its cost.

    Scoring it as an observation would make a model that predicts machine load
    look accurate. Here the censored rows are given absurd node counts: if they
    were scored, R^2 would collapse.
    """

    clean = [_row(c, 0, int(10 ** (4 + 0.5 * c))) for c in range(1, 11)]
    coefficients = fit(clean, FEATURES)
    poisoned = clean + [_row(c, 0, 10, censored=True) for c in range(1, 11)]

    baseline = score(coefficients, clean, FEATURES)["r2"]
    result = score(coefficients, poisoned, FEATURES)
    assert baseline == pytest.approx(1.0, abs=1e-5)
    # The poisoned rows move the score by nothing at all, not merely by little.
    assert result["r2"] == baseline
    assert result["n_uncensored"] == 10
    assert result["n_censored"] == 10


def test_underprediction_of_a_censored_floor_is_counted_as_an_error():
    """Above the floor says nothing; below it is provably wrong.

    A censored solve had already reached `nodes` when it was cut, so its true
    cost is at least that. A prediction under the floor is an unambiguous
    underestimate -- the one thing censoring still lets us measure.
    """

    rows = [_row(c, 0, int(10 ** (4 + 0.5 * c))) for c in range(1, 11)]
    coefficients = fit(rows, FEATURES)

    below = [_row(5, 0, 10**12, censored=True)]  # true cost far above prediction
    above = [_row(5, 0, 1, censored=True)]  # floor beneath prediction, uninformative
    assert score(coefficients, rows + below, FEATURES)[
        "censored_underprediction_rate"
    ] == pytest.approx(1.0)
    assert score(coefficients, rows + above, FEATURES)[
        "censored_underprediction_rate"
    ] == pytest.approx(0.0)


def test_a_censored_row_is_never_counted_as_bought():
    """It did not produce a proof, whatever it spent getting there."""

    rows = [_row(3, 0, 100, censored=True), _row(3, 0, 100)]
    coefficients = fit([_row(3, 0, 100)], FEATURES)
    result = compare_triggers(rows, coefficients, FEATURES, budget=10**6)
    assert result["card_cap_best"]["attempts"] == 2
    assert result["card_cap_best"]["solved"] == 1


def test_prediction_is_the_dot_product_of_the_design_row():
    coefficients = [1.0, 2.0, 3.0]
    assert predict(coefficients, _row(4, 5, 0), FEATURES) == 1.0 + 2 * 4 + 3 * 5


def test_the_budget_rule_drops_a_position_it_cannot_afford():
    """The whole point: an expensive position is declined before it is paid for."""

    cheap = _row(2, 0, 100)
    dear = _row(20, 0, 10**12)
    coefficients = fit(
        [_row(c, 0, int(10 ** (4 + 0.5 * c))) for c in range(1, 21)], FEATURES
    )
    # Budget 10**7: the cheap position predicts 10**5 and clears it even after
    # the 3.3x safety margin; the dear one predicts 10**14 and cannot.
    result = compare_triggers([cheap, dear], coefficients, FEATURES, budget=10**7)
    assert result["cost_predicted"]["attempts"] == 1
    assert math.isfinite(result["cost_predicted"]["nodes"])


def test_transfer_reports_both_distributions_separately():
    """A model can fit its own data and fail elsewhere; the point is to see it.

    Here the second distribution follows a different cost law, so the in-sample
    score must stay high while the transferred one collapses. Reporting a single
    blended number would hide exactly this.
    """

    from .validate_cost_trigger import transfer

    home = [_row(c, 0, int(10 ** (4 + 0.5 * c))) for c in range(1, 11)]
    away = [_row(c, 0, int(10 ** (4 + 2.0 * c))) for c in range(1, 11)]
    result = transfer(home, away, FEATURES)
    assert result["in_distribution"]["r2"] > 0.99
    assert result["transferred"]["r2"] < 0.0


def test_study_rows_keep_censored_positions_as_floors(tmp_path):
    """The study writes `nodes: None` when its own budget ran out.

    Dropping those would silently discard the most expensive positions -- the
    ones that decide whether a budget is enough.
    """

    import json

    from .validate_cost_trigger import study_rows

    path = tmp_path / "study.json"
    path.write_text(
        json.dumps(
            {
                "study_nodes": 20_000_000,
                "rows": [
                    {"cards_left": 5, "unrevealed": 1, "nodes": 1234, "censored": False},
                    {"cards_left": 9, "unrevealed": 3, "nodes": None, "censored": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = study_rows(path)
    assert len(rows) == 2
    assert rows[1]["censored"] is True
    # The floor is the budget it exhausted. Zero would make the single most
    # expensive position in the corpus read as the cheapest.
    assert rows[1]["nodes"] == 20_000_000
