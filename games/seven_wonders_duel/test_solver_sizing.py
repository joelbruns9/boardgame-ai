"""Sizing the solver's two caps from a priced corpus and a box measurement."""

from __future__ import annotations

import json
import math

import pytest

from .solver_corpus import price
from .solver_sizing import render_env, size

MODEL = {
    # Predicts 10^6 nodes for every position: weights are zero, so the corpus
    # rows below differ only in what they COST, which is what is being priced.
    "intercept": 6.0,
    "margin_decades": 0.4,
    "features": ["f0", "f1"],
    "coefficients": {"f0": 0.0, "f1": 0.0},
}


def _corpus(costs):
    """`None` cost means the position never completed at any budget measured."""

    return {
        "feature_names": ["f0", "f1"],
        "rows": [
            {"features": [0.0, 0.0], "true_nodes": c, "declined": c is None}
            for c in costs
        ],
    }


def test_the_margin_sits_between_the_bar_and_admission():
    """`affordable` is `predict + margin <= log10(bar)`, so the effective bar is
    the flag divided by 10**margin. A pricer that forgot the margin would admit
    positions the engine refuses and report proofs no run could get."""

    corpus = _corpus([1e5])
    # predict 10^6, margin 0.4 -> needs a bar of at least 10^6.4 = 2.51M.
    assert price(corpus, MODEL, attempt_nodes=3_000_000, max_nodes=1e9, games=1)["attempts"] == 1
    assert price(corpus, MODEL, attempt_nodes=2_000_000, max_nodes=1e9, games=1)["attempts"] == 0


def test_a_decline_costs_the_whole_timeout():
    """A decline is not free -- the budget was spent synchronously before it was
    reached -- and pricing it as free is what makes a wide bar look cheap."""

    out = price(_corpus([1e9]), MODEL, attempt_nodes=1e7, max_nodes=4e6, games=1)
    assert out["proofs"] == 0
    assert out["nodes"] == 4e6
    assert out["wasted_fraction"] == 1.0


def test_a_position_the_study_never_resolved_never_completes():
    """Its buffer cost is a FLOOR, not a cost. Pricing it as affordable at the
    cap it failed at is the one answer that is certainly wrong."""

    out = price(_corpus([None]), MODEL, attempt_nodes=1e7, max_nodes=1e12, games=1)
    assert out["proofs"] == 0
    assert out["nodes"] == 1e12


def test_raising_the_timeout_converts_declines_into_proofs():
    corpus = _corpus([1e5, 5e6, 5e7])
    tight = price(corpus, MODEL, attempt_nodes=1e7, max_nodes=1e6, games=1)
    loose = price(corpus, MODEL, attempt_nodes=1e7, max_nodes=1e8, games=1)
    assert tight["proofs"] == 1 and loose["proofs"] == 3
    assert loose["wasted_fraction"] < tight["wasted_fraction"]


def test_narrowing_the_bar_cannot_add_attempts():
    """Admission is monotone in the bar. Anything else means the pricer and
    `CostModel::affordable` disagree about the direction of the comparison."""

    corpus = _corpus([1e5] * 4)
    wide = price(corpus, MODEL, attempt_nodes=1e8, max_nodes=1e9, games=1)
    narrow = price(corpus, MODEL, attempt_nodes=1e6, max_nodes=1e9, games=1)
    assert narrow["attempts"] <= wide["attempts"]


def test_the_budget_is_threads_times_wall_times_rate_times_share():
    result = size(
        _corpus([1e5]), MODEL, rate=1e6, threads=10,
        generation_wall_seconds=1000, games=1, target_share=0.8,
        bars=(10_000_000,),
    )
    assert result["budget_nodes"] == pytest.approx(10 * 1000 * 1e6 * 0.8)


def test_a_box_that_cannot_afford_any_candidate_is_told_so():
    """Silently returning the cheapest option would launch a run whose solver
    competes with generation for cores it does not have."""

    with pytest.raises(SystemExit, match="no candidate fits"):
        size(
            _corpus([1e9] * 50), MODEL, rate=1e3, threads=1,
            generation_wall_seconds=1, games=1, target_share=0.8,
            bars=(10_000_000,),
        )


def test_an_unmeasured_input_is_refused_rather_than_defaulted():
    """A zero rate or wall means the box was never measured. Proceeding would
    size the caps against a machine that does not exist."""

    for kwargs in (
        {"rate": 0.0, "threads": 4, "generation_wall_seconds": 100},
        {"rate": 1e6, "threads": 4, "generation_wall_seconds": 0.0},
    ):
        with pytest.raises(SystemExit, match="not measured"):
            size(_corpus([1e5]), MODEL, games=1, target_share=0.8,
                 bars=(10_000_000,), **kwargs)


def test_the_chosen_pair_maximises_proofs_within_the_budget():
    corpus = _corpus([1e5, 1e7, 1e8])
    result = size(
        corpus, MODEL, rate=1e6, threads=10, generation_wall_seconds=10_000,
        games=1, target_share=0.8, bars=(10_000_000,),
    )
    chosen = result["chosen"]
    assert chosen["fits"]
    best_possible = max(row["proofs"] for row in result["considered"] if row["fits"])
    assert chosen["proofs"] == best_possible


def test_the_env_file_sets_what_the_launcher_reads(tmp_path):
    """Including the CLOCK. It is derived from the node budget and this box's
    rate; leaving it out would let stage 6b re-derive it from a different rate,
    or a stale environment carry a value that binds."""

    from pathlib import Path

    result = size(
        _corpus([1e5]), MODEL, rate=1e6, threads=10,
        generation_wall_seconds=1000, games=1, target_share=0.8,
        bars=(10_000_000,),
    )
    text = render_env(result["chosen"])
    setup = (Path(__file__).resolve().parents[2] / "setup_cloud_7wd.sh").read_text(
        encoding="utf-8"
    )
    exported = [
        line.removeprefix("export ").split("=", 1)[0]
        for line in text.splitlines()
        if line.startswith("export ")
    ]
    assert set(exported) == {
        "ENDGAME_SOLVER_ATTEMPT_NODES",
        "ENDGAME_SOLVER_MAX_NODES",
        "ENDGAME_SOLVER_MAX_SECS",
    }
    for name in exported:
        assert f'{name}="${{{name}:-' in setup, (
            f"the sizing exports {name}, but the launcher never reads it"
        )


def test_the_clock_stays_slack_against_the_node_budget():
    """If the clock binds, a node-censored decline becomes a load-dependent one
    and the buffer stops being a function of its seeds."""

    result = size(
        _corpus([1e5]), MODEL, rate=1e6, threads=10,
        generation_wall_seconds=1000, games=1, target_share=0.8,
        bars=(10_000_000,),
    )
    chosen = result["chosen"]
    reachable = chosen["max_secs"] * 1e6      # nodes the clock allows at this rate
    assert reachable > chosen["max_nodes"], (
        "the wall clock would stop the solve before the node budget does"
    )
