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


def _corpus(costs, *, collecting_attempt_nodes: int | None = 1_000_000_000,
            collecting_games: int = 1, collecting_model: dict | None = MODEL):
    """`None` cost means the position never completed at any budget measured.

    The collecting bar defaults WIDE so these fixtures exercise pricing rather
    than the admission ceiling; the ceiling tests set it deliberately. Pass
    `None` for the bar to get a corpus that has to infer it.

    `collecting_model` defaults to MODEL -- the same one the tests price with --
    so coverage holds and the fixtures exercise pricing. The coverage tests pass
    a different one deliberately.

    `collecting_games` is 1 so per-game figures equal per-corpus ones and the
    arithmetic in these tests stays readable; the scaling has its own test.
    """

    return {
        "feature_names": ["f0", "f1"],
        "collecting_attempt_nodes": collecting_attempt_nodes,
        "collecting_games": collecting_games,
        "collecting_model": collecting_model,
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


# --- what a corpus can and cannot see --------------------------------------
#
# A corpus holds the positions a run ATTEMPTED. Everything its bar refused is
# absent -- not recorded as expensive, absent -- so a wider candidate bar prices
# identically to the collecting one and reads as "widening buys nothing". The
# truth is "this corpus cannot see what widening would buy".


def test_a_bar_above_the_corpus_ceiling_is_refused_not_priced():
    from .solver_corpus import price

    corpus = _corpus([1e5])
    corpus["collecting_attempt_nodes"] = 5_000_000
    with pytest.raises(ValueError, match="admission ceiling"):
        price(corpus, MODEL, attempt_nodes=10_000_000, max_nodes=1e9, games=1)


def test_the_ceiling_is_read_rather_than_inferred_when_recorded():
    """Inference lands just BELOW the true bar, because the largest prediction
    in a corpus approaches the collecting bar without reaching it. On cloud2
    that inferred 39,975,202 against a true 40,000,000 -- excluding the run's
    own settings, which is the one candidate that must always be priceable."""

    from .solver_corpus import admission_ceiling

    corpus = _corpus([1e5], collecting_attempt_nodes=None)
    inferred = admission_ceiling(corpus, MODEL)
    corpus["collecting_attempt_nodes"] = 40_000_000
    assert admission_ceiling(corpus, MODEL) == 40_000_000
    assert inferred != 40_000_000, "the fixture must exercise the difference"


def test_the_collecting_runs_own_settings_are_always_priceable():
    """The status quo is what every other candidate is compared against."""

    from .solver_corpus import price

    corpus = _corpus([1e5, 1e7])
    corpus["collecting_attempt_nodes"] = 40_000_000
    out = price(corpus, MODEL, attempt_nodes=40_000_000, max_nodes=40_000_000, games=1)
    assert out["attempts"] == 2


def test_unpriceable_bars_are_dropped_with_a_reason_not_silently():
    """Dropping them silently would leave a grid whose top end simply vanished,
    which looks identical to a grid that was never asked for."""

    from .solver_sizing import candidates

    corpus = _corpus([1e5])
    corpus["collecting_attempt_nodes"] = 5_000_000
    rows = candidates(corpus, MODEL, games=1, bars=(5_000_000, 50_000_000))
    assert rows, "the priceable bar was dropped too"
    assert all(row["attempt_nodes"] == 5_000_000 for row in rows)


def test_a_grid_entirely_above_the_ceiling_stops_rather_than_returning_nothing():
    from .solver_sizing import size

    corpus = _corpus([1e5])
    corpus["collecting_attempt_nodes"] = 1_000_000
    with pytest.raises(SystemExit, match="above this corpus"):
        size(corpus, MODEL, rate=1e6, threads=4, generation_wall_seconds=100,
             games=1, target_share=0.8, bars=(50_000_000,))


# --- the drain tail: a stall that is absorbed is not a stall ----------------
#
# The scheduler cannot end an iteration while a game is parked on a solve, so
# the worst case is `max_nodes / rate` with one thread running alone. Measured
# across 97 cloud2 iterations, though, the drain tail below 25% of peak
# occupancy was a median 16.5% of generation wall at a 40M cap -- games
# finishing unevenly, nothing to do with solving -- while idle after the last NN
# batch, where a solve-induced stall WOULD show, was 3s median and 3s worst.
#
# So a solve shorter than the drain lands in a window that is already idle. The
# sizer reports the stall and flags the ones that exceed it; it does not filter,
# because filtering would cost hard proofs to avoid a cost the run already pays.


def _sized(bar, **kwargs):
    """Drive the real inputs rather than patching TIMEOUT_MULTIPLES.

    `candidates(..., multiples=TIMEOUT_MULTIPLES)` binds the module constant as
    a DEFAULT ARGUMENT, so rebinding it on the module after import changes
    nothing -- a test that patched it silently measured the 1x row.
    """

    from .solver_sizing import size

    return size(_corpus([1e5]), MODEL, rate=1e6, threads=10,
                generation_wall_seconds=1000, games=1, target_share=0.8,
                bars=(bar,), **kwargs)


def _row(result, max_nodes):
    return next(r for r in result["considered"] if r["max_nodes"] == max_nodes)


def test_the_worst_case_stall_is_reported_for_every_candidate():
    """`max_nodes / rate`: one solve running the cap to exhaustion, alone."""

    result = _sized(10_000_000)
    for row in result["considered"]:
        assert row["worst_stall_seconds"] == pytest.approx(row["max_nodes"] / 1e6)


def test_a_stall_inside_the_drain_is_not_flagged():
    """1,000s wall x 16.5% = 165s of drain; half of that is the threshold."""

    result = _sized(10_000_000)
    assert result["drain_seconds"] == pytest.approx(165.0)
    # 10M at 1e6 nodes/s is 10s, comfortably inside.
    assert not _row(result, 10_000_000)["stall_exceeds_drain"]


def test_a_stall_beyond_the_drain_is_flagged_but_still_offered():
    """Flagged, not filtered. The operator decides; a filter would silently cost
    hard proofs to avoid a cost the run may already be paying in idle time."""

    result = _sized(10_000_000)
    # 320M at 1e6 nodes/s is 320s against a 165s drain.
    row = _row(result, 320_000_000)
    assert row["stall_exceeds_drain"]
    assert row in result["considered"], "a flagged candidate must stay selectable"


def test_a_measured_drain_overrides_the_cloud2_default():
    """The default belongs to cloud2's geometry -- 1,000 games over 256 slots --
    and the box sweep picks a different one."""

    assert not _row(_sized(10_000_000, drain_fraction=0.90),
                    320_000_000)["stall_exceeds_drain"]
    assert _row(_sized(10_000_000, drain_fraction=0.01),
                320_000_000)["stall_exceeds_drain"]


# --- review findings, each with the case that reproduced it ----------------


def test_coverage_is_a_property_of_the_bar_AND_the_model():
    """A recorded 40M bar does not establish coverage under a refitted model.

    Reproduced on the shipped corpus: seed 116260767 move 64 was NOT attempted
    -- required 48,162,395 under the collecting model, above the 40M bar -- and
    the refit puts it at 34,617,215, inside the bar and absent from the corpus.

    It also disproves the "none newly admitted" claim I made when comparing the
    two models ON the corpus: the corpus contains only what the old model
    admitted, so newly-admitted positions cannot appear in it by construction.
    """

    from .solver_corpus import admission_ceiling

    corpus = _corpus([1e5])
    corpus["collecting_attempt_nodes"] = 40_000_000
    assert admission_ceiling(corpus, MODEL) == 40_000_000
    # A model that predicts one decade CHEAPER admits positions the collecting
    # run refused, so the ceiling must shrink by that factor.
    cheaper = dict(MODEL, intercept=MODEL["intercept"] - 1.0)
    assert admission_ceiling(corpus, cheaper) == pytest.approx(4_000_000)


def test_an_identical_model_does_not_shrink_the_ceiling():
    """The guard must not penalise the case it was not written for."""

    from .solver_corpus import admission_ceiling

    corpus = _corpus([1e5])
    corpus["collecting_attempt_nodes"] = 40_000_000
    assert admission_ceiling(corpus, dict(MODEL)) == 40_000_000


def test_a_model_that_only_differs_in_its_fit_block_is_the_same_model():
    """`affordable` reads the intercept, the weights and the margin. A refit
    that changed none of them changes no decision, and a whole-file comparison
    would shrink the ceiling for nothing."""

    from .solver_corpus import admission_ceiling

    corpus = _corpus([1e5])
    corpus["collecting_attempt_nodes"] = 40_000_000
    annotated = dict(MODEL, fit={"held_out_r2": 0.99}, comment="rewritten")
    assert admission_ceiling(corpus, annotated) == 40_000_000


def test_demand_scales_with_the_target_iteration_size():
    """`price` divided corpus nodes by the REQUESTED games and `size`
    multiplied by the same number, so they cancelled: 100, 1,000 and 10,000
    games all reported identical demand while capacity grew with the wall,
    making larger iterations look free."""

    from .solver_corpus import price

    corpus = _corpus([1e5, 1e6], collecting_games=10)
    small = price(corpus, MODEL, attempt_nodes=1e8, max_nodes=1e9, games=10)
    large = price(corpus, MODEL, attempt_nodes=1e8, max_nodes=1e9, games=100)
    assert large["nodes_for_games"] == pytest.approx(10 * small["nodes_for_games"])
    assert small["nodes_per_game"] == pytest.approx(large["nodes_per_game"])


def test_a_corpus_without_a_game_count_is_refused():
    """Per-game demand cannot be derived from it, and guessing would reinstate
    exactly the cancellation above."""

    from .solver_corpus import price

    corpus = _corpus([1e5])
    del corpus["collecting_games"]
    with pytest.raises(SystemExit, match="collecting game count"):
        price(corpus, MODEL, attempt_nodes=1e8, max_nodes=1e9, games=100)
