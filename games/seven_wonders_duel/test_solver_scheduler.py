"""The endgame solver inside the scheduler production actually runs.

Every solver test before this one drove `self_play_many_mock`, which routes to
`self_play::run_many` -- a scheduler that passes `None` where the pipelined ones
pass a `SolverPool` (`self_play.rs:2528` against `:2696`). No pool means no slot
ever parks, so nothing exercised the pump, the harvest, `resume_after_solve`, or
the `SolvePending` stage at all. `test_async_solver.py` varied
`set_solver_threads` across 0, 1, 2 and 4 and compared runs that all took the
same synchronous path: it could not fail.

These drive `self_play_many_net`, which routes to
`run_many_pipelined_sharded` -- the scheduler `self_play_many_flat_net` uses in
production. The only difference from production is the adapter boundary (a
Python callback here, packed bytes there), and that boundary has its own gates
in `test_f4_scheduler.py` and the flat-batch tests.

The property under test is **record identity**: async solving is a throughput
change, so a run at any thread count must produce byte-identical records to the
synchronous one. Anything else is a silent buffer corruption, not a crash.
"""

from __future__ import annotations

import pytest

from pathlib import Path

from .rust_bridge import rust_games_for_self_play

REPO_ROOT = Path(__file__).resolve().parents[2]
from .test_f4_scheduler import _common, _row_eval

swr = pytest.importorskip("seven_wonders_rust")

SEEDS = [2026081940, 2026081941, 2026081942, 2026081943, 2026081944, 2026081945]
FIRST = [i % 2 for i in range(len(SEEDS))]
MAX_CARDS = 9


@pytest.fixture(autouse=True)
def _reset_globals():
    yield
    swr.set_endgame_solver(0)
    swr.set_solver_threads(0)


def _records(
    threads: int,
    *,
    max_nodes: int = 3_000_000,
    fallback: bool = False,
    full_fraction: float = 0.3,
    shards: int = 2,
    slots: int = 4,
    attempt_nodes: int = 0,
):
    """One run through the production scheduler, with the solver on.

    `attempt_nodes` 0 means "same as max_nodes", which is what one shared number
    always meant -- so every caller predating the split is unaffected.
    """

    swr.set_endgame_solver(max_nodes, 120.0, MAX_CARDS, True, attempt_nodes)
    swr.set_solver_threads(threads)

    def adapter(rows):
        return [_row_eval(tokens, actor, legal) for tokens, actor, legal in rows]

    records, _ = swr.self_play_many_net(
        adapter=adapter,
        games=rust_games_for_self_play(SEEDS, FIRST),
        game_seeds=SEEDS,
        solve_endgames=True,
        solver_fallback_research=fallback,
        scheduler_workers=shards,
        max_active_slots=slots,
        **(_common(leaf_batch=1, global_batch_cap=8)
           | {"full_search_fraction": full_fraction}),
    )
    return records


def _moves(records):
    return [move for record in records for move in record["moves"]]


def _attempted(records):
    return [move for move in _moves(records) if move["solver_attempted"]]


# --- the harness must be able to fail -------------------------------------


def test_the_solver_actually_runs_on_this_path():
    """Guards every assertion below from passing vacuously.

    If no solve is attempted, identity between thread counts is trivially true
    and these tests would stay green through any bug in the async machinery --
    which is exactly how the previous gate passed for weeks.
    """

    attempted = _attempted(_records(0))
    assert attempted, "no solve was attempted; the rest of this file proves nothing"
    assert any(
        move["solver_masked"] for move in attempted
    ), "no solve produced a mask; the overlay never reached a record"


def test_solving_is_what_makes_the_thread_count_reachable():
    """With the solver off, threads cannot matter -- so this is the control.

    It fixes the interpretation of the identity tests: they compare runs that
    genuinely take different paths, not runs that both fall through.
    """

    swr.set_endgame_solver(0)
    swr.set_solver_threads(4)

    def adapter(rows):
        return [_row_eval(tokens, actor, legal) for tokens, actor, legal in rows]

    records, _ = swr.self_play_many_net(
        adapter=adapter,
        games=rust_games_for_self_play(SEEDS, FIRST),
        game_seeds=SEEDS,
        solve_endgames=False,
        scheduler_workers=2,
        max_active_slots=4,
        **_common(leaf_batch=1, global_batch_cap=8),
    )
    assert not _attempted(records)


# --- the gate -------------------------------------------------------------


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_async_records_match_the_synchronous_ones(threads):
    """THE gate, on the real scheduler.

    A solve dispatched to the pool parks its slot; the outcome comes back on a
    channel and resumes it. Timing therefore differs between thread counts while
    the records must not, because async is purely a throughput change.
    """

    assert _records(threads) == _records(0), f"{threads} solver threads diverged"


@pytest.mark.parametrize("threads", [1, 4])
def test_identity_holds_when_almost_every_move_is_cheap(threads):
    """A mixed schedule interleaves parked and searching slots.

    Every move full is the easy case: solves are dense and the pump has little
    else to do. Cheap moves are where a lost or misrouted outcome shows up.
    """

    assert _records(threads, full_fraction=0.05) == _records(0, full_fraction=0.05)


@pytest.mark.parametrize("threads", [1, 4])
def test_identity_holds_across_shard_counts(threads):
    """One SolverPool is built per scheduler loop, so shards multiply threads.

    A four-thread, three-shard run has twelve solver threads against a two-shard
    run's eight, and the records still must not move.
    """

    assert _records(threads, shards=3, slots=6) == _records(0, shards=3, slots=6)


def _metrics(threads: int, *, solve: bool, max_nodes: int = 3_000_000):
    """`_records`, but handing back the scheduler metrics instead."""

    swr.set_endgame_solver(max_nodes, 120.0, MAX_CARDS, True)
    swr.set_solver_threads(threads)

    def adapter(rows):
        return [_row_eval(tokens, actor, legal) for tokens, actor, legal in rows]

    _records_out, metrics = swr.self_play_many_net(
        adapter=adapter,
        games=rust_games_for_self_play(SEEDS, FIRST),
        game_seeds=SEEDS,
        solve_endgames=solve,
        scheduler_workers=2,
        max_active_slots=4,
        **(_common(leaf_batch=1, global_batch_cap=8)),
    )
    return metrics


def test_parked_slot_time_is_measured_rather_than_invisible():
    """How much slot capacity the solver consumes had NO instrument.

    `waiting_slot_ns` counts slots with an outstanding NN REQUEST, which is a
    different thing, and it read 0 in both the cloud2 run and the laptop soak.
    Parked slots were meanwhile counted as `ready` -- the metric meaning "able
    to produce work" included slots that structurally cannot, because a parked
    slot yields no evaluation group until its solve returns.

    Written to fail two ways: silent zero with the solver ON (the counter
    exported but never set, which is the failure this tree keeps finding), and
    non-zero with the solver OFF (charging time to the wrong bucket).
    """

    off = _metrics(0, solve=False)
    assert off["parked_slot_ns"] == 0, (
        "no solve ran, so no slot can have been parked"
    )

    on = _metrics(4, solve=True)
    assert on["parked_slot_ns"] > 0, (
        "the solver ran but parked slot-time is zero -- the counter is exported "
        "and never set, which is exactly the structurally-zero failure"
    )
    # Parked time is slot-time, so it cannot exceed the live slot-time it is
    # drawn from.
    assert on["parked_slot_ns"] <= on["live_slot_ns"]


def test_parking_loses_no_games():
    """A parked slot returns no evaluation group, which once read as 'finished'.

    That bug retired games mid-play; the symptom would be short records rather
    than an error.
    """

    for threads in (0, 4):
        records = _records(threads)
        assert len(records) == len(SEEDS)
        for record in records:
            assert record["moves"], "a game with no moves was retired early"
            assert record["winner"] is not None or record["scores"] is not None


# --- the fallback re-search, on the real scheduler -------------------------


def _declines(records):
    return [m for m in _moves(records) if m["solver_stop"] == "nodes"]


def test_a_tiny_budget_produces_the_declines_the_fallback_needs():
    """The fallback only fires on a declined solve, so a budget that never
    declines would make the tests below vacuous."""

    assert _declines(_records(0, max_nodes=20_000))


@pytest.mark.parametrize("threads", [1, 4])
def test_the_fallback_is_identical_sync_and_async(threads):
    """The fallback moves a slot SolvePending -> NeedRoot -> Searching.

    That transition exists only on this path -- `resume_after_solve` is never
    reached without a pool -- so this is the first test that can see it.
    """

    common = {"fallback": True, "max_nodes": 20_000, "full_fraction": 0.05}
    assert _records(threads, **common) == _records(0, **common)


def test_the_fallback_changes_the_games_it_touches():
    """It must actually alter a move, or the flag is dead configuration."""

    common = {"max_nodes": 20_000, "full_fraction": 0.05}
    assert _records(0, fallback=True, **common) != _records(0, fallback=False, **common)


def test_a_re_searched_row_still_reports_its_failed_solve():
    """The carried overlay survives the trip back through NeedRoot.

    Without it the row reads as a position the trigger never selected, and the
    declines vanish from the statistics that size the solver's budget.
    """

    records = _records(0, fallback=True, max_nodes=20_000, full_fraction=0.05)
    declines = _declines(records)
    assert declines
    for move in declines:
        assert move["solver_attempted"] is True
        assert move["solver_nodes"] > 0
        assert move["solver_value"] is None


# --- exclude_parked_from_budget -------------------------------------------
#
# A slot parked on a solve yields no evaluation group until its solve returns.
# Under the historical behaviour it holds its budget token anyway, so no
# replacement game can search in its place. These drive the FLAT entry point,
# which is the only one exposing the flag and the one production uses.


def _flat(exclude: bool, *, slots: int = 4, shards: int = 2, max_nodes: int = 2_000_000):
    from .control_table import ensure_rust_table
    from .test_coalescer import _deterministic_adapter

    ensure_rust_table()
    swr.set_endgame_solver(max_nodes, 120.0, MAX_CARDS, True)
    swr.set_solver_threads(2)
    return swr.self_play_many_flat_net(
        adapter=_deterministic_adapter,
        games=rust_games_for_self_play(SEEDS, FIRST),
        game_seeds=SEEDS,
        global_batch_cap=256,
        leaf_batch=1,
        cheap_sims_min=16,
        cheap_sims_max=16,
        full_sims_min=48,
        full_sims_max=48,
        full_search_fraction=0.5,
        top_k=8,
        draft_prior=0.0,
        iteration=1,
        scheduler_workers=shards,
        max_active_slots=slots,
        max_moves=256,
        force=True,
        solve_endgames=True,
        exclude_parked_from_budget=exclude,
    )


def test_excluding_parked_slots_changes_no_record():
    """THE gate. This is a throughput change and nothing else.

    If it moved a single record it would be a silent buffer corruption rather
    than a crash -- the same property `test_async_records_match_the_synchronous_ones`
    exists for.
    """

    off, _ = _flat(False)
    on, _ = _flat(True)
    assert on == off, "releasing a parked slot's token changed the games played"


def test_the_control_actually_parks_something():
    """Guards the gate above from passing vacuously.

    With no parking there is nothing to exclude, and every assertion here would
    hold through any bug in the accounting.
    """

    _records, metrics = _flat(False)
    assert metrics["parked_slot_ns"] > 0, (
        "no slot parked, so this file proves nothing about excluding parked slots"
    )


def test_excluding_parked_slots_releases_the_token():
    """The point of the change: a parked slot's token goes back to the budget.

    Asserted on the RELEASE EVENT, not on peak live slots. Whether the freed
    token is then taken up depends on a refill happening while that solve is
    still outstanding, which is a timing question -- an earlier version of this
    test asserted peak live and failed 2 runs in 3 under load. Whether the
    release happened is not a timing question.
    """

    _off_r, off = _flat(False, slots=4)
    _on_r, on = _flat(True, slots=4)

    assert off["parked_off_budget_events"] == 0, (
        "a slot released its token with the flag OFF"
    )
    assert on["parked_off_budget_events"] > 0, (
        "no parked slot released its token; the flag reached nothing"
    )
    # The cap still binds when parked slots keep their tokens.
    assert off["max_live_slots"] <= 4


def test_the_flag_is_off_by_default():
    """It changes what `max_active_slots` MEANS -- concurrent games becomes
    concurrent SEARCHING games -- so a slot count measured under one semantic is
    not comparable under the other. Defaulting it on would silently invalidate
    every slot number ever measured."""

    _records, metrics = _flat(False, slots=4)
    assert metrics["max_live_slots"] <= 4


# --- two caps: the attempt bar and the timeout ------------------------------
#
# One number used to serve both, with `margin_decades` the only way to separate
# them -- `predict + margin <= log10(budget)` put the attempt bar at
# `budget / 10**margin`, so raising the timeout widened admission by the same
# factor unless the margin was moved to compensate, by hand, in log space.
#
# The two decisions fail in opposite directions. A bar set too high admits
# hopeless positions that each burn the full timeout for nothing (45.9% of all
# solver nodes on cloud2 iteration 96). A timeout set too low discards work
# already done on positions the model merely underestimated.


def _install_cost_model():
    """The shipped model, which is what makes the bar mean anything.

    Without a model installed `solver_wants` falls back to the card cap and the
    node bar is not consulted at all -- so a test of the bar that forgot this
    would pass while measuring nothing.
    """

    from .phase_d import configure_endgame_cost_model

    return configure_endgame_cost_model(
        REPO_ROOT / "games/seven_wonders_duel/endgame_cost_model.json"
    )


def test_an_unset_attempt_bar_reproduces_the_single_number():
    swr.set_endgame_solver(40_000_000, 75.0, MAX_CARDS, True)
    assert swr.endgame_solver() == (40_000_000, 75.0, MAX_CARDS, True, 40_000_000), (
        "0 must resolve to the timeout, or every run before the split changes"
    )


def test_the_bar_and_the_timeout_are_reported_separately():
    swr.set_endgame_solver(320_000_000, 900.0, MAX_CARDS, True, 16_000_000)
    max_nodes, _secs, _cards, _mask, bar = swr.endgame_solver()
    assert (max_nodes, bar) == (320_000_000, 16_000_000)


def test_a_bar_above_the_timeout_is_refused():
    """Every position admitted above the timeout spends it in full and answers
    nothing -- the exact waste the split exists to remove, and a factor-sized
    typo rather than a digit-sized one."""

    with pytest.raises(ValueError, match="exceeds"):
        swr.set_endgame_solver(40_000_000, 75.0, MAX_CARDS, True, 80_000_000)


def test_narrowing_the_bar_attempts_fewer_positions():
    """The behaviour the split is FOR, on the production scheduler.

    Same timeout throughout, so what changes is admission and not the solve.

    COUNTS, not a subset -- and the difference is the point. A successful solve
    masks the policy target, which changes the move sampled, which changes every
    position after it. So a narrower bar does not merely drop rows from the same
    game: it plays a different game, and positions appear that the wider bar
    never reached. (Measured here: 57 attempts against 64, with one position
    unique to the narrow run.)

    That makes the attempt bar a TARGET-CHANGING knob, not a free one like slots
    or the batch cap. It cannot be A/B'd on wall clock alone, and two runs either
    side of it do not share a buffer definition.
    """

    if _install_cost_model() is None:
        pytest.skip("no cost model installed; the card cap ignores the node bar")

    def attempted(bar):
        records = _records(0, max_nodes=3_000_000, attempt_nodes=bar)
        return {
            (game, move["i"])   # the move index key is `i` on the Rust row
            for game, record in enumerate(records)
            for move in record["moves"]
            if move["solver_attempted"]
        }

    wide = attempted(3_000_000)
    narrow = attempted(300_000)
    assert wide, "nothing was attempted at the wide bar; the test proves nothing"
    assert len(narrow) < len(wide), (
        f"narrowing the bar 10x did not shrink the attempted set "
        f"({len(narrow)} vs {len(wide)})"
    )


def test_every_attempted_solve_records_what_was_predicted():
    """The attempt bar filters on the prediction, and until it was recorded the
    bar could not be tuned from a run's own output: the only costs observable
    were those of positions the bar had already admitted.

    Asserted on the DECLINES too -- they are exactly the rows a different bar
    would move, and a prediction logged only for solves that succeeded would be
    a sample selected by the outcome it is meant to predict.
    """

    if _install_cost_model() is None:
        pytest.skip("no cost model installed; nothing predicts anything")

    records = _records(0, max_nodes=200_000, full_fraction=0.3)
    attempted = [m for m in _moves(records) if m["solver_attempted"]]
    assert attempted, "nothing was attempted; the assertion below is vacuous"
    for move in attempted:
        assert move["solver_predicted_nodes"] is not None, (
            "an attempted solve carries no prediction"
        )
        assert move["solver_predicted_nodes"] > 0

    declined = [m for m in attempted if m["solver_stop"] == "nodes"]
    if declined:
        assert all(m["solver_predicted_nodes"] is not None for m in declined), (
            "declines carry no prediction, so the residual cannot be measured "
            "on the rows a different bar would move"
        )


def test_a_move_with_no_solve_carries_no_prediction():
    """A 0 would read as 'predicted to cost nothing'."""

    records = _records(0, max_nodes=200_000, full_fraction=0.3)
    unattempted = [m for m in _moves(records) if not m["solver_attempted"]]
    assert unattempted
    assert all(m["solver_predicted_nodes"] is None for m in unattempted)
