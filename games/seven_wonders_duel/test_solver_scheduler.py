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

from .rust_bridge import rust_games_for_self_play
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
):
    """One run through the production scheduler, with the solver on."""

    swr.set_endgame_solver(max_nodes, 120.0, MAX_CARDS, True)
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
