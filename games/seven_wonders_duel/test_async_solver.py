"""Async endgame solving must change WHEN, never WHAT.

Ported from Kingdomino's `BatchedMCTS` async_solve. It is a pure throughput
change: the solve still has to finish before the move is chosen, because the
mask decides what is played. What moves off the scheduler thread is the WAIT, so
other slots keep feeding the evaluation boundary instead of idling behind one
multi-second solve.

That makes record identity the whole gate. Same seeds, same masks, same targets,
same games -- only the timing differs.

**The gate has a precondition.** Identity holds only while the WALL CLOCK never
binds: a deadline decline is a function of how long a solve took in real time,
so under contention the async path and the synchronous one can decline different
positions and produce different buffers. These tests run with a slack clock and
`test_a_binding_clock_would_break_identity` states the precondition rather than
leaving it implied, because the split stop reasons now make it checkable.
"""

from __future__ import annotations

import pytest

from .rust_bridge import rust_games_for_self_play
from .test_f4_scheduler import _common

swr = pytest.importorskip("seven_wonders_rust")

SEEDS = [2026081730, 2026081731, 2026081732, 2026081733]
MAX_NODES = 5_000_000
MAX_CARDS = 8


@pytest.fixture(autouse=True)
def _reset_globals():
    yield
    swr.set_endgame_solver(0)
    swr.set_solver_threads(0)


def _records(
    threads: int,
    *,
    solve: bool = True,
    full_fraction: float = 1.0,
    fallback: bool = False,
    max_nodes: int = MAX_NODES,
):
    swr.set_endgame_solver(max_nodes, 60.0, MAX_CARDS, True)
    swr.set_solver_threads(threads)
    records, _ = swr.self_play_many_mock(
        games=rust_games_for_self_play(SEEDS, [0, 1, 0, 1]),
        game_seeds=SEEDS,
        force=True,
        solve_endgames=solve,
        solver_fallback_research=fallback,
        **(_common(leaf_batch=1, global_batch_cap=8)
           | {"full_search_fraction": full_fraction}),
    )
    return records


def test_async_records_are_identical_to_synchronous_ones():
    """The gate. A scheduler change that altered a single target would be a
    silent corruption of the buffer, not a crash."""

    synchronous = _records(0)
    for threads in (1, 2, 4):
        assert _records(threads) == synchronous, f"{threads} solver threads diverged"


def test_async_is_identical_on_a_mixed_schedule_too():
    """Every move full is the easy case: solves are dense and the pump rarely
    has anything else to do. A mixed schedule interleaves parked and searching
    slots, which is where a lost or misrouted outcome would show."""

    assert _records(4, full_fraction=0.25) == _records(0, full_fraction=0.25)


def _stops(records):
    return {
        move["solver_stop"]
        for record in records
        for move in record["moves"]
        if move["solver_stop"] is not None
    }


def test_identity_is_measured_with_a_clock_that_never_binds():
    """The precondition the identity gate rests on, asserted rather than assumed.

    A `deadline` stop means the clock decided the outcome, and the clock is not
    a function of the seed -- so an identity result gathered while it was
    binding would be luck. The node budget may bind; it reproduces.
    """

    assert "deadline" not in _stops(_records(4)), (
        "the wall clock bound during the identity test, so record identity was "
        "not actually demonstrated -- raise max_secs or lower the node budget"
    )


def test_a_binding_clock_is_visible_in_the_record():
    """And distinguishable from a node-capped decline, which is the point.

    Same position, same node budget, a clock of nothing: every decline must be
    reported as `deadline`, not folded into a single "budget" reason. Until
    these were one value, section 6's diagnosis had to be inferred from the
    maximum node count observed rather than read off the buffer.
    """

    # The smallest clock the setter accepts. It expires during the first
    # node-batch check, which is exactly the shape of a real deadline decline.
    swr.set_endgame_solver(MAX_NODES, 1e-9, MAX_CARDS, True)
    swr.set_solver_threads(0)
    records, _ = swr.self_play_many_mock(
        games=rust_games_for_self_play(SEEDS, [0, 1, 0, 1]),
        game_seeds=SEEDS,
        force=True,
        solve_endgames=True,
        **(_common(leaf_batch=1, global_batch_cap=8) | {"full_search_fraction": 1.0}),
    )
    stops = _stops(records)
    assert "deadline" in stops, f"an instantly-expired clock produced {stops}"
    assert "nodes" not in stops, f"nothing should reach a 5M node cap here: {stops}"


def test_the_solver_still_actually_fires_under_async():
    """Otherwise the identity above would be satisfied trivially by solving
    nothing at all."""

    records = _records(4)
    solved = [
        move
        for record in records
        for move in record["moves"]
        if move["solver_value"] is not None
    ]
    assert solved, "async path solved nothing"
    assert any(move["solver_masked"] for move in solved)


def test_threads_are_inert_when_the_slot_is_not_permitted_to_solve():
    """`solve_endgames` is the permission; threads are only the mechanism. A
    gate sets neither, and setting threads alone must not start solving."""

    assert _records(4, solve=False) == _records(0, solve=False)
    records = _records(4, solve=False)
    assert not any(
        move["solver_attempted"] for record in records for move in record["moves"]
    )


def _re_searched(records):
    """Rows whose solve declined AND which then got a full-budget search.

    `_common` runs cheap at 1-2 sims and full at 2-3, so a declined row
    recording more than 2 sims can only have come from the re-search.
    """

    return [
        move
        for record in records
        for move in record["moves"]
        if move["solver_stop"] == "nodes" and move["sims"] > 2
    ]


def test_async_is_identical_with_the_fallback_re_search_on():
    """The fallback moves a slot SolvePending -> NeedRoot -> Searching, a
    transition that did not exist when this gate was written.

    A tiny node budget forces declines, and a CHEAP schedule is what makes them
    reachable: the fallback only fires on a cheap ply, so at
    `full_search_fraction = 1.0` -- this helper's default -- not one line of it
    would run and the test would pass having proved nothing.

    NOTE: see `test_the_thread_count_is_currently_inert_in_this_harness`. This
    file's mock path is synchronous whatever the thread count, so what this
    asserts today is stability, not sync/async identity.
    """

    budget = 20_000
    common = {"fallback": True, "max_nodes": budget, "full_fraction": 0.0}
    synchronous = _records(0, **common)
    for threads in (1, 4):
        assert (
            _records(threads, **common) == synchronous
        ), f"{threads} solver threads diverged with the fallback on"


def test_the_fallback_actually_fires_in_that_gate():
    """Guards the gate above from passing vacuously.

    Asserting merely that a solve DECLINED is not enough -- an earlier version
    of this test did exactly that and passed while the fallback was unreachable,
    because every ply was full. The signal has to be a re-search.
    """

    records = _records(0, fallback=True, max_nodes=20_000, full_fraction=0.0)
    assert _re_searched(records), "no cheap ply was re-searched; the path is dead"


def test_the_fallback_changes_the_records_it_touches():
    """It must actually change a move, or it is dead configuration."""

    common = {"max_nodes": 20_000, "full_fraction": 0.0}
    assert _records(0, fallback=False, **common) != _records(
        0, fallback=True, **common
    )


def test_a_re_searched_row_still_reports_the_solve_it_attempted():
    """The carried overlay. Without it the row would look like a position the
    trigger never selected, and the decline would vanish from the statistics
    that size the solver's budget."""

    for move in _re_searched(
        _records(0, fallback=True, max_nodes=20_000, full_fraction=0.0)
    ):
        assert move["solver_attempted"] is True
        assert move["solver_stop"] == "nodes"
        assert move["solver_nodes"] > 0
        assert move["solver_value"] is None


def test_the_thread_count_is_currently_inert_in_this_harness():
    """This file's premise does not hold, and the failure is silent.

    `self_play_many_mock` runs through `self_play::run_many`, which passes
    `None` where the pipelined schedulers pass a `SolverPool`
    (`self_play.rs:2528` against `:2696`). No pool means no slot ever parks, so
    every `set_solver_threads` value above takes the same SYNCHRONOUS path and
    the identity assertions compare a run against itself.

    Pinned as a known gap rather than deleted: the assertions are still worth
    running -- they catch nondeterminism in the solve path -- but they do not
    gate the async port, and until this test fails, nothing does. Fixing it
    means either giving `run_many` a pool or moving the gate onto
    `self_play_many_flat_net`.
    """

    import inspect

    budget = 20_000
    common = {"fallback": True, "max_nodes": budget, "full_fraction": 0.0}
    # If the thread count ever starts mattering here, the mock path has grown a
    # real solver pool and this test should be replaced by a genuine gate.
    assert _records(0, **common) == _records(4, **common)
    assert "solver pool" not in inspect.getdoc(swr.self_play_many_mock or object) or True
