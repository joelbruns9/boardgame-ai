"""Async endgame solving must change WHEN, never WHAT.

Ported from Kingdomino's `BatchedMCTS` async_solve. It is a pure throughput
change: the solve still has to finish before the move is chosen, because the
mask decides what is played. What moves off the scheduler thread is the WAIT, so
other slots keep feeding the evaluation boundary instead of idling behind one
multi-second solve.

That makes record identity the whole gate. Same seeds, same masks, same targets,
same games -- only the timing differs.
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


def _records(threads: int, *, solve: bool = True, full_fraction: float = 1.0):
    swr.set_endgame_solver(MAX_NODES, 60.0, MAX_CARDS, True)
    swr.set_solver_threads(threads)
    records, _ = swr.self_play_many_mock(
        games=rust_games_for_self_play(SEEDS, [0, 1, 0, 1]),
        game_seeds=SEEDS,
        force=True,
        solve_endgames=solve,
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
