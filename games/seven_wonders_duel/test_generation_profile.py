"""The generation profile must read the log the run actually writes.

Two failure modes matter more than the arithmetic. A share computed against the
wrong denominator looks plausible at any value, and a hint whose input is always
zero can never fire -- so the tests pin the denominator and drive each hint from
both sides.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .generation_profile import (
    main,
    manifest_batch_cap,
    profile_row,
    render,
)


def _row(iteration=0, wall_ns=1_000, **scheduler):
    """A log row shaped like Phase D's, with one shard-second of wall."""

    base = {
        "scheduler_wall_ns": wall_ns,
        "sched_solve_wait_ns": 0,
        "py_call_ns": 0,
        "rust_tree_ns": 0,
        "queue_wait_ns": 0,
        "encode_pack_ns": 0,
        "live_slot_ns": 0,
        "batch_rows": [40, 44, 42],
        "max_batch_rows": 44,
        "max_active_slots": 256,
        "max_live_slots": 200,
        "scheduler_workers": 4,
    }
    base.update(scheduler)
    return {
        "iteration": iteration,
        "generation_performance": {
            "seconds": 3000.0,
            "games_per_second": 0.32,
            "rust_scheduler": base,
            "summary": {
                "solver": {"attempted": 11047, "masked": 10522, "nodes": 5, "stops": {}}
            },
        },
    }


def test_shares_are_fractions_of_shard_wall_not_of_wall_clock():
    """The counters sum across shards and so does the envelope; dividing by the
    iteration's wall-clock seconds instead would understate every share by the
    worker count, which at 4 shards is a 4x error that still looks like a
    percentage."""

    profile = profile_row(_row(wall_ns=1_000, sched_solve_wait_ns=250))
    assert profile["solve_wait_share"] == pytest.approx(0.25)


def test_a_row_without_scheduler_metrics_is_skipped_not_zeroed():
    """A Python-backend or bot-only iteration recorded no scheduler block.
    Reporting it as 0% everywhere would read as a healthy iteration."""

    assert profile_row({"iteration": 3, "generation_performance": {}}) is None
    assert profile_row({"iteration": 3}) is None


def test_mean_concurrency_is_measured_not_the_configured_cap():
    profile = profile_row(_row(wall_ns=1_000, live_slot_ns=42_000))
    assert profile["mean_live_slots"] == pytest.approx(42.0)
    assert profile["slot_cap"] == 256


def test_the_solver_hint_fires_only_when_solves_are_on_the_critical_path():
    blocked = render([profile_row(_row(wall_ns=1_000, sched_solve_wait_ns=300))])
    assert "SOLVER IS ON THE CRITICAL PATH" in blocked

    fine = render([profile_row(_row(wall_ns=1_000, sched_solve_wait_ns=10))])
    assert "SOLVER IS ON THE CRITICAL PATH" not in fine
    assert "not blocking generation" in fine


def test_a_deadline_stop_is_always_called_out():
    """A deadline decline makes which positions got a proof depend on machine
    load, so the buffer stops being a function of its seeds."""

    row = _row()
    row["generation_performance"]["summary"]["solver"]["stops"] = {"deadline": 7}
    assert "DEADLINE" in render([profile_row(row)])
    assert "DEADLINE" not in render([profile_row(_row())])


def test_the_batch_width_hint_needs_a_cap_and_uses_it():
    """`global_batch_cap` is not exported by the scheduler, so it comes from the
    manifest. Read as zero the comparison silently passes at every batch size --
    the hint would be dead code rather than a check."""

    narrow = render([profile_row(_row(), batch_cap=2048)])
    assert "paid per call" in narrow

    wide = render([profile_row(_row(), batch_cap=64)])
    assert "paid per call" not in wide

    dead = render([profile_row(_row(), batch_cap=0)])
    assert "paid per call" not in dead


def test_the_batch_cap_is_read_from_the_run_manifest(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"config": {"rust_global_batch_cap": 2048}}), encoding="utf-8"
    )
    assert manifest_batch_cap(tmp_path) == 2048
    assert manifest_batch_cap(tmp_path / "absent") == 0


def test_a_half_written_row_does_not_kill_the_tool(tmp_path, capsys):
    """This is meant to be run against a LIVE run, whose last line may be
    partial at the moment it is read."""

    log = tmp_path / "training_log.jsonl"
    log.write_text(
        json.dumps(_row(iteration=0)) + "\n" + '{"iteration": 1, "generation_per',
        encoding="utf-8",
    )
    assert main([str(tmp_path)]) == 0
    assert "0.320" in capsys.readouterr().out


def test_it_refuses_a_directory_that_is_not_a_run(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path)])


# ── Solver capacity ──────────────────────────────────────────────────────────
# "Does the solving fit inside the iteration" is the question the CPU split
# turns on. The scheduler never records solver BUSY time -- only the time it
# blocked -- so this is an estimate from the node count, and must say so.


def test_the_capacity_estimate_is_skipped_rather_than_guessed():
    """Without a measured node rate there is no honest answer. Printing one
    anyway would put a fabricated utilisation next to measured shares."""

    from .generation_profile import solver_budget

    profile = profile_row(_row())
    assert solver_budget(profile, node_rate=0.0, threads_total=12) is None
    assert solver_budget(profile, node_rate=1e6, threads_total=0) is None
    assert "ESTIMATED" not in render([profile])


def test_the_capacity_estimate_flags_solving_that_does_not_fit():
    from .generation_profile import solver_budget

    row = _row()
    row["generation_performance"]["summary"]["solver"]["nodes"] = 10**11
    profile = profile_row(row)
    # 1e11 nodes at 1e6/s/thread = 100,000 thread-seconds, against
    # 12 threads x 3000s = 36,000 available.
    budget = solver_budget(profile, node_rate=1e6, threads_total=12)
    assert budget["utilisation"] == pytest.approx(100_000 / 36_000)
    assert "does NOT fit" in render([profile], node_rate=1e6, solver_threads_total=12)


def test_the_capacity_estimate_flags_cores_bought_and_unused():
    row = _row()
    row["generation_performance"]["summary"]["solver"]["nodes"] = 10**9
    text = render([profile_row(row)], node_rate=1e6, solver_threads_total=12)
    assert "mostly idle" in text
    assert "ESTIMATED" in text, "an estimate must never read as a measurement"
