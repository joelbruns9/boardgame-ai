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


def _row(iteration=0, wall_ns=1_000, nested=True, **scheduler):
    """A log row, with one shard-second of wall.

    `nested=True` is the shape the CLOUD RUNS write: training_adapter reports
    {"performance": ..., "summary": ..., "model": ...} and az_loop stores it
    verbatim. `nested=False` is Phase D's own flat shape. The first version of
    this file only had the flat one -- invented from the consumer rather than
    read off the producer -- so the profiler reported "no scheduler metrics"
    against a real run while every test passed.
    """

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
        "mean_wave_width": 1.0,
    }
    base.update(scheduler)
    performance = {
        "seconds": 3000.0,
        "games_per_second": 0.32,
        "rust_scheduler": base,
    }
    summary = {
        "solver": {"attempted": 11047, "masked": 10522, "nodes_total": 5,
                   "nodes_on_declines": 0, "stops": {}}
    }
    if nested:
        reported = {"performance": performance, "summary": summary, "model": {}}
    else:
        reported = {**performance, "summary": summary}
    return {"iteration": iteration, "generation_performance": reported}


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


def test_the_batch_width_hint_does_not_tell_the_operator_to_chase_width():
    """Measured 2026-08-20: 2 shards beat 1 shard by 19% at HALF the batch
    width. The hint used to say "raise concurrency or lower the shard count",
    which the sweep contradicted. A tool that prints advice must not print
    advice its own project has measured to be wrong."""

    text = render([profile_row(_row(), batch_cap=2048)])
    assert "lower the shard count" not in text
    assert "does not predict throughput" in text


def test_the_batch_width_hint_needs_a_cap_and_uses_it():
    """`global_batch_cap` is not exported by the scheduler, so it comes from the
    manifest. Read as zero the comparison silently passes at every batch size --
    the hint would be dead code rather than a check."""

    narrow = render([profile_row(_row(), batch_cap=2048)])
    assert "cap is not binding" in narrow

    wide = render([profile_row(_row(), batch_cap=64)])
    assert "cap is not binding" not in wide

    dead = render([profile_row(_row(), batch_cap=0)])
    assert "cap is not binding" not in dead


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
    row["generation_performance"]["summary"]["solver"]["nodes_total"] = 10**11
    profile = profile_row(row)
    # 1e11 nodes at 1e6/s/thread = 100,000 thread-seconds, against
    # 12 threads x 3000s = 36,000 available.
    budget = solver_budget(profile, node_rate=1e6, threads_total=12)
    assert budget["utilisation"] == pytest.approx(100_000 / 36_000)
    assert "does NOT fit" in render([profile], node_rate=1e6, solver_threads_total=12)


def test_the_capacity_estimate_flags_cores_bought_and_unused():
    row = _row()
    row["generation_performance"]["summary"]["solver"]["nodes_total"] = 10**9
    text = render([profile_row(row)], node_rate=1e6, solver_threads_total=12)
    assert "mostly idle" in text
    assert "ESTIMATED" in text, "an estimate must never read as a measurement"


def test_both_row_shapes_are_read():
    """The nested shape is what the cloud writes; the flat one is Phase D's own
    loop. Reading only one reports a healthy run as having no metrics."""

    for nested in (True, False):
        profile = profile_row(_row(nested=nested, wall_ns=1_000, sched_solve_wait_ns=250))
        assert profile is not None, f"nested={nested} row was not read"
        assert profile["solve_wait_share"] == pytest.approx(0.25)
        assert profile["games_per_second"] == pytest.approx(0.32)
        assert profile["solver_attempted"] == 11047


def test_the_nested_key_matches_what_the_adapter_actually_reports():
    """Pinned to the PRODUCER, not to this file's fixture.

    `training_adapter.generate` builds the metrics dict; if its key names change,
    the profiler goes blind in exactly the way it did on the first cloud run --
    silently, with a message that reads like the run's fault.
    """

    source = (
        Path(__file__).resolve().parent / "training_adapter.py"
    ).read_text(encoding="utf-8")
    block = source[source.index("return GenerationResult(") :]
    block = block[: block.index("def assemble_replay")]
    for key in ('"performance": dict(loop.last_generation_stats)', '"summary": summarize_records'):
        assert key in block, (
            f"training_adapter no longer reports {key!r}; generation_profile "
            "reads that key path and will report no metrics without it"
        )


def test_an_unreadable_log_reports_what_it_did_find(tmp_path, capsys):
    """The failure message must name keys, not just say no.

    "no iteration recorded scheduler metrics" was true, uninformative, and
    worded like the run's fault while the reader was looking one nesting level
    too shallow. It cost a debugging round trip on a rented box.
    """

    row = {
        "iteration": 4,
        "generation_performance": {"performance": {"seconds": 1.0}, "summary": {}},
        "stats": {"generation": {"games_per_second": 0.3}},
    }
    log = tmp_path / "training_log.jsonl"
    log.write_text(json.dumps(row), encoding="utf-8")

    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    # Names the top-level keys, the nested keys, and where to look.
    assert "generation_performance keys" in out
    assert "'performance'" in out and "'summary'" in out
    assert "performance: ['seconds']" in out
    assert "stats.generation" in out


def test_the_diagnosis_says_so_when_the_key_is_absent_entirely(tmp_path, capsys):
    log = tmp_path / "training_log.jsonl"
    log.write_text(json.dumps({"iteration": 1}), encoding="utf-8")
    assert main([str(tmp_path)]) == 0
    assert "NO 'generation_performance' key" in capsys.readouterr().out


def test_phase_d_config_defaults_are_not_readable_as_class_attributes():
    """PhaseDConfig is @dataclass(slots=True), so PhaseDConfig.rust_slots is a
    slot descriptor rather than 16. Code that compared class attributes to
    measured values never matched anything and silently took its fallback."""

    import dataclasses

    from . import phase_d as pd
    from .f4_phase_d_sweep import field_default

    assert not isinstance(getattr(pd.PhaseDConfig, "rust_slots"), int)
    assert field_default("rust_slots") == next(
        f.default for f in dataclasses.fields(pd.PhaseDConfig) if f.name == "rust_slots"
    )
    assert isinstance(field_default("rust_slots"), int)


def test_wave_width_is_reported_and_flagged_when_it_is_one():
    """Wave width is leaves in flight for ONE game; batch width is leaves summed
    across games. At 1.00 a 1600-sim move is 1600 sequential round trips, and no
    amount of slots or workers can lift rows-per-live-game above 1."""

    one = render([profile_row(_row(mean_wave_width=1.0))])
    assert "wave width 1.00" in one
    assert "leaf-batch" in one

    batched = render([profile_row(_row(mean_wave_width=8.0))])
    assert "wave width" not in batched

    # Absent from an older metrics block: report nothing rather than 0.
    missing = _row()
    del missing["generation_performance"]["performance"]["rust_scheduler"]["mean_wave_width"]
    assert "wave width" not in render([profile_row(missing)])


def test_the_solver_node_key_matches_what_summarize_solver_emits():
    """Pinned to the PRODUCER, not to this file's fixture.

    The profiler read `nodes`; `_summarize_solver` emits `nodes_total`. The
    capacity estimate therefore reported an idle solver on a run performing
    ~10,000 solves an iteration -- a wrong answer that looked like a
    measurement, on the exact question it was built to answer.
    """

    from pathlib import Path

    source = (Path(__file__).with_name("phase_d.py")).read_text(encoding="utf-8")
    block = source[source.index("def _summarize_solver") :]
    block = block[: block.index("\ndef ")]
    for key in ("nodes_total", "nodes_on_declines", "attempted", "masked"):
        assert f'"{key}"' in block, f"_summarize_solver no longer emits {key}"

    profile = profile_row(_row())
    assert profile["solver_nodes"] == 5, "the fixture's nodes_total must be read"


def test_a_solver_doing_work_is_not_reported_as_idle():
    """The failure that reached the box: nodes read as 0, so the estimate said
    0% of the pool on an iteration with ~10,000 solves."""

    row = _row()
    row["generation_performance"]["summary"]["solver"]["nodes_total"] = 2_000_000_000
    text = render([profile_row(row)], node_rate=2e6, solver_threads_total=14)
    assert "ESTIMATED from 2,000,000,000 nodes" in text
    # Substring checks are a trap here: "1,000 thread-seconds" contains
    # "0 thread-seconds". Assert the computed value instead.
    from .generation_profile import solver_budget

    budget = solver_budget(profile_row(row), node_rate=2e6, threads_total=14)
    assert budget["thread_seconds"] == pytest.approx(1000.0)
    assert budget["utilisation"] > 0
