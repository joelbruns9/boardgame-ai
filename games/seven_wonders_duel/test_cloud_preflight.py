"""W6.4: the preflight sizes the run at its cap, not at its first iteration."""

from __future__ import annotations

import argparse
import json
import os

from .cloud_preflight import (
    EXAMPLE_BYTES,
    GIB,
    L_MIN_VRAM_BYTES,
    RECORD_BYTES,
    device_floor_bytes,
    disk_sizing,
    effective_memory_bytes,
    evaluate,
    host_sizing,
)


def _device(total_gib: float | None):
    """An explicit device report so tests never depend on the local GPU."""

    if total_gib is None:
        return {"device": "cuda", "available": False, "total_bytes": 0}
    return {
        "device": "cuda",
        "available": True,
        "name": "test-gpu",
        "total_bytes": int(total_gib * GIB),
    }


def _args(**overrides):
    base = dict(
        d_model=384,
        layers=8,
        heads=6,
        device="cuda",
        replay_window_cap_games=20_000,
        example_cache_gb=4.0,
        example_cache_examples=250_000,
        memory_budget_gb=64.0,
        memory_headroom_gb=2.0,
        output=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_sizing_uses_the_window_cap_not_the_current_window():
    small = host_sizing(
        max_window_games=1_000,
        example_cache_bytes=0,
        memory_budget_bytes=0,
        headroom_bytes=0,
    )
    capped = host_sizing(
        max_window_games=20_000,
        example_cache_bytes=0,
        memory_budget_bytes=0,
        headroom_bytes=0,
    )
    assert capped.window_bytes == 20_000 * RECORD_BYTES
    assert capped.required_bytes - small.required_bytes == 19_000 * RECORD_BYTES


def test_a_box_that_fits_today_but_not_at_the_cap_is_refused():
    # 20k games of records is ~2.3 GiB, plus a 4 GiB cache, 2 GiB of process and
    # 2 GiB of headroom: comfortable at 16 GiB, refused at 8.
    _report, failures = evaluate(_args(memory_budget_gb=16.0), _device(24))
    assert not failures

    _report, failures = evaluate(_args(memory_budget_gb=8.0), _device(24))
    assert any("host memory" in failure for failure in failures)
    assert any("maximum scheduled window" in failure for failure in failures)


def test_the_cache_flag_and_the_legacy_count_agree_on_the_same_ceiling():
    # 250k examples x 17.8 KB is 4.45 GB decimal, which is 4.144 GiB; the flag
    # is in GiB like every other memory flag in the loop.
    by_count, _ = evaluate(
        _args(example_cache_gb=0.0, example_cache_examples=250_000), _device(24)
    )
    by_bytes, _ = evaluate(
        _args(example_cache_gb=250_000 * EXAMPLE_BYTES / GIB), _device(24)
    )
    assert abs(
        by_bytes["host"]["cache_bytes"] - by_count["host"]["cache_bytes"]
    ) < 0.01 * GIB


def test_l_has_a_hard_vram_floor_and_smaller_models_do_not():
    assert device_floor_bytes(384) == L_MIN_VRAM_BYTES
    assert device_floor_bytes(128) == 0


def test_l_is_refused_when_no_device_is_visible():
    _report, failures = evaluate(_args(), _device(None))
    assert any("no CUDA device" in failure for failure in failures)


def test_l_is_refused_on_an_eight_gigabyte_box():
    # W0 measured L at 7,978 of 8,192 MiB for a single training model; a gate
    # holds two. This is the instance filter that failure implies.
    _report, failures = evaluate(_args(), _device(8))
    assert any("requires 16 GiB" in failure for failure in failures)
    _report, failures = evaluate(_args(), _device(24))
    assert not failures


def test_s_fallback_does_not_trip_the_vram_floor():
    _report, failures = evaluate(
        _args(d_model=128, layers=4, heads=4), _device(8)
    )
    assert not any("device:" in failure for failure in failures)


def test_a_deliberate_cpu_run_is_not_subject_to_the_vram_floor():
    _report, failures = evaluate(_args(device="cpu"), _device(None))
    assert not failures


def test_report_records_the_model_it_sized_for():
    report, _ = evaluate(_args(memory_budget_gb=64.0), _device(24))
    assert report["model"]["d_model"] == 384
    assert report["model"]["layers"] == 8
    assert report["model"]["heads"] == 6
    assert report["device"]["required_bytes"] == L_MIN_VRAM_BYTES


def test_the_report_can_be_written_before_the_run_directory_exists(tmp_path):
    """The preflight runs *before* anything creates the run directory.

    Its natural output path is inside that directory, so writing the report
    raised FileNotFoundError on every fresh box -- and the launcher reported it
    as "Preflight refused this box. Destroy the instance and rent a bigger one."
    """

    from .cloud_preflight import main

    output = tmp_path / "runs" / "seven_wonders_duel" / "cloud" / "preflight.json"
    assert not output.parent.exists()
    status = main(
        [
            "--device",
            "cpu",
            "--memory-budget-gb",
            "64",
            "--output",
            str(output),
        ]
    )
    assert status == 0
    assert json.loads(output.read_text(encoding="utf-8"))["passed"]


def test_a_rented_slice_is_sized_by_its_cgroup_not_by_the_host():
    """vast.ai sells "48 of 192 cores and 64 GB"; the container sees 192/251 GB.

    `psutil.virtual_memory().total` reads /proc/meminfo, which is the host's, so
    a check written against it believes it has ~4x the memory the cgroup will
    actually allow -- disabling the refusal on precisely the machines it exists
    to protect.
    """

    host = 251 * GIB
    slice_limit = 64 * GIB
    assert effective_memory_bytes(host, slice_limit) == slice_limit
    # A cgroup ceiling above host memory is a no-op, not a promise.
    assert effective_memory_bytes(host, 512 * GIB) == host
    # No cgroup at all (a bare-metal box, or Windows) keeps the host figure.
    assert effective_memory_bytes(host, None) == host


def test_oversubscribed_threads_are_reported_without_failing_the_run():
    """Costs throughput, does not stop a run -- and `nproc` cannot reveal it."""

    from .cloud_preflight import thread_oversubscription_note

    # The real case: 192 threads visible, 48 sold.
    note = thread_oversubscription_note(192, 48.0)
    assert note is not None
    assert "192" in note and "48" in note and "~4.0x" in note
    assert "OMP_NUM_THREADS=12" in note

    # Dedicated box, or a quota close enough that the pool is not silly.
    assert thread_oversubscription_note(48, 48.0) is None
    assert thread_oversubscription_note(48, 40.0) is None
    # No cgroup quota at all: nothing is knowable, so say nothing.
    assert thread_oversubscription_note(192, None) is None


def test_the_oversubscription_note_reaches_the_report_and_fails_nothing():
    args = _args(memory_budget_gb=64.0)
    limits = {"memory_bytes": 64 * GIB, "cpus": 0.5}  # far below any real host
    report, failures = evaluate(args, _device(24), _disk(200), limits)
    assert any("oversubscribe" in note for note in report["advice"])
    assert not failures
    assert report["passed"]


def test_cgroup_files_are_parsed_and_sentinels_mean_no_limit(tmp_path, monkeypatch):
    from . import cloud_preflight

    unlimited = tmp_path / "unlimited"
    unlimited.write_text("max\n", encoding="utf-8")
    limited = tmp_path / "limited"
    limited.write_text(f"{64 * GIB}\n", encoding="utf-8")
    v1_sentinel = tmp_path / "v1"
    # cgroup v1 writes a value near 2^63 instead of "max".
    v1_sentinel.write_text("9223372036854771712\n", encoding="utf-8")

    read = cloud_preflight._read_first_int
    assert read((str(unlimited),)) is None
    assert read((str(v1_sentinel),)) is None
    assert read((str(limited),)) == 64 * GIB
    assert read((str(tmp_path / "absent"),)) is None
    # Falls through to the second path when the first is missing.
    assert read((str(tmp_path / "absent"), str(limited))) == 64 * GIB


def _disk(free_gib: float):
    return {"path": "/run", "available": True, "free_bytes": int(free_gib * GIB)}


def _plan(**overrides):
    base = dict(
        iterations=200,
        games_per_iteration=1_000,
        seed_games=5_000,
        parameters=14_900_000,
        promotion_every=5,
        disk_budget_bytes=0,
        headroom_bytes=5 * GIB,
    )
    base.update(overrides)
    return disk_sizing(**base)


def test_the_disk_estimate_is_dominated_by_unpruned_per_iteration_checkpoints():
    sizing = _plan()
    # 200 iterations x 2 files x ~60 MB is ~24 GB; game records over the same
    # run are ~6.6 GB. If that ordering ever inverts, the advice attached to
    # this budget ("rent a bigger disk") is aimed at the wrong term.
    assert sizing.checkpoint_bytes > 2 * sizing.buffer_bytes
    assert 20 * GIB < sizing.checkpoint_bytes < 25 * GIB
    assert 30 * GIB < sizing.required_bytes < 40 * GIB


def test_halving_the_iteration_size_doubles_the_checkpoint_bill():
    """Why the launcher runs 1,000-game iterations rather than 500.

    Checkpoints are per iteration and records are per game, so the same 200k
    games cost twice the checkpoints when split into twice as many iterations.
    """

    big = _plan(iterations=200, games_per_iteration=1_000)
    small = _plan(iterations=400, games_per_iteration=500)
    assert small.total_games == big.total_games
    assert small.checkpoint_bytes == 2 * big.checkpoint_bytes
    assert small.required_bytes > big.required_bytes + 20 * GIB


def test_a_box_with_too_little_disk_is_refused_before_the_run_starts():
    args = _args(
        memory_budget_gb=64.0,
        iterations=200,
        games_per_iteration=1_000,
        seed_games=5_000,
        promotion_every=5,
        parameters=14_900_000,
        disk_budget_gb=0.0,
        disk_headroom_gb=5.0,
        run_dir="/run",
    )
    _report, failures = evaluate(args, _device(24), _disk(30))
    assert any("disk:" in failure for failure in failures)
    assert any("Nothing here is prunable at runtime" in failure for failure in failures)

    _report, failures = evaluate(args, _device(24), _disk(120))
    assert not failures


def test_an_unsized_run_makes_no_disk_claim():
    """`--iterations 0` (the default) must not refuse a box on headroom alone."""

    sizing = _plan(iterations=0)
    assert sizing.required_bytes == 0
    assert sizing.fits

    _report, failures = evaluate(_args(memory_budget_gb=64.0), _device(24), _disk(1))
    assert not failures
