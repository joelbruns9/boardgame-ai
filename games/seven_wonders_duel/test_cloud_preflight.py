"""W6.4: the preflight sizes the run at its cap, not at its first iteration."""

from __future__ import annotations

import argparse

from .cloud_preflight import (
    EXAMPLE_BYTES,
    GIB,
    L_MIN_VRAM_BYTES,
    RECORD_BYTES,
    device_floor_bytes,
    disk_sizing,
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
