"""W6.4: the preflight sizes the run at its cap, not at its first iteration."""

from __future__ import annotations

import argparse

from .cloud_preflight import (
    EXAMPLE_BYTES,
    GIB,
    L_MIN_VRAM_BYTES,
    RECORD_BYTES,
    device_floor_bytes,
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
    assert report["model"] == {"d_model": 384, "layers": 8, "heads": 6}
    assert report["device"]["required_bytes"] == L_MIN_VRAM_BYTES
