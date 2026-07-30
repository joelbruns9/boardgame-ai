"""W6.4: refuse an unrunnable box at setup rather than at 3 a.m.

Two independent budgets, both sized against what the run will *become* rather
than what it starts as:

* **Host RSS.** The replay window grows on a schedule (W1.1), so the number that
  matters is the window at its cap, not the one the first iteration uses. The
  example cache is bounded in calibrated retained bytes (W2.3), and the
  calibration factor exists because summing ``nbytes`` understates the real cost
  by ~26%.
* **Device VRAM.** W0 measured L at 7,978 of 8,192 MiB physical on an 8 GB
  laptop -- it fits, barely, with no room for the gate's second model. For
  vast.ai this is a hard instance filter, not a preference.

The sizing is a pure function so it can be tested without a GPU; only
:func:`device_report` touches CUDA.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys

GIB = 1024**3

# A1, measured through the real derivation path on 100 games from a real buffer.
RECORD_BYTES = 122 * 1024
"""Retained bytes per ``GameRecord`` held in the replay window."""

EXAMPLE_BYTES = 17_800
"""Retained bytes per ``Example``; ``nbytes`` says 13.1 KB and is wrong."""

PROCESS_OVERHEAD_BYTES = 2 * GIB
"""Interpreter, torch, CUDA context, and the Rust engine's own arenas.

Deliberately a flat figure: it does not scale with the window or the cache, and
pretending to model it precisely would make the whole estimate look more exact
than it is.
"""

L_MIN_VRAM_BYTES = 16 * GIB
"""W6.4's hard floor for the shipped 384x8x6 model."""

L_D_MODEL = 384


@dataclass(frozen=True)
class HostSizing:
    max_window_games: int
    window_bytes: int
    cache_bytes: int
    overhead_bytes: int
    headroom_bytes: int
    required_bytes: int
    budget_bytes: int

    @property
    def fits(self) -> bool:
        return self.budget_bytes <= 0 or self.required_bytes <= self.budget_bytes

    def as_dict(self) -> dict[str, object]:
        return {
            "max_window_games": self.max_window_games,
            "window_bytes": self.window_bytes,
            "cache_bytes": self.cache_bytes,
            "overhead_bytes": self.overhead_bytes,
            "headroom_bytes": self.headroom_bytes,
            "required_bytes": self.required_bytes,
            "budget_bytes": self.budget_bytes,
            "fits": self.fits,
        }


def host_sizing(
    *,
    max_window_games: int,
    example_cache_bytes: int,
    memory_budget_bytes: int,
    headroom_bytes: int,
) -> HostSizing:
    """Peak host RSS the run will reach once every schedule is at its cap."""

    window_bytes = int(max_window_games) * RECORD_BYTES
    required = (
        window_bytes
        + int(example_cache_bytes)
        + PROCESS_OVERHEAD_BYTES
        + int(headroom_bytes)
    )
    return HostSizing(
        max_window_games=int(max_window_games),
        window_bytes=window_bytes,
        cache_bytes=int(example_cache_bytes),
        overhead_bytes=PROCESS_OVERHEAD_BYTES,
        headroom_bytes=int(headroom_bytes),
        required_bytes=required,
        budget_bytes=int(memory_budget_bytes),
    )


def device_floor_bytes(d_model: int) -> int:
    """The VRAM this width may not launch below."""

    return L_MIN_VRAM_BYTES if d_model >= L_D_MODEL else 0


def device_report(device: str) -> dict[str, object]:
    """Physical device memory, or an explicit 'no device' report."""

    if not device.startswith("cuda"):
        return {"device": device, "available": False, "total_bytes": 0}
    try:
        import torch
    except ImportError:
        return {"device": device, "available": False, "total_bytes": 0}
    if not torch.cuda.is_available():
        return {"device": device, "available": False, "total_bytes": 0}
    _free, total = torch.cuda.mem_get_info()
    return {
        "device": device,
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "total_bytes": int(total),
    }


def evaluate(
    args, device_info: dict[str, object] | None = None
) -> tuple[dict[str, object], list[str]]:
    cache_bytes = (
        int(args.example_cache_gb * GIB)
        if args.example_cache_gb > 0
        else int(args.example_cache_examples * EXAMPLE_BYTES)
    )
    budget_bytes = int(args.memory_budget_gb * GIB)
    if budget_bytes <= 0:
        try:
            import psutil

            budget_bytes = int(psutil.virtual_memory().total * 0.85)
        except ImportError:
            budget_bytes = 0
    sizing = host_sizing(
        max_window_games=args.replay_window_cap_games,
        example_cache_bytes=cache_bytes,
        memory_budget_bytes=budget_bytes,
        headroom_bytes=int(args.memory_headroom_gb * GIB),
    )
    device = device_info if device_info is not None else device_report(args.device)
    # The floor is about the GPU the run will actually use. A deliberate CPU run
    # is not subject to it; a CUDA run that cannot see a device is.
    floor = device_floor_bytes(args.d_model) if args.device.startswith("cuda") else 0

    failures: list[str] = []
    if not sizing.fits:
        failures.append(
            f"host memory: the run needs about "
            f"{sizing.required_bytes / GIB:.1f} GiB at its maximum scheduled "
            f"window of {sizing.max_window_games:,} games "
            f"({sizing.window_bytes / GIB:.1f} GiB of records + "
            f"{sizing.cache_bytes / GIB:.1f} GiB of example cache + "
            f"{sizing.overhead_bytes / GIB:.1f} GiB process + "
            f"{sizing.headroom_bytes / GIB:.1f} GiB headroom) but the budget is "
            f"{sizing.budget_bytes / GIB:.1f} GiB. Lower "
            f"--replay-window-cap-games or --example-cache-gb, or rent more RAM."
        )
    if floor:
        if not device["available"]:
            failures.append(
                f"device: {args.d_model}x model requires "
                f"{floor / GIB:.0f} GiB of VRAM and no CUDA device is visible"
            )
        elif int(device["total_bytes"]) < floor:
            failures.append(
                f"device: {device.get('name', args.device)} has "
                f"{int(device['total_bytes']) / GIB:.1f} GiB of VRAM; the "
                f"{args.d_model}-wide model requires {floor / GIB:.0f} GiB. "
                "This is an instance filter, not a preference -- destroy this "
                "instance and rent one with more VRAM."
            )
    report = {
        "host": sizing.as_dict(),
        "device": {**device, "required_bytes": floor},
        "model": {"d_model": args.d_model, "layers": args.layers, "heads": args.heads},
        "passed": not failures,
        "failures": failures,
    }
    return report, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W6.4 launch preflight")
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replay-window-cap-games", type=int, default=20_000)
    parser.add_argument("--example-cache-gb", type=float, default=0.0)
    parser.add_argument("--example-cache-examples", type=int, default=250_000)
    parser.add_argument("--memory-budget-gb", type=float, default=0.0)
    parser.add_argument("--memory-headroom-gb", type=float, default=2.0)
    parser.add_argument("--output", help="write the report as JSON")
    args = parser.parse_args(argv)

    report, failures = evaluate(args)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    host = report["host"]
    print(
        f"host: needs {host['required_bytes'] / GIB:.1f} GiB at the "
        f"{host['max_window_games']:,}-game window cap, budget "
        f"{host['budget_bytes'] / GIB:.1f} GiB"
    )
    device = report["device"]
    if device["available"]:
        print(
            f"device: {device.get('name')} "
            f"{int(device['total_bytes']) / GIB:.1f} GiB, floor "
            f"{int(device['required_bytes']) / GIB:.0f} GiB"
        )
    else:
        print(f"device: {device['device']} not available")
    if failures:
        print("\nPREFLIGHT FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\npreflight passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
