"""W6.4: refuse an unrunnable box at setup rather than at 3 a.m.

Three independent budgets, all sized against what the run will *become* rather
than what it starts as:

* **Host RSS.** The replay window grows on a schedule (W1.1), so the number that
  matters is the window at its cap, not the one the first iteration uses. The
  example cache is bounded in calibrated retained bytes (W2.3), and the
  calibration factor exists because summing ``nbytes`` understates the real cost
  by ~26%.
* **Device VRAM.** W0 measured L at 7,978 of 8,192 MiB physical on an 8 GB
  laptop -- it fits, barely, with no room for the gate's second model. For
  vast.ai this is a hard instance filter, not a preference.
* **Disk.** Every iteration writes two checkpoints that are never pruned --
  ``candidate_NNNN.pt`` and the anchor's ``learner_NNNN.pt`` -- and an L
  checkpoint is 59.7 MB measured. A 200-iteration run therefore spends ~24 GB on
  checkpoints before a single game record is written. Disk is chosen when the
  instance is rented and is the one budget here that cannot be lowered by a
  flag afterwards.

The sizing is a pure function so it can be tested without a GPU; only
:func:`device_report` touches CUDA and only :func:`disk_report` touches the
filesystem.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
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

CHECKPOINT_BYTES_PER_PARAMETER = 4.03
"""Measured on both shipped widths: 4.15 MB at 1.03 M params, 59.66 MB at 14.9 M.

fp32 weights plus a small constant of config and metadata; the ratio is stable
enough across a 14x span that a per-parameter figure beats a hard-coded size.
"""

CHECKPOINTS_PER_ITERATION = 2
"""``candidate_NNNN.pt`` and ``learner_NNNN.pt``, neither of which is pruned.

The learner snapshot is the self-anchor's reference series (W7c), so it cannot
simply be deleted; the candidate is what makes an interrupted iteration
restartable. Both are kept for the life of the run.
"""

RECORD_DISK_BYTES = 32 * 1024
"""On-disk JSONL bytes per game record: 2,613 MB over 84,000 games in run 03."""

LOG_ROW_BYTES = 457 * 1024
"""One `training_log.jsonl` row: 96 MB over 210 rows in run 03."""


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


@dataclass(frozen=True)
class DiskSizing:
    iterations: int
    total_games: int
    parameters: int
    checkpoint_bytes: int
    hof_bytes: int
    buffer_bytes: int
    log_bytes: int
    headroom_bytes: int
    required_bytes: int
    budget_bytes: int

    @property
    def fits(self) -> bool:
        return self.budget_bytes <= 0 or self.required_bytes <= self.budget_bytes

    def as_dict(self) -> dict[str, object]:
        return {
            "iterations": self.iterations,
            "total_games": self.total_games,
            "parameters": self.parameters,
            "checkpoint_bytes": self.checkpoint_bytes,
            "hof_bytes": self.hof_bytes,
            "buffer_bytes": self.buffer_bytes,
            "log_bytes": self.log_bytes,
            "headroom_bytes": self.headroom_bytes,
            "required_bytes": self.required_bytes,
            "budget_bytes": self.budget_bytes,
            "fits": self.fits,
        }


def parameter_count(
    d_model: int,
    layers: int,
    heads: int,
    pooled_readout: bool = False,
    reply_head: bool = False,
) -> int:
    """Parameters of the model this run will build.

    Counted rather than assumed: W0 lost a run to a checkpoint whose width was
    inferred instead of read, and every disk figure here scales with it.
    """

    from .train import build_model

    # The architecture switches are part of the size. `--pooled-readout` adds a
    # 3d x d projection and `--reply-head` another head; leaving them out here
    # understates every disk and memory figure this module exists to produce,
    # which is the same class of miss as the six checkpoint rebuild sites --
    # a value crossing a boundary and silently defaulting.
    model = build_model(
        "transformer", d_model, layers, heads, pooled_readout, reply_head
    )
    return sum(parameter.numel() for parameter in model.parameters())


def disk_sizing(
    *,
    iterations: int,
    games_per_iteration: int,
    seed_games: int,
    parameters: int,
    promotion_every: int,
    disk_budget_bytes: int,
    headroom_bytes: int,
) -> DiskSizing:
    """Bytes the run will have written by its last iteration.

    Nothing in the run deletes anything, so this is a total rather than a peak.
    The HOF term assumes every scheduled gate promotes -- the archive keeps one
    checkpoint per promotion and prunes none, and being wrong in the cheap
    direction here costs a few GB of stated requirement rather than a dead run
    on day three.
    """

    iterations = max(0, int(iterations))
    if iterations == 0:
        # No planned length, no claim: an unsized disk check that failed on
        # headroom alone would refuse boxes for a run it knows nothing about.
        return DiskSizing(
            iterations=0,
            total_games=0,
            parameters=int(parameters),
            checkpoint_bytes=0,
            hof_bytes=0,
            buffer_bytes=0,
            log_bytes=0,
            headroom_bytes=0,
            required_bytes=0,
            budget_bytes=int(disk_budget_bytes),
        )
    total_games = max(0, int(seed_games)) + iterations * max(0, int(games_per_iteration))
    checkpoint = int(parameters * CHECKPOINT_BYTES_PER_PARAMETER)
    checkpoint_bytes = iterations * CHECKPOINTS_PER_ITERATION * checkpoint
    gates = iterations // max(1, int(promotion_every))
    hof_bytes = gates * checkpoint
    buffer_bytes = total_games * RECORD_DISK_BYTES
    log_bytes = iterations * LOG_ROW_BYTES
    required = (
        checkpoint_bytes
        + hof_bytes
        + buffer_bytes
        + log_bytes
        + int(headroom_bytes)
    )
    return DiskSizing(
        iterations=iterations,
        total_games=total_games,
        parameters=int(parameters),
        checkpoint_bytes=checkpoint_bytes,
        hof_bytes=hof_bytes,
        buffer_bytes=buffer_bytes,
        log_bytes=log_bytes,
        headroom_bytes=int(headroom_bytes),
        required_bytes=required,
        budget_bytes=int(disk_budget_bytes),
    )


def disk_report(path: str) -> dict[str, object]:
    """Free bytes on the filesystem the run directory will live on."""

    import os
    import shutil

    probe = os.path.abspath(path)
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        return {"path": probe, "available": False, "free_bytes": 0}
    return {"path": probe, "available": True, "free_bytes": int(usage.free)}


def _read_first_int(paths: tuple[str, ...]) -> int | None:
    """First readable file among `paths`, parsed as an integer.

    Returns None for "max"/unset/unparseable, which cgroup uses for no limit.
    """

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read().strip().split()[0]
        except (OSError, IndexError):
            continue
        if raw in ("max", "-1"):
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        # cgroup v1 writes a sentinel near 2^63 rather than "max".
        return None if value <= 0 or value >= 1 << 62 else value
    return None


def container_limits() -> dict[str, object]:
    """The cgroup's memory ceiling and CPU quota, when there is one.

    A rented box is usually a *slice* of a host: vast.ai sells "48 of 192 cores
    and 64 GB", and inside the container `psutil.virtual_memory().total` and
    `nproc` both report the **host's** figures. Sizing against those makes this
    check believe it has four times the memory the cgroup will actually allow,
    which disables the refusal on exactly the machines it exists to protect.
    """

    memory = _read_first_int(
        (
            "/sys/fs/cgroup/memory.max",  # v2
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # v1
        )
    )
    cpu_quota = None
    for path, period_path in (
        ("/sys/fs/cgroup/cpu.max", None),  # v2: "quota period" on one line
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                fields = handle.read().strip().split()
        except OSError:
            continue
        if not fields or fields[0] in ("max", "-1"):
            break
        try:
            quota = int(fields[0])
            if period_path is None:
                period = int(fields[1]) if len(fields) > 1 else 100_000
            else:
                with open(period_path, "r", encoding="utf-8") as handle:
                    period = int(handle.read().strip())
        except (ValueError, OSError, IndexError):
            break
        if quota > 0 and period > 0:
            cpu_quota = quota / period
        break
    cpuset = _cpuset_count()
    affinity = _affinity_count()
    return {
        "memory_bytes": memory,
        "cpus": cpu_quota,
        "cpuset_cpus": cpuset,
        "affinity_cpus": affinity,
        "effective_cpus": effective_cpu_count(cpu_quota, cpuset, affinity),
    }


def _parse_cpu_list(text: str) -> int | None:
    """Count CPUs in a cgroup/sysfs list such as ``0-3,8,12-15``."""

    text = text.strip()
    if not text:
        return None
    total = 0
    for part in text.split(","):
        if "-" in part:
            low, _, high = part.partition("-")
            try:
                total += int(high) - int(low) + 1
            except ValueError:
                return None
        else:
            try:
                int(part)
            except ValueError:
                return None
            total += 1
    return total or None


def _cpuset_count() -> int | None:
    """CPUs the cgroup's cpuset allows.

    A container constrained by **cpuset** commonly has no CFS quota at all, so
    reading `cpu.max` alone reports "unlimited" and callers fall back to the
    host's core count. That is the case this exists to catch.
    """

    for path in (
        "/sys/fs/cgroup/cpuset.cpus.effective",  # v2
        "/sys/fs/cgroup/cpuset/cpuset.effective_cpus",  # v1
        "/sys/fs/cgroup/cpuset.cpus",
    ):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                count = _parse_cpu_list(handle.read())
        except OSError:
            continue
        if count:
            return count
    return None


def _affinity_count() -> int | None:
    """CPUs this process may actually be scheduled on."""

    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:  # pragma: no cover - not on Windows/macOS
        return None
    try:
        return len(getter(0))
    except OSError:  # pragma: no cover
        return None


def effective_cpu_count(
    quota: float | None, cpuset: int | None, affinity: int | None
) -> int:
    """Usable parallelism: the **minimum** of every applicable limit.

    Oversubscribing a thread pool past this is the one regime where row-parallel
    packing loses rather than wins, and each limit can bind independently -- a
    slice can be quota-limited, cpuset-limited, affinity-limited, or any
    combination.
    """

    visible = os.cpu_count() or 1
    candidates = [visible]
    if quota:
        candidates.append(max(1, int(math.floor(quota))))
    if cpuset:
        candidates.append(cpuset)
    if affinity:
        candidates.append(affinity)
    return max(1, min(candidates))


def thread_oversubscription_note(
    visible_cpus: int, allowed_cpus: float | None
) -> str | None:
    """Warn when torch will size its thread pool from cores this run cannot use.

    Not a failure: oversubscription costs throughput, it does not stop a run.
    But it is invisible from inside the container -- `nproc` and `os.cpu_count()`
    both report the host -- so nothing else will ever tell the operator.
    """

    if not allowed_cpus or not visible_cpus or visible_cpus <= allowed_cpus * 1.5:
        return None
    return (
        f"cpu: {visible_cpus} threads are visible but this container may use "
        f"{allowed_cpus:.0f}. torch sizes its thread pool from the visible count, "
        f"so it will oversubscribe by ~{visible_cpus / allowed_cpus:.1f}x. Consider "
        f"OMP_NUM_THREADS={max(1, int(allowed_cpus // 4))} and measure it in the sweep."
    )


def effective_memory_bytes(host_bytes: int, cgroup_bytes: int | None) -> int:
    """What this process may actually use: the smaller of the two, when both exist.

    A cgroup limit above host memory is a no-op, not a promise, so `min` is
    right in both directions.
    """

    if not cgroup_bytes:
        return int(host_bytes)
    if not host_bytes:
        return int(cgroup_bytes)
    return int(min(host_bytes, cgroup_bytes))


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
    args,
    device_info: dict[str, object] | None = None,
    disk_info: dict[str, object] | None = None,
    limits: dict[str, object] | None = None,
) -> tuple[dict[str, object], list[str]]:
    limits = container_limits() if limits is None else limits
    cache_bytes = (
        int(args.example_cache_gb * GIB)
        if args.example_cache_gb > 0
        else int(args.example_cache_examples * EXAMPLE_BYTES)
    )
    budget_bytes = int(args.memory_budget_gb * GIB)
    if budget_bytes <= 0:
        try:
            import psutil

            # `virtual_memory().total` reads /proc/meminfo, which inside a
            # container is the HOST's memory. On a rented slice of a shared box
            # that overstates the budget several-fold, so the cgroup ceiling --
            # the number that will actually OOM-kill this process -- wins.
            budget_bytes = int(
                effective_memory_bytes(
                    psutil.virtual_memory().total, limits["memory_bytes"]
                )
                * 0.85
            )
        except ImportError:
            budget_bytes = 0
    sizing = host_sizing(
        max_window_games=args.replay_window_cap_games,
        example_cache_bytes=cache_bytes,
        memory_budget_bytes=budget_bytes,
        headroom_bytes=int(args.memory_headroom_gb * GIB),
    )
    device = device_info if device_info is not None else device_report(args.device)
    disk = (
        disk_info
        if disk_info is not None
        else disk_report(getattr(args, "run_dir", ".") or ".")
    )
    disk_budget = int(getattr(args, "disk_budget_gb", 0.0) * GIB)
    if disk_budget <= 0 and disk.get("available"):
        disk_budget = int(disk["free_bytes"])
    planned_iterations = int(getattr(args, "iterations", 0) or 0)
    parameters = int(getattr(args, "parameters", 0) or 0)
    if not parameters and planned_iterations > 0:
        parameters = parameter_count(
            args.d_model,
            args.layers,
            args.heads,
            # Read off the run's own flags: a preflight that sizes a different
            # architecture than the run builds is worse than no preflight.
            bool(getattr(args, "pooled_readout", False)),
            bool(getattr(args, "reply_head", False)),
        )
    disk_sizing_result = disk_sizing(
        iterations=planned_iterations,
        games_per_iteration=getattr(args, "games_per_iteration", 0),
        seed_games=getattr(args, "seed_games", 0),
        parameters=parameters,
        promotion_every=getattr(args, "promotion_every", 5),
        disk_budget_bytes=disk_budget,
        headroom_bytes=int(getattr(args, "disk_headroom_gb", 5.0) * GIB),
    )
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
    if not disk_sizing_result.fits:
        failures.append(
            f"disk: the run will write about "
            f"{disk_sizing_result.required_bytes / GIB:.0f} GiB over "
            f"{disk_sizing_result.iterations} iterations "
            f"({disk_sizing_result.checkpoint_bytes / GIB:.0f} GiB of "
            f"per-iteration checkpoints + "
            f"{disk_sizing_result.hof_bytes / GIB:.0f} GiB of HOF archive + "
            f"{disk_sizing_result.buffer_bytes / GIB:.0f} GiB of game records + "
            f"{disk_sizing_result.log_bytes / GIB:.1f} GiB of log + "
            f"{disk_sizing_result.headroom_bytes / GIB:.0f} GiB headroom) but "
            f"only {disk_sizing_result.budget_bytes / GIB:.0f} GiB is available "
            f"at {disk.get('path')}. Nothing here is prunable at runtime -- rent "
            "a bigger disk, or lower --iterations and resume for more."
        )
    import os

    visible_cpus = os.cpu_count() or 0
    allowed_cpus = limits.get("cpus")
    note = thread_oversubscription_note(visible_cpus, allowed_cpus)
    advice: list[str] = [note] if note else []

    report = {
        "host": sizing.as_dict(),
        "limits": {
            "cgroup_memory_bytes": limits.get("memory_bytes"),
            "cgroup_cpus": allowed_cpus,
            "visible_cpus": visible_cpus,
        },
        "advice": advice,
        "device": {**device, "required_bytes": floor},
        "disk": {**disk, **disk_sizing_result.as_dict()},
        "model": {
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "parameters": parameters,
        },
        "passed": not failures,
        "failures": failures,
    }
    return report, failures


def build_parser() -> argparse.ArgumentParser:
    """Every preflight flag, inspectable without running the preflight.

    The launcher's flags are checked against this rather than against a copy of
    the list kept in a test, so adding one here cannot leave the check behind.
    """

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
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="planned iterations; 0 skips the disk budget entirely",
    )
    parser.add_argument("--games-per-iteration", type=int, default=0)
    parser.add_argument("--seed-games", type=int, default=0)
    parser.add_argument("--promotion-every", type=int, default=5)
    parser.add_argument(
        "--run-dir",
        default=".",
        help="the run directory, to pick the filesystem whose free space counts",
    )
    parser.add_argument(
        "--disk-budget-gb",
        type=float,
        default=0.0,
        help="0 measures this box's free space",
    )
    parser.add_argument("--disk-headroom-gb", type=float, default=5.0)
    parser.add_argument(
        "--parameters",
        type=int,
        default=0,
        help="parameter count; 0 builds the model and counts them",
    )
    parser.add_argument("--output", help="write the report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    report, failures = evaluate(args)
    if args.output:
        # The preflight runs before anything creates the run directory, and its
        # natural output path is inside it. Writing the report must not be the
        # thing that fails -- on a fresh box that surfaced as a FileNotFoundError
        # wearing the "refused this box" message, which is the opposite of what
        # this check is for.
        import os

        parent = os.path.dirname(os.path.abspath(args.output))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    host = report["host"]
    limits = report["limits"]
    cgroup = limits["cgroup_memory_bytes"]
    print(
        f"host: needs {host['required_bytes'] / GIB:.1f} GiB at the "
        f"{host['max_window_games']:,}-game window cap, budget "
        f"{host['budget_bytes'] / GIB:.1f} GiB"
        + (f" (cgroup limit {cgroup / GIB:.0f} GiB)" if cgroup else "")
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
    disk = report["disk"]
    if disk["iterations"]:
        print(
            f"disk: needs {disk['required_bytes'] / GIB:.0f} GiB for "
            f"{disk['iterations']} iterations / {disk['total_games']:,} games, "
            f"available {disk['budget_bytes'] / GIB:.0f} GiB"
        )
    for note in report["advice"]:
        print(f"note: {note}")
    if failures:
        print("\nPREFLIGHT FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\npreflight passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
