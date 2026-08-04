"""Packing-only thread sweep (CORE_UTILIZATION_PLAN.md step 2b / step 7).

Row-parallel packing was 22.8% of generation wall before it was parallelised, so
its thread count needs tuning per machine. Doing that through
``f4_throughput_bench`` costs minutes of generation per rung; packing is a pure
CPU function of a batch of states, so it can be timed directly in seconds.

Two things this answers:

1. **Where the plateau starts** on this machine, so the expensive end-to-end
   sweep can hold the thread count fixed instead of carrying it as a grid
   dimension.
2. **Whether the optimum moves with batch size.** Rayon parallelises across rows,
   so rows-per-batch *is* the available parallel work. Within a single
   generation run that already varies ~4x between the steady window
   (``steady_rows_per_batch`` ~440) and the drain tail (~95), so the chosen
   thread count has to be robust across the range regardless. If the surface is
   flat in rows, the dimension can be dropped.

**Thread counts come from the cgroup, not ``os.cpu_count()``.** On a shared
cloud slice the host's core count is not what the container may use, and
oversubscribing past the real quota is the one regime where this change can
*lose* throughput -- see ``cloud_preflight.container_limits``.

Usage::

    python -m games.seven_wonders_duel.f4_pack_sweep --output runs/pack_sweep.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
from pathlib import Path

import seven_wonders_rust

from .cloud_preflight import container_limits
from .rust_bridge import rust_game_for_self_play


def effective_cpus() -> tuple[int, dict]:
    """Usable parallelism, preferring the cgroup quota over the host count.

    ``os.cpu_count()`` reports the host on a container-limited slice, which is
    exactly the case where oversubscription hurts.
    """

    host = os.cpu_count() or 1
    detail = {"host_cpu_count": host, "cgroup_cpus": None}
    try:
        limits = container_limits()
        quota = limits.get("cpus")
        if quota:
            detail["cgroup_cpus"] = quota
            return max(1, int(math.floor(quota))), detail
    except Exception as error:  # pragma: no cover - platform dependent
        detail["cgroup_error"] = repr(error)
    return host, detail


def thread_rungs(cpus: int) -> list[int]:
    """A short ladder derived from the machine, not a fixed list.

    1 is skipped: ``pack_routed`` already takes a serial path at one thread
    because rayon's dispatch made it a measured 0.971x regression.
    """

    candidates = {2, max(2, cpus // 4), max(2, cpus // 2), cpus}
    return sorted(c for c in candidates if 2 <= c <= cpus)


def build_corpus(rows: int, seed: int, plies: int) -> list:
    """Distinct mid-game states, so token counts vary as they do in production."""

    games = []
    for index in range(rows):
        # `RustGame.__new__` wants the full deal; `rust_bridge` owns the setup
        # that produces a legal one, and is the same path self-play uses.
        game = rust_game_for_self_play(seed + index, first_player=index % 2)
        # Walk a deterministic but state-dependent path so the corpus spans
        # phases rather than repeating one position `rows` times.
        step = 0
        while step < plies:
            legal = game.legal_action_indices()
            if not legal:
                break
            game.apply_index(legal[(index + step) % len(legal)])
            step += 1
        games.append(game)
    return games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="batch widths to test. Defaults span the drain tail (~95) and the "
        "steady window (~440) seen in real generation.",
    )
    parser.add_argument("--threads", type=int, nargs="+", default=None)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--plies", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    cpus, cpu_detail = effective_cpus()
    threads = args.threads or thread_rungs(cpus)
    print(f"effective cpus: {cpus}  ({cpu_detail})")
    print(f"thread rungs:   {threads}")

    # One corpus, sliced. Building a fresh corpus per width would vary the
    # *states* along with the batch size, so a width comparison would measure
    # two things at once -- the states at width 512 are not the states at 64.
    full_corpus = build_corpus(max(args.rows), args.seed, args.plies)

    results = []
    for rows in args.rows:
        corpus = full_corpus[:rows]
        for count in threads:
            samples = [
                seven_wonders_rust.bench_pack_routed(corpus, args.iterations, count)
                for _ in range(args.repetitions)
            ]
            # Median, not mean: a single scheduling hiccup should not move a
            # rung, and these runs are short enough to catch one.
            seconds = statistics.median(samples)
            results.append(
                {
                    "rows": rows,
                    "threads": count,
                    "seconds": seconds,
                    "spread": (max(samples) - min(samples)) / seconds if seconds else 0.0,
                    "rows_per_second": rows * args.iterations / seconds,
                }
            )
            spread = (max(samples) - min(samples)) / seconds if seconds else 0.0
            # Print the spread: without it a wandering argmax reads as a real
            # width dependence when it is just run-to-run noise.
            print(
                f"  rows={rows:5d} threads={count:3d}  {seconds*1000:8.1f} ms"
                f"  {rows * args.iterations / seconds:12.0f} rows/s"
                f"  spread {spread*100:5.1f}%"
            )

    # Per-width best, and whether the choice actually depends on width.
    best = {}
    for rows in args.rows:
        at_width = [r for r in results if r["rows"] == rows]
        best[rows] = min(at_width, key=lambda r: r["seconds"])["threads"]
    flat = len(set(best.values())) == 1

    # A rung within tolerance of the best everywhere is a safe fixed choice even
    # when the argmax wanders, which matters more than the argmax itself: the
    # measured curve is a plateau, not a peak.
    # Tolerance is the measured noise floor, not a constant. The argmax wanders
    # between widths whenever run-to-run spread exceeds the gap between rungs,
    # and calling that "the optimum depends on width" would push a whole extra
    # dimension into the expensive throughput sweep for no reason.
    noise_floor = statistics.median(r["spread"] for r in results)
    tolerance = max(0.05, noise_floor)
    worst_case = {}
    for count in threads:
        losses = []
        for rows in args.rows:
            at_width = [r for r in results if r["rows"] == rows]
            floor = min(r["seconds"] for r in at_width)
            mine = next(r["seconds"] for r in at_width if r["threads"] == count)
            losses.append(mine / floor - 1.0)
        worst_case[count] = max(losses)
    robust = [
        {"threads": c, "worst_case_loss": w}
        for c, w in worst_case.items()
        if w <= tolerance
    ]
    # No rung inside tolerance means pick the least-bad one, not the largest.
    # The argmax wandering between widths says the surface is noisy, and the
    # highest thread count is not a safe default when it loses at narrow widths.
    fallback = min(worst_case, key=worst_case.get)

    summary = {
        "cpu": cpu_detail | {"effective": cpus, "platform": platform.processor()},
        "rows": args.rows,
        "threads": threads,
        "iterations": args.iterations,
        "repetitions": args.repetitions,
        "results": results,
        "best_threads_per_width": best,
        "optimum_independent_of_width": flat,
        "noise_floor": noise_floor,
        "tolerance": tolerance,
        "worst_case_loss_by_threads": worst_case,
        "indistinguishable_from_best": robust,
        "recommended_threads": (
            min(r["threads"] for r in robust) if robust else fallback
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"best per width:      {best}")
    print(f"width-independent:   {flat}")
    print(f"noise floor:         {noise_floor*100:.1f}%  (tolerance {tolerance*100:.1f}%)")
    print(f"within tolerance:    {[r['threads'] for r in robust]}")
    losses = {k: round(v, 3) for k, v in worst_case.items()}
    print(f"worst-case loss:     {losses}")
    print(f"recommended:         {summary['recommended_threads']}")
    if len(robust) > 1:
        print(
            "  Several rungs are within measurement noise of the best: this is a "
            "plateau, not a peak. Fix the thread count and keep it out of the "
            "throughput sweep's grid."
        )
    elif not flat:
        print(
            "  The optimum moves with batch width by more than the noise floor, "
            "so thread count belongs in the throughput sweep's grid."
        )
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
