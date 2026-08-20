"""Where a training iteration's generation time actually went.

Every number here is already in `training_log.jsonl`. The Rust scheduler
records a full time breakdown per shard and Phase D stores it verbatim under
`generation_performance.rust_scheduler`; nothing surfaced it, so the questions
it answers were being answered by inference instead:

* **Does the solver block generation?** `sched_solve_wait_ns` is time the
  scheduler loop spent blocked on `SolverPool::wait_one`, which it only calls
  when no group is waiting to be batched and no batch is in flight. A small
  share means the async overlay is working; a large one means solves are on the
  critical path and the thread split is wrong.
* **Is the GPU fed?** `py_call_ns` is time inside the evaluation callback, and
  `batch_rows` is how wide each batch was. Small batches with a high call share
  is per-call overhead, not compute -- more concurrency, not a bigger GPU.
* **How concurrent is it really?** `live_slot_ns / scheduler_wall_ns` is the
  time-weighted mean number of games in flight. The slot *cap* is a ceiling, not
  a measurement, and the two differ whenever games retire faster than the pool
  refills.

All `*_ns` counters are per-thread and summed across shards, and
`scheduler_wall_ns` is the summed per-shard envelope, so every ratio below is a
fraction of shard-thread time and is comparable across worker counts. That is
the only reason these can be read as shares at all.

Usage:
    python -m games.seven_wonders_duel.generation_profile <run-dir> [--last N]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _rows(run_dir: Path) -> list[dict[str, Any]]:
    log = run_dir / "training_log.jsonl"
    if not log.is_file():
        raise SystemExit(f"no training log at {log}")
    rows = []
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A row still being written when this ran. Skip rather than fail:
            # this tool is meant to be run against a LIVE run.
            continue
    return rows


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return float(ordered[index])


def profile_row(row: dict[str, Any], batch_cap: int = 0) -> dict[str, Any] | None:
    """One iteration's generation breakdown, or None if it recorded none.

    `batch_cap` comes from the run manifest, not the metrics: the scheduler
    exports `max_active_slots` but not `global_batch_cap`, and a cap read as
    zero would silently disable the check that compares batch width against
    it -- a hint that can never fire is worse than no hint.
    """

    reported = row.get("generation_performance") or {}
    # TWO SHAPES, and the nested one is what the cloud runs write.
    #
    # `training_adapter.generate` reports metrics as
    #     {"performance": <loop.last_generation_stats>, "summary": ..., "model": ...}
    # and the az_loop controller stores that dict verbatim, so the scheduler
    # block lives at generation_performance.performance.rust_scheduler. Phase D's
    # own loop, used by smoke runs and the toy harness, assigns
    # last_generation_stats straight into generation_performance, leaving it
    # flat. Reading only one of the two silently reports "no scheduler metrics"
    # against a run that recorded them for every iteration.
    performance = reported.get("performance")
    if not isinstance(performance, dict):
        performance = reported
    scheduler = performance.get("rust_scheduler") or {}
    if not scheduler:
        return None

    wall = float(scheduler.get("scheduler_wall_ns") or 0.0)
    share = lambda key: (float(scheduler.get(key) or 0.0) / wall) if wall else 0.0
    batches = [int(value) for value in (scheduler.get("batch_rows") or [])]
    # `summary` sits beside `performance` in the adapter's shape and inside it
    # in the flat one.
    summary = reported.get("summary") or performance.get("summary") or {}
    solver = summary.get("solver") or {}

    return {
        "iteration": row.get("iteration", -1),
        "seconds": float(performance.get("seconds") or 0.0),
        "games_per_second": float(performance.get("games_per_second") or 0.0),
        # The share that answers "is the solver on the critical path".
        "solve_wait_share": share("sched_solve_wait_ns"),
        "eval_call_share": share("py_call_ns"),
        "queue_wait_share": share("queue_wait_ns"),
        "tree_share": share("rust_tree_ns"),
        "encode_share": share("encode_pack_ns"),
        # Leaves a SINGLE game's search has in flight at once, as opposed to
        # batch width, which is leaves summed across concurrent games. 1.0 means
        # every game issues one leaf and waits for the GPU before issuing the
        # next -- so a 1600-sim move is 1600 sequential round trips, and the
        # only thing hiding that latency is other games.
        "mean_wave_width": float(scheduler.get("mean_wave_width") or 0.0),
        "batch_mean": statistics.fmean(batches) if batches else 0.0,
        "batch_p95": _percentile(batches, 0.95),
        "batch_max": int(scheduler.get("max_batch_rows") or 0),
        "batches": len(batches),
        "batch_cap": int(batch_cap),
        # Occupancy, not the ceiling: `max_active_slots` is what was allowed.
        "mean_live_slots": (float(scheduler.get("live_slot_ns") or 0.0) / wall)
        if wall
        else 0.0,
        "max_live_slots": int(scheduler.get("max_live_slots") or 0),
        "slot_cap": int(scheduler.get("max_active_slots") or 0),
        "workers": int(scheduler.get("scheduler_workers") or 0),
        "solver_attempted": int(solver.get("attempted") or 0),
        "solver_masked": int(solver.get("masked") or 0),
        "solver_deadline_stops": int((solver.get("stops") or {}).get("deadline", 0)),
        # `nodes_total`, not `nodes`: _summarize_solver reports totals under
        # explicit names. Reading a key that does not exist gave 0 and the
        # capacity estimate then reported an idle solver on a run doing ~10,000
        # solves an iteration -- a wrong answer that looked like a measurement.
        "solver_nodes": int(
            solver.get("nodes_total") or solver.get("nodes") or 0
        ),
        "solver_nodes_on_declines": int(solver.get("nodes_on_declines") or 0),
    }


def manifest_batch_cap(run_dir: Path) -> int:
    """The run's configured `--rust-global-batch-cap`, or 0 if unknown."""

    manifest = run_dir / "run_manifest.json"
    if not manifest.is_file():
        return 0
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return int((payload.get("config") or {}).get("rust_global_batch_cap") or 0)


def solver_budget(
    profile: dict[str, Any], node_rate: float, threads_total: int
) -> dict[str, float] | None:
    """Does the iteration's solving fit in the iteration's solver capacity?

    ESTIMATED, not measured: the scheduler records how long it BLOCKED on solves
    (`sched_solve_wait_ns`), never how long the solver threads were busy. This
    divides the iteration's node count by a measured per-thread node rate, so it
    is only as good as that rate -- stage 6b of the launcher measures one on the
    box. Labelled as an estimate everywhere it is printed for that reason.
    """

    if node_rate <= 0 or threads_total <= 0 or profile["seconds"] <= 0:
        return None
    thread_seconds = profile["solver_nodes"] / node_rate
    capacity = threads_total * profile["seconds"]
    return {
        "thread_seconds": thread_seconds,
        "capacity_thread_seconds": capacity,
        "utilisation": thread_seconds / capacity,
    }


def diagnose(rows: list[dict[str, Any]]) -> str:
    """What the log actually contains, when nothing could be read from it.

    "no iteration recorded scheduler metrics" was a dead end for everyone: it
    named no key, so it could not distinguish a run that genuinely has none from
    a reader looking at the wrong nesting level -- which is what it was. A tool
    that cannot find its data should report what it DID find.
    """

    if not rows:
        return "  the log has no parseable rows at all"
    last = rows[-1]
    lines = [
        f"  rows in log: {len(rows)}",
        f"  keys in the last row: {sorted(last)}",
    ]
    reported = last.get("generation_performance")
    if reported is None:
        lines.append("  the last row has NO 'generation_performance' key")
    elif isinstance(reported, dict):
        lines.append(f"  generation_performance keys: {sorted(reported)}")
        for key, value in sorted(reported.items()):
            if isinstance(value, dict):
                lines.append(f"    {key}: {sorted(value)}")
    stats = last.get("stats")
    if isinstance(stats, dict):
        lines.append(f"  stats keys: {sorted(stats)}")
        generation = stats.get("generation")
        if isinstance(generation, dict):
            lines.append(f"    stats.generation: {sorted(generation)}")
    lines.append(
        "  Send this block back: it names the path the scheduler block is under."
    )
    return "\n".join(lines)


def render(
    profiles: list[dict[str, Any]],
    node_rate: float = 0.0,
    solver_threads_total: int = 0,
    rows: list[dict[str, Any]] | None = None,
) -> str:
    if not profiles:
        return "\n".join(
            [
                "no iteration in this log recorded scheduler metrics.",
                "What the log does contain:",
                diagnose(rows or []),
            ]
        )

    lines = [
        "",
        "  iter  games/s   batch(mean/p95/max)   live_slots/cap  wave   "
        "solve_wait  eval_call  tree   solved",
    ]
    for profile in profiles:
        lines.append(
            f"  {profile['iteration']:<5} {profile['games_per_second']:<8.3f} "
            f"{profile['batch_mean']:>6.0f}/{profile['batch_p95']:>5.0f}/"
            f"{profile['batch_max']:<6} "
            f"{profile['mean_live_slots']:>6.0f}/{profile['slot_cap']:<6} "
            f"{profile['mean_wave_width']:>4.2f} "
            f"{profile['solve_wait_share']:>9.1%} {profile['eval_call_share']:>9.1%} "
            f"{profile['tree_share']:>6.1%} "
            f"{profile['solver_masked']}/{profile['solver_attempted']}"
        )

    latest = profiles[-1]
    lines += ["", "  Reading the latest iteration:"]

    if latest["solve_wait_share"] >= 0.10:
        lines.append(
            f"  * SOLVER IS ON THE CRITICAL PATH: {latest['solve_wait_share']:.1%} of "
            "shard time is blocked waiting for solves with nothing else to do. "
            "Give the solver more threads, lower its node budget, or raise "
            "concurrency so another game can run while one solves."
        )
    else:
        lines.append(
            f"  * solver is not blocking generation ({latest['solve_wait_share']:.1%} "
            "of shard time blocked); the async overlay is doing its job."
        )

    if latest["solver_deadline_stops"]:
        lines.append(
            f"  * {latest['solver_deadline_stops']} solve(s) stopped on the DEADLINE. "
            "Which positions got a proof now depends on machine load, so the "
            "buffer is no longer a function of its seeds."
        )

    if latest["batch_cap"] and latest["batch_mean"] < 0.25 * latest["batch_cap"]:
        lines.append(
            f"  * batches average {latest['batch_mean']:.0f} against a cap of "
            f"{latest['batch_cap']}, so the cap is not binding. NOTE: batch "
            "width on its own does not predict throughput here -- measured "
            "2026-08-20, 2 shards ran 19% FASTER than 1 shard at half the batch "
            "width, and cloud6 ran 234-wide batches at the same games/s as this "
            "run does at 43. Total leaves in flight was near-constant across "
            "shard counts. Treat width as a diagnostic, not a target."
        )

    budget = solver_budget(latest, node_rate, solver_threads_total)
    if budget is not None:
        lines.append(
            f"  * solver load (ESTIMATED from {latest['solver_nodes']:,} nodes at "
            f"{node_rate:,.0f} nodes/s/thread): "
            f"{budget['thread_seconds']:,.0f} thread-seconds against "
            f"{budget['capacity_thread_seconds']:,.0f} available "
            f"({budget['utilisation']:.0%} of the solver pool)."
        )
        if latest["solver_nodes"] and latest["solver_nodes_on_declines"]:
            share = latest["solver_nodes_on_declines"] / latest["solver_nodes"]
            lines.append(
                f"    {share:.0%} of those nodes went on solves that DECLINED "
                "(hit the budget without a proof). Raising the budget buys "
                "proofs on exactly those; lowering it wastes less on them."
            )
        if budget["utilisation"] >= 1.0:
            lines.append(
                "    The solving does NOT fit in the iteration: solves must be "
                "queueing, and the only reason generation is not stalling on "
                "them is that some are being declined."
            )
        elif budget["utilisation"] <= 0.30:
            lines.append(
                "    The solver pool is mostly idle. Those cores are bought and "
                "unused: either raise the node budget (more proof per position) "
                "or move threads to generation."
            )

    if 0 < latest["mean_wave_width"] <= 1.01:
        lines.append(
            "  * wave width 1.00: each game has exactly ONE leaf in flight, so a "
            "1600-sim move is 1600 sequential GPU round trips. Batch width comes "
            "only from running many games at once, which caps rows per live game "
            "at 1. --leaf-batch is the only knob that lifts that ceiling."
        )

    if latest["slot_cap"] and latest["mean_live_slots"] < 0.6 * latest["slot_cap"]:
        lines.append(
            f"  * mean concurrency {latest['mean_live_slots']:.0f} is well under the "
            f"{latest['slot_cap']}-slot cap, so raising the cap alone will not "
            "raise batch width -- the pool is not refilling fast enough to use "
            "what it already has."
        )

    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--last", type=int, default=10, help="how many iterations to show"
    )
    parser.add_argument("--json", action="store_true", help="emit the raw profile")
    parser.add_argument(
        "--solver-node-rate",
        type=float,
        default=0.0,
        help="measured solver nodes per second PER THREAD (stage 6b of the "
        "launcher measures this on the box). Enables the solver capacity "
        "estimate, which is skipped rather than guessed without it.",
    )
    parser.add_argument(
        "--solver-threads-total",
        type=int,
        default=0,
        help="solver threads in total, i.e. --solver-threads x scheduler workers",
    )
    args = parser.parse_args(argv)

    batch_cap = manifest_batch_cap(args.run_dir)
    rows = _rows(args.run_dir)
    profiles = [
        profile
        for profile in (profile_row(row, batch_cap) for row in rows)
        if profile is not None
    ]
    if args.last > 0:
        profiles = profiles[-args.last :]

    if args.json:
        print(json.dumps(profiles, indent=2))
    else:
        print(
            render(
                profiles,
                args.solver_node_rate,
                args.solver_threads_total,
                rows=rows,
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
