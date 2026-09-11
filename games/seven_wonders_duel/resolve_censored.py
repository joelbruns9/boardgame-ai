#!/usr/bin/env python3
"""Measure what the endgame solver's DECLINES actually cost.

A position that exhausts its node budget is right-censored: we learn only that
it costs at least the budget, never what it costs. That is the one part of the
corpus a trigger most needs to know, because a decline spends the whole budget
and buys nothing -- cloud2 spent 46% of all solver nodes on 3.2% of attempts.

Refitting the cost model on production buffers cannot answer it. Every censored
row is censored at the SAME cap, so raising `--endgame-solver-max-nodes` against
that corpus only inflates the cost of declines and never converts one into a
solve. The measurement has to be made, not inferred:

    python -m games.seven_wonders_duel.resolve_censored BUFFER --threads 8

**Sized before it is run.** On a laptop 3070, 8 sampled positions from cloud2's
`iter_0096` measured 1.49M nodes/s and 66s each, with true costs of 46M-140M --
only 1.2x to 3.5x past the 40M cap that censored them. So 259 positions is
roughly 4.7 hours single-threaded and ~35 minutes at 8 threads. This is a tail,
not a monster.

**Threads are real.** `RustGame.solve_endgame` wraps the search in `py.detach()`,
so a thread pool genuinely parallelises; the replay that builds each position
holds the GIL, but it is milliseconds against a minute of solving.

**Give it a slack clock.** Two of the eight sampled positions stopped on the
90-second DEADLINE rather than the node budget, which censors them a second time
and for a machine-load-dependent reason. `--max-secs` defaults high for that
reason; a deadline stop is reported separately and never treated as a cost.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .buffer import read_records, replay
from .rust_bridge import rust_game_from_state


def censored_positions(buffer_path: Path, limit: int | None = None) -> list[dict]:
    """Replay each game once and capture every position that hit the node cap.

    The RustGame is built eagerly during the walk because `buffer.replay` hands
    out ONE live game and then mutates it -- keeping the Python object would
    leave every captured position pointing at the end of the record.
    """

    out: list[dict] = []
    for record in read_records(buffer_path):
        wanted = {
            move.i: move
            for move in record.moves
            if getattr(move, "solver_attempted", False)
            and getattr(move, "solver_stop", None) == "nodes"
        }
        if not wanted:
            continue

        def visit(game, move, _wanted=wanted, _record=record):
            target = _wanted.get(move.i)
            if target is None:
                return
            out.append(
                {
                    "game_seed": _record.seed,
                    "move_index": move.i,
                    "iteration": _record.iteration,
                    "censored_at": getattr(target, "solver_nodes", None),
                    "rust": rust_game_from_state(game),
                }
            )

        replay(record, on_state=visit)
        if limit is not None and len(out) >= limit:
            return out[:limit]
    return out


def resolve(
    positions: list[dict],
    *,
    max_nodes: int,
    max_secs: float,
    threads: int,
    progress: bool = True,
) -> list[dict]:
    done = threading.Lock()
    counter = {"n": 0}

    def one(entry: dict) -> dict:
        started = time.perf_counter()
        result = entry["rust"].solve_endgame(max_nodes, max_secs, "exact", "star1")
        elapsed = time.perf_counter() - started
        row = {
            "game_seed": entry["game_seed"],
            "move_index": entry["move_index"],
            "iteration": entry["iteration"],
            "censored_at": entry["censored_at"],
            "nodes": result.get("nodes"),
            "stop": result.get("stop"),
            "regime": result.get("regime"),
            "seconds": elapsed,
        }
        if progress:
            with done:
                counter["n"] += 1
                print(
                    f"  [{counter['n']}/{len(positions)}] "
                    f"nodes {row['nodes']:>14,}  stop {str(row['stop']):>9}  "
                    f"{elapsed:7.2f}s",
                    flush=True,
                )
        return row

    with ThreadPoolExecutor(max_workers=threads) as pool:
        return list(pool.map(one, positions))


def summarise(rows: list[dict], max_nodes: int, censored_at: int) -> dict[str, Any]:
    """What the declines cost, and what a larger budget would have bought.

    `budget_frontier` is the point of the whole exercise: for each candidate
    `--endgame-solver-max-nodes`, how many of these declines would instead have
    completed. Costs are a fixed property of a position, so one generous pass
    yields every smaller budget exactly.
    """

    finished = [r for r in rows if r["stop"] is None]
    node_capped = [r for r in rows if r["stop"] == "nodes"]
    deadline = [r for r in rows if r["stop"] == "deadline"]
    costs = sorted(r["nodes"] for r in finished if r["nodes"])

    frontier = []
    for budget in (censored_at * m for m in (1, 2, 4, 8, 16, 20)):
        completes = sum(1 for c in costs if c <= budget)
        # A position that still declines costs exactly the budget.
        spend = sum(min(c, budget) for c in costs) + (
            len(rows) - completes
        ) * budget
        frontier.append(
            {
                "max_nodes": int(budget),
                "completes": completes,
                "of": len(rows),
                "completed_fraction": completes / max(1, len(rows)),
                "nodes_spent": int(spend),
            }
        )

    summary: dict[str, Any] = {
        "positions": len(rows),
        "completed": len(finished),
        "still_node_capped": len(node_capped),
        "deadline_stopped": len(deadline),
        "study_max_nodes": max_nodes,
        "censored_at": censored_at,
        "seconds_total": sum(r["seconds"] for r in rows),
        "budget_frontier": frontier,
    }
    if costs:
        summary["cost_nodes"] = {
            "min": costs[0],
            "median": statistics.median(costs),
            "p90": costs[min(len(costs) - 1, int(0.9 * len(costs)))],
            "max": costs[-1],
            "multiple_of_cap_median": statistics.median(costs) / censored_at,
        }
    total_nodes = sum(r["nodes"] or 0 for r in rows)
    total_secs = sum(r["seconds"] for r in rows)
    if total_secs > 0:
        summary["measured_nodes_per_second"] = total_nodes / total_secs
    if deadline:
        summary["warning"] = (
            f"{len(deadline)} positions stopped on the WALL CLOCK, not the node "
            "budget. Their cost is still unmeasured, and for a machine-load "
            "dependent reason -- re-run those with a larger --max-secs."
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("buffer", help="a self-play buffer jsonl")
    parser.add_argument("--out", default="", help="write the full result as JSON")
    parser.add_argument(
        "--limit", type=int, default=0, help="stop after N positions (0 = all)"
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=800_000_000,
        help="study budget. 20x cloud2's 40M cap; nothing in the sample came "
        "close, so a decline here is a genuinely expensive position.",
    )
    parser.add_argument(
        "--max-secs",
        type=float,
        default=1200.0,
        help="per-position wall clock. Deliberately slack: a deadline stop "
        "censors a position a SECOND time, for a machine-load-dependent reason.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="solve_endgame releases the GIL, so these are real.",
    )
    parser.add_argument(
        "--censored-at",
        type=int,
        default=40_000_000,
        help="the budget the RUN was censored at, for the frontier.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.buffer)

    print(f"scanning {path.name} for declines...", flush=True)
    positions = censored_positions(path, args.limit or None)
    if not positions:
        print("no censored positions; nothing to measure")
        return 0
    print(
        f"{len(positions)} positions, {args.threads} threads, "
        f"budget {args.max_nodes:,} nodes / {args.max_secs}s each",
        flush=True,
    )

    started = time.perf_counter()
    rows = resolve(
        positions,
        max_nodes=args.max_nodes,
        max_secs=args.max_secs,
        threads=args.threads,
    )
    wall = time.perf_counter() - started

    summary = summarise(rows, args.max_nodes, args.censored_at)
    summary["wall_seconds"] = wall
    print("\n" + json.dumps(summary, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
