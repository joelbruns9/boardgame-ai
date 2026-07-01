"""Benchmark deck=0 exact solving at open-loop MCTS leaves.

This is measurement-only. It does not change training behavior.

Example:
    python -m games.kingdomino.bench_deck0_leaf_exact \
      --positions runs/kingdomino/benchmarks/network_positions_50.pkl \
      --sims 1600 --leaf-max-secs 0.25
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime

import kingdomino_rust

from games.kingdomino.bench_endgame_tail import _load_positions


FIELDS = (
    "elapsed_secs",
    "deck0_leaf_hits",
    "deck0_unique_solves",
    "deck0_cache_hits",
    "deck0_timeouts",
    "terminal_leaf_hits",
    "exact_solve_secs",
    "network_leaf_evals",
    "fallback_count",
    "arena_nodes",
)


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = min(len(vals) - 1, int(round(q * (len(vals) - 1))))
    return vals[idx]


def _run_one(state, *, exact: bool, args, seed: int) -> dict:
    result = kingdomino_rust.bench_open_loop_deck0_leaf_exact(
        state,
        n_sims=args.sims,
        exact_deck0=exact,
        leaf_max_secs=args.leaf_max_secs,
        final_only=args.final_only,
        fpu=args.fpu,
        cpuct=args.cpuct,
        seed=seed,
        score_scale=args.score_scale,
        margin_gain=args.margin_gain,
        alpha=args.alpha,
        ordering=args.ordering,
        parallel=not args.serial,
    )
    row = dict(zip(FIELDS, result))
    row["exact_deck0"] = bool(exact)
    return row


def _summarize(label: str, rows: list[dict]) -> None:
    elapsed = [float(r["elapsed_secs"]) for r in rows]
    total_hits = sum(int(r["deck0_leaf_hits"]) for r in rows)
    unique_solves = sum(int(r["deck0_unique_solves"]) for r in rows)
    cache_hits = sum(int(r["deck0_cache_hits"]) for r in rows)
    timeouts = sum(int(r["deck0_timeouts"]) for r in rows)
    solve_secs = sum(float(r["exact_solve_secs"]) for r in rows)
    net_evals = sum(int(r["network_leaf_evals"]) for r in rows)
    print(f"\n=== {label} ===")
    print(
        f"wall: total={sum(elapsed):.3f}s "
        f"p50={_pct(elapsed, 0.50)*1000:.1f}ms "
        f"p90={_pct(elapsed, 0.90)*1000:.1f}ms "
        f"max={max(elapsed, default=0.0)*1000:.1f}ms"
    )
    print(
        f"deck0 leaves={total_hits} unique_solves={unique_solves} "
        f"cache_hits={cache_hits} timeouts={timeouts} "
        f"exact_solve_secs={solve_secs:.3f}s network_leaf_evals={net_evals}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", default="runs/kingdomino/benchmarks/network_positions_50.pkl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--sims", type=int, default=1600)
    ap.add_argument("--leaf-max-secs", type=float, default=0.25)
    ap.add_argument("--final-only", action="store_true",
                    help="only exact-solve deck=0 FINAL_PLACEMENT leaves; "
                         "PLACE_AND_SELECT deck=0 leaves use the ordinary leaf evaluator")
    ap.add_argument("--fpu", type=float, default=-0.2)
    ap.add_argument("--cpuct", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-scale", type=float, default=160.0)
    ap.add_argument("--margin-gain", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--ordering", default="lookahead2_clustered")
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--label", default="deck0_leaf_exact")
    args = ap.parse_args()

    positions = _load_positions(args.positions)[: args.n]
    print(
        f"Loaded {len(positions)} positions; sims={args.sims}, "
        f"leaf_max_secs={args.leaf_max_secs:g}, "
        f"final_only={args.final_only}, "
        f"solver={'serial' if args.serial else 'parallel'}"
    )

    rows = []
    t0 = time.perf_counter()
    for i, (pos_seed, state) in enumerate(positions, 1):
        run_seed = args.seed ^ (int(pos_seed) * 0x9E3779B1)
        base = _run_one(state, exact=False, args=args, seed=run_seed)
        exact = _run_one(state, exact=True, args=args, seed=run_seed)
        base.update(position_seed=pos_seed, position_index=i, mode="baseline")
        exact.update(position_seed=pos_seed, position_index=i, mode="deck0_exact")
        rows.extend([base, exact])
        print(
            f"[{i}/{len(positions)}] seed={pos_seed} "
            f"base={base['elapsed_secs']*1000:.1f}ms "
            f"exact={exact['elapsed_secs']*1000:.1f}ms "
            f"deck0_hits={exact['deck0_leaf_hits']} "
            f"unique={exact['deck0_unique_solves']} "
            f"solve={exact['exact_solve_secs']*1000:.1f}ms",
            flush=True,
        )

    base_rows = [r for r in rows if r["mode"] == "baseline"]
    exact_rows = [r for r in rows if r["mode"] == "deck0_exact"]
    _summarize("baseline", base_rows)
    _summarize("deck0 exact leaves", exact_rows)
    base_total = sum(float(r["elapsed_secs"]) for r in base_rows)
    exact_total = sum(float(r["elapsed_secs"]) for r in exact_rows)
    print(
        f"\nDelta: {exact_total - base_total:+.3f}s "
        f"({(exact_total / max(base_total, 1e-9) - 1.0) * 100:+.1f}%)"
    )

    out_dir = os.path.join("runs", "kingdomino", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"{args.label}_{ts}.csv")
    fieldnames = ("position_index", "position_seed", "mode", "exact_deck0") + FIELDS
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"CSV: {csv_path}")
    print(f"Benchmark wall-clock: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
