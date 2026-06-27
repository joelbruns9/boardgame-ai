"""Benchmark: full (uncapped) deck=4 endgame minimax tree-size distribution.

Measurement only — does NOT touch production solver behavior, defaults, or the
MCTS hook. It exercises the same `solve_endgame_ab` (alpha-beta + move ordering)
the production solver uses, via the benchmark-only `RustGameState.measure_endgame_tree`
method, which adds timing and a high safety ceiling around that search.

Purpose: decide whether unconditional always-solve at deck=4 roots is shippable
now (just raise the node budget) or needs the Rayon/YBW parallel stack (OPT-6)
first. See `exact_endgame_solver.md`.

Usage:
    python -m games.kingdomino.bench_endgame_tail --n 300 --hard-cap 50000000 --alpha 0.8
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from datetime import datetime

from games.kingdomino.game import GameState, Phase
from games.kingdomino.endgame_solver import _rust_state_from_python

# Production value-formula constants (see exact_endgame_solver.md / mcts config).
SCORE_SCALE = 100.0
MARGIN_GAIN = 2.0

# Log-spaced node-count histogram buckets (upper-exclusive except the last).
_BUCKETS = [
    ("<100k", 0, 100_000),
    ("100k-500k", 100_000, 500_000),
    ("500k-1M", 500_000, 1_000_000),
    ("1M-2M", 1_000_000, 2_000_000),
    ("2M-5M", 2_000_000, 5_000_000),
    ("5M-10M", 5_000_000, 10_000_000),
    ("10M-50M", 10_000_000, 50_000_000),
    (">50M", 50_000_000, float("inf")),
]


def gen_deck4_positions(n: int, base_seed: int):
    """Yield (seed, GameState) for n real deck=4 PLACE_AND_SELECT positions.

    One position per game: each game is played with random legal moves until the
    first time it reaches len(deck)==4 in PLACE_AND_SELECT, then snapshotted. The
    move-selection RNG is derived deterministically from the game seed, so the
    whole run is reproducible from --base-seed.
    """
    seed = base_seed
    produced = 0
    while produced < n:
        move_rng = random.Random(0xC0FFEE ^ (seed * 2654435761 & 0xFFFFFFFF))
        state = GameState.new(seed=seed)
        found = None
        while state.phase != Phase.GAME_OVER:
            if len(state.deck) == 4 and state.phase == Phase.PLACE_AND_SELECT:
                found = state
                break
            state = state.step(move_rng.choice(state.legal_actions()))
        seed += 1
        if found is not None:
            produced += 1
            yield (seed - 1, found)


def _pct(sorted_vals, q):
    """Nearest-rank percentile (q in [0,1]) over a pre-sorted list."""
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def _fmt_nodes(n, capped):
    return f">{n:,}" if capped else f"{n:,}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300, help="number of deck=4 positions")
    ap.add_argument("--hard-cap", type=int, default=50_000_000, help="node safety ceiling")
    ap.add_argument("--alpha", type=float, default=0.8, help="value-frame alpha (affects pruning)")
    ap.add_argument("--base-seed", type=int, default=0, help="first game seed")
    ap.add_argument("--serial", action="store_true",
                    help="use the serial solver (true single-traversal node counts) "
                         "instead of the YBW parallel solver (wall-clock)")
    args = ap.parse_args()
    parallel = not args.serial

    print(f"Generating {args.n} real deck=4 positions (base_seed={args.base_seed})...", flush=True)
    positions = list(gen_deck4_positions(args.n, args.base_seed))

    rows = []  # (seed, nodes, elapsed_ms, fully_solved, value)
    bench_t0 = time.perf_counter()
    for i, (seed, state) in enumerate(positions, 1):
        rs = _rust_state_from_python(state)
        if rs is None:
            print(f"  [warn] seed={seed}: could not build RustGameState, skipping", flush=True)
            continue
        value, solved, nodes, elapsed_ms = rs.measure_endgame_tree(
            args.hard_cap, SCORE_SCALE, MARGIN_GAIN, args.alpha, parallel
        )
        rows.append((seed, int(nodes), float(elapsed_ms), bool(solved), float(value)))
        if i % 25 == 0 or not solved:
            tag = "" if solved else "  <-- HIT CAP"
            print(f"  [{i}/{len(positions)}] seed={seed} nodes={int(nodes):,} "
                  f"time={elapsed_ms:.0f}ms{tag}", flush=True)
    bench_wall = time.perf_counter() - bench_t0

    if not rows:
        print("No positions measured.")
        return

    # ── persist raw data ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("runs", "kingdomino", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"endgame_tail_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "nodes", "elapsed_ms", "fully_solved", "value"])
        for r in rows:
            w.writerow(r)

    # ── aggregate ──
    total = len(rows)
    solved_rows = [r for r in rows if r[3]]
    capped_rows = [r for r in rows if not r[3]]
    nodes_sorted = sorted(r[1] for r in rows)
    times_sorted = sorted(r[2] for r in rows)

    # Effective throughput from solved rows only (capped rows are time-truncated).
    thr_samples = sorted(
        r[1] / (r[2] / 1000.0) for r in solved_rows if r[2] > 0
    )
    median_thr = _pct(thr_samples, 0.5) if thr_samples else 0.0

    # Histogram (capped positions land in the >50M bucket regardless of exact count).
    hist = [0] * len(_BUCKETS)
    for r in rows:
        n = r[1]
        capped = not r[3]
        for bi, (_label, lo, hi) in enumerate(_BUCKETS):
            if capped and bi == len(_BUCKETS) - 1:
                hist[bi] += 1
                break
            if (not capped) and lo <= n < hi:
                hist[bi] += 1
                break

    worst5 = sorted(rows, key=lambda r: r[1], reverse=True)[:5]

    def nodes_pct(q):
        # If the percentile rank lands on a capped row, report it as ">value".
        v = _pct(nodes_sorted, q)
        # Determine whether the row at/above this count is capped: count how many
        # capped rows have nodes >= v.
        return v

    cap_str = f"{args.hard_cap/1e6:.0f}M"
    max_capped = (not nodes_sorted) or any(
        (not r[3]) and r[1] == nodes_sorted[-1] for r in rows
    )

    print()
    mode = "serial" if args.serial else "parallel(YBW)"
    print(f"=== deck=4 endgame tree size (n={total}, alpha={args.alpha}, "
          f"hard_cap={cap_str}, solver={mode}) ===")
    print(f"Fully solved within hard_cap: {len(solved_rows)}/{total} "
          f"({100.0*len(solved_rows)/total:.1f}%)")
    print(f"Hit hard_cap ({cap_str}):        {len(capped_rows)}/{total} "
          f"({100.0*len(capped_rows)/total:.1f}%)")
    print()
    print("Node count:  "
          f"p50={nodes_pct(0.50):,}  p75={nodes_pct(0.75):,}  p90={nodes_pct(0.90):,}  "
          f"p95={nodes_pct(0.95):,}  p99={nodes_pct(0.99):,}  "
          f"max={_fmt_nodes(nodes_sorted[-1], max_capped)}")
    print("Time (ms):   "
          f"p50={_pct(times_sorted,0.50):.0f}  p75={_pct(times_sorted,0.75):.0f}  "
          f"p90={_pct(times_sorted,0.90):.0f}  p95={_pct(times_sorted,0.95):.0f}  "
          f"p99={_pct(times_sorted,0.99):.0f}  max={times_sorted[-1]:.0f}")
    print(f"Effective throughput (median, solved): ~{median_thr/1000.0:,.0f}k nodes/sec")
    print()
    print("Histogram:")
    hist_max = max(hist) or 1
    for (label, _lo, _hi), count in zip(_BUCKETS, hist):
        bar = "#" * int(round(20 * count / hist_max))
        extra = "  (hit cap — see worst-5)" if label == ">50M" and count else ""
        print(f"  {label:<10}: {count:>4}  {bar}{extra}")
    print()
    print("Worst 5 positions:")
    for seed, nodes, elapsed_ms, solved, _value in worst5:
        tag = "  (hit cap)" if not solved else ""
        print(f"  seed={seed:<8} nodes={_fmt_nodes(nodes, not solved):<14} "
              f"time={elapsed_ms:.0f}ms{tag}")
    print()
    print(f"CSV: {csv_path}")
    print(f"Benchmark wall-clock: {bench_wall:.1f}s")


if __name__ == "__main__":
    main()
