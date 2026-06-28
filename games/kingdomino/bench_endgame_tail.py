"""Benchmark: full (uncapped) deck=4 endgame minimax tree-size distribution.

Measurement only — does NOT touch production solver behavior, defaults, or the
MCTS hook. It exercises the same `solve_endgame_ab` (alpha-beta + move ordering)
the production solver uses, via the benchmark-only `RustGameState.measure_endgame_tree`
method, which adds timing and a high safety ceiling around that search.

Purpose: decide whether unconditional always-solve at deck=4 roots is shippable
now (just raise the node budget) or needs the Rayon/YBW parallel stack (OPT-6)
first. See `exact_endgame_solver.md`.

Usage:
    python -m games.kingdomino.bench_endgame_tail --n 300 --max-secs 60 --alpha 0.8
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

# Log-spaced solve-time histogram buckets in ms (upper-exclusive except the last).
# The final bucket also collects positions that hit the wall-clock deadline.
_BUCKETS = [
    ("<50ms", 0.0, 50.0),
    ("50-100ms", 50.0, 100.0),
    ("100-250ms", 100.0, 250.0),
    ("250-500ms", 250.0, 500.0),
    ("500ms-1s", 500.0, 1_000.0),
    ("1-3s", 1_000.0, 3_000.0),
    ("3-10s", 3_000.0, 10_000.0),
    (">10s/deadline", 10_000.0, float("inf")),
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300, help="number of deck=4 positions")
    ap.add_argument("--max-secs", type=float, default=60.0,
                    help="per-position wall-clock safety ceiling (seconds); a "
                         "position that hits it is reported as not fully solved")
    ap.add_argument("--alpha", type=float, default=0.8, help="value-frame alpha (affects pruning)")
    ap.add_argument("--base-seed", type=int, default=0, help="first game seed")
    ap.add_argument("--serial", action="store_true",
                    help="use the serial solver instead of the YBW parallel solver")
    args = ap.parse_args()
    parallel = not args.serial

    print(f"Generating {args.n} real deck=4 positions (base_seed={args.base_seed})...", flush=True)
    positions = list(gen_deck4_positions(args.n, args.base_seed))

    rows = []  # (seed, elapsed_ms, fully_solved, value)
    bench_t0 = time.perf_counter()
    for i, (seed, state) in enumerate(positions, 1):
        rs = _rust_state_from_python(state)
        if rs is None:
            print(f"  [warn] seed={seed}: could not build RustGameState, skipping", flush=True)
            continue
        value, solved, elapsed_secs = rs.measure_endgame_tree(
            args.max_secs, SCORE_SCALE, MARGIN_GAIN, args.alpha, parallel
        )
        elapsed_ms = elapsed_secs * 1000.0
        rows.append((seed, float(elapsed_ms), bool(solved), float(value)))
        if i % 25 == 0 or not solved:
            tag = "" if solved else "  <-- HIT DEADLINE"
            print(f"  [{i}/{len(positions)}] seed={seed} "
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
        w.writerow(["seed", "elapsed_ms", "fully_solved", "value"])
        for r in rows:
            w.writerow(r)

    # ── aggregate ──
    total = len(rows)
    solved_rows = [r for r in rows if r[2]]
    capped_rows = [r for r in rows if not r[2]]
    times_sorted = sorted(r[1] for r in rows)

    # Solve-time histogram (deadline positions land in the final bucket).
    hist = [0] * len(_BUCKETS)
    for r in rows:
        ms = r[1]
        capped = not r[2]
        for bi, (_label, lo, hi) in enumerate(_BUCKETS):
            if capped and bi == len(_BUCKETS) - 1:
                hist[bi] += 1
                break
            if (not capped) and lo <= ms < hi:
                hist[bi] += 1
                break

    worst5 = sorted(rows, key=lambda r: r[1], reverse=True)[:5]

    print()
    mode = "serial" if args.serial else "parallel(YBW)"
    print(f"=== deck=4 endgame solve time (n={total}, alpha={args.alpha}, "
          f"max_secs={args.max_secs:g}, solver={mode}) ===")
    print(f"Fully solved within deadline: {len(solved_rows)}/{total} "
          f"({100.0*len(solved_rows)/total:.1f}%)")
    print(f"Hit deadline ({args.max_secs:g}s):     {len(capped_rows)}/{total} "
          f"({100.0*len(capped_rows)/total:.1f}%)")
    print()
    print("Time (ms):   "
          f"p50={_pct(times_sorted,0.50):.0f}  p75={_pct(times_sorted,0.75):.0f}  "
          f"p90={_pct(times_sorted,0.90):.0f}  p95={_pct(times_sorted,0.95):.0f}  "
          f"p99={_pct(times_sorted,0.99):.0f}  max={times_sorted[-1]:.0f}")
    print()
    print("Histogram:")
    hist_max = max(hist) or 1
    for (label, _lo, _hi), count in zip(_BUCKETS, hist):
        bar = "#" * int(round(20 * count / hist_max))
        extra = "  (hit deadline — see worst-5)" if _hi == float("inf") and count else ""
        print(f"  {label:<14}: {count:>4}  {bar}{extra}")
    print()
    print("Worst 5 positions:")
    for seed, elapsed_ms, solved, _value in worst5:
        tag = "  (hit deadline)" if not solved else ""
        print(f"  seed={seed:<8} time={elapsed_ms:.0f}ms{tag}")
    print()
    print(f"CSV: {csv_path}")
    print(f"Benchmark wall-clock: {bench_wall:.1f}s")


if __name__ == "__main__":
    main()
