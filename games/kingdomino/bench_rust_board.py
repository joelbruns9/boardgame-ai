"""
bench_rust_board.py — micro-benchmark RustBoard vs Python Board.

Measures the two hottest operations from the profile:
  - legal_placements (the 63.7%-subtree function)
  - is_legal_placement (15% self-time, called 11.5M times)

It builds a set of realistic mid-game board positions (by playing random legal
moves), then times each board type doing the same work on the same positions.

Run (PowerShell, one line):
  python -m games.kingdomino.bench_rust_board
  python -m games.kingdomino.bench_rust_board --positions 200 --reps 50
"""
from __future__ import annotations

import random
import sys
import time

from games.kingdomino.board import Board
from games.kingdomino.dominoes import DOMINOES

import kingdomino_rust


def _build_positions(n_positions: int, seed: int = 0):
    """Build N (python_board, rust_board) pairs at random mid-game states,
    kept in lockstep so they represent identical positions."""
    rng = random.Random(seed)
    all_ids = list(DOMINOES.keys())
    pairs = []
    for _ in range(n_positions):
        pb = Board()
        rb = kingdomino_rust.RustBoard(7, 7)
        n_moves = rng.randint(4, 18)
        for _ in range(n_moves):
            did = rng.choice(all_ids)
            domino = DOMINOES[did]
            placements = pb.legal_placements(domino)
            if not placements:
                continue
            chosen = rng.choice(placements)
            pb.place(domino, chosen)
            ta, ca = int(domino.a.terrain), int(domino.a.crowns)
            tb, cb = int(domino.b.terrain), int(domino.b.crowns)
            rb.place(ta, ca, tb, cb,
                     chosen.x1, chosen.y1, chosen.x2, chosen.y2, chosen.flipped)
        pairs.append((pb, rb))
    return pairs


def bench(n_positions: int = 100, reps: int = 30, seed: int = 0):
    pairs = _build_positions(n_positions, seed)
    all_ids = list(DOMINOES.keys())
    rng = random.Random(seed + 1)
    # Fixed list of dominoes to query per position (same for both backends).
    query_dominoes = [DOMINOES[rng.choice(all_ids)] for _ in range(reps)]

    # ── Python legal_placements ──
    t0 = time.perf_counter()
    py_count = 0
    for pb, _ in pairs:
        for domino in query_dominoes:
            py_count += len(pb.legal_placements(domino))
    py_time = time.perf_counter() - t0

    # ── Rust legal_placements ──
    # Pre-extract raw ints so we time the Rust call, not Python attribute access.
    query_raw = [(int(d.a.terrain), int(d.a.crowns),
                  int(d.b.terrain), int(d.b.crowns)) for d in query_dominoes]
    t0 = time.perf_counter()
    rust_count = 0
    for _, rb in pairs:
        for (ta, ca, tb, cb) in query_raw:
            rust_count += len(rb.legal_placements(ta, ca, tb, cb))
    rust_time = time.perf_counter() - t0

    n_calls = n_positions * reps
    print(f"\n=== legal_placements: {n_positions} positions × {reps} dominoes "
          f"= {n_calls:,} calls ===")
    print(f"  python: {py_time*1e3:8.1f} ms   "
          f"({py_time/n_calls*1e6:7.2f} us/call)   total_moves={py_count:,}")
    print(f"  rust:   {rust_time*1e3:8.1f} ms   "
          f"({rust_time/n_calls*1e6:7.2f} us/call)   total_moves={rust_count:,}")
    print(f"  speedup: {py_time/rust_time:.1f}x")
    if py_count != rust_count:
        print(f"  !! WARNING: move counts differ ({py_count} vs {rust_count}) — "
              f"check equivalence")

    # ── score() benchmark ──
    t0 = time.perf_counter()
    for pb, _ in pairs:
        for _ in range(reps):
            s = pb.score()
            _ = s.total
    py_score_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _, rb in pairs:
        for _ in range(reps):
            t, h, m = rb.score(True, True)
            _ = t + h + m
    rust_score_time = time.perf_counter() - t0

    print(f"\n=== score: {n_calls:,} calls ===")
    print(f"  python: {py_score_time*1e3:8.1f} ms   "
          f"({py_score_time/n_calls*1e6:7.2f} us/call)")
    print(f"  rust:   {rust_score_time*1e3:8.1f} ms   "
          f"({rust_score_time/n_calls*1e6:7.2f} us/call)")
    print(f"  speedup: {py_score_time/rust_score_time:.1f}x")


if __name__ == "__main__":
    n_positions = 100
    reps = 30
    for a in sys.argv[1:]:
        if a.startswith("--positions="):
            n_positions = int(a.split("=", 1)[1])
        elif a.startswith("--reps="):
            reps = int(a.split("=", 1)[1])
    bench(n_positions=n_positions, reps=reps)