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
import pickle
import random
import time
from datetime import datetime

import numpy as np

from games.kingdomino.game import GameState, Phase
from games.kingdomino.endgame_solver import _rust_state_from_python

# Production value-formula constants (see exact_endgame_solver.md / mcts config).
SCORE_SCALE = 160.0
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

# Finer buckets for --no-time-limit runs, where the distribution is much wider
# (positions run to completion, up to the 120s safety abort).
_BUCKETS_NOLIMIT = [
    ("<100ms", 0.0, 100.0),
    ("100-500ms", 100.0, 500.0),
    ("500ms-1s", 500.0, 1_000.0),
    ("1-3s", 1_000.0, 3_000.0),
    ("3-10s", 3_000.0, 10_000.0),
    ("10-30s", 10_000.0, 30_000.0),
    ("30-120s", 30_000.0, 120_000.0),
    (">120s(abort)", 120_000.0, float("inf")),
]

# --no-time-limit: run each position to completion, but abort any single position
# after this many seconds so one pathological tree cannot stall the whole run.
_NO_LIMIT_ABORT_SECS = 120.0


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


def gen_deck4_positions_network(n, base_seed, *, checkpoint, channels, blocks,
                                bilinear_dim, sims, device):
    """Yield (seed, GameState) for n deck=4 PLACE_AND_SELECT positions reached by
    PLAYING with the trained network (not random play).

    Each game is played move-by-move with OpenLoopMCTS at `sims` simulations; the
    first time the game reaches len(deck)==4 in PLACE_AND_SELECT it is snapshotted
    and yielded (the deck=4 root is captured BEFORE searching it, so the exact
    solver never runs during generation). The early-move temperature schedule
    (τ=1 for the first 20 plies, then greedy) mirrors training self-play, so the
    captured positions match the distribution the network actually reaches.
    """
    import torch
    from games.kingdomino.network import KingdominoNet
    from games.kingdomino.mcts_az import (
        OpenLoopMCTS, make_serial_evaluator, run_pimc_open_loop, select_move,
    )

    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = (ckpt["model_state"]
                  if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt)
    net = KingdominoNet(channels=channels, blocks=blocks, bilinear_dim=bilinear_dim)
    net.load_state_dict(state_dict)
    net.to(device).eval()
    evaluator = make_serial_evaluator(net, device=device)
    # Exact solving disabled: we capture the deck=4 root before it is searched, so
    # the solver would never fire anyway — disabling it just makes that explicit.
    mcts = OpenLoopMCTS(evaluator, n_simulations=sims, dirichlet_epsilon=0.0,
                        exact_endgame_enabled=False)

    produced = 0
    seed = base_seed
    while produced < n:
        np_rng = np.random.default_rng(seed)
        state = GameState.new(seed=seed)
        seed += 1
        move_num = 0
        found = None
        while state.phase != Phase.GAME_OVER:
            if len(state.deck) == 4 and state.phase == Phase.PLACE_AND_SELECT:
                found = state
                break
            legal = state.legal_actions()
            if len(legal) == 1:
                action = legal[0]
            else:
                visit_counts, _, _ = run_pimc_open_loop(
                    mcts, state, add_noise=False, rng=np_rng)
                temp = 1.0 if move_num < 20 else 0.0
                action = select_move(visit_counts, temperature=temp, rng=np_rng)
            state = state.step(action)
            move_num += 1
        if found is not None:
            produced += 1
            if produced % 10 == 0 or produced == n:
                print(f"  generated {produced}/{n} network positions "
                      f"(seed={seed - 1})...", flush=True)
            yield (seed - 1, found)


def _state_snapshot(seed: int, state: GameState) -> dict:
    b0, b1 = state.boards
    castle_x, castle_y = b0.castle_pos
    return {
        "seed": int(seed),
        "deck": list(state.deck),
        "current_row": list(state.current_row),
        "pending_claims": [(int(c.player), int(c.domino_id)) for c in state.pending_claims],
        "next_claims": [(int(c.player), int(c.domino_id)) for c in state.next_claims],
        "phase": int(state.phase),
        "actor_index": int(state.actor_index),
        "initial_pick_count": int(state.initial_pick_count),
        "start_player": int(state.start_player),
        "board0_terrain": b0.terrain.astype("uint8", copy=False).ravel().tolist(),
        "board0_crowns": b0.crowns.astype("uint8", copy=False).ravel().tolist(),
        "board1_terrain": b1.terrain.astype("uint8", copy=False).ravel().tolist(),
        "board1_crowns": b1.crowns.astype("uint8", copy=False).ravel().tolist(),
        "harmony": bool(state.config.harmony),
        "middle_kingdom": bool(state.config.middle_kingdom),
        "castle_x": int(castle_x),
        "castle_y": int(castle_y),
    }


def _rust_state_from_snapshot(snap: dict):
    import kingdomino_rust

    return kingdomino_rust.RustGameState.from_parts(
        list(snap["deck"]),
        list(snap["current_row"]),
        [tuple(c) for c in snap["pending_claims"]],
        [tuple(c) for c in snap["next_claims"]],
        int(snap["phase"]),
        int(snap["actor_index"]),
        int(snap["initial_pick_count"]),
        int(snap["start_player"]),
        list(snap["board0_terrain"]),
        list(snap["board0_crowns"]),
        list(snap["board1_terrain"]),
        list(snap["board1_crowns"]),
        bool(snap.get("harmony", True)),
        bool(snap.get("middle_kingdom", True)),
        int(snap.get("castle_x", 7)),
        int(snap.get("castle_y", 7)),
    )


def _save_positions(path: str, positions):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = [_state_snapshot(seed, state) for seed, state in positions]
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(payload)} positions to {path}", flush=True)


def _load_positions(path: str):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    positions = [(int(snap["seed"]), _rust_state_from_snapshot(snap)) for snap in payload]
    print(f"Loaded {len(positions)} positions from {path}", flush=True)
    return positions


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
    ap.add_argument("--from-network", type=str, default=None,
                    help="Checkpoint path. Generate positions by PLAYING games with "
                         "the network instead of random play.")
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--bilinear-dim", type=int, default=64)
    ap.add_argument("--sims", type=int, default=400,
                    help="MCTS simulations per move when generating positions "
                         "with --from-network.")
    ap.add_argument("--device", type=str, default="cuda",
                    help="torch device for --from-network position generation.")
    ap.add_argument("--no-time-limit", action="store_true",
                    help=f"Run solver to completion (max {_NO_LIMIT_ABORT_SECS:g}s "
                         "per position). Reports the true tree-size distribution.")
    ap.add_argument("--save-positions", type=str, default=None,
                    help="Save generated GameState snapshots to a .pkl file for reuse.")
    ap.add_argument("--load-positions", type=str, default=None,
                    help="Load positions from a .pkl file instead of generating.")
    ap.add_argument("--label", type=str, default=None,
                    help="Append a label to the output CSV filename.")
    ap.add_argument("--ordering", type=str, default="lookahead2_clustered",
                    choices=("baseline", "denial", "lookahead", "lookahead2",
                             "lookahead2_adaptive8", "lookahead2_adaptive",
                             "lookahead2_adaptive16", "lookahead2_adaptive20",
                             "lookahead2_clustered", "lookahead1_clustered",
                             "combined"),
                    help="Rust solver move-ordering variant to benchmark.")
    args = ap.parse_args()
    parallel = not args.serial

    # --no-time-limit overrides --max-secs with the per-position safety abort.
    max_secs = _NO_LIMIT_ABORT_SECS if args.no_time_limit else args.max_secs
    buckets = _BUCKETS_NOLIMIT if args.no_time_limit else _BUCKETS

    loaded_rust = False
    if args.load_positions:
        source = "loaded"
        ckpt_name = ""
        positions = _load_positions(args.load_positions)
        loaded_rust = True
    elif args.from_network:
        source = "network"
        ckpt_name = os.path.splitext(os.path.basename(args.from_network))[0]
        print(f"Generating {args.n} deck=4 positions from network "
              f"({ckpt_name}, sims={args.sims}, device={args.device})...", flush=True)
        positions = list(gen_deck4_positions_network(
            args.n, args.base_seed,
            checkpoint=args.from_network, channels=args.channels,
            blocks=args.blocks, bilinear_dim=args.bilinear_dim,
            sims=args.sims, device=args.device,
        ))
    else:
        source = "random"
        ckpt_name = ""
        print(f"Generating {args.n} real deck=4 positions "
              f"(base_seed={args.base_seed})...", flush=True)
        positions = list(gen_deck4_positions(args.n, args.base_seed))

    if args.save_positions:
        if loaded_rust:
            print("[warn] --save-positions ignored with --load-positions", flush=True)
        else:
            _save_positions(args.save_positions, positions)

    rows = []  # (seed, elapsed_ms, fully_solved, value)
    bench_t0 = time.perf_counter()
    for i, (seed, state) in enumerate(positions, 1):
        rs = state if loaded_rust else _rust_state_from_python(state)
        if rs is None:
            print(f"  [warn] seed={seed}: could not build RustGameState, skipping", flush=True)
            continue
        value, solved, elapsed_secs = rs.measure_endgame_tree(
            max_secs, SCORE_SCALE, MARGIN_GAIN, args.alpha, parallel, args.ordering
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
    label_part = f"_{args.label}" if args.label else ""
    csv_path = os.path.join(out_dir, f"endgame_tail_{ts}{label_part}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "elapsed_ms", "fully_solved", "value", "source", "ordering", "label"])
        for r in rows:
            w.writerow((r[0], r[1], r[2], r[3], source, args.ordering, args.label or ""))

    # ── aggregate ──
    total = len(rows)
    solved_rows = [r for r in rows if r[2]]
    capped_rows = [r for r in rows if not r[2]]
    times_sorted = sorted(r[1] for r in rows)

    # Solve-time histogram (deadline positions land in the final bucket).
    hist = [0] * len(buckets)
    for r in rows:
        ms = r[1]
        capped = not r[2]
        for bi, (_label, lo, hi) in enumerate(buckets):
            if capped and bi == len(buckets) - 1:
                hist[bi] += 1
                break
            if (not capped) and lo <= ms < hi:
                hist[bi] += 1
                break

    worst5 = sorted(rows, key=lambda r: r[1], reverse=True)[:5]

    print()
    mode = "serial" if args.serial else "parallel(YBW)"
    if args.no_time_limit:
        print(f"=== deck=4 endgame solve time (n={total}, alpha={args.alpha}, "
              f"NO TIME LIMIT, solver={mode}, ordering={args.ordering}) ===")
        if source == "network":
            print(f"Source: network positions (checkpoint: {ckpt_name}, "
                  f"sims={args.sims})")
        elif source == "loaded":
            print(f"Source: loaded positions ({args.load_positions})")
        else:
            print(f"Source: random positions (base_seed={args.base_seed})")
        print(f"Fully solved: {len(solved_rows)}/{total} "
              f"({100.0*len(solved_rows)/total:.1f}%)")
        print(f"Hit {max_secs:g}s safety abort: {len(capped_rows)}/{total} "
              f"({100.0*len(capped_rows)/total:.1f}%)")
    else:
        print(f"=== deck=4 endgame solve time (n={total}, alpha={args.alpha}, "
              f"max_secs={max_secs:g}, source={source}, solver={mode}, "
              f"ordering={args.ordering}) ===")
        print(f"Fully solved within deadline: {len(solved_rows)}/{total} "
              f"({100.0*len(solved_rows)/total:.1f}%)")
        print(f"Hit deadline ({max_secs:g}s):     {len(capped_rows)}/{total} "
              f"({100.0*len(capped_rows)/total:.1f}%)")
    print()
    print("Time (ms):   "
          f"p50={_pct(times_sorted,0.50):.0f}  p75={_pct(times_sorted,0.75):.0f}  "
          f"p90={_pct(times_sorted,0.90):.0f}  p95={_pct(times_sorted,0.95):.0f}  "
          f"p99={_pct(times_sorted,0.99):.0f}  max={times_sorted[-1]:.0f}")
    print()
    print("Histogram:")
    hist_max = max(hist) or 1
    for (label, _lo, _hi), count in zip(buckets, hist):
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
