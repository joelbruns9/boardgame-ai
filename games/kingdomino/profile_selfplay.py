"""
DIAGNOSTIC — Profiling script for the pre-Rust A3 worker architecture.
The bottlenecks identified here (IPC overhead, Python tree work) were
addressed by the Rust BatchedMCTS engine in self_play.py.

Not relevant for current training pipeline. Kept for historical reference.

────────────────────────────────────────────────────────────────────────
Original module docstring follows:

profile_selfplay.py — find where the WORKER's per-leaf Python time goes.

The A3 workers do MCTS tree work + encode_state + game-engine steps, then ship
the leaf to the server for the forward.  The forward and the IPC are NOT the
worker's CPU cost.  So to profile the worker bottleneck we run real self-play
with a STUB evaluator (returns zeros instantly) in place of the network: what's
left is exactly the per-leaf CPU the workers actually pay, and the measured
rate is the single-core pure-tree-work ceiling.

Two outputs:
  1. Clean (un-profiled) timing → leaf-evals/sec on ONE core with inference
     free.  Compare to the whole system's evals/sec (~1700 across 12 workers,
     ~140/worker): if this ceiling is far above ~140, the workers are gated by
     inference round-trip latency, not tree-work CPU; if it's close, you're
     genuinely tree-work bound and a faster engine pays off directly.
  2. cProfile breakdown by self-time (tottime) → which functions burn the CPU,
     i.e. exactly what to optimize in place or port to a compiled extension.

Run:  python -m games.kingdomino.profile_selfplay --games 2 --sims 200
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np

from games.kingdomino.action_codec import NUM_JOINT_ACTIONS
from games.kingdomino.mcts_az import AlphaZeroMCTS
from games.kingdomino.self_play import play_selfplay_game, _game_rngs


# Stub evaluator: stands in for the IPC+forward so the profile reflects ONLY
# the worker's tree/engine/encode cost.  Reuse one zero array → near-zero cost.
_ZERO_LOGITS = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)
_N_EVALS = [0]


def _stub_eval(mb, ob, flat, idxs):
    _N_EVALS[0] += 1
    return 0.0, np.zeros(len(idxs), dtype=np.float32)


def _run(n_games: int, sims: int, det: int, temp_moves: int, base_seed: int) -> None:
    mcts = AlphaZeroMCTS(_stub_eval, c_puct=1.5, n_simulations=sims,
                         dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    for i in range(n_games):
        seed = base_seed + i
        py_rng, np_rng = _game_rngs(seed)
        play_selfplay_game(mcts, n_determinizations=det, temp_moves=temp_moves,
                           seed=seed, py_rng=py_rng, np_rng=np_rng)


def main() -> None:
    p = argparse.ArgumentParser(description="Worker hot-path CPU profiler")
    p.add_argument("--games", type=int, default=2, help="games for clean timing")
    p.add_argument("--profile_games", type=int, default=1,
                   help="games under cProfile (keep small; cProfile is slow)")
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--determinizations", type=int, default=1)
    p.add_argument("--temp_moves", type=int, default=20)
    p.add_argument("--top", type=int, default=25)
    a = p.parse_args()

    # ── 1. Clean timing: single-core pure-tree-work throughput ──
    _N_EVALS[0] = 0
    t0 = time.perf_counter()
    _run(a.games, a.sims, a.determinizations, a.temp_moves, base_seed=1_000)
    dt = time.perf_counter() - t0
    evals = _N_EVALS[0]
    print(f"\n[clean] {a.games} games, sims={a.sims}, det={a.determinizations}: "
          f"{evals} leaf-evals in {dt:.2f}s")
    print(f"[clean] single-core pure-tree-work ceiling: {evals/dt:,.0f} leaf-evals/sec")
    print(f"        (system did ~1700/sec across 12 workers ≈ ~140/worker; "
          f"if this ceiling >> 140, workers are latency-bound, not tree-CPU-bound)")

    # ── 2. cProfile breakdown by self-time ──
    print(f"\n[profile] running {a.profile_games} game(s) under cProfile…")
    _N_EVALS[0] = 0
    pr = cProfile.Profile()
    pr.enable()
    _run(a.profile_games, a.sims, a.determinizations, a.temp_moves, base_seed=9_000)
    pr.disable()

    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf).strip_dirs()
    print(f"\n===== TOP {a.top} BY SELF-TIME (tottime — where the CPU actually goes) =====")
    st.sort_stats("tottime").print_stats(a.top)
    print("\n".join(buf.getvalue().splitlines()))

    buf2 = io.StringIO()
    st2 = pstats.Stats(pr, stream=buf2).strip_dirs()
    print(f"\n===== TOP 15 BY CUMULATIVE TIME (call hierarchy) =====")
    st2.sort_stats("cumulative").print_stats(15)
    print("\n".join(buf2.getvalue().splitlines()[:32]))


if __name__ == "__main__":
    main()