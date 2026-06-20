"""
production_probe.py — Phase 1: measure the CPU leaf-production ceiling with
inference and IPC removed entirely.

WHY
  Every throughput number so far has been entangled with the inference funnel
  and the GPU.  This probe answers the one governing question:
      "If leaf evaluation were FREE, how many leaves/sec can the CPU tree
       search manufacture — on one core, and across N cores?"
  That ceiling decides whether the next win is widening the IPC funnel
  (batched-send) or speeding up the tree work itself (encode_state / legal).

HOW
  A mock evaluator returns (0.0, zeros) instantly: uniform priors over legal
  actions, zero value, NO network, NO IPC, NO server.  The MCTS still does all
  of its real work — encode_state, legal_actions, PUCT selection, backup,
  step(), redeterminize — so the measured rate is the production ceiling.

  Two independent measurements:
    profile  — ONE in-process self-play game under cProfile; prints top
               functions by self-time and the share of the known hot paths.
    scale    — N INDEPENDENT worker processes (no server, no queues, no
               pickling between them), each running mock self-play for a fixed
               wall-clock budget; reports per-worker and aggregate leaves/sec
               and the scaling efficiency vs one worker.

INTERPRETING (vs the measured A3 plateau of ~2,250 evals/s):
  scale total ~2.5k–4k     → CPU tree work is already the wall; the funnel is
                             NOT the main problem. Optimize encode_state/legal.
  scale total ~8k–15k      → the IPC funnel is leaving a lot on the table;
                             batched-send (N leaves/request) is worth building.
  1 worker fast, N poor    → multiprocessing / memory-bandwidth /
                             oversubscription; more workers won't help.

CAVEATS
  - Uniform priors make a broad, shallow tree; real-net trees concentrate and
    go deeper, so real descents are a bit longer. Treat this as an OPTIMISTIC
    ceiling for per-leaf select cost. encode_state runs ~once per expanded leaf
    regardless of shape, so its share is representative.
  - cProfile covers only the single in-process run; the scaling run is pure
    wall-clock (no per-call instrumentation, no profiler overhead).
  - Compare the 1-worker rate against serial+leaf_batch=6 (~541 evals/s): if
    they're close, the GPU forward was already negligible in-process and tree
    work is the true wall.

USAGE (PowerShell, one line each)
  python -m games.kingdomino.production_probe --mode profile --sims 800
  python -m games.kingdomino.production_probe --mode scale --sims 800 --workers 1,2,4,8 --secs 8
  python -m games.kingdomino.production_probe --mode both --sims 800 --workers 1,2,4,8 --secs 8
"""
from __future__ import annotations

import argparse
import cProfile
import multiprocessing as mp
import pstats
import random
import time
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np


# ── mock evaluator (module-level so it is picklable under spawn) ─────────────
class _CountingMock:
    """4-arg Evaluator seam: (mb, ob, flat, idxs) -> (value, gathered_logits).
    Returns zeros instantly (uniform priors, zero value) and counts calls.
    Each call == one leaf evaluation, so the counter is an exact leaf count."""
    __slots__ = ("n",)

    def __init__(self):
        self.n = 0

    def __call__(self, mb, ob, flat, idxs):
        self.n += 1
        return 0.0, np.zeros(len(idxs), dtype=np.float32)


class _CountingMockBatched:
    """Batched evaluator seam for the RUST engine: (mb (K,...), ob, flat (K,261),
    idxs_list) -> (values (K,) f64, [zeros f64]).  Counts total leaves (sum of K)
    so the counter is an exact leaf count, matching _CountingMock's semantics."""
    __slots__ = ("n",)

    def __init__(self):
        self.n = 0

    def __call__(self, mb, ob, flat, idxs_list):
        k = len(idxs_list)
        self.n += k
        values = np.zeros(k, dtype=np.float64)
        gathered = [np.zeros(len(idxs_list[i]), dtype=np.float64) for i in range(k)]
        return values, gathered


def _play_mock_games(engine: str, sims: int, n_det: int, temp_moves: int,
                     leaf_batch: int, batch_slots: int, time_budget_s: float,
                     base_seed: int) -> Tuple[int, int, float]:
    """Play mock self-play games until the time budget elapses.
    Returns (total_leaves, games_played, elapsed_seconds).  engine selects the
    Python AlphaZeroMCTS path or the Rust RustMCTS path (in-process leaf eval)."""
    from games.kingdomino.self_play import _game_rngs

    if engine == "batched":
        if n_det != 1:
            raise ValueError("batched probe requires determinizations=1")
        import kingdomino_rust
        mock = _CountingMockBatched()
        t0 = time.time()
        games = 0
        while time.time() - t0 < time_budget_s:
            n = max(1, batch_slots)
            bm = kingdomino_rust.BatchedMCTS(
                n, n, base_seed + games, sims,
                leaf_batch=max(1, leaf_batch), virtual_loss=1,
                cpuct=1.5, fpu=0.0, dirichlet_eps=0.25,
                temp_moves=temp_moves)
            while not bm.done():
                mb, ob, flat, idxs_list = bm.step()
                values, gathered = mock(mb, ob, flat, idxs_list)
                bm.update(values, gathered)
            games += n
        return mock.n, games, time.time() - t0

    if engine == "rust":
        import kingdomino_rust
        from games.kingdomino.self_play import play_selfplay_game_rust
        mock = _CountingMockBatched()
        rm = kingdomino_rust.RustMCTS()
        t0 = time.time()
        games = 0
        while time.time() - t0 < time_budget_s:
            seed = base_seed + games
            py_rng, np_rng = _game_rngs(seed)
            play_selfplay_game_rust(
                rm, mock, n_simulations=sims, n_determinizations=n_det,
                temp_moves=temp_moves, c_puct=1.5, dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25, leaf_batch=leaf_batch, virtual_loss=1,
                seed=seed, py_rng=py_rng, np_rng=np_rng)
            games += 1
        return mock.n, games, time.time() - t0

    from games.kingdomino.mcts_az import AlphaZeroMCTS
    from games.kingdomino.self_play import play_selfplay_game
    mock = _CountingMock()
    mcts = AlphaZeroMCTS(mock, n_simulations=sims, virtual_loss=1)
    t0 = time.time()
    games = 0
    while time.time() - t0 < time_budget_s:
        seed = base_seed + games
        py_rng, np_rng = _game_rngs(seed)
        play_selfplay_game(mcts, n_determinizations=n_det, temp_moves=temp_moves,
                           seed=seed, py_rng=py_rng, np_rng=np_rng,
                           leaf_batch=leaf_batch)
        games += 1
    return mock.n, games, time.time() - t0


def _scale_worker(args) -> Dict:
    engine, sims, n_det, temp_moves, leaf_batch, batch_slots, time_budget_s, base_seed, wid = args
    # one compute thread per worker (match the A3 worker discipline)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    leaves, games, secs = _play_mock_games(
        engine, sims, n_det, temp_moves, leaf_batch, batch_slots, time_budget_s,
        base_seed + wid * 1_000_003)
    return {"wid": wid, "leaves": leaves, "games": games, "secs": secs}


# ── profile mode ─────────────────────────────────────────────────────────────
_HOT = {
    "encode_state": "encode_state (total)",
    "_encode_board_spatial": "  board spatial encode",
    "_compute_bag": "  compute_bag",
    "_domino_in_hand": "  domino_in_hand",
    "legal_placements": "legal_placements",
    "is_legal_placement": "is_legal_placement",
    "legal_actions": "legal_actions",
    "redeterminize": "redeterminize",
    "step": "GameState.step",
    "copy": "copy (Board/GameState)",
    "_select_child": "PUCT _select_child",
    "_simulate": "_simulate (serial)",
    "_simulate_batch": "_simulate_batch (LP)",
    "compute_target_z": "compute_target_z",
}


def run_profile(engine: str, sims: int, n_det: int, temp_moves: int, leaf_batch: int,
                batch_slots: int, n_games: int) -> None:
    print(f"\n=== PROFILE [{engine}]: {n_games} game(s), sims={sims}, n_det={n_det}, "
          f"leaf_batch={leaf_batch} (mock evaluator, in-process) ===")
    from games.kingdomino.self_play import _game_rngs

    if engine == "batched":
        import kingdomino_rust
        mock = _CountingMockBatched()

        def _run():
            bm = kingdomino_rust.BatchedMCTS(
                max(1, batch_slots), n_games, 20000, sims,
                leaf_batch=max(1, leaf_batch), virtual_loss=1,
                cpuct=1.5, fpu=0.0, dirichlet_eps=0.25,
                temp_moves=temp_moves)
            while not bm.done():
                mb, ob, flat, idxs_list = bm.step()
                values, gathered = mock(mb, ob, flat, idxs_list)
                bm.update(values, gathered)

    elif engine == "rust":
        # cProfile only sees Python frames; the Rust tree work is opaque to it,
        # so the hot-path table will be dominated by the leaf-eval boundary and
        # encode/asarray.  Use scale mode for the headline leaves/sec number.
        import kingdomino_rust
        from games.kingdomino.self_play import play_selfplay_game_rust
        mock = _CountingMockBatched()
        rm = kingdomino_rust.RustMCTS()

        def _run():
            for g in range(n_games):
                py_rng, np_rng = _game_rngs(20000 + g)
                play_selfplay_game_rust(
                    rm, mock, n_simulations=sims, n_determinizations=n_det,
                    temp_moves=temp_moves, c_puct=1.5, dirichlet_alpha=0.3,
                    dirichlet_epsilon=0.25, leaf_batch=leaf_batch, virtual_loss=1,
                    seed=20000 + g, py_rng=py_rng, np_rng=np_rng)
    else:
        from games.kingdomino.mcts_az import AlphaZeroMCTS
        from games.kingdomino.self_play import play_selfplay_game
        mock = _CountingMock()
        mcts = AlphaZeroMCTS(mock, n_simulations=sims, virtual_loss=1)

        def _run():
            for g in range(n_games):
                py_rng, np_rng = _game_rngs(20000 + g)
                play_selfplay_game(mcts, n_determinizations=n_det,
                                   temp_moves=temp_moves, seed=20000 + g,
                                   py_rng=py_rng, np_rng=np_rng, leaf_batch=leaf_batch)

    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    _run()
    pr.disable()
    dt = time.time() - t0

    leaves = mock.n
    print(f"leaves={leaves}  wall={dt:.2f}s  "
          f"rate={leaves/dt:,.0f} leaves/s (single thread, profiler ON — "
          f"true rate is higher; see scale mode)")

    st = pstats.Stats(pr)
    total_tt = sum(v[2] for v in st.stats.values())  # total self-time

    # Known hot paths: self-time (tottime) AND subtree time (cumtime).
    # Self-time = work done IN the function; subtree = it plus everything it
    # calls. legal_placements has small self-time but a huge subtree, because
    # the cost lives in is_legal_placement → is_empty/half_connects.
    print(f"\nHot-path shares  (self-time of {total_tt:.2f}s total; "
          f"subtree as % of {dt:.2f}s wall):")
    print(f"  {'function':<28} {'self':>8} {'self%':>6}   {'subtree':>8} {'wall%':>6}")
    agg_tt: Dict[str, float] = {k: 0.0 for k in _HOT}
    agg_ct: Dict[str, float] = {k: 0.0 for k in _HOT}
    for (fn_file, fn_line, fn_name), (cc, nc, tt, ct, callers) in st.stats.items():
        for key in _HOT:
            if fn_name == key:
                agg_tt[key] += tt
                agg_ct[key] += ct
    for key, label in _HOT.items():
        if agg_ct[key] <= 0:
            continue
        self_pct = 100.0 * agg_tt[key] / total_tt if total_tt else 0.0
        wall_pct = 100.0 * agg_ct[key] / dt if dt else 0.0
        print(f"  {label:<28} {agg_tt[key]:7.3f}s {self_pct:5.1f}%   "
              f"{agg_ct[key]:7.3f}s {wall_pct:5.1f}%")

    print("\nTop 18 functions by self-time:")
    buf = StringIO()
    st_sorted = pstats.Stats(pr, stream=buf)
    st_sorted.sort_stats("tottime").print_stats(18)
    # keep only the data rows for a compact view
    for line in buf.getvalue().splitlines():
        s = line.strip()
        if s and (s[0].isdigit() or "percall" in line or "function calls" in line
                  or "Ordered by" in line):
            print("  " + line.rstrip())


# ── scale mode ───────────────────────────────────────────────────────────────
def run_scale(engine: str, sims: int, n_det: int, temp_moves: int, leaf_batch: int,
              batch_slots: int, worker_counts: List[int], secs: float) -> None:
    print(f"\n=== SCALE [{engine}]: independent processes, NO server/IPC, "
          f"sims={sims}, n_det={n_det}, leaf_batch={leaf_batch}, "
          f"{secs:.0f}s/worker ===")
    print(f"{'workers':>8} {'leaves/s total':>15} {'leaves/s/worker':>17} "
          f"{'efficiency':>11} {'games':>7}")
    base_rate = None
    for w in worker_counts:
        args = [(engine, sims, n_det, temp_moves, leaf_batch, batch_slots,
                 secs, 50000 + rep, rep)
                for rep in range(w)]
        ctx = mp.get_context("spawn")
        t0 = time.time()
        with ctx.Pool(processes=w) as pool:
            results = pool.map(_scale_worker, args)
        wall = time.time() - t0
        total_leaves = sum(r["leaves"] for r in results)
        total_games = sum(r["games"] for r in results)
        # each worker ran ~secs; aggregate rate = total leaves / mean worker secs
        mean_secs = np.mean([r["secs"] for r in results])
        total_rate = total_leaves / mean_secs
        per_worker = total_rate / w
        if base_rate is None:
            base_rate = per_worker
        eff = 100.0 * per_worker / base_rate if base_rate else 100.0
        print(f"{w:>8} {total_rate:>15,.0f} {per_worker:>17,.0f} "
              f"{eff:>10.0f}% {total_games:>7}")
    print("\n  efficiency = per-worker rate vs the 1-worker rate (100% = "
          "perfect scaling; falling % = oversubscription / memory bandwidth).")
    print("  Compare total against the A3 plateau (~2,250 evals/s) and the "
          "GPU ceiling (~45,000 evals/s).")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-1 CPU leaf-production probe (inference removed).")
    p.add_argument("--mode", choices=["profile", "scale", "both"], default="both")
    p.add_argument("--engine", choices=["python", "rust", "batched"], default="python",
                   help="python = AlphaZeroMCTS; rust = RustMCTS (in-process leaf "
                        "eval, no IPC); batched = synchronized Rust BatchedMCTS.")
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--determinizations", type=int, default=1)
    p.add_argument("--temp_moves", type=int, default=10)
    p.add_argument("--leaf_batch", type=int, default=1,
                   help="1 = pure serial tree work (the clean ceiling).")
    p.add_argument("--batch_slots", type=int, default=32,
                   help="concurrent slots for --engine batched.")
    p.add_argument("--workers", type=str, default="1,2,4,8",
                   help="comma-separated worker counts for scale mode.")
    p.add_argument("--secs", type=float, default=8.0,
                   help="wall-clock budget per worker in scale mode.")
    p.add_argument("--profile_games", type=int, default=1)
    return p


def main() -> None:
    a = _build_argparser().parse_args()
    if a.mode in ("profile", "both"):
        run_profile(a.engine, a.sims, a.determinizations, a.temp_moves,
                    a.leaf_batch, a.batch_slots, a.profile_games)
    if a.mode in ("scale", "both"):
        wc = [int(x) for x in a.workers.split(",") if x.strip()]
        run_scale(a.engine, a.sims, a.determinizations, a.temp_moves,
                  a.leaf_batch, a.batch_slots, wc, a.secs)


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
