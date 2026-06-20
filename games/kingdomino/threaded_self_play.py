"""
DEPRECATED — Use self_play.py with engine=batched_open_loop instead.

This module implements a threaded self-play topology (A1 architecture)
that predates the Rust BatchedMCTS engine. It uses Python threads to
run multiple games concurrently, with a shared in-process inference
server.

Why deprecated:
- Python's GIL prevents true parallelism for CPU-bound tree work
- sample_batch threading confirmed 0.44-0.50x (2x regression) due to
  GIL contention at fine granularity
- The Rust engine achieves the same batching benefit without threads
  by running 32 game slots simultaneously in a single thread

Kept for reference. Do not use for training.

────────────────────────────────────────────────────────────────────────
Original module docstring follows:

threaded_self_play.py — single-process, multi-threaded AlphaZero self-play (A1).

Many game threads run concurrently in ONE process, all sharing a
LocalInferenceService.  While a thread is blocked waiting for its leaf
evaluation, the batcher coalesces requests from the other threads into one GPU
forward — so the GPU sees batches of up to `game_threads` instead of 1, and the
threads cover each other's inference waits.

This is the A1 design from the plan.  It reuses, unchanged:
  - AlphaZeroMCTS (via the evaluator seam — the client IS the evaluator)
  - play_selfplay_game, _game_rngs, train_step, benchmark_vs, AZPlayer,
    ReplayBuffer, save_checkpoint  (all from self_play.py)

Graduation to A3 (2-4 worker processes x many threads) is a backend swap:
replace LocalInferenceService with RemoteInferenceService and run this game pool
inside each worker process.  MCTS, the game pool, and the client call sites do
not change — see inference_service.py.

Does NOT import evaluation.py.
"""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from typing import List, Tuple

import numpy as np
import torch

from games.kingdomino.network import KingdominoNet
from games.kingdomino.mcts_az import AlphaZeroMCTS
from games.kingdomino.self_play import (
    SelfPlayConfig, ReplayBuffer, Example,
    play_selfplay_game, train_step, benchmark_vs,
    AZPlayer, save_checkpoint, make_mcts, _game_rngs,
)
from games.kingdomino.inference_service import LocalInferenceService
from games.kingdomino.bots import GreedyBot


# ─────────────────────────────────────────────────────────────────────────────
# Threaded game generation
# ─────────────────────────────────────────────────────────────────────────────
def run_threaded_self_play_games(
    service: LocalInferenceService,
    cfg: SelfPlayConfig,
    n_games: int,
    game_seed_start: int,
    game_threads: int,
    fail_fast: bool = True,
    verbose: bool = False,
) -> Tuple[List[List[Example]], List[Tuple[int, int]]]:
    """Generate `n_games` self-play games using `game_threads` concurrent threads.

    Each thread builds its own AlphaZeroMCTS (separate trees) but shares the one
    inference client / service.  Per-game RNGs come from _game_rngs(seed), so a
    game's training targets depend only on its seed.

    Determinism: results are sorted by seed before returning, so replay-buffer
    insertion order is independent of thread scheduling (and matches the serial
    loop's game_seed order) — important for run-to-run reproducibility.

    fail_fast (default True): if any game raises, abort the iteration with a
    RuntimeError rather than silently training on a biased subset.  Set False to
    tolerate failures (e.g. long robustness runs); even then, zero successful
    games is always fatal.
    """
    seed_q: "queue.Queue[int]" = queue.Queue()
    for i in range(n_games):
        seed_q.put(game_seed_start + i)

    # store (seed, examples, scores) so we can sort deterministically
    results: List[Tuple[int, List[Example], Tuple[int, int]]] = []
    results_lock = threading.Lock()
    errors: List[str] = []
    done_count = [0]

    client = service.make_client(worker_id=0)

    def worker() -> None:
        mcts = AlphaZeroMCTS(
            client,
            c_puct=cfg.c_puct,
            n_simulations=cfg.n_simulations,
            dirichlet_alpha=cfg.dirichlet_alpha,
            dirichlet_epsilon=cfg.dirichlet_epsilon,
        )
        while True:
            try:
                seed = seed_q.get_nowait()
            except queue.Empty:
                return
            try:
                py_rng, np_rng = _game_rngs(seed)
                examples, scores = play_selfplay_game(
                    mcts,
                    n_determinizations=cfg.n_determinizations,
                    temp_moves=cfg.temp_moves,
                    seed=seed,
                    py_rng=py_rng,
                    np_rng=np_rng,
                )
                with results_lock:
                    results.append((seed, examples, scores))
                    done_count[0] += 1
                    if verbose and n_games >= 20 and \
                            done_count[0] % max(1, n_games // 10) == 0:
                        print(f"    {done_count[0]}/{n_games} games done …")
            except Exception as exc:
                with results_lock:
                    errors.append(f"seed {seed}: {exc!r}")

    threads = [threading.Thread(target=worker, name=f"game-{i}", daemon=True)
               for i in range(max(1, game_threads))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Fail-fast: a real bug could fail many games; silently training on the
    # survivors biases the data, so abort by default.
    if errors and fail_fast:
        msg = "\n".join(errors[:10])
        raise RuntimeError(
            f"{len(errors)}/{n_games} self-play game(s) failed:\n{msg}")
    if errors and verbose:
        print(f"    [warning] {len(errors)} game(s) failed; "
              f"{len(results)} succeeded (fail_fast=False)")
        for e in errors[:3]:
            print(f"      - {e}")
    if not results:
        raise RuntimeError(
            f"All {n_games} self-play games failed; nothing to train on.")

    results.sort(key=lambda r: r[0])           # deterministic insertion order
    all_examples = [r[1] for r in results]
    all_scores = [r[2] for r in results]
    return all_examples, all_scores


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def run_threaded_self_play_training(
    cfg: SelfPlayConfig,
    game_threads: int = 16,
    max_batch: int = 16,
    max_wait_ms: float = 3.0,
    fail_fast: bool = True,
    verbose: bool = True,
) -> dict:
    """A1 training loop: threaded self-play → train → benchmark → checkpoint.

    Drop-in analogue of run_self_play_training, with in-process batched
    inference.  Same SelfPlayConfig and checkpoint format.

    `max_batch` should be ~`game_threads`: a thread holds at most one eval in
    flight, so the realized batch can never exceed the number of game threads.
    Setting max_batch >> game_threads wastes nothing but does nothing.
    """
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Training network (gradients here).
    net = KingdominoNet(channels=cfg.channels, blocks=cfg.blocks,
                        bilinear_dim=cfg.bilinear_dim).to(cfg.device)
    if cfg.warm_start_path:
        ckpt = torch.load(cfg.warm_start_path, map_location=cfg.device)
        sd = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(sd)
        if verbose:
            print(f"Warm-started from {cfg.warm_start_path}")

    if max_batch < game_threads and verbose:
        print(f"[note] max_batch ({max_batch}) < game_threads ({game_threads}); "
              f"batches are capped by game_threads, so max_batch≈game_threads is "
              f"the useful setting.")

    # The service deep-copies `net` and updates only via update_weights, so its
    # inference network is independent of the trainer's — identical semantics to
    # the remote backend.
    service = LocalInferenceService(net, device=cfg.device,
                                    max_batch=max_batch, max_wait_ms=max_wait_ms)
    service.start()

    buffer = ReplayBuffer(cfg.buffer_capacity)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    np_rng = np.random.default_rng(cfg.seed)
    history = {"value_loss": [], "policy_loss": [], "benchmark": [],
               "selfplay_score_diff": [], "games_per_sec": []}
    game_seed = cfg.seed * 1_000_003

    if verbose:
        print(f"Threaded self-play: game_threads={game_threads}, "
              f"max_batch={max_batch}, max_wait_ms={max_wait_ms}, "
              f"device={cfg.device}")

    try:
        for it in range(1, cfg.n_iterations + 1):
            if verbose:
                print(f"\n{'='*60}\nIteration {it}/{cfg.n_iterations}\n{'='*60}")

            # Push current weights and BARRIER until they are actually serving,
            # so the first self-play inference of this iteration cannot use the
            # previous iteration's weights.  Then reset per-iteration stats.
            target_version = service.update_weights(net.state_dict())
            service.wait_for_version(target_version, timeout_s=30.0)
            service.reset_stats()

            # ── 1. Self-play (threaded) ──
            t0 = time.time()
            all_examples, all_scores = run_threaded_self_play_games(
                service, cfg, n_games=cfg.games_per_iteration,
                game_seed_start=game_seed, game_threads=game_threads,
                fail_fast=fail_fast,
                verbose=verbose and cfg.games_per_iteration >= 20,
            )
            game_seed += cfg.games_per_iteration
            for exs in all_examples:
                buffer.add(exs)

            elapsed = time.time() - t0
            gps = len(all_examples) / elapsed if elapsed > 0 else 0.0
            history["games_per_sec"].append(gps)
            if all_scores:
                history["selfplay_score_diff"].append(
                    float(np.mean([s[0] - s[1] for s in all_scores])))

            if verbose:
                s = service.stats()
                print(f"  self-play: {len(all_examples)}/{cfg.games_per_iteration} "
                      f"games ({gps:.2f} games/sec), buffer={len(buffer)}")
                print(f"  inference: mean_batch={s['mean_batch']:.1f}/"
                      f"{s['max_batch_cap']} (fill {s['fill_ratio']:.0%}), "
                      f"service_busy={s['service_busy_fraction']:.0%}, "
                      f"{s['requests_per_sec']:.0f} evals/sec")

            # ── 2. Train ──
            if len(buffer) < cfg.min_buffer_to_train:
                if verbose:
                    print(f"  buffer below warmup ({len(buffer)}/"
                          f"{cfg.min_buffer_to_train}); skipping training")
            elif cfg.train_steps_per_iteration <= 0:
                if verbose:
                    print("  train: train_steps_per_iteration=0; skipping")
            else:
                net.train()
                v_sum = p_sum = 0.0
                for _ in range(cfg.train_steps_per_iteration):
                    batch = buffer.sample_batch(cfg.batch_size, np_rng,
                                                device=cfg.device,
                                                augment_d4=cfg.augment)
                    v, p = train_step(net, batch, optimizer,
                                      value_weight=cfg.value_weight,
                                      policy_weight=cfg.policy_weight,
                                      grad_clip=cfg.grad_clip)
                    v_sum += v
                    p_sum += p
                n = cfg.train_steps_per_iteration
                history["value_loss"].append(v_sum / n)
                history["policy_loss"].append(p_sum / n)
                net.eval()
                if verbose:
                    print(f"  train: value_loss={v_sum/n:.4f}  "
                          f"policy_loss={p_sum/n:.4f}")

            # ── 3. Benchmark + checkpoint ──
            if cfg.benchmark_every and it % cfg.benchmark_every == 0:
                net.eval()
                bench_dets = (cfg.benchmark_determinizations
                              if cfg.benchmark_determinizations is not None
                              else cfg.n_determinizations)
                az = AZPlayer(make_mcts(net, cfg, cfg.benchmark_sims),
                              n_determinizations=bench_dets, np_rng=np_rng)
                stats = benchmark_vs(az, GreedyBot(), cfg.benchmark_seeds,
                                     seed=cfg.seed + 99, verbose=False)
                history["benchmark"].append((it, stats["az_win_rate"]))
                if verbose:
                    print(f"  benchmark vs Greedy: {stats['az_win_rate']:.1%} "
                          f"({stats['az_wins']}-{stats['draws']}-"
                          f"{stats['opp_wins']} over {stats['n_games']})")

            if cfg.checkpoint_dir:
                os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                save_checkpoint(
                    os.path.join(cfg.checkpoint_dir, f"iter_{it:04d}.pt"),
                    net, cfg, it, history)
    finally:
        service.stop()
        if verbose:
            print("Inference service stopped.")

    return {"net": net, "history": history, "buffer": buffer}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Threaded (A1) AlphaZero self-play")
    # Threading / inference
    p.add_argument("--game_threads", type=int, default=16,
                   help="Concurrent self-play game threads (≈ inference batch).")
    p.add_argument("--max_batch", type=int, default=None,
                   help="Inference batch cap (default: = game_threads).")
    p.add_argument("--max_wait_ms", type=float, default=3.0)
    # Network
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--bilinear_dim", type=int, default=64)
    # Search
    p.add_argument("--sims", type=int, default=50)
    p.add_argument("--determinizations", type=int, default=1)
    p.add_argument("--c_puct", type=float, default=None,
                   help="PUCT exploration constant (default: SelfPlayConfig default)")
    p.add_argument("--dirichlet_alpha", type=float, default=None)
    p.add_argument("--dirichlet_epsilon", type=float, default=None)
    p.add_argument("--temp_moves", type=int, default=None,
                   help="Plies of temperature-1 sampling before greedy")
    # Loop
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--games_per_iter", type=int, default=200)
    p.add_argument("--train_steps", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--grad_clip", type=float, default=None,
                   help="Max global grad norm; <=0 disables")
    p.add_argument("--value_weight", type=float, default=None)
    p.add_argument("--policy_weight", type=float, default=None)
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--min_buffer", type=int, default=None)
    # Benchmark
    p.add_argument("--benchmark_seeds", type=int, default=20)
    p.add_argument("--benchmark_sims", type=int, default=50)
    p.add_argument("--benchmark_every", type=int, default=5)
    p.add_argument("--benchmark_determinizations", type=int, default=None)
    # Misc
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warm_start", default=None)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--allow_failed_games", action="store_true",
                   help="Tolerate per-game failures instead of fail-fast")
    a = p.parse_args()

    game_threads = a.game_threads
    max_batch = a.max_batch if a.max_batch is not None else game_threads
    min_buf = a.min_buffer if a.min_buffer is not None else a.games_per_iter * 52

    # Only override SelfPlayConfig defaults for flags the user actually set.
    optional = {}
    for name in ("c_puct", "dirichlet_alpha", "dirichlet_epsilon", "temp_moves",
                 "weight_decay", "grad_clip", "value_weight", "policy_weight",
                 "benchmark_determinizations"):
        val = getattr(a, name)
        if val is not None:
            optional[name] = val

    cfg = SelfPlayConfig(
        channels=a.channels, blocks=a.blocks, bilinear_dim=a.bilinear_dim,
        n_simulations=a.sims, n_determinizations=a.determinizations,
        batch_size=a.batch_size, lr=a.lr, buffer_capacity=a.buffer,
        n_iterations=a.iterations, games_per_iteration=a.games_per_iter,
        train_steps_per_iteration=a.train_steps, min_buffer_to_train=min_buf,
        benchmark_seeds=a.benchmark_seeds, benchmark_sims=a.benchmark_sims,
        benchmark_every=a.benchmark_every, augment=not a.no_augment,
        device=a.device, seed=a.seed, warm_start_path=a.warm_start,
        checkpoint_dir=a.checkpoint_dir, **optional,
    )
    run_threaded_self_play_training(
        cfg, game_threads=game_threads, max_batch=max_batch,
        max_wait_ms=a.max_wait_ms, fail_fast=not a.allow_failed_games,
        verbose=True)