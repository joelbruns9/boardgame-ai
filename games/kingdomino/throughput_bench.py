"""
throughput_bench.py â€” measure self-play GENERATION throughput across the three
backends, to quantify whether the throughput layer actually helps on YOUR
hardware.

It generates a fixed number of self-play games per backend (no training,
weight-static â€” one fixed net), times it, and reports games/sec.  For the A1 and
A3 backends it also prints the inference batch stats, so you can see WHY a config
is or isn't faster:

    mean_batch / cap   â€” average realized batch vs the cap.  Near the cap â‡’ the
                         GPU is being fed; near 1 â‡’ starved.
    fill               â€” mean_batch / cap as a percentage.
    evals/sec          â€” leaf evaluations served per second (the real GPU-feed
                         metric to watch rise vs serial/A1).
    max_batch_seen     â€” largest single batch the server actually assembled.

IMPORTANT: run this on the device you train on.  On a GPU, A3 should show larger
batches and higher evals/sec than A1, and both above serial.  On CPU the batching
win does not exist (a batched CPU forward isn't faster), and IPC overhead can make
A3 look SLOWER than serial â€” so CPU numbers are a mechanism check, not a verdict.

Examples
  # GPU, compare all three at concurrency 16, 64 games each:
  python -m games.kingdomino.throughput_bench --device cuda \
      --games 64 --sims 200 --channels 64 --blocks 6 \
      --game_threads 16 --workers 8 --games_per_worker 2

  # Just A3, sweep games_per_worker by re-running:
  python -m games.kingdomino.throughput_bench --device cuda --run a3 \
      --workers 8 --games_per_worker 4 --games 128 --sims 800
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from games.kingdomino.network import KingdominoNet
from games.kingdomino.mcts_az import (
    AlphaZeroMCTS, make_serial_evaluator, make_batched_evaluator,
)
from games.kingdomino.self_play import (
    SelfPlayConfig, play_selfplay_game, play_selfplay_game_rust,
    play_selfplay_games_batched, make_rust_evaluator, _game_rngs,
    configure_torch_performance,
)


def compile_net_for_inference(
    net, *, backend: str = "inductor", mode: str = "reduce-overhead"
):
    """Compile a live inference net while preserving the normal call contract."""
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    print(f"compiling net with torch.compile(backend={backend!r}, "
          f"mode={mode!r}, dynamic=True)...",
          flush=True)
    kwargs = {"backend": backend, "dynamic": True}
    if backend == "inductor":
        kwargs["mode"] = mode
    return torch.compile(net, **kwargs)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Backends
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_serial(cfg: SelfPlayConfig, net, n_games: int, base: int,
               leaf_batch: int = 1) -> Tuple[float, Optional[dict]]:
    ev = make_serial_evaluator(net, device=cfg.device)
    # Only build the batched evaluator when it will be used (leaf_batch>1);
    # leaf_batch=1 routes through the single evaluator (bit-identical serial).
    bev = make_batched_evaluator(net, device=cfg.device) if leaf_batch > 1 else None
    mcts = AlphaZeroMCTS(
        ev, batched_evaluator=bev, c_puct=cfg.c_puct, n_simulations=cfg.n_simulations,
        dirichlet_alpha=cfg.dirichlet_alpha, dirichlet_epsilon=cfg.dirichlet_epsilon)
    t0 = time.time()
    for i in range(n_games):
        seed = base + i
        py_rng, np_rng = _game_rngs(seed)
        play_selfplay_game(mcts, n_determinizations=cfg.n_determinizations,
                           temp_moves=cfg.temp_moves, seed=seed,
                           py_rng=py_rng, np_rng=np_rng, leaf_batch=leaf_batch)
        done = i + 1
        if done == 1 or done % max(1, n_games // 8) == 0 or done == n_games:
            rate = done / (time.time() - t0)
            tag = f"serial lb={leaf_batch}" if leaf_batch > 1 else "serial"
            print(f"    [{tag}] {done}/{n_games} games ({rate:.3f}/s)", flush=True)
    dt = time.time() - t0
    return n_games / dt if dt > 0 else 0.0, None


def run_open_loop(
    cfg: SelfPlayConfig, net, n_games: int, base: int,
) -> Tuple[float, Optional[dict]]:
    """Open-loop serial self-play throughput (one OpenLoopMCTS per game,
    fresh determinization per simulation — no cross-game batching)."""
    from games.kingdomino.self_play import make_open_loop_mcts

    ol_mcts = make_open_loop_mcts(net, cfg, cfg.n_simulations)
    t0 = time.time()
    for i in range(n_games):
        seed = base + i
        py_rng, np_rng = _game_rngs(seed)
        play_selfplay_game(
            ol_mcts,
            n_determinizations=1,
            temp_moves=cfg.temp_moves,
            seed=seed,
            py_rng=py_rng,
            np_rng=np_rng,
            open_loop=True,
        )
        done = i + 1
        if done == 1 or done % max(1, n_games // 8) == 0 or done == n_games:
            rate = done / (time.time() - t0)
            print(f"    [open_loop] {done}/{n_games} games ({rate:.3f}/s)",
                  flush=True)
    dt = time.time() - t0
    return n_games / dt if dt > 0 else 0.0, None


def run_a1(cfg: SelfPlayConfig, net, n_games: int, base: int,
           game_threads: int, max_batch: int, max_wait_ms: float
           ) -> Tuple[float, Optional[dict]]:
    from games.kingdomino.inference_service import LocalInferenceService
    from games.kingdomino.threaded_self_play import run_threaded_self_play_games

    svc = LocalInferenceService(net, device=cfg.device, max_batch=max_batch,
                                max_wait_ms=max_wait_ms, debug_checks=False).start()
    try:
        v = svc.update_weights(net.state_dict())
        svc.wait_for_version(v, timeout_s=30.0)
        svc.reset_stats()
        t0 = time.time()
        run_threaded_self_play_games(
            svc, cfg, n_games=n_games, game_seed_start=base,
            game_threads=game_threads, fail_fast=True, verbose=False)
        dt = time.time() - t0
        return (n_games / dt if dt > 0 else 0.0), svc.stats()
    finally:
        svc.stop()


def run_a3(cfg: SelfPlayConfig, net, n_games: int, base: int,
           n_workers: int, games_per_worker: int, max_batch: int,
           max_wait_ms: float, reps: int = 1, warmup: int = 0
           ) -> List[Tuple[float, Optional[dict]]]:
    """Run A3 generation `warmup + reps` times, reusing ONE server + pool so
    spawn/CUDA-init cost is paid once and excluded from every timed rep.  The
    workload (game seeds) is held identical across reps so the only thing that
    varies is the system itself â€” that turns the spread of `games/s` into a
    clean read on run-to-run noise and thermal drift.  Returns the per-rep
    (games/s, stats) for the timed reps only (warmup discarded)."""
    from games.kingdomino.inference_service import RemoteInferenceServer
    from games.kingdomino.parallel_self_play import (
        _init_worker, _generate_parallel, _worker_cleanup,
    )
    mk = dict(channels=cfg.channels, blocks=cfg.blocks, bilinear_dim=cfg.bilinear_dim)
    server = RemoteInferenceServer(
        n_workers=n_workers, model_kwargs=mk, device=cfg.device,
        max_batch=max_batch, max_wait_ms=max_wait_ms, debug_checks=False)
    server.start(initial_state_dict=net.state_dict(), wait_until_loaded=True)
    request_q, response_qs = server.worker_handles()
    counter = mp.Value("i", 0)
    pool = mp.Pool(
        processes=n_workers, initializer=_init_worker,
        initargs=(request_q, response_qs, counter, cfg, games_per_worker,
                  120.0, False),
        maxtasksperchild=None)
    results: List[Tuple[float, Optional[dict]]] = []
    try:
        total = warmup + reps
        for r in range(total):
            is_warm = r < warmup
            t0 = time.time()
            _generate_parallel(pool, cfg, n_games=n_games, game_seed_start=base,
                               n_workers=n_workers, fail_fast=True, verbose=False)
            dt = time.time() - t0
            gps = n_games / dt if dt > 0 else 0.0
            st = server.get_stats()
            tag = "warmup    " if is_warm else f"rep {r - warmup + 1:>2}/{reps}"
            print(f"    [{tag}] {gps:>7.4f} games/s   {_fmt_stats(st)}", flush=True)
            if not is_warm:
                results.append((gps, st))
        return results
    finally:
        try:
            pool.map(_worker_cleanup, range(n_workers), chunksize=1)
        except Exception:
            pass
        pool.terminate()
        pool.join()
        server.stop()


def run_rust(cfg: SelfPlayConfig, net, n_games: int, base: int,
             n_threads: int, max_batch: int, max_wait_ms: float,
             reps: int = 1, warmup: int = 0,
             leaf_batch: int = 1) -> List[Tuple[float, Optional[dict]]]:
    """Rust-engine generation.  `n_threads` in-process game threads each run their
    own RustMCTS (which releases the GIL during tree work via py.detach), sharing
    ONE in-process LocalInferenceService that COALESCES leaves across the games
    into batched forwards â€” so mean_batch can rise to ~n_threads.  No remote/IPC
    server.  Timing/reporting match run_a3 and the stats come straight from the
    service, so the row is directly comparable to the Python backends."""
    import kingdomino_rust  # noqa: F401  (per-thread RustMCTS built in body)
    from games.kingdomino.inference_service import LocalInferenceService
    from games.kingdomino.self_play import make_rust_coalescing_evaluator

    svc = LocalInferenceService(net, device=cfg.device, max_batch=max_batch,
                                max_wait_ms=max_wait_ms, debug_checks=False).start()
    try:
        v = svc.update_weights(net.state_dict())
        svc.wait_for_version(v, timeout_s=30.0)
        evaluator = make_rust_coalescing_evaluator(svc.make_client())

        results: List[Tuple[float, Optional[dict]]] = []
        total = warmup + reps
        for r in range(total):
            is_warm = r < warmup
            svc.reset_stats()
            work: "queue.Queue[int]" = queue.Queue()
            for i in range(n_games):
                work.put(base + i)
            done = [0]
            done_lock = threading.Lock()

            def body() -> None:
                rm = kingdomino_rust.RustMCTS()   # per-thread tree
                while True:
                    try:
                        seed = work.get_nowait()
                    except queue.Empty:
                        return
                    py_rng, np_rng = _game_rngs(seed)
                    play_selfplay_game_rust(
                        rm, evaluator,
                        n_simulations=cfg.n_simulations,
                        n_determinizations=cfg.n_determinizations,
                        temp_moves=cfg.temp_moves, c_puct=cfg.c_puct,
                        dirichlet_alpha=cfg.dirichlet_alpha,
                        dirichlet_epsilon=cfg.dirichlet_epsilon,
                        leaf_batch=leaf_batch, virtual_loss=1,
                        seed=seed, py_rng=py_rng, np_rng=np_rng,
                    )
                    with done_lock:
                        done[0] += 1
                        d = done[0]
                    if d == 1 or d % max(1, n_games // 8) == 0 or d == n_games:
                        print(f"    [{'warmup' if is_warm else 'rep'}] "
                              f"{d}/{n_games} games", flush=True)

            threads = [threading.Thread(target=body, name=f"rust-game-{i}", daemon=True)
                       for i in range(n_threads)]
            t0 = time.time()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            dt = time.time() - t0
            gps = n_games / dt if dt > 0 else 0.0
            st = svc.stats()
            tag = "warmup    " if is_warm else f"rep {r - warmup + 1:>2}/{reps}"
            print(f"    [{tag}] {gps:>7.4f} games/s   {_fmt_stats(st)}", flush=True)
            if not is_warm:
                results.append((gps, st))
        return results
    finally:
        svc.stop()


def run_batched(cfg: SelfPlayConfig, net, n_games: int, base: int,
                reps: int = 1, warmup: int = 0
                ) -> List[Tuple[float, Optional[dict]]]:
    """Run the synchronized Rust BatchedMCTS backend."""
    results: List[Tuple[float, Optional[dict]]] = []
    total = warmup + reps
    for r in range(total):
        is_warm = r < warmup
        t0 = time.time()
        _examples, _scores, stats = play_selfplay_games_batched(
            net, cfg, n_games=n_games, game_seed_start=base)
        dt = time.time() - t0
        gps = n_games / dt if dt > 0 else 0.0
        stats = dict(stats)
        stats["requests_per_sec"] = stats.get("total_evals", 0) / dt if dt > 0 else 0.0
        tag = "warmup    " if is_warm else f"rep {r - warmup + 1:>2}/{reps}"
        print(f"    [{tag}] {gps:>7.4f} games/s   {_fmt_stats(stats)}", flush=True)
        total = max(1e-9, stats.get("elapsed", 0.0))
        print(
            f"        timing: step={stats.get('step_sec', 0.0):.1f}s "
            f"({stats.get('step_sec', 0.0)/total:.0%}), "
            f"eval={stats.get('eval_sec', 0.0):.1f}s "
            f"({stats.get('eval_sec', 0.0)/total:.0%}), "
            f"update={stats.get('update_sec', 0.0):.1f}s "
            f"({stats.get('update_sec', 0.0)/total:.0%})",
            flush=True,
        )
        if "eval_forward_sec" in stats:
            print(
                f"        eval: h2d={stats.get('eval_h2d_sec', 0.0):.1f}s, "
                f"forward={stats.get('eval_forward_sec', 0.0):.1f}s, "
                f"readback={stats.get('eval_readback_sec', 0.0):.1f}s, "
                f"calls={stats.get('eval_calls', 0)}",
                flush=True,
            )
        if not is_warm:
            results.append((gps, stats))
    return results


def _summarize_reps(results: List[Tuple[float, Optional[dict]]],
                    games_per_iter: int) -> None:
    """Print the games/s distribution, a robust iteration-time estimate, and a
    thermal-drift flag."""
    import statistics as _st
    gps = [g for g, _ in results]
    n = len(gps)
    mean = _st.mean(gps); med = _st.median(gps)
    sd = _st.pstdev(gps) if n > 1 else 0.0
    cv = 100 * sd / mean if mean else 0.0
    print(f"\n  â”€â”€ baseline over {n} timed rep(s), warmup excluded â”€â”€")
    print(f"    games/s : mean {mean:.4f}  median {med:.4f}  "
          f"min {min(gps):.4f}  max {max(gps):.4f}  std {sd:.4f}  (CV {cv:.1f}%)")
    if med > 0:
        sec = games_per_iter / med
        print(f"    â†’ at the median, a {games_per_iter}-game iteration â‰ˆ "
              f"{sec/60:.1f} min;  40 iterations â‰ˆ {40*sec/3600:.1f} h")
    last = results[-1][1]
    if last:
        print(f"    inference (cumulative snapshot): fill "
              f"{last.get('fill_ratio',0)*100:.0f}%, busy "
              f"{last.get('service_busy_fraction',0)*100:.0f}%, mean_batch "
              f"{last.get('mean_batch',0):.1f}/{last.get('max_batch_cap','?')}, "
              f"{last.get('requests_per_sec',0):.0f} evals/s")
    # thermal-drift heuristic: best rep first, meaningful slide to the worst
    if n >= 3 and gps[0] == max(gps) and (gps[0] - gps[-1]) / gps[0] > 0.05:
        print("    âš  games/s slides downward across reps â€” likely thermal "
              "throttling on the laptop. The sustained (later-rep) rate, not the "
              "first-rep peak, is your real training baseline.")
    if any((s or {}).get('requests_per_sec', 0) == 0 for _, s in results):
        print("    âš  a rep reported 0 evals/s (stats-capture glitch like log "
              "iter 15) â€” exclude it from the baseline rather than averaging it in.")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Reporting
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _fmt_stats(s: Optional[dict]) -> str:
    if not s:
        return f"{'-':>10}{'-':>7}{'-':>10}{'-':>9}{'-':>8}"
    return (f"{s['mean_batch']:>6.1f}/{s['max_batch_cap']:<3}"
            f"{s['fill_ratio']*100:>6.0f}%"
            f"{s['requests_per_sec']:>10.0f}"
            f"{s['max_batch_seen']:>9}"
            f"{s.get('service_busy_fraction', 0.0)*100:>7.0f}%")


def main() -> None:
    p = argparse.ArgumentParser(description="Self-play throughput benchmark")
    p.add_argument("--run", default="serial,a1,a3",
                   help="comma list of backends to run: "
                        "serial,open_loop,a1,a3,batched,batched_open_loop")
    p.add_argument("--games", type=int, default=32, help="games per backend")
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--determinizations", type=int, default=1)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--bilinear_dim", type=int, default=64)
    p.add_argument("--temp_moves", type=int, default=20)
    # A1 concurrency
    p.add_argument("--game_threads", type=int, default=16)
    # A3 concurrency
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--games_per_worker", type=int, default=2)
    # batching
    p.add_argument("--max_batch", type=int, default=None,
                   help="default: max(game_threads, workers*games_per_worker)")
    p.add_argument("--max_wait_ms", type=float, default=3.0)
    p.add_argument("--leaf_batch", type=int, default=1,
                   help="leaf-parallel batch (virtual loss). Applies to the "
                        "SERIAL backend's in-process batched forward. A1/A3 "
                        "route leaves through the inference server and need a "
                        "separate batched-send entry point (not yet wired), so "
                        "they ignore this and stay at leaf_batch=1.")
    p.add_argument("--batch_slots", type=int, default=32,
                   help="concurrent slots for the batched backend")
    p.add_argument("--eval_pad_to_batch", type=int, default=0,
                   help="pad live inference batches to this size; 0 disables. "
                        "Useful for fixed-shape CUDA graph experiments.")
    p.add_argument("--pin_transfer", action="store_true",
                   help="stage batched evaluator inputs in pinned host memory "
                        "before CUDA transfer")
    p.add_argument("--profile_eval_timing", action="store_true",
                   help="synchronize CUDA around evaluator H2D/forward/readback "
                        "to collect detailed timing")
    p.add_argument("--no_tf32", action="store_true",
                   help="disable TF32 CUDA matmul/convolution")
    p.add_argument("--amp_inference", action="store_true",
                   help="use CUDA float16 autocast for batched/rust inference")
    p.add_argument("--compile_net", action="store_true",
                   help="compile the live inference net with torch.compile")
    p.add_argument("--compile_backend", default="inductor",
                   help="torch.compile backend for --compile_net")
    p.add_argument("--compile_mode", default="reduce-overhead",
                   help="torch.compile mode for --compile_net")
    p.add_argument("--device", default="cuda")
    p.add_argument("--engine", choices=["python", "rust"], default="python",
                   help="python = inference-server backends (serial/a1/a3); "
                        "rust = in-process RustMCTS per game thread, NO inference "
                        "server. Concurrency = workers Ã— games_per_worker threads.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--serial_games", type=int, default=8,
                   help="games for the serial baseline (its per-game rate is "
                        "constant, and serial is ~Nx slower than batched "
                        "backends, so keep this small; default 8)")
    p.add_argument("--repeat", type=int, default=1,
                   help="A3 timed reps (reuses one server/pool). Use >=5 for a "
                        "clean baseline; the spread is your noise floor.")
    p.add_argument("--warmup", type=int, default=0,
                   help="A3 untimed warmup reps before timing (excludes CUDA "
                        "init + lets clocks settle). Use >=1 for a baseline.")
    p.add_argument("--games_per_iter", type=int, default=200,
                   help="for the iteration-time estimate in baseline summaries")
    a = p.parse_args()

    runs = [r.strip() for r in a.run.split(",") if r.strip()]
    # With batched-send, each worker request can carry leaf_batch leaves, so the
    # server forward can grow to workers Ã— games_per_worker Ã— leaf_batch. Size the
    # GPU batch cap accordingly so it doesn't throttle the new throughput.
    lb = max(1, a.leaf_batch)
    max_batch = a.max_batch if a.max_batch is not None else max(
        a.game_threads, a.workers * a.games_per_worker * lb)

    torch.manual_seed(a.seed)
    cfg = SelfPlayConfig(
        channels=a.channels, blocks=a.blocks, bilinear_dim=a.bilinear_dim,
        n_simulations=a.sims, n_determinizations=a.determinizations,
        temp_moves=a.temp_moves, device=a.device, seed=a.seed,
        leaf_batch=a.leaf_batch, batch_slots=a.batch_slots,
        eval_pad_to_batch=a.eval_pad_to_batch,
        pin_transfer=a.pin_transfer,
        profile_eval_timing=a.profile_eval_timing,
        allow_tf32=not a.no_tf32, inference_amp=a.amp_inference)
    configure_torch_performance(cfg)
    net = KingdominoNet(channels=cfg.channels, blocks=cfg.blocks,
                        bilinear_dim=cfg.bilinear_dim).to(cfg.device).eval()
    if a.compile_net:
        net = compile_net_for_inference(
            net, backend=a.compile_backend, mode=a.compile_mode)
    base = a.seed * 1_000_003

    if a.leaf_batch > 1 and "a1" in runs:
        print(f"NOTE: A1 (local) ignores leaf_batch; it batches across game "
              f"threads in-process. leaf_batch drives SERIAL and A3.")

    display_cap = a.batch_slots * max(1, a.leaf_batch) if runs == ["batched"] else max_batch
    print(f"device={cfg.device}  net={a.channels}ch/{a.blocks}b  sims={a.sims}  "
          f"det={a.determinizations}  games/backend={a.games}  max_batch={display_cap}")
    print(f"{'backend':<20}{'games/s':>9}   "
          f"{'mean_batch':>10}{'fill':>7}{'evals/s':>10}{'maxseen':>9}{'busy':>8}")
    print("-" * 86)

    if a.engine == "rust":
        # Rust engine: in-process RustMCTS per game thread, no inference server.
        # Concurrency mirrors A3's total (workers Ã— games_per_worker) but as
        # threads in one process.  Output row is directly comparable to A3.
        n_threads = max(1, a.workers * a.games_per_worker)
        total = a.warmup + a.repeat
        print(f"running RUST ({a.games} games, {n_threads} in-process game threads "
              f"= {a.workers}wÃ—{a.games_per_worker}g, no inference server)â€¦",
              flush=True)
        results = run_rust(cfg, net, a.games, base, n_threads,
                           max_batch, a.max_wait_ms,
                           reps=a.repeat, warmup=a.warmup, leaf_batch=a.leaf_batch)
        label = f"rust {n_threads}t"
        if total <= 1 and results:
            gps, st = results[0]
            print(f"{label:<20}{gps:>9.3f}   {_fmt_stats(st)}", flush=True)
        elif results:
            _summarize_reps(results, a.games_per_iter)
        print("-" * 78)
        return

    if "serial" in runs:
        sg = min(a.serial_games, a.games)
        lb_tag = f"leaf_batch={a.leaf_batch}" if a.leaf_batch > 1 else "batch-1"
        print(f"running serial ({sg} games, {a.sims} sims)â€¦ "
              f"[{lb_tag}{', slowest backend' if a.leaf_batch == 1 else ''}]",
              flush=True)
        gps, st = run_serial(cfg, net, sg, base, leaf_batch=a.leaf_batch)
        row = f"serial(lb={a.leaf_batch})" if a.leaf_batch > 1 else "serial"
        print(f"{row:<20}{gps:>9.3f}   {_fmt_stats(st)}", flush=True)

    if "open_loop" in runs:
        sg = min(a.serial_games, a.games)
        print(f"running open_loop ({sg} games, {a.sims} sims)…", flush=True)
        gps, st = run_open_loop(cfg, net, sg, base)
        print(f"{'open_loop':<20}{gps:>9.3f}   {_fmt_stats(st)}", flush=True)

    if "a1" in runs:
        print(f"running A1 ({a.games} games, {a.game_threads} threads)â€¦",
              flush=True)
        gps, st = run_a1(cfg, net, a.games, base, a.game_threads,
                         max_batch, a.max_wait_ms)
        print(f"{'A1 threads='+str(a.game_threads):<20}{gps:>9.3f}   {_fmt_stats(st)}",
              flush=True)

    if "a3" in runs:
        total = a.warmup + a.repeat
        if total <= 1:
            print(f"running A3 ({a.games} games, {a.workers}wÃ—{a.games_per_worker}g, "
                  f"spawning workersâ€¦)", flush=True)
        else:
            print(f"running A3 ({a.games} games, {a.workers}wÃ—{a.games_per_worker}g, "
                  f"{a.warmup} warmup + {a.repeat} timed reps)â€¦", flush=True)
        results = run_a3(cfg, net, a.games, base, a.workers,
                         a.games_per_worker, max_batch, a.max_wait_ms,
                         reps=a.repeat, warmup=a.warmup)
        label = f"A3 {a.workers}wÃ—{a.games_per_worker}g"
        if total <= 1 and results:
            gps, st = results[0]
            print(f"{label:<20}{gps:>9.3f}   {_fmt_stats(st)}", flush=True)
        elif results:
            _summarize_reps(results, a.games_per_iter)

    if "batched" in runs:
        total = a.warmup + a.repeat
        print(f"running BATCHED ({a.games} games, slots={a.batch_slots}, "
              f"leaf_batch={a.leaf_batch})...", flush=True)
        results = run_batched(cfg, net, a.games, base,
                              reps=a.repeat, warmup=a.warmup)
        label = f"batched {a.batch_slots}s"
        if total <= 1 and results:
            gps, st = results[0]
            print(f"{label:<20}{gps:>9.3f}   {_fmt_stats(st)}", flush=True)
        elif results:
            _summarize_reps(results, a.games_per_iter)

    if "batched_open_loop" in runs:
        total = a.warmup + a.repeat
        print(f"running BATCHED_OPEN_LOOP ({a.games} games, slots={a.batch_slots}, "
              f"leaf_batch={a.leaf_batch})...", flush=True)
        # play_selfplay_games_batched reads cfg.engine to set BatchedMCTS
        # open_loop=True; save/restore so other backends are unaffected.
        prev_engine = cfg.engine
        cfg.engine = "batched_open_loop"
        results = run_batched(cfg, net, a.games, base,
                              reps=a.repeat, warmup=a.warmup)
        cfg.engine = prev_engine
        label = f"batched_ol {a.batch_slots}s"
        if total <= 1 and results:
            gps, st = results[0]
            print(f"{label:<20}{gps:>9.3f}   {_fmt_stats(st)}", flush=True)
        elif results:
            _summarize_reps(results, a.games_per_iter)

    print("-" * 78)
    print("Read: A3 helps if games/s rises with mean_batch & evals/s climbing vs "
          "A1/serial.\nLow fill + low games/s => GPU starved (raise workers / "
          "games_per_worker).")


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()

