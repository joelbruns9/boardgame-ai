"""
DEPRECATED — Use self_play.py with engine=batched_open_loop instead.

This module implements a multiprocess IPC topology (A3 architecture):
one GPU inference server + N CPU workers. It was superseded by the
Rust BatchedMCTS engine which achieves superior throughput in a single
process by batching leaf evaluations across 32 concurrent game slots.

Why deprecated:
- batched_open_loop requires workers=1 (multiple workers do uncoordinated
  GPU forwards that thrash rather than coalesce)
- The GIL-based IPC overhead measured 1.74x on the old Python path;
  the Rust engine is 48x faster than the Python baseline
- Throughput on GPU-bound hardware: self_play.py + BatchedMCTS is faster

When this might become relevant again:
- If profiling on a very fast GPU (e.g. RTX 5090) confirms CPU tree work
  is the bottleneck and double-buffer alone is insufficient, a proper
  central-inference-server topology (workers send batched leaves to one
  GPU server) would be the right approach. This file is the reference
  implementation for that pattern.

Do not update this file without first validating against self_play.py
via the correctness_oracle.py equivalence tests.

────────────────────────────────────────────────────────────────────────
Original module docstring follows:

parallel_self_play.py — multiprocess throughput layer for AlphaZero Kingdomino
self-play (A3).

This REPLACES the earlier bespoke ``InferenceServer(Process)`` that lived here.
It builds entirely on the reviewed inference layer in ``inference_service.py``
(shared BatchEvaluator, request futures, model-version tracking, and a real
weight barrier) and on the A1 game-thread pool pattern in
``threaded_self_play.py``.  The graduation is exactly the one threaded_self_play
documents: take its game-thread pool and run it inside N worker PROCESSES, each
holding one RemoteInferenceWorkerClient.

TOPOLOGY
  Parent process (owns the GPU for training + the inference server):
    RemoteInferenceServer  ── 1 dedicated process holding KingdominoNet on the
                              GPU; coalesces leaf requests from every worker
                              thread into one forward pass, routes results back.
  N worker processes (torch-free in the sense that matters: no model, no GPU
  forward — they only run MCTS tree work + encode_state + legal_actions):
    each builds ONE RemoteInferenceWorkerClient → ONE shared, thread-safe
    InferenceClient → ``games_per_worker`` game threads, each with its own
    AlphaZeroMCTS.  While one thread blocks on inference, the others keep their
    core busy — which is the whole point (cores, not RAM, are the bottleneck
    once workers hold no network).

WHY NOT ONE GAME PER WORKER
  A single blocking game per worker leaves the core idle for the large fraction
  of wall time spent waiting on inference.  ``games_per_worker`` threads sharing
  one client cover each others' waits and let the server batch across the whole
  fleet (realized batch ≤ n_workers × games_per_worker).

CORRECTNESS
  A game's training examples are a deterministic function of its seed alone
  (_game_rngs(seed) — the SAME derivation the serial loop uses), regardless of
  which worker/thread runs it.  Results are sorted by seed before they enter the
  replay buffer, so buffer insertion order matches the serial loop and is
  independent of scheduling.  The only divergence from serial is batched-vs-
  batch-1 floating point at the leaf, identical in spirit to the A1 caveat.

  Each iteration begins with update_weights(...) + wait_for_version(...), so the
  first leaf evaluation of an iteration cannot be served under the previous
  iteration's weights (the race the threaded review flagged).  Training runs
  after self-play within an iteration, so weights never change mid-iteration:
  every game in an iteration is evaluated under one consistent model version.

  DOES NOT IMPORT evaluation.py.

START METHOD
  Workers are SPAWNED, not forked.  The parent initializes CUDA (for training),
  and fork-after-CUDA-init yields a poisoned CUDA state in children; spawn gives
  clean workers.  Spawn also keeps the server's queues and the worker pool on a
  single multiprocessing context.  run_parallel_self_play_training forces spawn
  before any process or queue is created.

USAGE
  python -m games.kingdomino.parallel_self_play \\
      --workers 8 --games_per_worker 4 \\
      --sims 800 --determinizations 1 \\
      --iterations 40 --games_per_iter 200 --train_steps 400 \\
      --channels 64 --blocks 6 --device cuda --checkpoint_dir checkpoints
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from typing import List, Optional, Tuple

import numpy as np
import torch

from games.kingdomino.network import KingdominoNet
from games.kingdomino.mcts_az import AlphaZeroMCTS
from games.kingdomino.self_play import (
    SelfPlayConfig, ReplayBuffer, Example,
    play_selfplay_game, play_selfplay_game_rust, make_rust_evaluator,
    play_selfplay_games_batched,
    train_step, benchmark_vs, benchmark_vs_rust,
    AZPlayer, OpenLoopAZPlayer, save_checkpoint, make_mcts, make_open_loop_mcts,
    _game_rngs,
    configure_torch_performance,
    # Per-iteration logging helpers (shared with the serial loop — imported,
    # not duplicated).
    _grad_norm, _policy_params, _diag_metrics, _log_row, _derive_log_path,
    _new_history, _compact_summary,
)
from games.kingdomino.inference_service import (
    RemoteInferenceServer, RemoteInferenceWorkerClient, make_ipc_batched_evaluator,
)
from games.kingdomino.bots import GreedyBot


# ─────────────────────────────────────────────────────────────────────────────
# Worker-process state + entry points (module-level so they are picklable under
# spawn).  Populated once per process by _init_worker; consumed by the per-
# iteration drain task.
# ─────────────────────────────────────────────────────────────────────────────
_W: dict = {}   # per-process globals; never shared across processes


def _state_dict_bytes(net) -> bytes:
    """Serialize a net's weights to bytes for shipping to workers (Rust engine)."""
    import io
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    return buf.getvalue()


def _load_state_bytes(net, state_bytes: bytes, device: str) -> None:
    import io
    sd = torch.load(io.BytesIO(state_bytes), map_location=device)
    net.load_state_dict(sd)
    net.eval()


def _init_worker(request_q, response_qs, worker_counter,
                 cfg: SelfPlayConfig, games_per_worker: int,
                 default_timeout_s: float, debug_checks: bool,
                 model_kwargs: dict, init_state_bytes: Optional[bytes]) -> None:
    """Pool initializer — runs ONCE per worker process.

    Atomically claims a worker_id (0..n_workers-1) via the shared counter.
    Python engine: builds the one RemoteInferenceWorkerClient bound to that id's
    response queue + the shared thread-safe InferenceClient.
    Rust engine: builds an IN-PROCESS net (no IPC server/client) loaded with the
    initial weights; per-iteration weights arrive with each task.
    """
    # One math thread per worker.  With N worker processes, each defaulting its
    # BLAS/OpenMP pool to ~all cores, you get N×cores threads contending for
    # cores — oversubscription that slows the numpy work (encode_state, the
    # root softmax).  Workers run tiny numpy ops and no torch compute, so one
    # thread each is correct.  (torch.set_num_threads is reliable here; the env
    # vars are best-effort since numpy's BLAS may already be imported, but the
    # ops are small enough that BLAS rarely parallelizes them anyway.)
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_var, "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    with worker_counter.get_lock():
        worker_id = worker_counter.value
        worker_counter.value += 1

    # Fix 2: the leaf-value blend (margin_gain/alpha) and terminal-value params
    # (score_scale) are now bound at construction from cfg — make_rust_evaluator,
    # make_*_mcts, and BatchedMCTS all take them explicitly.  The old per-worker
    # `mcts_az.MARGIN_GAIN = cfg.margin_gain` global override (needed because
    # spawned workers re-import mcts_az fresh) is therefore gone — the exact
    # multiprocessing fragility this fix removes.
    _W.clear()
    _W.update(worker_id=worker_id, cfg=cfg, gpw=max(1, int(games_per_worker)))

    if cfg.engine in ("rust", "batched", "batched_open_loop", "open_loop"):
        # In-process net (no remote/IPC server).  Per-iteration weights arrive
        # with each task and are loaded into this net before playing.
        #   - batched / open_loop: just hold the net; the per-thread search
        #     (BatchedMCTS / OpenLoopMCTS) calls it directly via make_*_mcts.
        #   - rust: additionally stand up a per-worker LocalInferenceService that
        #     COALESCES leaf evals across this worker's game threads (RustMCTS
        #     releases the GIL during tree work, so threads overlap).
        # Open-loop is serial (no leaf-batching), so it needs no coalescing
        # service — a plain in-process net + make_serial_evaluator is correct.
        from games.kingdomino.network import KingdominoNet
        net = KingdominoNet(**model_kwargs).to(cfg.device).eval()
        if init_state_bytes is not None:
            _load_state_bytes(net, init_state_bytes, cfg.device)
        if cfg.engine in ("batched", "batched_open_loop", "open_loop"):
            _W.update(net=net)
            return

        from games.kingdomino.inference_service import LocalInferenceService
        from games.kingdomino.self_play import make_rust_coalescing_evaluator
        lb = max(1, int(getattr(cfg, "leaf_batch", 1)))
        cap = max(1, games_per_worker * lb)   # all threads' leaves can co-batch
        svc = LocalInferenceService(net, device=cfg.device, max_batch=cap,
                                    max_wait_ms=3.0, debug_checks=False,
                                    margin_gain=cfg.margin_gain, alpha=cfg.alpha).start()
        v = svc.update_weights(net.state_dict())
        svc.wait_for_version(v, timeout_s=30.0)
        evaluator = make_rust_coalescing_evaluator(svc.make_client())
        _W.update(net=net, svc=svc, evaluator=evaluator)
        return

    if worker_id >= len(response_qs):
        # Would only happen if the pool recycled a process (we set
        # maxtasksperchild=None, so it shouldn't).  Fail loudly rather than
        # index past the response-queue list.
        raise RuntimeError(
            f"worker_id {worker_id} >= response_qs count {len(response_qs)}; "
            f"pool worker recycling is not supported.")

    wclient = RemoteInferenceWorkerClient(
        request_q, response_qs, worker_id,
        default_timeout_s=default_timeout_s, debug_checks=debug_checks)
    client = wclient.make_client()   # thread-safe; callable as the MCTS Evaluator
    _W.update(wclient=wclient, client=client)


def _worker_play_seed_list(payload):
    """Per-iteration task: play this worker's assigned seeds.

    `payload` is (seeds, state_bytes, iteration).  For the Rust engine,
    state_bytes carries this iteration's weights (loaded into the in-process net
    before playing); for the Python engine it is None (the inference server holds
    the weights).  `iteration` is stamped on every Example for buffer-age
    tracking (ReplayBuffer.mean_age).

    The seed list is handed to the task directly (no shared cross-process queue,
    no sentinels), so termination is trivially correct.  Inside the process,
    `games_per_worker` threads drain a LOCAL queue.Queue and cover each others'
    inference waits.  Each thread owns its own search tree; Python threads share
    the one inference client, Rust threads share the one in-process net (torch
    releases the GIL during the forward pass, so concurrent leaf evals overlap).
    Returns (results, errors).
    """
    seeds, state_bytes, iteration = payload
    cfg: SelfPlayConfig = _W["cfg"]
    gpw: int = _W["gpw"]
    lb = max(1, int(getattr(cfg, "leaf_batch", 1)))

    if cfg.engine in ("batched", "batched_open_loop"):
        net = _W["net"]
        if state_bytes is not None:
            _load_state_bytes(net, state_bytes, cfg.device)
        if seeds:
            all_examples, all_scores, _stats = play_selfplay_games_batched(
                net, cfg, n_games=len(seeds), game_seed_start=min(seeds),
                iteration=iteration)
            return (
                [(min(seeds) + i, exs, score)
                 for i, (exs, score) in enumerate(zip(all_examples, all_scores))],
                [],
            )
        return [], []

    if cfg.engine == "rust":
        import kingdomino_rust  # noqa: F401  (per-thread RustMCTS built in body)
        net = _W["net"]
        svc = _W["svc"]
        rust_eval = _W["evaluator"]   # coalescing client → batches across threads
        if state_bytes is not None:
            _load_state_bytes(net, state_bytes, cfg.device)
            v = svc.update_weights(net.state_dict())
            svc.wait_for_version(v, timeout_s=30.0)
    elif cfg.engine == "open_loop":
        net = _W["net"]   # in-process; each thread builds its own OpenLoopMCTS
        if state_bytes is not None:
            _load_state_bytes(net, state_bytes, cfg.device)
    else:
        client = _W["client"]

    local_q: "queue.Queue[int]" = queue.Queue()
    for s in seeds:
        local_q.put(s)

    results: List[Tuple[int, List[Example], Tuple[int, int]]] = []
    errors: List[Tuple[int, str, str]] = []
    lock = threading.Lock()

    def body() -> None:
        if cfg.engine == "rust":
            import kingdomino_rust
            rust_mcts = kingdomino_rust.RustMCTS()   # per-thread tree
        elif cfg.engine == "open_loop":
            # Per-thread OpenLoopMCTS over the worker's in-process net.  Serial
            # (no leaf-batching); torch releases the GIL during the forward, so
            # threads still overlap their inference.
            ol_mcts = make_open_loop_mcts(net, cfg, cfg.n_simulations)
        else:
            # leaf_batch>1 sends N leaves per inference round trip (batched-send),
            # amortizing the per-request IPC cost.
            mcts = AlphaZeroMCTS(
                client,
                batched_evaluator=(make_ipc_batched_evaluator(client) if lb > 1 else None),
                c_puct=cfg.c_puct,
                n_simulations=cfg.n_simulations,
                dirichlet_alpha=cfg.dirichlet_alpha,
                dirichlet_epsilon=cfg.dirichlet_epsilon,
                score_scale=cfg.score_scale,
                margin_gain=cfg.margin_gain,
                alpha=cfg.alpha,
            )
        while True:
            try:
                seed = local_q.get_nowait()
            except queue.Empty:
                return
            try:
                py_rng, np_rng = _game_rngs(seed)
                if cfg.engine == "rust":
                    examples, scores = play_selfplay_game_rust(
                        rust_mcts, rust_eval,
                        n_simulations=cfg.n_simulations,
                        n_determinizations=cfg.n_determinizations,
                        temp_moves=cfg.temp_moves, c_puct=cfg.c_puct,
                        dirichlet_alpha=cfg.dirichlet_alpha,
                        dirichlet_epsilon=cfg.dirichlet_epsilon,
                        leaf_batch=lb, virtual_loss=1,
                        seed=seed, py_rng=py_rng, np_rng=np_rng,
                        score_scale=cfg.score_scale,
                        margin_gain=cfg.margin_gain, alpha=cfg.alpha,
                        iteration=iteration,
                    )
                elif cfg.engine == "open_loop":
                    examples, scores = play_selfplay_game(
                        ol_mcts,
                        n_determinizations=1,   # ignored internally; required by signature
                        temp_moves=cfg.temp_moves,
                        seed=seed, py_rng=py_rng, np_rng=np_rng,
                        leaf_batch=1,            # not used by open-loop
                        open_loop=True,
                        iteration=iteration,
                    )
                else:
                    examples, scores = play_selfplay_game(
                        mcts,
                        n_determinizations=cfg.n_determinizations,
                        temp_moves=cfg.temp_moves,
                        seed=seed, py_rng=py_rng, np_rng=np_rng,
                        leaf_batch=lb,
                        iteration=iteration,
                    )
                with lock:
                    results.append((seed, examples, scores))
            except Exception as exc:
                with lock:
                    errors.append((seed, repr(exc), traceback.format_exc()))

    threads = [
        threading.Thread(target=body, name=f"game-w{_W['worker_id']}-t{i}",
                         daemon=True)
        for i in range(gpw)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


def _worker_cleanup(_ignored: int) -> None:
    """Best-effort: stop this process's worker client / inference service."""
    wclient = _W.get("wclient")
    if wclient is not None:
        try:
            wclient.stop()
        except Exception:
            pass
    svc = _W.get("svc")
    if svc is not None:
        try:
            svc.stop()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Parent-side parallel game generation
# ─────────────────────────────────────────────────────────────────────────────
def _generate_parallel(
    pool, cfg: SelfPlayConfig, n_games: int, game_seed_start: int,
    n_workers: int, fail_fast: bool, verbose: bool,
    state_bytes: Optional[bytes] = None,
    iteration: int = 0,
) -> Tuple[List[List[Example]], List[Tuple[int, int]], List[Tuple[int, str, str]]]:
    """Split the iteration's seeds into one list per worker, run them, aggregate.

    Round-robin assignment keeps the per-worker chunk sizes balanced to within
    one game.  Each worker's task is self-contained (its own seed list), so
    termination is trivially correct no matter how the pool schedules tasks
    across processes — no shared queue, no sentinels.  Game data depends only on
    seed, and results are sorted by seed before returning, so buffer insertion
    order matches the serial loop regardless of scheduling.
    """
    seeds = list(range(game_seed_start, game_seed_start + n_games))
    seed_lists = [seeds[i::n_workers] for i in range(n_workers)]   # round-robin

    # Rust engine: ship this iteration's weights with each task (no IPC server).
    # Python engine: state_bytes is None (the server already holds the weights).
    # iteration is stamped on every Example for buffer-age tracking.
    payloads = [(sl, state_bytes, iteration) for sl in seed_lists]
    out = pool.map(_worker_play_seed_list, payloads, chunksize=1)

    results: List[Tuple[int, List[Example], Tuple[int, int]]] = []
    errors: List[Tuple[int, str, str]] = []
    for res, errs in out:
        results.extend(res)
        errors.extend(errs)

    if errors and fail_fast:
        msg = "\n".join(f"  seed {s}: {e}" for s, e, _tb in errors[:10])
        raise RuntimeError(
            f"{len(errors)}/{n_games} self-play game(s) failed:\n{msg}\n"
            f"(first traceback)\n{errors[0][2]}")
    if errors and verbose:
        print(f"    [warning] {len(errors)} game(s) failed; "
              f"{len(results)} succeeded (fail_fast=False)")
        for s, e, _tb in errors[:3]:
            print(f"      - seed {s}: {e}")
    if not results:
        raise RuntimeError(
            f"All {n_games} self-play games failed; nothing to train on.")

    results.sort(key=lambda r: r[0])          # deterministic buffer order (by seed)
    all_examples = [r[1] for r in results]
    all_scores = [r[2] for r in results]
    return all_examples, all_scores, errors


# ─────────────────────────────────────────────────────────────────────────────
# Main A3 training loop
# ─────────────────────────────────────────────────────────────────────────────
def run_parallel_self_play_training(
    cfg: SelfPlayConfig,
    n_workers: int = 8,
    games_per_worker: int = 4,
    max_batch: Optional[int] = None,
    max_wait_ms: float = 3.0,
    default_timeout_s: float = 60.0,
    debug_checks: bool = True,
    fail_fast: bool = True,
    eval_vs_checkpoint: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Multiprocess (A3) AlphaZero training loop.  Drop-in for the serial /
    threaded loops: same SelfPlayConfig, same checkpoint format, same train /
    benchmark / checkpoint stages.  Only self-play generation differs.

    `max_batch` defaults to n_workers × games_per_worker — the maximum number of
    leaf evaluations that can be in flight at once (each game thread holds at
    most one).  A larger cap wastes nothing but does nothing.
    """
    configure_torch_performance(cfg)
    # Fix 2: leaf-value/terminal-value params are bound at construction from cfg
    # (make_*_mcts / make_rust_evaluator / BatchedMCTS forward them), so the parent
    # no longer mutates the mcts_az module globals — and neither do the workers.
    # Spawn workers (see module docstring: CUDA + fork in the parent poisons
    # children; spawn keeps the server queues and worker pool on one context).
    # Safe to force here because this function owns the process and creates all
    # multiprocessing resources below.
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    if max_batch is None:
        # batched-send: each in-flight request can carry cfg.leaf_batch leaves, so
        # the server forward can reach workers × games_per_worker × leaf_batch.
        lb = max(1, int(getattr(cfg, "leaf_batch", 1)))
        max_batch = n_workers * games_per_worker * lb

    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Training network — gradients live here, in the parent process.
    net = KingdominoNet(channels=cfg.channels, blocks=cfg.blocks,
                        bilinear_dim=cfg.bilinear_dim).to(cfg.device)
    if cfg.warm_start_path:
        ckpt = torch.load(cfg.warm_start_path, map_location=cfg.device)
        sd = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(sd)
        if verbose:
            print(f"Warm-started from {cfg.warm_start_path}")

    # ── Inference (Python: a server process holding the net; Rust: none — each
    # worker holds an in-process net and calls it directly for leaf eval). ──
    model_kwargs = dict(channels=cfg.channels, blocks=cfg.blocks,
                        bilinear_dim=cfg.bilinear_dim)
    use_rust = (cfg.engine == "rust")
    use_batched = (cfg.engine in ("batched", "batched_open_loop"))
    use_open_loop = (cfg.engine == "open_loop")
    if cfg.engine == "batched_open_loop" and n_workers > 1:
        raise ValueError(
            "--engine batched_open_loop requires --workers 1; parallelism "
            "comes from --batch_slots, not workers.")
    if use_batched and n_workers != 1:
        raise ValueError("--engine batched/batched_open_loop in parallel_self_play currently requires --workers 1")
    if use_rust or use_batched or use_open_loop:
        # In-process worker nets, no inference server (weights shipped per
        # iteration via init_state_bytes / per-task state_bytes).
        server = None
        request_q, response_qs = None, []
        init_state_bytes = _state_dict_bytes(net)
        if verbose and use_rust:
            print("Rust engine: in-process worker nets, no inference server "
                  "(weights shipped per iteration).")
        if verbose and use_batched:
            print("Batched engine: one worker process, one synchronized "
                  f"BatchedMCTS with batch_slots={cfg.batch_slots}.")
        if verbose and use_open_loop:
            print("Open-loop engine: in-process worker nets, no inference server; "
                  "each game thread runs its own OpenLoopMCTS (deck resampled "
                  "per simulation, n_determinizations/leaf_batch ignored).")
    else:
        server = RemoteInferenceServer(
            n_workers=n_workers, model_kwargs=model_kwargs, device=cfg.device,
            max_batch=max_batch, max_wait_ms=max_wait_ms, debug_checks=debug_checks,
            margin_gain=cfg.margin_gain, alpha=cfg.alpha)
        server.start(initial_state_dict=net.state_dict(), wait_until_loaded=True)
        request_q, response_qs = server.worker_handles()
        init_state_bytes = None

    # ── Worker pool (persistent across iterations: avoids re-spawning + re-
    # importing torch every iteration). ──
    worker_counter = mp.Value("i", 0)
    pool = mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(request_q, response_qs, worker_counter, cfg,
                  games_per_worker, default_timeout_s, debug_checks,
                  model_kwargs, init_state_bytes),
        maxtasksperchild=None,   # never recycle: worker_id is claimed once
    )

    buffer = ReplayBuffer(cfg.buffer_capacity, n_sample_workers=cfg.sample_workers)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    np_rng = np.random.default_rng(cfg.seed)
    history = _new_history()
    log_path = _derive_log_path(cfg)
    if verbose:
        print(f"Per-iteration log: {log_path}")
    game_seed = cfg.seed * 1_000_003

    # Fixed diagnostic probe batch (shared by policy_entropy + win_brier_diag),
    # sampled once on the first training iteration; its own RNG keeps np_rng
    # (training/benchmark stream) unperturbed.  Mirrors the serial loop.
    diag_entropy_batch = None
    diag_rng = np.random.default_rng(cfg.seed + 7919)

    # Optional head-to-head benchmark opponent: a FROZEN net from a checkpoint
    # (e.g. the warm-start baseline). Far more informative than GreedyBot once the
    # net already sits near the Greedy ceiling — it measures strength delta vs a
    # fixed reference. None => fall back to GreedyBot (original behavior).
    bench_opponent_net = None
    if eval_vs_checkpoint:
        bench_opponent_net = KingdominoNet(
            channels=cfg.channels, blocks=cfg.blocks,
            bilinear_dim=cfg.bilinear_dim).to(cfg.device).eval()
        _oc = torch.load(eval_vs_checkpoint, map_location=cfg.device)
        _osd = _oc.get("model_state", _oc) if isinstance(_oc, dict) else _oc
        bench_opponent_net.load_state_dict(_osd)
        for _p in bench_opponent_net.parameters():
            _p.requires_grad_(False)
        if verbose:
            print(f"Benchmark opponent: frozen checkpoint {eval_vs_checkpoint} "
                  f"(head-to-head at benchmark_sims={cfg.benchmark_sims}).")

    if verbose:
        print(f"A3 self-play: workers={n_workers} × games_per_worker="
              f"{games_per_worker} = {n_workers * games_per_worker} concurrent "
              f"games, max_batch={max_batch}, max_wait_ms={max_wait_ms}, "
              f"sims={cfg.n_simulations}, det={cfg.n_determinizations}, "
              f"device={cfg.device}")

    try:
        for it in range(1, cfg.n_iterations + 1):
            if verbose:
                print(f"\n{'='*60}\nIteration {it}/{cfg.n_iterations}\n{'='*60}")

            # Per-iteration metrics for the structured log; None = not computed
            # this iteration.  Filled below (mirrors the serial loop).
            trained = False
            pol_m = own_m = opp_m = win_m = None
            win_brier_m = baseline_brier_m = None
            gn_pol = gn_win = gn_own = gn_opp = None
            entropy = win_brier_diag = None
            bench_win_rate = bench_score_margin = bench_win_brier = None

            # ── 1. Self-play (parallel) ──
            net.eval()
            if use_rust or use_batched or use_open_loop:
                # Ship this iteration's weights with the tasks (no server).
                state_bytes = _state_dict_bytes(net)
            else:
                # Push current weights and BARRIER until they are actually
                # serving, so this iteration's first leaf eval is not stale.
                target_version = server.update_weights(net.state_dict())
                if not server.wait_for_version(target_version, timeout_s=30.0):
                    raise RuntimeError(
                        f"Inference server did not apply weight version "
                        f"{target_version} within 30s (iteration {it}).")
                state_bytes = None

            t0 = time.time()
            all_examples, all_scores, _errs = _generate_parallel(
                pool, cfg, n_games=cfg.games_per_iteration,
                game_seed_start=game_seed, n_workers=n_workers,
                fail_fast=fail_fast,
                verbose=verbose and cfg.games_per_iteration >= 20,
                state_bytes=state_bytes, iteration=it,
            )
            game_seed += cfg.games_per_iteration
            for exs in all_examples:
                buffer.add(exs)

            elapsed = time.time() - t0
            gps = len(all_examples) / elapsed if elapsed > 0 else 0.0
            diffs_arr = np.array([s[0] - s[1] for s in all_scores], dtype=np.float32)
            sp_score_diff_mean = float(diffs_arr.mean()) if len(diffs_arr) else 0.0
            sp_score_diff_std = float(diffs_arr.std()) if len(diffs_arr) else 0.0
            buffer_size = len(buffer)
            buffer_mean_age = buffer.mean_age(it)
            history["sp_score_diff_mean"].append(sp_score_diff_mean)
            history["sp_score_diff_std"].append(sp_score_diff_std)
            history["games_per_sec"].append(gps)
            history["buffer_size"].append(buffer_size)
            history["buffer_mean_age"].append(buffer_mean_age)
            if verbose:
                print(f"  self-play: {len(all_examples)}/"
                      f"{cfg.games_per_iteration} games ({gps:.2f} games/sec), "
                      f"buffer={buffer_size}")
                s = server.get_stats() if server is not None else None
                if s:
                    print(f"  inference: mean_batch={s['mean_batch']:.1f}/"
                          f"{s['max_batch_cap']} (fill {s['fill_ratio']:.0%}), "
                          f"service_busy={s['service_busy_fraction']:.0%}, "
                          f"{s['requests_per_sec']:.0f} evals/sec, "
                          f"max_batch_seen={s['max_batch_seen']}")
                elif server is not None:
                    print("  inference: (no stats published yet this iteration)")

            # ── 2. Train ──
            if len(buffer) < cfg.min_buffer_to_train:
                if verbose:
                    print(f"  buffer below warmup ({len(buffer)}/"
                          f"{cfg.min_buffer_to_train}); skipping training")
            elif cfg.train_steps_per_iteration <= 0:
                if verbose:
                    print("  train: train_steps_per_iteration=0; skipping")
            else:
                trained = True
                net.train()
                p_sum = o_sum = q_sum = w_sum = 0.0
                brier_sum = baseline_sum = 0.0
                gnp_sum = gnw_sum = gno_sum = gnq_sum = 0.0
                for _ in range(cfg.train_steps_per_iteration):
                    batch = buffer.sample_batch(cfg.batch_size, np_rng,
                                                device=cfg.device,
                                                augment_d4=cfg.augment)
                    (policy_loss, own_loss, opp_loss, win_loss,
                     win_brier, baseline_brier) = train_step(
                        net, batch, optimizer,
                        policy_weight=cfg.policy_weight,
                        lambda_score=cfg.lambda_score,
                        lambda_w=cfg.lambda_w,
                        score_scale=cfg.score_scale,
                        grad_clip=cfg.grad_clip,
                    )
                    # Grad norms: grads still populated after step (train_step
                    # zeros at the START of each step), so read them here.
                    gnp_sum += _grad_norm(_policy_params(net))
                    gnw_sum += _grad_norm(net.win_mlp.parameters())
                    gno_sum += _grad_norm(net.own_score_mlp.parameters())
                    gnq_sum += _grad_norm(net.opponent_score_mlp.parameters())
                    p_sum += policy_loss; o_sum += own_loss
                    q_sum += opp_loss; w_sum += win_loss
                    brier_sum += win_brier; baseline_sum += baseline_brier
                n = cfg.train_steps_per_iteration
                pol_m, own_m, opp_m, win_m = p_sum/n, o_sum/n, q_sum/n, w_sum/n
                win_brier_m, baseline_brier_m = brier_sum/n, baseline_sum/n
                gn_pol, gn_win = gnp_sum/n, gnw_sum/n
                gn_own, gn_opp = gno_sum/n, gnq_sum/n
                history["policy_loss"].append(pol_m)
                history["own_loss"].append(own_m)
                history["opp_loss"].append(opp_m)
                history["win_loss"].append(win_m)
                history["win_brier"].append(win_brier_m)
                history["baseline_brier"].append(baseline_brier_m)
                history["grad_norm_policy"].append(gn_pol)
                history["grad_norm_win"].append(gn_win)
                history["grad_norm_own"].append(gn_own)
                history["grad_norm_opp"].append(gn_opp)
                if verbose:
                    print(f"  train: policy={pol_m:.4f}  own={own_m:.4f}  "
                          f"opp={opp_m:.4f}  win={win_m:.4f}  "
                          f"brier={win_brier_m:.4f}  base={baseline_brier_m:.4f}")

                # ── Diagnostic batch (policy_entropy + win_brier_diag) ──
                if diag_entropy_batch is None:
                    diag_entropy_batch = buffer.sample_batch(
                        min(256, len(buffer)), diag_rng,
                        device=cfg.device, augment_d4=False)
                entropy, win_brier_diag = _diag_metrics(net, diag_entropy_batch)
                history["policy_entropy"].append(entropy)
                history["win_brier_diag"].append(win_brier_diag)
                net.eval()
                if verbose:
                    print(f"  diag: entropy={entropy:.4f}  "
                          f"brier_diag={win_brier_diag:.4f}")

            # ── 3. Benchmark + checkpoint ──
            # Benchmark uses the freshly-trained net directly (serial batch-1
            # evaluator), not the server — so it reflects the current weights
            # regardless of server timing.  The server is idle during this.
            if cfg.benchmark_every and it % cfg.benchmark_every == 0:
                net.eval()
                bench_dets = (cfg.benchmark_determinizations
                              if cfg.benchmark_determinizations is not None
                              else cfg.n_determinizations)
                if cfg.engine == "batched_open_loop":
                    # Rust-backed lockstep benchmark (~50x faster than the Python
                    # OpenLoopMCTS player; see benchmark_vs_rust docstring).  The
                    # frozen checkpoint opponent, if any, also plays via RustMCTS.
                    opp_label = "checkpoint" if bench_opponent_net is not None else "Greedy"
                    stats = benchmark_vs_rust(
                        net, cfg, cfg.benchmark_seeds, seed=cfg.seed + 99,
                        opponent_net=bench_opponent_net, opp_rng_seed=cfg.seed + 12345,
                        verbose=False)
                else:
                    if cfg.engine == "open_loop":
                        az = OpenLoopAZPlayer(
                            make_open_loop_mcts(net, cfg, cfg.benchmark_sims),
                            np_rng=np_rng)
                    else:
                        az = AZPlayer(make_mcts(net, cfg, cfg.benchmark_sims),
                                      n_determinizations=bench_dets, np_rng=np_rng)
                    if bench_opponent_net is not None:
                        # Fresh fixed RNG each benchmark => the frozen opponent plays
                        # reproducibly, so the win-rate trajectory is a clean measure
                        # of the current net improving against a fixed reference.
                        opp_rng = np.random.default_rng(cfg.seed + 12345)
                        if cfg.engine == "open_loop":
                            opponent = OpenLoopAZPlayer(
                                make_open_loop_mcts(bench_opponent_net, cfg, cfg.benchmark_sims),
                                np_rng=opp_rng)
                        else:
                            opponent = AZPlayer(
                                make_mcts(bench_opponent_net, cfg, cfg.benchmark_sims),
                                n_determinizations=bench_dets, np_rng=opp_rng)
                        opp_label = "checkpoint"
                    else:
                        opponent = GreedyBot()
                        opp_label = "Greedy"
                    stats = benchmark_vs(az, opponent, cfg.benchmark_seeds,
                                         seed=cfg.seed + 99, verbose=False)
                bench_win_rate = stats["az_win_rate"]
                bench_score_margin = stats["mean_margin"]
                # Brier on the fixed diagnostic batch at benchmark time.
                if diag_entropy_batch is not None:
                    _, bench_win_brier = _diag_metrics(net, diag_entropy_batch)
                    net.eval()
                history["benchmark"].append((it, bench_win_rate))
                history["score_margin"].append(bench_score_margin)
                if verbose:
                    print(f"  benchmark vs {opp_label}: {stats['az_win_rate']:.1%} "
                          f"({stats['az_wins']}-{stats['draws']}-"
                          f"{stats['opp_wins']} over {stats['n_games']}), "
                          f"mean_margin={stats['mean_margin']:+.1f}")

            if cfg.checkpoint_dir:
                os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                save_checkpoint(
                    os.path.join(cfg.checkpoint_dir, f"iter_{it:04d}.pt"),
                    net, cfg, it, history)

            # ── 4. Structured log row + compact summary (END of iteration) ──
            row = {
                "iter": it,
                "timestamp": time.time(),
                "policy_loss": pol_m,
                "own_loss": own_m,
                "opp_loss": opp_m,
                "win_loss": win_m,
                "win_brier": win_brier_m,
                "baseline_brier": baseline_brier_m,
                "grad_norm_policy": gn_pol,
                "grad_norm_win": gn_win,
                "grad_norm_own": gn_own,
                "grad_norm_opp": gn_opp,
                "sp_score_diff_mean": sp_score_diff_mean,
                "sp_score_diff_std": sp_score_diff_std,
                "games_per_sec": gps,
                "buffer_size": buffer_size,
                "buffer_mean_age": buffer_mean_age,
                "policy_entropy": entropy,
                "win_brier_diag": win_brier_diag,
                "bench_win_rate": bench_win_rate,
                "bench_score_margin": bench_score_margin,
                "bench_win_brier": bench_win_brier,
            }
            _log_row(log_path, row)
            if verbose:
                print(_compact_summary(
                    it, sp_games=len(all_examples), row=row,
                    trained=trained, buf_n=buffer_size,
                    min_buf=cfg.min_buffer_to_train))
    finally:
        # Best-effort clean shutdown of worker clients, then the pool, then the
        # server.  terminate() is fine even if a task is mid-flight.
        buffer.close()   # stop the sample_batch thread pool
        try:
            pool.map(_worker_cleanup, range(n_workers), chunksize=1)
        except Exception:
            pass
        pool.terminate()
        pool.join()
        if server is not None:
            server.stop()
        if verbose:
            print("Worker pool stopped." if server is None
                  else "Inference server + worker pool stopped.")

    return {"net": net, "history": history, "buffer": buffer}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parallel (A3) multiprocess AlphaZero self-play for Kingdomino")
    # Parallelism / inference
    p.add_argument("--workers", type=int, default=8,
                   help="Number of self-play worker processes (~ CPU cores).")
    p.add_argument("--games_per_worker", type=int, default=4,
                   help="Concurrent game threads per worker (cover inference waits).")
    p.add_argument("--leaf_batch", type=int, default=1,
                   help="Leaf-parallel batch per search (batched-send): N leaves "
                        "per inference round trip. Validate divergence with "
                        "policy_compare before using >1 (4-6 is the safe zone).")
    p.add_argument("--max_batch", type=int, default=None,
                   help="Server batch cap (default: workers x games_per_worker x leaf_batch).")
    p.add_argument("--max_wait_ms", type=float, default=3.0,
                   help="Max time the server waits to fill a batch.")
    p.add_argument("--timeout_s", type=float, default=60.0,
                   help="Per-request inference timeout in the worker client.")
    p.add_argument("--no_debug_checks", action="store_true",
                   help="Disable per-request shape/finiteness asserts (slightly faster).")
    # Network
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--bilinear_dim", type=int, default=64)
    # Search
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--determinizations", type=int, default=1)
    p.add_argument("--c_puct", type=float, default=None)
    p.add_argument("--dirichlet_alpha", type=float, default=None)
    p.add_argument("--dirichlet_epsilon", type=float, default=None)
    p.add_argument("--temp_moves", type=int, default=None)
    p.add_argument("--fpu", type=float, default=None,
                   help="first-play-urgency value for unvisited children")
    p.add_argument("--virtual_loss", type=int, default=None,
                   help="virtual loss magnitude (leaf-parallel / batched paths)")
    # Leaf-value blend (overrides mcts_az.MARGIN_GAIN / mcts_az.ALPHA module
    # constants in BOTH the parent and every spawned worker process).
    p.add_argument("--margin_gain", type=float, default=None,
                   help="scales (own_norm-opp_norm) before tanh in leaf value")
    p.add_argument("--alpha", type=float, default=None,
                   help="weight on margin vs win term (0.8 = margin-dominant)")
    # Loop
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--games_per_iter", "--games", dest="games_per_iter",
                   type=int, default=200,
                   help="self-play games per iteration (--games is an alias)")
    p.add_argument("--train_steps", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--sample_workers", type=int, default=1,
                   help="threads for ReplayBuffer.sample_batch densify+augment "
                        "(1 = serial; >1 measured to REGRESS this GIL-bound "
                        "workload ~2x — kept for experimentation)")
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
    p.add_argument("--no_tf32", action="store_true",
                   help="disable TF32 CUDA matmul/convolution")
    p.add_argument("--amp_inference", action="store_true",
                   help="use CUDA float16 autocast for batched inference")
    p.add_argument("--engine",
                   choices=["python", "open_loop", "rust", "batched", "batched_open_loop"],
                   default="python",
                   help="python = AlphaZeroMCTS via the IPC inference server; "
                        "open_loop = OpenLoopMCTS with in-process per-worker nets "
                        "(no server; deck resampled per simulation, "
                        "n_determinizations/leaf_batch ignored); "
                        "rust = RustMCTS with in-process per-worker nets (no "
                        "server); batched = one worker with synchronized "
                        "BatchedMCTS; batched_open_loop = batched with "
                        "per-simulation deck resampling (the fast open-loop path).")
    p.add_argument("--batch_slots", type=int, default=32,
                   help="concurrent slots for --engine batched; use --workers 1.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warm_start", default=None)
    p.add_argument("--eval_vs_checkpoint", default=None,
                   help="Benchmark head-to-head vs this frozen checkpoint instead "
                        "of GreedyBot (e.g. your warm-start baseline). Far more "
                        "informative once the net is near the Greedy ceiling.")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_path", default=None,
                   help="per-iteration JSONL log path (default: auto-derive "
                        "{checkpoint_dir}/training_log.jsonl)")
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--allow_failed_games", action="store_true",
                   help="Tolerate per-game failures instead of fail-fast.")
    return p


def main() -> None:
    a = _build_argparser().parse_args()

    min_buf = a.min_buffer if a.min_buffer is not None else a.games_per_iter * 52

    # Only override SelfPlayConfig defaults for flags the user actually set.
    optional = {}
    for name in ("c_puct", "dirichlet_alpha", "dirichlet_epsilon", "temp_moves",
                 "fpu", "virtual_loss", "margin_gain", "alpha",
                 "weight_decay", "grad_clip", "value_weight", "policy_weight",
                 "benchmark_determinizations"):
        val = getattr(a, name)
        if val is not None:
            optional[name] = val

    cfg = SelfPlayConfig(
        channels=a.channels, blocks=a.blocks, bilinear_dim=a.bilinear_dim,
        n_simulations=a.sims, n_determinizations=a.determinizations,
        batch_size=a.batch_size, sample_workers=a.sample_workers,
        lr=a.lr, buffer_capacity=a.buffer,
        n_iterations=a.iterations, games_per_iteration=a.games_per_iter,
        train_steps_per_iteration=a.train_steps, min_buffer_to_train=min_buf,
        benchmark_seeds=a.benchmark_seeds, benchmark_sims=a.benchmark_sims,
        benchmark_every=a.benchmark_every, augment=not a.no_augment,
        device=a.device, seed=a.seed, warm_start_path=a.warm_start,
        checkpoint_dir=a.checkpoint_dir, log_path=a.log_path,
        leaf_batch=a.leaf_batch,
        batch_slots=a.batch_slots,
        allow_tf32=not a.no_tf32, inference_amp=a.amp_inference,
        engine=a.engine, **optional,
    )
    run_parallel_self_play_training(
        cfg, n_workers=a.workers, games_per_worker=a.games_per_worker,
        max_batch=a.max_batch, max_wait_ms=a.max_wait_ms,
        default_timeout_s=a.timeout_s, debug_checks=not a.no_debug_checks,
        fail_fast=not a.allow_failed_games,
        eval_vs_checkpoint=a.eval_vs_checkpoint, verbose=True)


if __name__ == "__main__":
    mp.freeze_support()                       # required on Windows under spawn
    mp.set_start_method("spawn", force=True)  # before any server/queue/pool
    main()
