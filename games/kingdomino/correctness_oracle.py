"""
correctness_oracle.py — verifies the A3 parallel self-play loop reproduces the
serial loop's training data.

(Updated for the inference_service.py-based A3 path.  The bespoke InferenceServer
that used to live in parallel_self_play.py has been retired; the parallel path
now uses inference_service.RemoteInferenceServer + RemoteInferenceWorkerClient
and the worker pool in parallel_self_play.py.)

WHY THIS EXISTS
The entire justification for the throughput layer is "same data, generated
faster".  If a subtle difference — a mis-threaded RNG, a different example-
construction path, a labeling bug — crept into the parallel code, it would
silently change the training distribution and you would only notice after a
multi-hour run produced a worse agent.  This oracle makes that failure mode loud
and cheap to catch.

WHAT EQUIVALENCE MEANS HERE
A self-play game's output is a deterministic function of (game_seed, evaluator
outputs, the shared game/encoder/MCTS code).  Both loops derive each game's RNGs
from game_seed alone (_game_rngs) and advance through the SAME seeds.  The engine,
encoder, action codec, MCTS, and example construction are literally the same code
(parallel_self_play imports play_selfplay_game from self_play).  So the ONLY thing
that can differ is the evaluator:

  - Serial uses a batch-1 network forward.
  - Parallel batches leaf evaluations across the fleet on the inference server.

Batched vs unbatched matmul/conv are NOT bit-identical (FP non-associativity,
~1e-6) — a property of PyTorch, not of our code.  Those tiny differences feed
PUCT argmax and can diverge the tree.  So:

  STRICT equivalence (bit-identical examples) holds iff the server's effective
  batch size equals serial's (i.e. 1), which is the case with a SINGLE worker,
  one game thread, and max_batch=1.

  With more concurrency, data is statistically equivalent but not bit-identical;
  that is inherent to batched inference, not a bug.

TWO TESTS
  1. PIPELINE ORACLE (rigorous, batch-independent): replace the network with a
     deterministic, batch-independent evaluator served by a mock server through
     the REAL worker protocol.  Batching is then irrelevant, so serial and
     parallel MUST produce bit-identical examples for ANY worker count.  This
     isolates and verifies OUR code (RNG threading, encoding, example
     construction, labeling, seed→buffer ordering) across the process boundary.

  2. REAL-NET ORACLE (production path): run the actual RemoteInferenceServer with
     the real network, ONE worker, games_per_worker=1, max_batch=1, on CPU.
     Confirms the real serial and real parallel paths agree bit-for-bit when
     batch sizes match.

Run:  python -m games.kingdomino.correctness_oracle
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from multiprocessing import Process, Queue as MPQueue, Event as MPEvent
from typing import Dict, List, Tuple

import numpy as np
import torch

from games.kingdomino.action_codec import NUM_JOINT_ACTIONS
from games.kingdomino.network import KingdominoNet
from games.kingdomino.mcts_az import AlphaZeroMCTS, make_serial_evaluator
from games.kingdomino.self_play import (
    SelfPlayConfig, Example, play_selfplay_game, _game_rngs,
)
from games.kingdomino.inference_service import (
    RemoteInferenceServer, _RemoteRequest, _RemoteResult,
)
from games.kingdomino.parallel_self_play import (
    _init_worker, _generate_parallel, _worker_cleanup,
)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic, batch-independent evaluator
# ─────────────────────────────────────────────────────────────────────────────
def deterministic_eval(
    mb: np.ndarray, ob: np.ndarray, flat: np.ndarray, idxs: np.ndarray
) -> Tuple[float, np.ndarray]:
    """A pure, batch-independent stand-in for the network.

    Output depends ONLY on the single position's bytes — never on a batch — so
    serial (batch-1) and parallel (any batch size) get identical results.  This
    removes PyTorch's batched-matmul FP non-associativity as a confound, leaving
    the oracle to test exclusively OUR pipeline code.

    Deterministic across processes: only float32 reductions and a NumPy Generator
    seeded from the input (no Python hash(), no global RNG).
    """
    mb = np.ascontiguousarray(mb, dtype=np.float32)
    ob = np.ascontiguousarray(ob, dtype=np.float32)
    flat = np.ascontiguousarray(flat, dtype=np.float32)

    s = float(mb.sum()) - float(ob.sum()) + float(flat.sum())
    value = float(np.tanh(s * 1e-3))

    seed = int(abs(float(flat.sum())) * 1000.0) % (2**31 - 1)
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal(NUM_JOINT_ACTIONS).astype(np.float32)
    # Gather legal logits (in idxs order) to mirror the real evaluator seam.
    # The full vector is still computed from the position alone, so the gathered
    # result stays batch-independent and identical across serial/parallel.
    return value, logits[idxs]


# ─────────────────────────────────────────────────────────────────────────────
# Mock A3 server: serves deterministic_eval through the real worker protocol
# ─────────────────────────────────────────────────────────────────────────────
def _mock_remote_server_loop(request_q: MPQueue, response_qs: List[MPQueue],
                             stop: MPEvent) -> None:
    """Drop-in for the real server's request/response protocol, but with no
    network and no batch effects: reads _RemoteRequest, replies _RemoteResult
    computed by deterministic_eval per position."""
    while not stop.is_set():
        try:
            r: _RemoteRequest = request_q.get(timeout=0.2)
        except Exception:
            continue
        if r is None or getattr(r, "worker_id", None) is None:
            continue
        value, logits = deterministic_eval(r.mb, r.ob, r.flat, r.idxs)
        response_qs[r.worker_id].put(_RemoteResult(
            request_id=r.request_id, worker_id=r.worker_id,
            value=value, logits=logits, model_version=0))


class MockRemoteServer:
    """worker_handles()-compatible stand-in for RemoteInferenceServer that serves
    the deterministic batch-independent evaluator.  Same queue surface, so the
    real worker pool (_init_worker / _generate_parallel) runs against it
    unchanged."""

    def __init__(self, n_workers: int):
        self.request_q: MPQueue = MPQueue()
        self.response_qs: List[MPQueue] = [MPQueue() for _ in range(n_workers)]
        self._stop: MPEvent = MPEvent()
        self._proc = Process(
            target=_mock_remote_server_loop,
            args=(self.request_q, self.response_qs, self._stop),
            daemon=True, name="MockRemoteServer")

    def start(self) -> "MockRemoteServer":
        self._proc.start()
        return self

    def worker_handles(self) -> Tuple[MPQueue, List[MPQueue]]:
        return self.request_q, self.response_qs

    def stop(self) -> None:
        self._stop.set()
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()


# ─────────────────────────────────────────────────────────────────────────────
# Serial / parallel game runners keyed by seed
# ─────────────────────────────────────────────────────────────────────────────
def play_games_serial(
    cfg: SelfPlayConfig, seeds: List[int], evaluator,
) -> Dict[int, Tuple[List[Example], Tuple[int, int]]]:
    """Play each seed serially with the given evaluator — exactly what the serial
    loop does per game (one mcts, _game_rngs per game)."""
    mcts = AlphaZeroMCTS(
        evaluator, c_puct=cfg.c_puct, n_simulations=cfg.n_simulations,
        dirichlet_alpha=cfg.dirichlet_alpha, dirichlet_epsilon=cfg.dirichlet_epsilon,
    )
    out: Dict[int, Tuple[List[Example], Tuple[int, int]]] = {}
    for seed in seeds:
        py_rng, np_rng = _game_rngs(seed)
        examples, scores = play_selfplay_game(
            mcts, n_determinizations=cfg.n_determinizations,
            temp_moves=cfg.temp_moves, seed=seed, py_rng=py_rng, np_rng=np_rng,
        )
        out[seed] = (examples, scores)
    return out


def _play_parallel_by_seed(
    server, cfg: SelfPlayConfig, seeds: List[int], n_workers: int, *,
    games_per_worker: int = 1, timeout_s: float = 60.0, debug_checks: bool = True,
) -> Dict[int, Tuple[List[Example], Tuple[int, int]]]:
    """Run `seeds` through the REAL A3 worker pool (the production
    _init_worker / _worker_play_seed_list / _generate_parallel path) against
    `server`, recovering the seed→result association.

    _generate_parallel returns examples sorted by seed and the oracle uses a
    contiguous seed range, so all_examples[i] corresponds to seeds[i].
    """
    base = seeds[0]
    n = len(seeds)
    assert seeds == list(range(base, base + n)), \
        "oracle requires a contiguous seed range (matches _generate_parallel)"

    request_q, response_qs = server.worker_handles()
    counter = mp.Value("i", 0)
    pool = mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(request_q, response_qs, counter, cfg, games_per_worker,
                  timeout_s, debug_checks),
        maxtasksperchild=None)
    try:
        all_examples, all_scores, _errs = _generate_parallel(
            pool, cfg, n_games=n, game_seed_start=base, n_workers=n_workers,
            fail_fast=True, verbose=False)
        return {base + i: (all_examples[i], all_scores[i]) for i in range(n)}
    finally:
        try:
            pool.map(_worker_cleanup, range(n_workers), chunksize=1)
        except Exception:
            pass
        pool.terminate()
        pool.join()


# ─────────────────────────────────────────────────────────────────────────────
# Example comparison
# ─────────────────────────────────────────────────────────────────────────────
def examples_equal(a: Example, b: Example) -> Tuple[bool, str]:
    """Exact equality of two training examples.  Returns (equal, reason)."""
    if not np.array_equal(a.my_board, b.my_board):
        return False, "my_board differs"
    if not np.array_equal(a.opp_board, b.opp_board):
        return False, "opp_board differs"
    if not np.array_equal(a.flat, b.flat):
        return False, "flat differs"
    if not np.array_equal(a.policy_idx, b.policy_idx):
        return False, "policy_idx differs"
    if not np.array_equal(a.policy_val, b.policy_val):
        return False, "policy_val differs"
    if not np.array_equal(a.legal_idx, b.legal_idx):
        return False, "legal_idx differs"
    if a.z != b.z:
        return False, f"z differs ({a.z} != {b.z})"
    return True, ""


def compare_runs(
    serial: Dict[int, Tuple[List[Example], Tuple[int, int]]],
    parallel: Dict[int, Tuple[List[Example], Tuple[int, int]]],
) -> Tuple[bool, List[str]]:
    """Compare two seed→(examples, scores) maps exactly.  Returns (ok, errors)."""
    errors: List[str] = []
    if set(serial) != set(parallel):
        errors.append(f"seed sets differ: serial={set(serial)} parallel={set(parallel)}")
        return False, errors
    for seed in sorted(serial):
        s_ex, s_sc = serial[seed]
        p_ex, p_sc = parallel[seed]
        if s_sc != p_sc:
            errors.append(f"seed {seed}: scores differ {s_sc} != {p_sc}")
        if len(s_ex) != len(p_ex):
            errors.append(f"seed {seed}: #examples {len(s_ex)} != {len(p_ex)}")
            continue
        for i, (ea, eb) in enumerate(zip(s_ex, p_ex)):
            ok, reason = examples_equal(ea, eb)
            if not ok:
                errors.append(f"seed {seed} example {i}: {reason}")
                break  # one diff per game is enough to flag
    return (len(errors) == 0), errors


# ─────────────────────────────────────────────────────────────────────────────
# The two oracle tests
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline_oracle(
    cfg: SelfPlayConfig, seeds: List[int], worker_counts=(1, 3), verbose=True,
) -> bool:
    """TEST 1 — batch-independent deterministic evaluator.  Serial vs parallel
    must be bit-identical for every worker count in `worker_counts`."""
    if verbose:
        print(f"\n[pipeline oracle] {len(seeds)} seeds, deterministic evaluator")
    serial = play_games_serial(cfg, seeds, deterministic_eval)
    if verbose:
        total_ex = sum(len(ex) for ex, _ in serial.values())
        print(f"  serial: {len(serial)} games, {total_ex} examples")

    all_ok = True
    for nw in worker_counts:
        server = MockRemoteServer(n_workers=nw).start()
        try:
            parallel = _play_parallel_by_seed(server, cfg, seeds, nw)
        finally:
            server.stop()
        ok, errors = compare_runs(serial, parallel)
        all_ok = all_ok and ok
        if verbose:
            status = "MATCH ✓" if ok else "MISMATCH ✗"
            print(f"  parallel({nw} workers) vs serial: {status}")
            for e in errors[:5]:
                print(f"      - {e}")
    return all_ok


def run_realnet_oracle(
    cfg: SelfPlayConfig, seeds: List[int], verbose=True,
) -> bool:
    """TEST 2 — real network through the production RemoteInferenceServer, 1
    worker, games_per_worker=1, max_batch=1, CPU.  Confirms the real serial and
    real parallel paths agree bit-for-bit when batch sizes match."""
    if verbose:
        print(f"\n[real-net oracle] {len(seeds)} seeds, real network, "
              f"1 worker, max_batch=1, CPU")

    # Fixed-weight network shared by both paths.
    torch.manual_seed(cfg.seed)
    net = KingdominoNet(channels=cfg.channels, blocks=cfg.blocks,
                        bilinear_dim=cfg.bilinear_dim).to("cpu").eval()

    # Serial: batch-1 evaluator wrapping the net.
    serial = play_games_serial(cfg, seeds, make_serial_evaluator(net, device="cpu"))

    # Parallel: real server, 1 worker, max_batch=1 (so the server forward is
    # batch-1 too, matching serial numerically).
    model_kwargs = dict(channels=cfg.channels, blocks=cfg.blocks,
                        bilinear_dim=cfg.bilinear_dim)
    server = RemoteInferenceServer(
        n_workers=1, model_kwargs=model_kwargs, device="cpu",
        max_batch=1, max_wait_ms=20.0, debug_checks=True)
    server.start(initial_state_dict=net.state_dict(), wait_until_loaded=True)
    try:
        parallel = _play_parallel_by_seed(server, cfg, seeds, n_workers=1,
                                          games_per_worker=1)
    finally:
        server.stop()

    ok, errors = compare_runs(serial, parallel)
    if verbose:
        status = "MATCH ✓" if ok else "MISMATCH ✗"
        print(f"  parallel(1 worker, batch-1) vs serial: {status}")
        for e in errors[:5]:
            print(f"      - {e}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _oracle_cfg(**overrides) -> SelfPlayConfig:
    base = dict(
        channels=16, blocks=2, bilinear_dim=16,
        n_simulations=16, n_determinizations=1, temp_moves=6,
        c_puct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
        device="cpu", seed=7,
    )
    base.update(overrides)
    return SelfPlayConfig(**base)


def main(verbose: bool = True) -> bool:
    cfg = _oracle_cfg()
    base_seed = cfg.seed * 1_000_003
    seeds = [base_seed + i for i in range(6)]

    t0 = time.time()
    ok1 = run_pipeline_oracle(cfg, seeds, worker_counts=(1, 3), verbose=verbose)
    # Real-net oracle is heavier (torch + IPC); fewer seeds/sims.
    rn_cfg = _oracle_cfg(n_simulations=8, temp_moves=4)
    rn_seeds = seeds[:3]
    ok2 = run_realnet_oracle(rn_cfg, rn_seeds, verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"pipeline oracle:  {'PASS' if ok1 else 'FAIL'}")
        print(f"real-net oracle:  {'PASS' if ok2 else 'FAIL'}")
        print(f"total: {time.time()-t0:.1f}s")
        print('='*60)
    return ok1 and ok2


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    p = argparse.ArgumentParser(description="A3 self-play correctness oracle")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    import sys
    sys.exit(0 if main(verbose=not a.quiet) else 1)