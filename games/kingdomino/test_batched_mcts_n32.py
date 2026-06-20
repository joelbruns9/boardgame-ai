"""
test_batched_mcts_n32.py - M7 gate for Rust BatchedMCTS.

This exercises the synchronized multi-slot driver with recycling:
  - deterministic gate: n_slots=32, n_games=64, dirichlet_eps=0, temp_moves=0
    and every completed game's policy targets match a RustMCTS.search replay.
  - production-settings smoke: Dirichlet noise and temperature on, checking
    example shapes and target invariants rather than bit identity.

Run:
  python -m games.kingdomino.test_batched_mcts_n32
"""
from __future__ import annotations

import math
import sys

import numpy as np

import kingdomino_rust

from games.kingdomino.encoder import FLAT_SIZE
from games.kingdomino.test_batched_mcts_n1 import (
    CPUCT,
    FPU,
    LEAF_BATCH,
    VIRTUAL_LOSS,
    _example_policy,
    _reference_policies,
    mock_batched,
)

N_SLOTS = 32
N_GAMES = 64
SIM_COUNTS = (50, 200)


def _run_batched_many(
    *,
    n_slots: int,
    n_games: int,
    base_seed: int,
    n_sims: int,
    dirichlet_eps: float,
    temp_moves: int,
):
    batched = kingdomino_rust.BatchedMCTS(
        n_slots,
        n_games,
        base_seed,
        n_sims,
        leaf_batch=LEAF_BATCH,
        virtual_loss=VIRTUAL_LOSS,
        cpuct=CPUCT,
        fpu=FPU,
        dirichlet_eps=dirichlet_eps,
        temp_moves=temp_moves,
    )
    finished = []
    batch_sizes = []
    ticks = 0
    while not batched.done():
        mb, ob, flat, idxs_list = batched.step()
        batch_sizes.append(int(mb.shape[0]))
        values, gathered = mock_batched(mb, ob, flat, idxs_list)
        finished.extend(batched.update(values, gathered))
        ticks += 1
        if ticks > 500_000:
            raise AssertionError("BatchedMCTS did not finish within the tick guard")
    return finished, batch_sizes, ticks


def _compare_game(seed: int, examples, scores, n_sims: int, verbose: bool) -> bool:
    expected, expected_scores = _reference_policies(seed, n_sims)
    scores = tuple(int(x) for x in scores)
    if scores != tuple(int(x) for x in expected_scores):
        print(f"    seed={seed}: score mismatch {scores} vs {expected_scores}")
        return False
    if len(examples) != len(expected):
        print(f"    seed={seed}: move count mismatch {len(examples)} vs {len(expected)}")
        return False

    for move_num, (example, exp) in enumerate(zip(examples, expected)):
        got = _example_policy(example)
        if set(got) != set(exp):
            print(f"    seed={seed} move={move_num}: policy support mismatch")
            return False
        keys = sorted(exp)
        got_v = np.array([got[k] for k in keys], dtype=np.float32)
        exp_v = np.array([exp[k] for k in keys], dtype=np.float32)
        if not np.array_equal(got_v, exp_v):
            max_abs = float(np.max(np.abs(got_v - exp_v)))
            print(f"    seed={seed} move={move_num}: policy values differ max_abs={max_abs:.3g}")
            if verbose:
                diffs = [(k, float(got[k]), float(exp[k])) for k in keys if got[k] != exp[k]]
                print(f"      sample diffs: {diffs[:8]}")
            return False
    return True


def _check_example_invariants(example) -> bool:
    # Phase 3R extended the BatchedMCTS training tuple from 7 to 10 elements:
    # the per-actor score targets own_score / opp_score / win_target were appended
    # after z.  Unpack all 10 so this test documents (and validates) the current
    # contract rather than silently ignoring the new fields.
    if len(example) != 10:
        print(f"    example tuple has {len(example)} fields, expected 10")
        return False
    mb, ob, flat, pidx, pval, lidx, z, own_score, opp_score, win_target = example
    mb = np.asarray(mb)
    ob = np.asarray(ob)
    flat = np.asarray(flat)
    pidx = np.asarray(pidx, dtype=np.int32)
    pval = np.asarray(pval, dtype=np.float32)
    lidx = np.asarray(lidx, dtype=np.int32)

    if mb.shape != (9, 13, 13) or ob.shape != (9, 13, 13) or flat.shape != (FLAT_SIZE,):
        return False
    if pidx.ndim != 1 or pval.ndim != 1 or lidx.ndim != 1:
        return False
    if len(pidx) != len(pval) or len(lidx) == 0 or len(pidx) == 0:
        return False
    if not np.isfinite(pval).all() or not np.isfinite(float(z)):
        return False
    if not -1.0 <= float(z) <= 1.0:
        return False
    if not np.isclose(float(pval.sum()), 1.0, atol=1e-5):
        return False
    legal = set(int(x) for x in lidx)
    if any(int(x) not in legal for x in pidx):
        return False
    # New Phase 3R score targets: own_score/opp_score are raw (un-normalized)
    # final scores as Python floats; win_target is 1.0 win / 0.5 draw / 0.0 loss.
    if not isinstance(own_score, float) or not isinstance(opp_score, float):
        print(f"    own_score/opp_score not float: {type(own_score)}, {type(opp_score)}")
        return False
    if not (math.isfinite(own_score) and math.isfinite(opp_score)):
        return False
    if not 0.0 <= float(win_target) <= 1.0:
        print(f"    win_target out of range: {win_target}")
        return False
    return True


def _deterministic_gate(n_sims: int, verbose: bool) -> bool:
    finished, batch_sizes, ticks = _run_batched_many(
        n_slots=N_SLOTS,
        n_games=N_GAMES,
        base_seed=0,
        n_sims=n_sims,
        dirichlet_eps=0.0,
        temp_moves=0,
    )
    if len(finished) != N_GAMES:
        print(f"  n_sims={n_sims}: expected {N_GAMES} games, got {len(finished)}")
        return False
    by_seed = {int(seed): (list(examples), scores) for seed, examples, scores in finished}
    expected_seeds = set(range(N_GAMES))
    if set(by_seed) != expected_seeds:
        print(f"  n_sims={n_sims}: seed set mismatch")
        return False

    ok = True
    checked_positions = 0
    for seed in range(N_GAMES):
        examples, scores = by_seed[seed]
        ok = _compare_game(seed, examples, scores, n_sims, verbose) and ok
        checked_positions += len(examples)

    mean_batch = float(np.mean([b for b in batch_sizes if b > 0]))
    max_batch = max(batch_sizes)
    print(
        f"  n_sims={n_sims:<4d}: {'PASS' if ok else 'FAIL'} "
        f"({checked_positions} positions, ticks={ticks}, mean_batch={mean_batch:.1f}, "
        f"max_batch={max_batch})"
    )
    return ok


def _production_smoke(verbose: bool) -> bool:
    finished, batch_sizes, ticks = _run_batched_many(
        n_slots=N_SLOTS,
        n_games=N_GAMES,
        base_seed=10_000,
        n_sims=50,
        dirichlet_eps=0.25,
        temp_moves=20,
    )
    ok = len(finished) == N_GAMES
    n_examples = 0
    for seed, examples, scores in finished:
        if len(examples) == 0:
            print(f"  smoke seed={seed}: no examples")
            ok = False
        scores = tuple(int(x) for x in scores)
        if not all(math.isfinite(x) for x in scores):
            print(f"  smoke seed={seed}: non-finite scores {scores}")
            ok = False
        for example in examples:
            n_examples += 1
            if not _check_example_invariants(example):
                print(f"  smoke seed={seed}: bad example invariant")
                ok = False
                break

    if verbose:
        print(f"  smoke batch sizes: first={batch_sizes[:8]} last={batch_sizes[-8:]}")
    print(
        f"  production smoke: {'PASS' if ok else 'FAIL'} "
        f"({len(finished)} games, {n_examples} examples, ticks={ticks}, "
        f"mean_batch={np.mean([b for b in batch_sizes if b > 0]):.1f})"
    )
    return ok


def run(verbose: bool = False) -> bool:
    print("=== BatchedMCTS M7 N=32 recycling gate ===")
    ok = True
    for n_sims in SIM_COUNTS:
        ok = _deterministic_gate(n_sims, verbose=verbose) and ok
    ok = _production_smoke(verbose=verbose) and ok
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    sys.exit(0 if run(verbose=verbose) else 1)
