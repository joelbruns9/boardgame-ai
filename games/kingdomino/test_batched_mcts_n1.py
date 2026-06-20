"""
test_batched_mcts_n1.py - M6 gate for Rust BatchedMCTS.

The current BatchedMCTS API is a full-game driver, so the N=1 equivalence gate
replays the same deterministic Rust-side game with RustMCTS.search and compares
every policy target emitted by BatchedMCTS against the reference search.

Run:
  python -m games.kingdomino.test_batched_mcts_n1
"""
from __future__ import annotations

import hashlib
import math
import sys

import numpy as np

import kingdomino_rust

GAME_OVER = 3
LEAF_BATCH = 6
VIRTUAL_LOSS = 1
MOCK_CONST = 0.001
CPUCT = 1.5
FPU = 0.0


def _state_hash(mb, ob, flat) -> int:
    h = hashlib.md5()
    h.update(np.ascontiguousarray(mb, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(ob, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(flat, dtype=np.float32).tobytes())
    val = int.from_bytes(h.digest()[:8], "little")
    return (val % 8000) - 4000


def _mock_value(mb, ob, flat) -> float:
    return math.tanh(_state_hash(mb, ob, flat) / 1000.0)


def mock_batched(mb, ob, flat, idxs_list):
    # f32 to match the halved-D2H contract: both consumers here (RustMCTS.search
    # and BatchedMCTS) widen f32→f64 on entry, so the comparison stays bit-exact.
    k = mb.shape[0]
    values = np.empty(k, dtype=np.float32)
    gathered = []
    for i in range(k):
        values[i] = _mock_value(mb[i], ob[i], flat[i])
        gathered.append((idxs_list[i].astype(np.float64) * MOCK_CONST).astype(np.float32))
    return values, gathered


def _policy_from_pairs(pairs):
    total = sum(int(c) for _, c in pairs)
    if total <= 0:
        raise AssertionError("reference search produced no visits")
    return {
        int(idx): np.float32(int(cnt) / total)
        for idx, cnt in pairs
        if int(cnt) > 0
    }


def _choose_greedy(pairs):
    return min(((int(idx), int(cnt)) for idx, cnt in pairs), key=lambda t: (-t[1], t[0]))[0]


def _step_by_index(state, joint_idx: int):
    actions = state.legal_actions()
    indices = state.legal_action_indices()
    for action, idx in zip(actions, indices):
        if int(idx) == joint_idx:
            placement, pick = action
            return state.step(placement, pick)
    raise AssertionError(f"selected index {joint_idx} is not legal")


def _reference_policies(seed: int, n_sims: int):
    state = kingdomino_rust.batched_new_game(seed)
    mcts = kingdomino_rust.RustMCTS()
    policies = []
    move_num = 0
    while int(state.phase) != GAME_OVER:
        det_seed = kingdomino_rust.batched_det_seed(seed, move_num)
        det = state.redeterminize(seed=det_seed)
        pairs = mcts.search(
            det,
            mock_batched,
            n_sims,
            dirichlet_eps=0.0,
            fpu=FPU,
            cpuct=CPUCT,
            leaf_batch=LEAF_BATCH,
            virtual_loss=VIRTUAL_LOSS,
        )
        policies.append(_policy_from_pairs(pairs))
        state = _step_by_index(state, _choose_greedy(pairs))
        move_num += 1
    return policies, state.scores()


def _run_batched(seed: int, n_sims: int):
    batched = kingdomino_rust.BatchedMCTS(
        1,
        1,
        seed,
        n_sims,
        leaf_batch=LEAF_BATCH,
        virtual_loss=VIRTUAL_LOSS,
        cpuct=CPUCT,
        fpu=FPU,
        dirichlet_eps=0.0,
        temp_moves=0,
    )
    finished = []
    ticks = 0
    while not batched.done():
        mb, ob, flat, idxs_list = batched.step()
        values, gathered = mock_batched(mb, ob, flat, idxs_list)
        finished.extend(batched.update(values, gathered))
        ticks += 1
        if ticks > 100_000:
            raise AssertionError("BatchedMCTS did not finish within the tick guard")
    if len(finished) != 1:
        raise AssertionError(f"expected one finished game, got {len(finished)}")
    game_seed, examples, scores = finished[0]
    if int(game_seed) != seed:
        raise AssertionError(f"finished seed mismatch: got {game_seed}, expected {seed}")
    return list(examples), tuple(int(x) for x in scores), ticks


def _example_policy(example):
    pidx = np.asarray(example[3], dtype=np.int32)
    pval = np.asarray(example[4], dtype=np.float32)
    return {int(i): np.float32(v) for i, v in zip(pidx, pval)}


def _compare(seed: int, n_sims: int, max_positions: int | None, verbose: bool = False):
    expected, expected_scores = _reference_policies(seed, n_sims)
    examples, scores, ticks = _run_batched(seed, n_sims)
    if scores != expected_scores:
        print(f"  seed={seed} sims={n_sims}: score mismatch {scores} vs {expected_scores}")
        return False, 0
    if len(examples) != len(expected):
        print(f"  seed={seed} sims={n_sims}: move count mismatch {len(examples)} vs {len(expected)}")
        return False, 0

    pairs = list(zip(examples, expected))
    if max_positions is not None:
        pairs = pairs[:max_positions]

    for move_num, (example, exp) in enumerate(pairs):
        got = _example_policy(example)
        if set(got) != set(exp):
            print(
                f"  seed={seed} sims={n_sims} move={move_num}: "
                f"policy support mismatch got={sorted(got)[:8]} exp={sorted(exp)[:8]}"
            )
            return False, move_num
        keys = sorted(exp)
        got_v = np.array([got[k] for k in keys], dtype=np.float32)
        exp_v = np.array([exp[k] for k in keys], dtype=np.float32)
        if not np.array_equal(got_v, exp_v):
            max_abs = float(np.max(np.abs(got_v - exp_v)))
            print(
                f"  seed={seed} sims={n_sims} move={move_num}: "
                f"policy values differ max_abs={max_abs:.3g}"
            )
            if verbose:
                diffs = [(k, float(got[k]), float(exp[k])) for k in keys if got[k] != exp[k]]
                print(f"    sample diffs: {diffs[:8]}")
            return False, move_num

    if verbose:
        print(
            f"  seed={seed} sims={n_sims}: checked {len(pairs)}/{len(examples)} "
            f"moves, {ticks} ticks"
        )
    return True, len(pairs)


def run(
    seeds=range(6),
    sim_counts=(50, 200, 800),
    positions_per_seed: int = 48,
    verbose: bool = False,
) -> bool:
    print("=== BatchedMCTS M6 N=1 equivalence vs RustMCTS.search (mock) ===")
    seeds = tuple(seeds)
    ok = True
    for n_sims in sim_counts:
        checked = 0
        sim_ok = True
        for seed in seeds:
            passed, n_checked = _compare(
                seed,
                n_sims,
                max_positions=positions_per_seed,
                verbose=verbose,
            )
            checked += n_checked
            sim_ok = sim_ok and passed
            print(
                f"  seed={seed:<3d} n_sims={n_sims:<4d}: "
                f"{'PASS' if passed else 'FAIL'} ({n_checked} positions)"
            )
        required = len(seeds) * positions_per_seed
        sim_ok = sim_ok and checked == required
        print(
            f"  n_sims={n_sims:<4d}: {checked}/{required} positions "
            f"bit-identical"
        )
        ok = ok and sim_ok
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    sys.exit(0 if run(verbose=verbose) else 1)
