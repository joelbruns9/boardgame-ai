"""
test_rust_mcts_equiv.py — Milestone 4, Phase 1: verify the Rust MCTS tree
reproduces the Python AlphaZeroMCTS tree EXACTLY under a deterministic mock
evaluator (serial / leaf_batch=1, no Dirichlet noise).

WHY A MOCK: with identical evaluator outputs and identical child ordering, the
two trees must produce bit-identical root visit-count vectors after N
simulations.  No torch, no GroupNorm batching noise — any divergence is a tree
bug, not floating-point.  This is the gate before the batched path or the real
network.

THE EVALUATOR IS A PYTHON CALLABLE for BOTH trees.  The Python tree
(AlphaZeroMCTS) calls it directly; the Rust tree calls back into Python for leaf
eval.  So the mock — and the state hash — run in Python in both cases, on the
encoded arrays, which Milestone 2 proved are bit-identical between the engines.
Cross-language hash identity is therefore automatic; we still define the hash
explicitly (stable md5, NOT Python's salted hash()) and test it standalone.

The only values each tree computes INDEPENDENTLY are terminal values
(Rust tanh vs Python compute_target_z) and the legal-prior softmax (Rust vs
numpy).  If those agree to the last bit, visit counts are bit-identical.

Run (PowerShell):
  python -m games.kingdomino.test_rust_mcts_equiv
"""
from __future__ import annotations

import hashlib
import math
import random
import sys

import numpy as np

from games.kingdomino.game import GameState, Phase
from games.kingdomino.encoder import encode_state
from games.kingdomino.action_codec import encode_action
from games.kingdomino.mcts_az import AlphaZeroMCTS
from games.kingdomino.test_rust_game_equiv import _rust_from_python, _translate

import kingdomino_rust

MOCK_CONST = 0.001  # priors = softmax(joint_index * MOCK_CONST)
CPUCT = 1.5
FPU = 0.0


# ─── deterministic mock evaluator ───────────────────────────────────────────
def state_hash(mb, ob, flat) -> int:
    """Stable, cross-process hash of the encoded position, mapped to a bounded
    range so tanh(hash/1000) spreads across (-1, 1) instead of saturating.

    Uses md5 (NOT Python's hash(), which is salted per process).  Operates on
    the encoded arrays, which are bit-identical between the Python and Rust
    engines (Milestone 2), so the same position hashes identically regardless of
    which engine produced the encoding.
    """
    h = hashlib.md5()
    h.update(np.ascontiguousarray(mb, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(ob, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(flat, dtype=np.float32).tobytes())
    val = int.from_bytes(h.digest()[:8], "little")
    return (val % 8000) - 4000  # → tanh input in (-4, 4)


def mock_value(mb, ob, flat) -> float:
    return math.tanh(state_hash(mb, ob, flat) / 1000.0)


def mock_single(mb, ob, flat, idxs):
    """Single-leaf evaluator for the Python AlphaZeroMCTS seam:
    (mb, ob, flat, idxs) -> (value, gathered_legal_logits).

    Values are rounded through f32 precision (then widened back to f64) so the
    Python tree sees BIT-IDENTICAL numbers to the Rust tree, which is now fed f32
    by mock_batched and widens f32→f64 on entry (the halved-D2H contract).  Kept
    as f64 arrays here because AlphaZeroMCTS softmaxes in f64."""
    gathered = (idxs.astype(np.float64) * MOCK_CONST).astype(np.float32).astype(np.float64)
    return float(np.float32(mock_value(mb, ob, flat))), gathered


def mock_batched(mb, ob, flat, idxs_list):
    """Batched evaluator for the Rust tree's leaf callback:
    (mb (K,9,13,13), ob (K,9,13,13), flat (K,261), idxs_list) ->
    (values (K,) f32, [gathered_i f32]).  Identical math to mock_single.
    Returns f32 — the Rust tree casts to f64 on entry (halved-D2H contract)."""
    k = mb.shape[0]
    values = np.empty(k, dtype=np.float32)   # f32-truncated; Rust widens to f64
    gathered = []
    for i in range(k):
        values[i] = mock_value(mb[i], ob[i], flat[i])
        # Compute in f64 then truncate to f32 — same rounding as mock_single, so
        # both trees receive identical numbers after Rust's f32→f64 widening.
        gathered.append((idxs_list[i].astype(np.float64) * MOCK_CONST).astype(np.float32))
    return values, gathered


# Uniform mock (value 0, uniform priors).  Used by the OPEN-LOOP equivalence
# gate: open-loop resamples determinizations independently per engine, so a
# high-variance value function (the hash mock) makes the two engines chase noise
# over different deck samples → large TV even when both are correct.  A uniform
# mock removes that value noise, isolating search-STRUCTURE consistency (the same
# choice as test_open_loop_mcts TEST 13).
def mock_uniform_single(mb, ob, flat, idxs):
    return 0.0, np.zeros(len(idxs), dtype=np.float64)


def mock_uniform_batched(mb, ob, flat, idxs_list):
    # f32 for the Rust tree (halved-D2H contract); zeros are exact in both
    # precisions, so the open-loop TV gate is unchanged.
    k = mb.shape[0]
    return (np.zeros(k, dtype=np.float32),
            [np.zeros(len(idxs_list[i]), dtype=np.float32) for i in range(k)])


# ─── standalone hash check (run before the tree test) ───────────────────────
def check_hash() -> bool:
    ok = True
    py = GameState.new(seed=0)
    rs = _rust_from_python(py)
    rng = random.Random(123)
    # Walk a few plies into a placement phase, checking hash agreement each step.
    for _ in range(15):
        if py.phase == Phase.GAME_OVER:
            break
        for player in (0, 1):
            mb_p, ob_p, flat_p = encode_state(py, player)
            mb_r, ob_r, flat_r = rs.encode(player)
            hp = state_hash(mb_p, ob_p, flat_p)
            hr = state_hash(np.asarray(mb_r), np.asarray(ob_r), np.asarray(flat_r))
            if hp != hr:
                print(f"  HASH MISMATCH py={hp} rust-encoded={hr}")
                ok = False
            if hp != state_hash(mb_p, ob_p, flat_p):  # determinism
                print("  HASH NON-DETERMINISTIC")
                ok = False
        a = rng.choice(py.legal_actions())
        py = py.step(a)
        rs = rs.step(*_translate(a))
    print(f"  hash standalone check: {'PASS' if ok else 'FAIL'}")
    return ok


# ─── collect snapshot states across phases ──────────────────────────────────
def _collect_states(seed: int, max_snapshots: int = 8):
    """Walk one game, returning (py_state, rs_state) snapshots spread across the
    game (so we exercise INITIAL_SELECTION, early/mid/late PLACE, and FINAL)."""
    py = GameState.new(seed=seed)
    rs = _rust_from_python(py)
    rng = random.Random(seed * 2654435761 & 0xFFFFFFFF)
    snaps = []
    ply = 0
    # Snapshot at a spread of plies; ~52 plies per game.
    targets = {0, 2, 8, 16, 26, 36, 44, 49}
    while py.phase != Phase.GAME_OVER:
        if ply in targets:
            snaps.append((py, rs))
            if len(snaps) >= max_snapshots:
                break
        a = rng.choice(py.legal_actions())
        py = py.step(a)
        rs = rs.step(*_translate(a))
        ply += 1
    return snaps


# ─── one search comparison ──────────────────────────────────────────────────
def _compare(py_state, rs_state, n_sims: int):
    """Return (match: bool, detail: str). Compares root visit counts keyed by
    joint index."""
    py_mcts = AlphaZeroMCTS(mock_single, c_puct=CPUCT, n_simulations=n_sims, fpu=FPU)
    vc, _ = py_mcts.search(py_state, add_noise=False, leaf_batch=1)
    py_counts = {encode_action(a, py_state): c for a, c in vc.items()}

    rs_mcts = kingdomino_rust.RustMCTS()
    rust_pairs = rs_mcts.search(rs_state, mock_batched, n_sims,
                                dirichlet_eps=0.0, fpu=FPU, cpuct=CPUCT)
    rust_counts = {int(idx): int(c) for idx, c in rust_pairs}

    if py_counts == rust_counts:
        return True, ""

    keys = sorted(set(py_counts) | set(rust_counts))
    diffs = [(k, py_counts.get(k, 0), rust_counts.get(k, 0))
             for k in keys if py_counts.get(k, 0) != rust_counts.get(k, 0)]
    total_abs = sum(abs(p - r) for _, p, r in diffs)
    sample = diffs[:6]
    return False, (f"{len(diffs)} indices differ (total |Δ|={total_abs}); "
                   f"set_match={set(py_counts) == set(rust_counts)}; "
                   f"sample(idx,py,rust)={sample}")


# ─── open-loop equivalence: Rust batched_open_loop vs Python OpenLoopMCTS ─────
def _py_state_from_rust(rs) -> GameState:
    """Build a fresh-game (INITIAL_SELECTION) Python GameState matching a Rust
    state's public fields (current_row / deck / start_player).  Only valid at
    move 0, where boards are empty and there are no claims."""
    from games.kingdomino.game import GameConfig
    from games.kingdomino.board import Board
    cfg = GameConfig()
    return GameState(
        config=cfg,
        boards=[Board(cfg.canvas_size), Board(cfg.canvas_size)],
        deck=list(rs.deck()),
        current_row=list(rs.current_row()),
        pending_claims=[],
        next_claims=[],
        phase=Phase.INITIAL_SELECTION,
        actor_index=0,
        initial_pick_count=0,
        start_player=rs.start_player,  # int attribute (getter), not a method
    )


def _total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def run_open_loop_equiv(seeds=(0, 1, 2), n_sims: int = 200) -> bool:
    """Open-loop equivalence GATE: the Rust BatchedMCTS open-loop tree and the
    Python OpenLoopMCTS must produce statistically consistent move-0 root visit
    distributions (TV < 0.10) under the same mock evaluator, Dirichlet off.

    NOT bit-identical: the two engines derive per-simulation determinization
    RNGs differently, so this is a statistical (TV) gate, not exact equality.
    Compared at move 0 (the shared public start position); after move 0 the two
    engines pick possibly-different moves and diverge.  Uses the UNIFORM mock so
    the gate measures search-structure consistency, not value-driven
    determinization noise (see mock_uniform_single).
    """
    from games.kingdomino.mcts_az import OpenLoopMCTS, visit_counts_to_policy
    from games.kingdomino.action_codec import NUM_JOINT_ACTIONS

    print("\n=== MCTS open-loop: Rust batched_open_loop vs Python OpenLoopMCTS ===")
    max_tv = 0.0
    ok = True
    for seed in seeds:
        rs0 = kingdomino_rust.batched_new_game(seed)
        py0 = _py_state_from_rust(rs0)
        # Sanity: the constructed Python state must encode identically to Rust's.
        mb_p, ob_p, flat_p = encode_state(py0, 0)
        mb_r, ob_r, flat_r = rs0.encode(0)
        if state_hash(mb_p, ob_p, flat_p) != state_hash(
            np.asarray(mb_r), np.asarray(ob_r), np.asarray(flat_r)
        ):
            print(f"  [seed {seed}] FAIL: constructed Python state != Rust state")
            ok = False
            continue

        # Python OpenLoopMCTS root visit distribution at move 0.
        ol = OpenLoopMCTS(mock_uniform_single, n_simulations=n_sims,
                          dirichlet_epsilon=0.0, fpu=FPU, c_puct=CPUCT)
        vc, _ = ol.search(py0, add_noise=False, rng=np.random.default_rng(seed))
        policy_py = visit_counts_to_policy(vc, py0, temperature=1.0).astype(np.float64)

        # Rust BatchedMCTS open-loop: play one game, take the move-0 example.
        batched = kingdomino_rust.BatchedMCTS(
            1, 1, seed, n_sims, leaf_batch=1, virtual_loss=1, cpuct=CPUCT,
            fpu=FPU, dirichlet_alpha=0.3, dirichlet_eps=0.0, temp_moves=0,
            open_loop=True,
        )
        finished = []
        while not batched.done():
            mb, ob, flat, idxs_list = batched.step()
            vals, gathered = mock_uniform_batched(
                np.asarray(mb), np.asarray(ob), np.asarray(flat),
                [np.asarray(x) for x in idxs_list],
            )
            finished.extend(batched.update(vals, gathered))
        ex0 = finished[0][1][0]  # (mb,ob,flat,pidx,pval,lidx,z,own,opp,win)
        pidx = np.asarray(ex0[3], dtype=np.int64)
        pval = np.asarray(ex0[4], dtype=np.float64)
        policy_rs = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float64)
        policy_rs[pidx] = pval

        tv = _total_variation(policy_py, policy_rs)
        max_tv = max(max_tv, tv)
        print(f"  [seed {seed}] move-0 visit-dist TV = {tv:.4f}  "
              f"(py nonzero={int((policy_py>0).sum())}, rust nonzero={len(pidx)})")
        if tv >= 0.10:
            ok = False

    print(f"  max pairwise TV = {max_tv:.4f}  "
          f"({'PASS' if max_tv < 0.10 else 'FAIL'} at 0.10 gate)")
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def run(seeds=(0, 1, 2), sim_counts=(50, 200, 800), verbose: bool = False) -> bool:
    print("=== MCTS Phase 1: Rust tree vs Python AlphaZeroMCTS (mock evaluator) ===")
    if not check_hash():
        print("RESULT: FAIL (hash check)")
        return False

    results = {n: [0, 0] for n in sim_counts}  # n_sims -> [matches, total]
    for seed in seeds:
        snaps = _collect_states(seed)
        for si, (py_state, rs_state) in enumerate(snaps):
            for n in sim_counts:
                match, detail = _compare(py_state, rs_state, n)
                results[n][1] += 1
                if match:
                    results[n][0] += 1
                elif verbose:
                    print(f"  [seed {seed} snap {si} phase {py_state.phase.name} "
                          f"n_sims {n}] MISMATCH: {detail}")

    all_ok = True
    for n in sim_counts:
        m, t = results[n]
        print(f"  n_sims={n:<4d}: {m}/{t} states bit-identical")
        all_ok = all_ok and (m == t)
    print(f"RESULT: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    ok_closed = run(verbose=verbose)
    ok_open = run_open_loop_equiv()
    sys.exit(0 if (ok_closed and ok_open) else 1)
