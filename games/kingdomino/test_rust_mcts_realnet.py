"""
test_rust_mcts_realnet.py — Milestone 4, Phase 2: real-network sanity check.

Run the Python AlphaZeroMCTS tree and the Rust RustMCTS tree on the same
positions with the REAL trained network (serial / leaf_batch=1, Dirichlet noise
disabled with eps=0 for determinism), and assert the root visit-count vectors
agree within a tolerance of 5% of total visits per action.

We expect near-identity: the network forward is batch-1 and uses GroupNorm
(batch-independent), so both trees feed it bit-identical inputs (Milestone 2)
and get bit-identical (value, logits) back.  The only divergence is the legal
prior softmax — Python softmaxes the float32 logits via numpy, Rust widens them
to f64 — a ~1e-7 perturbation that can flip a handful of near-tie selections.
Anything beyond 5% would be a bug, not noise.

Run (PowerShell):
  python -m games.kingdomino.test_rust_mcts_realnet
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from games.kingdomino.game import Phase
from games.kingdomino.action_codec import encode_action
from games.kingdomino.network import KingdominoNet
from games.kingdomino.mcts_az import AlphaZeroMCTS, make_serial_evaluator
from games.kingdomino.test_rust_mcts_equiv import _collect_states

import kingdomino_rust

# Pre-261 (four-head, FLAT_SIZE=257-era pick_rank) checkpoint.  NOTE: after the
# pick_pos feature change (FLAT_SIZE 259 → 261) this Phase 4b checkpoint's flat
# input layer (259-wide) no longer load_state_dict's into the current 261-wide
# KingdominoNet — this diagnostic needs a fresh checkpoint from a post-261
# training run.  Not in the required gate; left as-is until the cloud run
# produces a current-architecture net.
CKPT = "checkpoints_ol_4b/iter_0005.pt"
N_SIMS = 200
CPUCT = 1.5
FPU = 0.0
TOL_FRAC = 0.05  # max per-action |Δvisits| allowed, as a fraction of total visits


def load_net(path: str) -> KingdominoNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = KingdominoNet(channels=cfg["channels"], blocks=cfg["blocks"],
                        bilinear_dim=cfg["bilinear_dim"])
    net.load_state_dict(ck["model_state"])
    net.eval()
    return net


def make_real_batched(net: KingdominoNet, device: str = "cpu"):
    """Batched evaluator for the Rust tree's leaf callback, returning f32 values
    and f32 gathered logits (the halved-D2H contract; the Rust tree widens to
    f64).  Same four-head network and leaf-value formula as make_serial_evaluator
    / make_rust_evaluator up to the float32→f64 widening of the logits."""
    import games.kingdomino.mcts_az as _mcts_az
    net = net.to(device).eval()

    def ev(mb, ob, flat, idxs_list):
        with torch.no_grad():
            mb_t = torch.from_numpy(np.ascontiguousarray(mb)).to(device)
            ob_t = torch.from_numpy(np.ascontiguousarray(ob)).to(device)
            flat_t = torch.from_numpy(np.ascontiguousarray(flat)).to(device)
            own, opp, win_prob, logits = net(mb_t, ob_t, flat_t)
        mg = float(_mcts_az.MARGIN_GAIN)
        al = float(_mcts_az.ALPHA)
        margin_val = torch.tanh((own - opp) * mg)
        win_val = 2.0 * win_prob - 1.0
        values = (al * margin_val + (1.0 - al) * win_val).reshape(-1).float().cpu().numpy()
        full = logits.float().cpu().numpy()
        gathered = [full[i][idxs_list[i]] for i in range(len(idxs_list))]
        return values, gathered

    return ev


def run(seeds=(0, 1), n_positions: int = 10, verbose: bool = False) -> bool:
    net = load_net(CKPT)
    serial_eval = make_serial_evaluator(net, device="cpu")
    rust_eval = make_real_batched(net, device="cpu")

    positions = []
    for s in seeds:
        for py_state, rs_state in _collect_states(s):
            if py_state.phase != Phase.GAME_OVER:
                positions.append((py_state, rs_state))
            if len(positions) >= n_positions:
                break
        if len(positions) >= n_positions:
            break

    print(f"=== MCTS Phase 2: real net ({CKPT}), {len(positions)} positions, "
          f"n_sims={N_SIMS}, tol={TOL_FRAC:.0%} ===")

    rs_mcts = kingdomino_rust.RustMCTS()
    worst_frac = 0.0
    worst_detail = ""
    failures = 0

    for pi, (py_state, rs_state) in enumerate(positions):
        py_mcts = AlphaZeroMCTS(serial_eval, c_puct=CPUCT, n_simulations=N_SIMS, fpu=FPU)
        vc, _ = py_mcts.search(py_state, add_noise=False, leaf_batch=1)
        py_counts = {encode_action(a, py_state): c for a, c in vc.items()}

        rust_pairs = rs_mcts.search(rs_state, rust_eval, N_SIMS,
                                    dirichlet_eps=0.0, fpu=FPU, cpuct=CPUCT, leaf_batch=1)
        rust_counts = {int(idx): int(c) for idx, c in rust_pairs}

        total = sum(py_counts.values())  # == N_SIMS
        keys = set(py_counts) | set(rust_counts)
        max_abs = max((abs(py_counts.get(k, 0) - rust_counts.get(k, 0)) for k in keys), default=0)
        frac = max_abs / total if total else 0.0
        if frac > worst_frac:
            worst_frac = frac
            worst_detail = (f"pos {pi} ({py_state.phase.name}): max |Δ|={max_abs} "
                            f"of {total} = {frac:.3%}")
        ok = frac <= TOL_FRAC and set(py_counts) == set(rust_counts)
        if not ok:
            failures += 1
        if verbose:
            print(f"  pos {pi:2d} {py_state.phase.name:<17s} legal={len(keys):3d} "
                  f"max|Δ|={max_abs:3d}/{total} = {frac:6.2%}  {'ok' if ok else 'FAIL'}")

    print(f"  worst divergence: {worst_detail}")
    print(f"  positions over tolerance: {failures}/{len(positions)}")
    ok = failures == 0
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    ok = run(verbose=verbose)
    sys.exit(0 if ok else 1)
