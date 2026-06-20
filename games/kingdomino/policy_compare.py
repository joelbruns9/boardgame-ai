"""
policy_compare.py — does a search produce the same policy targets as serial?

Leaf parallelization (virtual loss) breaks bit-exactness with serial MCTS by
construction, so the existing oracle can no longer be the whole story.  This
harness is the statistical replacement: it measures how far one search's
visit-count policy (pi ∝ root visits — the actual training target) drifts from
another's, over a fixed set of positions, using KL divergence, total-variation
distance, and top-1 move agreement.

It is useful BEFORE leaf parallelization exists, because it establishes the two
reference points that turn "looks fine" into a pass/fail bar:

  * determinism mode — same search, same seeds, twice.  Must be ~0 (TV, KL ≈ 0).
    Validates the harness and the search's reproducibility.  Once leaf
    parallelization lands, leaf_batch=1 vs serial must also be ~0 here — that's
    the bit-exact gate that separates an implementation bug from the
    approximation.

  * variance mode — serial vs serial with DIFFERENT determinization + Dirichlet
    seeds.  This is the natural run-to-run spread of the targets (sampled deck +
    root noise).  It is the THRESHOLD: when leaf_batch=N is compared against
    serial, its divergence must stay at or under this floor.  If virtual loss
    pushes it well past the floor, it is degrading the targets — the exact
    failure mode to catch on a probe in minutes, not after a multi-day run.

Run today (serial only) to bank the references:
  python -m games.kingdomino.policy_compare --mode determinism --device cuda --sims 800
  python -m games.kingdomino.policy_compare --mode variance     --device cuda --sims 800
  # optionally on your trained net (sharper policies → tighter floor):
  #   ... --warm_start checkpoints/iter_0040.pt

Once leaf parallelization exists (mcts.search / run_pimc gain a leaf_batch arg),
add `--leaf_batch N`: the harness compares serial vs leaf-parallel against the
floor banked above.
"""
from __future__ import annotations

import argparse
import random
import statistics
from typing import Dict, List

import numpy as np
import torch

from games.kingdomino.game import GameState, Phase
from games.kingdomino.action_codec import encode_action
from games.kingdomino.mcts_az import (
    AlphaZeroMCTS, make_serial_evaluator, make_batched_evaluator, run_pimc,
)
from games.kingdomino.network import KingdominoNet


# ── position set ──────────────────────────────────────────────────────────
def gen_positions(n_games: int, per_game: int, seed0: int) -> List[GameState]:
    """Collect decision positions (PLACE_AND_SELECT) spread across early/mid/late
    game, by random playout.  States are immutable (step() is functional), so
    snapshots are safe to keep."""
    out: List[GameState] = []
    for g in range(n_games):
        st = GameState.new(seed=seed0 + g)
        rng = random.Random(9000 + g)
        decisions: List[GameState] = []
        steps = 0
        while st.phase != Phase.GAME_OVER and steps < 400:
            la = st.legal_actions()
            if not la:
                break
            if st.phase == Phase.PLACE_AND_SELECT and len(la) > 1:
                decisions.append(st)
            st = st.step(la[rng.randrange(len(la))])
            steps += 1
        # even spread across the game's decision points
        if decisions:
            idx = np.linspace(0, len(decisions) - 1, num=min(per_game, len(decisions)))
            out.extend(decisions[int(round(i))] for i in idx)
    return out


# ── policy from a search ──────────────────────────────────────────────────
def policy_counts(mcts: AlphaZeroMCTS, state: GameState, *, n_det: int,
                  add_noise: bool, det_seed: int, noise_seed: int,
                  leaf_batch: int = 1) -> Dict[object, int]:
    py_rng = random.Random(det_seed)
    np_rng = np.random.default_rng(noise_seed) if add_noise else None
    vc, _ = run_pimc(mcts, state, py_rng, n_determinizations=n_det,
                     add_noise=add_noise, np_rng=np_rng, leaf_batch=leaf_batch)
    return vc


def _vec(vc: Dict[object, int], order: List[object]) -> np.ndarray:
    v = np.array([float(vc.get(a, 0.0)) for a in order], dtype=np.float64)
    s = v.sum()
    return v / s if s > 0 else v


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p + eps; q = q + eps
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


# ── compare two search configs over the position set ──────────────────────
def compare(mcts, positions, cfg_a, cfg_b):
    kls, tvs, top1 = [], [], []
    for st in positions:
        order = sorted(st.legal_actions(), key=lambda a: encode_action(a, st))
        p = _vec(policy_counts(mcts, st, **cfg_a), order)
        q = _vec(policy_counts(mcts, st, **cfg_b), order)
        kls.append(_kl(p, q)); tvs.append(_tv(p, q))
        top1.append(int(np.argmax(p) == np.argmax(q)))
    return kls, tvs, top1


def _report(name, kls, tvs, top1):
    n = len(kls)
    print(f"\n  {name}  (n={n} positions)")
    print(f"    KL(serial||other) : mean {statistics.mean(kls):.4f}  "
          f"median {statistics.median(kls):.4f}  max {max(kls):.4f}")
    print(f"    total-variation   : mean {statistics.mean(tvs):.4f}  "
          f"median {statistics.median(tvs):.4f}  max {max(tvs):.4f}")
    print(f"    top-1 agreement   : {100*sum(top1)/n:.1f}%")


def main() -> None:
    p = argparse.ArgumentParser(description="Policy-divergence validator for "
                                            "leaf parallelization")
    p.add_argument("--mode", choices=["determinism", "variance"],
                   default="variance",
                   help="determinism: same seeds twice (expect ~0). "
                        "variance: serial vs serial, different seeds (the floor).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--bilinear_dim", type=int, default=64)
    p.add_argument("--warm_start", default=None,
                   help="checkpoint to load (sharper net → tighter, more "
                        "realistic floor). Random init if omitted.")
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--n_det", type=int, default=1)
    p.add_argument("--no_noise", action="store_true",
                   help="disable root Dirichlet noise (training uses it on)")
    p.add_argument("--games", type=int, default=6)
    p.add_argument("--per_game", type=int, default=5)
    p.add_argument("--leaf_batch", type=int, default=1,
                   help="leaf-parallel batch for the 'other' run (needs step 5)")
    a = p.parse_args()

    net = KingdominoNet(channels=a.channels, blocks=a.blocks,
                        bilinear_dim=a.bilinear_dim)
    if a.warm_start:
        sd = torch.load(a.warm_start, map_location="cpu")
        model_sd = sd.get("model_state", sd)  # trainer saves as {model_state, ...}
        net.load_state_dict(model_sd)
        print(f"loaded net from {a.warm_start}")
    # The serial evaluator backs the leaf_batch=1 reference (bit-exact serial);
    # the batched evaluator backs leaf_batch>1 (one real forward over N leaves —
    # the production path). leaf_batch=1 never calls the batched one.
    mcts = AlphaZeroMCTS(
        make_serial_evaluator(net, device=a.device),
        batched_evaluator=make_batched_evaluator(net, device=a.device),
        n_simulations=a.sims)

    positions = gen_positions(a.games, a.per_game, seed0=0)
    add_noise = not a.no_noise
    print(f"mode={a.mode}  sims={a.sims}  n_det={a.n_det}  noise={add_noise}  "
          f"leaf_batch={a.leaf_batch}  positions={len(positions)}  "
          f"net={'trained' if a.warm_start else 'random-init'}")

    # run A is always serial reference
    cfg_a = dict(n_det=a.n_det, add_noise=add_noise, det_seed=1000,
                 noise_seed=2000, leaf_batch=1)
    if a.mode == "determinism":
        cfg_b = dict(cfg_a)                      # identical → expect ~0
        label = "serial vs serial (SAME seeds)"
    else:
        cfg_b = dict(n_det=a.n_det, add_noise=add_noise, det_seed=5000,
                     noise_seed=6000, leaf_batch=a.leaf_batch)
        label = (f"serial vs {'leaf_batch='+str(a.leaf_batch) if a.leaf_batch>1 else 'serial'}"
                 f" (DIFFERENT seeds — the floor)")

    # per-position seeds so each position gets its own world/noise
    def per_pos(cfg, i):
        c = dict(cfg); c["det_seed"] += i; c["noise_seed"] += i; return c
    kls, tvs, top1 = [], [], []
    for i, st in enumerate(positions):
        order = sorted(st.legal_actions(), key=lambda x: encode_action(x, st))
        pv = _vec(policy_counts(mcts, st, **per_pos(cfg_a, i)), order)
        qv = _vec(policy_counts(mcts, st, **per_pos(cfg_b, i)), order)
        kls.append(_kl(pv, qv)); tvs.append(_tv(pv, qv))
        top1.append(int(np.argmax(pv) == np.argmax(qv)))
    _report(label, kls, tvs, top1)

    if a.mode == "determinism":
        ok = max(tvs) < 1e-6
        print(f"\n  → {'PASS' if ok else 'FAIL'}: same-seed divergence "
              f"{'is ~0 (reproducible)' if ok else 'is NONZERO — determinism bug'}"
              f"  [max TV {max(tvs):.2e}].")
        print("    (On GPU a tiny nonzero is just cuDNN FP noise. The STRICT "
              "bit-exact gate for leaf_batch=1 == serial is correctness_oracle.py "
              "on CPU — run that once leaf parallelization lands.)")
    else:
        print("\n  → This is your reference floor. When leaf_batch>1 is wired, "
              "its divergence vs serial must stay at or under these numbers to be "
              "safe; well above = virtual loss is degrading the targets.")


if __name__ == "__main__":
    main()