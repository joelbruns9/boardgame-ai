"""
DIAGNOSTIC — Probe to test whether more simulations sharpen MCTS targets.

This was used to investigate the policy-loss plateau observed early in
training. Conclusion: sims=200 produced unlearnable targets; sims=1600
broke the plateau at cloud run iteration 42. See training_parameters.md
for the full analysis.

Not needed for routine training. Kept for reference.

────────────────────────────────────────────────────────────────────────
Original module docstring follows:

sim_concentration_probe.py — does raising simulations actually sharpen the MCTS
targets, and is eval-caching worth building?

Tests the hypothesis behind the flat policy loss: with too few simulations over
Kingdomino's large branching, the visit-count policy target stays near-uniform
(nothing for the policy head to learn).  For a position, it runs MCTS at several
simulation budgets and reports, per budget:

  legal        : number of legal actions at the root (branching)
  top1         : visit share of the single most-visited action
  eff_actions  : exp(entropy of visit distribution) = "effective # of moves"
                 (near `legal` = mushy/uniform target; near 1-5 = sharp target)
  evals        : total network evaluations during the search
  unique       : distinct encoded positions evaluated
  txp_hit%     : 1 - unique/evals within this search (transposition redundancy)

It also tracks a CUMULATIVE cache across every search in the run (all budgets,
positions, determinizations) and reports the hit rate a persistent per-worker
eval cache would see — the number that decides whether caching is worth building.

If higher budgets drive top1 up and eff_actions down toward a handful, the
targets sharpen and the sims hypothesis is confirmed.  If eff_actions stays near
`legal` regardless of budget, sims are not the (only) problem.

Usage:
  python -m games.kingdomino.sim_concentration_probe \\
      --checkpoint checkpoints/iter_0030.pt --device cuda \\
      --channels 64 --blocks 6 --sims 50 200 800 1500

Does NOT import evaluation.py.
"""
from __future__ import annotations

import argparse
import math
import random
from typing import Dict, Tuple

import numpy as np
import torch

from games.kingdomino.game import GameState, Phase
from games.kingdomino.encoder import encode_state, redeterminize
from games.kingdomino.network import KingdominoNet
from games.kingdomino.mcts_az import AlphaZeroMCTS, make_serial_evaluator


class CountingCachingEvaluator:
    """Wraps an Evaluator with a resettable lossless cache so we can measure
    eval recurrence cleanly.  `reset_cache()` clears everything; each probe row
    starts fresh so its hit rate means exactly "what a per-move cache would save
    at this (sims, determinizations)" — no contamination across rows."""

    def __init__(self, base_eval):
        self._base = base_eval
        self.reset_cache()

    def reset_cache(self) -> None:
        self._cache: Dict[bytes, Tuple[float, np.ndarray]] = {}
        self.total = 0
        self.unique = 0

    @staticmethod
    def _key(mb, ob, flat, idxs) -> bytes:
        return mb.tobytes() + ob.tobytes() + flat.tobytes() + idxs.tobytes()

    def __call__(self, mb, ob, flat, idxs):
        k = self._key(mb, ob, flat, idxs)
        self.total += 1
        if k not in self._cache:
            self.unique += 1
            self._cache[k] = self._base(mb, ob, flat, idxs)
        return self._cache[k]

    def hit_rate(self) -> float:
        return 1.0 - (self.unique / self.total) if self.total else 0.0


def _visit_stats(visit_counts: Dict[object, int]):
    counts = np.array(list(visit_counts.values()), dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return len(visit_counts), 0.0, float(len(visit_counts))
    p = counts / total
    top1 = float(p.max())
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    eff_actions = float(math.exp(entropy))
    return len(visit_counts), top1, eff_actions


def advance(state: GameState, n_moves: int, rng: random.Random) -> GameState:
    """Advance the game by random legal joint actions to reach a mid-game spot."""
    s = state
    for _ in range(n_moves):
        if s.phase == Phase.GAME_OVER:
            break
        actions = s.legal_actions()
        if not actions:
            break
        s = s.step(rng.choice(actions))
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint .pt to load (else random init).")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--bilinear_dim", type=int, default=64)
    p.add_argument("--sims", type=int, nargs="+", default=[50, 200, 800, 1500])
    p.add_argument("--positions", type=int, default=3,
                   help="Distinct game positions to probe.")
    p.add_argument("--advance", type=int, default=8,
                   help="Random plies between probed positions.")
    p.add_argument("--determinizations", type=int, default=1,
                   help="Determinizations per position (exercises cross-world cache).")
    p.add_argument("--c_puct", type=float, default=1.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    net = KingdominoNet(channels=a.channels, blocks=a.blocks,
                        bilinear_dim=a.bilinear_dim)
    if a.checkpoint:
        ckpt = torch.load(a.checkpoint, map_location=a.device)
        sd = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(sd)
        print(f"Loaded {a.checkpoint}")
    else:
        print("No checkpoint: using random-init network "
              "(concentration trend is still meaningful).")
    net = net.to(a.device).eval()

    evaluator = CountingCachingEvaluator(make_serial_evaluator(net, a.device))
    py_rng = random.Random(a.seed)
    np_rng = np.random.default_rng(a.seed)

    print(f"\n{'pos':>3} {'sims':>5} {'legal':>6} {'top1':>6} "
          f"{'eff_act':>8} {'evals':>7} {'unique':>7} {'cache_hit%':>10}")
    print("-" * 64)

    state = GameState.new(seed=a.seed)
    for pos in range(a.positions):
        state = advance(state, a.advance, py_rng)
        if state.phase == Phase.GAME_OVER:
            print(f"(reached game over at position {pos}; stopping)")
            break
        for sims in a.sims:
            # Fresh cache for this row → hit rate = what a per-move cache would
            # save at this (sims, determinizations), counting transpositions
            # within each search plus reuse across determinizations.
            evaluator.reset_cache()
            legals, top1s, effs = [], [], []
            for _ in range(max(1, a.determinizations)):
                mcts = AlphaZeroMCTS(evaluator, c_puct=a.c_puct,
                                     n_simulations=sims)
                det = redeterminize(state, np_rng)
                visit_counts, _root = mcts.search(det, add_noise=False, rng=np_rng)
                n_legal, top1, eff = _visit_stats(visit_counts)
                legals.append(n_legal); top1s.append(top1); effs.append(eff)
            print(f"{pos:>3} {sims:>5} {np.mean(legals):>6.0f} "
                  f"{np.mean(top1s):>6.2f} {np.mean(effs):>8.1f} "
                  f"{evaluator.total:>7} {evaluator.unique:>7} "
                  f"{evaluator.hit_rate()*100:>9.1f}%")

    print("-" * 64)
    print("Read: if eff_act falls toward a handful as sims rise, the targets "
          "sharpen (sims hypothesis confirmed). cache_hit% is the per-move eval "
          "redundancy — high with --determinizations > 1 means an eval cache "
          "pays off (esp. given the info-set-safe encoding).")


if __name__ == "__main__":
    main()