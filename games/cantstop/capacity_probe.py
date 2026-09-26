"""Is the value net capacity-limited? Same frozen data, different net sizes.

One dataset of self-play from a trained net, split BY GAME into train and
held-out. Nets of several sizes are trained on it with the same budget and
a decaying learning rate; the question is whether bigger nets reach a lower
held-out loss. If they do, capacity binds and the next run should use a
bigger net. If every size lands on the same floor, the limit is the targets
or the dice, not the net.

Held-out loss includes the targets' own entropy (an irreducible floor), so
only DIFFERENCES between sizes matter.

    python -m games.cantstop.capacity_probe --net runs/lr1e4/iter_0120.pt
"""

import argparse
import json
import math
import time

import numpy as np
import torch

from .engine import RuleSet
from .model import CantStopNet, NetEvaluator, load_net, masked_soft_cross_entropy, seat_mask_tensor
from .portable_rng import PortableRng
from .rust_pool import game_seeds, run_pool
from .self_play import td_targets

SIZES = {
    "256x2 (92k)": (256, 256),
    "512-512-256-256 (0.51M, current)": (512, 512, 256, 256),
    "1024x4 (3.3M)": (1024, 1024, 1024, 1024),
    "1024x6 (5.4M)": (1024,) * 6,
}


def dataset(net_path, games, lam, seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ev = NetEvaluator(load_net(net_path, device=device), device=device)
    rules = RuleSet.make(2)
    results = run_pool([rules] * games, game_seeds(PortableRng(seed), games),
                       [ev])
    xs = [r.features for r in results if len(r)]
    ys = [td_targets(r, lam) for r in results if len(r)]
    return xs, ys


def train_one(hidden, x_tr, y_tr, x_ho, y_ho, epochs, batch, lr, device,
              seed):
    torch.manual_seed(seed)
    net = CantStopNet(hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    steps = epochs * math.ceil(len(x_tr) / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps,
                                                       eta_min=lr / 100)
    m_tr, m_ho = seat_mask_tensor(x_tr), seat_mask_tensor(x_ho)
    gen = torch.Generator(device=device).manual_seed(seed)
    best, curve = float("inf"), []
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(x_tr), device=device, generator=gen)
        for i in range(0, len(x_tr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = masked_soft_cross_entropy(net(x_tr[idx]), y_tr[idx],
                                             m_tr[idx])
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            ho = float(masked_soft_cross_entropy(net(x_ho), y_ho, m_ho))
            tr = float(masked_soft_cross_entropy(net(x_tr[:len(x_ho)]),
                                                 y_tr[:len(x_ho)],
                                                 m_tr[:len(x_ho)]))
        best = min(best, ho)
        curve.append((round(tr, 5), round(ho, 5)))
    params = sum(p.numel() for p in net.parameters())
    # Cross-entropy includes the targets' own entropy, a floor no net can go
    # below; CE - H is the part a better net could still remove (review).
    with torch.no_grad():
        t = y_ho.clamp_min(1e-12)
        h_ho = float(-(y_ho * torch.log(t)).sum(dim=-1).mean())
    return {"params": params, "best_heldout": best, "final_train": curve[-1][0],
            "final_heldout": curve[-1][1], "heldout_target_entropy": h_ho,
            "best_heldout_minus_entropy": best - h_ho, "curve": curve}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--net", required=True)
    ap.add_argument("--games", type=int, default=20_000)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--heldout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = time.perf_counter()
    xs, ys = dataset(args.net, args.games, args.lam, args.seed)
    gen_seconds = time.perf_counter() - t
    # Split by GAME: rows of one game are correlated, so a row-level split
    # would leak and flatter the larger nets.
    order = np.random.default_rng(args.seed).permutation(len(xs))
    cut = int(len(xs) * args.heldout)
    ho, tr = order[:cut], order[cut:]
    cat = lambda idx, arrs: torch.from_numpy(
        np.concatenate([arrs[i] for i in idx])).to(device)
    x_tr, y_tr, x_ho, y_ho = cat(tr, xs), cat(tr, ys), cat(ho, xs), cat(ho, ys)
    report = {"games": len(xs), "train_rows": len(x_tr),
              "heldout_rows": len(x_ho), "gen_seconds": round(gen_seconds),
              "sizes": {}}
    for name, hidden in SIZES.items():
        t = time.perf_counter()
        r = train_one(hidden, x_tr, y_tr, x_ho, y_ho, args.epochs, args.batch,
                      args.lr, device, args.seed)
        r["seconds"] = round(time.perf_counter() - t)
        report["sizes"][name] = r
        print(f"{name:34} params {r['params']:>9,}  best held-out "
              f"{r['best_heldout']:.5f} (CE-H "
              f"{r['best_heldout_minus_entropy']:.5f})  "
              f"final train {r['final_train']:.5f} "
              f"held-out {r['final_heldout']:.5f}  ({r['seconds']}s)",
              flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
