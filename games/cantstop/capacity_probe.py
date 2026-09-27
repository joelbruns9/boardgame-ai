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


def dataset_from_state(path):
    """The Phase 4 replay buffer as a dataset: rows grouped by SOURCE GAME
    (for a split that cannot leak), with each game's variant. Returns
    (xs, ys, variants, rule_names). Plan review: the probe must fit the
    data the run actually trains on, not regenerate 2p games."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    chunks = st["buffer"]["chunks"]
    x = np.concatenate([c[0] for c in chunks])
    y = np.concatenate([c[1] for c in chunks])
    game = np.concatenate([c[2]["game"] for c in chunks])
    variant = np.concatenate([c[2]["variant"] for c in chunks])
    xs, ys, vs = [], [], []
    for g in np.unique(game):
        m = game == g
        xs.append(x[m])
        ys.append(y[m])
        vs.append(int(variant[m][0]))
    return xs, ys, vs, st["config"]["rule_sets"]


def train_one(hidden, x_tr, y_tr, x_ho, y_ho, epochs, batch, lr, device,
              seed, v_ho=None, names=None):
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
        per_variant = {}
        if v_ho is not None:
            for v in torch.unique(v_ho).tolist():
                m = v_ho == v
                ce = float(masked_soft_cross_entropy(net(x_ho[m]), y_ho[m],
                                                     m_ho[m]))
                hv = float(-(y_ho[m] * torch.log(t[m])).sum(dim=-1).mean())
                per_variant[names[v] if names else v] = {
                    "rows": int(m.sum()), "ce": ce, "ce_minus_entropy": ce - hv}
    return {"params": params, "best_heldout": best, "final_train": curve[-1][0],
            "final_heldout": curve[-1][1], "heldout_target_entropy": h_ho,
            "best_heldout_minus_entropy": best - h_ho,
            "per_variant_final": per_variant, "curve": curve}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--net", default=None,
                    help="generate a 2p dataset with this net (legacy mode)")
    ap.add_argument("--data", default=None,
                    help="a Phase 4 state.pt: use its replay buffer instead")
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
    if args.data:
        xs, ys, vs, names = dataset_from_state(args.data)
    elif args.net:
        xs, ys = dataset(args.net, args.games, args.lam, args.seed)
        vs, names = [0] * len(xs), ["2p base (generated)"]
    else:
        raise SystemExit("give --data (Phase 4 state.pt) or --net")
    gen_seconds = time.perf_counter() - t
    # Split by GAME: rows of one game are correlated, so a row-level split
    # would leak and flatter the larger nets.
    order = np.random.default_rng(args.seed).permutation(len(xs))
    cut = int(len(xs) * args.heldout)
    ho, tr = order[:cut], order[cut:]
    cat = lambda idx, arrs: torch.from_numpy(
        np.concatenate([arrs[i] for i in idx])).to(device)
    x_tr, y_tr, x_ho, y_ho = cat(tr, xs), cat(tr, ys), cat(ho, xs), cat(ho, ys)
    v_ho = torch.from_numpy(np.concatenate(
        [np.full(len(xs[i]), vs[i]) for i in ho])).to(device)
    report = {"games": len(xs), "train_rows": len(x_tr),
              "heldout_rows": len(x_ho), "gen_seconds": round(gen_seconds),
              "sizes": {}}
    for name, hidden in SIZES.items():
        t = time.perf_counter()
        r = train_one(hidden, x_tr, y_tr, x_ho, y_ho, args.epochs, args.batch,
                      args.lr, device, args.seed, v_ho, names)
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
