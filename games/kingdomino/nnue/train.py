"""Train the two-head NNUE eval on the Kingdomino self-play buffer (Step 2a).

This is the Kingdomino training ENTRYPOINT. The net (`net.py`) and the training
*math* here (loss, loop, metrics) are game-agnostic in principle, but this CLI is
coupled to Kingdomino via the `data` loader import and the KD-specific input-layout
metadata it saves; a second game would swap the loader (and could factor the loop
into a shared trainer). Two heads, two separate labels (the plan is explicit these
are NOT interchangeable): outcome <- win_target (log-loss / Brier), margin <- own-opp
(MAE). Split is game-honest (whole held-out iterations). Reports both heads against
trivial baselines so "did a dense net learn Kingdomino value?" is answerable.

NOTE: the outcome sigmoid estimates EXPECTED MATCH SCORE (P(win)+0.5*P(draw)), not
literal P(win); the searcher converts via 2*sigmoid-1 + a P0-frame sign flip.
Run10's labels are score-only (no official tiebreaker cascade) -> a small, KNOWN
label noise on the ~2% of near-draw positions (which are exactly the close games
where tiebreakers/calibration matter most); fix at the Rust source before any
strength-focused retraining, not here.

Example:
    PYTHONPATH=. python -m games.kingdomino.nnue.train \
        --buffer runs/kingdomino/cloud_80x6_run10/buffer_final.pkl --epochs 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from games.kingdomino.nnue import data as kd_data
from games.kingdomino.nnue.net import TwoHeadNNUE, config_of


def _avg_rank(a):
    """Average (fractional) ranks, so tied values share the mean of their positions."""
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    avg = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sa[j] == sa[i]:
            j += 1
        avg[i:j] = (i + j - 1) / 2.0
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = avg
    return out


def _metrics(net, X, outcome, margin_raw, margin_scale, device, batch=8192):
    """Validation metrics for both heads (numpy in, dict out)."""
    net.eval()
    probs = np.empty(len(X), np.float32)
    mpred = np.empty(len(X), np.float32)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = torch.from_numpy(X[s:s + batch]).to(device)
            p, m = net.evaluate(xb)
            probs[s:s + batch] = p.cpu().numpy()
            mpred[s:s + batch] = m.cpu().numpy() * margin_scale  # denormalize to points
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    brier = float(np.mean((probs - outcome) ** 2))
    logloss = float(np.mean(-(outcome * np.log(p) + (1 - outcome) * np.log(1 - p))))
    decisive = outcome != 0.5
    acc = float(np.mean((probs[decisive] > 0.5) == (outcome[decisive] > 0.5)))
    mae = float(np.mean(np.abs(mpred - margin_raw)))
    # Ranking: does predicted margin order positions like the true margin? Spearman
    # over AVERAGE ranks — Kingdomino margins are discrete and heavily tied, so
    # average-rank (not arbitrary distinct ranks) is the correct tie handling.
    if len(margin_raw) > 2:
        rp, rt = _avg_rank(mpred), _avg_rank(margin_raw)
        spearman = float(np.corrcoef(rp, rt)[0, 1])
    else:
        spearman = float("nan")
    return {"brier": brier, "logloss": logloss, "acc": acc,
            "margin_mae": mae, "margin_spearman": spearman}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default="runs/kingdomino/cloud_80x6_run10/buffer_final.pkl")
    ap.add_argument("--limit", type=int, default=0, help="subsample N examples (0 = all)")
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of ITERATIONS held out")
    ap.add_argument("--acc-width", type=int, default=256)
    ap.add_argument("--tail-hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--margin-scale", type=float, default=40.0,
                    help="raw margin is divided by this for the margin target")
    ap.add_argument("--margin-loss-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="games/kingdomino/nnue/checkpoints/dense_v1.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"loading buffer {args.buffer} ...")
    t0 = time.time()
    examples = kd_data.load_examples(args.buffer)
    if args.limit and args.limit < len(examples):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(examples), size=args.limit, replace=False)
        examples = [examples[i] for i in idx]
    X, outcome, margin, iteration = kd_data.build_arrays(examples)
    print(f"  {len(X):,} examples, input_dim={X.shape[1]}, "
          f"assembled in {time.time() - t0:.1f}s")

    tr, va, held = kd_data.iteration_split(iteration, args.val_frac, args.seed)
    print(f"  split: {tr.sum():,} train / {va.sum():,} val "
          f"(held-out iterations {held})")

    # Trivial baselines on the val set (what the net must beat). The MAE-optimal
    # constant is the training MEDIAN (not mean); Brier's is the base rate.
    base_rate = float(outcome[tr].mean())
    base_brier = float(np.mean((base_rate - outcome[va]) ** 2))
    median_margin = float(np.median(margin[tr]))
    base_mae = float(np.mean(np.abs(median_margin - margin[va])))
    print(f"  baselines (val): outcome base-rate {base_rate:.3f} -> Brier {base_brier:.4f}; "
          f"margin predict-median -> MAE {base_mae:.2f} pts")

    net = TwoHeadNNUE(X.shape[1], args.acc_width, args.tail_hidden).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  net: input {X.shape[1]} -> acc {args.acc_width} -> tail {args.tail_hidden}, "
          f"{n_params:,} params, device {device}")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    Xtr = torch.from_numpy(X[tr])
    otr = torch.from_numpy(outcome[tr])
    mtr = torch.from_numpy(margin[tr] / args.margin_scale)
    Xva, ova, mva = X[va], outcome[va], margin[va]
    n_tr = len(Xtr)

    best = None
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(n_tr)
        t_ep = time.time()
        run = 0.0
        for s in range(0, n_tr, args.batch_size):
            b = perm[s:s + args.batch_size]
            xb = Xtr[b].to(device)
            ob = otr[b].to(device)
            mb = mtr[b].to(device)
            logit, mpred = net(xb)
            loss = bce(logit, ob) + args.margin_loss_weight * mse(mpred, mb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(b)
        m = _metrics(net, Xva, ova, mva, args.margin_scale, device)
        print(f"  epoch {epoch:3d} | loss {run / n_tr:.4f} | "
              f"val Brier {m['brier']:.4f} logloss {m['logloss']:.4f} acc {m['acc']:.3f} | "
              f"margin MAE {m['margin_mae']:.2f} rho {m['margin_spearman']:.3f} | "
              f"{time.time() - t_ep:.1f}s")
        if best is None or m["brier"] < best["brier"]:
            best = {**m, "epoch": epoch, "state": {k: v.cpu().clone()
                                                   for k, v in net.state_dict().items()}}

    print("\nbest val:", {k: round(v, 4) for k, v in best.items()
                          if k not in ("state", "epoch")}, "at epoch", best["epoch"])
    print(f"  vs baseline Brier {base_brier:.4f} (net {best['brier']:.4f}), "
          f"MAE {base_mae:.2f} (net {best['margin_mae']:.2f} pts)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best["state"],
        "config": config_of(net),
        "margin_scale": args.margin_scale,
        "input_layout": {"my_board": kd_data.BOARD_SIZE, "opp_board": kd_data.BOARD_SIZE,
                         "flat": X.shape[1] - 2 * kd_data.BOARD_SIZE},
        "val_metrics": {k: v for k, v in best.items() if k not in ("state",)},
        "baselines": {"outcome_brier": base_brier, "margin_mae": base_mae},
        "args": vars(args),
    }, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
