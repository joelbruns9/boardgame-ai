"""Train the Step-3 sparse NNUE on replayable Kingdomino trajectories.

This entrypoint deliberately leaves the dense Step-2 trainer untouched.  It
derives or strict-loads disposable CSR artifacts, applies random D4 feature
permutations at batch time, and trains the two exported heads plus the final-score
auxiliary heads.  The reserved test split is never opened here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn

from .sparse_data import (
    AUX_SCORE_NAMES,
    MARGIN_SCALE,
    TARGET_SCHEMA,
    derive_split,
    load_packed,
)
from .sparse_encoder import CORE_SIZE, core_schema_hash
from .sparse_net import SparseNNUE, sparse_config_of
from .summary_encoder import SUMMARY_SIZE, summary_schema_hash


def _load_or_derive(source: str, cache_dir: str, split: str, max_games: int):
    suffix = f"_{max_games}games" if max_games else ""
    path = Path(cache_dir) / f"{split}{suffix}.npz"
    if path.exists():
        print(f"loading packed {split}: {path}")
        try:
            return load_packed(path)
        except ValueError as exc:
            # Encoded artifacts are disposable caches. Keep the strict failure
            # visible, then rebuild from the replayable source of truth.
            print(f"  stale packed cache ({exc}); rebuilding from source")
    print(f"deriving packed {split} from replay source ...")
    t0 = time.time()
    data = derive_split(source, split, path, max_games=max_games)
    print(
        f"  {len(data):,} positions / {data.metadata['game_count']:,} games, "
        f"{len(data.indices):,} active indices in {time.time() - t0:.1f}s -> {path}"
    )
    return data


def validation_metrics(net, data, device, batch_size=4096):
    net.eval()
    probs = np.empty(len(data), np.float32)
    margins = np.empty(len(data), np.float32)
    aux = np.empty_like(data.aux_scores)
    bonus = np.empty_like(data.aux_bonus)
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            rows = np.arange(start, min(start + batch_size, len(data)))
            b = data.batch(rows, device=device)
            logit, margin, ap, bp = net(b["indices"], b["offsets"], b["summary"])
            probs[rows] = torch.sigmoid(logit).cpu().numpy()
            margins[rows] = margin.cpu().numpy()
            aux[rows] = ap.cpu().numpy()
            bonus[rows] = torch.sigmoid(bp).cpu().numpy()
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    return {
        "brier": float(np.mean((probs - data.outcome) ** 2)),
        "logloss": float(np.mean(-(data.outcome * np.log(p) + (1 - data.outcome) * np.log(1 - p)))),
        "margin_mae": float(np.mean(np.abs(margins - data.margin)) * MARGIN_SCALE),
        "aux_mae_normalized": float(np.mean(np.abs(aux - data.aux_scores))),
        "bonus_accuracy": float(np.mean((bonus >= 0.5) == (data.aux_bonus >= 0.5))),
    }


def train_epoch(
    net,
    data,
    optimizer,
    device,
    batch_size,
    rng,
    *,
    margin_weight=1.0,
    aux_score_weight=0.25,
    aux_bonus_weight=0.05,
):
    net.train()
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    order = rng.permutation(len(data))
    running = 0.0
    for start in range(0, len(order), batch_size):
        rows = order[start:start + batch_size]
        d4 = rng.integers(0, 8, size=len(rows))
        b = data.batch(rows, d4_choices=d4, device=device)
        outcome, margin, aux_scores, aux_bonus = net(
            b["indices"], b["offsets"], b["summary"]
        )
        loss = (
            bce(outcome, b["outcome"])
            + margin_weight * mse(margin, b["margin"])
            + aux_score_weight * mse(aux_scores, b["aux_scores"])
            + aux_bonus_weight * bce(aux_bonus, b["aux_bonus"])
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running += float(loss.detach()) * len(rows)
    return running / len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="runs/kingdomino/nnue_data/pilot_50k")
    ap.add_argument("--cache-dir", default="runs/kingdomino/nnue_data/packed_v3")
    ap.add_argument("--max-games", type=int, default=0,
                    help="whole games per split for a smoke run (0 = all)")
    ap.add_argument("--acc-width", type=int, default=256)
    ap.add_argument("--tail-hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--margin-loss-weight", type=float, default=1.0)
    ap.add_argument("--aux-score-loss-weight", type=float, default=0.25)
    ap.add_argument("--aux-bonus-loss-weight", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="games/kingdomino/nnue/checkpoints/sparse_v3_pilot.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    train = _load_or_derive(args.source, args.cache_dir, "train", args.max_games)
    val = _load_or_derive(args.source, args.cache_dir, "val", args.max_games)
    base_rate = float(train.outcome.mean())
    base_brier = float(np.mean((val.outcome - base_rate) ** 2))
    median_margin = float(np.median(train.margin))
    base_margin_mae = float(np.mean(np.abs(val.margin - median_margin)) * MARGIN_SCALE)
    print(
        f"train {len(train):,} / val {len(val):,}; baseline Brier {base_brier:.4f}, "
        f"margin MAE {base_margin_mae:.2f} points; device {device}"
    )

    net = SparseNNUE(CORE_SIZE, SUMMARY_SIZE, args.acc_width, args.tail_hidden).to(device)
    print(f"net parameters: {sum(p.numel() for p in net.parameters()):,}")
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(
            net, train, optimizer, device, args.batch_size, rng,
            margin_weight=args.margin_loss_weight,
            aux_score_weight=args.aux_score_loss_weight,
            aux_bonus_weight=args.aux_bonus_loss_weight,
        )
        metrics = validation_metrics(net, val, device)
        print(
            f"epoch {epoch:3d} loss {loss:.4f} | Brier {metrics['brier']:.4f} "
            f"logloss {metrics['logloss']:.4f} | margin MAE {metrics['margin_mae']:.2f} | "
            f"aux nMAE {metrics['aux_mae_normalized']:.3f} bonus acc "
            f"{metrics['bonus_accuracy']:.3f} | {time.time() - t0:.1f}s"
        )
        if best is None or metrics["brier"] < best["brier"]:
            best = {
                **metrics,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
            }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best["state"],
            "config": sparse_config_of(net),
            "core_schema_hash": core_schema_hash(),
            "summary_schema_hash": summary_schema_hash(),
            "target_schema": TARGET_SCHEMA,
            "margin_scale": MARGIN_SCALE,
            "aux_score_names": AUX_SCORE_NAMES,
            "data_provenance": {"train": train.metadata, "val": val.metadata},
            "val_metrics": {k: v for k, v in best.items() if k != "state"},
            "baselines": {"outcome_brier": base_brier, "margin_mae": base_margin_mae},
            "args": vars(args),
        },
        out,
    )
    print(f"best epoch {best['epoch']} Brier {best['brier']:.4f}; saved -> {out}")


if __name__ == "__main__":
    main()
