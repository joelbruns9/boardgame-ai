"""Capacity bake-off: supervised comparison of net sizes on a FROZEN buffer.

Question: is 80x6 the binding constraint — can a bigger/deeper net extract
more from the SAME search data? Self-play runs can't answer this (data
quality, staleness, and gating move together); freezing the dataset isolates
capacity. Each arm trains from scratch on the identical train split with the
identical loss (self_play.train_step) until its HELD-OUT loss plateaus, then
reports held-out policy CE, win Brier/BCE, and score MSE, overall and by
game phase.

Split is by contiguous BLOCK, not by example: examples from one game share
their outcome targets (z/own/opp/win), so an example-level split lets the
value heads cheat off sibling positions. The buffer inserts each game's
examples contiguously, so block splitting keeps ~(1 - block/examples_per_game)
of games intact; boundary games leak partially, but the bias is identical
across arms, so between-arm DELTAS (the only numbers that matter here) are
unaffected.

The trained arm checkpoints are playable and saved in standard format — the
winning arm doubles as a distillation warm start for a bigger-net run.

Usage (box, GPU free):
  nohup python -m games.kingdomino.capacity_bakeoff \
      --buffer runs/kingdomino/cloud_80x6_run7/buffer_final.pkl \
      --out_dir runs/kingdomino/capacity_bakeoff \
      --arms 80x6,96x6,80x10 --device cuda \
      > runs/kingdomino/capacity_bakeoff.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from games.kingdomino.network import KingdominoNet
from games.kingdomino.self_play import (
    Example,
    FLAT_LAYOUT,
    NUM_JOINT_ACTIONS,
    ReplayBuffer,
    masked_log_softmax,
    train_step,
)

# Buffers pickled by `python -m games.kingdomino.self_play` reference Example
# under __main__; register it there so those pickles load here too.
sys.modules["__main__"].Example = Example


def load_examples(path: str) -> list:
    import pickle
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["data"]


def block_split(examples: list, holdout_frac: float, block: int, seed: int):
    """Assign contiguous blocks to train/holdout with a seeded RNG."""
    n_blocks = (len(examples) + block - 1) // block
    rng = np.random.default_rng(seed)
    holdout_blocks = set(
        rng.choice(n_blocks, size=max(1, int(round(n_blocks * holdout_frac))),
                   replace=False).tolist())
    train, hold = [], []
    for b in range(n_blocks):
        chunk = examples[b * block:(b + 1) * block]
        (hold if b in holdout_blocks else train).extend(chunk)
    return train, hold


def densify_eval(examples: list, device: str, batch: int = 512):
    """Yield deterministic, UN-augmented eval batches (same tensor contract as
    ReplayBuffer.sample_batch)."""
    for i in range(0, len(examples), batch):
        chunk = examples[i:i + batch]
        mbs, obs, flats, pols, masks = [], [], [], [], []
        own_ss, opp_ss, win_ts = [], [], []
        for ex in chunk:
            policy = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)
            policy[ex.policy_idx] = ex.policy_val
            mask = np.zeros(NUM_JOINT_ACTIONS, dtype=bool)
            mask[ex.legal_idx] = True
            mbs.append(ex.my_board.astype(np.float32))
            obs.append(ex.opp_board.astype(np.float32))
            flats.append(ex.flat.astype(np.float32))
            pols.append(policy)
            masks.append(mask)
            own_ss.append(float(ex.own_score))
            opp_ss.append(float(ex.opp_score))
            win_ts.append(float(ex.win_target))
        to = lambda a: torch.from_numpy(np.stack(a)).to(device)
        yield (to(mbs).float(), to(obs).float(), to(flats).float(),
               to(pols).float(), to(np.stack(masks)),
               torch.tensor(own_ss, dtype=torch.float32, device=device),
               torch.tensor(opp_ss, dtype=torch.float32, device=device),
               torch.tensor(win_ts, dtype=torch.float32, device=device))


@torch.no_grad()
def evaluate(net, examples, device, score_scale: float,
             lambda_score: float, lambda_w: float):
    """Held-out metrics: policy CE, score MSE, win BCE + Brier, per-phase CE."""
    net.eval()
    prog_idx = FLAT_LAYOUT["game_progress"].start
    tot = {"n": 0, "policy_ce": 0.0, "own_mse": 0.0, "opp_mse": 0.0,
           "win_bce": 0.0, "win_brier": 0.0}
    phase = {p: [0.0, 0] for p in ("early", "mid", "end")}  # [ce_sum, n]
    for (mb, ob, flat, pol, mask, own_t, opp_t, win_t) in densify_eval(
            examples, device):
        n = mb.shape[0]
        own_p, opp_p, win_p, logits = net(mb, ob, flat)
        logp = masked_log_softmax(logits, mask)
        logp = torch.where(mask, logp, torch.zeros_like(logp))
        ce_rows = -(pol * logp).sum(dim=1)                       # (B,)
        tot["policy_ce"] += float(ce_rows.sum())
        tot["own_mse"] += float(F.mse_loss(own_p, own_t / score_scale,
                                           reduction="sum"))
        tot["opp_mse"] += float(F.mse_loss(opp_p, opp_t / score_scale,
                                           reduction="sum"))
        win_p = win_p.clamp(1e-6, 1 - 1e-6)
        tot["win_bce"] += float(F.binary_cross_entropy(
            win_p, win_t, reduction="sum"))
        tot["win_brier"] += float(((win_p - win_t) ** 2).sum())
        tot["n"] += n
        prog = flat[:, prog_idx]
        for name, lo, hi in (("early", -1.0, 1 / 3), ("mid", 1 / 3, 2 / 3),
                             ("end", 2 / 3, 10.0)):
            sel = (prog >= lo) & (prog < hi)
            if sel.any():
                phase[name][0] += float(ce_rows[sel].sum())
                phase[name][1] += int(sel.sum())
    n = max(1, tot["n"])
    out = {k: v / n for k, v in tot.items() if k != "n"}
    out["combined"] = (out["policy_ce"]
                       + lambda_score * (out["own_mse"] + out["opp_mse"])
                       + lambda_w * out["win_bce"])
    for p, (s, c) in phase.items():
        out[f"policy_ce_{p}"] = s / max(1, c)
    net.train()
    return out


def run_arm(arm: str, train_buf: ReplayBuffer, holdout: list,
            args) -> dict:
    channels, blocks = (int(x) for x in arm.split("x"))
    net = KingdominoNet(channels=channels, blocks=blocks,
                        bilinear_dim=args.bilinear_dim,
                        score_scale=args.score_scale).to(args.device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"\n=== arm {arm}: {n_params/1e6:.2f}M params ===", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + channels * 1000 + blocks)

    best = None
    best_state = None
    best_step = 0
    evals_since_best = 0
    lr_drops = 0
    t0 = time.time()
    step = 0
    while step < args.max_steps:
        for _ in range(args.eval_every):
            batch = train_buf.sample_batch(args.batch, rng, device=args.device,
                                           augment_d4=True)
            train_step(net, batch, opt,
                       policy_weight=1.0, lambda_score=args.lambda_score,
                       lambda_w=args.lambda_w, score_scale=args.score_scale,
                       grad_clip=args.grad_clip)
            step += 1
        m = evaluate(net, holdout, args.device, args.score_scale,
                     args.lambda_score, args.lambda_w)
        improved = best is None or m["combined"] < best["combined"] - 1e-4
        if improved:
            best, best_step, evals_since_best = m, step, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
        else:
            evals_since_best += 1
        lr_now = opt.param_groups[0]["lr"]
        print(f"[{arm}] step {step}: held-out ce={m['policy_ce']:.4f} "
              f"brier={m['win_brier']:.4f} combined={m['combined']:.4f} "
              f"{'*' if improved else f'(stale {evals_since_best})'} "
              f"lr={lr_now:.1e} ({(time.time()-t0)/60:.1f} min)", flush=True)
        if evals_since_best >= args.patience:
            if lr_drops < 2:
                lr_drops += 1
                evals_since_best = 0
                for g in opt.param_groups:
                    g["lr"] *= 0.3
                net.load_state_dict({k: v.to(args.device)
                                     for k, v in best_state.items()})
                print(f"[{arm}] plateau: lr -> {opt.param_groups[0]['lr']:.1e}, "
                      f"rewound to best", flush=True)
            else:
                print(f"[{arm}] early stop at step {step}", flush=True)
                break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"bakeoff_{arm}.pt"
    torch.save({
        "model_state": best_state,
        "config": {"channels": channels, "blocks": blocks,
                   "bilinear_dim": args.bilinear_dim,
                   "score_scale": args.score_scale},
        "bakeoff": {"arm": arm, "best_step": best_step,
                    "buffer": args.buffer, "seed": args.seed},
    }, ckpt_path)
    result = {"arm": arm, "params_m": n_params / 1e6, "best_step": best_step,
              "elapsed_min": (time.time() - t0) / 60.0,
              "checkpoint": str(ckpt_path), **best}
    print(f"RESULT {arm}: ce={best['policy_ce']:.4f} "
          f"brier={best['win_brier']:.4f} combined={best['combined']:.4f} "
          f"phases=[{best['policy_ce_early']:.4f}/{best['policy_ce_mid']:.4f}"
          f"/{best['policy_ce_end']:.4f}] @step {best_step}", flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--buffer", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--arms", default="80x6,96x6,80x10",
                   help="comma-separated CHANNELSxBLOCKS")
    p.add_argument("--holdout_frac", type=float, default=0.10)
    p.add_argument("--split_block", type=int, default=400)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_score", type=float, default=0.5)
    p.add_argument("--lambda_w", type=float, default=0.25)
    p.add_argument("--score_scale", type=float, default=160.0)
    p.add_argument("--bilinear_dim", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=40000)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--patience", type=int, default=4,
                   help="evals without held-out improvement before an lr drop "
                        "(x0.3, rewind to best); stops after 2 drops + patience")
    p.add_argument("--sample_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260709)
    args = p.parse_args()

    examples = load_examples(args.buffer)
    train, hold = block_split(examples, args.holdout_frac, args.split_block,
                              args.seed)
    print(f"buffer {args.buffer}: {len(examples)} examples -> "
          f"train {len(train)} / holdout {len(hold)} "
          f"(block={args.split_block})", flush=True)

    train_buf = ReplayBuffer(len(train), n_sample_workers=args.sample_workers)
    train_buf.data = train

    results = [run_arm(arm.strip(), train_buf, hold, args)
               for arm in args.arms.split(",") if arm.strip()]

    out = Path(args.out_dir) / "bakeoff_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nheld-out summary (lower is better):", flush=True)
    for r in sorted(results, key=lambda r: r["combined"]):
        print(f"  {r['arm']:>7s} ({r['params_m']:.2f}M): "
              f"ce={r['policy_ce']:.4f} brier={r['win_brier']:.4f} "
              f"combined={r['combined']:.4f}", flush=True)
    print("BAKEOFF DONE", flush=True)


if __name__ == "__main__":
    main()
