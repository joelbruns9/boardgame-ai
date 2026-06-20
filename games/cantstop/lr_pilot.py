#!/usr/bin/env python
"""
lr_pilot.py — Learning-rate pilot for Can't Stop self-play.

WHY
---
Before committing a long weekend run, find a learning rate that actually
MOVES the network. The self_play default is 3e-5, which the code's own
comment flags as fine-tuning-style ("consider 1e-4 to 3e-4 if train loss
is flat across epochs"). A flat-loss LR wastes a weekend producing noise.

WHAT IT MEASURES
----------------
For each candidate LR, the pilot trains the SAME data from the SAME
starting model, then reports two things:
  (a) the loss / entropy / policy-match trajectory  (diagnostic), and
  (b) the win rate of the trained model vs its own starting point
      (the decision metric — the same currency the accept gate uses).

Win rate is what decides. Policy/value loss can fall while playing
strength stalls or regresses (overfitting the targets, value
miscalibration). Prefer the LR that WINS, not the one with the lowest loss.

FAIRNESS
--------
train_on_buffer's internal train/val split uses an UNSEEDED random.shuffle,
so a naive loop would train each LR on different data. The pilot re-seeds
random / numpy / torch immediately before each training call, so the split
and the per-epoch batch order are identical across LRs. Each LR starts from
a fresh clone of the same initial checkpoint. (Eval games are NOT seeded —
see the noise note in the summary; treat sub-~5% win-rate gaps as noise and
lean on the loss/entropy trends to corroborate.)

USAGE  (drop this file in games/cantstop/, run from the repo root)
------------------------------------------------------------------
  # generate a fresh pilot buffer and sweep the default LRs:
  python -m games.cantstop.lr_pilot \
      --model models/cantstop/best_model.pt \
      --games 2000 --sims 20 --epochs 5 \
      --eval-games 400 --eval-sims 20 \
      --lrs 3e-5 1e-4 3e-4 \
      --save-records pilot_buffer.pkl

  # re-sweep a tighter range on IDENTICAL data (no regeneration):
  python -m games.cantstop.lr_pilot \
      --model models/cantstop/best_model.pt \
      --records pilot_buffer.pkl \
      --lrs 1e-4 2e-4 3e-4 5e-4 --epochs 8

NOTES
-----
- Augmentation: the pilot calls train_on_buffer with augmentation at its
  default (on), so it faithfully includes your symmetry hook. If your tree
  made augment a *required* positional arg, add it to the train_on_buffer
  call below.
- This harness imports the production functions unchanged; it does not
  modify self_play.py.
"""

import os
import sys
import io
import re
import time
import pickle
import random
import argparse
import multiprocessing as mp

import numpy as np
import torch

# Make the games package importable for `python path/to/lr_pilot.py` too.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.model import CantStopNet
from games.cantstop.self_play import (
    generate_games_parallel,
    train_on_buffer,
    evaluate_networks,
    ReplayBuffer,
)
from games.cantstop.inference_server import InferenceServerManager


# Fixed seed → identical train/val split and batch order for every LR.
PILOT_SEED = 1234


# ---- per-epoch metric capture (non-invasive) ----

class _Tee:
    """Mirror writes to the real stdout AND capture them, so training
    progress is visible live and the per-epoch lines can be parsed after."""

    def __init__(self, real):
        self.real = real
        self.buf = io.StringIO()

    def write(self, s):
        self.real.write(s)
        self.buf.write(s)

    def flush(self):
        self.real.flush()

    def getvalue(self):
        return self.buf.getvalue()


# Matches train_on_buffer's exact per-epoch print format.
_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|\s+Train:\s+([\d.]+)\s+\|\s+"
    r"Val:\s+([\d.]+)\s+\|\s+Policy match \(vs MCTS argmax\):\s+([\d.]+)"
    r"\s+\|\s+Entropy:\s+([\d.]+)"
)


def _parse_epochs(captured):
    rows = []
    for m in _EPOCH_RE.finditer(captured):
        rows.append({
            'epoch':        int(m.group(1)),
            'train':        float(m.group(3)),
            'val':          float(m.group(4)),
            'policy_match': float(m.group(5)),
            'entropy':      float(m.group(6)),
        })
    return rows


# ---- helpers ----

def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fresh_model(model_state, device):
    m = CantStopNet().to(device)
    m.load_state_dict(model_state)
    return m


def _f(x, d=4):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else "N/A"


def _pct(x):
    return f"{x:.1%}" if isinstance(x, (int, float)) else "N/A"


def get_records(args, device):
    """Load a saved pilot buffer, or generate a fresh one via the same
    GPU inference-server path the real run uses."""
    if args.records and os.path.exists(args.records):
        print(f"  Loading buffer from {args.records} ...")
        with open(args.records, 'rb') as f:
            records = pickle.load(f)
        print(f"  Loaded {len(records):,} records.")
        return records

    print(f"  Generating {args.games:,} games @ {args.sims} sims "
          f"for the pilot buffer ...")
    server_device = device if device == 'cuda' else 'cpu'
    server = InferenceServerManager(
        model_path=args.model,
        device=server_device,
        num_workers=args.workers,
        mp_context=mp.get_context('spawn'),
    )
    server.start()
    time.sleep(2.0)
    if not server.is_alive():
        raise RuntimeError("Inference server failed to start. "
                           "Check the model path and CUDA availability.")
    try:
        # Match the real run's default Fix-C generation path.
        records = generate_games_parallel(
            server_manager=server,
            num_games=args.games,
            num_simulations=args.sims,
            temp_mult=args.temp,
            num_workers=args.workers,
            iteration_seed=PILOT_SEED,
            target_inflight=1,
            warmup_sims=0,
            games_per_worker=args.games_per_worker,
            batch_sync_searches=args.batch_sync_searches,
        )
    finally:
        server.stop()
    print(f"  Generated {len(records):,} records.")

    if args.save_records:
        with open(args.save_records, 'wb') as f:
            pickle.dump(records, f)
        print(f"  Saved buffer to {args.save_records} (reuse with --records).")
    return records


# ---- main pilot ----

def run_pilot(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.workers is None:
        args.workers = min(mp.cpu_count(), 8)

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LR PILOT")
    print(f"  model:  {args.model}")
    print(f"  device: {device} | workers: {args.workers}")
    print(f"  LRs:    {', '.join(f'{lr:g}' for lr in args.lrs)}")
    print(f"  epochs: {args.epochs} | eval: {args.eval_games} games "
          f"@ {args.eval_sims} sims vs initial")
    print(f"{'='*60}")

    # Every LR starts from an identical clone of this state.
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    init_state = ckpt['model_state']

    records = get_records(args, device)
    if not records:
        raise RuntimeError("No training records — generation produced nothing. "
                           "Try a larger --games.")

    results = []
    for lr in args.lrs:
        print(f"\n{'='*60}\n  LR = {lr:g}\n{'='*60}")

        # Identical data split + batch order for every LR.
        _seed_all(PILOT_SEED)

        buf = ReplayBuffer(max_size=len(records) * 3)  # headroom for augmentation
        buf.add(records)

        model = _fresh_model(init_state, device)

        # Capture train_on_buffer's per-epoch lines while still showing them.
        tee = _Tee(sys.stdout)
        old_stdout = sys.stdout
        sys.stdout = tee
        try:
            best_val = train_on_buffer(
                model, buf, device,
                epochs=args.epochs,
                lr=lr,
                sample_size=args.sample,   # None => full fixed buffer
            )
        finally:
            sys.stdout = old_stdout
        epochs = _parse_epochs(tee.getvalue())

        # The deciding metric: does this LR's model beat its starting point?
        win_rate = evaluate_networks(
            model,
            old_model_path=args.model,
            num_games=args.eval_games,
            eval_sims=args.eval_sims,
            num_workers=args.workers,
            output_dir=args.output,
        )

        last = epochs[-1] if epochs else {}
        results.append({
            'lr':           lr,
            'best_val':     best_val,
            'final_train':  last.get('train'),
            'final_val':    last.get('val'),
            'policy_match': last.get('policy_match'),
            'entropy':      last.get('entropy'),
            'win_rate':     win_rate,
        })

    _report(results, args)
    return results


def _report(results, args):
    data_desc = (f"loaded {args.records}" if args.records
                 else f"{args.games} generated games")
    print(f"\n{'='*78}")
    print(f"  LR PILOT SUMMARY")
    print(f"  data: {data_desc} | epochs: {args.epochs} | "
          f"eval: {args.eval_games} games @ {args.eval_sims} sims vs initial")
    print(f"{'='*78}")
    hdr = (f"  {'LR':>8} | {'fin.train':>9} | {'best_val':>8} | "
           f"{'pol.match':>9} | {'entropy':>7} | {'win% vs init':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        print(f"  {r['lr']:>8.0e} | "
              f"{_f(r['final_train']):>9} | "
              f"{_f(r['best_val']):>8} | "
              f"{_f(r['policy_match'], 3):>9} | "
              f"{_f(r['entropy'], 3):>7} | "
              f"{_pct(r['win_rate']):>12}")
    print(f"{'='*78}")

    valid = [r for r in results if isinstance(r['win_rate'], (int, float))]
    if valid:
        best = max(valid, key=lambda r: r['win_rate'])
        lo = min(r['lr'] for r in valid)
        hi = max(r['lr'] for r in valid)
        print(f"\n  Strongest LR by win rate vs initial: {best['lr']:.0e} "
              f"({best['win_rate']:.1%}).")
        if best['win_rate'] < 0.52:
            print("  ⚠️  No LR clearly beats the starting model (all within eval noise).")
            print("     Likely the pilot buffer is too small to learn from — bump --games")
            print("     before concluding LR doesn't matter. Or widen the LR span.")
        elif best['lr'] == lo:
            print("  → Lowest LR tested is strongest; the 3e-5 default may be fine, but")
            print("     test a LOWER value to confirm you're not already past the peak.")
        elif best['lr'] == hi:
            print("  → Highest LR tested is strongest; push HIGHER — you may not have")
            print("     hit the ceiling yet.")
        else:
            print("  → Interior optimum; this LR is a solid pick for the real run.")
    print("\n  Decision metric is win rate. A lower-loss LR with a flat/declining")
    print("  win rate is overfitting the targets, not improving play.\n")


if __name__ == "__main__":
    mp.freeze_support()  # Windows
    p = argparse.ArgumentParser(
        description="Learning-rate pilot for Can't Stop self-play"
    )
    p.add_argument("--model", required=True,
                   help="Initial checkpoint (the run's starting point)")
    p.add_argument("--output", default="lr_pilot_out",
                   help="Scratch dir for temp eval checkpoints")
    p.add_argument("--lrs", type=float, nargs="+", default=[3e-5, 1e-4, 3e-4],
                   help="Learning rates to sweep (space-separated)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--sample", type=int, default=None,
                   help="Cap records used per LR "
                        "(default: full buffer, fixed across LRs)")
    # Generation (only used when --records is not supplied)
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--sims", type=int, default=20)
    p.add_argument("--temp", type=float, default=1.0,
                   help="Temperature multiplier for pilot game generation")
    p.add_argument("--games-per-worker", type=int, default=2,
                   dest="games_per_worker")
    p.add_argument("--batch-sync-searches", type=int, default=8,
                   dest="batch_sync_searches")
    p.add_argument("--records", type=str, default=None,
                   help="Load a pre-generated records pickle instead of generating")
    p.add_argument("--save-records", type=str, default=None, dest="save_records",
                   help="Save the generated buffer here for reuse")
    # Eval
    p.add_argument("--eval-games", type=int, default=400, dest="eval_games")
    p.add_argument("--eval-sims", type=int, default=20, dest="eval_sims")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    run_pilot(args)