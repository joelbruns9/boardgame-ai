"""Average the weights of a range of checkpoints (SWA / model-soup).

Rationale: checkpoints from a converged net training at a fixed lr are samples
scattered around the loss-basin optimum (the "noise ball"). Their weight-space
mean sits nearer the basin center than any individual sample, so the averaged
net can be stronger than every checkpoint that went into it — strength
harvested without a single training step. BN running stats are averaged too,
which is safe when the source nets are near-identical (same basin); do NOT use
this across warm-start boundaries or architecture changes.

CPU-only and light (loads one checkpoint at a time); safe to run on a box
whose GPU is busy training.

Usage (average run5 iters 6..90 into an averaged checkpoint):
  python -m games.kingdomino.average_checkpoints \
      --dir runs/kingdomino/cloud_80x6_run5 --first 6 --last 90 \
      --out runs/kingdomino/cloud_80x6_run5/avg_0006_0090.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="Checkpoint directory (iter_XXXX.pt files)")
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--last", type=int, required=True)
    p.add_argument("--stride", type=int, default=1,
                   help="Use every Nth checkpoint (1 = all). The mean barely "
                        "changes above ~20 samples; stride to save I/O if desired.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ckpt_dir = Path(args.dir)
    paths = []
    for it in range(args.first, args.last + 1, args.stride):
        cp = ckpt_dir / f"iter_{it:04d}.pt"
        if cp.exists():
            paths.append(cp)
        else:
            print(f"  [skip] missing {cp.name}")
    if len(paths) < 2:
        raise SystemExit(f"Need >= 2 checkpoints, found {len(paths)}.")
    print(f"averaging {len(paths)} checkpoints: {paths[0].name} .. {paths[-1].name}")

    mean_state: dict[str, torch.Tensor] = {}
    template = None  # full checkpoint dict to carry config/metadata forward
    n = 0
    for cp in paths:
        ck = torch.load(cp, map_location="cpu")
        sd = ck["model_state"]
        n += 1
        if template is None:
            template = ck
            for k, v in sd.items():
                # float64 accumulator: 85 fp32 additions stay exact to the ulp.
                mean_state[k] = v.double() if v.is_floating_point() else v.clone()
        else:
            for k, v in sd.items():
                if v.is_floating_point():
                    # running mean: m += (x - m) / n
                    mean_state[k] += (v.double() - mean_state[k]) / n
                else:
                    # integer buffers (e.g. BN num_batches_tracked): keep last.
                    mean_state[k] = v.clone()

    out_state = {k: (v.float() if v.is_floating_point() else v)
                 for k, v in mean_state.items()}
    assert template is not None
    template["model_state"] = out_state
    template["averaged_from"] = {
        "dir": str(ckpt_dir), "first": args.first, "last": args.last,
        "stride": args.stride, "n_checkpoints": len(paths),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, out)
    print(f"wrote {out}  (config carried from {paths[-1].name}; "
          f"'averaged_from' metadata embedded)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
