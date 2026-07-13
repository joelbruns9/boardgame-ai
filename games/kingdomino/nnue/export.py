"""Export a trained TwoHeadNNUE to a neutral, versioned format the Rust searcher
reads (Step 2b). Produces two files next to the checkpoint:

  <name>.knnue          compact binary: a fixed 32-byte little-endian header
                        (magic, format version, dims, margin_scale) followed by the
                        weight tensors as raw little-endian f32 in a FIXED order.
                        Rust reads this directly — no JSON/serde dependency.
  <name>.manifest.json  human-readable contract: format version, tensor
                        names/shapes/dtype/order, activation + output semantics,
                        margin scale, encoder layout + signature, and provenance.

Output semantics (frozen here, mirrored in the Rust eval):
  a  = relu(accumulator @ x + b)         x = [my_board | opp_board | flat] (actor frame)
  h  = relu(tail1 @ relu(tail0 @ a))
  outcome_logit = out @ h ;  expected_score = sigmoid(outcome_logit)  in [0,1]
                                            = P(win) + 0.5*P(draw)   (NOT P(win))
  actor_value  = 2*expected_score - 1    in [-1,1]
  p0_value     = actor_value if actor==P0 else -actor_value     (leaf value)
  margin_pred  = (margin_head @ h) * margin_scale               (auxiliary, points)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

from games.kingdomino.encoder import (
    FLAT_SIZE, FLAT_LAYOUT, NUM_BOARD_CHANNELS, CANVAS_SIZE,
)

KNNUE_MAGIC = 0x4B4E4E55  # "KNNU"
FORMAT_VERSION = 1

# Tensor order in the binary blob (PyTorch Linear weight is (out, in), row-major).
TENSOR_ORDER = [
    "accumulator.weight", "accumulator.bias",
    "tail.0.weight", "tail.0.bias",
    "tail.2.weight", "tail.2.bias",
    "outcome_head.weight", "outcome_head.bias",
    "margin_head.weight", "margin_head.bias",
]


def _encoder_signature() -> str:
    """Stable hash of the encoder layout, so a Rust/Python encoder mismatch is
    caught rather than silently feeding the net garbage features."""
    layout = f"{NUM_BOARD_CHANNELS}x{CANVAS_SIZE}x{CANVAS_SIZE}+flat{FLAT_SIZE};"
    layout += ";".join(f"{k}:{v.start}:{v.stop}" for k, v in sorted(FLAT_LAYOUT.items()))
    return hashlib.sha256(layout.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="games/kingdomino/nnue/checkpoints/dense_v1.pt")
    ap.add_argument("--out", default=None, help="output basename (default: alongside ckpt)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    cfg = ckpt["config"]
    margin_scale = float(ckpt["margin_scale"])
    board_size = NUM_BOARD_CHANNELS * CANVAS_SIZE * CANVAS_SIZE
    assert cfg["input_dim"] == 2 * board_size + FLAT_SIZE, "input_dim / encoder mismatch"

    base = Path(args.out) if args.out else Path(args.ckpt).with_suffix("")
    bin_path = base.with_suffix(".knnue")
    man_path = base.with_suffix(".manifest.json")

    # ── binary: header + tensors ──
    header = struct.pack(
        "<IIIIIIIf",
        KNNUE_MAGIC, FORMAT_VERSION,
        cfg["input_dim"], cfg["acc_width"], cfg["tail_hidden"],
        board_size, FLAT_SIZE, margin_scale,
    )
    tensors = {}
    blob = bytearray()
    for name in TENSOR_ORDER:
        t = sd[name].detach().cpu().numpy().astype("<f4")  # little-endian f32
        tensors[name] = list(t.shape)
        blob += t.reshape(-1).tobytes()
    bin_path.write_bytes(bytes(header) + bytes(blob))

    # ── manifest: the human/tooling-readable contract ──
    manifest = {
        "format": "knnue",
        "format_version": FORMAT_VERSION,
        "magic": KNNUE_MAGIC,
        "header": {
            "fields": ["magic:u32", "version:u32", "input_dim:u32", "acc_width:u32",
                       "tail_hidden:u32", "board_size:u32", "flat_size:u32",
                       "margin_scale:f32"],
            "byte_order": "little-endian",
            "size_bytes": len(header),
        },
        "dtype": "f32-le",
        "tensor_order": TENSOR_ORDER,
        "tensor_shapes": tensors,  # PyTorch Linear weight = (out, in), row-major
        "config": cfg,
        "activation": "relu(accumulator) -> relu(tail0) -> relu(tail1); "
                      "sigmoid(outcome_head) = expected_score; identity(margin_head)",
        "output_semantics": {
            "expected_score": "sigmoid(outcome_logit) = P(win)+0.5*P(draw) in [0,1], actor frame",
            "leaf_value_p0": "2*expected_score-1, sign-flipped when actor != P0",
            "margin_pred_points": "margin_head_output * margin_scale, actor frame",
            "margin_scale": margin_scale,
        },
        "encoder": {
            "board_channels": NUM_BOARD_CHANNELS,
            "canvas": CANVAS_SIZE,
            "flat_size": FLAT_SIZE,
            "feature_order": "[my_board | opp_board | flat], each board C-order (9,13,13)",
            "signature": _encoder_signature(),
        },
        "provenance": {
            "checkpoint": str(args.ckpt),
            "val_metrics": ckpt.get("val_metrics"),
            "baselines": ckpt.get("baselines"),
            "train_args": ckpt.get("args"),
        },
    }
    man_path.write_text(json.dumps(manifest, indent=2))

    n_floats = (len(blob)) // 4
    print(f"wrote {bin_path} ({bin_path.stat().st_size:,} bytes, {n_floats:,} f32) "
          f"+ {man_path.name}")
    print(f"  input_dim={cfg['input_dim']} acc_width={cfg['acc_width']} "
          f"tail_hidden={cfg['tail_hidden']} margin_scale={margin_scale} "
          f"encoder_sig={manifest['encoder']['signature']}")


if __name__ == "__main__":
    main()
