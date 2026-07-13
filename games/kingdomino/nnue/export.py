"""Export a trained TwoHeadNNUE to a neutral, versioned format the Rust searcher
reads (Step 2b). Produces two files next to the checkpoint:

  <name>.knnue          compact binary: a fixed 40-byte little-endian header
                        (magic bytes "KNNU", format version, dims, margin_scale,
                        encoder signature) followed by the weight tensors as raw
                        little-endian f32 in a FIXED order. Rust reads this directly
                        — no JSON/serde dependency — and ENFORCES the magic, version,
                        encoder signature, dims, and tensor finiteness.
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
    CH_TERRAIN_START, CH_TERRAIN_END, CH_CROWNS, CH_CASTLE, CH_OCCUPIED,
)
from games.kingdomino.network import KingdominoNet

KNNUE_MAGIC = b"KNNU"       # exact on-disk bytes (header[0:4])
FORMAT_VERSION = 2
HEADER_FMT = "<4sIIIIIIfQ"  # magic, version, input_dim, acc_width, tail_hidden,
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # board_size, flat_size, margin_scale, sig(u64) = 44

# Tensor order in the binary blob (PyTorch Linear weight is (out, in), row-major).
TENSOR_ORDER = [
    "accumulator.weight", "accumulator.bias",
    "tail.0.weight", "tail.0.bias",
    "tail.2.weight", "tail.2.bias",
    "outcome_head.weight", "outcome_head.bias",
    "margin_head.weight", "margin_head.bias",
]


def encoder_signature() -> int:
    """64-bit hash of the encoder CONTRACT the net was trained against: board
    dimensions, the semantic board-channel ordering (CH_* boundaries), the flat
    layout (names + slice bounds), and the checkpoint/encoder migration version.
    Embedded in the .knnue header and enforced by Rust — so a reordered flat block,
    a remapped board channel, or an encoder migration all invalidate stale exports
    loudly instead of silently feeding the net wrong features. (It cannot catch a
    change that alters channel *meaning* while preserving all of the above; bump
    KingdominoNet.checkpoint_version for those.)"""
    parts = [
        f"board={NUM_BOARD_CHANNELS}x{CANVAS_SIZE}x{CANVAS_SIZE}",
        f"chan=terrain:{CH_TERRAIN_START}:{CH_TERRAIN_END},crowns:{CH_CROWNS},"
        f"castle:{CH_CASTLE},occupied:{CH_OCCUPIED}",
        f"flat={FLAT_SIZE}:" + ";".join(
            f"{k}:{v.start}:{v.stop}" for k, v in sorted(FLAT_LAYOUT.items())),
        f"ckpt_version={KingdominoNet.checkpoint_version}",
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return int(digest, 16)


def export_state(state_dict, config, margin_scale, out_base, provenance=None):
    """Write <out_base>.knnue + .manifest.json from raw weights. Returns the two
    paths. Rejects non-finite / non-positive metadata and non-finite tensors here,
    so a bad checkpoint never becomes a bad artifact."""
    margin_scale = float(margin_scale)
    if not np.isfinite(margin_scale) or margin_scale <= 0:
        raise ValueError(f"margin_scale must be finite and positive, got {margin_scale}")
    board_size = NUM_BOARD_CHANNELS * CANVAS_SIZE * CANVAS_SIZE
    if config["input_dim"] != 2 * board_size + FLAT_SIZE:
        raise ValueError("config input_dim does not match the encoder layout")

    out_base = Path(out_base)
    bin_path = out_base.with_suffix(".knnue")
    man_path = out_base.with_suffix(".manifest.json")
    sig = encoder_signature()

    header = struct.pack(
        HEADER_FMT, KNNUE_MAGIC, FORMAT_VERSION,
        config["input_dim"], config["acc_width"], config["tail_hidden"],
        board_size, FLAT_SIZE, margin_scale, sig,
    )
    tensors, blob = {}, bytearray()
    for name in TENSOR_ORDER:
        t = state_dict[name].detach().cpu().numpy().astype("<f4")
        if not np.isfinite(t).all():
            raise ValueError(f"tensor {name} contains non-finite values")
        tensors[name] = list(t.shape)
        blob += t.reshape(-1).tobytes()
    bin_path.write_bytes(header + bytes(blob))

    manifest = {
        "format": "knnue",
        "format_version": FORMAT_VERSION,
        "magic": KNNUE_MAGIC.decode(),
        "header": {
            "struct": HEADER_FMT,
            "fields": ["magic:4s", "version:u32", "input_dim:u32", "acc_width:u32",
                       "tail_hidden:u32", "board_size:u32", "flat_size:u32",
                       "margin_scale:f32", "encoder_sig:u64"],
            "byte_order": "little-endian",
            "size_bytes": HEADER_SIZE,
        },
        "dtype": "f32-le",
        "tensor_order": TENSOR_ORDER,
        "tensor_shapes": tensors,  # PyTorch Linear weight = (out, in), row-major
        "config": config,
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
            "channel_order": {"terrain": [CH_TERRAIN_START, CH_TERRAIN_END],
                              "crowns": CH_CROWNS, "castle": CH_CASTLE,
                              "occupied": CH_OCCUPIED},
            "feature_order": "[my_board | opp_board | flat], each board C-order (9,13,13)",
            "checkpoint_version": KingdominoNet.checkpoint_version,
            "signature_u64": sig,
        },
        "provenance": provenance or {},
    }
    man_path.write_text(json.dumps(manifest, indent=2))
    return bin_path, man_path


def export_checkpoint(ckpt_path, out_base=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base = Path(out_base) if out_base else Path(ckpt_path).with_suffix("")
    return export_state(
        ck["state_dict"], ck["config"], ck["margin_scale"], base,
        provenance={
            "checkpoint": str(ckpt_path),
            "val_metrics": ck.get("val_metrics"),
            "baselines": ck.get("baselines"),
            "train_args": ck.get("args"),
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="games/kingdomino/nnue/checkpoints/dense_v1.pt")
    ap.add_argument("--out", default=None, help="output basename (default: alongside ckpt)")
    args = ap.parse_args()
    bin_path, man_path = export_checkpoint(args.ckpt, args.out)
    n_floats = (bin_path.stat().st_size - HEADER_SIZE) // 4
    print(f"wrote {bin_path} ({bin_path.stat().st_size:,} bytes, {n_floats:,} f32) "
          f"+ {man_path.name}")
    print(f"  format v{FORMAT_VERSION}, encoder_sig=0x{encoder_signature():016x}")


if __name__ == "__main__":
    main()
