"""Export a trained Step-3 sparse NNUE to the Rust inference format.

The accumulator table is already feature-major in PyTorch ``EmbeddingBag``
layout: ``(feature_count, accumulator_width)``.  Rows are therefore written
verbatim, making every Rust feature delta one contiguous vector add/subtract.
Auxiliary heads are training-only and intentionally omitted.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from .sparse_encoder import CORE_SIZE, core_schema_hash
from .summary_encoder import SUMMARY_SIZE, summary_schema_hash

MAGIC = b"KNSP"
FORMAT_VERSION = 3
HEADER_FMT = "<4sIIIIIfQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 44 bytes

TENSOR_ORDER = (
    "accumulator.weight",
    "accumulator_bias",
    "tail.0.weight",
    "tail.0.bias",
    "tail.2.weight",
    "tail.2.bias",
    "outcome_head.weight",
    "outcome_head.bias",
    "margin_head.weight",
    "margin_head.bias",
)


def _u64_hash(text: str) -> int:
    if len(text) != 16:
        raise ValueError(f"schema hash must be 16 hex characters, got {text!r}")
    return int(text, 16)


def _expected_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    nf = int(config["feature_count"])
    ns = int(config["summary_size"])
    aw = int(config["acc_width"])
    th = int(config["tail_hidden"])
    return {
        "accumulator.weight": (nf, aw),
        "accumulator_bias": (aw,),
        "tail.0.weight": (th, aw + ns),
        "tail.0.bias": (th,),
        "tail.2.weight": (th, th),
        "tail.2.bias": (th,),
        "outcome_head.weight": (1, th),
        "outcome_head.bias": (1,),
        "margin_head.weight": (1, th),
        "margin_head.bias": (1,),
    }


def export_state(
    state_dict,
    config: dict,
    margin_scale: float,
    out_base,
    *,
    checkpoint_hashes: dict,
    provenance: dict | None = None,
):
    margin_scale = float(margin_scale)
    if not np.isfinite(margin_scale) or margin_scale <= 0:
        raise ValueError("margin_scale must be finite and positive")
    if int(config["feature_count"]) != CORE_SIZE or int(config["summary_size"]) != SUMMARY_SIZE:
        raise ValueError("network dimensions do not match the frozen sparse encoder")
    expected_hashes = {
        "core_schema_hash": core_schema_hash(),
        "summary_schema_hash": summary_schema_hash(),
    }
    if checkpoint_hashes != expected_hashes:
        raise ValueError(f"checkpoint schema hashes {checkpoint_hashes} != {expected_hashes}")

    shapes = _expected_shapes(config)
    tensors: dict[str, np.ndarray] = {}
    blob = bytearray()
    for name in TENSOR_ORDER:
        if name not in state_dict:
            raise ValueError(f"checkpoint is missing inference tensor {name}")
        arr = state_dict[name].detach().cpu().numpy().astype("<f4")
        if tuple(arr.shape) != shapes[name]:
            raise ValueError(f"{name} shape {arr.shape} != {shapes[name]}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values")
        tensors[name] = arr
        blob += arr.reshape(-1).tobytes()

    out_base = Path(out_base)
    bin_path = out_base.with_suffix(".knnue")
    manifest_path = out_base.with_suffix(".manifest.json")
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        FORMAT_VERSION,
        int(config["feature_count"]),
        int(config["summary_size"]),
        int(config["acc_width"]),
        int(config["tail_hidden"]),
        margin_scale,
        _u64_hash(expected_hashes["core_schema_hash"]),
        _u64_hash(expected_hashes["summary_schema_hash"]),
    )
    bin_path.write_bytes(header + blob)

    manifest = {
        "format": "knnue-sparse",
        "format_version": FORMAT_VERSION,
        "magic": MAGIC.decode(),
        "header": {"struct": HEADER_FMT, "size_bytes": HEADER_SIZE, "byte_order": "little-endian"},
        "config": config,
        **expected_hashes,
        "margin_scale": margin_scale,
        "dtype": "f32-le",
        "tensor_order": list(TENSOR_ORDER),
        "tensor_shapes": {name: list(shapes[name]) for name in TENSOR_ORDER},
        "accumulator_layout": "feature-major: (feature_count, acc_width), contiguous row per feature",
        "activation": "relu(accumulator) + summary -> relu(tail0) -> relu(tail1)",
        "output_semantics": {
            "expected_score": "sigmoid(outcome_logit), actor frame",
            "leaf_value_p0": "2*expected_score-1, sign-flipped for actor P1",
            "margin_points": "margin_head * margin_scale, actor frame",
        },
        "omitted_training_heads": ["aux_score_head", "aux_bonus_head"],
        "provenance": provenance or {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return bin_path, manifest_path


def export_checkpoint(checkpoint, out_base=None):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = Path(out_base) if out_base else Path(checkpoint).with_suffix("")
    hashes = {
        "core_schema_hash": ck.get("core_schema_hash"),
        "summary_schema_hash": ck.get("summary_schema_hash"),
    }
    return export_state(
        ck["state_dict"],
        ck["config"],
        ck["margin_scale"],
        base,
        checkpoint_hashes=hashes,
        provenance={
            "checkpoint": str(checkpoint),
            "val_metrics": ck.get("val_metrics"),
            "baselines": ck.get("baselines"),
            "data_provenance": ck.get("data_provenance"),
            "train_args": ck.get("args"),
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/kingdomino/nnue_data/sparse_v3_pilot.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    binary, manifest = export_checkpoint(args.ckpt, args.out)
    print(f"wrote {binary} ({binary.stat().st_size:,} bytes) + {manifest.name}")


if __name__ == "__main__":
    main()
