from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from games.kingdomino import action_codec, encoder
from games.kingdomino.dominoes import DOMINOES
from games.kingdomino.network import KingdominoNet


EXPECTED_CHECKPOINT_VERSION = 2
EXPECTED_FLAT_SIZE = 261
EXPECTED_BOARD_SHAPE = (9, 13, 13)
EXPECTED_CHANNELS = 48
EXPECTED_BLOCKS = 6
EXPECTED_VALUE_HEADS = ["own_score", "opp_score", "win_prob"]


def _state_dict_from_checkpoint(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model_state", "state_dict", "model"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
        if ckpt and all(isinstance(k, str) for k in ckpt.keys()):
            if any(k.endswith(".weight") or k.endswith(".bias") for k in ckpt.keys()):
                return ckpt
    raise ValueError(
        "Checkpoint does not contain a recognizable state dict "
        "(expected model_state/state_dict/model or a raw state_dict)."
    )


def _infer_arch_from_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, Any]:
    try:
        channels = int(sd["stem.0.weight"].shape[0])
        blocks = 1 + max(
            int(k.split(".")[1])
            for k in sd
            if k.startswith("res_blocks.") and k.endswith(".conv1.weight")
        )
        bilinear_dim = int(sd["W"].shape[0])
        value_hidden = int(sd["own_score_mlp.0.weight"].shape[0])
        pick_hidden = int(sd["pick_mlp.0.weight"].shape[0])
        flat_policy_hidden = int(sd["flat_policy_mlp.0.weight"].shape[0])
    except Exception as exc:
        raise ValueError(f"Could not infer architecture from state_dict: {exc}") from exc

    has_batch_norm = any(k.endswith("running_mean") for k in sd)
    return {
        "channels": channels,
        "blocks": blocks,
        "bilinear_dim": bilinear_dim,
        "value_hidden": value_hidden,
        "pick_hidden": pick_hidden,
        "flat_policy_hidden": flat_policy_hidden,
        "norm": "batch" if has_batch_norm else "group",
    }


def _model_kwargs(ckpt: Any, sd: dict[str, torch.Tensor]) -> dict[str, Any]:
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    inferred = _infer_arch_from_state_dict(sd)
    out = dict(inferred)
    for key in (
        "channels",
        "blocks",
        "bilinear_dim",
        "value_hidden",
        "pick_hidden",
        "flat_policy_hidden",
        "norm",
        "score_scale",
    ):
        if key in cfg:
            out[key] = cfg[key]
    return out


def ruleset_hash() -> str:
    payload = [
        {
            "id": int(did),
            "a": {
                "terrain": domino.a.terrain.name,
                "crowns": int(domino.a.crowns),
            },
            "b": {
                "terrain": domino.b.terrain.name,
                "crowns": int(domino.b.crowns),
            },
        }
        for did, domino in sorted(DOMINOES.items())
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:8]


def _check(name: str, got: Any, expected: Any) -> None:
    if got != expected:
        raise AssertionError(f"{name}: got {got!r}, expected {expected!r}")


def verify_contract(checkpoint: Path) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    sd = _state_dict_from_checkpoint(ckpt)
    kwargs = _model_kwargs(ckpt, sd)

    checkpoint_version = (
        ckpt.get("checkpoint_version", ckpt.get("config", {}).get("checkpoint_version", 2))
        if isinstance(ckpt, dict)
        else 2
    )

    model = KingdominoNet(**kwargs)
    model.load_state_dict(sd)
    model.eval()

    with torch.inference_mode():
        my_board = torch.zeros((1, *EXPECTED_BOARD_SHAPE), dtype=torch.float32)
        opp_board = torch.zeros((1, *EXPECTED_BOARD_SHAPE), dtype=torch.float32)
        flat = torch.zeros((1, encoder.FLAT_SIZE), dtype=torch.float32)
        own, opp, win, policy = model(my_board, opp_board, flat)

    contract = {
        "checkpoint_version": int(checkpoint_version),
        "FLAT_SIZE": int(encoder.FLAT_SIZE),
        "NUM_JOINT_ACTIONS": int(action_codec.NUM_JOINT_ACTIONS),
        "board_shape": (
            int(encoder.NUM_BOARD_CHANNELS),
            int(encoder.CANVAS_SIZE),
            int(encoder.CANVAS_SIZE),
        ),
        "channels": int(kwargs["channels"]),
        "blocks": int(kwargs["blocks"]),
        "value_heads": EXPECTED_VALUE_HEADS,
        "policy_head_size": int(policy.shape[-1]),
        "ruleset_hash": ruleset_hash(),
        "bilinear_dim": int(kwargs["bilinear_dim"]),
    }

    _check("checkpoint_version", contract["checkpoint_version"], EXPECTED_CHECKPOINT_VERSION)
    _check("FLAT_SIZE", contract["FLAT_SIZE"], EXPECTED_FLAT_SIZE)
    _check("board_shape", contract["board_shape"], EXPECTED_BOARD_SHAPE)
    _check("channels", contract["channels"], EXPECTED_CHANNELS)
    _check("blocks", contract["blocks"], EXPECTED_BLOCKS)
    _check("value_heads", contract["value_heads"], EXPECTED_VALUE_HEADS)
    _check("policy_head_size", contract["policy_head_size"], action_codec.NUM_JOINT_ACTIONS)
    _check("own_score_shape", tuple(own.shape), (1,))
    _check("opp_score_shape", tuple(opp.shape), (1,))
    _check("win_prob_shape", tuple(win.shape), (1,))
    if not (0.0 <= float(win.item()) <= 1.0):
        raise AssertionError(f"win_prob dummy forward outside [0,1]: {float(win.item())}")

    return contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a Kingdomino checkpoint contract.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        contract = verify_contract(args.checkpoint)
    except Exception as exc:
        print(f"CONTRACT FAILED: {exc}", file=sys.stderr)
        return 1

    for key in (
        "checkpoint_version",
        "FLAT_SIZE",
        "NUM_JOINT_ACTIONS",
        "board_shape",
        "channels",
        "blocks",
        "value_heads",
        "policy_head_size",
        "ruleset_hash",
    ):
        print(f"{key}: {contract[key]}")
    print("CONTRACT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
