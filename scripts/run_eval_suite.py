from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games.kingdomino.action_codec import NUM_JOINT_ACTIONS, encode_action
from games.kingdomino.network import KingdominoNet
from games.kingdomino.print_model_contract import ruleset_hash
from games.kingdomino.round_robin_eval import checkpoint_config, checkpoint_state_dict
from games.kingdomino.web_app import state_from_debug_json


DEFAULT_SUITE = Path("data/kingdomino/eval_suite_v1.jsonl")


def load_suite(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            validate_record(rec, line_no)
            records.append(rec)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def validate_record(rec: dict[str, Any], line_no: int) -> None:
    required = {
        "position_id",
        "ruleset_hash",
        "public_state",
        "source",
        "phase",
        "tiles_remaining",
        "expected_exact_value",
        "notes",
    }
    missing = required - set(rec)
    if missing:
        raise ValueError(f"line {line_no}: missing keys {sorted(missing)}")
    if rec["ruleset_hash"] != ruleset_hash():
        raise ValueError(
            f"line {line_no}: ruleset_hash {rec['ruleset_hash']} != current {ruleset_hash()}"
        )
    state = state_from_debug_json(rec["public_state"])
    if len(state.deck) != int(rec["tiles_remaining"]):
        raise ValueError(f"line {line_no}: tiles_remaining mismatch")
    if state.phase.name == "GAME_OVER":
        raise ValueError(f"line {line_no}: eval suite cannot contain GAME_OVER roots")
    if not state.legal_actions():
        raise ValueError(f"line {line_no}: root has no legal actions")


def make_model(args) -> KingdominoNet:
    torch.manual_seed(int(args.seed))
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = checkpoint_config(ckpt)
        kwargs = {
            "channels": int(args.channels if args.channels is not None else cfg.get("channels", 96)),
            "blocks": int(args.blocks if args.blocks is not None else cfg.get("blocks", 8)),
            "bilinear_dim": int(args.bilinear_dim if args.bilinear_dim is not None else cfg.get("bilinear_dim", 64)),
            "score_scale": float(cfg.get("score_scale", args.score_scale)),
        }
        net = KingdominoNet(**kwargs)
        net.load_state_dict(checkpoint_state_dict(ckpt))
        return net
    return KingdominoNet(
        channels=int(args.channels or 8),
        blocks=int(args.blocks or 1),
        bilinear_dim=int(args.bilinear_dim or 16),
        score_scale=float(args.score_scale),
    )


def legal_policy_metrics(logits: np.ndarray, legal_idx: list[int]) -> tuple[int, float, float]:
    legal_logits = logits[np.asarray(legal_idx, dtype=np.int64)].astype(np.float64)
    legal_logits -= legal_logits.max()
    exp = np.exp(legal_logits)
    probs = exp / exp.sum()
    top_local = int(np.argmax(probs))
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return int(legal_idx[top_local]), float(probs[top_local]), entropy


def evaluate_record(rec: dict[str, Any], net: KingdominoNet, device: str, score_scale: float) -> dict[str, Any]:
    state = state_from_debug_json(rec["public_state"])
    actor = state.current_actor
    from games.kingdomino.encoder import encode_state

    mb_np, ob_np, flat_np = encode_state(state, actor)
    legal_idx = sorted({encode_action(a, state) for a in state.legal_actions()})
    with torch.inference_mode():
        mb_t = torch.from_numpy(mb_np).unsqueeze(0).to(device)
        ob_t = torch.from_numpy(ob_np).unsqueeze(0).to(device)
        flat_t = torch.from_numpy(flat_np).unsqueeze(0).to(device)
        own, opp, win, logits = net(mb_t, ob_t, flat_t)
    logits_np = logits.detach().cpu().numpy()[0]
    top_idx, top_prob, entropy = legal_policy_metrics(logits_np, legal_idx)
    own_raw = float(own.item() * score_scale)
    opp_raw = float(opp.item() * score_scale)
    win_prob = float(win.item())
    margin_value = math.tanh((float(own.item()) - float(opp.item())) * 2.0)
    leaf_value_actor = 0.8 * margin_value + 0.2 * (2.0 * win_prob - 1.0)
    leaf_value_p0 = leaf_value_actor if actor == 0 else -leaf_value_actor

    exact = rec["expected_exact_value"]
    exact_error = None if exact is None else float(leaf_value_p0 - float(exact))
    return {
        "position_id": rec["position_id"],
        "phase": rec["phase"],
        "source": rec["source"],
        "tiles_remaining": rec["tiles_remaining"],
        "actor": int(actor),
        "n_legal": len(legal_idx),
        "top_action_idx": top_idx,
        "top_action_prob": top_prob,
        "policy_entropy": entropy,
        "own_score_est": own_raw,
        "opp_score_est": opp_raw,
        "win_prob": win_prob,
        "leaf_value_player0": leaf_value_p0,
        "expected_exact_value": exact,
        "exact_value_error": exact_error,
    }


def summarize(rows: list[dict[str, Any]], suite: Path, checkpoint: str | None) -> dict[str, Any]:
    exact_errors = [
        abs(float(r["exact_value_error"]))
        for r in rows
        if r["exact_value_error"] is not None
    ]
    by_phase = {}
    for phase in sorted({r["phase"] for r in rows}):
        phase_rows = [r for r in rows if r["phase"] == phase]
        by_phase[phase] = {
            "count": len(phase_rows),
            "mean_policy_entropy": float(np.mean([r["policy_entropy"] for r in phase_rows])),
            "mean_top_action_prob": float(np.mean([r["top_action_prob"] for r in phase_rows])),
        }
    return {
        "suite": str(suite),
        "checkpoint": checkpoint or "random_init",
        "positions": len(rows),
        "ruleset_hash": ruleset_hash(),
        "mean_policy_entropy": float(np.mean([r["policy_entropy"] for r in rows])),
        "mean_top_action_prob": float(np.mean([r["top_action_prob"] for r in rows])),
        "exact_positions": len(exact_errors),
        "mean_abs_exact_value_error": None if not exact_errors else float(np.mean(exact_errors)),
        "by_phase": by_phase,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a checkpoint over Kingdomino eval suite.")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out", type=Path, default=Path("eval_results/eval_suite_v1_summary.json"))
    parser.add_argument("--details_out", type=Path, default=Path("eval_results/eval_suite_v1_details.jsonl"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--bilinear_dim", type=int, default=None)
    parser.add_argument("--score_scale", type=float, default=160.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    records = load_suite(args.suite)
    net = make_model(args).to(args.device)
    net.eval()
    score_scale = float(getattr(net, "score_scale", args.score_scale) or args.score_scale)
    rows = [evaluate_record(rec, net, args.device, score_scale) for rec in records]
    summary = summarize(rows, args.suite, args.checkpoint or None)
    write_json(args.out, summary)
    args.details_out.parent.mkdir(parents=True, exist_ok=True)
    with args.details_out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.selftest:
        if summary["positions"] != len(records):
            raise SystemExit("selftest failed: position count mismatch")
        if not all(0 <= r["top_action_idx"] < NUM_JOINT_ACTIONS for r in rows):
            raise SystemExit("selftest failed: top_action_idx out of range")
        if any(not np.isfinite(r["policy_entropy"]) for r in rows):
            raise SystemExit("selftest failed: non-finite entropy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
