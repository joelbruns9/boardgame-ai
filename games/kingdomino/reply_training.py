"""Equal-step control/treatment fine-tuning for opponent-reply labels.

This is an isolated pilot runner.  It never writes ``current_best`` and never
invokes promotion.  Both arms consume the exact same ordinary replay batches;
the treatment alone receives the grouped-pick reply loss.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from games.kingdomino.network import masked_log_softmax
from games.kingdomino.promotion import DEFAULT_CURRENT_BEST, sha256_file
from games.kingdomino.reply_pilot import (
    DEFAULT_MERGED,
    _read_jsonl,
    decode_array_blob,
    validate_reply_example,
)
from games.kingdomino.denial_search import load_checkpoint_network
from games.kingdomino.self_play import ReplayBuffer, train_step


DEFAULT_DIR = Path("runs/kingdomino/reply_pilot/training")


@dataclass(frozen=True)
class ReplyExample:
    example_id: str
    my_board: np.ndarray
    opp_board: np.ndarray
    flat: np.ndarray
    legal_indices: np.ndarray
    group_indices: tuple[np.ndarray, ...]
    target: np.ndarray
    baseline_conditionals: tuple[np.ndarray, ...]


class ReplyDataset:
    def __init__(self, path: str | Path, *, accepted_only: bool = True):
        self.path = Path(path)
        rows = _read_jsonl(self.path)
        examples = []
        for row in rows:
            validate_reply_example(row)
            if accepted_only and not bool(row["quality_accept"]):
                continue
            legal_indices = np.asarray(
                [int(item["action_idx"]) for item in row["legal_actions"]], dtype=np.int64)
            group_indices = []
            baseline_conditionals = []
            for pick_row in row["per_pick"]:
                conditional = pick_row["baseline_conditional_placements"]
                group_indices.append(np.asarray(
                    [int(item["action_idx"]) for item in conditional], dtype=np.int64))
                baseline_conditionals.append(np.asarray(
                    [float(item["conditional_probability"]) for item in conditional],
                    dtype=np.float64))
            examples.append(ReplyExample(
                example_id=str(row["example_id"]),
                my_board=decode_array_blob(row["encoded_state"]["my_board"]),
                opp_board=decode_array_blob(row["encoded_state"]["opp_board"]),
                flat=decode_array_blob(row["encoded_state"]["flat"]),
                legal_indices=legal_indices,
                group_indices=tuple(group_indices),
                target=np.asarray(row["denial_policy_target"], dtype=np.float32),
                baseline_conditionals=tuple(baseline_conditionals),
            ))
        if not examples:
            raise ValueError(f"reply dataset has no usable examples: {self.path}")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def sample(self, batch_size: int, rng: np.random.Generator, device: str):
        indices = rng.integers(0, len(self.examples), size=int(batch_size))
        rows = [self.examples[int(index)] for index in indices]
        batch = {
            "my_board": torch.from_numpy(np.stack([row.my_board for row in rows])).to(
                device=device, dtype=torch.float32),
            "opp_board": torch.from_numpy(np.stack([row.opp_board for row in rows])).to(
                device=device, dtype=torch.float32),
            "flat": torch.from_numpy(np.stack([row.flat for row in rows])).to(
                device=device, dtype=torch.float32),
            "legal_indices": [torch.as_tensor(row.legal_indices, device=device)
                              for row in rows],
            "group_indices": [
                [torch.as_tensor(group, device=device) for group in row.group_indices]
                for row in rows
            ],
            "target": [torch.as_tensor(row.target, device=device) for row in rows],
            "baseline_conditionals": [row.baseline_conditionals for row in rows],
            "example_ids": [row.example_id for row in rows],
        }
        return batch, [int(index) for index in indices]


def grouped_reply_loss(logits: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    """Cross-entropy over summed complete-action probability by pick group."""
    if logits.ndim != 2 or logits.shape[0] != len(batch["group_indices"]):
        raise ValueError("reply logits and batch size are not aligned")
    losses = []
    for row_index, (legal, groups, target) in enumerate(zip(
        batch["legal_indices"], batch["group_indices"], batch["target"]
    )):
        if len(groups) != int(target.numel()) or not groups:
            raise ValueError("reply groups and target are not aligned")
        legal_log_z = torch.logsumexp(logits[row_index].index_select(0, legal), dim=0)
        group_logp = torch.stack([
            torch.logsumexp(logits[row_index].index_select(0, group), dim=0) - legal_log_z
            for group in groups
        ])
        losses.append(-(target * group_logp).sum())
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite grouped reply loss")
    return loss


def placement_drift(logits: torch.Tensor, batch: dict[str, Any]) -> dict[str, float]:
    """Within-pick entropy and KL(q_current || q_generation_baseline)."""
    entropies = []
    kls = []
    with torch.no_grad():
        for row_index, (groups, baselines) in enumerate(zip(
            batch["group_indices"], batch["baseline_conditionals"]
        )):
            for group, baseline in zip(groups, baselines):
                log_q = torch.log_softmax(logits[row_index].index_select(0, group), dim=0)
                q = torch.exp(log_q).double().cpu().numpy()
                baseline = np.asarray(baseline, dtype=np.float64)
                baseline = np.maximum(baseline, 1e-300)
                entropies.append(float(-(q * np.log(np.maximum(q, 1e-300))).sum()))
                kls.append(float((q * (np.log(np.maximum(q, 1e-300)) - np.log(baseline))).sum()))

    def percentile(values: list[float], q: float) -> float:
        return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0

    return {
        "within_group_entropy_median": percentile(entropies, 50),
        "within_group_entropy_p90": percentile(entropies, 90),
        "kl_to_baseline_median": percentile(kls, 50),
        "kl_to_baseline_p90": percentile(kls, 90),
        "placement_groups": len(entropies),
    }


def _ordinary_losses(net, batch, *, policy_weight: float, lambda_score: float,
                     lambda_w: float, score_scale: float):
    mb, ob, flat, policy, legal_mask, _z, own_t, opp_t, win_t = batch
    if not legal_mask.any(dim=1).all():
        raise ValueError("ordinary batch contains a row with no legal actions")
    if not torch.allclose(policy.sum(dim=1), torch.ones(
        policy.shape[0], device=policy.device), atol=1e-4
    ):
        raise ValueError("ordinary policy target row does not sum to one")
    own_pred, opp_pred, win_prob, logits = net(mb, ob, flat)
    own_loss = F.mse_loss(own_pred, own_t / score_scale)
    opp_loss = F.mse_loss(opp_pred, opp_t / score_scale)
    win_loss = F.binary_cross_entropy(win_prob, win_t)
    logp = masked_log_softmax(logits, legal_mask)
    logp = torch.where(legal_mask, logp, torch.zeros_like(logp))
    policy_loss = -(policy * logp).sum(dim=1).mean()
    total = (policy_weight * policy_loss
             + lambda_score * (own_loss + opp_loss)
             + lambda_w * win_loss)
    return total, policy_loss, own_loss, opp_loss, win_loss


def treatment_train_step(
    net, ordinary_batch, reply_batch, optimizer, *, lambda_reply: float,
    policy_weight: float = 1.0, lambda_score: float = 0.5,
    lambda_w: float = 0.25, score_scale: float = 160.0, grad_clip: float = 1.0,
) -> dict[str, float]:
    ordinary_total, policy_loss, own_loss, opp_loss, win_loss = _ordinary_losses(
        net, ordinary_batch, policy_weight=policy_weight,
        lambda_score=lambda_score, lambda_w=lambda_w, score_scale=score_scale)
    _own, _opp, _win, reply_logits = net(
        reply_batch["my_board"], reply_batch["opp_board"], reply_batch["flat"])
    reply_loss = grouped_reply_loss(reply_logits, reply_batch)
    total = ordinary_total + float(lambda_reply) * reply_loss
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite treatment loss")
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip))
    if not math.isfinite(grad_norm):
        raise FloatingPointError("non-finite treatment gradient norm")
    optimizer.step()
    return {
        "total_loss": float(total.item()),
        "policy_loss": float(policy_loss.item()),
        "own_loss": float(own_loss.item()),
        "opp_loss": float(opp_loss.item()),
        "win_loss": float(win_loss.item()),
        "reply_loss": float(reply_loss.item()),
        "grad_norm": grad_norm,
    }


def evaluate_reply(net, dataset: ReplyDataset, *, device: str,
                   batch_size: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    batch, _indices = dataset.sample(min(batch_size, len(dataset)), rng, device)
    was_training = net.training
    net.eval()
    with torch.no_grad():
        _own, _opp, _win, logits = net(
            batch["my_board"], batch["opp_board"], batch["flat"])
        loss = float(grouped_reply_loss(logits, batch).item())
        drift = placement_drift(logits, batch)
    net.train(was_training)
    return {"reply_loss": loss, **drift}


def evaluate_ordinary(net, batch, *, policy_weight: float, lambda_score: float,
                      lambda_w: float, score_scale: float) -> dict[str, float]:
    """Evaluate both arms on the exact same frozen ordinary-replay batch."""
    was_training = net.training
    net.eval()
    with torch.no_grad():
        total, policy, own, opp, win = _ordinary_losses(
            net, batch, policy_weight=policy_weight, lambda_score=lambda_score,
            lambda_w=lambda_w, score_scale=score_scale)
    net.train(was_training)
    return {
        "total_loss": float(total.item()),
        "policy_loss": float(policy.item()),
        "own_loss": float(own.item()),
        "opp_loss": float(opp.item()),
        "win_loss": float(win.item()),
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps < 1 or args.batch_size < 1 or not 0.0 < args.reply_fraction <= 1.0:
        raise ValueError("steps/batch_size must be positive and reply_fraction must be in (0,1]")
    if args.lambda_reply <= 0.0:
        raise ValueError("lambda_reply must be positive for the treatment arm")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256_file(args.checkpoint)

    sidecar = Path(args.reply_train).with_suffix(".manifest.json")
    if sidecar.exists():
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        expected = manifest.get("checkpoint_sha256")
        if expected and expected != checkpoint_sha:
            raise ValueError("reply labels were generated from a different checkpoint")

    train_dataset = ReplyDataset(args.reply_train)
    validation_dataset = ReplyDataset(args.reply_validation)
    buffer = ReplayBuffer(capacity=args.buffer_capacity,
                          n_sample_workers=args.sample_workers)
    buffer.load(args.replay_buffer)
    if not len(buffer):
        raise ValueError("ordinary replay buffer is empty")

    control, checkpoint_config = load_checkpoint_network(args.checkpoint, args.device)
    treatment, treatment_config = load_checkpoint_network(args.checkpoint, args.device)
    if checkpoint_config != treatment_config:
        raise AssertionError("control and treatment checkpoint configs differ")
    control.train(); treatment.train()
    control_optimizer = torch.optim.Adam(control.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay)
    treatment_optimizer = torch.optim.Adam(treatment.parameters(), lr=args.lr,
                                           weight_decay=args.weight_decay)
    ordinary_rng = np.random.default_rng(args.seed)
    reply_rng = np.random.default_rng(args.seed + 1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    ordinary_validation_rng = np.random.default_rng(args.seed + 3)
    ordinary_validation_batch = buffer.sample_batch(
        min(args.validation_batch_size, len(buffer)), ordinary_validation_rng,
        device=args.device, augment_d4=False)

    def evaluate_all(net):
        return {
            "reply": evaluate_reply(
                net, validation_dataset, device=args.device,
                batch_size=args.validation_batch_size, seed=args.seed + 2),
            "ordinary": evaluate_ordinary(
                net, ordinary_validation_batch, policy_weight=args.policy_weight,
                lambda_score=args.lambda_score, lambda_w=args.lambda_w,
                score_scale=args.score_scale),
        }

    before = {"control": evaluate_all(control), "treatment": evaluate_all(treatment)}
    history_path = output_dir / "training_steps.jsonl"
    history_path.unlink(missing_ok=True)
    reply_batch_size = max(1, round(args.batch_size * args.reply_fraction))
    started = time.perf_counter()
    for step in range(args.steps):
        ordinary_batch, ordinary_meta = buffer.sample_batch(
            args.batch_size, ordinary_rng, device=args.device,
            augment_d4=not args.no_augment, return_metadata=True)
        reply_batch, reply_indices = train_dataset.sample(
            reply_batch_size, reply_rng, args.device)
        control_metrics = train_step(
            control, ordinary_batch, control_optimizer,
            policy_weight=args.policy_weight, lambda_score=args.lambda_score,
            lambda_w=args.lambda_w, score_scale=args.score_scale,
            grad_clip=args.grad_clip,
        )
        treatment_metrics = treatment_train_step(
            treatment, ordinary_batch, reply_batch, treatment_optimizer,
            lambda_reply=args.lambda_reply, policy_weight=args.policy_weight,
            lambda_score=args.lambda_score, lambda_w=args.lambda_w,
            score_scale=args.score_scale, grad_clip=args.grad_clip,
        )
        control_names = (
            "policy_loss", "own_loss", "opp_loss", "win_loss",
            "win_brier", "baseline_brier",
        )
        with history_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({
                "step": step,
                "ordinary_sample": ordinary_meta,
                "reply_indices": reply_indices,
                "reply_example_ids": reply_batch["example_ids"],
                "control": dict(zip(control_names, control_metrics)),
                "treatment": treatment_metrics,
            }, sort_keys=True, separators=(",", ":")) + "\n")
    elapsed = time.perf_counter() - started

    after = {"control": evaluate_all(control), "treatment": evaluate_all(treatment)}
    common = {
        "base_checkpoint": str(args.checkpoint),
        "base_checkpoint_sha256": checkpoint_sha,
        "checkpoint_config": checkpoint_config,
        "pilot": {
            "steps": args.steps, "batch_size": args.batch_size,
            "reply_batch_size": reply_batch_size,
            "reply_fraction": args.reply_fraction,
            "lambda_reply": args.lambda_reply, "seed": args.seed,
            "lr": args.lr, "weight_decay": args.weight_decay,
        },
    }
    control_path = output_dir / "control.pt"
    treatment_path = output_dir / "treatment.pt"
    _atomic_torch_save({**common, "config": checkpoint_config,
                        "arm": "control", "model_state": control.state_dict()},
                       control_path)
    _atomic_torch_save({**common, "config": checkpoint_config,
                        "arm": "treatment", "model_state": treatment.state_dict()},
                       treatment_path)
    report = {
        **common,
        "status": "trained_not_promoted",
        "reply_train": str(args.reply_train),
        "reply_validation": str(args.reply_validation),
        "replay_buffer": str(args.replay_buffer),
        "reply_train_examples": len(train_dataset),
        "reply_validation_examples": len(validation_dataset),
        "before": before,
        "after": after,
        "elapsed_seconds": elapsed,
        "history": str(history_path),
        "history_sha256": sha256_file(history_path),
        "control_checkpoint": str(control_path),
        "control_sha256": sha256_file(control_path),
        "treatment_checkpoint": str(treatment_path),
        "treatment_sha256": sha256_file(treatment_path),
        "current_best_updated": False,
    }
    report_path = output_dir / "pilot_training_report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(report_path)
    buffer.close()
    return report


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CURRENT_BEST))
    parser.add_argument("--reply-train", default=str(DEFAULT_MERGED))
    parser.add_argument("--reply-validation", required=True)
    parser.add_argument("--replay-buffer", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reply-fraction", type=float, default=0.15)
    parser.add_argument("--lambda-reply", type=float, default=0.15)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=1_000_000)
    parser.add_argument("--sample-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20_260_719)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--lambda-score", type=float, default=0.5)
    parser.add_argument("--lambda-w", type=float, default=0.25)
    parser.add_argument("--score-scale", type=float, default=160.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report = run_pilot(args)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
