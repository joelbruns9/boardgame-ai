"""Optimise Welcome To networks from searched S2 trajectories.

S0's trainer deliberately remains a GreedyBot pipeline shakedown.  This module
is the S2 boundary: it splits complete searched games, replays their MCTS visit
targets, evaluates those distributions, performs AdamW updates, and writes a
resume-capable checkpoint.  A caller may supply the S0 network or omit it and
bootstrap AlphaZero from random weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from games.welcome_to import network as nw
from games.welcome_to import self_play
from games.welcome_to import train as s0_train
from games.welcome_to import training


CHECKPOINT_FORMAT = "welcome_to_s2"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class S2TrainConfig:
    val_fraction: float = 0.10
    epochs: int = 4
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    shuffle_buffer: int = 8192
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        if self.epochs <= 0 or self.batch_size <= 0 or self.shuffle_buffer <= 0:
            raise ValueError("epochs, batch_size, and shuffle_buffer must be positive")
        for name, value in (
            ("lr", self.lr),
            ("weight_decay", self.weight_decay),
            ("grad_clip", self.grad_clip),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.lr == 0.0 or self.grad_clip == 0.0:
            raise ValueError("lr and grad_clip must be positive")


def _json_safe(value: Any) -> Any:
    """Replace undefined numeric diagnostics with strict-JSON ``null``."""
    if isinstance(value, Mapping):
        return {str(name): _json_safe(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def split_trajectories(
    trajectories: Sequence[self_play.SelfPlayTrajectory],
    val_fraction: float,
    seed: int,
) -> tuple[list[self_play.SelfPlayTrajectory], list[self_play.SelfPlayTrajectory]]:
    """Split by whole game so adjacent states never leak across the boundary."""
    if len(trajectories) < 2:
        raise ValueError("S2 training needs at least two complete trajectories")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    order = list(trajectories)
    random.Random(seed).shuffle(order)
    val_games = min(len(order) - 1, max(1, round(len(order) * val_fraction)))
    return order[val_games:], order[:val_games]


def _masked_rank(out: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]):
    mask = torch.stack(
        [batch[f"rank_mask_{rank}"] for rank in range(training.MAX_RANKS)], dim=-1
    )
    target = torch.stack(
        [batch[f"rank_p_{rank}"] for rank in range(training.MAX_RANKS)], dim=-1
    )
    log_policy = nw.masked_log_softmax(out["rank_logits"], mask)
    return mask, target, log_policy


@torch.no_grad()
def evaluate(
    net: nw.WelcomeToNet,
    trajectories: Sequence[self_play.SelfPlayTrajectory],
    device: torch.device | str,
    batch_size: int = 512,
) -> dict[str, float]:
    """Evaluate S2 against visit distributions and terminal outcomes.

    The sampled action is reported only as a diagnostic.  It is not the policy
    label: changing the self-play temperature changes that draw without changing
    the MCTS evidence.  ``policy_visit_best`` is tie-aware and therefore avoids
    repeating S0's arbitrary-argmax evaluation defect.
    """
    if not trajectories:
        raise ValueError("S2 evaluation needs at least one trajectory")
    torch_device = torch.device(device)
    net.eval()
    samples = 0
    policy_ce = 0.0
    target_entropy = 0.0
    predicted_entropy = 0.0
    visit_best = 0
    sampled_action_top1 = 0
    rank_ce = 0.0
    rank_entropy = 0.0
    rank_best = 0
    sq_err: dict[str, float] = {}
    sums: dict[str, float] = {}
    sq_sums: dict[str, float] = {}
    supports: dict[str, float] = {}

    rng = random.Random(0)
    for raw in self_play.iter_batches(
        trajectories,
        batch_size,
        rng,
        shuffle_buffer=batch_size * 2,
    ):
        batch = nw.to_tensors(raw, torch_device)
        out = net(
            batch["sheet_planes"],
            batch["sheet_scalars"],
            batch["viewer_plane"],
            batch["global_scalars"],
        )
        rows = int(batch["policy"].shape[0])
        samples += rows

        logits = out["policy_logits"].masked_fill(batch["legal"] <= 0, -1e9)
        log_policy = torch.log_softmax(logits, dim=-1)
        predicted = log_policy.exp()
        target = batch["policy"]
        row_ce = -(target * log_policy).sum(-1)
        row_target_entropy = -(
            target * target.clamp_min(torch.finfo(target.dtype).tiny).log()
        ).sum(-1)
        row_predicted_entropy = -(predicted * log_policy).sum(-1)
        policy_ce += float(row_ce.sum())
        target_entropy += float(row_target_entropy.sum())
        predicted_entropy += float(row_predicted_entropy.sum())

        predicted_action = logits.argmax(-1)
        target_at_prediction = target.gather(1, predicted_action[:, None]).squeeze(1)
        visit_best += int((target_at_prediction == target.max(-1).values).sum())
        sampled_action_top1 += int((predicted_action == batch["action"]).sum())

        _, rank_target, rank_log_policy = _masked_rank(out, batch)
        row_rank_ce = -(rank_target * rank_log_policy).sum(-1)
        row_rank_entropy = -(
            rank_target
            * rank_target.clamp_min(torch.finfo(rank_target.dtype).tiny).log()
        ).sum(-1)
        rank_ce += float(row_rank_ce.sum())
        rank_entropy += float(row_rank_entropy.sum())
        predicted_rank = rank_log_policy.argmax(-1)
        rank_at_prediction = rank_target.gather(
            1, predicted_rank[:, None]
        ).squeeze(1)
        rank_best += int((rank_at_prediction == rank_target.max(-1).values).sum())

        for name in nw.PER_SEAT_HEAD_TARGETS + nw.GLOBAL_HEAD_TARGETS:
            target_value, prediction = batch[name], out[name]
            if name in nw.PER_SEAT_HEAD_TARGETS:
                mask = batch["seat_valid"]
                mask_name = training.MASKED_TARGETS.get(name)
                if mask_name is not None:
                    mask = mask * batch[mask_name]
            else:
                mask = torch.ones_like(target_value)
            sq_err[name] = sq_err.get(name, 0.0) + float(
                (mask * (prediction - target_value) ** 2).sum()
            )
            sums[name] = sums.get(name, 0.0) + float((mask * target_value).sum())
            sq_sums[name] = sq_sums.get(name, 0.0) + float(
                (mask * target_value**2).sum()
            )
            supports[name] = supports.get(name, 0.0) + float(mask.sum())

    if samples == 0:
        raise ValueError("S2 held-out games produced no searched learner roots")
    mean_policy_ce = policy_ce / samples
    mean_target_entropy = target_entropy / samples
    mean_rank_ce = rank_ce / samples
    mean_rank_entropy = rank_entropy / samples
    metrics = {
        "eval_samples": float(samples),
        "policy_cross_entropy": mean_policy_ce,
        "policy_target_entropy": mean_target_entropy,
        "policy_kl": max(0.0, mean_policy_ce - mean_target_entropy),
        "policy_predicted_entropy": predicted_entropy / samples,
        "policy_visit_best": visit_best / samples,
        "sampled_action_top1": sampled_action_top1 / samples,
        "rank_cross_entropy": mean_rank_ce,
        "rank_target_entropy": mean_rank_entropy,
        "rank_kl": max(0.0, mean_rank_ce - mean_rank_entropy),
        "rank_best": rank_best / samples,
    }
    for name, count in supports.items():
        metrics[f"support_{name}"] = count
        if count <= 0.0:
            metrics[f"r2_{name}"] = float("nan")
            continue
        variance = sq_sums[name] / count - (sums[name] / count) ** 2
        mse = sq_err[name] / count
        metrics[f"r2_{name}"] = 1.0 - mse / variance if variance > 0.0 else float("nan")
    return metrics


def fit(
    trajectories: Sequence[self_play.SelfPlayTrajectory],
    *,
    net: Optional[nw.WelcomeToNet] = None,
    net_config: Optional[nw.NetConfig] = None,
    config: Optional[S2TrainConfig] = None,
    device: Optional[torch.device | str] = None,
    optimizer_state: Optional[dict[str, Any]] = None,
    epochs_completed: int = 0,
    log: bool = True,
) -> tuple[nw.WelcomeToNet, torch.optim.AdamW, dict[str, Any]]:
    """Train one S2 candidate from random or supplied weights."""
    config = config or S2TrainConfig()
    if epochs_completed < 0:
        raise ValueError("epochs_completed must be non-negative")
    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(config.seed)
    if net is None:
        net = nw.WelcomeToNet(net_config)
        initialization = "random"
    else:
        if net_config is not None and net.config != net_config:
            raise ValueError("net and net_config disagree")
        initialization = "checkpoint"
    net = net.to(torch_device)
    train_set, val_set = split_trajectories(
        trajectories, config.val_fraction, config.seed
    )
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    state[name] = value.to(torch_device)
        # The requested run configuration owns the resumed schedule. Adam's
        # moments resume; stale checkpoint hyperparameters do not silently
        # override an intentional LR or weight-decay change.
        for group in optimizer.param_groups:
            group["lr"] = config.lr
            group["weight_decay"] = config.weight_decay

    before = evaluate(net, val_set, torch_device)
    rng = random.Random(config.seed)
    history: list[dict[str, float]] = []
    total_steps = 0
    total_samples = 0
    started = time.perf_counter()
    for local_epoch in range(config.epochs):
        net.train()
        sums: dict[str, float] = {}
        epoch_samples = 0
        steps = 0
        max_gradient_norm = 0.0
        for raw in self_play.iter_batches(
            train_set,
            config.batch_size,
            rng,
            shuffle_buffer=config.shuffle_buffer,
        ):
            batch = nw.to_tensors(raw, torch_device)
            optimizer.zero_grad(set_to_none=True)
            out = net(
                batch["sheet_planes"],
                batch["sheet_scalars"],
                batch["viewer_plane"],
                batch["global_scalars"],
            )
            total, parts = nw.losses(out, batch)
            if not torch.isfinite(total):
                raise FloatingPointError("S2 loss became non-finite")
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                net.parameters(), config.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()

            rows = int(batch["policy"].shape[0])
            epoch_samples += rows
            steps += 1
            max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
            sums["total"] = sums.get("total", 0.0) + float(total.detach()) * rows
            for name, part in parts.items():
                sums[name] = sums.get(name, 0.0) + float(part.detach()) * rows
        if steps == 0 or epoch_samples == 0:
            raise ValueError("S2 training games produced no searched learner roots")
        total_steps += steps
        total_samples += epoch_samples
        row = {
            "epoch": float(epochs_completed + local_epoch + 1),
            "steps": float(steps),
            "samples": float(epoch_samples),
            "max_gradient_norm": max_gradient_norm,
            **{f"loss_{name}": value / epoch_samples for name, value in sums.items()},
        }
        history.append(row)
        if log:
            print(
                f"epoch {int(row['epoch'])}  steps {steps}  samples {epoch_samples}  "
                f"loss {row['loss_total']:.4f}  policy {row['loss_policy']:.4f}  "
                f"grad {max_gradient_norm:.3f}"
            )

    after = evaluate(net, val_set, torch_device)
    metrics: dict[str, Any] = dict(after)
    metrics.update(
        {
            "initialization": initialization,
            "train_games": float(len(train_set)),
            "val_games": float(len(val_set)),
            "optimizer_steps": float(total_steps),
            "training_samples": float(total_samples),
            "epochs_completed": float(epochs_completed + config.epochs),
            "train_seconds": time.perf_counter() - started,
            "history": history,
            "before": before,
        }
    )
    return net, optimizer, metrics


def save_checkpoint(
    path: str | Path,
    net: nw.WelcomeToNet,
    optimizer: torch.optim.Optimizer,
    config: S2TrainConfig,
    metrics: Mapping[str, Any],
    *,
    source: str,
) -> Path:
    """Atomically write a checkpoint that both S2 and the generic loader read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in net.state_dict().items()
        },
        "net_config": asdict(net.config),
        "train_config": asdict(config),
        "optimizer_state": optimizer.state_dict(),
        "epochs_completed": int(metrics["epochs_completed"]),
        "metrics": dict(metrics),
        "source": source,
    }
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def load_training_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[nw.WelcomeToNet, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not an S2 training checkpoint")
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"S2 checkpoint version {payload.get('version')} != {CHECKPOINT_VERSION}"
        )
    net = nw.WelcomeToNet(nw.NetConfig(**payload["net_config"]))
    net.load_state_dict(payload["state_dict"])
    return net.to(device), payload


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(
        description="Train a Welcome To candidate from S2 searched trajectories."
    )
    parser.add_argument("--trajectories", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init", help="S0 or S2 checkpoint; omit for random weights")
    source.add_argument("--resume", help="S2 checkpoint including optimizer state")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=S2TrainConfig().epochs)
    parser.add_argument("--batch-size", type=int, default=S2TrainConfig().batch_size)
    parser.add_argument("--lr", type=float, default=S2TrainConfig().lr)
    parser.add_argument("--weight-decay", type=float, default=S2TrainConfig().weight_decay)
    parser.add_argument("--grad-clip", type=float, default=S2TrainConfig().grad_clip)
    parser.add_argument("--val-fraction", type=float, default=S2TrainConfig().val_fraction)
    parser.add_argument("--shuffle-buffer", type=int, default=S2TrainConfig().shuffle_buffer)
    parser.add_argument("--seed", type=int, default=S2TrainConfig().seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    trajectories = self_play.read_trajectories(args.trajectories)
    optimizer_state = None
    epochs_completed = 0
    if args.resume:
        net, payload = load_training_checkpoint(args.resume, args.device)
        optimizer_state = payload["optimizer_state"]
        epochs_completed = int(payload["epochs_completed"])
        source_name = str(args.resume)
    elif args.init:
        net = s0_train.load(args.init, args.device)
        source_name = str(args.init)
    else:
        net = None
        source_name = "random"

    config = S2TrainConfig(
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
    )
    net, optimizer, metrics = fit(
        trajectories,
        net=net,
        config=config,
        device=args.device,
        optimizer_state=optimizer_state,
        epochs_completed=epochs_completed,
    )
    path = save_checkpoint(
        args.out, net, optimizer, config, metrics, source=source_name
    )
    metrics_path = path.with_suffix(path.suffix + ".metrics.json")
    json_metrics = _json_safe(metrics)
    metrics_path.write_text(
        json.dumps(json_metrics, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote S2 checkpoint to {path}")
    print(json.dumps(json_metrics, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
