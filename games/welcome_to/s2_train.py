"""Optimise Welcome To networks from searched S2 trajectories.

S0's trainer deliberately remains a GreedyBot pipeline shakedown.  This module
is the S2 boundary: it splits complete searched games, replays their MCTS visit
targets, evaluates those distributions, performs AdamW updates, and writes a
resume-capable checkpoint.  A caller may supply the S0 network or omit it and
bootstrap AlphaZero from random weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch.nn import functional as F

from games.welcome_to import network as nw
from games.welcome_to import s2_replay
from games.welcome_to import self_play
from games.welcome_to import train as s0_train
from games.welcome_to import training


CHECKPOINT_FORMAT = "welcome_to_s2"
CHECKPOINT_VERSION = 2
LEGACY_CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class S2TrainConfig:
    val_fraction: float = 0.10
    val_split_salt: str = "welcome_to_s2"
    train_steps: int = 200
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    log_every: int = 50
    max_eval_games: int = 256
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        if not self.val_split_salt:
            raise ValueError("val_split_salt must be non-empty")
        if (
            self.train_steps <= 0
            or self.batch_size <= 0
            or self.log_every <= 0
            or self.max_eval_games <= 0
        ):
            raise ValueError(
                "train_steps, batch_size, log_every, and max_eval_games must be positive"
            )
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


def stable_is_validation(
    iteration: int, game_seed: int, val_fraction: float, salt: str
) -> tuple[bool, int]:
    """Return a durable game-level holdout assignment and its hash value."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    digest = hashlib.blake2b(
        f"{salt}|{iteration}|{game_seed}".encode(), digest_size=8
    ).digest()
    value = struct.unpack("<Q", digest)[0]
    return value / 2.0**64 < val_fraction, value


def split_trajectories(
    trajectories: Sequence[self_play.SelfPlayTrajectory],
    val_fraction: float,
    seed: int,
    *,
    iterations: Optional[Sequence[int]] = None,
    salt: Optional[str] = None,
) -> tuple[list[self_play.SelfPlayTrajectory], list[self_play.SelfPlayTrajectory]]:
    """Stably split whole games so replay-window changes cannot leak rows.

    The assignment is a hash of ``(salt, iteration, game seed)`` rather than a
    shuffle of the current list. Thus a game remains validation-only for its
    entire lifetime in the durable replay window and across process resumes.
    """
    if len(trajectories) < 2:
        raise ValueError("S2 training needs at least two complete trajectories")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    if iterations is None:
        iterations = [0] * len(trajectories)
    if len(iterations) != len(trajectories):
        raise ValueError("trajectory iterations must align with trajectories")
    split_salt = salt if salt is not None else str(seed)
    assigned = [
        stable_is_validation(iteration, game.seed, val_fraction, split_salt)
        for game, iteration in zip(trajectories, iterations)
    ]
    validation = [
        game for game, (held, _digest) in zip(trajectories, assigned) if held
    ]
    train = [
        game for game, (held, _digest) in zip(trajectories, assigned) if not held
    ]
    # Tiny smoke corpora can miss one side of a probabilistic split. Keep them
    # runnable with a deterministic fallback; production buffers are thousands
    # of games and do not use this branch.
    if not validation:
        index = min(range(len(assigned)), key=lambda i: assigned[i][1])
        validation = [trajectories[index]]
        train = [game for i, game in enumerate(trajectories) if i != index]
    elif not train:
        index = max(range(len(assigned)), key=lambda i: assigned[i][1])
        train = [trajectories[index]]
        validation = [game for i, game in enumerate(trajectories) if i != index]
    return train, validation


def _bounded_evaluation_set(
    trajectories: Sequence[self_play.SelfPlayTrajectory],
    limit: int,
    *,
    iterations: Optional[Sequence[int]],
    salt: str,
    domain: str,
) -> list[self_play.SelfPlayTrajectory]:
    """Choose a deterministic, order-independent fixed-size diagnostic set."""
    if limit <= 0:
        raise ValueError("evaluation-game limit must be positive")
    if iterations is None:
        iterations = [0] * len(trajectories)
    if len(iterations) != len(trajectories):
        raise ValueError("evaluation iterations must align with trajectories")
    ranked = []
    for game, iteration in zip(trajectories, iterations):
        digest = hashlib.blake2b(
            f"{salt}|{domain}|{iteration}|{game.seed}".encode(), digest_size=8
        ).digest()
        ranked.append((struct.unpack("<Q", digest)[0], iteration, game.seed, game))
    ranked.sort(key=lambda item: item[:3])
    return [item[3] for item in ranked[:limit]]


def _masked_rank(out: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]):
    mask = torch.stack(
        [batch[f"rank_mask_{rank}"] for rank in range(training.MAX_RANKS)], dim=-1
    )
    target = torch.stack(
        [batch[f"rank_p_{rank}"] for rank in range(training.MAX_RANKS)], dim=-1
    )
    log_policy = nw.masked_log_softmax(out["rank_logits"], mask)
    return mask, target, log_policy


def _load_optimizer_state_compatible(
    optimizer: torch.optim.Optimizer,
    saved: dict[str, Any],
    net: nw.WelcomeToNet,
    saved_parameter_names: Optional[Sequence[str]] = None,
) -> None:
    """Expand legacy Adam moments for the appended per-seat output rows."""
    current = optimizer.state_dict()
    saved_ids = [item for group in saved["param_groups"] for item in group["params"]]
    current_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    if len(saved_ids) != len(current_parameters):
        raise ValueError("checkpoint optimizer parameter count changed")
    names = {id(parameter): name for name, parameter in net.named_parameters()}
    current_parameter_names = tuple(names[id(parameter)] for parameter in current_parameters)
    if saved_parameter_names is not None and tuple(saved_parameter_names) != (
        current_parameter_names
    ):
        raise ValueError(
            "checkpoint optimizer parameter names/order do not match the network"
        )
    migrated = {
        "state": {key: dict(value) for key, value in saved["state"].items()},
        "param_groups": [dict(group) for group in saved["param_groups"]],
    }
    final_index = max(
        index
        for index, module in enumerate(net.per_seat_head)
        if isinstance(module, torch.nn.Linear)
    )
    expandable = {
        f"per_seat_head.{final_index}.weight",
        f"per_seat_head.{final_index}.bias",
    }
    old_rows = len(nw.LEGACY_PER_SEAT_HEAD_TARGETS)
    new_rows = len(nw.PER_SEAT_HEAD_TARGETS)
    for saved_id, parameter in zip(saved_ids, current_parameters):
        state = migrated["state"].get(saved_id, {})
        for key, value in list(state.items()):
            if not torch.is_tensor(value) or value.ndim == 0 or value.shape == parameter.shape:
                continue
            name = names[id(parameter)]
            if (
                name not in expandable
                or value.shape[0] != old_rows
                or parameter.shape[0] != new_rows
                or value.shape[1:] != parameter.shape[1:]
            ):
                raise ValueError(
                    f"checkpoint optimizer tensor {name}.{key} has shape "
                    f"{tuple(value.shape)}, expected {tuple(parameter.shape)}"
                )
            expanded = value.new_zeros(parameter.shape)
            expanded[:old_rows].copy_(value)
            state[key] = expanded
    optimizer.load_state_dict(migrated)


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
    binary_ce: dict[str, float] = {}
    binary_correct: dict[str, float] = {}

    rng = random.Random(0)
    rust_loader = self_play.rust_training_loader(
        trajectories, batch_size, shuffle_seed=None
    )
    raw_batches = (
        self_play.iter_rust_training_batches(rust_loader)
        if rust_loader is not None
        else self_play.iter_batches(
            trajectories,
            batch_size,
            rng,
            shuffle_buffer=batch_size * 2,
        )
    )
    for raw in raw_batches:
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
            if name in training.BINARY_TARGETS:
                probability = torch.sigmoid(prediction)
                binary_ce[name] = binary_ce.get(name, 0.0) + float(
                    (
                        mask
                        * F.binary_cross_entropy_with_logits(
                            prediction, target_value, reduction="none"
                        )
                    ).sum()
                )
                binary_correct[name] = binary_correct.get(name, 0.0) + float(
                    (mask * ((probability >= 0.5) == (target_value >= 0.5))).sum()
                )
                prediction = probability
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
            metrics[f"target_mean_{name}"] = float("nan")
            metrics[f"target_std_{name}"] = float("nan")
            if name in training.BINARY_TARGETS:
                metrics[f"bce_{name}"] = float("nan")
                metrics[f"brier_{name}"] = float("nan")
                metrics[f"accuracy_{name}"] = float("nan")
                metrics[f"positive_rate_{name}"] = float("nan")
            else:
                metrics[f"r2_{name}"] = float("nan")
            continue
        target_mean = sums[name] / count
        variance = max(0.0, sq_sums[name] / count - target_mean**2)
        metrics[f"target_mean_{name}"] = target_mean
        metrics[f"target_std_{name}"] = math.sqrt(variance)
        if name in training.BINARY_TARGETS:
            metrics[f"bce_{name}"] = binary_ce[name] / count
            metrics[f"brier_{name}"] = sq_err[name] / count
            metrics[f"accuracy_{name}"] = binary_correct[name] / count
            metrics[f"positive_rate_{name}"] = target_mean
            continue
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
    optimizer_parameter_names: Optional[Sequence[str]] = None,
    optimizer_steps_completed: int = 0,
    training_runs_completed: int = 0,
    replay_metrics: Optional[Mapping[str, Any]] = None,
    trajectory_iterations: Optional[Sequence[int]] = None,
    log: bool = True,
) -> tuple[nw.WelcomeToNet, torch.optim.AdamW, dict[str, Any]]:
    """Train one S2 candidate for a fixed number of replay updates.

    Like 7WD's ``train_steps``, the optimization budget is independent of the
    replay-window size. Each minibatch is sampled uniformly with replacement;
    the production path performs both the index draw and row decode in Rust.
    """
    config = config or S2TrainConfig()
    if optimizer_steps_completed < 0 or training_runs_completed < 0:
        raise ValueError("completed training counters must be non-negative")
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
    train_set, val_pool = split_trajectories(
        trajectories,
        config.val_fraction,
        config.seed,
        iterations=trajectory_iterations,
        salt=config.val_split_salt,
    )
    if trajectory_iterations is None:
        val_iterations = [0] * len(val_pool)
    else:
        iteration_by_seed = {
            game.seed: iteration
            for game, iteration in zip(trajectories, trajectory_iterations)
        }
        val_iterations = [iteration_by_seed[game.seed] for game in val_pool]
    val_set = _bounded_evaluation_set(
        val_pool,
        config.max_eval_games,
        iterations=val_iterations,
        salt=config.val_split_salt,
        domain="validation",
    )
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    if optimizer_state is not None:
        _load_optimizer_state_compatible(
            optimizer,
            optimizer_state,
            net,
            optimizer_parameter_names,
        )
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
    pretrain_newest = None
    newest_iteration = None
    if trajectory_iterations is not None:
        newest_iteration = max(trajectory_iterations)
        newest_games = [
            game
            for game, iteration in zip(trajectories, trajectory_iterations)
            if iteration == newest_iteration
        ]
        if newest_games:
            newest_eval = _bounded_evaluation_set(
                newest_games,
                config.max_eval_games,
                iterations=[newest_iteration] * len(newest_games),
                salt=config.val_split_salt,
                domain="newest",
            )
            pretrain_newest = evaluate(net, newest_eval, torch_device)
    sample_seed = (
        config.seed
        ^ 0x5332_5245_504C_4159
        ^ optimizer_steps_completed
        ^ (training_runs_completed << 32)
    ) & ((1 << 64) - 1)
    rng = random.Random(sample_seed)
    history: list[dict[str, float]] = []
    total_steps = 0
    total_samples = 0
    loader_seconds = 0.0
    tensor_transfer_seconds = 0.0
    rust_loader = self_play.rust_training_loader(
        train_set,
        config.batch_size,
        shuffle_seed=None,
    )
    loader_name = "rust_wts" if rust_loader is not None else "python_replay"
    if rust_loader is not None:
        rust_loader.reset_random(sample_seed, config.train_steps)
        raw_batches = self_play.iter_rust_training_batches(rust_loader)
    else:
        raw_batches = self_play.iter_random_batches(
            train_set,
            config.batch_size,
            config.train_steps,
            rng,
        )
    started = time.perf_counter()
    net.train()
    sums: dict[str, float] = {}
    window_samples = 0
    window_steps = 0
    max_gradient_norm = 0.0
    iterator = iter(raw_batches)
    for local_step in range(config.train_steps):
        loader_started = time.perf_counter()
        try:
            raw = next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"training loader ended after {local_step}/{config.train_steps} steps"
            ) from exc
        loader_seconds += time.perf_counter() - loader_started
        transfer_started = time.perf_counter()
        batch = nw.to_tensors(raw, torch_device)
        tensor_transfer_seconds += time.perf_counter() - transfer_started
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
        total_steps += 1
        total_samples += rows
        window_steps += 1
        window_samples += rows
        max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
        sums["total"] = sums.get("total", 0.0) + float(total.detach()) * rows
        for name, part in parts.items():
            sums[name] = sums.get(name, 0.0) + float(part.detach()) * rows

        if window_steps == config.log_every or local_step + 1 == config.train_steps:
            row = {
                "step": float(optimizer_steps_completed + total_steps),
                "steps": float(window_steps),
                "samples": float(window_samples),
                "max_gradient_norm": max_gradient_norm,
                **{
                    f"loss_{name}": value / window_samples
                    for name, value in sums.items()
                },
            }
            history.append(row)
            if log:
                print(
                    f"step {int(row['step'])}  samples {total_samples}  "
                    f"loss {row['loss_total']:.4f}  "
                    f"policy {row['loss_policy']:.4f}  "
                    f"grad {max_gradient_norm:.3f}"
                )
            sums = {}
            window_samples = 0
            window_steps = 0
            max_gradient_norm = 0.0

    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("training loader yielded more than the fixed step budget")

    after = evaluate(net, val_set, torch_device)
    train_positions = sum(len(game.searches) for game in train_set)
    buffer_positions = sum(len(game.searches) for game in trajectories)
    replay = dict(replay_metrics or {})
    new_positions = int(replay.get("newest_positions", buffer_positions))
    metrics: dict[str, Any] = dict(after)
    metrics.update(
        {
            "initialization": initialization,
            "train_games": float(len(train_set)),
            "val_games": float(len(val_set)),
            "val_pool_games": float(len(val_pool)),
            "max_eval_games": float(config.max_eval_games),
            "optimizer_steps": float(total_steps),
            "optimizer_steps_completed": float(
                optimizer_steps_completed + total_steps
            ),
            "training_runs_completed": float(training_runs_completed + 1),
            "training_samples": float(total_samples),
            "buffer_positions": float(buffer_positions),
            "train_positions": float(train_positions),
            "new_positions": float(new_positions),
            "samples_per_new_position": (
                total_samples / new_positions if new_positions else None
            ),
            "buffer_passes": total_samples / train_positions,
            "training_loader": loader_name,
            "loader_seconds": loader_seconds,
            "tensor_transfer_seconds": tensor_transfer_seconds,
            "train_seconds": time.perf_counter() - started,
            "history": history,
            "before": before,
            "pretrain_newest_iteration": newest_iteration,
            "pretrain_newest_metrics": pretrain_newest,
            "replay": replay,
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
    parameter_names = {id(parameter): name for name, parameter in net.named_parameters()}
    optimizer_parameter_names = [
        parameter_names[id(parameter)]
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
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
        "optimizer_parameter_names": optimizer_parameter_names,
        "optimizer_steps_completed": int(metrics["optimizer_steps_completed"]),
        "training_runs_completed": int(metrics["training_runs_completed"]),
        # Read compatibility for old continuation scripts. This counter now
        # means completed fixed-step training calls, not literal epochs.
        "epochs_completed": int(metrics["training_runs_completed"]),
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
    version = int(payload.get("version", -1))
    if version not in (LEGACY_CHECKPOINT_VERSION, CHECKPOINT_VERSION):
        raise ValueError(
            f"unsupported S2 checkpoint version {payload.get('version')}"
        )
    net = nw.WelcomeToNet(nw.NetConfig(**payload["net_config"]))
    nw.load_state_dict_compatible(net, payload["state_dict"])
    return net.to(device), payload


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(
        description="Train a Welcome To candidate from S2 searched trajectories."
    )
    replay_source = parser.add_mutually_exclusive_group(required=True)
    replay_source.add_argument(
        "--trajectories",
        action="append",
        help="trajectory prefix; repeat to train from multiple explicit corpora",
    )
    replay_source.add_argument(
        "--replay-root",
        help="run directory containing durable iter_NNNN generation directories",
    )
    parser.add_argument("--replay-through-iteration", type=int)
    parser.add_argument("--replay-window-coefficient", type=float, default=16.0)
    parser.add_argument("--replay-window-exponent", type=float, default=0.6)
    parser.add_argument("--replay-window-cap-games", type=int, default=20_000)
    parser.add_argument("--replay-window-floor-games", type=int, default=500)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init", help="S0 or S2 checkpoint; omit for random weights")
    source.add_argument("--resume", help="S2 checkpoint including optimizer state")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-steps", type=int, default=S2TrainConfig().train_steps)
    parser.add_argument("--batch-size", type=int, default=S2TrainConfig().batch_size)
    parser.add_argument("--lr", type=float, default=S2TrainConfig().lr)
    parser.add_argument("--weight-decay", type=float, default=S2TrainConfig().weight_decay)
    parser.add_argument("--grad-clip", type=float, default=S2TrainConfig().grad_clip)
    parser.add_argument("--val-fraction", type=float, default=S2TrainConfig().val_fraction)
    parser.add_argument("--val-split-salt", default=S2TrainConfig().val_split_salt)
    parser.add_argument("--log-every", type=int, default=S2TrainConfig().log_every)
    parser.add_argument(
        "--max-eval-games", type=int, default=S2TrainConfig().max_eval_games
    )
    parser.add_argument("--seed", type=int, default=S2TrainConfig().seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    if args.replay_root:
        corpus = s2_replay.assemble_replay(
            args.replay_root,
            through_iteration=args.replay_through_iteration,
            coefficient=args.replay_window_coefficient,
            exponent=args.replay_window_exponent,
            cap_games=args.replay_window_cap_games,
            floor_games=args.replay_window_floor_games,
        )
    else:
        corpus = s2_replay.explicit_replay(args.trajectories)
    trajectories = corpus.trajectories
    optimizer_state = None
    optimizer_parameter_names = None
    optimizer_steps_completed = 0
    training_runs_completed = 0
    if args.resume:
        net, payload = load_training_checkpoint(args.resume, args.device)
        optimizer_state = payload["optimizer_state"]
        optimizer_parameter_names = payload.get("optimizer_parameter_names")
        optimizer_steps_completed = int(payload.get("optimizer_steps_completed", 0))
        training_runs_completed = int(
            payload.get("training_runs_completed", payload.get("epochs_completed", 0))
        )
        source_name = str(args.resume)
    elif args.init:
        net = s0_train.load(args.init, args.device)
        source_name = str(args.init)
    else:
        net = None
        source_name = "random"

    config = S2TrainConfig(
        val_fraction=args.val_fraction,
        val_split_salt=args.val_split_salt,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        max_eval_games=args.max_eval_games,
        seed=args.seed,
    )
    net, optimizer, metrics = fit(
        trajectories,
        net=net,
        config=config,
        device=args.device,
        optimizer_state=optimizer_state,
        optimizer_parameter_names=optimizer_parameter_names,
        optimizer_steps_completed=optimizer_steps_completed,
        training_runs_completed=training_runs_completed,
        replay_metrics=corpus.metrics(),
        trajectory_iterations=corpus.trajectory_iterations,
    )
    path = save_checkpoint(
        args.out, net, optimizer, config, metrics, source=source_name
    )
    s2_replay.write_replay_manifest(
        path.with_suffix(path.suffix + ".replay.json"), corpus
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
