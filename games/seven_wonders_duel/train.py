"""Phase B trainer (plan §4): six-head training over buffer records.

Kingdomino trainer skeleton carried over: game-honest splits, trivial-baseline
comparisons printed next to net metrics, early stop, JSON summary. Checkpoints
embed ENCODER_SIGNATURE — a loader must refuse a checkpoint whose signature
disagrees with the live encoder (export discipline, spec §5.8).

Usage:
  python -m games.seven_wonders_duel.train --buffer <records.jsonl> [--model mlp]
      [--overfit] [--epochs N] [--out runs/phase_b]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import struct
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .buffer import read_records
from .dataset import Example, collate, examples_from_records
from .encoder import ENCODER_SIGNATURE
from .mlp import SWDMlp
from .net import LEGACY_HEADS, SWDNet, masked_policy_log_softmax

AUX_WEIGHT_DEFAULT = 0.2

#: Multiplier on every head that fits a per-GAME label rather than a
#: per-position one: value, joint7, margin, military, science.
#:
#: Those five share one label per game across all ~16 of its rows, so an
#: iteration that produces ~16,500 policy labels produces only ~1,000
#: independent outcome labels -- and at the defaults they carry
#: 1.0 + 0.2*4 = 1.8 of the loss weight against the policy head's 1.0. That is
#: the side of the objective best placed to memorise, and on a shared trunk
#: memorising it drags the representation the policy head depends on.
#:
#: 1.0 is the historical behaviour. Lower it to test whether the outcome heads
#: are what stalls a run; see ablate_value_head.py.
VALUE_WEIGHT_DEFAULT = 1.0


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    aux_weight: float = AUX_WEIGHT_DEFAULT,
    value_weight: float = VALUE_WEIGHT_DEFAULT,
    value_bootstrap: float = 0.0,
    solver_value_target: bool = True,
    row_weights: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    log_policy = masked_policy_log_softmax(outputs["policy"], batch["legal_mask"])
    # Targets are zero on illegal actions where log_policy is -inf; read only
    # legal positions so 0 * -inf never produces NaN.
    safe_log = torch.where(
        batch["legal_mask"], log_policy, torch.zeros_like(log_policy)
    )
    per_row = -(batch["policy"] * safe_log).sum(dim=-1)
    has_policy = batch["has_policy"]
    # Per-row weights, head by head. Duplicating a row would have been simpler
    # and is wrong: every head averages over the batch, so a row duplicated to
    # emphasise its exact VALUE would also count its policy and all four
    # auxiliary targets twice.
    policy_w = (
        batch.get("policy_weight") if row_weights else None
    )
    if policy_w is None:
        policy_w = torch.ones_like(per_row)
    value_w = batch.get("value_weight") if row_weights else None
    if value_w is None:
        value_w = torch.ones_like(per_row)
    if has_policy.any():
        weights = policy_w[has_policy]
        policy_loss = (per_row[has_policy] * weights).sum() / weights.sum().clamp(min=1e-9)
    else:
        policy_loss = per_row.new_zeros(())
    solver_rows = batch.get("value_solver_valid") if solver_value_target else None
    has_solver = solver_rows is not None and bool(solver_rows.any())
    if value_bootstrap > 0.0 and "value_soft" in batch:
        # Blend the realised outcome with the search's own estimate. The outcome
        # is one sample of a probability; fitting it hard produces a head that is
        # confidently wrong off-distribution (cloud3: holdout value loss tripled
        # while accuracy moved 4 points -- pure overconfidence). Rows without a
        # search keep the hard label, so nothing is invented for them.
        hard = F.one_hot(batch["value_class"], num_classes=3).float()
        target = torch.where(
            batch["value_soft_valid"].unsqueeze(1),
            (1.0 - value_bootstrap) * hard + value_bootstrap * batch["value_soft"],
            hard,
        )
    elif has_solver:
        target = F.one_hot(batch["value_class"], num_classes=3).float()
    else:
        target = None
    if has_solver:
        # A proven value REPLACES the outcome outright rather than blending with
        # it -- at full weight, and regardless of `value_bootstrap`. The realised
        # result of an endgame the solver has settled is a sample of this number
        # produced by two players who may both then err; there is nothing in it
        # the exact value does not already contain, and averaging the two can
        # only move the target away from the truth.
        target = torch.where(solver_rows.unsqueeze(1), batch["value_solver"], target)
    if target is None:
        per_value = F.cross_entropy(
            outputs["value"], batch["value_class"], reduction="none"
        )
    else:
        per_value = F.cross_entropy(outputs["value"], target, reduction="none")
    value_loss = (per_value * value_w).sum() / value_w.sum().clamp(min=1e-9)
    joint7_loss = F.cross_entropy(outputs["joint7"], batch["joint7"])
    margin_valid = batch["margin_valid"]
    if margin_valid.any():
        margin_loss = F.mse_loss(
            outputs["margin"][margin_valid], batch["margin"][margin_valid]
        )
    else:
        margin_loss = outputs["margin"].new_zeros(())
    military_loss = F.mse_loss(outputs["military"], batch["military_final"])
    science_loss = F.mse_loss(outputs["science"], batch["sci_final"])
    total = (
        policy_loss
        + value_weight * value_loss
        + value_weight
        * aux_weight
        * (joint7_loss + margin_loss + military_loss + science_loss)
    )
    return total, {
        "total": float(total.detach()),
        "policy": float(policy_loss.detach()),
        "value": float(value_loss.detach()),
        "joint7": float(joint7_loss.detach()),
        "margin": float(margin_loss.detach()),
        "military": float(military_loss.detach()),
        "science": float(science_loss.detach()),
    }


def _validate_precision(precision: str) -> None:
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")


def _evaluation_autocast(device: str, precision: str):
    """Match bf16 training during validation without changing fp32 defaults."""

    _validate_precision(precision)
    if precision == "bf16" and str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _training_autocast(device: str, precision: str):
    """Preserve the historical CUDA-fp16 baseline; opt into bf16 explicitly."""

    _validate_precision(precision)
    if not str(device).startswith("cuda"):
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


@torch.no_grad()
def evaluate(
    model,
    examples: list[Example],
    device: str,
    batch_size: int = 512,
    aux_weight: float = AUX_WEIGHT_DEFAULT,
    value_weight: float = VALUE_WEIGHT_DEFAULT,
    value_bootstrap: float = 0.0,
    precision: str = "fp32",
):
    model.eval()
    sums: dict[str, float] = {}
    correct = {"value": 0, "joint7": 0, "policy_top1": 0}
    abs_err = {"margin": 0.0, "military": 0.0, "science": 0.0}
    margin_rows = 0
    policy_rows = 0
    count = 0
    for start in range(0, len(examples), batch_size):
        batch = collate(examples[start : start + batch_size], device)
        with _evaluation_autocast(device, precision):
            outputs = model(batch)
            # `solver_value_target=False` for the same reason validation drops
            # `value_bootstrap`: a held-out number has to mean the same thing
            # across runs, and against the realised outcome. Scoring solver rows
            # against their own exact value would make the metric easier exactly
            # where the training target was easier, and a run with the solver on
            # would post a better validation loss without playing better.
            _, parts = compute_losses(
                outputs,
                batch,
                aux_weight,
                value_weight,
                value_bootstrap,
                solver_value_target=False,
                # Unweighted for the same reason: a held-out number has to mean
                # the same thing across runs. Upweighting solved rows in
                # validation would make the metric easier exactly where training
                # was, which is how a run posts a better number without playing
                # better.
                row_weights=False,
            )
        rows = batch["value_class"].shape[0]
        for key, value in parts.items():
            sums[key] = sums.get(key, 0.0) + value * rows
        count += rows
        correct["value"] += int(
            (outputs["value"].argmax(-1) == batch["value_class"]).sum()
        )
        correct["joint7"] += int((outputs["joint7"].argmax(-1) == batch["joint7"]).sum())
        masked = outputs["policy"].masked_fill(~batch["legal_mask"], float("-inf"))
        top1 = masked.argmax(-1)
        target_top = batch["policy"].argmax(-1)
        has = batch["has_policy"]
        correct["policy_top1"] += int((top1[has] == target_top[has]).sum())
        policy_rows += int(has.sum())
        valid = batch["margin_valid"]
        if valid.any():
            abs_err["margin"] += float(
                (outputs["margin"][valid] - batch["margin"][valid]).abs().sum()
            )
            margin_rows += int(valid.sum())
        abs_err["military"] += float(
            (outputs["military"] - batch["military_final"]).abs().sum()
        )
        abs_err["science"] += float(
            (outputs["science"] - batch["sci_final"]).abs().mean(dim=-1).sum()
        )
    metrics = {key: value / count for key, value in sums.items()}
    metrics["value_acc"] = correct["value"] / count
    metrics["joint7_acc"] = correct["joint7"] / count
    metrics["policy_top1"] = correct["policy_top1"] / max(policy_rows, 1)
    metrics["margin_mae"] = abs_err["margin"] / max(margin_rows, 1)
    metrics["military_mae"] = abs_err["military"] / count
    metrics["science_mae"] = abs_err["science"] / count
    model.train()
    return metrics


def baselines(examples: list[Example]) -> dict[str, float]:
    """What each head must beat: majority-class rates for the classifiers,
    predict-the-mean MAE for the regressions, uniform policy cross-entropy
    (mean of log(n_legal) over policy-bearing examples)."""

    def base_rate(values):
        counts: dict[int, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        return max(counts.values()) / len(values)

    def mean_mae(values):
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum(abs(v - mean) for v in values) / len(values)

    policy_examples = [e for e in examples if e.has_policy]
    margins = [e.margin for e in examples if e.margin_valid]
    sci = [e.sci_final_my for e in examples] + [e.sci_final_opp for e in examples]
    return {
        "value_base_rate": base_rate([e.value_class for e in examples]),
        "joint7_base_rate": base_rate([e.joint7_class for e in examples]),
        "policy_uniform_loss": sum(math.log(len(e.legal)) for e in policy_examples)
        / max(len(policy_examples), 1),
        "margin_mae": mean_mae(margins),
        "military_mae": mean_mae([e.military_final for e in examples]),
        "science_mae": mean_mae(sci),
    }


def game_honest_split(examples: list[Example], val_frac: float, seed: int = 0):
    """Held-out validation that never shares a game with training.

    When examples carry iteration labels (Phase D self-play), whole recent
    iterations are held out — the KD `iteration_split` discipline, so val
    measures generalization across agent generations, not just across games.
    Unlabeled buffers (bot games) fall back to a by-game split.
    """

    # Curriculum seed games are intentionally unlabeled (iteration=None). They
    # remain training-only and must not disable the honest recent-iteration
    # holdout once at least two self-play generations exist.
    iterations = {e.iteration for e in examples if e.iteration is not None}
    if len(iterations) > 1:
        ordered = sorted(iterations)
        labeled = [e for e in examples if e.iteration is not None]
        total = len(labeled)
        by_iteration = {
            it: sum(1 for e in examples if e.iteration == it) for it in ordered
        }
        val_iterations: set[int] = set()
        held = 0
        for it in reversed(ordered):
            if held >= total * val_frac:
                break
            val_iterations.add(it)
            held += by_iteration[it]
        train = [e for e in examples if e.iteration not in val_iterations]
        val = [e for e in examples if e.iteration in val_iterations]
        return train, val

    keys = sorted({e.game_key for e in examples})
    rng = random.Random(seed)
    rng.shuffle(keys)
    val_keys = set(keys[: max(1, int(len(keys) * val_frac))])
    train = [e for e in examples if e.game_key not in val_keys]
    val = [e for e in examples if e.game_key in val_keys]
    return train, val


def stable_is_validation(
    iteration: int | None, game_key: int, val_frac: float, salt: str = ""
) -> bool:
    """Assign one game to train/validation by a hash of its identity.

    The assignment depends only on ``(salt, iteration, game_key)``, so a game
    keeps the same side for as long as it lives in the replay window.  The old
    per-iteration reshuffle (``phase_d_game_honest_split`` reseeded from
    ``seed + iteration``) let a game validate at one iteration, train at the
    next and validate again at the third -- which contaminates the holdout and
    understates validation loss on older data.

    ``blake2b`` rather than :func:`hash` because the builtin is randomized per
    process; the split has to survive a resume.  Curriculum seed games carry
    ``iteration is None`` and are always training-only.
    """

    if val_frac <= 0.0 or iteration is None:
        return False
    digest = hashlib.blake2b(
        f"{salt}|{iteration}|{game_key}".encode(), digest_size=8
    ).digest()
    return struct.unpack("<Q", digest)[0] / 2.0**64 < val_frac


def stable_game_split(
    examples: list[Example], val_frac: float, salt: str = ""
) -> tuple[list[Example], list[Example]]:
    """Split by :func:`stable_is_validation`, never sharing a game."""

    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must lie in [0, 1)")
    decisions: dict[tuple[int | None, int], bool] = {}
    train: list[Example] = []
    val: list[Example] = []
    for example in examples:
        key = (example.iteration, example.game_key)
        held = decisions.get(key)
        if held is None:
            held = stable_is_validation(
                example.iteration, example.game_key, val_frac, salt
            )
            decisions[key] = held
        (val if held else train).append(example)
    return train, val


def make_checkpoint(model, config: dict) -> dict:
    model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    return {
        "model_state": model.state_dict(),
        "config": config,
        "encoder_signature": ENCODER_SIGNATURE,
    }


def migrate_state_dict(old_state: dict, model) -> dict:
    """Additive-schema warm start (spec §5.8a): load every parameter that still
    matches, zero-initialize parameters with no counterpart (new token types'
    entity embeddings and feature projections), and zero-pad grown embedding
    tables (the type-embedding table when a type is appended).

    Zero-init makes the new tokens' pre-activation contribution exactly zero.
    Note the honest caveat (also in the spec): zero-VALUE tokens still
    participate in attention normalization, so switch-on is near-neutral, not
    bit-neutral — exact bit-neutrality requires masking the new type until
    enabled. Returns a report of what was loaded / grown / zero-initialized.
    """

    new_state = model.state_dict()
    report = {"loaded": [], "grown": [], "zeroed": []}
    for key, tensor in new_state.items():
        if key in old_state and old_state[key].shape == tensor.shape:
            new_state[key] = old_state[key]
            report["loaded"].append(key)
        elif (
            key in old_state
            and old_state[key].ndim == tensor.ndim
            and old_state[key].shape[1:] == tensor.shape[1:]
            and old_state[key].shape[0] < tensor.shape[0]
        ):
            grown = torch.zeros_like(tensor)
            grown[: old_state[key].shape[0]] = old_state[key]
            new_state[key] = grown
            report["grown"].append(key)
        elif (
            key in old_state
            and old_state[key].ndim == 2
            and tensor.ndim == 2
            and old_state[key].shape[0] == tensor.shape[0]
            and old_state[key].shape[1] < tensor.shape[1]
        ):
            # A Linear whose INPUT width grew: appending features to a token's
            # schema widens `[out, in]` along dim 1. Zero the new columns so the
            # appended features contribute exactly nothing and the model computes
            # what it did before.
            #
            # This only aligns because new features are APPENDED. Inserted
            # mid-vector, every later column shifts and the old weights land on
            # the wrong features -- which is worse than useless, and is why the
            # 2026-08-17 tempo features were moved to the end of GLOBAL_FEATURES
            # after this branch was found missing. Before that, growth along the
            # input dim fell through to the `else` below and zeroed the whole
            # projection, which the caller's guard correctly refused.
            grown = torch.zeros_like(tensor)
            grown[:, : old_state[key].shape[1]] = old_state[key]
            new_state[key] = grown
            report["grown"].append(key)
        else:
            new_state[key] = torch.zeros_like(tensor)
            report["zeroed"].append(key)
    model.load_state_dict(new_state)
    return report


def load_checkpoint(
    path, model, *, migrate: bool = False, checkpoint: dict | None = None
) -> dict:
    """Load a checkpoint, optionally reusing an already-read payload.

    Gate construction reads metadata and weights together; accepting the
    payload avoids a second ``torch.load`` of each large checkpoint.
    """

    if checkpoint is None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["encoder_signature"] != ENCODER_SIGNATURE:
        if not migrate:
            raise ValueError(
                "checkpoint encoder signature does not match the live encoder — "
                "the encoding schema changed since this model was trained "
                "(pass migrate=True for an additive-schema warm start)"
            )
        checkpoint["migration"] = migrate_state_dict(checkpoint["model_state"], model)
        return checkpoint
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def build_model(name: str, d_model: int, layers: int, heads: int | None = None):
    """Build a model. ``heads=None`` derives the width-appropriate head count.

    Rebuilding a *saved* model must pass the head count its checkpoint recorded
    (`heads_from_config`), never the derived default: attention parameter shapes
    are head-count independent, so a mismatch loads cleanly and silently
    computes something else.
    """

    if name == "transformer":
        return SWDNet(d_model=d_model, layers=layers, heads=heads)
    if name == "mlp":
        return SWDMlp(d_model=d_model)
    raise ValueError(f"unknown model: {name}")


def heads_from_config(config: dict) -> int:
    """Head count for rebuilding a checkpoint, honouring pre-`heads` files."""

    return int(config.get("heads", LEGACY_HEADS))


def train_loop(
    model,
    train_examples: list[Example],
    val_examples: list[Example] | None,
    *,
    device: str,
    epochs: int,
    batch_size: int = 512,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    aux_weight: float = AUX_WEIGHT_DEFAULT,
    value_weight: float = VALUE_WEIGHT_DEFAULT,
    value_bootstrap: float = 0.0,
    patience: int = 8,
    precision: str = "fp32",
    log=print,
):
    """Offline epoch trainer for a fixed buffer (Phase B gate, ``train.py`` CLI).

    The self-play loop uses :func:`train_steps` instead. Epochs make sense here,
    where the dataset is fixed and training runs once; inside the loop they made
    training cost track buffer size and re-presented old positions on every
    iteration. Note this function restores the best-validation weights
    unconditionally, which is appropriate for a one-shot offline fit but was the
    source of run 02 silently discarding seven of its eight epochs per
    iteration.
    """

    model.to(device).train()
    # AdamW decouples decay (`w -= lr*lambda*w`, a fixed fractional shrink);
    # Adam folds it into the gradient as L2, so it passes through the adaptive
    # denominator and its RELATIVE strength grows as gradients shrink. The
    # Kingdomino loop uses Adam, and never showed 7WD's late-run gap climb.
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"unknown optimizer: {optimizer_name!r}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.startswith("cuda")
    _validate_precision(precision)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and precision == "fp32"
    )
    rng = random.Random(0)
    best = {"val_total": float("inf"), "epoch": -1, "state": None}
    history = []
    for epoch in range(epochs):
        rng.shuffle(train_examples)
        start_time = time.time()
        running: dict[str, float] = {}
        batches = 0
        for start in range(0, len(train_examples), batch_size):
            batch = collate(train_examples[start : start + batch_size], device)
            optimizer.zero_grad(set_to_none=True)
            with _training_autocast(device, precision):
                outputs = model(batch)
                total, parts = compute_losses(outputs, batch, aux_weight, value_weight, value_bootstrap)
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + value
            batches += 1
        scheduler.step()
        train_parts = {k: v / batches for k, v in running.items()}
        row = {"epoch": epoch, "train": train_parts, "secs": time.time() - start_time}
        if val_examples:
            # `value_bootstrap` is deliberately NOT forwarded: validation must
            # score against the real game outcome, not the blended target the
            # arm trains on. Passing it here would make any soft-target run look
            # better by grading itself on its own softened labels.
            val_metrics = evaluate(
                model,
                val_examples,
                device,
                batch_size,
                aux_weight,
                precision=precision,
            )
            row["val"] = val_metrics
            log(
                f"epoch {epoch}: train total {train_parts['total']:.4f} "
                f"(policy {train_parts['policy']:.4f} value {train_parts['value']:.4f}) "
                f"| val total {val_metrics['total']:.4f} "
                f"value_acc {val_metrics['value_acc']:.3f} "
                f"joint7_acc {val_metrics['joint7_acc']:.3f} "
                f"policy_top1 {val_metrics['policy_top1']:.3f} "
                f"[{row['secs']:.0f}s]"
            )
            if val_metrics["total"] < best["val_total"] - 1e-4:
                best = {
                    "val_total": val_metrics["total"],
                    "epoch": epoch,
                    "state": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                }
            elif epoch - best["epoch"] >= patience:
                log(f"early stop at epoch {epoch} (best epoch {best['epoch']})")
                history.append(row)
                break
        else:
            log(
                f"epoch {epoch}: train total {train_parts['total']:.4f} "
                f"(policy {train_parts['policy']:.4f} value {train_parts['value']:.4f} "
                f"joint7 {train_parts['joint7']:.4f}) [{row['secs']:.0f}s]"
            )
        history.append(row)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return history


def train_steps(
    model,
    train_examples: list[Example],
    val_examples: list[Example] | None,
    *,
    device: str,
    steps: int,
    batch_size: int = 512,
    lr: float = 2e-4,
    warmup_steps: int = 0,
    weight_decay: float = 1e-4,
    aux_weight: float = AUX_WEIGHT_DEFAULT,
    value_weight: float = VALUE_WEIGHT_DEFAULT,
    value_bootstrap: float = 0.0,
    validate_every: int = 100,
    optimizer_state: dict | None = None,
    restore_best_val: bool = False,
    seed: int = 0,
    precision: str = "fp32",
    cosine_decay: bool = False,
    grad_clip: float = 0.0,
    optimizer_name: str = "adamw",
    batch_getter=None,
    log=print,
) -> tuple[list[dict], dict]:
    """Fixed-budget training on uniform random minibatches from the replay.

    Replaces the epoch loop.  An epoch presents every buffered position once,
    so as the buffer grows old positions are re-presented on every subsequent
    iteration -- run 02 reached ~113 presentations per new position while
    adding only ~18k new examples.  Here the budget is ``steps`` optimizer
    updates regardless of buffer size, which makes training pressure per unit
    of new data an explicit, logged quantity rather than a side effect.

    ``optimizer_state`` carries AdamW moments across self-play iterations.  A
    cold start (``None``) warms the learning rate up over ``warmup_steps``; a
    warm start skips warmup, because re-warming every iteration would just
    reproduce the sawtooth the cosine restart already caused.

    ``restore_best_val`` defaults to *off*.  Run 02 had it unconditionally on
    (via ``train_loop``), so from iteration 3 onward every candidate was the
    epoch-0 weights and the other seven epochs were computed and discarded.
    Validation stays diagnostic here until arena games show it predicts
    strength.

    Returns ``(history, optimizer_state)``.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if not train_examples:
        raise ValueError("train_steps needs at least one training example")
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warm = optimizer_state is not None
    if warm:
        optimizer.load_state_dict(optimizer_state)
        # A resumed state carries the LR that was saved with it; the schedule
        # below is the single source of truth, so re-assert it.
        for group in optimizer.param_groups:
            group["lr"] = lr
    use_amp = device.startswith("cuda")
    _validate_precision(precision)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and precision == "fp32"
    )
    rng = random.Random(seed)
    population = range(len(train_examples))
    best = {"val_total": float("inf"), "step": -1, "state": None}
    history: list[dict] = []
    running: dict[str, float] = {}
    running_grad_norm = 0.0
    # Steps whose gradients overflowed under GradScaler. Their norm is
    # meaningless -- `scaler.step` skips the update entirely -- so they are
    # counted rather than averaged in.
    norm_steps = 0
    overflow_steps = 0
    window_start = time.time()
    window_steps = 0

    def learning_rate(step: int) -> float:
        if not warm and warmup_steps > 0 and step < warmup_steps:
            return lr * min(1.0, (step + 1) / warmup_steps)
        if not cosine_decay:
            return lr
        decay_start = 0 if warm else min(warmup_steps, steps)
        decay_steps = max(1, steps - decay_start)
        progress = min(1.0, max(0.0, (step + 1 - decay_start) / decay_steps))
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    for step in range(steps):
        current_lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        sampled = rng.choices(population, k=batch_size)
        batch = (
            batch_getter(sampled, device)
            if batch_getter is not None
            else collate([train_examples[i] for i in sampled], device)
        )
        optimizer.zero_grad(set_to_none=True)
        with _training_autocast(device, precision):
            outputs = model(batch)
            total, parts = compute_losses(outputs, batch, aux_weight, value_weight, value_bootstrap)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_norm_sq = sum(
            float(parameter.grad.detach().float().norm(2).item()) ** 2
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        # An overflowed step yields inf/nan here. Averaging it in poisons the
        # whole reporting window with inf, which then cannot be serialised --
        # a scaler overflow is a routine AMP event and must not be able to
        # kill a multi-hour run at the logging step.
        step_grad_norm = math.sqrt(grad_norm_sq)
        if math.isfinite(step_grad_norm):
            running_grad_norm += step_grad_norm
            norm_steps += 1
        else:
            overflow_steps += 1
        scaler.step(optimizer)
        scaler.update()
        for key, value in parts.items():
            running[key] = running.get(key, 0.0) + value
        window_steps += 1

        done = step + 1
        if done % validate_every and done != steps:
            continue
        train_parts = {k: v / window_steps for k, v in running.items()}
        row = {
            "step": done,
            "lr": current_lr,
            "train": train_parts,
            "secs": time.time() - window_start,
            "grad_norm": (
                running_grad_norm / norm_steps if norm_steps else None
            ),
            "grad_overflow_steps": overflow_steps,
        }
        running = {}
        running_grad_norm = 0.0
        norm_steps = 0
        overflow_steps = 0
        window_steps = 0
        if val_examples:
            # `value_bootstrap` is deliberately NOT forwarded: validation must
            # score against the real game outcome, not the blended target the
            # arm trains on. Passing it here would make any soft-target run look
            # better by grading itself on its own softened labels.
            val_metrics = evaluate(
                model,
                val_examples,
                device,
                batch_size,
                aux_weight,
                precision=precision,
            )
            row["val"] = val_metrics
            log(
                f"step {done}: train total {train_parts['total']:.4f} "
                f"(policy {train_parts['policy']:.4f} value {train_parts['value']:.4f}) "
                f"| val total {val_metrics['total']:.4f} "
                f"value_acc {val_metrics['value_acc']:.3f} "
                f"joint7_acc {val_metrics['joint7_acc']:.3f} "
                f"policy_top1 {val_metrics['policy_top1']:.3f} "
                f"[{row['secs']:.0f}s]"
            )
            if val_metrics["total"] < best["val_total"] - 1e-4:
                best = {
                    "val_total": val_metrics["total"],
                    "step": done,
                    "state": (
                        {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }
                        if restore_best_val
                        else None
                    ),
                }
        else:
            log(
                f"step {done}: train total {train_parts['total']:.4f} "
                f"(policy {train_parts['policy']:.4f} value {train_parts['value']:.4f} "
                f"joint7 {train_parts['joint7']:.4f}) [{row['secs']:.0f}s]"
            )
        history.append(row)
        window_start = time.time()

    if restore_best_val and best["state"] is not None:
        log(f"restoring best-validation weights from step {best['step']}")
        model.load_state_dict(best["state"])
    return history, optimizer.state_dict()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", nargs="+", required=True)
    parser.add_argument("--model", choices=("transformer", "mlp"), default="transformer")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument(
        "--heads",
        type=int,
        default=None,
        help="attention heads (default: 64 dims per head, floor 4)",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=AUX_WEIGHT_DEFAULT)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--overfit", action="store_true", help="no split, no early stop")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model (falls back with a warning if the "
        "backend is unavailable on this platform)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="model-call precision; bf16 is opt-in",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    records = [record for path in args.buffer for record in read_records(path)]
    print(f"loaded {len(records)} games; featurizing (encoder {ENCODER_SIGNATURE[:12]})")
    examples = examples_from_records(records)
    print(f"{len(examples)} decision states")
    base = baselines(examples)
    print(f"baselines: {json.dumps({k: round(v, 4) for k, v in base.items()})}")

    if args.overfit:
        train_examples, val_examples = examples, None
    else:
        train_examples, val_examples = game_honest_split(examples, args.val_frac)
        print(f"split: {len(train_examples)} train / {len(val_examples)} val states")

    model = build_model(args.model, args.d_model, args.layers, args.heads)
    params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {params:,} params on {args.device}")
    if args.compile:
        # Compilation errors surface lazily at the first forward, so probe a
        # trivial compiled function on the target device before committing.
        # (Verified on this project's Windows box: triton is unavailable there
        # and the probe correctly falls back to eager; compile pays off on the
        # Linux training boxes.)
        try:
            probe = torch.compile(lambda x: x * 2 + 1)
            probe(torch.zeros(4, device=args.device))
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as error:  # backend availability varies by platform
            print(f"torch.compile unavailable, running eager: {type(error).__name__}")
    history = train_loop(
        model,
        train_examples,
        val_examples,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        aux_weight=args.aux_weight,
        patience=args.patience,
        precision=args.precision,
    )
    final = evaluate(
        model,
        val_examples or train_examples,
        args.device,
        args.batch_size,
        args.aux_weight,
        precision=args.precision,
    )
    print(f"final: {json.dumps({k: round(v, 4) for k, v in final.items()})}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        source_model = getattr(model, "_orig_mod", model)
        config = {
            "model": args.model,
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": (
                int(source_model.attention_heads)
                if hasattr(source_model, "attention_heads")
                else None
            ),
            "precision": args.precision,
            "weight_decay": args.weight_decay,
            "aux_weight": args.aux_weight,
        }
        torch.save(make_checkpoint(model, config), out / f"{args.model}.pt")
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "config": config,
                    "baselines": base,
                    "final": final,
                    "history": history[-5:],
                    "encoder_signature": ENCODER_SIGNATURE,
                },
                indent=2,
                default=float,
            )
        )
        print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
