"""Fitted W5 weight: set the served mix from held-out evidence, once per iteration.

The served policy is ``flat + alpha * W5`` over the legal moves, which is the
log-linear pool ``p ∝ p_flat * p_W5 ** alpha``. Letting AdamW train the gate that
sets alpha does not work at this run's scale: Adam moves one scalar by roughly
the learning rate per update whatever the gradient says, so 5e-5 x 190 steps
caps it near 0.01 per iteration, and weight decay pulls it back.

So alpha is not trained. After each iteration's training step it is FITTED:
on held-out positions that carry a search target, both heads' logits are already
known, and the question "which single alpha best predicts the search targets on
positions neither head trained on" is one-dimensional and convex --

    CE(alpha) = mean_i [ logsumexp_j (f_ij + alpha r_ij) - sum_j t_ij (f_ij + alpha r_ij) ]

is a log-sum-exp of functions affine in alpha, so its derivative

    CE'(alpha) = mean_i [ E_{p(alpha)}[r_i] - E_{t}[r_i] ]

is non-decreasing and a bisection on its sign finds the unique minimiser on
``[0, alpha_max]``. No forward pass beyond the one that produces the logits.

The fitted value is then moved toward by at most ``step`` per iteration. The
flat head trains against the COMBINED policy, so it has adapted to the previous
alpha; a jump would leave its compensation stale for an iteration and shift every
prior at once.

What this cannot see, and the reason it is not the promotion test: cross-entropy
against search targets is an average over ordinary moves. The plan's caveat for
W5 is that its gains, if any, sit in rare decisive actions, which this does not
weigh. A sustained alpha above 1 is a trigger to run the W5-only arena
(``policy_source="action"``), not a verdict.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import torch

from .codec import NUM_ACTIONS

#: Below this many held-out policy positions the fit is noise, and the previous
#: alpha is kept rather than chased.
DEFAULT_MIN_POSITIONS = 256
#: The fit's cost is one forward pass over this many positions. Held-out rows
#: beyond it are subsampled deterministically.
DEFAULT_MAX_POSITIONS = 16_384
_BISECTION_STEPS = 60


@dataclass
class AlphaFit:
    """One iteration's fit, as it is logged."""

    previous: float
    applied: float
    fitted: float | None = None
    positions: int = 0
    #: Held-out cross-entropy against the search targets, for reading the fit.
    loss_flat_only: float | None = None
    loss_at_previous: float | None = None
    loss_at_fitted: float | None = None
    loss_w5_only: float | None = None
    skipped: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class HeldOutLogits:
    """Legal-move logits for held-out positions, padded to one width.

    ``flat`` is the flat head alone, ``residual`` the W5 scorer alone, ``target``
    the search policy; padding is marked by ``mask`` (True = a real legal move).
    """

    flat: torch.Tensor
    residual: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor

    @property
    def positions(self) -> int:
        return int(self.flat.shape[0])


def cross_entropy(logits: HeldOutLogits, alpha: float) -> float:
    return float(_cross_entropy(logits.flat + alpha * logits.residual, logits))


def w5_only_cross_entropy(logits: HeldOutLogits) -> float:
    return float(_cross_entropy(logits.residual, logits))


def _cross_entropy(combined: torch.Tensor, logits: HeldOutLogits) -> torch.Tensor:
    masked = combined.masked_fill(~logits.mask, float("-inf"))
    log_z = torch.logsumexp(masked, dim=1)
    expected = (logits.target * combined.masked_fill(~logits.mask, 0.0)).sum(dim=1)
    return (log_z - expected).mean()


def _derivative(logits: HeldOutLogits, alpha: float) -> float:
    combined = (logits.flat + alpha * logits.residual).masked_fill(
        ~logits.mask, float("-inf")
    )
    probabilities = torch.softmax(combined, dim=1)
    residual = logits.residual.masked_fill(~logits.mask, 0.0)
    model_mean = (probabilities * residual).sum(dim=1)
    target_mean = (logits.target * residual).sum(dim=1)
    return float((model_mean - target_mean).mean())


def fit_alpha(logits: HeldOutLogits, alpha_max: float) -> float:
    """The alpha in ``[0, alpha_max]`` minimising held-out cross-entropy."""

    if not math.isfinite(alpha_max) or alpha_max <= 0:
        raise ValueError("alpha_max must be finite and positive")
    if logits.positions == 0:
        raise ValueError("cannot fit alpha on zero positions")
    if _derivative(logits, 0.0) >= 0.0:
        return 0.0
    if _derivative(logits, alpha_max) <= 0.0:
        return float(alpha_max)
    low, high = 0.0, float(alpha_max)
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if _derivative(logits, middle) < 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def step_toward(previous: float, fitted: float, step: float, alpha_max: float) -> float:
    """Move at most ``step`` from ``previous`` toward ``fitted``, inside the range."""

    if not math.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    moved = min(previous + step, max(previous - step, fitted))
    return min(float(alpha_max), max(0.0, moved))


@torch.no_grad()
def collect_held_out_logits(
    model,
    examples,
    device: str,
    *,
    batch_size: int = 512,
    precision: str = "fp32",
    max_positions: int = DEFAULT_MAX_POSITIONS,
    seed: int = 0,
) -> HeldOutLogits:
    """Run the model over held-out policy positions and keep legal-move logits.

    The flat head's logits are recovered as ``policy - alpha * residual`` rather
    than by a second pass with the gate zeroed: one forward, and exact up to
    float rounding because the served policy is literally that sum.
    """

    from .train import _evaluation_autocast, collate, control_table_for

    scorer = getattr(model, "action_scorer", None)
    if scorer is None:
        raise ValueError("fitting alpha needs a model with the W5 action scorer")
    if getattr(model, "policy_source", "combined") != "combined":
        raise ValueError("fit alpha on the combined policy, not an evaluation view")
    policy_examples = [example for example in examples if example.has_policy]
    if len(policy_examples) > max_positions:
        policy_examples = random.Random(seed).sample(policy_examples, max_positions)
    alpha = scorer.alpha_value()
    control_labels = control_table_for(model)
    was_training = model.training
    model.eval()
    parts: dict[str, list[torch.Tensor]] = {"flat": [], "residual": [], "target": [], "mask": []}
    try:
        for start in range(0, len(policy_examples), batch_size):
            batch = collate(
                policy_examples[start : start + batch_size],
                device,
                contextual_actions=True,
                control_table=control_labels,
            )
            with _evaluation_autocast(device, precision):
                outputs = model(batch)
            served = outputs["policy"].float()
            residual = outputs["action_policy"].float()
            flat = served - alpha * residual
            indices = batch["legal_indices"]
            real = ~batch["legal_pad_mask"]
            safe = torch.where(real, indices, torch.zeros_like(indices)).clamp(
                max=NUM_ACTIONS - 1
            )
            target = batch["policy"].float().gather(1, safe) * real
            parts["flat"].append(flat.gather(1, safe).cpu().double())
            parts["residual"].append(residual.gather(1, safe).cpu().double())
            parts["target"].append(target.cpu().double())
            parts["mask"].append(real.cpu())
    finally:
        model.train(was_training)
    if not parts["flat"]:
        empty = torch.zeros((0, 1), dtype=torch.float64)
        return HeldOutLogits(empty, empty, empty, torch.zeros((0, 1), dtype=torch.bool))
    width = max(tensor.shape[1] for tensor in parts["flat"])

    def stack(name: str, fill) -> torch.Tensor:
        return torch.cat(
            [
                torch.nn.functional.pad(tensor, (0, width - tensor.shape[1]), value=fill)
                for tensor in parts[name]
            ]
        )

    target = stack("target", 0.0)
    # Targets are distributions; renormalise defensively so a row whose mass sat
    # partly on a pruned index still scores as a distribution over its legal set.
    target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return HeldOutLogits(
        flat=stack("flat", 0.0),
        residual=stack("residual", 0.0),
        target=target,
        mask=stack("mask", False),
    )


def format_alpha_fit(fit: dict) -> str:
    """One log line for an iteration's fit."""

    if fit.get("skipped"):
        return f"W5 alpha: kept {fit['applied']:.3f} ({fit['skipped']})"
    return (
        f"W5 alpha: fitted {fit['fitted']:.3f} -> applied {fit['applied']:.3f} "
        f"(was {fit['previous']:.3f}) | held-out CE flat {fit['loss_flat_only']:.4f} "
        f"W5 {fit['loss_w5_only']:.4f} mixed {fit['loss_at_fitted']:.4f} "
        f"on {fit['positions']} positions"
    )


def refit_alpha(
    model,
    examples,
    device: str,
    *,
    alpha_max: float,
    step: float,
    batch_size: int = 512,
    precision: str = "fp32",
    min_positions: int = DEFAULT_MIN_POSITIONS,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    seed: int = 0,
) -> AlphaFit:
    """Fit alpha on ``examples`` and apply the capped step to ``model`` in place."""

    scorer = model.action_scorer
    previous = scorer.alpha_value()
    if not math.isclose(scorer.gate_max, alpha_max, rel_tol=1e-9):
        raise ValueError(
            f"model was built with action_gate_max={scorer.gate_max}, but the "
            f"controller fits up to {alpha_max}; alpha could not land where fitted"
        )
    logits = collect_held_out_logits(
        model,
        examples,
        device,
        batch_size=batch_size,
        precision=precision,
        max_positions=max_positions,
        seed=seed,
    )
    if logits.positions < min_positions:
        return AlphaFit(
            previous=previous,
            applied=previous,
            positions=logits.positions,
            skipped=f"{logits.positions} held-out policy positions < {min_positions}",
        )
    fitted = fit_alpha(logits, alpha_max)
    applied = scorer.set_alpha(step_toward(previous, fitted, step, alpha_max))
    return AlphaFit(
        previous=previous,
        applied=applied,
        fitted=fitted,
        positions=logits.positions,
        loss_flat_only=cross_entropy(logits, 0.0),
        loss_at_previous=cross_entropy(logits, previous),
        loss_at_fitted=cross_entropy(logits, fitted),
        loss_w5_only=w5_only_cross_entropy(logits),
    )
