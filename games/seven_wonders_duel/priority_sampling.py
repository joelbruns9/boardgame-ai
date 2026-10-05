"""G3 (`MODEL_GROWTH_PLAN.md`): controlled decisive-pattern sampling.

Recipe (Braun 4.2.4; KataGo's policy-surprise weighting): of every minibatch
draw, ``1 - uniform_share`` (70%) is by PRIORITY and ``uniform_share`` (30%) is
uniform; a row's priority is capped at ``cap`` (2x) the mean; every sampled row
keeps unit loss weight -- priority changes how OFTEN a row is seen, never how
much one presentation counts.

Priority signals, from one no-gradient pass of the model about to train:

* **policy surprise** -- KL(search target || model policy) over the legal set,
  on rows with a policy label;
* **value correction** -- |search root value - model value|, on rows with a
  recorded search value;
* **proof membership** -- solver-proven, certain-win and G1-retained rows are
  pinned at the cap: their labels are exact and rare, which is the case the
  workstream exists for.

Each continuous signal is normalised to mean 1 over the rows that carry it and
the available ones are averaged, so neither dominates by scale.

The pass costs one forward over the training rows per refresh (minutes at
run07's buffer size); `sampling_report` records what it produced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

UNIFORM_SHARE_DEFAULT = 0.3
CAP_DEFAULT = 2.0


@dataclass(frozen=True)
class Signals:
    surprise: np.ndarray  # [N] KL, NaN where the row has no policy label
    value_correction: np.ndarray  # [N] |root - model|, NaN where no root value
    proof: np.ndarray  # [N] bool


def is_proof_row(example) -> bool:
    return (
        example.solver_value is not None
        or bool(getattr(example, "certain_win", False))
        or bool(getattr(example, "retained_proof", False))
    )


@torch.no_grad()
def row_signals(model, examples, device: str, *, batch_size: int = 512) -> Signals:
    """One forward pass of `model` over `examples`; the model is left in the
    mode it was in."""

    from .dataset import collate
    from .net import masked_policy_log_softmax
    from .train import control_table_for

    was_training = model.training
    model.eval()
    control = control_table_for(model)
    contextual = bool(getattr(model, "action_residual", False))
    n = len(examples)
    surprise = np.full(n, np.nan)
    value_correction = np.full(n, np.nan)
    try:
        for start in range(0, n, batch_size):
            chunk = examples[start:start + batch_size]
            batch = collate(chunk, device, contextual_actions=contextual,
                            control_table=control)
            outputs = model(batch)
            log_policy = masked_policy_log_softmax(
                outputs["policy"].float(), batch["legal_mask"]
            )
            target = batch["policy"]
            legal = batch["legal_mask"]
            safe_log = torch.where(legal, log_policy, torch.zeros_like(log_policy))
            safe_target_log = torch.where(
                target > 0, target.clamp_min(1e-12).log(), torch.zeros_like(target)
            )
            kl = (target * (safe_target_log - safe_log)).sum(dim=-1)
            wdl = torch.softmax(outputs["value"].float(), dim=-1)
            model_value = (wdl[:, 0] - wdl[:, 2]).cpu().numpy()
            kl = kl.cpu().numpy()
            for offset, example in enumerate(chunk):
                row = start + offset
                if example.has_policy:
                    surprise[row] = max(0.0, float(kl[offset]))
                if example.root_value is not None:
                    value_correction[row] = abs(float(example.root_value) - float(model_value[offset]))
    finally:
        model.train(was_training)
    proof = np.array([is_proof_row(e) for e in examples], dtype=bool)
    return Signals(surprise, value_correction, proof)


def _normalised(values: np.ndarray) -> np.ndarray:
    present = ~np.isnan(values)
    out = np.full_like(values, np.nan)
    if present.any():
        mean = values[present].mean()
        out[present] = values[present] / mean if mean > 0 else 1.0
    return out


def priorities(signals: Signals, cap: float = CAP_DEFAULT) -> np.ndarray:
    """Per-row priority RELATIVE TO THE MEAN: the average of the normalised
    signals a row carries (1.0 when it carries none), rescaled to mean 1, then
    proof rows raised to `cap`. Normalised before the raise, so the raise does
    not move the mean it is measured against -- doing it the other way round
    left proof rows at ~1.8x instead of the 2x cap."""

    stacked = np.vstack([_normalised(signals.surprise), _normalised(signals.value_correction)])
    counts = (~np.isnan(stacked)).sum(axis=0)
    sums = np.nansum(stacked, axis=0)
    prio = np.where(counts > 0, sums / np.maximum(counts, 1), 1.0)
    mean = prio.mean()
    prio = prio / mean if mean > 0 else np.ones_like(prio)
    return np.where(signals.proof, np.maximum(prio, cap), prio)


def mixture(prio: np.ndarray, *, uniform_share: float = UNIFORM_SHARE_DEFAULT,
            cap: float = CAP_DEFAULT) -> np.ndarray:
    """Sampling probabilities: `uniform_share` uniform plus the rest in
    proportion to `prio` capped at `cap`. `prio` is relative to the mean, as
    `priorities` returns it, so the cap is "`cap` x the mean priority". Sums
    to 1."""

    if not 0.0 <= uniform_share <= 1.0:
        raise ValueError("uniform_share must be in [0, 1]")
    if cap < 1.0:
        raise ValueError("cap must be >= 1 (it is relative to the mean)")
    n = len(prio)
    capped = np.minimum(prio, cap)
    return uniform_share / n + (1.0 - uniform_share) * capped / capped.sum()


def sampling_report(probabilities: np.ndarray, signals: Signals) -> dict:
    """What the mixture does, in presentations relative to uniform."""

    n = len(probabilities)
    relative = probabilities * n
    report = {
        "rows": n,
        # Kish effective sample size, as a share of the rows.
        "effective_share": float(1.0 / (n * (probabilities ** 2).sum())),
        "max_relative": float(relative.max()),
        "proof_rows": int(signals.proof.sum()),
        "proof_relative": float(relative[signals.proof].mean()) if signals.proof.any() else None,
        "surprise_mean": float(np.nanmean(signals.surprise)) if not np.isnan(signals.surprise).all() else None,
        "value_correction_mean": (
            float(np.nanmean(signals.value_correction))
            if not np.isnan(signals.value_correction).all() else None
        ),
    }
    return report


def sample_weights(model, examples, device: str, *,
                   uniform_share: float = UNIFORM_SHARE_DEFAULT,
                   cap: float = CAP_DEFAULT) -> tuple[np.ndarray, dict]:
    """The probabilities `train.train_steps(sample_weights=)` draws by, and
    their report."""

    signals = row_signals(model, examples, device)
    probabilities = mixture(priorities(signals, cap), uniform_share=uniform_share, cap=cap)
    return probabilities, sampling_report(probabilities, signals)
