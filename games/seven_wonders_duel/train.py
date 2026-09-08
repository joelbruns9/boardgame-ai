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

#: Weight on the opponent-reply head. Small on purpose: it adds no information
#: (Q already integrates the reply), only supervision density and pressure on the
#: trunk to encode opponent intent. KataGo's auxiliary targets sit in this range.
REPLY_WEIGHT_DEFAULT = 0.15

# Independent supervision for W5's legal-action scorer. Zero preserves every
# historical training recipe; prototype runs opt in explicitly. This loss is
# required while the served residual gate is exactly zero, because the ordinary
# policy loss can otherwise update only the gate and not the scorer behind it.
CONTROL_WEIGHT_DEFAULT = 1.0
ACTION_POLICY_WEIGHT_DEFAULT = 0.0

#: Weight on W4's hierarchical winner x victory-type head.
#:
#: In the same range as the reply head, and for the same reason: it fits a
#: per-GAME label, so it carries about one independent observation per game
#: however many rows it is asked about. Zero is a legal setting and makes the
#: head a dead read-out -- with `hierarchical_value_detach` the head cannot
#: touch the trunk either way, so zero and detached together mean the module
#: is present, untrained and inert.
HIER_VALUE_WEIGHT_DEFAULT = 0.15

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


def control_table_for(model):
    """The W3 table to label batches with, or None when the head is absent.

    Derived from the MODEL rather than passed per call site: the auxiliary head
    is silently untrainable without labels -- its loss is exactly 0 and no head
    parameter moves -- which is a failure that looks like a working run. Every
    trainer and the validation path go through this.
    """

    if not getattr(model, "control_head", False):
        return None
    from .control_table import default_table

    return default_table()


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    aux_weight: float = AUX_WEIGHT_DEFAULT,
    value_weight: float = VALUE_WEIGHT_DEFAULT,
    value_bootstrap: float = 0.0,
    solver_value_target: bool = True,
    row_weights: bool = True,
    reply_weight: float = REPLY_WEIGHT_DEFAULT,
    action_policy_weight: float = ACTION_POLICY_WEIGHT_DEFAULT,
    control_weight: float = CONTROL_WEIGHT_DEFAULT,
    hier_value_weight: float = HIER_VALUE_WEIGHT_DEFAULT,
    hier_value_replaces_joint7: bool = False,
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
    control_loss = outputs["policy"].new_zeros(())
    if "control_reach_logit" in outputs and "control_reachable" in batch:
        slot_mask = batch["control_slot_valid"].float()
        denominator = slot_mask.sum().clamp(min=1.0)
        reach = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["control_reach_logit"], batch["control_reachable"],
            reduction="none",
        )
        # Distance is meaningless where the slot is unreachable, so it is
        # regressed only where the LABEL says it exists -- never against a
        # sentinel standing in for "no answer".
        distance_mask = slot_mask * batch["control_reachable"]
        distance = (outputs["control_distance_pred"] - batch["control_distance"]) ** 2
        control_loss = (
            (reach * slot_mask).sum() / denominator
            + (distance * distance_mask).sum() / distance_mask.sum().clamp(min=1.0)
        )
    action_policy_loss = outputs["policy"].new_zeros(())
    if "action_policy" in outputs:
        action_log = masked_policy_log_softmax(
            outputs["action_policy"], batch["legal_mask"]
        )
        safe_action_log = torch.where(
            batch["legal_mask"], action_log, torch.zeros_like(action_log)
        )
        per_action_row = -(batch["policy"] * safe_action_log).sum(dim=-1)
        if has_policy.any():
            weights = policy_w[has_policy]
            action_policy_loss = (
                per_action_row[has_policy] * weights
            ).sum() / weights.sum().clamp(min=1e-9)
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
    # The REPLACEMENT arm. `joint7` and W4's head fit the same per-game label,
    # so running both trains two heads on one observation and mostly re-weights
    # the outcome objective against policy -- which measures the weight, not the
    # parameterisation. Dropping the flat term makes the comparison the one
    # worth making: same information, same weight, structured or not.
    #
    # The flat head's PARAMETERS are then frozen wherever they were inherited,
    # and its outputs go stale. Nothing in search reads them, but the advisor
    # does, which is why this is recorded in the checkpoint config rather than
    # left for a reader to infer.
    reported_joint7 = float(joint7_loss.detach())
    if hier_value_replaces_joint7:
        joint7_loss = joint7_loss.new_zeros(())
    margin_valid = batch["margin_valid"]
    if margin_valid.any():
        margin_loss = F.mse_loss(
            outputs["margin"][margin_valid], batch["margin"][margin_valid]
        )
    else:
        margin_loss = outputs["margin"].new_zeros(())
    military_loss = F.mse_loss(outputs["military"], batch["military_final"])
    science_loss = F.mse_loss(outputs["science"], batch["sci_final"])
    hier_value_loss = outputs["policy"].new_zeros(())
    if "hier_joint7" in outputs:
        # ONE term, not two. The head emits log P(outcome) + log P(type|outcome)
        # already summed into the seven joint classes, so the negative
        # log-likelihood of the true class trains the outcome factor and the
        # conditional factor together, weighted exactly as the data weights
        # them. Fitting the marginal separately would double-count the rows and
        # let the two factors disagree, which is the defect this head exists to
        # remove.
        hier_value_loss = F.nll_loss(outputs["hier_joint7"], batch["joint7"])
    reply_loss = outputs["policy"].new_zeros(())
    if "reply" in outputs and batch.get("has_reply") is not None:
        rows = batch["has_reply"]
        if rows.any():
            # Same masked path as the policy head, so an illegal action cannot
            # turn 0 * -inf into a NaN. The mask is the REPLY's legal set, which
            # belongs to a different position than `legal_mask`.
            reply_log = masked_policy_log_softmax(
                outputs["reply"], batch["reply_mask"]
            )
            safe_reply = torch.where(
                batch["reply_mask"], reply_log, torch.zeros_like(reply_log)
            )
            per_reply = -(batch["reply"] * safe_reply).sum(dim=-1)
            reply_loss = per_reply[rows].mean()
    total = (
        policy_loss
        + action_policy_weight * action_policy_loss
        + control_weight * control_loss
        + value_weight * value_loss
        + value_weight
        * aux_weight
        * (joint7_loss + margin_loss + military_loss + science_loss)
        + reply_weight * reply_loss
        + hier_value_weight * hier_value_loss
    )
    return total, {
        "total": float(total.detach()),
        "policy": float(policy_loss.detach()),
        "action_policy": float(action_policy_loss.detach()),
        "control": float(control_loss.detach()),
        "value": float(value_loss.detach()),
        "joint7": reported_joint7,
        "margin": float(margin_loss.detach()),
        "military": float(military_loss.detach()),
        "science": float(science_loss.detach()),
        "reply": float(reply_loss.detach()),
        "hier_value": float(hier_value_loss.detach()),
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
    action_policy_weight: float = ACTION_POLICY_WEIGHT_DEFAULT,
    control_weight: float = CONTROL_WEIGHT_DEFAULT,
    hier_value_weight: float = HIER_VALUE_WEIGHT_DEFAULT,
    hier_value_replaces_joint7: bool = False,
):
    model.eval()
    control_labels = control_table_for(model)
    sums: dict[str, float] = {}
    correct = {"value": 0, "joint7": 0, "policy_top1": 0}
    abs_err = {"margin": 0.0, "military": 0.0, "science": 0.0}
    margin_rows = 0
    policy_rows = 0
    count = 0
    for start in range(0, len(examples), batch_size):
        batch = collate(
            examples[start : start + batch_size], device,
            contextual_actions=bool(getattr(model, "action_residual", False)),
            control_table=control_labels,
        )
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
                control_weight=control_weight,
                hier_value_weight=hier_value_weight,
                hier_value_replaces_joint7=hier_value_replaces_joint7,
                solver_value_target=False,
                # Unweighted for the same reason: a held-out number has to mean
                # the same thing across runs. Upweighting solved rows in
                # validation would make the metric easier exactly where training
                # was, which is how a run posts a better number without playing
                # better.
                row_weights=False,
                action_policy_weight=action_policy_weight,
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


# Architecture switches that change which parameters exist. A checkpoint whose
# config omits one cannot rebuild its own weights, so they are read off the model
# rather than trusted from the caller's dict.
ARCHITECTURE_SWITCHES = (
    "pooled_readout",
    "reply_head",
    "action_residual",
    "action_exposes",
    "control_head",
    "slot_embedding",
    "graph_module",
    "hierarchical_value",
    # Not a presence flag but an architecture fact all the same: it decides
    # whether the head's loss reaches the shared trunk, so two checkpoints with
    # identical parameters can have been trained on different objectives.
    "hierarchical_value_detach",
)


def model_from_config(config: dict, *, name: str = "transformer", **fallbacks):
    """Build the model a checkpoint's config describes.

    ONE place that knows which config keys are architecture. Every reader that
    rebuilds a saved model should come through here, because the failure mode is
    not a crash at the call site -- it is a model that loads and computes
    something else, or a strict load that fails hours into a run.

    Adding `pooled_readout` / `reply_head` to the model left SIX separate
    rebuild sites constructing a model the weights no longer fit, each of which
    had enumerated by hand the config keys it happened to know about. Adding a
    switch now means adding it here, once.

    `fallbacks` supplies `d_model` / `layers` for configs that omit them. A
    config with neither is refused by name rather than dying in `int(None)`,
    which reads as a bug in this helper rather than as a checkpoint that does
    not describe its own width.
    """

    missing = [
        key
        for key in ("d_model", "layers")
        if config.get(key, fallbacks.get(key)) is None
    ]
    if missing:
        raise ValueError(
            f"checkpoint config has no {', '.join(missing)} and no fallback was "
            "supplied; the model cannot be rebuilt from it"
        )
    # Named explicitly rather than left to `int(None)`, whose TypeError names
    # neither the field nor the checkpoint.
    for field in ("d_model", "layers"):
        if config.get(field, fallbacks.get(field)) is None:
            raise ValueError(
                f"checkpoint config has no {field!r} and no fallback was given; "
                "cannot rebuild the model"
            )
    return build_model(
        name,
        int(config.get("d_model", fallbacks.get("d_model"))),
        int(config.get("layers", fallbacks.get("layers"))),
        heads_from_config(config),
        pooled_readout_from_config(config),
        reply_head_from_config(config),
        action_residual_from_config(config),
        control_head_from_config(config),
        slot_embedding_from_config(config),
        graph_module_from_config(config),
        action_exposes=action_exposes_from_config(config),
        **graph_shape_from_config(config),
        **hierarchical_value_from_config(config),
    )


def make_checkpoint(model, config: dict) -> dict:
    """Package weights with the config needed to rebuild them.

    The architecture switches are derived from the MODEL, not taken from
    ``config``. Four separate call sites assembled that dict by hand, and adding
    ``pooled_readout`` / ``reply_head`` to the model left three of them behind --
    each producing a checkpoint that saved ``readout_proj`` weights alongside a
    config denying they existed, which the strict reload then rejected. Deriving
    here ends the class of bug: a caller can no longer forget a switch, because
    it never had the chance to state one.

    A caller that does state one must agree with the model, or the checkpoint
    would misdescribe its own weights.
    """

    model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    config = dict(config)
    for switch in ARCHITECTURE_SWITCHES:
        actual = bool(getattr(model, switch, False))
        stated = config.get(switch)
        if stated is not None and bool(stated) != actual:
            raise ValueError(
                f"checkpoint config says {switch}={stated!r} but the model was "
                f"built with {switch}={actual!r}; the checkpoint would not be "
                "able to rebuild its own weights"
            )
        config[switch] = actual
    # The W2 shape is architecture too, and equally derived from the model: a
    # config that named a different `graph_alpha` than the weights were trained
    # under would rebuild a net that computes something else.
    if bool(getattr(model, "graph_module", False)):
        for field in GRAPH_SHAPE_DEFAULTS:
            actual = getattr(model, field)
            stated = config.get(field)
            if stated is not None and type(actual)(stated) != actual:
                raise ValueError(
                    f"checkpoint config says {field}={stated!r} but the model "
                    f"was built with {field}={actual!r}; the checkpoint would "
                    "not be able to rebuild its own weights"
                )
            config[field] = actual
    # A TRAINING-recipe fact, not an architecture one, so it is carried through
    # rather than derived from the model: it shapes no parameter, but it changes
    # what the flat `joint7` head MEANS -- under that arm the head is frozen
    # wherever it was inherited and its outputs are stale, which a reader of the
    # checkpoint has to be able to see.
    out = {
        "model_state": model.state_dict(),
        "config": config,
        "encoder_signature": ENCODER_SIGNATURE,
    }
    # Only when the model actually consumes control features. A model without
    # them is not tied to any table, and recording one would invent a constraint
    # that later rejects a perfectly loadable checkpoint.
    if _reads_control_features(model):
        from .control_table import table_content_digest
        from .encoder import control_features_enabled

        # Which ARM this is. Both arms share a signature and a width by design,
        # so without this a baseline and an inputs model are indistinguishable
        # on disk -- and a result could be attributed to the wrong one.
        out["control_features"] = "on" if control_features_enabled() else "off"
        if control_features_enabled():
            out["control_table_digest"] = table_content_digest()
    # Which REVEAL arm this is, on the same argument and unconditionally: the
    # channels are in the schema whether or not they are switched on, so a file
    # that does not say which arm trained it is indistinguishable on disk from
    # the other arm -- and the default is OFF, so an unmarked reveal-arm model
    # would be served zeros in channels it was trained to read.
    from .reveal_risk import reveal_features_enabled

    out["reveal_features"] = "on" if reveal_features_enabled() else "off"
    return out


def _reads_control_features(model) -> bool:
    """Does this model depend on the control table's CONTENTS?

    True for the auxiliary head, and true for any model whose encoder inputs
    include the control channels -- which, once the schema carries them, is every
    model trained after W3.
    """

    from .encoder import CONTROL_FEATURES, TABLEAU_FEATURES

    return bool(set(CONTROL_FEATURES) & set(TABLEAU_FEATURES))


#: Parameters whose ZERO value is the designed switch-neutral start, not a
#: reset for want of a counterpart.
#:
#: The W1 slot table adds nothing to the token sequence, so at zero the model
#: computes bit-for-bit what it did before -- stronger than the near-neutrality
#: an appended token type gets, since there is no extra token to dilute
#: attention normalization. And unlike a zeroed MLP it still trains: a lookup's
#: gradient does not pass through its own value.
NEUTRAL_ZERO_PARAMETERS = frozenset({"embedder.slot.weight"})


def migrate_state_dict(old_state: dict, model) -> dict:
    """Additive-schema warm start (spec §5.8a): load every parameter that still
    matches, zero-initialize parameters with no counterpart (new token types'
    entity embeddings and feature projections), and zero-pad grown embedding
    tables (the type-embedding table when a type is appended). New W5 action
    scorer parameters retain their ordinary random initialization behind their
    exactly-zero gate; zeroing every layer of a residual MLP would destroy the
    gradients the independent action-policy loss is meant to train.

    ``NEUTRAL_ZERO_PARAMETERS`` are also zeroed, but reported apart: their zero
    is the designed switch-neutral start rather than a lost counterpart, so a
    reader that refuses a partly-random migration should still accept them.

    Zero-init makes the new tokens' pre-activation contribution exactly zero.
    Note the honest caveat (also in the spec): zero-VALUE tokens still
    participate in attention normalization, so switch-on is near-neutral, not
    bit-neutral — exact bit-neutrality requires masking the new type until
    enabled. Returns a report of what was loaded, grown, normally initialized,
    or zero-initialized.
    """

    new_state = model.state_dict()
    removed = sorted(set(old_state) - set(new_state))
    if removed:
        preview = ", ".join(removed[:5])
        suffix = " ..." if len(removed) > 5 else ""
        raise ValueError(
            "checkpoint migration is additive only; the target would remove "
            f"{len(removed)} parameter(s): {preview}{suffix}"
        )
    report = {
        "loaded": [],
        "grown": [],
        "initialized": [],
        "zeroed": [],
        # Zeroed BY DESIGN rather than for want of a counterpart. Kept apart
        # because readers act on the difference: `phase_e` refuses a migration
        # that zeroed anything, on the grounds that the net is then partly
        # random, which is exactly untrue of these.
        "neutral": [],
    }
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
        elif key == "action_scorer.exposed.weight":
            # W5b's OUTPUT projection, zero by design: switching the branch on
            # reproduces the W5a scorer exactly, and the branch still trains,
            # because a zeroed output projection receives gradient where a
            # zeroed gate over the whole branch would not.
            new_state[key] = torch.zeros_like(tensor)
            report["initialized"].append(key)
        elif key.startswith(
            ("action_scorer.", "control_scorer.", "graph.", "hier_value.")
        ):
            # ``tensor`` is the new model's initialized value. In particular,
            # action_scorer.gate was constructed as exact zero while the layers
            # behind it retain symmetry-breaking initialization.
            #
            # The W2 graph module is exempt for the same reason as W5's
            # scorer, and it is the sharper case: zeroing it is not a soft
            # start but a permanent one. A zeroed LayerNorm emits zeros
            # whatever it is fed, so every downstream activation, and every
            # gradient that would revive them, is zero as well. Its neutrality
            # comes from `graph_alpha`, which is a gate outside the state dict.
            #
            # The W3 control head is exempt for the same reason and one more:
            # zeroing a plain MLP is not a soft start, it is a dead one. With
            # every weight and bias zero, GELU(0) = 0, so the second layer sees
            # only zeros and BOTH layers' gradients are exactly zero -- the head
            # could never train. It needs no gate because it feeds no shared
            # output: policy and value are bit-identical whether it is present
            # or not, so ordinary initialization is already switch-neutral.
            report["initialized"].append(key)
        elif key in NEUTRAL_ZERO_PARAMETERS:
            new_state[key] = torch.zeros_like(tensor)
            report["neutral"].append(key)
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
    signature_changed = checkpoint["encoder_signature"] != ENCODER_SIGNATURE
    model_state = model.state_dict()
    stored_state = checkpoint["model_state"]
    architecture_changed = (
        set(model_state) != set(stored_state)
        or any(
            key not in stored_state or stored_state[key].shape != tensor.shape
            for key, tensor in model_state.items()
        )
    )
    if signature_changed or architecture_changed:
        if not migrate:
            detail = (
                "encoder signature changed since this model was trained"
                if signature_changed
                else "model architecture differs from the checkpoint"
            )
            raise ValueError(
                f"checkpoint migration required — {detail} "
                "(pass migrate=True for an additive warm start)"
            )
        checkpoint["migration"] = migrate_state_dict(stored_state, model)
        _check_control_table(checkpoint, migrating=True)
        return checkpoint
    model.load_state_dict(stored_state)
    _check_control_table(checkpoint, migrating=False)
    return checkpoint


def _check_control_table(checkpoint: dict, *, migrating: bool) -> None:
    """Refuse a checkpoint trained against a DIFFERENT control table.

    The encoder signature pins the feature-name schema; it says nothing about
    the cells behind those names. Serving a model against a regenerated table is
    silent: the inputs keep their names and change their meaning, and the model
    is simply worse for reasons nothing reports.

    A checkpoint predating the digest carries none, and is accepted -- its
    signature already differs, so it cannot reach the non-migrating path.
    """

    # Arm mismatch is not fatal -- a baseline checkpoint is perfectly loadable
    # under either setting -- but serving it under the other arm silently
    # changes what the model is shown, so it is surfaced.
    arm = checkpoint.get("control_features")
    if arm is not None:
        from .encoder import control_features_enabled

        live = "on" if control_features_enabled() else "off"
        if arm != live:
            checkpoint["control_features_mismatch"] = {
                "trained_with": arm, "loaded_with": live,
            }

    # Same for the reveal arm. Surfaced rather than fatal for the same reason:
    # a model is loadable under either setting, and serving it under the other
    # one is a silent change in what it is shown, not an error.
    reveal_arm = checkpoint.get("reveal_features")
    if reveal_arm is not None:
        from .reveal_risk import reveal_features_enabled

        live = "on" if reveal_features_enabled() else "off"
        if reveal_arm != live:
            checkpoint["reveal_features_mismatch"] = {
                "trained_with": reveal_arm, "loaded_with": live,
            }

    recorded = checkpoint.get("control_table_digest")
    if recorded is None:
        return
    from .control_table import table_content_digest

    current = table_content_digest()
    if recorded == current:
        return
    if migrating:
        # A migration is already an explicit "this model and this code differ".
        # Record the mismatch rather than blocking a deliberate warm start.
        checkpoint.setdefault("migration", {})["control_table_changed"] = {
            "trained_with": recorded,
            "loaded_with": current,
        }
        return
    raise ValueError(
        "checkpoint was trained against control table "
        f"{recorded[:12]} but {current[:12]} is installed; the control features "
        "keep their names and change their meaning. Restore the matching table "
        "or migrate deliberately."
    )


def build_model(
    name: str,
    d_model: int,
    layers: int,
    heads: int | None = None,
    pooled_readout: bool = False,
    reply_head: bool = False,
    action_residual: bool = False,
    control_head: bool = False,
    slot_embedding: bool = False,
    graph_module: bool = False,
    graph_layers: int = 2,
    graph_bases: int = 4,
    graph_alpha: float = 1e-3,
    hierarchical_value: bool = False,
    hierarchical_value_detach: bool = True,
    # APPENDED, not inserted. `model_from_config` and three phase_d sites pass
    # the earlier arguments POSITIONALLY, so a parameter added in the middle
    # silently receives `slot_embedding` and every later flag shifts by one --
    # a model that builds cleanly and is not the one asked for.
    action_exposes: bool = False,
):
    """Build a model. ``heads=None`` derives the width-appropriate head count.

    Rebuilding a *saved* model must pass the head count its checkpoint recorded
    (`heads_from_config`), never the derived default: attention parameter shapes
    are head-count independent, so a mismatch loads cleanly and silently
    computes something else.
    """

    if name == "transformer":
        return SWDNet(
            d_model=d_model,
            layers=layers,
            heads=heads,
            pooled_readout=pooled_readout,
            reply_head=reply_head,
            action_residual=action_residual,
            action_exposes=action_exposes,
            control_head=control_head,
            slot_embedding=slot_embedding,
            graph_module=graph_module,
            graph_layers=graph_layers,
            graph_bases=graph_bases,
            graph_alpha=graph_alpha,
            hierarchical_value=hierarchical_value,
            hierarchical_value_detach=hierarchical_value_detach,
        )
    if name == "mlp":
        return SWDMlp(d_model=d_model)
    raise ValueError(f"unknown model: {name}")


def reply_head_from_config(config: dict) -> bool:
    """Reply-head presence for rebuilding a checkpoint, honouring older files."""

    return bool(config.get("reply_head", False))


def action_residual_from_config(config: dict) -> bool:
    """W5 scorer presence, false for every checkpoint predating the prototype."""

    return bool(config.get("action_residual", False))


def control_head_from_config(config: dict) -> bool:
    """W3 auxiliary control head, false for every checkpoint predating it.

    Append-only and default-off, so an inherited checkpoint rebuilds as exactly
    the model it was saved as."""

    return bool(config.get("control_head", False))


def slot_embedding_from_config(config: dict) -> bool:
    """W1 learned Age/slot embedding, false for every checkpoint predating it.

    Append-only and default-off, so an inherited checkpoint rebuilds as exactly
    the model it was saved as -- and, because the table is zero-initialized,
    switching it ON by migration reproduces that model's outputs bit for bit
    until training moves it."""

    return bool(config.get("slot_embedding", False))


def hierarchical_value_from_config(config: dict) -> dict:
    """W4 head presence and its gradient path, false for older checkpoints.

    `detach` defaults TRUE, matching the model, so a config that names the head
    without naming its gradient path rebuilds the safe arm rather than the one
    that can move the trunk.
    """

    if not config.get("hierarchical_value", False):
        return {"hierarchical_value": False}
    return {
        "hierarchical_value": True,
        "hierarchical_value_detach": bool(
            config.get("hierarchical_value_detach", True)
        ),
    }


def action_exposes_from_config(config: dict) -> bool:
    """W5b's uncovering edge, false for every checkpoint predating it."""

    return bool(config.get("action_exposes", False))


def graph_module_from_config(config: dict) -> bool:
    """W2 tableau graph, false for every checkpoint predating it."""

    return bool(config.get("graph_module", False))


#: Shape and gate of the W2 module. Not booleans, so they travel beside
#: `ARCHITECTURE_SWITCHES` rather than in it -- but they are architecture all
#: the same: `graph_layers` and `graph_bases` decide which parameters exist,
#: and `graph_alpha` decides what the model COMPUTES with them. A rebuild that
#: dropped alpha would load every weight and silently serve a different net,
#: which is the failure `heads` is documented for.
GRAPH_SHAPE_DEFAULTS = {
    "graph_layers": 2,
    "graph_bases": 4,
    "graph_alpha": 1e-3,
}


def graph_shape_from_config(config: dict) -> dict:
    if not graph_module_from_config(config):
        return {}
    return {
        "graph_layers": int(config.get("graph_layers", GRAPH_SHAPE_DEFAULTS["graph_layers"])),
        "graph_bases": int(config.get("graph_bases", GRAPH_SHAPE_DEFAULTS["graph_bases"])),
        "graph_alpha": float(config.get("graph_alpha", GRAPH_SHAPE_DEFAULTS["graph_alpha"])),
    }


def pooled_readout_from_config(config: dict) -> bool:
    """Readout mode for rebuilding a checkpoint, honouring pre-flag files.

    Same hazard as `heads_from_config`: absent the flag a rebuild would default
    to the CLS-only readout and load a pooled checkpoint with `readout_proj`
    missing, so the weights would be there and the computation would not.
    """

    return bool(config.get("pooled_readout", False))


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
    # Matches :func:`train_steps`. Its absence here was a NameError on every
    # call -- the body has read `optimizer_name` since the Adam/AdamW split was
    # introduced, so this entry point could not run at all.
    optimizer_name: str = "adamw",
    action_policy_weight: float = ACTION_POLICY_WEIGHT_DEFAULT,
    control_weight: float = CONTROL_WEIGHT_DEFAULT,
    hier_value_weight: float = HIER_VALUE_WEIGHT_DEFAULT,
    hier_value_replaces_joint7: bool = False,
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
    control_labels = control_table_for(model)
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
            batch = collate(
                train_examples[start : start + batch_size], device,
                contextual_actions=bool(getattr(model, "action_residual", False)),
                control_table=control_labels,
            )
            optimizer.zero_grad(set_to_none=True)
            with _training_autocast(device, precision):
                outputs = model(batch)
                total, parts = compute_losses(
                    outputs,
                    batch,
                    aux_weight,
                    value_weight,
                    value_bootstrap,
                    action_policy_weight=action_policy_weight,
                    control_weight=control_weight,
                    hier_value_weight=hier_value_weight,
                    hier_value_replaces_joint7=hier_value_replaces_joint7,
                )
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
                action_policy_weight=action_policy_weight,
                # The SAME objective the step optimised. Validation selects the
                # early stop and the restored best checkpoint, so grading it
                # under a different recipe -- weight 0.15, replacement off --
                # picks the winner of a race nobody ran. In the replacement arm
                # it graded the stale flat head.
                hier_value_weight=hier_value_weight,
                hier_value_replaces_joint7=hier_value_replaces_joint7,
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
    action_policy_weight: float = ACTION_POLICY_WEIGHT_DEFAULT,
    control_weight: float = CONTROL_WEIGHT_DEFAULT,
    hier_value_weight: float = HIER_VALUE_WEIGHT_DEFAULT,
    hier_value_replaces_joint7: bool = False,
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
    control_labels = control_table_for(model)
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
            else collate(
                [train_examples[i] for i in sampled], device,
                contextual_actions=bool(getattr(model, "action_residual", False)),
                control_table=control_labels,
            )
        )
        optimizer.zero_grad(set_to_none=True)
        with _training_autocast(device, precision):
            outputs = model(batch)
            total, parts = compute_losses(
                outputs,
                batch,
                aux_weight,
                value_weight,
                value_bootstrap,
                action_policy_weight=action_policy_weight,
                control_weight=control_weight,
                hier_value_weight=hier_value_weight,
                hier_value_replaces_joint7=hier_value_replaces_joint7,
            )
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
                action_policy_weight=action_policy_weight,
                # The SAME objective the step optimised. Validation selects the
                # early stop and the restored best checkpoint, so grading it
                # under a different recipe -- weight 0.15, replacement off --
                # picks the winner of a race nobody ran. In the replacement arm
                # it graded the stale flat head.
                hier_value_weight=hier_value_weight,
                hier_value_replaces_joint7=hier_value_replaces_joint7,
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


def _build_arg_parser() -> argparse.ArgumentParser:
    """Every offline-trainer flag. Constructed here so `build_arg_parser`
    can hand it to a test without entering `main`."""

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
    parser.add_argument(
        "--action-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable the W5a contextual legal-action residual",
    )
    parser.add_argument(
        "--slot-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable the W1 learned Age/slot embedding (zero-initialized, so "
        "a warm start reproduces the inherited model until it trains)",
    )
    parser.add_argument(
        "--hierarchical-value",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="build the W4 hierarchical winner x victory-type head (shadow "
        "only; `value` and `joint7` stay authoritative)",
    )
    parser.add_argument(
        "--hierarchical-value-detach",
        action=argparse.BooleanOptionalAction,
        # UNSET, not True: this and the weight live outside the state dict, so
        # a warm start cannot recover them by loading, and argparse cannot
        # otherwise tell "omitted" from "passed the default". Omitting it used
        # to be indistinguishable from asking for detached, so an explicit
        # --no-hierarchical-value-detach was overwritten by the inherited value
        # AFTER validation had already approved the combination -- producing
        # exactly the replacement-with-a-detached-head this refuses.
        default=None,
        help="learn the W4 head from a stop-gradient readout, so it cannot "
        "move a trunk weight. --no-hierarchical-value-detach is the arm that "
        "lets it shape representations, and the only one that can change "
        "playing strength in either direction",
    )
    parser.add_argument(
        "--hier-value-replaces-joint7",
        action="store_true",
        help="drop the flat joint7 loss so W4 is the only victory-type "
        "supervision. The comparison worth making -- running both trains two "
        "heads on one per-game label and measures the weight, not the "
        "parameterisation. Leaves the flat head's outputs stale.",
    )
    parser.add_argument(
        "--hier-value-weight",
        type=float,
        default=None,
        help="loss weight for the W4 head (default 0.15 when the head is on, "
        "else 0; a present-but-unweighted head is throughput buying nothing)",
    )
    parser.add_argument(
        "--graph-module",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable the W2 tableau graph module",
    )
    # Defaulted to None, not to the value: the gate and the shape live outside
    # the state dict, so a warm start cannot recover them by loading weights,
    # and argparse cannot otherwise tell "omitted" from "passed the default".
    # An omitted flag therefore INHERITS; an explicit one overrides, including
    # an explicit zero.
    parser.add_argument("--graph-layers", type=int, default=None)
    parser.add_argument("--graph-bases", type=int, default=None)
    parser.add_argument(
        "--graph-alpha",
        type=float,
        default=None,
        help="W2 residual gate (default 1e-3 for a new graph, else inherited); "
        "exactly 0 is inert and ungradiented, so it is the ablation setting "
        "rather than a training one",
    )
    parser.add_argument(
        "--action-exposes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="W5b: let each action reach the contextual tokens of the slots it "
        "uncovers. Requires --action-residual; zero-initialised, so switching "
        "it on reproduces the W5a scorer until it trains.",
    )
    parser.add_argument(
        "--action-policy-weight",
        type=float,
        default=ACTION_POLICY_WEIGHT_DEFAULT,
        help="independent policy loss for the W5 scorer (zero is off)",
    )
    parser.add_argument(
        "--action-residual-only",
        action="store_true",
        help="freeze the inherited network and train only the W5 scorer "
        "(plus its gate when --train-action-gate is set)",
    )
    parser.add_argument(
        "--train-action-gate",
        action="store_true",
        help="allow the bounded W5 gate to change served policy logits",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="warm-start from this checkpoint; its architecture is preserved",
    )
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
    return parser

def resolve_graph_args(args, stored: dict | None) -> None:
    """Settle the W2 shape and gate, in place, and refuse the ambiguous case.

    The three graph flags default to None rather than to a value, because they
    live OUTSIDE the state dict: loading every weight successfully does not
    restore them, so argparse's inability to distinguish "omitted" from "passed
    the default" is not cosmetic here. An omitted flag used to overwrite a
    warm-started checkpoint's saved gate with 1e-3, which silently turned a
    saved zero-gate ablation on before the first training step.

    * With an inherited graph, the SHAPE is the shape its weights have. A flag
      that disagrees is refused rather than resolved in either direction:
      obeying it would fail to load, and ignoring it would run an arm the
      operator did not ask for.
    * The GATE is a knob a warm start may legitimately turn -- it is the
      ablation control -- but only when asked. Omitted inherits.
    * Anything still unset falls back to the fresh-graph defaults.

    Called twice: once with the warm-start config, once with None to fill the
    remainder. It is idempotent, which is what makes that safe.
    """

    if stored is not None and graph_module_from_config(stored):
        inherited = graph_shape_from_config(stored)
        for field in ("graph_layers", "graph_bases"):
            asked = getattr(args, field)
            if asked is not None and asked != inherited[field]:
                raise SystemExit(
                    f"--{field.replace('_', '-')} {asked} disagrees with the "
                    f"warm-start checkpoint's {inherited[field]}; its graph "
                    "weights have that shape, so the run must either inherit "
                    "it or start from scratch"
                )
            setattr(args, field, inherited[field])
        if args.graph_alpha is None:
            args.graph_alpha = inherited["graph_alpha"]
        return
    for field, fallback in GRAPH_SHAPE_DEFAULTS.items():
        if getattr(args, field) is None:
            setattr(args, field, fallback)


def matched_hier_value_weight(value_weight: float, aux_weight: float) -> float:
    """The weight that makes W4 a REPLACEMENT for the flat joint7 term.

    `joint7` enters the total at `value_weight * aux_weight`; W4 enters at
    `hier_value_weight`. The two losses are the same functional -- the negative
    log-likelihood of the same true class under a seven-way distribution,
    differing only in how the distribution is parameterised -- so their
    gradients are directly comparable and equal coefficients really do mean
    equal weight. That is what makes the replacement arm a clean test of the
    PARAMETERISATION rather than of the weight.

    Derived, never a constant: `--aux-weight` and `--value-weight` are run
    knobs, so a hard-coded 0.2 would silently stop matching the moment either
    moved.
    """

    return value_weight * aux_weight


def resolve_hier_value_args(args) -> None:
    """Settle the W4 recipe, in place, and refuse the combinations that lie.

    Called AFTER any warm-start inheritance, which is the whole point: the
    first version validated first and inherited afterwards, so an explicit
    request to attach a previously detached head passed validation and was then
    silently reverted -- leaving replacement enabled with a detached head, the
    exact configuration validation exists to prohibit. Resolving before
    validating makes that unrepresentable.

    Both fields default to None from the parser so "omitted" is distinguishable
    from "passed the default"; they live outside the state dict and cannot be
    recovered by loading weights.
    """

    if args.hierarchical_value_detach is None:
        args.hierarchical_value_detach = True
    if args.hier_value_weight is None:
        if args.hier_value_replaces_joint7:
            # The replacement arm holds the weight fixed and varies only the
            # structure, so it takes the coefficient it is replacing.
            args.hier_value_weight = matched_hier_value_weight(
                getattr(args, "value_weight", VALUE_WEIGHT_DEFAULT),
                args.aux_weight,
            )
        else:
            args.hier_value_weight = (
                HIER_VALUE_WEIGHT_DEFAULT if args.hierarchical_value else 0.0
            )
    elif args.hier_value_replaces_joint7:
        matched = matched_hier_value_weight(
            getattr(args, "value_weight", VALUE_WEIGHT_DEFAULT), args.aux_weight
        )
        if abs(args.hier_value_weight - matched) > 1e-12:
            # Warned, not refused: a deliberate sweep of the replacement arm's
            # weight is a legitimate experiment. But shipping a mismatch by
            # accident makes the arm vary structure AND weight, which is the
            # confound replacement exists to remove, so it cannot pass quietly.
            print(
                f"WARNING: --hier-value-weight {args.hier_value_weight} does not "
                f"match the flat joint7 coefficient it replaces ({matched} = "
                f"value_weight x aux_weight). The replacement arm then varies "
                "the weight as well as the parameterisation, and a difference "
                "cannot be attributed to either."
            )
    if args.hier_value_weight < 0 or not math.isfinite(args.hier_value_weight):
        raise SystemExit("--hier-value-weight must be finite and non-negative")
    if args.hier_value_weight > 0 and not args.hierarchical_value:
        raise SystemExit("--hier-value-weight requires --hierarchical-value")
    if args.hierarchical_value and args.hier_value_weight == 0:
        raise SystemExit(
            "--hierarchical-value requires a positive --hier-value-weight; the "
            "head is shadow-only, so an unweighted one is parameters and "
            "throughput buying nothing"
        )
    if args.hier_value_replaces_joint7:
        if not args.hierarchical_value:
            raise SystemExit(
                "--hier-value-replaces-joint7 requires --hierarchical-value"
            )
        if args.hierarchical_value_detach:
            # Replacing the flat term with a detached head would remove the
            # trunk's ONLY victory-type supervision and put nothing back --
            # not an arm, a deletion.
            raise SystemExit(
                "--hier-value-replaces-joint7 requires "
                "--no-hierarchical-value-detach; a detached head cannot replace "
                "supervision it never delivers to the trunk"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    """The offline trainer's parser, separated so flag resolution is testable.

    A run's architecture is decided here and in `resolve_graph_args`; both had
    to be reachable without also reaching the trainer, which has its own
    pre-existing launch defect.
    """

    return _build_arg_parser()


def main(argv=None) -> int:
    parser = build_arg_parser()
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

    effective_d_model, effective_layers, effective_heads = (
        args.d_model,
        args.layers,
        args.heads,
    )
    effective_pooled = False
    effective_reply = False
    initial = None
    if args.action_residual and args.model != "transformer":
        raise SystemExit("--action-residual is available only for --model transformer")
    if args.slot_embedding and args.model != "transformer":
        raise SystemExit("--slot-embedding is available only for --model transformer")
    if args.graph_module and args.model != "transformer":
        raise SystemExit("--graph-module is available only for --model transformer")
    if args.hierarchical_value and args.model != "transformer":
        raise SystemExit(
            "--hierarchical-value is available only for --model transformer"
        )
    if not math.isfinite(args.action_policy_weight) or args.action_policy_weight < 0:
        raise SystemExit("--action-policy-weight must be finite and non-negative")
    if args.init_checkpoint:
        if args.model != "transformer":
            raise SystemExit("--init-checkpoint W5 migration requires --model transformer")
        initial = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        stored = initial.get("config", {})
        effective_d_model = int(stored.get("d_model", args.d_model))
        effective_layers = int(stored.get("layers", args.layers))
        effective_heads = heads_from_config(stored)
        effective_pooled = pooled_readout_from_config(stored)
        effective_reply = reply_head_from_config(stored)
        # Warm starts are additive. An existing W5 path is never silently
        # removed merely because the new flag was omitted.
        args.action_residual = bool(
            args.action_residual or action_residual_from_config(stored)
        )
        args.action_exposes = bool(
            args.action_exposes or action_exposes_from_config(stored)
        )
        args.slot_embedding = bool(
            args.slot_embedding or slot_embedding_from_config(stored)
        )
        args.graph_module = bool(
            args.graph_module or graph_module_from_config(stored)
        )
        inherited_hier = hierarchical_value_from_config(stored)
        args.hierarchical_value = bool(
            args.hierarchical_value or inherited_hier["hierarchical_value"]
        )
        if inherited_hier["hierarchical_value"] and args.hierarchical_value_detach is None:
            # Omitted INHERITS; explicit overrides. Moving a head from detached
            # to attached is the whole representation-learning experiment, so
            # it must be expressible.
            args.hierarchical_value_detach = inherited_hier[
                "hierarchical_value_detach"
            ]
        resolve_graph_args(args, stored)
        print(
            "warm-start architecture: "
            f"d{effective_d_model} L{effective_layers} h{effective_heads} "
            f"pooled={effective_pooled} reply={effective_reply} "
            f"slots={args.slot_embedding} graph={args.graph_module}"
        )
    resolve_graph_args(args, None)
    resolve_hier_value_args(args)
    model = build_model(
        args.model,
        effective_d_model,
        effective_layers,
        effective_heads,
        effective_pooled,
        effective_reply,
        args.action_residual,
        action_exposes=args.action_exposes,
        slot_embedding=args.slot_embedding,
        graph_module=args.graph_module,
        graph_layers=args.graph_layers,
        graph_bases=args.graph_bases,
        graph_alpha=args.graph_alpha,
        hierarchical_value=args.hierarchical_value,
        hierarchical_value_detach=args.hierarchical_value_detach,
    )
    if initial is not None:
        load_checkpoint(
            args.init_checkpoint,
            model,
            migrate=True,
            checkpoint=initial,
        )
    if getattr(model, "action_scorer", None) is not None:
        model.action_scorer.gate.requires_grad_(args.train_action_gate)
    if args.action_residual_only:
        if getattr(model, "action_scorer", None) is None:
            raise SystemExit("--action-residual-only requires --action-residual")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.action_scorer.parameters():
            parameter.requires_grad_(True)
        model.action_scorer.gate.requires_grad_(args.train_action_gate)
        if args.action_policy_weight <= 0:
            raise SystemExit(
                "--action-residual-only requires a positive "
                "--action-policy-weight"
            )
    if args.action_policy_weight > 0 and not args.action_residual:
        raise SystemExit("--action-policy-weight requires --action-residual")
    if args.train_action_gate and not args.action_residual:
        raise SystemExit("--train-action-gate requires --action-residual")
    if args.action_exposes and not args.action_residual:
        raise SystemExit(
            "--action-exposes requires --action-residual; it is a branch of "
            "that scorer, not a scorer of its own"
        )
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
        action_policy_weight=args.action_policy_weight,
        hier_value_weight=args.hier_value_weight,
        hier_value_replaces_joint7=args.hier_value_replaces_joint7,
    )
    final = evaluate(
        model,
        val_examples or train_examples,
        args.device,
        args.batch_size,
        args.aux_weight,
        precision=args.precision,
        action_policy_weight=args.action_policy_weight,
        hier_value_weight=args.hier_value_weight,
        hier_value_replaces_joint7=args.hier_value_replaces_joint7,
    )
    print(f"final: {json.dumps({k: round(v, 4) for k, v in final.items()})}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        source_model = getattr(model, "_orig_mod", model)
        config = {
            "model": args.model,
            "d_model": effective_d_model,
            "layers": effective_layers,
            "heads": (
                int(source_model.attention_heads)
                if hasattr(source_model, "attention_heads")
                else None
            ),
            # Must travel with the weights: a rebuild without it loads a
            # pooled checkpoint minus `readout_proj` and computes something else.
            "pooled_readout": bool(getattr(source_model, "pooled_readout", False)),
            "reply_head": bool(getattr(source_model, "reply_head", False)),
            "precision": args.precision,
            "weight_decay": args.weight_decay,
            "aux_weight": args.aux_weight,
            "action_policy_weight": args.action_policy_weight,
            "hier_value_weight": args.hier_value_weight,
            "hier_value_replaces_joint7": bool(args.hier_value_replaces_joint7),
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
