"""Equal-step control/treatment fine-tuning for opponent-reply labels.

This is an isolated pilot runner.  It never writes ``current_best`` and never
invokes promotion.  Both arms consume the exact same ordinary replay batches;
the treatment can receive grouped-pick reply loss, direct distillation of the
searched reply-state value, or both.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
    root_state_key: str
    my_board: np.ndarray
    opp_board: np.ndarray
    flat: np.ndarray
    legal_indices: np.ndarray
    group_indices: tuple[np.ndarray, ...]
    target: np.ndarray
    baseline_conditionals: tuple[np.ndarray, ...]
    parent_searched_rank: int
    search_value_target: float
    search_value_standard_error: float


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
            searched_rows = list(row["per_pick"])
            if not searched_rows:
                raise ValueError("reply example has no searched pick values")
            searched_best = max(
                searched_rows, key=lambda item: float(item["searched_value_actor"]))
            search_value_target = float(searched_best["searched_value_actor"])
            search_value_standard_error = float(searched_best["mc_standard_error"])
            parent_searched_rank = int(row["parent_searched_rank"])
            if (not math.isfinite(search_value_target)
                    or not math.isfinite(search_value_standard_error)
                    or search_value_standard_error < 0.0
                    or parent_searched_rank < 1):
                raise ValueError("invalid searched reply-state value metadata")
            examples.append(ReplyExample(
                example_id=str(row["example_id"]),
                root_state_key=str(row["root_state_key"]),
                my_board=decode_array_blob(row["encoded_state"]["my_board"]),
                opp_board=decode_array_blob(row["encoded_state"]["opp_board"]),
                flat=decode_array_blob(row["encoded_state"]["flat"]),
                legal_indices=legal_indices,
                group_indices=tuple(group_indices),
                target=np.asarray(row["denial_policy_target"], dtype=np.float32),
                baseline_conditionals=tuple(baseline_conditionals),
                parent_searched_rank=parent_searched_rank,
                search_value_target=search_value_target,
                search_value_standard_error=search_value_standard_error,
            ))
        if not examples:
            raise ValueError(f"reply dataset has no usable examples: {self.path}")
        self.examples = examples
        self.indices_by_parent_rank: dict[int, np.ndarray] = {
            rank: np.asarray([
                index for index, row in enumerate(examples)
                if row.parent_searched_rank == rank
            ], dtype=np.int64)
            for rank in sorted({row.parent_searched_rank for row in examples})
        }
        indices_by_root: dict[str, list[int]] = {}
        for index, row in enumerate(examples):
            indices_by_root.setdefault(row.root_state_key, []).append(index)
        rank_pairs = []
        for root_key in sorted(indices_by_root):
            root_indices = indices_by_root[root_key]
            anchors = sorted(
                (index for index in root_indices
                 if examples[index].parent_searched_rank == 1),
                key=lambda index: examples[index].example_id)
            if not anchors:
                continue
            anchor = anchors[0]
            rank_pairs.extend(
                (anchor, index)
                for index in root_indices
                if examples[index].parent_searched_rank > 1)
        self.rank_pairs = tuple(rank_pairs)
        self.rank_pairs_by_secondary_rank: dict[int, tuple[tuple[int, int], ...]] = {
            rank: tuple(pair for pair in self.rank_pairs
                        if examples[pair[1]].parent_searched_rank == rank)
            for rank in sorted({
                examples[pair[1]].parent_searched_rank for pair in self.rank_pairs
            })
        }

    def __len__(self) -> int:
        return len(self.examples)

    def _batch(self, indices: np.ndarray, device: str) -> dict[str, Any]:
        rows = [self.examples[int(index)] for index in indices]
        return {
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
            "parent_searched_rank": torch.as_tensor(
                [row.parent_searched_rank for row in rows], device=device,
                dtype=torch.int64),
            "search_value_target": torch.as_tensor(
                [row.search_value_target for row in rows], device=device,
                dtype=torch.float32),
            "search_value_standard_error": torch.as_tensor(
                [row.search_value_standard_error for row in rows], device=device,
                dtype=torch.float32),
        }

    def sample(
        self, batch_size: int, rng: np.random.Generator, device: str,
        *, balance_parent_ranks: bool = False,
    ):
        if balance_parent_ranks:
            ranks = np.asarray(sorted(self.indices_by_parent_rank), dtype=np.int64)
            sampled_ranks = rng.choice(ranks, size=int(batch_size), replace=True)
            indices = np.asarray([
                rng.choice(self.indices_by_parent_rank[int(rank)])
                for rank in sampled_ranks
            ], dtype=np.int64)
        else:
            indices = rng.integers(0, len(self.examples), size=int(batch_size))
        batch = self._batch(indices, device)
        return batch, [int(index) for index in indices]

    def all(self, device: str) -> dict[str, Any]:
        return self._batch(np.arange(len(self.examples), dtype=np.int64), device)

    def sample_rank_pairs(
        self, pair_count: int, rng: np.random.Generator, device: str,
    ) -> tuple[dict[str, Any], list[int]]:
        if not self.rank_pairs:
            raise ValueError("reply dataset has no within-root rank-1/secondary pairs")
        ranks = np.asarray(sorted(self.rank_pairs_by_secondary_rank), dtype=np.int64)
        sampled_ranks = rng.choice(ranks, size=int(pair_count), replace=True)
        sampled_pairs = [
            self.rank_pairs_by_secondary_rank[int(rank)][
                int(rng.integers(0, len(self.rank_pairs_by_secondary_rank[int(rank)])))]
            for rank in sampled_ranks
        ]
        anchor_indices = [pair[0] for pair in sampled_pairs]
        secondary_indices = [pair[1] for pair in sampled_pairs]
        indices = np.asarray(anchor_indices + secondary_indices, dtype=np.int64)
        batch = self._batch(indices, device)
        batch["rank_pair_count"] = int(pair_count)
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


def mcts_actor_value(
    own: torch.Tensor, opp: torch.Tensor, win_prob: torch.Tensor,
    *, alpha: float, margin_gain: float,
) -> torch.Tensor:
    """Differentiable actor-frame leaf value used by Python and Rust MCTS."""
    own = own.reshape(-1)
    opp = opp.reshape(-1)
    win_prob = win_prob.reshape(-1)
    win_value = 2.0 * win_prob - 1.0
    margin = torch.tanh((own - opp) * float(margin_gain))
    return ((1.0 - float(alpha)) * win_value
            + float(alpha) * win_value.pow(4) * margin)


def searched_reply_value_loss(
    own: torch.Tensor, opp: torch.Tensor, win_prob: torch.Tensor,
    batch: dict[str, Any], *, alpha: float, margin_gain: float,
    huber_delta: float = 0.1,
) -> torch.Tensor:
    """Huber-distill the forced-tree value in the encoded reply actor's frame."""
    predicted = mcts_actor_value(
        own, opp, win_prob, alpha=alpha, margin_gain=margin_gain)
    target = batch["search_value_target"].reshape(-1)
    if predicted.shape != target.shape:
        raise ValueError("searched reply value prediction and target are not aligned")
    loss = F.huber_loss(
        predicted, target, reduction="mean", delta=float(huber_delta))
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite searched reply value loss")
    return loss


def searched_reply_value_gap_loss(
    own: torch.Tensor, opp: torch.Tensor, win_prob: torch.Tensor,
    batch: dict[str, Any], *, alpha: float, margin_gain: float,
    huber_delta: float = 0.1,
) -> torch.Tensor:
    """Match within-root value gaps while anchoring rank 1 against aux gradients."""
    pair_count = int(batch.get("rank_pair_count", 0))
    if pair_count < 1:
        raise ValueError("searched reply value-gap loss requires rank pairs")
    predicted = mcts_actor_value(
        own, opp, win_prob, alpha=alpha, margin_gain=margin_gain)
    target = batch["search_value_target"].reshape(-1)
    if len(predicted) != 2 * pair_count or target.shape != predicted.shape:
        raise ValueError("rank-pair reply values are not aligned")
    # Rank 1 defines the local value origin but ordinary replay remains solely
    # responsible for updating it. Without detach, the symmetric pairwise
    # gradient lowers rank 1 while raising the secondary state, which can pass
    # the relative gap objective through an unwanted primary-value shift.
    predicted_gap = predicted[pair_count:] - predicted[:pair_count].detach()
    target_gap = target[pair_count:] - target[:pair_count]
    loss = F.huber_loss(
        predicted_gap, target_gap, reduction="mean", delta=float(huber_delta))
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite searched reply value-gap loss")
    return loss


def rank1_value_anchor_loss(
    own: torch.Tensor, opp: torch.Tensor, win_prob: torch.Tensor,
    batch: dict[str, Any], anchor_values: torch.Tensor, *,
    alpha: float, margin_gain: float, huber_delta: float = 0.1,
) -> torch.Tensor:
    """Keep treatment rank-1 values aligned with a fixed or control anchor."""
    pair_count = int(batch.get("rank_pair_count", 0))
    if pair_count < 1:
        raise ValueError("rank-1 value anchor requires rank pairs")
    predicted = mcts_actor_value(
        own, opp, win_prob, alpha=alpha, margin_gain=margin_gain)
    anchor_values = anchor_values.reshape(-1).detach()
    if len(predicted) != 2 * pair_count or len(anchor_values) != pair_count:
        raise ValueError("rank-1 anchor values are not aligned with rank pairs")
    loss = F.huber_loss(
        predicted[:pair_count], anchor_values, reduction="mean",
        delta=float(huber_delta))
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite rank-1 value anchor loss")
    return loss


def actor_value_anchor_loss(
    own: torch.Tensor, opp: torch.Tensor, win_prob: torch.Tensor,
    anchor_values: torch.Tensor, *, alpha: float, margin_gain: float,
    huber_delta: float = 0.1,
) -> torch.Tensor:
    """Distill actor-frame values from an ordinary-only control arm."""
    predicted = mcts_actor_value(
        own, opp, win_prob, alpha=alpha, margin_gain=margin_gain)
    anchor_values = anchor_values.reshape(-1).detach()
    if predicted.shape != anchor_values.shape:
        raise ValueError("actor-value anchor predictions are not aligned")
    loss = F.huber_loss(
        predicted, anchor_values, reduction="mean", delta=float(huber_delta))
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite actor-value anchor loss")
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
    lambda_search_value: float = 0.0,
    lambda_search_value_gap: float = 0.0,
    lambda_rank1_anchor: float = 0.0,
    rank1_anchor_values: Optional[torch.Tensor] = None,
    lambda_ordinary_value_anchor: float = 0.0,
    ordinary_value_anchor_values: Optional[torch.Tensor] = None,
    policy_weight: float = 1.0, lambda_score: float = 0.5,
    lambda_w: float = 0.25, score_scale: float = 160.0, grad_clip: float = 1.0,
    search_value_alpha: float = 0.5, search_value_margin_gain: float = 2.0,
    search_value_huber_delta: float = 0.1,
) -> dict[str, float]:
    ordinary_total, policy_loss, own_loss, opp_loss, win_loss = _ordinary_losses(
        net, ordinary_batch, policy_weight=policy_weight,
        lambda_score=lambda_score, lambda_w=lambda_w, score_scale=score_scale)
    # The treatment sees an extra reply-state batch that the control does not.
    # Use frozen BatchNorm statistics for this auxiliary forward so those extra
    # states cannot change treatment-only running buffers. Gradients still flow
    # through every affine parameter and the shared trunk.
    was_training = net.training
    net.eval()
    try:
        reply_own, reply_opp, reply_win, reply_logits = net(
            reply_batch["my_board"], reply_batch["opp_board"], reply_batch["flat"])
    finally:
        net.train(was_training)
    reply_loss = grouped_reply_loss(reply_logits, reply_batch)
    search_value_loss = searched_reply_value_loss(
        reply_own, reply_opp, reply_win, reply_batch,
        alpha=search_value_alpha, margin_gain=search_value_margin_gain,
        huber_delta=search_value_huber_delta)
    if int(reply_batch.get("rank_pair_count", 0)) > 0:
        search_value_gap_loss = searched_reply_value_gap_loss(
            reply_own, reply_opp, reply_win, reply_batch,
            alpha=search_value_alpha, margin_gain=search_value_margin_gain,
            huber_delta=search_value_huber_delta)
    else:
        search_value_gap_loss = search_value_loss * 0.0
    if float(lambda_rank1_anchor) > 0.0:
        if rank1_anchor_values is None:
            raise ValueError("rank-1 anchor values are required when its loss is enabled")
        rank1_anchor_loss = rank1_value_anchor_loss(
            reply_own, reply_opp, reply_win, reply_batch, rank1_anchor_values,
            alpha=search_value_alpha, margin_gain=search_value_margin_gain,
            huber_delta=search_value_huber_delta)
    else:
        rank1_anchor_loss = search_value_loss * 0.0
    if float(lambda_ordinary_value_anchor) > 0.0:
        if ordinary_value_anchor_values is None:
            raise ValueError("ordinary value anchor targets are required when enabled")
        anchor_count = len(ordinary_value_anchor_values)
        mb, ob, flat = ordinary_batch[:3]
        was_training = net.training
        net.eval()
        try:
            anchor_own, anchor_opp, anchor_win, _ = net(
                mb[:anchor_count], ob[:anchor_count], flat[:anchor_count])
        finally:
            net.train(was_training)
        ordinary_value_anchor_loss = actor_value_anchor_loss(
            anchor_own, anchor_opp, anchor_win, ordinary_value_anchor_values,
            alpha=search_value_alpha, margin_gain=search_value_margin_gain,
            huber_delta=search_value_huber_delta)
    else:
        ordinary_value_anchor_loss = search_value_loss * 0.0
    total = (ordinary_total
             + float(lambda_reply) * reply_loss
             + float(lambda_search_value) * search_value_loss
             + float(lambda_search_value_gap) * search_value_gap_loss
             + float(lambda_rank1_anchor) * rank1_anchor_loss
             + float(lambda_ordinary_value_anchor) * ordinary_value_anchor_loss)
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
        "search_value_loss": float(search_value_loss.item()),
        "search_value_gap_loss": float(search_value_gap_loss.item()),
        "rank1_anchor_loss": float(rank1_anchor_loss.item()),
        "ordinary_value_anchor_loss": float(ordinary_value_anchor_loss.item()),
        "grad_norm": grad_norm,
    }


def evaluate_reply(
    net, dataset: ReplyDataset, *, device: str,
    alpha: float, margin_gain: float, huber_delta: float,
) -> dict[str, Any]:
    batch = dataset.all(device)
    was_training = net.training
    net.eval()
    with torch.no_grad():
        own, opp, win, logits = net(
            batch["my_board"], batch["opp_board"], batch["flat"])
        loss = float(grouped_reply_loss(logits, batch).item())
        predicted = mcts_actor_value(
            own, opp, win, alpha=alpha, margin_gain=margin_gain)
        target = batch["search_value_target"].reshape(-1)
        value_loss = float(F.huber_loss(
            predicted, target, reduction="mean", delta=float(huber_delta)).item())
        drift = placement_drift(logits, batch)
    net.train(was_training)
    predicted_np = predicted.detach().double().cpu().numpy()
    target_np = target.detach().double().cpu().numpy()
    ranks_np = batch["parent_searched_rank"].detach().cpu().numpy()
    residual = predicted_np - target_np
    correlation = (
        float(np.corrcoef(predicted_np, target_np)[0, 1])
        if len(predicted_np) > 1
        and float(np.std(predicted_np)) > 0.0
        and float(np.std(target_np)) > 0.0
        else 0.0
    )
    by_rank = {}
    for rank in sorted(set(int(value) for value in ranks_np)):
        selected = ranks_np == rank
        rank_residual = residual[selected]
        by_rank[str(rank)] = {
            "examples": int(selected.sum()),
            "mean_prediction_minus_target": float(rank_residual.mean()),
            "median_prediction_minus_target": float(np.median(rank_residual)),
            "mae": float(np.abs(rank_residual).mean()),
            "rmse": float(np.sqrt(np.mean(rank_residual ** 2))),
        }
    pair_metrics: dict[str, Any] = {"pairs": 0, "by_secondary_rank": {}}
    if dataset.rank_pairs:
        anchor_indices = np.asarray(
            [pair[0] for pair in dataset.rank_pairs], dtype=np.int64)
        secondary_indices = np.asarray(
            [pair[1] for pair in dataset.rank_pairs], dtype=np.int64)
        predicted_gaps = predicted_np[secondary_indices] - predicted_np[anchor_indices]
        target_gaps = target_np[secondary_indices] - target_np[anchor_indices]
        gap_residual = predicted_gaps - target_gaps
        pair_metrics = {
            "pairs": len(dataset.rank_pairs),
            "mean_prediction_minus_target": float(gap_residual.mean()),
            "median_prediction_minus_target": float(np.median(gap_residual)),
            "mae": float(np.abs(gap_residual).mean()),
            "rmse": float(np.sqrt(np.mean(gap_residual ** 2))),
            "by_secondary_rank": {},
        }
        for rank in sorted(dataset.rank_pairs_by_secondary_rank):
            selected = np.asarray([
                dataset.examples[secondary].parent_searched_rank == rank
                for _anchor, secondary in dataset.rank_pairs
            ], dtype=bool)
            rank_residual = gap_residual[selected]
            pair_metrics["by_secondary_rank"][str(rank)] = {
                "pairs": int(selected.sum()),
                "mean_prediction_minus_target": float(rank_residual.mean()),
                "mae": float(np.abs(rank_residual).mean()),
                "rmse": float(np.sqrt(np.mean(rank_residual ** 2))),
            }
    return {
        "reply_loss": loss,
        "search_value_huber_loss": value_loss,
        "search_value_prediction_mean": float(predicted_np.mean()),
        "search_value_target_mean": float(target_np.mean()),
        "search_value_mean_prediction_minus_target": float(residual.mean()),
        "search_value_median_prediction_minus_target": float(np.median(residual)),
        "search_value_mae": float(np.abs(residual).mean()),
        "search_value_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "search_value_correlation": correlation,
        "search_value_by_parent_rank": by_rank,
        "search_value_gap": pair_metrics,
        **drift,
    }


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
    if (args.steps < 1 or args.batch_size < 1 or args.ordinary_anchor_batch_size < 1
            or not 0.0 < args.reply_fraction <= 1.0):
        raise ValueError(
            "steps/batch sizes must be positive and reply_fraction must be in (0,1]")
    if (args.lambda_reply < 0.0 or args.lambda_search_value < 0.0
            or args.lambda_search_value_gap < 0.0
            or args.lambda_rank1_anchor < 0.0
            or args.lambda_ordinary_value_anchor < 0.0):
        raise ValueError("reply auxiliary loss weights must be non-negative")
    if (args.lambda_reply == 0.0 and args.lambda_search_value == 0.0
            and args.lambda_search_value_gap == 0.0
            and args.lambda_rank1_anchor == 0.0
            and args.lambda_ordinary_value_anchor == 0.0):
        raise ValueError("the treatment arm requires at least one reply auxiliary loss")
    if args.lambda_rank1_anchor > 0.0 and args.lambda_search_value_gap == 0.0:
        raise ValueError("rank-1 anchoring requires the paired value-gap loss")
    if args.rank1_anchor_source not in {"control", "searched"}:
        raise ValueError("rank1_anchor_source must be 'control' or 'searched'")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256_file(args.checkpoint)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

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
                alpha=float(checkpoint_config.get("alpha", 0.5)),
                margin_gain=float(checkpoint_config.get("margin_gain", 2.0)),
                huber_delta=args.search_value_huber_delta),
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
        if args.lambda_search_value_gap > 0.0:
            reply_batch, reply_indices = train_dataset.sample_rank_pairs(
                max(1, reply_batch_size // 2), reply_rng, args.device)
        else:
            reply_batch, reply_indices = train_dataset.sample(
                reply_batch_size, reply_rng, args.device,
                balance_parent_ranks=args.balance_reply_ranks)
        rank1_anchor_values = None
        if args.lambda_rank1_anchor > 0.0:
            pair_count = int(reply_batch.get("rank_pair_count", 0))
            if pair_count < 1:
                raise ValueError("rank-1 anchoring requires paired reply batches")
            if args.rank1_anchor_source == "searched":
                rank1_anchor_values = reply_batch[
                    "search_value_target"][:pair_count].detach()
            else:
                # Match the treatment auxiliary forward's frozen BatchNorm mode
                # and leave control running statistics untouched.
                control_was_training = control.training
                control.eval()
                try:
                    with torch.no_grad():
                        anchor_own, anchor_opp, anchor_win, _ = control(
                            reply_batch["my_board"], reply_batch["opp_board"],
                            reply_batch["flat"])
                        rank1_anchor_values = mcts_actor_value(
                            anchor_own, anchor_opp, anchor_win,
                            alpha=float(checkpoint_config.get("alpha", 0.5)),
                            margin_gain=float(checkpoint_config.get("margin_gain", 2.0)),
                        )[:pair_count].detach()
                finally:
                    control.train(control_was_training)
        ordinary_value_anchor_values = None
        if args.lambda_ordinary_value_anchor > 0.0:
            anchor_count = min(args.ordinary_anchor_batch_size, args.batch_size)
            mb, ob, flat = ordinary_batch[:3]
            control_was_training = control.training
            control.eval()
            try:
                with torch.no_grad():
                    anchor_own, anchor_opp, anchor_win, _ = control(
                        mb[:anchor_count], ob[:anchor_count], flat[:anchor_count])
                    ordinary_value_anchor_values = mcts_actor_value(
                        anchor_own, anchor_opp, anchor_win,
                        alpha=float(checkpoint_config.get("alpha", 0.5)),
                        margin_gain=float(checkpoint_config.get("margin_gain", 2.0)),
                    ).detach()
            finally:
                control.train(control_was_training)
        control_metrics = train_step(
            control, ordinary_batch, control_optimizer,
            policy_weight=args.policy_weight, lambda_score=args.lambda_score,
            lambda_w=args.lambda_w, score_scale=args.score_scale,
            grad_clip=args.grad_clip,
        )
        treatment_metrics = treatment_train_step(
            treatment, ordinary_batch, reply_batch, treatment_optimizer,
            lambda_reply=args.lambda_reply,
            lambda_search_value=args.lambda_search_value,
            lambda_search_value_gap=args.lambda_search_value_gap,
            lambda_rank1_anchor=args.lambda_rank1_anchor,
            rank1_anchor_values=rank1_anchor_values,
            lambda_ordinary_value_anchor=args.lambda_ordinary_value_anchor,
            ordinary_value_anchor_values=ordinary_value_anchor_values,
            policy_weight=args.policy_weight,
            lambda_score=args.lambda_score, lambda_w=args.lambda_w,
            score_scale=args.score_scale, grad_clip=args.grad_clip,
            search_value_alpha=float(checkpoint_config.get("alpha", 0.5)),
            search_value_margin_gain=float(
                checkpoint_config.get("margin_gain", 2.0)),
            search_value_huber_delta=args.search_value_huber_delta,
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
            "lambda_reply": args.lambda_reply,
            "lambda_search_value": args.lambda_search_value,
            "lambda_search_value_gap": args.lambda_search_value_gap,
            "lambda_rank1_anchor": args.lambda_rank1_anchor,
            "rank1_anchor_source": args.rank1_anchor_source,
            "lambda_ordinary_value_anchor": args.lambda_ordinary_value_anchor,
            "ordinary_anchor_batch_size": args.ordinary_anchor_batch_size,
            "search_value_huber_delta": args.search_value_huber_delta,
            "balance_reply_ranks": bool(args.balance_reply_ranks),
            "seed": args.seed,
            "deterministic": bool(args.deterministic),
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
    parser.add_argument("--lambda-search-value", type=float, default=0.0)
    parser.add_argument("--lambda-search-value-gap", type=float, default=0.0)
    parser.add_argument("--lambda-rank1-anchor", type=float, default=0.0)
    parser.add_argument(
        "--rank1-anchor-source", choices=("control", "searched"),
        default="searched")
    parser.add_argument("--lambda-ordinary-value-anchor", type=float, default=0.0)
    parser.add_argument("--ordinary-anchor-batch-size", type=int, default=64)
    parser.add_argument("--search-value-huber-delta", type=float, default=0.1)
    parser.add_argument("--balance-reply-ranks", action="store_true")
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
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report = run_pilot(args)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
