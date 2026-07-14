"""Game-agnostic sparse NNUE used by the Step-3 training path.

The first layer is represented as an ``EmbeddingBag(mode="sum")`` over active
binary feature indices.  Its weight rows are the columns of the conventional
dense first-layer matrix, so the result is exactly::

    z = bias + sum(weight[feature] for feature in active_features)

The 171-value Kingdomino summary is supplied separately and concatenated only
after ``relu(z)``.  Auxiliary heads shape the shared representation during
training but are deliberately separate from the two exported search heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SparseNNUE(nn.Module):
    """Sparse accumulator + small tail with two search and two auxiliary heads.

    Auxiliary output order is a generic contract chosen by the data adapter:
    ``aux_scores`` has six continuous values and ``aux_bonus_logits`` four
    binary logits.  Kingdomino supplies, in actor-relative ``[my, opp]`` order,
    ``(territory, largest, crowns)`` and ``(harmony, middle)`` respectively.
    """

    def __init__(
        self,
        feature_count: int,
        summary_size: int,
        acc_width: int = 256,
        tail_hidden: int = 32,
        aux_score_size: int = 6,
        aux_bonus_size: int = 4,
    ):
        super().__init__()
        if min(feature_count, summary_size, acc_width, tail_hidden) <= 0:
            raise ValueError("network dimensions must be positive")
        self.feature_count = int(feature_count)
        self.summary_size = int(summary_size)
        self.acc_width = int(acc_width)
        self.tail_hidden = int(tail_hidden)
        self.aux_score_size = int(aux_score_size)
        self.aux_bonus_size = int(aux_bonus_size)

        # include_last_offset=True gives an unambiguous CSR contract: offsets has
        # B+1 entries, including the final index count.
        self.accumulator = nn.EmbeddingBag(
            self.feature_count,
            self.acc_width,
            mode="sum",
            include_last_offset=True,
        )
        self.accumulator_bias = nn.Parameter(torch.zeros(self.acc_width))
        self.tail = nn.Sequential(
            nn.Linear(self.acc_width + self.summary_size, self.tail_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.tail_hidden, self.tail_hidden),
            nn.ReLU(inplace=True),
        )
        self.outcome_head = nn.Linear(self.tail_hidden, 1)
        self.margin_head = nn.Linear(self.tail_hidden, 1)
        self.aux_score_head = nn.Linear(self.tail_hidden, self.aux_score_size)
        self.aux_bonus_head = nn.Linear(self.tail_hidden, self.aux_bonus_size)

    def forward(
        self,
        active_indices: torch.Tensor,
        offsets: torch.Tensor,
        summary: torch.Tensor,
    ):
        """Return outcome logit, normalized margin, and auxiliary predictions.

        ``active_indices`` is flattened int64 CSR data, ``offsets`` is int64 with
        length ``batch+1``, and ``summary`` is float with shape ``(batch, S)``.
        All outputs remain in the actor-relative frame.
        """
        if summary.ndim != 2 or summary.shape[1] != self.summary_size:
            raise ValueError(
                f"summary must have shape (batch, {self.summary_size}), got {tuple(summary.shape)}"
            )
        if offsets.ndim != 1 or offsets.numel() != summary.shape[0] + 1:
            raise ValueError("offsets must be a 1-D CSR array of length batch+1")
        z = self.accumulator(active_indices, offsets) + self.accumulator_bias
        h = self.tail(torch.cat((torch.relu(z), summary), dim=1))
        return (
            self.outcome_head(h).squeeze(-1),
            self.margin_head(h).squeeze(-1),
            self.aux_score_head(h),
            self.aux_bonus_head(h),
        )

    @torch.no_grad()
    def evaluate(self, active_indices, offsets, summary):
        """Return actor-frame expected score and normalized margin only."""
        outcome_logit, margin, _, _ = self(active_indices, offsets, summary)
        return torch.sigmoid(outcome_logit), margin


def sparse_config_of(net: SparseNNUE) -> dict:
    return {
        "feature_count": net.feature_count,
        "summary_size": net.summary_size,
        "acc_width": net.acc_width,
        "tail_hidden": net.tail_hidden,
        "aux_score_size": net.aux_score_size,
        "aux_bonus_size": net.aux_bonus_size,
    }
