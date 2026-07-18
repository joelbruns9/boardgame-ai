"""Flat control model on the same token features (plan §4, §2 control-model row).

Architecturally independent of the transformer — no attention: per-type token
projections are mean-pooled per type, concatenated, and fed to an MLP trunk
with the same six heads. Exists as a fast baseline and as monoculture
insurance (run5/run10 lesson): a sparring partner whose blind spots differ.
"""

from __future__ import annotations

import torch
from torch import nn

from .dataset import TOKEN_TYPES
from .net import Heads, TokenEmbedder


class SWDMlp(nn.Module):
    def __init__(self, d_model: int = 128, hidden: int = 512, layers: int = 3):
        super().__init__()
        self.embedder = TokenEmbedder(d_model)
        trunk: list[nn.Module] = []
        width = d_model * len(TOKEN_TYPES)
        for _ in range(layers):
            trunk.extend([nn.Linear(width, hidden), nn.GELU()])
            width = hidden
        self.trunk = nn.Sequential(*trunk)
        self.heads = Heads(hidden)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens = self.embedder(batch)  # [B, T, d], padding rows zeroed
        type_ids = batch["type_ids"]
        real = ~batch["pad_mask"]
        pooled = []
        for type_index in range(len(TOKEN_TYPES)):
            mask = (type_ids == type_index) & real
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
            summed = (tokens * mask.unsqueeze(-1)).sum(dim=1)
            pooled.append(summed / counts)
        return self.heads(self.trunk(torch.cat(pooled, dim=-1)))
