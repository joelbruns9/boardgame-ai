"""Encoder-only set transformer over entity tokens (plan §4, spec §5.8a).

Input contract = ``dataset.collate`` tensors. Per-token embedding is the sum of
(per-type entity embedding) + (per-type feature projection) + (type embedding)
+ (aux card embedding, used by WONDER burials). Pre-LN transformer layers, no
positional encoding — structure lives in token features. Readout is the GLOBAL
token (always position 0).

Heads: policy (NUM_ACTIONS logits, legality-masked downstream), value (W/D/L),
joint winner×victory-type (7), VP-margin regression, final military position,
final science counts (2). Aux heads per the KataGo lesson (§2).

Per-type input projections are the §5.8a forward-compat hook: adding a token
type later = one new embedding row + one zero-initialized projection.
"""

from __future__ import annotations

import torch
from torch import nn

from .codec import NUM_ACTIONS
from .dataset import ENTITY_SPACES, FEATURE_COUNTS, NUM_AUX_CARDS, TOKEN_TYPES


class TokenEmbedder(nn.Module):
    """Shared by the transformer and the MLP control model.

    Per-type modules are keyed by token-type NAME (ModuleDict), so state-dict
    keys stay stable when a new token type is appended — the §5.8a additive-
    migration hook (`train.migrate_state_dict` zero-initializes exactly the
    keys that have no counterpart in an older checkpoint).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.entity = nn.ModuleDict(
            {
                token_type.value: nn.Embedding(space, d_model)
                for token_type, space in zip(TOKEN_TYPES, ENTITY_SPACES)
            }
        )
        self.feature = nn.ModuleDict(
            {
                token_type.value: nn.Linear(count, d_model)
                for token_type, count in zip(TOKEN_TYPES, FEATURE_COUNTS)
            }
        )
        self.type_embedding = nn.Embedding(len(TOKEN_TYPES), d_model)
        # padding_idx keeps the "no aux entity" row at zero permanently —
        # it receives no gradient, so real tokens never drift it.
        self.aux = nn.Embedding(NUM_AUX_CARDS, d_model, padding_idx=0)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        type_ids = batch["type_ids"]
        out = self.type_embedding(type_ids) + self.aux(batch["aux_ids"])
        per_type = torch.zeros_like(out)
        for type_index, token_type in enumerate(TOKEN_TYPES):
            mask = type_ids == type_index
            if not mask.any():
                continue
            entity = self.entity[token_type.value]
            feature = self.feature[token_type.value]
            rows = entity(batch["entity_ids"][mask])
            rows = rows + feature(batch["features"][mask][:, : feature.in_features])
            per_type[mask] = rows
        out = out + per_type
        return out.masked_fill(batch["pad_mask"].unsqueeze(-1), 0.0)


class Heads(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.policy = nn.Linear(d_model, NUM_ACTIONS)
        self.value = nn.Linear(d_model, 3)
        self.joint7 = nn.Linear(d_model, 7)
        self.margin = nn.Linear(d_model, 1)
        self.military = nn.Linear(d_model, 1)
        self.science = nn.Linear(d_model, 2)

    def forward(self, readout: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "policy": self.policy(readout),
            "value": self.value(readout),
            "joint7": self.joint7(readout),
            "margin": self.margin(readout).squeeze(-1),
            "military": self.military(readout).squeeze(-1),
            "science": self.science(readout),
        }


class SWDNet(nn.Module):
    def __init__(self, d_model: int = 128, layers: int = 4, heads: int = 4):
        super().__init__()
        self.embedder = TokenEmbedder(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.heads = Heads(d_model)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens = self.embedder(batch)
        encoded = self.encoder(tokens, src_key_padding_mask=batch["pad_mask"])
        readout = self.final_norm(encoded[:, 0])  # GLOBAL token
        return self.heads(readout)


def masked_policy_log_softmax(
    logits: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    return torch.log_softmax(masked, dim=-1)
