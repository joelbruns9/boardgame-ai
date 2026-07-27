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
from .dataset import (
    ENTITY_SPACES,
    FEATURE_COUNTS,
    MAX_FEATURES,
    NUM_AUX_CARDS,
    TOKEN_TYPES,
)


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
        #: Fused inference tensors, built by `fuse()`. `None` = use the per-type
        #: loop, which is the training path and stays untouched.
        self._fused: dict[str, torch.Tensor] | None = None
        # The cache copies `entity`/`feature` but reads `type_embedding`/`aux`
        # live, so anything that rewrites parameters in place would leave it
        # PARTIALLY stale — a silently wrong forward rather than a loud one.
        # Invalidate on every such operation instead of documenting the hazard.
        self.register_load_state_dict_post_hook(
            lambda module, incompatible_keys: module.unfuse()
        )

    def fuse(self) -> None:
        """Build the fused inference tensors from the per-type parameters.

        The per-type loop in `forward` costs two *host synchronisations* per token
        type — `mask.any()`, and boolean-mask indexing, whose output shape is
        data-dependent — plus a handful of small kernels each. At 9 types that is
        ~18 syncs and ~50 launches per forward, which Phase 0 measured as 73% of
        the forward's dispatch on a batch the GPU computes in tens of
        microseconds.

        Fusing removes all of it:

        * the 9 entity tables become one table with per-type id offsets, so the
          lookup is a single gather;
        * the 9 feature projections become one `[n_types * d_model, MAX_FEATURES]`
          matmul, zero-padded beyond each type's own `in_features` so the unused
          columns contribute exactly nothing, followed by a gather that picks each
          token's own type slice.

        Parameters are *not* moved or renamed: `entity`/`feature` remain the
        canonical modules, so every existing checkpoint loads unchanged and
        training keeps the original numerics. Call `fuse()` again after loading
        weights or moving devices — the cache is a snapshot.
        """

        device = self.type_embedding.weight.device
        dtype = self.type_embedding.weight.dtype
        offsets = []
        running = 0
        for space in ENTITY_SPACES:
            offsets.append(running)
            running += space
        # Detached on purpose: the cache is a snapshot, so it must never be part
        # of an autograd graph that an optimizer step would then invalidate.
        entity_weight = torch.cat(
            [
                self.entity[token_type.value].weight.detach()
                for token_type in TOKEN_TYPES
            ],
            dim=0,
        )
        feature_weight = torch.zeros(
            len(TOKEN_TYPES), self.d_model, MAX_FEATURES, device=device, dtype=dtype
        )
        feature_bias = torch.zeros(
            len(TOKEN_TYPES), self.d_model, device=device, dtype=dtype
        )
        for index, token_type in enumerate(TOKEN_TYPES):
            linear = self.feature[token_type.value]
            feature_weight[index, :, : linear.in_features] = linear.weight.detach()
            feature_bias[index] = linear.bias.detach()
        self._fused = {
            "entity_weight": entity_weight,
            "entity_offsets": torch.tensor(offsets, device=device, dtype=torch.long),
            # One Linear over the padded feature width, producing every type's
            # projection at once.
            "feature_weight": feature_weight.reshape(-1, MAX_FEATURES),
            "feature_bias": feature_bias.reshape(-1),
        }

    def unfuse(self) -> None:
        """Drop the fused cache and return to the per-type loop."""

        self._fused = None

    def _apply(self, *args, **kwargs):
        """`.to()`, `.cuda()`, `.float()` etc. all land here — drop the cache.

        The cached tensors are not parameters, so they would not be moved or cast
        with everything else.
        """

        self.unfuse()
        return super()._apply(*args, **kwargs)

    def train(self, mode: bool = True):
        """Entering training mode drops the fused cache.

        The cache is a detached snapshot: training through it would update the
        real parameters while the forward kept reading stale copies, i.e. it would
        silently learn nothing. Making `train()` invalidate it means that failure
        mode cannot happen, and `forward` additionally never takes the fused path
        while `self.training` is set.
        """

        if mode:
            self.unfuse()
        return super().train(mode)

    def _forward_fused(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        fused = self._fused
        assert fused is not None
        type_ids = batch["type_ids"]
        out = self.type_embedding(type_ids) + self.aux(batch["aux_ids"])
        global_ids = fused["entity_offsets"][type_ids] + batch["entity_ids"]
        out = out + nn.functional.embedding(global_ids, fused["entity_weight"])
        projected = nn.functional.linear(
            batch["features"][..., :MAX_FEATURES],
            fused["feature_weight"],
            fused["feature_bias"],
        ).unflatten(-1, (len(TOKEN_TYPES), self.d_model))
        picker = type_ids.unsqueeze(-1).unsqueeze(-1).expand(
            *type_ids.shape, 1, self.d_model
        )
        out = out + projected.gather(-2, picker).squeeze(-2)
        return out.masked_fill(batch["pad_mask"].unsqueeze(-1), 0.0)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._fused is not None and not self.training:
            return self._forward_fused(batch)
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


def fuse_for_inference(model: nn.Module) -> bool:
    """Switch `model`'s token embedder to its fused inference path.

    Returns whether anything was fused, so a caller can report honestly instead
    of assuming. Only valid in eval mode — see `TokenEmbedder.fuse`. The fused
    path is arithmetically the same computation with a different reduction order,
    so outputs move by ~2e-6; that is a numerical change, not an exact refactor,
    and it can flip a search decision at a tie.
    """

    embedder = getattr(model, "embedder", None)
    if embedder is None or not hasattr(embedder, "fuse"):
        return False
    embedder.fuse()
    return True


def masked_policy_log_softmax(
    logits: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    return torch.log_softmax(masked, dim=-1)
