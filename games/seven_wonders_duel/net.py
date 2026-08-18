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
        #: Autograd version counters of the copied parameters at `fuse()` time.
        self._fused_versions: tuple[tuple[int, int], ...] = ()
        # The cache copies `entity`/`feature` but reads `type_embedding`/`aux`
        # live, so anything that rewrites parameters in place would leave it
        # PARTIALLY stale — a silently wrong forward rather than a loud one.
        # Invalidate on every such operation instead of documenting the hazard.
        self.register_load_state_dict_post_hook(
            lambda module, incompatible_keys: module.unfuse()
        )

    #: Fall back to the per-type loop when the fused projection's temporary
    #: ``[rows, tokens, n_types, d_model]`` would exceed this. The fused path
    #: computes all nine type projections and selects one, so that temporary
    #: grows with model width and batch size: ~87 MB at d128 with 256 rows of 74
    #: tokens, ~262 MB at d384, ~524 MB at d384 with a 512-row cap. A safety
    #: valve — no measured configuration reaches it.
    MAX_PROJECTION_BYTES = 512 * 1024 * 1024

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

        if self.training:
            raise RuntimeError(
                "fuse() is inference-only and the cache is a detached snapshot; "
                "call model.eval() before fusing"
            )
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
        self._fused_versions = self._parameter_versions()
        self._fused = {
            "entity_weight": entity_weight,
            "entity_offsets": torch.tensor(offsets, device=device, dtype=torch.long),
            # One Linear over the padded feature width, producing every type's
            # projection at once.
            "feature_weight": feature_weight.reshape(-1, MAX_FEATURES),
            "feature_bias": feature_bias.reshape(-1),
        }

    def _snapshot_sources(self):
        """Exactly the parameters `fuse()` copies."""

        for token_type in TOKEN_TYPES:
            yield self.entity[token_type.value].weight
            yield self.feature[token_type.value].weight
            yield self.feature[token_type.value].bias

    def _parameter_versions(self) -> tuple[tuple[int, int], ...]:
        """Per-parameter (version counter, storage address).

        The address catches rebinding — `parameter.data = other` swaps the
        storage without touching the version counter — which the counter alone
        misses.
        """

        return tuple(
            (tensor._version, tensor.data_ptr())
            for tensor in self._snapshot_sources()
        )

    def _snapshot_is_current(self) -> bool:
        """Has any copied parameter changed since `fuse()`?

        `train()`, `load_state_dict()` and `_apply()` cover the ordinary ways
        parameters change — but not all of them. An optimizer step taken while
        still in eval mode, or an EMA/SWA `copy_`/`lerp_`, would leave a
        *partially* stale cache: the copied `entity`/`feature` go stale while
        `type_embedding`/`aux` are still read live, which is a silently wrong
        forward rather than a loud one. Autograd's per-tensor version counter
        observes those writes, and the storage address observes rebinding, so
        consulting both is cheap certainty for them — 54 integer reads against a
        forward measured in milliseconds.

        **Not covered: in-place writes routed through `.data`**, i.e.
        `parameter.data.add_(x)` or `parameter.data.copy_(x)`. Unlike
        `.detach()`, which shares the version counter, each `.data` access
        returns a fresh view with its own counter, so such a write is invisible
        to *any* counter-based guard and leaves the storage address unchanged.
        Nothing observes it short of comparing the parameter values themselves,
        which costs the reads and the device sync that fusing exists to avoid.

        The contract is therefore: **mutating a fused module through `.data` is
        unsupported — call `unfuse()` (or `load_state_dict`/`train()`) after any
        such write.** `test_fused_cache_invalidation_contract` pins both halves.
        In this codebase nothing writes through `.data`: training builds its own
        model and `Evaluator` fuses a model it then only reads.
        """

        return self._fused_versions == self._parameter_versions()

    def unfuse(self) -> None:
        """Drop the fused cache and return to the per-type loop."""

        self._fused = None
        self._fused_versions = ()

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

    def _fits_projection_budget(self, batch: dict[str, torch.Tensor]) -> bool:
        rows, tokens = batch["type_ids"].shape
        needed = (
            rows * tokens * len(TOKEN_TYPES) * self.d_model
            * batch["features"].element_size()
        )
        return needed <= self.MAX_PROJECTION_BYTES

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
            if not self._snapshot_is_current():
                # A copied parameter was written in place behind the cache's
                # back. Drop it rather than serve a stale answer.
                self.unfuse()
            elif self._fits_projection_budget(batch):
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
    def __init__(self, d_model: int, reply: bool = False):
        super().__init__()
        self.policy = nn.Linear(d_model, NUM_ACTIONS)
        #: Predicts the OPPONENT's improved policy at the next decision.
        #:
        #: It adds no information -- Q already integrates the opponent's reply,
        #: which is why "do not take X, it uncovers Y for them" is already
        #: implicit in the recorded target. What it adds is supervision density
        #: and explicit pressure on the trunk to encode opponent intent, i.e. a
        #: better PRIOR, which is where the oracle probe located the error (raw
        #: net |err| 0.221 against search's 0.096).
        #:
        #: Optional because it is ~3% of parameters at the cloud config and
        #: because the plan requires every network change to be independently
        #: ablatable.
        self.reply = nn.Linear(d_model, NUM_ACTIONS) if reply else None
        self.value = nn.Linear(d_model, 3)
        self.joint7 = nn.Linear(d_model, 7)
        self.margin = nn.Linear(d_model, 1)
        self.military = nn.Linear(d_model, 1)
        self.science = nn.Linear(d_model, 2)

    def forward(self, readout: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {
            "policy": self.policy(readout),
            "value": self.value(readout),
            "joint7": self.joint7(readout),
            "margin": self.margin(readout).squeeze(-1),
            "military": self.military(readout).squeeze(-1),
            "science": self.science(readout),
        }
        if self.reply is not None:
            out["reply"] = self.reply(readout)
        return out


#: Head count every checkpoint written before `heads` was configurable used.
#: Readers MUST apply this to a checkpoint whose config has no ``heads`` key --
#: not `default_heads`, which disagrees at d_model >= 384.  Attention parameter
#: shapes (`in_proj_weight` [3D, D], `out_proj` [D, D]) do not depend on the head
#: count, so a wrong value loads silently and changes what the network computes.
LEGACY_HEADS = 4


def default_heads(d_model: int) -> int:
    """Head count for a new model of this width: 64 dimensions per head.

    64 is the transformer-standard head width and what ZeusAI used (768/12).
    The former hard-coded 4 gave 96- and 128-dim heads at d_model 384 and 512,
    which would have handicapped every wide arm of the sizing experiment
    against the narrow baseline it is being compared to.

    The `max` floor keeps **d_model 128 at 4 heads**, exactly as every existing
    checkpoint was built, so the sizing baseline stays bit-for-bit comparable to
    run 03 rather than quietly becoming a different 2-head model.  It also keeps
    the narrow test models (d_model 32/64) legal, where the bare ratio would ask
    for 0 heads.  At d_model >= 256 the floor is inactive and the ratio governs.
    """

    return max(4, d_model // 64)


class SWDNet(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        layers: int = 4,
        heads: int | None = None,
        pooled_readout: bool = False,
        reply_head: bool = False,
    ):
        super().__init__()
        heads = default_heads(d_model) if heads is None else int(heads)
        if heads <= 0 or d_model % heads:
            raise ValueError(
                f"d_model={d_model} is not divisible by heads={heads}"
            )
        # NOT `self.heads` -- that name is the output-head bundle assigned below.
        self.attention_heads = heads
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
        #: Concatenate masked mean- and max-pools over the real tokens with the
        #: GLOBAL token, then project back to `d_model` so `Heads` is unchanged.
        #:
        #: The point is MAX. Attention is an averaging operator, so "is there
        #: ANY token with property X" -- an existential -- is something it
        #: approximates poorly, and 7WD is full of them: is there any card that
        #: completes their sixth science symbol, any single card that swings the
        #: game, any wonder that ends it. The encoder hand-codes two of these
        #: (`sci_win_feasible`, `mil_win_feasible`); this generalises the pattern
        #: instead of adding a third bespoke flag.
        #:
        #: It must travel with the weights. Like the attention-head count, the
        #: readout changes what the model COMPUTES while leaving most parameter
        #: shapes alone, so a checkpoint rebuilt without it would load with only
        #: `readout_proj` missing and silently compute something else.
        self.pooled_readout = bool(pooled_readout)
        self.readout_proj = (
            nn.Linear(3 * d_model, d_model) if self.pooled_readout else None
        )
        self.reply_head = bool(reply_head)
        self.heads = Heads(d_model, reply=self.reply_head)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens = self.embedder(batch)
        encoded = self.encoder(tokens, src_key_padding_mask=batch["pad_mask"])
        normed = self.final_norm(encoded)
        if self.readout_proj is None:
            readout = normed[:, 0]  # GLOBAL token
        else:
            # Masking is load-bearing in both pools: a padding token must not
            # dilute the mean, and must never win the max. `-inf` fill is what
            # makes the max ignore it; the row can never be all-padding because
            # the GLOBAL token is always present, but the clamp keeps a divide
            # by zero impossible rather than merely unlikely.
            real = ~batch["pad_mask"]
            counts = real.sum(1, keepdim=True).clamp(min=1)
            weights = real.unsqueeze(-1)
            mean = (normed * weights).sum(1) / counts
            maxed = normed.masked_fill(~weights, float("-inf")).max(1).values
            readout = self.readout_proj(
                torch.cat([normed[:, 0], mean, maxed], dim=-1)
            )
        return self.heads(readout)


def fusion_is_profitable(device) -> bool:
    """Is fusing measured to pay on this device?

    Fusing trades ~9× the projection arithmetic for ~18 fewer host syncs and ~50
    fewer kernel launches per forward. That is overwhelmingly worth it where
    launches and syncs cost something, and simply extra work where they do not.
    Measured on an RTX 3070 laptop (Phase 3b review response):

    * **CUDA** — 3.17× at d128/8 rows, 1.17× at d128/256 rows, and 1.01–1.07× for
      d256L8 and d384L12 at width. A win or neutral everywhere measured, never a
      loss: bigger models converge towards neutral rather than regressing.
    * **CPU** — 1.31× at 8 rows but **0.90× at 64**. No launch overhead to
      recover, so the extra arithmetic is pure cost. Off by default here.

    Callers who have measured their own configuration can override with
    ``force=True``.
    """

    return str(device).startswith("cuda")


def fuse_for_inference(model: nn.Module, *, force: bool = False) -> bool:
    """Switch `model`'s token embedder to its fused inference path.

    Returns whether anything was fused, so a caller can report honestly rather
    than assume. Declines where fusing is not measured to pay unless `force`.
    Only valid in eval mode — see `TokenEmbedder.fuse`. The fused path is
    arithmetically the same computation with a different reduction order, so
    outputs move by ~2e-6; that is a numerical change, not an exact refactor, and
    it can flip a search decision at a tie.
    """

    embedder = getattr(model, "embedder", None)
    if embedder is None or not hasattr(embedder, "fuse"):
        return False
    if not force and not fusion_is_profitable(embedder.type_embedding.weight.device):
        return False
    embedder.fuse()
    return True


def masked_policy_log_softmax(
    logits: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    return torch.log_softmax(masked, dim=-1)
