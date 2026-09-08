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

from .codec import NUM_ACTIONS, NUM_ACTION_FAMILIES
from .dataset import (
    ENTITY_SPACES,
    JOINT7_CLASSES,
    FEATURE_COUNTS,
    MAX_FEATURES,
    NUM_AUX_CARDS,
    TOKEN_TYPES,
)
from .encoder import GLOBAL_FEATURES, TABLEAU_FEATURES, TokenType
from .slot_identity import (
    AGE_OF_SLOT,
    AGE_SLOT_IDS,
    MAX_AGE,
    MAX_COVERED,
    MAX_SLOTS_PER_AGE,
    MAX_SLOT_ROW,
    MAX_SLOT_X,
    NUM_AGE_SLOTS,
    NUM_RELATIONS,
    WITHIN_AGE_INDEX,
    covered_planes,
    relation_planes,
)


# --- Workstream 1: learned Age/slot identity --------------------------------
#
# The identity is DERIVED from features the encoder already emits rather than
# appended to the schema: `row` and `x` on the tableau token, and the Age
# one-hot on the GLOBAL token, together name exactly one printed location. That
# keeps `ENCODER_SIGNATURE` still, so every existing checkpoint and every
# materialized buffer stays loadable, and it needs no matching change in the
# Rust encoder -- the appended-column alternative would have cost all three for
# an index the batch already determines.
#
# The price is that this module reads feature COLUMNS. Both lookups are by
# NAME, so an appended feature cannot silently shift them; only reordering an
# existing tuple could, which `migrate_state_dict` already forbids for its own
# reasons.
_TABLEAU_TYPE_INDEX = TOKEN_TYPES.index(TokenType.TABLEAU)
_AGE_COLUMNS = tuple(GLOBAL_FEATURES.index(f"age_{age}") for age in (1, 2, 3))
_ROW_COLUMN = TABLEAU_FEATURES.index("row")
_X_COLUMN = TABLEAU_FEATURES.index("x")
#: How many present slots overlap and cover this one. A slot becomes reachable
#: when its last coverer is taken, so `coverers == 1` is what makes an action's
#: removal an UNCOVERING rather than a step towards one.
_COVERERS_COLUMN = TABLEAU_FEATURES.index("coverers")


def _slot_lookup_table() -> torch.Tensor:
    """``[age, row, x] -> slot id + 1``; 0 means "no slot", i.e. embedding row 0.

    Age 0 is the padding plane: a row whose token 0 is not a GLOBAL token (a
    synthetic bench batch) resolves there rather than indexing out of bounds.
    """

    table = torch.zeros(
        MAX_AGE + 1, MAX_SLOT_ROW + 1, MAX_SLOT_X + 1, dtype=torch.long
    )
    for (age, row, x), slot_id in AGE_SLOT_IDS.items():
        table[age, row, x] = slot_id + 1
    return table


class TokenEmbedder(nn.Module):
    """Shared by the transformer and the MLP control model.

    Per-type modules are keyed by token-type NAME (ModuleDict), so state-dict
    keys stay stable when a new token type is appended — the §5.8a additive-
    migration hook (`train.migrate_state_dict` zero-initializes exactly the
    keys that have no counterpart in an older checkpoint).
    """

    def __init__(
        self,
        d_model: int,
        slot_embedding: bool = False,
        slot_index: bool = False,
    ):
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
        #: W1: a learned identity per printed tableau location, per Age.
        #:
        #: Zero-initialized, so loading an inherited checkpoint reproduces its
        #: computation EXACTLY -- not merely near-neutrally, as an added token
        #: type would, because this adds nothing to the sequence and therefore
        #: nothing to attention normalization. Unlike a zeroed MLP a zero table
        #: is still trainable: the gradient of a lookup does not pass through
        #: its own value, so every row that a batch touches moves on step one.
        #: That is why it needs no warm-up gate.
        #:
        #: `padding_idx=0` pins the "not a tableau token" row at zero for good.
        self.slot_embedding = bool(slot_embedding)
        self.slot = None
        if self.slot_embedding:
            self.slot = nn.Embedding(NUM_AGE_SLOTS + 1, d_model, padding_idx=0)
            nn.init.zeros_(self.slot.weight)
        #: W2 needs the same identity to order its graph nodes but none of the
        #: table, so the INDEX and the EMBEDDING are separate requests.
        self.slot_index = bool(slot_embedding or slot_index)
        if self.slot_index:
            self.register_buffer(
                "slot_lookup", _slot_lookup_table(), persistent=False
            )
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

    def slot_ids(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """W1: the Age/slot identity of every token, 0 where there is none.

        The index is reconstructed, not carried: `row` and `x` are on the
        tableau token itself, and the Age is the one-hot on the GLOBAL token,
        which is always position 0. A row whose token 0 carries no Age -- the
        Wonder draft, or a synthetic bench batch -- resolves to Age 0, whose
        lookup plane is empty, and every token there scores the zero row.

        Non-tableau tokens read whatever their own type happens to hold in the
        `row`/`x` columns, which is meaningless; `clamp` keeps that in bounds
        and the type mask discards it. Clamping is not a silent repair here,
        because no value it changes survives the mask.
        """

        assert self.slot_index, "slot_ids needs the lookup table"
        features = batch["features"]
        type_ids = batch["type_ids"]
        age = self.row_ages(batch)
        rows = features[..., _ROW_COLUMN].round().long().clamp(0, MAX_SLOT_ROW)
        xs = features[..., _X_COLUMN].round().long().clamp(0, MAX_SLOT_X)
        ids = self.slot_lookup[age.unsqueeze(-1).expand_as(rows), rows, xs]
        return ids.where(type_ids == _TABLEAU_TYPE_INDEX, torch.zeros_like(ids))

    @staticmethod
    def row_ages(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """The Age each row was encoded under, 0 where the row names none.

        Read from the GLOBAL token's one-hot, which is always position 0.
        """

        age_onehot = batch["features"][:, 0, list(_AGE_COLUMNS)]
        # `argmax` alone would call an all-zero row Age I. Ages are exclusive,
        # so the sum is 1 exactly when one is set.
        return torch.where(
            age_onehot.sum(-1) > 0.5,
            age_onehot.argmax(-1) + 1,
            torch.zeros_like(batch["type_ids"][:, 0]),
        )

    def _slot_contribution(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.slot(self.slot_ids(batch))

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
        if self.slot is not None:
            out = out + self._slot_contribution(batch)
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
        if self.slot is not None:
            out = out + self._slot_contribution(batch)
        return out.masked_fill(batch["pad_mask"].unsqueeze(-1), 0.0)


class TableauGraphLayer(nn.Module):
    """One relational message-passing step over the printed tableau graph.

    Per-relation transforms, in the basis-decomposed form R-GCN uses:
    ``W_e = sum_b a[e, b] V_b``. The point of the decomposition is cost. With
    15 relation types, a full ``d x d`` matrix each would be 15 projections per
    layer, and the plan requires this module to be cheap enough to sit in front
    of every forward on the generation path; four shared bases keep the
    per-relation transform real while paying for four.

    The algebra also reorders into something much cheaper than it looks:

        sum_e W_e (A_e h) = sum_b V_b (sum_e a[e, b] A_e) h

    so the per-relation adjacencies never have to be materialized separately.
    ``sum_e a[e, b] A_e`` is one embedding lookup over the static relation
    matrix, and what remains is four small matmuls.

    Messages are averaged over the PRESENT slots rather than over each
    relation's own degree. Per-relation normalization would make a node with
    one coverer and a node with two send messages of the same size, which is
    exactly the distinction "how exposed am I" needs.
    """

    def __init__(self, d_model: int, bases: int = 4):
        super().__init__()
        self.mix = nn.Embedding(NUM_RELATIONS, bases)
        self.basis = nn.Parameter(torch.empty(bases, d_model, d_model))
        self.self_transform = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.mix.weight, std=1.0)
        nn.init.normal_(self.basis, std=d_model ** -0.5)

    def forward(
        self,
        nodes: torch.Tensor,
        relations: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        normed = self.norm(nodes)
        # [rows, i, j, bases], zeroed where the SOURCE slot is absent.
        coefficients = self.mix(relations) * present[:, None, :, None]
        degree = present.sum(-1).clamp(min=1)
        coefficients = coefficients / degree[:, None, None, None].to(normed.dtype)
        aggregate = torch.einsum("rijb,rjd->rbid", coefficients.to(normed.dtype), normed)
        message = torch.einsum("rbid,bde->rie", aggregate, self.basis)
        update = nn.functional.gelu(message + self.self_transform(normed))
        return nodes + update * present.unsqueeze(-1)


class TableauGraph(nn.Module):
    """W2: a tableau-only graph module ahead of the main Transformer.

    Nodes are the present tableau slots, ordered by their within-Age slot
    index, so the edge set is one STATIC matrix per Age -- looked up, never
    rebuilt per position. Every non-tableau token passes through untouched.

    Why this is not just more layers. Learned slots (W1) make the cover graph
    learnable, but the model still has to discover each relationship
    statistically, and a five-step cover chain would need five neural layers
    merely to move information along it. The transitive edges make that one
    hop. Neither replaces W3: these make the topology easier to LEARN, while
    the control questions are exactly COMPUTABLE, and a learned answer is
    wrong precisely in the rare high-regret positions self-play seldom visits.

    The residual gate follows the plan: ``output = input + alpha * update``.
    At ``alpha = 0`` the module is exactly inert AND receives no gradient, so
    zero is the equivalence and ablation setting, not the training one; the
    default is the plan's small nonzero value, which lets the graph parameters
    learn from the first step.
    """

    def __init__(
        self,
        d_model: int,
        layers: int = 2,
        bases: int = 4,
        alpha: float = 1e-3,
    ):
        super().__init__()
        if layers <= 0:
            raise ValueError("a graph module with no layers is not a module")
        self.layers = nn.ModuleList(
            TableauGraphLayer(d_model, bases) for _ in range(layers)
        )
        self.alpha = float(alpha)
        self.register_buffer(
            "relation_planes",
            torch.tensor(relation_planes(), dtype=torch.long),
            persistent=False,
        )
        # `slot id - 1 -> within-Age index` and `-> Age`. Global ids are
        # consecutive across Ages, so the arithmetic would work today; the
        # tables are read instead, because "every Age has exactly 20 slots" is
        # a fact about the current layouts rather than a rule of the game.
        self.register_buffer(
            "within_age", torch.tensor(WITHIN_AGE_INDEX, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "slot_age", torch.tensor(AGE_OF_SLOT, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        slot_ids: torch.Tensor,
        row_ages: torch.Tensor,
    ) -> torch.Tensor:
        rows, _, width = tokens.shape
        nodes_wide = MAX_SLOTS_PER_AGE
        is_slot = slot_ids > 0
        global_ids = (slot_ids - 1).clamp(min=0)
        within = self.within_age[global_ids]
        # Everything that is not a tableau token is parked in a sink node that
        # is dropped before any message passing. Without it they would all
        # collide on node 0 and overwrite a real slot.
        node_index = torch.where(is_slot, within, torch.full_like(within, nodes_wide))
        picker = node_index.unsqueeze(-1).expand(-1, -1, width)

        nodes = tokens.new_zeros(rows, nodes_wide + 1, width)
        nodes.scatter_(1, picker, tokens)
        nodes = nodes[:, :nodes_wide]
        present = tokens.new_zeros(
            rows, nodes_wide + 1, dtype=torch.bool
        ).scatter_(1, node_index, torch.ones_like(node_index, dtype=torch.bool))
        present = present[:, :nodes_wide]

        relations = self.relation_planes[row_ages]
        updated = nodes
        for layer in self.layers:
            updated = layer(updated, relations, present.to(updated.dtype))
        update = torch.cat(
            [updated - nodes, tokens.new_zeros(rows, 1, width)], dim=1
        ).gather(1, picker)
        return tokens + self.alpha * update


class ContextualActionResidual(nn.Module):
    """W5a shared scorer over the legal actions in one state.

    The source card and optional Wonder are gathered from the already-contextual
    Transformer sequence. Action-family embeddings distinguish operations over
    the same card. The scorer returns the legal-only candidate axis; ``SWDNet``
    scatters it back into the frozen 1,202-action interface.
    """

    def __init__(self, d_model: int, exposes: bool = False):
        super().__init__()
        self.d_model = d_model
        #: W5b. The action's CONSEQUENCE, not just its identity: the contextual
        #: tokens of the slots this action uncovers.
        #:
        #: W5a says "this is a build of the Sawmill". W5b says "...and it
        #: uncovers those two slots". Every reviewed failure in the plan is in
        #: the second sentence -- 908370787 is a burial that uncovered a threat.
        #:
        #: Zero-initialised, so switching it on reproduces the W5a scorer
        #: exactly until it trains. Unlike a gate that would also freeze the
        #: branch, a zeroed OUTPUT projection still leaves the branch's own
        #: gradients alive: they flow through the non-zero input weights.
        self.exposes = bool(exposes)
        self.exposed = nn.Linear(d_model, d_model, bias=False) if exposes else None
        if self.exposed is not None:
            nn.init.zeros_(self.exposed.weight)
            self.register_buffer(
                "covered_planes",
                torch.tensor(covered_planes(), dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "within_age", torch.tensor(WITHIN_AGE_INDEX, dtype=torch.long),
                persistent=False,
            )
        self.family = nn.Embedding(NUM_ACTION_FAMILIES, d_model)
        self.source = nn.Linear(d_model, d_model, bias=False)
        self.wonder = nn.Linear(d_model, d_model, bias=False)
        self.action_norm = nn.LayerNorm(d_model)
        self.action_key = nn.Linear(d_model, d_model, bias=False)
        self.state_query = nn.Linear(d_model, d_model, bias=False)
        self.family_bias = nn.Embedding(NUM_ACTION_FAMILIES, 1)
        # Only the gate is zero. The scorer retains ordinary initialization so
        # its independent auxiliary loss has useful gradients from step one.
        # It is frozen for the shadow arm by default; training entry points must
        # opt in before the served policy can move away from the inherited net.
        self.gate = nn.Parameter(torch.zeros(()), requires_grad=False)

    @staticmethod
    def _gather(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        picker = indices.unsqueeze(-1).expand(*indices.shape, tokens.shape[-1])
        return tokens.gather(1, picker)

    def uncovered_tokens(
        self,
        tokens: torch.Tensor,
        batch: dict[str, torch.Tensor],
        slot_ids: torch.Tensor,
    ):
        """W5b: the contextual tokens of the slots each action would uncover.

        Reached through the geometry rather than a new encoder feature. The
        action's source token is already known (W5a); its slot identity is
        already known (W1); which slots that one covers is a property of the
        printed shape (`slot_identity.covered_slots`). So the edge costs two
        gathers and no schema change -- which also means no Rust change, since
        nothing new crosses the boundary.

        A covered slot counts only when this action's removal is what makes it
        reachable, i.e. when the slot has exactly ONE coverer. Two coverers and
        the card stays buried; the action is a step towards uncovering it, not
        an uncovering, and treating those alike is the difference between "this
        hands them the sixth symbol" and "this might, eventually".

        Returns `(covered_token, uncovers)` -- token indices and the mask --
        rather than only their embedding, so a test can check WHICH slots were
        found against the printed geometry. A branch that selected nothing
        would otherwise pass every neutrality and gradient test in the file.
        """

        assert self.exposed is not None
        rows, token_count, width = tokens.shape
        actions = batch["action_source_indices"]

        is_slot = slot_ids > 0
        within = self.within_age[(slot_ids - 1).clamp(min=0)]
        sink = torch.full_like(within, MAX_SLOTS_PER_AGE)
        node_index = torch.where(is_slot, within, sink)
        # Slot -> token, the inverse of `slot_ids`. Built by scatter, with the
        # non-tableau tokens parked in a sink that is then dropped: without it
        # they would all collide on slot 0 and overwrite a real one.
        positions = torch.arange(token_count, device=tokens.device)
        slot_token = tokens.new_zeros(
            (rows, MAX_SLOTS_PER_AGE + 1), dtype=torch.long
        )
        slot_token.scatter_(
            1, node_index, positions.expand(rows, token_count)
        )
        slot_present = tokens.new_zeros(
            (rows, MAX_SLOTS_PER_AGE + 1), dtype=torch.bool
        )
        slot_present.scatter_(
            1, node_index, torch.ones_like(node_index, dtype=torch.bool)
        )

        source_slot = within.gather(1, actions)
        source_is_slot = is_slot.gather(1, actions) & batch["action_source_present"].bool()
        ages = TokenEmbedder.row_ages(batch)
        planes = self.covered_planes[ages]                       # [rows, N, K]
        picker = source_slot.unsqueeze(-1).expand(
            *source_slot.shape, MAX_COVERED
        )
        covered = planes.gather(1, picker)                       # [rows, A, K]
        covered_exists = (covered >= 0) & source_is_slot.unsqueeze(-1)
        covered_index = torch.where(
            covered_exists, covered, torch.full_like(covered, MAX_SLOTS_PER_AGE)
        )

        flat = covered_index.reshape(rows, -1)
        covered_token = slot_token.gather(1, flat).reshape_as(covered_index)
        covered_here = slot_present.gather(1, flat).reshape_as(covered_index)

        coverers = (
            batch["features"][..., _COVERERS_COLUMN]
            .gather(1, covered_token.reshape(rows, -1))
            .reshape_as(covered_index)
        )
        # The uncovering condition. `round` because the feature is stored as a
        # count in a float tensor, not because the value is uncertain.
        uncovers = covered_exists & covered_here & (coverers.round() == 1)

        return covered_token, uncovers

    def _exposed_context(
        self,
        tokens: torch.Tensor,
        batch: dict[str, torch.Tensor],
        slot_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Sum of the contextual tokens each action actually uncovers."""

        rows, _, width = tokens.shape
        covered_token, uncovers = self.uncovered_tokens(tokens, batch, slot_ids)
        gathered = self._gather(tokens, covered_token.reshape(rows, -1))
        gathered = gathered.reshape(rows, covered_token.shape[1], MAX_COVERED, width)
        return (gathered * uncovers.unsqueeze(-1)).sum(dim=2)

    def forward(
        self,
        tokens: torch.Tensor,
        readout: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        source = self._gather(tokens, batch["action_source_indices"])
        source = source * batch["action_source_present"].unsqueeze(-1)
        wonder = self._gather(tokens, batch["action_wonder_indices"])
        wonder = wonder * batch["action_wonder_present"].unsqueeze(-1)
        action = self.family(batch["action_families"])
        action = action + self.source(source) + self.wonder(wonder)
        if self.exposed is not None:
            action = action + self.exposed(
                self._exposed_context(tokens, batch, batch["slot_ids"])
            )
        action = self.action_key(self.action_norm(action))
        query = self.state_query(readout).unsqueeze(1)
        score = (query * action).sum(dim=-1) / (self.d_model ** 0.5)
        score = score + self.family_bias(batch["action_families"]).squeeze(-1)
        return score.masked_fill(batch["legal_pad_mask"], 0.0)


class ControlHead(nn.Module):
    """W3 auxiliary target: predict exact positional control, per tableau slot.

    Reads the TOKEN SEQUENCE, not the pooled readout. The point of the arm is to
    push the trunk to represent control *per slot*, co-located with the card
    identity on that slot -- a pooled prediction could be right on aggregate
    while carrying no per-card structure, which is the failure the global
    fractions were dropped for.

    Two outputs per slot, matching the encoder-input design so the arms stay
    comparable: whether the seat can force the take, and how many of its turns
    that costs. Distance is only defined where the flag is set, so its loss is
    masked by the label rather than regressed against a sentinel.

    Droppable at inference: this head costs nothing in self-play and needs no
    table on the Rust side, which is why it is the cheapest arm to run first.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, tokens: torch.Tensor,
                batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        index = batch["control_token_index"]
        picker = index.unsqueeze(-1).expand(*index.shape, tokens.shape[-1])
        gathered = tokens.gather(1, picker)
        out = self.mlp(gathered)
        return {
            "control_reach_logit": out[..., 0],
            "control_distance_pred": out[..., 1],
        }


def _check_joint7_layout() -> None:
    """The factorisation hard-codes `JOINT7_CLASSES`'s ORDER.

    `HierarchicalValue` concatenates the three my-* classes, then the three
    opp-* classes, then draw. Reordering that tuple would leave the arithmetic
    valid and the meaning wrong -- a silent remap of victory types onto the
    wrong outcome -- so the assumption is checked here rather than trusted.
    """

    expected = (
        *(f"my_{kind}" for kind in ("civilian", "scientific", "military")),
        *(f"opp_{kind}" for kind in ("civilian", "scientific", "military")),
        "draw",
    )
    if tuple(JOINT7_CLASSES) != expected:
        raise RuntimeError(
            "JOINT7_CLASSES changed order; HierarchicalValue's factorisation "
            f"assumes {expected} and would silently mislabel victory types "
            f"under {tuple(JOINT7_CLASSES)}"
        )


_check_joint7_layout()


class HierarchicalValue(nn.Module):
    """W4: ONE distribution over winner and victory type.

    `value` and `joint7` are separate linear heads today, so nothing makes them
    agree: the model can serve 60% win while its 7-way distribution marginalises
    to 55%. Neither is wrong on its own terms and there is no fact of the matter
    about which to believe, which is the defect -- a distributional backup, or
    an advisor panel, has to pick one.

    This head emits the factors instead of the products::

        P(class) = P(outcome) * P(type | outcome)

    so the 7-way distribution and its win/draw/loss marginal are the same
    object seen from two sides, and consistency is arithmetic rather than
    something training has to discover. Draw carries no type, which is why the
    conditionals are two 3-way heads rather than one.

    **Shadow only.** `value` and `joint7` stay authoritative for search; nothing
    reads `hier_*` except the loss and diagnostics. The plan's rule is that a
    replacement head earns its promotion in an arena, and this has not run one.

    `detach` decides the one way a shadow head can still cost strength. Attached,
    its loss reaches the shared trunk and may shape representations for better
    (the KataGo lesson this project follows) or worse (gradient interference
    with policy and value). Detached, it learns from a stop-gradient copy and
    provably cannot move a single trunk weight -- so it costs throughput and
    nothing else, at the price of the representation benefit that is the usual
    reason to want an auxiliary head at all. Detached is the default because a
    strong incumbent makes those two risks asymmetric.
    """

    def __init__(self, d_model: int, detach: bool = True):
        super().__init__()
        self.detach = bool(detach)
        self.outcome = nn.Linear(d_model, 3)
        self.type_win = nn.Linear(d_model, 3)
        self.type_loss = nn.Linear(d_model, 3)

    def forward(self, readout: torch.Tensor) -> dict[str, torch.Tensor]:
        source = readout.detach() if self.detach else readout
        outcome = torch.log_softmax(self.outcome(source), dim=-1)
        win = torch.log_softmax(self.type_win(source), dim=-1)
        loss = torch.log_softmax(self.type_loss(source), dim=-1)
        # log P(class) for the seven joint classes, in JOINT7_CLASSES order --
        # the order `_check_joint7_layout` pins at import.
        joint = torch.cat(
            [
                outcome[:, 0:1] + win,
                outcome[:, 2:3] + loss,
                outcome[:, 1:2],
            ],
            dim=-1,
        )
        return {
            # Log-probabilities, not logits: these are already normalised, and
            # calling them logits invites a second softmax somewhere downstream.
            "hier_value": outcome,
            "hier_type_win": win,
            "hier_type_loss": loss,
            "hier_joint7": joint,
        }


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
        action_residual: bool = False,
        action_exposes: bool = False,
        control_head: bool = False,
        hierarchical_value: bool = False,
        hierarchical_value_detach: bool = True,
        slot_embedding: bool = False,
        graph_module: bool = False,
        graph_layers: int = 2,
        graph_bases: int = 4,
        graph_alpha: float = 1e-3,
    ):
        super().__init__()
        heads = default_heads(d_model) if heads is None else int(heads)
        if heads <= 0 or d_model % heads:
            raise ValueError(
                f"d_model={d_model} is not divisible by heads={heads}"
            )
        # NOT `self.heads` -- that name is the output-head bundle assigned below.
        self.attention_heads = heads
        #: W1. Recorded on the model because it changes which parameters
        #: exist, so a checkpoint rebuilt without it would load everything but
        #: `embedder.slot.weight` and quietly compute the pre-W1 network.
        self.slot_embedding = bool(slot_embedding)
        #: W2. Its node ordering is W1's slot identity, so it asks the embedder
        #: for the INDEX; it does not need the learned table and can run
        #: without it, which is what makes the two independently ablatable.
        self.graph_module = bool(graph_module)
        self.graph_layers = int(graph_layers)
        self.graph_bases = int(graph_bases)
        self.graph_alpha = float(graph_alpha)
        self.embedder = TokenEmbedder(
            d_model,
            slot_embedding=self.slot_embedding,
            slot_index=self.graph_module or bool(action_exposes and action_residual),
        )
        self.graph = (
            TableauGraph(
                d_model,
                layers=self.graph_layers,
                bases=self.graph_bases,
                alpha=self.graph_alpha,
            )
            if self.graph_module
            else None
        )
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
        self.action_residual = bool(action_residual)
        #: W5b. Needs W1's slot INDEX (not its table) to find what an action
        #: uncovers, so it asks the embedder for one exactly as W2 does.
        self.action_exposes = bool(action_exposes and action_residual)
        self.heads = Heads(d_model, reply=self.reply_head)
        self.action_scorer = (
            ContextualActionResidual(d_model, exposes=self.action_exposes)
            if self.action_residual
            else None
        )
        self.control_head = bool(control_head)
        self.control_scorer = ControlHead(d_model) if self.control_head else None
        #: W4. Recorded on the model like every other switch, and `detach` with
        #: it: it changes which gradients exist, so a rebuild that dropped it
        #: would train a different thing while loading every weight.
        self.hierarchical_value = bool(hierarchical_value)
        self.hierarchical_value_detach = bool(hierarchical_value_detach)
        self.hier_value = (
            HierarchicalValue(d_model, detach=self.hierarchical_value_detach)
            if self.hierarchical_value
            else None
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens = self.embedder(batch)
        if self.graph is not None:
            tokens = self.graph(
                tokens,
                self.embedder.slot_ids(batch),
                self.embedder.row_ages(batch),
            )
        encoded = self.encoder(tokens, src_key_padding_mask=batch["pad_mask"])
        normed = self.final_norm(encoded)
        if self.readout_proj is None:
            readout = normed[:, 0]  # GLOBAL token
        else:
            # Masking is load-bearing in both pools: a padding token must not
            # dilute the mean, and must never win the max. `-inf` fill is what
            # makes the max ignore it; the row can never be all-padding because
            # The GLOBAL token is always present, so an all-pad row cannot
            # occur. The clamp guards the MEAN's divide-by-zero only: the max
            # below would still take -inf on such a row and send it through
            # `readout_proj`. Stated precisely because the previous comment
            # claimed the clamp made failure impossible, which it does not.
            real = ~batch["pad_mask"]
            counts = real.sum(1, keepdim=True).clamp(min=1)
            weights = real.unsqueeze(-1)
            mean = (normed * weights).sum(1) / counts
            maxed = normed.masked_fill(~weights, float("-inf")).max(1).values
            readout = self.readout_proj(
                torch.cat([normed[:, 0], mean, maxed], dim=-1)
            )
        out = self.heads(readout)
        if self.action_scorer is not None:
            if self.action_scorer.exposed is not None:
                # Computed once and passed in, rather than recomputed inside the
                # scorer: W1 and W2 may already have it for this batch.
                batch = {**batch, "slot_ids": self.embedder.slot_ids(batch)}
            candidate_logits = self.action_scorer(normed, readout, batch)
            # Under autocast the scorer returns bf16 while `policy` is fp32, and
            # `scatter_add_` requires both to match -- so W5a crashed outright at
            # `--precision bf16`, which is what every cloud run uses. Match the
            # destination rather than the source: the residual is added to
            # `policy`, so fp32 is the dtype it has to end up in anyway.
            candidate_logits = candidate_logits.to(out["policy"].dtype)
            # Padded candidates land in a disposable sink, never action 0.
            # This also isolates their gradients if scorer masking changes.
            residual = out["policy"].new_zeros((candidate_logits.shape[0], NUM_ACTIONS + 1))
            residual.scatter_add_(1, batch["legal_indices"], candidate_logits)
            residual = residual[:, :NUM_ACTIONS]
            out["action_policy"] = residual
            alpha = torch.tanh(self.action_scorer.gate)
            out["policy"] = out["policy"] + alpha * residual
        if self.control_scorer is not None and "control_token_index" in batch:
            out.update(self.control_scorer(normed, batch))
        if self.hier_value is not None:
            out.update(self.hier_value(readout))
        return out


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
