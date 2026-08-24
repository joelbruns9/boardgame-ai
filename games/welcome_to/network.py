"""
The Welcome To policy/value network, and the loss it is trained against.

SHAPE
─────
The encoder hands over four arrays, and the shape of the network follows from
the middle two::

    sheet_planes    (B, 4, 12, 3, 12)     per seat
    sheet_scalars   (B, 4, 45)            per seat
    viewer_plane    (B, 1, 3, 12)
    global_scalars  (B, 358)

Each seat's planes and scalars go through **one shared sheet encoder**, run four
times::

    for each seat s:  [planes_s, scalars_s] --> SHARED sheet MLP --> h_s

    trunk_in = [h_0, h_1, h_2, h_3, viewer_plane, global_scalars] --> trunk --> h

The parameters therefore see 4x the data, and a rare situation in seat 3 trains
exactly the weights a common one in seat 0 does.  The seats are **concatenated,
not pooled**: pooling would be seat-count-invariant for free, but it destroys
identity, and identity -- *which* opponent finishes plan 2 first -- is what the
plan race is about.  Padded seats are zero and carry ``seat_valid = 0``.

MLP, not convolution.  The 3x12 grid has no symmetry group: streets are not
interchangeable and the strictly-ascending rule breaks reflection.  Weight
sharing across *seats* is the sharing that pays here.

THE PER-SEAT HEAD IS CONTEXTUAL
───────────────────────────────
An isolated ``h_s`` **cannot** carry several of its own targets, and the reason
is structural rather than a matter of accuracy: ``h_s`` comes from the shared
sheet encoder, which sees only that seat's planes and scalars, while
``final_score`` includes plan-race premiums and the temp-agency rank -- both
cross-seat -- and how many turns a seat has left depends on when *anyone* ends
the game.  So the head reads the seat **and** the context::

    z_s = concat(h_s, h)        h = the main trunk output
    out = per_seat_head(z_s)    weights still shared across seats

Emitting seat-indexed outputs straight from the trunk would work too, and would
lose the shared-weight benefit that makes the per-seat structure cheap.

WHAT THE HEADS ARE FOR
──────────────────────
Only the policy and the value are consulted at play time.  Every other head
exists to constrain what the trunk is allowed to *forget*: Welcome To has an
unusually long causal path -- writing a 15 into box 0 on turn 3 costs nine future
placements and changes your turn-3 score by zero -- and ``permits`` is that
consequence made predictable from turn 3, with a short clean gradient path.  The
trunk learns capacity-consequence reasoning from the permits head and the value
head inherits it.  ``AUX_TARGETS_SPEC.md`` is the spec of record.

No auxiliary head is ever blended into the leaf value.  Deciding that a permit
is worth 0.3 points would be hand-tuning a valuation, and it is redundant
besides: if predicted permits are informative about final score, the score head
already uses them, because both read the same ``h``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from games.welcome_to import encoder as enc
from games.welcome_to import training
from games.welcome_to.macro_codec import NUM_MACRO_ACTIONS

#: Per-seat regression outputs, in head order.  Derived from the target set so
#: that adding a target and forgetting the head is impossible.
PER_SEAT_HEAD_TARGETS: tuple[str, ...] = tuple(
    name
    for name in training.PER_SEAT_TARGETS
    if name != "seat_valid" and not name.endswith("_mask")
)
#: Global regression outputs, in head order.  ``rank_logits`` is separate: it is
#: a masked softmax over finishing positions, not a regression.
GLOBAL_HEAD_TARGETS: tuple[str, ...] = ("turns_left",)

#: Loss weights **by group**, per AUX_TARGETS_SPEC §8.  Weighting per target
#: would be twenty-odd independent knobs, which is not a tunable object; these
#: five are, and they are what an ablation switches off.
#:
#: The weight is applied to the group's **mean**, once -- not to each of its
#: members.  Multiplying every member by the group coefficient would make a
#: group's real influence its coefficient times its size: ``components`` (8
#: targets at 0.2) would outweigh ``capacity`` (4 at 0.3), and the table would
#: not describe the model it configures.  Worse, it makes the weighting drift as
#: the target set changes -- AUX_TARGETS_SPEC §10 step 4 adds nine targets to
#: ``plan_race``, which would have quietly tripled that group's pull.
LOSS_WEIGHTS: dict[str, float] = {
    "policy": 1.0,
    "objective": 1.0,   # score, rank -- score dominant early
    "capacity": 0.3,
    "plan_race": 0.3,
    "components": 0.2,
}

_GROUP_OF: dict[str, str] = {
    "policy": "policy",
    "rank": "objective",
    "score": "objective",
    "permits": "capacity",
    "houses": "capacity",
    "capacity_left": "capacity",
    "turns_left": "capacity",
    "plans_completed": "plan_race",
    "turns_to_plan_0": "plan_race",
    "turns_to_plan_1": "plan_race",
    "turns_to_plan_2": "plan_race",
    **{
        f"score_{part}": "components"
        for part in (
            "parks",
            "pools",
            "estates",
            "plans",
            "temp",
            "bis",
            "permits",
            "roundabouts",
        )
    },
}
assert set(_GROUP_OF) == (
    {"policy", "rank"} | set(PER_SEAT_HEAD_TARGETS) | set(GLOBAL_HEAD_TARGETS)
)
assert set(_GROUP_OF.values()) == set(LOSS_WEIGHTS)


@dataclass(frozen=True, slots=True)
class NetConfig:
    """Widths.  ``PROJECT_PLAN.md`` M2 says start near 4M parameters."""

    sheet_hidden: int = 256
    sheet_out: int = 128
    trunk_hidden: int = 768
    trunk_blocks: int = 2
    head_hidden: int = 256
    dropout: float = 0.0


def _mlp(sizes: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.LayerNorm(b))
            layers.append(nn.ReLU(inplace=True))
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class _ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = F.relu(self.fc1(h), inplace=True)
        return x + self.drop(self.fc2(h))


class WelcomeToNet(nn.Module):
    """Shared sheet encoder, main trunk, per-seat and global heads."""

    def __init__(self, config: Optional[NetConfig] = None) -> None:
        super().__init__()
        self.config = config or NetConfig()
        c = self.config

        plane_floats = enc.SHEET_PLANES * enc.NUM_STREETS * enc.MAX_STREET_LEN
        sheet_in = plane_floats + enc.NUM_SHEET_SCALAR
        self.sheet_encoder = _mlp(
            [sheet_in, c.sheet_hidden, c.sheet_out], c.dropout
        )

        viewer_floats = enc.NUM_STREETS * enc.MAX_STREET_LEN
        trunk_in = enc.MAX_SEATS * c.sheet_out + viewer_floats + enc.NUM_GLOBAL_SCALAR
        self.trunk_in = nn.Sequential(
            nn.Linear(trunk_in, c.trunk_hidden),
            nn.LayerNorm(c.trunk_hidden),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            *[_ResidualBlock(c.trunk_hidden, c.dropout) for _ in range(c.trunk_blocks)]
        )
        self.trunk_out = nn.LayerNorm(c.trunk_hidden)

        self.policy_head = _mlp([c.trunk_hidden, c.head_hidden, NUM_MACRO_ACTIONS])
        # contextual: the seat AND the game it is in
        self.per_seat_head = _mlp(
            [c.sheet_out + c.trunk_hidden, c.head_hidden, len(PER_SEAT_HEAD_TARGETS)]
        )
        self.global_head = _mlp(
            [c.trunk_hidden, c.head_hidden, len(GLOBAL_HEAD_TARGETS) + training.MAX_RANKS]
        )

    def _features(
        self,
        sheet_planes: Tensor,
        sheet_scalars: Tensor,
        viewer_plane: Tensor,
        global_scalars: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, seats = sheet_scalars.shape[0], sheet_scalars.shape[1]
        flat = torch.cat(
            [sheet_planes.reshape(batch, seats, -1), sheet_scalars], dim=-1
        )
        h_seat = self.sheet_encoder(flat)                      # (B, seats, sheet_out)

        trunk_input = torch.cat(
            [
                h_seat.reshape(batch, -1),
                viewer_plane.reshape(batch, -1),
                global_scalars,
            ],
            dim=-1,
        )
        h = self.trunk_out(self.trunk(self.trunk_in(trunk_input)))
        return h_seat, h

    def forward_inference(
        self,
        sheet_planes: Tensor,
        sheet_scalars: Tensor,
        viewer_plane: Tensor,
        global_scalars: Tensor,
    ) -> dict[str, Tensor]:
        """Search-only heads: policy, rank, and score.

        This keeps the checkpoint and numerical path identical to ``forward``
        while avoiding construction of the unused auxiliary-target mapping.
        The shared per-seat head is still evaluated because score is one of its
        channels; splitting it would change the learned parameterization.
        """
        h_seat, h = self._features(
            sheet_planes, sheet_scalars, viewer_plane, global_scalars
        )
        batch, seats = sheet_scalars.shape[0], sheet_scalars.shape[1]
        context = h.unsqueeze(1).expand(batch, seats, h.shape[-1])
        per_seat = self.per_seat_head(torch.cat([h_seat, context], dim=-1))
        global_out = self.global_head(h)
        return {
            "policy_logits": self.policy_head(h),
            "rank_logits": global_out[:, len(GLOBAL_HEAD_TARGETS) :],
            "score": per_seat[..., PER_SEAT_HEAD_TARGETS.index("score")],
        }

    def forward(
        self,
        sheet_planes: Tensor,
        sheet_scalars: Tensor,
        viewer_plane: Tensor,
        global_scalars: Tensor,
    ) -> dict[str, Tensor]:
        """Returns raw outputs; the loss applies the masks.

        ``rank_logits`` comes back **unmasked** on purpose.  Masking dead
        finishing positions is the caller's job and has to happen on the logits,
        before any softmax -- see :func:`masked_log_softmax`.
        """
        h_seat, h = self._features(
            sheet_planes, sheet_scalars, viewer_plane, global_scalars
        )
        batch, seats = sheet_scalars.shape[0], sheet_scalars.shape[1]

        context = h.unsqueeze(1).expand(batch, seats, h.shape[-1])
        per_seat = self.per_seat_head(torch.cat([h_seat, context], dim=-1))

        global_out = self.global_head(h)
        n_global = len(GLOBAL_HEAD_TARGETS)
        out: dict[str, Tensor] = {
            "policy_logits": self.policy_head(h),
            "rank_logits": global_out[:, n_global:],
            "per_seat": per_seat,
        }
        for i, name in enumerate(GLOBAL_HEAD_TARGETS):
            out[name] = global_out[:, i]
        for i, name in enumerate(PER_SEAT_HEAD_TARGETS):
            out[name] = per_seat[..., i]
        return out


# ──────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────
def masked_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    """Log-softmax over the live classes only.

    The mask goes on the **logits**.  Softmaxing over all four positions and
    zeroing the dead ones afterwards leaves the live ones summing to less than
    one, which corrupts the expected utility *and* its variance in the same
    direction -- so the value and the confidence gate would both be wrong, and
    neither would look like noise.
    """
    return torch.log_softmax(logits.masked_fill(mask <= 0, -1e9), dim=-1)


def rank_probabilities(logits: Tensor, mask: Tensor) -> Tensor:
    """The finishing-position distribution.  Sums to 1 over the live classes."""
    return masked_log_softmax(logits, mask).exp() * (mask > 0)


def losses(
    out: Mapping[str, Tensor], batch: Mapping[str, Tensor]
) -> tuple[Tensor, dict[str, Tensor]]:
    """Total loss and its parts.

    Returned ``parts`` carries one entry per target, unweighted, plus a
    ``group_*`` entry per group holding the mean the weight was applied to.

    Two masking rules are load-bearing, and both fail silently when broken:

    * **every per-seat loss is masked by ``seat_valid``.**  Zero is a value;
      absent is not.  Because the per-seat head is shared, teaching it that a
      nonexistent player scores zero contaminates the real seats too.
    * **every masked loss is normalised by the mask sum**, via
      :func:`training.masked_mean`.  Dividing by the batch size instead
      discounts rare events in proportion to their rarity -- and the plan
      targets, whose mask is 1 in roughly one sample in seven at bootstrap, are
      exactly the ones the race argument says matter.
    """
    parts: dict[str, Tensor] = {}
    seat_valid = batch["seat_valid"]                                # (B, seats)

    # Policy: S0's one-hot teacher action and S2's MCTS visit distribution use
    # the same soft-target cross entropy. `datagen.batch` always supplies the
    # dense target, promoting old teacher samples to one-hot without changing
    # their loss.
    logits = out["policy_logits"].masked_fill(batch["legal"] <= 0, -1e9)
    if "policy" in batch:
        log_policy = torch.log_softmax(logits, dim=-1)
        parts["policy"] = -(batch["policy"] * log_policy).sum(-1).mean()
    else:  # direct unit-test/legacy callers predating S2
        parts["policy"] = F.cross_entropy(logits, batch["action"])

    # rank: cross-entropy against the distribution over finishing positions
    rank_mask = torch.stack(
        [batch[f"rank_mask_{r}"] for r in range(training.MAX_RANKS)], dim=-1
    )
    rank_target = torch.stack(
        [batch[f"rank_p_{r}"] for r in range(training.MAX_RANKS)], dim=-1
    )
    log_p = masked_log_softmax(out["rank_logits"], rank_mask)
    parts["rank"] = -(rank_target * log_p * rank_mask).sum(-1).mean()

    for name in PER_SEAT_HEAD_TARGETS:
        error = (out[name] - batch[name]) ** 2
        mask = seat_valid
        mask_name = training.MASKED_TARGETS.get(name)
        if mask_name is not None:
            mask = mask * batch[mask_name]
        parts[name] = training.masked_mean(error, mask)

    for name in GLOBAL_HEAD_TARGETS:
        parts[name] = ((out[name] - batch[name]) ** 2).mean()

    # One coefficient per group, applied to the group's mean.  See LOSS_WEIGHTS.
    grouped: dict[str, list[Tensor]] = {}
    for name, part in parts.items():
        grouped.setdefault(_GROUP_OF[name], []).append(part)

    total = None
    for group, members in grouped.items():
        mean = members[0]
        for member in members[1:]:
            mean = mean + member
        mean = mean / len(members)
        parts[f"group_{group}"] = mean
        weighted = LOSS_WEIGHTS[group] * mean
        total = weighted if total is None else total + weighted
    assert total is not None
    return total, parts


def to_tensors(
    batch: Mapping[str, "object"], device: str | torch.device = "cpu"
) -> dict[str, Tensor]:
    """Move a :func:`games.welcome_to.datagen.batch` dict onto ``device``."""
    out: dict[str, Tensor] = {}
    for name, value in batch.items():
        tensor = torch.as_tensor(value)
        if name == "action":
            tensor = tensor.long()
        else:
            tensor = tensor.float()
        out[name] = tensor.to(device)
    return out


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
