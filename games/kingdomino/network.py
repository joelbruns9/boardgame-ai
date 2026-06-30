"""
network.py — the KingdominoNet policy+value network.

ARCHITECTURE OVERVIEW
─────────────────────
Inputs (from encoder.py):
    my_board   (B, 9, 13, 13)   current player's board, castle-centred
    opp_board  (B, 9, 13, 13)   opponent's board, same frame
    flat       (B, 261)         non-spatial features

Outputs:
    own_score  (B,)             predicted NORMALIZED own final score
                                (raw_score / score_scale); no final activation
    opp_score  (B,)             predicted NORMALIZED opponent final score
    win_prob   (B,)             win probability in (0, 1), from a sigmoid
    policy     (B, 3390)        raw logits over joint (placement, pick)
                                actions; masking happens downstream via
                                action_codec.legal_mask

SHARED TRUNK (not stacked input)
Unlike AlphaZero chess/Go — where both players' pieces sit on one physical
board and are stacked as input planes — Kingdomino has two *separate*
boards.  Cell (3,4) on my board and (3,4) on the opponent's are unrelated
physical locations.  So we run each board through the SAME ResNet trunk
(shared weights, "what does a kingdom look like" learned once) and combine
their summaries afterward, rather than stacking and convolving across the
two unrelated boards.

SCORE HEADS (own_score_head, opponent_score_head)
Average+max pool each board's trunk features → (B, C) each, concatenate
with the flat vector, MLP → linear scalar (no final activation).
Output is the predicted NORMALIZED score: raw_score / score_scale.
score_scale defaults to 100.0. The two heads share the same MLP structure
but have independent weights.

WIN HEAD
Same input as score heads: [global context (4C), flat]. MLP → sigmoid
scalar in (0, 1). Predicts win probability from the encoded player's
perspective.

MARGIN (derived, not a head)
margin_value = tanh((own_norm - opp_norm) * MARGIN_GAIN)
where MARGIN_GAIN is a hyperparameter set at inference time (default 2.0).
margin_value is computed in mcts_az.py _evaluate, not here.

BILINEAR JOINT POLICY HEAD
Produces a (placement, pick) joint distribution via Q = P·W·Kᵀ where:
    P  (B, 678, D)  placement representations
        [:676]  spatial, from a 1×1 conv on my_board features, laid out to
                match the codec: index = direction*169 + y*13 + x
        [676]   DISCARD     — learned embedding
        [677]   NO_PLACEMENT — learned embedding
    K  (B, 5, D)    pick representations
        [:4]    one per current_row slot; input = slot tile features
                (from flat) + slot one-hot (encodes pick-order rank, since
                current_row is sorted by domino id) + global board context
        [4]     NO_PICK     — learned embedding
    W  (D, D)       learned bilinear interaction
Q reshaped row-major to (B, 3390) gives joint_idx = placement*5 + pick,
exactly matching action_codec.make_joint_idx.

BATCHNORM NOTE
The trunk uses BatchNorm (AlphaZero standard).  Callers MUST switch to
eval() mode for inference / self-play so running statistics are used rather
than batch statistics.  If BN causes value miscalibration as the data
distribution shifts during self-play (a known AlphaZero subtlety), switch
`norm` to "group" (GroupNorm) — a one-argument change, batch-independent.

TRAINING ISOLATION
Does NOT import evaluation.py.  The network never sees the heuristic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from games.kingdomino.encoder import FLAT_SIZE, FLAT_LAYOUT, CANVAS_SIZE, NUM_BOARD_CHANNELS
from games.kingdomino.action_codec import (
    NUM_JOINT_ACTIONS,
    NUM_SPATIAL_PLACEMENTS, NUM_DIRECTIONS, NUM_PICK_SLOTS, PICK_AXIS_SIZE,
)

# Tile-feature width per current_row slot (terrain-A 6 + crowns 1 +
# terrain-B 6 + crowns 1 + present 1 = 15).  Derived from the encoder layout.
_ROW_SLOT_WIDTH = (FLAT_LAYOUT['current_row'].stop
                   - FLAT_LAYOUT['current_row'].start) // NUM_PICK_SLOTS


def _make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        # 8 groups is a reasonable default; must divide channels.
        groups = 8 if channels % 8 == 0 else 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown norm '{norm}'; expected 'batch' or 'group'.")


class ResBlock(nn.Module):
    """Standard pre-activation-free residual block: conv-norm-relu-conv-norm + skip."""

    def __init__(self, channels: int, norm: str = "group"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = _make_norm(norm, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _make_norm(norm, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class KingdominoNet(nn.Module):
    # Architecture version stamped into checkpoints so the loader (in
    # self_play.py) can detect migrations.  Bumped to 2 for the four-head
    # rewrite (single tanh value head → own/opp score + win heads).
    checkpoint_version: int = 2

    def __init__(
        self,
        channels: int = 96,
        blocks: int = 8,
        bilinear_dim: int = 64,
        value_hidden: int = 256,
        pick_hidden: int = 128,
        flat_policy_hidden: int = 256,
        norm: str = "group",
        score_scale: float = 100.0,
    ):
        super().__init__()
        self.channels = channels
        self.bilinear_dim = D = bilinear_dim
        # Divisor mapping a raw board score to the normalized target the score
        # heads predict (raw_score / score_scale).  Stored for callers that
        # denormalize predictions; not applied inside forward.
        self.score_scale = score_scale

        # ── Shared trunk ──
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_BOARD_CHANNELS, channels, 3, padding=1, bias=False),
            _make_norm(norm, channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList(
            [ResBlock(channels, norm) for _ in range(blocks)]
        )

        # Global context = [avg_my, max_my, avg_opp, max_opp] → 4C.  Max pooling
        # preserves "is this feature present anywhere?" signals that average
        # pooling washes out — relevant for Harmony (all six terrains present)
        # and Middle Kingdom (centred geometry), where region shape matters.
        ctx_dim = 4 * channels

        # ── Score / win heads ──  input: [global context (4C), flat (FLAT_SIZE)]
        # Three independent MLPs sharing the same structure as the old value
        # head.  own/opp predict normalized scores (no final activation); win
        # is sigmoid-activated in forward().
        def _scalar_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(ctx_dim + FLAT_SIZE, value_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(value_hidden, value_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(value_hidden // 2, 1),
            )

        self.own_score_mlp = _scalar_head()
        self.opponent_score_mlp = _scalar_head()
        self.win_mlp = _scalar_head()

        # ── Policy: flat → D context, injected into BOTH placement and pick
        # representations.  This is what lets the placement head condition on
        # the tile in hand (domino_in_hand lives in `flat`); without it the
        # placement logits depend only on board geometry, not on which terrain/
        # crown tile is actually being placed.
        self.flat_policy_mlp = nn.Sequential(
            nn.Linear(FLAT_SIZE, flat_policy_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(flat_policy_hidden, D),
        )

        # ── Policy head: placement representation ──
        # 1×1 conv → (4*D) channels = D per direction.
        self.placement_conv = nn.Conv2d(channels, NUM_DIRECTIONS * D, 1)
        # DISCARD and NO_PLACEMENT embeddings (non-spatial placements).
        self.special_placement = nn.Parameter(torch.zeros(2, D))

        # ── Policy head: pick representation ──
        # per-slot input: tile features + slot one-hot + global context (4C)
        pick_in = _ROW_SLOT_WIDTH + NUM_PICK_SLOTS + ctx_dim
        self.pick_mlp = nn.Sequential(
            nn.Linear(pick_in, pick_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pick_hidden, D),
        )
        self.no_pick = nn.Parameter(torch.zeros(D))

        # ── Bilinear interaction ──
        self.W = nn.Parameter(torch.empty(D, D))

        # Cached pick-slot one-hot (moves with the module via .to(device);
        # avoids rebuilding torch.eye every forward).
        self.register_buffer("pick_slot_eye",
                             torch.eye(NUM_PICK_SLOTS), persistent=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # Bilinear W small so initial policy is near-uniform across legal moves.
        nn.init.normal_(self.W, std=0.01)
        nn.init.normal_(self.special_placement, std=0.01)
        nn.init.normal_(self.no_pick, std=0.01)
        # Small-init each scalar head's final layer so initial outputs sit near
        # 0 (and win_prob near sigmoid(0)=0.5) with healthy, non-vanishing
        # gradients.  Without this the large activations from max-pooling drive
        # the pre-activation logits far from 0, which for the win head saturates
        # the sigmoid where its gradient vanishes and training can freeze.
        for head in (self.own_score_mlp, self.opponent_score_mlp):
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)
        nn.init.normal_(self.win_mlp[-1].weight, std=0.001)
        nn.init.zeros_(self.win_mlp[-1].bias)

    # ── trunk ──
    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.res_blocks:
            x = block(x)
        return x

    def _features(self, my_board, opp_board):
        """Run both boards through the shared trunk, return (f_my, f_opp, g).

        g is the global context of shape (B, 4C): avg+max pool of each board.
        """
        B = my_board.shape[0]
        both = torch.cat([my_board, opp_board], dim=0)   # (2B, 9, 13, 13)
        feats = self._trunk(both)                         # (2B, C, 13, 13)
        f_my, f_opp = feats[:B], feats[B:]                # each (B, C, 13, 13)
        g = torch.cat([
            f_my.mean(dim=(2, 3)),  f_my.amax(dim=(2, 3)),
            f_opp.mean(dim=(2, 3)), f_opp.amax(dim=(2, 3)),
        ], dim=1)                                         # (B, 4C)
        return f_my, f_opp, g

    def forward(self, my_board, opp_board, flat):
        """
        my_board, opp_board: (B, 9, 13, 13)
        flat:                (B, 261)
        returns: own_score (B,), opp_score (B,), win_prob (B,),
                 policy_logits (B, 3390)
        own_score/opp_score are normalized scores (no activation); win_prob is
        in (0, 1).
        """
        B = my_board.shape[0]
        D = self.bilinear_dim

        f_my, _, g = self._features(my_board, opp_board)

        # ── Score / win heads ──  input: [global context (4C), flat]
        head_in = torch.cat([g, flat], dim=1)
        own_score = self.own_score_mlp(head_in).squeeze(-1)            # (B,)
        opp_score = self.opponent_score_mlp(head_in).squeeze(-1)       # (B,)
        win_prob = torch.sigmoid(self.win_mlp(head_in).squeeze(-1))    # (B,) in (0,1)

        # ── Flat policy context (carries domino_in_hand etc. into the heads) ──
        flat_ctx = self.flat_policy_mlp(flat)                    # (B, D)

        # ── Placement representation P (B, 678, D) ──
        p_spatial = self.placement_conv(f_my)                    # (B, 4D, 13, 13)
        # Reshape to match codec layout: index = direction*169 + y*13 + x.
        p_spatial = p_spatial.view(B, NUM_DIRECTIONS, D, CANVAS_SIZE, CANVAS_SIZE)
        p_spatial = p_spatial.permute(0, 1, 3, 4, 2).contiguous()  # (B, 4, 13, 13, D)
        p_spatial = p_spatial.view(B, NUM_SPATIAL_PLACEMENTS, D)   # (B, 676, D)
        p_special = self.special_placement.unsqueeze(0).expand(B, 2, D)  # (B, 2, D)
        P = torch.cat([p_spatial, p_special], dim=1)             # (B, 678, D)
        P = P + flat_ctx.unsqueeze(1)  # condition every placement on tile-in-hand

        # ── Pick representation K (B, 5, D) ──
        row = flat[:, FLAT_LAYOUT['current_row']]                # (B, 60)
        row = row.view(B, NUM_PICK_SLOTS, _ROW_SLOT_WIDTH)       # (B, 4, 15)
        slot_onehot = self.pick_slot_eye.unsqueeze(0).expand(B, -1, -1)  # (B, 4, 4)
        g_broadcast = g.unsqueeze(1).expand(B, NUM_PICK_SLOTS, g.shape[1])  # (B, 4, 4C)
        k_in = torch.cat([row, slot_onehot, g_broadcast], dim=2)  # (B, 4, 15+4+4C)
        k_picks = self.pick_mlp(k_in)                            # (B, 4, D)
        k_nopick = self.no_pick.view(1, 1, D).expand(B, 1, D)    # (B, 1, D)
        K = torch.cat([k_picks, k_nopick], dim=1)                # (B, 5, D)
        K = K + flat_ctx.unsqueeze(1)  # same flat context for the pick side

        # ── Bilinear: Q[b,p,k] = sum_{d,e} P[b,p,d] W[d,e] K[b,k,e] ──
        Q = torch.einsum('bpd,de,bke->bpk', P, self.W, K)        # (B, 678, 5)
        policy_logits = Q.reshape(B, NUM_JOINT_ACTIONS)          # (B, 3390)

        return own_score, opp_score, win_prob, policy_logits

    def forward_legal(self, my_board, opp_board, flat, legal_idx):
        """Score/win heads plus logits for only the provided legal joint indices.

        legal_idx: (B, L) int64 tensor where joint_idx = placement_idx * 5 + pick_idx.
        Returns own_score (B,), opp_score (B,), win_prob (B,), legal_logits
        (B, L).  Intended for MCTS inference, where only the legal logits are
        consumed.
        """
        B = my_board.shape[0]
        D = self.bilinear_dim
        legal_idx = legal_idx.to(device=my_board.device, dtype=torch.long)

        f_my, _, g = self._features(my_board, opp_board)
        head_in = torch.cat([g, flat], dim=1)
        own_score = self.own_score_mlp(head_in).squeeze(-1)
        opp_score = self.opponent_score_mlp(head_in).squeeze(-1)
        win_prob = torch.sigmoid(self.win_mlp(head_in).squeeze(-1))

        flat_ctx = self.flat_policy_mlp(flat)

        p_spatial = self.placement_conv(f_my)
        p_spatial = p_spatial.view(B, NUM_DIRECTIONS, D, CANVAS_SIZE, CANVAS_SIZE)
        p_spatial = p_spatial.permute(0, 1, 3, 4, 2).contiguous()
        p_spatial = p_spatial.view(B, NUM_SPATIAL_PLACEMENTS, D)
        p_special = self.special_placement.unsqueeze(0).expand(B, 2, D)
        P = torch.cat([p_spatial, p_special], dim=1)
        P = P + flat_ctx.unsqueeze(1)

        row = flat[:, FLAT_LAYOUT['current_row']]
        row = row.view(B, NUM_PICK_SLOTS, _ROW_SLOT_WIDTH)
        slot_onehot = self.pick_slot_eye.unsqueeze(0).expand(B, -1, -1)
        g_broadcast = g.unsqueeze(1).expand(B, NUM_PICK_SLOTS, g.shape[1])
        k_in = torch.cat([row, slot_onehot, g_broadcast], dim=2)
        k_picks = self.pick_mlp(k_in)
        k_nopick = self.no_pick.view(1, 1, D).expand(B, 1, D)
        K = torch.cat([k_picks, k_nopick], dim=1)
        K = K + flat_ctx.unsqueeze(1)

        placement_idx = legal_idx // PICK_AXIS_SIZE
        pick_idx = legal_idx % PICK_AXIS_SIZE
        batch = torch.arange(B, device=my_board.device).unsqueeze(1)
        p_legal = P[batch, placement_idx]
        k_legal = K[batch, pick_idx]
        legal_logits = torch.einsum('bld,de,ble->bl', p_legal, self.W, k_legal)

        return own_score, opp_score, win_prob, legal_logits


def masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Numerically stable log-softmax over legal actions only.

    logits:     (B, 3390)
    legal_mask: (B, 3390) bool — True for legal joint indices
    Illegal positions get -inf before softmax, so they receive zero
    probability and contribute nothing to the loss.
    """
    legal_mask = legal_mask.to(device=logits.device, dtype=torch.bool)
    if not legal_mask.any(dim=-1).all():
        raise ValueError(
            "masked_log_softmax received a row with no legal actions "
            "(terminal states must not be used for policy training)."
        )
    # Use HALF of finfo.min as the fill, not finfo.min itself.  log_softmax
    # computes (logit - logsumexp); with finfo.min the subtraction of a small
    # positive logsumexp overflows below float range to -inf, and a later
    # 0 * (-inf) in a masked cross-entropy becomes NaN.  Half-min leaves
    # headroom so illegal entries stay finite (and exp() still underflows to 0,
    # so legal log-probs are unchanged).
    fill = torch.finfo(logits.dtype).min / 2
    masked = torch.where(legal_mask, logits, torch.full_like(logits, fill))
    return F.log_softmax(masked, dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = KingdominoNet()
    print(f"KingdominoNet parameters: {count_parameters(net):,}")
    B = 4
    mb = torch.randn(B, NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE)
    ob = torch.randn(B, NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE)
    flat = torch.randn(B, FLAT_SIZE)
    net.eval()
    with torch.no_grad():
        own, opp, win, p = net(mb, ob, flat)
    print(f"checkpoint_version = {KingdominoNet.checkpoint_version}")
    print(f"own_score shape {tuple(own.shape)}, range [{own.min():.3f}, {own.max():.3f}]")
    print(f"opp_score shape {tuple(opp.shape)}, range [{opp.min():.3f}, {opp.max():.3f}]")
    print(f"win_prob  shape {tuple(win.shape)}, range [{win.min():.3f}, {win.max():.3f}] (expect in (0,1))")
    print(f"policy    shape {tuple(p.shape)}")
