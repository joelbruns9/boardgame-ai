"""Permutation-equivariant policy and vector-value network for Kingdomino."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Final

import numpy as np
import torch
from torch import nn

from .action_codec import (
    ACTION_CODEC_VERSION,
    NUM_ACTIONS,
    d4_action_permutation,
)
from .encoder import (
    D4_ELEMENTS,
    ENCODER_VERSION,
    MAX_PLAYERS,
    NUM_BOARD_CHANNELS,
    NUM_DOMINO_FEATURES,
    NUM_GLOBAL_FEATURES,
    NUM_PLAYER_DOMINO_CHANNELS,
    NUM_PLAYER_FEATURES,
    PLAYER_FEATURE_INDEX,
    EncodedState,
)


NETWORK_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    board_channels: int = 32
    player_dim: int = 96
    domino_dim: int = 48
    global_dim: int = 64
    attention_heads: int = 4
    policy_hidden: int = 128

    def __post_init__(self) -> None:
        if self.player_dim % self.attention_heads:
            raise ValueError("player_dim must be divisible by attention_heads.")

    def manifest_fields(self) -> dict[str, object]:
        return {
            "network_version": NETWORK_VERSION,
            "encoder_version": ENCODER_VERSION,
            "action_codec_version": ACTION_CODEC_VERSION,
            "policy_size": NUM_ACTIONS,
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class NetworkOutput:
    policy_logits: torch.Tensor
    score: torch.Tensor
    rank_logits: torch.Tensor
    win_logits: torch.Tensor
    win_probs: torch.Tensor


def encoded_to_tensors(
    encoded: EncodedState, *, device: torch.device | str | None = None
) -> tuple[torch.Tensor, ...]:
    arrays = encoded.arrays()
    return tuple(
        torch.as_tensor(array, device=device).unsqueeze(0) for array in arrays
    )


class PlayerSelfAttention(nn.Module):
    """Shared masked attention with no player-position parameters."""

    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dimension // heads
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.output = nn.Linear(dimension, dimension)
        self.norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, 2 * dimension),
            nn.GELU(),
            nn.Linear(2 * dimension, dimension),
        )
        self.ff_norm = nn.LayerNorm(dimension)

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, players, _dimension = tensor.shape
        return tensor.view(batch, players, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, players: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        query = self._split(self.query(players))
        key = self._split(self.key(players))
        value = self._split(self.value(players))
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        logits = logits.masked_fill(~presence[:, None, None, :], -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        context = torch.matmul(weights, value).transpose(1, 2).contiguous()
        context = context.view(players.shape)
        interacted = self.norm(players + self.output(context))
        interacted = self.ff_norm(interacted + self.feed_forward(interacted))
        return interacted * presence.unsqueeze(-1)


class KingdominoNetwork(nn.Module):
    """S3/S4-equivariant network with invariant policy and vector values."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        config = config or NetworkConfig()
        self.config = config
        channels = config.board_channels
        self.board_encoder = nn.Sequential(
            nn.Conv2d(NUM_BOARD_CHANNELS, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.player_domino_encoder = nn.Sequential(
            nn.Linear(
                NUM_DOMINO_FEATURES + NUM_PLAYER_DOMINO_CHANNELS,
                config.domino_dim,
            ),
            nn.GELU(),
            nn.Linear(config.domino_dim, config.domino_dim),
            nn.GELU(),
        )
        self.player_encoder = nn.Sequential(
            nn.Linear(
                channels + NUM_PLAYER_FEATURES + config.domino_dim,
                config.player_dim,
            ),
            nn.GELU(),
            nn.Linear(config.player_dim, config.player_dim),
            nn.GELU(),
        )
        self.interaction = PlayerSelfAttention(
            config.player_dim, config.attention_heads
        )
        self.global_domino_encoder = nn.Sequential(
            nn.Linear(NUM_DOMINO_FEATURES, config.domino_dim),
            nn.GELU(),
            nn.Linear(config.domino_dim, config.domino_dim),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(
                NUM_GLOBAL_FEATURES + config.domino_dim,
                config.global_dim,
            ),
            nn.GELU(),
            nn.Linear(config.global_dim, config.global_dim),
            nn.GELU(),
        )
        policy_input = 2 * config.player_dim + config.global_dim
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input, config.policy_hidden),
            nn.GELU(),
            nn.Linear(config.policy_hidden, NUM_ACTIONS),
        )
        value_input = 2 * config.player_dim + config.global_dim
        self.value_trunk = nn.Sequential(
            nn.Linear(value_input, config.player_dim),
            nn.GELU(),
        )
        self.score_head = nn.Linear(config.player_dim, 1)
        self.rank_head = nn.Linear(config.player_dim, MAX_PLAYERS)
        self.win_head = nn.Linear(config.player_dim, 1)

        policy_permutations = np.stack(
            [
                d4_action_permutation(transform_id)
                for transform_id in range(len(D4_ELEMENTS))
            ]
        )
        self.register_buffer(
            "d4_policy_permutations",
            torch.from_numpy(policy_permutations.copy()),
            persistent=False,
        )

    def _forward_single_orientation(
        self,
        boards: torch.Tensor,
        player_features: torch.Tensor,
        player_dominoes: torch.Tensor,
        domino_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, players = boards.shape[:2]
        if players != MAX_PLAYERS:
            raise ValueError(f"Expected {MAX_PLAYERS} padded player slots.")
        presence = player_features[
            :, :, PLAYER_FEATURE_INDEX["present"]
        ].bool()
        actor = player_features[
            :, :, PLAYER_FEATURE_INDEX["current_actor"]
        ]

        board_embedding = self.board_encoder(
            boards.reshape(batch * players, *boards.shape[2:])
        ).reshape(batch, players, -1)
        material = domino_features[:, None].expand(-1, players, -1, -1)
        owned_domino_input = torch.cat((material, player_dominoes), dim=-1)
        owned_domino_embedding = self.player_domino_encoder(
            owned_domino_input
        ).mean(dim=2)
        player_input = torch.cat(
            (board_embedding, player_features, owned_domino_embedding), dim=-1
        )
        player_embedding = self.player_encoder(player_input) * presence.unsqueeze(-1)
        player_embedding = self.interaction(player_embedding, presence)

        count = presence.sum(dim=1, keepdim=True).clamp_min(1).to(boards.dtype)
        pooled_players = player_embedding.sum(dim=1) / count
        actor_embedding = (player_embedding * actor.unsqueeze(-1)).sum(dim=1)
        domino_pool = self.global_domino_encoder(domino_features).mean(dim=1)
        global_embedding = self.global_encoder(
            torch.cat((global_features, domino_pool), dim=-1)
        )

        policy_logits = self.policy_head(
            torch.cat((actor_embedding, pooled_players, global_embedding), dim=-1)
        )
        pooled_for_players = pooled_players[:, None].expand(-1, players, -1)
        global_for_players = global_embedding[:, None].expand(-1, players, -1)
        value_features = self.value_trunk(
            torch.cat(
                (player_embedding, pooled_for_players, global_for_players), dim=-1
            )
        )
        present_float = presence.to(value_features.dtype)
        score = self.score_head(value_features).squeeze(-1) * present_float
        rank_logits = self.rank_head(value_features) * present_float.unsqueeze(-1)
        win_logits = self.win_head(value_features).squeeze(-1)
        return policy_logits, score, rank_logits, win_logits, presence

    @staticmethod
    def _transform_boards(
        boards: torch.Tensor, rotation: int, reflected: bool
    ) -> torch.Tensor:
        transformed = torch.rot90(boards, k=rotation, dims=(-2, -1))
        if reflected:
            transformed = torch.flip(transformed, dims=(-1,))
        return transformed

    def forward(
        self,
        boards: torch.Tensor,
        player_features: torch.Tensor,
        player_dominoes: torch.Tensor,
        domino_features: torch.Tensor,
        global_features: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> NetworkOutput:
        """Average the D4 orbit after aligning policy coordinates.

        This Reynolds projection makes policy output D4-equivariant and every
        value output D4-invariant for arbitrary learned weights.
        """

        aligned_policies: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []
        ranks: list[torch.Tensor] = []
        wins: list[torch.Tensor] = []
        presence: torch.Tensor | None = None
        for transform_id, (rotation, reflected) in enumerate(D4_ELEMENTS):
            result = self._forward_single_orientation(
                self._transform_boards(boards, rotation, reflected),
                player_features,
                player_dominoes,
                domino_features,
                global_features,
            )
            policy, score, rank, win, presence = result
            # permutation maps aligned/original source -> transformed target;
            # gather transformed logits at those targets to align them back.
            permutation = self.d4_policy_permutations[transform_id]
            aligned_policies.append(policy[:, permutation])
            scores.append(score)
            ranks.append(rank)
            wins.append(win)
        assert presence is not None
        policy_logits = torch.stack(aligned_policies).mean(dim=0)
        score = torch.stack(scores).mean(dim=0)
        rank_logits = torch.stack(ranks).mean(dim=0)
        win_logits = torch.stack(wins).mean(dim=0)
        if legal_mask is not None:
            if legal_mask.shape != policy_logits.shape:
                raise ValueError("legal_mask must match the policy logits shape.")
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -torch.inf)
        present_float = presence.to(win_logits.dtype)
        player_count = presence.sum(dim=1)
        valid_rank_class = (
            torch.arange(MAX_PLAYERS, device=rank_logits.device)[None, :]
            < player_count[:, None]
        )
        rank_logits = rank_logits.masked_fill(
            presence[:, :, None] & ~valid_rank_class[:, None, :],
            -torch.inf,
        )
        rank_logits = torch.where(
            presence[:, :, None], rank_logits, torch.zeros_like(rank_logits)
        )
        masked_win_logits = win_logits.masked_fill(~presence, -torch.inf)
        win_probs = torch.softmax(masked_win_logits, dim=-1) * present_float
        return NetworkOutput(
            policy_logits=policy_logits,
            score=score,
            rank_logits=rank_logits,
            win_logits=masked_win_logits,
            win_probs=win_probs,
        )

    @torch.no_grad()
    def predict_encoded(
        self,
        encoded: EncodedState,
        *,
        legal_mask: np.ndarray | None = None,
        device: torch.device | str | None = None,
    ) -> NetworkOutput:
        if device is None:
            device = next(self.parameters()).device
        tensors = encoded_to_tensors(encoded, device=device)
        mask_tensor = (
            None
            if legal_mask is None
            else torch.as_tensor(legal_mask, device=device).unsqueeze(0)
        )
        return self(*tensors, legal_mask=mask_tensor)
