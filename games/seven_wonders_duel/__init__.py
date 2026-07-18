"""7 Wonders Duel rules engine and AI project."""

from .bots import GreedyBot, RandomBot, play_game, play_series
from .codec import (
    NUM_ACTIONS,
    decode_action,
    encode_action,
    legal_action_indices,
    legal_action_mask,
)
from .data import BackType, back_type_of
from .engine import (
    Action,
    ActionUse,
    apply_action,
    legal_actions,
    resolve_pending_choice,
    score_player,
    start_next_age,
)
from .game import (
    ChanceKind,
    GameState,
    HiddenInformationError,
    ResolvedChance,
    StepResult,
    VictoryType,
    new_game,
)
from .pool import (
    UnseenPool,
    enumerate_card_reveal,
    enumerate_great_library,
    enumerate_wonder_flip,
    resample_hidden,
    unseen_pool,
)

__all__ = [
    "Action",
    "ActionUse",
    "BackType",
    "ChanceKind",
    "GameState",
    "GreedyBot",
    "HiddenInformationError",
    "NUM_ACTIONS",
    "RandomBot",
    "ResolvedChance",
    "StepResult",
    "UnseenPool",
    "VictoryType",
    "apply_action",
    "back_type_of",
    "decode_action",
    "encode_action",
    "legal_action_indices",
    "legal_action_mask",
    "enumerate_card_reveal",
    "enumerate_great_library",
    "enumerate_wonder_flip",
    "legal_actions",
    "new_game",
    "play_game",
    "play_series",
    "resample_hidden",
    "resolve_pending_choice",
    "score_player",
    "start_next_age",
    "unseen_pool",
]
