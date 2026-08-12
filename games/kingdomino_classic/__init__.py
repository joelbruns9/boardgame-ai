"""Classic 3-4 player Kingdomino research package.

This package is intentionally separate from :mod:`games.kingdomino`, which is
the established two-player Mighty Duel implementation.  The implementations
may share proven utilities later, after the classic rules engine and its model
interfaces are stable.
"""

from .config import GAME_ID, RULES_VERSION, ClassicGameConfig
from .game import ClassicGameState, GameResult, Phase, PickAction, TurnAction

__all__ = [
    "ClassicGameConfig",
    "ClassicGameState",
    "GAME_ID",
    "GameResult",
    "Phase",
    "PickAction",
    "RULES_VERSION",
    "TurnAction",
]
