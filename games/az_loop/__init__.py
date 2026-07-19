"""Game-agnostic AlphaZero loop plumbing shared by board-game adapters."""

from .core import (
    Agent,
    GameAdapter,
    GameJob,
    MatchOutcome,
    play_match,
    run_jobs,
)
from .elo import EloLedger
from .hof import HOFEntry, HallOfFame
from .manifest import RunManifest
from .schedule import LinearSchedule, ReplayWindow
from .sprt import SPRT, SPRTResult

__all__ = [
    "Agent",
    "EloLedger",
    "GameAdapter",
    "GameJob",
    "HOFEntry",
    "HallOfFame",
    "LinearSchedule",
    "MatchOutcome",
    "ReplayWindow",
    "RunManifest",
    "SPRT",
    "SPRTResult",
    "play_match",
    "run_jobs",
]
