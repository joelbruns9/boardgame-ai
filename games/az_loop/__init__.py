"""Game-agnostic AlphaZero loop plumbing shared by board-game adapters."""

from .checkpoint_lifecycle import (
    CheckpointArtifact,
    artifact_for,
    atomic_copy,
    atomic_write_bytes,
    install,
    sha256_file,
)
from .contract import (
    AnchorRequest,
    AnchorResult,
    AssembleRequest,
    GenerateRequest,
    GenerationResult,
    LifecycleAdapter,
    PromotionRequest,
    PromotionResult,
    ReplayResult,
    TrainRequest,
    TrainingResult,
)
from .core import (
    Agent,
    GameAdapter,
    GameJob,
    MatchOutcome,
    play_match,
    run_jobs,
    run_jobs_in_processes,
)
from .elo import EloLedger
from .hof import HOFEntry, HallOfFame
from .manifest import RunManifest
from .run_controller import ControllerConfig, RunController, RunStore
from .run_log import RunLog
from .schedule import LinearSchedule, ReplayWindow
from .sprt import SPRT, SPRTResult
from .training_control import (
    BootstrapPolicy,
    GeneratorMode,
    GeneratorSource,
    GeneratorState,
    PromotionAction,
    TransitionResult,
    decide_transition,
    initial_state,
    select_generator_source,
)

__all__ = [
    "Agent",
    "AnchorRequest",
    "AnchorResult",
    "AssembleRequest",
    "BootstrapPolicy",
    "CheckpointArtifact",
    "ControllerConfig",
    "EloLedger",
    "GameAdapter",
    "GameJob",
    "GenerateRequest",
    "GenerationResult",
    "GeneratorMode",
    "GeneratorSource",
    "GeneratorState",
    "HOFEntry",
    "HallOfFame",
    "LifecycleAdapter",
    "LinearSchedule",
    "MatchOutcome",
    "PromotionAction",
    "PromotionRequest",
    "PromotionResult",
    "ReplayResult",
    "ReplayWindow",
    "RunController",
    "RunLog",
    "RunManifest",
    "RunStore",
    "SPRT",
    "SPRTResult",
    "TrainRequest",
    "TrainingResult",
    "TransitionResult",
    "artifact_for",
    "atomic_copy",
    "atomic_write_bytes",
    "decide_transition",
    "initial_state",
    "install",
    "play_match",
    "run_jobs",
    "run_jobs_in_processes",
    "select_generator_source",
    "sha256_file",
]
