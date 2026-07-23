"""Typed request/result contract between the shared controller and a game.

The controller owns *when* each operation happens and *what* lifecycle
transition follows.  A game adapter owns *how* each happens and which existing
metrics it returns.  These objects replace passing an unstructured dict through
the controller.

The Protocol is named :class:`LifecycleAdapter` deliberately: ``games.az_loop``
already defines ``core.GameAdapter`` -- the small engine/match boundary
(``new_game``/``step``/``terminal``/``outcome``) used by orchestration and
evaluation.  That is an unrelated concern; a lifecycle adapter composes with a
game adapter, it does not replace it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .checkpoint_lifecycle import CheckpointArtifact
from .training_control import GeneratorSource


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    iteration: int
    generator_checkpoint: Path
    generator_source: GeneratorSource


@dataclass(frozen=True, slots=True)
class AssembleRequest:
    iteration: int


@dataclass(frozen=True, slots=True)
class TrainRequest:
    iteration: int
    learner_checkpoint: Path
    replay: "ReplayResult"


@dataclass(frozen=True, slots=True)
class PromotionRequest:
    iteration: int
    candidate_checkpoint: Path
    best_checkpoint: Path


@dataclass(frozen=True, slots=True)
class AnchorRequest:
    iteration: int
    checkpoint: Path


@dataclass(frozen=True, slots=True)
class GenerationResult:
    generated_games: int
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReplayResult:
    training_games: int
    payload: Any = None  # opaque handle the adapter hands back to ``train``
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    candidate: CheckpointArtifact  # immutable candidate_XXXX.pt snapshot
    trained: bool = True
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PromotionResult:
    decision: str  # "accept" | "continue" | "reject"
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnchorResult:
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)


class LifecycleAdapter(Protocol):
    """Game-specific operations the controller sequences.

    The controller never interprets cards, tiles, network heads, Rust engines,
    replay record classes, or model architectures.  It calls these in a fixed
    order and applies the resulting lifecycle transition.
    """

    name: str

    def initialize_learner(self, *, seed: int) -> CheckpointArtifact:
        """Write a fresh untrained checkpoint and describe it.

        The controller installs the returned file as both ``latest`` and
        ``current_best`` so a bootstrap run starts from identical weights.
        """
        ...

    def generate(self, request: GenerateRequest) -> GenerationResult: ...

    def assemble_replay(self, request: AssembleRequest) -> ReplayResult: ...

    def train(self, request: TrainRequest) -> TrainingResult: ...

    def evaluate_promotion(self, request: PromotionRequest) -> PromotionResult: ...

    def evaluate_anchors(self, request: AnchorRequest) -> AnchorResult | None:
        """Optional anchor gate; return ``None`` when not run this iteration."""
        ...

    def archive_best(self, artifact: CheckpointArtifact) -> None:
        """Copy an *outgoing* trained best into HOF before it is overwritten.

        Never called for an ``untrained`` outgoing best (bootstrap weights must
        not enter the protected HOF).
        """
        ...

    def on_learner_reset(self, best_checkpoint: Path) -> None:
        """Hook for a revert-reset: clear persisted optimizer state, etc."""
        ...

    def autosave(self, iteration: int) -> None:
        """Atomically export the replay buffer.

        The controller decides *when* this runs (cadence) and guarantees a
        failure here is non-fatal; the adapter only performs the write.
        """
        ...

    def contract(self) -> dict[str, Any]: ...
