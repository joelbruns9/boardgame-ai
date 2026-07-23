"""Seven Wonders Duel implementation of the shared lifecycle adapter.

This is the thin game-specific layer the shared ``games.az_loop`` controller
sequences.  Every method delegates to an existing :class:`PhaseDLoop` operation;
the controller owns *when* each runs and which lifecycle transition follows.

The one behavioral change from the legacy loop lives in :meth:`train`: the
learner is loaded from the checkpoint the controller selects (``latest.pt``),
not from ``current_best.pt``, so a candidate continues the rolling learner
instead of restarting from the protected best every iteration.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from games.az_loop import (
    AnchorResult,
    GenerationResult,
    PromotionResult,
    ReplayResult,
    TrainingResult,
    artifact_for,
)
from games.az_loop.checkpoint_lifecycle import TRAINED, UNTRAINED
from games.az_loop.contract import (
    AnchorRequest,
    AssembleRequest,
    GenerateRequest,
    PromotionRequest,
    TrainRequest,
)

from .phase_d import summarize_records
from .train import make_checkpoint

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from .phase_d import PhaseDLoop


class SevenWondersDuelLifecycleAdapter:
    name = "seven_wonders_duel"

    def __init__(self, loop: "PhaseDLoop"):
        self.loop = loop

    def initialize_learner(self, *, seed: int):
        loop = self.loop
        path = loop.checkpoint_dir / "_bootstrap_init.pt"
        torch.manual_seed(seed)
        checkpoint = make_checkpoint(
            loop._new_model(),
            {
                "model": "transformer",
                "d_model": loop.config.d_model,
                "layers": loop.config.layers,
                "iteration": -1,
            },
        )
        torch.save(checkpoint, path)
        return artifact_for(
            path, role="candidate", iteration=-1, training_state=UNTRAINED
        )

    def generate(self, request: GenerateRequest) -> GenerationResult:
        loop = self.loop
        model = loop.load_model(request.generator_checkpoint)
        records = loop.generate_iteration(model, request.iteration)
        return GenerationResult(
            generated_games=len(records),
            metrics={
                "performance": dict(loop.last_generation_stats),
                "summary": summarize_records(records),
            },
        )

    def assemble_replay(self, request: AssembleRequest) -> ReplayResult:
        records = self.loop.training_records(request.iteration)
        return ReplayResult(
            training_games=len(records),
            payload=records,
            metrics={"summary": summarize_records(records)},
        )

    def train(self, request: TrainRequest) -> TrainingResult:
        loop = self.loop
        candidate = loop.train_candidate(
            request.replay.payload,
            request.iteration,
            source_checkpoint=request.learner_checkpoint,
        )
        # The controller installs a returned candidate over both latest and
        # (on bootstrap/promote) the protected best.  Refuse to certify a
        # diverged or unreadable checkpoint as trained so a NaN run cannot
        # overwrite the frontier -- an interrupted train just re-runs from the
        # last committed row (see RunController's crash-recovery journal).
        self._validate_candidate(candidate, request.iteration)
        artifact = artifact_for(
            candidate,
            role="candidate",
            iteration=request.iteration,
            training_state=TRAINED,
        )
        return TrainingResult(
            candidate=artifact,
            trained=True,
            metrics=dict(loop.last_training_stats),
        )

    def _validate_candidate(self, candidate: Path, iteration: int) -> None:
        """Finite-metric + reload check before a candidate is certified trained."""

        for epoch in self.loop.last_training_stats.get("epochs") or []:
            for section in ("train", "val"):
                for key, value in (epoch.get(section) or {}).items():
                    if isinstance(value, (int, float)) and not math.isfinite(value):
                        raise RuntimeError(
                            f"iteration {iteration} training diverged: non-finite "
                            f"{section}.{key}={value}; refusing to advance latest.pt"
                        )
        try:
            model = self.loop.load_model(candidate)
        except Exception as exc:  # noqa: BLE001 - any reload failure is disqualifying
            raise RuntimeError(
                f"iteration {iteration} candidate {candidate} failed to reload: "
                f"{exc}"
            ) from exc
        for name, tensor in model.state_dict().items():
            if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
                raise RuntimeError(
                    f"iteration {iteration} candidate has non-finite weights in "
                    f"{name}; refusing to advance latest.pt"
                )

    def evaluate_promotion(self, request: PromotionRequest) -> PromotionResult:
        report = self.loop.promotion_gate(
            request.candidate_checkpoint, opponent=request.best_checkpoint
        )
        return PromotionResult(decision=report.decision, metrics=asdict(report))

    def evaluate_anchors(self, request: AnchorRequest) -> AnchorResult:
        reports = self.loop.anchor_gates(request.checkpoint)
        passed = bool(reports) and all(
            report.decision == "accept" for report in reports
        )
        return AnchorResult(
            passed=passed,
            metrics={"gates": [asdict(report) for report in reports]},
        )

    def archive_best(self, artifact) -> None:
        # Archive the OUTGOING trained best before it is overwritten.  The
        # controller never calls this for an untrained best.
        self.loop.hof.add(
            artifact.path, iteration=artifact.iteration, tag="promoted"
        )

    def on_learner_reset(self, best_checkpoint: Path) -> None:
        # Stage B (persistent optimizer clearing) lands in a later milestone.
        return None

    def autosave(self, iteration: int) -> None:
        # The controller owns cadence + failure policy; just do the atomic write.
        self.loop._save_replay_buffer()

    def contract(self) -> dict[str, Any]:
        return self.loop.adapter.contract()
