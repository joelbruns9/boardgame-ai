"""Game-agnostic iteration orchestration and resume for AZ training loops.

The controller sequences a :class:`~games.az_loop.contract.LifecycleAdapter`
through generate -> assemble replay -> train -> (optional) gate, then applies
the soft-gate lifecycle transition and the resulting atomic checkpoint effects.
It threads a :class:`~games.az_loop.training_control.GeneratorState` across
iterations and persists enough control state in each row to reconstruct that
relationship exactly on resume.

The controller understands checkpoints only as opaque files.  It never imports
torch, reads a payload, or interprets a game.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .checkpoint_lifecycle import (
    CURRENT_BEST,
    LATEST,
    TRAINED,
    UNTRAINED,
    CheckpointArtifact,
    artifact_for,
    install,
)
from .contract import (
    AnchorRequest,
    AssembleRequest,
    GenerateRequest,
    LifecycleAdapter,
    PromotionRequest,
    TrainRequest,
)
from .training_control import (
    BootstrapPolicy,
    GeneratorMode,
    GeneratorSource,
    GeneratorState,
    PromotionAction,
    decide_transition,
    initial_state,
    is_bootstrap_eligible,
    select_generator_source,
)


# Bump when the lifecycle row schema changes in a way consumers must notice.
LOG_SCHEMA_VERSION = 1


class RunStore(Protocol):
    """Minimal persistence surface the controller needs for append/resume."""

    def append_iteration(self, row: dict[str, Any]) -> None: ...

    def iterations(self) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    mode: GeneratorMode = GeneratorMode.STRICT_GATE
    bootstrap_policy: BootstrapPolicy = BootstrapPolicy.GATE
    promotion_every: int = 1
    revert_reset_after: int = 0
    anchor_gate_every_promotions: int = 0
    buffer_autosave_every: int = 0
    seed: int = 0
    iterations: int = 1

    def validate(self) -> None:
        if self.promotion_every < 0:
            raise ValueError("promotion_every must be non-negative")
        if self.revert_reset_after < 0:
            raise ValueError("revert_reset_after must be non-negative")
        if self.anchor_gate_every_promotions < 0:
            raise ValueError("anchor_gate_every_promotions must be non-negative")
        if self.buffer_autosave_every < 0:
            raise ValueError("buffer_autosave_every must be non-negative")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")


class RunController:
    def __init__(
        self,
        *,
        adapter: LifecycleAdapter,
        store: RunStore,
        checkpoint_dir: str | Path,
        config: ControllerConfig,
    ):
        config.validate()
        self.adapter = adapter
        self.store = store
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.latest_path = self.checkpoint_dir / "latest.pt"
        self.current_best_path = self.checkpoint_dir / "current_best.pt"
        self.state: GeneratorState = initial_state(config.mode)
        self.latest_artifact: CheckpointArtifact | None = None
        self.current_best_artifact: CheckpointArtifact | None = None

    # -- lifecycle -----------------------------------------------------------

    def initialize(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        rows = self.store.iterations()
        if rows:
            self._resume(rows)
        else:
            self._bootstrap_checkpoints()

    def _bootstrap_checkpoints(self) -> None:
        """Fresh run: init learner and install identical latest + best."""

        init = self.adapter.initialize_learner(seed=self.config.seed)
        self.latest_artifact = install(
            init.path,
            self.latest_path,
            role=LATEST,
            iteration=init.iteration,
            training_state=UNTRAINED,
        )
        self.current_best_artifact = install(
            init.path,
            self.current_best_path,
            role=CURRENT_BEST,
            iteration=init.iteration,
            training_state=UNTRAINED,
        )
        self.state = initial_state(self.config.mode)

    def _resume(self, rows: list[dict[str, Any]]) -> None:
        last = rows[-1]
        try:
            control_state = last["control_state"]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(
                "run manifest has iterations but no control_state; it predates "
                "the soft-gate schema and must be resumed under strict_gate"
            ) from exc
        self.state = GeneratorState.from_row(control_state)
        if self.state.mode != self.config.mode:
            raise ValueError(
                f"cannot resume a {self.state.mode.value} run as "
                f"{self.config.mode.value}; start a new run directory instead"
            )
        best_state = TRAINED if self.state.bootstrap_state == TRAINED else UNTRAINED
        self.current_best_artifact = self._verify_on_disk(
            self.current_best_path,
            expected_sha=str(last["current_best_sha256"]),
            role=CURRENT_BEST,
            iteration=self.state.current_best_iteration,
            training_state=best_state,
        )
        self.latest_artifact = self._verify_on_disk(
            self.latest_path,
            expected_sha=str(last["latest_sha256"]),
            role=LATEST,
            iteration=int(last["iteration"]),
            training_state=TRAINED if self.state.bootstrap_state == TRAINED else UNTRAINED,
        )

    def _verify_on_disk(
        self,
        path: Path,
        *,
        expected_sha: str,
        role: str,
        iteration: int,
        training_state: str,
    ) -> CheckpointArtifact:
        if not path.is_file():
            raise FileNotFoundError(
                f"{role} checkpoint missing on resume: {path}; refusing to "
                "substitute random weights for an established checkpoint"
            )
        artifact = artifact_for(
            path, role=role, iteration=iteration, training_state=training_state
        )
        if artifact.sha256 != expected_sha:
            raise ValueError(
                f"{role} checkpoint hash mismatch on resume: {path} is "
                f"{artifact.sha256[:12]} but the manifest recorded "
                f"{expected_sha[:12]}"
            )
        return artifact

    def run(self) -> list[dict[str, Any]]:
        self.initialize()
        completed = [int(row["iteration"]) for row in self.store.iterations()]
        start = max(completed, default=-1) + 1
        return [
            self.run_iteration(iteration)
            for iteration in range(start, start + self.config.iterations)
        ]

    # -- one iteration -------------------------------------------------------

    def run_iteration(self, iteration: int) -> dict[str, Any]:
        rows = self.store.iterations()
        state = self.state

        generator_source = select_generator_source(state)
        generator_checkpoint = (
            self.latest_path
            if generator_source == GeneratorSource.LATEST
            else self.current_best_path
        )
        generator_artifact = self._role_artifact(generator_source)

        generation = self.adapter.generate(
            GenerateRequest(
                iteration=iteration,
                generator_checkpoint=generator_checkpoint,
                generator_source=generator_source,
            )
        )
        replay = self.adapter.assemble_replay(AssembleRequest(iteration=iteration))
        training = self.adapter.train(
            TrainRequest(
                iteration=iteration,
                learner_checkpoint=self.latest_path,
                replay=replay,
            )
        )
        if not training.trained:
            raise RuntimeError(
                f"iteration {iteration} produced no trained learner; refusing to "
                "advance latest.pt from an untrained result"
            )

        # The candidate snapshot and latest.pt hold identical weights right now.
        self.latest_artifact = install(
            training.candidate.path,
            self.latest_path,
            role=LATEST,
            iteration=iteration,
            training_state=TRAINED,
        )

        bootstrap = is_bootstrap_eligible(state, self.config.bootstrap_policy)
        scheduled = self._promotion_scheduled(rows, bootstrap)
        gate_decision: str | None = None
        promotion_metrics: dict[str, Any] = {}
        if scheduled:
            promotion = self.adapter.evaluate_promotion(
                PromotionRequest(
                    iteration=iteration,
                    candidate_checkpoint=self.latest_path,
                    best_checkpoint=self.current_best_path,
                )
            )
            gate_decision = promotion.decision
            promotion_metrics = dict(promotion.metrics)

        transition = decide_transition(
            state,
            policy=self.config.bootstrap_policy,
            promotion_scheduled=scheduled,
            gate_decision=gate_decision,
            revert_reset_after=self.config.revert_reset_after,
            iteration=iteration,
        )

        if transition.replace_best:
            outgoing = self.current_best_artifact
            if outgoing is not None and outgoing.training_state == TRAINED:
                self.adapter.archive_best(outgoing)
            self.current_best_artifact = install(
                self.latest_path,
                self.current_best_path,
                role=CURRENT_BEST,
                iteration=iteration,
                training_state=TRAINED,
            )

        if transition.reset_learner:
            self.latest_artifact = install(
                self.current_best_path,
                self.latest_path,
                role=LATEST,
                iteration=iteration,
                training_state=TRAINED,
            )
            self.adapter.on_learner_reset(self.current_best_path)

        anchor_metrics = self._maybe_run_anchors(transition.action, iteration)

        self.state = transition.next_state
        row = self._build_row(
            iteration=iteration,
            transition=transition,
            scheduled=scheduled,
            generator_source=generator_source,
            generator_artifact=generator_artifact,
            generation=generation,
            replay=replay,
            training=training,
            promotion_metrics=promotion_metrics,
            anchor_metrics=anchor_metrics,
        )
        self.store.append_iteration(row)
        self._emit_iteration_summary(row, generation, replay)
        self._maybe_autosave(iteration)
        return row

    @staticmethod
    def _emit_iteration_summary(row, generation, replay) -> None:
        """Compact one-line human summary for the run transcript."""

        print(
            f"iter {row['iteration']:03d} | gen {generation.generated_games} "
            f"({row['generator_source']}) | replay {replay.training_games} | "
            f"{row['promotion_action']} | best_iter {row['current_best_iteration']}"
        )

    def _maybe_autosave(self, iteration: int) -> None:
        """Owns autosave scheduling and failure policy; the adapter writes.

        A failed autosave warns and the run continues -- it must never terminate
        training.  Iterations are contiguous from 0, so ``iteration + 1`` is the
        count of completed iterations and the cadence is resume-stable.
        """

        every = self.config.buffer_autosave_every
        if every <= 0 or (iteration + 1) % every != 0:
            return
        try:
            self.adapter.autosave(iteration)
        except Exception as exc:  # noqa: BLE001 - autosave must never be fatal
            print(
                f"WARNING: buffer autosave failed after iteration {iteration}: {exc}"
            )

    # -- helpers -------------------------------------------------------------

    def _role_artifact(self, source: GeneratorSource) -> CheckpointArtifact | None:
        if source == GeneratorSource.LATEST:
            return self.latest_artifact
        return self.current_best_artifact

    def _promotion_scheduled(
        self, rows: list[dict[str, Any]], bootstrap: bool
    ) -> bool:
        if bootstrap or self.config.promotion_every <= 0:
            return False
        prior_eligible = sum(
            1
            for row in rows
            if row.get("promotion_action")
            != PromotionAction.BOOTSTRAP_PROMOTE.value
        )
        ordinal = prior_eligible + 1
        return ordinal % self.config.promotion_every == 0

    def _maybe_run_anchors(
        self, action: PromotionAction, iteration: int
    ) -> dict[str, Any] | None:
        if action not in (PromotionAction.PROMOTE, PromotionAction.BOOTSTRAP_PROMOTE):
            return None
        cadence = self.config.anchor_gate_every_promotions
        if cadence <= 0:
            return None
        promotions = 1 + sum(
            1
            for row in self.store.iterations()
            if row.get("promotion_action")
            in (
                PromotionAction.PROMOTE.value,
                PromotionAction.BOOTSTRAP_PROMOTE.value,
            )
        )
        if promotions % cadence != 0:
            return None
        result = self.adapter.evaluate_anchors(
            AnchorRequest(iteration=iteration, checkpoint=self.current_best_path)
        )
        if result is None:
            return None
        return {"passed": result.passed, **result.metrics}

    def _build_row(
        self,
        *,
        iteration: int,
        transition,
        scheduled: bool,
        generator_source: GeneratorSource,
        generator_artifact: CheckpointArtifact | None,
        generation,
        replay,
        training,
        promotion_metrics: dict[str, Any],
        anchor_metrics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        latest = self.latest_artifact
        best = self.current_best_artifact
        assert latest is not None and best is not None
        row: dict[str, Any] = {
            "iteration": iteration,
            "log_schema_version": LOG_SCHEMA_VERSION,
            "control_state": self.state.as_row(),
            "generator_mode": self.state.mode.value,
            "generator_source": generator_source.value,
            "generator_checkpoint": (
                str(generator_artifact.path) if generator_artifact else None
            ),
            "generator_sha256": (
                generator_artifact.sha256 if generator_artifact else None
            ),
            "learner_source": LATEST,
            "latest_checkpoint": str(latest.path),
            "latest_sha256": latest.sha256,
            "current_best_checkpoint": str(best.path),
            "current_best_sha256": best.sha256,
            "current_best_iteration": self.state.current_best_iteration,
            "candidate_checkpoint": str(training.candidate.path),
            "candidate_sha256": training.candidate.sha256,
            "bootstrap_state": self.state.bootstrap_state,
            "promotion_scheduled": scheduled,
            "promotion_action": transition.action.value,
            "consecutive_reverts": self.state.consecutive_reverts,
            "generated_games": generation.generated_games,
            "training_games": replay.training_games,
        }
        if generation.metrics:
            row["generation_performance"] = dict(generation.metrics)
        if replay.metrics:
            row["replay_summary"] = dict(replay.metrics)
        if training.metrics:
            row["training_performance"] = dict(training.metrics)
        if promotion_metrics:
            row["promotion_gate"] = promotion_metrics
        if anchor_metrics is not None:
            row["anchor_gates"] = anchor_metrics
        return row
