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

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from .checkpoint_lifecycle import (
    CURRENT_BEST,
    LATEST,
    TRAINED,
    UNTRAINED,
    CheckpointArtifact,
    artifact_for,
    atomic_copy,
    atomic_write_bytes,
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
    GateLadder,
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
from .stats import (
    GateStats,
    GenerationStats,
    IterationStats,
    ModelStats,
    OutcomeStats,
    ReplayStats,
    ResourceStats,
    TrainingStats,
)


# Bump when the lifecycle row schema changes in a way consumers must notice.
LOG_SCHEMA_VERSION = 2


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
    probation_reset_after: int = 0
    anchor_gate_every_promotions: int = 0
    anchor_every_iterations: int = 0
    buffer_autosave_every: int = 0
    seed: int = 0
    iterations: int = 1
    gate_ladder: GateLadder = field(default_factory=GateLadder)

    def validate(self) -> None:
        self.gate_ladder.validate()
        if self.promotion_every < 0:
            raise ValueError("promotion_every must be non-negative")
        if self.revert_reset_after < 0:
            raise ValueError("revert_reset_after must be non-negative")
        if self.anchor_gate_every_promotions < 0:
            raise ValueError("anchor_gate_every_promotions must be non-negative")
        if self.anchor_every_iterations < 0:
            raise ValueError("anchor_every_iterations must be non-negative")
        if self.buffer_autosave_every < 0:
            raise ValueError("buffer_autosave_every must be non-negative")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if self.probation_reset_after < 0:
            raise ValueError("probation_reset_after must be non-negative")
        for name, value in (
            ("revert_reset_after", self.revert_reset_after),
            ("probation_reset_after", self.probation_reset_after),
        ):
            if value > 0 and self.mode != GeneratorMode.SOFT_GATE:
                # Either reset copies current_best over latest, discarding the
                # rolling frontier.  That only makes sense for soft_gate; in
                # latest or current_best mode it would silently destroy the
                # learner the mode exists to keep.  Refuse it loudly.
                raise ValueError(
                    f"{name} is a soft_gate-only feature; it must be 0 "
                    f"for mode {self.mode.value}"
                )


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
        # Crash-recovery journal: a pending marker plus byte backups of the two
        # rolling checkpoints, taken before an iteration mutates them.  See
        # ``_begin_iteration`` / ``_reconcile_pending``.
        self.pending_path = self.checkpoint_dir / "pending_iteration.json"
        self.recovery_dir = self.checkpoint_dir / "_recovery"
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

        # A fresh run (no rows) fully re-establishes both rolling files below, so
        # any journal from an interrupted first iteration is stale -- drop it.
        self._clear_pending()
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
        self._validate_lifecycle_config(last)
        # Recover from an iteration interrupted after it began mutating the
        # rolling checkpoints but before its row was durably appended.  Must run
        # before the on-disk hash verification below, which assumes latest/best
        # match the last committed row.
        self._reconcile_pending(rows)
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

    # -- lifecycle-config guard ----------------------------------------------

    def _lifecycle_config(self) -> dict[str, Any]:
        """The lifecycle settings that must stay fixed for the life of a run.

        Persisted in every row so a resume can reject an invocation that would
        silently change semantics (e.g. dropping ``--promotion-every 3``, which
        would corrupt the resume-stable cadence ordinal).
        """

        return {
            "promotion_every": self.config.promotion_every,
            "bootstrap_policy": self.config.bootstrap_policy.value,
            "revert_reset_after": self.config.revert_reset_after,
            "probation_reset_after": self.config.probation_reset_after,
            "anchor_gate_every_promotions": self.config.anchor_gate_every_promotions,
            "anchor_every_iterations": self.config.anchor_every_iterations,
        }

    def _validate_lifecycle_config(self, last_row: dict[str, Any]) -> None:
        persisted = last_row.get("lifecycle_config")
        if not persisted:
            # Row predates this field; nothing to validate against.
            return
        current = self._lifecycle_config()
        mismatched = {
            key: (persisted[key], current[key])
            for key in current
            if key in persisted and persisted[key] != current[key]
        }
        if mismatched:
            details = "; ".join(
                f"{key}: run started with {was!r} but resumed with {now!r}"
                for key, (was, now) in sorted(mismatched.items())
            )
            raise ValueError(
                "cannot resume with changed lifecycle configuration ("
                f"{details}); repeat the original flags or start a new run "
                "directory"
            )

    # -- crash-recovery journal ----------------------------------------------

    def _begin_iteration(self, iteration: int) -> None:
        """Snapshot the rolling checkpoints before an iteration mutates them.

        Backs up ``latest``/``current_best`` beside the run and writes a pending
        marker.  If the iteration is interrupted before its row is committed,
        ``_reconcile_pending`` rolls the rolling files back to this snapshot so a
        resume finds them consistent with the last committed row instead of
        refusing the run on a hash mismatch.
        """

        assert self.latest_artifact is not None
        assert self.current_best_artifact is not None
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        atomic_copy(self.latest_path, self.recovery_dir / "latest.pt")
        atomic_copy(self.current_best_path, self.recovery_dir / "current_best.pt")
        atomic_write_bytes(
            self.pending_path,
            json.dumps(
                {
                    "iteration": iteration,
                    "latest_sha256": self.latest_artifact.sha256,
                    "current_best_sha256": self.current_best_artifact.sha256,
                }
            ).encode("utf-8"),
        )

    def _commit_iteration(self) -> None:
        """Clear the pending marker once the iteration's row is durable."""

        self._clear_pending()

    def _clear_pending(self) -> None:
        try:
            self.pending_path.unlink()
        except FileNotFoundError:
            pass

    def _reconcile_pending(self, rows: list[dict[str, Any]]) -> None:
        if not self.pending_path.is_file():
            return
        journal = json.loads(self.pending_path.read_text(encoding="utf-8"))
        pending_iter = int(journal["iteration"])
        last_iter = int(rows[-1]["iteration"])
        if pending_iter == last_iter:
            # The row was appended but the marker outlived the commit (crash
            # between append and clear).  On-disk already matches the last row;
            # roll forward by simply dropping the stale marker.
            self._clear_pending()
            return
        if pending_iter != last_iter + 1:
            raise ValueError(
                f"pending-iteration journal is for iteration {pending_iter}, "
                f"which is neither the last committed iteration ({last_iter}) "
                "nor its successor; refusing to guess how to reconcile"
            )
        # The successor iteration was interrupted before committing its row.
        # Restore both rolling files from the pre-iteration backups.
        self._restore_backup("latest.pt", self.latest_path)
        self._restore_backup("current_best.pt", self.current_best_path)
        rollback = getattr(self.adapter, "rollback_iteration", None)
        if rollback is not None:
            rollback(pending_iter)
        self._clear_pending()

    def _restore_backup(self, name: str, destination: Path) -> None:
        backup = self.recovery_dir / name
        if not backup.is_file():
            raise FileNotFoundError(
                f"pending iteration cannot be rolled back: backup {backup} is "
                "missing; the run directory is inconsistent"
            )
        atomic_copy(backup, destination)

    def run(self) -> list[dict[str, Any]]:
        self.initialize()
        completed = [int(row["iteration"]) for row in self.store.iterations()]
        start = max(completed, default=-1) + 1
        # Remembered so the per-iteration banner can say "3/60" -- `iterations`
        # is how many *more* to run, not a total, so the position in this
        # invocation is the only honest denominator.
        self._run_span = (start, self.config.iterations)
        return [
            self.run_iteration(iteration)
            for iteration in range(start, start + self.config.iterations)
        ]

    # -- one iteration -------------------------------------------------------

    def run_iteration(self, iteration: int) -> dict[str, Any]:
        rows = self.store.iterations()
        state = self.state

        self._emit_iteration_banner(iteration)

        # Snapshot the rolling checkpoints before any mutation so an interrupted
        # iteration can be rolled back on resume rather than bricking the run.
        self._begin_iteration(iteration)

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
        if training.skipped:
            # Buffer warmup: generate, record, and stop.  Nothing is installed,
            # so latest.pt keeps whatever it held, and crucially no gate runs --
            # comparing an untrained learner against the protected best burns
            # evaluation games to re-measure a checkpoint that has not moved.
            self.adapter.record_learner(iteration, self.latest_path)
            row = self._build_warmup_row(
                iteration=iteration,
                generator_source=generator_source,
                generator_artifact=generator_artifact,
                generation=generation,
                replay=replay,
                training=training,
            )
            self.store.append_iteration(row)
            self._commit_iteration()
            print(
                f"iter {iteration:03d} | gen {generation.generated_games} "
                f"({generator_source.value if hasattr(generator_source, 'value') else generator_source}) "
                f"| replay {replay.training_games} | training skipped: "
                f"{training.skip_reason}"
            )
            self._maybe_autosave(iteration)
            return row

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
        gate_stop_reason: str | None = None
        promotion_metrics: dict[str, Any] = {}
        ladder = self.config.gate_ladder
        # W5.8: the size is resolved from the ladder position and the games
        # clock *before* the match, never from the match's own results.
        gate_games = ladder.games(state.gate_rung)
        allow_step_up = self._total_games(rows) >= ladder.floor_games
        if scheduled:
            promotion = self.adapter.evaluate_promotion(
                PromotionRequest(
                    iteration=iteration,
                    candidate_checkpoint=self.latest_path,
                    best_checkpoint=self.current_best_path,
                    gate_games=gate_games,
                )
            )
            gate_decision = promotion.decision
            promotion_metrics = dict(promotion.metrics)
            gate_stop_reason = promotion_metrics.get("stop_reason")

        transition = decide_transition(
            state,
            policy=self.config.bootstrap_policy,
            promotion_scheduled=scheduled,
            gate_decision=gate_decision,
            revert_reset_after=self.config.revert_reset_after,
            iteration=iteration,
            ladder=ladder,
            allow_step_up=allow_step_up,
            probation_reset_after=self.config.probation_reset_after,
            gate_stop_reason=gate_stop_reason,
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

        # Persist the learner only after every lifecycle mutation. Candidate
        # snapshots are pre-transition and therefore lie about reset iterations.
        self.adapter.record_learner(iteration, self.latest_path)
        anchor_metrics = self._maybe_run_anchors(transition.action, iteration)
        measurement = self._measure(iteration)

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
        if measurement:
            row["measurements"] = measurement
            stats = row.get("stats")
            if isinstance(stats, dict):
                stats.setdefault("game_specific", {})["stagnation"] = measurement
        self.store.append_iteration(row)
        # The row is now durable; the pre-iteration snapshot is no longer needed.
        self._commit_iteration()
        self._emit_iteration_summary(row, generation, replay)
        self._emit_heartbeat(row, rows)
        self._maybe_autosave(iteration)
        return row

    def _emit_iteration_banner(self, iteration: int) -> None:
        """Rule-and-title header that delimits one iteration in the transcript.

        Without this the transcript is an undifferentiated wall of step lines:
        reading run 03 meant counting `iter NNNN` markers by eye to find where
        an iteration started.
        """

        start, span = getattr(self, "_run_span", (iteration, 0))
        position = (
            f" ({iteration - start + 1}/{span})" if span else ""
        )
        print()
        print("=" * 60)
        print(f"Iteration {iteration}{position}")
        print("=" * 60)

    @staticmethod
    def _emit_iteration_summary(row, generation, replay) -> None:
        """Indented per-phase block plus the compact one-line summary.

        Everything here already existed in the committed row; only the training
        log could show it, so diagnosing run 03 meant writing a script to read
        JSONL that the transcript had implicitly discarded. The block is built
        from ``row`` alone for the same reason ``heartbeat_line`` is: the row is
        the durable record, so the transcript can never disagree with it.
        """

        stats = row.get("stats") or {}
        out: list[str] = []

        def section(label: str, body: str) -> None:
            if body:
                out.append(f"  {label}: {body}")

        schedules = (row.get("training_performance") or {}).get("schedules") or {}
        if schedules:
            bits = [f"lr={float(schedules.get('learning_rate', 0.0)):.2e}"]
            for key, name in (
                ("curriculum_mix_fraction", "curriculum"),
                ("hof_opponent_fraction", "hof"),
                ("draft_prior", "draft_prior"),
                ("seed_retain_fraction", "seed_retain"),
            ):
                if schedules.get(key) is not None:
                    bits.append(f"{name}={float(schedules[key]):.3f}")
            bits.append(f"window={schedules.get('replay_window_iterations', '-')}it")
            section("schedule", " ".join(bits))

        gen = stats.get("generation") or {}
        if gen:
            section(
                "self-play",
                f"{generation.generated_games} games "
                f"({float(gen.get('games_per_second', 0.0) or 0.0):.2f}/s) "
                f"src={row.get('generator_source')} "
                f"sims={float(gen.get('mean_sims', 0.0) or 0.0):.1f} "
                f"batch={float(gen.get('mean_batch_size', 0.0) or 0.0):.0f}",
            )

        outcomes = (stats.get("outcomes") or {}).get("terminal_reason") or {}
        if outcomes:
            total = sum(outcomes.values()) or 1
            mix = "  ".join(
                f"{name}={count} ({count / total:.0%})"
                for name, count in sorted(
                    outcomes.items(), key=lambda kv: -kv[1]
                )
            )
            section("victories", mix)

        specific = stats.get("game_specific") or {}
        lengths = specific.get("victory_type_game_length") or {}
        if lengths:
            section(
                "victory length",
                "  ".join(
                    f"{name}={float(data.get('median', 0)):.0f}"
                    f"[{float(data.get('p10', 0)):.0f}-"
                    f"{float(data.get('p90', 0)):.0f}]"
                    for name, data in sorted(lengths.items())
                ),
            )
        science = specific.get("science") or {}
        if science:
            pairs = int(science.get("pairs_completed", 0) or 0)
            wins = int(science.get("sixth_symbol_wins", 0) or 0)
            section(
                "science",
                f"sixth_symbol_wins={wins} pairs_completed={pairs} "
                f"tokens={science.get('tokens_taken', 0)} "
                f"conversion={wins / pairs if pairs else 0.0:.1%}",
            )

        perf = row.get("training_performance") or {}
        training = stats.get("training") or {}
        if training:
            train_loss = (training.get("train_losses") or {}).get("total")
            val_loss = (training.get("validation_losses") or {}).get("total")
            # A run can train without a validation split (val_fraction 0, or a
            # window too small to hold one out), so every loss here is optional.
            def number(value: float | None, spec: str) -> str:
                return "-" if value is None else format(float(value), spec)

            gap = (
                f" gap={val_loss - train_loss:+.4f}"
                if train_loss is not None and val_loss is not None
                else ""
            )
            accuracies = training.get("accuracies") or {}
            section(
                "train",
                f"{training.get('steps', 0)} steps "
                f"loss={number(train_loss, '.4f')} "
                f"val={number(val_loss, '.4f')}{gap} "
                f"value_acc={number(accuracies.get('value_acc'), '.3f')} "
                f"top1={number(accuracies.get('policy_top1'), '.3f')} "
                f"gnorm={number(training.get('gradient_norm'), '.2f')}",
            )
            # `policy` is KD's trainable_examples: when it drops below
            # `examples`, part of the window is a seat the learner did not play
            # and only its value labels are being used.
            examples = int(perf.get("examples", 0) or 0)
            policy_examples = perf.get("policy_examples")
            policy = (
                f" policy={policy_examples}"
                f"({policy_examples / examples:.0%})"
                if policy_examples is not None and examples
                else ""
            )
            section(
                "replay",
                f"{replay.training_games} games "
                f"reuse={float(perf.get('samples_per_new_position', 0.0) or 0.0):.1f}x "
                f"passes={float(perf.get('buffer_passes', 0.0) or 0.0):.2f} "
                f"new={perf.get('new_examples', 0)}{policy} "
                f"derive={float(perf.get('replay_derivation_seconds', 0.0) or 0.0):.0f}s",
            )

        # The gate is the decision of record, so it gets its distance from its
        # own bar spelled out -- "0.580 vs 0.598 needed" is the sentence run 03
        # needed on screen and never had.
        for gate in stats.get("gates") or []:
            # Sequential gates report no Wilson bounds, so the interval and the
            # margin are both optional even though the score never is.
            lcb = gate.get("wilson_lcb")
            ucb = gate.get("wilson_ucb")
            interval = (
                f" [{float(lcb):.3f},{float(ucb):.3f}]"
                if lcb is not None and ucb is not None
                else ""
            )
            margin = (
                f" lcb_margin={float(lcb) - 0.5:+.3f}" if lcb is not None else ""
            )
            section(
                "gate",
                f"{float(gate.get('score_rate', 0.0) or 0.0):.3f}{interval} "
                f"n={gate.get('games')} vs {gate.get('opponent')} "
                f"-> {gate.get('decision')} ({gate.get('stop_reason')}){margin}",
            )

        for line in out:
            print(line)
        print(
            f"iter {row['iteration']:03d} | gen {generation.generated_games} "
            f"({row['generator_source']}) | replay {replay.training_games} | "
            f"{row['promotion_action']} | best_iter {row['current_best_iteration']}"
        )

    @staticmethod
    def heartbeat_line(row: dict[str, Any]) -> str:
        """One line of run health, from the committed row alone (W6.6).

        The transcript of a multi-day run is thousands of training-step lines.
        This is the line someone reconnecting over SSH greps for: is it still
        making games, is memory flat, is anything being promoted, and does it
        still beat the bots.
        """

        stats = row.get("stats") or {}
        generation = stats.get("generation") or {}
        resources = stats.get("resources") or {}
        gates = stats.get("gates") or []
        # Both kinds are fixed-N since W5.5, so the discriminator is the reason:
        # anchors are measurements against a threshold, promotion gates report
        # which side of the decision they landed on.
        anchors = [gate for gate in gates if gate.get("stop_reason") == "fixed_n"]
        promotion = next(
            (gate for gate in gates if gate.get("stop_reason") != "fixed_n"), None
        )

        parts = [
            f"HEARTBEAT iter={row.get('iteration', -1):04d}",
            f"games/s={float(generation.get('games_per_second', 0.0) or 0.0):.3f}",
        ]
        rss = resources.get("peak_rss_bytes") or 0
        if rss:
            parts.append(f"rss={int(rss) / 1024**3:.2f}GiB")
        vram = resources.get("vram_peak_physical_bytes") or 0
        if vram:
            parts.append(f"vram={int(vram) / 1024**3:.2f}GiB")
        parts.append(f"best_iter={row.get('current_best_iteration', -1)}")
        parts.append(f"action={row.get('promotion_action', '-')}")
        if promotion is not None:
            parts.append(
                f"gate={float(promotion.get('score_rate', 0.0)):.3f}"
                f"[{float(promotion.get('wilson_lcb', 0.0)):.3f},"
                f"{float(promotion.get('wilson_ucb', 1.0)):.3f}]"
                f"n={int(promotion.get('games', 0))}"
            )
        if anchors:
            worst = min(float(gate.get("score_rate", 0.0)) for gate in anchors)
            mean = sum(float(gate.get("score_rate", 0.0)) for gate in anchors) / len(
                anchors
            )
            parts.append(f"bots={mean:.3f}(min {worst:.3f})")

        # W7c. The games-indexed self-anchor is the one number that says whether
        # the run is still learning, and it is also W0's falsifier: a flat slope
        # at L's cost is the documented case for the S/fp32 fallback. It tracks
        # the learner's own trajectory; W7a tracked the promotion lineage and so
        # went silent whenever promotions stopped.
        stagnation = (stats.get("game_specific") or {}).get("stagnation") or {}
        anchor = stagnation.get("anchor") or {}
        if anchor:
            parts.append(
                f"self={float(anchor.get('score_rate', 0.0)):.3f}"
                f"[{float(anchor.get('wilson_lcb', 0.0)):.3f},"
                f"{float(anchor.get('wilson_ucb', 1.0)):.3f}]"
                f"lag={int(anchor.get('lag_games', 0))}"
            )
        verdict = stagnation.get("verdict") or {}
        if verdict:
            slope = verdict.get("slope_per_10k_games")
            if slope is not None:
                parts.append(f"slope={float(slope):+.4f}/10k")
            if verdict.get("stagnant"):
                parts.append("STAGNANT")
        ladder = stagnation.get("ladder") or {}
        if ladder.get("active"):
            parts.append(f"rung={ladder['active']}")

        # Learning and mix signals that turned before run 03 died: value_acc
        # rolled over ~25 iterations early, and the victory mix drifts away from
        # the reference long before a gate notices.
        accuracies = (stats.get("training") or {}).get("accuracies") or {}
        for key in ("value_acc", "val_value_acc"):
            if key in accuracies:
                parts.append(f"value_acc={float(accuracies[key]):.3f}")
                break
        terminal = (stats.get("outcomes") or {}).get("terminal_reason") or {}
        total = sum(int(count) for count in terminal.values())
        if total:
            mix = "/".join(
                f"{int(terminal.get(name, 0)) / total:.2f}"
                for name in ("civilian", "scientific", "military")
            )
            parts.append(f"mix={mix}")
        promotions = row.get("promotions_in_window")
        if promotions is not None:
            parts.append(f"promotions={promotions}")

        # W7b: the revert tally and the probation tally are two mechanisms
        # reading the same gate. Which one is advancing says whether the run is
        # being arrested by proven losses or by sustained inconclusiveness --
        # and if probations dominate, the gate is underpowered rather than the
        # learner bad. Absent when both are zero, so a healthy run stays quiet.
        control = row.get("control_state") or {}
        reverts = int(control.get("consecutive_reverts", 0) or 0)
        probations = int(control.get("probations_since_decisive", 0) or 0)
        if reverts or probations:
            parts.append(f"reverts={reverts} probations={probations}")
        return " ".join(parts)

    def _emit_heartbeat(self, row: dict[str, Any], rows=None) -> None:
        """Print the heartbeat and append it to a file that survives a reconnect."""

        if rows is not None:
            row = {**row, "promotions_in_window": self._recent_promotions(rows, row)}
        line = self.heartbeat_line(row)
        print(line)
        try:
            with open(
                self.checkpoint_dir.parent / "heartbeat.log", "a", encoding="utf-8"
            ) as handle:
                handle.write(line + "\n")
        except OSError as exc:  # noqa: BLE001 - telemetry must never stop a run
            print(f"WARNING: could not append to heartbeat.log: {exc}")

    @staticmethod
    def _recent_promotions(rows: list[dict[str, Any]], row: dict[str, Any]) -> str:
        """Promotions over the last 20k games, as `count/window` (W7a).

        A run that has stopped promoting is not necessarily stagnant -- the
        anchor decides that -- but the two together separate "the gate is too
        strict" from "the learner has stopped moving".
        """

        window_games = 20_000
        recent = list(rows) + [row]
        total = 0
        promotions = 0
        for past in reversed(recent):
            total += int(past.get("generated_games", 0) or 0)
            promotions += int(past.get("promotion_action") == "promote")
            if total >= window_games:
                break
        return f"{promotions}/{total}g"

    def _measure(self, iteration: int) -> dict[str, Any] | None:
        """Run the adapter's periodic measurements, if it has any.

        A measurement is telemetry: it never changes a lifecycle decision, and a
        failure in it must not end a multi-day run.
        """

        measure = getattr(self.adapter, "measure", None)
        if measure is None:
            return None
        try:
            return measure(iteration)
        except Exception as exc:  # noqa: BLE001 - telemetry is never fatal
            print(f"WARNING: measurement failed at iteration {iteration}: {exc}")
            return None

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

    @staticmethod
    def _total_games(rows: list[dict[str, Any]]) -> int:
        """Cumulative generated games, for the ladder's step-up floor."""

        return sum(int(row.get("generated_games", 0) or 0) for row in rows)

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
        due = self._anchors_due_on_promotion(action) or self._anchors_due_on_iteration(
            iteration
        )
        if not due:
            return None
        # On an iteration-cadence anchor there may have been no promotion at
        # all, so current_best can still be an early checkpoint.  Measure the
        # learner that self-play is actually using.
        checkpoint = (
            self.current_best_path
            if self._anchors_due_on_promotion(action)
            else self.latest_path
        )
        result = self.adapter.evaluate_anchors(
            AnchorRequest(iteration=iteration, checkpoint=checkpoint)
        )
        if result is None:
            return None
        return {"passed": result.passed, "checkpoint": str(checkpoint), **result.metrics}

    def _anchors_due_on_promotion(self, action: PromotionAction) -> bool:
        if action not in (PromotionAction.PROMOTE, PromotionAction.BOOTSTRAP_PROMOTE):
            return False
        cadence = self.config.anchor_gate_every_promotions
        if cadence <= 0:
            return False
        promotions = 1 + sum(
            1
            for row in self.store.iterations()
            if row.get("promotion_action")
            in (
                PromotionAction.PROMOTE.value,
                PromotionAction.BOOTSTRAP_PROMOTE.value,
            )
        )
        return promotions % cadence == 0

    def _anchors_due_on_iteration(self, iteration: int) -> bool:
        """Out-of-distribution tracking that does not depend on promotions.

        Self-play strength and strength against opponents outside the self-play
        distribution can move in opposite directions -- run 02's iteration 11
        beat iteration 0 head to head while losing ground against two of the
        five scripted bots -- so the bot suite has to be sampled on its own
        clock rather than only when the promotion gate happens to fire.
        """

        cadence = self.config.anchor_every_iterations
        return cadence > 0 and (iteration + 1) % cadence == 0

    def _build_warmup_row(
        self,
        *,
        iteration: int,
        generator_source: GeneratorSource,
        generator_artifact: CheckpointArtifact | None,
        generation,
        replay,
        training,
    ) -> dict[str, Any]:
        """Row for an iteration that generated games but did not train.

        Deliberately carries the same identity fields as a normal row so the
        log stays uniform, with ``training_skipped`` and ``promotion_action``
        making the no-op explicit rather than leaving a gap in the sequence.
        """

        latest = self.latest_artifact
        best = self.current_best_artifact
        assert latest is not None and best is not None
        row: dict[str, Any] = {
            "iteration": iteration,
            "log_schema_version": LOG_SCHEMA_VERSION,
            "control_state": self.state.as_row(),
            "lifecycle_config": self._lifecycle_config(),
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
            "bootstrap_state": self.state.bootstrap_state,
            "training_skipped": True,
            "training_skip_reason": training.skip_reason,
            "promotion_scheduled": False,
            "promotion_action": "not_scheduled",
            "consecutive_reverts": self.state.consecutive_reverts,
            "generated_games": generation.generated_games,
            "training_games": replay.training_games,
        }
        if generation.metrics:
            row["generation_performance"] = dict(generation.metrics)
        if replay.metrics:
            row["replay_summary"] = dict(replay.metrics)
        row["stats"] = self._build_iteration_stats(
            generation=generation,
            replay=replay,
            training=training,
            promotion_metrics={},
            anchor_metrics=None,
        ).to_dict()
        return row

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
            "lifecycle_config": self._lifecycle_config(),
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
            row["promotion_gate"] = {
                key: value
                for key, value in promotion_metrics.items()
                if not key.startswith("_")
            }
        if anchor_metrics is not None:
            row["anchor_gates"] = {
                key: value
                for key, value in anchor_metrics.items()
                if not key.startswith("_")
            }
        row["stats"] = self._build_iteration_stats(
            generation=generation,
            replay=replay,
            training=training,
            promotion_metrics=promotion_metrics,
            anchor_metrics=anchor_metrics,
        ).to_dict()
        return row

    @staticmethod
    def _build_iteration_stats(
        *,
        generation,
        replay,
        training,
        promotion_metrics: dict[str, Any],
        anchor_metrics: dict[str, Any] | None,
    ) -> IterationStats:
        """Assemble and validate the schema-v2 block at the shared boundary."""

        model_payload = generation.metrics.get("model", {})
        model = (
            model_payload
            if isinstance(model_payload, ModelStats)
            else ModelStats(**dict(model_payload))
        )
        gates = []
        if getattr(training, "stats", None) is None and training.metrics:
            training_stats = TrainingStats()
        else:
            training_stats = getattr(training, "stats", None) or TrainingStats()
        promotion_stats = promotion_metrics.get("_stats")
        if promotion_stats:
            gates.append(GateStats(**dict(promotion_stats)))
        if anchor_metrics:
            gates.extend(
                GateStats(**dict(item))
                for item in anchor_metrics.get("_stats", [])
            )
        resource_payload = (
            (anchor_metrics or {}).get("_resources")
            or promotion_metrics.get("_resources")
        )
        resources = (
            ResourceStats(**dict(resource_payload))
            if resource_payload
            else (
                getattr(training, "resources", None)
                or getattr(generation, "resources", None)
                or ResourceStats()
            )
        )
        replay_stats = getattr(replay, "stats", None) or ReplayStats(
            window_games=replay.training_games
        )
        if training.metrics:
            examples = int(training.metrics.get("examples", replay_stats.examples))
            new_examples = int(
                training.metrics.get("new_examples", replay_stats.new_examples)
            )
            replay_stats = replace(
                replay_stats,
                examples=examples,
                new_examples=new_examples,
                reuse_factor=(
                    examples / new_examples if new_examples else None
                ),
                derivation_seconds=float(
                    training.metrics.get(
                        "replay_derivation_seconds",
                        replay_stats.derivation_seconds,
                    )
                ),
            )
        stats = IterationStats(
            generation=getattr(generation, "stats", None)
            or GenerationStats(games=generation.generated_games),
            outcomes=getattr(generation, "outcomes", None)
            or OutcomeStats(),
            replay=replay_stats,
            training=training_stats,
            gates=gates,
            resources=resources,
            model=model,
            game_specific=dict(getattr(generation, "game_specific", {}) or {}),
        )
        stats.validate()
        return stats
