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
import time
from collections import Counter
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
    GenerationStats,
    GateStats,
    ModelStats,
    OutcomeStats,
    ReplayStats,
    ResourceMonitor,
    TrainingStats,
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

from .buffer import GameRecord, OPPONENT_TYPES, resolve_opponent_type
from .dataset import GameDerivationStats
from .phase_d import summarize_records
from .train import make_checkpoint

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from .phase_d import PhaseDLoop


def _distribution(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return float(ordered[index])

    return {
        "mean": sum(ordered) / len(ordered),
        "median": percentile(0.5),
        "p10": percentile(0.1),
        "p90": percentile(0.9),
    }


def _record_identity(record: GameRecord) -> tuple[int, int | None, str, int]:
    return (
        int(record.seed),
        record.iteration,
        str(record.trajectory_digest),
        len(record.moves),
    )


class _GameSpecificCollector:
    """Merge observations from required example derivation into W3 stats."""

    def __init__(self, records: list[GameRecord], target: dict[str, Any]):
        self._remaining = Counter(_record_identity(record) for record in records)
        self._expected = len(records)
        self._observed = 0
        self._age_three_civilian = 0
        self._target = target

    def observe(self, record: GameRecord, stats: GameDerivationStats) -> None:
        identity = _record_identity(record)
        if self._remaining[identity] <= 0:
            return
        self._remaining[identity] -= 1
        self._observed += 1
        reason = record.victory_type or "draw"
        target = self._target
        target["ending_age"][str(stats.ending_age)] = (
            target["ending_age"].get(str(stats.ending_age), 0) + 1
        )
        self._age_three_civilian += int(
            stats.ending_age == 3 and reason == "civilian"
        )
        target["science"]["sixth_symbol_wins"] += int(
            reason == "scientific" and stats.sixth_science_symbol
        )
        target["science"]["tokens_taken"] += stats.progress_tokens
        target["science"]["pairs_completed"] += stats.science_pairs
        target["military"]["max_absolute_track"] = max(
            target["military"]["max_absolute_track"],
            stats.max_absolute_track,
        )
        target["military"]["tokens_triggered"] += (
            stats.military_tokens_triggered
        )
        target["military"]["gold_pillaged"] += stats.military_gold_pillaged
        target["draft_wonder"]["wonders_built"] += stats.wonders_built
        target["draft_wonder"]["wonders_discarded"] += (
            stats.wonders_discarded
        )

    def finalize(self) -> None:
        if self._observed != self._expected:
            raise RuntimeError(
                "game-specific statistics were not collected during replay "
                f"derivation: observed {self._observed}/{self._expected} games"
            )
        if self._expected:
            self._target["draft_wonder"]["age_iii_completion_rate"] = (
                self._age_three_civilian / self._expected
            )


def _record_stats(
    records: list[GameRecord], performance: dict[str, Any]
):
    summary = summarize_records(records)
    lengths = [len(record.moves) for record in records]
    scheduler = performance.get("rust_scheduler", {})
    rows = int(scheduler.get("global_rows", 0))
    batches = int(scheduler.get("global_batches", 0))
    forced = int(scheduler.get("forced_rows", 0))
    seconds = float(performance.get("seconds", 0.0))
    opponent_mix = {opponent: 0 for opponent in OPPONENT_TYPES}
    terminal = {}
    winners = {"p0": 0, "p1": 0, "draw": 0}
    first_player_wins = 0
    margins = []
    by_opponent: dict[str, dict[str, Any]] = {}
    game_specific = {
        "victory_type_game_length": {},
        "ending_age": {},
        "ending_move_index": _distribution(lengths),
        "science": {
            "sixth_symbol_wins": 0,
            "tokens_taken": 0,
            "pairs_completed": 0,
        },
        "military": {
            "max_absolute_track": 0,
            "tokens_triggered": 0,
            "gold_pillaged": 0,
        },
        "draft_wonder": {
            "wonders_built": 0,
            "wonders_discarded": 0,
            "age_iii_completion_rate": 0.0,
        },
    }
    lengths_by_victory: dict[str, list[int]] = {}
    for record in records:
        opponent = resolve_opponent_type(record.agents)
        opponent_mix[opponent] += 1
        reason = record.victory_type or "draw"
        terminal[reason] = terminal.get(reason, 0) + 1
        lengths_by_victory.setdefault(reason, []).append(len(record.moves))
        winner_key = "draw" if record.winner is None else f"p{record.winner}"
        winners[winner_key] += 1
        if record.winner is not None and record.winner == record.first_player:
            first_player_wins += 1
        if record.scores is not None:
            margins.append(abs(record.scores[0] - record.scores[1]))
        bucket = by_opponent.setdefault(
            opponent, {"games": 0, "terminal_reason": {}, "wins_by_seat": {}}
        )
        bucket["games"] += 1
        bucket["terminal_reason"][reason] = (
            bucket["terminal_reason"].get(reason, 0) + 1
        )
        bucket["wins_by_seat"][winner_key] = (
            bucket["wins_by_seat"].get(winner_key, 0) + 1
        )

    game_specific["victory_type_game_length"] = {
        reason: {"count": len(values), **_distribution(values)}
        for reason, values in lengths_by_victory.items()
    }
    generation = GenerationStats(
        games=len(records),
        moves=summary["moves"],
        moves_per_game=_distribution(lengths),
        decisions_searched=summary["searched_moves"],
        mean_sims=summary["average_sims"],
        seconds=seconds,
        games_per_second=float(performance.get("games_per_second", 0.0)),
        nn_rows=rows,
        rows_per_second=rows / seconds if seconds else 0.0,
        forced_row_share=forced / rows if rows else 0.0,
        mean_batch_size=rows / batches if batches else 0.0,
        opponent_mix=opponent_mix,
    )
    outcomes = OutcomeStats(
        terminal_reason=terminal,
        winners=winners,
        first_player_win_rate=(
            first_player_wins / len(records) if records else None
        ),
        mean_margin=sum(margins) / len(margins) if margins else None,
        by_opponent_type=by_opponent,
    )
    return generation, outcomes, game_specific


class SevenWondersDuelLifecycleAdapter:
    name = "seven_wonders_duel"

    def __init__(self, loop: "PhaseDLoop"):
        self.loop = loop
        self._pending_game_stats: tuple[int, _GameSpecificCollector] | None = None

    def initialize_learner(self, *, seed: int):
        loop = self.loop
        path = loop.checkpoint_dir / "_bootstrap_init.pt"
        torch.manual_seed(seed)
        model = loop._new_model()
        checkpoint = make_checkpoint(
            model,
            {
                "model": "transformer",
                "d_model": loop.config.d_model,
                "layers": loop.config.layers,
                "heads": loop._built_heads(model),
                "precision": loop.config.precision,
                "iteration": -1,
            },
        )
        torch.save(checkpoint, path)
        return artifact_for(
            path, role="candidate", iteration=-1, training_state=UNTRAINED
        )

    def generate(self, request: GenerateRequest) -> GenerationResult:
        loop = self.loop
        loop.resource_monitor = ResourceMonitor()
        loop.phase_seconds = {}
        model = loop.load_model(request.generator_checkpoint)
        records = loop.generate_iteration(model, request.iteration)
        loop.sample_resources("post_generation")
        generation, outcomes, game_specific = _record_stats(
            records, loop.last_generation_stats
        )
        # The controller retains this result until training finishes. The
        # mutable game_specific payload is completed by the required replay
        # derivation before the controller validates and writes the row.
        self._pending_game_stats = (
            request.iteration,
            _GameSpecificCollector(records, game_specific),
        )
        model_stats = ModelStats(
            d_model=loop.config.d_model,
            layers=loop.config.layers,
            heads=loop._built_heads(model),
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            precision=loop.config.precision,
        )
        return GenerationResult(
            generated_games=len(records),
            metrics={
                "performance": dict(loop.last_generation_stats),
                "summary": summarize_records(records),
                "model": asdict(model_stats),
            },
            stats=generation,
            outcomes=outcomes,
            resources=loop.resource_stats(),
            game_specific=game_specific,
        )

    def assemble_replay(self, request: AssembleRequest) -> ReplayResult:
        records = self.loop.training_records(request.iteration)
        iterations = [
            record.iteration for record in records if record.iteration is not None
        ]
        newest = max(iterations, default=request.iteration)
        ages = [newest - value for value in iterations]
        selection = self.loop.window_selection(request.iteration)
        return ReplayResult(
            training_games=len(records),
            payload=records,
            metrics={"summary": summarize_records(records)},
            stats=ReplayStats(
                window_games=len(records),
                staleness=_distribution(ages),
                scheduled_window_games=(
                    selection.target_games if selection is not None else len(records)
                ),
                realized_window_games=(
                    selection.realised_games if selection is not None else len(records)
                ),
            ),
        )

    def train(self, request: TrainRequest) -> TrainingResult:
        loop = self.loop
        collector = None
        if (
            self._pending_game_stats is not None
            and self._pending_game_stats[0] == request.iteration
        ):
            collector = self._pending_game_stats[1]
        observer = collector.observe if collector is not None else None
        shortfall = loop.buffer_warmup_shortfall(request.replay.payload)
        if shortfall:
            # Warmup rows still require complete W3 statistics. Derive and cache
            # their trainable positions now, so the required replay does useful
            # buffer work instead of restoring the removed stats-only replay.
            replay_started = time.monotonic()
            examples = loop._cached_examples(
                request.replay.payload,
                on_record_derived=observer,
            )
            replay_seconds = time.monotonic() - replay_started
            loop.phase_seconds["replay_derivation"] = replay_seconds
            if collector is not None:
                collector.finalize()
                self._pending_game_stats = None
            # Nothing is installed on a skip, so hand back the incoming learner
            # as the (unused) candidate rather than a fresh snapshot.
            return TrainingResult(
                candidate=artifact_for(
                    Path(request.learner_checkpoint),
                    role="candidate",
                    iteration=request.iteration,
                    training_state=UNTRAINED,
                ),
                trained=False,
                skipped=True,
                skip_reason=shortfall,
                metrics={
                    "examples": len(examples),
                    "new_examples": sum(
                        example.iteration == request.iteration
                        for example in examples
                    ),
                    "replay_derivation_seconds": replay_seconds,
                },
                resources=loop.resource_stats(),
            )
        candidate = loop.train_candidate(
            request.replay.payload,
            request.iteration,
            source_checkpoint=request.learner_checkpoint,
            on_record_derived=observer,
        )
        if collector is not None:
            collector.finalize()
            self._pending_game_stats = None
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
            stats=self._training_stats(),
            resources=loop.resource_stats(),
        )

    def _training_stats(self) -> TrainingStats:
        performance = self.loop.last_training_stats
        history = performance.get("steps") or []
        last = history[-1] if history else {}
        train = {
            key: float(value)
            for key, value in (last.get("train") or {}).items()
            if isinstance(value, (int, float))
        }
        validation = {
            key: float(value)
            for key, value in (last.get("val") or {}).items()
            if isinstance(value, (int, float)) and "acc" not in key and "top1" not in key
        }
        accuracies = {
            key: float(value)
            for key, value in (last.get("val") or {}).items()
            if isinstance(value, (int, float)) and ("acc" in key or "top1" in key)
        }
        return TrainingStats(
            train_losses=train,
            validation_losses=validation,
            accuracies=accuracies,
            learning_rate=(
                float(last["lr"]) if last.get("lr") is not None else None
            ),
            gradient_norm=(
                float(last["grad_norm"])
                if last.get("grad_norm") is not None
                else None
            ),
            steps=int(self.loop.config.train_steps),
            seconds=float(performance.get("seconds", 0.0)),
            replay_derivation_seconds=float(
                performance.get("replay_derivation_seconds", 0.0)
            ),
            precision=self.loop.config.precision,
        )

    def _validate_candidate(self, candidate: Path, iteration: int) -> None:
        """Finite-metric + reload check before a candidate is certified trained."""

        history = self.loop.last_training_stats.get("steps") or []
        for window in history:
            for section in ("train", "val"):
                for key, value in (window.get(section) or {}).items():
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
        stats = GateStats(
            opponent=report.opponent,
            score_rate=report.score_rate,
            wilson_lcb=report.wilson_lcb,
            wilson_ucb=report.wilson_ucb,
            games=report.games,
            evaluated_games=report.evaluated_games,
            pairs=report.pairs,
            decision=report.decision,
            stop_reason=report.stop_reason,
            seconds=report.seconds,
            fixed_n=report.fixed_n,
        )
        metrics = asdict(report)
        metrics["_stats"] = asdict(stats)
        metrics["_resources"] = asdict(self.loop.resource_stats())
        return PromotionResult(
            decision=report.decision, metrics=metrics, stats=stats
        )

    def evaluate_anchors(self, request: AnchorRequest) -> AnchorResult:
        reports = self.loop.anchor_gates(request.checkpoint)
        passed = bool(reports) and all(
            report.decision == "accept" for report in reports
        )
        return AnchorResult(
            passed=passed,
            metrics={
                "gates": [asdict(report) for report in reports],
                "_stats": [
                    asdict(
                        GateStats(
                            opponent=report.opponent,
                            score_rate=report.score_rate,
                            wilson_lcb=report.wilson_lcb,
                            wilson_ucb=report.wilson_ucb,
                            games=report.games,
                            evaluated_games=report.evaluated_games,
                            pairs=report.pairs,
                            decision=report.decision,
                            stop_reason=report.stop_reason,
                            seconds=report.seconds,
                            fixed_n=True,
                        )
                    )
                    for report in reports
                ],
                "_resources": asdict(self.loop.resource_stats()),
            },
        )

    def archive_best(self, artifact) -> None:
        # Archive the OUTGOING trained best before it is overwritten.  The
        # controller never calls this for an untrained best.
        self.loop.hof.add(
            artifact.path, iteration=artifact.iteration, tag="promoted"
        )

    def on_learner_reset(self, best_checkpoint: Path) -> None:
        # A revert rewinds the weights to the protected best; the carried AdamW
        # moments describe the rejected trajectory, so drop them and let the
        # next iteration warm up cold from current_best.
        self.loop.clear_optimizer_state()

    def rollback_iteration(self, iteration: int) -> None:
        """Make an interrupted iteration safely restartable.

        Buffers and candidates are immutable once their row commits, so the
        pending journal proves these exact files are uncommitted. Adam moments
        may already describe the rejected candidate; dropping them is safer
        than pairing them with the restored learner weights.
        """

        loop = self.loop
        (loop.buffer_dir / f"iter_{iteration:04d}.jsonl").unlink(missing_ok=True)
        (loop.checkpoint_dir / f"candidate_{iteration:04d}.pt").unlink(
            missing_ok=True
        )
        loop.clear_optimizer_state()

    def autosave(self, iteration: int) -> None:
        # The controller owns cadence + failure policy; just do the atomic write.
        self.loop._save_replay_buffer()

    def contract(self) -> dict[str, Any]:
        return self.loop.adapter.contract()
