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

from .buffer import replay
from .engine import _science_symbols
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


def _record_stats(records, performance):
    summary = summarize_records(records)
    lengths = [len(record.moves) for record in records]
    scheduler = performance.get("rust_scheduler", {})
    rows = int(scheduler.get("global_rows", 0))
    batches = int(scheduler.get("global_batches", 0))
    forced = int(scheduler.get("forced_rows", 0))
    seconds = float(performance.get("seconds", 0.0))
    opponent_mix = {"current_best": 0, "hof": 0, "bot": 0}
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
    age_three = 0
    for record in records:
        kind = record.agents.get("kind", "self_play")
        opponent = "hof" if kind == "league" else (
            "bot" if "curriculum" in kind else "current_best"
        )
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

        maximum_track = 0

        def observe(game, _move):
            nonlocal maximum_track
            maximum_track = max(maximum_track, abs(game.conflict_position))

        game = replay(record, on_state=observe)
        maximum_track = max(maximum_track, abs(game.conflict_position))
        game_specific["ending_age"][str(game.age)] = (
            game_specific["ending_age"].get(str(game.age), 0) + 1
        )
        age_three += int(game.age == 3 and reason == "civilian")
        symbols = [len(_science_symbols(game, seat)) for seat in (0, 1)]
        game_specific["science"]["sixth_symbol_wins"] += int(
            reason == "scientific" and max(symbols) >= 6
        )
        game_specific["science"]["tokens_taken"] += sum(
            len(city.progress_tokens) for city in game.cities
        )
        game_specific["science"]["pairs_completed"] += sum(
            len(city.claimed_science_pairs) for city in game.cities
        )
        game_specific["military"]["max_absolute_track"] = max(
            game_specific["military"]["max_absolute_track"], maximum_track
        )
        game_specific["military"]["tokens_triggered"] += 4 - len(
            game.military_tokens_remaining
        )
        game_specific["military"]["gold_pillaged"] += 14 - sum(
            game.military_tokens_remaining.values()
        )
        game_specific["draft_wonder"]["wonders_built"] += sum(
            len(city.built_wonders) for city in game.cities
        )
        game_specific["draft_wonder"]["wonders_discarded"] += len(
            game.retired_wonders
        )
    game_specific["victory_type_game_length"] = {
        reason: {"count": len(values), **_distribution(values)}
        for reason, values in lengths_by_victory.items()
    }
    if records:
        game_specific["draft_wonder"]["age_iii_completion_rate"] = (
            age_three / len(records)
        )
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
        shortfall = loop.buffer_warmup_shortfall(request.replay.payload)
        if shortfall:
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
            )
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
