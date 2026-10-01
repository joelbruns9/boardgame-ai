"""Paired candidate/incumbent strength gate and atomic S2 promotion.

Both arms receive the same game seeds, seat-count schedule, and actual frozen
incumbent opponents.  The primary endpoint is the paired change in
``own score - best opponent score``; normalized rank must not regress. Raw
score, wins, plans, and plan-ending games are diagnostics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from games.az_loop import atomic_copy
from games.welcome_to import macro_codec as mc
from games.welcome_to import s2_league, s2_train, self_play
from games.welcome_to.game import GameState


GATE_FORMAT = "welcome_to_s2_promotion"
GATE_VERSION = 1


@dataclass(frozen=True, slots=True)
class GateConfig:
    games: int = 300
    simulations: int = 200
    inflight: int = 256
    max_batch: int = 256
    scheduler_workers: int = 8
    seed: int = 80_000
    confidence_z: float = 1.96
    secondary_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.games < 2:
            raise ValueError("promotion gate needs at least two paired games")
        if min(
            self.simulations,
            self.inflight,
            self.max_batch,
            self.scheduler_workers,
        ) <= 0:
            raise ValueError("gate search and scheduler sizes must be positive")
        if not math.isfinite(self.confidence_z) or self.confidence_z <= 0.0:
            raise ValueError("confidence_z must be finite and positive")
        if not math.isfinite(self.secondary_tolerance) or self.secondary_tolerance < 0:
            raise ValueError("secondary_tolerance must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class Estimate:
    mean: float
    stderr: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class ArmSummary:
    margin_vs_best: float
    normalized_rank: float
    raw_score: float
    win_rate: float
    plans_completed: float
    plan_end_fraction: float


@dataclass(frozen=True, slots=True)
class PromotionReport:
    format: str
    version: int
    decision: str
    games: int
    primary_margin_delta: Estimate
    secondary_rank_delta: Estimate
    primary_significant: bool
    secondary_not_regressed: bool
    candidate: ArmSummary
    incumbent: ArmSummary
    diagnostics: Mapping[str, Estimate]
    config: Mapping[str, Any]
    candidate_generation_seconds: Optional[float] = None
    incumbent_generation_seconds: Optional[float] = None

    @property
    def passed(self) -> bool:
        return self.decision == "promote"


@dataclass(frozen=True, slots=True)
class _GameMetrics:
    margin: float
    rank: float
    score: float
    win: float
    plans: float
    plan_end: float


def _estimate(values: Sequence[float], z: float) -> Estimate:
    if not values:
        raise ValueError("paired estimate needs observations")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    stderr = math.sqrt(variance / len(values))
    return Estimate(
        mean=mean,
        stderr=stderr,
        lower=mean - z * stderr,
        upper=mean + z * stderr,
    )


def secondary_not_regressed(estimate: Estimate, tolerance: float) -> bool:
    """Pass unless the interval establishes a material rank regression."""
    return estimate.upper >= -tolerance


def normalized_rank(scores: Sequence[int], seat: int = 0) -> float:
    """Linear rank utility with average ranks for ties, in [0, 1]."""
    if len(scores) < 2 or not 0 <= seat < len(scores):
        raise ValueError("normalized rank needs a valid seat in a multiplayer game")
    own = scores[seat]
    better = sum(score > own for score in scores)
    tied = sum(score == own for score in scores)
    rank = better + (tied - 1) / 2.0
    return (len(scores) - 1 - rank) / (len(scores) - 1)


def _game_metrics(trajectory: self_play.SelfPlayTrajectory) -> _GameMetrics:
    state = GameState.new(
        seed=trajectory.seed,
        config=trajectory.config,
        rng_kind=trajectory.rng,
    )
    for action in trajectory.actions:
        mc.apply_macro(state, action)
    if not state.is_terminal or tuple(state.scores()) != trajectory.scores:
        raise ValueError(f"gate trajectory {trajectory.seed} failed deterministic replay")
    own = state.scores()[trajectory.learner]
    opponents = [
        score for seat, score in enumerate(state.scores()) if seat != trajectory.learner
    ]
    plans = sum(
        trajectory.learner in state.plan_turns[slot] for slot in range(3)
    )
    reason = state.end_of_game_reason() or ""
    return _GameMetrics(
        margin=float(own - max(opponents)),
        rank=normalized_rank(state.scores(), trajectory.learner),
        score=float(own),
        win=float(trajectory.learner in state.winners()),
        plans=float(plans),
        plan_end=float("completed all three plans" in reason),
    )


def compare_trajectories(
    candidate_games: Sequence[self_play.SelfPlayTrajectory],
    incumbent_games: Sequence[self_play.SelfPlayTrajectory],
    config: GateConfig,
    *,
    candidate_generation_seconds: Optional[float] = None,
    incumbent_generation_seconds: Optional[float] = None,
) -> PromotionReport:
    """Calculate the fixed-N paired promotion decision from completed games."""
    candidate_by_seed = {game.seed: game for game in candidate_games}
    incumbent_by_seed = {game.seed: game for game in incumbent_games}
    if len(candidate_by_seed) != len(candidate_games) or len(incumbent_by_seed) != len(
        incumbent_games
    ):
        raise ValueError("promotion arms contain duplicate seeds")
    if candidate_by_seed.keys() != incumbent_by_seed.keys():
        raise ValueError("promotion arms do not contain identical game seeds")
    if len(candidate_by_seed) != config.games:
        raise ValueError(
            f"promotion expected {config.games} pairs, got {len(candidate_by_seed)}"
        )

    candidate_rows: list[_GameMetrics] = []
    incumbent_rows: list[_GameMetrics] = []
    for seed in sorted(candidate_by_seed):
        candidate_game = candidate_by_seed[seed]
        incumbent_game = incumbent_by_seed[seed]
        if candidate_game.players != incumbent_game.players:
            raise ValueError(f"paired seed {seed} used different seat counts")
        candidate_rows.append(_game_metrics(candidate_game))
        incumbent_rows.append(_game_metrics(incumbent_game))

    def arm(rows: Sequence[_GameMetrics]) -> ArmSummary:
        count = len(rows)
        return ArmSummary(
            margin_vs_best=sum(row.margin for row in rows) / count,
            normalized_rank=sum(row.rank for row in rows) / count,
            raw_score=sum(row.score for row in rows) / count,
            win_rate=sum(row.win for row in rows) / count,
            plans_completed=sum(row.plans for row in rows) / count,
            plan_end_fraction=sum(row.plan_end for row in rows) / count,
        )

    def deltas(name: str) -> list[float]:
        return [
            getattr(candidate, name) - getattr(incumbent, name)
            for candidate, incumbent in zip(candidate_rows, incumbent_rows)
        ]

    primary = _estimate(deltas("margin"), config.confidence_z)
    secondary = _estimate(deltas("rank"), config.confidence_z)
    primary_significant = primary.lower > 0.0
    # Reject on the secondary only when the data show a significant regression.
    # A bare mean-at-zero check rejects a truly neutral candidate half the time.
    secondary_passed = secondary_not_regressed(
        secondary, config.secondary_tolerance
    )
    diagnostics = {
        "raw_score_delta": _estimate(deltas("score"), config.confidence_z),
        "win_rate_delta": _estimate(deltas("win"), config.confidence_z),
        "plans_completed_delta": _estimate(deltas("plans"), config.confidence_z),
        "plan_end_fraction_delta": _estimate(deltas("plan_end"), config.confidence_z),
    }
    return PromotionReport(
        format=GATE_FORMAT,
        version=GATE_VERSION,
        decision=(
            "promote" if primary_significant and secondary_passed else "reject"
        ),
        games=config.games,
        primary_margin_delta=primary,
        secondary_rank_delta=secondary,
        primary_significant=primary_significant,
        secondary_not_regressed=secondary_passed,
        candidate=arm(candidate_rows),
        incumbent=arm(incumbent_rows),
        diagnostics=diagnostics,
        config=asdict(config),
        candidate_generation_seconds=candidate_generation_seconds,
        incumbent_generation_seconds=incumbent_generation_seconds,
    )


def gate_search_config(simulations: int):
    """Deterministic search used by both promotion arms.

    Generation noise is valuable for producing training diversity. It only adds
    variance to a fixed-N strength comparison and ceases to be common randomness
    as soon as the arms choose different moves.
    """
    return replace(
        self_play.default_search_config(simulations),
        dirichlet_alpha=None,
        dirichlet_concentration=None,
        dirichlet_weight=0.0,
        noise_fresh_fraction=0.0,
    )


def run_gate(
    candidate: torch.nn.Module,
    incumbent: torch.nn.Module,
    *,
    config: Optional[GateConfig] = None,
    device: torch.device | str = "cuda",
    cuda_events: bool = False,
) -> PromotionReport:
    """Generate both paired arms and return their fixed-N strength decision."""
    config = config or GateConfig()
    generation = self_play.SelfPlayConfig(
        games=config.games,
        inflight=config.inflight,
        max_batch=config.max_batch,
        scheduler_workers=config.scheduler_workers,
        seed=config.seed,
        opening_temperature_turns=0,
        opening_temperature=0.0,
        late_temperature=0.0,
    )
    search = gate_search_config(config.simulations)
    opponent = (self_play.Opponent("current_best", incumbent),)

    started = time.perf_counter()
    candidate_games, _ = self_play.generate(
        candidate,
        config=generation,
        search_config=search,
        opponents=opponent,
        search_policy_net=incumbent,
        device=device,
        cuda_events=cuda_events,
    )
    candidate_seconds = time.perf_counter() - started
    started = time.perf_counter()
    incumbent_games, _ = self_play.generate(
        incumbent,
        config=generation,
        search_config=search,
        opponents=opponent,
        search_policy_net=incumbent,
        device=device,
        cuda_events=cuda_events,
    )
    incumbent_seconds = time.perf_counter() - started
    return compare_trajectories(
        candidate_games,
        incumbent_games,
        config,
        candidate_generation_seconds=candidate_seconds,
        incumbent_generation_seconds=incumbent_seconds,
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=path.name + ".",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def validate_candidate(path: str | Path, device: torch.device | str = "cpu") -> None:
    """Refuse a corrupt, incompatible, or non-finite candidate before play."""
    model, payload = s2_train.load_training_checkpoint(path, device)
    for window in payload.get("metrics", {}).get("history", []):
        for name, value in window.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(
                    f"candidate checkpoint has non-finite training metric {name}={value}"
                )
    for name, tensor in model.state_dict().items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"candidate checkpoint has non-finite weights in {name}")


def _read_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("format") != GATE_FORMAT
        or int(payload.get("version", -1)) != GATE_VERSION
    ):
        raise ValueError(f"unsupported promotion record {path}")
    return payload


def _finish_promotion_intent(path: Path, payload: Mapping[str, Any]) -> bool:
    """Idempotently finish an intent whose old/new checkpoint hashes are pinned."""
    if payload.get("status") != "installing" or not payload.get("promoted"):
        raise ValueError("record is not a pending promotion intent")
    candidate_info = dict(payload["candidate"])
    incumbent_info = dict(payload["incumbent"])
    candidate = Path(candidate_info["path"])
    current_best = Path(incumbent_info["path"])
    archive = Path(str(payload["archive"]))
    candidate_sha = str(candidate_info["sha256"])
    incumbent_sha = str(incumbent_info["sha256_before"])
    if not candidate.is_file() or _sha256(candidate) != candidate_sha:
        raise RuntimeError("pending promotion candidate is missing or changed")
    if not current_best.is_file():
        raise RuntimeError("pending promotion current-best path is missing")
    current_sha = _sha256(current_best)
    if current_sha not in (incumbent_sha, candidate_sha):
        raise RuntimeError(
            "pending promotion current-best matches neither pinned checkpoint"
        )

    if not archive.exists():
        if current_sha != incumbent_sha:
            raise RuntimeError(
                "pending promotion installed the candidate without preserving the incumbent"
            )
        atomic_copy(current_best, archive)
    if _sha256(archive) != incumbent_sha:
        raise RuntimeError(f"promotion archive path contains different bytes: {archive}")

    if current_sha == incumbent_sha:
        atomic_copy(candidate, current_best)
    if _sha256(current_best) != candidate_sha:
        raise RuntimeError("atomic promotion did not install the candidate bytes")

    league_entry = None
    league_manifest = payload.get("league_manifest")
    if league_manifest is not None:
        iteration = payload.get("iteration")
        if iteration is None:
            raise ValueError("league registration needs the promotion iteration")
        league_entry = s2_league.S2League(str(league_manifest)).register(
            archive,
            archived_at_iteration=int(iteration),
        )
    complete = dict(payload)
    complete["status"] = "complete"
    complete["league_entry"] = (
        asdict(league_entry) if league_entry is not None else None
    )
    _write_json(path, complete)
    return True


def recover_pending_promotion(
    record_path: str | Path,
    *,
    candidate_path: str | Path | None = None,
    current_best_path: str | Path | None = None,
) -> bool:
    """Finish a crash-interrupted promotion without rerunning its strength gate."""
    path = Path(record_path)
    payload = _read_record(path)
    if payload.get("status", "complete") == "complete":
        return bool(payload.get("promoted"))
    if candidate_path is not None and Path(payload["candidate"]["path"]) != Path(
        candidate_path
    ).resolve():
        raise RuntimeError("pending promotion belongs to a different candidate")
    if current_best_path is not None and Path(payload["incumbent"]["path"]) != Path(
        current_best_path
    ).resolve():
        raise RuntimeError("pending promotion belongs to a different current-best path")
    return _finish_promotion_intent(path, payload)


def install_promotion(
    report: PromotionReport,
    candidate_path: str | Path,
    current_best_path: str | Path,
    *,
    archive_dir: str | Path,
    record_path: str | Path,
    league_manifest: str | Path | None = None,
    iteration: int | None = None,
) -> bool:
    """Durably record intent, then install a passing candidate idempotently.

    The caller owns ``record_path`` exclusively. A crash leaves an ``installing``
    record that :func:`recover_pending_promotion` completes before any new gate
    is run, so an installed candidate cannot be misreported as a later rejection.
    """
    candidate = Path(candidate_path).resolve()
    current_best = Path(current_best_path).resolve()
    record = Path(record_path)
    if league_manifest is not None and iteration is None:
        raise ValueError("league registration needs the promotion iteration")
    if not candidate.is_file() or not current_best.is_file():
        raise FileNotFoundError("candidate and current-best checkpoints must exist")
    if record.exists():
        existing = _read_record(record)
        if existing.get("status", "complete") == "installing":
            return recover_pending_promotion(
                record,
                candidate_path=candidate,
                current_best_path=current_best,
            )
        raise FileExistsError(f"promotion record already complete: {record}")
    candidate_sha = _sha256(candidate)
    incumbent_sha = _sha256(current_best)
    if candidate_sha == incumbent_sha and report.passed:
        raise ValueError("a passing promotion must change the current-best checkpoint")
    archive = Path(archive_dir) / f"best_{incumbent_sha[:16]}.pt"
    promoted = report.passed
    payload = {
        "format": GATE_FORMAT,
        "version": GATE_VERSION,
        "status": "installing" if promoted else "complete",
        "promoted": promoted,
        "candidate": {"path": str(candidate), "sha256": candidate_sha},
        "incumbent": {
            "path": str(current_best),
            "sha256_before": incumbent_sha,
        },
        "archive": str(archive.resolve()) if promoted else None,
        "league_manifest": (
            str(Path(league_manifest).resolve()) if league_manifest is not None else None
        ),
        "iteration": iteration,
        "league_entry": None,
        "report": asdict(report),
    }
    # This is the commit point. Every later filesystem step is idempotent and
    # recoverable from the pinned paths and hashes in this intent.
    _write_json(record, payload)
    if not promoted:
        return False
    return _finish_promotion_intent(record, payload)


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--current-best", required=True)
    parser.add_argument("--games", type=int, default=GateConfig().games)
    parser.add_argument("--simulations", type=int, default=GateConfig().simulations)
    parser.add_argument("--inflight", type=int, default=GateConfig().inflight)
    parser.add_argument("--max-batch", type=int, default=GateConfig().max_batch)
    parser.add_argument(
        "--scheduler-workers", type=int, default=GateConfig().scheduler_workers
    )
    parser.add_argument("--seed", type=int, default=GateConfig().seed)
    parser.add_argument("--confidence-z", type=float, default=GateConfig().confidence_z)
    parser.add_argument(
        "--secondary-tolerance", type=float, default=GateConfig().secondary_tolerance
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cuda-events", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--archive-dir")
    parser.add_argument("--league-manifest")
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    if args.promote and args.iteration is None:
        parser.error("--promote requires --iteration for durable league registration")

    report_path = Path(args.report or (str(args.candidate) + ".gate.json"))
    if args.promote and report_path.exists():
        existing = _read_record(report_path)
        if existing.get("status", "complete") == "installing":
            promoted = recover_pending_promotion(
                report_path,
                candidate_path=args.candidate,
                current_best_path=args.current_best,
            )
            completed = _read_record(report_path)
            print(json.dumps(completed["report"], indent=2, sort_keys=True, allow_nan=False))
            return 0 if promoted else 2
        raise RuntimeError(f"promotion record already complete: {report_path}")

    validate_candidate(args.candidate, args.device)
    validate_candidate(args.current_best, args.device)
    candidate, _ = s2_train.load_training_checkpoint(args.candidate, args.device)
    incumbent, _ = s2_train.load_training_checkpoint(args.current_best, args.device)
    config = GateConfig(
        games=args.games,
        simulations=args.simulations,
        inflight=args.inflight,
        max_batch=args.max_batch,
        scheduler_workers=args.scheduler_workers,
        seed=args.seed,
        confidence_z=args.confidence_z,
        secondary_tolerance=args.secondary_tolerance,
    )
    report = run_gate(
        candidate,
        incumbent,
        config=config,
        device=args.device,
        cuda_events=args.cuda_events,
    )
    if args.promote:
        install_promotion(
            report,
            args.candidate,
            args.current_best,
            archive_dir=args.archive_dir or Path(args.current_best).parent / "promoted",
            record_path=report_path,
            league_manifest=(
                args.league_manifest
                or Path(args.current_best).parent / "league.json"
            ),
            iteration=args.iteration,
        )
    else:
        _write_json(report_path, asdict(report))
    print(json.dumps(asdict(report), indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.passed else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
