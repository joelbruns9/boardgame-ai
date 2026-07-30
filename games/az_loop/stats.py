"""Versioned, game-agnostic statistics for AlphaZero training iterations.

Schema v1 rows remain readable as legacy dictionaries.  Schema v2 adds this
typed block so every run records the same operational and learning signals
without requiring game-specific throwaway parsers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import os
import platform
import statistics
import time
from typing import Any, Mapping

import psutil


SCHEMA_VERSION = 2


def wilson_interval(
    points: float, observations: int, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval; fractional points support draws/pair splits."""

    n = int(observations)
    if n <= 0:
        return 0.0, 1.0
    p = max(0.0, min(1.0, float(points) / n))
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return (
        max(0.0, (centre - margin) / denominator),
        min(1.0, (centre + margin) / denominator),
    )


@dataclass(slots=True)
class GenerationStats:
    games: int = 0
    moves: int = 0
    moves_per_game: dict[str, float] = field(default_factory=dict)
    decisions_searched: int = 0
    mean_sims: float = 0.0
    seconds: float = 0.0
    games_per_second: float = 0.0
    nn_rows: int = 0
    rows_per_second: float = 0.0
    forced_row_share: float = 0.0
    mean_batch_size: float = 0.0
    opponent_mix: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class OutcomeStats:
    terminal_reason: dict[str, int] = field(default_factory=dict)
    winners: dict[str, int] = field(default_factory=dict)
    first_player_win_rate: float | None = None
    mean_margin: float | None = None
    by_opponent_type: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class ReplayStats:
    window_games: int = 0
    examples: int = 0
    new_examples: int = 0
    reuse_factor: float | None = None
    staleness: dict[str, float] = field(default_factory=dict)
    scheduled_window_games: int = 0
    realized_window_games: int = 0
    derivation_seconds: float = 0.0


@dataclass(slots=True)
class TrainingStats:
    train_losses: dict[str, float] = field(default_factory=dict)
    validation_losses: dict[str, float] = field(default_factory=dict)
    accuracies: dict[str, float] = field(default_factory=dict)
    learning_rate: float | None = None
    gradient_norm: float | None = None
    steps: int = 0
    seconds: float = 0.0
    replay_derivation_seconds: float = 0.0
    precision: str = ""


@dataclass(slots=True)
class GateStats:
    opponent: str = ""
    score_rate: float = 0.0
    wilson_lcb: float = 0.0
    wilson_ucb: float = 1.0
    games: int = 0
    evaluated_games: int = 0
    pairs: int = 0
    decision: str = ""
    stop_reason: str = ""
    seconds: float = 0.0
    fixed_n: bool = False


@dataclass(slots=True)
class ResourceStats:
    rss_post_generation_bytes: int | None = None
    rss_post_training_bytes: int | None = None
    rss_post_gate_bytes: int | None = None
    peak_rss_bytes: int = 0
    cache_estimated_bytes: int = 0
    cache_raw_array_bytes: int = 0
    cache_calibration_factor: float = 1.0
    gpu_utilization_percent: float | None = None
    vram_peak_allocated_bytes: int = 0
    vram_peak_physical_bytes: int = 0
    vram_total_bytes: int | None = None
    phase_seconds: dict[str, float] = field(default_factory=dict)
    memory_pressure: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ModelStats:
    d_model: int = 0
    layers: int = 0
    heads: int = 0
    parameters: int = 0
    precision: str = ""


@dataclass(slots=True)
class IterationStats:
    generation: GenerationStats = field(default_factory=GenerationStats)
    outcomes: OutcomeStats = field(default_factory=OutcomeStats)
    replay: ReplayStats = field(default_factory=ReplayStats)
    training: TrainingStats = field(default_factory=TrainingStats)
    gates: list[GateStats] = field(default_factory=list)
    resources: ResourceStats = field(default_factory=ResourceStats)
    model: ModelStats = field(default_factory=ModelStats)
    game_specific: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"iteration stats schema must be {SCHEMA_VERSION}, "
                f"got {self.schema_version}"
            )
        for label, value in (
            ("generation.games", self.generation.games),
            ("generation.moves", self.generation.moves),
            ("replay.window_games", self.replay.window_games),
            ("training.steps", self.training.steps),
            ("model.parameters", self.model.parameters),
        ):
            if value < 0:
                raise ValueError(f"{label} must be non-negative")
        if self.model.d_model and not self.model.heads:
            raise ValueError("model.heads is required when model.d_model is set")
        for gate in self.gates:
            if not 0.0 <= gate.score_rate <= 1.0:
                raise ValueError("gate score_rate must lie in [0, 1]")
            if not 0.0 <= gate.wilson_lcb <= gate.wilson_ucb <= 1.0:
                raise ValueError("gate Wilson bounds are invalid")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IterationStats":
        stats = cls(
            generation=GenerationStats(**dict(payload.get("generation", {}))),
            outcomes=OutcomeStats(**dict(payload.get("outcomes", {}))),
            replay=ReplayStats(**dict(payload.get("replay", {}))),
            training=TrainingStats(**dict(payload.get("training", {}))),
            gates=[GateStats(**dict(row)) for row in payload.get("gates", [])],
            resources=ResourceStats(**dict(payload.get("resources", {}))),
            model=ModelStats(**dict(payload.get("model", {}))),
            game_specific=dict(payload.get("game_specific", {})),
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
        )
        stats.validate()
        return stats


def stats_from_row(row: Mapping[str, Any]) -> IterationStats | None:
    """Read schema-v2 stats while tolerating historical schema-v1 rows."""

    payload = row.get("stats")
    if payload is None:
        return None
    return IterationStats.from_dict(payload)


class ResourceMonitor:
    """Low-overhead process/GPU sampler shared by game adapters."""

    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.started = time.monotonic()
        self.peak_rss_bytes = 0
        self.peak_vram_allocated_bytes = 0
        self.peak_vram_physical_bytes = 0
        self.vram_total_bytes: int | None = None
        self.samples: dict[str, dict[str, int | float | None]] = {}
        self.memory_pressure: list[dict[str, Any]] = []
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self.sample("start")

    @staticmethod
    def _torch_memory() -> tuple[int, int, int | None, float | None]:
        try:
            import torch

            if not torch.cuda.is_available():
                return 0, 0, None, None
            allocated = int(torch.cuda.max_memory_allocated())
            free, total = torch.cuda.mem_get_info()
            try:
                utilization = float(torch.cuda.utilization())
            except Exception:
                utilization = None
            return allocated, int(total - free), int(total), utilization
        except Exception:
            return 0, 0, None, None

    def sample(self, phase: str) -> dict[str, int | float | None]:
        rss = int(self.process.memory_info().rss)
        allocated, physical, total, utilization = self._torch_memory()
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.peak_vram_allocated_bytes = max(
            self.peak_vram_allocated_bytes, allocated
        )
        self.peak_vram_physical_bytes = max(
            self.peak_vram_physical_bytes, physical
        )
        if total is not None:
            self.vram_total_bytes = total
        sample = {
            "rss_bytes": rss,
            "vram_allocated_bytes": allocated,
            "vram_physical_bytes": physical,
            "gpu_utilization_percent": utilization,
            "seconds_since_start": time.monotonic() - self.started,
        }
        self.samples[phase] = sample
        return sample

    def pressure(self, phase: str, **details: Any) -> None:
        self.memory_pressure.append(
            {"phase": phase, "time": time.time(), **details}
        )

    def as_stats(
        self,
        *,
        cache_estimated_bytes: int = 0,
        cache_raw_array_bytes: int = 0,
        cache_calibration_factor: float = 1.0,
        phase_seconds: Mapping[str, float] | None = None,
    ) -> ResourceStats:
        def rss(phase: str) -> int | None:
            value = self.samples.get(phase, {}).get("rss_bytes")
            return int(value) if value is not None else None

        return ResourceStats(
            rss_post_generation_bytes=rss("post_generation"),
            rss_post_training_bytes=rss("post_training"),
            rss_post_gate_bytes=rss("post_gate"),
            peak_rss_bytes=self.peak_rss_bytes,
            cache_estimated_bytes=int(cache_estimated_bytes),
            cache_raw_array_bytes=int(cache_raw_array_bytes),
            cache_calibration_factor=float(cache_calibration_factor),
            gpu_utilization_percent=(
                statistics.fmean(
                    float(sample["gpu_utilization_percent"])
                    for sample in self.samples.values()
                    if sample.get("gpu_utilization_percent") is not None
                )
                if any(
                    sample.get("gpu_utilization_percent") is not None
                    for sample in self.samples.values()
                )
                else None
            ),
            vram_peak_allocated_bytes=self.peak_vram_allocated_bytes,
            vram_peak_physical_bytes=self.peak_vram_physical_bytes,
            vram_total_bytes=self.vram_total_bytes,
            phase_seconds=dict(phase_seconds or {}),
            memory_pressure=list(self.memory_pressure),
        )


def hardware_identity() -> dict[str, Any]:
    """Stable-enough identity for separating laptop and cloud measurements."""

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_bytes": int(psutil.virtual_memory().total),
    }
