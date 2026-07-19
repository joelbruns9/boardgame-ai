"""Small explicit schedules and replay-window bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeVar


@dataclass(frozen=True, slots=True)
class LinearSchedule:
    start: float
    end: float
    duration: int

    def value(self, iteration: int) -> float:
        if iteration < 0:
            raise ValueError("iteration must be non-negative")
        if self.duration <= 0:
            return self.end
        fraction = min(1.0, iteration / self.duration)
        return self.start + fraction * (self.end - self.start)


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ReplayWindow:
    iterations: int = 20

    def select(self, values: Sequence[T], iteration_of) -> list[T]:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        present = [iteration_of(value) for value in values]
        numbered = [value for value in present if value is not None]
        if not numbered:
            return list(values)
        newest = max(numbered)
        oldest = newest - self.iterations + 1
        return [
            value
            for value in values
            if iteration_of(value) is not None and iteration_of(value) >= oldest
        ]

    def paths(self, buffer_dir: str | Path, current_iteration: int) -> list[Path]:
        root = Path(buffer_dir)
        oldest = max(0, current_iteration - self.iterations + 1)
        return [
            root / f"iter_{iteration:04d}.jsonl"
            for iteration in range(oldest, current_iteration + 1)
            if (root / f"iter_{iteration:04d}.jsonl").exists()
        ]
