"""Small explicit schedules and replay-window bookkeeping.

Two families live here.  ``LinearSchedule`` and ``ReplayWindow`` are keyed on
**iteration index**; ``GameSchedule`` and ``GrowingReplayWindow`` are keyed on
**cumulative games**.  The games-keyed pair is the one new work should use: an
iteration is not a unit of anything, and a schedule measured in iterations
silently rescales when ``games_per_iteration`` changes between runs.  The
iteration-keyed pair is retained because existing runs and their manifests are
expressed in it, and because per-move schedules (temperature) legitimately
have no games dimension.

See ``games_ledger.GamesLedger`` for the clock these read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TypeVar


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


@dataclass(frozen=True, slots=True)
class GameSchedule:
    """A linear anneal from ``start`` to ``end`` over ``duration_games``.

    The games-keyed twin of ``LinearSchedule``.  Identical arithmetic; the
    difference is entirely in what the caller passes, and that difference is the
    point: ``GameSchedule(0.15, 0.0, 10_000).value(total_games)`` means the same
    thing at 400 and at 800 games per iteration, whereas
    ``LinearSchedule(0.15, 0.0, 25).value(iteration)`` does not.

    ``duration_games <= 0`` pins the schedule at ``end`` -- the "already
    finished" reading, matching ``LinearSchedule``, so a zero duration disables
    a curriculum rather than dividing by zero.
    """

    start: float
    end: float
    duration_games: int

    def value(self, total_games: int) -> float:
        if total_games < 0:
            raise ValueError("total_games must be non-negative")
        if self.duration_games <= 0:
            return self.end
        fraction = min(1.0, total_games / self.duration_games)
        return self.start + fraction * (self.end - self.start)

    def finished(self, total_games: int) -> bool:
        return self.duration_games <= 0 or total_games >= self.duration_games


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


@dataclass(frozen=True, slots=True)
class WindowSelection:
    """What a growing window asked for, and what it actually got.

    Both numbers go in the stats row.  They differ whenever the run has not yet
    generated enough games to fill the target, and whenever whole-iteration
    granularity overshoots -- neither is an error, but a run whose realised
    window silently sits far below its target is a run whose window schedule is
    not doing what the config says, and that is only visible if both are logged.
    """

    target_games: int
    realised_games: int
    iterations: tuple[int, ...]

    @property
    def oldest_iteration(self) -> int | None:
        return min(self.iterations) if self.iterations else None

    @property
    def newest_iteration(self) -> int | None:
        return max(self.iterations) if self.iterations else None


@dataclass(frozen=True, slots=True)
class GrowingReplayWindow:
    """Replay window sized as a sublinear power law in cumulative games.

    ``window = clamp(coefficient * total_games ** exponent, floor, cap)``

    Small early and growing late.  Early on, a small window keeps training
    close to on-policy so the net adapts fast; later, a larger one reduces
    target variance and supplies the opponent diversity that mattered in
    Kingdomino.  ``exponent`` in 0.5-0.8 keeps growth sublinear, so the window
    never becomes "all games ever played" -- which would make the newest
    iteration an ever-shrinking fraction of each batch and stall adaptation.

    Motivating evidence, and its limits: run 03's window grew 1,400 -> 8,000
    games over iterations 0-20 and then froze at its iteration cap, and value
    accuracy peaked at iteration 35 and decayed after.  Suggestive of staleness,
    not proof of it -- the two are confounded with everything else that changed.

    ``cap_games`` is **not** a free parameter: it has to come from the memory
    budget, because the window is what the example cache holds.  At the measured
    17.8 KB per example and ~20 examples per game, a 20,000-game cap is roughly
    7 GB of host RSS.  Deriving it independently of that budget is how a run
    dies at iteration 70.
    """

    coefficient: float
    exponent: float
    cap_games: int
    floor_games: int = 1

    def __post_init__(self) -> None:
        if self.coefficient <= 0:
            raise ValueError("coefficient must be positive")
        if not 0.0 < self.exponent <= 1.0:
            raise ValueError("exponent must lie in (0, 1]")
        if self.cap_games <= 0:
            raise ValueError("cap_games must be positive")
        if self.floor_games <= 0:
            raise ValueError("floor_games must be positive")
        if self.floor_games > self.cap_games:
            raise ValueError("floor_games must not exceed cap_games")

    def games(self, total_games: int) -> int:
        """Target window size in games at a given point on the clock."""

        if total_games < 0:
            raise ValueError("total_games must be non-negative")
        raw = self.coefficient * (float(total_games) ** self.exponent)
        return int(min(float(self.cap_games), max(float(self.floor_games), raw)))

    def select(
        self,
        total_games: int,
        current_iteration: int,
        games_for: Callable[[int], int],
        available: Sequence[int],
    ) -> WindowSelection:
        """Choose whole iterations, newest first, until the target is met.

        Whole iterations rather than individual games, for two reasons: buffer
        files are the unit that is immutable and cacheable, and a partial
        iteration would make the selection depend on record order within a file,
        which no other part of the system promises to hold stable.

        The newest available iteration is always included even if it alone
        overshoots the target -- training on nothing is never the better
        alternative, and at the very start of a run the target is smaller than a
        single iteration by construction.
        """

        target = self.games(total_games)
        candidates = sorted(
            (iteration for iteration in available if iteration <= current_iteration),
            reverse=True,
        )
        chosen: list[int] = []
        realised = 0
        for iteration in candidates:
            if chosen and realised >= target:
                break
            chosen.append(iteration)
            realised += games_for(iteration)
        return WindowSelection(
            target_games=target,
            realised_games=realised,
            iterations=tuple(sorted(chosen)),
        )
