"""W7: detect a run that has stopped learning, and escalate a response.

The detection half rests on one property of a **games-indexed** anchor: the
opponent is whatever ``current_best`` was N games ago, so when learning stops the
anchor advances until it catches up and the score converges to exactly 0.500 --
a net against itself.  That makes the null self-calibrating.  A promotion-lagged
anchor cannot do this: when promotions stop, both pointers freeze and the score
is a constant that no threshold ever crosses.

Everything here is pure and game-agnostic.  Measurements come from whatever the
game uses to play a fixed-N match; the state survives a resume as a dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence


NORMAL = "normal"
STAGNANT = "stagnant"
INTERVENING = "intervening"
RE_MEASURE = "re_measure"
EXHAUSTED = "exhausted"


@dataclass(frozen=True, slots=True)
class AnchorMeasurement:
    """One fixed-N score against the games-lagged self."""

    games: int
    """Games clock when the measurement was taken."""

    score_rate: float
    lower: float
    upper: float
    anchor_games: int = 0
    """Games clock of the anchor checkpoint; the lag is ``games - anchor_games``."""

    iteration: int = -1

    def as_dict(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "score_rate": self.score_rate,
            "lower": self.lower,
            "upper": self.upper,
            "anchor_games": self.anchor_games,
            "iteration": self.iteration,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnchorMeasurement":
        return cls(
            games=int(payload["games"]),
            score_rate=float(payload["score_rate"]),
            lower=float(payload["lower"]),
            upper=float(payload["upper"]),
            anchor_games=int(payload.get("anchor_games", 0)),
            iteration=int(payload.get("iteration", -1)),
        )


def _slope_per_games(measurements: Sequence[AnchorMeasurement]) -> float | None:
    """OLS slope of score against the games clock, per 10k games.

    Per games rather than per measurement: the cadence can change mid-run (an
    intervention lengthens the window), and a slope per measurement would then
    mean two different things in one series.
    """

    if len(measurements) < 2:
        return None
    xs = [measurement.games / 10_000 for measurement in measurements]
    ys = [measurement.score_rate for measurement in measurements]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


@dataclass(frozen=True, slots=True)
class StagnationVerdict:
    stagnant: bool
    reasons: tuple[str, ...]
    slope_per_10k_games: float | None
    measurements_considered: int


@dataclass(frozen=True, slots=True)
class StagnationDetector:
    """Two independent triggers, both requiring several measurements.

    Never a single point: one anchor score straddling 0.500 is what an evenly
    matched pair of nets looks like on any given afternoon.
    """

    null: float = 0.50
    min_measurements: int = 3
    slope_epsilon: float = 0.005
    """Score gain per 10k games below which the run is not improving.

    A run that is learning moves the anchor score *up*, because the anchor lags.
    Zero would be too strict a boundary for a noisy estimate, so this is the
    smallest slope worth calling progress.
    """

    def verdict(
        self, measurements: Sequence[AnchorMeasurement]
    ) -> StagnationVerdict:
        recent = list(measurements)[-self.min_measurements :]
        if len(recent) < self.min_measurements:
            return StagnationVerdict(
                stagnant=False,
                reasons=("insufficient measurements",),
                slope_per_10k_games=_slope_per_games(recent),
                measurements_considered=len(recent),
            )
        reasons: list[str] = []
        # Interval trigger: not once in the recent window could the current net
        # be shown to beat its own past self.
        if all(measurement.lower <= self.null for measurement in recent):
            reasons.append(
                f"the last {len(recent)} anchor intervals all include "
                f"{self.null:.2f}: no measurement distinguishes the current net "
                "from its own past self"
            )
        slope = _slope_per_games(recent)
        if slope is not None and slope <= self.slope_epsilon:
            reasons.append(
                f"anchor slope {slope:+.4f}/10k games is at or below "
                f"{self.slope_epsilon:+.4f}"
            )
        return StagnationVerdict(
            stagnant=bool(reasons),
            reasons=tuple(reasons),
            slope_per_10k_games=slope,
            measurements_considered=len(recent),
        )


@dataclass(frozen=True, slots=True)
class Intervention:
    """One rung: a named, bounded change to the schedules.

    Multipliers rather than absolute values, so a rung means the same thing
    whatever the run was configured with.
    """

    name: str
    sims_multiplier: float = 1.0
    full_search_fraction_multiplier: float = 1.0
    window_multiplier: float = 1.0
    hof_fraction: float | None = None
    learning_rate_multiplier: float = 1.0
    rationale: str = ""


DEFAULT_LADDER: tuple[Intervention, ...] = (
    Intervention(
        name="raise_search_budget",
        sims_multiplier=1.5,
        full_search_fraction_multiplier=1.25,
        rationale=(
            "stagnation usually means the search-improved policy is no longer "
            "better than the raw policy at this budget, so buy search first"
        ),
    ),
    Intervention(
        name="jump_replay_window",
        window_multiplier=1.5,
        rationale="more diverse targets before changing what generates them",
    ),
    Intervention(
        name="raise_hof_fraction",
        hof_fraction=0.30,
        rationale="opponent diversity; what mattered most in Kingdomino",
    ),
    Intervention(
        name="lr_warm_restart",
        learning_rate_multiplier=3.0,
        rationale="last resort: the optimiser, not the data, is stuck",
    ),
)
"""The plan's four rungs, in its order. Model growth is NOT on this ladder."""


@dataclass(frozen=True, slots=True)
class LadderState:
    state: str = NORMAL
    rung: int = -1
    """Index of the active intervention; -1 when none is applied."""

    entered_at_games: int = 0
    """Games clock when the current state began; the measurement window runs
    from here so each rung's effect is attributable to that rung alone."""

    history: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "rung": self.rung,
            "entered_at_games": self.entered_at_games,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "LadderState":
        if not payload:
            return cls()
        return cls(
            state=str(payload.get("state", NORMAL)),
            rung=int(payload.get("rung", -1)),
            entered_at_games=int(payload.get("entered_at_games", 0)),
            history=tuple(payload.get("history", ())),
        )


@dataclass(frozen=True, slots=True)
class InterventionLadder:
    """Escalate one rung at a time, with a measurement window between rungs.

    Disabled by default (``enabled=False``): present in the code that launches
    so it can be switched on at launch or at a deliberate restart, because a
    running process will not pick up a mid-run change and restarting to get one
    is exactly what W6.5 refuses.
    """

    rungs: tuple[Intervention, ...] = DEFAULT_LADDER
    measurement_window_games: int = 20_000
    enabled: bool = False

    def active(self, state: LadderState) -> Intervention | None:
        if not self.enabled or state.rung < 0 or state.rung >= len(self.rungs):
            return None
        return self.rungs[state.rung]

    def advance(
        self, state: LadderState, verdict: StagnationVerdict, games: int
    ) -> LadderState:
        """Next ladder state after one anchor measurement."""

        if not self.enabled:
            # Detection still runs and still reports; only the response is off.
            return replace(state, state=STAGNANT if verdict.stagnant else NORMAL)

        if state.state in (NORMAL, STAGNANT):
            if not verdict.stagnant:
                return replace(state, state=NORMAL)
            return self._escalate(state, games, "stagnation detected")

        # A rung is applied; hold it until its window has elapsed so the next
        # measurement reflects this rung rather than the previous one.
        if games - state.entered_at_games < self.measurement_window_games:
            return state
        if not verdict.stagnant:
            # Drop the rung as well as the state: leaving it applied would keep
            # a recovered run on an intervention nobody chose and nothing needs.
            return LadderState(
                state=NORMAL,
                rung=-1,
                entered_at_games=games,
                history=state.history
                + (f"{self.rungs[state.rung].name} recovered",),
            )
        return self._escalate(state, games, f"rung {state.rung} did not recover")

    def _escalate(
        self, state: LadderState, games: int, reason: str
    ) -> LadderState:
        rung = state.rung + 1
        if rung >= len(self.rungs):
            # The last rung stays applied. Dropping every intervention at the
            # moment the ladder runs out would be a fifth, uncontrolled change
            # on top of the four that did not work.
            return replace(
                state,
                state=EXHAUSTED,
                entered_at_games=games,
                history=state.history + ("ladder exhausted",),
            )
        return LadderState(
            state=INTERVENING,
            rung=rung,
            entered_at_games=games,
            history=state.history + (f"{reason} -> {self.rungs[rung].name}",),
        )
