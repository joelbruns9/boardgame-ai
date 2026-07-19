"""Sequential probability-ratio testing for promotion and anchor gates."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class SPRTResult:
    decision: str
    games: int
    score: float
    score_rate: float
    log_likelihood_ratio: float
    lower_bound: float
    upper_bound: float


class SPRT:
    """Bernoulli SPRT with draws represented as half a win/half a loss."""

    def __init__(
        self,
        p0: float,
        p1: float,
        *,
        alpha: float = 0.05,
        beta: float = 0.05,
    ):
        if not 0.0 < p0 < p1 < 1.0:
            raise ValueError("require 0 < p0 < p1 < 1")
        if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
            raise ValueError("alpha and beta must lie in (0, 1)")
        self.p0 = p0
        self.p1 = p1
        self.lower = math.log(beta / (1.0 - alpha))
        self.upper = math.log((1.0 - beta) / alpha)
        self.games = 0
        self.score = 0.0
        self.llr = 0.0

    def update(self, outcome: float) -> SPRTResult:
        if outcome not in (0.0, 0.5, 1.0):
            raise ValueError("outcome must be loss=0, draw=0.5, or win=1")
        self.games += 1
        self.score += outcome
        self.llr += outcome * math.log(self.p1 / self.p0)
        self.llr += (1.0 - outcome) * math.log(
            (1.0 - self.p1) / (1.0 - self.p0)
        )
        if self.llr >= self.upper:
            decision = "accept"
        elif self.llr <= self.lower:
            decision = "reject"
        else:
            decision = "continue"
        return SPRTResult(
            decision=decision,
            games=self.games,
            score=self.score,
            score_rate=self.score / self.games,
            log_likelihood_ratio=self.llr,
            lower_bound=self.lower,
            upper_bound=self.upper,
        )

    def result(self) -> SPRTResult:
        decision = (
            "accept"
            if self.llr >= self.upper
            else "reject"
            if self.llr <= self.lower
            else "continue"
        )
        return SPRTResult(
            decision,
            self.games,
            self.score,
            self.score / self.games if self.games else 0.0,
            self.llr,
            self.lower,
            self.upper,
        )
