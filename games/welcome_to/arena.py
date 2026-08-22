"""
The paired-seed harness — permanent, and the gate every later change goes through.

``SELF_PLAY_PLAN.md`` S1: *"The paired-seed harness built here is permanent. Any
change to search or the network gets a paired-seed score against a fixed opponent
before it gets a training run."*

WHY PAIRED, AND WHY ONE SEAT
────────────────────────────
Per-game score variance in Welcome To is tens of points, so an unpaired
comparison over a few hundred games cannot resolve the 4-point difference the S1
gate is about.  Sharing the seed gives both arms the same deck and the same City
Plans, which removes most of that.

And **one seat is substituted, not the whole table.** Swapping every seat at once
changes something other than the policy under test: how *correlated* the seats
are.  Seats running the same deterministic policy off the same shared stacks
converge on each other's sheets, and correlated sheets complete plans on the same
turn — so they *share* first-place plan values instead of racing for them, and
they tie on the temp-agency rank.  Those are scoring rules worth 6-14 and 7/4/1
points moving for a reason unrelated to skill.  (Measured on the S0 gate before
it was fixed: 0.34 mean sheet divergence for an all-net table against GreedyBot's
0.80.)

The counterfactual is not perfectly clean even so — the substituted seat changes
when the game ends and who wins which plan, so the opponents' games are not
bit-identical across arms.  The deck, the plans and the opponents' policies are
shared, which removes most of it, and the stderr reported here is what says
whether what is left matters.

HOW MANY GAMES THE GATE NEEDS
─────────────────────────────
**Measured**, GreedyBot against GreedyBot at two seats: the per-game paired delta
has a standard deviation of about **18 points**, so the standard error of the
mean is ``18 / sqrt(n)`` — about 4.1 at 60 games and 1.0 at 330.

A 4-point gate read off 60 games is therefore **inside its own noise** and means
nothing.  Run at least ~300 paired games before believing a pass, and read
``score_gap_stderr`` every time rather than the gap alone.

PLANS PER GAME IS THE NUMBER THAT MATTERS
─────────────────────────────────────────
GreedyBot completes **0.42** plans per game: it is structurally race-blind, and
no amount of tuning its placement heuristic changes that.  So "search completes
≥ 1.0 plans per game" is the first evidence of something the teacher cannot do,
while a score gain could in principle come from better placement alone.  Both are
reported; the plan number is the one to read first.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from games.welcome_to import macro_codec as mc
from games.welcome_to.bots import GreedyBot
from games.welcome_to.game import GameConfig, GameState

#: A contestant: given a state and the seat it plays, apply exactly one macro.
Player = Callable[[GameState, int, random.Random], None]


def greedy_player() -> Player:
    """GreedyBot, the fixed reference.  Its RNG is seeded per game by the caller."""

    def move(state: GameState, seat: int, rng: random.Random) -> None:
        bot = GreedyBot(rng)
        state.apply(bot.act(state))

    return move


def _seeded_bots(config: GameConfig, game_seed: int) -> dict[int, GreedyBot]:
    return {
        p: GreedyBot(random.Random(game_seed * 100 + p)) for p in range(config.players)
    }


@dataclass(frozen=True, slots=True)
class ArenaResult:
    games: int
    score_gap: float
    score_gap_stderr: float
    subject_score: float
    baseline_score: float
    subject_plans: float
    baseline_plans: float
    subject_wins: float

    def summary(self) -> str:
        return (
            f"{self.games} paired games | "
            f"score {self.subject_score:.2f} vs {self.baseline_score:.2f} "
            f"({self.score_gap:+.2f} ± {self.score_gap_stderr:.2f}) | "
            f"plans {self.subject_plans:.2f} vs {self.baseline_plans:.2f} | "
            f"win rate {self.subject_wins:.2f}"
        )


def _play(
    config: GameConfig,
    game_seed: int,
    seat: int,
    subject: Optional[Player],
) -> tuple[list[int], int, bool]:
    """One game.  ``subject`` plays ``seat``; GreedyBots play everyone else.

    ``subject=None`` is the baseline arm: a GreedyBot in ``seat`` too, on the
    same RNG stream, so the two arms differ in exactly one thing.
    """
    bots = _seeded_bots(config, game_seed)
    subject_rng = random.Random(game_seed * 7919 + seat)
    state = GameState.new(seed=game_seed, config=config)
    steps = 0
    while not state.is_terminal:
        actor = state.actor
        if actor == seat and subject is not None:
            subject(state, actor, subject_rng)
        else:
            state.apply(bots[actor].act(state))
        steps += 1
        if steps > 20000:  # pragma: no cover - a stuck engine, not a rules case
            raise RuntimeError("game did not terminate")

    plans = sum(1 for slot in range(3) if seat in state.plan_turns[slot])
    return state.scores(), plans, seat in state.winners()


def paired(
    subject: Player,
    games: int = 60,
    seats: int = 2,
    seed: int = 20_000,
    advanced: bool = True,
) -> ArenaResult:
    """``subject`` in one rotating seat against GreedyBots, paired on the seed.

    Two seats is the cheapest configuration that still has all four race
    mechanics live, which is why the S1 gate uses it.
    """
    config = GameConfig(players=seats, advanced=advanced)
    deltas: list[float] = []
    subject_scores: list[float] = []
    baseline_scores: list[float] = []
    subject_plans: list[float] = []
    baseline_plans: list[float] = []
    wins: list[float] = []

    for i in range(games):
        game_seed = seed + i
        seat = i % seats  # rotate, so no seat's luck is the whole result

        with_subject, plans_a, won = _play(config, game_seed, seat, subject)
        without, plans_b, _ = _play(config, game_seed, seat, None)

        subject_scores.append(float(with_subject[seat]))
        baseline_scores.append(float(without[seat]))
        subject_plans.append(float(plans_a))
        baseline_plans.append(float(plans_b))
        deltas.append(float(with_subject[seat] - without[seat]))
        wins.append(1.0 if won else 0.0)

    mean = sum(deltas) / games
    variance = sum((d - mean) ** 2 for d in deltas) / max(games - 1, 1)
    return ArenaResult(
        games=games,
        score_gap=mean,
        score_gap_stderr=math.sqrt(variance / games),
        subject_score=sum(subject_scores) / games,
        baseline_score=sum(baseline_scores) / games,
        subject_plans=sum(subject_plans) / games,
        baseline_plans=sum(baseline_plans) / games,
        subject_wins=sum(wins) / games,
    )


def s1_gate(result: ArenaResult) -> dict[str, bool]:
    """The two S1 conditions.  Read ``plans`` first — it is the structural one."""
    return {
        "score_beats_greedy_by_4": result.score_gap >= 4.0,
        "completes_a_plan_per_game": result.subject_plans >= 1.0,
    }
