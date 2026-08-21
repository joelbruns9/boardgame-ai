"""
Self-play training support: auxiliary targets and diversity measurement.

This module holds nothing that plays the game.  It exists to answer the two
questions that make Welcome To awkward to train on.

────────────────────────────────────────────────────────────────────────────
1. EARLY PLACEMENTS, LATE CONSEQUENCES
────────────────────────────────────────────────────────────────────────────
Writing a 15 into the first box of a street on turn 3 costs nine future
placements.  Your score on turn 3 is unchanged.  The bill arrives around turn 20
as permit refusals and as parks, pools and estates you never got to build.  No
search reaches that far — a 25-turn horizon with chance at every turn boundary is
out of reach at any realistic simulation count — so the *value function* has to
carry it, and a single scalar "final score" target is a thin, high-variance way to
teach it.

The fix is to decompose the target.  :func:`final_outcomes` reads, off a finished
game, the things whose causes are local and whose values are late:

* ``permits`` — the direct, low-noise signature of capacity mismanagement.  A
  model that can predict "this sheet will end up taking two refusals" has learned
  the thing that is hard to learn, and it can learn it from turn 3 onwards.
* ``houses`` and ``capacity_left`` — did this sheet get filled or did it seize up?
* per-component scores — eight gradients instead of one, each attributable to a
  different part of the sheet.
* ``plan_turns`` — feeds the turns-to-finish head that the three races turn on.

These are auxiliary heads, in the KataGo sense: they are never consulted at play
time, they exist to force the trunk to represent the long horizon.  Combined with
the capacity features the encoder already supplies, the network is asked to learn
what capacity is *worth* rather than to rediscover the arithmetic of the
ascending rule.

────────────────────────────────────────────────────────────────────────────
2. SELF-PLAY DIVERGENCE
────────────────────────────────────────────────────────────────────────────
In standard mode every seat sees the same three combinations.  A deterministic
policy playing itself from identical empty sheets therefore produces identical
sheets, forever: the game degenerates and the seats are worth one sample between
them.

What saves it is that divergence is *self-amplifying* — one differing choice makes
the sheets differ, and every later decision is then taken in a different state, so
the trajectories never re-converge.  The whole problem is therefore concentrated
in the opening.  :func:`sheet_divergence` measures whether it is actually
happening, and should be logged every iteration; if it decays as the policy
sharpens, the sampling temperature is too low too early.

The other half of the answer is an opponent *pool*: two different policies cannot
mirror each other, so a learner playing frozen checkpoints or GreedyBot has no
collapse problem at all.  Symmetric self-play, where the same policy fills every
seat, is the only configuration that needs the guards above.

Dropping to a single seat is **not** an available answer, tempting as it looks:
one seat costs the same per learner trajectory as two, and it switches scoring
rules (``TEMP_SOLO_SCORE`` for the 7/4/1 ranking, every City Plan paying its
first-place value), so the data describes a different game.  See
``SELF_PLAY_PLAN.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

from games.welcome_to.constants import NUM_BOXES
from games.welcome_to.game import GameState
from games.welcome_to.sheet import SheetScore

#: Target value for "this player never completed that plan".  Pair every
#: turns-to-finish target with its mask rather than regressing on the sentinel.
NEVER: int = -1


@dataclass(frozen=True, slots=True)
class PlayerOutcome:
    """Everything a finished game says about one seat."""

    player: int
    score: int
    components: SheetScore
    permits: int
    houses: int
    capacity_left: int
    #: Absolute turn on which this player completed each plan slot, or ``None``.
    plan_turns: tuple[Optional[int], ...]
    plans_completed: int
    #: Plan slots in the order this player completed them.
    plan_order: tuple[int, ...]
    final_turn: int
    rank: int
    won: bool


def final_outcomes(state: GameState) -> list[PlayerOutcome]:
    """Read the per-seat training targets off a finished game."""
    if not state.is_terminal:
        raise ValueError("outcomes are only defined for a finished game")

    scores = state.scores()
    ranking = state.ranking()
    winners = set(state.winners())
    rank_of = {player: i for i, player in enumerate(ranking)}

    out: list[PlayerOutcome] = []
    for player in range(state.config.players):
        sheet = state.sheets[player]
        turns = tuple(state.plan_turns[slot].get(player) for slot in range(3))
        completed = [(t, slot) for slot, t in enumerate(turns) if t is not None]
        completed.sort()
        out.append(
            PlayerOutcome(
                player=player,
                score=scores[player],
                components=state.score_breakdown(player),
                permits=sheet.permits,
                houses=sum(1 for row in sheet.numbers for n in row if n is not None),
                capacity_left=sum(sheet.placement_capacity()),
                plan_turns=turns,
                plans_completed=len(completed),
                plan_order=tuple(slot for _, slot in completed),
                final_turn=state.turn,
                rank=rank_of[player],
                won=player in winners,
            )
        )
    return out


def sample_targets(outcome: PlayerOutcome, turn: int) -> dict[str, float]:
    """Targets for one visited state, given the turn it was visited on.

    ``turns_to_plan_*`` are relative to ``turn`` so the head predicts "how many
    turns from here", which is the quantity that is comparable across a game.
    Each comes with a ``_mask`` companion that is 0 when the plan was never
    completed; train the head only where the mask is 1.
    """
    targets: dict[str, float] = {
        "score": float(outcome.score),
        "won": 1.0 if outcome.won else 0.0,
        "rank": float(outcome.rank),
        "permits": float(outcome.permits),
        "houses": float(outcome.houses),
        "capacity_left": float(outcome.capacity_left),
        "turns_left": float(max(0, outcome.final_turn - turn)),
        "plans_completed": float(outcome.plans_completed),
        "score_parks": float(outcome.components.parks),
        "score_pools": float(outcome.components.pools),
        "score_estates": float(outcome.components.estates),
        "score_plans": float(outcome.components.plans),
        "score_temp": float(outcome.components.temp),
        "score_bis": float(outcome.components.bis),
        "score_permits": float(outcome.components.permits),
        "score_roundabouts": float(outcome.components.roundabouts),
    }
    for slot, completed_on in enumerate(outcome.plan_turns):
        if completed_on is None:
            targets[f"turns_to_plan_{slot}"] = float(NEVER)
            targets[f"turns_to_plan_{slot}_mask"] = 0.0
        else:
            targets[f"turns_to_plan_{slot}"] = float(max(0, completed_on - turn))
            targets[f"turns_to_plan_{slot}_mask"] = 1.0
    targets["first_plan"] = float(outcome.plan_order[0]) if outcome.plan_order else float(NEVER)
    targets["first_plan_mask"] = 1.0 if outcome.plan_order else 0.0
    return targets


#: The auxiliary target names, for building heads.  ``score`` and ``won`` are the
#: real value heads; the rest are there to shape the trunk.
TARGET_NAMES: tuple[str, ...] = tuple(
    sample_targets(
        PlayerOutcome(
            player=0,
            score=0,
            components=SheetScore(),
            permits=0,
            houses=0,
            capacity_left=0,
            plan_turns=(None, None, None),
            plans_completed=0,
            plan_order=(),
            final_turn=1,
            rank=0,
            won=False,
        ),
        turn=1,
    )
)


# ──────────────────────────────────────────────────────────────────────────
# Diversity
# ──────────────────────────────────────────────────────────────────────────
def sheet_divergence(state: GameState) -> float:
    """Mean fraction of house boxes on which two seats differ.

    ``0.0`` means every seat wrote exactly the same sheet — the degenerate case
    that symmetric self-play falls into when the policy is confident and the
    sampling temperature is low.  Log this per self-play iteration; it is the
    canary for the whole problem.

    Returns ``0.0`` for a single-seat game, where the question does not arise.
    """
    players = state.config.players
    if players < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for a, b in combinations(range(players), 2):
        left, right = state.sheets[a], state.sheets[b]
        differing = sum(
            1
            for x in range(3)
            for y in range(len(left.numbers[x]))
            if left.numbers[x][y] != right.numbers[x][y]
        )
        total += differing / NUM_BOXES
        pairs += 1
    return total / pairs


def first_divergence_turn(state: GameState) -> Optional[int]:
    """The earliest turn on which any two seats wrote a different box.

    ``None`` if the seats never diverged.  Divergence is self-amplifying, so this
    is the number that matters: if it is not small, the opening temperature is
    doing its job.
    """
    players = state.config.players
    if players < 2:
        return None
    earliest: Optional[int] = None
    for a, b in combinations(range(players), 2):
        left, right = state.sheets[a], state.sheets[b]
        for x in range(3):
            for y in range(len(left.numbers[x])):
                if left.numbers[x][y] == right.numbers[x][y]:
                    continue
                turns = [
                    t
                    for t in (left.written_turn[x][y], right.written_turn[x][y])
                    if t >= 0
                ]
                if not turns:
                    continue
                candidate = min(turns)
                if earliest is None or candidate < earliest:
                    earliest = candidate
    return earliest


def diversity_report(states: list[GameState]) -> dict[str, float]:
    """Summarise divergence over a batch of finished self-play games."""
    if not states:
        return {}
    divergences = [sheet_divergence(s) for s in states]
    firsts = [first_divergence_turn(s) for s in states]
    seen = [t for t in firsts if t is not None]
    spreads = [max(s.scores()) - min(s.scores()) for s in states]
    return {
        "games": float(len(states)),
        "mean_sheet_divergence": sum(divergences) / len(divergences),
        "identical_games": sum(1 for d in divergences if d == 0.0) / len(divergences),
        "mean_first_divergence_turn": (sum(seen) / len(seen)) if seen else float("nan"),
        "mean_score_spread": sum(spreads) / len(spreads),
    }
