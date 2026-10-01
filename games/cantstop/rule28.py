"""Rule of 28 opponents: an outside yardstick that owes nothing to our net.

Glenn & Aloi, "A Generalized Heuristic for Can't Stop" (FLAIRS 2009), give
the Rule of 28 (Keller 1986) and two genetic-algorithm champions tuned for
solitaire Can't Stop (fewest turns to claim three columns):

    RULE_OF_28           10.74 turns (paper)
    CONSTANT_CHAMPION     9.12 turns (paper Table 1)
    LINEAR_CHAMPION       9.05 turns (paper Table 2)

Head to head in 2p (paper Table 4, 10k games): the constant champion beats
the Rule of 28 0.74 moving first and 0.64 moving second.

**Stopping.** Sum, over the columns holding a runner, the column's progress
weight times one more than the spaces advanced THIS turn; if all three runners
are placed, add the difficulty scores (all odd / all even / all >= 7 /
all <= 7; column 7 counts as both high and low). Stop at or above the
threshold. ``bank_wins`` (default on, not in the paper) also stops whenever
stopping wins the game; the paper's solitaire numbers are reproduced with it
off. A blocked stop (blocking variant) rolls on, as the rules force.

**Moves.** Each move scores the sum of the space weights it advances over,
minus the marker penalty for every runner it newly places. Space weight j
(1-based) on column c of length l is ``floor(slope * j / l + intercept)``;
the constant-weight genomes have slope 0. The first maximal move in
``engine.legal_moves`` order wins ties.

The heuristics ignore the opponents' positions (the paper plays them head to
head exactly so), which is what makes them a plausible human-style opponent.
"""

from dataclasses import dataclass
import math

from .engine import (COLUMN_HEIGHTS, COLUMNS, Phase, apply_move, can_stop,
                     legal_moves, random_dice, roll, stop)
from .decision_search import winning_bank


def _mirror(values):
    """Weights for columns 2..7 extended symmetrically to 2..12."""
    return {c: values[min(c, 14 - c) - 2] for c in COLUMNS}


@dataclass(frozen=True)
class Rule28Params:
    name: str
    progress: tuple          # progress weight, columns 2..7 (mirrored)
    move_slope: tuple        # per-space move weight slope, columns 2..7
    move_intercept: tuple    # per-space move weight intercept, columns 2..7
    odd: int
    even: int
    high: int
    low: int
    marker: int
    threshold: int

    def progress_weight(self, col):
        return _mirror(self.progress)[col]

    def space_weight(self, col, j):
        i = min(col, 14 - col) - 2
        return math.floor(self.move_slope[i] * j / COLUMN_HEIGHTS[col]
                          + self.move_intercept[i])


# Keller's rule: progress weight |7 - c| + 1, move weight 6 - |7 - c|,
# all odd +2, all even -2, all high / all low +4, marker penalty 6, 28.
RULE_OF_28 = Rule28Params(
    "rule_of_28", progress=(6, 5, 4, 3, 2, 1), move_slope=(0,) * 6,
    move_intercept=(1, 2, 3, 4, 5, 6), odd=2, even=-2, high=4, low=4,
    marker=6, threshold=28)

# Paper Table 1.
CONSTANT_CHAMPION = Rule28Params(
    "constant_champion", progress=(7, 7, 3, 2, 2, 1), move_slope=(0,) * 6,
    move_intercept=(7, 0, 2, 0, 4, 3), odd=7, even=1, high=6, low=5,
    marker=6, threshold=29)

# Paper Table 2 (low = high, as the genome ties them).
LINEAR_CHAMPION = Rule28Params(
    "linear_champion", progress=(7, 6, 4, 3, 2, 1),
    move_slope=(64, 24, 28, 8, 18, 12), move_intercept=(7, 1, 2, 1, 3, 4),
    odd=1, even=0, high=-4, low=-4, marker=4, threshold=24)

HEURISTICS = {p.name: p for p in (RULE_OF_28, CONSTANT_CHAMPION,
                                  LINEAR_CHAMPION)}


def progress_value(state, params):
    """The stop score of the active player's current runners."""
    me = state.progress[state.active_player]
    value = sum((pos - me[c] + 1) * params.progress_weight(c)
                for c, pos in state.runners.items())
    cols = list(state.runners)
    if len(cols) == 3:
        if all(c % 2 for c in cols):
            value += params.odd
        if not any(c % 2 for c in cols):
            value += params.even
        if all(c >= 7 for c in cols):
            value += params.high
        if all(c <= 7 for c in cols):
            value += params.low
    return value


def move_value(state, move, params):
    """Space weights advanced over, minus the penalty per newly placed runner."""
    value = 0
    counts = {}
    for c in move:
        counts[c] = counts.get(c, 0) + 1
    for c, k in counts.items():
        start = state.position(c)
        value += sum(params.space_weight(c, j)
                     for j in range(start + 1, start + k + 1))
        if c not in state.runners:
            value -= params.marker
    return value


def choose_move(state, params):
    moves = legal_moves(state, state.dice)
    return max(moves, key=lambda m: move_value(state, m, params))


def should_stop(state, params, bank_wins=True):
    if not can_stop(state):
        return False
    if bank_wins and winning_bank(state):
        return True
    return progress_value(state, params) >= params.threshold


class Rule28Player:
    """A seat played by a Rule of 28 heuristic (``arena``-style turn player)."""

    def __init__(self, params=RULE_OF_28, bank_wins=True):
        self.params, self.bank_wins = params, bank_wins

    def play_turn(self, state, rng):
        """Play the active player's whole turn, rolling from ``rng``."""
        while True:
            if not roll(state, random_dice(rng)):
                return                                 # busted
            apply_move(state, choose_move(state, self.params))
            if should_stop(state, self.params, self.bank_wins):
                stop(state)
                return


def solitaire_turns(params, rng, bank_wins=False, max_turns=1000):
    """Turns one player needs to claim three columns, alone on the board."""
    from .engine import GameState, RuleSet
    state = GameState(RuleSet.make(2))
    player = Rule28Player(params, bank_wins)
    turns = 0
    while not state.game_over:
        if turns >= max_turns:
            raise RuntimeError("solitaire game did not finish")
        state.active_player = 0
        state.phase = Phase.AWAIT_ROLL
        player.play_turn(state, rng)
        turns += 1
    return turns
