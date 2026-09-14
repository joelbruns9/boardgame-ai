"""Can't Stop rules engine, parameterized over every supported variant.

Variants: 2-4 players; base (3 columns to win) or extended columns
(2p: 5, 3p: 4, 4p: 3 -- identical to base); blocking on or off. That is 10
distinct rule sets, listed in ``ALL_RULESETS``.

Blocking: a turn cannot end by stopping while any of the active player's
runners is on the same space as another player's saved marker. Runners may
land on and pass those spaces freely mid-turn, so two players' saved markers
never share a space.

Turn flow (``state.phase``):

    AWAIT_ROLL --roll(dice)--> AWAIT_MOVE --apply_move--> AWAIT_DECISION
         ^            |                                     |      |
         |            +-- no legal move: bust --------------+      |
         +---------- roll again (from AWAIT_DECISION) ------+      |
         +---------- stop (if can_stop) -> next player ------------+

The engine never draws randomness itself: callers pass dice to ``roll``.
``random_dice`` is a convenience for callers holding a ``random.Random``.

Board coordinates: columns are the dice sums 2..12. A position on a column
is the number of spaces climbed, 0 (off the board) up to the column height
(the top). Reaching the top and stopping claims the column.
"""

from dataclasses import dataclass
from enum import IntEnum

# ---- Board constants ----

COLUMNS = tuple(range(2, 13))
COLUMN_HEIGHTS = {2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13,
                  8: 11, 9: 9, 10: 7, 11: 5, 12: 3}
MAX_RUNNERS = 3

BASE_COLUMNS_TO_WIN = 3
EXTENDED_COLUMNS_TO_WIN = {2: 5, 3: 4, 4: 3}


# ---- Rules ----

@dataclass(frozen=True)
class RuleSet:
    """One supported rule variant. Construct via ``RuleSet.make``."""
    num_players: int
    columns_to_win: int
    blocking: bool

    def __post_init__(self):
        if self.num_players not in EXTENDED_COLUMNS_TO_WIN:
            raise ValueError(f"num_players must be 2-4, got {self.num_players}")
        allowed = {BASE_COLUMNS_TO_WIN, EXTENDED_COLUMNS_TO_WIN[self.num_players]}
        if self.columns_to_win not in allowed:
            raise ValueError(
                f"{self.num_players} players cannot play to "
                f"{self.columns_to_win} columns (allowed: {sorted(allowed)})")

    @classmethod
    def make(cls, num_players, extended=False, blocking=False):
        cols = (EXTENDED_COLUMNS_TO_WIN[num_players] if extended
                else BASE_COLUMNS_TO_WIN)
        return cls(num_players, cols, bool(blocking))


ALL_RULESETS = tuple(sorted(
    {RuleSet.make(n, ext, blk)
     for n in (2, 3, 4) for ext in (False, True) for blk in (False, True)},
    key=lambda r: (r.num_players, r.columns_to_win, r.blocking)))


class Phase(IntEnum):
    AWAIT_ROLL = 0      # turn start, or the player chose to roll again
    AWAIT_MOVE = 1      # dice rolled, at least one legal move
    AWAIT_DECISION = 2  # move applied; stop (if legal) or roll again
    GAME_OVER = 3


# ---- State ----

class GameState:
    """Complete game state.

    progress[p][col]  saved position of player p on col (0 = off board).
                      Zero on claimed columns.
    claimed_by[col]   player who claimed col, or None.
    runners[col]      absolute position of the active player's runner on col.
    dice              the current roll (sorted 4-tuple) while AWAIT_MOVE.
    """
    __slots__ = ("rules", "active_player", "progress", "claimed_by",
                 "runners", "dice", "phase", "winner")

    def __init__(self, rules):
        self.rules = rules
        self.active_player = 0
        self.progress = [dict.fromkeys(COLUMNS, 0)
                         for _ in range(rules.num_players)]
        self.claimed_by = dict.fromkeys(COLUMNS)
        self.runners = {}
        self.dice = None
        self.phase = Phase.AWAIT_ROLL
        self.winner = None

    def clone(self):
        new = GameState.__new__(GameState)
        new.rules = self.rules
        new.active_player = self.active_player
        new.progress = [p.copy() for p in self.progress]
        new.claimed_by = self.claimed_by.copy()
        new.runners = self.runners.copy()
        new.dice = self.dice
        new.phase = self.phase
        new.winner = self.winner
        return new

    @property
    def game_over(self):
        return self.phase == Phase.GAME_OVER

    def claimed_columns(self, player):
        return [c for c in COLUMNS if self.claimed_by[c] == player]

    def position(self, col):
        """Active player's current position on col, runner included."""
        return self.runners.get(col, self.progress[self.active_player][col])

    def __repr__(self):
        r = self.rules
        lines = [f"Can't Stop {r.num_players}p to {r.columns_to_win}"
                 f"{' blocking' if r.blocking else ''} | {self.phase.name}"
                 f" | active {self.active_player} | dice {self.dice}"
                 f" | runners {self.runners}"]
        for p in range(r.num_players):
            saved = {c: v for c, v in self.progress[p].items() if v}
            lines.append(f"  p{p} claimed {self.claimed_columns(p)} saved {saved}")
        if self.winner is not None:
            lines.append(f"  winner p{self.winner}")
        return "\n".join(lines)


# ---- Dice ----

def random_dice(rng):
    return tuple(rng.randint(1, 6) for _ in range(4))


def dice_pairings(dice):
    """Distinct (low, high) sum pairs from the three ways to split 4 dice."""
    d0, d1, d2, d3 = dice
    out = []
    for a, b in ((d0 + d1, d2 + d3), (d0 + d2, d1 + d3), (d0 + d3, d1 + d2)):
        pair = (a, b) if a <= b else (b, a)
        if pair not in out:
            out.append(pair)
    return out


# ---- Moves ----
#
# A move is a sorted tuple of the columns advanced, one entry per step:
#   (6, 8)  one step on 6 and one on 8
#   (7, 7)  two steps on 7
#   (6,)    one step on 6
# Moves are normalized to their effect: a double on a column with only one
# space left is (7,), so equal outcomes compare equal.

def _column_open(state, col):
    """Active player may place/advance a runner on col (ignoring the cap)."""
    return (state.claimed_by[col] is None
            and state.position(col) < COLUMN_HEIGHTS[col])


def legal_moves(state, dice):
    """Legal moves for the active player given dice. Empty list = bust.

    Rules: for each way of splitting the dice, use both sums if possible;
    otherwise use whichever single sum is usable. When both sums are
    individually usable but the runner cap forbids opening both, either one
    may be played alone.
    """
    runners = state.runners
    free_slots = MAX_RUNNERS - len(runners)
    moves = set()
    for a, b in dice_pairings(dice):
        if a == b:
            if not _column_open(state, a):
                continue
            if a not in runners and free_slots == 0:
                continue
            room = COLUMN_HEIGHTS[a] - state.position(a)
            moves.add((a, a) if room >= 2 else (a,))
            continue
        ok_a = _column_open(state, a) and (a in runners or free_slots > 0)
        ok_b = _column_open(state, b) and (b in runners or free_slots > 0)
        new_needed = (a not in runners) + (b not in runners)
        if ok_a and ok_b and new_needed <= free_slots:
            moves.add((a, b))
        else:
            if ok_a:
                moves.add((a,))
            if ok_b:
                moves.add((b,))
    return sorted(moves)


# ---- Transitions ----

def _require(state, *phases):
    if state.phase not in phases:
        raise ValueError(f"illegal in phase {state.phase.name}")


def roll(state, dice):
    """Roll for the active player. Busts (and passes the turn) if no legal
    move exists. Returns the legal moves; empty means the roll busted."""
    _require(state, Phase.AWAIT_ROLL, Phase.AWAIT_DECISION)
    dice = tuple(sorted(dice))
    moves = legal_moves(state, dice)
    if not moves:
        bust(state)
        return moves
    state.dice = dice
    state.phase = Phase.AWAIT_MOVE
    return moves


def apply_move(state, move):
    _require(state, Phase.AWAIT_MOVE)
    move = tuple(sorted(move))
    if move not in legal_moves(state, state.dice):
        raise ValueError(f"illegal move {move} for dice {state.dice}")
    for col in move:
        state.runners[col] = state.position(col) + 1
    state.dice = None
    state.phase = Phase.AWAIT_DECISION


def stop_blocked(state):
    """Blocking variant: a runner shares a space with another player's
    saved marker, so the turn cannot end by stopping."""
    if not state.rules.blocking:
        return False
    me = state.active_player
    for col, pos in state.runners.items():
        for p, prog in enumerate(state.progress):
            if p != me and prog[col] == pos:
                return True
    return False


def can_stop(state):
    return state.phase == Phase.AWAIT_DECISION and not stop_blocked(state)


def stop(state):
    """Bank runners, claim finished columns, check for a win, pass the turn."""
    _require(state, Phase.AWAIT_DECISION)
    if stop_blocked(state):
        raise ValueError("cannot stop: a runner is on another player's marker")
    me = state.active_player
    for col, pos in state.runners.items():
        if pos >= COLUMN_HEIGHTS[col]:
            state.claimed_by[col] = me
            for prog in state.progress:
                prog[col] = 0
        else:
            state.progress[me][col] = pos
    state.runners = {}
    if len(state.claimed_columns(me)) >= state.rules.columns_to_win:
        state.winner = me
        state.phase = Phase.GAME_OVER
        return
    _next_player(state)


def bust(state):
    """Lose the runners and pass the turn."""
    _require(state, Phase.AWAIT_ROLL, Phase.AWAIT_DECISION)
    state.runners = {}
    _next_player(state)


def _next_player(state):
    state.active_player = (state.active_player + 1) % state.rules.num_players
    state.dice = None
    state.phase = Phase.AWAIT_ROLL
