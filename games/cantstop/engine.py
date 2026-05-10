# engine.py
# Can't Stop game engine — performance optimized.
#
# Optimizations applied:
# - __slots__ on GameState for memory efficiency
# - Integer player IDs (0, 1) instead of strings
# - all_claimed maintained as running set on state
# - can_use_column inlined into get_valid_moves and apply_move
# - state.runners used directly (dict lookup is already O(1))
# - max_turns reduced with heuristic fallback
# - Normalized move tuples at source
# - Unambiguous move representation: (a,b) normal, (a,a) double, (a,) partial

import random

# ---- BOARD CONSTANTS ----
COLUMN_HEIGHTS = {
    2: 3,  3: 5,  4: 7,  5: 9,  6: 11,
    7: 13, 8: 11, 9: 9, 10: 7, 11: 5, 12: 3
}

COLUMNS_TO_WIN = 3
MAX_RUNNERS = 3
ALL_COLUMNS = list(COLUMN_HEIGHTS.keys())


# ---- GAME STATE ----
class GameState:
    """
    Holds complete game state at one moment in time.
    Uses __slots__ for memory efficiency.
    Players are integers: 0, 1.
    """
    __slots__ = [
        'players',
        'active_player',
        'claimed',
        'all_claimed',
        'progress',
        'runners',
        'dice',
        'game_over',
        'winner',
    ]

    def __init__(self, num_players=2):
        self.players = list(range(num_players))
        self.active_player = 0
        self.claimed = {p: set() for p in self.players}
        self.all_claimed = set()
        self.progress = {p: {} for p in self.players}
        self.runners = {}
        self.dice = []
        self.game_over = False
        self.winner = None

    def roll_dice(self):
        """Roll 4 fresh dice. Only call at start of a new turn."""
        self.dice = [random.randint(1, 6) for _ in range(4)]
        return self.dice

    def get_current_progress(self, player, column):
        """
        Total progress on a column including active runner.
        Only adds runner if player is the active player.
        """
        saved = self.progress[player].get(column, 0)
        runner = self.runners.get(column, 0) if player == self.active_player else 0
        return saved + runner

    def save_snapshot(self):
        """Lightweight snapshot of mutable state."""
        return {
            'active_player': self.active_player,
            'claimed':       {p: set(cols) for p, cols in self.claimed.items()},
            'all_claimed':   set(self.all_claimed),
            'progress':      {p: dict(prog) for p, prog in self.progress.items()},
            'runners':       dict(self.runners),
            'dice':          list(self.dice),
            'game_over':     self.game_over,
            'winner':        self.winner,
        }

    def restore_snapshot(self, snap):
        """Restore state from a snapshot."""
        self.active_player = snap['active_player']
        self.claimed =     {p: set(cols) for p, cols in snap['claimed'].items()}
        self.all_claimed = set(snap['all_claimed'])
        self.progress =    {p: dict(prog) for p, prog in snap['progress'].items()}
        self.runners =     dict(snap['runners'])
        self.dice =        list(snap['dice'])
        self.game_over =   snap['game_over']
        self.winner =      snap['winner']

    def clone(self):
        """Full independent copy. Used for rollouts."""
        new = GameState(len(self.players))
        new.active_player = self.active_player
        new.claimed =     {p: set(cols) for p, cols in self.claimed.items()}
        new.all_claimed = set(self.all_claimed)
        new.progress =    {p: dict(prog) for p, prog in self.progress.items()}
        new.runners =     dict(self.runners)
        new.dice =        list(self.dice)
        new.game_over =   self.game_over
        new.winner =      self.winner
        return new

    def heuristic_value(self, player):
        """
        Estimate win probability when game doesn't finish
        within max_turns. Returns float 0.0 to 1.0.
        Uses relative advantage — empty board returns 0.5.
        """
        opponent = 1 - player

        my_claimed  = len(self.claimed[player])
        opp_claimed = len(self.claimed[opponent])
        total_claimed = my_claimed + opp_claimed

        claimed_score = 0.5 if total_claimed == 0 else my_claimed / total_claimed

        my_progress = sum(
            steps / COLUMN_HEIGHTS[col]
            for col, steps in self.progress[player].items()
        )
        opp_progress = sum(
            steps / COLUMN_HEIGHTS[col]
            for col, steps in self.progress[opponent].items()
        )
        total_progress = my_progress + opp_progress

        progress_score = 0.5 if total_progress == 0 else my_progress / total_progress

        return 0.8 * claimed_score + 0.2 * progress_score

    def __repr__(self):
        lines = [
            f"\n=== Can't Stop ===",
            f"Active player: {self.active_player}",
            f"Dice: {self.dice}",
            f"Runners: {self.runners}",
        ]
        for p in self.players:
            lines.append(
                f"Player {p} - claimed: {sorted(self.claimed[p])} "
                f"| progress: {self.progress[p]}"
            )
        if self.game_over:
            lines.append(f"GAME OVER - Winner: Player {self.winner}")
        return "\n".join(lines)


# ---- MOVE CALCULATION ----
def get_possible_moves(dice):
    """All unique normalized column pairs from 4 dice."""
    splits = [
        tuple(sorted((dice[0] + dice[1], dice[2] + dice[3]))),
        tuple(sorted((dice[0] + dice[2], dice[1] + dice[3]))),
        tuple(sorted((dice[0] + dice[3], dice[1] + dice[2]))),
    ]
    return list(set(splits))


def get_valid_moves(state):
    """
    Filter possible moves to legal ones.

    Move formats returned:
        (a, b)  — normal move, two different columns
        (a, a)  — true double from dice, same column twice
        (a,)    — partial move, one column blocked/full

    Uses state.runners directly — dict lookup is O(1).
    Uses state.all_claimed directly — no recomputation.
    """
    possible = get_possible_moves(state.dice)
    player = state.active_player
    all_claimed = state.all_claimed
    num_runners = len(state.runners)
    valid = []

    for col_a, col_b in possible:

        if col_a == col_b:
            # True double from dice
            col = col_a
            if col in all_claimed:
                continue
            current = (state.progress[player].get(col, 0) +
                      state.runners.get(col, 0))
            if current >= COLUMN_HEIGHTS[col]:
                continue
            if col not in state.runners and num_runners >= MAX_RUNNERS:
                continue
            valid.append((col, col))

        else:
            # Two different columns
            can_a = (
                col_a not in all_claimed and
                state.progress[player].get(col_a, 0) +
                state.runners.get(col_a, 0) < COLUMN_HEIGHTS[col_a]
            )
            can_b = (
                col_b not in all_claimed and
                state.progress[player].get(col_b, 0) +
                state.runners.get(col_b, 0) < COLUMN_HEIGHTS[col_b]
            )

            if not can_a and not can_b:
                continue

            new_needed = (
                (1 if can_a and col_a not in state.runners else 0) +
                (1 if can_b and col_b not in state.runners else 0)
            )

            if can_a and can_b and num_runners + new_needed <= MAX_RUNNERS:
                # Full move — both columns usable and runner limit ok
                valid.append((col_a, col_b))
            elif can_a and not can_b:
                # Only col_a usable
                if col_a in state.runners or num_runners < MAX_RUNNERS:
                    valid.append((col_a,))
            elif can_b and not can_a:
                # Only col_b usable
                if col_b in state.runners or num_runners < MAX_RUNNERS:
                    valid.append((col_b,))
            else:
                # Both usable but runner limit prevents new columns
                if can_a and col_a in state.runners:
                    valid.append((col_a,))
                if can_b and col_b in state.runners:
                    valid.append((col_b,))

    return list(set(valid))


def apply_move(state, move):
    """
    Advance runners based on chosen move.

    Move formats:
        (6, 8)  — normal: advance columns 6 and 8 by 1 each
        (7, 7)  — true double: advance column 7 by 2 steps
        (6,)    — partial: advance column 6 by 1 step only
    """
    player = state.active_player
    all_claimed = state.all_claimed
    num_runners = len(state.runners)

    # ---- PARTIAL MOVE ----
    if len(move) == 1:
        col = move[0]
        if col in all_claimed:
            return state
        current = (state.progress[player].get(col, 0) +
                  state.runners.get(col, 0))
        if current >= COLUMN_HEIGHTS[col]:
            return state
        if col not in state.runners and num_runners >= MAX_RUNNERS:
            return state
        state.runners[col] = state.runners.get(col, 0) + 1
        return state

    col_a, col_b = move

    # ---- TRUE DOUBLE MOVE ----
    if col_a == col_b:
        col = col_a
        if col in all_claimed:
            return state
        current = (state.progress[player].get(col, 0) +
                  state.runners.get(col, 0))
        if current >= COLUMN_HEIGHTS[col]:
            return state
        if col not in state.runners and num_runners >= MAX_RUNNERS:
            return state
        steps = min(2, COLUMN_HEIGHTS[col] - current)
        state.runners[col] = state.runners.get(col, 0) + steps

    # ---- NORMAL MOVE ----
    else:
        for col in (col_a, col_b):
            if col in all_claimed:
                continue
            current = (state.progress[player].get(col, 0) +
                      state.runners.get(col, 0))
            if current >= COLUMN_HEIGHTS[col]:
                continue
            if col not in state.runners and num_runners >= MAX_RUNNERS:
                continue
            state.runners[col] = state.runners.get(col, 0) + 1
            num_runners += 1

    return state


def stop_turn(state):
    """
    Save runner progress. Claim completed columns.
    Update all_claimed incrementally. Pass to next player.
    """
    player = state.active_player

    for col, runner_steps in list(state.runners.items()):
        saved = state.progress[player].get(col, 0)
        new_pos = saved + runner_steps

        if new_pos >= COLUMN_HEIGHTS[col] and col not in state.all_claimed:
            state.claimed[player].add(col)
            state.all_claimed.add(col)
            state.progress[player].pop(col, None)

            if len(state.claimed[player]) >= COLUMNS_TO_WIN:
                state.game_over = True
                state.winner = player
        else:
            state.progress[player][col] = new_pos

    state.runners = {}

    if not state.game_over:
        _next_player(state)

    return state


def bust_turn(state):
    """Lose runners. Pass to next player."""
    state.runners = {}
    _next_player(state)
    return state


def _next_player(state):
    """Rotate to next player."""
    state.active_player = (state.active_player + 1) % len(state.players)


# ---- TEST ----
if __name__ == "__main__":
    import time
    import copy

    print("Performance benchmark...\n")

    state = GameState(2)

    start = time.time()
    for _ in range(10000):
        cloned = state.clone()
    clone_time = time.time() - start

    start = time.time()
    for _ in range(10000):
        snap = state.save_snapshot()
        state.restore_snapshot(snap)
    snap_time = time.time() - start

    start = time.time()
    for _ in range(10000):
        copied = copy.deepcopy(state)
    deep_time = time.time() - start

    print(f"10,000 iterations:")
    print(f"  deepcopy:          {deep_time:.3f}s")
    print(f"  clone():           {clone_time:.3f}s")
    print(f"  snapshot/restore:  {snap_time:.3f}s")
    print(f"  clone speedup:     {deep_time/clone_time:.1f}x")
    print(f"  snapshot speedup:  {deep_time/snap_time:.1f}x")

    print("\nSimulating 1000 random games...")

    start = time.time()
    winners = {0: 0, 1: 0}
    for _ in range(1000):
        state = GameState(2)
        turns = 0
        while not state.game_over and turns < 150:
            turns += 1
            state.roll_dice()
            valid = get_valid_moves(state)
            if not valid:
                bust_turn(state)
                continue
            apply_move(state, random.choice(valid))
            if random.random() < 0.5:
                stop_turn(state)
        if state.winner is not None:
            winners[state.winner] += 1

    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Games/second: {1000/elapsed:.0f}")
    print(f"  Winners: {winners}")