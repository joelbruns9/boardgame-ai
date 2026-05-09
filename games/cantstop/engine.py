# engine.py
# Can't Stop game engine with efficient state cloning.
# Clone/restore pattern replaces deepcopy for performance.

import random

# ---- BOARD CONSTANTS ----
COLUMN_HEIGHTS = {
    2: 3,  3: 5,  4: 7,  5: 9,  6: 11,
    7: 13, 8: 11, 9: 9, 10: 7, 11: 5, 12: 3
}

COLUMNS_TO_WIN = 3
MAX_RUNNERS = 3


# ---- GAME STATE ----
class GameState:
    def __init__(self, player_ids):
        self.players = player_ids
        self.active_player = player_ids[0]

        # Permanent progress saved after stopping
        # {player: {column: steps}}
        self.claimed = {p: set() for p in player_ids}
        self.progress = {p: {} for p in player_ids}

        # Temporary runners this turn
        # {column: steps_this_turn}
        self.runners = {}

        # Current dice roll
        self.dice = []

        self.game_over = False
        self.winner = None

    def roll_dice(self):
        """Roll 4 fresh dice. Only call at start of a new turn."""
        self.dice = [random.randint(1, 6) for _ in range(4)]
        return self.dice

    def get_current_progress(self, player, column):
        saved = self.progress[player].get(column, 0)
        # Only add runner if this player is currently active
        # Runners belong to the active player only
        runner = self.runners.get(column, 0) if player == self.active_player else 0
        return saved + runner

    def save_snapshot(self):
        """
        Save a lightweight snapshot of mutable state.
        Much faster than deepcopy — only saves what can change.

        Think of this as writing down the current board position
        on a notepad so you can restore it after exploring a move.
        """
        return {
            "active_player": self.active_player,
            "claimed": {p: set(cols) for p, cols in self.claimed.items()},
            "progress": {p: dict(prog) for p, prog in self.progress.items()},
            "runners": dict(self.runners),
            "dice": list(self.dice),
            "game_over": self.game_over,
            "winner": self.winner,
        }

    def restore_snapshot(self, snapshot):
        """
        Restore state from a snapshot.
        Undoes any moves made since the snapshot was taken.
        """
        self.active_player = snapshot["active_player"]
        self.claimed = {p: set(cols) for p, cols in snapshot["claimed"].items()}
        self.progress = {p: dict(prog) for p, prog in snapshot["progress"].items()}
        self.runners = dict(snapshot["runners"])
        self.dice = list(snapshot["dice"])
        self.game_over = snapshot["game_over"]
        self.winner = snapshot["winner"]

    def clone(self):
        """
        Create a full independent copy of the game state.
        Used when we need a truly separate state (e.g. for rollouts
        that will play out to completion independently).

        Faster than deepcopy because we know exactly what to copy.
        """
        new = GameState(list(self.players))
        new.active_player = self.active_player
        new.claimed = {p: set(cols) for p, cols in self.claimed.items()}
        new.progress = {p: dict(prog) for p, prog in self.progress.items()}
        new.runners = dict(self.runners)
        new.dice = list(self.dice)
        new.game_over = self.game_over
        new.winner = self.winner
        return new

    def __repr__(self):
        lines = [
            f"\n=== Can't Stop ===",
            f"Active player: {self.active_player}",
            f"Dice: {self.dice}",
            f"Runners: {self.runners}",
        ]
        for p in self.players:
            lines.append(
                f"{p} - claimed: {sorted(self.claimed[p])} "
                f"| progress: {self.progress[p]}"
            )
        if self.game_over:
            lines.append(f"GAME OVER - Winner: {self.winner}")
        return "\n".join(lines)


# ---- MOVE CALCULATION ----
def get_possible_moves(dice):
    splits = [
        tuple(sorted((dice[0] + dice[1], dice[2] + dice[3]))),
        tuple(sorted((dice[0] + dice[2], dice[1] + dice[3]))),
        tuple(sorted((dice[0] + dice[3], dice[1] + dice[2]))),
    ]
    return list(set(splits))


def get_valid_moves(state):
    """
    Filter possible moves to only legal ones.
    Handles double moves like (7,7) correctly —
    same column twice = 1 runner, 2 steps.
    """
    possible = get_possible_moves(state.dice)
    player = state.active_player
    valid = []

    # All claimed columns across all players
    all_claimed = set()
    for p in state.players:
        all_claimed.update(state.claimed[p])

    for col_a, col_b in possible:
        # Get unique columns in this move
        unique_cols = set([col_a, col_b])

        # Check each unique column is usable
        usable_cols = {
            col for col in unique_cols
            if _can_use_column(state, player, col, all_claimed)
        }

        if not usable_cols:
            continue

        # How many NEW runners would this move need?
        # Only count unique columns not already having a runner
        new_runners_needed = len(
            usable_cols - set(state.runners.keys())
        )

        current_runners = len(state.runners)

        if current_runners + new_runners_needed > MAX_RUNNERS:
            # Can't place enough new runners for the full move
            # But maybe we can still use columns we already have runners on
            partial_cols = usable_cols & set(state.runners.keys())
            if partial_cols:
                # Add partial moves using only existing runner columns
                for col in partial_cols:
                    valid.append(tuple(sorted((col, col))))
            continue

        valid.append(tuple(sorted((col_a, col_b))))

    # Normalize and deduplicate
    return list(set(valid))


def can_use_column_check(state, col, all_claimed):
    """Can this column accept a runner?"""
    if col in all_claimed:
        return False
    player = state.active_player
    current = state.get_current_progress(player, col)
    return current < COLUMN_HEIGHTS[col]


def _can_use_column(state, player, col, all_claimed):
    """Internal helper — can this player use this column?"""
    if col in all_claimed:
        return False
    current = state.get_current_progress(player, col)
    return current < COLUMN_HEIGHTS[col]


def apply_move(state, move):
    """
    Advance runners based on chosen column pair.
    Double moves like (7,7) advance one column by 2 steps
    but still only use 1 runner.
    Modifies state in place.
    """
    player = state.active_player

    all_claimed = set()
    for p in state.players:
        all_claimed.update(state.claimed[p])

    col_a, col_b = move

    if col_a == col_b:
        # Double move — same column twice, advance by 2 steps
        col = col_a
        if not _can_use_column(state, player, col, all_claimed):
            return state
        if col not in state.runners and len(state.runners) >= MAX_RUNNERS:
            return state
        # Advance by 2 but cap at column height
        current = state.get_current_progress(player, col)
        steps = min(2, COLUMN_HEIGHTS[col] - current)
        state.runners[col] = state.runners.get(col, 0) + steps
    else:
        # Normal move — two different columns, each advances by 1
        for col in [col_a, col_b]:
            if not _can_use_column(state, player, col, all_claimed):
                continue
            if col not in state.runners and len(state.runners) >= MAX_RUNNERS:
                continue
            state.runners[col] = state.runners.get(col, 0) + 1

    return state


def stop_turn(state):
    player = state.active_player

    # Build all claimed columns across all players
    all_claimed = set()
    for p in state.players:
        all_claimed.update(state.claimed[p])

    for col, runner_steps in list(state.runners.items()):
        saved = state.progress[player].get(col, 0)
        new_pos = saved + runner_steps

        if new_pos >= COLUMN_HEIGHTS[col] and col not in all_claimed:
            # Column completed — claim it, no need to save progress
            state.claimed[player].add(col)
            state.progress[player].pop(col, None)

            if len(state.claimed[player]) >= COLUMNS_TO_WIN:
                state.game_over = True
                state.winner = player
        else:
            # Column not yet complete — save progress
            state.progress[player][col] = new_pos

    state.runners = {}

    if not state.game_over:
        _next_player(state)

    return state


def bust_turn(state):
    """Lose all runner progress. Pass to next player."""
    state.runners = {}
    _next_player(state)
    return state


def _next_player(state):
    """Rotate to next player."""
    idx = state.players.index(state.active_player)
    state.active_player = state.players[(idx + 1) % len(state.players)]


# ---- TEST ----
if __name__ == "__main__":
    print("Testing clone vs snapshot performance...\n")

    import time

    state = GameState(["Alice", "Bob"])

    # Benchmark clone
    start = time.time()
    for _ in range(10000):
        cloned = state.clone()
    clone_time = time.time() - start

    # Benchmark snapshot/restore
    start = time.time()
    for _ in range(10000):
        snap = state.save_snapshot()
        state.restore_snapshot(snap)
    snap_time = time.time() - start

    # Benchmark deepcopy for comparison
    import copy
    start = time.time()
    for _ in range(10000):
        copied = copy.deepcopy(state)
    deep_time = time.time() - start

    print(f"10,000 iterations:")
    print(f"  deepcopy:          {deep_time:.3f} sec")
    print(f"  clone():           {clone_time:.3f} sec")
    print(f"  snapshot/restore:  {snap_time:.3f} sec")
    print(f"\nClone speedup over deepcopy: {deep_time/clone_time:.1f}x")
    print(f"Snapshot speedup over deepcopy: {deep_time/snap_time:.1f}x")

    # Test a random game still works
    print("\nSimulating random game to verify engine still works...")
    state = GameState(["Alice", "Bob"])
    turns = 0
    while not state.game_over and turns < 200:
        turns += 1
        state.roll_dice()
        valid = get_valid_moves(state)
        if not valid:
            bust_turn(state)
        else:
            import random
            apply_move(state, random.choice(valid))
            if random.random() < 0.5:
                stop_turn(state)
    print(state)