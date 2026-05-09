# engine.py
# The brain of our Can't Stop game engine.

# ---- IMPORTS ----
# random lets us simulate dice rolls
import random

# ---- THE BOARD ----
# Each column has a different number of steps to claim it.
# Column 7 is longest (hardest), 2 and 12 are shortest (easiest).
COLUMN_HEIGHTS = {
    2: 3,  3: 5,  4: 7,  5: 9,  6: 11,
    7: 13, 8: 11, 9: 9, 10: 7, 11: 5, 12: 3
}

# Number of columns a player must claim to win
COLUMNS_TO_WIN = 3

# Maximum runners a player can have active at once
MAX_RUNNERS = 3


# ---- GAME STATE ----
# This class holds everything about a game at one moment in time.
# Think of it like a photograph of the board.

class GameState:
    def __init__(self, player_ids):
        """
        Set up a brand new game.
        player_ids: a list of player identifiers e.g. ["Alice", "Bob"]
        """

        # Who is playing?
        self.players = player_ids

        # Whose turn is it? Start with the first player.
        self.active_player = player_ids[0]

        # Which columns has each player fully claimed?
        # e.g. {"Alice": [7, 9], "Bob": [2]}
        self.claimed = {p: [] for p in player_ids}

        # Where is each player's permanent progress on each column?
        # e.g. {"Alice": {7: 5, 9: 3}, "Bob": {2: 2}}
        # This is progress that is SAVED after stopping
        self.progress = {p: {} for p in player_ids}

        # Where are the active runners THIS turn?
        # Runners are temporary — lost if player busts
        # e.g. {7: 6, 9: 4}  means runner on col 7 at step 6
        self.runners = {}

        # What did the dice show this roll?
        self.dice = []

        # Is the game over?
        self.game_over = False
        self.winner = None

    def roll_dice(self):
        """Roll 4 dice. Returns the values."""
        self.dice = [random.randint(1, 6) for _ in range(4)]
        return self.dice

    def get_current_progress(self, player, column):
        """
        How far is a player on a given column RIGHT NOW?
        Combines their saved progress + any active runner.
        """
        saved = self.progress[player].get(column, 0)
        runner = self.runners.get(column, 0)
        return saved + runner

    def __repr__(self):
        """
        This is what prints when you do print(state).
        Like a scoreboard snapshot.
        """
        lines = [
            f"\n=== Can't Stop ===",
            f"Active player: {self.active_player}",
            f"Dice: {self.dice}",
            f"Runners: {self.runners}",
        ]
        for p in self.players:
            lines.append(f"{p} - claimed: {self.claimed[p]} | progress: {self.progress[p]}")
        if self.game_over:
            lines.append(f"GAME OVER - Winner: {self.winner}")
        return "\n".join(lines)


# ---- MOVE CALCULATION ----
def get_possible_moves(dice):
    """
    Given 4 dice, return all unique column pair combinations.
    Same as before — the three ways to split 4 dice into 2 pairs.
    """
    splits = [
        (dice[0] + dice[1], dice[2] + dice[3]),
        (dice[0] + dice[2], dice[1] + dice[3]),
        (dice[0] + dice[3], dice[1] + dice[2]),
    ]
    return list(set(splits))


# ---- VALID MOVES ----
def get_valid_moves(state):
    """
    Not all possible moves are valid.
    A move is invalid if:
    - The column is already claimed by someone
    - Adding a runner would exceed MAX_RUNNERS active runners
    - The runner is already at the top of the column

    Returns a list of valid (col_a, col_b) pairs.
    Think of this as filtering the rulebook:
    "you CAN pair these dice, but are you ALLOWED to move there?"
    """
    possible = get_possible_moves(state.dice)
    player = state.active_player
    valid = []

    for col_a, col_b in possible:
        move_valid = False

        for col in [col_a, col_b]:
            # Skip columns already fully claimed
            if any(col in state.claimed[p] for p in state.players):
                continue

            # Current position on this column
            current_pos = state.get_current_progress(player, col)

            # Already at the top?
            if current_pos >= COLUMN_HEIGHTS[col]:
                continue

            # Would this add a NEW runner beyond our limit?
            new_runner = col not in state.runners
            active_runners = len(state.runners)
            if new_runner and active_runners >= MAX_RUNNERS:
                continue

            move_valid = True

        if move_valid:
            valid.append((col_a, col_b))

    return valid


# ---- APPLY A MOVE ----
def apply_move(state, move):
    """
    Actually move the runners based on the chosen column pair.
    move: a tuple like (7, 9) meaning advance on columns 7 and 9.
    """
    player = state.active_player

    for col in move:
        # Skip claimed columns
        if any(col in state.claimed[p] for p in state.players):
            continue

        # Skip if already at top
        current_pos = state.get_current_progress(player, col)
        if current_pos >= COLUMN_HEIGHTS[col]:
            continue

        # Skip if would exceed runner limit
        if col not in state.runners and len(state.runners) >= MAX_RUNNERS:
            continue

        # Advance the runner by 1
        state.runners[col] = state.runners.get(col, 0) + 1

    return state


# ---- STOP: SAVE PROGRESS ----
def stop_turn(state):
    """
    Player chose to stop. Save runner positions as permanent progress.
    Check if any columns are now fully claimed.
    Then pass to next player.
    """
    player = state.active_player

    for col, runner_pos in state.runners.items():
        # Add runner progress to saved progress
        saved = state.progress[player].get(col, 0)
        new_pos = saved + runner_pos
        state.progress[player][col] = new_pos

        # Has this player reached the top of this column?
        if new_pos >= COLUMN_HEIGHTS[col]:
            state.claimed[player].append(col)
            # Remove from progress — it's fully claimed now
            del state.progress[player][col]

            # Check win condition
            if len(state.claimed[player]) >= COLUMNS_TO_WIN:
                state.game_over = True
                state.winner = player

    # Clear runners for next turn
    state.runners = {}

    # Pass to next player (if game not over)
    if not state.game_over:
        _next_player(state)

    return state


# ---- BUST: LOSE PROGRESS ----
def bust_turn(state):
    """
    Player busted — no valid moves available.
    Runners are lost. Progress is NOT saved.
    Pass to next player.
    """
    # Just clear runners — no progress saved
    state.runners = {}
    _next_player(state)
    return state


# ---- NEXT PLAYER ----
def _next_player(state):
    """Rotate to the next player in order."""
    current_index = state.players.index(state.active_player)
    next_index = (current_index + 1) % len(state.players)
    state.active_player = state.players[next_index]


# ---- TEST: PLAY A RANDOM GAME ----
# This simulates a full game with random decisions
# to verify all the rules work correctly together.

if __name__ == "__main__":
    print("Simulating a random game of Can't Stop...\n")

    state = GameState(["Alice", "Bob"])
    turn_count = 0
    max_turns = 200  # safety limit

    while not state.game_over and turn_count < max_turns:
        turn_count += 1
        player = state.active_player

        # Roll dice
        dice = state.roll_dice()

        # Get valid moves
        valid = get_valid_moves(state)

        if not valid:
            # No valid moves — bust!
            print(f"Turn {turn_count}: {player} rolled {dice} — BUST!")
            bust_turn(state)
        else:
            # Pick a random valid move
            move = random.choice(valid)
            apply_move(state, move)

            # Randomly decide to stop or continue (50/50)
            if random.random() < 0.5 or not get_valid_moves(state):
                print(f"Turn {turn_count}: {player} rolled {dice}, moved {move}, STOPS")
                stop_turn(state)
            else:
                print(f"Turn {turn_count}: {player} rolled {dice}, moved {move}, continues...")

    print(state)