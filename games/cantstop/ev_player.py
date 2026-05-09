# ev_player.py
# Tests three player strategies against each other

import random
import sys
import os
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_valid_moves, apply_move,
    stop_turn, bust_turn, COLUMN_HEIGHTS
)
from games.cantstop.ev_table import build_ev_table

EV_TABLE = build_ev_table()


# ---- HELPER: CURRENT RUNNER COLUMNS ----
def get_active_combo(state):
    """
    Returns the EV table key for the current runner columns.
    Uses actual runner columns if we have them,
    otherwise returns None (no lookup possible yet).
    """
    runners = list(state.runners.keys())
    if len(runners) == 3:
        return tuple(sorted(runners))
    return None


def get_total_runner_progress(state):
    """
    Total WEIGHTED progress across active runner columns.
    Measured in fraction of column completed — same units as break even.
    
    e.g. runner at step 4 on column 7 (height 13) = 4/13 = 0.308
    """
    player = state.active_player
    total = 0
    for col, runner_steps in state.runners.items():
        saved = state.progress[player].get(col, 0)
        total_steps = saved + runner_steps
        total += total_steps / COLUMN_HEIGHTS[col]
    return total


# ---- CHOOSE BEST MOVE ----
def choose_move(valid_moves, state):
    """
    Pick the move that advances on columns we already have
    runners on — maximizing progress on existing columns
    before opening new ones.
    """
    runners = set(state.runners.keys())

    def move_score(move):
        # Count how many columns in this move overlap with existing runners
        overlap = sum(1 for col in move if col in runners)
        # Also prefer middle columns (closer to 7)
        centrality = sum(6 - abs(7 - col) for col in move)
        return (overlap, centrality)

    return max(valid_moves, key=move_score)


# ---- PLAYER STRATEGIES ----

def random_player(state):
    """
    Makes completely random decisions.
    Returns (move, decision) where decision is stop/continue/bust.
    """
    valid = get_valid_moves(state)
    if not valid:
        return None, "bust"

    move = random.choice(valid)
    decision = random.choice(["stop", "continue"])
    return move, decision


def ev_player(state, use_corrected=False):
    """
    Makes decisions based on EV break even threshold.

    Strategy:
    1. Pick the best valid move (favor existing runner columns)
    2. After moving, check if total progress exceeds break even
    3. If yes → stop. If no → continue.

    use_corrected=False → your Excel formula
    use_corrected=True  → mathematically correct formula
    """
    valid = get_valid_moves(state)
    if not valid:
        return None, "bust"

    # Pick best move
    move = choose_move(valid, state)

    # Simulate applying this move on a copy to check progress
    state_copy = copy.deepcopy(state)
    apply_move(state_copy, move)

    # What columns do we now have runners on?
    runner_cols = tuple(sorted(state_copy.runners.keys()))

    # Look up EV for these columns if we have exactly 3
    if len(runner_cols) == 3:
        entry = EV_TABLE.get(runner_cols)
    else:
        entry = None

    if not entry:
        # Can't look up EV — just continue
        return move, "continue"

    # Total progress at risk if we bust
    total_progress = get_total_runner_progress(state_copy)

    # Calculate break even threshold
    prob_adv = entry["prob_adv"]
    avg_prog = entry["avg_prog"]

    if use_corrected:
        break_even = (prob_adv * avg_prog) / (1 - prob_adv)
    else:
        break_even = avg_prog / (1 - prob_adv)

    # Stop if we've exceeded break even
    decision = "stop" if total_progress >= break_even else "continue"

    return move, decision


# ---- GAME SIMULATOR ----

def play_game(strategy_a, strategy_b):
    """
    Plays one full game between two strategies.
    Handles the full turn loop including continue decisions.
    Returns the winner ("A" or "B").
    """
    state = GameState(["A", "B"])
    strategies = {"A": strategy_a, "B": strategy_b}
    max_turns = 1000

    for _ in range(max_turns):
        if state.game_over:
            break

        player = state.active_player
        strategy = strategies[player]

        # Roll dice
        state.roll_dice()
        valid = get_valid_moves(state)

        if not valid:
            bust_turn(state)
            continue

        # Get decision
        move, decision = strategy(state)

        if decision == "bust" or move is None:
            bust_turn(state)
            continue

        # Apply the move
        apply_move(state, move)

        if decision == "stop":
            stop_turn(state)
        # If continue — loop back, same player rolls again

    return state.winner


# ---- TOURNAMENT ----

def run_tournament(strategy_a, strategy_b, name_a, name_b, games=5000):
    """Runs many games and reports win rates."""
    wins = {"A": 0, "B": 0, None: 0}

    for _ in range(games):
        if random.random() < 0.5:
            winner = play_game(strategy_a, strategy_b)
            wins["A"] += winner == "A"
            wins["B"] += winner == "B"
            wins[None] += winner is None
        else:
            winner = play_game(strategy_b, strategy_a)
            wins["A"] += winner == "B"
            wins["B"] += winner == "A"
            wins[None] += winner is None

    total = games - wins[None]
    print(f"\n{'='*45}")
    print(f"  {name_a} vs {name_b}")
    print(f"  {games:,} games played")
    print(f"{'='*45}")
    print(f"  {name_a:<25} {wins['A']:>5} wins  ({100*wins['A']/total:.1f}%)")
    print(f"  {name_b:<25} {wins['B']:>5} wins  ({100*wins['B']/total:.1f}%)")


# ---- DIAGNOSTIC MODE ----

def watch_game(strategy, turns=15):
    """Watch a single player's decisions in detail."""
    state = GameState(["A", "B"])

    print("Watching EV player decisions...\n")
    turn = 0

    while not state.game_over and turn < turns:
        player = state.active_player
        if player != "A":
            # Skip B's turns for clarity
            state.roll_dice()
            valid = get_valid_moves(state)
            if not valid:
                bust_turn(state)
            else:
                move, decision = strategy(state)
                apply_move(state, move)
                if decision == "stop":
                    stop_turn(state)
            continue

        turn += 1
        state.roll_dice()
        valid = get_valid_moves(state)

        if not valid:
            print(f"Turn {turn} | Dice: {state.dice} | BUST")
            bust_turn(state)
            continue

        # Peek at what EV player will decide
        state_copy = copy.deepcopy(state)
        move, decision = strategy(state_copy)

        # Apply for display
        display_state = copy.deepcopy(state)
        apply_move(display_state, move)
        progress = get_total_runner_progress(display_state)
        runner_cols = tuple(sorted(display_state.runners.keys()))
        entry = EV_TABLE.get(runner_cols, {})

        print(f"Turn {turn} | Dice: {state.dice}")
        print(f"  Valid moves:  {valid}")
        print(f"  Chosen move:  {move}")
        print(f"  Runners after move: {display_state.runners}")
        print(f"  Progress at risk:   {progress:.1f}")
        if entry:
            be = entry['avg_prog'] / (1 - entry['prob_adv'])
            print(f"  Break even:   {be:.3f}")
        print(f"  Decision:     {decision}")
        print()

        # Actually apply to real state
        apply_move(state, move)
        if decision == "stop":
            stop_turn(state)


# ---- MAIN ----
if __name__ == "__main__":
    print("Building EV table...")

    rand    = lambda s: random_player(s)
    excel   = lambda s: ev_player(s, use_corrected=False)
    correct = lambda s: ev_player(s, use_corrected=True)

    # Watch a game first to verify logic
    watch_game(correct, turns=10)

    print("\nRunning tournaments (5,000 games each)...")

    run_tournament(rand,   excel,   "Random",    "Excel EV")
    run_tournament(rand,   correct, "Random",    "Corrected EV")
    run_tournament(excel,  correct, "Excel EV",  "Corrected EV")

    print("\nDone!")