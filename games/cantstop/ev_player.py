# ev_player.py
# EV-based player for Can't Stop.
# Updated for integer player IDs (0, 1) and clone() instead of deepcopy.

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_valid_moves, apply_move,
    stop_turn, bust_turn, COLUMN_HEIGHTS
)
from games.cantstop.ev_table import build_ev_table

_EV_TABLE = None


def _get_ev_table():
    global _EV_TABLE
    if _EV_TABLE is None:
        _EV_TABLE = build_ev_table()
    return _EV_TABLE


# ---- HELPER: WEIGHTED PROGRESS ----
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
    runners = state.runners

    def move_score(move):
        overlap = sum(1 for col in move if col in runners)
        centrality = sum(6 - abs(7 - col) for col in set(move))
        return (overlap, centrality)

    return max(valid_moves, key=move_score)


# ---- RANDOM PLAYER ----
def random_player(state):
    """Makes completely random decisions."""
    valid = get_valid_moves(state)
    if not valid:
        return None, "bust"
    move = random.choice(valid)
    decision = random.choice(["stop", "continue"])
    return move, decision


# ---- EV PLAYER ----
def ev_player(state, use_corrected=False):
    """
    Makes decisions based on EV break even threshold.

    Strategy:
    1. Pick the best valid move (favor existing runner columns)
    2. After moving, check if total weighted progress exceeds break even
    3. If yes → stop. If no → continue.

    use_corrected=False → Excel formula: avg_prog / (1 - prob_adv)
    use_corrected=True  → correct formula: (prob_adv * avg_prog) / (1 - prob_adv)
    """
    valid = get_valid_moves(state)
    if not valid:
        return None, "bust"

    move = choose_move(valid, state)

    # Use clone() instead of deepcopy — much faster
    state_copy = state.clone()
    apply_move(state_copy, move)

    # Look up EV for current runner columns if we have exactly 3
    runner_cols = tuple(sorted(state_copy.runners.keys()))
    entry = _get_ev_table().get(runner_cols) if len(runner_cols) == 3 else None

    if not entry:
        return move, "continue"

    total_progress = get_total_runner_progress(state_copy)
    prob_adv = entry["prob_adv"]
    avg_prog = entry["avg_prog"]

    if use_corrected:
        break_even = (prob_adv * avg_prog) / (1 - prob_adv)
    else:
        break_even = avg_prog / (1 - prob_adv)

    decision = "stop" if total_progress >= break_even else "continue"
    return move, decision


# ---- GAME SIMULATOR ----
def play_game(strategy_a, strategy_b):
    """
    Plays one full game between two strategies.
    strategy_a plays as player 0, strategy_b as player 1.
    Returns the winner (0 or 1) or None if timeout.
    """
    state = GameState(2)
    strategies = {0: strategy_a, 1: strategy_b}
    max_turns = 200  # 99th percentile is 133, max observed 156

    for _ in range(max_turns):
        if state.game_over:
            break

        player = state.active_player
        strategy = strategies[player]

        state.roll_dice()
        valid = get_valid_moves(state)

        if not valid:
            bust_turn(state)
            continue

        move, decision = strategy(state)

        if decision == "bust" or move is None:
            bust_turn(state)
            continue

        apply_move(state, move)

        if decision == "stop":
            stop_turn(state)

    return state.winner


# ---- TOURNAMENT ----
def run_tournament(strategy_a, strategy_b, name_a, name_b, games=5000):
    """Runs many games and reports win rates."""
    wins = {0: 0, 1: 0, None: 0}

    for _ in range(games):
        if random.random() < 0.5:
            winner = play_game(strategy_a, strategy_b)
            wins[0] += winner == 0
            wins[1] += winner == 1
            wins[None] += winner is None
        else:
            winner = play_game(strategy_b, strategy_a)
            # When sides are swapped, strategy_b is player 0
            wins[0] += winner == 1
            wins[1] += winner == 0
            wins[None] += winner is None

    total = games - wins[None]
    print(f"\n{'='*45}")
    print(f"  {name_a} vs {name_b}")
    print(f"  {games:,} games played")
    print(f"{'='*45}")
    print(f"  {name_a:<25} {wins[0]:>5} wins  ({100*wins[0]/total:.1f}%)")
    print(f"  {name_b:<25} {wins[1]:>5} wins  ({100*wins[1]/total:.1f}%)")


# ---- WATCH GAME ----
def watch_game(strategy, turns=15):
    """Watch a single player's decisions in detail."""
    state = GameState(2)
    print("Watching EV player decisions...\n")
    turn = 0

    while not state.game_over and turn < turns:
        player = state.active_player

        if player != 0:
            state.roll_dice()
            valid = get_valid_moves(state)
            if not valid:
                bust_turn(state)
            else:
                move, decision = strategy(state)
                if move:
                    apply_move(state, move)
                    if decision == "stop":
                        stop_turn(state)
                else:
                    bust_turn(state)
            continue

        turn += 1
        state.roll_dice()
        valid = get_valid_moves(state)

        if not valid:
            print(f"Turn {turn} | Dice: {state.dice} | BUST")
            bust_turn(state)
            continue

        state_copy = state.clone()
        move, decision = strategy(state_copy)

        display = state.clone()
        apply_move(display, move)
        progress = get_total_runner_progress(display)
        runner_cols = tuple(sorted(display.runners.keys()))
        entry = _get_ev_table().get(runner_cols, {})

        print(f"Turn {turn} | Dice: {state.dice}")
        print(f"  Valid moves:        {valid}")
        print(f"  Chosen move:        {move}")
        print(f"  Runners after move: {display.runners}")
        print(f"  Progress at risk:   {progress:.3f}")
        if entry:
            be = entry['avg_prog'] / (1 - entry['prob_adv'])
            print(f"  Break even:         {be:.3f}")
        print(f"  Decision:           {decision}")
        print()

        apply_move(state, move)
        if decision == "stop":
            stop_turn(state)


# ---- MAIN ----
if __name__ == "__main__":
    print("Building EV table...")

    rand    = lambda s: random_player(s)
    excel   = lambda s: ev_player(s, use_corrected=False)
    correct = lambda s: ev_player(s, use_corrected=True)

    watch_game(correct, turns=10)

    print("\nRunning tournaments (5,000 games each)...")

    run_tournament(rand,    excel,   "Random",       "Excel EV")
    run_tournament(rand,    correct, "Random",       "Corrected EV")
    run_tournament(excel,   correct, "Excel EV",     "Corrected EV")

    print("\nDone!")