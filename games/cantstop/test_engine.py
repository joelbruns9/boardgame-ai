# test_engine.py
# Automated test suite for the Can't Stop game engine.
# Updated for integer player IDs (0, 1) and all_claimed tracking.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_possible_moves, get_valid_moves,
    apply_move, stop_turn, bust_turn, COLUMN_HEIGHTS
)

# ---- TEST FRAMEWORK ----
passed = 0
failed = 0

def check(test_name, condition, details=""):
    global passed, failed
    if condition:
        print(f"  PASS  {test_name}")
        passed += 1
    else:
        print(f"  FAIL  {test_name}")
        if details:
            print(f"        {details}")
        failed += 1

def section(name):
    print(f"\n{name}")
    print("-" * 50)


# ========================================
# DICE AND MOVE GENERATION
# ========================================
section("Dice and Move Generation")

state = GameState(2)
state.dice = [3, 3, 3, 3]
moves = get_possible_moves(state.dice)
check(
    "All same dice [3,3,3,3] → one unique move (6,6)",
    len(moves) == 1 and moves[0] == (6, 6),
    f"Got: {moves}"
)

state.dice = [1, 2, 3, 4]
moves = get_possible_moves(state.dice)
check(
    "Standard dice [1,2,3,4] → 3 unique moves",
    len(moves) == 3,
    f"Got: {moves}"
)

state.dice = [2, 6, 1, 5]
moves = get_possible_moves(state.dice)
all_sorted = all(m[0] <= m[1] for m in moves)
check(
    "All moves are sorted tuples (lower, higher)",
    all_sorted,
    f"Got: {moves}"
)

state = GameState(2)
state.dice = [3, 4, 3, 4]
moves = get_valid_moves(state)
check(
    "Double move (7,7) appears in valid moves for [3,4,3,4]",
    (7, 7) in moves,
    f"Got: {moves}"
)

# ========================================
# RUNNER LIMITS
# ========================================
section("Runner Limits")

state = GameState(2)
state.runners = {6: 1, 7: 1, 8: 1}
state.dice = [1, 2, 4, 5]
moves = get_valid_moves(state)
new_cols = set()
for move in moves:
    for col in move:
        if col not in state.runners:
            new_cols.add(col)
check(
    "With 3 runners, no new columns can be added",
    len(new_cols) == 0,
    f"New columns found: {new_cols}, moves: {moves}"
)

state = GameState(2)
state.runners = {6: 1, 7: 1}
state.dice = [1, 2, 3, 4]
moves = get_valid_moves(state)
check(
    "With 2 runners, moves are returned",
    len(moves) > 0,
    f"Got: {moves}"
)

# ========================================
# APPLY MOVE
# ========================================
section("Apply Move")

# Partial move gives 1 step, not 2
state = GameState(2)
state.claimed[1] = {8}
state.all_claimed = {8}
state.runners = {6: 1}
state.dice = [3, 3, 2, 6]

# Debug — verify state before calling get_valid_moves
print(f"\n  Debug: all_claimed={state.all_claimed}")
print(f"  Debug: runners={state.runners}")
print(f"  Debug: num_runners={len(state.runners)}")

moves = get_valid_moves(state)
print(f"  Debug: moves={moves}")

check(
    "Partial move (6,) generated when one column blocked",
    (6,) in moves and (6, 6) not in moves,
    f"Moves: {moves}"
)

# Apply partial move gives exactly 1 step
state2 = GameState(2)
state2.runners = {6: 1}
apply_move(state2, (6,))
check(
    "Partial move (6,) advances column 6 by exactly 1",
    state2.runners.get(6) == 2,
    f"Runners: {state2.runners}"
)

# Apply true double gives 2 steps
state3 = GameState(2)
apply_move(state3, (6, 6))
check(
    "True double (6,6) advances column 6 by 2",
    state3.runners.get(6) == 2,
    f"Runners: {state3.runners}"
)

# Partial move and true double produce different step counts
# when starting from step 1 (partial adds 1, double adds 2)
state4 = GameState(2)
state4.runners = {7: 5}  # already at step 5
snap = state4.save_snapshot()

apply_move(state4, (7,))  # partial — should give step 6
partial_result = state4.runners.get(7)
state4.restore_snapshot(snap)

apply_move(state4, (7, 7))  # true double — should give step 7
double_result = state4.runners.get(7)

check(
    "Partial (7,) gives 1 step, true double (7,7) gives 2 steps",
    partial_result == 6 and double_result == 7,
    f"Partial: {partial_result}, Double: {double_result}"
)

# ========================================
# STOP TURN
# ========================================
section("Stop Turn")

state = GameState(2)
state.runners = {7: 3, 9: 2}
stop_turn(state)
check(
    "Stop saves runner progress to permanent progress",
    state.progress[0].get(7) == 3 and state.progress[0].get(9) == 2,
    f"Progress: {state.progress[0]}"
)
check(
    "Stop clears runners",
    state.runners == {},
    f"Runners after stop: {state.runners}"
)

state = GameState(2)
state.runners = {2: 3}
stop_turn(state)
check(
    "Reaching column top claims the column",
    2 in state.claimed[0],
    f"Claimed: {state.claimed[0]}"
)
check(
    "Claimed column added to all_claimed",
    2 in state.all_claimed,
    f"all_claimed: {state.all_claimed}"
)
check(
    "Claimed column removed from progress",
    2 not in state.progress[0],
    f"Progress: {state.progress[0]}"
)

state = GameState(2)
state.claimed[0] = {4, 6}
state.all_claimed = {4, 6}
state.runners = {2: 3}
stop_turn(state)
check(
    "Claiming 3rd column wins the game",
    state.game_over and state.winner == 0,
    f"game_over: {state.game_over}, winner: {state.winner}"
)

state = GameState(2)
state.runners = {7: 4}
stop_turn(state)
state.active_player = 0
state.runners = {7: 3}
stop_turn(state)
check(
    "Progress accumulates correctly across turns",
    state.progress[0].get(7) == 7,
    f"Progress on col 7: {state.progress[0].get(7)}"
)

# ========================================
# BUST TURN
# ========================================
section("Bust Turn")

state = GameState(2)
state.progress[0] = {7: 5}
state.runners = {7: 3, 9: 2}
bust_turn(state)
check(
    "Bust clears runners",
    state.runners == {},
    f"Runners: {state.runners}"
)
check(
    "Bust preserves previously saved progress",
    state.progress[0].get(7) == 5,
    f"Progress: {state.progress[0]}"
)

state = GameState(2)
state.runners = {}
try:
    bust_turn(state)
    check("Bust with no runners handles gracefully", True)
except Exception as e:
    check("Bust with no runners handles gracefully", False, str(e))

# ========================================
# CLAIMED COLUMN EXCLUSION
# ========================================
section("Claimed Column Exclusion")

state = GameState(2)
state.claimed[1] = {7}
state.all_claimed = {7}
state.dice = [3, 4, 3, 4]
moves = get_valid_moves(state)
cols_in_moves = {col for move in moves for col in move}
check(
    "Claimed column 7 excluded from valid moves",
    7 not in cols_in_moves,
    f"Moves: {moves}, columns: {cols_in_moves}"
)

# ========================================
# ALL_CLAIMED MAINTENANCE
# ========================================
section("all_claimed Maintenance")

state = GameState(2)
state.runners = {2: 3}
stop_turn(state)
check(
    "all_claimed updated when column claimed via stop",
    2 in state.all_claimed,
    f"all_claimed: {state.all_claimed}"
)

state = GameState(2)
state.claimed[0] = {4, 6}
state.all_claimed = {4, 6}
state.claimed[1] = {8}
state.all_claimed.add(8)
check(
    "all_claimed reflects both players claimed columns",
    state.all_claimed == {4, 6, 8},
    f"all_claimed: {state.all_claimed}"
)

# ========================================
# CLONE AND SNAPSHOT
# ========================================
section("Clone and Snapshot")

state = GameState(2)
state.runners = {7: 3}
state.all_claimed = {4}
cloned = state.clone()
cloned.runners[7] = 99
cloned.all_claimed.add(6)
check(
    "Clone is independent from original — runners",
    state.runners[7] == 3,
    f"Original runners: {state.runners}"
)
check(
    "Clone is independent from original — all_claimed",
    6 not in state.all_claimed,
    f"Original all_claimed: {state.all_claimed}"
)

state = GameState(2)
state.runners = {7: 3, 9: 2}
state.progress[0] = {6: 5}
state.all_claimed = {4}
snap = state.save_snapshot()
state.runners = {5: 1}
state.progress[0] = {}
state.all_claimed = {4, 6}
state.restore_snapshot(snap)
check(
    "Snapshot restore recovers runners",
    state.runners == {7: 3, 9: 2},
    f"Runners after restore: {state.runners}"
)
check(
    "Snapshot restore recovers progress",
    state.progress[0].get(6) == 5,
    f"Progress after restore: {state.progress[0]}"
)
check(
    "Snapshot restore recovers all_claimed",
    state.all_claimed == {4},
    f"all_claimed after restore: {state.all_claimed}"
)

# ========================================
# PLAYER ROTATION
# ========================================
section("Player Rotation")

state = GameState(2)
state.runners = {7: 1}
stop_turn(state)
check(
    "Active player switches after stop",
    state.active_player == 1,
    f"Active player: {state.active_player}"
)

state = GameState(2)
bust_turn(state)
check(
    "Active player switches after bust",
    state.active_player == 1,
    f"Active player: {state.active_player}"
)

state = GameState(2)
bust_turn(state)
bust_turn(state)
check(
    "Player rotation wraps correctly 0→1→0",
    state.active_player == 0,
    f"Active player: {state.active_player}"
)

# ========================================
# HEURISTIC VALUE
# ========================================
section("Heuristic Value")

state = GameState(2)
state.claimed[0] = {4, 6}
state.all_claimed = {4, 6}
h = state.heuristic_value(0)
check(
    "Heuristic near-win position is high (>0.5)",
    h > 0.5,
    f"Heuristic: {h:.3f}"
)

state = GameState(2)
state.claimed[1] = {4, 6}
state.all_claimed = {4, 6}
h = state.heuristic_value(0)
check(
    "Heuristic losing position is low (<0.5)",
    h < 0.5,
    f"Heuristic: {h:.3f}"
)

state = GameState(2)
h = state.heuristic_value(0)
check(
    "Heuristic empty board is ~0.5",
    0.3 <= h <= 0.7,
    f"Heuristic: {h:.3f}"
)

# ========================================
# FULL GAME SIMULATION
# ========================================
section("Full Game Simulation")

import random
import time

def simulate_random_game():
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
    return state

start = time.time()
results = [simulate_random_game() for _ in range(100)]
elapsed = time.time() - start

all_finished = all(g.game_over for g in results)
all_valid_winner = all(g.winner in [0, 1] for g in results)
all_claimed_3 = all(len(g.claimed[g.winner]) >= 3 for g in results)

check(
    "100 random games all complete with game_over=True",
    all_finished,
    f"Unfinished: {sum(1 for g in results if not g.game_over)}"
)
check(
    "All winners are valid players (0 or 1)",
    all_valid_winner,
    f"Invalid winners: {[g.winner for g in results if g.winner not in [0,1]]}"
)
check(
    "All winners claimed 3+ columns",
    all_claimed_3,
    f"Under-claimed: {[len(g.claimed[g.winner]) for g in results if len(g.claimed[g.winner]) < 3]}"
)

no_overflow = all(
    all(
        g.progress[p].get(col, 0) <= COLUMN_HEIGHTS[col]
        for p in g.players
        for col in COLUMN_HEIGHTS
    )
    for g in results
)
check(
    "No progress exceeds column height",
    no_overflow
)

check(
    "all_claimed consistent with claimed sets in all games",
    all(
        g.all_claimed == g.claimed[0] | g.claimed[1]
        for g in results
    )
)

print(f"\n  ({elapsed:.2f}s for 100 games)")

section("Game Length Distribution")

import time
turn_counts = []
for _ in range(200):
    state = GameState(2)
    turns = 0
    while not state.game_over and turns < 500:
        turns += 1
        state.roll_dice()
        valid = get_valid_moves(state)
        if not valid:
            bust_turn(state)
            continue
        apply_move(state, random.choice(valid))
        if random.random() < 0.5:
            stop_turn(state)
    turn_counts.append(turns)

check(
    "Games finish well under 200 turns (max observed ~156)",
    max(turn_counts) < 200,
    f"Max turns: {max(turn_counts)}"
)
check(
    "Average game length is reasonable (50-120 turns)",
    50 <= sum(turn_counts)/len(turn_counts) <= 120,
    f"Average: {sum(turn_counts)/len(turn_counts):.1f}"
)

# ========================================
# SUMMARY
# ========================================
print(f"\n{'='*50}")
print(f"  Results: {passed} passed, {failed} failed")
if failed == 0:
    print(f"  All {passed} tests passed!")
else:
    print(f"  {failed} test(s) need attention")
print(f"{'='*50}\n")