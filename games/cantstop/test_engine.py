# test_engine.py
# Automated test suite for the Can't Stop game engine.
# Run with: python games/cantstop/test_engine.py
#
# Each test checks one specific behavior.
# If all tests pass you'll see: "All X tests passed!"
# If something breaks you'll see exactly which test failed and why.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_possible_moves, get_valid_moves,
    apply_move, stop_turn, bust_turn, COLUMN_HEIGHTS
)

# ---- TEST FRAMEWORK ----
# Simple pass/fail tracker — no external libraries needed

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

# All same dice should give exactly one unique move
state = GameState(["A", "B"])
state.dice = [3, 3, 3, 3]  # all pairs sum to 6
moves = get_possible_moves(state.dice)
check(
    "All same dice [3,3,3,3] → one unique move (6,6)",
    len(moves) == 1 and moves[0] == (6, 6),
    f"Got: {moves}"
)

# Standard dice should give 3 unique moves
state.dice = [1, 2, 3, 4]
moves = get_possible_moves(state.dice)
check(
    "Standard dice [1,2,3,4] → 3 unique moves",
    len(moves) == 3,
    f"Got: {moves}"
)

# All moves should be sorted tuples (lower, higher)
state.dice = [2, 6, 1, 5]
moves = get_possible_moves(state.dice)
all_sorted = all(m[0] <= m[1] for m in moves)
check(
    "All moves are sorted tuples (lower, higher)",
    all_sorted,
    f"Got: {moves}"
)

# Double move should appear in valid moves
state = GameState(["A", "B"])
state.dice = [3, 4, 3, 4]  # pairs: (7,7), (6,8)
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

# With 3 runners active, can only use existing runner columns
state = GameState(["A", "B"])
state.runners = {6: 1, 7: 1, 8: 1}  # 3 runners placed
state.dice = [1, 2, 4, 5]  # possible: (3,9), (5,7), (6,6)
moves = get_valid_moves(state)
# Only (6,6) and (7,x) moves should be valid — no new columns
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

# With 2 runners, one new column allowed
state = GameState(["A", "B"])
state.runners = {6: 1, 7: 1}  # 2 runners
state.dice = [1, 2, 3, 4]  # various options
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

# Normal move advances two columns by 1 each
state = GameState(["A", "B"])
state.dice = [3, 4, 3, 4]
apply_move(state, (6, 8))
check(
    "Normal move (6,8) advances both columns by 1",
    state.runners.get(6) == 1 and state.runners.get(8) == 1,
    f"Runners: {state.runners}"
)

# Double move advances one column by 2
state = GameState(["A", "B"])
state.dice = [3, 4, 3, 4]
apply_move(state, (7, 7))
check(
    "Double move (7,7) advances column 7 by 2 steps",
    state.runners.get(7) == 2,
    f"Runners: {state.runners}"
)
check(
    "Double move (7,7) uses only 1 runner",
    len(state.runners) == 1,
    f"Runner count: {len(state.runners)}"
)

# Double move caps at column height
state = GameState(["A", "B"])
state.progress["A"][2] = 2  # column 2 height is 3, already at 2
state.dice = [1, 1, 1, 1]  # double move on column 2
apply_move(state, (2, 2))
total = state.progress["A"].get(2, 0) + state.runners.get(2, 0)
check(
    "Double move capped at column height (col 2, height 3)",
    total <= COLUMN_HEIGHTS[2],
    f"Total progress: {total}, height: {COLUMN_HEIGHTS[2]}"
)

# ========================================
# STOP TURN
# ========================================
section("Stop Turn")

# Stopping saves runner progress permanently
state = GameState(["A", "B"])
state.runners = {7: 3, 9: 2}
stop_turn(state)
check(
    "Stop saves runner progress to permanent progress",
    state.progress["A"].get(7) == 3 and state.progress["A"].get(9) == 2,
    f"Progress: {state.progress['A']}"
)
check(
    "Stop clears runners",
    state.runners == {},
    f"Runners after stop: {state.runners}"
)

# Stopping at top of column claims it
state = GameState(["A", "B"])
state.runners = {2: 3}  # column 2 height is 3 — exactly at top
stop_turn(state)
check(
    "Reaching column top claims the column",
    2 in state.claimed["A"],
    f"Claimed: {state.claimed['A']}"
)
check(
    "Claimed column removed from progress",
    2 not in state.progress["A"],
    f"Progress: {state.progress['A']}"
)

# Claiming 3 columns wins the game
state = GameState(["A", "B"])
state.claimed["A"] = {4, 6}  # already claimed 2
state.runners = {2: 3}  # claiming column 2 gives 3 total
stop_turn(state)
check(
    "Claiming 3rd column wins the game",
    state.game_over and state.winner == "A",
    f"game_over: {state.game_over}, winner: {state.winner}"
)

# Progress accumulates correctly across turns
state = GameState(["A", "B"])
state.runners = {7: 4}
stop_turn(state)  # saves 4 steps on col 7
# Now it's B's turn — simulate B's turn ending
state.runners = {7: 3}
# Manually switch back to A
state.active_player = "A"
state.runners = {7: 3}
stop_turn(state)  # should add 3 more to existing 4
check(
    "Progress accumulates correctly across turns",
    state.progress["A"].get(7) == 7,
    f"Progress on col 7: {state.progress['A'].get(7)}"
)

# ========================================
# BUST TURN
# ========================================
section("Bust Turn")

# Bust loses runners but preserves saved progress
state = GameState(["A", "B"])
state.progress["A"] = {7: 5}  # saved from previous turn
state.runners = {7: 3, 9: 2}  # this turn's runners — lost on bust
bust_turn(state)
check(
    "Bust clears runners",
    state.runners == {},
    f"Runners: {state.runners}"
)
check(
    "Bust preserves previously saved progress",
    state.progress["A"].get(7) == 5,
    f"Progress: {state.progress['A']}"
)

# Bust with no runners
state = GameState(["A", "B"])
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

# Opponent's claimed column shouldn't appear in valid moves
state = GameState(["A", "B"])
state.claimed["B"] = {7}  # B claimed column 7
state.dice = [3, 4, 3, 4]  # would normally give (7,7) and (6,8)
moves = get_valid_moves(state)
cols_in_moves = {col for move in moves for col in move}
check(
    "Claimed column 7 excluded from valid moves",
    7 not in cols_in_moves,
    f"Moves: {moves}, columns: {cols_in_moves}"
)

# ========================================
# CLONE AND SNAPSHOT
# ========================================
section("Clone and Snapshot")

# Clone is independent — modifying clone doesn't affect original
state = GameState(["A", "B"])
state.runners = {7: 3}
cloned = state.clone()
cloned.runners[7] = 99
check(
    "Clone is independent from original",
    state.runners[7] == 3,
    f"Original runners: {state.runners}"
)

# Snapshot round-trip restores exactly
state = GameState(["A", "B"])
state.runners = {7: 3, 9: 2}
state.progress["A"] = {6: 5}
snap = state.save_snapshot()
state.runners = {5: 1}  # corrupt the state
state.progress["A"] = {}
state.restore_snapshot(snap)
check(
    "Snapshot restore recovers runners",
    state.runners == {7: 3, 9: 2},
    f"Runners after restore: {state.runners}"
)
check(
    "Snapshot restore recovers progress",
    state.progress["A"].get(6) == 5,
    f"Progress after restore: {state.progress['A']}"
)

# ========================================
# PLAYER ROTATION
# ========================================
section("Player Rotation")

# After stop, active player switches
state = GameState(["A", "B"])
state.runners = {7: 1}
stop_turn(state)
check(
    "Active player switches after stop",
    state.active_player == "B",
    f"Active player: {state.active_player}"
)

# After bust, active player switches
state = GameState(["A", "B"])
bust_turn(state)
check(
    "Active player switches after bust",
    state.active_player == "B",
    f"Active player: {state.active_player}"
)

# Rotation wraps correctly A→B→A
state = GameState(["A", "B"])
bust_turn(state)  # A→B
bust_turn(state)  # B→A
check(
    "Player rotation wraps correctly A→B→A",
    state.active_player == "A",
    f"Active player: {state.active_player}"
)

# ========================================
# FULL GAME SIMULATION
# ========================================
section("Full Game Simulation")

import random

def simulate_random_game():
    state = GameState(["A", "B"])
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
    return state

# Run 100 random games — all should complete with a winner
results = [simulate_random_game() for _ in range(100)]
all_finished = all(g.game_over for g in results)
all_valid_winner = all(g.winner in ["A", "B"] for g in results)
all_claimed_3 = all(len(g.claimed[g.winner]) >= 3 for g in results)

check(
    "100 random games all complete with game_over=True",
    all_finished,
    f"Unfinished: {sum(1 for g in results if not g.game_over)}"
)
check(
    "All winners are valid players",
    all_valid_winner,
    f"Invalid winners: {[g.winner for g in results if g.winner not in ['A','B']]}"
)
check(
    "All winners claimed exactly 3+ columns",
    all_claimed_3,
    f"Under-claimed: {[len(g.claimed[g.winner]) for g in results if len(g.claimed[g.winner]) < 3]}"
)

# No runner should exceed column height
no_overflow = all(
    all(
        g.progress[p].get(col, 0) <= COLUMN_HEIGHTS[col]
        for p in g.players
        for col in COLUMN_HEIGHTS
    )
    for g in results
)
check(
    "No progress exceeds column height in any game",
    no_overflow
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