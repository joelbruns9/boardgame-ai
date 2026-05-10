# mc_player.py
# Monte Carlo player for Can't Stop.
#
# Key design decisions:
# - snapshot/restore for move exploration (fast)
# - clone() for full rollouts (necessary)
# - unified (move, decision) evaluation
# - separate rollout modes for stop vs continue

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_valid_moves, apply_move,
    stop_turn, bust_turn, COLUMN_HEIGHTS
)
from games.cantstop.ev_player import (
    run_tournament, random_player, ev_player
)


# ---- ROLLOUT FROM TURN START ----
def rollout_from_turn_start(state, perspective_player):
    """
    Complete a game from the START of a fresh turn.
    The active player rolls and makes decisions using EV player.
    
    This is used to evaluate the "stop" decision —
    we've banked our progress and it's now a fresh turn.
    """
    state = state.clone()
    max_turns = 500

    for _ in range(max_turns):
        if state.game_over:
            break

        # Fresh turn — roll dice
        state.roll_dice()
        valid = get_valid_moves(state)

        if not valid:
            bust_turn(state)
            continue

        move, decision = ev_player(state, use_corrected=False)

        if move is None:
            bust_turn(state)
            continue

        apply_move(state, move)

        if decision == "stop":
            stop_turn(state)

    return 1 if state.winner == perspective_player else 0


# ---- ROLLOUT FROM MID-TURN (CONTINUE) ----
def rollout_from_mid_turn(state, perspective_player):
    """
    Complete a game from the MIDDLE of a turn.
    The active player already has runners placed and
    must roll again (they chose to continue).
    
    This is used to evaluate the "continue" decision —
    runners are at risk, player rolls again now.
    """
    state = state.clone()
    max_turns = 500
    mid_turn = True  # first iteration is a continuation

    for _ in range(max_turns):
        if state.game_over:
            break

        if mid_turn:
            # Roll again — this is the continuation roll
            state.roll_dice()
            valid = get_valid_moves(state)
            mid_turn = False

            if not valid:
                # Busted immediately on continuation roll
                bust_turn(state)
                # Now it's opponent's turn — continue normally
                continue

            move, decision = ev_player(state, use_corrected=False)

            if move is None:
                bust_turn(state)
                continue

            apply_move(state, move)

            if decision == "stop":
                stop_turn(state)
            # If continue — loop back, same player rolls again

        else:
            # Normal turn start
            state.roll_dice()
            valid = get_valid_moves(state)

            if not valid:
                bust_turn(state)
                continue

            move, decision = ev_player(state, use_corrected=False)

            if move is None:
                bust_turn(state)
                continue

            apply_move(state, move)

            if decision == "stop":
                stop_turn(state)

    return 1 if state.winner == perspective_player else 0


# ---- EVALUATE STOP ----
def evaluate_stop(state, perspective_player, num_simulations):
    """
    Win probability if we stop now and bank current progress.
    Simulates from a fresh turn for the next player.
    """
    if state.game_over:
        return 1.0 if state.winner == perspective_player else 0.0

    wins = sum(
        rollout_from_turn_start(state, perspective_player)
        for _ in range(num_simulations)
    )
    return wins / num_simulations


# ---- EVALUATE CONTINUE ----
def evaluate_continue(state, perspective_player, num_simulations):
    """
    Win probability if we continue rolling with current runners.
    Simulates rolling again immediately from current position.
    """
    if state.game_over:
        return 1.0 if state.winner == perspective_player else 0.0

    wins = sum(
        rollout_from_mid_turn(state, perspective_player)
        for _ in range(num_simulations)
    )
    return wins / num_simulations


# ---- MONTE CARLO PLAYER ----
def mc_player(state, num_simulations=50):
    """
    Evaluates all (move, decision) combinations in one unified pass.
    Uses snapshot/restore for move exploration.
    Uses separate rollout functions for stop vs continue.
    """
    player = state.active_player
    valid = get_valid_moves(state)

    if not valid:
        return None, "bust"

    best_score = -1
    best_move = valid[0]
    best_decision = "stop"

    for move in valid:

        # Save pre-move state
        snap = state.save_snapshot()

        # Apply this move
        apply_move(state, move)

        # ---- EVALUATE STOP ----
        # Stop means saving progress and ending turn
        stop_snap = state.save_snapshot()
        stop_turn(state)
        stop_score = evaluate_stop(state, player, num_simulations)

        # Restore to after-move position
        state.restore_snapshot(stop_snap)

        # ---- EVALUATE CONTINUE ----
        # Continue means rolling again from current runner position
        continue_score = evaluate_continue(state, player, num_simulations)

        # Restore to pre-move position
        state.restore_snapshot(snap)

        # ---- PICK BEST FOR THIS MOVE ----
        if stop_score >= continue_score:
            move_score = stop_score
            move_decision = "stop"
        else:
            move_score = continue_score
            move_decision = "continue"

        # ---- UPDATE BEST OVERALL ----
        if move_score > best_score:
            best_score = move_score
            best_move = move
            best_decision = move_decision

    return best_move, best_decision


# ---- WATCH A GAME ----
def watch_mc_game(turns=5, num_simulations=50):
    """Watch MC player decisions in detail."""
    state = GameState(["A", "B"])

    print(f"Watching MC player ({num_simulations} sims/decision)...\n")
    turn = 0

    while not state.game_over and turn < turns:
        player = state.active_player

        if player != "A":
            state.roll_dice()
            valid = get_valid_moves(state)
            if not valid:
                bust_turn(state)
            else:
                move, decision = ev_player(state, use_corrected=False)
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

        print(f"Turn {turn} | Dice: {state.dice} | Runners: {state.runners}")
        print(f"  Valid moves: {valid}")

        for move in valid:
            snap = state.save_snapshot()
            apply_move(state, move)

            stop_snap = state.save_snapshot()
            stop_turn(state)
            s_score = evaluate_stop(state, player, num_simulations)
            state.restore_snapshot(stop_snap)

            c_score = evaluate_continue(state, player, num_simulations)
            state.restore_snapshot(snap)

            print(f"  {move}: stop={s_score:.1%}  continue={c_score:.1%}")

        move, decision = mc_player(state, num_simulations)
        print(f"  Chosen: {move} → {decision}")
        print()

        apply_move(state, move)
        if decision == "stop":
            stop_turn(state)


# ---- MAIN ----
if __name__ == "__main__":
    print("Monte Carlo player (EV rollouts, unified evaluation)\n")

    watch_mc_game(turns=5, num_simulations=50)

    print("\nRunning tournaments (1,000 games each)...\n")

    mc_50 = lambda s: mc_player(s, num_simulations=50)
    rand  = lambda s: random_player(s)
    ev    = lambda s: ev_player(s, use_corrected=False)

    # Uncomment to run tournaments
    run_tournament(rand, mc_50, "Random",    "MC (50 sims)", games=1000)
    run_tournament(ev,   mc_50, "EV Player", "MC (50 sims)", games=1000)

    print("\nDone!")