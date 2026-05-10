# mc_player.py
# Monte Carlo player for Can't Stop.
#
# Architecture:
# - Single unified rollout function (mid_turn flag replaces two functions)
# - Lightweight fast rollout policy (no EV table lookups)
# - snapshot/restore for move exploration
# - clone() for full rollouts
# - Unified (move, decision) evaluation

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_valid_moves, apply_move,
    stop_turn, bust_turn, COLUMN_HEIGHTS, MAX_RUNNERS
)
from games.cantstop.ev_player import run_tournament, random_player, ev_player


# ---- FAST ROLLOUT POLICY ----
# Used inside rollouts — must be fast.
# Captures good play without expensive EV table lookups.

def fast_move(state):
    """
    Pick a move quickly — prefer columns we already have runners on.
    No EV table lookup needed.
    """
    valid = get_valid_moves(state)
    if not valid:
        return None

    runner_cols = state.runners

    def move_score(move):
        # Count overlap with existing runners
        overlap = sum(1 for col in move if col in runner_cols)
        # Prefer middle columns (closer to 7 = more likely to roll)
        centrality = sum(6 - abs(7 - col) for col in set(move))
        return (overlap, centrality)

    return max(valid, key=move_score)


def fast_stop_decision(state, turns_taken):
    """
    Decide stop/continue quickly using simple thresholds.
    No weighted progress calculation needed.

    Logic:
    - More runners placed → more to lose → stop sooner
    - More turns taken this sequence → more progress → stop sooner
    - Random element to avoid determinism
    """
    num_runners = len(state.runners)
    total_steps = sum(state.runners.values())

    # Base stop probability increases with runners and steps taken
    # 0 runners: never stop (nothing at risk yet)
    # 3 runners, many steps: high stop probability
    stop_prob = (num_runners / MAX_RUNNERS) * (total_steps / 6)

    # Cap between 0.1 and 0.9 — never completely deterministic
    stop_prob = max(0.1, min(0.9, stop_prob))

    return random.random() < stop_prob


# ---- UNIFIED ROLLOUT ----
def rollout(state, perspective_player, mid_turn=False):
    """
    Play a full game to completion using fast rollout policy.

    mid_turn=False: start of fresh turn — roll dice first
    mid_turn=True:  middle of turn — player has runners, rolls again now

    Single function replaces rollout_from_turn_start and
    rollout_from_mid_turn — avoids code duplication.

    Returns 1 if perspective_player wins, 0 otherwise.
    Uses heuristic_value if max_turns reached.
    """
    state = state.clone()
    max_turns = 200  # consistent with play_game, covers all observed games
    turns_this_sequence = 0

    for turn in range(max_turns):
        if state.game_over:
            break

        # Roll dice
        state.roll_dice()
        valid = get_valid_moves(state)

        if not valid:
            bust_turn(state)
            turns_this_sequence = 0
            continue

        # Pick move
        move = fast_move(state)
        if move is None:
            bust_turn(state)
            turns_this_sequence = 0
            continue

        apply_move(state, move)
        turns_this_sequence += 1

        # Stop or continue
        if fast_stop_decision(state, turns_this_sequence):
            stop_turn(state)
            turns_this_sequence = 0

    if state.game_over:
        return 1 if state.winner == perspective_player else 0
    else:
        # Game didn't finish — use heuristic
        return state.heuristic_value(perspective_player)


# ---- EVALUATE STOP ----
def evaluate_stop(state, perspective_player, num_simulations):
    """Win probability if we stop now and bank current progress."""
    if state.game_over:
        return 1.0 if state.winner == perspective_player else 0.0

    snap = state.save_snapshot()
    stop_turn(state)
    wins = sum(
        rollout(state, perspective_player, mid_turn=False)
        for _ in range(num_simulations)
    )
    state.restore_snapshot(snap)
    return wins / num_simulations


# ---- EVALUATE CONTINUE ----
def evaluate_continue(state, perspective_player, num_simulations):
    """Win probability if we roll again from current runner position."""
    if state.game_over:
        return 1.0 if state.winner == perspective_player else 0.0

    wins = sum(
        rollout(state, perspective_player, mid_turn=True)
        for _ in range(num_simulations)
    )
    return wins / num_simulations


# ---- MONTE CARLO PLAYER ----
def mc_player(state, num_simulations=50):
    """
    Evaluates all (move, decision) combinations.
    Uses snapshot/restore for move exploration.
    Uses unified rollout for evaluation.
    """
    player = state.active_player
    valid = get_valid_moves(state)

    if not valid:
        return None, "bust"

    best_score = -1
    best_move = valid[0]
    best_decision = "stop"

    for move in valid:
        snap = state.save_snapshot()
        apply_move(state, move)

        # Evaluate stop
        stop_score = evaluate_stop(state, player, num_simulations)

        # Evaluate continue
        cont_score = evaluate_continue(state, player, num_simulations)

        state.restore_snapshot(snap)

        if stop_score >= cont_score:
            move_score = stop_score
            move_decision = "stop"
        else:
            move_score = cont_score
            move_decision = "continue"

        if move_score > best_score:
            best_score = move_score
            best_move = move
            best_decision = move_decision

    return best_move, best_decision


# ---- WATCH A GAME ----
def watch_mc_game(turns=5, num_simulations=50):
    """Watch MC player decisions in detail."""
    state = GameState(2)
    print(f"Watching MC player ({num_simulations} sims/decision)...\n")
    turn = 0

    while not state.game_over and turn < turns:
        player = state.active_player

        if player != 0:
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
            s = evaluate_stop(state, player, num_simulations)
            c = evaluate_continue(state, player, num_simulations)
            state.restore_snapshot(snap)
            print(f"  {move}: stop={s:.1%}  continue={c:.1%}")

        move, decision = mc_player(state, num_simulations)
        print(f"  Chosen: {move} → {decision}\n")

        apply_move(state, move)
        if decision == "stop":
            stop_turn(state)


# ---- MAIN ----
if __name__ == "__main__":
    import time

    print("Timing fast rollout policy...\n")

    state = GameState(2)
    state.roll_dice()

    start = time.time()
    move, decision = mc_player(state, num_simulations=50)
    elapsed = time.time() - start

    print(f"One MC decision (50 sims): {elapsed:.3f}s")
    print(f"Estimated per game (~50 decisions): {elapsed*50:.1f}s")
    print(f"Estimated 1000 games: {elapsed*50*1000/3600:.2f} hours")

    print("\nWatching decisions...\n")
    watch_mc_game(turns=5, num_simulations=50)

    print("\nRunning tournaments (1,000 games each)...\n")

    mc_10 = lambda s: mc_player(s, num_simulations=10)
    rand  = lambda s: random_player(s)
    ev    = lambda s: ev_player(s, use_corrected=False)

    run_tournament(rand, mc_10, "Random",    "MC (10 sims)", games=1000)
    run_tournament(ev,   mc_10, "EV Player", "MC (10 sims)", games=1000)

    print("\nDone!")