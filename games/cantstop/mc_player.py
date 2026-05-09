# mc_player.py
# Monte Carlo player for Can't Stop.
# Makes decisions by simulating many random games from the current position
# and choosing whatever leads to the highest win rate.
#
# This is the bridge between our rule-based EV player and the ML model.
# The neural network will eventually learn to replicate what Monte Carlo
# discovers through simulation — but instantly, without running games.

import random
import copy
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


# ---- RANDOM ROLLOUT ----
def random_rollout(state, perspective_player):
    """
    Play out a game to completion using random decisions.
    Returns 1 if perspective_player wins, 0 if they lose.

    This is the core of Monte Carlo — a single simulation
    from the current position to the end of the game.

    Think of it like mentally simulating one possible future.
    """
    # Deep copy so we don't modify the real game state
    state = copy.deepcopy(state)
    max_turns = 500  # safety limit

    for _ in range(max_turns):
        if state.game_over:
            break

        state.roll_dice()
        valid = get_valid_moves(state)

        if not valid:
            bust_turn(state)
            continue

        # Random move
        move = random.choice(valid)
        apply_move(state, move)

        # Random stop/continue decision
        if random.random() < 0.5:
            stop_turn(state)

    return 1 if state.winner == perspective_player else 0


# ---- EVALUATE A POSITION ----
def evaluate_position(state, perspective_player, num_simulations):
    """
    Estimate win probability from the current position
    by running num_simulations random rollouts.

    Returns a float between 0.0 and 1.0.
    The higher the number, the better the position.
    """
    wins = sum(
        random_rollout(state, perspective_player)
        for _ in range(num_simulations)
    )
    return wins / num_simulations


# ---- MONTE CARLO PLAYER ----
def mc_player(state, num_simulations=50):
    """
    Makes decisions using Monte Carlo simulation.

    For move selection:
        Tries each valid move, simulates num_simulations games
        from each resulting position, picks the move with
        highest estimated win rate.

    For stop/continue:
        Simulates num_simulations games from "stopped" position
        vs "continued" position, picks whichever wins more.

    num_simulations: how many rollouts per decision.
        Higher = stronger but slower.
        50 is a good balance for testing.
        200+ for serious play.
    """
    player = state.active_player
    valid = get_valid_moves(state)

    if not valid:
        return None, "bust"

    # ---- STEP 1: EVALUATE EACH VALID MOVE ----
    move_scores = {}

    for move in valid:
        # Apply this move to a copy
        state_after_move = copy.deepcopy(state)
        apply_move(state_after_move, move)

        # Evaluate the resulting position
        score = evaluate_position(state_after_move, player, num_simulations)
        move_scores[move] = score

    # Pick the move with highest win rate
    best_move = max(move_scores, key=lambda m: move_scores[m])
    best_score = move_scores[best_move]

    # ---- STEP 2: STOP OR CONTINUE? ----
    # Apply the best move to a copy
    state_after_best = copy.deepcopy(state)
    apply_move(state_after_best, best_move)

    # Evaluate "stop now" — save progress and end turn
    state_if_stop = copy.deepcopy(state_after_best)
    stop_turn(state_if_stop)
    stop_score = evaluate_position(state_if_stop, player, num_simulations)

    # Evaluate "continue" — keep rolling from this position
    continue_score = evaluate_position(state_after_best, player, num_simulations)

    # Pick whichever is better
    decision = "stop" if stop_score >= continue_score else "continue"

    return best_move, decision


# ---- WATCH A GAME ----
def watch_mc_game(turns=8, num_simulations=50):
    """Watch the Monte Carlo player's decisions in detail."""
    state = GameState(["A", "B"])

    print(f"Watching Monte Carlo player ({num_simulations} simulations per decision)...\n")
    turn = 0

    while not state.game_over and turn < turns:
        player = state.active_player

        if player != "A":
            # B plays random
            state.roll_dice()
            valid = get_valid_moves(state)
            if not valid:
                bust_turn(state)
            else:
                move = random.choice(valid)
                apply_move(state, move)
                if random.random() < 0.5:
                    stop_turn(state)
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

        # Show win rate for each move
        for move in valid:
            state_copy = copy.deepcopy(state)
            apply_move(state_copy, move)
            score = evaluate_position(state_copy, player, num_simulations)
            print(f"  Move {move}: estimated win rate {score:.1%}")

        move, decision = mc_player(state, num_simulations)
        print(f"  Chosen: {move} → {decision}")
        print()

        apply_move(state, move)
        if decision == "stop":
            stop_turn(state)


# ---- MAIN ----
if __name__ == "__main__":
    print("Building Monte Carlo player...\n")

    # Watch a few decisions first
    watch_mc_game(turns=5, num_simulations=50)

    print("\nRunning tournaments (1,000 games each)...")
    print("Note: fewer games than before — MC is slower to run\n")

    mc_50  = lambda s: mc_player(s, num_simulations=50)
    mc_200 = lambda s: mc_player(s, num_simulations=200)
    rand   = lambda s: random_player(s)
    ev     = lambda s: ev_player(s, use_corrected=False)

    # MC vs Random
    run_tournament(rand, mc_50, "Random", "MC (50 sims)", games=1000)

    # MC vs EV player — the key test
    run_tournament(ev, mc_50, "EV Player", "MC (50 sims)", games=1000)

    # Does more simulations help?
    run_tournament(mc_50, mc_200, "MC (50 sims)", "MC (200 sims)", games=500)

    print("\nDone!")