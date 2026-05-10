# mc_player.py
# Monte Carlo player for Can't Stop.
#
# Architecture:
# - Single unified rollout function (mid_turn flag replaces two functions)
# - Lightweight fast rollout policy (no EV table lookups)
# - snapshot/restore for move exploration
# - clone() for full rollouts
# - Unified (move, decision) evaluation
# - Parallel rollouts via multiprocessing (mc_player_parallel)
# - Parallel tournament runner (run_parallel_tournament)

import random
import sys
import os
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_valid_moves, apply_move,
    stop_turn, bust_turn, COLUMN_HEIGHTS, MAX_RUNNERS
)
from games.cantstop.ev_player import run_tournament, random_player, ev_player, play_game


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


# ============================================================
# PARALLEL SUPPORT — module-level functions required for pickle
# ============================================================

def _rollout_task(snap, player_id, mid_turn):
    """Rollout worker: rebuild state from snapshot and run rollout."""
    state = GameState(2)
    state.restore_snapshot(snap)
    return rollout(state, player_id, mid_turn=mid_turn)


def _spec_to_strategy(spec):
    """Build a strategy callable from a spec dict. Called inside worker processes."""
    t = spec['type']
    if t == 'random':
        return random_player
    elif t == 'ev':
        c = spec.get('corrected', False)
        return lambda s: ev_player(s, use_corrected=c)
    elif t == 'mc':
        n = spec.get('sims', 50)
        return lambda s: mc_player(s, num_simulations=n)
    raise ValueError(f"Unknown strategy type: {t!r}")


def _run_game_batch(args):
    """
    Run a batch of games. Worker function for parallel tournaments.
    Returns (wins_a, wins_b, draws).
    """
    spec_a, spec_b, swaps, seed = args
    random.seed(seed)
    strat_a = _spec_to_strategy(spec_a)
    strat_b = _spec_to_strategy(spec_b)

    wins_a = wins_b = draws = 0
    for swap in swaps:
        if not swap:
            winner = play_game(strat_a, strat_b)
            if winner == 0:   wins_a += 1
            elif winner == 1: wins_b += 1
            else:             draws += 1
        else:
            winner = play_game(strat_b, strat_a)
            if winner == 0:   wins_b += 1
            elif winner == 1: wins_a += 1
            else:             draws += 1
    return wins_a, wins_b, draws


# ---- PARALLEL MC PLAYER ----

def mc_player_parallel(state, num_simulations=50, pool=None):
    """
    MC player using a persistent process pool for rollouts.

    All rollouts for all (move, stop/continue) combinations are batched
    into a single pool.starmap call to minimize per-call overhead.
    Falls back to sequential mc_player when pool is None.
    """
    if pool is None:
        return mc_player(state, num_simulations)

    player = state.active_player
    valid = get_valid_moves(state)

    if not valid:
        return None, "bust"

    # Build all rollout tasks across all moves in one pass
    tasks = []
    entries = []  # (move, stop_fixed_or_None, stop_slice, cont_slice)

    for move in valid:
        snap0 = state.save_snapshot()
        apply_move(state, move)

        if state.game_over:
            # Move completed a column immediately — stop is trivially known
            stop_fixed = 1.0 if state.winner == player else 0.0
            mid_snap = state.save_snapshot()
            state.restore_snapshot(snap0)
            s0 = len(tasks)
            tasks.extend([(mid_snap, player, True)] * num_simulations)
            entries.append((move, stop_fixed, None, slice(s0, s0 + num_simulations)))
            continue

        mid_snap = state.save_snapshot()
        stop_turn(state)

        if state.game_over:
            stop_fixed = 1.0 if state.winner == player else 0.0
            state.restore_snapshot(snap0)
            s0 = len(tasks)
            tasks.extend([(mid_snap, player, True)] * num_simulations)
            entries.append((move, stop_fixed, None, slice(s0, s0 + num_simulations)))
        else:
            stop_snap = state.save_snapshot()
            state.restore_snapshot(snap0)
            s_stop = len(tasks)
            tasks.extend([(stop_snap, player, False)] * num_simulations)
            s_cont = len(tasks)
            tasks.extend([(mid_snap, player, True)] * num_simulations)
            entries.append((move, None, slice(s_stop, s_cont), slice(s_cont, s_cont + num_simulations)))

    # Single parallel dispatch for all rollouts
    chunksize = max(1, len(tasks) // (pool._processes * 4))
    results = pool.starmap(_rollout_task, tasks, chunksize=chunksize)

    best_score = -1
    best_move = valid[0]
    best_decision = "stop"

    for move, stop_fixed, stop_sl, cont_sl in entries:
        stop_score = stop_fixed if stop_fixed is not None else sum(results[stop_sl]) / num_simulations
        cont_score = sum(results[cont_sl]) / num_simulations

        if stop_score >= cont_score:
            score, decision = stop_score, "stop"
        else:
            score, decision = cont_score, "continue"

        if score > best_score:
            best_score = score
            best_move = move
            best_decision = decision

    return best_move, best_decision


# ---- PARALLEL TOURNAMENT ----

def run_parallel_tournament(spec_a, spec_b, name_a, name_b, games=1000, workers=None):
    """
    Run a tournament using multiple worker processes (game-level parallelism).

    Strategies are specified as dicts so they can be pickled:
        {"type": "random"}
        {"type": "ev", "corrected": False}
        {"type": "mc", "sims": 10}

    Each worker runs a batch of full games independently.
    """
    import time

    if workers is None:
        workers = mp.cpu_count()

    rng = random.Random(42)
    swaps = [rng.random() < 0.5 for _ in range(games)]

    batch_size = max(1, (games + workers - 1) // workers)
    batches = []
    for i in range(0, games, batch_size):
        batch_swaps = swaps[i:i + batch_size]
        seed = rng.randint(0, 2**31)
        batches.append((spec_a, spec_b, batch_swaps, seed))

    t0 = time.time()
    with mp.Pool(workers) as pool:
        results = pool.map(_run_game_batch, batches)
    elapsed = time.time() - t0

    total_a = sum(r[0] for r in results)
    total_b = sum(r[1] for r in results)
    total = total_a + total_b

    print(f"\n{'='*45}")
    print(f"  {name_a} vs {name_b}  ({workers} workers, {elapsed:.1f}s)")
    print(f"  {games:,} games played")
    print(f"{'='*45}")
    print(f"  {name_a:<25} {total_a:>5} wins  ({100*total_a/total:.1f}%)")
    print(f"  {name_b:<25} {total_b:>5} wins  ({100*total_b/total:.1f}%)")


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

    mp.freeze_support()  # required for Windows frozen executables

    print("=" * 50)
    print("Timing: sequential vs parallel MC player")
    print("=" * 50)

    state = GameState(2)
    state.roll_dice()

    # Sequential baseline
    N_TIMING = 20
    start = time.time()
    for _ in range(N_TIMING):
        state2 = state.clone()
        mc_player(state2, num_simulations=50)
    seq_time = (time.time() - start) / N_TIMING
    print(f"\nSequential  mc_player(50 sims): {seq_time*1000:.1f}ms/decision")

    # Parallel within-decision
    workers = mp.cpu_count()
    with mp.Pool(workers) as pool:
        # Warm up pool
        mc_player_parallel(state, num_simulations=50, pool=pool)

        start = time.time()
        for _ in range(N_TIMING):
            state2 = state.clone()
            mc_player_parallel(state2, num_simulations=50, pool=pool)
        par_time = (time.time() - start) / N_TIMING

    print(f"Parallel    mc_player(50 sims): {par_time*1000:.1f}ms/decision  ({workers} workers)")
    print(f"Speedup: {seq_time/par_time:.1f}x")

    print("\n" + "=" * 50)
    print("Tournaments — parallel (game-level, 1,000 games)")
    print("=" * 50)

    mc10  = {"type": "mc",     "sims": 10}
    rand  = {"type": "random"}
    ev    = {"type": "ev",     "corrected": False}

    run_parallel_tournament(rand, mc10, "Random",    "MC (10 sims)", games=1000)
    run_parallel_tournament(ev,   mc10, "EV Player", "MC (10 sims)", games=1000)

    print("\nDone!")
