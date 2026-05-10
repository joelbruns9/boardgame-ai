# generate_training_data.py
# Generates training data for the Can't Stop neural network.
# Runs overnight using all available CPU cores.
#
# Improvements over v1:
# - Epsilon-greedy exploration (15% random moves) for dataset diversity
# - Records valid_moves and step_index for future RL training
# - Buffered writes for faster I/O
# - imap_unordered for better CPU utilization
# - EV table built once per worker via initializer
# - Pre-generates all batch args upfront (fixes early termination bug)

import json
import os
import sys
import time
import random
import multiprocessing as mp
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_valid_moves, apply_move,
    stop_turn, bust_turn, COLUMN_HEIGHTS
)
from games.cantstop.ev_player import ev_player, get_total_runner_progress
from games.cantstop.ev_table import build_ev_table

# ---- EXPLORATION RATE ----
EPSILON = 0.15

# ---- WRITE BUFFER SIZE ----
BUFFER_SIZE = 1_000

# ---- AVERAGE RECORDS PER GAME ----
# Used to estimate batch count — measured at ~77
AVG_RECORDS_PER_GAME = 77


# ---- STATE SERIALIZATION ----
def serialize_state(state, move, decision, valid_moves, step_index):
    """Convert game state + decision into a training record."""
    player = state.active_player
    opponent = 1 - player

    return {
        "active_player":    player,
        "dice":             list(state.dice),
        "runners":          dict(state.runners),
        "progress_active":  dict(state.progress[player]),
        "progress_opponent":dict(state.progress[opponent]),
        "claimed_active":   sorted(state.claimed[player]),
        "claimed_opponent": sorted(state.claimed[opponent]),
        "score_active":     len(state.claimed[player]),
        "score_opponent":   len(state.claimed[opponent]),
        "valid_moves":      [list(m) for m in valid_moves],
        "move":             list(move),
        "decision":         decision,
        "weighted_progress":round(get_total_runner_progress(state), 6),
        "step_index":       step_index,
        "is_exploration":   False,
        "outcome":          None,
    }


# ---- SINGLE GAME GENERATOR ----
def generate_game():
    """
    Play one full game recording every decision point.
    Uses epsilon-greedy exploration (EPSILON % random moves).
    """
    state = GameState(2)
    records = []
    step_index = 0
    max_turns = 200

    for _ in range(max_turns):
        if state.game_over:
            break

        state.roll_dice()
        valid = get_valid_moves(state)

        if not valid:
            bust_turn(state)
            continue

        player = state.active_player
        is_exploration = random.random() < EPSILON

        if is_exploration:
            move = random.choice(valid)
            decision = random.choice(["stop", "continue"])
        else:
            move, decision = ev_player(state, use_corrected=False)
            if move is None:
                bust_turn(state)
                continue

        record = serialize_state(state, move, decision, valid, step_index)
        record["is_exploration"] = is_exploration
        records.append((record, player))
        step_index += 1

        apply_move(state, move)
        if decision == "stop":
            stop_turn(state)

    # Fill in outcomes
    winner = state.winner
    labeled = []
    for record, player in records:
        record["outcome"] = 1 if player == winner else 0
        labeled.append(record)

    return labeled


# ---- WORKER INITIALIZER ----
def worker_init():
    """Warm up EV table once per worker process."""
    from games.cantstop.ev_player import ev_player
    from games.cantstop.engine import GameState
    state = GameState(2)
    state.dice = [1, 2, 3, 4]
    ev_player(state)


# ---- WORKER FUNCTION ----
def worker_generate_batch(args):
    """Generate a batch of games. Returns list of training records."""
    batch_size, seed = args
    random.seed(seed)
    all_records = []
    for _ in range(batch_size):
        all_records.extend(generate_game())
    return all_records


# ---- MAIN GENERATOR ----
def generate_training_data(
    target_records=5_000_000,
    batch_size=50,
    workers=None,
    output_path=None
):
    """Generate training data using all available CPU cores."""
    if workers is None:
        workers = mp.cpu_count()

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            "data", "cantstop",
            f"training_data_{timestamp}.jsonl"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Calculate how many batches we need upfront
    records_per_batch = batch_size * AVG_RECORDS_PER_GAME
    num_batches = (target_records // records_per_batch) + workers * 2

    print(f"\n{'='*55}")
    print(f"  Can't Stop Training Data Generator v2")
    print(f"{'='*55}")
    print(f"  Target records:  {target_records:,}")
    print(f"  CPU workers:     {workers}")
    print(f"  Batch size:      {batch_size} games/batch")
    print(f"  Num batches:     {num_batches:,}")
    print(f"  Exploration:     {EPSILON:.0%} random moves")
    print(f"  Output:          {output_path}")
    print(f"{'='*55}\n")

    # Pre-generate all batch args upfront
    rng = random.Random(42)
    batch_args = [
        (batch_size, rng.randint(0, 2**31))
        for _ in range(num_batches)
    ]

    total_records = 0
    total_games = 0
    start_time = time.time()
    write_buffer = []

    with open(output_path, 'w') as f:
        with mp.Pool(workers, initializer=worker_init) as pool:

            for batch_records in pool.imap_unordered(
                worker_generate_batch,
                batch_args,
                chunksize=1
            ):
                for record in batch_records:
                    write_buffer.append(json.dumps(record))
                    total_records += 1

                total_games += batch_size

                # Flush buffer when full
                if len(write_buffer) >= BUFFER_SIZE:
                    f.write('\n'.join(write_buffer) + '\n')
                    write_buffer.clear()

                # Progress update
                elapsed = time.time() - start_time
                rate = total_records / elapsed if elapsed > 0 else 0
                eta = (target_records - total_records) / rate / 3600 \
                    if rate > 0 else 0
                pct = 100 * total_records / target_records

                print(
                    f"\r  Records: {total_records:>8,} / {target_records:,}"
                    f"  ({pct:.1f}%)"
                    f"  Rate: {rate:,.0f}/s"
                    f"  ETA: {eta:.1f}h",
                    end="", flush=True
                )

                if total_records >= target_records:
                    break

        # Flush remaining buffer
        if write_buffer:
            f.write('\n'.join(write_buffer) + '\n')

    elapsed = time.time() - start_time
    print(f"\n\n{'='*55}")
    print(f"  Complete!")
    print(f"  Total records: {total_records:,}")
    print(f"  Total games:   {total_games:,}")
    print(f"  Time:          {elapsed/3600:.2f} hours")
    print(f"  Final rate:    {total_records/elapsed:,.0f} records/sec")
    print(f"  Output:        {output_path}")
    print(f"{'='*55}\n")


# ---- QUICK TEST ----
def quick_test():
    """Verify everything works before running overnight."""
    print("Quick test — generating 100 games...\n")

    start = time.time()
    records = worker_generate_batch((100, 42))
    elapsed = time.time() - start

    print(f"  Generated {len(records):,} records from 100 games")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Rate: {len(records)/elapsed:,.0f} records/second (single core)")

    explored = sum(1 for r in records if r.get('is_exploration'))
    print(f"  Exploration: {explored}/{len(records)} ({100*explored/len(records):.1f}%)")

    print(f"\n  Sample record:")
    sample = records[10]
    for key, val in sample.items():
        print(f"    {key}: {val}")

    workers = mp.cpu_count()
    estimated = len(records) / elapsed * workers * 3600 * 8
    print(f"\n  Estimated records in 8 hours ({workers} cores): {int(estimated):,}")


# ---- ENTRY POINT ----
if __name__ == "__main__":
    mp.freeze_support()

    import argparse
    parser = argparse.ArgumentParser(
        description="Generate Can't Stop training data"
    )
    parser.add_argument("--test",    action="store_true",
                        help="Run quick test only")
    parser.add_argument("--records", type=int, default=5_000_000,
                        help="Target records")
    parser.add_argument("--workers", type=int, default=None,
                        help="CPU workers (default: all)")
    parser.add_argument("--batch",   type=int, default=50,
                        help="Games per batch")
    parser.add_argument("--output",  type=str, default=None,
                        help="Output file path")
    args = parser.parse_args()

    if args.test:
        quick_test()
    else:
        generate_training_data(
            target_records=args.records,
            batch_size=args.batch,
            workers=args.workers,
            output_path=args.output,
        )