# generate_training_data.py
# Generates training data for the Can't Stop neural network.
# Runs overnight using all available CPU cores.
#
# Output: data/cantstop/training_data.jsonl
# Format: one JSON record per line, each representing one decision point.
#
# Each record contains:
#   - state: full game state at decision time
#   - move: which move was chosen
#   - decision: stop or continue
#   - outcome: 1 if this player won, 0 if they lost
#   - ev_prob: EV player's estimated win probability (from weighted progress)

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


# ---- STATE SERIALIZATION ----
def serialize_state(state, move, decision):
    """
    Convert game state + decision into a training record.
    Neural network will learn to predict outcome from this.
    """
    player = state.active_player
    opponent = 1 - player

    return {
        # Who is deciding
        "active_player": player,

        # Dice rolled this turn
        "dice": list(state.dice),

        # Runner positions (current turn progress at risk)
        "runners": dict(state.runners),

        # Saved progress per column per player
        "progress_active": dict(state.progress[player]),
        "progress_opponent": dict(state.progress[opponent]),

        # Columns claimed
        "claimed_active": list(state.claimed[player]),
        "claimed_opponent": list(state.claimed[opponent]),

        # Scores (columns claimed count)
        "score_active": len(state.claimed[player]),
        "score_opponent": len(state.claimed[opponent]),

        # The decision made
        "move": list(move),
        "decision": decision,

        # Weighted progress at risk (our EV metric)
        "weighted_progress": get_total_runner_progress(state),

        # Outcome filled in after game ends
        "outcome": None,
    }


# ---- SINGLE GAME GENERATOR ----
def generate_game(ev_table, use_mc=False):
    """
    Play one full game recording every decision point.
    Returns list of training records with outcomes filled in.

    use_mc=False: use EV player for all decisions (fast)
    use_mc=True:  use MC player for some decisions (slower, higher quality)
    """
    state = GameState(2)
    records = []
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

        # Get decision from EV player
        move, decision = ev_player(state, use_corrected=False)

        if move is None:
            bust_turn(state)
            continue

        # Record this decision point BEFORE applying the move
        record = serialize_state(state, move, decision)
        records.append((record, player))

        # Apply the decision
        apply_move(state, move)

        if decision == "stop":
            stop_turn(state)

    # Fill in outcomes now that we know who won
    winner = state.winner
    labeled = []
    for record, player in records:
        record["outcome"] = 1 if player == winner else 0
        labeled.append(record)

    return labeled


# ---- WORKER FUNCTION ----
def worker_generate_batch(args):
    """
    Worker function — generates a batch of games.
    Must be top-level for multiprocessing pickling.

    Returns list of training records.
    """
    batch_size, seed = args
    random.seed(seed)

    # Build EV table once per worker
    ev_table = build_ev_table()

    all_records = []
    for _ in range(batch_size):
        records = generate_game(ev_table)
        all_records.extend(records)

    return all_records


# ---- PROGRESS TRACKER ----
class ProgressTracker:
    def __init__(self, target_records, output_path):
        self.target = target_records
        self.output_path = output_path
        self.total_records = 0
        self.total_games = 0
        self.start_time = time.time()

    def update(self, new_records):
        self.total_records += len(new_records)
        self.total_games += 1

        elapsed = time.time() - self.start_time
        rate = self.total_records / elapsed if elapsed > 0 else 0
        eta_seconds = (self.target - self.total_records) / rate if rate > 0 else 0
        eta_hours = eta_seconds / 3600

        pct = 100 * self.total_records / self.target

        print(
            f"\r  Records: {self.total_records:>8,} / {self.target:,}"
            f"  ({pct:.1f}%)"
            f"  Rate: {rate:.0f}/s"
            f"  ETA: {eta_hours:.1f}h",
            end="", flush=True
        )

    def done(self):
        elapsed = time.time() - self.start_time
        print(f"\n\n  Finished in {elapsed/3600:.2f} hours")
        print(f"  Total records: {self.total_records:,}")
        print(f"  Total games:   {self.total_games:,}")
        print(f"  Output: {self.output_path}")


# ---- MAIN GENERATOR ----
def generate_training_data(
    target_records=5_000_000,
    batch_size=50,
    workers=None,
    output_path=None
):
    """
    Generate training data using all available CPU cores.

    target_records: how many decision records to generate
    batch_size: games per worker batch (tune for memory efficiency)
    workers: number of CPU cores (default: all available)
    output_path: where to save the data
    """
    if workers is None:
        workers = mp.cpu_count()

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            "data", "cantstop",
            f"training_data_{timestamp}.jsonl"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Can't Stop Training Data Generator")
    print(f"{'='*55}")
    print(f"  Target records:  {target_records:,}")
    print(f"  CPU workers:     {workers}")
    print(f"  Batch size:      {batch_size} games/batch")
    print(f"  Output:          {output_path}")
    print(f"{'='*55}\n")

    tracker = ProgressTracker(target_records, output_path)
    rng = random.Random(42)

    with open(output_path, 'w') as f:
        with mp.Pool(workers) as pool:
            while tracker.total_records < target_records:
                # Generate seeds for this round of batches
                batch_args = [
                    (batch_size, rng.randint(0, 2**31))
                    for _ in range(workers * 2)  # keep workers busy
                ]

                # Run batches in parallel
                results = pool.map(worker_generate_batch, batch_args)

                # Write records to file
                for batch_records in results:
                    for record in batch_records:
                        f.write(json.dumps(record) + '\n')
                        tracker.total_records += 1

                    tracker.total_games += batch_size

                # Print progress
                elapsed = time.time() - tracker.start_time
                rate = tracker.total_records / elapsed if elapsed > 0 else 0
                eta = (target_records - tracker.total_records) / rate / 3600 if rate > 0 else 0
                pct = 100 * tracker.total_records / target_records

                print(
                    f"\r  Records: {tracker.total_records:>8,} / {target_records:,}"
                    f"  ({pct:.1f}%)"
                    f"  Rate: {rate:,.0f}/s"
                    f"  ETA: {eta:.1f}h",
                    end="", flush=True
                )

                if tracker.total_records >= target_records:
                    break

    tracker.done()


# ---- QUICK TEST ----
def quick_test():
    """Verify everything works before running overnight."""
    print("Quick test — generating 100 games...\n")

    start = time.time()
    records = worker_generate_batch((100, 42))
    elapsed = time.time() - start

    print(f"  Generated {len(records):,} records from 100 games")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Rate: {len(records)/elapsed:,.0f} records/second")
    print(f"\n  Sample record:")

    sample = records[0]
    for key, val in sample.items():
        print(f"    {key}: {val}")

    # Estimate overnight capacity
    rate = len(records) / elapsed
    hours = 8
    estimated = int(rate * 3600 * hours)
    print(f"\n  Estimated records in {hours} hours: {estimated:,}")
    print(f"  Estimated games in {hours} hours: {estimated // (len(records)//100):,}")


# ---- ENTRY POINT ----
if __name__ == "__main__":
    mp.freeze_support()  # Windows requirement

    import argparse
    parser = argparse.ArgumentParser(description="Generate Can't Stop training data")
    parser.add_argument("--test", action="store_true", help="Run quick test only")
    parser.add_argument("--records", type=int, default=5_000_000, help="Target records")
    parser.add_argument("--workers", type=int, default=None, help="CPU workers")
    parser.add_argument("--batch", type=int, default=50, help="Games per batch")
    args = parser.parse_args()

    if args.test:
        quick_test()
    else:
        generate_training_data(
            target_records=args.records,
            batch_size=args.batch,
            workers=args.workers,
        )