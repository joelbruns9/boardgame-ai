# validate_training_data.py
# Validates training data quality before large generation runs.
# Run this on a small sample before launching overnight generation.
#
# Checks:
#   1. Schema completeness — all required fields present
#   2. Value ranges — all values within legal bounds
#   3. Outcome balance — roughly 50/50 wins per player
#   4. Decision distribution — stop/continue ratio is sensible
#   5. Position diversity — states aren't all identical
#   6. Outcome consistency — within each game, outcomes are consistent
#   7. Move legality — recorded moves were legal given the dice
#   8. Progress consistency — weighted progress matches recorded state

import json
import sys
import os
import random
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_valid_moves, apply_move,
    COLUMN_HEIGHTS, COLUMNS_TO_WIN
)
from games.cantstop.ev_player import get_total_runner_progress
from games.cantstop.generate_training_data import (
    generate_game, worker_generate_batch
)
from games.cantstop.ev_table import build_ev_table

# ---- TEST FRAMEWORK ----
passed = 0
failed = 0
warnings = 0

def check(name, condition, details=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        if details:
            print(f"        {details}")
        failed += 1

def warn(name, condition, details=""):
    global warnings
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  WARN  {name}")
        if details:
            print(f"        {details}")
        warnings += 1

def section(name):
    print(f"\n{name}")
    print("-" * 55)


# ---- GENERATE SAMPLE DATA ----
def generate_sample(num_games=500):
    """Generate a sample dataset for validation."""
    print(f"\nGenerating {num_games} games for validation...")
    records = []
    game_outcomes = []  # track (game_id, player, outcome) for consistency check

    # Generate games and tag each record with a game ID
    for game_id in range(num_games):
        game_records = generate_game()
        for rec in game_records:
            rec['game_id'] = game_id
        records.extend(game_records)

    print(f"  Generated {len(records):,} records from {num_games} games")
    print(f"  Average {len(records)/num_games:.1f} records per game\n")
    return records


# ---- VALIDATION CHECKS ----

REQUIRED_FIELDS = [
    "active_player", "dice", "runners", "progress_active",
    "progress_opponent", "claimed_active", "claimed_opponent",
    "score_active", "score_opponent", "move", "decision",
    "weighted_progress", "outcome"
]

def validate_schema(records):
    section("Schema Completeness")

    missing_fields = []
    for i, rec in enumerate(records):
        for field in REQUIRED_FIELDS:
            if field not in rec:
                missing_fields.append((i, field))

    check(
        "All required fields present in every record",
        len(missing_fields) == 0,
        f"Missing: {missing_fields[:5]}"
    )

    # Check types
    type_errors = []
    for i, rec in enumerate(records[:100]):
        if not isinstance(rec['dice'], list) or len(rec['dice']) != 4:
            type_errors.append((i, 'dice', rec['dice']))
        if rec['outcome'] not in [0, 1]:
            type_errors.append((i, 'outcome', rec['outcome']))
        if rec['decision'] not in ['stop', 'continue']:
            type_errors.append((i, 'decision', rec['decision']))
        if rec['active_player'] not in [0, 1]:
            type_errors.append((i, 'active_player', rec['active_player']))

    check(
        "All field types are correct",
        len(type_errors) == 0,
        f"Type errors: {type_errors[:3]}"
    )


def validate_value_ranges(records):
    section("Value Ranges")

    dice_errors = []
    score_errors = []
    progress_errors = []
    col_errors = []

    for rec in records:
        # Dice values 1-6
        if any(d < 1 or d > 6 for d in rec['dice']):
            dice_errors.append(rec['dice'])

        # Scores 0-3
        if rec['score_active'] < 0 or rec['score_active'] > COLUMNS_TO_WIN:
            score_errors.append(rec['score_active'])
        if rec['score_opponent'] < 0 or rec['score_opponent'] > COLUMNS_TO_WIN:
            score_errors.append(rec['score_opponent'])

        # Progress values within column heights
        for col_str, steps in rec['progress_active'].items():
            col = int(col_str)
            if col not in COLUMN_HEIGHTS or steps > COLUMN_HEIGHTS[col]:
                progress_errors.append((col, steps))

        # Claimed columns are valid (2-12)
        for col in rec['claimed_active'] + rec['claimed_opponent']:
            if col not in COLUMN_HEIGHTS:
                col_errors.append(col)

    check(
        "All dice values in range 1-6",
        len(dice_errors) == 0,
        f"Bad dice: {dice_errors[:3]}"
    )
    check(
        "All scores in range 0-3",
        len(score_errors) == 0,
        f"Bad scores: {score_errors[:3]}"
    )
    check(
        "All progress values within column heights",
        len(progress_errors) == 0,
        f"Bad progress: {progress_errors[:3]}"
    )
    check(
        "All claimed columns are valid (2-12)",
        len(col_errors) == 0,
        f"Bad columns: {col_errors[:3]}"
    )


def validate_outcome_balance(records):
    section("Outcome Balance")

    outcomes = [rec['outcome'] for rec in records]
    wins = sum(outcomes)
    total = len(outcomes)
    win_rate = wins / total

    check(
        "Outcome balance roughly 50/50 (45-55%)",
        0.45 <= win_rate <= 0.55,
        f"Win rate: {win_rate:.3f} ({wins}/{total})"
    )

    # Check per-player balance
    p0_records = [r for r in records if r['active_player'] == 0]
    p1_records = [r for r in records if r['active_player'] == 1]

    p0_wins = sum(r['outcome'] for r in p0_records) / len(p0_records) if p0_records else 0
    p1_wins = sum(r['outcome'] for r in p1_records) / len(p1_records) if p1_records else 0

    warn(
        f"Player 0 win rate reasonable (40-60%): {p0_wins:.3f}",
        0.40 <= p0_wins <= 0.60,
        f"Player 0 win rate: {p0_wins:.3f}"
    )
    warn(
        f"Player 1 win rate reasonable (40-60%): {p1_wins:.3f}",
        0.40 <= p1_wins <= 0.60,
        f"Player 1 win rate: {p1_wins:.3f}"
    )


def validate_decision_distribution(records):
    section("Decision Distribution")

    decisions = Counter(rec['decision'] for rec in records)
    total = len(records)
    stop_rate = decisions['stop'] / total
    cont_rate = decisions['continue'] / total

    print(f"  Stop:     {decisions['stop']:,} ({stop_rate:.1%})")
    print(f"  Continue: {decisions['continue']:,} ({cont_rate:.1%})")

    warn(
        "Stop/continue ratio is sensible (20-70% stops)",
        0.20 <= stop_rate <= 0.70,
        f"Stop rate: {stop_rate:.3f}"
    )

    # Early game vs late game decisions
    early = [r for r in records if r['score_active'] + r['score_opponent'] == 0]
    late  = [r for r in records if r['score_active'] + r['score_opponent'] >= 3]

    if early and late:
        early_stop = sum(1 for r in early if r['decision'] == 'stop') / len(early)
        late_stop  = sum(1 for r in late  if r['decision'] == 'stop') / len(late)
        print(f"  Early game stop rate: {early_stop:.1%}")
        print(f"  Late game stop rate:  {late_stop:.1%}")

        warn(
            "Late game stops more than early game (reasonable play)",
            late_stop >= early_stop * 0.8,
            f"Early: {early_stop:.1%}, Late: {late_stop:.1%}"
        )


def validate_position_diversity(records):
    section("Position Diversity")

    # Check unique dice combinations
    dice_combos = Counter(tuple(sorted(rec['dice'])) for rec in records)
    unique_dice = len(dice_combos)

    check(
        "Many unique dice combinations (>50)",
        unique_dice > 50,
        f"Unique dice combos: {unique_dice}"
    )

    # Check unique board positions (simplified)
    positions = set()
    for rec in records:
        pos_key = (
            rec['active_player'],
            tuple(sorted(rec['claimed_active'])),
            tuple(sorted(rec['claimed_opponent'])),
            tuple(sorted(rec['runners'].items())),
        )
        positions.add(pos_key)

    check(
        "High position diversity (>1000 unique positions)",
        len(positions) > 1000,
        f"Unique positions: {len(positions):,}"
    )

    # Check game phase distribution
    scores = [(r['score_active'], r['score_opponent']) for r in records]
    phase_counts = Counter(sum(s) for s in scores)
    print(f"\n  Game phase distribution (total columns claimed):")
    for phase in sorted(phase_counts.keys()):
        pct = 100 * phase_counts[phase] / len(records)
        bar = '█' * int(pct / 2)
        print(f"    {phase} claimed: {bar} {pct:.1f}%")


def validate_outcome_consistency(records):
    section("Outcome Consistency")

    # Group by game_id
    games = defaultdict(list)
    for rec in records:
        if 'game_id' in rec:
            games[rec['game_id']].append(rec)

    if not games:
        print("  SKIP  No game_id field — run with generate_sample()")
        return

    inconsistent = []
    for game_id, game_records in games.items():
        # All records from same game should have consistent outcomes
        # player 0 wins → all player 0 records should have outcome=1
        #               → all player 1 records should have outcome=0
        winners = set()
        for rec in game_records:
            if rec['outcome'] == 1:
                winners.add(rec['active_player'])
            else:
                losers_player = 1 - rec['active_player']

        if len(winners) > 1:
            inconsistent.append(game_id)

    check(
        "Outcomes consistent within each game",
        len(inconsistent) == 0,
        f"Inconsistent games: {inconsistent[:5]}"
    )

    # Verify exactly one winner per game
    multi_winner = []
    for game_id, game_records in list(games.items())[:100]:
        p0_outcomes = [r['outcome'] for r in game_records if r['active_player'] == 0]
        p1_outcomes = [r['outcome'] for r in game_records if r['active_player'] == 1]

        if p0_outcomes and p1_outcomes:
            p0_win = p0_outcomes[0]
            p1_win = p1_outcomes[0]
            if p0_win == p1_win:  # both winning or both losing is wrong
                multi_winner.append(game_id)

    check(
        "Exactly one winner per game (outcomes are zero-sum)",
        len(multi_winner) == 0,
        f"Games with non-zero-sum outcomes: {multi_winner[:5]}"
    )


def validate_move_legality(records, sample_size=200):
    section("Move Legality")

    illegal_moves = []
    sample = random.sample(records, min(sample_size, len(records)))

    for rec in sample:
        # Reconstruct state from record
        state = GameState(2)
        state.active_player = rec['active_player']
        state.dice = list(rec['dice'])

        # Restore progress
        state.progress[0] = {int(k): v for k, v in rec['progress_active'].items()} \
            if rec['active_player'] == 0 else \
            {int(k): v for k, v in rec['progress_opponent'].items()}
        state.progress[1] = {int(k): v for k, v in rec['progress_opponent'].items()} \
            if rec['active_player'] == 0 else \
            {int(k): v for k, v in rec['progress_active'].items()}

        # Restore claimed
        state.claimed[0] = set(rec['claimed_active']) if rec['active_player'] == 0 \
            else set(rec['claimed_opponent'])
        state.claimed[1] = set(rec['claimed_opponent']) if rec['active_player'] == 0 \
            else set(rec['claimed_active'])
        state.all_claimed = state.claimed[0] | state.claimed[1]

        if rec['runners']:
            first_key = next(iter(rec['runners']))
            state.runners = {int(k): v for k, v in rec['runners'].items()} \
                if isinstance(first_key, str) else dict(rec['runners'])
        else:
            state.runners = {}


        # Check move was legal
        valid = get_valid_moves(state)
        move = tuple(rec['move'])

        if valid and move not in valid:
            illegal_moves.append({
                'move': move,
                'valid': valid,
                'dice': rec['dice']
            })

    check(
        f"All sampled moves were legal ({sample_size} checked)",
        len(illegal_moves) == 0,
        f"Illegal moves found: {illegal_moves[:2]}"
    )


def validate_weighted_progress(records, sample_size=200):
    section("Weighted Progress Consistency")

    errors = []
    sample = random.sample(records, min(sample_size, len(records)))

    for rec in sample:
        if not rec['runners']:
            if rec['weighted_progress'] != 0:
                errors.append(('no runners but progress > 0', rec))
            continue

        # Manually calculate expected weighted progress
        player = rec['active_player']
        progress_active = {int(k): v for k, v in rec['progress_active'].items()}
        if rec['runners']:
            first_key = next(iter(rec['runners']))
            runners = {int(k): v for k, v in rec['runners'].items()} \
                if isinstance(first_key, str) else dict(rec['runners'])
        else:
            runners = {}

        expected = 0
        for col, runner_steps in runners.items():
            saved = progress_active.get(col, 0)
            total_steps = saved + runner_steps
            expected += total_steps / COLUMN_HEIGHTS[col]

        actual = rec['weighted_progress']
        if abs(actual - expected) > 0.001:
            errors.append(('mismatch', actual, expected))

    check(
        f"Weighted progress matches recorded state ({sample_size} checked)",
        len(errors) == 0,
        f"Errors: {errors[:2]}"
    )


# ---- MAIN ----
if __name__ == "__main__":
    print("=" * 55)
    print("  Can't Stop Training Data Validator")
    print("=" * 55)

    records = generate_sample(num_games=500)

    validate_schema(records)
    validate_value_ranges(records)
    validate_outcome_balance(records)
    validate_decision_distribution(records)
    validate_position_diversity(records)
    validate_outcome_consistency(records)
    validate_move_legality(records)
    validate_weighted_progress(records)

    print(f"\n{'='*55}")
    print(f"  Results: {passed} passed, {failed} failed, {warnings} warnings")
    if failed == 0:
        print(f"  Data quality: GOOD — safe to run overnight generation")
    else:
        print(f"  Data quality: ISSUES FOUND — fix before overnight run")
    print(f"{'='*55}\n")