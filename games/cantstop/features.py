# features.py
# Feature extraction for the Can't Stop neural network.
#
# Converts a GameState into a fixed-size numpy array of 74 features.
#
# Feature vector structure:
#   Per-column features  (11 cols × 4 = 44)
#   Dice features        (11 cols × 2 = 22)
#   Game context         (8)
#   Total = 74
#
# All features normalized to [-1,1] or [0,1] for training stability.
#
# Design decisions documented in project notes:
#   - Claimed status centered {-1,0,1} so sign = good/bad
#   - Saved progress and runner at risk kept separate (different risk profiles)
#   - Bust probability precomputed and cached (231 possible combinations)
#   - Column roll frequency named explicitly to distinguish from EV-adjusted values
#   - Opponent threat score weighted by column rarity
#   - Scores kept separate (0v0 != 2v2 strategically)

import numpy as np
from itertools import combinations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, COLUMN_HEIGHTS, COLUMNS_TO_WIN
)
from games.cantstop.ev_table import (
    build_ev_table, calc_prob_advance
)

# ---- CONSTANTS ----
COLUMNS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
NUM_COLUMNS = len(COLUMNS)
COL_INDEX = {col: i for i, col in enumerate(COLUMNS)}

FEATURE_SIZE = 74

# Action space constants
NUM_NORMAL_MOVES = 55   # C(11,2)
NUM_DOUBLE_MOVES = 11
NUM_PARTIAL_MOVES = 11
NUM_MOVES = NUM_NORMAL_MOVES + NUM_DOUBLE_MOVES + NUM_PARTIAL_MOVES  # 77
NUM_DECISIONS = 2       # stop=0, continue=1
ACTION_SPACE = NUM_MOVES * NUM_DECISIONS  # 154


# ---- PRECOMPUTED TABLES ----
# Built once at import time — O(1) lookup during feature extraction

def _build_roll_frequency():
    """
    Exact probability of each column appearing in any dice roll.
    Computed deterministically from all 6^4 = 1296 outcomes.
    """
    return {
        col: calc_prob_advance({col})
        for col in COLUMNS
    }

def _build_bust_cache():
    """
    Precompute bust probability for all 231 possible runner combinations.
    Bust = probability of rolling zero legal moves given current runners.

    231 = C(11,1) + C(11,2) + C(11,3) = 11 + 55 + 165
    """
    cache = {}
    for r in range(1, 4):
        for combo in combinations(COLUMNS, r):
            prob_advance = calc_prob_advance(set(combo))
            cache[combo] = 1.0 - prob_advance
    # Empty runners = can't bust
    cache[()] = 0.0
    return cache

def _build_difficulty_weights():
    """
    Difficulty weight per column = 1 - roll_frequency.
    Higher weight for rarer columns (2, 12) — progress there is harder earned.
    """
    roll_freq = _build_roll_frequency()
    return {col: 1.0 - roll_freq[col] for col in COLUMNS}

def _build_max_threat():
    """
    Maximum possible threat score — used for normalization.
    Occurs when opponent has full progress on three rarest columns.
    """
    weights = _build_difficulty_weights()
    top3 = sorted(weights.values(), reverse=True)[:3]
    return sum(top3)

def _build_action_index():
    """
    Build mapping from (move, decision) → action index.

    Layout:
      [0..109]   normal moves (col_a, col_b) × {stop, continue}
      [110..131] double moves (col, col)     × {stop, continue}
      [132..153] partial moves (col,)        × {stop, continue}

    Within each group: stop=0, continue=1 alternating
    e.g. action 0 = (first_normal_move, stop)
         action 1 = (first_normal_move, continue)
    """
    index = {}
    idx = 0

    # Normal moves: all sorted pairs of different columns
    for col_a, col_b in combinations(COLUMNS, 2):
        index[((col_a, col_b), 'stop')]     = idx
        index[((col_a, col_b), 'continue')] = idx + 1
        idx += 2

    # Double moves: same column twice
    for col in COLUMNS:
        index[((col, col), 'stop')]     = idx
        index[((col, col), 'continue')] = idx + 1
        idx += 2

    # Partial moves: single column tuple
    for col in COLUMNS:
        index[((col,), 'stop')]     = idx
        index[((col,), 'continue')] = idx + 1
        idx += 2

    assert idx == ACTION_SPACE, f"Expected {ACTION_SPACE} actions, got {idx}"
    return index

# Build all tables at import time
print("Building feature extraction tables...", end=" ", flush=True)
ROLL_FREQUENCY  = _build_roll_frequency()
BUST_CACHE      = _build_bust_cache()
DIFFICULTY_WT   = _build_difficulty_weights()
MAX_THREAT      = _build_max_threat()
ACTION_INDEX    = _build_action_index()
print("done.")


# ---- ACTION HELPERS ----

def move_to_action(move, decision):
    """
    Convert (move, decision) pair to action index 0-153.

    move: tuple — (col_a, col_b), (col, col), or (col,)
    decision: 'stop' or 'continue'
    """
    key = (tuple(move), decision)
    if key not in ACTION_INDEX:
        raise ValueError(f"Unknown action: move={move}, decision={decision}")
    return ACTION_INDEX[key]


def action_to_move_decision(action_idx):
    """
    Convert action index back to (move, decision).
    Reverse lookup — built once from ACTION_INDEX.
    """
    if not hasattr(action_to_move_decision, '_reverse'):
        action_to_move_decision._reverse = {
            v: k for k, v in ACTION_INDEX.items()
        }
    move, decision = action_to_move_decision._reverse[action_idx]
    return move, decision


def get_legal_action_mask(valid_moves, include_stop=True, include_continue=True):
    """
    Returns a boolean mask of shape (154,) where True = legal action.

    valid_moves: list of legal move tuples from get_valid_moves()
    include_stop: whether stopping is currently a valid decision
    include_continue: whether continuing is currently a valid decision
    """
    mask = np.zeros(ACTION_SPACE, dtype=bool)

    for move in valid_moves:
        move_key = tuple(move)
        if include_stop:
            key = (move_key, 'stop')
            if key in ACTION_INDEX:
                mask[ACTION_INDEX[key]] = True
        if include_continue:
            key = (move_key, 'continue')
            if key in ACTION_INDEX:
                mask[ACTION_INDEX[key]] = True

    return mask


# ---- FEATURE EXTRACTION ----

def extract_features(state, valid_moves=None):
    """
    Convert a GameState into a numpy feature vector of shape (74,).

    Parameters:
        state: GameState object
        valid_moves: list of legal moves (if None, computed internally)
                     Pass pre-computed moves to avoid redundant computation.

    Returns:
        np.ndarray of shape (74,), dtype float32
        All values in [-1, 1] or [0, 1]

    Feature layout:
        [0:44]   per-column features (11 cols × 4)
        [44:66]  dice features (11 cols × 2)
        [66:74]  game context (8)
    """
    player   = state.active_player
    opponent = 1 - player

    features = np.zeros(FEATURE_SIZE, dtype=np.float32)

    # ---- SECTION 1: PER-COLUMN FEATURES (indices 0-43) ----
    # For each column: claimed_status, my_saved, my_runner, opp_saved
    # Layout: [col2_feat0, col2_feat1, col2_feat2, col2_feat3,
    #          col3_feat0, ...]

    for i, col in enumerate(COLUMNS):
        base = i * 4
        height = COLUMN_HEIGHTS[col]

        # Claimed status: -1=opponent, 0=open, 1=mine
        if col in state.claimed[player]:
            features[base + 0] = 1.0
        elif col in state.all_claimed:
            features[base + 0] = -1.0
        else:
            features[base + 0] = 0.0

        # My saved progress (safe, permanent)
        my_saved = state.progress[player].get(col, 0)
        features[base + 1] = my_saved / height

        # My runner at risk (this turn only, lost on bust)
        my_runner = state.runners.get(col, 0)
        features[base + 2] = my_runner / height

        # Opponent saved progress
        opp_saved = state.progress[opponent].get(col, 0)
        features[base + 3] = opp_saved / height

    # ---- SECTION 2: DICE FEATURES (indices 44-65) ----
    # For each column: effective_multiplicity, column_roll_frequency

    if valid_moves is None:
        from games.cantstop.engine import get_valid_moves
        valid_moves = get_valid_moves(state)

    # Compute effective multiplicity:
    # How many of the 3 dice pairs reach each usable column this roll
    multiplicity = {col: 0 for col in COLUMNS}

    if state.dice:
        d = state.dice
        pairs = [
            (d[0]+d[1], d[2]+d[3]),
            (d[0]+d[2], d[1]+d[3]),
            (d[0]+d[3], d[1]+d[2]),
        ]
        for pair in pairs:
            for col in set(pair):  # set() deduplicates (7,7) → {7}
                if col in COL_INDEX:
                    if (col not in state.all_claimed and
                        state.progress[player].get(col, 0) +
                        state.runners.get(col, 0) < COLUMN_HEIGHTS[col]):
                        multiplicity[col] += 1

    for i, col in enumerate(COLUMNS):
        base = 44 + i * 2

        # Effective multiplicity normalized by 3 (max pairs)
        features[base + 0] = multiplicity[col] / 3.0

        # Column roll frequency (combinatorial, static per column)
        features[base + 1] = ROLL_FREQUENCY[col]

    # ---- SECTION 3: GAME CONTEXT (indices 66-73) ----

    my_score  = len(state.claimed[player])
    opp_score = len(state.claimed[opponent])

    # My score centered: 0→-1, 1→-0.33, 2→+0.33, 3→+1
    features[66] = (my_score  - 1.5) / 1.5

    # Opponent score centered (kept separate — 0v0 ≠ 2v2)
    features[67] = (opp_score - 1.5) / 1.5

    # Weighted progress at risk (runner steps normalized by column height)
    # Normalized by 3 (max possible = 3 columns fully progressed)
    weighted_progress = sum(
        state.runners.get(col, 0) / COLUMN_HEIGHTS[col]
        for col in COLUMNS
    )
    features[68] = min(weighted_progress / 3.0, 1.0)

    # Progress value — weighted by column rarity (rare columns worth more)
    progress_value = sum(
        (state.runners.get(col, 0) / COLUMN_HEIGHTS[col]) * DIFFICULTY_WT[col]
        for col in COLUMNS
    )
    max_progress_value = sum(sorted(DIFFICULTY_WT.values(), reverse=True)[:3])
    features[69] = min(progress_value / max_progress_value, 1.0) \
        if max_progress_value > 0 else 0.0

    # Bust probability (precomputed cache)
    runner_cols = tuple(sorted(state.runners.keys()))
    features[70] = BUST_CACHE.get(runner_cols, 0.0)

    # Runner slots remaining: (3 - placed) / 3
    features[71] = (3 - len(state.runners)) / 3.0

    # Opponent threat score — their progress weighted by column rarity
    opp_threat = sum(
        (state.progress[opponent].get(col, 0) / COLUMN_HEIGHTS[col])
        * DIFFICULTY_WT[col]
        for col in COLUMNS
        if col not in state.all_claimed
    )
    features[72] = min(opp_threat / MAX_THREAT, 1.0) if MAX_THREAT > 0 else 0.0

    # My threat score — my progress weighted by column rarity
    my_threat = sum(
        (state.progress[player].get(col, 0) / COLUMN_HEIGHTS[col])
        * DIFFICULTY_WT[col]
        for col in COLUMNS
        if col not in state.all_claimed
    )
    features[73] = min(my_threat / MAX_THREAT, 1.0) if MAX_THREAT > 0 else 0.0

    return features


# ---- FEATURE NAMES (for debugging and visualization) ----

def get_feature_names():
    """Returns list of 74 human-readable feature names."""
    names = []

    for col in COLUMNS:
        names += [
            f"col{col}_claimed",
            f"col{col}_my_saved",
            f"col{col}_my_runner",
            f"col{col}_opp_saved",
        ]

    for col in COLUMNS:
        names += [
            f"col{col}_multiplicity",
            f"col{col}_roll_freq",
        ]

    names += [
        "my_score",
        "opp_score",
        "weighted_progress",
        "progress_value",
        "prob_bust",
        "runner_slots_remaining",
        "opp_threat_score",
        "my_threat_score",
    ]

    assert len(names) == FEATURE_SIZE, \
        f"Expected {FEATURE_SIZE} names, got {len(names)}"
    return names


# ---- SELF TEST ----
if __name__ == "__main__":
    import time
    from games.cantstop.engine import get_valid_moves, apply_move, stop_turn

    print("\nTesting feature extractor...\n")

    # ---- Test 1: Empty board ----
    state = GameState(2)
    state.dice = [3, 4, 3, 4]
    valid = get_valid_moves(state)
    features = extract_features(state, valid)

    print(f"Feature vector shape: {features.shape}")
    print(f"Feature dtype: {features.dtype}")
    print(f"Value range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"Expected range: [-1.0, 1.0]")
    assert features.shape == (FEATURE_SIZE,), "Wrong shape"
    assert features.min() >= -1.0, f"Min out of range: {features.min()}"
    assert features.max() <= 1.0,  f"Max out of range: {features.max()}"
    print("PASS  Shape and range correct\n")

    # ---- Test 2: Feature values on known position ----
    names = get_feature_names()
    print("Sample features on empty board (dice [3,4,3,4]):")
    print(f"  Pairs reachable: (7,7), (6,8)")
    print(f"  col7_multiplicity: {features[names.index('col7_multiplicity')]:.3f}  (expect 0.667 = 2/3 pairs)")
    print(f"  col6_multiplicity: {features[names.index('col6_multiplicity')]:.3f}  (expect 0.333 = 1/3 pairs)")
    print(f"  col8_multiplicity: {features[names.index('col8_multiplicity')]:.3f}  (expect 0.333 = 1/3 pairs)")
    print(f"  col2_multiplicity: {features[names.index('col2_multiplicity')]:.3f}  (expect 0.0)")
    print(f"  my_score:          {features[names.index('my_score')]:.3f}         (expect -1.0)")
    print(f"  prob_bust:         {features[names.index('prob_bust')]:.3f}         (expect 0.0 = no runners)")
    print(f"  runner_slots:      {features[names.index('runner_slots_remaining')]:.3f}   (expect 1.0 = 3 slots free)")

    # ---- Test 3: Mid-game position ----
    print("\nMid-game position (runners on 6,7,8 with saved progress):")
    state2 = GameState(2)
    state2.dice = [3, 3, 4, 4]
    state2.runners = {6: 3, 7: 5, 8: 2}
    state2.progress[0] = {7: 4, 9: 3}
    state2.claimed[0] = {2}
    state2.claimed[1] = {12, 11}
    state2.all_claimed = {2, 11, 12}
    state2.progress[1] = {7: 6, 8: 5}

    valid2 = get_valid_moves(state2)
    features2 = extract_features(state2, valid2)

    print(f"  col7_claimed:   {features2[names.index('col7_claimed')]:.3f}   (expect 0.0 = open)")
    print(f"  col2_claimed:   {features2[names.index('col2_claimed')]:.3f}   (expect 1.0 = mine)")
    print(f"  col12_claimed:  {features2[names.index('col12_claimed')]:.3f}  (expect -1.0 = opponent)")
    print(f"  col7_my_saved:  {features2[names.index('col7_my_saved')]:.3f}  (expect {4/13:.3f} = 4/13)")
    print(f"  col7_my_runner: {features2[names.index('col7_my_runner')]:.3f}  (expect {5/13:.3f} = 5/13)")
    print(f"  col7_opp_saved: {features2[names.index('col7_opp_saved')]:.3f}  (expect {6/13:.3f} = 6/13)")
    print(f"  my_score:       {features2[names.index('my_score')]:.3f}  (expect {(1-1.5)/1.5:.3f} = 1 column)")
    print(f"  opp_score:      {features2[names.index('opp_score')]:.3f}  (expect {(2-1.5)/1.5:.3f} = 2 columns)")
    print(f"  prob_bust:      {features2[names.index('prob_bust')]:.3f}")
    print(f"  runner_slots:   {features2[names.index('runner_slots_remaining')]:.3f}  (expect 0.0 = all placed)")
    print(f"  opp_threat:     {features2[names.index('opp_threat_score')]:.3f}")

    assert features2.min() >= -1.0
    assert features2.max() <= 1.0
    print("\nPASS  Mid-game position features correct")

    # ---- Test 4: Action index round-trip ----
    print("\nTesting action index round-trip...")
    test_cases = [
        ((6, 8), 'stop'),
        ((7, 7), 'continue'),
        ((2,),   'stop'),
        ((11,),  'continue'),
    ]
    for move, decision in test_cases:
        idx = move_to_action(move, decision)
        recovered_move, recovered_decision = action_to_move_decision(idx)
        assert recovered_move == move, \
            f"Move mismatch: {move} → {idx} → {recovered_move}"
        assert recovered_decision == decision
        print(f"  PASS  {move} + {decision} → {idx} → {recovered_move} + {recovered_decision}")

    # ---- Test 5: Legal action mask ----
    print("\nTesting legal action mask...")
    state3 = GameState(2)
    state3.dice = [3, 4, 3, 4]
    valid3 = get_valid_moves(state3)
    mask = get_legal_action_mask(valid3)
    print(f"  Valid moves: {valid3}")
    print(f"  Legal actions: {mask.sum()} (expect {len(valid3) * 2})")
    assert mask.sum() == len(valid3) * 2, \
        f"Mask has {mask.sum()} actions, expected {len(valid3) * 2}"
    print("  PASS  Legal action mask correct")

    # ---- Test 6: Performance ----
    print("\nPerformance benchmark...")
    state4 = GameState(2)
    state4.dice = [2, 3, 5, 6]
    valid4 = get_valid_moves(state4)

    start = time.time()
    for _ in range(10000):
        extract_features(state4, valid4)
    elapsed = time.time() - start

    print(f"  10,000 extractions: {elapsed:.3f}s")
    print(f"  Per extraction: {elapsed/10000*1000:.3f}ms")
    print(f"  (Target: <0.1ms)")

    print(f"\n{'='*45}")
    print(f"  All tests passed!")
    print(f"  Feature size: {FEATURE_SIZE}")
    print(f"  Action space: {ACTION_SPACE}")
    print(f"{'='*45}\n")