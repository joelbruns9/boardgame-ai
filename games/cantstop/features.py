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
#   - Saved progress and runner at risk kept separate
#   - Bust probability precomputed and cached (231 possible runner combos)
#   - Column roll frequency named explicitly to distinguish from EV values
#   - Opponent threat score weighted by column rarity
#   - Scores kept separate (0v0 != 2v2 strategically)
#
# Hot-path optimizations applied vs prior version:
#   - Hoist attribute lookups (state.progress[player], state.runners, etc.)
#     to local variables in extract_features's tight per-column loop.
#   - Precompute MAX_PROGRESS_VALUE (was recomputed every call).
#   - Build ACTION_INDEX_REVERSE eagerly at module load (was lazy with
#     function-attribute trick that added a hasattr check per call).
#   - Build MOVE_TO_ACTION_PAIR map (move_key -> (stop_idx, continue_idx))
#     so get_legal_action_mask does ONE dict lookup per valid move
#     instead of two, plus avoids constructing string keys.
#   - Lift the lazy import of get_valid_moves to module top.
#   - Replace `set(pair)` per-iteration allocation with explicit handling
#     of the (a, a) double case.
#   - Remove the dead `col in COL_INDEX` check (pair sums are always in
#     range 2..12 = COLUMNS by dice math).
#
# No correctness changes — the engine bug-fix that emits more partial
# moves (col,) is already handled by the existing partial-move slots in
# the action space (indices 132..153).

import numpy as np
from itertools import combinations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, COLUMN_HEIGHTS, COLUMNS_TO_WIN,
    get_valid_moves,  # lifted from inside extract_features for hot-path use
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


def _build_move_to_pair_index(action_index):
    """
    Build move_key -> (stop_idx, continue_idx) for fast mask construction.

    This lets get_legal_action_mask do ONE dict lookup per valid move
    instead of building two ((move, 'stop'), (move, 'continue')) tuples
    and doing two lookups.
    """
    pairs = {}
    # Reconstruct from action_index: for each move key, find its stop and
    # continue indices.
    for (move_key, decision), idx in action_index.items():
        existing = pairs.get(move_key)
        if existing is None:
            existing = [None, None]
            pairs[move_key] = existing
        if decision == 'stop':
            existing[0] = idx
        else:
            existing[1] = idx
    # Freeze to tuples (cheaper to index, immutable).
    return {k: tuple(v) for k, v in pairs.items()}


def _build_reverse_action_index(action_index):
    """Eagerly build the action_idx -> (move, decision) reverse map."""
    return {v: k for k, v in action_index.items()}


# Build all tables at import time
print("Building feature extraction tables...", end=" ", flush=True)
ROLL_FREQUENCY  = _build_roll_frequency()
BUST_CACHE      = _build_bust_cache()
DIFFICULTY_WT   = _build_difficulty_weights()
MAX_THREAT      = _build_max_threat()
ACTION_INDEX    = _build_action_index()

# Derived tables — built once, used in hot paths.
MOVE_TO_ACTION_PAIR = _build_move_to_pair_index(ACTION_INDEX)
ACTION_INDEX_REVERSE = _build_reverse_action_index(ACTION_INDEX)

# Precomputed normalization constants (were recomputed per call before).
MAX_PROGRESS_VALUE = sum(sorted(DIFFICULTY_WT.values(), reverse=True)[:3])

# Per-column constants as parallel arrays for tight loops.
# These are 11-element lookups indexed by COL_INDEX position.
_HEIGHTS_BY_POS  = [COLUMN_HEIGHTS[c] for c in COLUMNS]
_INV_HEIGHTS     = [1.0 / h for h in _HEIGHTS_BY_POS]  # multiply > divide
_ROLL_FREQ_BY_POS = [ROLL_FREQUENCY[c] for c in COLUMNS]
_DIFFICULTY_BY_POS = [DIFFICULTY_WT[c] for c in COLUMNS]

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
    Uses precomputed reverse map — O(1) lookup, no first-call overhead.
    """
    return ACTION_INDEX_REVERSE[action_idx]


def get_legal_action_mask(valid_moves, include_stop=True, include_continue=True):
    """
    Returns a boolean mask of shape (154,) where True = legal action.

    valid_moves: list of legal move tuples from get_valid_moves()
    include_stop: whether stopping is currently a valid decision
    include_continue: whether continuing is currently a valid decision

    Optimization: uses precomputed MOVE_TO_ACTION_PAIR map so each
    move requires ONE dict lookup yielding both (stop_idx, cont_idx),
    instead of building 2 composite-key tuples and doing 2 lookups.
    """
    mask = np.zeros(ACTION_SPACE, dtype=bool)
    pair_map = MOVE_TO_ACTION_PAIR  # local-bind

    for move in valid_moves:
        move_key = move if isinstance(move, tuple) else tuple(move)
        idx_pair = pair_map.get(move_key)
        if idx_pair is None:
            continue
        stop_idx, cont_idx = idx_pair
        if include_stop and stop_idx is not None:
            mask[stop_idx] = True
        if include_continue and cont_idx is not None:
            mask[cont_idx] = True

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
    # ---- Local-bind hot lookups (each access becomes faster) ----
    player   = state.active_player
    opponent = 1 - player
    claimed         = state.claimed
    claimed_mine    = claimed[player]
    all_claimed     = state.all_claimed
    progress_mine   = state.progress[player]
    progress_opp    = state.progress[opponent]
    runners         = state.runners
    dice            = state.dice

    # Locally-bound parallel arrays for the per-column loop.
    columns          = COLUMNS
    col_index        = COL_INDEX
    heights          = _HEIGHTS_BY_POS
    inv_heights      = _INV_HEIGHTS
    roll_freq_arr    = _ROLL_FREQ_BY_POS
    difficulty_arr   = _DIFFICULTY_BY_POS

    features = np.zeros(FEATURE_SIZE, dtype=np.float32)

    # ---- SECTION 1: PER-COLUMN FEATURES (indices 0-43) ----
    # For each column: claimed_status, my_saved, my_runner, opp_saved.
    #
    # Optimization: skip the assignment when the value is zero (which
    # is the default from np.zeros init). Wins ~3x on empty/early-game
    # states where most columns have no progress. Mid-game and late-
    # game positions get no measurable speedup but no penalty either.
    for i in range(NUM_COLUMNS):
        col   = columns[i]
        inv_h = inv_heights[i]
        base  = i * 4

        # Claimed status: -1=opponent, 0=open, 1=mine.
        if col in claimed_mine:
            features[base] = 1.0
        elif col in all_claimed:
            features[base] = -1.0
        # else implicitly 0.0 from np.zeros init

        # Skip writes when the value is zero — features[...] is already
        # 0.0 from np.zeros, so no-write is identical to write-zero but
        # avoids the numpy __setitem__ overhead.
        my_saved = progress_mine.get(col, 0)
        if my_saved:
            features[base + 1] = my_saved * inv_h

        my_runner = runners.get(col, 0)
        if my_runner:
            features[base + 2] = my_runner * inv_h

        opp_saved = progress_opp.get(col, 0)
        if opp_saved:
            features[base + 3] = opp_saved * inv_h

    # ---- SECTION 2: DICE FEATURES (indices 44-65) ----
    # For each column: effective_multiplicity, column_roll_frequency

    if valid_moves is None:
        valid_moves = get_valid_moves(state)  # import lifted to module top

    # Compute effective multiplicity:
    # How many of the 3 dice pair-partitions reach each usable column.
    # multiplicity[col] in [0, 3].
    multiplicity = [0] * NUM_COLUMNS

    if dice:
        d0, d1, d2, d3 = dice
        # Three pairings; for each pairing, increment multiplicity for
        # each column-sum it produces. Set-dedup is replaced by an
        # explicit equality check for the (a == b) double case.
        for sa, sb in (
            (d0 + d1, d2 + d3),
            (d0 + d2, d1 + d3),
            (d0 + d3, d1 + d2),
        ):
            # col is always in 2..12 = COLUMNS, so the `col in COL_INDEX`
            # check from the prior version was dead.
            # Check usability for col sa.
            if (sa not in all_claimed and
                progress_mine.get(sa, 0) + runners.get(sa, 0) < COLUMN_HEIGHTS[sa]):
                multiplicity[col_index[sa]] += 1
            if sa != sb:
                if (sb not in all_claimed and
                    progress_mine.get(sb, 0) + runners.get(sb, 0) < COLUMN_HEIGHTS[sb]):
                    multiplicity[col_index[sb]] += 1
            # If sa == sb (true double from dice), the second column is
            # the same as the first — already counted once. This matches
            # the old `set(pair)` dedup semantics.

    for i in range(NUM_COLUMNS):
        base = 44 + i * 2
        features[base]     = multiplicity[i] / 3.0
        features[base + 1] = roll_freq_arr[i]

    # ---- SECTION 3: GAME CONTEXT (indices 66-73) ----

    my_score  = len(claimed_mine)
    opp_score = len(claimed[opponent])

    # Score centered: 0→-1, 1→-0.33, 2→+0.33, 3→+1
    features[66] = (my_score  - 1.5) / 1.5
    features[67] = (opp_score - 1.5) / 1.5

    # Weighted progress at risk (runner steps normalized by column height)
    # and progress value (weighted by rarity). Fuse the two loops since
    # both iterate over runners.
    weighted_progress = 0.0
    progress_value    = 0.0
    for col, steps in runners.items():
        i = col_index.get(col)
        if i is None:
            continue
        inv_h = inv_heights[i]
        runner_frac = steps * inv_h
        weighted_progress += runner_frac
        progress_value    += runner_frac * difficulty_arr[i]

    # Normalize: max possible = 3 columns at full progress.
    if weighted_progress > 3.0:
        weighted_progress = 3.0
    features[68] = weighted_progress / 3.0

    if MAX_PROGRESS_VALUE > 0:
        v = progress_value / MAX_PROGRESS_VALUE
        features[69] = v if v < 1.0 else 1.0

    # Bust probability (precomputed cache).
    # sorted() on dict.keys() handles dicts of any size; for 0-3 keys
    # this is cheap. Use tuple() to match cache keys.
    runner_cols = tuple(sorted(runners))
    features[70] = BUST_CACHE.get(runner_cols, 0.0)

    # Runner slots remaining: (3 - placed) / 3
    features[71] = (3 - len(runners)) * (1.0 / 3.0)

    # Threat scores — fuse the two opp/my loops since both iterate COLUMNS
    # filtered by all_claimed. We iterate progress dicts directly instead
    # of all COLUMNS since most columns have zero progress.
    opp_threat = 0.0
    for col, steps in progress_opp.items():
        if col in all_claimed:
            continue
        i = col_index.get(col)
        if i is None:
            continue
        opp_threat += steps * inv_heights[i] * difficulty_arr[i]

    my_threat = 0.0
    for col, steps in progress_mine.items():
        if col in all_claimed:
            continue
        i = col_index.get(col)
        if i is None:
            continue
        my_threat += steps * inv_heights[i] * difficulty_arr[i]

    if MAX_THREAT > 0:
        inv_max_threat = 1.0 / MAX_THREAT
        v = opp_threat * inv_max_threat
        features[72] = v if v < 1.0 else 1.0
        v = my_threat * inv_max_threat
        features[73] = v if v < 1.0 else 1.0

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
    names = get_feature_names()

    # ---- Test 1: Empty board, shape and range ----
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

    # ---- Test 2: Known feature values on empty board ----
    print("Sample features on empty board (dice [3,4,3,4]):")
    print(f"  Pairs reachable: (7,7), (6,8), and (7,7) again")
    expected_col7 = 2 / 3.0
    print(f"  col7_multiplicity: {features[names.index('col7_multiplicity')]:.3f}  (expect {expected_col7:.3f})")
    assert abs(features[names.index('col7_multiplicity')] - expected_col7) < 1e-6
    expected_col6 = 1 / 3.0
    print(f"  col6_multiplicity: {features[names.index('col6_multiplicity')]:.3f}  (expect {expected_col6:.3f})")
    assert abs(features[names.index('col6_multiplicity')] - expected_col6) < 1e-6
    assert features[names.index('col2_multiplicity')] == 0.0
    assert features[names.index('my_score')] == -1.0
    assert features[names.index('prob_bust')] == 0.0
    assert features[names.index('runner_slots_remaining')] == 1.0
    print(f"  All empty-board invariants: PASS")

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

    assert features2[names.index('col7_claimed')]  == 0.0
    assert features2[names.index('col2_claimed')]  == 1.0
    assert features2[names.index('col12_claimed')] == -1.0
    assert abs(features2[names.index('col7_my_saved')]  - 4/13) < 1e-6
    assert abs(features2[names.index('col7_my_runner')] - 5/13) < 1e-6
    assert abs(features2[names.index('col7_opp_saved')] - 6/13) < 1e-6
    assert abs(features2[names.index('my_score')]  - (1 - 1.5) / 1.5) < 1e-6
    assert abs(features2[names.index('opp_score')] - (2 - 1.5) / 1.5) < 1e-6
    assert features2[names.index('runner_slots_remaining')] == 0.0
    assert features2.min() >= -1.0
    assert features2.max() <= 1.0
    print("  All mid-game invariants: PASS")

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
        print(f"  PASS  {move} + {decision} → {idx}")

    # ---- Test 5: Legal action mask ----
    print("\nTesting legal action mask...")
    state3 = GameState(2)
    state3.dice = [3, 4, 3, 4]
    valid3 = get_valid_moves(state3)
    mask = get_legal_action_mask(valid3)
    print(f"  Valid moves: {valid3}")
    print(f"  Legal actions: {mask.sum()} (expect {len(valid3) * 2})")
    assert mask.sum() == len(valid3) * 2
    print("  PASS")

    # ---- Test 6: Partial move handling (post-engine-bugfix) ----
    print("\nTesting partial moves in mask (engine bug fix coverage)...")
    state6 = GameState(2)
    state6.dice = [1, 2, 1, 3]
    state6.runners = {5: 1, 9: 1}  # 2 runners already placed
    valid6 = get_valid_moves(state6)
    print(f"  Valid moves under cap: {sorted(valid6)}")
    assert (3,) in valid6 or (4,) in valid6, \
        f"Expected partials in valid: {valid6}"
    mask6 = get_legal_action_mask(valid6)
    # Each partial occupies 2 action slots (stop+continue)
    for partial in valid6:
        if len(partial) == 1:
            stop_idx = move_to_action(partial, 'stop')
            cont_idx = move_to_action(partial, 'continue')
            assert mask6[stop_idx], f"stop bit missing for {partial}"
            assert mask6[cont_idx], f"continue bit missing for {partial}"
    print("  Partial moves correctly encoded in mask: PASS")

    # ---- Test 7: BUST_CACHE coverage ----
    print("\nTesting BUST_CACHE coverage...")
    expected = 1 + 11 + 55 + 165   # () + C(11,1) + C(11,2) + C(11,3)
    assert len(BUST_CACHE) == expected, \
        f"BUST_CACHE has {len(BUST_CACHE)}, expected {expected}"
    # Every possible 0-3 runner combo is present
    for r in range(1, 4):
        for combo in combinations(COLUMNS, r):
            assert combo in BUST_CACHE, f"Missing {combo}"
    assert () in BUST_CACHE
    print(f"  BUST_CACHE has all {expected} entries: PASS")

    # ---- Test 8: Determinism — same state, same features ----
    state8 = GameState(2)
    state8.dice = [2, 3, 5, 6]
    state8.runners = {7: 3}
    state8.progress[0] = {5: 2}
    state8.claimed[0] = {2}
    state8.all_claimed = {2}
    v8 = get_valid_moves(state8)
    f8a = extract_features(state8, v8)
    f8b = extract_features(state8, v8)
    assert np.array_equal(f8a, f8b), "Non-deterministic features!"
    print("\nDeterminism: PASS")

    # ---- Test 9: Performance ----
    print("\nPerformance benchmark...")
    state4 = GameState(2)
    state4.dice = [2, 3, 5, 6]
    valid4 = get_valid_moves(state4)

    N = 10000
    start = time.time()
    for _ in range(N):
        extract_features(state4, valid4)
    elapsed = time.time() - start
    print(f"  {N:,} extract_features:   {elapsed*1000:7.1f} ms "
          f"({elapsed*1000/N*1000:.1f} µs/call)")

    start = time.time()
    for _ in range(N):
        get_legal_action_mask(valid4)
    mask_elapsed = time.time() - start
    print(f"  {N:,} get_legal_action_mask: {mask_elapsed*1000:7.1f} ms "
          f"({mask_elapsed*1000/N*1000:.1f} µs/call)")

    start = time.time()
    for _ in range(N):
        move_to_action((6, 8), 'stop')
    a2m_elapsed = time.time() - start
    print(f"  {N:,} move_to_action:     {a2m_elapsed*1000:7.1f} ms")

    start = time.time()
    for _ in range(N):
        action_to_move_decision(42)
    inv_elapsed = time.time() - start
    print(f"  {N:,} action_to_move:     {inv_elapsed*1000:7.1f} ms")

    print(f"\n{'='*45}")
    print(f"  All tests passed!")
    print(f"  Feature size: {FEATURE_SIZE}")
    print(f"  Action space: {ACTION_SPACE}")
    print(f"{'='*45}\n")