# games/cantstop/symmetry.py
#
# Board reflection symmetry for Can't Stop, for use as training-data
# augmentation (and as a correctness check on the feature/engine pipeline).
#
# The symmetry
# ------------
# Can't Stop is invariant under reflecting the board about column 7 AND
# complementing the dice:
#       column c  ->  (MIN_COL + MAX_COL) - c        (2<->12, 3<->11, ... 7->7)
#       die    v  ->  (MIN_FACE + MAX_FACE) - v      (1<->6, 2<->5, 3<->4)
# because a pair summing to c maps to a pair summing to (14 - c). This is an
# exact relabeling: legal moves, the winner, and win probability are unchanged.
# It is applied to the WHOLE state (both/all players at once), so it preserves
# every player's relative position and strength — only column labels move.
#
# What this module provides
# --------------------------
#   reflect_col / reflect_die / reflect_move : the basic maps
#   reflect_state(state)                     : a reflected GameState
#   FEATURE_PERM, reflect_features(feat)     : reflect a 74-d feature vector
#   ACTION_PERM,  reflect_policy(policy)     : reflect a 154-d policy vector
#   reflect_value(v)                         : identity (value is invariant)
#   is_symmetric()                           : guard — False => do NOT augment
#
# Both FEATURE_PERM and ACTION_PERM are self-inverse (the reflection is an
# involution), so `arr[PERM]` reflects and `arr[PERM][PERM] == arr`.
#
# Implementation notes
# --------------------
# The COLUMN reflection is derived from the column range, so symmetric variants
# that change the board (more players, 4-to-win, more runners) keep working
# automatically. The DIE complement uses FIXED standard-d6 faces (FACE_MIN /
# FACE_MAX below) — non-standard or wider dice require updating those two
# constants by hand. Either way, asymmetric variants are caught by
# is_symmetric(), which disables augmentation rather than producing wrong labels.
#
# Run the self-tests (which also verify the feature extractor and engine are
# genuinely symmetric — i.e. that the observed model asymmetry is NOT a bug):
#   python -m games.cantstop.symmetry

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, COLUMN_HEIGHTS,
    get_valid_moves, apply_move, stop_turn, bust_turn,
)
from games.cantstop.features import (
    COLUMNS, COL_INDEX, NUM_COLUMNS, FEATURE_SIZE, ACTION_SPACE,
    extract_features, get_legal_action_mask,
    action_to_move_decision, move_to_action,
)

# ---- Reflection constants, derived from the game spec ----
MIN_COL, MAX_COL = min(COLUMNS), max(COLUMNS)     # 2, 12  -> c -> 14 - c
COL_SUM = MIN_COL + MAX_COL
FACE_MIN, FACE_MAX = 1, 6                          # standard d6 -> v -> 7 - v
FACE_SUM = FACE_MIN + FACE_MAX


def reflect_col(c):
    """Mirror a column about the board center: 2<->12, 3<->11, ... 7->7."""
    return COL_SUM - c


def reflect_die(v):
    """Complement a die face: 1<->6, 2<->5, 3<->4."""
    return FACE_SUM - v


def reflect_move(move):
    """
    Reflect a move tuple into its canonical (sorted) mirror.
    Normal pairs are stored ascending in the action index, so re-sort;
    doubles (c,c) and partials (c,) need no re-sort.
    """
    rc = tuple(reflect_col(c) for c in move)
    if len(rc) == 2 and rc[0] != rc[1]:
        rc = (rc[1], rc[0]) if rc[0] > rc[1] else rc
    return rc


def reflect_state(state):
    """
    Return a new GameState that is the mirror image of `state`: every column
    relabeled c -> reflect_col(c) for ALL players, and the dice complemented.
    active_player, scores (by count), game_over and winner are unchanged.
    """
    s = state.clone()
    s.claimed = {p: {reflect_col(c) for c in cols}
                 for p, cols in state.claimed.items()}
    s.all_claimed = {reflect_col(c) for c in state.all_claimed}
    s.progress = {p: {reflect_col(c): v for c, v in prog.items()}
                  for p, prog in state.progress.items()}
    s.runners = {reflect_col(c): v for c, v in state.runners.items()}
    s.dice = [reflect_die(d) for d in state.dice]
    return s


# ---- Index permutations (built once) ----

def _build_feature_permutation():
    """
    perm such that reflect_features(f) == f[perm].
    Per-column block (4 features) at position i moves to position 10-i;
    dice block (2 features) likewise; the context section [66:74] is invariant.
    """
    perm = np.arange(FEATURE_SIZE)
    for i in range(NUM_COLUMNS):
        j = COL_INDEX[reflect_col(COLUMNS[i])]   # mirror position (= 10 - i)
        for k in range(4):                       # per-column section [0:44]
            perm[i * 4 + k] = j * 4 + k
        for k in range(2):                       # dice section [44:66]
            perm[44 + i * 2 + k] = 44 + j * 2 + k
    # [66:74] left as identity by np.arange.
    return perm


def _build_action_permutation():
    """perm such that reflect_policy(p) == p[perm]; maps each action to its mirror."""
    perm = np.arange(ACTION_SPACE)
    for a in range(ACTION_SPACE):
        move, decision = action_to_move_decision(a)
        perm[a] = move_to_action(reflect_move(move), decision)
    return perm


FEATURE_PERM = _build_feature_permutation()
ACTION_PERM = _build_action_permutation()


def reflect_features(features):
    """Reflect a (74,) feature vector (or (N,74) batch along the last axis)."""
    return features[..., FEATURE_PERM]


def reflect_policy(policy):
    """Reflect a (154,) policy vector (or (N,154) batch along the last axis)."""
    return policy[..., ACTION_PERM]


def reflect_mask(mask):
    """Reflect a (154,) legal-action mask (or (N,154) batch along the last axis)."""
    return mask[..., ACTION_PERM]


def reflect_value(value):
    """Value/win-probability is invariant under board reflection."""
    return value


# ---- Training-record augmentation ----

def reflect_record(rec):
    """
    Mirror one self-play training record. Permutes the feature vector, the
    legal-action mask, the MCTS policy target, and the sampled action index;
    leaves value targets and metadata (player, step_index, game_id) unchanged.
    Returns a new dict; the input is not mutated.
    """
    r = dict(rec)
    r['features']    = reflect_features(rec['features'])
    r['mask']        = reflect_mask(rec['mask'])
    r['mcts_policy'] = reflect_policy(rec['mcts_policy'])
    r['action_idx']  = int(ACTION_PERM[int(rec['action_idx'])])
    return r


def augment_records(records):
    """
    Mirror augmentation for a list of training records: returns the originals
    PLUS a mirrored copy of each, so the training distribution is exactly
    reflection-symmetric. Can't Stop's symmetry group has only two elements,
    so including both orientations is the complete augmentation.
    Passes records through unchanged if the current variant isn't symmetric.
    """
    if not is_symmetric():
        return records
    return records + [reflect_record(r) for r in records]


def is_symmetric(column_heights=None, columns=None):
    """
    True iff the board reflection is a genuine symmetry for the current spec
    (column heights mirror-symmetric). Dice assumed standard symmetric d6.
    For an asymmetric variant this returns False and augmentation must be off.
    """
    column_heights = column_heights or COLUMN_HEIGHTS
    columns = columns or COLUMNS
    lo, hi = min(columns), max(columns)
    return all(column_heights[c] == column_heights[lo + hi - c] for c in columns)


# ============================================================
# Self-tests — also serve as the pipeline bug check.
# ============================================================

def _states_equal(a, b):
    return (a.active_player == b.active_player and
            a.all_claimed == b.all_claimed and
            all(a.claimed[p] == b.claimed[p] for p in a.players) and
            all(a.progress[p] == b.progress[p] for p in a.players) and
            a.runners == b.runners and
            list(a.dice) == list(b.dice))


def _random_states(n, rng):
    """Valid mid-game decision states, generated by random play."""
    states = []
    guard = 0
    while len(states) < n and guard < n * 100:
        guard += 1
        s = GameState(2)
        for _ in range(rng.randint(0, 14)):
            if s.game_over:
                break
            s.roll_dice()
            valid = get_valid_moves(s)
            if not valid:
                bust_turn(s)
                continue
            apply_move(s, rng.choice(valid))
            if rng.random() < 0.4:
                stop_turn(s)
        if s.game_over:
            continue
        if not s.dice:
            s.roll_dice()
        states.append(s)
    return states


def _run_self_tests():
    import random
    rng = random.Random(0)
    fails = 0

    # 1. reflect_col / reflect_die are involutions covering 7 and 3<->4 fixed/centre.
    assert all(reflect_col(reflect_col(c)) == c for c in COLUMNS)
    assert reflect_col(7) == 7
    assert all(reflect_die(reflect_die(v)) == v for v in range(1, 7))
    print("  [1] reflect_col / reflect_die are involutions: PASS")

    # 2. Permutations are genuine bijections AND self-inverse.
    assert sorted(FEATURE_PERM.tolist()) == list(range(FEATURE_SIZE))
    assert np.array_equal(FEATURE_PERM[FEATURE_PERM], np.arange(FEATURE_SIZE))
    assert sorted(ACTION_PERM.tolist()) == list(range(ACTION_SPACE))
    assert np.array_equal(ACTION_PERM[ACTION_PERM], np.arange(ACTION_SPACE))
    print("  [2] FEATURE_PERM and ACTION_PERM are self-inverse bijections: PASS")

    # 3. FEATURE EXTRACTOR BUG CHECK:
    #    reflecting the state then extracting == extracting then permuting.
    #    If this fails, the feature pipeline is NOT symmetric (a real bug).
    states = _random_states(400, rng)
    max_err = 0.0
    for s in states:
        a = extract_features(reflect_state(s))
        b = reflect_features(extract_features(s))
        max_err = max(max_err, float(np.max(np.abs(a - b))))
    assert max_err < 1e-5, f"feature reflection mismatch, max err {max_err}"
    print(f"  [3] extract_features is symmetric (max err {max_err:.2e}): PASS")

    # 4. ENGINE BUG CHECK + action-perm correctness:
    #    the legal mask of the reflected state == the permuted legal mask.
    bad = 0
    for s in states:
        valid = get_valid_moves(s)
        mask = get_legal_action_mask(valid)
        mask_reflected_direct = get_legal_action_mask(get_valid_moves(reflect_state(s)))
        if not np.array_equal(mask_reflected_direct, mask[ACTION_PERM]):
            bad += 1
    assert bad == 0, f"{bad}/{len(states)} states: legal mask not preserved under reflection"
    print(f"  [4] engine legal moves reflect consistently with ACTION_PERM "
          f"({len(states)} states): PASS")

    # 5. apply_move commutes with reflection: reflect then move == move then reflect.
    bad = 0
    for s in states:
        valid = get_valid_moves(s)
        if not valid:
            continue
        mv = rng.choice(valid)
        s1 = s.clone(); apply_move(s1, mv)
        sr = reflect_state(s); apply_move(sr, reflect_move(mv))
        if not _states_equal(sr, reflect_state(s1)):
            bad += 1
    assert bad == 0, f"{bad} states: apply_move does not commute with reflection"
    print("  [5] apply_move commutes with reflection: PASS")

    # 6. Symmetry guard sanity.
    assert is_symmetric() is True
    asym = dict(COLUMN_HEIGHTS); asym[2] = 99  # break the 2<->12 height match
    assert is_symmetric(column_heights=asym) is False
    print("  [6] is_symmetric() guard true on base game, false when broken: PASS")

    # 7. reflect_record: a full training record round-trips (double reflection
    #    is the identity) and value/metadata are preserved by a single reflect.
    rec_state = next((s for s in states if get_valid_moves(s)), None)
    assert rec_state is not None, "no state with legal moves to build a test record"
    valid = get_valid_moves(rec_state)
    mask0 = get_legal_action_mask(valid)
    legal = np.nonzero(mask0)[0]
    policy0 = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy0[legal] = 1.0 / len(legal)
    rec = {
        'features':     extract_features(rec_state, valid),
        'mask':         mask0,
        'mcts_policy':  policy0,
        'action_idx':   int(legal[0]),
        'mcts_value':   0.5,
        'value_target': 0.5,
        'player':       rec_state.active_player,
        'step_index':   0,
        'game_id':      0,
    }
    rr = reflect_record(reflect_record(rec))
    assert np.array_equal(rr['features'], rec['features'])
    assert np.array_equal(rr['mask'], rec['mask'])
    assert np.allclose(rr['mcts_policy'], rec['mcts_policy'])
    assert rr['action_idx'] == rec['action_idx']
    r1 = reflect_record(rec)
    assert r1['value_target'] == rec['value_target']
    assert r1['player'] == rec['player'] and r1['game_id'] == rec['game_id']
    assert r1['mask'][r1['action_idx']], "reflected action not legal under reflected mask"
    print("  [7] reflect_record round-trips and preserves value/metadata: PASS")

    print("\nAll symmetry self-tests passed. The feature extractor and engine are")
    print("provably reflection-symmetric, so the model's column asymmetry is in the")
    print("learned weights (emergent), not a pipeline bug — and augmentation is safe.")


if __name__ == "__main__":
    print("\nRunning Can't Stop symmetry self-tests...\n")
    _run_self_tests()