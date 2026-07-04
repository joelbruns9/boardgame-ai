"""
augmentation.py — D4 data augmentation for Kingdomino training tuples.

Kingdomino boards are invariant under the dihedral group D4: 4 rotations
× 2 reflections = 8 transforms total.  Each (encoded_state, policy, z)
training tuple has 7 equivalent friends produced by applying each
non-identity D4 element, expanding training data 8× at zero labelling cost.

WHAT TRANSFORMS, WHAT DOESN'T
─────────────────────────────
Spatial (transforms):
    my_board, opp_board  shape (9, 13, 13)     spatial planes rotate
    policy[:676*5]                              the spatial slice of the
                                                placement axis rotates AND
                                                its direction channels are
                                                permuted (see direction
                                                permutations below).

Invariant (unchanged):
    flat                 shape (FLAT_SIZE,)     tile encodings, bag, phase
    policy[676*5:678*5]                         DISCARD and NO_PLACEMENT —
                                                non-spatial actions
    policy pick axis (5 entries within each     pick slot index is unrelated
    placement)                                  to board orientation
    z                    scalar                 rotation is a coordinate
                                                change, not a strategic one

Both boards rotate together by the same D4 element.  Rotating them
independently would describe a joint state that never occurs in real play;
the policy is defined in a single coordinate frame.

DIRECTION PERMUTATIONS
──────────────────────
When the board rotates, the four "place B in direction d from A" semantics
also rotate.  For CCW 90°, "B right of A" becomes "B up of A".  The
permutation table below was derived by tracing each direction's offset
(Δx, Δy) through the rotation matrix and h-flip, then verified empirically
by the test suite.

CORRECTNESS CONTRACTS (all verified by tests)
─────────────────────────────────────────────
1. augment(inverse_of(t), augment(t, x)) == x          (byte-identical)
2. Castle remains at (CASTLE_CENTER, CASTLE_CENTER) under every transform.
3. Terrain channels remain one-hot at every occupied non-castle cell.
4. flat (shape (FLAT_SIZE,)), z, own_score, opp_score, and win_target are
   byte-identical across all 8 transforms.
5. A policy with mass exclusively on a single legal action remains exactly
   one-hot after transform (proving the spatial+direction transformation is
   consistent with the action codec's indexing).
6. The 8 transforms form a closed group: composing any two gives one of the 8.
7. own_score, opp_score, and win_target are scalars, byte-identical across
   all 8 transforms (final scores and game outcome do not change under
   rotation/reflection of the board).
8. DISCARD and NO_PLACEMENT joint indices are unchanged under all 8 transforms.
9. The pick axis (which of the PICK_AXIS_SIZE pick slots) is unchanged under
   all 8 transforms.
10. ROT90 CCW maps direction 'right' (0) to 'up' (3); FLIP_H maps 'right' to
    'left' — verified by concrete action-level tests.

TRAINING ISOLATION
This module does NOT import evaluation.py.  Augmentation is a pure
geometric operation on encoder/codec outputs.
"""
from __future__ import annotations
from typing import List, Tuple

import numpy as np

from games.kingdomino.encoder import CANVAS_SIZE, NUM_BOARD_CHANNELS, FLAT_SIZE
from games.kingdomino.action_codec import (
    NUM_DIRECTIONS, NUM_JOINT_ACTIONS, NUM_SPATIAL_PLACEMENTS,
    PICK_AXIS_SIZE, PLACEMENT_AXIS_SIZE,
)

# Optional Rust fast path for augment().  Falls back to the numpy implementation
# below when the extension isn't built — keeps the module importable everywhere.
try:
    from kingdomino_rust import d4_augment as _rust_d4_augment
    from kingdomino_rust import d4_augment_mask as _rust_d4_augment_mask
    from kingdomino_rust import d4_inverse_transform_id as _rust_inverse
    _RUST_AUGMENT_AVAILABLE = True
except ImportError:
    _RUST_AUGMENT_AVAILABLE = False


# ─── D4 group structure ───────────────────────────────────────────────────
NUM_D4_TRANSFORMS = 8

# Each entry: (CCW rotation count, h_flip, direction permutation).
# Convention: apply k rotations FIRST, then h_flip if True.  The direction
# permutation maps "new direction index → old direction index" so that:
#     new_direction_channel[d] = old_direction_channel[perm[d]]
_D4_ELEMENTS: Tuple[Tuple[int, bool, Tuple[int, int, int, int]], ...] = (
    # transform_id   rotation k   h_flip   direction_perm
    (0, False, (0, 1, 2, 3)),   # 0 : IDENTITY
    (1, False, (1, 2, 3, 0)),   # 1 : ROT90  (CCW)
    (2, False, (2, 3, 0, 1)),   # 2 : ROT180
    (3, False, (3, 0, 1, 2)),   # 3 : ROT270 (CCW)
    (0, True,  (2, 1, 0, 3)),   # 4 : FLIP_H
    (1, True,  (3, 2, 1, 0)),   # 5 : ROT90  + FLIP_H
    (2, True,  (0, 3, 2, 1)),   # 6 : ROT180 + FLIP_H  (= FLIP_V)
    (3, True,  (1, 0, 3, 2)),   # 7 : ROT270 + FLIP_H
)

# Inverse table.  Derivation:
#   Pure rotations: 0 and 2 (180°) are involutions; 1 ↔ 3 are inverses.
#   Flip-containing elements (4-7): each is its own inverse, because
#   (R^k · H)² = R^k · (H R^k) · H = R^k · R^(-k) · H · H = I,
#   using the dihedral relation H R^k = R^(-k) H and H² = I.
_INVERSE_TRANSFORM: Tuple[int, ...] = (0, 3, 2, 1, 4, 5, 6, 7)


def inverse_transform_id(transform_id: int) -> int:
    """Return the D4 transform_id that undoes `transform_id`."""
    if not 0 <= transform_id < NUM_D4_TRANSFORMS:
        raise ValueError(
            f"transform_id must be in [0, {NUM_D4_TRANSFORMS}); got {transform_id}."
        )
    if _RUST_AUGMENT_AVAILABLE:
        return _rust_inverse(transform_id)
    return _INVERSE_TRANSFORM[transform_id]


# ─── core transforms ──────────────────────────────────────────────────────
def _transform_spatial(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """Apply k CCW rotations then optional h-flip to a (C, H, W) tensor.

    Returns a contiguous copy so subsequent indexing doesn't hit negative
    strides (which numpy allows but PyTorch consumers won't).
    """
    out = np.rot90(arr, k=k, axes=(1, 2))
    if flip:
        out = out[:, :, ::-1]
    return np.ascontiguousarray(out)


def _transform_policy(
    policy: np.ndarray,
    k: int,
    flip: bool,
    dir_perm: Tuple[int, int, int, int],
) -> np.ndarray:
    """Apply the D4 transform to a flat policy of shape (NUM_JOINT_ACTIONS,).

    The policy layout is:  joint_idx = placement_idx * PICK_AXIS_SIZE + pick_idx
    where placement_idx ∈ [0, 676) is direction * 169 + y * 13 + x (spatial),
    placement_idx == 676 is DISCARD (non-spatial),
    placement_idx == 677 is NO_PLACEMENT (non-spatial).

    We rotate the (y, x) part of the spatial slice and permute the direction
    axis.  DISCARD, NO_PLACEMENT, and the pick axis are unchanged.
    """
    pol = policy.reshape(PLACEMENT_AXIS_SIZE, PICK_AXIS_SIZE)
    spatial = pol[:NUM_SPATIAL_PLACEMENTS]   # (676, 5)
    special = pol[NUM_SPATIAL_PLACEMENTS:]   # (2, 5)  — DISCARD, NO_PLACEMENT

    # Unflatten the spatial placements into a (direction, y, x, pick) tensor.
    spatial = spatial.reshape(NUM_DIRECTIONS, CANVAS_SIZE, CANVAS_SIZE, PICK_AXIS_SIZE)

    # Apply the spatial rotation on (y, x) = axes (1, 2).
    spatial = np.rot90(spatial, k=k, axes=(1, 2))
    if flip:
        spatial = spatial[:, :, ::-1]

    # Permute direction axis: new[d, ...] = post-rotation[perm[d], ...].
    spatial = spatial[list(dir_perm)]

    # Flatten back and re-attach the unchanged special entries.
    spatial = spatial.reshape(NUM_SPATIAL_PLACEMENTS, PICK_AXIS_SIZE)
    out = np.concatenate([spatial, special], axis=0).reshape(NUM_JOINT_ACTIONS)
    return np.ascontiguousarray(out)


# ─── public API ───────────────────────────────────────────────────────────
# (my_board, opp_board, flat, policy, z, own_score, opp_score, win_target)
TrainingTuple = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    float, float, float, float,
]

# Sentinel for the four-head scalar targets: a caller that forgets to pass
# own_score/opp_score/win_target would otherwise get FAKE labels (0/0/draw) on
# every position with no error — exactly the silent training-corruption class
# this guards against.  Required now; raise with a fix-it message if omitted.
_REQUIRED = object()


def augment(
    my_board: np.ndarray,
    opp_board: np.ndarray,
    flat: np.ndarray,
    policy: np.ndarray,
    z: float,
    transform_id: int,
    own_score: float = _REQUIRED,
    opp_score: float = _REQUIRED,
    win_target: float = _REQUIRED,
) -> TrainingTuple:
    """Apply one of the 8 D4 transforms to a complete training tuple.

    Inputs come from `encode_state` (my_board, opp_board, flat) plus a
    policy distribution over NUM_JOINT_ACTIONS=3390 indices and the scalar
    targets z, own_score, opp_score, win_target.

    own_score / opp_score / win_target are REQUIRED (sentinel defaults that
    raise if omitted) — passing the real four-head targets is mandatory for
    training; diagnostic/test callers must pass explicit placeholders
    (own_score=0.0, opp_score=0.0, win_target=0.5).

    Returns a new tuple with spatial components transformed consistently:
    the boards rotate, the spatial portion of the policy rotates and its
    direction channels permute.  flat, z, own_score, opp_score, and
    win_target are unchanged (rotation-invariant scalars passed through
    byte-identically).
    """
    if not 0 <= transform_id < NUM_D4_TRANSFORMS:
        raise ValueError(
            f"transform_id must be in [0, {NUM_D4_TRANSFORMS}); got {transform_id}."
        )
    _board_shape = (NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE)
    if my_board.shape != _board_shape:
        raise ValueError(f"bad my_board shape: {my_board.shape}, expected {_board_shape}")
    if opp_board.shape != _board_shape:
        raise ValueError(f"bad opp_board shape: {opp_board.shape}, expected {_board_shape}")
    if policy.shape != (NUM_JOINT_ACTIONS,):
        raise ValueError(f"bad policy shape: {policy.shape}, expected {(NUM_JOINT_ACTIONS,)}")
    if flat.shape != (FLAT_SIZE,):
        raise ValueError(
            f"bad flat shape: {flat.shape}, expected ({FLAT_SIZE},). "
            f"Note: FLAT_SIZE is now {FLAT_SIZE} after the symmetric pending "
            "encoder migration."
        )

    # Fix 3: four-head scalar targets are required — fail loudly rather than
    # silently fabricating labels.
    if own_score is _REQUIRED:
        raise ValueError(
            "own_score is required. Pass own_score=compute_target_own_score(state, player) "
            "or own_score=0.0 explicitly if this is a test/diagnostic call."
        )
    if opp_score is _REQUIRED:
        raise ValueError(
            "opp_score is required. Pass opp_score=compute_target_opponent_score(state, player) "
            "or opp_score=0.0 explicitly if this is a test/diagnostic call."
        )
    if win_target is _REQUIRED:
        raise ValueError(
            "win_target is required. Pass win_target=compute_target_win(state, player) "
            "or win_target=0.5 explicitly if this is a test/diagnostic call."
        )

    if _RUST_AUGMENT_AVAILABLE:
        # Rust does the two board transforms, the policy rotate+dir-permute, and
        # the flat copy in one call (byte-identical to the numpy path below).
        # Scalars stay in Python — they are rotation-invariant.
        mb_t, ob_t, fl_t, pol_t = _rust_d4_augment(
            np.ascontiguousarray(my_board, dtype=np.float32),
            np.ascontiguousarray(opp_board, dtype=np.float32),
            np.ascontiguousarray(flat, dtype=np.float32),
            np.ascontiguousarray(policy, dtype=np.float32),
            transform_id,
        )
    else:
        k, flip, dir_perm = _D4_ELEMENTS[transform_id]
        mb_t = _transform_spatial(my_board, k, flip)
        ob_t = _transform_spatial(opp_board, k, flip)
        fl_t = flat.copy()  # invariant, but copy to avoid downstream aliasing
        pol_t = _transform_policy(policy, k, flip, dir_perm)

    return (
        mb_t,
        ob_t,
        fl_t,
        pol_t,
        z,
        own_score,   # invariant scalar — passed through unchanged
        opp_score,   # invariant scalar — passed through unchanged
        win_target,  # invariant scalar — passed through unchanged
    )


def augment_mask(mask: np.ndarray, transform_id: int) -> np.ndarray:
    """Apply a D4 transform to a legal-action mask of shape (NUM_JOINT_ACTIONS,).

    Needed only if policy training uses masked_log_softmax: the legal mask is
    itself a 3390-length action-space vector and must be transformed by the
    SAME element as the policy target, or the mask and target will disagree
    about which spatial cells/directions are which after rotation.

    Do NOT substitute `policy > 0` for a legal mask: in MCTS many legal actions
    receive zero visits (especially when simulations < legal moves), and that
    would wrongly mark zero-visit legal actions as illegal.
    """
    if mask.dtype != np.bool_:
        raise ValueError(
            f"legal mask must be dtype bool, got {mask.dtype}. "
            f"Use mask.astype(bool) explicitly before calling augment_mask."
        )
    if mask.shape != (NUM_JOINT_ACTIONS,):
        raise ValueError(f"bad mask shape: {mask.shape}, expected {(NUM_JOINT_ACTIONS,)}")
    if not 0 <= transform_id < NUM_D4_TRANSFORMS:
        raise ValueError(
            f"transform_id must be in [0, {NUM_D4_TRANSFORMS}); got {transform_id}."
        )
    if _RUST_AUGMENT_AVAILABLE:
        # Rust does the bool spatial+direction transform GIL-free — this is what
        # lets the threaded sample_batch path actually run in parallel (the Python
        # _transform_policy is the dominant GIL-bound cost otherwise).
        return _rust_d4_augment_mask(np.ascontiguousarray(mask, dtype=bool),
                                     transform_id)
    k, flip, dir_perm = _D4_ELEMENTS[transform_id]
    return _transform_policy(mask.astype(np.float32), k, flip, dir_perm).astype(bool)


def augment_all(
    my_board: np.ndarray,
    opp_board: np.ndarray,
    flat: np.ndarray,
    policy: np.ndarray,
    z: float,
    own_score: float,
    opp_score: float,
    win_target: float,
) -> List[TrainingTuple]:
    """Return all 8 D4 augmentations of a training tuple.

    Convenient for offline data generation where you want every orbit
    expanded.  For online training, sample one transform per step from a
    uniform distribution over [0, NUM_D4_TRANSFORMS) and call `augment`.

    own_score / opp_score / win_target are required (forwarded to augment,
    which now mandates them); diagnostic callers pass 0.0, 0.0, 0.5.
    """
    return [
        augment(my_board, opp_board, flat, policy, z, t,
                own_score=own_score, opp_score=opp_score, win_target=win_target)
        for t in range(NUM_D4_TRANSFORMS)
    ]


# ─── module self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"NUM_D4_TRANSFORMS = {NUM_D4_TRANSFORMS}")
    print(f"_INVERSE_TRANSFORM = {_INVERSE_TRANSFORM}")
    print("Direction permutations:")
    for t, (k, h, perm) in enumerate(_D4_ELEMENTS):
        name = ["IDENTITY", "ROT90", "ROT180", "ROT270",
                "FLIP_H", "ROT90+H", "ROT180+H", "ROT270+H"][t]
        print(f"  {t}: {name:10s} k={k} flip={h}  perm={perm}")
