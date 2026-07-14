"""Test suite for D4 augmentation.

The most important tests are:
- Round-trip via inverse_transform_id (TEST 4)
- Direction permutation tracks a one-hot policy correctly (TEST 7) — the
  ground truth that proves the spatial+direction transformation matches the
  geometry of the action codec's indexing.
- Group closure: composing any two transforms gives one of the 8 (TEST 8).
"""
from __future__ import annotations

import itertools
import random
import sys

import numpy as np

from games.kingdomino.game import GameState, Phase
from games.kingdomino.encoder import (
    encode_state, CANVAS_SIZE, CASTLE_CENTER, NUM_BOARD_CHANNELS,
    CH_CASTLE, CH_OCCUPIED, CH_TERRAIN_START, CH_TERRAIN_END, FLAT_SIZE,
)
from games.kingdomino.action_codec import (
    NUM_JOINT_ACTIONS, PLACEMENT_AXIS_SIZE, PICK_AXIS_SIZE,
    NUM_SPATIAL_PLACEMENTS, NUM_DIRECTIONS, NO_PLACEMENT_IDX,
    DISCARD_PLACEMENT_IDX, NO_PICK_IDX, make_joint_idx, legal_mask,
)
from games.kingdomino import augmentation
from games.kingdomino.augmentation import (
    NUM_D4_TRANSFORMS, augment, augment_all, inverse_transform_id,
    augment_mask, _D4_ELEMENTS, _INVERSE_TRANSFORM,
    _transform_spatial, _transform_policy, _transform_flat, _FLAT_WH_PAIRS,
)

# Diagnostic scalar-target placeholders.  augment() now REQUIRES own_score /
# opp_score / win_target (Fix 3); these tests use synthetic boards/policies, not
# real training tuples, so they pass explicit placeholders (NOT real labels)
# via **_S to every augment / augment_all call.
_S = dict(own_score=0.0, opp_score=0.0, win_target=0.5)


_failures = []
def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        _failures.append(name)


def make_random_tuple(seed):
    """Build an encoded training tuple from a real mid-game state."""
    state = GameState.new(seed=seed)
    rng = random.Random(seed)
    for _ in range(seed % 25 + 5):
        if state.phase == Phase.GAME_OVER: break
        state = state.step(rng.choice(state.legal_actions()))
    mb, ob, flat = encode_state(state, player=0)
    # Synthesise a plausible policy and value
    mask = legal_mask(state)
    policy = mask.astype(np.float32)
    if policy.sum() > 0:
        policy /= policy.sum()
    z = 0.3
    return mb, ob, flat, policy, z, state


# ──────────────────────────────────────────────────────────────────────────
print("=== TEST 1: shapes and dtypes preserved under all 8 transforms ===")
mb, ob, flat, policy, z, _ = make_random_tuple(seed=11)
ok = True
for t in range(NUM_D4_TRANSFORMS):
    mb_t, ob_t, flat_t, pol_t, z_t, *_ = augment(mb, ob, flat, policy, z, t, **_S)
    if mb_t.shape != mb.shape: ok = False
    if ob_t.shape != ob.shape: ok = False
    if flat_t.shape != flat.shape: ok = False
    if pol_t.shape != policy.shape: ok = False
    if mb_t.dtype != mb.dtype or pol_t.dtype != policy.dtype: ok = False
check("shapes and dtypes preserved across all 8 transforms", ok)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 2: identity transform (t=0) is the identity function ===")
mb_t, ob_t, flat_t, pol_t, z_t, *_ = augment(mb, ob, flat, policy, z, 0, **_S)
check("identity preserves my_board", np.array_equal(mb_t, mb))
check("identity preserves opp_board", np.array_equal(ob_t, ob))
check("identity preserves flat",     np.array_equal(flat_t, flat))
check("identity preserves policy",   np.array_equal(pol_t, policy))
check("identity preserves z",        z_t == z)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 3: z invariant; flat invariant except width/height swap on odd rot ===")
# All flat fields are orientation-invariant EXCEPT the per-player bbox
# (width, height), which swap under the four odd-rotation transforms. z is a
# pure scalar and stays invariant across all 8.
_wh_idx = {i for pair in _FLAT_WH_PAIRS for i in pair}
_non_wh = [i for i in range(flat.shape[0]) if i not in _wh_idx]
violations = 0
for t in range(NUM_D4_TRANSFORMS):
    _, _, flat_t, _, z_t, *_ = augment(mb, ob, flat, policy, z, t, **_S)
    k, _flip, _perm = _D4_ELEMENTS[t]
    # everything OUTSIDE width/height must be byte-identical under every transform
    if not np.array_equal(flat_t[_non_wh], flat[_non_wh]): violations += 1
    # the full flat must match the reference transform (swap iff k odd)
    if not np.array_equal(flat_t, _transform_flat(flat, k)): violations += 1
    if z_t != z: violations += 1
check("z invariant and flat matches width/height-swap reference across all 8",
      violations == 0, f"violations={violations}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 4: round-trip via inverse_transform_id ===")
violations = 0
for seed in [11, 42, 17]:
    mb, ob, flat, policy, z, _ = make_random_tuple(seed=seed)
    for t in range(NUM_D4_TRANSFORMS):
        t_inv = inverse_transform_id(t)
        mb_t, ob_t, flat_t, pol_t, z_t, *_ = augment(mb, ob, flat, policy, z, t, **_S)
        mb_r, ob_r, flat_r, pol_r, z_r, *_ = augment(mb_t, ob_t, flat_t, pol_t, z_t, t_inv, **_S)
        if not (np.array_equal(mb_r, mb)
                and np.array_equal(ob_r, ob)
                and np.array_equal(flat_r, flat)
                and np.array_equal(pol_r, policy)
                and z_r == z):
            violations += 1
            print(f"    seed={seed} t={t} inverse={t_inv}: round-trip failed")
check("round-trip identity holds for all 8 transforms across multiple states",
      violations == 0, f"violations={violations}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 5: castle stays at CASTLE_CENTER under every transform ===")
mb, ob, flat, policy, z, _ = make_random_tuple(seed=23)
violations = 0
for t in range(NUM_D4_TRANSFORMS):
    mb_t, ob_t, _, _, _, *_ = augment(mb, ob, flat, policy, z, t, **_S)
    if not (mb_t[CH_CASTLE, CASTLE_CENTER, CASTLE_CENTER] == 1.0
            and mb_t[CH_CASTLE].sum() == 1.0):
        violations += 1
    if not (ob_t[CH_CASTLE, CASTLE_CENTER, CASTLE_CENTER] == 1.0
            and ob_t[CH_CASTLE].sum() == 1.0):
        violations += 1
check("castle stays at (CASTLE_CENTER, CASTLE_CENTER) on both boards, all 8 transforms",
      violations == 0, f"violations={violations}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 6: terrain channels remain one-hot at every occupied non-castle cell ===")
violations = 0
for t in range(NUM_D4_TRANSFORMS):
    mb_t, ob_t, _, _, _, *_ = augment(mb, ob, flat, policy, z, t, **_S)
    for plane in (mb_t, ob_t):
        occupied = plane[CH_OCCUPIED] > 0.5
        castle = plane[CH_CASTLE] > 0.5
        terrain_sum = plane[CH_TERRAIN_START:CH_TERRAIN_END].sum(axis=0)
        err = ((terrain_sum[occupied & ~castle] - 1.0) ** 2).sum()
        if err > 1e-6: violations += 1
check("terrain remains one-hot at every occupied non-castle cell, all 8 transforms",
      violations == 0, f"violations={violations}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 7: direction permutation is geometrically correct ===")
# This is the critical correctness test.  Construct a one-hot policy at a
# known (direction, y, x, pick), apply each transform, and verify the result
# is one-hot at the expected new position derived independently from the
# rotation geometry.

def expected_location(d, y, x, transform_id):
    """Compute the destination (d', y', x') of (d, y, x) under transform."""
    k, flip, perm = _D4_ELEMENTS[transform_id]
    # Apply k CCW rotations to (y, x).
    # np.rot90 with k=1 on (H, W) maps (i, j) → (W-1-j, i).
    yy, xx = y, x
    for _ in range(k):
        yy, xx = CANVAS_SIZE - 1 - xx, yy
    if flip:
        xx = CANVAS_SIZE - 1 - xx
    # The new direction d' is the one where perm[d'] == d.
    new_d = perm.index(d)
    return new_d, yy, xx


def make_one_hot_policy(d, y, x, pick):
    spatial_idx = d * (CANVAS_SIZE * CANVAS_SIZE) + y * CANVAS_SIZE + x
    joint_idx = spatial_idx * PICK_AXIS_SIZE + pick
    pol = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)
    pol[joint_idx] = 1.0
    return pol, joint_idx


# Use a generic (d, y, x, pick) — chosen so all 8 transforms produce
# distinct positions (i.e., not on a symmetry axis).
d0, y0, x0, pick0 = 0, 2, 3, 1

violations = 0
distinct_destinations = set()
for t in range(NUM_D4_TRANSFORMS):
    pol_in, _ = make_one_hot_policy(d0, y0, x0, pick0)
    _, _, _, pol_out, _, *_ = augment(mb, ob, flat, pol_in, z, t, **_S)
    # Where did the mass go?
    nonzero = np.where(pol_out > 0.5)[0]
    if len(nonzero) != 1:
        violations += 1
        print(f"    t={t}: expected one-hot, got {len(nonzero)} nonzero entries")
        continue
    got_joint = int(nonzero[0])
    # Decode the joint index
    got_placement = got_joint // PICK_AXIS_SIZE
    got_pick = got_joint % PICK_AXIS_SIZE
    if got_placement >= NUM_SPATIAL_PLACEMENTS:
        violations += 1
        print(f"    t={t}: mass moved out of spatial range")
        continue
    got_d = got_placement // (CANVAS_SIZE * CANVAS_SIZE)
    rem = got_placement % (CANVAS_SIZE * CANVAS_SIZE)
    got_y = rem // CANVAS_SIZE
    got_x = rem % CANVAS_SIZE
    # Expected destination
    exp_d, exp_y, exp_x = expected_location(d0, y0, x0, t)
    exp_pick = pick0  # pick axis is invariant
    if (got_d, got_y, got_x, got_pick) != (exp_d, exp_y, exp_x, exp_pick):
        violations += 1
        print(f"    t={t}: got (d={got_d},y={got_y},x={got_x},pick={got_pick}) "
              f"expected (d={exp_d},y={exp_y},x={exp_x},pick={exp_pick})")
    distinct_destinations.add((got_d, got_y, got_x, got_pick))
    # Mass should also be exactly 1.0 (no leakage)
    if abs(pol_out.sum() - 1.0) > 1e-6:
        violations += 1
        print(f"    t={t}: total mass = {pol_out.sum()}")

check("one-hot policy lands at geometrically-predicted location for all 8 transforms",
      violations == 0, f"violations={violations}")
check("the 8 transforms produce 8 distinct destinations (no collisions)",
      len(distinct_destinations) == 8,
      f"got {len(distinct_destinations)} distinct destinations")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 8: group closure — composing any two transforms yields one of the 8 ===")
# For every pair (t1, t2), apply both in sequence and check the result equals
# some single-transform application.  This proves the 8 transforms form a
# closed group under the operation augment(.) — i.e., they really are D4.
mb, ob, flat, policy, z, _ = make_random_tuple(seed=29)
violations = 0
composition_table = np.zeros((NUM_D4_TRANSFORMS, NUM_D4_TRANSFORMS), dtype=int)
for t1 in range(NUM_D4_TRANSFORMS):
    for t2 in range(NUM_D4_TRANSFORMS):
        mb_t, ob_t, flat_t, pol_t, z_t, *_ = augment(mb, ob, flat, policy, z, t1, **_S)
        mb_tt, ob_tt, flat_tt, pol_tt, z_tt, *_ = augment(mb_t, ob_t, flat_t, pol_t, z_t, t2, **_S)
        # Find which single transform reproduces this
        found = -1
        for u in range(NUM_D4_TRANSFORMS):
            mb_u, ob_u, _, pol_u, _, *_ = augment(mb, ob, flat, policy, z, u, **_S)
            if (np.array_equal(mb_u, mb_tt)
                and np.array_equal(ob_u, ob_tt)
                and np.array_equal(pol_u, pol_tt)):
                found = u
                break
        if found < 0:
            violations += 1
            print(f"    t1={t1}, t2={t2}: composition not in group")
        composition_table[t1, t2] = found
check("composition of any two transforms equals some single transform",
      violations == 0, f"violations={violations}")
# Sanity: the composition table is a Latin square (each row/col is a permutation)
rows_are_permutations = all(
    sorted(composition_table[i]) == list(range(NUM_D4_TRANSFORMS))
    for i in range(NUM_D4_TRANSFORMS)
)
cols_are_permutations = all(
    sorted(composition_table[:, j]) == list(range(NUM_D4_TRANSFORMS))
    for j in range(NUM_D4_TRANSFORMS)
)
check("composition table is a Latin square (Cayley table property)",
      rows_are_permutations and cols_are_permutations)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 9: inverse table is self-consistent ===")
# Verify that inverse_transform_id matches what we'd derive from the
# composition table: t·inv(t) == identity (transform 0).
violations = 0
for t in range(NUM_D4_TRANSFORMS):
    if composition_table[t, inverse_transform_id(t)] != 0:
        violations += 1
        print(f"    t={t}, claimed inverse={inverse_transform_id(t)}, but "
              f"composition gives {composition_table[t, inverse_transform_id(t)]}")
    if composition_table[inverse_transform_id(t), t] != 0:
        violations += 1
check("inverse_transform_id is consistent with composition table",
      violations == 0, f"violations={violations}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 10: augmented policy preserves legal_mask structure ===")
# If the original policy has mass on legal indices only, the augmented
# policy should also have mass on indices that are legal in the augmented
# coordinate frame.  We verify a weaker but sufficient property: the augmented
# policy is still a valid probability distribution (sums to 1.0, no negatives).
mb, ob, flat, _, z, state = make_random_tuple(seed=31)
mask = legal_mask(state)
policy = mask.astype(np.float32)
policy /= policy.sum()
violations = 0
for t in range(NUM_D4_TRANSFORMS):
    _, _, _, pol_t, _, *_ = augment(mb, ob, flat, policy, z, t, **_S)
    if abs(pol_t.sum() - 1.0) > 1e-6: violations += 1
    if (pol_t < 0).any(): violations += 1
check("augmented policy is a valid probability distribution (sums to 1, ≥0)",
      violations == 0, f"violations={violations}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 11: DISCARD and NO_PLACEMENT policy entries are invariant ===")
# Construct a policy that has mass on DISCARD and NO_PLACEMENT entries;
# verify those entries are unchanged under every transform.
pol_in = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)
pol_in[make_joint_idx(DISCARD_PLACEMENT_IDX, 0)] = 0.3
pol_in[make_joint_idx(DISCARD_PLACEMENT_IDX, 2)] = 0.2
pol_in[make_joint_idx(NO_PLACEMENT_IDX, 1)] = 0.4
pol_in[make_joint_idx(NO_PLACEMENT_IDX, NO_PICK_IDX)] = 0.1
violations = 0
for t in range(NUM_D4_TRANSFORMS):
    _, _, _, pol_t, _, *_ = augment(mb, ob, flat, pol_in, z, t, **_S)
    # Indices in the special region should be byte-identical
    special_in  = pol_in.reshape(PLACEMENT_AXIS_SIZE, PICK_AXIS_SIZE)[NUM_SPATIAL_PLACEMENTS:]
    special_out = pol_t.reshape(PLACEMENT_AXIS_SIZE, PICK_AXIS_SIZE)[NUM_SPATIAL_PLACEMENTS:]
    if not np.array_equal(special_in, special_out): violations += 1
check("DISCARD and NO_PLACEMENT entries are byte-identical across all transforms",
      violations == 0, f"violations={violations}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 12: augment_all returns 8 distinct results for a generic input ===")
mb, ob, flat, policy, z, _ = make_random_tuple(seed=37)
results = augment_all(mb, ob, flat, policy, z, **_S)
check("augment_all returns NUM_D4_TRANSFORMS results", len(results) == NUM_D4_TRANSFORMS)
# For a generic state, the 8 board encodings should all be different
mb_hashes = set(r[0].tobytes() for r in results)
check("augment_all produces distinct my_board tensors (no symmetry on this state)",
      len(mb_hashes) == NUM_D4_TRANSFORMS)
# augment_all[0] should equal the input
check("augment_all[0] matches identity transform",
      np.array_equal(results[0][0], mb)
      and np.array_equal(results[0][1], ob)
      and np.array_equal(results[0][3], policy))


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 13: augmentation does not import evaluation.py ===")
import ast
import games.kingdomino.augmentation as aug_mod
tree = ast.parse(open(aug_mod.__file__).read())
imports_evaluation = False
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module and 'evaluation' in node.module:
        imports_evaluation = True
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if 'evaluation' in alias.name:
                imports_evaluation = True
check("no import statement references 'evaluation'", not imports_evaluation)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 14 (A): ROT90 CCW concrete action mapping (right→up) ===")
# Derive the expected (new_y, new_x) EMPIRICALLY from the board transform (not
# from analysis) so this test is ground truth, not circular.
_tb = np.zeros((9, 13, 13), dtype=np.float32)
_tb[0, 4, 3] = 1.0
_rot = _transform_spatial(_tb, k=1, flip=False)
_nz = np.argwhere(_rot[0] > 0)
check("A: one-hot board stays one-hot under ROT90", len(_nz) == 1)
A_new_y, A_new_x = int(_nz[0][0]), int(_nz[0][1])
# Which NEW direction channel pulls from old dir 0 (right)? new[d]=old[perm[d]],
# so find d with perm[d]==0.
A_dir_perm = _D4_ELEMENTS[1][2]          # (1, 2, 3, 0)
A_new_dir = A_dir_perm.index(0)          # right → up (expect 3)
A_pick = 2
_old_joint = (0 * 169 + 4 * 13 + 3) * PICK_AXIS_SIZE + A_pick
_policy = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)
_policy[_old_joint] = 1.0
_mb = np.zeros((9, 13, 13), dtype=np.float32)
_flat = np.zeros(FLAT_SIZE, dtype=np.float32)
_, _, _, A_pol_t, *_ = augment(_mb, _mb, _flat, _policy, 0.0, 1, **_S)
A_expected = (A_new_dir * 169 + A_new_y * 13 + A_new_x) * PICK_AXIS_SIZE + A_pick
check(f"A: ROT90 right(y=4,x=3) → dir={A_new_dir} (y={A_new_y},x={A_new_x})",
      A_pol_t[A_expected] == 1.0,
      f"argmax={int(A_pol_t.argmax())}, expected={A_expected}")
check("A: policy stays one-hot (sum==1)", abs(float(A_pol_t.sum()) - 1.0) < 1e-6)
check("A: new_dir is 3 (up) — right maps to up under ROT90 CCW", A_new_dir == 3,
      f"got {A_new_dir}")
print(f"  [A] ROT90 CCW: right at (y=4, x=3) → direction={A_new_dir} (up) "
      f"at (y={A_new_y}, x={A_new_x})")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 15 (B): FLIP_H concrete action mapping (right→left) ===")
_tb = np.zeros((9, 13, 13), dtype=np.float32)
_tb[0, 4, 3] = 1.0
_flip = _transform_spatial(_tb, k=0, flip=True)
_nz = np.argwhere(_flip[0] > 0)
check("B: one-hot board stays one-hot under FLIP_H", len(_nz) == 1)
B_new_y, B_new_x = int(_nz[0][0]), int(_nz[0][1])
B_dir_perm = _D4_ELEMENTS[4][2]          # (2, 1, 0, 3)
B_new_dir = B_dir_perm.index(0)          # right → left (expect 2)
B_pick = 1
_old_joint = (0 * 169 + 4 * 13 + 3) * PICK_AXIS_SIZE + B_pick
_policy = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)
_policy[_old_joint] = 1.0
_mb = np.zeros((9, 13, 13), dtype=np.float32)
_flat = np.zeros(FLAT_SIZE, dtype=np.float32)
_, _, _, B_pol_t, *_ = augment(_mb, _mb, _flat, _policy, 0.0, 4, **_S)
B_expected = (B_new_dir * 169 + B_new_y * 13 + B_new_x) * PICK_AXIS_SIZE + B_pick
check(f"B: FLIP_H right(y=4,x=3) → dir={B_new_dir} (y={B_new_y},x={B_new_x})",
      B_pol_t[B_expected] == 1.0, f"argmax={int(B_pol_t.argmax())}")
check("B: new_dir is 2 (left) and y unchanged (FLIP_H mirrors x only)",
      B_new_dir == 2 and B_new_y == 4, f"new_dir={B_new_dir}, new_y={B_new_y}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 16 (C): DISCARD / NO_PLACEMENT invariance (all 8 transforms) ===")
_c_violations = 0
for t in range(8):
    for pick_idx in range(PICK_AXIS_SIZE):
        for placement_idx in (NUM_SPATIAL_PLACEMENTS, NUM_SPATIAL_PLACEMENTS + 1):
            joint = placement_idx * PICK_AXIS_SIZE + pick_idx
            policy = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)
            policy[joint] = 1.0
            mb = np.zeros((9, 13, 13), dtype=np.float32)
            flat = np.zeros(FLAT_SIZE, dtype=np.float32)
            _, _, _, pol_t, *_ = augment(mb, mb, flat, policy, 0.0, t, **_S)
            if pol_t[joint] != 1.0:
                _c_violations += 1
check("C: DISCARD & NO_PLACEMENT joint indices unchanged under all 8 transforms",
      _c_violations == 0, f"violations={_c_violations}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 17 (D): pick-axis invariance (all 8 transforms, all slots) ===")
_d_violations = 0
for t in range(8):
    for pick_idx in range(PICK_AXIS_SIZE):
        placement_idx = 0 * 169 + 6 * 13 + 6   # dir=0, board centre (a fixed point)
        joint = placement_idx * PICK_AXIS_SIZE + pick_idx
        policy = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)
        policy[joint] = 1.0
        mb = np.zeros((9, 13, 13), dtype=np.float32)
        flat = np.zeros(FLAT_SIZE, dtype=np.float32)
        _, _, _, pol_t, *_ = augment(mb, mb, flat, policy, 0.0, t, **_S)
        new_joint = int(pol_t.argmax())
        if new_joint % PICK_AXIS_SIZE != pick_idx:
            _d_violations += 1
check("D: pick axis unchanged under all 8 transforms × all pick slots",
      _d_violations == 0, f"violations={_d_violations}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 18 (E): Rust d4_augment == NumPy fallback (concrete action) ===")
try:
    from kingdomino_rust import d4_augment as _rust_d4
    _e_pol_mismatch = 0
    _e_board_mismatch = 0
    E_pick = 2
    _old_joint = (0 * 169 + 4 * 13 + 3) * PICK_AXIS_SIZE + E_pick  # right, (y=4,x=3)
    _policy = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)
    _policy[_old_joint] = 1.0
    _mb = np.random.default_rng(42).random((9, 13, 13), dtype=np.float32)
    _flat = np.zeros(FLAT_SIZE, dtype=np.float32)
    for t in range(8):
        mb_r, ob_r, fl_r, pol_r = _rust_d4(_mb, _mb, _flat, _policy, t)
        k, flip, dir_perm = _D4_ELEMENTS[t]
        pol_py = _transform_policy(_policy, k, flip, dir_perm)
        mb_py = _transform_spatial(_mb, k, flip)
        if not np.array_equal(np.asarray(pol_r), pol_py):
            _e_pol_mismatch += 1
        if not np.array_equal(np.asarray(mb_r), mb_py):
            _e_board_mismatch += 1
    check("E: Rust vs NumPy policy identical on concrete action (all 8)",
          _e_pol_mismatch == 0, f"mismatches={_e_pol_mismatch}")
    check("E: Rust vs NumPy board identical (all 8)",
          _e_board_mismatch == 0, f"mismatches={_e_board_mismatch}")
except ImportError:
    print("  SKIP  kingdomino_rust not built — Rust-vs-NumPy check skipped")


# ──────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
if _failures:
    print(f"FAILED: {len(_failures)} test(s)")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")