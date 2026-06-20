"""Test suite for the action codec.

The critical test is round-trip identity verified through the engine: for every
legal action, encode then decode and step the engine — the resulting board state
must match.  This is stronger than object equality (which fails for symmetric
dominoes) and is exactly what matters for self-play training.
"""
from __future__ import annotations

import random
import sys
import numpy as np

from games.kingdomino.board import Placement
from games.kingdomino.dominoes import DOMINOES, Terrain
from games.kingdomino.game import GameState, Phase, PickAction, TurnAction
from games.kingdomino import action_codec as ac
from games.kingdomino.action_codec import (
    NUM_JOINT_ACTIONS, PLACEMENT_AXIS_SIZE, PICK_AXIS_SIZE,
    NUM_SPATIAL_PLACEMENTS, DISCARD_PLACEMENT_IDX, NO_PLACEMENT_IDX,
    NO_PICK_IDX, NUM_PICK_SLOTS, NUM_DIRECTIONS,
    encode_action, decode_action, legal_mask, make_joint_idx, split_joint_idx,
)


_failures = []
def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        _failures.append(name)


def boards_equivalent(b1, b2):
    """Two boards are equivalent if their terrain, crowns, and occupancy match."""
    return (
        np.array_equal(b1.terrain, b2.terrain) and
        np.array_equal(b1.crowns,  b2.crowns) and
        b1.castle_pos == b2.castle_pos
    )


def states_equivalent(s1, s2):
    """Compare game state in a way that ignores object identity."""
    if s1.phase != s2.phase:
        return False
    if s1.current_row != s2.current_row:
        return False
    if [(c.player, c.domino_id) for c in s1.pending_claims] != \
       [(c.player, c.domino_id) for c in s2.pending_claims]:
        return False
    if [(c.player, c.domino_id) for c in s1.next_claims] != \
       [(c.player, c.domino_id) for c in s2.next_claims]:
        return False
    if s1.actor_index != s2.actor_index:
        return False
    if s1.initial_pick_count != s2.initial_pick_count:
        return False
    for b1, b2 in zip(s1.boards, s2.boards):
        if not boards_equivalent(b1, b2):
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────
print("=== TEST 1: constants and arithmetic ===")
check("NUM_JOINT_ACTIONS = 3390", NUM_JOINT_ACTIONS == 3390)
check("PLACEMENT_AXIS_SIZE = 678", PLACEMENT_AXIS_SIZE == 678)
check("PICK_AXIS_SIZE = 5", PICK_AXIS_SIZE == 5)
check("NUM_SPATIAL_PLACEMENTS = 676", NUM_SPATIAL_PLACEMENTS == 676)
check("DISCARD = 676", DISCARD_PLACEMENT_IDX == 676)
check("NO_PLACEMENT = 677", NO_PLACEMENT_IDX == 677)
check("NO_PICK = 4", NO_PICK_IDX == 4)

# Round-trip on joint index helpers
for p in [0, 100, 676, 677]:
    for k in [0, 2, 4]:
        idx = make_joint_idx(p, k)
        p2, k2 = split_joint_idx(idx)
        if (p2, k2) != (p, k):
            check(f"split(make({p},{k})) round-trip", False, f"got ({p2},{k2})")
            break
else:
    check("make_joint_idx / split_joint_idx round-trip on all sampled pairs", True)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 2: legal_mask is the correct size, dtype, and bit-count ===")
state = GameState.new(seed=0)
m = legal_mask(state)
check("mask shape (NUM_JOINT_ACTIONS,)", m.shape == (NUM_JOINT_ACTIONS,))
check("mask dtype bool", m.dtype == np.bool_)
n_legal = m.sum()
n_engine = len(state.legal_actions())
check(f"INITIAL_SELECTION mask bit-count matches legal_actions ({n_legal} vs {n_engine})",
      n_legal == n_engine)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 3: INITIAL_SELECTION encoding uses NO_PLACEMENT_IDX ===")
state = GameState.new(seed=1)
for action in state.legal_actions():
    idx = encode_action(action, state)
    p_idx, k_idx = split_joint_idx(idx)
    if p_idx != NO_PLACEMENT_IDX:
        check(f"INITIAL_SELECTION uses NO_PLACEMENT_IDX", False,
              f"got placement_idx={p_idx} for {action}")
        break
    if k_idx == NO_PICK_IDX:
        check(f"INITIAL_SELECTION does not use NO_PICK_IDX", False)
        break
else:
    check("every INITIAL_SELECTION action has placement_idx=NO_PLACEMENT_IDX", True)
    check("every INITIAL_SELECTION action has a real pick_idx", True)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 4: round-trip identity through the engine, INITIAL_SELECTION ===")
state = GameState.new(seed=2)
violations = 0
for action in state.legal_actions():
    idx = encode_action(action, state)
    decoded = decode_action(idx, state)
    # For INITIAL_SELECTION the decoded action should be identical
    if not (isinstance(decoded, PickAction) and decoded.domino_id == action.domino_id):
        violations += 1
check("encode/decode round-trip preserves PickAction exactly", violations == 0,
      f"violations={violations}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 5: round-trip identity through the engine, PLACE_AND_SELECT ===")
# Drive the game forward into PLACE_AND_SELECT and exercise round-trip.
rng = random.Random(42)
state = GameState.new(seed=7)
while state.phase != Phase.PLACE_AND_SELECT and state.phase != Phase.GAME_OVER:
    actions = state.legal_actions()
    state = state.step(rng.choice(actions))

violations = 0
tested = 0
# Iterate over several PLACE_AND_SELECT decision points
for _ in range(40):
    if state.phase != Phase.PLACE_AND_SELECT:
        break
    actions = state.legal_actions()
    # Test every legal action at this decision point
    for action in actions:
        idx = encode_action(action, state)
        decoded = decode_action(idx, state)
        # Step both — engine should produce equivalent states
        try:
            s_orig = state.step(action)
            s_dec = state.step(decoded)
        except Exception as e:
            violations += 1
            print(f"    EXCEPTION on action {action}: {e}")
            continue
        if not states_equivalent(s_orig, s_dec):
            violations += 1
            print(f"    state mismatch for action {action}")
        tested += 1
    # Step forward to the next decision point
    state = state.step(rng.choice(actions))

check(f"PLACE_AND_SELECT engine-equivalent round-trip ({tested} actions tested)",
      violations == 0, f"violations={violations}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 6: round-trip identity through the engine, FINAL_PLACEMENT ===")
# Drive to FINAL_PLACEMENT
rng = random.Random(99)
state = GameState.new(seed=11)
while state.phase != Phase.FINAL_PLACEMENT and state.phase != Phase.GAME_OVER:
    actions = state.legal_actions()
    state = state.step(rng.choice(actions))

violations = 0
tested = 0
while state.phase == Phase.FINAL_PLACEMENT:
    actions = state.legal_actions()
    for action in actions:
        idx = encode_action(action, state)
        p_idx, k_idx = split_joint_idx(idx)
        # FINAL_PLACEMENT must use NO_PICK_IDX
        if k_idx != NO_PICK_IDX:
            violations += 1
            print(f"    FINAL_PLACEMENT non-NO_PICK pick_idx={k_idx}")
        decoded = decode_action(idx, state)
        try:
            s_orig = state.step(action)
            s_dec = state.step(decoded)
        except Exception as e:
            violations += 1
            print(f"    EXCEPTION on action {action}: {e}")
            continue
        if not states_equivalent(s_orig, s_dec):
            violations += 1
            print(f"    state mismatch for action {action}")
        tested += 1
    state = state.step(rng.choice(actions))

check(f"FINAL_PLACEMENT engine-equivalent round-trip ({tested} actions tested)",
      violations == 0, f"violations={violations}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 7: legal_mask covers EXACTLY the engine's legal_actions ===")
# For a variety of states, the mask bit-count must equal len(legal_actions).
rng = random.Random(123)
state = GameState.new(seed=5)
mismatches = 0
tested_states = 0
for _ in range(50):
    if state.phase == Phase.GAME_OVER:
        break
    m = legal_mask(state)
    n_mask = int(m.sum())
    n_engine = len(state.legal_actions())
    if n_mask != n_engine:
        mismatches += 1
        print(f"    {state.phase.name}: mask={n_mask}, engine={n_engine}")
    tested_states += 1
    state = state.step(rng.choice(state.legal_actions()))
check(f"mask cardinality matches engine across {tested_states} states", mismatches == 0,
      f"mismatches={mismatches}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 8: legal_mask only marks indices that decode to legal actions ===")
# Every True bit in the mask must decode to an action the engine accepts.
rng = random.Random(456)
state = GameState.new(seed=8)
violations = 0
for _ in range(30):
    if state.phase == Phase.GAME_OVER:
        break
    m = legal_mask(state)
    legal_set = set(state.legal_actions())  # objects; works for PickAction (frozen)
    # For TurnAction we can't use simple set membership due to Placement equality.
    # Instead, we step the engine and trust that legal_mask + decode succeeds.
    for idx in np.where(m)[0]:
        try:
            decoded = decode_action(int(idx), state)
            state.step(decoded)
        except Exception as e:
            violations += 1
            print(f"    idx {idx} decoded action raised: {e}")
    state = state.step(rng.choice(state.legal_actions()))
check("every True mask bit decodes to an engine-accepted action", violations == 0,
      f"violations={violations}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 9: random masked bits decode and step cleanly across whole game ===")
# Play many random games, sampling uniformly from masked actions.  Verifies
# end-to-end stability of the codec across all phases.
n_games = 5
total_steps = 0
violations = 0
for game_seed in range(20, 20 + n_games):
    state = GameState.new(seed=game_seed)
    rng = random.Random(game_seed * 7)
    while state.phase != Phase.GAME_OVER:
        m = legal_mask(state)
        idxs = np.where(m)[0]
        if len(idxs) == 0:
            violations += 1
            print(f"    game {game_seed}: empty mask at phase {state.phase.name}")
            break
        idx = int(rng.choice(idxs))
        try:
            decoded = decode_action(idx, state)
            state = state.step(decoded)
        except Exception as e:
            violations += 1
            print(f"    game {game_seed}: step failed: {e}")
            break
        total_steps += 1
check(f"{n_games} random games played via codec, {total_steps} steps total",
      violations == 0, f"violations={violations}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 10: error handling on out-of-range / wrong-phase indices ===")
state = GameState.new(seed=0)  # INITIAL_SELECTION

# Out of range
caught = False
try: decode_action(NUM_JOINT_ACTIONS, state)
except ValueError: caught = True
check("decode_action raises on idx >= NUM_JOINT_ACTIONS", caught)

caught = False
try: decode_action(-1, state)
except ValueError: caught = True
check("decode_action raises on negative idx", caught)

# Wrong phase for placement
caught = False
try: decode_action(make_joint_idx(0, 0), state)  # spatial placement in INITIAL_SELECTION
except ValueError: caught = True
check("decode_action raises for spatial placement in INITIAL_SELECTION", caught)

# Wrong phase for NO_PICK
caught = False
try: decode_action(make_joint_idx(NO_PLACEMENT_IDX, NO_PICK_IDX), state)
except ValueError: caught = True
check("decode_action raises for NO_PICK in INITIAL_SELECTION", caught)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 11: discard action encoding round-trips ===")
# Build a board state where a discard occurs.  We synthesize this by stepping
# until we find a state with a forced discard in the legal action set, OR
# directly synthesizing a TurnAction(placement=None, ...).
state = GameState.new(seed=33)
rng = random.Random(33)
# Step into PLACE_AND_SELECT
while state.phase != Phase.PLACE_AND_SELECT:
    state = state.step(rng.choice(state.legal_actions()))

# Synthesize a discard action manually (always legal to encode, even if
# discard isn't currently in legal_actions — encoding must still work).
synthetic_discard = TurnAction(placement=None, pick_domino_id=state.current_row[0])
idx = encode_action(synthetic_discard, state)
p_idx, k_idx = split_joint_idx(idx)
check("synthetic discard encodes to DISCARD_PLACEMENT_IDX",
      p_idx == DISCARD_PLACEMENT_IDX)
check("synthetic discard pick_idx is the right slot", k_idx == 0)

decoded = decode_action(idx, state)
check("decoded discard has placement=None",
      isinstance(decoded, TurnAction) and decoded.placement is None)
check("decoded discard has correct pick_domino_id",
      decoded.pick_domino_id == state.current_row[0])


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 12: decode_action validates spatial placement legality ===")
# Set up a PLACE_AND_SELECT state so the current actor has a pending claim.
state = GameState.new(seed=44)
rng = random.Random(44)
while state.phase != Phase.PLACE_AND_SELECT:
    state = state.step(rng.choice(state.legal_actions()))

# Pick an index that decodes to "place A-half on the castle cell" — always
# illegal because that cell is already occupied by the castle.  In the
# castle-centred frame, the castle sits at (CASTLE_CENTER, CASTLE_CENTER) = (6,6).
# direction=0 (right), out_y=6, out_x=6 → idx = 0 * 169 + 6 * 13 + 6 = 84.
illegal_placement_idx = 6 * 13 + 6  # direction 0, cell (6,6)
illegal_joint_idx = make_joint_idx(illegal_placement_idx, 0)

# Sanity check: this index is NOT in the legal mask (it's illegal by construction)
mask = legal_mask(state)
check("illegal placement is not in legal_mask",
      not mask[illegal_joint_idx])

# With validate=True (default), decoding should raise
caught = False
try:
    decode_action(illegal_joint_idx, state)
except ValueError as e:
    caught = True
    msg = str(e)
check("decode_action raises on illegal placement with validate=True (default)",
      caught)
check("error message mentions legality and unmasked logits hint",
      caught and "not legal" in msg and "unmasked" in msg)

# With validate=False, decoding should succeed and return a syntactically
# valid TurnAction (even though stepping it would fail)
try:
    decoded = decode_action(illegal_joint_idx, state, validate=False)
    no_raise = True
except Exception:
    no_raise = False
check("decode_action(validate=False) does not raise on illegal placement",
      no_raise)
check("decode_action(validate=False) returns a TurnAction",
      no_raise and isinstance(decoded, TurnAction))
check("decoded TurnAction has the decoded placement, not None",
      no_raise and decoded.placement is not None)

# Sanity check: legal placements should still work with validate=True
violations = 0
for action in state.legal_actions():
    idx = encode_action(action, state)
    try:
        decode_action(idx, state, validate=True)
    except ValueError:
        violations += 1
check("validate=True does not reject legal actions (no regression)",
      violations == 0, f"violations={violations}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 13: discard (placement=None) is unaffected by validate ===")
# Synthesize a discard TurnAction and round-trip with both validate settings.
state = GameState.new(seed=55)
rng = random.Random(55)
while state.phase != Phase.PLACE_AND_SELECT:
    state = state.step(rng.choice(state.legal_actions()))
synthetic_discard = TurnAction(placement=None, pick_domino_id=state.current_row[0])
idx = encode_action(synthetic_discard, state)
for v in (True, False):
    try:
        decoded = decode_action(idx, state, validate=v)
        ok = isinstance(decoded, TurnAction) and decoded.placement is None
    except Exception as e:
        ok = False
        print(f"    discard decode raised under validate={v}: {e}")
    check(f"discard decodes cleanly with validate={v}", ok)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 14: codec does not import evaluation.py ===")
import ast
import games.kingdomino.action_codec as ac_mod
tree = ast.parse(open(ac_mod.__file__).read())
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
print(f"\n{'='*60}")
if _failures:
    print(f"FAILED: {len(_failures)} test(s)")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")