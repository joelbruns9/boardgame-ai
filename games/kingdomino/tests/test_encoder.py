"""Test suite for the encoder.

Each test prints PASS/FAIL and exits non-zero on any failure.
"""
from __future__ import annotations
import math
import random
import sys
import numpy as np

from games.kingdomino.dominoes import DOMINOES, Terrain
from games.kingdomino.board import Board
from games.kingdomino.game import GameState, Phase, Claim, PickAction, TurnAction
from games.kingdomino import encoder
from games.kingdomino.encoder import (
    encode_state, compute_target_z,
    FLAT_LAYOUT, FLAT_SIZE,
    CANVAS_SIZE, CASTLE_CENTER, NUM_BOARD_CHANNELS,
    CH_CASTLE, CH_OCCUPIED, CH_CROWNS, CH_TERRAIN_START, CH_TERRAIN_END,
    TILE_FEAT_SIZE, ROW_SLOT_SIZE, CLAIM_SLOT_SIZE,
    NUM_DOMINOES, MAX_CROWNS,
)


_failures = []
def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        _failures.append(name)


def play_random_to_phase(state, target_phase, rng, max_steps=80):
    """Step the game forward with random legal actions until target_phase is reached."""
    steps = 0
    while state.phase != target_phase and state.phase != Phase.GAME_OVER:
        actions = state.legal_actions()
        if not actions:
            break
        state = state.step(rng.choice(actions))
        steps += 1
        if steps > max_steps:
            break
    return state


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 1: shapes & dtypes ===")
state = GameState.new(seed=0)
mb, ob, flat = encode_state(state, player=0)
check("my_board shape (9,13,13)", mb.shape == (9, 13, 13))
check("opp_board shape (9,13,13)", ob.shape == (9, 13, 13))
check("flat shape matches FLAT_SIZE", flat.shape == (FLAT_SIZE,))
check("my_board dtype float32", mb.dtype == np.float32)
check("opp_board dtype float32", ob.dtype == np.float32)
check("flat dtype float32", flat.dtype == np.float32)
check("no NaN in my_board", not np.isnan(mb).any())
check("no NaN in opp_board", not np.isnan(ob).any())
check("no NaN in flat", not np.isnan(flat).any())


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 2: castle always at (CASTLE_CENTER, CASTLE_CENTER) ===")
# Try several different starting seeds and game-progression states
rng = random.Random(42)
for seed in [0, 1, 7, 99, 314]:
    state = GameState.new(seed=seed)
    state = play_random_to_phase(state, Phase.PLACE_AND_SELECT, random.Random(seed))
    # Play forward a random number of steps
    for _ in range(rng.randint(0, 30)):
        actions = state.legal_actions()
        if not actions or state.phase == Phase.GAME_OVER:
            break
        state = state.step(rng.choice(actions))
    if state.phase == Phase.GAME_OVER:
        continue
    mb, ob, _ = encode_state(state, player=0)
    check(f"seed={seed}: my castle mask centred",
          mb[CH_CASTLE, CASTLE_CENTER, CASTLE_CENTER] == 1.0
          and mb[CH_CASTLE].sum() == 1.0)
    check(f"seed={seed}: opp castle mask centred",
          ob[CH_CASTLE, CASTLE_CENTER, CASTLE_CENTER] == 1.0
          and ob[CH_CASTLE].sum() == 1.0)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 3: terrain channels are one-hot and aligned with crowns/occupied ===")
state = GameState.new(seed=2)
rng = random.Random(2)
# Play ~25 actions to get a populated board
for _ in range(40):
    if state.phase == Phase.GAME_OVER:
        break
    actions = state.legal_actions()
    if not actions:
        break
    state = state.step(rng.choice(actions))
mb, ob, _ = encode_state(state, player=0)

# Every occupied non-castle cell must have exactly one terrain channel set,
# and the crown value must be in [0, 1].
for board_name, plane in [("my", mb), ("opp", ob)]:
    occupied = plane[CH_OCCUPIED] > 0.5
    terrain_sum = plane[CH_TERRAIN_START:CH_TERRAIN_END].sum(axis=0)
    castle_mask = plane[CH_CASTLE] > 0.5
    # Occupied cells split into: castle (terrain_sum=0) and terrain cells (terrain_sum=1)
    non_castle_occupied = occupied & ~castle_mask
    ok_onehot = ((terrain_sum[non_castle_occupied] - 1.0) ** 2).sum() < 1e-6
    check(f"{board_name}: every non-castle occupied cell is one-hot terrain", ok_onehot)
    ok_unocc = terrain_sum[~occupied].sum() == 0.0
    check(f"{board_name}: no terrain set on unoccupied cells", ok_unocc)
    crowns_plane = plane[CH_CROWNS]
    in_range = (crowns_plane >= 0.0).all() and (crowns_plane <= 1.0).all()
    check(f"{board_name}: crown values in [0,1]", in_range)
    # Crowns only on occupied cells
    ok_crowns_loc = (crowns_plane[~occupied]).sum() == 0.0
    check(f"{board_name}: crowns zero on unoccupied cells", ok_crowns_loc)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 4: bag computation matches set(state.deck) ===")
# After every step, the public-info bag should equal the engine's deck contents.
state = GameState.new(seed=11)
rng = random.Random(11)
for step in range(30):
    if state.phase == Phase.GAME_OVER:
        break
    _, _, flat = encode_state(state, player=0)
    bag = flat[FLAT_LAYOUT['bag']]
    public_set = {i + 1 for i in np.where(bag > 0.5)[0]}
    deck_set = set(state.deck)
    if public_set != deck_set:
        check(f"step={step}: bag matches deck set", False,
              f"diff={public_set ^ deck_set}")
        break
    actions = state.legal_actions()
    if not actions:
        break
    state = state.step(rng.choice(actions))
else:
    check("bag matches set(state.deck) across 30 steps", True)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 5: encoder does not import evaluation.py ===")
import ast
import games.kingdomino.encoder as enc_mod
tree = ast.parse(open(enc_mod.__file__).read())
imports_evaluation = False
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        if node.module and 'evaluation' in node.module:
            imports_evaluation = True
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if 'evaluation' in alias.name:
                imports_evaluation = True
check("no import statement references 'evaluation' (AST scan)", not imports_evaluation)
check("evaluation module not loaded into sys.modules via encoder import",
      "games.kingdomino.evaluation" not in sys.modules)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 6: information-set safety — encoding is invariant under deck shuffle ===")
# If I shuffle state.deck (preserving its members) the encoding must NOT change,
# because no encoder feature should depend on deck order.
state = GameState.new(seed=33)
rng = random.Random(33)
# Step into mid-game
for _ in range(20):
    if state.phase == Phase.GAME_OVER: break
    a = state.legal_actions()
    if not a: break
    state = state.step(rng.choice(a))

mb_a, ob_a, flat_a = encode_state(state, player=0)

# Shuffle the (otherwise unobserved) deck order. GameState.copy returns a
# fresh instance whose lists are independent — we mutate the deck list in place.
state2 = state.copy()
random.Random(999).shuffle(state2.deck)
assert set(state2.deck) == set(state.deck), "shuffle should preserve membership"
assert state2.deck != state.deck, "shuffle should change order (statistically near-certain)"

mb_b, ob_b, flat_b = encode_state(state2, player=0)
check("my_board identical under deck shuffle",  np.array_equal(mb_a,  mb_b))
check("opp_board identical under deck shuffle", np.array_equal(ob_a,  ob_b))
check("flat identical under deck shuffle",      np.array_equal(flat_a, flat_b))


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 7: phase one-hot is correct in all three phases ===")
state = GameState.new(seed=4)
_, _, flat = encode_state(state, player=0)
ph = flat[FLAT_LAYOUT['phase']]
check("phase=INITIAL_SELECTION → [1,0,0]", np.array_equal(ph, [1.,0.,0.]))

# Step to PLACE_AND_SELECT
state = play_random_to_phase(state, Phase.PLACE_AND_SELECT, random.Random(4))
if state.phase == Phase.PLACE_AND_SELECT:
    _, _, flat = encode_state(state, player=0)
    ph = flat[FLAT_LAYOUT['phase']]
    check("phase=PLACE_AND_SELECT → [0,1,0]", np.array_equal(ph, [0.,1.,0.]))

# Step to FINAL_PLACEMENT
state = play_random_to_phase(state, Phase.FINAL_PLACEMENT, random.Random(4))
if state.phase == Phase.FINAL_PLACEMENT:
    _, _, flat = encode_state(state, player=0)
    ph = flat[FLAT_LAYOUT['phase']]
    check("phase=FINAL_PLACEMENT → [0,0,1]", np.array_equal(ph, [0.,0.,1.]))


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 8: terminal raises and target_z is correct ===")
# Force into terminal
state = GameState.new(seed=5)
rng = random.Random(5)
while state.phase != Phase.GAME_OVER:
    a = state.legal_actions()
    if not a: break
    state = state.step(rng.choice(a))

check("phase reaches GAME_OVER", state.phase == Phase.GAME_OVER)
try:
    encode_state(state, player=0)
    check("encode_state raises on terminal", False)
except ValueError:
    check("encode_state raises on terminal", True)

scores = state.scores()
margin = scores[0] - scores[1]
z = compute_target_z(state, player=0, sigma=30.0)
expected = math.tanh(margin / 30.0)
check("compute_target_z matches tanh(margin/30)", abs(z - expected) < 1e-9,
      f"got {z:.6f}, expected {expected:.6f}, margin={margin}")
z_opp = compute_target_z(state, player=1, sigma=30.0)
check("z is antisymmetric across players", abs(z + z_opp) < 1e-9,
      f"p0={z:.6f}, p1={z_opp:.6f}, sum={z+z_opp:.2e}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 9: D4 symmetry — encoder is rotation-friendly ===")
# We verify that applying rot90 to the spatial output yields another valid
# encoding (same channel structure, castle stays centred, terrain still one-hot,
# crowns/occupied still consistent).  We are NOT verifying that the rotated
# encoding corresponds to a rotated *board* — that requires rotating the board
# itself, which is a training-side concern.  What we check here is that the
# encoder's output is in a form the training loop can rotate freely.
state = GameState.new(seed=7)
rng = random.Random(7)
for _ in range(25):
    if state.phase == Phase.GAME_OVER: break
    a = state.legal_actions()
    if not a: break
    state = state.step(rng.choice(a))

mb, _, _ = encode_state(state, player=0)
for k in range(4):
    rot = np.rot90(mb, k=k, axes=(1, 2))
    # Castle stays at centre under any k (since centre is a fixed point)
    check(f"rot{k*90}: castle stays at centre",
          rot[CH_CASTLE, CASTLE_CENTER, CASTLE_CENTER] == 1.0
          and rot[CH_CASTLE].sum() == 1.0)
    # Terrain still one-hot at non-castle occupied cells
    occupied = rot[CH_OCCUPIED] > 0.5
    castle = rot[CH_CASTLE] > 0.5
    tsum = rot[CH_TERRAIN_START:CH_TERRAIN_END].sum(axis=0)
    err = ((tsum[occupied & ~castle] - 1.0) ** 2).sum()
    check(f"rot{k*90}: terrain still one-hot at occupied non-castle cells", err < 1e-6)

# Also check reflection
ref = mb[:, :, ::-1]
check("horizontal flip: castle stays centred",
      ref[CH_CASTLE, CASTLE_CENTER, CASTLE_CENTER] == 1.0
      and ref[CH_CASTLE].sum() == 1.0)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 10: player-perspective swap exchanges boards correctly ===")
state = GameState.new(seed=8)
rng = random.Random(8)
for _ in range(20):
    if state.phase == Phase.GAME_OVER: break
    a = state.legal_actions()
    if not a: break
    state = state.step(rng.choice(a))

mb0, ob0, flat0 = encode_state(state, player=0)
mb1, ob1, flat1 = encode_state(state, player=1)

check("encoding p1's my_board == p0's opp_board", np.array_equal(mb1, ob0))
check("encoding p1's opp_board == p0's my_board", np.array_equal(ob1, mb0))
# fill ratios should swap
fp0_mine = flat0[FLAT_LAYOUT['my_fill_ratio']][0]
fp0_opp  = flat0[FLAT_LAYOUT['opp_fill_ratio']][0]
fp1_mine = flat1[FLAT_LAYOUT['my_fill_ratio']][0]
fp1_opp  = flat1[FLAT_LAYOUT['opp_fill_ratio']][0]
check("fill ratios swap with perspective",
      abs(fp0_mine - fp1_opp) < 1e-9 and abs(fp0_opp - fp1_mine) < 1e-9)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 11: domino_in_hand and pending_claim alignment ===")
# When it's player 0's turn to place, flat[domino_in_hand] should match
# the domino they're about to place.
state = GameState.new(seed=12)
rng = random.Random(12)
state = play_random_to_phase(state, Phase.PLACE_AND_SELECT, random.Random(12))
# We're in PLACE_AND_SELECT; pending_claims has 4 entries; current actor is
# whoever holds pending_claims[actor_index].
acting_player = state.current_actor
in_hand_id = state.pending_claims[state.actor_index].domino_id
_, _, flat = encode_state(state, player=acting_player)
expected_tile = encoder._encode_tile(in_hand_id)
check("domino_in_hand matches current actor's pending claim",
      np.array_equal(flat[FLAT_LAYOUT['domino_in_hand']], expected_tile))

# From the opponent's perspective, domino_in_hand should be all zeros
_, _, flat_opp = encode_state(state, player=1 - acting_player)
check("domino_in_hand is zero from non-actor's perspective",
      not flat_opp[FLAT_LAYOUT['domino_in_hand']].any())


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 12: game_progress monotone increase across game ===")
state = GameState.new(seed=14)
rng = random.Random(14)
prev_progress = 0.0
violations = 0
for _ in range(80):
    if state.phase == Phase.GAME_OVER: break
    _, _, flat = encode_state(state, player=0)
    p = flat[FLAT_LAYOUT['game_progress']][0]
    if p < prev_progress - 1e-9:
        violations += 1
    prev_progress = p
    a = state.legal_actions()
    if not a: break
    state = state.step(rng.choice(a))
check("game_progress monotonically non-decreasing", violations == 0,
      f"violations={violations}")
# A random game produces some discards (cells never filled), so game_progress
# at termination is (1 - 2*discards/96). What we can guarantee is that the
# value is in [0, 1] and is non-trivial (> 0.5) by end of game.
check("game_progress at end-of-game is in [0, 1]",
      0.0 <= prev_progress <= 1.0, f"final progress={prev_progress:.4f}")
check("game_progress is non-trivial at end-of-game (>0.5)",
      prev_progress > 0.5, f"final progress={prev_progress:.4f}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 13: redeterminize preserves bag and changes order ===")
from games.kingdomino.encoder import redeterminize
state = GameState.new(seed=21)
# Step into mid-game where deck has many tiles left
rng = random.Random(21)
for _ in range(15):
    if state.phase == Phase.GAME_OVER: break
    state = state.step(rng.choice(state.legal_actions()))

orig_deck_set = set(state.deck)
orig_deck_list = list(state.deck)

new_state = redeterminize(state, random.Random(42))
check("redeterminize preserves bag membership",
      set(new_state.deck) == orig_deck_set)
check("redeterminize changes deck order (high-probability with many tiles)",
      list(new_state.deck) != orig_deck_list)
check("redeterminize does not mutate original state",
      list(state.deck) == orig_deck_list)
check("redeterminize preserves all public information",
      new_state.phase == state.phase
      and new_state.current_row == state.current_row
      and [(c.player, c.domino_id) for c in new_state.pending_claims]
          == [(c.player, c.domino_id) for c in state.pending_claims]
      and [(c.player, c.domino_id) for c in new_state.next_claims]
          == [(c.player, c.domino_id) for c in state.next_claims])

# CRITICAL property: encoder output is byte-identical under redeterminize.
# This is the same as TEST 6 (deck shuffle invariance), but via the public
# redeterminize API — proving the helper does what it's documented to do.
mb_a, ob_a, flat_a = encode_state(state, player=0)
mb_b, ob_b, flat_b = encode_state(new_state, player=0)
check("encode_state output is byte-identical under redeterminize",
      np.array_equal(mb_a, mb_b)
      and np.array_equal(ob_a, ob_b)
      and np.array_equal(flat_a, flat_b))


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 14: config validation in encode_state ===")
import dataclasses
from games.kingdomino.game import GameConfig

state = GameState.new(seed=0)

# Override config to flip the mighty_duel flag
state.config = dataclasses.replace(state.config, mighty_duel=False)
caught = False
try:
    encode_state(state, player=0)
except ValueError as e:
    caught = "mighty_duel" in str(e).lower() or "Mighty Duel" in str(e)
check("encode_state rejects non-Mighty-Duel config with informative error",
      caught)

# Restore mighty_duel, set players=3 (still illegal for this encoder)
state.config = dataclasses.replace(state.config, mighty_duel=True, players=3)
caught = False
try:
    encode_state(state, player=0)
except ValueError as e:
    caught = "players" in str(e).lower()
check("encode_state rejects non-2-player config with informative error",
      caught)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 15: compute_target_z enforces terminal + config ===")
# Non-terminal state — should raise
state = GameState.new(seed=7)
caught = False
try:
    compute_target_z(state, player=0)
except ValueError as e:
    caught = "terminal" in str(e).lower() or "GAME_OVER" in str(e)
check("compute_target_z raises on non-terminal state", caught)

# Play to terminal
rng = random.Random(7)
while state.phase != Phase.GAME_OVER:
    a = state.legal_actions()
    if not a: break
    state = state.step(rng.choice(a))

# Terminal but with bad config — should raise
saved_config = state.config
state.config = dataclasses.replace(saved_config, mighty_duel=False)
caught = False
try:
    compute_target_z(state, player=0)
except ValueError as e:
    caught = "Mighty Duel" in str(e) or "mighty_duel" in str(e).lower()
check("compute_target_z raises on non-Mighty-Duel terminal state", caught)

# Restore and verify the happy path still works
state.config = saved_config
try:
    z = compute_target_z(state, player=0)
    ok = -1.0 <= z <= 1.0
except Exception:
    ok = False
check("compute_target_z returns z ∈ [-1, 1] on valid terminal", ok)

# Invalid player index
caught = False
try:
    compute_target_z(state, player=5)
except ValueError:
    caught = True
check("compute_target_z raises on invalid player index", caught)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 16: pick_pos_0..3 targeted table (both perspectives + Rust) ===")


def _pos_of(flat):
    return tuple(float(flat[FLAT_LAYOUT[f"pick_pos_{k}"]][0]) for k in range(4))


def _base_state(phase, rng_seed):
    """A real state in the requested phase (valid pending_claims/current_actor)."""
    if phase == Phase.INITIAL_SELECTION:
        return GameState.new(seed=rng_seed)
    return play_random_to_phase(GameState.new(seed=rng_seed), phase,
                                random.Random(rng_seed))


# (phase, next_claims as [(player, domino_id)], expected pos from player-0 view)
_PICK_CASES = [
    (Phase.INITIAL_SELECTION, [],                                  (0.0, 0.0, 0.0, 0.0)),
    (Phase.INITIAL_SELECTION, [(0, 12)],                           (0.0, 0.0, 0.0, 0.0)),
    (Phase.PLACE_AND_SELECT,  [],                                  (0.0, 0.0, 0.0, 0.0)),
    (Phase.PLACE_AND_SELECT,  [(0, 20)],                           (1.0, 0.0, 0.0, 0.0)),
    (Phase.PLACE_AND_SELECT,  [(1, 10)],                           (-1.0, 0.0, 0.0, 0.0)),
    (Phase.PLACE_AND_SELECT,  [(0, 30), (1, 10)],                  (-1.0, 1.0, 0.0, 0.0)),
    (Phase.PLACE_AND_SELECT,  [(0, 5), (1, 20)],                   (1.0, -1.0, 0.0, 0.0)),
    (Phase.PLACE_AND_SELECT,  [(0, 5), (0, 40), (1, 20)],          (1.0, -1.0, 1.0, 0.0)),
    (Phase.PLACE_AND_SELECT,  [(0, 5), (0, 40), (1, 20), (1, 45)], (1.0, -1.0, 1.0, -1.0)),
    (Phase.PLACE_AND_SELECT,  [(1, 5), (1, 15), (0, 25), (0, 35)], (-1.0, -1.0, 1.0, 1.0)),
    (Phase.FINAL_PLACEMENT,   [(0, 5), (1, 20)],                   (0.0, 0.0, 0.0, 0.0)),
]

for ci, (phase, claims, exp0) in enumerate(_PICK_CASES, start=1):
    base = _base_state(phase, 50 + ci)
    if base.phase != phase:
        check(f"case {ci}: reached {phase.name}", False, f"got {base.phase.name}")
        continue
    base.next_claims = [Claim(p, d) for (p, d) in claims]

    _, _, f0 = encode_state(base, player=0)
    got0 = _pos_of(f0)
    check(f"case {ci} ({phase.name}, {claims}) p0 == {exp0}", got0 == exp0,
          f"got {got0}")

    # Perspective flip: player 1 must see the exact negation (0s stay 0).
    _, _, f1 = encode_state(base, player=1)
    got1 = _pos_of(f1)
    exp1 = tuple(-v for v in exp0)
    check(f"case {ci} p1 perspective flip == {exp1}", got1 == exp1, f"got {got1}")


# Rust cross-check via lockstep: pick_pos is determinization-independent, so we
# step a Python GameState and a mirrored RustGameState together and compare the
# pick_pos flat slice at every non-terminal state across a few games.
try:
    import kingdomino_rust as _kr
    from games.kingdomino.game import PickAction as _PA, TurnAction as _TA
    _OFF0 = FLAT_LAYOUT['pick_pos_0'].start  # 257

    def _rs_from_py(py):
        return _kr.RustGameState(py.start_player, list(py.deck), list(py.current_row),
                                 py.config.harmony, py.config.middle_kingdom)

    def _translate(action):
        if isinstance(action, _PA):
            return (None, action.domino_id)
        p = action.placement
        pt = None if p is None else (p.x1, p.y1, p.x2, p.y2, p.flipped)
        return (pt, action.pick_domino_id)

    max_diff = 0.0
    plies = 0
    for g in range(4):
        py = GameState.new(seed=300 + g)
        rs = _rs_from_py(py)
        rng = random.Random(g * 2654435761 & 0xFFFFFFFF)
        while py.phase != Phase.GAME_OVER:
            _, _, fpy = encode_state(py, player=0)
            frs = np.asarray(rs.encode(0)[2])
            d = float(np.abs(fpy[_OFF0:_OFF0 + 4] - frs[_OFF0:_OFF0 + 4]).max())
            max_diff = max(max_diff, d)
            plies += 1
            a = rng.choice(py.legal_actions())
            py = py.step(a)
            rs = rs.step(*_translate(a))
    check(f"Rust pick_pos matches Python within 1e-6 ({plies} states)",
          max_diff < 1e-6, f"max_diff={max_diff:.2e}")
except ImportError:
    print("  SKIP  kingdomino_rust not built — Rust pick_pos cross-check skipped")


# ──────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
if _failures:
    print(f"FAILED: {len(_failures)} test(s)")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")