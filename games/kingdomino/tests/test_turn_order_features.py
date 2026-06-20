"""Turn-order and claim features in encode_state (updated for pick_pos_0..3).

Verifies the four next-round pick-POSITION features (pick_pos_0..pick_pos_3,
+1 encoded player / -1 opponent / 0 unknown-or-no-next-round) that replaced the
old two rank scalars, plus that the existing turn-order / claim signals and
information-set safety are intact.

Run with:  python -m games.kingdomino.tests.test_turn_order_features
"""
from __future__ import annotations

import random
import sys

import numpy as np

from games.kingdomino.game import GameState, Phase, PickAction, Claim
from games.kingdomino.encoder import (
    encode_state,
    redeterminize,
    FLAT_LAYOUT,
    FLAT_SIZE,
)


_failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        _failures.append(name)


def pos(flat) -> tuple:
    """Read the four pick_pos scalars out of the flat vector as a tuple."""
    return tuple(float(flat[FLAT_LAYOUT[f"pick_pos_{k}"]][0]) for k in range(4))


def play_random_to_phase(state, target_phase, rng, max_steps=200):
    """Step forward with random legal actions until target_phase (or terminal)."""
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
print("\n=== TEST 1: FLAT_SIZE is 261 ===")
check("FLAT_SIZE == 261", FLAT_SIZE == 261, f"got {FLAT_SIZE}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 2: pick_pos_0..3 feature slots exist with correct size ===")
for k in range(4):
    key = f"pick_pos_{k}"
    present = key in FLAT_LAYOUT
    check(f"FLAT_LAYOUT contains {key}", present)
    if present:
        sl = FLAT_LAYOUT[key]
        check(f"{key} size is 1", (sl.stop - sl.start) == 1,
              f"got {sl.stop - sl.start}")
# the four positions must be contiguous and pick_pos_3 the final feature
check("pick_pos_0..3 are ascending & contiguous",
      FLAT_LAYOUT["pick_pos_0"].stop == FLAT_LAYOUT["pick_pos_1"].start
      and FLAT_LAYOUT["pick_pos_1"].stop == FLAT_LAYOUT["pick_pos_2"].start
      and FLAT_LAYOUT["pick_pos_2"].stop == FLAT_LAYOUT["pick_pos_3"].start)
check("pick_pos_3 is the final feature",
      FLAT_LAYOUT["pick_pos_3"].stop == FLAT_SIZE)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 3: INITIAL_SELECTION, no next_claims -> all 0.0 ===")
state = GameState.new(seed=0, start_player=0)
check("phase is INITIAL_SELECTION", state.phase == Phase.INITIAL_SELECTION)
check("next_claims empty", len(state.next_claims) == 0)
_, _, flat = encode_state(state, player=0)
check("pick_pos all 0.0", pos(flat) == (0.0, 0.0, 0.0, 0.0), f"got {pos(flat)}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 4: INITIAL_SELECTION WITH a committed pick -> STILL all 0.0 ===")
# KEY behavioral change vs the old pick_rank encoding: opening-phase claims are
# first-round placement commitments, not next-round tempo signals, so they are
# NOT surfaced as pick_pos.  The old _pick_ranks WOULD have ranked them.
state = GameState.new(seed=0, start_player=0)
state = state.step(PickAction(state.current_row[0]))  # one opening pick
check("still INITIAL_SELECTION", state.phase == Phase.INITIAL_SELECTION)
check("one next_claim committed", len(state.next_claims) == 1)
_, _, flat = encode_state(state, player=0)
check("pick_pos still all 0.0 during INITIAL_SELECTION",
      pos(flat) == (0.0, 0.0, 0.0, 0.0), f"got {pos(flat)}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 5: PLACE_AND_SELECT next_claims -> +1/-1 by domino_id order ===")
# Reach a real PLACE_AND_SELECT state (valid pending_claims/current_actor), then
# override next_claims deterministically: P0 holds domino 10, P1 holds domino 20.
rng = random.Random(5)
base = play_random_to_phase(GameState.new(seed=5), Phase.PLACE_AND_SELECT, rng)
check("reached PLACE_AND_SELECT", base.phase == Phase.PLACE_AND_SELECT)
base.next_claims = [Claim(0, 10), Claim(1, 20)]
_, _, flat0 = encode_state(base, player=0)
check("p0: pick_pos = (+1, -1, 0, 0)", pos(flat0) == (1.0, -1.0, 0.0, 0.0),
      f"got {pos(flat0)}")
_, _, flat1 = encode_state(base, player=1)
check("p1 perspective flip: pick_pos = (-1, +1, 0, 0)",
      pos(flat1) == (-1.0, 1.0, 0.0, 0.0), f"got {pos(flat1)}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 6: fully committed -> two +1s, two -1s, sums to 0 + flip ===")
base.next_claims = [Claim(0, 5), Claim(0, 40), Claim(1, 20), Claim(1, 45)]
_, _, flat0 = encode_state(base, player=0)
# sorted by domino_id: 5(P0), 20(P1), 40(P0), 45(P1) -> (+1, -1, +1, -1)
check("p0: pick_pos = (+1, -1, +1, -1)", pos(flat0) == (1.0, -1.0, 1.0, -1.0),
      f"got {pos(flat0)}")
check("p0: fully-committed sums to 0", abs(sum(pos(flat0))) < 1e-6)
_, _, flat1 = encode_state(base, player=1)
check("p1 perspective is exact negation of p0",
      pos(flat1) == tuple(-v for v in pos(flat0)), f"got {pos(flat1)}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 7: FINAL_PLACEMENT -> all 0.0 (no next round) ===")
rng = random.Random(7)
state = play_random_to_phase(GameState.new(seed=3), Phase.FINAL_PLACEMENT, rng)
check("reached FINAL_PLACEMENT", state.phase == Phase.FINAL_PLACEMENT,
      f"got {state.phase.name}")
if state.phase == Phase.FINAL_PLACEMENT:
    for p in (0, 1):
        _, _, flat = encode_state(state, player=p)
        check(f"p{p} pick_pos all 0.0", pos(flat) == (0.0, 0.0, 0.0, 0.0),
              f"got {pos(flat)}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 8: actor_flag regression (1.0 for current actor) ===")
state = GameState.new(seed=0, start_player=0)
actor = state.current_actor
_, _, flat_actor = encode_state(state, player=actor)
_, _, flat_other = encode_state(state, player=1 - actor)
check("actor_flag == 1.0 for current actor",
      float(flat_actor[FLAT_LAYOUT["actor_flag"]][0]) == 1.0)
check("actor_flag == 0.0 for the other player",
      float(flat_other[FLAT_LAYOUT["actor_flag"]][0]) == 0.0)


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 9: encode_state output shapes ===")
state = GameState.new(seed=1)
mb, ob, flat = encode_state(state, player=0)
check("my_board shape (9,13,13)", mb.shape == (9, 13, 13), f"got {mb.shape}")
check("opp_board shape (9,13,13)", ob.shape == (9, 13, 13), f"got {ob.shape}")
check("flat shape (261,)", flat.shape == (261,), f"got {flat.shape}")


# ──────────────────────────────────────────────────────────────────────────
print("\n=== TEST 10: information-set safety (invariant under deck reshuffle) ===")
# Mid-game PLACE_AND_SELECT state with next_claims populated, so the pick_pos
# features are actually exercised by the invariance check.
rng = random.Random(123)
base = play_random_to_phase(GameState.new(seed=11), Phase.PLACE_AND_SELECT, rng)
base = base.step(rng.choice(base.legal_actions()))  # commit at least one next claim
check("mid-game state has next_claims", len(base.next_claims) >= 1)
all_identical = True
for s in range(10):
    shuffle_rng = random.Random(1000 + s)
    redet = redeterminize(base, shuffle_rng)
    for p in (0, 1):
        a = encode_state(base, p)
        b = encode_state(redet, p)
        if not all(np.array_equal(x, y) for x, y in zip(a, b)):
            all_identical = False
            break
    if not all_identical:
        break
check("encode_state identical across 10 deck reshuffles (both perspectives)",
      all_identical)


# ──────────────────────────────────────────────────────────────────────────
def main():
    if _failures:
        print(f"\n{len(_failures)} FAILED: {_failures}")
        sys.exit(1)
    print("\nAll turn-order feature tests passed")


main()
