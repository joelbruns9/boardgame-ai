"""
test_rust_encode_equiv.py — verify RustGameState.encode produces bit-for-bit
identical output to encoder.encode_state across many random game states.

Milestone 2 gate.  These encodings are deterministic integer/float fills — no
network involved — so the target is EXACT equality (np.array_equal).  Any
mismatch is a bug, not floating-point noise.

We reuse Milestone 1's lockstep walk (a RustGameState mirrored from a Python
GameState, stepped with identical random actions) so the two states are
guaranteed equal at every ply; then we encode BOTH from each player's
perspective and compare all three outputs (my_board, opp_board, flat).

THE BAG CROSS-CHECK (the one place the two implementations derive differently):
  - Python encoder._compute_bag derives the remaining-tile bag from
    board.domino_id per cell (plus row/claims).
  - RustGameState.encode derives it from deck membership (the deck is exactly
    the complement of row ∪ claims ∪ placed).
These should always agree, but they are independent derivations, so we check
explicitly that Python's bag slice equals a deck-membership vector — catching
any silent divergence at its source — in addition to comparing the full flat
vectors.

Run (PowerShell, one line):
  python -m games.kingdomino.test_rust_encode_equiv
"""
from __future__ import annotations

import random
import sys

import numpy as np

from games.kingdomino.game import GameState, Phase
from games.kingdomino.encoder import encode_state, FLAT_LAYOUT, NUM_DOMINOES
from games.kingdomino.test_rust_game_equiv import _rust_from_python, _translate

import kingdomino_rust

_BAG = FLAT_LAYOUT["bag"]  # slice into the flat vector


def _deck_membership_bag(deck) -> np.ndarray:
    """The bag as derived purely from deck membership (Rust's approach)."""
    out = np.zeros(NUM_DOMINOES, dtype=np.float32)
    for did in deck:
        out[int(did) - 1] = 1.0
    return out


def run_equiv(n_games: int = 1000, seed: int = 0, verbose: bool = False) -> bool:
    mb_mismatches = 0
    ob_mismatches = 0
    flat_mismatches = 0
    bag_derivation_mismatches = 0  # Python domino_id-bag vs deck-membership bag
    shape_dtype_mismatches = 0
    states_encoded = 0

    for g in range(n_games):
        py = GameState.new(seed=seed + g)
        rs = _rust_from_python(py)
        rng = random.Random((seed + g) * 2654435761 & 0xFFFFFFFF)

        while py.phase != Phase.GAME_OVER:
            for player in (0, 1):
                py_mb, py_ob, py_flat = encode_state(py, player)
                rs_mb, rs_ob, rs_flat = rs.encode(player)
                rs_mb = np.asarray(rs_mb)
                rs_ob = np.asarray(rs_ob)
                rs_flat = np.asarray(rs_flat)
                states_encoded += 1

                # Shapes / dtypes must match before value comparison is meaningful.
                if (rs_mb.shape != py_mb.shape or rs_ob.shape != py_ob.shape
                        or rs_flat.shape != py_flat.shape
                        or rs_mb.dtype != py_mb.dtype or rs_flat.dtype != py_flat.dtype):
                    shape_dtype_mismatches += 1
                    if verbose and shape_dtype_mismatches <= 5:
                        print(f"  [g{g} p{player}] shape/dtype: "
                              f"mb {rs_mb.shape}/{rs_mb.dtype} vs {py_mb.shape}/{py_mb.dtype}, "
                              f"flat {rs_flat.shape}/{rs_flat.dtype} vs {py_flat.shape}/{py_flat.dtype}")
                    continue

                if not np.array_equal(py_mb, rs_mb):
                    mb_mismatches += 1
                    if verbose and mb_mismatches <= 5:
                        _report_spatial(g, player, "my_board", py_mb, rs_mb)
                if not np.array_equal(py_ob, rs_ob):
                    ob_mismatches += 1
                    if verbose and ob_mismatches <= 5:
                        _report_spatial(g, player, "opp_board", py_ob, rs_ob)
                if not np.array_equal(py_flat, rs_flat):
                    flat_mismatches += 1
                    if verbose and flat_mismatches <= 5:
                        _report_flat(g, player, py_flat, rs_flat)

                # Bag derivation cross-check: Python's (domino_id-based) bag slice
                # must equal the deck-membership bag.  Only depends on the state,
                # so check it once per state (player 0).
                if player == 0:
                    py_bag = py_flat[_BAG]
                    deck_bag = _deck_membership_bag(py.deck)
                    if not np.array_equal(py_bag, deck_bag):
                        bag_derivation_mismatches += 1
                        if verbose and bag_derivation_mismatches <= 5:
                            only_pid = sorted((np.nonzero(py_bag)[0] + 1).tolist())
                            only_deck = sorted((np.nonzero(deck_bag)[0] + 1).tolist())
                            print(f"  [g{g}] BAG DERIVATION DIVERGES: "
                                  f"domino_id-bag={only_pid} deck-bag={only_deck}")

            action = rng.choice(py.legal_actions())
            py = py.step(action)
            rs = rs.step(*_translate(action))

    print(f"\n=== RustGameState.encode equivalence: {n_games} games, "
          f"{states_encoded} encodings ===")
    print(f"  my_board  mismatches:        {mb_mismatches}")
    print(f"  opp_board mismatches:        {ob_mismatches}")
    print(f"  flat      mismatches:        {flat_mismatches}")
    print(f"  shape/dtype mismatches:      {shape_dtype_mismatches}")
    print(f"  bag-derivation divergences:  {bag_derivation_mismatches}")
    ok = (mb_mismatches == 0 and ob_mismatches == 0 and flat_mismatches == 0
          and shape_dtype_mismatches == 0 and bag_derivation_mismatches == 0)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def _report_spatial(g, player, name, py_arr, rs_arr):
    diff = np.argwhere(py_arr != rs_arr)
    c, y, x = diff[0]
    print(f"  [g{g} p{player}] {name} differs at (c={c},y={y},x={x}): "
          f"py={py_arr[c, y, x]} rust={rs_arr[c, y, x]} ({len(diff)} cells)")


def _report_flat(g, player, py_flat, rs_flat):
    diff = np.nonzero(py_flat != rs_flat)[0]
    # Name the field each differing index falls in.
    fields = []
    for idx in diff[:8]:
        field = next((k for k, sl in FLAT_LAYOUT.items() if sl.start <= idx < sl.stop), "?")
        fields.append(f"{idx}({field}):py={py_flat[idx]:.6g},rust={rs_flat[idx]:.6g}")
    print(f"  [g{g} p{player}] flat differs at {len(diff)} idxs: {fields}")


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    n = 1000
    for a in sys.argv[1:]:
        if a.startswith("--games="):
            n = int(a.split("=", 1)[1])
    ok = run_equiv(n_games=n, verbose=verbose)
    sys.exit(0 if ok else 1)
