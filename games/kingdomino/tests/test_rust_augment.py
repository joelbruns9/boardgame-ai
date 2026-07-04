"""
test_rust_augment.py — verify the Rust fast paths added in the perf phase:

  A. kingdomino_rust.d4_augment  ==  augmentation.augment  (byte-identical
     array components) for all 8 D4 transforms.
  B. RustGameState.legal_mask()  ==  action_codec.legal_mask(PythonGameState)
     at every ply of a full game (driven in lockstep).

Both are equivalence gates: the Rust path must produce bit-for-bit the same
arrays the numpy/Python path produces, or training data would silently differ
depending on whether the extension is built.

Run (PowerShell, one line):
  python -m games.kingdomino.tests.test_rust_augment
"""
from __future__ import annotations

import random
import sys

import numpy as np

import kingdomino_rust as kr
from kingdomino_rust import d4_augment

from games.kingdomino.game import GameState, Phase, PickAction, TurnAction
from games.kingdomino.augmentation import augment, NUM_D4_TRANSFORMS
from games.kingdomino.action_codec import legal_mask as py_legal_mask
from games.kingdomino.encoder import FLAT_SIZE


# ─── A. d4_augment byte-identical to augment() ──────────────────────────────
def test_rust_augment_matches_python() -> bool:
    """d4_augment (Rust) byte-identical to augment (Python) for all 8 transforms."""
    my_board = np.random.default_rng(0).random((9, 13, 13), dtype=np.float32)
    opp_board = np.random.default_rng(1).random((9, 13, 13), dtype=np.float32)
    flat = np.random.default_rng(2).random((FLAT_SIZE,), dtype=np.float32)
    policy = np.random.default_rng(3).random((3390,), dtype=np.float32)

    ok = True
    for t in range(NUM_D4_TRANSFORMS):
        mb_r, ob_r, fl_r, pol_r = d4_augment(my_board, opp_board, flat, policy, t)
        py_result = augment(my_board, opp_board, flat, policy, 0.0, t,
                            own_score=0.0, opp_score=0.0, win_target=0.5)  # diagnostic placeholders
        mb_p, ob_p, fl_p, pol_p = py_result[0], py_result[1], py_result[2], py_result[3]

        for name, r, p in (("my_board", mb_r, mb_p), ("opp_board", ob_r, ob_p),
                           ("flat", fl_r, fl_p), ("policy", pol_r, pol_p)):
            if not np.array_equal(r, p):
                ok = False
                n_diff = int(np.count_nonzero(np.asarray(r) != np.asarray(p)))
                print(f"  [t={t}] {name} differs ({n_diff} elements)")
    if ok:
        print("PASS: d4_augment Rust == Python for all 8 transforms")
    else:
        print("FAIL: d4_augment Rust != Python")
    return ok


# ─── B. RustGameState.legal_mask() == action_codec.legal_mask() ─────────────
def _rust_from_python(py: GameState):
    return kr.RustGameState(
        py.start_player, list(py.deck), list(py.current_row),
        py.config.harmony, py.config.middle_kingdom,
    )


def _translate(action) -> tuple:
    """Convert a Python engine action to RustGameState.step(*args)."""
    if isinstance(action, PickAction):
        return (None, action.domino_id)
    p = action.placement
    ptuple = None if p is None else (p.x1, p.y1, p.x2, p.y2, p.flipped)
    return (ptuple, action.pick_domino_id)


def test_rust_legal_mask_matches_python(n_games: int = 5, seed: int = 7) -> bool:
    """RustGameState.legal_mask() byte-identical to action_codec.legal_mask(Python)
    at every ply across several full games driven in lockstep."""
    ok = True
    plies = 0
    for g in range(n_games):
        py = GameState.new(seed=seed + g)
        rs = _rust_from_python(py)
        rng = random.Random((seed + g) * 2654435761 & 0xFFFFFFFF)

        while py.phase != Phase.GAME_OVER:
            # py_legal_mask dispatches to the PYTHON encode_action loop (GameState
            # has no legal_mask()); rs.legal_mask() is the Rust method.
            mask_py = py_legal_mask(py)
            mask_rs = rs.legal_mask()
            if not np.array_equal(mask_py, mask_rs):
                ok = False
                n_diff = int(np.count_nonzero(mask_py != mask_rs))
                print(f"  [game {g}] mask mismatch at phase {py.phase.name} "
                      f"({n_diff} indices differ)")
                break

            action = rng.choice(py.legal_actions())
            py = py.step(action)
            rs = rs.step(*_translate(action))
            plies += 1

    if ok:
        print(f"PASS: RustGameState.legal_mask() == action_codec.legal_mask() "
              f"across {n_games} games ({plies} plies)")
    else:
        print("FAIL: legal_mask Rust != Python")
    return ok


def main() -> int:
    a = test_rust_augment_matches_python()
    b = test_rust_legal_mask_matches_python()
    return 0 if (a and b) else 1


if __name__ == "__main__":
    sys.exit(main())
