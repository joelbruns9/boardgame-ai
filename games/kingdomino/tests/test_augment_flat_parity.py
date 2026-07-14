"""Rust d4_augment on the NEW 333-flat with REALISTIC values.

test_augmentation.py TEST 18 checks Rust==NumPy for board+policy, but uses an
all-zeros flat and never compares the returned flat — so Rust's handling of the
board-summary flat fields is unverified. A stale FLAT_SIZE or a wrong flat
transform would slip through.

Every flat field is rotation-invariant EXCEPT the per-player bbox (width,
height), which SWAP under odd (90°/270°) rotations. This uses a real encoded
mid-game state (nonzero board summaries, mb != ob, width != height) and a
realistic multi-action policy, and for all 8 transforms asserts the returned flat
matches the NumPy `_transform_flat` reference (i.e. width/height swapped exactly
under the four odd rotations, invariant otherwise), plus board/policy parity.
"""
import random

import numpy as np
import pytest

from games.kingdomino.game import GameState, Phase
from games.kingdomino.encoder import encode_state
from games.kingdomino.action_codec import encode_action
from games.kingdomino.augmentation import (
    _transform_spatial, _transform_policy, _transform_flat, _D4_ELEMENTS,
    _FLAT_WH_PAIRS,
)

NUM_JOINT_ACTIONS = 3390


def _realistic_tuple(seed):
    st = GameState.new(seed=seed)
    rng = random.Random(seed)
    for _ in range(20):
        if st.phase == Phase.GAME_OVER:
            break
        st = st.step(rng.choice(st.legal_actions()))
    mb, ob, flat = encode_state(st, player=0)
    mb = np.ascontiguousarray(mb, dtype=np.float32)
    ob = np.ascontiguousarray(ob, dtype=np.float32)
    flat = np.ascontiguousarray(flat, dtype=np.float32)
    policy = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)
    idxs = [encode_action(a, st) for a in st.legal_actions()[:8]]
    for w, j in zip(range(len(idxs), 0, -1), idxs):
        policy[j] += float(w)
    if policy.sum() > 0:
        policy /= policy.sum()
    return mb, ob, flat, policy


def test_rust_d4_augment_flat_parity_and_wh_swap():
    rust_d4 = pytest.importorskip("kingdomino_rust").d4_augment

    saw_nonzero_flat = False
    saw_real_swap = False  # a fixture where width != height, so odd rotations DO change flat
    for seed in range(600, 616):
        mb, ob, flat, policy = _realistic_tuple(seed)
        saw_nonzero_flat |= bool(np.any(flat != 0))
        wh_differs = any(flat[wi] != flat[hi] for wi, hi in _FLAT_WH_PAIRS)
        assert not np.array_equal(mb, ob), "fixture: opp board must differ from mine"
        for t in range(8):
            mb_r, ob_r, fl_r, pol_r = rust_d4(mb, ob, flat, policy, t)
            k, flip, dir_perm = _D4_ELEMENTS[t]
            fl_ref = _transform_flat(flat, k)
            # flat matches the NumPy reference: width/height swapped iff k odd.
            assert np.array_equal(np.asarray(fl_r), fl_ref), f"flat mismatch under t={t}"
            if k % 2 == 1 and wh_differs:
                assert not np.array_equal(np.asarray(fl_r), flat), \
                    f"odd rotation t={t} must swap width/height when they differ"
                saw_real_swap = True
            elif k % 2 == 0:
                assert np.array_equal(np.asarray(fl_r), flat), \
                    f"even transform t={t} must leave flat unchanged"
            # board + policy match the validated NumPy reference.
            assert np.array_equal(np.asarray(mb_r), _transform_spatial(mb, k, flip))
            assert np.array_equal(np.asarray(ob_r), _transform_spatial(ob, k, flip))
            assert np.array_equal(np.asarray(pol_r),
                                  _transform_policy(policy, k, flip, dir_perm))
    assert saw_nonzero_flat, "expected nonzero board-summary flat in the fixtures"
    assert saw_real_swap, "expected at least one fixture with width != height to exercise the swap"


if __name__ == "__main__":
    test_rust_d4_augment_flat_parity_and_wh_swap()
    print("PASS: Rust d4_augment swaps width/height under odd rotations and matches NumPy.")
