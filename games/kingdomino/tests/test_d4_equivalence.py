"""D4 augmentation identity gate: encode(transform(s)) == Dmap(encode(s)).

Validates the group action on the state (transform_state) and the induced index
permutations for BOTH encoders, over random trajectories x all 8 D4 elements.
This is the blocker before any sparse/summary data augmentation.
"""
import random

import numpy as np
import pytest

from games.kingdomino.game import GameState, Phase
from games.kingdomino.nnue import d4
from games.kingdomino.nnue.sparse_encoder import encode_core, BOARD_SIZE
from games.kingdomino.nnue.summary_encoder import encode_summary


def _states(seeds=range(25)):
    out = []
    for seed in seeds:
        st = GameState.new(seed=seed)
        rng = random.Random(seed * 5 + 1)
        while st.phase != Phase.GAME_OVER:
            out.append(st)
            st = st.step(rng.choice(st.legal_actions()))
        out.append(st)
    return out


# ── group action on the state is geometry-preserving ─────────────────────────
def test_transform_state_preserves_invariants():
    changed = 0
    for st in _states(range(12)):
        for k, flip in d4.D4_ELEMENTS:
            t = d4.transform_state(st, k, flip)
            for orig, tb in zip(st.boards, t.boards):
                assert tb.castle_pos == orig.castle_pos == (7, 7)
                assert len(tb.occupied_cells()) == len(orig.occupied_cells())
                # D4 preserves adjacency/connectivity -> identical official score
                so = orig.score(st.config.harmony, st.config.middle_kingdom)
                stf = tb.score(st.config.harmony, st.config.middle_kingdom)
                assert so.total == stf.total
                assert so.largest_territory_size == stf.largest_territory_size
                assert so.total_crowns == stf.total_crowns
                if (k, flip) != (0, False) and not np.array_equal(tb.terrain, orig.terrain):
                    changed += 1
    assert changed > 0, "transforms never moved a tile — test is vacuous"


def test_identity_element_is_identity():
    p = d4.sparse_perm(0, False)
    assert np.array_equal(p, np.arange(len(p)))
    assert not d4.axis_swap(0, False)
    assert d4._dir_map(0, False) == [0, 1, 2, 3]


def test_cell_perm_is_a_bijection_for_all_elements():
    for k, flip in d4.D4_ELEMENTS:
        fwd = d4.cell_perm(k, flip)
        assert sorted(fwd.tolist()) == list(range(len(fwd)))
        # castle-adjacent cells stay adjacent to the (fixed) castle centre
        assert set(d4._dir_map(k, flip)) == {0, 1, 2, 3}


# ── the core identity: sparse ────────────────────────────────────────────────
def test_sparse_augmentation_identity():
    for st in _states(range(25)):
        for persp in (0, 1):
            base = encode_core(st, persp)
            for k, flip in d4.D4_ELEMENTS:
                got = set(encode_core(d4.transform_state(st, k, flip), persp).tolist())
                want = d4.apply_sparse(base.tolist(), k, flip)
                assert got == want, f"sparse mismatch persp={persp} k={k} flip={flip}"


def test_sparse_nonboard_banks_are_invariant():
    """Every active index >= BOARD_SIZE must be identical before/after transform
    (only the board bank is geometric)."""
    for st in _states(range(15)):
        base = encode_core(st, 0)
        nb = {i for i in base.tolist() if i >= BOARD_SIZE}
        for k, flip in d4.D4_ELEMENTS:
            t = encode_core(d4.transform_state(st, k, flip), 0)
            assert {i for i in t.tolist() if i >= BOARD_SIZE} == nb


# ── the core identity: summary ───────────────────────────────────────────────
def test_summary_augmentation_identity():
    for st in _states(range(25)):
        for persp in (0, 1):
            base = encode_summary(st, persp)
            for k, flip in d4.D4_ELEMENTS:
                dmap = d4.summary_perm(k, flip)
                got = encode_summary(d4.transform_state(st, k, flip), persp)
                want = dmap(base)
                assert np.allclose(got, want, atol=1e-6), \
                    f"summary mismatch persp={persp} k={k} flip={flip} " \
                    f"maxdiff={np.abs(got - want).max():.4g}"


def test_summary_width_height_actually_swaps_under_rotation():
    """Guard against a vacuous summary test: find a state whose width != height and
    confirm the 90-degree Dmap really swaps them."""
    st = next(s for s in _states(range(25))
              if abs(s.boards[0].occupied_bbox()[2] - s.boards[0].occupied_bbox()[0]
                     - (s.boards[0].occupied_bbox()[3] - s.boards[0].occupied_bbox()[1])) > 0)
    assert d4.axis_swap(1, False)          # a single rot90 swaps axes
    v = encode_summary(st, 0)
    assert v[20] != v[21]                  # width != height in the base block (idx 20/21)
    swapped = d4.summary_perm(1, False)(v)
    assert swapped[20] == v[21] and swapped[21] == v[20]
