"""Gates for the 171-value NNUE summary encoder.

Enforces: exact size/layout, value ranges with ZERO clipping (fixed catalog
norms), seat-swap symmetry (summary(s,P0)==summary(swap(s),P1)), the base block
matching the reused _encode_board_summary, and game_progress reaching 1.0 at
terminal (the reason for the (placed+discards)/48 redefinition).
"""
import random

import numpy as np
import pytest

from games.kingdomino.game import GameState, Phase
from games.kingdomino.encoder import _encode_board_summary
from games.kingdomino.nnue import summary_encoder as sm
from games.kingdomino.nnue.sparse_encoder import swap_players


def _states(seeds=range(30)):
    out = []
    for seed in seeds:
        st = GameState.new(seed=seed)
        rng = random.Random(seed * 3 + 7)
        while st.phase != Phase.GAME_OVER:
            out.append(st)
            st = st.step(rng.choice(st.legal_actions()))
        out.append(st)
    return out


def test_size_and_hash():
    assert sm.SUMMARY_SIZE == 171
    for st in _states(range(4)):
        v = sm.encode_summary(st, 0)
        assert v.shape == (171,) and v.dtype == np.float32
    assert len(sm.summary_schema_hash()) == 16


def test_range_and_zero_clipping():
    """Every value in [-1, 1]; since the code never clips, staying <= 1 proves the
    fixed catalog/rules norm caps are true upper bounds (0 clipping)."""
    hi = -np.inf
    lo = np.inf
    for st in _states(range(30)):
        for persp in (0, 1):
            v = sm.encode_summary(st, persp)
            assert np.isfinite(v).all()
            hi = max(hi, float(v.max()))
            lo = min(lo, float(v.min()))
    assert hi <= 1.0 + 1e-6, f"a feature exceeded its norm cap (max {hi}) -> clipping"
    assert lo >= -1.0 - 1e-6, f"a feature below -1 (min {lo})"


def test_seat_swap_symmetry():
    """summary(s, P0) == summary(swap_players(s), P1), and vice versa."""
    saw_asym = False
    for st in _states(range(30)):
        sw = swap_players(st)
        assert np.allclose(sm.encode_summary(st, 0), sm.encode_summary(sw, 1), atol=1e-6)
        assert np.allclose(sm.encode_summary(st, 1), sm.encode_summary(sw, 0), atol=1e-6)
        if not np.allclose(sm.encode_summary(st, 0), sm.encode_summary(st, 1), atol=1e-6):
            saw_asym = True
    assert saw_asym, "no asymmetric summary seen; seat-swap test would be vacuous"


def test_base_block_matches_encoder():
    """The first 50 values are exactly _encode_board_summary for [my, opp]."""
    for st in _states(range(10)):
        for persp in (0, 1):
            v = sm.encode_summary(st, persp)
            assert np.array_equal(v[:25], _encode_board_summary(st, persp))
            assert np.array_equal(v[25:50], _encode_board_summary(st, 1 - persp))


def test_game_progress_reaches_one_at_terminal():
    """The redefinition ((placed+discards)/48) must hit 1.0 at game end even with
    discards, and increase monotonically over a game."""
    saw_terminal = False
    prog_idx = sm.BASE_SIZE + 2 * sm.EXT_PER + 12 + 24 + 4  # game_progress position
    for seed in range(20):
        st = GameState.new(seed=seed)
        rng = random.Random(seed)
        prev = -1.0
        while st.phase != Phase.GAME_OVER:
            p = float(sm.encode_summary(st, 0)[prog_idx])
            assert p >= prev - 1e-6, "game_progress went backwards"
            prev = p
            st = st.step(rng.choice(st.legal_actions()))
        final = float(sm.encode_summary(st, 0)[prog_idx])
        assert abs(final - 1.0) < 1e-6, f"terminal game_progress {final} != 1.0"
        saw_terminal = True
    assert saw_terminal


def test_deterministic():
    st = _states(range(3))[10]
    a = sm.encode_summary(st, 0)
    b = sm.encode_summary(st.copy(), 0)
    assert np.array_equal(a, b)
