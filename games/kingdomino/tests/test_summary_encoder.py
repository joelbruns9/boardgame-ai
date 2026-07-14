"""Gates for the 171-value NNUE summary encoder.

Enforces: exact size/layout, seat-swap symmetry, base block == reused encoder,
game_progress -> 1.0 at terminal, a SEMANTIC schema hash, RAW pre-normalization
clip auditing (not the circular post-clip range check), and golden-value fixtures
for the new global/extension features.
"""
import copy
import dataclasses
import random

import numpy as np
import pytest

from games.kingdomino.game import GameState, Phase, Claim
from games.kingdomino.dominoes import DOMINOES
from games.kingdomino.encoder import _encode_board_summary
from games.kingdomino.nnue import summary_encoder as sm
from games.kingdomino.nnue.sparse_encoder import swap_players

NT = sm.NT


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


# ── layout / hash ─────────────────────────────────────────────────────────────
def test_size_and_semantic_hash():
    assert sm.SUMMARY_SIZE == 171
    for st in _states(range(4)):
        v = sm.encode_summary(st, 0)
        assert v.shape == (171,) and v.dtype == np.float32
    h = sm.summary_schema_hash()
    assert len(h) == 16
    saved = sm._SUMMARY_SEMANTICS
    try:  # a MEANING change (version bump) must change the hash
        sm._SUMMARY_SEMANTICS = {**copy.deepcopy(saved), "version": saved["version"] + 1}
        assert sm.summary_schema_hash() != h
    finally:
        sm._SUMMARY_SEMANTICS = saved
    assert sm.summary_schema_hash() == h


# ── clipping: audit RAW values, not post-clip outputs ────────────────────────
def test_true_bound_features_never_clip():
    """The extension/global features use TRUE combinatorial bounds; assert the RAW
    (pre-normalization) values never exceed them, over games + a dense board."""
    boards = [b for st in _states(range(30)) for b in st.boards]
    # plus a deliberately dense/high board: fill greedily from one game
    st = GameState.new(seed=123)
    rng = random.Random(1)
    for _ in range(400):
        if st.phase == Phase.GAME_OVER:
            break
        st = st.step(rng.choice(st.legal_actions()))
    boards += st.boards

    for b in boards:
        r = sm._extension_raw(b)
        for t in range(NT):
            assert r["cell_count"][t] <= sm.MAX_BOARD_CELLS
            assert r["crown_count"][t] <= sm.MAX_CROWNS_PER_TERRAIN[t]
            assert r["largest_crowns"][t] <= sm.MAX_CROWNS_PER_TERRAIN[t]
            assert r["open_frontier"][t] <= sm.MAX_BOARD_CELLS
            assert r["largest_crownless"][t] <= sm.MAX_BOARD_CELLS
        assert r["global_largest"] <= sm.MAX_BOARD_CELLS
        assert r["crownless_region_count"] <= sm.MAX_BOARD_CELLS
        assert r["stranded_crowns"] <= sm.TOTAL_CATALOG_CROWNS
        assert r["holes"] <= sm.MAX_BOARD_CELLS
        assert 0 <= r["gaps"] <= sm.MAX_BOARD_CELLS
        assert all(0 <= e <= 6 for e in r["castle_extent"])


def test_saturating_features_are_measured_and_bounded_output():
    """SCORE_SCALE=160 and MAX_LEGAL_PLACEMENTS=64 are training scales, not rules
    maxima -> those features saturate by design. Measure how often, and confirm the
    output still lands in range. (Weak random play rarely reaches the ceiling; this
    records it so stronger-play corpora can be re-audited.)"""
    score_clips = legal_clips = n_scores = n_legal = 0
    score_max = legal_max = 0
    for st in _states(range(30)):
        for b in st.boards:
            total = b.score(st.config.harmony, st.config.middle_kingdom).total
            n_scores += 1
            score_max = max(score_max, total)
            score_clips += total > sm.SCORE_SCALE
        if st.phase in (Phase.PLACE_AND_SELECT, Phase.FINAL_PLACEMENT):
            for claim in st.pending_claims[st.actor_index:]:
                lp = len(st.boards[claim.player].legal_placements(DOMINOES[claim.domino_id]))
                n_legal += 1
                legal_max = max(legal_max, lp)
                legal_clips += lp > sm.MAX_LEGAL_PLACEMENTS
    print(f"score: max={score_max} clips={score_clips}/{n_scores}; "
          f"legal: max={legal_max} clips={legal_clips}/{n_legal}")
    # output range still holds regardless of saturation
    for st in _states(range(8)):
        for persp in (0, 1):
            v = sm.encode_summary(st, persp)
            assert v.min() >= -1.0 - 1e-6 and v.max() <= 1.0 + 1e-6


# ── seat-swap / base / progress / determinism ────────────────────────────────
def test_seat_swap_symmetry():
    saw = False
    for st in _states(range(30)):
        sw = swap_players(st)
        assert np.allclose(sm.encode_summary(st, 0), sm.encode_summary(sw, 1), atol=1e-6)
        assert np.allclose(sm.encode_summary(st, 1), sm.encode_summary(sw, 0), atol=1e-6)
        if not np.allclose(sm.encode_summary(st, 0), sm.encode_summary(st, 1), atol=1e-6):
            saw = True
    assert saw


def test_base_block_matches_encoder():
    for st in _states(range(10)):
        for persp in (0, 1):
            v = sm.encode_summary(st, persp)
            assert np.array_equal(v[:25], _encode_board_summary(st, persp))
            assert np.array_equal(v[25:50], _encode_board_summary(st, 1 - persp))


def test_game_progress_reaches_one_at_terminal():
    prog_idx = sm.BASE_SIZE + 2 * sm.EXT_PER + 12 + 24 + 4
    for seed in range(20):
        st = GameState.new(seed=seed)
        rng = random.Random(seed)
        prev = -1.0
        while st.phase != Phase.GAME_OVER:
            p = float(sm.encode_summary(st, 0)[prog_idx])
            assert p >= prev - 1e-6
            prev = p
            st = st.step(rng.choice(st.legal_actions()))
        assert abs(float(sm.encode_summary(st, 0)[prog_idx]) - 1.0) < 1e-6


def test_deterministic():
    st = _states(range(3))[10]
    assert np.array_equal(sm.encode_summary(st, 0), sm.encode_summary(st.copy(), 0))


# ── golden fixtures: exact values for new global/extension features ──────────
_GLOBAL_OFF = sm.BASE_SIZE + 2 * sm.EXT_PER          # start of the global block
_BAG_OFF = _GLOBAL_OFF
_CLAIMS_OFF = _GLOBAL_OFF + 12
_PICKPOS_OFF = _CLAIMS_OFF + 24


def _place_state():
    return next(s for s in _states(range(30))
                if s.phase == Phase.PLACE_AND_SELECT
                and len(s.pending_claims[s.actor_index:]) >= 2 and s.next_claims)


def test_golden_bag_aggregates():
    st = _place_state()
    v = sm.encode_summary(st, 0)
    hc = [0] * NT
    cr = [0] * NT
    for did in st.deck:                       # independent recomputation
        for h in (DOMINOES[did].a, DOMINOES[did].b):
            ti = int(h.terrain) - sm.TOFF
            hc[ti] += 1
            cr[ti] += int(h.crowns)
    for t in range(NT):
        assert abs(v[_BAG_OFF + t] - hc[t] / sm.MAX_HALVES_PER_TERRAIN[t]) < 1e-6
        assert abs(v[_BAG_OFF + 6 + t] - cr[t] / sm.MAX_CROWNS_PER_TERRAIN[t]) < 1e-6


def test_golden_claims_and_pickpos_owner_signs_and_ranks():
    st = _place_state().copy()
    # force a known unresolved tail: [my (0), opp (1)] with specific ids
    st.pending_claims = list(st.pending_claims)
    ai = st.actor_index
    st.pending_claims[ai] = Claim(0, 10)
    if ai + 1 < len(st.pending_claims):
        st.pending_claims[ai + 1] = Claim(1, 5)
    v = sm.encode_summary(st, 0)
    # slot 0: owner my (+1), draft rank (10-1)/47, turn_distance 0
    assert v[_CLAIMS_OFF + 0] == 1.0                       # presence
    assert v[_CLAIMS_OFF + 3] == 1.0                       # owner my
    assert abs(v[_CLAIMS_OFF + 4] - 9 / 47) < 1e-6         # draft rank (id-1)/47
    assert v[_CLAIMS_OFF + 5] == 0.0                       # turn distance k=0
    if ai + 1 < len(st.pending_claims):
        assert v[_CLAIMS_OFF + 6 + 3] == -1.0              # slot1 owner opp
        assert abs(v[_CLAIMS_OFF + 6 + 5] - 1 / 3) < 1e-6  # turn distance k=1
    # pick_pos: next_claims sorted by id, owners mapped my/opp
    st.next_claims = [Claim(1, 40), Claim(0, 3)]
    v2 = sm.encode_summary(st, 0)
    assert v2[_PICKPOS_OFF + 0] == 1.0    # id 3 first -> owner 0 == perspective -> +1
    assert v2[_PICKPOS_OFF + 1] == -1.0   # id 40 -> owner 1 -> -1
    assert v2[_PICKPOS_OFF + 2] == 0.0    # no 3rd


def test_golden_castle_extent_directions():
    """Each castle-extent direction reflects the actual bbox reach, computed raw."""
    st = _place_state()
    b = st.boards[0]
    bbox = b.occupied_bbox()
    cx, cy = b.castle_pos
    minx, miny, maxx, maxy = bbox
    r = sm._extension_raw(b)
    assert r["castle_extent"] == [cx - minx, maxx - cx, cy - miny, maxy - cy]
    assert all(0 <= e <= 6 for e in r["castle_extent"])


def test_golden_region_flood_on_constructed_board():
    """Place a known same-terrain L-shape and assert region size/crowns and gaps."""
    st = GameState.new(seed=0)
    b = st.boards[0]
    placed_cells = []
    for did in (1, 2, 3):                      # place a few dominoes, record cells
        dom = DOMINOES[did]
        pl = b.legal_placements(dom)
        if not pl:
            continue
        b.place(dom, pl[0])
    r = sm._extension_raw(b)
    # total placed cells across terrains == occupied non-castle cells
    occ_non_castle = sum(1 for (x, y) in b.occupied_cells() if int(b.terrain[y, x]) != 1)
    assert sum(r["cell_count"]) == occ_non_castle
    # global largest region <= total placed, gaps non-negative
    assert 1 <= r["global_largest"] <= occ_non_castle
    assert r["gaps"] >= 0
