"""Gates for the NNUE sparse-core encoder (the lossless public-state features).

Enforces the contracts the two-perspective / Markov-quotient design rests on:
  * schema size and index bounds,
  * seat-swap symmetry (encode(s,P0) == encode(swap(s),P1)),
  * losslessness (decode(encode(s)) == the engine's public fingerprint),
  * completeness (a one-field change changes the encoding),
  * inventory agreement with the 48-ID conservation ledger.
"""
import random

import numpy as np
import pytest

from games.kingdomino.game import GameState, Phase, Claim
from games.kingdomino.nnue import sparse_encoder as se


def _states(seeds=range(40), max_plies=200):
    """Mid-game PLACE/FINAL/INITIAL states with asymmetric boards."""
    out = []
    for seed in seeds:
        st = GameState.new(seed=seed)
        rng = random.Random(seed * 3 + 1)
        while st.phase != Phase.GAME_OVER:
            out.append(st)
            st = st.step(rng.choice(st.legal_actions()))
        out.append(st)  # terminal too
    return out


def test_schema_size_and_bounds():
    assert se.CORE_SIZE == 5710
    assert se.NUM_HALF == 16 and se.NUM_CELLS == 169 and se.NUM_DOMINOES == 48
    for st in _states(range(6)):
        for persp in (0, 1):
            idx = se.encode_core(st, persp)
            assert idx.dtype == np.int32
            assert (idx >= 0).all() and (idx < se.CORE_SIZE).all()
            assert len(idx) == len(set(idx.tolist()))  # unique


def test_seat_swap_invariant():
    """encode(s, P0) must equal encode(swap_players(s), P1), and vice versa."""
    saw_asymmetric = False
    for st in _states(range(40)):
        sw = se.swap_players(st)
        a0 = se.encode_core(st, 0)
        b1 = se.encode_core(sw, 1)
        assert np.array_equal(a0, b1), f"seat-swap broke (P0 vs swap/P1) at phase {st.phase}"
        a1 = se.encode_core(st, 1)
        b0 = se.encode_core(sw, 0)
        assert np.array_equal(a1, b0), "seat-swap broke (P1 vs swap/P0)"
        # confirm the fixtures actually exercise asymmetric boards (else vacuous)
        if not np.array_equal(se.encode_core(st, 0), se.encode_core(st, 1)):
            saw_asymmetric = True
    assert saw_asymmetric, "no asymmetric state seen; seat-swap test would be vacuous"


def test_lossless_fingerprint_roundtrip():
    """decode(encode(s, persp)) reconstructs exactly the engine's public state."""
    for st in _states(range(40)):
        for persp in (0, 1):
            fp = se.decode(se.encode_core(st, persp))
            assert fp == se.public_fingerprint(st, persp), f"lossless gate failed at {st.phase}"


def test_one_field_mutation_changes_encoding():
    """A single change to any public field must change the active index set."""
    st = next(s for s in _states(range(10))
              if s.phase == Phase.PLACE_AND_SELECT and s.deck and s.next_claims)
    base = se.encode_core(st, 0)

    def _changed(mut):
        s = st.copy()
        mut(s)
        return not np.array_equal(se.encode_core(s, 0), base)

    # bag: drop one hidden domino
    assert _changed(lambda s: setattr(s, "deck", list(s.deck[:-1])))
    # current row: drop one tile
    assert _changed(lambda s: setattr(s, "current_row", list(s.current_row[:-1])))
    # next-claim owner: flip who owns the first next claim
    assert _changed(lambda s: setattr(
        s, "next_claims",
        [Claim(1 - s.next_claims[0].player, s.next_claims[0].domino_id)] + list(s.next_claims[1:])))
    # discard flag: force one on
    assert _changed(lambda s: setattr(s, "discards", [s.discards[0] + 1, s.discards[1]]))
    # rules flag: disable harmony
    def _dis_harmony(s):
        s.config = s.config.__class__(**{**vars(s.config), "harmony": False}) \
            if hasattr(s.config, "__dict__") else s.config
    # config may be a frozen dataclass; use replace if so
    import dataclasses
    assert _changed(lambda s: setattr(s, "config", dataclasses.replace(s.config, harmony=False)))


def test_inventory_matches_conservation_ledger():
    """The row/pending/next/bag banks reconstruct exactly the visible unplaced-ID
    sets, so |unplaced-in-banks| + placed + discards == 48 (ties to conservation)."""
    for st in _states(range(20)):
        if st.phase == Phase.GAME_OVER:
            continue
        fp = se.decode(se.encode_core(st, 0))
        banks = set(fp.row) | {i for _, i in fp.pending} | {i for _, i in fp.next_claims} | set(fp.bag)
        placed = set()
        for b in st.boards:
            g = np.asarray(b.domino_id)
            placed |= {int(v) for v in np.unique(g[g > 0])}
        # banks are the UNPLACED public ids; placed are on boards; discards vanished.
        assert banks.isdisjoint(placed)
        assert len(banks) + len(placed) + sum(st.discards) == 48
