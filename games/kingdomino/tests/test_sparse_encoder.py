"""Gates for the NNUE sparse-core encoder (the lossless public-state features).

Hardened per review so the proof is strong and NON-circular before the training
corpus is generated:
  * golden fixtures assert exact indices via RAW arithmetic (no encoder helpers),
  * seat-swap is enforced on every enumerated hand-relevant case, not just "some
    asymmetric state",
  * completeness covers every bank incl. board position/half-type, id REPLACEMENT,
    owner, actor, slot, phase/terminal, both rules, both roles,
  * inventory uses a Counter (cross-bank duplication can't hide), incl. terminal,
  * hidden-deck-order and perspective/id validation have direct regression gates.
"""
import dataclasses
import random
from collections import Counter, defaultdict

import numpy as np
import pytest

from games.kingdomino.game import GameState, Phase, Claim
from games.kingdomino.dominoes import DOMINOES
from games.kingdomino.nnue import sparse_encoder as se


# ── state collection ─────────────────────────────────────────────────────────
def _states(seeds=range(40)):
    out = []
    for seed in seeds:
        st = GameState.new(seed=seed)
        rng = random.Random(seed * 3 + 1)
        while st.phase != Phase.GAME_OVER:
            out.append(st)
            st = st.step(rng.choice(st.legal_actions()))
        out.append(st)
    return out


def _first(pred, seeds=range(60)):
    for st in _states(seeds):
        if pred(st):
            return st
    raise AssertionError("no state matched predicate")


# ── schema + semantic hash ───────────────────────────────────────────────────
def test_schema_size_and_semantic_hash():
    assert se.CORE_SIZE == 5710
    assert se.NUM_HALF == 16 and se.NUM_CELLS == 169 and se.NUM_DOMINOES == 48
    h = se.core_schema_hash()
    assert isinstance(h, str) and len(h) == 16
    # the hash must cover MEANINGS, not just dims: perturbing a semantic descriptor
    # (here, the version) must change it.
    import copy
    saved = se._SEMANTICS
    try:
        se._SEMANTICS = {**copy.deepcopy(saved), "version": saved["version"] + 1}
        assert se.core_schema_hash() != h
    finally:
        se._SEMANTICS = saved
    assert se.core_schema_hash() == h  # restored


# ── golden fixtures (RAW arithmetic; break circularity with encoder helpers) ──
def test_golden_board_indices_raw():
    """Place a real domino and assert its board features at RAW-computed indices."""
    st = GameState.new(seed=0)
    b = st.boards[0]
    dom = DOMINOES[7]
    b.place(dom, b.legal_placements(dom)[0])
    idx = set(se.encode_core(st, 0).tolist())
    placed = [(x, y) for (x, y) in b.occupied_cells() if int(b.terrain[y, x]) != 1]
    assert len(placed) == 2
    for (x, y) in placed:
        dx, dy = x - 7, y - 7                        # raw castle-relative
        assert -6 <= dx <= 6 and -6 <= dy <= 6
        cell = (dy + 6) * 13 + (dx + 6)              # raw row-major
        half = se.HALF_TYPES.index((int(b.terrain[y, x]), int(b.crowns[y, x])))  # via .index
        expected = 0 + (0 * 169 + cell) * 16 + half  # BOARD_OFF=0, role0=my
        assert expected in idx, f"board feature for ({dx},{dy}) missing"
    # opponent (board1) is empty -> no role-1 board features
    assert all(i >= se.ROW_OFF or i // (169 * 16) == 0 for i in idx if i < se.ROW_OFF)


def test_golden_decode_raw():
    """A hand-built index decodes to the RAW (dx,dy,half) it was built from."""
    dx, dy, half = 1, -2, 5
    cell = (dy + 6) * 13 + (dx + 6)
    board_idx = (0 * 169 + cell) * 16 + half          # role0
    row_idx = 2 * 169 * 16 + (7 - 1)                  # ROW_OFF raw + domino 7
    fp = se.decode([board_idx, row_idx])
    assert fp.board_my == ((cell, half),)
    r_dy, r_dx = divmod(cell, 13)                     # raw invert
    assert (r_dx - 6, r_dy - 6) == (dx, dy)
    assert fp.row == (7,)
    assert se.ROW_OFF == 2 * 169 * 16                 # bank offset is what we assumed


# ── seat-swap: universal + every enumerated case ─────────────────────────────
def test_seat_swap_universal():
    saw_asymmetric = False
    for st in _states(range(40)):
        assert np.array_equal(se.encode_core(st, 0), se.encode_core(se.swap_players(st), 1))
        assert np.array_equal(se.encode_core(st, 1), se.encode_core(se.swap_players(st), 0))
        if not np.array_equal(se.encode_core(st, 0), se.encode_core(st, 1)):
            saw_asymmetric = True
    assert saw_asymmetric


def test_seat_swap_enumerated_cases():
    """Every case the plan flags as historically fragile must be present AND obey
    seat-swap -- not merely 'some asymmetric state occurred'."""
    buckets = defaultdict(list)
    for st in _states(range(150)):
        p = st.phase
        if p == Phase.INITIAL_SELECTION:
            buckets[f"init_{st.initial_pick_count}"].append(st)
            if st.next_claims and not st.pending_claims[st.actor_index:]:
                buckets["next_without_pending"].append(st)
        elif p == Phase.PLACE_AND_SELECT:
            buckets[f"actor_index_{st.actor_index}"].append(st)
            tail = [c.player for c in st.pending_claims[st.actor_index:]]
            if len(tail) >= 2 and tail[0] == tail[1]:
                buckets["consecutive_same_owner"].append(st)
            if len(set(tail)) == 2 and any(tail[i] != tail[i + 1] for i in range(len(tail) - 1)):
                buckets["interleaved_ownership"].append(st)
            if st.actor_index == 0:
                buckets["round_start_promotion"].append(st)
        elif p == Phase.FINAL_PLACEMENT:
            buckets["final_placement"].append(st)
        if sum(st.discards) > 0:
            buckets["forced_discard"].append(st)
        if p == Phase.GAME_OVER:
            buckets["terminal"].append(st)

    required = ["init_0", "init_1", "init_2", "init_3",
               "actor_index_0", "actor_index_1", "actor_index_2", "actor_index_3",
               "next_without_pending", "consecutive_same_owner", "interleaved_ownership",
               "round_start_promotion", "final_placement", "forced_discard", "terminal"]
    for case in required:
        sample = buckets.get(case, [])
        assert sample, f"case not covered by fixtures: {case}"
        for st in sample[:25]:
            assert np.array_equal(se.encode_core(st, 0), se.encode_core(se.swap_players(st), 1)), \
                f"seat-swap failed for case {case}"


# ── losslessness ─────────────────────────────────────────────────────────────
def test_lossless_fingerprint_roundtrip():
    for st in _states(range(40)):
        for persp in (0, 1):
            assert se.decode(se.encode_core(st, persp)) == se.public_fingerprint(st, persp)


# ── completeness: every bank ─────────────────────────────────────────────────
def _rich_place():
    return _first(lambda s: s.phase == Phase.PLACE_AND_SELECT
                  and len(s.pending_claims[s.actor_index:]) >= 2
                  and s.next_claims and len(s.deck) >= 2)


def test_completeness_nonboard_mutations():
    st = _rich_place()
    base = se.encode_core(st, 0)

    def changed(mut):
        s = st.copy()
        mut(s)
        return not np.array_equal(se.encode_core(s, 0), base)

    # A different catalog id for replacement mutations. All 48 are conserved, so a
    # currently-PLACED id is the natural "not in row/deck/pending/next" choice (the
    # completeness gate only needs the encoding to change, not a legal state).
    placed = set()
    for b in st.boards:
        g = np.asarray(b.domino_id)
        placed |= {int(v) for v in np.unique(g[g > 0])}
    assert placed, "expected a mid-game state with placed dominoes"
    unused = min(placed)
    # id REPLACEMENT (not just removal), for row / bag / next
    assert changed(lambda s: setattr(s, "current_row", [unused] + list(s.current_row[1:])))
    assert changed(lambda s: setattr(s, "deck", [unused] + list(s.deck[1:])))
    assert changed(lambda s: setattr(s, "next_claims",
                                     [Claim(s.next_claims[0].player, unused)] + list(s.next_claims[1:])))
    # pending id and owner
    tail0 = st.pending_claims[st.actor_index]
    assert changed(lambda s: s.pending_claims.__setitem__(s.actor_index, Claim(tail0.player, unused)))
    assert changed(lambda s: s.pending_claims.__setitem__(s.actor_index, Claim(1 - tail0.player, tail0.domino_id)))
    # next owner
    assert changed(lambda s: setattr(s, "next_claims",
                                     [Claim(1 - s.next_claims[0].player, s.next_claims[0].domino_id)]
                                     + list(s.next_claims[1:])))
    # removals too
    assert changed(lambda s: setattr(s, "current_row", list(s.current_row[:-1])))
    assert changed(lambda s: setattr(s, "deck", list(s.deck[:-1])))
    # actor + turn slot (advancing actor_index shifts pending tail, actor bit, slot)
    assert changed(lambda s: setattr(s, "actor_index", s.actor_index + 1))
    # discard flag per role, both roles
    assert changed(lambda s: setattr(s, "discards", [s.discards[0] + 1, s.discards[1]]))
    assert changed(lambda s: setattr(s, "discards", [s.discards[0], s.discards[1] + 1]))
    # rules: harmony and middle kingdom independently
    assert changed(lambda s: setattr(s, "config", dataclasses.replace(s.config, harmony=False)))
    assert changed(lambda s: setattr(s, "config", dataclasses.replace(s.config, middle_kingdom=False)))


def test_completeness_board_position_halftype_and_role():
    """Board bank is sensitive to position, half-type, and which player owns a cell."""
    def placed_on(board_player, dom_id, which_placement=0):
        s = GameState.new(seed=0)
        b = s.boards[board_player]
        dom = DOMINOES[dom_id]
        b.place(dom, b.legal_placements(dom)[which_placement])
        return se.encode_core(s, 0)

    a = placed_on(0, 7, 0)
    b_pos = placed_on(0, 7, 1)             # same domino, different placement -> different cells
    b_half = placed_on(0, 12, 0)           # different domino (half-types) at first placement
    b_role = placed_on(1, 7, 0)            # same domino/pos on the OPPONENT board
    assert not np.array_equal(a, b_pos), "board bank ignores position"
    assert not np.array_equal(a, b_half), "board bank ignores half-type"
    assert not np.array_equal(a, b_role), "board bank ignores which player owns the cell"


def test_phase_and_terminal_bits():
    seen = set()
    for st in _states(range(30)):
        idx = set(se.encode_core(st, 0).tolist())
        assert (se.PHASE_OFF + int(st.phase)) in idx  # exactly the current phase bit
        for ph in range(4):
            if ph != int(st.phase):
                assert (se.PHASE_OFF + ph) not in idx
        seen.add(int(st.phase))
    assert seen == {0, 1, 2, 3}  # INITIAL, PLACE, FINAL, TERMINAL all exercised


# ── inventory (Counter; cross-bank duplication cannot hide) ───────────────────
def test_inventory_counter_and_conservation():
    for st in _states(range(20)):
        fp = se.decode(se.encode_core(st, 0))
        ids = Counter(fp.row)
        ids.update(i for _, i in fp.pending)
        ids.update(i for _, i in fp.next_claims)
        ids.update(fp.bag)
        assert all(c == 1 for c in ids.values()), "an unplaced id appears in two banks"
        placed = set()
        for b in st.boards:
            g = np.asarray(b.domino_id)
            placed |= {int(v) for v in np.unique(g[g > 0])}
        assert set(ids).isdisjoint(placed)
        assert len(ids) + len(placed) + sum(st.discards) == 48


# ── information-set: hidden deck ORDER must not leak ─────────────────────────
def test_hidden_deck_order_invariant():
    st = _rich_place()
    base = se.encode_core(st, 0)
    for k in range(5):
        s = st.copy()
        rng = random.Random(k)
        s.deck = list(s.deck)
        rng.shuffle(s.deck)
        assert np.array_equal(se.encode_core(s, 0), base), "deck ORDER leaked into features"


# ── boundary validation ──────────────────────────────────────────────────────
def test_rejects_bad_perspective_and_ids():
    st = _rich_place()
    for bad in (-1, 2, "x"):
        with pytest.raises(ValueError):
            se.encode_core(st, bad)
    with pytest.raises(ValueError):
        s = st.copy(); s.deck = list(s.deck) + [999]; se.encode_core(s, 0)
    with pytest.raises(ValueError):
        s = st.copy(); s.next_claims = [Claim(3, s.next_claims[0].domino_id)] + list(s.next_claims[1:])
        se.encode_core(s, 0)
