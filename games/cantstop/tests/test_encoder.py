"""Correctness tests for the value-net feature encoder.

Run: python -m pytest games/cantstop/tests/test_encoder.py -q
"""

import random

import numpy as np
import pytest

from games.cantstop.engine import (
    ALL_RULESETS, COLUMNS, GameState, RuleSet, apply_move, can_stop,
    legal_moves, random_dice, roll, stop,
)
from games.cantstop.encoder import (
    FEATURE_SIZE, MAX_SEATS, NUM_COLUMNS, PER_SEAT, encode_batch,
    encode_board, seat_mask, seat_order, seat_to_slot, to_absolute,
)


def fresh(num_players=3, extended=True, blocking=True):
    return GameState(RuleSet.make(num_players, extended, blocking))


def seat_block(vec, slot):
    return vec[slot * PER_SEAT:(slot + 1) * PER_SEAT]


def globals_of(vec):
    return vec[MAX_SEATS * PER_SEAT:]


def seat_order_for(active, n):
    return [(active + k) % n for k in range(n)]


def play_to_random_end_of_turn(rng, rules, max_turns=200):
    """Play random legal moves and return a state at some end of turn."""
    s = GameState(rules)
    targets = []
    for _ in range(max_turns):
        if s.game_over:
            break
        # roll() busts and passes the turn itself when nothing is legal.
        moves = roll(s, random_dice(rng))
        if not moves:
            targets.append(s.clone())
            continue
        apply_move(s, rng.choice(moves))
        if can_stop(s) and rng.random() < 0.35:
            stop(s)
            if not s.game_over:
                targets.append(s.clone())
    return rng.choice(targets) if targets else s


# ---- shape and contract ----

def test_feature_size_matches_layout():
    assert PER_SEAT == 2 * NUM_COLUMNS + 2
    assert FEATURE_SIZE == MAX_SEATS * PER_SEAT + 2


def test_encode_returns_float32_of_declared_size():
    v = encode_board(fresh())
    assert v.shape == (FEATURE_SIZE,)
    assert v.dtype == np.float32


def test_mid_turn_board_is_rejected():
    """The net only ever sees end-of-turn boards; catch misuse loudly."""
    s = fresh()
    roll(s, (1, 1, 2, 2))
    apply_move(s, legal_moves(s, s.dice)[0])
    assert s.runners
    with pytest.raises(ValueError, match="end-of-turn"):
        encode_board(s)


def test_encode_batch_matches_row_by_row():
    rng = random.Random(7)
    states = [play_to_random_end_of_turn(rng, r) for r in ALL_RULESETS]
    batch = encode_batch(states)
    assert batch.shape == (len(states), FEATURE_SIZE)
    for i, s in enumerate(states):
        assert np.array_equal(batch[i], encode_board(s))


def test_out_parameter_clears_stale_values():
    s = fresh()
    scratch = np.full(FEATURE_SIZE, 9.0, dtype=np.float32)
    encode_board(s, out=scratch)
    assert np.array_equal(scratch, encode_board(s))


# ---- what the features mean ----

def test_progress_is_a_fraction_of_column_height():
    s = fresh(num_players=2, extended=False, blocking=False)
    s.progress[0][7] = 5          # column 7 is 13 tall
    s.progress[0][2] = 3          # column 2 is 3 tall: full height, unclaimed
    v = encode_board(s)
    me = seat_block(v, 0)
    assert me[COLUMNS.index(7)] == pytest.approx(5 / 13, abs=1e-6)
    assert me[COLUMNS.index(2)] == pytest.approx(1.0, abs=1e-6)


def test_claimed_column_sets_its_flag():
    s = fresh(num_players=2)
    s.claimed_by[7] = 0
    v = encode_board(s)
    assert seat_block(v, 0)[NUM_COLUMNS + COLUMNS.index(7)] == 1.0
    assert seat_block(v, 1)[NUM_COLUMNS + COLUMNS.index(7)] == 0.0


def test_columns_needed_counts_down_as_columns_are_claimed():
    s = fresh(num_players=3, extended=True)   # needs 4
    idx = 2 * NUM_COLUMNS
    assert seat_block(encode_board(s), 0)[idx] == pytest.approx(4 / 5)
    s.claimed_by[7] = s.active_player
    s.claimed_by[6] = s.active_player
    assert seat_block(encode_board(s), 0)[idx] == pytest.approx(2 / 5)


def test_columns_needed_bridges_variants():
    """The point of the feature: 'two from winning' reads the same in a
    3-column and a 5-column rule set, even though raw claim counts differ."""
    base = GameState(RuleSet.make(2, extended=False))   # needs 3
    ext = GameState(RuleSet.make(2, extended=True))     # needs 5
    base.claimed_by[7] = 0
    for col in (7, 6, 8):
        ext.claimed_by[col] = 0
    idx = 2 * NUM_COLUMNS
    assert (seat_block(encode_board(base), 0)[idx]
            == pytest.approx(seat_block(encode_board(ext), 0)[idx]))


def test_absent_seats_are_entirely_zero():
    for n in (2, 3, 4):
        v = encode_board(fresh(num_players=n, extended=False))
        for slot in range(n):
            assert seat_block(v, slot)[2 * NUM_COLUMNS + 1] == 1.0
        for slot in range(n, MAX_SEATS):
            assert not seat_block(v, slot).any()


def test_blocking_flag_is_the_only_difference_between_paired_rulesets():
    on = encode_board(fresh(num_players=3, extended=True, blocking=True))
    off = encode_board(fresh(num_players=3, extended=True, blocking=False))
    diff = np.flatnonzero(on != off)
    assert diff.tolist() == [MAX_SEATS * PER_SEAT]


def test_columns_to_win_global_distinguishes_base_from_extended():
    base = globals_of(encode_board(fresh(2, extended=False)))
    ext = globals_of(encode_board(fresh(2, extended=True)))
    assert base[1] != ext[1]


# ---- seat-relative framing ----

def test_seat_order_starts_at_the_player_to_move():
    s = fresh(num_players=3)
    s.active_player = 2
    assert seat_order(s) == [2, 0, 1]


def test_seat_to_slot_inverts_seat_order():
    """Training labels go through this: a winner's absolute seat becomes the
    encoding slot the net is asked to predict. Getting it wrong mislabels
    every row in a way self-consistent code cannot notice."""
    for n in (2, 3, 4):
        for active in range(n):
            s = fresh(num_players=n, extended=False)
            s.active_player = active
            order = seat_order(s)
            for seat in range(n):
                assert order[seat_to_slot(s, seat)] == seat
            assert seat_to_slot(s, active) == 0


def test_seat_mask_reads_the_present_flag_not_a_neighbour():
    """A live seat one column from winning still has columns-needed > 0, so a
    mask read off the wrong index usually looks right. Pin the index with a
    seat that needs nothing more: it is still present."""
    s = fresh(num_players=2, extended=False)     # needs 3
    for col in (2, 3, 4):
        s.claimed_by[col] = s.active_player
    mask = seat_mask(encode_board(s))
    assert mask.tolist() == [[True, True, False, False]]


def test_encoding_is_seat_relative():
    """The same board seen by different seats must place each seat's own
    features in slot 0. This is what lets one net play every seat."""
    s = fresh(num_players=3, extended=True, blocking=False)
    s.progress[0][7] = 4
    s.progress[1][6] = 2
    s.progress[2][8] = 6

    blocks = {}
    for seat in range(3):
        s.active_player = seat
        blocks[seat] = [seat_block(encode_board(s), slot) for slot in range(3)]

    for seat in range(3):
        for slot, other in enumerate(seat_order_for(seat, 3)):
            assert np.array_equal(blocks[seat][slot], blocks[other][0])


def test_two_boards_differing_only_in_seat_identity_encode_alike():
    """Seat-relative framing is a deliberate collapse: seat 0 leading while
    seat 0 moves is the same position as seat 1 leading while seat 1 moves."""
    a = fresh(num_players=2, extended=False, blocking=False)
    a.progress[0][7] = 5
    b = fresh(num_players=2, extended=False, blocking=False)
    b.progress[1][7] = 5
    b.active_player = 1
    assert np.array_equal(encode_board(a), encode_board(b))


# ---- to_absolute ----

def test_to_absolute_undoes_the_rotation():
    s = fresh(num_players=3)
    s.active_player = 2
    out = to_absolute([0.5, 0.3, 0.2, 0.0], s)
    assert out.shape == (3,)
    assert out[2] == pytest.approx(0.5)   # slot 0 is the seat to move
    assert out[0] == pytest.approx(0.3)
    assert out[1] == pytest.approx(0.2)


def test_to_absolute_renormalizes_after_dropping_absent_seats():
    s = fresh(num_players=2, extended=False)
    out = to_absolute([0.4, 0.2, 0.3, 0.1], s)   # 0.4 of the mass is absent
    assert out.sum() == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.4 / 0.6)


def test_to_absolute_handles_a_batch():
    s = fresh(num_players=3)
    s.active_player = 1
    rel = np.array([[0.5, 0.3, 0.2, 0.0], [0.1, 0.1, 0.8, 0.0]])
    out = to_absolute(rel, s)
    assert out.shape == (2, 3)
    for i in range(2):
        assert np.array_equal(out[i], to_absolute(rel[i], s))


def test_to_absolute_rejects_the_wrong_width():
    with pytest.raises(ValueError, match="seat outputs"):
        to_absolute([0.5, 0.5], fresh(num_players=2))


def test_to_absolute_rejects_empty_mass_on_live_seats():
    s = fresh(num_players=2, extended=False)
    with pytest.raises(ValueError, match="probability mass"):
        to_absolute([0.0, 0.0, 0.5, 0.5], s)


# ---- whole-corpus properties ----

def test_every_ruleset_encodes_finite_values_in_range():
    rng = random.Random(11)
    for rules in ALL_RULESETS:
        for _ in range(20):
            v = encode_board(play_to_random_end_of_turn(rng, rules))
            assert np.all(np.isfinite(v)), rules
            assert v.min() >= 0.0 and v.max() <= 1.0, rules


def canonical_key(state):
    """Everything an end-of-turn board carries, in seat-relative order."""
    order = seat_order(state)
    return (state.rules,
            tuple((tuple(state.progress[p][c] for c in COLUMNS),
                   tuple(state.claimed_by[c] == p for c in COLUMNS))
                  for p in order))


def test_distinct_boards_get_distinct_encodings():
    """Guards against silent information loss: if two genuinely different
    positions collide, the net cannot tell them apart no matter how it
    trains. Collected over every rule set."""
    rng = random.Random(19)
    seen = {}
    collisions = []
    for rules in ALL_RULESETS:
        for _ in range(120):
            s = play_to_random_end_of_turn(rng, rules)
            key = canonical_key(s)
            vec = encode_board(s).tobytes()
            if vec in seen and seen[vec] != key:
                collisions.append((seen[vec], key))
            seen[vec] = key
    assert not collisions, (
        f"{len(collisions)} colliding boards, e.g. {collisions[0]}")
    assert len(seen) > 300, f"corpus too small to be meaningful: {len(seen)}"
