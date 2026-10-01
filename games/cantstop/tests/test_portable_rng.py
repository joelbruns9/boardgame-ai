"""M0 gate: the portable RNG is identical in Python and Rust.

Without this every later equivalence test is meaningless -- two engines fed
"the same seed" would be playing different games.

The Rust tests are skipped rather than failed when the extension is not built,
so the suite still runs on a machine without a Rust toolchain. Build it with:

    cd games/cantstop/cantstop_rust && maturin develop --release

Run: python -m pytest games/cantstop/tests/test_portable_rng.py -q
"""

import pytest

from games.cantstop.engine import random_dice
from games.cantstop.portable_rng import PortableRng
from games.seven_wonders_duel.portable_rng import (
    PortableRng as SevenWondersRng,
)

rust = pytest.importorskip("cantstop_rust", reason="run maturin develop first")

SEEDS = (0, 1, 2, 12345, 2**63, 2**64 - 1, 0xDEADBEEFCAFEF00D)


# ---- the Python stream is the shared one ----

def test_matches_the_seven_wonders_stream_bit_for_bit():
    """Transcribed constants get transcribed wrong; check, do not trust."""
    for seed in SEEDS:
        ours, theirs = PortableRng(seed), SevenWondersRng(seed)
        for _ in range(200):
            assert ours.next_u64() == theirs.next_u64(), seed
        # 7WD's copy keeps the state private; compare through the attribute
        # it actually exposes rather than assuming a shared API.
        assert ours.state == theirs._state


def test_state_round_trips():
    a = PortableRng(99)
    for _ in range(10):
        a.next_u64()
    b = PortableRng(0)
    b.state = a.state
    assert [b.next_u64() for _ in range(5)] == [a.next_u64() for _ in range(5)]


def test_clone_is_independent():
    a = PortableRng(5)
    a.next_u64()
    b = a.clone()
    assert a.next_u64() == b.next_u64()
    a.next_u64()
    assert a.state != b.state


def test_randint_covers_its_range_and_no_more():
    rng = PortableRng(3)
    seen = {rng.randint(1, 6) for _ in range(500)}
    assert seen == {1, 2, 3, 4, 5, 6}


def test_randrange_rejects_a_useless_bound():
    with pytest.raises(ValueError, match="randrange requires"):
        PortableRng(0).randrange(0)


def test_randint_rejects_an_inverted_range():
    with pytest.raises(ValueError, match="randint requires"):
        PortableRng(0).randint(6, 1)


def test_it_is_a_drop_in_for_random_dice():
    """engine.random_dice must accept either generator unchanged."""
    dice = random_dice(PortableRng(7))
    assert len(dice) == 4
    assert all(1 <= d <= 6 for d in dice)


# ---- Python and Rust agree ----

def test_rust_next_u64_matches():
    for seed in SEEDS:
        py, rs = PortableRng(seed), rust.Rng(seed)
        for step in range(500):
            assert py.next_u64() == rs.next_u64(), (seed, step)
            assert py.state == rs.state, (seed, step)


def test_rust_next_float_matches_exactly():
    """f64 division, so this is bit-equality, not approximate."""
    for seed in SEEDS[:4]:
        py, rs = PortableRng(seed), rust.Rng(seed)
        for _ in range(200):
            assert py.next_float() == rs.next_float()


def test_rust_randrange_and_randint_match():
    for seed in SEEDS[:4]:
        py, rs = PortableRng(seed), rust.Rng(seed)
        for n in (1, 2, 6, 7, 11, 126):
            assert py.randrange(n) == rs.randrange(n), (seed, n)
        for _ in range(100):
            assert py.randint(1, 6) == rs.randint(1, 6)


def test_rust_dice_match_engine_random_dice():
    """The draw order and count must agree, not just the distribution: four
    randint(1, 6) calls, unsorted, in order."""
    for seed in SEEDS:
        py, rs = PortableRng(seed), rust.Rng(seed)
        for _ in range(100):
            expected = random_dice(py)
            # Vec<u8> arrives as bytes, not list. Comparing a tuple of ints
            # against bytes is ALWAYS unequal even element-wise identical, so
            # unpack before comparing -- this is the trap the Kingdomino port
            # recorded.
            got = tuple(rs.roll_dice())
            assert expected == got, seed
            assert py.state == rs.state


def test_the_bytes_trap_is_real():
    """Pin the reason the test above unpacks: a bare comparison silently
    passes as 'not equal' forever, so a broken port would look broken in a
    way that is easy to misread, or a correct one would look wrong."""
    rs = rust.Rng(1)
    raw = rs.roll_dice()
    assert isinstance(raw, bytes)
    assert list(raw) != raw          # the trap itself
    assert tuple(raw) == tuple(list(raw))


def test_rust_shuffle_matches():
    for seed in SEEDS[:4]:
        py, rs = PortableRng(seed), rust.Rng(seed)
        seq = list(range(32))
        py.shuffle(seq)
        assert seq == list(rs.shuffle(list(range(32))))
        assert py.state == rs.state
