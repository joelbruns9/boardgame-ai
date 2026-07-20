"""Golden contract for the portable SplitMix64 RNG (PHASE_F.md F3.0).

These values are the reference stream the Rust F3 searcher must reproduce
bit-for-bit. `next_u64` for seed 0 begins with the canonical SplitMix64(0)
output 0xE220A8397B1DCDAF, so a correct Rust port (same constants) matches by
construction. Do not "update to match" a divergent Rust port — a mismatch means
the port is wrong.
"""

from games.seven_wonders_duel.portable_rng import PortableRng


def test_next_u64_golden():
    assert PortableRng(0).next_u64() == 0xE220A8397B1DCDAF  # canonical splitmix64(0)
    assert [PortableRng(0).next_u64() for _ in range(1)] == [16294208416658607535]
    r = PortableRng(0)
    assert [r.next_u64() for _ in range(5)] == [
        16294208416658607535,
        7960286522194355700,
        487617019471545679,
        17909611376780542444,
        1961750202426094747,
    ]
    r = PortableRng(12345)
    assert [r.next_u64() for _ in range(3)] == [
        2454886589211414944,
        3778200017661327597,
        2205171434679333405,
    ]


def test_next_float_golden():
    r = PortableRng(0)
    assert [r.next_float() for _ in range(3)] == [
        0.8833108082136426,
        0.43152799704850997,
        0.026433771592597743,
    ]
    r = PortableRng(0)
    assert all(0.0 <= r.next_float() < 1.0 for _ in range(1000))


def test_gumbel_golden():
    r = PortableRng(7)
    assert [r.gumbel() for _ in range(3)] == [
        0.7051848236225707,
        4.0786199258627525,
        -0.8373431815918142,
    ]


def test_randrange_and_shuffle_golden():
    r = PortableRng(0)
    assert [r.randrange(10) for _ in range(8)] == [5, 0, 9, 4, 7, 0, 3, 0]
    r = PortableRng(42)
    seq = list(range(8))
    r.shuffle(seq)
    assert seq == [3, 1, 6, 2, 4, 0, 7, 5]
    assert sorted(seq) == list(range(8))  # a permutation
