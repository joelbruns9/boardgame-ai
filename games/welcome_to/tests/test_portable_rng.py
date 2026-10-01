"""The engine's generator (``RUST_PORT_PLAN.md`` M0-B).

`portable_rng.py` says it is the same stream as
``seven_wonders_duel/portable_rng.py``, which was written for exactly this
reason.  This asserts that rather than trusting the transcription — the two
files are separate on purpose (each game's package stands alone, as Kingdomino's
and 7WD's do), and separate copies are how two implementations drift.

The Rust half of the parity is in ``test_rust_engine_equiv.py``, which skips
when the crate is not built; this half always runs.
"""

from __future__ import annotations

import pytest

from games.seven_wonders_duel.portable_rng import PortableRng as SevenWondersRng
from games.welcome_to.portable_rng import (
    SEARCH_SEED_DOMAIN,
    PortableRng,
    derive_search_seed,
)


@pytest.mark.parametrize("seed", [0, 1, 7, 12345, 2**64 - 1])
def test_the_stream_matches_the_seven_wonders_implementation(seed):
    ours, theirs = PortableRng(seed), SevenWondersRng(seed)
    assert [ours.next_u64() for _ in range(32)] == [
        theirs.next_u64() for _ in range(32)
    ]


def test_the_shuffle_matches_the_seven_wonders_implementation():
    """The deck shuffle *is* the deal, so one swap of difference is a different
    game from the same seed."""
    for seed in (0, 7, 123456789):
        for n in (2, 5, 81, 82):
            ours, theirs = list(range(n)), list(range(n))
            PortableRng(seed).shuffle(ours)
            SevenWondersRng(seed).shuffle(theirs)
            assert ours == theirs


def test_choice_is_one_draw_and_not_randoms_rejection_loop():
    """⚠ Deliberately **not** ``random.Random.choice``, which draws through
    ``_randbelow``.  This is one ``next_u64 % n``, so Rust reproduces it in a
    line — and the whole point is that the two agree, so the definition has to
    be the simple one."""
    rng, reference = PortableRng(99), PortableRng(99)
    seq = list("abcdefg")
    for _ in range(20):
        assert rng.choice(seq) == seq[reference.next_u64() % len(seq)]


def test_the_state_is_one_integer_and_a_clone_is_exact():
    """What makes ``GameState.copy`` exact and what the M1 gate compares."""
    rng = PortableRng(5)
    for _ in range(10):
        rng.next_u64()
    clone = rng.clone()
    assert clone.state == rng.state == rng.getstate()
    assert [clone.next_u64() for _ in range(8)] == [rng.next_u64() for _ in range(8)]

    restored = PortableRng(0)
    restored.setstate(clone.state)
    assert restored.next_u64() == clone.next_u64()


def test_an_empty_sequence_and_a_zero_range_are_refused():
    with pytest.raises(IndexError):
        PortableRng(1).choice([])
    with pytest.raises(ValueError):
        PortableRng(1).randrange(0)


def test_search_seeds_do_not_collide_over_a_contiguous_seed_block():
    """The regression the XOR derivation had, stated as the run that hits it.

    S2 hands out a contiguous block of game seeds and indexes searches by the
    learner's decision count, so ``game_seed ^ DOMAIN ^ index`` gave decision
    ``i`` of game ``s`` the tape of decision ``0`` of game ``s ^ i``: identical
    determinizations and, at equal root width, an identical Dirichlet vector.
    """
    seeds = {
        derive_search_seed(game_seed, index)
        for game_seed in range(1_000, 1_064)
        for index in range(64)
    }
    assert len(seeds) == 64 * 64

    # The exact pair the old derivation collided on.
    assert derive_search_seed(1_001, 0) != derive_search_seed(1_000, 1)


def test_a_search_seed_is_a_draw_of_the_stream_it_names():
    """O(1) state skipping, spelled out against the generator itself."""
    for index in range(8):
        rng = PortableRng(0xB3EC4 ^ SEARCH_SEED_DOMAIN)
        drawn = [rng.next_u64() for _ in range(index + 1)][-1]
        assert derive_search_seed(0xB3EC4, index) == drawn
