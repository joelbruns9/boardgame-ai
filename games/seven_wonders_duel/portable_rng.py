"""Portable SplitMix64 RNG — the reference random stream the Rust F3 searcher
mirrors bit-for-bit (PHASE_F.md F3.0).

Deliberately NOT ``random.Random``: the Mersenne Twister and ``gammavariate``
cannot be reproduced in Rust, so the searcher's Gumbel noise and chance sampling
draw from this instead. Every derivation here (state transition, uniform, Gumbel,
``randrange``, Fisher–Yates ``shuffle``) is defined so the Rust port produces an
identical stream from the same seed. The constants match Kingdomino's
``search.rs::splitmix64``.
"""

from __future__ import annotations

import math

_MASK64 = (1 << 64) - 1
_TWO53 = float(1 << 53)
_GAMMA = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_CLAMP = 1e-12  # guards log() against a zero argument; Rust applies the same


class PortableRng:
    """A reproducible SplitMix64 stream. Mutable state is a single u64."""

    __slots__ = ("_state",)

    def __init__(self, seed: int):
        self._state = seed & _MASK64

    def next_u64(self) -> int:
        self._state = (self._state + _GAMMA) & _MASK64
        z = self._state
        z = ((z ^ (z >> 30)) * _MIX1) & _MASK64
        z = ((z ^ (z >> 27)) * _MIX2) & _MASK64
        return z ^ (z >> 31)

    def next_float(self) -> float:
        """Uniform in [0, 1) from the top 53 bits (exactly representable)."""
        return (self.next_u64() >> 11) / _TWO53

    def gumbel(self) -> float:
        """A Gumbel(0, 1) key via ``-log(-log(1 - U))`` — the same pipeline as the
        old ``-log(gammavariate(1, 1))`` (``gammavariate(1, 1) == -log(1 - U)``),
        with an explicit clamp Rust mirrors."""
        gamma = -math.log(max(1.0 - self.next_float(), _CLAMP))
        return -math.log(max(gamma, _CLAMP))

    def randrange(self, n: int) -> int:
        """Integer in [0, n) via modulo (matches Rust ``splitmix64() % n``)."""
        if n <= 0:
            raise ValueError("randrange requires n > 0")
        return self.next_u64() % n

    def shuffle(self, seq: list) -> None:
        """In-place Fisher–Yates (Durstenfeld), high index to low, so the Rust
        port reproduces the permutation."""
        for i in range(len(seq) - 1, 0, -1):
            j = self.next_u64() % (i + 1)
            seq[i], seq[j] = seq[j], seq[i]

    def getrandbits(self, k: int) -> int:
        """Low ``k`` bits (k <= 64). Used to reseed the open-mode determinizer."""
        if not 0 < k <= 64:
            raise ValueError("getrandbits supports 1..64 bits")
        return self.next_u64() & ((1 << k) - 1)
