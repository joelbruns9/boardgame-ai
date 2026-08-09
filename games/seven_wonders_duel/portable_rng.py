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

    def normal(self) -> float:
        """Standard normal via the Marsaglia POLAR method.

        Polar rather than Box-Muller deliberately: Box-Muller needs ``cos``,
        which is not required to be correctly rounded and is the most likely
        place for a Rust/Python last-bit disagreement. Polar uses only
        multiplication, ``log`` and ``sqrt`` -- ``log`` is already proven across
        the boundary by ``gumbel`` (``rng.rs::gumbel_golden_matches_python``) and
        IEEE-754 requires ``sqrt`` to be correctly rounded.

        The second variate the method produces is DISCARDED rather than cached.
        Caching would put a spare-value flag in the RNG state that the Rust port
        would also have to mirror; throwing it away keeps the state a single u64
        and the stream trivially reproducible.
        """

        while True:
            u = 2.0 * self.next_float() - 1.0
            v = 2.0 * self.next_float() - 1.0
            s = u * u + v * v
            if s >= 1.0 or s == 0.0:
                continue
            return u * math.sqrt(-2.0 * math.log(s) / s)

    def gamma(self, alpha: float) -> float:
        """Gamma(``alpha``, 1) by Marsaglia-Tsang (2000).

        This is the primitive the module docstring says ``gammavariate`` could
        not provide portably, and it is what Dirichlet root noise is built from.

        ``alpha >= 1`` is the parity-tested path and the only one production
        uses -- 7WD's ~5.6 mean legal moves put the AlphaZero convention
        (alpha ~ 10 / branching) near 1.8. Below 1 the standard boost
        ``Gamma(a) = Gamma(a+1) * U**(1/a)`` applies, but it introduces ``pow``,
        which carries no correct-rounding guarantee, so treat sub-1 alphas as
        unverified across the boundary.
        """

        # `NaN <= 0.0` is False, so a bare positivity check admits NaN --
        # and with d = NaN every comparison in the rejection loop below is
        # false, so it never terminates. Infinity is as bad: c collapses to
        # 0, every draw returns inf, and inf/inf makes the whole Dirichlet
        # vector NaN, which silently pins selection to the first edge.
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("gamma requires a finite alpha > 0")
        if alpha < 1.0:
            boosted = self.gamma(alpha + 1.0)
            return boosted * (max(self.next_float(), _CLAMP) ** (1.0 / alpha))
        d = alpha - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = self.normal()
            v = 1.0 + c * x
            if v <= 0.0:
                continue
            v = v * v * v
            u = self.next_float()
            # Squeeze: pure arithmetic, so it short-circuits most draws without
            # touching a transcendental at all.
            if u < 1.0 - 0.0331 * (x * x) * (x * x):
                return d * v
            if math.log(max(u, _CLAMP)) < 0.5 * x * x + d * (
                1.0 - v + math.log(max(v, _CLAMP))
            ):
                return d * v

    def dirichlet(self, alpha: float, n: int) -> list[float]:
        """Symmetric Dirichlet over ``n`` categories, as normalised Gammas.

        Returns a uniform vector if every draw underflows to zero, which keeps
        the caller's ``(1-eps)*prior + eps*noise`` blend well-formed rather than
        dividing by zero on a pathological sample.
        """

        if n <= 0:
            raise ValueError("dirichlet requires n > 0")
        draws = [self.gamma(alpha) for _ in range(n)]
        total = sum(draws)
        if total <= 0.0:
            return [1.0 / n] * n
        return [draw / total for draw in draws]
