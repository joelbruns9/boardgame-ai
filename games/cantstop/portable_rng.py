"""Portable SplitMix64 — the dice stream, mirrored bit-for-bit in Rust.

Deliberately NOT ``random.Random``. The Mersenne Twister cannot be reproduced
in Rust, so "same seed" would give two different games and the M1 equivalence
gate would have nothing to compare. ``random.Random.randint`` is doubly
unusable: it draws through ``_randbelow``'s rejection loop, whose number of
draws depends on rejected values.

This is the same stream as ``seven_wonders_duel/portable_rng.py``;
``tests/test_portable_rng.py`` asserts the two agree bit-for-bit rather than
trusting the transcription. Only what Can't Stop needs is here -- the engine
never draws randomness itself, so this serves the *drivers* (self-play, arena,
tests) and nothing else.

``PortableRng`` deliberately offers ``randint`` with ``random.Random``'s
signature so ``engine.random_dice`` accepts either generator unchanged. The
two produce different streams from the same seed; only this one is portable.
"""

_MASK64 = (1 << 64) - 1
_TWO53 = float(1 << 53)
_GAMMA = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB


class PortableRng:
    """A reproducible SplitMix64 stream. Mutable state is a single u64."""

    __slots__ = ("_state",)

    def __init__(self, seed):
        self._state = seed & _MASK64

    @property
    def state(self):
        """The whole of the state, which is what makes ``clone`` exact and
        lets the equivalence gate compare the two engines' generators
        directly -- one integer here against 625 words for the Twister."""
        return self._state

    @state.setter
    def state(self, value):
        self._state = value & _MASK64

    def clone(self):
        return PortableRng(self._state)

    def __repr__(self):                      # pragma: no cover - debug aid
        return f"PortableRng(0x{self._state:016x})"

    # ---- primitives ----

    def next_u64(self):
        self._state = (self._state + _GAMMA) & _MASK64
        z = self._state
        z = ((z ^ (z >> 30)) * _MIX1) & _MASK64
        z = ((z ^ (z >> 27)) * _MIX2) & _MASK64
        return z ^ (z >> 31)

    def next_float(self):
        """Uniform in [0, 1) from the top 53 bits."""
        return (self.next_u64() >> 11) / _TWO53

    def randrange(self, n):
        """Integer in [0, n) by plain modulo.

        Modulo, not rejection sampling: the tiny bias is irrelevant for dice
        and a rejection loop would make the *number of draws* depend on the
        values drawn, which is precisely what makes ``random.Random``
        unportable.
        """
        if n <= 0:
            raise ValueError("randrange requires n > 0")
        return self.next_u64() % n

    def randint(self, a, b):
        """Inclusive [a, b], matching ``random.Random.randint``'s signature
        (but not its stream)."""
        if b < a:
            raise ValueError("randint requires a <= b")
        return a + self.randrange(b - a + 1)

    def choice(self, seq):
        if not seq:
            raise IndexError("cannot choose from an empty sequence")
        return seq[self.randrange(len(seq))]

    def random(self):
        """``random.Random``'s spelling of :meth:`next_float`, so drivers that
        hold either generator do not have to know which one they have."""
        return self.next_float()

    def shuffle(self, seq):
        """In-place Fisher-Yates, high index to low."""
        for i in range(len(seq) - 1, 0, -1):
            j = self.randrange(i + 1)
            seq[i], seq[j] = seq[j], seq[i]
