"""The static-table signature -- ``RUST_PORT_PLAN.md`` M0-D.

Both engines hash the rule tables they were built from and compare at load.  A
silent table divergence (one deck count off, one plan score transposed, a codec
offset shifted) produces a legal-looking game that is a *different game*, and it
would pass every per-action gate: the two engines would agree perfectly about a
world neither of them shares with BGA.

⚠ **The signature is over integers, not text.** Every table is flattened to a
stream of ``i64`` and folded with FNV-1a; nothing depends on formatting,
encoding or float repr, all three of which are places Rust and Python may
legitimately differ.  Strings appearing in plan parameters are mapped through
:data:`_PARAM_CODES`, so an unrecognised one raises rather than hashing to
whatever ``str`` happens to produce.

Change a table on purpose and this number changes; that is the point.  Update
the expected value in ``tests/test_tables.py``, and rebuild the Rust crate --
the mismatch is a build reminder, not a bug.
"""

from __future__ import annotations

from typing import Iterator

from games.welcome_to import action_codec as codec
from games.welcome_to.constants import (
    BIS_BOXES,
    BIS_SCORES,
    BOX_OFFSET,
    CARD_TABLE,
    ESTATE_ROW_BOXES,
    ESTATE_ROW_SCORES,
    EXTREMITY_POSITIONS,
    FENCE_OFFSET,
    FENCE_SIZES,
    MAX_ESTATE_SIZE,
    MAX_NUMBER,
    MIN_NUMBER,
    NUM_BOXES,
    NUM_FENCES,
    NUM_STREETS,
    PARK_BOXES,
    PARK_SCORES,
    PERMIT_BOXES,
    PERMIT_SCORES,
    POOL_BOXES,
    POOL_POSITIONS,
    POOL_SCORES,
    ROUNDABOUT,
    ROUNDABOUT_BOXES,
    ROUNDABOUT_SCORES,
    SOLO_CARD_ID,
    SOLO_DECK_MIDDLE,
    STREET_SIZES,
    TEMP_BOXES,
    TEMP_DELTAS,
    TEMP_RANK_SCORES,
    TEMP_SOLO_SCORE,
    TEMP_SOLO_THRESHOLD,
)
from games.welcome_to.plans import PLANS

#: Bump when the *stream layout* changes (not when a table's values change).
SIGNATURE_VERSION: int = 1

#: Plan parameters that are strings.  Fixed codes, so the stream stays integral.
_PARAM_CODES: dict[str, int] = {
    "park": 1,
    "pool": 2,
    "pool&park": 3,
    "iceCream": 4,
    "without": 5,
    "with": 6,
    "christmas": 7,
    "easterEgg": 8,
}

#: The codec sections, in layout order.  Named explicitly rather than iterating
#: ``OFFSET``, so that a section renamed or reordered is a signature change.
_CODEC_SECTIONS: tuple[str, ...] = (
    "CHOOSE_STACK",
    "PERMIT_REFUSAL",
    "ROUNDABOUT_OPEN",
    "ROUNDABOUT_POS",
    "WRITE",
    "SURVEYOR_FENCE",
    "ESTATE_ROW",
    "PARK_STREET",
    "POOL_BUILD",
    "BIS",
    "CHOOSE_PLAN",
    "VALIDATE_ESTATE",
    "PASS_ROUNDABOUT",
    "PASS_SURVEYOR",
    "PASS_ESTATE",
    "PASS_PARK",
    "PASS_POOL",
    "PASS_BIS",
    "PASS_PLAN",
    "RESHUFFLE_YES",
    "RESHUFFLE_NO",
)

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = (1 << 64) - 1


def _param_value(param) -> int:
    if isinstance(param, bool):  # bool is an int; nothing uses one, so refuse
        raise TypeError("plan parameters are ints or known strings")
    if isinstance(param, int):
        return param
    try:
        return _PARAM_CODES[param]
    except KeyError:
        raise ValueError(
            f"plan parameter {param!r} has no signature code; add it to "
            "_PARAM_CODES in both engines"
        ) from None


def signature_stream() -> Iterator[int]:
    """Every static table, flattened, in a fixed order.  Mirrored in Rust."""
    yield SIGNATURE_VERSION

    # -- cards ------------------------------------------------------------
    yield len(CARD_TABLE)
    for number, effect in CARD_TABLE:
        yield -1 if number is None else number
        yield int(effect)
    yield SOLO_CARD_ID
    yield SOLO_DECK_MIDDLE

    # -- geometry ---------------------------------------------------------
    yield NUM_STREETS
    yield from STREET_SIZES
    yield from FENCE_SIZES
    yield from BOX_OFFSET
    yield from FENCE_OFFSET
    yield NUM_BOXES
    yield NUM_FENCES
    yield MIN_NUMBER
    yield MAX_NUMBER
    yield ROUNDABOUT
    yield from TEMP_DELTAS

    # -- score tracks -----------------------------------------------------
    yield from PARK_BOXES
    for row in PARK_SCORES:
        yield len(row)
        yield from row
    yield POOL_BOXES
    yield from POOL_SCORES
    for x, y in POOL_POSITIONS:
        yield x
        yield y
    yield BIS_BOXES
    yield from BIS_SCORES
    yield TEMP_BOXES
    yield from TEMP_RANK_SCORES
    yield TEMP_SOLO_THRESHOLD
    yield TEMP_SOLO_SCORE
    yield MAX_ESTATE_SIZE
    yield from ESTATE_ROW_BOXES
    for row in ESTATE_ROW_SCORES:
        yield len(row)
        yield from row
    yield PERMIT_BOXES
    yield from PERMIT_SCORES
    yield ROUNDABOUT_BOXES
    yield from ROUNDABOUT_SCORES
    for x, y in EXTREMITY_POSITIONS:
        yield x
        yield y

    # -- plans ------------------------------------------------------------
    yield len(PLANS)
    for plan in PLANS:
        yield plan.id
        yield int(plan.variant)
        yield plan.stack
        yield plan.scores[0]
        yield plan.scores[1]
        yield int(plan.kind)
        yield len(plan.params)
        for param in plan.params:
            yield _param_value(param)

    # -- codec layout -----------------------------------------------------
    yield codec.NUM_ACTIONS
    for name in _CODEC_SECTIONS:
        yield codec.OFFSET[name]
    for i, j in codec.EXPERT_PAIRS:
        yield i
        yield j


def table_signature() -> int:
    """FNV-1a 64 over :func:`signature_stream`, each value as 8 little-endian
    bytes.  ``welcome_to_rust::tables::table_signature`` computes the same.

    FNV rather than SHA-256 deliberately: the crate's dependency list starts at
    pyo3 only (``RUST_PORT_PLAN.md`` §6), and this hash guards against a
    transcription slip, not an adversary."""
    h = _FNV_OFFSET
    for value in signature_stream():
        word = value & _MASK64
        for shift in range(0, 64, 8):
            h ^= (word >> shift) & 0xFF
            h = (h * _FNV_PRIME) & _MASK64
    return h


if __name__ == "__main__":  # pragma: no cover - a convenience for the Rust side
    print(f"0x{table_signature():016x}")
