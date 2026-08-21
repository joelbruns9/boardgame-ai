"""
Static rule data for *Welcome To...* (Blue Cocker Games), transcribed from the
Board Game Arena implementation under ``BGA Files/welcometo``.

Every table in this module is a direct transcription of a PHP source of truth;
the docstring on each table names the file it came from so the mapping can be
re-checked whenever the BGA code is updated.

SCOPE
─────
Base game, with the *advanced* variant (extra City Plans + roundabouts) and the
*expert* variant (draft a number card and an effect card, pass the third on)
available as configuration switches.  The three seasonal boards (Ice Cream,
Christmas, Easter) are deliberately **not** implemented: their plan/effect
tables are kept in ``plans.py`` for id fidelity but are never dealt.

GEOMETRY
────────
A player sheet is three streets of 10 / 11 / 12 house boxes.  Two flat index
spaces are used throughout the engine and the encoder:

    box index    0..32   house boxes,   ``BOX_OFFSET[x] + y``
    fence index  0..29   fence slots,   ``FENCE_OFFSET[x] + j``

A fence slot ``(x, j)`` sits between house boxes ``(x, j)`` and ``(x, j + 1)``,
so street ``x`` has ``STREET_SIZES[x] - 1`` of them.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Final


# ──────────────────────────────────────────────────────────────────────────
# Construction card effects  (modules/php/constants.inc.php)
# ──────────────────────────────────────────────────────────────────────────
class Effect(IntEnum):
    SURVEYOR = 1
    ESTATE = 2
    PARK = 3
    POOL = 4
    TEMP = 5
    BIS = 6
    SOLO = 7  # solo-mode marker card; never appears as a playable effect


#: Column order of :data:`DECK_COUNTS`, copied from
#: ``ConstructionCards::$actions``.
DECK_EFFECT_ORDER: Final = (
    Effect.SURVEYOR,
    Effect.POOL,
    Effect.TEMP,
    Effect.BIS,
    Effect.PARK,
    Effect.ESTATE,
)

#: ``ConstructionCards::$deck`` — how many copies of each (number, effect) pair
#: exist.  Sums to 81 cards.
DECK_COUNTS: Final = {
    #     SURVEYOR, POOL, TEMP, BIS, PARK, ESTATE
    1:  (1, 0, 0, 0, 1, 1),
    2:  (1, 0, 0, 0, 1, 1),
    3:  (1, 1, 1, 1, 0, 0),
    4:  (0, 1, 1, 1, 1, 1),
    5:  (2, 0, 0, 0, 2, 2),
    6:  (2, 1, 1, 1, 1, 1),
    7:  (1, 1, 1, 1, 2, 2),
    8:  (2, 1, 1, 1, 2, 2),
    9:  (1, 1, 1, 1, 2, 2),
    10: (2, 1, 1, 1, 1, 1),
    11: (2, 0, 0, 0, 2, 2),
    12: (0, 1, 1, 1, 1, 1),
    13: (1, 1, 1, 1, 0, 0),
    14: (1, 0, 0, 0, 1, 1),
    15: (1, 0, 0, 0, 1, 1),
}


#: The fifteen printed house numbers, and lookups into the (15, 6) deck matrix
#: used by :mod:`games.welcome_to.deck_knowledge`.
CARD_NUMBERS: Final = tuple(sorted(DECK_COUNTS))
NUMBER_INDEX: Final = {n: i for i, n in enumerate(CARD_NUMBERS)}
EFFECT_INDEX: Final = {e: i for i, e in enumerate(DECK_EFFECT_ORDER)}


def _build_card_table() -> tuple[tuple[int | None, Effect], ...]:
    cards: list[tuple[int | None, Effect]] = []
    for number in sorted(DECK_COUNTS):
        counts = DECK_COUNTS[number]
        for effect, n in zip(DECK_EFFECT_ORDER, counts):
            cards.extend([(number, effect)] * n)
    return tuple(cards)


#: Every construction card as ``(number, effect)``.  A "card id" anywhere in the
#: engine is an index into this tuple.  Index :data:`SOLO_CARD_ID` is the extra
#: solo-mode card and is only shuffled in when ``config.players == 1``.
CARD_TABLE: Final = _build_card_table() + ((None, Effect.SOLO),)
SOLO_CARD_ID: Final = len(CARD_TABLE) - 1
NUM_BASE_CARDS: Final = SOLO_CARD_ID  # 81

#: ``ConstructionCards::soloSetupNewGame`` keeps the solo card out of the top
#: half of the deck by shifting it down by this many positions.
SOLO_DECK_MIDDLE: Final = 42


# ──────────────────────────────────────────────────────────────────────────
# Sheet geometry  (Houses::$streetSizes and the Zone subclasses)
# ──────────────────────────────────────────────────────────────────────────
STREET_SIZES: Final = (10, 11, 12)
NUM_STREETS: Final = 3
MAX_STREET_LEN: Final = 12
BOX_OFFSET: Final = (0, 10, 21)
NUM_BOXES: Final = 33

FENCE_SIZES: Final = (9, 10, 11)  # ``Surveyor::$cols``
FENCE_OFFSET: Final = (0, 9, 19)
NUM_FENCES: Final = 30

#: House numbers.  1..15 are printed on the cards; the temp agency can push a
#: written number down to 0 or up to 17 (``Player::getAvailableNumbersOfCombination``).
MIN_NUMBER: Final = 0
MAX_NUMBER: Final = 17
NUM_NUMBER_VALUES: Final = MAX_NUMBER - MIN_NUMBER + 1  # 18

#: Sentinel written into a house box for a roundabout (``ROUNDABOUT`` in
#: ``constants.inc.php``).  Roundabouts break the ascending-order chain and
#: never belong to a housing estate.
ROUNDABOUT: Final = 100

#: Temp agency modifiers, in the order used by the action codec.
TEMP_DELTAS: Final = (0, -2, -1, 1, 2)


def box_index(x: int, y: int) -> int:
    return BOX_OFFSET[x] + y


def box_coords(index: int) -> tuple[int, int]:
    for x in range(NUM_STREETS - 1, -1, -1):
        if index >= BOX_OFFSET[x]:
            return x, index - BOX_OFFSET[x]
    raise ValueError(f"bad box index {index}")


def fence_index(x: int, j: int) -> int:
    return FENCE_OFFSET[x] + j


def fence_coords(index: int) -> tuple[int, int]:
    for x in range(NUM_STREETS - 1, -1, -1):
        if index >= FENCE_OFFSET[x]:
            return x, index - FENCE_OFFSET[x]
    raise ValueError(f"bad fence index {index}")


# ──────────────────────────────────────────────────────────────────────────
# Score tracks
# ──────────────────────────────────────────────────────────────────────────

#: ``Actions/Park.php`` — parks per street and the cumulative value of having
#: built ``n`` of them.  ``PARK_SCORES[x][n]``.
PARK_BOXES: Final = (3, 4, 5)
PARK_SCORES: Final = (
    (0, 2, 4, 10),
    (0, 2, 4, 6, 14),
    (0, 2, 4, 6, 8, 18),
)

#: ``Actions/Pool.php`` — nine pool boxes; value of having built ``n`` pools.
POOL_BOXES: Final = 9
POOL_SCORES: Final = (0, 3, 6, 9, 13, 17, 21, 26, 31, 36)

#: The nine printed pool locations, as ``(street, box)``.
POOL_POSITIONS: Final = (
    (0, 2), (0, 6), (0, 7),
    (1, 0), (1, 3), (1, 7),
    (2, 1), (2, 6), (2, 10),
)
POOL_POSITION_SET: Final = frozenset(POOL_POSITIONS)

#: ``Actions/Bis.php`` — nine bis boxes; the value is a PENALTY (subtracted).
BIS_BOXES: Final = 9
BIS_SCORES: Final = (0, 1, 3, 6, 9, 12, 16, 20, 24, 28)

#: ``Actions/Temp.php`` — eleven temp boxes.  In a multiplayer game the players
#: are ranked by how many they crossed off: most = 7, second = 4, third = 1.
#: In solo, six or more is worth a flat 7.
TEMP_BOXES: Final = 11
TEMP_RANK_SCORES: Final = (7, 4, 1)
TEMP_SOLO_THRESHOLD: Final = 6
TEMP_SOLO_SCORE: Final = 7

#: ``Actions/RealEstate.php`` — one row per estate size 1..6.  Crossing boxes in
#: a row raises the per-estate value of that size.
ESTATE_ROW_BOXES: Final = (1, 2, 3, 4, 4, 4)
ESTATE_ROW_SCORES: Final = (
    (1, 3),
    (2, 3, 4),
    (3, 4, 5, 6),
    (4, 5, 6, 7, 8),
    (5, 6, 7, 8, 10),
    (6, 7, 8, 10, 12),
)
MAX_ESTATE_SIZE: Final = 6

#: ``Actions/PermitRefusal.php`` — three boxes; PENALTY.  The third one also
#: ends the game.
PERMIT_BOXES: Final = 3
PERMIT_SCORES: Final = (0, 0, 3, 5)

#: ``Actions/Roundabout.php`` — two boxes (advanced variant only); PENALTY.
ROUNDABOUT_BOXES: Final = 2
ROUNDABOUT_SCORES: Final = (0, 3, 8)

#: The six boxes checked by the "Extremities" advanced plan.
EXTREMITY_POSITIONS: Final = (
    (0, 0), (0, 9),
    (1, 0), (1, 10),
    (2, 0), (2, 11),
)
