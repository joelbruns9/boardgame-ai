"""
Action codec -- the flat policy index space.

The engine speaks integers.  ``GameState.legal_actions()`` returns indices into
a single fixed space of :data:`NUM_ACTIONS` slots and ``GameState.step(i)``
consumes one, so there is no second representation that can drift out of sync
with the rules.  This module owns the layout and the helpers that turn an index
into something a human (or a test) can read.

LAYOUT
──────
Every slot below is a *semantic* action, not a phase-local option number.  The
same index always means the same thing, which is what makes a policy head over
these logits learnable; which subset is legal is entirely up to the phase.

    CHOOSE_STACK      6    pick a construction-card combination
                             standard mode : slots 0..2 are the three stacks
                             expert / solo : slots 0..5 are the ordered pairs
                                             in EXPERT_PAIRS -- take the number
                                             from one card, the effect from another
    PERMIT_REFUSAL    1    no combination is playable: take a refusal
    ROUNDABOUT_OPEN   1    (advanced) declare that you want to build a roundabout
    ROUNDABOUT_POS   33    ...then choose the empty box for it
    WRITE           165    write a number: 5 temp-agency deltas x 33 boxes.
                           Delta slot 0 is "no modifier" and is the only legal
                           one unless the chosen effect is TEMP.
    SURVEYOR_FENCE   30    draw an estate fence in one of the 30 fence slots
    ESTATE_ROW        6    cross a box in one of the six estate-value rows
    PARK_STREET       3    build a park in a street
    POOL_BUILD        1    build the pool under the house just written
    BIS              66    duplicate a neighbour's number: 33 boxes x 2 sides
                           (side 0 copies from the left, side 1 from the right)
    CHOOSE_PLAN       3    validate the City Plan in slot 0, 1 or 2
    VALIDATE_ESTATE  33    hand one housing estate to the plan being validated,
                           identified by the box its leftmost house sits in
    PASS_*            7    one dedicated pass per optional decision, so that a
                           "pass" logit never has to mean two different things
    RESHUFFLE_YES/NO  2    first plan of the game: reshuffle the deck or not

BIS ENCODING
────────────
A bis is stored as (box, side) rather than (box, number) because the number is
implied by the neighbour, and because the *same* empty box can legally take two
different numbers when its two neighbours differ.  33 x 2 stays compact where
33 x 18 would not.

ESTATE ENCODING
───────────────
A plan that asks for housing estates is resolved one estate at a time: the
engine asks for an estate of a specific size, the policy names it by its
leftmost box, and the engine moves on to the next required size.  Because every
requirement asks for an *exact* size, any estate of the requested size keeps the
rest of the plan satisfiable, so this sequential form never traps the player
(see ``plans._estates_available_for``).
"""
from __future__ import annotations

from typing import Final

from games.welcome_to.constants import (
    NUM_BOXES,
    NUM_FENCES,
    TEMP_DELTAS,
    box_coords,
    box_index,
    fence_coords,
    fence_index,
)

#: Ordered (number-card, effect-card) stack pairs used by expert and solo mode,
#: in the order produced by ``ConstructionCards::getPossibleCombinations``.
EXPERT_PAIRS: Final = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))

NUM_STACK_SLOTS: Final = 6
NUM_TEMP_DELTAS: Final = len(TEMP_DELTAS)  # 5
NUM_ESTATE_ROWS: Final = 6
NUM_PLAN_SLOTS: Final = 3
NUM_BIS_SIDES: Final = 2


def _layout() -> tuple[dict[str, int], int]:
    sizes = (
        ("CHOOSE_STACK", NUM_STACK_SLOTS),
        ("PERMIT_REFUSAL", 1),
        ("ROUNDABOUT_OPEN", 1),
        ("ROUNDABOUT_POS", NUM_BOXES),
        ("WRITE", NUM_TEMP_DELTAS * NUM_BOXES),
        ("SURVEYOR_FENCE", NUM_FENCES),
        ("ESTATE_ROW", NUM_ESTATE_ROWS),
        ("PARK_STREET", 3),
        ("POOL_BUILD", 1),
        ("BIS", NUM_BOXES * NUM_BIS_SIDES),
        ("CHOOSE_PLAN", NUM_PLAN_SLOTS),
        ("VALIDATE_ESTATE", NUM_BOXES),
        ("PASS_ROUNDABOUT", 1),
        ("PASS_SURVEYOR", 1),
        ("PASS_ESTATE", 1),
        ("PASS_PARK", 1),
        ("PASS_POOL", 1),
        ("PASS_BIS", 1),
        ("PASS_PLAN", 1),
        ("RESHUFFLE_YES", 1),
        ("RESHUFFLE_NO", 1),
    )
    offsets: dict[str, int] = {}
    cursor = 0
    for name, size in sizes:
        offsets[name] = cursor
        cursor += size
    return offsets, cursor


OFFSET, NUM_ACTIONS = _layout()

A_CHOOSE_STACK: Final = OFFSET["CHOOSE_STACK"]
A_PERMIT_REFUSAL: Final = OFFSET["PERMIT_REFUSAL"]
A_ROUNDABOUT_OPEN: Final = OFFSET["ROUNDABOUT_OPEN"]
A_ROUNDABOUT_POS: Final = OFFSET["ROUNDABOUT_POS"]
A_WRITE: Final = OFFSET["WRITE"]
A_SURVEYOR_FENCE: Final = OFFSET["SURVEYOR_FENCE"]
A_ESTATE_ROW: Final = OFFSET["ESTATE_ROW"]
A_PARK_STREET: Final = OFFSET["PARK_STREET"]
A_POOL_BUILD: Final = OFFSET["POOL_BUILD"]
A_BIS: Final = OFFSET["BIS"]
A_CHOOSE_PLAN: Final = OFFSET["CHOOSE_PLAN"]
A_VALIDATE_ESTATE: Final = OFFSET["VALIDATE_ESTATE"]
A_PASS_ROUNDABOUT: Final = OFFSET["PASS_ROUNDABOUT"]
A_PASS_SURVEYOR: Final = OFFSET["PASS_SURVEYOR"]
A_PASS_ESTATE: Final = OFFSET["PASS_ESTATE"]
A_PASS_PARK: Final = OFFSET["PASS_PARK"]
A_PASS_POOL: Final = OFFSET["PASS_POOL"]
A_PASS_BIS: Final = OFFSET["PASS_BIS"]
A_PASS_PLAN: Final = OFFSET["PASS_PLAN"]
A_RESHUFFLE_YES: Final = OFFSET["RESHUFFLE_YES"]
A_RESHUFFLE_NO: Final = OFFSET["RESHUFFLE_NO"]


# ──────────────────────────────────────────────────────────────────────────
# Encoders
# ──────────────────────────────────────────────────────────────────────────
def choose_stack(slot: int) -> int:
    return A_CHOOSE_STACK + slot


def roundabout_pos(x: int, y: int) -> int:
    return A_ROUNDABOUT_POS + box_index(x, y)


def write(delta_slot: int, x: int, y: int) -> int:
    """``delta_slot`` indexes :data:`~games.welcome_to.constants.TEMP_DELTAS`."""
    return A_WRITE + delta_slot * NUM_BOXES + box_index(x, y)


def surveyor_fence(x: int, j: int) -> int:
    return A_SURVEYOR_FENCE + fence_index(x, j)


def estate_row(row: int) -> int:
    return A_ESTATE_ROW + row


def park_street(x: int) -> int:
    return A_PARK_STREET + x


def bis(x: int, y: int, side: int) -> int:
    return A_BIS + box_index(x, y) * NUM_BIS_SIDES + side


def choose_plan(slot: int) -> int:
    return A_CHOOSE_PLAN + slot


def validate_estate(x: int, y_start: int) -> int:
    return A_VALIDATE_ESTATE + box_index(x, y_start)


# ──────────────────────────────────────────────────────────────────────────
# Decoders
# ──────────────────────────────────────────────────────────────────────────
def decode_stack(index: int) -> int:
    return index - A_CHOOSE_STACK


def decode_roundabout_pos(index: int) -> tuple[int, int]:
    return box_coords(index - A_ROUNDABOUT_POS)


def decode_write(index: int) -> tuple[int, int, int]:
    """Returns ``(delta_slot, x, y)``."""
    rel = index - A_WRITE
    return rel // NUM_BOXES, *box_coords(rel % NUM_BOXES)


def decode_surveyor_fence(index: int) -> tuple[int, int]:
    return fence_coords(index - A_SURVEYOR_FENCE)


def decode_estate_row(index: int) -> int:
    return index - A_ESTATE_ROW


def decode_park_street(index: int) -> int:
    return index - A_PARK_STREET


def decode_bis(index: int) -> tuple[int, int, int]:
    """Returns ``(x, y, side)``."""
    rel = index - A_BIS
    x, y = box_coords(rel // NUM_BIS_SIDES)
    return x, y, rel % NUM_BIS_SIDES


def decode_plan(index: int) -> int:
    return index - A_CHOOSE_PLAN


def decode_validate_estate(index: int) -> tuple[int, int]:
    return box_coords(index - A_VALIDATE_ESTATE)


# ──────────────────────────────────────────────────────────────────────────
# Human readable
# ──────────────────────────────────────────────────────────────────────────
_SINGLETONS = {
    A_PERMIT_REFUSAL: "permit-refusal",
    A_ROUNDABOUT_OPEN: "roundabout-open",
    A_POOL_BUILD: "build-pool",
    A_PASS_ROUNDABOUT: "pass(roundabout)",
    A_PASS_SURVEYOR: "pass(surveyor)",
    A_PASS_ESTATE: "pass(estate)",
    A_PASS_PARK: "pass(park)",
    A_PASS_POOL: "pass(pool)",
    A_PASS_BIS: "pass(bis)",
    A_PASS_PLAN: "pass(plan)",
    A_RESHUFFLE_YES: "reshuffle=yes",
    A_RESHUFFLE_NO: "reshuffle=no",
}


def describe(index: int) -> str:
    """A short label for logs, tests and debugging output."""
    if index in _SINGLETONS:
        return _SINGLETONS[index]
    if index < A_PERMIT_REFUSAL:
        return f"stack[{decode_stack(index)}]"
    if A_ROUNDABOUT_POS <= index < A_WRITE:
        x, y = decode_roundabout_pos(index)
        return f"roundabout@{x},{y}"
    if A_WRITE <= index < A_SURVEYOR_FENCE:
        d, x, y = decode_write(index)
        return f"write(delta={TEMP_DELTAS[d]:+d})@{x},{y}"
    if A_SURVEYOR_FENCE <= index < A_ESTATE_ROW:
        x, j = decode_surveyor_fence(index)
        return f"fence@{x},{j}"
    if A_ESTATE_ROW <= index < A_PARK_STREET:
        return f"estate-row[{decode_estate_row(index)}]"
    if A_PARK_STREET <= index < A_POOL_BUILD:
        return f"park@street{decode_park_street(index)}"
    if A_BIS <= index < A_CHOOSE_PLAN:
        x, y, side = decode_bis(index)
        return f"bis@{x},{y}({'left' if side == 0 else 'right'})"
    if A_CHOOSE_PLAN <= index < A_VALIDATE_ESTATE:
        return f"plan[{decode_plan(index)}]"
    if A_VALIDATE_ESTATE <= index < A_PASS_ROUNDABOUT:
        x, y = decode_validate_estate(index)
        return f"give-estate@{x},{y}"
    raise ValueError(f"action index {index} out of range")
