"""Workstream 1: a stable name for every printed tableau location.

The pyramid is a fixed printed shape, so a location has the same structural
role in every game of an Age: "the left face-down slot under the apex" is one
thing whether it holds a Guild or a Palace. The encoder already gives the
network `row` and `x` as numbers, and numbers say where a slot is, not which
slot it *is* -- two coordinates cannot be told apart from their arithmetic.
`net.TokenEmbedder` attaches a learned embedding to the identity below.

Keyed by Age because the shapes differ. Age I and Age III both open at
``(row 0, x 5)`` and diverge by row 3, so a shared numbering would hand one
name to two different structural roles -- exactly the confusion the table
exists to remove.

**Why this is not in `data.py`, where the geometry lives.**
`control_table.rule_identity` digests the BYTES of `data.py` and
`tableau_control.py` to decide whether an installed control table still means
what the current code means. Appending anything at all to `data.py` -- even a
derived constant that changes no rule -- invalidates every control table and
every checkpoint bound to one, and the failure is a hard refusal at import.
The digest is right to be that blunt; the answer is to derive here instead.
"""

from __future__ import annotations

from .data import TABLEAU_LAYOUTS


def _age_slot_ids() -> dict[tuple[int, int, int], int]:
    """``(age, row, x) -> 0-based identity``, numbered per Age in layout order.

    Built from ``TABLEAU_LAYOUTS`` rather than written out, so it cannot drift
    from the geometry it names.
    """

    ids: dict[tuple[int, int, int], int] = {}
    running = 0
    for age in sorted(TABLEAU_LAYOUTS):
        for index, slot in enumerate(TABLEAU_LAYOUTS[age]):
            ids[(age, slot.row, slot.x)] = running + index
        running += len(TABLEAU_LAYOUTS[age])
    return ids


AGE_SLOT_IDS = _age_slot_ids()
NUM_AGE_SLOTS = len(AGE_SLOT_IDS)
MAX_AGE = max(age for age, _, _ in AGE_SLOT_IDS)
MAX_SLOT_ROW = max(row for _, row, _ in AGE_SLOT_IDS)
MAX_SLOT_X = max(x for _, _, x in AGE_SLOT_IDS)


# --- Workstream 2: the printed cover graph ----------------------------------
#
# Node ordering is the WITHIN-AGE slot index above, which is what makes the
# graph module cheap: the relation between two locations is a property of the
# printed shape, so it is one static matrix per Age rather than something to
# rebuild per position. That is the substantive sense in which W2 depends on
# W1 -- not the learned table, but the canonical numbering it fixed.
#
# The edges are STRUCTURAL and ignore which cards are still present. A slot two
# rows down is a distance-2 descendant whether or not the card between them has
# been taken. The dynamic half -- what is accessible now, who reaches it first
# -- is already carried by the `accessible` / `coverers` features and, exactly,
# by W3's control channels; duplicating it here would make the graph a worse
# copy of a solver.

SLOTS_PER_AGE = {age: len(layout) for age, layout in TABLEAU_LAYOUTS.items()}
MAX_SLOTS_PER_AGE = max(SLOTS_PER_AGE.values())

#: ``global id -> Age`` and ``global id -> index within its Age``.
AGE_OF_SLOT = [0] * NUM_AGE_SLOTS
WITHIN_AGE_INDEX = [0] * NUM_AGE_SLOTS
for _key, _id in AGE_SLOT_IDS.items():
    _age, _row, _x = _key
    AGE_OF_SLOT[_id] = _age
    WITHIN_AGE_INDEX[_id] = _id - min(
        value for (age, _, _), value in AGE_SLOT_IDS.items() if age == _age
    )


def _cover_matrix(age: int) -> list[list[bool]]:
    """``C[i][j]`` = slot ``i`` directly covers slot ``j``.

    Direction follows ``data.covering_slots``: a card in row ``r + 1`` covers
    the cards it overlaps in row ``r``. So the coverers of a slot sit BELOW it
    in the row numbering, which is the same convention the encoder's `coverers`
    feature counts.
    """

    layout = TABLEAU_LAYOUTS[age]
    return [
        [
            other.row == slot.row + 1 and abs(other.x - slot.x) == 1
            for slot in layout
        ]
        for other in layout
    ]


def _reachable_powers(cover: list[list[bool]]) -> list[list[list[bool]]]:
    """``powers[k][i][j]`` = ``j`` is ``k`` cover-steps below ``i`` (k >= 1)."""

    size = len(cover)
    powers = [cover]
    while any(any(row) for row in powers[-1]):
        previous = powers[-1]
        step = [
            [
                any(previous[i][m] and cover[m][j] for m in range(size))
                for j in range(size)
            ]
            for i in range(size)
        ]
        if not any(any(row) for row in step):
            break
        powers.append(step)
    return [None, *powers]  # 1-based: powers[k] is k steps


def _max_distance() -> int:
    return max(
        len(_reachable_powers(_cover_matrix(age))) - 1
        for age in TABLEAU_LAYOUTS
    )


#: The plan names transitive edges "at distance 2, 3, 4, or 5". Age III is seven
#: rows deep, so distance 6 exists; truncating at 5 would quietly declare the
#: deepest pair in the game structurally unrelated. The range is taken from the
#: layouts instead of written down.
MAX_COVER_DISTANCE = _max_distance()

RELATION_NAMES = (
    "none",
    "self",
    "covers",
    "covered_by",
    "sibling",
    *(f"descendant_{k}" for k in range(2, MAX_COVER_DISTANCE + 1)),
    *(f"ancestor_{k}" for k in range(2, MAX_COVER_DISTANCE + 1)),
)
RELATION_IDS = {name: index for index, name in enumerate(RELATION_NAMES)}
NUM_RELATIONS = len(RELATION_NAMES)


def relation_matrix(age: int) -> list[list[int]]:
    """``M[i][j]`` = the relation slot ``j`` bears to slot ``i``, as an id.

    Read as "what is j to me": ``covers`` means i covers j, ``covered_by``
    means j covers i. The categories are mutually exclusive by construction --
    a cover relation crosses rows and a sibling relation does not, and a
    transitive distance is fixed by the row difference -- so the priority order
    below only ever settles direct-versus-transitive, never a genuine tie.
    """

    cover = _cover_matrix(age)
    size = len(cover)
    powers = _reachable_powers(cover)
    shares_child = [
        [
            any(cover[i][c] and cover[j][c] for c in range(size))
            for j in range(size)
        ]
        for i in range(size)
    ]
    shares_parent = [
        [
            any(cover[p][i] and cover[p][j] for p in range(size))
            for j in range(size)
        ]
        for i in range(size)
    ]
    matrix = [[RELATION_IDS["none"]] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i == j:
                matrix[i][j] = RELATION_IDS["self"]
            elif cover[i][j]:
                matrix[i][j] = RELATION_IDS["covers"]
            elif cover[j][i]:
                matrix[i][j] = RELATION_IDS["covered_by"]
            elif shares_child[i][j] or shares_parent[i][j]:
                matrix[i][j] = RELATION_IDS["sibling"]
            else:
                for k in range(2, len(powers)):
                    if powers[k][i][j]:
                        matrix[i][j] = RELATION_IDS[f"descendant_{k}"]
                        break
                    if powers[k][j][i]:
                        matrix[i][j] = RELATION_IDS[f"ancestor_{k}"]
                        break
    return matrix


def relation_planes() -> list[list[list[int]]]:
    """One ``[MAX_SLOTS_PER_AGE, MAX_SLOTS_PER_AGE]`` plane per Age, plus a
    plane 0 of pure ``none`` for the "row has no Age" case.

    Padded to a common size so the planes stack into one tensor; the padding
    indices name slots no Age has, and nothing is ever present there.
    """

    size = MAX_SLOTS_PER_AGE
    planes = [[[RELATION_IDS["none"]] * size for _ in range(size)]]
    for age in sorted(TABLEAU_LAYOUTS):
        source = relation_matrix(age)
        plane = [[RELATION_IDS["none"]] * size for _ in range(size)]
        for i, row in enumerate(source):
            for j, value in enumerate(row):
                plane[i][j] = value
        planes.append(plane)
    return planes


# --- Workstream 5b: what an action uncovers ---------------------------------

#: The most slots any one slot covers. Two, in every printed Age: a card
#: overlaps the two below it. Derived rather than asserted, so a future layout
#: cannot quietly overflow the tensor built from it.
MAX_COVERED = max(
    max(sum(1 for value in row if value) for row in _cover_matrix(age))
    for age in TABLEAU_LAYOUTS
)


def covered_slots(age: int) -> list[list[int]]:
    """For each slot, the slots it COVERS -- the ones taking it can uncover.

    Direction matters and is easy to invert: `_cover_matrix(age)[i][j]` is "i
    covers j", and a card covers the two beneath it in the row numbering. So
    removing slot `i` is what can make its `covered_slots` reachable, and the
    coverers of `i` are a different set entirely.

    Padded to `MAX_COVERED` with `-1`, because a slot on the bottom row covers
    nothing and the tensor built from this is rectangular.
    """

    cover = _cover_matrix(age)
    size = len(cover)
    out = []
    for i in range(size):
        covered = [j for j in range(size) if cover[i][j]]
        out.append(covered + [-1] * (MAX_COVERED - len(covered)))
    return out


def covered_planes() -> list[list[list[int]]]:
    """One `[MAX_SLOTS_PER_AGE, MAX_COVERED]` plane per Age, plus an empty
    plane 0 for a row that names no Age.

    Values are WITHIN-AGE slot indices, or -1. Padded to a common size so the
    planes stack into one tensor, exactly as `relation_planes` is.
    """

    size = MAX_SLOTS_PER_AGE
    empty = [[-1] * MAX_COVERED for _ in range(size)]
    planes = [empty]
    for age in sorted(TABLEAU_LAYOUTS):
        source = covered_slots(age)
        plane = [[-1] * MAX_COVERED for _ in range(size)]
        for i, covered in enumerate(source):
            plane[i] = list(covered)
        planes.append(plane)
    return planes
