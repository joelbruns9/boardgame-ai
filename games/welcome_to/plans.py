"""
City Plans.

Transcribed from ``BGA Files/welcometo/modules/php/PlanCards.php`` and the
``modules/php/Plans/*.php`` classes.  Plan ids are the *array indices* of
``PlanCards::$plans``, so they can be compared directly against a BGA game log.

Three plans are in play at a time, one drawn from each of the three plan stacks.
Completing one is worth its first value; every player who completes it on a
later turn gets the second value.  Players who complete it on the *same* turn as
the first finisher all get the first value -- BGA ranks by turn number, not by
seat order, which is what makes the simultaneous turn faithful to serialise
(see :mod:`games.welcome_to.game`).

EXPANSIONS
──────────
The Ice Cream / Christmas / Easter plans are listed so that plan ids line up
with BGA, but :data:`SUPPORTED_VARIANTS` excludes them and
:func:`available_plan_ids` never deals them.  Their predicates raise.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Optional

from games.welcome_to.constants import (
    EXTREMITY_POSITIONS,
    NUM_STREETS,
    PARK_BOXES,
    STREET_SIZES,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from games.welcome_to.sheet import Estate, Pos, Sheet


class Variant(IntEnum):
    BASIC = 1
    ADVANCED = 2
    ICE_CREAM = 3
    CHRISTMAS = 4
    EASTER = 5


class PlanKind(IntEnum):
    ESTATE = 0
    FULL_STREET = 1
    FIVE_BIS = 2
    SEVEN_TEMP = 3
    EXTREMITIES = 4
    DECORATIVE = 5
    COMPLETE_STREET = 6
    UNSUPPORTED = 7


@dataclass(frozen=True, slots=True)
class Plan:
    id: int
    variant: Variant
    stack: int  # 1, 2 or 3
    scores: tuple[int, int]  # (first to finish, everyone later)
    kind: PlanKind
    params: tuple = ()

    @property
    def is_automatic(self) -> bool:
        """``AbstractPlan::$automatic`` -- true when validating needs no choice.

        Only :attr:`PlanKind.ESTATE` asks the player which housing estates to
        spend; every other plan resolves itself.
        """
        return self.kind is not PlanKind.ESTATE

    @property
    def required_sizes(self) -> tuple[int, ...]:
        return self.params if self.kind is PlanKind.ESTATE else ()


_E = PlanKind.ESTATE
_U = PlanKind.UNSUPPORTED

#: ``PlanCards::$plans``, index-for-index.
PLANS: tuple[Plan, ...] = (
    Plan(0, Variant.BASIC, 1, (8, 4), _E, (1, 1, 1, 1, 1, 1)),
    Plan(1, Variant.BASIC, 1, (8, 4), _E, (2, 2, 2, 2)),
    Plan(2, Variant.BASIC, 1, (8, 4), _E, (3, 3, 3)),
    Plan(3, Variant.BASIC, 1, (6, 3), _E, (4, 4)),
    Plan(4, Variant.BASIC, 1, (8, 4), _E, (5, 5)),
    Plan(5, Variant.BASIC, 1, (10, 6), _E, (6, 6)),
    Plan(6, Variant.BASIC, 2, (11, 6), _E, (1, 1, 1, 6)),
    Plan(7, Variant.BASIC, 2, (10, 6), _E, (5, 2, 2)),
    Plan(8, Variant.BASIC, 2, (12, 7), _E, (3, 3, 4)),
    Plan(9, Variant.BASIC, 2, (8, 4), _E, (3, 6)),
    Plan(10, Variant.BASIC, 2, (9, 5), _E, (4, 5)),
    Plan(11, Variant.BASIC, 2, (9, 5), _E, (4, 1, 1, 1)),
    Plan(12, Variant.BASIC, 3, (12, 7), _E, (1, 2, 6)),
    Plan(13, Variant.BASIC, 3, (13, 7), _E, (1, 4, 5)),
    Plan(14, Variant.BASIC, 3, (7, 3), _E, (3, 4)),
    Plan(15, Variant.BASIC, 3, (7, 3), _E, (2, 5)),
    Plan(16, Variant.BASIC, 3, (11, 6), _E, (1, 2, 2, 3)),
    Plan(17, Variant.BASIC, 3, (13, 7), _E, (2, 3, 5)),
    Plan(18, Variant.ADVANCED, 1, (8, 4), PlanKind.FULL_STREET, (2,)),
    Plan(19, Variant.ADVANCED, 1, (6, 3), PlanKind.FULL_STREET, (0,)),
    Plan(20, Variant.ADVANCED, 1, (8, 3), PlanKind.FIVE_BIS),
    Plan(21, Variant.ADVANCED, 1, (6, 3), PlanKind.SEVEN_TEMP),
    Plan(22, Variant.ADVANCED, 1, (7, 4), PlanKind.EXTREMITIES),
    Plan(23, Variant.ADVANCED, 2, (7, 4), PlanKind.DECORATIVE, ("park",)),
    Plan(24, Variant.ADVANCED, 2, (10, 5), PlanKind.COMPLETE_STREET),
    Plan(25, Variant.ADVANCED, 2, (7, 4), PlanKind.DECORATIVE, ("pool",)),
    Plan(26, Variant.ADVANCED, 2, (10, 5), PlanKind.DECORATIVE, ("pool&park", 2)),
    Plan(27, Variant.ADVANCED, 2, (8, 3), PlanKind.DECORATIVE, ("pool&park", 1)),
    # -- seasonal boards, listed for id fidelity only, never dealt --
    Plan(28, Variant.ICE_CREAM, 3, (6, 4), _U, ("iceCream",)),
    Plan(29, Variant.ICE_CREAM, 3, (7, 3), _U, ("without", 3, 4, 5)),
    Plan(30, Variant.ICE_CREAM, 3, (8, 4), _U, ("with", 4, 4, 4)),
    Plan(31, Variant.CHRISTMAS, 3, (10, 5), _U, ("with", 6, 6)),
    Plan(32, Variant.CHRISTMAS, 3, (14, 7), _U, ("christmas",)),
    Plan(33, Variant.CHRISTMAS, 3, (10, 5), _U, ("without", 3, 3)),
    Plan(34, Variant.EASTER, 3, (7, 3), _U, ("easterEgg",)),
    Plan(35, Variant.EASTER, 3, (10, 5), _U, ("without", 2, 3, 4)),
    Plan(36, Variant.EASTER, 3, (8, 4), _U, ("with", 3, 3, 3)),
)

NUM_PLANS: int = len(PLANS)
SUPPORTED_VARIANTS: frozenset[Variant] = frozenset({Variant.BASIC, Variant.ADVANCED})


def available_plan_ids(stack: int, advanced: bool) -> list[int]:
    """``AbstractPlan::isAvailable`` restricted to the base board.

    With the advanced variant on, stacks 1 and 2 gain five extra plans each;
    stack 3 is always the six basic plans because the cards that would replace
    it belong to the seasonal boards.
    """
    out = []
    for plan in PLANS:
        if plan.stack != stack:
            continue
        if plan.variant not in SUPPORTED_VARIANTS:
            continue
        if plan.variant is Variant.ADVANCED and not advanced:
            continue
        out.append(plan.id)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Completion predicates
# ──────────────────────────────────────────────────────────────────────────
def can_be_scored(plan: Plan, sheet: "Sheet") -> bool:
    """Whether ``sheet`` currently satisfies ``plan``.

    This is the per-sheet half of ``AbstractPlan::canBeScored``; the caller is
    responsible for the other half (that the player has not already scored this
    plan) because that lives in the game, not the sheet.
    """
    kind = plan.kind

    if kind is PlanKind.ESTATE:
        return _estates_available_for(plan.required_sizes, sheet) is not None

    if kind is PlanKind.FULL_STREET:
        x = plan.params[0]
        if any(sheet.top_fences[x]):
            return False
        return all(n is not None for n in sheet.numbers[x])

    if kind is PlanKind.FIVE_BIS:
        return any(c >= 5 for c in sheet.bis_count_per_street())

    if kind is PlanKind.SEVEN_TEMP:
        return sheet.temps >= 7

    if kind is PlanKind.EXTREMITIES:
        return all(
            sheet.numbers[x][y] is not None and not sheet.top_fences[x][y]
            for x, y in EXTREMITY_POSITIONS
        )

    if kind is PlanKind.COMPLETE_STREET:
        parks = sheet.street_parks_complete()
        pools = sheet.street_pools_complete()
        return any(
            parks[x] and pools[x] and sheet.has_roundabout_in_street(x)
            for x in range(NUM_STREETS)
        )

    if kind is PlanKind.DECORATIVE:
        what = plan.params[0]
        if what == "park":
            # "two streets must have all of the parks built"
            return sum(sheet.street_parks_complete()) >= 2
        if what == "pool":
            return sum(sheet.street_pools_complete()) >= 2
        if what == "pool&park":
            x = plan.params[1]
            return sheet.street_parks_complete()[x] and sheet.street_pools_complete()[x]
        raise NotImplementedError(f"seasonal decorative plan {plan.id} is not supported")

    raise NotImplementedError(f"plan {plan.id} belongs to an unsupported expansion")


def progress(plan: Plan, sheet: "Sheet") -> tuple[float, int]:
    """How close ``sheet`` is to completing ``plan``: ``(fraction, steps_left)``.

    The plan race is the main interaction in a low-interaction game, and it
    is decided by *who gets there first* — so a player needs to know not only
    whether a plan is done but how far off it is, for themselves and for
    everybody else.  Completion status alone cannot express "two parks away";
    this can, and it is computed for opponents from their public sheets too.

    ``fraction`` is 1.0 exactly when :func:`can_be_scored` is true.  ``steps_left``
    counts the marks still needed, in the natural unit of the plan (estates for
    an estate plan, boxes for the track plans) — the two are separate because a
    fraction hides whether the remaining work is one mark or six.
    """
    kind = plan.kind

    if kind is PlanKind.ESTATE:
        required = plan.required_sizes
        supply = Counter(size for _, _, size in sheet.free_estates())
        need = Counter(required)
        matched = sum(min(need[size], supply[size]) for size in need)
        left = len(required) - matched
        return matched / len(required), left

    if kind is PlanKind.FULL_STREET:
        x = plan.params[0]
        size = STREET_SIZES[x]
        if any(sheet.top_fences[x]):
            return 0.0, size  # a plan already ate part of this street
        built = sum(1 for n in sheet.numbers[x] if n is not None)
        return built / size, size - built

    if kind is PlanKind.FIVE_BIS:
        best = max(sheet.bis_count_per_street())
        return min(best, 5) / 5, max(0, 5 - best)

    if kind is PlanKind.SEVEN_TEMP:
        return min(sheet.temps, 7) / 7, max(0, 7 - sheet.temps)

    if kind is PlanKind.EXTREMITIES:
        done = sum(
            1
            for x, y in EXTREMITY_POSITIONS
            if sheet.numbers[x][y] is not None and not sheet.top_fences[x][y]
        )
        return done / len(EXTREMITY_POSITIONS), len(EXTREMITY_POSITIONS) - done

    if kind is PlanKind.COMPLETE_STREET:
        best_left = None
        best_cap = 1
        for x in range(NUM_STREETS):
            cap = PARK_BOXES[x] + 3 + 1
            left = (
                (PARK_BOXES[x] - sheet.parks[x])
                + (3 - sheet.pools[x])
                + (0 if sheet.has_roundabout_in_street(x) else 1)
            )
            if best_left is None or left < best_left:
                best_left, best_cap = left, cap
        best_left = best_left or 0
        return 1.0 - best_left / best_cap, best_left

    if kind is PlanKind.DECORATIVE:
        what = plan.params[0]
        if what == "park":
            needs = sorted(PARK_BOXES[x] - sheet.parks[x] for x in range(NUM_STREETS))
            left = sum(needs[:2])
            return 1.0 - left / (PARK_BOXES[0] + PARK_BOXES[1]), left
        if what == "pool":
            needs = sorted(3 - sheet.pools[x] for x in range(NUM_STREETS))
            left = sum(needs[:2])
            return 1.0 - left / 6, left
        if what == "pool&park":
            x = plan.params[1]
            cap = PARK_BOXES[x] + 3
            left = (PARK_BOXES[x] - sheet.parks[x]) + (3 - sheet.pools[x])
            return 1.0 - left / cap, left
        raise NotImplementedError(f"seasonal decorative plan {plan.id} is not supported")

    raise NotImplementedError(f"plan {plan.id} belongs to an unsupported expansion")


def _estates_available_for(
    sizes: tuple[int, ...], sheet: "Sheet"
) -> Optional[Counter]:
    """Return the per-size supply if ``sizes`` can be covered, else ``None``.

    ``EstatePlan::canBeScored`` does a multiset difference between the required
    sizes and the sizes of the estates that no plan has consumed yet.  Because
    every requirement asks for an *exact* size, feasibility is pure counting --
    which is what lets the engine hand the estates over one at a time without
    ever painting the player into a corner.
    """
    supply = Counter(size for _, _, size in sheet.free_estates())
    need = Counter(sizes)
    for size, count in need.items():
        if supply[size] < count:
            return None
    return supply


def estates_matching_size(
    sheet: "Sheet", size: int, already_chosen: tuple["Estate", ...] = ()
) -> list["Estate"]:
    """Free estates of exactly ``size`` that have not been picked yet this validation."""
    chosen = set(already_chosen)
    return [e for e in sheet.free_estates() if e[2] == size and e not in chosen]


def validation_cells(
    plan: Plan, sheet: "Sheet", chosen_estates: tuple["Estate", ...] = ()
) -> list["Pos"]:
    """Houses this plan consumes, which get a top fence and cannot be reused.

    ``EstatePlan``, ``FullStreetPlan`` and ``ExtremitiesPlan`` are the three that
    spend houses; the rest score off tracks and consume nothing.
    """
    if plan.kind is PlanKind.ESTATE:
        cells: list[Pos] = []
        for x, start, size in chosen_estates:
            cells.extend((x, start + k) for k in range(size))
        return cells

    if plan.kind is PlanKind.FULL_STREET:
        x = plan.params[0]
        return [(x, y) for y in range(STREET_SIZES[x])]

    if plan.kind is PlanKind.EXTREMITIES:
        return list(EXTREMITY_POSITIONS)

    return []
