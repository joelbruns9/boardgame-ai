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
    FENCE_SIZES,
    MAX_ESTATE_SIZE,
    NUM_STREETS,
    PARK_BOXES,
    POOL_POSITIONS,
    STREET_SIZES,
    TEMP_BOXES,
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

#: The plan ids :func:`available_plan_ids` can actually deal, in id order.
#:
#: :data:`PLANS` holds 37 entries so that ids line up with BGA, but nine of them
#: are seasonal boards that no supported variant ever puts on the table.  A
#: 37-wide one-hot would therefore carry nine permanently dead input slots.
#:
#: This is the **advanced superset**, deliberately, and it is sized that way
#: whether or not the game being encoded has the advanced variant on: a
#: base-rules game is then a strict subset of the same input space, with ten
#: slots dark, and one set of network weights reads both.
DEALT_PLAN_IDS: tuple[int, ...] = tuple(
    plan.id for plan in PLANS if plan.variant in SUPPORTED_VARIANTS
)
NUM_DEALT_PLANS: int = len(DEALT_PLAN_IDS)

_DENSE_INDEX: dict[int, int] = {pid: i for i, pid in enumerate(DEALT_PLAN_IDS)}


def dense_index(plan_id: int) -> int:
    """Position of ``plan_id`` in :data:`DEALT_PLAN_IDS`.

    Raises for a seasonal id rather than returning a placeholder: an id that
    cannot be dealt reaching the encoder means the caller built a state this
    engine does not support, and silently folding it into slot 0 would put a
    real plan's weights under an unsupported one.
    """
    try:
        return _DENSE_INDEX[plan_id]
    except KeyError:
        raise ValueError(
            f"plan {plan_id} belongs to an unsupported expansion and is never dealt"
        ) from None


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


# ──────────────────────────────────────────────────────────────────────────
# Encoder v3: requirements, feasibility and a hard turn bound
#
# ENCODER_V3_SPEC.md §2, §3, §6.1 and §6.2.  Read §6.1 before touching
# `feasible`: five review rounds of that spec produced four unsound death
# tests, every one of them by reasoning about *numbers* while forgetting that
# a roundabout, a bis or a fence puts a house on the sheet without one.
# ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Requirements:
    """What a plan still wants from one sheet -- "how much" AND "where".

    The v2 spec described a plan by a single 7-vector of counts, which answered
    "how much" and never "where".  Measured against the dealt pool that loses
    almost everything: 18 of the 28 plans are estate plans whose 18 distinct
    size multisets all collapse to "N fences needed", and 9 of the remaining 10
    are street- or box-bound.  Exactly one -- ``SEVEN_TEMP`` -- survived intact.

    So the counts are split by locus: per-street where the plan binds to a
    street, per-size where it binds to estate sizes, and ``target_boxes`` where
    it binds to named boxes.
    """

    #: Sheet-wide.
    temps_needed: int
    #: Descriptor of estate work remaining.  NOT a bound on fences: one SURVEYOR
    #: fence can raise the estate match by two, so this may not be used as a
    #: lower bound on anything.  See :func:`turns_lower_bound`.
    fences_needed: int
    #: ``need[s] - free supply[s]`` for sizes 1..6, clipped at zero.
    estate_shortfall: tuple[int, ...]

    #: Per street.
    parks_needed: tuple[int, ...]
    pools_needed: tuple[int, ...]
    houses_needed: tuple[int, ...]
    bis_needed: tuple[int, ...]
    roundabout_needed: tuple[int, ...]
    #: 1 where street *x* is **alive** for this plan -- not provably unable to
    #: contribute.  A street whose work is already done is alive, not dead; the
    #: remaining work lives in the other five vectors.
    street_serves: tuple[int, ...]

    #: Boxes this plan still needs *written*, for the plan-target planes.
    #: Non-empty only for ``FULL_STREET`` and ``EXTREMITIES``, whose targets are
    #: fixed by the plan itself.  Estate plans are deliberately absent: which
    #: boxes they would consume depends on a choice, and marking "boxes whose
    #: filling would create a needed size" would be a search, not a feature.
    target_boxes: tuple[tuple[int, int], ...]


def _regions(sheet: "Sheet", x: int) -> list[range]:
    """Maximal spans of street ``x`` between existing fences and the street ends.

    Fences are only ever added, never removed, so a region is the widest piece
    of street that could still become a single estate.
    """
    out: list[range] = []
    size = STREET_SIZES[x]
    start = 0
    for j in range(size):
        if j == size - 1 or (j < FENCE_SIZES[x] and sheet.fences[x][j]):
            out.append(range(start, j + 1))
            start = j + 1
    return out


def reachable_estate_counts(sheet: "Sheet") -> list[int]:
    """Loose upper bound on estates of each size 1..6 this sheet could still hold.

    Deliberately the **loose** bound of ``ENCODER_V3_SPEC.md`` §13.2, shipped
    ahead of a tighter one because three sibling death tests were unsound in
    earlier drafts and the conservative default is the right posture until the
    soundness fuzz is green on this form.

    It counts, per fence-delimited region, ``floor(usable / s)`` estates of size
    ``s``, where ``usable`` is the boxes not already spent on a City Plan.  Two
    deliberate over-counts, both in the safe direction:

    * it ignores whether the fences that would carve the region are *legal*
      (``surveyor_zones`` refuses to split a bis pair or two same-plan houses);
    * it ignores contiguity, so a region may be credited with more estates than
      it could really yield.

    **Critically it counts boxes, not free boxes.** ``Sheet.estates`` is bounded
    by fences, not by writes, so a single fence re-partitions an already-built
    run into estates of new sizes while consuming nothing.  A bound counting
    only empty boxes called a fully written sheet dead when one fence completed
    the plan.
    """
    counts = [0] * MAX_ESTATE_SIZE
    for x in range(NUM_STREETS):
        for region in _regions(sheet, x):
            usable = sum(1 for y in region if not sheet.top_fences[x][y])
            for size in range(1, MAX_ESTATE_SIZE + 1):
                counts[size - 1] += usable // size
    return counts


def _pool_boxes_alive(sheet: "Sheet", x: int) -> int:
    """Pool positions in street ``x`` that could still take a house."""
    spans = sheet.span_if_roundabout()
    alive = 0
    for px, py in POOL_POSITIONS:
        if px != x:
            continue
        if sheet.numbers[x][py] is not None:
            continue
        if spans[x][py] > 0 or sheet.bis_reachable(x, py):
            alive += 1
    return alive


def feasible(plan: Plan, sheet: "Sheet") -> bool:
    """Is ``plan`` still reachable on ``sheet``?  **Sound, not complete.**

    Returns ``False`` only when the plan is *provably* unreachable, and ``True``
    otherwise -- so it will miss some real deaths and must never report one that
    is not real.  A false death is a feature that lies to the network; a missed
    death is a feature that is merely weak.

    ``tests/plan_reachability.py`` is the independent oracle this is checked
    against, one-sidedly: ``feasible`` false requires the oracle to agree, never
    the converse.

    Every span-derived claim below is guarded against **all** the ways a house
    reaches a box that no drawn number can:

    * a **roundabout** -- ``available_locations(None)`` ignores numeric fit and
      the sentinel counts as a built house (hence ``span_if_roundabout``);
    * a **bis** -- ``bis_candidates`` has no ascending-order check at all
      (hence ``bis_reachable``);
    * a **fence** -- creates estates of new sizes consuming no box at all
      (hence :func:`reachable_estate_counts`).

    And no bound here subtracts from ``BIS_BOXES`` or ``TEMP_BOXES``.  Those two
    tracks **saturate rather than gate**: ``legal_actions`` never reads their
    counters, and the apply path clamps with ``min(marks + 1, CAP)``.
    ``PERMIT_BOXES`` and ``ROUNDABOUT_BOXES`` do gate, via ``can_take_permit``
    and ``can_build_roundabout``.
    """
    kind = plan.kind

    if kind is PlanKind.ESTATE:
        supply = Counter(size for _, _, size in sheet.free_estates())
        reachable = reachable_estate_counts(sheet)
        for size, count in Counter(plan.required_sizes).items():
            if count > supply[size] + reachable[size - 1]:
                return False
        return True

    if kind is PlanKind.FULL_STREET:
        # Exact on its own, and the only clause: a house already spent on
        # another City Plan can never be un-spent.  A capacity clause was tried
        # and removed -- with roundabouts and bis both able to fill
        # numerically-dead boxes, no cheap capacity bound is sound.
        return not any(sheet.top_fences[plan.params[0]])

    if kind is PlanKind.EXTREMITIES:
        spans = sheet.span_if_roundabout()
        for x, y in EXTREMITY_POSITIONS:
            if sheet.top_fences[x][y]:
                return False
            if sheet.numbers[x][y] is not None:
                continue
            if spans[x][y] == 0 and not sheet.bis_reachable(x, y):
                return False
        return True

    if kind is PlanKind.FIVE_BIS:
        reach = sheet.bis_reach()
        counts = sheet.bis_count_per_street()
        return any(counts[x] + reach[x] >= 5 for x in range(NUM_STREETS))

    if kind is PlanKind.SEVEN_TEMP:
        # The temp track is eleven boxes and nothing consumes it but temp marks,
        # so seven is always still reachable.  Kept for uniformity.
        return TEMP_BOXES >= 7

    if kind is PlanKind.COMPLETE_STREET:
        for x in range(NUM_STREETS):
            if sheet.pools[x] + _pool_boxes_alive(sheet, x) < 3:
                continue
            if not sheet.has_roundabout_in_street(x):
                if not sheet.can_build_roundabout():
                    continue
            return True
        return False

    if kind is PlanKind.DECORATIVE:
        what = plan.params[0]
        if what == "park":
            return True  # park boxes are a pure track; nothing can block them
        if what == "pool":
            ok = sum(
                1
                for x in range(NUM_STREETS)
                if sheet.pools[x] + _pool_boxes_alive(sheet, x) >= 3
            )
            return ok >= 2
        if what == "pool&park":
            x = plan.params[1]
            return sheet.pools[x] + _pool_boxes_alive(sheet, x) >= 3
        raise NotImplementedError(f"seasonal decorative plan {plan.id} is not supported")

    raise NotImplementedError(f"plan {plan.id} belongs to an unsupported expansion")


def requirements(plan: Plan, sheet: "Sheet") -> Requirements:
    """What ``plan`` still wants from ``sheet``, split by locus.  Spec §3."""
    kind = plan.kind

    temps_needed = 0
    fences_needed = 0
    estate_shortfall = [0] * MAX_ESTATE_SIZE
    parks = [0, 0, 0]
    pools = [0, 0, 0]
    houses = [0, 0, 0]
    bis = [0, 0, 0]
    roundabout = [0, 0, 0]
    serves = [0, 0, 0]
    targets: list[tuple[int, int]] = []

    alive = feasible(plan, sheet)
    done = can_be_scored(plan, sheet)

    if kind is PlanKind.ESTATE:
        supply = Counter(size for _, _, size in sheet.free_estates())
        for size, count in Counter(plan.required_sizes).items():
            estate_shortfall[size - 1] = max(0, count - supply[size])
        fences_needed = progress(plan, sheet)[1]
        serves = [1 if alive else 0] * NUM_STREETS

    elif kind is PlanKind.FULL_STREET:
        x = plan.params[0]
        serves[x] = 1 if alive else 0
        houses[x] = sum(1 for n in sheet.numbers[x] if n is None)
        targets = [
            (x, y) for y in range(STREET_SIZES[x]) if sheet.numbers[x][y] is None
        ]

    elif kind is PlanKind.EXTREMITIES:
        for x, y in EXTREMITY_POSITIONS:
            serves[x] = 1 if alive else 0
            if sheet.numbers[x][y] is None:
                houses[x] += 1
                targets.append((x, y))

    elif kind is PlanKind.FIVE_BIS:
        reach = sheet.bis_reach()
        counts = sheet.bis_count_per_street()
        for x in range(NUM_STREETS):
            if counts[x] + reach[x] >= 5:
                serves[x] = 1
                bis[x] = max(0, 5 - counts[x])

    elif kind is PlanKind.SEVEN_TEMP:
        temps_needed = max(0, 7 - sheet.temps)
        # Sheet-wide: no street contributes, which is correct and is not "dead".

    elif kind is PlanKind.COMPLETE_STREET:
        for x in range(NUM_STREETS):
            if sheet.pools[x] + _pool_boxes_alive(sheet, x) < 3:
                continue
            if not sheet.has_roundabout_in_street(x) and not sheet.can_build_roundabout():
                continue
            serves[x] = 1
            parks[x] = PARK_BOXES[x] - sheet.parks[x]
            pools[x] = 3 - sheet.pools[x]
            roundabout[x] = 0 if sheet.has_roundabout_in_street(x) else 1

    elif kind is PlanKind.DECORATIVE:
        what = plan.params[0]
        streets = range(NUM_STREETS) if what != "pool&park" else (plan.params[1],)
        wants_pool = what in ("pool", "pool&park")
        for x in streets:
            if wants_pool and sheet.pools[x] + _pool_boxes_alive(sheet, x) < 3:
                continue
            serves[x] = 1
            if what in ("park", "pool&park"):
                parks[x] = PARK_BOXES[x] - sheet.parks[x]
            if wants_pool:
                pools[x] = 3 - sheet.pools[x]
    else:
        raise NotImplementedError(f"plan {plan.id} belongs to an unsupported expansion")

    if done:
        # A completed plan wants nothing.
        #
        # Note this is gated on `done` ALONE, not on `done or not alive`.  Every
        # field except `street_serves` states what the plan still *wants*, and
        # that stays true of a plan that has become unreachable -- a dead estate
        # plan is still short a 3 and a 6.  Aliveness has exactly one home, the
        # `feasible` scalar, mirrored by `street_serves`; blanking the demand
        # vectors as well would make the two fields redundant and destroy the
        # information a reader needs to see WHY a plan died.
        temps_needed = 0
        fences_needed = 0
        estate_shortfall = [0] * MAX_ESTATE_SIZE
        parks = [0, 0, 0]
        pools = [0, 0, 0]
        houses = [0, 0, 0]
        bis = [0, 0, 0]
        roundabout = [0, 0, 0]
        targets = []

    if not alive:
        # No street can contribute to a plan that cannot complete.  Per-kind
        # branches above judge streets independently -- a `pool` plan needs two
        # viable streets and may have one -- so the whole-plan verdict has to be
        # applied here rather than left to each branch.
        serves = [0, 0, 0]
    elif done:
        # ...but a COMPLETED plan's streets stay ALIVE.  A street whose work is
        # finished is the one that contributed most, and reading it as "cannot
        # contribute" was the defect the aliveness definition exists to fix.
        pass

    return Requirements(
        temps_needed=temps_needed,
        fences_needed=fences_needed,
        estate_shortfall=tuple(estate_shortfall),
        parks_needed=tuple(parks),
        pools_needed=tuple(pools),
        houses_needed=tuple(houses),
        bis_needed=tuple(bis),
        roundabout_needed=tuple(roundabout),
        street_serves=tuple(serves),
        target_boxes=tuple(targets),
    )


def turns_lower_bound(plan: Plan, sheet: "Sheet") -> int:
    """Fewest turns in which ``plan`` could still complete.  A hard bound.

    Two terms, both sound:

    ``effect_term``
        A turn takes **one** combination and so applies **one** effect mark, so
        the marks still needed *sum*.  An earlier draft took a max over effects;
        sound, but needlessly weak on the one feature sold as a hard bound -- and
        it contradicted the argument for the rate features, which sum for exactly
        this reason.

    ``house_term``
        Houses needed divided by **three**, the absolute per-turn ceiling:
        ``ROUNDABOUT_OPEN`` is legal before the write and a roundabout counts as
        a built house, so a turn can be roundabout -> write -> bis.  Three is a
        constant, deliberately not ``max_houses_this_turn`` -- a *lower bound on
        turns* must not fluctuate with the current offer.

    The estate contribution is ``0 if done else 1``, not the number of missing
    estates.  One SURVEYOR fence can raise the estate match by two -- a fenced
    run of six against a plan needing ``(3, 3)`` scores nothing, and one fence in
    its middle scores both -- so "one missing estate is one fence" is not a lower
    bound.  Weak but sound is the correct trade here.
    """
    req = requirements(plan, sheet)
    effect_term = req.temps_needed + sum(req.parks_needed) + sum(req.pools_needed)
    if any(req.estate_shortfall):
        effect_term += 1
    houses = sum(req.houses_needed)
    house_term = -(-houses // 3)  # ceiling division
    return max(effect_term, house_term)
