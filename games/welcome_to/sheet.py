"""
The player sheet: houses, fences and every score track.

This is the own-sheet half of Welcome To.  Everything a player can mark is
here, along with the legality rules that depend only on one player's own sheet
(where a number may be written, which fences may be drawn, what counts as a
housing estate) and the part of scoring that is purely local.

Two scoring components are deliberately NOT here because they depend on the
whole table rather than one sheet:

* the temp-agency track, which is scored by *rank* across players
  (:func:`games.welcome_to.game.temp_scores`), and
* City Plans, which are scored by who got there first
  (:class:`games.welcome_to.game.GameState`).

Source of truth: ``BGA Files/welcometo/modules/php/Houses.php`` and
``modules/php/Actions/*.php``.  Method docstrings name the PHP function each
rule was transcribed from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from games.welcome_to.constants import (
    BIS_BOXES,
    BIS_SCORES,
    MAX_NUMBER,
    MIN_NUMBER,
    ESTATE_ROW_BOXES,
    ESTATE_ROW_SCORES,
    FENCE_SIZES,
    MAX_ESTATE_SIZE,
    NUM_STREETS,
    PARK_BOXES,
    PARK_SCORES,
    PERMIT_BOXES,
    PERMIT_SCORES,
    POOL_BOXES,
    POOL_POSITION_SET,
    POOL_SCORES,
    ROUNDABOUT,
    ROUNDABOUT_BOXES,
    ROUNDABOUT_SCORES,
    STREET_SIZES,
)

def _row_capacity(row: list[Optional[int]], size: int) -> int:
    """The most houses that could still be written in one street's ``row``.

    Factored out of :meth:`Sheet.placement_capacity` so that
    :meth:`Sheet.capacity_if_roundabout` can evaluate a *hypothetical* row
    through the identical rule.  Two copies of this arithmetic would drift, and
    the hypothetical is only meaningful if it is the same function.
    """
    total = 0
    y = 0
    while y < size:
        if row[y] is not None:
            y += 1
            continue
        start = y
        while y < size and row[y] is None:
            y += 1
        run = y - start

        low = MIN_NUMBER - 1
        left = start - 1
        if left >= 0 and row[left] is not None and row[left] != ROUNDABOUT:
            low = row[left]
        high = MAX_NUMBER + 1
        if y < size and row[y] is not None and row[y] != ROUNDABOUT:
            high = row[y]

        total += min(run, max(0, high - low - 1))
    return total


#: ``(street, box)``
Pos = tuple[int, int]
#: ``(street, first box, size)`` -- a housing estate.
Estate = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class SheetScore:
    """Everything a single sheet is worth.

    ``plans`` and ``temp`` are filled in by the game, which is why
    :meth:`Sheet.local_score` leaves them at zero.
    """

    parks: int = 0
    pools: int = 0
    estates: int = 0
    plans: int = 0
    temp: int = 0
    bis: int = 0
    permits: int = 0
    roundabouts: int = 0

    @property
    def total(self) -> int:
        """``Player::computeScore`` -- the three penalties are subtracted."""
        return (
            self.plans
            + self.parks
            + self.pools
            + self.temp
            + self.estates
            - self.bis
            - self.permits
            - self.roundabouts
        )


@dataclass
class Sheet:
    """One player's score sheet.

    ``numbers[x][y]`` is ``None`` for an empty box, otherwise the written number
    (0..17) or :data:`~games.welcome_to.constants.ROUNDABOUT`.

    ``fences[x][j]`` is the *estate* fence between boxes ``j`` and ``j + 1``;
    ``top_fences[x][y]`` marks a house consumed by a City Plan, which stops it
    being reused by a later plan and stops a fence being drawn through it.
    """

    numbers: list[list[Optional[int]]] = field(default_factory=list)
    is_bis: list[list[bool]] = field(default_factory=list)
    written_turn: list[list[int]] = field(default_factory=list)
    fences: list[list[bool]] = field(default_factory=list)
    top_fences: list[list[bool]] = field(default_factory=list)
    parks: list[int] = field(default_factory=lambda: [0, 0, 0])
    pools: list[int] = field(default_factory=lambda: [0, 0, 0])
    estate_marks: list[int] = field(default_factory=lambda: [0] * 6)
    temps: int = 0
    bis_marks: int = 0
    permits: int = 0
    roundabouts: int = 0

    @classmethod
    def new(cls) -> "Sheet":
        return cls(
            numbers=[[None] * n for n in STREET_SIZES],
            is_bis=[[False] * n for n in STREET_SIZES],
            written_turn=[[-1] * n for n in STREET_SIZES],
            fences=[[False] * n for n in FENCE_SIZES],
            top_fences=[[False] * n for n in STREET_SIZES],
        )

    def copy(self) -> "Sheet":
        return Sheet(
            numbers=[list(r) for r in self.numbers],
            is_bis=[list(r) for r in self.is_bis],
            written_turn=[list(r) for r in self.written_turn],
            fences=[list(r) for r in self.fences],
            top_fences=[list(r) for r in self.top_fences],
            parks=list(self.parks),
            pools=list(self.pools),
            estate_marks=list(self.estate_marks),
            temps=self.temps,
            bis_marks=self.bis_marks,
            permits=self.permits,
            roundabouts=self.roundabouts,
        )

    # ------------------------------------------------------------------
    # Writing numbers
    # ------------------------------------------------------------------
    def available_locations(self, number: Optional[int] = None) -> list[Pos]:
        """``Houses::getAvailableLocations``.

        Numbers must increase strictly from left to right within a street.  A
        roundabout resets the chain in both directions, so it acts as a divider
        rather than as a number.  ``number=None`` asks for *every* empty box,
        which is how the engine finds roundabout sites and detects a full sheet.
        """
        result: list[Pos] = []
        for x, size in enumerate(STREET_SIZES):
            row = self.numbers[x]
            ok = [False] * size

            highest = -1
            for y in range(size):
                n = row[y]
                if n is None:
                    ok[y] = number is None or number > highest
                else:
                    ok[y] = False
                    highest = -1 if n == ROUNDABOUT else n

            lowest = 18
            for y in range(size - 1, -1, -1):
                n = row[y]
                if n is None:
                    ok[y] = True if number is None else (ok[y] and number < lowest)
                else:
                    lowest = 18 if n == ROUNDABOUT else n

            result.extend((x, y) for y in range(size) if ok[y])
        return result

    def has_free_box(self) -> bool:
        """Cheap ``available_locations(None)`` emptiness test (end-of-game check)."""
        return any(n is None for row in self.numbers for n in row)

    def box_spans(self) -> list[list[int]]:
        """How many distinct numbers each empty box could still legally take.

        For an empty box, the strictly-ascending rule bounds it by the nearest
        written number to its left and right; a roundabout or the end of the
        street removes that bound.  The span is the width of the open interval
        between them, so a box between a 5 and a 6 has span 0 and is dead.

        Filled boxes report 0.  This is the local view of the same quantity
        :meth:`placement_capacity` totals up per street.
        """
        spans: list[list[int]] = []
        for x, size in enumerate(STREET_SIZES):
            row = self.numbers[x]
            street: list[int] = []
            for y in range(size):
                if row[y] is not None:
                    street.append(0)
                    continue
                low = MIN_NUMBER - 1
                for k in range(y - 1, -1, -1):
                    if row[k] is not None:
                        if row[k] != ROUNDABOUT:
                            low = row[k]
                        break
                high = MAX_NUMBER + 1
                for k in range(y + 1, size):
                    if row[k] is not None:
                        if row[k] != ROUNDABOUT:
                            high = row[k]
                        break
                street.append(max(0, high - low - 1))
            spans.append(street)
        return spans

    def placement_capacity(self) -> list[int]:
        """Per street, the most houses that could still be written there.

        Each maximal run of empty boxes is bounded by the nearest number to its
        left and right.  A run of ``L`` boxes between bounds ``low`` and ``high``
        can hold at most ``min(L, high - low - 1)`` houses, because what goes in
        must strictly ascend — a four-box gap between a 5 and a 7 fits one house,
        not four.

        This is the resource a careless write burns, and it is why a uniformly
        random player is not a meaningful baseline in this game: writing 15 into
        the first box of a street destroys nine placements at once.  It is
        exposed as a feature and used by :class:`~games.welcome_to.bots.GreedyBot`.
        """
        return [
            _row_capacity(self.numbers[x], size)
            for x, size in enumerate(STREET_SIZES)
        ]

    def total_span(self) -> int:
        """Sum of :meth:`box_spans` — remaining freedom, which does not saturate.

        :meth:`placement_capacity` counts how many houses still *fit*, and
        saturates: a four-box gap between a 5 and a 17 has the same capacity as
        the same gap between a 5 and a 6 once the gap is short.  Total span keeps
        discriminating, and measurement says the two are complementary rather than
        redundant — adding this term to a capacity-greedy baseline is worth about
        eight points a game.
        """
        return sum(sum(row) for row in self.box_spans())

    # ------------------------------------------------------------------
    # Hypotheticals: what a roundabout or a bis could still reach
    #
    # Every one of these exists because a house can reach a box that no *drawn
    # number* can.  `build_roundabout` writes the sentinel and
    # `available_locations(None)` ignores numeric fit; `bis_candidates` has no
    # ascending-order check at all.  Four review rounds produced unsound
    # feasibility tests by reasoning about numbers and forgetting both.
    # ------------------------------------------------------------------
    def span_if_roundabout(self, available: bool = True) -> list[list[int]]:
        """:meth:`box_spans`, but allowing one roundabout elsewhere in the street.

        A roundabout is written into its box but is skipped when
        :meth:`box_spans` looks for a bounding number, so placing one *between*
        an empty box and the number hemming it in removes that bound entirely.
        This is what makes the deferred-roundabout line representable: a box the
        plain span calls dead is alive for the price of a roundabout.

        Only one roundabout may be placed, so a box can have its left bound or
        its right bound removed, never both.  ``available=False`` returns
        :meth:`box_spans` unchanged -- pass it when the variant has no
        roundabouts or both are already spent, so the difference against
        :meth:`box_spans` never reports option value for an option that is gone.
        """
        spans = self.box_spans()
        if not available or not self.can_build_roundabout():
            return spans

        for x, size in enumerate(STREET_SIZES):
            row = self.numbers[x]
            for y in range(size):
                if row[y] is not None:
                    continue
                bounds = self.gap_bounds(x, y)
                assert bounds is not None
                first, last, low, high = bounds
                best = spans[x][y]
                # A roundabout anywhere in [first, y-1] is the first written box
                # seen scanning left, and it carries no number -- so the left
                # bound falls away.
                if y > first:
                    best = max(best, max(0, high - (MIN_NUMBER - 1) - 1))
                if y < last:
                    best = max(best, max(0, (MAX_NUMBER + 1) - low - 1))
                spans[x][y] = best
        return spans

    def capacity_if_roundabout(self, available: bool = True) -> list[int]:
        """Per street, the best :meth:`placement_capacity` one roundabout can buy.

        The roundabout *consumes* the box it sits in, so the returned figure
        counts one fewer empty box than the street currently holds.  Any caller
        comparing this against a count of empties must compare against the
        hypothetical sheet's own count, not the current one.
        """
        capacity = self.placement_capacity()
        if not available or not self.can_build_roundabout():
            return capacity

        for x, size in enumerate(STREET_SIZES):
            row = self.numbers[x]
            best = capacity[x]
            for r in range(size):
                if row[r] is not None:
                    continue
                hypothetical = list(row)
                hypothetical[r] = ROUNDABOUT
                best = max(best, _row_capacity(hypothetical, size))
            capacity[x] = best
        return capacity

    def bis_reachable(self, x: int, y: int) -> bool:
        """Could ``(x, y)`` **ever** be filled by a bis?  A sound over-estimate.

        Deliberately permissive, because the only caller is a feasibility test
        that may declare death solely when even the over-estimate says no:

        * an *empty* neighbour counts, since it may be written later and a bis
          copies whatever number ends up there;
        * an unfenced slot counts, since fences are only ever added, never
          removed, so assuming it stays open can only over-count.

        A roundabout neighbour never counts -- ``bis_candidates`` refuses to
        duplicate one -- and neither does a slot that is already fenced.

        ⚠ **There is deliberately no ``bis_marks`` test.** The bis track
        saturates, it does not gate: ``legal_actions`` offers ``bis_candidates``
        at ``ACTION_BIS`` without reading ``bis_marks``, and the apply path uses
        ``min(bis_marks + 1, BIS_BOXES)``.  Past nine marks a bis is free of its
        own penalty and still writes a house, exactly like the temp agency past
        eleven.  Capping here would under-count reachability and make every
        caller's death test unsound.
        """
        if self.numbers[x][y] is not None:
            return False
        size = STREET_SIZES[x]
        if y > 0 and not self.fences[x][y - 1]:
            if self.numbers[x][y - 1] != ROUNDABOUT:
                return True
        if y < size - 1 and not self.fences[x][y]:
            if self.numbers[x][y + 1] != ROUNDABOUT:
                return True
        return False

    def bis_reach(self) -> list[int]:
        """Per street, an upper bound on boxes a bis could still fill.

        The only bound is the supply of reachable boxes.  ``BIS_BOXES`` does
        **not** enter it -- see :meth:`bis_reachable` -- and an earlier draft
        that took ``min(BIS_BOXES - bis_marks, ...)`` was unsound.
        """
        return [
            sum(1 for y in range(size) if self.bis_reachable(x, y))
            for x, size in enumerate(STREET_SIZES)
        ]

    def gap_bounds(self, x: int, y: int) -> Optional[tuple[int, int, int, int]]:
        """``(first box, last box, bounding low number, bounding high number)``.

        The maximal run of empty boxes containing ``(x, y)``, and the numbers
        that hem it in.  A roundabout or the end of the street removes a bound.
        ``None`` if the box is already written.
        """
        row = self.numbers[x]
        size = len(row)
        if row[y] is not None:
            return None
        first = y
        while first > 0 and row[first - 1] is None:
            first -= 1
        last = y
        while last < size - 1 and row[last + 1] is None:
            last += 1

        low = MIN_NUMBER - 1
        if first > 0 and row[first - 1] is not None and row[first - 1] != ROUNDABOUT:
            low = row[first - 1]
        high = MAX_NUMBER + 1
        if last < size - 1 and row[last + 1] is not None and row[last + 1] != ROUNDABOUT:
            high = row[last + 1]
        return first, last, low, high

    def positional_fit(self, number: int, x: int, y: int) -> Optional[float]:
        """How well ``number`` belongs at ``(x, y)``: ``0.0`` is a perfect fit.

        Numbers must ascend, so a number's natural home is proportional to where
        it sits between the two numbers bounding its gap — a 15 belongs at the
        right-hand end, a 2 at the left.  Writing a high number early on the left
        is what destroys a street, and this is the cheap measure of that mistake.

        Returns the negated distance from the ideal box (higher is better), or
        ``None`` if the box is written or its gap can take nothing.
        """
        bounds = self.gap_bounds(x, y)
        if bounds is None:
            return None
        first, last, low, high = bounds
        if high - low <= 1:
            return None
        ideal = first + (last - first) * (number - low) / (high - low)
        return -abs(y - ideal)

    def bis_candidates(self) -> list[tuple[int, int, int, int]]:
        """``Houses::getAvailableLocationsForBis``.

        Returns ``(x, y, number, side)`` where ``side`` is 0 when the number is
        copied from the neighbour on the left and 1 from the right.  A bis must
        sit directly next to the house it duplicates with no estate fence in
        between, and a roundabout can never be duplicated.

        Both sides can be legal at the same box with two different numbers,
        which is why ``side`` and not ``number`` is what the action codec
        stores.
        """
        out: list[tuple[int, int, int, int]] = []
        for x, size in enumerate(STREET_SIZES):
            row = self.numbers[x]
            fence = self.fences[x]
            for y in range(size):
                if row[y] is not None:
                    continue
                if y > 0:
                    left = row[y - 1]
                    if left is not None and left != ROUNDABOUT and not fence[y - 1]:
                        out.append((x, y, left, 0))
                if y < size - 1:
                    right = row[y + 1]
                    if right is not None and right != ROUNDABOUT and not fence[y]:
                        out.append((x, y, right, 1))
        return out

    def bis_number_at(self, x: int, y: int, side: int) -> Optional[int]:
        """The number a bis at ``(x, y)`` copied from ``side`` would be, else ``None``."""
        row = self.numbers[x]
        fence = self.fences[x]
        if row[y] is not None:
            return None
        if side == 0:
            if y == 0 or fence[y - 1]:
                return None
            left = row[y - 1]
            return None if left is None or left == ROUNDABOUT else left
        if y == len(row) - 1 or fence[y]:
            return None
        right = row[y + 1]
        return None if right is None or right == ROUNDABOUT else right

    def write(self, number: int, pos: Pos, turn: int, is_bis: bool = False) -> None:
        """``Houses::add`` -- unchecked; callers validate first."""
        x, y = pos
        self.numbers[x][y] = number
        self.is_bis[x][y] = is_bis
        self.written_turn[x][y] = turn

    # ------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------
    def surveyor_zones(self) -> list[Pos]:
        """``Actions/Surveyor::getAvailableZones``.

        A fence may go in any empty fence slot, except that it may not split two
        houses already spent on the same City Plan, and it may not split a bis
        pair (two neighbours showing the same number).
        """
        out: list[Pos] = []
        for x in range(NUM_STREETS):
            row = self.numbers[x]
            for j in range(FENCE_SIZES[x]):
                if self.fences[x][j]:
                    continue
                if row[j] is not None:
                    if self.top_fences[x][j] and self.top_fences[x][j + 1]:
                        continue
                    if row[j + 1] is not None and row[j] == row[j + 1]:
                        continue
                out.append((x, j))
        return out

    def park_streets(self) -> list[int]:
        """Streets that still have an unbuilt park (``Actions/Park``)."""
        return [x for x in range(NUM_STREETS) if self.parks[x] < PARK_BOXES[x]]

    def estate_rows(self) -> list[int]:
        """Estate-value rows that still have a box to cross (``Actions/RealEstate``)."""
        return [i for i in range(6) if self.estate_marks[i] < ESTATE_ROW_BOXES[i]]

    def can_build_pool_at(self, pos: Pos) -> bool:
        """``Actions/Pool::canBuild`` -- the house just written must be on a pool."""
        return pos in POOL_POSITION_SET and self.pool_count < POOL_BOXES

    @property
    def pool_count(self) -> int:
        return sum(self.pools)

    def can_take_permit(self) -> bool:
        return self.permits < PERMIT_BOXES

    def can_build_roundabout(self) -> bool:
        return self.roundabouts < ROUNDABOUT_BOXES

    def build_roundabout(self, pos: Pos, turn: int) -> None:
        """``WriteNumberTrait::buildRoundabout``.

        A roundabout writes the sentinel into the box, fences both of its sides
        (silently skipping a side that falls off the end of the street) and
        crosses off one of the two roundabout penalty boxes.
        """
        x, y = pos
        self.write(ROUNDABOUT, pos, turn)
        for j in (y - 1, y):
            if 0 <= j < FENCE_SIZES[x]:
                self.fences[x][j] = True
        self.roundabouts = min(self.roundabouts + 1, ROUNDABOUT_BOXES)

    # ------------------------------------------------------------------
    # Housing estates
    # ------------------------------------------------------------------
    def estates(self) -> list[Estate]:
        """``Actions/RealEstate::getEstates``.

        An estate is a run of boxes bounded by fences (or the ends of the
        street) in which every box is built and none of them is a roundabout.
        """
        out: list[Estate] = []
        for x, size in enumerate(STREET_SIZES):
            row = self.numbers[x]
            start = 0
            full = True
            for y in range(size):
                n = row[y]
                if n is None or n == ROUNDABOUT:
                    full = False
                if y == size - 1 or self.fences[x][y]:
                    if full:
                        out.append((x, start, y - start + 1))
                    full = True
                    start = y + 1
        return out

    def free_estates(self) -> list[Estate]:
        """Estates no City Plan has consumed (``EstatePlan::getAvailableEstates``)."""
        out: list[Estate] = []
        for x, start, size in self.estates():
            if not any(self.top_fences[x][start + k] for k in range(size)):
                out.append((x, start, size))
        return out

    def estate_size_counts(self) -> list[int]:
        """``RealEstate::getAssocSizeNumber`` -- estates of size 1..6 (bigger ignored)."""
        mult = [0] * MAX_ESTATE_SIZE
        for _, _, size in self.estates():
            if size <= MAX_ESTATE_SIZE:
                mult[size - 1] += 1
        return mult

    def free_estate_size_counts(self) -> list[int]:
        """:meth:`estate_size_counts` over :meth:`free_estates` only.

        The two are different questions and the encoder needs both.
        :meth:`estate_size_counts` is the **scoring** multiset -- a top fence
        stops a City Plan re-using an estate, it does not remove its real-estate
        points, so :meth:`estate_score` counts every estate.  This one is the
        **plan-eligibility** multiset, which is what ``EstatePlan::canBeScored``
        matches against.
        """
        mult = [0] * MAX_ESTATE_SIZE
        for _, _, size in self.free_estates():
            if size <= MAX_ESTATE_SIZE:
                mult[size - 1] += 1
        return mult

    def mark_top_fences(self, cells: Iterable[Pos]) -> None:
        for x, y in cells:
            self.top_fences[x][y] = True

    def bis_count_per_street(self) -> list[int]:
        counts = [0, 0, 0]
        for x, size in enumerate(STREET_SIZES):
            counts[x] = sum(1 for y in range(size) if self.is_bis[x][y])
        return counts

    def has_roundabout_in_street(self, x: int) -> bool:
        return any(n == ROUNDABOUT for n in self.numbers[x])

    def street_pools_complete(self) -> list[bool]:
        """``Actions/Pool::getCompleted`` -- all three pools of a street built."""
        return [self.pools[x] == 3 for x in range(NUM_STREETS)]

    def street_parks_complete(self) -> list[bool]:
        return [self.parks[x] == PARK_BOXES[x] for x in range(NUM_STREETS)]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def park_score(self) -> int:
        return sum(PARK_SCORES[x][self.parks[x]] for x in range(NUM_STREETS))

    def pool_score(self) -> int:
        return POOL_SCORES[self.pool_count]

    def estate_score(self) -> int:
        mult = self.estate_size_counts()
        return sum(
            mult[i] * ESTATE_ROW_SCORES[i][self.estate_marks[i]]
            for i in range(MAX_ESTATE_SIZE)
        )

    def bis_penalty(self) -> int:
        return BIS_SCORES[min(self.bis_marks, BIS_BOXES)]

    def permit_penalty(self) -> int:
        return PERMIT_SCORES[min(self.permits, PERMIT_BOXES)]

    def roundabout_penalty(self) -> int:
        return ROUNDABOUT_SCORES[min(self.roundabouts, ROUNDABOUT_BOXES)]

    def local_score(self) -> SheetScore:
        """Everything except City Plans and the temp-agency rank."""
        return SheetScore(
            parks=self.park_score(),
            pools=self.pool_score(),
            estates=self.estate_score(),
            bis=self.bis_penalty(),
            permits=self.permit_penalty(),
            roundabouts=self.roundabout_penalty(),
        )

    def tiebreak_key(self) -> tuple[int, ...]:
        """``EndOfGameTrait::stComputeScores``.

        Ties go to the most completed estates, then the most size-1 estates,
        then size-2, and so on.  Larger tuple wins.
        """
        estates = self.estates()
        counts = [0] * MAX_ESTATE_SIZE
        for _, _, size in estates:
            if 1 <= size <= MAX_ESTATE_SIZE:
                counts[size - 1] += 1
        return (len(estates), *counts)

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------
    def pretty(self) -> str:
        lines = []
        for x, size in enumerate(STREET_SIZES):
            cells = []
            for y in range(size):
                n = self.numbers[x][y]
                if n is None:
                    text = " ." if not self.top_fences[x][y] else " ^"
                elif n == ROUNDABOUT:
                    text = " O"
                else:
                    text = f"{n:>2}"
                mark = "*" if self.is_bis[x][y] else ("^" if self.top_fences[x][y] and n is not None else " ")
                cells.append(text + mark)
                if y < FENCE_SIZES[x]:
                    cells.append("|" if self.fences[x][y] else " ")
            lines.append("".join(cells))
        lines.append(
            f"parks={self.parks} pools={self.pools} temps={self.temps} "
            f"bis={self.bis_marks} permits={self.permits} "
            f"roundabouts={self.roundabouts} estate_marks={self.estate_marks}"
        )
        return "\n".join(lines)
