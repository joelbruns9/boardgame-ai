"""Readable 5x5 classic Kingdomino board and scoring oracle.

Placement follows the same three checks as the BGA implementation's
``dominoFitsInPosition``: both cells are empty, at least one half connects to
the castle or matching terrain, and the resulting kingdom fits in a 5x5 box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import ClassicGameConfig
from .dominoes import Domino, HalfTile, Terrain

Coord = tuple[int, int]

_DIRECTIONS: tuple[Coord, ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))


@dataclass(frozen=True, order=True, slots=True)
class Placement:
    """The two occupied cells and orientation of a placed domino."""

    x1: int
    y1: int
    x2: int
    y2: int
    flipped: bool = False

    @property
    def cells(self) -> tuple[Coord, Coord]:
        return (self.x1, self.y1), (self.x2, self.y2)


@dataclass(frozen=True, slots=True)
class Cell:
    terrain: Terrain
    crowns: int
    domino_id: int | None


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    territory_score: int
    harmony_bonus: int
    middle_kingdom_bonus: int
    largest_territory_size: int
    total_crowns: int

    @property
    def total(self) -> int:
        return (
            self.territory_score
            + self.harmony_bonus
            + self.middle_kingdom_bonus
        )

    @property
    def tiebreak_key(self) -> tuple[int, int, int]:
        """BGA ranking order: score, largest territory, then crowns."""

        return self.total, self.largest_territory_size, self.total_crowns


class ClassicBoard:
    """Sparse, castle-centred board constrained by a 5x5 bounding box."""

    def __init__(self) -> None:
        self._cells: dict[Coord, Cell] = {
            (0, 0): Cell(Terrain.CASTLE, crowns=0, domino_id=None)
        }
        self._placed_domino_ids: set[int] = set()

    def copy(self) -> "ClassicBoard":
        board = ClassicBoard()
        board._cells = dict(self._cells)
        board._placed_domino_ids = set(self._placed_domino_ids)
        return board

    @property
    def placed_domino_ids(self) -> frozenset[int]:
        return frozenset(self._placed_domino_ids)

    @property
    def placed_domino_count(self) -> int:
        return len(self._placed_domino_ids)

    @property
    def occupied_count(self) -> int:
        return len(self._cells)

    def cell_at(self, coord: Coord) -> Cell | None:
        return self._cells.get(coord)

    def occupied_cells(self) -> tuple[Coord, ...]:
        return tuple(sorted(self._cells, key=lambda coord: (coord[1], coord[0])))

    def occupied_bbox(
        self, extra: Iterable[Coord] = ()
    ) -> tuple[int, int, int, int]:
        coords = [*self._cells, *extra]
        xs = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        return min(xs), min(ys), max(xs), max(ys)

    def bbox_fits(self, extra: Iterable[Coord] = ()) -> bool:
        min_x, min_y, max_x, max_y = self.occupied_bbox(extra)
        size = ClassicGameConfig.board_size
        return max_x - min_x + 1 <= size and max_y - min_y + 1 <= size

    def _half_connects(self, coord: Coord, half: HalfTile) -> bool:
        x, y = coord
        for dx, dy in _DIRECTIONS:
            neighbour = self._cells.get((x + dx, y + dy))
            if neighbour is not None and (
                neighbour.terrain == Terrain.CASTLE
                or neighbour.terrain == half.terrain
            ):
                return True
        return False

    @staticmethod
    def _halves(
        domino: Domino, placement: Placement
    ) -> tuple[HalfTile, HalfTile]:
        if placement.flipped:
            return domino.b, domino.a
        return domino.a, domino.b

    def is_legal_placement(self, domino: Domino, placement: Placement) -> bool:
        first, second = placement.cells
        if domino.id in self._placed_domino_ids:
            return False
        if abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1:
            return False
        if first in self._cells or second in self._cells:
            return False
        if not self.bbox_fits((first, second)):
            return False

        first_half, second_half = self._halves(domino, placement)
        return self._half_connects(first, first_half) or self._half_connects(
            second, second_half
        )

    def _frontier(self) -> set[Coord]:
        frontier: set[Coord] = set()
        for x, y in self._cells:
            for dx, dy in _DIRECTIONS:
                coord = (x + dx, y + dy)
                if coord not in self._cells:
                    frontier.add(coord)
        return frontier

    def legal_placements(self, domino: Domino) -> list[Placement]:
        """Return every physically distinct legal placement deterministically."""

        placements: list[Placement] = []
        seen_assignments: set[tuple[tuple[int, int, int, int], ...]] = set()
        for x1, y1 in sorted(self._frontier(), key=lambda c: (c[1], c[0])):
            for dx, dy in _DIRECTIONS:
                x2, y2 = x1 + dx, y1 + dy
                if (x2, y2) in self._cells:
                    continue
                for flipped in (False, True):
                    placement = Placement(x1, y1, x2, y2, flipped)
                    if not self.is_legal_placement(domino, placement):
                        continue
                    first_half, second_half = self._halves(domino, placement)
                    assignment = tuple(
                        sorted(
                            (
                                (x1, y1, int(first_half.terrain), first_half.crowns),
                                (x2, y2, int(second_half.terrain), second_half.crowns),
                            )
                        )
                    )
                    if assignment in seen_assignments:
                        continue
                    seen_assignments.add(assignment)
                    placements.append(placement)
        placements.sort()
        return placements

    def place(self, domino: Domino, placement: Placement) -> None:
        if not self.is_legal_placement(domino, placement):
            raise ValueError(f"Illegal placement for domino {domino.id}: {placement}")
        first_half, second_half = self._halves(domino, placement)
        for coord, half in zip(placement.cells, (first_half, second_half)):
            self._cells[coord] = Cell(half.terrain, half.crowns, domino.id)
        self._placed_domino_ids.add(domino.id)

    def score(
        self, *, harmony: bool = True, middle_kingdom: bool = True
    ) -> ScoreBreakdown:
        visited: set[Coord] = set()
        territory_score = 0
        largest_territory_size = 0
        total_crowns = 0

        for start, cell in self._cells.items():
            if start in visited or cell.terrain == Terrain.CASTLE:
                continue
            stack = [start]
            visited.add(start)
            size = 0
            crowns = 0
            while stack:
                x, y = stack.pop()
                current = self._cells[(x, y)]
                size += 1
                crowns += current.crowns
                for dx, dy in _DIRECTIONS:
                    neighbour_coord = (x + dx, y + dy)
                    neighbour = self._cells.get(neighbour_coord)
                    if (
                        neighbour_coord not in visited
                        and neighbour is not None
                        and neighbour.terrain == cell.terrain
                    ):
                        visited.add(neighbour_coord)
                        stack.append(neighbour_coord)
            territory_score += size * crowns
            largest_territory_size = max(largest_territory_size, size)
            total_crowns += crowns

        harmony_bonus = (
            5
            if harmony
            and self.placed_domino_count == ClassicGameConfig.dominoes_per_player
            and self.occupied_count == ClassicGameConfig.board_size**2
            else 0
        )

        min_x, min_y, max_x, max_y = self.occupied_bbox()
        middle_bonus = (
            10
            if middle_kingdom and -min_x == max_x and -min_y == max_y
            else 0
        )
        return ScoreBreakdown(
            territory_score=territory_score,
            harmony_bonus=harmony_bonus,
            middle_kingdom_bonus=middle_bonus,
            largest_territory_size=largest_territory_size,
            total_crowns=total_crowns,
        )

    def state_key(self) -> tuple[tuple[int, int, int, int, int | None], ...]:
        return tuple(
            (
                x,
                y,
                int(cell.terrain),
                cell.crowns,
                cell.domino_id,
            )
            for (x, y), cell in sorted(self._cells.items())
        )

    def assert_invariants(self) -> None:
        castle = self._cells.get((0, 0))
        if castle != Cell(Terrain.CASTLE, crowns=0, domino_id=None):
            raise AssertionError("The castle must remain at (0, 0).")
        if not self.bbox_fits():
            raise AssertionError("The occupied kingdom exceeds its 5x5 bounding box.")

        counts: dict[int, int] = {}
        for cell in self._cells.values():
            if cell.domino_id is not None:
                counts[cell.domino_id] = counts.get(cell.domino_id, 0) + 1
        if set(counts) != self._placed_domino_ids:
            raise AssertionError("Placed domino index disagrees with occupied cells.")
        if any(count != 2 for count in counts.values()):
            raise AssertionError("Every placed domino must occupy exactly two cells.")
