"""
REFERENCE COPY — Original Board implementation before optimization.
Used by test_board_equiv.py as a correctness oracle to verify that
the optimized Board in board.py produces identical results for all
legal move generation and placement validation.

Do not use this in production code — import from board.py instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from .dominoes import Domino, HalfTile, Terrain

Coord = Tuple[int, int]

_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True, slots=True)
class Placement:
    x1: int
    y1: int
    x2: int
    y2: int
    flipped: bool = False

    @property
    def cells(self) -> tuple[Coord, Coord]:
        return (self.x1, self.y1), (self.x2, self.y2)


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    territory_score: int
    harmony_bonus: int
    middle_kingdom_bonus: int

    @property
    def total(self) -> int:
        return self.territory_score + self.harmony_bonus + self.middle_kingdom_bonus


class Board:
    """Compact floating Kingdomino board on a larger canvas.

    A 15x15 canvas keeps placement arithmetic simple while still enforcing the
    kingdom size by bounding-box checks. The castle starts at (7, 7), but the
    final occupied kingdom can be off-center within its 7x7 bounding frame.

    Performance notes vs the original implementation:
    - The set of occupied cells and the occupied bounding box are tracked
      incrementally on place(), so occupied_bbox()/bbox_fits() no longer call
      np.nonzero on every legality check.
    - legal_placements() only considers the frontier (empty cells adjacent to
      occupied cells) instead of scanning the whole canvas, and de-duplicates
      physically-identical placements.
    - score() only scans the occupied bounding box.
    Array storage convention is unchanged: [row, col] == [y, x].
    """

    def __init__(self, canvas_size: int = 15, castle_pos: Coord | None = None):
        if canvas_size < 7 or canvas_size % 2 == 0:
            raise ValueError("canvas_size should be an odd integer >= 7")
        self.canvas_size = canvas_size
        self.terrain = np.zeros((canvas_size, canvas_size), dtype=np.int8)
        self.crowns = np.zeros((canvas_size, canvas_size), dtype=np.int8)
        self.domino_id = np.zeros((canvas_size, canvas_size), dtype=np.int16)
        self.castle_pos = castle_pos or (canvas_size // 2, canvas_size // 2)
        cx, cy = self.castle_pos
        self.terrain[cy, cx] = Terrain.CASTLE
        self.domino_id[cy, cx] = -1

        # Incremental occupancy tracking (includes the castle, matching the
        # original occupied_cells() which counted CASTLE as non-empty).
        self._occupied: set[Coord] = {(cx, cy)}
        self._min_x = self._max_x = cx
        self._min_y = self._max_y = cy

    def copy(self) -> "Board":
        b = Board(self.canvas_size, self.castle_pos)
        b.terrain = self.terrain.copy()
        b.crowns = self.crowns.copy()
        b.domino_id = self.domino_id.copy()
        b._occupied = set(self._occupied)
        b._min_x, b._max_x = self._min_x, self._max_x
        b._min_y, b._max_y = self._min_y, self._max_y
        return b

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.canvas_size and 0 <= y < self.canvas_size

    def is_empty(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.terrain[y, x] == Terrain.EMPTY

    def occupied_cells(self) -> list[Coord]:
        # Deterministic row-major order to mirror the original np.nonzero order.
        return sorted(self._occupied, key=lambda c: (c[1], c[0]))

    def occupied_bbox(self, extra: Iterable[Coord] = ()) -> Optional[tuple[int, int, int, int]]:
        have = bool(self._occupied)
        if have:
            min_x, min_y = self._min_x, self._min_y
            max_x, max_y = self._max_x, self._max_y
        else:
            min_x = min_y = self.canvas_size
            max_x = max_y = -1
        for ex, ey in extra:
            have = True
            if ex < min_x:
                min_x = ex
            if ex > max_x:
                max_x = ex
            if ey < min_y:
                min_y = ey
            if ey > max_y:
                max_y = ey
        if not have:
            return None
        return min_x, min_y, max_x, max_y

    def bbox_fits(self, extra: Iterable[Coord] = (), max_size: int = 7) -> bool:
        bbox = self.occupied_bbox(extra)
        if bbox is None:
            return True
        min_x, min_y, max_x, max_y = bbox
        return (max_x - min_x + 1) <= max_size and (max_y - min_y + 1) <= max_size

    def adjacent_coords(self, x: int, y: int) -> Iterable[Coord]:
        for dx, dy in _DIRECTIONS:
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny):
                yield nx, ny

    def half_connects(self, x: int, y: int, half: HalfTile) -> bool:
        for nx, ny in self.adjacent_coords(x, y):
            t = int(self.terrain[ny, nx])
            if t == Terrain.CASTLE or t == half.terrain:
                return True
        return False

    def is_legal_placement(self, domino: Domino, placement: Placement) -> bool:
        (x1, y1), (x2, y2) = placement.cells
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return False
        if not self.is_empty(x1, y1) or not self.is_empty(x2, y2):
            return False
        if not self.bbox_fits(((x1, y1), (x2, y2)), max_size=7):
            return False

        h1, h2 = (domino.b, domino.a) if placement.flipped else (domino.a, domino.b)
        return self.half_connects(x1, y1, h1) or self.half_connects(x2, y2, h2)

    def _frontier(self) -> set[Coord]:
        frontier: set[Coord] = set()
        for ox, oy in self._occupied:
            for nx, ny in self.adjacent_coords(ox, oy):
                if self.terrain[ny, nx] == Terrain.EMPTY:
                    frontier.add((nx, ny))
        return frontier

    def legal_placements(self, domino: Domino) -> list[Placement]:
        """Generate every legal, physically-distinct placement of `domino`.

        Any legal placement must have at least one half adjacent to an occupied
        cell (castle or matching terrain), so at least one of its two cells is
        in the frontier. We therefore enumerate (frontier cell, empty neighbour)
        pairs in both half-orientations and verify with is_legal_placement.
        Placements that put the same terrain/crowns on the same cells are the
        same move and are collapsed to a single entry.
        """
        moves: list[Placement] = []
        seen: set = set()
        for fx, fy in self._frontier():
            for dx, dy in _DIRECTIONS:
                gx, gy = fx + dx, fy + dy
                if not self.is_empty(gx, gy):
                    continue
                for flipped in (False, True):
                    p = Placement(fx, fy, gx, gy, flipped)
                    if not self.is_legal_placement(domino, p):
                        continue
                    h1, h2 = (domino.b, domino.a) if flipped else (domino.a, domino.b)
                    c1 = (fx, fy, int(h1.terrain), int(h1.crowns))
                    c2 = (gx, gy, int(h2.terrain), int(h2.crowns))
                    key = (c1, c2) if c1 <= c2 else (c2, c1)
                    if key in seen:
                        continue
                    seen.add(key)
                    moves.append(p)
        return moves

    def place(self, domino: Domino, placement: Placement) -> None:
        if not self.is_legal_placement(domino, placement):
            raise ValueError(f"Illegal placement for domino {domino.id}: {placement}")
        h1, h2 = (domino.b, domino.a) if placement.flipped else (domino.a, domino.b)
        for (x, y), h in zip(placement.cells, (h1, h2)):
            self.terrain[y, x] = h.terrain
            self.crowns[y, x] = h.crowns
            self.domino_id[y, x] = domino.id
            self._occupied.add((x, y))
            if x < self._min_x:
                self._min_x = x
            if x > self._max_x:
                self._max_x = x
            if y < self._min_y:
                self._min_y = y
            if y > self._max_y:
                self._max_y = y

    def score(self, harmony: bool = True, middle_kingdom: bool = True) -> ScoreBreakdown:
        visited = np.zeros_like(self.terrain, dtype=bool)
        territory_score = 0
        min_x, min_y, max_x, max_y = self._min_x, self._min_y, self._max_x, self._max_y
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                t = int(self.terrain[y, x])
                if visited[y, x] or t in (Terrain.EMPTY, Terrain.CASTLE):
                    continue
                stack = [(x, y)]
                visited[y, x] = True
                area = 0
                crowns = 0
                while stack:
                    cx, cy = stack.pop()
                    area += 1
                    crowns += int(self.crowns[cy, cx])
                    for nx, ny in self.adjacent_coords(cx, cy):
                        if not visited[ny, nx] and int(self.terrain[ny, nx]) == t:
                            visited[ny, nx] = True
                            stack.append((nx, ny))
                territory_score += area * crowns

        harmony_bonus = 0
        middle_bonus = 0
        if self._occupied:
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            occupied = len(self._occupied)
            if harmony and width == 7 and height == 7 and occupied == 49:
                harmony_bonus = 5
            if middle_kingdom:
                cx, cy = self.castle_pos
                if width == 7 and height == 7 and (cx, cy) == (min_x + 3, min_y + 3):
                    middle_bonus = 10
        return ScoreBreakdown(territory_score, harmony_bonus, middle_bonus)

    def pretty(self) -> str:
        symbols = {
            Terrain.EMPTY: " . ", Terrain.CASTLE: " K ", Terrain.WHEAT: "Wh", Terrain.FOREST: "Fo",
            Terrain.WATER: "Wa", Terrain.GRASS: "Gr", Terrain.SWAMP: "Sw", Terrain.MINE: "Mi",
        }
        bbox = self.occupied_bbox()
        if bbox is None:
            xs = ys = range(self.canvas_size)
        else:
            min_x, min_y, max_x, max_y = bbox
            pad = 1
            xs = range(max(0, min_x - pad), min(self.canvas_size, max_x + pad + 1))
            ys = range(max(0, min_y - pad), min(self.canvas_size, max_y + pad + 1))
        lines = []
        for y in ys:
            row = []
            for x in xs:
                t = Terrain(int(self.terrain[y, x]))
                c = int(self.crowns[y, x])
                cell = symbols[t]
                row.append(f"{cell}{c}" if t not in (Terrain.EMPTY, Terrain.CASTLE) else f"{cell} ")
            lines.append(" ".join(row))
        return "\n".join(lines)
