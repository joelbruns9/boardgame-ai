from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Tuple


class Terrain(IntEnum):
    EMPTY = 0
    CASTLE = 1
    WHEAT = 2
    FOREST = 3
    WATER = 4
    GRASS = 5
    SWAMP = 6
    MINE = 7


@dataclass(frozen=True, slots=True)
class HalfTile:
    terrain: Terrain
    crowns: int


@dataclass(frozen=True, slots=True)
class Domino:
    id: int
    a: HalfTile
    b: HalfTile


_RAW_DOMINOES: Dict[int, Tuple[str, int, str, int]] = {
    1:  ('WHEAT', 0, 'WHEAT', 0),
    2:  ('WHEAT', 0, 'WHEAT', 0),
    3:  ('FOREST', 0, 'FOREST', 0),
    4:  ('FOREST', 0, 'FOREST', 0),
    5:  ('FOREST', 0, 'FOREST', 0),
    6:  ('FOREST', 0, 'FOREST', 0),
    7:  ('WATER', 0, 'WATER', 0),
    8:  ('WATER', 0, 'WATER', 0),
    9:  ('WATER', 0, 'WATER', 0),
    10: ('GRASS', 0, 'GRASS', 0),
    11: ('GRASS', 0, 'GRASS', 0),
    12: ('SWAMP', 0, 'SWAMP', 0),
    13: ('WHEAT', 0, 'FOREST', 0),
    14: ('WHEAT', 0, 'WATER', 0),
    15: ('WHEAT', 0, 'GRASS', 0),
    16: ('WHEAT', 0, 'SWAMP', 0),
    17: ('FOREST', 0, 'WATER', 0),
    18: ('FOREST', 0, 'GRASS', 0),
    19: ('WHEAT', 1, 'FOREST', 0),
    20: ('WHEAT', 1, 'WATER', 0),
    21: ('WHEAT', 1, 'GRASS', 0),
    22: ('WHEAT', 1, 'SWAMP', 0),
    23: ('WHEAT', 1, 'MINE', 0),
    24: ('FOREST', 1, 'WHEAT', 0),
    25: ('FOREST', 1, 'WHEAT', 0),
    26: ('FOREST', 1, 'WHEAT', 0),
    27: ('FOREST', 1, 'WHEAT', 0),
    28: ('FOREST', 1, 'WATER', 0),
    29: ('FOREST', 1, 'GRASS', 0),
    30: ('WATER', 1, 'WHEAT', 0),
    31: ('WATER', 1, 'WHEAT', 0),
    32: ('WATER', 1, 'FOREST', 0),
    33: ('WATER', 1, 'FOREST', 0),
    34: ('WATER', 1, 'FOREST', 0),
    35: ('WATER', 1, 'FOREST', 0),
    36: ('WHEAT', 0, 'GRASS', 1),
    37: ('WATER', 0, 'GRASS', 1),
    38: ('WHEAT', 0, 'SWAMP', 1),
    39: ('GRASS', 0, 'SWAMP', 1),
    40: ('MINE', 1, 'WHEAT', 0),
    41: ('WHEAT', 0, 'GRASS', 2),
    42: ('WATER', 0, 'GRASS', 2),
    43: ('WHEAT', 0, 'SWAMP', 2),
    44: ('GRASS', 0, 'SWAMP', 2),
    45: ('MINE', 2, 'WHEAT', 0),
    46: ('SWAMP', 0, 'MINE', 2),
    47: ('SWAMP', 0, 'MINE', 2),
    48: ('WHEAT', 0, 'MINE', 3),
}

DOMINOES: Dict[int, Domino] = {
    i: Domino(i, HalfTile(Terrain[t1], c1), HalfTile(Terrain[t2], c2))
    for i, (t1, c1, t2, c2) in _RAW_DOMINOES.items()
}


def terrain_frequency() -> dict[Terrain, int]:
    counts = {t: 0 for t in Terrain if t not in (Terrain.EMPTY, Terrain.CASTLE)}
    for d in DOMINOES.values():
        counts[d.a.terrain] += 1
        counts[d.b.terrain] += 1
    return counts
