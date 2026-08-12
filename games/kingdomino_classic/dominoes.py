"""Canonical base-game domino material.

The immutable catalogue is shared with the established Mighty Duel package.
Its 48 entries are equivalence-tested against the BGA material file in this
package's test suite.
"""

from games.kingdomino.dominoes import (
    DOMINOES,
    Domino,
    HalfTile,
    Terrain,
    terrain_frequency,
)

__all__ = [
    "DOMINOES",
    "Domino",
    "HalfTile",
    "Terrain",
    "terrain_frequency",
]
