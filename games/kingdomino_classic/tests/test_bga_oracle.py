from __future__ import annotations

from pathlib import Path
import re

from games.kingdomino_classic.dominoes import DOMINOES, Terrain


_BGA_TERRAIN = {
    "field": Terrain.WHEAT,
    "forest": Terrain.FOREST,
    "lake": Terrain.WATER,
    "grassland": Terrain.GRASS,
    "swamp": Terrain.SWAMP,
    "mountain": Terrain.MINE,
}


def test_domino_catalogue_matches_checked_in_bga_material() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    material_path = repo_root / "BGA Files" / "kingdomino" / "material.inc.php"
    material = material_path.read_text(encoding="utf-8")
    entry_pattern = re.compile(
        r"(?P<id>\d+)\s*=>\s*array\(\s*"
        r'"left"\s*=>\s*array\("terrain"\s*=>\s*"(?P<left>\w+)",\s*'
        r'"crowns"\s*=>\s*(?P<left_crowns>\d+)\),\s*'
        r'"right"\s*=>\s*array\("terrain"\s*=>\s*"(?P<right>\w+)",\s*'
        r'"crowns"\s*=>\s*(?P<right_crowns>\d+)\)\)',
        re.MULTILINE,
    )

    parsed = {
        int(match.group("id")): (
            _BGA_TERRAIN[match.group("left")],
            int(match.group("left_crowns")),
            _BGA_TERRAIN[match.group("right")],
            int(match.group("right_crowns")),
        )
        for match in entry_pattern.finditer(material)
        if int(match.group("id")) <= 48
    }
    canonical = {
        domino_id: (
            domino.a.terrain,
            domino.a.crowns,
            domino.b.terrain,
            domino.b.crowns,
        )
        for domino_id, domino in DOMINOES.items()
    }

    assert parsed == canonical
