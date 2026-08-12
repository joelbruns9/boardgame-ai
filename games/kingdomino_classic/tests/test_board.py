from __future__ import annotations

import pytest

from games.kingdomino_classic.board import ClassicBoard, Placement
from games.kingdomino_classic.dominoes import DOMINOES, Terrain


def test_first_domino_must_touch_the_castle() -> None:
    board = ClassicBoard()

    adjacent = Placement(1, 0, 2, 0)
    detached = Placement(2, 0, 3, 0)

    assert board.is_legal_placement(DOMINOES[1], adjacent)
    assert not board.is_legal_placement(DOMINOES[1], detached)


def test_matching_terrain_connection_and_scoring() -> None:
    board = ClassicBoard()
    board.place(DOMINOES[1], Placement(1, 0, 2, 0))
    board.place(DOMINOES[19], Placement(1, 1, 2, 1))

    score = board.score(harmony=False, middle_kingdom=False)

    assert score.territory_score == 3
    assert score.largest_territory_size == 3
    assert score.total_crowns == 1
    assert score.total == 3


def test_placement_cannot_expand_beyond_five_cells() -> None:
    board = ClassicBoard()
    board.place(DOMINOES[1], Placement(1, 0, 2, 0))
    board.place(DOMINOES[2], Placement(3, 0, 4, 0))

    beyond_grid = Placement(5, 0, 6, 0)

    assert not board.is_legal_placement(DOMINOES[13], beyond_grid)


def test_physically_identical_halves_do_not_duplicate_moves() -> None:
    board = ClassicBoard()
    placements = board.legal_placements(DOMINOES[1])
    assignments = set()
    for placement in placements:
        cells = tuple(sorted(placement.cells))
        assert cells not in assignments
        assignments.add(cells)


def test_place_rejects_a_domino_that_is_already_on_the_board() -> None:
    board = ClassicBoard()
    board.place(DOMINOES[1], Placement(1, 0, 2, 0))

    with pytest.raises(ValueError, match="Illegal placement"):
        board.place(DOMINOES[1], Placement(-1, 0, -2, 0))


def test_castle_cell_is_stable() -> None:
    board = ClassicBoard()

    castle = board.cell_at((0, 0))

    assert castle is not None
    assert castle.terrain == Terrain.CASTLE
    board.assert_invariants()
