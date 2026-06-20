from games.kingdomino.board import Board, Placement
from games.kingdomino.dominoes import DOMINOES


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_valid_castle_connection():
    board = Board()
    cx, cy = board.castle_pos

    domino = DOMINOES[1]  # WHEAT/WHEAT
    placement = Placement(cx + 1, cy, cx + 2, cy, flipped=False)

    assert_true(
        board.is_legal_placement(domino, placement),
        "Domino adjacent to castle should be legal",
    )


def test_invalid_no_connection():
    board = Board()
    cx, cy = board.castle_pos

    domino = DOMINOES[1]
    placement = Placement(cx + 3, cy + 3, cx + 4, cy + 3, flipped=False)

    assert_true(
        not board.is_legal_placement(domino, placement),
        "Disconnected domino should be illegal",
    )


def test_invalid_overlap():
    board = Board()
    cx, cy = board.castle_pos

    domino = DOMINOES[1]
    legal = Placement(cx + 1, cy, cx + 2, cy, flipped=False)
    board.place(domino, legal)

    overlap = Placement(cx + 1, cy, cx + 1, cy + 1, flipped=False)

    assert_true(
        not board.is_legal_placement(domino, overlap),
        "Overlapping placement should be illegal",
    )


def test_valid_matching_terrain_connection():
    board = Board()
    cx, cy = board.castle_pos

    wheat = DOMINOES[1]
    board.place(wheat, Placement(cx + 1, cy, cx + 2, cy, flipped=False))

    second_wheat = DOMINOES[2]
    placement = Placement(cx + 3, cy, cx + 4, cy, flipped=False)

    assert_true(
        board.is_legal_placement(second_wheat, placement),
        "Matching terrain adjacency should be legal",
    )


def test_invalid_nonmatching_terrain_connection():
    board = Board()
    cx, cy = board.castle_pos

    wheat = DOMINOES[1]
    board.place(wheat, Placement(cx + 1, cy, cx + 2, cy, flipped=False))

    water = DOMINOES[7]
    placement = Placement(cx + 3, cy, cx + 4, cy, flipped=False)

    assert_true(
        not board.is_legal_placement(water, placement),
        "Nonmatching terrain adjacency should be illegal",
    )


def main():
    test_valid_castle_connection()
    test_invalid_no_connection()
    test_invalid_overlap()
    test_valid_matching_terrain_connection()
    test_invalid_nonmatching_terrain_connection()
    print("Placement tests passed")


if __name__ == "__main__":
    main()