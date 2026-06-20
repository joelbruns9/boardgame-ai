from games.kingdomino.board import Board, Placement
from games.kingdomino.dominoes import DOMINOES


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_castle_can_be_corner_of_7x7():
    board = Board()
    castle = board.castle_pos

    # Fill a 7x7 footprint with castle at one corner.
    for x in range(castle[0], castle[0] + 7):
        for y in range(castle[1], castle[1] + 7):
            if (x, y) != castle:
                board.terrain[x, y] = 2  # WHEAT
                board.crowns[x, y] = 0
                board.domino_id[x, y] = 99

    assert_true(board.bbox_fits(), f"Expected 7x7 bbox to fit, got {board.occupied_bbox()}")


def test_bounding_box_rejects_8_wide():
    board = Board()
    castle = board.castle_pos

    board.terrain[castle[0] + 7, castle[1]] = 2  # WHEAT, creates 8-wide bbox
    board.domino_id[castle[0] + 7, castle[1]] = 99

    assert_true(not board.bbox_fits(), f"Expected oversized bbox to fail, got {board.occupied_bbox()}")

def test_domino_frequency_counts():
    counts = {
        "WHEAT": 0,
        "FOREST": 0,
        "WATER": 0,
        "GRASS": 0,
        "SWAMP": 0,
        "MINE": 0,
    }

    for domino in DOMINOES.values():
        counts[domino.a.terrain.name] += 1
        counts[domino.b.terrain.name] += 1

    expected = {
        "WHEAT": 26,
        "FOREST": 22,
        "WATER": 18,
        "GRASS": 14,
        "SWAMP": 10,
        "MINE": 6,
    }

    assert_true(counts == expected, f"Bad terrain counts: {counts}")

def main():
    test_domino_frequency_counts()
    test_castle_can_be_corner_of_7x7()
    test_bounding_box_rejects_8_wide()
    print("Rule tests passed")


if __name__ == "__main__":
    main()