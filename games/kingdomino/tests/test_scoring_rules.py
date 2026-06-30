from games.kingdomino.board import Board


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected={expected}, actual={actual}")


def set_cell(board, x, y, terrain, crowns=0, domino_id=99):
    board.terrain[y, x] = terrain
    board.crowns[y, x] = crowns
    board.domino_id[y, x] = domino_id
    board._occupied.add((x, y))
    board._cell[(x, y)] = int(terrain)
    board._min_x = min(board._min_x, x)
    board._max_x = max(board._max_x, x)
    board._min_y = min(board._min_y, y)
    board._max_y = max(board._max_y, y)


def test_single_region_score():
    board = Board()
    cx, cy = board.castle_pos

    # 3 wheat cells connected, 2 total crowns = 6 points
    set_cell(board, cx + 1, cy, 2, crowns=1)
    set_cell(board, cx + 2, cy, 2, crowns=1)
    set_cell(board, cx + 3, cy, 2, crowns=0)

    score = board.score()
    assert_equal(score.territory_score, 6, "Single region score failed")


def test_separate_regions_same_terrain():
    board = Board()
    cx, cy = board.castle_pos

    # Region 1: 2 wheat cells, 1 crown = 2
    set_cell(board, cx + 1, cy, 2, crowns=1)
    set_cell(board, cx + 2, cy, 2, crowns=0)

    # Region 2: separate wheat cell, 2 crowns = 2
    set_cell(board, cx, cy + 2, 2, crowns=2)

    score = board.score()
    assert_equal(score.territory_score, 4, "Separate same-terrain regions failed")


def test_harmony_bonus():
    board = Board()
    cx, cy = board.castle_pos

    # Fill exact 7x7 kingdom around castle.
    for x in range(cx - 3, cx + 4):
        for y in range(cy - 3, cy + 4):
            if (x, y) != board.castle_pos:
                set_cell(board, x, y, 2, crowns=0)

    score = board.score()
    assert_equal(score.harmony_bonus, 5, "Harmony bonus failed")


def test_middle_kingdom_bonus():
    board = Board()
    cx, cy = board.castle_pos

    # Occupied bbox from cx-3..cx+3 and cy-3..cy+3 makes castle centered.
    set_cell(board, cx - 3, cy - 3, 2)
    set_cell(board, cx + 3, cy + 3, 2)

    score = board.score()
    assert_equal(score.middle_kingdom_bonus, 10, "Middle Kingdom bonus failed")


def test_no_middle_kingdom_when_castle_off_center():
    board = Board()
    cx, cy = board.castle_pos

    # Bbox makes castle corner-ish, not centered.
    set_cell(board, cx + 6, cy + 6, 2)

    score = board.score()
    assert_equal(score.middle_kingdom_bonus, 0, "Off-center castle should not get Middle Kingdom")


def main():
    test_single_region_score()
    test_separate_regions_same_terrain()
    test_harmony_bonus()
    test_middle_kingdom_bonus()
    test_no_middle_kingdom_when_castle_off_center()
    print("Scoring tests passed")


if __name__ == "__main__":
    main()
