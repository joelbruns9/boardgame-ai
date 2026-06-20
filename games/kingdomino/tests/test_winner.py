"""Tests for the authoritative winner-determination cascade (Phase 0.1).

Covers game.determine_winner and encoder.compute_target_win across every level
of the official Kingdomino tiebreaker cascade:

    1. total score
    2. largest single connected territory (tile count, crowns ignored)
    3. total crowns
    4. draw

Run via:  python -m games.kingdomino.tests.test_winner
"""
from games.kingdomino.board import Board
from games.kingdomino.dominoes import Terrain
from games.kingdomino.encoder import compute_target_win
from games.kingdomino.game import (
    GameConfig,
    GameState,
    Phase,
    determine_winner,
)


# ─── test fixtures ─────────────────────────────────────────────────────────
def put(board: Board, x: int, y: int, terrain: int, crowns: int = 0) -> None:
    """Set one occupied cell, keeping Board.score()'s bookkeeping correct.

    The public Board API only places full dominoes, but winner tests need
    arbitrary terminal shapes. This mirrors exactly the per-cell bookkeeping
    Board.place() maintains (terrain/crowns arrays plus the incremental
    occupancy/bbox tracking score() scans), so the constructed board scores
    identically to one built move-by-move.
    """
    board.terrain[y, x] = int(terrain)
    board.crowns[y, x] = crowns
    board.domino_id[y, x] = 99
    board._occupied.add((x, y))
    board._cell[(x, y)] = int(terrain)
    board._min_x = min(board._min_x, x)
    board._max_x = max(board._max_x, x)
    board._min_y = min(board._min_y, y)
    board._max_y = max(board._max_y, y)


def strip(board: Board, x0: int, y: int, length: int, terrain: int, crowns=()) -> None:
    """Place a horizontal run of one terrain — a single connected territory.

    `crowns` lists the crown count per cell (left to right); missing trailing
    entries default to 0. All cells share a terrain and a row, so they form one
    connected region of size `length`.
    """
    crowns = list(crowns) + [0] * (length - len(crowns))
    for i in range(length):
        put(board, x0 + i, y, terrain, crowns[i])


def make_terminal_state(board0: Board, board1: Board) -> GameState:
    return GameState(
        config=GameConfig(),
        boards=[board0, board1],
        deck=[],
        current_row=[],
        pending_claims=[],
        next_claims=[],
        phase=Phase.GAME_OVER,
    )


def swapped(state: GameState) -> GameState:
    """Same game with the two boards exchanged — used for symmetry checks."""
    s = state.copy()
    s.boards = [s.boards[1], s.boards[0]]
    return s


# Strips are placed at x >= 8, y >= 8 (castle is at (7, 7)), so the castle is
# always the min corner of the bbox — never centred and never part of a full
# 7x7 — guaranteeing no Harmony/Middle-Kingdom bonus distorts these scores.
WHEAT, FOREST, WATER = int(Terrain.WHEAT), int(Terrain.FOREST), int(Terrain.WATER)


def assert_eq(actual, expected, msg):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")


# ─── cascade level 1: score ────────────────────────────────────────────────
def case_score_decided():
    """P0 scores 30, P1 scores 20 → P0 wins on raw score."""
    b0 = Board()
    strip(b0, 8, 8, 5, WHEAT, crowns=[2, 1, 1, 1, 1])  # 5 tiles * 6 crowns = 30
    b1 = Board()
    strip(b1, 8, 8, 5, WHEAT, crowns=[1, 1, 1, 1, 0])  # 5 tiles * 4 crowns = 20

    sb0 = b0.score()
    sb1 = b1.score()
    assert_eq((sb0.total, sb1.total), (30, 20), "score-decided totals")
    return make_terminal_state(b0, b1)


# ─── cascade level 2: largest territory ────────────────────────────────────
def case_territory_decided():
    """Both score 20; P0 largest territory 6 tiles, P1 largest 4 → P0 wins."""
    b0 = Board()
    strip(b0, 8, 8, 6, WHEAT)                            # 6 tiles, 0 crowns -> 0
    strip(b0, 8, 10, 4, FOREST, crowns=[2, 1, 1, 1])    # 4 tiles * 5 crowns = 20
    b1 = Board()
    strip(b1, 8, 8, 4, FOREST, crowns=[2, 1, 1, 1])     # 4 tiles * 5 crowns = 20

    sb0, sb1 = b0.score(), b1.score()
    assert_eq((sb0.total, sb1.total), (20, 20), "territory-decided totals")
    assert_eq((sb0.largest_territory_size, sb1.largest_territory_size), (6, 4),
              "territory-decided largest territory")
    # crowns equal, so the decision must come from territory, not crowns
    assert_eq((sb0.total_crowns, sb1.total_crowns), (5, 5),
              "territory-decided crowns (should be tied)")
    return make_terminal_state(b0, b1)


# ─── cascade level 3: total crowns ─────────────────────────────────────────
def case_crowns_decided():
    """Both score 10, both largest territory 5; P0 has 3 crowns, P1 has 2."""
    b0 = Board()
    strip(b0, 8, 8, 5, WHEAT, crowns=[1])    # 5 * 1 = 5
    strip(b0, 8, 10, 4, FOREST, crowns=[1])  # 4 * 1 = 4
    strip(b0, 8, 12, 1, WATER, crowns=[1])   # 1 * 1 = 1   -> total 10, crowns 3
    b1 = Board()
    strip(b1, 8, 8, 5, WHEAT, crowns=[1])    # 5 * 1 = 5
    strip(b1, 8, 10, 5, FOREST, crowns=[1])  # 5 * 1 = 5   -> total 10, crowns 2

    sb0, sb1 = b0.score(), b1.score()
    assert_eq((sb0.total, sb1.total), (10, 10), "crowns-decided totals")
    assert_eq((sb0.largest_territory_size, sb1.largest_territory_size), (5, 5),
              "crowns-decided largest territory (should be tied)")
    assert_eq((sb0.total_crowns, sb1.total_crowns), (3, 2),
              "crowns-decided total crowns")
    return make_terminal_state(b0, b1)


# ─── cascade level 4: draw ─────────────────────────────────────────────────
def case_draw():
    """Identical boards: equal score, territory, crowns → draw."""
    def build():
        b = Board()
        strip(b, 8, 8, 5, WHEAT, crowns=[2, 2])  # 5 * 4 = 20, crowns 4, largest 5
        return b

    b0, b1 = build(), build()
    sb0, sb1 = b0.score(), b1.score()
    assert_eq((sb0.total, sb1.total), (20, 20), "draw totals")
    assert_eq((sb0.largest_territory_size, sb1.largest_territory_size), (5, 5),
              "draw largest territory")
    assert_eq((sb0.total_crowns, sb1.total_crowns), (4, 4), "draw crowns")
    return make_terminal_state(b0, b1)


# ─── tests ─────────────────────────────────────────────────────────────────
def test_determine_winner_cascade_levels():
    assert_eq(determine_winner(case_score_decided()), 0, "score level -> P0")
    assert_eq(determine_winner(case_territory_decided()), 0, "territory level -> P0")
    assert_eq(determine_winner(case_crowns_decided()), 0, "crowns level -> P0")
    assert_eq(determine_winner(case_draw()), None, "all tied -> draw")


def test_determine_winner_symmetry():
    """Swapping the two boards must flip the winner index for every decided
    case, and keep a draw a draw."""
    for name, builder in (("score", case_score_decided),
                          ("territory", case_territory_decided),
                          ("crowns", case_crowns_decided)):
        state = builder()
        assert_eq(determine_winner(state), 0, f"{name}: P0 wins")
        assert_eq(determine_winner(swapped(state)), 1, f"{name}: mirror -> P1 wins")

    draw = case_draw()
    assert_eq(determine_winner(swapped(draw)), None, "draw: mirror still draw")


def test_compute_target_win_values():
    win = case_score_decided()       # P0 wins
    assert_eq(compute_target_win(win, 0), 1.0, "winner target 1.0")
    assert_eq(compute_target_win(win, 1), 0.0, "loser target 0.0")

    draw = case_draw()
    assert_eq(compute_target_win(draw, 0), 0.5, "draw target P0 = 0.5")
    assert_eq(compute_target_win(draw, 1), 0.5, "draw target P1 = 0.5")

    # crowns-decided still yields a discrete win/loss
    crowns = case_crowns_decided()
    assert_eq(compute_target_win(crowns, 0), 1.0, "crowns winner target 1.0")
    assert_eq(compute_target_win(crowns, 1), 0.0, "crowns loser target 0.0")


def test_compute_target_win_requires_terminal_state():
    state = GameState.new(seed=0)  # phase = INITIAL_SELECTION, not terminal
    assert state.phase != Phase.GAME_OVER
    try:
        compute_target_win(state, 0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "compute_target_win should raise ValueError on a non-terminal state"
        )


def main():
    test_determine_winner_cascade_levels()
    test_determine_winner_symmetry()
    test_compute_target_win_values()
    test_compute_target_win_requires_terminal_state()
    print("All winner-determination tests passed")


if __name__ == "__main__":
    main()
