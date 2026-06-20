import math
from dataclasses import dataclass

from games.kingdomino.dominoes import DOMINOES


# NOTE ON INDEXING CONVENTION
# ---------------------------
# Board stores numpy arrays in [row, col] == [y, x] order
# (see board.py: self.terrain[cy, cx], self.terrain[y, x], ...).
# However Board.occupied_cells() returns (x, y) tuples.
# Therefore any DIRECT array read must use board.terrain[y, x] / board.crowns[y, x].
# Board.occupied_bbox() returns (min_x, min_y, max_x, max_y) -- unpack in that order.


def total_score(score):
    return score.territory_score + score.harmony_bonus + score.middle_kingdom_bonus


@dataclass(frozen=True)
class EvalBreakdown:
    total: float
    score: float
    occupied: float
    crowns: float
    terrain_potential: float
    bonus_potential: float
    discard_penalty: float


def evaluate_board(board):
    """
    Static board evaluation.

    This is intentionally heuristic. It is not training truth.
    It gives shallow MCTS a better leaf value than current score alone.
    """

    score_obj = board.score()
    score_value = total_score(score_obj)

    occupied_cells = board.occupied_cells()
    occupied = len(occupied_cells)

    crowns = int(board.crowns.sum())

    terrain_potential = estimate_terrain_potential(board)
    bonus_potential = estimate_bonus_potential(board)
    discard_penalty = estimate_discard_pressure(board)

    total = (
        1.00 * score_value
        + 0.35 * occupied
        + 1.50 * crowns
        + 0.75 * terrain_potential
        + 1.00 * bonus_potential
        - 2.00 * discard_penalty
    )

    return EvalBreakdown(
        total=total,
        score=score_value,
        occupied=occupied,
        crowns=crowns,
        terrain_potential=terrain_potential,
        bonus_potential=bonus_potential,
        discard_penalty=discard_penalty,
    )


def evaluate_state(state, player):
    """
    Return value from player's perspective, squashed to [-1, 1].
    """

    my_eval = evaluate_board(state.boards[player]).total
    opp_eval = evaluate_board(state.boards[1 - player]).total

    margin = my_eval - opp_eval
    return math.tanh(margin / 50.0)


def estimate_terrain_potential(board):
    """
    Reward terrain groups that have crowns and room to grow.

    This approximates future scoring potential:
    - crowns are valuable only if attached to regions
    - open edges around crowned terrain are useful
    """

    total = 0.0
    visited = set()

    for x, y in board.occupied_cells():
        terrain = board.terrain[y, x]

        # Skip castle / empty-like cells.
        if terrain <= 1:
            continue

        if (x, y) in visited:
            continue

        region = flood_region(board, x, y, terrain, visited)
        size = len(region)
        crowns = sum(int(board.crowns[ry, rx]) for rx, ry in region)
        liberties = count_region_liberties(board, region)

        if crowns > 0:
            total += size * crowns
            total += 0.25 * liberties * crowns
        else:
            # Crownless regions are still mildly useful if expandable.
            total += 0.05 * size * liberties

    return total


def estimate_bonus_potential(board):
    """
    Estimate Harmony and Middle Kingdom potential before final scoring.
    """

    occupied = board.occupied_cells()
    if not occupied:
        return 0.0

    min_x, min_y, max_x, max_y = board.occupied_bbox()
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    bonus = 0.0

    # Middle Kingdom potential: castle near center of current bbox.
    cx, cy = board.castle_pos
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_distance = abs(cx - center_x) + abs(cy - center_y)

    bonus += max(0.0, 5.0 - center_distance)

    # Harmony potential: reward compactness / low gaps inside bbox.
    bbox_area = width * height
    gaps = bbox_area - len(occupied)

    if width <= 7 and height <= 7:
        bonus += max(0.0, 8.0 - 0.5 * gaps)

    return bonus


def estimate_discard_pressure(board):
    """
    Penalize cramped boards with low remaining placement flexibility.

    This is intentionally rough. Later we can make it domino-aware.
    """

    occupied = board.occupied_cells()
    if not occupied:
        return 0.0

    min_x, min_y, max_x, max_y = board.occupied_bbox()
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    pressure = 0.0

    if width >= 7:
        pressure += 2.0
    if height >= 7:
        pressure += 2.0

    # Penalize high occupancy with remaining gaps, since random play can trap holes.
    bbox_area = width * height
    gaps = bbox_area - len(occupied)

    if gaps > 0 and len(occupied) > 30:
        pressure += 0.25 * gaps

    return pressure


def flood_region(board, start_x, start_y, terrain, visited):
    stack = [(start_x, start_y)]
    region = []

    while stack:
        x, y = stack.pop()

        if (x, y) in visited:
            continue

        if board.terrain[y, x] != terrain:
            continue

        visited.add((x, y))
        region.append((x, y))

        for nx, ny in board.adjacent_coords(x, y):
            if board.in_bounds(nx, ny) and (nx, ny) not in visited:
                if board.terrain[ny, nx] == terrain:
                    stack.append((nx, ny))

    return region


def count_region_liberties(board, region):
    liberties = set()

    for x, y in region:
        for nx, ny in board.adjacent_coords(x, y):
            if board.in_bounds(nx, ny) and board.is_empty(nx, ny):
                liberties.add((nx, ny))

    return len(liberties)


def score_action_prior(state, action, player=None):
    """
    Cheap prior score for move ordering / future progressive widening.

    This should be fast and not require deep copying every action.
    """

    score = 0.0

    placement = getattr(action, "placement", None)
    pick_id = getattr(action, "pick_domino_id", None)

    # Placement/discard component.
    if hasattr(action, "placement"):
        if placement is None:
            score -= 20.0
        else:
            score += 5.0
            score += placement_shape_prior(placement)

    # Draft component.
    if pick_id is not None:
        domino = DOMINOES[pick_id]
        crowns = domino.a.crowns + domino.b.crowns

        score += 8.0 * crowns
        score += 0.06 * pick_id

        # Mildly prefer double-terrain tiles because they are easier to grow.
        if domino.a.terrain == domino.b.terrain:
            score += 1.0

    return score


def placement_shape_prior(placement):
    """
    Tiny geometric prior.

    Prefer compact placements slightly. This is weak on purpose.
    """

    x1, y1 = placement.x1, placement.y1
    x2, y2 = placement.x2, placement.y2

    # Domino halves should always be adjacent, but keep this robust.
    distance = abs(x1 - x2) + abs(y1 - y2)

    if distance != 1:
        return -100.0

    # No strong preference yet.
    return 0.0