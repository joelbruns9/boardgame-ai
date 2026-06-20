import math
from dataclasses import dataclass
from functools import lru_cache

from games.kingdomino.dominoes import DOMINOES, Terrain


# NOTE ON INDEXING CONVENTION
# ---------------------------
# Board stores numpy arrays in [row, col] == [y, x] order
# (see board.py: self.terrain[cy, cx], self.terrain[y, x], ...).
# However Board.occupied_cells() returns (x, y) tuples.
# Therefore any DIRECT array read must use board.terrain[y, x] / board.crowns[y, x].
# Board.occupied_bbox() returns (min_x, min_y, max_x, max_y) -- unpack in that order.


TERRAINS = tuple(t for t in Terrain if t not in (Terrain.EMPTY, Terrain.CASTLE))


# Full-deck crown supply by terrain.  This lets the action prior understand that
# a large crownless wheat/water/etc. region can be valuable if crowns for that
# terrain are still available.
TOTAL_CROWNS_BY_TERRAIN = {t: 0 for t in TERRAINS}
for _domino in DOMINOES.values():
    TOTAL_CROWNS_BY_TERRAIN[_domino.a.terrain] += _domino.a.crowns
    TOTAL_CROWNS_BY_TERRAIN[_domino.b.terrain] += _domino.b.crowns


# Heuristic weights.  These are intentionally collected here so ladder sweeps can
# tune them without digging through the implementation.
PLACEMENT_BASE_BONUS = 4.0
DISCARD_PENALTY = 30.0
IMMEDIATE_DELTA_WEIGHT = 1.10
STRUCTURE_DELTA_WEIGHT = 0.75
TERRAIN_CONNECTION_WEIGHT = 1.40
CROWN_TO_REGION_WEIGHT = 1.15
COMMITTED_TERRAIN_WEIGHT = 1.75
CROWN_READY_WEIGHT = 0.22
OPEN_EDGE_WEIGHT = 0.20
BOUNDING_BOX_EXPANSION_PENALTY = 1.20
MIDGAME_SCATTER_PENALTY = 1.25
PICK_BOARD_FIT_WEIGHT = 1.20
PICK_CROWN_WEIGHT = 5.00
PICK_ID_WEIGHT = 0.035
TURN_ORDER_BASE_WEIGHT = 0.22
TURN_ORDER_URGENCY_WEIGHT = 0.48
FUTURE_ROW_DISCOUNT = 0.35
UNSEEN_CROWN_DISCOUNT = 0.05


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


@dataclass(frozen=True)
class RegionInfo:
    terrain: Terrain
    cells: tuple[tuple[int, int], ...]
    size: int
    crowns: int
    liberties: int


@dataclass(frozen=True)
class TerrainInfo:
    terrain: Terrain
    total_cells: int
    total_crowns: int
    largest_size: int
    largest_crowns: int
    largest_liberties: int
    regions: tuple[RegionInfo, ...]

    @property
    def commitment(self) -> float:
        """How much this board is already invested in this terrain."""
        return (
            self.largest_size
            + 4.0 * self.largest_crowns
            + 0.45 * self.largest_liberties
            + 0.20 * self.total_cells
            + 1.50 * self.total_crowns
        )


@dataclass(frozen=True)
class BoardProfile:
    terrain: dict[Terrain, TerrainInfo]
    region_by_cell: dict[tuple[int, int], RegionInfo]
    top_terrains: tuple[Terrain, ...]
    occupied_count: int

    def info(self, terrain: Terrain) -> TerrainInfo:
        return self.terrain.get(
            terrain,
            TerrainInfo(terrain, 0, 0, 0, 0, 0, ()),
        )

    def commitment(self, terrain: Terrain) -> float:
        return self.info(terrain).commitment


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
        if (x, y) in visited:
            continue

        terrain = Terrain(int(board.terrain[y, x]))

        # Skip castle / empty cells. Use the explicit enum check rather than a
        # magic numeric threshold so this stays correct if the enum is reordered.
        if terrain in (Terrain.EMPTY, Terrain.CASTLE):
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
    # Normalize once so the function behaves identically whether `terrain` is
    # passed as a numpy int (estimate_terrain_potential) or a Terrain enum
    # (build_board_profile). Terrain is an IntEnum, so int() is exact and the
    # cell comparisons below are plain int-vs-int.
    terrain = int(terrain)

    stack = [(start_x, start_y)]
    region = []

    while stack:
        x, y = stack.pop()

        if (x, y) in visited:
            continue

        if int(board.terrain[y, x]) != terrain:
            continue

        visited.add((x, y))
        region.append((x, y))

        for nx, ny in board.adjacent_coords(x, y):
            if board.in_bounds(nx, ny) and (nx, ny) not in visited:
                if int(board.terrain[ny, nx]) == terrain:
                    stack.append((nx, ny))

    return region


def count_region_liberties(board, region):
    liberties = set()

    for x, y in region:
        for nx, ny in board.adjacent_coords(x, y):
            if board.in_bounds(nx, ny) and board.is_empty(nx, ny):
                liberties.add((nx, ny))

    return len(liberties)


def build_board_profile(board) -> BoardProfile:
    """Summarize terrain commitments and connected regions on a board."""

    visited = set()
    region_by_cell = {}
    regions_by_terrain = {t: [] for t in TERRAINS}

    occupied_cells = board.occupied_cells()

    for x, y in occupied_cells:
        terrain = Terrain(int(board.terrain[y, x]))
        if terrain in (Terrain.EMPTY, Terrain.CASTLE) or (x, y) in visited:
            continue

        cells = tuple(flood_region(board, x, y, terrain, visited))
        crowns = sum(int(board.crowns[cy, cx]) for cx, cy in cells)
        liberties = count_region_liberties(board, cells)
        info = RegionInfo(terrain, cells, len(cells), crowns, liberties)
        regions_by_terrain[terrain].append(info)
        for cell in cells:
            region_by_cell[cell] = info

    terrain_info = {}
    for terrain, regions in regions_by_terrain.items():
        if not regions:
            continue
        largest = max(regions, key=lambda r: (r.size, r.crowns, r.liberties))
        terrain_info[terrain] = TerrainInfo(
            terrain=terrain,
            total_cells=sum(r.size for r in regions),
            total_crowns=sum(r.crowns for r in regions),
            largest_size=largest.size,
            largest_crowns=largest.crowns,
            largest_liberties=largest.liberties,
            regions=tuple(regions),
        )

    ranked = sorted(
        terrain_info.values(),
        key=lambda info: info.commitment,
        reverse=True,
    )
    top_terrains = tuple(info.terrain for info in ranked[:3] if info.commitment > 0)

    return BoardProfile(
        terrain=terrain_info,
        region_by_cell=region_by_cell,
        top_terrains=top_terrains,
        occupied_count=len(occupied_cells),
    )


def current_actor_player(state):
    actor = getattr(state, "current_actor", None)
    if callable(actor):
        actor = actor()
        return getattr(actor, "player", actor)
    if isinstance(actor, int):
        return actor

    # Fallbacks for older state implementations.
    if hasattr(state, "pending_claims") and getattr(state, "pending_claims", None):
        return state.pending_claims[state.actor_index].player
    return 0


def current_domino_id(state):
    if hasattr(state, "current_actor_domino"):
        value = state.current_actor_domino
        domino = value() if callable(value) else value
        if domino is not None:
            return domino.id

    if hasattr(state, "pending_claims") and getattr(state, "pending_claims", None):
        return state.pending_claims[state.actor_index].domino_id

    return None


def crown_counts_in_domino_ids(domino_ids) -> dict[Terrain, int]:
    counts = {t: 0 for t in TERRAINS}
    for domino_id in domino_ids or []:
        domino = DOMINOES[domino_id]
        counts[domino.a.terrain] += domino.a.crowns
        counts[domino.b.terrain] += domino.b.crowns
    return counts


def already_claimed_or_played_ids(state) -> set[int]:
    ids = set()
    ids.update(getattr(state, "current_row", []) or [])
    ids.update(getattr(state, "deck", []) or [])
    for claim in getattr(state, "pending_claims", []) or []:
        ids.add(claim.domino_id)
    for claim in getattr(state, "next_claims", []) or []:
        ids.add(claim.domino_id)
    # Invert the known-not-played set.  This is used only as a rough crown
    # opportunity signal; exactness is not critical for move ordering.
    return set(DOMINOES) - ids


def remaining_crown_opportunity(state, terrain: Terrain) -> float:
    """
    Estimate future crown access for a terrain.

    current_row is visible and actionable now.

    state.deck is the hidden remaining draw pile. The engine stores it in true
    shuffled order, but a real player/advisor should only treat it as an
    unordered remaining pool. Therefore this function must not use state.deck[:4]
    as the known next row.
    """

    current_counts = crown_counts_in_domino_ids(getattr(state, "current_row", []) or [])

    remaining_pool = list(getattr(state, "deck", []) or [])
    remaining_counts = crown_counts_in_domino_ids(remaining_pool)

    # Approximate the expected crown content of the next revealed row without
    # peeking at deck order. If 4 dominoes will be drawn from an unordered pool,
    # expected crowns by terrain are pool crowns * 4 / pool_size.
    if remaining_pool:
        next_row_probability = min(1.0, 4.0 / len(remaining_pool))
    else:
        next_row_probability = 0.0

    expected_next_row_counts = {
        t: remaining_counts[t] * next_row_probability
        for t in TERRAINS
    }

    used_ids = already_claimed_or_played_ids(state)
    used_crowns = crown_counts_in_domino_ids(used_ids)
    unseen_remaining = max(0, TOTAL_CROWNS_BY_TERRAIN[terrain] - used_crowns[terrain])

    return (
        current_counts[terrain]
        + FUTURE_ROW_DISCOUNT * expected_next_row_counts[terrain]
        + UNSEEN_CROWN_DISCOUNT * unseen_remaining
    )


def terrain_crown_opportunities(state) -> dict[Terrain, float]:
    return {terrain: remaining_crown_opportunity(state, terrain) for terrain in TERRAINS}


def score_action_prior(state, action, player=None):
    """
    Heuristic prior for move ordering and progressive widening.

    The prior now models three human-style ideas:
      1. early structure: bunch matching terrain into expandable regions;
      2. terrain commitment: after a few rounds, lean into the 1-3 terrain types
         that already look like scoring engines;
      3. draft urgency: value picks and turn order when visible/future crown
         opportunities match the player's board.
    """

    if player is None:
        player = current_actor_player(state)

    placement = getattr(action, "placement", None)
    pick_id = getattr(action, "pick_domino_id", None)
    if pick_id is None:
        pick_id = getattr(action, "domino_id", None)

    board = state.boards[player]
    profile = build_board_profile(board)
    crown_opportunity = terrain_crown_opportunities(state)

    score = 0.0

    # Placement/discard component.
    if hasattr(action, "placement"):
        if placement is None:
            score -= DISCARD_PENALTY
        else:
            domino_id = current_domino_id(state)
            domino = DOMINOES[domino_id]
            score += PLACEMENT_BASE_BONUS
            score += placement_strategy_prior(
                state=state,
                board=board,
                profile=profile,
                crown_opportunity=crown_opportunity,
                domino=domino,
                placement=placement,
            )

    # Draft component.
    if pick_id is not None:
        score += pick_strategy_prior(
            state=state,
            profile=profile,
            crown_opportunity=crown_opportunity,
            pick_id=pick_id,
        )

    return score


def placement_strategy_prior(state, board, profile, crown_opportunity, domino, placement):
    """Score the placement part of an action from a strategic-board perspective."""

    before_score = total_score(board.score())
    before_structure = board_structure_value(board, profile, crown_opportunity)

    board_after = board.copy()
    board_after.place(domino, placement)
    after_profile = build_board_profile(board_after)
    after_score = total_score(board_after.score())
    after_structure = board_structure_value(board_after, after_profile, crown_opportunity)

    immediate_delta = after_score - before_score
    structure_delta = after_structure - before_structure

    h1, h2 = (domino.b, domino.a) if placement.flipped else (domino.a, domino.b)
    halves = ((placement.x1, placement.y1, h1), (placement.x2, placement.y2, h2))

    local = 0.0
    terrains_placed = []
    connected_terrain_edges = 0

    for x, y, half in halves:
        terrain = half.terrain
        terrains_placed.append(terrain)
        if terrain in (Terrain.EMPTY, Terrain.CASTLE):
            continue

        neighbor_regions = []
        same_edges = 0
        castle_edges = 0
        for nx, ny in board.adjacent_coords(x, y):
            neighbor_terrain = Terrain(int(board.terrain[ny, nx]))
            if neighbor_terrain == terrain:
                same_edges += 1
                region = profile.region_by_cell.get((nx, ny))
                if region is not None and region not in neighbor_regions:
                    neighbor_regions.append(region)
            elif neighbor_terrain == Terrain.CASTLE:
                castle_edges += 1

        connected_terrain_edges += same_edges
        connected_size = sum(r.size for r in neighbor_regions)
        connected_crowns = sum(r.crowns for r in neighbor_regions)
        opportunity = crown_opportunity.get(terrain, 0.0)
        commitment = profile.commitment(terrain)

        # Bunch matching terrain together, especially for terrain already acting
        # as a scoring engine.
        local += TERRAIN_CONNECTION_WEIGHT * same_edges
        local += 0.18 * connected_size
        local += COMMITTED_TERRAIN_WEIGHT * math.log1p(commitment) * same_edges

        # Adding crowns to an existing large region is the classic high-leverage
        # Kingdomino move.
        if half.crowns > 0:
            local += PICK_CROWN_WEIGHT * half.crowns
            local += CROWN_TO_REGION_WEIGHT * half.crowns * max(1, connected_size)
            local += 0.80 * half.crowns * math.log1p(commitment)

        # Crownless expansion is useful when crowns remain for that terrain.
        if half.crowns == 0:
            local += CROWN_READY_WEIGHT * max(1, connected_size + 1) * opportunity

        # Castle-only connections are legal, but usually less strategically
        # meaningful than joining same-terrain regions.
        if same_edges == 0 and castle_edges > 0 and profile.occupied_count > 5:
            local -= 0.75

        # Midgame: do not keep scattering new low-upside terrain once the board
        # has obvious main scoring candidates.
        if (
            profile.occupied_count >= 13
            and terrain not in profile.top_terrains
            and half.crowns == 0
            and same_edges == 0
        ):
            local -= MIDGAME_SCATTER_PENALTY

    # Prefer double-terrain dominoes and adjacent same-terrain halves because
    # they create/extend larger crown-ready regions.
    if terrains_placed[0] == terrains_placed[1]:
        local += 1.50
        if connected_terrain_edges > 0:
            local += 1.50

    # Keep regions open.  We measure after-placement liberties because the move
    # may either preserve growth lanes or bury a region against the edge.
    for terrain in set(terrains_placed):
        if terrain in (Terrain.EMPTY, Terrain.CASTLE):
            continue
        info = after_profile.info(terrain)
        opportunity = crown_opportunity.get(terrain, 0.0)
        local += OPEN_EDGE_WEIGHT * min(info.largest_liberties, 8) * min(2.0, 0.5 + opportunity)

    local += bbox_shape_delta_prior(board, board_after)

    return (
        IMMEDIATE_DELTA_WEIGHT * immediate_delta
        + STRUCTURE_DELTA_WEIGHT * structure_delta
        + local
    )


def board_structure_value(board, profile, crown_opportunity):
    """Evaluate long-term structure independent of current score."""

    value = 0.0
    for terrain in TERRAINS:
        info = profile.info(terrain)
        if info.total_cells == 0:
            continue

        opportunity = crown_opportunity.get(terrain, 0.0)

        # Crowned regions are already scoring.  Crownless large regions are only
        # valuable if crown opportunity remains for that terrain.
        if info.largest_crowns > 0:
            value += info.largest_size * info.largest_crowns
            value += 0.18 * info.largest_liberties * info.largest_crowns
        else:
            value += CROWN_READY_WEIGHT * info.largest_size * opportunity
            value += 0.06 * info.largest_liberties * opportunity

        # Commitment bonus encourages the bot to turn 1-3 terrains into real
        # scoring plans instead of making six disconnected mini-regions.
        if terrain in profile.top_terrains:
            value += 0.40 * math.log1p(info.commitment) * max(1.0, opportunity)

        # Fragmentation penalty: multiple separate regions of the same terrain
        # are harder to turn into one high-multiplier score.
        value -= 0.45 * max(0, len(info.regions) - 1)

    return value


def bbox_shape_delta_prior(board_before, board_after):
    before = board_before.occupied_bbox()
    after = board_after.occupied_bbox()
    if before is None or after is None:
        return 0.0

    before_min_x, before_min_y, before_max_x, before_max_y = before
    after_min_x, after_min_y, after_max_x, after_max_y = after

    before_width = before_max_x - before_min_x + 1
    before_height = before_max_y - before_min_y + 1
    after_width = after_max_x - after_min_x + 1
    after_height = after_max_y - after_min_y + 1

    before_area = before_width * before_height
    after_area = after_width * after_height
    area_growth = max(0, after_area - before_area)
    edge_pressure = max(0, after_width - before_width) + max(0, after_height - before_height)

    # Early expansion is sometimes necessary; late expansion toward the 7x7 edge
    # should be more cautious because it increases future discard risk.
    occupied_after = len(board_after.occupied_cells())
    late_multiplier = 1.0 + max(0, occupied_after - 17) / 20.0

    penalty = BOUNDING_BOX_EXPANSION_PENALTY * late_multiplier * edge_pressure
    penalty += 0.12 * late_multiplier * area_growth

    # Small compactness bonus if a move adds cells without enlarging the bbox.
    if edge_pressure == 0:
        penalty -= 0.85

    return -penalty


def pick_strategy_prior(state, profile, crown_opportunity, pick_id):
    """Score the draft/pick part of an action."""

    domino = DOMINOES[pick_id]
    terrains = (domino.a.terrain, domino.b.terrain)
    crowns_by_terrain = {
        terrain: (domino.a.crowns if domino.a.terrain == terrain else 0)
        + (domino.b.crowns if domino.b.terrain == terrain else 0)
        for terrain in set(terrains)
    }
    total_crowns = domino.a.crowns + domino.b.crowns

    value = 0.0

    # Intrinsic tile value.
    value += PICK_CROWN_WEIGHT * total_crowns
    value += PICK_ID_WEIGHT * pick_id
    if domino.a.terrain == domino.b.terrain:
        value += 1.0

    # Fit the picked tile to my existing board plan.  Crowns matching a large
    # crown-ready region get a major boost; blank terrain matching a committed
    # terrain gets a smaller structure boost.
    for terrain in set(terrains):
        info = profile.info(terrain)
        commitment = profile.commitment(terrain)
        terrain_crowns = crowns_by_terrain.get(terrain, 0)

        if terrain_crowns > 0:
            crown_ready_size = max(info.largest_size, 1)
            value += PICK_BOARD_FIT_WEIGHT * terrain_crowns * crown_ready_size
            value += 0.75 * terrain_crowns * math.log1p(commitment)
        else:
            value += 0.30 * math.log1p(commitment)
            if terrain in profile.top_terrains:
                value += 0.80

    # Turn order: low-number tiles act earlier next round.  This becomes much
    # more valuable when the next selection row appears to contain crown tiles
    # that match my committed/crown-ready terrains.
    value += turn_order_prior(state, profile, crown_opportunity, pick_id)

    return value


def turn_order_prior(state, profile, crown_opportunity, pick_id):
    current_row = sorted(getattr(state, "current_row", []) or [])
    if pick_id not in current_row or len(current_row) <= 1:
        return 0.0

    # rank 0 is earliest next turn, rank n-1 is latest.
    rank = current_row.index(pick_id)
    n = len(current_row)
    early_score = (n - 1 - rank) / (n - 1)

    # Generic benefit of acting earlier.
    value = TURN_ORDER_BASE_WEIGHT * early_score

    urgency = 0.0
    for terrain in profile.top_terrains or TERRAINS:
        info = profile.info(terrain)
        if info.total_cells == 0:
            continue
        opportunity = crown_opportunity.get(terrain, 0.0)
        if opportunity <= 0:
            continue

        # Large low/no-crown regions have the most to gain from securing an
        # upcoming crown tile.  Existing crowned regions also benefit, but the
        # urgency is slightly lower because they already score.
        crown_need = 1.0 if info.largest_crowns == 0 else 0.55
        urgency += crown_need * opportunity * max(1.0, info.largest_size / 5.0)

    value += TURN_ORDER_URGENCY_WEIGHT * early_score * urgency
    return value


def placement_shape_prior(placement):
    """
    Tiny geometric prior retained for compatibility with older experiments.

    Most placement scoring now lives in placement_strategy_prior().
    """

    x1, y1 = placement.x1, placement.y1
    x2, y2 = placement.x2, placement.y2

    # Domino halves should always be adjacent, but keep this robust.
    distance = abs(x1 - x2) + abs(y1 - y2)

    if distance != 1:
        return -100.0

    return 0.0