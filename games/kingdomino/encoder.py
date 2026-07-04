"""
State encoder for the Kingdomino value/policy network.

DESIGN CONTRACT
───────────────
1. Information-set safe.  Never reads state.deck ORDER. The remaining-bag
   feature reads deck MEMBERSHIP (which tiles are still unseen) — public
   information every player can track — but never the order of future draws,
   so the encoding is identical for any determinization that respects the
   information set.

   IMPORTANT — necessary but not sufficient for fair play.  The encoder
   being information-set safe means the *features* a network sees from a
   single state are public-information-only.  But for training a model
   meant to play fairly from public information, the *search* must also
   respect the boundary.  If MCTS simulates forward from a real GameState
   whose deck order is fixed, future tile reveals leak through simulation
   outcomes even though the encoder is clean.

   AlphaZero MCTS must either (a) run only from self-play states where
   the deck order is part of the sampled world, or (b) redeterminize the
   hidden deck at the root of every search by sampling a fresh shuffle of
   the current bag.  Option (b) is what `redeterminize()` (below) is for.
   This module flags the responsibility; enforcement happens in the
   search code, not here.

2. Training-isolated.  This module does NOT import evaluation.py.  The
   heuristic evaluator must never influence the network's training signal,
   so no code path in the encoder can reach it.  A grep on
   "from games.kingdomino.evaluation" should never match anything here.

3. Castle-centered.  Both boards are projected onto a 13×13 canvas with
   the castle fixed at index (CASTLE_CENTER, CASTLE_CENTER) = (6, 6).  This
   is a coordinate normalisation, not a strategy constraint: a kingdom can
   legally extend at most 6 cells from the castle in any single direction
   (with the 7×7 bbox constraint applied across both axes simultaneously),
   so 13×13 is exactly the smallest canvas that holds every legal kingdom.

4. D4-friendly.  Spatial planes can be rotated and reflected freely (np.rot90,
   array slicing).  Flat features are rotation-invariant — they are NOT
   transformed during D4 augmentation.

5. Reusable across players.  encode_state(state, player) returns the boards
   in (my, opp) order, so the same network weights process both perspectives.
"""
from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import numpy as np

from games.kingdomino.dominoes import DOMINOES, Terrain
from games.kingdomino.game import GameState, Phase, determine_winner

# ─── shape constants ──────────────────────────────────────────────────────
CANVAS_SIZE = 13
CASTLE_CENTER = CANVAS_SIZE // 2  # 6

# Terrain channels cover only placeable terrain (WHEAT..MINE).
# EMPTY/CASTLE are handled by the occupied and castle mask channels.
NUM_PLACEABLE_TERRAINS = 6
TERRAIN_INDEX_OFFSET = int(Terrain.WHEAT)  # 2 — first placeable terrain

MAX_CROWNS = 3
NUM_DOMINOES = 48
MAX_PHASE_SLOTS = 4  # current_row, pending_claims, next_claims all max at 4

# Spatial channel indices (channels-first, (C, H, W) convention to match PyTorch)
CH_TERRAIN_START = 0
CH_TERRAIN_END = 6  # exclusive
CH_CROWNS = 6
CH_CASTLE = 7
CH_OCCUPIED = 8
NUM_BOARD_CHANNELS = 9

# Flat per-tile encoding: half-A one-hot (6) + crowns-A (1) + half-B one-hot (6) + crowns-B (1)
TILE_FEAT_SIZE = 2 * (NUM_PLACEABLE_TERRAINS + 1)  # 14
ROW_SLOT_SIZE = TILE_FEAT_SIZE + 1                  # + present flag = 15
CLAIM_SLOT_SIZE = TILE_FEAT_SIZE + 2                # + is_mine flag + status flag = 16
PENDING_SUMMARY_SIZE = TILE_FEAT_SIZE + 4           # + present + turn_distance + active + remaining_count
BOARD_SUMMARY_SIZE = 25
# Normaliser for the board-summary score features.  MUST match the training
# score_scale (self_play/network) so the summary inputs and the score-head
# targets live on the same scale, and high scores (~160) don't saturate.
SCORE_SCALE = 160.0
MAX_BOARD_CELLS = 48.0
MAX_TOTAL_CROWNS = 24.0
MAX_LEGAL_PLACEMENTS = 64.0


# ─── flat-vector layout ───────────────────────────────────────────────────
def _build_flat_layout() -> Tuple[dict, int]:
    """Build the flat vector layout. Returns (layout_dict, total_size).

    The last four features (pick_pos_0..pick_pos_3) make the NEXT-ROUND pick
    INTERLEAVING explicit. Element k says who acts at next-round pick position k
    (0=earliest domino_id, 3=latest): +1.0 = encoded player, -1.0 = opponent,
    0.0 = not yet committed (or no next round).  This is richer than the old two
    rank scalars, which only told who picks first — in Mighty Duel each player
    picks twice per round and the full interleaving (e.g. holding positions
    (1,3) vs (2,4)) is strategically distinct.  See _pick_positions.
    """
    sizes = [
        ('my_next_pending',  PENDING_SUMMARY_SIZE),                  # 18
        ('opp_next_pending', PENDING_SUMMARY_SIZE),                  # 18
        ('my_board_summary',  BOARD_SUMMARY_SIZE),                   # 25
        ('opp_board_summary', BOARD_SUMMARY_SIZE),                   # 25
        ('current_row',    ROW_SLOT_SIZE   * MAX_PHASE_SLOTS),       # 60
        ('pending_claims', CLAIM_SLOT_SIZE * MAX_PHASE_SLOTS),       # 64
        ('next_claims',    CLAIM_SLOT_SIZE * MAX_PHASE_SLOTS),       # 64
        ('bag',            NUM_DOMINOES),                            # 48
        ('phase',          3),                                       # init/place/final one-hot
        ('game_progress',  1),                                       # placed_cells / 96
        ('my_fill_ratio',  1),                                       # compactness (Harmony signal)
        ('opp_fill_ratio', 1),
        ('actor_flag',     1),                                       # is it my turn?
        ('pick_pos_0',     1),   # next-round pick pos 0: +1 me / -1 opp / 0 unknown
        ('pick_pos_1',     1),
        ('pick_pos_2',     1),
        ('pick_pos_3',     1),
    ]
    layout, offset = {}, 0
    for name, size in sizes:
        layout[name] = slice(offset, offset + size)
        offset += size
    return layout, offset


FLAT_LAYOUT, FLAT_SIZE = _build_flat_layout()
# FLAT_SIZE = 333 with current constants


# ─── primitives ───────────────────────────────────────────────────────────
def _encode_tile(domino_id: Optional[int]) -> np.ndarray:
    """Encode one domino as a 14-float vector. Returns zeros for None/missing."""
    out = np.zeros(TILE_FEAT_SIZE, dtype=np.float32)
    if not domino_id:  # None or 0
        return out
    dom = DOMINOES[domino_id]
    out[int(dom.a.terrain) - TERRAIN_INDEX_OFFSET] = 1.0
    out[NUM_PLACEABLE_TERRAINS] = dom.a.crowns / MAX_CROWNS
    out[NUM_PLACEABLE_TERRAINS + 1 + int(dom.b.terrain) - TERRAIN_INDEX_OFFSET] = 1.0
    out[2 * NUM_PLACEABLE_TERRAINS + 1] = dom.b.crowns / MAX_CROWNS
    return out


def _encode_row_slot(domino_id: Optional[int]) -> np.ndarray:
    out = np.zeros(ROW_SLOT_SIZE, dtype=np.float32)
    if not domino_id:
        return out
    out[:TILE_FEAT_SIZE] = _encode_tile(domino_id)
    out[TILE_FEAT_SIZE] = 1.0  # present
    return out


def _encode_claim_slot(claim, current_player: int, status_flag: float) -> np.ndarray:
    """Encode a Claim slot: tile features (14) + is_mine (1) + status (1)."""
    out = np.zeros(CLAIM_SLOT_SIZE, dtype=np.float32)
    if claim is None:
        return out
    out[:TILE_FEAT_SIZE] = _encode_tile(claim.domino_id)
    out[TILE_FEAT_SIZE] = 1.0 if claim.player == current_player else 0.0
    out[TILE_FEAT_SIZE + 1] = status_flag
    return out


def _next_pending_summary(state: GameState, owner: int) -> tuple[Optional[int], int, int]:
    """Return (domino_id, distance, remaining_count) for owner's next claim.

    Current-round unresolved pending_claims take priority. If owner has no
    unresolved current-round claim, fall forward to their earliest next_claims
    commitment so claimed-but-unplaced tiles stay visible across round
    boundaries.

    distance is measured in placement-order slots from the current actor_index:
      0 = this owner is placing now
      1 = after one more pending placement
      ...
    remaining_count counts owner's claims remaining in the chosen claim source.
    """
    current_remaining = []
    if state.phase in (Phase.PLACE_AND_SELECT, Phase.FINAL_PLACEMENT):
        current_remaining = [
            (idx, claim)
            for idx, claim in enumerate(state.pending_claims)
            if idx >= state.actor_index
        ]
        own_current = [(idx, claim) for idx, claim in current_remaining
                       if claim.player == owner]
        if own_current:
            idx, claim = own_current[0]
            return claim.domino_id, idx - state.actor_index, len(own_current)

    if state.phase in (Phase.INITIAL_SELECTION, Phase.PLACE_AND_SELECT):
        next_order = sorted(state.next_claims, key=lambda c: c.domino_id)
        own_next = [(idx, claim) for idx, claim in enumerate(next_order)
                    if claim.player == owner]
        if own_next:
            idx, claim = own_next[0]
            return claim.domino_id, len(current_remaining) + idx, len(own_next)

    return None, 0, 0


def _encode_pending_summary(state: GameState, owner: int) -> np.ndarray:
    out = np.zeros(PENDING_SUMMARY_SIZE, dtype=np.float32)
    domino_id, distance, remaining_count = _next_pending_summary(state, owner)
    if domino_id is None:
        return out
    out[:TILE_FEAT_SIZE] = _encode_tile(domino_id)
    out[TILE_FEAT_SIZE] = 1.0
    out[TILE_FEAT_SIZE + 1] = min(float(distance), 3.0) / 3.0
    out[TILE_FEAT_SIZE + 2] = 1.0 if distance == 0 else 0.0
    out[TILE_FEAT_SIZE + 3] = min(float(remaining_count), 2.0) / 2.0
    return out


def _board_component_facts(board) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (score_by_terrain, largest_by_terrain, total_crowns)."""
    score_by = np.zeros(NUM_PLACEABLE_TERRAINS, dtype=np.float32)
    largest_by = np.zeros(NUM_PLACEABLE_TERRAINS, dtype=np.float32)
    total_crowns = 0
    visited = np.zeros_like(board.terrain, dtype=bool)
    bbox = board.occupied_bbox()
    if bbox is None:
        return score_by, largest_by, total_crowns
    min_x, min_y, max_x, max_y = bbox
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            t = int(board.terrain[y, x])
            if visited[y, x] or t in (int(Terrain.EMPTY), int(Terrain.CASTLE)):
                continue
            terrain_idx = t - TERRAIN_INDEX_OFFSET
            stack = [(x, y)]
            visited[y, x] = True
            area = 0
            crowns = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                crowns += int(board.crowns[cy, cx])
                for nx, ny in board.adjacent_coords(cx, cy):
                    if not visited[ny, nx] and int(board.terrain[ny, nx]) == t:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            score_by[terrain_idx] += float(area * crowns)
            largest_by[terrain_idx] = max(largest_by[terrain_idx], float(area))
            total_crowns += crowns
    return score_by, largest_by, total_crowns


def _bonus_state_features(state: GameState, board, owner: int) -> tuple[np.ndarray, np.ndarray]:
    """Return factual bonus states for harmony and middle kingdom.

    Layout per bonus: [currently_awarded, still_possible, impossible].  Both
    tests are EXACT in the safe direction — they never mark a truly-possible
    position impossible.

    Harmony needs occupied == 49 (a full 7×7), i.e. all 24 dominoes placed →
    zero discards.  So harmony is impossible the instant this player discards
    (a forced discard permanently caps the board below 49 cells).

    Middle kingdom needs the castle centred in a 7×7 bbox at game end; it does
    NOT require a full fill, so discards are irrelevant.  It becomes impossible
    once the bbox extends outside the castle-centred 7×7 target (it can then
    never end as a castle-centred 7×7).
    """
    occupied = len(board.occupied_cells())
    bbox = board.occupied_bbox()
    if bbox is None:
        min_x = max_x = board.castle_pos[0]
        min_y = max_y = board.castle_pos[1]
    else:
        min_x, min_y, max_x, max_y = bbox
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    harmony = np.zeros(3, dtype=np.float32)
    if state.config.harmony:
        awarded = width == 7 and height == 7 and occupied == 49
        impossible = state.discards[owner] > 0
        if awarded:
            harmony[0] = 1.0
        elif impossible:
            harmony[2] = 1.0
        else:
            harmony[1] = 1.0

    middle = np.zeros(3, dtype=np.float32)
    if state.config.middle_kingdom:
        cx, cy = board.castle_pos
        awarded = (
            width == 7 and height == 7
            and (cx, cy) == (min_x + 3, min_y + 3)
        )
        outside_target = (
            min_x < cx - 3 or max_x > cx + 3
            or min_y < cy - 3 or max_y > cy + 3
        )
        if awarded:
            middle[0] = 1.0
        elif outside_target:
            middle[2] = 1.0
        else:
            middle[1] = 1.0
    return harmony, middle


def _encode_board_summary(state: GameState, player: int) -> np.ndarray:
    board = state.boards[player]
    out = np.zeros(BOARD_SUMMARY_SIZE, dtype=np.float32)
    score = board.score(state.config.harmony, state.config.middle_kingdom)
    score_by, largest_by, total_crowns = _board_component_facts(board)
    harmony, middle = _bonus_state_features(state, board, player)
    bbox = board.occupied_bbox()
    if bbox is None:
        width = height = 1
        occupied = 1
    else:
        min_x, min_y, max_x, max_y = bbox
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        occupied = len(board.occupied_cells())
    next_domino, _, _ = _next_pending_summary(state, player)
    legal_count = 0
    if next_domino is not None:
        legal_count = len(board.legal_placements(DOMINOES[next_domino]))

    off = 0
    out[off] = min(float(score.total), SCORE_SCALE) / SCORE_SCALE; off += 1
    out[off:off + NUM_PLACEABLE_TERRAINS] = np.minimum(score_by, SCORE_SCALE) / SCORE_SCALE
    off += NUM_PLACEABLE_TERRAINS
    out[off:off + NUM_PLACEABLE_TERRAINS] = np.minimum(largest_by, MAX_BOARD_CELLS) / MAX_BOARD_CELLS
    off += NUM_PLACEABLE_TERRAINS
    out[off] = min(float(total_crowns), MAX_TOTAL_CROWNS) / MAX_TOTAL_CROWNS; off += 1
    out[off:off + 3] = harmony; off += 3
    out[off:off + 3] = middle; off += 3
    out[off] = min(float(width), 7.0) / 7.0; off += 1
    out[off] = min(float(height), 7.0) / 7.0; off += 1
    out[off] = min(float(max(0, 49 - occupied)), MAX_BOARD_CELLS) / MAX_BOARD_CELLS; off += 1
    out[off] = min(float(legal_count), MAX_LEGAL_PLACEMENTS) / MAX_LEGAL_PLACEMENTS; off += 1
    out[off] = 1.0 if next_domino is not None and legal_count == 0 else 0.0
    return out


def _encode_board_spatial(board) -> np.ndarray:
    """Encode one Board as a (9, 13, 13) tensor, castle pinned to (CASTLE_CENTER, CASTLE_CENTER).

    Channel layout:
        0..5 : terrain one-hot (WHEAT, FOREST, WATER, GRASS, SWAMP, MINE)
        6    : crowns / MAX_CROWNS
        7    : castle mask (always 1 at the centre after centring)
        8    : occupied mask (1 wherever a tile or the castle sits)

    Indexing convention follows the project's documented rule
    (board.terrain[y, x]).  Output uses PyTorch-style (C, H, W) = (C, y, x).
    """
    out = np.zeros((NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    cx, cy = board.castle_pos  # canvas coords

    # Castle anchor — always at the output centre regardless of canvas position
    out[CH_CASTLE,    CASTLE_CENTER, CASTLE_CENTER] = 1.0
    out[CH_OCCUPIED,  CASTLE_CENTER, CASTLE_CENTER] = 1.0

    for x, y in board.occupied_cells():
        if (x, y) == (cx, cy):
            continue  # castle already handled above

        # Translate canvas coords → castle-centred output coords
        out_x = x - cx + CASTLE_CENTER
        out_y = y - cy + CASTLE_CENTER
        if not (0 <= out_x < CANVAS_SIZE and 0 <= out_y < CANVAS_SIZE):
            raise ValueError(
                f"Board cell ({x}, {y}) maps to out-of-canvas output "
                f"({out_x}, {out_y}) on a 13×13 canvas. "
                f"Castle at ({cx}, {cy}). This indicates a board "
                f"invariant violation — legal kingdoms must fit within "
                f"6 cells of the castle in all directions."
            )

        terrain_val = int(board.terrain[y, x])
        crowns_val  = int(board.crowns[y, x])

        # The castle cell itself was handled before this loop (the (x,y)==(cx,cy)
        # guard above), so an occupied cell here must be a placed half with real
        # terrain.  EMPTY or CASTLE terrain on a non-castle occupied cell is a
        # board-invariant violation — fail loudly rather than silently skip.
        if terrain_val == int(Terrain.EMPTY):
            raise ValueError(
                f"occupied_cells() returned cell ({x}, {y}) with "
                f"EMPTY terrain — board invariant violation."
            )
        if terrain_val == int(Terrain.CASTLE):
            raise ValueError(
                f"occupied_cells() returned non-castle cell ({x}, {y}) "
                f"with CASTLE terrain — board invariant violation."
            )

        terrain_ch = terrain_val - TERRAIN_INDEX_OFFSET
        out[terrain_ch, out_y, out_x] = 1.0
        out[CH_CROWNS,  out_y, out_x] = crowns_val / MAX_CROWNS
        out[CH_OCCUPIED, out_y, out_x] = 1.0

    return out


def _fill_ratio(board) -> float:
    """Filled cells (incl. castle) divided by occupied bbox area.

    Encodes 'how compact is the kingdom' — directly relevant to the Harmony
    bonus, which requires zero gaps within the kingdom's 7×7 footprint.
    Returns 0.0 for an empty board.
    """
    occupied = board.occupied_cells()
    if not occupied:
        return 0.0
    bbox = board.occupied_bbox()
    if bbox is None:
        return 0.0
    min_x, min_y, max_x, max_y = bbox
    area = (max_x - min_x + 1) * (max_y - min_y + 1)
    if area == 0:
        return 0.0
    return len(occupied) / area


def _compute_bag(state: GameState) -> np.ndarray:
    """Return the 48-float binary bag vector (1.0 = tile still unseen).

    IMPLEMENTATION NOTE: this function reads state.deck membership (not deck
    order) to determine which tiles remain. Deck membership is public
    information — every player knows which tiles have been revealed (in
    current_row, claims, or placed on boards) and can infer the remaining
    unseen set. Only the ORDER of future draws is hidden, and this function
    never reads that order.

    Reading deck membership directly is simpler and more robust than
    reconstructing it from public observations (the indirect approach had a
    silent bug where discarded tiles — which leave no public trace — were
    re-added to the bag). The information-set contract is preserved:
    encode_state(state, p) is byte-identical to
    encode_state(redeterminize(state, rng), p) because redeterminize only
    changes deck ORDER, not membership.
    """
    out = np.zeros(NUM_DOMINOES, dtype=np.float32)
    for did in state.deck:
        out[int(did) - 1] = 1.0
    return out


def _pick_positions(state: GameState, player: int) -> np.ndarray:
    """Compute next-round pick position features for the encoded player.

    Returns a (4,) float32 array where element k encodes who acts at pick
    position k (0=earliest, 3=latest) in the NEXT round:
        +1.0  encoded player acts at position k
        -1.0  opponent acts at position k
         0.0  not yet committed, or no next round

    Positions are determined by sorting all committed next_claims by domino_id
    ascending (lower id = earlier pick). Uncommitted positions (fewer than 4
    claims) get 0.0.

    Returns all zeros for INITIAL_SELECTION (opening claims are first-round
    placement commitments, not next-round tempo signals — they are not
    interleaved action positions in the same sense), FINAL_PLACEMENT, and
    GAME_OVER (no next round).

    Antisymmetric: once all four are committed the array holds two +1s and two
    -1s and sums to 0.0; the opponent's perspective is the exact negation, so
    encode_state(state, 0) and encode_state(state, 1) have negated pick_pos.

    Information-set safe: reads only next_claims (player + domino_id, both
    public) and phase; never the hidden deck order.
    """
    out = np.zeros(4, dtype=np.float32)
    if state.phase in (Phase.INITIAL_SELECTION,
                       Phase.FINAL_PLACEMENT,
                       Phase.GAME_OVER):
        return out

    # Sort all committed next_claims by domino_id ascending.
    # Lower domino_id = earlier next-round pick position.
    committed = sorted(state.next_claims, key=lambda c: c.domino_id)
    for k, claim in enumerate(committed):
        if k >= 4:
            break  # safety: never more than 4 claims
        out[k] = 1.0 if claim.player == player else -1.0

    return out


# ─── public API ───────────────────────────────────────────────────────────
def encode_state(state: GameState, player: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode `state` from `player`'s perspective.

    Returns
    -------
    my_board  : (9, 13, 13) float32 — current player's board, castle-centred
    opp_board : (9, 13, 13) float32 — opponent's board, castle-centred
    flat      : (FLAT_SIZE,) float32 — global features (see FLAT_LAYOUT)

    Raises
    ------
    ValueError if state.phase is GAME_OVER (terminal states get their value
    target directly from compute_target_z; they should never be fed to the
    network).
    """
    if state.phase == Phase.GAME_OVER:
        raise ValueError("encode_state is not defined for terminal states; "
                         "use compute_target_z for the value target instead.")
    # Config validation — the spatial shape and game_progress scaling assume
    # 2-player Mighty Duel: 48 dominoes total, 24 placed per player, 96
    # placeable cells in total.  If you add support for standard 2-player
    # rules (where each player places only 12 dominoes) or for 3-4 player
    # variants, this assumption breaks and the encoder needs revisiting.
    if state.config.players != 2 or not state.config.mighty_duel:
        raise ValueError(
            f"This encoder currently assumes 2-player Mighty Duel "
            f"(state.config.players=2, mighty_duel=True). "
            f"Got players={state.config.players}, "
            f"mighty_duel={state.config.mighty_duel}."
        )
    if len(state.boards) != 2:
        raise ValueError(f"Expected 2 boards, got {len(state.boards)}.")
    if not 0 <= player < 2:
        raise ValueError(f"Invalid player index {player}.")
    opponent = 1 - player

    # ── Spatial encodings ──
    my_board  = _encode_board_spatial(state.boards[player])
    opp_board = _encode_board_spatial(state.boards[opponent])

    # ── Flat encoding ──
    flat = np.zeros(FLAT_SIZE, dtype=np.float32)

    # 1. Symmetric pending-placement summaries expose each side's next claimed
    # but unplaced tile, even before that side is the current actor.
    flat[FLAT_LAYOUT['my_next_pending']] = _encode_pending_summary(state, player)
    flat[FLAT_LAYOUT['opp_next_pending']] = _encode_pending_summary(state, opponent)

    # 1b. Rule-derived board summaries.  These are mirrored my/opp facts:
    # scoring components, conservative bonus feasibility, geometry, remaining
    # placement opportunities, and neutral legality for the next pending tile.
    flat[FLAT_LAYOUT['my_board_summary']] = _encode_board_summary(state, player)
    flat[FLAT_LAYOUT['opp_board_summary']] = _encode_board_summary(state, opponent)

    # 2. Current row (visible dominoes available for next-round drafting).
    row_buf = np.zeros(ROW_SLOT_SIZE * MAX_PHASE_SLOTS, dtype=np.float32)
    for i in range(MAX_PHASE_SLOTS):
        did = state.current_row[i] if i < len(state.current_row) else None
        row_buf[i * ROW_SLOT_SIZE:(i + 1) * ROW_SLOT_SIZE] = _encode_row_slot(did)
    flat[FLAT_LAYOUT['current_row']] = row_buf

    # 3. Pending claims (this round's placement order + who placed what).
    #    status_flag = 1.0 if this claim has already been resolved (placed).
    pend_buf = np.zeros(CLAIM_SLOT_SIZE * MAX_PHASE_SLOTS, dtype=np.float32)
    for i in range(MAX_PHASE_SLOTS):
        claim = state.pending_claims[i] if i < len(state.pending_claims) else None
        already_placed = 1.0 if (claim is not None and i < state.actor_index) else 0.0
        pend_buf[i * CLAIM_SLOT_SIZE:(i + 1) * CLAIM_SLOT_SIZE] = (
            _encode_claim_slot(claim, player, already_placed)
        )
    flat[FLAT_LAYOUT['pending_claims']] = pend_buf

    # 4. Next claims (what's been committed for next round so far).
    #    status_flag = 1.0 just means 'slot filled' for next_claims.
    nxt_buf = np.zeros(CLAIM_SLOT_SIZE * MAX_PHASE_SLOTS, dtype=np.float32)
    for i in range(MAX_PHASE_SLOTS):
        claim = state.next_claims[i] if i < len(state.next_claims) else None
        slot_filled = 1.0 if claim is not None else 0.0
        nxt_buf[i * CLAIM_SLOT_SIZE:(i + 1) * CLAIM_SLOT_SIZE] = (
            _encode_claim_slot(claim, player, slot_filled)
        )
    flat[FLAT_LAYOUT['next_claims']] = nxt_buf

    # 5. Bag (information-set safe).
    flat[FLAT_LAYOUT['bag']] = _compute_bag(state)

    # 6. Phase one-hot (GAME_OVER excluded — checked above).
    phase_oh = np.zeros(3, dtype=np.float32)
    phase_oh[int(state.phase)] = 1.0  # Phase indices 0/1/2 align with one-hot positions
    flat[FLAT_LAYOUT['phase']] = phase_oh

    # 7. Game progress: total cells placed (excluding castles) / 96.
    #    Each player places 24 dominoes × 2 cells = 48 cells; total = 96.
    placed_cells = sum(len(b.occupied_cells()) - 1 for b in state.boards)
    flat[FLAT_LAYOUT['game_progress']] = placed_cells / 96.0

    # 8. Per-board fill ratios (compactness → Harmony bonus signal).
    flat[FLAT_LAYOUT['my_fill_ratio']]  = _fill_ratio(state.boards[player])
    flat[FLAT_LAYOUT['opp_fill_ratio']] = _fill_ratio(state.boards[opponent])

    # 9. Actor flag: is the encoded player the one about to act?
    flat[FLAT_LAYOUT['actor_flag']] = 1.0 if state.current_actor == player else 0.0

    # 10. Next-round pick positions (full interleaving, 4 positions; see
    #     _pick_positions).  +1 = encoded player acts here, -1 = opponent,
    #     0 = not yet determined.  All zeros during INITIAL_SELECTION (opening
    #     claims are not next-round tempo) and FINAL_PLACEMENT/GAME_OVER (no
    #     next round).
    pos = _pick_positions(state, player)
    flat[FLAT_LAYOUT['pick_pos_0']] = pos[0]
    flat[FLAT_LAYOUT['pick_pos_1']] = pos[1]
    flat[FLAT_LAYOUT['pick_pos_2']] = pos[2]
    flat[FLAT_LAYOUT['pick_pos_3']] = pos[3]

    return my_board, opp_board, flat


def compute_target_z(state: GameState, player: int, sigma: float = 30.0) -> float:
    """Compute the value-head training target for a terminal state.

    Uses the "Leader" heuristic from Goodman et al. 2023 (FDG): the score
    margin from the encoded player's perspective, soft-normalised through
    tanh.  This blends the binary outcome (sign of the margin → win/loss)
    with the magnitude (a 30-point blowout is a stronger signal than a
    1-point squeaker).  Both Score+ and Leader heuristics beat pure
    win/loss across 10 tabletop games in the paper, with Leader best in
    games that have adversarial counter-moves — which is exactly the
    drafting dynamic in Kingdomino.

    sigma controls the softness of the margin.  Tune by measuring the
    standard deviation of score margins in your training-game distribution
    (a sweep is cheap; 20–40 is a reasonable starting range).

    z is bounded in (-1, 1):
        z → +1  for large wins
        z →  0  for tied / very close games
        z → -1  for large losses

    Defined only for terminal states.  Non-terminal value targets come from
    the network's own predictions via MCTS bootstrapping, not from this
    function.
    """
    if state.phase != Phase.GAME_OVER:
        raise ValueError(
            f"compute_target_z is defined for terminal states only; "
            f"got state.phase={state.phase.name}.  Non-terminal value "
            f"targets come from MCTS bootstrapping, not from this function."
        )
    if state.config.players != 2 or not state.config.mighty_duel:
        raise ValueError(
            f"compute_target_z currently assumes 2-player Mighty Duel "
            f"(state.config.players=2, mighty_duel=True). "
            f"Got players={state.config.players}, "
            f"mighty_duel={state.config.mighty_duel}."
        )
    if not 0 <= player < 2:
        raise ValueError(f"Invalid player index {player}.")
    scores = state.scores()
    opponent = 1 - player
    margin = scores[player] - scores[opponent]
    return math.tanh(margin / sigma)


def compute_target_own_score(state: GameState, player: int) -> float:
    """Encoded player's RAW final score at a terminal state, as a float.

    Returns the integer board score (territory + harmony + middle-kingdom)
    unnormalized. The network's own_score_head predicts the NORMALIZED score
    (raw / score_scale); normalization by score_scale is the caller's
    responsibility at train time, not here — this keeps the engine target a
    plain game quantity independent of any network hyperparameter.

    Terminal-state only (mirrors compute_target_z); 2-player Mighty Duel only.
    """
    if state.phase != Phase.GAME_OVER:
        raise ValueError(
            f"compute_target_own_score is defined for terminal states only; "
            f"got state.phase={state.phase.name}."
        )
    if state.config.players != 2 or not state.config.mighty_duel:
        raise ValueError(
            f"compute_target_own_score currently assumes 2-player Mighty Duel "
            f"(state.config.players=2, mighty_duel=True). "
            f"Got players={state.config.players}, "
            f"mighty_duel={state.config.mighty_duel}."
        )
    if not 0 <= player < 2:
        raise ValueError(f"Invalid player index {player}.")
    scores = state.scores()
    return float(scores[player])


def compute_target_opponent_score(state: GameState, player: int) -> float:
    """Opponent's RAW final score at a terminal state, as a float.

    Same contract as compute_target_own_score but for the opponent of `player`.
    Returns the unnormalized integer board score; normalization by score_scale
    is the caller's responsibility at train time, not here.

    Terminal-state only (mirrors compute_target_z); 2-player Mighty Duel only.
    """
    if state.phase != Phase.GAME_OVER:
        raise ValueError(
            f"compute_target_opponent_score is defined for terminal states "
            f"only; got state.phase={state.phase.name}."
        )
    if state.config.players != 2 or not state.config.mighty_duel:
        raise ValueError(
            f"compute_target_opponent_score currently assumes 2-player Mighty "
            f"Duel (state.config.players=2, mighty_duel=True). "
            f"Got players={state.config.players}, "
            f"mighty_duel={state.config.mighty_duel}."
        )
    if not 0 <= player < 2:
        raise ValueError(f"Invalid player index {player}.")
    scores = state.scores()
    return float(scores[1 - player])


def compute_target_win(state: GameState, player: int) -> float:
    """Win/draw/loss target for `player` at a terminal state.

    Returns 1.0 if `player` won, 0.0 if they lost, 0.5 for a draw, applying the
    full official tiebreaker cascade via game.determine_winner (score → largest
    territory → total crowns → draw). Unlike compute_target_z (a soft score
    margin), this is the discrete game outcome that correctly resolves score
    ties.

    Terminal-state only — raises ValueError otherwise, mirroring
    compute_target_z. Non-terminal value targets come from MCTS bootstrapping.
    """
    if state.phase != Phase.GAME_OVER:
        raise ValueError(
            f"compute_target_win is defined for terminal states only; "
            f"got state.phase={state.phase.name}."
        )
    if state.config.players != 2 or not state.config.mighty_duel:
        raise ValueError(
            f"compute_target_win assumes 2-player Mighty Duel "
            f"(state.config.players=2, mighty_duel=True). "
            f"Got players={state.config.players}, "
            f"mighty_duel={state.config.mighty_duel}."
        )
    if not 0 <= player < 2:
        raise ValueError(f"Invalid player index {player}.")
    winner = determine_winner(state)
    if winner is None:
        return 0.5
    return 1.0 if winner == player else 0.0


def redeterminize(state: GameState, rng: random.Random) -> GameState:
    """Return a copy of `state` with the hidden deck reshuffled.

    Bag membership is preserved (only order changes), so every piece of
    public information — boards, current_row, claims, scores, phase — is
    byte-identical to the input.  By the encoder's information-set
    contract, `encode_state(state, p)` and `encode_state(redeterminize(s),
    p)` produce identical tensors.

    This is the operation that closes the information-set loop on the
    *search* side.  Calling it at the root of each MCTS search (and
    optionally at chance-reveal boundaries inside the tree) ensures the
    search cannot benefit from knowing the true deck order of the real
    game — a leak the encoder alone cannot prevent.

    This is the simplest determinization strategy (often called "single
    determinization" or PIMC).  Higher-quality alternatives exist —
    multiple determinizations per decision averaged together (IS-MCTS),
    or full information-set MCTS that doesn't commit to a single
    determinization — but all of them use this primitive as their building
    block.  The full design lives in the search code; the encoder only
    provides the primitive.
    """
    new_state = state.copy()
    rng.shuffle(new_state.deck)
    return new_state


# ─── self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Module smoke test — runs when this file is executed directly.
    import random
    from games.kingdomino.game import GameState

    print(f"FLAT_SIZE = {FLAT_SIZE}")
    print("FLAT_LAYOUT offsets:")
    for k, sl in FLAT_LAYOUT.items():
        print(f"  {k:18s} {sl.start:4d}..{sl.stop:<4d} (size {sl.stop - sl.start})")

    rng = random.Random(0)
    state = GameState.new(seed=0)

    # Initial-selection phase encoding
    mb, ob, flat = encode_state(state, player=0)
    assert mb.shape == (9, 13, 13) and ob.shape == (9, 13, 13)
    assert flat.shape == (FLAT_SIZE,)
    print(f"\nfresh state | bag count = {int(flat[FLAT_LAYOUT['bag']].sum())} "
          f"(expected 44 = 48 - 4 in current_row)")
    print(f"  castle mask sum (my)  = {mb[CH_CASTLE].sum()} (expect 1.0)")
    print(f"  occupied sum (my)     = {mb[CH_OCCUPIED].sum()} (expect 1.0; just castle)")
    print(f"  terrain plane sum (my)= {mb[:CH_TERRAIN_END].sum()} (expect 0.0)")
    print(f"  phase one-hot         = {flat[FLAT_LAYOUT['phase']]}")
    print(f"  actor_flag (p0)       = {flat[FLAT_LAYOUT['actor_flag']][0]}")
    print(f"  game_progress         = {flat[FLAT_LAYOUT['game_progress']][0]:.4f}")
    print(f"  pick_pos_0..3 = {flat[FLAT_LAYOUT['pick_pos_0']][0]:.1f} "
          f"{flat[FLAT_LAYOUT['pick_pos_1']][0]:.1f} "
          f"{flat[FLAT_LAYOUT['pick_pos_2']][0]:.1f} "
          f"{flat[FLAT_LAYOUT['pick_pos_3']][0]:.1f} "
          f"(expect 0.0 0.0 0.0 0.0 at fresh INITIAL_SELECTION state)")
    print("OK")
