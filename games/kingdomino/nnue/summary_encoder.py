"""171-value real-valued SUMMARY for the Kingdomino NNUE (Step 3).

Recomputed per node and concatenated at the tail (it bypasses the incremental
accumulator). Perspective-relative: every per-player block is ordered
[my (perspective), opponent]; seat-independent blocks stay global but any owner
indicator is my/opp, never absolute P0/P1.

Layout (frozen; see NNUE_STEP3_FEATURES.md, definitions APPROVED 2026-07-13):
  base       50  = _encode_board_summary x [my, opp]  (25 each, reused verbatim)
  extension  78  = 39 x [my, opp]  (region flood-fill features)
  global     43  = bag aggregates(12) + unresolved claims(24) + pick_pos(4)
                   + game_progress + fill_ratio my/opp (3)
  total     171

Normalizations are FIXED catalog/rules constants (never pilot-derived) and are in
summary_schema_hash(). Range: values in [-1, 1] (owner/pick_pos are -1/0/+1, the
rest in [0, 1]); test_summary_encoder asserts ranges and measures clip frequency.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from games.kingdomino.game import Phase
from games.kingdomino.dominoes import DOMINOES
from games.kingdomino.encoder import (
    _encode_board_summary, SCORE_SCALE, MAX_BOARD_CELLS, MAX_TOTAL_CROWNS,
    MAX_LEGAL_PLACEMENTS, NUM_PLACEABLE_TERRAINS, TERRAIN_INDEX_OFFSET,
)

NT = NUM_PLACEABLE_TERRAINS          # 6
TOFF = TERRAIN_INDEX_OFFSET          # 2 (WHEAT); placeable terrain codes 2..7
CASTLE = 7                           # board grid coord of the castle on each axis
_MIN, _MAX = CASTLE - 6, CASTLE + 6  # reachable board indices 1..13
BBOX_MAX = 7                         # 7x7 kingdom

# Catalog-derived fixed normalization maxima (true upper bounds -> no clipping).
MAX_CROWNS_PER_TERRAIN = [0] * NT
MAX_HALVES_PER_TERRAIN = [0] * NT
for _d in DOMINOES.values():
    for _h in (_d.a, _d.b):
        _ti = int(_h.terrain) - TOFF
        MAX_CROWNS_PER_TERRAIN[_ti] += int(_h.crowns)
        MAX_HALVES_PER_TERRAIN[_ti] += 1
# Rules-derived caps (generous true maxima; clip frequency asserted 0 in tests).
CAP_REGION_COUNT = 24.0
CAP_FRONTIER = 24.0
MAX_TURN_DISTANCE = 3.0

BASE_SIZE = 50
EXT_PER = 39
GLOBAL_SIZE = 43
SUMMARY_SIZE = BASE_SIZE + 2 * EXT_PER + GLOBAL_SIZE   # 171
assert SUMMARY_SIZE == 171

_NEI = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _flood_regions(board):
    """List of (terrain_idx, size, crowns) for every connected same-terrain region."""
    terr = np.asarray(board.terrain)
    cr = np.asarray(board.crowns)
    H, W = terr.shape
    seen = np.zeros((H, W), bool)
    regions = []
    for y in range(H):
        for x in range(W):
            t = int(terr[y, x])
            if seen[y, x] or t < TOFF:      # skip EMPTY(0) and CASTLE(1)
                continue
            stack = [(x, y)]
            seen[y, x] = True
            size = crowns = 0
            while stack:
                cx, cy = stack.pop()
                size += 1
                crowns += int(cr[cy, cx])
                for dx, dy in _NEI:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < W and 0 <= ny < H and not seen[ny, nx] and int(terr[ny, nx]) == t:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            regions.append((t - TOFF, size, crowns))
    return regions


def _extension_raw(board) -> dict:
    """Un-normalized extension features for one board."""
    terr = np.asarray(board.terrain)
    H, W = terr.shape
    regions = _flood_regions(board)

    cell_count = [0] * NT
    crown_count = [0] * NT
    largest_size = [0] * NT
    largest_crowns = [0] * NT                 # crowns of the max-AREA region (tie: max crowns)
    largest_crownless = [0] * NT
    crownless_region_count = 0
    stranded_crowns = 0
    global_largest = 0
    for ti, size, crowns in regions:
        cell_count[ti] += size
        crown_count[ti] += crowns
        global_largest = max(global_largest, size)
        if size > largest_size[ti] or (size == largest_size[ti] and crowns > largest_crowns[ti]):
            largest_size[ti] = size
            largest_crowns[ti] = crowns
        if crowns == 0:
            crownless_region_count += 1
            largest_crownless[ti] = max(largest_crownless[ti], size)
        if size == 1:
            stranded_crowns += crowns

    bbox = board.occupied_bbox()
    cx, cy = board.castle_pos
    if bbox is None:
        minx = maxx = cx
        miny = maxy = cy
    else:
        minx, miny, maxx, maxy = bbox
    occ = board.occupied_cells()

    # open frontier: unique empty, in-bounds, bbox-admissible cells adjacent to a terrain.
    frontier = [set() for _ in range(NT)]
    for (x, y) in occ:
        t = int(terr[y, x])
        if t < TOFF:
            continue
        for dx, dy in _NEI:
            nx, ny = x + dx, y + dy
            if not (_MIN <= nx <= _MAX and _MIN <= ny <= _MAX):
                continue
            if int(terr[ny, nx]) != 0:
                continue
            if (max(maxx, nx) - min(minx, nx) + 1) <= BBOX_MAX and \
               (max(maxy, ny) - min(miny, ny) + 1) <= BBOX_MAX:
                frontier[t - TOFF].add((nx, ny))
    open_frontier = [len(f) for f in frontier]

    # enclosed single holes: empty cells inside bbox with all 4 neighbors occupied.
    holes = 0
    for y in range(miny, maxy + 1):
        for x in range(minx, maxx + 1):
            if int(terr[y, x]) == 0 and all(int(terr[y + dy, x + dx]) != 0 for dx, dy in _NEI):
                holes += 1

    gaps = (maxx - minx + 1) * (maxy - miny + 1) - len(occ)
    castle_extent = [cx - minx, maxx - cx, cy - miny, maxy - cy]

    return dict(cell_count=cell_count, crown_count=crown_count, largest_crowns=largest_crowns,
                global_largest=global_largest, crownless_region_count=crownless_region_count,
                stranded_crowns=stranded_crowns, open_frontier=open_frontier,
                holes=holes, gaps=gaps, castle_extent=castle_extent,
                largest_crownless=largest_crownless)


def _extension_vec(board) -> np.ndarray:
    r = _extension_raw(board)
    out = []
    out += [c / MAX_BOARD_CELLS for c in r["cell_count"]]                       # 6
    out += [r["crown_count"][t] / MAX_CROWNS_PER_TERRAIN[t] for t in range(NT)]  # 6
    out += [r["largest_crowns"][t] / MAX_CROWNS_PER_TERRAIN[t] for t in range(NT)]  # 6
    out.append(r["global_largest"] / MAX_BOARD_CELLS)                          # 1
    out.append(r["crownless_region_count"] / CAP_REGION_COUNT)                 # 1
    out.append(r["stranded_crowns"] / MAX_TOTAL_CROWNS)                        # 1
    out += [f / CAP_FRONTIER for f in r["open_frontier"]]                      # 6
    out.append(r["holes"] / MAX_BOARD_CELLS)                                   # 1
    out.append(r["gaps"] / MAX_BOARD_CELLS)                                    # 1
    out += [e / 6.0 for e in r["castle_extent"]]                              # 4
    out += [c / MAX_BOARD_CELLS for c in r["largest_crownless"]]               # 6
    v = np.array(out, dtype=np.float32)
    assert len(v) == EXT_PER
    return v


def _bag_terrain(state) -> np.ndarray:
    halfcount = [0] * NT
    crowns = [0] * NT
    for did in state.deck:
        d = DOMINOES[did]
        for h in (d.a, d.b):
            ti = int(h.terrain) - TOFF
            halfcount[ti] += 1
            crowns[ti] += int(h.crowns)
    out = [halfcount[t] / MAX_HALVES_PER_TERRAIN[t] for t in range(NT)]
    out += [crowns[t] / MAX_CROWNS_PER_TERRAIN[t] for t in range(NT)]
    return np.array(out, dtype=np.float32)


def _unresolved_claims(state, perspective) -> np.ndarray:
    """Up to 4 unresolved claims in fixed action order, 6 values each (owner my/opp)."""
    out = np.zeros(4 * 6, dtype=np.float32)
    if state.phase in (Phase.PLACE_AND_SELECT, Phase.FINAL_PLACEMENT):
        tail = state.pending_claims[state.actor_index:]
        for k, claim in enumerate(tail[:4]):
            board = state.boards[claim.player]
            legal = len(board.legal_placements(DOMINOES[claim.domino_id]))
            base = k * 6
            out[base + 0] = 1.0                                              # presence
            out[base + 1] = min(legal, MAX_LEGAL_PLACEMENTS) / MAX_LEGAL_PLACEMENTS
            out[base + 2] = 1.0 if legal == 0 else 0.0                       # forced discard
            out[base + 3] = 1.0 if claim.player == perspective else -1.0     # owner (my/opp)
            out[base + 4] = (claim.domino_id - 1) / (len(DOMINOES) - 1)      # draft-priority rank
            out[base + 5] = min(k, MAX_TURN_DISTANCE) / MAX_TURN_DISTANCE    # turn distance
    return out


def _pick_pos(state, perspective) -> np.ndarray:
    """Next-round draft order: element k = owner of next-round pick k (+1 my/-1 opp/0)."""
    out = np.zeros(4, dtype=np.float32)
    for k, claim in enumerate(sorted(state.next_claims, key=lambda c: c.domino_id)[:4]):
        out[k] = 1.0 if claim.player == perspective else -1.0
    return out


def _progress_and_fill(state, perspective) -> np.ndarray:
    opp = 1 - perspective
    placed_halves = [0, 0]
    for p in (0, 1):
        g = np.asarray(state.boards[p].domino_id)
        placed_halves[p] = int((g > 0).sum())
    total_placed_dominoes = sum(placed_halves) / 2.0
    game_progress = (total_placed_dominoes + sum(state.discards)) / len(DOMINOES)

    def fill(p):
        b = state.boards[p]
        bbox = b.occupied_bbox()
        if bbox is None:
            return 0.0
        minx, miny, maxx, maxy = bbox
        area = (maxx - minx + 1) * (maxy - miny + 1)
        return placed_halves[p] / area if area else 0.0

    return np.array([game_progress, fill(perspective), fill(opp)], dtype=np.float32)


def encode_summary(state, perspective: int) -> np.ndarray:
    """The 171-value perspective-relative summary vector (float32)."""
    if perspective not in (0, 1):
        raise ValueError(f"perspective must be 0 or 1, got {perspective}")
    opp = 1 - perspective
    parts = [
        _encode_board_summary(state, perspective),   # base my (25)
        _encode_board_summary(state, opp),            # base opp (25)
        _extension_vec(state.boards[perspective]),    # ext my (39)
        _extension_vec(state.boards[opp]),            # ext opp (39)
        _bag_terrain(state),                          # 12
        _unresolved_claims(state, perspective),       # 24
        _pick_pos(state, perspective),                # 4
        _progress_and_fill(state, perspective),       # 3
    ]
    v = np.concatenate(parts).astype(np.float32)
    assert len(v) == SUMMARY_SIZE, f"summary size {len(v)} != {SUMMARY_SIZE}"
    return v


def summary_schema_hash() -> str:
    spec = {
        "size": SUMMARY_SIZE, "base": BASE_SIZE, "ext_per": EXT_PER, "global": GLOBAL_SIZE,
        "score_scale": SCORE_SCALE, "max_board_cells": MAX_BOARD_CELLS,
        "max_total_crowns": MAX_TOTAL_CROWNS, "max_legal": MAX_LEGAL_PLACEMENTS,
        "max_crowns_per_terrain": MAX_CROWNS_PER_TERRAIN,
        "max_halves_per_terrain": MAX_HALVES_PER_TERRAIN,
        "cap_region_count": CAP_REGION_COUNT, "cap_frontier": CAP_FRONTIER,
        "max_turn_distance": MAX_TURN_DISTANCE, "bbox_max": BBOX_MAX,
        "order": "base[my,opp] ext[my,opp] bag claims pick_pos progress_fill",
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
