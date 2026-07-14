"""D4 geometric transforms on a Kingdomino GameState + induced feature-index maps.

The castle sits at the exact centre (7,7) of the 15x15 canvas, so the eight D4
symmetries (4 rotations x optional reflection) act as `np.rot90`/`np.fliplr` about
the array centre and keep the castle fixed. A domino half keeps its terrain and
crown count under any D4 op, so the ONLY thing that moves is the cell it occupies.

This module provides the group action on the *state* and the matching permutations
on the encoder outputs, so a test can assert the augmentation identity

    encode(transform(s))  ==  Dmap(encode(s))

for both the 5,710-value sparse core and the 171-value summary. The permutations
are derived from the SAME numpy ops that transform the board (via a labelled-cell
probe), so they cannot silently disagree with the transform.
"""
from __future__ import annotations

import numpy as np

from .sparse_encoder import (
    CELL_SIDE, NUM_CELLS, NUM_HALF, NUM_ROLES, BOARD_OFF, BOARD_SIZE, CORE_SIZE,
    _MIN, _MAX, CASTLE, _cell_index,
)

# 8 elements as (num_rot90 k, reflect flip_lr). Reflection last: fliplr(rot90(a,k)).
D4_ELEMENTS: list[tuple[int, bool]] = [(k, f) for f in (False, True) for k in range(4)]


def _apply_grid(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, k)
    if flip:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def coord_fwd(k: int, flip: bool) -> dict:
    """Map (x,y) -> (x',y') for every reachable cell (castle fixed), derived from
    the SAME numpy ops so it can't drift from the encoders' cell permutation."""
    lab = np.full((CASTLE * 2 + 1, CASTLE * 2 + 1), -1, dtype=np.int64)
    for y in range(_MIN, _MAX + 1):
        for x in range(_MIN, _MAX + 1):
            lab[y, x] = x * 100 + y            # y <= 13 < 100, so decodable
    lab_t = _apply_grid(lab, k, flip)
    fwd = {}
    for y in range(_MIN, _MAX + 1):
        for x in range(_MIN, _MAX + 1):
            ox, oy = divmod(int(lab_t[y, x]), 100)   # tile now at (x,y) came from (ox,oy)
            fwd[(ox, oy)] = (x, y)
    return fwd


# ── group action on the state ────────────────────────────────────────────────
def transform_board(board, k: int, flip: bool):
    """Return a new Board with every placed half moved to its D4 image. The Board
    keeps occupancy/bbox caches as the source of truth (not just the terrain
    array), so we rebuild from coordinates rather than rotating arrays. Adjacency,
    connectivity and the castle position are preserved -> still a legal kingdom."""
    b = type(board)(board.canvas_size, board.castle_pos)   # fresh: castle placed
    fwd = coord_fwd(k, flip)
    cx, cy = board.castle_pos
    minx = maxx = cx
    miny = maxy = cy
    for (x, y) in board.occupied_cells():
        if (x, y) == (cx, cy):
            continue                                       # castle already placed
        nx, ny = fwd[(x, y)]
        t = int(board.terrain[y, x])
        b.terrain[ny, nx] = t
        b.crowns[ny, nx] = board.crowns[y, x]
        b.domino_id[ny, nx] = board.domino_id[y, x]
        b._occupied.add((nx, ny))
        b._cell[(nx, ny)] = t
        minx, maxx = min(minx, nx), max(maxx, nx)
        miny, maxy = min(miny, ny), max(maxy, ny)
    b._min_x, b._max_x, b._min_y, b._max_y = minx, maxx, miny, maxy
    return b


def transform_state(state, k: int, flip: bool):
    """Apply a D4 element to every player's board; the rest of the state (deck,
    claims, phase, actor, discards, rules) is geometry-independent and unchanged."""
    s = state.copy()
    s.boards = [transform_board(b, k, flip) for b in state.boards]
    return s


# ── induced cell permutation (empirical, matches the numpy ops exactly) ───────
def cell_perm(k: int, flip: bool) -> np.ndarray:
    """`fwd[c] = c'`: the reachable-cell index a tile at cell `c` moves to under
    (k, flip). Built by transforming a grid of cell labels with the same ops."""
    label = np.full((CASTLE * 2 + 1, CASTLE * 2 + 1), -1, dtype=np.int64)
    for y in range(_MIN, _MAX + 1):
        for x in range(_MIN, _MAX + 1):
            label[y, x] = _cell_index(x, y)
    lab_t = _apply_grid(label, k, flip)
    fwd = np.full(NUM_CELLS, -1, dtype=np.int64)
    for y in range(_MIN, _MAX + 1):
        for x in range(_MIN, _MAX + 1):
            c_src = int(lab_t[y, x])          # tile now sitting at (x,y) came from c_src
            fwd[c_src] = _cell_index(x, y)     # ...and lands on this reachable cell
    assert (fwd >= 0).all() and len(set(fwd.tolist())) == NUM_CELLS
    return fwd


def sparse_perm(k: int, flip: bool) -> np.ndarray:
    """Permutation `p` over the 5,710 core indices s.t.
    encode_core(transform(s)) == { p[i] for i in encode_core(s) }.
    Board indices permute by cell; every other bank is geometry-invariant."""
    p = np.arange(CORE_SIZE, dtype=np.int64)
    fwd = cell_perm(k, flip)
    for role in range(NUM_ROLES):
        base = role * NUM_CELLS
        for c in range(NUM_CELLS):
            src = BOARD_OFF + (base + c) * NUM_HALF
            dst = BOARD_OFF + (base + int(fwd[c])) * NUM_HALF
            for h in range(NUM_HALF):
                p[src + h] = dst + h
    return p


def apply_sparse(indices, k: int, flip: bool) -> set:
    p = sparse_perm(k, flip)
    return {int(p[i]) for i in indices}


# ── direction / axis maps for the summary ────────────────────────────────────
# castle_extent order is [L=-x, R=+x, U=-y, D=+y]; base block carries width,height.
_DIR_PROBE = {0: (CASTLE - 1, CASTLE), 1: (CASTLE + 1, CASTLE),
              2: (CASTLE, CASTLE - 1), 3: (CASTLE, CASTLE + 1)}


def _dir_map(k: int, flip: bool) -> list[int]:
    """dir_map[old] = new: which of {0:L,1:R,2:U,3:D} an original direction maps to."""
    fwd = cell_perm(k, flip)
    out = [-1, -1, -1, -1]
    for old, (px, py) in _DIR_PROBE.items():
        c_new = int(fwd[_cell_index(px, py)])
        nx, ny = c_new % CELL_SIDE + _MIN, c_new // CELL_SIDE + _MIN
        if nx < CASTLE:
            out[old] = 0
        elif nx > CASTLE:
            out[old] = 1
        elif ny < CASTLE:
            out[old] = 2
        else:
            out[old] = 3
    assert sorted(out) == [0, 1, 2, 3]
    return out


def axis_swap(k: int, flip: bool) -> bool:
    """True iff the transform swaps the x/y axes (width<->height); i.e. an L probe
    lands on a vertical direction. Equivalent to k being odd, derived empirically."""
    return _dir_map(k, flip)[0] in (2, 3)


def summary_perm(k: int, flip: bool):
    """Return a callable Dmap: given a 171-vector v = encode_summary(s), produce the
    vector encode_summary should yield for transform(s). Width/height swap under an
    axis swap; castle_extent directions permute; everything else is invariant."""
    from .summary_encoder import BASE_SIZE, EXT_PER

    swap = axis_swap(k, flip)
    dm = _dir_map(k, flip)
    ext_castle = [29, 30, 31, 32]           # L,R,U,D within a 39-value extension block
    ext_starts = [BASE_SIZE, BASE_SIZE + EXT_PER]     # my, opp extension blocks
    base_wh = [(20, 21), (25 + 20, 25 + 21)]          # (width,height) per 25-value base sub-block

    def dmap(v: np.ndarray) -> np.ndarray:
        w = v.copy()
        if swap:
            for wi, hi in base_wh:
                w[wi], w[hi] = v[hi], v[wi]
        for start in ext_starts:
            src = [v[start + ext_castle[old]] for old in range(4)]
            for old in range(4):
                w[start + ext_castle[dm[old]]] = src[old]
        return w

    return dmap
