"""Sparse feature encoder for the Kingdomino NNUE (Step 3) — the LOSSLESS CORE.

Encodes the future-relevant public Markov state as a set of active binary feature
indices, in a PERSPECTIVE-RELATIVE frame (roles are my/opp w.r.t. the perspective
player, never absolute P0/P1). This is the 5,710-feature core from
NNUE_STEP3_FEATURES.md; the 171-value real-valued summary is a separate module.

Design contracts enforced by tests (test_sparse_encoder.py):
  * seat-swap: encode(s, P0) == encode(swap_players(s), P1)   [structural symmetry]
  * fingerprint: decode(encode(s, persp)) == public_fingerprint(s, persp)  [lossless]
  * one-field mutation changes the encoding                    [completeness]
  * inventory: row/pending/next/bag banks agree with the 48-ID conservation ledger

Reads a Python GameState (rich accessors, easy to construct/swap for the gates).
A RustGameState path (for replay-time feature derivation in training) mirrors the
same banks and is validated to match.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from games.kingdomino.game import Phase, Claim
from games.kingdomino.dominoes import DOMINOES

# ── catalog-derived schema constants (frozen; part of the schema hash) ────────
_IDS = sorted(DOMINOES)                       # 1..48, contiguous
NUM_DOMINOES = len(_IDS)                       # 48
assert _IDS == list(range(1, NUM_DOMINOES + 1))

# The 16 (terrain, crowns) half-types that actually occur, sorted -> 0..15.
HALF_TYPES = sorted({(int(h.terrain), int(h.crowns))
                     for d in DOMINOES.values() for h in (d.a, d.b)})
HALF_IDX = {ht: i for i, ht in enumerate(HALF_TYPES)}
NUM_HALF = len(HALF_TYPES)                     # 16

# Board geometry: 7x7 kingdom, castle can sit at a bbox edge -> occupied cells
# reach +-6 from the castle. The Python Board grid is 15x15 with castle at (7,7);
# reachable board indices are 1..13 (offset -6..+6) -> a 13x13 = 169 relative grid.
CELL_SIDE = 13
NUM_CELLS = CELL_SIDE * CELL_SIDE              # 169
CASTLE = 7                                     # board index of the castle on each axis
_MIN, _MAX = CASTLE - 6, CASTLE + 6            # 1..13 inclusive (reachable)

NUM_ROLES = 2                                  # 0 = my (perspective), 1 = opp
_CASTLE_TERRAIN = 1                            # occupied_cells() includes the castle; skip it


def _placed_halves(board):
    """(x, y, half_type_index) for each PLACED domino half (castle excluded)."""
    for (x, y) in board.occupied_cells():
        t = int(board.terrain[y, x])
        if t == _CASTLE_TERRAIN:
            continue
        yield x, y, HALF_IDX[(t, int(board.crowns[y, x]))]

# ── bank layout (cumulative offsets) ─────────────────────────────────────────
BOARD_OFF = 0
BOARD_SIZE = NUM_ROLES * NUM_CELLS * NUM_HALF  # 2*169*16 = 5408
ROW_OFF = BOARD_OFF + BOARD_SIZE               # current-row ID membership (48)
PENDING_OFF = ROW_OFF + NUM_DOMINOES           # owner x ID, unresolved only (96)
NEXT_OFF = PENDING_OFF + NUM_ROLES * NUM_DOMINOES   # owner x ID (96)
BAG_OFF = NEXT_OFF + NUM_ROLES * NUM_DOMINOES  # deck ID membership (48)
PHASE_OFF = BAG_OFF + NUM_DOMINOES             # one-hot incl. terminal (4)
ACTOR_OFF = PHASE_OFF + 4                      # my_to_move / opp_to_move (2)
SLOT_OFF = ACTOR_OFF + 2                       # pick-order slot within round (4)
DISC_OFF = SLOT_OFF + 4                        # discard/harmony-lost flag my/opp (2)
RULES_OFF = DISC_OFF + 2                       # harmony_enabled, middle_kingdom_enabled (2)
CORE_SIZE = RULES_OFF + 2                      # 5710


def _cell_index(x: int, y: int) -> int:
    """Castle-relative cell index in [0, 169). x, y are Board grid coords."""
    if not (_MIN <= x <= _MAX and _MIN <= y <= _MAX):
        raise ValueError(f"occupied cell ({x},{y}) outside the reachable 13x13 region")
    return (y - _MIN) * CELL_SIDE + (x - _MIN)


def _turn_slot(state) -> int:
    if state.phase == Phase.INITIAL_SELECTION:
        return min(state.initial_pick_count, 3)
    if state.phase in (Phase.PLACE_AND_SELECT, Phase.FINAL_PLACEMENT):
        return min(state.actor_index, 3)
    return 0  # terminal


def encode_core(state, perspective: int) -> np.ndarray:
    """Active sparse feature indices for `state` from `perspective`'s frame,
    returned sorted and unique (int32)."""
    opp = 1 - perspective
    idx: list[int] = []

    # Board cells: one feature per placed half-cell, (role, cell, half_type).
    for role, p in ((0, perspective), (1, opp)):
        for x, y, half in _placed_halves(state.boards[p]):
            idx.append(BOARD_OFF + (role * NUM_CELLS + _cell_index(x, y)) * NUM_HALF + half)

    # Current-row ID membership.
    for did in state.current_row:
        idx.append(ROW_OFF + (did - 1))

    # Pending claims: UNRESOLVED tail only ([actor_index:]); resolved ones are placed.
    for claim in state.pending_claims[state.actor_index:]:
        role = 0 if claim.player == perspective else 1
        idx.append(PENDING_OFF + role * NUM_DOMINOES + (claim.domino_id - 1))

    # Next-round claims (owner x ID).
    for claim in state.next_claims:
        role = 0 if claim.player == perspective else 1
        idx.append(NEXT_OFF + role * NUM_DOMINOES + (claim.domino_id - 1))

    # Bag (hidden deck) membership -- the SET is public, only order is hidden.
    for did in state.deck:
        idx.append(BAG_OFF + (did - 1))

    # Phase one-hot (incl. terminal).
    idx.append(PHASE_OFF + int(state.phase))

    # Actor one-hot (only meaningful pre-terminal).
    if state.phase != Phase.GAME_OVER:
        idx.append(ACTOR_OFF + (0 if state.current_actor == perspective else 1))

    # Turn slot within the round.
    idx.append(SLOT_OFF + _turn_slot(state))

    # Discard/harmony-lost flag (a discard is invisible on the board yet kills Harmony).
    for role, p in ((0, perspective), (1, opp)):
        if state.discards[p] > 0:
            idx.append(DISC_OFF + role)

    # Rules-config flags (schema is not silently bound to one variant).
    if state.config.harmony:
        idx.append(RULES_OFF + 0)
    if state.config.middle_kingdom:
        idx.append(RULES_OFF + 1)

    out = np.array(sorted(set(idx)), dtype=np.int32)
    if len(out) != len(idx):
        raise AssertionError("duplicate active feature index (banks overlap or double-emit)")
    return out


@dataclass(frozen=True)
class Fingerprint:
    """Canonical public state recovered from an index set (for the lossless gate)."""
    board_my: tuple      # sorted ((cell, half), ...)
    board_opp: tuple
    row: tuple           # sorted domino ids
    pending: tuple       # sorted ((role, id), ...)
    next_claims: tuple
    bag: tuple
    phase: int
    actor: int           # 0 my / 1 opp / -1 terminal (no actor)
    slot: int
    discards: tuple      # sorted roles with a discard
    rules: tuple         # sorted rule flags set


def decode(indices) -> Fingerprint:
    """Reconstruct the canonical public fingerprint from an active index set."""
    board = {0: [], 1: []}
    row, bag = [], []
    pending, nxt = [], []
    phase = actor = slot = None
    disc, rules = [], []
    for i in map(int, indices):
        if i < ROW_OFF:
            j = i - BOARD_OFF
            role, rem = divmod(j, NUM_CELLS * NUM_HALF)
            cell, half = divmod(rem, NUM_HALF)
            board[role].append((cell, half))
        elif i < PENDING_OFF:
            row.append(_IDS[i - ROW_OFF])
        elif i < NEXT_OFF:
            role, k = divmod(i - PENDING_OFF, NUM_DOMINOES)
            pending.append((role, _IDS[k]))
        elif i < BAG_OFF:
            role, k = divmod(i - NEXT_OFF, NUM_DOMINOES)
            nxt.append((role, _IDS[k]))
        elif i < PHASE_OFF:
            bag.append(_IDS[i - BAG_OFF])
        elif i < ACTOR_OFF:
            phase = i - PHASE_OFF
        elif i < SLOT_OFF:
            actor = i - ACTOR_OFF
        elif i < DISC_OFF:
            slot = i - SLOT_OFF
        elif i < RULES_OFF:
            disc.append(i - DISC_OFF)
        else:
            rules.append(i - RULES_OFF)
    return Fingerprint(
        board_my=tuple(sorted(board[0])), board_opp=tuple(sorted(board[1])),
        row=tuple(sorted(row)), pending=tuple(sorted(pending)),
        next_claims=tuple(sorted(nxt)), bag=tuple(sorted(bag)),
        phase=phase, actor=(-1 if actor is None else actor), slot=slot,
        discards=tuple(sorted(disc)), rules=tuple(sorted(rules)),
    )


def public_fingerprint(state, perspective: int) -> Fingerprint:
    """The engine's public state expressed in the same canonical form as decode(),
    read directly from the GameState (the independent side of the lossless gate)."""
    opp = 1 - perspective
    board = {0: [], 1: []}
    for role, p in ((0, perspective), (1, opp)):
        for x, y, half in _placed_halves(state.boards[p]):
            board[role].append((_cell_index(x, y), half))
    pending = [(0 if c.player == perspective else 1, c.domino_id)
               for c in state.pending_claims[state.actor_index:]]
    nxt = [(0 if c.player == perspective else 1, c.domino_id) for c in state.next_claims]
    actor = -1 if state.phase == Phase.GAME_OVER else (0 if state.current_actor == perspective else 1)
    disc = [role for role, p in ((0, perspective), (1, opp)) if state.discards[p] > 0]
    rules = ([0] if state.config.harmony else []) + ([1] if state.config.middle_kingdom else [])
    return Fingerprint(
        board_my=tuple(sorted(board[0])), board_opp=tuple(sorted(board[1])),
        row=tuple(sorted(state.current_row)), pending=tuple(sorted(pending)),
        next_claims=tuple(sorted(nxt)), bag=tuple(sorted(state.deck)),
        phase=int(state.phase), actor=actor, slot=_turn_slot(state),
        discards=tuple(sorted(disc)), rules=tuple(sorted(rules)),
    )


def swap_players(state):
    """Return a copy of `state` with the two players swapped everywhere: boards,
    claim owners (pending + next), discards, and start player. actor_index is a
    position in the pending order and is unchanged (current_actor flips because the
    claim owners flipped). The seat-swap gate requires this to be exact."""
    s = state.copy()
    s.boards = [state.boards[1].copy(), state.boards[0].copy()]
    s.pending_claims = [Claim(1 - c.player, c.domino_id) for c in state.pending_claims]
    s.next_claims = [Claim(1 - c.player, c.domino_id) for c in state.next_claims]
    s.discards = [state.discards[1], state.discards[0]]
    s.start_player = 1 - state.start_player
    return s


def core_schema_hash() -> str:
    """Stable hash of the frozen CORE feature layout. Included in the buffer/schema
    contract so a layout change invalidates trained artifacts loudly."""
    spec = {
        "half_types": HALF_TYPES, "num_half": NUM_HALF, "cell_side": CELL_SIDE,
        "num_cells": NUM_CELLS, "castle": CASTLE, "num_dominoes": NUM_DOMINOES,
        "offsets": {"board": BOARD_OFF, "row": ROW_OFF, "pending": PENDING_OFF,
                    "next": NEXT_OFF, "bag": BAG_OFF, "phase": PHASE_OFF,
                    "actor": ACTOR_OFF, "slot": SLOT_OFF, "disc": DISC_OFF,
                    "rules": RULES_OFF, "core_size": CORE_SIZE},
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
