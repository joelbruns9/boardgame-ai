"""
Action codec — bidirectional mapping between engine actions and policy indices.

The joint policy head (see encoder.py / network design) outputs a distribution
over NUM_JOINT_ACTIONS = 3390 indices, structured as a (placement, pick) pair:

PLACEMENT AXIS  (PLACEMENT_AXIS_SIZE = 678)
    0 .. 675       spatial placements.  Index = direction * 169 + y * 13 + x
                   where (y, x) is the A-half anchor in castle-centred coords
                   and direction ∈ {0:right, 1:down, 2:left, 3:up} encodes
                   where the B-half sits relative to A.
    676            DISCARD — represents the action `TurnAction(placement=None)`.
                   Whether discard is *legal* in a given state depends on the
                   engine, not the codec (see DISCARD SEMANTICS below).
    677            NO_PLACEMENT — placeholder used only in INITIAL_SELECTION
                   where the action is a pure PickAction with no placement.

PICK AXIS  (PICK_AXIS_SIZE = 5)
    0 .. 3         pick the domino currently in current_row[i]
    4              NO_PICK (used only in FINAL_PLACEMENT)

Joint index:  joint = placement_idx * PICK_AXIS_SIZE + pick_idx

PHASE-DEPENDENT LEGALITY
    INITIAL_SELECTION : placement_idx must be NO_PLACEMENT_IDX
                        pick_idx ∈ valid current_row slots
    PLACE_AND_SELECT  : placement_idx ∈ spatial or DISCARD
                        pick_idx ∈ valid current_row slots
    FINAL_PLACEMENT   : placement_idx ∈ spatial or DISCARD
                        pick_idx must be NO_PICK_IDX

CORRECTNESS CONTRACT
For any legal engine action `a` in state `s`:
    decode_action(encode_action(a, s), s)  physically equals  a
where "physically equals" means: stepping `s` with the decoded action
yields the same resulting board state as stepping with `a`.  Strict
object equality may fail for symmetric dominoes (where a + b are
indistinguishable halves), since the codec canonicalises to "A at anchor".

TRAINING ISOLATION
This module does NOT import evaluation.py.  The action indexing is a
pure function of the engine state and the action object.

DISCARD SEMANTICS — engine is the source of truth
The codec assigns a fixed index to "place nothing" (TurnAction.placement=None)
but does NOT decide when that action is legal.  Legality is whatever
`state.legal_actions()` exposes.  Under the current engine, discard is
forced-only: `legal_actions()` returns a discard option only when no legal
placement exists.  The policy therefore learns the forced-only game, which
matches the real Kingdomino rules.

If the engine ever changes to expose voluntary discard (e.g., for search
convenience), the codec keeps working mechanically — but the policy would
start learning a different game.  Keep the engine and the rule you want the
policy to learn aligned.  The codec is a passive reflection of whatever
`legal_actions()` returns.

PICK ENCODING AND OPEN-LOOP MCTS
─────────────────────────────────
Pick indices are SLOT-RELATIVE: pick_idx ∈ {0..3} identifies a position
in `current_row`, not a specific domino_id. This is the property that
makes open-loop MCTS (per-simulation deck resampling) valid for this
codec — see `_encode_pick` for the full explanation.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from games.kingdomino.board import Placement
from games.kingdomino.dominoes import DOMINOES
from games.kingdomino.encoder import CANVAS_SIZE, CASTLE_CENTER
from games.kingdomino.game import GameState, Phase, PickAction, TurnAction

# ─── axis sizes ───────────────────────────────────────────────────────────
NUM_DIRECTIONS = 4
NUM_CELLS = CANVAS_SIZE * CANVAS_SIZE                # 169
NUM_SPATIAL_PLACEMENTS = NUM_DIRECTIONS * NUM_CELLS  # 676

DISCARD_PLACEMENT_IDX = NUM_SPATIAL_PLACEMENTS       # 676
NO_PLACEMENT_IDX = NUM_SPATIAL_PLACEMENTS + 1        # 677
PLACEMENT_AXIS_SIZE = NUM_SPATIAL_PLACEMENTS + 2     # 678

NUM_PICK_SLOTS = 4
NO_PICK_IDX = NUM_PICK_SLOTS                         # 4
PICK_AXIS_SIZE = NUM_PICK_SLOTS + 1                  # 5

NUM_JOINT_ACTIONS = PLACEMENT_AXIS_SIZE * PICK_AXIS_SIZE  # 3390

# B-half position relative to A-half anchor, indexed by direction in 0..3.
_DIRECTION_DELTAS: Tuple[Tuple[int, int], ...] = (
    (+1,  0),  # 0: right
    ( 0, +1),  # 1: down
    (-1,  0),  # 2: left
    ( 0, -1),  # 3: up
)

EngineAction = Union[PickAction, TurnAction]


# ─── index arithmetic ─────────────────────────────────────────────────────
def make_joint_idx(placement_idx: int, pick_idx: int) -> int:
    """Combine a placement axis index and pick axis index into a joint index."""
    return placement_idx * PICK_AXIS_SIZE + pick_idx


def split_joint_idx(idx: int) -> Tuple[int, int]:
    """Split a joint index into (placement_idx, pick_idx)."""
    return divmod(idx, PICK_AXIS_SIZE)


def _spatial_placement_idx(direction: int, out_y: int, out_x: int) -> int:
    return direction * NUM_CELLS + out_y * CANVAS_SIZE + out_x


def _decode_spatial_placement_idx(idx: int) -> Tuple[int, int, int]:
    """Returns (direction, out_y, out_x)."""
    direction, rem = divmod(idx, NUM_CELLS)
    out_y, out_x = divmod(rem, CANVAS_SIZE)
    return direction, out_y, out_x


# ─── placement encoding / decoding ────────────────────────────────────────
def _encode_placement(placement: Placement, castle_pos: Tuple[int, int]) -> int:
    """Encode a `Placement` (canvas coords) as a spatial placement index.

    Canonical form: anchor = A-half cell, direction = where B-half sits.
    Handles both `placement.flipped` values transparently:
        flipped=False → A at (x1,y1), B at (x2,y2)
        flipped=True  → B at (x1,y1), A at (x2,y2)
    """
    if placement.flipped:
        a_x, a_y = placement.x2, placement.y2
        b_x, b_y = placement.x1, placement.y1
    else:
        a_x, a_y = placement.x1, placement.y1
        b_x, b_y = placement.x2, placement.y2

    cx, cy = castle_pos
    out_x = a_x - cx + CASTLE_CENTER
    out_y = a_y - cy + CASTLE_CENTER

    if not (0 <= out_x < CANVAS_SIZE and 0 <= out_y < CANVAS_SIZE):
        raise ValueError(
            f"A-half anchor ({a_x},{a_y}) maps to ({out_x},{out_y}) which is "
            f"outside the {CANVAS_SIZE}×{CANVAS_SIZE} crop centred on "
            f"castle {castle_pos}."
        )

    dx, dy = b_x - a_x, b_y - a_y
    try:
        direction = _DIRECTION_DELTAS.index((dx, dy))
    except ValueError:
        raise ValueError(
            f"B-half offset ({dx},{dy}) from A-half ({a_x},{a_y}) is not a "
            f"valid orthogonal step.  Allowed offsets: {_DIRECTION_DELTAS}."
        )

    return _spatial_placement_idx(direction, out_y, out_x)


def _decode_placement(idx: int, castle_pos: Tuple[int, int]) -> Placement:
    """Decode a spatial placement index → `Placement` in canvas coords.

    Always produces flipped=False with A at (x1,y1).  For symmetric dominoes,
    this may differ in object equality from the placement returned by
    `legal_placements()`, but is physically equivalent and accepted by
    `board.place()`.
    """
    if not 0 <= idx < NUM_SPATIAL_PLACEMENTS:
        raise ValueError(
            f"idx={idx} is outside the spatial placement range "
            f"[0, {NUM_SPATIAL_PLACEMENTS})."
        )

    direction, out_y, out_x = _decode_spatial_placement_idx(idx)

    cx, cy = castle_pos
    a_x = out_x - CASTLE_CENTER + cx
    a_y = out_y - CASTLE_CENTER + cy

    dx, dy = _DIRECTION_DELTAS[direction]
    b_x, b_y = a_x + dx, a_y + dy

    return Placement(x1=a_x, y1=a_y, x2=b_x, y2=b_y, flipped=False)


# ─── pick encoding / decoding ─────────────────────────────────────────────
def _encode_pick(pick_domino_id: Optional[int], state: GameState) -> int:
    """Encode a pick choice as a slot index into `state.current_row`.

    Returns the position of `pick_domino_id` in `current_row` (0-based),
    NOT the domino_id itself. This slot-relative encoding is the property
    that makes open-loop MCTS valid:

    SLOT-RELATIVE PICK ENCODING AND OPEN-LOOP MCTS
    ───────────────────────────────────────────────
    In open-loop MCTS each simulation resamples the hidden deck order, so
    the concrete domino at slot k of a future round's current_row will differ
    across simulations. A slot-relative action ("pick slot k") is a legal,
    well-defined instruction in every determinization — slot k always exists
    (current_row always has 4 entries at the root and within the current
    public row).

    At DEEP nodes (future rounds after hidden rows are revealed) the concrete
    domino at slot k varies across simulations. This is correct: open-loop
    averages over exactly that distribution of futures. The training target
    is extracted only at the ROOT, where current_row is public and identical
    across determinizations, so no target contamination occurs.

    If pick encoding were domino-id-relative instead ("pick domino 37"),
    replaying "pick domino 37" would fail in any determinization where 37
    is not in that future row — open-loop would break. Slot-relative
    encoding avoids this entirely.

    `pick_domino_id=None` → NO_PICK_IDX (FINAL_PLACEMENT only).
    """
    if pick_domino_id is None:
        return NO_PICK_IDX
    try:
        return state.current_row.index(pick_domino_id)
    except ValueError:
        raise ValueError(
            f"pick_domino_id={pick_domino_id} not found in "
            f"current_row={state.current_row}."
        )


def _decode_pick(pick_idx: int, state: GameState) -> Optional[int]:
    """Decode a pick-axis slot index → the `pick_domino_id` at that slot.

    This is the inverse of the slot-relative encoding in `_encode_pick`.
    Returns `state.current_row[pick_idx]` — the concrete domino ID at the
    given slot in THIS simulation's current_row. In open-loop MCTS this
    value varies across simulations at deep nodes (because the hidden deck
    differs per determinization), which is correct behaviour: the variation
    is what open-loop averages over. At the root current_row is public, so
    the returned domino_id is deterministic and target-safe.

    Returns None for NO_PICK_IDX (FINAL_PLACEMENT).
    """
    if pick_idx == NO_PICK_IDX:
        return None
    if not 0 <= pick_idx < NUM_PICK_SLOTS:
        raise ValueError(
            f"pick_idx={pick_idx} is outside [0, {NUM_PICK_SLOTS})."
        )
    if pick_idx >= len(state.current_row):
        raise ValueError(
            f"pick_idx={pick_idx} references current_row slot that isn't "
            f"filled (current_row has {len(state.current_row)} tiles)."
        )
    return state.current_row[pick_idx]


# ─── public API ───────────────────────────────────────────────────────────
def encode_action(action: EngineAction, state: GameState) -> int:
    """Convert an engine action to a flat joint policy index."""
    if isinstance(action, PickAction):
        if state.phase != Phase.INITIAL_SELECTION:
            raise ValueError(
                f"PickAction is only valid in INITIAL_SELECTION; "
                f"state.phase={state.phase.name}."
            )
        pick_idx = _encode_pick(action.domino_id, state)
        return make_joint_idx(NO_PLACEMENT_IDX, pick_idx)

    if isinstance(action, TurnAction):
        if state.phase not in (Phase.PLACE_AND_SELECT, Phase.FINAL_PLACEMENT):
            raise ValueError(
                f"TurnAction is only valid in PLACE_AND_SELECT/FINAL_PLACEMENT; "
                f"state.phase={state.phase.name}."
            )
        # Placement component
        if action.placement is None:
            placement_idx = DISCARD_PLACEMENT_IDX
        else:
            current_board = state.boards[state.current_actor]
            placement_idx = _encode_placement(action.placement,
                                              current_board.castle_pos)
            # Symmetric-domino canonicalization: when the two halves are
            # identical (terrain AND crowns — dominoes 1..12), the placement can
            # be encoded with the anchor at either cell, giving two different
            # joint indices for the SAME physical move.  Collapse to the smaller
            # index so encode_action is representation-invariant and
            # deterministic, rather than depending on which representative
            # legal_placements happened to return.
            domino = DOMINOES[state.pending_claims[state.actor_index].domino_id]
            if domino.a == domino.b:
                p = action.placement
                swapped = Placement(p.x2, p.y2, p.x1, p.y1, p.flipped)
                placement_idx = min(
                    placement_idx,
                    _encode_placement(swapped, current_board.castle_pos),
                )
        # Pick component
        if state.phase == Phase.FINAL_PLACEMENT:
            pick_idx = NO_PICK_IDX  # ignore action.pick_domino_id; engine does too
        else:
            pick_idx = _encode_pick(action.pick_domino_id, state)
        return make_joint_idx(placement_idx, pick_idx)

    raise TypeError(f"Unsupported action type: {type(action).__name__}")


def decode_action(idx: int, state: GameState, validate: bool = True) -> EngineAction:
    """Convert a flat joint policy index back to an engine action.

    `state` is required to look up current_row tiles and the current actor's
    castle position.

    When `validate=True` (default), spatial placements are checked against
    `board.is_legal_placement` for the current actor's domino — if the
    decoded placement isn't actually legal, `ValueError` is raised.  This is
    defence in depth: it catches bugs where a caller samples from unmasked
    logits and would otherwise produce a syntactically valid TurnAction with
    an illegal Placement, failing later inside `board.place()` with a less
    diagnostic error.

    Pass `validate=False` only when you explicitly want to inspect what an
    arbitrary index decodes to without legality enforcement (debug tools,
    visualisations).  Search and self-play code should always leave
    validation on; the cost is microseconds.

    Note: discard (placement=None) and NO_PLACEMENT/NO_PICK structural
    checks always run regardless of `validate`.  `validate` controls only
    the *spatial placement legality* check, since that's the gap where
    unmasked sampling can produce a "syntactically valid but illegal"
    action.
    """
    if not 0 <= idx < NUM_JOINT_ACTIONS:
        raise ValueError(
            f"idx={idx} is outside [0, {NUM_JOINT_ACTIONS})."
        )
    placement_idx, pick_idx = split_joint_idx(idx)

    if state.phase == Phase.INITIAL_SELECTION:
        if placement_idx != NO_PLACEMENT_IDX:
            raise ValueError(
                "INITIAL_SELECTION requires placement_idx=NO_PLACEMENT_IDX."
            )
        if pick_idx == NO_PICK_IDX:
            raise ValueError(
                "INITIAL_SELECTION requires a real pick (pick_idx ≠ NO_PICK)."
            )
        domino_id = _decode_pick(pick_idx, state)
        return PickAction(domino_id=domino_id)

    if state.phase in (Phase.PLACE_AND_SELECT, Phase.FINAL_PLACEMENT):
        if placement_idx == NO_PLACEMENT_IDX:
            raise ValueError(
                f"NO_PLACEMENT_IDX is not legal in {state.phase.name}."
            )
        if placement_idx == DISCARD_PLACEMENT_IDX:
            placement = None
        else:
            current_board = state.boards[state.current_actor]
            placement = _decode_placement(placement_idx,
                                          current_board.castle_pos)
            if validate:
                # Look up the domino the current actor is placing — it's the
                # one in their pending claim — and ask the board whether this
                # placement is legal for it.  This catches index-from-unmasked
                # bugs at the codec boundary, not deep inside board.place().
                domino_id = state.pending_claims[state.actor_index].domino_id
                domino = DOMINOES[domino_id]
                if not current_board.is_legal_placement(domino, placement):
                    raise ValueError(
                        f"Decoded placement {placement} is not legal for "
                        f"domino_id={domino_id} in {state.phase.name}.  "
                        f"This usually means a caller sampled from unmasked "
                        f"logits; check that legal_mask() was applied before "
                        f"sampling.  (Pass validate=False to suppress this "
                        f"check for debug inspection.)"
                    )
        if state.phase == Phase.FINAL_PLACEMENT:
            if pick_idx != NO_PICK_IDX:
                raise ValueError(
                    "FINAL_PLACEMENT requires pick_idx=NO_PICK_IDX."
                )
            pick_domino_id = None
        else:
            if pick_idx == NO_PICK_IDX:
                raise ValueError(
                    "PLACE_AND_SELECT requires a real pick "
                    "(pick_idx ≠ NO_PICK)."
                )
            pick_domino_id = _decode_pick(pick_idx, state)
        return TurnAction(placement=placement, pick_domino_id=pick_domino_id)

    raise ValueError(
        f"decode_action is undefined for phase {state.phase.name}."
    )


def legal_mask(state: GameState) -> np.ndarray:
    """Boolean mask of shape (NUM_JOINT_ACTIONS,) marking legal joint indices.

    For each action returned by `state.legal_actions()`, the corresponding
    joint index is True; all others are False.  Use this to mask network
    output logits before softmax.
    """
    # Rust fast path: a RustGameState builds the mask directly from its
    # already-computed joint indices (no Python encode_action loop).  Duck-typed
    # dispatch — Python's GameState has no legal_mask(), so AttributeError is the
    # natural fallback branch (zero-cost when the Rust path succeeds).
    try:
        return state.legal_mask()
    except AttributeError:
        pass

    mask = np.zeros(NUM_JOINT_ACTIONS, dtype=bool)
    if state.phase == Phase.GAME_OVER:
        return mask  # no legal actions in a terminal state
    for action in state.legal_actions():
        idx = encode_action(action, state)
        if mask[idx]:
            # Two different engine actions mapped to the same joint index.
            # Should not happen — flag it loudly if it does.
            raise RuntimeError(
                f"Action collision at joint idx {idx} in {state.phase.name}. "
                f"This indicates an indexing bug in the codec."
            )
        mask[idx] = True
    return mask


# ─── module self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"PLACEMENT_AXIS_SIZE = {PLACEMENT_AXIS_SIZE}")
    print(f"PICK_AXIS_SIZE      = {PICK_AXIS_SIZE}")
    print(f"NUM_JOINT_ACTIONS   = {NUM_JOINT_ACTIONS}")
    print(f"DISCARD_PLACEMENT_IDX = {DISCARD_PLACEMENT_IDX}")
    print(f"NO_PLACEMENT_IDX      = {NO_PLACEMENT_IDX}")
    print(f"NO_PICK_IDX           = {NO_PICK_IDX}")