"""Fixed action codec and exact D4 policy transforms for Classic Kingdomino.

The placement axis uses every undirected adjacent edge on the centred 9x9
canvas.  Three assignment modes distinguish A-at-canonical-endpoint,
B-at-canonical-endpoint, and identical halves.  The identical mode is fixed by
D4; the other two swap whenever a transform reverses the edge's canonical
endpoint.  This avoids duplicate identical-half actions while retaining a
state-independent D4 permutation.

The pick axis has four stable slots plus NO_PICK.  A slot is the domino's rank
in the original four-tile row, reconstructed from available and already claimed
tiles, so remaining choices never shift left as a draft progresses.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

import numpy as np

from .board import Coord, Placement
from .dominoes import DOMINOES
from .encoder import (
    CANVAS_RADIUS,
    CANVAS_SIZE,
    NUM_D4_TRANSFORMS,
    transform_coord_d4,
)
from .game import Action, ClassicGameState, Phase, PickAction, TurnAction


ACTION_CODEC_VERSION: Final = 1

HORIZONTAL_EDGES: Final = CANVAS_SIZE * (CANVAS_SIZE - 1)
VERTICAL_EDGES: Final = (CANVAS_SIZE - 1) * CANVAS_SIZE
EDGES_PER_ASSIGNMENT: Final = HORIZONTAL_EDGES + VERTICAL_EDGES

A_AT_ANCHOR: Final = 0
B_AT_ANCHOR: Final = 1
IDENTICAL_HALVES: Final = 2
NUM_ASSIGNMENT_MODES: Final = 3

NUM_SPATIAL_PLACEMENTS: Final = NUM_ASSIGNMENT_MODES * EDGES_PER_ASSIGNMENT
DISCARD_PLACEMENT: Final = NUM_SPATIAL_PLACEMENTS
NO_PLACEMENT: Final = NUM_SPATIAL_PLACEMENTS + 1
PLACEMENT_AXIS_SIZE: Final = NUM_SPATIAL_PLACEMENTS + 2

PICK_SLOTS: Final = 4
NO_PICK: Final = PICK_SLOTS
PICK_AXIS_SIZE: Final = PICK_SLOTS + 1
NUM_ACTIONS: Final = PLACEMENT_AXIS_SIZE * PICK_AXIS_SIZE


def _coord_key(coord: Coord) -> tuple[int, int]:
    x, y = coord
    return y, x


def _canonical_edge(first: Coord, second: Coord) -> tuple[Coord, Coord, bool]:
    if abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1:
        raise ValueError("Placement cells must be orthogonally adjacent.")
    if _coord_key(first) <= _coord_key(second):
        return first, second, False
    return second, first, True


def _edge_index(anchor: Coord, other: Coord) -> int:
    x, y = anchor
    other_x, other_y = other
    column = x + CANVAS_RADIUS
    row = y + CANVAS_RADIUS
    if not 0 <= column < CANVAS_SIZE or not 0 <= row < CANVAS_SIZE:
        raise ValueError("Placement anchor is outside the action canvas.")
    if (other_x, other_y) == (x + 1, y):
        if column >= CANVAS_SIZE - 1:
            raise ValueError("Horizontal placement exits the action canvas.")
        return row * (CANVAS_SIZE - 1) + column
    if (other_x, other_y) == (x, y + 1):
        if row >= CANVAS_SIZE - 1:
            raise ValueError("Vertical placement exits the action canvas.")
        return HORIZONTAL_EDGES + row * CANVAS_SIZE + column
    raise ValueError("A canonical edge must point right or down.")


def _decode_edge(edge_index: int) -> tuple[Coord, Coord]:
    if not 0 <= edge_index < EDGES_PER_ASSIGNMENT:
        raise ValueError("edge_index is outside the placement axis.")
    if edge_index < HORIZONTAL_EDGES:
        row, column = divmod(edge_index, CANVAS_SIZE - 1)
        anchor = column - CANVAS_RADIUS, row - CANVAS_RADIUS
        return anchor, (anchor[0] + 1, anchor[1])
    row, column = divmod(edge_index - HORIZONTAL_EDGES, CANVAS_SIZE)
    anchor = column - CANVAS_RADIUS, row - CANVAS_RADIUS
    return anchor, (anchor[0], anchor[1] + 1)


def draft_slots(state: ClassicGameState) -> tuple[int | None, ...]:
    """Return stable original-row slots, including already claimed dominoes."""

    if state.phase not in (Phase.INITIAL_DRAFT, Phase.PLACE_AND_DRAFT):
        return (None,) * PICK_SLOTS
    domino_ids = [*state.draft_row, *(claim.domino_id for claim in state.next_claims)]
    if len(domino_ids) != PICK_SLOTS or len(set(domino_ids)) != PICK_SLOTS:
        raise AssertionError("An active draft must reconstruct exactly four slots.")
    return tuple(sorted(domino_ids))


def _pick_index(state: ClassicGameState, domino_id: int | None) -> int:
    if domino_id is None:
        return NO_PICK
    slots = draft_slots(state)
    try:
        return slots.index(domino_id)
    except ValueError as error:
        raise ValueError(
            f"Domino {domino_id} is not in the active draft row."
        ) from error


def _placement_index(state: ClassicGameState, placement: Placement | None) -> int:
    if placement is None:
        return DISCARD_PLACEMENT
    if not state.pending_claims:
        raise ValueError("A spatial placement requires a pending domino.")
    domino = DOMINOES[state.pending_claims[0].domino_id]
    first, second = placement.cells
    anchor, other, reversed_endpoints = _canonical_edge(first, second)
    first_half, second_half = (
        (domino.b, domino.a) if placement.flipped else (domino.a, domino.b)
    )
    anchor_half = second_half if reversed_endpoints else first_half
    if domino.a == domino.b:
        assignment = IDENTICAL_HALVES
    elif anchor_half == domino.a:
        assignment = A_AT_ANCHOR
    else:
        assignment = B_AT_ANCHOR
    return assignment * EDGES_PER_ASSIGNMENT + _edge_index(anchor, other)


def _encode_action_unchecked(state: ClassicGameState, action: Action) -> int:
    if isinstance(action, PickAction):
        placement_index = NO_PLACEMENT
        pick_index = _pick_index(state, action.domino_id)
    elif isinstance(action, TurnAction):
        placement_index = _placement_index(state, action.placement)
        pick_index = _pick_index(state, action.pick_domino_id)
    else:
        raise TypeError(f"Unsupported action type: {type(action).__name__}.")
    return placement_index * PICK_AXIS_SIZE + pick_index


def encode_action(state: ClassicGameState, action: Action) -> int:
    """Encode one legal action; reject representations not legal in ``state``."""

    legal = state.legal_actions()
    if action not in legal:
        raise ValueError(f"Cannot encode illegal action: {action!r}.")
    return _encode_action_unchecked(state, action)


def legal_action_indices(state: ClassicGameState) -> tuple[int, ...]:
    indices = tuple(
        _encode_action_unchecked(state, action) for action in state.legal_actions()
    )
    if len(indices) != len(set(indices)):
        raise AssertionError("Distinct legal actions collided in the fixed codec.")
    return indices


def legal_action_mask(state: ClassicGameState) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    mask[list(legal_action_indices(state))] = True
    return mask


def decode_action(state: ClassicGameState, action_index: int) -> Action:
    """Decode a legal index back to the engine's exact action representation."""

    if not 0 <= action_index < NUM_ACTIONS:
        raise ValueError(f"action_index must be in [0, {NUM_ACTIONS}).")
    action_by_index = {
        _encode_action_unchecked(state, action): action
        for action in state.legal_actions()
    }
    try:
        return action_by_index[action_index]
    except KeyError as error:
        raise ValueError(
            f"Action index {action_index} is illegal in this state."
        ) from error


def _transform_placement_index_d4(
    placement_index: int, transform_id: int
) -> int:
    if placement_index >= NUM_SPATIAL_PLACEMENTS:
        return placement_index
    assignment, edge_index = divmod(placement_index, EDGES_PER_ASSIGNMENT)
    first, second = _decode_edge(edge_index)
    transformed_first = transform_coord_d4(first, transform_id)
    transformed_second = transform_coord_d4(second, transform_id)
    anchor, other, reversed_endpoints = _canonical_edge(
        transformed_first, transformed_second
    )
    if reversed_endpoints and assignment != IDENTICAL_HALVES:
        assignment = B_AT_ANCHOR if assignment == A_AT_ANCHOR else A_AT_ANCHOR
    return assignment * EDGES_PER_ASSIGNMENT + _edge_index(anchor, other)


def transform_action_index_d4(action_index: int, transform_id: int) -> int:
    if not 0 <= action_index < NUM_ACTIONS:
        raise ValueError(f"action_index must be in [0, {NUM_ACTIONS}).")
    if not 0 <= transform_id < NUM_D4_TRANSFORMS:
        raise ValueError(
            f"transform_id must be in [0, {NUM_D4_TRANSFORMS}); got {transform_id}."
        )
    placement_index, pick_index = divmod(action_index, PICK_AXIS_SIZE)
    transformed_placement = _transform_placement_index_d4(
        placement_index, transform_id
    )
    return transformed_placement * PICK_AXIS_SIZE + pick_index


@lru_cache(maxsize=NUM_D4_TRANSFORMS)
def d4_action_permutation(transform_id: int) -> np.ndarray:
    """Return ``source_index -> transformed_index`` for the whole policy."""

    permutation = np.fromiter(
        (
            transform_action_index_d4(action_index, transform_id)
            for action_index in range(NUM_ACTIONS)
        ),
        dtype=np.int64,
        count=NUM_ACTIONS,
    )
    if len(np.unique(permutation)) != NUM_ACTIONS:
        raise AssertionError("A D4 action transform must be a permutation.")
    permutation.setflags(write=False)
    return permutation


def transform_policy_d4(policy: np.ndarray, transform_id: int) -> np.ndarray:
    if policy.shape != (NUM_ACTIONS,):
        raise ValueError(f"policy must have shape ({NUM_ACTIONS},).")
    transformed = np.empty_like(policy)
    transformed[d4_action_permutation(transform_id)] = policy
    return transformed


def transform_action_mask_d4(mask: np.ndarray, transform_id: int) -> np.ndarray:
    if mask.shape != (NUM_ACTIONS,):
        raise ValueError(f"mask must have shape ({NUM_ACTIONS},).")
    transformed = np.empty_like(mask)
    transformed[d4_action_permutation(transform_id)] = mask
    return transformed
