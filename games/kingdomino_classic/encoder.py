"""Exactly player-equivariant state encoding for Classic Kingdomino.

Players occupy a single padded axis.  No feature depends on an absolute player
index, and opponents are never assigned distinct semantic roles.  Relabeling
players therefore only permutes that axis.  The fourth 3p slot is identically
zero and is identified solely by the per-player presence feature.

Spatial boards use a castle-centred 9x9 canvas.  Every legal 5x5 kingdom fits
on this canvas, and all eight D4 transforms act exactly around its centre.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Final, Sequence

import numpy as np

from .board import ClassicBoard, Coord
from .dominoes import DOMINOES, Terrain
from .game import Claim, ClassicGameState, Phase


MAX_PLAYERS: Final = 4
NUM_DOMINOES: Final = 48
ENCODER_VERSION: Final = 1
CANVAS_RADIUS: Final = 4
CANVAS_SIZE: Final = 2 * CANVAS_RADIUS + 1

BOARD_CHANNEL_NAMES: Final = (
    "castle",
    "wheat",
    "forest",
    "water",
    "grass",
    "swamp",
    "mine",
    "crowns",
)
NUM_BOARD_CHANNELS: Final = len(BOARD_CHANNEL_NAMES)

PLAYER_FEATURE_NAMES: Final = (
    "present",
    "current_actor",
    "initial_order_0",
    "initial_order_1",
    "initial_order_2",
    "initial_order_3",
    "pending_order_0",
    "pending_order_1",
    "pending_order_2",
    "pending_order_3",
    "placed_fraction",
    "forced_discard_fraction",
    "territory_score_scaled",
    "total_score_scaled",
    "largest_territory_fraction",
    "crown_fraction",
    "harmony_eligible",
    "castle_centered",
)
NUM_PLAYER_FEATURES: Final = len(PLAYER_FEATURE_NAMES)
PLAYER_FEATURE_INDEX: Final = {
    name: index for index, name in enumerate(PLAYER_FEATURE_NAMES)
}

PLAYER_DOMINO_CHANNEL_NAMES: Final = (
    "placed",
    "pending",
    "next",
    "forced_discard",
)
NUM_PLAYER_DOMINO_CHANNELS: Final = len(PLAYER_DOMINO_CHANNEL_NAMES)

_TERRAINS: Final = (
    Terrain.WHEAT,
    Terrain.FOREST,
    Terrain.WATER,
    Terrain.GRASS,
    Terrain.SWAMP,
    Terrain.MINE,
)
DOMINO_FEATURE_NAMES: Final = tuple(
    chain(
        (f"a_{terrain.name.lower()}" for terrain in _TERRAINS),
        ("a_crowns",),
        (f"b_{terrain.name.lower()}" for terrain in _TERRAINS),
        ("b_crowns", "number", "in_deck", "in_draft", "unclaimed_discard"),
    )
)
NUM_DOMINO_FEATURES: Final = len(DOMINO_FEATURE_NAMES)
DOMINO_FEATURE_INDEX: Final = {
    name: index for index, name in enumerate(DOMINO_FEATURE_NAMES)
}

GLOBAL_FEATURE_NAMES: Final = (
    "phase_initial_draft",
    "phase_place_and_draft",
    "phase_final_placement",
    "phase_game_over",
    "three_players",
    "four_players",
    "harmony",
    "middle_kingdom",
)
NUM_GLOBAL_FEATURES: Final = len(GLOBAL_FEATURE_NAMES)
GLOBAL_FEATURE_INDEX: Final = {
    name: index for index, name in enumerate(GLOBAL_FEATURE_NAMES)
}

# Transform convention: k counter-clockwise quarter turns, followed by an
# optional left-right reflection.  IDs 0..3 are rotations; 4..7 add reflection.
D4_ELEMENTS: Final = tuple(
    (rotation, reflected)
    for reflected in (False, True)
    for rotation in range(4)
)
NUM_D4_TRANSFORMS: Final = len(D4_ELEMENTS)
_D4_INVERSES: Final = (0, 3, 2, 1, 4, 5, 6, 7)


@dataclass(frozen=True, slots=True)
class EncodedState:
    """Dense model inputs with an explicit, shared four-player axis."""

    boards: np.ndarray
    player_features: np.ndarray
    player_dominoes: np.ndarray
    domino_features: np.ndarray
    global_features: np.ndarray

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.boards,
            self.player_features,
            self.player_dominoes,
            self.domino_features,
            self.global_features,
        )


def _validate_player_order(
    state: ClassicGameState, player_order: Sequence[int] | None
) -> tuple[int, ...]:
    order = (
        tuple(range(state.config.players))
        if player_order is None
        else tuple(player_order)
    )
    if len(order) != state.config.players or set(order) != set(
        range(state.config.players)
    ):
        raise ValueError(
            "player_order must be a permutation of every real player exactly once."
        )
    return order


def _encode_board(board: ClassicBoard) -> np.ndarray:
    encoded = np.zeros(
        (NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32
    )
    for x, y in board.occupied_cells():
        if not -CANVAS_RADIUS <= x <= CANVAS_RADIUS or not (
            -CANVAS_RADIUS <= y <= CANVAS_RADIUS
        ):
            raise ValueError(f"Board coordinate {(x, y)} is outside the 9x9 canvas.")
        cell = board.cell_at((x, y))
        assert cell is not None
        row = y + CANVAS_RADIUS
        column = x + CANVAS_RADIUS
        if cell.terrain == Terrain.CASTLE:
            encoded[0, row, column] = 1.0
        else:
            encoded[int(cell.terrain) - 1, row, column] = 1.0
            encoded[-1, row, column] = cell.crowns / 3.0
    return encoded


def _claims_by_player(
    claims: Sequence[Claim], players: int
) -> tuple[list[int], ...]:
    result = tuple([] for _ in range(players))
    for claim in claims:
        result[claim.player].append(claim.domino_id)
    return result


def _player_features(state: ClassicGameState, player: int) -> np.ndarray:
    features = np.zeros(NUM_PLAYER_FEATURES, dtype=np.float32)
    index = PLAYER_FEATURE_INDEX
    features[index["present"]] = 1.0
    if not state.is_terminal and state.current_actor == player:
        features[index["current_actor"]] = 1.0

    if state.phase == Phase.INITIAL_DRAFT:
        order = (player - state.start_player) % state.config.players
        features[index[f"initial_order_{order}"]] = 1.0

    for order, claim in enumerate(state.pending_claims):
        if claim.player == player:
            features[index[f"pending_order_{order}"]] = 1.0

    board = state.boards[player]
    breakdown = board.score(
        harmony=state.config.harmony,
        middle_kingdom=state.config.middle_kingdom,
    )
    centered = board.score(harmony=False, middle_kingdom=True)
    features[index["placed_fraction"]] = board.placed_domino_count / 12.0
    features[index["forced_discard_fraction"]] = (
        len(state.forced_discards[player]) / 12.0
    )
    features[index["territory_score_scaled"]] = breakdown.territory_score / 200.0
    features[index["total_score_scaled"]] = breakdown.total / 200.0
    features[index["largest_territory_fraction"]] = (
        breakdown.largest_territory_size / 24.0
    )
    features[index["crown_fraction"]] = breakdown.total_crowns / 20.0
    features[index["harmony_eligible"]] = float(
        not state.forced_discards[player]
    )
    features[index["castle_centered"]] = float(
        centered.middle_kingdom_bonus > 0
    )
    return features


def _static_domino_features() -> np.ndarray:
    features = np.zeros(
        (NUM_DOMINOES, NUM_DOMINO_FEATURES), dtype=np.float32
    )
    for domino_id, domino in DOMINOES.items():
        row = features[domino_id - 1]
        row[_TERRAINS.index(domino.a.terrain)] = 1.0
        row[6] = domino.a.crowns / 3.0
        row[7 + _TERRAINS.index(domino.b.terrain)] = 1.0
        row[13] = domino.b.crowns / 3.0
        row[14] = domino_id / NUM_DOMINOES
    return features


_STATIC_DOMINO_FEATURES: Final = _static_domino_features()


def encode_state(
    state: ClassicGameState,
    *,
    player_order: Sequence[int] | None = None,
) -> EncodedState:
    """Encode public state without assigning semantic roles to player slots.

    ``player_order`` selects which source player appears in each leading output
    slot.  It exists to make player relabeling explicit and testable.  The
    default is engine order; model code must treat that axis equivariantly.
    """

    state.assert_invariants()
    order = _validate_player_order(state, player_order)
    boards = np.zeros(
        (MAX_PLAYERS, NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE),
        dtype=np.float32,
    )
    player_features = np.zeros(
        (MAX_PLAYERS, NUM_PLAYER_FEATURES), dtype=np.float32
    )
    player_dominoes = np.zeros(
        (MAX_PLAYERS, NUM_DOMINOES, NUM_PLAYER_DOMINO_CHANNELS),
        dtype=np.float32,
    )

    pending = _claims_by_player(state.pending_claims, state.config.players)
    next_claims = _claims_by_player(state.next_claims, state.config.players)
    for output_slot, player in enumerate(order):
        board = state.boards[player]
        boards[output_slot] = _encode_board(board)
        player_features[output_slot] = _player_features(state, player)
        for domino_id in board.placed_domino_ids:
            player_dominoes[output_slot, domino_id - 1, 0] = 1.0
        for domino_id in pending[player]:
            player_dominoes[output_slot, domino_id - 1, 1] = 1.0
        for domino_id in next_claims[player]:
            player_dominoes[output_slot, domino_id - 1, 2] = 1.0
        for domino_id in state.forced_discards[player]:
            player_dominoes[output_slot, domino_id - 1, 3] = 1.0

    domino_features = _STATIC_DOMINO_FEATURES.copy()
    for domino_id in state.deck:
        domino_features[domino_id - 1, DOMINO_FEATURE_INDEX["in_deck"]] = 1.0
    for domino_id in state.draft_row:
        domino_features[domino_id - 1, DOMINO_FEATURE_INDEX["in_draft"]] = 1.0
    for domino_id in state.unclaimed_discards:
        domino_features[
            domino_id - 1, DOMINO_FEATURE_INDEX["unclaimed_discard"]
        ] = 1.0

    global_features = np.zeros(NUM_GLOBAL_FEATURES, dtype=np.float32)
    global_features[int(state.phase)] = 1.0
    global_features[GLOBAL_FEATURE_INDEX["three_players"]] = float(
        state.config.players == 3
    )
    global_features[GLOBAL_FEATURE_INDEX["four_players"]] = float(
        state.config.players == 4
    )
    global_features[GLOBAL_FEATURE_INDEX["harmony"]] = float(state.config.harmony)
    global_features[GLOBAL_FEATURE_INDEX["middle_kingdom"]] = float(
        state.config.middle_kingdom
    )
    return EncodedState(
        boards=boards,
        player_features=player_features,
        player_dominoes=player_dominoes,
        domino_features=domino_features,
        global_features=global_features,
    )


def permute_encoded_players(
    encoded: EncodedState, permutation: Sequence[int]
) -> EncodedState:
    """Apply ``new_slot -> old_slot`` permutation to the padded player axis."""

    permutation = tuple(permutation)
    if len(permutation) != MAX_PLAYERS or set(permutation) != set(
        range(MAX_PLAYERS)
    ):
        raise ValueError("permutation must contain each of the four slots once.")
    slots = list(permutation)
    return EncodedState(
        boards=np.ascontiguousarray(encoded.boards[slots]),
        player_features=np.ascontiguousarray(encoded.player_features[slots]),
        player_dominoes=np.ascontiguousarray(encoded.player_dominoes[slots]),
        domino_features=encoded.domino_features.copy(),
        global_features=encoded.global_features.copy(),
    )


def padded_player_permutation(
    real_player_permutation: Sequence[int], players: int
) -> tuple[int, ...]:
    """Extend an S3/S4 real-player permutation over the padded four slots."""

    permutation = tuple(real_player_permutation)
    if players not in (3, 4) or len(permutation) != players or set(
        permutation
    ) != set(range(players)):
        raise ValueError("real_player_permutation does not match player count.")
    return (*permutation, *range(players, MAX_PLAYERS))


def _d4_element(transform_id: int) -> tuple[int, bool]:
    if not 0 <= transform_id < NUM_D4_TRANSFORMS:
        raise ValueError(
            f"transform_id must be in [0, {NUM_D4_TRANSFORMS}); got {transform_id}."
        )
    return D4_ELEMENTS[transform_id]


def inverse_d4_transform_id(transform_id: int) -> int:
    _d4_element(transform_id)
    return _D4_INVERSES[transform_id]


def transform_coord_d4(coord: Coord, transform_id: int) -> Coord:
    """Apply the same D4 convention used by ``numpy.rot90`` on board tensors."""

    rotation, reflected = _d4_element(transform_id)
    x, y = coord
    for _ in range(rotation):
        x, y = y, -x
    if reflected:
        x = -x
    return x, y


def transform_state_d4(
    state: ClassicGameState, transform_id: int
) -> ClassicGameState:
    """Rules-level D4 oracle used to verify encoder commutation exactly."""

    _d4_element(transform_id)
    transformed = state.copy()
    transformed.boards = [
        board.transformed_coordinates(
            lambda coord, transform_id=transform_id: transform_coord_d4(
                coord, transform_id
            )
        )
        for board in state.boards
    ]
    transformed.assert_invariants()
    return transformed


def transform_encoded_d4(
    encoded: EncodedState, transform_id: int
) -> EncodedState:
    """Apply a D4 transform; every non-spatial feature remains invariant."""

    rotation, reflected = _d4_element(transform_id)
    boards = np.rot90(encoded.boards, k=rotation, axes=(-2, -1))
    if reflected:
        boards = boards[..., ::-1]
    return EncodedState(
        boards=np.ascontiguousarray(boards),
        player_features=encoded.player_features.copy(),
        player_dominoes=encoded.player_dominoes.copy(),
        domino_features=encoded.domino_features.copy(),
        global_features=encoded.global_features.copy(),
    )
