from __future__ import annotations

from itertools import permutations
import random

import numpy as np
import pytest

from games.kingdomino_classic import ClassicGameConfig, ClassicGameState
from games.kingdomino_classic.encoder import (
    BOARD_CHANNEL_NAMES,
    CANVAS_RADIUS,
    CANVAS_SIZE,
    DOMINO_FEATURE_INDEX,
    GLOBAL_FEATURE_INDEX,
    MAX_PLAYERS,
    NUM_BOARD_CHANNELS,
    NUM_D4_TRANSFORMS,
    NUM_DOMINOES,
    NUM_DOMINO_FEATURES,
    NUM_GLOBAL_FEATURES,
    NUM_PLAYER_DOMINO_CHANNELS,
    NUM_PLAYER_FEATURES,
    PLAYER_FEATURE_INDEX,
    EncodedState,
    encode_state,
    inverse_d4_transform_id,
    padded_player_permutation,
    permute_state_players,
    permute_encoded_players,
    transform_encoded_d4,
    transform_state_d4,
)


def _assert_encoded_equal(left: EncodedState, right: EncodedState) -> None:
    for left_array, right_array in zip(left.arrays(), right.arrays()):
        np.testing.assert_array_equal(left_array, right_array)


def _progressed_state(
    players: int, *, seed: int, actions: int = 23
) -> ClassicGameState:
    state = ClassicGameState.new(
        config=ClassicGameConfig(players=players),
        seed=seed,
        start_player=seed % players,
    )
    rng = random.Random(seed * 10_007 + players)
    for _ in range(actions):
        if state.is_terminal:
            break
        state = state.step(rng.choice(state.legal_actions()))
    return state


def _trajectory_snapshots(players: int, seed: int) -> list[ClassicGameState]:
    state = ClassicGameState.new(
        config=ClassicGameConfig(players=players),
        seed=seed,
        start_player=(seed + 1) % players,
    )
    rng = random.Random(91_019 * seed + players)
    snapshots = [state]
    while not state.is_terminal:
        previous_phase = state.phase
        state = state.step(rng.choice(state.legal_actions()))
        if (
            len(state.history) % 11 == 0
            or len(state.history) <= players
            or state.phase != previous_phase
            or state.is_terminal
        ):
            snapshots.append(state)
    return snapshots


@pytest.mark.parametrize("players", [3, 4])
def test_shapes_dtype_and_complete_inventory_partition(players: int) -> None:
    encoded = encode_state(_progressed_state(players, seed=13))

    assert encoded.boards.shape == (
        MAX_PLAYERS,
        NUM_BOARD_CHANNELS,
        CANVAS_SIZE,
        CANVAS_SIZE,
    )
    assert encoded.player_features.shape == (MAX_PLAYERS, NUM_PLAYER_FEATURES)
    assert encoded.player_dominoes.shape == (
        MAX_PLAYERS,
        NUM_DOMINOES,
        NUM_PLAYER_DOMINO_CHANNELS,
    )
    assert encoded.domino_features.shape == (
        NUM_DOMINOES,
        NUM_DOMINO_FEATURES,
    )
    assert encoded.global_features.shape == (NUM_GLOBAL_FEATURES,)
    assert all(array.dtype == np.float32 for array in encoded.arrays())

    global_zone_count = encoded.domino_features[
        :,
        [
            DOMINO_FEATURE_INDEX["in_deck"],
            DOMINO_FEATURE_INDEX["in_draft"],
            DOMINO_FEATURE_INDEX["unclaimed_discard"],
        ],
    ].sum(axis=1)
    player_zone_count = encoded.player_dominoes.sum(axis=(0, 2))
    np.testing.assert_array_equal(
        global_zone_count + player_zone_count,
        np.ones(NUM_DOMINOES, dtype=np.float32),
    )


@pytest.mark.parametrize("players", [3, 4])
def test_every_real_player_uses_the_same_schema(players: int) -> None:
    encoded = encode_state(_progressed_state(players, seed=21))
    castle = BOARD_CHANNEL_NAMES.index("castle")
    centre = CANVAS_RADIUS
    present = PLAYER_FEATURE_INDEX["present"]

    for player in range(players):
        assert encoded.player_features[player, present] == 1.0
        assert encoded.boards[player, castle, centre, centre] == 1.0
        assert encoded.boards[player, castle].sum() == 1.0


def test_three_player_padding_is_exactly_zero() -> None:
    encoded = encode_state(_progressed_state(3, seed=8))

    assert not encoded.boards[3].any()
    assert not encoded.player_features[3].any()
    assert not encoded.player_dominoes[3].any()
    assert encoded.global_features[GLOBAL_FEATURE_INDEX["three_players"]] == 1.0
    assert encoded.global_features[GLOBAL_FEATURE_INDEX["four_players"]] == 0.0

    moved = permute_encoded_players(encoded, (3, 0, 1, 2))
    assert not moved.boards[0].any()
    assert not moved.player_features[0].any()
    assert not moved.player_dominoes[0].any()
    np.testing.assert_array_equal(moved.domino_features, encoded.domino_features)
    np.testing.assert_array_equal(moved.global_features, encoded.global_features)


@pytest.mark.parametrize("harmony", [False, True])
@pytest.mark.parametrize("middle_kingdom", [False, True])
def test_scoring_configuration_is_encoded_explicitly(
    harmony: bool, middle_kingdom: bool
) -> None:
    state = ClassicGameState.new(
        config=ClassicGameConfig(
            players=4,
            harmony=harmony,
            middle_kingdom=middle_kingdom,
        ),
        seed=12,
        start_player=0,
    )
    encoded = encode_state(state)

    assert encoded.global_features[GLOBAL_FEATURE_INDEX["harmony"]] == float(
        harmony
    )
    assert encoded.global_features[
        GLOBAL_FEATURE_INDEX["middle_kingdom"]
    ] == float(middle_kingdom)


@pytest.mark.parametrize("players", [3, 4])
def test_all_real_player_permutations_commute_with_encoding(players: int) -> None:
    for state in _trajectory_snapshots(players, seed=34):
        encoded = encode_state(state)
        for permutation in permutations(range(players)):
            direct = encode_state(state, player_order=permutation)
            transformed = permute_encoded_players(
                encoded, padded_player_permutation(permutation, players)
            )
            _assert_encoded_equal(direct, transformed)


@pytest.mark.parametrize("players", [3, 4])
def test_rules_level_player_relabeling_commutes_with_encoding(players: int) -> None:
    for state in _trajectory_snapshots(players, seed=61):
        encoded = encode_state(state)
        for permutation in permutations(range(players)):
            relabeled = encode_state(permute_state_players(state, permutation))
            transformed = permute_encoded_players(
                encoded, padded_player_permutation(permutation, players)
            )
            _assert_encoded_equal(relabeled, transformed)


@pytest.mark.parametrize("players", [3, 4])
def test_d4_state_and_tensor_transforms_commute_exactly(players: int) -> None:
    for seed in (5, 17):
        for state in _trajectory_snapshots(players, seed):
            encoded = encode_state(state)
            for transform_id in range(NUM_D4_TRANSFORMS):
                direct = encode_state(transform_state_d4(state, transform_id))
                transformed = transform_encoded_d4(encoded, transform_id)
                _assert_encoded_equal(direct, transformed)


@pytest.mark.parametrize("players", [3, 4])
def test_every_d4_transform_has_an_exact_inverse(players: int) -> None:
    encoded = encode_state(_progressed_state(players, seed=55))

    for transform_id in range(NUM_D4_TRANSFORMS):
        transformed = transform_encoded_d4(encoded, transform_id)
        restored = transform_encoded_d4(
            transformed, inverse_d4_transform_id(transform_id)
        )
        _assert_encoded_equal(restored, encoded)


@pytest.mark.parametrize("players", [3, 4])
def test_player_permutations_and_d4_transforms_commute(players: int) -> None:
    encoded = encode_state(_progressed_state(players, seed=89))

    for real_permutation in permutations(range(players)):
        permutation = padded_player_permutation(real_permutation, players)
        permuted = permute_encoded_players(encoded, permutation)
        for transform_id in range(NUM_D4_TRANSFORMS):
            left = transform_encoded_d4(permuted, transform_id)
            right = permute_encoded_players(
                transform_encoded_d4(encoded, transform_id), permutation
            )
            _assert_encoded_equal(left, right)


def test_encoder_does_not_leak_hidden_deck_order() -> None:
    state = _progressed_state(4, seed=144, actions=14)
    shuffled = state.copy()
    random.Random(901).shuffle(shuffled.deck)

    assert state.deck != shuffled.deck
    _assert_encoded_equal(encode_state(state), encode_state(shuffled))


def test_terminal_encoding_has_no_actor_and_preserves_player_axis() -> None:
    state = _trajectory_snapshots(3, 233)[-1]
    encoded = encode_state(state)
    actor = PLAYER_FEATURE_INDEX["current_actor"]

    assert state.is_terminal
    assert not encoded.player_features[:, actor].any()
    assert encoded.global_features[GLOBAL_FEATURE_INDEX["phase_game_over"]] == 1.0


def test_invalid_permutations_and_transforms_are_rejected() -> None:
    state = _progressed_state(3, seed=3)
    encoded = encode_state(state)

    with pytest.raises(ValueError, match="player_order"):
        encode_state(state, player_order=(0, 0, 1))
    with pytest.raises(ValueError, match="four slots"):
        permute_encoded_players(encoded, (0, 1, 2))
    with pytest.raises(ValueError, match="transform_id"):
        transform_encoded_d4(encoded, NUM_D4_TRANSFORMS)
