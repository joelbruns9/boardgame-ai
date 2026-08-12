from __future__ import annotations

import random

import numpy as np
import pytest

from games.kingdomino_classic import (
    ClassicGameConfig,
    ClassicGameState,
    Phase,
    PickAction,
)
from games.kingdomino_classic.action_codec import (
    DISCARD_PLACEMENT,
    NO_PICK,
    NO_PLACEMENT,
    NUM_ACTIONS,
    NUM_D4_TRANSFORMS,
    PICK_AXIS_SIZE,
    PICK_SLOTS,
    d4_action_permutation,
    decode_action,
    draft_slots,
    encode_action,
    legal_action_indices,
    legal_action_mask,
    transform_action_index_d4,
    transform_action_mask_d4,
    transform_policy_d4,
)
from games.kingdomino_classic.encoder import (
    inverse_d4_transform_id,
    transform_state_d4,
)


def _trajectory(players: int, seed: int) -> list[ClassicGameState]:
    state = ClassicGameState.new(
        config=ClassicGameConfig(players=players),
        seed=seed,
        start_player=seed % players,
    )
    rng = random.Random(seed * 65_537 + players)
    states = [state]
    while not state.is_terminal:
        state = state.step(rng.choice(state.legal_actions()))
        states.append(state)
    return states


@pytest.mark.parametrize("players", [3, 4])
def test_every_legal_action_round_trips_without_collisions(players: int) -> None:
    for state in _trajectory(players, seed=7):
        actions = state.legal_actions()
        indices = legal_action_indices(state)

        assert len(indices) == len(actions)
        assert len(indices) == len(set(indices))
        assert legal_action_mask(state).sum() == len(actions)
        for action, action_index in zip(actions, indices):
            assert encode_action(state, action) == action_index
            assert decode_action(state, action_index) == action


@pytest.mark.parametrize("players", [3, 4])
def test_draft_slots_do_not_shift_as_tiles_are_claimed(players: int) -> None:
    state = ClassicGameState.new(
        config=ClassicGameConfig(players=players), seed=19, start_player=0
    )
    original_slots = draft_slots(state)
    assert tuple(state.draft_row) == original_slots

    while state.phase == Phase.INITIAL_DRAFT:
        available_before = set(state.draft_row)
        slot_by_domino = {
            domino_id: original_slots.index(domino_id)
            for domino_id in available_before
        }
        actions = state.legal_actions()
        for action in actions:
            assert isinstance(action, PickAction)
            assert encode_action(state, action) % PICK_AXIS_SIZE == slot_by_domino[
                action.domino_id
            ]
        state = state.step(actions[0])
        if state.phase == Phase.INITIAL_DRAFT:
            assert draft_slots(state) == original_slots


def test_three_player_third_pick_has_two_stable_choices() -> None:
    state = ClassicGameState.new(
        config=ClassicGameConfig(players=3), seed=27, start_player=0
    )
    original = draft_slots(state)
    state = state.step(state.legal_actions()[0])
    state = state.step(state.legal_actions()[0])

    indices = legal_action_indices(state)
    assert len(indices) == 2
    assert {
        action_index % PICK_AXIS_SIZE for action_index in indices
    } == {
        original.index(domino_id) for domino_id in state.draft_row
    }


def test_four_player_final_pick_is_forced_but_keeps_its_original_slot() -> None:
    state = ClassicGameState.new(
        config=ClassicGameConfig(players=4), seed=31, start_player=0
    )
    original = draft_slots(state)
    for _ in range(3):
        state = state.step(state.legal_actions()[0])

    indices = legal_action_indices(state)
    assert len(indices) == 1
    assert indices[0] // PICK_AXIS_SIZE == NO_PLACEMENT
    assert indices[0] % PICK_AXIS_SIZE == original.index(state.draft_row[0])


@pytest.mark.parametrize("players", [3, 4])
def test_legal_mask_commutes_with_every_d4_transform(players: int) -> None:
    trajectory = _trajectory(players, seed=43)
    sampled_states = trajectory[::9]
    for phase in Phase:
        phase_state = next(
            (state for state in trajectory if state.phase == phase), None
        )
        if phase_state is not None and phase_state not in sampled_states:
            sampled_states.append(phase_state)
    if trajectory[-1] not in sampled_states:
        sampled_states.append(trajectory[-1])

    for state in sampled_states:
        mask = legal_action_mask(state)
        for transform_id in range(NUM_D4_TRANSFORMS):
            direct = legal_action_mask(transform_state_d4(state, transform_id))
            transformed = transform_action_mask_d4(mask, transform_id)
            np.testing.assert_array_equal(direct, transformed)


def test_d4_action_permutations_are_bijective_and_exactly_invertible() -> None:
    expected = np.arange(NUM_ACTIONS)
    policy = np.linspace(-1.0, 1.0, NUM_ACTIONS, dtype=np.float32)

    for transform_id in range(NUM_D4_TRANSFORMS):
        permutation = d4_action_permutation(transform_id)
        np.testing.assert_array_equal(np.sort(permutation), expected)
        inverse = inverse_d4_transform_id(transform_id)
        restored_indices = np.fromiter(
            (
                transform_action_index_d4(
                    transform_action_index_d4(index, transform_id), inverse
                )
                for index in range(NUM_ACTIONS)
            ),
            dtype=np.int64,
            count=NUM_ACTIONS,
        )
        np.testing.assert_array_equal(restored_indices, expected)
        restored_policy = transform_policy_d4(
            transform_policy_d4(policy, transform_id), inverse
        )
        np.testing.assert_array_equal(restored_policy, policy)


def test_nonspatial_placement_and_pick_axes_are_d4_invariant() -> None:
    for placement in (DISCARD_PLACEMENT, NO_PLACEMENT):
        for pick in range(PICK_SLOTS + 1):
            action_index = placement * PICK_AXIS_SIZE + pick
            for transform_id in range(NUM_D4_TRANSFORMS):
                assert (
                    transform_action_index_d4(action_index, transform_id)
                    == action_index
                )

    assert NO_PICK == PICK_SLOTS


def test_illegal_and_out_of_range_actions_are_rejected() -> None:
    state = ClassicGameState.new(
        config=ClassicGameConfig(players=3), seed=2, start_player=0
    )
    unavailable = next(
        domino_id for domino_id in range(1, 49) if domino_id not in state.draft_row
    )

    with pytest.raises(ValueError, match="illegal action"):
        encode_action(state, PickAction(unavailable))
    with pytest.raises(ValueError, match="action_index"):
        decode_action(state, NUM_ACTIONS)
    with pytest.raises(ValueError, match="illegal in this state"):
        decode_action(state, DISCARD_PLACEMENT * PICK_AXIS_SIZE)
