"""The flat action space: no collisions, exact round trips, readable labels."""
from __future__ import annotations

import pytest

from games.welcome_to import action_codec as codec
from games.welcome_to.constants import FENCE_SIZES, STREET_SIZES, TEMP_DELTAS


def test_layout_is_contiguous_and_the_expected_size():
    assert codec.NUM_ACTIONS == 357
    offsets = sorted(codec.OFFSET.values())
    assert offsets == list(codec.OFFSET.values()), "blocks must be in order"
    assert offsets[0] == 0
    assert max(offsets) < codec.NUM_ACTIONS


def test_every_index_has_a_description():
    seen = set()
    for i in range(codec.NUM_ACTIONS):
        label = codec.describe(i)
        assert label
        seen.add(i)
    assert len(seen) == codec.NUM_ACTIONS

    with pytest.raises(ValueError):
        codec.describe(codec.NUM_ACTIONS)


def test_stack_choices_round_trip():
    for slot in range(6):
        assert codec.decode_stack(codec.choose_stack(slot)) == slot


def test_write_actions_round_trip_and_do_not_collide():
    seen = set()
    for delta_slot in range(len(TEMP_DELTAS)):
        for x, size in enumerate(STREET_SIZES):
            for y in range(size):
                index = codec.write(delta_slot, x, y)
                assert index not in seen
                seen.add(index)
                assert codec.decode_write(index) == (delta_slot, x, y)


def test_bis_actions_round_trip_and_do_not_collide():
    seen = set()
    for x, size in enumerate(STREET_SIZES):
        for y in range(size):
            for side in (0, 1):
                index = codec.bis(x, y, side)
                assert index not in seen
                seen.add(index)
                assert codec.decode_bis(index) == (x, y, side)


def test_fence_actions_round_trip():
    for x, size in enumerate(FENCE_SIZES):
        for j in range(size):
            index = codec.surveyor_fence(x, j)
            assert codec.decode_surveyor_fence(index) == (x, j)


def test_estate_and_plan_actions_round_trip():
    for row in range(6):
        assert codec.decode_estate_row(codec.estate_row(row)) == row
    for slot in range(3):
        assert codec.decode_plan(codec.choose_plan(slot)) == slot
    for x in range(3):
        assert codec.decode_park_street(codec.park_street(x)) == x
    for x, size in enumerate(STREET_SIZES):
        for y in range(size):
            index = codec.validate_estate(x, y)
            assert codec.decode_validate_estate(index) == (x, y)
            assert codec.decode_roundabout_pos(codec.roundabout_pos(x, y)) == (x, y)


def test_the_pass_slots_are_all_distinct():
    passes = {
        codec.A_PASS_ROUNDABOUT,
        codec.A_PASS_SURVEYOR,
        codec.A_PASS_ESTATE,
        codec.A_PASS_PARK,
        codec.A_PASS_POOL,
        codec.A_PASS_BIS,
        codec.A_PASS_PLAN,
    }
    assert len(passes) == 7
    assert codec.A_RESHUFFLE_YES != codec.A_RESHUFFLE_NO


def test_expert_pairs_are_the_six_ordered_choices():
    assert len(codec.EXPERT_PAIRS) == 6
    assert len(set(codec.EXPERT_PAIRS)) == 6
    for i, j in codec.EXPERT_PAIRS:
        assert i != j
        assert 0 <= i < 3 and 0 <= j < 3
