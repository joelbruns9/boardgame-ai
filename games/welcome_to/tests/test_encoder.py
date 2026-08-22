"""Encoder shape, seat symmetry, determinism and information-set safety."""
from __future__ import annotations

import random

import numpy as np
import pytest

from games.welcome_to import action_codec as codec
from games.welcome_to import encoder as enc
from games.welcome_to.bots import GreedyBot
from games.welcome_to.constants import CARD_TABLE, Effect, NUM_BASE_CARDS
from games.welcome_to.plans import PLANS, PlanKind, progress
from games.welcome_to.game import GameConfig, GameState, Phase


def _card_with(number=None, effect=None) -> int:
    for i, (n, e) in enumerate(CARD_TABLE[:NUM_BASE_CARDS]):
        if (number is None or n == number) and (effect is None or e is effect):
            return i
    raise AssertionError(f"no card number={number} effect={effect}")


def _game(players: int = 2, **kwargs) -> GameState:
    return GameState.new(seed=21, config=GameConfig(players=players, **kwargs))


def _at_turn_boundary(players: int, turn: int, seed: int = 5) -> GameState:
    """A played-in position where the public snapshot equals the live sheets.

    Seats have to differ for a symmetry test to say anything, and the snapshot
    has to be current for the comparison to be exact -- mid-turn, an opponent's
    block *legitimately* lags its owner's.  ``_begin_turn`` refreshes the
    snapshot, so the start of a turn is both.
    """
    bots = [GreedyBot(random.Random(seed * 100 + i)) for i in range(players)]
    state = GameState.new(seed=seed, config=GameConfig(players=players, advanced=True))
    while not state.is_terminal and state.turn < turn:
        state.apply(bots[state.actor].act(state))
    assert not state.is_terminal
    assert state.phase is Phase.CHOOSE_CARDS and state.actor == 0
    for live, public in zip(state.sheets, state.public_sheets):
        assert live.numbers == public.numbers, "not actually a turn boundary"
    return state


# ──────────────────────────────────────────────────────────────────────────
# Shape
# ──────────────────────────────────────────────────────────────────────────
def test_encoding_has_the_declared_shape():
    planes, sheets, viewer, glob = enc.encode_state(_game())
    assert planes.shape == enc.SHEET_PLANES_SHAPE
    assert sheets.shape == (enc.MAX_SEATS, enc.NUM_SHEET_SCALAR)
    assert viewer.shape == enc.VIEWER_PLANE_SHAPE
    assert glob.shape == (enc.NUM_GLOBAL_SCALAR,)
    assert all(a.dtype == np.float32 for a in (planes, sheets, viewer, glob))


def test_scalar_blocks_add_up():
    assert enc.NUM_SHEET_SCALAR == sum(n for _, n in enc.SHEET_SCALAR_BLOCKS)
    assert enc.NUM_GLOBAL_SCALAR == sum(n for _, n in enc.GLOBAL_SCALAR_BLOCKS)


def test_block_names_are_unique_across_the_two_vectors():
    """``block_slice`` indexes one of two arrays; a shared name would be a trap."""
    sheet = {name for name, _ in enc.SHEET_SCALAR_BLOCKS}
    glob = {name for name, _ in enc.GLOBAL_SCALAR_BLOCKS}
    assert not (sheet & glob)
    for name in sheet:
        assert enc.block_axis(name) == "sheet"
    for name in glob:
        assert enc.block_axis(name) == "global"
    with pytest.raises(KeyError):
        enc.block_slice("no_such_block")


def test_shape_does_not_depend_on_the_player_count():
    shapes = set()
    for players in (1, 2, 3, 4):
        shapes.add(tuple(a.shape for a in enc.encode_state(_game(players))))
    assert len(shapes) == 1


def test_encoding_is_finite_and_bounded():
    state = _game(4)
    for _ in range(40):
        if state.is_terminal:
            break
        for array in enc.encode_state(state):
            assert np.isfinite(array).all()
            assert array.min() >= -1.0 and array.max() <= 6.0
        state.apply(state.legal_actions()[0])


def test_encoding_is_deterministic():
    state = _game(3)
    a, b = enc.encode_state(state, 1), enc.encode_state(state, 1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_batch_matches_single():
    states = [_game(2), _game(3)]
    batched = enc.encode_batch(states)
    for i, s in enumerate(states):
        for array, one in zip(batched, enc.encode_state(s)):
            assert np.array_equal(array[i], one)


# ──────────────────────────────────────────────────────────────────────────
# Seat symmetry -- ENCODER_V2_SPEC §9.3, and the most valuable test here.
#
# Every seat is encoded by the same function, so the block describing player p
# must not depend on who is looking.  One helper reaching for
# ``state.sheets[p]`` instead of ``sheet_for(viewer, p)`` breaks symmetry and
# leaks information at the same time; this catches both at once.
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("players", [2, 3, 4])
def test_a_seats_block_is_the_same_whoever_encodes_it(players):
    state = _at_turn_boundary(players, turn=6)
    is_viewer = enc.block_slice("is_viewer")

    encodings = {v: enc.encode_state(state, v) for v in range(players)}
    for target in range(players):
        blocks = []
        for viewer in range(players):
            planes, sheets, _, _ = encodings[viewer]
            k = enc.seat_order(state, viewer).index(target)
            scalars = sheets[k].copy()
            scalars[is_viewer] = 0.0  # the one legitimate difference
            blocks.append((planes[k], scalars))
        for planes, scalars in blocks[1:]:
            assert np.array_equal(planes, blocks[0][0]), f"planes differ for {target}"
            assert np.array_equal(scalars, blocks[0][1]), f"scalars differ for {target}"


def test_the_seats_are_actually_different_from_each_other():
    """Guards the test above: identical sheets would make symmetry vacuous."""
    state = _at_turn_boundary(3, turn=6)
    _, sheets, _, _ = enc.encode_state(state, 0)
    assert not np.array_equal(sheets[0], sheets[1])


def test_the_viewer_is_first_on_the_seat_axis():
    state = _at_turn_boundary(3, turn=4)
    is_viewer = enc.block_slice("is_viewer")
    for viewer in range(3):
        assert enc.seat_order(state, viewer)[0] == viewer
        _, sheets, _, _ = enc.encode_state(state, viewer)
        assert list(sheets[:, is_viewer].ravel()) == [1.0, 0.0, 0.0, 0.0][:4]


def test_absent_seats_are_absent_rather_than_zero_valued():
    """M4: a padded seat contributes nothing, flagged, not a seat scoring zero."""
    state = _game(2)
    planes, sheets, _, glob = enc.encode_state(state, 0)
    valid = enc.block_slice("seat_valid")
    assert list(sheets[:, valid].ravel()) == [1.0, 1.0, 0.0, 0.0]
    assert np.all(sheets[2:] == 0.0)
    assert np.all(planes[2:] == 0.0)
    assert list(glob[enc.block_slice("seat_validity")]) == [1.0, 1.0, 0.0, 0.0]


# ──────────────────────────────────────────────────────────────────────────
# Sheet planes
# ──────────────────────────────────────────────────────────────────────────
def test_the_validity_plane_masks_the_short_streets():
    planes, _, _, _ = enc.encode_state(_game())
    assert planes[0, enc.P_VALID, 0].sum() == 10
    assert planes[0, enc.P_VALID, 1].sum() == 11
    assert planes[0, enc.P_VALID, 2].sum() == 12


def test_written_houses_show_up_on_the_occupancy_plane():
    state = _game()
    state.sheets[0].write(9, (1, 4), turn=1)
    planes, _, _, _ = enc.encode_state(state, 0)
    assert planes[0, enc.P_WRITTEN, 1, 4] == 1.0
    assert planes[0, enc.P_NUMBER, 1, 4] == np.float32(9 / 17.0)


def test_the_pool_plane_marks_the_nine_printed_pools():
    planes, _, _, _ = enc.encode_state(_game())
    assert planes[0, enc.P_POOL].sum() == 9


def test_the_span_plane_marks_dead_boxes():
    state = _game()
    sheet = state.sheets[0]
    sheet.write(5, (0, 3), turn=1)
    sheet.write(6, (0, 5), turn=1)
    planes, _, _, _ = enc.encode_state(state, 0)
    assert planes[0, enc.P_SPAN, 0, 4] == 0.0, "box 4 is stuck between a 5 and a 6"
    assert planes[0, enc.P_SPAN, 1, 0] > 0.0, "an untouched street still has room"


def test_the_positional_fit_plane_prefers_a_numbers_natural_home():
    state = _game()
    # every stack offers a 15, so "the numbers on offer" are unambiguous
    for slot in range(3):
        state.stack_new[0][slot] = _card_with(number=15)
        state.stack_old[0][slot] = _card_with(effect=Effect.SURVEYOR)

    plane = enc.encode_state(state, 0)[0][0, enc.P_FIT]
    assert plane[0, 9] > plane[0, 0], "a 15 belongs at the right-hand end"


def test_a_perfect_fit_is_the_strongest_fit_not_the_absence_of_one():
    """`positional_fit` returns 0.0 for a perfect fit and None for no fit.

    Truthiness collapses those two -- ``fit or -99.0`` turns the best possible
    placement into the worst -- so they have to be told apart explicitly.  A 6
    between a 5 and a 7 is the exactly-ideal box and must read 1.0, the maximum
    the plane can carry.
    """
    state = _game()
    sheet = state.sheets[0]
    sheet.write(5, (0, 3), turn=1)
    sheet.write(7, (0, 5), turn=1)
    for slot in range(3):
        state.stack_new[0][slot] = _card_with(number=6)
        state.stack_old[0][slot] = _card_with(effect=Effect.SURVEYOR)

    assert sheet.positional_fit(6, 0, 4) == 0.0, "the engine calls this perfect"
    plane = enc.encode_state(state, 0)[0][0, enc.P_FIT]
    assert plane[0, 4] == 1.0


def test_the_fit_plane_collapses_once_a_street_is_ruined():
    state = _game()
    before = enc.encode_state(state, 0)[0][0, enc.P_FIT, 0].sum()
    state.sheets[0].write(15, (0, 0), turn=1)
    after = enc.encode_state(state, 0)[0][0, enc.P_FIT, 0].sum()
    assert after < before, "nothing on offer fits well in what is left"


def test_the_writable_plane_is_live_before_a_combination_is_chosen():
    """§4.1: the v1 plane was blank at CHOOSE_CARDS -- exactly when it is needed."""
    state = _game()
    assert state.phase is Phase.CHOOSE_CARDS
    planes, _, viewer_plane, _ = enc.encode_state(state, 0)
    assert planes[0, enc.P_WRITABLE].sum() > 0.0
    assert viewer_plane.sum() == 0.0, "nothing is locked in yet"


def test_the_viewer_plane_carries_the_locked_in_choice():
    state = _game()
    state.stack_new[0][0] = _card_with(number=6)
    state.stack_old[0][0] = _card_with(effect=Effect.SURVEYOR)
    state.apply(codec.choose_stack(0))

    _, _, viewer_plane, _ = enc.encode_state(state, 0)
    legal = set(state.sheets[0].available_locations(6))
    assert legal
    for x in range(3):
        for y in range(12):
            assert viewer_plane[0, x, y] == float((x, y) in legal)


# ──────────────────────────────────────────────────────────────────────────
# Information-set safety
# ──────────────────────────────────────────────────────────────────────────
def test_next_turns_effect_reaches_the_encoding():
    """It is printed on the number face, so it is public and must be encoded.

    An earlier version of this engine treated it as hidden and asserted the
    opposite.  It is not hidden: swapping the top card of a stack for one with the
    same number and a different effect changes what the player knows about next
    turn, and the encoding has to move with it.
    """
    state = _game()
    state.stack_new[0][0] = _card_with(number=6, effect=Effect.SURVEYOR)
    a = enc.encode_state(state, 0)[3]

    alt = state.copy()
    alt.stack_new[0][0] = _card_with(number=6, effect=Effect.BIS)
    b = enc.encode_state(alt, 0)[3]

    block = enc.block_slice("next_effects")
    assert not np.array_equal(a[block], b[block])

    # the pair in play is untouched -- only next turn's promise moved
    assert state.visible_cards(0) == alt.visible_cards(0)
    stacks = enc.block_slice("stacks")
    assert np.array_equal(a[stacks], b[stacks])


def test_the_deck_order_cannot_change_the_encoding():
    state = _game()
    a = enc.encode_state(state, 0)
    alt = state.copy()
    alt.deck[alt.deck_pos:] = list(reversed(alt.deck[alt.deck_pos:]))
    b = enc.encode_state(alt, 0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_this_turns_houses_are_hidden_from_the_other_seats():
    """The other half of §9.3, which the boundary symmetry test cannot see.

    At a turn boundary the live sheet and the public snapshot are equal, so a
    helper reaching for ``state.sheets[seat]`` instead of ``sheet_for`` would
    pass the symmetry test unnoticed.  Mid-turn they differ, and that is the
    only moment the leak is visible -- so the comparison is made here, over the
    scalar block as well as the planes.
    """
    state = _game()
    state.stack_new[0][0] = _card_with(number=6)
    state.stack_old[0][0] = _card_with(effect=Effect.SURVEYOR)
    state.apply(codec.choose_stack(0))

    k = enc.seat_order(state, 1).index(0)
    before_planes, before_sheets, _, _ = enc.encode_state(state, 1)

    state.apply(codec.write(0, 0, 3))

    after_planes, after_sheets, _, _ = enc.encode_state(state, 1)
    assert np.array_equal(after_planes[k], before_planes[k]), "the house leaked"
    assert np.array_equal(after_sheets[k], before_sheets[k]), "a scalar leaked"
    assert after_planes[k, enc.P_WRITTEN].sum() == 0.0

    # player 0 sees it on their own block
    own_planes, own_sheets, _, _ = enc.encode_state(state, 0)
    assert own_planes[0, enc.P_WRITTEN, 0, 3] == 1.0
    free = enc.block_slice("free_boxes")
    assert own_sheets[0, free] < before_sheets[k, free], "the writer's own block moved"


# ──────────────────────────────────────────────────────────────────────────
# The races
# ──────────────────────────────────────────────────────────────────────────
def test_the_plan_block_moves_when_that_seat_advances():
    state = _game(2)
    plans = enc.block_slice("plans")
    before = enc.encode_state(state, 0)[1][1, plans]

    # opponent 1 crosses temp boxes; the snapshot other players see is
    # public_sheets, so move that too (a real turn boundary would)
    state.sheets[1].temps = 7
    state.public_sheets[1].temps = 7
    after = enc.encode_state(state, 0)[1][1, plans]

    if any(PLANS[p].kind is PlanKind.SEVEN_TEMP for p in state.plan_ids):
        assert not np.array_equal(before, after)
    else:  # no temp plan in this deal: the block is allowed to be unchanged
        assert before.shape == after.shape


def test_the_plan_block_reports_this_seats_own_progress():
    state = _game(2)
    state.sheets[0].temps = 7
    block = enc.encode_state(state, 0)[1][0, enc.block_slice("plans")].reshape(3, 3)
    for slot, plan_id in enumerate(state.plan_ids):
        fraction, _ = progress(PLANS[plan_id], state.sheets[0])
        assert block[slot, 0] == np.float32(fraction)
        assert block[slot, 2] == 0.0, "nothing is banked yet"


def test_capacity_block_falls_when_a_street_is_blocked():
    state = _game()
    capacity = enc.block_slice("capacity")
    before = enc.encode_state(state, 0)[1][0, capacity]
    state.sheets[0].write(15, (0, 0), turn=1)  # nine placements destroyed
    after = enc.encode_state(state, 0)[1][0, capacity]
    assert after[0] < before[0]
    assert after[3] < before[3], "the total must fall too"


def test_the_reshuffle_race_block_reports_the_option_is_open():
    state = _game()
    block = enc.encode_state(state, 0)[3][enc.block_slice("reshuffle_race")]
    assert block[0] == 1.0, "nobody has completed a plan yet"
    assert block[1] == 0.0, "and no reshuffle is queued"

    state.plan_turns[0][1] = state.turn - 1  # an opponent got there first
    block = enc.encode_state(state, 0)[3][enc.block_slice("reshuffle_race")]
    assert block[0] == 0.0, "the reshuffle option is gone"


def test_a_hidden_reshuffle_vote_does_not_reach_a_later_seat():
    """Transient global state leaks the same way a sheet does, and is easier to
    miss: the mid-turn sheet test above watches the per-seat block, and this one
    lives in `global_scalars`.

    The table-wide `reshuffle_next_turn` flips the moment any player votes yes,
    mid-turn, and is consumed at the start of the next turn -- so it is *never*
    legitimately public while true.  A later serial actor reading it would learn
    both that an earlier player voted, and that they completed a plan this turn,
    which `plan_turns_for` is at pains to hide.
    """
    state = _game(3, advanced=True)
    state.plan_turns[0][0] = state.turn  # seat 0 completed a plan THIS turn
    state.phase = Phase.ASK_RESHUFFLE
    state.actor = 0
    block = enc.block_slice("reshuffle_race")

    before = [enc.encode_state(state, v)[3][block].tolist() for v in range(3)]
    state.apply(codec.A_RESHUFFLE_YES)
    after = [enc.encode_state(state, v)[3][block].tolist() for v in range(3)]

    assert state.reshuffle_next_turn, "the vote did take effect in the engine"
    assert after[0][1] == 1.0, "the voter knows their own vote"
    assert before[1] == after[1] and before[2] == after[2], "the vote leaked"


def test_a_banked_plan_shows_on_the_seat_that_banked_it():
    state = _game(2)
    state.plan_turns[0][1] = state.turn - 1  # seat 1 completed plan slot 0
    _, sheets, _, glob = enc.encode_state(state, 0)
    plans = enc.block_slice("plans")
    assert sheets[0, plans].reshape(3, 3)[0, 2] == 0.0, "the viewer did not"
    assert sheets[1, plans].reshape(3, 3)[0, 2] == 1.0, "seat 1 did"

    identity = glob[enc.block_slice("plan_identity")].reshape(3, -1)
    assert identity[0, -1] == 0.0, "the first-place value is claimed"
    assert identity[1, -1] == 1.0, "the other slots are still open"
