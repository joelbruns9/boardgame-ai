"""Encoder shape, determinism and information-set safety."""
from __future__ import annotations

import numpy as np

from games.welcome_to import action_codec as codec
from games.welcome_to import encoder as enc
from games.welcome_to.constants import CARD_TABLE, Effect, NUM_BASE_CARDS
from games.welcome_to.plans import progress
from games.welcome_to.game import GameConfig, GameState


def _card_with(number=None, effect=None) -> int:
    for i, (n, e) in enumerate(CARD_TABLE[:NUM_BASE_CARDS]):
        if (number is None or n == number) and (effect is None or e is effect):
            return i
    raise AssertionError(f"no card number={number} effect={effect}")


def _game(players: int = 2, **kwargs) -> GameState:
    return GameState.new(seed=21, config=GameConfig(players=players, **kwargs))


# ──────────────────────────────────────────────────────────────────────────
# Shape
# ──────────────────────────────────────────────────────────────────────────
def test_encoding_has_the_declared_shape():
    spatial, scalar = enc.encode_state(_game())
    assert spatial.shape == enc.SPATIAL_SHAPE
    assert scalar.shape == (enc.NUM_SCALAR,)
    assert spatial.dtype == np.float32 and scalar.dtype == np.float32


def test_scalar_blocks_add_up():
    assert enc.NUM_SCALAR == sum(size for _, size in enc.SCALAR_BLOCKS)


def test_shape_does_not_depend_on_the_player_count():
    shapes = set()
    for players in (1, 2, 3, 4):
        spatial, scalar = enc.encode_state(_game(players))
        shapes.add((spatial.shape, scalar.shape))
    assert len(shapes) == 1


def test_encoding_is_finite_and_bounded():
    state = _game(4)
    for _ in range(40):
        if state.is_terminal:
            break
        spatial, scalar = enc.encode_state(state)
        assert np.isfinite(spatial).all() and np.isfinite(scalar).all()
        assert spatial.min() >= -1.0 and spatial.max() <= 6.0
        assert scalar.min() >= -2.0 and scalar.max() <= 3.0
        state.apply(state.legal_actions()[0])


def test_encoding_is_deterministic():
    state = _game(3)
    a = enc.encode_state(state, 1)
    b = enc.encode_state(state, 1)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_batch_matches_single():
    states = [_game(2), _game(3)]
    spatial, scalar = enc.encode_batch(states)
    for i, s in enumerate(states):
        one_spatial, one_scalar = enc.encode_state(s)
        assert np.array_equal(spatial[i], one_spatial)
        assert np.array_equal(scalar[i], one_scalar)


# ──────────────────────────────────────────────────────────────────────────
# Board planes
# ──────────────────────────────────────────────────────────────────────────
def test_the_validity_plane_masks_the_short_streets():
    spatial, _ = enc.encode_state(_game())
    assert spatial[0, 0].sum() == 10
    assert spatial[0, 1].sum() == 11
    assert spatial[0, 2].sum() == 12


def test_written_houses_show_up_on_the_occupancy_plane():
    state = _game()
    state.sheets[0].write(9, (1, 4), turn=1)
    spatial, _ = enc.encode_state(state, 0)
    assert spatial[1, 1, 4] == 1.0
    assert spatial[2, 1, 4] == np.float32(9 / 17.0)


def test_the_pool_plane_marks_the_nine_printed_pools():
    spatial, _ = enc.encode_state(_game())
    assert spatial[7].sum() == 9


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
    a_scalar = enc.encode_state(state, 0)[1]

    alt = state.copy()
    alt.stack_new[0][0] = _card_with(number=6, effect=Effect.BIS)
    b_scalar = enc.encode_state(alt, 0)[1]

    block = enc.block_slice("next_effects")
    assert not np.array_equal(a_scalar[block], b_scalar[block])

    # the pair in play is untouched -- only next turn's promise moved
    assert state.visible_cards(0) == alt.visible_cards(0)
    stacks = enc.block_slice("stacks")
    assert np.array_equal(a_scalar[stacks], b_scalar[stacks])


def test_the_deck_order_cannot_change_the_encoding():
    state = _game()
    a = enc.encode_state(state, 0)
    alt = state.copy()
    alt.deck[alt.deck_pos:] = list(reversed(alt.deck[alt.deck_pos:]))
    b = enc.encode_state(alt, 0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_this_turns_houses_are_hidden_from_the_other_seats():
    state = _game()
    state.stack_new[0][0] = _card_with(number=6)
    state.stack_old[0][0] = _card_with(effect=Effect.SURVEYOR)
    state.apply(codec.choose_stack(0))
    state.apply(codec.write(0, 0, 3))

    # player 1 looks at player 0: opponent planes start after the own-sheet ones
    spatial, _ = enc.encode_state(state, 1)
    assert spatial[enc.OWN_PLANES].sum() == 0.0, "player 0's new house leaked"

    # player 0 sees it on their own planes
    own, _ = enc.encode_state(state, 0)
    assert own[1, 0, 3] == 1.0


def test_the_plan_race_block_moves_when_an_opponent_advances():
    state = _game(2)
    before = enc.encode_state(state, 0)[1][enc.block_slice("plan_race")]

    # opponent 1 crosses temp boxes; the snapshot other players see is
    # public_sheets, so move that too (a real turn boundary would)
    state.sheets[1].temps = 7
    state.public_sheets[1].temps = 7
    after = enc.encode_state(state, 0)[1][enc.block_slice("plan_race")]

    plans = [p for p in state.plan_ids]
    from games.welcome_to.plans import PLANS, PlanKind

    if any(PLANS[p].kind is PlanKind.SEVEN_TEMP for p in plans):
        assert not np.array_equal(before, after)
    else:  # no temp plan in this deal: the block is allowed to be unchanged
        assert before.shape == after.shape


def test_plan_race_reports_the_viewer_first():
    state = _game(2)
    state.sheets[0].temps = 7
    block = enc.encode_state(state, 0)[1][enc.block_slice("plan_race")]
    # layout is 3 plans x 4 seats x 2 values, viewer at seat offset 0
    reshaped = block.reshape(3, 4, 2)
    for slot, plan_id in enumerate(state.plan_ids):
        from games.welcome_to.plans import PLANS

        fraction, _ = progress(PLANS[plan_id], state.sheets[0])
        assert reshaped[slot, 0, 0] == np.float32(fraction)


def test_absent_seats_leave_their_blocks_at_zero():
    state = _game(2)
    block = enc.encode_state(state, 0)[1][enc.block_slice("plan_race")].reshape(3, 4, 2)
    assert np.all(block[:, 2:, :] == 0.0)


def test_capacity_block_falls_when_a_street_is_blocked():
    state = _game()
    before = enc.encode_state(state, 0)[1][enc.block_slice("capacity")]
    state.sheets[0].write(15, (0, 0), turn=1)  # nine placements destroyed
    after = enc.encode_state(state, 0)[1][enc.block_slice("capacity")]
    assert after[0] < before[0]
    assert after[3] < before[3], "the total must fall too"


def test_the_span_plane_marks_dead_boxes():
    state = _game()
    sheet = state.sheets[0]
    sheet.write(5, (0, 3), turn=1)
    sheet.write(6, (0, 5), turn=1)
    spatial = enc.encode_state(state, 0)[0]
    assert spatial[10, 0, 4] == 0.0, "box 4 is stuck between a 5 and a 6"
    assert spatial[10, 1, 0] > 0.0, "an untouched street still has room"


def test_the_reshuffle_race_block_reports_the_option_is_open():
    state = _game()
    block = enc.encode_state(state, 0)[1][enc.block_slice("reshuffle_race")]
    assert block[0] == 1.0, "nobody has completed a plan yet"
    assert block[1] == 0.0, "and no reshuffle is queued"

    state.plan_turns[0][1] = state.turn - 1  # an opponent got there first
    block = enc.encode_state(state, 0)[1][enc.block_slice("reshuffle_race")]
    assert block[0] == 0.0, "the reshuffle option is gone"


def test_the_positional_fit_plane_prefers_a_numbers_natural_home():
    state = _game()
    # rig the offer so exactly one number is available and it is a high one
    state.stack_new[0][0] = _card_with(number=15)
    state.stack_old[0][0] = _card_with(effect=Effect.SURVEYOR)
    state.apply(codec.choose_stack(0))

    plane = enc.encode_state(state, 0)[0][11]
    left, right = plane[0, 0], plane[0, 9]
    assert right > left, "a 15 belongs at the right-hand end of a street"


def test_the_fit_plane_collapses_once_a_street_is_ruined():
    state = _game()
    before = enc.encode_state(state, 0)[0][11, 0].sum()
    state.sheets[0].write(15, (0, 0), turn=1)
    after = enc.encode_state(state, 0)[0][11, 0].sum()
    assert after < before, "nothing on offer fits well in what is left"
