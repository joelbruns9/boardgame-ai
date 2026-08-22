"""Engine flow: phases, effects, scoring and information sets."""
from __future__ import annotations

import random

import pytest

from games.welcome_to import action_codec as codec
from games.welcome_to.constants import (
    CARD_TABLE,
    Effect,
    NUM_BASE_CARDS,
    ROUNDABOUT,
    STREET_SIZES,
)
from games.welcome_to.game import GameConfig, GameState, IllegalAction, Phase
from games.welcome_to.plans import PLANS


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def _card_with(number=None, effect=None) -> int:
    for i, (n, e) in enumerate(CARD_TABLE[:NUM_BASE_CARDS]):
        if (number is None or n == number) and (effect is None or e is effect):
            return i
    raise AssertionError(f"no card number={number} effect={effect}")


def _force_combination(state: GameState, slot: int, number: int, effect: Effect) -> None:
    """Rig one standard-mode stack so a test can pick a known (number, effect)."""
    assert state.config.standard
    state.stack_new[0][slot] = _card_with(number=number)
    state.stack_old[0][slot] = _card_with(effect=effect)
    assert state.combination(slot) == (number, effect)


def _fill_sheet(state: GameState, player: int) -> None:
    sheet = state.sheets[player]
    for x, size in enumerate(STREET_SIZES):
        for y in range(size):
            sheet.numbers[x][y] = y


def _two_player() -> GameState:
    return GameState.new(seed=11, config=GameConfig(players=2))


# ──────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────
def test_a_fresh_standard_game_offers_three_stacks():
    state = _two_player()
    assert state.phase is Phase.CHOOSE_CARDS
    assert state.actor == 0
    assert state.turn == 1
    assert state.legal_actions() == [
        codec.choose_stack(0),
        codec.choose_stack(1),
        codec.choose_stack(2),
    ]


def test_standard_setup_fills_both_halves_of_every_stack():
    state = _two_player()
    for i in range(3):
        assert state.stack_new[0][i] is not None
        assert state.stack_old[0][i] is not None
        number, effect = state.combination(i)
        assert 1 <= number <= 15
        assert effect in tuple(Effect)[:6]


def test_three_distinct_plans_are_in_play():
    state = _two_player()
    assert len(state.plan_ids) == 3
    stacks = sorted(PLANS[pid].stack for pid in state.plan_ids)
    assert stacks == [1, 2, 3]


def test_expert_needs_two_players():
    with pytest.raises(ValueError):
        GameState.new(config=GameConfig(players=1, expert=True))


# ──────────────────────────────────────────────────────────────────────────
# Writing a number
# ──────────────────────────────────────────────────────────────────────────
def test_choosing_a_stack_moves_to_writing():
    state = _two_player()
    _force_combination(state, 0, number=5, effect=Effect.SURVEYOR)
    state.apply(codec.choose_stack(0))

    assert state.phase is Phase.WRITE_NUMBER
    assert state.ctx.number == 5
    assert state.ctx.effect is Effect.SURVEYOR
    # a non-temp effect writes exactly its own number, anywhere on an empty sheet
    assert len(state.legal_actions()) == 33
    for action in state.legal_actions():
        delta_slot, _, _ = codec.decode_write(action)
        assert delta_slot == 0


def test_temp_agency_offers_plus_or_minus_two():
    state = _two_player()
    _force_combination(state, 0, number=5, effect=Effect.TEMP)
    state.apply(codec.choose_stack(0))
    assert len(state.legal_actions()) == 5 * 33

    state.apply(codec.write(2, 0, 3))  # delta -1
    assert state.sheets[0].numbers[0][3] == 4
    assert state.sheets[0].temps == 1, "temp box is crossed automatically"


def test_temp_agency_cannot_leave_the_zero_to_seventeen_range():
    state = _two_player()
    _force_combination(state, 0, number=1, effect=Effect.TEMP)
    state.apply(codec.choose_stack(0))
    numbers = {
        state.ctx.number + [0, -2, -1, 1, 2][codec.decode_write(a)[0]]
        for a in state.legal_actions()
    }
    assert numbers == {0, 1, 2, 3}, "1 - 2 would be negative"


def test_writing_respects_the_ascending_rule():
    state = _two_player()
    _force_combination(state, 0, number=8, effect=Effect.SURVEYOR)
    state.sheets[0].write(8, (0, 4), turn=1)
    state.apply(codec.choose_stack(0))
    boxes = {codec.decode_write(a)[1:] for a in state.legal_actions()}
    assert (0, 3) not in boxes and (0, 5) not in boxes
    assert (1, 0) in boxes


# ──────────────────────────────────────────────────────────────────────────
# Effects
# ──────────────────────────────────────────────────────────────────────────
def test_surveyor_offers_every_open_fence_slot():
    state = _two_player()
    _force_combination(state, 0, number=5, effect=Effect.SURVEYOR)
    state.apply(codec.choose_stack(0))
    state.apply(codec.write(0, 0, 3))

    assert state.phase is Phase.ACTION_SURVEYOR
    assert len(state.legal_actions()) == 31  # 30 slots + pass
    state.apply(codec.surveyor_fence(1, 4))
    assert state.sheets[0].fences[1][4]


def test_estate_effect_offers_the_six_value_rows():
    state = _two_player()
    _force_combination(state, 0, number=5, effect=Effect.ESTATE)
    state.apply(codec.choose_stack(0))
    state.apply(codec.write(0, 0, 3))

    assert state.phase is Phase.ACTION_ESTATE
    assert len(state.legal_actions()) == 7
    state.apply(codec.estate_row(3))
    assert state.sheets[0].estate_marks[3] == 1


def test_park_is_offered_only_on_the_street_just_built_on():
    state = _two_player()
    _force_combination(state, 0, number=5, effect=Effect.PARK)
    state.apply(codec.choose_stack(0))
    state.apply(codec.write(0, 1, 5))

    assert state.phase is Phase.ACTION_PARK
    assert state.legal_actions() == [codec.park_street(1), codec.A_PASS_PARK]
    state.apply(codec.park_street(1))
    assert state.sheets[0].parks == [0, 1, 0]


def test_park_is_skipped_when_its_street_is_finished():
    state = _two_player()
    state.sheets[0].parks = [0, 4, 0]  # street 1 is full
    _force_combination(state, 0, number=5, effect=Effect.PARK)
    state.apply(codec.choose_stack(0))
    state.apply(codec.write(0, 1, 5))
    assert state.phase is not Phase.ACTION_PARK


def test_pool_needs_the_house_to_sit_on_one():
    state = _two_player()
    _force_combination(state, 0, number=5, effect=Effect.POOL)
    state.apply(codec.choose_stack(0))
    state.apply(codec.write(0, 0, 2))  # (0, 2) is a printed pool

    assert state.phase is Phase.ACTION_POOL
    assert state.legal_actions() == [codec.A_POOL_BUILD, codec.A_PASS_POOL]
    state.apply(codec.A_POOL_BUILD)
    assert state.sheets[0].pools == [1, 0, 0]


def test_pool_is_skipped_off_a_pool_box():
    state = _two_player()
    _force_combination(state, 0, number=5, effect=Effect.POOL)
    state.apply(codec.choose_stack(0))
    state.apply(codec.write(0, 0, 3))  # not a pool
    assert state.phase is not Phase.ACTION_POOL
    assert state.sheets[0].pools == [0, 0, 0]


def test_bis_duplicates_a_neighbour_and_costs_a_penalty_box():
    state = _two_player()
    _force_combination(state, 0, number=5, effect=Effect.BIS)
    state.apply(codec.choose_stack(0))
    state.apply(codec.write(0, 0, 3))

    assert state.phase is Phase.ACTION_BIS
    legal = set(state.legal_actions())
    assert codec.bis(0, 2, 1) in legal
    assert codec.bis(0, 4, 0) in legal
    assert codec.A_PASS_BIS in legal

    state.apply(codec.bis(0, 4, 0))
    sheet = state.sheets[0]
    assert sheet.numbers[0][4] == 5
    assert sheet.is_bis[0][4]
    assert sheet.bis_marks == 1


def test_bis_may_be_declined():
    state = _two_player()
    _force_combination(state, 0, number=5, effect=Effect.BIS)
    state.apply(codec.choose_stack(0))
    state.apply(codec.write(0, 0, 3))
    state.apply(codec.A_PASS_BIS)
    assert state.sheets[0].bis_marks == 0


# ──────────────────────────────────────────────────────────────────────────
# Permit refusal
# ──────────────────────────────────────────────────────────────────────────
def test_a_full_sheet_forces_a_permit_refusal():
    state = _two_player()
    _fill_sheet(state, 0)
    assert state.legal_actions() == [codec.A_PERMIT_REFUSAL]

    state.apply(codec.A_PERMIT_REFUSAL)
    assert state.sheets[0].permits == 1
    assert state.actor == 1, "a refusal ends the turn"


def test_refusal_stays_open_rather_than_forcing_the_temp_agency():
    state = _two_player()
    _force_combination(state, 0, number=5, effect=Effect.TEMP)
    # every street blocked for 5, but 3 / 4 / 6 / 7 still fit
    sheet = state.sheets[0]
    for x, size in enumerate(STREET_SIZES):
        sheet.numbers[x][0] = 5
    state.apply(codec.choose_stack(0))

    assert sheet.available_locations(5) == []
    assert codec.A_PERMIT_REFUSAL in state.legal_actions()


def test_refusal_is_not_offered_when_the_plain_number_fits():
    state = _two_player()
    _force_combination(state, 0, number=5, effect=Effect.TEMP)
    state.apply(codec.choose_stack(0))
    assert codec.A_PERMIT_REFUSAL not in state.legal_actions()


# ──────────────────────────────────────────────────────────────────────────
# Roundabouts (advanced variant)
# ──────────────────────────────────────────────────────────────────────────
def _advanced_game() -> GameState:
    return GameState.new(seed=12, config=GameConfig(players=2, advanced=True))


def test_roundabouts_only_exist_in_the_advanced_variant():
    assert codec.A_ROUNDABOUT_OPEN not in _two_player().legal_actions()
    assert codec.A_ROUNDABOUT_OPEN in _advanced_game().legal_actions()


def test_building_a_roundabout_returns_to_the_card_choice():
    state = _advanced_game()
    state.apply(codec.A_ROUNDABOUT_OPEN)
    assert state.phase is Phase.ROUNDABOUT_PLACE
    assert len(state.legal_actions()) == 34  # 33 boxes + pass

    state.apply(codec.roundabout_pos(1, 4))
    assert state.phase is Phase.CHOOSE_CARDS
    assert state.sheets[0].numbers[1][4] == ROUNDABOUT
    assert state.sheets[0].roundabouts == 1
    assert state.sheets[0].fences[1][3] and state.sheets[0].fences[1][4]
    assert codec.A_ROUNDABOUT_OPEN not in state.legal_actions(), "one per turn"


def test_a_third_roundabout_is_never_offered():
    state = _advanced_game()
    state.sheets[0].roundabouts = 2
    assert codec.A_ROUNDABOUT_OPEN not in state.legal_actions()


# ──────────────────────────────────────────────────────────────────────────
# Turn structure
# ──────────────────────────────────────────────────────────────────────────
def test_the_turn_advances_once_every_seat_has_played():
    state = _two_player()
    steps = 0
    while state.turn == 1 and not state.is_terminal:
        state.apply(state.legal_actions()[0])
        steps += 1
        assert steps < 200
    assert state.turn == 2
    assert state.actor == 0


def test_stacks_rotate_so_this_turns_number_card_shows_its_effect_next_turn():
    state = _two_player()
    fresh = list(state.stack_new[0])
    while state.turn == 1 and not state.is_terminal:
        state.apply(state.legal_actions()[0])
    assert state.stack_old[0] == fresh
    assert all(c not in fresh for c in state.stack_new[0])


def test_illegal_actions_are_rejected():
    state = _two_player()
    with pytest.raises(IllegalAction):
        state.apply(codec.A_PASS_PLAN)


# ──────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────
def test_finishing_a_plan_on_the_same_turn_shares_first_place():
    state = GameState.new(seed=3, config=GameConfig(players=3))
    state.plan_turns[0] = {0: 2, 1: 2, 2: 5}
    plan = PLANS[state.plan_ids[0]]
    scores = state.plan_scores()
    assert scores[0] == plan.scores[0]
    assert scores[1] == plan.scores[0]
    assert scores[2] == plan.scores[1]


def test_the_solo_ghost_never_scores_but_does_take_first_place():
    state = GameState.new(seed=3, config=GameConfig(players=1))
    state.plan_turns[0] = {-1: 1, 0: 4}
    plan = PLANS[state.plan_ids[0]]
    assert state.plan_scores() == [plan.scores[1]]


def test_temp_agency_is_scored_by_rank():
    state = GameState.new(seed=4, config=GameConfig(players=3))
    state.sheets[0].temps = 5
    state.sheets[1].temps = 5
    state.sheets[2].temps = 2
    assert state.temp_scores() == [7, 7, 4]

    state.sheets[0].temps = 0
    assert state.temp_scores() == [0, 7, 4], "no temps means no rank at all"


def test_solo_temp_agency_is_a_flat_threshold():
    state = GameState.new(seed=4, config=GameConfig(players=1))
    state.sheets[0].temps = 5
    assert state.temp_scores() == [0]
    state.sheets[0].temps = 6
    assert state.temp_scores() == [7]


def test_total_is_the_sum_minus_the_three_penalties():
    state = GameState.new(seed=6, config=GameConfig(players=2))
    sheet = state.sheets[0]
    sheet.parks = [3, 0, 0]  # +10
    sheet.pools = [1, 0, 0]  # +3
    sheet.bis_marks = 2  # -3
    sheet.permits = 2  # -3
    breakdown = state.score_breakdown(0)
    assert breakdown.parks == 10
    assert breakdown.pools == 3
    assert breakdown.total == 10 + 3 - 3 - 3


# ──────────────────────────────────────────────────────────────────────────
# Information sets
# ──────────────────────────────────────────────────────────────────────────
def test_other_players_sheets_are_frozen_at_the_start_of_the_turn():
    state = _two_player()
    _force_combination(state, 0, number=5, effect=Effect.SURVEYOR)
    state.apply(codec.choose_stack(0))
    state.apply(codec.write(0, 0, 3))

    # player 1 must not see what player 0 just wrote
    assert state.sheet_for(1, 0).numbers[0][3] is None
    # player 0 sees their own live sheet
    assert state.sheet_for(0, 0).numbers[0][3] == 5


def test_plan_completions_this_turn_are_private():
    state = _two_player()
    state.plan_turns[0] = {0: state.turn, 1: state.turn - 1}
    assert state.plan_turns_for(1, 0) == {1: state.turn - 1}
    assert state.plan_turns_for(0, 0) == {0: state.turn, 1: state.turn - 1}


def test_copy_is_independent():
    state = _two_player()
    clone = state.copy()
    clone.apply(clone.legal_actions()[0])
    assert state.phase is Phase.CHOOSE_CARDS
    assert state.sheets[0].numbers == GameState.new(
        seed=11, config=GameConfig(players=2)
    ).sheets[0].numbers


def test_step_does_not_mutate_the_original():
    state = _two_player()
    before = state.phase
    state.step(state.legal_actions()[0])
    assert state.phase is before


def test_redeterminize_keeps_everything_visible_and_reshuffles_the_rest():
    state = _two_player()
    for _ in range(6):  # get a few cards into the discard
        state.apply(state.legal_actions()[0])

    alt = state.redeterminize(random.Random(7))
    assert alt.visible_cards() == state.visible_cards()
    assert alt.discard == state.discard
    assert alt.stack_old == state.stack_old
    assert alt.deck_pos == state.deck_pos

    def pool(s):
        return sorted(s.deck[s.deck_pos:] + [c for c in s.stack_new[0] if c is not None])

    assert pool(alt) == pool(state), "cards are permuted, never invented"


# ──────────────────────────────────────────────────────────────────────────
# Variants
# ──────────────────────────────────────────────────────────────────────────
def test_expert_mode_offers_six_ordered_card_pairs():
    state = GameState.new(seed=5, config=GameConfig(players=2, expert=True))
    assert state.config.choice_slots == 6
    assert state.legal_actions() == [codec.choose_stack(s) for s in range(6)]
    assert state.stack_old == []


def test_solo_mode_also_drafts_two_of_three_cards():
    state = GameState.new(seed=5, config=GameConfig(players=1))
    assert state.config.solo and not state.config.standard
    assert state.legal_actions() == [codec.choose_stack(s) for s in range(6)]


def test_solo_deck_carries_the_extra_card():
    state = GameState.new(seed=5, config=GameConfig(players=1))
    assert state.deck_remaining + 3 == 82


def test_declining_a_roundabout_is_sticky():
    """Otherwise a bot can oscillate CHOOSE_CARDS <-> ROUNDABOUT_PLACE forever.

    BGA's state machine sends ``pass`` from ST_ROUNDABOUT straight back to
    ST_CHOOSE_CARDS, where the offer is live again.  A human would never loop on
    it; greedy did, 1489 times in one measured game, and MCTS would grow an
    infinite no-op branch.  Declining is therefore made sticky for the turn.
    """
    state = _advanced_game()
    assert codec.A_ROUNDABOUT_OPEN in state.legal_actions()

    state.apply(codec.A_ROUNDABOUT_OPEN)
    state.apply(codec.A_PASS_ROUNDABOUT)

    assert state.phase is Phase.CHOOSE_CARDS
    assert codec.A_ROUNDABOUT_OPEN not in state.legal_actions()
    assert state.ctx.roundabout_declined


def test_the_offer_returns_on_the_next_turn():
    state = _advanced_game()
    state.apply(codec.A_ROUNDABOUT_OPEN)
    state.apply(codec.A_PASS_ROUNDABOUT)
    while state.turn == 1 and not state.is_terminal:
        state.apply(state.legal_actions()[0])
    assert not state.ctx.roundabout_declined
    assert codec.A_ROUNDABOUT_OPEN in state.legal_actions()


def test_a_greedy_advanced_game_terminates_promptly():
    """The livelock regression: this used to run for thousands of no-op steps."""
    from games.welcome_to.bots import GreedyBot

    state = GameState.new(
        seed=0, config=GameConfig(players=1, advanced=True, solo_rules=False)
    )
    bot = GreedyBot(random.Random(0))
    steps = 0
    while not state.is_terminal:
        state.apply(bot.act(state))
        steps += 1
        assert steps < 400, "roundabout livelock is back"


def test_redeterminize_gives_a_different_world_each_call():
    """Regression: it used to return the same shuffle every time.

    ``copy()`` clones the RNG state exactly, so when ``rng`` defaulted to the
    copy's own generator every determinization started from the same seed.  A
    search built on that explores one fixed future while looking healthy, so
    ``rng`` is now required and the caller must advance it.
    """
    state = _two_player()
    for _ in range(6):
        state.apply(state.legal_actions()[0])

    search_rng = random.Random(1234)
    worlds = {
        tuple(state.redeterminize(search_rng).deck[state.deck_pos :])
        for _ in range(20)
    }
    assert len(worlds) > 15, f"only {len(worlds)} distinct worlds in 20 draws"


def test_redeterminize_requires_an_rng():
    state = _two_player()
    with pytest.raises(TypeError):
        state.redeterminize()  # type: ignore[call-arg]


def test_redeterminize_gives_the_copy_its_own_forward_rng():
    """Two determinizations must not share RNG state into the rollout.

    ``_reform_deck`` uses ``state.rng``; if both copies carried the source's
    generator they would apply the same permutation pattern at a mid-rollout
    reshuffle, correlating simulations that are meant to be independent.
    """
    state = _two_player()
    for _ in range(6):
        state.apply(state.legal_actions()[0])

    search_rng = random.Random(99)
    a = state.redeterminize(search_rng)
    b = state.redeterminize(search_rng)
    assert a.rng.getstate() != b.rng.getstate()
    assert a.rng.getstate() != state.rng.getstate()
