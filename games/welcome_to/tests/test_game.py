"""Engine flow: phases, effects, scoring and information sets."""
from __future__ import annotations

import collections
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
from games.welcome_to.bots import GreedyBot
from games.welcome_to.game import (
    BoundaryOutcome,
    GameConfig,
    GameState,
    IllegalAction,
    Phase,
)
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


def test_reshuffle_votes_are_recorded_per_seat_and_cleared_each_turn():
    """The table-wide flag is the OR of these; only the votes are viewer-safe."""
    from games.welcome_to.game import Phase

    state = GameState.new(seed=3, config=GameConfig(players=3, advanced=True))
    assert state.reshuffle_votes == {}
    assert not state.reshuffle_vote_for(0)

    state.plan_turns[0][1] = state.turn
    state.phase = Phase.ASK_RESHUFFLE
    state.actor = 1
    state.apply(codec.A_RESHUFFLE_YES)

    assert state.reshuffle_votes == {1: True}
    assert state.reshuffle_vote_for(1) and not state.reshuffle_vote_for(0)
    assert state.reshuffle_next_turn

    copied = state.copy()
    copied.reshuffle_votes[2] = True
    assert state.reshuffle_votes == {1: True}, "copy() must not share the dict"


# ──────────────────────────────────────────────────────────────────────────
# The turn boundary, in three parts -- SEARCH_SPEC.md §6.3
#
# The point of the split is that a search must never reimplement the draw: a
# boundary does four different things and a reimplementation is a correctness
# bug waiting on a reshuffle turn.  So there is a test per case, and one that
# says the three-part path *is* the engine's own path.
# ──────────────────────────────────────────────────────────────────────────
def _mid_game(players: int = 2, turn: int = 4, seed: int = 3) -> GameState:
    bots = [GreedyBot(random.Random(seed * 10 + i)) for i in range(players)]
    state = GameState.new(seed=seed, config=GameConfig(players=players, advanced=True))
    while not state.is_terminal and state.turn < turn:
        state.apply(bots[state.actor].act(state))
    assert not state.is_terminal
    return state


def _afterstate(**kwargs) -> GameState:
    """A boundary afterstate: everything settled except which cards come off."""
    state = _mid_game(**kwargs)
    assert state.prepare_turn_boundary(), "the game ended instead"
    return state


def _live_cards(state: GameState) -> collections.Counter:
    """Every card that still exists.  ``deck[:deck_pos]`` is already dealt."""
    cards = list(state.deck[state.deck_pos :]) + list(state.discard)
    for group in list(state.stack_new) + list(state.stack_old):
        cards += [c for c in group if c is not None]
    cards += [c for c in state.expert_pending if c is not None]
    return collections.Counter(cards)


def test_prepare_promotes_the_numbers_and_reveals_nothing():
    """§6.2's certainty, as a transition: *this* turn's numbers become *next*
    turn's effects, before any card is drawn.  That is why ``next_effects`` is
    known rather than guessed, and why the table is never reshuffled material."""
    state = _mid_game()
    numbers = list(state.stack_new[0])
    aside = [c for c in state.stack_old[0] if c is not None]
    turn, discarded = state.turn, len(state.discard)

    assert state.prepare_turn_boundary() is True
    assert state.turn == turn + 1
    assert state.stack_old[0] == numbers, "this turn's numbers were not promoted"
    assert all(c is None for c in state.stack_new[0]), "a card was revealed early"
    assert len(state.discard) == discarded + len(aside)


def test_an_ordinary_boundary_draws_three():
    state = _afterstate()
    assert not state.reshuffle_next_turn and state.deck_remaining >= 3
    outcome = state.sample_boundary_outcome(random.Random(0))
    assert len(outcome.draws) == 3
    assert outcome.reformed is False


def test_an_exact_empty_deck_reforms_and_still_draws_three():
    """The deck is empty *at the start* of the draw, so ``_draw`` reforms first.

    Reachable and not rare: consumption is a multiple of three and reform
    happens at zero, so the deck passes through empty once per cycle.
    """
    state = _afterstate()
    state.discard.extend(state.deck[state.deck_pos :])
    state.deck_pos = len(state.deck)
    assert state.deck_remaining == 0 and state.discard

    before = _live_cards(state)
    outcome = state.sample_boundary_outcome(random.Random(0))
    assert len(outcome.draws) == 3
    assert outcome.reformed is True

    state.apply_boundary_outcome(outcome)
    assert _live_cards(state) == before, "the reform invented or lost a card"
    assert state.deck_remaining > 0


def test_a_queued_reshuffle_draws_six_in_two_batches():
    """⚠ The case a "draw three from the histogram" search gets wrong.

    ``_reshuffle_decks`` reforms, draws **3**, runs a discard cycle -- which
    promotes those three into the aside slots -- and then ``_draw_step`` draws
    **3 more**.  So both halves of every stack are freshly drawn, and the
    boundary reveals six cards, not three.
    """
    state = _afterstate()
    state.reshuffle_next_turn = True
    before = _live_cards(state)

    outcome = state.sample_boundary_outcome(random.Random(0))
    assert len(outcome.draws) == 6, "a reshuffle boundary is not three cards"
    assert outcome.reformed is True

    state.apply_boundary_outcome(outcome)
    assert state.reshuffle_next_turn is False, "the queued reshuffle was not consumed"
    assert _live_cards(state) == before
    assert list(state.stack_old[0]) == list(outcome.draws[:3])
    assert list(state.stack_new[0]) == list(outcome.draws[3:])


def test_a_boundary_can_end_the_game_before_revealing_anything():
    """The fourth case, and the one a search that assumes "a boundary reveals
    cards" gets wrong: ``prepare_turn_boundary`` returns False and there is no
    outcome to sample at all."""
    state = _mid_game()
    _fill_sheet(state, 0)  # ``isEndOfGame``: a player with no free box

    assert state.prepare_turn_boundary() is False
    assert state.is_terminal and state.phase is Phase.GAME_OVER
    with pytest.raises(IllegalAction):
        state.sample_boundary_outcome(random.Random(0))


@pytest.mark.parametrize("case", ["ordinary", "exact-empty reform", "queued reshuffle"])
def test_the_three_part_boundary_is_the_engine_s_own_boundary(case):
    """The guarantee the whole split exists for, on each of the three reveal
    cases: ``prepare`` / ``apply`` must land on exactly the state the engine
    reaches by itself.

    The control side records what it drew (through the same ``draw`` hook
    ``sample_boundary_outcome`` uses, which is why a reform is not a special
    case here) and the split side replays it, so the *only* thing that can
    differ is the logic.
    """
    compared = 0
    for seed in range(6):
        for players in (2, 3):
            control = _afterstate(players=players, seed=seed)
            if case == "exact-empty reform":
                control.discard.extend(control.deck[control.deck_pos :])
                control.deck_pos = len(control.deck)
            elif case == "queued reshuffle":
                control.reshuffle_next_turn = True
            split = control.copy()

            drawn: list[int] = []
            real = control._draw

            def record() -> int:
                card = real()
                drawn.append(card)
                return card

            control._reveal_step(record)
            control._open_turn()

            split.apply_boundary_outcome(BoundaryOutcome(draws=tuple(drawn)))
            compared += 1

            assert split.turn == control.turn
            assert split.actor == control.actor
            assert split.phase is control.phase
            assert split.stack_new == control.stack_new
            assert split.stack_old == control.stack_old
            assert split.table_cards(0) == control.table_cards(0)
            assert split.reshuffle_next_turn == control.reshuffle_next_turn
            assert split.deck_remaining == control.deck_remaining
            assert split.solo_card_drawn == control.solo_card_drawn
            assert _live_cards(split) == _live_cards(control)
            assert split.legal_actions() == control.legal_actions()
    assert compared == 12, "the comparison was skipped"


def test_the_reformed_flag_means_the_deck_was_actually_reformed():
    """Pinned against the real call, because it is inferable and must not be.

    ``_reform_deck`` has exactly two call sites -- ``_reshuffle_decks``, which
    always reforms, and ``_draw``, which reforms on an empty undrawn region --
    and both are checked directly.  Inferring it from how ``deck_pos`` moved
    reads plausible and is wrong: a reform resets the cursor, so the arithmetic
    stops meaning anything.  Verified 4,329/4,329 over 50 games at 2 and 3 seats.
    """
    calls = []
    original = GameState._reform_deck

    def spy(self):
        calls.append(1)
        original(self)

    GameState._reform_deck = spy
    try:
        for seed in (3, 5, 8):
            base = _afterstate(seed=seed)
            for variant in (None, "empty", "queued"):
                state = base.copy()
                if variant == "empty":
                    state.discard.extend(state.deck[state.deck_pos :])
                    state.deck_pos = len(state.deck)
                elif variant == "queued":
                    state.reshuffle_next_turn = True
                calls.clear()
                outcome = state.sample_boundary_outcome(random.Random(7))
                assert outcome.reformed is bool(calls), (
                    f"{variant}: flag {outcome.reformed}, "
                    f"actual reforms {len(calls)}"
                )
    finally:
        GameState._reform_deck = original


def test_sampling_an_outcome_does_not_touch_the_afterstate():
    """A chance node samples many outcomes from one afterstate; if sampling
    consumed the deck, the second sample would be drawn from a different game."""
    state = _afterstate()
    before = (
        state.deck_pos,
        list(state.deck),
        list(state.discard),
        [list(g) for g in state.stack_new],
        [list(g) for g in state.stack_old],
        state.reshuffle_next_turn,
    )
    outcomes = {state.sample_boundary_outcome(random.Random(k)).draws for k in range(8)}
    after = (
        state.deck_pos,
        list(state.deck),
        list(state.discard),
        [list(g) for g in state.stack_new],
        [list(g) for g in state.stack_old],
        state.reshuffle_next_turn,
    )
    assert before == after, "sampling mutated the afterstate"
    assert len(outcomes) > 1, "repeated samples are not independent"


def test_an_outcome_carries_only_cards_the_boundary_makes_public():
    """§7.3's non-anticipativity rule, held structurally.

    Every card in an outcome ends up face-up on the table, and the order of what
    is left is not recorded -- so an outcome *cannot* leak the future deck, and
    a scenario built from one cannot become clairvoyant by accident.
    """
    for reshuffle in (False, True):
        state = _afterstate()
        state.reshuffle_next_turn = reshuffle
        outcome = state.sample_boundary_outcome(random.Random(2))
        state.apply_boundary_outcome(outcome)
        assert set(outcome.draws) <= set(state.table_cards(0))


def test_applying_an_outcome_is_deterministic():
    state = _afterstate()
    outcome = state.sample_boundary_outcome(random.Random(4))
    ends = []
    for _ in range(3):
        applied = state.copy()
        applied.apply_boundary_outcome(outcome)
        ends.append((applied.table_cards(0), applied.deck_remaining, applied.phase))
    assert ends[0] == ends[1] == ends[2]


def test_an_outcome_from_another_boundary_is_refused():
    """Rather than silently producing a state no deal could reach."""
    state = _afterstate()
    outcome = state.sample_boundary_outcome(random.Random(5))

    with pytest.raises(IllegalAction):
        state.copy().apply_boundary_outcome(
            BoundaryOutcome(draws=outcome.draws[:2])  # too few
        )
    with pytest.raises(IllegalAction):
        state.copy().apply_boundary_outcome(
            BoundaryOutcome(draws=outcome.draws + outcome.draws)  # too many
        )
    on_table = next(c for c in state.stack_old[0] if c is not None)
    with pytest.raises(IllegalAction):
        state.copy().apply_boundary_outcome(
            BoundaryOutcome(draws=(on_table,) + outcome.draws[1:])  # not in the deck
        )


def test_the_boundary_must_be_prepared_before_it_is_sampled_or_applied():
    state = _mid_game()  # cards still on the table: not an afterstate
    with pytest.raises(IllegalAction):
        state.sample_boundary_outcome(random.Random(0))
    with pytest.raises(IllegalAction):
        state.apply_boundary_outcome(BoundaryOutcome(draws=(0, 1, 2)))


@pytest.mark.parametrize("players", [2, 3, 4])
def test_standard_play_never_exhausts_the_deck_mid_triple(players):
    """Why "mid-draw exhaustion" is not a standard-mode case.

    Consumption stays a multiple of three -- six at setup, three per ordinary
    turn, six per requested reshuffle -- so the deck is either empty before the
    first of three draws (the reform case above) or holds at least three.
    Measured here at every ``_draw_step``; the spec's figure is 56,205
    observations over 120 games with no exceptions.
    """
    seen = []
    original = GameState._draw_step

    def spy(self, draw=None):
        seen.append(self.deck_remaining)
        original(self, draw)

    GameState._draw_step = spy
    try:
        for seed in range(4):
            bots = [GreedyBot(random.Random(seed * 10 + i)) for i in range(players)]
            state = GameState.new(
                seed=seed, config=GameConfig(players=players, advanced=True)
            )
            while not state.is_terminal:
                state.apply(bots[state.actor].act(state))
    finally:
        GameState._draw_step = original

    assert seen, "no draw step ran"
    offenders = [n for n in seen if n % 3 != 0]
    assert not offenders, f"{len(offenders)} of {len(seen)} draw steps mid-triple"


def test_a_mid_draw_reform_still_works_if_expert_mode_ever_needs_it():
    """Generic, and *not* search-critical -- kept because ``_draw`` supports it
    and expert mode is still in the engine.  Standard mode cannot reach it (the
    test above), so it is driven directly."""
    state = _afterstate()
    keep = state.deck[state.deck_pos]
    state.discard.extend(state.deck[state.deck_pos + 1 :])
    state.deck = state.deck[: state.deck_pos] + [keep]
    assert state.deck_remaining == 1 and state.discard

    before = _live_cards(state)
    outcome = state.sample_boundary_outcome(random.Random(0))
    assert len(outcome.draws) == 3, "the draw did not survive a mid-triple reform"
    state.apply_boundary_outcome(outcome)
    assert _live_cards(state) == before
