"""Action-codec gates: round-trip, mask exactness, block layout (CODEC_SPEC.md §3.3).

The in-repo gate runs a reduced game count for CI speed; the full ≥10k-game
Phase A sweep uses the same checks via a longer seed range.
"""

import os
import random

import pytest

from games.seven_wonders_duel.codec import (
    BUILD_BASE,
    CARD_TO_WONDER_BASE,
    DESTROY_BASE,
    DISCARD_BASE,
    MAUSOLEUM_BASE,
    NEXT_AGE_BASE,
    NUM_ACTIONS,
    PROGRESS_BOARD_BASE,
    PROGRESS_LIBRARY_BASE,
    decode_action,
    encode_action,
    legal_action_indices,
    legal_action_mask,
)
from games.seven_wonders_duel.data import CARD_IDS, PROGRESS_IDS, WONDER_IDS
from games.seven_wonders_duel.engine import (
    Action,
    ActionUse,
    apply_action,
    legal_actions,
)
from games.seven_wonders_duel.game import PendingChoiceKind, Phase, new_game

# CI default 150 games (≥10k states); the full spec §3.3 gate is the same test
# at SWD_GATE_GAMES=10000.
N_GAMES = int(os.environ.get("SWD_GATE_GAMES", "150"))


def _playing_game(seed=1):
    game = new_game(seed)
    while game.phase is Phase.WONDER_DRAFT:
        game.pick_wonder(game.legal_wonder_choices()[0])
    return game


def _give_wonder(game, player, wonder_name):
    for city in game.cities:
        if wonder_name in city.wonders:
            city.wonders.remove(wonder_name)
    game.cities[player].wonders[0:0] = [wonder_name]


def _put_in_accessible_slot(game, card_name):
    target = game.tableau.accessible_slot_ids()[0]
    source = next(
        (slot_id for slot_id, card in game.tableau.cards.items() if card.card_name == card_name),
        None,
    )
    if source is None:
        game.tableau.cards[target].card_name = card_name
        return target
    game.tableau.cards[target].card_name, game.tableau.cards[source].card_name = (
        game.tableau.cards[source].card_name,
        game.tableau.cards[target].card_name,
    )
    return target


def _assert_state_round_trips(game, actions):
    indices = []
    for action in actions:
        index = encode_action(game, action)
        assert 0 <= index < NUM_ACTIONS
        assert decode_action(game, index) == action
        indices.append(index)
    assert len(set(indices)) == len(indices), "legal actions must map injectively"
    assert tuple(sorted(indices)) == legal_action_indices(game)
    mask = legal_action_mask(game)
    assert sum(mask) == len(indices)
    assert all(mask[i] for i in indices)


# --- block layout goldens ---------------------------------------------------


def test_block_layout_matches_spec_exactly():
    assert NUM_ACTIONS == 1202
    assert (BUILD_BASE, DISCARD_BASE, CARD_TO_WONDER_BASE) == (12, 85, 158)
    assert (DESTROY_BASE, MAUSOLEUM_BASE) == (1034, 1107)
    assert (PROGRESS_BOARD_BASE, PROGRESS_LIBRARY_BASE, NEXT_AGE_BASE) == (
        1180,
        1190,
        1200,
    )
    game = new_game(1)
    # Draft block: wonder id 0 is The Appian Way.
    if "The Appian Way" in game.wonder_offer:
        action = Action(None, ActionUse.DRAFT_WONDER, wonder_name="The Appian Way")
        assert encode_action(game, action) == 0
    assert WONDER_IDS["The Appian Way"] == 0
    assert CARD_IDS["Lumber Yard"] == 0
    assert CARD_IDS["Tacticians Guild"] == 72
    assert PROGRESS_IDS["Urbanism"] == 9
    # Card→wonder extremes: (card 0, wonder 0) and (card 72, wonder 11).
    assert CARD_TO_WONDER_BASE + 0 * 12 + 0 == 158
    assert CARD_TO_WONDER_BASE + 72 * 12 + 11 == 1033


def test_build_and_discard_encode_by_card_identity():
    game = _playing_game(30)
    slot = game.tableau.accessible_slot_ids()[0]
    card_id = CARD_IDS[game.tableau.cards[slot].card_name]
    assert (
        encode_action(game, Action(slot, ActionUse.DISCARD_FOR_COINS))
        == DISCARD_BASE + card_id
    )


# --- §3.3.1 + §3.3.2: round-trip and mask exactness over sampled games ------


def test_round_trip_and_mask_exactness_over_random_games():
    block_coverage = set()
    states_checked = 0
    for seed in range(N_GAMES):
        game = new_game(seed, first_player=seed % 2)
        rng = random.Random(seed * 7919 + 1)
        while game.phase is not Phase.COMPLETE:
            actions = legal_actions(game)
            assert actions, "non-complete state must have legal actions"
            _assert_state_round_trips(game, actions)
            states_checked += 1
            for action in actions:
                block_coverage.add(action.use)
            apply_action(game, rng.choice(actions))
    assert states_checked >= 10_000, states_checked
    assert {
        ActionUse.DRAFT_WONDER,
        ActionUse.CONSTRUCT_BUILDING,
        ActionUse.DISCARD_FOR_COINS,
        ActionUse.CONSTRUCT_WONDER,
        ActionUse.CHOOSE_NEXT_START_PLAYER,
    } <= block_coverage


# --- pending-choice blocks (crafted; random games hit these rarely) ---------


def test_destroy_block_round_trips_for_zeus():
    game = _playing_game(400)
    _give_wonder(game, 0, "The Statue of Zeus")
    game.cities[0].coins = 100
    game.cities[1].buildings = ["Lumber Yard", "Clay Pool"]
    slot = game.tableau.accessible_slot_ids()[0]
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_WONDER, "The Statue of Zeus"))
    assert game.pending_choice is not None
    assert game.pending_choice.kind is PendingChoiceKind.DESTROY_OPPONENT_BROWN
    _assert_state_round_trips(game, legal_actions(game))
    expected = {DESTROY_BASE + CARD_IDS["Lumber Yard"], DESTROY_BASE + CARD_IDS["Clay Pool"]}
    assert set(legal_action_indices(game)) == expected


def test_mausoleum_block_round_trips():
    game = _playing_game(400)
    _give_wonder(game, 1, "The Mausoleum")
    game.cities[1].coins = 100
    discard_slot = game.tableau.accessible_slot_ids()[0]
    discarded = game.tableau.cards[discard_slot].card_name
    apply_action(game, Action(discard_slot, ActionUse.DISCARD_FOR_COINS))
    assert game.active_player == 1
    build_slot = game.tableau.accessible_slot_ids()[0]
    apply_action(game, Action(build_slot, ActionUse.CONSTRUCT_WONDER, "The Mausoleum"))
    assert game.pending_choice is not None
    assert game.pending_choice.kind is PendingChoiceKind.BUILD_FROM_DISCARD_FREE
    _assert_state_round_trips(game, legal_actions(game))
    assert MAUSOLEUM_BASE + CARD_IDS[discarded] in legal_action_indices(game)


def test_progress_board_block_round_trips_for_science_pair():
    game = _playing_game(400)
    game.cities[0].coins = 100
    game.cities[0].buildings = ["Workshop"]  # set square
    slot = _put_in_accessible_slot(game, "Laboratory")  # second set square
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_BUILDING))
    assert game.pending_choice is not None
    assert game.pending_choice.kind is PendingChoiceKind.CHOOSE_AVAILABLE_PROGRESS
    _assert_state_round_trips(game, legal_actions(game))
    expected = {
        PROGRESS_BOARD_BASE + PROGRESS_IDS[name]
        for name in game.available_progress_tokens
    }
    assert set(legal_action_indices(game)) == expected


def test_progress_library_block_round_trips():
    game = _playing_game(400)
    _give_wonder(game, 0, "The Great Library")
    game.cities[0].coins = 100
    slot = game.tableau.accessible_slot_ids()[0]
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_WONDER, "The Great Library"))
    assert game.pending_choice is not None
    assert game.pending_choice.kind is PendingChoiceKind.CHOOSE_UNUSED_PROGRESS
    _assert_state_round_trips(game, legal_actions(game))
    expected = {
        PROGRESS_LIBRARY_BASE + PROGRESS_IDS[name]
        for name in game.pending_choice.options
    }
    assert set(legal_action_indices(game)) == expected


def test_pending_block_decode_requires_matching_pending_kind():
    game = _playing_game(400)
    _give_wonder(game, 0, "The Great Library")
    game.cities[0].coins = 100
    slot = game.tableau.accessible_slot_ids()[0]
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_WONDER, "The Great Library"))
    with pytest.raises(ValueError):
        decode_action(game, DESTROY_BASE)  # destroy block against a library pending


# --- §7.6 pinned test: NEXT_AGE_STARTER is actor-relative -------------------


def test_next_age_starter_is_actor_relative():
    game = _playing_game(11)
    rng = random.Random(99)
    while game.phase is Phase.PLAY_AGE:
        apply_action(game, rng.choice(legal_actions(game)))
    assert game.phase is Phase.CHOOSE_NEXT_START_PLAYER
    chooser = game.active_player
    self_first = Action(None, ActionUse.CHOOSE_NEXT_START_PLAYER, starting_player=chooser)
    opp_first = Action(
        None, ActionUse.CHOOSE_NEXT_START_PLAYER, starting_player=1 - chooser
    )
    assert encode_action(game, self_first) == NEXT_AGE_BASE
    assert encode_action(game, opp_first) == NEXT_AGE_BASE + 1
    assert decode_action(game, NEXT_AGE_BASE) == self_first
    assert decode_action(game, NEXT_AGE_BASE + 1) == opp_first
