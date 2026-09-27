"""The two pieces of the anchored BGA replay that can silently go wrong."""

from __future__ import annotations

import collections
import random

from .bga_replay import Move, _action_for, _force_card
from .codec import decode_action, legal_action_indices
from .engine import ActionUse, apply_action
from .game import Phase, new_game


def _play_into_age(seed: int, moves: int):
    game = new_game(seed, 0)
    rng = random.Random(seed)
    while game.phase is Phase.WONDER_DRAFT or moves > 0:
        if game.phase is not Phase.WONDER_DRAFT:
            moves -= 1
        apply_action(game, decode_action(game, rng.choice(legal_action_indices(game))))
    return game


def _hidden_multiset(game):
    names = [c.card_name for c in game.tableau.cards.values() if c.present and not c.revealed]
    names += list(game.removed_age_cards.get(game.age, ()))
    names += list(game.unused_guilds)
    return collections.Counter(names)


def test_forcing_a_hidden_card_is_a_swap_that_keeps_the_deal_valid():
    for seed in range(12):
        game = _play_into_age(seed, 3)
        hidden = [s for s, c in game.tableau.cards.items() if c.present and not c.revealed]
        before = _hidden_multiset(game)
        slot = hidden[0]
        # A card hidden elsewhere, or set aside for this Age.
        other = next(
            (game.tableau.cards[s].card_name for s in hidden[1:]
             if game.tableau.cards[s].card_name != game.tableau.cards[slot].card_name),
            None,
        ) or game.removed_age_cards[game.age][0]
        assert _force_card(game, slot, other)
        assert game.tableau.cards[slot].card_name == other
        assert _hidden_multiset(game) == before


def test_forcing_a_card_that_is_not_hidden_anywhere_is_refused():
    game = _play_into_age(3, 3)
    slot = next(s for s, c in game.tableau.cards.items() if c.present and not c.revealed)
    before = game.tableau.cards[slot].card_name
    assert not _force_card(game, slot, "The Pyramids")  # a wonder, never a card
    assert game.tableau.cards[slot].card_name == before


def test_packet_names_map_to_the_engine_action_including_bga_capitalisation():
    game = _play_into_age(5, 0)
    for index in legal_action_indices(game):
        action = decode_action(game, index)
        if action.use is not ActionUse.CONSTRUCT_BUILDING:
            continue
        name = game.tableau.cards[tuple(action.slot_id)].card_name
        bga = " ".join(w.capitalize() for w in name.split())  # "Chamber Of Commerce"
        move = Move("p", "constructBuilding", {"buildingName": bga, "wonderName": "None"}, 1)
        assert _action_for(game, move) == index
        sold = Move("p", "discardBuilding", {"buildingName": name}, 1)
        assert decode_action(game, _action_for(game, sold)).use is ActionUse.DISCARD_FOR_COINS
        return
    raise AssertionError("no buildable card in the opening position")
