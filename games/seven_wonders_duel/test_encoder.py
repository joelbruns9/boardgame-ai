"""Encoder gates: purity, mirror symmetry, determinism, feature spot checks
(CODEC_SPEC.md §2, §5.1, §5.8)."""

import dataclasses
import hashlib
import json
import random

from games.seven_wonders_duel.data import BackType
from games.seven_wonders_duel.encoder import (
    ENCODER_SIGNATURE,
    GLOBAL_FEATURES,
    TABLEAU_FEATURES,
    Encoding,
    TokenType,
    encode,
)
from games.seven_wonders_duel.engine import Action, ActionUse, apply_action, legal_actions
from games.seven_wonders_duel.game import Phase, new_game
from games.seven_wonders_duel.pool import resample_hidden


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


def _mirrored(game):
    mirror = game.clone()
    mirror.cities = (mirror.cities[1], mirror.cities[0])
    mirror.conflict_position = -game.conflict_position
    mirror.military_tokens_remaining = {
        -position: penalty for position, penalty in game.military_tokens_remaining.items()
    }
    mirror.active_player = 1 - game.active_player
    mirror.first_player = 1 - game.first_player
    if mirror.pending_choice is not None:
        mirror.pending_choice = dataclasses.replace(
            mirror.pending_choice, player=1 - mirror.pending_choice.player
        )
    if mirror.winner is not None:
        mirror.winner = 1 - mirror.winner
    if mirror.final_scores is not None:
        mirror.final_scores = (mirror.final_scores[1], mirror.final_scores[0])
    return mirror


def _global_value(encoding: Encoding, name: str) -> float:
    token = encoding.tokens[0]
    assert token.type is TokenType.GLOBAL
    return token.features[GLOBAL_FEATURES.index(name)]


def _digest(encoding: Encoding) -> str:
    payload = json.dumps(
        [
            (t.type.value, t.entity_id, t.aux_id, t.features)
            for t in encoding.tokens
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# --- determinism + purity ---------------------------------------------------


def test_encoding_is_deterministic():
    game = _playing_game(30)
    observation = game.observation(0)
    assert encode(observation) == encode(observation)


def test_encoding_is_pure_under_hidden_reassignment():
    game = _playing_game(30)
    apply_action(game, Action((4, 1), ActionUse.DISCARD_FOR_COINS))
    reference = encode(game.observation(0))
    for sample_seed in range(10):
        clone = game.clone()
        resample_hidden(clone, random.Random(sample_seed))
        assert encode(clone.observation(0)) == reference
        assert encode(clone.observation(1)) == reference  # viewer-independent


# --- mirror gate (spec §2) --------------------------------------------------


def test_mirror_states_encode_identically_across_trajectories():
    for seed in (3, 8):
        game = new_game(seed, first_player=seed % 2)
        rng = random.Random(seed)
        move = 0
        while game.phase is not Phase.COMPLETE:
            if move % 8 == 0:
                # The net-facing tokens must match exactly; Encoding.actor is
                # absolute-seat bookkeeping and flips by construction.
                assert (
                    encode(game.observation(0)).tokens
                    == encode(_mirrored(game).observation(0)).tokens
                )
            apply_action(game, rng.choice(legal_actions(game)))
            move += 1


def test_mirror_holds_with_military_asymmetry_and_pending():
    game = _playing_game(400)
    game.conflict_position = 5
    del game.military_tokens_remaining[4]
    _give_wonder(game, 0, "The Great Library")
    game.cities[0].coins = 100
    slot = game.tableau.accessible_slot_ids()[0]
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_WONDER, "The Great Library"))
    assert game.pending_choice is not None
    assert (
        encode(game.observation(0)).tokens
        == encode(_mirrored(game).observation(0)).tokens
    )


# --- feature spot checks ----------------------------------------------------


def test_initial_economy_features():
    game = _playing_game(30)
    encoding = encode(game.observation(0))
    assert _global_value(encoding, "my_coins") == 7
    assert _global_value(encoding, "opp_coins") == 7
    for resource in ("wood", "clay", "stone", "glass", "papyrus"):
        assert _global_value(encoding, f"my_trade_price_{resource}") == 2
    assert _global_value(encoding, "my_discard_income") == 2
    assert _global_value(encoding, "my_next_token_dist") == 4
    assert _global_value(encoding, "my_next_token_penalty") == 2
    assert _global_value(encoding, "opp_next_token_dist") == 4
    assert _global_value(encoding, "my_sci_win_feasible") == 1
    assert _global_value(encoding, "my_mil_win_feasible") == 1
    assert _global_value(encoding, "decision_main_turn") == 1


def test_opponent_production_raises_my_trade_price():
    game = _playing_game(30)
    game.cities[1 - game.active_player].buildings = ["Sawmill"]  # two wood
    encoding = encode(game.observation(0))
    assert _global_value(encoding, "my_trade_price_wood") == 4
    assert _global_value(encoding, "opp_trade_price_wood") == 2


def test_chain_free_build_is_visible_on_the_tableau_token():
    game = _playing_game(30)
    actor = game.active_player
    game.cities[actor].buildings = ["Baths"]  # chain to Aqueduct
    slot = _put_in_accessible_slot(game, "Aqueduct")
    encoding = encode(game.observation(0))
    index = TABLEAU_FEATURES.index("my_chain_free")
    cost_index = TABLEAU_FEATURES.index("my_cost")
    aqueduct = next(
        t
        for t in encoding.tokens
        if t.type is TokenType.TABLEAU and t.entity_id == 43  # Aqueduct card id
    )
    assert aqueduct.features[index] == 1.0
    assert aqueduct.features[cost_index] == 0.0


def test_face_down_tableau_tokens_carry_back_type_and_no_card_features():
    game = _playing_game(30)
    encoding = encode(game.observation(0))
    face_down = [
        t for t in encoding.tokens if t.type is TokenType.TABLEAU and t.entity_id >= 73
    ]
    assert len(face_down) == 8
    assert all(t.entity_id == 73 + 0 for t in face_down)  # AGE_I back id 0
    cost_index = TABLEAU_FEATURES.index("my_cost")
    assert all(t.features[cost_index] == 0.0 for t in face_down)


def test_library_candidates_are_flagged_progress_tokens():
    game = _playing_game(400)
    _give_wonder(game, 0, "The Great Library")
    game.cities[0].coins = 100
    slot = game.tableau.accessible_slot_ids()[0]
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_WONDER, "The Great Library"))
    encoding = encode(game.observation(1))
    candidates = [
        t
        for t in encoding.tokens
        if t.type is TokenType.PROGRESS and t.features[3] == 1.0
    ]
    assert len(candidates) == 3
    assert _global_value(encoding, "decision_great_library_pick") == 1


def test_pool_tokens_cover_future_ages_with_cost_aggregates():
    game = _playing_game(30)
    encoding = encode(game.observation(0))
    pools = {t.entity_id: t for t in encoding.tokens if t.type is TokenType.POOL}
    assert set(pools) == {0, 1, 2, 3}  # all four backs present at Age I start
    age2 = pools[1]
    assert age2.features[0] == 23  # full Age II pool
    my_mean = age2.features[2]
    opp_mean = age2.features[4]
    assert my_mean > 0 and my_mean == opp_mean  # symmetric start economy


def test_draft_phase_has_offer_and_wonder_pool_tokens():
    game = new_game(9)
    apply_action(game, legal_actions(game)[0])
    encoding = encode(game.observation(0))
    offers = [t for t in encoding.tokens if t.type is TokenType.DRAFT_OFFER]
    assert len(offers) == 3
    wonder_pool = [t for t in encoding.tokens if t.type is TokenType.POOL_WONDER]
    assert len(wonder_pool) == 1
    assert wonder_pool[0].features[0] == 8
    assert _global_value(encoding, "decision_wonder_draft") == 1


# --- signature + golden -----------------------------------------------------


def test_encoder_signature_is_pinned():
    # Bump ENCODER_VERSION and this pin together on any schema change (§5.8).
    assert (
        ENCODER_SIGNATURE
        == "76268a5f53f3ea0c3a18142ea3189df9b7f46e8e7d8117623720b989353054f5"
    )


def test_golden_encoding_digest_is_stable():
    game = _playing_game(30)
    assert (
        _digest(encode(game.observation(0)))
        == "9d75df0d87a4d9725c2d4eb9e217b8bd6d2ac86d4cd4553535baac203ed4be60"
    )
