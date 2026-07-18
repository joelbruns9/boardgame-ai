"""Encoder gates: purity, mirror symmetry, determinism, feature spot checks
(CODEC_SPEC.md §2, §5.1, §5.8)."""

import dataclasses
import hashlib
import json
import os
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
from games.seven_wonders_duel.codec import legal_action_indices
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
    # CI default: 2 games sampled every 8th move. The full spec §2 gate (≥1k
    # states) is the same test at SWD_MIRROR_GAMES=16 (checks every move).
    n_games = int(os.environ.get("SWD_MIRROR_GAMES", "2"))
    step = 8 if n_games <= 2 else 1
    for seed in range(3, 3 + n_games):
        game = new_game(seed, first_player=seed % 2)
        rng = random.Random(seed)
        move = 0
        while game.phase is not Phase.COMPLETE:
            if move % step == 0:
                # The net-facing tokens must match exactly; Encoding.actor is
                # absolute-seat bookkeeping and flips by construction.
                mirror = _mirrored(game)
                assert (
                    encode(game.observation(0)).tokens
                    == encode(mirror.observation(0)).tokens
                )
                # Identity-indexed masks are seat-free, so the mirrored state
                # must expose the exact same legal index set (spec §2 gate).
                assert legal_action_indices(game) == legal_action_indices(mirror)
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


def _force_into_unused(game, token_names):
    """Ensure the given progress tokens are off-board (in unused), swapping
    with board tokens as needed while keeping the 5/5 split consistent."""

    available = list(game.available_progress_tokens)
    unused = list(game.unused_progress_tokens)
    for name in token_names:
        if name in available:
            swap = next(t for t in unused if t not in token_names)
            available[available.index(name)] = swap
            unused[unused.index(swap)] = name
    game.available_progress_tokens = tuple(available)
    game.unused_progress_tokens = tuple(unused)


def test_great_library_candidates_count_toward_race_clocks():
    base = _playing_game(400)
    _give_wonder(base, 0, "The Great Library")
    base.cities[0].coins = 100
    _force_into_unused(base, ["Law", "Strategy"])
    others = [
        t for t in base.unused_progress_tokens if t not in ("Law", "Strategy")
    ]
    slot = base.tableau.accessible_slot_ids()[0]

    def build_with(drawn):
        clone = base.clone()
        apply_action(
            clone,
            Action(slot, ActionUse.CONSTRUCT_WONDER, "The Great Library"),
            chance_outcomes=[tuple(drawn)],
        )
        return encode(clone.observation(0))

    without = build_with(others[:3])
    with_law = build_with(["Law", *others[:2]])
    with_strategy = build_with(["Strategy", *others[:2]])

    # Law among the drawn candidates is one pick away: it must count as an
    # obtainable missing symbol for the pending player.
    assert (
        _global_value(with_law, "my_sci_missing_obtainable")
        == _global_value(without, "my_sci_missing_obtainable") + 1
    )
    # Strategy among the candidates unlocks the +1/red headroom in the bound.
    assert _global_value(with_strategy, "my_mil_shields_obtainable") > _global_value(
        without, "my_mil_shields_obtainable"
    )


def test_new_global_fields_token_flags_and_guild_score():
    game = _playing_game(30)
    encoding = encode(game.observation(0))
    assert _global_value(encoding, "my_token_2coin_remaining") == 1
    assert _global_value(encoding, "my_token_5coin_remaining") == 1
    assert _global_value(encoding, "opp_token_2coin_remaining") == 1
    assert _global_value(encoding, "my_score_guild") == 0
    actor = game.active_player
    del game.military_tokens_remaining[4 if actor == 0 else -4]
    encoding = encode(game.observation(0))
    assert _global_value(encoding, "my_token_2coin_remaining") == 0
    assert _global_value(encoding, "my_next_token_dist") == 7
    assert _global_value(encoding, "my_next_token_penalty") == 5


# --- signature + golden -----------------------------------------------------


def test_encoder_signature_is_pinned():
    # Bump ENCODER_VERSION and this pin together on any schema change (§5.8).
    assert (
        ENCODER_SIGNATURE
        == "7d68ff20f280700f0c7a04d2411cded734c51b3e312a80578824d7dbb0098be2"
    )


def test_golden_encoding_digest_is_stable():
    game = _playing_game(30)
    assert (
        _digest(encode(game.observation(0)))
        == "3dc18d90782fac2a906eecaa3bf6d8118beee76909f8b1a13b1111ac24772e60"
    )


def test_golden_digests_cover_draft_and_pending_states():
    draft = new_game(9)
    apply_action(draft, legal_actions(draft)[0])
    assert (
        _digest(encode(draft.observation(0)))
        == "12c235e52cc19f80430a55ede7f4ec9c3b4a0922d7a031955ad424442bd190c1"
    )

    library = _playing_game(400)
    _give_wonder(library, 0, "The Great Library")
    library.cities[0].coins = 100
    slot = library.tableau.accessible_slot_ids()[0]
    apply_action(library, Action(slot, ActionUse.CONSTRUCT_WONDER, "The Great Library"))
    assert library.pending_choice is not None
    assert (
        _digest(encode(library.observation(0)))
        == "13d9638d8f1720232eba47c8d9e8987a15fe5f2de45b87c3fb9ed65c50ba4e99"
    )
