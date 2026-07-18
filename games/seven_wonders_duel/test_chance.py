"""Chance-event contract tests: CODEC_SPEC.md §4 and §7 engine additions."""

import random

import pytest

from games.seven_wonders_duel.data import (
    AGE_II_CARDS,
    BackType,
    PROGRESS_IDS,
    back_type_of,
)
from games.seven_wonders_duel.engine import (
    Action,
    ActionUse,
    apply_action,
    legal_actions,
)
from games.seven_wonders_duel.game import (
    ChanceKind,
    HiddenInformationError,
    PendingChoiceKind,
    Phase,
    new_game,
)
from games.seven_wonders_duel.pool import (
    enumerate_card_reveal,
    enumerate_great_library,
    enumerate_wonder_flip,
    resample_hidden,
    unseen_pool,
)


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


def _advance_until(game, predicate, limit=400):
    for _ in range(limit):
        if predicate(game):
            return
        actions = legal_actions(game)
        if not actions:
            raise AssertionError("no legal actions before predicate held")
        apply_action(game, actions[0])
    raise AssertionError("predicate did not hold within move limit")


# --- §7.1 back type ---------------------------------------------------------


def test_observation_exposes_back_type_without_identity():
    game = _playing_game(30)
    observation = game.observation(0)
    for public in observation.tableau:
        assert public.back is BackType.AGE_I
    hidden = [c for c in observation.tableau if c.present and not c.revealed]
    assert hidden and all(c.card_name is None for c in hidden)


def test_age_three_guild_backs_are_public_and_match_hidden_identity():
    game = _playing_game(5)
    _advance_until(game, lambda g: g.age == 3 or g.phase is Phase.COMPLETE)
    if game.phase is Phase.COMPLETE:
        pytest.skip("seed ended before Age III")
    observation = game.observation(1)
    guild_backed = [c for c in observation.tableau if c.back is BackType.GUILD]
    assert len(guild_backed) == 3
    for public in guild_backed:
        true_card = game.tableau.cards[public.slot_id]
        assert back_type_of(true_card.card_name) is BackType.GUILD


# --- §7.2 burial mapping ----------------------------------------------------


def test_wonder_burial_mapping_is_recorded_and_observable():
    game = _playing_game(400)
    _give_wonder(game, 0, "The Sphinx")
    game.cities[0].coins = 100
    slot = game.tableau.accessible_slot_ids()[0]
    buried = game.tableau.cards[slot].card_name
    apply_action(game, Action(slot, ActionUse.CONSTRUCT_WONDER, "The Sphinx"))
    assert game.wonder_burials["The Sphinx"] == buried
    assert ("The Sphinx", buried) in game.observation(1).wonder_burials


# --- §7.3/7.4 chance events, outcomes, barrier ------------------------------


def test_uncovering_emits_card_reveal_events_in_canonical_order():
    game = _playing_game(30)
    first = apply_action(game, Action((4, 1), ActionUse.DISCARD_FOR_COINS))
    assert first.events == ()
    second = apply_action(game, Action((4, 3), ActionUse.DISCARD_FOR_COINS))
    kinds = [event.kind for event in second.events]
    assert kinds == [ChanceKind.CARD_REVEAL]
    event = second.events[0]
    assert event.context == ((3, 2), BackType.AGE_I)
    assert event.outcome == game.tableau.cards[(3, 2)].card_name


def test_search_barrier_blocks_unresolved_reveal():
    game = _playing_game(30)
    apply_action(game, Action((4, 1), ActionUse.DISCARD_FOR_COINS))
    clone = game.clone()
    clone.search_barrier = True
    with pytest.raises(HiddenInformationError):
        apply_action(clone, Action((4, 3), ActionUse.DISCARD_FOR_COINS))


def test_supplied_reveal_outcome_overrides_and_swaps_consistently():
    game = _playing_game(30)
    apply_action(game, Action((4, 1), ActionUse.DISCARD_FOR_COINS))
    clone = game.clone()
    clone.search_barrier = True
    pool = unseen_pool(clone.observation(0))
    true_card = clone.tableau.cards[(3, 2)].card_name
    replacement = next(
        name for name, _ in enumerate_card_reveal(pool, BackType.AGE_I)
        if name != true_card
    )
    result = apply_action(
        clone,
        Action((4, 3), ActionUse.DISCARD_FOR_COINS),
        chance_outcomes=[replacement],
    )
    assert result.events[0].outcome == replacement
    assert clone.tableau.cards[(3, 2)].card_name == replacement
    # The swap keeps the world consistent: no card duplicated or lost.
    everywhere = (
        [c.card_name for c in clone.tableau.cards.values() if c.present]
        + list(clone.removed_age_cards[1])
        + clone.discard_pile
    )
    assert len(everywhere) == len(set(everywhere))
    assert true_card in everywhere


def test_reveal_outcome_must_come_from_the_unseen_pool():
    game = _playing_game(30)
    apply_action(game, Action((4, 1), ActionUse.DISCARD_FOR_COINS))
    visible = next(
        c.card_name for c in game.tableau.cards.values() if c.present and c.revealed
    )
    with pytest.raises(ValueError):
        apply_action(
            game,
            Action((4, 3), ActionUse.DISCARD_FOR_COINS),
            chance_outcomes=[visible],
        )


def test_great_library_draw_is_an_event_with_canonical_sorted_options():
    game = _playing_game(400)
    _give_wonder(game, 0, "The Great Library")
    game.cities[0].coins = 100
    slot = game.tableau.accessible_slot_ids()[0]
    result = apply_action(
        game, Action(slot, ActionUse.CONSTRUCT_WONDER, "The Great Library")
    )
    draws = [e for e in result.events if e.kind is ChanceKind.GREAT_LIBRARY_DRAW]
    assert len(draws) == 1
    options = draws[0].outcome
    assert len(options) == 3
    assert set(options) <= set(game.unused_progress_tokens)
    assert list(options) == sorted(options, key=PROGRESS_IDS.__getitem__)
    assert game.pending_choice is not None
    assert game.pending_choice.kind is PendingChoiceKind.CHOOSE_UNUSED_PROGRESS
    assert game.pending_choice.options == options


def test_great_library_respects_barrier_and_supplied_outcome():
    game = _playing_game(400)
    _give_wonder(game, 0, "The Great Library")
    game.cities[0].coins = 100
    slot = game.tableau.accessible_slot_ids()[0]

    barred = game.clone()
    barred.search_barrier = True
    with pytest.raises(HiddenInformationError):
        apply_action(barred, Action(slot, ActionUse.CONSTRUCT_WONDER, "The Great Library"))

    chosen = game.clone()
    subset = tuple(sorted(chosen.unused_progress_tokens)[:3])
    result = apply_action(
        chosen,
        Action(slot, ActionUse.CONSTRUCT_WONDER, "The Great Library"),
        chance_outcomes=[subset],
    )
    draws = [e for e in result.events if e.kind is ChanceKind.GREAT_LIBRARY_DRAW]
    assert set(draws[0].outcome) == set(subset)


def test_fourth_draft_pick_emits_wonder_group_reveal():
    game = new_game(9)
    result = None
    for _ in range(4):
        result = apply_action(game, legal_actions(game)[0])
    assert result is not None
    flips = [e for e in result.events if e.kind is ChanceKind.WONDER_GROUP_REVEAL]
    assert len(flips) == 1
    assert set(flips[0].outcome) == set(game.wonder_offer)


def test_wonder_flip_barrier_and_override():
    game = new_game(9)
    for _ in range(3):
        apply_action(game, legal_actions(game)[0])

    barred = game.clone()
    barred.search_barrier = True
    with pytest.raises(HiddenInformationError):
        apply_action(barred, legal_actions(barred)[0])

    chosen = game.clone()
    pool = unseen_pool(chosen.observation(0))
    outcome = next(iter(enumerate_wonder_flip(pool)))[0]
    apply_action(chosen, legal_actions(chosen)[0], chance_outcomes=[outcome])
    assert tuple(chosen.wonder_offer) == outcome
    drafted = {w for city in chosen.cities for w in city.wonders}
    assert not (set(outcome) & drafted)


def test_age_transition_emits_age_deal_and_respects_barrier():
    game = _playing_game(11)
    _advance_until(
        game,
        lambda g: g.phase in (Phase.CHOOSE_NEXT_START_PLAYER, Phase.COMPLETE),
    )
    if game.phase is Phase.COMPLETE:
        pytest.skip("seed ended during Age I")

    barred = game.clone()
    barred.search_barrier = True
    with pytest.raises(HiddenInformationError):
        apply_action(barred, legal_actions(barred)[0])

    supplied = game.clone()
    deal = tuple(card.name for card in AGE_II_CARDS[:20])
    result = apply_action(supplied, legal_actions(supplied)[0], chance_outcomes=[deal])
    deals = [e for e in result.events if e.kind is ChanceKind.AGE_DEAL]
    assert deals[0].context == (2,)
    assert deals[0].outcome == deal
    dealt_names = {c.card_name for c in supplied.tableau.cards.values()}
    assert dealt_names == set(deal)
    assert set(supplied.removed_age_cards[2]) == {
        card.name for card in AGE_II_CARDS[20:]
    }

    simulator = game.clone()
    result = apply_action(simulator, legal_actions(simulator)[0])
    deals = [e for e in result.events if e.kind is ChanceKind.AGE_DEAL]
    assert len(deals) == 1 and len(deals[0].outcome) == 20


# --- §7.5 UnseenPool + resample_hidden --------------------------------------


def test_unseen_pool_sizes_at_age_one_start():
    game = _playing_game(30)
    pool = unseen_pool(game.observation(0))
    assert len(pool.cards[BackType.AGE_I]) == 11  # 8 face-down + 3 removed
    assert len(pool.cards[BackType.AGE_II]) == 23
    assert len(pool.cards[BackType.AGE_III]) == 20
    assert len(pool.cards[BackType.GUILD]) == 7
    assert set(game.removed_age_cards[1]) <= pool.cards[BackType.AGE_I]
    assert len(pool.offboard_progress) == 5
    reveal = enumerate_card_reveal(pool, BackType.AGE_I)
    assert len(reveal) == 11 and abs(sum(p for _, p in reveal) - 1.0) < 1e-12
    library = enumerate_great_library(pool)
    assert len(library) == 10


def test_unseen_wonder_pool_during_draft_round_zero():
    game = new_game(9)
    apply_action(game, legal_actions(game)[0])
    pool = unseen_pool(game.observation(0))
    assert len(pool.wonders) == 8
    assert len(enumerate_wonder_flip(pool)) == 70


def test_resample_hidden_preserves_the_visible_projection():
    game = _playing_game(30)
    apply_action(game, Action((4, 1), ActionUse.DISCARD_FOR_COINS))
    before = (game.observation(0), game.observation(1))
    changed = False
    for sample_seed in range(20):
        clone = game.clone()
        resample_hidden(clone, random.Random(sample_seed))
        assert (clone.observation(0), clone.observation(1)) == before
        hidden_before = tuple(
            card.card_name
            for card in game.tableau.cards.values()
            if card.present and not card.revealed
        )
        hidden_after = tuple(
            card.card_name
            for card in clone.tableau.cards.values()
            if card.present and not card.revealed
        )
        if hidden_before != hidden_after:
            changed = True
    assert changed


def test_resample_hidden_during_draft_and_full_playthrough():
    game = new_game(9)
    apply_action(game, legal_actions(game)[0])
    clone = game.clone()
    observation_before = clone.observation(0)
    resample_hidden(clone, random.Random(123))
    assert clone.observation(0) == observation_before
    # A resampled world must still be a playable, rules-consistent game.
    _advance_until(clone, lambda g: g.phase is Phase.COMPLETE, limit=600)
    assert clone.victory_type is not None


def test_resampled_worlds_cover_reveal_marginals():
    """Light marginal check (the full chi-squared gate is Phase A tier)."""

    game = _playing_game(30)
    apply_action(game, Action((4, 1), ActionUse.DISCARD_FOR_COINS))
    pool = unseen_pool(game.observation(0))
    expected = {name for name, _ in enumerate_card_reveal(pool, BackType.AGE_I)}
    seen: set[str] = set()
    for sample_seed in range(200):
        clone = game.clone()
        resample_hidden(clone, random.Random(sample_seed))
        seen.add(clone.tableau.cards[(3, 2)].card_name)
    assert seen == expected
