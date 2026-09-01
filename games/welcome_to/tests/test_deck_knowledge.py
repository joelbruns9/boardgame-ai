"""Exact card counting: the deck composition and what each stack reveals next."""
from __future__ import annotations

import random

import numpy as np
import pytest

from games.welcome_to import deck_knowledge as dk
from games.welcome_to.constants import (
    CARD_TABLE,
    DECK_COUNTS,
    DECK_EFFECT_ORDER,
    EFFECT_INDEX,
    NUMBER_INDEX,
    NUM_BASE_CARDS,
    Effect,
)
from games.welcome_to.game import GameConfig, GameState

#: Numbers whose printed cards never carry POOL, TEMP or BIS.
NO_WET_EFFECTS = (1, 2, 5, 11, 14, 15)


def _card_with(number=None, effect=None) -> int:
    for i, (n, e) in enumerate(CARD_TABLE[:NUM_BASE_CARDS]):
        if (number is None or n == number) and (effect is None or e is effect):
            return i
    raise AssertionError(f"no card number={number} effect={effect}")


def _game(players: int = 2, **kwargs) -> GameState:
    return GameState.new(seed=4, config=GameConfig(players=players, **kwargs))


# ──────────────────────────────────────────────────────────────────────────
# The printed deck
# ──────────────────────────────────────────────────────────────────────────
def test_the_matrix_is_the_printed_deck():
    assert dk.DECK_MATRIX.shape == (15, 6)
    assert dk.DECK_MATRIX.sum() == NUM_BASE_CARDS == 81
    for number, counts in DECK_COUNTS.items():
        for effect, n in zip(DECK_EFFECT_ORDER, counts):
            assert dk.DECK_MATRIX[NUMBER_INDEX[number], EFFECT_INDEX[effect]] == n


def test_the_numbers_that_carry_no_wet_effects():
    """The number/effect correlation, which the joint histogram preserves."""
    for number in NO_WET_EFFECTS:
        row = dk.DECK_MATRIX[NUMBER_INDEX[number]]
        for effect in (Effect.POOL, Effect.TEMP, Effect.BIS):
            assert row[EFFECT_INDEX[effect]] == 0
    for number in set(DECK_COUNTS) - set(NO_WET_EFFECTS):
        row = dk.DECK_MATRIX[NUMBER_INDEX[number]]
        assert row[EFFECT_INDEX[Effect.POOL]] > 0


def test_three_and_thirteen_carry_no_dry_effects():
    for number in (3, 13):
        row = dk.DECK_MATRIX[NUMBER_INDEX[number]]
        assert row[EFFECT_INDEX[Effect.PARK]] == 0
        assert row[EFFECT_INDEX[Effect.ESTATE]] == 0


# ──────────────────────────────────────────────────────────────────────────
# Deck composition is exact, not estimated
# ──────────────────────────────────────────────────────────────────────────
def test_every_card_on_the_table_is_ruled_out():
    state = _game()
    assert state.discard == []
    # six cards on the table in standard mode, all fully identified
    assert len([c for c in state.table_cards(0) if c is not None]) == 6
    assert dk.deck_composition(state, 0).sum() == NUM_BASE_CARDS - 6


def test_the_composition_tracks_the_real_deck_all_game():
    state = _game()
    rng = random.Random(1)
    for _ in range(80):
        if state.is_terminal:
            break
        deck = dk.deck_composition(state, 0)
        assert deck.sum() == pytest.approx(state.deck_remaining, abs=1e-3)
        assert (deck >= 0).all()
        state.apply(rng.choice(state.legal_actions()))


def test_composition_plus_discard_plus_table_is_the_whole_deck():
    state = _game()
    rng = random.Random(5)
    for _ in range(60):
        if state.is_terminal:
            break
        total = (
            dk.deck_composition(state, 0).sum()
            + dk.discard_composition(state, 0).sum()
            + len([c for c in state.table_cards(0) if c is not None])
        )
        assert total == NUM_BASE_CARDS
        state.apply(rng.choice(state.legal_actions()))


def test_solo_deck_carries_one_card_the_matrix_does_not_know_about():
    state = _game(players=1)
    deck = dk.deck_composition(state, 0)
    # the solo marker is in the deck but is not a printed construction card,
    # and solo shows one card per stack rather than two
    assert deck.sum() == pytest.approx(state.deck_remaining - 1, abs=1e-3)


def test_a_reshuffle_puts_the_discard_back():
    state = _game()
    rng = random.Random(2)
    for _ in range(40):
        if state.is_terminal:
            break
        state.apply(rng.choice(state.legal_actions()))
    assert state.discard, "expected cards to have been discarded"

    before = dk.deck_composition(state, 0)
    counterfactual = dk.after_reshuffle_composition(state, 0)
    assert counterfactual.sum() > before.sum()
    assert np.array_equal(
        counterfactual,
        before
        + dk.discard_composition(state, 0)
        + dk.aside_composition(state, 0),
    )

    # The reshuffle resolves at the NEXT turn boundary, and _begin_turn runs
    # _discard_step() BEFORE _reshuffle_decks().  Reproduce that order -- calling
    # _reform_deck() on its own reproduces the bug this test used to encode,
    # leaving the three aside cards out of the pool.
    state._discard_step()
    state._reform_deck()
    assert np.array_equal(dk.deck_composition(state, 0), counterfactual)


# ──────────────────────────────────────────────────────────────────────────
# Next turn is partly certain
# ──────────────────────────────────────────────────────────────────────────
def test_next_turns_effect_is_known_not_guessed():
    """The number face prints its own effect, so there is nothing to infer."""
    state = _game()
    rows = dk.known_next_effects(state, 0)
    assert rows.shape == (3, 6)
    assert np.array_equal(rows.sum(axis=1), np.ones(3)), "one-hot per stack"

    for i, effect in enumerate(state.next_effects(0)):
        assert effect is not None
        assert rows[i, EFFECT_INDEX[effect]] == 1.0
        # and it is the effect of the card on top of the stack
        assert CARD_TABLE[state.stack_new[0][i]][1] is effect


def test_this_turns_effect_becomes_next_turns_from_the_same_card():
    """The card showing a number now supplies the effect after the flip."""
    state = _game()
    promised = state.next_effects(0)
    while state.turn == 1 and not state.is_terminal:
        state.apply(state.legal_actions()[0])
    assert [e for _, e in state.visible_cards(0)] == promised


def test_the_next_number_is_a_distribution_over_what_is_left():
    state = _game()
    dist = dk.next_number_distribution(state, 0)
    assert dist.shape == (15,)
    assert dist.sum() == pytest.approx(1.0)
    # 8 and 9 are the most common numbers in the printed deck (nine copies each)
    assert dist[NUMBER_INDEX[8]] > dist[NUMBER_INDEX[1]]


def test_expert_and_solo_promise_nothing_about_next_turn():
    for state in (_game(players=1), _game(players=3, expert=True)):
        assert state.next_effects(0) == [None, None, None]
        assert dk.known_next_effects(state, 0).sum() == 0.0


# ──────────────────────────────────────────────────────────────────────────
# Information-set safety
# ──────────────────────────────────────────────────────────────────────────
def test_the_deck_order_is_never_read():
    state = _game()
    before = dk.deck_composition(state, 0).copy()
    alt = state.copy()
    alt.deck[alt.deck_pos:] = list(reversed(alt.deck[alt.deck_pos:]))
    assert np.array_equal(dk.deck_composition(alt, 0), before)


def test_expert_mode_never_counts_the_shared_discard():
    """``getAllDatas`` sends each expert client only its own cards."""
    state = GameState.new(seed=8, config=GameConfig(players=3, expert=True))
    rng = random.Random(3)
    for _ in range(40):
        if state.is_terminal:
            break
        state.apply(rng.choice(state.legal_actions()))
    assert state.discard, "expected discards to have accumulated"
    # only the player's own three cards are ruled out, never anyone else's
    assert dk.deck_composition(state, 0).sum() == NUM_BASE_CARDS - 3
    assert dk.discard_composition(state, 0).sum() == 0.0


def test_summarise_runs():
    assert "deck" in dk.summarise(_game(), 0)


# ──────────────────────────────────────────────────────────────────────────
# Regressions for the two defects found by external review, 2026-08-21
# ──────────────────────────────────────────────────────────────────────────
def test_after_reshuffle_matches_the_pool_the_engine_actually_reforms():
    """The counterfactual must equal the pool ``_reform_deck`` really sees.

    It used to be ``deck + discard``, which undercounts by the three aside
    cards: ``_begin_turn`` discards them *before* reforming.  Asserted against
    the engine rather than against a restatement of the formula.
    """
    state = _game()
    rng = random.Random(5)
    for _ in range(40):
        if state.is_terminal:
            break
        state.apply(rng.choice(state.legal_actions()))

    aside = [c for c in state.stack_old[0] if c is not None]
    assert aside, "expected aside cards to be on the table"
    expected_size = state.deck_remaining + len(state.discard) + len(aside)
    assert dk.after_reshuffle_composition(state, 0).sum() == expected_size

    counterfactual = dk.after_reshuffle_composition(state, 0)
    state._discard_step()
    state._reform_deck()
    assert np.array_equal(dk.deck_composition(state, 0), counterfactual)


def test_after_reshuffle_is_not_merely_deck_plus_discard():
    """Pin the specific regression: the old formula is three cards short."""
    state = _game()
    rng = random.Random(6)
    for _ in range(30):
        if state.is_terminal:
            break
        state.apply(rng.choice(state.legal_actions()))

    old_formula = dk.deck_composition(state, 0) + dk.discard_composition(state, 0)
    correct = dk.after_reshuffle_composition(state, 0)
    assert correct.sum() - old_formula.sum() == 3


# ──────────────────────────────────────────────────────────────────────────
# Encoder v3: prefix sums and exact supply rates (ENCODER_V3_SPEC.md §7.1, §9.3)
# ──────────────────────────────────────────────────────────────────────────
import itertools


def _fresh(seed: int = 3, **cfg):
    return GameState.new(seed=seed, config=GameConfig(players=2, advanced=True, **cfg))


def test_prefix_sums_are_cumulative_and_total_the_deck():
    state = _fresh()
    deck, reform, reshuffled = dk.number_prefix_sums(state, 0)
    for p in (deck, reform, reshuffled):
        assert p.shape == (16,)
        assert p[0] == 0
        assert all(p[i] <= p[i + 1] for i in range(15))
    assert deck[-1] == dk.deck_composition(state, 0).sum()
    assert reshuffled[-1] == deck[-1] + reform[-1]


def test_prefix_range_subtraction_matches_a_direct_count():
    """The subtraction the fit planes make, checked against counting by hand."""
    state = _fresh(seed=11)
    deck, _, _ = dk.number_prefix_sums(state, 0)
    composition = dk.deck_composition(state, 0).sum(axis=1)
    for low, high in ((7, 9), (-1, 18), (0, 4), (12, 16), (5, 6)):
        expected = sum(
            composition[NUMBER_INDEX[n]]
            for n in range(1, 16)
            if low < n < high
        )
        got = deck[min(high - 1, 15)] - deck[max(low, 0)]
        assert got == expected, (low, high)


def _brute_force_effect_rate(state, player, effect_index: int) -> float:
    """P(effect appears among three drawn) by enumerating the literal deck."""
    remaining = state.deck[state.deck_pos :]
    cards = [c for c in remaining if CARD_TABLE[c][1] in EFFECT_INDEX]
    hits = 0
    total = 0
    for combo in itertools.combinations(range(len(cards)), 3):
        total += 1
        if any(EFFECT_INDEX[CARD_TABLE[cards[i]][1]] == effect_index for i in combo):
            hits += 1
    return hits / total


def test_effect_supply_rate_is_exact_not_the_independent_approximation():
    """Three cards are drawn WITHOUT replacement; `1 - (1-p)**3` is not it."""
    exact = 1.0 - dk._p_none_in_next_three(9, 40, 0, 0)
    approximation = 1.0 - (1.0 - 9 / 40) ** 3
    # The spec's worked example: 0.545 exact against 0.535 approximate.
    assert abs(exact - 0.545) < 5e-4
    assert abs(approximation - 0.5345) < 5e-4
    assert exact > approximation


def test_effect_supply_rate_matches_brute_force_over_the_real_deck():
    state = _fresh(seed=5)
    rates = dk.effect_supply_rate(state, 0)
    for e in range(dk.NUM_EFFECTS):
        # `effect_supply_rate` returns float32; the arithmetic itself is f64.
        assert abs(float(rates[e]) - _brute_force_effect_rate(state, 0, e)) < 1e-6


def test_a_near_empty_deck_still_reveals_three_cards():
    """`_draw` reforms mid-draw, so `D < 3` is not a degenerate case."""
    # Two cards left, one of them a hit: drawing both makes the hit certain.
    assert dk._p_none_in_next_three(1, 2, 0, 30) == 0.0
    # An empty deck draws all three from the reform pool.
    assert dk._p_none_in_next_three(0, 0, 5, 30) == (25 * 24 * 23) / (30 * 29 * 28)
    # And the hits sitting in the reform pool are what decide it, not the deck.
    assert dk._p_none_in_next_three(0, 0, 0, 30) == 1.0


def test_reveals_to_reform_counts_the_reveal_that_finds_the_deck_empty():
    """`floor(D/3) + 1`: `_draw` reforms only when it FINDS the deck empty."""
    state = _fresh()
    for remaining, expected in ((0, 1), (3, 2), (4, 2), (6, 3), (7, 3)):
        state.deck_pos = len(state.deck) - remaining
        assert state.deck_remaining == remaining
        assert dk.reveals_to_reform(state) == expected
