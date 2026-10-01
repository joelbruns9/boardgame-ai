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
# Encoder v3 step 3 -- ENCODER_V3_SPEC.md §7.1, §7.5, §9.3
# ──────────────────────────────────────────────────────────────────────────
import itertools  # noqa: E402


def _played(players: int = 2, plies: int = 30, seed: int = 5, **kwargs) -> GameState:
    state = _game(players=players, **kwargs)
    rng = random.Random(seed)
    for _ in range(plies):
        if state.is_terminal:
            break
        state.apply(rng.choice(state.legal_actions()))
    assert not state.is_terminal
    return state


def _drain_to(state: GameState, construction_left: int) -> GameState:
    """Move undrawn cards into the discard until ``construction_left`` remain.

    A consistent state -- every card is still somewhere -- unlike advancing
    ``deck_pos``, which would delete cards from the bookkeeping.
    """
    state = state.copy()
    cards = state.deck[state.deck_pos:]
    state.discard.extend(cards[construction_left:])
    state.deck = state.deck[: state.deck_pos] + cards[:construction_left]
    return state


def test_prefix_sums_interval_counts_match_brute_force():
    state = _played()
    deck, pool, reshuffled = dk.number_prefix_sums(state, 0)
    per_number = [
        dk.deck_composition(state, 0).sum(axis=1),
        (dk.discard_composition(state, 0) + dk.aside_composition(state, 0)).sum(axis=1),
        dk.after_reshuffle_composition(state, 0).sum(axis=1),
    ]
    for prefix, counts in zip((deck, pool, reshuffled), per_number):
        assert prefix.shape == (16,) and prefix[0] == 0
        for low in range(-3, 21):
            for high in range(-3, 21):
                expected = sum(
                    int(counts[NUMBER_INDEX[n]]) for n in range(1, 16) if low < n < high
                )
                assert dk.count_in_open_interval(prefix, low, high) == expected
    assert deck[-1] == state.deck_remaining
    assert np.array_equal(reshuffled, deck + pool)


def _brute_ordered_draws(deck_labels, pool_labels, k, draws=3):
    out = np.zeros((k,) * draws)
    m = min(len(deck_labels), draws)
    r = draws - m
    first = list(itertools.permutations(range(len(deck_labels)), m))
    rest = list(itertools.permutations(range(len(pool_labels)), r))
    if not first or not rest:
        return out
    weight = 1.0 / (len(first) * len(rest))
    for a in first:
        for b in rest:
            idx = tuple(deck_labels[i] for i in a) + tuple(pool_labels[j] for j in b)
            out[idx] += weight
    return out


@pytest.mark.parametrize("deck_size", [0, 1, 2, 3, 4, 7])
def test_ordered_draws_match_enumeration_including_the_mid_draw_reform(deck_size):
    """§7.5 R4: D < 3 reforms mid-draw and still yields three cards -- never 0."""
    rng = random.Random(100 + deck_size)
    k = 4
    for _ in range(25):
        deck_labels = [rng.randrange(k) for _ in range(deck_size)]
        pool_labels = [rng.randrange(k) for _ in range(rng.randint(3, 6))]
        deck = np.bincount(deck_labels, minlength=k)
        pool = np.bincount(pool_labels, minlength=k)
        got = dk.ordered_draw_distribution(deck, pool)
        want = _brute_ordered_draws(deck_labels, pool_labels, k)
        assert np.allclose(got, want, atol=1e-12)
        assert got.sum() == pytest.approx(1.0, abs=1e-12)


def test_supply_rate_is_the_hypergeometric_form_when_the_deck_suffices():
    state = _played()
    counts = dk.deck_composition(state, 0).sum(axis=0).astype(np.float64)
    total = counts.sum()
    assert total >= 3
    rate = dk.effect_supply_rate(state, 0)
    for e in range(dk.NUM_EFFECTS):
        rest = total - counts[e]
        want = 1 - rest * (rest - 1) * (rest - 2) / (total * (total - 1) * (total - 2))
        assert rate[e] == pytest.approx(want, abs=1e-12)


def _engine_turn_plus_two_rate(state: GameState, samples: int, seed: int) -> np.ndarray:
    """Oracle: run the engine's real boundary and read the cards it draws last.

    The final three construction cards drawn are the ones promoted to aside on
    the following boundary, i.e. turn+2's effects -- on an ordinary boundary, a
    mid-draw reform and a queued reshuffle alike.
    """
    after = state.copy()
    assert after.prepare_turn_boundary()
    rng = random.Random(seed)
    hits = np.zeros(dk.NUM_EFFECTS)
    for _ in range(samples):
        draws = after.sample_boundary_outcome(rng).draws
        for e in {EFFECT_INDEX[CARD_TABLE[c][1]] for c in draws[-3:]}:
            hits[e] += 1
    return hits / samples


def _assert_close_to_engine(state: GameState, seed: int) -> None:
    rate = dk.effect_supply_rate(state, state.actor)
    engine = _engine_turn_plus_two_rate(state, samples=6000, seed=seed)
    # binomial sd <= 0.0065 at 6000 samples; 0.03 is > 4.5 sd
    assert np.abs(rate - engine).max() < 0.03, (rate, engine)


@pytest.mark.parametrize("left", [0, 1, 2, 5])
def test_supply_rate_matches_the_engine_across_the_reform(left):
    _assert_close_to_engine(_drain_to(_played(), left), seed=left)


def test_the_viewers_own_reshuffle_vote_switches_to_the_reformed_pool():
    state = _drain_to(_played(), 1)
    before = dk.effect_supply_rate(state, state.actor)
    state.reshuffle_votes[state.actor] = True
    state.reshuffle_next_turn = True
    after = dk.effect_supply_rate(state, state.actor)
    assert not np.allclose(before, after)
    _assert_close_to_engine(state, seed=99)


def test_another_seats_vote_is_not_read():
    state = _played()
    before = dk.effect_supply_rate(state, 0)
    state.reshuffle_votes[1] = True
    state.reshuffle_next_turn = True
    assert np.array_equal(dk.effect_supply_rate(state, 0), before)


def test_boundary_pool_is_what_the_engine_reforms():
    state = _drain_to(_played(plies=12), 0)
    pool = dk.boundary_pool_composition(state, 0)
    probe = state.copy()
    probe._discard_step()
    probe._reform_deck()
    assert np.array_equal(dk._histogram(probe.deck), pool)


def test_boundary_features_refuse_expert_one_seat_and_afterstates():
    with pytest.raises(ValueError, match=r"2\+ player"):
        dk.effect_supply_rate(_game(expert=True), 0)
    for solo_rules in (True, False):
        with pytest.raises(ValueError, match=r"2\+ player"):
            dk.effect_supply_rate(_game(players=1, solo_rules=solo_rules), 0)
    after = _played().copy()
    assert after.prepare_turn_boundary()
    with pytest.raises(ValueError, match="afterstate"):
        dk.effect_supply_rate(after, 0)


# `reveals_to_reform` is not part of step 3; this test is carried over from
# `welcome-to-engine` 5a73fe6, whose step-3 block was otherwise superseded by
# 9d4cf9b.  The function it covers is still live.
def test_reveals_to_reform_counts_the_reveal_that_finds_the_deck_empty():
    """`floor(D/3) + 1`: `_draw` reforms only when it FINDS the deck empty."""
    state = _game(advanced=True)
    for remaining, expected in ((0, 1), (3, 2), (4, 2), (6, 3), (7, 3)):
        state.deck_pos = len(state.deck) - remaining
        assert state.deck_remaining == remaining
        assert dk.reveals_to_reform(state) == expected
