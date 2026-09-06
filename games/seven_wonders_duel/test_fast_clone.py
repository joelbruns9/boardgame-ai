"""``fast_clone`` must be indistinguishable from ``GameState.clone``.

The fast path shares every immutable field instead of copying it, so the failure
mode is *aliasing*: a mutable object reachable from both the copy and the
original. These tests look for that directly -- structural equality against
``deepcopy``, then identity checks on everything the engine writes through, then
a played-out game where divergence would surface as a rules difference.
"""

from __future__ import annotations

import random

import pytest

from .codec import decode_action, legal_action_indices
from .engine import apply_action
from .fast_clone import check_layout, fast_clone
from .game import Phase, new_game


def _states(count: int = 40, seed: int = 20260905):
    """Positions from a random playout, across all phases."""

    rng = random.Random(seed)
    game = new_game(seed=seed)
    out = [game.clone()]
    while game.phase is not Phase.COMPLETE and len(out) < count:
        indices = list(legal_action_indices(game))
        if not indices:
            break
        apply_action(game, decode_action(game, rng.choice(indices)))
        out.append(game.clone())
    return out


def test_layout_is_pinned():
    check_layout()


@pytest.mark.parametrize("index", range(0, 40, 7))
def test_fields_match_deepcopy(index):
    states = _states()
    state = states[min(index, len(states) - 1)]
    fast, slow = fast_clone(state), state.clone()

    assert fast.phase is slow.phase
    assert fast.active_player == slow.active_player
    assert fast.age == slow.age
    assert fast.conflict_position == slow.conflict_position
    assert fast.winner == slow.winner
    assert fast.victory_type == slow.victory_type
    assert fast.discard_pile == slow.discard_pile
    assert fast.buried_cards == slow.buried_cards
    assert fast.retired_wonders == slow.retired_wonders
    assert fast.wonder_offer == slow.wonder_offer
    assert fast.military_tokens_remaining == slow.military_tokens_remaining
    assert fast.wonder_burials == slow.wonder_burials
    assert fast.pending_choice == slow.pending_choice
    assert fast.tableau.age == slow.tableau.age
    assert set(fast.tableau.cards) == set(slow.tableau.cards)
    for slot_id, card in fast.tableau.cards.items():
        other = slow.tableau.cards[slot_id]
        assert (card.card_name, card.revealed, card.present) == (
            other.card_name, other.revealed, other.present
        )
    for seat in (0, 1):
        a, b = fast.cities[seat], slow.cities[seat]
        assert a.coins == b.coins
        assert a.buildings == b.buildings
        assert a.wonders == b.wonders
        assert a.built_wonders == b.built_wonders
        assert a.progress_tokens == b.progress_tokens
        assert a.claimed_science_pairs == b.claimed_science_pairs


def test_no_mutable_state_is_shared():
    """The whole risk of the fast path, checked by identity."""

    state = _states()[-1]
    copy_ = fast_clone(state)

    assert copy_ is not state
    assert copy_.tableau is not state.tableau
    assert copy_.tableau.cards is not state.tableau.cards
    for slot_id, card in copy_.tableau.cards.items():
        assert card is not state.tableau.cards[slot_id]
    assert copy_.cities is not state.cities
    for seat in (0, 1):
        assert copy_.cities[seat] is not state.cities[seat]
        assert copy_.cities[seat].buildings is not state.cities[seat].buildings
        assert copy_.cities[seat].wonders is not state.cities[seat].wonders
        assert (copy_.cities[seat].built_wonders
                is not state.cities[seat].built_wonders)
        assert (copy_.cities[seat].progress_tokens
                is not state.cities[seat].progress_tokens)
        assert (copy_.cities[seat].claimed_science_pairs
                is not state.cities[seat].claimed_science_pairs)
    assert copy_.discard_pile is not state.discard_pile
    assert copy_.buried_cards is not state.buried_cards
    assert copy_.retired_wonders is not state.retired_wonders
    assert copy_.wonder_offer is not state.wonder_offer
    assert copy_.military_tokens_remaining is not state.military_tokens_remaining
    assert copy_.wonder_burials is not state.wonder_burials
    assert copy_.age_decks is not state.age_decks
    assert copy_.removed_age_cards is not state.removed_age_cards
    assert copy_.rng is not state.rng


def test_mutating_the_copy_leaves_the_original_alone():
    state = _states()[-1]
    before_coins = state.cities[0].coins
    before_buildings = list(state.cities[0].buildings)
    before_discard = list(state.discard_pile)
    before_present = {
        slot_id: card.present for slot_id, card in state.tableau.cards.items()
    }

    copy_ = fast_clone(state)
    copy_.cities[0].coins += 17
    copy_.cities[0].buildings.append("Palace")
    copy_.cities[0].claimed_science_pairs.add(next(iter(
        copy_.cities[0].claimed_science_pairs), None
    ) or "x")
    copy_.discard_pile.append("Senate")
    copy_.retired_wonders.add("The Pyramids")
    copy_.military_tokens_remaining.clear()
    for card in copy_.tableau.cards.values():
        card.present = not card.present

    assert state.cities[0].coins == before_coins
    assert state.cities[0].buildings == before_buildings
    assert state.discard_pile == before_discard
    assert {
        slot_id: card.present for slot_id, card in state.tableau.cards.items()
    } == before_present


def test_rng_stream_is_preserved_and_independent():
    """``clone``'s contract is the exact future stream, not merely a fresh RNG."""

    state = _states()[-1]
    fast, slow = fast_clone(state), state.clone()
    original = [state.rng.random() for _ in range(8)]
    assert [fast.rng.random() for _ in range(8)] == original
    assert [slow.rng.random() for _ in range(8)] == original


def test_playouts_agree_action_for_action():
    """Divergence would show up as different legal moves or a different result."""

    for seed in (1, 7, 20260905):
        rng_a, rng_b = random.Random(seed), random.Random(seed)
        fast = fast_clone(new_game(seed=seed))
        slow = new_game(seed=seed).clone()
        for _ in range(400):
            if slow.phase is Phase.COMPLETE:
                break
            a = list(legal_action_indices(fast))
            b = list(legal_action_indices(slow))
            assert a == b
            index_a = rng_a.choice(a)
            index_b = rng_b.choice(b)
            assert index_a == index_b
            apply_action(fast, decode_action(fast, index_a))
            apply_action(slow, decode_action(slow, index_b))
            fast = fast_clone(fast)
        assert fast.phase is slow.phase
        assert fast.winner == slow.winner
        assert fast.victory_type == slow.victory_type
        assert fast.conflict_position == slow.conflict_position
