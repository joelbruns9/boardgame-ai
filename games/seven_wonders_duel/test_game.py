import pytest

from games.seven_wonders_duel.data import GUILD_CARDS
from games.seven_wonders_duel.game import Phase, new_game


def _finish_wonder_draft(game):
    pickers = []
    while game.phase is Phase.WONDER_DRAFT:
        pickers.append(game.active_player)
        game.pick_wonder(game.legal_wonder_choices()[0])
    return pickers


def test_seeded_setup_is_reproducible_and_local():
    first = new_game(1729)
    second = new_game(1729)
    different = new_game(1730)
    assert first.setup_fingerprint() == second.setup_fingerprint()
    assert first.setup_fingerprint() != different.setup_fingerprint()


def test_setup_component_counts_and_age_three_composition():
    game = new_game(4)
    assert len(game.available_progress_tokens) == 5
    assert len(game.unused_progress_tokens) == 5
    assert len(game.wonder_groups[0]) == len(game.wonder_groups[1]) == 4
    assert len(game.unused_wonders) == 4
    assert all(len(game.age_decks[age]) == 20 for age in (1, 2, 3))
    assert all(len(game.removed_age_cards[age]) == 3 for age in (1, 2, 3))
    assert len(game.selected_guilds) == 3
    assert len(game.unused_guilds) == 4
    guild_names = {card.name for card in GUILD_CARDS}
    assert len(guild_names & set(game.age_decks[3])) == 3


@pytest.mark.parametrize(
    ("first_player", "expected_order"),
    [(0, [0, 1, 1, 0, 1, 0, 0, 1]), (1, [1, 0, 0, 1, 0, 1, 1, 0])],
)
def test_wonder_draft_uses_two_reversed_snake_rounds(first_player, expected_order):
    game = new_game(99, first_player=first_player)
    assert _finish_wonder_draft(game) == expected_order
    assert [len(city.wonders) for city in game.cities] == [4, 4]
    assert game.phase is Phase.PLAY_AGE
    assert game.active_player == first_player
    assert game.wonder_offer == []


def test_observation_during_draft_hides_future_offer_and_all_age_cards():
    game = new_game(12)
    observation = game.observation(0)
    assert observation.wonder_offer == game.wonder_groups[0]
    assert not set(game.wonder_groups[1]) & set(observation.wonder_offer)
    assert observation.tableau == ()


def test_play_observation_redacts_face_down_and_setup_removed_cards():
    game = new_game(21)
    _finish_wonder_draft(game)
    observation = game.observation(0)
    visible_names = {card.card_name for card in observation.tableau if card.card_name}
    hidden = [card for card in observation.tableau if card.present and not card.revealed]
    assert len(observation.tableau) == 20
    assert len(visible_names) == 12
    assert len(hidden) == 8
    assert all(card.card_name is None for card in hidden)
    assert not visible_names & set(game.removed_age_cards[1])
    assert all(name not in repr(observation) for name in game.removed_age_cards[1])


def test_only_accessible_card_can_be_taken_and_uncovered_card_is_reported():
    game = new_game(30)
    _finish_wonder_draft(game)
    tableau = game.tableau
    assert len(tableau.accessible_slot_ids()) == 6
    with pytest.raises(ValueError):
        tableau.take_accessible((0, 5))

    # Removing both cards that cover row 3 / x=2 exposes that face-down card.
    # Revelation itself is a chance event owned by the engine: take_accessible
    # only reports the newly accessible slots, and reveal() flips them.
    assert tableau.cards[(3, 2)].revealed is False
    _, newly = tableau.take_accessible((4, 1))
    assert newly == ()
    _, newly = tableau.take_accessible((4, 3))
    assert newly == ((3, 2),)
    assert tableau.cards[(3, 2)].revealed is False
    tableau.reveal((3, 2))
    assert tableau.cards[(3, 2)].revealed is True
    assert tableau.is_accessible((3, 2))


def test_observations_are_immutable_snapshots_not_live_city_lists():
    game = new_game(77)
    before = game.observation(0)
    game.pick_wonder(game.legal_wonder_choices()[0])
    after = game.observation(0)
    assert before.cities[0].wonders == ()
    assert sum(len(city.wonders) for city in after.cities) == 1
