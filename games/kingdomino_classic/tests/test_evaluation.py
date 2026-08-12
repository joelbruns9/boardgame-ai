from __future__ import annotations

import pytest

from games.kingdomino_classic import ClassicGameConfig
from games.kingdomino_classic.baselines import RandomBot
from games.kingdomino_classic.evaluation import (
    Participant,
    play_game,
    run_challenger_tournament,
    run_seat_balanced_tournament,
)


def _random_participant(name: str) -> Participant:
    return Participant(name=name, make_bot=RandomBot)


@pytest.mark.parametrize("players", [3, 4])
def test_lineup_rotation_balances_every_named_participant(players: int) -> None:
    config = ClassicGameConfig(players=players)
    participants = [_random_participant(f"bot_{index}") for index in range(players)]

    report = run_seat_balanced_tournament(
        participants, config=config, seeds=[10, 11]
    )

    assert report.games == 2 * players
    for participant in participants:
        standing = report.standing(participant.name)
        assert standing.seat_games == 2 * players
        assert standing.seat_counts == (2,) * players


def test_challenger_tournament_aggregates_the_repeated_field() -> None:
    config = ClassicGameConfig(players=3)
    report = run_challenger_tournament(
        _random_participant("challenger"),
        _random_participant("field"),
        config=config,
        seeds=[7, 8],
    )

    challenger = report.standing("challenger")
    field = report.standing("field")
    assert report.equal_strength_win_share == pytest.approx(1 / 3)
    assert report.win_share_lift("challenger") == pytest.approx(
        challenger.win_share - 1 / 3
    )
    assert report.to_dict()["equal_strength_win_share"] == pytest.approx(1 / 3)
    assert challenger.seat_games == 6
    assert challenger.seat_counts == (2, 2, 2)
    assert field.seat_games == 12
    assert field.seat_counts == (4, 4, 4)


def test_tournament_is_reproducible() -> None:
    config = ClassicGameConfig(players=3)
    participants = [_random_participant(name) for name in ("a", "b", "c")]

    first = run_seat_balanced_tournament(participants, config=config, seeds=[4])
    second = run_seat_balanced_tournament(participants, config=config, seeds=[4])

    assert first == second
    assert first.to_dict() == second.to_dict()


def test_play_game_rejects_the_wrong_number_of_seats() -> None:
    with pytest.raises(ValueError, match="Expected 4 seated participants"):
        play_game(
            [_random_participant("a")],
            config=ClassicGameConfig(players=4),
            seed=1,
        )


def test_tournament_rejects_an_empty_seed_set() -> None:
    config = ClassicGameConfig(players=3)
    participants = [_random_participant(name) for name in ("a", "b", "c")]

    with pytest.raises(ValueError, match="at least one seed"):
        run_seat_balanced_tournament(participants, config=config, seeds=[])
