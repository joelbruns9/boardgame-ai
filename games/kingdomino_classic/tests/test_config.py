from __future__ import annotations

import pytest

from games.kingdomino_classic import GAME_ID, RULES_VERSION, ClassicGameConfig


@pytest.mark.parametrize(
    (
        "players",
        "selections_per_round",
        "unclaimed_dominoes_per_round",
        "final_selection_is_forced",
    ),
    [(3, 3, 1, False), (4, 4, 0, True)],
)
def test_player_count_derives_classic_setup_sizes(
    players: int,
    selections_per_round: int,
    unclaimed_dominoes_per_round: int,
    final_selection_is_forced: bool,
) -> None:
    config = ClassicGameConfig(players=players)

    assert config.board_size == 5
    assert config.dominoes_per_player == 12
    assert config.deck_size == 48
    assert config.draft_row_size == 4
    assert config.selections_per_round == selections_per_round
    assert config.unclaimed_dominoes_per_round == unclaimed_dominoes_per_round
    assert config.final_selection_is_forced is final_selection_is_forced


@pytest.mark.parametrize("players", [0, 1, 2, 5])
def test_unsupported_player_counts_are_rejected(players: int) -> None:
    with pytest.raises(ValueError, match="exactly 3 or 4 players"):
        ClassicGameConfig(players=players)  # type: ignore[arg-type]


def test_manifest_identity_is_complete_and_stable() -> None:
    config = ClassicGameConfig(players=3, harmony=False, middle_kingdom=True)

    assert config.configuration_key == "3p-h0-m1"
    assert config.manifest_fields() == {
        "game_id": GAME_ID,
        "rules_version": RULES_VERSION,
        "players": 3,
        "harmony": False,
        "middle_kingdom": True,
        "board_size": 5,
        "draft_row_size": 4,
        "deck_size": 48,
        "selections_per_round": 3,
        "unclaimed_dominoes_per_round": 1,
        "final_selection_is_forced": False,
    }
