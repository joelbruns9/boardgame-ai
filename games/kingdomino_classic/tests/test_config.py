from __future__ import annotations

import pytest

from games.kingdomino_classic import GAME_ID, RULES_VERSION, ClassicGameConfig


@pytest.mark.parametrize(
    ("players", "deck_size", "draft_row_size"),
    [(3, 36, 3), (4, 48, 4)],
)
def test_player_count_derives_classic_setup_sizes(
    players: int, deck_size: int, draft_row_size: int
) -> None:
    config = ClassicGameConfig(players=players)

    assert config.board_size == 5
    assert config.dominoes_per_player == 12
    assert config.deck_size == deck_size
    assert config.draft_row_size == draft_row_size


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
        "draft_row_size": 3,
        "deck_size": 36,
    }
