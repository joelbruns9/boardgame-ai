"""Rules configuration for the unified 3-4 player Kingdomino model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar, Literal


GAME_ID = "kingdomino_classic"
RULES_VERSION = 1


@dataclass(frozen=True, slots=True)
class ClassicGameConfig:
    """A supported classic Kingdomino rules configuration.

    The first model intentionally covers only the standard 5x5 three- and
    four-player game.  Two-player Mighty Duel remains in ``games.kingdomino``.
    Derived quantities are properties so manifests contain only choices that
    actually distinguish training configurations.
    """

    players: Literal[3, 4] = 4
    harmony: bool = True
    middle_kingdom: bool = True

    board_size: ClassVar[int] = 5
    dominoes_per_player: ClassVar[int] = 12
    max_players: ClassVar[int] = 4
    total_dominoes: ClassVar[int] = 48

    def __post_init__(self) -> None:
        if self.players not in (3, 4):
            raise ValueError(
                "ClassicGameConfig supports exactly 3 or 4 players; "
                f"got {self.players}."
            )

    @property
    def draft_row_size(self) -> int:
        """Number of face-up dominoes in every three- or four-player draft."""

        return 4

    @property
    def deck_size(self) -> int:
        """Number of dominoes used across the twelve draft rounds."""

        return self.total_dominoes

    @property
    def selections_per_round(self) -> int:
        """Number of claimed dominoes before a draft row is resolved."""

        return self.players

    @property
    def unclaimed_dominoes_per_round(self) -> int:
        """Dominoes discarded after all players have selected."""

        return self.draft_row_size - self.selections_per_round

    @property
    def final_selection_is_forced(self) -> bool:
        """Whether the final player has only one domino remaining to select."""

        return self.players == 4

    @property
    def configuration_key(self) -> str:
        """Stable short key for configs, reports, and evaluation strata."""

        harmony = "h1" if self.harmony else "h0"
        middle = "m1" if self.middle_kingdom else "m0"
        return f"{self.players}p-{harmony}-{middle}"

    def manifest_fields(self) -> dict[str, object]:
        """Serializable identity fields to stamp into generated artifacts."""

        return {
            "game_id": GAME_ID,
            "rules_version": RULES_VERSION,
            **asdict(self),
            "board_size": self.board_size,
            "draft_row_size": self.draft_row_size,
            "deck_size": self.deck_size,
            "selections_per_round": self.selections_per_round,
            "unclaimed_dominoes_per_round": self.unclaimed_dominoes_per_round,
            "final_selection_is_forced": self.final_selection_is_forced,
        }
