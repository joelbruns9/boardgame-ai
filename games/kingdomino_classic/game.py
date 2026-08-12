"""Three- and four-player classic Kingdomino state machine.

The draft transitions mirror the BGA implementation in
``BGA Files/kingdomino/kingdomino.game.php``.  Each row contains four sorted
dominoes.  Three-player games make three claims and discard the ownerless
remainder; four-player games claim all four.  Claimed domino number determines
the next placement order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import random

from .board import ClassicBoard, Placement, ScoreBreakdown
from .config import ClassicGameConfig
from .dominoes import DOMINOES


class Phase(IntEnum):
    INITIAL_DRAFT = 0
    PLACE_AND_DRAFT = 1
    FINAL_PLACEMENT = 2
    GAME_OVER = 3


@dataclass(frozen=True, slots=True)
class Claim:
    player: int
    domino_id: int


@dataclass(frozen=True, slots=True)
class PickAction:
    domino_id: int


@dataclass(frozen=True, slots=True)
class TurnAction:
    """Place the current domino, then claim from the next row when present."""

    placement: Placement | None
    pick_domino_id: int | None


Action = PickAction | TurnAction


@dataclass(frozen=True, slots=True)
class GameResult:
    scores: tuple[int, ...]
    tiebreak_keys: tuple[tuple[int, int, int], ...]
    ranks: tuple[int, ...]
    winners: tuple[int, ...]
    win_shares: tuple[float, ...]


@dataclass
class ClassicGameState:
    config: ClassicGameConfig
    boards: list[ClassicBoard]
    deck: list[int]
    draft_row: list[int]
    pending_claims: list[Claim]
    next_claims: list[Claim]
    unclaimed_discards: list[int]
    forced_discards: list[list[int]]
    phase: Phase
    initial_pick_count: int = 0
    start_player: int = 0
    initial_player_order: list[int] = field(default_factory=list)
    history: list[Action] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        *,
        seed: int | None = None,
        config: ClassicGameConfig | None = None,
        start_player: int | None = None,
    ) -> "ClassicGameState":
        config = config or ClassicGameConfig()
        rng = random.Random(seed)
        deck = list(DOMINOES)
        rng.shuffle(deck)
        if start_player is None:
            start_player = rng.randrange(config.players)
        if not 0 <= start_player < config.players:
            raise ValueError(
                f"start_player must be in [0, {config.players}); got {start_player}."
            )

        state = cls(
            config=config,
            boards=[ClassicBoard() for _ in range(config.players)],
            deck=deck[config.draft_row_size :],
            draft_row=sorted(deck[: config.draft_row_size]),
            pending_claims=[],
            next_claims=[],
            unclaimed_discards=[],
            forced_discards=[[] for _ in range(config.players)],
            phase=Phase.INITIAL_DRAFT,
            start_player=start_player,
            initial_player_order=[
                (start_player + offset) % config.players
                for offset in range(config.players)
            ],
        )
        state.assert_invariants()
        return state

    @property
    def current_actor(self) -> int:
        if self.phase == Phase.INITIAL_DRAFT:
            return self.initial_player_order[self.initial_pick_count]
        if self.phase in (Phase.PLACE_AND_DRAFT, Phase.FINAL_PLACEMENT):
            if not self.pending_claims:
                raise AssertionError("A placement phase requires a pending claim.")
            return self.pending_claims[0].player
        raise ValueError("There is no current actor after game over.")

    @property
    def is_terminal(self) -> bool:
        return self.phase == Phase.GAME_OVER

    def copy(self) -> "ClassicGameState":
        return ClassicGameState(
            config=self.config,
            boards=[board.copy() for board in self.boards],
            deck=list(self.deck),
            draft_row=list(self.draft_row),
            pending_claims=list(self.pending_claims),
            next_claims=list(self.next_claims),
            unclaimed_discards=list(self.unclaimed_discards),
            forced_discards=[list(ids) for ids in self.forced_discards],
            phase=self.phase,
            initial_pick_count=self.initial_pick_count,
            start_player=self.start_player,
            initial_player_order=list(self.initial_player_order),
            history=list(self.history),
        )

    def legal_actions(self) -> list[Action]:
        if self.phase == Phase.GAME_OVER:
            return []
        if self.phase == Phase.INITIAL_DRAFT:
            return [PickAction(domino_id) for domino_id in self.draft_row]

        claim = self.pending_claims[0]
        placements = self.boards[claim.player].legal_placements(
            DOMINOES[claim.domino_id]
        )
        placement_options: list[Placement | None] = placements or [None]
        if self.phase == Phase.FINAL_PLACEMENT:
            return [TurnAction(placement, None) for placement in placement_options]
        return [
            TurnAction(placement, domino_id)
            for placement in placement_options
            for domino_id in self.draft_row
        ]

    def step(self, action: Action) -> "ClassicGameState":
        if self.phase == Phase.GAME_OVER:
            raise ValueError("Cannot act after game over.")

        state = self.copy()
        state.history.append(action)
        if state.phase == Phase.INITIAL_DRAFT:
            state._apply_initial_pick(action)
        else:
            state._apply_turn(action)
        state.assert_invariants()
        return state

    def _apply_initial_pick(self, action: Action) -> None:
        if not isinstance(action, PickAction):
            raise TypeError("Initial draft requires PickAction.")
        if action.domino_id not in self.draft_row:
            raise ValueError(f"Domino {action.domino_id} is not available.")

        player = self.current_actor
        self.draft_row.remove(action.domino_id)
        self.next_claims.append(Claim(player, action.domino_id))
        self.initial_pick_count += 1
        if self.initial_pick_count == self.config.selections_per_round:
            self._resolve_completed_draft()

    def _apply_turn(self, action: Action) -> None:
        if not isinstance(action, TurnAction):
            raise TypeError("Placement phases require TurnAction.")

        claim = self.pending_claims[0]
        board = self.boards[claim.player]
        domino = DOMINOES[claim.domino_id]
        placements = board.legal_placements(domino)
        if placements:
            if action.placement is None or action.placement not in placements:
                raise ValueError("A legal placement must be selected.")
            board.place(domino, action.placement)
        else:
            if action.placement is not None:
                raise ValueError("An unplaceable domino must be discarded.")
            self.forced_discards[claim.player].append(claim.domino_id)

        if self.phase == Phase.PLACE_AND_DRAFT:
            if action.pick_domino_id not in self.draft_row:
                raise ValueError("A displayed domino must be selected.")
            self.draft_row.remove(action.pick_domino_id)
            self.next_claims.append(Claim(claim.player, action.pick_domino_id))
        elif action.pick_domino_id is not None:
            raise ValueError("There is no draft during final placement.")

        self.pending_claims.pop(0)
        if self.pending_claims:
            return
        if self.phase == Phase.FINAL_PLACEMENT:
            self.phase = Phase.GAME_OVER
        else:
            self._resolve_completed_draft()

    def _resolve_completed_draft(self) -> None:
        if len(self.next_claims) != self.config.selections_per_round:
            raise AssertionError("A draft resolved with the wrong number of claims.")
        expected_unclaimed = self.config.unclaimed_dominoes_per_round
        if len(self.draft_row) != expected_unclaimed:
            raise AssertionError("A draft resolved with the wrong remainder.")
        self.unclaimed_discards.extend(self.draft_row)
        self.draft_row = []
        self.pending_claims = sorted(
            self.next_claims, key=lambda claim: claim.domino_id
        )
        self.next_claims = []

        if self.deck:
            if len(self.deck) < self.config.draft_row_size:
                raise AssertionError("The draw pile ended with an incomplete row.")
            self.draft_row = sorted(self.deck[: self.config.draft_row_size])
            del self.deck[: self.config.draft_row_size]
            self.phase = Phase.PLACE_AND_DRAFT
        else:
            self.phase = Phase.FINAL_PLACEMENT

    def score_breakdowns(self) -> tuple[ScoreBreakdown, ...]:
        return tuple(
            board.score(
                harmony=self.config.harmony,
                middle_kingdom=self.config.middle_kingdom,
            )
            for board in self.boards
        )

    def scores(self) -> tuple[int, ...]:
        return tuple(score.total for score in self.score_breakdowns())

    def result(self) -> GameResult:
        if not self.is_terminal:
            raise ValueError("Official results are defined only at game over.")
        breakdowns = self.score_breakdowns()
        keys = tuple(score.tiebreak_key for score in breakdowns)
        best = max(keys)
        winners = tuple(player for player, key in enumerate(keys) if key == best)
        ranks = tuple(1 + sum(other > key for other in keys) for key in keys)
        share = 1.0 / len(winners)
        win_shares = tuple(
            share if player in winners else 0.0
            for player in range(self.config.players)
        )
        return GameResult(
            scores=tuple(score.total for score in breakdowns),
            tiebreak_keys=keys,
            ranks=ranks,
            winners=winners,
            win_shares=win_shares,
        )

    def returns(self) -> tuple[float, ...]:
        """Terminal shared-win returns for future vector-value training."""

        return self.result().win_shares

    def state_key(self) -> tuple[object, ...]:
        return (
            self.config.configuration_key,
            int(self.phase),
            self.start_player,
            tuple(self.initial_player_order),
            self.initial_pick_count,
            tuple(self.deck),
            tuple(self.draft_row),
            tuple((claim.player, claim.domino_id) for claim in self.pending_claims),
            tuple((claim.player, claim.domino_id) for claim in self.next_claims),
            tuple(self.unclaimed_discards),
            tuple(tuple(ids) for ids in self.forced_discards),
            tuple(board.state_key() for board in self.boards),
        )

    def _inventory_ids(self) -> list[int]:
        inventory = [*self.deck, *self.draft_row, *self.unclaimed_discards]
        inventory.extend(claim.domino_id for claim in self.pending_claims)
        inventory.extend(claim.domino_id for claim in self.next_claims)
        for board in self.boards:
            inventory.extend(board.placed_domino_ids)
        for ids in self.forced_discards:
            inventory.extend(ids)
        return inventory

    def assert_invariants(self) -> None:
        players = self.config.players
        if set(self.initial_player_order) != set(range(players)):
            raise AssertionError(
                "Initial player order must contain every player once."
            )
        if self.start_player != self.initial_player_order[0]:
            raise AssertionError("Start player must lead the initial player order.")
        if len(self.boards) != players or len(self.forced_discards) != players:
            raise AssertionError("Per-player state does not match player count.")
        if self.draft_row != sorted(self.draft_row):
            raise AssertionError("The displayed draft row must stay sorted.")
        if self.pending_claims != sorted(
            self.pending_claims, key=lambda claim: claim.domino_id
        ):
            raise AssertionError("Pending claims must follow domino-number order.")
        if len(self.deck) % self.config.draft_row_size != 0:
            raise AssertionError("The draw pile must contain complete future rows.")

        for board in self.boards:
            board.assert_invariants()
        for claim in (*self.pending_claims, *self.next_claims):
            if not 0 <= claim.player < players:
                raise AssertionError("A claim references an invalid player.")

        inventory = self._inventory_ids()
        expected = set(DOMINOES)
        if len(inventory) != len(expected) or set(inventory) != expected:
            raise AssertionError("The 48-domino inventory is not conserved uniquely.")

        owned_counts = [
            self.boards[player].placed_domino_count
            + len(self.forced_discards[player])
            + sum(claim.player == player for claim in self.pending_claims)
            + sum(claim.player == player for claim in self.next_claims)
            for player in range(players)
        ]
        if any(count > self.config.dominoes_per_player for count in owned_counts):
            raise AssertionError("A player owns more than twelve dominoes.")

        if self.phase == Phase.INITIAL_DRAFT:
            if self.pending_claims:
                raise AssertionError("Initial draft cannot have current claims.")
            if len(self.draft_row) + len(self.next_claims) != 4:
                raise AssertionError("Initial draft row does not conserve four dominoes.")
            if self.initial_pick_count != len(self.next_claims):
                raise AssertionError("Initial pick count disagrees with claims.")
        elif self.phase == Phase.PLACE_AND_DRAFT:
            if len(self.pending_claims) + len(self.next_claims) != players:
                raise AssertionError("Placement round does not conserve player claims.")
            if len(self.draft_row) + len(self.next_claims) != 4:
                raise AssertionError("Future draft row does not conserve four dominoes.")
        elif self.phase == Phase.FINAL_PLACEMENT:
            if self.draft_row or self.next_claims or self.deck:
                raise AssertionError("Final placement cannot retain future dominoes.")
        elif self.phase == Phase.GAME_OVER:
            if self.pending_claims or self.next_claims or self.draft_row or self.deck:
                raise AssertionError("Game over cannot retain unresolved dominoes.")
            if any(
                count != self.config.dominoes_per_player for count in owned_counts
            ):
                raise AssertionError("Every player must finish with twelve claims.")

        if self.config.players == 4 and self.unclaimed_discards:
            raise AssertionError("Four-player games cannot have unclaimed discards.")
        if len(self.unclaimed_discards) > 12:
            raise AssertionError("More than twelve draft rows were discarded.")
        if self.phase == Phase.GAME_OVER:
            expected_unclaimed = (
                12 * self.config.unclaimed_dominoes_per_round
            )
            if len(self.unclaimed_discards) != expected_unclaimed:
                raise AssertionError(
                    "Terminal unclaimed-discard count disagrees with player count."
                )
