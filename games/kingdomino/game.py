from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import random
from typing import Optional

from .board import Board, Placement
from .dominoes import DOMINOES


class Phase(IntEnum):
    INITIAL_SELECTION = 0
    PLACE_AND_SELECT = 1
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
    # placement=None means the current domino is discarded because it cannot be placed
    placement: Optional[Placement]
    pick_domino_id: Optional[int]


@dataclass(frozen=True, slots=True)
class GameConfig:
    players: int = 2
    board_size: int = 7
    canvas_size: int = 15
    harmony: bool = True
    middle_kingdom: bool = True
    mighty_duel: bool = True


@dataclass
class GameState:
    config: GameConfig
    boards: list[Board]
    deck: list[int]
    current_row: list[int]
    pending_claims: list[Claim]
    next_claims: list[Claim]
    phase: Phase
    actor_index: int = 0
    initial_pick_count: int = 0
    start_player: int = 0
    history: list[object] = field(default_factory=list)
    # Per-player count of dominoes discarded (forced when a claimed tile has no
    # legal placement).  A discard permanently forfeits the Harmony bonus for
    # that player (Harmony needs all 24 dominoes placed → a full 7×7).  Tracked
    # explicitly because it cannot be reconstructed from board/claim state alone.
    discards: list[int] = field(default_factory=lambda: [0, 0])

    @classmethod
    def new(cls, seed: int | None = None, config: GameConfig | None = None, start_player: int | None = None) -> "GameState":
        config = config or GameConfig()
        rng = random.Random(seed)
        deck = list(DOMINOES.keys())
        rng.shuffle(deck)
        row = sorted(deck[:4])
        deck = deck[4:]
        if start_player is None:
            start_player = rng.randrange(config.players)
        return cls(
            config=config,
            boards=[Board(config.canvas_size) for _ in range(config.players)],
            deck=deck,
            current_row=row,
            pending_claims=[],
            next_claims=[],
            phase=Phase.INITIAL_SELECTION,
            start_player=start_player,
        )

    @property
    def current_actor(self) -> int:
        if self.phase == Phase.INITIAL_SELECTION:
            # Mighty Duel opening pick order: P1, P2, P2, P1.
            order = [self.start_player, 1 - self.start_player, 1 - self.start_player, self.start_player]
            return order[self.initial_pick_count]
        if self.phase in (Phase.PLACE_AND_SELECT, Phase.FINAL_PLACEMENT):
            return self.pending_claims[self.actor_index].player
        raise ValueError("No current actor after game over")

    def copy(self) -> "GameState":
        return GameState(
            config=self.config,
            boards=[b.copy() for b in self.boards],
            deck=list(self.deck),
            current_row=list(self.current_row),
            pending_claims=list(self.pending_claims),
            next_claims=list(self.next_claims),
            phase=self.phase,
            actor_index=self.actor_index,
            initial_pick_count=self.initial_pick_count,
            start_player=self.start_player,
            history=list(self.history),
            discards=list(self.discards),
        )

    def legal_actions(self) -> list[PickAction | TurnAction]:
        # Lazy import: action_codec imports this module, so importing it at top
        # level would be circular.  By call time both modules are loaded.
        from .action_codec import encode_action

        if self.phase == Phase.GAME_OVER:
            return []
        if self.phase == Phase.INITIAL_SELECTION:
            actions: list[PickAction | TurnAction] = [PickAction(d) for d in self.current_row]
        else:
            claim = self.pending_claims[self.actor_index]
            domino = DOMINOES[claim.domino_id]
            placements = self.boards[claim.player].legal_placements(domino)
            # Kingdomino allows/forces a discard when the domino cannot be placed.
            placement_options: list[Optional[Placement]] = placements if placements else [None]
            if self.phase == Phase.FINAL_PLACEMENT:
                actions = [TurnAction(p, None) for p in placement_options]
            else:
                actions = [TurnAction(p, pick) for p in placement_options for pick in self.current_row]

        # Canonical ordering: ascending joint action index.  This makes the
        # search tree's child iteration — and thus PUCT tie-breaking — fully
        # deterministic and identical to the Rust engine, rather than depending
        # on the set-iteration order of board.legal_placements().  Legal joint
        # indices are unique (action_codec.legal_mask enforces no collisions),
        # so the sort is total.  (No-op for INITIAL_SELECTION: already ascending.)
        actions.sort(key=lambda a: encode_action(a, self))
        return actions

    def step(self, action: PickAction | TurnAction) -> "GameState":
        s = self.copy()
        s.history.append(action)

        if s.phase == Phase.INITIAL_SELECTION:
            if not isinstance(action, PickAction):
                raise TypeError("Expected PickAction")
            if action.domino_id not in s.current_row:
                raise ValueError("Picked domino not available")
            s.current_row.remove(action.domino_id)
            s.next_claims.append(Claim(s.current_actor, action.domino_id))
            s.initial_pick_count += 1
            if s.initial_pick_count == 4:
                s.pending_claims = sorted(s.next_claims, key=lambda c: c.domino_id)
                s.next_claims = []
                s.current_row = sorted(s.deck[:4])
                s.deck = s.deck[4:]
                s.actor_index = 0
                s.phase = Phase.PLACE_AND_SELECT
            return s

        if not isinstance(action, TurnAction):
            raise TypeError("Expected TurnAction")
        claim = s.pending_claims[s.actor_index]
        if action.placement is not None:
            s.boards[claim.player].place(DOMINOES[claim.domino_id], action.placement)
        else:
            # Forced discard: the claimed tile had no legal placement.
            s.discards[claim.player] += 1

        if s.phase == Phase.PLACE_AND_SELECT:
            if action.pick_domino_id not in s.current_row:
                raise ValueError("Picked domino not available")
            s.current_row.remove(action.pick_domino_id)
            s.next_claims.append(Claim(claim.player, action.pick_domino_id))

        s.actor_index += 1
        if s.actor_index >= len(s.pending_claims):
            if s.phase == Phase.FINAL_PLACEMENT:
                s.phase = Phase.GAME_OVER
            else:
                s.pending_claims = sorted(s.next_claims, key=lambda c: c.domino_id)
                s.next_claims = []
                s.actor_index = 0
                if s.deck:
                    s.current_row = sorted(s.deck[:4])
                    s.deck = s.deck[4:]
                    s.phase = Phase.PLACE_AND_SELECT
                else:
                    s.current_row = []
                    s.phase = Phase.FINAL_PLACEMENT
        return s

    def scores(self) -> list[int]:
        return [b.score(self.config.harmony, self.config.middle_kingdom).total for b in self.boards]


def determine_winner(state: GameState) -> Optional[int]:
    """Authoritative winner determination via the official tiebreaker cascade.

    Returns the winning player index (0 or 1), or None for a true draw.

    Cascade (official Kingdomino rules):
        1. Highest total score wins.
        2. Score-tied: most tiles in a single connected territory wins.
        3. Still tied: most total crowns wins.
        4. Still tied: draw → None.

    This is the SINGLE SOURCE OF TRUTH for who won a game. Self-play winner
    logic (benchmark/eval) and encoder.compute_target_win both route through it,
    so the cascade is implemented in exactly one place. It scores the current
    boards directly, so it is meaningful only at a terminal state — callers that
    need a terminal guard (e.g. compute_target_win) enforce it themselves.

    Only defined for the 2-player game (the project's Mighty Duel target).
    """
    if len(state.boards) != 2:
        raise ValueError(
            f"determine_winner is defined for the 2-player game; "
            f"got {len(state.boards)} boards."
        )
    # Tuple comparison applies the cascade in order: total score, then largest
    # single territory (tile count), then total crowns.
    keys = []
    for b in state.boards:
        sb = b.score(state.config.harmony, state.config.middle_kingdom)
        keys.append((sb.total, sb.largest_territory_size, sb.total_crowns))
    if keys[0] > keys[1]:
        return 0
    if keys[1] > keys[0]:
        return 1
    return None
