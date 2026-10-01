"""Thin Welcome To adapter for the shared ``games.az_loop`` seam.

Simpler than the Kingdomino adapter because this engine already speaks the flat
action index natively — there is no encode/decode wrapper to keep in sync.

``games.az_loop.core.GameAdapter`` types ``outcome`` for two seats; Welcome To
runs 1 to N, so the scores tuple is variable length. Anything in the loop that
indexes it positionally by seat is fine; anything that assumes exactly two is not.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from games.welcome_to.action_codec import NUM_ACTIONS
from games.welcome_to.game import GameConfig, GameState


class WelcomeToLoopAdapter:
    name = "welcome_to"

    def __init__(self, config: Optional[GameConfig] = None) -> None:
        #: Default is the *advanced* two-seat game.  Training runs multiplayer
        #: throughout and samples 2-4 seats per game; two is the cheapest
        #: configuration that still has every race mechanic live.
        #:
        #: Do not default this to ``players=1``.  A one-seat game silently
        #: switches scoring rules -- ``TEMP_SOLO_SCORE`` replaces the 7/4/1
        #: ranking and every City Plan pays its first-place value -- so a
        #: single-seat corpus trains on a different game.  See SELF_PLAY_PLAN.md.
        self.config = config or GameConfig(
            players=2, advanced=True, solo_rules=False
        )

    def new_game(self, seed: int, first_player: int = 0) -> GameState:
        # Welcome To has no first-player advantage to rotate: every seat sees the
        # same three stacks and acts on its own sheet, so the argument is ignored.
        return GameState.new(seed=seed, config=self.config)

    def actor(self, state: GameState) -> int:
        return state.actor

    def legal_actions(self, state: GameState) -> Sequence[int]:
        return state.legal_actions()

    def step(self, state: GameState, action: int) -> GameState:
        return state.step(action)

    def terminal(self, state: GameState) -> bool:
        return state.is_terminal

    def outcome(self, state: GameState) -> tuple[Optional[int], tuple[int, ...], str]:
        scores = tuple(state.scores())
        winners = state.winners()
        # a shared best score is a genuine draw, reported as no winner
        winner = winners[0] if len(winners) == 1 else None
        return winner, scores, "score"

    def contract(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "players": self.config.players,
            "advanced": self.config.advanced,
            "expert": self.config.expert,
            "solo_rules": self.config.solo_rules,
            "action_space": NUM_ACTIONS,
            "state_transition": "immutable",
        }
