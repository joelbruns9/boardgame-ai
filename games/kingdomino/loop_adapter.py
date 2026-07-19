"""Thin Kingdomino adapter for the shared ``games.az_loop`` regression seam."""

from __future__ import annotations

from typing import Any

from .action_codec import NUM_JOINT_ACTIONS, decode_action, encode_action
from .game import GameState, Phase, determine_winner


class KingdominoLoopAdapter:
    name = "kingdomino"

    def new_game(self, seed: int, first_player: int = 0) -> GameState:
        return GameState.new(seed=seed, start_player=first_player)

    def actor(self, state: GameState) -> int:
        return state.current_actor

    def legal_actions(self, state: GameState) -> tuple[int, ...]:
        return tuple(encode_action(action, state) for action in state.legal_actions())

    def step(self, state: GameState, action: int) -> GameState:
        return state.step(decode_action(action, state))

    def terminal(self, state: GameState) -> bool:
        return state.phase is Phase.GAME_OVER

    def outcome(
        self, state: GameState
    ) -> tuple[int | None, tuple[int, int], str]:
        scores = tuple(state.scores())
        return determine_winner(state), scores, "score"

    def contract(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "players": 2,
            "action_space": NUM_JOINT_ACTIONS,
            "state_transition": "immutable",
        }
