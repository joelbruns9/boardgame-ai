"""Seven Wonders Duel engine adapter for shared AZ loop orchestration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .buffer import SPEC_VERSION
from .codec import NUM_ACTIONS, decode_action, legal_action_indices
from .encoder import ENCODER_SIGNATURE
from .engine import apply_action
from .game import GameState, Phase, new_game
from .search import state_actor


class SevenWondersDuelLoopAdapter:
    name = "seven_wonders_duel"

    def new_game(self, seed: int, first_player: int = 0) -> GameState:
        return new_game(seed, first_player=first_player)

    def actor(self, state: GameState) -> int:
        return state_actor(state)

    def legal_actions(self, state: GameState) -> tuple[int, ...]:
        return legal_action_indices(state)

    def step(self, state: GameState, action: int) -> GameState:
        apply_action(state, decode_action(state, action))
        return state

    def terminal(self, state: GameState) -> bool:
        return state.phase is Phase.COMPLETE

    def outcome(
        self, state: GameState
    ) -> tuple[int | None, tuple[int, int] | None, str]:
        victory = state.victory_type.value if state.victory_type else "unknown"
        return state.winner, state.final_scores, victory

    def contract(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        root = Path(__file__).resolve().parent
        for name in ("data.py", "rules.py", "engine.py", "codec.py"):
            digest.update(name.encode())
            digest.update((root / name).read_bytes())
        return {
            "adapter": self.name,
            "players": 2,
            "action_space": NUM_ACTIONS,
            "encoder_signature": ENCODER_SIGNATURE,
            "codec_spec_version": SPEC_VERSION,
            "ruleset_hash": "sha256:" + digest.hexdigest(),
            "state_transition": "mutable",
            "chance_model": "explicit_enumerable_plus_seeded_age_deal",
        }
