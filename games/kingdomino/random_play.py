from __future__ import annotations

import random

from .game import GameState, Phase


def play_random_game(seed: int = 0, max_steps: int = 500, verbose: bool = False) -> GameState:
    rng = random.Random(seed)
    state = GameState.new(seed=seed)
    steps = 0
    while state.phase != Phase.GAME_OVER and steps < max_steps:
        actions = state.legal_actions()
        if not actions:
            raise RuntimeError(f"No legal actions in phase {state.phase}; actor={state.current_actor}")
        state = state.step(rng.choice(actions))
        steps += 1
    if verbose:
        print(f"steps={steps} scores={state.scores()}")
        for i, board in enumerate(state.boards):
            print(f"\nPlayer {i}\n{board.pretty()}")
    return state


if __name__ == "__main__":
    play_random_game(seed=1, verbose=True)
