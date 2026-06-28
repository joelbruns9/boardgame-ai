from __future__ import annotations

import argparse
import random
import time

from games.kingdomino.endgame_solver import count_endgame_nodes, exact_endgame_value
from games.kingdomino.game import GameState, Phase


def _sample_state(seed: int, max_hidden: int) -> GameState | None:
    rng = random.Random(seed)
    state = GameState.new(seed=seed)
    while state.phase != Phase.GAME_OVER:
        if state.phase in (Phase.PLACE_AND_SELECT, Phase.FINAL_PLACEMENT):
            if len(state.deck) <= max_hidden:
                return state
        legal = state.legal_actions()
        if not legal:
            return None
        state = state.step(rng.choice(legal))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample late-game states and measure exact endgame viability."
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--max-hidden", type=int, default=6)
    parser.add_argument("--max-nodes", type=int, default=50_000,
                        help="node cap for count_endgame_nodes diagnostics")
    parser.add_argument("--max-secs", type=float, default=3.0,
                        help="wall-clock budget for the exact endgame solver")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score-scale", type=float, default=100.0)
    parser.add_argument("--margin-gain", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    args = parser.parse_args()

    print("seed,phase,hidden,current_row,legal,node_count,solved,seconds,value0")
    found = 0
    for i in range(args.samples):
        seed = args.seed + i
        state = _sample_state(seed, args.max_hidden)
        if state is None:
            continue
        found += 1
        t0 = time.perf_counter()
        nodes = count_endgame_nodes(state, max_nodes=args.max_nodes)
        value0, solved = exact_endgame_value(
            state,
            max_secs=args.max_secs,
            rng=random.Random(seed),
            score_scale=args.score_scale,
            margin_gain=args.margin_gain,
            alpha=args.alpha,
        )
        dt = time.perf_counter() - t0
        print(
            f"{seed},{state.phase.name},{len(state.deck)},{len(state.current_row)},"
            f"{len(state.legal_actions())},{nodes},{int(solved)},{dt:.6f},{value0:+.6f}"
        )

    if found == 0:
        print("No matching late-game states found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
