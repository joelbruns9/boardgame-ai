"""Compare stateless and incremental sparse NNUE search on identical positions."""
from __future__ import annotations

import argparse
import random
import time

import kingdomino_rust as kr

from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.game import GameState, Phase


def positions(count: int):
    out = []
    for seed in range(1000):
        state = GameState.new(seed=seed)
        rng = random.Random(seed * 17 + 5)
        ply = 0
        while state.phase != Phase.GAME_OVER:
            if (
                state.phase == Phase.PLACE_AND_SELECT
                and 12 <= ply <= 38
                and len(state.legal_actions()) >= 2
                and ply % 7 == 0
            ):
                rust = _rust_state_from_python(state)
                if rust is not None:
                    out.append(rust)
                    if len(out) >= count:
                        return out
            state = state.step(rng.choice(state.legal_actions()))
            ply += 1
    raise RuntimeError(f"found only {len(out)} benchmark positions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/kingdomino/nnue_data/sparse_v3_pilot.knnue")
    ap.add_argument("--depths", default="2,3")
    ap.add_argument("--positions", type=int, default=4)
    ap.add_argument("--chance-samples", type=int, default=8)
    ap.add_argument(
        "--evals",
        default="sparse_nnue_ref,sparse_nnue,sparse_nnue_q",
        help="comma-separated RustSearch evaluator names",
    )
    args = ap.parse_args()
    states = positions(args.positions)
    evals = [name.strip() for name in args.evals.split(",") if name.strip()]
    for depth in [int(x) for x in args.depths.split(",")]:
        result = {}
        actions = {}
        for name in evals:
            search = kr.RustSearch(
                depth=depth,
                enum_cap=1,
                chance_samples=args.chance_samples,
                seed=23,
                eval=name,
                nnue_path=args.model,
            )
            total_nodes = 0
            chosen = []
            start = time.perf_counter()
            for i, state in enumerate(states):
                chosen.append(search.choose_action(state, i))
                total_nodes += search.nodes
            elapsed = time.perf_counter() - start
            result[name] = (elapsed, total_nodes, total_nodes / elapsed)
            actions[name] = chosen
        if {"sparse_nnue_ref", "sparse_nnue"} <= actions.keys():
            assert actions["sparse_nnue_ref"] == actions["sparse_nnue"]
            assert result["sparse_nnue_ref"][1] == result["sparse_nnue"][1]
        print(f"depth {depth}:")
        for name in evals:
            elapsed, nodes, nps = result[name]
            print(f"  {name:18s} {elapsed:8.3f}s {nodes:10,d} nodes {nps:10,.0f} n/s")
        if {"sparse_nnue", "sparse_nnue_q"} <= actions.keys():
            agreement = sum(
                a == b
                for a, b in zip(actions["sparse_nnue"], actions["sparse_nnue_q"])
            )
            print(
                f"  quantized action agreement {agreement}/{len(states)}; "
                f"speedup {result['sparse_nnue'][0] / result['sparse_nnue_q'][0]:.2f}x"
            )


if __name__ == "__main__":
    main()
