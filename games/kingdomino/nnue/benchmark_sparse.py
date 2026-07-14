"""Compare stateless and incremental sparse NNUE search on identical positions."""
from __future__ import annotations

import argparse
from collections import Counter
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
        "--timed-secs",
        type=float,
        default=0.0,
        help="if >0, also run deadline-safe sparse_nnue_q searches at this per-move budget",
    )
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--selective-widths", default="",
                    help="comma-separated upper-tree beam widths for timed sweeps")
    ap.add_argument("--selective-min-depth", type=int, default=4)
    ap.add_argument("--selective-root-width", type=int, default=None,
                    help="optional explicit root cap; default keeps every root action")
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

    if args.timed_secs > 0:
        widths = [None] + [
            int(x) for x in args.selective_widths.split(",") if x.strip()
        ]
        full_actions = None
        for width in widths:
            reports = []
            wall_start = time.perf_counter()
            for state in states:
                search = kr.RustSearch(
                    depth=args.max_depth,
                    enum_cap=1,
                    chance_samples=args.chance_samples,
                    seed=23,
                    eval="sparse_nnue_q",
                    nnue_path=args.model,
                )
                reports.append(
                    search.choose_action_timed(
                        state,
                        max_secs=args.timed_secs,
                        max_depth=args.max_depth,
                        aspiration_window=0.25,
                        selective_width=width,
                        selective_root_width=(args.selective_root_width
                                              if width is not None else None),
                        selective_min_depth=args.selective_min_depth,
                    )
                )
            wall = time.perf_counter() - wall_start
            depths = Counter(r.completed_depth for r in reports)
            total_nodes = sum(r.nodes for r in reports)
            final_nodes = sum(r.last_iteration_nodes for r in reports)
            label = "full" if width is None else f"selective-{width}"
            actions = [r.action for r in reports]
            if width is None:
                full_actions = actions
            print(
                f"timed {label}: {len(reports)} positions x {args.timed_secs:.3f}s, "
                f"wall {wall:.3f}s, completed depths {dict(sorted(depths.items()))}"
            )
            print(
                f"  timeouts {sum(r.timed_out for r in reports)}/{len(reports)}, "
                f"nodes {total_nodes:,} ({total_nodes / wall:,.0f} n/s), "
                f"last-complete iteration nodes {final_nodes:,}, "
                f"ordering evals {sum(r.ordering_evals for r in reports):,}, "
                f"selective pruned {sum(r.selective_pruned for r in reports):,}, "
                f"Star cutoffs {sum(r.star_cutoffs for r in reports):,}, "
                f"TT cutoffs {sum(r.tt_cutoffs for r in reports):,}, "
                f"exact extensions {sum(r.exact_extensions for r in reports):,}"
            )
            if full_actions is not None and width is not None:
                agreement = sum(a == b for a, b in zip(actions, full_actions))
                print(f"  root action agreement with full-width: {agreement}/{len(actions)}")


if __name__ == "__main__":
    main()
