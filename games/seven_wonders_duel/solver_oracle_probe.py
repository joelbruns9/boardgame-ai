"""Score the value head against PROVEN endgame values.

Every value diagnostic in this package so far has been self-referential. The
cloud6 stall was measured with search self-agreement (0.973) against net
self-agreement (0.777), and `gumbel_target_kl` compares the search's target
with the net's own prior -- both ask whether the system agrees with itself, and
neither can tell a confidently wrong value head from a correct one.

The endgame solver supplies external truth. At a solved position the value is
not an estimate: it is the game's value under optimal play by both sides. So
for the last few plies of every game we can ask the only question that matters
-- how wrong is the net, and does searching fix it?

Three quantities per solved position, all actor-relative in [-1, 1]:

* `net_root_value`  -- the network's raw opinion of the root, before search;
* `root_value`      -- the search's backed-up value after `sims` simulations;
* `solver_value`    -- the truth.

The interesting number is whether search closes the gap. If the net is far from
truth and the search is close, the value head is the problem and the search is
compensating. If both are far, the search is inheriting the net's error and no
amount of it will help -- which is the shape the cloud6 diagnosis predicts.

Reports by cards-remaining, because difficulty is not uniform: a 2-card
position is nearly a lookup and a 8-card one is a real search problem.

Usage:

    python -m games.seven_wonders_duel.solver_oracle_probe \
        --checkpoint "runs/cloud runs/cloud6_capture/cloud6/checkpoints/current_best.pt" \
        --games 60 --out oracle_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .phase_e import load_evaluator
from .rust_bridge import rust_flat_batch_adapter, rust_games_for_self_play


def _summarise(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "mean": sum(ordered) / n,
        "median": ordered[n // 2],
        "p90": ordered[min(n - 1, int(0.9 * n))],
    }


def _sign(value: float, tolerance: float = 1e-9) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def collect(
    *,
    evaluator,
    games: int,
    seed_base: int,
    max_nodes: int,
    max_cards: int,
    full_sims: tuple[int, int],
    top_k: int,
    slots: int,
    batch_cap: int,
) -> list[dict]:
    """Play `games` self-play games with the solver on, and return one row per
    position it proved."""

    import seven_wonders_rust as swr

    # Mask the policy as production would: the point is to measure the value
    # head under the conditions it will actually be trained in, not under a
    # configuration nothing runs.
    swr.set_endgame_solver(max_nodes, 60.0, max_cards, True)
    swr.set_cheap_top_k(0)
    seeds = [seed_base + index for index in range(games)]
    first_players = [index % 2 for index in range(games)]
    try:
        records, _ = swr.self_play_many_flat_net(
            adapter=rust_flat_batch_adapter(evaluator),
            games=rust_games_for_self_play(seeds, first_players),
            game_seeds=seeds,
            global_batch_cap=batch_cap,
            leaf_batch=1,
            cheap_sims_min=16,
            cheap_sims_max=24,
            full_sims_min=full_sims[0],
            full_sims_max=full_sims[1],
            # Every move a full search. Cheap moves are never solved (they emit
            # no training example), so a mixed schedule would only shrink the
            # sample without changing what it measures.
            full_search_fraction=1.0,
            top_k=top_k,
            draft_prior=0.0,
            iteration=-1,
            force=True,
            age_deal_samples=32,
            cheap_double_reveal_offsets=0,
            max_active_slots=slots,
        )
    finally:
        # Process-global, so leaving it set would silently change every later
        # generation call in this process.
        swr.set_endgame_solver(0)

    rows: list[dict] = []
    for record in records:
        moves = record["moves"]
        for position, move in enumerate(moves):
            if move["solver_value"] is None:
                continue
            rows.append(
                {
                    "regime": move["solver_regime"],
                    "legal": len(move["legal"]),
                    "optimal": sum(1 for p in move["policy_target"] if p > 0.0),
                    "sims": move["sims"],
                    "nodes": move["solver_nodes"],
                    "moves_from_end": len(moves) - position - 1,
                    "truth": float(move["solver_value"]),
                    "net": float(move["net_root_value"]),
                    "search": float(move["root_value"]),
                }
            )
    return rows


def report(rows: list[dict]) -> dict:
    """Net-vs-truth and search-vs-truth, overall and by distance from the end."""

    def block(subset: list[dict]) -> dict:
        net_error = [abs(r["net"] - r["truth"]) for r in subset]
        search_error = [abs(r["search"] - r["truth"]) for r in subset]
        return {
            "positions": len(subset),
            "net_abs_error": _summarise(net_error),
            "search_abs_error": _summarise(search_error),
            # The blunt version of the same question: does it even know who is
            # winning? A sign disagreement on a PROVEN position is not a
            # calibration issue.
            "net_sign_agrees": sum(
                _sign(r["net"]) == _sign(r["truth"]) for r in subset
            ),
            "search_sign_agrees": sum(
                _sign(r["search"]) == _sign(r["truth"]) for r in subset
            ),
            # Does searching move the estimate toward the truth at all? This is
            # the number that separates "bad value head, search compensates"
            # from "bad value head, search inherits it".
            "search_improves": sum(
                abs(r["search"] - r["truth"]) < abs(r["net"] - r["truth"])
                for r in subset
            ),
            "mean_truth": (
                sum(r["truth"] for r in subset) / len(subset) if subset else None
            ),
        }

    buckets: dict[str, list[dict]] = {}
    for row in rows:
        key = f"{min(row['moves_from_end'], 9)}"
        buckets.setdefault(key, []).append(row)

    return {
        "overall": block(rows),
        "by_regime": {
            regime: block([r for r in rows if r["regime"] == regime])
            for regime in sorted({r["regime"] for r in rows})
        },
        "by_moves_from_end": {
            key: block(buckets[key]) for key in sorted(buckets, key=int)
        },
        "tie_structure": {
            "mean_legal": (
                sum(r["legal"] for r in rows) / len(rows) if rows else None
            ),
            "mean_optimal_fraction": (
                sum(r["optimal"] / max(1, r["legal"]) for r in rows) / len(rows)
                if rows
                else None
            ),
        },
        "solver_cost": {
            "positions": len(rows),
            "total_nodes": sum(r["nodes"] for r in rows),
            "nodes": _summarise([float(r["nodes"]) for r in rows]),
        },
    }


def _print(summary: dict, games: int, *, values_interpretable: bool = True) -> None:
    overall = summary["overall"]
    count = overall["positions"]
    print(f"{count} proven positions over {games} games")
    if not count:
        return
    print(
        f"  solver cost: {summary['solver_cost']['total_nodes']:,} nodes "
        f"({summary['solver_cost']['total_nodes'] / games:,.0f}/game), "
        f"median {summary['solver_cost']['nodes']['median']:,.0f}/position"
    )
    print(
        f"  ties: {summary['tie_structure']['mean_optimal_fraction']:.0%} of "
        f"{summary['tie_structure']['mean_legal']:.1f} legal moves proven optimal"
    )
    if not values_interpretable:
        print("")
        print("  value comparison suppressed: the net is off-distribution")
        return
    print("")
    print("  |value - truth|      mean   median      p90")
    for label, key in (("net (no search)", "net_abs_error"), ("search", "search_abs_error")):
        stats = overall[key]
        print(
            f"  {label:<18} {stats['mean']:>7.3f} {stats['median']:>8.3f} "
            f"{stats['p90']:>8.3f}"
        )
    print("")
    print(
        f"  knows who is winning: net {overall['net_sign_agrees']}/{count} "
        f"({overall['net_sign_agrees'] / count:.0%}), "
        f"search {overall['search_sign_agrees']}/{count} "
        f"({overall['search_sign_agrees'] / count:.0%})"
    )
    print(
        f"  search moves toward truth: {overall['search_improves']}/{count} "
        f"({overall['search_improves'] / count:.0%})"
    )
    print("")
    print("  by moves from the end of the game:")
    print("    from_end   n   net_err  search_err  net_sign  search_sign")
    for key, block in summary["by_moves_from_end"].items():
        n = block["positions"]
        print(
            f"    {key:>8} {n:>3}   {block['net_abs_error']['mean']:>7.3f} "
            f"{block['search_abs_error']['mean']:>11.3f} "
            f"{block['net_sign_agrees'] / n:>9.0%} {block['search_sign_agrees'] / n:>12.0%}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--inference-precision", default="fp32")
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--seed-base", type=int, default=20260816)
    parser.add_argument("--max-nodes", type=int, default=5_000_000)
    parser.add_argument("--max-cards", type=int, default=8)
    parser.add_argument("--full-sims", type=int, nargs=2, default=[64, 128])
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--global-batch-cap", type=int, default=512)
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="serve a checkpoint whose encoder signature no longer matches the "
        "live encoder. The VALUE numbers are then uninterpretable -- a net fed "
        "shifted features looks wrong for a reason that has nothing to do with "
        "its training -- and are suppressed. Solver cost and tie structure are "
        "far less sensitive and are still reported.",
    )
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluator = load_evaluator(
        str(args.checkpoint),
        args.device,
        precision=args.inference_precision,
        migrate=args.migrate,
    )
    if args.migrate:
        print(
            "WARNING: --migrate. This checkpoint was trained on a different "
            "encoder, so it is being served off-distribution. Its value error "
            "against the solver measures that mismatch as much as the value "
            "head, and is NOT reported.\n"
        )
    rows = collect(
        evaluator=evaluator,
        games=args.games,
        seed_base=args.seed_base,
        max_nodes=args.max_nodes,
        max_cards=args.max_cards,
        full_sims=tuple(args.full_sims),
        top_k=args.top_k,
        slots=args.slots,
        batch_cap=args.global_batch_cap,
    )
    summary = report(rows)
    _print(summary, args.games, values_interpretable=not args.migrate)
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "games": args.games,
                    "seed_base": args.seed_base,
                    "max_cards": args.max_cards,
                    "summary": summary,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
