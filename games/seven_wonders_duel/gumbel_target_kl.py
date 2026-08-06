#!/usr/bin/env python3
"""How much does the Gumbel search actually move the policy?

Gumbel AlphaZero's guarantee is that the improved policy is never *worse* than
the prior, at any simulation budget. It says nothing about how much better.
That distinction is the whole question behind the cloud2/cloud3 stall: both runs
stopped learning around 15,000-30,000 games, `policy_top1` pinned at ~0.64, and
the diagnosis on the table is that a 20-simulation search over ~10 root
candidates cannot re-rank the prior enough to teach the network anything new.

This measures it instead of arguing it. For each generated row we have the prior
and the improved policy over the same legal set, so the improvement the search
produced is exactly

    KL(policy_target || prior)

in nats. Zero means the search returned its own input and the row carries no
training signal beyond what the network already believed.

The sweep varies cheap-move root width (`top_k`) and cheap-move budget, holding
everything else at production values, and reports KL split by cheap vs full
moves. Read it like this:

  * cheap KL near zero while full KL is materially higher
        -> the budget/width ratio is the binding constraint, and the cheap-move
           `top_k` column says which width fixes it
  * cheap and full KL comparable and both non-trivial
        -> search is contributing at both budgets, the stall is elsewhere
           (model capacity or the encoder), and no amount of width helps

`top1_agree` is the fraction of rows where the search's argmax equals the
prior's argmax -- a coarser, more legible companion to KL. If it is ~1.0 the
search never changed its mind about the best move even when it shifted mass.

Everything here runs on a laptop against a saved checkpoint. No box time.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .phase_e import load_evaluator
from .rust_bridge import rust_flat_batch_adapter, rust_games_for_self_play


def _kl(target: list[float], prior: list[float]) -> float:
    """KL(target || prior) in nats, over one row's legal set.

    Both arrive renormalised over the same legal actions, so no alignment or
    masking is needed. Zero-probability prior entries would be a bug rather than
    a case to handle -- the network's softmax cannot produce them -- but the
    guard keeps a malformed row from poisoning a whole sweep cell with `inf`.
    """

    total = 0.0
    for t, p in zip(target, prior):
        if t <= 0.0:
            continue
        if p <= 0.0:
            return float("nan")
        total += t * math.log(t / p)
    return total


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


def measure(
    *,
    evaluator,
    games: int,
    seed_base: int,
    search: dict,
    cheap_top_k: int,
    cheap_sims: tuple[int, int],
    slots: int,
    batch_cap: int,
) -> dict:
    import seven_wonders_rust as swr

    # Cheap width is process-wide (it is a self-play-only knob), so it must be
    # set before the call rather than passed into it.
    swr.set_cheap_top_k(cheap_top_k)

    adapter = rust_flat_batch_adapter(evaluator)
    seeds = [seed_base + index for index in range(games)]
    first_players = [index % 2 for index in range(games)]

    records, _ = swr.self_play_many_flat_net(
        adapter=adapter,
        games=rust_games_for_self_play(seeds, first_players),
        game_seeds=seeds,
        global_batch_cap=batch_cap,
        leaf_batch=1,
        cheap_sims_min=cheap_sims[0],
        cheap_sims_max=cheap_sims[1],
        full_sims_min=search["full_sims_min"],
        full_sims_max=search["full_sims_max"],
        full_search_fraction=search["full_search_fraction"],
        top_k=search["top_k"],
        draft_prior=0.0,
        iteration=-1,
        force=True,
        age_deal_samples=search["age_deal_samples"],
        cheap_double_reveal_offsets=0,
        max_active_slots=slots,
    )

    buckets: dict[str, list[float]] = {"cheap": [], "full": []}
    agree: dict[str, list[int]] = {"cheap": [], "full": []}
    sims_seen: dict[str, list[int]] = {"cheap": [], "full": []}
    # Move-index thirds, to see whether the improvement dies out as the game
    # converges -- which is where the temperature floor also stops mattering.
    thirds: dict[int, list[float]] = {0: [], 1: [], 2: []}

    for record in records:
        moves = record["moves"] if isinstance(record, dict) else record.moves
        length = max(1, len(moves))
        for move in moves:
            target = move.get("policy_target")
            prior = move.get("prior")
            if not target or not prior or len(target) != len(prior):
                continue  # bot rows carry no search
            value = _kl(target, prior)
            if math.isnan(value):
                continue
            key = "full" if move.get("full_search") else "cheap"
            buckets[key].append(value)
            sims_seen[key].append(int(move.get("sims", 0)))
            agree[key].append(
                int(
                    max(range(len(target)), key=target.__getitem__)
                    == max(range(len(prior)), key=prior.__getitem__)
                )
            )
            thirds[min(2, (3 * int(move["i"])) // length)].append(value)

    out = {
        "cheap_top_k": cheap_top_k,
        "cheap_sims": list(cheap_sims),
        "games": games,
    }
    for key in ("cheap", "full"):
        out[key] = _summarise(buckets[key])
        out[key]["top1_agree"] = (
            sum(agree[key]) / len(agree[key]) if agree[key] else None
        )
        out[key]["mean_sims"] = (
            sum(sims_seen[key]) / len(sims_seen[key]) if sims_seen[key] else None
        )
    out["by_game_third"] = {
        str(third): _summarise(values) for third, values in thirds.items()
    }
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--inference-precision",
        default="fp32",
        help="must match the training run's --precision; the evaluator defaults "
        "to fp32 while production runs bf16, and profiling the wrong one has "
        "already produced a wrong answer once",
    )
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--seed-base", type=int, default=20260806)
    parser.add_argument(
        "--cheap-top-k",
        type=int,
        nargs="+",
        default=[16, 8, 4, 2],
        help="cheap-move root widths to sweep. 16 is the production value, which "
        "7WD's ~10 legal actions clip to; at 20 sims that leaves every candidate "
        "with a single simulation and therefore no opponent reply in its Q.",
    )
    parser.add_argument(
        "--cheap-sims",
        type=int,
        nargs=2,
        default=[16, 24],
        help="cheap-move budget range, held fixed across the width sweep so the "
        "width effect is not confounded with a budget change",
    )
    parser.add_argument("--full-sims", type=int, nargs=2, default=[64, 128])
    parser.add_argument("--full-search-fraction", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--age-deal-samples", type=int, default=32)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--global-batch-cap", type=int, default=512)
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluator = load_evaluator(
        str(args.checkpoint), args.device, precision=args.inference_precision
    )
    evaluator.max_batch = args.global_batch_cap

    search = {
        "full_sims_min": args.full_sims[0],
        "full_sims_max": args.full_sims[1],
        "full_search_fraction": args.full_search_fraction,
        "top_k": args.top_k,
        "age_deal_samples": args.age_deal_samples,
    }

    results = []
    print(
        f"{'cheap_k':>8} {'c_sims':>7} {'cheap_KL':>9} {'cheap_ag':>9} "
        f"{'full_KL':>9} {'full_ag':>8}   (KL in nats, mean)"
    )
    for width in args.cheap_top_k:
        row = measure(
            evaluator=evaluator,
            games=args.games,
            seed_base=args.seed_base,
            search=search,
            cheap_top_k=width,
            cheap_sims=tuple(args.cheap_sims),
            slots=args.slots,
            batch_cap=args.global_batch_cap,
        )
        results.append(row)
        cheap, full = row["cheap"], row["full"]
        print(
            f"{width:>8} {row['cheap_sims'][0]}-{row['cheap_sims'][1]:<5} "
            f"{cheap['mean']:>9.4f} {cheap['top1_agree']:>9.3f} "
            f"{full['mean']:>9.4f} {full['top1_agree']:>8.3f}"
        )

    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
