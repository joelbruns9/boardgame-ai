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
from .rust_bridge import (
    rust_flat_batch_adapter,
    rust_game_from_prefix,
    rust_games_for_self_play,
    rust_scalar_net_adapter,
)


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
    cheap_double_reveal_offsets: int,
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
        cheap_double_reveal_offsets=cheap_double_reveal_offsets,
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


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def reference_compare(
    *,
    evaluator,
    games: int,
    seed_base: int,
    search: dict,
    cheap_sims: tuple[int, int],
    cheap_double_reveal_offsets: int,
    deep_sims: int,
    samples: int,
    slots: int,
    batch_cap: int,
) -> dict:
    """Is the cheap policy target closer to a deep search than the prior is?

    KL(target || prior) measures how far the search moved the policy, not
    whether it moved it anywhere useful -- and `sigma_vector` min-max normalises
    completed Q across root actions, stretching it over ~5 logits whether or not
    the underlying Q differences mean anything. So a large cheap-move KL is
    equally consistent with an informative search and with noise amplified to
    fill the range.

    A deep search on the same position breaks the tie. Treating it as the best
    available stand-in for the truth:

        KL(deep || cheap) < KL(deep || prior)  ->  the cheap search moved the
                                                   policy toward the truth
        KL(deep || cheap) > KL(deep || prior)  ->  it moved it away, and the
                                                   half-nat is damage

    The deep search runs with an independent Gumbel seed, so agreement is not
    manufactured by sharing noise with the cheap search.
    """

    import seven_wonders_rust as swr

    swr.set_cheap_top_k(0)
    seeds = [seed_base + index for index in range(games)]
    first_players = [index % 2 for index in range(games)]

    records, _ = swr.self_play_many_flat_net(
        adapter=rust_flat_batch_adapter(evaluator),
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
        cheap_double_reveal_offsets=cheap_double_reveal_offsets,
        max_active_slots=slots,
    )

    # Positions to re-examine, strided across games and game stages rather than
    # taken from the front of the corpus -- the opening is the most stereotyped
    # part of a 7WD game and would flatter any search.
    wanted: list[dict] = []
    for record in records:
        moves = record["moves"]
        for position, move in enumerate(moves):
            if not move.get("policy_target") or not move.get("prior"):
                continue
            wanted.append(
                {
                    "seed": record["seed"],
                    "first_player": record["first_player"],
                    "prefix": [m["action"] for m in moves[:position]],
                    "cheap": move["policy_target"],
                    "prior": move["prior"],
                    "legal": move["legal"],
                    "budget": "full" if move.get("full_search") else "cheap",
                    "sims": int(move.get("sims", 0)),
                }
            )
    stride = max(1, len(wanted) // max(1, samples))
    wanted = wanted[::stride][:samples]

    # `self_play_many_flat_net` cannot be asked for a single move -- max_moves=1
    # raises rather than returning a partial record -- so the reference search
    # runs per position through the single-search entry point. That path takes a
    # scalar adapter, so it is one evaluation per simulation rather than batched;
    # at a few hundred positions that is minutes, which is the right trade for
    # not playing every position out to the end of the game at the deep budget.
    adapter = rust_scalar_net_adapter(evaluator)
    acc: dict[str, dict[str, list]] = {
        budget: {"kl_target": [], "kl_prior": [], "agree_target": [], "agree_prior": [], "sims": []}
        for budget in ("cheap", "full")
    }

    for index, item in enumerate(wanted):
        try:
            _, rust_state = rust_game_from_prefix(
                item["seed"], item["first_player"], item["prefix"]
            )
        except Exception:  # pragma: no cover - a position we cannot replay
            continue
        result = rust_state.closed_search_resumable_net(
            adapter,
            deep_sims,
            search["top_k"],
            (item["seed"] ^ 0x5DEE) + index,
            force=True,
        )
        target = list(result[4])
        # The rebuilt root must present the same legal set, or we would be
        # comparing distributions over different action spaces.
        if len(target) != len(item["cheap"]):
            continue
        from_target = _kl(target, item["cheap"])
        from_prior = _kl(target, item["prior"])
        if math.isnan(from_target) or math.isnan(from_prior):
            continue
        bucket = acc[item["budget"]]
        bucket["kl_target"].append(from_target)
        bucket["kl_prior"].append(from_prior)
        bucket["sims"].append(item["sims"])
        deep_best = _argmax(target)
        bucket["agree_target"].append(int(_argmax(item["cheap"]) == deep_best))
        bucket["agree_prior"].append(int(_argmax(item["prior"]) == deep_best))

    def _mean(values):
        return sum(values) / len(values) if values else None

    out = {"deep_sims": deep_sims, "positions_sampled": len(wanted)}
    for budget, bucket in acc.items():
        out[budget] = {
            "positions_compared": len(bucket["kl_target"]),
            "mean_sims": _mean(bucket["sims"]),
            "kl_deep_from_target": _mean(bucket["kl_target"]),
            "kl_deep_from_prior": _mean(bucket["kl_prior"]),
            "agree_target_with_deep": _mean(bucket["agree_target"]),
            "agree_prior_with_deep": _mean(bucket["agree_prior"]),
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
    parser.add_argument(
        "--cheap-double-reveal-offsets",
        type=int,
        default=0,
        help="fixed-support cap X on pure double card-reveal edges, CHEAP "
        "moves only; 0 is the exhaustive shipped behaviour",
    )
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--global-batch-cap", type=int, default=512)
    parser.add_argument(
        "--deep-sims",
        type=int,
        default=0,
        help="run the reference comparison instead of the width sweep: re-search sampled cheap-move positions at this budget and ask whether the cheap target sits closer to it than the prior does. 0 disables.",
    )
    parser.add_argument("--deep-samples", type=int, default=200)
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

    if args.deep_sims:
        row = reference_compare(
            evaluator=evaluator,
            games=args.games,
            seed_base=args.seed_base,
            search=search,
            cheap_sims=tuple(args.cheap_sims),
            cheap_double_reveal_offsets=args.cheap_double_reveal_offsets,
            deep_sims=args.deep_sims,
            samples=args.deep_samples,
            slots=args.slots,
            batch_cap=args.global_batch_cap,
        )
        print(f"reference search: {args.deep_sims} sims\n")
        print(
            f"{'budget':>8} {'sims':>6} {'n':>5} {'agree(target,deep)':>19} "
            f"{'agree(prior,deep)':>18} {'lift':>7}"
        )
        for budget in ("cheap", "full"):
            cell = row[budget]
            if not cell["positions_compared"]:
                continue
            lift = cell["agree_target_with_deep"] - cell["agree_prior_with_deep"]
            print(
                f"{budget:>8} {cell['mean_sims']:>6.0f} "
                f"{cell['positions_compared']:>5} "
                f"{cell['agree_target_with_deep']:>19.3f} "
                f"{cell['agree_prior_with_deep']:>18.3f} {lift:>+7.3f}"
            )
        print(
            "\n  agree(target,deep) is the fraction of positions where the "
            "training target's\n  best move matches a "
            f"{args.deep_sims}-simulation reference. 1 - that value is the\n"
            "  label noise the network is being asked to fit."
        )
        if args.out:
            args.out.write_text(json.dumps(row, indent=2), encoding="utf-8")
            print(f"wrote {args.out}")
        return 0

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
            cheap_double_reveal_offsets=args.cheap_double_reveal_offsets,
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
