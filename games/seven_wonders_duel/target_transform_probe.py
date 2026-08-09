#!/usr/bin/env python3
"""Which target transform should we record, and what `c_scale` fits a budget?

Supersedes `target_entropy_probe.py`, which had three defects this fixes:

1. **It swept `c_scale` by re-running the search.** Sigma feeds the sequential
   halving key (`search.py:594`, `tree.rs:625`), so those arms selected different
   survivors and produced different Q -- they were not one search under different
   label transforms, which is what the comparison assumed. Here the search runs
   ONCE per position and every transform is applied offline to the same
   `(prior, completed_q, visits)` vector. No halving confound, and arbitrarily
   many transforms for one generation pass.
2. **It aggregated rows away**, so nothing could be re-cut by action count or
   re-tested. Rows persist, with `game_id`, so results can be re-analysed and
   confidence intervals can cluster on the game rather than pretending positions
   from one game are independent.
3. **It scored sharpness with entropy in nats**, which conflates a small legal
   set with a confident target: 7WD is forced ~5% of the time and has <=2 legal
   moves ~21% of the time. Entropy is normalised by `log(K)` here and forced
   decisions are excluded.

THE PRESCRIPTIVE METRIC IS CALIBRATION, and it is deliberately free of the
circularity that would follow from scoring a target against another
sigma-transformed distribution: a reference built with some `c_scale` would
simply prefer candidates sharing it. The reference contributes only quantities
no transform touches -- its completed Q, and the argmax thereof.

  * calibration: bin by `target_max`, compare stated confidence with how often
    the target's argmax matches the reference's. Confidence above accuracy means
    the transform is too sharp, and `c_scale` should come down.
  * regret: `max(ref_q) - ref_q[argmax(target)]`, in game-value units, plus
    expected regret under the full target distribution. Expected regret is the
    metric that rewards spreading mass over genuinely equal moves while
    punishing spreading it over unequal ones -- entropy cannot tell those apart.

Two-stage by design: `collect` searches and writes rows; `analyse` reads rows and
sweeps transforms. Re-running the sweep costs no GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from .phase_e import load_evaluator
from .rust_bridge import (
    rust_flat_batch_adapter,
    rust_game_from_prefix,
    rust_games_for_self_play,
    rust_scalar_net_adapter,
)


# ---------------------------------------------------------------- transforms


def sigma_minmax(completed_q, max_visits, c_visit, c_scale, tau=0.0):
    """Production transform: min-max to [0, 1], scaled by (c_visit + n) * c_scale.

    `tau` is the span floor. At 0 this is exactly `_sigma`. Above 0 a raw span
    below `tau` is no longer stretched to fill the whole range, so tiny Q
    differences stop being magnified into confident logits while genuinely large
    spreads are untouched.
    """

    low = min(completed_q)
    raw_span = max(completed_q) - low
    span = max(raw_span, tau, 1e-8)
    scale = (c_visit + max_visits) * c_scale
    return [scale * (q - low) / span for q in completed_q]


def target_from_sigma(prior, sigma):
    """softmax(log_prior + sigma), matching `_gumbel_root`'s target exactly."""

    logits = [
        math.log(max(p, 1e-12)) + s for p, s in zip(prior, sigma)
    ]
    peak = max(logits)
    weights = [math.exp(v - peak) for v in logits]
    total = sum(weights)
    return [w / total for w in weights]


def parse_transform(spec: str) -> dict:
    """`minmax:0.1` or `spanfloor:0.1:0.15` -> a named transform."""

    parts = spec.split(":")
    kind = parts[0]
    if kind == "minmax":
        return {"name": spec, "c_scale": float(parts[1]), "tau": 0.0}
    if kind == "spanfloor":
        return {"name": spec, "c_scale": float(parts[1]), "tau": float(parts[2])}
    raise ValueError(f"unknown transform {spec!r}; expected minmax:C or spanfloor:C:TAU")


# ------------------------------------------------------------------- collect


def collect(args) -> dict:
    import seven_wonders_rust as swr

    evaluator = load_evaluator(
        args.checkpoint, args.device, precision=args.inference_precision
    )
    swr.set_cheap_top_k(0)
    seeds = [args.seed_base + index for index in range(args.games)]
    first_players = [index % 2 for index in range(args.games)]

    records, _ = swr.self_play_many_flat_net(
        adapter=rust_flat_batch_adapter(evaluator),
        games=rust_games_for_self_play(seeds, first_players),
        game_seeds=seeds,
        global_batch_cap=args.global_batch_cap,
        leaf_batch=1,
        cheap_sims_min=args.cheap_sims[0],
        cheap_sims_max=args.cheap_sims[1],
        full_sims_min=args.full_sims[0],
        full_sims_max=args.full_sims[1],
        full_search_fraction=args.full_search_fraction,
        top_k=args.top_k,
        draft_prior=0.0,
        iteration=-1,
        force=True,
        age_deal_samples=args.age_deal_samples,
        cheap_double_reveal_offsets=0,
        max_active_slots=args.slots,
    )

    # Strided across games and stages: the opening is the most stereotyped part
    # of a 7WD game and would flatter any measurement of move equivalence.
    wanted: list[dict] = []
    for record in records:
        moves = record["moves"]
        for position, move in enumerate(moves):
            if not move.get("full_search") or not move.get("policy_target"):
                continue
            wanted.append(
                {
                    "game_id": int(record["seed"]),
                    "first_player": record["first_player"],
                    "prefix": [m["action"] for m in moves[:position]],
                    # `closed_search_batched_net` does not return the prior, but
                    # the prior is the net's policy at this state -- identical at
                    # re-search time, so the recorded one is the same vector.
                    "prior": list(move["prior"]),
                }
            )
    stride = max(1, len(wanted) // max(1, args.samples))
    wanted = wanted[::stride][: args.samples]
    print(f"sampled {len(wanted)} full-search positions from {args.games} games")

    adapter = rust_scalar_net_adapter(evaluator)
    rows: list[dict] = []
    for index, item in enumerate(wanted):
        try:
            _, rust_state = rust_game_from_prefix(
                item["game_id"], item["first_player"], item["prefix"]
            )
        except Exception:  # pragma: no cover - a position we cannot replay
            continue
        candidate = rust_state.closed_search_batched_net(
            adapter,
            1,
            args.candidate_sims,
            args.top_k,
            (item["game_id"] ^ 0x5DEE) + index,
            1.5,
            args.c_visit,
            args.search_c_scale,
            True,
            False,
            False,
        )
        try:
            _, reference_state = rust_game_from_prefix(
                item["game_id"], item["first_player"], item["prefix"]
            )
        except Exception:  # pragma: no cover
            continue
        # Independent seed: the reference must not manufacture agreement by
        # sharing the candidate's Gumbel draw.
        reference = reference_state.closed_search_batched_net(
            adapter,
            1,
            args.reference_sims,
            args.top_k,
            (item["game_id"] ^ 0x9F17) + index,
            1.5,
            args.c_visit,
            args.search_c_scale,
            True,
            False,
            False,
        )
        prior = item["prior"]
        completed_q = list(candidate[-2])
        reference_q = list(reference[-2])
        visits = list(candidate[3])
        # A length mismatch means the rebuilt root presented a different legal
        # set, so the vectors are not over the same action space.
        if not completed_q or len(reference_q) != len(completed_q):
            continue
        if len(prior) != len(completed_q):
            continue
        rows.append(
            {
                "game_id": item["game_id"],
                "legal": len(completed_q),
                "prior": prior,
                "completed_q": completed_q,
                "reference_q": reference_q,
                "max_visits": max(visits) if visits else 0,
            }
        )
        if (index + 1) % 25 == 0:
            print(f"  searched {index + 1}/{len(wanted)}")

    payload = {
        "candidate_sims": args.candidate_sims,
        "reference_sims": args.reference_sims,
        "search_c_scale": args.search_c_scale,
        "c_visit": args.c_visit,
        "rows": rows,
    }
    args.rows.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.rows}")
    return payload


# ------------------------------------------------------------------- analyse


def _bootstrap_by_game(rows, statistic, iterations, rng):
    """Percentile CI resampling GAMES, not positions.

    Positions from one game share a trajectory and a network state; treating
    them as independent would understate the interval.
    """

    by_game: dict[int, list] = {}
    for row in rows:
        by_game.setdefault(row["game_id"], []).append(row)
    games = list(by_game)
    if len(games) < 2:
        return None, None
    samples = []
    for _ in range(iterations):
        drawn = []
        for _ in range(len(games)):
            drawn.extend(by_game[rng.choice(games)])
        value = statistic(drawn)
        if value is not None:
            samples.append(value)
    if not samples:
        return None, None
    samples.sort()
    return (
        samples[int(0.025 * len(samples))],
        samples[min(len(samples) - 1, int(0.975 * len(samples)))],
    )


def evaluate_transform(rows, transform, c_visit, bootstrap, rng):
    scored = []
    for row in rows:
        # Forced decisions carry no information about sharpness: entropy is 0
        # and the argmax is trivially correct.
        if row["legal"] < 2:
            continue
        sigma = sigma_minmax(
            row["completed_q"],
            row["max_visits"],
            c_visit,
            transform["c_scale"],
            transform["tau"],
        )
        target = target_from_sigma(row["prior"], sigma)
        reference_q = row["reference_q"]
        best_reference = max(range(len(reference_q)), key=lambda i: reference_q[i])
        chosen = max(range(len(target)), key=lambda i: target[i])
        best_value = reference_q[best_reference]
        entropy = -sum(p * math.log(p) for p in target if p > 0)
        scored.append(
            {
                "game_id": row["game_id"],
                "legal": row["legal"],
                "target_max": max(target),
                "agree": int(chosen == best_reference),
                "regret": best_value - reference_q[chosen],
                "expected_regret": sum(
                    p * (best_value - q) for p, q in zip(target, reference_q)
                ),
                "norm_entropy": entropy / math.log(row["legal"]),
            }
        )
    if not scored:
        return None

    def _confidence(subset):
        return statistics.mean(r["target_max"] for r in subset) if subset else None

    def _accuracy(subset):
        return statistics.mean(r["agree"] for r in subset) if subset else None

    def _gap(subset):
        if not subset:
            return None
        return _confidence(subset) - _accuracy(subset)

    summary = {
        "name": transform["name"],
        "n": len(scored),
        "confidence": _confidence(scored),
        "accuracy": _accuracy(scored),
        "calibration_gap": _gap(scored),
        "mean_regret": statistics.mean(r["regret"] for r in scored),
        "mean_expected_regret": statistics.mean(
            r["expected_regret"] for r in scored
        ),
        "mean_norm_entropy": statistics.mean(r["norm_entropy"] for r in scored),
    }
    if bootstrap:
        low, high = _bootstrap_by_game(scored, _gap, bootstrap, rng)
        summary["calibration_gap_ci"] = [low, high]
        low, high = _bootstrap_by_game(
            scored,
            lambda s: statistics.mean(r["expected_regret"] for r in s) if s else None,
            bootstrap,
            rng,
        )
        summary["expected_regret_ci"] = [low, high]

    # Calibration curve: does stated confidence track actual accuracy?
    buckets = []
    edges = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]
    for low_edge, high_edge in edges:
        subset = [r for r in scored if low_edge <= r["target_max"] < high_edge]
        if len(subset) < 5:
            continue
        buckets.append(
            {
                "range": f"{low_edge:.2f}-{high_edge:.2f}",
                "n": len(subset),
                "confidence": _confidence(subset),
                "accuracy": _accuracy(subset),
            }
        )
    summary["calibration_curve"] = buckets
    return summary


def analyse(args) -> int:
    payload = json.loads(args.rows.read_text(encoding="utf-8"))
    rows = payload["rows"]
    c_visit = payload.get("c_visit", 50.0)
    rng = random.Random(20260807)
    transforms = [parse_transform(spec) for spec in args.transform]

    print(
        f"\ncandidate_sims={payload['candidate_sims']}  "
        f"reference_sims={payload['reference_sims']}  rows={len(rows)}"
        f"  (forced decisions excluded from all metrics)"
    )
    print(
        "\n  transform            n   conf   acc    gap    E[regret]  regret  nH"
    )
    results = []
    for transform in transforms:
        summary = evaluate_transform(rows, transform, c_visit, args.bootstrap, rng)
        if not summary:
            continue
        results.append(summary)
        gap = summary["calibration_gap"]
        ci = summary.get("calibration_gap_ci")
        ci_text = (
            f" [{ci[0]:+.3f},{ci[1]:+.3f}]" if ci and ci[0] is not None else ""
        )
        print(
            f"  {summary['name']:<18} {summary['n']:>4}"
            f"  {summary['confidence']:.3f}  {summary['accuracy']:.3f}"
            f"  {gap:+.3f}{ci_text}"
            f"  {summary['mean_expected_regret']:.4f}"
            f"  {summary['mean_regret']:.4f}"
            f"  {summary['mean_norm_entropy']:.3f}"
        )

    if results:
        best = min(results, key=lambda r: abs(r["calibration_gap"]))
        print(
            f"\n  best calibrated: {best['name']}"
            f"  (confidence {best['confidence']:.3f} vs accuracy"
            f" {best['accuracy']:.3f}, gap {best['calibration_gap']:+.3f})"
        )
        lowest = min(results, key=lambda r: r["mean_expected_regret"])
        print(f"  lowest expected regret: {lowest['name']}")
        print("\n  calibration curve for the best-calibrated transform:")
        for bucket in best["calibration_curve"]:
            print(
                f"    target_max {bucket['range']}  n={bucket['n']:>4}"
                f"  says {bucket['confidence']:.3f}"
                f"  is right {bucket['accuracy']:.3f}"
            )

    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    collector = sub.add_parser("collect", help="search positions, persist rows")
    collector.add_argument("--checkpoint", type=Path, required=True)
    collector.add_argument("--rows", type=Path, required=True)
    collector.add_argument("--device", default="cpu")
    collector.add_argument("--inference-precision", default="fp32")
    collector.add_argument("--games", type=int, default=60)
    collector.add_argument("--samples", type=int, default=200)
    collector.add_argument("--seed-base", type=int, default=20260807)
    collector.add_argument(
        "--candidate-sims",
        type=int,
        default=96,
        help="the budget you are choosing a c_scale FOR",
    )
    collector.add_argument("--reference-sims", type=int, default=512)
    collector.add_argument(
        "--search-c-scale",
        type=float,
        default=0.1,
        help="c_scale used INSIDE the search (it feeds sequential halving). "
        "Transforms are swept offline; this only sets how the Q was produced.",
    )
    collector.add_argument("--c-visit", type=float, default=50.0)
    collector.add_argument("--cheap-sims", type=int, nargs=2, default=[16, 24])
    collector.add_argument("--full-sims", type=int, nargs=2, default=[64, 128])
    collector.add_argument("--full-search-fraction", type=float, default=0.25)
    collector.add_argument("--top-k", type=int, default=16)
    collector.add_argument("--age-deal-samples", type=int, default=32)
    collector.add_argument("--slots", type=int, default=32)
    collector.add_argument("--global-batch-cap", type=int, default=512)

    analyser = sub.add_parser("analyse", help="sweep transforms over saved rows")
    analyser.add_argument("--rows", type=Path, required=True)
    analyser.add_argument(
        "--transform",
        action="append",
        default=None,
        help="repeatable: minmax:C_SCALE or spanfloor:C_SCALE:TAU",
    )
    analyser.add_argument("--bootstrap", type=int, default=400)
    analyser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "collect":
        collect(args)
        return 0
    if not args.transform:
        args.transform = [
            "minmax:0.1",
            "minmax:0.07",
            "minmax:0.05",
            "minmax:0.03",
            "minmax:0.02",
            "spanfloor:0.1:0.10",
            "spanfloor:0.1:0.25",
        ]
    return analyse(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
