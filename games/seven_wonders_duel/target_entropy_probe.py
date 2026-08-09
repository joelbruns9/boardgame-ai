#!/usr/bin/env python3
"""Is the Gumbel policy target sharp because the moves differ, or by construction?

`_sigma` min-max normalises completed Q across the root's actions and multiplies
by `(c_visit + max_visits) * c_scale`.  Both halves of that are suspicious at
budget:

  * min-max always maps best -> 1 and worst -> 0, so the target's SHAPE carries
    no information about how much the actions actually differ.  Four moves worth
    0.30/0.29/0.28/0.27 produce the same normalised spread as 0.9/0.6/0.3/0.0.
  * the scale grows with visits, so the same relative gaps turn into ever larger
    logit gaps: ~9.6 nats at 96 sims, ~43 at 1024.

Together those predict that at high budget the target becomes near one-hot even
in positions where every move is equivalent -- which would train the network to
be confidently arbitrary, degrade its calibration, and feed back into a search
that uses the prior for both Gumbel top-k and interior PUCT.

This measures it on identical positions across arms.  For each sampled position
we re-run the real searcher and record, from the SAME call, the raw completed Q
(pre-normalisation) and the policy target it produced.  The diagnostic is the
join: **target entropy binned by raw Q spread**.  Read it like this:

  * entropy falls as raw spread rises, and stays high when the spread is small
        -> the target is sharp because the moves differ.  Working as intended.
  * entropy is uniformly low regardless of raw spread
        -> min-max is manufacturing confidence, and more sims makes it worse.

Arms let you price the fix: the same positions at a larger budget, and at a
smaller `c_scale`, which is the knob that would hold target entropy roughly
constant as sims rise.

Laptop-only, one checkpoint, no box time.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from .phase_e import load_evaluator
from .rust_bridge import (
    rust_flat_batch_adapter,
    rust_game_from_prefix,
    rust_games_for_self_play,
    rust_scalar_net_adapter,
)


def _entropy(distribution: list[float]) -> float:
    """Shannon entropy in nats, over whatever support the caller passes."""

    total = 0.0
    for probability in distribution:
        if probability > 0:
            total -= probability * math.log(probability)
    return total


def _spread(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def collect_positions(
    *,
    evaluator,
    games: int,
    seed_base: int,
    cheap_sims: tuple[int, int],
    full_sims: tuple[int, int],
    full_search_fraction: float,
    top_k: int,
    age_deal_samples: int,
    slots: int,
    batch_cap: int,
    samples: int,
) -> list[dict]:
    """Play production-shaped self-play, then stride-sample full-search rows.

    Strided rather than taken from the front: the opening is the most
    stereotyped part of a 7WD game and would flatter any measurement of how much
    the moves differ.
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
        full_sims_min=full_sims[0],
        full_sims_max=full_sims[1],
        full_search_fraction=full_search_fraction,
        top_k=top_k,
        draft_prior=0.0,
        iteration=-1,
        force=True,
        age_deal_samples=age_deal_samples,
        cheap_double_reveal_offsets=0,
        max_active_slots=slots,
    )

    wanted: list[dict] = []
    for record in records:
        moves = record["moves"]
        for position, move in enumerate(moves):
            # Only full-search moves become training examples, so only their
            # targets can damage the network.
            if not move.get("full_search") or not move.get("policy_target"):
                continue
            wanted.append(
                {
                    "seed": record["seed"],
                    "first_player": record["first_player"],
                    "prefix": [m["action"] for m in moves[:position]],
                    "legal": len(move["legal"]),
                }
            )
    stride = max(1, len(wanted) // max(1, samples))
    return wanted[::stride][:samples]


def run_arm(
    *,
    evaluator,
    positions: list[dict],
    sims: int,
    c_scale: float,
    c_visit: float,
    top_k: int,
    leaf_batch: int,
) -> list[dict]:
    """Re-search each position, taking target and raw completed Q from one call.

    Same call for both so they cannot drift apart: the whole question is whether
    THIS target came from THOSE Q values.
    """

    # `closed_search_batched_net` crosses the boundary through the SCALAR
    # adapter (one call per leaf), not the flat-batch family.
    adapter = rust_scalar_net_adapter(evaluator)
    rows: list[dict] = []
    for index, item in enumerate(positions):
        try:
            _, rust_state = rust_game_from_prefix(
                item["seed"], item["first_player"], item["prefix"]
            )
        except Exception:  # pragma: no cover - a position we cannot replay
            continue
        result = rust_state.closed_search_batched_net(
            adapter,
            leaf_batch,
            sims,
            top_k,
            (item["seed"] ^ 0x5DEE) + index,
            1.5,  # c_puct, production default
            c_visit,
            c_scale,
            True,  # force_expand_root_chance, as the rust generator does
            False,  # conflict_free_waves
            False,  # round_robin_candidates
        )
        visits = list(result[3])
        target = list(result[4])
        completed_q = list(result[-2])
        if not target or len(completed_q) != len(target):
            continue
        rows.append(
            {
                "legal": len(target),
                "max_visits": max(visits) if visits else 0,
                "q_spread": _spread(completed_q),
                "target_entropy": _entropy(target),
                "target_max": max(target),
                # What `_sigma` should have applied, for a consistency check
                # against the entropy we actually observe.
                "sigma_span": (c_visit + (max(visits) if visits else 0)) * c_scale,
            }
        )
    return rows


def _report(name: str, rows: list[dict]) -> dict:
    if not rows:
        print(f"{name}: no positions")
        return {}
    entropies = [row["target_entropy"] for row in rows]
    spreads = [row["q_spread"] for row in rows]
    summary = {
        "n": len(rows),
        "mean_target_entropy": statistics.mean(entropies),
        "median_target_entropy": statistics.median(entropies),
        "mean_target_max": statistics.mean(row["target_max"] for row in rows),
        "mean_q_spread": statistics.mean(spreads),
        "mean_sigma_span": statistics.mean(row["sigma_span"] for row in rows),
        "mean_max_visits": statistics.mean(row["max_visits"] for row in rows),
        "frac_near_one_hot": sum(row["target_max"] > 0.95 for row in rows) / len(rows),
    }
    print(f"\n{name}")
    print(
        f"  n={summary['n']}  mean_max_visits={summary['mean_max_visits']:.0f}"
        f"  sigma_span={summary['mean_sigma_span']:.1f} nats"
    )
    print(
        f"  target entropy: mean {summary['mean_target_entropy']:.3f} nats"
        f"  median {summary['median_target_entropy']:.3f}"
    )
    print(
        f"  target argmax mass: mean {summary['mean_target_max']:.3f}"
        f"   P(>0.95) = {summary['frac_near_one_hot']:.3f}"
    )

    # The join that answers the question: does sharpness track how much the
    # moves actually differ, or is it flat across the raw-spread range?
    ordered = sorted(rows, key=lambda row: row["q_spread"])
    quartiles = []
    size = max(1, len(ordered) // 4)
    print("     raw Q spread quartile -> mean target entropy")
    for q in range(4):
        chunk = ordered[q * size : (q + 1) * size] if q < 3 else ordered[3 * size :]
        if not chunk:
            continue
        bucket = {
            "q_spread_mean": statistics.mean(row["q_spread"] for row in chunk),
            "target_entropy_mean": statistics.mean(
                row["target_entropy"] for row in chunk
            ),
            "n": len(chunk),
        }
        quartiles.append(bucket)
        print(
            f"       Q{q + 1}  spread {bucket['q_spread_mean']:.4f}"
            f"  ->  entropy {bucket['target_entropy_mean']:.3f}  (n={bucket['n']})"
        )
    summary["quartiles"] = quartiles
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--inference-precision",
        default="fp32",
        help="must match the training run's --precision",
    )
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--seed-base", type=int, default=20260807)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--cheap-sims", type=int, nargs=2, default=[16, 24])
    parser.add_argument("--full-sims", type=int, nargs=2, default=[64, 128])
    parser.add_argument("--full-search-fraction", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--age-deal-samples", type=int, default=32)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--global-batch-cap", type=int, default=512)
    parser.add_argument("--leaf-batch", type=int, default=1)
    parser.add_argument("--c-visit", type=float, default=50.0)
    parser.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="SIMS:C_SCALE",
        help="repeatable; defaults to 96:0.1 (production), 1024:0.1, 1024:0.01",
    )
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arms = args.arm or ["96:0.1", "1024:0.1", "1024:0.01"]
    parsed_arms = []
    for arm in arms:
        sims_text, _, scale_text = arm.partition(":")
        parsed_arms.append((int(sims_text), float(scale_text)))

    evaluator = load_evaluator(
        args.checkpoint, args.device, precision=args.inference_precision
    )
    positions = collect_positions(
        evaluator=evaluator,
        games=args.games,
        seed_base=args.seed_base,
        cheap_sims=tuple(args.cheap_sims),
        full_sims=tuple(args.full_sims),
        full_search_fraction=args.full_search_fraction,
        top_k=args.top_k,
        age_deal_samples=args.age_deal_samples,
        slots=args.slots,
        batch_cap=args.global_batch_cap,
        samples=args.samples,
    )
    print(f"sampled {len(positions)} full-search positions from {args.games} games")

    out: dict = {"positions": len(positions), "arms": {}}
    for sims, c_scale in parsed_arms:
        rows = run_arm(
            evaluator=evaluator,
            positions=positions,
            sims=sims,
            c_scale=c_scale,
            c_visit=args.c_visit,
            top_k=args.top_k,
            leaf_batch=args.leaf_batch,
        )
        out["arms"][f"sims{sims}_cscale{c_scale}"] = _report(
            f"sims={sims}  c_scale={c_scale}", rows
        )

    if args.out:
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
