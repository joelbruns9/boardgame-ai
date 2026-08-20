"""Does leaf batching cost playing strength? A paired, same-net A/B.

Kingdomino settled this with head-to-head games and no solver at all: run a
match, read the win rate, and keep the setting that wins. Its notes record
`--leaf_batch 6` as "the best known quality setting" with degradation "around
8" -- an interior optimum found by playing, not by proving.

This is the same measurement with one improvement 7WD's engine allows.
`leaf_batch_by_player` sets the batch PER SEAT, so both arms can play inside the
SAME game against each other: identical network, identical position stream,
identical seeds, differing only in how many leaves one side has in flight. KD
had to hold a fixed reference opponent and compare two separate match results,
which spends power on the reference's noise.

Three things make the number mean what it looks like:

* **Seat-swapped.** 7WD is not seat-symmetric -- the first player picks first --
  so every seed is played twice with the arms exchanged. Reporting a win rate
  without this measures the seat advantage as much as the setting.
* **Deterministic actions.** Temperature sampling would add variance that has
  nothing to do with batching. Arena games play the argmax.
* **Paired seeds.** Both arms meet the same deals, so deal luck cancels rather
  than averaging out slowly.

The null is 50%. A binomial interval that excludes 50% is degradation (or gain);
one that contains it means this many games could not tell, which is a different
statement from "no effect" and is reported as such.

Usage:
    python -m games.seven_wonders_duel.leaf_batch_ab \\
        --checkpoint runs/.../current_best.pt --leaf-batches 2,4,6,8 --games 200
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from . import phase_d as pd
from .f4_phase_d_sweep import config_from_manifest, geometry_from_checkpoint
from .rust_bridge import rust_flat_batch_adapter, rust_games_for_self_play


def wilson_interval(wins: float, games: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: correct near 0 and 1, where normal approximation
    produces bounds outside [0, 1] and reads as false precision."""

    if games <= 0:
        return (0.0, 1.0)
    phat = wins / games
    denominator = 1 + z * z / games
    centre = (phat + z * z / (2 * games)) / denominator
    margin = (
        z
        * math.sqrt(phat * (1 - phat) / games + z * z / (4 * games * games))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def play_pairing(
    loop: "pd.PhaseDLoop",
    model,
    seeds: list[int],
    batch_a: int,
    batch_b: int,
    swap: bool,
) -> list[dict[str, Any]]:
    """One block of games with `batch_a` on seat 0 (or seat 1 when swapped)."""

    import seven_wonders_rust as swr

    config = loop.config
    p0, p1 = (batch_b, batch_a) if swap else (batch_a, batch_b)
    jobs = [pd.GameJob(index=i, seed=seed) for i, seed in enumerate(seeds)]
    evaluator = pd.Evaluator(
        model, config.device, config.rust_global_batch_cap,
        precision=config.precision,
    )
    adapter = rust_flat_batch_adapter(evaluator)
    # First player alternates with the seat swap so the arm under test does not
    # always move first -- the seat advantage is real and would otherwise load
    # entirely onto one arm.
    raw_records, _ = swr.self_play_many_flat_net(
        adapter=adapter,
        games=rust_games_for_self_play(seeds, [0] * len(seeds)),
        game_seeds=seeds,
        global_batch_cap=config.rust_global_batch_cap,
        leaf_batch=1,
        leaf_batch_p0=p0,
        leaf_batch_p1=p1,
        cheap_sims_min=config.cheap_sims_min,
        cheap_sims_max=config.cheap_sims_max,
        full_sims_min=config.full_sims_min,
        full_sims_max=config.full_sims_max,
        # Every move at the full budget: the setting under test is a search
        # setting, and a cheap/full mix would dilute it by whatever fraction of
        # moves ran cheap.
        full_search_fraction=1.0,
        top_k=config.top_k,
        draft_prior=0.0,
        iteration=-1,
        force=config.force_root_chance,
        age_deal_samples=config.age_deal_samples,
        max_inflight_batches=config.rust_max_inflight_batches,
        scheduler_workers=config.rust_scheduler_workers,
        max_active_slots=config.rust_slots,
        # No temperature, no root noise: arena conditions. Dirichlet would add
        # variance unrelated to the setting being measured.
        deterministic_actions=True,
        dirichlet_epsilon=0.0,
        puct_root=config.selfplay_search_mode == "puct",
        cheap_puct_root=None,
        virtual_loss_root=config.virtual_loss_root,
        conflict_free_waves=config.conflict_free_waves,
        round_robin_candidates=config.round_robin_candidates,
        solve_endgames=False,
    )
    rows = []
    for job, raw in zip(jobs, raw_records):
        winner = raw["winner"] if isinstance(raw, dict) else raw.winner
        # `winner` is a seat; translate to "did arm A win".
        seat_of_a = 1 if swap else 0
        rows.append(
            {
                "seed": job.seed,
                "swap": swap,
                "winner_seat": winner,
                "a_won": None if winner is None or winner < 0 else int(winner == seat_of_a),
            }
        )
    return rows


def summarise(rows: list[dict[str, Any]], batch_a: int, batch_b: int) -> dict[str, Any]:
    decided = [row for row in rows if row["a_won"] is not None]
    wins = sum(row["a_won"] for row in decided)
    draws = len(rows) - len(decided)
    low, high = wilson_interval(wins, len(decided)) if decided else (0.0, 1.0)
    return {
        "leaf_batch_a": batch_a,
        "leaf_batch_b": batch_b,
        "games": len(rows),
        "decided": len(decided),
        "draws": draws,
        "a_wins": wins,
        "a_win_rate": wins / len(decided) if decided else 0.0,
        "ci95": [low, high],
        # The null is a coin flip. Excluding it is the whole verdict.
        "separates_from_even": not (low <= 0.5 <= high),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config-from-manifest",
        default="",
        help="run_manifest.json of the run being tuned, so the A/B runs that "
        "run's search rather than PhaseDConfig defaults",
    )
    parser.add_argument(
        "--leaf-batches",
        default="2,4,6,8",
        help="values to test, each against leaf_batch=1 on the other seat",
    )
    parser.add_argument("--games", type=int, default=200,
                        help="paired games per value; half run seat-swapped")
    parser.add_argument("--seed-base", type=int, default=20260820)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    values = [int(part) for part in args.leaf_batches.split(",") if part.strip()]
    if any(value < 1 for value in values):
        raise SystemExit("--leaf-batches must be positive")

    geometry = geometry_from_checkpoint(args.checkpoint)
    work = Path(args.out).parent if args.out else Path("runs/leaf_batch_ab")
    if args.config_from_manifest:
        config = config_from_manifest(
            args.config_from_manifest,
            output=work,
            device=args.device,
            games=args.games,
            precision=args.precision,
            geometry=geometry,
        )
    else:
        print(
            "WARNING: no --config-from-manifest; search settings take "
            "PhaseDConfig defaults and may not match the run being tuned.",
            flush=True,
        )
        config = pd.PhaseDConfig(
            run_dir=str(work / "run"), device=args.device,
            games_per_iteration=args.games, seed_games=0, iterations=1,
            precision=args.precision, **geometry,
        )
    # Batching a PUCT root is the thing under test, so it must be permitted
    # here even though a training run would have to opt in deliberately.
    config.virtual_loss_root = True
    config.validate()

    loop = pd.PhaseDLoop(config)
    loop.buffer_dir.mkdir(parents=True, exist_ok=True)
    model = loop.load_model(args.checkpoint)

    half = max(1, args.games // 2)
    results = []
    for value in values:
        started = time.monotonic()
        seeds = [args.seed_base + i for i in range(half)]
        rows = play_pairing(loop, model, seeds, value, 1, swap=False)
        rows += play_pairing(loop, model, seeds, value, 1, swap=True)
        summary = summarise(rows, value, 1)
        summary["seconds"] = time.monotonic() - started
        results.append(summary)
        verdict = (
            "DIFFERS from even"
            if summary["separates_from_even"]
            else "cannot separate from even at this sample size"
        )
        print(
            f"leaf_batch {value} vs 1: {summary['a_wins']}/{summary['decided']} "
            f"= {summary['a_win_rate']:.3f} "
            f"[{summary['ci95'][0]:.3f},{summary['ci95'][1]:.3f}] "
            f"({summary['draws']} draws, {summary['seconds']:.0f}s) -- {verdict}",
            flush=True,
        )

    payload = {
        "checkpoint": str(args.checkpoint),
        "games_per_value": args.games,
        "search": {
            "mode": config.selfplay_search_mode,
            "full_sims": [config.full_sims_min, config.full_sims_max],
            "top_k": config.top_k,
        },
        "results": results,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
