"""Joint sweep of the free scheduler axes on the real Phase D generation path.

The axes here are the ones that do not change what the search computes, so they
can be chosen on wall clock alone: `rust_slots`, `rust_global_batch_cap` and
`rust_max_inflight_batches`. They are swept *jointly* because they interact --
the cap binds as slots rise, so sweeping slots alone at a fixed cap finds a
ceiling that belongs to the cap and misattributes it to slots.

Two things this measures that the F4 benchmark cannot:

* **the real job mix.** Phase D spends ~15% of its games on curriculum bots,
  split across `(bot type, seat)` groups that go to the Rust scheduler as
  separate calls. A small group cannot fill a large slot pool, so the benefit of
  more slots is diluted by exactly the fraction of games that are bot games.
  Per-group timings are recorded so that dilution is visible rather than baked
  into one number.
* **whether these axes are really free.** They do not change the search, but
  they do change batch composition, and a different batch shape can change
  floating-point reductions on CUDA. Trajectory fingerprints are compared across
  every point; divergence is reported, not asserted away, because on the real
  net it is information about the axis rather than a bug.

Order is reversed on alternate repetitions so that thermal drift on a laptop GPU
averages out instead of loading onto whichever point runs last.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path

import torch

from . import phase_d as pd
from .train import heads_from_config


def timed_scheduler_calls():
    """Wrap the Rust entry point to record (games, seconds) per group call."""

    import seven_wonders_rust as swr

    real = swr.self_play_many_flat_net
    calls: list[dict] = []

    def wrapper(*args, **kwargs):
        games = len(kwargs.get("game_seeds", ()))
        # Per-game routing puts bot and neural games in one call, so a call is
        # no longer either/or: count the bot games inside it. The scalar form is
        # still used by the seed buffer and the arena, where a call is uniform.
        per_game = list(kwargs.get("bots_p0") or ()) + list(kwargs.get("bots_p1") or ())
        if per_game:
            bot_games = sum(
                1
                for left, right in zip(kwargs["bots_p0"], kwargs["bots_p1"])
                if left is not None or right is not None
            )
            bot = "per-game"
        else:
            bot = kwargs.get("bot_p0") or kwargs.get("bot_p1")
            bot_games = games if bot else 0
        started = time.monotonic()
        result = real(*args, **kwargs)
        calls.append(
            {
                "games": games,
                "bot_games": bot_games,
                "seconds": time.monotonic() - started,
                "bot": bot,
            }
        )
        return result

    swr.self_play_many_flat_net = wrapper
    return calls, (lambda: setattr(swr, "self_play_many_flat_net", real))


def geometry_from_checkpoint(path) -> dict[str, int]:
    """The model width the checkpoint was trained at.

    Read, never assumed or passed as a flag. `_load_model_checkpoint` refuses a
    width mismatch -- W0 lost a run to a checkpoint whose width was inferred --
    so a sweep left on `PhaseDConfig`'s 128x4 defaults cannot load an L
    checkpoint at all, which is how this stage died on its first cloud box.
    `w5_gate_slots_sweep` and `w5_gate_bench` already do exactly this.
    """

    stored = torch.load(path, map_location="cpu", weights_only=False).get("config", {})
    return {
        "d_model": int(stored.get("d_model", 384)),
        "layers": int(stored.get("layers", 8)),
        "heads": heads_from_config(stored),
    }


def run_point(loop, model, iteration, jobs, destination, slots, cap, inflight):
    loop.config.rust_slots = slots
    loop.config.rust_global_batch_cap = cap
    loop.config.rust_max_inflight_batches = inflight

    calls, restore = timed_scheduler_calls()
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.monotonic()
        records = loop._generate_iteration_rust(model, iteration, destination, jobs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.monotonic() - started
    finally:
        restore()

    # Uniform calls split cleanly; a per-game call carries both kinds at once,
    # in which case there is one pool and the split is no longer a timing.
    neural = [call for call in calls if call["bot"] is None]
    bots = [call for call in calls if call["bot"] not in (None, "per-game")]
    mixed = [call for call in calls if call["bot"] == "per-game"]
    fingerprint = tuple(
        (
            record.winner,
            record.trajectory_digest,
            record.final_digest,
            tuple(move.action for move in record.moves),
        )
        for record in records
    )
    stats = {
        "slots": slots,
        "global_batch_cap": cap,
        "max_inflight_batches": inflight,
        "wall_seconds": wall,
        "games": len(records),
        "games_per_second": len(records) / wall if wall else 0.0,
        "games_per_hour": 3600 * len(records) / wall if wall else 0.0,
        "neural_games": sum(call["games"] for call in neural),
        "neural_seconds": sum(call["seconds"] for call in neural),
        "bot_games": sum(call["games"] for call in bots),
        "bot_seconds": sum(call["seconds"] for call in bots),
        "bot_groups": len(bots),
        "mixed_calls": len(mixed),
        "mixed_games": sum(call["games"] for call in mixed),
        "mixed_bot_games": sum(call["bot_games"] for call in mixed),
        "mixed_seconds": sum(call["seconds"] for call in mixed),
        "scheduler_calls": len(calls),
        "calls": calls,
    }
    for prefix in ("neural", "bot"):
        games = stats[f"{prefix}_games"]
        seconds = stats[f"{prefix}_seconds"]
        stats[f"{prefix}_games_per_second"] = games / seconds if seconds else 0.0
    return stats, fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--warmup-games", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="must match the run being configured: bf16 is 1.69x on L, so a "
        "geometry chosen at fp32 is chosen against the wrong cost curve",
    )
    parser.add_argument("--slots", default="16,32,48")
    parser.add_argument("--caps", default="256,512")
    parser.add_argument("--inflight", default="1,2")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    numbers = lambda text: [int(part) for part in text.split(",") if part.strip()]
    grid = list(
        itertools.product(numbers(args.slots), numbers(args.caps), numbers(args.inflight))
    )

    geometry = geometry_from_checkpoint(args.checkpoint)
    config = pd.PhaseDConfig(
        run_dir=str(output / "run"),
        device=args.device,
        games_per_iteration=args.games,
        seed_games=0,
        iterations=1,
        precision=args.precision,
        **geometry,
    )
    loop = pd.PhaseDLoop(config)
    loop.buffer_dir.mkdir(parents=True, exist_ok=True)
    model = loop.load_model(args.checkpoint)

    def jobs_for(count: int, iteration: int):
        return [
            pd.GameJob(index=index, seed=config.seed + iteration * 1_000_000 + index)
            for index in range(count)
        ]

    if args.warmup_games:
        print(f"warmup: {args.warmup_games} games", flush=True)
        run_point(
            loop, model, args.iteration, jobs_for(args.warmup_games, 999),
            output / "warmup.jsonl", *grid[0],
        )

    jobs = jobs_for(args.games, args.iteration)
    if args.games <= max(numbers(args.slots)):
        print(
            f"WARNING: {args.games} games <= {max(numbers(args.slots))} slots; the "
            "pool cannot refill at the top of the grid, so that point measures "
            "activation rather than throughput",
            flush=True,
        )

    results: list[dict] = []
    fingerprints: set = set()
    for repetition in range(args.repetitions):
        order = grid if repetition % 2 == 0 else list(reversed(grid))
        for position, (slots, cap, inflight) in enumerate(order):
            stats, fingerprint = run_point(
                loop, model, args.iteration, jobs,
                output / f"r{repetition}_{position:02d}_s{slots}_c{cap}_i{inflight}.jsonl",
                slots, cap, inflight,
            )
            stats["repetition"] = repetition
            results.append(stats)
            fingerprints.add(fingerprint)
            if stats["mixed_calls"]:
                detail = (
                    f"| 1 pool: {stats['mixed_games']} games "
                    f"({stats['mixed_bot_games']} bot) in "
                    f"{stats['scheduler_calls']} call(s)"
                )
            else:
                detail = (
                    f"| neural {stats['neural_games_per_second']:.3f} g/s, "
                    f"bots {stats['bot_games_per_second']:.3f} g/s over "
                    f"{stats['bot_groups']} groups"
                )
            print(
                f"slots={slots:<3} cap={cap:<4} inflight={inflight}  "
                f"{stats['wall_seconds']:7.1f}s  {stats['games_per_hour']:7.0f} games/h  "
                + detail,
                flush=True,
            )

    summary = []
    for point in grid:
        slots, cap, inflight = point
        matching = [
            row for row in results
            if (row["slots"], row["global_batch_cap"], row["max_inflight_batches"]) == point
        ]
        summary.append(
            {
                "slots": slots,
                "global_batch_cap": cap,
                "max_inflight_batches": inflight,
                "median_seconds": statistics.median(row["wall_seconds"] for row in matching),
                "median_games_per_hour": statistics.median(
                    row["games_per_hour"] for row in matching
                ),
                "runs": len(matching),
            }
        )
    summary.sort(key=lambda row: row["median_seconds"])
    # Compare against Phase D's current defaults when they are in the grid --
    # the number that matters is "what would changing the config buy" -- and
    # fall back to the first grid point when they are not.
    defaults = (
        pd.PhaseDConfig.rust_slots,
        pd.PhaseDConfig.rust_global_batch_cap,
        pd.PhaseDConfig.rust_max_inflight_batches,
    )
    key = lambda row: (row["slots"], row["global_batch_cap"], row["max_inflight_batches"])
    base = next((row for row in summary if key(row) == defaults), None) or next(
        row for row in summary if key(row) == grid[0]
    )
    baseline_label = "/".join(str(part) for part in key(base))
    for row in summary:
        row["speedup_vs_baseline"] = base["median_seconds"] / row["median_seconds"]
    payload_baseline = baseline_label

    payload = {
        "config": {
            "checkpoint": args.checkpoint,
            "games": args.games,
            "repetitions": args.repetitions,
            "grid": [list(point) for point in grid],
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "opponent_fraction": config.opponent_fraction,
        },
        "runs": results,
        "summary": summary,
        "baseline": payload_baseline,
        "distinct_trajectory_sets": len(fingerprints),
    }
    (output / "phase_d_sweep.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("\nbest first:")
    for row in summary:
        print(
            f"  slots={row['slots']:<3} cap={row['global_batch_cap']:<4} "
            f"inflight={row['max_inflight_batches']}  "
            f"{row['median_games_per_hour']:7.0f} games/h  "
            f"({row['speedup_vs_baseline']:.2f}x vs {baseline_label})"
        )
    print(
        f"\ndistinct trajectory sets across all points: {len(fingerprints)} "
        "(1 = these axes changed nothing the search saw)"
    )


if __name__ == "__main__":
    main()
