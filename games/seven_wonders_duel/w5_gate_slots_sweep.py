"""Joint sweep of the gate's scheduler axes: ``gate_slots`` x ``global_batch_cap``.

Companion to :mod:`f4_phase_d_sweep`, which sweeps the same axes on the
*generation* path.  The gate needs its own sweep because its job mix is not
generation's:

* **Generation is diluted, the gate is not.**  Generation spends part of every
  iteration on curriculum-bot games that arrive as separate, small scheduler
  calls, so a large slot pool is partly wasted on groups that cannot fill it.
  A gate is one uniform call -- every game is neural, both seat legs share the
  active pool (``_rust_model_gate_waves``) -- so the generation optimum is not
  transferable, and measurement confirmed it: the gate wants 144 slots where
  generation ships 48.
* **The gate runs at fixed sims.**  ``full_search_fraction=0`` there, so every
  move costs ``gate_sims`` simulations rather than generation's playout-capped
  average.  Rows per batch, and therefore where the cap starts binding, differ
  for that reason alone.

The two axes are swept **jointly**, and the first run showed why: the cap's
*sign* depends on the slot count.  On the laptop 3070 at 64 sims, d128 L4,
100-game gates (medians of two repetitions, games/s):

    slots \\ cap     256      512     1024
    48 (shipped)   0.605    0.571    0.581
    96             0.647    0.706    0.757
    144            0.752    0.816    0.840
    192            0.714    0.789    0.828

Raising the cap at the shipped 48 slots *costs* 4%; at 144 slots the same
change *gains* 12%.  Sweeping either axis alone would have concluded the
shipped setting was already optimal.

One trap this harness walks into if ``--games`` is small, visible as
**bit-identical** ``mean_batch_rows`` between adjacent slot counts: the pool
cannot hold more games in flight than the gate plays, so every slot count above
``--games`` measures the same configuration.  The 144-vs-192 gap above is
overhead on unusable slots, not an optimum.

**The optimum is stable in gate size**, which is what makes one sweep enough.
144 slots / 1024 cap wins at 100, 200 and 600 games, and the gain over the
shipped 48/256 barely moves -- 1.39x at 100 games, 1.37x at 600.  At 600 games
the full row reads 0.627 / 0.782 / 0.756 games/s at 48 / 144 / 288 slots and
cap 256, and 0.739 / 0.859 / 0.842 at cap 1024.

An earlier reading of this data claimed the gain decayed with gate size (1.45x
at 100 games, 1.17x at 200).  It does not: those two numbers were measured
against *different* baselines (48/256 and 48/1024).  Against one baseline the
gain is flat.  Compare like with like -- ``--baseline-slots`` and
``--baseline-cap`` exist so the reported speedup names its denominator.

Throughput is flat from 144 slots through 288 at every gate size measured,
while mean batch rows keep climbing (196 to 293 at 200 games, 233 to 350 at
600): the ceiling is the serial scheduler thread, not the slot count.

Both sides of the match are the *same* checkpoint, so every point does an
identical amount of work no matter how the games turn out, and wall clock is
comparable across the grid.  These axes do not change what the search computes,
so the sweep is decided on wall clock alone -- but batch composition does change
float reductions, so ``score_rate`` is reported per point rather than asserted
constant.

Order is reversed on alternate repetitions so thermal drift on a laptop GPU
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

from games.az_loop import hardware_identity

from .phase_d import PhaseDConfig, PhaseDLoop
from .train import heads_from_config


def _measure(args, slots: int, cap: int, index: int) -> dict:
    """One grid point: a single fixed-N gate, timed end to end."""

    config = PhaseDConfig(
        run_dir=str(args.work_dir / f"slots{slots}_cap{cap}_{index}"),
        device=args.device,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        precision=args.precision,
        gate_backend="rust",
        gate_sims=args.sims,
        gate_max_games=args.games,
        gate_slots=slots,
        rust_slots=slots,
        # The gate cap is the one under test; generation's stays at its own
        # default so a point cannot be flattered by a setting the gate does not
        # actually run under.
        gate_global_batch_cap=cap,
        rust_max_inflight_batches=args.max_inflight_batches,
        rust_scheduler_workers=args.scheduler_workers,
        promotion_every=0,
        seed_games=0,
        memory_budget_gb=args.memory_budget_gb,
        vram_budget_gb=args.vram_budget_gb,
    )
    loop = PhaseDLoop(config)
    started = time.monotonic()
    result = loop.promotion_gate(args.checkpoint, opponent=args.checkpoint)
    wall = time.monotonic() - started
    stats = dict(loop.last_gate_stats)
    scheduler = dict(stats.get("scheduler") or {})
    # The per-batch arrays are large and say nothing at this altitude; the
    # summary statistics they were reduced to are what the sweep compares.
    for bulky in ("batch_rows", "batch_live_slots", "batch_submit_ns"):
        scheduler.pop(bulky, None)
    return {
        "slots": slots,
        "global_batch_cap": cap,
        "repetition": index,
        "wall_seconds": wall,
        "match_seconds": float(stats.get("seconds", wall)),
        "games": args.games,
        "games_per_second": args.games / wall if wall else 0.0,
        "match_games_per_second": (
            args.games / float(stats["seconds"])
            if stats.get("seconds")
            else 0.0
        ),
        "setup_seconds": wall - float(stats.get("seconds", wall)),
        "score_rate": float(result.score_rate),
        "moves": stats.get("moves"),
        "scheduler": scheduler,
    }


def run(args) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stored = checkpoint.get("config", {})
    args.d_model = int(stored.get("d_model", 384))
    args.layers = int(stored.get("layers", 8))
    args.heads = heads_from_config(stored)

    grid = list(itertools.product(sorted(set(args.slots)), sorted(set(args.caps))))
    rows: list[dict] = []
    for repetition in range(args.repetitions):
        ordered = grid if repetition % 2 == 0 else list(reversed(grid))
        for slots, cap in ordered:
            row = _measure(args, slots, cap, repetition)
            rows.append(row)
            print(
                f"  slots={slots:<4} cap={cap:<5} "
                f"{row['games_per_second']:.3f} games/s "
                f"(match {row['match_games_per_second']:.3f}, "
                f"setup {row['setup_seconds']:.0f}s) "
                f"batch={float(row['scheduler'].get('mean_batch_rows', 0.0)):.0f}"
                f"/{row['scheduler'].get('max_batch_rows', '-')}",
                flush=True,
            )

    # Median over repetitions, so one thermal outlier cannot pick the winner.
    aggregated: list[dict] = []
    for slots, cap in grid:
        points = [
            row for row in rows if row["slots"] == slots and row["global_batch_cap"] == cap
        ]
        aggregated.append(
            {
                "slots": slots,
                "global_batch_cap": cap,
                "games_per_second": statistics.median(
                    row["games_per_second"] for row in points
                ),
                "match_games_per_second": statistics.median(
                    row["match_games_per_second"] for row in points
                ),
                "setup_seconds": statistics.median(
                    row["setup_seconds"] for row in points
                ),
                "mean_batch_rows": statistics.median(
                    float(row["scheduler"].get("mean_batch_rows", 0.0))
                    for row in points
                ),
                "max_batch_rows": max(
                    int(row["scheduler"].get("max_batch_rows", 0) or 0)
                    for row in points
                ),
                "score_rates": sorted({row["score_rate"] for row in points}),
            }
        )
    best = max(aggregated, key=lambda row: row["games_per_second"])
    baseline = next(
        (
            row
            for row in aggregated
            if row["slots"] == args.baseline_slots
            and row["global_batch_cap"] == args.baseline_cap
        ),
        None,
    )
    try:
        gpu_name = (
            torch.cuda.get_device_name()
            if args.device.startswith("cuda") and torch.cuda.is_available()
            else None
        )
    except Exception:
        gpu_name = None
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "architecture": {
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "precision": args.precision,
        },
        "hardware": {**hardware_identity(), "gpu": gpu_name},
        "settings": {
            "gate_games": args.games,
            "sims": args.sims,
            "max_inflight_batches": args.max_inflight_batches,
            "scheduler_workers": args.scheduler_workers,
            "repetitions": args.repetitions,
        },
        "measurements": rows,
        "grid": aggregated,
        "best": best,
        "baseline": baseline,
        "speedup_vs_baseline": (
            best["games_per_second"] / baseline["games_per_second"]
            if baseline and baseline["games_per_second"]
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--slots", type=int, nargs="+", default=[48, 96, 144, 192])
    parser.add_argument("--caps", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--max-inflight-batches", type=int, default=1)
    parser.add_argument("--scheduler-workers", type=int, default=1)
    parser.add_argument("--memory-budget-gb", type=float, default=0.0)
    parser.add_argument("--vram-budget-gb", type=float, default=0.0)
    parser.add_argument("--baseline-slots", type=int, default=48)
    parser.add_argument("--baseline-cap", type=int, default=256)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["grid"], indent=2))
    print(f"best: {payload['best']}")
    if payload["speedup_vs_baseline"]:
        print(f"speedup vs baseline: {payload['speedup_vs_baseline']:.2f}x")


if __name__ == "__main__":
    main()
