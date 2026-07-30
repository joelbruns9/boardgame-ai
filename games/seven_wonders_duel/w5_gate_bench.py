"""Cloud-hardware gate-cost calibration for the shipped L architecture."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import torch

from games.az_loop import hardware_identity

from .phase_d import PhaseDConfig, PhaseDLoop
from .train import heads_from_config


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
        if denominator
        else 0.0
    )
    return mean_y - slope * mean_x, slope


def run(args) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stored = checkpoint.get("config", {})
    d_model = int(stored.get("d_model", 384))
    layers = int(stored.get("layers", 8))
    heads = heads_from_config(stored)
    if (d_model, layers, heads) != (384, 8, 6) and not args.allow_non_l:
        raise ValueError(
            f"W5.7 requires L=384x8x6, checkpoint is {d_model}x{layers}x{heads}"
        )
    sizes = sorted(set(args.sizes))
    if len(sizes) < 3:
        raise ValueError("W5.7 requires at least three distinct gate sizes")
    rows = []
    for games in sizes:
        config = PhaseDConfig(
            run_dir=str(args.work_dir / f"gate_{games}"),
            device=args.device,
            d_model=d_model,
            layers=layers,
            heads=heads,
            precision=args.precision,
            gate_backend="rust",
            gate_sims=args.sims,
            gate_max_games=games,
            gate_slots=args.slots,
            rust_slots=args.slots,
            rust_global_batch_cap=args.global_batch_cap,
            rust_max_inflight_batches=args.max_inflight_batches,
            rust_scheduler_workers=args.scheduler_workers,
            promotion_every=0,
            seed_games=0,
            hof_opponent_fraction=0.15,
            memory_budget_gb=args.memory_budget_gb,
            vram_budget_gb=args.vram_budget_gb,
        )
        loop = PhaseDLoop(config)
        started = time.monotonic()
        result = loop.promotion_gate(args.checkpoint, opponent=args.checkpoint)
        wall = time.monotonic() - started
        rows.append(
            {
                "gate_games": games,
                "wall_seconds": wall,
                "result": asdict(result),
                "performance": dict(loop.last_gate_stats),
                "resources": asdict(loop.resource_stats()),
            }
        )
    intercept, slope = _linear_fit(
        [float(row["gate_games"]) for row in rows],
        [float(row["wall_seconds"]) for row in rows],
    )
    recommendation = None
    if args.iteration_seconds > 0:
        affordable = []
        for games in sizes:
            gate_seconds = intercept + slope * games
            share = gate_seconds / (
                args.promotion_every * args.iteration_seconds + gate_seconds
            )
            if share <= args.max_gate_fraction:
                affordable.append((games, share, gate_seconds))
        chosen = max(affordable, default=None)
        if chosen is not None:
            recommendation = {
                # W5.8: this is the ladder's *ceiling*, not a single cap --
                # lower rungs are always affordable if the top one is.
                "gate_ladder_ceiling_games": chosen[0],
                "predicted_gate_fraction": chosen[1],
                "predicted_gate_seconds": chosen[2],
                "criterion": (
                    f"largest measured rung at or below "
                    f"{args.max_gate_fraction:.1%} of wall time"
                ),
            }
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
            "d_model": d_model,
            "layers": layers,
            "heads": heads,
            "precision": args.precision,
        },
        "hardware": {**hardware_identity(), "gpu": gpu_name},
        "settings": {
            "sims": args.sims,
            "slots": args.slots,
            "global_batch_cap": args.global_batch_cap,
            "max_inflight_batches": args.max_inflight_batches,
            "scheduler_workers": args.scheduler_workers,
            "hof_opponent_fraction": 0.15,
        },
        "measurements": rows,
        "fit": {
            "fixed_seconds": intercept,
            "seconds_per_game": slope,
        },
        "recommendation": recommendation,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--slots", type=int, default=48)
    parser.add_argument("--global-batch-cap", type=int, default=256)
    parser.add_argument("--max-inflight-batches", type=int, default=2)
    parser.add_argument("--scheduler-workers", type=int, default=1)
    parser.add_argument("--memory-budget-gb", type=float, default=0.0)
    parser.add_argument("--vram-budget-gb", type=float, default=0.0)
    parser.add_argument("--iteration-seconds", type=float, default=0.0)
    parser.add_argument("--promotion-every", type=int, default=5)
    parser.add_argument("--max-gate-fraction", type=float, default=0.10)
    parser.add_argument("--allow-non-l", action="store_true")
    args = parser.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
