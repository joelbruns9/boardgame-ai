"""W6.2b: play the shipped precision before trusting it.

W0 chose bf16 on training-loss and throughput evidence, but every arena it ran
was fp32, and bf16 changed 43 of 64 L trajectories.  So the configuration this
plan ships has never played a scored game.

This is the cheapest possible check on that: the **same checkpoint** on both
sides, one evaluator in bf16 and one in fp32, seat-paired and fixed-N.  Because
the weights are identical the null is known exactly -- 0.500 -- so there is no
seed-variance problem of the kind that limited W0's width claim, and no need for
a reference opponent.  A few hundred games costs minutes on a rented GPU.

If the interval excludes 0.500, bf16 is not merely a different rounding of the
same policy and the fallback is L/fp32 at 1.69x the cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from games.az_loop import hardware_identity, wilson_interval

from .phase_d import PhaseDConfig, PhaseDLoop
from .train import heads_from_config


def run(
    checkpoint: Path,
    *,
    games: int,
    device: str,
    sims: int,
    slots: int,
    global_batch_cap: int,
    work_dir: Path,
    z: float = 1.96,
) -> dict:
    import torch

    stored = torch.load(checkpoint, map_location="cpu", weights_only=False).get(
        "config", {}
    )
    d_model = int(stored.get("d_model", 384))
    layers = int(stored.get("layers", 8))
    heads = heads_from_config(stored)

    config = PhaseDConfig(
        run_dir=str(work_dir),
        device=device,
        d_model=d_model,
        layers=layers,
        heads=heads,
        # The loop-level precision is irrelevant here: each side is built with
        # its own, which is the entire point of the comparison.
        precision="fp32",
        gate_backend="rust",
        gate_sims=sims,
        gate_max_games=games,
        gate_slots=slots,
        rust_slots=slots,
        rust_global_batch_cap=global_batch_cap,
        promotion_every=0,
        seed_games=0,
    )
    loop = PhaseDLoop(config)
    spec = loop._model_agent_spec(checkpoint, "precision_arena")
    report, outcomes = loop._wilson_model_match(
        spec,
        spec,
        seed_offset=52_000_000,
        games=games,
        precisions=("bf16", "fp32"),
    )

    lower, upper = wilson_interval(
        report.score_rate * report.pairs, report.pairs, z=z
    )
    null_inside = lower <= 0.50 <= upper
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "architecture": {"d_model": d_model, "layers": layers, "heads": heads},
        "arms": {"candidate": "bf16", "opponent": "fp32"},
        "games": len(outcomes),
        "pairs": report.pairs,
        "bf16_score_rate": report.score_rate,
        "wilson": {"lower": lower, "upper": upper, "z": z},
        "null": 0.50,
        "null_inside_interval": null_inside,
        "moves_per_game": report.moves_per_game,
        "seconds": report.seconds,
        "pair_scores": list(report.pair_scores),
        "hardware": hardware_identity(),
        "verdict": (
            "bf16 is indistinguishable from fp32 at this sample size; ship bf16"
            if null_inside
            else "bf16 differs from fp32 beyond the interval; ship L/fp32 and "
            "accept the 1.69x cost"
        ),
        "passed": null_inside,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--slots", type=int, default=48)
    parser.add_argument("--global-batch-cap", type=int, default=256)
    args = parser.parse_args(argv)

    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number (seat pairs)")
    if not args.checkpoint.is_file():
        # argparse exits 2, which is what lets a caller tell "could not run" from
        # this tool's exit 1, "ran and the precisions disagreed". They are
        # opposite conclusions and must not share an exit code.
        parser.error(f"--checkpoint {args.checkpoint} does not exist")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report = run(
        args.checkpoint,
        games=args.games,
        device=args.device,
        sims=args.sims,
        slots=args.slots,
        global_batch_cap=args.global_batch_cap,
        work_dir=args.work_dir,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    wilson = report["wilson"]
    print(
        f"bf16 vs fp32 over {report['games']} games "
        f"({report['pairs']} pairs): {report['bf16_score_rate']:.3f} "
        f"[{wilson['lower']:.3f}, {wilson['upper']:.3f}] against a null of 0.500"
    )
    print(report["verdict"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
