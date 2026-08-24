"""CUDA-only production-geometry sweep for S2 in-flight game count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import torch

from games.welcome_to import self_play
from games.welcome_to import train


def sweep(
    checkpoint: str | Path,
    *,
    inflight: Sequence[int] = (8, 16, 32),
    games: int = 32,
    simulations: int = 200,
    seed: int = 15_000,
    device: str = "cuda",
) -> list[dict]:
    """Run identical games at each scheduler width and retain no corpus."""
    if device != "cuda":
        raise ValueError("the S2 in-flight sweep is intentionally CUDA-only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if games <= 0 or simulations <= 0 or not inflight:
        raise ValueError("games, simulations, and in-flight arms must be positive")
    if any(width <= 0 or width > games for width in inflight):
        raise ValueError("each in-flight arm must be in [1, games]")

    net = train.load(checkpoint, device).eval()
    reference: Optional[dict[int, str]] = None
    results: list[dict] = []
    for width in inflight:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        trajectories, metrics = self_play.generate(
            net,
            config=self_play.SelfPlayConfig(
                games=games,
                inflight=width,
                max_batch=width,
                seed=seed,
            ),
            search_config=self_play.default_search_config(simulations),
            device=device,
        )
        torch.cuda.synchronize()
        fingerprints = {
            trajectory.seed: trajectory.to_json() for trajectory in trajectories
        }
        if reference is None:
            reference = fingerprints
            agreement = 1.0
            mismatches: list[int] = []
        else:
            mismatches = sorted(
                game_seed
                for game_seed in reference
                if fingerprints.get(game_seed) != reference[game_seed]
            )
            agreement = (games - len(mismatches)) / games
        row = {
            "inflight": float(width),
            "max_batch": float(width),
            "simulations": float(simulations),
            "trajectory_agreement": agreement,
            "trajectory_mismatches": mismatches,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            **metrics,
        }
        results.append(row)
        print(
            f"inflight={width:>2}  {metrics['games_per_hour']:.1f} games/h  "
            f"{metrics['evaluator_rows_per_second']:.1f} rows/s  "
            f"batch={metrics['mean_batch']:.2f} "
            f"p90={metrics['batch_p90']:.0f}  "
            f"VRAM={row['cuda_peak_allocated_mib']:.0f} MiB  "
            f"agreement={agreement:.3f}",
            flush=True,
        )
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inflight", default="8,16,32")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=15_000)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    widths = tuple(int(value) for value in args.inflight.split(",") if value)
    results = sweep(
        args.checkpoint,
        inflight=widths,
        games=args.games,
        simulations=args.simulations,
        seed=args.seed,
        device=args.device,
    )
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(results, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        print(f"wrote sweep to {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
