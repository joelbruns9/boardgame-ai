"""A/B the fused embedder + vectorised gather on the real Phase D generation path.

Calls `PhaseDLoop._generate_iteration_rust` directly, with exactly the jobs
`generate_iteration` would build, so the measured path includes what the F4
benchmark does not: production `rust_slots`, randomised 64-128 sim budgets, the
annealed draft prior, and the curriculum-bot groups.

The two toggles are patched at the `phase_d` module boundary -- production code
is untouched.  Arms are interleaved (A B B A ...) so thermal drift on a laptop
GPU cannot be mistaken for an effect, and every arm runs the same seeds, so the
games themselves are identical work.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from . import phase_d as pd


def make_arm_patches(fused: bool, gather: bool, precision: str = "fp32"):
    """Return (enter, exit) that force the two toggles for one arm."""

    real_evaluator = pd.Evaluator
    real_adapter = pd.rust_flat_batch_adapter
    forced_precision = precision

    def evaluator(
        model,
        device="cpu",
        max_batch=512,
        fuse_embedder=True,
        precision="fp32",
    ):
        return real_evaluator(
            model,
            device,
            max_batch,
            fuse_embedder=fused,
            precision=forced_precision,
        )

    def adapter(evaluator_, **kwargs):
        kwargs["vectorized_gather"] = gather
        return real_adapter(evaluator_, **kwargs)

    def enter():
        pd.Evaluator = evaluator
        pd.rust_flat_batch_adapter = adapter

    def leave():
        pd.Evaluator = real_evaluator
        pd.rust_flat_batch_adapter = real_adapter

    return enter, leave


def run_arm(
    loop,
    model,
    iteration,
    jobs,
    destination,
    fused,
    gather,
    precision="fp32",
):
    enter, leave = make_arm_patches(fused, gather, precision)
    previous_precision = loop.config.precision
    loop.config.precision = precision
    enter()
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.monotonic()
        records = loop._generate_iteration_rust(model, iteration, destination, jobs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.monotonic() - started
    finally:
        leave()
        loop.config.precision = previous_precision
    stats = dict(loop.last_generation_stats)
    stats["wall_seconds"] = wall
    stats["records"] = len(records)
    stats["moves"] = sum(len(record.moves) for record in records)
    stats["moves_per_second"] = stats["moves"] / wall if wall else 0.0
    # Fingerprint the trajectories so a speed change that also changed the games
    # cannot pass unnoticed.
    fingerprint = []
    for record in records:
        fingerprint.append(
            (
                record.winner,
                record.trajectory_digest,
                record.final_digest,
                tuple(move.action for move in record.moves),
                tuple(move.sims for move in record.moves),
            )
        )
    return stats, tuple(fingerprint), records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--warmup-games", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--arms",
        default="before,after",
        help="comma list from before,fused,gather,after,fp32,bf16",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    config = pd.PhaseDConfig(
        run_dir=str(output / "run"),
        device=args.device,
        games_per_iteration=args.games,
        seed_games=0,
        iterations=1,
    )
    loop = pd.PhaseDLoop(config)
    loop.buffer_dir.mkdir(parents=True, exist_ok=True)
    model = loop.load_model(args.checkpoint)

    def jobs_for(count: int, iteration: int):
        return [
            pd.GameJob(index=index, seed=config.seed + iteration * 1_000_000 + index)
            for index in range(count)
        ]

    arm_settings = {
        "before": (False, False, "fp32"),
        "fused": (True, False, "fp32"),
        "gather": (False, True, "fp32"),
        "after": (True, True, "fp32"),
        "fp32": (True, True, "fp32"),
        "bf16": (True, True, "bf16"),
    }
    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    for name in arms:
        if name not in arm_settings:
            raise SystemExit(f"unknown arm: {name}")

    if args.warmup_games:
        print(f"warmup: {args.warmup_games} games", flush=True)
        run_arm(
            loop,
            model,
            args.iteration,
            jobs_for(args.warmup_games, 999),
            output / "warmup.jsonl",
            *arm_settings[arms[-1]],
        )

    jobs = jobs_for(args.games, args.iteration)
    # A B B A ... so a monotone drift cancels within each arm.
    order: list[str] = []
    for repetition in range(args.repetitions):
        order.extend(arms if repetition % 2 == 0 else list(reversed(arms)))

    results: list[dict] = []
    fingerprints: dict[str, set] = {}
    for position, name in enumerate(order):
        fused, gather, precision = arm_settings[name]
        stats, fingerprint, _ = run_arm(
            loop,
            model,
            args.iteration,
            jobs,
            output / f"{position:02d}_{name}.jsonl",
            fused,
            gather,
        )
        stats["arm"] = name
        stats["position"] = position
        stats["fused_embedder"] = fused
        stats["vectorized_gather"] = gather
        stats["precision"] = precision
        results.append(stats)
        fingerprints.setdefault(name, set()).add(fingerprint)
        print(
            f"[{position}] {name:7s} {stats['wall_seconds']:7.2f}s  "
            f"{stats['games_per_second']:.3f} games/s  "
            f"{stats['moves_per_second']:.1f} moves/s  "
            f"({stats['rust_games']} neural + {stats['rust_bot_games']} bot, "
            f"{stats['rust_chunks']} groups)",
            flush=True,
        )

    summary = {}
    for name in arms:
        times = [row["wall_seconds"] for row in results if row["arm"] == name]
        rates = [row["games_per_second"] for row in results if row["arm"] == name]
        summary[name] = {
            "runs": len(times),
            "median_seconds": statistics.median(times),
            "seconds": times,
            "median_games_per_second": statistics.median(rates),
            "games_per_second": rates,
        }
    base = summary[arms[0]]["median_seconds"]
    for name in arms:
        summary[name]["speedup_vs_" + arms[0]] = base / summary[name]["median_seconds"]

    # Every repeat within one arm must be deterministic. Different precision
    # arms may legitimately take different discrete trajectories, so quantify
    # that divergence instead of treating it as a timing failure.
    identical = {name: len(values) == 1 for name, values in fingerprints.items()}
    all_fingerprints = set().union(*fingerprints.values())
    consistent = len(all_fingerprints) == 1
    representatives = {
        name: next(iter(values)) for name, values in fingerprints.items()
    }
    base_fingerprint = representatives[arms[0]]
    divergence = {}
    for name, candidate in representatives.items():
        changed = sum(
            left != right
            for left, right in zip(base_fingerprint, candidate)
        )
        divergence[name] = {
            "changed_games": changed,
            "games": len(base_fingerprint),
            "rate": changed / len(base_fingerprint) if base_fingerprint else 0.0,
        }
    mixed_precision = len({arm_settings[name][2] for name in arms}) > 1
    payload = {
        "config": {
            "checkpoint": args.checkpoint,
            "games": args.games,
            "device": args.device,
            "repetitions": args.repetitions,
            "arms": arms,
            "rust_slots": config.rust_slots,
            "full_sims": [config.full_sims_min, config.full_sims_max],
            "cheap_sims": [config.cheap_sims_min, config.cheap_sims_max],
            "full_search_fraction": config.full_search_fraction,
            "opponent_fraction": config.opponent_fraction,
            "top_k": config.top_k,
            "d_model": config.d_model,
            "layers": config.layers,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
        },
        "runs": results,
        "summary": summary,
        "self_consistent_within_arm": identical,
        "trajectory_divergence_vs_first_arm": divergence,
        # What the fingerprint covers: the *discrete* trajectory -- winner,
        # chained state digests, action sequence, per-move sim counts. It does
        # not compare policy targets or root values, which are continuous and
        # drift at ~1e-5 under the fused embedder, nor every other record field.
        # So this is "discrete trajectories identical", not "records identical";
        # the record-level comparison lives in
        # `test_f4_phase3b_fused.assert_records_identical`.
        "discrete_trajectories_identical_across_runs": consistent,
        "distinct_fingerprints": len(all_fingerprints),
    }
    (output / "phase_d_ab.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(
        "discrete trajectories identical across all runs: "
        f"{consistent} (within arm: {identical})"
    )
    if not all(identical.values()):
        raise SystemExit(
            "determinism invariant failed within an arm: "
            f"{identical}"
        )
    if not consistent and not mixed_precision:
        raise SystemExit(
            f"equivalence invariant failed: {len(all_fingerprints)} distinct "
            "trajectory sets across runs -- the arms did not do the same work, "
            "so the timing comparison is meaningless"
        )


if __name__ == "__main__":
    main()
