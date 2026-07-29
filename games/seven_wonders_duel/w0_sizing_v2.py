"""Validity-repair tools for the W0 V2 sizing decision.

V1 artifacts remain immutable.  This module consumes explicit checkpoints and
uses fixed experimental budgets so selection, stopping, and provenance are
visible in every JSON result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from games.kingdomino.promotion import wilson_lower_bound

from .buffer import read_records
from .dataset import examples_from_record
from .phase_d import PhaseDConfig, PhaseDLoop
from .train import build_model, heads_from_config
from .w0_sizing import (
    _evaluate_tensor_cache,
    _load_agent,
    _pack_examples,
)


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson_interval(points: float, samples: int) -> tuple[float, float]:
    return (
        wilson_lower_bound(points, samples),
        1.0 - wilson_lower_bound(samples - points, samples),
    )


class _FixedPairCollector:
    """Collect paired-seat scores without any data-dependent stopping."""

    decision = "continue"

    def __init__(self):
        self.game_scores: list[float] = []
        self.pair_scores: list[float] = []

    def update(self, score: float):
        self.game_scores.append(float(score))
        if len(self.game_scores) % 2 == 0:
            self.pair_scores.append(
                (self.game_scores[-2] + self.game_scores[-1]) / 2.0
            )
        return self


def fixed_arena_one(args) -> None:
    if args.games <= 0 or args.games % 2:
        raise ValueError("--games must be a positive even number")
    output = Path(args.output)
    result_path = output / "arena_cells" / f"{args.cell}.json"
    if result_path.is_file() and not args.force:
        print(f"fixed arena already complete: {result_path}", flush=True)
        return

    candidate_path = Path(args.candidate_checkpoint)
    opponent_path = Path(args.opponent_checkpoint)
    config = PhaseDConfig(
        run_dir=str(output / "arena_runtime"),
        device=args.device,
        precision=args.precision,
        d_model=128,
        layers=4,
        seed_games=0,
        promotion_every=0,
        gate_sims=args.sims,
        gate_max_games=args.games,
        rust_slots=args.rust_slots,
        rust_global_batch_cap=args.batch_cap,
        eval_search_mode="gumbel",
        search_mode="closed",
        top_k=16,
    )
    loop = PhaseDLoop(config)
    candidate = _load_agent(
        candidate_path,
        args.candidate,
        args.sims,
        config.search_mode,
        config.top_k,
    )
    opponent = _load_agent(
        opponent_path,
        args.opponent,
        args.sims,
        config.search_mode,
        config.top_k,
    )
    collector = _FixedPairCollector()
    print(
        f"fixed arena {args.cell}: {args.candidate} vs {args.opponent}, "
        f"{args.games} games, precision={args.precision}",
        flush=True,
    )
    started = time.monotonic()
    outcomes = loop._rust_model_gate_waves(
        candidate, opponent, collector, args.seed_offset
    )
    seconds = time.monotonic() - started
    if len(outcomes) != args.games or len(collector.game_scores) != args.games:
        raise AssertionError(
            f"fixed arena returned {len(outcomes)} games, expected {args.games}"
        )
    if len(collector.pair_scores) != args.games // 2:
        raise AssertionError("fixed arena did not retain complete seat pairs")
    pair_points = sum(collector.pair_scores)
    lower, upper = _wilson_interval(pair_points, len(collector.pair_scores))
    payload = {
        "cell": args.cell,
        "candidate": args.candidate,
        "opponent": args.opponent,
        "candidate_checkpoint": str(candidate_path),
        "candidate_checkpoint_sha256": _sha256(candidate_path),
        "opponent_checkpoint": str(opponent_path),
        "opponent_checkpoint_sha256": _sha256(opponent_path),
        "precision": args.precision,
        "sims": args.sims,
        "games": len(collector.game_scores),
        "pairs": len(collector.pair_scores),
        "game_points": sum(collector.game_scores),
        "score_rate": sum(collector.game_scores) / len(collector.game_scores),
        "pair_points": pair_points,
        "pair_score_rate": pair_points / len(collector.pair_scores),
        "wilson_lower_fixed_n": lower,
        "wilson_upper_fixed_n": upper,
        "stopping": "fixed_n",
        "seed_offset": args.seed_offset,
        "seconds": seconds,
        "game_scores": collector.game_scores,
        "pair_scores": collector.pair_scores,
        "outcomes": [asdict(outcome) for outcome in outcomes],
    }
    _atomic_json(result_path, payload)
    print(
        f"completed {args.cell}: score={payload['score_rate']:.3f} "
        f"fixed-N CI=[{lower:.3f}, {upper:.3f}] in {seconds / 60:.1f}m",
        flush=True,
    )


def aggregate_fixed_arenas(args) -> None:
    root = Path(args.output) / "arena_cells"
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob(f"{args.match}_*.json"))
    ]
    if len(rows) != args.expected_cells:
        raise ValueError(
            f"{args.match}: found {len(rows)} cells, "
            f"expected {args.expected_cells}"
        )
    precisions = {row["precision"] for row in rows}
    sims = {row["sims"] for row in rows}
    games_per_cell = {row["games"] for row in rows}
    if len(precisions) != 1 or len(sims) != 1 or len(games_per_cell) != 1:
        raise ValueError("arena cells do not share precision/sims/fixed N")
    pair_scores = [
        float(score) for row in rows for score in row["pair_scores"]
    ]
    pair_points = sum(pair_scores)
    lower, upper = _wilson_interval(pair_points, len(pair_scores))
    cell_rates = [float(row["pair_score_rate"]) for row in rows]
    payload = {
        "match": args.match,
        "candidate": rows[0]["candidate"],
        "opponent": rows[0]["opponent"],
        "precision": next(iter(precisions)),
        "sims": next(iter(sims)),
        "cells": len(rows),
        "games_per_cell": next(iter(games_per_cell)),
        "games": sum(int(row["games"]) for row in rows),
        "pairs": len(pair_scores),
        "pair_points": pair_points,
        "score_rate": pair_points / len(pair_scores),
        "wilson_lower_fixed_n": lower,
        "wilson_upper_fixed_n": upper,
        "cell_mean": statistics.mean(cell_rates),
        "cell_sd": statistics.stdev(cell_rates),
        "cell_min": min(cell_rates),
        "cell_max": max(cell_rates),
        "cells_above_half": sum(rate > 0.5 for rate in cell_rates),
        "stopping": "fixed_n",
        "cell_results": rows,
    }
    _atomic_json(Path(args.output) / "arenas" / f"{args.match}.json", payload)
    print(
        f"{args.match}: score={payload['score_rate']:.3f}, "
        f"CI=[{lower:.3f}, {upper:.3f}], "
        f"cells above 0.5={payload['cells_above_half']}/{len(rows)}",
        flush=True,
    )


def prepare_holdout(args) -> None:
    output = Path(args.output)
    destination = output / "holdout.pt"
    metadata_path = output / "holdout.json"
    if destination.is_file() and metadata_path.is_file() and not args.force:
        print(f"holdout already complete: {destination}", flush=True)
        return
    buffer_dir = Path(args.buffer_dir)
    iterations = list(range(args.first_iteration, args.last_iteration + 1))
    paths = [buffer_dir / f"iter_{iteration:04d}.jsonl" for iteration in iterations]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing holdout buffers: {missing}")
    examples = []
    games = 0
    started = time.monotonic()
    for path in paths:
        records = read_records(path)
        games += len(records)
        for record in records:
            examples.extend(
                examples_from_record(record, record_fast_moves=False)
            )
    packed = _pack_examples(examples, 0.0, "w0-v2-external-holdout")
    packed["metadata"] = {
        "games": games,
        "examples": len(examples),
        "record_fast_moves": False,
        "source_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths
        ],
    }
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(packed, temporary)
    temporary.replace(destination)
    metadata = {
        **packed["metadata"],
        "first_iteration": args.first_iteration,
        "last_iteration": args.last_iteration,
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "seconds": time.monotonic() - started,
        "use": "post-selection audit only; never used for LR selection",
    }
    _atomic_json(metadata_path, metadata)
    print(
        f"prepared untouched holdout: {games:,} games, "
        f"{len(examples):,} examples in {metadata['seconds'] / 60:.1f}m",
        flush=True,
    )


@torch.no_grad()
def evaluate_holdout(args) -> None:
    output = Path(args.output)
    result_path = output / "holdout_results" / f"{args.label}.json"
    if result_path.is_file() and not args.force:
        print(f"holdout result already complete: {result_path}", flush=True)
        return
    holdout_path = output / "holdout.pt"
    packed = torch.load(
        holdout_path, map_location="cpu", mmap=True, weights_only=False
    )
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    model = build_model(
        config.get("model", "transformer"),
        int(config["d_model"]),
        int(config["layers"]),
        heads_from_config(config),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device)
    indices = torch.arange(
        len(packed["storage"]["value_class"]), dtype=torch.long
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    metrics = _evaluate_tensor_cache(
        model,
        packed["storage"],
        indices,
        args.device,
        args.batch_size,
        args.precision,
    )
    seconds = time.monotonic() - started
    payload = {
        "label": args.label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "holdout_sha256": _sha256(holdout_path),
        "precision": args.precision,
        "batch_size": args.batch_size,
        "examples": len(indices),
        "metrics": metrics,
        "seconds": seconds,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None
        ),
    }
    _atomic_json(result_path, payload)
    print(
        f"holdout {args.label}: total={metrics['total']:.6f} "
        f"over {len(indices):,} examples in {seconds:.1f}s",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    arena = subparsers.add_parser("fixed-arena-one")
    arena.add_argument("--output", required=True)
    arena.add_argument("--cell", required=True)
    arena.add_argument("--candidate", required=True)
    arena.add_argument("--opponent", required=True)
    arena.add_argument("--candidate-checkpoint", required=True)
    arena.add_argument("--opponent-checkpoint", required=True)
    arena.add_argument("--games", type=int, default=88)
    arena.add_argument("--sims", type=int, default=64)
    arena.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    arena.add_argument("--device", default="cuda")
    arena.add_argument("--rust-slots", type=int, default=48)
    arena.add_argument("--batch-cap", type=int, default=256)
    arena.add_argument("--seed-offset", type=int, required=True)
    arena.add_argument("--force", action="store_true")

    aggregate = subparsers.add_parser("aggregate-fixed-arenas")
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--match", required=True)
    aggregate.add_argument("--expected-cells", type=int, default=9)

    holdout = subparsers.add_parser("prepare-holdout")
    holdout.add_argument("--output", required=True)
    holdout.add_argument("--buffer-dir", required=True)
    holdout.add_argument("--first-iteration", type=int, default=36)
    holdout.add_argument("--last-iteration", type=int, default=40)
    holdout.add_argument("--force", action="store_true")

    evaluate = subparsers.add_parser("evaluate-holdout")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--label", required=True)
    evaluate.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=int, default=512)
    evaluate.add_argument("--force", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "fixed-arena-one":
        fixed_arena_one(args)
    elif args.command == "aggregate-fixed-arenas":
        aggregate_fixed_arenas(args)
    elif args.command == "prepare-holdout":
        prepare_holdout(args)
    else:
        evaluate_holdout(args)


if __name__ == "__main__":
    main()
