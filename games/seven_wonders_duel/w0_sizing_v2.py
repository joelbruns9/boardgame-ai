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
import math
from pathlib import Path
import statistics
import time

import torch

from games.kingdomino.promotion import wilson_lower_bound

from .buffer import read_records
from .dataset import examples_from_record
from .inference import Evaluator
from .phase_d import MatchOutcome, PhaseDConfig, PhaseDLoop
from .rust_bridge import rust_flat_batch_adapter, rust_games_for_self_play
from .train import build_model, heads_from_config
from .w0_sizing import (
    _evaluate_tensor_cache,
    _load_agent,
    _pack_examples,
    _tensor_batch,
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


def _root_priors_from_digest(digest: list[float]) -> list[float]:
    """Extract root edge priors from the production search-tree digest."""

    def read_node(cursor: int, capture: bool) -> tuple[int, list[float]]:
        cursor += 4  # visits, value sum, actor, terminal
        fingerprint_size = int(digest[cursor])
        cursor += 1 + fingerprint_size
        edge_count = int(digest[cursor])
        cursor += 1
        priors = []
        for _ in range(edge_count):
            prior = float(digest[cursor + 3])
            children = int(digest[cursor + 5])
            cursor += 6
            if capture:
                priors.append(prior)
            for _ in range(children):
                key_parts = int(digest[cursor])
                cursor += 1
                for _ in range(key_parts):
                    part_size = int(digest[cursor])
                    cursor += 1 + part_size
                cursor += 2  # samples, probability
                cursor, _ = read_node(cursor, False)
        return cursor, priors

    end, priors = read_node(0, True)
    if end != len(digest):
        raise ValueError(f"tree digest has {len(digest) - end} trailing values")
    if not priors:
        raise ValueError("tree digest has no root priors")
    return priors


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


def summarize_training(args) -> None:
    """Apply the preregistered two-stage adaptive LR selection rule."""

    output = Path(args.output)
    all_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "training").glob("*.json"))
    ]
    metrics_to_summarize = (
        "total",
        "value_acc",
        "joint7_acc",
        "policy_top1",
    )
    result = {
        "rules": {
            "stage_1": (
                "retain every seed-0 LR within 1.0% of the arm's minimum"
            ),
            "stage_2": (
                "select the lower LR among three-seed mean validation totals "
                "within 0.5% of the minimum mean"
            ),
            "holdout_use": (
                "post-selection audit only; external holdout cannot change LR"
            ),
        },
        "arms": {},
    }
    cache_hashes = {row["tensor_cache_sha256"] for row in all_rows}
    precisions = {row["precision"] for row in all_rows}
    if len(cache_hashes) != 1 or len(precisions) != 1:
        raise ValueError("training rows do not share one tensor cache/precision")

    for arm in ("S", "M", "L"):
        sweeps = [
            row
            for row in all_rows
            if row["arm"] == arm
            and row["kind"] == "sweep"
            and row["seed"] == 0
        ]
        if not sweeps:
            raise ValueError(f"{arm}: no sweep rows")
        minimum = min(row["metrics"]["total"] for row in sweeps)
        contenders = [
            row
            for row in sweeps
            if row["metrics"]["total"] <= minimum * 1.01
        ]
        contender_summaries = []
        for sweep in sorted(contenders, key=lambda row: row["learning_rate"]):
            lr = float(sweep["learning_rate"])
            rows = [
                row
                for row in all_rows
                if row["arm"] == arm
                and math.isclose(
                    float(row["learning_rate"]), lr, rel_tol=0.0, abs_tol=1e-15
                )
                and int(row["seed"]) in (0, 1, 2)
            ]
            by_seed = {int(row["seed"]): row for row in rows}
            if set(by_seed) != {0, 1, 2}:
                raise ValueError(
                    f"{arm} lr={lr:g}: expected seeds 0/1/2, "
                    f"found {sorted(by_seed)}"
                )
            ordered = [by_seed[seed] for seed in (0, 1, 2)]
            totals = [float(row["metrics"]["total"]) for row in ordered]
            contender_summaries.append(
                {
                    "learning_rate": lr,
                    "seed0_sweep_total": float(sweep["metrics"]["total"]),
                    "validation_total_mean": statistics.mean(totals),
                    "validation_total_sd": statistics.stdev(totals),
                    "validation_total_min": min(totals),
                    "validation_total_max": max(totals),
                    "rows": ordered,
                }
            )
        best_mean = min(
            row["validation_total_mean"] for row in contender_summaries
        )
        tied = [
            row
            for row in contender_summaries
            if row["validation_total_mean"] <= best_mean * 1.005
        ]
        selected = min(tied, key=lambda row: row["learning_rate"])
        selected_rows = selected["rows"]
        metric_summary = {}
        for metric in metrics_to_summarize:
            values = [
                float(row["metrics"][metric]) for row in selected_rows
            ]
            metric_summary[metric] = {
                "mean": statistics.mean(values),
                "sd": statistics.stdev(values),
                "min": min(values),
                "max": max(values),
            }
        result["arms"][arm] = {
            "sweep": sorted(
                [
                    {
                        "learning_rate": float(row["learning_rate"]),
                        "validation_total": float(row["metrics"]["total"]),
                    }
                    for row in sweeps
                ],
                key=lambda row: row["learning_rate"],
            ),
            "stage_1_minimum": minimum,
            "stage_1_contenders": [
                float(row["learning_rate"]) for row in contenders
            ],
            "contender_summaries": contender_summaries,
            "selected_learning_rate": selected["learning_rate"],
            "selected_tie_candidates": [
                row["learning_rate"] for row in tied
            ],
            "metrics": metric_summary,
            "training_seconds_mean": statistics.mean(
                float(row["seconds"]) for row in selected_rows
            ),
            "training_milliseconds_per_step_mean": statistics.mean(
                float(row["history"][-1]["secs"]) / int(row["steps"]) * 1000
                for row in selected_rows
            ),
            "cuda_peak_allocated_bytes_max": max(
                int(row["cuda_peak_allocated_bytes"]) for row in selected_rows
            ),
            "cuda_peak_reserved_bytes_max": max(
                int(row["cuda_peak_reserved_bytes"]) for row in selected_rows
            ),
            "selected_checkpoints": [
                {
                    "seed": int(row["seed"]),
                    "path": row["checkpoint"],
                    "sha256": row["checkpoint_sha256"],
                }
                for row in selected_rows
            ],
        }
    result["tensor_cache_sha256"] = next(iter(cache_hashes))
    result["precision"] = next(iter(precisions))
    _atomic_json(output / "training_summary.json", result)
    for arm, row in result["arms"].items():
        print(
            f"{arm}: selected lr={row['selected_learning_rate']:.0e}, "
            f"val.total={row['metrics']['total']['mean']:.6f}"
            f"+/-{row['metrics']['total']['sd']:.6f}",
            flush=True,
        )


def search_lift_one(args) -> None:
    """Play fixed-N raw-policy vs production-search games with one checkpoint."""

    if args.games <= 0 or args.games % 2:
        raise ValueError("--games must be a positive even number")
    output = Path(args.output)
    result_path = output / "search_lift" / f"{args.label}.json"
    if result_path.is_file() and not args.force:
        print(f"search lift already complete: {result_path}", flush=True)
        return
    checkpoint_path = Path(args.checkpoint)
    spec = _load_agent(
        checkpoint_path, args.label, args.sims, "closed", args.top_k
    )
    model = build_model("transformer", spec.d_model, spec.layers, spec.heads)
    model.load_state_dict(spec.model_state)
    evaluator = Evaluator(
        model,
        args.device,
        args.batch_cap,
        precision=args.precision,
    )
    adapter = rust_flat_batch_adapter(evaluator)

    import seven_wonders_rust as swr

    game_scores: list[float] = []
    pair_scores: list[float] = []
    outcomes = []
    raw_search_disagreements = 0
    decisions = 0
    started = time.monotonic()
    for start in range(0, args.games // 2, args.rust_slots):
        pair_indices = list(
            range(start, min(start + args.rust_slots, args.games // 2))
        )
        seeds = [args.seed_offset + pair for pair in pair_indices]
        first_players = [pair % 2 for pair in pair_indices]
        for search_seat in (0, 1):
            games = rust_games_for_self_play(seeds, first_players)
            actions_played = [0] * len(games)
            live = list(range(len(games)))
            move_index = 0
            while live:
                results = swr.search_many_flat_net(
                    adapter,
                    [games[slot] for slot in live],
                    [
                        seeds[slot] + move_index * 1_000_003
                        for slot in live
                    ],
                    args.batch_cap,
                    1,
                    args.sims,
                    args.top_k,
                    force=False,
                    age_deal_samples=0,
                    puct_root=False,
                )
                for slot, result in zip(live, results):
                    legal = games[slot].legal_action_indices()
                    search_policy = result["policy"]
                    raw_policy = _root_priors_from_digest(result["digest"])
                    if len(raw_policy) != len(legal):
                        raise AssertionError("root priors do not align to legal actions")
                    raw_best = max(range(len(legal)), key=lambda i: raw_policy[i])
                    search_best = max(
                        range(len(legal)), key=lambda i: search_policy[i]
                    )
                    raw_search_disagreements += raw_best != search_best
                    decisions += 1
                    choice = (
                        search_best
                        if games[slot].actor == search_seat
                        else raw_best
                    )
                    games[slot].apply_index(legal[choice])
                    actions_played[slot] += 1
                move_index += 1
                live = [slot for slot in live if not games[slot].is_complete()]

            for pair, seed, first_player, game, actions in zip(
                pair_indices, seeds, first_players, games, actions_played
            ):
                score = (
                    0.5
                    if game.winner is None
                    else float(game.winner == search_seat)
                )
                game_scores.append(score)
                outcomes.append(
                    asdict(
                        MatchOutcome(
                            seed=seed,
                            first_player=first_player,
                            agents=(
                                ("search", "raw")
                                if search_seat == 0
                                else ("raw", "search")
                            ),
                            winner=game.winner,
                            scores=(
                                tuple(game.final_scores)
                                if game.final_scores is not None
                                else None
                            ),
                            victory_type=game.victory_type or "unknown",
                            actions=actions,
                        )
                    )
                )
        # Legs are appended in seat blocks, so pair the matching seed explicitly.
        block = len(pair_indices)
        for offset in range(block):
            pair_scores.append(
                (game_scores[-2 * block + offset] + game_scores[-block + offset])
                / 2.0
            )
    seconds = time.monotonic() - started
    lower, upper = _wilson_interval(sum(pair_scores), len(pair_scores))
    payload = {
        "label": args.label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "precision": args.precision,
        "sims": args.sims,
        "games": len(game_scores),
        "pairs": len(pair_scores),
        "search_score_rate": sum(pair_scores) / len(pair_scores),
        "wilson_lower_fixed_n": lower,
        "wilson_upper_fixed_n": upper,
        "raw_search_action_disagreements": raw_search_disagreements,
        "decisions": decisions,
        "raw_search_action_disagreement_rate": (
            raw_search_disagreements / decisions
        ),
        "stopping": "fixed_n",
        "seed_offset": args.seed_offset,
        "seconds": seconds,
        "pair_scores": pair_scores,
        "outcomes": outcomes,
    }
    _atomic_json(result_path, payload)
    print(
        f"search lift {args.label}: score={payload['search_score_rate']:.3f}, "
        f"CI=[{lower:.3f}, {upper:.3f}], "
        f"action changes={payload['raw_search_action_disagreement_rate']:.1%}",
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


@torch.no_grad()
def forward_bench(args) -> None:
    """Measure a labelled, device-resident production-model forward at b256."""

    output = Path(args.output)
    result_path = (
        output / "forward_bench" / f"{args.label}_{args.precision}.json"
    )
    if result_path.is_file() and not args.force:
        print(f"forward benchmark already complete: {result_path}", flush=True)
        return
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
    evaluator = Evaluator(
        model,
        args.device,
        args.batch_size,
        precision=args.precision,
    )
    packed = torch.load(
        Path(args.cache_dir) / "tensor_examples.pt",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    batch = _tensor_batch(
        packed["storage"], torch.arange(args.batch_size), args.device
    )
    observed_dtypes: list[str] = []

    def record_dtype(_module, _inputs, output_tensor):
        observed_dtypes.append(str(output_tensor.dtype))

    hook = evaluator.model.heads.value.register_forward_hook(record_dtype)
    try:
        for _ in range(args.warmups):
            with evaluator.autocast():
                evaluator.model(batch)
        torch.cuda.synchronize()
        observed_dtypes.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            with evaluator.autocast():
                evaluator.model(batch)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    finally:
        hook.remove()
    expected_dtype = (
        "torch.bfloat16" if args.precision == "bf16" else "torch.float32"
    )
    if set(observed_dtypes) != {expected_dtype}:
        raise AssertionError(
            f"{args.precision} observed dtypes {sorted(set(observed_dtypes))}"
        )
    payload = {
        "label": args.label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "parameters": sum(p.numel() for p in evaluator.model.parameters()),
        "d_model": int(config["d_model"]),
        "layers": int(config["layers"]),
        "heads": heads_from_config(config),
        "precision": args.precision,
        "observed_dtype": expected_dtype,
        "batch_size": args.batch_size,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "seconds": seconds,
        "milliseconds_per_batch": seconds * 1000.0 / args.repetitions,
        "rows_per_second": (
            args.batch_size * args.repetitions / seconds
        ),
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None
        ),
        "scope": (
            "isolated fused model forward with a device-resident real b256 "
            "batch; excludes encoding, transfer, search, and scheduling"
        ),
    }
    _atomic_json(result_path, payload)
    print(
        f"forward {args.label} {args.precision}: "
        f"{payload['rows_per_second']:,.0f} rows/s, "
        f"{payload['milliseconds_per_batch']:.2f} ms/batch",
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

    summary = subparsers.add_parser("summarize-training")
    summary.add_argument("--output", required=True)

    lift = subparsers.add_parser("search-lift-one")
    lift.add_argument("--output", required=True)
    lift.add_argument("--checkpoint", required=True)
    lift.add_argument("--label", required=True)
    lift.add_argument("--games", type=int, default=176)
    lift.add_argument("--sims", type=int, default=64)
    lift.add_argument("--top-k", type=int, default=16)
    lift.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    lift.add_argument("--device", default="cuda")
    lift.add_argument("--rust-slots", type=int, default=48)
    lift.add_argument("--batch-cap", type=int, default=256)
    lift.add_argument("--seed-offset", type=int, required=True)
    lift.add_argument("--force", action="store_true")

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

    forward = subparsers.add_parser("forward-bench")
    forward.add_argument("--output", required=True)
    forward.add_argument("--cache-dir", required=True)
    forward.add_argument("--checkpoint", required=True)
    forward.add_argument("--label", required=True)
    forward.add_argument(
        "--precision", choices=("fp32", "bf16"), required=True
    )
    forward.add_argument("--device", default="cuda")
    forward.add_argument("--batch-size", type=int, default=256)
    forward.add_argument("--warmups", type=int, default=20)
    forward.add_argument("--repetitions", type=int, default=100)
    forward.add_argument("--force", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "fixed-arena-one":
        fixed_arena_one(args)
    elif args.command == "aggregate-fixed-arenas":
        aggregate_fixed_arenas(args)
    elif args.command == "summarize-training":
        summarize_training(args)
    elif args.command == "search-lift-one":
        search_lift_one(args)
    elif args.command == "prepare-holdout":
        prepare_holdout(args)
    elif args.command == "evaluate-holdout":
        evaluate_holdout(args)
    else:
        forward_bench(args)


if __name__ == "__main__":
    main()
