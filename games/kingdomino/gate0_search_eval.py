"""Gate 0: fixed-network progressive-search versus open-loop search.

The checkpoint, network evaluator, seeds, row budget, and all ordinary MCTS
parameters are identical. Only the deck-8/12 progressive search treatment is
rotated between seats. This isolates whether the proposed teacher search
improves play before any v2 training work begins.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

from games.kingdomino.elo_rating import (
    EloConfig,
    play_search_ab_games_with_diagnostics,
)
from games.kingdomino.promotion import (
    _net_from_checkpoint,
    match_stats_from_pair,
    pair_scores_from_games,
    sha256_file,
)
from games.kingdomino.symmetric_checkpoint_eval import (
    _atomic_games_jsonl,
    _atomic_json,
    _now_iso,
    _parse_decks,
    progressive_diagnostic_errors,
)


MIN_GATE0_GAMES = 2048


def _student_t_critical(probability: float, degrees_freedom: int) -> float:
    """Accurate large-sample t quantile without adding a SciPy dependency."""
    if not 0.5 < float(probability) < 1.0:
        raise ValueError("probability must be between 0.5 and 1")
    if int(degrees_freedom) < 1:
        raise ValueError("degrees_freedom must be positive")
    z = NormalDist().inv_cdf(float(probability))
    df = float(degrees_freedom)
    # Cornish-Fisher expansion. Gate 0 has df=1023, where the omitted term is
    # negligible; retain the method name and inputs in the result artifact.
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
    )


def paired_score_summary(
        scores: Iterable[float], confidence: float = 0.95) -> dict[str, Any]:
    """Pair-aware t intervals over actual per-seed scores in {0, 0.5, 1}."""
    values = [float(value) for value in scores]
    if len(values) < 2:
        raise ValueError("paired uncertainty requires at least two seat pairs")
    if any(value not in (0.0, 0.5, 1.0) for value in values):
        raise ValueError("paired scores must be in {0, 0.5, 1}")
    if not 0.5 < float(confidence) < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")

    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / math.sqrt(len(values))
    df = len(values) - 1
    one_sided_t = _student_t_critical(float(confidence), df)
    two_sided_t = _student_t_critical((1.0 + float(confidence)) / 2.0, df)

    return {
        "method": "paired_score_t_interval_cornish_fisher",
        "observation_unit": "complete_same_seed_seat_pair",
        "confidence": float(confidence),
        "pairs": len(values),
        "mean": mean,
        "sample_standard_deviation": sample_sd,
        "standard_error": standard_error,
        "one_sided_lower": max(0.0, mean - one_sided_t * standard_error),
        "one_sided_upper": min(1.0, mean + one_sided_t * standard_error),
        "two_sided_interval": [
            max(0.0, mean - two_sided_t * standard_error),
            min(1.0, mean + two_sided_t * standard_error),
        ],
    }


def gate0_verdict(interval: dict[str, Any]) -> str:
    """Apply the preregistered hard Gate-0 decision rule."""
    if float(interval["one_sided_lower"]) > 0.5:
        return "proceed"
    if float(interval["one_sided_upper"]) <= 0.5:
        return "fail"
    return "inconclusive"


def gate0_diagnostic_errors(
        diagnostics: dict[str, Any], decks: tuple[int, ...],
        paired_seeds: int) -> list[str]:
    """Fail closed if the treatment was absent or leaked onto both seats."""
    errors = progressive_diagnostic_errors(diagnostics, decks)
    orientations = diagnostics.get("orientations", [])
    if not isinstance(orientations, list) or len(orientations) != 2:
        return errors + ["expected exactly two diagnostic orientations"]

    # Each player owns two roots at each treated deck count. The mechanism
    # counter increments only if a simulation actually crosses the chance
    # boundary, so depth may make it lower; it can never exceed this one-seat
    # capacity unless the treatment leaked onto the control seat.
    maximum_per_orientation = 2 * int(paired_seeds)
    for index, (row, expected_seat) in enumerate(zip(orientations, (0, 1))):
        actual_seat = row.get("progressive_chance_seat")
        if actual_seat != expected_seat:
            errors.append(
                f"orientation {index} progressive seat is {actual_seat}; "
                f"expected {expected_seat}"
            )
        orientation_errors = progressive_diagnostic_errors(row, decks)
        errors.extend(f"orientation {index}: {error}"
                      for error in orientation_errors)
        for deck in decks:
            key = f"progressive_chance_search_count_deck{deck}"
            actual = int(row.get(key, 0))
            if actual > maximum_per_orientation:
                errors.append(
                    f"orientation {index} {key}={actual}; expected "
                    f"at most {maximum_per_orientation} for exactly one "
                    "treated seat"
                )
    return errors


def run_gate0_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if int(args.games) <= 0 or int(args.games) % 2:
        raise ValueError("--games must be a positive even number (seat pairs)")
    if int(args.games) < 4:
        raise ValueError("--games must contain at least two complete seat pairs")
    if int(args.games) < MIN_GATE0_GAMES and not bool(args.allow_smoke):
        raise ValueError(
            f"Gate 0 requires at least {MIN_GATE0_GAMES} games; use "
            "--allow-smoke only for mechanism verification"
        )
    existing = [
        path for path in (
            output_dir / "manifest.json",
            output_dir / "games.jsonl",
            output_dir / "result.json",
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite evaluation artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    decks = tuple(args.chance_progressive_decks)
    paired_seeds = int(args.games) // 2
    cfg = EloConfig(
        games_per_anchor=paired_seeds,
        sims=int(args.sims),
        device=str(args.device),
        n_slots=int(args.batch_slots),
        leaf_batch=int(args.leaf_batch),
        c_puct=float(args.c_puct),
        fpu=float(args.fpu),
        margin_gain=float(args.margin_gain),
        alpha=float(args.alpha),
        seed=int(args.seed),
        verbose=False,
        progressive_chance_decks=decks,
        progressive_chance_width_schedule=str(args.chance_width_schedule),
        progressive_chance_n_init=int(args.chance_n_init),
        progressive_chance_d_min=int(args.chance_d_min),
        progressive_chance_deck8_cap=int(args.chance_deck8_cap),
        progressive_chance_deck12_cap=int(args.chance_deck12_cap),
        progressive_chance_max_init_fraction=float(
            args.chance_max_init_fraction),
    )
    checkpoint_hash = sha256_file(checkpoint)
    manifest = {
        "schema_version": 1,
        "gate": "gate0_fixed_network_search_ab",
        "status": "running",
        "started_at": _now_iso(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "network_is_identical": True,
        "network_load_count": 1,
        "treatment": "chance_progressive",
        "control": "open_loop",
        "search_is_asymmetric": True,
        "games": int(args.games),
        "paired_seeds": paired_seeds,
        "execution_mode": (
            "gate" if int(args.games) >= MIN_GATE0_GAMES else "smoke_only"),
        "seed_start": int(args.seed),
        "decision_rule": {
            "confidence": float(args.confidence),
            "proceed": "one_sided_lower > 0.5",
            "fail": "one_sided_upper <= 0.5",
            "otherwise": "inconclusive",
        },
        "parameters": {
            "sims_nn_rows_per_move": cfg.sims,
            "batch_slots": cfg.n_slots,
            "leaf_batch": cfg.leaf_batch,
            "c_puct": cfg.c_puct,
            "fpu": cfg.fpu,
            "margin_gain": cfg.margin_gain,
            "alpha": cfg.alpha,
            "progressive_chance_decks": list(decks),
            "progressive_chance_width_schedule":
                cfg.progressive_chance_width_schedule,
            "progressive_chance_n_init": cfg.progressive_chance_n_init,
            "progressive_chance_d_min": cfg.progressive_chance_d_min,
            "progressive_chance_deck8_cap": cfg.progressive_chance_deck8_cap,
            "progressive_chance_deck12_cap": cfg.progressive_chance_deck12_cap,
            "progressive_chance_max_init_fraction":
                cfg.progressive_chance_max_init_fraction,
            "orientation_progressive_seats": [0, 1],
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)

    print(
        f"Loading one checkpoint and running {args.games} games "
        f"({paired_seeds} paired seeds): progressive search vs open loop...",
        flush=True,
    )
    net = _net_from_checkpoint(checkpoint, cfg.device)
    started = time.perf_counter()
    pair, games, diagnostics = play_search_ab_games_with_diagnostics(
        net,
        "chance_progressive",
        "open_loop",
        paired_seeds,
        int(args.seed),
        cfg,
    )
    elapsed = time.perf_counter() - started
    stats = match_stats_from_pair(
        pair, games, candidate_name="chance_progressive")
    pair_scores = pair_scores_from_games(games, "chance_progressive")
    uncertainty = paired_score_summary(
        pair_scores, confidence=float(args.confidence))
    verdict = (
        gate0_verdict(uncertainty)
        if int(args.games) >= MIN_GATE0_GAMES else "smoke_only"
    )
    errors = gate0_diagnostic_errors(diagnostics, decks, paired_seeds)

    result = {
        **manifest,
        "status": "complete" if not errors else "invalid",
        "completed_at": _now_iso(),
        "elapsed_seconds": elapsed,
        "games_per_second": int(args.games) / elapsed if elapsed else 0.0,
        "valid": not errors,
        "errors": errors,
        "verdict": verdict if not errors else "invalid",
        "match": asdict(stats),
        "paired_uncertainty": uncertainty,
        "raw_pair": asdict(pair),
        "progressive_diagnostics": diagnostics,
        "artifacts": {
            "manifest": "manifest.json",
            "games": "games.jsonl",
            "result": "result.json",
        },
    }
    _atomic_games_jsonl(output_dir / "games.jsonl", games)
    _atomic_json(output_dir / "result.json", result)
    print(json.dumps({
        "valid": result["valid"],
        "verdict": result["verdict"],
        "games": stats.games,
        "paired_seeds": stats.pairs,
        "progressive_pair_score_rate": stats.pair_score_rate,
        "paired_uncertainty": uncertainty,
        "mean_margin": stats.mean_margin,
        "elapsed_seconds": elapsed,
        "progressive_diagnostics": diagnostics,
        "result": str(output_dir / "result.json"),
    }, indent=2, sort_keys=True), flush=True)
    if errors:
        raise RuntimeError("Gate 0 evaluation invalid: " + "; ".join(errors))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gate 0 fixed-network progressive-versus-open-loop paired match"
        ))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="allow fewer than 2,048 games; never emits a strength verdict",
    )
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-slots", type=int, default=48)
    parser.add_argument("--leaf-batch", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20360000)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--fpu", type=float, default=-0.2)
    parser.add_argument("--margin-gain", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--chance-progressive-decks", type=_parse_decks, default=(8, 12))
    parser.add_argument("--chance-width-schedule", default="4,8,16,32,64,70")
    parser.add_argument("--chance-n-init", type=int, default=2)
    parser.add_argument("--chance-d-min", type=int, default=4)
    parser.add_argument("--chance-deck8-cap", type=int, default=16)
    parser.add_argument("--chance-deck12-cap", type=int, default=16)
    parser.add_argument("--chance-max-init-fraction", type=float, default=0.25)
    return parser


def main() -> None:
    run_gate0_evaluation(build_parser().parse_args())


if __name__ == "__main__":
    main()
