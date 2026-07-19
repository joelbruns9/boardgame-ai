"""P0 step 1: matched-clock NNUE-vs-AlphaZero scaling characterization.

This driver reuses :mod:`games.kingdomino.nnue.match` for every game.  It first
calibrates AZ simulations against measured NNUE decision wall-time at each
deadline, then runs a disjoint paired/seat-swapped screening match.  A small,
disjoint confirmation is run only at clocks whose screening result is close to
the narrow/flat/widen decision boundary or is visibly non-monotonic.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from games.kingdomino.nnue.match import (
    _sha256,
    nnue_participant,
    run_paired,
    write_report,
)
from games.kingdomino.promotion import wilson_lower_bound
from games.kingdomino.round_robin_eval import build_open_loop_checkpoint_participant


DEFAULT_CLOCKS = (0.1, 0.5, 2.0, 10.0)
DEFAULT_NNUE = Path("runs/kingdomino/nnue_data/sparse_v3_pilot.knnue")
DEFAULT_AZ = Path("runs/kingdomino/best_checkpoint/current_best.pt")
DEFAULT_OUT = Path("runs/kingdomino/nnue_loop/clock_scaling_p0_step1.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _wilson_interval(points: float, games: int) -> tuple[float, float]:
    lower = wilson_lower_bound(points, games)
    upper = 1.0 - wilson_lower_bound(games - points, games)
    return lower, upper


def _relative_clock_error(report: dict) -> float:
    timing = report["timing"]
    nnue = timing["NNUE"]["decision_mean_seconds"]
    az = timing["AlphaZero"]["decision_mean_seconds"]
    if nnue <= 0.0:
        return math.inf
    return abs(az / nnue - 1.0)


def _clock_match_audit(report: dict, tolerance: float) -> dict:
    nnue = report["timing"]["NNUE"]["decision_mean_seconds"]
    az = report["timing"]["AlphaZero"]["decision_mean_seconds"]
    ratio = az / nnue if nnue > 0.0 else None
    error = abs(ratio - 1.0) if ratio is not None else None
    return {
        "tolerance_fraction": tolerance,
        "az_to_nnue_mean_ratio": ratio,
        "relative_error": error,
        "within_tolerance": error is not None and error <= tolerance,
    }


def _settings(
    *,
    provenance: dict,
    clock: float,
    az_sims: int,
    max_depth: int,
    chance_samples: int,
    device: str,
    c_puct: float,
    sweep_index: int,
    tag: str,
    seed_start: int,
    paired_seeds: int,
    calibration_trial: int | None = None,
) -> dict:
    settings = {
        "kind": "nnue_vs_alphazero_clock_scaling_p0_step1",
        **provenance,
        "nnue_move_secs": clock,
        "az_sims": az_sims,
        "device": device,
        "c_puct": c_puct,
        "max_depth": max_depth,
        "chance_samples": chance_samples,
        "full_width_ordering": True,
        "selective_width": None,
        "selective_root_width": None,
        "sweep_index": sweep_index,
        "clock_seconds": clock,
        "set_tag": tag,
        "seed_start": seed_start,
        "paired_seeds": paired_seeds,
    }
    if calibration_trial is not None:
        settings["calibration_trial"] = calibration_trial
    return settings


def _run_match(
    *,
    nnue_path: Path,
    az_path: Path,
    provenance: dict,
    clock: float,
    az_sims: int,
    max_depth: int,
    chance_samples: int,
    device: str,
    c_puct: float,
    sweep_index: int,
    tag: str,
    seed_start: int,
    paired_seeds: int,
    calibration_trial: int | None = None,
) -> dict:
    # Measurement only: retain the operational public-state chance model,
    # quantized pilot artifact, full-width ordering, and no selective width.
    nnue = nnue_participant(
        "NNUE",
        nnue_path,
        move_secs=clock,
        max_depth=max_depth,
        chance_samples=chance_samples,
        full_width_ordering=True,
        selective_width=None,
        selective_root_width=None,
    )
    az = build_open_loop_checkpoint_participant(
        az_path,
        name="AlphaZero",
        device=device,
        sims=az_sims,
        c_puct=c_puct,
        temperature=0.0,
        channels_override=None,
        blocks_override=None,
        bilinear_dim_override=None,
    )
    return run_paired(
        nnue,
        az,
        seed_start=seed_start,
        paired_seeds=paired_seeds,
        settings=_settings(
            provenance=provenance,
            clock=clock,
            az_sims=az_sims,
            max_depth=max_depth,
            chance_samples=chance_samples,
            device=device,
            c_puct=c_puct,
            sweep_index=sweep_index,
            tag=tag,
            seed_start=seed_start,
            paired_seeds=paired_seeds,
            calibration_trial=calibration_trial,
        ),
    )


def _next_sims(trials: list[dict], current_sims: int) -> int:
    """Choose another simulation count from measured means, not a cost guess."""
    samples = [
        (
            trial["az_sims"],
            trial["timing"]["AlphaZero"]["decision_mean_seconds"],
            trial["timing"]["NNUE"]["decision_mean_seconds"],
        )
        for trial in trials
    ]
    target = samples[-1][2]
    below = [sample for sample in samples if sample[1] < target]
    above = [sample for sample in samples if sample[1] > target]
    if below and above:
        lo = max(below, key=lambda sample: sample[1])
        hi = min(above, key=lambda sample: sample[1])
        if hi[1] > lo[1]:
            fraction = (target - lo[1]) / (hi[1] - lo[1])
            proposed = round(lo[0] + fraction * (hi[0] - lo[0]))
        else:
            proposed = current_sims
    else:
        az_mean = samples[-1][1]
        ratio = target / az_mean if az_mean > 0.0 else 2.0
        proposed = round(current_sims * min(8.0, max(0.125, ratio)))

    tried = {sample[0] for sample in samples}
    proposed = max(1, proposed)
    if proposed in tried:
        direction = 1 if samples[-1][1] < target else -1
        proposed = max(1, proposed + direction)
        while proposed in tried and proposed > 1:
            proposed += direction
    return max(1, proposed)


def calibrate_clock(
    *,
    initial_sims: int,
    tolerance: float,
    max_trials: int,
    run_kwargs: dict,
) -> dict:
    trials: list[dict] = []
    sims = max(1, initial_sims)
    for trial_index in range(max_trials):
        print(
            f"calibration {run_kwargs['clock']:g}s trial {trial_index + 1}: "
            f"AZ sims={sims}",
            flush=True,
        )
        match = _run_match(
            az_sims=sims,
            tag="calibration",
            calibration_trial=trial_index,
            **run_kwargs,
        )
        trial = {
            "trial_index": trial_index,
            "az_sims": sims,
            "timing": match["timing"],
            "clock_match": _clock_match_audit(match, tolerance),
            "pair": match["pair"],
            "wall_seconds": match["wall_seconds"],
            "settings": match["settings"],
        }
        trials.append(trial)
        audit = trial["clock_match"]
        print(
            f"  means NNUE={match['timing']['NNUE']['decision_mean_seconds']:.3f}s "
            f"AZ={match['timing']['AlphaZero']['decision_mean_seconds']:.3f}s "
            f"error={audit['relative_error']:.1%}",
            flush=True,
        )
        if trial["clock_match"]["within_tolerance"]:
            break
        sims = _next_sims(trials, sims)

    chosen = min(
        trials,
        key=lambda trial: (
            trial["clock_match"]["relative_error"]
            if trial["clock_match"]["relative_error"] is not None
            else math.inf
        ),
    )
    return {
        "chosen_az_sims": chosen["az_sims"],
        "matched": chosen["clock_match"]["within_tolerance"],
        "chosen_trial_index": chosen["trial_index"],
        "trials": trials,
    }


def _combine_pairs(reports: Iterable[dict]) -> dict:
    pairs = [report["pair"] for report in reports]
    games = sum(pair["games"] for pair in pairs)
    wins = sum(pair["a_wins"] for pair in pairs)
    draws = sum(pair["draws"] for pair in pairs)
    losses = sum(pair["b_wins"] for pair in pairs)
    a_score = sum(pair["a_score_sum"] for pair in pairs)
    b_score = sum(pair["b_score_sum"] for pair in pairs)
    points = wins + 0.5 * draws
    lower, upper = _wilson_interval(points, games)
    return {
        "games": games,
        "nnue_wins": wins,
        "draws": draws,
        "nnue_losses": losses,
        "nnue_points": points,
        "nnue_points_rate": points / games,
        "nnue_points_lcb_95": lower,
        "nnue_points_ucb_95": upper,
        "nnue_score_sum": a_score,
        "az_score_sum": b_score,
        "avg_margin_nnue": (a_score - b_score) / games,
    }


def _screening_verdict(rows: list[dict], flat_margin_points: float) -> dict:
    valid = [row for row in rows if row["included_in_verdict"]]
    if len(valid) != len(rows):
        return {
            "classification": "inconclusive_clock_mismatch",
            "explanation": "At least one realized screening clock exceeded the matching tolerance.",
        }
    clocks = np.asarray([row["clock_seconds"] for row in valid], dtype=np.float64)
    margins = np.asarray(
        [row["performance"]["avg_margin_nnue"] for row in valid], dtype=np.float64
    )
    slope = float(np.polyfit(np.log10(clocks), margins, 1)[0])
    endpoint_delta = float(margins[-1] - margins[0])
    if endpoint_delta > flat_margin_points and slope > 0.0:
        classification = "narrow"
    elif endpoint_delta < -flat_margin_points and slope < 0.0:
        classification = "widen"
    else:
        classification = "stay_flat"
    return {
        "classification": classification,
        "metric": "NNUE average score margin versus log10(matched clock)",
        "flat_band_endpoint_margin_points": flat_margin_points,
        "endpoint_margin_delta_points": endpoint_delta,
        "margin_slope_points_per_clock_decade": slope,
        "explanation": (
            f"NNUE margin changed by {endpoint_delta:+.2f} points from "
            f"{clocks[0]:g}s to {clocks[-1]:g}s; fitted slope {slope:+.2f} "
            "points per clock decade."
        ),
    }


def _auto_confirmation(
    rows: list[dict], flat_margin_points: float, max_clocks: int = 2
) -> tuple[list[float], dict[str, str]]:
    verdict = _screening_verdict(rows, flat_margin_points)
    if verdict["classification"] == "inconclusive_clock_mismatch":
        return [], {}
    margins = [row["performance"]["avg_margin_nnue"] for row in rows]
    endpoint_delta = margins[-1] - margins[0]
    reasons: dict[float, str] = {}

    # Endpoints determine the flat-band classification; confirm them only when
    # the observed endpoint delta is close enough that more games could cross it.
    if abs(endpoint_delta) <= 2.0 * flat_margin_points:
        reasons[rows[-1]["clock_seconds"]] = "endpoint trend is near the flat-band boundary"
        reasons[rows[0]["clock_seconds"]] = "baseline endpoint anchors a boundary-sensitive trend"

    expected_sign = 0 if endpoint_delta == 0 else (1 if endpoint_delta > 0 else -1)
    for index, delta in enumerate(np.diff(margins)):
        sign = 0 if delta == 0 else (1 if delta > 0 else -1)
        if expected_sign and sign and sign != expected_sign:
            clock = rows[index + 1]["clock_seconds"]
            reasons.setdefault(clock, "screening curve is non-monotonic at this clock")

    ordered = sorted(
        reasons,
        key=lambda clock: (
            0 if clock == rows[-1]["clock_seconds"] else 1,
            -clock,
        ),
    )[:max_clocks]
    return ordered, {str(clock): reasons[clock] for clock in ordered}


def _summary_row(clock_result: dict) -> dict:
    screening = clock_result["screening"]
    nnue_timing = screening["timing"]["NNUE"]
    az_timing = screening["timing"]["AlphaZero"]
    search = nnue_timing.get("search")
    return {
        "clock_seconds": clock_result["clock_seconds"],
        **clock_result["performance"],
        "nnue_completed_depth_mean": (
            search["completed_depth_mean"] if search is not None else None
        ),
        "nnue_completed_depth_median": (
            search["completed_depth_median"] if search is not None else None
        ),
        "nnue_nodes_mean": search["nodes_mean"] if search is not None else None,
        "nnue_last_iteration_nodes_mean": (
            search["last_iteration_nodes_mean"] if search is not None else None
        ),
        "nnue_timeout_rate": search["timeout_rate"] if search is not None else None,
        "nnue_chance_share": search["chance_share"] if search is not None else None,
        "az_sims": clock_result["az_sims"],
        "nnue_decision_mean_seconds": nnue_timing["decision_mean_seconds"],
        "nnue_decision_p95_seconds": nnue_timing["decision_p95_seconds"],
        "az_decision_mean_seconds": az_timing["decision_mean_seconds"],
        "az_decision_p95_seconds": az_timing["decision_p95_seconds"],
        "clock_match": clock_result["screening_clock_match"],
        "confirmation_games": (
            clock_result["confirmation"]["pair"]["games"]
            if clock_result["confirmation"] is not None
            else 0
        ),
        "performance_sets": (
            ["screening", "confirmation"]
            if clock_result["confirmation"] is not None
            else ["screening"]
        ),
        # Exact confirmation timing/telemetry remains available in the detailed
        # per-clock block. Quantiles cannot be merged from aggregate summaries,
        # so this compact row names its telemetry source explicitly.
        "telemetry_set": "screening",
    }


def run_sweep(args: argparse.Namespace) -> dict:
    clocks = sorted(set(float(clock) for clock in args.clocks))
    if clocks != sorted(DEFAULT_CLOCKS) and not args.allow_nonstandard_clocks:
        raise ValueError(f"P0 step 1 clocks must be {DEFAULT_CLOCKS}")
    if any(clock <= 0.0 or not math.isfinite(clock) for clock in clocks):
        raise ValueError("clocks must be finite and positive")
    if not 0.0 < args.clock_tolerance < 1.0:
        raise ValueError("clock tolerance must be in (0, 1)")
    if (
        args.screening_paired_seeds < 1
        or args.calibration_paired_seeds < 1
        or args.confirmation_paired_seeds < 1
    ):
        raise ValueError("all paired seed counts must be >= 1")
    if args.max_calibration_trials < 1:
        raise ValueError("max calibration trials must be >= 1")
    if args.baseline_az_sims < 1:
        raise ValueError("baseline AZ sims must be >= 1")
    if args.flat_margin_points < 0.0 or not math.isfinite(args.flat_margin_points):
        raise ValueError("flat margin points must be finite and non-negative")

    nnue_path = Path(args.nnue).resolve()
    az_path = Path(args.az_checkpoint).resolve()
    if not nnue_path.is_file() or not az_path.is_file():
        raise FileNotFoundError("NNUE artifact and AZ checkpoint must both exist")
    provenance = {
        "nnue": str(nnue_path),
        "nnue_sha256": _sha256(nnue_path),
        "az_checkpoint": str(az_path),
        "az_checkpoint_sha256": _sha256(az_path),
    }

    initial_overrides = {}
    for item in args.initial_az_sims:
        clock_text, sims_text = item.split("=", 1)
        initial_overrides[float(clock_text)] = int(sims_text)

    clock_results = []
    previous_clock = None
    previous_sims = None
    for index, clock in enumerate(clocks):
        if clock in initial_overrides:
            initial_sims = initial_overrides[clock]
        elif previous_sims is not None:
            initial_sims = max(1, round(previous_sims * clock / previous_clock))
        else:
            initial_sims = max(1, round(args.baseline_az_sims * clock / clocks[0]))

        common = {
            "nnue_path": nnue_path,
            "az_path": az_path,
            "provenance": provenance,
            "clock": clock,
            "max_depth": args.max_depth,
            "chance_samples": args.chance_samples,
            "device": args.device,
            "c_puct": args.c_puct,
            "sweep_index": index,
        }
        calibration = calibrate_clock(
            initial_sims=initial_sims,
            tolerance=args.clock_tolerance,
            max_trials=args.max_calibration_trials,
            run_kwargs={
                **common,
                "seed_start": args.calibration_seed_start,
                "paired_seeds": args.calibration_paired_seeds,
            },
        )
        az_sims = calibration["chosen_az_sims"]
        print(
            f"screening {clock:g}s: AZ sims={az_sims}, "
            f"paired seeds={args.screening_paired_seeds}",
            flush=True,
        )
        screening = _run_match(
            **common,
            az_sims=az_sims,
            tag="screening",
            seed_start=args.screening_seed_start,
            paired_seeds=args.screening_paired_seeds,
        )
        audit = _clock_match_audit(screening, args.clock_tolerance)
        print(
            f"  screening points={screening['pair']['a_points_rate']:.3f}, "
            f"margin={screening['pair']['avg_margin_a']:+.2f}, "
            f"clock error={audit['relative_error']:.1%}",
            flush=True,
        )
        clock_results.append({
            "sweep_index": index,
            "clock_seconds": clock,
            "az_sims": az_sims,
            "calibration": calibration,
            "screening_clock_match": audit,
            "screening": screening,
            "confirmation": None,
            "confirmation_clock_match": None,
            "performance": _combine_pairs([screening]),
            "included_in_verdict": audit["within_tolerance"],
        })
        previous_clock, previous_sims = clock, az_sims

    if args.confirmation_clocks is None:
        confirmation_clocks, confirmation_reasons = _auto_confirmation(
            clock_results, args.flat_margin_points
        )
        confirmation_mode = "automatic_decision_sensitive_only"
    else:
        confirmation_clocks = sorted(set(args.confirmation_clocks))
        confirmation_reasons = {
            str(clock): "explicitly requested confirmation clock"
            for clock in confirmation_clocks
        }
        confirmation_mode = "explicit"

    by_clock = {result["clock_seconds"]: result for result in clock_results}
    unknown = set(confirmation_clocks) - set(by_clock)
    if unknown:
        raise ValueError(f"confirmation clocks not present in sweep: {sorted(unknown)}")
    for confirm_index, clock in enumerate(confirmation_clocks):
        result = by_clock[clock]
        print(
            f"confirmation {clock:g}s: AZ sims={result['az_sims']}, "
            f"paired seeds={args.confirmation_paired_seeds}",
            flush=True,
        )
        confirmation = _run_match(
            nnue_path=nnue_path,
            az_path=az_path,
            provenance=provenance,
            clock=clock,
            az_sims=result["az_sims"],
            max_depth=args.max_depth,
            chance_samples=args.chance_samples,
            device=args.device,
            c_puct=args.c_puct,
            sweep_index=result["sweep_index"],
            tag="confirmation",
            seed_start=args.confirmation_seed_start + confirm_index * args.confirmation_paired_seeds,
            paired_seeds=args.confirmation_paired_seeds,
        )
        audit = _clock_match_audit(confirmation, args.clock_tolerance)
        result["confirmation"] = confirmation
        result["confirmation_clock_match"] = audit
        if audit["within_tolerance"] and result["included_in_verdict"]:
            result["performance"] = _combine_pairs([result["screening"], confirmation])
        else:
            result["included_in_verdict"] = False

    verdict = _screening_verdict(clock_results, args.flat_margin_points)
    summary = [_summary_row(result) for result in clock_results]
    return {
        "schema": "kingdomino_nnue_clock_scaling_p0_step1_v1",
        "created_utc": _utc_now(),
        "question": "Does the NNUE-vs-AZ gap narrow, stay flat, or widen across matched clocks?",
        "verdict": verdict,
        "settings": {
            **provenance,
            "clocks_seconds": clocks,
            "clock_tolerance_fraction": args.clock_tolerance,
            "calibration_paired_seeds": args.calibration_paired_seeds,
            "calibration_seed_start": args.calibration_seed_start,
            "screening_paired_seeds": args.screening_paired_seeds,
            "screening_seed_start": args.screening_seed_start,
            "confirmation_paired_seeds": args.confirmation_paired_seeds,
            "confirmation_seed_start": args.confirmation_seed_start,
            "device": args.device,
            "c_puct": args.c_puct,
            "max_depth": args.max_depth,
            "chance_samples": args.chance_samples,
            "full_width_ordering": True,
            "selective_width": None,
            "reserved_test_split_opened": False,
            "chance_share_definition": "sum(chance_nodes) / (sum(nodes) + sum(chance_nodes))",
        },
        "confirmation_policy": {
            "mode": confirmation_mode,
            "clocks_seconds": confirmation_clocks,
            "reasons": confirmation_reasons,
        },
        "summary": summary,
        "clocks": clock_results,
    }


def _print_summary(report: dict, out: Path) -> None:
    print(f"report: {out}")
    print("clock  points (95% Wilson)  margin   depth  NNUE mean  AZ mean  AZ sims")
    for row in report["summary"]:
        depth = row["nnue_completed_depth_mean"]
        depth_text = f"{depth:5.2f}" if depth is not None else " null"
        print(
            f"{row['clock_seconds']:>5g}s "
            f"{row['nnue_points_rate']:.3f} "
            f"[{row['nnue_points_lcb_95']:.3f}, {row['nnue_points_ucb_95']:.3f}] "
            f"{row['avg_margin_nnue']:+7.2f} {depth_text} "
            f"{row['nnue_decision_mean_seconds']:9.3f}s "
            f"{row['az_decision_mean_seconds']:7.3f}s "
            f"{row['az_sims']:7d}"
        )
    print(json.dumps(report["verdict"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnue", default=str(DEFAULT_NNUE))
    parser.add_argument("--az-checkpoint", default=str(DEFAULT_AZ))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--clocks", nargs="+", type=float, default=list(DEFAULT_CLOCKS))
    parser.add_argument("--allow-nonstandard-clocks", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--chance-samples", type=int, default=16)
    parser.add_argument("--clock-tolerance", type=float, default=0.15)
    parser.add_argument("--baseline-az-sims", type=int, default=13)
    parser.add_argument(
        "--initial-az-sims",
        action="append",
        default=[],
        metavar="CLOCK=SIMS",
        help="optional measured-calibration starting point override",
    )
    parser.add_argument("--max-calibration-trials", type=int, default=4)
    parser.add_argument("--calibration-paired-seeds", type=int, default=2)
    parser.add_argument("--calibration-seed-start", type=int, default=20260600)
    parser.add_argument("--screening-paired-seeds", type=int, default=4)
    parser.add_argument("--screening-seed-start", type=int, default=20260800)
    parser.add_argument("--confirmation-paired-seeds", type=int, default=4)
    parser.add_argument("--confirmation-seed-start", type=int, default=20261800)
    parser.add_argument(
        "--confirmation-clocks",
        nargs="*",
        type=float,
        default=None,
        help="omit for automatic decision-sensitive confirmation; pass empty to disable",
    )
    parser.add_argument(
        "--flat-margin-points",
        type=float,
        default=5.0,
        help="endpoint margin-change band classified as flat",
    )
    args = parser.parse_args()
    report = run_sweep(args)
    out = write_report(report, args.out)
    _print_summary(report, out)


if __name__ == "__main__":
    main()
