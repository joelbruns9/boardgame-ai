"""Fixed-eval depth-conversion fitness probe for Kingdomino.

The independent variable is the completed full-width search depth. Every
depth-controlled participant uses direct fixed-depth Rust make/unmake recursion
with the same ``pick_aware`` evaluation and chance model. Telemetry must show
that every searched move ran at its requested depth or the experiment is
classified as invalid.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import time

import kingdomino_rust as kr

from games.kingdomino.nnue.match import _sha256, run_paired, write_report
from games.kingdomino.promotion import wilson_lower_bound
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.game import Phase
from games.kingdomino.round_robin_eval import (
    Participant,
    build_open_loop_checkpoint_participant,
)
from games.kingdomino.rust_expectiminimax import (
    OperationalRustSearchBot,
    RustExpectiminimax,
    pick_aware,
)


DEFAULT_STEP1 = Path("runs/kingdomino/nnue_loop/clock_scaling_p0_step1.json")
DEFAULT_AZ = Path("runs/kingdomino/best_checkpoint/current_best.pt")
DEFAULT_OUT = Path("runs/kingdomino/nnue_loop/depth_conversion.json")
DEFAULT_DEPTHS = (1, 2, 3)
DEFAULT_AZ_SIMS = (65, 260)
WITHIN_ROW_CAVEAT = (
    "Depth 1-3 is within-row tactical depth: prior telemetry shows depth near 3 "
    "rarely crosses a round/chance boundary. If depth does not help even inside "
    "the revealed row, the negative fitness result is strong. If useful depth "
    "would begin only beyond the next unknown row, that depth is not practically "
    "reachable by this full-width design and is itself a negative fitness result."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _wilson_interval(points: float, games: int) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    return (
        wilson_lower_bound(points, games),
        1.0 - wilson_lower_bound(games - points, games),
    )


class FixedDepthPickAwareBot:
    """Strict fixed-ply adapter around the direct make/unmake oracle."""

    def __init__(self, depth: int, *, chance_samples: int, enum_cap: int):
        self.depth = depth
        self.search = RustExpectiminimax(
            depth=depth,
            chance_samples=chance_samples,
            enum_cap=enum_cap,
            eval_fn=pick_aware,
        )
        self.last_report = None

    def choose_action(self, state, actions=None, rng=None):
        legal = state.legal_actions() if actions is None else actions
        if len(legal) == 1:
            return legal[0]
        rust_state = _rust_state_from_python(state)
        if rust_state is None:
            raise RuntimeError("could not convert GameState to RustGameState")
        t0 = time.perf_counter()
        rust_action = self.search.choose_action(kr.SearchEngine(rust_state), rng=rng)
        elapsed = time.perf_counter() - t0
        self.last_report = SimpleNamespace(
            completed_depth=self.depth,
            timed_out=False,
            elapsed_secs=elapsed,
            nodes=self.search.nodes,
            last_iteration_nodes=self.search.nodes,
            chance_nodes=None,
        )
        rust_placement, rust_pick = rust_action
        if state.phase == Phase.INITIAL_SELECTION:
            for action in legal:
                if action.domino_id == rust_pick:
                    return action
        else:
            domino_id = state.pending_claims[state.actor_index].domino_id
            rust_key = OperationalRustSearchBot._placement_key(
                rust_placement, domino_id
            )
            for action in legal:
                placement = action.placement
                py_tuple = None if placement is None else (
                    placement.x1,
                    placement.y1,
                    placement.x2,
                    placement.y2,
                    placement.flipped,
                )
                if (
                    action.pick_domino_id == rust_pick
                    and OperationalRustSearchBot._placement_key(
                        py_tuple, domino_id
                    ) == rust_key
                ):
                    return action
        raise RuntimeError(f"fixed-depth search returned illegal action {rust_action!r}")


def _pick_aware_participant(
    depth: int, *, chance_samples: int, enum_cap: int
) -> Participant:
    name = f"pick_aware_d{depth}"

    def make_bot() -> FixedDepthPickAwareBot:
        return FixedDepthPickAwareBot(
            depth,
            chance_samples=chance_samples,
            enum_cap=enum_cap,
        )

    return Participant(
        name=name,
        make_bot=make_bot,
        kind="operational_pick_aware_fixed_depth",
        source="RustExpectiminimax(eval_fn=pick_aware)",
    )


def _depth_completion(timing: dict, target_depth: int) -> dict:
    search = timing.get("search")
    if not search:
        return {
            "target_depth": target_depth,
            "report_count": 0,
            "completed_depth_min": None,
            "completed_depth_max": None,
            "completed_depth_histogram": {},
            "timeout_count": None,
            "all_nonforced_moves_completed_target": False,
        }
    completed = (
        search["report_count"] > 0
        and search["completed_depth_min"] == target_depth
        and search["completed_depth_max"] == target_depth
        and search["timeout_count"] == 0
    )
    return {
        "target_depth": target_depth,
        "report_count": search["report_count"],
        "completed_depth_min": search["completed_depth_min"],
        "completed_depth_max": search["completed_depth_max"],
        "completed_depth_histogram": search["completed_depth_histogram"],
        "timeout_count": search["timeout_count"],
        "all_nonforced_moves_completed_target": completed,
    }


def _pairing_summary(
    match: dict,
    *,
    higher_depth: int,
    lower_depth: int | None,
    paired_seeds: int,
) -> dict:
    pair = match["pair"]
    games = pair["games"]
    points = pair["a_wins"] + 0.5 * pair["draws"]
    lower, upper = _wilson_interval(points, games)
    higher_name = pair["a"]
    opponent_name = pair["b"]
    higher_timing = match["timing"][higher_name]
    opponent_timing = match["timing"][opponent_name]
    summary = {
        "higher_depth": higher_depth,
        "lower_depth": lower_depth,
        "paired_seeds": paired_seeds,
        "requested_games": match["requested_games"],
        "completed_games": match["completed_games"],
        "failed_games": match["failed_games"],
        "higher_depth_wins": pair["a_wins"],
        "draws": pair["draws"],
        "higher_depth_losses": pair["b_wins"],
        "higher_depth_points_rate": pair["a_points_rate"],
        "higher_depth_points_lcb_95": lower,
        "higher_depth_points_ucb_95": upper,
        "avg_margin_higher_depth": pair["avg_margin_a"],
        "higher_depth_timing": higher_timing,
        "opponent_timing": opponent_timing,
        "higher_depth_completion": _depth_completion(higher_timing, higher_depth),
    }
    if lower_depth is not None:
        summary["lower_depth_completion"] = _depth_completion(
            opponent_timing, lower_depth
        )
        summary["fixed_depth_control_valid"] = (
            summary["higher_depth_completion"][
                "all_nonforced_moves_completed_target"
            ]
            and summary["lower_depth_completion"][
                "all_nonforced_moves_completed_target"
            ]
        )
    else:
        summary["fixed_depth_control_valid"] = summary["higher_depth_completion"][
            "all_nonforced_moves_completed_target"
        ]
    return summary


def _positive_conversion(item: dict) -> bool:
    return (
        item["higher_depth_points_rate"] > 0.5
        and item["avg_margin_higher_depth"] > 0.0
    )


def _fitness_verdict(core: dict) -> dict:
    policy = {
        "clear_direct_depth3_vs_depth1_lcb_min": 0.5,
        "positive_pairing_points_min_exclusive": 0.5,
        "positive_pairing_margin_min_exclusive": 0.0,
        "depth_converts_requires": (
            "valid fixed-depth control; positive d2>d1 and d3>d2 adjacent "
            "pairings; d3>d1 Wilson LCB > 0.5"
        ),
        "weak_conversion_requires": (
            "valid fixed-depth control; positive d3>d1; at least two of the "
            "three higher-vs-lower pairings positive"
        ),
    }
    valid = all(item["fixed_depth_control_valid"] for item in core.values())
    positives = {key: _positive_conversion(item) for key, item in core.items()}
    positive_count = sum(positives.values())
    direct = core["d3_vs_d1"]
    adjacent_monotonic = positives["d2_vs_d1"] and positives["d3_vs_d2"]
    direct_clear = (
        positives["d3_vs_d1"]
        and direct["higher_depth_points_lcb_95"] > 0.5
    )

    if not valid:
        classification = "invalid_depth_control"
        reasoning = (
            "At least one searched move failed to complete its requested fixed "
            "depth; no fitness inference is valid."
        )
        fork = "Repair the fixed-depth driver and rerun; do not route the project yet."
    elif adjacent_monotonic and direct_clear:
        classification = "depth_converts"
        reasoning = (
            "The depth curve is strictly positive across both adjacent steps, "
            "and depth 3 clearly beats depth 1 by its 95% Wilson lower bound."
        )
        fork = (
            "Proceed to NNUE_P0_EVAL_DIAGNOSIS_PROMPT.md, then gate any retrain "
            "on beating pick_aware under matched search. Standalone strength remains live."
        )
    elif positives["d3_vs_d1"] and positive_count >= 2:
        classification = "weak_conversion"
        reasoning = (
            "Added depth has a positive directional effect, but the pairwise "
            "curve is not both monotonic and statistically clear."
        )
        fork = (
            "Re-weight toward eval-as-generator, exact endgames, and AZ curriculum. "
            "An eval retrain still matters for generator/relabeler roles, not as "
            "evidence that feasible depth can out-search AZ."
        )
    else:
        classification = "no_conversion"
        reasoning = (
            "Depth 3 does not directionally beat depth 1 with a positive margin "
            "in a broadly monotonic pairwise curve."
        )
        fork = (
            "Re-weight toward eval-as-generator, exact endgames, and AZ curriculum. "
            "An eval retrain still matters for generator/relabeler roles, not in "
            "pursuit of standalone depth-based superiority."
        )

    return {
        "classification": classification,
        "fixed_depth_control_valid": valid,
        "pairing_positive": positives,
        "positive_pairing_count": positive_count,
        "adjacent_depth_curve_strictly_monotonic": adjacent_monotonic,
        "depth3_vs_depth1_clear": direct_clear,
        "policy": policy,
        "reasoning": reasoning,
        "recommended_fork": fork,
        "do_not_act_in_this_probe": True,
    }


def _step1_baseline(step1: dict, az_sims: int, source_report: Path) -> dict:
    for item in step1["clocks"]:
        if int(item["az_sims"]) == az_sims:
            performance = item["performance"]
            return {
                "source_report": str(source_report),
                "az_sims": az_sims,
                "pilot_clock_seconds": item["clock_seconds"],
                "games": performance["games"],
                "pilot_points_rate": performance["nnue_points_rate"],
                "pilot_avg_margin_vs_az": performance["avg_margin_nnue"],
                "performance_sets": (
                    "screening confirmation"
                    if item.get("confirmation") is not None
                    else "screening"
                ),
                "comparison_caveat": (
                    "Step-1 baseline used fewer and different paired seeds; gap "
                    "closure is a directional ceiling comparison, not a paired estimate."
                ),
            }
    raise ValueError(f"Step-1 report has no exact AZ simulation baseline for {az_sims}")


def _gap_closure(summary: dict, baseline: dict) -> dict:
    old_margin = baseline["pilot_avg_margin_vs_az"]
    new_margin = summary["avg_margin_higher_depth"]
    improvement = new_margin - old_margin
    return {
        "baseline": baseline,
        "depth3_pick_aware_points_rate": summary["higher_depth_points_rate"],
        "depth3_pick_aware_avg_margin_vs_az": new_margin,
        "points_rate_improvement": (
            summary["higher_depth_points_rate"] - baseline["pilot_points_rate"]
        ),
        "margin_improvement_points": improvement,
        "fraction_of_step1_margin_deficit_closed": (
            improvement / abs(old_margin) if old_margin < 0.0 else None
        ),
        "remaining_margin_to_even": max(0.0, -new_margin),
    }


def run_probe(args: argparse.Namespace) -> dict:
    depths = tuple(sorted(set(args.depths)))
    if depths != DEFAULT_DEPTHS:
        raise ValueError("this predeclared probe requires exactly depths 1,2,3")
    az_sims_values = tuple(args.az_sims)
    if (
        len(az_sims_values) != 2
        or len(set(az_sims_values)) != 2
        or any(sims < 1 for sims in az_sims_values)
    ):
        raise ValueError("provide exactly two distinct AZ simulation budgets")

    step1_path = Path(args.step1_report).resolve()
    az_path = Path(args.az_checkpoint).resolve()
    if not step1_path.is_file() or not az_path.is_file():
        raise FileNotFoundError("Step-1 report and current-best AZ checkpoint must exist")
    step1 = json.loads(step1_path.read_text(encoding="utf-8"))
    az_hash = _sha256(az_path)
    expected_az_hash = step1["settings"]["az_checkpoint_sha256"]
    if az_hash != expected_az_hash:
        raise ValueError("current-best AZ hash differs from the completed Step-1 artifact")

    core_specs = (
        ("d2_vs_d1", 2, 1),
        ("d3_vs_d2", 3, 2),
        ("d3_vs_d1", 3, 1),
    )
    core_matches = {}
    core_summaries = {}
    for key, high, low in core_specs:
        print(f"{key}: {args.paired_seeds} paired seeds", flush=True)
        match = run_paired(
            _pick_aware_participant(
                high,
                chance_samples=args.chance_samples,
                enum_cap=args.enum_cap,
            ),
            _pick_aware_participant(
                low,
                chance_samples=args.chance_samples,
                enum_cap=args.enum_cap,
            ),
            seed_start=args.core_seed_start,
            paired_seeds=args.paired_seeds,
            settings={
                "kind": "pick_aware_fixed_depth_ablation",
                "pairing": key,
                "higher_depth": high,
                "lower_depth": low,
                "seed_start": args.core_seed_start,
                "paired_seeds": args.paired_seeds,
            },
        )
        summary = _pairing_summary(
            match,
            higher_depth=high,
            lower_depth=low,
            paired_seeds=args.paired_seeds,
        )
        core_matches[key] = match
        core_summaries[key] = summary
        print(
            f"  higher-depth points={summary['higher_depth_points_rate']:.3f} "
            f"[{summary['higher_depth_points_lcb_95']:.3f}, "
            f"{summary['higher_depth_points_ucb_95']:.3f}] "
            f"margin={summary['avg_margin_higher_depth']:+.2f} "
            f"fixed={summary['fixed_depth_control_valid']}",
            flush=True,
        )

    best_depth = max(depths)
    az_matches = {}
    az_summaries = {}
    for sims in az_sims_values:
        key = f"d{best_depth}_vs_az_{sims}sims"
        print(f"{key}: {args.paired_seeds} paired seeds", flush=True)
        az = build_open_loop_checkpoint_participant(
            str(az_path),
            name=f"AlphaZero_{sims}sims",
            device=args.device,
            sims=sims,
            c_puct=args.c_puct,
            temperature=0.0,
            channels_override=None,
            blocks_override=None,
            bilinear_dim_override=None,
        )
        match = run_paired(
            _pick_aware_participant(
                best_depth,
                chance_samples=args.chance_samples,
                enum_cap=args.enum_cap,
            ),
            az,
            seed_start=args.az_seed_start,
            paired_seeds=args.paired_seeds,
            settings={
                "kind": "pick_aware_depth_ceiling_vs_current_best_az",
                "depth": best_depth,
                "az_sims": sims,
                "seed_start": args.az_seed_start,
                "paired_seeds": args.paired_seeds,
            },
        )
        summary = _pairing_summary(
            match,
            higher_depth=best_depth,
            lower_depth=None,
            paired_seeds=args.paired_seeds,
        )
        baseline = _step1_baseline(step1, sims, step1_path)
        summary["gap_closure_vs_step1_pilot"] = _gap_closure(summary, baseline)
        az_matches[key] = match
        az_summaries[key] = summary
        print(
            f"  d{best_depth} points={summary['higher_depth_points_rate']:.3f} "
            f"[{summary['higher_depth_points_lcb_95']:.3f}, "
            f"{summary['higher_depth_points_ucb_95']:.3f}] "
            f"margin={summary['avg_margin_higher_depth']:+.2f} "
            f"fixed={summary['fixed_depth_control_valid']}",
            flush=True,
        )

    verdict = _fitness_verdict(core_summaries)
    rust_binary = Path(kr.__file__).resolve()
    driver_path = Path(__file__).resolve()
    oracle_source = driver_path.parents[1] / "rust_expectiminimax.py"
    return {
        "schema": "kingdomino_nnue_depth_conversion_fitness_probe_v1",
        "created_utc": _utc_now(),
        "question": "Does fixed search depth convert to strength for a decent Kingdomino eval?",
        "fitness_verdict": verdict,
        "within_row_depth_caveat": WITHIN_ROW_CAVEAT,
        "settings": {
            "driver": "RustExpectiminimax(depth=N, eval_fn=pick_aware)",
            "driver_path": str(driver_path),
            "driver_sha256": _sha256(driver_path),
            "depths": list(depths),
            "depth_control": (
                "direct fixed-depth recursion; no wall deadline, iterative deepening, "
                "or operational exact-tail extension; every non-forced move reports "
                "exactly the requested completed_depth"
            ),
            "eval": "pick_aware",
            "eval_definition": "score margin plus claimed-domino crown potential",
            "oracle_source": str(oracle_source),
            "oracle_source_sha256": _sha256(oracle_source),
            "chance_samples": args.chance_samples,
            "enum_cap": args.enum_cap,
            "chance_handling": "inside one public-state expectiminimax tree",
            "encoder_order_blind": True,
            "representative_row_shortcut": False,
            "full_width": True,
            "move_ordering": "canonical Rust engine legal-action order",
            "selective_pruning": None,
            "d4_included": False,
            "d4_omission_reason": "The predeclared 1/2/3 within-row probe answers the fitness gate without expanding scope.",
            "pick_blind_control_included": False,
            "pick_blind_omission_reason": "Optional control omitted to concentrate power on the required pick_aware pairings.",
            "core_seed_set": {
                "tag": "screening_common_across_depth_pairings",
                "seed_start": args.core_seed_start,
                "paired_seeds_per_pairing": args.paired_seeds,
            },
            "az_seed_set": {
                "tag": "ceiling_common_across_az_budgets",
                "seed_start": args.az_seed_start,
                "paired_seeds_per_pairing": args.paired_seeds,
            },
            "az_checkpoint": str(az_path),
            "az_checkpoint_sha256": az_hash,
            "step1_expected_az_checkpoint_sha256": expected_az_hash,
            "az_sims": list(az_sims_values),
            "device": args.device,
            "c_puct": args.c_puct,
            "step1_report": str(step1_path),
            "step1_report_sha256": _sha256(step1_path),
            "rust_extension_binary": str(rust_binary),
            "rust_extension_binary_sha256": _sha256(rust_binary),
            "reserved_test_split_opened": False,
            "confirmation_set": None,
        },
        "core_depth_pairings": core_summaries,
        "az_ceiling": az_summaries,
        "detailed_matches": {
            "core_depth_pairings": core_matches,
            "az_ceiling": az_matches,
        },
        "result_routes_only": verdict["recommended_fork"],
    }


def _print_summary(report: dict, out: Path) -> None:
    print(f"report: {out}")
    print("pairing             points (95% Wilson)      margin   fixed")
    for key, item in report["core_depth_pairings"].items():
        print(
            f"{key:<19} {item['higher_depth_points_rate']:.3f} "
            f"[{item['higher_depth_points_lcb_95']:.3f}, "
            f"{item['higher_depth_points_ucb_95']:.3f}] "
            f"{item['avg_margin_higher_depth']:+8.2f}   "
            f"{item['fixed_depth_control_valid']}"
        )
    for key, item in report["az_ceiling"].items():
        print(
            f"{key:<19} {item['higher_depth_points_rate']:.3f} "
            f"[{item['higher_depth_points_lcb_95']:.3f}, "
            f"{item['higher_depth_points_ucb_95']:.3f}] "
            f"{item['avg_margin_higher_depth']:+8.2f}   "
            f"{item['fixed_depth_control_valid']}"
        )
    print(json.dumps(report["fitness_verdict"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--step1-report", default=str(DEFAULT_STEP1))
    parser.add_argument("--az-checkpoint", default=str(DEFAULT_AZ))
    parser.add_argument("--depths", type=int, nargs="+", default=list(DEFAULT_DEPTHS))
    parser.add_argument("--az-sims", type=int, nargs="+", default=list(DEFAULT_AZ_SIMS))
    parser.add_argument("--paired-seeds", type=int, default=24)
    parser.add_argument("--core-seed-start", type=int, default=20263000)
    parser.add_argument("--az-seed-start", type=int, default=20264000)
    parser.add_argument("--chance-samples", type=int, default=16)
    parser.add_argument("--enum-cap", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--c-puct", type=float, default=1.5)
    args = parser.parse_args()
    if args.paired_seeds < 1:
        parser.error("--paired-seeds must be >= 1")
    report = run_probe(args)
    out = write_report(report, args.out)
    _print_summary(report, out)


if __name__ == "__main__":
    main()
