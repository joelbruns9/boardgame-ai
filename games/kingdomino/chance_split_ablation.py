"""Sampled explicit-chance-tree ablation for the completed deck=8 probe.

Runs only the new ``sampled_split`` arm, then pairs it by frozen position,
budget, and common-random-number seed with the existing control and exhaustive
panel arms.  This isolates explicit public-observation subtrees and sampled
balanced chance traversal from the incremental 70-row bootstrap.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Optional, Sequence

from games.kingdomino.chance_leverage_probe import (
    DEFAULT_BUDGETS,
    DEFAULT_DIR,
    DEFAULT_POSITIONS,
    _atomic_json,
    _mean_ci,
    _paired_metrics,
    _run_arm,
    _wilson,
)
from games.kingdomino.denial_search import load_checkpoint_network
from games.kingdomino.denial_signal_sweep import file_sha256, load_frozen_positions
from games.kingdomino.promotion import DEFAULT_CURRENT_BEST, sha256_file
from games.kingdomino.self_play import make_rust_evaluator


DEFAULT_BASE = DEFAULT_DIR / "deck8_causal_leverage_v1.json"
DEFAULT_OUTPUT = DEFAULT_DIR / "deck8_sampled_split_ablation_v1.json"
ABLATION_VERSION = "deck8-sampled-explicit-split-ablation-v1"


def _comparison_summary(
    left_runs: Sequence[dict[str, Any]],
    right_runs: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    pairs = [_paired_metrics(left, right) for left, right in zip(left_runs, right_runs)]
    tv = [float(row["visit_total_variation"]) for row in pairs]
    top_changed = sum(bool(row["top_action_changed"]) for row in pairs)
    pick_changed = sum(bool(row["top_pick_changed"]) for row in pairs)
    return {
        "pairs": len(pairs),
        "mean_visit_total_variation": statistics.fmean(tv) if tv else 0.0,
        "mean_visit_total_variation_bootstrap_95": _mean_ci(tv, seed),
        "material_tv_ge_0_10_rate": sum(value >= 0.10 for value in tv) / max(1, len(tv)),
        "top_action_change_rate": top_changed / max(1, len(pairs)),
        "top_action_change_wilson_95": _wilson(top_changed, len(pairs)),
        "top_pick_change_rate": pick_changed / max(1, len(pairs)),
        "top_action_changes": top_changed,
        "top_pick_changes": pick_changed,
    }


def build_summary(
    sampled_results: Sequence[dict[str, Any]],
    base_results: Sequence[dict[str, Any]],
    budgets: Sequence[int],
) -> dict[str, Any]:
    sampled_by_key = {
        (int(row["position_index"]), int(row["configured_budget"])): row
        for row in sampled_results
    }
    base_by_key = {
        (int(row["position_index"]), int(row["configured_budget"]), row["arm"]): row
        for row in base_results
    }
    summary: dict[str, Any] = {}
    for budget in budgets:
        indices = sorted(
            position_index
            for position_index, row_budget in sampled_by_key
            if row_budget == budget
        )
        sampled = [sampled_by_key[(index, budget)] for index in indices]
        control = [base_by_key[(index, budget, "control")] for index in indices]
        panel_extra = [base_by_key[(index, budget, "panel_extra")] for index in indices]
        panel_charged = [base_by_key[(index, budget, "panel_charged")] for index in indices]
        summary[str(budget)] = {
            "positions_completed": len(indices),
            "sampled_split": {
                "mean_nn_rows": statistics.fmean(row["python_nn_rows"] for row in sampled)
                if sampled else 0.0,
                "mean_elapsed_seconds": statistics.fmean(
                    row["elapsed_seconds"] for row in sampled
                ) if sampled else 0.0,
                "mean_chance_nodes": statistics.fmean(
                    row["diagnostics"]["chance_nodes"] for row in sampled
                ) if sampled else 0.0,
                "mean_reaching_path_fraction": statistics.fmean(
                    row["diagnostics"]["probe_paths_reaching_first_reveal_fraction"]
                    for row in sampled
                ) if sampled else 0.0,
            },
            "control_vs_sampled_split": _comparison_summary(
                control, sampled, seed=budget + 11
            ),
            "sampled_split_vs_panel_extra": _comparison_summary(
                sampled, panel_extra, seed=budget + 23
            ),
            "sampled_split_vs_panel_charged": _comparison_summary(
                sampled, panel_charged, seed=budget + 37
            ),
            "control_vs_panel_extra": _comparison_summary(
                control, panel_extra, seed=budget + 41
            ),
        }
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_path = Path(args.base_artifact)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if not base.get("completed") or len(base.get("results", [])) != 1280:
        raise ValueError("base causal-leverage artifact is not the completed 1,280-cell run")
    base_provenance = base["provenance"]
    budgets = tuple(args.budgets)
    base_budgets = list(base_provenance["budgets"])
    if any(budget not in base_budgets for budget in budgets):
        raise ValueError("every ablation budget must exist in the completed base artifact")

    positions_path = Path(args.positions_path)
    positions = load_frozen_positions(positions_path)
    positions = positions[: args.limit or None]
    checkpoint = Path(args.checkpoint)
    if file_sha256(positions_path) != base_provenance["positions_sha256"]:
        raise ValueError("frozen position hash differs from the completed base artifact")
    if sha256_file(checkpoint) != base_provenance["checkpoint_sha256"]:
        raise ValueError("checkpoint hash differs from the completed base artifact")

    provenance = {
        "version": ABLATION_VERSION,
        "base_artifact_path": str(base_path),
        "base_artifact_sha256": file_sha256(base_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": base_provenance["checkpoint_sha256"],
        "positions_path": str(positions_path),
        "positions_sha256": base_provenance["positions_sha256"],
        "position_count_requested": len(positions),
        "budgets": list(budgets),
        "arm": "sampled_split",
        "arm_definition": (
            "exhaustive 70-row support at every reached reveal; explicit public "
            "observation subtrees; sampled backup; balanced traversal; zero bootstrap rows"
        ),
        "paired_seed_source": "exact seed recorded by each base control cell",
        "leaf_batch": int(base_provenance["leaf_batch"]),
        "fpu": float(base_provenance["fpu"]),
        "cpuct": float(base_provenance["cpuct"]),
    }
    output = Path(args.output)
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("provenance") != provenance:
            raise ValueError(f"existing ablation artifact provenance differs: {output}")
    else:
        payload = {"provenance": provenance, "results": [], "summary": {}}
    completed = {
        (int(row["position_index"]), int(row["configured_budget"]))
        for row in payload["results"]
    }
    base_by_key = {
        (int(row["position_index"]), int(row["configured_budget"]), row["arm"]): row
        for row in base["results"]
    }

    net, checkpoint_cfg = load_checkpoint_network(checkpoint, args.device)
    margin_gain = float(checkpoint_cfg.get("margin_gain", 2.0))
    alpha = float(checkpoint_cfg.get("alpha", 0.5))
    evaluator = make_rust_evaluator(
        net,
        device=args.device,
        margin_gain=margin_gain,
        alpha=alpha,
    )
    _run_arm(
        positions[0][0], evaluator, arm="control", configured_budget=1,
        seed=0, leaf_batch=1, fpu=provenance["fpu"], cpuct=provenance["cpuct"],
        margin_gain=margin_gain, alpha=alpha,
    )

    for budget in budgets:
        for position_index, (state, source) in enumerate(positions):
            key = (position_index, budget)
            if key in completed:
                continue
            control = base_by_key[(position_index, budget, "control")]
            result = _run_arm(
                state,
                evaluator,
                arm="sampled_split",
                configured_budget=budget,
                seed=int(control["seed"]),
                leaf_batch=provenance["leaf_batch"],
                fpu=provenance["fpu"],
                cpuct=provenance["cpuct"],
                margin_gain=margin_gain,
                alpha=alpha,
            )
            result.update({"position_index": position_index, "position_source": source})
            diagnostics = result["diagnostics"]
            if (
                diagnostics["probe_sampled_split_requested"] != 1.0
                or diagnostics["probe_full_panel_bootstrap_rows"] != 0.0
                or diagnostics["initialization_nn_evaluations"] != 0.0
            ):
                raise RuntimeError("sampled-split arm violated its zero-bootstrap contract")
            payload["results"].append(result)
            payload["summary"] = build_summary(
                payload["results"], base["results"], budgets
            )
            _atomic_json(output, payload)
            completed.add(key)
            print(
                f"budget={budget} position={position_index + 1}/{len(positions)} "
                f"chance_nodes={int(diagnostics['chance_nodes'])} "
                f"nn={result['python_nn_rows']} secs={result['elapsed_seconds']:.2f}",
                flush=True,
            )
    payload["summary"] = build_summary(payload["results"], base["results"], budgets)
    payload["completed"] = True
    _atomic_json(output, payload)
    return payload


def _parse_budgets(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one budget is required")
    return values


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", default=str(DEFAULT_BASE))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CURRENT_BEST))
    parser.add_argument("--positions-path", default=str(DEFAULT_POSITIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--budgets", type=_parse_budgets, default=DEFAULT_BUDGETS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    payload = run(_parse_args(argv))
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
