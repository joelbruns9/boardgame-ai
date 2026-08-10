"""Compare frozen-network search arms with a completed deck-8 boundary oracle."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from games.kingdomino.chance_correct_search_probe import _arm
from games.kingdomino.deck8_oracle import apply_prefix_actions, validate_boundary_state
from games.kingdomino.denial_search import load_checkpoint_network
from games.kingdomino.denial_signal_sweep import file_sha256, load_frozen_positions
from games.kingdomino.promotion import DEFAULT_CURRENT_BEST, sha256_file
from games.kingdomino.self_play import make_rust_evaluator


SCHEMA_VERSION = "kd-deck8-oracle-compare-v5"


def require_exhausted_nn_budget(
    name: str, arm: dict[str, Any], nn_eval_budget: int
) -> None:
    """Reject an arm that stopped before the preregistered equal-work budget."""
    if nn_eval_budget <= 0:
        return
    diagnostics = arm["chance_diagnostics"]
    if int(arm["nn_evaluations"]) != nn_eval_budget or not bool(
        diagnostics["nn_eval_budget_hit"]
    ):
        simulations_completed = int(diagnostics.get("simulations_completed", 0))
        simulations_requested = int(diagnostics.get("simulations_requested", 0))
        ordinary_nn_evaluations = int(diagnostics.get("ordinary_nn_evaluations", 0))
        unused = int(
            diagnostics.get(
                "nn_eval_budget_unused",
                max(0, nn_eval_budget - int(arm["nn_evaluations"])),
            )
        )
        raise RuntimeError(
            f"arm {name} did not exhaust the NN budget "
            f"({arm['nn_evaluations']}/{nn_eval_budget}, unused={unused}, "
            f"simulations={simulations_completed}/{simulations_requested}, "
            f"ordinary_nn_evaluations={ordinary_nn_evaluations}); increase --sims "
            "only if ordinary leaf evaluations are still being produced, otherwise "
            "inspect terminal saturation"
        )


def rotated_arm_order(
    arm_names: list[str], seed_index: int, rotation_offset: int = 0
) -> list[str]:
    """Cycle timing position without changing an arm's search seed."""
    if not arm_names:
        return []
    rotation = (rotation_offset + seed_index) % len(arm_names)
    return arm_names[rotation:] + arm_names[:rotation]


def a1c_node_diagnostics(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct per-chance-node admission evidence from Rust's flat map."""
    prefix = "a1c_node_"
    suffix = "_initialized_cycles"
    node_ids = sorted(
        int(key[len(prefix) : -len(suffix)])
        for key in diagnostics
        if key.startswith(prefix) and key.endswith(suffix)
    )
    rows = []
    for node_id in node_ids:
        visits = int(diagnostics[f"{prefix}{node_id}_visits"])
        preinit_visits = int(diagnostics[f"{prefix}{node_id}_preinit_visits"])
        rows.append(
            {
                "node_id": node_id,
                "visits": visits,
                "preinit_visits": preinit_visits,
                "preinit_visit_fraction": (
                    0.0 if visits == 0 else preinit_visits / visits
                ),
                "initialized_cycles": int(
                    diagnostics[f"{prefix}{node_id}_initialized_cycles"]
                ),
            }
        )
    return rows


def exact_separation(exact_actor_values: dict[int, float]) -> dict[str, Any]:
    order = sorted(exact_actor_values, key=lambda idx: (-exact_actor_values[idx], idx))
    best_value = exact_actor_values[order[0]]
    best_actions = [
        idx
        for idx in order
        if math.isclose(
            exact_actor_values[idx], best_value, rel_tol=0.0, abs_tol=1e-12
        )
    ]
    best_set = set(best_actions)
    first_inferior = next((idx for idx in order if idx not in best_set), None)
    return {
        "best_action_idx": best_actions[0],
        "best_action_indices": best_actions,
        "best_value_actor": best_value,
        "runner_up_action_idx": first_inferior,
        "top_gap_actor": (
            None
            if first_inferior is None
            else best_value - exact_actor_values[first_inferior]
        ),
        "all_actions_tied": first_inferior is None,
    }


def score_arm_against_oracle(
    arm: dict[str, Any],
    exact_actor_values: dict[int, float],
) -> dict[str, Any]:
    best_value = max(exact_actor_values.values())
    best_actions = sorted(
        idx
        for idx, value in exact_actor_values.items()
        if math.isclose(value, best_value, rel_tol=0.0, abs_tol=1e-12)
    )
    best_action = best_actions[0]
    selected = int(arm["top_action_idx"])
    if selected not in exact_actor_values:
        raise ValueError(f"search selected action {selected} absent from oracle")
    children = {int(row["action_idx"]): row for row in arm["children"]}
    common = sorted(set(children) & set(exact_actor_values))

    def pairwise(field: str) -> tuple[int, int]:
        correct = 0
        eligible = 0
        for offset, left in enumerate(common):
            for right in common[offset + 1:]:
                exact_gap = exact_actor_values[left] - exact_actor_values[right]
                left_value = children[left].get(field)
                right_value = children[right].get(field)
                if exact_gap == 0.0 or left_value is None or right_value is None:
                    continue
                search_gap = float(left_value) - float(right_value)
                if search_gap == 0.0:
                    continue
                eligible += 1
                correct += int((exact_gap > 0.0) == (search_gap > 0.0))
        return correct, eligible

    q_correct, q_pairs = pairwise("q_actor")
    visit_correct, visit_pairs = pairwise("visits")
    selected_value = exact_actor_values[selected]
    selected_rank = 1 + sum(
        not math.isclose(value, selected_value, rel_tol=0.0, abs_tol=1e-12)
        and value > selected_value
        for value in exact_actor_values.values()
    )
    diagnostics = arm.get("chance_diagnostics", {})
    node_diagnostics = a1c_node_diagnostics(diagnostics)
    return {
        "selected_action_idx": selected,
        "exact_best_action_idx": best_action,
        "exact_best_action_indices": best_actions,
        "top1_exact": selected in best_actions,
        "selected_exact_rank": selected_rank,
        "selected_exact_value_actor": exact_actor_values[selected],
        "exact_best_value_actor": exact_actor_values[best_action],
        "exact_regret_actor": exact_actor_values[best_action] - exact_actor_values[selected],
        "q_pairwise_correct": q_correct,
        "q_pairwise_pairs": q_pairs,
        "q_pairwise_accuracy": None if q_pairs == 0 else q_correct / q_pairs,
        "visit_pairwise_correct": visit_correct,
        "visit_pairwise_pairs": visit_pairs,
        "visit_pairwise_accuracy": (
            None if visit_pairs == 0 else visit_correct / visit_pairs
        ),
        "nn_evaluations": int(arm["nn_evaluations"]),
        "rust_nn_evaluations": int(
            diagnostics.get("nn_evaluations", arm["nn_evaluations"])
        ),
        "python_rust_nn_accounting_match": int(arm["nn_evaluations"])
        == int(diagnostics.get("nn_evaluations", arm["nn_evaluations"])),
        "simulations_completed": int(
            diagnostics.get("simulations_completed", arm.get("sims", 0))
        ),
        "initialization_nn_evaluations": int(
            diagnostics.get("initialization_nn_evaluations", 0)
        ),
        "initialization_nn_fraction": float(
            diagnostics.get("initialization_nn_fraction", 0.0)
        ),
        "ordinary_nn_evaluations": int(
            diagnostics.get("ordinary_nn_evaluations", arm["nn_evaluations"])
        ),
        "nn_evaluator_calls": int(
            diagnostics.get("nn_evaluator_calls", arm.get("evaluator_calls", 0))
        ),
        "nn_max_batch_size": int(
            diagnostics.get(
                "nn_max_batch_size", arm.get("evaluator_max_batch_size", 0)
            )
        ),
        "nn_mean_batch_size": float(
            diagnostics.get(
                "nn_mean_batch_size",
                int(arm["nn_evaluations"])
                / max(1, int(diagnostics.get("nn_evaluator_calls", 0))),
            )
        ),
        "initialization_evaluator_calls": int(
            diagnostics.get("initialization_evaluator_calls", 0)
        ),
        "initialization_max_batch_size": int(
            diagnostics.get("initialization_max_batch_size", 0)
        ),
        "initialization_blocked_cycles": int(
            diagnostics.get("initialization_blocked_cycles", 0)
        ),
        "initialization_nn_budget_blocked_cycles": int(
            diagnostics.get("initialization_nn_budget_blocked_cycles", 0)
        ),
        "initialization_nn_budget_blocked_rows": int(
            diagnostics.get("initialization_nn_budget_blocked_rows", 0)
        ),
        "a1c_initialized_cycles": int(diagnostics.get("a1c_initialized_cycles", 0)),
        "a1c_preinit_visit_fraction": float(
            diagnostics.get("a1c_preinit_visit_fraction", 0.0)
        ),
        "a1c_reached_chance_nodes": int(
            diagnostics.get("a1c_reached_chance_nodes", 0)
        ),
        "a1c_initialized_chance_nodes": int(
            diagnostics.get("a1c_initialized_chance_nodes", 0)
        ),
        "a1c_uninitialized_chance_nodes": int(
            diagnostics.get("a1c_uninitialized_chance_nodes", 0)
        ),
        "a1c_uninitialized_node_visits": int(
            diagnostics.get("a1c_uninitialized_node_visits", 0)
        ),
        "a1c_nodes": node_diagnostics,
        "nn_eval_budget_hit": bool(diagnostics.get("nn_eval_budget_hit", False)),
        "nn_eval_budget_unused": int(diagnostics.get("nn_eval_budget_unused", 0)),
        "simulation_limit_hit": bool(diagnostics.get("simulation_limit_hit", False)),
        "search_waves": int(diagnostics.get("search_waves", 0)),
        "elapsed_seconds": float(arm["elapsed_seconds"]),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    q_correct = sum(int(row["q_pairwise_correct"]) for row in rows)
    q_pairs = sum(int(row["q_pairwise_pairs"]) for row in rows)
    visit_correct = sum(int(row["visit_pairwise_correct"]) for row in rows)
    visit_pairs = sum(int(row["visit_pairwise_pairs"]) for row in rows)
    q_seed_accuracies = [
        float(row["q_pairwise_accuracy"])
        for row in rows
        if row["q_pairwise_accuracy"] is not None
    ]
    visit_seed_accuracies = [
        float(row["visit_pairwise_accuracy"])
        for row in rows
        if row["visit_pairwise_accuracy"] is not None
    ]
    return {
        "searches": len(rows),
        "aggregation_unit": "search seed nested within one strategic boundary",
        "top1_exact_rate": statistics.fmean(float(row["top1_exact"]) for row in rows),
        "mean_exact_regret_actor": statistics.fmean(
            float(row["exact_regret_actor"]) for row in rows
        ),
        "max_exact_regret_actor": max(float(row["exact_regret_actor"]) for row in rows),
        "modal_selected_action_idx": min(
            {int(row["selected_action_idx"]) for row in rows},
            key=lambda idx: (
                -sum(int(row["selected_action_idx"]) == idx for row in rows), idx
            ),
        ),
        "mean_seed_q_pairwise_accuracy": (
            None if not q_seed_accuracies else statistics.fmean(q_seed_accuracies)
        ),
        "mean_seed_visit_pairwise_accuracy": (
            None if not visit_seed_accuracies else statistics.fmean(visit_seed_accuracies)
        ),
        "descriptive_pooled_q_pairwise_accuracy": (
            None if q_pairs == 0 else q_correct / q_pairs
        ),
        "descriptive_pooled_q_pairs": q_pairs,
        "descriptive_pooled_visit_pairwise_accuracy": (
            None if visit_pairs == 0 else visit_correct / visit_pairs
        ),
        "descriptive_pooled_visit_pairs": visit_pairs,
        "mean_nn_evaluations": statistics.fmean(
            float(row["nn_evaluations"]) for row in rows
        ),
        "mean_simulations_completed": statistics.fmean(
            float(row["simulations_completed"]) for row in rows
        ),
        "mean_initialization_nn_evaluations": statistics.fmean(
            float(row["initialization_nn_evaluations"]) for row in rows
        ),
        "mean_initialization_nn_fraction": statistics.fmean(
            float(row["initialization_nn_fraction"]) for row in rows
        ),
        "mean_ordinary_nn_evaluations": statistics.fmean(
            float(row["ordinary_nn_evaluations"]) for row in rows
        ),
        "mean_nn_evaluator_calls": statistics.fmean(
            float(row["nn_evaluator_calls"]) for row in rows
        ),
        "mean_nn_max_batch_size": statistics.fmean(
            float(row["nn_max_batch_size"]) for row in rows
        ),
        "mean_nn_mean_batch_size": statistics.fmean(
            float(row["nn_mean_batch_size"]) for row in rows
        ),
        "mean_initialization_evaluator_calls": statistics.fmean(
            float(row["initialization_evaluator_calls"]) for row in rows
        ),
        "mean_initialization_max_batch_size": statistics.fmean(
            float(row["initialization_max_batch_size"]) for row in rows
        ),
        "mean_a1c_initialized_cycles": statistics.fmean(
            float(row["a1c_initialized_cycles"]) for row in rows
        ),
        "mean_a1c_preinit_visit_fraction": statistics.fmean(
            float(row["a1c_preinit_visit_fraction"]) for row in rows
        ),
        "mean_a1c_reached_chance_nodes": statistics.fmean(
            float(row["a1c_reached_chance_nodes"]) for row in rows
        ),
        "mean_a1c_initialized_chance_nodes": statistics.fmean(
            float(row["a1c_initialized_chance_nodes"]) for row in rows
        ),
        "mean_a1c_uninitialized_chance_nodes": statistics.fmean(
            float(row["a1c_uninitialized_chance_nodes"]) for row in rows
        ),
        "mean_a1c_uninitialized_node_visits": statistics.fmean(
            float(row["a1c_uninitialized_node_visits"]) for row in rows
        ),
        "searches_with_uninitialized_chance_nodes": sum(
            int(row["a1c_uninitialized_chance_nodes"] > 0) for row in rows
        ),
        "all_nn_eval_budgets_hit": all(bool(row["nn_eval_budget_hit"]) for row in rows),
        "all_python_rust_nn_accounting_match": all(
            bool(row["python_rust_nn_accounting_match"]) for row in rows
        ),
        "all_simulation_limits_avoided": all(
            not bool(row["simulation_limit_hit"]) for row in rows
        ),
        "total_initialization_blocked_cycles": sum(
            int(row["initialization_blocked_cycles"]) for row in rows
        ),
        "total_initialization_nn_budget_blocked_cycles": sum(
            int(row["initialization_nn_budget_blocked_cycles"]) for row in rows
        ),
        "total_initialization_nn_budget_blocked_rows": sum(
            int(row["initialization_nn_budget_blocked_rows"]) for row in rows
        ),
        "mean_seconds": statistics.fmean(float(row["elapsed_seconds"]) for row in rows),
    }


def build_arm_specs(
    *,
    include_a1c: bool,
    a1c_exposures: str = "4,8",
    a1c_sampling: str = "balanced,iid",
) -> dict[str, dict[str, Any]]:
    """Build the legacy CLI arm matrix without changing its ordering."""
    specs: dict[str, dict[str, Any]] = {
        "x0": {
            "exposure": 0,
            "backup": "hajek",
            "traversal": "balanced",
            "panel_mode": "lazy",
            "panel_sampling": "balanced",
            "enum_max_rows": 70,
            "leaf_batch": 8,
        },
        "x1_hajek_balanced": {
            "exposure": 1,
            "backup": "hajek",
            "traversal": "balanced",
            "panel_mode": "lazy",
            "panel_sampling": "balanced",
            "enum_max_rows": 70,
            "leaf_batch": 8,
        },
    }
    if include_a1c:
        exposures = [int(value) for value in str(a1c_exposures).split(",")]
        samplings = [value.strip() for value in str(a1c_sampling).split(",")]
        if any(exposure <= 0 for exposure in exposures):
            raise ValueError("--a1c-exposures must contain positive integers")
        if any(sampling not in {"balanced", "iid"} for sampling in samplings):
            raise ValueError("--a1c-sampling must contain balanced and/or iid")
        for exposure in exposures:
            for sampling in samplings:
                specs[f"a1c_x{exposure}_{sampling}"] = {
                    "exposure": exposure,
                    "backup": "sampled",
                    "traversal": "balanced",
                    "panel_mode": "a1c",
                    "panel_sampling": sampling,
                    # Force sampled panels at deck=8; the exact 70-row oracle is
                    # the external target, not a counterfactual search arm.
                    "enum_max_rows": 1,
                    "leaf_batch": 8,
                }
    return specs


def compare_boundary(
    boundary: Any,
    evaluator: Any,
    exact_actor_values: dict[int, float],
    specs: dict[str, dict[str, Any]],
    *,
    simulation_ceiling: int,
    nn_eval_budget: int,
    seed: int,
    seed_count: int,
    fpu: float,
    cpuct: float,
    margin_gain: float,
    alpha: float,
    a1c_init_visits: int,
    a1c_widening_c: float,
    a1c_init_max_fraction: float,
    rotation_offset: int = 0,
    warm: bool = True,
) -> dict[str, Any]:
    """Run and score one boundary with a caller-owned loaded evaluator."""
    if any(spec.get("panel_mode") == "a1c" for spec in specs.values()) and nn_eval_budget <= 0:
        raise ValueError("A1c comparisons require a positive NN evaluation budget")
    if warm:
        # Warm the evaluator outside measured arms.
        _arm(
            boundary, evaluator, sims=1, exposure=0, enum_max_rows=70,
            seed=int(seed), margin_gain=margin_gain, alpha=alpha,
            fpu=float(fpu), cpuct=float(cpuct), backup="hajek",
            traversal="balanced",
        )
    records = []
    arm_names = list(specs)
    for seed_index in range(int(seed_count)):
        search_seed = int(seed) + seed_index
        arm_rows = {}
        execution_order = rotated_arm_order(
            arm_names, seed_index, rotation_offset=rotation_offset
        )
        for name in execution_order:
            spec = specs[name]
            arm = _arm(
                boundary,
                evaluator,
                sims=int(simulation_ceiling),
                exposure=int(spec["exposure"]),
                enum_max_rows=int(spec["enum_max_rows"]),
                seed=search_seed,
                margin_gain=margin_gain,
                alpha=alpha,
                fpu=float(fpu),
                cpuct=float(cpuct),
                backup=str(spec["backup"]),
                traversal=str(spec["traversal"]),
                panel_mode=str(spec["panel_mode"]),
                panel_sampling=str(spec["panel_sampling"]),
                init_visits=int(a1c_init_visits),
                widening_c=float(a1c_widening_c),
                init_max_fraction=float(a1c_init_max_fraction),
                leaf_batch=int(spec["leaf_batch"]),
                nn_eval_budget=int(nn_eval_budget),
            )
            require_exhausted_nn_budget(name, arm, int(nn_eval_budget))
            arm_rows[name] = score_arm_against_oracle(arm, exact_actor_values)
        records.append(
            {
                "seed_index": seed_index,
                "seed": search_seed,
                "arm_execution_order": execution_order,
                "arms": arm_rows,
            }
        )
    return {
        "records": records,
        "aggregate": {
            name: _aggregate([record["arms"][name] for record in records])
            for name in specs
        },
        "exact": exact_separation(exact_actor_values),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    nn_eval_budget = int(getattr(args, "nn_eval_budget", 0))
    if bool(getattr(args, "include_a1c", False)) and nn_eval_budget <= 0:
        raise ValueError(
            "--include-a1c requires --nn-eval-budget > 0 for the primary "
            "equal-work oracle comparison"
        )
    oracle_path = Path(args.oracle_summary)
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    if oracle.get("status") != "complete":
        raise ValueError("oracle summary must be complete")
    exact_actor_values = {
        int(row["action_idx"]): float(row["expected_value_actor"])
        for row in oracle["actions"]
    }
    if len(exact_actor_values) != int(oracle["legal_actions"]):
        raise ValueError("oracle summary does not contain every legal action")

    positions_path = Path(args.positions_path)
    positions = load_frozen_positions(positions_path)
    root, _source = positions[int(oracle["position_index"])]
    boundary = apply_prefix_actions(root, [int(x) for x in oracle["prefix_actions"]])
    validate_boundary_state(boundary)

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"comparison output already exists: {output}")
    net, checkpoint_cfg = load_checkpoint_network(args.checkpoint, args.device)
    margin_gain = float(checkpoint_cfg.get("margin_gain", 2.0))
    alpha = float(checkpoint_cfg.get("alpha", 0.5))
    evaluator = make_rust_evaluator(
        net,
        device=args.device,
        margin_gain=margin_gain,
        alpha=alpha,
    )
    specs = build_arm_specs(
        include_a1c=bool(getattr(args, "include_a1c", False)),
        a1c_exposures=str(args.a1c_exposures),
        a1c_sampling=str(args.a1c_sampling),
    )
    comparison = compare_boundary(
        boundary,
        evaluator,
        exact_actor_values,
        specs,
        simulation_ceiling=int(args.sims),
        nn_eval_budget=nn_eval_budget,
        seed=int(args.seed),
        seed_count=int(args.seed_count),
        fpu=float(args.fpu),
        cpuct=float(args.cpuct),
        margin_gain=margin_gain,
        alpha=alpha,
        a1c_init_visits=int(args.a1c_init_visits),
        a1c_widening_c=float(args.a1c_widening_c),
        a1c_init_max_fraction=float(args.a1c_init_max_fraction),
        rotation_offset=int(getattr(args, "arm_order_rotation_offset", 0)),
    )
    records = comparison["records"]
    aggregate = comparison["aggregate"]
    exact_order = sorted(exact_actor_values, key=lambda x: (-exact_actor_values[x], x))
    exact_summary = exact_separation(exact_actor_values)
    result = {
        "schema_version": SCHEMA_VERSION,
        "scope": "single-boundary search convergence diagnostic; not an independent position sample",
        "selection_reason": str(args.selection_reason),
        "oracle_summary": str(oracle_path.resolve()),
        "oracle_sha256": file_sha256(oracle_path),
        "oracle_id": oracle["oracle_id"],
        "positions_path": str(positions_path.resolve()),
        "positions_sha256": file_sha256(positions_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "configuration": {
            "simulation_ceiling": int(args.sims),
            "nn_eval_budget": nn_eval_budget,
            "comparison_basis": "equal_nn_work" if nn_eval_budget > 0 else "equal_simulations",
            "arm_ordering": "cyclic_rotation_by_seed_index_plus_offset",
            "arm_order_rotation_offset": int(
                getattr(args, "arm_order_rotation_offset", 0)
            ),
            "seed": int(args.seed),
            "seed_count": int(args.seed_count), "fpu": float(args.fpu),
            "cpuct": float(args.cpuct), "device": str(args.device), "arms": specs,
        },
        "exact": {
            **exact_summary,
            "action_values_actor": {
                str(idx): exact_actor_values[idx] for idx in exact_order
            },
        },
        "records": records,
        "aggregate": aggregate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-summary", required=True)
    parser.add_argument(
        "--positions-path",
        default="runs/kingdomino/denial_search/signal_positions.jsonl",
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CURRENT_BEST))
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-reason", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sims", type=int, default=4800)
    parser.add_argument(
        "--nn-eval-budget",
        type=int,
        default=0,
        help="Hard per-search NN-row budget; 0 disables it. Required with --include-a1c.",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument(
        "--arm-order-rotation-offset",
        type=int,
        default=0,
        help="Added to the per-seed cyclic arm-order rotation (default preserves legacy behavior).",
    )
    parser.add_argument("--fpu", type=float, default=-0.2)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--include-a1c", action="store_true")
    parser.add_argument("--a1c-exposures", default="4,8")
    parser.add_argument("--a1c-sampling", default="balanced,iid")
    parser.add_argument("--a1c-init-visits", type=int, default=32)
    parser.add_argument("--a1c-widening-c", type=float, default=0.25)
    parser.add_argument("--a1c-init-max-fraction", type=float, default=0.25)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps({"exact": result["exact"], "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
