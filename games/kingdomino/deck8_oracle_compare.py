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


SCHEMA_VERSION = "kd-deck8-oracle-compare-v2"
A1C_COMPARISON_READY = False


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
        "mean_seconds": statistics.fmean(float(row["elapsed_seconds"]) for row in rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if bool(getattr(args, "include_a1c", False)) and not A1C_COMPARISON_READY:
        raise RuntimeError(
            "A1c oracle comparison is disabled until the harness matches total NN "
            "work; wave-safe admission and leaf_batch parity alone do not make equal "
            "simulations an equal-compute comparison"
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

    specs = {
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
    if getattr(args, "include_a1c", False):
        exposures = [int(value) for value in str(args.a1c_exposures).split(",")]
        samplings = [value.strip() for value in str(args.a1c_sampling).split(",")]
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
    # Warm the evaluator outside measured arms.
    _arm(
        boundary, evaluator, sims=1, exposure=0, enum_max_rows=70,
        seed=int(args.seed), margin_gain=margin_gain, alpha=alpha,
        fpu=float(args.fpu), cpuct=float(args.cpuct), backup="hajek",
        traversal="balanced",
    )
    records = []
    for seed_index in range(int(args.seed_count)):
        seed = int(args.seed) + seed_index
        arm_rows = {}
        for name, spec in specs.items():
            arm = _arm(
                boundary,
                evaluator,
                sims=int(args.sims),
                exposure=int(spec["exposure"]),
                enum_max_rows=int(spec["enum_max_rows"]),
                seed=seed,
                margin_gain=margin_gain,
                alpha=alpha,
                fpu=float(args.fpu),
                cpuct=float(args.cpuct),
                backup=str(spec["backup"]),
                traversal=str(spec["traversal"]),
                panel_mode=str(spec["panel_mode"]),
                panel_sampling=str(spec["panel_sampling"]),
                init_visits=int(args.a1c_init_visits),
                widening_c=float(args.a1c_widening_c),
                init_max_fraction=float(args.a1c_init_max_fraction),
                leaf_batch=int(spec["leaf_batch"]),
            )
            arm_rows[name] = score_arm_against_oracle(arm, exact_actor_values)
        records.append({"seed_index": seed_index, "seed": seed, "arms": arm_rows})

    aggregate = {
        name: _aggregate([record["arms"][name] for record in records])
        for name in specs
    }
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
            "sims": int(args.sims), "seed": int(args.seed),
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
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--seed-count", type=int, default=8)
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
