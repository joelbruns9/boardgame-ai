"""Deck=8 first-selection causal reach/leverage experiment.

This is a diagnostic harness, not a self-play path.  It freezes independent,
untreated current-network trajectories and compares five paired search arms at
800/1600/3200/6400 configured work units:

* control: ordinary open-loop search;
* pulse_positive / pulse_negative: free persistent Q=+/-1 interventions at the
  first reveal-triggering action reached by search;
* panel_charged: one exhaustive 70-row panel with 70 ordinary paths removed;
* panel_extra: the same panel in addition to the full ordinary-path budget.

The JSON artifact is rewritten after every completed arm and can be resumed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from games.kingdomino.denial_search import (
    AZBatchEvaluator,
    DenialSearch,
    SearchConfig,
    generate_az_midgame_positions,
    load_checkpoint_network,
)
from games.kingdomino.denial_signal_sweep import (
    file_sha256,
    load_frozen_positions,
    write_frozen_positions,
)
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.promotion import DEFAULT_CURRENT_BEST, sha256_file
from games.kingdomino.self_play import make_rust_evaluator


DEFAULT_DIR = Path("runs/kingdomino/chance_correct_a1")
DEFAULT_POSITIONS = DEFAULT_DIR / "deck8_first_selection_positions_v1.jsonl"
DEFAULT_OUTPUT = DEFAULT_DIR / "deck8_causal_leverage_v1.json"
DEFAULT_BUDGETS = (800, 1600, 3200, 6400)
ARM_ORDER = (
    "control",
    "pulse_positive",
    "pulse_negative",
    "panel_charged",
    "panel_extra",
)
PROBE_VERSION = "deck8-first-selection-causal-leverage-v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def freeze_positions(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.positions_path)
    if output.exists() and not args.force:
        records = load_frozen_positions(output)
        if len(records) != args.positions:
            raise ValueError(
                f"existing frozen set has {len(records)} positions, expected "
                f"{args.positions}; pass --force to regenerate"
            )
        return {
            "path": str(output),
            "sha256": file_sha256(output),
            "positions": len(records),
        }

    net, checkpoint_cfg = load_checkpoint_network(args.checkpoint, args.device)
    trajectory_evaluator = AZBatchEvaluator(
        net,
        device=args.device,
        batch_size=args.trajectory_leaf_batch,
        margin_gain=float(checkpoint_cfg.get("margin_gain", 2.0)),
        alpha=float(checkpoint_cfg.get("alpha", 0.5)),
    )
    search = DenialSearch(
        trajectory_evaluator,
        checkpoint_path=args.checkpoint,
        config=SearchConfig(
            pick_plies=8,
            chance_k=4,
            seed=args.position_seed,
            placement_top_k=2,
            root_search_sims=args.trajectory_sims,
            policy_temperature=0.10,
        ),
    )
    generated = generate_az_midgame_positions(
        search,
        count=args.positions,
        seed=args.position_seed,
        min_deck=8,
        max_deck=8,
    )
    checkpoint_hash = sha256_file(args.checkpoint)
    records = []
    for state, source in generated:
        if len(state.deck) != 8 or state.actor_index != 0:
            raise RuntimeError("generator returned a non-target position")
        records.append(
            (
                state,
                {
                    **source,
                    "trajectory_engine": "untreated-greedy-az-mcts",
                    "trajectory_sims": int(args.trajectory_sims),
                    "trajectory_seed_policy": "fixed search seed; independent game seeds",
                    "checkpoint_sha256": checkpoint_hash,
                },
            )
        )
    return write_frozen_positions(records, output)


def _run_arm(
    state,
    evaluator,
    *,
    arm: str,
    configured_budget: int,
    seed: int,
    leaf_batch: int,
    fpu: float,
    cpuct: float,
    margin_gain: float,
    alpha: float,
) -> dict[str, Any]:
    import kingdomino_rust as kr

    if arm == "panel_charged":
        search_sims = configured_budget
        mode = "full_panel_charged"
    elif arm == "panel_extra":
        search_sims = configured_budget
        mode = "full_panel_extra"
    else:
        search_sims = configured_budget
        mode = arm
    if search_sims <= 0:
        raise ValueError("configured budget must exceed the 70-row panel charge")

    python_nn_rows = 0
    evaluator_calls = 0
    evaluator_max_batch = 0

    def counted_evaluator(my_board, opp_board, flat, legal_indices):
        nonlocal python_nn_rows, evaluator_calls, evaluator_max_batch
        batch = int(my_board.shape[0])
        python_nn_rows += batch
        evaluator_calls += 1
        evaluator_max_batch = max(evaluator_max_batch, batch)
        return evaluator(my_board, opp_board, flat, legal_indices)

    started = time.perf_counter()
    children, root_value0, raw_diagnostics = kr.advisor_chance_leverage_probe(
        _rust_state_from_python(state),
        counted_evaluator,
        int(search_sims),
        mode=mode,
        fpu=float(fpu),
        cpuct=float(cpuct),
        seed=int(seed) & 0xFFFF_FFFF_FFFF_FFFF,
        leaf_batch=int(leaf_batch),
        virtual_loss=1,
        margin_gain=float(margin_gain),
        alpha=float(alpha),
    )
    diagnostics = {str(key): float(value) for key, value in raw_diagnostics.items()}
    rust_nn_rows = int(diagnostics["nn_evaluations"])
    if python_nn_rows != rust_nn_rows:
        raise RuntimeError(
            f"Rust/Python NN accounting mismatch: rust={rust_nn_rows}, "
            f"python={python_nn_rows}"
        )
    actor = int(state.current_actor)
    total_visits = sum(int(row[1]) for row in children)
    rows = []
    for action_idx, visits, value_sum0, prior in children:
        visits = int(visits)
        q0 = None if visits == 0 else float(value_sum0) / visits
        rows.append(
            {
                "action_idx": int(action_idx),
                "visits": visits,
                "visit_share": visits / max(1, total_visits),
                "q_actor": None if q0 is None else (q0 if actor == 0 else -q0),
                "prior": float(prior),
            }
        )
    top = max(rows, key=lambda row: (row["visits"], row["prior"], -row["action_idx"]))
    return {
        "arm": arm,
        "rust_mode": mode,
        "configured_budget": int(configured_budget),
        "ordinary_simulation_ceiling": int(search_sims),
        "seed": int(seed),
        "elapsed_seconds": time.perf_counter() - started,
        "root_value_player0": float(root_value0),
        "top_action_idx": int(top["action_idx"]),
        "top_pick_rank": int(top["action_idx"]) % 5,
        "python_nn_rows": python_nn_rows,
        "python_evaluator_calls": evaluator_calls,
        "python_evaluator_max_batch": evaluator_max_batch,
        "diagnostics": diagnostics,
        "children": rows,
    }


def _shares(run: dict[str, Any]) -> dict[int, float]:
    return {int(row["action_idx"]): float(row["visit_share"]) for row in run["children"]}


def _paired_metrics(control: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    left, right = _shares(control), _shares(arm)
    indices = set(left) | set(right)
    target_action = int(arm["diagnostics"].get("probe_target_root_action_idx", -1))
    return {
        "top_action_changed": control["top_action_idx"] != arm["top_action_idx"],
        "top_pick_changed": control["top_pick_rank"] != arm["top_pick_rank"],
        "visit_total_variation": 0.5
        * sum(abs(left.get(index, 0.0) - right.get(index, 0.0)) for index in indices),
        "target_root_action_idx": target_action,
        "target_root_visit_share_delta": (
            None
            if target_action < 0
            else right.get(target_action, 0.0) - left.get(target_action, 0.0)
        ),
    }


def _wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _mean_ci(values: Sequence[float], seed: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(10_000, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def build_summary(results: Sequence[dict[str, Any]], budgets: Sequence[int]) -> dict[str, Any]:
    by_key = {
        (int(row["position_index"]), int(row["configured_budget"]), row["arm"]): row
        for row in results
    }
    summary: dict[str, Any] = {}
    for budget in budgets:
        controls = [
            row for row in results
            if row["arm"] == "control" and int(row["configured_budget"]) == budget
        ]
        controls.sort(key=lambda row: int(row["position_index"]))
        reached = [row for row in controls if row["diagnostics"]["probe_first_reveal_reached"] == 1.0]
        first_sims = [row["diagnostics"]["probe_first_reveal_simulation"] for row in reached]
        budget_summary: dict[str, Any] = {
            "positions_completed": len(controls),
            "control_reach": {
                "positions_reached": len(reached),
                "rate": len(reached) / max(1, len(controls)),
                "wilson_95": _wilson(len(reached), len(controls)),
                "first_reach_sim_median": None if not first_sims else statistics.median(first_sims),
                "first_reach_sim_p90": None if not first_sims else float(np.percentile(first_sims, 90)),
                "mean_reaching_path_fraction": statistics.fmean(
                    row["diagnostics"]["probe_paths_reaching_first_reveal_fraction"]
                    for row in controls
                ) if controls else 0.0,
                "mean_max_decision_depth": statistics.fmean(
                    row["diagnostics"]["probe_max_decision_depth"] for row in controls
                ) if controls else 0.0,
            },
            "arms": {},
        }
        for arm_name in ARM_ORDER[1:]:
            pairs = []
            arm_runs = []
            for control in controls:
                key = (int(control["position_index"]), budget, arm_name)
                if key not in by_key:
                    continue
                arm_run = by_key[key]
                arm_runs.append(arm_run)
                pairs.append(_paired_metrics(control, arm_run))
            tv = [row["visit_total_variation"] for row in pairs]
            changed = sum(row["top_action_changed"] for row in pairs)
            pick_changed = sum(row["top_pick_changed"] for row in pairs)
            committed = sum(
                row["diagnostics"]["probe_full_panel_committed"] == 1.0
                for row in arm_runs
            )
            deltas = [
                row["target_root_visit_share_delta"]
                for row in pairs
                if row["target_root_visit_share_delta"] is not None
            ]
            budget_summary["arms"][arm_name] = {
                "pairs_completed": len(pairs),
                "top_action_change_rate": changed / max(1, len(pairs)),
                "top_action_change_wilson_95": _wilson(changed, len(pairs)),
                "top_pick_change_rate": pick_changed / max(1, len(pairs)),
                "mean_visit_total_variation": statistics.fmean(tv) if tv else 0.0,
                "mean_visit_total_variation_bootstrap_95": _mean_ci(tv, budget + len(arm_name)),
                "mean_target_root_visit_share_delta": statistics.fmean(deltas) if deltas else None,
                "panel_commit_rate": committed / max(1, len(arm_runs)),
                "mean_nn_rows": statistics.fmean(row["python_nn_rows"] for row in arm_runs)
                if arm_runs else 0.0,
                "mean_elapsed_seconds": statistics.fmean(row["elapsed_seconds"] for row in arm_runs)
                if arm_runs else 0.0,
            }
        summary[str(budget)] = budget_summary
    return summary


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    positions_path = Path(args.positions_path)
    positions = load_frozen_positions(positions_path)
    positions = positions[: args.limit or None]
    budgets = tuple(args.budgets)
    checkpoint = Path(args.checkpoint)
    provenance = {
        "version": PROBE_VERSION,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "positions_path": str(positions_path),
        "positions_sha256": file_sha256(positions_path),
        "position_count_requested": len(positions),
        "budgets": list(budgets),
        "arms": list(ARM_ORDER),
        "budget_accounting": {
            "control_and_pulses": "B ordinary simulation paths",
            "panel_charged": "B-70 ordinary paths plus one complete 70-row bootstrap",
            "panel_extra": "B ordinary paths plus one complete 70-row bootstrap",
            "root_evaluation": "reported separately and common to every arm",
        },
        "common_random_numbers": "same seed across all arms for each position/budget",
        "leaf_batch": int(args.leaf_batch),
        "fpu": float(args.fpu),
        "cpuct": float(args.cpuct),
        "base_seed": int(args.search_seed),
    }
    output = Path(args.output)
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("provenance") != provenance:
            raise ValueError(f"existing artifact provenance differs: {output}")
    else:
        payload = {"provenance": provenance, "results": [], "summary": {}}
    completed = {
        (int(row["position_index"]), int(row["configured_budget"]), row["arm"])
        for row in payload["results"]
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
    # Exclude one-time device/kernel initialization from the first timed arm.
    _run_arm(
        positions[0][0], evaluator, arm="control", configured_budget=1,
        seed=args.search_seed, leaf_batch=1, fpu=args.fpu, cpuct=args.cpuct,
        margin_gain=margin_gain, alpha=alpha,
    )

    for budget in budgets:
        for position_index, (state, source) in enumerate(positions):
            if len(state.deck) != 8 or state.actor_index != 0:
                raise ValueError(f"position {position_index} is not a deck=8 first-selection root")
            seed = int(args.search_seed) + 104_729 * (position_index + 1) + 1_000_003 * budget
            for arm in ARM_ORDER:
                key = (position_index, budget, arm)
                if key in completed:
                    continue
                result = _run_arm(
                    state,
                    evaluator,
                    arm=arm,
                    configured_budget=budget,
                    seed=seed,
                    leaf_batch=args.leaf_batch,
                    fpu=args.fpu,
                    cpuct=args.cpuct,
                    margin_gain=margin_gain,
                    alpha=alpha,
                )
                result.update({"position_index": position_index, "position_source": source})
                payload["results"].append(result)
                payload["summary"] = build_summary(payload["results"], budgets)
                _atomic_json(output, payload)
                completed.add(key)
                print(
                    f"budget={budget} position={position_index + 1}/{len(positions)} "
                    f"arm={arm} reach={int(result['diagnostics']['probe_first_reveal_reached'])} "
                    f"first={int(result['diagnostics']['probe_first_reveal_simulation'])} "
                    f"panel={int(result['diagnostics']['probe_full_panel_committed'])} "
                    f"nn={result['python_nn_rows']} secs={result['elapsed_seconds']:.2f}",
                    flush=True,
                )
    payload["summary"] = build_summary(payload["results"], budgets)
    payload["completed"] = True
    _atomic_json(output, payload)
    return payload


def _parse_budgets(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or len(set(values)) != len(values) or any(value <= 70 for value in values):
        raise argparse.ArgumentTypeError("budgets must be unique integers greater than 70")
    return values


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "run"), default="run")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CURRENT_BEST))
    parser.add_argument("--positions-path", default=str(DEFAULT_POSITIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--budgets", type=_parse_budgets, default=DEFAULT_BUDGETS)
    parser.add_argument("--position-seed", type=int, default=20260830)
    parser.add_argument("--trajectory-sims", type=int, default=32)
    parser.add_argument("--trajectory-leaf-batch", type=int, default=1024)
    parser.add_argument("--search-seed", type=int, default=20260831)
    parser.add_argument("--leaf-batch", type=int, default=8)
    parser.add_argument("--fpu", type=float, default=-0.2)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.mode == "freeze":
        result = freeze_positions(args)
        print(
            f"frozen {result['positions']} deck=8 first-selection positions -> "
            f"{result['path']} sha={result['sha256']}",
            flush=True,
        )
    else:
        result = run_experiment(args)
        print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
