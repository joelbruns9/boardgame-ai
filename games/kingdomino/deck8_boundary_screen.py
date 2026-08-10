"""Screen frozen deck-8 continuations for stable search disagreements.

This is a selection harness, not an oracle.  It derives both incumbent- and
lazy-one-reveal-directed continuations to the final pre-reveal actor, deduplicates
the resulting public boundaries, and repeats both searches on paired seeds.
Exact solving is reserved for boundaries where both arms are individually
stable but select different actions.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from games.kingdomino.action_codec import encode_action
from games.kingdomino.chance_correct_search_probe import _arm
from games.kingdomino.deck8_oracle import validate_boundary_state
from games.kingdomino.denial_search import chance_public_state_key_v1, load_checkpoint_network
from games.kingdomino.denial_signal_sweep import file_sha256, load_frozen_positions
from games.kingdomino.promotion import DEFAULT_CURRENT_BEST, sha256_file
from games.kingdomino.self_play import make_rust_evaluator


SCHEMA_VERSION = "kd-deck8-boundary-screen-v1"
ARM_SPECS = {
    "x0": {"exposure": 0, "backup": "hajek", "traversal": "balanced"},
    "x1_hajek_balanced": {
        "exposure": 1,
        "backup": "hajek",
        "traversal": "balanced",
    },
}


def selection_summary(action_indices: list[int]) -> dict[str, Any]:
    if not action_indices:
        raise ValueError("selection summary requires at least one action")
    counts = {idx: action_indices.count(idx) for idx in sorted(set(action_indices))}
    mode = min(counts, key=lambda idx: (-counts[idx], idx))
    return {
        "actions": [int(x) for x in action_indices],
        "counts": {str(idx): count for idx, count in counts.items()},
        "mode": int(mode),
        "mode_count": int(counts[mode]),
        "mode_fraction": float(counts[mode] / len(action_indices)),
        "unanimous": len(counts) == 1,
    }


def disagreement_verdict(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    x0 = arms["x0"]["selection"]
    x1 = arms["x1_hajek_balanced"]["selection"]
    stable = bool(x0["unanimous"] and x1["unanimous"])
    disagree = int(x0["mode"]) != int(x1["mode"])
    return {
        "both_arms_unanimous": stable,
        "modal_disagreement": disagree,
        "stable_disagreement": stable and disagree,
    }


def _select(
    state,
    evaluator,
    *,
    spec: dict[str, Any],
    sims: int,
    seed: int,
    margin_gain: float,
    alpha: float,
    fpu: float,
    cpuct: float,
) -> tuple[int, dict[str, Any]]:
    result = _arm(
        state,
        evaluator,
        sims=sims,
        exposure=int(spec["exposure"]),
        enum_max_rows=70,
        seed=seed,
        margin_gain=margin_gain,
        alpha=alpha,
        fpu=fpu,
        cpuct=cpuct,
        backup=str(spec["backup"]),
        traversal=str(spec["traversal"]),
    )
    return int(result["top_action_idx"]), result


def _derive_boundary(
    root,
    evaluator,
    *,
    spec: dict[str, Any],
    sims: int,
    seed: int,
    margin_gain: float,
    alpha: float,
    fpu: float,
    cpuct: float,
) -> tuple[Any, list[int], list[dict[str, Any]]]:
    state = root.copy()
    prefix: list[int] = []
    steps = []
    ply = 0
    while state.actor_index < len(state.pending_claims) - 1:
        action_idx, search = _select(
            state, evaluator, spec=spec, sims=sims, seed=seed + ply,
            margin_gain=margin_gain, alpha=alpha, fpu=fpu, cpuct=cpuct,
        )
        legal = {int(encode_action(action, state)): action for action in state.legal_actions()}
        if action_idx not in legal:
            raise RuntimeError(f"search selected illegal action {action_idx}")
        steps.append({
            "ply": ply,
            "actor": int(state.current_actor),
            "action_idx": action_idx,
            "legal_actions": len(legal),
            "nn_evaluations": int(search["nn_evaluations"]),
            "elapsed_seconds": float(search["elapsed_seconds"]),
        })
        prefix.append(action_idx)
        state = state.step(legal[action_idx])
        ply += 1
    validate_boundary_state(state)
    return state, prefix, steps


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"screen output already exists: {output}")
    positions_path = Path(args.positions_path)
    positions = load_frozen_positions(positions_path)
    requested = [
        int(x) for x in str(args.position_indices).split(",") if x.strip()
    ]
    if not requested:
        requested = [i for i, (state, _source) in enumerate(positions) if len(state.deck) == 8]
    for index in requested:
        if not 0 <= index < len(positions):
            raise ValueError(f"position index {index} outside corpus")
        if len(positions[index][0].deck) != 8:
            raise ValueError(f"position {index} is not deck=8")

    net, checkpoint_cfg = load_checkpoint_network(args.checkpoint, args.device)
    margin_gain = float(checkpoint_cfg.get("margin_gain", 2.0))
    alpha = float(checkpoint_cfg.get("alpha", 0.5))
    evaluator = make_rust_evaluator(
        net, device=args.device, margin_gain=margin_gain, alpha=alpha
    )
    # Warm up outside recorded searches.
    _select(
        positions[requested[0]][0], evaluator, spec=ARM_SPECS["x0"], sims=1,
        seed=int(args.seed), margin_gain=margin_gain, alpha=alpha,
        fpu=float(args.fpu), cpuct=float(args.cpuct),
    )

    boundary_by_key: dict[str, dict[str, Any]] = {}
    for position_index in requested:
        root, source = positions[position_index]
        for line_offset, line_arm in enumerate(ARM_SPECS):
            boundary, prefix, steps = _derive_boundary(
                root,
                evaluator,
                spec=ARM_SPECS[line_arm],
                sims=int(args.sims),
                seed=int(args.seed) + 10_000 * position_index + 100 * line_offset,
                margin_gain=margin_gain,
                alpha=alpha,
                fpu=float(args.fpu),
                cpuct=float(args.cpuct),
            )
            key = chance_public_state_key_v1(boundary).hex()
            origin = {
                "position_index": position_index,
                "source": source,
                "line_arm": line_arm,
                "prefix_actions": prefix,
                "steps": steps,
            }
            if key in boundary_by_key:
                boundary_by_key[key]["origins"].append(origin)
            else:
                boundary_by_key[key] = {
                    "state": boundary,
                    "boundary_key_hex": key,
                    "actor": int(boundary.current_actor),
                    "legal_actions": len(boundary.legal_actions()),
                    "current_row": [int(x) for x in boundary.current_row],
                    "deck": sorted(int(x) for x in boundary.deck),
                    "origins": [origin],
                }

    boundary_rows = []
    for boundary_index, key in enumerate(sorted(boundary_by_key)):
        entry = boundary_by_key[key]
        state = entry.pop("state")
        arms = {}
        for arm_offset, (name, spec) in enumerate(ARM_SPECS.items()):
            searches = []
            selected = []
            for seed_index in range(int(args.seed_count)):
                seed = int(args.seed) + 1_000_000 + 10_000 * boundary_index + seed_index
                action_idx, result = _select(
                    state, evaluator, spec=spec, sims=int(args.sims), seed=seed,
                    margin_gain=margin_gain, alpha=alpha, fpu=float(args.fpu),
                    cpuct=float(args.cpuct),
                )
                selected.append(action_idx)
                searches.append({
                    "seed_index": seed_index,
                    "seed": seed,
                    "top_action_idx": action_idx,
                    "nn_evaluations": int(result["nn_evaluations"]),
                    "elapsed_seconds": float(result["elapsed_seconds"]),
                })
            arms[name] = {
                "selection": selection_summary(selected),
                "mean_nn_evaluations": statistics.fmean(
                    row["nn_evaluations"] for row in searches
                ),
                "mean_seconds": statistics.fmean(row["elapsed_seconds"] for row in searches),
                "searches": searches,
            }
        entry["boundary_index"] = boundary_index
        entry["arms"] = arms
        entry["verdict"] = disagreement_verdict(arms)
        boundary_rows.append(entry)

    result = {
        "schema_version": SCHEMA_VERSION,
        "scope": "search-disagreement selection only; no exact values",
        "selection_reason": str(args.selection_reason),
        "positions_path": str(positions_path.resolve()),
        "positions_sha256": file_sha256(positions_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "configuration": {
            "position_indices": requested,
            "sims": int(args.sims),
            "seed": int(args.seed),
            "seed_count": int(args.seed_count),
            "fpu": float(args.fpu),
            "cpuct": float(args.cpuct),
            "arms": ARM_SPECS,
        },
        "derived_lines": 2 * len(requested),
        "unique_boundaries": len(boundary_rows),
        "stable_disagreements": sum(
            int(row["verdict"]["stable_disagreement"]) for row in boundary_rows
        ),
        "boundaries": boundary_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positions-path",
        default="runs/kingdomino/denial_search/signal_positions.jsonl",
    )
    parser.add_argument("--position-indices", default="")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CURRENT_BEST))
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-reason", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sims", type=int, default=4800)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--fpu", type=float, default=-0.2)
    parser.add_argument("--cpuct", type=float, default=1.5)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    compact = {
        "derived_lines": result["derived_lines"],
        "unique_boundaries": result["unique_boundaries"],
        "stable_disagreements": result["stable_disagreements"],
        "disagreements": [
            {
                "boundary_index": row["boundary_index"],
                "origins": row["origins"],
                "x0": row["arms"]["x0"]["selection"],
                "x1": row["arms"]["x1_hajek_balanced"]["selection"],
            }
            for row in result["boundaries"]
            if row["verdict"]["stable_disagreement"]
        ],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
