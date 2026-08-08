"""Equal-compute A1 probe for the one-reveal fixed-support search.

This is deliberately narrower than self-play: it compares root selections from
the incumbent open-loop search and opt-in one-reveal arms on frozen public
positions. Every candidate arm receives the same simulation budget and the
artifact records actual NN rows so terminal-leaf differences cannot be hidden.
A larger reference search is reported separately and is not called an exact
oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from games.kingdomino.denial_search import load_checkpoint_network
from games.kingdomino.denial_signal_sweep import (
    file_sha256,
    load_frozen_positions,
)
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.promotion import DEFAULT_CURRENT_BEST, sha256_file
from games.kingdomino.self_play import make_rust_evaluator


def _arm(
    state,
    evaluator,
    *,
    sims: int,
    exposure: int,
    enum_max_rows: int,
    seed: int,
    margin_gain: float,
    alpha: float,
    fpu: float,
    cpuct: float,
) -> dict[str, Any]:
    import kingdomino_rust as kr

    nn_evaluations = 0

    def counted_evaluator(my_board, opp_board, flat, legal_indices):
        nonlocal nn_evaluations
        nn_evaluations += int(my_board.shape[0])
        return evaluator(my_board, opp_board, flat, legal_indices)

    started = time.perf_counter()
    children, root_value0 = kr.advisor_open_loop_search(
        _rust_state_from_python(state),
        counted_evaluator,
        int(sims),
        dirichlet_eps=0.0,
        fpu=float(fpu),
        cpuct=float(cpuct),
        seed=int(seed) & 0xFFFF_FFFF_FFFF_FFFF,
        leaf_batch=8,
        virtual_loss=1,
        margin_gain=float(margin_gain),
        alpha=float(alpha),
        chance_exposure=int(exposure),
        chance_enum_max_rows=int(enum_max_rows),
    )
    actor = int(state.current_actor)
    rows = []
    for action_idx, visits, value_sum0, prior in children:
        q0 = None if not visits else float(value_sum0) / int(visits)
        rows.append({
            "action_idx": int(action_idx),
            "pick_rank": int(action_idx) % 5,
            "visits": int(visits),
            "q_actor": None if q0 is None else (q0 if actor == 0 else -q0),
            "prior": float(prior),
        })
    top = max(rows, key=lambda x: (x["visits"], x["prior"], -x["action_idx"]))
    return {
        "sims": int(sims),
        "chance_exposure": int(exposure),
        "chance_enum_max_rows": int(enum_max_rows),
        "root_value_player0": float(root_value0),
        "nn_evaluations": int(nn_evaluations),
        "top_action_idx": top["action_idx"],
        "top_pick_rank": top["pick_rank"],
        "elapsed_seconds": time.perf_counter() - started,
        "children": rows,
    }


def _reference_metrics(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    candidate_by_idx = {row["action_idx"]: row for row in candidate["children"]}
    reference_rows = [row for row in reference["children"] if row["q_actor"] is not None]
    reference_by_idx = {row["action_idx"]: row for row in reference_rows}
    common = sorted(set(candidate_by_idx) & set(reference_by_idx))
    selected = candidate["top_action_idx"]
    min_reference_visits = max(2, int(reference["sims"] * 0.01))
    established = [
        reference_by_idx[idx] for idx in common
        if reference_by_idx[idx]["visits"] >= min_reference_visits
    ]
    best_q = max((row["q_actor"] for row in established), default=None)
    selected_q = reference_by_idx.get(selected, {}).get("q_actor")
    q_pairs = 0
    q_correct = 0
    visit_pairs = 0
    visit_correct = 0
    for offset, left in enumerate(common):
        cq_left = candidate_by_idx[left]["q_actor"]
        rq_left = reference_by_idx[left]["q_actor"]
        if cq_left is None:
            continue
        for right in common[offset + 1:]:
            cq_right = candidate_by_idx[right]["q_actor"]
            rq_right = reference_by_idx[right]["q_actor"]
            if cq_right is None or abs(rq_left - rq_right) <= 1e-9:
                continue
            q_pairs += 1
            q_correct += int((cq_left - cq_right) * (rq_left - rq_right) > 0.0)
            candidate_visit_delta = (
                candidate_by_idx[left]["visits"] - candidate_by_idx[right]["visits"]
            )
            reference_visit_delta = (
                reference_by_idx[left]["visits"] - reference_by_idx[right]["visits"]
            )
            if candidate_visit_delta and reference_visit_delta:
                visit_pairs += 1
                visit_correct += int(candidate_visit_delta * reference_visit_delta > 0)
    return {
        "top1_agrees": selected == reference["top_action_idx"],
        "pick_agrees": candidate["top_pick_rank"] == reference["top_pick_rank"],
        "regret_actor": (
            None if best_q is None or selected_q is None else float(best_q - selected_q)
        ),
        "regret_reference_min_visits": min_reference_visits,
        "q_pairwise_pairs": q_pairs,
        "q_pairwise_correct": q_correct,
        "q_pairwise_accuracy": None if not q_pairs else q_correct / q_pairs,
        "visit_pairwise_pairs": visit_pairs,
        "visit_pairwise_correct": visit_correct,
        "visit_pairwise_accuracy": (
            None if not visit_pairs else visit_correct / visit_pairs
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    positions = load_frozen_positions(args.positions_path)
    if args.positions > 0:
        positions = positions[:args.positions]
    if not positions:
        raise ValueError("the frozen position set is empty")
    exposures = [int(x) for x in args.exposures.split(",")]
    if not exposures or exposures[0] != 0 or any(x < 0 for x in exposures):
        raise ValueError("--exposures must start with the incumbent 0 arm")

    net, checkpoint_cfg = load_checkpoint_network(checkpoint, args.device)
    evaluator = make_rust_evaluator(
        net,
        device=args.device,
        margin_gain=float(checkpoint_cfg.get("margin_gain", 2.0)),
        alpha=float(checkpoint_cfg.get("alpha", 0.5)),
    )
    margin_gain = float(checkpoint_cfg.get("margin_gain", 2.0))
    alpha = float(checkpoint_cfg.get("alpha", 0.5))
    # Exclude one-time CUDA/kernel initialization from the first arm's latency.
    _arm(
        positions[0][0], evaluator, sims=1, exposure=0,
        enum_max_rows=args.enum_max_rows, seed=args.seed,
        margin_gain=margin_gain, alpha=alpha,
        fpu=args.fpu, cpuct=args.cpuct,
    )
    records = []
    for index, (state, source) in enumerate(positions):
        seed = int(args.seed) + 104729 * (index + 1)
        arms = {
            f"x{exposure}": _arm(
                state,
                evaluator,
                sims=args.sims,
                exposure=exposure,
                enum_max_rows=args.enum_max_rows,
                seed=seed,
                margin_gain=margin_gain,
                alpha=alpha,
                fpu=args.fpu,
                cpuct=args.cpuct,
            )
            for exposure in exposures
        }
        reference_openloop = _arm(
            state,
            evaluator,
            sims=args.reference_sims,
            exposure=0,
            enum_max_rows=args.enum_max_rows,
            seed=seed,
            margin_gain=margin_gain,
            alpha=alpha,
            fpu=args.fpu,
            cpuct=args.cpuct,
        )
        reference_hybrid = _arm(
            state,
            evaluator,
            sims=args.reference_sims,
            exposure=args.reference_exposure,
            enum_max_rows=args.enum_max_rows,
            seed=seed,
            margin_gain=margin_gain,
            alpha=alpha,
            fpu=args.fpu,
            cpuct=args.cpuct,
        )
        references = {
            "openloop": reference_openloop,
            "hybrid": reference_hybrid,
        }
        metrics = {
            reference_name: {
                name: _reference_metrics(arm, reference)
                for name, arm in arms.items()
            }
            for reference_name, reference in references.items()
        }
        records.append({
            "position_index": index,
            "source": source,
            "deck_size": len(state.deck),
            "actor": int(state.current_actor),
            "arms": arms,
            "references": references,
            "reference_agreement": {
                "top1": (
                    reference_openloop["top_action_idx"]
                    == reference_hybrid["top_action_idx"]
                ),
                "pick": (
                    reference_openloop["top_pick_rank"]
                    == reference_hybrid["top_pick_rank"]
                ),
            },
            "metrics": metrics,
        })
        print(
            f"A1 {index + 1}/{len(positions)} deck={len(state.deck)} "
            + " ".join(
                f"{name}:a{arm['top_action_idx']}/p{arm['top_pick_rank']}"
                for name, arm in arms.items()
            )
        )

    aggregate = {}
    for reference_name in ("openloop", "hybrid"):
        aggregate[reference_name] = {}
        for exposure in exposures:
            name = f"x{exposure}"
            metrics = [record["metrics"][reference_name][name] for record in records]
            regrets = [m["regret_actor"] for m in metrics if m["regret_actor"] is not None]
            q_pair_correct = sum(m["q_pairwise_correct"] for m in metrics)
            q_pair_total = sum(m["q_pairwise_pairs"] for m in metrics)
            visit_pair_correct = sum(m["visit_pairwise_correct"] for m in metrics)
            visit_pair_total = sum(m["visit_pairwise_pairs"] for m in metrics)
            elapsed = [record["arms"][name]["elapsed_seconds"] for record in records]
            nn_evals = [record["arms"][name]["nn_evaluations"] for record in records]
            aggregate[reference_name][name] = {
                "positions": len(records),
                "top1_agreement": statistics.fmean(m["top1_agrees"] for m in metrics),
                "pick_agreement": statistics.fmean(m["pick_agrees"] for m in metrics),
                "q_pairwise_accuracy": (
                    None if not q_pair_total else q_pair_correct / q_pair_total
                ),
                "visit_pairwise_accuracy": (
                    None if not visit_pair_total else visit_pair_correct / visit_pair_total
                ),
                "mean_regret_actor": None if not regrets else statistics.fmean(regrets),
                "p90_regret_actor": (
                    None if not regrets else sorted(regrets)[math.ceil(0.9 * len(regrets)) - 1]
                ),
                "mean_seconds": statistics.fmean(elapsed),
                "mean_nn_evaluations": statistics.fmean(nn_evals),
            }
    aggregate["reference_agreement"] = {
        "top1": statistics.fmean(r["reference_agreement"]["top1"] for r in records),
        "pick": statistics.fmean(r["reference_agreement"]["pick"] for r in records),
    }
    return {
        "schema_version": 1,
        "scope": "A1 equal-compute frozen-position probe; no training or promotion",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "positions_path": str(args.positions_path),
        "positions_sha256": file_sha256(args.positions_path),
        "configuration": vars(args),
        "reference_label": "separate larger incumbent and one-reveal searches; neither is exact",
        "aggregate": aggregate,
        "positions": records,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CURRENT_BEST))
    parser.add_argument(
        "--positions-path",
        default="runs/kingdomino/denial_search/signal_positions.jsonl",
    )
    parser.add_argument("--output", default="runs/kingdomino/chance_correct_a1/search_probe.json")
    parser.add_argument("--positions", type=int, default=12)
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--exposures", default="0,1,2,4")
    parser.add_argument("--enum-max-rows", type=int, default=70)
    parser.add_argument("--reference-sims", type=int, default=512)
    parser.add_argument("--reference-exposure", type=int, default=4)
    parser.add_argument("--fpu", type=float, default=-0.2)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
