"""P3 laptop advisor throughput benchmark for capacity candidates.

Runs the same Rust open-loop search and leaf-batch setting as the live advisor
on a frozen difficult-root suite.  Candidate weights may be random because P3
measures architectural throughput, not strength.  The 80x6 baseline uses the
incumbent checkpoint.  Fixed-simulation timings are converted to projected
15-second and 60-second counts, with a predeclared maximum 3x slowdown gate.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import kingdomino_rust as kr
import numpy as np
import torch

from games.kingdomino.bga_reanalysis_corpus import DEFAULT_CORPUS
from games.kingdomino.capacity_bakeoff import ArmSpec, parse_arm
from games.kingdomino.deep_target_screen import _read_jsonl
from games.kingdomino.denial_search import load_checkpoint_network
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.late_model_policy_audit import prepare_bga_state
from games.kingdomino.network import KingdominoNet
from games.kingdomino.self_play import make_rust_evaluator


DEFAULT_SUITE = Path(
    "runs/kingdomino/placement_audit/deep_target_stage3_cohort_v2.json"
)
DEFAULT_CHECKPOINT = Path("runs/kingdomino/best_checkpoint/current_best.pt")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(position_id: str, repeat: int, base_seed: int) -> int:
    payload = f"p3:{position_id}:{repeat}:{base_seed}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def summarize(
    rows: list[dict[str, Any]], budgets: list[float], max_slowdown: float
) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    if "80x6" not in by_arm:
        raise ValueError("P3 requires the 80x6 baseline")
    baseline_by_key = {
        (row["position_id"], row["repeat"]): row for row in by_arm["80x6"]
    }
    output: dict[str, Any] = {}
    for arm, arm_rows in by_arm.items():
        total_sims = sum(int(row["simulations"]) for row in arm_rows)
        total_seconds = sum(float(row["elapsed_seconds"]) for row in arm_rows)
        rate = total_sims / total_seconds
        slowdowns = [
            float(row["elapsed_seconds"])
            / float(baseline_by_key[(row["position_id"], row["repeat"])]["elapsed_seconds"])
            for row in arm_rows
        ]
        overall_slowdown = total_seconds / sum(
            float(baseline_by_key[(row["position_id"], row["repeat"])]["elapsed_seconds"])
            for row in arm_rows
        )
        output[arm] = {
            "searches": len(arm_rows),
            "total_simulations": total_sims,
            "total_seconds": total_seconds,
            "simulations_per_second": rate,
            "overall_slowdown_vs_80x6": overall_slowdown,
            "median_paired_slowdown": statistics.median(slowdowns),
            "max_paired_slowdown": max(slowdowns),
            "projected_simulations": {
                str(int(budget) if budget.is_integer() else budget): rate * budget
                for budget in budgets
            },
            "deployment_floor_pass": overall_slowdown <= max_slowdown,
        }
    return output


def _network_for_arm(
    spec: ArmSpec, checkpoint: Path, device: str, random_seed: int
) -> tuple[KingdominoNet, str]:
    if spec.name == "80x6":
        net, _config = load_checkpoint_network(checkpoint, device)
        return net, str(checkpoint)
    torch.manual_seed(random_seed + spec.channels * 100 + spec.blocks)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed + spec.channels * 100 + spec.blocks)
    net = KingdominoNet(
        channels=spec.channels,
        blocks=spec.blocks,
        global_pooling=spec.global_pooling,
    ).to(device).eval()
    return net, f"random_seed:{random_seed}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", default="80x6,80x6+gp,128x8+gp,128x10+gp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--positions", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=1_600)
    parser.add_argument("--warmup-simulations", type=int, default=128)
    parser.add_argument("--leaf-batch", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=20260813)
    parser.add_argument("--random-weight-seed", type=int, default=20260813)
    parser.add_argument("--budgets", default="15,60")
    parser.add_argument("--max-slowdown", type=float, default=3.0)
    args = parser.parse_args()

    if args.leaf_batch != 8:
        raise ValueError("the live advisor harness uses leaf_batch=8")
    specs = [parse_arm(value) for value in args.arms.split(",") if value.strip()]
    budgets = [float(value) for value in args.budgets.split(",") if value.strip()]
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    entries = [
        entry for entry in suite["entries"] if int(entry.get("deck_count", 0)) > 4
    ][: args.positions]
    corpus = {str(row["position_id"]): row for row in _read_jsonl(args.corpus)}
    prepared = []
    for entry in entries:
        source = corpus[str(entry["position_id"])]
        rules = source["state"].get("rules", {})
        state = prepare_bga_state(
            source["state"],
            harmony=bool(rules.get("harmony", True)),
            middle=bool(rules.get("middle_kingdom", True)),
        )
        if len(state.deck) <= 4:
            raise ValueError("P3 suite manifest deck count disagrees with reconstructed state")
        prepared.append((entry, _rust_state_from_python(state)))

    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    rows: list[dict[str, Any]] = []
    for spec in specs:
        net, weight_source = _network_for_arm(
            spec, args.checkpoint, args.device, args.random_weight_seed
        )
        evaluator = make_rust_evaluator(net, device=args.device, alpha=0.5)
        warm_entry, warm_state = prepared[0]
        kr.advisor_open_loop_search(
            warm_state,
            evaluator,
            args.warmup_simulations,
            dirichlet_eps=0.0,
            cpuct=1.5,
            seed=_seed(str(warm_entry["position_id"]), -1, args.base_seed),
            leaf_batch=args.leaf_batch,
            alpha=0.5,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        for entry, rust_state in prepared:
            for repeat in range(args.repeats):
                search_seed = _seed(
                    str(entry["position_id"]), repeat, args.base_seed
                )
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                started = time.perf_counter()
                kr.advisor_open_loop_search(
                    rust_state,
                    evaluator,
                    args.simulations,
                    dirichlet_eps=0.0,
                    cpuct=1.5,
                    seed=search_seed,
                    leaf_batch=args.leaf_batch,
                    alpha=0.5,
                )
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                row = {
                    "arm": spec.name,
                    "position_id": str(entry["position_id"]),
                    "repeat": repeat,
                    "seed": search_seed,
                    "simulations": args.simulations,
                    "elapsed_seconds": elapsed,
                    "simulations_per_second": args.simulations / elapsed,
                    "weight_source": weight_source,
                }
                rows.append(row)
                print(
                    f"{spec.name} {entry['position_id']} r{repeat}: "
                    f"{args.simulations / elapsed:.1f} sims/s",
                    flush=True,
                )
        del evaluator, net
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    arm_summary = summarize(rows, budgets, args.max_slowdown)
    result = {
        "schema": "kingdomino-advisor-capacity-throughput/v1",
        "device": args.device,
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if args.device.startswith("cuda") and torch.cuda.is_available()
            else None
        ),
        "suite": str(args.suite),
        "suite_sha256": _sha256(args.suite),
        "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "positions": len(prepared),
        "repeats": args.repeats,
        "simulations_per_search": args.simulations,
        "leaf_batch": args.leaf_batch,
        "wall_clock_budgets_seconds": budgets,
        "maximum_slowdown_floor": args.max_slowdown,
        "5090_selfplay_measurement": "deferred until rental; hardware unavailable locally",
        "arms": arm_summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(arm_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
