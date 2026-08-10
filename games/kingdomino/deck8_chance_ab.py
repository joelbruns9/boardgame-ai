"""Paired whole-game A/B for the production deck=8 chance panel.

The same frozen network and deck seeds play twice: treatment as seat 0, then
treatment as seat 1.  Only the treatment seat enables exhaustive deck=8 panels.
Separate all-off/all-on cohorts measure throughput without being interpreted as
strength games.  The output is one atomic, provenance-rich JSON artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from games.kingdomino.encoder import CANVAS_SIZE, FLAT_SIZE, NUM_BOARD_CHANNELS
from games.kingdomino.network import KingdominoNet
from games.kingdomino.round_robin_eval import (
    checkpoint_config,
    checkpoint_state_dict,
    load_checkpoint,
)
from games.kingdomino.self_play import make_rust_evaluator


SCHEMA_VERSION = 1
TREATMENT = "deck8_panel"
CONTROL = "open_loop_sampled"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def bootstrap_mean_interval(
    values: list[float], *, seed: int, resamples: int = 20_000
) -> dict[str, float]:
    """Paired-seed bootstrap interval; the seed pair is the resampling unit."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def summarize_match(
    treatment_seat0: list[dict[str, int]],
    treatment_seat1: list[dict[str, int]],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_orientation = {
        0: {row["seed"]: row for row in treatment_seat0},
        1: {row["seed"]: row for row in treatment_seat1},
    }
    if set(by_orientation[0]) != set(by_orientation[1]):
        raise ValueError("paired orientations do not contain identical seed sets")

    wins = losses = draws = 0
    seat_rows: dict[str, dict[str, float | int]] = {}
    pair_points: list[float] = []
    pair_margins: list[float] = []
    pairs: list[dict[str, Any]] = []
    for treatment_seat in (0, 1):
        orientation = list(by_orientation[treatment_seat].values())
        outcomes = [row["outcome0"] if treatment_seat == 0 else -row["outcome0"]
                    for row in orientation]
        margins = [row["score0"] - row["score1"] if treatment_seat == 0
                   else row["score1"] - row["score0"] for row in orientation]
        seat_rows[str(treatment_seat)] = {
            "games": len(orientation),
            "wins": sum(value > 0 for value in outcomes),
            "losses": sum(value < 0 for value in outcomes),
            "draws": sum(value == 0 for value in outcomes),
            "mean_margin": float(np.mean(margins)),
        }

    for seed in sorted(by_orientation[0]):
        games = []
        points = []
        margins = []
        for treatment_seat in (0, 1):
            row = by_orientation[treatment_seat][seed]
            outcome = row["outcome0"] if treatment_seat == 0 else -row["outcome0"]
            margin = (row["score0"] - row["score1"] if treatment_seat == 0
                      else row["score1"] - row["score0"])
            wins += int(outcome > 0)
            losses += int(outcome < 0)
            draws += int(outcome == 0)
            points.append(1.0 if outcome > 0 else 0.5 if outcome == 0 else 0.0)
            margins.append(float(margin))
            games.append({
                "treatment_seat": treatment_seat,
                "treatment_score": row[f"score{treatment_seat}"],
                "control_score": row[f"score{1 - treatment_seat}"],
                "treatment_outcome": int(outcome),
                "treatment_margin": int(margin),
            })
        pair_points.append(float(np.mean(points)))
        pair_margins.append(float(np.mean(margins)))
        pairs.append({
            "seed": seed,
            "paired_points": pair_points[-1],
            "paired_margin": pair_margins[-1],
            "games": games,
        })

    games = wins + losses + draws
    decisive = wins + losses
    return {
        "paired_seeds": len(pairs),
        "games": games,
        "treatment_wins": wins,
        "control_wins": losses,
        "draws": draws,
        "treatment_points_rate": (wins + 0.5 * draws) / games,
        "treatment_decisive_win_rate": wins / decisive if decisive else 0.5,
        "paired_points": bootstrap_mean_interval(
            pair_points, seed=bootstrap_seed
        ),
        "paired_score_margin": bootstrap_mean_interval(
            pair_margins, seed=bootstrap_seed + 1
        ),
        "seat_breakdown": seat_rows,
        "pairs": pairs,
    }


class CountingEvaluator:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.batch_sizes: list[int] = []

    def __call__(self, my, opp, flat, legal):
        self.batch_sizes.append(int(len(my)))
        return self.evaluator(my, opp, flat, legal)

    def summary(self) -> dict[str, Any]:
        sizes = np.asarray(self.batch_sizes, dtype=np.int64)
        histogram = Counter(int(value) for value in self.batch_sizes)
        return {
            "calls": len(self.batch_sizes),
            "rows": int(sizes.sum()) if sizes.size else 0,
            "mean_batch": float(sizes.mean()) if sizes.size else 0.0,
            "p50_batch": float(np.quantile(sizes, 0.50)) if sizes.size else 0.0,
            "p90_batch": float(np.quantile(sizes, 0.90)) if sizes.size else 0.0,
            "max_batch": int(sizes.max()) if sizes.size else 0,
            "batch_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        }


def run_cohort(
    evaluator,
    *,
    mode: str,
    treatment_seat: int | None,
    n_games: int,
    seed_start: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    import kingdomino_rust

    if mode not in {"off", "both", "seat"}:
        raise ValueError(f"unknown chance mode: {mode}")
    if mode == "seat" and treatment_seat not in (0, 1):
        raise ValueError("seat mode requires treatment_seat 0 or 1")
    counted = CountingEvaluator(evaluator)
    mcts = kingdomino_rust.BatchedMCTS(
        min(int(settings["batch_slots"]), n_games),
        n_games,
        seed_start,
        int(settings["sims"]),
        leaf_batch=int(settings["leaf_batch"]),
        virtual_loss=1,
        cpuct=float(settings["c_puct"]),
        fpu=float(settings["fpu"]),
        dirichlet_alpha=0.3,
        dirichlet_eps=0.0,
        temp_moves=0,
        open_loop=True,
        score_scale=float(settings["score_scale"]),
        margin_gain=float(settings["margin_gain"]),
        alpha=float(settings["alpha"]),
        exact_endgame_max_secs=float(settings["exact_endgame_max_secs"]),
        async_solve=True,
        solver_cpus=int(settings["solver_cpus"]),
        deck8_chance_enumeration=(mode != "off"),
        deck8_chance_enumeration_seat=(treatment_seat if mode == "seat" else -1),
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    results: list[dict[str, int]] = []
    ticks = 0
    last_report = 0
    while not mcts.done():
        my, opp, flat, legal = mcts.step()
        values, gathered = counted(my, opp, flat, legal)
        for seed, _examples, scores in mcts.update(values, gathered):
            results.append({
                "seed": int(seed),
                "score0": int(scores[0]),
                "score1": int(scores[1]),
                "outcome0": int(scores[2]),
            })
        if len(results) >= last_report + max(1, n_games // 4):
            last_report = len(results)
            print(f"  {mode} seat={treatment_seat}: {len(results)}/{n_games} games", flush=True)
        ticks += 1
        if ticks > 2_000_000:
            raise RuntimeError("A/B cohort exceeded tick guard")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    results.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in results] != list(range(seed_start, seed_start + n_games)):
        raise RuntimeError("cohort returned an incomplete or unexpected seed set")
    return {
        "mode": mode,
        "treatment_seat": treatment_seat,
        "games": results,
        "wall_seconds": seconds,
        "games_per_second": n_games / seconds,
        "inference": counted.summary(),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "deck8_chance_panel_count": int(mcts.deck8_chance_panel_count),
        "deck8_chance_bootstrap_rows": int(mcts.deck8_chance_bootstrap_rows),
        "deck8_chance_budget_blocked_count": int(
            mcts.deck8_chance_budget_blocked_count
        ),
        "exact_solve_count": int(mcts.exact_solve_count),
        "exact_fallback_count": int(mcts.exact_fallback_count),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="runs/kingdomino/best_checkpoint/current_best.pt",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--paired-seeds", type=int, default=64)
    parser.add_argument("--throughput-games", type=int, default=16)
    # Deliberately disjoint from the 2026081200 smoke block, whose outcomes are
    # visible during harness validation.
    parser.add_argument("--seed", type=int, default=2_026_082_200)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--batch-slots", type=int, default=32)
    parser.add_argument("--leaf-batch", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-inference", action="store_true")
    parser.add_argument("--solver-cpus", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    if output.exists() or output.with_suffix(output.suffix + ".tmp").exists():
        raise FileExistsError(f"refusing to overwrite existing A/B output: {output}")
    if args.paired_seeds <= 0 or args.throughput_games <= 0:
        raise ValueError("paired and throughput game counts must be positive")
    if args.sims < 71:
        raise ValueError("treatment requires at least 71 budget units")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    repo = Path(__file__).resolve().parents[2]
    checkpoint_path = (repo / args.checkpoint).resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint_config(checkpoint)
    architecture = {
        "channels": int(config.get("channels", 96)),
        "blocks": int(config.get("blocks", 8)),
        "bilinear_dim": int(config.get("bilinear_dim", 64)),
    }
    net = KingdominoNet(**architecture)
    net.load_state_dict(checkpoint_state_dict(checkpoint))
    net.to(args.device).eval()
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(config.get("allow_tf32", True))

    evaluator = make_rust_evaluator(
        net,
        device=args.device,
        amp=bool(args.amp_inference),
        margin_gain=float(config.get("margin_gain", 2.0)),
        alpha=float(config.get("alpha", 0.5)),
    )
    # Warm CUDA kernels and allocations outside every measured cohort.
    board = np.zeros(
        (1, NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32
    )
    flat = np.zeros((1, FLAT_SIZE), dtype=np.float32)
    evaluator(board, board, flat, [np.asarray([0], dtype=np.int64)])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    settings = {
        "sims": int(args.sims),
        "budget_definition": "panel rows and real paths share the same move budget",
        "batch_slots": int(args.batch_slots),
        "leaf_batch": int(args.leaf_batch),
        "c_puct": float(config.get("c_puct", 1.5)),
        "fpu": float(config.get("fpu", -0.2)),
        "score_scale": float(config.get("score_scale", 160.0)),
        "margin_gain": float(config.get("margin_gain", 2.0)),
        "alpha": float(config.get("alpha", 0.5)),
        "exact_endgame_max_secs": float(config.get("exact_endgame_max_secs", 3.0)),
        "solver_cpus": int(args.solver_cpus),
        "dirichlet_epsilon": 0.0,
        "temperature": 0.0,
        "open_loop": True,
        "amp_inference": bool(args.amp_inference),
    }

    print("Running direct paired strength match...", flush=True)
    treatment_seat0 = run_cohort(
        evaluator, mode="seat", treatment_seat=0,
        n_games=args.paired_seeds, seed_start=args.seed, settings=settings,
    )
    treatment_seat1 = run_cohort(
        evaluator, mode="seat", treatment_seat=1,
        n_games=args.paired_seeds, seed_start=args.seed, settings=settings,
    )
    print("Running matched throughput cohorts...", flush=True)
    throughput_seed = args.seed + 1_000_000
    throughput_off = run_cohort(
        evaluator, mode="off", treatment_seat=None,
        n_games=args.throughput_games, seed_start=throughput_seed, settings=settings,
    )
    throughput_on = run_cohort(
        evaluator, mode="both", treatment_seat=None,
        n_games=args.throughput_games, seed_start=throughput_seed, settings=settings,
    )

    match = summarize_match(
        treatment_seat0["games"], treatment_seat1["games"],
        bootstrap_seed=args.seed + 2_000_000,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "hypothesis": "deck8 panel improves whole-game strength at a matched 400-unit move budget",
        "treatment": TREATMENT,
        "control": CONTROL,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "architecture": architecture,
        },
        "provenance": {
            "git_commit": git_commit(repo),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "rust_sha256": sha256_file(
                repo / "games/kingdomino/kingdomino_rust/src/lib.rs"
            ),
        },
        "settings": settings,
        "match": match,
        "strength_cohorts": {
            "treatment_seat0": treatment_seat0,
            "treatment_seat1": treatment_seat1,
        },
        "throughput": {
            "shared_seed_start": throughput_seed,
            "control_all_off": throughput_off,
            "treatment_all_on": throughput_on,
            "treatment_to_control_wall_ratio": (
                throughput_on["wall_seconds"] / throughput_off["wall_seconds"]
            ),
            "treatment_to_control_games_per_second_ratio": (
                throughput_on["games_per_second"] / throughput_off["games_per_second"]
            ),
        },
        "interpretation_rule": (
            "Use paired points and score-margin intervals as the primary strength screen; "
            "do not claim improvement if both intervals include the null. Throughput cohorts "
            "measure cost only and are not strength evidence."
        ),
    }
    atomic_json(output, payload)
    print(json.dumps({
        "output": str(output),
        "wins": match["treatment_wins"],
        "losses": match["control_wins"],
        "draws": match["draws"],
        "points": match["paired_points"],
        "margin": match["paired_score_margin"],
        "throughput_ratio": payload["throughput"]["treatment_to_control_games_per_second_ratio"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
