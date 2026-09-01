"""Frozen-buffer capacity Test A for Kingdomino.

Every arm sees the same contiguous-block train/holdout split, optimization
seeds, sampled minibatches, number of example presentations, and two-point LR
grid.  Policy CE is the primary selection metric.  Final uncertainty is a
paired contiguous-block bootstrap over seed-mean held-out metrics; it is not a
game-clustered interval because the legacy Example format has no game IDs.

Example::

    python -m games.kingdomino.capacity_bakeoff \
      --buffer runs/kingdomino/cloud_80x6_run10/buffer_final.pkl \
      --out_dir runs/kingdomino/final_capacity_test_a \
      --arms 80x6,80x6+gp,128x8+gp,128x10+gp --device cuda
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from games.kingdomino.network import KingdominoNet
from games.kingdomino.self_play import (
    Example,
    FLAT_LAYOUT,
    NUM_JOINT_ACTIONS,
    ReplayBuffer,
    masked_log_softmax,
    train_step,
)


# Old self-play pickles can reference Example through __main__.
sys.modules["__main__"].Example = Example


@dataclass(frozen=True)
class ArmSpec:
    name: str
    channels: int
    blocks: int
    global_pooling: bool

    @property
    def slug(self) -> str:
        return self.name.replace("+", "_")


def parse_arm(value: str) -> ArmSpec:
    match = re.fullmatch(r"(\d+)x(\d+)(\+gp)?", value.strip().lower())
    if not match:
        raise ValueError(f"invalid arm {value!r}; expected CHANNELSxBLOCKS[+gp]")
    channels, blocks = int(match.group(1)), int(match.group(2))
    pooling = bool(match.group(3))
    name = f"{channels}x{blocks}{'+gp' if pooling else ''}"
    return ArmSpec(name, channels, blocks, pooling)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_examples(path: Path) -> list[Any]:
    import pickle

    with path.open("rb") as handle:
        return pickle.load(handle)["data"]


def block_split(
    examples: list[Any], holdout_frac: float, block: int, seed: int
) -> tuple[list[Any], list[Any], list[tuple[int, list[Any]]], list[int]]:
    """Return train rows, flat holdout, labeled holdout blocks, and IDs."""
    n_blocks = (len(examples) + block - 1) // block
    rng = np.random.default_rng(seed)
    holdout_ids = sorted(
        rng.choice(
            n_blocks,
            size=max(1, int(round(n_blocks * holdout_frac))),
            replace=False,
        ).tolist()
    )
    holdout_set = set(holdout_ids)
    train: list[Any] = []
    holdout: list[Any] = []
    holdout_blocks: list[tuple[int, list[Any]]] = []
    for block_id in range(n_blocks):
        chunk = examples[block_id * block : (block_id + 1) * block]
        if block_id in holdout_set:
            holdout.extend(chunk)
            holdout_blocks.append((block_id, chunk))
        else:
            train.extend(chunk)
    return train, holdout, holdout_blocks, holdout_ids


def densify_eval(examples: list[Any], device: str, batch: int = 512):
    """Yield deterministic, unaugmented evaluation batches."""
    for offset in range(0, len(examples), batch):
        chunk = examples[offset : offset + batch]
        mbs, obs, flats, pols, masks = [], [], [], [], []
        own_ss, opp_ss, win_ts = [], [], []
        for example in chunk:
            policy = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)
            policy[example.policy_idx] = example.policy_val
            mask = np.zeros(NUM_JOINT_ACTIONS, dtype=bool)
            mask[example.legal_idx] = True
            mbs.append(example.my_board.astype(np.float32))
            obs.append(example.opp_board.astype(np.float32))
            flats.append(example.flat.astype(np.float32))
            pols.append(policy)
            masks.append(mask)
            own_ss.append(float(example.own_score))
            opp_ss.append(float(example.opp_score))
            win_ts.append(float(example.win_target))

        def to(values):
            return torch.from_numpy(np.stack(values)).to(device)

        yield (
            to(mbs).float(),
            to(obs).float(),
            to(flats).float(),
            to(pols).float(),
            to(masks),
            torch.tensor(own_ss, dtype=torch.float32, device=device),
            torch.tensor(opp_ss, dtype=torch.float32, device=device),
            torch.tensor(win_ts, dtype=torch.float32, device=device),
        )


@torch.no_grad()
def evaluate(
    net: KingdominoNet,
    examples: list[Any],
    device: str,
    score_scale: float,
    lambda_score: float,
    lambda_w: float,
) -> dict[str, float]:
    """Held-out policy CE, score losses, win losses, and per-phase CE."""
    net.eval()
    progress_index = FLAT_LAYOUT["game_progress"].start
    totals = {
        "n": 0,
        "policy_ce": 0.0,
        "own_mse": 0.0,
        "opp_mse": 0.0,
        "win_bce": 0.0,
        "win_brier": 0.0,
    }
    phase = {name: [0.0, 0] for name in ("early", "mid", "end")}
    for my, opp, flat, policy, mask, own_t, opp_t, win_t in densify_eval(
        examples, device
    ):
        count = my.shape[0]
        own_p, opp_p, win_p, logits = net(my, opp, flat)
        logp = masked_log_softmax(logits, mask)
        logp = torch.where(mask, logp, torch.zeros_like(logp))
        ce_rows = -(policy * logp).sum(dim=1)
        totals["policy_ce"] += float(ce_rows.sum())
        totals["own_mse"] += float(
            F.mse_loss(own_p, own_t / score_scale, reduction="sum")
        )
        totals["opp_mse"] += float(
            F.mse_loss(opp_p, opp_t / score_scale, reduction="sum")
        )
        win_p = win_p.clamp(1e-6, 1 - 1e-6)
        totals["win_bce"] += float(
            F.binary_cross_entropy(win_p, win_t, reduction="sum")
        )
        totals["win_brier"] += float(((win_p - win_t) ** 2).sum())
        totals["n"] += count
        progress = flat[:, progress_index]
        for name, low, high in (
            ("early", -1.0, 1 / 3),
            ("mid", 1 / 3, 2 / 3),
            ("end", 2 / 3, 10.0),
        ):
            selected = (progress >= low) & (progress < high)
            if selected.any():
                phase[name][0] += float(ce_rows[selected].sum())
                phase[name][1] += int(selected.sum())
    count = max(1, int(totals["n"]))
    result = {key: value / count for key, value in totals.items() if key != "n"}
    result["combined"] = (
        result["policy_ce"]
        + lambda_score * (result["own_mse"] + result["opp_mse"])
        + lambda_w * result["win_bce"]
    )
    for name, (total, phase_count) in phase.items():
        result[f"policy_ce_{name}"] = total / max(1, phase_count)
    net.train()
    return result


def evaluate_blocks(
    net: KingdominoNet,
    blocks: list[tuple[int, list[Any]]],
    args: argparse.Namespace,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for block_id, examples in blocks:
        metrics = evaluate(
            net,
            examples,
            args.device,
            args.score_scale,
            args.lambda_score,
            args.lambda_w,
        )
        rows.append({"block_id": block_id, "n": len(examples), **metrics})
    return rows


def run_trial(
    spec: ArmSpec,
    seed: int,
    lr: float,
    train_buffer: ReplayBuffer,
    holdout: list[Any],
    holdout_blocks: list[tuple[int, list[Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lr_slug = f"{lr:.10g}".replace(".", "p")
    trial_slug = f"{spec.slug}_seed{seed}_lr{lr_slug}"
    progress_path = output_dir / f"{trial_slug}.progress.pt"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    net = KingdominoNet(
        channels=spec.channels,
        blocks=spec.blocks,
        bilinear_dim=args.bilinear_dim,
        score_scale=args.score_scale,
        global_pooling=spec.global_pooling,
    ).to(args.device)
    parameter_count = sum(parameter.numel() for parameter in net.parameters())
    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=args.weight_decay
    )
    # Same seed means every arm/LR sees the same sampled examples and D4 draws.
    rng = np.random.default_rng(seed)
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_step = 0
    step = 0
    elapsed_before = 0.0
    if args.resume and progress_path.exists():
        progress = torch.load(
            progress_path, map_location=args.device, weights_only=False
        )
        net.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        rng.bit_generator.state = progress["numpy_rng_state"]
        best_metrics = progress["best_metrics"]
        best_state = progress["best_state"]
        best_step = int(progress["best_step"])
        step = int(progress["step"])
        elapsed_before = float(progress.get("elapsed_seconds", 0.0))
        print(f"resumed {trial_slug} at step {step}", flush=True)
    started = time.time()
    print(
        f"\n=== {spec.name} seed={seed} lr={lr:.3g}: "
        f"{parameter_count / 1e6:.3f}M params ===",
        flush=True,
    )
    while step < args.max_steps:
        steps_now = min(args.eval_every, args.max_steps - step)
        for _ in range(steps_now):
            batch = train_buffer.sample_batch(
                args.batch, rng, device=args.device, augment_d4=True
            )
            train_step(
                net,
                batch,
                optimizer,
                policy_weight=1.0,
                lambda_score=args.lambda_score,
                lambda_w=args.lambda_w,
                score_scale=args.score_scale,
                grad_clip=args.grad_clip,
            )
        step += steps_now
        metrics = evaluate(
            net,
            holdout,
            args.device,
            args.score_scale,
            args.lambda_score,
            args.lambda_w,
        )
        improved = (
            best_metrics is None
            or metrics["policy_ce"] < best_metrics["policy_ce"] - args.min_delta
        )
        if improved:
            best_metrics = metrics
            best_step = step
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in net.state_dict().items()
            }
        print(
            f"[{spec.name} s{seed} lr={lr:.3g}] step {step}: "
            f"ce={metrics['policy_ce']:.6f} brier={metrics['win_brier']:.6f} "
            f"{'*' if improved else ''} ({(time.time() - started) / 60:.1f}m)",
            flush=True,
        )
        torch.save(
            {
                "schema": "kingdomino-capacity-test-a-progress/v2",
                "arm": spec.name,
                "optimization_seed": seed,
                "lr": lr,
                "step": step,
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "numpy_rng_state": rng.bit_generator.state,
                "best_metrics": best_metrics,
                "best_state": best_state,
                "best_step": best_step,
                "elapsed_seconds": elapsed_before + time.time() - started,
            },
            progress_path,
        )

    assert best_metrics is not None and best_state is not None
    net.load_state_dict(best_state)
    block_metrics = evaluate_blocks(net, holdout_blocks, args)
    checkpoint_path = output_dir / f"{trial_slug}.pt"
    torch.save(
        {
            "model_state": best_state,
            "config": {
                "channels": spec.channels,
                "blocks": spec.blocks,
                "bilinear_dim": args.bilinear_dim,
                "score_scale": args.score_scale,
                "global_pooling": spec.global_pooling,
            },
            "bakeoff": {
                "schema": "kingdomino-capacity-test-a-trial/v2",
                "arm": spec.name,
                "best_step": best_step,
                "presentations": args.max_steps * args.batch,
                "buffer": str(args.buffer),
                "optimization_seed": seed,
                "split_seed": args.split_seed,
                "lr": lr,
            },
        },
        checkpoint_path,
    )
    result: dict[str, Any] = {
        "schema": "kingdomino-capacity-test-a-trial/v2",
        "arm": spec.name,
        "channels": spec.channels,
        "blocks": spec.blocks,
        "global_pooling": spec.global_pooling,
        "params_m": parameter_count / 1e6,
        "optimization_seed": seed,
        "lr": lr,
        "max_steps": args.max_steps,
        "batch": args.batch,
        "example_presentations": args.max_steps * args.batch,
        "best_step": best_step,
        "elapsed_minutes": (elapsed_before + time.time() - started) / 60,
        "checkpoint": str(checkpoint_path),
        "block_metrics": block_metrics,
        **best_metrics,
    }
    (output_dir / f"{trial_slug}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _weighted_mean(rows: list[dict[str, Any]], metric: str) -> float:
    total_n = sum(int(row["n"]) for row in rows)
    return sum(float(row[metric]) * int(row["n"]) for row in rows) / total_n


def paired_block_bootstrap(
    control_trials: list[dict[str, Any]],
    candidate_trials: list[dict[str, Any]],
    metric: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap control-candidate; positive values favor the candidate."""
    if len(control_trials) != len(candidate_trials):
        raise ValueError("paired arms must have the same optimization-seed count")
    control_by_seed = {
        int(trial["optimization_seed"]): trial for trial in control_trials
    }
    candidate_by_seed = {
        int(trial["optimization_seed"]): trial for trial in candidate_trials
    }
    if set(control_by_seed) != set(candidate_by_seed):
        raise ValueError("paired arms must use identical optimization seeds")
    seeds = sorted(control_by_seed)
    block_ids = [
        int(row["block_id"]) for row in control_by_seed[seeds[0]]["block_metrics"]
    ]
    control_values, candidate_values, weights = [], [], []
    for index, block_id in enumerate(block_ids):
        control_rows = [control_by_seed[value]["block_metrics"][index] for value in seeds]
        candidate_rows = [
            candidate_by_seed[value]["block_metrics"][index] for value in seeds
        ]
        if any(int(row["block_id"]) != block_id for row in control_rows + candidate_rows):
            raise ValueError("held-out block ordering differs across trials")
        control_values.append(statistics.fmean(float(row[metric]) for row in control_rows))
        candidate_values.append(
            statistics.fmean(float(row[metric]) for row in candidate_rows)
        )
        block_weights = {int(row["n"]) for row in control_rows + candidate_rows}
        if len(block_weights) != 1:
            raise ValueError("held-out block sizes differ across trials")
        weights.append(block_weights.pop())

    control_array = np.asarray(control_values, dtype=np.float64)
    candidate_array = np.asarray(candidate_values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    difference = control_array - candidate_array
    estimate = float(np.average(difference, weights=weight_array))
    control_mean = float(np.average(control_array, weights=weight_array))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        chosen = rng.integers(0, len(block_ids), size=len(block_ids))
        draws[draw] = np.average(difference[chosen], weights=weight_array[chosen])
    return {
        "paired_blocks": len(block_ids),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "estimate_control_minus_candidate": estimate,
        "relative_improvement": estimate / control_mean,
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
    }


def summarize_trials(
    trials: list[dict[str, Any]], specs: list[ArmSpec], args: argparse.Namespace
) -> dict[str, Any]:
    selected: dict[str, list[dict[str, Any]]] = {}
    arm_summaries: dict[str, Any] = {}
    required_seeds = set(
        getattr(
            args,
            "optimization_seeds",
            {int(trial["optimization_seed"]) for trial in trials},
        )
    )
    for spec in specs:
        arm_trials = [trial for trial in trials if trial["arm"] == spec.name]
        by_lr: dict[float, list[dict[str, Any]]] = {}
        for trial in arm_trials:
            by_lr.setdefault(float(trial["lr"]), []).append(trial)
        complete_by_lr = {
            lr: values
            for lr, values in by_lr.items()
            if {int(trial["optimization_seed"]) for trial in values}
            == required_seeds
        }
        if not complete_by_lr:
            raise ValueError(
                f"{spec.name} has no LR with the complete seed set "
                f"{sorted(required_seeds)}"
            )
        lr_scores = {
            lr: statistics.fmean(trial["policy_ce"] for trial in values)
            for lr, values in complete_by_lr.items()
        }
        best_lr = min(lr_scores, key=lambda lr: (lr_scores[lr], lr))
        selected_trials = sorted(
            complete_by_lr[best_lr],
            key=lambda trial: int(trial["optimization_seed"]),
        )
        selected[spec.name] = selected_trials
        arm_summaries[spec.name] = {
            "selected_lr": best_lr,
            "lr_seed_mean_policy_ce": {str(lr): value for lr, value in lr_scores.items()},
            "incomplete_lr_seed_sets_excluded": {
                str(lr): sorted(int(trial["optimization_seed"]) for trial in values)
                for lr, values in by_lr.items()
                if lr not in complete_by_lr
            },
            "optimization_seeds": [
                int(trial["optimization_seed"]) for trial in selected_trials
            ],
            "seed_mean_policy_ce": statistics.fmean(
                trial["policy_ce"] for trial in selected_trials
            ),
            "seed_mean_win_brier": statistics.fmean(
                trial["win_brier"] for trial in selected_trials
            ),
            "policy_ce_seed_range": max(trial["policy_ce"] for trial in selected_trials)
            - min(trial["policy_ce"] for trial in selected_trials),
            "checkpoints": [trial["checkpoint"] for trial in selected_trials],
        }

    control_name = "80x6"
    if control_name not in selected:
        raise ValueError("Test A requires the 80x6 control arm")
    comparisons: dict[str, Any] = {}
    for spec in specs:
        if spec.name == control_name:
            continue
        control_trials = selected[control_name]
        candidate_trials = selected[spec.name]
        policy = paired_block_bootstrap(
            control_trials,
            candidate_trials,
            "policy_ce",
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        brier = paired_block_bootstrap(
            control_trials,
            candidate_trials,
            "win_brier",
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + 1,
        )
        control_by_seed = {
            int(trial["optimization_seed"]): trial for trial in control_trials
        }
        candidate_by_seed = {
            int(trial["optimization_seed"]): trial for trial in candidate_trials
        }
        paired_gaps = [
            float(control_by_seed[seed]["policy_ce"])
            - float(candidate_by_seed[seed]["policy_ce"])
            for seed in sorted(control_by_seed)
        ]
        paired_gap_spread = max(paired_gaps) - min(paired_gaps)
        policy_pass = (
            float(policy["relative_improvement"]) >= args.policy_relative_floor
            and float(policy["ci95_lower"]) > 0
            and float(policy["estimate_control_minus_candidate"])
            > paired_gap_spread
        )
        # No material Brier regression: the paired interval must not establish
        # a regression (positive is improvement; an entirely negative interval
        # would establish harm).
        brier_noninferior = float(brier["ci95_upper"]) >= 0
        comparisons[spec.name] = {
            "policy_ce": policy,
            "win_brier": brier,
            "paired_policy_gap_by_seed": paired_gaps,
            "paired_policy_gap_seed_spread": paired_gap_spread,
            "policy_gate_pass": policy_pass,
            "brier_noninferior": brier_noninferior,
            "arm_pass": policy_pass and brier_noninferior,
            "large_arm": spec.channels >= 128,
        }

    large_passes = [
        name
        for name, comparison in comparisons.items()
        if comparison["large_arm"] and comparison["arm_pass"]
    ]
    depth_comparison = None
    if "128x8+gp" in selected and "128x10+gp" in selected:
        depth_comparison = paired_block_bootstrap(
            selected["128x8+gp"],
            selected["128x10+gp"],
            "policy_ce",
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + 2,
        )
    return {
        "schema": "kingdomino-capacity-test-a-summary/v2",
        "uncertainty_unit": "paired contiguous 400-example blocks; not game-clustered",
        "policy_relative_floor": args.policy_relative_floor,
        "arm_summaries": arm_summaries,
        "comparisons_vs_80x6": comparisons,
        "large_passing_arms": large_passes,
        "test_a_pass": bool(large_passes),
        "depth_128x8_control_minus_128x10": depth_comparison,
        "depth_selection_requires_p3": True,
    }


def _parse_numbers(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--arms", default="80x6,80x6+gp,128x8+gp,128x10+gp")
    parser.add_argument("--holdout_frac", type=float, default=0.10)
    parser.add_argument("--split_block", type=int, default=400)
    parser.add_argument("--split_seed", type=int, default=20260813)
    parser.add_argument("--seeds", default="20260813,20260814")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_multipliers", default="1,0.5")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_score", type=float, default=0.5)
    parser.add_argument("--lambda_w", type=float, default=0.25)
    parser.add_argument("--score_scale", type=float, default=160.0)
    parser.add_argument("--bilinear_dim", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=40_000)
    parser.add_argument("--eval_every", type=int, default=1_000)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--sample_workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260813)
    parser.add_argument("--policy_relative_floor", type=float, default=0.015)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--trials_only",
        action="store_true",
        help="run requested trials without requiring a complete Test A summary",
    )
    parser.add_argument(
        "--summarize_existing",
        action="store_true",
        help="summarize completed trial JSON files without training",
    )
    args = parser.parse_args()

    specs = [parse_arm(value) for value in args.arms.split(",") if value.strip()]
    seeds = _parse_numbers(args.seeds, int)
    multipliers = _parse_numbers(args.lr_multipliers, float)
    if len(seeds) < 2:
        raise ValueError("Test A requires at least two optimization seeds")
    args.optimization_seeds = seeds
    if args.split_block != 400:
        raise ValueError("the signed-off Test A block size is 400")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_existing:
        trials = []
        for trial_path in sorted(args.out_dir.glob("*.json")):
            payload = json.loads(trial_path.read_text(encoding="utf-8"))
            if (
                isinstance(payload, dict)
                and payload.get("schema") == "kingdomino-capacity-test-a-trial/v2"
            ):
                trials.append(payload)
        summary = summarize_trials(trials, specs, args)
        split_manifest = json.loads(
            (args.out_dir / "split_manifest.json").read_text(encoding="utf-8")
        )
        summary.update({"split_manifest": split_manifest, "trials": trials})
        (args.out_dir / "test_a_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    key: summary[key]
                    for key in (
                        "test_a_pass",
                        "large_passing_arms",
                        "arm_summaries",
                        "comparisons_vs_80x6",
                        "depth_128x8_control_minus_128x10",
                    )
                },
                indent=2,
            ),
            flush=True,
        )
        return

    buffer_hash = _sha256(args.buffer)
    print(f"loading frozen buffer {args.buffer} sha256={buffer_hash}", flush=True)
    examples = load_examples(args.buffer)
    train, holdout, holdout_blocks, holdout_ids = block_split(
        examples, args.holdout_frac, args.split_block, args.split_seed
    )
    split_manifest = {
        "schema": "kingdomino-capacity-test-a-split/v2",
        "buffer": str(args.buffer),
        "buffer_sha256": buffer_hash,
        "examples": len(examples),
        "train_examples": len(train),
        "holdout_examples": len(holdout),
        "total_contiguous_blocks": (len(examples) + args.split_block - 1)
        // args.split_block,
        "holdout_block_ids": holdout_ids,
        "split_block": args.split_block,
        "split_seed": args.split_seed,
        "uncertainty_unit": "paired contiguous blocks; not game-clustered",
    }
    (args.out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{len(examples)} examples -> {len(train)} train / {len(holdout)} holdout "
        f"({len(holdout_blocks)} blocks)",
        flush=True,
    )
    train_buffer = ReplayBuffer(len(train), n_sample_workers=args.sample_workers)
    train_buffer.data = train

    trials: list[dict[str, Any]] = []
    for spec in specs:
        for multiplier in multipliers:
            lr = args.lr * multiplier
            lr_slug = f"{lr:.10g}".replace(".", "p")
            for seed in seeds:
                trial_path = args.out_dir / f"{spec.slug}_seed{seed}_lr{lr_slug}.json"
                if args.resume and trial_path.exists():
                    print(f"resuming completed trial {trial_path.name}", flush=True)
                    trials.append(json.loads(trial_path.read_text(encoding="utf-8")))
                    continue
                result = run_trial(
                    spec,
                    seed,
                    lr,
                    train_buffer,
                    holdout,
                    holdout_blocks,
                    args,
                )
                trials.append(result)
                (args.out_dir / "all_trials.json").write_text(
                    json.dumps(trials, indent=2) + "\n", encoding="utf-8"
                )

    if args.trials_only:
        print(f"completed {len(trials)} requested trials", flush=True)
        return

    summary = summarize_trials(trials, specs, args)
    summary.update({"split_manifest": split_manifest, "trials": trials})
    (args.out_dir / "test_a_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in (
        "test_a_pass", "large_passing_arms", "arm_summaries",
        "comparisons_vs_80x6", "depth_128x8_control_minus_128x10",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
