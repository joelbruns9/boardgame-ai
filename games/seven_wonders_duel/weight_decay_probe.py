#!/usr/bin/env python3
"""Is `--weight-decay 0.5` regularising the run, or pinning it?

cloud4 and cloud5 run `weight_decay=0.5`. `PhaseDConfig` defaults to `1e-4`
(`phase_d.py:287`) and cloud3 ran `1e-2`, so production is **5000x the code's own
default** -- a value introduced as an overfitting fix. The symptoms since then
look more like underfitting than overfitting:

  * train loss barely moves (cloud5: 1.6422 -> 1.5942 over 34 iterations)
  * the train/val gap is small and stable, +0.05 to +0.07
  * `gnorm` climbs steadily (2.18 -> 3.03), which is what you get when the
    gradient has to fight a large decoupled shrinkage term

AdamW decouples decay as `w -= lr*lambda*w` (`train.py:434`), so at lr 5e-5 and
lambda 0.5 every step pulls weights 2.5e-5 toward zero regardless of gradient --
about 0.5% per 190-step iteration, applied forever.

WHAT THIS MEASURES. Identical data, identical split, identical initial weights,
identical batch order; only `weight_decay` differs between arms. Read it as:

  * lower wd reaches lower TRAIN loss but higher VAL loss
        -> 0.5 is doing its job; overfitting is the real constraint
  * lower wd reaches lower train AND val loss
        -> 0.5 is over-regularising and has been costing the run learning
  * all arms land together
        -> weight decay is not the binding constraint; look elsewhere

The parameter L2 norm is reported alongside, because it makes the mechanism
visible rather than inferred: if the 0.5 arm's norm is collapsing while its loss
is flat, the model is being held down rather than converged.

Runs on a laptop against a checkpoint and a slice of a run's buffer. No box time,
no GPU contention with anything else -- but it does want the GPU to itself.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .buffer import from_json_line
from .dataset import examples_from_records
from .train import (
    build_model,
    heads_from_config,
    load_checkpoint,
    stable_game_split,
    train_steps,
)


def load_records(paths: list[Path], limit: int | None) -> list:
    """Read GameRecords from one or more JSONL files, newest file last."""

    records = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(from_json_line(line))
        if limit and len(records) >= limit:
            break
    if limit:
        records = records[:limit]
    return records


def parameter_norm(model) -> float:
    """L2 norm over all trainable parameters, the quantity decay acts on."""

    total = 0.0
    for parameter in model.parameters():
        if parameter.requires_grad:
            total += float(parameter.detach().float().pow(2).sum())
    return math.sqrt(total)


def run_arm(
    *,
    weight_decay: float,
    checkpoint_path: Path,
    train_examples,
    val_examples,
    args,
) -> dict:
    """One decay setting, from the same starting weights as every other arm."""

    # Rebuild under the architecture the checkpoint was trained with. The head
    # count MUST come from the checkpoint: attention parameter shapes do not
    # encode it, so a mismatch loads cleanly and computes something else
    # (`phase_d.py:2752-2759`).
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stored = checkpoint.get("config", {})
    model = build_model(
        "transformer",
        int(stored["d_model"]),
        int(stored["layers"]),
        heads_from_config(stored),
    )
    load_checkpoint(checkpoint_path, model, checkpoint=checkpoint)
    model.to(args.device)
    start_norm = parameter_norm(model)
    started = time.monotonic()
    history, _optimizer_state = train_steps(
        model,
        train_examples,
        val_examples,
        device=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=weight_decay,
        value_bootstrap=args.value_bootstrap,
        validate_every=args.validate_every,
        # Cold optimizer for every arm. Production carries AdamW moments across
        # iterations, but inheriting one arm's moments would confound the next.
        optimizer_state=None,
        restore_best_val=False,
        seed=args.seed,
        precision=args.precision,
        log=lambda *_args, **_kwargs: None,
    )
    elapsed = time.monotonic() - started
    end_norm = parameter_norm(model)
    return {
        "weight_decay": weight_decay,
        "seconds": elapsed,
        "param_norm_start": start_norm,
        "param_norm_end": end_norm,
        "param_norm_ratio": end_norm / start_norm if start_norm else None,
        "history": history,
        "final": last_validated(history),
        "validated_rows": sum(1 for row in history if row.get("val")),
    }


def last_validated(history: list[dict]) -> dict:
    """The last row that actually carries validation metrics.

    `train_steps` only attaches `val` on `validate_every` boundaries, so the
    final row is often train-only.
    """

    for row in reversed(history):
        if row.get("val"):
            return row
    return history[-1] if history else {}


def _metrics(row: dict) -> tuple[float | None, float | None, float | None, float | None]:
    """(train_total, val_total, val_value_acc, val_policy_top1) from one row.

    History rows nest their parts: `row["train"]` and `row["val"]` are dicts of
    loss components and accuracies, not scalars.
    """

    train = row.get("train") or {}
    val = row.get("val") or {}
    return (
        train.get("total"),
        val.get("total"),
        val.get("value_acc"),
        val.get("policy_top1"),
    )


def report(results: list[dict]) -> None:
    print("\n  wd         train      val    value_acc   top1   |w|start   |w|end   ratio")
    for result in results:
        train_loss, val_loss, value_acc, top1 = _metrics(result["final"])

        def fmt(value, width=8, places=4):
            return (
                f"{value:{width}.{places}f}"
                if isinstance(value, (int, float))
                else " " * width
            )

        print(
            f"  {result['weight_decay']:<8.0e}"
            f"{fmt(train_loss)}  {fmt(val_loss)}  {fmt(value_acc, 8, 3)}"
            f"  {fmt(top1, 6, 3)}  {result['param_norm_start']:8.1f}"
            f"  {result['param_norm_end']:8.1f}"
            f"  {result['param_norm_ratio']:.3f}"
        )

    # The verdict depends on how train and val move TOGETHER, so state it.
    scored = []
    for result in results:
        train_loss, val_loss, _acc, _top1 = _metrics(result["final"])
        if isinstance(train_loss, (int, float)) and isinstance(val_loss, (int, float)):
            scored.append((result["weight_decay"], train_loss, val_loss))
    if len(scored) < 2:
        return
    scored.sort(key=lambda item: item[0])
    heaviest, heavy_train, heavy_val = scored[-1]
    lightest, light_train, light_val = scored[0]
    print(
        f"\n  wd {heaviest:.0e} vs {lightest:.0e}:"
        f"  train {heavy_train - light_train:+.4f}   val {heavy_val - light_val:+.4f}"
    )
    if light_train < heavy_train and light_val < heavy_val:
        print("  -> lighter decay wins on BOTH: 0.5 is over-regularising.")
    elif light_train < heavy_train and light_val > heavy_val:
        print("  -> lighter decay fits harder but generalises worse: 0.5 is earning its keep.")
    else:
        print("  -> arms are not separated: weight decay is not the binding constraint.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--buffer",
        type=Path,
        nargs="+",
        required=True,
        help="one or more iteration JSONL files from a run's buffers/ directory",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        default="bf16",
        help="must match the run being diagnosed",
    )
    parser.add_argument("--max-games", type=int, default=8000)
    parser.add_argument(
        "--weight-decays",
        type=float,
        nargs="+",
        default=[0.5, 1e-2, 1e-4],
        help="production, cloud3's value, and the code default",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1500,
        help="chosen so total passes over the slice approximate production's ~6x "
        "cumulative reuse, rather than one iteration's 190",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=63)
    parser.add_argument("--value-bootstrap", type=float, default=0.5)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--val-split-salt", default="swd-v1")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    records = load_records(list(args.buffer), args.max_games)
    print(f"loaded {len(records)} records from {len(args.buffer)} file(s)")

    derived = time.monotonic()
    examples = examples_from_records(records, record_fast_moves=False)
    print(
        f"derived {len(examples)} examples in {time.monotonic() - derived:.0f}s"
    )

    # Split ONCE, so every arm sees byte-identical train and val sets.
    train_examples, val_examples = stable_game_split(
        examples, args.val_fraction, args.val_split_salt
    )
    passes = args.steps * args.batch_size / max(1, len(train_examples))
    print(
        f"train {len(train_examples)}  val {len(val_examples)}  "
        f"steps {args.steps} x batch {args.batch_size} = {passes:.1f} passes"
    )

    results = []
    for weight_decay in args.weight_decays:
        print(f"\n--- weight_decay = {weight_decay:g} ---")
        result = run_arm(
            weight_decay=weight_decay,
            checkpoint_path=args.checkpoint,
            train_examples=train_examples,
            val_examples=val_examples,
            args=args,
        )
        results.append(result)
        print(
            f"    {result['seconds']:.0f}s   "
            f"|w| {result['param_norm_start']:.1f} -> {result['param_norm_end']:.1f}"
        )

    report(results)

    if args.out:
        args.out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
