#!/usr/bin/env python3
"""Why does a run stop learning? Retrain its own data and change one thing.

cloud2 and cloud3 both stalled the same way: the train/val gap climbs steadily,
`value_acc` peaks and declines about twelve iterations before `policy_top1`
does, gates start rejecting, and a full weight reset repairs it completely --
for about twenty iterations, after which it returns. Two rentals established the
pattern and neither established the cause.

The hypothesis tested here is that the OUTCOME-SIDE heads are what break. Five
of the six heads -- value, joint7, margin, military, science -- fit a per-GAME
label: one outcome shared across all ~16 rows of that game. An iteration
producing ~16,500 policy labels produces only ~1,000 independent outcome labels,
and at the defaults those five carry 1.0 + 0.2*4 = 1.8 of the loss weight against
the policy head's 1.0. That is the side of the objective best placed to
memorise, and on a shared trunk memorising it drags the representation the
policy head depends on. It also predicts the observed ordering: value degrades
first, policy follows.

The competing hypothesis is that nothing about the objective is wrong and the
model is out of capacity -- which implies a different and far more expensive
next run.

This separates them on a laptop. It holds a fixed pool of the run's own games
and trains repeatedly on it, varying one thing per arm:

    baseline   the run's configuration, unchanged
    decay      --weight-decay raised to a value that actually acts
    value      --value-weight lowered

If the outcome-side story holds, `decay` and `value` flatten the gap curve while
`baseline` reproduces the climb. If all three lie on top of each other, the
objective is not the problem and the answer is capacity.

WHY A FIXED POOL. The buffer files are a sample of the run, not a contiguous
window, so replaying them as successive windows would not reproduce what the run
held. A fixed pool asks the narrower question this test actually needs answered
-- under repeated exposure to real data, which arm resists memorising it -- and
asks it under deliberately accelerated pressure: with ~5,000 games the pool is
about a sixth of the run's 30,000-game window, so each round makes roughly six
times the passes the run made. Expect the climb to arrive sooner and steeper
than in the logs. Arms are compared to each other, never to the run's numbers.

WHAT IT CANNOT SHOW: the data never responds to the learner. This reproduces
memorisation but not the self-play feedback loop. A null result rules out
memorisation; it does not rule out something inside the loop.

WHICH GAP. `total` is the weighted objective, so it differs between arms by
construction and cannot be compared across them. The per-head components in
`parts` are unweighted, so this reports gaps per head: `gap_value` and
`gap_policy` mean the same thing in every arm.

ON --weight-decay: AdamW's decay step is `lr * lambda * w`. At the run's lr of
5e-5, lambda=1e-2 is 5e-7 per step -- about 0.6% cumulative shrink over an entire
run, against gradient updates ~100x larger. cloud3 ran believing it had raised
regularisation 100x; it had raised it to approximately zero. Values near 0.5 are
where decay begins to constrain anything at this learning rate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .buffer import read_records
from .dataset import examples_from_record
from .train import (
    build_model,
    heads_from_config,
    load_checkpoint,
    stable_game_split,
    train_steps,
)


def load_pool(buffer_dir: Path) -> list:
    paths = sorted(buffer_dir.glob("iter_[0-9][0-9][0-9][0-9].jsonl"))
    if not paths:
        raise FileNotFoundError(f"no iter_NNNN.jsonl files in {buffer_dir}")
    records: list = []
    for path in paths:
        loaded = read_records(path)
        records.extend(loaded)
        print(f"  {path.name}: {len(loaded)} games")
    return records


def run_arm(
    *,
    name: str,
    checkpoint: Path,
    train_examples: list,
    val_examples: list,
    rounds: int,
    device: str,
    steps: int,
    batch_size: int,
    lr: float,
    warmup_steps: int,
    weight_decay: float,
    value_weight: float,
    aux_weight: float,
    precision: str,
    grad_clip: float,
    optimizer_name: str,
    seed: int,
    log=print,
) -> list[dict]:
    import torch

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = raw.get("config") or {}
    model = build_model(
        config.get("model", "transformer"),
        config.get("d_model", 128),
        config.get("layers", 4),
        heads_from_config(config),
    )
    load_checkpoint(checkpoint, model)

    # Every arm starts from the same weights and sees the same split in the same
    # order. The run's optimizer state is deliberately NOT restored: its moments
    # accumulated under the baseline decay and would contaminate the others from
    # the first step.
    optimizer_state = None
    history: list[dict] = []

    for index in range(rounds):
        started = time.monotonic()
        rows, optimizer_state = train_steps(
            model,
            train_examples,
            val_examples,
            device=device,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            # Warm up once, at the start, exactly as the run does on resume.
            warmup_steps=warmup_steps if index == 0 else 0,
            weight_decay=weight_decay,
            aux_weight=aux_weight,
            value_weight=value_weight,
            grad_clip=grad_clip,
            optimizer_name=optimizer_name,
            optimizer_state=optimizer_state,
            seed=seed + index,
            precision=precision,
            log=lambda *a, **k: None,
        )
        final = rows[-1] if rows else {}
        train_parts = final.get("train") or {}
        val_parts = final.get("val") or {}
        entry = {
            "arm": name,
            "round": index,
            "train_policy": train_parts.get("policy"),
            "train_value": train_parts.get("value"),
            "val_policy": val_parts.get("policy"),
            "val_value": val_parts.get("value"),
            "value_acc": val_parts.get("value_acc"),
            "policy_top1": val_parts.get("policy_top1"),
            "joint7_acc": val_parts.get("joint7_acc"),
            "grad_norm": final.get("grad_norm"),
            "seconds": round(time.monotonic() - started, 1),
        }
        for head in ("policy", "value"):
            train_side = entry[f"train_{head}"]
            val_side = entry[f"val_{head}"]
            entry[f"gap_{head}"] = (
                val_side - train_side
                if train_side is not None and val_side is not None
                else None
            )
        history.append(entry)
        log(
            f"  {name:>8} round {index:>3}  "
            f"gap_value {entry['gap_value']:+.4f}  "
            f"gap_policy {entry['gap_policy']:+.4f}  "
            f"value_acc {entry['value_acc']:.3f}  "
            f"top1 {entry['policy_top1']:.3f}  "
            f"gnorm {entry['grad_norm']:.2f}  [{entry['seconds']:.0f}s]"
        )
    return history


def _slope(history: list[dict], key: str) -> float | None:
    """Least-squares slope per round -- the number the whole test is about."""

    points = [(h["round"], h[key]) for h in history if h.get(key) is not None]
    if len(points) < 3:
        return None
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--train-steps", type=int, default=190)
    parser.add_argument("--train-warmup-steps", type=int, default=63)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--baseline-weight-decay",
        type=float,
        default=1e-2,
        help="cloud3's setting, inert at lr 5e-5 (see module docstring)",
    )
    parser.add_argument(
        "--decay-arm",
        type=float,
        default=2.0,
        help="weight decay for the decay arm. Choose it to match CUMULATIVE "
        "shrink (lr * lambda * steps), not the value you would run in "
        "production: this test is ~2,850 steps against a real run's ~11,400, so "
        "lambda=2.0 here reproduces what lambda=0.5 does over a full run. A "
        "nominal 0.5 in this budget is ~7% shrink and could not bend anything, "
        "which would read as a null result rather than a weak arm.",
    )
    parser.add_argument("--value-arm", type=float, default=0.4)
    parser.add_argument(
        "--clip-arm",
        type=float,
        default=1.0,
        help="global gradient-norm clip for the clip arm. 1.0 is Kingdomino's "
        "setting; 7WD does not clip at all and its gnorm runs 2-4.",
    )
    parser.add_argument(
        "--adam-weight-decay",
        type=float,
        default=1e-4,
        help="weight decay for the adam arm. 1e-4 is Kingdomino's, and it is NOT "
        "comparable to an AdamW lambda: Adam's L2 enters the gradient and passes "
        "through the adaptive denominator rather than scaling with lr.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["baseline", "decay", "value"],
        choices=["baseline", "decay", "value", "clip", "adam"],
    )
    parser.add_argument("--out", type=Path, default=Path("ablation.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(f"loading pool from {args.buffer_dir}")
    records = load_pool(args.buffer_dir)
    # Derivation is minutes, and every arm needs the identical example set, so
    # it is cached against the pool's size and newest file rather than repeated
    # on each invocation.
    newest = max(path.stat().st_mtime_ns for path in args.buffer_dir.glob("*.jsonl"))
    cache = args.buffer_dir / f".derived_{len(records)}_{newest}.pkl"
    derive_started = time.monotonic()
    if cache.exists():
        import pickle

        examples = pickle.loads(cache.read_bytes())
        print(f"  reused {cache.name} [{time.monotonic() - derive_started:.0f}s]")
    else:
        examples = []
        for record in records:
            examples.extend(examples_from_record(record))
        import pickle

        cache.write_bytes(pickle.dumps(examples, protocol=5))
    print(
        f"  {len(records)} games -> {len(examples)} examples "
        f"[{time.monotonic() - derive_started:.0f}s]"
    )

    # One split, shared by every arm, never sharing a game across the boundary.
    train_examples, val_examples = stable_game_split(
        examples, args.val_fraction, salt="ablation"
    )
    passes = args.train_steps * args.train_batch_size / max(1, len(train_examples))
    print(
        f"  train {len(train_examples)} / val {len(val_examples)} "
        f"-- {passes:.2f} passes per round\n"
    )

    # (weight_decay, value_weight, grad_clip, optimizer)
    #
    # `clip` and `adam` restore what the Kingdomino loop does and 7WD does not:
    # KD clips global gradient norm at 1.0 every step and uses Adam's coupled
    # L2. 7WD clips nothing -- while its gnorm ran 1.44 -> 3.40 -- and uses
    # AdamW's decoupled decay at a learning rate 20x lower, where `lr*lambda` is
    # 5e-7 per step. Both are regressions from a configuration with two working
    # projects behind it, and neither has been tested here.
    settings = {
        "baseline": (args.baseline_weight_decay, 1.0, 0.0, "adamw"),
        "decay": (args.decay_arm, 1.0, 0.0, "adamw"),
        "value": (args.baseline_weight_decay, args.value_arm, 0.0, "adamw"),
        "clip": (args.baseline_weight_decay, 1.0, args.clip_arm, "adamw"),
        "adam": (args.adam_weight_decay, 1.0, 0.0, "adam"),
    }

    results: dict[str, list[dict]] = {}
    for name in args.arms:
        weight_decay, value_weight, grad_clip, optimizer_name = settings[name]
        shrink = 1.0 - (1.0 - args.learning_rate * weight_decay) ** (
            args.rounds * args.train_steps
        )
        print(
            f"arm {name}: optimizer={optimizer_name} weight_decay={weight_decay} "
            f"value_weight={value_weight} grad_clip={grad_clip or 'off'}"
            + (f" (cumulative shrink {shrink:.1%})" if optimizer_name == "adamw" else "")
        )
        results[name] = run_arm(
            name=name,
            checkpoint=args.checkpoint,
            train_examples=train_examples,
            val_examples=val_examples,
            rounds=args.rounds,
            device=args.device,
            steps=args.train_steps,
            batch_size=args.train_batch_size,
            lr=args.learning_rate,
            warmup_steps=args.train_warmup_steps,
            weight_decay=weight_decay,
            value_weight=value_weight,
            aux_weight=args.aux_weight,
            precision=args.precision,
            grad_clip=grad_clip,
            optimizer_name=optimizer_name,
            seed=args.seed,
        )
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print()

    print(
        f"{'arm':>10} {'gap_value/rd':>13} {'gap_policy/rd':>14} "
        f"{'value_acc/rd':>13} {'final top1':>11}"
    )
    for name, history in results.items():
        if not history:
            continue
        print(
            f"{name:>10} {_slope(history, 'gap_value'):>+13.5f} "
            f"{_slope(history, 'gap_policy'):>+14.5f} "
            f"{_slope(history, 'value_acc'):>+13.5f} "
            f"{history[-1]['policy_top1']:>11.3f}"
        )
    print(
        "\nSlopes are per round, on the UNWEIGHTED per-head losses, so they mean\n"
        "the same thing in every arm. Compare arms to each other: the run's own\n"
        "numbers are not a reference, because this pool is deliberately smaller\n"
        "and therefore harder to resist memorising.\n\n"
        "  baseline climbs, decay and/or value flatten -> the objective is the\n"
        "      mechanism, and the flattening arm names which side of it\n"
        "  all three climb together                    -> not the objective;\n"
        "      the next question is capacity"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
