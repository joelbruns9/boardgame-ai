"""G3 offline A/B: one arm of uniform vs priority sampling on a fixed window.

Warm-starts from a checkpoint (run07's `candidate_0060` by default), re-derives
a window of run07 buffers under the current G1/G2 code, trains ONE arm for a
fixed number of steps with run07's loss settings, and saves the result. Run it
twice (``--arm uniform`` and ``--arm priority``, same seed and window), then
score both with the G0 suite::

    python -m games.seven_wonders_duel.tactical_suite evaluate \\
        --cases <cases.jsonl> --checkpoint <out>.pt --sims 0,64,800 ...

Success (plan, G3): the decisive classes improve -- own_win / must_block /
reveal_trap / forced_loss / predecessor -- while `ordinary` calibration does
not get worse. Validation loss is printed but is NOT the judge: uniform
validation rows are exactly what priority sampling shows less often.

Memory: ~13 KB per row, ~22k rows per run07 iteration -- a 10-iteration window
is ~3 GB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .buffer import read_records
from .dataset import RETAIN_PROOFS_PER_GAME_DEFAULT, derive_records_rust
from .train import (
    VALUE_TARGET_CONTRACT_DEFAULT,
    VALUE_TARGET_CONTRACTS,
    load_checkpoint,
    make_checkpoint,
    model_from_config,
    stable_game_split,
    train_steps,
)

#: run07's training settings (run_manifest.json), so the two arms differ only
#: in how rows are drawn.
RUN07 = {
    "lr": 5e-5,
    "weight_decay": 0.5,
    "batch_size": 512,
    "warmup_steps": 63,
    "aux_weight": 0.2,
    "value_weight": 1.0,
    "action_policy_weight": 0.5,
    "value_bootstrap": 0.5,
    "short_term_value_weight": 0.25,
    "hier_value_weight": 0.2,
    "hier_value_replaces_joint7": True,
    "outlook_bootstrap": 0.5,
    "precision": "bf16",
    "val_fraction": 0.05,
    "val_split_salt": "swd-v1",
}


def _iterations(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def derive_window(buffers_dir: Path, iterations: list[int], retain: int, log=print) -> list:
    examples = []
    for iteration in iterations:
        records = read_records(buffers_dir / f"iter_{iteration:04d}.jsonl")
        for rows, _stats in derive_records_rust(records, retain_proofs_per_game=retain):
            examples.extend(rows)
        log(f"derived iteration {iteration}: {len(examples)} rows so far")
    return examples


def run(args) -> dict:
    started = time.time()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_config(checkpoint["config"])
    load_checkpoint(args.checkpoint, model, checkpoint=checkpoint)
    model.to(args.device)

    examples = derive_window(args.buffers_dir, _iterations(args.iterations), args.retain_proofs_per_game)
    train_examples, val_examples = stable_game_split(
        examples, RUN07["val_fraction"], RUN07["val_split_salt"]
    )
    weights, report = None, None
    if args.arm == "priority":
        from .priority_sampling import sample_weights

        weights, report = sample_weights(
            model, train_examples, args.device,
            uniform_share=args.uniform_share, cap=args.cap,
        )
        print(f"[g3] {report}", flush=True)
    history, _optimizer_state = train_steps(
        model,
        train_examples,
        val_examples,
        device=args.device,
        steps=args.steps,
        batch_size=RUN07["batch_size"],
        lr=RUN07["lr"],
        warmup_steps=RUN07["warmup_steps"],
        weight_decay=RUN07["weight_decay"],
        aux_weight=RUN07["aux_weight"],
        value_weight=RUN07["value_weight"],
        value_bootstrap=RUN07["value_bootstrap"],
        short_term_value_weight=RUN07["short_term_value_weight"],
        action_policy_weight=RUN07["action_policy_weight"],
        hier_value_weight=RUN07["hier_value_weight"],
        hier_value_replaces_joint7=RUN07["hier_value_replaces_joint7"],
        outlook_bootstrap=RUN07["outlook_bootstrap"],
        value_target_contract=args.value_target_contract,
        validate_every=args.validate_every,
        seed=args.seed,
        precision=RUN07["precision"],
        sample_weights=weights,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(make_checkpoint(model, dict(checkpoint["config"])), args.out)
    summary = {
        "arm": args.arm,
        "init": str(args.checkpoint),
        "iterations": args.iterations,
        "rows": {"train": len(train_examples), "val": len(val_examples)},
        "steps": args.steps,
        "seed": args.seed,
        "value_target_contract": args.value_target_contract,
        "priority_report": report,
        "final": history[-1] if history else None,
        "minutes": round((time.time() - started) / 60, 1),
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", choices=("uniform", "priority"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--buffers-dir", type=Path, required=True)
    parser.add_argument("--iterations", default="91-100", help="window, e.g. 91-100")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--retain-proofs-per-game", type=int,
                        default=RETAIN_PROOFS_PER_GAME_DEFAULT)
    parser.add_argument("--value-target-contract", choices=VALUE_TARGET_CONTRACTS,
                        default=VALUE_TARGET_CONTRACT_DEFAULT)
    parser.add_argument("--uniform-share", type=float, default=0.3)
    parser.add_argument("--cap", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
