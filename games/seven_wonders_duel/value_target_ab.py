"""Offline A/B of the value target's outcome share: flat vs distance-scaled.

One arm per call. Warm-starts from a checkpoint that never trained on the
window (run07 `candidate_0080` for window 91-100), re-derives the window under
the current code with every G0-sealed game withheld (`g3_offline_ab`), trains
a fixed number of steps with run07's loss settings except the value target's
outcome share, and saves the checkpoint plus a held-out report.

Arms (same seed, window and steps):

* ``--value-bootstrap 0.5`` -- the flat run07/run08 target;
* ``--value-bootstrap 0.75`` -- the flat, lower-outcome alternative;
* ``--value-bootstrap 0.5 --outcome-share-decay 0.97 --outcome-share-floor
  0.2`` -- the realised result's share is 0.5 at the last move, decays per move
  toward the start, floored at 0.2 (before the 25% short-term mix).

The held-out report is the judge here, by game distance to the end:

* realised-outcome log loss and Brier of the flat head's win probability on
  the validation GAMES. A proper scoring rule, so on unseen games it rewards
  the true probability and does not favour the outcome-heavy arm;
* the train-minus-validation gap of the same, which is the memorisation the
  schedule is meant to remove;
* absolute error against exact endgame proofs on validation rows.

Then score each checkpoint on the G0 suite (`tactical_suite evaluate --split
sealed`) and compare pairwise (`tactical_suite compare`).

What this cannot show: whether results propagate back across generations.
That is an online effect (better late values improve later searches' roots).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch

from .dataset import RETAIN_PROOFS_PER_GAME_DEFAULT, collate
from .g3_offline_ab import RUN07, _iterations, derive_window
from .tactical_suite import SEALED_FRACTION
from .train import (
    _evaluation_autocast,
    control_table_for,
    load_checkpoint,
    make_checkpoint,
    model_from_config,
    stable_game_split,
    train_steps,
)

def _bucket(plies) -> str:
    """Moves left to the end: last age's tail, the rest of it, mid, opening."""

    if plies is None:
        return "unknown"
    if plies < 10:
        return "0-9"
    if plies < 30:
        return "10-29"
    if plies < 50:
        return "30-49"
    return "50+"


@torch.no_grad()
def held_out_report(model, examples, device: str, batch_size: int = 512) -> dict:
    """Outcome log loss / Brier by distance bucket, and proof error."""

    model.eval()
    table = control_table_for(model)
    sums: dict[str, dict[str, float]] = {}
    proof = {"rows": 0, "abs_error": 0.0}
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = collate(
            chunk, device,
            contextual_actions=bool(getattr(model, "action_residual", False)),
            control_table=table,
        )
        with _evaluation_autocast(device, RUN07["precision"]):
            outputs = model(batch)
        probs = torch.softmax(outputs["value"].float(), dim=-1)
        p = (probs[:, 0] + 0.5 * probs[:, 1]).clamp(1e-6, 1 - 1e-6)
        cls = batch["value_class"].long()
        y = torch.where(cls == 0, 1.0, torch.where(cls == 1, 0.5, 0.0)).to(p)
        logloss = -(y * p.log() + (1 - y) * (1 - p).log())
        brier = (p - y) ** 2
        for row, example in enumerate(chunk):
            if getattr(example, "outcome_free", False):
                continue
            for key in ("all", _bucket(getattr(example, "plies_to_end", None))):
                tally = sums.setdefault(key, {"rows": 0, "logloss": 0.0, "brier": 0.0})
                tally["rows"] += 1
                tally["logloss"] += float(logloss[row])
                tally["brier"] += float(brier[row])
            if example.solver_value is not None and example.solver_exact:
                proof["rows"] += 1
                proof["abs_error"] += abs(float(2 * p[row] - 1) - float(example.solver_value))
    report = {
        key: {
            "rows": tally["rows"],
            "logloss": tally["logloss"] / tally["rows"],
            "brier": tally["brier"] / tally["rows"],
        }
        for key, tally in sums.items()
    }
    report["proof"] = {
        "rows": proof["rows"],
        "abs_error": proof["abs_error"] / max(proof["rows"], 1),
    }
    return report


def _sample_by_game(examples, games: int, seed: int) -> list:
    keys = sorted({(e.iteration, e.game_key) for e in examples})
    keep = set(random.Random(seed).sample(keys, min(games, len(keys))))
    return [e for e in examples if (e.iteration, e.game_key) in keep]


def run(args) -> dict:
    started = time.time()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_config(checkpoint["config"])
    # An encoder-7 start (run07's candidates) loads additively, as in
    # `pretrain.base_model`: the grown G10a columns start at exact zero.
    load_checkpoint(args.checkpoint, model, checkpoint=checkpoint, migrate=True)
    migration = checkpoint.get("migration")
    if migration is not None and migration["zeroed"]:
        raise ValueError(f"{args.checkpoint}: migration would zero {migration['zeroed'][:5]}")
    model.to(args.device)

    examples, reserved = derive_window(
        args.buffers_dir, _iterations(args.iterations), args.retain_proofs_per_game,
        tactic_labels=args.tactic_labels,
    )
    train_examples, val_examples = stable_game_split(
        examples, args.val_fraction, RUN07["val_split_salt"]
    )
    # The memorisation gap: the same metric on a same-sized sample of TRAINING
    # games, which the arm has fitted.
    val_games = len({(e.iteration, e.game_key) for e in val_examples})
    train_probe = _sample_by_game(train_examples, val_games, args.seed)
    before = {
        "val": held_out_report(model, val_examples, args.device),
        "train": held_out_report(model, train_probe, args.device),
    }
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
        value_bootstrap=args.value_bootstrap,
        short_term_value_weight=RUN07["short_term_value_weight"],
        action_policy_weight=RUN07["action_policy_weight"],
        hier_value_weight=RUN07["hier_value_weight"],
        hier_value_replaces_joint7=RUN07["hier_value_replaces_joint7"],
        outlook_bootstrap=RUN07["outlook_bootstrap"],
        validate_every=args.validate_every,
        seed=args.seed,
        precision=RUN07["precision"],
        outcome_share_decay=args.outcome_share_decay,
        outcome_share_floor=args.outcome_share_floor,
    )
    after = {
        "val": held_out_report(model, val_examples, args.device),
        "train": held_out_report(model, train_probe, args.device),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(make_checkpoint(model, dict(checkpoint["config"])), args.out)
    games = len({(e.iteration, e.game_key) for e in train_examples})
    summary = {
        "init": str(args.checkpoint),
        "iterations": args.iterations,
        "value_bootstrap": args.value_bootstrap,
        "outcome_share_decay": args.outcome_share_decay,
        "outcome_share_floor": args.outcome_share_floor,
        "rows": {"train": len(train_examples), "val": len(val_examples)},
        "games": {"train": games, "val": val_games},
        "samples_per_train_game": args.steps * RUN07["batch_size"] / max(games, 1),
        "g0_sealed_games_withheld": reserved,
        "g0_sealed_fraction": SEALED_FRACTION,
        "steps": args.steps,
        "seed": args.seed,
        "before": before,
        "after": after,
        "history": history,
        "minutes": round((time.time() - started) / 60, 1),
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--buffers-dir", type=Path, required=True)
    parser.add_argument("--iterations", default="91-100")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--retain-proofs-per-game", type=int,
                        default=RETAIN_PROOFS_PER_GAME_DEFAULT)
    parser.add_argument("--tactic-labels", action=argparse.BooleanOptionalAction,
                        default=True)
    # Twice run07's 5%: the judge is a per-game metric, so more held-out games.
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--value-bootstrap", type=float, default=0.5)
    parser.add_argument("--outcome-share-decay", type=float, default=0.0)
    parser.add_argument("--outcome-share-floor", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    summary = run(build_parser().parse_args(argv))
    summary.pop("history", None)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
