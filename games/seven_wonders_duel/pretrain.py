"""Final-run preparation, steps 3-4 (`MODEL_GROWTH_PLAN.md`): windowed pretrain
on the corrected run07 buffers, and the G14 initialisation check.

Walks a range of run07 iterations in RAM-sized windows (default 10 iterations,
~3 GB), deriving each window under the current corrections -- G1 retention,
G2 targets, G2b tactic labels (on), and an optional G8.2 reanalysis overlay --
and trains ``--steps-per-window`` steps on it with run07's loss settings. The
model AND optimizer carry from window to window, so the run behaves like one
continuous pass over the data with bounded memory. G0's sealed games are
withheld from every window (`g3_offline_ab.derive_window`).

``--init`` is the G14 axis -- run the same command three times, changing only
it, and score the results on sealed G0:

* ``checkpoint`` -- the base (decision 13: candidate_0100) + fresh optimizer;
* ``random``     -- a fresh network of the base's architecture;
* ``reset-value`` -- the base with its value heads (value, joint7, margin,
  military, science, W4) re-initialised.

W5's served weight (alpha) is not trained; Phase D FITS it on held-out rows
after each training step (`action_alpha`). Pretrain does the same after every
window, for every arm -- without it a random arm's alpha stays at its built
zero for the whole run, so the G14 comparison would also be W5-on versus
W5-off (review of 8014a6c, #1). ``--alpha-step`` defaults to a jump straight
to the fitted value: windows are few, and a capped step would leave the random
arm short of where its heads put the optimum.

``--grow-layers N`` is the capacity probe (plan W8): the base's transformer
gets ``N - layers`` new layers appended on top, each an exact no-op at the
start -- pre-norm residual blocks whose two output projections (attention
``out_proj`` and feed-forward ``linear2``, weights and biases) are zero -- so
the grown net computes exactly what the base did, and training decides what
the new layers add. Only the output projections are zeroed: a block zeroed
throughout gets no gradient and would stay a no-op forever.

Each window's checkpoint is saved, so a stopped run restarts from the last
completed window with ``--resume``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from .action_alpha import format_alpha_fit, refit_alpha
from .g3_offline_ab import RUN07, derive_window
from .targeted_reanalysis import load_overlay
from .train import (
    load_checkpoint,
    make_checkpoint,
    model_from_config,
    stable_game_split,
    train_steps,
)

INITS = ("checkpoint", "random", "reset-value")
#: Value-side modules `--init reset-value` re-initialises.
VALUE_MODULES = ("heads.value", "heads.joint7", "heads.margin", "heads.military",
                 "heads.science", "hier_value")


def _windows(first: int, last: int, size: int) -> list[list[int]]:
    return [list(range(lo, min(lo + size, last + 1))) for lo in range(first, last + 1, size)]


#: The parameters of a new layer that are zeroed so the layer starts as a no-op.
GROWN_ZERO_SUFFIXES = (
    "self_attn.out_proj.weight", "self_attn.out_proj.bias", "linear2.weight", "linear2.bias",
)


def grow_layers(model_config: dict, state: dict, layers: int, seed: int):
    """A model with `layers` transformer layers whose first ones carry `state`
    and whose new top layers are exact no-ops (see the module docstring)."""

    old = int(model_config["layers"])
    if layers <= old:
        raise ValueError(f"--grow-layers {layers} must exceed the base's {old} layers")
    config = dict(model_config, layers=layers)
    torch.manual_seed(seed)
    model = model_from_config(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    new_prefixes = tuple(f"encoder.layers.{i}." for i in range(old, layers))
    if unexpected or any(not key.startswith(new_prefixes) for key in missing):
        raise RuntimeError(
            f"growing {old} -> {layers} layers: unexpected {unexpected[:3]}, "
            f"missing outside the new layers {[k for k in missing if not k.startswith(new_prefixes)][:3]}"
        )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith(new_prefixes) and name.endswith(GROWN_ZERO_SUFFIXES):
                parameter.zero_()
    return model, config


def base_model(base: Path, init: str, seed: int, grow_to: int | None = None):
    checkpoint = torch.load(base, map_location="cpu", weights_only=False)
    torch.manual_seed(seed)
    model = model_from_config(checkpoint["config"])
    if init == "random":
        if grow_to is not None:
            raise ValueError("--grow-layers grows a trained base; use --init checkpoint")
        return model, checkpoint["config"]
    # An encoder-schema change since the base was trained (G10a appended 22
    # tableau channels) is migrated ADDITIVELY: the grown input columns load as
    # exact zeros, so the net computes what it did until training uses them.
    # Anything else the migration would have to zero means a partly fresh net,
    # which is refused here rather than trained on.
    load_checkpoint(base, model, checkpoint=checkpoint, migrate=True)
    migration = checkpoint.get("migration")
    if migration is not None:
        if migration["zeroed"]:
            raise ValueError(
                f"{base}: migrating would zero {migration['zeroed'][:5]}; not an "
                "additive warm start"
            )
        print(f"[pretrain] migrated {base}: grown {migration['grown']}", flush=True)
    config = checkpoint["config"]
    if grow_to is not None:
        if init != "checkpoint":
            raise ValueError("--grow-layers grows a trained base; use --init checkpoint")
        return grow_layers(config, model.state_dict(), grow_to, seed)
    if init == "reset-value":
        reset = 0
        for name, module in model.named_modules():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in VALUE_MODULES):
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
                    reset += 1
        if reset == 0:
            raise RuntimeError("--init reset-value found no value modules to reset")
    return model, config


def run(args) -> dict:
    if args.presentations_per_row is not None and args.presentations_per_row <= 0:
        args.presentations_per_row = None
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "progress.json"
    progress = json.loads(state_path.read_text()) if args.resume and state_path.exists() else {
        "completed_windows": [], "history": [],
    }
    model, config = base_model(args.base, args.init, args.seed, args.grow_layers)
    optimizer_state = None
    if progress["completed_windows"]:
        last = out_dir / f"window_{progress['completed_windows'][-1]}.pt"
        saved = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer_state = saved.get("optimizer_state")
    model.to(args.device)
    overlay = load_overlay(args.reanalysis_overlay) if args.reanalysis_overlay else None
    first, _, last = args.iterations.partition("-")
    windows = _windows(int(first), int(last or first), args.window)
    for index, window in enumerate(windows):
        tag = f"{window[0]:04d}_{window[-1]:04d}"
        if tag in progress["completed_windows"]:
            continue
        started = time.time()
        examples, reserved = derive_window(
            args.buffers_dir, window, args.retain_proofs_per_game,
            tactic_labels=args.tactic_labels, reanalysis_overlay=overlay,
        )
        train_examples, val_examples = stable_game_split(
            examples, RUN07["val_fraction"], RUN07["val_split_salt"]
        )
        del examples
        # Exposure control: a fixed number of presentations per training row
        # (default 2) rather than a fixed step count, so a wider window does
        # not mean more passes over each row.
        steps = (
            max(1, math.ceil(args.presentations_per_row * len(train_examples)
                             / RUN07["batch_size"]))
            if args.presentations_per_row is not None
            else args.steps_per_window
        )
        history, optimizer_state = train_steps(
            model,
            train_examples,
            val_examples,
            device=args.device,
            steps=steps,
            batch_size=RUN07["batch_size"],
            lr=RUN07["lr"],
            warmup_steps=RUN07["warmup_steps"] if index == 0 and optimizer_state is None else 0,
            weight_decay=RUN07["weight_decay"],
            aux_weight=RUN07["aux_weight"],
            value_weight=RUN07["value_weight"],
            value_bootstrap=RUN07["value_bootstrap"],
            short_term_value_weight=RUN07["short_term_value_weight"],
            action_policy_weight=RUN07["action_policy_weight"],
            hier_value_weight=RUN07["hier_value_weight"],
            hier_value_replaces_joint7=RUN07["hier_value_replaces_joint7"],
            outlook_bootstrap=RUN07["outlook_bootstrap"],
            validate_every=args.validate_every,
            optimizer_state=optimizer_state,
            seed=args.seed + index,
            precision=args.precision,
        )
        alpha_fit = None
        scorer = getattr(model, "action_scorer", None)
        if config.get("fit_action_alpha") and scorer is not None:
            # Before the window checkpoint, so a resumed run starts from it.
            alpha_fit = refit_alpha(
                model,
                val_examples,
                args.device,
                alpha_max=scorer.gate_max,
                step=args.alpha_step if args.alpha_step else scorer.gate_max,
                batch_size=RUN07["batch_size"],
                precision=args.precision,
                seed=args.seed + index,
            ).as_dict()
            print(format_alpha_fit(alpha_fit), flush=True)
        payload = make_checkpoint(model, dict(config))
        payload["optimizer_state"] = optimizer_state
        torch.save(payload, out_dir / f"window_{tag}.pt")
        final = history[-1] if history else {}
        progress["completed_windows"].append(tag)
        progress["history"].append({
            "window": tag,
            "rows": {"train": len(train_examples), "val": len(val_examples)},
            "steps": steps,
            "presentations_per_row": steps * RUN07["batch_size"] / max(1, len(train_examples)),
            "g0_sealed_games_withheld": reserved,
            "reanalysed_rows": sum(1 for e in train_examples if getattr(e, "reanalysed", False)),
            "val_total": (final.get("val") or {}).get("total"),
            "alpha_fit": alpha_fit,
            "minutes": round((time.time() - started) / 60, 1),
        })
        state_path.write_text(json.dumps(progress, indent=2))
        print(f"[pretrain] window {tag}: {progress['history'][-1]}", flush=True)
    final_path = out_dir / "pretrained.pt"
    torch.save(make_checkpoint(model, dict(config)), final_path)
    summary = {"init": args.init, "base": str(args.base), "iterations": args.iterations,
               "grow_layers": args.grow_layers,
               "window": args.window, "steps_per_window": args.steps_per_window,
               "presentations_per_row": args.presentations_per_row,
               "tactic_labels": args.tactic_labels,
               "reanalysis_overlay": [str(p) for p in args.reanalysis_overlay or []],
               "windows": progress["history"], "checkpoint": str(final_path)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", type=Path, required=True,
                        help="base checkpoint (architecture source for every --init)")
    parser.add_argument("--init", choices=INITS, default="checkpoint")
    parser.add_argument("--buffers-dir", type=Path, required=True)
    parser.add_argument("--iterations", default="41-100")
    parser.add_argument("--window", type=int, default=10, help="iterations per window")
    parser.add_argument("--steps-per-window", type=int, default=2000,
                        help="fixed steps per window (ignored with --presentations-per-row)")
    parser.add_argument("--presentations-per-row", type=float, default=2.0,
                        help="target times each training row is drawn per window; sets "
                        "the step count from the window's row count (default 2). Pass "
                        "a negative value to use --steps-per-window instead")
    parser.add_argument("--tactic-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reanalysis-overlay", type=Path, nargs="*", default=None)
    parser.add_argument("--grow-layers", type=int, default=None,
                        help="capacity probe: append no-op transformer layers to the base "
                        "up to this many (e.g. 12), then train as usual")
    parser.add_argument("--alpha-step", type=float, default=None,
                        help="largest W5 alpha change per window (default: jump to the "
                        "held-out fit)")
    parser.add_argument("--retain-proofs-per-game", type=int, default=4)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default=RUN07["precision"], choices=("fp32", "bf16"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    return parser


def main(argv=None) -> int:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
