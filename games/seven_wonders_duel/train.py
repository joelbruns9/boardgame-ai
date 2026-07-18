"""Phase B trainer (plan §4): six-head training over buffer records.

Kingdomino trainer skeleton carried over: game-honest splits, trivial-baseline
comparisons printed next to net metrics, early stop, JSON summary. Checkpoints
embed ENCODER_SIGNATURE — a loader must refuse a checkpoint whose signature
disagrees with the live encoder (export discipline, spec §5.8).

Usage:
  python -m games.seven_wonders_duel.train --buffer <records.jsonl> [--model mlp]
      [--overfit] [--epochs N] [--out runs/phase_b]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .buffer import read_records
from .dataset import Example, collate, examples_from_records
from .encoder import ENCODER_SIGNATURE
from .mlp import SWDMlp
from .net import SWDNet, masked_policy_log_softmax

AUX_WEIGHT_DEFAULT = 0.2


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    aux_weight: float = AUX_WEIGHT_DEFAULT,
) -> tuple[torch.Tensor, dict[str, float]]:
    log_policy = masked_policy_log_softmax(outputs["policy"], batch["legal_mask"])
    # Targets are zero on illegal actions where log_policy is -inf; read only
    # legal positions so 0 * -inf never produces NaN.
    safe_log = torch.where(
        batch["legal_mask"], log_policy, torch.zeros_like(log_policy)
    )
    per_row = -(batch["policy"] * safe_log).sum(dim=-1)
    has_policy = batch["has_policy"]
    policy_loss = (
        per_row[has_policy].mean() if has_policy.any() else per_row.new_zeros(())
    )
    value_loss = F.cross_entropy(outputs["value"], batch["value_class"])
    joint7_loss = F.cross_entropy(outputs["joint7"], batch["joint7"])
    margin_valid = batch["margin_valid"]
    if margin_valid.any():
        margin_loss = F.mse_loss(
            outputs["margin"][margin_valid], batch["margin"][margin_valid]
        )
    else:
        margin_loss = outputs["margin"].new_zeros(())
    military_loss = F.mse_loss(outputs["military"], batch["military_final"])
    science_loss = F.mse_loss(outputs["science"], batch["sci_final"])
    total = (
        policy_loss
        + value_loss
        + aux_weight * (joint7_loss + margin_loss + military_loss + science_loss)
    )
    return total, {
        "total": float(total.detach()),
        "policy": float(policy_loss.detach()),
        "value": float(value_loss.detach()),
        "joint7": float(joint7_loss.detach()),
        "margin": float(margin_loss.detach()),
        "military": float(military_loss.detach()),
        "science": float(science_loss.detach()),
    }


@torch.no_grad()
def evaluate(model, examples: list[Example], device: str, batch_size: int = 512):
    model.eval()
    sums: dict[str, float] = {}
    correct = {"value": 0, "joint7": 0, "policy_top1": 0}
    policy_rows = 0
    count = 0
    for start in range(0, len(examples), batch_size):
        batch = collate(examples[start : start + batch_size], device)
        outputs = model(batch)
        _, parts = compute_losses(outputs, batch)
        rows = batch["value_class"].shape[0]
        for key, value in parts.items():
            sums[key] = sums.get(key, 0.0) + value * rows
        count += rows
        correct["value"] += int(
            (outputs["value"].argmax(-1) == batch["value_class"]).sum()
        )
        correct["joint7"] += int((outputs["joint7"].argmax(-1) == batch["joint7"]).sum())
        masked = outputs["policy"].masked_fill(~batch["legal_mask"], float("-inf"))
        top1 = masked.argmax(-1)
        target_top = batch["policy"].argmax(-1)
        has = batch["has_policy"]
        correct["policy_top1"] += int((top1[has] == target_top[has]).sum())
        policy_rows += int(has.sum())
    metrics = {key: value / count for key, value in sums.items()}
    metrics["value_acc"] = correct["value"] / count
    metrics["joint7_acc"] = correct["joint7"] / count
    metrics["policy_top1"] = correct["policy_top1"] / max(policy_rows, 1)
    model.train()
    return metrics


def baselines(examples: list[Example]) -> dict[str, float]:
    """What the heads must beat: majority-class rates and uniform policy."""

    def base_rate(values):
        counts: dict[int, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        return max(counts.values()) / len(values)

    mean_legal = sum(len(e.legal) for e in examples) / len(examples)
    return {
        "value_base_rate": base_rate([e.value_class for e in examples]),
        "joint7_base_rate": base_rate([e.joint7_class for e in examples]),
        "policy_uniform_loss": math.log(mean_legal),
    }


def game_honest_split(examples: list[Example], val_frac: float, seed: int = 0):
    keys = sorted({e.game_key for e in examples})
    rng = random.Random(seed)
    rng.shuffle(keys)
    val_keys = set(keys[: max(1, int(len(keys) * val_frac))])
    train = [e for e in examples if e.game_key not in val_keys]
    val = [e for e in examples if e.game_key in val_keys]
    return train, val


def make_checkpoint(model, config: dict) -> dict:
    return {
        "model_state": model.state_dict(),
        "config": config,
        "encoder_signature": ENCODER_SIGNATURE,
    }


def load_checkpoint(path, model) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["encoder_signature"] != ENCODER_SIGNATURE:
        raise ValueError(
            "checkpoint encoder signature does not match the live encoder — "
            "the encoding schema changed since this model was trained"
        )
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def build_model(name: str, d_model: int, layers: int):
    if name == "transformer":
        return SWDNet(d_model=d_model, layers=layers)
    if name == "mlp":
        return SWDMlp(d_model=d_model)
    raise ValueError(f"unknown model: {name}")


def train_loop(
    model,
    train_examples: list[Example],
    val_examples: list[Example] | None,
    *,
    device: str,
    epochs: int,
    batch_size: int = 512,
    lr: float = 2e-4,
    aux_weight: float = AUX_WEIGHT_DEFAULT,
    patience: int = 8,
    log=print,
):
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(0)
    best = {"val_total": float("inf"), "epoch": -1, "state": None}
    history = []
    for epoch in range(epochs):
        rng.shuffle(train_examples)
        start_time = time.time()
        running: dict[str, float] = {}
        batches = 0
        for start in range(0, len(train_examples), batch_size):
            batch = collate(train_examples[start : start + batch_size], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                outputs = model(batch)
                total, parts = compute_losses(outputs, batch, aux_weight)
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + value
            batches += 1
        scheduler.step()
        train_parts = {k: v / batches for k, v in running.items()}
        row = {"epoch": epoch, "train": train_parts, "secs": time.time() - start_time}
        if val_examples:
            val_metrics = evaluate(model, val_examples, device, batch_size)
            row["val"] = val_metrics
            log(
                f"epoch {epoch}: train total {train_parts['total']:.4f} "
                f"(policy {train_parts['policy']:.4f} value {train_parts['value']:.4f}) "
                f"| val total {val_metrics['total']:.4f} "
                f"value_acc {val_metrics['value_acc']:.3f} "
                f"joint7_acc {val_metrics['joint7_acc']:.3f} "
                f"policy_top1 {val_metrics['policy_top1']:.3f} "
                f"[{row['secs']:.0f}s]"
            )
            if val_metrics["total"] < best["val_total"] - 1e-4:
                best = {
                    "val_total": val_metrics["total"],
                    "epoch": epoch,
                    "state": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                }
            elif epoch - best["epoch"] >= patience:
                log(f"early stop at epoch {epoch} (best epoch {best['epoch']})")
                history.append(row)
                break
        else:
            log(
                f"epoch {epoch}: train total {train_parts['total']:.4f} "
                f"(policy {train_parts['policy']:.4f} value {train_parts['value']:.4f} "
                f"joint7 {train_parts['joint7']:.4f}) [{row['secs']:.0f}s]"
            )
        history.append(row)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return history


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", nargs="+", required=True)
    parser.add_argument("--model", choices=("transformer", "mlp"), default="transformer")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--aux-weight", type=float, default=AUX_WEIGHT_DEFAULT)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--overfit", action="store_true", help="no split, no early stop")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    records = [record for path in args.buffer for record in read_records(path)]
    print(f"loaded {len(records)} games; featurizing (encoder {ENCODER_SIGNATURE[:12]})")
    examples = examples_from_records(records)
    print(f"{len(examples)} decision states")
    base = baselines(examples)
    print(f"baselines: {json.dumps({k: round(v, 4) for k, v in base.items()})}")

    if args.overfit:
        train_examples, val_examples = examples, None
    else:
        train_examples, val_examples = game_honest_split(examples, args.val_frac)
        print(f"split: {len(train_examples)} train / {len(val_examples)} val states")

    model = build_model(args.model, args.d_model, args.layers)
    params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {params:,} params on {args.device}")
    history = train_loop(
        model,
        train_examples,
        val_examples,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        aux_weight=args.aux_weight,
    )
    final = evaluate(model, val_examples or train_examples, args.device, args.batch_size)
    print(f"final: {json.dumps({k: round(v, 4) for k, v in final.items()})}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        config = {
            "model": args.model,
            "d_model": args.d_model,
            "layers": args.layers,
            "aux_weight": args.aux_weight,
        }
        torch.save(make_checkpoint(model, config), out / f"{args.model}.pt")
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "config": config,
                    "baselines": base,
                    "final": final,
                    "history": history[-5:],
                    "encoder_signature": ENCODER_SIGNATURE,
                },
                indent=2,
                default=float,
            )
        )
        print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
