"""Resumable W0.1 model-size experiment.

The ``all`` command is an orchestrator: preparation, every training arm, and
every arena pairing run in separate child processes. That keeps the 3-4 GB
example cache from being retained between GPU jobs and makes an interrupted
overnight run resume from the last completed JSON artifact.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

from games.kingdomino.promotion import wilson_lower_bound

from .buffer import read_records
from .dataset import (
    MAX_FEATURES,
    NUM_ACTIONS,
    collate,
    examples_from_record,
    solver_value_distribution,
)
from .net import default_heads
from .phase_d import ModelAgentSpec, PhaseDConfig, PhaseDLoop
from .train import (
    _evaluation_autocast,
    build_model,
    compute_losses,
    heads_from_config,
    make_checkpoint,
    stable_game_split,
    stable_is_validation,
    train_steps,
)


ARMS = {
    "S": {"d_model": 128, "layers": 4},
    "M": {"d_model": 256, "layers": 6},
    "L": {"d_model": 384, "layers": 8},
}
LEARNING_RATES = (1e-4, 2e-4, 4e-4)
FINAL_SEEDS = (0, 1, 2)
PAIRINGS = (("S", "M"), ("M", "L"), ("S", "L"))
INFERENCE_ROWS_PER_SECOND = {
    "S": {"fp32": 14_118, "bf16": 32_182},
    "M": {"fp32": 3_645, "bf16": 9_835},
    "L": {"fp32": 1_430, "bf16": 4_356},
}
STEP_MS_B512 = {"S": 38.7, "M": 122.8, "L": 251.6}
VRAM_GB = {"S": 0.79, "M": 2.19, "L": 4.31}
PARAMETERS = {"S": 1_030_000, "M": 5_210_000, "L": 14_900_000}


@contextlib.contextmanager
def _keep_system_awake():
    """Prevent Windows sleep only while the unattended orchestrator is alive."""

    if sys.platform != "win32":
        yield
        return
    import ctypes

    es_continuous = 0x80000000
    es_system_required = 0x00000001
    kernel32 = ctypes.windll.kernel32
    result = kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    if not result:
        raise OSError("SetThreadExecutionState failed")
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(es_continuous)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _job_stem(kind: str, arm: str, lr: float, seed: int) -> str:
    return f"{kind}_{arm}_lr{lr:.0e}_seed{seed}"


def _job_paths(output: Path, kind: str, arm: str, lr: float, seed: int):
    stem = _job_stem(kind, arm, lr, seed)
    return output / "training" / f"{stem}.json", output / "checkpoints" / f"{stem}.pt"


def prepare_cache(args) -> None:
    output = Path(args.output)
    cache = output / "examples.pt"
    metadata_path = output / "dataset.json"
    if cache.is_file() and metadata_path.is_file() and not args.force:
        print(f"dataset cache already complete: {cache}", flush=True)
        return

    buffer_dir = Path(args.buffer_dir)
    paths = sorted(buffer_dir.glob("iter_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no iteration buffers under {buffer_dir}")

    selected: list[Path] = []
    games = 0
    for path in reversed(paths):
        count = sum(1 for _ in path.open("r", encoding="utf-8"))
        if games + count > args.games:
            raise ValueError(
                f"{path} would cross the exact {args.games}-game boundary; "
                "partial iteration files are intentionally unsupported"
            )
        selected.append(path)
        games += count
        if games == args.games:
            break
    if games != args.games:
        raise ValueError(f"found only {games} recent games, need {args.games}")
    selected.reverse()

    print(
        f"deriving examples from {games:,} games in "
        f"{selected[0].name}..{selected[-1].name}",
        flush=True,
    )
    started = time.monotonic()
    examples = []
    for index, path in enumerate(selected, start=1):
        records = read_records(path)
        for record in records:
            examples.extend(
                examples_from_record(record, record_fast_moves=False)
            )
        print(
            f"[{index:02d}/{len(selected)}] {path.name}: "
            f"{len(examples):,} examples",
            flush=True,
        )

    train, val = stable_game_split(examples, args.val_fraction, args.split_salt)
    metadata = {
        "games": games,
        "examples": len(examples),
        "train_examples": len(train),
        "validation_examples": len(val),
        "val_fraction": args.val_fraction,
        "split_salt": args.split_salt,
        "record_fast_moves": False,
        "source_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in selected
        ],
        "derivation_seconds": time.monotonic() - started,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".pt.tmp")
    torch.save({"metadata": metadata, "examples": examples}, temporary)
    temporary.replace(cache)
    metadata["cache_bytes"] = cache.stat().st_size
    _atomic_json(metadata_path, metadata)
    print(
        f"cached {len(examples):,} byte-identical examples to {cache} "
        f"({metadata['cache_bytes'] / 2**30:.2f} GiB)",
        flush=True,
    )


def _pack_examples(examples, val_fraction: float, split_salt: str) -> dict:
    """Pack ragged examples once so every optimizer step avoids Python collation."""

    rows = len(examples)
    max_tokens = max(len(example.type_ids) for example in examples)
    storage = {
        "type_ids": torch.zeros(rows, max_tokens, dtype=torch.int8),
        "entity_ids": torch.zeros(rows, max_tokens, dtype=torch.int16),
        "aux_ids": torch.zeros(rows, max_tokens, dtype=torch.int16),
        "features": torch.zeros(
            rows, max_tokens, MAX_FEATURES, dtype=torch.float16
        ),
        "pad_mask": torch.ones(rows, max_tokens, dtype=torch.bool),
        "legal_mask": torch.zeros(rows, NUM_ACTIONS, dtype=torch.bool),
        "policy": torch.zeros(rows, NUM_ACTIONS, dtype=torch.float32),
        "has_policy": torch.zeros(rows, dtype=torch.bool),
        "value_weight": torch.ones(rows, dtype=torch.float32),
        "policy_weight": torch.ones(rows, dtype=torch.float32),
        "value_class": torch.zeros(rows, dtype=torch.int8),
        # Mirrors `collate`: the audit asserts this path is value-identical to
        # it, so a target added there has to be added here too.
        "value_soft": torch.zeros(rows, 3, dtype=torch.float32),
        "value_soft_valid": torch.zeros(rows, dtype=torch.bool),
        "value_solver": torch.zeros(rows, 3, dtype=torch.float32),
        "value_solver_valid": torch.zeros(rows, dtype=torch.bool),
        "joint7": torch.zeros(rows, dtype=torch.int8),
        "margin": torch.zeros(rows, dtype=torch.float32),
        "margin_valid": torch.zeros(rows, dtype=torch.bool),
        "military_final": torch.zeros(rows, dtype=torch.float32),
        "sci_final": torch.zeros(rows, 2, dtype=torch.float32),
    }
    train_indices: list[int] = []
    val_indices: list[int] = []
    decisions: dict[tuple[int | None, int], bool] = {}
    for row, example in enumerate(examples):
        count = len(example.type_ids)
        storage["type_ids"][row, :count].copy_(torch.tensor(example.type_ids))
        storage["entity_ids"][row, :count].copy_(
            torch.tensor(example.entity_ids)
        )
        storage["aux_ids"][row, :count].copy_(torch.tensor(example.aux_ids))
        storage["features"][row, :count].copy_(torch.tensor(example.features))
        storage["pad_mask"][row, :count] = False
        legal = torch.tensor(example.legal, dtype=torch.long)
        storage["legal_mask"][row, legal] = True
        storage["policy"][row, legal] = torch.tensor(
            example.policy_target, dtype=torch.float32
        )
        storage["has_policy"][row] = example.has_policy
        storage["value_weight"][row] = example.value_weight
        storage["policy_weight"][row] = example.policy_weight
        storage["value_class"][row] = example.value_class
        if example.root_value is not None:
            probability = (1.0 + float(example.root_value)) / 2.0
            probability = min(1.0, max(0.0, probability))
            storage["value_soft"][row, 0] = probability
            storage["value_soft"][row, 2] = 1.0 - probability
            storage["value_soft_valid"][row] = True
        proven = solver_value_distribution(example)
        if proven is not None:
            storage["value_solver"][row] = torch.tensor(proven)
            storage["value_solver_valid"][row] = True
        storage["joint7"][row] = example.joint7_class
        storage["margin"][row] = example.margin
        storage["margin_valid"][row] = example.margin_valid
        storage["military_final"][row] = example.military_final
        storage["sci_final"][row, 0] = example.sci_final_my
        storage["sci_final"][row, 1] = example.sci_final_opp
        key = (example.iteration, example.game_key)
        held = decisions.get(key)
        if held is None:
            held = stable_is_validation(
                example.iteration,
                example.game_key,
                val_fraction,
                split_salt,
            )
            decisions[key] = held
        (val_indices if held else train_indices).append(row)
    return {
        "storage": storage,
        "train_indices": torch.tensor(train_indices, dtype=torch.long),
        "val_indices": torch.tensor(val_indices, dtype=torch.long),
        "max_tokens": max_tokens,
    }


def tensorize_cache(args) -> None:
    output = Path(args.output)
    path = output / "tensor_examples.pt"
    metadata_path = output / "tensor_examples.json"
    if path.is_file() and metadata_path.is_file() and not args.force:
        print(f"tensor cache already complete: {path}", flush=True)
        return
    source = torch.load(
        output / "examples.pt", map_location="cpu", weights_only=False
    )
    examples = source["examples"]
    print(
        f"packing {len(examples):,} examples for value-identical indexed batching",
        flush=True,
    )
    started = time.monotonic()
    payload = _pack_examples(examples, args.val_fraction, args.split_salt)
    payload["metadata"] = source["metadata"]
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    metadata = {
        "examples": len(examples),
        "train_examples": len(payload["train_indices"]),
        "validation_examples": len(payload["val_indices"]),
        "max_tokens": payload["max_tokens"],
        "bytes": path.stat().st_size,
        "seconds": time.monotonic() - started,
        "sha256": _sha256(path),
        "value_semantics": "identical to dataset.collate after dtype expansion",
    }
    _atomic_json(metadata_path, metadata)
    print(
        f"packed tensor cache: {metadata['bytes'] / 2**30:.2f} GiB in "
        f"{metadata['seconds'] / 60:.1f} min",
        flush=True,
    )


def _tensor_batch(storage: dict, indices, device: str) -> dict:
    index = torch.as_tensor(indices, dtype=torch.long)
    batch = {key: value.index_select(0, index) for key, value in storage.items()}
    token_count = int((~batch["pad_mask"]).sum(dim=1).max())
    for key in ("type_ids", "entity_ids", "aux_ids", "features", "pad_mask"):
        batch[key] = batch[key][:, :token_count]
    batch["type_ids"] = batch["type_ids"].long()
    batch["entity_ids"] = batch["entity_ids"].long()
    batch["aux_ids"] = batch["aux_ids"].long()
    batch["features"] = batch["features"].float()
    batch["value_class"] = batch["value_class"].long()
    batch["joint7"] = batch["joint7"].long()
    if device != "cpu":
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }
    return batch


def audit_tensor_cache(args) -> None:
    """Verify the packed cache against production collation on real examples."""

    output = Path(args.output)
    cache_root = Path(args.cache_dir or args.output)
    source = torch.load(
        cache_root / "examples.pt", map_location="cpu", weights_only=False
    )
    packed = torch.load(
        cache_root / "tensor_examples.pt",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    examples = source["examples"]
    train, val = stable_game_split(
        examples, args.val_fraction, args.split_salt
    )
    train_indices = packed["train_indices"]
    val_indices = packed["val_indices"]
    expected_train = torch.tensor(
        [
            index
            for index, example in enumerate(examples)
            if not stable_is_validation(
                example.iteration,
                example.game_key,
                args.val_fraction,
                args.split_salt,
            )
        ],
        dtype=torch.long,
    )
    expected_val = torch.tensor(
        [
            index
            for index, example in enumerate(examples)
            if stable_is_validation(
                example.iteration,
                example.game_key,
                args.val_fraction,
                args.split_salt,
            )
        ],
        dtype=torch.long,
    )
    if not torch.equal(train_indices, expected_train):
        raise AssertionError("packed train indices differ from stable_game_split")
    if not torch.equal(val_indices, expected_val):
        raise AssertionError(
            "packed validation indices differ from stable_game_split"
        )
    if len(train) != len(train_indices) or len(val) != len(val_indices):
        raise AssertionError("packed split lengths differ from object split")

    rng = random.Random(args.audit_seed)
    population = range(len(train))
    digest = hashlib.sha256()
    checked_rows = 0
    checked_batches = []
    for batch_index in range(args.audit_batches):
        local_indices = rng.choices(population, k=args.batch_size)
        global_indices = train_indices.index_select(
            0, torch.tensor(local_indices, dtype=torch.long)
        )
        expected = collate([train[index] for index in local_indices])
        actual = _tensor_batch(packed["storage"], global_indices, "cpu")
        if actual.keys() != expected.keys():
            raise AssertionError("packed and collated batch keys differ")
        for key in expected:
            if actual[key].dtype != expected[key].dtype:
                raise AssertionError(
                    f"batch {batch_index} {key} dtype differs: "
                    f"{actual[key].dtype} != {expected[key].dtype}"
                )
            if not torch.equal(actual[key], expected[key]):
                raise AssertionError(
                    f"batch {batch_index} {key} values differ"
                )
        local_array = np.asarray(local_indices, dtype=np.int64)
        global_array = global_indices.numpy()
        digest.update(local_array.tobytes())
        digest.update(global_array.tobytes())
        checked_rows += len(local_indices)
        checked_batches.append(
            {
                "batch": batch_index,
                "local_sha256": hashlib.sha256(
                    local_array.tobytes()
                ).hexdigest(),
                "global_sha256": hashlib.sha256(
                    global_array.tobytes()
                ).hexdigest(),
            }
        )

    tensor_metadata = _read_json(cache_root / "tensor_examples.json")
    payload = {
        "cache_root": str(cache_root),
        "examples": len(examples),
        "train_examples": len(train),
        "validation_examples": len(val),
        "split_indices_exact": True,
        "real_batches_exact": True,
        "audit_batches": args.audit_batches,
        "batch_size": args.batch_size,
        "checked_rows": checked_rows,
        "audit_seed": args.audit_seed,
        "sample_sequence_sha256": digest.hexdigest(),
        "batches": checked_batches,
        "tensor_cache_sha256": tensor_metadata["sha256"],
        "source_files": source["metadata"]["source_files"],
    }
    _atomic_json(output / "pipeline_audit.json", payload)
    print(
        f"pipeline audit passed: {args.audit_batches} real batches, "
        f"{checked_rows:,} sampled rows, exact split/order/tensors",
        flush=True,
    )


@torch.no_grad()
def _evaluate_tensor_cache(
    model,
    storage: dict,
    indices: torch.Tensor,
    device: str,
    batch_size: int,
    precision: str,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    correct = {"value": 0, "joint7": 0, "policy_top1": 0}
    abs_err = {"margin": 0.0, "military": 0.0, "science": 0.0}
    margin_rows = 0
    policy_rows = 0
    count = 0
    for start in range(0, len(indices), batch_size):
        batch = _tensor_batch(
            storage, indices[start : start + batch_size], device
        )
        with _evaluation_autocast(device, precision):
            outputs = model(batch)
            _, parts = compute_losses(outputs, batch)
        rows = batch["value_class"].shape[0]
        for key, value in parts.items():
            sums[key] = sums.get(key, 0.0) + value * rows
        count += rows
        correct["value"] += int(
            (outputs["value"].argmax(-1) == batch["value_class"]).sum()
        )
        correct["joint7"] += int(
            (outputs["joint7"].argmax(-1) == batch["joint7"]).sum()
        )
        masked = outputs["policy"].masked_fill(
            ~batch["legal_mask"], float("-inf")
        )
        has = batch["has_policy"]
        correct["policy_top1"] += int(
            (masked.argmax(-1)[has] == batch["policy"].argmax(-1)[has]).sum()
        )
        policy_rows += int(has.sum())
        valid = batch["margin_valid"]
        if valid.any():
            abs_err["margin"] += float(
                (outputs["margin"][valid] - batch["margin"][valid]).abs().sum()
            )
            margin_rows += int(valid.sum())
        abs_err["military"] += float(
            (outputs["military"] - batch["military_final"]).abs().sum()
        )
        abs_err["science"] += float(
            (outputs["science"] - batch["sci_final"]).abs().mean(dim=-1).sum()
        )
    metrics = {key: value / count for key, value in sums.items()}
    metrics["value_acc"] = correct["value"] / count
    metrics["joint7_acc"] = correct["joint7"] / count
    metrics["policy_top1"] = correct["policy_top1"] / max(policy_rows, 1)
    metrics["margin_mae"] = abs_err["margin"] / max(margin_rows, 1)
    metrics["military_mae"] = abs_err["military"] / count
    metrics["science_mae"] = abs_err["science"] / count
    model.train()
    return metrics


def _load_examples(output: Path, val_fraction: float, split_salt: str):
    payload = torch.load(
        output / "examples.pt", map_location="cpu", weights_only=False
    )
    examples = payload["examples"]
    train, val = stable_game_split(examples, val_fraction, split_salt)
    return examples, train, val, payload["metadata"]


def train_one(args) -> None:
    output = Path(args.output)
    cache_root = Path(args.cache_dir or args.output)
    result_path, checkpoint_path = _job_paths(
        output, args.kind, args.arm, args.lr, args.seed
    )
    if result_path.is_file() and checkpoint_path.is_file() and not args.force:
        print(f"training job already complete: {result_path.name}", flush=True)
        return

    spec = ARMS[args.arm]
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    packed = torch.load(
        cache_root / "tensor_examples.pt",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    storage = packed["storage"]
    train_indices = packed["train_indices"]
    val_indices = packed["val_indices"]
    dataset_meta = packed["metadata"]
    tensor_metadata = _read_json(cache_root / "tensor_examples.json")
    model = build_model(
        "transformer",
        spec["d_model"],
        spec["layers"],
        heads=None,
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    warmup_steps = round(args.steps * args.warmup_fraction)
    print(
        f"{args.kind} {args.arm}: d={spec['d_model']} L={spec['layers']} "
        f"H={default_heads(spec['d_model'])} params={parameters:,} "
        f"lr={args.lr:g} seed={args.seed} precision={args.precision} "
        f"steps={args.steps} warmup={warmup_steps}",
        flush=True,
    )
    started = time.monotonic()

    def batch_getter(local_indices, device):
        local = torch.as_tensor(local_indices, dtype=torch.long)
        global_indices = train_indices.index_select(0, local)
        return _tensor_batch(storage, global_indices, device)

    history, _optimizer = train_steps(
        model,
        range(len(train_indices)),
        None,
        device=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=warmup_steps,
        validate_every=args.steps,
        restore_best_val=False,
        seed=args.seed,
        precision=args.precision,
        cosine_decay=True,
        batch_getter=batch_getter,
        log=lambda message: print(message, flush=True),
    )
    metrics = _evaluate_tensor_cache(
        model,
        storage,
        val_indices,
        args.device,
        args.batch_size,
        args.precision,
    )
    history[-1]["val"] = metrics
    elapsed = time.monotonic() - started
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = make_checkpoint(
        model,
        {
            "model": "transformer",
            "d_model": spec["d_model"],
            "layers": spec["layers"],
            "heads": default_heads(spec["d_model"]),
            "precision": args.precision,
            "w0_arm": args.arm,
            "w0_kind": args.kind,
            "learning_rate": args.lr,
            "seed": args.seed,
            "steps": args.steps,
            "warmup_steps": warmup_steps,
            "cosine_decay": True,
            "dataset_source_files": [
                row["name"] for row in dataset_meta["source_files"]
            ],
            "tensor_cache_sha256": tensor_metadata["sha256"],
        },
    )
    torch.save(checkpoint, checkpoint_path)
    result = {
        "kind": args.kind,
        "arm": args.arm,
        **spec,
        "heads": default_heads(spec["d_model"]),
        "parameters": parameters,
        "learning_rate": args.lr,
        "seed": args.seed,
        "precision": args.precision,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "warmup_steps": warmup_steps,
        "warmup_fraction": args.warmup_fraction,
        "cosine_decay": True,
        "batching": "packed_tensor_v1",
        "cache_root": str(cache_root),
        "tensor_cache_sha256": tensor_metadata["sha256"],
        "train_examples": len(train_indices),
        "validation_examples": len(val_indices),
        "metrics": metrics,
        "history": history,
        "seconds": elapsed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None
        ),
        "cuda_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved())
            if torch.cuda.is_available()
            else None
        ),
    }
    _atomic_json(result_path, result)
    print(
        f"completed {result_path.name}: val.total={metrics['total']:.6f} "
        f"in {elapsed / 60:.1f} min",
        flush=True,
    )
    del model, packed, storage, train_indices, val_indices, checkpoint, _optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def select_learning_rates(args) -> dict[str, dict[str, Any]]:
    output = Path(args.output)
    selection_path = output / "selection.json"
    selection: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        rows = []
        for lr in LEARNING_RATES:
            result_path, checkpoint_path = _job_paths(output, "sweep", arm, lr, 0)
            row = _read_json(result_path)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            rows.append(row)
        best_loss = min(row["metrics"]["total"] for row in rows)
        tied = [
            row
            for row in rows
            if row["metrics"]["total"] <= best_loss * (1.0 + args.lr_tie_fraction)
        ]
        chosen = min(tied, key=lambda row: row["learning_rate"])
        selection[arm] = {
            "best_learning_rate": chosen["learning_rate"],
            "chosen_val_total": chosen["metrics"]["total"],
            "minimum_val_total": best_loss,
            "tie_fraction": args.lr_tie_fraction,
            "tie_candidates": [row["learning_rate"] for row in tied],
            "decision": (
                "lowest LR within relative tie band"
                if len(tied) > 1
                else "unique minimum validation total"
            ),
        }
    _atomic_json(selection_path, selection)
    return selection


def _final_rows(output: Path, arm: str, selection: dict) -> list[dict]:
    lr = selection[arm]["best_learning_rate"]
    rows = []
    for seed in FINAL_SEEDS:
        kind = "sweep" if seed == 0 else "final"
        result_path, _checkpoint_path = _job_paths(output, kind, arm, lr, seed)
        rows.append(_read_json(result_path))
    return rows


def _best_checkpoint(output: Path, arm: str, selection: dict) -> tuple[Path, dict]:
    rows = _final_rows(output, arm, selection)
    best = min(rows, key=lambda row: row["metrics"]["total"])
    return Path(best["checkpoint"]), best


def _wilson_interval(points: float, samples: int, z: float = 1.96):
    lower = wilson_lower_bound(points, samples, z)
    upper = 1.0 - wilson_lower_bound(samples - points, samples, z)
    return lower, upper


@dataclass
class _WilsonResult:
    decision: str


class _PairedWilson:
    """Sequential stop checked on two-seat seed pairs, with pairs as N."""

    def __init__(self, minimum_games: int):
        self.minimum_games = minimum_games
        self.game_scores: list[float] = []
        self.pair_scores: list[float] = []
        self.decision = "continue"
        self.lower = 0.0
        self.upper = 1.0

    def update(self, score: float):
        self.game_scores.append(float(score))
        if len(self.game_scores) % 2:
            return _WilsonResult(self.decision)
        self.pair_scores.append(
            (self.game_scores[-2] + self.game_scores[-1]) / 2.0
        )
        self.lower, self.upper = _wilson_interval(
            sum(self.pair_scores), len(self.pair_scores)
        )
        if len(self.game_scores) >= self.minimum_games:
            if self.lower > 0.5:
                self.decision = "accept"
            elif self.upper < 0.5:
                self.decision = "reject"
        return _WilsonResult(self.decision)


def _load_agent(path: Path, name: str, sims: int, mode: str, top_k: int):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    return ModelAgentSpec(
        name=name,
        model_state=checkpoint["model_state"],
        d_model=int(config["d_model"]),
        layers=int(config["layers"]),
        heads=heads_from_config(config),
        sims=sims,
        mode=mode,
        top_k=top_k,
    )


def arena_one(args) -> None:
    output = Path(args.output)
    result_path = output / "arenas" / f"{args.smaller}_vs_{args.larger}.json"
    if result_path.is_file() and not args.force:
        print(f"arena already complete: {result_path.name}", flush=True)
        return
    selection = _read_json(output / "selection.json")
    smaller_path, smaller_row = _best_checkpoint(
        output, args.smaller, selection
    )
    larger_path, larger_row = _best_checkpoint(output, args.larger, selection)

    config = PhaseDConfig(
        run_dir=str(output / "arena_runtime"),
        device=args.device,
        precision=args.precision,
        d_model=128,
        layers=4,
        seed_games=0,
        promotion_every=0,
        gate_sims=args.sims,
        gate_max_games=args.max_games,
        rust_slots=args.rust_slots,
        rust_global_batch_cap=args.batch_cap,
        eval_search_mode="gumbel",
        search_mode="closed",
        top_k=16,
    )
    loop = PhaseDLoop(config)
    candidate = _load_agent(
        larger_path, args.larger, args.sims, config.search_mode, config.top_k
    )
    opponent = _load_agent(
        smaller_path, args.smaller, args.sims, config.search_mode, config.top_k
    )
    test = _PairedWilson(args.minimum_games)
    seed_offset = 80_000_000 + PAIRINGS.index(
        (args.smaller, args.larger)
    ) * 1_000_000
    print(
        f"arena {args.larger} (candidate) vs {args.smaller}: "
        f"precision={args.precision}, cap={args.max_games}",
        flush=True,
    )
    started = time.monotonic()
    outcomes = loop._rust_model_gate_waves(
        candidate, opponent, test, seed_offset
    )
    elapsed = time.monotonic() - started
    stop_reason = (
        "wilson_accept"
        if test.decision == "accept"
        else "wilson_reject"
        if test.decision == "reject"
        else "cap"
    )
    payload = {
        "smaller": args.smaller,
        "larger": args.larger,
        "candidate": args.larger,
        "candidate_checkpoint": str(larger_path),
        "candidate_seed": larger_row["seed"],
        "opponent_checkpoint": str(smaller_path),
        "opponent_seed": smaller_row["seed"],
        "precision": args.precision,
        "sims": args.sims,
        "games": len(test.game_scores),
        "pairs": len(test.pair_scores),
        "points": sum(test.game_scores),
        "score_rate": (
            sum(test.game_scores) / len(test.game_scores)
            if test.game_scores
            else 0.0
        ),
        "pair_points": sum(test.pair_scores),
        "pair_score_rate": (
            sum(test.pair_scores) / len(test.pair_scores)
            if test.pair_scores
            else 0.0
        ),
        "wilson_lower": test.lower,
        "wilson_upper": test.upper,
        "decision": test.decision,
        "stop_reason": stop_reason,
        "minimum_games": args.minimum_games,
        "max_games": args.max_games,
        "seconds": elapsed,
        "game_scores": test.game_scores,
        "pair_scores": test.pair_scores,
        "outcomes": [asdict(outcome) for outcome in outcomes],
    }
    _atomic_json(result_path, payload)
    print(
        f"completed {result_path.name}: score={payload['score_rate']:.3f} "
        f"CI=[{test.lower:.3f}, {test.upper:.3f}] games={len(outcomes)} "
        f"stop={stop_reason} time={elapsed / 60:.1f}m",
        flush=True,
    )


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def write_report(args) -> None:
    output = Path(args.output)
    dataset = _read_json(output / "dataset.json")
    selection = _read_json(output / "selection.json")
    precision_ab_path = output / "precision_ab" / "phase_d_ab.json"
    precision_ab = (
        _read_json(precision_ab_path) if precision_ab_path.is_file() else None
    )
    sweep = {
        arm: {
            f"{lr:.0e}": _read_json(
                _job_paths(output, "sweep", arm, lr, 0)[0]
            )["metrics"]["total"]
            for lr in LEARNING_RATES
        }
        for arm in ARMS
    }
    finals = {arm: _final_rows(output, arm, selection) for arm in ARMS}
    arenas = {
        f"{small}_vs_{large}": _read_json(
            output / "arenas" / f"{small}_vs_{large}.json"
        )
        for small, large in PAIRINGS
    }
    raw = {
        "dataset": dataset,
        "selection": selection,
        "sweep": sweep,
        "finals": finals,
        "arenas": arenas,
        "precision_ab": precision_ab,
        "constants": {
            "parameters": PARAMETERS,
            "step_ms_b512": STEP_MS_B512,
            "vram_gb": VRAM_GB,
            "inference_rows_per_second": INFERENCE_ROWS_PER_SECOND,
        },
        "judgment_calls": [
            {
                "decision": "Use bf16 for the sizing run.",
                "why": (
                    "The production A/B preserved every discrete trajectory and "
                    "measured a repeatable 1.2% median end-to-end throughput gain "
                    "over fp32. The gain is modest, but it is positive and the "
                    "fidelity criterion passed."
                ),
            },
            {
                "decision": "Include complete iter_0070 in the recent-game window.",
                "why": (
                    "The brief asks for the most recent complete games. Although "
                    "the run manifest stopped after iteration 69, iter_0070 is a "
                    "complete 400-game buffer; excluding it would make the input "
                    "window older for an unrelated downstream gate failure."
                ),
            },
            {
                "decision": "Pack examples into an indexed tensor cache after the first fit.",
                "why": (
                    "The original Python-object collation path took 107.5 minutes "
                    "for one S fit and left the GPU mostly idle. The packed path "
                    "was verified tensor-for-tensor against dataset.collate and "
                    "preserves the same random sampled indices, while reducing "
                    "the next S fit to 4.3 minutes. The first completed artifact "
                    "was retained rather than retrained."
                ),
            },
            {
                "decision": "Use linear warmup followed by cosine decay only in the sizing harness.",
                "why": (
                    "The brief requested the existing cosine schedule, but the "
                    "production fixed-step trainer only had warmup plus constant "
                    "LR. An opt-in flag preserves production behavior while "
                    "implementing the intended controlled offline experiment."
                ),
            },
            {
                "decision": "Warm up for one third of 4,000 steps.",
                "why": (
                    "This preserves the production 100/300 warmup proportion "
                    "rather than inventing a new width-dependent schedule."
                ),
            },
            {
                "decision": "Treat a seed pair, not an individual game, as Wilson N.",
                "why": (
                    "The protocol says confidence bounds are on pairs. Each pair "
                    "is the mean of the two seat-swapped games, preserving the "
                    "designed experimental unit."
                ),
            },
            {
                "decision": "Reuse the selected LR sweep seed-0 run as final seed 0.",
                "why": (
                    "The dataset, split, initialization seed, schedule, and "
                    "training budget are identical, so retraining would add no "
                    "information and would waste GPU time."
                ),
            },
            {
                "decision": "Define an LR tie as within 0.5% relative validation loss.",
                "why": (
                    "The sweep has one seed, so its seed noise is not directly "
                    "estimable before choosing final LRs. A narrow deterministic "
                    "band implements the brief's lower-LR tie break without "
                    "letting materially worse fits qualify."
                ),
            },
        ],
    }
    _atomic_json(output / "raw_results.json", raw)

    metrics = ("total", "value_acc", "joint7_acc", "policy_top1")
    final_summary = {}
    for arm, rows in finals.items():
        final_summary[arm] = {}
        for metric in metrics:
            values = [row["metrics"][metric] for row in rows]
            mean, sd = _mean_sd(values)
            final_summary[arm][metric] = {
                "mean": mean,
                "sd": sd,
                "min": min(values),
                "max": max(values),
            }

    dominated: set[str] = set()
    readings = []
    for small, large in PAIRINGS:
        row = arenas[f"{small}_vs_{large}"]
        if row["wilson_lower"] > 0.5:
            readings.append(
                f"{large} demonstrated a strength gain over {small}; both remain "
                "trade-off points because the stronger arm is slower."
            )
        elif row["wilson_upper"] < 0.5:
            dominated.add(large)
            readings.append(
                f"{large} was slower and significantly weaker than {small} in "
                "this arena, so it is dominated by that comparison."
            )
        else:
            readings.append(
                f"{large} vs {small} was inconclusive at the cap; no strength "
                "dominance is established."
            )
    frontier = [arm for arm in ARMS if arm not in dominated]

    lines = [
        "# W0 model-size experiment",
        "",
    ]
    if precision_ab is not None:
        fp32 = precision_ab["summary"]["fp32"]
        bf16 = precision_ab["summary"]["bf16"]
        divergence = precision_ab["trajectory_divergence_vs_first_arm"]["bf16"]
        lines += [
            "## Precision A/B",
            "",
            f"- fp32 median: {fp32['median_seconds']:.2f}s "
            f"({fp32['median_games_per_second']:.3f} games/s)",
            f"- bf16 median: {bf16['median_seconds']:.2f}s "
            f"({bf16['median_games_per_second']:.3f} games/s)",
            f"- End-to-end speedup: {bf16['speedup_vs_fp32']:.2f}×",
            f"- Discrete trajectory divergence: "
            f"{divergence['changed_games']}/{divergence['games']} "
            f"({divergence['rate']:.1%})",
            "",
        ]
    lines += [
        "## Dataset",
        "",
        f"- Games: {dataset['games']:,} ({dataset['source_files'][0]['name']} "
        f"through {dataset['source_files'][-1]['name']})",
        f"- Examples: {dataset['examples']:,}; train "
        f"{dataset['train_examples']:,}; validation "
        f"{dataset['validation_examples']:,}",
        f"- Cache: {dataset['cache_bytes'] / 2**30:.2f} GiB; derivation "
        f"{dataset['derivation_seconds'] / 60:.1f} min",
        "",
        "## LR sweep",
        "",
        "| arm | 1e-4 | 2e-4 | 4e-4 | selected | rule |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        choice = selection[arm]
        lines.append(
            f"| {arm} | {sweep[arm]['1e-04']:.5f} | "
            f"{sweep[arm]['2e-04']:.5f} | {sweep[arm]['4e-04']:.5f} | "
            f"{choice['best_learning_rate']:.0e} | {choice['decision']} |"
        )
    lines += [
        "",
        "## Final seeded fits",
        "",
        "| arm | best LR | val total | value acc | joint7 acc | policy top-1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        summary = final_summary[arm]
        lines.append(
            f"| {arm} | {selection[arm]['best_learning_rate']:.0e} | "
            f"{summary['total']['mean']:.5f} ± {summary['total']['sd']:.5f} | "
            f"{summary['value_acc']['mean']:.4f} ± "
            f"{summary['value_acc']['sd']:.4f} | "
            f"{summary['joint7_acc']['mean']:.4f} ± "
            f"{summary['joint7_acc']['sd']:.4f} | "
            f"{summary['policy_top1']['mean']:.4f} ± "
            f"{summary['policy_top1']['sd']:.4f} |"
        )
    lines += [
        "",
        "Ranges and per-seed raw values are in `raw_results.json`.",
        "",
        "## Paired 64-sim arenas",
        "",
        "| candidate | opponent | score | 95% Wilson (pairs) | games | stop | precision |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for small, large in PAIRINGS:
        row = arenas[f"{small}_vs_{large}"]
        lines.append(
            f"| {large} | {small} | {row['score_rate']:.3f} | "
            f"[{row['wilson_lower']:.3f}, {row['wilson_upper']:.3f}] | "
            f"{row['games']} | {row['stop_reason']} | {row['precision']} |"
        )
    lines += [
        "",
        "## Throughput and memory",
        "",
        "| arm | params | AMP step @b512 | VRAM | fp32 rows/s | bf16 rows/s | bf16 inference cost vs S |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        cost = (
            INFERENCE_ROWS_PER_SECOND["S"]["bf16"]
            / INFERENCE_ROWS_PER_SECOND[arm]["bf16"]
        )
        lines.append(
            f"| {arm} | {PARAMETERS[arm] / 1e6:.2f}M | "
            f"{STEP_MS_B512[arm]:.1f} ms | {VRAM_GB[arm]:.2f} GB | "
            f"{INFERENCE_ROWS_PER_SECOND[arm]['fp32']:,} | "
            f"{INFERENCE_ROWS_PER_SECOND[arm]['bf16']:,} | {cost:.2f}× |"
        )
    rows_per_iteration = 2_450_000
    lines += [
        "",
        "At the measured 2.45M NN rows per 400-game iteration, pure bf16 "
        "forward time on this laptop is:",
        "",
    ]
    for arm in ARMS:
        seconds = rows_per_iteration / INFERENCE_ROWS_PER_SECOND[arm]["bf16"]
        lines.append(f"- {arm}: {seconds:.0f}s of dense-forward work")
    lines += [
        "",
        "Cloud wall time cannot be claimed before the W5 box-specific sweep. "
        "The portable cost signal is the relative inference multiplier above; "
        "on the rented GPU, multiply its measured S forward share by 1.00×, "
        "3.27×, or 7.39× for S/M/L respectively.",
        "",
        "## Pareto reading",
        "",
        f"Frontier candidates under the arena evidence: {', '.join(frontier)}.",
        "",
    ]
    lines.extend(f"- {reading}" for reading in readings)
    lines += [
        "",
        "This is a frontier description, not a model-size recommendation.",
        "",
        "## Judgment calls",
        "",
    ]
    for row in raw["judgment_calls"]:
        lines.append(f"- **{row['decision']}** {row['why']}")
    lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output / 'report.md'}", flush=True)


def _run_child(args, command: list[str]) -> None:
    full = [
        sys.executable,
        "-m",
        "games.seven_wonders_duel.w0_sizing",
        *command,
        "--output",
        args.output,
        "--buffer-dir",
        args.buffer_dir,
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--games",
        str(args.games),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--warmup-fraction",
        str(args.warmup_fraction),
        "--val-fraction",
        str(args.val_fraction),
        "--split-salt",
        args.split_salt,
        "--lr-tie-fraction",
        str(args.lr_tie_fraction),
    ]
    if args.force:
        full.append("--force")
    print("$ " + subprocess.list2cmdline(full), flush=True)
    subprocess.run(full, check=True)


def run_all(args) -> None:
    with _keep_system_awake():
        _run_child(args, ["prepare"])
        _run_child(args, ["tensorize"])
        for arm in ARMS:
            for lr in LEARNING_RATES:
                _run_child(
                    args,
                    [
                        "train-one",
                        "--kind",
                        "sweep",
                        "--arm",
                        arm,
                        "--lr",
                        str(lr),
                        "--seed",
                        "0",
                    ],
                )
        selection = select_learning_rates(args)
        for arm in ARMS:
            lr = selection[arm]["best_learning_rate"]
            for seed in (1, 2):
                _run_child(
                    args,
                    [
                        "train-one",
                        "--kind",
                        "final",
                        "--arm",
                        arm,
                        "--lr",
                        str(lr),
                        "--seed",
                        str(seed),
                    ],
                )
        for smaller, larger in PAIRINGS:
            _run_child(
                args,
                [
                    "arena-one",
                    "--smaller",
                    smaller,
                    "--larger",
                    larger,
                    "--minimum-games",
                    str(args.minimum_games),
                    "--max-games",
                    str(args.max_games),
                    "--sims",
                    str(args.sims),
                    "--rust-slots",
                    str(args.rust_slots),
                    "--batch-cap",
                    str(args.batch_cap),
                ],
            )
        write_report(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "all",
            "prepare",
            "tensorize",
            "audit-cache",
            "train-one",
            "arena-one",
            "report",
        ),
    )
    parser.add_argument(
        "--output",
        default="games/seven_wonders_duel/runs/w0_sizing",
    )
    parser.add_argument(
        "--buffer-dir",
        default="games/seven_wonders_duel/runs/laptop_training_03/buffers",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="directory containing examples.pt/tensor_examples.pt; defaults to output",
    )
    parser.add_argument("--games", type=int, default=12_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--split-salt", default="w0-sizing-v1")
    parser.add_argument("--lr-tie-fraction", type=float, default=0.005)
    parser.add_argument("--kind", choices=("sweep", "final"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smaller", choices=tuple(ARMS))
    parser.add_argument("--larger", choices=tuple(ARMS))
    parser.add_argument("--minimum-games", type=int, default=60)
    parser.add_argument("--max-games", type=int, default=800)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--rust-slots", type=int, default=48)
    parser.add_argument("--batch-cap", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--audit-batches", type=int, default=8)
    parser.add_argument("--audit-seed", type=int, default=20260729)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "all":
        run_all(args)
    elif args.command == "prepare":
        prepare_cache(args)
    elif args.command == "tensorize":
        tensorize_cache(args)
    elif args.command == "audit-cache":
        audit_tensor_cache(args)
    elif args.command == "train-one":
        if args.kind is None or args.arm is None or args.lr is None or args.seed is None:
            raise SystemExit("train-one requires --kind, --arm, --lr, and --seed")
        train_one(args)
    elif args.command == "arena-one":
        if args.smaller is None or args.larger is None:
            raise SystemExit("arena-one requires --smaller and --larger")
        arena_one(args)
    else:
        write_report(args)


if __name__ == "__main__":
    main()
