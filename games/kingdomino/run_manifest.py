"""Run provenance artifacts for Kingdomino self-play training.

Milestone 5.5 makes each checkpoint traceable to the code, rules, schedules,
architecture, and machine context that produced it.  This module writes the
static run artifacts once at startup and appends checkpoint records as training
progresses.
"""
from __future__ import annotations

import dataclasses
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from games.kingdomino import action_codec, encoder
from games.kingdomino.network import KingdominoNet
from games.kingdomino.print_model_contract import ruleset_hash


MANIFEST_VERSION = 1

ARTIFACTS = {
    "run_manifest": "run_manifest.json",
    "git_commit": "git_commit.txt",
    "dirty_diff": "dirty_diff.patch",
    "model_contract": "model_contract.json",
    "ruleset_hash": "ruleset_hash.json",
    "schedule_config": "schedule_config.json",
    "hardware_benchmark": "hardware_benchmark.json",
}

SCHEDULE_FIELDS = {
    "lr": ("lr", "lr_schedule"),
    "alpha": ("alpha", "alpha_schedule"),
    "n_simulations": ("n_simulations", "sims_schedule"),
    "games_per_iteration": ("games_per_iteration", "games_per_iter_schedule"),
    "c_puct": ("c_puct", "c_puct_schedule"),
    "dirichlet_epsilon": ("dirichlet_epsilon", "dirichlet_epsilon_schedule"),
    "temp_moves": ("temp_moves", "temp_moves_schedule"),
    "train_steps_per_iteration": (
        "train_steps_per_iteration",
        "train_steps_schedule",
    ),
    "buffer_capacity": ("buffer_capacity", "buffer_capacity_schedule"),
    "fast_game_fraction": ("fast_game_fraction", "fast_game_fraction_schedule"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _run_git(args: list[str]) -> tuple[str, str, int]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=_repo_root(),
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:
        return "", str(exc), 1
    return proc.stdout, proc.stderr, proc.returncode


def _git_snapshot() -> dict[str, Any]:
    commit, commit_err, commit_rc = _run_git(["rev-parse", "HEAD"])
    branch, branch_err, branch_rc = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status, status_err, status_rc = _run_git(["status", "--porcelain"])
    diff, diff_err, diff_rc = _run_git(["diff", "--binary", "HEAD"])

    status_lines = [line for line in status.splitlines() if line.strip()]
    errors = {
        "commit": commit_err.strip() if commit_rc else "",
        "branch": branch_err.strip() if branch_rc else "",
        "status": status_err.strip() if status_rc else "",
        "diff": diff_err.strip() if diff_rc else "",
    }
    return {
        "commit": commit.strip() if commit_rc == 0 else "unknown",
        "branch": branch.strip() if branch_rc == 0 else "unknown",
        "dirty": bool(status_lines),
        "status_porcelain": status_lines,
        "dirty_diff": diff if diff_rc == 0 else "",
        "errors": {k: v for k, v in errors.items() if v},
    }


def _parameter_count(net: KingdominoNet | None) -> int | None:
    if net is None:
        return None
    return int(sum(p.numel() for p in net.parameters()))


def _norm_name(net: KingdominoNet | None) -> str | None:
    if net is None:
        return None
    has_batch = any(module.__class__.__name__ == "BatchNorm2d" for module in net.modules())
    has_group = any(module.__class__.__name__ == "GroupNorm" for module in net.modules())
    if has_batch:
        return "batch"
    if has_group:
        return "group"
    return "unknown"


def _model_contract(cfg: Any, net: KingdominoNet | None) -> dict[str, Any]:
    return {
        "network": "KingdominoNet",
        "checkpoint_version": int(KingdominoNet.checkpoint_version),
        "channels": int(getattr(cfg, "channels")),
        "blocks": int(getattr(cfg, "blocks")),
        "bilinear_dim": int(getattr(cfg, "bilinear_dim")),
        "norm": _norm_name(net),
        "parameter_count": _parameter_count(net),
        "board_shape": [
            int(encoder.NUM_BOARD_CHANNELS),
            int(encoder.CANVAS_SIZE),
            int(encoder.CANVAS_SIZE),
        ],
        "flat_size": int(encoder.FLAT_SIZE),
        "policy_head_size": int(action_codec.NUM_JOINT_ACTIONS),
        "value_heads": ["own_score", "opp_score", "win_prob"],
        "score_scale": float(getattr(cfg, "score_scale")),
        "policy_head_trained": True,
    }


def _parse_schedule(text: str, cast_name: str) -> list[list[Any]]:
    text = (text or "").strip()
    if not text:
        return []
    cast = float if cast_name == "float" else int
    out: list[list[Any]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            out.append(["parse_error", part])
            continue
        at, value = part.split(":", 1)
        try:
            out.append([int(at.strip()), cast(value.strip())])
        except ValueError:
            out.append(["parse_error", part])
    out.sort(key=lambda pair: pair[0] if isinstance(pair[0], int) else sys.maxsize)
    return out


def _schedule_config(cfg: Any) -> dict[str, Any]:
    schedules = {}
    for field, (base_field, schedule_field) in SCHEDULE_FIELDS.items():
        base_value = getattr(cfg, base_field)
        schedule_text = getattr(cfg, schedule_field, "")
        cast_name = "float" if isinstance(base_value, float) else "int"
        schedules[field] = {
            "base_field": base_field,
            "base_value": base_value,
            "schedule_field": schedule_field,
            "raw": schedule_text,
            "parsed": _parse_schedule(schedule_text, cast_name),
        }
    return {
        "step_indexing": "zero_based",
        "iteration_mapping": "schedule_step = iteration - 1",
        "selection_rule": "use the greatest schedule key <= schedule_step; otherwise use the base config value",
        "application_order": [
            "compute iteration-local config before self-play",
            "apply optimizer learning rate schedule",
            "apply replay buffer capacity schedule",
            "run self-play, training, diagnostics, benchmark, checkpoint, and logging with iteration-local values",
        ],
        "schedules": schedules,
    }


def _hardware_benchmark(cfg: Any) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_devices = []
    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            cuda_devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "major": int(props.major),
                    "minor": int(props.minor),
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )
    return {
        "captured_at_utc": _utc_now(),
        "requested_device": getattr(cfg, "device", None),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_devices": cuda_devices,
        "forward_benchmark": None,
        "note": "Hardware identity is captured automatically; explicit throughput sweeps remain external benchmark commands.",
    }


def _relative_artifacts() -> dict[str, str]:
    return dict(ARTIFACTS)


def initialize_run_manifest(
    cfg: Any,
    run_dir: str | Path,
    *,
    log_path: str | Path,
    net: KingdominoNet | None = None,
) -> dict[str, Any]:
    """Write startup provenance artifacts and return checkpoint metadata."""
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    git = _git_snapshot()
    contract = _model_contract(cfg, net)
    rules = {
        "ruleset_hash": ruleset_hash(),
        "source": "DOMINOES payload in games.kingdomino.dominoes",
        "hash_algorithm": "sha256",
        "hash_length": 8,
    }
    schedules = _schedule_config(cfg)
    hardware = _hardware_benchmark(cfg)
    config = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(vars(cfg))
    run_id = out_dir.name

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "repo_root": str(_repo_root()),
        "run_dir": str(out_dir.resolve()),
        "checkpoint_dir": str(out_dir.resolve()),
        "log_path": str(Path(log_path).resolve()),
        "command": sys.argv,
        "git": {
            "commit": git["commit"],
            "branch": git["branch"],
            "dirty": git["dirty"],
            "status_porcelain": git["status_porcelain"],
            "errors": git["errors"],
        },
        "config": config,
        "artifacts": _relative_artifacts(),
        "checkpoints": [],
    }

    _write_json(out_dir / ARTIFACTS["model_contract"], contract)
    _write_json(out_dir / ARTIFACTS["ruleset_hash"], rules)
    _write_json(out_dir / ARTIFACTS["schedule_config"], schedules)
    _write_json(out_dir / ARTIFACTS["hardware_benchmark"], hardware)
    _write_text(out_dir / ARTIFACTS["git_commit"], git["commit"] + "\n")
    _write_text(out_dir / ARTIFACTS["dirty_diff"], git["dirty_diff"])
    _write_json(out_dir / ARTIFACTS["run_manifest"], manifest)

    return checkpoint_manifest_metadata(out_dir, manifest=manifest, rules=rules)


def checkpoint_manifest_metadata(
    run_dir: str | Path,
    *,
    manifest: dict[str, Any] | None = None,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = Path(run_dir)
    if manifest is None:
        manifest_path = out_dir / ARTIFACTS["run_manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if rules is None:
        rules_path = out_dir / ARTIFACTS["ruleset_hash"]
        rules = json.loads(rules_path.read_text(encoding="utf-8"))

    return {
        "manifest_version": manifest.get("manifest_version"),
        "run_id": manifest.get("run_id"),
        "run_manifest_path": str((out_dir / ARTIFACTS["run_manifest"]).resolve()),
        "git_commit": manifest.get("git", {}).get("commit"),
        "git_dirty": manifest.get("git", {}).get("dirty"),
        "ruleset_hash": rules.get("ruleset_hash"),
        "model_contract_path": str((out_dir / ARTIFACTS["model_contract"]).resolve()),
        "schedule_config_path": str((out_dir / ARTIFACTS["schedule_config"]).resolve()),
        "hardware_benchmark_path": str((out_dir / ARTIFACTS["hardware_benchmark"]).resolve()),
    }


def record_checkpoint(
    run_dir: str | Path,
    checkpoint_path: str | Path,
    iteration: int,
) -> None:
    """Append/update a checkpoint record in run_manifest.json."""
    out_dir = Path(run_dir)
    manifest_path = out_dir / ARTIFACTS["run_manifest"]
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = Path(checkpoint_path)
    record = {
        "iteration": int(iteration),
        "name": checkpoint.name,
        "path": str(checkpoint.resolve()),
        "saved_at_utc": _utc_now(),
    }

    checkpoints = [
        item
        for item in manifest.get("checkpoints", [])
        if item.get("name") != record["name"]
    ]
    checkpoints.append(record)
    checkpoints.sort(key=lambda item: int(item.get("iteration", 0)))
    manifest["checkpoints"] = checkpoints
    manifest["last_checkpoint"] = record
    _write_json(manifest_path, manifest)
