"""Auditable launcher and validator for the frozen chance-progressive cloud run.

The defaults live in ``configs/chance_progressive_cloud_v1.json``.  This module
does not provision a vendor instance; it makes the experiment itself
reproducible once a clean checkout and the pinned, Git-ignored base checkpoint
are present.  ``run`` is a dry run unless ``--execute`` is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "chance_progressive_cloud_v1.json"
PHASE_NAMES = ("g4", "phase_a", "phase_b_prefill", "phase_b")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_config(data)
    return data


def validate_config(cfg: dict[str, Any]) -> None:
    """Fail closed if a frozen load-bearing invariant drifts."""
    errors: list[str] = []
    if cfg.get("schema_version") != 2:
        errors.append("schema_version must be 2")
    progressive = cfg.get("progressive", {})
    expected_progressive = {
        "decks": "8,12",
        "w0": 4,
        "n_init": 2,
        "d_min": 4,
        "deck8_width_cap": 16,
        "deck12_width_cap": 16,
        "max_init_fraction": 0.25,
    }
    for key, expected in expected_progressive.items():
        if progressive.get(key) != expected:
            errors.append(f"progressive.{key} must be {expected!r}")
    common = cfg.get("common", {})
    for key, expected in {
        "games_per_iteration": 400,
        "train_steps_per_iteration": 300,
        "sims": 4800,
        "fast_move_sims": 200,
        "full_search_fraction": 0.25,
        "game_cpus": 2,
        "chance_replay_weight": 1.0,
    }.items():
        if common.get(key) != expected:
            errors.append(f"common.{key} must be {expected!r}")
    if cfg.get("phase_a", {}).get("measurement_iterations") != "5,8":
        errors.append("phase_a.measurement_iterations must be '5,8'")
    prefill = cfg.get("phase_b_prefill", {})
    for key, expected in {
        "iterations": 23,
        "games_per_iteration": 400,
        "train_steps_per_iteration": 0,
        "minimum_buffer_examples": 175000,
    }.items():
        if prefill.get(key) != expected:
            errors.append(f"phase_b_prefill.{key} must be {expected!r}")
    phase_b = cfg.get("phase_b", {})
    for key, expected in {
        "iterations": 24,
        "train_steps_per_iteration": 192,
        "promotion_every": 5,
        "promotion_games": 384,
        "promotion_sims": 400,
        "promotion_min_win_rate": 0.55,
        "promotion_min_lcb": 0.50,
        "soft_gate_revert_win_rate": 0.48,
        "stop_after_consecutive_reverts": 2,
    }.items():
        if phase_b.get(key) != expected:
            errors.append(f"phase_b.{key} must be {expected!r}")
    checkpoint_sha = cfg.get("base_checkpoint", {}).get("sha256", "")
    if len(checkpoint_sha) != 64:
        errors.append("base_checkpoint.sha256 must be a full SHA-256")
    if errors:
        raise ValueError("invalid frozen cloud config:\n- " + "\n- ".join(errors))


def _path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _common_args(cfg: dict[str, Any], *, python: str, checkpoint_dir: Path,
                 batch_slots: int) -> list[str]:
    c = cfg["common"]
    return [
        python, "-m", "games.kingdomino.self_play",
        "--engine", "batched_open_loop",
        "--device", "cuda",
        "--async_solve",
        "--game_cpus", str(c["game_cpus"]),
        "--sims", str(c["sims"]),
        "--playout_cap_randomization",
        "--full_search_fraction", str(c["full_search_fraction"]),
        "--fast_move_sims", str(c["fast_move_sims"]),
        "--fast_move_dirichlet_epsilon", "0.0",
        "--fast_move_temp_moves", "0",
        "--channels", str(c["channels"]),
        "--blocks", str(c["blocks"]),
        "--bilinear_dim", str(c["bilinear_dim"]),
        "--batch_slots", str(batch_slots),
        "--leaf_batch", str(c["leaf_batch"]),
        "--virtual_loss", str(c["virtual_loss"]),
        "--batch_size", str(c["batch_size"]),
        "--lr", str(c["learning_rate"]),
        "--weight_decay", str(c["weight_decay"]),
        "--buffer", str(c["buffer_capacity"]),
        "--lambda_score", str(c["lambda_score"]),
        "--lambda_w", str(c["lambda_w"]),
        "--score_scale", str(c["score_scale"]),
        "--policy_weight", str(c["policy_weight"]),
        "--grad_clip", str(c["grad_clip"]),
        "--margin_gain", str(c["margin_gain"]),
        "--alpha", str(c["alpha"]),
        "--c_puct", str(c["c_puct"]),
        "--temp_moves", str(c["temp_moves"]),
        "--fpu", str(c["fpu"]),
        "--exact_endgame_max_secs", str(c["exact_endgame_max_secs"]),
        "--endgame_oversample", str(c["endgame_oversample"]),
        "--chance_split_oversample", str(c["chance_replay_weight"]),
        "--policy_target_pruning",
        "--benchmark_every", "0",
        "--elo_every", "0",
        "--checkpoint_dir", str(checkpoint_dir),
        "--save_buffer", str(checkpoint_dir / "buffer_final.pkl"),
        "--buffer_autosave_every", str(c["buffer_autosave_every"]),
        "--exact_fallback_positions", str(checkpoint_dir / "exact_fallbacks.jsonl"),
    ]


def _progressive_args(cfg: dict[str, Any]) -> list[str]:
    p = cfg["progressive"]
    return [
        "--chance_progressive_decks", p["decks"],
        "--chance_w0", str(p["w0"]),
        "--chance_n_init", str(p["n_init"]),
        "--chance_d_min", str(p["d_min"]),
        "--chance_width_schedule", p["width_schedule"],
        "--chance_deck8_width_cap", str(p["deck8_width_cap"]),
        "--chance_deck12_width_cap", str(p["deck12_width_cap"]),
        "--chance_max_init_fraction", str(p["max_init_fraction"]),
    ]


def phase_paths(run_root: Path) -> dict[str, Path]:
    return {phase: run_root / phase for phase in PHASE_NAMES}


def build_command(cfg: dict[str, Any], phase: str, *, repo_root: Path,
                  run_root: Path, python: str,
                  batch_slots: int) -> list[str]:
    if phase not in PHASE_NAMES:
        raise ValueError(f"unknown phase {phase!r}")
    paths = phase_paths(run_root)
    out = paths[phase]
    base_checkpoint = _path(repo_root, cfg["base_checkpoint"]["path"])
    args = _common_args(
        cfg, python=python, checkpoint_dir=out,
        batch_slots=batch_slots,
    )
    section = cfg[phase]
    common = cfg["common"]
    args += [
        "--iterations", str(section["iterations"]),
        "--games_per_iter", str(section.get(
            "games_per_iteration", common["games_per_iteration"])),
        "--train_steps", str(section.get(
            "train_steps_per_iteration", common["train_steps_per_iteration"])),
        "--min_buffer", str(section.get("min_buffer", common["min_buffer"])),
        "--seed", str(section["seed"]),
    ]

    if phase in ("g4", "phase_a", "phase_b_prefill"):
        args += [
            "--warm_start_current_best",
            "--current_best_path", str(base_checkpoint),
            "--selfplay_generator_mode", "current_best",
        ]
    if phase == "g4":
        args += _progressive_args(cfg)
    elif phase == "phase_a":
        args += [
            "--measurement_iterations", section["measurement_iterations"],
            "--measurement_games", str(section["measurement_games"]),
            "--measurement_sims", str(section["measurement_sims"]),
            "--measurement_seed", str(section["measurement_seed"]),
            "--measurement_confidence_z", str(section["measurement_confidence_z"]),
            "--measurement_stop_ucb", str(section["measurement_stop_ucb"]),
        ]
    elif phase == "phase_b_prefill":
        args += _progressive_args(cfg)
    else:
        local_best = out / "run_local_best" / "current_best.pt"
        args += [
            "--warm_start", str(base_checkpoint),
            "--warm_buffer", str(paths["phase_b_prefill"] / "buffer_final.pkl"),
            "--current_best_path", str(local_best),
            "--selfplay_generator_mode", "soft_gate",
            "--hof_dir", str(out / "run_local_best" / "hof"),
            "--promotion_every", str(section["promotion_every"]),
            "--promotion_games", str(section["promotion_games"]),
            "--promotion_sims", str(section["promotion_sims"]),
            "--promotion_seed", str(section["promotion_seed"]),
            "--promotion_min_win_rate", str(section["promotion_min_win_rate"]),
            "--promotion_min_lcb", str(section["promotion_min_lcb"]),
            "--promotion_fixed_suite", str(_path(
                repo_root, section["promotion_fixed_suite"])),
            "--promotion_fixed_suite_tolerance",
            str(section["promotion_fixed_suite_tolerance"]),
            "--soft_gate_revert_win_rate",
            str(section["soft_gate_revert_win_rate"]),
            "--soft_gate_stop_after_reverts",
            str(section["stop_after_consecutive_reverts"]),
        ]
        args += _progressive_args(cfg)
    return args


def _verify_base_checkpoint(cfg: dict[str, Any], repo_root: Path) -> Path:
    path = _path(repo_root, cfg["base_checkpoint"]["path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"missing Git-ignored base checkpoint: {path}; transfer it separately"
        )
    actual = _sha256(path)
    expected = cfg["base_checkpoint"]["sha256"].lower()
    if actual != expected:
        raise ValueError(f"base checkpoint SHA mismatch: expected {expected}, got {actual}")
    return path


def _git(*args: str, repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo_root, text=True, stderr=subprocess.STDOUT,
    ).strip()


def _verify_source(cfg: dict[str, Any], repo_root: Path,
                   expected_commit: str) -> str:
    head = _git("rev-parse", "HEAD", repo_root=repo_root)
    expected = _git("rev-parse", expected_commit, repo_root=repo_root)
    if head != expected:
        raise ValueError(f"HEAD {head} does not equal expected commit {expected}")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", cfg["minimum_source_commit"], head],
        cwd=repo_root,
    )
    if _git("status", "--porcelain", "--untracked-files=no", repo_root=repo_root):
        raise ValueError("tracked worktree changes present; cloud execution requires a clean checkout")
    return head


def _prepare_phase_b(cfg: dict[str, Any], repo_root: Path,
                     run_root: Path) -> None:
    paths = phase_paths(run_root)
    prefill_checkpoint = paths["phase_b_prefill"] / "iter_0023.pt"
    prefill_buffer = paths["phase_b_prefill"] / "buffer_final.pkl"
    prefill_log = paths["phase_b_prefill"] / "training_log.jsonl"
    for path in (prefill_checkpoint, prefill_buffer, prefill_log):
        if not path.is_file():
            raise FileNotFoundError(f"Phase B prefill prerequisite missing: {path}")
    rows = _read_log(prefill_log)
    expected_rows = int(cfg["phase_b_prefill"]["iterations"])
    if len(rows) != expected_rows:
        raise ValueError(
            f"Phase B prefill has {len(rows)} rows; expected {expected_rows}"
        )
    minimum_buffer = int(cfg["phase_b_prefill"]["minimum_buffer_examples"])
    final_buffer = int(rows[-1].get("buffer_size") or 0)
    if final_buffer < minimum_buffer:
        raise ValueError(
            f"Phase B prefill buffer has {final_buffer} examples; "
            f"requires at least {minimum_buffer}"
        )
    if any(int(row.get("train_steps_per_iteration") or 0) != 0 for row in rows):
        raise ValueError("Phase B prefill log contains nonzero training steps")
    prefill_validation = validate_phase(cfg, "phase_b_prefill", run_root)
    if (not prefill_validation["valid"]
            or prefill_validation.get("status") != "complete"):
        raise ValueError(
            "Phase B prefill failed validation: "
            + json.dumps(prefill_validation, sort_keys=True)
        )
    source = _verify_base_checkpoint(cfg, repo_root)
    destination = paths["phase_b"] / "run_local_best" / "current_best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) != cfg["base_checkpoint"]["sha256"]:
            raise ValueError(
                "run-local current_best already exists with a non-baseline hash; "
                "refusing to overwrite possible promotion history"
            )
    else:
        shutil.copy2(source, destination)


def _write_launch_manifest(path: Path, *, cfg_path: Path, cfg: dict[str, Any],
                           phase: str, command: list[str], source_commit: str,
                           base_checkpoint: Path) -> None:
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase": phase,
        "source_commit": source_commit,
        "config_path": str(cfg_path),
        "config_sha256": _sha256(cfg_path),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": _sha256(base_checkpoint),
        "command": command,
        "cwd": str(REPO_ROOT),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def execute_phase(cfg_path: Path, cfg: dict[str, Any], phase: str, *,
                  repo_root: Path, run_root: Path, python: str,
                  batch_slots: int,
                  expected_commit: str | None, execute: bool) -> int:
    command = build_command(
        cfg, phase, repo_root=repo_root, run_root=run_root, python=python,
        batch_slots=batch_slots,
    )
    print(shlex.join(command))
    if not execute:
        return 0
    if not expected_commit:
        raise ValueError("--expected-commit is required with --execute")
    source_commit = _verify_source(cfg, repo_root, expected_commit)
    base_checkpoint = _verify_base_checkpoint(cfg, repo_root)
    output = phase_paths(run_root)[phase]
    log_path = output / "training_log.jsonl"
    if log_path.exists() and log_path.stat().st_size:
        raise FileExistsError(f"refusing to append to an existing phase log: {log_path}")
    if any(output.glob("iter_*.pt")):
        raise FileExistsError(f"refusing to reuse phase directory with checkpoints: {output}")
    if phase == "phase_b":
        _prepare_phase_b(cfg, repo_root, run_root)
    output.mkdir(parents=True, exist_ok=True)
    _write_launch_manifest(
        output / "cloud_launch_manifest.json",
        cfg_path=cfg_path, cfg=cfg, phase=phase, command=command,
        source_commit=source_commit, base_checkpoint=base_checkpoint,
    )
    return subprocess.call(command, cwd=repo_root)


def _read_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"training log not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def validate_phase(cfg: dict[str, Any], phase: str, run_root: Path) -> dict[str, Any]:
    output = phase_paths(run_root)[phase]
    rows = _read_log(output / "training_log.jsonl")
    errors: list[str] = []
    expected_iterations = int(cfg[phase]["iterations"])
    if not rows:
        errors.append("training log contains no rows")
    for row in rows:
        if int(row.get("game_cpus") or 0) != 2:
            errors.append(f"iteration {row.get('iter')}: game_cpus drifted from 2")
        if phase == "phase_a":
            if (row.get("chance_progressive_decks")
                    or row.get("sampled_chance_split_decks")
                    or row.get("deck8_chance_enumeration")):
                errors.append(f"iteration {row.get('iter')}: chance treatment was not off")
            if row.get("generator_mode") != "current_best":
                errors.append(f"iteration {row.get('iter')}: generator was not current_best")
        else:
            if row.get("chance_progressive_decks") != [8, 12]:
                errors.append(f"iteration {row.get('iter')}: progressive decks drifted")
            pcfg = row.get("chance_progressive_config", {})
            expected_progressive = {
                "w0": 4,
                "n_init": 2,
                "d_min": 4,
                "width_schedule": [4, 8, 16, 32, 64, 70],
                "deck8_width_cap": 16,
                "deck12_width_cap": 16,
                "max_init_fraction": 0.25,
            }
            if any(pcfg.get(key) != expected
                   for key, expected in expected_progressive.items()):
                errors.append(f"iteration {row.get('iter')}: progressive config drifted")
            if float(row.get("chance_split_oversample") or 0.0) != 1.0:
                errors.append(f"iteration {row.get('iter')}: replay weight drifted")
            if (phase == "phase_b_prefill"
                    and row.get("generator_mode") != "current_best"):
                errors.append(
                    f"iteration {row.get('iter')}: prefill generator was not current_best")
            if (phase == "phase_b_prefill"
                    and int(row.get("train_steps_per_iteration") or 0) != 0):
                errors.append(
                    f"iteration {row.get('iter')}: prefill performed training")
            if phase == "phase_b" and row.get("generator_mode") != "soft_gate":
                errors.append(f"iteration {row.get('iter')}: generator was not soft_gate")
            if (phase == "phase_b"
                    and int(row.get("train_steps_per_iteration") or 0) != 192):
                errors.append(
                    f"iteration {row.get('iter')}: Phase B train steps drifted from 192")
    result: dict[str, Any] = {
        "phase": phase,
        "rows": len(rows),
        "expected_rows": expected_iterations,
        "last_iteration": rows[-1].get("iter") if rows else None,
        "errors": errors,
    }
    if phase == "g4" and rows:
        row = rows[-1]
        for key in (
            "progressive_chance_search_count_deck8",
            "progressive_chance_search_count_deck12",
            "progressive_chance_admission_count",
            "progressive_chance_widening_count",
            "progressive_chance_treated_examples_deck8",
            "progressive_chance_treated_examples_deck12",
            "exact_solve_count",
        ):
            if int(row.get(key) or 0) <= 0:
                errors.append(f"G4 mechanism check failed: {key} <= 0")
        init_fraction = float(row.get("progressive_chance_init_fraction") or 0.0)
        result["progressive_chance_init_fraction"] = init_fraction
        if init_fraction > cfg["progressive"]["max_init_fraction"] + 1e-12:
            errors.append("G4 initialization fraction exceeded the 25% guardrail")
        speed = float(row.get("games_per_sec") or 0.0)
        result["games_per_second"] = speed
        if speed >= cfg["g4"]["proceed_games_per_second"]:
            result["recommendation"] = "proceed"
        elif speed >= cfg["g4"]["reassess_games_per_second"]:
            result["recommendation"] = "proceed_slower"
        else:
            result["recommendation"] = "destroy_and_reassess"
            errors.append("G4 throughput is below the frozen 0.20 games/s floor")
    elif phase == "phase_a":
        for iteration in (5, 8):
            if len(rows) >= iteration:
                artifact = output / f"measurement_iter_{iteration:04d}.json"
                if not artifact.is_file():
                    errors.append(f"missing measurement artifact: {artifact}")
        if len(rows) == expected_iterations:
            result["status"] = "complete"
        elif rows and rows[-1].get("measurement_stop_requested"):
            result["status"] = "stopped_by_measurement_ucb"
        else:
            result["status"] = "partial"
    elif phase == "phase_b_prefill":
        final_buffer = int(rows[-1].get("buffer_size") or 0) if rows else 0
        result["buffer_size"] = final_buffer
        result["minimum_buffer_examples"] = int(
            cfg["phase_b_prefill"]["minimum_buffer_examples"])
        if len(rows) == expected_iterations:
            result["status"] = "complete"
            if final_buffer < result["minimum_buffer_examples"]:
                errors.append("completed prefill buffer is below the frozen minimum")
        else:
            result["status"] = "partial"
        for key in (
            "progressive_chance_search_count_deck8",
            "progressive_chance_search_count_deck12",
            "progressive_chance_admission_count",
            "progressive_chance_widening_count",
            "progressive_chance_treated_examples_deck8",
            "progressive_chance_treated_examples_deck12",
        ):
            if sum(int(row.get(key) or 0) for row in rows) <= 0:
                errors.append(f"prefill mechanism check failed: {key} <= 0")
        if any(float(row.get("progressive_chance_init_fraction") or 0.0)
               > cfg["progressive"]["max_init_fraction"] + 1e-12
               for row in rows):
            errors.append("prefill initialization fraction exceeded 25%")
    else:
        if rows and rows[-1].get("automatic_stop_requested") \
                and int(rows[-1].get("consecutive_gate_reverts") or 0) >= 2:
            result["status"] = "stopped_after_consecutive_reverts"
        elif len(rows) == expected_iterations:
            result["status"] = "complete"
        else:
            result["status"] = "partial"
    if len(rows) > expected_iterations:
        errors.append("more log rows than frozen phase allows")
    result["valid"] = not errors
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-config")
    run = sub.add_parser("run")
    run.add_argument("--phase", choices=PHASE_NAMES, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--batch-slots", type=int, required=True)
    run.add_argument("--expected-commit")
    run.add_argument("--execute", action="store_true")
    check = sub.add_parser("validate")
    check.add_argument("--phase", choices=PHASE_NAMES, required=True)
    check.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)

    cfg_path = args.config.resolve()
    cfg = _load_config(cfg_path)
    if args.command == "validate-config":
        print(json.dumps({"valid": True, "config": str(cfg_path)}, indent=2))
        return 0
    run_root = _path(REPO_ROOT, args.run_root).resolve()
    if args.command == "run":
        if args.batch_slots <= 0:
            raise ValueError("batch-slots must be positive")
        return execute_phase(
            cfg_path, cfg, args.phase, repo_root=REPO_ROOT,
            run_root=run_root, python=args.python,
            batch_slots=args.batch_slots,
            expected_commit=args.expected_commit, execute=args.execute,
        )
    result = validate_phase(cfg, args.phase, run_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
