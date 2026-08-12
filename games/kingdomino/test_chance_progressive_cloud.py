from __future__ import annotations

import json
import hashlib
from pathlib import Path

from games.kingdomino.chance_progressive_cloud import (
    DEFAULT_CONFIG,
    _load_config,
    _prepare_phase_b,
    build_command,
    validate_phase,
)


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_frozen_phase_commands_preserve_treatment_boundary(tmp_path: Path) -> None:
    cfg = _load_config(DEFAULT_CONFIG)
    repo = tmp_path / "repo"
    run_root = repo / "runs" / "cloud"
    common = dict(
        repo_root=repo,
        run_root=run_root,
        python="python",
        batch_slots=96,
    )

    phase_a = build_command(cfg, "phase_a", **common)
    assert "--chance_progressive_decks" not in phase_a
    assert _option(phase_a, "--selfplay_generator_mode") == "current_best"
    assert _option(phase_a, "--measurement_iterations") == "5,8"
    assert _option(phase_a, "--measurement_games") == "384"
    assert _option(phase_a, "--measurement_sims") == "400"
    assert _option(phase_a, "--measurement_stop_ucb") == "0.5"

    prefill = build_command(cfg, "phase_b_prefill", **common)
    assert _option(prefill, "--iterations") == "23"
    assert _option(prefill, "--games_per_iter") == "400"
    assert _option(prefill, "--train_steps") == "0"
    assert _option(prefill, "--selfplay_generator_mode") == "current_best"
    assert _option(prefill, "--chance_progressive_decks") == "8,12"
    assert "--warm_buffer" not in prefill

    phase_b = build_command(cfg, "phase_b", **common)
    assert _option(phase_b, "--chance_progressive_decks") == "8,12"
    assert _option(phase_b, "--chance_d_min") == "4"
    assert _option(phase_b, "--chance_deck8_width_cap") == "16"
    assert _option(phase_b, "--chance_deck12_width_cap") == "16"
    assert _option(phase_b, "--chance_split_oversample") == "1.0"
    assert _option(phase_b, "--game_cpus") == "2"
    assert _option(phase_b, "--selfplay_generator_mode") == "soft_gate"
    assert _option(phase_b, "--train_steps") == "192"
    assert _option(phase_b, "--promotion_min_win_rate") == "0.55"
    assert _option(phase_b, "--soft_gate_stop_after_reverts") == "2"
    assert Path(_option(phase_b, "--current_best_path")).parts[-3:] == (
        "phase_b", "run_local_best", "current_best.pt")
    assert Path(_option(phase_b, "--warm_start")).parts[-2:] == (
        "best_checkpoint", "current_best.pt")
    assert Path(_option(phase_b, "--warm_buffer")).parts[-2:] == (
        "phase_b_prefill", "buffer_final.pkl")


def test_g4_validator_requires_mechanism_and_classifies_speed(tmp_path: Path) -> None:
    cfg = _load_config(DEFAULT_CONFIG)
    run_root = tmp_path / "cloud"
    g4 = run_root / "g4"
    g4.mkdir(parents=True)
    row = {
        "iter": 1,
        "game_cpus": 2,
        "games_per_sec": 0.31,
        "chance_progressive_decks": [8, 12],
        "chance_progressive_config": {
            "w0": 4,
            "n_init": 2,
            "d_min": 4,
            "width_schedule": [4, 8, 16, 32, 64, 70],
            "deck8_width_cap": 16,
            "deck12_width_cap": 16,
            "max_init_fraction": 0.25,
        },
        "chance_split_oversample": 1.0,
        "progressive_chance_init_fraction": 0.10,
        "progressive_chance_search_count_deck8": 1,
        "progressive_chance_search_count_deck12": 1,
        "progressive_chance_admission_count": 1,
        "progressive_chance_widening_count": 1,
        "progressive_chance_treated_examples_deck8": 1,
        "progressive_chance_treated_examples_deck12": 1,
        "exact_solve_count": 1,
    }
    (g4 / "training_log.jsonl").write_text(json.dumps(row) + "\n")

    result = validate_phase(cfg, "g4", run_root)
    assert result["valid"] is True
    assert result["recommendation"] == "proceed"

    row["progressive_chance_widening_count"] = 0
    row["games_per_sec"] = 0.19
    (g4 / "training_log.jsonl").write_text(json.dumps(row) + "\n")
    result = validate_phase(cfg, "g4", run_root)
    assert result["valid"] is False
    assert result["recommendation"] == "destroy_and_reassess"


def test_phase_a_validator_accepts_preregistered_ucb_stop(tmp_path: Path) -> None:
    cfg = _load_config(DEFAULT_CONFIG)
    run_root = tmp_path / "cloud"
    phase_a = run_root / "phase_a"
    phase_a.mkdir(parents=True)
    rows = []
    for iteration in range(1, 6):
        rows.append({
            "iter": iteration,
            "game_cpus": 2,
            "chance_progressive_decks": [],
            "generator_mode": "current_best",
            "measurement_stop_requested": iteration == 5,
        })
    (phase_a / "training_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    (phase_a / "measurement_iter_0005.json").write_text("{}\n")

    result = validate_phase(cfg, "phase_a", run_root)
    assert result["valid"] is True
    assert result["status"] == "stopped_by_measurement_ucb"


def test_phase_b_prefill_validator_requires_full_untrained_treatment_buffer(
    tmp_path: Path,
) -> None:
    cfg = _load_config(DEFAULT_CONFIG)
    run_root = tmp_path / "cloud"
    out = run_root / "phase_b_prefill"
    out.mkdir(parents=True)
    rows = []
    for iteration in range(1, 24):
        rows.append({
            "iter": iteration,
            "game_cpus": 2,
            "buffer_size": min(200000, iteration * 8700),
            "train_steps_per_iteration": 0,
            "generator_mode": "current_best",
            "chance_progressive_decks": [8, 12],
            "chance_progressive_config": {
                "w0": 4,
                "n_init": 2,
                "d_min": 4,
                "width_schedule": [4, 8, 16, 32, 64, 70],
                "deck8_width_cap": 16,
                "deck12_width_cap": 16,
                "max_init_fraction": 0.25,
            },
            "chance_split_oversample": 1.0,
            "progressive_chance_init_fraction": 0.02,
            "progressive_chance_search_count_deck8": 1,
            "progressive_chance_search_count_deck12": 1,
            "progressive_chance_admission_count": 1,
            "progressive_chance_widening_count": 1,
            "progressive_chance_treated_examples_deck8": 1,
            "progressive_chance_treated_examples_deck12": 1,
        })
    (out / "training_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))

    result = validate_phase(cfg, "phase_b_prefill", run_root)
    assert result["valid"] is True
    assert result["status"] == "complete"
    assert result["buffer_size"] == 200000

    repo_root = tmp_path / "repo"
    baseline = repo_root / "baseline.pt"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"pinned baseline")
    cfg["base_checkpoint"] = {
        "path": "baseline.pt",
        "sha256": hashlib.sha256(b"pinned baseline").hexdigest(),
    }
    (out / "iter_0023.pt").write_bytes(b"checkpoint")
    (out / "buffer_final.pkl").write_bytes(b"buffer")
    _prepare_phase_b(cfg, repo_root, run_root)
    local_best = run_root / "phase_b" / "run_local_best" / "current_best.pt"
    assert local_best.read_bytes() == b"pinned baseline"

    rows[-1]["train_steps_per_iteration"] = 192
    (out / "training_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    result = validate_phase(cfg, "phase_b_prefill", run_root)
    assert result["valid"] is False
