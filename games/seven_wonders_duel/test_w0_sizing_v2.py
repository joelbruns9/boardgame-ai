"""Structural tests for W0 V2's fixed-budget strength measurements."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from .w0_sizing_v2 import (
    _FixedPairCollector,
    _root_priors_from_digest,
    aggregate_fixed_arenas,
    summarize_training,
)


def test_fixed_pair_collector_never_stops_early() -> None:
    collector = _FixedPairCollector()
    for index in range(88):
        result = collector.update(float(index % 3 == 0))
        assert result.decision == "continue"
    assert len(collector.game_scores) == 88
    assert len(collector.pair_scores) == 44


def test_extracts_root_priors_and_skips_child_subtrees() -> None:
    # Root with two edges. The first has a child node with its own edge.
    digest = [
        3, 0.2, 0, 0, 2, 41, 42, 2,
        10, 2, 0.1, 0.25, 0, 1,
        1, 1, 7, 1, 0.5,
        1, -0.1, 1, 0, 0, 1,
        99, 1, -0.1, 0.9, 0, 0,
        11, 1, 0.0, 0.75, 0, 0,
    ]
    assert _root_priors_from_digest(digest) == [0.25, 0.75]


def test_fixed_arena_aggregate_requires_and_weights_all_cells(tmp_path) -> None:
    cells = tmp_path / "arena_cells"
    cells.mkdir()
    for candidate_seed in range(3):
        for opponent_seed in range(3):
            name = f"M_vs_S_c{candidate_seed}_o{opponent_seed}"
            rate = 0.4 + 0.025 * (candidate_seed * 3 + opponent_seed)
            payload = {
                "cell": name,
                "candidate": "M",
                "opponent": "S",
                "precision": "fp32",
                "sims": 64,
                "games": 88,
                "pairs": 44,
                "pair_score_rate": rate,
                "pair_scores": [rate] * 44,
            }
            (cells / f"{name}.json").write_text(json.dumps(payload))

    aggregate_fixed_arenas(
        SimpleNamespace(
            output=str(tmp_path),
            match="M_vs_S",
            expected_cells=9,
        )
    )
    result = json.loads(
        (tmp_path / "arenas" / "M_vs_S.json").read_text()
    )
    assert result["games"] == 792
    assert result["pairs"] == 396
    assert result["cells"] == 9
    assert result["score_rate"] == pytest.approx(0.5)
    assert result["cell_min"] == pytest.approx(0.4)
    assert result["cell_max"] == pytest.approx(0.6)


def test_training_summary_uses_seed_means_and_lower_lr_tie_break(tmp_path) -> None:
    training = tmp_path / "training"
    training.mkdir()
    for arm in ("S", "M", "L"):
        rows = [
            (1e-4, 0, "sweep", 0.997),
            (2e-4, 0, "sweep", 0.990),
            (4e-4, 0, "sweep", 1.200),
            (1e-4, 1, "final", 0.990),
            (1e-4, 2, "final", 0.990),
            (2e-4, 1, "final", 0.987),
            (2e-4, 2, "final", 0.987),
        ]
        for lr, seed, kind, total in rows:
            payload = {
                "arm": arm,
                "kind": kind,
                "seed": seed,
                "learning_rate": lr,
                "tensor_cache_sha256": "cache",
                "precision": "fp32",
                "metrics": {
                    "total": total,
                    "value_acc": 0.6,
                    "joint7_acc": 0.5,
                    "policy_top1": 0.4,
                },
                "seconds": 10.0,
                "steps": 100,
                "history": [{"secs": 9.0}],
                "cuda_peak_allocated_bytes": 100,
                "cuda_peak_reserved_bytes": 200,
                "checkpoint": f"{arm}_{lr}_{seed}.pt",
                "checkpoint_sha256": f"{arm}_{lr}_{seed}",
            }
            name = f"{kind}_{arm}_lr{lr:.0e}_seed{seed}.json"
            (training / name).write_text(json.dumps(payload))

    summarize_training(SimpleNamespace(output=str(tmp_path)))
    result = json.loads((tmp_path / "training_summary.json").read_text())
    for arm in ("S", "M", "L"):
        assert result["arms"][arm]["selected_learning_rate"] == 1e-4
        assert result["arms"][arm]["selected_tie_candidates"] == [1e-4, 2e-4]
