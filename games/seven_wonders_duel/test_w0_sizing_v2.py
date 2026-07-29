"""Structural tests for W0 V2's fixed-budget strength measurements."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from .w0_sizing_v2 import _FixedPairCollector, aggregate_fixed_arenas


def test_fixed_pair_collector_never_stops_early() -> None:
    collector = _FixedPairCollector()
    for index in range(88):
        result = collector.update(float(index % 3 == 0))
        assert result.decision == "continue"
    assert len(collector.game_scores) == 88
    assert len(collector.pair_scores) == 44


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
