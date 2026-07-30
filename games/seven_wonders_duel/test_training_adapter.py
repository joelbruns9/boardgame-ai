"""Focused tests for the 7WD lifecycle adapter's candidate validation.

These use a lightweight fake loop so the finite-metric / reload / finite-weight
gate can be exercised without a full self-play + training pipeline.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from games.seven_wonders_duel.training_adapter import (
    SevenWondersDuelLifecycleAdapter,
    _record_stats,
)


class _FakeModel:
    def __init__(self, weights: torch.Tensor):
        self._weights = weights

    def state_dict(self):
        return {"block.weight": self._weights}


class _FakeLoop:
    def __init__(self, *, stats: dict, weights: torch.Tensor, reload_error: bool = False):
        self.last_training_stats = stats
        self._weights = weights
        self._reload_error = reload_error

    def load_model(self, path):
        if self._reload_error:
            raise RuntimeError("corrupt checkpoint")
        return _FakeModel(self._weights)


def _adapter(loop: _FakeLoop) -> SevenWondersDuelLifecycleAdapter:
    return SevenWondersDuelLifecycleAdapter(loop)


def test_validate_passes_for_finite_metrics_and_weights():
    loop = _FakeLoop(
        stats={"steps": [{"train": {"total": 0.9, "policy": 0.4}, "val": {"total": 1.0}}]},
        weights=torch.tensor([0.1, -0.2, 0.3]),
    )
    _adapter(loop)._validate_candidate(Path("candidate.pt"), 3)  # must not raise


def test_validate_rejects_non_finite_training_metric():
    loop = _FakeLoop(
        stats={"steps": [{"train": {"total": float("nan")}}]},
        weights=torch.tensor([0.0]),
    )
    with pytest.raises(RuntimeError, match="training diverged"):
        _adapter(loop)._validate_candidate(Path("candidate.pt"), 3)


def test_validate_rejects_non_finite_weights():
    loop = _FakeLoop(
        stats={"steps": [{"train": {"total": 0.5}}]},
        weights=torch.tensor([0.0, float("inf")]),
    )
    with pytest.raises(RuntimeError, match="non-finite weights"):
        _adapter(loop)._validate_candidate(Path("candidate.pt"), 3)


def test_validate_rejects_unreadable_checkpoint():
    loop = _FakeLoop(
        stats={"steps": [{"train": {"total": 0.5}}]},
        weights=torch.tensor([0.0]),
        reload_error=True,
    )
    with pytest.raises(RuntimeError, match="failed to reload"):
        _adapter(loop)._validate_candidate(Path("candidate.pt"), 3)


def test_w3_opponent_mix_uses_explicit_non_lossy_categories():
    records = [
        SimpleNamespace(
            seed=index,
            agents={"opponent_type": opponent, "kind": "legacy_is_ignored"},
            moves=(),
            victory_type=None,
            winner=None,
            first_player=0,
            scores=None,
        )
        for index, opponent in enumerate(
            ("current_best", "hof", "bot", "hof_bot")
        )
    ]

    generation, outcomes, _game_specific = _record_stats(records, {})

    assert generation.opponent_mix == {
        "current_best": 1,
        "hof": 1,
        "bot": 1,
        "hof_bot": 1,
    }
    assert set(outcomes.by_opponent_type) == {
        "current_best",
        "hof",
        "bot",
        "hof_bot",
    }
