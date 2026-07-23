"""Focused tests for the 7WD lifecycle adapter's candidate validation.

These use a lightweight fake loop so the finite-metric / reload / finite-weight
gate can be exercised without a full self-play + training pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from games.seven_wonders_duel.training_adapter import SevenWondersDuelLifecycleAdapter


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
        stats={"epochs": [{"train": {"total": 0.9, "policy": 0.4}, "val": {"total": 1.0}}]},
        weights=torch.tensor([0.1, -0.2, 0.3]),
    )
    _adapter(loop)._validate_candidate(Path("candidate.pt"), 3)  # must not raise


def test_validate_rejects_non_finite_training_metric():
    loop = _FakeLoop(
        stats={"epochs": [{"train": {"total": float("nan")}}]},
        weights=torch.tensor([0.0]),
    )
    with pytest.raises(RuntimeError, match="training diverged"):
        _adapter(loop)._validate_candidate(Path("candidate.pt"), 3)


def test_validate_rejects_non_finite_weights():
    loop = _FakeLoop(
        stats={"epochs": [{"train": {"total": 0.5}}]},
        weights=torch.tensor([0.0, float("inf")]),
    )
    with pytest.raises(RuntimeError, match="non-finite weights"):
        _adapter(loop)._validate_candidate(Path("candidate.pt"), 3)


def test_validate_rejects_unreadable_checkpoint():
    loop = _FakeLoop(
        stats={"epochs": [{"train": {"total": 0.5}}]},
        weights=torch.tensor([0.0]),
        reload_error=True,
    )
    with pytest.raises(RuntimeError, match="failed to reload"):
        _adapter(loop)._validate_candidate(Path("candidate.pt"), 3)
