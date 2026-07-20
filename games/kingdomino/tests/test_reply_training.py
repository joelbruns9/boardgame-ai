from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from games.kingdomino.reply_training import (
    grouped_reply_loss,
    placement_drift,
    treatment_train_step,
)


def _group_batch():
    return {
        "legal_indices": [torch.tensor([0, 1, 2, 3])],
        "group_indices": [[torch.tensor([0, 1]), torch.tensor([2, 3])]],
        "target": [torch.tensor([0.7, 0.3])],
        "baseline_conditionals": [(np.array([0.5, 0.5]), np.array([0.5, 0.5]))],
    }


def test_grouped_reply_loss_depends_on_group_mass_not_within_group_split():
    batch = _group_batch()
    equal = torch.zeros((1, 4), requires_grad=True)
    redistributed = torch.tensor([[
        math.log(1.5), math.log(0.5), math.log(1.8), math.log(0.2),
    ]], requires_grad=True)

    equal_loss = grouped_reply_loss(equal, batch)
    redistributed_loss = grouped_reply_loss(redistributed, batch)

    assert float(redistributed_loss.item()) == pytest.approx(
        float(equal_loss.item()), abs=1e-6)
    redistributed_loss.backward()
    assert redistributed.grad is not None
    assert torch.isfinite(redistributed.grad).all()


def test_placement_drift_reports_redistribution_inside_pick_groups():
    batch = _group_batch()
    baseline = placement_drift(torch.zeros((1, 4)), batch)
    shifted = placement_drift(torch.tensor([[4.0, -4.0, 3.0, -3.0]]), batch)

    assert baseline["kl_to_baseline_p90"] == pytest.approx(0.0, abs=1e-12)
    assert shifted["kl_to_baseline_median"] > 0.5
    assert shifted["within_group_entropy_median"] < baseline["within_group_entropy_median"]


class _TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(3, 8)
        self.own = torch.nn.Linear(8, 1)
        self.opp = torch.nn.Linear(8, 1)
        self.win = torch.nn.Linear(8, 1)
        self.policy = torch.nn.Linear(8, 4)

    def forward(self, _mb, _ob, flat):
        hidden = torch.tanh(self.trunk(flat))
        return (
            self.own(hidden), self.opp(hidden), torch.sigmoid(self.win(hidden)),
            self.policy(hidden),
        )


def test_treatment_step_is_finite_and_updates_shared_model():
    torch.manual_seed(11)
    net = _TinyNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    batch_size = 3
    flat = torch.randn(batch_size, 3)
    legal = torch.ones((batch_size, 4), dtype=torch.bool)
    policy = torch.full((batch_size, 4), 0.25)
    ordinary = (
        torch.zeros((batch_size, 1)), torch.zeros((batch_size, 1)), flat,
        policy, legal, torch.zeros(batch_size),
        torch.ones((batch_size, 1)), torch.zeros((batch_size, 1)),
        torch.ones((batch_size, 1)),
    )
    reply = {
        **_group_batch(),
        "my_board": torch.zeros((1, 1)),
        "opp_board": torch.zeros((1, 1)),
        "flat": torch.randn(1, 3),
    }
    before = [parameter.detach().clone() for parameter in net.parameters()]

    metrics = treatment_train_step(
        net, ordinary, reply, optimizer, lambda_reply=0.15, score_scale=160.0)

    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["reply_loss"] >= 0.0
    assert any(not torch.equal(old, new) for old, new in zip(before, net.parameters()))
