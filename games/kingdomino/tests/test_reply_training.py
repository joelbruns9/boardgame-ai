from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from games.kingdomino.reply_training import (
    actor_value_anchor_loss,
    grouped_reply_loss,
    mcts_actor_value,
    placement_drift,
    rank1_value_anchor_loss,
    searched_reply_value_loss,
    searched_reply_value_gap_loss,
    treatment_train_step,
)


def _group_batch():
    return {
        "legal_indices": [torch.tensor([0, 1, 2, 3])],
        "group_indices": [[torch.tensor([0, 1]), torch.tensor([2, 3])]],
        "target": [torch.tensor([0.7, 0.3])],
        "baseline_conditionals": [(np.array([0.5, 0.5]), np.array([0.5, 0.5]))],
        "search_value_target": torch.tensor([0.4]),
        "parent_searched_rank": torch.tensor([2]),
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


def test_searched_reply_value_loss_uses_mcts_actor_value():
    batch = _group_batch()
    own = torch.tensor([0.2], requires_grad=True)
    opp = torch.tensor([-0.1], requires_grad=True)
    win = torch.tensor([0.7], requires_grad=True)
    predicted = mcts_actor_value(
        own, opp, win, alpha=0.5, margin_gain=2.0)
    batch["search_value_target"] = predicted.detach().clone()

    loss = searched_reply_value_loss(
        own, opp, win, batch, alpha=0.5, margin_gain=2.0)

    assert float(loss.item()) == pytest.approx(0.0, abs=1e-12)
    (predicted.sum() + loss).backward()
    assert own.grad is not None
    assert win.grad is not None


def test_searched_reply_value_gap_loss_anchors_on_paired_rank1_state():
    batch = {
        "search_value_target": torch.tensor([0.1, 0.7]),
        "rank_pair_count": 1,
    }
    own = torch.tensor([0.0, 0.0], requires_grad=True)
    opp = torch.tensor([0.0, 0.0], requires_grad=True)
    # alpha=0 makes actor value exactly 2*win-1: [0.1, 0.6], leaving a
    # non-zero gap residual against the [0.1, 0.7] teacher pair.
    win = torch.tensor([0.55, 0.8], requires_grad=True)

    loss = searched_reply_value_gap_loss(
        own, opp, win, batch, alpha=0.0, margin_gain=2.0)

    assert float(loss.item()) > 0.0
    loss.backward()
    assert win.grad is not None
    assert torch.isfinite(win.grad).all()
    assert float(win.grad[0].item()) == pytest.approx(0.0, abs=1e-12)
    assert abs(float(win.grad[1].item())) > 0.0


def test_rank1_value_anchor_loss_only_updates_rank1_side():
    batch = {"rank_pair_count": 1}
    own = torch.tensor([0.0, 0.0], requires_grad=True)
    opp = torch.tensor([0.0, 0.0], requires_grad=True)
    win = torch.tensor([0.6, 0.8], requires_grad=True)

    loss = rank1_value_anchor_loss(
        own, opp, win, batch, torch.tensor([0.1]),
        alpha=0.0, margin_gain=2.0)

    assert float(loss.item()) > 0.0
    loss.backward()
    assert win.grad is not None
    assert abs(float(win.grad[0].item())) > 0.0
    assert float(win.grad[1].item()) == pytest.approx(0.0, abs=1e-12)


def test_actor_value_anchor_loss_matches_control_values():
    own = torch.tensor([0.1, -0.2], requires_grad=True)
    opp = torch.tensor([0.0, 0.1], requires_grad=True)
    win = torch.tensor([0.6, 0.4], requires_grad=True)
    target = mcts_actor_value(
        own, opp, win, alpha=0.5, margin_gain=2.0).detach()

    loss = actor_value_anchor_loss(
        own, opp, win, target, alpha=0.5, margin_gain=2.0)

    assert float(loss.item()) == pytest.approx(0.0, abs=1e-12)


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


class _TinyBatchNormNet(_TinyNet):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(8)

    def forward(self, _mb, _ob, flat):
        hidden = torch.tanh(self.norm(self.trunk(flat)))
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
    assert metrics["search_value_loss"] >= 0.0
    assert any(not torch.equal(old, new) for old, new in zip(before, net.parameters()))


def test_treatment_reply_forward_does_not_update_batchnorm_twice():
    torch.manual_seed(13)
    net = _TinyBatchNormNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    flat = torch.randn(3, 3)
    ordinary = (
        torch.zeros((3, 1)), torch.zeros((3, 1)), flat,
        torch.full((3, 4), 0.25), torch.ones((3, 4), dtype=torch.bool),
        torch.zeros(3), torch.ones((3, 1)), torch.zeros((3, 1)),
        torch.ones((3, 1)),
    )
    reply = {
        **_group_batch(),
        "my_board": torch.zeros((1, 1)),
        "opp_board": torch.zeros((1, 1)),
        "flat": torch.randn(1, 3),
    }
    before = int(net.norm.num_batches_tracked.item())

    treatment_train_step(
        net, ordinary, reply, optimizer, lambda_reply=0.15, score_scale=160.0)

    assert int(net.norm.num_batches_tracked.item()) == before + 1


def test_value_only_treatment_updates_model_without_reply_policy_weight():
    torch.manual_seed(17)
    net = _TinyNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    flat = torch.randn(2, 3)
    ordinary = (
        torch.zeros((2, 1)), torch.zeros((2, 1)), flat,
        torch.full((2, 4), 0.25), torch.ones((2, 4), dtype=torch.bool),
        torch.zeros(2), torch.ones((2, 1)), torch.zeros((2, 1)),
        torch.ones((2, 1)),
    )
    reply = {
        **_group_batch(),
        "my_board": torch.zeros((1, 1)),
        "opp_board": torch.zeros((1, 1)),
        "flat": torch.randn(1, 3),
    }
    before = [parameter.detach().clone() for parameter in net.parameters()]

    metrics = treatment_train_step(
        net, ordinary, reply, optimizer, lambda_reply=0.0,
        lambda_search_value=1.0, score_scale=160.0)

    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["search_value_loss"] >= 0.0
    assert any(not torch.equal(old, new) for old, new in zip(before, net.parameters()))
