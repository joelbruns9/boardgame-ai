"""Plan-deal curriculum, the helper schedule, and pool-rule-steered playouts."""

from __future__ import annotations

import argparse
import collections
import random

import pytest

from games.welcome_to import deal_curriculum as dc
from games.welcome_to import s2_run, self_play
from games.welcome_to.plans import available_plan_ids

wr = pytest.importorskip("welcome_to_rust")


def _jobs(n=400, seed=50_000):
    return list(zip(range(seed, seed + n), self_play.seat_counts(n)))


def test_deals_are_legal_deterministic_and_honour_fraction_and_exclusions():
    completion = {str(pid): 0.0 for stack in (1, 2, 3) for pid in available_plan_ids(stack, True)}
    jobs = _jobs()
    excluded = {jobs[0][0], jobs[1][0]}
    deals = dc.plan_deals(jobs, completion, fraction=0.25, seed=7, exclude=excluded)
    assert deals == dc.plan_deals(list(reversed(jobs)), completion, fraction=0.25, seed=7, exclude=excluded)
    assert abs(len(deals) - 0.25 * (len(jobs) - 2)) <= 1
    assert not excluded & set(deals)
    for plans in deals.values():
        for stack, pid in zip((1, 2, 3), plans):
            assert pid in available_plan_ids(stack, True)
    assert dc.plan_deals(jobs, completion, fraction=0.0, seed=7) == {}
    assert dc.plan_deals(jobs, {}, fraction=0.5, seed=7) == {}


def test_deals_favour_the_plans_the_learner_completes():
    easy = available_plan_ids(3, True)[0]
    completion = {str(pid): 0.0 for stack in (1, 2, 3) for pid in available_plan_ids(stack, True)}
    completion[str(easy)] = 0.7
    deals = dc.plan_deals(_jobs(2000), completion, fraction=1.0, seed=3)
    share = sum(plans[2] == easy for plans in deals.values()) / len(deals)
    others = len(available_plan_ids(3, True))
    expected = (0.7 + dc.DEFAULT_FLOOR) / (0.7 + others * dc.DEFAULT_FLOOR)
    assert share == pytest.approx(expected, abs=0.04)
    unlearned = collections.Counter(plans[2] for plans in deals.values())
    assert all(unlearned[pid] > 0 for pid in available_plan_ids(3, True)), "the floor keeps every plan in play"


def test_every_helper_falls_to_zero_at_the_end_iteration():
    args = argparse.Namespace(helpers_end_iteration=11)
    shares = [s2_run.helper_share(0.25, i, args) for i in range(1, 15)]
    assert shares[0] == pytest.approx(0.25)
    assert all(a >= b for a, b in zip(shares, shares[1:]))
    assert shares[10:] == [0.0] * 4
    constant = argparse.Namespace(helpers_end_iteration=0)
    assert s2_run.helper_share(0.2, 30, constant) == 0.2
    assert s2_run.helper_share(0.0, 1, args) == 0.0


def test_steered_playouts_apply_the_pool_rule_to_the_learner_only(monkeypatch):
    import numpy as np
    import torch

    from games.welcome_to import mcts, network as nw, rust_search, sibling_probe as sp

    calls = []
    monkeypatch.setattr(sp, "_POOL_STEER", lambda state, choice: (calls.append(int(state.actor)), choice)[1])
    torch.manual_seed(1)
    net = nw.WelcomeToNet(nw.NetConfig(sheet_hidden=16, sheet_out=8, trunk_hidden=24, trunk_blocks=1, head_hidden=16)).eval()
    packed = rust_search.PackedNetEvaluator(net, torch.device("cpu"), mcts.SearchConfig(simulations=2))
    states = [wr.RustGameState(s, players=2, advanced=True, expert=False, solo_rules=False) for s in (1, 2)]
    sp._finish_all(states, [2, 2], packed, steer=[True, False])
    assert calls and set(calls) == {0}, "only the learner's moves in steered games pass through the rule"
    assert all(s.is_terminal for s in states)
    natural = [wr.RustGameState(s, players=2, advanced=True, expert=False, solo_rules=False) for s in (1, 2)]
    sp._finish_all(natural, [2, 2], packed)
    assert natural[1].scores(None) == states[1].scores(None), "an unsteered game is untouched"
