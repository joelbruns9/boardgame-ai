"""Pool rescue: the pool rule and the two-arm runner."""

from __future__ import annotations

import random

import pytest
import torch

from games.welcome_to import macro_codec as mc
from games.welcome_to import network as nw
from games.welcome_to import placement_assist as pa
from games.welcome_to import plans as pl
from games.welcome_to import pool_rescue as pr
from games.welcome_to.game import GameConfig, GameState, Phase

pytest.importorskip("welcome_to_rust")

_SMALL = nw.NetConfig(sheet_hidden=16, sheet_out=8, trunk_hidden=24, trunk_blocks=1, head_hidden=16)


def _pool_positions(count: int = 150):
    """(state, random legal choice) at decisions where seat 0 has a live pool plan."""
    rng = random.Random(7)
    out, seed = [], 0
    while len(out) < count and seed < 400:
        seed += 1
        state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
        if not set(state.plan_ids) & pr.POOL_PLAN_IDS:
            continue
        while not state.is_terminal and len(out) < count:
            legal = mc.legal_macros(state)
            if state.actor == 0 and pr.needed_streets(state, 0)[2] and rng.random() < 0.5:
                out.append((state.copy(), rng.choice(legal)))
            mc.apply_macro(state, rng.choice(legal))
    return out


def test_the_rule_is_legal_and_never_kills_a_live_pool_plan_when_it_can_avoid_it():
    positions = _pool_positions()
    assert len(positions) > 50
    changed = 0
    for state, choice in positions:
        picked = pr.pool_choice(state, choice)
        assert picked in mc.legal_macros(state)
        changed += picked != choice
        if not pa.is_write(picked):
            continue
        live = pr.needed_streets(state, 0)[2]
        writes = [m for m in mc.legal_macros(state) if pa.is_write(m)]
        safe = [m for m in writes if not pr._kills(state, pr._resolved(state, m)[0], live)]
        if safe:
            assert picked in safe, "the rule played a killing write while a safe one existed"
    assert changed > 0, "the rule never intervened; the test has no power"


def test_the_rule_builds_a_needed_pool_instead_of_passing():
    rng = random.Random(3)
    seen = 0
    for seed in range(1, 600):
        state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
        if not set(state.plan_ids) & pr.POOL_PLAN_IDS:
            continue
        while not state.is_terminal:
            if (
                state.actor == 0
                and state.phase is Phase.ACTION_POOL
                and state._pool_available()
                and any(pid in pr.POOL_PLAN_IDS and 0 not in state.plan_turns[k] for k, pid in enumerate(state.plan_ids))
            ):
                assert pr.pool_choice(state, pr._PASS_POOL) == pr._BUILD_POOL
                seen += 1
            mc.apply_macro(state, rng.choice(mc.legal_macros(state)))
        if seen >= 5:
            break
    assert seen > 0, "random play reached no pool prompt in a needed street"


def test_the_two_arm_runner_reports_paired_pool_deal_deltas(tmp_path):
    torch.manual_seed(9)
    net = nw.WelcomeToNet(_SMALL).eval()
    results = pr.run("unused.pt", tmp_path, games=12, simulations=2, seed=9_100, workers=2, device="cpu", load=lambda _p: net)
    assert set(results["arms"]) == {"normal", "pool"}
    assert results["arms"]["normal"]["assisted_decisions"] == 0
    assert results["arms"]["pool"]["pool_plan_games"] == results["arms"]["normal"]["pool_plan_games"]
    if results["arms"]["pool"]["pool_plan_games"] >= 2:
        assert results["pool_vs_normal_on_pool_deals"]["pool_plan_done"] is not None
