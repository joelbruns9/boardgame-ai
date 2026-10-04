"""Encoder v4: plan characteristics, pool-target planes, no absolute seat, and the
shared plan encoder."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from games.welcome_to import encoder as enc
from games.welcome_to import macro_codec as mc
from games.welcome_to import network as nw
from games.welcome_to import plans as pl
from games.welcome_to.constants import POOL_POSITIONS
from games.welcome_to.game import GameConfig, GameState


def test_characteristics_describe_what_each_plan_asks_for():
    by_id = {p.id: pl.plan_characteristics(p) for p in pl.PLANS if p.id in pl.DEALT_PLAN_IDS}
    assert all(len(v) == pl.NUM_PLAN_CHARACTERISTICS for v in by_id.values())
    kind = lambda pid: by_id[pid][:7]
    assert kind(14) == (1, 0, 0, 0, 0, 0, 0)            # estates 3+4
    assert by_id[14][16:22] == (0, 0, 1 / 6, 1 / 6, 0, 0) and by_id[14][22] == 2 / 6
    assert by_id[18][7:10] == (0, 0, 1)                   # full bottom street
    assert by_id[25][10] == 1 and by_id[25][11] == 0       # pools in 2 streets
    assert by_id[23][10] == 0 and by_id[23][11] == 1       # parks in 2 streets
    assert by_id[24][10:13] == (1, 1, 1)                   # complete street: pools, parks, roundabout
    assert by_id[27][7:10] == (0, 1, 0)                    # park+pool middle street
    assert by_id[20][13] == 1 and by_id[21][14] == 1       # bis, temp
    for p in pl.PLANS:
        if p.id in pl.DEALT_PLAN_IDS:
            assert by_id[p.id][23:26] == tuple(float(p.stack == s) for s in (1, 2, 3))


def test_pool_targets_are_the_live_pool_boxes_of_needed_streets_only():
    rng = random.Random(5)
    checked = 0
    for seed in range(1, 200):
        state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True, solo_rules=False))
        while not state.is_terminal:
            sheet = state.sheets[0]
            for pid in state.plan_ids:
                plan = pl.PLANS[pid]
                boxes = pl.pool_target_boxes(plan, sheet)
                if plan.kind not in (pl.PlanKind.DECORATIVE, pl.PlanKind.COMPLETE_STREET) or (
                    plan.kind is pl.PlanKind.DECORATIVE and plan.params[0] == "park"
                ):
                    assert boxes == []
                    continue
                for x, y in boxes:
                    assert (x, y) in POOL_POSITIONS and sheet.numbers[x][y] is None
                    assert sheet.pools[x] < 3
                if not pl.feasible(plan, sheet):
                    assert boxes == [], "a dead pool plan needs no box"
                checked += 1
            mc.apply_macro(state, rng.choice(mc.legal_macros(state)))
        if checked > 2000:
            break
    assert checked > 200


def test_the_pool_plan_planes_carry_the_pool_targets():
    rng = random.Random(8)
    for seed in range(1, 400):
        state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True, solo_rules=False))
        slot = next((k for k, pid in enumerate(state.plan_ids) if pid == 25), None)
        if slot is None:
            continue
        planes, *_ = enc.encode_state(state, 0)
        marked = {(x, y) for x in range(3) for y in range(12) if planes[0, enc.P_PLAN_TARGET[slot], x, y] > 0}
        assert marked == set(pl.pool_target_boxes(pl.PLANS[25], state.sheets[0]))
        assert marked, "a fresh sheet needs every pool box for 'pools in two streets'"
        return
    pytest.skip("no deal with plan 25 in range")


def test_no_input_depends_on_the_absolute_viewer_seat():
    state = GameState.new(seed=3, config=GameConfig(players=3, advanced=True, solo_rules=False))
    assert "seat" not in dict(enc.GLOBAL_SCALAR_BLOCKS)
    assert enc.ENCODER_ABI_VERSION == 4 and enc.NUM_GLOBAL_SCALAR == 439


def test_the_plan_encoder_is_shared_across_slots():
    torch.manual_seed(0)
    net = nw.WelcomeToNet(nw.NetConfig(sheet_hidden=16, sheet_out=8, trunk_hidden=24, trunk_blocks=1, head_hidden=16, plan_hidden=16, plan_out=8)).eval()
    global_scalars = torch.randn(2, enc.NUM_GLOBAL_SCALAR)
    sheet_scalars = torch.randn(2, enc.MAX_SEATS, enc.NUM_SHEET_SCALAR)
    swapped_g, swapped_s = global_scalars.clone(), sheet_scalars.clone()
    a, b = enc.plan_identity_slot_slice(0), enc.plan_identity_slot_slice(2)
    swapped_g[:, a], swapped_g[:, b] = global_scalars[:, b], global_scalars[:, a]
    pa_, pb_ = enc.plan_slot_slice(0), enc.plan_slot_slice(2)
    swapped_s[:, :, pa_], swapped_s[:, :, pb_] = sheet_scalars[:, :, pb_], sheet_scalars[:, :, pa_]

    def plan_embeddings(g, s):
        slots = torch.stack(
            [torch.cat([g[:, i], s[:, :, p].reshape(2, -1)], dim=-1) for i, p in zip(net._plan_identity, net._plan_progress)],
            dim=1,
        )
        return net.plan_encoder(slots)

    with torch.no_grad():
        h = plan_embeddings(global_scalars, sheet_scalars)
        hs = plan_embeddings(swapped_g, swapped_s)
    assert torch.allclose(h[:, 0], hs[:, 2]) and torch.allclose(h[:, 2], hs[:, 0])
    assert torch.allclose(h[:, 1], hs[:, 1])
