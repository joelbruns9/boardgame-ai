"""Hygiene-rescue diagnostic: the assistance rule and the three-arm runner."""

from __future__ import annotations

import random

import pytest
import torch

from games.welcome_to import hygiene_rescue as hr
from games.welcome_to import macro_codec as mc
from games.welcome_to import network as nw
from games.welcome_to import self_play
from games.welcome_to.game import GameConfig, GameState

pytest.importorskip("welcome_to_rust")

_SMALL = nw.NetConfig(
    sheet_hidden=16, sheet_out=8, trunk_hidden=24, trunk_blocks=1, head_hidden=16
)


def _write_positions(count: int = 120):
    """(state, net-like random write choice) pairs from random 2-3p games."""
    rng = random.Random(4)
    out = []
    seed = 0
    while len(out) < count:
        seed += 1
        state = GameState.new(seed=seed, config=GameConfig(players=2 + seed % 2, advanced=True))
        while not state.is_terminal and len(out) < count:
            legal = mc.legal_macros(state)
            writes = [m for m in legal if hr._is_write(m)]
            if writes and rng.random() < 0.3:
                out.append((state.copy(), rng.choice(writes)))
            mc.apply_macro(state, rng.choice(legal))
    return out


def test_assistance_keeps_the_card_and_maximises_hygiene():
    changed = 0
    for state, choice in _write_positions():
        picked = hr.assisted_choice(state, choice)
        assert picked in mc.legal_macros(state)
        slot = mc.decode_macro_write(choice)[0]
        assert mc.decode_macro_write(picked)[0] == slot
        same_slot = [
            m for m in mc.legal_macros(state)
            if hr._is_write(m) and mc.decode_macro_write(m)[0] == slot
        ]
        best = max(hr.hygiene_key(state, m) for m in same_slot)
        assert hr.hygiene_key(state, picked) == best
        if hr.hygiene_key(state, choice) == best:
            assert picked == choice, "a tie must keep the seat's own choice"
        changed += picked != choice
        # idempotent: the assisted move is its own assistance
        assert hr.assisted_choice(state, picked) == picked
    assert changed > 0, "random writes were never improved; the test has no power"


def test_assistance_leaves_non_writes_alone():
    state = GameState.new(seed=3, config=GameConfig(players=2, advanced=True))
    for macro in mc.legal_macros(state):
        if not hr._is_write(macro):
            assert hr.assisted_choice(state, macro) == macro


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    torch.manual_seed(5)
    net = nw.WelcomeToNet(_SMALL).eval()
    out = tmp_path_factory.mktemp("rescue")
    results = hr.run(
        "unused.pt", out, games=6, simulations=2, assist_through=8, seed=4_400,
        inflight=6, workers=2, device="cpu", load=lambda _path: net,
    )
    games = {
        name: sorted(
            (self_play.SelfPlayTrajectory.from_json(line)
             for line in (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()),
            key=lambda t: t.seed,
        )
        for name in hr.ARMS
    }
    return results, games, out


def test_every_arm_plays_the_same_deals(arms):
    _results, games, _out = arms
    deals = {name: [(t.seed, t.players) for t in g] for name, g in games.items()}
    assert deals["normal"] == deals["focal"] == deals["all"]


def _assisted_writes(trajectory, seats, through):
    """Every write by ``seats`` up to ``through`` must be its own assistance."""
    state = GameState.new(seed=trajectory.engine_seed, config=trajectory.config, rng_kind=trajectory.rng)
    checked = 0
    for action in trajectory.actions:
        if hr._is_write(action) and state.turn <= through and state.actor in seats:
            assert hr.assisted_choice(state, action) == action
            checked += 1
        mc.apply_macro(state, action)
    return checked


def test_assistance_reaches_exactly_the_arm_s_seats_through_turn_t(arms):
    results, games, _out = arms
    assert results["arms"]["normal"]["assisted_decisions"] == 0
    assert results["arms"]["focal"]["assisted_decisions"] > 0
    assert (
        results["arms"]["all"]["assisted_decisions"]
        > results["arms"]["focal"]["assisted_decisions"]
    )
    # the assistant saw exactly the arm's writes through turn T -- no other
    # seat, no later turn -- and every one of them is its own assistance
    all_writes = sum(_assisted_writes(t, range(t.players), 8) for t in games["all"])
    focal_writes = sum(_assisted_writes(t, {0}, 8) for t in games["focal"])
    assert results["arms"]["all"]["assisted_decisions"] == all_writes
    assert results["arms"]["focal"]["assisted_decisions"] == focal_writes


def test_results_carry_paired_deltas_and_resume(arms, tmp_path):
    results, _games, out = arms
    for name in ("focal", "all"):
        delta = results["arms"][name]["vs_normal"]["end_turn"]
        assert delta["n"] == 6.0 and delta["lower"] <= delta["mean"] <= delta["upper"]
    again = hr.run(
        "unused.pt", out, games=6, simulations=2, assist_through=8, seed=4_400,
        inflight=6, workers=2, device="cpu",
        load=lambda _path: nw.WelcomeToNet(_SMALL).eval(),
    )
    assert again["arms"]["all"]["vs_normal"] == results["arms"]["all"]["vs_normal"]
