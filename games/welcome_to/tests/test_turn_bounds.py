"""One-turn bounds that the encoder still uses -- spec §6.2 and §8.

Retained from `test_plan_threat.py` when the §6.4 threat predicates were deleted
(review 2026-09-25).  The ceiling test is now checked against the REAL engine
(`engine_turn_oracle.engine_one_turn_sheets`) rather than the deleted
hand-written turn model, so it no longer compares a model against itself.

**The one-turn ceiling is tested over EVERY `steps_left`, not just a range some
caller enumerates.**  Restricting it that way makes the test structurally unable
to catch a ceiling that is too low -- the defect review found in the first
draft, where `ESTATE` was given a ceiling of 3 and one SURVEYOR fence can close
two steps at once.
"""
from __future__ import annotations

import random

import pytest

from games.welcome_to.bots import GreedyBot
from games.welcome_to.constants import Effect
from games.welcome_to.game import (
    GameConfig,
    GameState,
    Phase,
    bis_usable,
    max_houses_this_turn,
    one_turn_ceiling,
)
from games.welcome_to.plans import DEALT_PLAN_IDS, PLANS, progress
from games.welcome_to.sheet import Sheet
from games.welcome_to.tests.engine_turn_oracle import engine_one_turn_sheets


def _late_turn_starts(seeds, *, min_turn: int = 18) -> list[GameState]:
    """States at a turn start with dense sheets, where the engine walk is cheap."""
    out = []
    for seed in seeds:
        state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
        bots = [GreedyBot(random.Random(seed * 7 + i)) for i in range(2)]
        while not state.is_terminal:
            if state.turn >= min_turn and state.phase is Phase.CHOOSE_CARDS:
                out.append(state.copy())
                break
            state.apply(bots[state.actor].act(state))
    return out


def test_no_single_turn_beats_the_ceiling():
    checked = 0
    for state in _late_turn_starts(range(6)):
        seat = state.actor
        sheet = state.sheets[seat]
        for candidate in engine_one_turn_sheets(state, seat, cap=400_000):
            for plan_id in DEALT_PLAN_IDS:
                plan = PLANS[plan_id]
                gained = progress(plan, sheet)[1] - progress(plan, candidate)[1]
                assert gained <= one_turn_ceiling(plan), (
                    f"plan {plan_id} ({plan.kind.name}) advanced {gained} steps "
                    f"in one turn, above its ceiling of {one_turn_ceiling(plan)}"
                )
                checked += 1
    assert checked > 0


def _sheet(rows) -> Sheet:
    sheet = Sheet.new()
    for x, row in enumerate(rows):
        for y, n in enumerate(row):
            sheet.numbers[x][y] = n
    return sheet


def test_a_single_fence_can_close_two_estate_steps():
    """Why ESTATE gets no early exit: one SURVEYOR fence, two steps."""
    sheet = _sheet([[1, 2, 3, 4, 5, 6]])
    sheet.fences[0][5] = True
    plan = PLANS[2]  # requires (3, 3, 3)
    before = progress(plan, sheet)[1]
    assert [sz for _, _, sz in sheet.free_estates()] == [6]
    sheet.fences[0][2] = True  # split 6 into 3 + 3
    assert before - progress(plan, sheet)[1] == 2
    assert one_turn_ceiling(plan) == len(plan.required_sizes)


def test_a_turn_can_place_three_houses():
    """roundabout -> choose + write -> bis, which an earlier draft capped at two.

    ⚠ A bis needs a BIS combination to be **offered**, not merely a candidate on
    the sheet -- so this searches for a state that actually offers one.
    """
    for seed in range(300):
        state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
        if not any(e is Effect.BIS for _, e in state.visible_cards(0)):
            continue
        state.sheets[0].numbers[0][3] = 8  # a neighbour for the bis to copy
        if max_houses_this_turn(state, 0, 0) == 3:
            return
    pytest.fail("no seed in 0..299 offered a BIS with room for three houses")


def test_three_houses_needs_a_bis_OFFER_not_just_a_candidate():
    """The defect review found: independent predicates summed to an illegal 3."""
    for seed in range(300):
        state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
        if any(e is Effect.BIS for _, e in state.visible_cards(0)):
            continue
        state.sheets[0].numbers[0][3] = 8  # a bis candidate exists on the sheet...
        assert bis_usable(state, 0, 0)
        # ...but no BIS is offered, so no single turn reaches three houses.
        assert max_houses_this_turn(state, 0, 0) == 2
        return
    pytest.fail("every seed offered a BIS")
