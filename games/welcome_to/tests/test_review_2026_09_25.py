"""Permanent regressions for the 2026-09-25 external review.

`reviews/welcome-to-v3-advisor-f362584.md`.  Each test pins the reviewer's own
counterexample where one was given, so the defect cannot come back unnoticed.
Encoder findings F1-F6 live here; the advisor's F7-F10 live in
`test_bga_extract.py` with the capture machinery they need.
"""
from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from games.welcome_to import action_codec as ac
from games.welcome_to import deck_knowledge as dk
from games.welcome_to import encoder as enc
from games.welcome_to.bots import GreedyBot
from games.welcome_to.constants import (
    EXTREMITY_POSITIONS,
    PARK_BOXES,
    ROUNDABOUT_BOXES,
    STREET_SIZES,
    Effect,
)
from games.welcome_to.game import GameConfig, GameState, Phase, max_houses_this_turn
from games.welcome_to.plans import PLANS, PlanKind, can_be_scored, feasible, requirements
from games.welcome_to.sheet import Sheet
from games.welcome_to.tests.plan_reachability import can_ever_be_scored


# ──────────────────────────────────────────────────────────────────────────
# F1 -- EXTREMITIES: a roundabout can fill the extremity box itself
# ──────────────────────────────────────────────────────────────────────────
def _extremity_sheet() -> Sheet:
    """The reviewer's case: street 0 reads ``_ 0`` with a fence between them."""
    sheet = Sheet.new()
    sheet.write(0, (0, 1), turn=1)
    sheet.fences[0][0] = True
    for x, y in EXTREMITY_POSITIONS:
        if (x, y) != (0, 0):
            sheet.write(1 if y == 0 else 15, (x, y), turn=1)
    return sheet


def test_f1_extremity_reachable_only_by_roundabout_is_alive():
    plan = next(p for p in PLANS if p.kind is PlanKind.EXTREMITIES)
    sheet = _extremity_sheet()
    after = sheet.copy()
    after.build_roundabout((0, 0), turn=2)
    assert can_be_scored(plan, after)
    assert can_ever_be_scored(sheet, plan, max_states=2000)
    assert feasible(plan, sheet)


def test_f1_the_death_still_fires_once_no_roundabout_is_left():
    plan = next(p for p in PLANS if p.kind is PlanKind.EXTREMITIES)
    sheet = _extremity_sheet()
    sheet.roundabouts = ROUNDABOUT_BOXES
    assert not feasible(plan, sheet)


# ──────────────────────────────────────────────────────────────────────────
# F2 -- an empty integer gap takes nothing, temp or not
# ──────────────────────────────────────────────────────────────────────────
def _state_with_temp_next() -> GameState:
    for seed in range(200):
        state = GameState.new(seed=seed, config=GameConfig(players=2))
        if Effect.TEMP in state.next_effects(0):
            return state
    raise AssertionError("no seed offers TEMP next turn")


def test_f2_temp_and_next_turn_fit_are_zero_on_an_empty_gap():
    state = _state_with_temp_next()
    state.sheets[0].write(7, (0, 0), turn=0)
    state.sheets[0].write(8, (0, 2), turn=0)
    state.public_sheets = [s.copy() for s in state.sheets]
    planes = enc.encode_state(state, 0)[0]
    assert not any((0, 1) in state.sheets[0].available_locations(v) for v in range(18))
    assert planes[0, enc.P_FIT_TEMP, 0, 1] == 0.0
    assert planes[0, enc.P_FIT_NEXT_TURN, 0, 1] == 0.0
    assert planes[0, enc.P_FIT_DECK, 0, 1] == 0.0


def test_f2_a_one_value_gap_is_still_widened_by_temp():
    state = _state_with_temp_next()
    state.sheets[0].write(7, (0, 0), turn=0)
    state.sheets[0].write(9, (0, 2), turn=0)  # only 8 fits
    state.public_sheets = [s.copy() for s in state.sheets]
    planes = enc.encode_state(state, 0)[0]
    assert planes[0, enc.P_FIT_TEMP, 0, 1] > planes[0, enc.P_FIT_DECK, 0, 1] > 0.0


# ──────────────────────────────────────────────────────────────────────────
# F4 -- a dead alternative still states what it wants
# ──────────────────────────────────────────────────────────────────────────
def test_f4_dead_pool_and_park_plan_keeps_its_demand():
    sheet = Sheet.new()
    sheet.pools[1] = 2
    for y in range(STREET_SIZES[1]):
        sheet.write(y + 1, (1, y), turn=0)
    plan = PLANS[27]  # pool&park in street 1
    req = requirements(plan, sheet)
    assert not feasible(plan, sheet)
    assert req.pools_needed[1] == 1
    assert req.parks_needed[1] == PARK_BOXES[1]
    assert req.street_serves == (0, 0, 0)


def test_f4_five_bis_states_every_streets_shortfall():
    sheet = Sheet.new()
    plan = next(p for p in PLANS if p.kind is PlanKind.FIVE_BIS)
    # Street 0 packed with numbers: no bis can reach it, the others still can.
    for y in range(STREET_SIZES[0]):
        sheet.write(y + 1, (0, y), turn=0)
    req = requirements(plan, sheet)
    assert req.street_serves[0] == 0 and req.bis_needed[0] == 5
    assert req.street_serves[1] == 1 and req.bis_needed[1] == 5


# ──────────────────────────────────────────────────────────────────────────
# F5 -- the house ceiling honours the acting viewer's phase
# ──────────────────────────────────────────────────────────────────────────
_HOUSE_PHASES = (
    Phase.CHOOSE_CARDS,
    Phase.ROUNDABOUT_PLACE,
    Phase.WRITE_NUMBER,
    Phase.ACTION_BIS,
)


def _built(sheet: Sheet) -> int:
    return sum(1 for row in sheet.numbers for n in row if n is not None)


def _engine_remaining_houses(state: GameState) -> int:
    """Most houses the REAL engine lets the actor still build this turn."""
    seat, turn, start = state.actor, state.turn, _built(state.sheets[state.actor])
    best = 0

    def walk(current: GameState, in_bis: bool) -> None:
        nonlocal best
        best = max(best, _built(current.sheets[seat]) - start)
        if (
            current.actor != seat
            or current.turn != turn
            or current.is_terminal
            or current.phase not in _HOUSE_PHASES
            or in_bis
        ):
            return
        was_bis = current.phase is Phase.ACTION_BIS
        for action in current.legal_actions():
            walk(current.step(action), was_bis)

    walk(state, False)
    return best


def _actor_states(seeds, *, per_game: int = 6) -> list[GameState]:
    out = []
    for seed in seeds:
        rng = random.Random(seed)
        state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
        bots = [GreedyBot(random.Random(seed * 10 + i)) for i in range(2)]
        picked = 0
        while not state.is_terminal and picked < per_game:
            if state.turn >= 8 and state.phase in _HOUSE_PHASES and rng.random() < 0.25:
                out.append(state.copy())
                picked += 1
            state.apply(bots[state.actor].act(state))
    return out


def test_f5_house_ceiling_matches_an_engine_walk_at_every_phase():
    states = _actor_states(range(6))
    phases = {s.phase for s in states}
    assert {Phase.CHOOSE_CARDS, Phase.WRITE_NUMBER} <= phases
    for state in states:
        assert max_houses_this_turn(state, state.actor, state.actor) == (
            _engine_remaining_houses(state)
        ), f"phase {state.phase.name}"


def test_f5_action_bis_offers_only_the_bis():
    for seed in range(100):
        state = GameState.new(seed=seed, config=GameConfig(players=2))
        slots = [i for i, (_, e) in enumerate(state.visible_cards(0)) if e is Effect.BIS]
        if not slots:
            continue
        state.apply(ac.choose_stack(slots[0]))
        state.apply(state.legal_actions()[0])
        assert state.phase is Phase.ACTION_BIS
        expected = 1 if state.sheets[0].bis_candidates() else 0
        assert max_houses_this_turn(state, 0, 0) == expected
        return
    raise AssertionError("no seed offers BIS")


def test_max_houses_is_blind_to_an_opponents_live_sheet():
    """§3.6: the opponent's live mid-turn sheet must not move the feature."""
    state = GameState.new(seed=5, config=GameConfig(players=2, advanced=True))
    before = max_houses_this_turn(state, 0, 1)
    for x, size in enumerate(STREET_SIZES):
        for y in range(size):
            state.sheets[1].write(y, (x, y), turn=state.turn)
    assert max_houses_this_turn(state, 0, 1) == before


def test_max_houses_refuses_expert_mode():
    state = GameState.new(seed=5, config=GameConfig(players=2, expert=True))
    with pytest.raises(ValueError, match="standard mode only"):
        max_houses_this_turn(state, 0, 0)


# ──────────────────────────────────────────────────────────────────────────
# F3 -- the viewer's own reshuffle vote
# ──────────────────────────────────────────────────────────────────────────
def _voted_state() -> GameState:
    """Viewer 0 completes plan 0 (SEVEN_TEMP) and votes yes."""
    state = GameState.new(seed=0, config=GameConfig(players=2, advanced=True))
    state.plan_ids = (21, 23, 14)
    state.sheets[0].temps = 7
    state.public_sheets = [s.copy() for s in state.sheets]
    state.apply(ac.choose_stack(state.playable_slots()[0]))
    state.apply(state.legal_actions()[0])
    while state.phase is not Phase.CHOOSE_PLAN:
        state.apply(state.legal_actions()[-1])
    state.apply(ac.choose_plan(0))
    assert state.phase is Phase.ASK_RESHUFFLE
    state.apply(ac.A_RESHUFFLE_YES)
    assert state.reshuffle_vote_for(0)
    return state


def test_f3_next_effects_are_blanked_after_the_viewers_vote():
    state = _voted_state()
    glob = enc.encode_state(state, 0)[3]
    block = enc.block_slice("next_effects")
    assert not glob[block].any()


def test_f3_plane_18_uses_the_reshuffled_pool_after_a_vote():
    state = _voted_state()
    view = enc._DeckView(state, 0)
    assert view.viewer_voted
    sheet = state.sheet_for(0, 0)
    x, y = next(
        (x, y) for x, size in enumerate(STREET_SIZES) for y in range(size)
        if sheet.numbers[x][y] is None
    )
    _f, _l, low, high = sheet.gap_bounds(x, y)
    expected = 1.0 - dk.two_triple_probability(
        view.reshuffled_matrix,
        enc._interval_miss(low, high),
        enc._interval_miss(low - 2, high + 2),
    )
    planes = enc.encode_state(state, 0)[0]
    assert planes[0, enc.P_FIT_NEXT_TURN, x, y] == np.float32(expected)


# ──────────────────────────────────────────────────────────────────────────
# F6 and the draw kernel
# ──────────────────────────────────────────────────────────────────────────
def _exact_two_triple(cards, miss_non_temp, miss_temp) -> float:
    """Brute force: effects from draws 0-2, numbers from draws 3-5, all distinct."""
    hit = total = 0
    for draw in itertools.permutations(range(len(cards)), 6):
        total += 1
        hit += all(
            (miss_temp if cards[draw[i]][1] == dk._TEMP else miss_non_temp)[
                cards[draw[3 + i]][0]
            ]
            for i in range(3)
        )
    return hit / total


def test_f6_two_triple_matches_the_reviewers_example():
    """Six cards, sheet writable only at 8: the true P(some stack fits) is 0.75;
    the old same-card model said 0.50."""
    e = dk.EFFECT_INDEX
    cards = [(2, e[Effect.TEMP]), (3, e[Effect.TEMP]), (6, e[Effect.PARK]),
             (7, e[Effect.SURVEYOR]), (8, e[Effect.BIS]), (9, e[Effect.ESTATE])]
    fits_temp = [int(any(v == 8 for v in (n, n - 2, n - 1, n + 1, n + 2))) for n in range(1, 16)]
    miss_non = [0 if n == 8 else 1 for n in range(1, 16)]
    miss_temp = [1 - f for f in fits_temp]
    matrix = np.zeros((15, 6))
    for n, eff in cards:
        matrix[n, eff] += 1
    assert 1 - _exact_two_triple(cards, miss_non, miss_temp) == pytest.approx(0.75)
    approx = 1 - dk.two_triple_probability(matrix, miss_non, miss_temp)
    assert abs(approx - 0.75) < 0.03


def test_f6_two_triple_error_is_bounded_on_tiny_pools():
    """Measured 2026-09-25: worst |error| 0.020 over 18 pools of 8-10 cards (the
    worst case -- six of nine cards drawn), 0.0007-0.0055 at 15-81 cards."""
    rng = random.Random(1)
    for _ in range(6):
        cards = [(rng.randrange(15), rng.randrange(6)) for _ in range(8)]
        matrix = np.zeros((15, 6))
        for n, eff in cards:
            matrix[n, eff] += 1
        miss_non = [int(rng.random() < 0.6) for _ in range(15)]
        miss_temp = [m | int(rng.random() < 0.5) for m in miss_non]
        exact = _exact_two_triple(cards, miss_non, miss_temp)
        assert abs(dk.two_triple_probability(matrix, miss_non, miss_temp) - exact) < 0.03


def test_next_draw_probability_equals_the_literal_joint():
    """Throughput #1: the inclusion-exclusion kernel against the K^3 oracle."""
    rng = np.random.default_rng(20260925)
    for case in range(600):
        k = 6 if case % 2 else 15
        d = case % 12  # D = 0, 1, 2 are the reforming cases
        deck = np.bincount(rng.integers(k, size=d), minlength=k).astype(np.float64)
        pool = np.bincount(rng.integers(k, size=40), minlength=k).astype(np.float64)
        masks = rng.integers(2, size=(3, k)).astype(np.float64)
        num, den = dk.ordered_draw_counts(deck, pool)
        assert dk.next_draw_probability(deck, pool, masks) == dk.draw_probability(
            num, den, masks
        )


def test_writable_values_matches_available_locations():
    rng = random.Random(4)
    for seed in range(8):
        state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
        bots = [GreedyBot(random.Random(seed * 3 + i)) for i in range(2)]
        while not state.is_terminal:
            if rng.random() < 0.2:
                sheet = state.sheets[state.actor]
                assert sheet.writable_values() == [
                    bool(sheet.available_locations(v)) for v in range(18)
                ]
            state.apply(bots[state.actor].act(state))


# ──────────────────────────────────────────────────────────────────────────
# §3.2 -- roundabout rescue is about THIS turn
# ──────────────────────────────────────────────────────────────────────────
def test_rescue_is_zero_once_the_viewer_has_declined_the_roundabout():
    for seed in range(300):
        state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
        sheet = state.sheets[0]
        for y in range(STREET_SIZES[0]):
            sheet.write(min(17, y), (0, y), turn=0)
        for y in range(STREET_SIZES[1]):
            sheet.write(min(17, y + 6), (1, y), turn=0)
        # Street 2: one free box hemmed by 16 and 17 -- nothing fits until a
        # roundabout next to it removes a bound.
        for y in range(STREET_SIZES[2]):
            if y != 1:
                sheet.write(16 if y == 0 else 17, (2, y), turn=0)
        sheet.numbers[2][2] = None
        state.public_sheets = [s.copy() for s in state.sheets]
        if not enc._rescue_this_turn(state, 0, 0, sheet):
            continue
        state.apply(ac.A_ROUNDABOUT_OPEN)
        state.apply(ac.A_PASS_ROUNDABOUT)
        assert state.ctx.roundabout_declined
        assert not enc._rescue_this_turn(state, 0, 0, state.sheets[0])
        return
    pytest.skip("no seed produced a rescuable offer on the constructed sheet")


# ──────────────────────────────────────────────────────────────────────────
# §10.6 on the review's own states -- the random-play gate reaches the
# viewer-voted branch only a dozen times in 35 games
# ──────────────────────────────────────────────────────────────────────────
def _rust_matches(state: GameState) -> None:
    wr = pytest.importorskip("welcome_to_rust")
    from games.welcome_to import rust_encode_equiv as eq
    from games.welcome_to import snapshot

    rs = wr.RustGameState.from_snapshot(snapshot.to_snapshot(state))
    eq.compare_state(state, rs, where="review 2026-09-25 case")


def test_rust_matches_python_after_the_viewers_vote():
    state = _voted_state()
    _rust_matches(state)
    # ...and at the viewer's next decision after voting, with the vote queued.
    while state.actor == 0 and not state.is_terminal:
        state.apply(state.legal_actions()[-1])
    _rust_matches(state)


def test_rust_matches_python_on_an_empty_gap():
    state = _state_with_temp_next()
    state.sheets[0].write(7, (0, 0), turn=0)
    state.sheets[0].write(8, (0, 2), turn=0)
    state.public_sheets = [s.copy() for s in state.sheets]
    _rust_matches(state)


def test_rust_matches_python_at_every_house_phase():
    for state in _actor_states(range(3), per_game=4):
        _rust_matches(state)


def test_two_triple_fast_form_equals_its_definition():
    """The three-sum shortcut against `masked_draw_numerator` per sequence."""
    rng = np.random.default_rng(7)
    for _ in range(300):
        matrix = rng.integers(0, 4, size=(15, 6)).astype(np.float64)
        mask_non = [int(v) for v in rng.integers(2, size=15)]
        mask_temp = [int(v) for v in rng.integers(2, size=15)]
        rows = [int(v) for v in matrix.sum(axis=1)]
        total = sum(rows)
        temp = int(matrix[:, dk._TEMP].sum())
        classes = (total - temp, temp)
        masks = (mask_non, mask_temp)
        num = 0
        for seq in itertools.product((0, 1), repeat=3):
            effect_num, used = 1, [0, 0]
            for t in seq:
                effect_num *= classes[t] - used[t]
                used[t] += 1
            if effect_num > 0:
                num += effect_num * dk.masked_draw_numerator(rows, [masks[t] for t in seq])[0]
        expected = dk._probability(num, dk.falling(total, 3) ** 2)
        assert dk.two_triple_probability(matrix, mask_non, mask_temp) == expected


def test_roundabout_writable_masks_equal_a_copied_sheet():
    """Throughput #3's street-local shortcut against the literal construction."""
    rng = random.Random(9)
    checked = 0
    for seed in range(6):
        state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
        bots = [GreedyBot(random.Random(seed * 5 + i)) for i in range(2)]
        while not state.is_terminal:
            sheet = state.sheets[state.actor]
            if rng.random() < 0.15:
                assert sheet.writable_mask() == sum(
                    1 << v for v, ok in enumerate(sheet.writable_values()) if ok
                )
                literal = []
                for pos in sheet.available_locations(None):
                    copy = sheet.copy()
                    copy.build_roundabout(pos, turn=0)
                    literal.append(copy.writable_mask())
                assert sheet.roundabout_writable_masks() == literal
                checked += 1
            state.apply(bots[state.actor].act(state))
    assert checked > 20
