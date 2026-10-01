"""Focused, non-mutating reproduction probes for the 2026-09-25 review.

Run from the repository root: .venv/Scripts/python.exe -m reviews.welcome_to_v3_review_probes
Assertions describe the reviewed implementation's defects, not desired behavior.
"""
import random
from itertools import permutations

import numpy as np

from games.welcome_to import action_codec as ac, encoder as enc, deck_knowledge as dk
from games.welcome_to.bga_extract import state_from_bga_payload, StaleGamedata, _TurnMarks, _CardPool, _deck_from_ledger
from games.welcome_to.constants import CARD_TABLE, Effect, EXTREMITY_POSITIONS, STREET_SIZES
from games.welcome_to.game import GameConfig, GameState, Phase, max_houses_this_turn, _numbers_for
from games.welcome_to.plans import PLANS, PlanKind, feasible, can_be_scored, requirements
from games.welcome_to.sheet import Sheet
from games.welcome_to.tests.plan_reachability import can_ever_be_scored
from games.welcome_to.tests.test_bga_extract import bga_payload, _capture_points
from games.welcome_to.bots import GreedyBot
from games.welcome_to.rust_encoder import encode_state as rust_encode
from games.welcome_to.snapshot import to_snapshot
import welcome_to_rust as wr


def masked_counts(counts, masks):
    """Integer inclusion-exclusion for up to three arbitrary binary masks."""
    n = len(masks)
    total = int(np.sum(counts))
    den = 1
    for i in range(n):
        den *= total - i
    if not n:
        return 1, 1
    s = [int(np.dot(counts, m)) for m in masks]
    if n == 1:
        return s[0], den
    pair = int(np.dot(counts, masks[0] * masks[1]))
    if n == 2:
        return s[0] * s[1] - pair, den
    pair02 = int(np.dot(counts, masks[0] * masks[2]))
    pair12 = int(np.dot(counts, masks[1] * masks[2]))
    triple = int(np.dot(counts, masks[0] * masks[1] * masks[2]))
    return s[0] * s[1] * s[2] - pair * s[2] - pair02 * s[1] - pair12 * s[0] + 2 * triple, den


def check_draw_alternative():
    rng = np.random.default_rng(20260925)
    for case in range(5000):
        k = 6 if case % 2 else 15
        d = case % 82
        deck = np.bincount(rng.integers(k, size=d), minlength=k).astype(np.float64)
        pool = np.bincount(rng.integers(k, size=81-d), minlength=k).astype(np.float64)
        masks = rng.integers(2, size=(3, k)).astype(np.float64)
        num, den = dk.ordered_draw_counts(deck, pool)
        assert np.all(num >= 0) and np.all(num == np.rint(num))
        split = min(d, 3)
        a, da = masked_counts(deck, masks[:split])
        b, db = masked_counts(pool, masks[split:])
        expected = dk.draw_probability(num, den, masks)
        actual = (a * b) / (da * db)
        assert expected == actual, (case, expected, actual)
    print('DRAW_ALTERNATIVE: 5000 exact float matches, including D=0,1,2; nonnegative integer numerators')

    # An illustration of why the two three-card draws cannot be replaced by
    # three cards each supplying both its own number and its own effect.
    cards = [(3, Effect.TEMP), (4, Effect.TEMP), (7, Effect.PARK),
             (8, Effect.SURVEYOR), (9, Effect.BIS), (10, Effect.ESTATE)]
    hits = sum(any(8 in _numbers_for(cards[p[i+3]][0], cards[p[i]][1])
                   for i in range(3)) for p in permutations(range(6)))
    same_card = sum(any(8 in _numbers_for(cards[i][0], cards[i][1]) for i in p)
                    for p in permutations(range(6), 3))
    print('TWO_DRAW_PAIRING:', {'six_draw_fit': hits / 720, 'same_card_shortcut': same_card / 120})


def main():
    # A legal 0 written via TEMP leaves the fenced left extremity unfillable
    # by either a normal write or bis; a roundabout at that box still works.
    sh = Sheet.new()
    sh.write(0, (0, 1), turn=1)
    sh.fences[0][0] = True
    for x, y in EXTREMITY_POSITIONS:
        if (x, y) != (0, 0):
            sh.write(1 if y == 0 else 15, (x, y), turn=1)
    plan = next(p for p in PLANS if p.kind == PlanKind.EXTREMITIES)
    before = feasible(plan, sh)
    after = sh.copy()
    after.build_roundabout((0, 0), turn=2)
    assert not before and can_be_scored(plan, after)
    assert can_ever_be_scored(sh, plan, max_states=2000)
    print('EXTREMITIES:', {'plan': plan.id, 'feasible_before': before,
          'scoreable_after_roundabout': can_be_scored(plan, after), 'oracle': True})

    # Find an ordinary initial offer with TEMP printed on a top card.
    state = next(GameState.new(seed=s, config=GameConfig(players=2)) for s in range(50)
                 if Effect.TEMP in GameState.new(seed=s, config=GameConfig(players=2)).next_effects(0))
    state.sheets[0].write(7, (0, 0), turn=0)
    state.sheets[0].write(8, (0, 2), turn=0)
    py = enc.encode_state(state, 0)
    rs = rust_encode(wr.RustGameState.from_snapshot(to_snapshot(state)), 0)
    assert all(np.array_equal(a, b) for a, b in zip(py, rs))
    legal = [n for n in range(18) if (0, 1) in state.sheets[0].available_locations(n)]
    assert not legal and py[0][0, enc.P_FIT_TEMP, 0, 1] > 0 and py[0][0, enc.P_FIT_NEXT_TURN, 0, 1] > 0
    print('EMPTY_INTEGER_GAP:', {'legal_values': legal, 'fit_temp': float(py[0][0, enc.P_FIT_TEMP, 0, 1]),
          'fit_next_turn': float(py[0][0, enc.P_FIT_NEXT_TURN, 0, 1]), 'rust_equal': True})

    sh = Sheet.new()
    sh.pools[1] = 2
    for y in range(STREET_SIZES[1]):
        sh.write(y + 1, (1, y), turn=0)
    plan = PLANS[27]
    req = requirements(plan, sh)
    assert not feasible(plan, sh) and req.pools_needed[1] == 0 and req.parks_needed[1] == 0
    print('DEAD_REQUIREMENTS:', {'plan': plan.id, 'pools_needed': req.pools_needed,
          'parks_needed': req.parks_needed, 'actual_pool_shortfall': 1})

    # At ACTION_BIS only the optional bis remains, not a fresh whole turn.
    state = next(GameState.new(seed=s, config=GameConfig(players=2)) for s in range(50)
                 if any(e == Effect.BIS for _, e in GameState.new(seed=s, config=GameConfig(players=2)).visible_cards(0)))
    slot = next(i for i, (_, e) in enumerate(state.visible_cards(0)) if e == Effect.BIS)
    state.apply(ac.choose_stack(slot))
    state.apply(state.legal_actions()[0])
    assert state.phase is Phase.ACTION_BIS
    print('MID_TURN_HOUSE_CEILING:', {'phase': state.phase.name,
          'feature': max_houses_this_turn(state, 0, 0), 'remaining_legal_max': 1})
    safe_before = max_houses_this_turn(state, 0, 1)
    for x, size in enumerate(STREET_SIZES):
        for y in range(size):
            state.sheets[1].write(y, (x, y), turn=state.turn)
    assert max_houses_this_turn(state, 0, 1) == safe_before
    print('OPPONENT_HOUSE_CEILING_LEAK: unchanged after hidden live-sheet mutation')

    # scorePlan updates a DOM stamp but leaves gamedatas.planValidations stale.
    state = None
    for seed in range(1, 12):
        state = next((s for s in _capture_points(seed=seed, players=2, advanced=False, limit=600,
                     bot=GreedyBot(rng=random.Random(seed)))
                      if s.phase is Phase.ASK_RESHUFFLE), None)
        if state is not None:
            break
    assert state is not None
    payload = bga_payload(state)
    for slot in payload['bga']['planValidations']:
        for pid in list(slot):
            if slot[pid]['turn'] == state.turn:
                del slot[pid]
    try:
        rebuilt, _, _ = state_from_bga_payload(payload)
        print('STALE_PLAN_VALIDATION:', {'unexpected_phase': rebuilt.phase.name})
    except StaleGamedata as exc:
        print('STALE_PLAN_VALIDATION:', str(exc))

    state = next(s for s in _capture_points(seed=6, players=2, advanced=False, limit=200)
                 if s.phase is Phase.ACTION_SURVEYOR)
    payload = bga_payload(state)
    wire_sheet = payload['bga']['players']['100']['scoreSheet']
    wire_sheet['houses'].extend(payload['dom']['houses'])
    wire_sheet['scribbles'].extend(payload['dom']['scribbles'])
    try:
        rebuilt, _, _ = state_from_bga_payload(payload)
        print('RELOAD_MID_TURN:', {'unexpected_phase': rebuilt.phase.name})
    except StaleGamedata as exc:
        print('RELOAD_MID_TURN:', str(exc))

    # Two adjacent estates are distinct because of their fence, despite one
    # continuous top-fence run after handing both over to a City Plan.
    state = GameState.new(seed=0, config=GameConfig(players=2))
    state.ctx.pending_sizes = [3, 3, 3]
    for y in range(9):
        state.sheets[0].write(y + 1, (0, y), turn=0)
    for y in (2, 5, 8):
        state.sheets[0].fences[0][y] = True
    assert can_be_scored(PLANS[2], state.sheets[0])
    top = [{'type': 'top-fence', 'x': 0, 'y': y, 'turn': 1} for y in range(9)]
    obs = {'turn': 1, 'my_turn_marks': {'houses': [], 'scribbles': top}}
    marks = _TurnMarks(obs, Phase.ASK_RESHUFFLE)
    try:
        marks._validate_action(state)
        raise AssertionError('Expected adjacent estate replay to fail')
    except StaleGamedata as exc:
        print('ADJACENT_ESTATES:', str(exc))

    # On a natural reform the JS DOM history still contains the old cards.
    # mergeLedger clears storage then scans that full history again.
    state = GameState.new(seed=10, config=GameConfig(players=2))
    state.discard = list(state.deck[state.deck_pos:])
    state.deck = []
    state.deck_pos = 0
    state._begin_turn()
    pool = _CardPool()
    table_faces = [tuple(map(int, CARD_TABLE[c])) for c in state.table_cards(0)]
    for face in table_faces:
        pool.take(*face)
    _, deck, warning = _deck_from_ledger(pool, seen=[tuple(map(int, c)) for c in CARD_TABLE[:81]],
                                       table_faces=table_faces, cards_left=state.deck_remaining,
                                       rng=random.Random(0))
    assert len(deck) != state.deck_remaining
    print('NATURAL_REFORM_LEDGER:', {'actual_deck': state.deck_remaining,
          'reconstructed_deck': len(deck), 'warning': warning})

    # Isolate vote reconstruction from the stale-validation bug: supply the
    # current validations, as though capture of the DOM plan stamps was fixed.
    state = GameState.new(seed=0, config=GameConfig(players=2, advanced=True))
    state.plan_ids = (21, 23, 14)
    state.sheets[0].temps = 7
    state.sheets[0].parks = [3, 4, 0]
    state.public_sheets = [s.copy() for s in state.sheets]
    state.apply(ac.choose_stack(state.playable_slots()[0]))
    state.apply(state.legal_actions()[0])
    if state.phase is not Phase.CHOOSE_PLAN:
        state.apply(state.legal_actions()[-1])
    assert state.phase is Phase.CHOOSE_PLAN
    state.apply(ac.choose_plan(0))
    assert state.phase is Phase.ASK_RESHUFFLE
    state.apply(ac.A_RESHUFFLE_YES)
    assert state.phase is Phase.CHOOSE_PLAN and state.reshuffle_vote_for(0)
    rebuilt, _, _ = state_from_bga_payload(bga_payload(state))
    assert not rebuilt.reshuffle_vote_for(0)
    print('LOST_VIEWER_VOTE:', {'actual': state.reshuffle_vote_for(0),
          'reconstructed': rebuilt.reshuffle_vote_for(0), 'phase': state.phase.name})
    check_draw_alternative()


if __name__ == '__main__':
    main()
