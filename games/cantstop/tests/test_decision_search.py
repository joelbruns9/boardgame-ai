import json

import numpy as np
import pytest

from games.cantstop.decision_search import (
    Action, TurnTableBackend, actions, force_action, rng_stream,
)
from games.cantstop.decision_compare import load_suite, compare_position, main
from games.cantstop.engine import (
    Phase, apply_move, stop, roll, random_dice, reflect_state, reflect_move,
)
from games.cantstop.snapshot import snapshot, from_snapshot
from games.cantstop.solver import TurnSolver, ProgressHeuristic, runners_key
from games.cantstop.rust_solver import flat_evaluator
from games.cantstop.turn_search import TurnSearchConfig

ROWS = load_suite(split='development')
SMALL = [r for r in ROWS if 'near_claim' in r['tags']]


@pytest.mark.parametrize('row', SMALL, ids=lambda r:r['id'])
def test_all_action_values_match_python_and_zero_budget(row):
    s = from_snapshot(row['snapshot']); before = snapshot(s)
    evaluator = ProgressHeuristic()
    python = TurnSolver(s, evaluator)
    a = TurnTableBackend(evaluator).evaluate(s)
    b = TurnTableBackend(evaluator, search_config=TurnSearchConfig(expansions=0)).evaluate(s)
    assert a == b
    np.testing.assert_array_equal(a.value, python.value(s))
    assert a.selected.move == python.choose_move(s)
    for option in a.options:
        child = force_action(s, option.action)
        np.testing.assert_array_equal(option.value, python.value(child))
        decision = TurnTableBackend(evaluator).evaluate(child)
        key = runners_key(child.runners)
        for choice in decision.options:
            value = (python.stop_values[key] if choice.action.kind == 'stop'
                     else python.roll_values[key])
            np.testing.assert_array_equal(choice.value, value)
    assert snapshot(s) == before


@pytest.mark.parametrize('row', ROWS, ids=lambda r:r['id'])
def test_forced_transitions_match_engine_and_do_not_mutate(row):
    s = from_snapshot(row['snapshot']); before = snapshot(s)
    effects = []
    for a in actions(s):
        actual = force_action(s, a)
        expected = s.clone()
        if a.kind == 'move': apply_move(expected, a.move)
        elif a.kind == 'stop': stop(expected)
        else: expected.phase, expected.dice = Phase.AWAIT_ROLL, None
        assert snapshot(actual) == snapshot(expected)
        if a.kind == 'roll':
            roll(actual, (1,2,3,4)); roll(expected, (1,2,3,4))
            assert snapshot(actual) == snapshot(expected)
        effects.append(repr(snapshot(actual)))
        actual.progress[0][2] = 999
        assert snapshot(s) == before
    assert len(effects) == len(set(effects))
    with pytest.raises(ValueError): force_action(s, Action('invalid'))


def test_terminal_and_fifth_column_are_exact_without_roll():
    row = next(r for r in ROWS if r['id']=='2p5n_winning_bank')
    s = from_snapshot(row['snapshot'])
    assert actions(s) == (Action('stop'),)
    decision = TurnTableBackend(flat_evaluator).evaluate(s)
    assert decision.selected == Action('stop')
    assert decision.value == (1.0,0.0)
    child = force_action(s, Action('stop'))
    assert child.game_over and len(child.claimed_columns(0)) == 5
    def forbidden(_): raise AssertionError('terminal invoked evaluator')
    done = TurnTableBackend(forbidden).evaluate(child)
    assert done.value == (1.0,0.0) and done.selected is None and done.options == ()
    with pytest.raises(ValueError): force_action(s, Action('roll'))
    with pytest.raises(ValueError): force_action(child, Action('roll'))


def test_blocked_stop_and_roll_before_dice():
    row = next(r for r in ROWS if r['id']=='3p4b_three_runners')
    s = from_snapshot(row['snapshot'])
    assert actions(s) == (Action('roll'),)
    with pytest.raises(ValueError): force_action(s, Action('stop'))
    child = force_action(s, Action('roll'))
    assert child.runners == s.runners and child.dice is None
    assert child.active_player == s.active_player and child.phase == Phase.AWAIT_ROLL


@pytest.mark.parametrize('row', SMALL + [r for r in ROWS if 'three_runners' in r['tags']], ids=lambda r:r['id'])
def test_reflection_and_cyclic_seat_symmetry(row):
    s = from_snapshot(row['snapshot']); backend = TurnTableBackend(ProgressHeuristic())
    a = backend.evaluate(s)
    mirrored = backend.evaluate(reflect_state(s))
    np.testing.assert_allclose(mirrored.value, a.value, atol=2e-14, rtol=0)
    original = {o.action:o.value for o in a.options}
    for o in mirrored.options:
        key = Action('move', reflect_move(o.action.move)) if o.action.kind == 'move' else o.action
        np.testing.assert_allclose(o.value, original[key], atol=2e-14, rtol=0)
    # Ties need not reflect to the identical chosen action: compare optimal sets.
    best = max(v[s.active_player] for v in original.values())
    mapped = (Action('move',reflect_move(mirrored.selected.move))
              if mirrored.selected.kind == 'move' else mirrored.selected)
    assert abs(original[mapped][s.active_player]-best)<2e-14
    n=s.rules.num_players; rotated=s.clone()
    rotated.active_player=(s.active_player+1)%n
    rotated.progress=[s.progress[(p-1)%n].copy() for p in range(n)]
    rotated.claimed_by={c:None if p is None else (p+1)%n for c,p in s.claimed_by.items()}
    b=backend.evaluate(rotated)
    np.testing.assert_allclose(b.value,np.roll(a.value,1),atol=2e-14,rtol=0)
    for o in b.options:
        np.testing.assert_allclose(o.value,np.roll(original[o.action],1),atol=2e-14,rtol=0)
    # Only cyclic relabeling preserves the engine's fixed turn order.


def test_ties_and_selected_action_use_existing_solver_order():
    s = from_snapshot(SMALL[0]['snapshot'])
    python = TurnSolver(s,flat_evaluator)
    backend = TurnTableBackend(flat_evaluator)
    result = backend.evaluate(s)
    assert result == backend.evaluate(s)
    assert result.selected.move == python.choose_move(s)
    values=[o.value[s.active_player] for o in result.options]
    assert values == sorted(values,reverse=True)


def test_nonzero_search_and_game_rng_independence():
    s=from_snapshot(SMALL[0]['snapshot'])
    game=rng_stream(17,'game'); expected=rng_stream(17,'game')
    search=rng_stream(17,'search'); training=rng_stream(17,'training')
    assert len({game.state,search.state,training.state})==3
    for _ in range(100): search.next_u64(); training.next_u64()
    cfg=TurnSearchConfig(expansions=1)
    a=TurnTableBackend(ProgressHeuristic(),search_config=cfg).evaluate(s)
    b=TurnTableBackend(ProgressHeuristic(),search_config=cfg).evaluate(s)
    assert a==b
    assert [random_dice(game) for _ in range(20)]==[random_dice(expected) for _ in range(20)]
    assert rng_stream(17,'search',1).state != rng_stream(17,'search',0).state


def test_report_deterministic_and_suite_splits(tmp_path):
    backend=TurnTableBackend(ProgressHeuristic())
    assert compare_position(SMALL[0],backend,backend)==compare_position(SMALL[0],backend,backend)
    heldout=load_suite(split='heldout')
    assert {r['id'] for r in ROWS}.isdisjoint(r['id'] for r in heldout)
    assert {repr(r['snapshot']) for r in ROWS}.isdisjoint(repr(r['snapshot']) for r in heldout)
    assert len({tuple(r['snapshot'][0]) for r in ROWS})==10
    suite=tmp_path/'suite.json'; out=tmp_path/'report.json'
    suite.write_text(json.dumps({'format':'cantstop-search-decisions-v1','positions':[SMALL[0]]}))
    main(['--suite',str(suite),'--out',str(out)])
    report=json.loads(out.read_text())
    assert report['status']=='complete' and report['summary']['identical']==1
    with pytest.raises(SystemExit): main(['--suite',str(suite),'--out',str(out)])


def test_pre_roll_values_and_invalid_domain():
    s = from_snapshot(SMALL[0]['snapshot'])
    s.phase, s.dice = Phase.AWAIT_ROLL, None
    backend = TurnTableBackend(ProgressHeuristic())
    result = backend.evaluate(s)
    assert result.selected == Action('roll')
    assert result.options[0].value == result.value
    np.testing.assert_array_equal(result.value, TurnSolver(s,ProgressHeuristic()).value(s))
    with pytest.raises(ValueError): rng_stream(1,'unknown')
    with pytest.raises(ValueError): rng_stream(1,'search',-1)
