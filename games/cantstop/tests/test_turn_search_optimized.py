"""Exact equivalence to the pre-optimization controller, including tie order."""
from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import pytest

from games.cantstop.advisor_adapter import parse_state
from games.cantstop.engine import Phase
from games.cantstop.rust_solver import hashed_evaluator
from games.cantstop.solver import ProgressHeuristic
from games.cantstop.turn_search import WholeTurnSearch, TurnSearchConfig, _Node
from games.cantstop.tests.turn_search_reference import WholeTurnSearch as Reference
from games.cantstop.tests.test_turn_search import small


@pytest.mark.parametrize('n', [2,3,4])
@pytest.mark.parametrize('k,depth', [(0,1),(1,1),(4,1),(8,2),(12,3)])
@pytest.mark.parametrize('explore', [0,1,4])
def test_identical_branch_sequence_and_values(n, k, depth, explore):
    state = small(n)
    cfg = TurnSearchConfig(expansions=k, depth=depth, explore_every=explore)
    old = Reference(state, hashed_evaluator, cfg)
    new = WholeTurnSearch(state, hashed_evaluator, cfg)
    assert new.trace == old.trace
    for a,b in zip(new.nodes, old.nodes, strict=True):
        np.testing.assert_array_equal(a.values,b.values)
    np.testing.assert_array_equal(new.value(state), old.value(state))
    assert new.choose_move(state) == old.choose_move(state)
    assert new.stats['evaluator_rows'] == old.stats['evaluator_rows']
    assert all(n.reach_evaluations == 0 for n in new.nodes if n.depth == depth)


@pytest.mark.parametrize('k,depth', [(4,1),(8,2)])
def test_real_capture_same_values_with_feature_evaluator(k, depth):
    state = parse_state(json.loads((Path(__file__).parent/'fixtures/bga_923128580_stale_scores.json').read_text()))
    cfg = TurnSearchConfig(expansions=k, depth=depth)
    old = Reference(state, ProgressHeuristic(), cfg)
    new = WholeTurnSearch(state, ProgressHeuristic(), cfg)
    assert new.trace == old.trace
    for a,b in zip(new.nodes, old.nodes, strict=True):
        np.testing.assert_array_equal(a.values,b.values)
    np.testing.assert_array_equal(new.value(state), old.value(state))
    assert new.choose_move(state) == old.choose_move(state)
    assert new.stats['reach_evaluations'] < k*(k+1)//2


@pytest.mark.parametrize('influence', [0.0, 1.0, 1e-200, 2e-323])
@pytest.mark.parametrize('explore', [False, True])
def test_candidate_ties_and_underflow_match_python(influence, explore):
    node = _Node.__new__(_Node)
    node.state = SimpleNamespace(active_player=0)
    node.candidate_tables = 0
    node._reach = np.array([.9,.8,.8,.7,0.0,0.0])
    node.expanded = np.array([True,False,False,False,False,False])
    node.values = np.array([[.5,.5],[.4,.6],[.4,.6],[.8,.2],[.9,.1],[.9,.1]])
    candidates = [(influence*float(w), float(node.values[i,0]), node,i)
                  for i,w in enumerate(node._reach) if not node.expanded[i]]
    expected = max(candidates, key=(lambda c:(c[1],c[0])) if explore else (lambda c:(c[0],c[1])))
    from cantstop_rust import select_leaf_candidate
    actual = select_leaf_candidate(node._reach.tolist(), node.values[:,0].tolist(),
                                   np.flatnonzero(node.expanded).tolist(), influence, explore)
    assert actual == (expected[0], expected[1], expected[3])


def test_stop_roll_change_is_preserved():
    state = small(); state.phase = Phase.AWAIT_DECISION; state.dice = None
    state.claimed_by[6] = None; state.progress[1][10] = 0; state.runners = {10:4}
    cfg = TurnSearchConfig(expansions=8, depth=2)
    old = Reference(state, hashed_evaluator, cfg)
    new = WholeTurnSearch(state, hashed_evaluator, cfg)
    assert new.trace == old.trace
    assert not new.should_stop(state) and not old.should_stop(state)


def test_reach_cache_invalidates_on_backup():
    search = WholeTurnSearch(small(), hashed_evaluator, TurnSearchConfig(expansions=0))
    node = search.root
    a = node.reach(); assert node.reach() is a
    assert node.reach_evaluations == 1
    node.backup()
    b = node.reach(); assert b is not a
    np.testing.assert_array_equal(a,b)
    assert node.reach_evaluations == 2
