import numpy as np
import pytest

from games.cantstop.engine import GameState, RuleSet, Phase, COLUMN_HEIGHTS, apply_move
from games.cantstop.lookahead import refine
from games.cantstop.rust_solver import RustTurnSolver, hashed_evaluator
from games.cantstop.solver import ProgressHeuristic, TurnSolver
from games.cantstop.turn_search import TurnSearchConfig, WholeTurnSearch
from games.cantstop.search_arena import play_game, summarize


def small(n=2, move=True):
    rules = RuleSet(n, 5 if n == 2 else (4 if n == 3 else 3), False)
    s = GameState(rules)
    columns = iter(range(2,13))
    for p in range(n):
        for _ in range(rules.columns_to_win-1):
            s.claimed_by[next(columns)] = p
    for c in columns:
        for p in range(n):
            s.progress[p][c] = COLUMN_HEIGHTS[c]-1
    s.active_player = n-1
    if move:
        s.phase = Phase.AWAIT_MOVE
        s.dice = (5,5,6,6)
    return s


@pytest.mark.parametrize('n', [2,3,4])
def test_zero_budget_exactly_matches_existing_solver(n):
    s = small(n)
    a = WholeTurnSearch(s, hashed_evaluator, TurnSearchConfig(expansions=0))
    b = RustTurnSolver(s, hashed_evaluator)
    assert np.array_equal(a.value(s), b.value(s))
    assert a.choose_move(s) == b.choose_move(s)
    child = s.clone(); apply_move(child, a.choose_move(s))
    assert a.should_stop(child) == b.should_stop(child)
    assert a.stats['expansions'] == 0


@pytest.mark.parametrize('n', [2,3,4])
def test_one_expansion_matches_independent_python_reference(n):
    s = small(n)
    expected = TurnSolver(s, hashed_evaluator)
    refine(expected, s, hashed_evaluator, 1, offset=False)
    actual = WholeTurnSearch(s, hashed_evaluator, TurnSearchConfig(expansions=1))
    np.testing.assert_allclose(actual.value(s), expected.value(s), atol=1e-12, rtol=0)
    assert actual.choose_move(s) == expected.choose_move(s)
    assert actual.stats['expansions'] == 1
    # Next turn begins BEFORE its dice; absolute player order is preserved.
    child = actual.nodes[1]
    assert child.state.phase == Phase.AWAIT_ROLL and child.state.dice is None
    assert child.state.active_player == (s.active_player+1) % n


def test_depth_and_upward_propagation():
    s = small()
    a = WholeTurnSearch(s, hashed_evaluator, TurnSearchConfig(expansions=20, depth=2))
    assert a.stats['expansions'] <= 20
    assert a.stats['depth_reached'] == 2
    for node in a.nodes[1:]:
        np.testing.assert_array_equal(node.parent.values[node.leaf], node.value())
    assert all(np.isclose(sum(t['root_value']), 1) for t in a.trace)
    assert all(0 <= x <= 1 for t in a.trace for x in t['root_value'])


def test_soft_time_and_cancellation_leave_valid_root():
    s = small()
    ticks = iter([0, 2, 2])
    a = WholeTurnSearch(s, hashed_evaluator, TurnSearchConfig(seconds=1), clock=lambda:next(ticks))
    assert a.stats['stop_reason'] == 'time_budget' and a.stats['expansions'] == 0
    b = WholeTurnSearch(s, hashed_evaluator, cancelled=lambda:True)
    assert b.stats['stop_reason'] == 'cancelled' and b.stats['expansions'] == 0
    assert a.choose_move(s) == b.choose_move(s)


def test_fixed_budget_deterministic_and_terminal_win():
    s = small()
    cfg = TurnSearchConfig(expansions=4, depth=2)
    a = WholeTurnSearch(s, hashed_evaluator, cfg)
    b = WholeTurnSearch(s, hashed_evaluator, cfg)
    assert a.trace == b.trace
    child = s.clone(); apply_move(child, a.choose_move(s))
    assert a.should_stop(child)
    assert a.value(child)[s.active_player] == 1


@pytest.mark.parametrize('kwargs', [{'expansions':-1},{'depth':0},{'seconds':float('nan')},{'explore_every':-1}])
def test_bad_config(kwargs):
    with pytest.raises(ValueError): TurnSearchConfig(**kwargs)


def test_arena_seating_and_costs():
    records = [play_game(small().rules, ProgressHeuristic(), i, i, TurnSearchConfig(expansions=1), start=small(move=False)) for i in range(2)]
    summary = summarize(records, 2)
    assert summary['games'] == 2
    assert sum(summary['wins']) == 2
    assert [r['games'] for r in summary['by_seat']] == [1,1]
    assert summary['costs']['control']['expansions'] == 0
    assert all(r['challenger_won'] == (r['winner_seat'] == r['challenger_seat']) for r in records)

def test_future_search_changes_current_stop_decision():
    s = small()
    s.phase = Phase.AWAIT_DECISION; s.dice = None
    s.claimed_by[6] = None
    s.progress[1][10] = 0
    s.runners = {10:4}
    plain = WholeTurnSearch(s, hashed_evaluator, TurnSearchConfig(expansions=0))
    deeper = WholeTurnSearch(s, hashed_evaluator, TurnSearchConfig(expansions=8, depth=2))
    assert plain.should_stop(s)
    assert not deeper.should_stop(s)
    assert not np.array_equal(plain.value(s), deeper.value(s))


def test_bad_rebackup_preserves_existing_table():
    s = small()
    a = WholeTurnSearch(s, hashed_evaluator, TurnSearchConfig(expansions=0))
    before = a.value(s).copy()
    with pytest.raises(ValueError):
        a.root.solver.rebackup([[float('nan'), 0]])
    np.testing.assert_array_equal(a.value(s), before)


@pytest.mark.parametrize('n', [2,3,4])
def test_blocking_backup_matches_reference(n):
    s = small(n)
    s.rules = RuleSet(n, s.rules.columns_to_win, True)
    for c, owner in s.claimed_by.items():
        if owner is None:
            for p in range(n):
                s.progress[p][c] = max(0, COLUMN_HEIGHTS[c]-1-p)
    expected = TurnSolver(s, hashed_evaluator)
    refine(expected, s, hashed_evaluator, 1, offset=False)
    actual = WholeTurnSearch(s, hashed_evaluator, TurnSearchConfig(expansions=1))
    np.testing.assert_allclose(actual.value(s), expected.value(s), atol=1e-12, rtol=0)
    assert actual.choose_move(s) == expected.choose_move(s)
