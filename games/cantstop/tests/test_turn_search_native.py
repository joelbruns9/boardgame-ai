"""Native throughput APIs: layout, cache invalidation and no bulk board export."""
from types import SimpleNamespace
import numpy as np
import pytest
import cantstop_rust

from games.cantstop.tests.test_turn_search import small
from games.cantstop.turn_search import WholeTurnSearch, TurnSearchConfig
from games.cantstop.solver import ProgressHeuristic
from games.cantstop.snapshot import snapshot


@pytest.mark.parametrize('players', [2,3,4])
def test_single_leaf_matches_bulk_export(players):
    solver = cantstop_rust.TurnSolver(snapshot(small(players)))
    expected = solver.leaf_snapshots()
    assert len(expected) == solver.num_leaves
    assert [solver.leaf_snapshot(i) for i in range(solver.num_leaves)] == expected
    for i in [-1, solver.num_leaves]:
        with pytest.raises((ValueError,OverflowError)):
            solver.leaf_snapshot(i)


@pytest.mark.parametrize('byte_input', [False,True])
def test_native_ranking_cache_and_updated_scores(byte_input):
    solver = cantstop_rust.TurnSolver(snapshot(small()))
    values = np.full((solver.num_leaves,2), .5)
    if byte_input: solver.set_leaf_values_bytes(values.astype('<f8').tobytes())
    else: solver.set_leaf_values(values.tolist())
    assert solver.reach_evaluations == 0
    for explore in [False,True]:
        reach = solver.leaf_reach()
        for excluded in [[],[0],list(range(solver.num_leaves))]:
            assert solver.best_candidate(.2,explore,excluded) == cantstop_rust.select_leaf_candidate(
                reach,values[:,1].tolist(),excluded,.2,explore)
    assert solver.reach_evaluations == 1
    values[0] = [.1,.9]
    solver.rebackup(values.tolist())
    assert solver.best_candidate(0,True,[]) == (0.0,.9,0)
    assert solver.reach_evaluations == 2
    before = solver.best_candidate(1,False,[])
    with pytest.raises(ValueError): solver.rebackup([[float('nan'),0]])
    assert solver.best_candidate(1,False,[]) == before
    assert solver.reach_evaluations == 2


def test_feature_path_only_materializes_selected_boards(monkeypatch):
    import games.cantstop.turn_search as module
    selected = []
    class Guard:
        def __init__(self, state): self.inner = cantstop_rust.TurnSolver(state)
        def __getattr__(self, name): return getattr(self.inner,name)
        def leaf_snapshots(self): raise AssertionError('bulk board export on feature path')
        def leaf_snapshot(self,index):
            selected.append(index)
            return self.inner.leaf_snapshot(index)
    monkeypatch.setattr(module, '_rust', lambda:SimpleNamespace(TurnSolver=Guard))
    result = WholeTurnSearch(small(),ProgressHeuristic(),TurnSearchConfig(expansions=8,depth=2))
    assert len(selected) == result.stats['expansions']
    assert all(node.boards is None for node in result.nodes)


@pytest.mark.parametrize('reach,values,excluded,influence', [
    ([.5],[.5,.5],[],1), ([float('nan')],[.5],[],1),
    ([.5],[float('inf')],[],1), ([.5],[.5],[1],1), ([.5],[.5],[],-1)])
def test_native_ranking_rejects_bad_inputs(reach,values,excluded,influence):
    with pytest.raises(ValueError):
        cantstop_rust.select_leaf_candidate(reach,values,excluded,influence,False)
