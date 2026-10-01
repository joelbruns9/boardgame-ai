import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from games.advisor.contract import RecommendRequest
from games.cantstop.advisor_adapter import CantStopAdvisor, parse_state
from games.cantstop.engine import Phase, apply_move
from games.cantstop.solver import ProgressHeuristic
from games.cantstop.tests.test_advisor import wire


class Counted(ProgressHeuristic):
    calls = 0

    def evaluate_features(self, features, reference):
        self.calls += 1
        return super().evaluate_features(features, reference)


def request(turn="turn1", table="table1"):
    return RecommendRequest(max_sims=1, options={"table_id": table, "turn_id": turn})


def solve(adapter, state, req=None):
    return adapter.open_search(state, req or request()).advance(1, threading.Event())


def values(result):
    return {k: v.q_value for k, v in result.entries.items()}


def test_later_rolls_equal_fresh_solves_without_evaluation():
    ev = Counted()
    adapter = CantStopAdvisor(evaluator=ev)
    state = parse_state(wire())
    try:
        solve(adapter, state)
        apply_move(state, (7,))
        for phase in [Phase.AWAIT_DECISION, Phase.AWAIT_MOVE]:
            state.phase = phase
            state.dice = (3, 3, 4, 4) if phase == Phase.AWAIT_MOVE else None
            cached = solve(adapter, state)
            fresh = solve(CantStopAdvisor(evaluator=ProgressHeuristic()), state, RecommendRequest())
            assert values(cached) == pytest.approx(values(fresh), abs=1e-12)
            assert "Cached turn" in cached.stop_reason
        assert ev.calls == 1
        assert adapter.turn_cache.stats()["hits"] == 2
    finally:
        adapter.turn_cache.close()


@pytest.mark.parametrize("change", ["turn", "table", "board", "rules", "model", "regression"])
def test_cache_invalidations(change):
    ev = Counted()
    adapter = CantStopAdvisor(evaluator=ev)
    state = parse_state(wire())
    req = request()
    try:
        solve(adapter, state, req)
        if change == "turn": req = request("turn2")
        elif change == "table": req = request(table="table2")
        elif change == "board": state.progress[0][2] = 1
        elif change == "rules":
            payload = wire(); payload["rules"]["blocking"] = False
            state = parse_state(payload)
        elif change == "model": req = RecommendRequest(engine="heuristic", options=req.options)
        elif change == "regression": state.runners[6] -= 1
        assert "Current-turn solve" in solve(adapter, state, req).stop_reason
        assert adapter.turn_cache.stats()["builds"] == 2
    finally:
        adapter.turn_cache.close()


def test_concurrent_requests_build_once_and_legacy_is_uncached():
    ev = Counted()
    adapter = CantStopAdvisor(evaluator=ev)
    state = parse_state(wire())
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: solve(adapter, state), range(3)))
        assert ev.calls == 1
        assert all(values(r) == values(results[0]) for r in results)
        solve(adapter, state, RecommendRequest())
        solve(adapter, state, RecommendRequest())
        assert ev.calls == 3
    finally:
        adapter.turn_cache.close()


def test_cache_capacity_and_expiry():
    adapter = CantStopAdvisor(evaluator=Counted(), cache_entries=1)
    state = parse_state(wire())
    try:
        solve(adapter, state)
        solve(adapter, state, request(table="table2"))
        assert adapter.turn_cache.stats()["entries"] == 1
        with adapter._lock:
            next(iter(adapter.turn_cache.entries.values())).deadline = 0
        assert "Current-turn solve" in solve(adapter, state, request(table="table2")).stop_reason
    finally:
        adapter.turn_cache.close()
    tiny = CantStopAdvisor(evaluator=Counted(), cache_positions=1)
    solve(tiny, state)
    assert tiny.turn_cache.stats()["entries"] == 0


def test_solver_can_be_reused_and_expired_on_other_threads():
    ev = Counted()
    adapter = CantStopAdvisor(evaluator=ev, cache_ttl=0.1)
    state = parse_state(wire())
    solve(adapter, state)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(solve, adapter, state).result()
    assert "Cached turn" in result.stop_reason
    assert ev.calls == 1
    timer = next(iter(adapter.turn_cache.entries.values())).timer
    timer.join(timeout=2)
    assert not timer.is_alive()
    assert adapter.turn_cache.stats()["entries"] == 0

@pytest.mark.parametrize("target", [3, 4, 5])
@pytest.mark.parametrize("blocking", [False, True])
def test_winning_column_recommends_stop_with_certainty(target, blocking):
    from games.cantstop.engine import COLUMN_HEIGHTS
    payload = {"rules": {"num_players": 2, "columns_to_win": target, "blocking": blocking},
               "active_player": 1, "phase": "diceChoice", "progress": [{}, {}],
               "claimed": [[], list(range(2, target+1))],
               "runners": {"8": COLUMN_HEIGHTS[8]-1}, "dice": [4,4,4,4]}
    if target == 4:
        payload["rules"]["num_players"] = 3
        payload["progress"].append({})
        payload["claimed"].append([])
    ev = Counted()
    adapter = CantStopAdvisor(evaluator=ev)
    try:
        state = parse_state(payload)
        result = solve(adapter, state)
        assert values(result) == {"move:8|stop": 1.0}
        actions = adapter.action_views(state)
        assert len(actions) == 1 and actions[0].fields["wins_game"]
        child = parse_state(actions[0].fields["after_move"])
        calls = ev.calls
        result = solve(adapter, child)
        assert values(result) == {"stop": 1.0}
        assert ev.calls == calls
        assert adapter.action_views(child)[0].fields["wins_game"]
        fresh_ev = Counted()
        fresh = CantStopAdvisor(evaluator=fresh_ev)
        assert values(solve(fresh, child)) == {"stop": 1.0}
        assert fresh_ev.calls == 0
    finally:
        adapter.turn_cache.close()


def test_winning_column_still_requires_legal_stop():
    payload = {"rules": {"num_players": 2, "columns_to_win": 5, "blocking": True},
               "active_player": 1, "phase": "continueChoice", "progress": [{"7": 12}, {}],
               "claimed": [[], [2,3,4,5]], "runners": {"8": 11, "7": 12}, "dice": []}
    adapter = CantStopAdvisor(evaluator=Counted())
    state = parse_state(payload)
    assert [a.action_id for a in adapter.action_views(state)] == ["roll"]
    try:
        assert set(solve(adapter, state).entries) == {"roll"}
    finally:
        adapter.turn_cache.close()
