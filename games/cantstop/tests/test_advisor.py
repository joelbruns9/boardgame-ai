import json
import threading
import pytest

from games.advisor.contract import RecommendRequest
from games.advisor.ranking import build_recommendations
from games.cantstop.advisor_adapter import CantStopAdvisor, parse_state
from games.cantstop.engine import Phase, apply_move, legal_moves
from games.cantstop.solver import ProgressHeuristic
from games.cantstop.rust_solver import RustTurnSolver


def wire(n=3, phase="diceChoice"):
    return {"rules": {"num_players": n, "columns_to_win": 4 if n == 3 else 3, "blocking": True},
            "active_player": n-1, "phase": phase, "progress": [{} for _ in range(n)],
            "claimed": [[] for _ in range(n)], "runners": {"6":10, "7":12, "8":10},
            "dice": [1,2,5,6] if phase == "diceChoice" else []}


@pytest.mark.parametrize("n", [2,3,4])
@pytest.mark.parametrize("phase", ["diceChoice", "continueChoice"])
def test_round_trip_and_solver_agreement(n, phase):
    adapter = CantStopAdvisor(evaluator=ProgressHeuristic())
    state = adapter.state_from_wire(wire(n, phase))
    assert adapter.state_key(state) == adapter.state_key(adapter.state_from_wire(adapter.state_to_public(state)))
    req = RecommendRequest(max_sims=1)
    result = adapter.open_search(state, req).advance(1, threading.Event())
    solver = RustTurnSolver(state, ProgressHeuristic())
    ranked = build_recommendations(result, adapter.action_views(state), top_k=12)
    expected = ("move:"+",".join(map(str, solver.choose_move(state)))) if phase == "diceChoice" else ("stop" if solver.should_stop(state) else "roll")
    assert ranked[0].action_id.split("|")[0] == expected
    assert result.root_value == pytest.approx(2*solver.value(state)[state.active_player]-1)
    assert all(r.visits == 0 for r in ranked)
    assert all(-1 <= r.q_value <= 1 for r in ranked)


def test_blocking_forces_roll():
    payload = wire(phase="continueChoice")
    payload["progress"][0]["6"] = 10
    adapter = CantStopAdvisor()
    assert [a.action_id for a in adapter.action_views(parse_state(payload))] == ["roll"]


@pytest.mark.parametrize("mutate", [
    lambda p: p["rules"].update(blocking="false"),
    lambda p: p.update(active_player=3),
    lambda p: p.update(dice=[0,1,2,3]),
    lambda p: p.update(runners={"6":12}),
    lambda p: p.update(progress=[{}]),
    lambda p: p.update(claimed=[[2],[2],[]]),
    lambda p: p["progress"][2].update({"6":10}),
])
def test_bad_states_rejected(mutate):
    p = wire()
    mutate(p)
    with pytest.raises(ValueError):
        parse_state(p)


def capture():
    return {"format":"bga-cantstop-v1", "playerorder":["10","20","30"],
            "players":{"10":{"color":"ff0000","score":0}, "20":{"color":"0000ff","score":0},
                       "30":{"color":"00ff00","score":0}}, "active_player":"30",
            "required_column_count":4, "blocking":True, "movement_variant_raw":2,
            "phase":"continueChoice", "dice":[],
            "markers":[{"column":6,"height":3,"color":"00ff00"},
                       {"column":6,"height":1,"color":"000000"}]}


def test_bga_runner_absolute_not_increment():
    state = parse_state(capture())
    assert state.progress[2][6] == 8
    assert state.runners[6] == 10
    assert state.active_player == 2


def test_bga_rejects_duplicate_marker():
    c = capture()
    c["markers"].append(dict(c["markers"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        parse_state(c)


def test_bga_no_guessed_movement_flag():
    c = capture()
    del c["blocking"]
    assert parse_state(c).rules.blocking is True


def test_cancel_before_work():
    adapter = CantStopAdvisor()
    stop = threading.Event(); stop.set()
    snap = adapter.open_search(parse_state(wire()), RecommendRequest()).advance(1, stop)
    assert snap.partial and snap.sims_done == 0


def test_shared_http_contract_and_log(tmp_path):
    import socket
    import time
    import urllib.request
    import urllib.error
    import uvicorn
    from games.advisor.app import create_advisor_app
    app = create_advisor_app(CantStopAdvisor(evaluator=ProgressHeuristic()), log_dir=tmp_path)
    sock = socket.socket(); sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    worker = threading.Thread(target=server.run, kwargs={'sockets':[sock]}, daemon=True)
    worker.start()
    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{port}'+path, data=data,
                                     headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    try:
        deadline = time.monotonic()+10
        while not server.started and time.monotonic() < deadline:
            time.sleep(.02)
        assert server.started
        assert call('/health')['game_id'] == 'cantstop'
        r = call('/api/recommend', {'state':wire(), 'max_sims':1})
        assert r['ok'] and r['recommendations'] and r['sims_done'] == 1
        assert all(x['is_legal'] for x in r['recommendations'])
        job = call('/api/recommend/start', {'state':wire(), 'max_sims':1})
        deadline = time.monotonic()+10
        while job['status'] not in ('done','error') and time.monotonic() < deadline:
            time.sleep(.02)
            job = call('/api/recommend/poll?job_id='+job['job_id'])
        assert job['status'] == 'done'
        assert job['snapshot']['recommendations'] == r['recommendations']
        assert call('/api/game_log', {'state':wire(), 'table_id':'test'})['appended']
        assert json.loads((tmp_path/'table_test.jsonl').read_text())['state'] == wire()
        bad = wire(); bad['dice'] = []
        with pytest.raises(urllib.error.HTTPError) as error:
            call('/api/recommend', {'state':bad})
        assert error.value.code == 400
    finally:
        server.should_exit = True
        worker.join(timeout=10)
        sock.close()

@pytest.mark.parametrize("flag,expected", [(0,False),("0",False),(2,True),("2",True)])
def test_bga_variant_detection(flag, expected):
    c = capture()
    c.pop("blocking")
    c["movement_variant_raw"] = flag
    assert parse_state(c).rules.blocking is expected


@pytest.mark.parametrize("flag", [1,"1",None,True,False,3,"",0.0])
def test_bga_unsupported_variants(flag):
    c = capture()
    c.pop("blocking")
    c["movement_variant_raw"] = flag
    with pytest.raises(ValueError):
        parse_state(c)


def test_bga_variant_conflict():
    c = capture()
    c["blocking"] = False
    with pytest.raises(ValueError, match="conflicts"):
        parse_state(c)

def test_move_follow_up_uses_one_solve(monkeypatch):
    import games.cantstop.rust_solver as rust
    original = rust.RustTurnSolver
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(rust, "RustTurnSolver", counted)
    adapter = CantStopAdvisor(evaluator=ProgressHeuristic())
    state = parse_state(wire())
    snap = adapter.open_search(state, RecommendRequest(max_sims=1)).advance(1, threading.Event())
    assert len(calls) == 1
    solver = original(state, ProgressHeuristic())
    for view in adapter.action_views(state):
        child = parse_state(view.fields["after_move"])
        decision = view.fields["decision"]
        expected = "then stop and bank" if decision == "stop" else "then roll again"
        assert snap.entries[view.action_id].follow_up.startswith(expected)
        stop_value, roll_value = solver.stop_roll(child)
        value = stop_value if decision == "stop" else roll_value
        assert snap.entries[view.action_id].q_value == pytest.approx(2*value[state.active_player]-1)
    for move in legal_moves(state, state.dice):
        prefix = "move:" + ",".join(map(str, move))
        assert prefix + "|stop" in snap.entries
        assert prefix + "|roll" in snap.entries


def test_forced_roll_follow_up():
    payload = wire()
    payload["progress"][0]["6"] = 10
    payload["progress"][0]["8"] = 10
    adapter = CantStopAdvisor(evaluator=ProgressHeuristic())
    snap = adapter.open_search(parse_state(payload), RecommendRequest(max_sims=1)).advance(1, threading.Event())
    assert snap.entries["move:7|roll"].follow_up == "then roll again (stopping is blocked)"
    assert "move:7|stop" not in snap.entries


def test_live_capture_stale_metadata_scores():
    from pathlib import Path
    from games.cantstop.engine import legal_moves
    raw = json.loads((Path(__file__).parent / "fixtures/bga_923128580_stale_scores.json").read_text())
    state = parse_state(raw)
    assert set(state.claimed_columns(0)) == {10}
    assert set(state.claimed_columns(1)) == {2}
    assert state.progress[0][7] == 3
    assert state.progress[0][8] == 7
    assert state.progress[1][7] == 8
    assert state.progress[1][9] == 3
    assert set(legal_moves(state, state.dice)) == {(8,), (4, 6)}
    adapter = CantStopAdvisor(evaluator=ProgressHeuristic())
    result = adapter.open_search(state, RecommendRequest(max_sims=1)).advance(1, threading.Event())
    assert set(result.entries) == {"move:8|stop", "move:8|roll", "move:4,6|stop", "move:4,6|roll"}
    assert all(entry.follow_up for entry in result.entries.values())
    for player in raw["players"].values():
        player.pop("score")
    assert adapter.state_key(parse_state(raw)) == adapter.state_key(state)
