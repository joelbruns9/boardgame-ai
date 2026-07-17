import threading

import pytest

from games.kingdomino.game import GameState
from games.kingdomino import web_app
from games.kingdomino.test_web_app_exact_advisor import _forced_discard_deck4_state


@pytest.fixture(autouse=True)
def clean_search_jobs():
    with web_app._SEARCH_JOBS_LOCK:
        for job in web_app._SEARCH_JOBS.values():
            job._stop.set()
        web_app._SEARCH_JOBS.clear()
        web_app._STATE_TO_JOB.clear()
    yield
    with web_app._SEARCH_JOBS_LOCK:
        for job in web_app._SEARCH_JOBS.values():
            job._stop.set()
        web_app._SEARCH_JOBS.clear()
        web_app._STATE_TO_JOB.clear()


def _request(state):
    return web_app.RecommendStartRequest(
        state=web_app.state_to_debug_json(state),
        engine="nn",
        nn_sims=20,
        max_sims=20,
        chunk_sims=10,
        fragility_at_sims=0,
    )


def test_start_poll_done_and_dedupe(monkeypatch):
    state = GameState.new(seed=7)
    entered = threading.Event()
    release = threading.Event()

    def fake_worker(job):
        entered.set()
        release.wait(timeout=2)
        if job._stop.is_set():
            web_app._set_job_status(job, "cancelled")
            return
        snapshot = {
            "ok": True,
            "engine": "nn-mcts-rust",
            "checkpoint_path": "fake.pt",
            "value": 0.1,
            "root_win_prob": 0.55,
            "root_value_player0": 0.1,
            "root_inference": {},
            "search_ms": 1,
            "num_simulations": 20,
            "total_visits": 20.0,
            "draft_matrix": None,
            "recommendations": [],
        }
        web_app._publish_job_snapshot(job, snapshot, 20)
        web_app._set_job_status(job, "done", bump=True)

    monkeypatch.setattr(web_app, "_run_search_job", fake_worker)
    first = web_app.recommend_start(_request(state))
    assert entered.wait(timeout=1)
    second = web_app.recommend_start(_request(state))
    assert second["job_id"] == first["job_id"]

    initial = web_app.recommend_poll(first["job_id"], -1)
    assert initial["status"] == "running"
    assert initial["version"] == 0
    release.set()
    job = web_app._SEARCH_JOBS[first["job_id"]]
    job._thread.join(timeout=2)

    done = web_app.recommend_poll(first["job_id"], -1)
    assert done["status"] == "done"
    assert done["sims_done"] == done["sims_target"] == 20
    assert done["draft_matrix"] is None
    assert "recommendations" in done
    unchanged = web_app.recommend_poll(first["job_id"], done["version"])
    assert unchanged["changed"] is False
    assert "recommendations" not in unchanged


def test_stop_cancels_running_job(monkeypatch):
    state = GameState.new(seed=8)
    entered = threading.Event()

    def fake_worker(job):
        entered.set()
        job._stop.wait(timeout=2)
        web_app._set_job_status(job, "cancelled")

    monkeypatch.setattr(web_app, "_run_search_job", fake_worker)
    started = web_app.recommend_start(_request(state))
    assert entered.wait(timeout=1)
    stopped = web_app.recommend_stop(web_app.RecommendStopRequest(job_id=started["job_id"]))
    assert stopped["status"] == "cancelled"
    assert web_app._SEARCH_JOBS[started["job_id"]]._stop.is_set()


def test_dedupe_includes_complete_start_parameters(monkeypatch):
    state = GameState.new(seed=81)
    entered = threading.Event()

    def fake_worker(job):
        entered.set()
        job._stop.wait(timeout=2)
        web_app._set_job_status(job, "cancelled")

    monkeypatch.setattr(web_app, "_run_search_job", fake_worker)
    first = web_app.recommend_start(_request(state))
    assert entered.wait(timeout=1)
    deeper = _request(state)
    deeper.max_sims = 30
    second = web_app.recommend_start(deeper)

    assert second["job_id"] != first["job_id"]
    assert web_app.recommend_poll(first["job_id"], -1)["status"] == "cancelled"


def test_exact_eligible_job_publishes_one_solved_snapshot(monkeypatch):
    state = _forced_discard_deck4_state()
    snapshot = {
        "ok": True,
        "engine": "exact",
        "value": 0.5,
        "root_win_prob": 0.75,
        "root_margin_pts": 3.0,
        "search_ms": 4,
        "num_simulations": 0,
        "recommendations": [{"rank": 1, "action_id": "exact-1"}],
        "exact": {"solved": True, "deck_count": 4},
    }
    monkeypatch.setattr(web_app, "_recommend_exact", lambda *args, **kwargs: snapshot)

    started = web_app.recommend_start(_request(state))
    job = web_app._SEARCH_JOBS[started["job_id"]]
    job._thread.join(timeout=2)
    done = web_app.recommend_poll(job.job_id, -1)

    assert done["status"] == "done"
    assert done["version"] == 1
    assert done["sims_done"] == 0
    assert done["engine"] == "exact"
    assert done["exact"]["solved"] is True
    assert done["recommendations"][0]["action_id"] == "exact-1"


def test_exact_timeout_continues_same_job_with_nn(monkeypatch):
    state = _forced_discard_deck4_state()
    disabled_exact_hooks = []

    def timeout(*args, **kwargs):
        raise web_app.ExactTimeout("exact budget exhausted")

    def fake_python_search(state, evaluator, req, sims, *, disable_exact_endgame=False):
        disabled_exact_hooks.append(disable_exact_endgame)
        action = state.legal_actions()[0]
        return {action: float(sims)}, 0.2, {action: (0.4, 0.6)}

    monkeypatch.setattr(web_app, "_recommend_exact", timeout)
    monkeypatch.setattr(web_app, "_load_nn_evaluator", lambda req: (object(), object(), "fake.pt"))
    monkeypatch.setattr(web_app, "_root_trajectory", lambda *args: {})
    monkeypatch.setattr(web_app, "_python_stream_search", fake_python_search)

    started = web_app.recommend_start(_request(state))
    job = web_app._SEARCH_JOBS[started["job_id"]]
    job._thread.join(timeout=2)
    done = web_app.recommend_poll(job.job_id, -1)

    assert done["job_id"] == started["job_id"]
    assert done["status"] == "done"
    assert done["engine"] == "nn-mcts"
    assert done["sims_done"] == 20
    assert done["exact_fallback"] is True
    assert done["reason"] == "exact budget exhausted"
    assert disabled_exact_hooks == [True, True]


def test_exact_job_cancels_between_bounded_solver_calls(monkeypatch):
    state = _forced_discard_deck4_state()
    second_solve_entered = threading.Event()
    release_second_solve = threading.Event()
    calls = 0

    def fake_exact_margin(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            second_solve_entered.set()
            release_second_solve.wait(timeout=2)
        return 0, False, True

    monkeypatch.setattr(web_app, "_cached_exact_margin", fake_exact_margin)
    started = web_app.recommend_start(_request(state))
    assert second_solve_entered.wait(timeout=1)
    stopped = web_app.recommend_stop(web_app.RecommendStopRequest(job_id=started["job_id"]))
    release_second_solve.set()
    job = web_app._SEARCH_JOBS[started["job_id"]]
    job._thread.join(timeout=2)

    assert stopped["status"] == "cancelled"
    assert web_app.recommend_poll(job.job_id, -1)["status"] == "cancelled"
    assert calls == 2


def test_rust_handle_accumulates_visits():
    np = pytest.importorskip("numpy")
    rust = pytest.importorskip("kingdomino_rust")
    if not hasattr(rust, "AdvisorSearchHandle"):
        pytest.skip("Rust extension has not been rebuilt with the streaming handle")
    from games.kingdomino.endgame_solver import _rust_state_from_python

    class UniformEvaluator:
        def __call__(self, boards, opponents, flat, legal_indices):
            del opponents, flat
            return (
                np.zeros((boards.shape[0],), dtype=np.float32),
                [np.zeros((len(indices),), dtype=np.float32) for indices in legal_indices],
            )

    state = GameState.new(seed=11)
    handle = rust.AdvisorSearchHandle(
        _rust_state_from_python(state), UniformEvaluator(),
        dirichlet_eps=0.0, seed=11, leaf_batch=4, alpha=0.5,
    )
    first, _value1 = handle.advance(8)
    second, _value2 = handle.advance(8)
    visits1 = sum(row[1] for row in first)
    visits2 = sum(row[1] for row in second)
    assert visits1 == 8
    assert visits2 == 16
    assert handle.sims_done == 16
    assert handle.transpositions >= 1
