"""7WD advisor adapter against the shared host, with a tiny real net.

Proves the validated seam holds end-to-end: the host drives the Gumbel closed
tree through open_search/advance, ranks by visits, and the wire round-trips.
"""

from __future__ import annotations

import threading

import pytest

from games.advisor import JobManager, RecommendRequest

from .advisor_adapter import SevenWondersAdvisor


@pytest.fixture(scope="module")
def adapter():
    from .inference import Evaluator
    from .train import build_model

    return SevenWondersAdvisor(evaluator=Evaluator(build_model("transformer", 32, 1), "cpu"))


def _pos(adapter, prefix=()):
    return adapter.state_from_wire(
        {"seed": 7, "first_player": 0, "prefix": list(prefix)}
    )


def test_wire_round_trip_is_stable(adapter):
    pos = _pos(adapter)
    public = adapter.state_to_public(pos)
    reparsed = adapter.state_from_wire(
        {"seed": public["seed"], "first_player": public["first_player"], "prefix": public["prefix"]}
    )
    assert adapter.state_key(pos) == adapter.state_key(reparsed)
    assert public["phase"] == "WONDER_DRAFT"
    assert public["actor"] == 0
    assert len(public["legal_actions"]) > 0


def test_action_ids_are_identity_indexed(adapter):
    pos = _pos(adapter)
    from .codec import legal_action_indices

    views = adapter.action_views(pos)
    assert [v.action_id for v in views] == [str(i) for i in legal_action_indices(pos.game)]


def test_blocking_recommend_ranks_by_visits(adapter):
    mgr = JobManager(adapter)
    resp = mgr.run_blocking(
        _pos(adapter), RecommendRequest(engine="auto", max_sims=200, chunk_sims=50, top_k=4, seed=1)
    )
    assert resp.ok
    assert resp.sims_done == 200
    visits = [r.visits for r in resp.recommendations]
    assert visits == sorted(visits, reverse=True)
    assert -1.0 <= resp.root_value <= 1.0
    assert all(-1.0 <= r.q_value <= 1.0 for r in resp.recommendations)


def test_streaming_reaches_target_monotonically(adapter):
    import time

    mgr = JobManager(adapter, chunk_default=40)
    job = mgr.start(
        _pos(adapter), RecommendRequest(engine="auto", max_sims=240, chunk_sims=40, seed=2)
    )
    seen = []
    for _ in range(2000):
        polled = mgr.poll(job.job_id)
        if polled.snapshot is not None:
            seen.append(polled.sims_done)
        if polled.status in ("done", "error", "cancelled"):
            break
        time.sleep(0.003)
    assert polled.status == "done", polled.error
    assert polled.sims_done == 240
    assert all(seen[i] <= seen[i + 1] for i in range(len(seen) - 1))


def test_open_search_runs_no_sims_until_advance(adapter):
    pos = _pos(adapter)
    handle = adapter.open_search(pos, RecommendRequest(engine="auto", max_sims=100, seed=3))
    stop = threading.Event()
    first = handle.advance(0, stop)  # zero-sim advance: read the seeded root only
    assert first.sims_done == 0
    assert all(stats.visits == 0 for stats in first.entries.values())
    handle.close()


def test_unknown_engine_rejected(adapter):
    with pytest.raises(ValueError):
        adapter.open_search(_pos(adapter), RecommendRequest(engine="mystery", max_sims=10))


def test_state_to_public_reports_cities_and_advances(adapter):
    pos = adapter.state_from_wire({"seed": 7, "first_player": 0, "prefix": []})
    public = adapter.state_to_public(pos)
    assert public["origin"] == "replay"
    assert [c["player"] for c in public["cities"]] == [0, 1]
    first = public["legal_actions"][0]["action_id"]
    advanced = adapter.state_to_public(
        adapter.state_from_wire({"seed": 7, "first_player": 0, "prefix": [int(first)]})
    )
    assert advanced["actor"] != public["actor"]  # turn passed


def test_web_app_builds_with_routes_and_static():
    from pathlib import Path

    from . import web_app

    paths = {getattr(r, "path", "") for r in web_app.app.routes}
    assert {"/", "/health", "/api/state", "/api/recommend"} <= paths
    assert {"/api/recommend/start", "/api/recommend/poll", "/api/recommend/stop"} <= paths
    assert (Path(web_app.__file__).with_name("web_static") / "index.html").exists()
