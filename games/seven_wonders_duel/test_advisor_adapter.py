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


# --- Rust-backed searcher (ADVISOR_RUST_UNIFICATION.md step 4) ---------------


def _swr():
    return pytest.importorskip("seven_wonders_rust")


def _new_adapter():
    """A fresh advisor per test: the Rust handle holds its own arena, so sharing
    the module-scoped fixture across searches would confuse ownership."""
    from .inference import Evaluator
    from .train import build_model

    return SevenWondersAdvisor(
        evaluator=Evaluator(build_model("transformer", 32, 1), "cpu")
    )


def _req(**over):
    from types import SimpleNamespace

    base = dict(
        engine="nn",
        seed=0,
        options={},
        max_sims=99999,
        checkpoint_path=None,
        device="cpu",
        top_k=8,
        temperature=0.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _scraped_state(adapter):
    import json
    from pathlib import Path

    raw = json.loads(
        (Path(__file__).parent / "testdata" / "bga_892846644_greatlibrary.json").read_text(
            encoding="utf-8"
        )
    )
    return adapter.state_from_wire({k: raw[k] for k in ("bga", "args", "dom", "log")})


def test_rust_is_the_default_searcher():
    _swr()
    adapter = _new_adapter()
    handle = adapter.open_search(_scraped_state(adapter), _req())
    try:
        assert type(handle).__name__ == "_RustClosedHandle"
    finally:
        handle.close()


def test_python_searcher_still_selectable():
    adapter = _new_adapter()
    handle = adapter.open_search(
        _scraped_state(adapter), _req(options={"search_impl": "python"})
    )
    try:
        assert type(handle).__name__ == "_ClosedHandle"
    finally:
        handle.close()


def test_explicit_force_expand_falls_back_to_python():
    """The Rust path cannot force-expand the root chance layer (needs the F4.5
    forced-child cache). Honouring the request matters more than using Rust, so
    an explicit ask falls back rather than silently dropping it."""
    _swr()
    adapter = _new_adapter()
    handle = adapter.open_search(
        _scraped_state(adapter), _req(options={"force_expand_root_chance": True})
    )
    try:
        assert type(handle).__name__ == "_ClosedHandle"
    finally:
        handle.close()


def test_rust_and_python_searchers_produce_the_same_snapshot_shape():
    """Plumbing check only.

    Search equivalence itself is gated in test_puct_root, against the one-shot
    Rust search that is in turn gated against Python. Re-asserting a ranking
    here would be both redundant and unreliable: this fixture uses a random
    untrained net, so every action has an almost identical value and the order
    is noise. (Driven by the real checkpoint the two agree exactly -- 2927 vs
    2924 visits on the top action, Q equal to four decimals.)

    What this does check is that both handles expose the same actions, agree
    roughly on the root value, and report the sims they ran.
    """
    import threading

    _swr()
    snapshots = {}
    for impl in ("rust", "python"):
        adapter = _new_adapter()
        handle = adapter.open_search(
            _scraped_state(adapter), _req(options={"search_impl": impl})
        )
        try:
            snapshots[impl] = handle.advance(600, threading.Event())
        finally:
            handle.close()

    rust, python = snapshots["rust"], snapshots["python"]
    assert set(rust.entries) == set(python.entries)
    assert rust.sims_done >= 600 and python.sims_done >= 600
    assert sum(s.visits for s in rust.entries.values()) > 0
    # Both are the same tree over the same position: the root value should be in
    # the same region even under a random net.
    assert rust.root_value == pytest.approx(python.root_value, abs=0.15)
    for stats in rust.entries.values():
        assert -1.0 <= stats.q_value <= 1.0
        assert 0.0 <= stats.prior <= 1.0
