"""Host mechanics, proven against a fake adapter (no game, no torch).

If these pass, the resumable-search lifecycle, ranking, dedup, cancellation,
terminal handling, and annotators work for *any* AdvisorAdapter -- which is the
whole point of standardizing the host.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from games.advisor import (
    ActionStats,
    ActionView,
    AnnotationResult,
    EngineSpec,
    JobManager,
    RecommendRequest,
    SearchSnapshot,
)


@dataclass
class _FakeState:
    key: str
    n_actions: int
    terminal: bool = False


class _FakeHandle:
    """Deterministic handle: after ``done`` sims, action ``i`` has
    ``done // (i + 1)`` visits -- monotone in ``done`` and strictly ordered so
    action 0 is always the top recommendation."""

    def __init__(self, n: int, target: int):
        self._n = n
        self._target = target
        self._done = 0

    def advance(self, chunk_sims: int, stop_event: threading.Event) -> SearchSnapshot:
        for _ in range(chunk_sims):
            if stop_event.is_set():
                break
            self._done += 1
        entries = {
            str(i): ActionStats(self._done // (i + 1), 0.5 - 0.1 * i, 1.0 / self._n)
            for i in range(self._n)
        }
        return SearchSnapshot(
            sims_done=self._done,
            sims_target=self._target,
            root_value=0.25,
            entries=entries,
            partial=stop_event.is_set(),
        )

    def close(self) -> None:
        pass


class _TagAnnotator:
    name = "tag"

    def annotate(self, state, recommendations, req, *, deadline, stop_event):
        return AnnotationResult(
            name="tag",
            per_action={rec.action_id: {"seen": True} for rec in recommendations},
            summary={"count": len(recommendations)},
        )


class _FakeAdapter:
    game_id = "fake"

    def __init__(self, *, annotators=()):
        self._annotators = list(annotators)

    def state_from_wire(self, payload):
        return _FakeState(payload["key"], int(payload["n"]), payload.get("terminal", False))

    def state_to_public(self, state):
        return {"key": state.key, "n": state.n_actions}

    def state_key(self, state):
        return state.key

    def action_views(self, state):
        if state.terminal:
            return []
        return [ActionView(action_id=str(i), label=f"a{i}") for i in range(state.n_actions)]

    def open_search(self, state, req):
        return _FakeHandle(state.n_actions, req.max_sims)

    def engines(self):
        return {"fake": EngineSpec(key="fake", label="fake")}

    def annotators(self):
        return self._annotators

    def contract(self):
        return {"game_id": self.game_id}


def _drain(manager, job, limit=1000):
    import time

    for _ in range(limit):
        polled = manager.poll(job.job_id)
        if polled.status in ("done", "error", "cancelled"):
            return polled
        time.sleep(0.002)
    raise AssertionError("job never finished")


def test_blocking_reaches_target_and_ranks_by_visits():
    mgr = JobManager(_FakeAdapter())
    state = _FakeState("s1", 4)
    resp = mgr.run_blocking(state, RecommendRequest(max_sims=300, chunk_sims=50, top_k=4))
    assert resp.ok
    assert resp.sims_done == 300
    visits = [r.visits for r in resp.recommendations]
    assert visits == sorted(visits, reverse=True)
    assert resp.recommendations[0].action_id == "0"
    assert abs(sum(r.visit_frac for r in resp.recommendations) - 1.0) < 1e-9
    assert resp.root_value == 0.25  # actor-frame passthrough


def test_streaming_is_monotone_and_reaches_target():
    mgr = JobManager(_FakeAdapter(), chunk_default=25)
    state = _FakeState("s2", 3)
    job = mgr.start(state, RecommendRequest(max_sims=200, chunk_sims=25))
    seen = []
    import time

    for _ in range(1000):
        polled = mgr.poll(job.job_id)
        if polled.snapshot is not None:
            seen.append(polled.sims_done)
        if polled.status in ("done", "error", "cancelled"):
            break
        time.sleep(0.002)
    assert polled.status == "done"
    assert polled.sims_done == 200
    assert all(seen[i] <= seen[i + 1] for i in range(len(seen) - 1))


def test_start_dedups_same_state_and_params():
    mgr = JobManager(_FakeAdapter())
    state = _FakeState("s3", 3)
    req = RecommendRequest(max_sims=100000, chunk_sims=10)
    a = mgr.start(state, req)
    b = mgr.start(state, req)
    assert a.job_id == b.job_id
    mgr.stop(a.job_id)


def test_cancellation_stops_before_target():
    mgr = JobManager(_FakeAdapter(), chunk_default=5)
    state = _FakeState("s4", 3)
    job = mgr.start(state, RecommendRequest(max_sims=1_000_000, chunk_sims=5))
    import time

    time.sleep(0.02)
    assert mgr.stop(job.job_id)
    final = _drain(mgr, job)
    assert final.status == "cancelled"
    assert final.sims_done < 1_000_000


def test_terminal_state_returns_no_recommendations():
    mgr = JobManager(_FakeAdapter())
    state = _FakeState("s5", 0, terminal=True)
    resp = mgr.run_blocking(state, RecommendRequest(max_sims=100))
    assert resp.ok
    assert resp.recommendations == []
    assert resp.summary.get("terminal") is True


def test_annotators_decorate_settled_recommendations():
    mgr = JobManager(_FakeAdapter(annotators=[_TagAnnotator()]))
    state = _FakeState("s6", 3)
    resp = mgr.run_blocking(state, RecommendRequest(max_sims=60, chunk_sims=20, top_k=3))
    assert all(r.annotations.get("tag") == {"seen": True} for r in resp.recommendations)
    assert resp.summary.get("tag") == {"count": 3}
