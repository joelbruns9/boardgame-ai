"""Resumable search jobs: the host half of the :class:`SearchHandle` contract.

A job owns the crank loop for one position.  ``run_blocking`` opens a handle,
advances it to the target, and returns; ``start``/``poll``/``stop`` do the same
work in a background thread so a client watches recommendations sharpen and can
cancel at any time.  Both paths share one worker so blocking and streaming
never diverge.

The manager is game-agnostic -- it drives an :class:`AdvisorAdapter` and never
inspects a state, an action, or a search internal.  Deduplication, TTL
reaping, cancellation, and annotator application live here so every game gets
them for free.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .contract import (
    AdvisorAdapter,
    Annotator,
    RecommendRequest,
    RecommendResponse,
    SearchHandle,
    SearchSnapshot,
)
from .ranking import response_from_snapshot, terminal_response

_ACTIVE_STATUSES = ("queued", "running")


@dataclass
class SearchJob:
    """One in-flight or finished search.  Worker-owned fields (prefixed ``_``)
    never escape through the API; the reaper discards them on cleanup."""

    job_id: str
    state_key: str
    params_key: str
    status: str
    version: int
    sims_done: int
    sims_target: int
    snapshot: RecommendResponse | None
    error: str | None
    started_at: float
    updated_at: float
    #: Last time a CLIENT asked about this job. Distinct from ``updated_at``,
    #: which the worker bumps on every publish and so never goes stale while a
    #: search runs. Only this field detects an abandoned search.
    polled_at: float = 0.0
    #: Streaming jobs are the only ones a client can abandon; a blocking request
    #: holds the caller for its whole duration and must never be idle-reaped.
    streaming: bool = False
    _stop: threading.Event = field(default_factory=threading.Event)
    _future: Future | None = None
    _handle: SearchHandle | None = None
    _state: Any = None
    _req: RecommendRequest | None = None
    _annotation_results: dict[str, AnnotationResult] = field(default_factory=dict)
    _annotation_futures: list[tuple[Future, float]] = field(default_factory=list)
    _annotation_errors: list[str] = field(default_factory=list)
    _annotation_states: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.status in _ACTIVE_STATUSES


class JobManager:
    """Owns *when* search happens; the adapter owns *how*.

    ``chunk_default`` bounds a single ``advance`` and therefore the latency of
    both progress updates and cancellation.  ``ttl_secs`` is how long a
    finished job stays pollable before the reaper drops it.

    ``publish_target_ms`` turns the caller's ``chunk_sims`` into a *floor*
    rather than a fixed step: the loop grows the chunk until a publish costs
    about this long.  Simulations are the wrong unit for a refresh rate --
    their cost varies by position width and by machine, and the caller cannot
    know either.  A 7WD panel polling every 700ms was being served a publish
    every ~54ms, so roughly thirteen of every fourteen were built, ranked and
    thrown away.  Set to 0 to keep the exact chunk the caller asked for.
    """

    def __init__(
        self,
        adapter: AdvisorAdapter,
        *,
        chunk_default: int = 200,
        publish_target_ms: float = 300.0,
        max_chunk: int = 20_000,
        ttl_secs: float = 60.0,
        idle_timeout_secs: float = 120.0,
        sweep_interval_secs: float = 15.0,
        search_workers: int = 4,
        annotator_workers: int = 4,
    ):
        self._adapter = adapter
        self._chunk_default = chunk_default
        self._publish_target_ms = float(publish_target_ms)
        self._max_chunk = int(max_chunk)
        self._ttl_secs = ttl_secs
        self._idle_timeout_secs = idle_timeout_secs
        self._jobs: dict[str, SearchJob] = {}
        self._by_key: dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()
        # POOLS, NOT A THREAD PER JOB. Torch's CPU inference allocates
        # per-thread arenas that it does NOT return when the thread exits --
        # measured at ~8 MB for a thread that only runs forward passes, ~18 MB
        # once a whole search runs on one. An advisor panel starts one job per
        # position (~70 in a game of 7WD), so a thread per job leaked ~1.3 GB
        # per game, unbounded across games, until the machine thrashed. Reusing
        # a fixed set of workers makes that a one-off cost.
        #
        # The two pools are separate because a concurrent annotator must
        # overlap the search that spawned it; sharing one pool would let a
        # saturated search queue starve annotation (and, at one worker,
        # deadlock it).
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(search_workers)),
            thread_name_prefix="advisor-search",
        )
        self._annotator_pool = ThreadPoolExecutor(
            max_workers=max(1, int(annotator_workers)),
            thread_name_prefix="advisor-annotate",
        )
        # A browser tab that closes stops polling and never calls the API again,
        # so reaping only on request would leave the search running forever --
        # observed live as an advisor host burning 2,199 CPU-seconds on a
        # position nobody was looking at. A sweeper is the only thing that can
        # notice the absence of calls.
        self._sweeper = threading.Thread(
            target=self._sweep_forever, args=(sweep_interval_secs,), daemon=True
        )
        self._sweeper.start()

    def _sweep_forever(self, interval: float) -> None:
        while True:
            time.sleep(interval)
            try:
                self._reap()
            except Exception:  # a sweeper that dies silently is worse than noisy
                continue

    # -- keys ---------------------------------------------------------------

    @staticmethod
    def _params_key(req: RecommendRequest) -> str:
        return (
            f"{req.engine}|{req.max_sims}|{req.top_k}|{req.temperature}|"
            f"{req.checkpoint_path}|{req.seed}|{sorted(req.options.items())}"
        )

    # -- blocking path ------------------------------------------------------

    def run_blocking(self, state: Any, req: RecommendRequest) -> RecommendResponse:
        """Open a handle, advance to ``req.max_sims`` in chunks, annotate, and
        return.  Same worker the streaming path uses, minus the background
        thread and the poll surface."""

        views = self._adapter.action_views(state)
        if not views:
            return terminal_response(
                engine=req.engine, root_value=self._root_value_if_known(state)
            )
        job = self._new_job(state, req)
        self._advance_to_target(job)
        if job.error is not None:
            return RecommendResponse(
                ok=False,
                engine=req.engine,
                root_value=0.0,
                search_ms=0,
                sims_done=job.sims_done,
                sims_target=job.sims_target,
                error=job.error,
            )
        return job.snapshot  # type: ignore[return-value]

    # -- streaming path -----------------------------------------------------

    def start(self, state: Any, req: RecommendRequest) -> SearchJob:
        """Begin (or rejoin) a background search.  A live job for the same
        (state, params) is returned as-is instead of spawning a duplicate."""

        self._reap()
        views = self._adapter.action_views(state)
        with self._lock:
            key = (self._adapter.state_key(state), self._params_key(req))
            existing_id = self._by_key.get(key)
            if existing_id is not None:
                existing = self._jobs.get(existing_id)
                if existing is not None and existing.active:
                    return existing
            job = self._new_job(state, req, register_key=key, streaming=True)
            if not views:
                job.snapshot = terminal_response(
                    engine=req.engine, root_value=self._root_value_if_known(state)
                )
                job.status = "done"
                job.updated_at = time.time()
                return job
            job.status = "running"
            job._future = self._pool.submit(self._advance_to_target, job)
            return job

    def poll(self, job_id: str) -> SearchJob | None:
        self._reap()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.polled_at = time.time()  # proof a client still cares
            return job

    def stop(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job._stop.set()
            return True

    # -- worker (shared) ----------------------------------------------------

    def _new_job(
        self,
        state: Any,
        req: RecommendRequest,
        *,
        register_key: tuple[str, str] | None = None,
        streaming: bool = False,
    ) -> SearchJob:
        now = time.time()
        job = SearchJob(
            job_id=uuid.uuid4().hex,
            state_key=self._adapter.state_key(state),
            params_key=self._params_key(req),
            status="queued",
            version=0,
            sims_done=0,
            sims_target=int(req.max_sims),
            snapshot=None,
            error=None,
            started_at=now,
            updated_at=now,
            polled_at=now,
            streaming=streaming,
            _state=state,
            _req=req,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            if register_key is not None:
                self._by_key[register_key] = job.job_id
        return job

    def _advance_to_target(self, job: SearchJob) -> None:
        if job._stop.is_set():
            # Stopped while queued behind another search: opening a handle only
            # to close it again buys nothing.
            job.status = "cancelled"
            job.updated_at = time.time()
            return
        req = job._req
        assert req is not None
        state = job._state
        views = self._adapter.action_views(state)
        target = int(req.max_sims)
        floor = int(req.chunk_sims) or self._chunk_default
        chunk = floor
        started = time.perf_counter()
        handle: SearchHandle | None = None
        try:
            handle = self._adapter.open_search(state, req)
            job._handle = handle
            annotators: list[Annotator] = list(self._adapter.annotators())
            concurrent = [
                annotator
                for annotator in annotators
                if bool(getattr(annotator, "concurrent", False))
            ]
            sequential = [
                annotator
                for annotator in annotators
                if not bool(getattr(annotator, "concurrent", False))
            ]
            self._start_concurrent_annotators(job, concurrent)
            job.status = "running"
            while job.sims_done < target and not job._stop.is_set():
                step = min(chunk, target - job.sims_done)
                chunk_started = time.perf_counter()
                snap = handle.advance(step, job._stop)
                chunk = self._next_chunk(chunk, floor, time.perf_counter() - chunk_started)
                if snap.sims_done <= job.sims_done:
                    # No forward progress (terminal tree / immediate cancel):
                    # publish once and stop rather than spin the budget.
                    self._publish(job, snap, views, started, final=True)
                    break
                self._publish(job, snap, views, started, final=False)
                if snap.partial:
                    break
                if snap.stop_reason:
                    # A resource ceiling, not a cancellation: what is already
                    # published IS the answer, so settle on it rather than
                    # spinning against a handle that will not grow.
                    break
            else:
                # Loop fell through on the target/stop condition with the last
                # published snapshot already final; nothing more to do.
                pass
            self._settle(job, views, started, sequential)
            job.status = "cancelled" if job._stop.is_set() else "done"
        except Exception as exc:  # surfaced to the client, never raised out
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = "error"
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
            job._handle = None
            job.updated_at = time.time()

    def _next_chunk(self, chunk: int, floor: int, elapsed_secs: float) -> int:
        """Grow the chunk toward ``publish_target_ms`` of work per publish.

        Never below what the caller asked for -- that is their granularity for
        cancellation and progress -- and never more than doubling per step, so a
        single fast chunk (a terminal-heavy sample, a cold cache) cannot launch
        the next one into a multi-second stall. Shrinks the same way when a
        chunk overruns.
        """

        if self._publish_target_ms <= 0 or elapsed_secs <= 0:
            return chunk
        scale = (self._publish_target_ms / 1000.0) / elapsed_secs
        scaled = int(chunk * max(0.5, min(2.0, scale)))
        return max(floor, min(self._max_chunk, scaled))

    def _publish(
        self,
        job: SearchJob,
        snapshot: SearchSnapshot,
        views: list,
        started: float,
        *,
        final: bool,
    ) -> None:
        response = response_from_snapshot(
            snapshot,
            views,
            engine=job._req.engine,  # type: ignore[union-attr]
            top_k=job._req.top_k,  # type: ignore[union-attr]
            search_ms=int((time.perf_counter() - started) * 1000),
            # The person waiting is entitled to know the search stopped early
            # and why; silently capping would look like a stalled counter.
            # Adapter-level warnings (an adapter serving a stale checkpoint, say)
            # ride along for the same reason. `warnings()` is optional, so games
            # that have nothing to say need not implement it.
            warnings=(
                ([snapshot.stop_reason] if snapshot.stop_reason else [])
                + list(getattr(self._adapter, "warnings", list)())
            )
            or None,
        )
        with self._lock:
            response = _with_annotation_results(
                response, job._annotation_results.values()
            )
            response = _with_annotation_states(response, job._annotation_states)
            job.sims_done = snapshot.sims_done
            job.snapshot = response
            job.version += 1
            job.updated_at = time.time()

    def _start_concurrent_annotators(
        self, job: SearchJob, annotators: list[Annotator]
    ) -> None:
        """Start position-only analyses beside the main search.

        They receive an empty recommendation list by contract: a concurrent
        annotator must derive its answer from the immutable state. Its result is
        merged immediately when available and is also re-applied by every later
        publish, so a fresh MCTS snapshot cannot erase an early exact solve.
        """

        req = job._req
        assert req is not None
        for annotator in annotators:
            default_budget = float(getattr(annotator, "default_budget_secs", 10.0))
            budget = float(req.options.get("annotate_budget_secs", default_budget))
            if not (budget > 0.0):
                continue
            deadline = time.perf_counter() + budget
            with self._lock:
                job._annotation_states[annotator.name] = {
                    "status": "running",
                    "budget_secs": budget,
                }
            future = self._annotator_pool.submit(
                self._run_concurrent_annotator, job, annotator, deadline
            )
            job._annotation_futures.append((future, deadline))

    def _run_concurrent_annotator(
        self, job: SearchJob, annotator: Annotator, deadline: float
    ) -> None:
        try:
            result = annotator.annotate(
                job._state,
                [],
                job._req,  # type: ignore[arg-type]
                deadline=deadline,
                stop_event=job._stop,
            )
        except Exception as exc:
            with self._lock:
                detail = f"{type(exc).__name__}: {exc}"
                job._annotation_errors.append(f"{annotator.name}: {detail}")
                job._annotation_states[annotator.name] = {
                    "status": "error",
                    "detail": detail,
                }
                self._refresh_annotation_state(job)
            return
        if job._stop.is_set():
            return
        with self._lock:
            if result is None:
                job._annotation_states[annotator.name] = {"status": "no_result"}
                self._refresh_annotation_state(job)
                return
            job._annotation_results[result.name] = result
            final_status = str(result.summary.get("status", "complete"))
            job._annotation_states[annotator.name] = {"status": final_status}
            self._refresh_annotation_state(job, result)

    def _refresh_annotation_state(
        self, job: SearchJob, result: AnnotationResult | None = None
    ) -> None:
        """Publish an annotator lifecycle transition into the live snapshot.

        Caller holds ``self._lock``. A dedicated lifecycle map lets a UI tell
        running/skipped/timed-out apart even when no per-action answer exists.
        """

        if job.snapshot is None:
            return
        response = job.snapshot
        if result is not None:
            response = _with_annotation_results(response, [result])
        job.snapshot = _with_annotation_states(response, job._annotation_states)
        job.version += 1
        job.updated_at = time.time()

    def _wait_for_concurrent_annotators(self, job: SearchJob) -> None:
        """Wait only to each worker's declared deadline, staying cancellable."""

        for future, deadline in job._annotation_futures:
            while not future.done() and not job._stop.is_set():
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    future.result(timeout=min(0.05, remaining))
                except Exception:
                    break

    def _settle(
        self,
        job: SearchJob,
        views: list,
        started: float,
        annotators: list[Annotator],
    ) -> None:
        """Run annotators on the settled recommendations (bounded, cancellable)
        and fold their output into the final snapshot."""

        if job.snapshot is None or job._stop.is_set():
            return
        self._wait_for_concurrent_annotators(job)
        if not annotators:
            return
        response = job.snapshot
        recs = list(response.recommendations)
        summary = dict(response.summary)
        deadline = time.perf_counter() + float(
            job._req.options.get("annotate_budget_secs", 10.0)  # type: ignore[union-attr]
        )
        for annotator in annotators:
            if job._stop.is_set() or time.perf_counter() > deadline:
                break
            result = annotator.annotate(
                job._state,
                recs,
                job._req,  # type: ignore[arg-type]
                deadline=deadline,
                stop_event=job._stop,
            )
            if result is None:
                continue
            recs = [
                _with_annotation(rec, result.name, result.per_action.get(rec.action_id))
                for rec in recs
            ]
            if result.summary:
                summary[result.name] = result.summary
        with self._lock:
            job.snapshot = _replace_recs(response, recs, summary)
            job.version += 1
            job.updated_at = time.time()

    # -- housekeeping -------------------------------------------------------

    def _root_value_if_known(self, state: Any) -> float:
        return 0.0

    def _reap(self) -> None:
        now = time.time()
        with self._lock:
            # Stop abandoned streaming searches. Signalling _stop lets the
            # worker finish its chunk and release the handle; the job is then
            # dropped by the ttl rule below on a later pass.
            for job in self._jobs.values():
                if (
                    job.active
                    and job.streaming
                    and self._idle_timeout_secs > 0
                    and (now - max(job.polled_at, job.started_at))
                    > self._idle_timeout_secs
                ):
                    job.error = job.error or "abandoned: no client poll"
                    job._stop.set()
            drop = [
                job_id
                for job_id, job in self._jobs.items()
                if not job.active and (now - job.updated_at) > self._ttl_secs
            ]
            for job_id in drop:
                job = self._jobs.pop(job_id, None)
                if job is None:
                    continue
                key = (job.state_key, job.params_key)
                # Only if it still points at THIS job: a newer search on the
                # same position has already claimed the key, and dropping it
                # would cost that one its deduplication.
                if self._by_key.get(key) == job_id:
                    self._by_key.pop(key, None)


def _with_annotation(rec, name: str, blob):
    if not blob:
        return rec
    import dataclasses

    merged = dict(rec.annotations)
    merged[name] = blob
    return dataclasses.replace(rec, annotations=merged)


def _replace_recs(response: RecommendResponse, recs: list, summary: dict):
    import dataclasses

    return dataclasses.replace(response, recommendations=recs, summary=summary)


def _with_annotation_results(response: RecommendResponse, results) -> RecommendResponse:
    """Idempotently merge completed analyses into one response snapshot."""

    recs = list(response.recommendations)
    summary = dict(response.summary)
    for result in results:
        recs = [
            _with_annotation(rec, result.name, result.per_action.get(rec.action_id))
            for rec in recs
        ]
        if result.summary:
            summary[result.name] = result.summary
    return _replace_recs(response, recs, summary)


def _with_annotation_states(
    response: RecommendResponse, states: dict[str, dict[str, Any]]
) -> RecommendResponse:
    """Expose concurrent-analysis lifecycle without changing the wire schema."""

    if not states:
        return response
    summary = dict(response.summary)
    summary["annotation_status"] = {
        name: dict(status) for name, status in states.items()
    }
    return _replace_recs(response, list(response.recommendations), summary)
