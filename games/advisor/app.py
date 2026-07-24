"""FastAPI host: the transport shell every game's advisor reuses.

``create_advisor_app(adapter)`` returns an app that speaks the stable wire
contract -- a blocking recommend plus the start/poll/stop streaming trio -- and
delegates all game specifics to the :class:`AdvisorAdapter`.  A game's server
is then a two-liner::

    from games.advisor import create_advisor_app
    from games.seven_wonders_duel.advisor_adapter import SevenWondersAdvisor
    app = create_advisor_app(SevenWondersAdvisor())

Run with ``uvicorn games.seven_wonders_duel.web_app:app``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from .contract import AdvisorAdapter, RecommendRequest, RecommendResponse
from .jobs import JobManager, SearchJob

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - runtime setup hint
    raise RuntimeError(
        "Advisor host needs FastAPI. Install with: pip install fastapi uvicorn"
    ) from exc


class RecommendBody(BaseModel):
    """HTTP body for a recommend/start request.

    ``state`` is the game-specific public position the adapter parses; every
    other field maps onto :class:`RecommendRequest`.  ``options`` carries
    per-game knobs untouched by the host.
    """

    state: dict[str, Any]
    engine: str = "auto"
    max_sims: int = Field(default=800, ge=1, le=1_000_000)
    chunk_sims: int = Field(default=200, ge=1, le=100_000)
    top_k: int = Field(default=8, ge=1, le=200)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    checkpoint_path: str | None = None
    device: str = "cpu"
    seed: int = 0
    options: dict[str, Any] = Field(default_factory=dict)

    def to_request(self) -> RecommendRequest:
        return RecommendRequest(
            engine=self.engine,
            max_sims=self.max_sims,
            chunk_sims=self.chunk_sims,
            top_k=self.top_k,
            temperature=self.temperature,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            seed=self.seed,
            options=dict(self.options),
        )


class StateBody(BaseModel):
    state: dict[str, Any]


class StopBody(BaseModel):
    job_id: str


def _response_dict(response: RecommendResponse) -> dict[str, Any]:
    return dataclasses.asdict(response)


def _job_dict(job: SearchJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "version": job.version,
        "sims_done": job.sims_done,
        "sims_target": job.sims_target,
        "error": job.error,
        "snapshot": None if job.snapshot is None else _response_dict(job.snapshot),
    }


def create_advisor_app(
    adapter: AdvisorAdapter,
    *,
    chunk_default: int = 200,
    ttl_secs: float = 60.0,
    title: str | None = None,
    static_dir: str | Path | None = None,
) -> "FastAPI":
    """Build the advisor app for one game adapter.

    ``static_dir`` optionally points at a game's ``web_static/`` folder; when it
    exists it is mounted at ``/static`` and its ``index.html`` served at ``/``,
    so a game's UI is drop-in with no per-game transport code.
    """

    app = FastAPI(title=title or f"{adapter.game_id} advisor", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    manager = JobManager(adapter, chunk_default=chunk_default, ttl_secs=ttl_secs)

    static_path = Path(static_dir) if static_dir is not None else None
    if static_path is not None and static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

        @app.get("/")
        def index() -> "FileResponse":
            return FileResponse(str(static_path / "index.html"))

    def _parse(body: RecommendBody):
        try:
            state = adapter.state_from_wire(body.state)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"bad state: {exc}") from exc
        return state, body.to_request()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "game_id": adapter.game_id,
            "engines": {
                key: dataclasses.asdict(spec)
                for key, spec in dict(adapter.engines()).items()
            },
            "contract": adapter.contract(),
        }

    @app.post("/api/state")
    def state(body: StateBody) -> dict[str, Any]:
        try:
            parsed = adapter.state_from_wire(body.state)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"bad state: {exc}") from exc
        return adapter.state_to_public(parsed)

    @app.post("/api/recommend")
    def recommend(body: RecommendBody) -> dict[str, Any]:
        state, req = _parse(body)
        return _response_dict(manager.run_blocking(state, req))

    @app.post("/api/recommend/start")
    def recommend_start(body: RecommendBody) -> dict[str, Any]:
        state, req = _parse(body)
        return _job_dict(manager.start(state, req))

    @app.get("/api/recommend/poll")
    def recommend_poll(job_id: str) -> dict[str, Any]:
        job = manager.poll(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job_id {job_id!r}")
        return _job_dict(job)

    @app.post("/api/recommend/stop")
    def recommend_stop(body: StopBody) -> dict[str, Any]:
        return {"ok": manager.stop(body.job_id)}

    app.state.advisor_manager = manager  # exposed for tests/inspection
    return app
