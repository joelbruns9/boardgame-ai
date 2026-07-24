"""Shared, game-agnostic advisor host.

See :mod:`games.advisor.contract` for the seam between this host and a game's
``advisor_adapter.py``.  ``create_advisor_app`` builds the FastAPI transport;
:class:`JobManager` owns the resumable-search lifecycle; :mod:`ranking` turns a
snapshot into the wire envelope.
"""

from __future__ import annotations

from .contract import (
    ActionStats,
    ActionView,
    AdvisorAdapter,
    AnnotationResult,
    Annotator,
    EngineSpec,
    EngineState,
    Recommendation,
    RecommendRequest,
    RecommendResponse,
    SearchHandle,
    SearchSnapshot,
)
from .jobs import JobManager, SearchJob
from .ranking import (
    build_recommendations,
    response_from_snapshot,
    terminal_response,
)

__all__ = [
    "ActionStats",
    "ActionView",
    "AdvisorAdapter",
    "AnnotationResult",
    "Annotator",
    "EngineSpec",
    "EngineState",
    "JobManager",
    "Recommendation",
    "RecommendRequest",
    "RecommendResponse",
    "SearchHandle",
    "SearchJob",
    "SearchSnapshot",
    "build_recommendations",
    "response_from_snapshot",
    "terminal_response",
]


def create_advisor_app(*args, **kwargs):
    """Lazy proxy for :func:`games.advisor.app.create_advisor_app` so importing
    the package does not require FastAPI unless the app is actually built."""

    from .app import create_advisor_app as _factory

    return _factory(*args, **kwargs)
