"""Turn a raw :class:`SearchSnapshot` into the ranked response envelope.

Pure and game-agnostic: it joins snapshot entries to their :class:`ActionView`
by ``action_id``, ranks by the AlphaZero criterion (visits, Q as tie-break),
and truncates to ``top_k``.  No search, no game knowledge.
"""

from __future__ import annotations

from .contract import (
    ActionView,
    Recommendation,
    RecommendResponse,
    SearchSnapshot,
)


def build_recommendations(
    snapshot: SearchSnapshot,
    action_views: list[ActionView],
    *,
    top_k: int,
) -> list[Recommendation]:
    """Rank snapshot entries into recommendations, most-visited first."""

    views = {view.action_id: view for view in action_views}
    total_visits = sum(stats.visits for stats in snapshot.entries.values()) or 1

    ranked = sorted(
        snapshot.entries.items(),
        key=lambda item: (item[1].visits, item[1].q_value),
        reverse=True,
    )

    recommendations: list[Recommendation] = []
    for rank, (action_id, stats) in enumerate(ranked[: max(1, top_k)], start=1):
        view = views.get(action_id)
        recommendations.append(
            Recommendation(
                rank=rank,
                action_id=action_id,
                label=view.label if view is not None else action_id,
                kind=view.kind if view is not None else "",
                visits=stats.visits,
                visit_frac=stats.visits / total_visits,
                q_value=stats.q_value,
                prior=stats.prior,
                is_legal=view is not None,
                fields=dict(view.fields) if view is not None else {},
            )
        )
    return recommendations


def response_from_snapshot(
    snapshot: SearchSnapshot,
    action_views: list[ActionView],
    *,
    engine: str,
    top_k: int,
    search_ms: int,
    warnings: list[str] | None = None,
    meta: dict | None = None,
) -> RecommendResponse:
    """Assemble the full wire envelope from a snapshot (no annotators yet)."""

    return RecommendResponse(
        ok=True,
        engine=engine,
        root_value=snapshot.root_value,
        search_ms=search_ms,
        sims_done=snapshot.sims_done,
        sims_target=snapshot.sims_target,
        recommendations=build_recommendations(snapshot, action_views, top_k=top_k),
        warnings=list(warnings or []),
        meta=dict(meta or {}),
    )


def terminal_response(
    *, engine: str, root_value: float, warnings: list[str] | None = None
) -> RecommendResponse:
    """Response for a position with no decision to make (game over / no legal
    actions).  The host emits this without opening a search."""

    return RecommendResponse(
        ok=True,
        engine=engine,
        root_value=root_value,
        search_ms=0,
        sims_done=0,
        sims_target=0,
        recommendations=[],
        warnings=list(warnings or []),
        summary={"terminal": True},
    )
