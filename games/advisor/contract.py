"""Typed request/response contract between the shared advisor host and a game.

The host (``games.advisor``) owns *when* work happens and *what* the wire
surface looks like: HTTP transport, CORS, the start/poll/stop job lifecycle,
progress cadence, ranking, and the response envelope.  A game adapter owns
*how* each thing happens: turning scraped/entered JSON into an engine state,
labeling actions, and running the search.  These objects replace passing an
unstructured dict through the host.

Three adapters, one engine
--------------------------
A game already exposes ``az_loop.core.GameAdapter`` -- the small engine/match
boundary (``new_game``/``step``/``terminal``/``outcome``).  On top of that
same engine a game composes *serving shells*:

    * ``az_loop.contract.LifecycleAdapter`` -- serve the training loop
      (generate / train / evaluate / promote).
    * ``advisor.contract.AdvisorAdapter`` (this file) -- serve a human at a
      board: given a public position, return ranked recommendations that
      refine live as the search deepens.

``AdvisorAdapter`` is the third shell.  It composes with a ``GameAdapter``, it
does not replace it, and the host never interprets cards, tiles, network
heads, or search internals.

The resumable-handle contract
-----------------------------
The evaluator is exposed as a :class:`SearchHandle`, not a one-shot function.
``open_search`` builds a tree and does no work; the host then calls
``advance`` in chunks, publishing each :class:`SearchSnapshot` as the
recommendations sharpen, and can stop at any time via a ``stop_event``.  A
plain blocking request is just ``open -> advance(to max) -> read -> close``;
progressive display, interrupt-on-position-change, and "think more" all fall
out of the same method.  A one-shot function could express none of them.

Dependency direction
---------------------
``games.advisor`` imports nothing game-specific.  A game's
``advisor_adapter.py`` imports *this*.  If the host ever needs to import a
concrete game, the seam has leaked.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

# An engine's internal position object.  Opaque to the host by design; only the
# adapter that produced it ever interprets it.  Aliased for readable signatures.
EngineState = Any


# ─────────────────────────────────────────────────────────────────────────────
# Request
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class RecommendRequest:
    """One recommendation ask.

    ``max_sims`` is the search ceiling; ``chunk_sims`` is how many simulations
    a single ``SearchHandle.advance`` performs.  A blocking one-shot sets
    ``chunk_sims == max_sims``; a streaming job sets a small chunk and polls.

    ``options`` carries game-specific knobs (an exact-solver budget, a
    swindle/annotator toggle, a determinizer setting) so the frozen core never
    has to grow a field per game.  The host passes it through untouched; only
    the adapter reads it.
    """

    engine: str = "auto"
    max_sims: int = 800
    chunk_sims: int = 200
    top_k: int = 8
    temperature: float = 0.0
    checkpoint_path: str | None = None
    device: str = "cuda"
    seed: int = 0
    options: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ActionView:
    """A legal action as the host and UI see it.

    ``action_id`` is the stable join key used everywhere: it ties a snapshot
    entry, a recommendation, and an annotation to the same move, and survives
    across ``legal_actions()`` calls (raw engine action objects do not).  It
    must be reproducible from the public state alone.  ``fields`` holds the
    game-specific payload the UI renders (a placement, a picked tile, a wonder
    id); the host never inspects it.
    """

    action_id: str
    label: str
    kind: str = ""
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionStats:
    """Per-action search statistics, keyed by ``action_id`` in a snapshot."""

    visits: int
    q_value: float  # actor-frame edge in [-1, 1]
    prior: float


# ─────────────────────────────────────────────────────────────────────────────
# Search handle (the resumable evaluator)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SearchSnapshot:
    """The current state of an in-flight search.

    Returned by every ``advance``.  ``entries`` is the raw per-action tree
    readout; the host turns it into ranked :class:`Recommendation` objects.

    All values are **actor-framed**: ``root_value`` and every
    ``ActionStats.q_value`` are the edge for the player to move -- the human
    asking for advice.  The adapter absorbs the player-0<->actor sign flip (it
    is the only layer that knows whose turn it is), so the host renders
    ``(root_value + 1) / 2`` as "your win probability" without ever learning
    the seat.  ``partial`` is set when a chunk stopped early (deadline/cancel).
    """

    sims_done: int
    sims_target: int
    root_value: float  # actor-frame edge in [-1, 1]
    entries: dict[str, ActionStats]
    partial: bool = False


@runtime_checkable
class SearchHandle(Protocol):
    """A resumable search over one fixed position.

    ``open_search`` returns one of these having done *no* search work.  Each
    ``advance`` deepens the *same* tree by ``chunk_sims`` and returns the best
    guess so far; the tree persists between calls, so successive advances add
    work rather than restart it.  ``stop_event`` bounds cancellation latency to
    roughly one chunk.  The host always calls ``close`` exactly once.
    """

    def advance(self, chunk_sims: int, stop_event: threading.Event) -> SearchSnapshot: ...

    def close(self) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# Annotators (optional, bounded deep analyses)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class AnnotationResult:
    """Output of one annotator pass over a settled recommendation list.

    ``per_action`` maps ``action_id`` to a blob merged into that
    recommendation's ``annotations``; ``summary`` is a position-level blob.
    ``partial`` is set when the annotator's own budget ran out before it
    finished (partial analyses are surfaced as partial, never as complete).
    """

    name: str
    per_action: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    partial: bool = False


@runtime_checkable
class Annotator(Protocol):
    """A bounded, cancellable secondary analysis that decorates recommendations.

    This is the plug-in slot for per-game research value -- an exact endgame
    solve, trap/swindle search, a draft-danger matrix -- kept *out* of the
    host.  It runs after the main search settles, under its own deadline, and
    must honor ``stop_event``.  Caching by state key is the annotator's own
    concern.
    """

    name: str

    def annotate(
        self,
        state: EngineState,
        recommendations: Sequence["Recommendation"],
        req: RecommendRequest,
        *,
        deadline: float,
        stop_event: threading.Event,
    ) -> AnnotationResult | None: ...


# ─────────────────────────────────────────────────────────────────────────────
# Response envelope
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Recommendation:
    """One ranked move in a response.

    Built by the host from a :class:`SearchSnapshot` entry plus the matching
    :class:`ActionView`.  ``annotations`` is where annotator ``per_action``
    blobs land, keyed by annotator name.
    """

    rank: int
    action_id: str
    label: str
    kind: str
    visits: int
    visit_frac: float
    q_value: float
    prior: float
    is_legal: bool = True
    fields: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecommendResponse:
    """The stable wire envelope every advisor returns.

    Game-agnostic on purpose: a BGA extension or manual-entry UI renders this
    shape with only the per-action ``fields``/``annotations`` differing by
    game.  ``meta`` carries non-contractual extras (thread info, cache stats)
    that clients may ignore.
    """

    ok: bool
    engine: str
    root_value: float
    search_ms: int
    sims_done: int
    sims_target: int
    recommendations: list[Recommendation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Engine discovery
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class EngineSpec:
    """Metadata for one engine a game exposes (for ``/health`` and the UI).

    Declarative only -- dispatch happens inside ``open_search``.  ``streaming``
    flags engines that meaningfully refine across advances (an NN/MCTS search)
    versus those that settle in one chunk (a heuristic, or an atomic exact
    solve the host still drives through the same handle).
    """

    key: str
    label: str
    description: str = ""
    needs_checkpoint: bool = False
    default_sims: int = 800
    streaming: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# The adapter a game implements
# ─────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class AdvisorAdapter(Protocol):
    """Game-specific operations the advisor host calls.

    The host sequences these; it never interprets their contents.  A game
    implements this in its own package (e.g.
    ``games/seven_wonders_duel/advisor_adapter.py``) and depends on
    ``games.advisor`` -- never the reverse.
    """

    game_id: str

    def state_from_wire(self, payload: dict[str, Any]) -> EngineState:
        """Parse a public position (BGA scrape or manual entry) into an engine
        state.  Must not require hidden information the wire cannot carry."""
        ...

    def state_to_public(self, state: EngineState) -> dict[str, Any]:
        """Serialize an engine state back to the public JSON the UI renders."""
        ...

    def state_key(self, state: EngineState) -> str:
        """Canonical, hashable identity of a position, used by the host to
        deduplicate concurrent jobs and by annotators to key caches."""
        ...

    def action_views(self, state: EngineState) -> list[ActionView]:
        """Legal actions with stable ids and UI labels, in a deterministic
        order."""
        ...

    def engines(self) -> Mapping[str, EngineSpec]:
        """Engines this game offers, keyed by ``EngineSpec.key``.  Used for
        discovery/validation; ``"auto"`` may resolve to one of them."""
        ...

    def open_search(self, state: EngineState, req: RecommendRequest) -> SearchHandle:
        """Build a resumable search over ``state`` for ``req.engine``.  Does no
        search work until the host calls ``advance``."""
        ...

    def annotators(self) -> Sequence[Annotator]:
        """Optional deep-analysis plug-ins, applied after the main search
        settles.  Empty when the game has none."""
        ...

    def contract(self) -> dict[str, Any]:
        """Run provenance for the manifest/health surface (game id, engine
        list, checkpoint identity, versions)."""
        ...
