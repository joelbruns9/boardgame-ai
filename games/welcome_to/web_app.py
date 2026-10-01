"""Local Welcome To advisor server.

Composition only: the shared host (:mod:`games.advisor`) plus the Welcome To
adapter.  No transport, no job lifecycle, no ranking here -- that is the whole
point of the standardization.

Run from the project root::

    pip install fastapi uvicorn
    uvicorn games.welcome_to.web_app:app --port 8001

Environment:

    WTO_ADVISOR_CHECKPOINT   checkpoint served by default (S0 or S2 format)
    WTO_ADVISOR_DEVICE       cpu (default) or cuda
    WTO_ADVISOR_ROUNDABOUT_PASS
                             set to 1 to give the search back the roundabout
                             pass. Off by default, which is parity with
                             training (SEARCH_SPEC §5.1a): the pruned search
                             reads ROUNDABOUT_OPEN as "commit to a roundabout",
                             so it can rank one but cannot advise declining one.
                             Turn it on to ask that question.

``/health`` reports which checkpoint is loaded and whether its plan-outcome
heads are the untrained legacy ones, so "why does every plan read 50%?" is
answerable without reading the server's environment.

**Port 8001, not 8000.** The 7WD advisor owns 8000 and both hosts get left
running; a shared port would silently serve one game's extension from the
other's model.
"""

from __future__ import annotations

import os

from games.advisor import create_advisor_app

from .advisor_adapter import WelcomeToAdvisor


def _flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


adapter = WelcomeToAdvisor(
    default_checkpoint=os.environ.get("WTO_ADVISOR_CHECKPOINT"),
    device=os.environ.get("WTO_ADVISOR_DEVICE", "cpu"),
    prune_roundabout_pass=not _flag("WTO_ADVISOR_ROUNDABOUT_PASS"),
)

app = create_advisor_app(
    adapter,
    title="Welcome To Advisor",
    # A search step here is a leaf evaluation plus a rollout to the next turn
    # boundary, which is far more expensive than a 7WD simulation: measured 29
    # sims/s on CPU. A 200-simulation chunk would be seven seconds between
    # publishes and the panel would look frozen, so the grain starts much finer;
    # the host's own publish-rate governor grows it on a faster machine.
    chunk_default=8,
)
