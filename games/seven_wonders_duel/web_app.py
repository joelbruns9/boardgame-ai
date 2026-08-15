"""Local 7 Wonders Duel advisor server.

Composition only: the shared host (`games.advisor`) + the 7WD adapter + a lab
UI.  This is the two-liner the standardization was for -- no transport, no job
lifecycle, no ranking here.

Run from the project root::

    pip install fastapi uvicorn
    uvicorn games.seven_wonders_duel.web_app:app --reload --port 8000

Then open http://127.0.0.1:8000/ .  Optional environment:

    SWD_ADVISOR_CHECKPOINT   default checkpoint for the "nn" engine
    SWD_ADVISOR_DEVICE       cpu (default) or cuda
    SWD_ADVISOR_ALLOW_MIGRATION
                             serve a checkpoint trained under an older encoder
                             signature (off by default). Every response then
                             carries a warning: the schema shape is unchanged
                             but the meaning of some features moved, so the net
                             is answering off-distribution.
    SWD_ADVISOR_EXACT_ENDGAME
                             run the exact endgame solver at settle (off by
                             default). It costs the annotate budget on any
                             position inside its size gate, and nothing renders
                             its answer yet.

The checkpoint may also be supplied per-request from the UI.
"""

from __future__ import annotations

import os
from pathlib import Path

from games.advisor import create_advisor_app

from .advisor_adapter import SevenWondersAdvisor

def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


adapter = SevenWondersAdvisor(
    default_checkpoint=os.environ.get("SWD_ADVISOR_CHECKPOINT"),
    device=os.environ.get("SWD_ADVISOR_DEVICE", "cpu"),
    allow_encoder_migration=_flag("SWD_ADVISOR_ALLOW_MIGRATION"),
    exact_endgame=_flag("SWD_ADVISOR_EXACT_ENDGAME"),
)

app = create_advisor_app(
    adapter,
    title="7 Wonders Duel Advisor",
    static_dir=Path(__file__).with_name("web_static"),
)
