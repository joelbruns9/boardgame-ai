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

The checkpoint may also be supplied per-request from the UI.
"""

from __future__ import annotations

import os
from pathlib import Path

from games.advisor import create_advisor_app

from .advisor_adapter import SevenWondersAdvisor

adapter = SevenWondersAdvisor(
    default_checkpoint=os.environ.get("SWD_ADVISOR_CHECKPOINT"),
    device=os.environ.get("SWD_ADVISOR_DEVICE", "cpu"),
)

app = create_advisor_app(
    adapter,
    title="7 Wonders Duel Advisor",
    static_dir=Path(__file__).with_name("web_static"),
)
