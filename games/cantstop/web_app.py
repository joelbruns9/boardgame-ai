"""Serve via python -m uvicorn games.cantstop.web_app:app --port 8765."""
from fastapi import Body, HTTPException

from games.advisor import create_advisor_app
from .advisor_adapter import CantStopAdvisor


def build_app(advisor=None, **kwargs):
    """The shared advisor host plus Can't Stop's own win-probability route."""
    advisor = advisor or CantStopAdvisor()
    app = create_advisor_app(advisor, title="Can't Stop advisor", **kwargs)

    @app.post("/api/cantstop/win_probabilities")
    def win_probabilities(body: dict = Body(...)):
        """Every player's win probability, for the panel's player list."""
        try:
            return advisor.win_probabilities(body["state"], body.get("options"),
                                             body.get("device", "cuda"))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = build_app()
