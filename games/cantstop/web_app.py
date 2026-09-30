"""Serve via python -m uvicorn games.cantstop.web_app:app --port 8765."""
from games.advisor import create_advisor_app
from .advisor_adapter import CantStopAdvisor

app = create_advisor_app(CantStopAdvisor(), title="Can't Stop advisor")
