"""Every net evaluation runs on one long-lived thread.

Torch on CPU keeps ~8 MB per distinct thread that runs a forward pass (8.2 MB
measured for this net, 2026-10-02) and FastAPI retires idle route threads
after 10 s, so evaluating on the route's own thread leaks without bound.
Counts DISTINCT threads that reached the evaluator: counting live threads
proves nothing, since a retired thread is already gone.
"""

import json
import threading
from pathlib import Path

from games.cantstop.advisor_adapter import CantStopAdvisor
from games.cantstop.live_luck import LiveLuck
from games.cantstop.solver import ProgressHeuristic

FIXTURE = Path(__file__).parent / "fixtures" / "bga_packets_925113041.jsonl"


class RecordingEvaluator:
    def __init__(self):
        self.inner, self.threads = ProgressHeuristic(), set()

    def __call__(self, states):
        self.threads.add(threading.get_ident())
        return self.inner(states)


def test_route_threads_never_run_the_net(tmp_path):
    ev = RecordingEvaluator()
    advisor = CantStopAdvisor(evaluator=ev)
    rows = [json.loads(l) for l in FIXTURE.read_text(encoding="utf-8").splitlines()]
    caps = [r["extra"]["capture"] for r in rows if r["kind"] == "decision"]
    (tmp_path / "table_925113041.jsonl").write_text(FIXTURE.read_text(encoding="utf-8"),
                                                    encoding="utf-8")
    live = LiveLuck(advisor, tmp_path)
    calls = [lambda c=c: advisor.win_probabilities(c, None, "cpu") for c in caps[:6]]
    calls.append(lambda: live.summary("925113041", "cpu"))
    for call in calls:                     # each on a fresh, short-lived thread
        t = threading.Thread(target=call)
        t.start()
        t.join()
    assert ev.threads and len(ev.threads) == 1
    assert threading.get_ident() not in ev.threads
