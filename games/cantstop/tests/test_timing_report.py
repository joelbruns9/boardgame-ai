"""timing_report: the numbers that size the capture delay."""
from games.cantstop.timing_report import analyse, summarise


def test_tail_and_state_entry_rule():
    events = [
        {"t": 0, "kind": "state", "name": "diceChoice"},
        {"t": 30, "kind": "board"}, {"t": 80, "kind": "board"},
        {"t": 1280, "kind": "sent", "phase": "diceChoice", "fast": False, "waited": 1200},
        {"t": 2000, "kind": "state", "name": "continueChoice"},
        {"t": 2010, "kind": "board"},
        {"t": 2210, "kind": "sent", "phase": "continueChoice", "fast": True, "waited": 200},
    ]
    rows = analyse(events)
    assert [(r["read_after_state_ms"], r["still_before_read_ms"], r["anim_tail_after_state_ms"])
            for r in rows] == [(1280, 1200, 80), (210, 200, 10)]
    rule = summarise(rows)["state_entry_rule"]
    assert rule["+50ms"]["safe_share"] == 0.5 and rule["+100ms"]["safe_share"] == 1.0
    assert summarise(rows)["groups"]["diceChoice slow"]["reads"] == 1
