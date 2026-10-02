"""Summarise BGA page timing from an advisor "Export capture" file.

The extension's timing probe (extension_cantstop/timing_probe.js) records when
BGA enters each game state (``state``), when board markers/dice change in the
DOM (``board``), when the bridge first sees a new board (``sig``) and when it
reads a settled one (``sent``). This answers the capture-delay question with
measurements:

* how long the board had really been still when we read it (wasted wait),
* how long after BGA *enters* the decision state the board keeps moving
  (the animation tail a state-entry trigger would have to wait out),
* for a candidate rule "read at state entry + X ms": the share of reads that
  would have come after the board's last change (safe), per X.

    python -m games.cantstop.timing_report cantstop-bga-capture.json [more.json]
"""

import argparse
import json
from statistics import median

DECISION_STATES = ("diceChoice", "continueChoice")
CANDIDATE_WAITS_MS = (0, 50, 100, 200, 400)


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def analyse(events):
    """One row per settled read ('sent'), with the timings around it."""
    rows = []
    for i, e in enumerate(events):
        if e["kind"] != "sent":
            continue
        before = events[:i]
        states = [x for x in before if x["kind"] == "state" and x.get("name") == e.get("phase")]
        boards = [x for x in before if x["kind"] == "board"]
        if not states:
            continue
        entered = states[-1]["t"]
        # Board changes belonging to this decision: since the previous read.
        prev_sent = max((x["t"] for x in before if x["kind"] == "sent"), default=-1e18)
        moves = [b["t"] for b in boards if b["t"] > prev_sent]
        last_move = moves[-1] if moves else None
        rows.append({
            "phase": e.get("phase"), "fast": bool(e.get("fast")),
            "trigger": e.get("trigger", "poll"),
            "waited": e.get("waited"),
            "read_after_state_ms": e["t"] - entered,
            "still_before_read_ms": None if last_move is None else e["t"] - last_move,
            "anim_tail_after_state_ms": (None if last_move is None
                                         else max(0.0, last_move - entered)),
        })
    return rows


def summarise(rows):
    out = {"reads": len(rows), "groups": {}, "state_entry_rule": {}}
    for key in sorted({(r["phase"], r["fast"], r["trigger"]) for r in rows}):
        g = [r for r in rows if (r["phase"], r["fast"], r["trigger"]) == key]
        def stats(field):
            xs = [r[field] for r in g if r[field] is not None]
            return {"median": median(xs) if xs else None, "p95": _pct(xs, .95), "n": len(xs)}
        label = "event" if key[2] == "event" else ("fast" if key[1] else "slow")
        out["groups"][f"{key[0]} {label}"] = {
            "reads": len(g),
            "read_after_state_ms": stats("read_after_state_ms"),
            "still_before_read_ms": stats("still_before_read_ms"),
            "anim_tail_after_state_ms": stats("anim_tail_after_state_ms"),
        }
    tails = [r["anim_tail_after_state_ms"] for r in rows if r["anim_tail_after_state_ms"] is not None]
    current = [r["read_after_state_ms"] for r in rows]
    for x in CANDIDATE_WAITS_MS:
        out["state_entry_rule"][f"+{x}ms"] = {
            "safe_share": (sum(t <= x for t in tails) / len(tails)) if tails else None,
            "median_saving_ms": (median([c - x for c in current]) if current else None),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("exports", nargs="+")
    args = ap.parse_args(argv)
    rows = []
    for path in args.exports:
        timing = json.load(open(path, encoding="utf-8")).get("page_timing")
        if not timing:
            raise SystemExit(f"{path}: no page_timing (export from the updated extension)")
        print(f"{path}: hooks {timing['hooks']}, {len(timing['events'])} events")
        rows += analyse(timing["events"])
    print(json.dumps(summarise(rows), indent=1))


if __name__ == "__main__":
    main()
