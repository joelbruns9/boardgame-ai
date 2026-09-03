"""Merge the sharded triage summaries into one corpus-level liveness report.

`threat_corpus_measure.py --triage --shard I/N` writes one summary per shard, so
the corpus-level question -- what FRACTION of each (threat class, chain distance)
cell carries a decision at all -- can only be answered after the shards are
joined. That fraction is the whole point of triaging in full: the eleven measured
positions in the plan are a live subset of a pre-triage sample, so their regret
magnitudes are quotable and their RATE is not.

Reads the shard summaries if they exist and falls back to the per-position
artifacts on disk, so it reports honestly on a partial run -- a shard still
working, or one that died overnight, shows as missing coverage rather than
silently shrinking the denominator.

    python -m games.seven_wonders_duel.threat_corpus_triage_report

Writes `triage_report.json` beside the shard summaries and prints the table.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = "runs/seven_wonders_duel/threat_corpus/measured/triage"
DEFAULT_EPISODES = "runs/seven_wonders_duel/threat_corpus/episodes.json"


def corpus_rows(episodes_path: Path) -> list[dict]:
    corpus = json.loads(episodes_path.read_text(encoding="utf-8"))
    return [
        {
            "episode_id": episode["episode_id"], "table": episode["table"],
            "threat": episode["threat"], "distance": snapshot["distance"],
            "decision_row": snapshot["decision_row"],
            "target_card": episode.get("target_card"),
        }
        for episode in corpus["episodes"]
        for snapshot in episode["snapshots"]
    ]


def verdict_from_artifact(path: Path, args) -> dict | None:
    """Re-derive a verdict from a position artifact.

    The shard summaries are the primary source; this is the fallback for a
    position whose shard has not written its summary yet, and the reason a
    partial run still reports."""

    try:
        actions = json.loads(path.read_text(encoding="utf-8"))
        actions = actions["reference_values"]["actions"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None                      # a file being written right now
    if not actions:
        return None
    best = actions[0]["win_pct_weighted"]
    worst = actions[-1]["win_pct_weighted"]
    spread = best - worst
    live = (
        spread >= args.min_spread
        and args.live_floor <= best <= args.live_ceiling
        and len(actions) > 1
    )
    return {
        "live": live,
        "reason": (
            "live" if live
            else "no_spread" if spread < args.min_spread
            else "decided_lost" if best < args.live_floor
            else "decided_won"
        ),
        "best_pct": best, "worst_pct": worst, "spread": round(spread, 2),
        "legal_actions": len(actions),
        "source": "artifact",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--triage-dir", default=DEFAULT_DIR)
    parser.add_argument("--episodes", default=DEFAULT_EPISODES)
    parser.add_argument("--min-spread", type=float, default=3.0)
    parser.add_argument("--live-floor", type=float, default=10.0)
    parser.add_argument("--live-ceiling", type=float, default=90.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    triage_dir = Path(args.triage_dir)
    if not triage_dir.is_absolute():
        triage_dir = REPO_ROOT / triage_dir
    episodes_path = Path(args.episodes)
    if not episodes_path.is_absolute():
        episodes_path = REPO_ROOT / episodes_path

    rows = corpus_rows(episodes_path)
    verdicts: dict[tuple, dict] = {}

    # Shard summaries first -- they carry the timing and the failure records.
    shards = {}
    for summary_path in sorted(triage_dir.glob("summary_shard*.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shards[summary_path.name] = {
            "shard": summary["params"].get("shard"),
            "totals": summary["totals"],
        }
        for entry in summary["positions"]:
            if entry.get("status") == "ok":
                verdicts[(entry["episode_id"], entry["decision_row"])] = {
                    **entry, "source": "summary"
                }

    # Then any artifact not covered by a summary: a shard mid-flight.
    for row in rows:
        key = (row["episode_id"], row["decision_row"])
        if key in verdicts:
            continue
        artifact = triage_dir / f"{row['episode_id']}_r{row['decision_row']}.json"
        if artifact.exists():
            got = verdict_from_artifact(artifact, args)
            if got:
                verdicts[key] = {**row, **got}

    cells = defaultdict(lambda: {"total": 0, "triaged": 0, "live": 0,
                                 "reasons": Counter()})
    for row in rows:
        cell = cells[(row["threat"], row["distance"])]
        cell["total"] += 1
        got = verdicts.get((row["episode_id"], row["decision_row"]))
        if got:
            cell["triaged"] += 1
            cell["live"] += bool(got["live"])
            cell["reasons"][got["reason"]] += 1

    print(f"corpus: {len(rows)} snapshots, {len(verdicts)} triaged "
          f"({len(verdicts) / len(rows):.0%})\n")
    header = f"{'threat':<16}{'d':>2}{'total':>7}{'triaged':>9}{'live':>6}{'rate':>7}  reasons"
    print(header)
    print("-" * len(header))
    for (threat, distance), cell in sorted(cells.items()):
        rate = (f"{cell['live'] / cell['triaged']:.0%}"
                if cell["triaged"] else "--")
        reasons = " ".join(
            f"{name}={count}" for name, count in sorted(cell["reasons"].items())
        )
        print(f"{threat:<16}{distance:>2}{cell['total']:>7}{cell['triaged']:>9}"
              f"{cell['live']:>6}{rate:>7}  {reasons}")

    triaged = [v for v in verdicts.values()]
    live = [v for v in triaged if v["live"]]
    print(f"\noverall live rate: {len(live)}/{len(triaged)}"
          f" ({len(live) / len(triaged):.1%})" if triaged else "\nnothing triaged")
    if len(verdicts) < len(rows):
        print(f"INCOMPLETE -- {len(rows) - len(verdicts)} snapshots still "
              f"untriaged; rates above are over what has finished.")

    report = {
        "harness": "threat_corpus_triage_report",
        "snapshots": len(rows),
        "triaged": len(verdicts),
        "complete": len(verdicts) == len(rows),
        "thresholds": {"min_spread": args.min_spread,
                       "live_floor": args.live_floor,
                       "live_ceiling": args.live_ceiling},
        "overall": {"live": len(live), "triaged": len(triaged),
                    "rate": round(len(live) / len(triaged), 4) if triaged else None},
        "cells": [
            {"threat": threat, "distance": distance, **{
                k: (dict(v) if isinstance(v, Counter) else v)
                for k, v in cell.items()}}
            for (threat, distance), cell in sorted(cells.items())
        ],
        "shards": shards,
        "live_positions": sorted(
            ((v["episode_id"], v["decision_row"]) for v in live)
        ),
    }
    out = Path(args.out) if args.out else triage_dir / "triage_report.json"
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
