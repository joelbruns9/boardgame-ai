"""Run the reference measurement over the threat corpus, position by position.

`w9_reference_case.py` is built around ONE position: its `--walk-action` and
`--tracked` default to the reference case's own move and refutation. The corpus
has 248 episodes with different creating actions and different threatened cards,
so a batch needs those threaded per position rather than defaulted.

This driver does that, and writes one artifact per position plus a summary. It
is deliberately resumable -- each position's artifact is written as it completes
and skipped on a re-run -- because the batch is hours long and a laptop is not a
reliable place to hold a single process open.

What it measures per position:

  ref-values   the probability-weighted value of EVERY legal action, at a
               common ply, so action regret can be read. This is the expensive
               stage and the one that answers "does this threat cost anything".
  trace        (optional) the discovery profile of the specific refutation, for
               positions where the creating action and the refutation are both
               identifiable.

The refutation differs by chain distance, which is why it cannot be a constant:

  distance 0   the opponent simply BUILDS the threatened card next turn.
  distance 1   the opponent needs an extra-turn Wonder to uncover and take it in
               one turn, so the tracked action is that Wonder.

Cost, measured on `906378778` row 19 (15 legal actions, 7 chance worlds each):
70 minutes at `--ref-sims 1500`, 24 minutes at 600. Ranking was identical at 600
-- mean absolute change 0.44 points, no action moving 3+ ranks -- so 600 is the
default here. A position whose top two actions land within `--recheck-margin` is
re-run at `--recheck-sims`, because a tight margin is exactly where the cheaper
budget would not resolve the order.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def row_id_from(episode, snapshot):
    """A measurable row from an episode's snapshot."""

    return {
        "episode_id": episode["episode_id"], "table": episode["table"],
        "threat": episode["threat"], "target_card": episode["target_card"],
        "rows_spanned": episode["rows_spanned"],
    }

REPO_ROOT = Path(__file__).resolve().parents[2]


def tracked_for(row) -> str | None:
    """The action that exploits the threat, given chain distance.

    Distance 0: the opponent builds the threatened card itself. Distance 1: it
    must first remove the coverer, so the exploit is the extra-turn Wonder.
    Beyond that one extra turn is not enough and there is no single action to
    track.
    """

    if row["distance"] == 0:
        return row["target_card"]
    if row["distance"] == 1:
        wonders = row.get("opponent_unbuilt_extra_turn") or []
        return wonders[0] if wonders else None
    return None


def triage_position(row, args, log) -> dict:
    """Cheap liveness check: is there a decision here worth measuring?

    A position can have the right SHAPE and carry no signal. Measured on four
    military positions: one actor sat at 0-2% in every line (already lost), two
    at 96-99.8% with action spreads of 3.5 and 0.2 points (already won), and only
    one was a live decision. A threat can be structurally real and strategically
    irrelevant -- `907771438` is flagged for a military_win threat against an
    actor who is winning at 96%.

    Running the full measurement on those spends hours to learn nothing, so this
    filters first at a budget of seconds. It is deliberately crude: the question
    is only whether the actions differ enough for regret to exist.
    """

    out = Path(args.out_dir)
    if not out.is_absolute():
        out = REPO_ROOT / out
    (out / "triage").mkdir(parents=True, exist_ok=True)
    artifact = out / "triage" / f"{row['episode_id']}_r{row['decision_row']}.json"

    cmd = [
        sys.executable, "-m", "games.seven_wonders_duel.w9_reference_case",
        "--table", row["table"], "--decision-row", str(row["decision_row"]),
        "--no-verify-position", "--stages", "ref-values",
        "--ref-worlds", str(args.triage_worlds),
        "--ref-sims", str(args.triage_sims),
        "--ref-sample", "random", "--out", str(artifact), "--quiet",
    ]
    if args.allow_migration:
        cmd.append("--allow-migration")
    if args.checkpoint:
        cmd += ["--checkpoint", str(args.checkpoint)]
    if artifact.exists() and not args.force:
        elapsed = 0.0
        result = None
    else:
        started = time.perf_counter()
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            return {**row_id(row), "status": "failed", "seconds": round(elapsed, 1),
                    "stderr": result.stderr.strip()[-600:]}

    actions = json.loads(artifact.read_text(encoding="utf-8"))["reference_values"]["actions"]
    best = actions[0]["win_pct_weighted"]
    worst = actions[-1]["win_pct_weighted"]
    spread = best - worst
    live = (
        spread >= args.min_spread
        and args.live_floor <= best <= args.live_ceiling
        and len(actions) > 1
    )
    reason = (
        "live" if live
        else "no_spread" if spread < args.min_spread
        else "decided_lost" if best < args.live_floor
        else "decided_won"
    )
    entry = {
        **row_id(row), "status": "ok", "live": live, "reason": reason,
        "seconds": round(elapsed, 1), "legal_actions": len(actions),
        "best_pct": best, "worst_pct": worst, "spread": round(spread, 2),
        # A forced position has one legal action and therefore no margin. The
        # liveness test above already excludes it via `len(actions) > 1`; this
        # is the same fact, and indexing actions[1] for it crashed the shard.
        "top_margin": (round(best - actions[1]["win_pct_weighted"], 2)
                       if len(actions) > 1 else None),
    }
    log(
        f"  {'LIVE ' if live else 'skip '} {row['episode_id']} {row['table']} "
        f"row {row['decision_row']:<3} {row['threat']:<14} d={row['distance']} "
        f"{elapsed:>5.1f}s  {worst:>5.1f}-{best:<5.1f}%  spread={spread:>5.1f}  {reason}"
    )
    return entry


def row_id(row):
    return {
        "episode_id": row["episode_id"], "table": row["table"],
        "decision_row": row["decision_row"], "threat": row["threat"],
        "distance": row["distance"], "target_card": row.get("target_card"),
    }


def run_position(row, args, log) -> dict:
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / f"{row['episode_id']}_r{row['decision_row']}.json"

    if artifact.exists() and not args.force:
        log(f"  {row['episode_id']} row {row['decision_row']}: already done, skipping")
        return {"episode_id": row["episode_id"], "artifact": artifact.name,
                "status": "cached"}

    stages = ["ref-values"]
    tracked = tracked_for(row)
    if args.trace and tracked:
        stages.append("trace")

    cmd = [
        sys.executable, "-m", "games.seven_wonders_duel.w9_reference_case",
        "--table", row["table"],
        "--decision-row", str(row["decision_row"]),
        "--no-verify-position",
        "--stages", ",".join(stages),
        "--ref-worlds", str(args.ref_worlds),
        "--ref-sims", str(args.ref_sims),
        "--ref-sample", "random",
        "--out", str(artifact),
        "--quiet",
    ]
    if args.allow_migration:
        cmd.append("--allow-migration")
    if args.checkpoint:
        cmd += ["--checkpoint", str(args.checkpoint)]
    if tracked:
        cmd += ["--tracked", tracked]
    if args.trace and tracked:
        cmd += ["--trace-sims", str(args.trace_sims),
                "--trace-seeds", str(args.trace_seeds)]

    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        log(f"  {row['episode_id']} row {row['decision_row']}: FAILED "
            f"({elapsed / 60:.1f}m)\n{result.stderr.strip()[-400:]}")
        return {"episode_id": row["episode_id"], "status": "failed",
                "seconds": round(elapsed, 1),
                "stderr": result.stderr.strip()[-2000:]}

    report = json.loads(artifact.read_text(encoding="utf-8"))
    actions = report["reference_values"]["actions"]
    top = actions[0]["win_pct_weighted"]
    second = actions[1]["win_pct_weighted"] if len(actions) > 1 else top
    margin = top - second

    entry = {
        "episode_id": row["episode_id"], "table": row["table"],
        "decision_row": row["decision_row"], "threat": row["threat"],
        "distance": row["distance"], "target_card": row["target_card"],
        "tracked": tracked, "status": "ok", "seconds": round(elapsed, 1),
        "legal_actions": len(actions),
        "best": actions[0]["label"], "best_pct": top,
        "top_margin": round(margin, 2),
        "worst_pct": actions[-1]["win_pct_weighted"],
        "artifact": artifact.name,
    }

    # A margin this tight is not resolved by the cheap budget; re-run it alone.
    if margin < args.recheck_margin and args.recheck_sims > args.ref_sims:
        log(f"  {row['episode_id']}: margin {margin:.2f} < {args.recheck_margin}"
            f" -- re-running at {args.recheck_sims} sims")
        recheck = out / f"{row['episode_id']}_r{row['decision_row']}_recheck.json"
        cmd[cmd.index("--ref-sims") + 1] = str(args.recheck_sims)
        cmd[cmd.index("--out") + 1] = str(recheck)
        again = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if again.returncode == 0:
            deep = json.loads(recheck.read_text(encoding="utf-8"))
            deep_actions = deep["reference_values"]["actions"]
            entry["recheck"] = {
                "sims": args.recheck_sims,
                "best": deep_actions[0]["label"],
                "best_pct": deep_actions[0]["win_pct_weighted"],
                "rank_changed": deep_actions[0]["label"] != actions[0]["label"],
                "artifact": recheck.name,
            }
    log(
        f"  {row['episode_id']} row {row['decision_row']:<3} "
        f"{row['threat']:<14} d={row['distance']} "
        f"{elapsed / 60:>5.1f}m  {len(actions):>2} actions  "
        f"best={entry['best'][:34]:<36} margin={margin:>5.2f}"
    )
    return entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes",
                        default="runs/seven_wonders_duel/threat_corpus/episodes.json")
    parser.add_argument("--out-dir",
                        default="runs/seven_wonders_duel/threat_corpus/measured")
    parser.add_argument("--ref-sims", type=int, default=600)
    parser.add_argument("--ref-worlds", type=int, default=10)
    parser.add_argument("--recheck-margin", type=float, default=1.5,
                        help="re-run a position whose top two actions are closer "
                             "than this, where the cheap budget cannot order them")
    parser.add_argument("--recheck-sims", type=int, default=1500)
    parser.add_argument("--trace", action="store_true",
                        help="also run the discovery trace where a refutation "
                             "action is identifiable")
    parser.add_argument("--trace-sims", type=int, default=6000)
    parser.add_argument("--trace-seeds", type=int, default=3)
    parser.add_argument("--triage", action="store_true",
                        help="cheap liveness pass instead of the full measurement")
    parser.add_argument("--triage-sims", type=int, default=400)
    parser.add_argument("--triage-worlds", type=int, default=6)
    parser.add_argument("--min-spread", type=float, default=3.0,
                        help="points between best and worst action below which "
                             "no decision is at stake")
    parser.add_argument("--live-floor", type=float, default=10.0)
    parser.add_argument("--live-ceiling", type=float, default=90.0)
    parser.add_argument("--live-from", default=None,
                        help="a triage summary; measure only its live positions")
    parser.add_argument("--triage-sample", type=int, default=0,
                        help="triage roughly this many snapshots, stratified "
                             "across (threat class, chain distance)")
    parser.add_argument("--all-episodes", action="store_true",
                        help="triage every episode snapshot, not just the sample")
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="run only every Nth row, offset I (0-based). Rows "
                             "are disjoint across shards and each position "
                             "writes its own artifact, so N shards may run "
                             "concurrently against one out-dir. Give each its "
                             "own --summary-out; the summary is the only "
                             "shared write.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="measure THIS model rather than w9_reference_case's "
                             "default. Required to compare A/B arms: without it "
                             "every arm would be measured against the same "
                             "incumbent and report identical regret.")
    parser.add_argument("--allow-migration", action="store_true",
                        help="measure a checkpoint whose encoder signature has "
                             "moved, warm-started additively. Required after a "
                             "schema change (W3 added control channels): the "
                             "migrated model computes exactly what it did "
                             "before, since the new columns are zero, which is "
                             "the right BEFORE state for a regression baseline.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    log = (lambda *_: None) if args.quiet else (
        lambda m: print(m, file=sys.stderr, flush=True)
    )

    path = Path(args.episodes)
    if not path.is_absolute():
        path = REPO_ROOT / path
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if args.all_episodes or args.triage_sample:
        rows = [
            {**row_id_from(episode, snapshot), **snapshot}
            for episode in corpus["episodes"]
            for snapshot in episode["snapshots"]
        ]
    else:
        rows = corpus["sample"]

    if args.live_from:
        # Measure only positions a triage pass already classified as live.
        # Triage is cheap relative to the full measurement but not free, so its
        # verdict is reused rather than recomputed, and a position with no
        # decision at stake is never paid for twice.
        triage_path = Path(args.live_from)
        if not triage_path.is_absolute():
            triage_path = REPO_ROOT / triage_path
        verdicts = json.loads(triage_path.read_text(encoding="utf-8"))["positions"]
        keep = {
            (v["episode_id"], v["decision_row"])
            for v in verdicts if v.get("status") == "ok" and v.get("live")
        }
        rows = [r for r in rows if (r["episode_id"], r["decision_row"]) in keep]

    if args.triage_sample:
        # Spread the triage budget across (threat class, chain distance) so the
        # live RATE can be estimated per class. Triaging the first N in file
        # order would answer that for whichever class happens to sort first.
        cells = defaultdict(list)
        for row in rows:
            cells[(row["threat"], row["distance"])].append(row)
        total = sum(len(v) for v in cells.values())
        picked = []
        for key, group in sorted(cells.items()):
            share = max(2, round(args.triage_sample * len(group) / total))
            # Longer standoffs first: the players themselves treated those as real.
            picked.extend(
                sorted(group, key=lambda r: -r.get("rows_spanned", 0))[:share]
            )
        rows = picked
    if args.shard:
        # Interleave rather than block: cost varies more than fourfold across
        # the corpus, so contiguous blocks would finish at wildly different
        # times and the expensive shard would still be running at breakfast.
        index, count = (int(part) for part in args.shard.split("/"))
        if not 0 <= index < count:
            raise SystemExit(f"--shard {args.shard}: need 0 <= I < N")
        rows = rows[index::count]

    rows = rows[: args.limit] if args.limit else rows

    if args.triage:
        log(f"triaging {len(rows)} position(s) at {args.triage_sims} sims")
    else:
        log(f"measuring {len(rows)} position(s) at {args.ref_sims} sims"
            f"{' + trace' if args.trace else ''}")
    started = time.perf_counter()
    results = []
    for row in rows:
        results.append(
            triage_position(row, args, log) if args.triage
            else run_position(row, args, log)
        )
    elapsed = time.perf_counter() - started

    ok = [r for r in results if r.get("status") == "ok"]
    live = [r for r in ok if r.get("live")]
    report = {
        "harness": "threat_corpus_measure",
        "episodes_source": str(path.relative_to(REPO_ROOT)),
        "params": {
            "ref_sims": args.ref_sims, "ref_worlds": args.ref_worlds,
            "recheck_margin": args.recheck_margin,
            "recheck_sims": args.recheck_sims, "trace": bool(args.trace),
            "shard": args.shard,
        },
        "totals": {
            "positions": len(results),
            "ok": len(ok),
            "failed": sum(1 for r in results if r.get("status") == "failed"),
            "cached": sum(1 for r in results if r.get("status") == "cached"),
            "rechecked": sum(1 for r in ok if "recheck" in r),
            "rank_changed_on_recheck": sum(
                1 for r in ok if r.get("recheck", {}).get("rank_changed")
            ),
            "wall_clock_minutes": round(elapsed / 60, 1),
            **({
                "live": len(live),
                "decided": len(ok) - len(live),
                "live_fraction": round(len(live) / len(ok), 3) if ok else None,
                "by_reason": dict(Counter(r.get("reason") for r in ok)),
            } if args.triage else {}),
        },
        "live_positions": [r for r in live] if args.triage else None,
        "positions": results,
    }
    log("\n" + json.dumps(report["totals"], indent=2))
    payload = json.dumps(report, indent=2) + "\n"
    if args.summary_out:
        out = Path(args.summary_out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        log(f"wrote {out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
