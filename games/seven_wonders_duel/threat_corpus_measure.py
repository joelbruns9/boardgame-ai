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
from pathlib import Path

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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    log = (lambda *_: None) if args.quiet else (
        lambda m: print(m, file=sys.stderr, flush=True)
    )

    path = Path(args.episodes)
    if not path.is_absolute():
        path = REPO_ROOT / path
    corpus = json.loads(path.read_text(encoding="utf-8"))
    rows = corpus["sample"][: args.limit] if args.limit else corpus["sample"]

    log(f"measuring {len(rows)} position(s) at {args.ref_sims} sims"
        f"{' + trace' if args.trace else ''}")
    started = time.perf_counter()
    results = []
    for row in rows:
        results.append(run_position(row, args, log))
    elapsed = time.perf_counter() - started

    ok = [r for r in results if r.get("status") == "ok"]
    report = {
        "harness": "threat_corpus_measure",
        "episodes_source": str(path.relative_to(REPO_ROOT)),
        "params": {
            "ref_sims": args.ref_sims, "ref_worlds": args.ref_worlds,
            "recheck_margin": args.recheck_margin,
            "recheck_sims": args.recheck_sims, "trace": bool(args.trace),
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
        },
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
