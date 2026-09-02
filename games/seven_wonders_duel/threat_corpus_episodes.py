"""Collapse scanned rows into distinct threat EPISODES, and stratify.

`threat_corpus_scan.py` reports every decision row whose shape matches. Those
rows are not independent: a threat standing for four turns produces four rows of
the same game, same card, same slot. Counting them separately overstates the
corpus by roughly the length of the average standoff.

An **episode** is one physical run: `(table, threatened card, target slot)` over
consecutive decision rows. It carries a `snapshots` list -- one per chain
distance the threat was seen at -- rather than being emitted once per snapshot.
An earlier version appended each snapshot to `episodes`, turning 82 runs into
134 reported "episodes"; every statistic and every sample drawn from that was
inflated. All counting, sampling and any future train/test split keys on
`episode_id`.

Whether the opponent can act on a threat **immediately** is a function of chain
distance, not of victory type:

  distance 0   the card is uncovered by the actor's own move; the opponent
               simply takes it on its next turn. No extra-turn Wonder needed.
  distance 1   the opponent must remove the coverer AND take the card in one
               turn, so exactly one extra-turn Wonder is required.
  distance 2+  beyond one extra turn; the threat is real but not immediate.

An earlier version keyed this on threat class instead and got both ends
backwards -- exempting distance-1 terminal cards that DO need an extra turn,
while demanding one for distance-0 pairs that do not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

THREAT_RANK = {"science_win": 0, "military_win": 0, "science_pair": 1, "military_band": 1}


def needs_extra_turn(episode) -> bool:
    """Does acting on this threat immediately require an extra-turn Wonder?

    Distance alone decides. Threat class does not enter: a terminal card one
    removal away still needs the extra turn to be taken now, and a progress-token
    pair sitting uncovered does not.
    """

    return episode["distance"] >= 1


def resolvable_with_one_extra_turn(episode) -> bool:
    """Can a single extra-turn Wonder convert this threat into a taken card?

    One extra turn buys one additional action, so it covers exactly distance 1.
    Distance 0 needs none; distance 2+ needs more than one and is not immediate.
    """

    return episode["distance"] <= 1


def episode_id(table, card, slot, first_row) -> str:
    raw = f"{table}|{card}|{tuple(slot)}|{first_row}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def build_episodes(positions):
    """One entry per physical run, carrying every distance snapshot."""

    by_key = defaultdict(list)
    for position in positions:
        for hit in position["public_threats"]:
            key = (position["table"], hit.get("target_card"), tuple(hit["target"]))
            by_key[key].append((position, hit))

    episodes = []
    for (table, card, slot), entries in by_key.items():
        rows = sorted({p["decision_row"] for p, _ in entries})
        # Split on gaps: the same card threatened again later, after the
        # situation cleared, is a second episode rather than a continuation.
        runs, current = [], [rows[0]]
        for row in rows[1:]:
            if row - current[-1] <= 2:
                current.append(row)
            else:
                runs.append(current)
                current = [row]
        runs.append(current)

        for run in runs:
            members = [(p, h) for p, h in entries if p["decision_row"] in run]
            snapshots = []
            for distance in sorted({h["distance"] for _p, h in members}):
                at = [(p, h) for p, h in members if h["distance"] == distance]
                position, hit = min(at, key=lambda ph: ph[0]["decision_row"])
                # Exact action identity: a later regret search needs the codec
                # index and the target, not just the kind of move.
                actions = [
                    {
                        "action_index": h2["action_index"],
                        "action_use": h2["action_use"],
                        "action_label": h2.get("action_label"),
                        "removes": h2["removes"],
                    }
                    for p2, h2 in at if p2["decision_row"] == position["decision_row"]
                ]
                seen, unique = set(), []
                for entry in actions:
                    if entry["action_index"] in seen:
                        continue
                    seen.add(entry["action_index"])
                    unique.append(entry)
                snapshots.append({
                    "distance": distance,
                    "decision_row": position["decision_row"],
                    "observation_digest": position.get("observation_digest"),
                    "age": position["age"],
                    "actor": position["actor"],
                    "opponent_unbuilt_extra_turn":
                        position["opponent_unbuilt_extra_turn"],
                    "extra_turn_legality_verified":
                        position.get("extra_turn_legality_verified"),
                    "creating_actions": unique,
                })
            threat = members[0][1]["threat"]
            episodes.append({
                "episode_id": episode_id(table, card, slot, run[0]),
                "table": table,
                "target_card": card,
                "target_slot": list(slot),
                "threat": threat,
                "rows_spanned": len(run),
                "row_span": [run[0], run[-1]],
                "distances": [s["distance"] for s in snapshots],
                "min_distance": min(s["distance"] for s in snapshots),
                "snapshots": snapshots,
            })
    episodes.sort(key=lambda e: (THREAT_RANK[e["threat"]], e["min_distance"], e["table"]))
    return episodes


def eligible_snapshots(episode, require_extra_turn):
    """Snapshots whose threat the opponent could actually act on."""

    out = []
    for snapshot in episode["snapshots"]:
        probe = {"threat": episode["threat"], "distance": snapshot["distance"]}
        if not resolvable_with_one_extra_turn(probe):
            continue  # distance 2+: real, but not an immediate threat
        if (
            require_extra_turn
            and needs_extra_turn(probe)
            and not snapshot["opponent_unbuilt_extra_turn"]
        ):
            continue
        out.append(snapshot)
    return out


def stratify(episodes, per_cell, require_extra_turn):
    """A sample spread across (threat class, chain distance), one row per
    episode-snapshot but never two rows from the same episode in a cell."""

    cells = defaultdict(list)
    for episode in episodes:
        for snapshot in eligible_snapshots(episode, require_extra_turn):
            cells[(episode["threat"], snapshot["distance"])].append((episode, snapshot))

    sample, used = [], Counter()
    for key in sorted(cells, key=lambda k: (THREAT_RANK[k[0]], k[0], k[1])):
        # Longer standoffs first: a threat the players themselves lived with for
        # several turns is one they treated as real.
        for episode, snapshot in sorted(
            cells[key], key=lambda es: -es[0]["rows_spanned"]
        ):
            if used[key] >= per_cell:
                break
            sample.append({
                "episode_id": episode["episode_id"],
                "table": episode["table"],
                "threat": episode["threat"],
                "target_card": episode["target_card"],
                "target_slot": episode["target_slot"],
                "rows_spanned": episode["rows_spanned"],
                **snapshot,
            })
            used[key] += 1
    return sample, {f"{k[0]}@d{k[1]}": len(v) for k, v in sorted(cells.items())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scan", default="runs/seven_wonders_duel/threat_corpus/scan_all.json")
    parser.add_argument("--per-cell", type=int, default=3)
    parser.add_argument("--require-extra-turn", action="store_true", default=True)
    parser.add_argument("--allow-no-extra-turn", dest="require_extra_turn",
                        action="store_false")
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    log = (lambda *_: None) if args.quiet else (
        lambda m: print(m, file=sys.stderr, flush=True)
    )

    scan_path = Path(args.scan)
    if not scan_path.is_absolute():
        scan_path = REPO_ROOT / scan_path
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    positions = scan["positions"]

    episodes = build_episodes(positions)
    sample, cells = stratify(episodes, args.per_cell, args.require_extra_turn)

    log(f"{len(positions)} scanned rows -> {len(episodes)} EPISODES "
        f"across {len({e['table'] for e in episodes})} tables "
        f"({sum(len(e['snapshots']) for e in episodes)} snapshots)")
    log(f"\ncell sizes (threat@distance): {json.dumps(cells, indent=2)}")
    log(f"\nstratified sample: {len(sample)} rows from "
        f"{len({s['episode_id'] for s in sample})} distinct episodes")
    for row in sample:
        log(
            f"  {row['episode_id']}  {row['table']:<10} row {row['decision_row']:<3} "
            f"age {row['age']} {row['threat']:<14} d={row['distance']} "
            f"span={row['rows_spanned']:<2} {str(row['target_card'])[:20]:<22}"
            f"{','.join(row['opponent_unbuilt_extra_turn'])[:30]}"
        )

    report = {
        "harness": "threat_corpus_episodes",
        "source_scan": str(scan_path.relative_to(REPO_ROOT)),
        "note": (
            "one episode == one physical (table, card, slot) run; snapshots are "
            "nested, never counted as separate episodes"
        ),
        "params": {"per_cell": args.per_cell,
                   "require_extra_turn": bool(args.require_extra_turn)},
        "counts": {
            "scanned_rows": len(positions),
            "episodes": len(episodes),
            "snapshots": sum(len(e["snapshots"]) for e in episodes),
            "tables": len({e["table"] for e in episodes}),
            "sampled_rows": len(sample),
            "sampled_episodes": len({s["episode_id"] for s in sample}),
            "by_threat": dict(Counter(e["threat"] for e in episodes)),
            "by_min_distance": dict(Counter(e["min_distance"] for e in episodes)),
        },
        "cells": cells,
        "sample": sample,
        "episodes": episodes,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        log(f"\nwrote {out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
