"""Run the sudden-death certifier on recorded BGA decision rows.

Step 2's own test instrument, before anything is wired into search. It answers
the two questions the certifier has to get right on the reference case:

    positive   at the row where the extra-turn Wonder is gone, does it PROVE the
               forced science defeat?
    negative   one row earlier, with the Wonder still in hand, does it decline
               (REFUTED or UNKNOWN) rather than fire?

    python -m games.seven_wonders_duel.control_certify_probe \\
        --table 907773062 --rows 84,85,86 --max-plies 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--table", required=True)
    parser.add_argument("--rows", required=True,
                        help="comma-separated decision rows")
    parser.add_argument("--log-dir",
                        default="runs/seven_wonders_duel/bga_game_log")
    parser.add_argument("--resample-seed", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=400_000)
    parser.add_argument("--max-plies", type=int, default=12)
    parser.add_argument("--max-secs", type=float, default=30.0)
    parser.add_argument("--loser", type=int, default=None,
                        help="seat to prove cannot escape (default: side to move)")
    parser.add_argument("--no-require-threat", action="store_true",
                        help="attempt even when the cheap gate sees no threat")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from .control_certify import certify, science_symbols, sudden_death_threat
    from .w9_reference_case import load_position

    log_path = REPO_ROOT / args.log_dir / f"table_{args.table}.jsonl"
    rows = [int(r) for r in args.rows.split(",") if r.strip()]
    report = {"table": args.table, "rows": []}

    for row in rows:
        position = load_position(log_path, row, args.resample_seed, strict=False)
        game = position.game
        actor = position.actor
        loser = args.loser if args.loser is not None else actor
        threat = sudden_death_threat(game, loser)
        certificate = certify(
            game,
            loser=loser,
            max_nodes=args.max_nodes,
            max_plies=args.max_plies,
            max_secs=args.max_secs,
            require_threat=not args.no_require_threat,
        )
        entry = {
            "decision_row": row,
            "age": game.age,
            "actor": actor,
            "loser": loser,
            "legal": len(position.legal),
            "science_symbols": {
                "seat0": len(science_symbols(game, 0)),
                "seat1": len(science_symbols(game, 1)),
            },
            "conflict_position": game.conflict_position,
            "wonders_unbuilt": {
                str(seat): [
                    w for w in game.cities[seat].wonders
                    if w not in game.cities[seat].built_wonders
                ]
                for seat in (0, 1)
            },
            "threat": threat,
            "verdict": certificate.verdict.value,
            "nodes": certificate.nodes,
            "seconds": certificate.seconds,
            "stopped_by": certificate.stopped_by,
            "limits_hit": list(certificate.limits_hit),
            "victory_type": (
                certificate.victory_type.value
                if certificate.victory_type is not None else None
            ),
            "principal_line": certificate.principal_line,
        }
        report["rows"].append(entry)
        print(
            f"row {row:>3}  age {game.age}  actor {actor}  loser {loser}  "
            f"sci {entry['science_symbols']['seat0']}/"
            f"{entry['science_symbols']['seat1']}  "
            f"conflict {game.conflict_position:+d}  "
            f"threat={threat}  -> {certificate.verdict.value.upper()}"
            f" ({certificate.nodes} nodes, {certificate.seconds}s"
            # EVERY limit that fired, not just the binding one. `Certificate`
            # carries both because "the horizon was too short" and "the machine
            # was too slow" call for opposite fixes -- and then this line
            # printed `stopped_by` alone and reintroduced the same trap one
            # layer out: a run whose branches were truncated by the ply horizon
            # 88% of the time reported simply "nodes", and was read as needing a
            # bigger node budget.
            f"{', stopped by ' + certificate.stopped_by if certificate.stopped_by else ''}"
            f"{', limits ' + '+'.join(sorted(certificate.limits_hit)) if certificate.limits_hit else ''})"
        )
        if certificate.principal_line:
            print("      line: " + " | ".join(certificate.principal_line[:8]))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
