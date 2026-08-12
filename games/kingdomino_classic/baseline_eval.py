"""Run reproducible, seat-balanced Classic Kingdomino baseline matches."""

from __future__ import annotations

import argparse
import json

from .baselines import BASELINE_FACTORIES
from .config import ClassicGameConfig
from .evaluation import Participant, run_challenger_tournament


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenger", choices=BASELINE_FACTORIES, required=True)
    parser.add_argument("--field", choices=BASELINE_FACTORIES, required=True)
    parser.add_argument("--players", type=int, choices=(3, 4), required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--harmony", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--middle-kingdom", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--include-games",
        action="store_true",
        help="Include every game record instead of emitting only aggregate metrics.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")
    config = ClassicGameConfig(
        players=args.players,
        harmony=args.harmony,
        middle_kingdom=args.middle_kingdom,
    )
    challenger = Participant(
        args.challenger, BASELINE_FACTORIES[args.challenger]
    )
    field = Participant(args.field, BASELINE_FACTORIES[args.field])
    report = run_challenger_tournament(
        challenger,
        field,
        config=config,
        seeds=range(args.seed_start, args.seed_start + args.seeds),
    )
    payload = report.to_dict() if args.include_games else report.summary_dict()
    payload.update(
        {
            "challenger": args.challenger,
            "field": args.field,
            "seed_start": args.seed_start,
            "seed_count": args.seeds,
            "challenger_win_share_lift": report.win_share_lift(args.challenger),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
