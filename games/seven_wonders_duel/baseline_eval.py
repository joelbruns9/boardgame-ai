"""CLI for reproducible baseline matches."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .bots import GreedyBot, RandomBot, play_series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = play_series(
        GreedyBot(),
        RandomBot(args.seed ^ 0x5EED),
        games=args.games,
        seed=args.seed,
    )
    payload = {
        "bot_a": "greedy",
        "bot_b": "random",
        "seed": args.seed,
        **asdict(result),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

