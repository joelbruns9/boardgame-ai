"""Does the 2-turn lookahead play better? Same net, depth 2 vs depth 1.

Plan step 3 of the lookahead build: the arena measurement that decides
whether self-play should pay the lookahead's cost (measured ~24x at k=4).
Both players use the SAME net; only the search differs, so the result is
the value of the extra turn of search alone. Seats rotate every game.

    python -m games.cantstop.lookahead_arena --net runs/mvp2/iter_0010.pt \
        --k 4 --games 400
"""

import argparse
import json
import threading
import time

import psutil
import torch

from .arena import verdict
from .engine import RuleSet
from .model import NetEvaluator, load_net
from .portable_rng import PortableRng
from .rust_pool import PoolStats, play_match
from .self_play import PLAIN, Search


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--net", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--no-offset", action="store_true")
    ap.add_argument("--exact", action="store_true",
                    help="exact_root for the lookahead side (values only; "
                         "decisions are unchanged by it)")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--blocking", action="store_true")
    ap.add_argument("--in-flight", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = NetEvaluator(load_net(args.net, device=device), device=device)
    rules = RuleSet.make(args.players, args.extended, args.blocking)
    deep = Search(exact_root=args.exact, lookahead_k=args.k,
                  lookahead_offset=not args.no_offset)
    n = rules.num_players
    players = [net] * n
    searches = [deep] + [PLAIN] * (n - 1)

    proc, peak, done = psutil.Process(), [0], [False]

    def watch():
        while not done[0]:
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.2)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    stats = PoolStats()
    wins = play_match(rules, players, args.games, PortableRng(args.seed),
                      in_flight=args.in_flight, stats=stats,
                      searches=searches)
    done[0] = True
    result = verdict(wins, n)
    result.update(
        rules=str(rules), net=args.net, search=vars(deep),
        wall_seconds=round(stats.wall_seconds, 1),
        games_per_hour=round(3600 * args.games / stats.wall_seconds),
        peak_rss_gb=round(peak[0] / 2 ** 30, 2))
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
