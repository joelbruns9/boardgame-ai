"""Head-to-head evaluation of two nets, including against fixed personas.

Every match is seat-balanced (seats rotate each game) and reports the
first player's win rate with a Wilson interval.

    1. new vs old, both playing best (the before/after number);
    2. new vs X and old vs X for fixed opponents X = the old net playing
       conservative (+bias) and aggressive (-bias). If the personas did
       their job, the new net's margin should grow most against the style
       it was shown during training.

    python -m games.cantstop.eval_matches --new runs/td0_personas/iter_0120.pt \
        --old runs/lr1e4/iter_0120.pt --games 2000
"""

import argparse
import json
import time

import torch

from .arena import verdict
from .engine import RuleSet
from .model import NetEvaluator, load_net
from .portable_rng import PortableRng
from .rust_pool import play_match
from .self_play import PLAIN, Search


def match(rules, a, b, search_a, search_b, games, seed):
    t = time.perf_counter()
    wins = play_match(rules, [a, b], games, PortableRng(seed),
                      searches=[search_a, search_b])
    out = verdict(wins, rules.num_players)
    out["seconds"] = round(time.perf_counter() - t)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--bias", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    new = NetEvaluator(load_net(args.new, device=device), device=device)
    old = NetEvaluator(load_net(args.old, device=device), device=device)
    rules = RuleSet.make(2)
    cons = Search(stop_bias=args.bias)
    aggr = Search(stop_bias=-args.bias)
    plan = [
        ("new vs old (both best play)", new, old, PLAIN),
        ("new vs old-conservative", new, old, cons),
        ("old vs old-conservative", old, old, cons),
        ("new vs old-aggressive", new, old, aggr),
        ("old vs old-aggressive", old, old, aggr),
    ]
    report = {}
    for i, (name, a, b, opp_search) in enumerate(plan):
        r = match(rules, a, b, PLAIN, opp_search, args.games, args.seed + i)
        report[name] = r
        lo, hi = r["ci95"]
        print(f"{name:32} {r['wins']}  {r['win_rate']:.3f} "
              f"[{lo:.3f}, {hi:.3f}]  ({r['seconds']}s)", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
