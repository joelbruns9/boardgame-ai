"""Confirmation match: a final, pre-declared strength verdict.

Plan review (2026-09-26): 200-game monitoring matches (+/-6.9 pts near
50%) cannot resolve the 1-2 pt effects that drive decisions, and repeatedly
checking ten variants makes their intervals unsuitable as a stopping rule.
A confirmation is different by construction:

* the two checkpoints are chosen BEFORE the match (named on the command
  line, recorded with their hashes);
* the games are FRESH: seeds come from a confirmation namespace hashed
  with the checkpoint paths, never the training streams;
* the game count is rounded UP to complete seat cycles (a multiple of the
  player count), so every seat is played equally;
* the equivalence margin is declared up front (``--margin``, win-rate
  points around the even rate 1/n), and the verdict is one of
  better / worse / equivalent / inconclusive from the 95% interval.

~10,000 two-player games give about +/-1 pt near 50%.

    python -m games.cantstop.confirm --a runs/x/iter_0120.pt --b runs/y/iter_0120.pt \
        --games 10000 --margin 0.01 --out runs/confirm_x_vs_y.json
"""

import argparse
import math
import zlib

import torch

from .arena import verdict
from .engine import RuleSet
from .experiment import identity, write_json
from .model import NetEvaluator, load_net
from .portable_rng import PortableRng
from .rust_pool import play_match
from .self_play import Search


def seat_cycle_games(games, players):
    """Round up to a whole number of seat cycles."""
    return players * math.ceil(games / players)


def confirmation_seed(a, b, seed):
    """Fresh-game seed namespace, independent of every training stream."""
    return zlib.crc32(f"confirm|{a}|{b}|{seed}".encode())


def decide(ci, even, margin):
    """better / worse / equivalent / inconclusive for player A."""
    lo, hi = ci
    if lo > even + margin:
        return "better"
    if hi < even - margin:
        return "worse"
    if lo >= even - margin and hi <= even + margin:
        return "equivalent"
    return "inconclusive"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True, help="challenger checkpoint")
    ap.add_argument("--b", required=True, help="reference checkpoint "
                    "(fills every other seat)")
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--blocking", action="store_true")
    ap.add_argument("--games", type=int, default=10_000)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--bias-a", type=float, default=0.0)
    ap.add_argument("--bias-b", type=float, default=0.0)
    ap.add_argument("--mirror-a", action="store_true",
                    help="A averages each prediction with its column mirror")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rules = RuleSet.make(args.players, args.extended, args.blocking)
    n = rules.num_players
    games = seat_cycle_games(args.games, n)
    a = NetEvaluator(load_net(args.a, device=device), device=device,
                     mirror_average=args.mirror_a)
    b = NetEvaluator(load_net(args.b, device=device), device=device)
    search_a, search_b = Search(stop_bias=args.bias_a), Search(stop_bias=args.bias_b)
    seed = confirmation_seed(args.a, args.b, args.seed)
    wins = play_match(rules, [a] + [b] * (n - 1), games, PortableRng(seed),
                      searches=[search_a] + [search_b] * (n - 1))
    result = verdict(wins, n)
    result["decision"] = decide(result["ci95"], 1.0 / n, args.margin)
    report = {"meta": identity(nets={"a": args.a, "b": args.b},
                               rules=str(rules), games=games,
                               margin=args.margin, search_a=search_a,
                               search_b=search_b, mirror_a=args.mirror_a,
                               seed=seed),
              "result": result}
    write_json(args.out, report)
    lo, hi = result["ci95"]
    print(f"A {result['wins'][0]} of {games}: {result['win_rate']:.4f} "
          f"[{lo:.4f}, {hi:.4f}] vs even {1 / n:.4f} +/- {args.margin} "
          f"-> {result['decision']}")


if __name__ == "__main__":
    main()
