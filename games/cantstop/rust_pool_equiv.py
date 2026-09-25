"""M3 gate and benchmark for the Rust self-play pool.

Gate: the pool plays exactly the games ``self_play.play_game`` plays --
same winner, same training rows byte for byte, same winner slots, turns,
solves, evaluator rows and turn lengths -- for every game, whatever the
thread count and however many games are in flight.

Evaluators are deterministic per-row mocks that work on features, so the
Python path (boards -> ``encode_batch``) and the pool (Rust-encoded
features) score every leaf identically no matter how rows are batched:

    hashed      a pseudo-random value per leaf from a CRC of its row, so play
                is varied and games actually finish
    mover_wins  every leaf a win for the seat that just moved: every
                stop-or-roll choice sits on a rounding error (see
                rust_solver.py), so any backup difference changes the game

Benchmark: games per hour on the pool vs the Python loop with a real
(untrained) net, per rule set, CPU and GPU.

    python -m games.cantstop.rust_pool_equiv gate --games 4
    python -m games.cantstop.rust_pool_equiv bench --games 64
"""

import argparse
import time
import zlib

import numpy as np

from .encoder import encode_batch
from .engine import ALL_RULESETS, RuleSet
from .portable_rng import PortableRng
from .rust_equiv import Divergence
from .rust_pool import PoolStats, game_seeds, run_pool
from .self_play import play_game


class FeatureMock:
    """Base: a per-row function of (features, seat to move).

    ``rows`` counts boards scored through the Python path, which is how
    ``play_game`` reports ``evaluator_rows`` (``NetEvaluator`` keeps the same
    counter)."""

    rows = 0

    def __call__(self, boards):
        self.rows += len(boards)
        return self.evaluate_features(encode_batch(boards), boards[0])

    def evaluate_features(self, features, reference):
        n, a = reference.rules.num_players, reference.active_player
        return np.stack([self.row(f, n, a) for f in features])


class HashedMock(FeatureMock):
    def row(self, f, n, a):
        rng = PortableRng(zlib.crc32(f.tobytes() + bytes([a])))
        v = np.array([rng.next_float() + 1e-3 for _ in range(n)])
        return v / v.sum()


class MoverWinsMock(FeatureMock):
    def row(self, f, n, a):
        v = np.zeros(n)
        v[(a - 1) % n] = 1.0
        return v


MOCKS = {"hashed": HashedMock(), "mover_wins": MoverWinsMock()}

FIELDS = ("winner", "turns", "solves", "evaluator_rows", "turn_lengths",
          "turn_values")


def compare_results(py, rs, where):
    for name in FIELDS:
        a, b = getattr(py, name), getattr(rs, name)
        if a != b:
            raise Divergence(f"{name} differs at {where}: "
                             f"python={a!r} rust={b!r}")
    if not np.array_equal(py.winner_slots, rs.winner_slots):
        raise Divergence(f"winner_slots differ at {where}")
    if py.features.tobytes() != rs.features.tobytes():
        raise Divergence(f"training rows differ at {where}")


def python_games(rules_list, seeds, evaluate):
    return [play_game(r, evaluate, PortableRng(s))
            for r, s in zip(rules_list, seeds)]


def gate(rules_list, seeds, mock, threads_options=(1, 0), in_flight=0):
    """Python reference vs the pool at each thread count (0 = all cores)."""
    reference = python_games(rules_list, seeds, MOCKS[mock])
    for threads in threads_options:
        got = run_pool(rules_list, seeds, [MOCKS[mock]], threads=threads,
                       in_flight=in_flight)
        for i, (p, r) in enumerate(zip(reference, got)):
            compare_results(p, r, f"game {i} {rules_list[i]} seed={seeds[i]}"
                                  f" mock={mock} threads={threads}")
    return reference


def _schedule(games_per_ruleset, seed):
    rules_list = [r for r in ALL_RULESETS for _ in range(games_per_ruleset)]
    return rules_list, game_seeds(PortableRng(seed), len(rules_list))


def cmd_gate(args):
    rules_list, seeds = _schedule(args.games, args.seed)
    started = time.perf_counter()
    for mock in MOCKS:
        ref = gate(rules_list, seeds, mock)
        rows = sum(len(r) for r in ref)
        print(f"{mock}: {len(ref)} games, {rows} rows, identical at 1 thread "
              f"and all cores", flush=True)
    print(f"GREEN in {time.perf_counter() - started:.0f} s")


def cmd_bench(args):
    import torch
    from .model import CantStopNet, NetEvaluator
    torch.manual_seed(0)
    net = CantStopNet()
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rule_sets = [RuleSet.make(2), RuleSet.make(4, blocking=True),
                 RuleSet.make(2, extended=True)]
    for rules in rule_sets:
        seeds = game_seeds(PortableRng(args.seed), args.games)
        line = [f"{rules}"]
        if args.python_games:
            ev = NetEvaluator(net, device="cpu")
            t = time.perf_counter()
            python_games([rules] * args.python_games,
                         seeds[:args.python_games], ev)
            py_gps = args.python_games / (time.perf_counter() - t)
            line.append(f"python {3600 * py_gps:8.0f} games/h")
        for device in devices:
            ev = NetEvaluator(net, device=device)
            stats = PoolStats()
            run_pool([rules] * args.games, seeds, [ev], threads=args.threads,
                     stats=stats, in_flight=args.in_flight)
            gph = 3600 * args.games / stats.wall_seconds
            line.append(
                f"pool/{device} {gph:9.0f} games/h "
                f"(rust {stats.rust_seconds:5.1f}s eval "
                f"{stats.eval_seconds:5.1f}s, {stats.rounds} rounds, "
                f"{stats.rows / max(1, stats.rounds):6.0f} rows/round)")
        print("\n   ".join(line), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--games", type=int, default=4, help="per rule set")
    g.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("bench")
    b.add_argument("--games", type=int, default=256, help="per rule set")
    b.add_argument("--in-flight", type=int, default=64)
    b.add_argument("--python-games", type=int, default=4)
    b.add_argument("--threads", type=int, default=0)
    b.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    {"gate": cmd_gate, "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    main()
