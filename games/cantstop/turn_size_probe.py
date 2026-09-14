"""Measure turn-solver cost on game positions, per variant.

For each rule set, plays full games with the solver choosing every move
(heuristic leaf evaluator) and records, for every turn:
  positions   runner configurations the solve enumerated
  leaves      evaluator calls (end-of-turn boards) -- what a net would cost
  seconds     wall time of the Python solve
split by runners already placed (0 = turn start, before the first roll).

One solve covers the rest of the turn, so self-play pays the turn-start
(runners 0) cost once per turn; the 1-3 rows show how fast the table
shrinks after the first moves.

    python -m games.cantstop.turn_size_probe --games 2
"""

import argparse
import random
import statistics
import time
from collections import defaultdict

from .engine import (ALL_RULESETS, GameState, Phase, apply_move, random_dice,
                     roll, stop)
from .solver import ProgressHeuristic, TurnSolver


def probe(rules, games, rng, evaluate):
    """rows[runners placed] -> [(positions, leaves, secs, game fraction)]"""
    rows = defaultdict(list)
    turns_per_game = []
    for _ in range(games):
        s = GameState(rules)
        records = []  # (runners, positions, leaves, secs, turn index)
        turn = 0
        solver = None
        measured = set()
        while not s.game_over:
            if s.phase == Phase.AWAIT_ROLL and not s.runners:
                t0 = time.perf_counter()
                solver = TurnSolver(s, evaluate)
                records.append((0, solver.num_positions,
                                solver.evaluator_calls,
                                time.perf_counter() - t0, turn))
                measured = {0}
                turn += 1
            if s.phase == Phase.AWAIT_MOVE:
                apply_move(s, solver.choose_move(s))
                n = len(s.runners)
                if n not in measured:
                    measured.add(n)
                    t0 = time.perf_counter()
                    sub = TurnSolver(s, evaluate)
                    records.append((n, sub.num_positions, sub.evaluator_calls,
                                    time.perf_counter() - t0, turn - 1))
            elif s.phase == Phase.AWAIT_DECISION and solver.should_stop(s):
                stop(s)
            else:
                roll(s, random_dice(rng))
        turns_per_game.append(turn)
        for n, pos, leaves, secs, t in records:
            rows[n].append((pos, leaves, secs, t / max(turn, 1)))
    return rows, turns_per_game


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2,
                    help="full games per rule set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    evaluate = ProgressHeuristic()

    print(f"{'rules':<22}{'runners':>8}{'n':>5}"
          f"{'pos p50':>9}{'pos p90':>9}{'pos max':>9}"
          f"{'leaf p50':>9}{'sec p50':>9}{'sec p90':>9}{'sec max':>9}"
          f"{'sec 1st half':>13}{'sec 2nd half':>13}")
    for rules in ALL_RULESETS:
        rows, turns = probe(rules, args.games, rng, evaluate)
        name = (f"{rules.num_players}p to {rules.columns_to_win}"
                f"{' block' if rules.blocking else ''}")
        print(f"{name}: turns per game {turns}", flush=True)
        for n in sorted(rows):
            pos = [r[0] for r in rows[n]]
            leaf = [r[1] for r in rows[n]]
            sec = [r[2] for r in rows[n]]
            early = [r[2] for r in rows[n] if r[3] < .5] or [0]
            late = [r[2] for r in rows[n] if r[3] >= .5] or [0]
            print(f"{name:<22}{n:>8}{len(pos):>5}"
                  f"{pct(pos, .5):>9}{pct(pos, .9):>9}{max(pos):>9}"
                  f"{pct(leaf, .5):>9}{pct(sec, .5):>9.2f}"
                  f"{pct(sec, .9):>9.2f}{max(sec):>9.2f}"
                  f"{statistics.mean(early):>13.2f}{statistics.mean(late):>13.2f}",
                  flush=True)
            name = ""


if __name__ == "__main__":
    main()
