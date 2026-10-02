"""Is the net's turn-boundary residual a blind spot or a selection effect?

In the user's BGA games the net's value of the viewer's position drifts down
by ~6 win-probability points per game when checked one turn deeper (the
ledger's "residual", games/cantstop/luck.py). Two readings:

  selection   the net-following seat stops exactly where the net's value
              peaks, so its errors are selected in its favour (the
              optimizer's curse); a one-turn look-ahead then regresses them.
              Predicts the drift in ANY game where one seat follows the net,
              concentrated right after that seat's own stops.
  blind spot  the net misjudges the boards human opponents create.
              Predicts no comparable drift against a non-human opponent.

The control: the net against Rule of 28 (a fixed heuristic, so it never
selects on the net's errors) in the user's variant, every roll recorded and
run through the same ledger. The residual is split by what ended the turn
before the boundary -- the net seat's stop or bust, the opponent's -- for
both the simulation and the BGA games.

    python -m games.cantstop.residual_control --games 300 \\
        --log-dir ../boardgame-ai/runs/cantstop/bga_game_log
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .advisor_adapter import DEFAULT_CHECKPOINT, wins_on_stop
from .engine import (GameState, RuleSet, apply_move, can_stop, random_dice,
                     roll, stop)
from .luck import (DEFAULT_LOG_DIR, Step, Turn, build_ledger, load_game,
                   read_turns, turn_start)
from .rule28 import HEURISTICS, choose_move as r28_move, should_stop as r28_stop
from .snapshot import snapshot


class SolverCache:
    """Turn-start solves shared by play and ledger (the same tables)."""

    def __init__(self, evaluate):
        self.evaluate, self.cache = evaluate, {}

    def __call__(self, start, evaluate=None):
        from .rust_solver import RustTurnSolver
        key = json.dumps(snapshot(start))
        if key not in self.cache:
            if len(self.cache) > 2000:
                self.cache.clear()
            self.cache[key] = RustTurnSolver(start, self.evaluate)
        return self.cache[key]


def play_recorded(rules, solvers, rng, net_seat, params):
    """One game, net at ``net_seat`` and Rule of 28 elsewhere, recorded in
    the ledger's Turn/Step form."""
    state, turns = GameState(rules), []
    while not state.game_over:
        if len(turns) > 1000:
            raise RuntimeError("game did not finish")
        seat = state.active_player
        start = turn_start(state)
        turn = Turn(seat, start, [], "adjacent" if turns else "first")
        turns.append(turn)
        solver = solvers(start) if seat == net_seat else None
        while True:
            step = Step(dict(state.runners), None)
            turn.steps.append(step)
            dice = random_dice(rng)
            if not roll(state, dice):
                turn.end = "bust"
                break
            step.dice = state.dice
            move = solver.choose_move(state) if solver else r28_move(state, params)
            apply_move(state, move)
            step.move = move
            if wins_on_stop(state) or (can_stop(state) and (
                    solver.should_stop(state) if solver else r28_stop(state, params))):
                stop(state)
                step.then = "stop"
                turn.end = "win" if state.game_over else "stop"
                break
            step.then = "roll"
    return turns, state.winner


def boundary_split(ledger, turns, seat):
    """Residual on ``seat`` at each turn boundary, keyed by what ended the
    turn before it: ('self' | 'opp', 'stop' | 'bust')."""
    out = defaultdict(list)
    for e in ledger.entries:
        if e["kind"] != "residual":
            continue
        prev = turns[e["turn"] - 1]
        who = "self" if prev.seat == seat else "opp"
        out[(who, "bust" if prev.end == "bust" else "stop")].append(float(e["delta"][seat]))
    return out


def per_game_residual(ledger, seat):
    return sum(float(e["delta"][seat]) for e in ledger.entries if e["kind"] == "residual")


def describe(name, per_game, splits, n_games):
    xs = np.asarray(per_game)
    lines = [f"{name}: {n_games} games; residual on the net-following seat "
             f"{100 * xs.mean():+.2f} +/- {100 * 1.96 * xs.std(ddof=1) / np.sqrt(len(xs)):.2f} pts/game"]
    for key in (("self", "stop"), ("self", "bust"), ("opp", "stop"), ("opp", "bust")):
        v = np.asarray(splits.get(key, []))
        if len(v):
            lines.append(f"   after {'its own' if key[0] == 'self' else 'the opponent'}'s {key[1]:<4}"
                         f"  n={len(v):5d}  mean {100 * v.mean():+.2f}  "
                         f"+/- {100 * 1.96 * v.std(ddof=1) / np.sqrt(len(v)):.2f} pts per boundary"
                         f"  ({100 * v.sum() / n_games:+.2f} pts/game)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--opponent", default="rule_of_28", choices=sorted(HEURISTICS))
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    from .model import NetEvaluator, load_net
    evaluate = NetEvaluator(load_net(args.checkpoint, device=args.device), device=args.device)
    solvers = SolverCache(evaluate)
    rules = RuleSet(2, 5, False)            # the user's BGA variant
    params = HEURISTICS[args.opponent]
    rng = random.Random(args.seed)
    sim_games, sim_splits, wins, t0 = [], defaultdict(list), 0, time.time()
    for g in range(args.games):
        net_seat = g % 2
        turns, winner = play_recorded(rules, solvers, rng, net_seat, params)
        ledger = build_ledger(None, turns, evaluate, solver_cls=solvers)
        sim_games.append(per_game_residual(ledger, net_seat))
        for k, v in boundary_split(ledger, turns, net_seat).items():
            sim_splits[k] += v
        wins += winner == net_seat
        if (g + 1) % 25 == 0:
            xs = np.asarray(sim_games)
            print(f"{g + 1} games, {time.time() - t0:.0f} s: net wins {wins / (g + 1):.1%}, "
                  f"residual {100 * xs.mean():+.2f} pts/game", flush=True)

    bga_games, bga_splits = [], defaultdict(list)
    for path in sorted(args.log_dir.glob("table_*.jsonl")):
        try:
            game = load_game(path)
        except ValueError:
            continue
        if not game.opponents_logged or game.captures[0].state.rules != rules:
            continue
        turns = read_turns(path, game)
        if not game.opponents_logged:
            continue
        ledger = build_ledger(game, turns, evaluate, solver_cls=solvers)
        bga_games.append(per_game_residual(ledger, game.viewer_seat))
        for k, v in boundary_split(ledger, turns, game.viewer_seat).items():
            bga_splits[k] += v

    print()
    print(describe(f"Simulation, net vs {args.opponent} (net won {wins / args.games:.1%})",
                   sim_games, sim_splits, args.games))
    print()
    print(describe("BGA games, viewer following the net", bga_games, bga_splits, len(bga_games)))
    if args.out:
        args.out.write_text(json.dumps({
            "sim": {"per_game": sim_games, "splits": {"/".join(k): v for k, v in sim_splits.items()},
                    "net_wins": wins, "games": args.games},
            "bga": {"per_game": bga_games, "splits": {"/".join(k): v for k, v in bga_splits.items()}}}))


if __name__ == "__main__":
    main()
