"""Net vs Rule of 28 opponents, sweeping the net's own stop bias.

Two questions, one run:

* **Outside yardstick** (bias 0): how often does the net beat a published
  heuristic that owes nothing to it (``rule28``)?
* **Exploitability**: against an opponent that ignores the board and banks by
  a fixed rule, does shifting the net's risk attitude pay? ``stop_bias`` is
  ``self_play.Search.stop_bias`` -- stop when stop value + bias >= roll
  value, applied inside the backup so move choices plan for it.

Every bias arm plays the SAME games: game ``i`` has the same dice seed and
the same net seat in every arm (the net rotates through all seats), so arms
are compared per game (paired) as well as by rate. Pairing only helps while
the games still coincide; they diverge at the first differing decision.

    python -m games.cantstop.rule28_sweep --checkpoint runs/p4_pilot/iter_0080.pt \\
        --games 2000 --workers 6 --out runs/rule28_sweep_2p.json

The heuristics were written for 2 players and play any seat count unchanged
(they never look at opponents).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import math
from pathlib import Path
import time

from .engine import (GameState, RuleSet, apply_move, can_stop, random_dice,
                     roll, stop)
from .portable_rng import PortableRng
from .rule28 import HEURISTICS, Rule28Player

DEFAULT_BIASES = (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06)

_worker = {}


def _init_worker(checkpoint, device):
    import torch
    from .model import NetEvaluator, load_net
    torch.set_num_threads(1)
    _worker["ev"] = NetEvaluator(load_net(checkpoint, device=device),
                                 device=device)


def net_turn(state, evaluate, rng, bias):
    """One net turn with the Rust solver, solved after the opening roll."""
    from .rust_solver import RustTurnSolver
    if not roll(state, random_dice(rng)):
        return
    solver = RustTurnSolver(state, evaluate, stop_bias=bias)
    while True:
        apply_move(state, solver.choose_move(state))
        if can_stop(state) and solver.should_stop(state):
            stop(state)
            return
        if not roll(state, random_dice(rng)):
            return


def play_game(rules, evaluate, seed, net_seat, bias, opponent,
              max_turns=1000):
    state = GameState(rules)
    rng = PortableRng(seed)
    heuristic = Rule28Player(HEURISTICS[opponent])
    turns = 0
    while not state.game_over:
        if turns >= max_turns:
            raise RuntimeError(f"game seed {seed} exceeded {max_turns} turns")
        if state.active_player == net_seat:
            net_turn(state, evaluate, rng, bias)
        else:
            heuristic.play_turn(state, rng)
        turns += 1
    return {"won": state.winner == net_seat, "turns": turns}


def _run_chunk(rules_key, opponent, bias, games):
    rules = RuleSet(*rules_key)
    out = []
    for index, seed in games:
        seat = index % rules.num_players
        r = play_game(rules, _worker["ev"], seed, seat, bias, opponent)
        out.append((index, seat, r["won"], r["turns"]))
    return bias, out


def game_seeds(seed, games):
    rng = PortableRng(seed)
    return [rng.next_u64() for _ in range(games)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(results, biases, num_players):
    """Rates per arm, plus each arm paired against bias 0 game by game."""
    base = {i: w for i, _, w, _ in results.get(0.0, [])}
    arms = []
    others = [b for b in biases if b != 0.0]
    z = _z_bonferroni(len(others))
    for b in biases:
        rows = results.get(b, [])
        n, k = len(rows), sum(w for _, _, w, _ in rows)
        arm = {"bias": b, "games": n, "wins": k,
               "win_rate": k / n if n else None, "ci95": wilson(k, n),
               "mean_turns": (sum(t for *_, t in rows) / n) if n else None,
               "by_seat": [
                   {"seat": s,
                    "games": sum(1 for _, st, _, _ in rows if st == s),
                    "wins": sum(1 for _, st, w, _ in rows if st == s and w)}
                   for s in range(num_players)]}
        if base and b != 0.0:
            d = [int(w) - int(base[i]) for i, _, w, _ in rows if i in base]
            m = sum(d) / len(d)
            var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
            se = math.sqrt(var / len(d))
            arm["paired_vs_bias0"] = {
                "games": len(d), "difference": m, "standard_error": se,
                "simultaneous_ci95": (m - z * se, m + z * se),
                "method": f"paired by game seed and seat; Bonferroni over "
                          f"{len(others)} arms (normal approximation)"}
        arms.append(arm)
    return {"null_rate": 1 / num_players, "arms": arms}


def _z_bonferroni(k):
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - 0.05 / (2 * max(1, k)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--blocking", action="store_true")
    ap.add_argument("--opponent", choices=sorted(HEURISTICS),
                    default="rule_of_28")
    ap.add_argument("--biases", type=float, nargs="+",
                    default=list(DEFAULT_BIASES))
    ap.add_argument("--games", type=int, default=2000, help="per bias arm")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error("output exists; choose a new --out")
    if args.games % args.players:
        ap.error("--games must be a multiple of --players (seat balance)")
    if 0.0 not in args.biases:
        ap.error("include bias 0: the other arms are paired against it")

    from .experiment import identity, write_json
    rules = RuleSet.make(args.players, args.extended, args.blocking)
    seeds = game_seeds(args.seed, args.games)
    report = {"status": "running",
              "meta": identity(nets={"net": args.checkpoint},
                               rules=asdict(rules), opponent=args.opponent,
                               opponent_params=asdict(HEURISTICS[args.opponent]),
                               biases=args.biases, games_per_arm=args.games,
                               seed=args.seed, device=args.device),
              "summary": None}
    write_json(args.out, report)
    results = {b: [] for b in args.biases}
    jobs = [(b, list(enumerate(seeds))[i:i + args.chunk])
            for b in args.biases for i in range(0, args.games, args.chunk)]
    rules_key = (rules.num_players, rules.columns_to_win, rules.blocking)
    started = time.perf_counter()
    done = 0
    try:
        with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                                 initargs=(args.checkpoint, args.device)) as pool:
            futures = [pool.submit(_run_chunk, rules_key, args.opponent, b, g)
                       for b, g in jobs]
            for f in as_completed(futures):
                bias, rows = f.result()
                results[bias].extend(rows)
                done += len(rows)
                total = len(args.biases) * args.games
                rate = done / (time.perf_counter() - started)
                print(f"{done}/{total} games, {rate * 3600:.0f} games/h, "
                      f"eta {(total - done) / rate / 60:.0f} min", flush=True)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "incomplete"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for b in results:
            results[b].sort()
        report["summary"] = summarize(results, args.biases, args.players)
        report["games"] = {str(b): rows for b, rows in results.items()}
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.out, report)
    for arm in report["summary"]["arms"]:
        lo, hi = arm["ci95"]
        line = (f"bias {arm['bias']:+.3f}: {arm['win_rate']:.3f} "
                f"[{lo:.3f}, {hi:.3f}] over {arm['games']}")
        p = arm.get("paired_vs_bias0")
        if p:
            a, b = p["simultaneous_ci95"]
            line += (f"   vs bias 0: {100 * p['difference']:+.2f} pt "
                     f"[{100 * a:+.2f}, {100 * b:+.2f}]")
        print(line)


if __name__ == "__main__":
    main()
