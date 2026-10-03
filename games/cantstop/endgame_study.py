"""How good are the net's late-game win chances and decisions, judged exactly?

Every late turn small enough for the exact end-game solver (endgame.py) is
scored with no net involved:

  win %     at each solvable turn start: the net's instinct and the
            advisor's displayed value (exact turn solve over net leaves)
            against the exact value, signed for the seat in question
  decisions every choice inside those turns -- pairing, then stop/roll --
            the exact win chance lost by the advisor's choice against the
            exact best, split into too aggressive (rolled where stopping
            was better) and too timid (stopped where rolling was better)

Sources: the user's BGA games (2 players, 5 columns, no blocking; the
viewer's decisions there are the advisor's in ~97% of cases, so the
advisor's own choice is recomputed and compared), and simulated net-vs-net
games in the same variant for a larger sample.

    python -m games.cantstop.endgame_study --checkpoint <net.pt> \\
        --log-dir ../boardgame-ai/runs/cantstop/bga_game_log --selfplay 150
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .advisor_adapter import DEFAULT_CHECKPOINT, wins_on_stop
from .endgame import ExactEndgame, TooLarge
from .engine import (GameState, Phase, RuleSet, apply_move, can_stop,
                     random_dice, roll, stop)
from .luck import (DEFAULT_LOG_DIR, Step, Turn, after_move, at,
                   bust_probability, load_game, read_turns, turn_start)
from .residual_control import SolverCache

RULES = RuleSet(2, 5, False)


def selfplay_turns(solvers, rng):
    """One net-vs-net game, recorded as luck.py turns."""
    state, turns = GameState(RULES), []
    while not state.game_over:
        start = turn_start(state)
        turn = Turn(state.active_player, start, [], "adjacent" if turns else "first")
        turns.append(turn)
        solver = solvers(start)
        while True:
            step = Step(dict(state.runners), None)
            turn.steps.append(step)
            if not roll(state, random_dice(rng)):
                turn.end = "bust"
                break
            step.dice = state.dice
            step.move = solver.choose_move(state)
            apply_move(state, step.move)
            if wins_on_stop(state) or (can_stop(state) and solver.should_stop(state)):
                stop(state)
                step.then, turn.end = "stop", ("win" if state.game_over else "stop")
                break
            step.then = "roll"
    return turns


def score_game(turns, solvers, evaluate, source, game_id, seat_of_interest, budget):
    """Walk back from the last turn while the end game stays solvable."""
    eg, boards, decisions = ExactEndgame(budget=budget), [], []
    for ti in range(len(turns) - 1, -1, -1):
        turn = turns[ti]
        start = turn.start
        try:
            exact_start = eg.board_value(start)
        except TooLarge:
            break
        net_solver = solvers(start)
        exact = eg.turn_solver(start)
        prev = turns[ti - 1] if ti else None
        after_opp_stop = bool(prev and prev.seat != turn.seat and prev.end == "stop")
        seats = range(2) if seat_of_interest is None else [seat_of_interest]
        for seat in seats:
            boards.append({"source": source, "game": game_id, "turn": ti, "seat": seat,
                           "to_move": turn.seat == seat, "after_opp_stop": after_opp_stop,
                           "exact": float(exact_start[seat]),
                           "net": float(evaluate([start])[0][seat]),
                           "advisor": float(net_solver.value(start)[seat]),
                           "levels": eg.levels})
        if seat_of_interest is not None and turn.seat != seat_of_interest:
            continue
        me = turn.seat
        for step in turn.steps:
            if step.dice is None or step.runners is None:
                continue
            rolled = at(start, step.runners, Phase.AWAIT_MOVE, step.dice)
            # pairing: the advisor's move vs the exact best move, each followed
            # by exact play
            adv_move = net_solver.choose_move(rolled)
            ex_best = exact.value(rolled)[me]
            ex_adv = exact.value(after_move(rolled, adv_move))[me]
            decisions.append({"source": source, "game": game_id, "turn": ti, "kind": "pairing",
                              "win": float(ex_best), "regret": float(ex_best - ex_adv),
                              "after_opp_stop": after_opp_stop})
            # stop/roll at the position actually reached
            if step.move is None or not step.then:
                continue
            child = after_move(rolled, step.move)
            esv, erv = exact.stop_roll(child)
            nsv, nrv = net_solver.stop_roll(child)
            if esv is None or erv is None or nsv is None or nrv is None:
                continue
            advisor_rolls = nrv[me] > nsv[me]
            exact_rolls = erv[me] > esv[me]
            best = max(esv[me], erv[me])
            chosen = erv[me] if advisor_rolls else esv[me]
            decisions.append({"source": source, "game": game_id, "turn": ti, "kind": "stop_roll",
                              "win": float(best), "regret": float(best - chosen),
                              "advisor": "roll" if advisor_rolls else "stop",
                              "exact": "roll" if exact_rolls else "stop",
                              "bust_odds": bust_probability(start, child.runners),
                              "after_opp_stop": after_opp_stop})
    return boards, decisions


def report(boards, decisions):
    lines = []
    for source in sorted({b["source"] for b in boards}):
        bs = [b for b in boards if b["source"] == source]
        lines.append(f"\n== {source}: {len({b['game'] for b in bs})} games, "
                     f"{len({(b['game'], b['turn']) for b in bs})} solvable late turn starts ==")
        lines.append("Win % vs exact (pts, seat in question; + = too optimistic):")
        for name, keep in (("all", lambda b: True),
                           ("seat to move", lambda b: b["to_move"]),
                           ("seat waiting", lambda b: not b["to_move"]),
                           ("right after the opponent stopped, seat to move",
                            lambda b: b["to_move"] and b["after_opp_stop"])):
            sel = [b for b in bs if keep(b)]
            if not sel:
                continue
            net = np.array([b["net"] - b["exact"] for b in sel]) * 100
            adv = np.array([b["advisor"] - b["exact"] for b in sel]) * 100
            lines.append(f"  {name:<50} n={len(sel):5d}  net {net.mean():+.2f} (mean abs {np.abs(net).mean():.2f})"
                         f"   advisor {adv.mean():+.2f} (mean abs {np.abs(adv).mean():.2f})")
        ds = [d for d in decisions if d["source"] == source]
        lines.append("Decisions vs exact (win % lost by the advisor's choice):")
        for kind in ("pairing", "stop_roll"):
            sel = [d for d in ds if d["kind"] == kind]
            if not sel:
                continue
            r = np.array([d["regret"] for d in sel]) * 100
            wrong = r > 1e-7
            lines.append(f"  {kind:<10} n={len(sel):5d}  different from exact best: {wrong.sum():4d} "
                         f"({100 * wrong.mean():.1f}%)  mean loss {r.mean():.3f} pts/decision, "
                         f"worst {r.max():.2f}")
        sr = [d for d in ds if d["kind"] == "stop_roll"]
        if sr:
            per_game = defaultdict(float)
            for d in ds:
                per_game[d["game"]] += d["regret"] * 100
            lines.append(f"  total loss per game's solvable end game: {np.mean(list(per_game.values())):.2f} pts")
            lines.append("  stop/roll mistakes by the mover's exact win %:")
            for lo, hi in ((0, .2), (.2, .5), (.5, .8), (.8, 1.01)):
                sel = [d for d in sr if lo <= d["win"] < hi]
                if not sel:
                    continue
                aggressive = [d for d in sel if d["advisor"] == "roll" and d["exact"] == "stop"]
                timid = [d for d in sel if d["advisor"] == "stop" and d["exact"] == "roll"]
                lines.append(f"    {int(100 * lo):>3}-{min(100, int(100 * hi)):<3}%: n={len(sel):4d}  "
                             f"too aggressive {len(aggressive):3d} (loss {100 * sum(d['regret'] for d in aggressive):.2f})  "
                             f"too timid {len(timid):3d} (loss {100 * sum(d['regret'] for d in timid):.2f})")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--selfplay", type=int, default=150)
    ap.add_argument("--budget", type=int, default=10_000, help="levels per game's end game")
    ap.add_argument("--seed", type=int, default=2026100203)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True,
                    help="results so far are saved here as the study runs; rerunning with the "
                         "same --out resumes, skipping finished games")
    args = ap.parse_args(argv)
    from .model import NetEvaluator, load_net
    evaluate = NetEvaluator(load_net(args.checkpoint, device=args.device), device=args.device)
    # Reuse is within one game (play, then score its last turns), so a
    # small cache suffices; each entry is ~8 MB (measured: 128 entries held
    # 1.56 GB; the old cap of 2000 grew until the system ran out of memory).
    solvers = SolverCache(evaluate, max_size=48)
    saved = json.loads(args.out.read_text()) if args.out.exists() else {}
    boards, decisions = saved.get("boards", []), saved.get("decisions", [])
    done = set(saved.get("done", []))
    t0 = time.time()

    def save():
        text = report(boards, decisions)
        tmp = args.out.with_name(args.out.name + ".tmp")
        tmp.write_text(json.dumps({"boards": boards, "decisions": decisions, "done": sorted(done),
                                   "budget": args.budget, "seed": args.seed}))
        tmp.replace(args.out)
        args.out.with_suffix(".md").write_text(
            f"# End-game study (exact solver)\n\n{len(done)} games done.\n```\n{text}\n```\n",
            encoding="utf-8")
        return text

    if "bga" not in done:
        for path in sorted(args.log_dir.glob("table_*.jsonl")):
            try:
                game = load_game(path)
            except ValueError:
                continue
            if game.captures[0].state.rules != RULES:
                continue
            turns = read_turns(path, game)
            b, d = score_game(turns, solvers, evaluate, "BGA (your seat)", game.table_id,
                              game.viewer_seat, args.budget)
            boards += b
            decisions += d
        done.add("bga")
        save()
        print(f"BGA done in {time.time() - t0:.0f} s", flush=True)
    for g in range(args.selfplay):
        if f"sp{g}" in done:
            continue
        # One seed per game, so a resumed run plays the same games.
        turns = selfplay_turns(solvers, random.Random(args.seed * 1_000_003 + g))
        b, d = score_game(turns, solvers, evaluate, "self-play (both seats)", f"sp{g}", None, args.budget)
        boards += b
        decisions += d
        done.add(f"sp{g}")
        if (g + 1) % 10 == 0:
            save()
            print(f"self-play {g + 1}/{args.selfplay}, {time.time() - t0:.0f} s", flush=True)
    print(save())


if __name__ == "__main__":
    main()
