"""Forced stop-vs-roll benchmark: is the net's stop/roll choice right?

Plan review (2026-09-26): the open question is how well end-of-turn values
RANK stopping against rolling on. This measures it directly:

1. **Collect** real decision states (AWAIT_DECISION, stopping allowed, not a
   winning stop) from self-play with a frozen net. For each, record the
   solver's stop value, roll-on value, margin and choice for the mover.
2. **Stratify**: `random` (a uniform sample of all decisions -- the
   ordinary-play estimate), `small_margin` (|stop - roll| < 2 pts),
   `threat` (the next player can win in their very next turn with >= 10%),
   `contested` (a runner shares a column with an opponent's saved progress),
   and `disagree` (a second net chooses differently). Occurrence rates are
   kept; non-random strata are reported separately, never pooled as if
   ordinary.
3. **Force both actions** and play each to the end: *stop now* (bank, then
   the next player moves) vs *roll once now* (later decisions in this turn
   stay free -- not "never stop again"). Every seat is played by a frozen,
   identified controller; both branches of replicate j share its seed
   (common random numbers; the empirical pairing gain is reported, not
   assumed).
4. **Grade without reusing noise**: replicates split into halves A and B.
   A picks the better action; B grades it. A decision is *materially wrong*
   when the solver's choice differs from A's pick and B confirms the loss
   exceeds `--delta`. The reported loss is B's estimate.

The truth here is the value under the named continuation controller, not
perfect play. Agreement that grows with a stronger controller, or with a
full-round lookahead (not built yet), would point at the evaluator's
boundary; flips that vanish with more rollouts were noise.

    python -m games.cantstop.decision_benchmark --net runs/td0_personas/iter_0120.pt \
        --games 60 --per-stratum 40 --reps 200 --out runs/decisions.json
"""

import argparse
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .engine import (GameState, Phase, RuleSet, apply_move, can_stop,
                     random_dice, roll, stop)
from .experiment import identity, write_json
from .model import NetEvaluator, load_net
from .portable_rng import PortableRng
from .rust_pool import game_seeds, run_pool
from .rust_solver import RustTurnSolver
from .self_play import PLAIN, Search, play_turn
from .snapshot import from_snapshot, snapshot
from .solver import TurnSolver

STRATA = ("random", "small_margin", "threat", "contested", "disagree")


@dataclass
class Decision:
    snap: tuple                 # the AWAIT_DECISION state
    game: int
    mover: int
    stop_value: float           # mover's win probability, solver's view
    roll_value: float
    chose_stop: bool
    threat: float               # P(next player wins in their next turn)
    contested: bool
    disagree: bool = False
    strata: list = field(default_factory=list)

    @property
    def margin(self):
        return self.stop_value - self.roll_value


def _zero_evaluator(boards):
    """Scores every non-winning end-of-turn board 0: the solver's root is
    then exactly P(the mover wins before this turn ends)."""
    return np.zeros((len(boards), boards[0].rules.num_players))


def immediate_threat(state_after_stop):
    """P(the player to move wins during this turn) -- the review's
    immediate-threat bound, from an exact turn solve."""
    if state_after_stop.game_over:
        return 0.0
    ps = RustTurnSolver(state_after_stop, _zero_evaluator)
    return float(ps.value(state_after_stop)[state_after_stop.active_player])


def _contested(state):
    me = state.active_player
    return any(prog[c] > 0 for p, prog in enumerate(state.progress) if p != me
               for c in state.runners)


def collect(ev, rules, games, seed, compare_ev=None, max_turns=400):
    """Decision states from ``games`` self-play games with ``ev``."""
    out = []
    seeds = game_seeds(PortableRng(seed), games)
    for g in range(games):
        rng = PortableRng(seeds[g])
        state = GameState(rules)
        turns = 0
        while not state.game_over and turns < max_turns:
            if not roll(state, random_dice(rng)):
                turns += 1
                continue
            solver = RustTurnSolver(state, ev)
            other = RustTurnSolver(state, compare_ev) if compare_ev else None
            while True:
                apply_move(state, solver.choose_move(state))
                if can_stop(state):
                    stop_v, roll_v = solver.stop_roll(state)
                    if stop_v is not None and roll_v is not None:
                        a = state.active_player
                        after = state.clone()
                        stop(after)
                        if not after.game_over:
                            out.append(Decision(
                                snap=snapshot(state), game=g, mover=a,
                                stop_value=float(stop_v[a]),
                                roll_value=float(roll_v[a]),
                                chose_stop=bool(solver.should_stop(state)),
                                threat=immediate_threat(after),
                                contested=_contested(state),
                                disagree=(other is not None and
                                          other.should_stop(state)
                                          != solver.should_stop(state))))
                    if solver.should_stop(state):
                        stop(state)
                        break
                if not roll(state, random_dice(rng)):
                    break
            turns += 1
    return out


def stratify(decisions, per_stratum, seed, small=0.02, threat=0.10):
    """Tag and sample: each stratum draws up to ``per_stratum`` of its
    members uniformly without replacement. Returns (sample, occurrence)."""
    members = {
        "random": list(range(len(decisions))),
        "small_margin": [i for i, d in enumerate(decisions)
                         if abs(d.margin) < small],
        "threat": [i for i, d in enumerate(decisions) if d.threat >= threat],
        "contested": [i for i, d in enumerate(decisions) if d.contested],
        "disagree": [i for i, d in enumerate(decisions) if d.disagree],
    }
    occurrence = {k: len(v) / max(1, len(decisions)) for k, v in members.items()}
    chosen = {}
    for k, idx in members.items():
        order = list(idx)
        PortableRng(seed + STRATA.index(k)).shuffle(order)
        for i in order[:per_stratum]:
            chosen.setdefault(i, []).append(k)
    sample = []
    for i in sorted(chosen):
        d = decisions[i]
        d.strata = chosen[i]
        sample.append(d)
    return sample, occurrence


def _branches(d):
    state = from_snapshot(d.snap)
    after = state.clone()
    stop(after)
    return after, state              # stop branch start, roll branch start


def rollout(decisions, controller, reps, seed, opponent_search=PLAIN,
            chunk=16, progress=None):
    """For each decision, win indicators for the mover in both branches:
    arrays (reps,) for stop and roll, replicates paired by seed."""
    assert reps % 2 == 0, "reps must be even (halves A and B)"
    results = []
    for c0 in range(0, len(decisions), chunk):
        batch = decisions[c0:c0 + chunk]
        starts, seeds, seating_idx = [], [], []
        for k, d in enumerate(batch):
            s_stop, s_roll = _branches(d)
            rep_seeds = game_seeds(PortableRng(seed * 1_000_003 + c0 + k), reps)
            for branch in (s_stop, s_roll):
                starts.extend([branch] * reps)
                seeds.extend(rep_seeds)
            seating_idx.append(d.mover)
        n = batch[0].snap[0][0]
        rules_list = [s.rules for s in starts]
        searches = list(dict.fromkeys([PLAIN, opponent_search]))
        seating = []
        for k, d in enumerate(batch):
            row = [searches.index(opponent_search)] * n
            row[d.mover] = searches.index(PLAIN)
            seating.extend([row] * (2 * reps))
        res = run_pool(rules_list, seeds, [controller], starts=starts,
                       searches=searches, search_seating=seating)
        for k, d in enumerate(batch):
            base = 2 * reps * k
            wins = np.array([r.winner == d.mover
                             for r in res[base:base + 2 * reps]], dtype=float)
            results.append((wins[:reps], wins[reps:]))
        if progress:
            progress(c0 + len(batch), len(decisions))
    return results


def grade(decision, stop_wins, roll_wins, delta):
    """Select with half A, grade with half B."""
    h = len(stop_wins) // 2
    diff = roll_wins - stop_wins                     # paired, per replicate
    a, b = slice(0, h), slice(h, None)
    best_a_roll = diff[a].mean() > 0
    chose_roll = not decision.chose_stop
    value_b = {"stop": stop_wins[b].mean(), "roll": roll_wins[b].mean()}
    best_a = "roll" if best_a_roll else "stop"
    chosen = "roll" if chose_roll else "stop"
    loss_b = value_b[best_a] - value_b[chosen]
    se_paired = diff.std(ddof=1) / math.sqrt(len(diff))
    se_unpaired = math.sqrt(stop_wins.var(ddof=1) / len(stop_wins)
                            + roll_wins.var(ddof=1) / len(roll_wins))
    return {
        "margin": decision.margin, "chose_stop": decision.chose_stop,
        "diff_all": float(diff.mean()), "diff_a": float(diff[a].mean()),
        "diff_b": float(diff[b].mean()),
        "loss_b": float(loss_b),
        "wrong": bool(best_a != chosen and loss_b > delta),
        "se_paired": float(se_paired), "se_unpaired": float(se_unpaired),
    }


def summarize(sample, graded):
    out = {}
    for k in STRATA:
        rows = [g for d, g in zip(sample, graded) if k in d.strata]
        if not rows:
            continue
        losses = np.array([g["loss_b"] for g in rows])
        games = {d.game for d in sample if k in d.strata}
        pred = np.array([-g["margin"] for g in rows])      # predicted roll - stop
        real = np.array([g["diff_all"] for g in rows])
        out[k] = {
            "n": len(rows), "source_games": len(games),
            "mean_abs_margin": float(np.mean(np.abs([g["margin"] for g in rows]))),
            "chose_stop": float(np.mean([g["chose_stop"] for g in rows])),
            "wrong_fraction": float(np.mean([g["wrong"] for g in rows])),
            "mean_loss_b": float(losses.mean()),
            "mean_loss_b_se": float(losses.std(ddof=1) / math.sqrt(len(rows)))
            if len(rows) > 1 else None,
            "pred_vs_real_corr": float(np.corrcoef(pred, real)[0, 1])
            if len(rows) > 2 and pred.std() > 0 and real.std() > 0 else None,
            "pairing_se_ratio": float(np.mean([g["se_paired"] / g["se_unpaired"]
                                               for g in rows
                                               if g["se_unpaired"] > 0])),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--net", required=True, help="the net whose decisions are graded")
    ap.add_argument("--controller", default=None,
                    help="net playing every seat in the rollouts (default: --net)")
    ap.add_argument("--compare-net", default=None,
                    help="second net: tags decisions where it chooses differently")
    ap.add_argument("--opponent-bias", type=float, default=0.0,
                    help="stop bias for the mover's opponents in the rollouts")
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--blocking", action="store_true")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--per-stratum", type=int, default=40)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load = lambda p: NetEvaluator(load_net(p, device=device), device=device)
    ev = load(args.net)
    controller = load(args.controller) if args.controller else ev
    compare = load(args.compare_net) if args.compare_net else None
    rules = RuleSet.make(args.players, args.extended, args.blocking)
    nets = {"graded": args.net}
    if args.controller:
        nets["controller"] = args.controller
    if args.compare_net:
        nets["compare"] = args.compare_net
    meta = identity(nets=nets, rules=str(rules), games=args.games,
                    per_stratum=args.per_stratum, reps=args.reps,
                    delta=args.delta, seed=args.seed,
                    opponent_search=Search(stop_bias=args.opponent_bias),
                    controller_search=PLAIN)

    t = time.perf_counter()
    decisions = collect(ev, rules, args.games, args.seed, compare)
    sample, occurrence = stratify(decisions, args.per_stratum, args.seed)
    print(f"{len(decisions)} decisions, {len(sample)} sampled "
          f"({time.perf_counter() - t:.0f}s); occurrence "
          f"{ {k: round(v, 3) for k, v in occurrence.items()} }", flush=True)
    t = time.perf_counter()
    wins = rollout(sample, controller, args.reps, args.seed + 1,
                   Search(stop_bias=args.opponent_bias),
                   progress=lambda i, n: print(f"  rollouts {i}/{n} "
                                               f"({time.perf_counter() - t:.0f}s)",
                                               flush=True))
    graded = [grade(d, s, r, args.delta) for d, (s, r) in zip(sample, wins)]
    report = {"meta": meta, "decisions_total": len(decisions),
              "occurrence": occurrence, "strata": summarize(sample, graded),
              "rollout_seconds": round(time.perf_counter() - t),
              "rows": [{"game": d.game, "strata": d.strata, "threat": d.threat,
                        "contested": d.contested, "disagree": d.disagree,
                        **g} for d, g in zip(sample, graded)]}
    write_json(args.out, report)
    for k, v in report["strata"].items():
        print(f"{k:13} n={v['n']:3} wrong {v['wrong_fraction']:.3f}  "
              f"loss {v['mean_loss_b']:+.4f} (se {v['mean_loss_b_se'] or 0:.4f})  "
              f"|margin| {v['mean_abs_margin']:.3f}  pairing se ratio "
              f"{v['pairing_se_ratio']:.2f}", flush=True)


# ---- Python reference for the mid-turn start (tests) ----

def continue_game(state, evaluate, rng, max_turns=400):
    """Finish ``state``'s game in Python from AWAIT_DECISION by rolling on
    (the roll branch), exactly as the Rust pool does from that snapshot."""
    state = state.clone()
    turns = 0
    if state.phase == Phase.AWAIT_DECISION:
        if roll(state, random_dice(rng)):
            solver = TurnSolver(state, evaluate)
            while True:
                apply_move(state, solver.choose_move(state))
                if can_stop(state) and solver.should_stop(state):
                    stop(state)
                    break
                if not roll(state, random_dice(rng)):
                    break
        turns += 1
    while not state.game_over and turns < max_turns:
        play_turn(state, evaluate, rng)
        turns += 1
    return state


if __name__ == "__main__":
    main()
