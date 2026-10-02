"""Do the net's choices hold up on positions from the user's BGA games?

The residual control (residual_control.py) found one human-specific bias:
right after a human opponent stops, the net's instinct rates the viewer's
position ~0.6 points too high (the boards humans stop on are better for them
than the net thinks). The September headroom test found no decision gains
from deeper search -- but on self-play positions only.

This re-runs that test (decision_headroom.py's machinery, unchanged) on the
viewer's own decisions in logged BGA games: each candidate is played to the
END of the game many times (the net continues for every seat, shared dice
across candidates, dice-luck correction), so a choice is scored by real
outcomes instead of the net's instinct at the end of the turn. Actions are
selected on the first half of the samples and scored on the second; a
positive held-out gain means deeper play would have chosen differently and
better. Decisions are split by whether the turn began right after the
opponent's stop (the biased boards) or not.

    python -m games.cantstop.bga_headroom --checkpoint <net.pt> \\
        --log-dir ../boardgame-ai/runs/cantstop/bga_game_log --out runs/bga_headroom.json
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from .adaptive_search import atomic_json, source_identity
from .decision_headroom import candidate_keys, run_rollouts
from .decision_pilot import derived_seed, stage_of, variant_id
from .decision_search import TurnTableBackend, actions
from .engine import Phase, RuleSet
from .experiment import file_sha256, identity
from .luck import DEFAULT_LOG_DIR, after_move, at, load_game, read_turns
from .snapshot import from_snapshot, snapshot

AFTER_OPP_STOP, OTHER = "after_opponent_stop", "other_turns"
ROLLED_AHEAD = "rolled_while_ahead"


def collect_bga(log_dir, rules, viewer_only_logs=False):
    """Every viewer decision with a real choice, from fully logged games
    (and, with ``viewer_only_logs``, from logs without opponent turns: the
    viewer's own turns are still exact there; only the category is not)."""
    rows = []
    for path in sorted(Path(log_dir).glob("table_*.jsonl")):
        try:
            game = load_game(path)
        except ValueError:
            continue
        if (not (game.opponents_logged or viewer_only_logs)
                or game.captures[0].state.rules != rules):
            continue
        turns = read_turns(path, game)
        v = game.viewer_seat
        for ti, turn in enumerate(turns):
            if turn.seat != v:
                continue
            prev = turns[ti - 1] if ti else None
            category = (AFTER_OPP_STOP if prev is not None and prev.seat != v
                        and prev.end == "stop" and turn.link == "adjacent" else OTHER)
            for si, step in enumerate(turn.steps):
                if step.dice is None or step.runners is None:
                    continue
                rolled = at(turn.start, step.runners, Phase.AWAIT_MOVE, step.dice)
                states = [("dice", rolled, None if step.move is None else f"move:{','.join(map(str, step.move))}")]
                if step.move is not None and step.then:
                    states.append(("stop_roll", after_move(rolled, step.move), step.then))
                for phase, state, played in states:
                    if len(actions(state)) < 2:
                        continue
                    rows.append({"id": f"{game.table_id}_t{ti}_s{si}_{phase}",
                                 "variant": variant_id(rules), "stage": stage_of(state),
                                 "phase": phase, "runners": len(state.runners),
                                 "snapshot": snapshot(state), "category": category,
                                 "played": played,
                                 "origin": {"game_seed": int(game.table_id), "turn": ti,
                                            "table_id": game.table_id}})
    return rows


def rolled_while_ahead(rows, evaluator, min_win=0.70, min_bust=0.15):
    """Stop/roll decisions where the advisor rolled while the viewer was at
    ``min_win`` or better and the next roll busts ``min_bust`` or more."""
    from .luck import bust_probability
    from .residual_control import SolverCache
    solvers, out = SolverCache(evaluator), []
    for r in rows:
        if r["phase"] != "stop_roll":
            continue
        state = from_snapshot(r["snapshot"])
        start = state.clone()
        start.runners, start.phase = {}, Phase.AWAIT_ROLL
        sv, rv = solvers(start).stop_roll(state)
        if sv is None or rv is None:
            continue
        me, p = state.active_player, bust_probability(start, state.runners)
        if rv[me] > sv[me] and rv[me] >= min_win and p >= min_bust:
            out.append({**r, "category": ROLLED_AHEAD, "win": float(rv[me]), "bust_odds": p})
    return out


def pick(rows, seed, per_category):
    rng = random.Random(seed)
    out = []
    for cat, n in per_category.items():
        pool = [r for r in rows if r["category"] == cat]
        out += rng.sample(pool, min(n, len(pool)))
    return out


def category_summary(positions):
    lines = ["| Turns | Decisions | Choice changed | Mean held-out gain (pp) | +/- (95%) |",
             "|---|---:|---:|---:|---:|"]
    out = {}
    for cat in sorted({r["category"] for r in positions}) + ["all"]:
        rs = [r for r in positions if "result" in r and (cat == "all" or r["category"] == cat)]
        if not rs:
            continue
        gains = np.array([r["result"]["heldout_gain"] for r in rs])
        se = float(np.sqrt(sum(r["result"]["heldout_mc_se"] ** 2 for r in rs)) / len(rs))
        out[cat] = {"decisions": len(rs), "changed": sum(r["result"]["changed"] for r in rs),
                    "mean_gain": float(gains.mean()), "mc_se": se}
        lines.append(f"| {cat} | {len(rs)} | {out[cat]['changed']} | "
                     f"{100 * gains.mean():+.3f} | {100 * 1.96 * se:.3f} |")
    changed = [r for r in positions if "result" in r and r["result"]["changed"]]
    if changed:
        lines += ["", "Decisions where held-out play preferred another action:", "",
                  "| Position | Turns | Stage | Advisor | Preferred | Held-out gain (pp) |",
                  "|---|---|---|---|---|---:|"]
        for r in sorted(changed, key=lambda r: -r["result"]["heldout_gain"]):
            lines.append(f"| {r['id']} | {r['category']} | {r['stage']} | {r['baseline']['selected']} | "
                         f"{r['result']['selected_on_first_half']} | "
                         f"{100 * r['result']['heldout_gain']:+.2f} +/- {100 * 1.96 * r['result']['heldout_mc_se']:.2f} |")
    return out, "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026100202)
    p.add_argument("--after-opponent-stop", type=int, default=50)
    p.add_argument("--other", type=int, default=30)
    p.add_argument("--focus", choices=("categories", "rolled-ahead"), default="categories",
                   help="rolled-ahead: only stop/roll decisions where the advisor rolled while "
                        "the viewer was at 70%%+ and the next roll busts 15%%+")
    p.add_argument("--rolled-ahead", type=int, default=60)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--in-flight", type=int, default=64)
    p.add_argument("--max-rows", type=int, default=1_000_000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if args.out.exists() != args.resume:
        p.error("choose a new output or resume an existing output")
    rules = RuleSet(2, 5, False)            # the user's BGA variant
    sources = source_identity()
    for name in ("decision_headroom.py", "bga_headroom.py", "pool_rollout.py", "luck.py", "bga_packets.py"):
        sources[name] = file_sha256(Path(__file__).with_name(name))
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
           if k not in ("out", "resume", "checkpoint")}
    cfg.update(horizon="full_game", continuation="baseline", common_random_numbers=True,
               dice_luck=True, candidate_limit=3, selection="first half; baseline wins exact ties")
    signature = hashlib.sha256(json.dumps({"configuration": cfg, "source": sources,
                               "checkpoint": file_sha256(args.checkpoint)}, sort_keys=True).encode()).hexdigest()
    report = json.loads(args.out.read_text()) if args.resume else None
    if report and report["signature"] != signature:
        p.error("resume identity differs")
    from .model import NetEvaluator, load_net
    evaluator = NetEvaluator(load_net(args.checkpoint, device=args.device), device=args.device)
    if report is None:
        if args.focus == "rolled-ahead":
            rows = rolled_while_ahead(collect_bga(args.log_dir, rules, viewer_only_logs=True), evaluator)
            per_category = {ROLLED_AHEAD: args.rolled_ahead}
        else:
            rows = collect_bga(args.log_dir, rules)
            per_category = {AFTER_OPP_STOP: args.after_opponent_stop, OTHER: args.other}
        counts = defaultdict(int)
        for r in rows:
            counts[r["category"]] += 1
        print(f"collected {len(rows)} decisions: {dict(counts)}", flush=True)
        chosen = pick(rows, args.seed, per_category)
        report = {"format": "cantstop-bga-headroom-v1", "status": "running", "signature": signature,
                  "configuration": cfg, "meta": identity(nets={"shared": args.checkpoint}, source_sha256=sources),
                  "collected": dict(counts), "positions": [], "turn_starts": [],
                  "consistency": {"rows": []}, "invocations": []}
        for row in chosen:
            baseline = TurnTableBackend(evaluator).evaluate(from_snapshot(row["snapshot"])).to_dict()
            token = hashlib.sha256(row["id"].encode()).hexdigest()[:20]
            report["positions"].append({**row, "baseline": baseline, "candidates": candidate_keys(baseline),
                "legal_actions": len(baseline["options"]),
                "rollout_seed": derived_seed(args.seed, "bga-headroom", row["id"]),
                "sample_file": str(args.out.with_suffix(".samples") / (token + ".json"))})
    report["status"] = "running"
    report["invocations"].append({"resume": args.resume})
    started = time.perf_counter()
    atomic_json(args.out, report)
    try:
        run_rollouts(report, evaluator, args.out)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "incomplete"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["invocations"][-1]["wall_seconds"] = time.perf_counter() - started
        report["by_category"], table = category_summary(report["positions"])
        atomic_json(args.out, report)
        args.out.with_suffix(".md").write_text(
            "# Decision check on BGA positions\n\nStatus: " + report["status"] + ".\n\n" + table + "\n",
            encoding="utf-8")
    print(f"{report['status']}: {args.out}", flush=True)
    print(table)


if __name__ == "__main__":
    main()
