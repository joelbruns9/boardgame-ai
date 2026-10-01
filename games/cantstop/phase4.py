"""Phase 4: the all-variant generalist training loop.

Built on `train.py`'s pieces (generation, replay buffer, training steps,
schedules), with what the plan and its review require for a multi-night,
ten-variant run:

* **row-balanced schedule** (`schedule.RowSchedule`): ~equal rows per
  variant per iteration, games per variant adapted to measured lengths;
* **per-variant logs**: games, rows, turns, turn length, seat wins;
* **per-variant evaluation** every ``eval_every`` iterations, against the
  heuristic, the iteration-0 net (a permanent reference) and, once reached,
  a fixed later checkpoint (``reference_iter``) -- complete seat cycles;
* **probe monitor**: V vs one exact turn of search on fixed boards per
  variant, with a pre-declared alert threshold;
* **staged targets** (``exact_from``): sampled TD values until that
  iteration, exact after -- run it as a controlled branch, not a global
  trigger; the buffer's sampled/exact mixture is logged;
* **true resume**: ``state.pt`` holds the net, optimizer, replay buffer
  (with per-row variant / game / target-mode metadata -- also the dataset
  `capacity_probe --data` reads), RNG states, schedule, probes and
  iteration; ``--resume`` continues the same run exactly;
* **delayed personas** (``personas_from``): plain self-play until that
  iteration. An untrained net's values are nearly flat, so the aggressive
  persona (stop_bias < 0) always prefers rolling and never banks.

    python -m games.cantstop.phase4 --out runs/p4_pilot --iterations 100 \\
        --rows-per-variant 4000 --td-lambda 0 --conservative 0.2 --aggressive 0.1 \\
        --lr-schedule 1:1e-3 40:3e-4 80:1e-4 120:5e-5
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .engine import ALL_RULESETS
from .experiment import identity, write_json
from .model import CantStopNet, NetEvaluator, load_net, save_net
from .schedule import RowSchedule
from .self_play import (PLAIN, Search, TurnLimitExceeded,
                        stack_training, summarize)
from .train import (ReplayBuffer, generate, lr_at, parse_lr_schedule,
                    steps_for_passes, train_steps)
from .variant_eval import ProbeSet, evaluate_variants

# Share of each variant's attempted games allowed to hit the turn limit (dropped).
MAX_UNFINISHED = 0.05

# Arguments that define the run; a resume must not change them.
FIXED = ("rule_sets", "rows_per_variant", "hidden", "batch_size", "passes",
         "replay_window", "td_lambda", "exact_from", "conservative",
         "aggressive", "persona_bias", "personas_from", "reflect_augment",
         "seed")


def _check_unfinished(results, iteration):
    """A healthy variant must not hide stalling in another variant."""
    attempted, unfinished = {}, {}
    for result in results:
        key = str(result.rules)
        attempted[key] = attempted.get(key, 0) + 1
        if result.winner < 0:
            unfinished[key] = unfinished.get(key, 0) + 1
    failures = [f"{key}: {count}/{attempted[key]} ({count/attempted[key]:.1%})"
                for key, count in unfinished.items()
                if count > MAX_UNFINISHED * attempted[key]]
    if failures:
        raise TurnLimitExceeded(
            f"iteration {iteration}: turn-limit rate exceeds {MAX_UNFINISHED:.0%} "
            f"per variant: {'; '.join(failures)}")
    if not attempted:
        raise TurnLimitExceeded(f"iteration {iteration}: no games generated")
    return attempted, unfinished


def _per_variant(results, rule_sets):
    out = {}
    for rules in rule_sets:
        rs = [r for r in results if r.rules == rules]
        if not rs:
            continue
        s = summarize(rs)
        s["seat_wins"] = {str(k): v for k, v in s["seat_wins"].items()}
        out[str(rules)] = s
    return out


def _meta(results, rule_sets, iteration, exact):
    """Per-row metadata in stack_training's row order."""
    usable = [r for r in results if len(r)]
    index = {r: i for i, r in enumerate(rule_sets)}
    variant = np.concatenate([np.full(len(r), index[r.rules], np.int16)
                              for r in usable])
    game = np.concatenate([np.full(len(r), iteration * 1_000_000 + g, np.int64)
                           for g, r in enumerate(usable)])
    mode = np.full(len(variant), 1 if exact else 0, np.int8)
    return {"variant": variant, "game": game, "exact": mode}


def run(out_dir, iterations, rule_sets=ALL_RULESETS, rows_per_variant=4000,
        hidden=(512, 512, 256, 256), lr=1e-3, lr_schedule=None,
        batch_size=256, passes=5.0, replay_window=10, td_lambda=0.0,
        exact_from=None, conservative=0.0, aggressive=0.0, persona_bias=0.03,
        personas_from=None,
        reflect_augment=False, variant_weights=None,
        random_start_fraction=0.0, random_start_turns=8,
        eval_every=5, eval_games=200,
        reference_iter=40, probes_per_variant=50, probe_alert=0.05, seed=0,
        device=None, init_checkpoint=None, threads=0, in_flight=None,
        resume=False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {"rule_sets": [str(r) for r in rule_sets],
              "rows_per_variant": rows_per_variant, "hidden": list(hidden),
              "batch_size": batch_size, "passes": passes,
              "replay_window": replay_window, "td_lambda": td_lambda,
              "exact_from": exact_from, "conservative": conservative,
              "aggressive": aggressive, "persona_bias": persona_bias,
              "personas_from": personas_from,
              "reflect_augment": reflect_augment, "seed": seed,
              # Not FIXED: a resume may change the emphasis and the start mix.
              "variant_weights": {str(k): w for k, w
                                  in (variant_weights or {}).items()},
              "random_start_fraction": random_start_fraction,
              "random_start_turns": random_start_turns}
    device = torch.device(device or ("cuda" if torch.cuda.is_available()
                                     else "cpu"))
    state_path = out / "state.pt"

    if resume:
        st = torch.load(state_path, map_location=device, weights_only=False)
        for k in FIXED:
            # .get: runs saved before a key existed had its default (None).
            if st["config"].get(k) != config[k]:
                raise SystemExit(f"--resume must not change {k}: run has "
                                 f"{st['config'].get(k)!r}, got {config[k]!r}")
        net = CantStopNet(hidden=tuple(st["config"]["hidden"])).to(device)
        net.load_state_dict(st["net"])
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        opt.load_state_dict(st["optimizer"])
        buffer = ReplayBuffer.from_state(st["buffer"])
        rng, torch_rng = random.Random(), random.Random()
        rng.setstate(st["rng"])
        torch_rng.setstate(st["torch_rng"])
        schedule = RowSchedule(rule_sets, rows_per_variant,
                               weights=variant_weights)
        schedule.load(st["schedule"])
        probes = ProbeSet.from_state(st["probes"])
        start = st["iteration"] + 1
        write_json(out / f"run_meta_resume_{start:04d}.json",
                   identity(config=config, resumed_from=st["iteration"],
                            iterations=iterations, lr=lr,
                            lr_schedule=lr_schedule))
    else:
        if state_path.exists():
            raise SystemExit(f"{state_path} exists: use --resume, or a new --out")
        torch.manual_seed(seed)
        rng, torch_rng = random.Random(seed), random.Random(seed + 1)
        net = (load_net(init_checkpoint, device=str(device)) if init_checkpoint
               else CantStopNet(hidden=hidden)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        buffer = ReplayBuffer(window_iterations=replay_window)
        schedule = RowSchedule(rule_sets, rows_per_variant,
                               weights=variant_weights)
        probes = ProbeSet.build(rule_sets, probes_per_variant, seed)
        save_net(net, out / "iter_0000.pt")
        start = 1
        write_json(out / "run_meta.json", identity(
            nets={"init": init_checkpoint} if init_checkpoint else None,
            config=config, iterations=iterations, lr=lr,
            lr_schedule=lr_schedule, eval_every=eval_every,
            eval_games=eval_games, reference_iter=reference_iter,
            probes_per_variant=probes_per_variant, probe_alert=probe_alert,
            note="weights-only warm start" if init_checkpoint else "fresh"))

    def frozen(it):
        return NetEvaluator(load_net(out / f"iter_{it:04d}.pt",
                                     device=str(device)), device=str(device))

    log_path = out / "run.jsonl"
    for it in range(start, iterations + 1):
        it_lr = lr_at(it, lr, lr_schedule)
        for group in opt.param_groups:
            group["lr"] = it_lr
        exact = exact_from is not None and it >= exact_from
        search = Search(exact_root=exact)
        personas = personas_from is None or it >= personas_from
        games = schedule.games()

        started = time.time()
        results = generate(rule_sets, games, NetEvaluator(net, device=str(device)),
                           rng, threads=threads, in_flight=in_flight,
                           search=search,
                           conservative=conservative if personas else 0.0,
                           aggressive=aggressive if personas else 0.0,
                           persona_bias=persona_bias,
                           allow_unfinished=True,
                           random_start_fraction=random_start_fraction,
                           random_start_turns=random_start_turns)
        gen_seconds = time.time() - started
        # An untrained net plays erratically, and the aggressive persona
        # (stop_bias < 0) never banks while its values are flat, so a rare
        # game hits the turn limit. Drop those games (no winner, no
        # targets) rather than abort the run; a policy that stalls often
        # is still a failure.
        attempted, unfinished = _check_unfinished(results, it)
        results = [r for r in results if r.winner >= 0]
        schedule.update(results)

        x, y = stack_training(results, td_lambda)
        buffer.add(x, y, _meta(results, rule_sets, it, exact))
        it_steps = steps_for_passes(len(buffer), passes, replay_window,
                                    batch_size)
        started = time.time()
        loss = train_steps(net, buffer, opt, it_steps, batch_size, device,
                           torch_rng, reflect=reflect_augment)
        train_seconds = time.time() - started
        save_net(net, out / f"iter_{it:04d}.pt")

        exact_share = buffer.meta("exact")
        record = {
            "iteration": it, "lr": it_lr, "loss": loss,
            "exact_targets": exact, "personas": personas,
            "buffer_rows": len(buffer), "window_iterations": buffer.iterations,
            "buffer_exact_share": (float(exact_share.mean())
                                   if exact_share is not None else None),
            "new_rows": len(x), "train_steps": it_steps,
            "gen_seconds": round(gen_seconds, 1),
            "train_seconds": round(train_seconds, 1),
            "games": {str(r): g for r, g in games.items()},
            "variants": _per_variant(results, rule_sets),
            "unfinished": unfinished,
            "attempted": attempted,
            "random_start_fraction": random_start_fraction,
            "variant_weights": config["variant_weights"],
        }

        if eval_every and it % eval_every == 0:
            current = NetEvaluator(net, device=str(device))
            refs = {"iter0": frozen(0)}
            if reference_iter and it > reference_iter:
                refs[f"iter{reference_iter}"] = frozen(reference_iter)
            started = time.time()
            record["eval"] = evaluate_variants(
                current, rule_sets, eval_games, seed * 7919 + it, refs,
                PLAIN, threads, in_flight)
            record["probes"] = probes.measure(current, probe_alert)
            record["eval_seconds"] = round(time.time() - started, 1)

        torch.save({"iteration": it, "config": config,
                    "net": net.state_dict(), "optimizer": opt.state_dict(),
                    "buffer": buffer.state(), "rng": rng.getstate(),
                    "torch_rng": torch_rng.getstate(),
                    "schedule": schedule.state(), "probes": probes.state()},
                   state_path)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        print(json.dumps({k: record[k] for k in
                          ("iteration", "lr", "loss", "new_rows",
                           "gen_seconds", "exact_targets")}), flush=True)
    return net


def parse_variants(items):
    """``["3:4:b", "2:3"]`` -> rule sets: PLAYERS:COLUMNS_TO_WIN[:b]."""
    from .engine import RuleSet
    if not items:
        return ALL_RULESETS
    out = []
    for item in items:
        parts = item.split(":")
        if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] != "b"):
            raise SystemExit(f"variants are PLAYERS:COLUMNS[:b], got {item!r}")
        out.append(RuleSet(int(parts[0]), int(parts[1]), len(parts) == 3))
    return tuple(out)


def parse_weights(items):
    """``["3:4:b=2"]`` -> {(3, 4, True): 2.0}, keyed like the schedule."""
    from .schedule import rules_key
    out = {}
    for item in items or ():
        spec, sep, weight = item.partition("=")
        if not sep:
            raise SystemExit(f"variant weights are P:C[:b]=W, got {item!r}")
        (rules,) = parse_variants([spec])
        out[rules_key(rules)] = float(weight)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--iterations", type=int, required=True,
                   help="total iterations (a resume continues up to this)")
    p.add_argument("--resume", action="store_true",
                   help="continue the run in --out exactly from state.pt")
    p.add_argument("--variants", nargs="+", default=None,
                   metavar="P:C[:b]",
                   help="train only these rule sets, e.g. 3:4:b for the "
                        "3-player, 4-column, blocking specialist (default: "
                        "all ten)")
    p.add_argument("--rows-per-variant", type=int, default=4000)
    p.add_argument("--hidden", type=int, nargs="+",
                   default=[512, 512, 256, 256])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-schedule", nargs="+", default=None, metavar="ITER:LR")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--passes", type=float, default=5.0)
    p.add_argument("--replay-window", type=int, default=10)
    p.add_argument("--td-lambda", type=float, default=0.0)
    p.add_argument("--exact-from", type=int, default=None,
                   help="exact TD targets from this iteration on (default: "
                        "sampled throughout)")
    p.add_argument("--conservative", type=float, default=0.0)
    p.add_argument("--aggressive", type=float, default=0.0)
    p.add_argument("--persona-bias", type=float, default=0.03)
    p.add_argument("--personas-from", type=int, default=None,
                   help="plain self-play before this iteration, personas "
                        "from it on (default: from the first)")
    p.add_argument("--reflect-augment", action="store_true")
    p.add_argument("--variant-weights", nargs="+", default=None,
                   metavar="P:C[:b]=W",
                   help="row emphasis per variant, e.g. 3:4:b=2 (default 1)")
    p.add_argument("--random-start-fraction", type=float, default=0.0,
                   help="share of games starting from a random-play prefix")
    p.add_argument("--random-start-turns", type=int, default=8,
                   help="prefix length: up to this many turns per player")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-games", type=int, default=200,
                   help="per opponent per variant (rounded up to seat cycles)")
    p.add_argument("--reference-iter", type=int, default=40,
                   help="also evaluate against this frozen checkpoint once "
                        "the run passes it (0 = off)")
    p.add_argument("--probes-per-variant", type=int, default=50)
    p.add_argument("--probe-alert", type=float, default=0.05,
                   help="probe RMSE (win-prob) above which a variant is flagged")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--init-checkpoint", default=None)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--in-flight", type=int, default=None)
    args = p.parse_args(argv)
    run(out_dir=args.out, iterations=args.iterations,
        rule_sets=parse_variants(args.variants),
        rows_per_variant=args.rows_per_variant, hidden=tuple(args.hidden),
        lr=args.lr, lr_schedule=parse_lr_schedule(args.lr_schedule),
        batch_size=args.batch_size, passes=args.passes,
        replay_window=args.replay_window, td_lambda=args.td_lambda,
        exact_from=args.exact_from, conservative=args.conservative,
        aggressive=args.aggressive, persona_bias=args.persona_bias,
        personas_from=args.personas_from,
        reflect_augment=args.reflect_augment,
        variant_weights=parse_weights(args.variant_weights),
        random_start_fraction=args.random_start_fraction,
        random_start_turns=args.random_start_turns,
        eval_every=args.eval_every,
        eval_games=args.eval_games, reference_iter=args.reference_iter,
        probes_per_variant=args.probes_per_variant,
        probe_alert=args.probe_alert, seed=args.seed, device=args.device,
        init_checkpoint=args.init_checkpoint, threads=args.threads,
        in_flight=args.in_flight, resume=args.resume)


if __name__ == "__main__":
    main()
