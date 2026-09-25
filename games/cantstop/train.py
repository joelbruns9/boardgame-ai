"""MVP training loop: self-play with the current net, then fit the winner.

Phase 2 of ``VARIANT_SOLVER_PLAN.md``. Deliberately narrow and standalone --
it is *not* wired into ``games/az_loop``, because az_loop is built around
policy+value AlphaZero learners and integrating it would gate this on
plumbing rather than on the only question that matters here: does the loop
learn at all?

Run (from the repo root):

    python -m games.cantstop.train --out runs/mvp --iterations 10 --games 20

Generation is single-process for now. The fan-out below (which rule sets,
how many games each, how rows are collected) is kept separate from *how* the
games are executed precisely so a process pool -- and later the Rust engine --
can replace the execution without touching the schedule.
"""

import argparse
import json
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .arena import compare
from .encoder import FEATURE_SIZE
from .engine import ALL_RULESETS, RuleSet
from .model import (
    CantStopNet, NetEvaluator, load_net, masked_cross_entropy,
    masked_soft_cross_entropy, save_net,
    seat_mask_tensor,
)
from .self_play import play_game, stack_training, summarize
from .portable_rng import PortableRng
from .solver import ProgressHeuristic

# The MVP rule set: fewest seats and fewest columns to win, so games are as
# short as the variant space allows. The encoder is full width regardless.
MVP_RULES = RuleSet.make(2, extended=False, blocking=False)


class ReplayBuffer:
    """Most-recent rows, one chunk per ``add`` (per iteration).

    Two independent limits, either or both:

    * ``window_iterations`` -- keep the last K iterations' rows, as 7WD's
      ``--replay-window``. The run's default; it is what makes
      ``--passes`` mean "passes over each position's lifetime".
    * ``max_rows`` -- a hard row cap, trimmed by whole iterations. Rows, not
      games, because a 4-player or 5-column game contributes several times
      the rows of a 2-player base game.
    """

    def __init__(self, max_rows=None, window_iterations=None):
        if max_rows is None and window_iterations is None:
            raise ValueError("give max_rows, window_iterations, or both")
        self.max_rows = max_rows
        self.window_iterations = window_iterations
        self._chunks = deque()
        self._rows = 0

    def add(self, features, slots):
        if len(features) != len(slots):
            raise ValueError("features and labels disagree in length")
        if not len(features):
            return
        self._chunks.append((features, slots))
        self._rows += len(features)
        while len(self._chunks) > 1 and (
                (self.max_rows is not None and self._rows > self.max_rows)
                or (self.window_iterations is not None
                    and len(self._chunks) > self.window_iterations)):
            old_x, _ = self._chunks.popleft()
            self._rows -= len(old_x)

    @property
    def iterations(self):
        """Iterations actually held (realised window)."""
        return len(self._chunks)

    def __len__(self):
        return self._rows

    def arrays(self):
        if not self._chunks:
            return (np.zeros((0, FEATURE_SIZE), dtype=np.float32),
                    np.zeros(0, dtype=np.int64))
        return (np.concatenate([x for x, _ in self._chunks]),
                np.concatenate([y for _, y in self._chunks]))


def generate(rule_sets, games_per_ruleset, evaluate, rng, backend="auto",
             threads=0, stats=None, in_flight=None):
    """Play the scheduled games and return every result.

    Each game rolls from its own ``PortableRng``, seeded up front in
    schedule order, so both backends play exactly the same games:

    * ``"rust"`` -- the M3 pool: every game in flight at once, Rust doing
      the turns on ``threads`` cores, one batched forward per round;
    * ``"python"`` -- one game at a time through ``self_play.play_game``,
      the reference the M3 gate replays;
    * ``"auto"`` -- rust when the extension is built.
    """
    from . import rust_pool
    if backend == "auto":
        backend = "rust" if rust_pool.rust_available() else "python"
    if backend == "rust":
        return rust_pool.generate(
            rule_sets, games_per_ruleset, evaluate, rng, threads=threads,
            stats=stats, in_flight=in_flight or rust_pool.DEFAULT_IN_FLIGHT)
    if backend != "python":
        raise ValueError(f"unknown backend {backend!r}")
    schedule = [r for r in rule_sets for _ in range(games_per_ruleset)]
    seeds = rust_pool.game_seeds(rng, len(schedule))
    return [play_game(rules, evaluate, PortableRng(seed))
            for rules, seed in zip(schedule, seeds)]


def train_steps(net, buffer, opt, steps, batch_size, device, rng):
    """Sample minibatches from the buffer and fit. Returns the mean loss."""
    x_all, y_all = buffer.arrays()
    if not len(x_all):
        return float("nan")
    x_all = torch.from_numpy(x_all).to(device)
    y_all = torch.from_numpy(y_all).to(device)
    mask_all = seat_mask_tensor(x_all)
    # 1-D integer rows are winner slots; 2-D float rows are TD targets
    # (a distribution over slots, ``self_play.td_targets``).
    soft = y_all.ndim == 2
    loss_fn = masked_soft_cross_entropy if soft else masked_cross_entropy

    net.train()
    losses = []
    n = len(x_all)
    for _ in range(steps):
        idx = torch.from_numpy(
            np.array([rng.randrange(n) for _ in range(min(batch_size, n))])
        ).to(device)
        opt.zero_grad()
        loss = loss_fn(net(x_all[idx]), y_all[idx], mask_all[idx])
        loss.backward()
        opt.step()
        losses.append(loss.item())
    net.eval()
    return float(np.mean(losses))


def steps_for_passes(buffer_rows, passes, replay_window, batch_size):
    """Optimizer steps this iteration so that every position is sampled
    ``passes`` times on average over its whole life in the buffer.

    Each iteration draws ``passes / replay_window`` of the current buffer, so
    a row that lives the full window collects exactly ``passes`` -- whether
    it arrived in a thin early buffer or a full one. (A per-new-row rule,
    ``passes * new_rows``, instead hammers the first iterations' rows: they
    are resampled every iteration while the buffer is small, ~5 * H(30) = 20
    passes for iteration 1's rows at a 30-iteration window.) The cost is
    light training early, until the window fills.
    """
    samples = passes * buffer_rows / replay_window
    return max(1, math.ceil(samples / batch_size))


def run(out_dir, iterations, games, rule_sets=(MVP_RULES,), hidden=(256, 256),
        lr=1e-3, batch_size=256, steps=None, passes=5.0, replay_window=30,
        buffer_rows=None, arena_games=60, seed=0, device=None,
        init_checkpoint=None, backend="auto", threads=0, in_flight=None,
        td_lambda=0.7):
    """``steps=None`` derives each iteration's steps from ``passes`` and
    ``replay_window`` (``steps_for_passes``); an integer fixes them, as
    the pre-window loop did."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.jsonl"

    device = torch.device(device or ("cuda" if torch.cuda.is_available()
                                     else "cpu"))
    torch.manual_seed(seed)
    rng = random.Random(seed)
    torch_rng = random.Random(seed + 1)

    net = (load_net(init_checkpoint, device=str(device)) if init_checkpoint
           else CantStopNet(hidden=hidden))
    net.to(device)
    save_net(net, out / "iter_0000.pt")

    # The random init is a fixed reference point for the whole run: if the
    # trained net cannot beat where it started, nothing else in the log
    # matters.
    baseline_net = CantStopNet(**net.config())
    baseline_net.load_state_dict(net.state_dict())
    baseline = NetEvaluator(baseline_net, device=str(device))

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    buffer = ReplayBuffer(max_rows=buffer_rows,
                          window_iterations=replay_window)

    for it in range(1, iterations + 1):
        started = time.time()
        evaluate = NetEvaluator(net, device=str(device))
        results = generate(rule_sets, games, evaluate, rng, backend=backend,
                           threads=threads, in_flight=in_flight)
        gen_seconds = time.time() - started

        # Targets are fixed when rows enter the buffer, from the net that
        # played the game -- as TD-Gammon's were, and as 7WD's search targets.
        x, y = stack_training(results, td_lambda)
        buffer.add(x, y)
        it_steps = (steps if steps is not None else
                    steps_for_passes(len(buffer), passes, replay_window,
                                     batch_size))
        loss = train_steps(net, buffer, opt, it_steps, batch_size, device,
                           torch_rng)
        samples = it_steps * min(batch_size, len(buffer))

        record = {
            "iteration": it,
            "loss": loss,
            "buffer_rows": len(buffer),
            # 7WD's training-schedule columns: the realised window and how
            # hard this iteration leaned on the data it has.
            "window_iterations": buffer.iterations,
            "td_lambda": td_lambda,
            "new_rows": len(x),
            "train_steps": it_steps,
            "samples": samples,
            "samples_per_new_position": round(samples / len(x), 3),
            "buffer_passes": round(samples / len(buffer), 4),
            "gen_seconds": round(gen_seconds, 1),
            "seconds_per_game": round(gen_seconds / max(1, len(results)), 2),
            **summarize(results),
        }
        record["seat_wins"] = {str(k): v for k, v in
                               record["seat_wins"].items()}
        save_net(net, out / f"iter_{it:04d}.pt")

        if arena_games:
            current = NetEvaluator(net, device=str(device))
            record["vs_random_init"] = compare(
                rule_sets[0], current, baseline, arena_games, rng,
                backend=backend, threads=threads, in_flight=in_flight)
            record["vs_heuristic"] = compare(
                rule_sets[0], current, ProgressHeuristic(), arena_games, rng,
                backend=backend, threads=threads, in_flight=in_flight)

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        print(json.dumps(record, default=str), flush=True)

    return net


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/cantstop-mvp")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--games", type=int, default=20,
                   help="games per rule set per iteration")
    p.add_argument("--all-rulesets", action="store_true",
                   help="train on all 10 rule sets instead of the MVP one")
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--steps", type=int, default=None,
                   help="fixed optimizer steps per iteration; overrides "
                        "--passes")
    p.add_argument("--passes", type=float, default=5.0,
                   help="average times each position is sampled over its "
                        "life in the buffer; sets steps per iteration")
    p.add_argument("--td-lambda", type=float, default=0.7,
                   help="value-target lambda-return: 1 = game outcome only "
                        "(the pre-TD labels), 0 = one-step TD against the "
                        "next turn's solver value")
    p.add_argument("--replay-window", type=int, default=30,
                   help="iterations of self-play kept in the buffer")
    p.add_argument("--buffer-rows", type=int, default=None,
                   help="optional hard row cap on top of the window")
    p.add_argument("--arena-games", type=int, default=60,
                   help="0 to skip the head-to-head checks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--init-checkpoint", default=None)
    p.add_argument("--backend", choices=("auto", "rust", "python"),
                   default="auto",
                   help="rust = the M3 pool (many games at once, batched "
                        "forwards); auto = rust when the extension is built")
    p.add_argument("--threads", type=int, default=0,
                   help="pool worker threads; 0 = one per logical core")
    p.add_argument("--in-flight", type=int, default=None,
                   help="games live at once in the pool (default 64; 128 "
                        "measured slower on an 8 GB laptop GPU)")
    args = p.parse_args(argv)

    run(out_dir=args.out,
        iterations=args.iterations,
        games=args.games,
        rule_sets=tuple(ALL_RULESETS) if args.all_rulesets else (MVP_RULES,),
        hidden=tuple(args.hidden),
        lr=args.lr,
        batch_size=args.batch_size,
        steps=args.steps,
        passes=args.passes,
        replay_window=args.replay_window,
        td_lambda=args.td_lambda,
        buffer_rows=args.buffer_rows,
        arena_games=args.arena_games,
        seed=args.seed,
        device=args.device,
        init_checkpoint=args.init_checkpoint,
        backend=args.backend,
        threads=args.threads,
        in_flight=args.in_flight)


if __name__ == "__main__":
    main()
