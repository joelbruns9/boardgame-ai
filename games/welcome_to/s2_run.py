"""S2 run driver: generate -> train -> gate, one iteration at a time, resumably.

The three stages already exist as CLIs (``self_play``, ``s2_train``,
``s2_promotion``); this only sequences them over a run directory and records one
line of strength metrics per iteration.  Built for the encoder-v3 short laptop
run (ENCODER_V3_SPEC.md §12 step 8): random weights, no behaviour cloning
(decided 2026-09-14), judged on the absolute trend of the §11.2 strength
metrics rather than against GreedyBot or an old-encoder arm (decided
2026-09-26).

RUN DIRECTORY
-------------
::

    <run>/current_best.pt            the gate's incumbent; replaced on promotion
    <run>/league.json                promoted history (written by s2_promotion)
    <run>/promoted/                  archived previous bests
    <run>/iter_NNNN/trajectories.*   generation output, the replay root's unit
    <run>/candidate_iter_NNNN.pt     trained candidate (+ .metrics.json, .replay.json)
    <run>/candidate_iter_NNNN.pt.gate.json   the promotion record
    <run>/progress.jsonl             one line per finished iteration

Training is continuous: each candidate resumes from the previous candidate
(weights and optimizer), and the LEARNER generates -- iteration ``i`` plays with
candidate ``i - 1`` against the league, whose "current best" slot is
``current_best.pt``.  The gate runs every ``--gate-every`` iterations (and on the
last): it measures the latest candidate against ``current_best.pt`` and
promotes it if it is significantly better, which also enters it in the league.
Iteration 1 generates with ``current_best.pt``, a seeded random network.

⚠ Between gates nothing stops a regressing learner from generating; the next
gate is what catches it.  That is the AlphaZero trade-off, chosen 2026-09-26
over gating every iteration (which would leave a stale generator in place for
every iteration that fails to promote).

RESUMING
--------
Re-running the same command continues where it stopped.  An iteration whose
gate record exists is done; a candidate without a record is gated; generation
resumes inside ``self_play`` itself.  A crash mid-promotion is recovered by
``s2_promotion`` from its own install record.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from games.welcome_to import network as nw
from games.welcome_to import s2_promotion, s2_train, self_play

#: Generation metrics copied into ``progress.jsonl`` -- §11.2's strength
#: signals plus the throughput and length context needed to read them.
_GENERATION_FIELDS = (
    "plans_per_seat_game",
    "plan_ending_fraction",
    "learner_score",
    "learner_margin_vs_best",
    "roundabouts_per_seat_game",
    "permits_per_seat_game",
    "mean_decisions_per_game",
    "games_per_hour",
    "evaluator_rows_per_second",
)


def _iteration_dir(run: Path, iteration: int) -> Path:
    return run / f"iter_{iteration:04d}"


def _candidate(run: Path, iteration: int) -> Path:
    return run / f"candidate_iter_{iteration:04d}.pt"


def _gate_record(candidate: Path) -> Path:
    return Path(str(candidate) + ".gate.json")


def _generation_complete(directory: Path, games: int) -> bool:
    metrics = directory / "trajectories.jsonl.metrics.json"
    if not metrics.is_file():
        return False
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    return int(payload.get("total_games", 0)) >= games


def _gate_complete(record: Path) -> bool:
    if not record.is_file():
        return False
    return json.loads(record.read_text(encoding="utf-8")).get("status") != "installing"


def initialise(run: Path, seed: int) -> Path:
    """Write the iteration-0 generator: a seeded random network, S2 format."""
    best = run / "current_best.pt"
    if best.exists():
        return best
    torch.manual_seed(seed)
    net = nw.WelcomeToNet(nw.NetConfig())
    config = s2_train.S2TrainConfig(seed=seed)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    s2_train.save_checkpoint(
        best,
        net,
        optimizer,
        config,
        {"optimizer_steps_completed": 0, "training_runs_completed": 0},
        source="random",
    )
    return best


def run_iteration(run: Path, iteration: int, args: argparse.Namespace) -> dict[str, Any]:
    """One generate -> train -> gate cycle.  Each stage is skipped if done."""
    best = run / "current_best.pt"
    directory = _iteration_dir(run, iteration)
    candidate = _candidate(run, iteration)
    record = _gate_record(candidate)
    seed = args.seed + iteration * 100_000
    timings: dict[str, float] = {}

    previous = _candidate(run, iteration - 1)
    learner = previous if previous.exists() else best
    gated = iteration % args.gate_every == 0 or iteration == args.iterations

    if not _generation_complete(directory, args.games):
        started = time.perf_counter()
        code = self_play.main([
            "--checkpoint", str(learner),
            "--league-manifest", str(run / "league.json"),
            "--league-iteration", str(iteration),
            "--league-current-best", str(best),
            "--games", str(args.games),
            "--simulations", str(args.simulations),
            "--inflight", str(args.inflight),
            "--max-batch", str(args.inflight),
            "--scheduler-workers", str(args.workers),
            "--seed", str(seed),
            "--out", str(directory / "trajectories.jsonl"),
        ])
        if code != 0:
            raise RuntimeError(f"generation for iteration {iteration} exited {code}")
        timings["generation_seconds"] = time.perf_counter() - started

    if not candidate.exists():
        resume = learner
        started = time.perf_counter()
        code = s2_train.main([
            "--replay-root", str(run),
            "--replay-through-iteration", str(iteration),
            "--resume", str(resume),
            "--train-steps", str(args.train_steps),
            "--seed", str(seed),
            "--out", str(candidate),
        ])
        if code != 0:
            raise RuntimeError(f"training for iteration {iteration} exited {code}")
        timings["training_seconds"] = time.perf_counter() - started

    if gated and not _gate_complete(record):
        started = time.perf_counter()
        code = s2_promotion.main([
            "--candidate", str(candidate),
            "--current-best", str(best),
            "--games", str(args.gate_games),
            "--simulations", str(args.simulations),
            "--inflight", str(args.inflight),
            "--max-batch", str(args.inflight),
            "--scheduler-workers", str(args.workers),
            "--seed", str(seed + 1),
            "--promote",
            "--iteration", str(iteration),
            "--league-manifest", str(run / "league.json"),
            "--report", str(record),
        ])
        if code not in (0, 2):  # 2 is "gated, not promoted"
            raise RuntimeError(f"gate for iteration {iteration} exited {code}")
        timings["gate_seconds"] = time.perf_counter() - started

    generation = json.loads(
        (directory / "trajectories.jsonl.metrics.json").read_text(encoding="utf-8")
    )
    report: dict[str, Any] = {}
    if gated:
        gate = json.loads(record.read_text(encoding="utf-8"))
        report = gate.get("report", gate)
    line = {
        "iteration": iteration,
        "gated": gated,
        "promoted": report.get("decision") == "promote" if gated else None,
        "gate_margin_delta": report.get("primary_margin_delta"),
        "gate_rank_delta": report.get("secondary_rank_delta"),
        "gate_candidate": report.get("candidate"),
        "gate_incumbent": report.get("incumbent"),
        **{name: generation.get(name) for name in _GENERATION_FIELDS},
        **timings,
    }
    with (run / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")
    return line


def _finished(run: Path) -> set[int]:
    path = run / "progress.jsonl"
    if not path.is_file():
        return set()
    return {
        int(json.loads(line)["iteration"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(
        description="Run S2 iterations: generate, train, gate (resumable)."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--train-steps", type=int, default=400)
    parser.add_argument("--gate-games", type=int, default=300)
    parser.add_argument(
        "--gate-every",
        type=int,
        default=5,
        help="gate the latest candidate every N iterations, and on the last",
    )
    parser.add_argument("--inflight", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=60_000)
    args = parser.parse_args(argv)

    if args.gate_every <= 0:
        parser.error("--gate-every must be positive")
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "driver.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )
    initialise(run, args.seed)
    done = _finished(run)
    for iteration in range(1, args.iterations + 1):
        if iteration in done:
            continue
        line = run_iteration(run, iteration, args)
        gate = (
            f"gate: promoted {line['promoted']}, margin delta "
            f"{line['gate_margin_delta']['mean']:+.2f}"
            if line["gated"]
            else "no gate"
        )
        print(
            f"iter {iteration:3d}  plans/seat {line['plans_per_seat_game']:.3f}  "
            f"score {line['learner_score']:.1f}  {gate}",
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
