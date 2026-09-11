"""Two-stage driver for `f4_phase_d_sweep`: geometry first, then batching.

The generation sweep has six free axes -- slots, batch cap, inflight batches,
scheduler workers, the generation/solver core split, and the coalescing wait --
and `f4_phase_d_sweep` takes their full CARTESIAN PRODUCT. That was affordable
while every point ran `PhaseDConfig`'s laptop defaults. It stopped being
affordable the moment `--config-from-manifest` made each point run the RUN's
search: 3 slots x 2 caps x 2 inflight x 3 workers x 3 waits is 108 points, twice
over for repetitions, each one a full generation iteration at 1600 simulations a
move.

Cutting the grid is the only lever that does not cost fidelity somewhere, and
the cut has to respect which axes actually interact:

* **slots, caps, workers and the solver split interact strongly.** `rust_slots`
  is a GLOBAL budget divided among shards, so an optimum at one shard is not the
  optimum at four; the cap binds as slots rise, so sweeping slots at a fixed cap
  finds a ceiling that belongs to the cap; and solver threads come off the same
  cores the shards run on. These four are swept JOINTLY -- stage A.
* **inflight batches and the coalescing wait are about what happens to a batch
  once the geometry has produced it.** They are swept at stage A's winning
  geometry -- stage B.

That is 18 + 6 points where the product is 108, and the two stages answer the
two questions in the order their answers depend on each other.

WHAT THIS GIVES UP, stated rather than buried. Staging assumes the stage-B axes
do not REORDER the stage-A ranking. There is one known place that assumption is
weak: the coalescing wait pays off in proportion to how many shards there are to
merge across (4 shards measured 1.94x at zero wait and 3.79x at 2 ms), so a
shard count chosen at stage A's pinned wait of 0 is chosen where high shard
counts look worst, and stage B can then only pick a wait for a shard count stage
A already decided. `--stage-a-wait-ms` exists for exactly that: set it to the
wait the run intends and stage A ranks shards with coalescing already on. The
driver prints and records which way it ran, rather than leaving it to be
inferred.

The stage-A pin must be a point stage B also measures, and that is enforced. It
buys a free check: stage B re-measures the exact winning point, so a box that
drifted thermally between the stages says so in the output instead of silently
reordering stage B.

Output is the shape `sweep_launch_env.py` already reads -- `{"summary": [...]}`
sorted fastest-first -- so nothing downstream has to know this ran. `summary` is
stage B's, whose rows all carry stage A's winning geometry, which means even a
consumer that ignores the `staged` block reads the right slots, cap, workers and
solver split off row 0. The `staged` block adds the one thing that shape cannot
express: WHICH axes were varied, across both stages. Without it
`sweep_launch_env` would look at stage B's constant solver column, conclude the
split was never swept, and decline to pin a value stage A had in fact measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import f4_phase_d_sweep as sweep


# Each axis: the flag it is passed as, the name it is REPORTED under in
# `swept_axes`, and the summary key that says whether it actually varied.
#
# The solver is the one place those last two differ, and deliberately.
# `solver_threads_total` is what the flag takes, but it is also the product of
# the split and the worker count -- so a grid that holds the total fixed while
# sweeping workers has a varying total and a split nobody swept. The PER-SHARD
# column is what "was the split varied" actually means, and it is the column
# `sweep_launch_env` has always read for this decision. Matching it here is what
# lets the staged output stay a drop-in for the unstaged one.
AXES = {
    "slots": ("--slots", "slots", "slots"),
    "caps": ("--caps", "global_batch_cap", "global_batch_cap"),
    "inflight": ("--inflight", "max_inflight_batches", "max_inflight_batches"),
    "workers": ("--workers", "scheduler_workers", "scheduler_workers"),
    "solver": (
        "--solver-threads-total",
        "solver_threads_total",
        "solver_threads_per_shard",
    ),
    "wait": ("--inference-wait-ms", "inference_wait_ms", "inference_wait_ms"),
}


def _values(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _varied(summary: list[dict], key: str) -> bool:
    """Did this axis take more than one value in the rows actually measured?

    Read off the SUMMARY rather than off the command line, because
    `f4_phase_d_sweep` drops points -- fewer slots than shards, a positive wait
    at one shard -- and an axis whose every value but one was dropped was not
    swept, whatever the flag said.
    """

    return len({row.get(key) for row in summary}) > 1


def build_stage_argv(
    args, *, name: str, output: Path, axes: dict[str, str], solver_threads: int
) -> list[str]:
    """One `f4_phase_d_sweep` command line, as a list.

    Everything that is not an axis is forwarded identically to both stages, so
    the two run the same search, the same checkpoint and the same divisor. A
    setting that differed between them would make the stage-A winner a winner
    under conditions stage B never ran.

    `solver_threads` is PER SHARD and passed explicitly rather than forwarded,
    because it is not simply a constant. `f4_phase_d_sweep` falls back to it
    whenever the total axis is exactly `[0]` -- which is both "the operator gave
    a per-shard count and no total" and "stage A measured the solver off and off
    won". Stage B therefore passes the winning row's OWN per-shard figure, which
    reproduces either case exactly and invents neither.
    """

    argv = [
        "--checkpoint", args.checkpoint,
        "--output", str(output),
        "--games", str(args.games),
        "--iteration", str(args.iteration),
        "--repetitions", str(args.repetitions),
        "--warmup-games", str(args.warmup_games),
        "--device", args.device,
        "--precision", args.precision,
        "--sims-divisor", str(args.sims_divisor),
        "--solver-max-nodes", str(args.solver_max_nodes),
        "--solver-max-secs", str(args.solver_max_secs),
        "--solver-attempt-nodes", str(args.solver_attempt_nodes),
    ]
    if args.config_from_manifest:
        argv += ["--config-from-manifest", args.config_from_manifest]
    for override in args.config_override:
        argv += ["--config-override", override]
    for axis, value in axes.items():
        argv += [AXES[axis][0], value]
    argv += ["--solver-threads", str(solver_threads)]
    print("\n=== stage " + name + " ===\n  " + " ".join(argv) + "\n", flush=True)
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-games", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--config-from-manifest", default="")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument(
        "--sims-divisor",
        type=int,
        default=1,
        help="forwarded to both stages; see f4_phase_d_sweep.apply_sims_divisor",
    )
    # Stage A axes.
    parser.add_argument("--slots", default="128,256,512")
    parser.add_argument("--caps", default="1024,2048")
    parser.add_argument("--workers", default="1")
    parser.add_argument("--solver-threads-total", default="0")
    parser.add_argument("--solver-threads", type=int, default=0)
    parser.add_argument("--solver-max-nodes", type=int, default=0)
    parser.add_argument("--solver-max-secs", type=float, default=0.0)
    parser.add_argument("--solver-attempt-nodes", type=int, default=0)
    # Stage B axes.
    parser.add_argument("--inflight", default="1,2")
    parser.add_argument("--inference-wait-ms", default="0")
    # The stage-A pins: where on the stage-B axes stage A sits while it ranks
    # geometry.
    parser.add_argument(
        "--stage-a-inflight",
        default="1",
        help="inflight batches held fixed while stage A ranks geometry. Must be "
        "one of --inflight, so stage B re-measures the winning point.",
    )
    parser.add_argument(
        "--stage-a-wait-ms",
        default="0",
        help="coalescing wait held fixed while stage A ranks geometry. Must be "
        "one of --inference-wait-ms. 0 is the neutral value, and it is also "
        "where high shard counts look WORST, since the wait is what recovers "
        "cross-shard fragmentation. Set it to the wait the run intends when the "
        "shard count is the decision you care about.",
    )
    args = parser.parse_args(argv)

    # A pin outside the stage-B axis would mean stage B never measures the point
    # stage A picked, so the two stages could only be compared across a
    # configuration change. Refuse rather than produce that quietly.
    #
    # Compared as NUMBERS, not as strings: `--stage-a-wait-ms 0` against
    # `--inference-wait-ms 0.0,2` is the same point spelled twice, and refusing
    # it would be this check failing on its own formatting.
    for pin, axis, flag in (
        (args.stage_a_inflight, args.inflight, "--inflight"),
        (args.stage_a_wait_ms, args.inference_wait_ms, "--inference-wait-ms"),
    ):
        if float(pin) not in [float(value) for value in _values(axis)]:
            raise SystemExit(
                "stage A pins " + pin + " but " + flag + "=" + axis + " does not "
                "contain it, so stage B would never re-measure the winning "
                "point. Add it to " + flag + ", or move the pin."
            )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    stage_a = sweep.main(
        build_stage_argv(
            args,
            name="A (geometry: slots x caps x workers x solver split)",
            output=output / "stage_a",
            axes={
                "slots": args.slots,
                "caps": args.caps,
                "workers": args.workers,
                "solver": args.solver_threads_total,
                "inflight": args.stage_a_inflight,
                "wait": args.stage_a_wait_ms,
            },
            solver_threads=args.solver_threads,
        )
    )
    won = stage_a["summary"][0]
    print(
        f"\nstage A winner: slots={won['slots']} cap={won['global_batch_cap']} "
        f"workers={won['scheduler_workers']} "
        f"solver={won.get('solver_threads_total', 0)} "
        f"({won['median_games_per_hour']:.0f} games/h, "
        f"park={won.get('median_parked_slot_fraction', 0.0):.0%})",
        flush=True,
    )

    stage_b = sweep.main(
        build_stage_argv(
            args,
            name="B (batching: inflight x coalescing wait, at stage A's geometry)",
            output=output / "stage_b",
            axes={
                "slots": str(won["slots"]),
                "caps": str(won["global_batch_cap"]),
                "workers": str(won["scheduler_workers"]),
                "solver": str(won.get("solver_threads_total", 0)),
                "inflight": args.inflight,
                "wait": args.inference_wait_ms,
            },
            # The winning row's OWN per-shard count, not the flag's. See
            # `build_stage_argv`: this is what reproduces stage A exactly
            # whether the split was an axis or a forwarded constant.
            solver_threads=int(won.get("solver_threads_per_shard", 0)),
        )
    )
    best = stage_b["summary"][0]

    # The pin took, or the winner describes a geometry nobody measured. Cheap,
    # and the failure it catches -- a flag forwarded to the wrong stage -- looks
    # exactly like a legitimate result.
    for key in ("slots", "global_batch_cap", "scheduler_workers"):
        if best[key] != won[key]:
            raise SystemExit(
                f"stage B reports {key}={best[key]} where stage A won "
                f"{won[key]}; the geometry pin did not reach the harness"
            )

    # Stage B re-measured stage A's exact point, so the stages share one row. A
    # large disagreement there is the box drifting between them, and it makes
    # stage B's ranking -- and stage A's, equally -- untrustworthy at finer
    # margins than the drift itself.
    carried = next(
        (
            row
            for row in stage_b["summary"]
            if row["max_inflight_batches"] == int(args.stage_a_inflight)
            and float(row["inference_wait_ms"]) == float(args.stage_a_wait_ms)
        ),
        None,
    )
    drift = None
    if carried is not None and won["median_seconds"]:
        drift = carried["median_seconds"] / won["median_seconds"] - 1.0
        note = f"carryover: stage A's winning point re-measured {drift:+.1%} in stage B"
        if abs(drift) > 0.10:
            print(
                "WARNING: " + note + ". The two stages did not measure the same "
                "machine -- thermal drift, or something else on the box -- so "
                "differences smaller than that inside either stage are noise.",
                flush=True,
            )
        else:
            print(note, flush=True)

    swept = sorted(
        {
            reported
            for _flag, reported, varied_key in AXES.values()
            if _varied(stage_a["summary"], varied_key)
            or _varied(stage_b["summary"], varied_key)
        }
    )
    payload = {
        "config": dict(stage_b.get("config", {}), staged=True),
        # Stage B's, unchanged and fastest-first: its rows all carry stage A's
        # winning geometry, so a reader that knows nothing about staging still
        # takes the right settings off row 0.
        "summary": stage_b["summary"],
        "baseline": stage_b.get("baseline"),
        "staged": {
            "winner": best,
            # Which axes were VARIED, across both stages. The whole reason this
            # block exists: stage B's summary has a constant solver column, and
            # a consumer reading only that would conclude the split was never
            # measured and decline to pin what stage A chose.
            "swept_axes": swept,
            "carryover_drift": drift,
            "stage_a_pins": {
                "max_inflight_batches": int(args.stage_a_inflight),
                "inference_wait_ms": float(args.stage_a_wait_ms),
            },
            "stages": [
                {
                    "name": "geometry",
                    "output": str((output / "stage_a").resolve()),
                    "winner": won,
                    "summary": stage_a["summary"],
                },
                {
                    "name": "batching",
                    "output": str((output / "stage_b").resolve()),
                    "winner": best,
                    "summary": stage_b["summary"],
                },
            ],
        },
    }
    destination = output / "phase_d_sweep.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"\nstaged winner: slots={best['slots']} cap={best['global_batch_cap']} "
        f"inflight={best['max_inflight_batches']} "
        f"workers={best['scheduler_workers']} "
        f"solver={best.get('solver_threads_total', 0)} "
        f"wait={best['inference_wait_ms']:g} "
        f"({best['median_games_per_hour']:.0f} games/h)\n"
        f"axes actually varied: {', '.join(swept) or 'none'}\n"
        f"written: {destination}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
