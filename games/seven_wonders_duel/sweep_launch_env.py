"""Turn this box's two scheduler sweeps into an env file the launcher can source.

`setup_cloud_7wd.sh` stage 8b measures generation and the gate separately,
because the batch cap's sign depends on the slot count it runs at: at 48 slots
widening it costs ~4%, at 144 slots it gains ~12%. One value cannot serve both
paths, so there are two harnesses with two output shapes:

* `f4_phase_d_sweep` writes ``{"summary": [...]}`` sorted fastest-first, each
  row carrying ``slots`` / ``global_batch_cap`` / ``max_inflight_batches``;
* `w5_gate_slots_sweep` writes ``{"best": {...}}`` with ``slots`` /
  ``global_batch_cap``.

Neither is the production-manifest shape `f4_launch_flags` reads -- that comes
from `f4_cloud_finalize`, which is a different three-input workflow. Rather than
pretend `LAUNCH_FLAGS_JSON` can consume a sweep (it raises `KeyError`), this
writes the settings straight out as environment variables.

The output is deliberately a *file to source* rather than numbers to re-type.
W6.3 exists because the bench and Phase D spell the same settings differently
and the transcription step sits in the middle of the one workflow where an error
is both easy and expensive: the numbers look plausible either way, and the run
is 24 hours long.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex


def _require(payload: dict, key: str, source: Path) -> object:
    if key not in payload:
        raise SystemExit(
            f"{source} has no {key!r} block; it is not the output of the sweep "
            "this expects. Re-run stage 8b rather than hand-editing the JSON."
        )
    return payload[key]


def build_env(sweep_dir: Path, gate_rung: str) -> dict[str, object]:
    """The measured settings, as the environment variables the launcher reads."""

    generation_path = sweep_dir / "generation" / "phase_d_sweep.json"
    gate_path = sweep_dir / f"gate_{gate_rung}.json"
    for path in (generation_path, gate_path):
        if not path.is_file():
            raise SystemExit(f"missing sweep output: {path}")

    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    summary = _require(generation, "summary", generation_path)
    if not summary:
        raise SystemExit(f"{generation_path} has an empty summary")
    # `f4_phase_d_sweep` sorts by median wall seconds ascending, so the first
    # row is the fastest point. Sorting again here would silently disagree with
    # the harness if it ever changes its key.
    best_generation = summary[0]
    best_gate = _require(gate, "best", gate_path)

    # THE STAGED SWEEP, when `f4_staged_sweep` drove it.
    #
    # Its `summary` is stage B's, which already carries stage A's winning
    # geometry on every row -- so everything above is correct without this block
    # and a file written by the old harness still works. What the block adds is
    # WHICH AXES WERE VARIED. The two conditional emits below ask exactly that
    # question, and stage B's summary cannot answer it: the solver column is
    # constant there because stage A pinned it, not because nobody measured it.
    staged = generation.get("staged")
    swept: set[str] | None = None
    if isinstance(staged, dict):
        best_generation = staged.get("winner") or best_generation
        swept = set(staged.get("swept_axes") or ())

    env: dict[str, object] = {
        "RUST_SLOTS": int(best_generation["slots"]),
        "RUST_GLOBAL_BATCH_CAP": int(best_generation["global_batch_cap"]),
        "RUST_MAX_INFLIGHT_BATCHES": int(best_generation["max_inflight_batches"]),
        "GATE_SLOTS": int(best_gate["slots"]),
        "GATE_GLOBAL_BATCH_CAP": int(best_gate["global_batch_cap"]),
    }
    # Older sweep outputs have no worker axis; those measured at one shard and
    # said nothing about the shard count, so emitting a value would be inventing
    # a measurement.
    if "scheduler_workers" in best_generation:
        env["RUST_SCHEDULER_WORKERS"] = int(best_generation["scheduler_workers"])
    # The generation/solver core split, but ONLY when the sweep actually varied
    # it. A grid that held solver threads fixed measured one split and says
    # nothing about the others, so emitting its value would dress a constant up
    # as a result -- the same reason the worker axis is conditional above.
    splits = {
        row["solver_threads_per_shard"]
        for row in summary
        if "solver_threads_per_shard" in row
    }
    solver_was_swept = (
        "solver_threads_total" in swept if swept is not None else len(splits) > 1
    )
    if solver_was_swept and "solver_threads_per_shard" in best_generation:
        env["SOLVER_THREADS"] = int(best_generation["solver_threads_per_shard"])
    # The coalescing wait: emitted whenever the winning row HAS one, varied or
    # not.
    #
    # This deliberately does NOT follow the `len(splits) > 1` rule that guards
    # SOLVER_THREADS above, and the difference is the fallback. An unset
    # SOLVER_THREADS is DERIVED at stage 6b from the box's cores and the worker
    # count, which is a better answer than a pinned one. An unset
    # RUST_INFERENCE_WAIT_MS is 0, or whatever stale value the environment
    # happens to carry -- neither of which is what was measured.
    #
    # So a confirmation sweep pinned at 2 ms used to emit nothing and hand
    # production a 0 ms run, while every other number in the file described a
    # geometry measured at 2 ms. "Did the sweep optimise this axis" and "can
    # production reproduce what the sweep ran" are different questions; this
    # file answers the second.
    if "inference_wait_ms" in best_generation:
        env["RUST_INFERENCE_WAIT_MS"] = float(best_generation["inference_wait_ms"])
        # Whether it was actually a free axis, for the comment `render` writes.
        waits = {
            row["inference_wait_ms"] for row in summary if "inference_wait_ms" in row
        }
        env["_WAIT_WAS_SWEPT"] = (
            "inference_wait_ms" in swept if swept is not None else len(waits) > 1
        )
    # Provenance. `SKIP_SWEEPS` cannot serve as this marker: an operator sets
    # that by hand to skip measuring altogether, so it is true in exactly the
    # case this needs to detect.
    if "median_requests_per_forward" in best_generation:
        # Underscore-prefixed: read by `render` for the comment above, and
        # dropped before anything is exported.
        env["_REQUESTS_PER_FORWARD"] = float(
            best_generation["median_requests_per_forward"]
        )
    # The SEARCH BUDGET the geometry was chosen at, and whether it was staged.
    # Both are underscore-prefixed: they steer `render`'s comments and are never
    # exported, because neither is a launcher knob. A geometry measured at a
    # quarter of the run's simulations is still the right geometry to launch on
    # -- it is just not a measurement of the run's throughput, and the file that
    # carries it should say so where the operator reads it.
    divisor = int((generation.get("config") or {}).get("sims_divisor", 1) or 1)
    if divisor > 1:
        env["_SIMS_DIVISOR"] = divisor
    if swept is not None:
        env["_STAGED_AXES"] = ",".join(sorted(swept))
    env["SWEEP_MEASURED"] = "1"
    env["SWEEP_MEASURED_FROM"] = str(sweep_dir.resolve())
    return env


def render(env: dict[str, object]) -> str:
    lines = [
        "# Measured on this box (setup_cloud_7wd.sh stage 8b, or sweep_7wd.sh).",
        "# Source this, then re-run the launcher to launch on these numbers.",
    ]
    if "_STAGED_AXES" in env:
        lines += [
            "#",
            "# STAGED sweep (f4_staged_sweep): geometry -- slots, cap, workers,",
            "# solver split -- was ranked first, then the inflight and wait axes",
            "# were swept at the winning geometry. That is a SUM of two grids",
            "# rather than their product; the cost is that the second stage",
            "# cannot reorder the first. Axes that actually varied somewhere:",
            f"#   {env['_STAGED_AXES']}",
        ]
    if "_SIMS_DIVISOR" in env:
        divisor = env["_SIMS_DIVISOR"]
        lines += [
            "#",
            f"# ! Measured at 1/{divisor} of the run's SIMULATION budget, with the",
            "#   solver's node budget divided by the same factor. The geometry",
            "#   below is still the geometry to launch on: what the sweep ranks",
            "#   is points against each other, and the search algorithm, the",
            "#   cheap/full mix and the solver's share of slot occupancy all",
            "#   survive the division.",
            "#   What does NOT survive is the ABSOLUTE throughput. Any games/hour",
            f"#   in the sweep output is roughly {divisor}x the run's rate and is",
            "#   not a prediction. Read the run's own heartbeat for that.",
        ]
    if "RUST_SCHEDULER_WORKERS" in env and "SOLVER_THREADS" not in env:
        workers = env["RUST_SCHEDULER_WORKERS"]
        lines += [
            "#",
            "# SOLVER_THREADS is deliberately ABSENT, not forgotten. It is PER",
            f"# SHARD, so the total is {workers} x SOLVER_THREADS -- and leaving it",
            "# unset lets stage 6b derive it from this box's core count and the",
            "# worker count above, keeping the split tied to the geometry.",
            "# Pinning a value here would freeze a split that should follow it.",
            "# Set it only to override that derivation deliberately.",
            "# This sweep did not VARY the split, so there is nothing measured",
            "# to pin; sweep SWEEP_SOLVER_THREADS_CSV to get one.",
        ]
    elif "SOLVER_THREADS" in env:
        lines += [
            "#",
            "# SOLVER_THREADS is PER SHARD and was MEASURED here, so it is pinned",
            f"# rather than derived: total solver threads are "
            f"{env.get('RUST_SCHEDULER_WORKERS', '?')} x {env['SOLVER_THREADS']}.",
            "# The generation/solver split competes for the same cores, so the",
            "# winning value belongs to the worker count beside it -- change one",
            "# and the other is no longer measured.",
        ]
    if "RUST_INFERENCE_WAIT_MS" in env:
        engaged = env.get("_REQUESTS_PER_FORWARD")
        lines += [
            "#",
            "# RUST_INFERENCE_WAIT_MS is the evaluator COALESCING wait. 0 is a",
            "# real answer, not a disabled feature: the worker always merges what",
            "# is already queued and this only buys width by waiting for more.",
        ]
        if env.get("_WAIT_WAS_SWEPT"):
            lines.append("# This sweep VARIED the wait, so the value is a winner.")
        else:
            lines += [
                "# ! This sweep held the wait FIXED, so the value is not a",
                "#   winner -- it is what the measured points actually ran at.",
                "#   It is pinned anyway because the alternative is 0, and a run",
                "#   at 0 would not be the run that was measured.",
            ]
        if engaged is not None:
            # 1.00 means two very different things and the operator reading this
            # on a rented box should not have to work out which.
            #
            # At ONE shard there is a single submitter, so 1.00 is arithmetic,
            # not a failure -- nothing existed to merge with. Above one shard it
            # IS a failure: the coalescer was configured and merged nothing, and
            # any wait above 0 bought pure latency.
            workers = int(env.get("RUST_SCHEDULER_WORKERS", 1) or 1)
            ratio = float(engaged)
            if workers <= 1:
                lines += [
                    f"# The winning point merged {ratio:.2f} requests per forward,",
                    "# which at ONE shard is by construction -- a single submitter",
                    "# has nothing to merge with. Not a failed merge.",
                ]
            elif ratio <= 1.0:
                lines += [
                    f"# ! The winning point merged {ratio:.2f} requests per forward",
                    f"#   across {workers} shards. The coalescer was configured and",
                    "#   did NOT merge; a wait above 0 bought only latency. Check",
                    "#   the sweep before trusting this pick.",
                ]
            else:
                lines += [
                    f"# The winning point merged {ratio:.2f} requests per forward",
                    f"# across {workers} shards -- the coalescer is doing work here.",
                ]
    # QUOTED. This file is `source`d, so a value containing a space -- a run
    # directory under one, most obviously -- would otherwise split into two
    # words and export something that is not the measurement. The integers are
    # unaffected; the path is the reason.
    lines += [
        f"export {key}={shlex.quote(str(value))}"
        for key, value in env.items()
        if not key.startswith("_")
    ]
    # Pass 2 must not re-measure: the sweeps are the expensive part of setup.
    lines.append("export SKIP_SWEEPS=1")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--gate-rung", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="defaults to <sweep-dir>/measured_env.sh",
    )
    args = parser.parse_args(argv)

    env = build_env(args.sweep_dir, args.gate_rung)
    destination = args.output or (args.sweep_dir / "measured_env.sh")
    destination.write_text(render(env), encoding="utf-8")
    for key, value in env.items():
        # `_`-prefixed keys steer `render`'s comments and are never exported;
        # printing them invites someone to set one by hand.
        if key.startswith("_"):
            continue
        print(f"{key}={value}")
    print(f"written: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
