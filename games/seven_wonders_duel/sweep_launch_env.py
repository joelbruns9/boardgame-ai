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


def _require(payload: dict, key: str, source: Path) -> object:
    if key not in payload:
        raise SystemExit(
            f"{source} has no {key!r} block; it is not the output of the sweep "
            "this expects. Re-run stage 8b rather than hand-editing the JSON."
        )
    return payload[key]


def build_env(sweep_dir: Path, gate_rung: str) -> dict[str, int]:
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

    env = {
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
    return env


def render(env: dict[str, int]) -> str:
    lines = [
        "# Measured on this box (setup_cloud_7wd.sh stage 8b, or sweep_7wd.sh).",
        "# Source this, then re-run the launcher to launch on these numbers.",
    ]
    if "RUST_SCHEDULER_WORKERS" in env:
        workers = env["RUST_SCHEDULER_WORKERS"]
        lines += [
            "#",
            "# SOLVER_THREADS is deliberately ABSENT, not forgotten. It is PER",
            f"# SHARD, so the total is {workers} x SOLVER_THREADS -- and leaving it",
            "# unset lets stage 6b derive it from this box's core count and the",
            "# worker count above, keeping the split tied to the geometry.",
            "# Pinning a value here would freeze a split that should follow it.",
            "# Set it only to override that derivation deliberately.",
        ]
    lines += [f"export {key}={value}" for key, value in env.items()]
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
        print(f"{key}={value}")
    print(f"written: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
