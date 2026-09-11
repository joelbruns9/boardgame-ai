#!/usr/bin/env python3
"""Stage 0: what batch size does this GPU stop paying for, and does it merge?

Everything downstream aims at a number this produces. Stage 1 raises slots until
rows-per-forward plateaus -- but "plateaus" is meaningless without knowing where
the GPU stops rewarding a wider batch, and that knee is a property of the model
and the card, measurable in seconds with no games and no scheduler.

    python -m games.seven_wonders_duel.stage0_gpu_curve \\
        --checkpoint L.pt --buffer some_iter.jsonl

**Why this has to run before the slot sweep.** On a laptop 3070 the forward cost
is nearly flat past ~128 rows: 144 rows/s at batch 1 against 3,781 at 128, and
512 is WORSE at 2,790. Without that curve, a measured 44 rows/forward looks like
a number rather than a sixth of the achievable throughput -- and a sweep that
widens batches past the knee spends wall clock to buy nothing.

**And why the coalescer check belongs here.** cloud2 ran 256 slots and got 46.9
rows/forward; a laptop at 24 slots got 44.2. Ten times the slots bought nothing,
because that build had no cross-shard coalescer and four shards fragmented every
batch. Stage 1's premise -- raise slots, watch rows/forward rise -- is FALSE
under a dead coalescer, and it fails by concluding "slots do not help". So this
asserts merging before any slot number is trusted:

    requests > forwards  <=>  something merged.

The two halves answer different questions and both gate stage 1. The curve says
what to aim for; the liveness check says the mechanism that gets you there is
running.

**What the curve does NOT include.** It times the evaluator -- H2D, forward,
gather, D2H -- on real encodings, which is where the production boundary spends
126s of every 165s. It excludes the Rust-side packing that precedes it. That
makes the knee slightly optimistic in absolute rows/s and leaves its LOCATION,
the only thing stage 1 consumes, unaffected.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

DEFAULT_BATCHES = (1, 8, 16, 32, 64, 128, 256, 512, 1024)


def sample_encodings(buffer_path: Path, wanted: int) -> tuple[list, list]:
    """Real positions, not random tensors.

    Token counts vary by position (the production run measured a 0.19 padding
    ratio), and a batch of identical synthetic rows would understate the padded
    width the GPU actually sees.
    """

    from .buffer import read_records, replay
    from .codec import legal_action_indices
    from .encoder import encode

    encodings: list = []
    legals: list = []
    for record in read_records(buffer_path):
        if len(encodings) >= wanted:
            break

        def visit(game, move, _stop=wanted):
            if len(encodings) >= _stop:
                return
            actor = (
                game.pending_choice.player
                if game.pending_choice is not None
                else game.active_player
            )
            encodings.append(encode(game.observation(actor)))
            legals.append([int(a) for a in legal_action_indices(game)])

        replay(record, on_state=visit)
    return encodings[:wanted], legals[:wanted]


def forward_curve(
    evaluator,
    encodings: list,
    legals: list,
    batches: tuple[int, ...],
    *,
    device: str,
    repetitions: int = 5,
) -> list[dict[str, Any]]:
    import torch

    rows = []
    for batch in batches:
        if batch > len(encodings):
            continue
        chunk_e, chunk_l = encodings[:batch], legals[:batch]
        for _ in range(2):  # warm
            evaluator.evaluate(chunk_e, chunk_l)
        if device == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(repetitions):
            started = time.perf_counter()
            evaluator.evaluate(chunk_e, chunk_l)
            if device == "cuda":
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
        seconds = statistics.median(samples)
        rows.append(
            {
                "batch": batch,
                "ms_per_call": seconds * 1e3,
                "us_per_row": seconds / batch * 1e6,
                "rows_per_second": batch / seconds,
            }
        )
    return rows


def knee(curve: list[dict[str, Any]], tolerance: float = 0.05) -> dict[str, Any]:
    """The smallest batch within `tolerance` of the best rows/s.

    The SMALLEST, deliberately. Past the knee a wider batch buys nothing and
    costs latency -- which is why a coalescing wait that widened batches from
    101 to 163 rows measured SLOWER, 41.6 to 49.3 ms/position.
    """

    best = max(row["rows_per_second"] for row in curve)
    largest = curve[-1]["batch"]
    found = next(
        row for row in curve if row["rows_per_second"] >= best * (1.0 - tolerance)
    )
    # If the best throughput is at the LARGEST batch tested, the curve never
    # turned over: that is the edge of the grid, not a knee. Reporting it as one
    # hands stage 1 a target the GPU has not been shown to stop rewarding -- a
    # measurement that is really a statement about the range.
    return {
        "batch": found["batch"],
        "rows_per_second": found["rows_per_second"],
        "peak_rows_per_second": best,
        "tolerance": tolerance,
        "beyond_tested_range": found["batch"] == largest,
    }


def coalescer_live(checkpoint: str, device: str, precision: str) -> dict[str, Any]:
    """Does the boundary merge requests from different shards into one forward?

    Asserted as `requests > forwards`. Equality means every request became its
    own forward, which is the pre-coalescer behaviour and the state cloud2 ran
    in -- configured, reported, and not merging.
    """

    import seven_wonders_rust as swr

    from .control_table import ensure_rust_table
    from .phase_d import PhaseDConfig, PhaseDLoop
    from .rust_bridge import rust_flat_batch_adapter, rust_games_for_self_play
    from .search import Evaluator

    ensure_rust_table()
    config = PhaseDConfig(run_dir="unused", device=device, precision=precision)
    loop = PhaseDLoop.__new__(PhaseDLoop)
    loop.config = config
    model = loop.load_model(checkpoint)
    evaluator = Evaluator(model, device, 2048, precision=precision)

    seeds = [90210 + index for index in range(16)]
    _records, metrics = swr.self_play_many_flat_net(
        adapter=rust_flat_batch_adapter(evaluator),
        games=rust_games_for_self_play(seeds, [i % 2 for i in range(len(seeds))]),
        game_seeds=seeds,
        global_batch_cap=1024,
        leaf_batch=1,
        cheap_sims_min=16,
        cheap_sims_max=16,
        full_sims_min=32,
        full_sims_max=32,
        full_search_fraction=0.25,
        top_k=8,
        draft_prior=0.0,
        iteration=1,
        # More than one shard, or there is nothing to merge ACROSS.
        scheduler_workers=2,
        max_active_slots=len(seeds),
        max_moves=256,
        force=True,
    )
    requests = int(metrics.get("worker_requests", 0) or 0)
    forwards = int(metrics.get("boundary_forwards", 0) or 0)
    rows = int(metrics.get("boundary_forward_rows", 0) or 0)
    return {
        "worker_requests": requests,
        "boundary_forwards": forwards,
        "requests_per_forward": requests / max(1, forwards),
        "rows_per_forward": rows / max(1, forwards),
        "merging": requests > forwards,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--buffer", required=True, help="a self-play buffer, for real positions"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16", choices=("fp32", "bf16"))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--batches",
        default=",".join(str(b) for b in DEFAULT_BATCHES),
        help="comma-separated batch sizes",
    )
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--skip-coalescer-check",
        action="store_true",
        help="curve only. The check needs a scheduler run of a few seconds.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    batches = tuple(int(b) for b in args.batches.split(",") if b.strip())

    from .phase_d import PhaseDConfig, PhaseDLoop
    from .search import Evaluator

    config = PhaseDConfig(
        run_dir="unused", device=args.device, precision=args.precision
    )
    loop = PhaseDLoop.__new__(PhaseDLoop)
    loop.config = config
    model = loop.load_model(args.checkpoint)
    evaluator = Evaluator(model, args.device, max(batches), precision=args.precision)

    wanted = max(batches)
    print(f"sampling {wanted} real positions from {Path(args.buffer).name}...")
    encodings, legals = sample_encodings(Path(args.buffer), wanted)
    if len(encodings) < wanted:
        print(f"  only {len(encodings)} available; capping the curve there")

    curve = forward_curve(
        evaluator,
        encodings,
        legals,
        batches,
        device=args.device,
        repetitions=args.repetitions,
    )
    print(f"\n{'batch':>7} {'ms/call':>10} {'us/row':>10} {'rows/s':>12}")
    for row in curve:
        print(
            f"{row['batch']:>7} {row['ms_per_call']:>10.2f} "
            f"{row['us_per_row']:>10.1f} {row['rows_per_second']:>12,.0f}"
        )

    point = knee(curve)
    if point.get("beyond_tested_range"):
        print(
            "\nNO KNEE within the tested range: throughput was still rising "
            f"at batch {point['batch']} "
            f"({point['rows_per_second']:,.0f} rows/s). That is the edge of "
            "the grid, NOT a knee -- re-run with larger --batches before "
            "sizing anything from it."
        )
    else:
        print(
            f"\nKNEE: batch {point['batch']} at "
            f"{point['rows_per_second']:,.0f} rows/s "
            f"(peak {point['peak_rows_per_second']:,.0f})"
        )
        print(
            "  -> aim --rust-global-batch-cap comfortably ABOVE this, and raise "
            "--rust-slots until rows/forward approaches it."
        )

    summary: dict[str, Any] = {"curve": curve, "knee": point}

    if not args.skip_coalescer_check:
        print("\ncoalescer liveness (2 shards, real net)...")
        live = coalescer_live(args.checkpoint, args.device, args.precision)
        summary["coalescer"] = live
        print(
            f"  {live['worker_requests']:,} requests -> "
            f"{live['boundary_forwards']:,} forwards "
            f"({live['requests_per_forward']:.2f} per forward, "
            f"{live['rows_per_forward']:.1f} rows/forward)"
        )
        if 1.0 < live["requests_per_forward"] < 1.2:
            print(
                "  NOTE: barely above 1.00. Arrivals are sparse at this toy "
                "scale, so a low ratio says little; the check is for exactly "
                "1.00, which means nothing merged at all."
            )
        if not live["merging"]:
            print(
                "  FAIL: every request became its own forward. Slot count "
                "cannot raise batch size while this is true -- cloud2 ran 256 "
                "slots for 46.9 rows/forward for exactly this reason.",
            )
            return 1
        print("  merging: OK")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
