"""
batched_matrix_bench.py - repeatable M8 throughput matrix for BatchedMCTS.

Runs the real-net BatchedMCTS throughput backend across model sizes, slot
counts, and sim counts, then writes a CSV with games/s, batch fill, evals/s,
and timing breakdowns.

Example:
  python -m games.kingdomino.batched_matrix_bench --device cuda \
      --models 32x6,48x4,64x6 --slots 32,64 --sims 800 \
      --games 64 --leaf_batch 6 --amp_inference --out m8_matrix.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Tuple

import torch

from games.kingdomino.network import KingdominoNet
from games.kingdomino.self_play import SelfPlayConfig, configure_torch_performance
from games.kingdomino.throughput_bench import compile_net_for_inference, run_batched


def _parse_model(spec: str) -> Tuple[int, int]:
    s = spec.lower().strip()
    if "x" not in s:
        raise ValueError(f"model spec must look like 64x6, got {spec!r}")
    c, b = s.split("x", 1)
    return int(c), int(b)


def _csv_rows(results, *, channels: int, blocks: int, slots: int, sims: int,
              games: int, leaf_batch: int, amp: bool, compiled: bool,
              compile_backend: str, compile_mode: str,
              eval_pad_to_batch: int, pin_transfer: bool,
              profile_eval_timing: bool):
    gps_values = [gps for gps, _stats in results]
    for rep_idx, (gps, stats) in enumerate(results, start=1):
        stats = stats or {}
        yield {
            "channels": channels,
            "blocks": blocks,
            "batch_slots": slots,
            "leaf_batch": leaf_batch,
            "sims": sims,
            "games": games,
            "amp": int(amp),
            "compiled": int(compiled),
            "compile_backend": compile_backend if compiled else "",
            "compile_mode": compile_mode if compiled else "",
            "eval_pad_to_batch": eval_pad_to_batch,
            "pin_transfer": int(pin_transfer),
            "profile_eval_timing": int(profile_eval_timing),
            "rep": rep_idx,
            "games_per_sec": gps,
            "games_per_sec_mean": mean(gps_values),
            "games_per_sec_median": median(gps_values),
            "mean_batch": stats.get("mean_batch", 0.0),
            "max_batch_cap": stats.get("max_batch_cap", slots * leaf_batch),
            "fill_ratio": stats.get("fill_ratio", 0.0),
            "evals_per_sec": stats.get("requests_per_sec", 0.0),
            "max_batch_seen": stats.get("max_batch_seen", 0),
            "ticks": stats.get("ticks", 0),
            "elapsed": stats.get("elapsed", 0.0),
            "step_sec": stats.get("step_sec", 0.0),
            "eval_sec": stats.get("eval_sec", 0.0),
            "update_sec": stats.get("update_sec", 0.0),
            "eval_h2d_sec": stats.get("eval_h2d_sec", 0.0),
            "eval_forward_sec": stats.get("eval_forward_sec", 0.0),
            "eval_readback_sec": stats.get("eval_readback_sec", 0.0),
            "eval_calls": stats.get("eval_calls", 0),
        }


def _ints(csv_text: str) -> Iterable[int]:
    for part in csv_text.split(","):
        part = part.strip()
        if part:
            yield int(part)


def main() -> None:
    p = argparse.ArgumentParser(description="BatchedMCTS throughput matrix")
    p.add_argument("--models", default="32x6,48x4,64x6",
                   help="comma list like 32x6,48x4,64x6")
    p.add_argument("--slots", default="32,64",
                   help="comma list of BatchedMCTS slot counts")
    p.add_argument("--sims", default="800",
                   help="comma list of simulation counts")
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--leaf_batch", type=int, default=6)
    p.add_argument("--eval_pad_to_batch", type=int, default=0,
                   help="pad live inference batches to this size; 0 disables")
    p.add_argument("--pin_transfer", action="store_true",
                   help="stage evaluator inputs in pinned host memory before H2D")
    p.add_argument("--profile_eval_timing", action="store_true",
                   help="synchronize CUDA around evaluator H2D/forward/readback "
                        "to collect detailed timing")
    p.add_argument("--device", default="cuda")
    p.add_argument("--bilinear_dim", type=int, default=64)
    p.add_argument("--amp_inference", action="store_true")
    p.add_argument("--compile_net", action="store_true",
                   help="compile the live inference net with torch.compile")
    p.add_argument("--compile_backend", default="inductor",
                   help="torch.compile backend for --compile_net")
    p.add_argument("--compile_mode", default="reduce-overhead",
                   help="torch.compile mode for --compile_net")
    p.add_argument("--no_tf32", action="store_true")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="m8_batched_matrix.csv")
    args = p.parse_args()

    rows = []
    base = args.seed * 1_000_003
    started = time.time()
    for model_spec in [m.strip() for m in args.models.split(",") if m.strip()]:
        channels, blocks = _parse_model(model_spec)
        for sims in _ints(args.sims):
            for slots in _ints(args.slots):
                eval_pad_to_batch = int(args.eval_pad_to_batch)
                if (
                    args.compile_net
                    and args.compile_backend == "cudagraphs"
                    and eval_pad_to_batch <= 0
                ):
                    eval_pad_to_batch = slots * args.leaf_batch
                print(
                    f"\n=== model={channels}x{blocks} slots={slots} "
                    f"sims={sims} games={args.games} amp={args.amp_inference} "
                    f"compiled={args.compile_net} pad={eval_pad_to_batch} "
                    f"pin={args.pin_transfer} profile={args.profile_eval_timing} ===",
                    flush=True,
                )
                cfg = SelfPlayConfig(
                    channels=channels,
                    blocks=blocks,
                    bilinear_dim=args.bilinear_dim,
                    n_simulations=sims,
                    n_determinizations=1,
                    leaf_batch=args.leaf_batch,
                    batch_slots=slots,
                    eval_pad_to_batch=eval_pad_to_batch,
                    pin_transfer=args.pin_transfer,
                    profile_eval_timing=args.profile_eval_timing,
                    device=args.device,
                    seed=args.seed,
                    allow_tf32=not args.no_tf32,
                    inference_amp=args.amp_inference,
                )
                configure_torch_performance(cfg)
                net = KingdominoNet(
                    channels=channels,
                    blocks=blocks,
                    bilinear_dim=args.bilinear_dim,
                ).to(args.device).eval()
                if args.compile_net:
                    net = compile_net_for_inference(
                        net, backend=args.compile_backend,
                        mode=args.compile_mode,
                    )
                results = run_batched(
                    cfg, net, args.games, base,
                    reps=args.repeat, warmup=args.warmup,
                )
                rows.extend(_csv_rows(
                    results,
                    channels=channels,
                    blocks=blocks,
                    slots=slots,
                    sims=sims,
                    games=args.games,
                    leaf_batch=args.leaf_batch,
                    amp=args.amp_inference,
                    compiled=args.compile_net,
                    compile_backend=args.compile_backend,
                    compile_mode=args.compile_mode,
                    eval_pad_to_batch=eval_pad_to_batch,
                    pin_transfer=args.pin_transfer,
                    profile_eval_timing=args.profile_eval_timing,
                ))
                del net
                if str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()

    out = Path(args.out)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nWrote {len(rows)} row(s) to {out}")
    print(f"Total wall time: {(time.time() - started) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
