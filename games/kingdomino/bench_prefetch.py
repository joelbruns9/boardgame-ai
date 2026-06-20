"""
bench_prefetch.py — Item 19 A/B benchmark: prefetch thread for sample_batch.

Fills a replay buffer with real self-play examples, then runs N training steps
two ways and prints a before/after table:
  - WITHOUT prefetch: sample_batch and train_step run sequentially each step.
  - WITH prefetch:     a single background thread prepares batch N+1 while the
                       GPU runs train_step on batch N (the same pattern wired
                       into run_self_play_training behind --prefetch_batches).

It also measures sample_batch latency vs train_step latency in isolation and
reports the fraction of train_step time the prefetch can hide (the theoretical
ceiling on the speedup).

Run:
  python -m games.kingdomino.bench_prefetch --device cuda --train_steps 200
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from games.kingdomino.network import KingdominoNet
from games.kingdomino.self_play import (
    SelfPlayConfig, ReplayBuffer, train_step, play_selfplay_games_batched,
)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _fill_buffer(net, cfg, buffer, min_examples, device) -> None:
    """Generate real self-play examples until the buffer holds >= min_examples."""
    seed = 0
    while len(buffer) < min_examples:
        all_examples, _scores, _stats = play_selfplay_games_batched(
            net, cfg, n_games=8, game_seed_start=seed)
        for exs in all_examples:
            buffer.add(exs)
        seed += 8
    print(f"  buffer filled: {len(buffer)} examples")


def main() -> None:
    ap = argparse.ArgumentParser(description="Item 19 prefetch A/B benchmark")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train_steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--sample_workers", type=int, default=1)
    a = ap.parse_args()

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")

    torch.manual_seed(0)
    net = KingdominoNet(channels=a.channels, blocks=a.blocks).to(a.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    # Config used both to GENERATE buffer data (batched_open_loop self-play) and
    # to drive sample_batch (batch_size / augment / device).
    cfg = SelfPlayConfig(
        channels=a.channels, blocks=a.blocks,
        engine="batched_open_loop", device=a.device,
        n_simulations=50, n_determinizations=1,
        batch_slots=32, leaf_batch=6,
        batch_size=a.batch_size, sample_workers=a.sample_workers,
        augment=True,
    )

    print(f"\n=== Item 19: prefetch A/B  (channels={a.channels}, blocks={a.blocks}, "
          f"train_steps={a.train_steps}, batch_size={a.batch_size}, "
          f"device={a.device}) ===")
    print("\n[setup] generating self-play examples for the buffer ...", flush=True)
    buffer = ReplayBuffer(50_000, n_sample_workers=a.sample_workers)
    net.eval()
    _fill_buffer(net, cfg, buffer, max(2 * a.batch_size, 600), a.device)

    net.train()
    main_rng = np.random.default_rng(123)

    def sample(rng):
        return buffer.sample_batch(a.batch_size, rng, device=a.device,
                                   augment_d4=cfg.augment)

    # ── Isolated latencies (coverage ceiling) ──
    print("[latency] measuring sample_batch vs train_step in isolation ...",
          flush=True)
    warm = sample(main_rng)
    train_step(net, warm, optimizer)          # warm caches / cudnn autotune
    _sync(a.device)
    n_probe = 20
    t = time.perf_counter()
    for _ in range(n_probe):
        _ = sample(main_rng)
    _sync(a.device)
    sample_lat = (time.perf_counter() - t) / n_probe
    probe_batches = [sample(main_rng) for _ in range(n_probe)]
    _sync(a.device)
    t = time.perf_counter()
    for b in probe_batches:
        train_step(net, b, optimizer)
    _sync(a.device)
    train_lat = (time.perf_counter() - t) / n_probe
    coverage = min(sample_lat, train_lat) / max(1e-9, train_lat)

    # ── WITHOUT prefetch (sequential) ──
    print(f"[1/2] {a.train_steps} steps WITHOUT prefetch ...", flush=True)
    seq_rng = np.random.default_rng(7)
    _sync(a.device)
    t0 = time.perf_counter()
    for _ in range(a.train_steps):
        batch = sample(seq_rng)
        train_step(net, batch, optimizer)
    _sync(a.device)
    seq_time = time.perf_counter() - t0

    # ── WITH prefetch (background thread prepares batch N+1) ──
    print(f"[2/2] {a.train_steps} steps WITH prefetch ...", flush=True)
    pf_rng = np.random.default_rng(7)
    sample_fn = lambda: sample(pf_rng)
    executor = ThreadPoolExecutor(max_workers=1)
    _sync(a.device)
    t0 = time.perf_counter()
    next_future = executor.submit(sample_fn)            # prime
    for step in range(a.train_steps):
        batch = next_future.result()
        if step + 1 < a.train_steps:
            next_future = executor.submit(sample_fn)
        train_step(net, batch, optimizer)
    _sync(a.device)
    pf_time = time.perf_counter() - t0
    executor.shutdown(wait=False)

    # ── Report ──
    impr = (1.0 - pf_time / max(1e-9, seq_time)) * 100.0
    print("\n" + "=" * 64)
    print(f"{'metric':<22}{'no prefetch':>14}{'prefetch':>14}{'change':>14}")
    print("-" * 64)
    print(f"{'total train time (s)':<22}{seq_time:>14.3f}{pf_time:>14.3f}{impr:>+13.1f}%")
    print(f"{'time / step (ms)':<22}{seq_time/a.train_steps*1e3:>14.2f}"
          f"{pf_time/a.train_steps*1e3:>14.2f}{impr:>+13.1f}%")
    print("=" * 64)
    print("isolated latencies (per step):")
    print(f"  sample_batch : {sample_lat*1e3:7.2f} ms")
    print(f"  train_step   : {train_lat*1e3:7.2f} ms")
    print(f"  prefetch can hide up to {coverage:.0%} of train_step time "
          f"(sample_lat/train_lat ceiling)")
    print(f"\n(prefetch {'FASTER' if impr > 0 else 'SLOWER'} by {abs(impr):.1f}%)")


if __name__ == "__main__":
    main()
