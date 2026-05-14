# benchmark_gpu_batch.py
#
# Stage 1 confirmation: sweep batch size against the GPU model and
# measure forward-pass latency + throughput. Answers:
#
#   1. Does forward time stay flat with batch size (GPU underutilized)?
#   2. At what batch size does GPU throughput plateau?
#   3. What's the realistic batch target for async MCTS?
#
# No IPC, no workers. Pure model.forward on CUDA. This isolates GPU
# behavior from queue/pickle costs we measured in run #1.
#
# Why this matters: in the previous run, GPU forward at batch=32
# averaged 1.2ms. If forward at batch=128 is still ~1-2ms, the GPU
# is bored and async MCTS (which would get us to batch=128+) is a
# huge win. If forward at batch=128 is 8ms, the GPU is saturated and
# async MCTS would only help modestly.
#
# Usage:
#   python games\cantstop\benchmark_gpu_batch.py --model models\cantstop\best_model.pt

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def bench_batch(model, batch_size, feature_size, action_space,
                device, num_iters=200, warmup=20):
    """
    Time `num_iters` forward passes at the given batch size.

    Returns dict with per-call latency stats and throughput numbers.

    Includes cuda.synchronize() before stopping the clock — without it
    we'd just be measuring kernel-launch time, not actual GPU work.
    Also includes the cpu copy at the end since the real server pays
    that cost too.
    """
    rng = np.random.default_rng(42)
    feats_np = rng.standard_normal(
        (batch_size, feature_size)
    ).astype(np.float32)
    masks_np = np.ones((batch_size, action_space), dtype=bool)
    # Random subset masked to simulate real legality masks.
    masks_np[:, :action_space // 2] = (
        rng.random((batch_size, action_space // 2)) > 0.3
    )

    # Pre-move tensors to device — we want to measure forward pass,
    # not h2d copy. (The real server pays h2d once per batch; we'll
    # measure it separately below.)
    feats_t = torch.from_numpy(feats_np).to(device)
    masks_t = torch.from_numpy(masks_np).to(device)

    # Warmup — first call on a fresh batch size triggers cuDNN
    # heuristic selection + kernel autotuning. We don't want that
    # in the measurement.
    with torch.no_grad():
        for _ in range(warmup):
            v, l = model(feats_t, masks_t)
            p = F.softmax(l, dim=-1)
        if device == 'cuda':
            torch.cuda.synchronize()

    # ---- Forward pass only (model.forward + softmax + sync) ----
    times_fwd_us = []
    with torch.no_grad():
        for _ in range(num_iters):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            v, l = model(feats_t, masks_t)
            p = F.softmax(l, dim=-1)
            if device == 'cuda':
                torch.cuda.synchronize()
            dt_us = (time.perf_counter() - t0) * 1e6
            times_fwd_us.append(dt_us)

    # ---- Full cycle (forward + cpu copy back, what server actually does) ----
    times_full_us = []
    with torch.no_grad():
        for _ in range(num_iters):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            v, l = model(feats_t, masks_t)
            p = F.softmax(l, dim=-1)
            v_np = v.detach().cpu().numpy()
            p_np = p.detach().cpu().numpy()
            dt_us = (time.perf_counter() - t0) * 1e6
            times_full_us.append(dt_us)

    # ---- Per-batch h2d transfer cost ----
    # This is what an IPC payload pays when arriving on the server.
    times_h2d_us = []
    feats_cpu = torch.from_numpy(feats_np)  # already on cpu
    masks_cpu = torch.from_numpy(masks_np)
    for _ in range(num_iters):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        f_dev = feats_cpu.to(device)
        m_dev = masks_cpu.to(device)
        if device == 'cuda':
            torch.cuda.synchronize()
        dt_us = (time.perf_counter() - t0) * 1e6
        times_h2d_us.append(dt_us)

    def stats(xs):
        s = sorted(xs)
        n = len(s)
        return {
            'mean':   sum(s) / n,
            'median': s[n // 2],
            'p90':    s[int(0.90 * (n - 1))],
            'p99':    s[int(0.99 * (n - 1))],
            'min':    s[0],
        }

    fwd  = stats(times_fwd_us)
    full = stats(times_full_us)
    h2d  = stats(times_h2d_us)

    # Throughput at the median (steady-state)
    samples_per_sec = batch_size / (full['median'] / 1e6)

    return {
        'batch_size':  batch_size,
        'fwd':         fwd,
        'full':        full,
        'h2d':         h2d,
        'samples_per_sec': samples_per_sec,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', default='models/cantstop/best_model.pt'
    )
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument(
        '--batches', type=str,
        default='1,2,4,8,16,32,64,128,256,512',
        help='Comma-separated batch sizes to sweep.'
    )
    parser.add_argument('--iters', type=int, default=200)
    args = parser.parse_args()

    from games.cantstop.features import FEATURE_SIZE, ACTION_SPACE
    from games.cantstop.evaluate import load_model

    batch_sizes = [int(x) for x in args.batches.split(',')]

    print(f"\n{'='*72}")
    print(f"GPU BATCH SIZE SWEEP")
    print(f"{'='*72}")
    print(f"  Model:        {args.model}")
    print(f"  Device:       {args.device}")
    if args.device == 'cuda':
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
        # Make benchmark reproducible — disable cuDNN autotune randomness
        torch.backends.cudnn.benchmark = False
    print(f"  Iters/batch:  {args.iters} (after 20 warmup)")
    print(f"  Sweep:        {batch_sizes}")
    print(f"{'='*72}\n")

    model = load_model(args.model, args.device)
    model.eval()

    results = []
    for bs in batch_sizes:
        print(f"  benchmarking batch_size={bs:>4d}...", end=' ', flush=True)
        r = bench_batch(model, bs, FEATURE_SIZE, ACTION_SPACE,
                        args.device, num_iters=args.iters)
        results.append(r)
        print(f"fwd_median={r['fwd']['median']:>6.1f}µs  "
              f"full_median={r['full']['median']:>6.1f}µs  "
              f"thru={r['samples_per_sec']/1e3:>7.1f}k samples/s")

    # ---- Table ----
    print(f"\n{'='*72}")
    print(f"RESULTS TABLE")
    print(f"{'='*72}")
    print(f"  {'batch':>5} {'fwd µs':>10} {'full µs':>10} {'h2d µs':>9} "
          f"{'thru ks/s':>10} {'µs/sample':>10}")
    print(f"  {'-'*5:>5} {'-'*10:>10} {'-'*10:>10} {'-'*9:>9} "
          f"{'-'*10:>10} {'-'*10:>10}")
    for r in results:
        per_sample = r['full']['median'] / r['batch_size']
        print(f"  {r['batch_size']:>5d} "
              f"{r['fwd']['median']:>10.1f} "
              f"{r['full']['median']:>10.1f} "
              f"{r['h2d']['median']:>9.1f} "
              f"{r['samples_per_sec']/1e3:>10.1f} "
              f"{per_sample:>10.2f}")

    # ---- Analysis ----
    print(f"\n{'='*72}")
    print(f"ANALYSIS")
    print(f"{'='*72}")

    fwd_1   = next(r['fwd']['median'] for r in results if r['batch_size'] == 1) \
              if any(r['batch_size'] == 1 for r in results) else None
    fwd_max = results[-1]['fwd']['median']
    bs_max  = results[-1]['batch_size']

    if fwd_1 is not None:
        scaling = fwd_max / fwd_1
        ideal_scaling = bs_max  # linear scaling = no parallelism gain
        efficiency = ideal_scaling / scaling
        print(f"  Forward latency batch={bs_max} / batch=1 = "
              f"{scaling:.2f}x")
        print(f"  Linear scaling would be: {ideal_scaling:.0f}x")
        print(f"  GPU parallelism efficiency: {efficiency:.1f}x speedup "
              f"over serial")

        if scaling < 2.0:
            print(f"  ✓ GPU forward time barely grows with batch size — "
                  f"GPU is underutilized at small batches.")
            print(f"    Async MCTS that grows batches to {bs_max}+ "
                  f"will give large gains.")
        elif scaling < 8.0:
            print(f"  ◐ GPU forward time grows sublinearly — "
                  f"there's room to grow, but diminishing returns.")
        else:
            print(f"  ⚠ GPU forward time scales near-linearly — "
                  f"GPU is already saturated; async MCTS won't help "
                  f"much.")

    # Find the knee — smallest batch where throughput is within 80%
    # of the peak.
    peak_thru = max(r['samples_per_sec'] for r in results)
    knee_bs = None
    for r in results:
        if r['samples_per_sec'] >= 0.80 * peak_thru:
            knee_bs = r['batch_size']
            break
    if knee_bs is not None:
        print(f"\n  Peak throughput: {peak_thru/1e3:.1f}k samples/s "
              f"(at batch={results[-1]['batch_size'] if results[-1]['samples_per_sec'] == peak_thru else 'see table'})")
        print(f"  Throughput knee (80% of peak): batch={knee_bs}")
        print(f"  → Async MCTS target: aim for "
              f"~{max(knee_bs, 32)} concurrent in-flight sims per worker")

    # Effective amortization vs. current 32-batch operating point
    cur_32 = next((r for r in results if r['batch_size'] == 32), None)
    if cur_32 is not None:
        cur_thru = cur_32['samples_per_sec']
        max_thru = results[-1]['samples_per_sec']
        print(f"\n  Current operating point (batch=32 in your data): "
              f"{cur_thru/1e3:.1f}k samples/s")
        print(f"  If we push to batch={results[-1]['batch_size']}: "
              f"{max_thru/1e3:.1f}k samples/s "
              f"({max_thru / cur_thru:.1f}x improvement)")

    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()