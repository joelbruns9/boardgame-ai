# measure_inference.py
#
# Diagnoses the actual cost of the GPU inference server vs. local CPU
# inference, so we can decide between:
#   (A) Async MCTS — fixes GPU underutilization, makes queue cost irrelevant
#   (B) Shared-memory transport — fixes queue cost specifically
#   (C) Revert to local CPU inference
#
# Outputs three numbers per scenario:
#   1. queue round-trip latency (median + p99) from the worker's view
#   2. server-side batch size distribution (mean, p50, p99)
#   3. local CPU inference baseline at matched batch sizes (apples-to-apples)
#
# Run from project root:
#   python games\cantstop\measure_inference.py --model models\cantstop\best_model.pt
#
# Optional knobs:
#   --workers N        : number of concurrent worker processes (default 8)
#   --requests N       : how many requests each worker sends (default 200)
#   --batch-size N     : feature batch size per request (default 4 — typical
#                        MCTS batch_size_cap)
#   --device {cuda,cpu}: server device (default cuda if available)
#   --skip-cpu-baseline: skip the local CPU inference baseline run

import os
import sys
import time
import argparse
import multiprocessing as mp
from statistics import median

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---- WORKER: round-trip latency measurement ----
#
# Windows spawn context requires Queue objects to be inherited via
# Pool initargs — they CANNOT be pickled and sent via pool.map() args.
# So we stash the queues into module-level globals during init, then
# pull them out in the worker function.

_WORKER_REQUEST_QUEUE = None
_WORKER_RESPONSE_QUEUES = None


def _init_latency_worker(request_queue, response_queues):
    """Pool initializer — stash inherited queues for the worker."""
    global _WORKER_REQUEST_QUEUE, _WORKER_RESPONSE_QUEUES
    _WORKER_REQUEST_QUEUE = request_queue
    _WORKER_RESPONSE_QUEUES = response_queues


def _latency_worker(args):
    """
    A single worker that times round-trips to the inference server.

    Returns a list of per-request latency in microseconds.
    """
    (worker_id, num_requests, batch_size,
     feature_size, action_space, seed) = args

    # Pin RNG so each worker generates different but reproducible data.
    rng = np.random.default_rng(seed)

    from games.cantstop.inference_server import InferenceClient
    client = InferenceClient(
        _WORKER_REQUEST_QUEUE,
        _WORKER_RESPONSE_QUEUES[worker_id],
        worker_id,
    )

    # Pre-generate request payloads so we're not measuring numpy alloc.
    payloads = []
    for _ in range(num_requests):
        feats = rng.standard_normal(
            (batch_size, feature_size)
        ).astype(np.float32)
        masks = np.ones((batch_size, action_space), dtype=bool)
        # Random subset masked to simulate real legality masks.
        masks[:, :action_space // 2] = (
            rng.random((batch_size, action_space // 2)) > 0.3
        )
        payloads.append((feats, masks))

    latencies_us = []
    # Tiny warmup so we don't measure first-request CUDA-init weirdness.
    feats0, masks0 = payloads[0]
    client.infer(feats0, masks0)

    for feats, masks in payloads:
        t0 = time.perf_counter()
        client.infer(feats, masks)
        dt_us = (time.perf_counter() - t0) * 1e6
        latencies_us.append(dt_us)

    return latencies_us


# ---- INSTRUMENTED SERVER ----
# We can't modify inference_server.py without affecting the running
# project, so we copy its loop here with a single addition: log batch
# sizes per forward pass.

def _instrumented_server_loop(request_queue, response_queues,
                              control_queue, initial_model_path,
                              device, batch_log_queue):
    """
    Same as _server_loop in inference_server.py but pushes the
    per-batch size to `batch_log_queue` so the main process can
    summarize the distribution.
    """
    import queue
    import traceback
    import torch.nn.functional as F

    from games.cantstop.evaluate import load_model

    try:
        model = load_model(initial_model_path, device)
        model.eval()
    except Exception as e:
        print(f"[instrumented-server] FATAL: {e}", flush=True)
        traceback.print_exc()
        return

    print(f"[instrumented-server] started on device={device}", flush=True)

    SERVER_DRAIN_MAX = 64
    SERVER_POLL_INTERVAL = 0.05

    pending_shutdown = False

    while not pending_shutdown:
        # Drain control queue.
        try:
            while True:
                cmd, payload = control_queue.get_nowait()
                if cmd == 'shutdown':
                    pending_shutdown = True
                    break
        except queue.Empty:
            pass
        if pending_shutdown:
            break

        try:
            first = request_queue.get(timeout=SERVER_POLL_INTERVAL)
        except queue.Empty:
            continue
        if first[0] is None:
            pending_shutdown = True
            break

        collected = [first]
        for _ in range(SERVER_DRAIN_MAX - 1):
            try:
                req = request_queue.get_nowait()
            except queue.Empty:
                break
            if req[0] is None:
                pending_shutdown = True
                break
            collected.append(req)

        try:
            all_features = []
            all_masks = []
            chunk_sizes = []
            for (wid, rid, feats, msks) in collected:
                all_features.append(feats)
                all_masks.append(msks)
                chunk_sizes.append(feats.shape[0])

            features_np = np.concatenate(all_features, axis=0)
            masks_np = np.concatenate(all_masks, axis=0)

            features_t = torch.from_numpy(features_np).to(device)
            masks_t = torch.from_numpy(masks_np).to(device)

            # Time the actual forward pass — we want to know if the
            # GPU work itself is the bottleneck.
            t_fwd_start = time.perf_counter()
            with torch.no_grad():
                values, logits = model(features_t, masks_t)
                probs = F.softmax(logits, dim=-1)
                if device == 'cuda':
                    torch.cuda.synchronize()
                values_np = values.detach().cpu().numpy().astype(np.float32)
                probs_np = probs.detach().cpu().numpy().astype(np.float32)
            t_fwd_us = (time.perf_counter() - t_fwd_start) * 1e6

            offset = 0
            for (wid, rid, _, _), n in zip(collected, chunk_sizes):
                v_chunk = values_np[offset:offset + n]
                p_chunk = probs_np[offset:offset + n]
                offset += n
                response_queues[wid].put((rid, v_chunk, p_chunk))

            # Log: num_requests_in_batch, total_samples, forward_us
            try:
                batch_log_queue.put_nowait(
                    (len(collected), features_np.shape[0], t_fwd_us)
                )
            except Exception:
                pass

        except Exception as e:
            print(f"[instrumented-server] batch failed: {e}", flush=True)
            traceback.print_exc()
            err_msg = str(e)
            for (wid, rid, _, _) in collected:
                try:
                    response_queues[wid].put((rid, None, err_msg))
                except Exception:
                    pass

    print("[instrumented-server] shutdown", flush=True)


# ---- ANALYSIS HELPERS ----

def percentile(values, p):
    """Simple percentile — values need not be sorted."""
    if not values:
        return float('nan')
    s = sorted(values)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[max(0, min(k, len(s) - 1))]


def summarize_latencies(label, latencies_us):
    if not latencies_us:
        print(f"  {label}: no data")
        return
    n = len(latencies_us)
    mean = sum(latencies_us) / n
    print(f"  {label}: n={n}")
    print(f"    mean    = {mean:>8.1f} µs")
    print(f"    median  = {percentile(latencies_us, 50):>8.1f} µs")
    print(f"    p90     = {percentile(latencies_us, 90):>8.1f} µs")
    print(f"    p99     = {percentile(latencies_us, 99):>8.1f} µs")
    print(f"    max     = {max(latencies_us):>8.1f} µs")


def summarize_batches(batch_log):
    """batch_log: list of (num_requests, total_samples, forward_us)."""
    if not batch_log:
        print("  (no batch data captured)")
        return

    req_counts = [b[0] for b in batch_log]
    sample_counts = [b[1] for b in batch_log]
    forward_times = [b[2] for b in batch_log]

    print(f"  {len(batch_log)} batches observed")
    print(f"  Requests per batch:")
    print(f"    mean    = {sum(req_counts)/len(req_counts):>6.2f}")
    print(f"    median  = {percentile(req_counts, 50):>6.1f}")
    print(f"    p90     = {percentile(req_counts, 90):>6.1f}")
    print(f"    p99     = {percentile(req_counts, 99):>6.1f}")
    print(f"    max     = {max(req_counts):>6d}")
    print(f"  Samples per batch (= GPU batch size):")
    print(f"    mean    = {sum(sample_counts)/len(sample_counts):>6.2f}")
    print(f"    median  = {percentile(sample_counts, 50):>6.1f}")
    print(f"    p90     = {percentile(sample_counts, 90):>6.1f}")
    print(f"    p99     = {percentile(sample_counts, 99):>6.1f}")
    print(f"    max     = {max(sample_counts):>6d}")
    print(f"  Forward pass time (server-side, includes cuda.sync + cpu copy):")
    print(f"    mean    = {sum(forward_times)/len(forward_times):>6.1f} µs")
    print(f"    median  = {percentile(forward_times, 50):>6.1f} µs")
    print(f"    p99     = {percentile(forward_times, 99):>6.1f} µs")

    # Distribution histogram of GPU batch sizes
    buckets = [(1, 1), (2, 4), (5, 8), (9, 16), (17, 32),
               (33, 64), (65, 128), (129, 999)]
    print(f"  GPU batch size distribution:")
    for lo, hi in buckets:
        c = sum(1 for s in sample_counts if lo <= s <= hi)
        pct = 100 * c / len(sample_counts)
        bar = '█' * int(pct / 2)
        print(f"    [{lo:>3d}..{hi:>3d}]: {c:>5d} ({pct:>5.1f}%) {bar}")


def cpu_inference_baseline(model_path, batch_size, num_iters,
                           feature_size, action_space):
    """
    Time local CPU inference at the same batch size workers would
    naturally send. This is the apples-to-apples comparison: how long
    would the same forward pass take if we never crossed a queue?
    """
    import torch.nn.functional as F
    from games.cantstop.evaluate import load_model

    model = load_model(model_path, 'cpu')
    model.eval()

    rng = np.random.default_rng(12345)
    feats = rng.standard_normal(
        (batch_size, feature_size)
    ).astype(np.float32)
    masks = np.ones((batch_size, action_space), dtype=bool)

    feats_t = torch.from_numpy(feats)
    masks_t = torch.from_numpy(masks)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            v, l = model(feats_t, masks_t)
            _ = F.softmax(l, dim=-1)

    times_us = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            v, l = model(feats_t, masks_t)
            p = F.softmax(l, dim=-1)
            _ = v.numpy()
            _ = p.numpy()
        times_us.append((time.perf_counter() - t0) * 1e6)
    return times_us


# ---- MAIN ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', default='models/cantstop/best_model.pt'
    )
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--requests', type=int, default=200,
                        help='Round-trips per worker')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Feature batch size per request (MCTS '
                             'sends ~1-8 typically)')
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--skip-cpu-baseline', action='store_true')
    args = parser.parse_args()

    from games.cantstop.features import FEATURE_SIZE, ACTION_SPACE

    print(f"\n{'='*60}")
    print(f"INFERENCE PATH MEASUREMENT")
    print(f"{'='*60}")
    print(f"  Model:        {args.model}")
    print(f"  Server dev:   {args.device}")
    print(f"  Workers:      {args.workers}")
    print(f"  Req/worker:   {args.requests}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Feature size: {FEATURE_SIZE}")
    print(f"  Action space: {ACTION_SPACE}")
    print(f"{'='*60}")

    ctx = mp.get_context('spawn')
    request_queue = ctx.Queue()
    response_queues = [ctx.Queue() for _ in range(args.workers)]
    control_queue = ctx.Queue()
    batch_log_queue = ctx.Queue()

    # ---- Start instrumented server ----
    print(f"\n[1/3] Starting instrumented server "
          f"(device={args.device})...")
    server_proc = ctx.Process(
        target=_instrumented_server_loop,
        args=(request_queue, response_queues, control_queue,
              args.model, args.device, batch_log_queue),
        daemon=False,
    )
    server_proc.start()
    time.sleep(3.0)
    if not server_proc.is_alive():
        print("FATAL: server failed to start"); return

    # ---- Run workers ----
    print(f"\n[2/3] Running {args.workers} workers × "
          f"{args.requests} requests "
          f"({args.workers * args.requests} total round-trips)...")
    worker_args = [
        (i, args.requests, args.batch_size,
         FEATURE_SIZE, ACTION_SPACE, i * 1000 + 7)
        for i in range(args.workers)
    ]

    t_wall_start = time.perf_counter()
    with ctx.Pool(
        processes=args.workers,
        initializer=_init_latency_worker,
        initargs=(request_queue, response_queues),
    ) as pool:
        per_worker_latencies = pool.map(_latency_worker, worker_args)
    t_wall = time.perf_counter() - t_wall_start

    all_latencies = []
    for lst in per_worker_latencies:
        all_latencies.extend(lst)

    total_requests = len(all_latencies)
    throughput = total_requests / t_wall

    # ---- Shutdown server ----
    try:
        control_queue.put(('shutdown', None))
        request_queue.put((None, 0, None, None))
    except Exception:
        pass
    server_proc.join(timeout=5.0)
    if server_proc.is_alive():
        server_proc.terminate()

    # Drain batch log
    batch_log = []
    try:
        while True:
            batch_log.append(batch_log_queue.get_nowait())
    except Exception:
        pass

    # ---- Report ----
    print(f"\n{'='*60}")
    print(f"RESULTS — ROUND-TRIP LATENCY (worker-side)")
    print(f"{'='*60}")
    summarize_latencies("All requests pooled", all_latencies)
    print(f"  Wall time:    {t_wall:.2f}s")
    print(f"  Throughput:   {throughput:.0f} req/s aggregate")
    print(f"  Per-worker:   {throughput / args.workers:.0f} req/s")

    print(f"\n{'='*60}")
    print(f"RESULTS — SERVER-SIDE BATCHING")
    print(f"{'='*60}")
    summarize_batches(batch_log)

    if not args.skip_cpu_baseline:
        print(f"\n[3/3] Local CPU inference baseline "
              f"(batch={args.batch_size})...")
        cpu_times = cpu_inference_baseline(
            args.model, args.batch_size, 200,
            FEATURE_SIZE, ACTION_SPACE
        )
        print(f"\n{'='*60}")
        print(f"RESULTS — LOCAL CPU INFERENCE BASELINE")
        print(f"{'='*60}")
        summarize_latencies(
            f"Local model forward (no IPC, batch={args.batch_size})",
            cpu_times
        )

    # ---- Diagnosis ----
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS")
    print(f"{'='*60}")
    median_rt = percentile(all_latencies, 50)
    if not args.skip_cpu_baseline:
        median_cpu = percentile(cpu_times, 50)
        overhead = median_rt - median_cpu
        print(f"  Median queue overhead: "
              f"{overhead:>+7.1f} µs "
              f"(server route {median_rt:.0f} - cpu local {median_cpu:.0f})")
        if overhead > 1000:
            print(f"  ⚠ Queue overhead > 1ms — shared memory / structural"
                  f" fix may help.")
        elif overhead > 200:
            print(f"  ◐ Queue overhead modest. Async MCTS likely wins by"
                  f" letting batches grow.")
        else:
            print(f"  ✓ Queue overhead negligible. Bottleneck is "
                  f"elsewhere (batch sizes / sync MCTS).")

    if batch_log:
        sample_counts = [b[1] for b in batch_log]
        mean_batch = sum(sample_counts) / len(sample_counts)
        if mean_batch < 8:
            print(f"  ⚠ Mean GPU batch = {mean_batch:.1f}. GPU is "
                  f"underutilized. Async MCTS would let batches grow"
                  f" to {args.workers * 16}+.")
        elif mean_batch < 32:
            print(f"  ◐ Mean GPU batch = {mean_batch:.1f}. Room for"
                  f" growth — async MCTS would help.")
        else:
            print(f"  ✓ Mean GPU batch = {mean_batch:.1f}. GPU is being"
                  f" exercised.")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()