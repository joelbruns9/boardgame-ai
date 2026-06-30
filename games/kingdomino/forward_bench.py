"""
forward_bench.py — pure network forward throughput vs batch size.

Strips away MCTS, IPC, and batching policy: just times `net.forward` on
GPU-resident inputs at a range of batch sizes.  This isolates the one cost the
inference server can never go below — the forward itself — and answers the
questions that gate the leaf-parallelization decision:

  * peak evals/s the GPU sustains on this net  → the ceiling any batching
    change (leaf parallelization, more concurrency) can approach;
  * the batch size where evals/s plateaus       → the max_batch worth targeting;
  * how steeply forward time grows with batch   → flat means big headroom for
    fuller batches; linear-from-small means the win is modest.

Compare the peak here against the live A3 baseline (~1470 evals/s at fill ~74%):
if the peak is several times higher, the live system is starved and batching
changes have room; if the peak is ~1470, you're already forward-bound and
leaf parallelization buys little.

Inputs are placed on-device once per batch size, so this measures pure compute
and excludes the host->device copy (which prior server-side timing showed is
~0.2 ms — negligible).  Run:

  python -m games.kingdomino.forward_bench --device cuda --channels 64 --blocks 6
"""
from __future__ import annotations

import argparse
import time

import torch

from games.kingdomino.network import KingdominoNet
from games.kingdomino.encoder import encode_state
from games.kingdomino.game import GameState


def _input_shapes():
    """Get the real encoder output shapes so we don't hard-code constants."""
    mb, ob, flat = encode_state(GameState.new(seed=0), 0)
    return mb.shape, ob.shape, flat.shape[0]


def _maybe_channels_last(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled:
        return x.contiguous(memory_format=torch.channels_last)
    return x


def bench_batch(net, B, device, mb_shape, ob_shape, flat_size, iters, warmup,
                amp: bool = False, channels_last: bool = False):
    cuda = device.startswith("cuda")
    mb = _maybe_channels_last(torch.rand(B, *mb_shape, device=device), channels_last)
    ob = _maybe_channels_last(torch.rand(B, *ob_shape, device=device), channels_last)
    flat = torch.rand(B, flat_size, device=device)
    use_amp = bool(amp and cuda)
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                net(mb, ob, flat)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                # Return values (own, opp, win_prob, logits) intentionally discarded —
                # only forward time matters here.
                net(mb, ob, flat)
        if cuda:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    return dt * 1000.0, (B / dt if dt > 0 else 0.0)   # ms/forward, evals/s


def bench_legal_batch(net, B, legal_n, device, mb_shape, ob_shape, flat_size,
                      iters, warmup, amp: bool = False,
                      channels_last: bool = False):
    cuda = device.startswith("cuda")
    mb = _maybe_channels_last(torch.rand(B, *mb_shape, device=device), channels_last)
    ob = _maybe_channels_last(torch.rand(B, *ob_shape, device=device), channels_last)
    flat = torch.rand(B, flat_size, device=device)
    legal_idx = torch.randint(0, 3390, (B, legal_n), device=device)
    use_amp = bool(amp and cuda)
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                net.forward_legal(mb, ob, flat, legal_idx)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                # Return values (own, opp, win_prob, legal_logits) intentionally
                # discarded — only forward time matters here.
                net.forward_legal(mb, ob, flat, legal_idx)
        if cuda:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    return dt * 1000.0, (B / dt if dt > 0 else 0.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Pure forward throughput vs batch size")
    p.add_argument("--device", default="cuda")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--bilinear_dim", type=int, default=64)
    p.add_argument("--norm", choices=["group", "batch"], default="group",
                   help="normalization layer in the residual trunk")
    p.add_argument("--batches", default="1,2,4,8,16,24,32,48,64,96,128,192,256")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--baseline_evals", type=float, default=1470.0,
                   help="live A3 evals/s to compare the ceiling against")
    p.add_argument("--cudnn_benchmark", action="store_true",
                   help="enable cuDNN autotuner (best-case kernels for fixed shapes)")
    p.add_argument("--no_tf32", action="store_true",
                   help="disable TF32 CUDA matmul/convolution")
    p.add_argument("--amp_inference", action="store_true",
                   help="use CUDA float16 autocast during the forward benchmark")
    p.add_argument("--channels_last", action="store_true",
                   help="use channels-last memory format for board tensors and "
                        "conv weights")
    p.add_argument("--legal_counts", default=None,
                   help="optional comma list of legal action counts to benchmark "
                        "with net.forward_legal, e.g. 20,50,100")
    a = p.parse_args()

    if a.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = not a.no_tf32
        torch.backends.cudnn.allow_tf32 = not a.no_tf32
        if not a.no_tf32 and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    if a.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    mb_shape, ob_shape, flat_size = _input_shapes()
    net = KingdominoNet(channels=a.channels, blocks=a.blocks,
                        bilinear_dim=a.bilinear_dim,
                        norm=a.norm).to(a.device).eval()
    if a.channels_last:
        net = net.to(memory_format=torch.channels_last)
    batches = [int(x) for x in a.batches.split(",") if x.strip()]

    print(f"device={a.device}  net={a.channels}ch/{a.blocks}b norm={a.norm}  "
          f"mb={tuple(mb_shape)} flat={flat_size}  "
          f"cudnn_benchmark={a.cudnn_benchmark}  "
          f"tf32={not a.no_tf32}  amp={a.amp_inference}  "
          f"channels_last={a.channels_last}")
    print(f"{'batch':>6}{'ms/fwd':>10}{'evals/s':>12}{'us/sample':>12}{'vs live':>9}")
    print("-" * 49)

    best_b, best_eps = 0, 0.0
    prev_eps = None
    for B in batches:
        try:
            ms, eps = bench_batch(net, B, a.device, mb_shape, ob_shape,
                                  flat_size, a.iters, a.warmup,
                                  amp=a.amp_inference,
                                  channels_last=a.channels_last)
        except RuntimeError as e:
            msg = str(e).splitlines()[0][:42]
            print(f"{B:>6}   stopped (likely OOM): {msg}")
            break
        print(f"{B:>6}{ms:>10.2f}{eps:>12.0f}{1e6/eps:>12.1f}"
              f"{eps/a.baseline_evals:>8.1f}x")
        if eps > best_eps:
            best_b, best_eps = B, eps
        prev_eps = eps

    print("-" * 49)
    print(f"peak forward throughput : {best_eps:,.0f} evals/s at batch {best_b}")
    print(f"live A3 baseline        : {a.baseline_evals:,.0f} evals/s "
          f"(fill ~74%)  ->  headroom ~{best_eps/a.baseline_evals:.1f}x")
    print("Read: the knee (evals/s stops climbing) = max_batch worth targeting; "
          "peak evals/s = the ceiling batching changes can approach. If peak >> "
          "live, you're starved and leaf parallelization has room.")

    if a.legal_counts:
        legal_counts = [int(x) for x in a.legal_counts.split(",") if x.strip()]
        print("\nLegal-only policy head benchmark (three score/win heads + policy)")
        print(f"{'legal':>6}{'batch':>8}{'ms/fwd':>10}{'evals/s':>12}"
              f"{'speedup':>10}")
        print("-" * 48)
        for legal_n in legal_counts:
            for B in batches:
                try:
                    ms, eps = bench_legal_batch(
                        net, B, legal_n, a.device, mb_shape, ob_shape,
                        flat_size, a.iters, a.warmup, amp=a.amp_inference,
                        channels_last=a.channels_last)
                except RuntimeError as e:
                    msg = str(e).splitlines()[0][:32]
                    print(f"{legal_n:>6}{B:>8}   stopped: {msg}")
                    break
                speedup = eps / best_eps if best_eps > 0 else 0.0
                print(f"{legal_n:>6}{B:>8}{ms:>10.2f}{eps:>12.0f}"
                      f"{speedup:>9.2f}x")


if __name__ == "__main__":
    main()
