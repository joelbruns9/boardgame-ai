"""
bench_compile.py — Item 20 A/B benchmark: torch.compile on the inference net.

This benchmark answers ONE question for the cloud training command: should we
pass ``--compile``?  It does so in two sections:

  SECTION 1 — variable-shape microbenchmark (the decisive test).
    During self-play the filled inference batch varies tick to tick (mean ~45,
    range ~6–192, right-skewed) — it is NOT the fixed ``batch_slots*leaf_batch``.
    torch.compile recompiles whenever it sees a new input shape; a stream of
    novel shapes can trigger a *recompilation storm* that silently eats the
    1.2–1.5× kernel-fusion win (and, once the dynamo cache limit is exceeded,
    degrades to permanent eager fallback + guard thrash).  This section feeds the
    realistic right-skewed batch-size distribution through the net under three
    compile settings — eager, ``dynamic=False``, ``dynamic=True`` — with
    ``TORCH_LOGS=recompiles`` logging enabled programmatically, and counts the
    recompiles each one triggers.  Whichever compile mode (if any) is both
    *faster* than eager AND *recompile-stable* is the one to use.

  SECTION 2 — end-to-end self-play A/B (corroboration).
    Runs ``play_selfplay_games_batched`` (the real tick loop) WITHOUT and WITH
    ``compile_net`` and prints the games/sec delta.  The compiled run is executed
    TWICE: the first pays graph-capture/autotune, only the SECOND is reported.

The final verdict line is exactly ``USE --compile`` or ``DO NOT USE --compile``.

NOTE: torch.compile needs Triton (Linux cu128 wheel ships it; Windows does not),
so this is only meaningful on the cloud box.  Without Triton, inductor falls back
to eager → ~0% speedup → the bench will correctly say DO NOT USE.

Run:
  python -m games.kingdomino.bench_compile --device cuda --sims 200 --games 20
"""
from __future__ import annotations

# TORCH_LOGS must be set BEFORE `import torch` to take effect via the env path.
# We also call torch._logging.set_logs(recompiles=True) below (the programmatic
# equivalent) so this works regardless of import order; setdefault lets an
# explicit user TORCH_LOGS override win.
import os
os.environ.setdefault("TORCH_LOGS", "recompiles")

import argparse
import copy
import logging
import math
import statistics
import time

import numpy as np
import torch

from games.kingdomino.encoder import (
    CANVAS_SIZE, NUM_BOARD_CHANNELS, FLAT_SIZE,
)
from games.kingdomino.network import KingdominoNet
from games.kingdomino.self_play import SelfPlayConfig, play_selfplay_games_batched


# ── recompile accounting ────────────────────────────────────────────────────
class _RecompileCounter(logging.Handler):
    """Counts torch.compile recompile log records.

    With ``torch._logging.set_logs(recompiles=True)`` every guard-failure
    recompile emits a record under the ``torch._dynamo`` logger tree; we attach
    this handler there and tally records that mention a recompile.  This is the
    programmatic read of ``TORCH_LOGS=recompiles``.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        low = msg.lower()
        if "recompil" in low:  # matches "Recompiling", "recompile", "recompiles"
            self.records.append(msg)

    @property
    def count(self) -> int:
        return len(self.records)

    def reset(self) -> None:
        self.records.clear()


def _install_recompile_logging() -> _RecompileCounter:
    handler = _RecompileCounter()
    try:
        torch._logging.set_logs(recompiles=True)
    except Exception:
        pass
    logger = logging.getLogger("torch._dynamo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)          # child loggers propagate up to here
    return handler


# ── realistic batch-size distribution ───────────────────────────────────────
def _sample_batch_sizes(n: int, seed: int, lo: int = 6, hi: int = 192,
                        target_mean: float = 45.0) -> list[int]:
    """Right-skewed batch sizes (mean ~45, clipped to [lo, hi]).

    Lognormal gives the right skew (a long thin tail toward the 192 cap with most
    mass near the mode) seen in the real leaf-eval batch stream.
    """
    rng = np.random.default_rng(seed)
    # median ~ exp(mu); pick mu/sigma so the clipped mean lands near target_mean.
    sigma = 0.55
    mu = math.log(target_mean) - 0.5 * sigma * sigma
    raw = rng.lognormal(mean=mu, sigma=sigma, size=n)
    sizes = np.clip(np.rint(raw), lo, hi).astype(int)
    return sizes.tolist()


def _make_inputs(batch_sizes: list[int], device: str):
    """Pre-build synthetic (mb, ob, flat) tensors for each batch size on device.

    Random values are fine: we measure kernel/compile behaviour vs shape, not
    output correctness.  Pre-building keeps H2D out of the timed loop.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    cache: dict[int, tuple] = {}
    out = []
    for b in batch_sizes:
        if b not in cache:
            mb = torch.randn(b, NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE, generator=g)
            ob = torch.randn(b, NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE, generator=g)
            flat = torch.randn(b, FLAT_SIZE, generator=g)
            cache[b] = (mb.to(device), ob.to(device), flat.to(device))
        out.append(cache[b])
    return out


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _run_pass(net, inputs, device: str) -> tuple[float, list[float]]:
    """Run the net over the input sequence once; return (total_s, per_call_s)."""
    per_call = []
    _sync(device)
    t_start = time.perf_counter()
    with torch.inference_mode():
        for mb, ob, flat in inputs:
            t0 = time.perf_counter()
            net(mb, ob, flat)
            _sync(device)
            per_call.append(time.perf_counter() - t0)
    total = time.perf_counter() - t_start
    return total, per_call


def _bench_compile_mode(base_net, inputs, device, handler, *, mode: str,
                        dynamic) -> dict:
    """Benchmark one compile setting over the variable-shape stream.

    Two passes: pass 1 (warmup) absorbs first-time compiles and reveals how many
    recompiles the shape stream provokes; pass 2 (steady) is the reported
    throughput and should be recompile-free if the cache is stable.
    """
    net = copy.deepcopy(base_net).to(device).eval()
    torch._dynamo.reset()
    handler.reset()

    compiled = mode != "eager"
    if compiled:
        # Mirror production self_play: suppress dynamo capture errors → eager
        # fallback rather than a crash (perf feature, not correctness).
        torch._dynamo.config.suppress_errors = True
        net = torch.compile(net, dynamic=dynamic)

    warm_total, _ = _run_pass(net, inputs, device)
    warm_recompiles = handler.count

    handler.reset()
    steady_total, per_call = _run_pass(net, inputs, device)
    steady_recompiles = handler.count

    n = len(inputs)
    return {
        "mode": mode,
        "warm_total": warm_total,
        "warm_recompiles": warm_recompiles,
        "steady_total": steady_total,
        "steady_recompiles": steady_recompiles,
        "calls_per_sec": n / max(1e-9, steady_total),
        "median_call_ms": statistics.median(per_call) * 1e3,
        "p95_call_ms": (sorted(per_call)[int(0.95 * (n - 1))]) * 1e3,
    }


def _microbench(base_net, device: str, n_batches: int, seed: int) -> dict | None:
    """Section 1: variable-shape eager vs compile(dynamic=False/True)."""
    if not str(device).startswith("cuda"):
        print("\n[Section 1] SKIPPED — microbenchmark needs CUDA "
              "(torch.compile/Triton only helps on GPU).")
        return None

    batch_sizes = _sample_batch_sizes(n_batches, seed)
    uniq = sorted(set(batch_sizes))
    cache_limit = getattr(torch._dynamo.config, "cache_size_limit", None)

    print("\n" + "=" * 72)
    print("[Section 1] torch.compile under REALISTIC variable batch shapes")
    print("=" * 72)
    print(f"  stream: {n_batches} forwards, batch sizes drawn lognormal "
          f"(mean≈{statistics.mean(batch_sizes):.1f}, "
          f"min={min(batch_sizes)}, max={max(batch_sizes)}, "
          f"unique={len(uniq)})")
    print(f"  dynamo cache_size_limit = {cache_limit}  "
          f"(distinct static graphs before eviction/fallback)")
    print(f"  TORCH_LOGS={os.environ.get('TORCH_LOGS')!r} "
          f"(recompiles counted programmatically)")
    if not _triton_ok():
        print("  WARNING: Triton NOT importable — inductor will fall back to "
              "EAGER; expect ~0% speedup (and the verdict DO NOT USE).")

    handler = _install_recompile_logging()
    inputs = _make_inputs(batch_sizes, device)

    modes = [
        ("eager", None),
        ("compile(dynamic=False)", False),
        ("compile(dynamic=True)", True),
    ]
    results = []
    for name, dyn in modes:
        print(f"\n  running {name} ... (pass 1 warmup may compile per shape)",
              flush=True)
        try:
            r = _bench_compile_mode(base_net, inputs, device, handler,
                                    mode=name, dynamic=dyn)
        except Exception as e:  # one mode failing must not kill the bench
            print(f"    {name} FAILED: {type(e).__name__}: {e}")
            continue
        results.append(r)
        print(f"    warmup: {r['warm_total']:.2f}s, "
              f"{r['warm_recompiles']} recompiles | "
              f"steady: {r['steady_total']:.2f}s, "
              f"{r['steady_recompiles']} recompiles | "
              f"{r['calls_per_sec']:.0f} calls/s "
              f"(median {r['median_call_ms']:.2f}ms, p95 {r['p95_call_ms']:.2f}ms)")

    # table
    eager = next((r for r in results if r["mode"] == "eager"), None)
    print("\n  " + "-" * 70)
    print(f"  {'mode':<26}{'calls/s':>10}{'vs eager':>10}"
          f"{'warm recmp':>12}{'steady recmp':>14}")
    print("  " + "-" * 70)
    for r in results:
        spd = (r["calls_per_sec"] / eager["calls_per_sec"] - 1.0) * 100 if eager else 0.0
        print(f"  {r['mode']:<26}{r['calls_per_sec']:>10.0f}{spd:>+9.1f}%"
              f"{r['warm_recompiles']:>12}{r['steady_recompiles']:>14}")
    print("  " + "-" * 70)

    return {"results": results, "eager": eager, "unique_shapes": len(uniq),
            "cache_limit": cache_limit}


# ── Section 2: end-to-end self-play A/B (original benchmark) ─────────────────
def _bench_once(net, cfg, n_games, seed_start):
    _ex, _sc, stats = play_selfplay_games_batched(
        net, cfg, n_games=n_games, game_seed_start=seed_start)
    elapsed = max(1e-9, stats["elapsed"])
    return n_games / elapsed, stats


def _fmt_breakdown(stats) -> str:
    total = max(1e-9, stats["elapsed"])
    return (f"step={stats['step_sec']:.2f}s ({stats['step_sec']/total:5.1%})  "
            f"eval={stats['eval_sec']:.2f}s ({stats['eval_sec']/total:5.1%})  "
            f"update={stats['update_sec']:.2f}s ({stats['update_sec']/total:5.1%})")


def _endtoend(base_net, a) -> dict:
    print("\n" + "=" * 72)
    print(f"[Section 2] end-to-end self-play A/B  (channels={a.channels}, "
          f"blocks={a.blocks}, sims={a.sims}, games={a.games})")
    print("=" * 72)
    print("[1/3] baseline (no compile) ...", flush=True)
    gps_base, stats_base = _bench_once(copy.deepcopy(base_net), _cfg(a, False),
                                       a.games, seed_start=0)
    print("[2/3] compiled WARMUP (graph capture / per-shape recompiles) ...",
          flush=True)
    net_comp = copy.deepcopy(base_net)
    cfg_comp = _cfg(a, True)
    _bench_once(net_comp, cfg_comp, a.games, seed_start=0)   # warmup (discarded)
    print("[3/3] compiled MEASURED (steady state) ...", flush=True)
    gps_comp, stats_comp = _bench_once(net_comp, cfg_comp, a.games, seed_start=0)

    gps_impr = (gps_comp / gps_base - 1.0) * 100.0
    eval_impr = (1.0 - stats_comp["eval_sec"] / max(1e-9, stats_base["eval_sec"])) * 100.0
    print("\n" + "-" * 68)
    print(f"{'metric':<18}{'no compile':>16}{'compile':>16}{'change':>16}")
    print("-" * 68)
    print(f"{'games/sec':<18}{gps_base:>16.3f}{gps_comp:>16.3f}{gps_impr:>+15.1f}%")
    print(f"{'eval_sec':<18}{stats_base['eval_sec']:>16.2f}"
          f"{stats_comp['eval_sec']:>16.2f}{eval_impr:>+15.1f}%")
    print(f"{'elapsed_sec':<18}{stats_base['elapsed']:>16.2f}"
          f"{stats_comp['elapsed']:>16.2f}"
          f"{(1.0-stats_comp['elapsed']/max(1e-9,stats_base['elapsed']))*100:>+15.1f}%")
    print("-" * 68)
    print("tick timing breakdown:")
    print(f"  no compile : {_fmt_breakdown(stats_base)}")
    print(f"  compile    : {_fmt_breakdown(stats_comp)}")
    print(f"(compile {'FASTER' if gps_impr > 0 else 'SLOWER'} by "
          f"{abs(gps_impr):.1f}% games/sec)")
    return {"gps_impr": gps_impr}


def _cfg(a, compile_net: bool) -> SelfPlayConfig:
    return SelfPlayConfig(
        channels=a.channels, blocks=a.blocks,
        engine="batched_open_loop", device=a.device,
        n_simulations=a.sims, n_determinizations=1,
        batch_slots=a.batch_slots, leaf_batch=a.leaf_batch,
        profile_eval_timing=True,
        compile_net=compile_net,
        compile_dynamic={"auto": None, "on": True, "off": False}[a.compile_dynamic],
    )


def _triton_ok() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


# ── verdict ──────────────────────────────────────────────────────────────────
def _verdict(micro: dict | None, e2e: dict | None,
             speedup_thresh: float = 5.0) -> None:
    print("\n" + "#" * 72)
    print("# RECOMMENDATION")
    print("#" * 72)

    if not _triton_ok():
        print("Triton is not importable, so inductor cannot codegen GPU kernels "
              "and torch.compile\nfalls back to eager (no speedup).")
        print("\n>>> DO NOT USE --compile <<<")
        return

    reasons: list[str] = []
    use = False
    best_name = None

    if micro and micro["eager"]:
        eager = micro["eager"]
        # A compile mode is viable iff faster than eager by the threshold AND
        # recompile-stable in steady state (the storm has to be absent).
        candidates = []
        for r in micro["results"]:
            if r["mode"] == "eager":
                continue
            spd = (r["calls_per_sec"] / eager["calls_per_sec"] - 1.0) * 100
            stable = r["steady_recompiles"] <= 2
            candidates.append((spd, stable, r))
        candidates.sort(key=lambda c: c[0], reverse=True)
        viable = [c for c in candidates if c[0] >= speedup_thresh and c[1]]
        if viable:
            spd, _stable, r = viable[0]
            use = True
            best_name = r["mode"]
            reasons.append(f"{r['mode']} is {spd:+.1f}% faster than eager with "
                           f"only {r['steady_recompiles']} steady-state recompiles.")
        else:
            # explain why nothing qualified
            if candidates:
                spd, stable, r = candidates[0]
                if spd < speedup_thresh:
                    reasons.append(f"best compile mode ({r['mode']}) is only "
                                   f"{spd:+.1f}% vs eager (< {speedup_thresh:.0f}% "
                                   f"threshold) — kernel-fusion win is not real here.")
                if not stable:
                    reasons.append(f"{r['mode']} still recompiles "
                                   f"{r['steady_recompiles']}× in steady state — "
                                   f"a recompilation storm under variable shapes.")
        # flag dynamic=False storm explicitly if relevant
        df = next((r for r in micro["results"]
                   if r["mode"] == "compile(dynamic=False)"), None)
        if df and micro["unique_shapes"] > (micro["cache_limit"] or 8):
            reasons.append(
                f"dynamic=False sees {micro['unique_shapes']} unique shapes vs "
                f"cache_size_limit={micro['cache_limit']} -> guaranteed eviction "
                f"thrash; if compiling, use dynamic=True.")

    if e2e is not None:
        if e2e["gps_impr"] >= speedup_thresh:
            reasons.append(f"end-to-end self-play is {e2e['gps_impr']:+.1f}% games/s.")
            use = use or (micro is None)  # e2e alone can justify if no micro
        else:
            reasons.append(f"end-to-end self-play is only {e2e['gps_impr']:+.1f}% "
                           f"games/s (compile gain doesn't survive the full tick "
                           f"loop, where eval is not the sole bottleneck).")
            if micro is None:
                use = False

    for r in reasons:
        print(f"  - {r}")
    if use and best_name and "dynamic=True" in best_name:
        print("  (note: self_play calls torch.compile with default dynamic; the "
              "win\n   came from dynamic=True - consider exposing/forcing it.)")

    print("\n>>> " + ("USE --compile" if use else "DO NOT USE --compile") + " <<<")


def main() -> None:
    ap = argparse.ArgumentParser(description="Item 20 torch.compile A/B benchmark")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--batch_slots", type=int, default=32)
    ap.add_argument("--leaf_batch", type=int, default=6)
    ap.add_argument("--compile_dynamic", choices=["auto", "on", "off"],
                    default="auto",
                    help="dynamic= for the Section 2 end-to-end compiled run "
                         "(Section 1 always tests auto/on/off itself)")
    ap.add_argument("--micro_batches", type=int, default=96,
                    help="number of variable-shape forwards in Section 1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_micro", action="store_true")
    ap.add_argument("--skip_endtoend", action="store_true")
    a = ap.parse_args()

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")

    torch.manual_seed(0)
    base_net = KingdominoNet(channels=a.channels, blocks=a.blocks).to(a.device).eval()

    print(f"\n=== Item 20: torch.compile A/B  (channels={a.channels}, "
          f"blocks={a.blocks}, device={a.device}) ===")

    micro = None if a.skip_micro else _microbench(base_net, a.device,
                                                  a.micro_batches, a.seed)
    e2e = None if a.skip_endtoend else _endtoend(base_net, a)
    _verdict(micro, e2e)


if __name__ == "__main__":
    main()
