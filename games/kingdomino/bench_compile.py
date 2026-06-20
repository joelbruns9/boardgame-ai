"""
bench_compile.py — Item 20 A/B benchmark: torch.compile on the inference net.

Runs ``play_selfplay_games_batched`` (the batched_open_loop self-play tick loop)
WITHOUT and WITH ``torch.compile`` on the leaf-eval net, and prints a before/
after table plus the tick-timing breakdown (step / eval / update as % of total).

The compiled run is executed TWICE: the first run pays the one-time graph
capture / autotune cost (and any per-shape recompiles), so only the SECOND run
is reported as the steady-state number.

Run:
  python -m games.kingdomino.bench_compile --device cuda --sims 200 --games 20
"""
from __future__ import annotations

import argparse
import copy

import torch

from games.kingdomino.network import KingdominoNet
from games.kingdomino.self_play import SelfPlayConfig, play_selfplay_games_batched


def _bench_once(net, cfg, n_games, seed_start):
    """One play_selfplay_games_batched run; return (games_per_sec, stats)."""
    _ex, _sc, stats = play_selfplay_games_batched(
        net, cfg, n_games=n_games, game_seed_start=seed_start)
    elapsed = max(1e-9, stats["elapsed"])
    return n_games / elapsed, stats


def _fmt_breakdown(stats) -> str:
    total = max(1e-9, stats["elapsed"])
    return (f"step={stats['step_sec']:.2f}s ({stats['step_sec']/total:5.1%})  "
            f"eval={stats['eval_sec']:.2f}s ({stats['eval_sec']/total:5.1%})  "
            f"update={stats['update_sec']:.2f}s ({stats['update_sec']/total:5.1%})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Item 20 torch.compile A/B benchmark")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--batch_slots", type=int, default=32)
    ap.add_argument("--leaf_batch", type=int, default=6)
    a = ap.parse_args()

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")

    torch.manual_seed(0)
    base_net = KingdominoNet(channels=a.channels, blocks=a.blocks).to(a.device).eval()

    def _cfg(compile_net: bool) -> SelfPlayConfig:
        return SelfPlayConfig(
            channels=a.channels, blocks=a.blocks,
            engine="batched_open_loop", device=a.device,
            n_simulations=a.sims, n_determinizations=1,
            batch_slots=a.batch_slots, leaf_batch=a.leaf_batch,
            profile_eval_timing=True,   # populate eval_*_sec timing
            compile_net=compile_net,
        )

    # ── Baseline (no compile) ──
    print(f"\n=== Item 20: torch.compile A/B  (channels={a.channels}, "
          f"blocks={a.blocks}, sims={a.sims}, games={a.games}, device={a.device}) ===")
    print("\n[1/3] baseline (no compile) ...", flush=True)
    net_base = copy.deepcopy(base_net)
    gps_base, stats_base = _bench_once(net_base, _cfg(False), a.games, seed_start=0)

    # ── Compiled — warmup run (pays the capture cost), then measured run ──
    print("[2/3] compiled WARMUP (graph capture / per-shape recompiles) ...",
          flush=True)
    net_comp = copy.deepcopy(base_net)
    cfg_comp = _cfg(True)
    _bench_once(net_comp, cfg_comp, a.games, seed_start=0)   # warmup (discarded)
    print("[3/3] compiled MEASURED (steady state) ...", flush=True)
    gps_comp, stats_comp = _bench_once(net_comp, cfg_comp, a.games, seed_start=0)

    # ── Report ──
    gps_impr = (gps_comp / gps_base - 1.0) * 100.0
    eval_impr = (1.0 - stats_comp["eval_sec"] / max(1e-9, stats_base["eval_sec"])) * 100.0
    print("\n" + "=" * 68)
    print(f"{'metric':<18}{'no compile':>16}{'compile':>16}{'change':>16}")
    print("-" * 68)
    print(f"{'games/sec':<18}{gps_base:>16.3f}{gps_comp:>16.3f}{gps_impr:>+15.1f}%")
    print(f"{'eval_sec':<18}{stats_base['eval_sec']:>16.2f}"
          f"{stats_comp['eval_sec']:>16.2f}{eval_impr:>+15.1f}%")
    print(f"{'elapsed_sec':<18}{stats_base['elapsed']:>16.2f}"
          f"{stats_comp['elapsed']:>16.2f}"
          f"{(1.0-stats_comp['elapsed']/max(1e-9,stats_base['elapsed']))*100:>+15.1f}%")
    print("=" * 68)
    print("tick timing breakdown:")
    print(f"  no compile : {_fmt_breakdown(stats_base)}")
    print(f"  compile    : {_fmt_breakdown(stats_comp)}")
    print(f"\n(compile {'FASTER' if gps_impr > 0 else 'SLOWER'} by "
          f"{abs(gps_impr):.1f}% games/sec)")


if __name__ == "__main__":
    main()
