"""
bench_doublebuffer.py — Item 17 A/B benchmark: double-buffered self-play.

Runs play_selfplay_games_batched WITHOUT and WITH double_buffer over the same
n_games / config and prints a before/after table: games/s, the step/eval/update
tick-timing split, and the improvement %.

In the double-buffer path step_sec + eval_sec + update_sec OVERLAP (they run on
two threads), so their sum can exceed elapsed — that overlap is the speedup, and
the table shows each as a % of (step+eval+update) so the shift is visible.

Run:
  python -m games.kingdomino.bench_doublebuffer --device cuda --sims 200 --games 50 --channels 32 --blocks 4
"""
from __future__ import annotations

import argparse
import copy

import torch

from games.kingdomino.network import KingdominoNet
from games.kingdomino.self_play import SelfPlayConfig, play_selfplay_games_batched


def _bench_once(net, cfg, n_games, double_buffer):
    examples, _scores, stats = play_selfplay_games_batched(
        net, cfg, n_games=n_games, game_seed_start=0, double_buffer=double_buffer)
    elapsed = max(1e-9, stats["elapsed"])
    n_done = len(examples)
    return n_done / elapsed, n_done, stats


def _split(stats) -> str:
    s, e, u = stats["step_sec"], stats["eval_sec"], stats["update_sec"]
    tot = max(1e-9, s + e + u)
    return (f"step={s:.2f}s ({s/tot:5.1%})  eval={e:.2f}s ({e/tot:5.1%})  "
            f"update={u:.2f}s ({u/tot:5.1%})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Item 17 double-buffer A/B benchmark")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--batch_slots", type=int, default=32)
    ap.add_argument("--leaf_batch", type=int, default=6)
    a = ap.parse_args()

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")

    torch.manual_seed(0)
    base_net = KingdominoNet(channels=a.channels, blocks=a.blocks).to(a.device).eval()

    cfg = SelfPlayConfig(
        channels=a.channels, blocks=a.blocks,
        engine="batched_open_loop", device=a.device,
        n_simulations=a.sims, n_determinizations=1,
        batch_slots=a.batch_slots, leaf_batch=a.leaf_batch,
        profile_eval_timing=False,
    )

    print(f"\n=== Item 17: double-buffer A/B  (channels={a.channels}, "
          f"blocks={a.blocks}, sims={a.sims}, games={a.games}, "
          f"device={a.device}) ===")
    print("\n[1/2] single-buffer ...", flush=True)
    gps_s, done_s, stats_s = _bench_once(copy.deepcopy(base_net), cfg, a.games, False)
    print("[2/2] double-buffer ...", flush=True)
    gps_d, done_d, stats_d = _bench_once(copy.deepcopy(base_net), cfg, a.games, True)

    impr = (gps_d / gps_s - 1.0) * 100.0
    print("\n" + "=" * 70)
    print(f"{'metric':<20}{'single':>16}{'double':>16}{'change':>16}")
    print("-" * 70)
    print(f"{'games/sec':<20}{gps_s:>16.3f}{gps_d:>16.3f}{impr:>+15.1f}%")
    print(f"{'elapsed_sec':<20}{stats_s['elapsed']:>16.2f}{stats_d['elapsed']:>16.2f}"
          f"{(1.0-stats_d['elapsed']/max(1e-9,stats_s['elapsed']))*100:>+15.1f}%")
    print(f"{'games completed':<20}{done_s:>16d}{done_d:>16d}"
          f"{'  MATCH' if done_s == done_d else '  MISMATCH!':>16}")
    print(f"{'mean_batch':<20}{stats_s['mean_batch']:>16.1f}{stats_d['mean_batch']:>16.1f}"
          f"{'':>16}")
    print("=" * 70)
    print("tick timing breakdown (step/eval/update; double-buffer sums OVERLAP):")
    print(f"  single : {_split(stats_s)}")
    print(f"  double : {_split(stats_d)}")
    print(f"\n(double-buffer {'FASTER' if impr > 0 else 'SLOWER'} by {abs(impr):.1f}% "
          f"games/sec; game count {'matches' if done_s == done_d else 'MISMATCH'})")

    # ── Recommendation ──
    # Double-buffer overlaps CPU tree-work with GPU eval on two threads, so it
    # only helps when the two phases are comparably costly (a fast GPU that has
    # left the CPU as a co-bottleneck).  On the RTX 3070 it was -8.5% because the
    # GPU was the sole bottleneck — overlap bought nothing and added thread
    # overhead.  A game-count MISMATCH means the two paths didn't run the same
    # work, so the % is not comparable and we must not recommend turning it on.
    print("\n" + "#" * 70)
    print("# RECOMMENDATION")
    print("#" * 70)
    threshold = 3.0  # ignore sub-noise wins
    if done_s != done_d:
        print(f"  - game count MISMATCH ({done_s} vs {done_d}): results are not "
              f"comparable; rerun before trusting the delta.")
        print("\n>>> DO NOT USE --double_buffer <<<")
    elif impr >= threshold:
        print(f"  - double-buffer is {impr:+.1f}% games/s (> {threshold:.0f}% "
              f"threshold): the GPU now finishes eval fast enough that overlapping "
              f"CPU tree-work pays off.")
        print("\n>>> USE --double_buffer <<<")
    else:
        print(f"  - double-buffer is only {impr:+.1f}% games/s (< {threshold:.0f}% "
              f"threshold): GPU eval is still the bottleneck, so the overlap buys "
              f"little and adds thread overhead.")
        print("\n>>> DO NOT USE --double_buffer <<<")


if __name__ == "__main__":
    main()
