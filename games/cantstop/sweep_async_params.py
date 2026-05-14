# sweep_async_params.py
#
# Sweep test for the (warmup_sims, target_inflight) hyperparameters of
# the async MCTS scheduler.
#
# Why we need this: the simple async design (warmup=0, inflight=64) was
# 2.86x faster than sync but produced training data so bad that the
# resulting model won 10% vs the baseline (sync produced 48%). Warmup
# fixes the training-signal issue by guaranteeing early sims see real
# tree statistics, but introduces a speed/quality tradeoff. This script
# finds the sweet spot empirically.
#
# Each "cell" in the sweep runs ONE self_play iteration:
#   - generate N games with MCTS at chosen (warmup, inflight)
#   - train for K epochs on the resulting data
#   - eval against the baseline model
# We then collect the metrics that matter:
#   - games/s during generation (speed)
#   - val_loss + policy_match (training signal quality)
#   - win_rate (the final quality verdict)
#   - avg entropy (decisiveness indicator)
#
# Strategy is two-phased:
#   Phase 1: pin inflight=16, sweep warmup ∈ {0,4,8,16,32}
#            → pick smallest warmup where quality matches sync
#   Phase 2: pin warmup=winner, sweep inflight ∈ {1,4,8,16,32,64}
#            → pick largest inflight where quality stays good
#
# Usage:
#   python games\cantstop\sweep_async_params.py ^
#     --model models\cantstop\self_play\model_iter_003_accepted.pt ^
#     --games 200 --sims 200 --eval 50 --eval-sims 20 --epochs 3
#
# Output: sweep_results.csv with one row per cell, columns:
#   warmup, inflight, gen_time_s, games_per_s, val_loss, policy_match,
#   entropy, win_rate

import os
import sys
import csv
import time
import argparse
import subprocess
import re
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# Patterns to extract metrics from self_play.py log output.
RE_GEN = re.compile(
    r"Generated\s+([\d,]+)\s+records\s+in\s+([\d.]+)s\s+\(([\d.]+)\s+games/s\)"
)
RE_EPOCH = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|\s+Train:\s+([\d.]+)\s+\|\s+Val:\s+([\d.]+)\s+"
    r"\|\s+Policy match \(vs MCTS argmax\):\s+([\d.]+)\s+\|\s+Entropy:\s+([\d.]+)"
)
RE_WIN_RATE = re.compile(
    r"New model:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)"
)


def run_cell(args, warmup, inflight, log_dir):
    """
    Run one sweep cell — a single self_play_loop iteration — and
    return its metrics.

    Returns dict with keys:
        warmup, inflight, gen_time_s, games_per_s,
        val_loss, policy_match, entropy, win_rate

    Returns None if the run failed.
    """
    log_path = os.path.join(
        log_dir, f"w{warmup:02d}_i{inflight:02d}.log"
    )

    # Use a private output dir per cell so checkpoints/tmp files don't
    # collide if multiple cells write best_model.pt. We don't actually
    # care about the saved model — the metrics come from stdout.
    cell_out = os.path.join(log_dir, f"out_w{warmup:02d}_i{inflight:02d}")
    os.makedirs(cell_out, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "self_play.py"),
        "--model", args.model,
        "--output", cell_out,
        "--iterations", "1",
        "--games", str(args.games),
        "--sims", str(args.sims),
        "--epochs", str(args.epochs),
        "--eval", str(args.eval),
        "--eval-sims", str(args.eval_sims),
        "--floor", "0.0",          # never accept; we only want metrics
        "--inflight", str(inflight),
        "--warmup", str(warmup),
    ]

    print(f"\n{'='*60}")
    print(f"  CELL: warmup={warmup}, inflight={inflight}")
    print(f"  Log:  {log_path}")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, text=True
        )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  ⚠ Run failed (exit {proc.returncode}); see {log_path}")
        return None

    # Parse the log for metrics.
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    gen_match = RE_GEN.search(text)
    if not gen_match:
        print("  ⚠ Couldn't find 'Generated ... games/s' line")
        return None
    gen_time_s = float(gen_match.group(2))
    games_per_s = float(gen_match.group(3))

    # Take the LAST epoch's metrics (final training state).
    epoch_matches = RE_EPOCH.findall(text)
    if not epoch_matches:
        # epochs=0 case — no training metrics. Still record gen speed.
        val_loss = float('nan')
        policy_match = float('nan')
        entropy = float('nan')
    else:
        last = epoch_matches[-1]
        val_loss = float(last[3])
        policy_match = float(last[4])
        entropy = float(last[5])

    win_match = RE_WIN_RATE.search(text)
    if not win_match:
        win_rate = float('nan')
    else:
        win_rate = float(win_match.group(3)) / 100.0

    print(f"  ✓ Done in {elapsed:.1f}s | games/s={games_per_s:.2f} | "
          f"val_loss={val_loss:.4f} | policy_match={policy_match:.3f} | "
          f"win_rate={win_rate:.1%}")

    # Clean up per-cell output dir to save disk space.
    try:
        shutil.rmtree(cell_out)
    except Exception:
        pass

    return {
        "warmup":       warmup,
        "inflight":     inflight,
        "gen_time_s":   gen_time_s,
        "games_per_s":  games_per_s,
        "val_loss":     val_loss,
        "policy_match": policy_match,
        "entropy":      entropy,
        "win_rate":     win_rate,
    }


def write_csv(results, path):
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def print_table(results, title=""):
    if not results:
        print("  (no results)")
        return
    print(f"\n{title}")
    print(f"  {'warmup':>6} {'inflight':>8} {'games/s':>8} "
          f"{'val_loss':>9} {'policy_match':>13} {'entropy':>8} "
          f"{'win_rate':>9}")
    print(f"  {'-'*6:>6} {'-'*8:>8} {'-'*8:>8} {'-'*9:>9} "
          f"{'-'*13:>13} {'-'*8:>8} {'-'*9:>9}")
    for r in results:
        print(f"  {r['warmup']:>6d} {r['inflight']:>8d} "
              f"{r['games_per_s']:>8.2f} {r['val_loss']:>9.4f} "
              f"{r['policy_match']:>13.3f} {r['entropy']:>8.3f} "
              f"{r['win_rate']:>8.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--games",     type=int, default=200)
    parser.add_argument("--sims",      type=int, default=200)
    parser.add_argument("--epochs",    type=int, default=3)
    parser.add_argument("--eval",      type=int, default=50)
    parser.add_argument("--eval-sims", type=int, default=20, dest="eval_sims")
    parser.add_argument("--log-dir",   default="sweep_logs", dest="log_dir")
    parser.add_argument("--csv",       default="sweep_results.csv")
    parser.add_argument(
        "--phase1-warmups", type=str, default="0,4,8,16,32",
        help="Comma-separated warmup values for phase 1 (inflight pinned)."
    )
    parser.add_argument(
        "--phase1-inflight", type=int, default=16,
        help="Inflight value pinned during phase 1."
    )
    parser.add_argument(
        "--phase2-inflights", type=str, default="1,4,8,16,32,64",
        help="Comma-separated inflight values for phase 2 (warmup pinned)."
    )
    parser.add_argument(
        "--phase2-warmup", type=int, default=None,
        help="Warmup value pinned during phase 2. If omitted, picks the "
             "winner from phase 1 (smallest warmup whose val_loss is "
             "within 0.02 of the best)."
    )
    parser.add_argument(
        "--skip-phase2", action="store_true",
        help="Run only phase 1."
    )
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    all_results = []

    # ---- Phase 1 ----
    print(f"\n{'#'*60}")
    print(f"  PHASE 1: sweep warmup, inflight pinned at "
          f"{args.phase1_inflight}")
    print(f"{'#'*60}")
    phase1_warmups = [int(x) for x in args.phase1_warmups.split(",")]
    phase1 = []
    for w in phase1_warmups:
        r = run_cell(args, w, args.phase1_inflight, args.log_dir)
        if r is not None:
            phase1.append(r)
            all_results.append(r)
            # Write CSV after every cell so a crash doesn't lose data.
            write_csv(all_results, args.csv)

    print_table(phase1, "PHASE 1 RESULTS:")

    if args.skip_phase2:
        print("\nSkipping phase 2 (--skip-phase2).")
        return

    # Pick phase 2 warmup: smallest warmup whose val_loss is within
    # 0.02 of the best val_loss in phase 1. This rewards smaller
    # warmup (faster) when quality is comparable.
    if args.phase2_warmup is not None:
        phase2_warmup = args.phase2_warmup
        print(f"\nPhase 2 warmup fixed to {phase2_warmup} (--phase2-warmup).")
    elif phase1:
        best_vl = min(r["val_loss"] for r in phase1
                      if not (r["val_loss"] != r["val_loss"]))  # filter NaN
        # Tolerate quality within 0.02 val_loss — empirical "close enough"
        candidates = [r for r in phase1
                      if r["val_loss"] <= best_vl + 0.02]
        # Smallest warmup among candidates
        phase2_warmup = min(c["warmup"] for c in candidates)
        print(f"\nPhase 2 warmup = {phase2_warmup} "
              f"(smallest warmup within 0.02 val_loss of best={best_vl:.4f})")
    else:
        phase2_warmup = 16
        print(f"\nPhase 1 produced no results; phase 2 warmup defaults to 16.")

    # ---- Phase 2 ----
    print(f"\n{'#'*60}")
    print(f"  PHASE 2: sweep inflight, warmup pinned at {phase2_warmup}")
    print(f"{'#'*60}")
    phase2_inflights = [int(x) for x in args.phase2_inflights.split(",")]
    phase2 = []
    for i in phase2_inflights:
        # Skip duplicates with phase 1 (same warmup, same inflight)
        if any(r["warmup"] == phase2_warmup and r["inflight"] == i
               for r in phase1):
            print(f"  (skipping warmup={phase2_warmup} inflight={i} "
                  f"— already run in phase 1)")
            continue
        r = run_cell(args, phase2_warmup, i, args.log_dir)
        if r is not None:
            phase2.append(r)
            all_results.append(r)
            write_csv(all_results, args.csv)

    # Combine phase 1 cells with matching warmup into phase 2 view
    matching_phase1 = [r for r in phase1 if r["warmup"] == phase2_warmup]
    phase2_full = sorted(
        phase2 + matching_phase1, key=lambda r: r["inflight"]
    )
    print_table(phase2_full, f"PHASE 2 RESULTS (warmup={phase2_warmup}):")

    # ---- Summary ----
    print(f"\n{'#'*60}")
    print(f"  FULL RESULTS")
    print(f"{'#'*60}")
    print_table(
        sorted(all_results, key=lambda r: (r["warmup"], r["inflight"])),
        ""
    )

    # Recommend a winner.
    print(f"\n{'#'*60}")
    print(f"  RECOMMENDATION")
    print(f"{'#'*60}")
    # Find sync baseline (warmup=0, inflight=1) if present, else just
    # use the best val_loss across all cells.
    sync_baseline = next(
        (r for r in all_results
         if r["warmup"] == 0 and r["inflight"] == 1),
        None
    )
    if sync_baseline is None:
        sync_baseline = min(all_results, key=lambda r: r["val_loss"])
        print(f"  No (warmup=0, inflight=1) cell found; using best "
              f"val_loss cell as the quality reference.")
    print(f"  Quality reference: warmup={sync_baseline['warmup']}, "
          f"inflight={sync_baseline['inflight']}, "
          f"val_loss={sync_baseline['val_loss']:.4f}, "
          f"games/s={sync_baseline['games_per_s']:.2f}")
    # Best speed within val_loss tolerance
    qualified = [r for r in all_results
                 if r["val_loss"] <= sync_baseline["val_loss"] + 0.02]
    if qualified:
        winner = max(qualified, key=lambda r: r["games_per_s"])
        speedup = winner["games_per_s"] / max(
            sync_baseline["games_per_s"], 1e-9
        )
        print(f"  Best speed within 0.02 val_loss: "
              f"warmup={winner['warmup']}, "
              f"inflight={winner['inflight']}, "
              f"games/s={winner['games_per_s']:.2f} "
              f"({speedup:.2f}× over reference), "
              f"val_loss={winner['val_loss']:.4f}, "
              f"win_rate={winner['win_rate']:.1%}")
    else:
        print(f"  No cell met the val_loss tolerance.")

    print(f"\nResults CSV: {args.csv}")
    print(f"Per-cell logs: {args.log_dir}")


if __name__ == "__main__":
    main()