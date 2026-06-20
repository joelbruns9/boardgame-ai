"""
round_robin_eval.py
Round-robin tournament between multiple model checkpoints.

Purpose
-------
Identify which checkpoint is genuinely strongest. Per-iteration evals in
self_play_loop only compare against the immediately-previous accepted
model, which is structurally unable to detect chain drift or rock-paper-
scissors cycles between checkpoints. A round-robin pits every model
against every other model at higher sim counts than per-iteration eval,
producing a real ordering.

How it works
------------
Given N model paths, it plays every pair (N*(N-1)/2 pairs) in a worker
pool. Each pair plays `games_per_pair` games at `eval_sims` simulations
per move, with players alternating P0/P1 to remove first-mover bias.

Output
------
1. A win-rate matrix (rows = model, columns = opponent), printed as a
   table.
2. Each model's overall win rate (averaged across all opponents).
3. A ranked list, strongest first.
4. Statistical confidence flags on pair results (was the win rate gap
   plausibly real, or within noise?).
5. A CSV file with raw pair-by-pair results for later analysis.

Usage
-----
From the repo root:

    python -m games.cantstop.scripts.round_robin_eval \\
        --models models/cantstop/self_play/model_iter_001_accepted.pt \\
                 models/cantstop/self_play/model_iter_005_accepted.pt \\
                 models/cantstop/self_play/model_iter_012_accepted.pt \\
                 models/cantstop/self_play/best_model.pt \\
        --games-per-pair 400 \\
        --eval-sims 100 \\
        --workers 8 \\
        --output tournament_results.csv

Adjust paths and parameters as needed. With 6 models, 400 games/pair,
100 sims, 8 workers, expect roughly 1-2 hours of total runtime. Scale
roughly linearly with eval_sims and games_per_pair.

Calibration notes
-----------------
- games_per_pair=400: standard error ~2.5%% on a 50%% comparison from
  the formula sqrt(0.25/N), so ±5%% margin at 95%% CI. Greedy play
  introduces correlation across games (identical model responses to
  repeated positions), so effective sample size is somewhat lower —
  treat margins as roughly ±5-7%%. Bump to 800 for tighter resolution
  of matchups expected to be close (e.g. iter N vs iter N+1).
- eval_sims=100 (default): 5x the per-iteration default. Discriminates
  better between models with subtle policy differences. Raise to 300+
  for tournaments where deployment-style strength is the question.
- temperature=0 (greedy from MCTS visit counts). This is set inside
  _play_mcts_eval_game already.
"""

import os
import sys
import time
import math
import csv
import argparse
import itertools
import random
import multiprocessing as mp

import numpy as np
import torch

# Path: this file is at games/cantstop/scripts/, so three '..'s reach the
# project root where the `games` package lives.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from games.cantstop.evaluate import load_model
from games.cantstop.mcts import MCTS
from games.cantstop.self_play import _play_mcts_eval_game


# ============================================================
# Worker-process state and entry points
# ============================================================
# Workers are processes (multiprocessing.Pool). Each process keeps a
# per-process cache of (model_path -> MCTS instance) so we don't reload
# every game. Models are loaded lazily on first request from disk.

_worker_mcts_cache = None  # dict: model_path -> MCTS instance


def _init_worker():
    """Initializer for each worker process."""
    global _worker_mcts_cache
    _worker_mcts_cache = {}


def _get_mcts(model_path):
    """Return an MCTS instance for `model_path`, caching per-process."""
    global _worker_mcts_cache
    if model_path not in _worker_mcts_cache:
        model = load_model(model_path, 'cpu')
        _worker_mcts_cache[model_path] = MCTS(
            model,
            'cpu',
            target_inflight=1,
            warmup_sims=0,
        )
    return _worker_mcts_cache[model_path]


def _play_pair_chunk(args):
    """
    args: (model_a_path, model_b_path, start_idx, n_games, num_simulations, seed)
    """
    model_a_path, model_b_path, start_idx, n_games, num_simulations, seed = args
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    mcts_a = _get_mcts(model_a_path)
    mcts_b = _get_mcts(model_b_path)

    wins_a = wins_b = draws = 0
    for local_i in range(n_games):
        game_idx = start_idx + local_i
        try:
            if game_idx % 2 == 0:
                winner = _play_mcts_eval_game(mcts_a, mcts_b, num_simulations)
                if winner == 0:   wins_a += 1
                elif winner == 1: wins_b += 1
                else:             draws += 1
            else:
                winner = _play_mcts_eval_game(mcts_b, mcts_a, num_simulations)
                if winner == 0:   wins_b += 1
                elif winner == 1: wins_a += 1
                else:             draws += 1
        except Exception as e:
            import traceback
            print(f"  [worker] game {game_idx} failed: {e}", flush=True)
            traceback.print_exc()
            continue

    return wins_a, wins_b, draws


# ============================================================
# Tournament orchestration
# ============================================================

def run_pair(pool, model_a_path, model_b_path, games_per_pair, eval_sims,
             num_workers, base_seed):
    """
    Run all games for a single pair (A, B) across the worker pool.
    """
    # Split games_per_pair across workers. P0/P1 alternation is based on
    # global game index, so chunk sizes do not need to be even.
    per_worker = games_per_pair // num_workers
    leftover = games_per_pair % num_workers

    seed_rng = random.Random(base_seed)
    chunk_args = []
    start_idx = 0
    for w in range(num_workers):
        n = per_worker + (1 if w < leftover else 0)
        if n <= 0:
            continue
        chunk_seed = seed_rng.randint(0, 2**31 - 1)
        chunk_args.append((
            model_a_path,
            model_b_path,
            start_idx,
            n,
            eval_sims,
            chunk_seed,
        ))
        start_idx += n

    t0 = time.time()
    results = pool.map(_play_pair_chunk, chunk_args)
    elapsed = time.time() - t0

    wins_a = sum(r[0] for r in results)
    wins_b = sum(r[1] for r in results)
    draws  = sum(r[2] for r in results)
    return wins_a, wins_b, draws, elapsed


def wilson_ci(wins, total, z=1.96):
    """
    Wilson 95% confidence interval for win probability.

    More appropriate than normal-approximation when win counts are near
    0 or near total (Wilson handles boundary cases gracefully).
    Returns (low, high). z=1.96 ~ 95% two-sided.
    """
    if total == 0:
        return 0.0, 1.0
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def format_label(model_path):
    """Compact display label: filename without extension."""
    return os.path.splitext(os.path.basename(model_path))[0]


# ============================================================
# Result presentation
# ============================================================

def print_pair_matrix(model_paths, pair_results):
    """
    Print the win-rate matrix. Cell [i][j] = win rate of model i against
    model j. Diagonal is dashes. Matrix is symmetric in info but cells
    show winrate-of-row-vs-column for readability.
    """
    labels = [format_label(p) for p in model_paths]
    n = len(labels)

    # Build dense matrix from sparse pair_results dict.
    # pair_results: {(i,j): (wins_i, wins_j, draws)} with i<j.
    wr = np.full((n, n), np.nan)
    for (i, j), (wi, wj, dr) in pair_results.items():
        total = wi + wj
        if total == 0:
            continue
        wr[i, j] = wi / total
        wr[j, i] = wj / total

    # Column width sized to label or "100.0%"
    col_w = max(max(len(lab) for lab in labels), 7)

    # Header
    print()
    print("  Win rate matrix (row vs column, draws excluded)")
    print("  " + " " * col_w + " | " + " | ".join(lab.center(col_w) for lab in labels))
    print("  " + "-" * col_w + "-+-" + "-+-".join("-" * col_w for _ in labels))

    for i in range(n):
        cells = []
        for j in range(n):
            if i == j:
                cells.append("---".center(col_w))
            elif np.isnan(wr[i, j]):
                cells.append("n/a".center(col_w))
            else:
                cells.append(f"{wr[i, j]*100:5.1f}%".center(col_w))
        print(f"  {labels[i]:<{col_w}} | " + " | ".join(cells))


def compute_overall_rankings(model_paths, pair_results):
    """
    Compute each model's overall win rate (averaged over all opponents).
    Returns list of (model_path, label, overall_wr, total_wins, total_games)
    sorted by overall_wr descending.
    """
    n = len(model_paths)
    totals = [[0, 0] for _ in range(n)]  # [wins, games] per model

    for (i, j), (wi, wj, dr) in pair_results.items():
        # Draws are excluded from the win-rate denominator (consistent
        # with self_play_loop's evaluate_networks convention).
        played = wi + wj
        totals[i][0] += wi
        totals[i][1] += played
        totals[j][0] += wj
        totals[j][1] += played

    rows = []
    for idx, path in enumerate(model_paths):
        wins, games = totals[idx]
        wr = wins / games if games > 0 else 0.0
        rows.append((path, format_label(path), wr, wins, games))

    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


def print_rankings(rankings):
    """Print the ranked list."""
    print()
    print("  Overall ranking (decisive games, all opponents combined)")
    print("  " + "-" * 60)
    print(f"  {'Rank':<5} {'Model':<35} {'Win rate':>10} {'Games':>8}")
    print("  " + "-" * 60)
    for rank, (path, label, wr, wins, games) in enumerate(rankings, start=1):
        marker = "★ " if rank == 1 else "  "
        print(f"  {marker}{rank:<3} {label:<35} {wr*100:>9.1f}% {games:>8d}")
    print()


def print_pair_significance(model_paths, pair_results):
    """
    For each pair, print Wilson 95% CI on the win rate and a flag for
    whether the result is significantly different from 50%.
    """
    labels = [format_label(p) for p in model_paths]
    print()
    print("  Pair-by-pair detail with 95% confidence intervals")
    print("  " + "-" * 78)
    print(f"  {'Pair':<55} {'Result':>10} {'95% CI':>16}")
    print("  " + "-" * 78)

    pairs_sorted = sorted(pair_results.keys())
    for (i, j) in pairs_sorted:
        wi, wj, dr = pair_results[(i, j)]
        total = wi + wj
        if total == 0:
            continue
        wr = wi / total
        lo, hi = wilson_ci(wi, total)
        # Flag significance: does the CI cross 50%?
        if lo > 0.5:
            flag = "** wins"
        elif hi < 0.5:
            flag = "** loses"
        else:
            flag = "  noise"
        pair_label = f"{labels[i]:>26}  vs  {labels[j]:<26}"
        ci_str = f"[{lo*100:5.1f}, {hi*100:5.1f}]"
        print(f"  {pair_label} {wr*100:>9.1f}% {ci_str:>16}  {flag}")
    print()
    print("  ** = statistically significant (CI does not cross 50%)")
    print()


def write_csv(model_paths, pair_results, output_path):
    """Persist raw pair results so they can be re-analyzed later."""
    labels = [format_label(p) for p in model_paths]
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_a', 'model_a_path',
                         'model_b', 'model_b_path',
                         'wins_a', 'wins_b', 'draws',
                         'games_decisive', 'wr_a',
                         'wilson_lo', 'wilson_hi'])
        for (i, j) in sorted(pair_results.keys()):
            wi, wj, dr = pair_results[(i, j)]
            total = wi + wj
            wr = wi / total if total > 0 else 0.0
            lo, hi = wilson_ci(wi, total) if total > 0 else (0.0, 1.0)
            writer.writerow([labels[i], model_paths[i],
                             labels[j], model_paths[j],
                             wi, wj, dr,
                             total, f"{wr:.4f}",
                             f"{lo:.4f}", f"{hi:.4f}"])
    print(f"  Raw results written to: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Round-robin tournament between Can't Stop model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--models', nargs='+', required=True,
                        help='Two or more model checkpoint paths.')
    parser.add_argument('--games-per-pair', type=int, default=400,
                        dest='games_per_pair',
                        help='Games played between each pair (default 400). '
                             '400 gives roughly ±5%% margin on a 50%% '
                             'comparison; greedy correlation effectively '
                             'reduces this somewhat. Increase to 800+ for '
                             'tighter resolution of close matchups.')
    parser.add_argument('--eval-sims', type=int, default=100,
                        dest='eval_sims',
                        help='MCTS simulations per move (default 100).')
    parser.add_argument('--workers', type=int, default=None,
                        help='Worker processes (default min(cpu_count, 8)).')
    parser.add_argument('--output', type=str, default='tournament_results.csv',
                        help='Path to write raw pair results CSV.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Base RNG seed (default time-based).')
    args = parser.parse_args()

    if len(args.models) < 2:
        parser.error("Need at least 2 models for a tournament.")

    # Validate every path exists before starting (avoid 3-hour run that
    # fails on the last pair because one path was a typo).
    for p in args.models:
        if not os.path.exists(p):
            parser.error(f"Model file not found: {p}")

    num_workers = args.workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    base_seed = args.seed
    if base_seed is None:
        base_seed = int(time.time() * 1000) & 0x7FFFFFFF

    n = len(args.models)
    pairs = list(itertools.combinations(range(n), 2))

    # Rough runtime estimate so the user knows what they're in for.
    # Calibration: ~3s per game at 100 sims, /num_workers parallelism.
    est_per_game_s = 3.0 * (args.eval_sims / 100.0)
    est_pair_s = (args.games_per_pair * est_per_game_s) / max(num_workers, 1)
    est_total_min = (est_pair_s * len(pairs)) / 60.0

    print()
    print("=" * 72)
    print("  ROUND-ROBIN TOURNAMENT")
    print("=" * 72)
    print(f"  Models:          {n}")
    print(f"  Pairs:           {len(pairs)}")
    print(f"  Games per pair:  {args.games_per_pair}")
    print(f"  Eval sims:       {args.eval_sims}")
    print(f"  Workers:         {num_workers}")
    print(f"  Seed:            {base_seed}")
    print(f"  Output CSV:      {args.output}")
    print(f"  Est. runtime:    ~{est_total_min:.0f} minutes")
    print()
    print("  Models entered:")
    for i, p in enumerate(args.models):
        print(f"    [{i}] {format_label(p)}  ({p})")
    print("=" * 72)
    print()

    # Generate independent seeds per pair so each pair's dice sequence
    # is reproducible from its seed alone.
    pair_seed_rng = random.Random(base_seed)
    pair_seeds = {pair: pair_seed_rng.randint(0, 2**31 - 1)
                  for pair in pairs}

    pair_results = {}  # (i, j) with i<j → (wins_i, wins_j, draws)
    tournament_t0 = time.time()

    # Create the worker pool ONCE for the entire tournament. Workers
    # accumulate models in their per-process MCTS cache as pairs are
    # played, so a model loaded for an early pair is reused for every
    # subsequent pair it appears in — no redundant disk reads, no
    # process spawn overhead between pairs.
    with mp.Pool(processes=num_workers, initializer=_init_worker) as pool:
        for k, (i, j) in enumerate(pairs, start=1):
            label_i = format_label(args.models[i])
            label_j = format_label(args.models[j])
            print(f"  [Pair {k}/{len(pairs)}] {label_i} vs {label_j} ...",
                  end=' ', flush=True)

            wi, wj, dr, elapsed = run_pair(
                pool,
                args.models[i], args.models[j],
                games_per_pair=args.games_per_pair,
                eval_sims=args.eval_sims,
                num_workers=num_workers,
                base_seed=pair_seeds[(i, j)],
            )
            pair_results[(i, j)] = (wi, wj, dr)

            total = wi + wj
            wr_i = wi / total if total > 0 else 0.0
            lo, hi = wilson_ci(wi, total) if total > 0 else (0.0, 1.0)
            sig = "**" if (lo > 0.5 or hi < 0.5) else "  "
            print(f"{wi}-{wj}-{dr} ({wr_i*100:.1f}%) "
                  f"CI[{lo*100:.1f}, {hi*100:.1f}] {sig}  "
                  f"[{elapsed:.0f}s]")

    tournament_elapsed = time.time() - tournament_t0
    print()
    print(f"  Tournament complete in {tournament_elapsed/60:.1f} minutes.")
    print()

    # ---- Reports ----
    print_pair_matrix(args.models, pair_results)
    rankings = compute_overall_rankings(args.models, pair_results)
    print_rankings(rankings)
    print_pair_significance(args.models, pair_results)

    write_csv(args.models, pair_results, args.output)

    # ---- Final verdict ----
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    best_path, best_label, best_wr, best_wins, best_games = rankings[0]
    second_path, second_label, second_wr, *_ = rankings[1]
    gap = best_wr - second_wr
    print(f"  Strongest model:  {best_label}")
    print(f"                    ({best_path})")
    print(f"  Overall win rate: {best_wr*100:.1f}% across {best_games} games")
    print(f"  Gap to #2:        {gap*100:.1f} percentage points "
          f"(#2 = {second_label} at {second_wr*100:.1f}%)")
    if gap < 0.03:
        print()
        print(f"  ⚠ Top two are within 3 points — likely statistical tie.")
        print(f"    Consider this a cluster, not a clear winner.")
    print("=" * 72)
    print()


if __name__ == "__main__":
    mp.freeze_support()  # Required on Windows
    main()