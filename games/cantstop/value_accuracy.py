"""How accurate is each win-probability estimate? Rollouts as ground truth.

The backgammon method: the "true" value of a board under the current play
is what fraction of games the side to move wins when the game is played out
from it many times with fresh dice. Every estimator is scored against that:

    net        the net's raw output for the board (what the solver's leaves
               use today)
    search1    one turn of exact search from the board, averaged over every
               opening roll (the exact TD target)
    k4, k16    the same with the selective 2-turn lookahead refining 4 / 16
               leaves

If two-turn search is much closer to the truth than one-turn, lookahead
targets would teach the net a more accurate win probability.

Scope (review, 2026-09-25): the truth is the value under PLAIN one-turn
play by this net. Lower error against it is neither necessary nor
sufficient for a search that changes decisions to play better -- judge
search with head-to-head matches; use this for calibration.

Rollout noise is binomial and known, so it is subtracted: for each board the
unbiased estimate of an estimator's squared error is
    (est - p_hat)^2 - p_hat (1 - p_hat) / (R - 1).
Comparisons between estimators are PAIRED per board (same p_hat), which is
unbiased for the difference in true squared error.

    python -m games.cantstop.value_accuracy --net runs/overnight1/iter_0400.pt
"""

import argparse
import json
import math
import time

import numpy as np
import torch

from .encoder import decode_features, encode_batch, to_absolute
from .engine import RuleSet
from .model import NetEvaluator, load_net
from .portable_rng import PortableRng
from .rust_pool import game_seeds, run_pool
from .self_play import Search

ESTIMATORS = {
    "search1": Search(exact_root=True),
    "k4": Search(exact_root=True, lookahead_k=4),
    "k16": Search(exact_root=True, lookahead_k=16),
}


def sample_boards(rules, ev, games, count, rng):
    """End-of-turn boards from the net's own games (so the distribution is
    the one it is trained and asked on), drawn uniformly without
    replacement. Returns (boards, source game id per board).

    Review finding 3: the first version drew 3*count indices, SORTED them
    and kept the first count -- i.e. the lowest indices, so the boards came
    from the earliest-generated games (mean percentile 19%, not 50%)."""
    results = run_pool([rules] * games, game_seeds(rng, games), [ev])
    pool, game_of = [], []
    n = rules.num_players
    for g, r in enumerate(results):
        for row, slot in zip(r.features, r.winner_slots):
            active = (r.winner - int(slot)) % n
            pool.append(decode_features(row, active))
            game_of.append(g)
    idx = uniform_indices(len(pool), count, rng)
    return [pool[i] for i in idx], [game_of[i] for i in idx]


def uniform_indices(n, count, rng):
    """``count`` distinct indices from range(n), uniformly (sorted only
    AFTER the draw)."""
    if count > n:
        raise ValueError(f"asked for {count} of {n}")
    order = list(range(n))
    rng.shuffle(order)
    return sorted(order[:count])


def cluster_bootstrap(values, clusters, reps=2000, seed=0):
    """Standard error of the mean of per-board ``values`` resampling whole
    source GAMES, since boards from one game are correlated (review: the
    per-board SE ignored that)."""
    values = np.asarray(values, dtype=np.float64)
    ids = sorted(set(clusters))
    by = {c: values[[i for i, g in enumerate(clusters) if g == c]] for c in ids}
    gen = np.random.default_rng(seed)
    means = []
    for _ in range(reps):
        pick = gen.choice(len(ids), size=len(ids), replace=True)
        chunk = np.concatenate([by[ids[j]] for j in pick])
        means.append(chunk.mean())
    return float(np.std(means, ddof=1))


def net_estimates(ev, boards):
    feats = encode_batch(boards)
    probs = ev.relative_probs(feats)
    return [to_absolute(probs[i], b) for i, b in enumerate(boards)]


def search_estimates(ev, boards, search, in_flight):
    res = run_pool([b.rules for b in boards],
                   list(range(len(boards))), [ev], max_turns=1,
                   allow_unfinished=True, starts=boards, search=search,
                   in_flight=in_flight)
    return [np.asarray(r.turn_values[0]) for r in res]


def rollouts(ev, boards, reps, rng, in_flight):
    starts = [b for b in boards for _ in range(reps)]
    res = run_pool([b.rules for b in starts], game_seeds(rng, len(starts)),
                   [ev], starts=starts, in_flight=in_flight)
    wins = np.zeros(len(boards))
    for i, b in enumerate(boards):
        chunk = res[i * reps:(i + 1) * reps]
        wins[i] = sum(r.winner == b.active_player for r in chunk)
    return wins / reps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--net", required=True)
    ap.add_argument("--boards", type=int, default=60)
    ap.add_argument("--rollouts", type=int, default=300)
    ap.add_argument("--source-games", type=int, default=40)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--estimators", nargs="+", default=list(ESTIMATORS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ev = NetEvaluator(load_net(args.net, device=device), device=device)
    rules = RuleSet.make(args.players)
    rng = PortableRng(args.seed)
    started = time.perf_counter()

    if args.rollouts < 2:
        raise SystemExit("--rollouts must be at least 2 (noise correction)")
    boards, games_of = sample_boards(rules, ev, args.source_games,
                                     args.boards, rng)
    movers = [b.active_player for b in boards]
    est = {"net": [v[a] for v, a in zip(net_estimates(ev, boards), movers)]}
    cost = {"net": 0.0}
    for name in args.estimators:
        t = time.perf_counter()
        vals = search_estimates(ev, boards, ESTIMATORS[name],
                                in_flight=16 if "k" in name else 64)
        cost[name] = (time.perf_counter() - t) / len(boards)
        est[name] = [v[a] for v, a in zip(vals, movers)]
    t = time.perf_counter()
    p_hat = rollouts(ev, boards, args.rollouts, rng, in_flight=64)
    rollout_seconds = time.perf_counter() - t

    noise = p_hat * (1 - p_hat) / (args.rollouts - 1)
    sq = {k: (np.asarray(v) - p_hat) ** 2 - noise for k, v in est.items()}
    report = {"boards": len(boards), "source_games": len(set(games_of)),
              "rollouts": args.rollouts,
              "rollout_noise_rmse": float(math.sqrt(noise.mean())),
              "estimators": {}, "vs_net": {}, "vs_search1": {}}
    for k, v in sq.items():
        report["estimators"][k] = {
            "rmse": float(math.sqrt(max(v.mean(), 0.0))),
            "mse": float(v.mean()),
            "bias": float(np.mean(np.asarray(est[k]) - p_hat)),
            "seconds_per_board": round(cost[k], 4),
        }
    for base in ("net", "search1"):
        if base not in sq:
            continue
        for k in sq:
            if k == base:
                continue
            d = sq[k] - sq[base]
            report[f"vs_{base}"][k] = {
                "mse_change": float(d.mean()),
                "se_per_board": float(d.std(ddof=1) / math.sqrt(len(d))),
                "se": cluster_bootstrap(d, games_of),
                "relative": float(d.mean() / sq[base].mean())
                if sq[base].mean() > 0 else None,
            }
    report["seconds"] = round(time.perf_counter() - started, 1)
    report["rollout_seconds"] = round(rollout_seconds, 1)
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"report": report, "p_hat": p_hat.tolist(),
                   "source_game": games_of,
                       "estimates": {k: list(map(float, v))
                                     for k, v in est.items()}}, fh)


if __name__ == "__main__":
    main()
