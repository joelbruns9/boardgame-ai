"""Fixed-size three-way match, balanced over all six seating orders.

Default: specialist 40 vs generalist 80 vs heuristic, 3p/4-column/blocking.
Pairwise tests use independent games, seating-stratified standard errors,
and Bonferroni correction across all three comparisons. No early stopping.
"""
import argparse
import itertools
import math
from pathlib import Path
from statistics import NormalDist
import time
import zlib

NAMES = ('specialist', 'generalist', 'heuristic')
ORDERS = tuple(itertools.permutations(range(3)))  # seat -> player


def summarize(counts):
    """Six equal-size seating strata; winner counts indexed by player."""
    sizes = [sum(c) for c in counts]
    if len(counts) != 6 or min(sizes) < 2 or len(set(sizes)) != 1:
        raise ValueError('Need six equally sized strata with >=2 games each')
    n = sum(sizes)
    wins = [sum(c[i] for c in counts) for i in range(3)]
    z = NormalDist().inv_cdf(1 - 0.05 / 6)
    pairs = []
    for a, b in itertools.combinations(range(3), 2):
        difference = (wins[a] - wins[b]) / n
        # D is +1 for A wins, -1 for B wins, 0 for the third player.
        variance = sum((c[a] + c[b] - (c[a] - c[b]) ** 2 / m)
                       / (m - 1) / m for c, m in zip(counts, sizes)) / 36
        se = math.sqrt(max(0, variance))
        p = math.erfc(abs(difference) / se / math.sqrt(2)) if se else None
        pairs.append(dict(a=NAMES[a], b=NAMES[b], difference=difference,
                          se=se, p_two_sided=p,
                          p_bonferroni=min(1, 3*p) if p is not None else None,
                          simultaneous_ci95=[max(-1, difference-z*se),
                                             min(1, difference+z*se)] if se else None,
                          significant=p is not None and p < 0.05/3))
    return dict(games=n, wins=dict(zip(NAMES, wins)),
                win_rates={name: w/n for name, w in zip(NAMES, wins)},
                pairwise=pairs,
                method='Large-sample seating-stratified Wald tests; Bonferroni family alpha 0.05')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--specialist', default='runs/p4_spec_3p4b/iter_0040.pt')
    ap.add_argument('--generalist', default='runs/p4_pilot/iter_0080.pt')
    ap.add_argument('--games', type=int, default=12000)
    ap.add_argument('--seed', type=int, default=20260928)
    ap.add_argument('--out', default='runs/three_way_3p4b_12000.json')
    args = ap.parse_args()
    if args.games < 12 or args.games % 6:
        ap.error('--games must be a multiple of six and at least 12')
    if Path(args.out).exists():
        ap.error('Output exists; select a new --out to preserve results')

    import torch
    from .engine import RuleSet
    from .experiment import identity, write_json
    from .model import NetEvaluator, load_net
    from .portable_rng import PortableRng
    from .rust_pool import game_seeds, run_pool
    from .self_play import PLAIN
    from .solver import ProgressHeuristic

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rules = RuleSet(3, 4, True)
    players = [NetEvaluator(load_net(p, device=device), device=device)
               for p in (args.specialist, args.generalist)] + [ProgressHeuristic()]
    seed = zlib.crc32(f'three-way-confirmation|{args.seed}'.encode())
    seeds = game_seeds(PortableRng(seed), args.games)
    meta = identity(nets={'specialist': args.specialist, 'generalist': args.generalist},
                    rules=str(rules), games=args.games, seed=seed, input_seed=args.seed,
                    search=PLAIN, device=device, names=NAMES, seating_orders=ORDERS,
                    analysis='Fixed sample; three two-sided pairwise tests; Bonferroni alpha=.05')
    counts = [[0]*3 for _ in ORDERS]
    started = time.perf_counter()
    for offset in range(0, args.games, 600):
        batch_seeds = seeds[offset:offset+600]
        seating = [ORDERS[i % 6] for i in range(offset, offset+len(batch_seeds))]
        results = run_pool([rules]*len(batch_seeds), batch_seeds, players,
                           seating=seating, search=PLAIN)
        if len(results) != len(batch_seeds):
            raise RuntimeError('Missing game results')
        for i, (order, result) in enumerate(zip(seating, results), offset):
            if result.winner not in (0, 1, 2):
                raise RuntimeError('Unfinished game; no result may be silently dropped')
            counts[i % 6][order[result.winner]] += 1
        print(f'{offset+len(batch_seeds)}/{args.games} games complete', flush=True)
    report = dict(meta=meta, counts_by_seating=counts, result=summarize(counts),
                  seconds=round(time.perf_counter()-started, 1))
    write_json(args.out, report)
    for name, rate in report['result']['win_rates'].items():
        print(f'{name}: {rate:.2%}', flush=True)
    print(f'Results saved to {args.out}', flush=True)


if __name__ == '__main__':
    main()
