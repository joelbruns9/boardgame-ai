"""Matched full-game throughput and NN numerical checks; never an arena test."""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import t

from .adaptive_search import atomic_json, source_identity
from .experiment import file_sha256, identity
from .model import NetEvaluator, load_net
from .engine import GameState
from .pool_rollout import PoolRolloutBackend
from .progressive_search import EarlyTurnPolicy, ProgressiveConfig
from .rollout_search import RolloutBackend, RolloutConfig
from .snapshot import from_snapshot


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--suite', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--samples', type=int, default=128)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--positions', type=int, default=3)
    p.add_argument('--seed', type=int, default=2026100210)
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--in-flight', type=int, default=64)
    p.add_argument('--max-rows', type=int, default=1_000_000)
    p.add_argument('--equivalence-margin-pp', type=float, default=.5)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args(argv)
    if args.samples < 2 or args.repeats < 1 or args.positions < 1 or args.equivalence_margin_pp <= 0:
        p.error('require positive budgets and equivalence margin')
    if args.out.exists():
        p.error('choose a new benchmark output')
    rows = json.loads(Path(args.suite).read_text())['positions'][:args.positions]
    if len(rows) != args.positions:
        p.error('suite has too few positions')
    native = importlib.import_module('cantstop_rust.cantstop_rust')
    sources = source_identity()
    for name in ('pool_rollout.py', 'benchmark_audit_pool.py'):
        sources[name] = file_sha256(Path(__file__).with_name(name))
    sources['native_extension'] = file_sha256(native.__file__)
    ev = NetEvaluator(load_net(args.checkpoint, device=args.device), device=args.device)
    ev([GameState(from_snapshot(rows[0]['snapshot']).rules)])
    report = {'status': 'running', 'configuration': vars(args) | {'out': str(args.out)},
              'meta': identity(nets={'shared': args.checkpoint}, source_sha256=sources),
              'suite_sha256': file_sha256(args.suite), 'records': [], 'comparisons': []}
    atomic_json(args.out, report)
    started = time.perf_counter()
    try:
        for repeat in range(args.repeats):
            for position, row in enumerate(rows):
                cfg = RolloutConfig(samples=args.samples, horizon=None, seed=args.seed + repeat*100 + position,
                                    common_random_numbers=True, dice_luck=True)
                samples = {}
                order = ('serial', 'pool') if (repeat+position)%2 == 0 else ('pool', 'serial')
                for mode in order:
                    backend = (PoolRolloutBackend(ev, cfg, retain_samples=True, threads=args.threads,
                                                  in_flight=args.in_flight, max_rows=args.max_rows)
                               if mode == 'pool' else RolloutBackend(ev, cfg, retain_samples=True,
                                                                    policy_factory=EarlyTurnPolicy(ev, ProgressiveConfig())))
                    before, calls = ev.rows, ev.calls
                    tick = time.perf_counter()
                    decision = backend.evaluate(from_snapshot(row['snapshot']))
                    seconds = time.perf_counter() - tick
                    report['records'].append({'id': row['id'], 'repeat': repeat, 'backend': mode,
                                              'seconds': seconds, 'nn_rows': ev.rows-before, 'nn_calls': ev.calls-calls,
                                              'rollouts': sum(a['terminal_samples'] for a in backend.last_stats['actions']),
                                              'selected': decision.selected.key, 'stats': backend.last_stats})
                    samples[mode] = backend.last_samples
                    atomic_json(args.out, report)
                    print(f'{repeat+1}/{args.repeats} {row["id"]} {mode}: {seconds:.2f}s, {ev.rows-before:,} NN rows', flush=True)
                actor = from_snapshot(row['snapshot']).active_player
                for key in samples['serial']:
                    x = samples['pool'][key]['adjusted'][:, actor] - samples['serial'][key]['adjusted'][:, actor]
                    se = float(x.std(ddof=1)/math.sqrt(len(x)))
                    # Per-action 95% diagnostic interval; no multiplicity or strength claim.
                    radius = float(t.isf(.025, len(x)-1)*se)
                    mean = float(x.mean())
                    limit = args.equivalence_margin_pp/100
                    report['comparisons'].append({'id': row['id'], 'repeat': repeat, 'action': key,
                                                   'mean_backend_difference_pp': mean*100,
                                                   'low_pp': (mean-radius)*100, 'high_pp': (mean+radius)*100,
                                                   'standard_error_pp': se*100,
                                                   'within_declared_margin': abs(mean)+radius <= limit,
                                                   'raw_disagreements': int(np.any(samples['pool'][key]['raw'] != samples['serial'][key]['raw'], axis=1).sum()),
                                                   'max_adjusted_sample_difference': float(np.max(np.abs(samples['pool'][key]['adjusted']-samples['serial'][key]['adjusted'])))})
                atomic_json(args.out, report)
        totals = {mode: sum(r['seconds'] for r in report['records'] if r['backend'] == mode)
                  for mode in ('serial', 'pool')}
        report['summary'] = {'seconds': totals, 'speedup': totals['serial']/totals['pool'],
                             'all_checks_within_declared_margin': all(c['within_declared_margin'] for c in report['comparisons']),
                             'raw_disagreements': sum(c['raw_disagreements'] for c in report['comparisons']),
                             'interpretation': 'Matched-work throughput; approximate NN numerical diagnostic intervals, not arena evidence.'}
        report['status'] = 'complete'
    except BaseException as exc:
        report['status'] = 'incomplete'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        atomic_json(args.out, report)
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
