"""Fixed-budget variance/time ablation on saved decisions, not an arena."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .decision_compare import DEFAULT_SUITE, load_suite
from .experiment import identity, file_sha256, write_json
from .rollout_search import RolloutBackend, RolloutConfig
from .snapshot import from_snapshot

MODES = {'plain': (False, False), 'shared': (True, False),
         'luck': (False, True), 'both': (True, True)}
POSITIONS = ('2p5n_opening', '2p5n_three_runners', '2p5b_opening',
             '3p4b_opening', '3p4n_three_runners', '4p3n_three_runners')


def summarize(records, samples):
    summary = []
    for position in dict.fromkeys(r['position'] for r in records):
        for mode in MODES:
            group = [r for r in records if r['position'] == position and r['mode'] == mode]
            elapsed = float(np.mean([r['stats']['elapsed_seconds'] for r in group]))
            for pair in range(len(group[0]['stats']['differences'])):
                differences = [r['stats']['differences'][pair] for r in group]
                variance = float(np.mean([d['sample_variance'] for d in differences])) / samples
                between = float(np.var([d['mean'] for d in differences], ddof=1))
                summary.append({'position': position, 'mode': mode,
                    'a': differences[0]['a'], 'b': differences[0]['b'],
                    'batches': len(group), 'mean_difference': float(np.mean([d['mean'] for d in differences])),
                    'estimated_variance_of_mean': variance,
                    'between_batch_variance': between, 'mean_seconds': elapsed,
                    'variance_seconds': variance * elapsed,
                    'between_batch_variance_seconds': between * elapsed})
    for row in summary:
        base = next(r for r in summary if r['position'] == row['position'] and
                    r['a'] == row['a'] and r['b'] == row['b'] and r['mode'] == 'plain')
        row['variance_seconds_relative_to_plain'] = (row['variance_seconds'] / base['variance_seconds']
            if base['variance_seconds'] > 1e-20 else None)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--samples', type=int, default=32)
    p.add_argument('--batches', type=int, default=8)
    p.add_argument('--horizon', type=int, default=1)
    p.add_argument('--seed', type=int, default=20261001)
    p.add_argument('--position', action='append')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    if args.out.exists(): p.error('output exists')
    if args.samples < 2 or args.batches < 2: p.error('samples and batches must be at least two')
    RolloutConfig(samples=args.samples, horizon=args.horizon, seed=args.seed)
    selected = set(args.position or POSITIONS)
    rows = [r for r in load_suite() if r['id'] in selected]
    if {r['id'] for r in rows} != selected: p.error('unknown development position')
    from .model import NetEvaluator, load_net
    evaluator = NetEvaluator(load_net(args.checkpoint, device=args.device), device=args.device)
    report = {'status': 'running', 'meta': identity(nets={'shared': args.checkpoint},
        samples=args.samples, batches=args.batches, horizon=args.horizon, seed=args.seed,
        device=args.device, suite_sha256=file_sha256(DEFAULT_SUITE),
        source_sha256={name: file_sha256(Path(__file__).parent/name) for name in
            ('benchmark_rollout_variance.py', 'rollout_search.py', 'rust_solver.py',
             'decision_search.py', 'model.py', 'engine.py', 'cantstop_rust/src/lib.rs')}), 'records': []}
    write_json(args.out, report)
    try:
        for row in rows:
            state = from_snapshot(row['snapshot'])
            RolloutBackend(evaluator, RolloutConfig(samples=1, horizon=0)).evaluate(state)
            for batch in range(args.batches):
                order = list(MODES)
                offset = batch % len(order)
                for mode in order[offset:] + order[:offset]:
                    shared, luck = MODES[mode]
                    config = RolloutConfig(samples=args.samples, horizon=args.horizon,
                        seed=args.seed+batch, common_random_numbers=shared, dice_luck=luck)
                    backend = RolloutBackend(evaluator, config)
                    backend.evaluate(state)
                    report['records'].append({'position': row['id'], 'batch': batch,
                        'mode': mode, 'stats': backend.last_stats})
                write_json(args.out, report)
            print(f"completed {row['id']}", flush=True)
        report['summary'] = summarize(report['records'], args.samples)
        report['status'] = 'complete'
        report['interpretation'] = ('Exploratory fixed-position comparison. Lower variance_seconds is better. '
            'Sampling precision only; no strength or NN-bias claim. Shared-dice gains may depend on position. '
            'Multiple candidate pairs on one position are correlated, not independent replications.')
    except BaseException as exc:
        report['status'] = 'incomplete'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_json(args.out, report)


if __name__ == '__main__':
    main()
