"""Saved-position comparison harness; reports do not claim playing strength."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from .decision_search import TurnTableBackend
from .experiment import file_sha256, identity, write_json
from .snapshot import from_snapshot
from .turn_search import TurnSearchConfig
from .rollout_search import RolloutBackend, RolloutConfig
from .progressive_search import ProgressiveBackend, ProgressiveConfig, EarlyTurnPolicy, audit_filtering

DEFAULT_SUITE = Path(__file__).parent / 'tests/fixtures/search_decisions_v1.json'


def load_suite(path=DEFAULT_SUITE, split='development'):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if data['format'] != 'cantstop-search-decisions-v1':
        raise ValueError('unsupported fixture format')
    ids = [row['id'] for row in data['positions']]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate fixture id')
    if split not in ('development', 'heldout', 'all'):
        raise ValueError('unknown split')
    return [row for row in data['positions'] if split == 'all' or row['split'] == split]


def compare_position(row, baseline, candidate, *, audit_samples=0):
    state = from_snapshot(row['snapshot'])
    a = baseline.evaluate(state).to_dict()
    decision = candidate.evaluate(state)
    b = decision.to_dict()
    result = {'id': row['id'], 'split': row['split'], 'tags': row['tags'],
            'snapshot': row['snapshot'], 'baseline': a, 'candidate': b,
            'changed': a['selected'] != b['selected'], 'identical': a == b}
    if isinstance(candidate, RolloutBackend):
        result['rollout'] = candidate.last_stats
    if isinstance(candidate, ProgressiveBackend):
        result['progressive'] = candidate.last_stats
        if audit_samples:
            config = replace(candidate.rollout, samples=audit_samples, horizon=None, seed=candidate.rollout.seed+1000003)
            policy = EarlyTurnPolicy(candidate.evaluator, candidate.config)
            reference = RolloutBackend(candidate.evaluator, config, policy_factory=policy)
            ref_decision = reference.evaluate(state)
            if not state.game_over:
                result['filtering_audit'] = audit_filtering(decision, ref_decision)
            result['audit_reference'] = {'result': ref_decision.to_dict(), 'rollout': reference.last_stats}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, default=DEFAULT_SUITE)
    p.add_argument('--split', choices=('development', 'heldout', 'all'), default='development')
    p.add_argument('--checkpoint', help='Omit for deterministic heuristic verification')
    p.add_argument('--device', default='cpu')
    p.add_argument('--expansions', type=int, default=0)
    p.add_argument('--depth', type=int, default=1)
    p.add_argument('--backend', choices=('selective', 'rollout', 'progressive'), default='selective')
    p.add_argument('--common-random-numbers', action='store_true')
    p.add_argument('--dice-luck', action='store_true')
    p.add_argument('--samples', type=int, default=32)
    horizon = p.add_mutually_exclusive_group()
    horizon.add_argument('--horizon', type=int, default=1)
    horizon.add_argument('--full-game', action='store_true')
    p.add_argument('--seed', type=int, default=20260930)
    p.add_argument('--max-turns', type=int, default=1000)
    p.add_argument('--max-rolls', type=int, default=10000)
    p.add_argument('--stage-horizons', nargs='+', default=['0','1'])
    p.add_argument('--stage-samples', type=int, nargs='+', default=[8,32])
    p.add_argument('--margin', type=float)
    p.add_argument('--max-candidates', type=int)
    p.add_argument('--early-turns', type=int, default=0)
    p.add_argument('--early-expansions', type=int, default=0)
    p.add_argument('--early-depth', type=int, default=1)
    p.add_argument('--cache-entries', type=int, default=8)
    p.add_argument('--cache-positions', type=int, default=200000)
    p.add_argument('--audit-samples', type=int, default=0)
    p.add_argument('--position', action='append', help='Restrict to named fixture IDs (repeatable)')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    if args.out.exists():
        p.error('output exists; choose a new --out')
    if args.backend == 'selective' and (args.common_random_numbers or args.dice_luck):
        p.error('variance-reduction switches require rollout or progressive backend')
    if args.backend != 'selective' and args.expansions != 0:
        p.error('--expansions applies only to selective search')
    try:
        config = (RolloutConfig(common_random_numbers=args.common_random_numbers, dice_luck=args.dice_luck, samples=args.samples, horizon=None if args.full_game else args.horizon,
                    seed=args.seed, max_turns=args.max_turns, max_rolls=args.max_rolls)
                  if args.backend != 'selective' else TurnSearchConfig(expansions=args.expansions, depth=args.depth))
        progressive_config = None
        if args.backend == 'progressive':
            progressive_config = ProgressiveConfig(
                horizons=tuple(None if h=='full' else int(h) for h in args.stage_horizons),
                samples=tuple(args.stage_samples), margin=args.margin, max_candidates=args.max_candidates,
                early_turns=args.early_turns, expansions=args.early_expansions, depth=args.early_depth,
                cache_entries=args.cache_entries, cache_positions=args.cache_positions)
        if args.audit_samples < 0 or (args.audit_samples and args.backend != 'progressive'):
            p.error('--audit-samples requires progressive backend and a nonnegative count')
    except ValueError as exc:
        p.error(str(exc))
    if args.checkpoint:
        from .model import NetEvaluator, load_net
        evaluator = NetEvaluator(load_net(args.checkpoint, device=args.device), device=args.device)
    else:
        from .solver import ProgressHeuristic
        evaluator = ProgressHeuristic()
    rows = load_suite(args.suite, args.split)
    if args.position:
        unknown = set(args.position) - {row['id'] for row in rows}
        if unknown:
            p.error(f'position IDs missing from selected split: {sorted(unknown)}')
        rows = [row for row in rows if row['id'] in args.position]
    if not rows:
        p.error('empty fixture split')
    report = {'format': 'cantstop-decision-report-v1', 'status': 'running',
              'meta': identity(nets={'shared': args.checkpoint} if args.checkpoint else {},
                    evaluator='NN' if args.checkpoint else 'ProgressHeuristic',
                    device=args.device, backend=args.backend, search=asdict(config),
                    progressive=None if progressive_config is None else asdict(progressive_config), audit_samples=args.audit_samples, split=args.split,
                    suite_sha256=file_sha256(args.suite),
                    source_sha256={name: file_sha256(Path(__file__).parent / name)
                        for name in ('decision_search.py', 'decision_compare.py',
                                     'rust_solver.py', 'turn_search.py', 'rollout_search.py', 'progressive_search.py', 'engine.py', 'model.py',
                                     'cantstop_rust/src/lib.rs')}), 'positions': []}
    baseline = TurnTableBackend(evaluator)
    candidate = (RolloutBackend(evaluator, config) if args.backend == 'rollout'
                 else TurnTableBackend(evaluator, search_config=config))
    if progressive_config is not None:
        candidate = ProgressiveBackend(evaluator, progressive_config, config)
    write_json(args.out, report)
    try:
        for row in rows:
            report['positions'].append(compare_position(row, baseline, candidate, audit_samples=args.audit_samples))
            write_json(args.out, report)
            print(f"{len(report['positions'])}/{len(rows)} {row['id']}", flush=True)
        report['status'] = 'complete'
        report['summary'] = {'positions': len(rows),
            'identical': sum(r['identical'] for r in report['positions']),
            'changed': sum(r['changed'] for r in report['positions'])}
    except BaseException as exc:
        report['status'] = 'incomplete'
        report['error'] = f'{type(exc).__name__}: {exc}'
        if isinstance(candidate, (RolloutBackend, ProgressiveBackend)):
            report['failed_search'] = candidate.last_stats
            if isinstance(candidate, RolloutBackend):
                report['failed_rollout'] = candidate.last_stats
        raise
    finally:
        write_json(args.out, report)


if __name__ == '__main__':
    main()
