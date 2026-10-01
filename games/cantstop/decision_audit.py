"""Independent full-game audit of frozen pilot choices, with matched fresh controls."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import t

from .adaptive_search import atomic_json, source_identity
from .decision_compare import load_suite
from .decision_pilot import bucket_of, collect_positions, derived_seed, variant_id
from .decision_search import TurnTableBackend, actions
from .engine import ALL_RULESETS, GameState
from .experiment import file_sha256, identity
from .progressive_search import EarlyTurnPolicy, ProgressiveConfig
from .rollout_search import RolloutBackend, RolloutConfig
from .snapshot import from_snapshot


def pilot_source_changes(pilot, sources, *, allow_native_rebuild=False):
    changes = {k: {'pilot': v, 'current': sources.get(k)}
               for k, v in pilot['meta']['source_sha256'].items() if sources.get(k) != v}
    if changes and (not allow_native_rebuild or any(
            not k.endswith(('.pyd', '.so')) or v['current'] is None for k, v in changes.items())):
        raise ValueError('search source differs from pilot; only an explicit native rebuild is permitted')
    return changes


def recheck_pilot_baselines(pilot, rows, evaluator, tolerance=2e-6):
    if {r['id'] for r in rows} != set(pilot['baselines']):
        raise ValueError('pilot baseline suite differs')
    largest = 0.
    for row in rows:
        old = pilot['baselines'][row['id']]['decision']
        current = TurnTableBackend(evaluator).evaluate(from_snapshot(row['snapshot'])).to_dict()
        if (current['actor'], current['selected']) != (old['actor'], old['selected']):
            raise ValueError(f'native rebuild changes baseline choice: {row["id"]}')
        old_values = {x['action']: np.asarray(x['value']) for x in old['options']}
        current_values = {x['action']: np.asarray(x['value']) for x in current['options']}
        if old_values.keys() != current_values.keys():
            raise ValueError(f'native rebuild changes legal baseline actions: {row["id"]}')
        for key in old_values:
            if current_values[key].shape != old_values[key].shape or not np.isfinite(current_values[key]).all():
                raise ValueError('invalid rebuilt baseline values')
            error = float(np.max(np.abs(current_values[key] - old_values[key])))
            largest = max(largest, error)
            if error > tolerance:
                raise ValueError(f'native rebuild changes baseline values: {row["id"]}')
    return {'positions': len(rows), 'all_choices_match': True,
            'max_absolute_value_difference': largest, 'tolerance': tolerance}


def choose_focus(pilot, rows):
    """All disagreements plus decisions unresolved at the largest pilot ceiling."""
    highest = max(pilot['configuration']['budgets'])
    changed = set(pilot['summary']['changed_position_ids'])
    unresolved = {c['id'] for c in pilot['cells'] if c['budget'] == highest and c['adaptive']['fallback']}
    result = []
    for row in rows:
        if row['id'] in changed | unresolved:
            result.append({**row, 'audit_role': 'changed' if row['id'] in changed else 'unresolved',
                           'pilot_choices': [{'horizon': c['horizon'], 'budget': c['budget'],
                                              'selected': c['decision']['selected'], 'fallback': c['adaptive']['fallback']}
                                             for c in pilot['cells'] if c['id'] == row['id']]})
    return result


def intervals(samples, actor, *, alpha, total_pairs, looks):
    """Correct over all planned pairs/looks, including fresh control comparisons.

    Adjusted and raw t intervals are approximate. Raw Hoeffding intervals bound
    paired payoff differences in [-1,1]. No coverage claim for arbitrary
    nonnormal corrected streams; a zero observed variance stays unresolved.
    """
    keys = list(samples)
    n = len(samples[keys[0]]['raw'])
    if n < 2 or any(len(v[f]) != n for v in samples.values() for f in ('raw', 'adjusted')):
        raise ValueError('at least two aligned samples required')
    adjusted = {k: np.asarray(v['adjusted'])[:, actor] for k, v in samples.items()}
    raw = {k: np.asarray(v['raw'])[:, actor] for k, v in samples.items()}
    if any(not np.isfinite(x).all() for x in [*adjusted.values(), *raw.values()]):
        raise ValueError('nonfinite samples')
    if any(np.any((x != 0) & (x != 1)) for x in raw.values()):
        raise ValueError('full-game audit requires terminal raw payoffs')
    quantile = float(t.isf(alpha / (2 * total_pairs * looks), n - 1))
    result = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pair = {'a': a, 'b': b, 'samples': n, 'approximate': True}
            for name, values in (('adjusted', adjusted), ('raw', raw)):
                d = values[a] - values[b]
                mean = float(d.mean())
                se = float(d.std(ddof=1) / math.sqrt(n))
                radius = quantile * se if se > 1e-14 else None
                pair[name] = {'mean': mean, 'standard_error': se,
                              'low': None if radius is None else mean - radius,
                              'high': None if radius is None else mean + radius,
                              'min': float(d.min()), 'max': float(d.max())}
            radius = math.sqrt(2 * math.log(2 * total_pairs * looks / alpha) / n)
            pair['raw_hoeffding'] = {'low': max(-1., pair['raw']['mean'] - radius),
                                     'high': min(1., pair['raw']['mean'] + radius), 'radius': radius}
            result.append(pair)
    return result


def contrast(pairs, candidate, baseline):
    if candidate == baseline:
        return {'classification': 'unchanged', 'mean': 0., 'low': 0., 'high': 0.}
    pair = next(p for p in pairs if {p['a'], p['b']} == {candidate, baseline})
    x = pair['adjusted']
    sign = 1 if pair['a'] == candidate else -1
    mean = sign * x['mean']
    low, high = (x['low'], x['high']) if sign == 1 else (None if x['high'] is None else -x['high'],
                                                       None if x['low'] is None else -x['low'])
    classification = 'uncertain' if low is None else ('helps' if low > 0 else ('hurts' if high < 0 else 'uncertain'))
    return {'classification': classification, 'mean': mean, 'low': low, 'high': high,
            'standard_error': x['standard_error'], 'raw_mean': sign * pair['raw']['mean']}


def stage_result(row, baseline, samples, *, alpha, total_pairs, looks):
    actor = from_snapshot(row['snapshot']).active_player
    pairs = intervals(samples, actor, alpha=alpha, total_pairs=total_pairs, looks=looks)
    candidate_choices = sorted({c['selected'] for c in row.get('pilot_choices', [])})
    fixed_changes = [{'action': a, **contrast(pairs, a, baseline['selected'])}
                     for a in candidate_choices if a != baseline['selected']]
    baseline_comparisons = [{'action': a, **contrast(pairs, a, baseline['selected'])}
                            for a in samples if a != baseline['selected']]
    needs_extension = (any(c['classification'] == 'uncertain' for c in fixed_changes)
                       if fixed_changes else any(c['classification'] == 'uncertain' for c in baseline_comparisons))
    return {'samples_per_action': len(next(iter(samples.values()))['raw']),
            'means': {k: {field: np.asarray(v[field]).mean(axis=0).tolist() for field in ('raw', 'adjusted')}
                      for k, v in samples.items()},
            'pairs': pairs, 'fixed_changes': fixed_changes, 'baseline_comparisons': baseline_comparisons,
            'needs_extension': needs_extension,
            'adjusted_outside_unit_interval': {k: int(np.any((np.asarray(v['adjusted']) < 0) |
                                                             (np.asarray(v['adjusted']) > 1), axis=1).sum())
                                              for k, v in samples.items()}}


def prepare_suite(pilot, pilot_rows, evaluator, seed):
    focus = choose_focus(pilot, pilot_rows)
    variants = tuple(r for r in ALL_RULESETS if variant_id(r) in {variant_id(from_snapshot(x['snapshot']).rules) for x in focus})
    fresh = collect_positions(evaluator, seed=derived_seed(seed, 'fresh-audit-controls'), per_bucket=1,
                              games_per_variant=2, variants=variants)
    existing = {json.dumps(r['snapshot'], separators=(',', ':')) for r in pilot_rows}
    controls = []
    for row in focus:
        state = from_snapshot(row['snapshot'])
        matches = [r for r in fresh['positions'] if r['variant'] == variant_id(state.rules)
                   and (r['stage'], r['phase']) == bucket_of(state)
                   and json.dumps(r['snapshot'], separators=(',', ':')) not in existing]
        if not matches:
            raise ValueError(f'no fresh control matching {row["id"]}')
        control = matches[0]
        controls.append({**control, 'id': 'control_' + control['id'], 'audit_role': 'control',
                         'matched_focus': row['id'], 'split': 'heldout'})
    return {'format': 'cantstop-search-decisions-v1', 'purpose': 'independent full-game decision audit',
            'control_collection': fresh['collection'], 'positions': focus + controls}


def save_readable(out, report):
    lines = ['# Independent decision audit', '', f'Status: **{report["status"]}**.', '',
             'Terminal full-game outcomes, baseline continuation, fresh seeds. Adjusted confidence intervals are approximate',
             'and corrected across all planned action pairs and two looks. This is a small decision audit, not an arena result.', '',
             '| Position | Role | Samples/action | Frozen change or alternative vs baseline | Gain (percentage points) | Corrected interval | Result |',
             '| --- | --- | ---: | --- | ---: | --- | --- |']
    for row in report['positions']:
        if not row['stages']:
            continue
        stage = row['stages'][-1]
        comparisons = stage['fixed_changes'] or stage['baseline_comparisons']
        for c in comparisons:
            band = 'unresolved variance' if c['low'] is None else f'[{c["low"]*100:.3f}, {c["high"]*100:.3f}]'
            lines.append(f'| {row["id"]} | {row["audit_role"]} | {stage["samples_per_action"]} | '
                         f'{c["action"]} vs {row["baseline"]["selected"]} | {c["mean"]*100:.3f} | {band} | {c["classification"]} |')
    lines += ['', 'Interpretation applies only to the declared continuation policy. Control alternatives were not frozen pilot changes.',
              'Raw outcomes, conservative raw bounds, full per-action samples, code/checkpoint hashes, and batch costs are saved in the JSON and sample files.']
    if report.get('error'):
        lines += ['', 'Error: ' + report['error']]
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--backend', choices=('serial', 'pool'), default='serial')
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--in-flight', type=int, default=64)
    p.add_argument('--max-rows', type=int, default=1_000_000)
    p.add_argument('--seed', type=int, default=2026100101)
    p.add_argument('--budgets', type=int, nargs='+', default=[1024, 4096])
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--alpha', type=float, default=.05)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--allow-native-rebuild', action='store_true',
                   help='permit a recorded native-only change after rechecking every pilot baseline')
    args = p.parse_args(argv)
    if args.out.exists() != args.resume:
        p.error('choose a new output or resume an existing output')
    if len(args.budgets) != 2 or not 2 <= args.budgets[0] < args.budgets[1] or args.batch < 1 or not 0 < args.alpha < 1:
        p.error('require two increasing budgets, positive batch, and 0 < alpha < 1')
    if args.threads < 0 or args.in_flight < 1 or args.max_rows < 1:
        p.error('invalid pool resource limits')
    pilot = json.loads(args.pilot.read_text())
    if pilot['status'] != 'complete':
        p.error('audit requires a completed pilot')
    model_key = file_sha256(args.checkpoint)
    if model_key != pilot['meta']['nets']['shared']['sha256']:
        p.error('checkpoint differs from pilot')
    pilot_suite = Path(pilot['suite'])
    if file_sha256(pilot_suite) != pilot['suite_sha256']:
        p.error('pilot suite changed')
    sources = source_identity()
    sources['decision_pilot.py'] = file_sha256(Path(__file__).with_name('decision_pilot.py'))
    try:
        native_changes = pilot_source_changes(pilot, sources, allow_native_rebuild=args.allow_native_rebuild)
    except ValueError as exc:
        p.error(str(exc))
    sources['decision_audit.py'] = file_sha256(__file__)
    if args.backend == 'pool':
        sources['pool_rollout.py'] = file_sha256(Path(__file__).with_name('pool_rollout.py'))
        crate = Path(__file__).with_name('cantstop_rust')
        for name in ('src/audit_stream.rs', 'src/selfplay.rs', 'src/lib.rs', 'Cargo.toml', 'Cargo.lock'):
            sources['cantstop_rust/' + name] = file_sha256(crate / name)
        import cantstop_rust.cantstop_rust as native
        sources['native_extension'] = file_sha256(native.__file__)
    cfg = {'seed': args.seed, 'budgets': args.budgets, 'batch': args.batch, 'alpha': args.alpha,
           'horizon': 'full_game', 'continuation': 'baseline', 'dice_luck': True, 'common_random_numbers': True,
           'control_budget': args.budgets[0], 'extension_rule': 'uncertain fixed change; for unresolved focus, uncertain alternative vs baseline',
           'device': args.device, 'backend': args.backend,
           'threads': args.threads, 'in_flight': args.in_flight, 'max_rows': args.max_rows,
           'allow_native_rebuild': args.allow_native_rebuild}
    signature = hashlib.sha256(json.dumps({'pilot': file_sha256(args.pilot), 'model': model_key,
                                          'source': sources, 'configuration': cfg}, sort_keys=True).encode()).hexdigest()
    report = json.loads(args.out.read_text()) if args.resume else None
    if report and report['signature'] != signature:
        p.error('resume identity differs')
    from .model import NetEvaluator, load_net
    evaluator = NetEvaluator(load_net(args.checkpoint, device=args.device), device=args.device)
    evaluator([GameState(ALL_RULESETS[0])])
    compatibility = None
    if native_changes:
        compatibility = {'source_changes': native_changes,
                         'baseline_recheck': recheck_pilot_baselines(pilot, load_suite(pilot_suite), evaluator)}
    suite_path = args.out.with_suffix('.positions.json')
    if report is None:
        if suite_path.exists():
            p.error('audit collection already exists; choose a new output')
        suite = prepare_suite(pilot, load_suite(pilot_suite), evaluator, args.seed)
        atomic_json(suite_path, suite)
        rows = suite['positions']
        total_pairs = sum(len(actions(from_snapshot(r['snapshot']))) * (len(actions(from_snapshot(r['snapshot']))) - 1) // 2 for r in rows)
        report = {'format': 'cantstop-decision-audit-v1', 'status': 'running', 'signature': signature,
                  'configuration': cfg, 'total_pairs': total_pairs, 'suite': str(suite_path),
                  'suite_sha256': file_sha256(suite_path), 'positions': [], 'invocations': [],
                  'meta': identity(nets={'shared': args.checkpoint}, source_sha256=sources,
                                   pilot_source_sha256=pilot['meta']['source_sha256'],
                                   native_compatibility=compatibility, pilot_sha256=file_sha256(args.pilot))}
        for row in rows:
            state = from_snapshot(row['snapshot'])
            baseline = TurnTableBackend(evaluator).evaluate(state).to_dict()
            if row['audit_role'] != 'control' and baseline['selected'] != pilot['baselines'][row['id']]['decision']['selected']:
                raise ValueError('baseline choice differs from pilot')
            token = hashlib.sha256(row['id'].encode()).hexdigest()[:20]
            sample_path = args.out.with_suffix('.samples') / (token + '.json')
            atomic_json(sample_path, {'signature': signature, 'id': row['id'], 'seed': derived_seed(args.seed, 'independent-full-game-audit', row['id']),
                                      'samples': {a.key: {'raw': [], 'adjusted': []} for a in actions(state)}, 'batches': []})
            report['positions'].append({**row, 'baseline': baseline, 'sample_file': str(sample_path), 'stages': []})
    elif file_sha256(suite_path) != report['suite_sha256']:
        p.error('audit suite changed')
    report['status'] = 'running'
    report.pop('error', None)
    started = time.perf_counter()
    report['invocations'].append({'resume': args.resume})
    atomic_json(args.out, report)
    try:
        pilot_seeds = {c['seed'] for c in pilot['cells']}
        for stage_index, target in enumerate(args.budgets):
            for row in report['positions']:
                if len(row['stages']) > stage_index:
                    continue
                if stage_index and (row['audit_role'] == 'control' or not row['stages'][0]['needs_extension']):
                    continue
                data = json.loads(Path(row['sample_file']).read_text())
                if data['signature'] != signature or data['seed'] in pilot_seeds:
                    raise ValueError('sample identity differs or seed reused from pilot')
                state = from_snapshot(row['snapshot'])
                n = len(next(iter(data['samples'].values()))['raw'])
                if any(len(v[f]) != n for v in data['samples'].values() for f in ('raw', 'adjusted')) or n > target:
                    raise ValueError('unaligned or excess audit samples')
                while n < target:
                    count = min(args.batch, target - n)
                    rollout = RolloutConfig(samples=count, horizon=None, seed=data['seed'], sample_offset=n,
                                            dice_luck=True, common_random_numbers=True)
                    if args.backend == 'pool':
                        from .pool_rollout import PoolRolloutBackend
                        backend = PoolRolloutBackend(evaluator, rollout, retain_samples=True,
                                                     threads=args.threads, in_flight=args.in_flight,
                                                     max_rows=args.max_rows)
                    else:
                        policy = EarlyTurnPolicy(evaluator, ProgressiveConfig())
                        backend = RolloutBackend(evaluator, rollout, policy_factory=policy, retain_samples=True)
                    before = evaluator.rows
                    report['current_batch'] = {'id': row['id'], 'offset': n, 'samples': count, 'target': target}
                    atomic_json(args.out, report)
                    backend.evaluate(state)
                    if any(a['terminal_samples'] != count for a in backend.last_stats['actions']):
                        raise ValueError('audit trajectory failed to reach terminal outcome')
                    for key in data['samples']:
                        for field in ('raw', 'adjusted'):
                            data['samples'][key][field].extend(backend.last_samples[key][field].tolist())
                    data['batches'].append({'offset': n, 'samples': count, 'stats': backend.last_stats, 'nn_rows': evaluator.rows - before})
                    n += count
                    atomic_json(row['sample_file'], data)
                    report.pop('current_batch', None)
                    row['committed_samples_per_action'] = n
                    atomic_json(args.out, report)
                    print(f'{row["id"]} {row["audit_role"]}: {n}/{target} full-game samples/action, '
                          f'batch {backend.last_stats["elapsed_seconds"]:.1f}s', flush=True)
                row['stages'].append(stage_result(row, row['baseline'], data['samples'], alpha=args.alpha,
                                                   total_pairs=report['total_pairs'], looks=2))
                atomic_json(args.out, report)
                save_readable(args.out, report)
                print(f'completed planned look: {row["id"]} n={n}', flush=True)
        report['status'] = 'complete'
        changes = [c for r in report['positions'] for c in (r['stages'][-1]['fixed_changes'] if r['stages'] else [])]
        report['summary'] = {'positions': len(report['positions']), 'roles': dict(Counter(r['audit_role'] for r in report['positions'])),
                             'fixed_change_classifications': dict(Counter(c['classification'] for c in changes)),
                             'rollouts': sum(r['committed_samples_per_action'] * len(actions(from_snapshot(r['snapshot']))) for r in report['positions']),
                             'interpretation': 'Small policy-conditioned audit; sampling intervals are approximate and do not establish arena strength.'}
    except BaseException as exc:
        report['status'] = 'incomplete'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['invocations'][-1]['wall_seconds'] = time.perf_counter() - started
        atomic_json(args.out, report)
        save_readable(args.out, report)
    print(f'{report["status"]}: {args.out}', flush=True)


if __name__ == '__main__':
    main()
