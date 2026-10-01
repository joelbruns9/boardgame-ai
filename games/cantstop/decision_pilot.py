"""Matched horizon/budget pilot on stratified, ordinary baseline-play decisions.

Run ceilings independently: --budgets 32 128 512 means three arms, whose
adaptive schedules are (32,), (32,128), and (32,128,512). This is a cost and
decision-change experiment, not evidence that changed choices improve play.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .adaptive_search import AdaptiveBackend, AdaptiveConfig, atomic_json, source_identity
from .decision_compare import load_suite
from .decision_search import TurnTableBackend, actions, rng_stream
from .engine import ALL_RULESETS, GameState, Phase, apply_move, random_dice, roll, stop
from .experiment import file_sha256, identity
from .progressive_search import ProgressiveConfig
from .rollout_search import RolloutConfig
from .rust_solver import RustTurnSolver
from .snapshot import from_snapshot, snapshot


def variant_id(rules):
    return f'{rules.num_players}p{rules.columns_to_win}{"b" if rules.blocking else "n"}'


def derived_seed(seed, *labels):
    payload = json.dumps([seed, *labels], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')


def stage_of(state):
    most = max(len(state.claimed_columns(p)) for p in range(state.rules.num_players))
    return 'late' if most >= state.rules.columns_to_win - 1 else ('middle' if most else 'early')


def bucket_of(state):
    return stage_of(state), 'dice' if state.phase == Phase.AWAIT_MOVE else 'stop_roll'


def collect_positions(evaluator, *, seed, per_bucket=1, games_per_variant=2,
                      variants=ALL_RULESETS, max_turns=400, solver_factory=RustTurnSolver):
    """Reservoir sample each stage/phase bucket from complete baseline games.

    Stages use saved claims: early=no claims; late=a player needs one more;
    middle=everything between. Forced/winning/terminal choices are excluded.
    Sampling has its own stream and never changes the baseline game dice.
    """
    if min(per_bucket, games_per_variant, max_turns) < 1:
        raise ValueError('collection counts must be positive')
    started = time.perf_counter()
    reservoirs, seen, games = defaultdict(list), Counter(), []
    selection_rng = rng_stream(derived_seed(seed, 'position-selection'), 'search')
    unique = set()
    for rules in variants:
        name = variant_id(rules)
        for game_index in range(games_per_variant):
            game_seed = derived_seed(seed, 'collection-game', name, game_index)
            game_rng = rng_stream(game_seed, 'game')
            state = GameState(rules)
            turns = decisions = 0

            def consider():
                nonlocal decisions
                decisions += 1
                if len(actions(state)) < 2:
                    return
                snap = snapshot(state)
                encoded = json.dumps(snap, separators=(',', ':'))
                if encoded in unique:
                    return
                unique.add(encoded)
                stage, phase = bucket_of(state)
                key = name, stage, phase
                seen[key] += 1
                index = selection_rng.randrange(seen[key])
                row = {'id': f'{name}_g{game_index}_t{turns}_d{decisions}',
                       'split': 'development', 'variant': name, 'stage': stage,
                       'phase': phase, 'tags': [stage, phase, 'ordinary_baseline_play'],
                       'origin': {'game_seed': game_seed, 'game_index': game_index,
                                  'turn': turns, 'decision': decisions}, 'snapshot': snap}
                if len(reservoirs[key]) < per_bucket:
                    reservoirs[key].append(row)
                elif index < per_bucket:
                    reservoirs[key][index] = row

            while not state.game_over:
                if turns >= max_turns:
                    raise RuntimeError(f'{name} collection game {game_index} exceeded {max_turns} turns')
                turns += 1
                if not roll(state, random_dice(game_rng)):
                    continue
                solver = solver_factory(state, evaluator)
                while True:
                    consider()
                    apply_move(state, solver.choose_move(state))
                    consider()
                    if solver.should_stop(state):
                        stop(state)
                        break
                    if not roll(state, random_dice(game_rng)):
                        break
            games.append({'variant': name, 'game_index': game_index, 'seed': game_seed,
                          'turns': turns, 'decisions': decisions, 'winner': state.winner})
            print(f'collected {name} game {game_index + 1}/{games_per_variant}', flush=True)

    # Interleave variants and rotate buckets so a short timing prefix is diverse.
    buckets = [(s, p) for s in ('early', 'middle', 'late') for p in ('dice', 'stop_roll')]
    rows = []
    for sample_index in range(per_bucket):
        for slot in range(len(buckets)):
            for variant_index, rules in enumerate(variants):
                stage, phase = buckets[(slot + variant_index) % len(buckets)]
                pool = reservoirs[variant_id(rules), stage, phase]
                if sample_index < len(pool):
                    rows.append(pool[sample_index])
    coverage = [{'variant': variant_id(r), 'stage': s, 'phase': p,
                 'eligible_unique_seen': seen[variant_id(r), s, p],
                 'selected': len(reservoirs[variant_id(r), s, p]), 'requested': per_bucket}
                for r in variants for s, p in buckets]
    return {'format': 'cantstop-search-decisions-v1',
            'purpose': 'stratified decision pilot; not a representative strength benchmark',
            'collection': {'policy': 'unmodified baseline turn solver', 'seed': seed,
                           'games_per_variant': games_per_variant, 'per_bucket': per_bucket,
                           'max_turns': max_turns, 'stages': 'early: no claims; late: one claim from winning; middle: between',
                           'elapsed_seconds': time.perf_counter() - started,
                           'games': games, 'coverage': coverage,
                           'shortfall': sum(c['requested'] - c['selected'] for c in coverage)},
            'positions': rows}


def summarize(report):
    grouped = defaultdict(list)
    for cell in report['cells']:
        grouped[cell['horizon'], cell['budget']].append(cell)

    def stats(cells):
        times = [c['adaptive']['elapsed_seconds'] for c in cells]
        unresolved = sum(c['adaptive']['fallback'] for c in cells)
        changed = sum(c['changed'] for c in cells)
        return {'decisions': len(cells), 'changed_vs_baseline': changed,
                'changed_fraction': changed / len(cells),
                'resolved': len(cells) - unresolved,
                'unresolved': unresolved, 'unresolved_fraction': unresolved / len(cells),
                'decision_seconds_median': float(np.median(times)),
                'decision_seconds_p95': float(np.percentile(times, 95)),
                'decision_seconds_total': sum(times),
                'committed_rollouts': sum(sum(c['adaptive']['committed_samples'].values()) for c in cells),
                'nn_rows': sum(c['nn_rows'] for c in cells),
                'stop_reasons': dict(Counter(c['adaptive']['stop_reason'] for c in cells))}

    arms = []
    for (horizon, budget), cells in sorted(grouped.items()):
        strata = defaultdict(list)
        for c in cells:
            strata[c['variant'], c['stage'], c['phase']].append(c)
        arms.append({'horizon': horizon, 'budget': budget, **stats(cells),
                     'strata': [{'variant': v, 'stage': s, 'phase': p, **stats(cs)}
                                for (v, s, p), cs in sorted(strata.items())]})
    comparisons = []
    by_key = {(c['id'], c['horizon'], c['budget']): c for c in report['cells']}
    horizons = report['configuration']['horizons']
    for budget in report['configuration']['budgets']:
        for i, h1 in enumerate(horizons):
            for h2 in horizons[i + 1:]:
                pairs = [(by_key[id_, h1, budget], by_key[id_, h2, budget])
                         for id_ in report['position_ids']
                         if (id_, h1, budget) in by_key and (id_, h2, budget) in by_key]
                comparisons.append({'budget': budget, 'horizons': [h1, h2], 'paired_positions': len(pairs),
                                    'different_choices': sum(a['decision']['selected'] != b['decision']['selected'] for a, b in pairs),
                                    'different_choice_ids': [a['id'] for a, b in pairs if a['decision']['selected'] != b['decision']['selected']]})
    budget_comparisons = []
    budgets = report['configuration']['budgets']
    for horizon in horizons:
        for b1, b2 in zip(budgets, budgets[1:]):
            pairs = [(by_key[id_, horizon, b1], by_key[id_, horizon, b2])
                     for id_ in report['position_ids']
                     if (id_, horizon, b1) in by_key and (id_, horizon, b2) in by_key]
            changed_ids = [a['id'] for a, b in pairs if a['decision']['selected'] != b['decision']['selected']]
            budget_comparisons.append({'horizon': horizon, 'budgets': [b1, b2],
                                      'paired_positions': len(pairs), 'different_choices': len(changed_ids),
                                      'different_choice_ids': changed_ids})
    return {'arms': arms, 'horizon_comparisons': comparisons, 'budget_comparisons': budget_comparisons,
            'changed_position_ids': sorted({c['id'] for c in report['cells'] if c['changed']}),
            'completed_cells': len(report['cells']),
            'expected_cells': len(report['position_ids']) * len(horizons) * len(report['configuration']['budgets']),
            'baseline_seconds': sum(r['elapsed_seconds'] for r in report['baselines'].values()),
            'search_seconds': sum(a['decision_seconds_total'] for a in arms),
            'interpretation': 'Changes and approximate confidence do not establish improvement; independent audits come next.'}


def run_cells(report, rows, evaluator, *, out, state_dir, evaluator_key,
              resume=False, max_cells=None, backend_factory=AdaptiveBackend):
    cfg = report['configuration']
    completed = {(c['id'], c['horizon'], c['budget']) for c in report['cells']}
    settings = [(h, b) for h in cfg['horizons'] for b in cfg['budgets']]
    new_cells = 0
    try:
        for row_index, row in enumerate(rows):
            state = from_snapshot(row['snapshot'])
            if row['id'] not in report['baselines']:
                started = time.perf_counter()
                decision = TurnTableBackend(evaluator).evaluate(state).to_dict()
                report['baselines'][row['id']] = {'decision': decision, 'elapsed_seconds': time.perf_counter() - started}
            # Rotate arm order to spread warmup/load effects across arms.
            offset = row_index % len(settings)
            for horizon, budget in settings[offset:] + settings[:offset]:
                key = row['id'], horizon, budget
                if key in completed:
                    continue
                if max_cells is not None and new_cells >= max_cells:
                    report['status'] = 'partial'
                    return
                schedule = tuple(n for n in cfg['budgets'] if n <= budget)
                adaptive = AdaptiveConfig(budgets=schedule, confidence=cfg['confidence'], alpha=cfg['alpha'])
                rollout = RolloutConfig(horizon=horizon, seed=derived_seed(cfg['seed'], 'pilot-position', row['id']),
                                        dice_luck=cfg['dice_luck'], common_random_numbers=cfg['common_random_numbers'])
                continuation = ProgressiveConfig(early_turns=cfg['early_turns'], expansions=cfg['early_expansions'], depth=1)
                backend = backend_factory(evaluator, adaptive, rollout, continuation, evaluator_key=evaluator_key)
                token = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
                checkpoint = state_dir / (token + '.json')
                before = getattr(evaluator, 'rows', 0)
                report['current_cell'] = {'id': row['id'], 'horizon': horizon, 'budget': budget, 'checkpoint': str(checkpoint)}
                atomic_json(out, report)
                resuming_checkpoint = resume and checkpoint.exists()
                decision = backend.evaluate(state, checkpoint=checkpoint,
                                            resume=resuming_checkpoint, seconds=cfg['seconds']).to_dict()
                stage, phase = bucket_of(state)
                cell = {'id': row['id'], 'variant': variant_id(state.rules), 'stage': stage, 'phase': phase,
                        'horizon': horizon, 'budget': budget, 'schedule': list(schedule), 'seed': rollout.seed,
                        'decision': decision, 'changed': decision['selected'] != report['baselines'][row['id']]['decision']['selected'],
                        'adaptive': backend.last_stats, 'nn_rows': getattr(evaluator, 'rows', 0) - before,
                        'checkpoint': str(checkpoint), 'resumed_checkpoint': resuming_checkpoint,
                        'timing_scope': 'this invocation; earlier checkpoint work excluded' if resuming_checkpoint else 'fresh decision'}
                report['cells'].append(cell)
                completed.add(key)
                new_cells += 1
                report.pop('current_cell', None)
                report['summary'] = summarize(report)
                atomic_json(out, report)
                print(f'{len(completed)}/{report["summary"]["expected_cells"]} {row["id"]} H={horizon} budget={budget}: '
                      f'{decision["selected"]}, {backend.last_stats["stop_reason"]}, '
                      f'{backend.last_stats["elapsed_seconds"]:.2f}s', flush=True)
        report['status'] = 'complete'
    except BaseException as exc:
        report['status'] = 'incomplete'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['summary'] = summarize(report)
        atomic_json(out, report)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--budgets', type=int, nargs='+', default=[32, 128, 512])
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 2])
    p.add_argument('--positions-per-bucket', type=int, default=1)
    p.add_argument('--games-per-variant', type=int, default=2)
    p.add_argument('--variants', nargs='+', choices=[variant_id(r) for r in ALL_RULESETS])
    p.add_argument('--max-turns', type=int, default=400)
    p.add_argument('--suite', type=Path, help='Reuse saved development positions instead of collecting')
    p.add_argument('--position', action='append', help='Restrict saved suite to named IDs (repeatable)')
    p.add_argument('--confidence', choices=('paired_t', 'hoeffding'), default='paired_t')
    p.add_argument('--alpha', type=float, default=.05)
    p.add_argument('--dice-luck', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--common-random-numbers', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--early-turns', type=int, default=0)
    p.add_argument('--early-expansions', type=int, default=0)
    p.add_argument('--seconds', type=float, help='Optional soft time limit per cell; omit for fixed-work timing')
    p.add_argument('--seed', type=int, default=20260930)
    p.add_argument('--collect-only', action='store_true')
    p.add_argument('--max-cells', type=int, help='Stop after this many new cells, for timing; continue with --resume')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    if args.out.exists() != args.resume:
        p.error('use a new output, or --resume with an existing output')
    if args.position and args.suite is None:
        p.error('--position requires --suite')
    if args.suite and args.variants:
        p.error('--variants controls collection; use --position with --suite')
    try:
        AdaptiveConfig(budgets=tuple(args.budgets), confidence=args.confidence, alpha=args.alpha)
        if min(args.positions_per_bucket, args.games_per_variant, args.max_turns) < 1:
            raise ValueError('collection counts must be positive')
        if args.max_cells is not None and args.max_cells < 1:
            raise ValueError('max-cells must be positive')
        if not args.horizons or len(set(args.horizons)) != len(args.horizons) or any(h < 1 for h in args.horizons):
            raise ValueError('horizons must be distinct positive integers')
        if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds < 0):
            raise ValueError('seconds must be finite and nonnegative')
        ProgressiveConfig(early_turns=args.early_turns, expansions=args.early_expansions, depth=1)
    except ValueError as exc:
        p.error(str(exc))
    variants = tuple(r for r in ALL_RULESETS if args.variants is None or variant_id(r) in args.variants)
    cfg = {k: getattr(args, k) for k in ('budgets', 'horizons', 'confidence', 'alpha', 'dice_luck',
                                      'common_random_numbers', 'early_turns', 'early_expansions', 'seconds', 'seed')}
    cfg['collection'] = {'positions_per_bucket': args.positions_per_bucket, 'games_per_variant': args.games_per_variant,
                         'max_turns': args.max_turns, 'variants': [variant_id(r) for r in variants]}
    cfg['position_filter'] = sorted(set(args.position or []))
    cfg['suite'] = str(args.suite.resolve()) if args.suite else None
    cfg['device'] = args.device
    sources = source_identity()
    sources['decision_pilot.py'] = file_sha256(__file__)
    model_key = file_sha256(args.checkpoint)
    signature_data = {'configuration': cfg, 'model': model_key, 'source': sources}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    suite_path = args.suite or args.out.with_suffix('.positions.json')
    report = json.loads(args.out.read_text()) if args.resume else None
    if report is not None and report['signature'] != signature:
        p.error('resume configuration, checkpoint, or code differs')
    if args.resume and file_sha256(suite_path) != report['suite_sha256']:
        p.error('resume suite content differs')
    if not args.resume and not args.suite and suite_path.exists():
        p.error('saved collection already exists; choose a new output')

    from .model import NetEvaluator, load_net
    started = time.perf_counter()
    evaluator = NetEvaluator(load_net(args.checkpoint, device=args.device), device=args.device)
    evaluator([GameState(variants[0])])  # Warm up before measured decision work.
    if not args.suite and not args.resume:
        suite = collect_positions(evaluator, seed=args.seed, per_bucket=args.positions_per_bucket,
                                  games_per_variant=args.games_per_variant, variants=variants, max_turns=args.max_turns)
        suite['meta'] = identity(nets={'shared': args.checkpoint}, source_sha256=sources, collection=cfg['collection'])
        atomic_json(suite_path, suite)
        if suite['collection']['shortfall']:
            print(f'collection shortfall: {suite["collection"]["shortfall"]} requested positions missing; '
                  'see suite coverage or collect more games', flush=True)
    rows = load_suite(suite_path)
    if args.position:
        selected = set(args.position)
        if selected - {r['id'] for r in rows}:
            p.error('unknown development position')
        rows = [r for r in rows if r['id'] in selected]
    # Even user-provided correctness suites must not inflate resolution rates.
    rows = [r for r in rows if len(actions(from_snapshot(r['snapshot']))) >= 2]
    if not rows:
        p.error('no nontrivial development decisions')
    if report is None:
        report = {'format': 'cantstop-decision-pilot-v1', 'status': 'running', 'signature': signature,
                  'configuration': cfg, 'suite': str(suite_path), 'suite_sha256': file_sha256(suite_path),
                  'position_ids': [r['id'] for r in rows], 'baselines': {}, 'cells': [],
                  'meta': identity(nets={'shared': args.checkpoint}, source_sha256=sources), 'invocations': []}
    report.pop('error', None)
    report['status'] = 'positions_ready' if args.collect_only else 'running'
    report['invocations'].append({'resume': args.resume, 'max_cells': args.max_cells,
                                  'collect_only': args.collect_only})
    atomic_json(args.out, report)
    try:
        if not args.collect_only:
            run_cells(report, rows, evaluator, out=args.out, state_dir=args.out.with_suffix('.states'),
                      evaluator_key=model_key, resume=args.resume, max_cells=args.max_cells)
    finally:
        report['invocations'][-1]['wall_seconds'] = time.perf_counter() - started
        report['summary'] = summarize(report)
        atomic_json(args.out, report)
    print(f'{report["status"]}: {len(rows)} positions, {len(report["cells"])} cells; report: {args.out}', flush=True)


if __name__ == '__main__':
    main()
