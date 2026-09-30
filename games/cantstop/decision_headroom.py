"""Fresh baseline decisions: one-turn consistency and held-out full-game gains.

This measures a candidate search with baseline continuation, not arena strength
or an upper bound on all possible search improvements. Selection and evaluation
use independent halves of a fixed shared-dice stream; negative gains are kept.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .adaptive_search import atomic_json, source_identity
from .decision_pilot import derived_seed, stage_of, variant_id
from .decision_search import TurnTableBackend, actions, rng_stream
from .engine import ALL_RULESETS, GameState, Phase, apply_move, random_dice, roll, stop
from .experiment import file_sha256, identity
from .pool_rollout import PoolRolloutBackend
from .rollout_search import RolloutConfig
from .rust_solver import RustTurnSolver
from .snapshot import from_snapshot, snapshot


def collect(evaluator, *, seed, decisions=10, turn_starts=200,
            games_per_variant=30, variants=ALL_RULESETS, solver_factory=RustTurnSolver):
    """Frequency-weighted reservoirs per variant, independent of game dice.

    Repeated board occurrences are eligible; their frequency is part of the
    population. Decision roots exclude forced actions and winning banks.
    """
    if min(decisions, turn_starts, games_per_variant) < 1:
        raise ValueError('positive collection sizes required')
    pools, counts, games = defaultdict(list), Counter(), []
    started = time.perf_counter()
    for rules in variants:
        name = variant_id(rules)
        samplers = {kind: rng_stream(derived_seed(seed, 'reservoir', name, kind), 'search')
                    for kind in ('decision', 'turn_start')}
        for gi in range(games_per_variant):
            game_seed = derived_seed(seed, 'fresh-headroom-game', name, gi)
            dice_rng = rng_stream(game_seed, 'game')
            state = GameState(rules)
            turns = visits = 0

            def consider(kind):
                nonlocal visits
                visits += 1
                if kind == 'decision' and len(actions(state)) < 2:
                    return
                key = name, kind
                counts[key] += 1
                size = decisions if kind == 'decision' else turn_starts
                index = samplers[kind].randrange(counts[key])
                row = {'id': f'{name}_g{gi}_t{turns}_v{visits}', 'variant': name,
                       'stage': stage_of(state), 'phase': 'dice' if state.phase == Phase.AWAIT_MOVE else
                       ('stop_roll' if state.phase == Phase.AWAIT_DECISION else 'turn_start'),
                       'runners': len(state.runners), 'snapshot': snapshot(state),
                       'origin': {'game_seed': game_seed, 'game_index': gi, 'turn': turns}}
                if len(pools[key]) < size:
                    pools[key].append(row)
                elif index < size:
                    pools[key][index] = row

            while not state.game_over:
                if turns >= 400:
                    raise RuntimeError('collection game exceeded 400 turns')
                if state.runners or state.phase != Phase.AWAIT_ROLL:
                    raise RuntimeError('invalid turn-start boundary')
                consider('turn_start')
                turns += 1
                if not roll(state, random_dice(dice_rng)):
                    continue
                solver = solver_factory(state, evaluator)
                while True:
                    consider('decision')
                    apply_move(state, solver.choose_move(state))
                    consider('decision')
                    if solver.should_stop(state):
                        stop(state)
                        break
                    if not roll(state, random_dice(dice_rng)):
                        break
            games.append({'variant': name, 'game_index': gi, 'seed': game_seed,
                          'turns': turns, 'winner': state.winner})
        print(f'collected {name}: {len(pools[name, "decision"])} decisions, '
              f'{len(pools[name, "turn_start"])} turn starts from {games_per_variant} games', flush=True)
    rows = [pools[variant_id(r), 'decision'][i] for i in range(decisions) for r in variants
            if i < len(pools[variant_id(r), 'decision'])]
    starts = [x for r in variants for x in pools[variant_id(r), 'turn_start']]
    coverage = [{'variant': variant_id(r), 'kind': k, 'eligible_seen': counts[variant_id(r), k],
                 'selected': len(pools[variant_id(r), k]),
                 'requested': decisions if k == 'decision' else turn_starts}
                for r in variants for k in ('decision', 'turn_start')]
    if any(c['selected'] != c['requested'] for c in coverage):
        raise RuntimeError('collection shortfall; increase games-per-variant')
    return {'format': 'cantstop-headroom-suite-v1', 'positions': rows, 'turn_starts': starts,
            'collection': {'seed': seed, 'games_per_variant': games_per_variant, 'games': games,
                           'coverage': coverage, 'elapsed_seconds': time.perf_counter() - started,
                           'population': 'uniform eligible decision/turn-start occurrences within each variant; variants equal weight'}}


def candidate_keys(baseline, limit=3):
    actor = baseline['actor']
    ranked = sorted(baseline['options'], key=lambda o: (-o['value'][actor],
                    o['action'] != baseline['selected'], o['action']))
    # Preserve the baseline even if tiny floating differences reorder it.
    keys = [baseline['selected']]
    keys.extend(o['action'] for o in ranked if o['action'] != baseline['selected'])
    return keys[:limit]


def validate_samples(samples, n=None, seats=None):
    if not samples:
        raise ValueError('empty action samples')
    shapes = set()
    for fields in samples.values():
        raw, adjusted = (np.asarray(fields[k], dtype=float) for k in ('raw', 'adjusted'))
        if raw.ndim != 2 or raw.shape != adjusted.shape or not np.isfinite(adjusted).all():
            raise ValueError('invalid or nonfinite sample arrays')
        if not np.isin(raw, (0., 1.)).all() or not np.all(raw.sum(axis=1) == 1):
            raise ValueError('raw outcomes must be terminal one-hot winners')
        if not np.allclose(adjusted.sum(axis=1), 1., atol=1e-8, rtol=0):
            raise ValueError('adjusted seat values must sum to one')
        shapes.add(raw.shape)
    if len(shapes) != 1:
        raise ValueError('unaligned samples')
    count, width = next(iter(shapes))
    if (n is not None and count != n) or (seats is not None and width != seats):
        raise ValueError('wrong sample count or seat count')
    return count


def heldout(samples, baseline, actor):
    n = validate_samples(samples)
    if n < 4 or n % 2 or baseline not in samples:
        raise ValueError('require even aligned samples and included baseline')
    half = n // 2
    means = {k: float(np.mean(v['adjusted'][:half], axis=0)[actor]) for k, v in samples.items()}
    selected = min(means, key=lambda k: (-means[k], k != baseline, k))
    gain = (np.asarray(samples[selected]['adjusted'])[half:, actor] -
            np.asarray(samples[baseline]['adjusted'])[half:, actor])
    raw = (np.asarray(samples[selected]['raw'])[half:, actor] -
           np.asarray(samples[baseline]['raw'])[half:, actor])
    return {'selected_on_first_half': selected, 'changed': selected != baseline,
            'selection_samples': half, 'evaluation_samples': half, 'selection_means': means,
            'heldout_gain': float(gain.mean()), 'heldout_raw_gain': float(raw.mean()),
            'heldout_paired_sd': float(gain.std(ddof=1)),
            'heldout_mc_se': float(gain.std(ddof=1) / np.sqrt(half)),
            'heldout_raw_mc_se': float(raw.std(ddof=1) / np.sqrt(half))}


def summarize(rows, *, seed, bootstrap=5000):
    groups = defaultdict(list)
    for row in rows:
        if 'result' in row:
            groups[row['variant']].append(row)
    if not groups:
        return {'positions': 0}
    strata = []
    for name, rs in sorted(groups.items()):
        strata.append({'variant': name, 'positions': len(rs),
                       'changed': sum(r['result']['changed'] for r in rs),
                       'mean_gain': float(np.mean([r['result']['heldout_gain'] for r in rs])),
                       'source_games': len({r['origin']['game_seed'] for r in rs})})
    v = len(groups)
    mean = float(np.mean([s['mean_gain'] for s in strata]))
    mc_se = float(np.sqrt(sum(sum(r['result']['heldout_mc_se'] ** 2 for r in rs) / len(rs) ** 2
                              for rs in groups.values())) / v)
    # Resample source-game clusters within variants. Preserve all sampled roots
    # in each cluster, weighting by their count, to respect dependent roots.
    rng = np.random.default_rng(derived_seed(seed, 'source-game-bootstrap'))
    draws = np.zeros(bootstrap)
    for rs in groups.values():
        clusters = defaultdict(list)
        for row in rs:
            clusters[row['origin']['game_seed']].append(row['result']['heldout_gain'])
        cs = list(clusters.values())
        indices = rng.integers(len(cs), size=(bootstrap, len(cs)))
        totals = np.array([sum(c) for c in cs])
        sizes = np.array([len(c) for c in cs])
        draws += totals[indices].sum(axis=1) / sizes[indices].sum(axis=1) / v
    low, high = map(float, np.quantile(draws, (.025, .975)))
    return {'positions': sum(len(rs) for rs in groups.values()), 'variants': strata,
            'changed': sum(s['changed'] for s in strata), 'mean_heldout_gain': mean,
            'mean_heldout_raw_gain': float(np.mean([np.mean([r['result']['heldout_raw_gain'] for r in rs])
                                                   for rs in groups.values()])),
            'conditional_mc_se': mc_se, 'conditional_mc_95': [mean - 1.96 * mc_se, mean + 1.96 * mc_se],
            'source_game_cluster_bootstrap_95': [low, high], 'bootstrap_replicates': bootstrap,
            'bootstrap_degenerate_variants': [s['variant'] for s in strata if s['source_games'] < 2],
            'interpretation': 'Exploratory held-out candidate-search gain with baseline continuation; not arena strength or maximum headroom.'}


def consistency_summary(rows):
    def stats(rs):
        actor = np.array([r['residual'][r['actor']] for r in rs]) * 100
        all_abs = np.abs(np.concatenate([r['residual'] for r in rs])) * 100
        return {'boards': len(rs), 'actor_mean_pp': float(actor.mean()),
                'actor_mae_pp': float(np.abs(actor).mean()),
                'actor_abs_median_pp': float(np.median(np.abs(actor))),
                'actor_abs_p90_pp': float(np.quantile(np.abs(actor), .9)),
                'actor_abs_p95_pp': float(np.quantile(np.abs(actor), .95)),
                'actor_abs_max_pp': float(np.abs(actor).max()),
                'all_seats_mae_pp': float(all_abs.mean()),
                'actor_fraction_above_0_5pp': float(np.mean(np.abs(actor) > .5)),
                'actor_fraction_above_1pp': float(np.mean(np.abs(actor) > 1))}
    if not rows:
        return {'boards': 0}
    groups = defaultdict(list)
    for row in rows:
        groups[row['variant']].append(row)
    return {**stats(rows), 'variants': [{'variant': v, **stats(rs)} for v, rs in sorted(groups.items())],
            'interpretation': 'Exact one-turn backup minus NN on no-runner turn-start boards. Consistency does not establish accuracy.'}


def run_consistency(report, evaluator, out):
    done = {r['id'] for r in report['consistency']['rows']}
    for row in report['turn_starts']:
        if row['id'] in done:
            continue
        state = from_snapshot(row['snapshot'])
        if state.runners or state.phase != Phase.AWAIT_ROLL or state.game_over:
            raise ValueError('consistency requires nonterminal turn-start board')
        started, before = time.perf_counter(), evaluator.rows
        net = np.asarray(evaluator([state])[0], dtype=float)
        backed = RustTurnSolver(state, evaluator).value(state)
        if not np.isfinite(net).all() or not np.isfinite(backed).all():
            raise ValueError('nonfinite consistency values')
        report['consistency']['rows'].append({'id': row['id'], 'variant': row['variant'],
                  'actor': state.active_player, 'net': net.tolist(), 'turn_start': backed.tolist(),
                  'residual': (backed - net).tolist(), 'seconds': time.perf_counter() - started,
                  'nn_rows': evaluator.rows - before})
        if len(report['consistency']['rows']) % 25 == 0:
            report['consistency']['summary'] = consistency_summary(report['consistency']['rows'])
            atomic_json(out, report)
            print(f'consistency {len(report["consistency"]["rows"])}/{len(report["turn_starts"])}', flush=True)
    report['consistency']['summary'] = consistency_summary(report['consistency']['rows'])
    atomic_json(out, report)


def run_rollouts(report, evaluator, out, *, backend_factory=PoolRolloutBackend):
    cfg = report['configuration']
    for index, row in enumerate(report['positions']):
        path = Path(row['sample_file'])
        data = json.loads(path.read_text()) if path.exists() else {
            'signature': report['signature'], 'id': row['id'], 'seed': row['rollout_seed'],
            'samples': {k: {'raw': [], 'adjusted': []} for k in row['candidates']}, 'batches': []}
        if (data['signature'] != report['signature'] or data['id'] != row['id'] or
                data['seed'] != row['rollout_seed'] or set(data['samples']) != set(row['candidates'])):
            raise ValueError('sample identity differs')
        n = len(next(iter(data['samples'].values()))['raw'])
        if n:
            validate_samples(data['samples'], n, len(row['baseline']['value']))
        elif any(v[f] for v in data['samples'].values() for f in ('raw', 'adjusted')):
            raise ValueError('unaligned empty prefix')
        if n > cfg['samples']:
            raise ValueError('excess sample count')
        state = from_snapshot(row['snapshot'])
        candidates = tuple(a for a in actions(state) if a.key in row['candidates'])
        while n < cfg['samples']:
            count = min(cfg['batch'], cfg['samples'] - n)
            backend = backend_factory(evaluator, RolloutConfig(samples=count, horizon=None,
                        seed=row['rollout_seed'], sample_offset=n, dice_luck=True,
                        common_random_numbers=True), retain_samples=True, threads=cfg['threads'],
                        in_flight=cfg['in_flight'], max_rows=cfg['max_rows'])
            report['current_batch'] = {'id': row['id'], 'offset': n, 'samples': count}
            atomic_json(out, report)
            before = evaluator.rows
            backend.evaluate(state, candidates=candidates)
            if set(backend.last_samples) != set(data['samples']):
                raise ValueError('backend returned wrong action subset')
            validate_samples(backend.last_samples, count, state.rules.num_players)
            if any(a['terminal_samples'] != count for a in backend.last_stats['actions']):
                raise ValueError('unfinished trajectory')
            for key in data['samples']:
                for field in ('raw', 'adjusted'):
                    data['samples'][key][field].extend(backend.last_samples[key][field].tolist())
            data['batches'].append({'offset': n, 'samples': count, 'stats': backend.last_stats,
                                    'nn_rows': evaluator.rows - before})
            n += count
            atomic_json(path, data)  # sample file commits before report; safe on resume
            row['sample_sha256'] = file_sha256(path)
            row['committed_samples_per_action'] = n
            report.pop('current_batch', None)
            atomic_json(out, report)
            print(f'{index + 1}/{len(report["positions"])} {row["id"]}: {n}/{cfg["samples"]}/action, '
                  f'{backend.last_stats["elapsed_seconds"]:.1f}s', flush=True)
        row['sample_sha256'] = file_sha256(path)
        row['committed_samples_per_action'] = n
        row['result'] = heldout(data['samples'], row['baseline']['selected'], row['baseline']['actor'])
        report['summary'] = summarize(report['positions'], seed=cfg['seed'])
        atomic_json(out, report)


def readable(out, report):
    s, c = report.get('summary', {}), report['consistency'].get('summary', {})
    lines = ['# Baseline headroom diagnostic', '', f'Status: {report["status"]}.', '',
             'Fresh ordinary baseline decisions, equal weight per rule variant. NN guides the continuation; '
             'full-game payoffs use actual terminal winners. Shared simulated dice and dice-luck correction are enabled.', '',
             'Actions are chosen on the first half of samples and scored on the independent second half. '
             'Baseline is selectable, exact ties retain it, and negative held-out gains remain in the mean.', '']
    if c.get('boards'):
        lines += [f'One-turn consistency: {c["boards"]} turn-start boards, all without runners. '
                  f'Actor mean residual {c["actor_mean_pp"]:+.3f} pp; mean absolute residual {c["actor_mae_pp"]:.3f} pp; '
                  f'median {c["actor_abs_median_pp"]:.3f}, p95 {c["actor_abs_p95_pp"]:.3f}, max {c["actor_abs_max_pp"]:.3f} pp.', '',
                  '| Variant | Boards | Actor MAE (pp) | p95 abs (pp) |', '|---|---:|---:|---:|']
        lines += [f'| {v["variant"]} | {v["boards"]} | {v["actor_mae_pp"]:.3f} | {v["actor_abs_p95_pp"]:.3f} |' for v in c['variants']]
        lines += ['', 'Consistency does not establish value accuracy or rule out deeper-search improvements.', '']
    if s.get('positions'):
        lo, hi = s['source_game_cluster_bootstrap_95']
        lines += [f'Held-out search: {s["positions"]} positions, {s["changed"]} alternatives selected. '
                  f'Mean gain {100*s["mean_heldout_gain"]:+.3f} pp; exploratory source-game cluster bootstrap '
                  f'95% interval [{100*lo:+.3f}, {100*hi:+.3f}] pp. '
                  f'Conditional Monte Carlo SE {100*s["conditional_mc_se"]:.3f} pp.', '',
                  f'Raw held-out mean gain for the same selected actions: {100*s["mean_heldout_raw_gain"]:+.3f} pp.', '',
                  '| Variant | Decisions | Changes | Mean gain (pp) | Source games |', '|---|---:|---:|---:|---:|']
        lines += [f'| {v["variant"]} | {v["positions"]} | {v["changed"]} | {100*v["mean_gain"]:+.3f} | {v["source_games"]} |'
                  for v in s['variants']]
        lines += ['', 'The bootstrap is exploratory and can be unstable with few source games per variant. '
                  'The conditional Monte Carlo interval treats the sampled roots and selected actions as fixed. '
                  'Neither is an arena win-rate interval.', '',
                  'Only the baseline and its highest-valued rivals (at most three total actions) were tested. '
                  'Finite selection samples can miss a better candidate, and other legal actions can be excluded. '
                  'This estimates this candidate-search procedure with baseline continuation; it is not the maximum '
                  'headroom of root search or proof that training is the only remaining lever.', '']
    out.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=2026100201)
    p.add_argument('--decisions-per-variant', type=int, default=10)
    p.add_argument('--turn-starts-per-variant', type=int, default=200)
    p.add_argument('--games-per-variant', type=int, default=30)
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--in-flight', type=int, default=64)
    p.add_argument('--max-rows', type=int, default=1_000_000)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    if min(args.decisions_per_variant, args.turn_starts_per_variant, args.games_per_variant,
           args.batch, args.in_flight, args.max_rows) < 1 or args.samples < 4 or args.samples % 2 or args.threads < 0:
        p.error('positive sizes, even samples >= 4, and nonnegative threads required')
    if args.out.exists() != args.resume:
        p.error('choose a new output or resume an existing output')
    sources = source_identity()
    for name in ('decision_headroom.py', 'decision_pilot.py', 'pool_rollout.py'):
        sources[name] = file_sha256(Path(__file__).with_name(name))
    cfg = {k: v for k, v in vars(args).items() if k not in ('out', 'resume', 'checkpoint')}
    cfg.update(horizon='full_game', continuation='baseline', common_random_numbers=True,
               dice_luck=True, candidate_limit=3, selection='first half; baseline wins exact ties')
    signature = hashlib.sha256(json.dumps({'configuration': cfg, 'source': sources,
                           'checkpoint': file_sha256(args.checkpoint)}, sort_keys=True).encode()).hexdigest()
    report = json.loads(args.out.read_text()) if args.resume else None
    if report and report['signature'] != signature:
        p.error('resume identity differs')
    from .model import NetEvaluator, load_net
    evaluator = NetEvaluator(load_net(args.checkpoint, device=args.device), device=args.device)
    evaluator([GameState(ALL_RULESETS[0])])
    if report is None:
        suite = collect(evaluator, seed=args.seed, decisions=args.decisions_per_variant,
                        turn_starts=args.turn_starts_per_variant, games_per_variant=args.games_per_variant)
        suite_path = args.out.with_suffix('.positions.json')
        if suite_path.exists():
            p.error('collection file already exists; choose new output')
        atomic_json(suite_path, suite)
        report = {'format': 'cantstop-headroom-v1', 'status': 'running', 'signature': signature,
                  'configuration': cfg, 'suite': str(suite_path), 'suite_sha256': file_sha256(suite_path),
                  'meta': identity(nets={'shared': args.checkpoint}, source_sha256=sources),
                  'collection': suite['collection'], 'turn_starts': suite['turn_starts'],
                  'positions': [], 'consistency': {'rows': []}, 'invocations': []}
        # Freeze all candidate sets before any full-game sample is generated.
        for row in suite['positions']:
            baseline = TurnTableBackend(evaluator).evaluate(from_snapshot(row['snapshot'])).to_dict()
            token = hashlib.sha256(row['id'].encode()).hexdigest()[:20]
            report['positions'].append({**row, 'baseline': baseline, 'candidates': candidate_keys(baseline),
                 'legal_actions': len(baseline['options']), 'rollout_seed': derived_seed(args.seed, 'headroom-full-game', row['id']),
                 'sample_file': str(args.out.with_suffix('.samples') / (token + '.json'))})
    elif file_sha256(report['suite']) != report['suite_sha256']:
        p.error('suite changed')
    report['status'] = 'running'
    report.pop('error', None)
    started = time.perf_counter()
    report['invocations'].append({'resume': args.resume})
    atomic_json(args.out, report)
    try:
        run_consistency(report, evaluator, args.out)
        readable(args.out, report)
        run_rollouts(report, evaluator, args.out)
        report['status'] = 'complete'
        report['summary']['full_games'] = sum(r['committed_samples_per_action'] * len(r['candidates']) for r in report['positions'])
    except BaseException as exc:
        report['status'] = 'incomplete'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['invocations'][-1]['wall_seconds'] = time.perf_counter() - started
        atomic_json(args.out, report)
        readable(args.out, report)
    print(f'{report["status"]}: {args.out}', flush=True)


if __name__ == '__main__':
    main()
