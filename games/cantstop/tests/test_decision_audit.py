import json
import numpy as np
import pytest

from games.cantstop.decision_audit import choose_focus, contrast, intervals, stage_result
from games.cantstop.snapshot import snapshot
from games.cantstop.tests.test_rollout_search import one_roll


def samples(n=1024):
    rng = np.random.default_rng(9)
    noise = rng.normal(0, .02, n)
    p = np.column_stack((.5 + noise, .5 - noise))
    q = np.column_stack((.52 - noise, .48 + noise))
    raw = np.column_stack((rng.integers(0, 2, n), rng.integers(0, 2, n)))
    return {'stop': {'raw': raw.tolist(), 'adjusted': p.tolist()},
            'roll': {'raw': raw.tolist(), 'adjusted': q.tolist()}}


def test_corrected_intervals_and_oriented_contrasts():
    pairs = intervals(samples(), 0, alpha=.05, total_pairs=12, looks=2)
    gain = contrast(pairs, 'roll', 'stop')
    loss = contrast(pairs, 'stop', 'roll')
    assert gain['classification'] == 'helps'
    assert gain['low'] > 0 and gain['mean'] == pytest.approx(-loss['mean'])
    assert loss['classification'] == 'hurts'
    assert gain['low'] == pytest.approx(-loss['high'])
    assert pairs[0]['raw_hoeffding']['low'] < 0 < pairs[0]['raw_hoeffding']['high']
    assert contrast(pairs, 'stop', 'stop')['classification'] == 'unchanged'


def test_multiplicity_widens_intervals_and_degenerate_variance_stays_uncertain():
    a = intervals(samples(), 0, alpha=.05, total_pairs=1, looks=1)[0]['adjusted']
    b = intervals(samples(), 0, alpha=.05, total_pairs=12, looks=2)[0]['adjusted']
    assert b['low'] < a['low'] < a['high'] < b['high']
    constant = {key: {'raw': [[0, 1]] * 32, 'adjusted': [[.5, .5]] * 32} for key in ('stop', 'roll')}
    p = intervals(constant, 0, alpha=.05, total_pairs=1, looks=2)
    assert contrast(p, 'roll', 'stop')['classification'] == 'uncertain'


@pytest.mark.parametrize('kind', ['nonterminal', 'unaligned', 'nonfinite'])
def test_invalid_full_game_samples_rejected(kind):
    data = samples(32)
    if kind == 'nonterminal':
        data['stop']['raw'][0][0] = .5
    elif kind == 'unaligned':
        data['stop']['adjusted'].pop()
    else:
        data['roll']['adjusted'][0][0] = float('nan')
    with pytest.raises(ValueError):
        intervals(data, 0, alpha=.05, total_pairs=1, looks=2)


def test_frozen_choices_define_change_assessment_and_extension():
    row = {'snapshot': snapshot(one_roll()), 'pilot_choices': [{'selected': 'roll'}, {'selected': 'stop'}]}
    baseline = {'selected': 'stop'}
    stage = stage_result(row, baseline, samples(), alpha=.05, total_pairs=12, looks=2)
    assert [c['action'] for c in stage['fixed_changes']] == ['roll']
    assert stage['fixed_changes'][0]['classification'] == 'helps'
    assert not stage['needs_extension']
    row['pilot_choices'] = [{'selected': 'stop'}]
    stage = stage_result(row, baseline, samples(), alpha=.05, total_pairs=12, looks=2)
    assert not stage['fixed_changes']
    assert stage['baseline_comparisons'][0]['classification'] == 'helps'


def test_focus_selects_changed_and_largest_budget_unresolved_only():
    pilot = {'configuration': {'budgets': [32, 128, 512]},
             'summary': {'changed_position_ids': ['change']},
             'cells': [{'id': 'change', 'horizon': 1, 'budget': 512, 'decision': {'selected': 'roll'}, 'adaptive': {'fallback': False}},
                       {'id': 'close', 'horizon': 2, 'budget': 512, 'decision': {'selected': 'stop'}, 'adaptive': {'fallback': True}},
                       {'id': 'ordinary', 'horizon': 1, 'budget': 32, 'decision': {'selected': 'stop'}, 'adaptive': {'fallback': True}}]}
    rows = [{'id': id_} for id_ in ('change', 'close', 'ordinary')]
    focus = choose_focus(pilot, rows)
    assert [(r['id'], r['audit_role']) for r in focus] == [('change', 'changed'), ('close', 'unresolved')]


@pytest.mark.parametrize('backend', ['serial', 'pool'])
def test_cli_failure_resume_preserves_completed_sample_prefix(tmp_path, monkeypatch, backend):
    from games.cantstop import decision_audit as audit, model
    from games.cantstop.adaptive_search import atomic_json
    from games.cantstop.decision_search import TurnTableBackend, actions
    from games.cantstop.experiment import file_sha256
    from games.cantstop.rust_solver import flat_evaluator

    class Flat:
        rows = 0
        def __call__(self, states):
            self.rows += len(states)
            return flat_evaluator(states)
    monkeypatch.setattr(model, 'load_net', lambda *a, **k: None)
    monkeypatch.setattr(model, 'NetEvaluator', lambda *a, **k: Flat())
    state = one_roll()
    baseline = TurnTableBackend(flat_evaluator).evaluate(state).to_dict()
    other = next(a.key for a in actions(state) if a.key != baseline['selected'])
    row = {'id': 'tiny', 'split': 'development', 'snapshot': snapshot(state),
           'audit_role': 'changed', 'pilot_choices': [{'selected': other}]}
    suite = tmp_path / 'pilot.positions.json'
    atomic_json(suite, {'format': 'cantstop-search-decisions-v1', 'positions': [row]})
    checkpoint = tmp_path / 'net.pt'; checkpoint.write_text('mock checkpoint')
    pilot = tmp_path / 'pilot.json'
    atomic_json(pilot, {'status': 'complete', 'suite': str(suite), 'suite_sha256': file_sha256(suite),
                        'meta': {'nets': {'shared': {'sha256': file_sha256(checkpoint)}}, 'source_sha256': {}},
                        'baselines': {'tiny': {'decision': baseline}}, 'cells': [{'seed': 1}]})
    monkeypatch.setattr(audit, 'prepare_suite', lambda *a: {'format': 'cantstop-search-decisions-v1', 'positions': [row]})

    class Fake:
        calls = []
        fail = True
        def __init__(self, evaluator, config, **kwargs):
            self.config = config
        def evaluate(self, state):
            self.calls.append(self.config.sample_offset)
            if self.config.sample_offset == 2 and self.fail:
                raise RuntimeError('interrupted test')
            x = np.array([0., 1.])
            adjusted = .5 + x * .01
            self.last_samples = {a.key: {'raw': np.column_stack((x, 1-x)),
                                         'adjusted': np.column_stack((adjusted + (x*.01 if a.key == other else 0),
                                                                      1-adjusted - (x*.01 if a.key == other else 0)))}
                                 for a in actions(state)}
            self.last_stats = {'actions': [{'terminal_samples': 2} for a in actions(state)], 'elapsed_seconds': 1.}
    monkeypatch.setattr(audit, 'RolloutBackend', Fake)
    from games.cantstop import pool_rollout
    monkeypatch.setattr(pool_rollout, 'PoolRolloutBackend', Fake)
    out = tmp_path / 'audit.json'
    args = ['--pilot', str(pilot), '--checkpoint', str(checkpoint), '--device', 'cpu', '--backend', backend,
            '--budgets', '2', '4', '--batch', '2', '--out', str(out)]
    with pytest.raises(RuntimeError, match='interrupted test'):
        audit.main(args)
    report = json.loads(out.read_text())
    assert report['status'] == 'incomplete'
    sample_file = report['positions'][0]['sample_file']
    assert all(len(x['raw']) == 2 for x in json.loads(open(sample_file).read())['samples'].values())
    Fake.fail = False
    audit.main(args + ['--resume'])
    report = json.loads(out.read_text())
    assert report['status'] == 'complete'
    assert Fake.calls == [0, 2, 2]
    assert [s['samples_per_action'] for s in report['positions'][0]['stages']] == [2, 4]
    assert all(len(x['raw']) == 4 for x in json.loads(open(sample_file).read())['samples'].values())


def test_native_compatibility_is_explicit_and_never_allows_python_changes():
    from games.cantstop.decision_audit import pilot_source_changes
    pilot = {'meta': {'source_sha256': {'native.pyd': 'old', 'engine.py': 'same'}}}
    sources = {'native.pyd': 'new', 'engine.py': 'same'}
    with pytest.raises(ValueError):
        pilot_source_changes(pilot, sources)
    assert pilot_source_changes(pilot, sources, allow_native_rebuild=True)['native.pyd']['pilot'] == 'old'
    with pytest.raises(ValueError):
        pilot_source_changes(pilot, sources | {'engine.py': 'changed'}, allow_native_rebuild=True)
    with pytest.raises(ValueError):
        pilot_source_changes(pilot, {'engine.py': 'same'}, allow_native_rebuild=True)


def test_native_recheck_rejects_changed_choices_and_values():
    import copy
    from games.cantstop.decision_audit import recheck_pilot_baselines
    from games.cantstop.decision_search import TurnTableBackend
    from games.cantstop.rust_solver import flat_evaluator
    row = {'id': 'one', 'snapshot': snapshot(one_roll())}
    decision = TurnTableBackend(flat_evaluator).evaluate(one_roll()).to_dict()
    pilot = {'baselines': {'one': {'decision': decision}}}
    result = recheck_pilot_baselines(pilot, [row], flat_evaluator)
    assert result['positions'] == 1 and result['max_absolute_value_difference'] == 0
    changed = copy.deepcopy(pilot)
    changed['baselines']['one']['decision']['selected'] = 'different'
    with pytest.raises(ValueError, match='choice'):
        recheck_pilot_baselines(changed, [row], flat_evaluator)
    changed = copy.deepcopy(pilot)
    changed['baselines']['one']['decision']['options'][0]['value'][0] += .01
    with pytest.raises(ValueError, match='values'):
        recheck_pilot_baselines(changed, [row], flat_evaluator)
