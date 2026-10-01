import json

import pytest

from games.cantstop import decision_pilot as pilot
from games.cantstop.decision_search import Decision, actions
from games.cantstop.engine import ALL_RULESETS, Phase, can_stop, legal_moves
from games.cantstop.snapshot import from_snapshot, snapshot
from games.cantstop.tests.test_rollout_search import one_roll
from games.cantstop.rust_solver import flat_evaluator


class BankingSolver:
    """Cheap legal collection policy, to exercise full engine trajectories."""
    def __init__(self, state, evaluator):
        pass

    def choose_move(self, state):
        return max(legal_moves(state, state.dice), key=lambda m: (len(m), tuple(state.position(c) for c in m), m))

    def should_stop(self, state):
        return can_stop(state)


def test_collection_reproducible_stratified_legal_and_independent_of_reservoir_size():
    kwargs = dict(seed=19, games_per_variant=1, variants=(ALL_RULESETS[0],),
                  max_turns=400, solver_factory=BankingSolver)
    a = pilot.collect_positions(None, per_bucket=1, **kwargs)
    b = pilot.collect_positions(None, per_bucket=1, **kwargs)
    more = pilot.collect_positions(None, per_bucket=2, **kwargs)
    assert a['positions'] == b['positions']
    assert a['collection']['games'] == more['collection']['games']
    assert len(a['positions']) == 6
    assert a['collection']['shortfall'] == 0
    assert len({r['id'] for r in a['positions']}) == 6
    assert {(r['stage'], r['phase']) for r in a['positions']} == {
        (s, p) for s in ('early', 'middle', 'late') for p in ('dice', 'stop_roll')}
    for row in a['positions']:
        state = from_snapshot(row['snapshot'])
        assert not state.game_over and len(actions(state)) >= 2
        assert pilot.bucket_of(state) == (row['stage'], row['phase'])
        assert row['split'] == 'development'


def test_collection_fails_on_stalled_game():
    with pytest.raises(RuntimeError, match='exceeded 1 turns'):
        pilot.collect_positions(None, seed=0, games_per_variant=1, variants=(ALL_RULESETS[0],),
                                max_turns=1, solver_factory=BankingSolver)


def make_report():
    return {'configuration': {'budgets': [32, 128], 'horizons': [1, 2], 'seed': 10,
                              'confidence': 'paired_t', 'alpha': .05, 'dice_luck': True,
                              'common_random_numbers': True, 'early_turns': 0,
                              'early_expansions': 0, 'seconds': None},
            'position_ids': ['tiny'], 'baselines': {}, 'cells': []}


class FakeBackend:
    calls = []

    def __init__(self, evaluator, config, rollout, continuation, **kwargs):
        self.config, self.rollout = config, rollout
        self.calls.append((config, rollout, continuation))

    def evaluate(self, state, **kwargs):
        choices = actions(state)
        selected = choices[self.rollout.horizon - 1]
        self.last_stats = {'fallback': self.rollout.horizon == 1,
                           'stop_reason': 'sample_budget' if self.rollout.horizon == 1 else 'resolved',
                           'elapsed_seconds': self.rollout.horizon * .5,
                           'committed_samples': {a.key: self.config.budgets[-1] for a in choices}}
        return Decision(state.active_player, (.5, .5), selected, ())


def test_sweep_independent_ceiling_schedules_matched_seeds_and_resume(tmp_path):
    report = make_report()
    row = {'id': 'tiny', 'snapshot': snapshot(one_roll())}
    out = tmp_path / 'report.json'
    FakeBackend.calls = []
    pilot.run_cells(report, [row], flat_evaluator, out=out, state_dir=tmp_path / 'states',
                    evaluator_key='mock', max_cells=2, backend_factory=FakeBackend)
    assert report['status'] == 'partial' and len(report['cells']) == 2
    pilot.run_cells(report, [row], flat_evaluator, out=out, state_dir=tmp_path / 'states',
                    evaluator_key='mock', resume=True, backend_factory=FakeBackend)
    assert report['status'] == 'complete' and len(report['cells']) == 4
    assert len(FakeBackend.calls) == 4  # Finished arms were skipped.
    assert {c[0].budgets for c in FakeBackend.calls} == {(32,), (32, 128)}
    assert len({c[1].seed for c in FakeBackend.calls}) == 1
    assert all(c[2].depth == 1 for c in FakeBackend.calls)
    assert all(c[1].dice_luck and c[1].common_random_numbers for c in FakeBackend.calls)
    assert report['summary']['expected_cells'] == 4
    assert all(c['paired_positions'] == 1 and c['different_choices'] == 1
               for c in report['summary']['horizon_comparisons'])
    assert [a['unresolved'] for a in report['summary']['arms']] == [1, 1, 0, 0]
    assert all(c['paired_positions'] == 1 and c['different_choices'] == 0
               for c in report['summary']['budget_comparisons'])
    assert json.loads(out.read_text())['summary'] == report['summary']


def test_failure_is_saved_and_not_counted_as_a_completed_cell(tmp_path):
    class Broken(FakeBackend):
        def evaluate(self, state, **kwargs):
            raise RuntimeError('inference failed')
    report = make_report()
    out = tmp_path / 'failure.json'
    with pytest.raises(RuntimeError, match='inference failed'):
        pilot.run_cells(report, [{'id': 'tiny', 'snapshot': snapshot(one_roll())}], flat_evaluator,
                        out=out, state_dir=tmp_path / 'states', evaluator_key='mock', backend_factory=Broken)
    saved = json.loads(out.read_text())
    assert saved['status'] == 'incomplete'
    assert saved['summary']['completed_cells'] == 0
    assert saved['current_cell']['id'] == 'tiny'
    assert saved['error'] == 'RuntimeError: inference failed'


def test_partial_summary_pairs_only_matching_completed_positions():
    report = make_report()
    stats = {'fallback': False, 'stop_reason': 'resolved', 'elapsed_seconds': 1,
             'committed_samples': {'stop': 32, 'roll': 32}}
    cell = {'id': 'tiny', 'horizon': 1, 'budget': 32, 'variant': '2p3n', 'stage': 'late',
            'phase': 'stop_roll', 'changed': True, 'decision': {'selected': 'stop'}, 'adaptive': stats, 'nn_rows': 1}
    report['cells'] = [cell]
    summary = pilot.summarize(report)
    assert summary['arms'][0]['changed_vs_baseline'] == 1
    assert summary['arms'][0]['strata'][0]['stage'] == 'late'
    assert summary['horizon_comparisons'][0]['paired_positions'] == 0


def test_cli_reuses_suite_excludes_trivial_positions_and_rejects_changed_resume(tmp_path, monkeypatch):
    from games.cantstop.model import CantStopNet, save_net
    model = tmp_path / 'net.pt'
    save_net(CantStopNet(hidden=(8,)), model)
    state = one_roll()
    forced = state.clone()
    forced.phase, forced.dice = Phase.AWAIT_ROLL, None
    suite = tmp_path / 'suite.json'
    pilot.atomic_json(suite, {'format': 'cantstop-search-decisions-v1', 'positions': [
        {'id': 'tiny', 'split': 'development', 'snapshot': snapshot(state)},
        {'id': 'forced', 'split': 'development', 'snapshot': snapshot(forced)},
        {'id': 'heldout', 'split': 'heldout', 'snapshot': snapshot(state)}]})
    out = tmp_path / 'out.json'
    args = ['--checkpoint', str(model), '--suite', str(suite), '--device', 'cpu', '--out', str(out), '--collect-only']
    pilot.main(args)
    saved = json.loads(out.read_text())
    assert saved['position_ids'] == ['tiny']
    assert saved['status'] == 'positions_ready'
    pilot.main(args + ['--resume'])
    with pytest.raises(SystemExit):
        pilot.main(args + ['--resume', '--seed', '7'])
    suite.write_text(suite.read_text() + '\n')
    with pytest.raises(SystemExit):
        pilot.main(args + ['--resume'])


@pytest.mark.parametrize('extra', [['--budgets', '128', '32'], ['--horizons', '1', '1'],
                                 ['--early-turns', '1'], ['--max-cells', '0']])
def test_invalid_cli_arguments_fail_before_loading_model(tmp_path, extra):
    with pytest.raises(SystemExit):
        pilot.main(['--checkpoint', 'missing.pt', '--out', str(tmp_path / 'new.json'), *extra])
