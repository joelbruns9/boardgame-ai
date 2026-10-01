import json

import numpy as np
import pytest

from games.cantstop import decision_headroom as h
from games.cantstop.adaptive_search import atomic_json
from games.cantstop.engine import ALL_RULESETS, Phase
from games.cantstop.snapshot import from_snapshot, snapshot
from games.cantstop.tests.test_rollout_search import BankPolicy, one_roll


def samples(first=.04, second=-.02, n=8):
    base = np.full(n, .5)
    other = base + np.r_[np.full(n//2, first), np.full(n - n//2, second)]
    raw = np.tile([1., 0.], (n, 1))
    return {k: {'raw': raw.copy(), 'adjusted': np.column_stack((a, 1-a))}
            for k, a in [('stop', base), ('roll', other)]}


def test_selection_uses_only_first_half_and_keeps_negative_evaluation():
    a = h.heldout(samples(), 'stop', 0)
    assert a['selected_on_first_half'] == 'roll'
    assert a['heldout_gain'] == pytest.approx(-.02)
    b = h.heldout(samples(second=.2), 'stop', 0)
    assert b['selected_on_first_half'] == a['selected_on_first_half']
    assert b['selection_means'] == a['selection_means']


def test_baseline_wins_ties_and_future_gain_cannot_affect_selection():
    a = h.heldout(samples(first=0, second=.2), 'stop', 0)
    assert a['selected_on_first_half'] == 'stop'
    assert a['heldout_gain'] == a['heldout_mc_se'] == 0
    a = h.heldout(samples(first=-.01, second=.2), 'stop', 0)
    assert not a['changed']


def test_actor_absolute_seat_and_unclipped_values():
    data = samples(first=-.6, second=-.2)
    a = h.heldout(data, 'stop', 1)
    assert a['changed'] and a['heldout_gain'] == pytest.approx(.2)
    assert data['roll']['adjusted'][0, 0] < 0


@pytest.mark.parametrize('kind', ['raw', 'nonfinite', 'unaligned', 'seatsum', 'odd'])
def test_invalid_samples_fail(kind):
    data = samples(n=7 if kind == 'odd' else 8)
    if kind == 'raw':
        data['roll']['raw'][0] = [.5, .5]
    elif kind == 'nonfinite':
        data['roll']['adjusted'][0, 0] = float('nan')
    elif kind == 'unaligned':
        data['roll']['raw'] = data['roll']['raw'][:-1]
    elif kind == 'seatsum':
        data['roll']['adjusted'][0, 1] = .2
    with pytest.raises(ValueError):
        h.heldout(data, 'stop', 0)


def test_candidates_keep_baseline_even_when_float_reordering_occurs():
    b = {'actor': 1, 'selected': 'base', 'options': [
        {'action': 'a', 'value': [0, .6]}, {'action': 'b', 'value': [0, .5]},
        {'action': 'c', 'value': [0, .4]}, {'action': 'base', 'value': [0, .1]}]}
    assert h.candidate_keys(b) == ['base', 'a', 'b']


def test_collector_sampling_never_changes_game_dice_and_turn_starts_are_clean():
    args = dict(seed=83, games_per_variant=3, variants=ALL_RULESETS[:1], solver_factory=BankPolicy)
    a = h.collect(None, decisions=3, turn_starts=4, **args)
    b = h.collect(None, decisions=5, turn_starts=8, **args)
    assert a['collection']['games'] == b['collection']['games']
    assert len(a['positions']) == 3 and len(b['turn_starts']) == 8
    for row in b['turn_starts']:
        state = from_snapshot(row['snapshot'])
        assert state.phase == Phase.AWAIT_ROLL and not state.runners and not state.game_over
    assert all(len(h.actions(from_snapshot(r['snapshot']))) >= 2 for r in a['positions'])


def test_summary_equal_variant_weight_and_clusters():
    def row(v, g, gain):
        return {'variant': v, 'origin': {'game_seed': g}, 'result': {
            'changed': True, 'heldout_gain': gain, 'heldout_raw_gain': gain, 'heldout_mc_se': .01}}
    rs = [row('a', 1, .2), row('a', 1, .2), row('b', 2, -.1)]
    s = h.summarize(rs, seed=33, bootstrap=100)
    assert s['mean_heldout_gain'] == pytest.approx(.05)
    assert s['conditional_mc_se'] == pytest.approx(np.sqrt(.00005 + .0001)/2)
    assert s['bootstrap_degenerate_variants'] == ['a', 'b']


def test_resume_keeps_committed_prefix_and_failure_aborts_batch(tmp_path):
    class Eval:
        rows = 0
    class Fake:
        fail = True
        offsets = []
        def __init__(self, ev, config, **kwargs):
            self.config = config
        def evaluate(self, state, candidates):
            self.offsets.append(self.config.sample_offset)
            if self.config.sample_offset == 2 and self.fail:
                raise RuntimeError('one stalled game')
            n = self.config.samples
            raw = np.tile([1., 0.], (n, 1))
            self.last_samples = {a.key: {'raw': raw, 'adjusted': np.full((n, 2), .5)} for a in candidates}
            self.last_stats = {'actions': [{'terminal_samples': n} for a in candidates], 'elapsed_seconds': 1.}
    report = {'signature': 'fixed', 'configuration': {'samples': 4, 'batch': 2, 'threads': 0,
              'in_flight': 64, 'max_rows': 1000, 'seed': 39}, 'positions': [{
              'id': 'root', 'variant': '2p5n', 'origin': {'game_seed': 1},
              'snapshot': snapshot(one_roll()), 'rollout_seed': 9, 'candidates': ['stop', 'roll'],
              'sample_file': str(tmp_path / 'samples.json'),
              'baseline': {'selected': 'stop', 'actor': 0, 'value': [.5, .5]}}]}
    out = tmp_path / 'report.json'
    with pytest.raises(RuntimeError, match='one stalled game'):
        h.run_rollouts(report, Eval(), out, backend_factory=Fake)
    prefix = json.loads((tmp_path / 'samples.json').read_text())
    assert len(prefix['samples']['stop']['raw']) == 2
    Fake.fail = False
    h.run_rollouts(report, Eval(), out, backend_factory=Fake)
    data = json.loads((tmp_path / 'samples.json').read_text())
    assert Fake.offsets == [0, 2, 2]
    assert [b['offset'] for b in data['batches']] == [0, 2]
    assert data['samples']['stop']['raw'][:2] == prefix['samples']['stop']['raw']
    assert report['positions'][0]['result']['selected_on_first_half'] == 'stop'


def test_consistency_rejects_runner_board_before_evaluation(tmp_path):
    report = {'turn_starts': [{'id': 'bad', 'snapshot': snapshot(one_roll())}], 'consistency': {'rows': []}}
    with pytest.raises(ValueError, match='turn-start'):
        h.run_consistency(report, None, tmp_path / 'report.json')
