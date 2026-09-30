from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from games.cantstop.decision_search import Action, force_action
from games.cantstop.engine import Phase, roll, reflect_state
from games.cantstop.rollout_search import (
    RolloutBackend, RolloutConfig, SharedDice, expected_roll_value, dice_luck_delta,
)
from games.cantstop.rust_solver import RustTurnSolver, flat_evaluator, hashed_evaluator
from games.cantstop.tests.test_rollout_search import one_roll, stable_stats


@pytest.mark.parametrize('evaluate',[flat_evaluator,hashed_evaluator])
@pytest.mark.parametrize('mirror',[False,True])
def test_exhaustive_zero_mean_correction_even_with_inaccurate_values(evaluate,mirror):
    state=one_roll()
    if mirror: state=reflect_state(state)
    solver=RustTurnSolver(state,evaluate)
    expected=expected_roll_value(solver,state)
    deltas=[]
    for dice in product(range(1,7),repeat=4):
        after=force_action(state,Action('roll'))
        busted=not roll(after,dice)
        deltas.append(dice_luck_delta(expected,solver,after,busted))
    np.testing.assert_allclose(np.mean(deltas,axis=0),0,atol=2e-14,rtol=0)
    np.testing.assert_allclose(np.sum(deltas,axis=1),0,atol=2e-14,rtol=0)
    saved=solver.bust_value; saved[:]=99
    assert np.all(solver.bust_value<=1)


def test_shared_turn_coordinates_do_not_shift_after_different_turn_lengths():
    a,b=SharedDice(17),SharedDice(17)
    for i in range(30): a.at(0,i)
    assert [a.at(1,i) for i in range(10)]==[b.at(1,i) for i in range(10)]
    assert a.at(1,0)==a.at(1,0)
    counts=np.bincount([d for i in range(10000) for d in a.at(2,i)],minlength=7)[1:]
    np.testing.assert_allclose(counts/counts.sum(),1/6,atol=0.01,rtol=0)
    assert [a.at(1,i) for i in range(10)]!=[SharedDice(18).at(1,i) for i in range(10)]


@pytest.mark.parametrize('paired',[False,True])
def test_h0_variance_collapses_without_changing_paths(paired):
    cfg=RolloutConfig(samples=128,horizon=0,common_random_numbers=paired)
    plain=RolloutBackend(flat_evaluator,cfg); plain.evaluate(one_roll())
    corrected=RolloutBackend(flat_evaluator,replace(cfg,dice_luck=True)); corrected.evaluate(one_roll())
    assert plain.last_stats['trajectory_sha256']==corrected.last_stats['trajectory_sha256']
    for a,b in zip(plain.last_stats['actions'],corrected.last_stats['actions']):
        assert a['raw_mean']==b['raw_mean']
        np.testing.assert_allclose(b['sample_variance'],0,atol=1e-26)
        assert sum(b['mean'])==pytest.approx(1)
    assert corrected.last_stats['differences'][0]['sample_variance']<1e-26
    assert corrected.last_stats['differences'][0]['paired']==paired


def test_independent_seed_batches_keep_full_game_mean_and_do_not_clip():
    shifts=[]; outside=0
    for seed in range(24):
        cfg=RolloutConfig(samples=64,horizon=None,seed=seed,dice_luck=True)
        b=RolloutBackend(hashed_evaluator,cfg); b.evaluate(one_roll())
        shifts.append([a['correction_mean'][0] for a in b.last_stats['actions']])
        outside+=sum(a['adjusted_samples_outside_unit_interval'] for a in b.last_stats['actions'])
        for a in b.last_stats['actions']: assert sum(a['mean'])==pytest.approx(1,abs=1e-12)
    shifts=np.asarray(shifts)
    # Fixed test design, not a sequential significance test. Separate seed
    # batches provide the independent replicates for this uncertainty check.
    assert np.all(np.abs(shifts.mean(axis=0)) < 5*shifts.std(axis=0,ddof=1)/np.sqrt(len(shifts))+0.005)
    assert outside>0  # Adjusted samples were retained, not probability-clipped.


def test_reproducible_paired_statistics_and_single_sample():
    cfg=RolloutConfig(samples=64,horizon=None,dice_luck=True,common_random_numbers=True)
    a=RolloutBackend(flat_evaluator,cfg); b=RolloutBackend(flat_evaluator,cfg)
    assert a.evaluate(one_roll())==b.evaluate(one_roll())
    assert stable_stats(a.last_stats)==stable_stats(b.last_stats)
    diff=a.last_stats['differences'][0]
    assert diff['standard_error']==pytest.approx(np.sqrt(diff['sample_variance']/64))
    single=RolloutBackend(flat_evaluator,replace(cfg,samples=1)); single.evaluate(one_roll())
    assert single.last_stats['differences'][0]['sample_variance'] is None
    assert single.last_stats['differences'][0]['standard_error'] is None


@pytest.mark.parametrize('field',['dice_luck','common_random_numbers'])
def test_switch_validation(field):
    with pytest.raises(ValueError): RolloutConfig(**{field:1})

@pytest.mark.parametrize('fixture_id',['3p4n_near_claim','4p3b_near_claim'])
def test_multiplayer_absolute_seat_corrections(fixture_id):
    from games.cantstop.decision_compare import load_suite
    from games.cantstop.snapshot import from_snapshot
    from games.cantstop.engine import COLUMN_HEIGHTS
    state=from_snapshot(next(r['snapshot'] for r in load_suite() if r['id']==fixture_id))
    state.active_player=state.rules.num_players-1
    state.phase, state.dice=Phase.AWAIT_DECISION,None
    state.runners={c:COLUMN_HEIGHTS[c]-1 for c,p in state.claimed_by.items() if p is None}
    solver=RustTurnSolver(state,hashed_evaluator)
    expected=expected_roll_value(solver,state)
    corrections=[]
    for dice in product(range(1,7),repeat=4):
        after=state.clone(); after.phase=Phase.AWAIT_ROLL
        busted=not roll(after,dice)
        corrections.append(dice_luck_delta(expected,solver,after,busted))
    np.testing.assert_allclose(np.mean(corrections,axis=0),0,atol=2e-14,rtol=0)
    np.testing.assert_allclose(np.sum(corrections,axis=1),0,atol=2e-14,rtol=0)


def test_benchmark_reports_pair_variance_and_compute_cost():
    from games.cantstop.benchmark_rollout_variance import summarize, MODES
    records=[]
    for mode in MODES:
        for batch in range(2):
            records.append({'position':'test','mode':mode,'stats':{
                'elapsed_seconds':2.0,'differences':[{'a':'stop','b':'roll',
                    'mean':float(batch),'sample_variance':0.5 if mode=='plain' else 0.25}]}})
    summary=summarize(records,10)
    assert summary[0]['variance_seconds']==pytest.approx(0.1)
    assert summary[1]['variance_seconds_relative_to_plain']==pytest.approx(0.5)
    assert summary[0]['between_batch_variance']==pytest.approx(0.5)


def test_cli_requires_rollout_backend_and_records_both_switches(tmp_path):
    import json
    from games.cantstop.decision_compare import main
    from games.cantstop.snapshot import snapshot
    with pytest.raises(SystemExit):
        main(['--dice-luck','--out',str(tmp_path/'invalid.json')])
    suite=tmp_path/'suite.json'
    suite.write_text(json.dumps({'format':'cantstop-search-decisions-v1','positions':[
        {'id':'small','split':'development','tags':[],'snapshot':snapshot(one_roll())}]}))
    report=tmp_path/'report.json'
    main(['--suite',str(suite),'--backend','rollout','--dice-luck','--common-random-numbers',
          '--samples','4','--horizon','0','--out',str(report)])
    data=json.loads(report.read_text())
    assert data['meta']['search']['dice_luck'] is True
    assert data['meta']['search']['common_random_numbers'] is True
    assert data['positions'][0]['rollout']['differences'][0]['paired'] is True
