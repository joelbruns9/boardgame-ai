from dataclasses import replace
from itertools import product
import json

import numpy as np
import pytest

from games.cantstop.decision_search import Action
from games.cantstop.engine import Phase, roll
from games.cantstop.rollout_search import RolloutBackend, RolloutConfig, expected_roll_value, dice_luck_delta
from games.cantstop.progressive_search import ProgressiveBackend, ProgressiveConfig, EarlyTurnPolicy, audit_filtering
from games.cantstop.rust_solver import hashed_evaluator, flat_evaluator
from games.cantstop.solver import TurnSolver
from games.cantstop.lookahead import refine
from games.cantstop.tests.test_turn_search import small
from games.cantstop.tests.test_rollout_search import one_roll


def reversal():
    s=small(); s.phase=Phase.AWAIT_DECISION; s.dice=None
    s.claimed_by[6]=None; s.progress[1][10]=0; s.runners={10:4}
    return s


@pytest.mark.parametrize('strong',[False,True])
@pytest.mark.parametrize('variance',[False,True])
def test_unfiltered_final_matches_direct_reference(strong,variance):
    s=one_roll()
    cfg=ProgressiveConfig(horizons=(0,1),samples=(4,16),early_turns=int(strong),expansions=int(strong))
    rc=RolloutConfig(dice_luck=variance,common_random_numbers=variance)
    backend=ProgressiveBackend(hashed_evaluator,cfg,rc)
    result=backend.evaluate(s)
    ref=RolloutBackend(hashed_evaluator,replace(rc,samples=16,horizon=1),
                       policy_factory=EarlyTurnPolicy(hashed_evaluator,cfg))
    assert result==ref.evaluate(s)
    assert all(not stage['pruned'] for stage in backend.last_stats['stages'])
    assert backend.last_stats['status']=='complete'
    assert result==backend.evaluate(s)


def test_future_evaluation_reverses_choice_and_filter_audit_detects_loss():
    s=reversal(); rc=RolloutConfig(dice_luck=True,common_random_numbers=True)
    full=ProgressiveBackend(hashed_evaluator,ProgressiveConfig(samples=(32,256)),rc)
    result=full.evaluate(s)
    assert full.last_stats['baseline']['selected']=='stop'
    assert full.last_stats['stages'][0]['result']['selected']=='stop'
    assert result.selected==Action('roll')
    assert result.value[s.active_player]>0.25
    pruned=ProgressiveBackend(hashed_evaluator,ProgressiveConfig(samples=(32,256),max_candidates=1),rc)
    chosen=pruned.evaluate(s)
    assert len(pruned.last_stats['stages'][0]['result']['options'])==2
    assert pruned.last_stats['stages'][0]['pruned']==['roll']
    assert chosen.selected==Action('stop') and len(chosen.options)==1
    # Independent larger evaluation (still a policy-based estimate).
    reference=RolloutBackend(hashed_evaluator,replace(rc,horizon=1,samples=1024,seed=42)).evaluate(s)
    audit=audit_filtering(chosen,reference)
    assert audit['reference_winner_pruned']
    assert audit['estimated_filtering_regret']>0.08


def test_margin_is_applied_only_after_minimum_all_candidate_stage():
    b=ProgressiveBackend(hashed_evaluator,ProgressiveConfig(samples=(16,16),margin=0.01),
                         RolloutConfig(dice_luck=True))
    b.evaluate(reversal())
    assert len(b.last_stats['stages'][0]['rollout']['actions'])==2
    assert all(a['samples']==16 for a in b.last_stats['stages'][0]['rollout']['actions'])
    assert b.last_stats['stages'][0]['survivors']==['stop']
    assert b.last_stats['stages'][1]['horizon']==1


@pytest.mark.parametrize('n',[2,3,4])
def test_early_turn_search_matches_independent_python_backup(n):
    state=small(n)
    cfg=ProgressiveConfig(early_turns=1,expansions=1,depth=1)
    policy=EarlyTurnPolicy(hashed_evaluator,cfg)
    stronger=policy(state,hashed_evaluator,1)
    reference=TurnSolver(state,hashed_evaluator)
    refine(reference,state,hashed_evaluator,1,offset=False)
    np.testing.assert_allclose(stronger.value(state),reference.value(state),atol=1e-12,rtol=0)
    assert stronger.choose_move(state)==reference.choose_move(state)
    assert stronger.stats['expansions']==1
    assert stronger.stats['depth_reached']==1
    assert policy(state,hashed_evaluator,1) is stronger
    # Current remainder (0) and turns beyond configured window stay baseline.
    plain=policy(state,hashed_evaluator,0)
    assert policy(state,hashed_evaluator,2) is plain
    assert plain is not stronger
    assert policy.stats['strong_solves']==1 and policy.stats['expansions']==1


def test_stronger_table_control_variate_zero_mean_after_frontier_update():
    state=one_roll()
    solver=EarlyTurnPolicy(hashed_evaluator,ProgressiveConfig(early_turns=1,expansions=2))(state,hashed_evaluator,1)
    expected=expected_roll_value(solver,state)
    corrections=[]
    for dice in product(range(1,7),repeat=4):
        after=state.clone(); after.phase=Phase.AWAIT_ROLL
        busted=not roll(after,dice)
        corrections.append(dice_luck_delta(expected,solver,after,busted))
    np.testing.assert_allclose(np.mean(corrections,axis=0),0,atol=2e-14,rtol=0)
    np.testing.assert_array_equal(solver.bust_value,solver.root.values[0])


def test_cache_limits_config_separation_and_evaluator_identity():
    cfg=ProgressiveConfig(cache_entries=1,cache_positions=200000)
    policy=EarlyTurnPolicy(hashed_evaluator,cfg)
    state=one_roll(); first=policy(state,hashed_evaluator,0)
    assert policy(state,hashed_evaluator,0) is first
    other=state.clone(); other.active_player=1
    policy(other,hashed_evaluator,0)
    assert len(policy.cache)==1
    assert policy(state,hashed_evaluator,0) is not first
    assert policy.stats['peak_cached_entries']==1
    assert policy.stats['peak_cached_positions']<=cfg.cache_positions
    with pytest.raises(ValueError): policy(state,flat_evaluator,0)
    tiny=EarlyTurnPolicy(hashed_evaluator,replace(cfg,cache_positions=1))
    tiny(state,hashed_evaluator,0)
    assert not tiny.cache and tiny.positions==0


def test_subset_order_has_no_effect_and_invalid_subsets_fail():
    state=one_roll(); b=RolloutBackend(flat_evaluator,RolloutConfig(samples=8,horizon=0))
    assert b.evaluate(state)==b.evaluate(state,candidates=[Action('roll'),Action('stop')])
    for choices in ([],[Action('stop'),Action('stop')],[Action('bad')]):
        with pytest.raises(ValueError): b.evaluate(state,candidates=choices)


@pytest.mark.parametrize('kw',[
    {'horizons':()},{'horizons':(1,0)},{'horizons':(None,1)}, {'samples':(0,2)},
    {'margin':float('nan')},{'max_candidates':0},{'early_turns':1},
    {'expansions':1},{'cache_entries':-1},{'depth':-1},
])
def test_invalid_configuration(kw):
    with pytest.raises(ValueError): ProgressiveConfig(**kw)


def test_cli_stages_and_independent_full_game_audit(tmp_path):
    from games.cantstop.decision_compare import main
    from games.cantstop.snapshot import snapshot
    suite=tmp_path/'suite.json'
    suite.write_text(json.dumps({'format':'cantstop-search-decisions-v1','positions':[
        {'id':'small','split':'development','tags':[],'snapshot':snapshot(one_roll())}]}))
    out=tmp_path/'report.json'
    main(['--suite',str(suite),'--backend','progressive','--stage-horizons','0','1',
          '--stage-samples','4','8','--early-turns','1','--early-expansions','1',
          '--dice-luck','--common-random-numbers','--audit-samples','16','--out',str(out)])
    r=json.loads(out.read_text())
    assert r['status']=='complete'
    p=r['positions'][0]
    assert p['progressive']['policy']['strong_solves']>0
    assert p['audit_reference']['rollout']['config']['horizon'] is None
    assert p['audit_reference']['rollout']['config']['seed']!=p['progressive']['rollout']['seed']
    assert 'estimated_filtering_regret' in p['filtering_audit']


def test_progressive_full_game_last_stage_and_incomplete_failure():
    from games.cantstop.rollout_search import RolloutLimitExceeded
    cfg=ProgressiveConfig(horizons=(0,None),samples=(2,4))
    b=ProgressiveBackend(flat_evaluator,cfg,RolloutConfig(dice_luck=True))
    result=b.evaluate(one_roll())
    assert b.last_stats['stages'][-1]['horizon'] is None
    assert all(a['terminal_samples']==4 for a in b.last_stats['stages'][-1]['rollout']['actions'])
    assert result.selected in (Action('stop'),Action('roll'))
    failed=ProgressiveBackend(flat_evaluator,cfg,RolloutConfig(max_turns=1))
    with pytest.raises(RolloutLimitExceeded): failed.evaluate(one_roll())
    assert failed.last_stats['status']=='incomplete'
    assert len(failed.last_stats['stages'])==1
    assert 'finalists' not in failed.last_stats
