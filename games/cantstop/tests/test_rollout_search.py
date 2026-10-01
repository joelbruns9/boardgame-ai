import copy
from itertools import product
import json

import numpy as np
import pytest

from games.cantstop.decision_search import Action, TurnTableBackend, force_action, rng_stream
from games.cantstop.decision_compare import main
from games.cantstop.engine import (GameState, RuleSet, Phase, COLUMN_HEIGHTS,
    legal_moves, apply_move, can_stop, roll, stop)
from games.cantstop.rollout_search import RolloutBackend, RolloutConfig, RolloutLimitExceeded, search_dice
from games.cantstop.rust_solver import flat_evaluator
from games.cantstop.snapshot import snapshot


def one_roll():
    s=GameState(RuleSet(2,5,False))
    for c in range(2,6): s.claimed_by[c]=0
    for c in range(6,10): s.claimed_by[c]=1
    for c in (10,11,12):
        s.progress[0][c]=COLUMN_HEIGHTS[c]-2
        s.progress[1][c]=COLUMN_HEIGHTS[c]-1
    s.runners={c:COLUMN_HEIGHTS[c]-1 for c in (10,11,12)}
    s.phase=Phase.AWAIT_DECISION
    return s


class BankPolicy:
    """Cheap known policy for testing horizon mechanics independently."""
    def __init__(self,state,evaluator): pass
    def should_stop(self,state): return can_stop(state)
    def choose_move(self,state): return legal_moves(state,state.dice)[0]


def flat(states):
    return np.full((len(states),states[0].rules.num_players),1/states[0].rules.num_players)


def stable_stats(stats):
    result=copy.deepcopy(stats)
    result.pop('elapsed_seconds'); result.pop('endpoint_seconds')
    for a in result['actions']: a.pop('generation_seconds',None)
    return result


def test_h0_matches_exhaustive_dice_reference_and_baseline():
    s=one_roll(); before=snapshot(s)
    # Every legal roll reaches a winning top; otherwise the turn busts.
    exact=0
    for dice in product(range(1,7),repeat=4):
        child=force_action(s,Action('roll'))
        moves=roll(child,dice)
        if moves:
            for move in moves:
                c=child.clone(); apply_move(c,move); stop(c)
                assert c.game_over and c.winner==0
            exact+=1/1296
        else: exact+=0.5/1296
    baseline=TurnTableBackend(flat_evaluator).evaluate(s)
    expected={o.action.key:o.value[0] for o in baseline.options}
    assert abs(expected['roll']-exact)<1e-12
    backend=RolloutBackend(flat_evaluator,RolloutConfig(samples=4096,horizon=0))
    result=backend.evaluate(s)
    observed={o.action.key:o.value[0] for o in result.options}
    assert observed['stop']==0.5
    # Predeclared fixed-seed MC tolerance: >5 worst-case standard errors.
    assert abs(observed['roll']-exact)<0.022
    assert all(a['samples']==4096 for a in backend.last_stats['actions'])
    assert all(a['completed_turns']==4096 for a in backend.last_stats['actions'])
    assert snapshot(s)==before
    again=RolloutBackend(flat_evaluator,backend.config)
    assert result==again.evaluate(s)
    assert stable_stats(backend.last_stats)==stable_stats(again.last_stats)


@pytest.mark.parametrize('n',[2,3,4])
@pytest.mark.parametrize('h',[0,1,2,4])
@pytest.mark.parametrize('phase',[Phase.AWAIT_ROLL,Phase.AWAIT_MOVE,Phase.AWAIT_DECISION])
def test_horizon_and_endpoint_batch_perspectives(n,h,phase):
    s=GameState(RuleSet.make(n,extended=True))
    s.active_player=n-1
    if phase!=Phase.AWAIT_ROLL:
        roll(s,(1,2,3,4))
        if phase==Phase.AWAIT_DECISION: apply_move(s,legal_moves(s,s.dice)[0])
    calls=[]
    def endpoint(boards):
        assert all(b.phase==Phase.AWAIT_ROLL and not b.runners and not b.game_over for b in boards)
        assert len({(b.rules,b.active_player) for b in boards})==1
        assert all(b.active_player==(n-1+h+1)%n for b in boards)
        calls.append(len(boards))
        return flat(boards)
    b=RolloutBackend(endpoint,RolloutConfig(samples=3,horizon=h,endpoint_batch_size=4),solver_factory=BankPolicy)
    b.evaluate(s)
    assert sum(calls)==3*len(b.last_stats['actions'])
    assert max(calls)<=4
    for a in b.last_stats['actions']:
        assert a['completed_turns']==3*(h+1)
        assert a['terminal_samples']==0
        np.testing.assert_allclose(a['sample_variance'],0,atol=1e-30)


def test_full_game_mode_uses_actual_winners_not_endpoint_evaluator():
    s=one_roll()
    calls=[]
    def forbidden(boards): calls.append(boards); raise AssertionError('NN cutoff in full game')
    b=RolloutBackend(forbidden,RolloutConfig(samples=32,horizon=None),solver_factory=BankPolicy)
    result=b.evaluate(s)
    assert calls==[]
    assert all(a['terminal_samples']==32 for a in b.last_stats['actions'])
    assert all(sum(o.value)==1 for o in result.options)


def test_early_terminal_and_game_over_do_not_call_evaluator():
    s=one_roll(); s.runners[12]=COLUMN_HEIGHTS[12]
    def forbidden(_): raise AssertionError('terminal evaluator called')
    b=RolloutBackend(forbidden,RolloutConfig(samples=4,horizon=4))
    r=b.evaluate(s)
    assert r.selected==Action('stop') and r.value==(1.,0.)
    assert b.last_stats['actions'][0]['completed_turns']==4
    stop(s)
    assert b.evaluate(s).value==(1.,0.)
    assert b.last_stats['actions']==[]


def test_safety_limit_failure_never_silently_drops_samples():
    b=RolloutBackend(flat,RolloutConfig(samples=4,horizon=None,max_turns=1),solver_factory=BankPolicy)
    with pytest.raises(RolloutLimitExceeded): b.evaluate(one_roll())
    assert b.last_stats['status']=='incomplete'
    assert b.last_stats['actions'][0]['samples']==0
    class NeverBank(BankPolicy):
        def should_stop(self,s): return False
    s=GameState(RuleSet.make(2))
    b=RolloutBackend(flat,RolloutConfig(samples=2,horizon=3,max_rolls=1),solver_factory=NeverBank)
    with pytest.raises(RolloutLimitExceeded): b.evaluate(s)
    assert b.last_stats['status']=='incomplete'


def test_real_solver_full_game_and_sample_statistics():
    b=RolloutBackend(flat_evaluator,RolloutConfig(samples=32,horizon=None))
    b.evaluate(one_roll())
    assert all(a['terminal_samples']==32 for a in b.last_stats['actions'])
    for a in b.last_stats['actions']:
        p=a['mean'][0]
        assert a['sample_variance'][0]==pytest.approx(p*(1-p)*32/31)
        assert a['standard_error'][0]==pytest.approx((p*(1-p)/31)**0.5)


def test_search_dice_rejects_modulo_bias_and_rng_isolation():
    class Script:
        def __init__(self): self.values=iter([(1<<64)-1,0,1,2,3])
        def next_u64(self): return next(self.values)
    assert search_dice(Script())==(1,2,3,4)
    game=rng_stream(9,'game'); control=game.clone()
    RolloutBackend(flat_evaluator,RolloutConfig(samples=8,horizon=0)).evaluate(one_roll())
    assert [game.next_u64() for _ in range(20)]==[control.next_u64() for _ in range(20)]


@pytest.mark.parametrize('settings',[{'samples':0},{'samples':True},{'horizon':-1},
    {'seed':1.5},{'max_turns':0},{'max_rolls':0},{'endpoint_batch_size':0}])
def test_invalid_config(settings):
    with pytest.raises(ValueError): RolloutConfig(**settings)


def test_one_sample_uncertainty_and_bad_endpoints():
    b=RolloutBackend(flat_evaluator,RolloutConfig(samples=1,horizon=0))
    b.evaluate(one_roll())
    assert all(a['sample_variance'] is None and a['standard_error'] is None for a in b.last_stats['actions'])
    def invalid(states): return np.full((len(states),2),np.nan)
    b=RolloutBackend(invalid,RolloutConfig(samples=2,horizon=0),solver_factory=BankPolicy)
    with pytest.raises(ValueError,match='probability'): b.evaluate(one_roll())
    assert b.last_stats['status']=='incomplete'


def test_cli_complete_and_failure_report(tmp_path):
    suite=tmp_path/'suite.json'
    suite.write_text(json.dumps({'format':'cantstop-search-decisions-v1','positions':[
        {'id':'small','split':'development','tags':['reference'],'snapshot':snapshot(one_roll())}]}))
    output=tmp_path/'done.json'
    main(['--suite',str(suite),'--backend','rollout','--horizon','0','--samples','8','--out',str(output)])
    report=json.loads(output.read_text())
    assert report['status']=='complete'
    assert report['positions'][0]['rollout']['config']['samples']==8
    failed=tmp_path/'failed.json'
    with pytest.raises(RolloutLimitExceeded):
        main(['--suite',str(suite),'--backend','rollout','--full-game','--max-turns','1','--out',str(failed)])
    report=json.loads(failed.read_text())
    assert report['status']=='incomplete' and 'summary' not in report
    assert report['failed_rollout']['status']=='incomplete'


def test_h0_dice_candidates_keep_optimal_subsequent_stop_roll():
    s=one_roll()
    s.runners={10:5,11:3,12:1}
    s.phase=Phase.AWAIT_MOVE; s.dice=(5,5,6,6)
    exact=TurnTableBackend(flat_evaluator).evaluate(s)
    backend=RolloutBackend(flat_evaluator,RolloutConfig(samples=4096,horizon=0))
    result=backend.evaluate(s)
    estimates={o.action:o.value for o in result.options}
    for option in exact.options:
        np.testing.assert_allclose(estimates[option.action],option.value,atol=0.022,rtol=0)
    assert all(a['samples']==4096 for a in backend.last_stats['actions'])
