from dataclasses import replace
import json
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from games.cantstop.adaptive_search import AdaptiveBackend,AdaptiveConfig,paired_bounds
from games.cantstop.batched_evaluator import BatchingEvaluator
from games.cantstop.decision_search import Action
from games.cantstop.encoder import encode_batch,seat_mask
from games.cantstop.engine import GameState,RuleSet
from games.cantstop.rollout_search import RolloutBackend,RolloutConfig
from games.cantstop.rust_solver import flat_evaluator
from games.cantstop.tests.test_rollout_search import one_roll


def test_paired_intervals_known_gaps_ties_and_sample_alignment():
    raw={'a':np.tile([.9,.1],(64,1)), 'b':np.tile([.1,.9],(64,1))}
    keep,intervals=paired_bounds(raw,0,.05,3,2)
    assert keep==['a'] and intervals[0]['low']>0
    raw['b']=raw['a'].copy()
    assert paired_bounds(raw,0,.05,3,2)[0]==['a','b']
    raw['b']=raw['a'][:-1]
    with pytest.raises(ValueError): paired_bounds(raw,0,.05,3,2)


def test_chunked_sample_indices_match_uninterrupted_stream():
    s=one_roll(); cfg=RolloutConfig(samples=12,horizon=None,dice_luck=True,common_random_numbers=True)
    full=RolloutBackend(flat_evaluator,cfg,retain_samples=True); full.evaluate(s)
    parts=[]
    for offset,n in ((0,5),(5,7)):
        b=RolloutBackend(flat_evaluator,replace(cfg,samples=n,sample_offset=offset),retain_samples=True)
        b.evaluate(s); parts.append(b.last_samples)
    for key in full.last_samples:
        for field in ('raw','adjusted'):
            np.testing.assert_array_equal(full.last_samples[key][field],np.concatenate([p[key][field] for p in parts]))


def test_resume_exactly_preserves_samples_counts_and_decision(tmp_path):
    s=one_roll(); cfg=AdaptiveConfig(budgets=(4,8,16)); rc=RolloutConfig(horizon=1,dice_luck=True,common_random_numbers=True)
    path=tmp_path/'state.json'
    b=AdaptiveBackend(flat_evaluator,cfg,rc,evaluator_key='flat-v1')
    def cancelled():
        return path.exists() and json.loads(path.read_text())['next_stage']>=1
    b.evaluate(s,checkpoint=path,cancelled=cancelled)
    assert b.last_stats['stop_reason']=='cancelled'
    saved=json.loads(path.read_text())
    assert saved['next_stage']==1
    resumed=b.evaluate(s,checkpoint=path,resume=True)
    finished=json.loads(path.read_text())
    direct=AdaptiveBackend(flat_evaluator,cfg,rc,evaluator_key='flat-v1')
    other=tmp_path/'other.json'
    assert direct.evaluate(s,checkpoint=other)==resumed
    expected=json.loads(other.read_text())
    assert finished['samples']==expected['samples'] and finished['active']==expected['active']
    assert b.last_stats['committed_samples']==direct.last_stats['committed_samples']
    assert saved['samples']['roll']['raw']==finished['samples']['roll']['raw'][:4]
    with pytest.raises(ValueError): b.evaluate(s,checkpoint=path)
    different=AdaptiveBackend(flat_evaluator,cfg,rc,evaluator_key='other-model')
    with pytest.raises(ValueError): different.evaluate(s,checkpoint=path,resume=True)


def test_cancelled_batch_not_committed_and_time_memory_fallback(tmp_path):
    cfg=AdaptiveConfig(budgets=(4,8)); rc=RolloutConfig(horizon=1)
    b=AdaptiveBackend(flat_evaluator,cfg,rc,evaluator_key='flat')
    count=[0]
    def cancelled():
        count[0]+=1
        return count[0]>4
    b.evaluate(one_roll(),checkpoint=tmp_path/'cancel.json',cancelled=cancelled)
    assert all(n==0 for n in b.last_stats['committed_samples'].values())
    assert b.last_stats['fallback']
    b.evaluate(one_roll(),seconds=0)
    assert b.last_stats['stop_reason']=='time_budget'
    tiny=AdaptiveBackend(flat_evaluator,replace(cfg,max_sample_bytes=1),rc)
    with pytest.raises(MemoryError): tiny.evaluate(one_roll())
    assert tiny.last_stats['stop_reason']=='evaluation_failure'
    assert 'MemoryError' in tiny.last_stats['error']


def test_inference_failure_raises_unless_diagnostic_fallback_requested(monkeypatch):
    def fail(*args,**kwargs): raise RuntimeError('inference failed')
    monkeypatch.setattr(RolloutBackend,'evaluate',fail)
    b=AdaptiveBackend(flat_evaluator,AdaptiveConfig(budgets=(4,)))
    with pytest.raises(RuntimeError,match='inference failed'): b.evaluate(one_roll())
    assert not b.last_stats['fallback']
    b=AdaptiveBackend(flat_evaluator,AdaptiveConfig(budgets=(4,),failure_policy='fallback'))
    r=b.evaluate(one_roll())
    assert r.selected is not None and b.last_stats['fallback']
    assert 'inference failed' in b.last_stats['error']


class RelativeFlat:
    def relative_probs(self,features):
        mask=seat_mask(features).astype(float)
        return mask/mask.sum(axis=1,keepdims=True)


def test_batcher_mixed_seats_rules_and_memory_bounds():
    states=[GameState(RuleSet.make(n)) for n in (2,3,4)]
    for s in states: s.active_player=s.rules.num_players-1
    barrier=threading.Barrier(3)
    with BatchingEvaluator(RelativeFlat(),max_rows=16,max_pending_rows=32,wait_seconds=.03) as evaluator:
        def run(s):
            barrier.wait()
            return evaluator([s]*3)
        with ThreadPoolExecutor(max_workers=3) as pool:
            results=list(pool.map(run,states))
        for s,result in zip(states,results):
            np.testing.assert_allclose(result,1/s.rules.num_players)
            assert result.shape==(3,s.rules.num_players)
        assert evaluator.stats['max_batch_requests']>1
        assert evaluator.stats['peak_pending_rows']<=32
        assert evaluator.stats['max_batch_rows']<=16
    with pytest.raises(RuntimeError): evaluator([states[0]])


def test_batcher_errors_unblock_all_waiters():
    class Broken:
        def relative_probs(self,features): raise RuntimeError('gpu failed')
    with BatchingEvaluator(Broken(),wait_seconds=.02) as evaluator:
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(evaluator,[GameState(RuleSet.make(2))]) for _ in range(2)]
            for job in jobs:
                with pytest.raises(RuntimeError,match='gpu failed'): job.result(timeout=2)
        assert evaluator.pending_rows==0


def test_parallel_candidates_equal_serial():
    cfg=AdaptiveConfig(budgets=(4,8)); rc=RolloutConfig(horizon=1,dice_luck=True,common_random_numbers=True)
    serial=AdaptiveBackend(flat_evaluator,cfg,rc).evaluate(one_roll())
    with BatchingEvaluator(RelativeFlat()) as evaluator:
        parallel=AdaptiveBackend(evaluator,replace(cfg,workers=2),rc)
        assert parallel.evaluate(one_roll())==serial


@pytest.mark.parametrize('kw',[{'budgets':()},{'budgets':(4,4)},{'alpha':0},{'alpha':float('nan')},
    {'max_sample_bytes':0},{'workers':0}])
def test_invalid_adaptive_settings(kw):
    with pytest.raises(ValueError): AdaptiveConfig(**kw)


def test_adaptive_resolves_misleading_baseline_and_stops_allocating(monkeypatch):
    def fake(self,state,*,candidates=None):
        self.last_samples={a.key:{'raw':np.tile([.95,.05] if a.kind=='stop' else [.05,.95],(self.config.samples,1)),
            'adjusted':np.tile([.95,.05] if a.kind=='stop' else [.05,.95],(self.config.samples,1))} for a in candidates}
        self.last_stats={'status':'complete'}
    monkeypatch.setattr(RolloutBackend,'evaluate',fake)
    b=AdaptiveBackend(flat_evaluator,AdaptiveConfig(budgets=(4,64,256),confidence="hoeffding"))
    result=b.evaluate(one_roll())
    assert result.selected==Action('stop')
    assert not b.last_stats['fallback'] and b.last_stats['stop_reason']=='resolved'
    assert len(b.last_stats['stages'])==2
    assert all(n==64 for n in b.last_stats['committed_samples'].values())


def test_unresolved_tie_uses_baseline_choice(monkeypatch):
    def fake(self,state,*,candidates=None):
        self.last_samples={a.key:{'raw':np.full((self.config.samples,2),.5),
            'adjusted':np.full((self.config.samples,2),.5)} for a in candidates}
        self.last_stats={'status':'complete'}
    monkeypatch.setattr(RolloutBackend,'evaluate',fake)
    from games.cantstop.decision_search import TurnTableBackend
    expected=TurnTableBackend(flat_evaluator).evaluate(one_roll()).selected
    b=AdaptiveBackend(flat_evaluator,AdaptiveConfig(budgets=(4,8)))
    assert b.evaluate(one_roll()).selected==expected
    assert b.last_stats['fallback']


def test_resume_rejects_corrupted_empty_active_set(tmp_path):
    path=tmp_path/'state.json'
    b=AdaptiveBackend(flat_evaluator,AdaptiveConfig(budgets=(2,)),evaluator_key='flat')
    b.evaluate(one_roll(),checkpoint=path)
    data=json.loads(path.read_text()); data['active']=[]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): b.evaluate(one_roll(),checkpoint=path,resume=True)


def test_mask_optimization_matches_index_select_on_cpu_and_cuda():
    import torch
    from games.cantstop.model import seat_mask_tensor
    from games.cantstop.encoder import FEATURE_SIZE,seat_present_index
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        x=torch.randn(9,FEATURE_SIZE,device=device)
        idx=torch.tensor([seat_present_index(i) for i in range(4)],device=device)
        assert torch.equal(seat_mask_tensor(x),x.index_select(1,idx)>0)
        assert torch.equal(seat_mask_tensor(x[0]),x[:1].index_select(1,idx)>0)
        y=x[::2]
        assert torch.equal(seat_mask_tensor(y),y.index_select(1,idx)>0)


def test_adaptive_arena_records_search_decisions_and_preserves_start():
    from games.cantstop.adaptive_arena import play_game
    from games.cantstop.snapshot import snapshot
    state=one_roll(); before=snapshot(state)
    result=play_game(state.rules,flat_evaluator,7,0,AdaptiveConfig(budgets=(2,)),
                     RolloutConfig(horizon=0),start=state)
    assert result['winner_seat'] in (0,1)
    assert result['decisions'] and all('fallback' in d for d in result['decisions'])
    assert snapshot(state)==before


def test_cli_cooperative_cancellation_and_resume(tmp_path,monkeypatch):
    from games.cantstop import adaptive_compare
    from games.cantstop.model import CantStopNet,save_net
    from games.cantstop.snapshot import snapshot
    model=tmp_path/'model.pt'; save_net(CantStopNet(hidden=(8,)),model)
    suite=tmp_path/'suite.json'
    suite.write_text(json.dumps({'format':'cantstop-search-decisions-v1','positions':[
        {'id':'small','split':'development','tags':[],'snapshot':snapshot(one_roll())}]}))
    out=tmp_path/'report.json'
    args=['--checkpoint',str(model),'--device','cpu','--suite',str(suite),'--position','small',
          '--budgets','2','4','--horizon','0','--out',str(out)]
    original=adaptive_compare.as_completed
    def interrupt(_): raise KeyboardInterrupt()
    monkeypatch.setattr(adaptive_compare,'as_completed',interrupt)
    with pytest.raises(RuntimeError): adaptive_compare.main(args)
    report=json.loads(out.read_text())
    assert report['status']=='incomplete'
    assert any(e['error']=='cancelled by user' for e in report['errors'])
    monkeypatch.setattr(adaptive_compare,'as_completed',original)
    adaptive_compare.main(args+['--resume'])
    assert json.loads(out.read_text())['status']=='complete'


@pytest.mark.parametrize('mode',['progressive','rollout'])
def test_arena_accepts_other_decision_backends(mode):
    from games.cantstop.adaptive_arena import play_game
    from games.cantstop.progressive_search import ProgressiveBackend,ProgressiveConfig
    state=one_roll()
    def factory():
        if mode=='progressive':
            return ProgressiveBackend(flat_evaluator,ProgressiveConfig(horizons=(0,),samples=(2,)))
        return RolloutBackend(flat_evaluator,RolloutConfig(samples=2,horizon=0))
    result=play_game(state.rules,flat_evaluator,5,0,AdaptiveConfig(budgets=(2,)),
                     RolloutConfig(horizon=0),start=state,backend_factory=factory)
    assert result['decisions']
    assert all(d['samples']==0 if mode=='progressive' else d['samples']>0 for d in result['decisions'])


def test_corrected_t_resolves_close_reversal_while_raw_hoeffding_cannot(monkeypatch):
    from games.cantstop.decision_search import TurnTableBackend
    state=one_roll()
    baseline=TurnTableBackend(flat_evaluator).evaluate(state).selected
    other=Action('stop') if baseline.kind=='roll' else Action('roll')
    def fake(self,state,*,candidates=None):
        rng=np.random.default_rng(77)
        noise=rng.normal(0,.005,self.config.samples)
        self.last_samples={}
        for a in candidates:
            adjusted=.5 + (noise+.01 if a==other else np.zeros(len(noise)))
            raw=.5 + (rng.uniform(-.2,.2,len(noise)))
            self.last_samples[a.key]={'raw':np.column_stack((raw,1-raw)),
                'adjusted':np.column_stack((adjusted,1-adjusted))}
        self.last_stats={'status':'complete'}
    monkeypatch.setattr(RolloutBackend,'evaluate',fake)
    b=AdaptiveBackend(flat_evaluator,AdaptiveConfig(budgets=(32,128,512)))
    assert b.evaluate(state).selected==other
    assert b.last_stats['confidence_approximate']
    assert b.last_stats['stop_reason']=='resolved'
    assert len(b.last_stats['stages'])==1
    old=AdaptiveBackend(flat_evaluator,AdaptiveConfig(budgets=(32,128,512),confidence='hoeffding'))
    assert old.evaluate(state).selected==baseline
    assert old.last_stats['fallback']


def test_paired_t_matches_student_quantile_and_keeps_degenerate_samples():
    from scipy.stats import t
    d=np.linspace(-.02,.03,64)
    samples={'a':np.column_stack((d,1-d)), 'b':np.zeros((64,2))}
    keep,rows=paired_bounds(samples,0,.05,3,4,method='paired_t')
    expected=t.isf(.05/(2*6*3),63)*d.std(ddof=1)/8
    assert rows[0]['high']-rows[0]['mean_difference']==pytest.approx(expected)
    assert rows[0]['sample_kind']=='adjusted'
    constant={'a':np.ones((64,2)), 'b':np.zeros((64,2))}
    assert paired_bounds(constant,0,.05,3,2,method='paired_t')[0]==['a','b']
    assert paired_bounds({k:v[:16] for k,v in samples.items()},0,.05,3,2,method='paired_t')[1][0]['low'] is None


def test_t_interval_null_coverage_on_predeclared_normal_toy_streams():
    # Calibration smoke for the intended approximation, not a proof for game payoffs.
    rng=np.random.default_rng(102)
    false_eliminations=0
    for _ in range(200):
        d=rng.normal(0,.005,128)
        for n in (32,64,128):
            samples={'a':np.column_stack((d[:n],-d[:n])), 'b':np.zeros((n,2))}
            if len(paired_bounds(samples,0,.05,3,2,method='paired_t')[0])<2:
                false_eliminations+=1
                break
    assert false_eliminations<=20


def test_adaptive_horizon_zero_returns_exact_baseline_without_rollouts(monkeypatch):
    from games.cantstop.decision_search import TurnTableBackend
    def fail(*args,**kwargs): raise AssertionError('H=0 must not sample')
    monkeypatch.setattr(RolloutBackend,'evaluate',fail)
    state=one_roll()
    backend=AdaptiveBackend(flat_evaluator,rollout=RolloutConfig(horizon=0))
    assert backend.evaluate(state)==TurnTableBackend(flat_evaluator).evaluate(state)
    assert backend.last_stats['stop_reason']=='exact_baseline'
    assert not backend.last_stats['fallback']
    assert sum(backend.last_stats['committed_samples'].values())==0


@pytest.mark.parametrize('failure_policy',['raise','fallback'])
def test_arena_cannot_play_through_evaluation_failure(monkeypatch,failure_policy):
    from games.cantstop.adaptive_arena import play_game,ArenaEvaluationFailure,finalize_report
    def fail(*args,**kwargs): raise RuntimeError('CUDA out of memory')
    monkeypatch.setattr(RolloutBackend,'evaluate',fail)
    state=one_roll()
    with pytest.raises(ArenaEvaluationFailure) as error:
        play_game(state.rules,flat_evaluator,1,0,AdaptiveConfig(budgets=(32,),failure_policy=failure_policy),
                  RolloutConfig(horizon=1),start=state)
    report={'status':'complete','games':[], 'failed_game':error.value.game,
            'meta':{'start_fixture':None,'rules':{'num_players':2}}}
    finalize_report(report)
    assert report['result'] is None and report['status']=='incomplete'
    assert report['decision_summary']['evaluation_failures']==1
    assert 'CUDA out of memory' in report['failed_game']['decisions'][-1]['error']


def test_arena_summary_counts_budget_fallbacks_and_refuses_error_results():
    from games.cantstop.adaptive_arena import finalize_report
    game={'winner_seat':0,'challenger_seat':0,'decisions':[
        {'stop_reason':'sample_budget','fallback':True},
        {'stop_reason':'time_budget','fallback':True}]}
    report={'status':'complete','games':[game], 'meta':{'start_fixture':None,'rules':{'num_players':2}}}
    finalize_report(report)
    assert report['result'] is not None
    assert report['decision_summary']['fallbacks_by_reason']=={'sample_budget':1,'time_budget':1}
    game['decisions'].append({'stop_reason':'evaluation_failure','fallback':True})
    finalize_report(report)
    assert report['result'] is None and report['verdict_withheld']=='evaluation_failure'


def test_real_rollout_can_reverse_baseline_under_corrected_t():
    from games.cantstop.tests.test_progressive_search import reversal
    from games.cantstop.rust_solver import hashed_evaluator
    from games.cantstop.decision_search import TurnTableBackend
    state=reversal()
    assert TurnTableBackend(hashed_evaluator).evaluate(state).selected==Action('stop')
    backend=AdaptiveBackend(hashed_evaluator,AdaptiveConfig(budgets=(32,128,512)))
    assert backend.evaluate(state).selected==Action('roll')
    assert backend.last_stats['stop_reason']=='resolved'
    assert not backend.last_stats['fallback']


def test_arena_cli_saves_failure_and_no_verdict(tmp_path,monkeypatch):
    from games.cantstop import adaptive_arena
    from games.cantstop.model import CantStopNet,save_net
    checkpoint=tmp_path/'model.pt'; save_net(CantStopNet(hidden=(8,)),checkpoint)
    out=tmp_path/'arena.json'
    partial={'seed':1,'challenger_seat':0,'turns':1,'decisions':[
        {'stop_reason':'evaluation_failure','fallback':False,'error':'inference failed'}]}
    def fail(*args,**kwargs):
        raise adaptive_arena.ArenaEvaluationFailure('inference failed',partial)
    monkeypatch.setattr(adaptive_arena,'play_game',fail)
    with pytest.raises(adaptive_arena.ArenaEvaluationFailure):
        adaptive_arena.main(['--checkpoint',str(checkpoint),'--device','cpu','--games','2','--out',str(out)])
    report=json.loads(out.read_text())
    assert report['status']=='incomplete' and report['result'] is None
    assert report['decision_summary']['evaluation_failures']==1
    assert report['failed_game']==partial
