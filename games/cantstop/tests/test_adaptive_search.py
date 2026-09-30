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
    assert tiny.evaluate(one_roll()).selected is not None
    assert tiny.last_stats['stop_reason']=='evaluation_failure'
    assert 'MemoryError' in tiny.last_stats['error']


def test_inference_failure_retains_valid_baseline(monkeypatch):
    def fail(*args,**kwargs): raise RuntimeError('inference failed')
    monkeypatch.setattr(RolloutBackend,'evaluate',fail)
    b=AdaptiveBackend(flat_evaluator,AdaptiveConfig(budgets=(4,)))
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
    b=AdaptiveBackend(flat_evaluator,AdaptiveConfig(budgets=(4,64,256)))
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
    assert result['decisions'] and all(d['samples']>0 for d in result['decisions'])
