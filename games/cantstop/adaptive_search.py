"""Fixed-horizon adaptive allocation with simultaneous bounded-payoff intervals.

Variance-corrected estimates are reported, but elimination uses bounded raw
payoffs. No normal-distribution assumption or optional-stopping shortcut.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .decision_search import ActionValue, Decision, TurnTableBackend, actions
from .progressive_search import EarlyTurnPolicy, ProgressiveConfig
from .rollout_search import RolloutBackend, RolloutConfig, RolloutInterrupted
from .snapshot import snapshot


@dataclass(frozen=True)
class AdaptiveConfig:
    budgets: tuple = (32, 128, 512)
    alpha: float = 0.05
    max_sample_bytes: int = 64 * 1024 * 1024
    workers: int = 1

    def __post_init__(self):
        if not isinstance(self.budgets, tuple) or not self.budgets:
            raise ValueError('budgets must be a nonempty tuple')
        if type(self.workers) is not int or not 1 <= self.workers <= 8:
            raise ValueError('workers must be 1-8')
        previous = 0
        for n in self.budgets:
            if type(n) is not int or n <= previous:
                raise ValueError('cumulative sample budgets must strictly increase')
            previous = n
        if not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError('alpha must be between zero and one')
        if type(self.max_sample_bytes) is not int or self.max_sample_bytes < 1:
            raise ValueError('max_sample_bytes must be positive')


def paired_bounds(raw, actor, alpha, checks, total_actions):
    """Union-bound Hoeffding over every pair at each predeclared sample count.

    Each independent simulation-index pair has difference in [-1,1], whether
    its two actions share dice or use independent dice. Adaptive elimination
    selects subsets of these pre-existing streams, not a new outcome-dependent
    sampling distribution. Interval coverage concerns the fixed rollout policy.
    """
    keys = list(raw)
    pairs = max(1, total_actions * (total_actions-1)//2)
    n = len(raw[keys[0]])
    if any(len(raw[k]) != n for k in keys):
        raise ValueError('paired comparisons require aligned sample counts')
    radius = math.sqrt(2 * math.log(2 * pairs * checks / alpha) / n)
    eliminated, comparisons = set(), []
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            mean = float(np.mean(np.asarray(raw[a])[:,actor]-np.asarray(raw[b])[:,actor]))
            low, high = mean-radius, mean+radius
            comparisons.append({'a':a, 'b':b, 'raw_mean_difference':mean,
                                'low':low, 'high':high, 'samples':n})
            if low > 0: eliminated.add(b)
            if high < 0: eliminated.add(a)
    return [k for k in keys if k not in eliminated], comparisons


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    temp.replace(path)


def source_identity():
    root = Path(__file__).parent
    names = ('adaptive_search.py','rollout_search.py','progressive_search.py','turn_search.py',
             'rust_solver.py','decision_search.py','engine.py','encoder.py','model.py','portable_rng.py','batched_evaluator.py')
    result = {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    import cantstop_rust
    native=Path(cantstop_rust.__file__)
    files=[native] if native.suffix in ('.pyd','.so') else sorted(native.parent.glob('*.pyd'))+sorted(native.parent.glob('*.so'))
    for file in files:
        result[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
    return result


class AdaptiveBackend:
    def __init__(self, evaluator, config=AdaptiveConfig(), rollout=RolloutConfig(),
                 continuation=ProgressiveConfig(), *, evaluator_key=None):
        if config.workers > 1 and not getattr(evaluator,'thread_safe_batching',False):
            raise ValueError('parallel candidates require BatchingEvaluator')
        if rollout.sample_offset:
            raise ValueError('adaptive allocation manages sample_offset')
        self.evaluator, self.config, self.rollout = evaluator, config, rollout
        self.continuation, self.evaluator_key = continuation, evaluator_key
        self.last_stats = None

    def evaluate(self, state, *, checkpoint=None, resume=False, seconds=None,
                 cancelled=lambda:False, clock=time.perf_counter):
        if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
            raise ValueError('seconds must be finite and nonnegative')
        if checkpoint is not None and not self.evaluator_key:
            raise ValueError('checkpointing requires a stable evaluator identity')
        if resume and (checkpoint is None or not Path(checkpoint).exists()):
            raise ValueError('resume requires an existing checkpoint')
        if checkpoint is not None and Path(checkpoint).exists() and not resume:
            raise ValueError('checkpoint exists; use resume or a fresh path')
        start = clock(); deadline = None if seconds is None else start+seconds
        baseline = TurnTableBackend(self.evaluator).evaluate(state)
        legal = actions(state)
        identity = {'format':'cantstop-adaptive-v1', 'board':snapshot(state),
                    'adaptive':asdict(self.config), 'rollout':asdict(self.rollout),
                    'continuation':asdict(self.continuation), 'evaluator':self.evaluator_key,
                    'source':source_identity()}
        signature = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        data = {'signature':signature,'identity':identity,'next_stage':0,
                'active':[a.key for a in legal], 'samples':{a.key:{'raw':[], 'adjusted':[]} for a in legal},
                'stages':[], 'status':'running'}
        if resume:
            data = json.loads(Path(checkpoint).read_text(encoding='utf-8'))
            if (data['signature'] != signature or
                    hashlib.sha256(json.dumps(data['identity'],sort_keys=True).encode()).hexdigest()!=signature):
                raise ValueError('resume board, model, code, or configuration differs')
            self._validate_checkpoint(data, legal, state.rules.num_players)
        policy = EarlyTurnPolicy(self.evaluator,self.continuation)
        self.last_stats = {'status':'running','stages':data['stages'], 'policy':policy.stats,
                           'fallback':False,'confidence_method':'simultaneous paired raw-payoff Hoeffding',
                           'alpha':self.config.alpha,'configuration':asdict(self.config)}
        reason = 'sample_budget'
        try:
            required = len(legal)*self.config.budgets[-1]*state.rules.num_players*8*2
            self.last_stats['planned_numeric_sample_bytes'] = required
            if required > self.config.max_sample_bytes:
                raise MemoryError('numeric sample storage budget exceeded')
            if checkpoint is not None: atomic_json(checkpoint,data)
            while data['next_stage'] < len(self.config.budgets) and len(data['active']) > 1:
                if cancelled(): raise RolloutInterrupted('cancelled')
                if deadline is not None and clock() >= deadline: raise RolloutInterrupted('time_budget')
                stage = data['next_stage']; target = self.config.budgets[stage]
                previous = 0 if stage == 0 else self.config.budgets[stage-1]
                cfg = replace(self.rollout, samples=target-previous, sample_offset=previous)
                abort=threading.Event()
                candidates=[a for a in legal if a.key in data['active']]
                def generate(subset, local_policy):
                    backend = RolloutBackend(self.evaluator,cfg,policy_factory=local_policy,retain_samples=True,
                        cancelled=lambda: cancelled() or abort.is_set(),deadline=deadline,clock=clock)
                    try:
                        backend.evaluate(state,candidates=subset)
                    except BaseException:
                        abort.set()
                        raise
                    return backend.last_samples, backend.last_stats
                if self.config.workers == 1:
                    batch_samples, batch_stats = generate(candidates,policy)
                else:
                    # One immutable current-turn table is shared by all root
                    # candidate workers; otherwise parallelism repeats its NN
                    # work once per action. Future caches stay worker-local.
                    shared_root=policy(state,self.evaluator,0)
                    def worker_policy():
                        local=EarlyTurnPolicy(self.evaluator,self.continuation)
                        return lambda s,e,t: shared_root if t==0 else local(s,e,t)
                    with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
                        jobs=[pool.submit(generate,[a],worker_policy()) for a in candidates]
                        pieces=[]; failures=[]
                        for job in jobs:
                            try: pieces.append(job.result())
                            except Exception as exc: failures.append(exc)
                        if failures:
                            raise next((e for e in failures if not isinstance(e,RolloutInterrupted)),failures[0])
                    batch_samples={key:value for samples,_ in pieces for key,value in samples.items()}
                    batch_stats={'parallel_candidates':[stats for _,stats in pieces]}
                # Stage commits are transactional. Failed/partial batches never
                # change the checkpoint's aligned sample-prefix counts.
                new_samples=dict(data['samples'])
                for key in data['active']:
                    new_samples[key]={field:data['samples'][key][field]+batch_samples[key][field].tolist()
                                      for field in ('raw','adjusted')}
                raw = {k:new_samples[k]['raw'] for k in data['active']}
                keep, comparisons = paired_bounds(raw,state.active_player,self.config.alpha,
                                                    len(self.config.budgets),len(legal))
                data['samples']=new_samples
                data['stages'].append({'target':target,'active_before':data['active'],
                    'active_after':keep,'comparisons':comparisons,'rollout':batch_stats})
                data['active'] = keep
                data['next_stage'] += 1
                if checkpoint is not None: atomic_json(checkpoint,data)
            reason = 'resolved' if len(data['active']) <= 1 else 'sample_budget'
        except RolloutInterrupted as exc:
            reason = str(exc)
        except (MemoryError, RuntimeError, ValueError) as exc:
            reason = 'evaluation_failure'
            self.last_stats['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            policy.cache.clear()
        resolved = len(data['active']) <= 1
        selected = next((o.action for o in baseline.options if o.action.key in data['active']),None)
        options=[]
        for option in baseline.options:
            if option.action.key not in data['active']: continue
            values=data['samples'][option.action.key]['adjusted']
            value=tuple(map(float,np.mean(values,axis=0))) if values else option.value
            options.append(ActionValue(option.action,value))
        options.sort(key=lambda o:-o.value[state.active_player])
        value=next((o.value for o in options if o.action==selected),baseline.value)
        self.last_stats.update(status='complete' if reason in ('resolved','sample_budget') else 'interrupted',
            stop_reason=reason,fallback=not resolved,active=data['active'],
            committed_samples={k:len(v['raw']) for k,v in data['samples'].items()},
            elapsed_seconds=clock()-start,
            time_budget_overshoot_seconds=0 if deadline is None else max(0,clock()-deadline))
        data['status']=self.last_stats['status']; data['stop_reason']=reason
        if checkpoint is not None: atomic_json(checkpoint,data)
        return Decision(state.active_player,value,selected,tuple(options))

    def _validate_checkpoint(self,data,legal,n):
        keys={a.key for a in legal}; stage=data['next_stage']
        if type(stage) is not int or not 0 <= stage <= len(self.config.budgets):
            raise ValueError('invalid checkpoint stage')
        if len(data['stages']) != stage:
            raise ValueError('checkpoint stage history mismatch')
        if (set(data['samples']) != keys or not set(data['active']).issubset(keys)
                or len(set(data['active'])) != len(data['active']) or (keys and not data['active'])):
            raise ValueError('invalid checkpoint actions')
        for k,entry in data['samples'].items():
            raw=np.asarray(entry['raw'],dtype=float); adjusted=np.asarray(entry['adjusted'],dtype=float)
            if raw.shape != adjusted.shape or (raw.size and (raw.ndim!=2 or raw.shape[1]!=n)):
                raise ValueError('invalid checkpoint sample shape')
            if not np.isfinite(raw).all() or not np.isfinite(adjusted).all() or (raw<0).any() or (raw>1).any():
                raise ValueError('invalid checkpoint sample values')
            if len(raw) not in (0,*self.config.budgets[:stage]):
                raise ValueError('invalid checkpoint prefix length')
            if k in data['active'] and len(raw)!=(0 if stage==0 else self.config.budgets[stage-1]):
                raise ValueError('unaligned checkpoint sample count')
