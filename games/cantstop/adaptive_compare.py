"""Offline adaptive comparison, resumable per position, with bounded GPU batching."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import cProfile
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np

from .adaptive_search import AdaptiveBackend, AdaptiveConfig, atomic_json
from .batched_evaluator import BatchingEvaluator
from .decision_compare import DEFAULT_SUITE, load_suite
from .engine import GameState
from .experiment import identity, file_sha256
from .progressive_search import ProgressiveConfig
from .rollout_search import RolloutConfig
from .snapshot import from_snapshot


class Measurements:
    def __init__(self,device):
        self.device=device; self.stop=threading.Event(); self.gpu=[]; self.peak_rss=0
        self.thread=threading.Thread(target=self._run,daemon=True)
    def _run(self):
        import torch
        try:
            import psutil
            process=psutil.Process()
        except ImportError: process=None
        while not self.stop.is_set():
            if process is not None:
                info=process.memory_info()
                self.peak_rss=max(self.peak_rss,getattr(info,'peak_wset',info.rss))
            if str(self.device).startswith('cuda'):
                try: self.gpu.append(float(torch.cuda.utilization(self.device)))
                except (ImportError,ModuleNotFoundError,RuntimeError,AttributeError): pass
            self.stop.wait(.25)
    def __enter__(self): self.thread.start(); return self
    def __exit__(self,*args): self.stop.set(); self.thread.join()
    def result(self):
        import torch
        cuda=str(self.device).startswith('cuda')
        return {'peak_process_rss_bytes':self.peak_rss or None,
            'gpu_utilization_mean_percent':float(np.mean(self.gpu)) if self.gpu else None,
            'gpu_utilization_samples':len(self.gpu),
            'gpu_utilization_scope':'whole device, may include other processes; unavailable if NVML is missing',
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(self.device) if cuda else None,
            'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(self.device) if cuda else None}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--suite',type=Path,default=DEFAULT_SUITE)
    p.add_argument('--position',action='append')
    p.add_argument('--confidence',choices=('paired_t','hoeffding'),default='paired_t',
                   help='paired_t: approximate corrected-sample intervals; hoeffding: bounded raw intervals')
    p.add_argument('--budgets',type=int,nargs='+',default=[16,64])
    p.add_argument('--horizon',type=int,default=1)
    p.add_argument('--full-game',action='store_true')
    p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--dice-luck',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--common-random-numbers',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--early-turns',type=int,default=0)
    p.add_argument('--early-expansions',type=int,default=0)
    p.add_argument('--early-depth',type=int,default=1)
    p.add_argument('--parallel-candidates',type=int,default=1)
    p.add_argument('--parallel-positions',type=int,default=1)
    p.add_argument('--max-batch-rows',type=int,default=65536)
    p.add_argument('--max-pending-rows',type=int,default=131072)
    p.add_argument('--max-sample-bytes',type=int,default=64*1024*1024)
    p.add_argument('--seconds',type=float,help='Soft per-position budget, baseline solve included')
    p.add_argument('--seed',type=int,default=20260930)
    p.add_argument('--state-dir',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.exists() and not args.resume: p.error('output exists; use --resume or a new output')
    if not 1<=args.parallel_positions<=8: p.error('parallel positions must be 1-8')
    cfg=AdaptiveConfig(tuple(args.budgets),args.alpha,args.max_sample_bytes,args.parallel_candidates,confidence=args.confidence)
    rollout=RolloutConfig(horizon=None if args.full_game else args.horizon,seed=args.seed,
        dice_luck=args.dice_luck,common_random_numbers=args.common_random_numbers)
    continuation=ProgressiveConfig(early_turns=args.early_turns,expansions=args.early_expansions,depth=args.early_depth)
    selected=set(args.position or ['2p5n_three_runners','3p4n_three_runners','4p3n_three_runners'])
    rows=[r for r in load_suite(args.suite) if r['id'] in selected]
    if {r['id'] for r in rows}!=selected: p.error('unknown development position')
    state_dir=args.state_dir or args.out.with_suffix('.states')
    from .model import NetEvaluator,load_net
    model=NetEvaluator(load_net(args.checkpoint,device=args.device),device=args.device)
    model([GameState(from_snapshot(rows[0]['snapshot']).rules)])
    initial_rows=model.rows
    if args.device.startswith('cuda'):
        import torch
        torch.cuda.reset_peak_memory_stats(args.device)
    key=file_sha256(args.checkpoint)
    report={'status':'running','meta':identity(nets={'shared':args.checkpoint},
        adaptive=asdict(cfg),rollout=asdict(rollout),continuation=asdict(continuation),
        positions_parallel=args.parallel_positions,seed=args.seed,suite_sha256=file_sha256(args.suite),
        device=args.device,seconds=args.seconds),'positions':[],'errors':[]}
    atomic_json(args.out,report)
    batched=args.parallel_candidates>1 or args.parallel_positions>1
    context=BatchingEvaluator(model,max_rows=args.max_batch_rows,max_pending_rows=args.max_pending_rows) if batched else nullcontext(model)
    started=time.perf_counter()
    cancellation=threading.Event()
    with Measurements(args.device) as measurements, context as evaluator:
        def run(row):
            checkpoint=state_dir/(hashlib.sha256(row['id'].encode()).hexdigest()[:16]+'.json')
            backend=AdaptiveBackend(evaluator,cfg,rollout,continuation,evaluator_key=key)
            profile=cProfile.Profile() if args.profile else None
            if profile: profile.enable()
            try:
                result=backend.evaluate(from_snapshot(row['snapshot']),checkpoint=checkpoint,
                    resume=args.resume and checkpoint.exists(),seconds=args.seconds,cancelled=cancellation.is_set)
            finally:
                if profile:
                    profile.disable(); checkpoint.parent.mkdir(parents=True,exist_ok=True)
                    profile.dump_stats(str(checkpoint.with_suffix('.prof')))
            return {'id':row['id'],'decision':result.to_dict(),'adaptive':backend.last_stats,
                    'checkpoint':str(checkpoint)}
        with ThreadPoolExecutor(max_workers=args.parallel_positions) as pool:
            futures={pool.submit(run,row):row['id'] for row in rows}
            try:
                for future in as_completed(futures):
                    try: report['positions'].append(future.result())
                    except Exception as exc: report['errors'].append({'id':futures[future],'error':f'{type(exc).__name__}: {exc}'})
                    report['positions'].sort(key=lambda row:row['id'])
                    atomic_json(args.out,report)
                    print(f"completed {futures[future]}",flush=True)
            except KeyboardInterrupt:
                cancellation.set()
                report['errors'].append({'id':'run','error':'cancelled by user'})
                completed={r['id'] for r in report['positions']}
                for future in futures:
                    if futures[future] in completed: continue
                    try: report['positions'].append(future.result())
                    except Exception as exc: report['errors'].append({'id':futures[future],'error':str(exc)})
        report['batching']=evaluator.stats if batched else None
    times=[r['adaptive']['elapsed_seconds'] for r in report['positions']]
    report['measurements']=measurements.result()
    report['summary']={'wall_seconds':time.perf_counter()-started,'nn_rows':model.rows-initial_rows,
        'decision_seconds_median':float(np.median(times)) if times else None,
        'decision_seconds_p95':float(np.percentile(times,95)) if times else None,
        'resolved':sum(not r['adaptive']['fallback'] for r in report['positions']),
        'unresolved':sum(r['adaptive']['fallback'] for r in report['positions']),
        'committed_rollouts':sum(sum(r['adaptive']['committed_samples'].values()) for r in report['positions'])}
    report['status']='incomplete' if report['errors'] else 'complete'
    atomic_json(args.out,report)
    if report['errors']: raise RuntimeError('some positions failed; see saved report')


if __name__=='__main__': main()
