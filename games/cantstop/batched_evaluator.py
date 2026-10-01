"""One inference worker batches concurrent searches, including mixed seats/rules."""
from concurrent.futures import Future
import queue
import threading
import time

import numpy as np

from .encoder import encode_batch, to_absolute


class BatchingEvaluator:
    thread_safe_batching = True

    def __init__(self, evaluator, *, max_rows=65536, max_pending_rows=131072, wait_seconds=0.002):
        if max_rows < 1 or max_pending_rows < max_rows or wait_seconds < 0:
            raise ValueError('invalid inference batching limits')
        if not hasattr(evaluator,'relative_probs'):
            raise ValueError('batching requires a relative-probability evaluator')
        self.evaluator=evaluator
        self.max_rows,self.max_pending_rows,self.wait_seconds=max_rows,max_pending_rows,wait_seconds
        self.queue=queue.Queue()
        self.condition=threading.Condition()
        self.pending_rows=0
        self.closed=False
        self.stats={'batches':0,'requests':0,'rows':0,'max_batch_rows':0,
                    'max_batch_requests':0,'peak_pending_rows':0,'inference_seconds':0.0}
        self.thread=threading.Thread(target=self._run,name='cantstop-inference',daemon=True)
        self.thread.start()

    def __enter__(self): return self

    def __exit__(self,*args): self.close()

    def close(self):
        with self.condition:
            self.closed=True
            self.condition.notify_all()
        self.queue.put(None)
        self.thread.join()

    def __call__(self,states):
        if not states: raise ValueError('empty evaluation')
        if any(s.rules!=states[0].rules or s.active_player!=states[0].active_player for s in states):
            raise ValueError('each request must have one ruleset and active seat')
        return self.evaluate_features(encode_batch(states),states[0])

    def evaluate_features(self,features,reference):
        features=np.asarray(features,dtype=np.float32)
        if len(features)==0: raise ValueError('empty feature request')
        results=[]
        for start in range(0,len(features),self.max_rows):
            chunk=features[start:start+self.max_rows]
            future=Future()
            with self.condition:
                while not self.closed and self.pending_rows+len(chunk)>self.max_pending_rows:
                    self.condition.wait()
                if self.closed: raise RuntimeError('inference batcher is closed')
                self.pending_rows+=len(chunk)
                self.stats['peak_pending_rows']=max(self.stats['peak_pending_rows'],self.pending_rows)
                self.queue.put((chunk,future))
            relative=future.result()
            results.append(to_absolute(relative,reference))
        return np.concatenate(results,axis=0)

    def _run(self):
        carry=None
        while True:
            item=carry if carry is not None else self.queue.get()
            carry=None
            if item is None: return
            batch=[item]; rows=len(item[0]); deadline=time.perf_counter()+self.wait_seconds
            stopping=False
            while rows<self.max_rows:
                try: item=self.queue.get(timeout=max(0,deadline-time.perf_counter()))
                except queue.Empty: break
                if item is None:
                    stopping=True; break
                if rows+len(item[0])>self.max_rows:
                    carry=item; break
                batch.append(item); rows+=len(item[0])
            try:
                started=time.perf_counter()
                output=np.asarray(self.evaluator.relative_probs(np.concatenate([x[0] for x in batch])))
                if output.shape!=(rows,4) or not np.isfinite(output).all():
                    raise ValueError('invalid batched NN output')
                self.stats['inference_seconds']+=time.perf_counter()-started
                self.stats['batches']+=1; self.stats['requests']+=len(batch); self.stats['rows']+=rows
                self.stats['max_batch_rows']=max(self.stats['max_batch_rows'],rows)
                self.stats['max_batch_requests']=max(self.stats['max_batch_requests'],len(batch))
                offset=0
                for features,future in batch:
                    future.set_result(output[offset:offset+len(features)])
                    offset+=len(features)
            except BaseException as exc:
                for _,future in batch: future.set_exception(exc)
            finally:
                with self.condition:
                    self.pending_rows-=rows
                    self.condition.notify_all()
            if stopping: return
