"""CUDA-graph replay for the self-play forward.

Why: the flat boundary's forward is ~520 kernel launches, and on a fast GPU the
host cost of issuing them is most of the call. Measured on the run07 network
(W1/W2/W4/W5, bf16, laptop 3070): a 4-row call took 25.7 ms of which the GPU was
busy 4.9 ms. On the rented 5090 one thread doing that dispatch pinned at 100%
while the scheduler shards idled 60%. A graph replays all of them in one call.

A graph needs static shapes, so the caller pads to buckets:

* tokens and legal width -- by the adapter, through its own padding (the same
  pad values rows of unequal length already get), see
  ``_RustFlatBatchAdapter.build_device_batch``;
* rows -- here, to ``row_bucket(rows)``, by repeating row 0. Rows are
  independent through the whole network, so copies of a real row are valid
  input whose outputs are simply dropped.

One graph per distinct input signature (every tensor's shape and dtype), made
the first time that signature is seen, and sharing one memory pool: replays are
serialised on the evaluator's single worker thread, so no two can overlap.

Contract for callers:

* Outputs are VIEWS into the graph's static buffers. They are valid until the
  next call on this wrapper, which is how the adapter uses them (it copies to
  the host within the same call). Clone anything kept longer.
* Weights are captured by address. Loading new weights into the SAME module in
  place is picked up; replacing parameters is not. Self-play builds a fresh
  evaluator per checkpoint, which builds a fresh wrapper.
* Anything that cannot be graphed -- CPU, a batch wider than ``max_rows``, a
  failed capture -- runs eagerly, and a failed capture is remembered so it is
  not retried every call.
"""

from __future__ import annotations

import contextlib
import threading

import torch

#: Process-wide lock for GPU work while graphs are on. A capture is broken by
#: ANY concurrent CUDA work in the process, not just work on its own stream:
#: another thread's eager op on the legacy default stream fails it with
#: "operation not permitted when stream is capturing", and the first failure
#: leaves the allocator recording ("already recording to mempool_id"). With
#: several shard threads calling adapters at once this disabled graphs for the
#: whole of run07 (setup.log; reproduced with 4 threads on the laptop: 0
#: replays). Re-entrant, so the adapter can hold it around a whole call and the
#: wrapper can take it again inside. Contract: concurrent CUDA work in a process
#: that captures graphs must hold this lock.
GPU_LOCK = threading.RLock()

#: Smallest row bucket. Below this the saving per call is the same and more
#: buckets only cost memory and capture time.
MIN_ROW_BUCKET = 8
#: Token and legal-width buckets are multiples of this.
WIDTH_BUCKET = 8


def row_bucket(rows: int) -> int:
    """Pad rows to a bucket that wastes little: 8, 16, then steps of 32 to 512,
    of 128 to 2048, of 512 beyond. Powers of two cost up to 2x the GPU work --
    measured 0.72x at 160 rows on a device-bound GPU -- where these cost <20%."""

    for bucket in (MIN_ROW_BUCKET, 16):
        if rows <= bucket:
            return bucket
    for limit, step in ((512, 32), (2048, 128)):
        if rows <= limit:
            return -(-rows // step) * step
    return -(-rows // 512) * 512


def width_bucket(width: int) -> int:
    return max(WIDTH_BUCKET, -(-int(width) // WIDTH_BUCKET) * WIDTH_BUCKET)


class GraphedForward(torch.nn.Module):
    """Wrap a model whose forward takes and returns dicts of row-major tensors."""

    def __init__(self, model: torch.nn.Module, *, max_rows: int = 4096):
        super().__init__()
        self.model = model
        self.max_rows = int(max_rows)
        self._graphs: dict = {}
        self._failed: set = set()
        #: Set by the first failed capture. A failure can leave the caching
        #: allocator recording into the pool ("beginAllocateToPool: already
        #: recording"), after which every later capture fails too -- so stop
        #: trying and run eager for the life of this wrapper.
        self._disabled = False
        self._pool = None
        self.replays = 0
        self.eager_calls = 0
        self.captures = 0
        self.capture_failures = 0

    def __getattr__(self, name):
        # Callers read model attributes (`action_residual`, head switches)
        # straight off `evaluator.model`; keep answering them.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("model"), name)

    def forward(self, batch: dict) -> dict:
        with GPU_LOCK:
            return self._forward_locked(batch)

    def _forward_locked(self, batch: dict) -> dict:
        rows = _rows_of(batch)
        if (
            self._disabled
            or rows is None
            or rows > self.max_rows
            or not all(v.is_cuda for v in batch.values())
        ):
            return self._eager(batch)
        padded_rows = row_bucket(rows)
        padded = _pad_rows(batch, padded_rows)
        key = (
            padded_rows,
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None,
            tuple(
                (name, tuple(value.shape), value.dtype)
                for name, value in sorted(padded.items())
            ),
        )
        if key in self._failed:
            return self._eager(batch)
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._capture(key, padded)
            if entry is None:
                return self._eager(batch)
        graph, static_in, static_out = entry
        for name, value in padded.items():
            static_in[name].copy_(value)
        graph.replay()
        self.replays += 1
        return {
            name: (value[:rows] if _is_row_major(value, padded_rows) else value)
            for name, value in static_out.items()
        }

    def _eager(self, batch):
        self.eager_calls += 1
        return self.model(batch)

    def _capture(self, key, padded):
        # Autocast's weight-cast cache would hand the graph casts that are freed
        # when the capture's autocast context exits. Re-enter it uncached.
        autocast = (
            torch.autocast("cuda", dtype=key[2], cache_enabled=False)
            if key[1]
            else contextlib.nullcontext()
        )
        static_in = {name: value.clone() for name, value in padded.items()}
        # Drain work queued before the lock was taken, so nothing in flight can
        # touch the capture.
        torch.cuda.synchronize()
        original_stream = torch.cuda.current_stream()
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), autocast:
                for _ in range(2):
                    self.model(static_in)
            torch.cuda.current_stream().wait_stream(side)
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            # THREAD-LOCAL: the default ("global") mode fails the capture if ANY
            # other thread in the process makes an unsafe CUDA call meanwhile --
            # on the box something did, and the capture died with
            # cudaErrorStreamCaptureInvalidated. Nothing else in the process
            # touches this model's tensors, so the global guard buys nothing.
            with torch.cuda.graph(
                graph, pool=self._pool, capture_error_mode="thread_local"
            ), autocast:
                static_out = self.model(static_in)
        except Exception as error:  # noqa: BLE001 -- any failure means eager
            _recover_from_failed_capture(original_stream, self._pool)
            self.capture_failures += 1
            self._failed.add(key)
            self._disabled = True
            print(f"cuda graphs: capture failed at {key[0]} rows, running EAGER from "
                  f"now on: {type(error).__name__}: {str(error).splitlines()[0]}",
                  flush=True)
            return None
        if not isinstance(static_out, dict):
            self._failed.add(key)
            return None
        self.captures += 1
        entry = (graph, static_in, static_out)
        self._graphs[key] = entry
        return entry


def _recover_from_failed_capture(original_stream, pool) -> None:
    """Undo what a failed `torch.cuda.graph` leaves behind.

    Its `__exit__` raises from `capture_end()` BEFORE restoring the thread's
    stream, so every later op on this thread lands on the dead capture stream
    ("operation failed due to a previous error during capture"), and the
    caching allocator is left recording into the pool ("beginAllocateToPool:
    already recording" on the next capture). Both seen on the box.
    """

    if pool is not None:
        try:
            torch._C._cuda_endAllocateToPool(original_stream.device.index, pool)
        except Exception:  # noqa: BLE001 -- already ended is fine
            pass
    torch.cuda.set_stream(original_stream)
    # The capture error is still pending and is reported by the NEXT kernel
    # launch on this thread -- measured: one eager op fails, the one after it
    # succeeds. Spend that report on a throwaway probe, not on real work.
    for _ in range(3):
        try:
            torch.zeros(1, device=original_stream.device).add_(1).item()
            return
        except Exception:  # noqa: BLE001 -- this IS the pending error
            continue


def _rows_of(batch: dict):
    sizes = {value.shape[0] for value in batch.values() if value.dim() > 0}
    return sizes.pop() if len(sizes) == 1 else None


def _pad_rows(batch: dict, rows: int) -> dict:
    real = _rows_of(batch)
    if real == rows:
        return batch
    device = next(iter(batch.values())).device
    index = torch.zeros(rows, dtype=torch.long, device=device)
    index[:real] = torch.arange(real, device=device)
    return {name: value.index_select(0, index) for name, value in batch.items()}


def _is_row_major(value, rows: int) -> bool:
    return isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == rows
