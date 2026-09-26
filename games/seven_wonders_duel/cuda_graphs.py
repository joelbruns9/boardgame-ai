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

import torch

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
        self._pool = None
        self.replays = 0
        self.eager_calls = 0
        self.captures = 0

    def __getattr__(self, name):
        # Callers read model attributes (`action_residual`, head switches)
        # straight off `evaluator.model`; keep answering them.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("model"), name)

    def forward(self, batch: dict) -> dict:
        rows = _rows_of(batch)
        if (
            rows is None
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
            with torch.cuda.graph(graph, pool=self._pool), autocast:
                static_out = self.model(static_in)
        except Exception as error:  # noqa: BLE001 -- any failure means eager
            self._failed.add(key)
            print(f"cuda graphs: capture failed, running eager for {key[0]} rows: "
                  f"{type(error).__name__}: {error}", flush=True)
            return None
        if not isinstance(static_out, dict):
            self._failed.add(key)
            return None
        self.captures += 1
        entry = (graph, static_in, static_out)
        self._graphs[key] = entry
        return entry


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
