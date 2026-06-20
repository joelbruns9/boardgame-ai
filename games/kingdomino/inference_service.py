"""
inference_service.py — unified batched inference for Kingdomino self-play.

ONE client interface, two interchangeable backends
--------------------------------------------------
Game code (and MCTS) only ever sees an ``InferenceClient`` whose call signature
IS the existing Evaluator seam:  (mb, ob, flat) -> (value: float, logits: np.ndarray).
Behind that client sit two backends sharing one batching core, routing,
futures, versioning, stats, and shutdown:

  - LocalInferenceService  (A1): the batcher runs as a thread in THIS process and
    calls the live network directly.  No IPC, no pickling.  Run this first: one
    process, many game threads.

  - RemoteInferenceServer + RemoteInferenceWorkerClient (A3): the batcher runs in
    a dedicated process fed over multiprocessing queues.  The PARENT owns the
    server (queues + process).  Each self-play WORKER PROCESS constructs its own
    RemoteInferenceWorkerClient from the queue handles + its worker_id; that
    client owns a process-local RequestRouter and dispatcher thread reading its
    assigned response queue.  This is the clean "N worker processes x many game
    threads" topology — only queue handles cross the process boundary, never a
    router/lock/thread.

Both backends share ``BatchEvaluator`` (assemble -> forward -> split -> emit, with
versioning + stats) and ``RequestRouter`` (request_id -> future), so A1 -> A3 is a
backend swap, not a rewrite.

Design properties (all present from day one): request_id routing, batched tensor
assembly, per-request futures/events with timeout, model-version tracking
(requested vs applied), clean shutdown, timing metrics, and debug-mode shape /
finiteness validation at the inference boundary.

Does NOT import evaluation.py.
"""
from __future__ import annotations

import copy
import itertools
import queue
import threading
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue as MPQueue, Event as MPEvent
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from games.kingdomino.network import KingdominoNet
from games.kingdomino.encoder import NUM_BOARD_CHANNELS, CANVAS_SIZE, FLAT_SIZE
from games.kingdomino.action_codec import NUM_JOINT_ACTIONS
# Imported as a module (not `from … import MARGIN_GAIN`) so the leaf-value blend
# constants are read at call time — config overrides (mcts_az.MARGIN_GAIN =
# cfg.margin_gain, set at training startup) then propagate here with no extra
# wiring.  No circular import: mcts_az does not import inference_service.
import games.kingdomino.mcts_az as _mcts_az_module

_MB_SHAPE = (NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE)
_FLAT_SIZE = FLAT_SIZE   # computed in encoder; never hardcode (layout may change)


def clone_state_dict(state_dict: dict) -> dict:
    """Detached, CPU, independent-storage copy of a state dict.

    state_dict() tensors share storage with live parameters, so queuing one and
    then continuing to train would let the queued weights reflect *later*
    mutations.  Cloning onto CPU severs that link; the batcher loads onto its own
    device.
    """
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Protocol objects
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class InferenceResult:
    value: float
    logits: np.ndarray             # gathered LEGAL logits (n_legal,) float32
    model_version: int             # weight version this was ACTUALLY computed under


@dataclass
class BatchedInferenceResult:
    """Future payload for a batched (K-leaf) request — one per infer_batch call."""
    values: np.ndarray             # (K,) float
    logits_list: List[np.ndarray]  # K gathered LEGAL logit arrays
    model_version: int


class _PendingRequest:
    __slots__ = ("event", "result", "cancelled")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: Optional[InferenceResult] = None
        self.cancelled = False


# ─────────────────────────────────────────────────────────────────────────────
# Request routing (shared by both backends)
# ─────────────────────────────────────────────────────────────────────────────
class RequestRouter:
    """Thread-safe table mapping request_id -> _PendingRequest."""

    def __init__(self) -> None:
        self._table: Dict[int, _PendingRequest] = {}
        self._lock = threading.Lock()
        # Late/lost results: complete() called for an id no longer in the table
        # (already timed out, cancelled, or shut down).  A nonzero value is a
        # useful signal that timeouts or post-shutdown results are occurring.
        self.dropped_completions = 0

    def register(self, request_id: int) -> _PendingRequest:
        pending = _PendingRequest()
        with self._lock:
            self._table[request_id] = pending
        return pending

    def complete(self, request_id: int, result: InferenceResult) -> None:
        with self._lock:
            pending = self._table.pop(request_id, None)
            if pending is None:
                self.dropped_completions += 1
        if pending is not None:
            pending.result = result
            pending.event.set()

    def cancel(self, request_id: int) -> None:
        """Drop one request (e.g. it timed out) so its future doesn't leak."""
        with self._lock:
            pending = self._table.pop(request_id, None)
        if pending is not None:
            pending.cancelled = True
            pending.event.set()

    def cancel_all(self) -> None:
        with self._lock:
            pendings = list(self._table.values())
            self._table.clear()
        for p in pendings:
            p.cancelled = True
            p.event.set()


# ─────────────────────────────────────────────────────────────────────────────
# The client (the only thing game code / MCTS sees)
# ─────────────────────────────────────────────────────────────────────────────
# submit(request_id, worker_id, mb[K], ob[K], flat[K], idxs_list[K], batched).
# Always group-shaped: K=1 for infer(), K=N for infer_batch(). `batched` tells the
# backend which response type to emit so the future gets the right payload.
SubmitFn = Callable[
    [int, int, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], bool], None]


class InferenceClient:
    """Blocking, thread-safe evaluator backed by a batching service.

    Satisfies the MCTS Evaluator seam: callable as (mb, ob, flat) -> (value,
    logits).  Many game threads may share one client; each infer() owns its own
    future, so concurrent calls do not collide.

    IMMUTABILITY CONTRACT: the caller must not mutate mb/ob/flat until infer()
    returns.  MCTS passes freshly-encoded arrays it never reuses, so no copy is
    made on the hot path.  (The remote backend pickles on submit, isolating
    anyway.)
    """

    def __init__(self, submit: SubmitFn, router: RequestRouter,
                 version_getter: Callable[[], int], worker_id: int = 0,
                 default_timeout_s: float = 60.0, debug_checks: bool = True) -> None:
        self._submit = submit
        self._router = router
        self._version_getter = version_getter
        self._worker_id = worker_id
        self._counter = itertools.count()
        self._counter_lock = threading.Lock()
        self.default_timeout_s = default_timeout_s
        self.debug_checks = debug_checks

    def _next_request_id(self) -> int:
        with self._counter_lock:
            n = next(self._counter)
        return (self._worker_id << 40) | (n & ((1 << 40) - 1))

    def infer(self, mb: np.ndarray, ob: np.ndarray, flat: np.ndarray,
              idxs: np.ndarray,
              timeout_s: Optional[float] = None) -> Tuple[float, np.ndarray]:
        if self.debug_checks:
            assert mb.shape == _MB_SHAPE, f"mb shape {mb.shape} != {_MB_SHAPE}"
            assert ob.shape == _MB_SHAPE, f"ob shape {ob.shape} != {_MB_SHAPE}"
            assert flat.shape == (_FLAT_SIZE,), f"flat shape {flat.shape}"
            assert idxs.ndim == 1 and idxs.size > 0, f"idxs shape {idxs.shape}"

        timeout = self.default_timeout_s if timeout_s is None else timeout_s
        rid = self._next_request_id()
        pending = self._router.register(rid)
        try:
            # K=1 group; backend unwraps to a single-leaf message (bit-identical
            # to the pre-batched protocol).
            self._submit(rid, self._worker_id,
                         mb[None], ob[None], flat[None], [idxs], False)
        except Exception:
            self._router.cancel(rid)   # don't leak the future on a failed submit
            raise

        if not pending.event.wait(timeout):
            self._router.cancel(rid)
            raise TimeoutError(
                f"Inference request {rid} timed out after {timeout}s "
                f"(batcher dead, server crashed, or request lost?)."
            )
        if pending.cancelled or pending.result is None:
            raise RuntimeError("Inference cancelled (service shutting down).")
        return pending.result.value, pending.result.logits

    __call__ = infer

    def infer_batch(self, leaves: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                    timeout_s: Optional[float] = None
                    ) -> List[Tuple[float, np.ndarray]]:
        """Evaluate K leaves in ONE request/response round trip.

        leaves: list of (mb, ob, flat, idxs).  Returns a list aligned with
        `leaves` of (value, gathered_legal_logits) — identical results to calling
        infer() on each leaf, but with a single submit, a single future, and a
        single response, so the per-request IPC cost amortizes over K leaves.
        """
        if not leaves:
            return []
        if self.debug_checks:
            for mb, ob, flat, idxs in leaves:
                assert mb.shape == _MB_SHAPE, f"mb shape {mb.shape} != {_MB_SHAPE}"
                assert ob.shape == _MB_SHAPE, f"ob shape {ob.shape} != {_MB_SHAPE}"
                assert flat.shape == (_FLAT_SIZE,), f"flat shape {flat.shape}"
                assert idxs.ndim == 1 and idxs.size > 0, f"idxs shape {idxs.shape}"

        timeout = self.default_timeout_s if timeout_s is None else timeout_s
        mb = np.stack([l[0] for l in leaves])
        ob = np.stack([l[1] for l in leaves])
        flat = np.stack([l[2] for l in leaves])
        idxs_list = [l[3] for l in leaves]

        rid = self._next_request_id()
        pending = self._router.register(rid)
        try:
            self._submit(rid, self._worker_id, mb, ob, flat, idxs_list, True)
        except Exception:
            self._router.cancel(rid)
            raise

        if not pending.event.wait(timeout):
            self._router.cancel(rid)
            raise TimeoutError(
                f"Batched inference request {rid} ({len(leaves)} leaves) timed "
                f"out after {timeout}s.")
        if pending.cancelled or pending.result is None:
            raise RuntimeError("Inference cancelled (service shutting down).")
        res: BatchedInferenceResult = pending.result
        return [(float(res.values[i]), res.logits_list[i])
                for i in range(len(leaves))]

    @property
    def model_version(self) -> int:
        """The weight version actually serving inference (trustworthy)."""
        return self._version_getter()


def make_ipc_batched_evaluator(client: InferenceClient):
    """Adapt an InferenceClient into the MCTS BatchedEvaluator seam:
        (mbs (K,C,H,W), obs (K,C,H,W), flats (K,F), idxs_list[K])
          -> (values (K,), [gathered_legal_logits])
    One infer_batch round trip per call: K leaves cross the IPC boundary in a
    single request/response, so the per-leaf queue/pickle cost amortizes over K.
    Used by AlphaZeroMCTS._simulate_batch (leaf_batch>1)."""
    def ev(mbs: np.ndarray, obs: np.ndarray, flats: np.ndarray,
           idxs_list: List[np.ndarray]):
        leaves = [(mbs[i], obs[i], flats[i], idxs_list[i])
                  for i in range(len(idxs_list))]
        results = client.infer_batch(leaves)
        values = np.array([r[0] for r in results], dtype=np.float64)
        gathered = [r[1] for r in results]
        return values, gathered
    return ev


# ─────────────────────────────────────────────────────────────────────────────
# Batching core (shared by local thread AND remote process)
# ─────────────────────────────────────────────────────────────────────────────
# A pulled GROUP of K>=1 leaves sharing one request_id/future:
#   (rid, wid, mb[K,C,H,W], ob[K,C,H,W], flat[K,F], idxs_list(len K), batched)
# Single-leaf infer() produces K=1 groups; infer_batch() produces K=N. The
# `batched` flag tells emit which response/payload type to build.
PulledGroup = Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], bool]
PullFn = Callable[[float], Optional[PulledGroup]]
# emit(request_id, worker_id, values[K], gathered_logits_list[K], batched, version)
EmitFn = Callable[[int, int, np.ndarray, List[np.ndarray], bool, int], None]


class BatchEvaluator:
    """Collect requests into a batch, run one forward, split, emit results.

    Identical whether driven from an in-process queue (A1) or a multiprocessing
    queue (A3); only pull_one and emit differ.  Owns the network and the
    (applied) model-version counter; applies weight updates between batches so a
    forward never races load_state_dict.
    """

    def __init__(self, net: torch.nn.Module, device: str,
                 max_batch: int = 32, max_wait_ms: float = 3.0,
                 debug_checks: bool = True,
                 margin_gain: float = _mcts_az_module.MARGIN_GAIN,
                 alpha: float = _mcts_az_module.ALPHA) -> None:
        self.net = net.to(device).eval()
        self.device = device
        self.max_batch = max(1, int(max_batch))
        self.max_wait_s = max(0.0, max_wait_ms / 1000.0)
        self.debug_checks = debug_checks
        # Fix 2: leaf-value blend params bound at construction (no module-global
        # reads in _forward_and_emit).  Callers forward cfg.margin_gain/cfg.alpha.
        self._margin_gain = float(margin_gain)
        self._alpha = float(alpha)

        self.model_version = 0          # APPLIED version (weights actually loaded)
        self._pending_weights: Optional[dict] = None
        self._weights_lock = threading.Lock()

        self._stats_lock = threading.Lock()
        self.reset_stats()

    # ── weights ──
    def apply_weights(self, state_dict: dict, version: Optional[int] = None) -> None:
        """Queue a (cloned) weight update; applied by the run loop between batches.

        If `version` is given (remote path), the applied model_version is SET to
        it, so the version stays consistent even when intermediate updates are
        coalesced/dropped in a bounded weight queue.  If None (local path), the
        version simply increments per load.
        """
        with self._weights_lock:
            self._pending_weights = (clone_state_dict(state_dict), version)

    def _maybe_update_weights(self) -> Optional[int]:
        with self._weights_lock:
            pending = self._pending_weights
            self._pending_weights = None
        if pending is not None:
            sd, version = pending
            self.net.load_state_dict(sd)
            self.net.eval()
            with self._stats_lock:
                if version is None:
                    self.model_version += 1
                else:
                    self.model_version = version
                self.n_weight_updates += 1
                return self.model_version
        return None

    # ── stats ──
    def reset_stats(self) -> None:
        with self._stats_lock:
            self._t_start = time.monotonic()
            self.n_requests = 0
            self.n_batches = 0
            self.sum_batch = 0
            self.max_batch_seen = 0
            self.batch_processing_time_s = 0.0   # stack+transfer+forward+emit
            self.wait_time_s = 0.0               # idle (first req) + batch-fill wait
            self.n_weight_updates = 0

    def _record_batch(self, size: int, proc_dt: float) -> None:
        with self._stats_lock:
            self.n_batches += 1
            self.n_requests += size
            self.sum_batch += size
            if size > self.max_batch_seen:
                self.max_batch_seen = size
            self.batch_processing_time_s += proc_dt

    def _record_wait(self, dt: float) -> None:
        with self._stats_lock:
            self.wait_time_s += dt

    def stats(self) -> dict:
        with self._stats_lock:
            elapsed = max(1e-9, time.monotonic() - self._t_start)
            mean_batch = self.sum_batch / self.n_batches if self.n_batches else 0.0
            busy_denom = self.batch_processing_time_s + self.wait_time_s
            return {
                "requests": self.n_requests,
                "batches": self.n_batches,
                "mean_batch": mean_batch,
                "max_batch_seen": self.max_batch_seen,
                "max_batch_cap": self.max_batch,
                # near 1.0 => batches full => more threads won't help much;
                # near 0   => GPU starved => add game threads.
                "fill_ratio": (mean_batch / self.max_batch) if self.max_batch else 0.0,
                "requests_per_sec": self.n_requests / elapsed,
                # SERVICE busy (stack+transfer+forward+emit), not pure GPU.  Counts
                # both first-request idle AND batch-fill wait in the denominator.
                # near 1.0 + low throughput => tree-work bound => consider A3.
                "service_busy_fraction": (self.batch_processing_time_s / busy_denom)
                                          if busy_denom else 0.0,
                "batch_processing_time_s": self.batch_processing_time_s,
                "wait_time_s": self.wait_time_s,
                "model_version": self.model_version,
                "weight_updates": self.n_weight_updates,
            }

    # ── main loop ──
    def run(self, pull_one: PullFn, emit: EmitFn, stop_event,
            on_weights_applied: Optional[Callable[[int], None]] = None) -> None:
        while not stop_event.is_set():
            applied = self._maybe_update_weights()
            if applied is not None and on_weights_applied is not None:
                on_weights_applied(applied)

            t0 = time.monotonic()
            first = pull_one(self.max_wait_s if self.max_wait_s > 0 else 0.05)
            wait_dt = time.monotonic() - t0
            if first is None:
                self._record_wait(wait_dt)
                continue

            batch: List[PulledGroup] = [first]
            n_leaves = first[2].shape[0]
            deadline = time.monotonic() + self.max_wait_s
            t_fill0 = time.monotonic()
            while n_leaves < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                nxt = pull_one(remaining)
                if nxt is None:
                    break
                batch.append(nxt)
                n_leaves += nxt[2].shape[0]
            # account first-request idle + time spent filling the batch
            self._record_wait(wait_dt + (time.monotonic() - t_fill0))

            self._forward_and_emit(batch, emit)

    def _forward_and_emit(self, batch: List[PulledGroup], emit: EmitFn) -> None:
        t0 = time.monotonic()
        # Each group's arrays are already (K, ...) stacks (K=1 for single leaves),
        # so concatenation builds the full GPU batch across groups. For single-leaf
        # -only traffic this is identical to the old np.stack of (C,H,W) slabs.
        mb = np.concatenate([g[2] for g in batch]).astype(np.float32, copy=False)
        ob = np.concatenate([g[3] for g in batch]).astype(np.float32, copy=False)
        flat = np.concatenate([g[4] for g in batch]).astype(np.float32, copy=False)
        total = mb.shape[0]
        with torch.inference_mode():
            own, opp, win_prob, logits = self.net(
                torch.from_numpy(mb).to(self.device),
                torch.from_numpy(ob).to(self.device),
                torch.from_numpy(flat).to(self.device),
            )
            # Full leaf value formula.  Computed server-side so the IPC wire still
            # carries one scalar per leaf (own/opp/win_prob never cross the process
            # boundary).  margin_gain/alpha are bound at BatchEvaluator construction
            # (Fix 2) from cfg — no module-global reads here.
            mg = self._margin_gain
            al = self._alpha
            margin_val = torch.tanh((own - opp) * mg)          # (total[,1])
            win_val    = 2.0 * win_prob - 1.0                  # (total[,1])
            values_t   = al * margin_val + (1.0 - al) * win_val
            values = values_t.detach().cpu().numpy().reshape(-1)   # (total,)
            logits_np = logits.detach().cpu().numpy()              # (total, A)

        if self.debug_checks:
            assert logits_np.shape == (total, NUM_JOINT_ACTIONS), \
                f"logits shape {logits_np.shape} != {(total, NUM_JOINT_ACTIONS)}"
            assert np.isfinite(values).all(), "non-finite value from network"
            assert np.isfinite(logits_np).all(), "non-finite logits from network"

        version = self.model_version
        offset = 0
        for (rid, wid, gmb, _gob, _gflat, idxs_list, batched) in batch:
            k = gmb.shape[0]
            vals = values[offset:offset + k]
            # Gather each leaf's legal logits (bit-identical to the caller indexing
            # the full vector) so responses carry only sum(n_legal) floats.
            gathered = [logits_np[offset + j][idxs_list[j]] for j in range(k)]
            emit(rid, wid, vals, gathered, batched, version)
            offset += k
        # record LEAVES processed (== GPU batch size), so evals/s and mean_batch
        # keep their per-leaf meaning across single and batched requests.
        self._record_batch(total, time.monotonic() - t0)


# ─────────────────────────────────────────────────────────────────────────────
# Backend A1 — local, in-process
# ─────────────────────────────────────────────────────────────────────────────
class LocalInferenceService:
    """In-process batching service (A1).

    DEEP-COPIES the net it is given so it truly owns an independent inference
    network — weights change only via update_weights, never via shared mutation
    with a trainer.  This makes the semantics identical to the remote backend.
    """

    def __init__(self, net: torch.nn.Module, device: str = "cpu",
                 max_batch: int = 32, max_wait_ms: float = 3.0,
                 debug_checks: bool = True,
                 margin_gain: float = _mcts_az_module.MARGIN_GAIN,
                 alpha: float = _mcts_az_module.ALPHA) -> None:
        own_net = copy.deepcopy(net)
        self._router = RequestRouter()
        # Unbounded by type, but structurally bounded at runtime: every game
        # thread submits exactly one request then blocks on its future, so
        # in-flight requests can never exceed the number of concurrent game
        # threads.  An explicit maxsize would add deadlock risk (if set below the
        # thread count) with no benefit in this blocking-evaluator pattern.
        self._req_q: "queue.Queue[PulledRequest]" = queue.Queue()
        self._stop = threading.Event()
        self._closed = False
        self._debug_checks = debug_checks
        self._next_client_id = itertools.count()
        # Versioned weight updates + barrier (symmetric with RemoteInferenceServer)
        self._requested_version = 0
        self._applied_version = 0
        self._version_cond = threading.Condition()
        self.batcher = BatchEvaluator(own_net, device, max_batch, max_wait_ms,
                                      debug_checks=debug_checks,
                                      margin_gain=margin_gain, alpha=alpha)
        self._thread = threading.Thread(
            target=self._run, name="LocalInferenceBatcher", daemon=True)
        self._started = False

    def _on_weights_applied(self, version: int) -> None:
        with self._version_cond:
            self._applied_version = version
            self._version_cond.notify_all()

    def _run(self) -> None:
        def pull_one(timeout: float) -> Optional[PulledRequest]:
            try:
                return self._req_q.get(timeout=timeout)
            except queue.Empty:
                return None

        def emit(rid: int, _wid: int, values, gathered_list, batched: bool,
                 version: int) -> None:
            if batched:
                self._router.complete(rid, BatchedInferenceResult(
                    values, gathered_list, version))
            else:
                self._router.complete(rid, InferenceResult(
                    float(values[0]), gathered_list[0], version))

        self.batcher.run(pull_one, emit, self._stop,
                         on_weights_applied=self._on_weights_applied)

    def start(self) -> "LocalInferenceService":
        if not self._started:
            self._thread.start()
            self._started = True
        return self

    def make_client(self, worker_id: Optional[int] = None) -> InferenceClient:
        # Auto-assign a unique id when not given, so independent clients never
        # share a request-id prefix (which would collide in the router).
        if worker_id is None:
            worker_id = next(self._next_client_id)

        def submit(rid: int, wid: int, mb, ob, flat, idxs_list, batched) -> None:
            if self._closed:
                raise RuntimeError("Service is closed; cannot submit inference.")
            # Already group-shaped ((K,...) stacks + len-K idxs_list); enqueue as
            # a PulledGroup. No pickling (in-process), so this is cheap.
            self._req_q.put((rid, wid, mb, ob, flat, idxs_list, batched))

        return InferenceClient(submit, self._router,
                               lambda: self._applied_version, worker_id,
                               debug_checks=self._debug_checks)

    def update_weights(self, state_dict: dict) -> int:
        """Queue cloned, version-tagged weights; returns the REQUESTED version.
        Use wait_for_version to barrier until it is actually serving."""
        self._requested_version += 1
        v = self._requested_version
        self.batcher.apply_weights(state_dict, version=v)   # clones internally
        return v

    def wait_for_version(self, version: int, timeout_s: float = 30.0) -> bool:
        """Block until the batcher has applied at least `version`.  True if
        reached, False on timeout."""
        with self._version_cond:
            return self._version_cond.wait_for(
                lambda: self._applied_version >= version, timeout=timeout_s)

    def stats(self) -> dict:
        return self.batcher.stats()

    def reset_stats(self) -> None:
        self.batcher.reset_stats()

    def stop(self) -> None:
        self._closed = True
        self._router.cancel_all()        # wake any waiters promptly
        self._stop.set()
        if self._started:
            self._thread.join(timeout=5)

    def __enter__(self) -> "LocalInferenceService":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Backend A3 — remote, multiprocess (graduation path)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _RemoteRequest:
    request_id: int
    worker_id: int
    mb: np.ndarray
    ob: np.ndarray
    flat: np.ndarray
    idxs: np.ndarray               # legal joint-action indices (int64)


@dataclass
class _RemoteResult:
    request_id: int
    worker_id: int
    value: float
    logits: np.ndarray             # gathered LEGAL logits (n_legal,)
    model_version: int


@dataclass
class _RemoteBatchRequest:
    """K leaves in one message — amortizes the per-request queue/pickle cost."""
    request_id: int
    worker_id: int
    mb: np.ndarray                 # (K, C, H, W) float32
    ob: np.ndarray                 # (K, C, H, W) float32
    flat: np.ndarray               # (K, F) float32
    idxs_list: List[np.ndarray]    # K legal-index arrays (int64), variable length


@dataclass
class _RemoteBatchResult:
    request_id: int
    worker_id: int
    values: np.ndarray             # (K,)
    logits_list: List[np.ndarray]  # K gathered LEGAL logit arrays
    model_version: int


def _remote_server_loop(model_kwargs: dict, device: str,
                        max_batch: int, max_wait_ms: float,
                        request_q: MPQueue, response_qs: List[MPQueue],
                        weight_q: MPQueue, weight_ack_q: MPQueue,
                        stop_event, debug_checks: bool,
                        stats_q: Optional[MPQueue] = None,
                        margin_gain: float = _mcts_az_module.MARGIN_GAIN,
                        alpha: float = _mcts_az_module.ALPHA) -> None:
    """Server process entry point: builds the EXACT net from full model_kwargs and
    runs the shared BatchEvaluator.  Module-level so it is picklable under spawn."""
    net = KingdominoNet(**model_kwargs)
    batcher = BatchEvaluator(net, device, max_batch, max_wait_ms,
                             debug_checks=debug_checks,
                             margin_gain=margin_gain, alpha=alpha)

    _last_push = [0.0]
    _STATS_PUSH_INTERVAL = 0.5   # seconds; diagnostic cadence, not exact

    def pull_one(timeout: float) -> Optional[PulledGroup]:
        try:
            version, sd = weight_q.get_nowait()
            batcher.apply_weights(sd, version=version)
        except Exception:
            pass
        try:
            r = request_q.get(timeout=timeout)
        except Exception:
            return None
        if isinstance(r, _RemoteBatchRequest):
            return (r.request_id, r.worker_id, r.mb, r.ob, r.flat, r.idxs_list, True)
        # single-leaf _RemoteRequest -> K=1 group
        return (r.request_id, r.worker_id, r.mb[None], r.ob[None], r.flat[None],
                [r.idxs], False)

    def emit(rid: int, wid: int, values, gathered_list, batched: bool,
             version: int) -> None:
        if batched:
            response_qs[wid].put(_RemoteBatchResult(
                rid, wid, values, gathered_list, version))
        else:
            response_qs[wid].put(_RemoteResult(
                rid, wid, float(values[0]), gathered_list[0], version))
        # Throttled stats publish so the parent can read recent batch/throughput
        # numbers without a request/response round-trip.
        if stats_q is not None:
            now = time.monotonic()
            if now - _last_push[0] >= _STATS_PUSH_INTERVAL:
                _last_push[0] = now
                try:
                    stats_q.put_nowait(batcher.stats())
                except Exception:
                    pass
        # Throttled stats publish so the parent can read recent batch/throughput
        # numbers without a request/response round-trip.  Cheap: at most one dict
        # per _STATS_PUSH_INTERVAL, and only while inference is active.
        if stats_q is not None:
            now = time.monotonic()
            if now - _last_push[0] >= _STATS_PUSH_INTERVAL:
                _last_push[0] = now
                try:
                    stats_q.put_nowait(batcher.stats())
                except Exception:
                    pass

    def on_weights_applied(version: int) -> None:
        try:
            weight_ack_q.put_nowait(version)
        except Exception:
            pass
        # A weight apply marks a training-iteration boundary in the parent loop,
        # so reset per-window stats here.  This makes get_stats() report numbers
        # for the current iteration's self-play rather than the whole run.
        batcher.reset_stats()

    batcher.run(pull_one, emit, stop_event, on_weights_applied=on_weights_applied)


class RemoteInferenceServer:
    """Parent-owned multiprocess batching server (A3).

    Owns the queues, the weight channel, and the server process.  Does NOT own
    routers or dispatcher threads — those live in each worker via
    RemoteInferenceWorkerClient, constructed from worker_handles() inside the
    worker process.  Only queue handles cross the process boundary.
    """

    def __init__(self, n_workers: int, model_kwargs: dict, device: str = "cuda",
                 max_batch: int = 64, max_wait_ms: float = 3.0,
                 debug_checks: bool = True,
                 margin_gain: float = _mcts_az_module.MARGIN_GAIN,
                 alpha: float = _mcts_az_module.ALPHA) -> None:
        self.n_workers = n_workers
        self.model_kwargs = dict(model_kwargs)
        self.request_q: MPQueue = MPQueue()
        self.response_qs: List[MPQueue] = [MPQueue() for _ in range(n_workers)]
        self._weight_q: MPQueue = MPQueue(maxsize=2)
        self._weight_ack_q: MPQueue = MPQueue()
        self._stats_q: MPQueue = MPQueue()
        self.latest_stats: dict = {}
        self._stop: MPEvent = MPEvent()
        self.requested_version = 0
        self.applied_version = 0
        self._proc = Process(
            target=_remote_server_loop,
            args=(self.model_kwargs, device, max_batch, max_wait_ms,
                  self.request_q, self.response_qs, self._weight_q,
                  self._weight_ack_q, self._stop, debug_checks, self._stats_q,
                  float(margin_gain), float(alpha)),
            daemon=True, name="RemoteInferenceServer")
        self._started = False

    def start(self, initial_state_dict: Optional[dict] = None,
              wait_until_loaded: bool = True, timeout_s: float = 30.0
              ) -> "RemoteInferenceServer":
        """Start the server.  Pass initial weights or the server serves RANDOM
        weights until the first update_weights — so initial_state_dict is
        strongly recommended."""
        if self._started:
            return self
        self._proc.start()
        self._started = True
        if initial_state_dict is not None:
            v = self.update_weights(initial_state_dict)
            if wait_until_loaded:
                self.wait_for_version(v, timeout_s)
        return self

    def worker_handles(self) -> Tuple[MPQueue, List[MPQueue]]:
        """The handles to pass into a worker process so it can build its own
        RemoteInferenceWorkerClient.  These are picklable; routers/threads are not."""
        return self.request_q, self.response_qs

    def update_weights(self, state_dict: dict) -> int:
        """Queue cloned, version-tagged weights; returns the REQUESTED version
        (not yet applied — use wait_for_version to confirm).  Tagging keeps the
        applied version consistent with the requested numbering even if
        intermediate updates are dropped when the queue is full."""
        self.requested_version += 1
        v = self.requested_version
        sd = clone_state_dict(state_dict)
        if self._weight_q.full():
            try:
                self._weight_q.get_nowait()
            except Exception:
                pass
        self._weight_q.put((v, sd))
        return v

    def wait_for_version(self, version: int, timeout_s: float = 30.0) -> bool:
        """Block until the server acks applying at least `version`.  True if
        reached, False on timeout."""
        deadline = time.monotonic() + timeout_s
        while self.applied_version < version:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                v = self._weight_ack_q.get(timeout=remaining)
                self.applied_version = max(self.applied_version, v)
            except Exception:
                return False
        return True

    def get_stats(self) -> dict:
        """Most recent batch/throughput snapshot published by the server process
        (mean_batch, fill_ratio, requests_per_sec, service_busy_fraction, ...).

        Drains the stats channel keeping only the newest entry, so repeated calls
        reflect the latest snapshot.  Stats reset when weights are applied (each
        training iteration), so a call after an iteration's self-play reports that
        iteration's inference behaviour.  Returns {} until the server has
        published at least once (i.e. once some inference has run this window)."""
        while True:
            try:
                self.latest_stats = self._stats_q.get_nowait()
            except Exception:
                break
        return dict(self.latest_stats)

    def stop(self) -> None:
        self._stop.set()
        if self._started:
            self._proc.join(timeout=10)
            if self._proc.is_alive():
                self._proc.terminate()

    def __enter__(self) -> "RemoteInferenceServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


class RemoteInferenceWorkerClient:
    """Constructed INSIDE each self-play worker process from the server's queue
    handles.  Owns a process-local RequestRouter and dispatcher thread reading
    this worker's response queue, and submits to the shared request queue.

    One client per worker process (shared across that process's game threads):
    worker_id is the response-queue index, so two clients with the same worker_id
    would consume each other's responses.  Share one client per process.
    """

    def __init__(self, request_q: MPQueue, response_qs: List[MPQueue],
                 worker_id: int, default_timeout_s: float = 60.0,
                 debug_checks: bool = True) -> None:
        self._request_q = request_q
        self._response_q = response_qs[worker_id]
        self._worker_id = worker_id
        self._router = RequestRouter()
        self._applied_version = 0       # max version observed in results (trustworthy)
        self._stop = threading.Event()
        self._closed = False
        self._default_timeout_s = default_timeout_s
        self._debug_checks = debug_checks
        self._client_created = False
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, name=f"InferDispatch-{worker_id}",
            daemon=True)
        self._dispatcher.start()

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                res = self._response_q.get(timeout=0.2)
            except Exception:
                continue
            if res.model_version > self._applied_version:
                self._applied_version = res.model_version
            if isinstance(res, _RemoteBatchResult):
                self._router.complete(res.request_id, BatchedInferenceResult(
                    res.values, res.logits_list, res.model_version))
            else:
                self._router.complete(res.request_id, InferenceResult(
                    res.value, res.logits, res.model_version))

    def make_client(self) -> InferenceClient:
        # One shared client per worker process: worker_id is the response-queue
        # index, so a second client with the same id would consume the first's
        # responses and start a colliding request-id counter.  Share this client
        # across the worker's game threads (it is thread-safe).
        if self._client_created:
            raise RuntimeError(
                "RemoteInferenceWorkerClient serves one shared client per worker "
                "process; reuse it across game threads instead of creating more.")
        self._client_created = True

        def submit(rid: int, wid: int, mb, ob, flat, idxs_list, batched) -> None:
            if self._closed:
                raise RuntimeError("Worker client closed; cannot submit.")
            if batched:
                self._request_q.put(_RemoteBatchRequest(
                    rid, wid,
                    np.ascontiguousarray(mb, np.float32),
                    np.ascontiguousarray(ob, np.float32),
                    np.ascontiguousarray(flat, np.float32),
                    [np.ascontiguousarray(ix, np.int64) for ix in idxs_list]))
            else:
                # K=1 group -> single-leaf message (unwrap leading axis); the
                # single-leaf wire format is byte-identical to the pre-batch protocol.
                self._request_q.put(_RemoteRequest(
                    rid, wid,
                    np.ascontiguousarray(mb[0], np.float32),
                    np.ascontiguousarray(ob[0], np.float32),
                    np.ascontiguousarray(flat[0], np.float32),
                    np.ascontiguousarray(idxs_list[0], np.int64)))
        return InferenceClient(submit, self._router,
                               lambda: self._applied_version, self._worker_id,
                               default_timeout_s=self._default_timeout_s,
                               debug_checks=self._debug_checks)

    def stop(self) -> None:
        self._closed = True
        self._router.cancel_all()
        self._stop.set()
        self._dispatcher.join(timeout=2)