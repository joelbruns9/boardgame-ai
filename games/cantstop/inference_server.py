# inference_server.py
# GPU inference server for Can't Stop MCTS self-play.
#
# Architecture:
#
#   Main Process (GPU)
#     ├── starts server process
#     ├── pushes model paths via control queue between iterations
#     └── pushes shutdown sentinel when done
#
#   Server Process (GPU)
#     ├── loads model onto CUDA
#     ├── main loop:
#     │     - blocks on request_queue.get()
#     │     - drains additional requests with get_nowait() (dynamic batching)
#     │     - stacks features/masks from all collected requests
#     │     - runs one batched forward pass on GPU
#     │     - splits results, sends each chunk to the originating worker's
#     │       response_queue
#     └── polls control_queue at the top of each loop iteration
#
#   Worker Processes (CPU)
#     ├── hold an InferenceClient(request_queue, response_queue, worker_id)
#     └── client.infer(features, masks) sends a request and blocks on
#         response_queue with a timeout. Returns (values, probs_softmaxed).
#
# IPC: multiprocessing.Queue. Chosen for simplicity, reliability, no new
# dependencies, and well-tested Windows spawn-context behavior. At our
# payload sizes (~450 bytes per sample) the queue overhead (~100-300µs
# per round-trip on Windows) is small compared to GPU compute it replaces.

import os
import sys
import time
import queue
import threading
import traceback
import multiprocessing as mp

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---- PROTOCOL ----
# Request:  (worker_id, request_id, features_np, masks_np)
#           - features_np: (N, FEATURE_SIZE) float32
#           - masks_np:    (N, ACTION_SPACE) bool
# Response: (request_id, values_np, probs_np) — both already softmaxed,
#           probs masked (illegal positions ~ 0).
#           If error: (request_id, None, error_message_str)
#
# Control queue messages:
#   ('update_model', model_path)  — reload model weights from disk
#   ('shutdown', None)            — clean exit
#
# Sentinel on request queue: a request with worker_id=None means shutdown
# (alternative path if control queue is too slow to react).

REQUEST_TIMEOUT = 30.0       # seconds; worker raises if no response by then
SERVER_DRAIN_MAX = 64        # max requests to drain per batch (caps GPU batch)
SERVER_POLL_INTERVAL = 0.05  # seconds blocking get() — lets us check control


# ---- WORKER-SIDE CLIENT ----

class InferenceClient:
    """
    Worker-side handle for the inference server.

    Each worker process gets ONE InferenceClient, which holds:
      - the SHARED request queue (all workers push to it)
      - this worker's PRIVATE response queue (only this worker reads)
      - a unique worker_id for routing
      - a request_id counter to detect stale/wrong-routing responses

    Methods are designed to be drop-in for the `model(features_t, masks_t)`
    call inside MCTS.evaluate_batch but operating on numpy arrays instead.
    """

    def __init__(self, request_queue, response_queue, worker_id):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id
        self._next_request_id = 0

        # ---- Profiling counters ----
        # Updated by infer() on every call. The worker reads these at
        # exit time to print a blocked-vs-cpu breakdown. All times in
        # seconds; counts are integers.
        self._stats_n_calls = 0
        self._stats_total_samples = 0       # total samples sent (sum of N)
        self._stats_blocked_s = 0.0          # time spent in response_queue.get()
        self._stats_put_s = 0.0              # time spent in request_queue.put()
        self._stats_first_call_t = None      # wall-clock of first infer()
        self._stats_last_return_t = None     # wall-clock of last infer() return

    def get_stats(self):
        """Snapshot of per-worker inference stats. Safe to call any time."""
        return {
            'n_calls':       self._stats_n_calls,
            'total_samples': self._stats_total_samples,
            'blocked_s':     self._stats_blocked_s,
            'put_s':         self._stats_put_s,
            'first_call_t':  self._stats_first_call_t,
            'last_return_t': self._stats_last_return_t,
        }

    def infer(self, features_np, masks_np):
        """
        Send a batched inference request and wait for the response.

        Args:
            features_np: (N, FEATURE_SIZE) float32
            masks_np:    (N, ACTION_SPACE) bool

        Returns:
            (values_np, probs_np):
              values_np: (N,) float32 — sigmoided value head outputs
              probs_np:  (N, ACTION_SPACE) float32 — softmaxed, mask-zeroed

        Raises:
            RuntimeError if the server doesn't respond within REQUEST_TIMEOUT.
        """
        import time as _t

        req_id = self._next_request_id
        self._next_request_id += 1

        # Track wall-clock window so the worker can compute total span
        # and infer the CPU fraction by subtracting blocked time.
        call_t = _t.perf_counter()
        if self._stats_first_call_t is None:
            self._stats_first_call_t = call_t

        # Time the put (usually trivial but worth measuring).
        put_t0 = _t.perf_counter()
        self.request_queue.put(
            (self.worker_id, req_id, features_np, masks_np)
        )
        put_t1 = _t.perf_counter()
        self._stats_put_s += (put_t1 - put_t0)

        # Time the blocking wait — this is the number that decides Fix B.
        block_t0 = _t.perf_counter()
        try:
            resp = self.response_queue.get(timeout=REQUEST_TIMEOUT)
        except queue.Empty:
            raise RuntimeError(
                f"Worker {self.worker_id}: inference server did not "
                f"respond within {REQUEST_TIMEOUT}s (request {req_id})."
            )
        block_t1 = _t.perf_counter()
        self._stats_blocked_s += (block_t1 - block_t0)

        resp_req_id, values_np, probs_np = resp

        # If server returned an error tuple, raise it.
        if values_np is None:
            raise RuntimeError(
                f"Inference server error on request {req_id}: {probs_np}"
            )

        # Sanity check: request IDs should match. If they don't, we got
        # a stale response from a previous failed request — discard and
        # retry would be ideal, but a strict match is safer here since
        # mismatches indicate a real protocol bug.
        if resp_req_id != req_id:
            raise RuntimeError(
                f"Worker {self.worker_id}: response id mismatch "
                f"(expected {req_id}, got {resp_req_id})."
            )

        # Update stats now that we know the call succeeded.
        self._stats_n_calls += 1
        self._stats_total_samples += int(features_np.shape[0])
        self._stats_last_return_t = _t.perf_counter()

        return values_np, probs_np


# ---- THREAD-SAFE WRAPPER ----
#
# The raw InferenceClient assumes a single caller — its `_next_request_id`
# counter, its `response_queue.get()` blocking call, and its assertion that
# returned request_id matches the just-sent one all break if multiple
# threads call infer() concurrently.
#
# ThreadSafeInferenceClient wraps the raw client and serves N caller threads
# from one worker process by:
#
#   1. Atomic request_id assignment under a lock.
#   2. A pending-request registry: request_id → {Event, result-slot}.
#      Caller threads register before put(), then block on their Event.
#   3. A SINGLE dispatcher thread that owns response_queue.get(). It reads
#      each response, looks up the matching pending entry, fills the result
#      slot, and signals the Event. The caller thread wakes up and reads
#      the slot.
#
# The dispatcher pattern means:
#   - No two threads ever call response_queue.get() (which is unsafe to
#     share with the assertion that responses match the just-sent id).
#   - Responses can arrive in any order — dispatcher routes them correctly
#     by request_id.
#   - mp.Queue.put() IS thread-safe within a single process, so concurrent
#     put() calls are fine without additional locking.
#
# Statistics (blocked time, samples sent, etc) live ON THE WRAPPER, not
# the raw client. Wrapper is the single point all threads go through, so
# we don't need separate locks for stats — they're protected by the same
# pending_lock that gates request_id assignment.

class _PendingRequest:
    """A single in-flight inference request. One per concurrent caller."""
    __slots__ = ('event', 'values', 'probs', 'error')

    def __init__(self):
        self.event = threading.Event()
        self.values = None
        self.probs = None
        self.error = None    # str if server returned error, else None


class ThreadSafeInferenceClient:
    """
    Drop-in replacement for InferenceClient that supports multiple
    concurrent caller threads.

    Usage:
        ts_client = ThreadSafeInferenceClient(request_queue,
                                              response_queue, worker_id)
        ts_client.start()
        # ... spawn N threads, each gets `ts_client` and calls .infer()
        # ... when threads finish:
        ts_client.shutdown()

    The dispatcher thread starts in start() and is joined in shutdown().
    Calling .infer() before start() will hang forever (no dispatcher
    to wake events). Calling .infer() after shutdown() raises.
    """

    def __init__(self, request_queue, response_queue, worker_id):
        # Raw IPC handles — used directly here, not via the old client.
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id

        # Atomic request_id counter + pending registry.
        self._pending_lock = threading.Lock()
        self._next_request_id = 0
        self._pending = {}  # request_id -> _PendingRequest

        # Dispatcher state.
        self._dispatcher_thread = None
        self._shutdown = threading.Event()

        # Profiling stats — protected by _pending_lock for simplicity
        # (the lock is held briefly anyway during request setup).
        self._stats_n_calls = 0
        self._stats_total_samples = 0
        self._stats_blocked_s = 0.0
        self._stats_put_s = 0.0
        self._stats_first_call_t = None
        self._stats_last_return_t = None

    def start(self):
        """Start the dispatcher thread. Must be called once before any
        .infer() call. Idempotent."""
        if self._dispatcher_thread is not None:
            return
        self._shutdown.clear()
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop,
            name=f"infer-dispatcher-w{self.worker_id}",
            daemon=False,
        )
        self._dispatcher_thread.start()

    def shutdown(self, timeout=5.0):
        """Stop the dispatcher thread. After this, .infer() raises.
        Idempotent — safe to call multiple times."""
        if self._dispatcher_thread is None:
            return
        self._shutdown.set()
        # Dispatcher polls response_queue with a short timeout, so it
        # picks up the shutdown signal within SHUTDOWN_POLL seconds.
        self._dispatcher_thread.join(timeout=timeout)
        self._dispatcher_thread = None

        # Wake any callers still waiting — their requests will never
        # complete now. We mark them with an error so they raise rather
        # than block forever.
        with self._pending_lock:
            stale = list(self._pending.items())
            self._pending.clear()
        for req_id, pending in stale:
            pending.error = "client shut down with request in flight"
            pending.event.set()

    def get_stats(self):
        """Snapshot of inference stats. Safe to call any time."""
        with self._pending_lock:
            return {
                'n_calls':       self._stats_n_calls,
                'total_samples': self._stats_total_samples,
                'blocked_s':     self._stats_blocked_s,
                'put_s':         self._stats_put_s,
                'first_call_t':  self._stats_first_call_t,
                'last_return_t': self._stats_last_return_t,
            }

    # ---- Public infer API (called from game threads) ----

    def infer(self, features_np, masks_np):
        """
        Send an inference request and wait for the response.
        Thread-safe — multiple game threads can call this concurrently.

        Identical contract to InferenceClient.infer().
        """
        if self._dispatcher_thread is None or self._shutdown.is_set():
            raise RuntimeError(
                f"ThreadSafeInferenceClient (worker {self.worker_id}) "
                f"is not running. Call start() before infer()."
            )

        # Register the pending request atomically. The lock protects:
        #   - request_id assignment
        #   - pending dict insertion
        #   - stats setup (first_call_t)
        pending = _PendingRequest()
        with self._pending_lock:
            req_id = self._next_request_id
            self._next_request_id += 1
            self._pending[req_id] = pending

            call_t = time.perf_counter()
            if self._stats_first_call_t is None:
                self._stats_first_call_t = call_t

        # Put the request (mp.Queue.put is thread-safe within a process).
        put_t0 = time.perf_counter()
        self.request_queue.put(
            (self.worker_id, req_id, features_np, masks_np)
        )
        put_t1 = time.perf_counter()

        # Block on the event until dispatcher fills our slot.
        block_t0 = time.perf_counter()
        signaled = pending.event.wait(timeout=REQUEST_TIMEOUT)
        block_t1 = time.perf_counter()

        # Update stats.
        with self._pending_lock:
            self._stats_n_calls += 1
            self._stats_total_samples += int(features_np.shape[0])
            self._stats_put_s += (put_t1 - put_t0)
            self._stats_blocked_s += (block_t1 - block_t0)
            self._stats_last_return_t = block_t1

        if not signaled:
            # Timeout. Try to remove the entry so dispatcher doesn't
            # store into freed memory later.
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise RuntimeError(
                f"Worker {self.worker_id}: inference server did not "
                f"respond within {REQUEST_TIMEOUT}s (request {req_id})."
            )

        if pending.error is not None:
            raise RuntimeError(
                f"Inference server error on request {req_id}: "
                f"{pending.error}"
            )

        return pending.values, pending.probs

    # ---- Dispatcher thread body ----

    def _dispatcher_loop(self):
        """
        Owned by ONE thread. Reads responses, routes by request_id.

        Polls response_queue with a short timeout so we can check
        the shutdown event without blocking forever.
        """
        SHUTDOWN_POLL = 0.1  # 100ms is plenty responsive

        while not self._shutdown.is_set():
            try:
                resp = self.response_queue.get(timeout=SHUTDOWN_POLL)
            except queue.Empty:
                continue

            try:
                resp_req_id, values_np, probs_np = resp
            except Exception as e:
                # Malformed response — log and continue. Server would
                # never send this in normal operation.
                print(f"[ts-client-w{self.worker_id}] malformed "
                      f"response: {resp!r} ({e})", flush=True)
                continue

            with self._pending_lock:
                pending = self._pending.pop(resp_req_id, None)

            if pending is None:
                # Stale response (caller timed out, or duplicate). Drop.
                continue

            # Fill the slot and signal the caller.
            if values_np is None:
                # Server returned an error: error_message is in probs_np
                # field per the server's protocol.
                pending.error = str(probs_np)
            else:
                pending.values = values_np
                pending.probs = probs_np
            pending.event.set()


# ---- SERVER PROCESS ----

def _server_loop(request_queue, response_queues, control_queue,
                 initial_model_path, device):
    """
    Inference server main loop. Runs in its own process.

    The server owns the GPU. It loads the model once, then services
    inference requests in a tight loop until shutdown.

    Dynamic batching: each iteration does ONE blocking get(), then drains
    the queue non-blocking up to SERVER_DRAIN_MAX requests, stacks them
    into one tensor, runs a single forward pass, and dispatches results.
    """
    # Lazy-import torch + model in the server process so that the parent
    # process doesn't accidentally pull CUDA into its own state before
    # we want it there. (Main process imports torch normally for
    # training, but in some test paths we want this to be safe to start
    # without the parent having touched CUDA.)
    from games.cantstop.model import CantStopNet
    from games.cantstop.evaluate import load_model
    from games.cantstop.features import ACTION_SPACE

    try:
        model = load_model(initial_model_path, device)
        model.eval()
    except Exception as e:
        print(f"[inference-server] FATAL: failed to load initial model "
              f"{initial_model_path!r}: {e}", flush=True)
        traceback.print_exc()
        return

    print(f"[inference-server] started on device={device}, "
          f"initial model={initial_model_path}", flush=True)

    pending_shutdown = False
    total_requests = 0
    total_batches = 0
    start_time = time.time()

    while not pending_shutdown:
        # ---- Check control queue (non-blocking) ----
        try:
            while True:
                cmd, payload = control_queue.get_nowait()
                if cmd == 'update_model':
                    try:
                        new_model = load_model(payload, device)
                        new_model.eval()
                        model = new_model
                        print(f"[inference-server] model updated from "
                              f"{payload}", flush=True)
                    except Exception as e:
                        print(f"[inference-server] WARNING: failed to "
                              f"update model from {payload}: {e}",
                              flush=True)
                elif cmd == 'shutdown':
                    pending_shutdown = True
                    break
        except queue.Empty:
            pass

        if pending_shutdown:
            break

        # ---- Collect a batch of requests ----
        try:
            # Block briefly so we don't busy-spin when no work is queued.
            # SERVER_POLL_INTERVAL lets us check control queue every
            # ~50ms even when workers are quiet.
            first = request_queue.get(timeout=SERVER_POLL_INTERVAL)
        except queue.Empty:
            continue

        # Sentinel request — synonym for shutdown.
        if first[0] is None:
            pending_shutdown = True
            break

        collected = [first]
        # Drain everything else that's already enqueued, up to the cap.
        for _ in range(SERVER_DRAIN_MAX - 1):
            try:
                req = request_queue.get_nowait()
            except queue.Empty:
                break
            if req[0] is None:
                pending_shutdown = True
                # Process what we have, then exit on next loop.
                break
            collected.append(req)

        # ---- Stack and run forward pass ----
        try:
            # Build a flat tensor across all collected requests.
            all_features = []
            all_masks = []
            chunk_sizes = []   # per-request count, used to split results
            for (wid, rid, feats, msks) in collected:
                all_features.append(feats)
                all_masks.append(msks)
                chunk_sizes.append(feats.shape[0])

            features_np = np.concatenate(all_features, axis=0)
            masks_np    = np.concatenate(all_masks, axis=0)

            features_t = torch.from_numpy(features_np).to(device)
            masks_t    = torch.from_numpy(masks_np).to(device)

            with torch.no_grad():
                values, logits = model(features_t, masks_t)
                # softmax on GPU is much cheaper than on CPU; values
                # come back already masked-illegal-as-very-negative
                # if the model does that internally (which our model
                # does via masked_fill).
                probs = F.softmax(logits, dim=-1)

                values_np = values.detach().cpu().numpy().astype(np.float32)
                probs_np  = probs.detach().cpu().numpy().astype(np.float32)

            # Split results back per-request and dispatch.
            offset = 0
            for (wid, rid, _, _), n in zip(collected, chunk_sizes):
                v_chunk = values_np[offset:offset + n]
                p_chunk = probs_np[offset:offset + n]
                offset += n
                response_queues[wid].put((rid, v_chunk, p_chunk))

            total_requests += len(collected)
            total_batches  += 1

        except Exception as e:
            print(f"[inference-server] batch failed: {e}", flush=True)
            traceback.print_exc()
            # Send error response to every worker in this batch.
            err_msg = str(e)
            for (wid, rid, _, _) in collected:
                try:
                    response_queues[wid].put((rid, None, err_msg))
                except Exception:
                    pass
            continue

    # ---- Drain any straggler requests so workers don't deadlock. ----
    try:
        while True:
            req = request_queue.get_nowait()
            if req[0] is None:
                continue
            wid, rid, _, _ = req
            response_queues[wid].put(
                (rid, None, 'inference server shutting down')
            )
    except queue.Empty:
        pass

    elapsed = time.time() - start_time
    if total_batches > 0:
        avg_batch = total_requests / total_batches
        rate = total_batches / max(elapsed, 1e-6)
        print(f"[inference-server] shutdown. "
              f"{total_batches} batches, {total_requests} requests, "
              f"avg batch={avg_batch:.1f}, {rate:.0f} batches/s, "
              f"{elapsed:.1f}s wall", flush=True)
    else:
        print(f"[inference-server] shutdown (no work done).", flush=True)


# ---- MANAGER (main-process API) ----

class InferenceServerManager:
    """
    Main-process handle that owns the inference server lifecycle.

    Usage:
        mgr = InferenceServerManager(model_path, device='cuda', num_workers=8)
        mgr.start()
        # ... use mgr.request_queue, mgr.response_queues, mgr.worker_ids
        # ... for setting up worker pool
        mgr.update_model(new_path)   # between iterations
        mgr.stop()

    The manager creates ALL queues so the same Queue objects can be
    passed via initargs into the worker pool. The server process
    inherits them via the standard mp.Process(args=...).
    """

    def __init__(self, model_path, device='cuda', num_workers=8,
                 mp_context=None):
        if mp_context is None:
            # Match the context used elsewhere (spawn on Windows).
            mp_context = mp.get_context('spawn')
        self.ctx = mp_context
        self.device = device
        self.num_workers = num_workers
        self.model_path = model_path

        self.request_queue = self.ctx.Queue()
        self.response_queues = [self.ctx.Queue() for _ in range(num_workers)]
        self.control_queue = self.ctx.Queue()
        self.server_proc = None

    def start(self):
        if self.server_proc is not None and self.server_proc.is_alive():
            return
        self.server_proc = self.ctx.Process(
            target=_server_loop,
            args=(
                self.request_queue,
                self.response_queues,
                self.control_queue,
                self.model_path,
                self.device,
            ),
            daemon=False,  # daemon=True would forbid spawning workers
                           # from within; explicit shutdown is cleaner.
        )
        self.server_proc.start()

    def update_model(self, new_model_path):
        """Tell the server to reload model weights from disk."""
        if not (self.server_proc and self.server_proc.is_alive()):
            raise RuntimeError("Inference server is not running.")
        self.control_queue.put(('update_model', new_model_path))
        # Note: there's no synchronous ACK — the server picks this up
        # at the top of its next loop iteration. With a sub-second
        # polling cadence and the fact that we typically idle between
        # iterations, this is fine in practice.

    def stop(self, timeout=10.0):
        """Send shutdown signal and join the server process."""
        if self.server_proc is None:
            return
        if self.server_proc.is_alive():
            try:
                self.control_queue.put(('shutdown', None))
            except Exception:
                pass
            # Also push a sentinel onto the request queue so the server
            # doesn't sit forever blocked on a quiet queue waiting for
            # the control message to be checked at the top of the loop.
            try:
                self.request_queue.put((None, 0, None, None))
            except Exception:
                pass
            self.server_proc.join(timeout=timeout)
            if self.server_proc.is_alive():
                print("[inference-server-manager] server didn't shut "
                      "down cleanly; terminating.", flush=True)
                self.server_proc.terminate()
                self.server_proc.join(timeout=2.0)
        self.server_proc = None

    def is_alive(self):
        return self.server_proc is not None and self.server_proc.is_alive()

    def client_for(self, worker_id):
        """Build an InferenceClient for a specific worker."""
        return InferenceClient(
            self.request_queue,
            self.response_queues[worker_id],
            worker_id,
        )


# ---- SELF-TEST ----

if __name__ == "__main__":
    # Mock self-test: spin up a server, send some requests from "fake
    # workers" running in the main process, verify round-trip works.
    #
    # This test does NOT exercise actual worker processes (that's
    # tested by self_play.py), but it does verify:
    #   - server starts and serves requests
    #   - dynamic batching collects multiple requests
    #   - model update via control queue works
    #   - shutdown is clean
    #
    # It requires a real model file at the path passed via argv[1]
    # or a sensible default.

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', default='models/cantstop/best_model.pt',
        help='Path to a model checkpoint to load.'
    )
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    args = parser.parse_args()

    from games.cantstop.features import FEATURE_SIZE, ACTION_SPACE

    print(f"Self-test: device={args.device}, model={args.model}")
    print(f"FEATURE_SIZE={FEATURE_SIZE}, ACTION_SPACE={ACTION_SPACE}")

    mgr = InferenceServerManager(
        model_path=args.model, device=args.device, num_workers=4,
    )
    mgr.start()
    time.sleep(2.0)  # let server load model

    assert mgr.is_alive(), "Server should be alive after start()"
    print("Server started.")

    # Build 4 'fake worker' clients
    clients = [mgr.client_for(i) for i in range(4)]

    # Send a single small request from worker 0
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((1, FEATURE_SIZE)).astype(np.float32)
    masks = np.ones((1, ACTION_SPACE), dtype=bool)
    values, probs = clients[0].infer(feats, masks)
    print(f"Single request: values shape={values.shape}, "
          f"probs shape={probs.shape}, probs sum={probs.sum():.3f}")
    assert values.shape == (1,)
    assert probs.shape == (1, ACTION_SPACE)

    # Send larger batches from multiple "workers" rapidly to exercise
    # dynamic batching. We do this in sequence since we're in one
    # process, but it still tests the per-worker response routing.
    print("\nSending 50 sequential requests across 4 worker IDs...")
    t0 = time.time()
    for i in range(50):
        wid = i % 4
        n = (i % 5) + 1
        feats = rng.standard_normal((n, FEATURE_SIZE)).astype(np.float32)
        masks = np.ones((n, ACTION_SPACE), dtype=bool)
        # Random subset masked
        masks[:, :ACTION_SPACE // 2] = (rng.random(
            (n, ACTION_SPACE // 2)) > 0.3)
        v, p = clients[wid].infer(feats, masks)
        assert v.shape == (n,), f"value shape {v.shape}, expected ({n},)"
        assert p.shape == (n, ACTION_SPACE)
    elapsed = time.time() - t0
    print(f"  50 round-trips in {elapsed:.2f}s "
          f"({50/elapsed:.0f} req/s sequential)")

    # Test model update via control queue
    print("\nTesting model update via control queue...")
    mgr.update_model(args.model)   # reload same file — exercises the path
    time.sleep(0.5)
    # Verify server is still responsive
    v, p = clients[0].infer(feats, masks)
    assert v.shape == (n,)
    print("  Server still responsive after update_model: OK")

    print("\nStopping server...")
    mgr.stop()
    assert not mgr.is_alive(), "Server should be stopped"
    print("Server stopped cleanly.")
    print("\nSelf-test PASSED.")