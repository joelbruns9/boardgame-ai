"""Thread-safe request coalescing for Phase D Python self-play workers."""

from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading
import time
from typing import Sequence

from .codec import legal_action_indices
from .encoder import Encoding, encode
from .search import state_actor


@dataclass(slots=True)
class _Request:
    encodings: Sequence[Encoding]
    legal_lists: Sequence[Sequence[int]]
    done: threading.Event = field(default_factory=threading.Event)
    result: list | None = None
    error: BaseException | None = None


class CoalescingEvaluator:
    """Coalesce synchronous calls from game threads into model batches.

    Only the service thread touches the underlying evaluator/model.  Search
    remains synchronous, while concurrent games naturally fill GPU batches.
    """

    def __init__(
        self,
        evaluator,
        *,
        max_batch: int = 64,
        max_wait_ms: float = 2.0,
    ):
        if max_batch <= 0:
            raise ValueError("max_batch must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        self.evaluator = evaluator
        self.max_batch = max_batch
        self.max_wait_s = max_wait_ms / 1000.0
        self._queue: queue.Queue[_Request | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._closed = False
        self.batches = 0
        self.positions = 0

    def start(self) -> "CoalescingEvaluator":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._serve, name="swd-inference", daemon=True
        )
        self._thread.start()
        return self

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            if self._thread.is_alive():
                raise RuntimeError("inference service did not stop")

    def __enter__(self) -> "CoalescingEvaluator":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.close()

    def evaluate(self, encodings, legal_lists):
        if self._closed:
            raise RuntimeError("inference service is closed")
        if len(encodings) != len(legal_lists):
            raise ValueError("encodings and legal_lists must align")
        if not encodings:
            return []
        self.start()
        request = _Request(tuple(encodings), tuple(legal_lists))
        self._queue.put(request)
        request.done.wait()
        if request.error is not None:
            raise request.error
        return request.result

    def evaluate_states(self, states):
        encodings = []
        legals = []
        for state in states:
            actor = state_actor(state)
            encodings.append(encode(state.observation(actor)))
            legals.append(legal_action_indices(state))
        return self.evaluate(encodings, legals)

    def _serve(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return
            requests = [first]
            positions = len(first.encodings)
            deadline = time.monotonic() + self.max_wait_s
            while positions < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    request = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if request is None:
                    self._queue.put(None)
                    break
                requests.append(request)
                positions += len(request.encodings)
            encodings = [item for req in requests for item in req.encodings]
            legal_lists = [item for req in requests for item in req.legal_lists]
            try:
                results = self.evaluator.evaluate(encodings, legal_lists)
                if len(results) != len(encodings):
                    raise RuntimeError("evaluator returned a misaligned result batch")
                offset = 0
                for request in requests:
                    count = len(request.encodings)
                    request.result = results[offset : offset + count]
                    offset += count
                self.batches += 1
                self.positions += len(encodings)
            except BaseException as error:
                for request in requests:
                    request.error = error
            finally:
                for request in requests:
                    request.done.set()
