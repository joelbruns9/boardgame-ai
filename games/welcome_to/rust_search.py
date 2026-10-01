"""Packed Python boundary for the Rust-owned Welcome To search and M6 scheduler.

The native search owns states, observation keys, the tree and all accumulated
statistics.  This module is deliberately small: it turns M3's four packed
little-endian buffers into one Torch row and returns the frozen M0-E response.
M6 concatenates those rows into the frozen V2 batch-major representation; the
single-request M5 seam remains as a diagnostic control.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Sequence

import numpy as np
import torch

from games.welcome_to import encoder as enc
from games.welcome_to import mcts
from games.welcome_to import network as nw
from games.welcome_to import training
from games.welcome_to.portable_rng import PortableRng

try:
    import welcome_to_rust as wr
except ImportError:  # pragma: no cover - source checkouts need no Rust toolchain
    wr = None


LEAF = 0
POLICY = 1
EVALUATOR_ABI_VERSION = 3

if wr is not None and wr.EVALUATOR_ABI_VERSION != EVALUATOR_ABI_VERSION:
    raise ImportError(
        "welcome_to_rust evaluator ABI "
        f"{wr.EVALUATOR_ABI_VERSION} != Python {EVALUATOR_ABI_VERSION}; "
        "rebuild the extension"
    )


@dataclass(frozen=True, slots=True)
class RequestRecord:
    kind: int
    viewer: int
    seats: int
    request_id: int
    legal: tuple[int, ...]
    encoding: tuple[bytes, bytes, bytes, bytes]


class PackedNetEvaluator:
    """Answer M5's blocking packed ABI with one network row per request."""

    def __init__(
        self,
        net: nw.WelcomeToNet,
        device: Optional[torch.device] = None,
        config: Optional[mcts.SearchConfig] = None,
        *,
        policy_net: Optional[nw.WelcomeToNet] = None,
        record_requests: bool = False,
        cuda_events: bool = False,
        cuda_event_sample_every: int = 32,
    ) -> None:
        self.net = net
        # Promotion gates can hold the simulated-opponent model fixed while
        # comparing two learner value/policy models. Ordinary generation keeps
        # the historical single-network path.
        self.policy_net = policy_net if policy_net is not None else net
        self.device = device or next(net.parameters()).device
        self.config = config or mcts.SearchConfig()
        self.record_requests = record_requests
        self.requests: list[RequestRecord] = []
        self.calls = 0
        self.rows = 0
        self.batch_widths: dict[int, int] = {}
        self.kind_rows = {"leaf": 0, "policy": 0}
        self._external_request_id = 0
        # Reused page-locked staging keeps cloud H2D transfer asynchronous and
        # avoids allocating four host tensors at every inference wave.
        self._pinned_inputs: list[Optional[torch.Tensor]] = [None] * 4
        self._device_inputs: list[Optional[torch.Tensor]] = [None] * 4
        self._stage_ns = {
            "parse": 0,
            "tensor_transfer": 0,
            "network_submit": 0,
            "postprocess_sync": 0,
        }
        if cuda_event_sample_every <= 0:
            raise ValueError("cuda_event_sample_every must be positive")
        self._cuda_events = bool(cuda_events and self.device.type == "cuda")
        self._cuda_event_sample_every = int(cuda_event_sample_every)
        self._cuda_event_pairs: list[tuple[int, object, object]] = []
        self._cuda_forward_ms: list[tuple[int, float]] = []

    @staticmethod
    def _arrays(buffers: Sequence[bytes]) -> tuple[np.ndarray, ...]:
        if len(buffers) != 4:
            raise ValueError(f"expected four encoder buffers, got {len(buffers)}")
        shapes = (
            enc.SHEET_PLANES_SHAPE,
            (enc.MAX_SEATS, enc.NUM_SHEET_SCALAR),
            enc.VIEWER_PLANE_SHAPE,
            (enc.NUM_GLOBAL_SCALAR,),
        )
        arrays = tuple(
            np.frombuffer(raw, dtype="<f4").reshape(shape)
            for raw, shape in zip(buffers, shapes)
        )
        return arrays

    @staticmethod
    def _batch_arrays(batch: dict, rows: int) -> tuple[np.ndarray, ...]:
        shapes = (
            enc.SHEET_PLANES_SHAPE,
            (enc.MAX_SEATS, enc.NUM_SHEET_SCALAR),
            enc.VIEWER_PLANE_SHAPE,
            (enc.NUM_GLOBAL_SCALAR,),
        )
        names = (
            "sheet_planes",
            "sheet_scalars",
            "viewer_plane",
            "global_scalars",
        )
        arrays = []
        for name, shape in zip(names, shapes):
            raw = bytes(batch[name])
            expected = rows * int(np.prod(shape)) * np.dtype("<f4").itemsize
            if len(raw) != expected:
                raise ValueError(
                    f"{name} has {len(raw)} bytes, expected {expected} for {rows} rows"
                )
            arrays.append(np.frombuffer(raw, dtype="<f4").reshape((rows, *shape)))
        return tuple(arrays)

    def _network_batch(
        self, arrays: Sequence[np.ndarray], leaf_rows: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if len(self._cuda_event_pairs) >= 2048:
            self._drain_cuda_events()
        started = time.perf_counter_ns()
        if self.device.type == "cuda":
            tensors = []
            for index, array in enumerate(arrays):
                host = self._pinned_inputs[index]
                if host is None or tuple(host.shape) != array.shape:
                    host = torch.empty(
                        array.shape, dtype=torch.float32, pin_memory=True
                    )
                    self._pinned_inputs[index] = host
                    self._device_inputs[index] = torch.empty(
                        array.shape, dtype=torch.float32, device=self.device
                    )
                np.copyto(host.numpy(), array)
                device_input = self._device_inputs[index]
                assert device_input is not None
                device_input.copy_(host, non_blocking=True)
                tensors.append(device_input)
        else:
            tensors = [
                torch.from_numpy(np.array(array, copy=True)).to(self.device)
                for array in arrays
            ]
        self._stage_ns["tensor_transfer"] += time.perf_counter_ns() - started
        self.net.eval()
        self.policy_net.eval()
        started = time.perf_counter_ns()
        sample_event = self._cuda_events and (
            self.calls % self._cuda_event_sample_every == 0
        )
        cuda_start = None
        if sample_event:
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        with torch.inference_mode():
            if self.policy_net is self.net:
                out = self.net.forward_inference(*tensors, leaf_rows=leaf_rows)
            else:
                rows = int(tensors[0].shape[0])
                is_policy = torch.ones(rows, dtype=torch.bool, device=self.device)
                is_policy[leaf_rows] = False
                policy_rows = torch.nonzero(is_policy, as_tuple=False).flatten()
                policy_logits = tensors[0].new_empty((rows, nw.NUM_MACRO_ACTIONS))

                if leaf_rows.numel():
                    leaf_inputs = [value.index_select(0, leaf_rows) for value in tensors]
                    leaf_out = self.net.forward_inference(*leaf_inputs)
                    policy_logits.index_copy_(0, leaf_rows, leaf_out["policy_logits"])
                    rank_logits = leaf_out["rank_logits"]
                    scores = leaf_out["score"]
                else:
                    rank_logits = tensors[0].new_empty((0, training.MAX_RANKS))
                    scores = tensors[0].new_empty((0, enc.MAX_SEATS))

                if policy_rows.numel():
                    policy_inputs = [
                        value.index_select(0, policy_rows) for value in tensors
                    ]
                    no_values = torch.empty(
                        0, dtype=torch.long, device=self.device
                    )
                    policy_out = self.policy_net.forward_inference(
                        *policy_inputs, leaf_rows=no_values
                    )
                    policy_logits.index_copy_(
                        0, policy_rows, policy_out["policy_logits"]
                    )
                out = {
                    "policy_logits": policy_logits,
                    "rank_logits": rank_logits,
                    "score": scores,
                }
        if cuda_start is not None:
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_end.record()
            self._cuda_event_pairs.append((int(arrays[0].shape[0]), cuda_start, cuda_end))
        self._stage_ns["network_submit"] += time.perf_counter_ns() - started
        return out

    def _drain_cuda_events(self) -> None:
        """Resolve sampled device timings only after their response has synced."""
        for width, start, end in self._cuda_event_pairs:
            end.synchronize()
            self._cuda_forward_ms.append((width, float(start.elapsed_time(end))))
        self._cuda_event_pairs.clear()

    def _cuda_event_profile(self) -> dict[str, object]:
        self._drain_cuda_events()
        values = np.asarray(
            [milliseconds for _, milliseconds in self._cuda_forward_ms],
            dtype=np.float64,
        )
        by_width: dict[str, dict[str, float]] = {}
        for width in sorted({width for width, _ in self._cuda_forward_ms}):
            samples = np.asarray(
                [ms for sample_width, ms in self._cuda_forward_ms if sample_width == width],
                dtype=np.float64,
            )
            by_width[str(width)] = {
                "samples": float(len(samples)),
                "mean_ms": float(samples.mean()),
                "p90_ms": float(np.percentile(samples, 90)),
            }
        return {
            "cuda_events_enabled": self._cuda_events,
            "cuda_event_sample_every": self._cuda_event_sample_every,
            "cuda_forward_samples": int(len(values)),
            "cuda_forward_mean_ms": float(values.mean()) if len(values) else None,
            "cuda_forward_p50_ms": float(np.percentile(values, 50)) if len(values) else None,
            "cuda_forward_p90_ms": float(np.percentile(values, 90)) if len(values) else None,
            "cuda_forward_by_batch": by_width,
        }

    def profile(self) -> dict[str, object]:
        """Cumulative Python/GPU-boundary stages without adding CUDA syncs.

        CUDA launches are asynchronous, so ``network_submit_ms`` is launch time;
        the unavoidable device synchronization is charged to
        ``postprocess_sync_ms`` when logits/value heads return to the CPU.
        """
        return {
            "calls": self.calls,
            "rows": self.rows,
            "leaf_rows": self.kind_rows["leaf"],
            "policy_rows": self.kind_rows["policy"],
            "parse_ms": self._stage_ns["parse"] / 1e6,
            "tensor_transfer_ms": self._stage_ns["tensor_transfer"] / 1e6,
            "network_submit_ms": self._stage_ns["network_submit"] / 1e6,
            "postprocess_sync_ms": self._stage_ns["postprocess_sync"] / 1e6,
            "batch_widths": dict(sorted(self.batch_widths.items())),
            **self._cuda_event_profile(),
        }

    def reset_profile(self) -> None:
        self._drain_cuda_events()
        self.calls = 0
        self.rows = 0
        self.kind_rows = {"leaf": 0, "policy": 0}
        self.batch_widths.clear()
        self._cuda_forward_ms.clear()
        for name in self._stage_ns:
            self._stage_ns[name] = 0

    @torch.inference_mode()
    def evaluate_request(
        self,
        kind: int,
        buffers: Sequence[bytes],
        legal: Sequence[int],
        viewer: int,
        seats: int,
        request_id: int,
    ) -> tuple[bytes, Optional[float]]:
        if kind not in (LEAF, POLICY):
            raise ValueError(f"unknown evaluator request kind {kind}")
        if not 0 <= viewer < seats <= enc.MAX_SEATS:
            raise ValueError(f"invalid viewer/seats pair {viewer}/{seats}")
        packed = tuple(bytes(raw) for raw in buffers)
        if self.record_requests:
            self.requests.append(
                RequestRecord(
                    kind, viewer, seats, request_id, tuple(legal), packed  # type: ignore[arg-type]
                )
            )
        started = time.perf_counter_ns()
        arrays = self._arrays(packed)
        tensors = [array[np.newaxis, ...] for array in arrays]
        self._stage_ns["parse"] += time.perf_counter_ns() - started
        self.calls += 1
        self.rows += 1
        self.kind_rows["leaf" if kind == LEAF else "policy"] += 1
        self.batch_widths[1] = self.batch_widths.get(1, 0) + 1
        leaf_rows = torch.tensor(
            [0] if kind == LEAF else [], dtype=torch.long, device=self.device
        )
        out = self._network_batch(tensors, leaf_rows)
        started = time.perf_counter_ns()
        logits = out["policy_logits"][0].cpu().numpy()
        mask = np.zeros(nw.NUM_MACRO_ACTIONS, dtype=np.bool_)
        mask[np.asarray(legal, dtype=np.intp)] = True
        priors = mcts._masked_softmax(logits, mask)

        value: Optional[float] = None
        if kind == LEAF:
            rank_mask = torch.zeros((1, training.MAX_RANKS), device=self.device)
            rank_mask[0, :seats] = 1.0
            ranks = nw.rank_probabilities(out["rank_logits"], rank_mask)[0].cpu().numpy()
            scores = out["score"][0].cpu().numpy()
            value = mcts.blend_value(ranks, scores, seats, self.config)[0]
        self._stage_ns["postprocess_sync"] += time.perf_counter_ns() - started
        return priors.astype("<f4", copy=False).tobytes(), value

    @torch.inference_mode()
    def forward(self, batch: dict) -> dict:
        """Answer one mixed M0-E ``RequestBatchV2``.

        LEAF and POLICY rows deliberately share this call. Values are emitted
        as little-endian f64 so the Rust tree never quantizes its accumulator
        input; POLICY rows carry zero in that ignored column.
        """
        parse_started = time.perf_counter_ns()
        version = int(batch.get("version", -1))
        rows = int(batch.get("rows", -1))
        if version != EVALUATOR_ABI_VERSION:
            raise ValueError(
                f"evaluator request ABI {version}, expected {EVALUATOR_ABI_VERSION}"
            )
        if rows <= 0:
            raise ValueError(f"evaluator batch rows must be positive, got {rows}")
        arrays = self._batch_arrays(batch, rows)

        kind = np.frombuffer(bytes(batch["kind"]), dtype=np.uint8)
        seats = np.frombuffer(bytes(batch["seats"]), dtype=np.uint8)
        request_ids = np.frombuffer(bytes(batch["request_id"]), dtype="<u4")
        offsets = np.frombuffer(bytes(batch["legal_offsets"]), dtype="<u4")
        legal = np.frombuffer(bytes(batch["legal_indices"]), dtype="<u2")
        if len(kind) != rows or len(seats) != rows or len(request_ids) != rows:
            raise ValueError("kind, seats, and request_id must have one entry per row")
        if len(offsets) != rows + 1 or int(offsets[0]) != 0:
            raise ValueError("legal_offsets must be CSR offsets with rows + 1 entries")
        if np.any(offsets[1:] < offsets[:-1]) or int(offsets[-1]) != len(legal):
            raise ValueError("legal_offsets are not monotone or do not cover legal_indices")
        if np.any(kind > POLICY):
            raise ValueError("evaluator batch contains an unknown request kind")
        if np.any((seats < 2) | (seats > enc.MAX_SEATS)):
            raise ValueError("evaluator batch contains a seat count outside 2-4")
        if len(np.unique(request_ids)) != rows:
            raise ValueError("outstanding evaluator request_id values must be unique")
        if len(legal) and int(legal.max()) >= nw.NUM_MACRO_ACTIONS:
            raise ValueError("legal_indices contains an out-of-range macro")
        # Whole-batch checks, not a per-row Python loop.  This runs on the one
        # thread holding the GIL while every Rust search worker is parked at the
        # inference barrier, so a `np.unique` per row put ``rows`` sorts directly
        # in the critical path of the whole pool.  Tagging each macro with its
        # row makes one sort answer the same question for all of them.
        lengths = np.diff(offsets).astype(np.int64, copy=False)
        empty = np.flatnonzero(lengths == 0)
        if len(empty):
            raise ValueError(f"evaluator row {int(empty[0])} has no legal macros")
        row_of = np.repeat(np.arange(rows, dtype=np.int64), lengths)
        keyed = row_of * nw.NUM_MACRO_ACTIONS + legal.astype(np.int64, copy=False)
        if len(np.unique(keyed)) != len(keyed):
            # Only the failing batch pays for locating the row.
            ordered = np.sort(keyed)
            first = ordered[np.flatnonzero(np.diff(ordered) == 0)[0]]
            row = int(first) // nw.NUM_MACRO_ACTIONS
            raise ValueError(f"evaluator row {row} repeats a legal macro")
        self._stage_ns["parse"] += time.perf_counter_ns() - parse_started

        self.calls += 1
        self.rows += rows
        self.kind_rows["leaf"] += int(np.count_nonzero(kind == LEAF))
        self.kind_rows["policy"] += int(np.count_nonzero(kind == POLICY))
        self.batch_widths[rows] = self.batch_widths.get(rows, 0) + 1
        leaf_rows = np.flatnonzero(kind == LEAF)
        leaf_rows_device = torch.as_tensor(
            leaf_rows, dtype=torch.long, device=self.device
        )
        out = self._network_batch(arrays, leaf_rows_device)
        post_started = time.perf_counter_ns()
        # One device gather replaces the old Python row loop and transfers only
        # legal policy entries. Rust owns segmented softmax and value blending.
        legal_rows = torch.as_tensor(row_of, device=self.device)
        legal_actions = torch.as_tensor(
            legal.astype(np.int64, copy=False), device=self.device
        )
        legal_logits = out["policy_logits"][legal_rows, legal_actions].float().cpu()
        rank_logits = out["rank_logits"].float().cpu()
        scores_out = out["score"].float().cpu()

        self._stage_ns["postprocess_sync"] += time.perf_counter_ns() - post_started

        return {
            "version": EVALUATOR_ABI_VERSION,
            "rows": rows,
            "request_id": request_ids.astype("<u4", copy=False).tobytes(),
            "legal_offsets": offsets.astype("<u4", copy=False).tobytes(),
            "legal_logits": legal_logits.numpy().astype("<f4", copy=False).tobytes(),
            "leaf_request_id": request_ids[leaf_rows]
            .astype("<u4", copy=False)
            .tobytes(),
            "rank_logits": rank_logits.numpy().astype("<f4", copy=False).tobytes(),
            "scores": scores_out.numpy().astype("<f4", copy=False).tobytes(),
        }

    def policy_states(
        self, states: Sequence[object], seats: Sequence[int]
    ) -> tuple[list[np.ndarray], list[tuple[int, ...]]]:
        """Batch real-game opponent policies from Rust-owned states.

        Search POLICY rows already pass through :meth:`forward`; S2 also needs
        cheap policies for opponents' *actual* moves. Keeping this path batched
        avoids replacing the search callback bottleneck with one Python call per
        opponent decision. Sampling stays outside this method so every seat can
        own an independent portable RNG stream.
        """
        rows = len(states)
        if rows == 0:
            return [], []
        if len(seats) != rows:
            raise ValueError("states and seat counts are not row-aligned")
        encodings = []
        legals: list[tuple[int, ...]] = []
        for state, count in zip(states, seats):
            actor = int(state.actor)
            if not 0 <= actor < count <= enc.MAX_SEATS:
                raise ValueError(f"invalid actor/seats pair {actor}/{count}")
            encodings.append(tuple(bytes(raw) for raw in state.encode_state(actor)))
            legals.append(tuple(int(action) for action in state.legal_macros()))

        ids = np.arange(
            self._external_request_id,
            self._external_request_id + rows,
            dtype=np.uint64,
        ).astype("<u4")
        self._external_request_id = (self._external_request_id + rows) & 0xFFFFFFFF
        offsets = [0]
        flat_legal: list[int] = []
        for legal in legals:
            if not legal:
                raise ValueError("live opponent state has no legal macros")
            flat_legal.extend(legal)
            offsets.append(len(flat_legal))
        payload = {
            "version": EVALUATOR_ABI_VERSION,
            "rows": rows,
            "sheet_planes": b"".join(row[0] for row in encodings),
            "sheet_scalars": b"".join(row[1] for row in encodings),
            "viewer_plane": b"".join(row[2] for row in encodings),
            "global_scalars": b"".join(row[3] for row in encodings),
            "legal_indices": np.asarray(flat_legal, dtype="<u2").tobytes(),
            "legal_offsets": np.asarray(offsets, dtype="<u4").tobytes(),
            "kind": bytes([POLICY] * rows),
            "seats": bytes(int(count) for count in seats),
            "request_id": ids.tobytes(),
        }
        response = self.forward(payload)
        response_ids = np.frombuffer(response["request_id"], dtype="<u4")
        response_offsets = np.frombuffer(response["legal_offsets"], dtype="<u4")
        response_logits = np.frombuffer(response["legal_logits"], dtype="<f4")
        by_id = {int(request_id): row for row, request_id in enumerate(response_ids)}
        if len(by_id) != rows or set(by_id) != set(int(request_id) for request_id in ids):
            raise ValueError("opponent policy response request ids are not aligned")
        policies = []
        for request_id, row_legal in zip(ids, legals):
            row = by_id[int(request_id)]
            start, end = int(response_offsets[row]), int(response_offsets[row + 1])
            compact = response_logits[start:end]
            if len(compact) != len(row_legal):
                raise ValueError("opponent policy response legal segment is misaligned")
            # Actual-game opponent sampling does not cross the Rust scheduler;
            # retain its dense public return shape while sharing ABI v3.
            dense = np.zeros(nw.NUM_MACRO_ACTIONS, dtype=np.float32)
            shifted = compact - compact.max()
            weights = np.exp(shifted).astype(np.float32, copy=False)
            mass = weights.sum(dtype=np.float32)
            if not mass > 0:
                weights.fill(np.float32(1.0 / len(weights)))
            else:
                weights /= mass
            dense[np.asarray(row_legal, dtype=np.intp)] = weights
            policies.append(dense)
        return policies, legals


def _native_kwargs(config: mcts.SearchConfig) -> dict:
    return dict(
        simulations=config.simulations,
        c_puct=config.c_puct,
        alpha=config.alpha,
        margin_gain=config.margin_gain,
        confidence_power=config.confidence_power,
        prune_roundabout_pass=config.prune_roundabout_pass,
        chance_widening=config.chance_widening,
        chance_widening_alpha=config.chance_widening_alpha,
        max_particles=config.max_particles,
        noise_fresh_fraction=config.noise_fresh_fraction,
        dirichlet_weight=config.dirichlet_weight,
        temperature=config.temperature,
        noise_required=(
            config.dirichlet_alpha is not None
            or config.dirichlet_concentration is not None
        ),
    )


def native_search(config: mcts.SearchConfig):
    """Construct the blocking M5 diagnostic engine."""
    if wr is None:
        raise RuntimeError(
            "welcome_to_rust is not installed; run maturin develop --release in "
            "games/welcome_to/welcome_to_rust"
        )
    return wr.RustMcts(**_native_kwargs(config))


def native_scheduler(config: mcts.SearchConfig, capacity: int = 32):
    """Construct M6's persistent cross-game coalescing scheduler."""
    if wr is None:
        raise RuntimeError(
            "welcome_to_rust is not installed; run maturin develop --release in "
            "games/welcome_to/welcome_to_rust"
        )
    return wr.RustScheduler(capacity=capacity, **_native_kwargs(config))


def native_cloud_scheduler(
    config: mcts.SearchConfig, capacity: int = 256, workers: int = 8
):
    """Construct the resumable fixed-worker/global-broker scheduler."""
    if wr is None:
        raise RuntimeError(
            "welcome_to_rust is not installed; run maturin develop --release in "
            "games/welcome_to/welcome_to_rust"
        )
    return wr.RustCloudScheduler(
        capacity=capacity, workers=workers, **_native_kwargs(config)
    )


def root_noise(
    config: mcts.SearchConfig, width: int, rng: PortableRng
) -> tuple[Optional[np.ndarray], int]:
    """Generate Python's canonical Dirichlet vector and return the advanced tape.

    The Python oracle consumes one search-RNG draw to seed NumPy before its
    first determinization. Rust applies the resulting vector, so it must start
    from the returned state—not the original seed—or every later chance draw
    shifts by one.
    """
    if width < 2:
        return None, rng.state
    alpha = (
        config.dirichlet_concentration / width
        if config.dirichlet_concentration is not None
        else config.dirichlet_alpha
    )
    if alpha is None:
        return None, rng.state
    vector = np.random.default_rng(rng.randrange(1 << 32)).dirichlet([alpha] * width)
    return vector, rng.state
