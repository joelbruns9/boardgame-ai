"""Packed Python boundary for the Rust-owned Welcome To search and M6 scheduler.

The native search owns states, observation keys, the tree and all accumulated
statistics.  This module is deliberately small: it turns M3's four packed
little-endian buffers into one Torch row and returns the frozen M0-E response.
M6 concatenates those rows into the frozen V2 batch-major representation; the
single-request M5 seam remains as a diagnostic control.
"""

from __future__ import annotations

from dataclasses import dataclass
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
EVALUATOR_ABI_VERSION = 2

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
        record_requests: bool = False,
    ) -> None:
        self.net = net
        self.device = device or next(net.parameters()).device
        self.config = config or mcts.SearchConfig()
        self.record_requests = record_requests
        self.requests: list[RequestRecord] = []
        self.calls = 0
        self.rows = 0
        self.batch_widths: dict[int, int] = {}
        self._external_request_id = 0

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

    def _network_batch(self, arrays: Sequence[np.ndarray]) -> dict[str, torch.Tensor]:
        tensors = [
            torch.from_numpy(np.array(array, copy=True)).to(self.device)
            for array in arrays
        ]
        self.net.eval()
        return self.net(*tensors)

    @torch.no_grad()
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
        arrays = self._arrays(packed)
        tensors = [array[np.newaxis, ...] for array in arrays]
        self.calls += 1
        self.rows += 1
        self.batch_widths[1] = self.batch_widths.get(1, 0) + 1
        out = self._network_batch(tensors)
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
        return priors.astype("<f4", copy=False).tobytes(), value

    @torch.no_grad()
    def forward(self, batch: dict) -> dict:
        """Answer one mixed M0-E ``RequestBatchV2``.

        LEAF and POLICY rows deliberately share this call. Values are emitted
        as little-endian f64 so the Rust tree never quantizes its accumulator
        input; POLICY rows carry zero in that ignored column.
        """
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
        for row in range(rows):
            start, end = int(offsets[row]), int(offsets[row + 1])
            if start == end:
                raise ValueError(f"evaluator row {row} has no legal macros")
            row_legal = legal[start:end]
            if len(np.unique(row_legal)) != len(row_legal):
                raise ValueError(f"evaluator row {row} repeats a legal macro")

        self.calls += 1
        self.rows += rows
        self.batch_widths[rows] = self.batch_widths.get(rows, 0) + 1
        out = self._network_batch(arrays)
        logits = out["policy_logits"].cpu().numpy()
        priors = np.zeros((rows, nw.NUM_MACRO_ACTIONS), dtype=np.float32)
        for row in range(rows):
            start, end = int(offsets[row]), int(offsets[row + 1])
            mask = np.zeros(nw.NUM_MACRO_ACTIONS, dtype=np.bool_)
            mask[legal[start:end].astype(np.intp, copy=False)] = True
            priors[row] = mcts._masked_softmax(logits[row], mask)

        values = np.zeros(rows, dtype="<f8")
        leaf_rows = np.flatnonzero(kind == LEAF)
        if len(leaf_rows):
            rank_mask = torch.zeros(
                (len(leaf_rows), training.MAX_RANKS), device=self.device
            )
            for local_row, source_row in enumerate(leaf_rows):
                rank_mask[local_row, : int(seats[source_row])] = 1.0
            ranks = nw.rank_probabilities(
                out["rank_logits"][leaf_rows.tolist()], rank_mask
            ).cpu().numpy()
            scores = out["score"][leaf_rows.tolist()].cpu().numpy()
            for local_row, source_row in enumerate(leaf_rows):
                values[source_row] = mcts.blend_value(
                    ranks[local_row],
                    scores[local_row],
                    int(seats[source_row]),
                    self.config,
                )[0]

        return {
            "version": EVALUATOR_ABI_VERSION,
            "rows": rows,
            "request_id": request_ids.astype("<u4", copy=False).tobytes(),
            "priors": priors.astype("<f4", copy=False).tobytes(),
            "values": values.tobytes(),
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
        raw_priors = np.frombuffer(response["priors"], dtype="<f4").reshape(
            rows, nw.NUM_MACRO_ACTIONS
        )
        by_id = {int(request_id): row for row, request_id in enumerate(response_ids)}
        if len(by_id) != rows or set(by_id) != set(int(request_id) for request_id in ids):
            raise ValueError("opponent policy response request ids are not aligned")
        policies = [raw_priors[by_id[int(request_id)]].copy() for request_id in ids]
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
