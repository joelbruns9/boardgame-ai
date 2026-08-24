"""M5 Python boundary for the Rust-owned Welcome To search.

The native search owns states, observation keys, the tree and all accumulated
statistics.  This module is deliberately small: it turns M3's four packed
little-endian buffers into one Torch row and returns the frozen M0-E response.
M6 will batch these same requests; it must not reinterpret them.
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
        self.batch_widths: dict[int, int] = {1: 0}

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
        tensors = [
            torch.from_numpy(np.array(array, copy=True)).unsqueeze(0).to(self.device)
            for array in arrays
        ]
        self.net.eval()
        self.calls += 1
        self.rows += 1
        self.batch_widths[1] += 1
        out = self.net(*tensors)
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


def native_search(config: mcts.SearchConfig):
    """Construct the M5 engine, refusing every option it cannot honor."""
    if wr is None:
        raise RuntimeError(
            "welcome_to_rust is not installed; run maturin develop --release in "
            "games/welcome_to/welcome_to_rust"
        )
    return wr.RustMcts(
        simulations=config.simulations,
        c_puct=config.c_puct,
        alpha=config.alpha,
        margin_gain=config.margin_gain,
        confidence_power=config.confidence_power,
        prune_roundabout_pass=config.prune_roundabout_pass,
        chance_widening=config.chance_widening,
        noise_fresh_fraction=config.noise_fresh_fraction,
        dirichlet_weight=config.dirichlet_weight,
        temperature=config.temperature,
        noise_required=(
            config.dirichlet_alpha is not None
            or config.dirichlet_concentration is not None
        ),
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
