"""NumPy view of the packed M3 Rust encoder ABI.

The Rust extension deliberately returns bytes rather than depending on the
NumPy C ABI.  These arrays are read-only views of those immutable buffers;
callers that need writable storage can copy them explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .encoder import (
    MAX_SEATS,
    NUM_GLOBAL_SCALAR,
    NUM_SHEET_SCALAR,
    SHEET_PLANES_SHAPE,
    VIEWER_PLANE_SHAPE,
)


EncodedState = tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]


def encode_state(state: Any, player: int | None = None) -> EncodedState:
    """Encode a ``welcome_to_rust.RustGameState`` as four float32 arrays."""

    raw = state.encode_state(player)
    shapes = (
        SHEET_PLANES_SHAPE,
        (MAX_SEATS, NUM_SHEET_SCALAR),
        VIEWER_PLANE_SHAPE,
        (NUM_GLOBAL_SCALAR,),
    )
    return tuple(
        np.frombuffer(buffer, dtype="<f4").reshape(shape)
        for buffer, shape in zip(raw, shapes, strict=True)
    )  # type: ignore[return-value]
