"""What the evaluator boundary does with a batch it cannot decode.

An overnight run died here with `ValueError: both buffer length (0) and count
(-1) must not be 0` -- `torch.frombuffer` on an empty `legal_actions`, naming
neither the field nor the game. The cause was upstream (terminal children of a
forced chance edge reaching the network; fixed in `tree_resumable.rs`), but the
boundary should still say which invariant broke.
"""

from __future__ import annotations

import pytest
import torch

from .f4_cost_model import FEATURE_WIDTH, build_payload
from .rust_bridge import rust_flat_batch_adapter


class _StubEvaluator:
    device = "cpu"
    max_batch = 64


def _row(legal: list[int]) -> dict:
    return {
        "actor": 0,
        "tokens": [(1, 0, 0, [0.0] * FEATURE_WIDTH)],
        "legal": legal,
    }


@pytest.fixture
def adapter():
    return rust_flat_batch_adapter(_StubEvaluator())


def test_an_all_terminal_batch_names_the_invariant_it_broke(adapter):
    payload = build_payload([_row([]), _row([])])
    with pytest.raises(ValueError, match="carries no legal actions at all"):
        adapter.build_device_batch(payload)


def test_a_batch_with_any_legal_row_still_decodes(adapter):
    # Guards the obvious over-correction: rejecting rows individually rather
    # than only the batch that carries nothing at all.
    payload = build_payload([_row([]), _row([3, 7])])
    _, legal_lengths, legal_actions, _, _ = adapter.build_device_batch(payload)
    assert legal_lengths.tolist() == [0, 2]
    assert legal_actions.tolist() == [3, 7]
