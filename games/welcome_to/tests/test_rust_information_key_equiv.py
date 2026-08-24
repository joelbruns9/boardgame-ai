"""Fast M4 gate: constructed observation classes, never random non-collisions."""

from __future__ import annotations

import pytest

from games.welcome_to import mcts
from games.welcome_to.rust_key_equiv import run_gate

wr = pytest.importorskip(
    "welcome_to_rust",
    reason="the Rust key is not built; run maturin develop --release",
)


def test_the_information_key_abi_versions_agree():
    assert wr.INFORMATION_KEY_ABI_VERSION == mcts.INFORMATION_KEY_ABI_VERSION


def test_constructed_python_and_rust_partitions_agree():
    report = run_gate()
    assert report.collisions >= 20
    assert report.separations >= 40


def test_invalid_viewer_is_refused():
    state = wr.RustGameState(3, players=2, advanced=False, solo_rules=False)
    with pytest.raises(RuntimeError, match="outside 2 seats"):
        state.information_key(2)
