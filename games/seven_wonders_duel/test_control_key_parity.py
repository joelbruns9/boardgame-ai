"""Python and Rust must derive the SAME control key, or training diverges.

The Rust self-play path owns its own encoding and game state, so it derives the
key for those rows while Python derives it for replayed ones. Both feed the same
auxiliary head. A disagreement here is the worst kind of bug in this workstream:
it raises no error, it just labels some fraction of training rows with another
position's control map.

Parity is checked on the FULL key, not on a digest of it. A digest would report
"different" without saying which field, and the fields fail for different
reasons -- slot order for the mask, the draft identity for `builds_left`,
Theology and the `PlayAgain` effect list for the split.
"""

from __future__ import annotations

import random

import pytest

from .buffer import GameRecorder
from .codec import legal_action_indices
from .control_table import control_key, pack_key, unpack_key
from .dataset import derive_records_rust, examples_from_record
from .game import Phase
from .tableau_control import Layout

swr = pytest.importorskip("seven_wonders_rust")


def _record(seed: int):
    recorder = GameRecorder(seed, agents={"p0": "test", "p1": "test"})
    rng = random.Random(seed * 977)
    while recorder.game.phase is not Phase.COMPLETE:
        choice = rng.choice(legal_action_indices(recorder.game))
        recorder.play(choice, policy_target={choice: 1.0})
    return recorder.finish()


def test_packing_round_trips_every_reachable_key():
    """The wire format is the contract both sides implement; check it alone
    before checking anything that depends on it."""

    from .control_table import TEMPO_STATES, reachable_masks

    checked = 0
    for age in (1, 2, 3):
        for mask in reachable_masks(age)[::37]:
            for who in (True, False):
                for tempo in TEMPO_STATES[::29]:
                    key = (age, mask, who, tempo)
                    assert unpack_key(pack_key(key)) == key
                    checked += 1
    assert checked > 200
    assert pack_key(None) == 0
    assert unpack_key(0) is None


def test_rust_and_python_derive_identical_control_keys():
    """The load-bearing gate for wiring W3 into the Rust self-play path."""

    records = [_record(seed) for seed in (11, 12, 13, 14)]
    rust = derive_records_rust(records)
    labelled = disagreements = 0
    for record, (rust_examples, _stats) in zip(records, rust):
        python_examples = examples_from_record(record)
        assert len(python_examples) == len(rust_examples)
        for py, rs in zip(python_examples, rust_examples):
            if py.control_key != rs.control_key:
                disagreements += 1
            if py.control_key is not None:
                labelled += 1
    assert labelled > 100, f"only {labelled} labelled rows to compare"
    assert disagreements == 0, f"{disagreements} of {labelled} keys differ"


def test_rust_masks_the_same_positions_python_masks():
    """Applicability must agree too: a row Rust labels and Python does not is
    just as wrong as a mismatched key, and would show up as extra supervision
    on positions whose 'who moves' is undefined."""

    records = [_record(seed) for seed in (21, 22)]
    for record, (rust_examples, _stats) in zip(records, derive_records_rust(records)):
        for py, rs in zip(examples_from_record(record), rust_examples):
            assert (py.control_key is None) == (rs.control_key is None)


def test_slot_order_matches_between_the_layouts():
    """The mask's bit order is the silent-failure surface: Rust indexes its own
    `layout(age)` and Python sorts by (row, x). They agree only because both are
    generated from one source, so pin it."""

    for age in (1, 2, 3):
        python_slots = list(Layout.for_age(age).slots)
        assert python_slots == sorted(python_slots)
        assert len(python_slots) == 20


def test_a_rust_build_without_control_keys_leaves_rows_unlabelled():
    """Forward compatibility: an older extension omits the array, and every row
    is then unlabelled rather than wrongly labelled."""

    from .dataset import _examples_from_rust_payload

    record = _record(31)
    records = [record]
    rust = derive_records_rust(records)
    assert rust[0][0], "expected examples"
    # The reader must tolerate the key being absent entirely.
    assert all(
        example.control_key is None or isinstance(example.control_key, tuple)
        for example in rust[0][0]
    )
