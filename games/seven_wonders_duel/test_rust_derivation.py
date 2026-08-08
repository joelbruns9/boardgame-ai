"""Bit-exact and corruption gates for the Rust-default derive path."""

from __future__ import annotations

from dataclasses import replace
import random

import numpy as np
import pytest

from .buffer import GameRecorder, ReplayMismatchError
from .codec import legal_action_indices
from .dataset import derive_records_rust, examples_from_record
from .game import Phase


ARRAY_FIELDS = (
    "type_ids",
    "entity_ids",
    "aux_ids",
    "features",
    "legal",
    "policy_target",
)
SCALAR_FIELDS = (
    "has_policy",
    "value_class",
    "joint7_class",
    "margin",
    "margin_valid",
    "military_final",
    "sci_final_my",
    "sci_final_opp",
    "game_key",
    "iteration",
    "root_value",
)


def _record(seed: int):
    recorder = GameRecorder(seed, first_player=seed % 2, iteration=7)
    rng = random.Random(seed ^ 0xD3A1)
    while recorder.game.phase is not Phase.COMPLETE:
        legal = legal_action_indices(recorder.game)
        action = rng.choice(legal)
        visits = {candidate: offset + 1 for offset, candidate in enumerate(legal)}
        recorder.play(
            action,
            visits=visits,
            root_value=(rng.random() * 2.0 - 1.0),
            sims=sum(visits.values()),
            policy_excluded=(len(recorder._moves) % 3 != 0),
        )
    return recorder.finish()


def _assert_examples_equal(expected, actual):
    assert len(actual) == len(expected)
    for left, right in zip(expected, actual):
        for field in ARRAY_FIELDS:
            assert np.array_equal(getattr(left, field), getattr(right, field)), field
            assert getattr(right, field).dtype == getattr(left, field).dtype, field
        for field in SCALAR_FIELDS:
            assert getattr(right, field) == getattr(left, field), field


@pytest.mark.parametrize("record_fast_moves", [False, True])
def test_rust_derivation_is_bit_exact_with_python(record_fast_moves):
    records = [_record(seed) for seed in (13, 29, 47)]
    rust_rows = derive_records_rust(
        records, record_fast_moves=record_fast_moves, batch_games=2
    )
    assert len(rust_rows) == len(records)
    for record, (rust_examples, rust_stats) in zip(records, rust_rows):
        python_stats = []
        python_examples = examples_from_record(
            record,
            record_fast_moves=record_fast_moves,
            on_derived=python_stats.append,
        )
        _assert_examples_equal(python_examples, rust_examples)
        assert python_stats == [rust_stats]


@pytest.mark.parametrize(
    "corruption",
    ["mask", "actor", "action", "chance", "result", "final_digest", "trajectory_digest"],
)
def test_rust_derivation_rejects_corrupt_records(corruption):
    record = _record(91)
    if corruption == "mask":
        moves = list(record.moves)
        moves[0] = replace(moves[0], mask_hash="sha256:0000000000000000")
        record = replace(record, moves=tuple(moves))
    elif corruption == "actor":
        moves = list(record.moves)
        moves[0] = replace(moves[0], actor=1 - moves[0].actor)
        record = replace(record, moves=tuple(moves))
    elif corruption == "action":
        moves = list(record.moves)
        moves[5] = replace(moves[5], action=999_999)
        record = replace(record, moves=tuple(moves))
    elif corruption == "chance":
        record = replace(record, chance_log=record.chance_log[:-1])
    elif corruption == "result":
        record = replace(record, winner=0 if record.winner is None else 1 - record.winner)
    elif corruption == "final_digest":
        record = replace(record, final_digest="sha256:" + "0" * 64)
    else:
        record = replace(record, trajectory_digest="sha256:" + "0" * 64)

    with pytest.raises(ReplayMismatchError):
        derive_records_rust([record])


def test_rust_derivation_rejects_unknown_digest_version():
    record = replace(_record(117), digest_version="future-unknown-v9")

    with pytest.raises(ValueError, match="unsupported buffer digest version"):
        derive_records_rust([record])
