"""Fast structural tests for the resumable W0 sizing harness."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from .dataset import Example, MAX_FEATURES, collate
from .net import SWDNet
from .train import train_steps

from .w0_sizing import (
    _PairedWilson,
    _job_stem,
    _pack_examples,
    _tensor_batch,
    _wilson_interval,
    build_parser,
)


def test_wilson_interval_contains_even_score() -> None:
    lower, upper = _wilson_interval(15.0, 30)
    assert lower < 0.5 < upper
    assert lower == pytest.approx(1.0 - upper)


def test_paired_wilson_waits_for_minimum_and_accepts() -> None:
    test = _PairedWilson(minimum_games=60)
    for _ in range(59):
        assert test.update(1.0).decision == "continue"
    assert test.update(1.0).decision == "accept"
    assert len(test.pair_scores) == 30
    assert test.lower > 0.5


def test_paired_wilson_uses_mean_of_two_seats() -> None:
    test = _PairedWilson(minimum_games=60)
    for index in range(60):
        test.update(float(index % 2))
    assert test.pair_scores == [0.5] * 30
    assert test.decision == "continue"


def test_job_names_are_stable_and_unambiguous() -> None:
    assert _job_stem("sweep", "M", 0.0002, 0) == "sweep_M_lr2e-04_seed0"


def test_harness_defaults_to_bf16_and_full_budget() -> None:
    args = build_parser().parse_args(["all"])
    assert args.precision == "bf16"
    assert args.games == 12_000
    assert args.steps == 4_000
    assert args.max_games == 800


def _example(tokens: int, game_key: int) -> Example:
    legal = np.asarray([3, 17], dtype=np.int16)
    features = np.arange(tokens * MAX_FEATURES, dtype=np.float16).reshape(
        tokens, MAX_FEATURES
    )
    return Example(
        type_ids=np.zeros(tokens, dtype=np.int8),
        entity_ids=np.zeros(tokens, dtype=np.int16),
        aux_ids=np.zeros(tokens, dtype=np.int16),
        features=features,
        legal=legal,
        policy_target=np.asarray([0.25, 0.75], dtype=np.float32),
        has_policy=True,
        value_class=game_key % 3,
        joint7_class=game_key % 7,
        margin=0.25,
        margin_valid=True,
        military_final=-0.5,
        sci_final_my=0.5,
        sci_final_opp=1.0 / 3.0,
        game_key=game_key,
        iteration=70,
    )


def test_packed_batch_is_value_identical_to_collate() -> None:
    examples = [_example(3, 10), _example(5, 11)]
    packed = _pack_examples(examples, 0.05, "test")
    expected = collate([examples[1], examples[0]])
    actual = _tensor_batch(packed["storage"], [1, 0], "cpu")
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key].dtype == expected[key].dtype
        assert torch.equal(actual[key], expected[key]), key


def test_sizing_cosine_schedule_is_warmup_then_decay() -> None:
    examples = [_example(3, 10), _example(5, 11)]
    history, _ = train_steps(
        SWDNet(32, 1, 2),
        examples,
        None,
        device="cpu",
        steps=4,
        batch_size=2,
        lr=1e-3,
        warmup_steps=2,
        validate_every=1,
        cosine_decay=True,
        log=lambda *_: None,
    )
    assert [row["lr"] for row in history] == pytest.approx(
        [5e-4, 1e-3, 5e-4, 0.0]
    )
