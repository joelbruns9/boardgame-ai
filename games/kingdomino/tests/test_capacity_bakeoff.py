from __future__ import annotations

from argparse import Namespace

import pytest

from games.kingdomino.capacity_bakeoff import (
    ArmSpec,
    block_split,
    paired_block_bootstrap,
    parse_arm,
    summarize_trials,
)


def _trial(arm, seed, lr, ce_values, brier_values):
    blocks = [
        {
            "block_id": index,
            "n": 400,
            "policy_ce": ce,
            "win_brier": brier,
        }
        for index, (ce, brier) in enumerate(zip(ce_values, brier_values))
    ]
    return {
        "arm": arm,
        "optimization_seed": seed,
        "lr": lr,
        "policy_ce": sum(ce_values) / len(ce_values),
        "win_brier": sum(brier_values) / len(brier_values),
        "checkpoint": f"{arm}-{seed}-{lr}.pt",
        "block_metrics": blocks,
    }


def test_arm_parser_distinguishes_pooling_identity():
    assert parse_arm("128x8+gp") == ArmSpec("128x8+gp", 128, 8, True)
    assert parse_arm("80x6") == ArmSpec("80x6", 80, 6, False)
    with pytest.raises(ValueError):
        parse_arm("128x8+plain")


def test_block_split_keeps_whole_contiguous_blocks():
    examples = list(range(1_025))
    train, holdout, blocks, ids = block_split(examples, 0.4, 400, 9)
    assert len(train) + len(holdout) == len(examples)
    assert [block_id for block_id, _ in blocks] == ids
    assert all(chunk == examples[i * 400 : (i + 1) * 400] for i, chunk in blocks)


def test_paired_bootstrap_and_gate_use_seed_means_and_block_pairs():
    control = [
        _trial("80x6", 1, 1e-3, [1.0, 1.2, 0.8], [0.20, 0.21, 0.19]),
        _trial("80x6", 2, 1e-3, [1.02, 1.22, 0.82], [0.20, 0.21, 0.19]),
    ]
    candidate = [
        _trial("128x8+gp", 1, 1e-3, [0.90, 1.08, 0.72], [0.19, 0.20, 0.18]),
        _trial("128x8+gp", 2, 1e-3, [0.91, 1.09, 0.73], [0.19, 0.20, 0.18]),
        # An incomplete LR is excluded even if its lone seed looks better.
        _trial("128x8+gp", 1, 5e-4, [0.50, 0.50, 0.50], [0.10, 0.10, 0.10]),
    ]
    interval = paired_block_bootstrap(
        control, candidate[:2], "policy_ce", samples=1_000, seed=5
    )
    assert interval["estimate_control_minus_candidate"] > 0
    assert interval["ci95_lower"] > 0

    args = Namespace(
        bootstrap_samples=1_000,
        bootstrap_seed=5,
        policy_relative_floor=0.015,
        optimization_seeds=[1, 2],
    )
    summary = summarize_trials(
        control + candidate,
        [ArmSpec("80x6", 80, 6, False), ArmSpec("128x8+gp", 128, 8, True)],
        args,
    )
    assert summary["test_a_pass"] is True
    assert summary["large_passing_arms"] == ["128x8+gp"]
    assert summary["arm_summaries"]["128x8+gp"]["selected_lr"] == 1e-3
    assert summary["arm_summaries"]["128x8+gp"][
        "incomplete_lr_seed_sets_excluded"
    ] == {"0.0005": [1]}
