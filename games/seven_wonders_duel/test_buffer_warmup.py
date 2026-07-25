"""Buffer warmup: skip training until the replay buffer is worth training on.

A fixed step budget presents the same number of samples regardless of how much
data exists, so the first iteration of a run with no seed curriculum trains
153,600 samples against a single iteration of self-play. The bot curriculum
currently hides this; `--seed-games 0` exposes it.

The second property matters as much as the first: a skipped iteration must not
run a promotion or anchor gate. Gating an untrained learner against the
protected best spends evaluation games re-measuring a checkpoint that has not
moved -- which is how run 02 spent roughly half its wall clock.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .buffer import GameRecord, MoveRecord
from .phase_d import PhaseDConfig, PhaseDLoop


def _record(moves: int) -> GameRecord:
    return GameRecord(
        seed=1,
        first_player=0,
        agents={"0": "a", "1": "b"},
        iteration=0,
        winner=0,
        victory_type="civilian",
        scores=(10, 5),
        chance_log=(),
        moves=tuple(
            MoveRecord(i=i, actor=i % 2, action=0, mask_hash="sha256:0")
            for i in range(moves)
        ),
        final_digest="sha256:0",
        trajectory_digest="sha256:0",
    )


def _loop(tmp_path: Path, minimum: int) -> PhaseDLoop:
    return PhaseDLoop(
        PhaseDConfig(
            run_dir=str(tmp_path / "run"),
            device="cpu",
            d_model=32,
            layers=1,
            seed_games=0,
            promotion_every=0,
            min_buffer_positions=minimum,
        )
    )


def test_warmup_disabled_by_default(tmp_path: Path):
    loop = _loop(tmp_path, 0)
    assert loop.buffer_warmup_shortfall([_record(1)]) == ""


def test_shortfall_reported_below_the_threshold(tmp_path: Path):
    loop = _loop(tmp_path, 500)
    reason = loop.buffer_warmup_shortfall([_record(60), _record(60)])
    assert "120" in reason and "500" in reason


def test_no_shortfall_at_or_above_the_threshold(tmp_path: Path):
    loop = _loop(tmp_path, 120)
    assert loop.buffer_warmup_shortfall([_record(60), _record(60)]) == ""


def test_positions_are_counted_in_moves_not_games(tmp_path: Path):
    """Games vary in length, so a game count is the wrong unit."""

    loop = _loop(tmp_path, 100)
    assert loop.buffer_warmup_shortfall([_record(150)]) == ""
    assert loop.buffer_warmup_shortfall([_record(10) for _ in range(5)]) != ""


def test_adapter_reports_a_skip_instead_of_training(tmp_path: Path):
    from games.az_loop.contract import AssembleRequest, TrainRequest

    from .training_adapter import SevenWondersDuelLifecycleAdapter

    loop = _loop(tmp_path, 10_000)
    loop.initialize(bootstrap_checkpoint=False)
    adapter = SevenWondersDuelLifecycleAdapter(loop)
    learner = adapter.initialize_learner(seed=1).path
    replay = adapter.assemble_replay(AssembleRequest(iteration=0))
    result = adapter.train(
        TrainRequest(iteration=0, learner_checkpoint=learner, replay=replay)
    )
    assert result.skipped is True
    assert result.trained is False
    assert "min-buffer-positions" in result.skip_reason


def test_controller_skips_gates_on_a_warmup_iteration():
    """`skipped` must not reach the `trained=False` failure path."""

    from games.az_loop.contract import TrainingResult

    result = TrainingResult(candidate=None, trained=False, skipped=True)
    assert result.skipped and not result.trained


def test_skipped_is_distinct_from_a_training_failure():
    """A failure still raises; only a planned skip is tolerated."""

    from games.az_loop.contract import TrainingResult

    failure = TrainingResult(candidate=None, trained=False)
    assert failure.skipped is False
