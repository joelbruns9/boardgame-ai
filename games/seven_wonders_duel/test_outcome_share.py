"""Distance-scaled outcome share of the value target (`--outcome-share-decay`)."""

from __future__ import annotations

import random

import pytest
import torch

from .buffer import GameRecorder
from .codec import legal_action_indices
from .dataset import collate, examples_from_record
from .game import Phase
from .train import value_targets


def _record(seed: int, root=0.2):
    recorder = GameRecorder(seed, agents={"p0": "test", "p1": "test"})
    rng = random.Random(seed)
    while recorder.game.phase is not Phase.COMPLETE:
        choice = rng.choice(legal_action_indices(recorder.game))
        recorder.play(choice, policy_target={choice: 1.0}, root_value=root)
    return recorder.finish()


def _batch(plies, root=0.2, cls=0):
    rows = len(plies)
    p = (1 + root) / 2
    return {
        "value_class": torch.full((rows,), cls, dtype=torch.long),
        "value_soft": torch.tensor([[p, 0.0, 1 - p]]).repeat(rows, 1),
        "value_soft_valid": torch.ones(rows, dtype=torch.bool),
        "plies_to_end": torch.tensor(plies, dtype=torch.long),
    }


def test_derivation_counts_moves_left_to_the_end():
    record = _record(91)
    examples = examples_from_record(record)
    last = len(record.moves) - 1
    position = {move.i: n for n, move in enumerate(record.moves)}
    assert examples
    for e in examples:
        assert e.plies_to_end == last - position[e.move_index]
    batch = collate(examples)
    assert batch["plies_to_end"].tolist() == [e.plies_to_end for e in examples]


def test_the_rust_derivation_path_agrees():
    from .control_table import ensure_rust_table
    from .dataset import derive_records_rust

    ensure_rust_table()
    record = _record(92)
    (rust_rows, _stats), = derive_records_rust([record])
    python_rows = examples_from_record(record)
    assert [e.plies_to_end for e in rust_rows] == [e.plies_to_end for e in python_rows]


def test_share_decays_from_the_end_and_stops_at_the_floor():
    decay, floor, bootstrap = 0.9, 0.2, 0.5
    plies = [0, 1, 5, 40, -1]
    flat = value_targets(
        _batch(plies), value_bootstrap=bootstrap,
        outcome_share_decay=decay, outcome_share_floor=floor,
    )["flat"]
    p = 0.6  # search's win probability at root value 0.2
    for row, d in enumerate(plies):
        share = floor if d < 0 else max(floor, (1 - bootstrap) * decay ** d)
        # Realised win: target = share * 1 + (1 - share) * p on the win class.
        assert float(flat[row, 0]) == pytest.approx(share + (1 - share) * p)
        assert float(flat[row].sum()) == pytest.approx(1.0)


def test_off_is_the_flat_blend_exactly():
    batch = _batch([0, 3, 30])
    off = value_targets(batch, value_bootstrap=0.5)["flat"]
    zero = value_targets(
        batch, value_bootstrap=0.5, outcome_share_decay=0.0, outcome_share_floor=0.0
    )["flat"]
    assert torch.equal(off, zero)
    # Decay 1 would be the flat blend too; just under it is close at the end.
    near = value_targets(batch, value_bootstrap=0.5, outcome_share_decay=0.999999)["flat"]
    assert torch.allclose(near, off, atol=1e-4)


def test_proofs_still_replace_the_scheduled_target():
    batch = _batch([20, 20])
    batch["value_solver"] = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    batch["value_solver_valid"] = torch.tensor([True, False])
    batch["value_solver_exact"] = torch.tensor([True, False])
    flat = value_targets(
        batch, value_bootstrap=0.5, outcome_share_decay=0.95, outcome_share_floor=0.2,
    )["flat"]
    assert flat[0].tolist() == [0.0, 0.0, 1.0]
    assert float(flat[1, 0]) > 0.5


def test_missing_distances_are_refused():
    batch = _batch([0])
    del batch["plies_to_end"]
    with pytest.raises(ValueError, match="plies_to_end"):
        value_targets(batch, value_bootstrap=0.5, outcome_share_decay=0.9)


def test_config_validation():
    from .phase_d import PhaseDConfig

    PhaseDConfig(value_bootstrap=0.5, outcome_share_decay=0.97,
                 outcome_share_floor=0.2).validate()
    with pytest.raises(ValueError, match="outcome_share_floor"):
        PhaseDConfig(value_bootstrap=0.5, outcome_share_decay=0.97,
                     outcome_share_floor=0.6).validate()
    with pytest.raises(ValueError, match="needs outcome_share_decay"):
        PhaseDConfig(outcome_share_floor=0.2).validate()
    with pytest.raises(ValueError, match="outcome_share_decay"):
        PhaseDConfig(outcome_share_decay=1.0).validate()
