"""Short-term value targets: the TD(lambda) return over later recorded values."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

from .buffer import GameRecorder
from .codec import legal_action_indices
from .dataset import SHORT_TERM_DECAY, collate, examples_from_record, short_term_values
from .game import Phase


def _move(i, actor, root=None, solver=None):
    return SimpleNamespace(i=i, actor=actor, root_value=root, solver_value=solver, search_lambda=0.0)


def test_return_is_the_decayed_average_of_later_values_then_the_result():
    d = 0.5
    record = SimpleNamespace(
        winner=0,
        moves=[_move(0, 0, 0.1), _move(1, 1, 0.4), _move(2, 0, 0.3)],
    )
    t = short_term_values(record, decay=d)
    # Player 0 won: past the last move the return is +1 in player 0's frame.
    assert t[2] == pytest.approx(1.0)
    # Before move 2 is recorded: (1-d)*0.3 + d*1.0 = 0.65 (p0 frame). Move 1's
    # actor is player 1, so its target is -0.65.
    assert t[1] == pytest.approx(-0.65)
    # Move 1's value 0.4 is player 1's; in p0's frame -0.4:
    # (1-d)*(-0.4) + d*0.65 = 0.125, and move 0's actor is player 0.
    assert t[0] == pytest.approx(0.125)


def test_a_proof_is_preferred_and_a_valueless_move_passes_the_return_through():
    record = SimpleNamespace(
        winner=1,
        moves=[
            _move(0, 0, 0.9),
            _move(1, 1, root=-0.9, solver=1.0),  # proof overrides the search value
            _move(2, 0, None),  # a bot move: nothing recorded
        ],
    )
    d = 0.5
    t = short_term_values(record, decay=d)
    tail = -1.0  # player 1 won, p0 frame
    assert t[2] == pytest.approx(tail)
    assert t[1] == pytest.approx(-tail)  # move 2 recorded nothing: unchanged
    after_1 = (1 - d) * (-1.0) + d * tail  # the PROOF (+1 for player 1) = -1 in p0 frame
    assert t[0] == pytest.approx(after_1)


def test_a_drawn_game_ends_at_zero():
    record = SimpleNamespace(winner=None, moves=[_move(0, 0, None)])
    assert short_term_values(record)[0] == 0.0


def test_real_derivation_attaches_it_and_collate_maps_it():
    recorder = GameRecorder(83, agents={"p0": "test", "p1": "test"})
    rng = random.Random(8301)
    while recorder.game.phase is not Phase.COMPLETE:
        choice = rng.choice(legal_action_indices(recorder.game))
        recorder.play(choice, policy_target={choice: 1.0}, root_value=0.2)
    record = recorder.finish()
    examples = examples_from_record(record)
    expected = short_term_values(record, SHORT_TERM_DECAY)
    assert examples and all(e.short_term_value is not None for e in examples)
    for e in examples:
        assert e.short_term_value == pytest.approx(expected[e.move_index])
    batch = collate(examples[:4])
    assert batch["value_short_valid"].all()
    win = (1 + torch.tensor([e.short_term_value for e in examples[:4]])) / 2
    assert torch.allclose(batch["value_short"][:, 0], win)
    assert torch.allclose(batch["value_short"][:, 1], torch.zeros(4))


def _outputs(rows):
    return {
        "policy": torch.zeros(rows, 1202),
        "value": torch.zeros(rows, 3),
        "joint7": torch.zeros(rows, 7),
        "margin": torch.zeros(rows),
        "military": torch.zeros(rows),
        "science": torch.zeros(rows, 2),
    }


def test_the_loss_blends_it_in_and_zero_weight_changes_nothing():
    from .train import compute_losses

    recorder = GameRecorder(84, agents={"p0": "test", "p1": "test"})
    rng = random.Random(8401)
    while recorder.game.phase is not Phase.COMPLETE:
        choice = rng.choice(legal_action_indices(recorder.game))
        recorder.play(choice, policy_target={choice: 1.0}, root_value=-0.4)
    examples = examples_from_record(recorder.finish())[:6]
    batch = collate(examples)
    outputs = _outputs(len(examples))
    _, off = compute_losses(outputs, batch, value_bootstrap=0.5)
    _, zero = compute_losses(outputs, batch, value_bootstrap=0.5, short_term_value_weight=0.0)
    _, on = compute_losses(outputs, batch, value_bootstrap=0.5, short_term_value_weight=0.25)
    assert zero["value"] == pytest.approx(off["value"])
    # Uniform logits: cross-entropy against ANY distribution is log(3), so the
    # blend is checked through its gradient-free cousin instead -- a skewed head.
    skewed = _outputs(len(examples))
    skewed["value"] = torch.tensor([[3.0, 0.0, -3.0]]).repeat(len(examples), 1)
    _, off_s = compute_losses(skewed, batch, value_bootstrap=0.5)
    _, on_s = compute_losses(skewed, batch, value_bootstrap=0.5, short_term_value_weight=0.25)
    assert on_s["value"] != pytest.approx(off_s["value"])
    assert on["value"] == pytest.approx(off["value"])


def test_the_rust_derivation_path_attaches_the_same_targets():
    """The run derives with --derive-backend rust; both paths must agree."""

    from .control_table import ensure_rust_table
    from .dataset import derive_records_rust

    ensure_rust_table()
    recorder = GameRecorder(85, agents={"p0": "test", "p1": "test"})
    rng = random.Random(8501)
    while recorder.game.phase is not Phase.COMPLETE:
        choice = rng.choice(legal_action_indices(recorder.game))
        recorder.play(choice, policy_target={choice: 1.0}, root_value=rng.uniform(-1, 1))
    record = recorder.finish()
    python_rows = examples_from_record(record)
    (rust_rows, _stats), = derive_records_rust([record])
    assert [e.short_term_value for e in rust_rows] == pytest.approx(
        [e.short_term_value for e in python_rows]
    )
