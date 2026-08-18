"""The opponent-reply auxiliary target, and the join that produces it.

The head itself is ordinary. The pairing is not: fast decisions are dropped at
the example boundary, so "the next example" can be several actual decisions
later. Pairing there would silently supervise a position against a reply two
plies downstream, and every shape and sum check would still pass.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from .dataset import NUM_ACTIONS, collate, reply_targets
from .train import REPLY_WEIGHT_DEFAULT, build_model, compute_losses, reply_head_from_config
from .test_endgame_solver_self_play import _example


class _Move:
    def __init__(self, i, actor, policy=None, sims=64):
        self.i, self.actor, self.sims = i, actor, sims
        self.policy_target = policy


def _legal(moves, size=3):
    return {m.i: np.arange(size, dtype=np.int16) for m in moves}


def test_the_reply_comes_from_the_next_RAW_move_not_the_next_example():
    """The trap this file exists for. Move 1 is a cheap search that emits no
    example; move 2 is the next EXAMPLE. Row 0's reply must come from move 1's
    absence -- i.e. no target -- and never from move 2, which is two plies away.
    """

    moves = [
        _Move(0, 0, {0: 1.0}),
        _Move(1, 1, None, sims=16),   # cheap: no recorded target
        _Move(2, 0, {1: 1.0}),
    ]
    out = reply_targets(moves, _legal(moves))
    assert 0 not in out, "row 0 was paired past a cheap move"


def test_an_extra_turn_is_skipped_rather_than_blended():
    """The same actor moving twice makes the target 'what do I do next', a
    different question. Blending the two teaches neither, and the encoder's
    tempo primitives already carry the extra-turn signal."""

    moves = [_Move(0, 0, {0: 1.0}), _Move(1, 0, {1: 1.0})]
    assert reply_targets(moves, _legal(moves)) == {}


def test_the_mask_is_the_replys_legal_set_not_its_policy_keys():
    """Under PUCT the target holds only visited actions -- pruning zeroes more
    of them -- so its keys understate what was legal. A head trained on that
    would learn that unvisited legal moves are illegal."""

    moves = [_Move(0, 0, {0: 1.0}), _Move(1, 1, {2: 1.0})]
    legal, target = reply_targets(moves, _legal(moves, size=5))[0]
    assert len(legal) == 5, "mask collapsed onto the visited actions"
    assert target[2] == pytest.approx(1.0)
    assert target.sum() == pytest.approx(1.0)


def test_a_malformed_target_is_skipped_not_renormalised():
    """Renormalising a target that does not sum to one would invent a
    distribution rather than decline to supervise."""

    moves = [_Move(0, 0, {0: 1.0}), _Move(1, 1, {0: 0.3})]
    assert reply_targets(moves, _legal(moves)) == {}


def test_the_last_move_of_a_game_has_no_reply():
    moves = [_Move(0, 0, {0: 1.0})]
    assert reply_targets(moves, _legal(moves)) == {}


# --- the head ---------------------------------------------------------------


def _outputs(rows, reply=True):
    out = {
        "policy": torch.zeros(rows, NUM_ACTIONS),
        "value": torch.zeros(rows, 3),
        "joint7": torch.zeros(rows, 7),
        "margin": torch.zeros(rows),
        "military": torch.zeros(rows),
        "science": torch.zeros(rows, 2),
    }
    if reply:
        out["reply"] = torch.zeros(rows, NUM_ACTIONS)
    return out


def test_the_reply_loss_only_scores_rows_that_have_one():
    legal = np.asarray([0, 1], dtype=np.int16)
    with_reply = _example(reply_legal=legal, reply_target=np.asarray([1.0, 0.0], np.float32))
    batch = collate([with_reply, _example()])
    assert batch["has_reply"].tolist() == [True, False]
    _, parts = compute_losses(_outputs(2), batch)
    assert parts["reply"] > 0.0


def test_a_batch_with_no_replies_contributes_no_reply_loss():
    batch = collate([_example(), _example()])
    _, parts = compute_losses(_outputs(2), batch)
    assert parts["reply"] == 0.0


def test_the_head_is_absent_by_default_and_travels_with_the_weights():
    assert reply_head_from_config({}) is False
    assert reply_head_from_config({"reply_head": True}) is True
    plain = build_model("transformer", 32, 2, None, False, False)
    with_head = build_model("transformer", 32, 2, None, False, True)
    assert "reply" not in plain(_dummy_batch())
    assert "reply" in with_head(_dummy_batch())
    with pytest.raises(RuntimeError):
        plain.load_state_dict(with_head.state_dict())


def _dummy_batch(rows: int = 2, tokens: int = 4):
    from .dataset import MAX_FEATURES

    return {
        "type_ids": torch.zeros(rows, tokens, dtype=torch.long),
        "entity_ids": torch.zeros(rows, tokens, dtype=torch.long),
        "aux_ids": torch.zeros(rows, tokens, dtype=torch.long),
        "features": torch.zeros(rows, tokens, MAX_FEATURES),
        "pad_mask": torch.zeros(rows, tokens, dtype=torch.bool),
    }


def test_a_model_without_the_head_ignores_reply_targets_entirely():
    """Off must mean off: an existing run must not start paying a reply loss
    merely because the buffer now carries the target."""

    legal = np.asarray([0, 1], dtype=np.int16)
    batch = collate([_example(reply_legal=legal, reply_target=np.asarray([1.0, 0.0], np.float32))])
    _, parts = compute_losses(_outputs(1, reply=False), batch)
    assert parts["reply"] == 0.0
