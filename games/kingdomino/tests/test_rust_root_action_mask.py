from __future__ import annotations

import numpy as np
import pytest

from games.kingdomino.action_codec import encode_action
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.game import GameState


class _UniformEvaluator:
    def __call__(self, my_board, opp_board, flat, legal_indices):
        del my_board, opp_board, flat
        values = np.zeros(len(legal_indices), dtype=np.float32)
        policies = [
            np.ones(len(indices), dtype=np.float32) for indices in legal_indices
        ]
        return values, policies


def test_advisor_root_action_mask_survives_missing_child_recovery():
    rust = pytest.importorskip("kingdomino_rust")
    state = GameState.new(seed=123, start_player=0)
    rust_state = _rust_state_from_python(state)
    legal = sorted(int(encode_action(action, state)) for action in state.legal_actions())
    allowed = legal[:2]

    children, _root_value = rust.advisor_open_loop_search(
        rust_state,
        _UniformEvaluator(),
        32,
        seed=7,
        leaf_batch=4,
        root_allowed_actions=allowed,
    )

    assert sorted(int(child[0]) for child in children) == allowed
    assert sum(int(child[1]) for child in children) == 32


def test_advisor_root_action_mask_rejects_any_nonlegal_action():
    rust = pytest.importorskip("kingdomino_rust")
    state = GameState.new(seed=123, start_player=0)
    rust_state = _rust_state_from_python(state)
    legal = sorted(int(encode_action(action, state)) for action in state.legal_actions())

    with pytest.raises(ValueError, match="non-legal root action"):
        rust.advisor_open_loop_search(
            rust_state,
            _UniformEvaluator(),
            1,
            seed=7,
            root_allowed_actions=[legal[0], 65535],
        )
