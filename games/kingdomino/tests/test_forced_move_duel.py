"""Correctness gates for the forced-move disagreement-duel harness."""

import math

import numpy as np
import pytest

from games.kingdomino.action_codec import encode_action
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.forced_move_duel import (
    build_summary,
    extract_disagreements,
    prepare_forced_state,
)
from games.kingdomino.game import GameState, Phase
from games.kingdomino.nnue.sparse_encoder import swap_players
from games.kingdomino.denial_search import public_state_key


def _deck8_first_selection(seed=17):
    state = GameState.new(seed=seed)
    while not (
        state.phase == Phase.PLACE_AND_SELECT
        and state.actor_index == 0
        and len(state.deck) == 8
    ):
        assert state.phase != Phase.GAME_OVER
        state = state.step(state.legal_actions()[0])
    return state


def _zero_evaluator(my_board, opp_board, flat, legal_indices):
    batch = int(np.asarray(my_board).shape[0])
    return np.zeros(batch, dtype=np.float32), [
        np.zeros(len(indices), dtype=np.float32) for indices in legal_indices
    ]


def test_extract_disagreements_freezes_only_changed_top_actions():
    base = {"results": [
        {"position_index": 0, "configured_budget": 6400, "arm": "control",
         "top_action_idx": 10, "top_pick_rank": 0, "seed": 100},
        {"position_index": 1, "configured_budget": 6400, "arm": "control",
         "top_action_idx": 20, "top_pick_rank": 1, "seed": 101},
    ]}
    sampled = {"results": [
        {"position_index": 0, "configured_budget": 6400,
         "top_action_idx": 11, "top_pick_rank": 0},
        {"position_index": 1, "configured_budget": 6400,
         "top_action_idx": 20, "top_pick_rank": 1},
    ]}

    rows = extract_disagreements(base, sampled, budget=6400)

    assert rows == [{
        "position_index": 0,
        "control_action_idx": 10,
        "split_action_idx": 11,
        "control_pick_rank": 0,
        "split_pick_rank": 0,
        "pick_changed": False,
        "paired_search_seed": 100,
    }]


def test_prepare_forced_state_pairs_deck_order_and_exact_player_relabeling():
    root = _deck8_first_selection()
    action_idx = encode_action(root.legal_actions()[0], root)

    plain, chooser, deck_order = prepare_forced_state(
        root, action_idx=action_idx, deck_seed=1234, mirrored=False
    )
    mirrored, mirrored_chooser, mirrored_deck = prepare_forced_state(
        root, action_idx=action_idx, deck_seed=1234, mirrored=True
    )

    assert deck_order == mirrored_deck
    assert chooser == root.current_actor
    assert mirrored_chooser == 1 - chooser
    assert public_state_key(mirrored) == public_state_key(swap_players(plain))


def test_summary_averages_mirrors_then_seeds_then_positions():
    disagreements = [
        {"position_index": 0, "pick_changed": True},
        {"position_index": 1, "pick_changed": False},
    ]
    results = []
    # Position 0 has split-control margin +4. Position 1 has -2. Equal position
    # weighting therefore produces +1 regardless of the number of raw cells.
    for position_index, delta in ((0, 4), (1, -2)):
        for continuation_index in (0, 1):
            for mirror in (0, 1):
                common = {
                    "position_index": position_index,
                    "continuation_index": continuation_index,
                    "mirror": mirror,
                }
                results.append({
                    **common,
                    "arm": "control_forced",
                    "chooser_margin": 10,
                    "chooser_points": 0.5,
                })
                results.append({
                    **common,
                    "arm": "split_forced",
                    "chooser_margin": 10 + delta,
                    "chooser_points": 1.0 if delta > 0 else 0.0,
                })

    summary = build_summary(results, disagreements)

    assert summary["completed_seed_pairs"] == 4
    assert summary["completed_positions"] == 2
    assert math.isclose(
        summary["all_disagreements"]["mean_chooser_margin_delta"], 1.0
    )
    assert summary["pick_changing_disagreements"]["positions"] == 1
    assert math.isclose(
        summary["pick_changing_disagreements"]["mean_chooser_margin_delta"], 4.0
    )
    assert summary["mean_mirror_margin_abs_difference"] == 0.0


def test_batched_from_states_plays_only_the_supplied_continuation():
    import kingdomino_rust as kr

    root = _deck8_first_selection()
    child = root.step(root.legal_actions()[0])
    rust_state = _rust_state_from_python(child)
    mcts = kr.BatchedMCTS.from_states(
        [rust_state], [987654321], 1,
        leaf_batch=1,
        dirichlet_eps=0.0,
        temp_moves=0,
        exact_endgame_max_secs=0.0,
    )
    finished = []
    while not mcts.done():
        my, opp, flat, legal = mcts.step()
        values, gathered = _zero_evaluator(my, opp, flat, legal)
        finished.extend(mcts.update(values, gathered))

    assert len(finished) == 1
    seed, examples, scores = finished[0]
    assert seed == 987654321
    # A full game records 52 decisions. Starting immediately after the first
    # deck=8 selection leaves three current-round actions, four actions in each
    # of the next two rounds, and four final placements: fifteen in total.
    assert len(examples) == 15
    assert len(scores) == 3


def test_batched_from_states_rejects_duplicate_batch_seeds():
    import kingdomino_rust as kr

    root = _deck8_first_selection()
    rust_state = _rust_state_from_python(root.step(root.legal_actions()[0]))
    with pytest.raises(ValueError, match="unique game_seeds"):
        kr.BatchedMCTS.from_states([rust_state, rust_state], [5, 5], 1)
