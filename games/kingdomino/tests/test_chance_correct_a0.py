"""A0 information/probability gates for chance-correct Kingdomino search.

These tests cover the invariants that were not already pinned by
test_denial_search.py and the generic Rust search tests.  They intentionally run
before production observation-split topology exists: A1 may reuse these
contracts, but must not weaken them.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from games.kingdomino.action_codec import encode_action
from games.kingdomino.denial_search import (
    _as_pre_reveal_leaf,
    chance_public_state_key_v1,
    chance_rows,
)
from games.kingdomino.encoder import encode_state, redeterminize
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.game import GameState, Phase, PickAction, TurnAction
from games.kingdomino.mcts_az import make_serial_evaluator
from games.kingdomino.network import KingdominoNet


def _midgame_state(seed: int = 91001, plies: int = 15) -> GameState:
    state = GameState.new(seed=seed)
    rng = random.Random(seed ^ 0xA0)
    for _ in range(plies):
        if state.phase == Phase.GAME_OVER:
            break
        state = state.step(rng.choice(state.legal_actions()))
    return state


def _legal_indices(state: GameState) -> np.ndarray:
    return np.asarray(
        [encode_action(action, state) for action in state.legal_actions()],
        dtype=np.int64,
    )


def test_hidden_deck_permutation_preserves_raw_network_inference_exactly():
    """A0.1: equality must survive the forward pass, not only the encoder."""
    torch.manual_seed(20260808)
    network = KingdominoNet(channels=16, blocks=2, bilinear_dim=16).eval()
    evaluate = make_serial_evaluator(network, device="cpu")
    state = _midgame_state()
    assert len(state.deck) >= 8

    for player in (0, 1):
        encoded = encode_state(state, player)
        indices = _legal_indices(state)
        reference_value, reference_logits = evaluate(*encoded, indices)
        for seed in range(12):
            determinized = redeterminize(state, random.Random(seed + 77))
            assert sorted(determinized.deck) == sorted(state.deck)
            assert chance_public_state_key_v1(determinized) == chance_public_state_key_v1(state)
            candidate_value, candidate_logits = evaluate(
                *encode_state(determinized, player), _legal_indices(determinized)
            )
            assert candidate_value == reference_value
            assert np.array_equal(candidate_logits, reference_logits)


def _boundary_state(seed: int, *, deck_size: int | None = None) -> GameState:
    state = GameState.new(seed=seed)
    rng = random.Random(seed)
    while state.phase != Phase.GAME_OVER:
        at_boundary = (
            state.phase == Phase.PLACE_AND_SELECT
            and state.actor_index + 1 == len(state.pending_claims)
            and (deck_size is None or len(state.deck) == deck_size)
        )
        if at_boundary:
            return state
        state = state.step(rng.choice(state.legal_actions()))
    raise AssertionError("failed to find requested pre-reveal boundary")


def _install_public_row(child: GameState, pre_deal_bag: list[int], row: list[int]) -> GameState:
    out = child.copy()
    remaining = list(pre_deal_bag)
    for tile in row:
        remaining.remove(tile)
    out.current_row = sorted(row)
    out.deck = sorted(remaining)
    return out


def _preferred_rank_policy(state: GameState, rank: int) -> dict[int, float]:
    target = sorted(state.current_row)[rank]
    selected = []
    for action in state.legal_actions():
        picked = action.domino_id if isinstance(action, PickAction) else action.pick_domino_id
        if picked == target:
            selected.append(encode_action(action, state))
    assert selected
    probability = 1.0 / len(selected)
    return {index: probability for index in selected}


def test_distinct_revealed_rows_can_retain_distinct_post_reveal_policies():
    """A0.5: public observations must not alias one post-reveal policy cache."""
    boundary = _boundary_state(92001)
    pre_deal_bag = sorted(boundary.deck)
    child = boundary.step(boundary.legal_actions()[0])
    assert len(pre_deal_bag) - len(child.deck) == 4
    low_row = pre_deal_bag[:4]
    high_row = pre_deal_bag[-4:]
    low = _install_public_row(child, pre_deal_bag, low_row)
    high = _install_public_row(child, pre_deal_bag, high_row)

    low_key = chance_public_state_key_v1(low)
    high_key = chance_public_state_key_v1(high)
    assert low_key != high_key
    policy_cache = {
        low_key: _preferred_rank_policy(low, 0),
        high_key: _preferred_rank_policy(high, 3),
    }
    assert policy_cache[low_key] != policy_cache[high_key]
    assert len(policy_cache) == 2


def test_python_rust_public_keys_and_exact_chance_probabilities_agree():
    """A0.6: cross-language key material and enumerated P(row) are identical."""
    kingdomino_rust = pytest.importorskip("kingdomino_rust")
    state = _midgame_state(seed=93001, plies=19)
    rng = random.Random(17)
    checked = 0
    while state.phase != Phase.GAME_OVER and checked < 18:
        rust_state = _rust_state_from_python(state)
        assert rust_state is not None
        assert hasattr(rust_state, "chance_public_state_key_v1"), "Rust extension needs rebuilding"
        assert chance_public_state_key_v1(state) == bytes(rust_state.chance_public_state_key_v1())
        shuffled = state.copy()
        rng.shuffle(shuffled.deck)
        assert chance_public_state_key_v1(shuffled) == chance_public_state_key_v1(state)
        checked += 1
        state = state.step(rng.choice(state.legal_actions()))
    assert checked == 18

    boundary = _boundary_state(93002, deck_size=8)
    rust_boundary = _rust_state_from_python(boundary)
    expected_rows, mode = chance_rows(boundary.deck, 70, seed=999)
    assert mode == "enumerated" and len(expected_rows) == 70
    outcomes = rust_boundary.chance_outcomes(enum_cap=70, chance_samples=3, seed=999)
    assert [tuple(row) for row, _weight in outcomes] == expected_rows
    assert [weight for _row, weight in outcomes] == pytest.approx([1.0 / 70.0] * 70)
    assert sum(weight for _row, weight in outcomes) == pytest.approx(1.0)

    sampled = rust_boundary.chance_outcomes(enum_cap=1, chance_samples=9, seed=123)
    assert len(sampled) == 9
    assert all(weight == pytest.approx(1.0 / 9.0) for _row, weight in sampled)


def test_every_reachable_pre_reveal_bag_is_divisible_by_four():
    """A0.7: prove the panel precondition over complete reachable trajectories."""
    deal_count = 0
    for seed in range(64):
        state = GameState.new(seed=94000 + seed)
        rng = random.Random(0xA000 + seed)
        while state.phase != Phase.GAME_OVER:
            assert len(state.deck) % 4 == 0
            pre_deal_bag = list(state.deck)
            child = state.step(rng.choice(state.legal_actions()))
            if len(child.deck) != len(pre_deal_bag):
                assert len(pre_deal_bag) - len(child.deck) == 4
                assert len(pre_deal_bag) % 4 == 0
                pre_reveal = _as_pre_reveal_leaf(child, pre_deal_bag)
                assert len(pre_reveal.deck) % 4 == 0
                assert sorted(pre_reveal.deck) == sorted(pre_deal_bag)
                deal_count += 1
            state = child
    assert deal_count == 64 * 11
