from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import numpy as np
import pytest

from games.kingdomino.action_codec import encode_action
from games.kingdomino.encoder import encode_state, redeterminize
from games.kingdomino.game import GameState, Phase
from games.kingdomino.self_play import Example, ReplayBuffer
from games.kingdomino.web_app import state_from_debug_json, state_to_public_json


def _advance(seed: int, steps: int = 12) -> GameState:
    state = GameState.new(seed=seed)
    rng = random.Random(seed)
    for _ in range(steps):
        if state.phase == Phase.GAME_OVER:
            break
        state = state.step(rng.choice(state.legal_actions()))
    assert state.phase != Phase.GAME_OVER
    return state


def _encoded_tuple_equal(a, b) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def test_example_and_replay_buffer_schema_cannot_store_true_deck_order():
    """Training examples are encoded tensors plus sparse targets, not states.

    This is the buffer-storage half of the Milestone 4 no-leak gate: once a
    position enters the replay buffer, there is no GameState, deck, history, or
    public_state object where true hidden order could be recovered later.
    """
    field_names = {f.name for f in dataclasses.fields(Example)}
    forbidden = {
        "state",
        "game_state",
        "public_state",
        "deck",
        "deck_order",
        "hidden_deck",
        "history",
        "debug",
    }
    assert field_names.isdisjoint(forbidden)
    assert field_names == {
        "my_board",
        "opp_board",
        "flat",
        "policy_idx",
        "policy_val",
        "legal_idx",
        "z",
        "own_score",
        "opp_score",
        "win_target",
        "root_prior_idx",
        "root_prior_val",
        "root_visit_count",
        "iteration",
    }

    ex = Example(
        my_board=np.zeros((9, 13, 13), dtype=np.float16),
        opp_board=np.zeros((9, 13, 13), dtype=np.float16),
        flat=np.zeros(261, dtype=np.float16),
        policy_idx=np.array([0], dtype=np.int32),
        policy_val=np.array([1.0], dtype=np.float32),
        legal_idx=np.array([0, 1], dtype=np.int32),
        z=0.0,
        own_score=0.0,
        opp_score=0.0,
        win_target=0.5,
        root_prior_idx=np.array([0], dtype=np.int32),
        root_prior_val=np.array([1.0], dtype=np.float32),
        root_visit_count=np.array([1], dtype=np.int32),
        iteration=7,
    )
    buf = ReplayBuffer(capacity=4)
    buf.add([ex])
    assert len(buf) == 1
    assert isinstance(buf.data[0], Example)
    assert not any(hasattr(buf.data[0], name) for name in forbidden)


def test_self_play_storage_is_invariant_to_hidden_deck_order():
    """The tensors/targets stored at a root do not encode hidden deck order."""
    state = _advance(seed=123, steps=18)
    assert len(state.deck) >= 8

    shuffled = state.copy()
    shuffled.deck = list(reversed(shuffled.deck))
    assert set(shuffled.deck) == set(state.deck)
    assert shuffled.deck != state.deck

    assert _encoded_tuple_equal(encode_state(state, 0), encode_state(shuffled, 0))
    assert _encoded_tuple_equal(encode_state(state, 1), encode_state(shuffled, 1))

    legal_a = {encode_action(a, state) for a in state.legal_actions()}
    legal_b = {encode_action(a, shuffled) for a in shuffled.legal_actions()}
    assert legal_a == legal_b


def test_advisor_import_hidden_deck_order_is_encoding_inert():
    """Advisor JSON may carry a bag list, but the encoder consumes membership only."""
    state = _advance(seed=77, steps=14)
    payload = state_to_public_json(state, include_debug=True)
    assert "debug" in payload and "deck" in payload["debug"]

    reversed_payload = state_to_public_json(state, include_debug=True)
    reversed_payload["debug"]["deck"] = list(reversed(payload["debug"]["deck"]))
    assert reversed_payload["debug"]["deck"] != payload["debug"]["deck"]

    imported_a = state_from_debug_json(payload)
    imported_b = state_from_debug_json(reversed_payload)
    assert imported_a.deck != imported_b.deck
    assert set(imported_a.deck) == set(imported_b.deck)

    assert _encoded_tuple_equal(encode_state(imported_a, 0), encode_state(imported_b, 0))
    assert _encoded_tuple_equal(encode_state(imported_a, 1), encode_state(imported_b, 1))
    assert {encode_action(a, imported_a) for a in imported_a.legal_actions()} == {
        encode_action(a, imported_b) for a in imported_b.legal_actions()
    }


def test_extension_builds_advisor_deck_as_sorted_membership_not_observed_order():
    """Static guard for the BGA extension's hidden-bag construction."""
    content = Path("extension_kingdomino/content.js").read_text(encoding="utf-8")
    hidden_deck_pos = content.index("const hiddenDeck = allDominoIds")
    snippet = content[hidden_deck_pos:hidden_deck_pos + 220]
    assert ".filter((n) => !visibleSet[n])" in snippet
    assert ".sort((a, b) => a - b)" in snippet
    assert "deck: hiddenDeck" in content
    assert "deck_count: hiddenDeck.length" in content


def test_python_redeterminize_draws_independent_orders_from_same_public_bag():
    state = _advance(seed=91, steps=20)
    base_public = encode_state(state, 0)
    base_bag = set(state.deck)

    orders = set()
    for seed in range(12):
        det = redeterminize(state, random.Random(seed))
        assert set(det.deck) == base_bag
        assert det.current_row == state.current_row
        assert det.pending_claims == state.pending_claims
        assert det.next_claims == state.next_claims
        assert _encoded_tuple_equal(base_public, encode_state(det, 0))
        orders.add(tuple(det.deck))

    assert len(orders) > 1


def test_rust_redeterminize_draws_independent_orders_from_same_public_bag():
    kr = pytest.importorskip("kingdomino_rust")

    state = kr.batched_new_game(123)
    base_deck = set(state.deck())
    base_row = state.current_row()
    base_encoded = tuple(np.asarray(x) for x in state.encode(0))

    orders = set()
    for seed in range(12):
        det = state.redeterminize(seed)
        assert set(det.deck()) == base_deck
        assert det.current_row() == base_row
        det_encoded = tuple(np.asarray(x) for x in det.encode(0))
        assert _encoded_tuple_equal(base_encoded, det_encoded)
        orders.add(tuple(det.deck()))

    assert len(orders) > 1
