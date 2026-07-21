from __future__ import annotations

import copy

import numpy as np
import pytest

from games.kingdomino.action_codec import encode_action
from games.kingdomino.denial_search import (
    DenialSearch,
    EvalStats,
    SearchConfig,
    public_state_key,
)
from games.kingdomino.encoder import encode_state
from games.kingdomino.game import GameState, Phase
from games.kingdomino.reply_pilot import (
    _confirm_rust_parent_tree_labels,
    _confirm_rust_reply_label,
    serialize_rust_reply_example,
    validate_reply_example,
)
from games.kingdomino.tests.test_denial_search import _round_start


kingdomino_rust = pytest.importorskip("kingdomino_rust")
if not hasattr(kingdomino_rust, "denial_forced_tree"):
    pytest.skip("Rust extension needs rebuilding", allow_module_level=True)


def _actor_values_from_flat(flat):
    array = np.asarray(flat, dtype=np.float32)
    return np.tanh(array.sum(axis=1, dtype=np.float32) * np.float32(0.001)).astype(np.float32)


class _EncodedEvaluator:
    def __init__(self):
        self.policy_cache = {}
        self.leaf_cache = {}
        self.stats = EvalStats()

    def policies(self, states):
        for state in states:
            actions = state.legal_actions()
            probability = 1.0 / len(actions)
            self.policy_cache[public_state_key(state)] = {
                int(encode_action(action, state)): probability for action in actions
            }

    def policy(self, state):
        self.policies([state])
        return self.policy_cache[public_state_key(state)]

    def values_p0(self, states):
        out = {}
        for state in states:
            actor = int(state.current_actor)
            _mb, _ob, flat = encode_state(state, actor)
            actor_value = float(_actor_values_from_flat(flat[None, :])[0])
            out[public_state_key(state)] = actor_value if actor == 0 else -actor_value
        return out


def _rust_evaluator(_mb, _ob, flat, indices):
    values = _actor_values_from_flat(flat)
    logits = [np.zeros(len(index), dtype=np.float32) for index in indices]
    return values, logits


def _normalise_rust(result):
    shallow = dict(result)
    shallow["reply_labels"] = [
        {key: value for key, value in reply.items() if key != "_rust_state"}
        for reply in result["reply_labels"]
    ]
    out = copy.deepcopy(shallow)
    out["structure"].pop("rayon_threads", None)
    return out


def _by_pick(rows):
    return {row["pick_domino_id"]: row for row in rows}


def test_serial_rust_tree_matches_python_root_and_reply_values(tmp_path):
    state = next(
        candidate for seed in range(500, 1500)
        if (
            (candidate := _round_start(seed)).pending_claims[0].player
            != candidate.pending_claims[1].player
        )
    )
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    search = DenialSearch(
        _EncodedEvaluator(), checkpoint_path=str(checkpoint),
        config=SearchConfig(pick_plies=2, chance_k=1, root_search_sims=0,
                            placement_top_k=2),
    )
    python_root = search.search_position(state)
    python_replies = search.extract_reply_labels(state, root_label=python_root)
    rust = search.search_position_rust(
        state, rayon_threads=1, rust_evaluator=_rust_evaluator)

    py_root = _by_pick(python_root["per_pick"])
    rs_root = _by_pick(rust["per_pick"])
    assert rs_root.keys() == py_root.keys()
    for pick in py_root:
        assert rs_root[pick]["searched_value_player0"] == pytest.approx(
            py_root[pick]["searched_value_player0"], abs=1e-7)
        assert rs_root[pick]["mc_standard_error"] == pytest.approx(
            py_root[pick]["mc_standard_error"], abs=1e-12)
        assert rs_root[pick]["representative_action_idx"] == (
            py_root[pick]["representative"]["action_idx"])

    py_reply = {row["parent_pick_domino_id"]: row for row in python_replies}
    rs_reply = {row["parent_pick_domino_id"]: row for row in rust["reply_labels"]}
    assert rs_reply.keys() == py_reply.keys()
    for parent_pick in py_reply:
        py_rows = _by_pick(py_reply[parent_pick]["per_pick"])
        rs_rows = _by_pick(rs_reply[parent_pick]["per_pick"])
        assert rs_rows.keys() == py_rows.keys()
        for pick in py_rows:
            assert rs_rows[pick]["searched_value_player0"] == pytest.approx(
                py_rows[pick]["searched_value_player0"], abs=1e-7)
            assert rs_rows[pick]["selected_placement_action_idx"] == (
                py_rows[pick]["selected_placement_action_idx"])
        assert rs_reply[parent_pick]["denial_policy_target"] == pytest.approx(
            py_reply[parent_pick]["denial_policy_target"], abs=1e-7)


def test_rayon_thread_counts_are_output_identical(tmp_path):
    state = _round_start(1701)
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    search = DenialSearch(
        _EncodedEvaluator(), checkpoint_path=str(checkpoint),
        config=SearchConfig(pick_plies=3, chance_k=1, root_search_sims=0,
                            placement_top_k=2),
    )
    outputs = [
        _normalise_rust(search.search_position_rust(
            state, rayon_threads=threads, rust_evaluator=_rust_evaluator))
        for threads in (1, 2, 4)
    ]
    assert outputs[0] == outputs[1] == outputs[2]


def test_rust_matches_python_across_sampled_chance_boundary(tmp_path):
    state = GameState.new(seed=1801)
    while state.phase == Phase.INITIAL_SELECTION:
        state = state.step(state.legal_actions()[0])
    for _ in range(3):
        state = state.step(state.legal_actions()[0])
    assert state.actor_index == 3 and len(state.current_row) == 1
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    search = DenialSearch(
        _EncodedEvaluator(), checkpoint_path=str(checkpoint),
        config=SearchConfig(pick_plies=2, chance_k=1, root_search_sims=0,
                            placement_top_k=1),
    )
    python_root = search.search_position(state)
    rust = search.search_position_rust(
        state, rayon_threads=4, rust_evaluator=_rust_evaluator)

    assert rust["structure"]["chance_events"] > 0
    py_rows = _by_pick(python_root["per_pick"])
    rs_rows = _by_pick(rust["per_pick"])
    assert rs_rows.keys() == py_rows.keys()
    for pick in py_rows:
        assert rs_rows[pick]["searched_value_player0"] == pytest.approx(
            py_rows[pick]["searched_value_player0"], abs=1e-7)
        assert rs_rows[pick]["mc_standard_error"] == pytest.approx(
            py_rows[pick]["mc_standard_error"], abs=1e-7)


def test_rust_reply_serialization_is_self_contained(tmp_path):
    state = next(
        candidate for seed in range(1900, 2500)
        if (
            (candidate := _round_start(seed)).pending_claims[0].player
            != candidate.pending_claims[1].player
        )
    )
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    search = DenialSearch(
        _EncodedEvaluator(), checkpoint_path=str(checkpoint),
        config=SearchConfig(pick_plies=2, chance_k=1, root_search_sims=0,
                            placement_top_k=2),
    )
    result = search.search_position_rust(
        state, rayon_threads=2, rust_evaluator=_rust_evaluator)
    row = serialize_rust_reply_example(
        result["reply_labels"][0], rust_evaluator=_rust_evaluator,
        position_index=0, root_state_key=public_state_key(state), source={},
        calibration=False, min_top_two_margin=0.0,
        max_mc_standard_error=1.0, max_target_entropy=10.0,
        reject_ties=False,
    )

    validate_reply_example(row)
    assert row["state_backend"] == "rust-encoded-v1"
    assert "state" not in row
    assert row["quality_accept"]


def test_fixed_reply_research_reproduces_parent_tree_and_records_cross_seed(tmp_path):
    state = next(
        candidate for seed in range(2500, 3200)
        if ((candidate := _round_start(seed)).pending_claims[0].player
            != candidate.pending_claims[1].player)
    )
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    search = DenialSearch(
        _EncodedEvaluator(), checkpoint_path=str(checkpoint),
        config=SearchConfig(pick_plies=3, chance_k=1, seed=91,
                            root_search_sims=0, placement_top_k=2),
    )
    result = search.search_position_rust(
        state, rayon_threads=2, rust_evaluator=_rust_evaluator)
    label = result["reply_labels"][0]
    cross = _confirm_rust_reply_label(
        search, label, base_seed=91, seed_count=3,
        seed_stride=1_000_003, rayon_threads=2)
    label["cross_seed"] = cross
    assert cross["top_pick_agreement"] == 1.0
    assert cross["max_searched_seed_sd"] == 0.0

    row = serialize_rust_reply_example(
        label, rust_evaluator=_rust_evaluator,
        position_index=0, root_state_key=public_state_key(state), source={},
        calibration=False, min_top_two_margin=0.0,
        max_mc_standard_error=1.0, max_target_entropy=10.0,
        max_searched_seed_sd=0.0, min_top_pick_agreement=1.0,
        reject_ties=False)
    validate_reply_example(row)
    assert row["quality_accept"]


def test_full_parent_tree_confirmation_amortizes_all_reply_labels(tmp_path):
    state = next(
        candidate for seed in range(2500, 3200)
        if ((candidate := _round_start(seed)).pending_claims[0].player
            != candidate.pending_claims[1].player)
    )
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    search = DenialSearch(
        _EncodedEvaluator(), checkpoint_path=str(checkpoint),
        config=SearchConfig(pick_plies=3, chance_k=1, seed=91,
                            root_search_sims=0, placement_top_k=2),
    )
    result = search.search_position_rust(
        state, rayon_threads=2, rust_evaluator=_rust_evaluator)
    confirmation_calls = []
    search_position_rust = search.search_position_rust

    def tracked_search_position_rust(*args, **kwargs):
        confirmation_calls.append(kwargs.get("chance_seed"))
        return search_position_rust(*args, **kwargs)

    search.search_position_rust = tracked_search_position_rust
    confirmations = _confirm_rust_parent_tree_labels(
        search, state, root_result=None, primary_result=result,
        base_seed=91, seed_count=3, seed_stride=1_000_003,
        rayon_threads=2)

    assert confirmation_calls == [1_000_094, 2_000_097]
    assert set(confirmations) == {
        label["parent_pick_domino_id"] for label in result["reply_labels"]}
    for cross in confirmations.values():
        assert cross["confirmation_scope"] == "full-parent-tree"
        assert cross["seeds"] == [91, 1_000_094, 2_000_097]
        assert cross["top_pick_agreement"] == 1.0
        assert cross["max_searched_seed_sd"] == 0.0
