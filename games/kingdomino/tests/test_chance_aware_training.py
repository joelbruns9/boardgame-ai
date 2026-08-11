"""Focused tests for the deck=8/12 chance-aware training pilot."""

from __future__ import annotations

import numpy as np
import pytest

from games.kingdomino.encoder import FLAT_LAYOUT, FLAT_SIZE
from games.kingdomino.self_play import (
    Example,
    ReplayBuffer,
    SelfPlayConfig,
    _example_from_rust_tuple,
    _parse_sampled_chance_split_decks,
    _sampled_chance_split_deck_mask,
    play_selfplay_games_batched,
)


def _example(*, progress: float = 0.0, treated: bool = False,
             deck: int = 0) -> Example:
    flat = np.zeros(FLAT_SIZE, dtype=np.float16)
    flat[FLAT_LAYOUT["game_progress"].start] = progress
    empty_i32 = np.zeros(0, dtype=np.int32)
    return Example(
        my_board=np.zeros((9, 13, 13), dtype=np.float16),
        opp_board=np.zeros((9, 13, 13), dtype=np.float16),
        flat=flat,
        policy_idx=empty_i32,
        policy_val=np.zeros(0, dtype=np.float32),
        legal_idx=empty_i32,
        z=0.0,
        own_score=0.0,
        opp_score=0.0,
        win_target=0.5,
        sampled_chance_split_treated=treated,
        root_hidden_deck_count=deck,
        root_turn_slot=0,
    )


def test_sampled_chance_split_deck_parser_is_strict_and_stable():
    assert _parse_sampled_chance_split_decks("") == ()
    assert _parse_sampled_chance_split_decks("12,8,12") == (8, 12)
    assert _sampled_chance_split_deck_mask("8,12") == (1 << 8) | (1 << 12)
    with pytest.raises(ValueError, match="supports only deck counts 8 and 12"):
        _parse_sampled_chance_split_decks("4,8")
    with pytest.raises(ValueError, match="comma-separated"):
        _parse_sampled_chance_split_decks("eight")


def test_sampled_split_configuration_rejects_wrong_engine_and_panels():
    with pytest.raises(ValueError, match="requires engine='batched_open_loop'"):
        play_selfplay_games_batched(
            None,
            SelfPlayConfig(
                engine="batched",
                sampled_chance_split_decks="8,12",
            ),
            n_games=1,
            game_seed_start=1,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        play_selfplay_games_batched(
            None,
            SelfPlayConfig(
                engine="batched_open_loop",
                sampled_chance_split_decks="8,12",
                deck8_chance_enumeration=True,
            ),
            n_games=1,
            game_seed_start=1,
        )


def test_replay_weights_compose_by_max_not_multiplication():
    buffer = ReplayBuffer(capacity=8)
    buffer.add([
        _example(progress=0.1),
        _example(progress=0.1, treated=True, deck=12),
        _example(progress=0.9),
        _example(progress=0.9, treated=True, deck=8),
    ])

    weights = buffer._sampling_weights(
        endgame_oversample=2.0,
        chance_oversample=4.0,
    )
    assert weights is not None
    # Raw weights are [1, 4, 2, 4]. The final item proves the overlap is 4,
    # not the accidental product 8.
    assert weights == pytest.approx(np.asarray([1, 4, 2, 4]) / 11.0)
    treatment = buffer.chance_treatment_stats()
    assert treatment["treated_count"] == 2
    assert treatment["treated_fraction"] == 0.5
    assert treatment["treated_deck8_count"] == 1
    assert treatment["treated_deck12_count"] == 1
    assert treatment["treated_deck8_by_slot"] == [1, 0, 0, 0]
    assert treatment["treated_deck12_by_slot"] == [1, 0, 0, 0]


def test_replay_reports_actual_treated_draws_and_metadata():
    buffer = ReplayBuffer(capacity=4)
    buffer.add([_example(), _example(treated=True, deck=12)])
    buffer.reset_sampling_stats()
    _batch, metadata = buffer.sample_batch(
        64,
        np.random.default_rng(17),
        augment_d4=False,
        chance_split_oversample_weight=4.0,
        return_metadata=True,
    )
    stats = buffer.sampling_stats()
    assert stats["draw_count"] == 64
    assert stats["treated_draw_count"] == sum(
        metadata["sampled_chance_split_treated"])
    assert stats["treated_draw_fraction"] > 0.5
    assert set(metadata["root_hidden_deck_counts"]) <= {0, 12}


def test_rust_tuple_conversion_accepts_current_and_legacy_root_stats():
    ex = _example()
    base = (
        ex.my_board,
        ex.opp_board,
        ex.flat,
        np.asarray([4], dtype=np.int32),
        np.asarray([1.0], dtype=np.float32),
        np.asarray([4], dtype=np.int32),
    )
    priors = (
        np.asarray([4], dtype=np.int32),
        np.asarray([0.75], dtype=np.float32),
        np.asarray([32], dtype=np.int32),
    )
    tail = (0.25, 30.0, 20.0, 1.0, 0)

    current = _example_from_rust_tuple(
        base + (priors + (True, 12, 0),) + tail,
        iteration=9,
    )
    assert current.sampled_chance_split_treated is True
    assert current.root_hidden_deck_count == 12
    assert current.root_turn_slot == 0
    assert current.iteration == 9

    legacy = _example_from_rust_tuple(base + (priors,) + tail)
    assert legacy.sampled_chance_split_treated is False
    assert legacy.root_hidden_deck_count == 0
    assert legacy.root_turn_slot == 0


def test_rust_sampled_split_smoke_reaches_and_tags_both_decks():
    kingdomino_rust = pytest.importorskip("kingdomino_rust")
    mcts = kingdomino_rust.BatchedMCTS(
        1,
        1,
        20260810,
        800,
        leaf_batch=4,
        open_loop=True,
        dirichlet_eps=0.0,
        exact_endgame_max_secs=0.0,
        sampled_chance_split_deck_mask=(1 << 8) | (1 << 12),
    )
    finished = []
    ticks = 0
    max_batch = 0
    while not mcts.done():
        my, _opp, _flat, legal_indices = mcts.step()
        max_batch = max(max_batch, int(my.shape[0]))
        values = np.zeros(len(my), dtype=np.float32)
        # A uniform mock policy maximally fragments 800 visits and may never
        # reach the four-actions-deep reveal. This smoke tests routing/tagging,
        # not realistic network coverage, so deterministically focus one legal
        # continuation at every leaf. Real-network coverage is a separate
        # preflight metric before training.
        gathered = []
        for indices in legal_indices:
            logits = np.full(len(indices), -8.0, dtype=np.float32)
            if len(logits):
                logits[0] = 8.0
            gathered.append(logits)
        finished.extend(mcts.update(values, gathered))
        ticks += 1
        assert ticks < 100_000

    assert len(finished) == 1
    assert int(mcts.deck8_chance_bootstrap_rows) == 0
    assert max_batch <= 4
    assert int(mcts.sampled_chance_split_search_count_deck8) > 0
    assert int(mcts.sampled_chance_split_search_count_deck12) > 0
    assert int(mcts.sampled_chance_split_path_count_deck8) > 0
    assert int(mcts.sampled_chance_split_path_count_deck12) > 0

    raw_examples = finished[0][1]
    converted = [_example_from_rust_tuple(tup) for tup in raw_examples]
    treated8 = [ex for ex in converted
                if ex.sampled_chance_split_treated
                and ex.root_hidden_deck_count == 8]
    treated12 = [ex for ex in converted
                 if ex.sampled_chance_split_treated
                 and ex.root_hidden_deck_count == 12]
    assert len(treated8) == 4
    assert len(treated12) == 4
    assert {ex.root_turn_slot for ex in treated8} == {0, 1, 2, 3}
    assert {ex.root_turn_slot for ex in treated12} == {0, 1, 2, 3}
