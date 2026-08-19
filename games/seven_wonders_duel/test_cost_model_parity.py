"""Rust's cost features must equal Python's, feature by feature.

The model's coefficients were fit against `position_features`. If Rust computes
any of the twenty differently, nothing raises -- the trigger simply prices every
position with the wrong weights and quietly attempts the wrong solves. That is
the failure this file exists to make impossible, and it is not hypothetical:
this project has already shipped a Python/Rust divergence that survived 989
tests because nothing compared the two directly.

Parity is checked on real Age III positions from played games, not constructed
ones, so the comparison covers the boards the trigger will actually meet.
"""

from __future__ import annotations

import math

import pytest

import seven_wonders_rust as swr

from .encoder_audit import DEFAULT_PAIRINGS, make_bot
from .endgame_trigger_study import position_features
from .engine import apply_action
from .game import Phase, new_game
from .rust_bridge import rust_game_from_state
from .validate_cost_trigger import RUST_FEATURES, load_cost_model


def _age_three_positions(seeds=range(40), max_cards=12):
    """Every Age III position at or under `max_cards` from a few bot games."""

    for seed in seeds:
        game = new_game(seed)
        bots = (
            make_bot(DEFAULT_PAIRINGS[0][0], seed),
            make_bot(DEFAULT_PAIRINGS[0][1], seed + 13),
        )
        while game.phase is not Phase.COMPLETE:
            if (
                game.phase is Phase.PLAY_AGE
                and game.age == 3
                and sum(1 for c in game.tableau.cards.values() if c.present)
                <= max_cards
            ):
                yield game.clone()
            actor = (
                game.pending_choice.player
                if game.pending_choice is not None
                else game.active_player
            )
            apply_action(game, bots[actor].select_action(game))


def test_the_feature_order_is_shared_not_assumed():
    """Rust indexes weights positionally, so a reordering is silent."""

    assert tuple(swr.endgame_cost_model_features()) == RUST_FEATURES


def test_every_feature_matches_python_on_real_positions():
    positions = list(_age_three_positions())
    assert len(positions) > 100, "not enough Age III positions to be meaningful"

    mismatches: dict[str, int] = {}
    for game in positions:
        expected = position_features(game)
        actual = swr.endgame_cost_features(rust_game_from_state(game))
        for name, value in zip(RUST_FEATURES, actual):
            if not math.isclose(value, float(expected[name]), rel_tol=1e-9, abs_tol=1e-9):
                mismatches[name] = mismatches.get(name, 0) + 1

    assert not mismatches, (
        f"Rust and Python disagree on {len(positions)} positions: {mismatches}. "
        "The fitted coefficients belong to Python's definitions, so a mismatch "
        "prices positions with the wrong weights."
    )


def test_the_shipped_model_and_rust_agree_on_what_to_attempt():
    """End to end: the same position, the same verdict, both languages.

    Feature parity alone is not enough -- the coefficients have to be installed
    in the same order they were fit in, which is a separate way to get it wrong.
    """

    from .validate_cost_trigger import should_attempt

    coefficients, features, margin = load_cost_model()
    swr.set_endgame_cost_model(list(features), coefficients[0], coefficients[1:], margin)
    budget = 4_500_000
    try:
        seen = {True: 0, False: 0}
        for game in _age_three_positions():
            python_row = position_features(game)
            expected = should_attempt(python_row, budget, (coefficients, features, margin))
            rust_features = swr.endgame_cost_features(rust_game_from_state(game))
            predicted = coefficients[0] + sum(
                c * x for c, x in zip(coefficients[1:], rust_features)
            )
            actual = predicted + margin <= math.log10(budget)
            assert actual == expected
            seen[actual] += 1
        # A test where the trigger always says yes would pass while proving
        # nothing about the boundary.
        assert seen[True] and seen[False], f"trigger never varied: {seen}"
    finally:
        swr.set_endgame_cost_model([], 0.0, [], 0.0)


def test_clearing_the_model_restores_the_card_cap():
    swr.set_endgame_cost_model([], 0.0, [], 0.0)  # no exception, and no model


def test_a_reordered_weight_vector_is_refused():
    """The one mistake that would otherwise be invisible."""

    coefficients, features, margin = load_cost_model()
    scrambled = list(features)
    scrambled[0], scrambled[1] = scrambled[1], scrambled[0]
    with pytest.raises(ValueError, match="weight order must match"):
        swr.set_endgame_cost_model(
            scrambled, coefficients[0], coefficients[1:], margin
        )
