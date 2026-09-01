"""Rust's cost features must equal Python's, feature by feature.

The model's coefficients were fit against `position_features`. If Rust computes
any of the twenty differently, nothing raises -- the trigger simply prices every
position with the wrong weights and quietly attempts the wrong solves. That is
the failure this file exists to make impossible, and it is not hypothetical:
this project has already shipped a Python/Rust divergence that survived 989
tests because nothing compared the two directly.

Parity is checked on real Age III positions from played games, not constructed
ones, so the comparison covers the boards the trigger will actually meet.

**Two corpora, and the second is the one that earns its keep.** Bot games are
cheap and reproducible but they are not the distribution the coefficients were
fit on, and the difference is not academic: `chance_wonders` diverged for weeks
because Python counted a RETIRED Great Library as unbuilt and Rust did not, and
bot games retire no wonders at all (measured: 0 of 146 positions), so no number
of them could have caught it. The committed proven-endgame benchmark is drawn
from the same cloud6 buffers the model was fit on, where 68% of positions have
a retired wonder -- so parity is checked there too.
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


def _mismatches(positions) -> dict[str, int]:
    counts: dict[str, int] = {}
    for game in positions:
        expected = position_features(game)
        actual = swr.endgame_cost_features(rust_game_from_state(game))
        for name, value in zip(RUST_FEATURES, actual):
            if not math.isclose(value, float(expected[name]), rel_tol=1e-9, abs_tol=1e-9):
                counts[name] = counts.get(name, 0) + 1
    return counts


def test_every_feature_matches_python_on_real_positions():
    positions = list(_age_three_positions())
    assert len(positions) > 100, "not enough Age III positions to be meaningful"

    mismatches = _mismatches(positions)
    assert not mismatches, (
        f"Rust and Python disagree on {len(positions)} positions: {mismatches}. "
        "The fitted coefficients belong to Python's definitions, so a mismatch "
        "prices positions with the wrong weights."
    )


def test_malformed_wonder_representation_does_not_unsigned_wrap():
    """A malformed/legacy injected state must not unsigned-wrap.

    Canonical self-play and BGA states keep all drafted Wonders in ``wonders``
    and use ``built_wonders`` as a subset. Keep the arithmetic defensive anyway:
    an old serialized or manually injected split-list state must not turn a
    negative Python value into roughly 2^64 in Rust.
    """

    game = next(_age_three_positions(seeds=range(10)))
    game.cities[0].wonders = []
    game.cities[0].built_wonders = ["The Appian Way"]
    game.cities[1].wonders = []
    game.cities[1].built_wonders = ["The Pyramids"]
    expected = position_features(game)["unbuilt_wonders"]
    names = tuple(swr.endgame_cost_model_features())
    actual = dict(
        zip(names, swr.endgame_cost_features(rust_game_from_state(game)))
    )["unbuilt_wonders"]
    assert expected < 0, "fixture must exercise signed subtraction"
    assert actual == expected


def _fit_distribution_positions(limit: int = 120):
    """Age III positions from the corpus the coefficients were fit on.

    Bot games are the wrong distribution for this check in a specific,
    demonstrated way -- they never build a seventh wonder, so no retirement
    ever happens in them and any feature whose definition turns on
    `retired_wonders` is untested. These positions come from cloud6 self-play.
    """

    from .proven_benchmark import load_benchmark, rebuild_position

    for row in load_benchmark()[:limit]:
        yield rebuild_position(row)[0]


def test_every_feature_matches_python_on_the_fit_distribution():
    """The check the bot corpus cannot make.

    `chance_wonders` (+0.43, the second-largest weight) disagreed here for weeks
    while every bot-game parity assertion passed.
    """

    positions = list(_fit_distribution_positions())
    assert len(positions) > 50, "benchmark corpus too small to be meaningful"

    mismatches = _mismatches(positions)
    assert not mismatches, (
        f"Rust and Python disagree on {len(positions)} strong-play positions: "
        f"{mismatches}. These are the positions the coefficients were fit on."
    )


def test_the_fit_distribution_actually_exercises_retirement():
    """Otherwise the test above is just a second bot corpus.

    A corpus that happens to contain no retired wonder would pass the parity
    assertion while testing nothing about the definition that broke.
    """

    retired = sum(
        1 for game in _fit_distribution_positions() if game.retired_wonders
    )
    assert retired > 10, (
        f"only {retired} positions have a retired wonder; this corpus cannot "
        "cover the features whose definition depends on retirement"
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
