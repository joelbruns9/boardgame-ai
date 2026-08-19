"""The frozen proven-endgame benchmark.

This is the project's only ground truth for the value head, so the failure that
matters is a quiet one: a benchmark that reconstructs a *different* position than
it was solved at would report a confident number about nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .proven_benchmark import BENCHMARK_PATH, load_benchmark, rebuild_position


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return load_benchmark(BENCHMARK_PATH)


def test_the_committed_benchmark_is_large_enough_to_resolve_an_arm_difference(rows):
    """133 positions gave a standard error near 0.026, which cannot separate
    0.221 from 0.19 -- the size of difference actually at stake."""

    assert len(rows) >= 1000


def test_every_position_rebuilds_at_the_actor_it_was_solved_for(rows):
    """A drifting engine must fail loudly, not score a different position.

    Sampled rather than exhaustive: replaying a thousand prefixes is a minute of
    CPU, and a schema or engine change breaks all of them, not a scattered few.
    """

    for row in rows[::37]:
        game, actor = rebuild_position(row)
        assert actor == row["actor"]
        present = sum(1 for c in game.tableau.cards.values() if c.present)
        assert present == row["cards_left"]


def test_every_banked_position_is_fully_revealed(rows):
    """The property that makes a proof possible.

    An unrevealed card is a chance edge, and crossing one turns the solve into an
    expectimax whose scalar cannot be scored against a three-class head. If a
    face-down card ever appears here, the values are no longer all provable.
    """

    for row in rows[::37]:
        game, _ = rebuild_position(row)
        assert not any(
            card.present and not card.revealed
            for card in game.tableau.cards.values()
        )


def test_proven_values_are_decided(rows):
    """Exact proofs over a fully-revealed board are -1, 0 or +1."""

    assert {round(row["value"], 6) for row in rows} <= {-1.0, 0.0, 1.0}


def test_the_benchmark_is_not_one_sided(rows):
    """A set that is 90% wins would make sign agreement meaningless."""

    wins = sum(1 for row in rows if row["value"] > 0)
    assert 0.4 < wins / len(rows) < 0.6


def test_sign_agreement_ignores_drawn_positions():
    """A draw has no sign, and counting it as agreement would inflate the rate."""

    from .proven_benchmark import _sign_agreement

    assert _sign_agreement([0.5, -0.5], [1.0, -1.0]) == 1.0
    assert _sign_agreement([0.5, 0.5], [1.0, -1.0]) == 0.5
    # The drawn row is excluded, not scored as a hit.
    agreement, decisive = _sign_agreement(
        [0.5, 0.9], [1.0, 0.0], with_count=True
    )
    assert decisive == 1
    assert agreement == 1.0


def test_a_terminal_score_difference_is_a_distinct_error_type():
    """The distinction three call sites used to make by substring-matching.

    A caller reading positions BEFORE the end can accept a differing terminal
    score; none may accept a mask divergence, where the recorded actions after
    that point were chosen for a position that no longer exists. Subclassing
    keeps every existing `except ReplayMismatchError` working.
    """

    from .buffer import FinalDigestMismatchError, ReplayMismatchError

    assert issubclass(FinalDigestMismatchError, ReplayMismatchError)
