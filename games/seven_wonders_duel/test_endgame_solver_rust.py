"""Gate: the Rust endgame solver must answer exactly what the Python one does.

The solver's output becomes training labels that the loss treats as ground
truth, so "close enough" is not a category here. Every banked position is
compared on the regime, on the value of *every* legal action, and on the set of
actions proven optimal -- not just the root value, since a solver that agreed on
the root while mispricing the alternatives would still poison a policy label.

The corpus is a set of recipes (seed, bot pairing, ply) plus a fingerprint of
the state each lands on, so a position that the engine no longer produces is
reported as stale rather than silently skipped or, worse, compared against a
different position than the one that was solved.
"""

from __future__ import annotations

import pytest

from . import endgame_corpus as corpus

seven_wonders_rust = pytest.importorskip("seven_wonders_rust")


@pytest.fixture(scope="module")
def records():
    rows = corpus.load()
    if not rows:
        pytest.skip("no endgame corpus; run endgame_corpus --build")
    return rows


def test_corpus_still_regenerates(records):
    """A stale corpus gates nothing, and does it quietly."""

    stale = [r for r in records if corpus.regenerate(r) is None]
    assert not stale, (
        f"{len(stale)} of {len(records)} positions no longer regenerate; "
        "rebuild with: python -m games.seven_wonders_duel.endgame_corpus --build"
    )


def test_rust_matches_the_python_reference(records):
    report = corpus.check(corpus.rust_solver())
    assert report.problems == [], str(report)
    assert report.checked == len(records), str(report)


def test_both_regimes_are_covered(records):
    """Chance-free positions exercise alpha-beta; the rest exercise expectimax.

    Both matter, and the expectimax half more than a Kingdomino-shaped intuition
    suggests: most real 7WD endgames still contain chance, so a solver that only
    handled the chance-free ones would decline the majority of them.
    """

    report = corpus.check(corpus.rust_solver())
    assert report.regimes.get("exact", 0) > 0
    assert report.regimes.get("exact_expectimax", 0) > 0


def test_value_only_mode_agrees_on_the_root(records):
    """The cheap mode may leave alternatives as bounds -- never the root."""

    exact = corpus.rust_solver(policy_mode="exact")
    value_only = corpus.rust_solver(policy_mode="value_only")
    compared = 0
    for record in records:
        game = corpus.regenerate(record)
        if game is None:
            continue
        full, cheap = exact(game), value_only(game)
        if full is None or cheap is None:
            continue
        compared += 1
        assert cheap["root_value"] == pytest.approx(full["root_value"], abs=1e-9)
        assert cheap["regime"] == full["regime"]
    assert compared > 0
