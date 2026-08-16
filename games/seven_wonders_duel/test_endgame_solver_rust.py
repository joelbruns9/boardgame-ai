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


def _deep_position(*, cards: int):
    """An Age III position with roughly `cards` left -- too deep to solve fast."""

    from .encoder_audit import DEFAULT_PAIRINGS, make_bot
    from .engine import apply_action
    from .game import Phase, new_game

    for index in range(40):
        left, right = DEFAULT_PAIRINGS[index % len(DEFAULT_PAIRINGS)]
        game = new_game(index)
        bots = (make_bot(left, index), make_bot(right, index + 10_000))
        while game.phase is not Phase.COMPLETE:
            if game.phase is Phase.PLAY_AGE and game.age == 3:
                present = sum(1 for c in game.tableau.cards.values() if c.present)
                if present == cards:
                    return game.clone()
            actor = (
                game.pending_choice.player
                if game.pending_choice is not None
                else game.active_player
            )
            apply_action(game, bots[actor].select_action(game))
    pytest.skip(f"no Age III position with {cards} cards found")


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


def test_a_budget_of_zero_returns_no_answer(records):
    """The budget is a contract, not a hint.

    The clock is sampled every N nodes for throughput, which quietly made the
    deadline unenforceable for any position that finished inside one sampling
    window: `max_secs=0` still returned complete answers. A late answer is not
    a free bonus -- the caller asked for a bound because something downstream
    depends on it.
    """

    solve = corpus.rust_solver(max_secs=0.0)
    answered = 0
    for record in records:
        game = corpus.regenerate(record)
        if game is not None and solve(game) is not None:
            answered += 1
    assert answered == 0


def test_the_solve_releases_the_gil(records):
    """A multi-second solve must not stop the rest of the process.

    Holding the GIL through the solve freezes a threaded advisor host and
    serialises any thread-based solver pool -- the pool would run one position
    at a time while looking parallel.
    """

    import threading
    import time

    # Deliberately NOT a corpus position: those now solve in microseconds, so
    # the thread would see no contention either way and the test would pass
    # while proving nothing. This position is past the reach table's limit, so
    # the solver is guaranteed to spend the whole budget and then give up.
    game = _deep_position(cards=13)
    from .rust_bridge import rust_game_from_state

    rust_game = rust_game_from_state(game)
    ticks = 0
    stop = threading.Event()

    def count_ticks():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            time.sleep(0.001)

    ticker = threading.Thread(target=count_ticks, daemon=True)
    ticker.start()
    # A budget long enough that a GIL-holding solve would starve the ticker.
    started = time.perf_counter()
    rust_game.solve_endgame(2_000_000_000, 1.0, "exact")
    elapsed = time.perf_counter() - started
    stop.set()
    ticker.join(timeout=2.0)

    # If the solve returned early the test proves nothing, so say so instead.
    assert elapsed > 0.5, f"solve finished in {elapsed:.3f}s; not a real test"

    # Without the release this sits at ~0; with it the ticker runs throughout.
    assert ticks > 50, f"other Python threads only advanced {ticks} times"


def test_journaled_undo_restores_state_exactly(records):
    """The journal's whole risk is a mutation nobody recorded.

    A missed one leaves the search on a quietly wrong state -- no crash, no
    wrong answer until much later, and nothing in the solve output to show it.
    So the journaled path is compared against the snapshot path it replaces:
    full-state equality after undo, over every legal action to depth 3, walking
    the same tree the solver walks. Positions the journal declines still run
    through the snapshot path here, so the audit covers the real mix.
    """

    from .rust_bridge import rust_game_from_state

    checked = 0
    for record in records[:20]:
        game = corpus.regenerate(record)
        if game is None:
            continue
        problem = rust_game_from_state(game).journal_undo_audit(3)
        assert problem is None, f"seed={record['seed']} ply={record['ply']}: {problem}"
        checked += 1
    # Deep positions too: they reach wonders, revives and token crossings that a
    # nearly-finished board no longer offers.
    for cards in (9, 11, 13):
        game = _deep_position(cards=cards)
        problem = rust_game_from_state(game).journal_undo_audit(3)
        assert problem is None, f"{cards} cards: {problem}"
        checked += 1
    assert checked > 10


@pytest.mark.parametrize("pruning", ["none", "star1", "star2"])
def test_every_chance_pruning_setting_returns_the_same_values(records, pruning):
    """Pruning may change the node count and nothing else.

    Both star settings failed this on their first run, and for a reason worth
    keeping: the derived window clamps to the full value range, and a child that
    comes back at exactly -1 or +1 through such a window is reporting its true
    value, not failing against a bound. Decided endgames are full of exact -1s
    and +1s, so the root published bounds as values.
    """

    report = corpus.check(corpus.rust_solver(chance_pruning=pruning))
    assert report.problems == [], str(report)
    assert report.checked == len(records), str(report)


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
