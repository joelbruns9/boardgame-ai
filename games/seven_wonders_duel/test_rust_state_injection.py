"""Gate for `RustGame.from_state` — full mid-game state injection.

Why this file exists (ADVISOR_RUST_UNIFICATION.md §5, step 2). Every other
Python/Rust equivalence test reaches a position by replaying an action prefix
from a locked deal. The advisor cannot: it rebuilds a position from a public BGA
observation, with hidden information supplied by a determinizer and no action
history, so there is no seed and no prefix. Injection is the only way in — and a
wide serialization boundary whose entire value is exactness needs a test that
starts from an *injected* state, not a replayed one.

That distinction is not academic. The `pool.py`/`pool.rs` divergence that shipped
in 0aa0897 was invisible to every existing gate precisely because those gates
only ever exercise states Rust built itself.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from .codec import encode_action, legal_action_indices
from .data import PROGRESS_IDS, ScienceSymbol
from .engine import apply_action, legal_actions
from .game import PendingChoiceKind, Phase, VictoryType, new_game
from .rust_bridge import rust_game_from_state, rust_setup, rust_state

seven_wonders_rust = pytest.importorskip("seven_wonders_rust")

_TESTDATA = Path(__file__).parent / "testdata"


def _rust_replay(game):
    return seven_wonders_rust.RustGame(**rust_setup(game), library_draws=[])


def test_enum_declaration_order_matches_rust():
    """Enum fields cross the boundary as declaration indices, so the two
    languages must declare them in the same order. Reordering either side would
    otherwise silently remap — e.g. a scientific win becoming a military one."""
    assert [m.name for m in Phase] == [
        "WONDER_DRAFT",
        "PLAY_AGE",
        "CHOOSE_NEXT_START_PLAYER",
        "COMPLETE",
    ]
    assert [m.name for m in PendingChoiceKind] == [
        "DESTROY_OPPONENT_BROWN",
        "DESTROY_OPPONENT_GREY",
        "BUILD_FROM_DISCARD_FREE",
        "CHOOSE_UNUSED_PROGRESS",
        "CHOOSE_AVAILABLE_PROGRESS",
    ]
    assert [m.name for m in VictoryType] == [
        "MILITARY",
        "SCIENTIFIC",
        "CIVILIAN",
        "SHARED_CIVILIAN",
    ]
    assert [m.name for m in ScienceSymbol] == [
        "ARMILLARY_SPHERE",
        "WHEEL",
        "SUNDIAL",
        "MORTAR_AND_PESTLE",
        "SET_SQUARE",
        "QUILL_AND_INK",
        "LAW",
    ]


def test_injected_state_fingerprints_identically_to_replay():
    """THE gate. For every position reachable by replay, injecting the Python
    state must produce a byte-identical Rust state."""
    rng = random.Random(17)
    compared = 0
    seen_phases: set[str] = set()
    mismatches: list[tuple] = []

    for seed in range(40):
        game = new_game(seed, 0)
        rust = _rust_replay(game)
        for step in range(400):
            if game.phase is Phase.COMPLETE:
                break
            actions = legal_actions(game)
            if not actions:
                break
            action = rng.choice(actions)
            index = encode_action(game, action)
            apply_action(game, action)
            try:
                rust.apply_index(index)
            except BaseException:
                # Great Library draws must be pre-supplied to Rust; such lines
                # are covered by the replay-path tests, not this one.
                break
            injected = rust_game_from_state(game)
            compared += 1
            seen_phases.add(game.phase.name)
            if injected.fingerprint() != rust.fingerprint():
                mismatches.append((seed, step, game.phase.name))

    assert compared > 1500, f"corpus too small to mean anything: {compared}"
    # All four phases, or the corpus is not exercising the state machine.
    assert seen_phases == {
        "WONDER_DRAFT",
        "PLAY_AGE",
        "CHOOSE_NEXT_START_PLAYER",
        "COMPLETE",
    }, seen_phases
    assert not mismatches, f"{len(mismatches)} mismatches, first {mismatches[:3]}"


def test_taken_tableau_slots_keep_their_card_id():
    """Regression: zeroing the card id of an emptied slot mismatched 2,635 of
    3,204 injected positions. Both sides retain it after removal, and that
    retention is load-bearing -- it is how a card buried under a wonder still
    reads as visible to the unseen-card pool."""
    rng = random.Random(5)
    game = new_game(1, 0)
    rust = _rust_replay(game)
    for _ in range(60):
        if game.phase is Phase.COMPLETE:
            break
        actions = legal_actions(game)
        if not actions:
            break
        action = rng.choice(actions)
        index = encode_action(game, action)
        apply_action(game, action)
        try:
            rust.apply_index(index)
        except BaseException:
            pytest.skip("hit a Great Library before an emptied slot")
        taken = [
            c for c in game.tableau.cards.values() if not c.present and c.card_name
        ]
        if taken:
            payload = rust_state(game)
            ids = {slot[0] for slot in payload["tableau_slots"]}
            assert ids != {0}
            assert rust_game_from_state(game).fingerprint() == rust.fingerprint()
            return
    pytest.skip("no emptied slot reached")


# --- states Rust could not previously reach at all --------------------------


def _scraped_positions():
    """Determinized positions from real BGA captures. These have no seed and no
    prefix, so replay cannot construct them -- the whole point of injection."""
    from .advisor_adapter import SevenWondersAdvisor

    advisor = SevenWondersAdvisor()
    for name, keys in (
        ("bga_892846644_age3_reference", ("bga", "dom")),
        ("bga_892846644_greatlibrary", ("bga", "args", "dom", "log")),
    ):
        raw = json.loads((_TESTDATA / f"{name}.json").read_text(encoding="utf-8"))
        yield name, advisor.state_from_wire({k: raw[k] for k in keys}).game


@pytest.mark.parametrize("name", ["bga_892846644_age3_reference", "bga_892846644_greatlibrary"])
def test_scraped_position_injects_and_agrees_with_python(name):
    positions = dict((n, g) for n, g in _scraped_positions())
    game = positions[name]
    rust = rust_game_from_state(game)

    # Legality is the sharpest cheap check: it exercises costs, chains, wonders
    # and the pending choice all at once.
    assert list(rust.legal_action_indices()) == list(legal_action_indices(game))

    # And the unseen pool, which is where the pool.rs divergence lived.
    from .data import CARD_IDS, BackType

    py_pool = __import__(
        "games.seven_wonders_duel.pool", fromlist=["unseen_pool"]
    ).unseen_pool(game.observation(0))
    rs = rust.unseen_pool()
    order = [BackType.AGE_I, BackType.AGE_II, BackType.AGE_III, BackType.GUILD]
    for i, back in enumerate(order):
        assert sorted(rs[i]) == sorted(CARD_IDS[n] for n in py_pool.cards[back]), back
    assert sorted(rs[5]) == sorted(
        PROGRESS_IDS[n] for n in py_pool.offboard_progress
    )
