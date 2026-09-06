"""W3 as an encoder input: the delta is only control, and applicability is right.

Two things a reviewer of a schema change actually needs. First, that the encoding
changed in *exactly* the way claimed -- regenerating a golden digest is how an
unintended change hides, so stripping the appended control columns must reproduce
the pre-W3 digests bit for bit. Second, that `control_valid` is 0 on the right
states and 1 on the right states, decided per phase rather than asserted.

Agreement between Python and Rust is checked in `test_control_key_parity.py`.
Agreement alone cannot establish that both made the *right* decision, which is
what the expected-value cases below are for.
"""

from __future__ import annotations

import random

import pytest

from .codec import legal_actions
from .control_table import control_key_from_observation
from .reveal_risk import REVEAL_FEATURES
from .encoder import (
    CONTROL_FEATURES,
    GLOBAL_FEATURES,
    TABLEAU_FEATURES,
    TokenType,
    encode,
)
from .engine import apply_action
from .game import Phase, new_game
from .test_encoder import _digest, _give_wonder, _playing_game

_VALID = GLOBAL_FEATURES.index("control_valid")
_FIRST = TABLEAU_FEATURES.index(CONTROL_FEATURES[0])


def _strip_control(encoding):
    """The encoding as it was before W3: control columns removed.

    Both additions are APPENDED, so removing them must recover the old vectors
    exactly. If it does not, the schema change did more than it claimed.
    """

    import dataclasses

    tokens = []
    for token in encoding.tokens:
        features = token.features
        if token.type is TokenType.GLOBAL:
            features = features[:_VALID] + features[_VALID + 1:]
        elif token.type is TokenType.TABLEAU:
            # Strip BOTH appended blocks: control, then the reveal-risk
            # channels that follow it. The claim under test is "the pre-W3
            # digest survives removing everything W3 added", so a later
            # appended block has to come off too.
            features = (
                features[:_FIRST]
                + features[_FIRST + len(CONTROL_FEATURES):-len(REVEAL_FEATURES)]
            )
        tokens.append(dataclasses.replace(token, features=features))
    return dataclasses.replace(encoding, tokens=tuple(tokens))


# -- the delta is only control ---------------------------------------------


@pytest.mark.parametrize(
    "build, expected",
    [
        (
            lambda: _playing_game(30).observation(0),
            "22b9b0a8b0381f3de284b622a59ff8ae2626d926acb7c800bc541e79865fbe66",
        ),
    ],
)
def test_stripping_control_reproduces_the_pre_w3_digest(build, expected):
    """Executable evidence that the schema change is exactly what it says.

    These are the digests this suite pinned before W3. They are not re-derived
    from the current encoder -- that would be circular -- they are the literals
    the pre-W3 tests asserted.
    """

    assert _digest(_strip_control(encode(build()))) == expected


def test_stripping_control_reproduces_the_pre_w3_draft_digest():
    draft = new_game(9)
    apply_action(draft, legal_actions(draft)[0])
    assert (
        _digest(_strip_control(encode(draft.observation(0))))
        == "b48a15fd8f87d5c93c98a345aaa6c8fc2114bb4444b5b1d8fcda92c7699ac0d9"
    )


# -- applicability, per phase, with expected values -------------------------


def _valid(obs) -> float:
    global_token = next(t for t in encode(obs).tokens if t.type is TokenType.GLOBAL)
    return global_token.features[_VALID]


def test_wonder_draft_is_masked():
    """The draft emits no tableau tokens at all -- 0 against 20 present cards --
    and the drafter is a 50/50 coin flip to be the next tableau mover."""

    draft = new_game(9)
    assert draft.phase.value == "wonder_draft"
    obs = draft.observation(draft.active_player)
    assert _valid(obs) == 0.0
    assert control_key_from_observation(obs) is None


def test_clean_play_age_is_valid():
    game = _playing_game(30)
    assert game.phase.value == "play_age" and game.pending_choice is None
    assert control_key_from_observation(
        game.observation(game.active_player)
    ) is not None
    assert _valid(game.observation(game.active_player)) == 1.0


def test_pending_choice_is_masked():
    """`_finish_turn` defers `pending_extra_turn`, so the decision-maker is not
    the next tableau mover; taking Theology would also restate the tempo.

    The state is found by playing rather than constructed: a hand-built one is a
    state the engine may not consider legal, and the point is what real play
    produces."""

    from .loop_adapter import SevenWondersDuelLoopAdapter

    adapter = SevenWondersDuelLoopAdapter()
    rng = random.Random(11)
    seen = 0
    for seed in range(40):
        game = adapter.new_game(seed=3000 + seed)
        guard = 0
        while not adapter.terminal(game) and guard < 300:
            guard += 1
            legal = adapter.legal_actions(game)
            if not legal:
                break
            if game.pending_choice is not None and game.phase.value == "play_age":
                assert _valid(game.observation(game.pending_choice.player)) == 0.0
                seen += 1
                if seen >= 5:
                    return
            game = adapter.step(game, rng.choice(legal))
    assert seen, "no pending-choice PLAY_AGE state found to check"


def test_an_illegal_tempo_is_masked_not_an_error():
    """A constructed state with a fifth Wonder gives nine unbuilt, which no legal
    game reaches. That is an inapplicable position, not a table gap."""

    game = _playing_game(30)
    _give_wonder(game, game.active_player, "The Mausoleum")
    obs = game.observation(game.active_player)
    assert control_key_from_observation(obs) is None
    assert _valid(obs) == 0.0


def test_masked_positions_zero_every_control_channel():
    """A masked position must not leave a stale or partial control vector: zeros
    with `control_valid` 0 is the only reading that is not a claim."""

    draft = new_game(9)
    encoding = encode(draft.observation(draft.active_player))
    for token in encoding.tokens:
        if token.type is TokenType.TABLEAU:
            window = token.features[_FIRST:_FIRST + len(CONTROL_FEATURES)]
            assert all(value == 0.0 for value in window)


def test_valid_positions_appear_and_are_the_common_case():
    """Guards against a regression that masks everything: the features would
    then be trivially 'consistent' and entirely useless."""

    from .loop_adapter import SevenWondersDuelLoopAdapter

    adapter = SevenWondersDuelLoopAdapter()
    rng = random.Random(4)
    game = adapter.new_game(seed=808)
    valid = total = 0
    guard = 0
    while not adapter.terminal(game) and guard < 200:
        guard += 1
        legal = adapter.legal_actions(game)
        if not legal:
            break
        actor = (game.pending_choice.player if game.pending_choice is not None
                 else game.active_player)
        total += 1
        valid += int(_valid(game.observation(actor)) == 1.0)
        game = adapter.step(game, rng.choice(legal))
    assert total > 50
    assert 0.6 < valid / total < 0.95, f"valid fraction {valid / total:.2f}"


# -- input-off mode: the baseline arm ---------------------------------------


def test_input_off_zeroes_every_control_channel():
    """The baseline arm must show the network nothing.

    Not "discouraged from using control" -- unable to. With the inputs pinned to
    zero the projection's control columns receive exactly zero gradient, so they
    never move from their zero initialisation.
    """

    from .encoder import control_features_enabled, set_control_features

    game = _playing_game(30)
    obs = game.observation(game.active_player)
    assert _valid(obs) == 1.0, "this position is control-valid with the arm on"

    set_control_features(False)
    try:
        assert control_features_enabled() is False
        encoding = encode(obs)
        global_token = next(
            t for t in encoding.tokens if t.type is TokenType.GLOBAL
        )
        assert global_token.features[_VALID] == 0.0
        for token in encoding.tokens:
            if token.type is TokenType.TABLEAU:
                window = token.features[_FIRST:_FIRST + len(CONTROL_FEATURES)]
                assert all(value == 0.0 for value in window)
    finally:
        set_control_features(True)
    assert _valid(obs) == 1.0, "the arm must be restorable"


def test_input_off_changes_only_the_control_columns():
    """Off-mode is the on-mode encoding with those columns zeroed, and nothing
    else. If any other feature moved, the arms would differ by more than the
    thing under test."""

    from .encoder import set_control_features

    game = _playing_game(30)
    obs = game.observation(game.active_player)
    on = _strip_control(encode(obs))
    set_control_features(False)
    try:
        off = _strip_control(encode(obs))
    finally:
        set_control_features(True)
    assert _digest(on) == _digest(off)


def test_both_languages_agree_in_off_mode():
    """Setting one language only would be worse than either arm: the replay and
    self-play paths would disagree about what the model is shown."""

    import numpy as np

    from .buffer import GameRecorder
    from .codec import legal_action_indices
    from .control_table import ensure_rust_table
    from .dataset import derive_records_rust, examples_from_record
    from .encoder import set_control_features

    if not ensure_rust_table():
        pytest.skip("seven_wonders_rust not available")

    recorder = GameRecorder(505, agents={"p0": "t", "p1": "t"})
    rng = random.Random(5051)
    while recorder.game.phase.value != "complete":
        choice = rng.choice(legal_action_indices(recorder.game))
        recorder.play(choice, policy_target={choice: 1.0})
    record = recorder.finish()

    set_control_features(False)
    try:
        python_rows = examples_from_record(record)
        rust_rows = derive_records_rust([record])[0][0]
        assert len(python_rows) == len(rust_rows)
        for py, rs in zip(python_rows, rust_rows):
            a = np.asarray(py.features, dtype=np.float64)
            b = np.asarray(rs.features, dtype=np.float64)
            assert a.shape == b.shape
            assert np.abs(a - b).max() == 0.0
        # ...and Rust really is off, not merely equal by luck.
        import seven_wonders_rust

        assert seven_wonders_rust.control_features_enabled() is False
    finally:
        set_control_features(True)


# -- reveal risk, both languages --------------------------------------------


def _reveal_columns():
    """Indices of the reveal channels inside a tableau token's feature row."""

    from .encoder import TABLEAU_FEATURES
    from .reveal_risk import REVEAL_FEATURES

    return [TABLEAU_FEATURES.index(name) for name in REVEAL_FEATURES]


def test_both_languages_agree_with_reveal_on():
    """The gate that lets the reveal channels reach self-play.

    Off-mode agreement proves only that both languages emit zeros. Self-play
    encodes in Rust and the replay/A-B path encodes in Python, so what has to
    hold is agreement with the channels LIVE -- and on a whole game, because the
    counts are pure geometry that only some positions exercise.
    """

    import numpy as np

    from .buffer import GameRecorder
    from .codec import legal_action_indices
    from .control_table import ensure_rust_table
    from .dataset import TYPE_IDS, derive_records_rust, examples_from_record
    from .reveal_risk import set_reveal_features

    if not ensure_rust_table():
        pytest.skip("seven_wonders_rust not available")

    recorder = GameRecorder(707, agents={"p0": "t", "p1": "t"})
    rng = random.Random(7071)
    while recorder.game.phase.value != "complete":
        choice = rng.choice(legal_action_indices(recorder.game))
        recorder.play(choice, policy_target={choice: 1.0})
    record = recorder.finish()

    set_reveal_features(True)
    try:
        import seven_wonders_rust

        # Rust really is on, not agreeing by both being off.
        assert seven_wonders_rust.reveal_features_enabled() is True
        python_rows = examples_from_record(record)
        rust_rows = derive_records_rust([record])[0][0]
        assert len(python_rows) == len(rust_rows)
        nonzero = 0
        columns = _reveal_columns()
        for py, rs in zip(python_rows, rust_rows):
            a = np.asarray(py.features, dtype=np.float64)
            b = np.asarray(rs.features, dtype=np.float64)
            assert a.shape == b.shape
            assert np.abs(a - b).max() == 0.0
            tableau = np.asarray(py.type_ids) == TYPE_IDS[TokenType.TABLEAU]
            if tableau.any():
                nonzero += int(np.count_nonzero(a[tableau][:, columns]))
    finally:
        set_reveal_features(False)

    # A whole game must actually exercise the channels, or the comparison above
    # is agreement about zeros with extra steps.
    assert nonzero > 0, "no reveal channel was ever nonzero; the test proves nothing"


def _age_three_position(seed=11, rng_seed=1101):
    """Play randomly into Age III and return the game."""

    from .buffer import GameRecorder
    from .codec import legal_action_indices

    recorder = GameRecorder(seed, agents={"p0": "t", "p1": "t"})
    rng = random.Random(rng_seed)
    while recorder.game.phase.value != "complete":
        if recorder.game.age == 3 and recorder.game.phase is Phase.PLAY_AGE:
            return recorder.game
        choice = rng.choice(legal_action_indices(recorder.game))
        recorder.play(choice, policy_target={choice: 1.0})
    raise AssertionError("never reached Age III")


def test_reveal_risk_uses_the_uncovered_slots_own_back():
    """A Guild-backed slot cannot turn over an Age III card.

    Until 2026-09-07 the fractions came from every relevant back pooled and were
    scaled by the count, so uncovering a GUILD slot was priced with Age III
    cards it could never produce -- military risk on a pool (the Guilds) that
    carries no shields at all.
    """

    from .data import BackType, CARDS_BY_NAME, GUILD_CARDS
    from .encoder import _Derived
    from .pool import unseen_pool
    from .reveal_risk import decisive_fractions, newly_revealed_backs, set_reveal_features

    game = _age_three_position()
    # Put the opponent one shield from a military win, so the Age III pool is
    # genuinely dangerous and the test cannot pass vacuously.
    game.conflict_position = 8 if game.active_player == 0 else -8

    observation = game.observation(game.active_player)
    revealed = newly_revealed_backs(observation)
    uncovering = {slot: counts for slot, counts in revealed.items() if counts}
    assert uncovering, "position uncovers nothing; the test would prove nothing"

    # Force every uncovered slot to a Guild back, using guilds that are not
    # visible anywhere else so the unseen pool stays consistent.
    visible = {
        card.card_name for card in observation.tableau
        if card.present and card.card_name
    }
    for city in game.cities:
        visible.update(city.buildings)
    visible.update(game.discard_pile)
    spare = [g.name for g in GUILD_CARDS if g.name not in visible]
    hidden = [
        slot for slot, card in game.tableau.cards.items()
        if card.present and not card.revealed
    ]
    for slot in hidden:
        if not spare:
            break
        game.tableau.cards[slot].card_name = spare.pop()

    observation = game.observation(game.active_player)
    derived = _Derived(observation, unseen_pool(observation), game.active_player)
    seat = derived.actor

    pooled = decisive_fractions(derived, seat)
    guild_only = decisive_fractions(derived, seat, BackType.GUILD)

    # Non-vacuity: the pooled reading really is nonzero here, which is what the
    # old code would have charged the Guild slot with.
    assert pooled[1] > 0.0, "pool carries no military threat; nothing to get wrong"
    # And no guild card carries a shield, so the honest answer is zero.
    assert guild_only[1] == 0.0
    assert all(CARDS_BY_NAME[name].shields == 0 for name in derived.pool.cards[BackType.GUILD])

    set_reveal_features(True)
    try:
        encoding = encode(observation)
    finally:
        set_reveal_features(False)

    # Tableau tokens are emitted in sorted (row, x) order of the present slots,
    # so they can be matched back to slot ids.
    columns = _reveal_columns()
    order = sorted(card.slot_id for card in observation.tableau if card.present)
    tokens = [t for t in encoding.tokens if t.type is TokenType.TABLEAU]
    assert len(order) == len(tokens)

    revealed = newly_revealed_backs(observation)
    guild_only = [
        slot for slot, counts in revealed.items()
        if counts and set(counts) == {BackType.GUILD}
    ]
    assert guild_only, "no slot uncovers only Guild backs; the test proves nothing"

    for slot, token in zip(order, tokens):
        if slot not in guild_only:
            continue
        assert token.features[columns[0]] > 0.0  # it does uncover something
        assert token.features[columns[3]] == 0.0  # and no Guild card wins a war
        assert token.features[columns[4]] == 0.0
