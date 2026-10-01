"""The Rust engine and macro vocabulary are equivalent to Python's —
``RUST_PORT_PLAN.md`` M1 and M2.

⚠ **This module is a *sample* of the gate, not the gate.** The gate is 8,000
games and takes about an hour; a test suite that took half an hour would
stop being run. The gate is::

    python -m games.welcome_to.rust_equiv --games 8000

and what runs here is the same code over a few dozen games, plus every
constructed position — those are cheap and they cover the rules that played
games do not reach.

⚠ **Python is the oracle** (§3). A failure here means the *Rust* engine is
wrong, unless the diff shows otherwise; ``game.py`` is what the BGA differential
harness validates.

The whole module skips when the crate is not built, so a checkout without a
Rust toolchain still runs the rest of the suite. ``maturin develop --release``
in ``games/welcome_to/welcome_to_rust`` is what un-skips it.
"""

from __future__ import annotations

import copy

import pytest

from games.welcome_to import macro_codec as mc
from games.welcome_to import rust_equiv as eq
from games.welcome_to import snapshot as sn
from games.welcome_to import tables
from games.welcome_to.constants import box_coords
from games.welcome_to.game import GameConfig, GameState
from games.welcome_to.portable_rng import PortableRng

wr = pytest.importorskip(
    "welcome_to_rust",
    reason="the Rust engine is not built; run `maturin develop --release` in "
    "games/welcome_to/welcome_to_rust",
)


# ──────────────────────────────────────────────────────────────────────────
# M0 contracts — checked first, because M1 means nothing if they are broken
# ──────────────────────────────────────────────────────────────────────────
def test_the_static_tables_are_the_same_tables():
    """M0-D. A silent table divergence produces a legal-looking game that is a
    *different game*, and it would pass every per-action gate below."""
    assert wr.table_signature() == tables.table_signature()


def test_the_snapshot_schemas_are_the_same_version():
    """M0-C. Two differently-shaped dictionaries can be compared key by key and
    agree; the version is what stops that."""
    assert wr.snapshot_version() == sn.SNAPSHOT_VERSION


def test_the_action_space_is_the_same_size():
    from games.welcome_to import action_codec as codec

    assert wr.NUM_ACTIONS == codec.NUM_ACTIONS


@pytest.mark.parametrize("seed", [0, 1, 2**63 - 1, 2**64 - 1])
def test_the_portable_rng_streams_are_identical(seed):
    """M0-B. "Same seed" only means something if the two generators agree."""
    rng = PortableRng(seed)
    assert [rng.next_u64() for _ in range(16)] == wr.portable_rng_stream(seed, 16)


def test_the_portable_shuffle_permutes_identically():
    """The deck shuffle *is* the deal, so a shuffle that differed by one swap
    would give two different games from the same seed."""
    for seed in (0, 7, 123456789):
        for n in (2, 5, 81, 82):
            seq = list(range(n))
            rng = PortableRng(seed)
            rng.shuffle(seq)
            rust_seq, rust_state = wr.portable_rng_shuffle(seed, n)
            assert seq == rust_seq
            assert rng.state == rust_state


# ──────────────────────────────────────────────────────────────────────────
# M0-A — the supported configuration matrix, refused loudly
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"players": 2, "expert": True}, "expert"),
        ({"players": 1}, "one-seat"),
        ({"players": 5}, "5 seats"),
    ],
)
def test_unsupported_configurations_are_refused_by_name(kwargs, expected):
    """A silently-ignored flag is how a Rust self-play run stops being
    equivalent without anybody noticing, so the refusal names the reason."""
    with pytest.raises(ValueError, match=expected):
        wr.RustGameState(1, **kwargs)


# ──────────────────────────────────────────────────────────────────────────
# M1 — lockstep equivalence
# ──────────────────────────────────────────────────────────────────────────
def test_a_new_game_deals_the_same_cards_and_the_same_plans():
    """Before a single action: the same shuffle, the same three plans, the same
    generator state.  If this fails, nothing below is meaningful."""
    config = GameConfig(players=2, advanced=True, solo_rules=False)
    py = GameState.new(seed=5, config=config)
    rs = wr.RustGameState(5, players=2, advanced=True, solo_rules=False)
    assert sn.diff(sn.to_snapshot(py), rs.snapshot()) == []
    assert list(py.plan_ids) == rs.plan_ids
    assert py.rng.state == rs.rng_state


@pytest.mark.parametrize("config", eq.GATE_CONFIGS, ids=lambda c: f"{c.players}p-adv{int(c.advanced)}")
@pytest.mark.parametrize("driver", eq.DRIVERS)
def test_a_game_agrees_after_every_action(config, driver):
    """One game per (configuration, driver) — 18 games, the shape of the gate.

    Compares the complete M0-C snapshot after every action, the raw
    ``legal_actions`` order before it, and the boundary triple plus
    redeterminization once per turn. Information-set accessors are also checked
    after the first seat hand-off, where live and public observations differ.
    """
    report = eq.check_game(seed=41, config=config, driver=driver)
    assert report.steps > 20, "a game this short is not evidence of anything"


@pytest.mark.parametrize("case", list(eq.constructed_cases()), ids=lambda c: c[0])
def test_a_constructed_position_agrees(case):
    """The states play does not reach: two of ``isEndOfGame``'s three clauses,
    a queued reshuffle, an exact-empty deck, and both at once.

    Measured over 60 greedy games: 56 ended on the third permit refusal, four
    filled a sheet, none completed three plans.  Rare is exactly where a rules
    divergence survives a gate, so these are handed to Rust through the
    snapshot rather than played into.
    """
    name, state, expect = case
    eq.check_constructed(name, state, expect)


def test_a_snapshot_survives_the_round_trip_into_rust_and_back():
    """M0-C in both directions, which is what the constructed positions ride
    on: Python state -> snapshot -> Rust state -> snapshot."""
    config = GameConfig(players=3, advanced=True, solo_rules=False)
    py = eq._mid_game(17, config, turn=8)
    snapshot = sn.to_snapshot(py)
    rs = wr.RustGameState.from_snapshot(snapshot)
    assert sn.diff(snapshot, rs.snapshot()) == []
    assert py.legal_actions() == rs.legal_actions()
    assert list(py.scores()) == list(rs.scores())


@pytest.mark.parametrize("seed", [0, 12345])
def test_redeterminize_permutes_the_undrawn_deck_identically(seed):
    """The primitive MCTS calls at every root, and the one place a port can
    cheat without any gate noticing: it must permute *only* the undrawn deck,
    identically, and leave the caller's generator in the same place.

    The determinized copy also gets a fresh generator of its own, derived from
    the caller's — two determinizations that shared RNG state would apply the
    same permutation at a mid-rollout reform, correlating simulations that are
    meant to be independent.
    """
    config = GameConfig(players=2, advanced=True, solo_rules=False)
    py = eq._mid_game(23, config, turn=6)
    rs = wr.RustGameState.from_snapshot(sn.to_snapshot(py))

    search_rng = PortableRng(seed)
    rust_state = seed
    for _ in range(3):
        py_next = py.redeterminize(search_rng)
        rs_next, rust_state = rs.redeterminize(rust_state)
        assert sn.diff(sn.to_snapshot(py_next), rs_next.snapshot()) == []
        assert search_rng.state == rust_state
        # the composition is untouched; only the hidden order moved
        assert sorted(py_next.deck) == sorted(py.deck)
        assert py_next.deck[: py_next.deck_pos] == py.deck[: py.deck_pos]


def test_a_cpython_snapshot_is_refused_rather_than_silently_reseeded():
    """A ``random.Random`` state has no Rust counterpart.  Substituting a fresh
    generator would make the hand-off *look* successful while the two engines
    drew different cards from there on."""
    config = GameConfig(players=2, advanced=True, solo_rules=False)
    py = GameState.new(seed=3, config=config, rng_kind="cpython")
    with pytest.raises(ValueError, match="cpython"):
        wr.RustGameState.from_snapshot(sn.to_snapshot(py))


def test_an_unknown_snapshot_rng_is_refused_by_both_readers():
    """A typo must not silently become a fresh Mersenne Twister in Python."""
    config = GameConfig(players=2, advanced=True, solo_rules=False)
    raw = sn.to_snapshot(GameState.new(seed=3, config=config))
    raw["rng"]["kind"] = "splitmix65"
    with pytest.raises(ValueError, match="rng kind"):
        sn.from_snapshot(raw)
    with pytest.raises(ValueError, match="splitmix65"):
        wr.RustGameState.from_snapshot(raw)

    impossible_cpython = copy.deepcopy(raw)
    impossible_cpython["rng"] = {"kind": "cpython", "state": 17}
    with pytest.raises(ValueError, match="cannot carry"):
        sn.from_snapshot(impossible_cpython)


def test_the_played_gate_really_checks_redeterminize_and_a_midturn_information_set(
    monkeypatch,
):
    """Regression for two claims that used to exist only in the gate's prose.

    Checking accessors only at a turn boundary is vacuous for sheet visibility:
    live and public sheets are equal there. Redeterminize was not wired into the
    played-game path at all.
    """
    redetermined_turns: list[int] = []
    accessor_actors: list[int] = []
    accessor_asymmetries: list[bool] = []
    original_redeterminize = eq._check_redeterminize
    original_accessors = eq._check_accessors

    def record_redeterminize(py, rs, seed, config, step):
        redetermined_turns.append(py.turn)
        return original_redeterminize(py, rs, seed, config, step)

    def record_accessors(py, rs, where):
        accessor_actors.append(py.actor)
        accessor_asymmetries.append(eq._has_private_midturn_state(py))
        return original_accessors(py, rs, where)

    monkeypatch.setattr(eq, "_check_redeterminize", record_redeterminize)
    monkeypatch.setattr(eq, "_check_accessors", record_accessors)
    config = GameConfig(players=2, advanced=True, solo_rules=False)
    eq.check_game(seed=29, config=config, driver="no-refusal", check_macros=False)

    assert len(set(redetermined_turns)) > 1
    assert any(actor > 0 for actor in accessor_actors)
    assert any(accessor_asymmetries)


def test_the_macro_gate_sampling_interval_must_be_positive():
    config = GameConfig(players=2, advanced=True, solo_rules=False)
    with pytest.raises(ValueError, match="at least 1"):
        eq.check_game(seed=1, config=config, macro_apply_every=0)


def test_the_m1_gate_can_skip_m2_without_skipping_engine_checks(monkeypatch):
    """The corrected M1 rerun need not pay M2's sampled macro-apply cost."""
    macro_calls = 0
    redeterminize_calls = 0
    original_redeterminize = eq._check_redeterminize

    def reject_macro_check(*args, **kwargs):
        nonlocal macro_calls
        macro_calls += 1

    def record_redeterminize(*args, **kwargs):
        nonlocal redeterminize_calls
        redeterminize_calls += 1
        return original_redeterminize(*args, **kwargs)

    monkeypatch.setattr(eq, "_check_macros", reject_macro_check)
    monkeypatch.setattr(eq, "_check_redeterminize", record_redeterminize)
    config = GameConfig(players=2, advanced=False, solo_rules=False)
    reports = list(
        eq.gate(
            1,
            configs=(config,),
            drivers=("random",),
            check_macros=False,
        )
    )
    assert reports and macro_calls == 0
    assert redeterminize_calls > 1


# ──────────────────────────────────────────────────────────────────────────
# M2 — the macro vocabulary
# ──────────────────────────────────────────────────────────────────────────
def test_the_macro_layout_is_the_same_684():
    """The 684 indices are an ABI: a checkpoint's policy head is indexed by
    them, so a shifted section puts two decisions on one logit."""
    assert wr.NUM_MACRO_ACTIONS == mc.NUM_MACRO_ACTIONS == 684
    assert list(wr.PRIMITIVE_ACTIONS) == list(mc.PRIMITIVE_ACTIONS)
    assert len(wr.PRIMITIVE_ACTIONS) == 184


def test_every_macro_index_decodes_to_the_same_primitives():
    """All 684 of them, not a sample: this is a static table and checking it
    whole costs milliseconds."""
    for index in range(mc.NUM_MACRO_ACTIONS):
        assert list(mc.primitives_for(index)) == list(wr.macro_primitives(index)), index


def test_the_macro_index_arithmetic_agrees():
    for slot in range(mc.NUM_MACRO_SLOTS):
        assert mc.macro_refuse(slot) == wr.macro_refuse(slot)
        for delta in range(mc.NUM_TEMP_DELTAS):
            for box in range(33):
                x, y = box_coords(box)
                index = mc.macro_write(slot, delta, x, y)
                assert index == wr.macro_write(slot, delta, x, y)
                assert mc.decode_macro_write(index) == tuple(wr.decode_macro_write(index))


def test_a_subsumed_primitive_has_no_standalone_macro_index():
    """A bare ``WRITE`` has no macro meaning without the slot that preceded it;
    inventing one would put two different decisions on one logit."""
    from games.welcome_to import action_codec as codec

    for subsumed in (codec.A_CHOOSE_STACK, codec.A_WRITE, codec.A_PERMIT_REFUSAL,
                     codec.A_ROUNDABOUT_OPEN):
        assert wr.macro_from_primitive(subsumed) is None
        with pytest.raises(ValueError):
            mc.from_primitive(subsumed)
    assert wr.macro_from_primitive(codec.A_PASS_PLAN) == mc.from_primitive(codec.A_PASS_PLAN)


@pytest.mark.parametrize("config", eq.GATE_CONFIGS, ids=lambda c: f"{c.players}p-adv{int(c.advanced)}")
def test_the_macro_vocabulary_agrees_over_a_whole_game(config):
    """``legal_macros`` in order at every root, ``search_legal_macros`` at both
    settings of ``prune_roundabout_pass``, and every compound macro offered at
    each turn-opening root applied end to end. Other macros are one primitive
    and ride the M1 action-parity check.

    ``macro_apply_every=1`` here — the sample is small enough to afford the
    claim in full, which is what makes the gate's sampling a speed choice
    rather than the only thing that was ever checked.
    """
    eq.check_game(seed=77, config=config, driver="no-refusal", macro_apply_every=1)


def test_write_number_is_inside_a_macro_in_both_engines():
    """The macro layer never decides at ``WRITE_NUMBER``; an engine that
    answered there would be inventing a decision point, and a search built on it
    would ask the network a question the vocabulary has no logit for."""
    config = GameConfig(players=2, advanced=True, solo_rules=False)
    py = GameState.new(seed=4, config=config)
    rs = wr.RustGameState(4, players=2, advanced=True, solo_rules=False)
    slot = next(a for a in py.legal_actions() if a < 6)
    py.apply(slot)
    rs.apply(slot)
    assert py.phase.name == "WRITE_NUMBER"
    assert not mc.is_macro_root(py) and not rs.is_macro_root
    with pytest.raises(ValueError):
        mc.legal_macros(py)
    with pytest.raises(ValueError):
        rs.legal_macros()


def test_the_expert_configuration_has_no_macro_representation_in_either_engine():
    """Three choice slots is the vocabulary; expert's six ordered pairs have no
    macro form and are refused rather than silently truncated.  Rust refuses the
    configuration outright (M0-A), which is the same answer one step earlier."""
    config = GameConfig(players=2, advanced=False, expert=True, solo_rules=False)
    py = GameState.new(seed=2, config=config)
    with pytest.raises(ValueError, match="standard mode only"):
        mc.legal_macros(py)
    with pytest.raises(ValueError, match="expert"):
        wr.RustGameState(2, players=2, expert=True)


def test_the_harness_can_actually_see_a_divergence():
    """A negative control.  A gate that cannot fail is not a gate — and this one
    compares two dictionaries, which is exactly the shape of check that quietly
    stops comparing anything."""
    config = GameConfig(players=2, advanced=True, solo_rules=False)
    py = GameState.new(seed=1, config=config)
    rs = wr.RustGameState(2, players=2, advanced=True, solo_rules=False)
    assert sn.diff(sn.to_snapshot(py), rs.snapshot()) != []

    with pytest.raises(eq.Divergence):
        eq._compare(py, rs, seed=1, config=config, step=0, action=None)


def test_an_illegal_action_raises_the_same_class_on_both_sides():
    config = GameConfig(players=2, advanced=True, solo_rules=False)
    py = GameState.new(seed=9, config=config)
    rs = wr.RustGameState(9, players=2, advanced=True, solo_rules=False)
    illegal = next(a for a in range(wr.NUM_ACTIONS) if a not in py.legal_actions())
    with pytest.raises(ValueError):
        py.apply(illegal)
    with pytest.raises(ValueError):
        rs.apply(illegal)


def test_a_rejected_boundary_outcome_leaves_the_state_untouched():
    """The transactional contract: "raises, and also destroys the state you
    called it on" is not a contract worth having, in either language."""
    config = GameConfig(players=2, advanced=True, solo_rules=False)
    py = eq._mid_game(11, config, turn=4)
    rs = wr.RustGameState.from_snapshot(sn.to_snapshot(py))
    assert py.prepare_turn_boundary() and rs.prepare_turn_boundary()

    before = rs.snapshot()
    outcome = py.sample_boundary_outcome(PortableRng(2))
    too_many = list(outcome.draws) + [outcome.draws[0]]
    with pytest.raises(ValueError):
        rs.apply_boundary_outcome(too_many)
    assert sn.diff(before, rs.snapshot()) == []
