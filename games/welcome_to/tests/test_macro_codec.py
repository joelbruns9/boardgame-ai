"""The macro vocabulary: layout, end-to-end legality, and the S0 collapse."""
from __future__ import annotations

import random

import pytest

from games.welcome_to import action_codec as codec
from games.welcome_to import datagen
from games.welcome_to import macro_codec as mc
from games.welcome_to.bots import GreedyBot
from games.welcome_to.constants import NUM_BOXES, TEMP_DELTAS
from games.welcome_to.game import GameConfig, GameState, Phase


def _played(players: int = 3, seed: int = 4, turns: int = 6) -> GameState:
    bots = [GreedyBot(random.Random(seed * 100 + i)) for i in range(players)]
    state = GameState.new(seed=seed, config=GameConfig(players=players, advanced=True))
    while not state.is_terminal and state.turn < turns:
        state.apply(bots[state.actor].act(state))
    return state


def _macro_roots(players: int, seed: int):
    """Every state the macro layer decides at, over one whole game."""
    bots = [GreedyBot(random.Random(seed * 7 + i)) for i in range(players)]
    state = GameState.new(seed=seed, config=GameConfig(players=players, advanced=True))
    while not state.is_terminal:
        if state.phase is not Phase.WRITE_NUMBER:
            yield state
        state.apply(bots[state.actor].act(state))


# ──────────────────────────────────────────────────────────────────────────
# Layout -- frozen in ENCODER_V2_SPEC §10.6
# ──────────────────────────────────────────────────────────────────────────
def test_the_vocabulary_is_the_frozen_684():
    assert mc.NUM_MACRO_ACTIONS == 684
    assert mc.M_WRITE == 0
    assert mc.M_REFUSE == 495 == 3 * len(TEMP_DELTAS) * NUM_BOXES
    assert mc.M_DIRECT_REFUSE == 498
    assert mc.M_ROUNDABOUT_OPEN == 499
    assert mc.M_PRIMITIVE == 500
    assert len(mc.PRIMITIVE_ACTIONS) == 184


def test_the_primitives_are_the_codec_minus_exactly_what_the_macro_subsumes():
    subsumed = set(range(codec.A_CHOOSE_STACK, codec.A_CHOOSE_STACK + 6))
    subsumed |= set(range(codec.A_WRITE, codec.A_WRITE + 5 * NUM_BOXES))
    subsumed |= {codec.A_PERMIT_REFUSAL, codec.A_ROUNDABOUT_OPEN}
    assert len(subsumed) == 173
    assert set(mc.PRIMITIVE_ACTIONS) == set(range(codec.NUM_ACTIONS)) - subsumed
    # the roundabout *placement* stays a primitive: taking the roundabout is one
    # decision, siting it is another, and the turn returns to CHOOSE_CARDS after
    assert codec.roundabout_pos(0, 0) in mc.PRIMITIVE_ACTIONS


def test_macro_write_round_trips():
    seen = set()
    for slot in range(3):
        for delta in range(len(TEMP_DELTAS)):
            for box in range(NUM_BOXES):
                x, y = box // 12, box % 12
                if y >= 12:
                    continue
                index = mc.macro_write(slot, delta, *codec.box_coords(box))
                assert mc.decode_macro_write(index) == (
                    slot,
                    delta,
                    *codec.box_coords(box),
                )
                seen.add(index)
    assert seen == set(range(mc.M_WRITE, mc.M_REFUSE)), "the write block has holes"


def test_a_subsumed_primitive_has_no_standalone_index():
    """A bare WRITE has no macro meaning without the slot that preceded it."""
    for action in (codec.choose_stack(0), codec.write(0, 0, 0), codec.A_PERMIT_REFUSAL):
        with pytest.raises(ValueError):
            mc.from_primitive(action)


def test_every_primitive_index_round_trips():
    for action in mc.PRIMITIVE_ACTIONS:
        assert mc.to_primitive(mc.from_primitive(action)) == action


def test_a_macro_write_is_not_a_single_action():
    with pytest.raises(ValueError):
        mc.to_primitive(mc.macro_write(0, 0, 0, 0))


def test_expert_and_solo_are_refused_rather_than_truncated():
    """Six ordered pairs have no three-slot representation."""
    for config in (
        GameConfig(players=2, expert=True),
        GameConfig(players=1, solo_rules=True),
    ):
        state = GameState.new(seed=1, config=config)
        with pytest.raises(ValueError, match="standard mode"):
            mc.legal_macros(state)


def test_a_fourth_slot_has_no_index():
    with pytest.raises(ValueError):
        mc.macro_write(3, 0, 0, 0)
    with pytest.raises(ValueError):
        mc.macro_refuse(3)


# ──────────────────────────────────────────────────────────────────────────
# Legality -- enumerated end to end, never intersected
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("players", [2, 3, 4])
def test_every_legal_macro_actually_applies(players):
    """The contract: legal iff the *whole* primitive sequence is legal.

    An intersection of per-step masks would pass "slot s is legal" and "write 7
    in box 4 is legal" independently and admit the pair even when writing 7 is
    only reachable from a different slot.  `step_macro` raises on any illegal
    step, so applying all of them is the assertion.
    """
    total = 0
    for state in _macro_roots(players, seed=players * 11):
        macros = mc.legal_macros(state)
        assert macros, f"no legal macro at {state.phase.name}"
        assert len(set(macros)) == len(macros), "duplicate index"
        for macro in macros:
            mc.step_macro(state, macro)
            total += 1
    assert total > 500


def test_the_macro_mask_admits_nothing_the_engine_would_refuse():
    """The inverse of the test above: an illegal macro must not apply."""
    state = _played(players=2)
    while state.phase is Phase.WRITE_NUMBER:
        state.apply(state.legal_actions()[0])
    mask = mc.legal_mask(state)
    assert mask.shape == (mc.NUM_MACRO_ACTIONS,)

    refused = 0
    for macro in range(0, mc.M_PRIMITIVE, 7):  # sample the macro block
        if mask[macro]:
            continue
        with pytest.raises(Exception):
            mc.step_macro(state, macro)
        refused += 1
    assert refused > 10


def test_the_write_number_phase_is_never_a_decision():
    """Under this vocabulary the network is not asked there -- at all."""
    state = _played(players=2)
    while state.phase is not Phase.WRITE_NUMBER:
        state.apply(state.legal_actions()[0])
    assert not mc.is_macro_root(state)
    with pytest.raises(ValueError, match="inside a macro"):
        mc.legal_macros(state)


def test_taking_a_slot_to_refuse_is_a_different_action_from_refusing_outright():
    """§6.4: a slot can be playable via the temp agency while its *printed*
    number has nowhere to go, and `argWriteNumber` will not force the agency to
    be spent merely to have somewhere to write.  Which slot you burn is a real
    decision -- it is the effect you forgo."""
    found = False
    for players in (2, 3):
        for state in _macro_roots(players, seed=players * 3 + 1):
            if state.phase is not Phase.CHOOSE_CARDS:
                continue
            macros = mc.legal_macros(state)
            refusals = [m for m in macros if mc.M_REFUSE <= m < mc.M_DIRECT_REFUSE]
            if not refusals:
                continue
            found = True
            # the slot is playable -- otherwise it would be the direct refusal
            slot = refusals[0] - mc.M_REFUSE
            assert codec.choose_stack(slot) in state.legal_actions()
            assert mc.M_DIRECT_REFUSE not in macros
            assert mc.primitives_for(refusals[0]) == (
                codec.choose_stack(slot),
                codec.A_PERMIT_REFUSAL,
            )
            break
        if found:
            break
    assert found, "no macro refusal arose; the test never exercised its case"


def test_a_terminal_state_offers_nothing():
    bots = [GreedyBot(random.Random(i)) for i in range(2)]
    state = GameState.new(seed=2, config=GameConfig(players=2, advanced=True))
    while not state.is_terminal:
        state.apply(bots[state.actor].act(state))
    assert mc.legal_macros(state) == []


# ──────────────────────────────────────────────────────────────────────────
# The collapse -- how a primitive corpus becomes macro labels
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("players", [2, 3, 4])
def test_the_collapse_is_total_and_faithful(players):
    """Every primitive trajectory has exactly one macro reading.

    This is what lets the stored corpus stay primitive: the vocabulary can be
    changed without recapturing a single game.
    """
    def factory(rng: random.Random):
        bots = [GreedyBot(random.Random(rng.randrange(1 << 30))) for _ in range(players)]
        return lambda s: bots[s.actor].act(s)

    for trajectory in datagen.generate(
        3, factory, config=GameConfig(players=players, advanced=True), seed=5
    ):
        state = GameState.new(seed=trajectory.seed, config=trajectory.config)
        labels = []
        for visited, macro in mc.collapse(state, trajectory.actions):
            assert visited.phase is not Phase.WRITE_NUMBER
            assert macro in mc.legal_macros(visited), mc.describe(macro)
            labels.append(macro)
        assert state.is_terminal
        assert tuple(state.scores()) == trajectory.scores, "the collapse diverged"
        assert 0 < len(labels) < len(trajectory.actions), "nothing was collapsed"


def test_the_collapse_refuses_a_trajectory_that_ends_mid_macro():
    state = GameState.new(seed=1, config=GameConfig(players=2, advanced=True))
    slot = codec.decode_stack(state.legal_actions()[0])
    with pytest.raises(ValueError, match="unpaired"):
        list(mc.collapse(state, [codec.choose_stack(slot)]))


def test_replay_emits_one_sample_per_macro_and_labels_it_legally():
    def factory(rng: random.Random):
        bots = [GreedyBot(random.Random(rng.randrange(1 << 30))) for _ in range(3)]
        return lambda s: bots[s.actor].act(s)

    trajectory = datagen.generate(
        1, factory, config=GameConfig(players=3, advanced=True), seed=8
    )[0]
    samples = list(datagen.replay(trajectory))
    assert samples
    for sample in samples:
        assert sample.legal.shape == (mc.NUM_MACRO_ACTIONS,)
        assert sample.legal[sample.action], "a label was masked illegal"
    assert len(samples) < len(trajectory.actions)
