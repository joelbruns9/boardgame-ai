"""The macro vocabulary: layout, end-to-end legality, and the S0 collapse."""
from __future__ import annotations

import random

import pytest

from games.welcome_to import action_codec as codec
from games.welcome_to import datagen
from games.welcome_to import macro_codec as mc
from games.welcome_to.bots import GreedyBot
from games.welcome_to.constants import ESTATE_ROW_BOXES, NUM_BOXES, TEMP_DELTAS
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


# ──────────────────────────────────────────────────────────────────────────
# The search's action set -- SEARCH_SPEC §5.1 dominance pruning
# ──────────────────────────────────────────────────────────────────────────
#: The three that are pruned unconditionally.  ``PASS_ROUNDABOUT`` is the
#: fourth; it is pruned too, but behind a switch (SEARCH_SPEC §5.1a), so both
#: settings get tested.
_PRUNED = {
    Phase.ACTION_PARK: codec.A_PASS_PARK,
    Phase.ACTION_POOL: codec.A_PASS_POOL,
    Phase.ACTION_ESTATE: codec.A_PASS_ESTATE,
}
_ALL_PRUNED = {**_PRUNED, Phase.ROUNDABOUT_PLACE: codec.A_PASS_ROUNDABOUT}


def _search_corpus(games: int = 6, players: int = 3):
    for seed in range(games):
        yield from _macro_roots(players, seed + 40)


@pytest.mark.parametrize("prune_roundabout", [False, True])
def test_a_dominated_pass_is_pruned_exactly_when_it_has_an_alternative(prune_roundabout):
    """Both halves matter.  ``estate_rows()`` can be empty -- nothing settles that
    phase away, unlike park and pool -- and then the pass is the whole node."""
    table = _ALL_PRUNED if prune_roundabout else _PRUNED
    seen_pruned = set()
    for state in _search_corpus():
        pruned = table.get(state.phase)
        if pruned is None:
            continue
        legal = mc.legal_macros(state)
        search = mc.search_legal_macros(state, prune_roundabout)
        macro = mc.from_primitive(pruned)
        assert macro in legal, "the engine stopped offering the pass"
        if len(legal) > 1:
            assert macro not in search, f"{codec.describe(pruned)} survived pruning"
            assert search == [m for m in legal if m != macro]
            seen_pruned.add(state.phase)
        else:
            assert search == [macro], "the only legal action was pruned away"
    assert seen_pruned == set(table), f"never exercised: {set(table) - seen_pruned}"


def test_the_roundabout_pass_is_pruned_by_default_and_restorable():
    """SEARCH_SPEC §5.1a.  Pruned like the other three -- and the one that can be
    put back, because it is the one that interacts with a bootstrap prior."""
    macro = mc.from_primitive(codec.A_PASS_ROUNDABOUT)
    seen = False
    for state in _search_corpus():
        if state.phase is not Phase.ROUNDABOUT_PLACE:
            continue
        seen = True
        assert macro in mc.legal_macros(state), "the engine stopped offering it"
        if len(mc.legal_macros(state)) > 1:
            assert macro not in mc.search_legal_macros(state)
        assert macro in mc.search_legal_macros(state, prune_roundabout_pass=False)
    assert seen, "no ROUNDABOUT_PLACE state in the corpus"


def test_the_roundabout_placement_node_is_wide_and_never_collapses():
    """Why pruning its pass narrows the node instead of removing it.

    ``ROUNDABOUT_PLACE`` offers ``available_locations(None)`` -- *every empty box
    on the sheet*, because a roundabout may go anywhere -- so the node is as wide
    as the sheet is empty.  Measured over 485 visits in 25 GreedyBot games it
    ranges across 1..33 placements and is a singleton exactly **once**.  The
    forced-node collapse of §12 step 2 therefore never removes it.
    """
    widths = [
        len(state.sheets[state.actor].available_locations(None))
        for state in _search_corpus()
        if state.phase is Phase.ROUNDABOUT_PLACE
    ]
    assert widths, "no ROUNDABOUT_PLACE state in the corpus"
    assert max(widths) > 20, "the node is not wide anywhere"
    assert sum(1 for w in widths if w == 1) / len(widths) < 0.05


def test_the_pass_survives_when_it_is_the_only_thing_left():
    """``ACTION_ESTATE`` with every value row full -- the pass is the whole node.

    ⚠ **Measured, this is the only one of the four that is reachable at all**,
    which corrects the brief's ``PASS_PARK`` example:

    - ``ACTION_PARK`` and ``ACTION_POOL`` are settled away by ``_settle()`` when
      their build is unavailable, so the phase is never *entered* passless;
    - ``ROUNDABOUT_PLACE`` offers ``available_locations(None)``, which is exactly
      ``has_free_box()`` -- the same condition that put the roundabout on offer
      -- so its placement list is never empty either;
    - ``ACTION_ESTATE`` has no such guard: ``estate_rows()`` runs out after
      ``sum(ESTATE_ROW_BOXES) == 18`` marks and the phase is still entered.

    So the pruning's ``len(macros) < 2`` guard is load-bearing for exactly one
    phase, and this is it.
    """
    state = _played(players=2, seed=11, turns=4)
    state.phase = Phase.ACTION_ESTATE
    state.sheets[state.actor].estate_marks = list(ESTATE_ROW_BOXES)
    assert state.legal_actions() == [codec.A_PASS_ESTATE]

    pass_estate = mc.from_primitive(codec.A_PASS_ESTATE)
    assert mc.legal_macros(state) == [pass_estate]
    assert mc.search_legal_macros(state) == [pass_estate], "pruning emptied the node"

    state.sheets[state.actor].estate_marks = [0] + list(ESTATE_ROW_BOXES[1:])
    assert pass_estate not in mc.search_legal_macros(state), "one row is an alternative"


def test_a_full_estate_row_is_already_masked_out_by_the_engine():
    """The size-1 row takes exactly one mark and then leaves the legal set.

    ``estate_rows()`` is ``estate_marks[i] < ESTATE_ROW_BOXES[i]``, and
    ``ESTATE_ROW_BOXES`` is ``(1, 2, 3, 4, 4, 4)`` -- so no search-side masking
    is needed for this, and none is added.  Emptying the node needs *five* of the
    six rows full, i.e. 14 of the 18 boxes; measured over 320 ``ACTION_ESTATE``
    visits the node never fell below **3** rows, mean 4.74 from turn 15 on.  That
    is why pruning ``PASS_ESTATE`` narrows 7 to <= 6 and never removes the node.
    """
    state = _played(players=2, seed=11, turns=4)
    sheet = state.sheets[state.actor]
    sheet.estate_marks = [0, 0, 0, 0, 0, 0]
    assert sheet.estate_rows() == [0, 1, 2, 3, 4, 5]
    sheet.estate_marks[0] = 1  # size-1 row: one box, now full
    assert sheet.estate_rows() == [1, 2, 3, 4, 5], "a full row stayed on offer"
    assert sum(ESTATE_ROW_BOXES) == 18


def test_the_pruning_table_is_the_four_dominated_passes():
    assert mc._DOMINATED_PASS == {
        phase: mc.from_primitive(primitive) for phase, primitive in _ALL_PRUNED.items()
    }


def test_bis_and_surveyor_keep_their_pass():
    """Not dominated: bis fills a box and takes a penalty, and a fence can
    destroy an ``EstatePlan``'s required sizes."""
    seen = set()
    for state in _search_corpus():
        if state.phase is Phase.ACTION_BIS:
            assert mc.from_primitive(codec.A_PASS_BIS) in mc.search_legal_macros(state)
            seen.add(state.phase)
        if state.phase is Phase.ACTION_SURVEYOR:
            assert mc.from_primitive(codec.A_PASS_SURVEYOR) in mc.search_legal_macros(
                state
            )
            seen.add(state.phase)
    assert seen == {Phase.ACTION_BIS, Phase.ACTION_SURVEYOR}


def test_the_engines_own_legal_set_is_untouched_by_the_pruning():
    """The trap this file exists to guard: ``datagen.replay`` builds its training
    legal mask from ``legal_mask``, and GreedyBot takes the dominated actions
    anyway -- 78 ``PASS_ESTATE`` and 1775 ``PASS_ROUNDABOUT`` in the reference
    corpus.  Pruning there would make those recorded labels illegal."""
    seen = set()
    for state in _search_corpus():
        pruned = _ALL_PRUNED.get(state.phase)
        if pruned is None:
            continue
        macro = mc.from_primitive(pruned)
        assert macro in mc.legal_macros(state)
        assert mc.legal_mask(state)[macro], "legal_mask lost the pass"
        assert set(mc.legal_macros(state)) == set(
            mc.from_primitive(a) for a in state.legal_actions()
        )
        seen.add(state.phase)
    assert seen == set(_ALL_PRUNED)


def test_the_search_mask_is_the_search_list():
    for state in _search_corpus(games=3):
        mask = mc.search_legal_mask(state)
        assert set(mask.nonzero()[0].tolist()) == set(mc.search_legal_macros(state))
        assert not (mask & ~mc.legal_mask(state)).any(), "pruning invented an action"
