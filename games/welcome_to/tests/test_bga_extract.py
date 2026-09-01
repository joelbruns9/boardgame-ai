"""The BGA scrape, checked against the engine that produced it.

The mapper's job is to turn what a browser can see back into a
:class:`GameState`, and the only honest test of that is a round trip: play real
games, render each position the way BGA would send it, map it back, and demand
the two agree.  Hand-written fixtures would only prove the mapper agrees with
whatever the fixture author believed BGA does.

WHAT BGA SENDS, RECONSTRUCTED HERE
----------------------------------
``gamedatas.players[pid].scoreSheet`` is the sheet as it stood at the *start* of
the turn -- ``notif_updatePlayersData`` is the only thing that rewrites it and it
fires at ``stApplyTurn``.  So the synthetic payload is built from two sheets: the
turn-start snapshot (which is what BGA carries) and the delta the viewer has
marked since (which lives only in the DOM until the turn ends).  That split is
the mapper's central assumption, so the test has to reproduce it rather than
hand over a single finished sheet.

Positions are captured only when seat 0 is to act, which is not a convenience:
it is the situation a real capture is always in.  BGA is simultaneous and the
engine serialises the turn, so a capture taken while a *later* seat acts would
show the earlier seats' turn-start sheets and no way to know they had moved --
and the mapper reindexes the viewer to seat 0 precisely so that never arises.
"""

from __future__ import annotations

import random

import pytest

from games.welcome_to import action_codec as codec
from games.welcome_to import macro_codec as mc
from games.welcome_to.bga_extract import (
    PHASE_OF_STATE,
    StaleGamedata,
    UnsupportedBgaState,
    state_from_bga_payload,
    wire_from_bga_payload,
)
from games.welcome_to.constants import (
    CARD_TABLE,
    POOL_POSITIONS,
    ROUNDABOUT,
    STREET_SIZES,
)
from games.welcome_to.bots import GreedyBot
from games.welcome_to.game import GameConfig, GameState, Phase
from games.welcome_to.plans import PLANS
from games.welcome_to.sheet import Sheet

#: Engine phase -> the BGA state name a capture would carry there.
#:
#: ``writeNumber`` is listed explicitly because the mapping is not a bijection:
#: it maps *back* to ``CHOOSE_CARDS``, since the macro vocabulary folds the write
#: into the card choice and nothing has been marked yet at that state.
_STATE_OF_PHASE = {
    Phase.CHOOSE_CARDS: "chooseCards",
    Phase.ROUNDABOUT_PLACE: "buildRoundabout",
    Phase.WRITE_NUMBER: "writeNumber",
    Phase.ACTION_SURVEYOR: "actionSurveyor",
    Phase.ACTION_ESTATE: "actionEstate",
    Phase.ACTION_PARK: "actionPark",
    Phase.ACTION_POOL: "actionPool",
    Phase.ACTION_BIS: "actionBis",
    Phase.CHOOSE_PLAN: "choosePlan",
    Phase.VALIDATE_PLAN: "validatePlan",
    Phase.ASK_RESHUFFLE: "askReshuffle",
}
assert set(_STATE_OF_PHASE.values()) == set(PHASE_OF_STATE)

#: Any turn strictly before the current one; the mapper only ever compares
#: ``turn == current``, so the exact value of a past mark does not matter.
_PAST = 0


# ---------------------------------------------------------------------------
# Rendering a Sheet the way BGA would send it
# ---------------------------------------------------------------------------
def _houses(sheet: Sheet) -> list[dict]:
    out = []
    for x, size in enumerate(STREET_SIZES):
        for y in range(size):
            number = sheet.numbers[x][y]
            if number is None:
                continue
            out.append(
                {
                    "x": x,
                    "y": y,
                    "number": int(number),
                    "isBis": bool(sheet.is_bis[x][y]),
                    "turn": int(sheet.written_turn[x][y]),
                }
            )
    return out


def _scribbles(sheet: Sheet, turn: int) -> list[dict]:
    """Every mark on ``sheet``, as ``Scribbles::cast`` would report it.

    A :class:`Sheet` stores counters where BGA stores one row per mark, so the
    counters are expanded back out.  Which box a counter mark sits in is never
    read by the mapper (it counts them), but emitting plausible coordinates keeps
    the fixture honest about the shape of the real data.
    """

    out: list[dict] = []

    def add(kind: str, x, y=None):
        out.append({"type": kind, "x": x, "y": y, "turn": turn})

    for x, row in enumerate(sheet.fences):
        for j, drawn in enumerate(row):
            if drawn:
                add("estate-fence", x, j)
    for x, row in enumerate(sheet.top_fences):
        for y, marked in enumerate(row):
            if marked:
                add("top-fence", x, y)
    for x, count in enumerate(sheet.parks):
        for i in range(count):
            add("park", x, i)
    for x, count in enumerate(sheet.pools):
        # BGA draws the circle on the house, so the coordinates are a pool box in
        # that street.
        boxes = [p for p in POOL_POSITIONS if p[0] == x]
        for i in range(count):
            add("pool", x, boxes[i][1])
    for row, count in enumerate(sheet.estate_marks):
        for i in range(count):
            add("score-estate", row, i)
    for i in range(sheet.temps):
        add("score-temp", i)
    for i in range(sheet.bis_marks):
        add("score-bis", i)
    for i in range(sheet.permits):
        add("permit-refusal", i)
    for i in range(sheet.roundabouts):
        add("score-roundabout", i)
    return out


def _mark_key(mark: dict) -> tuple:
    return (mark["type"], mark["x"], mark["y"])


def _delta(before: Sheet, after: Sheet, turn: int) -> dict:
    """The marks the viewer has added since the turn began.

    Set difference rather than bookkeeping, because that is exactly what a DOM
    read gives: the marks carrying this turn's ``data-turn``.
    """
    old = {_mark_key(m) for m in _scribbles(before, _PAST)}
    new_scribbles = [
        dict(m, turn=turn)
        for m in _scribbles(after, turn)
        if _mark_key(m) not in old
    ]
    old_houses = {(h["x"], h["y"]) for h in _houses(before)}
    new_houses = [h for h in _houses(after) if (h["x"], h["y"]) not in old_houses]
    return {"houses": new_houses, "scribbles": new_scribbles}


def _stack_rows(state: GameState) -> list[list[dict]]:
    """``constructionCards``: ``[aside, top]`` per stack, ``getTopOf`` order."""
    rows = []
    for slot in range(3):
        aside = state.stack_old[0][slot]
        top = state.stack_new[0][slot]
        rows.append(
            [
                {"id": "a%d" % slot, "number": CARD_TABLE[aside][0], "action": int(CARD_TABLE[aside][1])},
                {"id": "t%d" % slot, "number": CARD_TABLE[top][0], "action": int(CARD_TABLE[top][1])},
            ]
        )
    return rows


def _selected_stack(state: GameState):
    """What ``Player::getSelectedCards`` would return: the slot taken this turn."""
    return state.turn_choice[state.actor]


def _state_args(state: GameState) -> dict:
    """``argPrivatePlayerTurn`` plus whatever the state adds.

    ``currentPlan`` matters: at ``validatePlan`` the plan has been chosen and
    NOT yet recorded in ``planValidations`` (``AbstractPlan::validate`` writes
    that row only after every estate is handed over), so its conditions are the
    only thing naming it.
    """
    args = {"selectedCards": _selected_stack(state)}
    if state.phase is Phase.VALIDATE_PLAN:
        plan = PLANS[state.plan_ids[state.ctx.plan_slot]]
        args["currentPlan"] = {"conditions": list(plan.required_sizes), "estates": []}
    return args


def bga_payload(state: GameState, *, ledger: str = "complete") -> dict:
    """Render ``state`` as the browser would capture it.

    ``ledger='short'`` drops the oldest half of the seen-card ledger, which is
    what a table joined or reloaded mid-game looks like.
    """

    assert state.actor == 0, "a capture is always taken while the viewer acts"
    players = {}
    for seat in range(state.config.players):
        start = state.public_sheets[seat]
        players[str(100 + seat)] = {
            "id": 100 + seat,
            "no": seat + 1,
            "name": "P%d" % seat,
            "score": 0,
            "scoreSheet": {
                "houses": _houses(start),
                "scribbles": _scribbles(start, _PAST),
            },
        }

    seen = [
        [CARD_TABLE[card][0], int(CARD_TABLE[card][1])]
        for card in state.deck[: state.deck_pos]
    ]
    if ledger == "short":
        seen = seen[len(seen) // 2 :]

    validations = []
    for slot in range(3):
        validations.append(
            {
                str(100 + seat): {"rank": 0, "turn": turn}
                for seat, turn in state.plan_turns[slot].items()
            }
        )

    return {
        "bga": {
            "me_id": "100",
            "turn": state.turn,
            "cardsLeft": state.deck_remaining,
            "options": {
                "standard": True,
                "advanced": state.config.advanced,
                "expert": False,
                "solo": False,
                "board": 0,
            },
            "players": players,
            "constructionCards": _stack_rows(state),
            "planCards": [{"id": p, "desc": ""} for p in state.plan_ids],
            "planValidations": validations,
            "gamestate": {
                "name": _STATE_OF_PHASE[state.phase],
                "args": _state_args(state),
            },
        },
        "dom": _delta(state.public_sheets[0], state.sheets[0], state.turn),
        "seen": seen,
    }


# ---------------------------------------------------------------------------
# Positions to test against
# ---------------------------------------------------------------------------
def _capture_points(seed: int, players: int, advanced: bool, limit: int, *, bot=None):
    """Play a real game, yielding every position seat 0 is asked to act in.

    Random play is the default because it is the rules fuzzer: it reaches odd
    sheets, permit refusals and blocked streets that a competent player never
    would. It is useless for the plan phases, though -- uniform play does not
    complete a City Plan in a two-player game at all (measured: zero
    ``CHOOSE_PLAN`` entries over 39 seeds) -- so those tests pass ``GreedyBot``,
    which does.
    """
    rng = random.Random(seed)
    state = GameState.new(
        seed=seed,
        config=GameConfig(players=players, advanced=advanced, solo_rules=False),
    )
    produced = 0
    while not state.is_terminal and produced < limit:
        if (
            state.actor == 0
            and state.phase in _STATE_OF_PHASE
            # BGA collects every estate for a plan in ONE client-side step, so
            # only the first of the engine's per-estate nodes is a state a
            # capture can be taken in.
            and not (
                state.phase is Phase.VALIDATE_PLAN and state.ctx.chosen_estates
            )
        ):
            yield state.copy()
            produced += 1
        state.apply(
            bot.act(state) if bot is not None else rng.choice(state.legal_actions())
        )
    return


def _same_sheet(left: Sheet, right: Sheet) -> None:
    assert left.numbers == right.numbers
    assert left.is_bis == right.is_bis
    assert left.fences == right.fences
    assert left.top_fences == right.top_fences
    assert left.parks == right.parks
    assert left.pools == right.pools
    assert left.estate_marks == right.estate_marks
    assert (left.temps, left.bis_marks, left.permits, left.roundabouts) == (
        right.temps,
        right.bis_marks,
        right.permits,
        right.roundabouts,
    )


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "seed,players,advanced",
    [(1, 2, False), (2, 3, True), (3, 4, True), (4, 2, True)],
)
def test_capture_round_trips_through_the_mapper(seed, players, advanced):
    """Every reachable capture rebuilds the position it was taken from.

    Everything the viewer can see has to survive: the sheets, the phase, the
    partially-resolved turn, the table, the plans and the deck's *composition*
    (its order is hidden and is re-shuffled by the mapper on purpose).
    """

    checked = 0
    for original in _capture_points(seed, players, advanced, limit=60):
        # At `writeNumber` the engine's own phase is inside a macro, so the
        # mapper backs up to the card choice it belongs to. The board is
        # identical there -- nothing has been written -- which is exactly what
        # the sheet comparisons below assert.
        inside_macro = original.phase is Phase.WRITE_NUMBER
        expected_phase = Phase.CHOOSE_CARDS if inside_macro else original.phase

        rebuilt, obs, warnings = state_from_bga_payload(
            bga_payload(original), rng=random.Random(0)
        )
        where = "%s turn %d" % (original.phase.name, original.turn)

        assert not warnings, where
        assert rebuilt.phase is expected_phase, where
        assert rebuilt.turn == original.turn, where
        assert rebuilt.actor == 0, where
        assert rebuilt.config.players == original.config.players
        assert rebuilt.plan_ids == original.plan_ids, where
        assert rebuilt.plan_turns == original.plan_turns, where

        for seat in range(players):
            _same_sheet(rebuilt.sheets[seat], original.sheets[seat])
            _same_sheet(rebuilt.public_sheets[seat], original.public_sheets[seat])

        assert rebuilt.visible_cards(0) == original.visible_cards(0), where
        assert rebuilt.next_effects(0) == original.next_effects(0), where

        # The mid-turn context is inferred, never carried, so it is the part
        # most likely to be silently wrong.
        if inside_macro:
            # Backed up before the combination was taken, so the context is
            # empty -- but the stack the player actually chose must still be one
            # the rebuilt position offers.
            assert rebuilt.ctx.slot is None, where
            assert original.ctx.slot in rebuilt.playable_slots(), where
            assert obs["selected_stack"] == original.ctx.slot, where
        else:
            assert rebuilt.ctx.number == original.ctx.number, where
            assert rebuilt.ctx.effect == original.ctx.effect, where
            assert rebuilt.ctx.slot == original.ctx.slot, where
            assert rebuilt.ctx.last_house == original.ctx.last_house, where
            assert rebuilt.ctx.pending_sizes == original.ctx.pending_sizes, where
            assert rebuilt.ctx.chosen_estates == original.ctx.chosen_estates, where

        # Order is hidden and deliberately resampled; composition is public
        # bookkeeping and must be exact.
        assert rebuilt.deck_remaining == original.deck_remaining, where
        assert sorted(CARD_TABLE[c] for c in rebuilt.discard) == sorted(
            CARD_TABLE[c] for c in original.discard
        ), where

        # And the position has to be *usable*: the search must offer exactly the
        # moves the real position offers.
        if not inside_macro:
            assert sorted(mc.search_legal_macros(rebuilt)) == sorted(
                mc.search_legal_macros(original)
            ), where
        else:
            assert mc.search_legal_macros(rebuilt), where
        checked += 1

    assert checked > 10, "the game ended before enough positions were captured"


def test_every_advisable_phase_is_exercised():
    """A round trip that never reaches the interesting phases proves little."""
    seen = set()
    for seed in range(1, 12):
        for state in _capture_points(seed, 3, True, limit=200):
            seen.add(state.phase)
    # ASK_RESHUFFLE needs a first plan completed on the very turn it is offered,
    # and VALIDATE_PLAN needs a non-automatic plan; both are rare under random
    # play, so they are checked separately below rather than demanded here.
    assert {
        Phase.CHOOSE_CARDS,
        Phase.WRITE_NUMBER,
        Phase.ACTION_SURVEYOR,
        Phase.ACTION_ESTATE,
        Phase.ACTION_PARK,
        Phase.ACTION_POOL,
        Phase.ACTION_BIS,
        Phase.CHOOSE_PLAN,
        Phase.ROUNDABOUT_PLACE,
    } <= seen


def test_plan_validation_round_trips_when_it_occurs():
    """The estate hand-over, which is the one mark set with no direct encoding.

    BGA validates a plan in a single call carrying every estate at once, while
    the engine asks for them one at a time. The mapper recovers the picks from
    the houses the validation consumed (this turn's ``top-fence`` marks), so it
    has to be exercised on a real validation rather than argued about.
    """

    seen_phases = set()
    checked = 0
    for seed in range(1, 12):
        bot = GreedyBot(rng=random.Random(seed))
        for state in _capture_points(seed, 2, False, limit=400, bot=bot):
            if state.phase not in (
                Phase.VALIDATE_PLAN,
                Phase.ASK_RESHUFFLE,
                Phase.CHOOSE_PLAN,
            ):
                continue
            rebuilt, _obs, _warn = state_from_bga_payload(
                bga_payload(state), rng=random.Random(0)
            )
            assert rebuilt.phase is state.phase, state.phase.name
            assert rebuilt.plan_turns == state.plan_turns, state.phase.name
            assert rebuilt.ctx.pending_sizes == state.ctx.pending_sizes
            assert rebuilt.ctx.chosen_estates == state.ctx.chosen_estates
            _same_sheet(rebuilt.sheets[0], state.sheets[0])
            seen_phases.add(state.phase)
            checked += 1

    assert checked, "GreedyBot never reached a plan decision"
    # The two that need the estate hand-over recovered from top-fence marks.
    assert Phase.VALIDATE_PLAN in seen_phases
    assert Phase.ASK_RESHUFFLE in seen_phases


# ---------------------------------------------------------------------------
# The deck ledger
# ---------------------------------------------------------------------------
def test_a_short_ledger_is_reported_rather_than_guessed_silently():
    """Joining or reloading mid-game leaves cards unaccounted for.

    The deck SIZE stays right -- BGA tells us that -- but which cards are in the
    discard becomes a guess, and every number forecast downstream inherits it.
    A silent approximation here is how a scrape starts lying, so it must come
    back as a warning while still producing a usable position.
    """

    state = next(_capture_points(seed=5, players=3, advanced=False, limit=1))
    for _ in range(30):  # get well past the first turn so the ledger has depth
        if state.is_terminal:
            break
        state.apply(random.Random(1).choice(state.legal_actions()))
    while not (state.actor == 0 and state.phase in _STATE_OF_PHASE):
        state.apply(random.Random(2).choice(state.legal_actions()))

    rebuilt, _obs, warnings = state_from_bga_payload(
        bga_payload(state, ledger="short"), rng=random.Random(0)
    )
    assert warnings and "ledger" in warnings[0]
    # Still a legal, playable position, and still the right deck size.
    assert rebuilt.deck_remaining == state.deck_remaining
    assert mc.search_legal_macros(rebuilt)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def _base_payload():
    state = next(_capture_points(seed=9, players=2, advanced=False, limit=1))
    return bga_payload(state)


def test_expert_and_seasonal_tables_are_refused_by_name():
    payload = _base_payload()
    payload["bga"]["options"]["standard"] = False
    with pytest.raises(UnsupportedBgaState, match="expert and solo"):
        wire_from_bga_payload(payload)

    payload = _base_payload()
    payload["bga"]["options"]["board"] = 2
    with pytest.raises(UnsupportedBgaState, match="seasonal"):
        wire_from_bga_payload(payload)


def test_idle_and_unknown_states_are_distinguished():
    """"Nothing to advise" and "I do not understand this table" are different
    problems for the person reading the panel."""
    payload = _base_payload()
    payload["bga"]["gamestate"]["name"] = "waitOthers"
    with pytest.raises(UnsupportedBgaState, match="nothing to advise"):
        wire_from_bga_payload(payload)

    payload = _base_payload()
    payload["bga"]["gamestate"]["name"] = "iceCream"
    with pytest.raises(UnsupportedBgaState, match="unknown private state"):
        wire_from_bga_payload(payload)


def test_a_spectator_frame_is_refused():
    payload = _base_payload()
    payload["bga"]["me_id"] = "999"
    with pytest.raises(UnsupportedBgaState, match="not seated"):
        wire_from_bga_payload(payload)


def test_a_capture_that_contradicts_the_rules_fails_loudly():
    """The replay is the check, not just the reconstruction.

    A house that could not have come from the chosen combination is exactly the
    shape of a stale capture -- BGA animates a number into its box, and a read
    taken mid-flight sees the previous turn's board with this turn's state. The
    mapper must refuse rather than hand the advisor a board nobody is playing.
    """

    state = None
    for candidate in _capture_points(seed=6, players=2, advanced=False, limit=200):
        if candidate.phase is Phase.ACTION_SURVEYOR:
            state = candidate
            break
    assert state is not None

    payload = bga_payload(state)
    for house in payload["dom"]["houses"]:
        house["number"] = (house["number"] + 7) % 16  # not a legal write here
    with pytest.raises(StaleGamedata):
        state_from_bga_payload(payload, rng=random.Random(0))


def test_a_capture_missing_the_chosen_combination_fails_loudly():
    """``selectedCards`` is the one field of the turn that has no DOM trace.

    The written house says *what* was written but not which of the three
    combinations produced it, and two stacks can offer the same number. So a
    capture that has moved past the card choice without carrying it is
    unreconstructable, and has to say so.
    """

    state = None
    for candidate in _capture_points(seed=6, players=2, advanced=False, limit=200):
        if candidate.phase is Phase.ACTION_SURVEYOR:
            state = candidate
            break
    assert state is not None

    payload = bga_payload(state)
    payload["bga"]["gamestate"]["args"]["selectedCards"] = None
    with pytest.raises(StaleGamedata, match="which combination"):
        state_from_bga_payload(payload, rng=random.Random(0))


def test_the_write_state_is_backed_up_to_the_card_choice():
    """At ``writeNumber`` the advisor answers the question one step earlier.

    ``WRITE_NUMBER`` lives inside a macro, so the search has no decision to
    offer there; and nothing has been written yet, so the position genuinely is
    the one at ``chooseCards``. Backing up therefore costs nothing and answers
    the more useful question -- whether the combination just taken was right.
    """

    state = None
    for candidate in _capture_points(seed=6, players=2, advanced=False, limit=200):
        if candidate.phase is Phase.WRITE_NUMBER:
            state = candidate
            break
    assert state is not None

    rebuilt, obs, _warn = state_from_bga_payload(
        bga_payload(state), rng=random.Random(0)
    )
    assert obs["state"] == "writeNumber"
    assert rebuilt.phase is Phase.CHOOSE_CARDS
    assert rebuilt.ctx.slot is None
    _same_sheet(rebuilt.sheets[0], state.sheets[0])
    # The combination the player took is still on the table and still legal, so
    # the advice covers what they actually did as well as the alternatives.
    assert obs["selected_stack"] in rebuilt.playable_slots()
