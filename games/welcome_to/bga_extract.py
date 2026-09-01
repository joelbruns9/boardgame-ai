"""BGA *Welcome To...* table -> engine ``GameState``.

The browser hands over ``window.gameui.gamedatas`` verbatim (plus a small DOM
patch and a card ledger); every rule for turning that into a position lives
here, in Python, for the same reason 7WD's does: a BGA UI change then breaks a
mapper loudly instead of silently producing a plausible-but-wrong board.

WHAT BGA GIVES US, AND WHAT IT WITHHOLDS
----------------------------------------
``gamedatas.players[pid].scoreSheet`` is **the sheet as it stood at the start of
the current turn**, for everyone including yourself.  ``notif_updatePlayersData``
is the only thing that rewrites it and it fires at ``stApplyTurn``; everything a
player scribbles mid-turn goes to the DOM only (``notif_addScribble``,
``addHouseNumber``).  That is not a defect to work around -- it is exactly
:attr:`GameState.public_sheets`, so the scrape lands on the engine's own
information-set boundary with nothing to hide by hand.

Your *own* current-turn progress is then read back off the DOM (``dom`` in the
payload) and **replayed as engine actions** rather than poked into a
:class:`~games.welcome_to.game.TurnCtx`.  Replay is what makes the
reconstruction checkable: the engine refuses an action it would not have
offered, so a capture that disagrees with the rules fails here instead of being
advised on.

SEAT 0 IS THE VIEWER
--------------------
Welcome To is simultaneous and the engine serialises a turn into consecutive
private turns, which is safe because nothing a player does during a turn changes
what another player may do during it (see :mod:`games.welcome_to.game`).  So the
seat *order* carries no information, and we are free to reindex it: the human
becomes seat 0 and the remaining players follow in BGA turn order.  That makes
the reconstruction self-consistent -- every other seat sits at its turn-start
sheet because nobody has acted yet this turn -- where keeping the real seat index
would leave the seats before the viewer looking as if they had skipped a turn.

THE DECK
--------
Composition, not order, is what the engine needs
(:mod:`games.welcome_to.deck_knowledge`), and composition is public bookkeeping
for anyone who watched the table.  The extension keeps a ledger of every card
face it has seen; ``discard`` is that ledger minus the six cards on the table,
and the undrawn deck is the printed 81 minus the ledger, shuffled.  A ledger
that started late (a page reload mid-game) leaves fewer identified cards than
``cardsLeft`` implies; the shortfall is drawn at random from the unseen pool and
reported as a **warning**, because a guessed discard is a real approximation and
silently rolling it in is how a scrape starts lying.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from games.welcome_to import action_codec as codec
from games.welcome_to.constants import (
    CARD_TABLE,
    Effect,
    NUM_BASE_CARDS,
    ROUNDABOUT,
    STREET_SIZES,
    TEMP_DELTAS,
)
from games.welcome_to.game import GameConfig, GameState, IllegalAction, Phase, TurnCtx
from games.welcome_to.plans import PLANS
from games.welcome_to.sheet import Sheet


class UnsupportedBgaState(ValueError):
    """A table the advisor deliberately will not read (variant, seat, phase)."""


class StaleGamedata(ValueError):
    """The capture contradicts itself and cannot be trusted."""


# ---------------------------------------------------------------------------
# BGA private state name  ->  engine phase
# ---------------------------------------------------------------------------
#: ``states.inc.php`` names, for the private states the advisor can advise at.
#:
#: ⚠ ``writeNumber`` maps to ``CHOOSE_CARDS``, not to ``WRITE_NUMBER``, and that
#: is deliberate on two counts.  Structurally, the macro vocabulary folds
#: ``CHOOSE_CARDS -> WRITE_NUMBER`` into one action -- you pick a combination
#: *for* a placement -- so ``WRITE_NUMBER`` is inside a macro and the search has
#: no decision to offer there at all (``macro_codec.legal_macros`` refuses it).
#: Factually, nothing has happened yet: BGA has logged which stack you took and
#: written nothing, and ``restart`` puts you back at ``chooseCards``.  So the
#: position at ``writeNumber`` *is* the position at ``chooseCards``, and backing
#: up to it loses nothing while gaining the more useful answer -- whether the
#: combination you just took was the right one.  The wire still reports
#: ``selected_stack`` so a UI can say which one that was.
PHASE_OF_STATE: dict[str, Phase] = {
    "chooseCards": Phase.CHOOSE_CARDS,
    "buildRoundabout": Phase.ROUNDABOUT_PLACE,
    "writeNumber": Phase.CHOOSE_CARDS,
    "actionSurveyor": Phase.ACTION_SURVEYOR,
    "actionEstate": Phase.ACTION_ESTATE,
    "actionPark": Phase.ACTION_PARK,
    "actionPool": Phase.ACTION_POOL,
    "actionBis": Phase.ACTION_BIS,
    "choosePlan": Phase.CHOOSE_PLAN,
    "validatePlan": Phase.VALIDATE_PLAN,
    "askReshuffle": Phase.ASK_RESHUFFLE,
}

#: States that exist but hold no decision the advisor can help with.  Named
#: rather than lumped in with "unknown" so the panel can say *why* it is quiet.
IDLE_STATES: frozenset = frozenset(
    {
        "playerTurn",
        "newTurn",
        "applyTurns",
        "confirmTurn",
        "waitOthers",
        "computeScores",
        "gameEnd",
        "gameSetup",
    }
)

#: Scribble types that are pure counters on a :class:`Sheet`.
_COUNTED_SCRIBBLES: dict[str, str] = {
    "score-temp": "temps",
    "score-bis": "bis_marks",
    "permit-refusal": "permits",
    "score-roundabout": "roundabouts",
}


# ---------------------------------------------------------------------------
# Card identity
# ---------------------------------------------------------------------------
class _CardPool:
    """Hands out distinct :data:`CARD_TABLE` indices for a ``(number, effect)``.

    The printed deck holds two copies of some pairs, so a face does not identify
    a card.  Every allocation comes from here, and running out means the capture
    claims more copies of a card than exist -- a real contradiction, raised
    rather than papered over.
    """

    def __init__(self) -> None:
        self._by_face: dict[tuple[int, int], list[int]] = {}
        for card_id in range(NUM_BASE_CARDS):
            number, effect = CARD_TABLE[card_id]
            self._by_face.setdefault((int(number), int(effect)), []).append(card_id)

    def take(self, number: int, effect: int) -> int:
        pool = self._by_face.get((int(number), int(effect)))
        if not pool:
            raise StaleGamedata(
                "capture names more copies of card (%s, %s) than the printed "
                "deck holds" % (number, Effect(int(effect)).name)
            )
        return pool.pop()

    def remaining(self) -> list[int]:
        return [card for pool in self._by_face.values() for card in pool]


def _face(card: dict) -> tuple[int, int]:
    """``(number, effect)`` off a BGA construction-card row."""
    try:
        return int(card["number"]), int(card["action"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StaleGamedata("unreadable construction card %r" % (card,)) from exc


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------
def _apply_house(sheet: Sheet, house: dict) -> None:
    x, y = int(house["x"]), int(house["y"])
    if not (0 <= x < 3 and 0 <= y < STREET_SIZES[x]):
        raise StaleGamedata("house out of the sheet: %r" % (house,))
    sheet.numbers[x][y] = int(house["number"])
    sheet.is_bis[x][y] = bool(house.get("isBis"))
    sheet.written_turn[x][y] = int(house.get("turn", -1))


def _apply_scribble(sheet: Sheet, scribble: dict) -> None:
    kind = str(scribble["type"])
    x = scribble.get("x")
    y = scribble.get("y")
    x = None if x in (None, "") else int(x)
    y = None if y in (None, "") else int(y)

    if kind in _COUNTED_SCRIBBLES:
        field = _COUNTED_SCRIBBLES[kind]
        setattr(sheet, field, getattr(sheet, field) + 1)
        return
    if kind == "estate-fence":
        sheet.fences[x][y] = True
        return
    if kind == "top-fence":
        sheet.top_fences[x][y] = True
        return
    if kind == "park":
        sheet.parks[x] += 1
        return
    if kind == "pool":
        # The circle drawn on the house, which is what Sheet.pools counts
        # (Actions/Pool::getCompleted reads exactly these).
        sheet.pools[x] += 1
        return
    if kind == "score-estate":
        sheet.estate_marks[x] += 1
        return
    if kind == "score-pool":
        # Derived from the `pool` circles (Sheet.pool_score sums Sheet.pools);
        # storing it twice would let the two disagree.
        return
    raise UnsupportedBgaState(
        "scribble type %r is not part of the base game; the seasonal boards "
        "(Ice Cream / Christmas / Easter) are out of scope" % (kind,)
    )


def _sheet_from_wire(wire: dict, *, up_to_turn: Optional[int] = None) -> Sheet:
    """Build a sheet from the wire's houses/scribbles.

    ``up_to_turn`` drops marks made on that turn or later, which is how the
    viewer's turn-start sheet is recovered from a capture that already includes
    their current-turn DOM marks.
    """
    sheet = Sheet.new()
    for house in wire.get("houses") or []:
        if up_to_turn is not None and int(house.get("turn", -1)) >= up_to_turn:
            continue
        _apply_house(sheet, house)
    for scribble in wire.get("scribbles") or []:
        if up_to_turn is not None and int(scribble.get("turn", -1)) >= up_to_turn:
            continue
        _apply_scribble(sheet, scribble)
    return sheet


# ---------------------------------------------------------------------------
# The normalized wire
# ---------------------------------------------------------------------------
def wire_from_bga_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """``{"bga": gamedatas, "dom": ..., "seen": ...}`` -> the observation wire.

    Pure re-shaping and validation: no engine object is built here, so the result
    is JSON, is what the game log stores, and is what
    :func:`state_from_observation` alone interprets.
    """

    g = payload.get("bga")
    if not isinstance(g, dict):
        raise StaleGamedata("payload has no `bga` gamedatas object")

    options = g.get("options") or {}
    if not options.get("standard", True):
        raise UnsupportedBgaState(
            "expert and solo tables have six ordered card pairs and no macro "
            "representation; the trained net covers standard 2-4 player games"
        )
    if int(options.get("board", 0) or 0) != 0:
        raise UnsupportedBgaState(
            "seasonal boards (Ice Cream / Christmas / Easter) are not implemented"
        )

    state_name = str(((g.get("gamestate") or {}).get("name")) or "")
    if state_name in IDLE_STATES:
        raise UnsupportedBgaState("nothing to advise in state %r" % (state_name,))
    if state_name not in PHASE_OF_STATE:
        raise UnsupportedBgaState("unknown private state %r" % (state_name,))

    me = str(g.get("me_id") or "")
    players = g.get("players") or {}
    players = {str(k): v for k, v in players.items()}
    if me not in players:
        raise UnsupportedBgaState(
            "this frame is not seated at the table (me_id %s); the advisor "
            "needs the playing seat" % (me or "missing",)
        )

    # BGA turn order is `player_no`; the viewer is rotated to seat 0.
    ordered = sorted(players.values(), key=lambda p: int(p["no"]))
    pids = [str(p["id"]) for p in ordered]
    seat_pids = pids[pids.index(me) :] + pids[: pids.index(me)]

    turn = int(g.get("turn", 1))

    stacks = g.get("constructionCards") or []
    if len(stacks) != 3:
        raise StaleGamedata("expected three stacks, got %d" % (len(stacks),))
    wire_stacks = []
    for i, stack in enumerate(stacks):
        if len(stack) < 2:
            raise StaleGamedata(
                "stack %d has %d card(s); a standard stack shows two -- the "
                "number card on top and the effect card flipped aside beside it"
                % (i, len(stack))
            )
        # getTopOf orders by state DESC, so [0] is the aside card whose EFFECT is
        # in play and [1] is the top card whose NUMBER is in play -- exactly what
        # ConstructionCards::getCombination reads.
        wire_stacks.append({"old": list(_face(stack[0])), "new": list(_face(stack[1]))})

    plan_ids = [int(p["id"]) for p in (g.get("planCards") or [])]
    if len(plan_ids) != 3:
        raise StaleGamedata("expected three City Plans, got %d" % (len(plan_ids),))

    raw_validations = g.get("planValidations") or []
    plan_turns: list[dict[str, int]] = []
    for slot in range(3):
        raw = (raw_validations[slot] if slot < len(raw_validations) else {}) or {}
        plan_turns.append(
            {
                str(seat_pids.index(str(pid))): int(v["turn"])
                for pid, v in raw.items()
                if str(pid) in seat_pids
            }
        )

    dom = payload.get("dom") or {}
    args = (g.get("gamestate") or {}).get("args") or {}
    my_seat = "0"

    return {
        "version": 1,
        "game": "welcome_to",
        "state": state_name,
        "turn": turn,
        "advanced": bool(options.get("advanced")),
        "cards_left": int(g.get("cardsLeft", 0)),
        "seats": [
            {
                "pid": pid,
                "name": str(players[pid].get("name", pid)),
                "score": int(players[pid].get("score") or 0),
                # BGA's own turn-start snapshot: `public_sheets`, verbatim.
                "sheet": _sheet_wire(players[pid].get("scoreSheet") or {}),
            }
            for pid in seat_pids
        ],
        # The viewer's current-turn marks, read off the DOM because gamedatas
        # will not carry them until stApplyTurn.
        "my_turn_marks": {
            "houses": list(dom.get("houses") or []),
            "scribbles": list(dom.get("scribbles") or []),
        },
        "stacks": wire_stacks,
        "plans": plan_ids,
        "plan_turns": plan_turns,
        "seen_cards": [[int(n), int(e)] for n, e in (payload.get("seen") or [])],
        "reshuffle_pending": bool(payload.get("reshuffle_pending", False)),
        "selected_stack": _selected_stack(args),
        # `argValidatePlan` sends the plan's own conditions, not its id
        # (Plans/EstatePlan.php:65-71). Only estate plans are non-automatic, and
        # their conditions ARE their required sizes, so this identifies which of
        # the three is being validated -- see `_TurnMarks._validating_slot`.
        "current_plan_conditions": _plan_conditions(args),
        "validated_plan_slots": [
            slot for slot in range(3) if plan_turns[slot].get(my_seat) == turn
        ],
    }


def _sheet_wire(score_sheet: dict) -> dict[str, Any]:
    """The raw houses/scribbles, carried through unchanged.

    Deliberately *not* pre-digested into a :class:`Sheet` snapshot: a logged line
    then still says what BGA said, so a mapping bug found later can be re-run
    against the original bytes.
    """
    return {
        "houses": [
            {
                "x": int(h["x"]),
                "y": int(h["y"]),
                "number": int(h["number"]),
                "isBis": bool(h.get("isBis")),
                "turn": int(h.get("turn", -1)),
            }
            for h in (score_sheet.get("houses") or [])
        ],
        "scribbles": [
            {
                "type": str(s["type"]),
                "x": None if s.get("x") in (None, "") else int(s["x"]),
                "y": None if s.get("y") in (None, "") else int(s["y"]),
                "turn": int(s.get("turn", -1)),
            }
            for s in (score_sheet.get("scribbles") or [])
        ],
    }


def _plan_conditions(args: dict) -> Optional[list[int]]:
    """The required estate sizes of the plan currently being validated."""
    current = args.get("currentPlan")
    if not isinstance(current, dict):
        return None
    conditions = current.get("conditions")
    if conditions is None:
        return None
    return [int(c) for c in conditions]


def _selected_stack(args: dict) -> Optional[int]:
    """The combination already taken this turn, if any (``selectedCards``).

    ``Player::getSelectedCards`` reads the turn's ``selectCard`` log entry, so
    this is null at ``chooseCards`` and an int at every later private state.
    """
    value = args.get("selectedCards")
    if value is None:
        return None
    if isinstance(value, (list, tuple)):  # expert mode ships a pair
        raise UnsupportedBgaState("expert-mode card selection is not supported")
    return int(value)


# ---------------------------------------------------------------------------
# Observation -> GameState
# ---------------------------------------------------------------------------
def state_from_observation(
    obs: dict[str, Any], *, rng: Optional[random.Random] = None
) -> tuple[GameState, list[str]]:
    """Rebuild an engine state from the observation wire.

    Returns the state and any warnings the human reading the numbers needs -- an
    incomplete card ledger being the one that actually happens.
    """

    rng = rng or random.Random(0)
    warnings: list[str] = []
    players = len(obs["seats"])
    if not 2 <= players <= 4:
        raise UnsupportedBgaState(
            "the trained net covers 2-4 seats; this table has %d" % (players,)
        )

    config = GameConfig(
        players=players,
        advanced=bool(obs["advanced"]),
        expert=False,
        solo_rules=False,
    )
    turn = int(obs["turn"])

    # -- sheets ------------------------------------------------------------
    # Every seat is at its turn-start sheet, the viewer included: their own
    # current-turn marks are replayed as actions below, not written in here.
    sheets = [_sheet_from_wire(seat["sheet"]) for seat in obs["seats"]]

    # -- the table ---------------------------------------------------------
    pool = _CardPool()
    stack_new: list[Optional[int]] = []
    stack_old: list[Optional[int]] = []
    for stack in obs["stacks"]:
        stack_new.append(pool.take(*stack["new"]))
        stack_old.append(pool.take(*stack["old"]))

    discard, deck, ledger_warning = _deck_from_ledger(
        pool,
        seen=[tuple(c) for c in obs["seen_cards"]],
        table_faces=[tuple(s["new"]) for s in obs["stacks"]]
        + [tuple(s["old"]) for s in obs["stacks"]],
        cards_left=int(obs["cards_left"]),
        rng=rng,
    )
    if ledger_warning:
        warnings.append(ledger_warning)

    # The viewer's own validations from THIS turn are dropped and re-applied by
    # the replay below. Leaving them in would make the plan unscorable
    # (`scorable_plan_slots` skips a plan you have already taken), so the replay
    # would find nothing to do and pass the turn instead -- the position would
    # come back one decision short, with no error.
    plan_turns = [
        {
            int(seat): int(t)
            for seat, t in slot.items()
            if not (int(seat) == 0 and int(t) == turn)
        }
        for slot in obs["plan_turns"]
    ]

    state = GameState(
        config=config,
        sheets=[s.copy() for s in sheets],
        public_sheets=sheets,
        deck=deck,
        deck_pos=0,
        discard=discard,
        stack_new=[stack_new],
        stack_old=[stack_old],
        expert_pending=[None] * players,
        plan_ids=tuple(int(p) for p in obs["plans"]),  # type: ignore[arg-type]
        plan_turns=plan_turns,
        turn=turn,
        actor=0,
        phase=Phase.CHOOSE_CARDS,
        ctx=TurnCtx(),
        turn_choice=[None] * players,
        reshuffle_next_turn=bool(obs.get("reshuffle_pending", False)),
        reshuffle_votes={},
        rng=random.Random(rng.getrandbits(64)),
    )

    target = PHASE_OF_STATE[obs["state"]]
    _replay_my_turn(state, obs, target)
    return state, warnings


def _deck_from_ledger(
    pool: _CardPool,
    *,
    seen: list[tuple[int, int]],
    table_faces: list[tuple[int, int]],
    cards_left: int,
    rng: random.Random,
) -> tuple[list[int], list[int], Optional[str]]:
    """Split the printed deck into ``(discard, undrawn)``.

    ``seen`` is every face the ledger recorded, table included.  The six faces on
    the table have already been allocated out of ``pool``, so only the surplus
    counts as discarded.
    """

    surplus: list[tuple[int, int]] = list(seen)
    for face in table_faces:
        if face in surplus:
            surplus.remove(face)

    discard: list[int] = []
    for face in surplus:
        try:
            discard.append(pool.take(*face))
        except StaleGamedata:
            # A ledger that double-counted a face is a browser-side bug, not a
            # reason to refuse the position: drop the extra and carry on.
            continue

    undrawn = pool.remaining()
    rng.shuffle(undrawn)

    warning = None
    shortfall = len(undrawn) - int(cards_left)
    if shortfall > 0:
        # The ledger missed cards that really were drawn (a table joined or
        # reloaded mid-game). Move the difference into the discard at random so
        # the deck SIZE is right even though the identities are guesses.
        discard.extend(undrawn[:shortfall])
        undrawn = undrawn[shortfall:]
        warning = (
            "card ledger is %d card(s) short of BGA's deck count, so that many "
            "discards are guessed; number forecasts and the reshuffle read are "
            "approximate. Reload the table at the start of a game to fix it."
            % (shortfall,)
        )
    elif shortfall < 0:
        warning = (
            "card ledger holds %d more discard(s) than BGA's deck count allows; "
            "the deck is short by that much" % (-shortfall,)
        )

    return discard, undrawn, warning


# ---------------------------------------------------------------------------
# Replaying the viewer's own current turn
# ---------------------------------------------------------------------------
def _replay_my_turn(state: GameState, obs: dict[str, Any], target: Phase) -> None:
    """Step the engine from turn start to the phase BGA is actually in.

    Each step is inferred from what the viewer has already marked this turn, and
    is applied through :meth:`GameState.apply`, which refuses anything the rules
    would not have offered.  So this both *reconstructs* the position and
    *checks* the capture: a DOM read that disagrees with the engine raises here
    rather than producing a board nobody would recognise.
    """

    marks = _TurnMarks(obs, target)
    guard = 0
    # Reaching the target phase is not the same as being finished with it. Two
    # of BGA's states are re-entered inside one turn -- `chooseCards` after a
    # roundabout, `choosePlan` after a validation -- so the loop also has to ask
    # whether the marks still hold something that must pass through this phase.
    while state.phase is not target or marks.pending_at(state.phase):
        guard += 1
        if guard > 32:
            raise StaleGamedata(
                "could not replay this turn onto BGA state %r (stuck in %s)"
                % (obs["state"], state.phase.name)
            )
        if state.actor != 0:
            raise StaleGamedata(
                "replay ran past the viewer's own turn before reaching BGA "
                "state %r" % (obs["state"],)
            )
        action = marks.next_action(state)
        try:
            state.apply(action)
        except IllegalAction as exc:
            raise StaleGamedata(
                "replay of your current turn hit an illegal step (%s); the "
                "capture is stale or the mapper is wrong" % (exc,)
            ) from exc


class _TurnMarks:
    """What the viewer has already marked this turn, consumed one action at a time.

    The DOM patch is a *set* of marks with no ordering, which is enough because
    the engine's phase says what kind of mark is next: at ``ACTION_SURVEYOR``
    only a fence can be pending, at ``ACTION_BIS`` only a bis house.  Anything
    the phase asks for and the marks do not hold is a pass.
    """

    def __init__(self, obs: dict[str, Any], target: Phase) -> None:
        turn = int(obs["turn"])
        self.target = target
        houses = [
            h for h in obs["my_turn_marks"]["houses"] if int(h.get("turn", -1)) == turn
        ]
        scribbles = [
            s
            for s in obs["my_turn_marks"]["scribbles"]
            if int(s.get("turn", -1)) == turn
        ]
        self.turn = turn
        self.selected_stack: Optional[int] = obs.get("selected_stack")
        self.roundabout = next(
            (h for h in houses if int(h["number"]) == ROUNDABOUT), None
        )
        self.written = next(
            (
                h
                for h in houses
                if int(h["number"]) != ROUNDABOUT and not h.get("isBis")
            ),
            None,
        )
        self.bis = next((h for h in houses if h.get("isBis")), None)
        self.by_type: dict[str, list[dict]] = {}
        for scribble in scribbles:
            self.by_type.setdefault(str(scribble["type"]), []).append(scribble)
        self.validated_slots: list[int] = list(obs.get("validated_plan_slots") or [])
        self.plan_conditions = obs.get("current_plan_conditions")
        self.top_fences = self.by_type.get("top-fence", [])

    # -- helpers -----------------------------------------------------------
    def _pop(self, kind: str) -> Optional[dict]:
        bucket = self.by_type.get(kind)
        if not bucket:
            return None
        return bucket.pop(0)

    def pending_at(self, phase: Phase) -> bool:
        """Whether a mark still has to be pushed *through* ``phase``.

        Only the two phases BGA re-enters mid-turn can answer yes.
        """
        if phase is Phase.CHOOSE_CARDS:
            return self.roundabout is not None
        if phase is Phase.CHOOSE_PLAN:
            return bool(self.validated_slots)
        return False

    def next_action(self, state: GameState) -> int:
        phase = state.phase

        if phase is Phase.CHOOSE_CARDS:
            # Two ways to open the roundabout dialog: the roundabout is already
            # on the sheet (BGA has moved on and we are catching up), or BGA is
            # sitting IN the dialog waiting for a box, in which case there is no
            # mark to find and the target phase is the only evidence.
            if self.roundabout is not None or self.target is Phase.ROUNDABOUT_PLACE:
                return codec.A_ROUNDABOUT_OPEN
            # A refusal taken with no combination chosen is the *direct* one --
            # nothing on the table was playable. With a slot chosen it belongs to
            # WRITE_NUMBER instead, so it must not be consumed here.
            if self.selected_stack is None:
                if self._pop("permit-refusal") is not None:
                    return codec.A_PERMIT_REFUSAL
                raise StaleGamedata(
                    "BGA is past the card choice but the capture does not say "
                    "which combination was taken"
                )
            return codec.choose_stack(int(self.selected_stack))

        if phase is Phase.ROUNDABOUT_PLACE:
            house = self.roundabout
            if house is None:
                return codec.A_PASS_ROUNDABOUT
            self.roundabout = None
            return codec.roundabout_pos(int(house["x"]), int(house["y"]))

        if phase is Phase.WRITE_NUMBER:
            house = self.written
            if house is None:
                if self._pop("permit-refusal") is not None:
                    return codec.A_PERMIT_REFUSAL
                raise StaleGamedata(
                    "BGA is past the write but no house was marked this turn"
                )
            self.written = None
            assert state.ctx.number is not None
            delta = int(house["number"]) - int(state.ctx.number)
            if delta not in TEMP_DELTAS:
                raise StaleGamedata(
                    "house %s cannot come from the chosen combination (number "
                    "%s)" % (house["number"], state.ctx.number)
                )
            return codec.write(
                TEMP_DELTAS.index(delta), int(house["x"]), int(house["y"])
            )

        if phase is Phase.ACTION_SURVEYOR:
            fence = self._pop("estate-fence")
            if fence is None:
                return codec.A_PASS_SURVEYOR
            return codec.surveyor_fence(int(fence["x"]), int(fence["y"]))

        if phase is Phase.ACTION_ESTATE:
            mark = self._pop("score-estate")
            if mark is None:
                return codec.A_PASS_ESTATE
            return codec.estate_row(int(mark["x"]))

        if phase is Phase.ACTION_PARK:
            mark = self._pop("park")
            if mark is None:
                return codec.A_PASS_PARK
            return codec.park_street(int(mark["x"]))

        if phase is Phase.ACTION_POOL:
            mark = self._pop("pool")
            if mark is None:
                return codec.A_PASS_POOL
            return codec.A_POOL_BUILD

        if phase is Phase.ACTION_BIS:
            house = self.bis
            if house is None:
                return codec.A_PASS_BIS
            self.bis = None
            return self._bis_action(state, house)

        if phase is Phase.CHOOSE_PLAN:
            if self.validated_slots:
                return codec.choose_plan(self.validated_slots.pop(0))
            # A plan being validated right now is NOT in `planValidations` yet:
            # AbstractPlan::validate writes that row only once every estate has
            # been handed over, and BGA collects them all in one client-side
            # step. So the evidence that a plan was chosen is the target state
            # itself, plus the conditions its args carry.
            if self.target is Phase.VALIDATE_PLAN:
                return codec.choose_plan(self._validating_slot(state))
            return codec.A_PASS_PLAN

        if phase is Phase.VALIDATE_PLAN:
            return self._validate_action(state)

        if phase is Phase.ASK_RESHUFFLE:
            # Which way the vote went is private to the player and BGA does not
            # publish it; "no" is the reading that leaves the deck as observed.
            return codec.A_RESHUFFLE_NO

        raise StaleGamedata("nothing to replay at phase %s" % (phase.name,))

    def _validating_slot(self, state: GameState) -> int:
        """Which of the three plans BGA is asking the player to spend estates on.

        ``argValidatePlan`` names the plan by its conditions rather than its id.
        Only estate plans are non-automatic and their conditions are exactly
        their required sizes, so among the plans this player can still score the
        match is normally unique. When it is not, refusing is the only honest
        answer: validating the wrong plan would consume the wrong houses.
        """
        if self.plan_conditions is None:
            raise StaleGamedata(
                "BGA is asking which estates to spend but the capture does not "
                "say which plan is being validated"
            )
        want = tuple(self.plan_conditions)
        matches = [
            slot
            for slot in state.scorable_plan_slots()
            if PLANS[state.plan_ids[slot]].required_sizes == want
        ]
        if len(matches) != 1:
            raise StaleGamedata(
                "cannot tell which City Plan is being validated: %d of the "
                "scorable plans ask for %s" % (len(matches), list(want))
            )
        return matches[0]

    def _bis_action(self, state: GameState, house: dict) -> int:
        """Which neighbour the bis copied, recovered from the written number."""
        x, y, number = int(house["x"]), int(house["y"]), int(house["number"])
        sheet = state.sheets[state.actor]
        for cx, cy, cnumber, side in sheet.bis_candidates():
            if (cx, cy) == (x, y) and cnumber == number:
                return codec.bis(x, y, side)
        raise StaleGamedata(
            "the bis house at (%d, %d) showing %d is not a legal duplication "
            "of either neighbour" % (x, y, number)
        )

    def _validate_action(self, state: GameState) -> int:
        """Which estate was handed to the plan being validated.

        The estates are not in the capture; the houses they consumed are, as
        this turn's ``top-fence`` marks.  Group those into runs and hand over the
        one whose length is the size the engine is currently asking for.
        """
        size = state.ctx.pending_sizes[0]
        runs = _contiguous_runs(self.top_fences)
        for index, (x, start, length) in enumerate(runs):
            if length == size:
                self.top_fences = [
                    f
                    for f in self.top_fences
                    if not (int(f["x"]) == x and start <= int(f["y"]) < start + length)
                ]
                return codec.validate_estate(x, start)
        raise StaleGamedata(
            "no run of %d consumed houses this turn to satisfy the plan being "
            "validated" % (size,)
        )


def _contiguous_runs(top_fences: list[dict]) -> list[tuple[int, int, int]]:
    """``(street, first box, length)`` for each run of consumed houses."""
    by_street: dict[int, list[int]] = {}
    for fence in top_fences:
        by_street.setdefault(int(fence["x"]), []).append(int(fence["y"]))
    runs: list[tuple[int, int, int]] = []
    for x, ys in sorted(by_street.items()):
        ys = sorted(ys)
        start = prev = ys[0]
        for y in ys[1:]:
            if y == prev + 1:
                prev = y
                continue
            runs.append((x, start, prev - start + 1))
            start = prev = y
        runs.append((x, start, prev - start + 1))
    return runs


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def state_from_bga_payload(
    payload: dict[str, Any], *, rng: Optional[random.Random] = None
) -> tuple[GameState, dict[str, Any], list[str]]:
    """``payload -> (state, observation, warnings)``, the whole pipeline."""
    obs = wire_from_bga_payload(payload)
    state, warnings = state_from_observation(obs, rng=rng)
    return state, obs, warnings


def plan_labels(plan_ids: list[int]) -> list[str]:
    """Short human names for the three City Plans in play."""
    out = []
    for slot, plan_id in enumerate(plan_ids):
        plan = PLANS[int(plan_id)]
        out.append("plan %d: %s %s" % (slot + 1, plan.kind.name.lower(), plan.params))
    return out
