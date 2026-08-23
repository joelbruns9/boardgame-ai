"""
The Welcome To game engine.

Rules source of truth: ``BGA Files/welcometo``.  Every state transition below
mirrors a state in ``states.inc.php`` and the trait method that implements it;
the docstrings name them.

SIMULTANEOUS TURNS, SERIALISED
──────────────────────────────
Welcome To is played simultaneously: every player sees the same three
construction-card combinations and fills in their own sheet at the same time.
This engine serialises a turn into ``players`` consecutive private turns, which
is safe here in a way it usually is not, because nothing one player does during
a turn can change what another player may do during the same turn:

* the three stacks are shared but not consumed -- everyone may take the same
  combination;
* City Plans are ranked by *turn number*, not by seat order, so two players who
  finish the same plan on the same turn both collect the first-place value
  (``AbstractPlan::getValidations``);
* the temp-agency ranking and the end-of-game check are both evaluated once, at
  the end of the turn.

What serialising *would* leak is information: player 3 must not see what
players 1 and 2 wrote this turn.  BGA hides exactly that (``Houses::getOfPlayer``
filters out the current turn for everyone but the current player), and so does
this engine: :attr:`GameState.public_sheets` is a snapshot taken at the start of
each turn, and :meth:`GameState.sheet_for` is what the encoder must read.  The
raw ``sheets`` list is ground truth and is *not* information-set safe.

The one genuine deviation is that expert mode passes the unused card to the
next player, so a serialised expert turn does change what a later player holds
-- but that card is passed at the *end* of the turn in BGA too
(``stApplyTurn``), which this engine reproduces.

CHANCE AND DETERMINIZATION
──────────────────────────
Very little is hidden.  A card's number face prints the effect from its own back
in the corners, so both cards of a stack are fully identified and the effect each
stack will offer *next* turn is already known (:meth:`GameState.next_effects`).
The only unknown is the order of the undrawn deck -- its composition is public
bookkeeping.  ``GameState`` stores the true shuffle, so search running forward
from a real state would cheat; :meth:`GameState.redeterminize` permutes it and is
what an MCTS root must call.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional

import numpy as np

from games.welcome_to import action_codec as codec
from games.welcome_to.constants import (
    BIS_BOXES,
    CARD_TABLE,
    Effect,
    NUM_BASE_CARDS,
    PERMIT_BOXES,
    SOLO_CARD_ID,
    SOLO_DECK_MIDDLE,
    TEMP_BOXES,
    TEMP_DELTAS,
    TEMP_RANK_SCORES,
    TEMP_SOLO_SCORE,
    TEMP_SOLO_THRESHOLD,
    MAX_NUMBER,
    MIN_NUMBER,
)
from games.welcome_to.plans import (
    PLANS,
    Plan,
    available_plan_ids,
    can_be_scored,
    estates_matching_size,
    validation_cells,
)
from games.welcome_to.sheet import Estate, Pos, Sheet, SheetScore

#: The mock player id BGA uses when the solo card validates every plan.
SOLO_MOCK_PLAYER: int = -1


class Phase(IntEnum):
    """One per decision point.  Mirrors the private states of ``states.inc.php``."""

    CHOOSE_CARDS = 0      # ST_CHOOSE_CARDS
    ROUNDABOUT_PLACE = 1  # ST_ROUNDABOUT
    WRITE_NUMBER = 2      # ST_WRITE_NUMBER
    ACTION_SURVEYOR = 3   # ST_ACTION_SURVEYOR
    ACTION_ESTATE = 4     # ST_ACTION_ESTATE
    ACTION_PARK = 5       # ST_ACTION_PARK
    ACTION_POOL = 6       # ST_ACTION_POOL
    ACTION_BIS = 7        # ST_ACTION_BIS
    CHOOSE_PLAN = 8       # ST_CHOOSE_PLAN
    VALIDATE_PLAN = 9     # ST_VALIDATE_PLAN
    ASK_RESHUFFLE = 10    # ST_ASK_RESHUFFLE
    GAME_OVER = 11        # ST_COMPUTE_SCORES


@dataclass(frozen=True, slots=True)
class GameConfig:
    players: int = 4
    #: Advanced variant: five extra City Plans in stacks 1 and 2, plus roundabouts.
    advanced: bool = False
    #: Expert variant: each player gets their own three cards, takes the number
    #: from one and the effect from another, and passes the third on.
    expert: bool = False
    #: Whether a one-player game uses the real solo rules (solo marker card,
    #: deck-out end condition, six ordered card pairs).  Turning it off gives
    #: standard three-stack rules with a single seat.
    #:
    #: Both one-seat modes exist so the engine covers what BGA offers; **neither
    #: is a training configuration.**  With one seat the scoring rules change --
    #: ``TEMP_SOLO_SCORE`` replaces the 7/4/1 ranking and every City Plan pays
    #: its first-place value -- so data captured there describes a different
    #: game.  Training runs 2-4 seats; see ``SELF_PLAY_PLAN.md``.
    solo_rules: bool = True

    @property
    def solo(self) -> bool:
        """Real solo mode, with the solo card and its scoring."""
        return self.players == 1 and self.solo_rules

    @property
    def single_player(self) -> bool:
        """One seat, whether or not the solo rules are switched on."""
        return self.players == 1

    @property
    def standard(self) -> bool:
        """``Globals::isStandard`` -- shared stacks of two cards each."""
        return not self.expert and not self.solo

    @property
    def stack_groups(self) -> int:
        """How many independent sets of three stacks exist."""
        return self.players if self.expert else 1

    @property
    def choice_slots(self) -> int:
        """3 shared stacks in standard mode, 6 ordered card pairs otherwise."""
        return 3 if self.standard else len(codec.EXPERT_PAIRS)


@dataclass
class TurnCtx:
    """Scratch state for the private turn currently being resolved."""

    slot: Optional[int] = None
    number: Optional[int] = None
    effect: Optional[Effect] = None
    last_house: Optional[Pos] = None
    built_roundabout: bool = False
    #: Set once the player has declined the roundabout this turn.  BGA's state
    #: machine sends ``pass`` from ST_ROUNDABOUT straight back to
    #: ST_CHOOSE_CARDS, where the offer is live again, so a bot can oscillate
    #: between the two forever without touching the sheet.  A human would not,
    #: but greedy does (1489 such cycles in one measured game) and MCTS would
    #: grow an infinite no-op branch.  Declining is therefore made sticky, which
    #: changes nothing a player could want: opening the prompt and passing has no
    #: effect on the game.
    roundabout_declined: bool = False
    refused: bool = False
    plan_slot: Optional[int] = None
    pending_sizes: tuple[int, ...] = ()
    chosen_estates: tuple[Estate, ...] = ()

    def copy(self) -> "TurnCtx":
        return TurnCtx(
            slot=self.slot,
            number=self.number,
            effect=self.effect,
            last_house=self.last_house,
            built_roundabout=self.built_roundabout,
            roundabout_declined=self.roundabout_declined,
            refused=self.refused,
            plan_slot=self.plan_slot,
            pending_sizes=self.pending_sizes,
            chosen_estates=self.chosen_estates,
        )


class IllegalAction(ValueError):
    pass


#: A replacement for :meth:`GameState._draw`, at the raw-draw level.
DrawFn = Callable[[], int]


@dataclass(frozen=True, slots=True)
class BoundaryOutcome:
    """One immediate chance outcome at a turn boundary -- ``SEARCH_SPEC.md`` §6.3.

    ``draws`` is the ordered sequence of **raw** draws the boundary made, which
    is deliberately not "the three cards on the table":

    - an ordinary boundary draws **three**;
    - a **queued reshuffle** draws **six, in two batches with a discard cycle
      between them**, so both the aside pair and the number pair on show
      afterwards are freshly drawn;
    - a solo card is a draw that resolves and forces another, so it appears here
      like any other card and replays as one.

    ⚠ **Every card in ``draws`` is one the boundary makes public**, and the
    order of the deck it leaves behind is not recorded. That is the
    non-anticipativity rule of §7.3, held structurally rather than by
    convention: an outcome cannot leak the future because it never contains it.
    """

    draws: tuple[int, ...]
    #: Whether the deck was reformed from the discard during this boundary.
    #: Determined by the afterstate, not by the draw -- see ``_reveal_step``.
    reformed: bool = False


@dataclass
class GameState:
    config: GameConfig
    sheets: list[Sheet]
    #: Snapshot of every sheet as it stood at the start of the current turn.
    #: This, not ``sheets``, is what other players are allowed to see.
    public_sheets: list[Sheet]
    deck: list[int]
    deck_pos: int
    discard: list[int]
    #: ``stack_new[g][i]`` -- the card on top of the stack, showing its NUMBER.
    #: Its own effect is printed in the corners of that face, so it is public and
    #: is what the stack will offer next turn.
    stack_new: list[list[Optional[int]]]
    #: ``stack_old[g][i]`` -- standard mode only: the card flipped aside beside
    #: the stack, showing its EFFECT.  Empty in expert and solo, where the single
    #: card in each stack serves both roles.
    stack_old: list[list[Optional[int]]]
    #: Expert mode: the card the previous player passed to each player.
    expert_pending: list[Optional[int]]
    #: The three City Plans in play, as plan ids into ``plans.PLANS``.
    plan_ids: tuple[int, int, int]
    #: ``plan_turns[slot][player]`` -- the turn on which that player completed it.
    plan_turns: list[dict[int, int]]
    turn: int
    actor: int
    phase: Phase
    ctx: TurnCtx
    #: Per-player combination slot chosen this turn (expert card passing needs it).
    turn_choice: list[Optional[int]]
    reshuffle_next_turn: bool
    #: How each player voted at ``ASK_RESHUFFLE`` this turn, by seat.  The
    #: table-wide :attr:`reshuffle_next_turn` is the OR of these and is
    #: **private until the turn resolves**: turns are serialised, so a later
    #: actor reading the aggregate would learn that an earlier one voted yes --
    #: and in the concurrent game nobody knows that, or even that the earlier
    #: player completed a plan.  An information-set-safe reader asks this.
    reshuffle_votes: dict[int, bool]
    rng: random.Random
    solo_card_drawn: bool = False
    #: Whether :meth:`prepare_turn_boundary` has run and no reveal has followed.
    #:
    #: ⚠ **An explicit lifecycle flag, not an inference.** This was read off
    #: "every ``stack_new`` slot is ``None``", which is the shape ``_discard_step``
    #: leaves but is an incidental property of the representation rather than a
    #: guarantee about where the state is in its life. It also could not refuse a
    #: second ``prepare``, which would increment the turn twice and discard the
    #: pair that was just promoted.
    boundary_prepared: bool = False

    # ──────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────
    @classmethod
    def new(
        cls,
        seed: Optional[int] = None,
        config: Optional[GameConfig] = None,
    ) -> "GameState":
        """``welcometo::setupNewGame``."""
        config = config or GameConfig()
        if config.players < 1:
            raise ValueError("need at least one player")
        if config.expert and config.players < 2:
            raise ValueError("expert rules need at least two players")

        rng = random.Random(seed)
        deck = list(range(NUM_BASE_CARDS))
        rng.shuffle(deck)

        plan_ids = tuple(
            rng.choice(available_plan_ids(stack, config.advanced)) for stack in (1, 2, 3)
        )

        sheets = [Sheet.new() for _ in range(config.players)]
        groups = config.stack_groups
        state = cls(
            config=config,
            sheets=sheets,
            public_sheets=[s.copy() for s in sheets],
            deck=deck,
            deck_pos=0,
            discard=[],
            stack_new=[[None, None, None] for _ in range(groups)],
            stack_old=[[None, None, None] for _ in range(groups)] if config.standard else [],
            expert_pending=[None] * config.players,
            plan_ids=plan_ids,  # type: ignore[arg-type]
            plan_turns=[{}, {}, {}],
            turn=1,
            actor=0,
            phase=Phase.CHOOSE_CARDS,
            ctx=TurnCtx(),
            turn_choice=[None] * config.players,
            reshuffle_next_turn=False,
            reshuffle_votes={},
            rng=rng,
        )

        if config.solo:
            state._solo_setup()
        # ``ConstructionCards::setupNewGame``: standard mode seeds one card in
        # each stack so that the first ``stNewTurn`` has something to flip.
        if config.standard:
            for i in range(3):
                state.stack_new[0][i] = state._draw_playable()
        elif config.expert:
            for p in range(config.players):
                state.expert_pending[p] = state._draw_playable()

        state._begin_turn()
        return state

    def _solo_setup(self) -> None:
        """``ConstructionCards::soloSetupNewGame``.

        The solo card is shuffled in and then pushed into the bottom half of the
        deck if it landed in the top half, so the solo player always gets a
        decent run before it fires.
        """
        self.deck.append(SOLO_CARD_ID)
        self.rng.shuffle(self.deck)
        index = self.deck.index(SOLO_CARD_ID)
        limit = len(self.deck) - 1 - SOLO_DECK_MIDDLE
        if index <= limit:
            self.deck.pop(index)
            self.deck.insert(min(index + SOLO_DECK_MIDDLE, len(self.deck)), SOLO_CARD_ID)

    def copy(self) -> "GameState":
        rng = random.Random()
        rng.setstate(self.rng.getstate())
        return GameState(
            config=self.config,
            sheets=[s.copy() for s in self.sheets],
            public_sheets=[s.copy() for s in self.public_sheets],
            deck=list(self.deck),
            deck_pos=self.deck_pos,
            discard=list(self.discard),
            stack_new=[list(g) for g in self.stack_new],
            stack_old=[list(g) for g in self.stack_old],
            expert_pending=list(self.expert_pending),
            plan_ids=self.plan_ids,
            plan_turns=[dict(d) for d in self.plan_turns],
            turn=self.turn,
            actor=self.actor,
            phase=self.phase,
            ctx=self.ctx.copy(),
            turn_choice=list(self.turn_choice),
            reshuffle_next_turn=self.reshuffle_next_turn,
            reshuffle_votes=dict(self.reshuffle_votes),
            rng=rng,
            solo_card_drawn=self.solo_card_drawn,
            boundary_prepared=self.boundary_prepared,
        )

    # ──────────────────────────────────────────────────────────────────
    # Deck
    # ──────────────────────────────────────────────────────────────────
    def _reform_deck(self) -> None:
        """``Pieces::reformDeckFromDiscard`` -- discard back into the deck, shuffled."""
        remaining = self.deck[self.deck_pos:]
        self.deck = remaining + self.discard
        self.discard = []
        self.rng.shuffle(self.deck)
        self.deck_pos = 0

    def _draw(self) -> int:
        if self.deck_pos >= len(self.deck):
            self._reform_deck()
        if self.deck_pos >= len(self.deck):
            raise RuntimeError("construction deck and discard are both empty")
        card = self.deck[self.deck_pos]
        self.deck_pos += 1
        return card

    def _replay_draw(self, card: int) -> int:
        """:meth:`_draw`, except *which* card comes off the top is dictated.

        The same reform-if-empty rule, the same bookkeeping; only the choice of
        card is replaced.  ``card`` is swapped to the top of the undrawn region
        and drawn from there, so the deck's **composition** stays exact while the
        order of what is left stays arbitrary -- which is correct, because that
        order is hidden and every simulation re-determinizes it anyway
        (``SEARCH_SPEC.md`` §7.3: a sampled outcome must not carry the future
        deck order).
        """
        if self.deck_pos >= len(self.deck):
            self._reform_deck()
        try:
            at = self.deck.index(card, self.deck_pos)
        except ValueError:
            raise IllegalAction(
                f"card {card} is not in the undrawn deck; this outcome does not "
                "belong to this boundary"
            ) from None
        self.deck[self.deck_pos], self.deck[at] = (
            self.deck[at],
            self.deck[self.deck_pos],
        )
        self.deck_pos += 1
        return card

    def _draw_playable(self, draw: Optional[DrawFn] = None) -> int:
        """Draw, resolving the solo card if it turns up (``ConstructionCards::drawAux``).

        ``draw`` replaces :meth:`_draw` for the duration -- at the *raw* draw
        level, deliberately, so that a solo card and the extra draw it forces are
        recorded and replayed like any other card rather than being a special
        case a second implementation has to know about.
        """
        take = draw or self._draw
        card = take()
        if card == SOLO_CARD_ID:
            self._on_solo_card()
            card = take()
        return card

    def _on_solo_card(self) -> None:
        """``ConstructionCards::soloCardDrawn`` -- the ghost claims every plan.

        Each plan is validated for a mock player on the *previous* turn, so the
        solo player can never take a first-place plan value afterwards.
        """
        self.solo_card_drawn = True
        for slot in range(3):
            self.plan_turns[slot].setdefault(SOLO_MOCK_PLAYER, self.turn - 1)

    @property
    def deck_remaining(self) -> int:
        return len(self.deck) - self.deck_pos

    # ──────────────────────────────────────────────────────────────────
    # Turn boundaries
    # ──────────────────────────────────────────────────────────────────
    def _begin_turn(self, draw: Optional[DrawFn] = None) -> None:
        """``stNewTurn`` -- discard, optionally reshuffle, draw, then hand over.

        Kept whole for ``setupNewGame``, which begins a turn without *ending*
        one: no turn increment and no end-of-game test.  A real boundary goes
        through :meth:`prepare_turn_boundary` instead.
        """
        self._discard_step()
        self._reveal_step(draw)
        self._open_turn()

    def _reveal_step(self, draw: Optional[DrawFn] = None) -> None:
        """The part of a boundary that reveals cards -- **all four cases of §6.3**.

        ⚠ Which case fires is **not** chance: a queued reshuffle is
        ``reshuffle_next_turn``, and an exact-empty reform is
        ``deck_remaining == 0``, both of which are settled by the prepared state
        before a card is seen.  Only *which cards* is chance.  That is what makes
        :meth:`sample_boundary_outcome` and :meth:`apply_boundary_outcome` a
        clean pair: they run this same code and differ only in where a card
        comes from.
        """
        if self.reshuffle_next_turn:
            self._reshuffle_decks(draw)
            self.reshuffle_next_turn = False
        self._draw_step(draw)

    def _open_turn(self) -> None:
        """The deterministic tail: hand the turn to seat 0 and settle."""
        self.boundary_prepared = False
        self.actor = 0
        self.ctx = TurnCtx()
        self.reshuffle_votes = {}
        self.turn_choice = [None] * self.config.players
        self.phase = Phase.CHOOSE_CARDS
        self.public_sheets = [s.copy() for s in self.sheets]
        self._settle()

    def _discard_step(self) -> None:
        """``ConstructionCards::discardAux``."""
        if self.config.standard:
            for i in range(3):
                old = self.stack_old[0][i]
                if old is not None:
                    self.discard.append(old)
                self.stack_old[0][i] = self.stack_new[0][i]
                self.stack_new[0][i] = None
        else:
            for group in self.stack_new:
                for i in range(3):
                    if group[i] is not None:
                        self.discard.append(group[i])
                    group[i] = None

    def _draw_step(self, draw: Optional[DrawFn] = None) -> None:
        """``ConstructionCards::drawAux``."""
        if self.config.expert:
            for p in range(self.config.players):
                for i in range(3):
                    if i == 0 and self.expert_pending[p] is not None:
                        self.stack_new[p][0] = self.expert_pending[p]
                        self.expert_pending[p] = None
                    else:
                        self.stack_new[p][i] = self._draw_playable(draw)
        else:
            for i in range(3):
                self.stack_new[0][i] = self._draw_playable(draw)

    def _reshuffle_decks(self, draw: Optional[DrawFn] = None) -> None:
        """``ConstructionCards::reshuffle`` -- offered once, after the first plan.

        In standard mode BGA reforms the deck and then burns a full draw/discard
        cycle, so the pair on show after a reshuffle is made of two freshly drawn
        cards.  That is reproduced exactly.  In expert mode BGA re-drafts a card
        per player on top of the one already passed; this engine only tops up a
        player whose pending slot is empty, which is the sane reading of an
        otherwise ambiguous corner of the original.
        """
        self._reform_deck()
        if self.config.standard:
            for i in range(3):
                self.stack_new[0][i] = self._draw_playable(draw)
            self._discard_step()
        elif self.config.expert:
            for p in range(self.config.players):
                if self.expert_pending[p] is None:
                    self.expert_pending[p] = self._draw_playable(draw)

    def _end_turn(self) -> None:
        """``stApplyTurn`` -- the three-part boundary, run straight through."""
        if not self.prepare_turn_boundary():
            return
        self._reveal_step()
        self._open_turn()

    # ──────────────────────────────────────────────────────────────────
    # The boundary, in three parts -- SEARCH_SPEC.md §6.3
    # ──────────────────────────────────────────────────────────────────
    def prepare_turn_boundary(self) -> bool:
        """Everything a boundary does **before a card is revealed**, in place.

        Returns ``True`` if a reveal follows and ``False`` if the game ended
        here, which is the fourth case of §6.3 and the one a search that assumes
        "a boundary reveals cards" gets wrong.

        What this settles, all of it deterministic and all of it observable:

        - expert card passing, and the turn counter;
        - ``isEndOfGame``, tested on the **pre-discard** deck, which matters in
          solo where the clause reads ``deck_remaining``;
        - ``discardAux``: the three aside cards go to the discard and the three
          **number** cards are promoted into the aside slots. So *this turn's
          numbers become next turn's effects* -- which is the whole reason
          ``next_effects`` is a certainty and not a posterior (§6.2), and why the
          cards on the table are never part of the material being reshuffled.

        Afterwards ``self`` is the boundary **afterstate**: ``stack_new`` empty,
        the effects for the coming turn already known and public, and exactly one
        thing left undecided -- which cards come off the deck.
        """
        if self.boundary_prepared:
            raise IllegalAction(
                "this boundary is already prepared; preparing twice would "
                "increment the turn again and discard the pair just promoted"
            )
        if self.config.expert:
            self._pass_unused_cards()

        self.turn += 1
        if self._is_end_of_game():
            self.phase = Phase.GAME_OVER
            self.public_sheets = [s.copy() for s in self.sheets]
            return False

        self._discard_step()
        self.boundary_prepared = True
        return True

    def _is_boundary_afterstate(self) -> bool:
        return self.boundary_prepared and not self.is_terminal

    def sample_boundary_outcome(self, rng: random.Random) -> "BoundaryOutcome":
        """One immediate outcome of this boundary.  **Does not modify ``self``.**

        ``rng`` re-permutes the undrawn deck before the draw, so repeated calls
        on the *same* afterstate give independent outcomes -- which is what a
        chance node needs, and what reading the top of an already-determinized
        deck would not give.  The discard is not permuted here because it does
        not need to be: it only ever enters play through ``_reform_deck``, which
        shuffles it with the probe's own generator.

        Runs the engine's real reveal path, so the ordinary draw, the
        exact-empty reform, the queued reshuffle and any solo card are handled
        because they are *the same code*, not because they were re-derived.
        """
        if not self._is_boundary_afterstate():
            raise IllegalAction(
                "sample_boundary_outcome needs a prepared boundary afterstate; "
                "call prepare_turn_boundary() first"
            )
        probe = self.redeterminize(rng)
        drawn: list[int] = []
        # ``_reform_deck`` has exactly two call sites, and both are visible from
        # here: ``_reshuffle_decks`` always reforms, and ``_draw`` reforms when
        # the undrawn region is empty as it is entered.  Checking both is exact,
        # where inferring one from how ``deck_pos`` moved is not -- a reform
        # resets the cursor, so the arithmetic stops meaning anything.
        reformed = probe.reshuffle_next_turn

        def record() -> int:
            nonlocal reformed
            if probe.deck_pos >= len(probe.deck):
                reformed = True
            card = probe._draw()
            drawn.append(card)
            return card

        probe._reveal_step(record)
        return BoundaryOutcome(draws=tuple(drawn), reformed=reformed)

    def apply_boundary_outcome(self, outcome: "BoundaryOutcome") -> None:
        """Apply ``outcome`` to this afterstate, in place, and open the turn.

        Deterministic: the same reveal path, with each draw served from
        ``outcome`` instead of the deck. Raises :class:`IllegalAction` if the
        outcome does not fit this boundary -- a wrong card, or the wrong number
        of them -- rather than silently producing a state no deal could reach.

        ⚠ **Transactional.** The replay runs on a copy and is adopted only once
        the outcome has been validated *whole*. It used to validate while
        applying, which meant a rejected outcome left the receiver mutated: too
        few cards raised part-way through the reveal, and too many were caught
        only after it had finished. Measured on a rejected outcome with one
        extra card: ``deck_remaining`` 66 -> 63 and three cards on the table.
        "Raises, and also destroys the state you called it on" is not a contract
        worth having on a public method.
        """
        if not self._is_boundary_afterstate():
            raise IllegalAction(
                "apply_boundary_outcome needs a prepared boundary afterstate; "
                "call prepare_turn_boundary() first"
            )
        staged = self.copy()
        pending = iter(outcome.draws)

        def replay() -> int:
            try:
                card = next(pending)
            except StopIteration:
                raise IllegalAction(
                    "the outcome ran out of cards; it does not belong to this "
                    "boundary"
                ) from None
            return staged._replay_draw(card)

        staged._reveal_step(replay)
        if next(pending, None) is not None:
            raise IllegalAction("the outcome has cards this boundary did not draw")
        staged._scramble_undrawn()
        staged._open_turn()
        self._adopt(staged)

    def _scramble_undrawn(self) -> None:
        """Shuffle what is left of the deck, so its order carries no artefact.

        ``_replay_draw`` swaps each named card to the top of the undrawn region,
        which leaves the remainder in an order determined by *which cards the
        outcome named* rather than by any deal. The composition is exact either
        way and the order is hidden, so nothing could read it correctly -- but
        "nothing reads it" was a docstring, not a property. Shuffling makes it
        one, and costs a shuffle of the undrawn tail per boundary.
        """
        tail = self.deck[self.deck_pos :]
        self.rng.shuffle(tail)
        self.deck = self.deck[: self.deck_pos] + tail

    def _adopt(self, other: "GameState") -> None:
        """Become ``other``, in place.  The commit half of a transaction."""
        self.sheets = other.sheets
        self.public_sheets = other.public_sheets
        self.deck = other.deck
        self.deck_pos = other.deck_pos
        self.discard = other.discard
        self.stack_new = other.stack_new
        self.stack_old = other.stack_old
        self.expert_pending = other.expert_pending
        self.plan_ids = other.plan_ids
        self.plan_turns = other.plan_turns
        self.turn = other.turn
        self.actor = other.actor
        self.phase = other.phase
        self.ctx = other.ctx
        self.turn_choice = other.turn_choice
        self.reshuffle_next_turn = other.reshuffle_next_turn
        self.reshuffle_votes = other.reshuffle_votes
        self.rng = other.rng
        self.solo_card_drawn = other.solo_card_drawn
        self.boundary_prepared = other.boundary_prepared

    def _pass_unused_cards(self) -> None:
        """``Player::giveThirdCardToNextPlayer``.

        A player who took a permit refusal never chose a pair, so all three of
        their cards are simply discarded.
        """
        moved: list[tuple[int, int]] = []
        for p in range(self.config.players):
            slot = self.turn_choice[p]
            if slot is None:
                continue
            i, j = codec.EXPERT_PAIRS[slot]
            spare = ({0, 1, 2} - {i, j}).pop()
            card = self.stack_new[p][spare]
            if card is None:
                continue
            self.stack_new[p][spare] = None
            moved.append(((p + 1) % self.config.players, card))
        for target, card in moved:
            self.expert_pending[target] = card

    def _is_end_of_game(self) -> bool:
        """``EndOfGameTrait::isEndOfGame``."""
        for p, sheet in enumerate(self.sheets):
            if not sheet.has_free_box():
                return True
            if all(p in self.plan_turns[slot] for slot in range(3)):
                return True
            if sheet.permits >= PERMIT_BOXES:
                return True
        if self.config.solo and self.deck_remaining == 0:
            return True
        return False

    def end_of_game_reason(self) -> Optional[str]:
        """Which ``isEndOfGame`` clause fired, for logging."""
        for p, sheet in enumerate(self.sheets):
            if not sheet.has_free_box():
                return f"player {p} filled every house"
            if all(p in self.plan_turns[slot] for slot in range(3)):
                return f"player {p} completed all three plans"
            if sheet.permits >= PERMIT_BOXES:
                return f"player {p} took a third permit refusal"
        if self.config.solo and self.deck_remaining == 0:
            return "deck exhausted (solo)"
        return None

    # ──────────────────────────────────────────────────────────────────
    # Card combinations
    # ──────────────────────────────────────────────────────────────────
    def _group_of(self, player: int) -> int:
        return player if self.config.expert else 0

    def combination(self, slot: int, player: Optional[int] = None) -> tuple[int, Effect]:
        """``ConstructionCards::getCombination`` -- the ``(number, effect)`` on offer."""
        player = self.actor if player is None else player
        g = self._group_of(player)
        if self.config.standard:
            # The number in play sits on top of the stack; the effect in play is
            # on the card that was flipped aside beside it last turn.
            number_card = self.stack_new[g][slot]
            effect_card = self.stack_old[g][slot]
            if number_card is None or effect_card is None:
                raise RuntimeError("stacks are not populated yet")
            return CARD_TABLE[number_card][0], CARD_TABLE[effect_card][1]
        i, j = codec.EXPERT_PAIRS[slot]
        number_card = self.stack_new[g][i]
        effect_card = self.stack_new[g][j]
        if number_card is None or effect_card is None:
            raise RuntimeError("stacks are not populated yet")
        return CARD_TABLE[number_card][0], CARD_TABLE[effect_card][1]

    def visible_cards(self, player: Optional[int] = None) -> list[tuple[Optional[int], Optional[Effect]]]:
        """The ``(number, effect)`` pair the player can read off each stack.

        In standard mode the number comes from the card on top of the stack and
        the effect from the card flipped aside beside it.  In expert and solo mode
        there is one card per stack and it supplies both faces.

        This is the pair *in play*; see :meth:`next_effects` for the effect each
        stack will offer next turn, which is already public.
        """
        player = self.actor if player is None else player
        out: list[tuple[Optional[int], Optional[Effect]]] = []
        for slot in range(3):
            try:
                number, effect = self.combination_faces(slot, player)
            except RuntimeError:
                number, effect = None, None
            out.append((number, effect))
        return out

    def combination_faces(
        self, slot: int, player: Optional[int] = None
    ) -> tuple[Optional[int], Optional[Effect]]:
        """The visible faces of stack ``slot``, without the expert pairing."""
        player = self.actor if player is None else player
        g = self._group_of(player)
        top = self.stack_new[g][slot]
        if not self.config.standard:
            if top is None:
                return None, None
            return CARD_TABLE[top][0], CARD_TABLE[top][1]
        aside = self.stack_old[g][slot]
        return (
            None if top is None else CARD_TABLE[top][0],
            None if aside is None else CARD_TABLE[aside][1],
        )

    def next_effects(self, player: Optional[int] = None) -> list[Optional[Effect]]:
        """The effect each stack will offer NEXT turn — known with certainty now.

        A construction card prints the effect from its own back in two corners of
        its *number* face (``.top-right-corner`` / ``.bottom-left-corner`` in
        ``wtoCards.scss``, keyed on ``data-action``).  So the card sitting on top
        of a stack, showing a number, already tells you which effect it will
        contribute when it is flipped aside next turn.  This is not a posterior,
        it is a fact, and a model not told it is playing blindfolded.

        ``None`` per slot in expert and solo mode, where all three cards are
        replaced every turn and nothing carries over.
        """
        if not self.config.standard:
            return [None, None, None]
        g = self._group_of(self.actor if player is None else player)
        return [None if c is None else CARD_TABLE[c][1] for c in self.stack_new[g]]

    def table_cards(self, player: Optional[int] = None) -> list[Optional[int]]:
        """Every card on the table.  All of them are fully identified.

        Both cards of a standard-mode stack have both faces public: the one
        flipped aside shows its effect and showed its number last turn, and the
        one on top shows its number and prints its own effect in the corners.
        """
        player = self.actor if player is None else player
        g = self._group_of(player)
        cards = list(self.stack_new[g])
        if self.config.standard:
            cards += list(self.stack_old[g])
        return cards

    def numbers_for(self, number: int, effect: Effect) -> list[int]:
        """``Player::getAvailableNumbersOfCombination`` -- candidate numbers, in codec order."""
        if effect is not Effect.TEMP:
            return [number]
        out = [number]
        for delta in TEMP_DELTAS[1:]:
            n = number + delta
            if MIN_NUMBER <= n <= MAX_NUMBER:
                out.append(n)
        return out

    def _writable(self, number: int, effect: Effect, sheet: Sheet) -> dict[int, list[Pos]]:
        result: dict[int, list[Pos]] = {}
        for n in self.numbers_for(number, effect):
            spots = sheet.available_locations(n)
            if spots:
                result[n] = spots
        return result

    def playable_slots(self, player: Optional[int] = None) -> list[int]:
        """``Player::getAvailableStacks`` -- combinations with somewhere to write."""
        player = self.actor if player is None else player
        sheet = self.sheets[player]
        out = []
        for slot in range(self.config.choice_slots):
            number, effect = self.combination(slot, player)
            if self._writable(number, effect, sheet):
                out.append(slot)
        return out

    # ──────────────────────────────────────────────────────────────────
    # Legal actions
    # ──────────────────────────────────────────────────────────────────
    @property
    def current_player(self) -> int:
        return self.actor

    @property
    def is_terminal(self) -> bool:
        return self.phase is Phase.GAME_OVER

    def legal_actions(self) -> list[int]:
        phase = self.phase
        if phase is Phase.GAME_OVER:
            return []
        sheet = self.sheets[self.actor]
        ctx = self.ctx

        if phase is Phase.CHOOSE_CARDS:
            actions = [codec.choose_stack(s) for s in self.playable_slots()]
            if not actions and sheet.can_take_permit():
                actions.append(codec.A_PERMIT_REFUSAL)
            if (
                self.config.advanced
                and ctx.last_house is None
                and not ctx.roundabout_declined
                and sheet.can_build_roundabout()
                and sheet.has_free_box()
            ):
                actions.append(codec.A_ROUNDABOUT_OPEN)
            return actions

        if phase is Phase.ROUNDABOUT_PLACE:
            actions = [
                codec.roundabout_pos(x, y) for x, y in sheet.available_locations(None)
            ]
            actions.append(codec.A_PASS_ROUNDABOUT)
            return actions

        if phase is Phase.WRITE_NUMBER:
            assert ctx.number is not None and ctx.effect is not None
            actions = []
            candidates = self.numbers_for(ctx.number, ctx.effect)
            for n in candidates:
                delta_slot = TEMP_DELTAS.index(n - ctx.number)
                for x, y in sheet.available_locations(n):
                    actions.append(codec.write(delta_slot, x, y))
            # ``argWriteNumber``: never force a player to spend the temp agency
            # just to have somewhere to write -- a refusal stays open when the
            # combination's own number has nowhere to go.
            if not sheet.available_locations(ctx.number) and sheet.can_take_permit():
                actions.append(codec.A_PERMIT_REFUSAL)
            return actions

        if phase is Phase.ACTION_SURVEYOR:
            return [codec.surveyor_fence(x, j) for x, j in sheet.surveyor_zones()] + [
                codec.A_PASS_SURVEYOR
            ]

        if phase is Phase.ACTION_ESTATE:
            return [codec.estate_row(r) for r in sheet.estate_rows()] + [
                codec.A_PASS_ESTATE
            ]

        if phase is Phase.ACTION_PARK:
            return [codec.park_street(x) for x in self._park_streets()] + [
                codec.A_PASS_PARK
            ]

        if phase is Phase.ACTION_POOL:
            return [codec.A_POOL_BUILD, codec.A_PASS_POOL]

        if phase is Phase.ACTION_BIS:
            actions = [
                codec.bis(x, y, side) for x, y, _, side in sheet.bis_candidates()
            ]
            actions.append(codec.A_PASS_BIS)
            return actions

        if phase is Phase.CHOOSE_PLAN:
            return [codec.choose_plan(s) for s in self.scorable_plan_slots()] + [
                codec.A_PASS_PLAN
            ]

        if phase is Phase.VALIDATE_PLAN:
            size = ctx.pending_sizes[0]
            return [
                codec.validate_estate(x, start)
                for x, start, _ in estates_matching_size(sheet, size, ctx.chosen_estates)
            ]

        if phase is Phase.ASK_RESHUFFLE:
            return [codec.A_RESHUFFLE_YES, codec.A_RESHUFFLE_NO]

        raise AssertionError(f"unhandled phase {phase}")

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(codec.NUM_ACTIONS, dtype=bool)
        idx = self.legal_actions()
        if idx:
            mask[np.asarray(idx, dtype=np.int64)] = True
        return mask

    def _park_streets(self) -> list[int]:
        """``Actions/Park::getAvailableZones`` -- same street as the house just written."""
        if self.ctx.last_house is None:
            return []
        street = self.ctx.last_house[0]
        return [x for x in self.sheets[self.actor].park_streets() if x == street]

    def _pool_available(self) -> bool:
        if self.ctx.last_house is None:
            return False
        return self.sheets[self.actor].can_build_pool_at(self.ctx.last_house)

    def scorable_plan_slots(self) -> list[int]:
        """``Player::getScorablePlans``."""
        sheet = self.sheets[self.actor]
        out = []
        for slot, plan_id in enumerate(self.plan_ids):
            if self.actor in self.plan_turns[slot]:
                continue
            if can_be_scored(PLANS[plan_id], sheet):
                out.append(slot)
        return out

    def _may_ask_reshuffle(self) -> bool:
        """``stAskReshuffle`` -- only before any plan has been completed on an earlier turn."""
        # Real solo skips it (BGA does).  A one-seat standard-rules game keeps
        # it: the decision itself -- does this deck suit my sheet? -- is decided
        # on your own sheet, even though winning the right to make it is a race.
        if self.config.solo:
            return False
        return not any(
            t < self.turn for turns in self.plan_turns for t in turns.values()
        )

    # ──────────────────────────────────────────────────────────────────
    # Stepping
    # ──────────────────────────────────────────────────────────────────
    def step(self, action: int) -> "GameState":
        """Apply ``action`` to a copy and return it (states are treated as values)."""
        nxt = self.copy()
        nxt.apply(action)
        return nxt

    def apply(self, action: int) -> None:
        """Apply ``action`` in place."""
        legal = self.legal_actions()
        if action not in legal:
            raise IllegalAction(
                f"{codec.describe(action)} is not legal in {self.phase.name} "
                f"for player {self.actor}"
            )
        self._dispatch(action)
        self._settle()

    def _dispatch(self, action: int) -> None:
        phase = self.phase
        sheet = self.sheets[self.actor]
        ctx = self.ctx

        if phase is Phase.CHOOSE_CARDS:
            if action == codec.A_PERMIT_REFUSAL:
                sheet.permits = min(sheet.permits + 1, PERMIT_BOXES)
                ctx.refused = True
                self.phase = Phase.CHOOSE_PLAN
                return
            if action == codec.A_ROUNDABOUT_OPEN:
                self.phase = Phase.ROUNDABOUT_PLACE
                return
            slot = codec.decode_stack(action)
            ctx.slot = slot
            ctx.number, ctx.effect = self.combination(slot)
            self.turn_choice[self.actor] = slot
            self.phase = Phase.WRITE_NUMBER
            return

        if phase is Phase.ROUNDABOUT_PLACE:
            if action == codec.A_PASS_ROUNDABOUT:
                ctx.roundabout_declined = True
            else:
                pos = codec.decode_roundabout_pos(action)
                sheet.build_roundabout(pos, self.turn)
                ctx.built_roundabout = True
                ctx.last_house = pos
            self.phase = Phase.CHOOSE_CARDS
            return

        if phase is Phase.WRITE_NUMBER:
            if action == codec.A_PERMIT_REFUSAL:
                sheet.permits = min(sheet.permits + 1, PERMIT_BOXES)
                ctx.refused = True
                self.phase = Phase.CHOOSE_PLAN
                return
            delta_slot, x, y = codec.decode_write(action)
            assert ctx.number is not None and ctx.effect is not None
            number = ctx.number + TEMP_DELTAS[delta_slot]
            sheet.write(number, (x, y), self.turn)
            ctx.last_house = (x, y)
            self.phase = _EFFECT_PHASE[ctx.effect]
            if ctx.effect is Effect.TEMP:
                # ``stActionTemp`` crosses a box off with no decision to make.
                sheet.temps = min(sheet.temps + 1, TEMP_BOXES)
                self.phase = Phase.CHOOSE_PLAN
            return

        if phase is Phase.ACTION_SURVEYOR:
            if action != codec.A_PASS_SURVEYOR:
                x, j = codec.decode_surveyor_fence(action)
                sheet.fences[x][j] = True
            self.phase = Phase.CHOOSE_PLAN
            return

        if phase is Phase.ACTION_ESTATE:
            if action != codec.A_PASS_ESTATE:
                row = codec.decode_estate_row(action)
                sheet.estate_marks[row] += 1
            self.phase = Phase.CHOOSE_PLAN
            return

        if phase is Phase.ACTION_PARK:
            if action != codec.A_PASS_PARK:
                x = codec.decode_park_street(action)
                sheet.parks[x] += 1
            self.phase = Phase.CHOOSE_PLAN
            return

        if phase is Phase.ACTION_POOL:
            if action == codec.A_POOL_BUILD:
                assert ctx.last_house is not None
                sheet.pools[ctx.last_house[0]] += 1
            self.phase = Phase.CHOOSE_PLAN
            return

        if phase is Phase.ACTION_BIS:
            if action != codec.A_PASS_BIS:
                x, y, side = codec.decode_bis(action)
                number = sheet.bis_number_at(x, y, side)
                assert number is not None
                sheet.write(number, (x, y), self.turn, is_bis=True)
                ctx.last_house = (x, y)
                sheet.bis_marks = min(sheet.bis_marks + 1, BIS_BOXES)
            self.phase = Phase.CHOOSE_PLAN
            return

        if phase is Phase.CHOOSE_PLAN:
            if action == codec.A_PASS_PLAN:
                self._finish_player_turn()
                return
            slot = codec.decode_plan(action)
            ctx.plan_slot = slot
            plan = PLANS[self.plan_ids[slot]]
            ctx.chosen_estates = ()
            if plan.is_automatic:
                self._validate_plan(plan, slot)
            else:
                ctx.pending_sizes = plan.required_sizes
                self.phase = Phase.VALIDATE_PLAN
            return

        if phase is Phase.VALIDATE_PLAN:
            x, start = codec.decode_validate_estate(action)
            size = ctx.pending_sizes[0]
            ctx.chosen_estates = ctx.chosen_estates + ((x, start, size),)
            ctx.pending_sizes = ctx.pending_sizes[1:]
            if not ctx.pending_sizes:
                assert ctx.plan_slot is not None
                self._validate_plan(PLANS[self.plan_ids[ctx.plan_slot]], ctx.plan_slot)
            return

        if phase is Phase.ASK_RESHUFFLE:
            self.reshuffle_votes[self.actor] = action == codec.A_RESHUFFLE_YES
            if action == codec.A_RESHUFFLE_YES:
                self.reshuffle_next_turn = True
            self.phase = Phase.CHOOSE_PLAN
            return

        raise AssertionError(f"unhandled phase {phase}")

    def _validate_plan(self, plan: Plan, slot: int) -> None:
        """``AbstractPlan::validate`` -- consume houses, record the turn."""
        sheet = self.sheets[self.actor]
        sheet.mark_top_fences(validation_cells(plan, sheet, self.ctx.chosen_estates))
        self.plan_turns[slot][self.actor] = self.turn
        self.ctx.plan_slot = None
        self.ctx.pending_sizes = ()
        self.ctx.chosen_estates = ()
        self.phase = Phase.ASK_RESHUFFLE

    def _finish_player_turn(self) -> None:
        """``stConfirmTurn`` / ``stWaitOther`` -- hand over to the next seat."""
        self.actor += 1
        if self.actor < self.config.players:
            self.ctx = TurnCtx()
            self.phase = Phase.CHOOSE_CARDS
        else:
            self._end_turn()

    def _settle(self) -> None:
        """Run out every transition that BGA resolves without asking the player."""
        for _ in range(64):
            if self.phase is Phase.ACTION_PARK and not self._park_streets():
                self.phase = Phase.CHOOSE_PLAN
                continue
            if self.phase is Phase.ACTION_POOL and not self._pool_available():
                self.phase = Phase.CHOOSE_PLAN
                continue
            if self.phase is Phase.CHOOSE_PLAN and not self.scorable_plan_slots():
                self._finish_player_turn()
                continue
            if self.phase is Phase.ASK_RESHUFFLE and not self._may_ask_reshuffle():
                self.phase = Phase.CHOOSE_PLAN
                continue
            return
        raise AssertionError("settle loop did not converge")

    # ──────────────────────────────────────────────────────────────────
    # Information sets
    # ──────────────────────────────────────────────────────────────────
    def sheet_for(self, viewer: int, target: int) -> Sheet:
        """The sheet ``viewer`` is allowed to see for ``target``.

        Your own sheet is live; everyone else's is frozen as of the start of the
        turn, mirroring ``Houses::getOfPlayer``.
        """
        return self.sheets[target] if viewer == target else self.public_sheets[target]

    def reshuffle_vote_for(self, viewer: int) -> bool:
        """Whether ``viewer`` knows a reshuffle is coming -- i.e. voted for one.

        Not :attr:`reshuffle_next_turn`, which is the table-wide OR.  The vote
        happens mid-turn and is consumed at the start of the next one, so the
        aggregate is **never** legitimately public while it is true: reading it
        tells a later serial actor that an earlier one voted yes, which in the
        concurrent game nobody knows.
        """
        return self.reshuffle_votes.get(viewer, False)

    def plan_turns_for(self, viewer: int, slot: int) -> dict[int, int]:
        """Plan completions ``viewer`` may see (``AbstractPlan::getValidations``)."""
        return {
            p: t
            for p, t in self.plan_turns[slot].items()
            if p == viewer or t < self.turn
        }

    def redeterminize(self, rng: random.Random) -> "GameState":
        """Resample everything the acting player cannot see.

        Only one thing is hidden: the order of the cards still to come.  Every
        card on the table is fully identified (see :meth:`table_cards`) and the
        deck's *composition* is public bookkeeping
        (:func:`games.welcome_to.deck_knowledge.deck_composition`), so this is a
        pure permutation of the undrawn deck -- it never invents or destroys a
        card, and never has to guess at a face.

        Expert mode is only partly determinized: the opponents' private stacks
        and the card about to be passed along are not resampled.

        MCTS must call this at the root, otherwise search reads the true shuffle
        and the trained policy inherits the cheat.

        ``rng`` is **required, and must be a search RNG that the caller advances
        between simulations.**  It was optional once, defaulting to the copy's
        own generator, and that was a silent trap: :meth:`copy` clones the RNG
        state exactly, so every ``state.redeterminize()`` began from the same
        seed and returned the *same* shuffle.  A search built on that would
        explore one fixed future while looking perfectly healthy.  Kingdomino's
        ``encoder.redeterminize(state, rng)`` has always required the argument;
        this now matches it.

        The returned state also gets a **fresh generator of its own**, derived
        from ``rng``.  Without that, two determinizations would carry identical
        RNG state into the rollout, so any later stochastic step -- a deck
        exhaustion or a mid-rollout reshuffle, both of which call
        :meth:`_reform_deck` -- would apply the same permutation pattern across
        simulations that were supposed to be independent.
        """
        nxt = self.copy()
        unseen = nxt.deck[nxt.deck_pos:]
        rng.shuffle(unseen)
        nxt.deck = nxt.deck[: nxt.deck_pos] + unseen
        nxt.rng = random.Random(rng.getrandbits(64))
        return nxt

    # ──────────────────────────────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────────────────────────────
    def _sheet_view(self, player: int, viewer: Optional[int]) -> Sheet:
        return self.sheets[player] if viewer is None else self.sheet_for(viewer, player)

    def _plan_turns_view(self, slot: int, viewer: Optional[int]) -> dict[int, int]:
        return (
            self.plan_turns[slot]
            if viewer is None
            else self.plan_turns_for(viewer, slot)
        )

    def plan_scores(self, viewer: Optional[int] = None) -> list[int]:
        """``AbstractPlan::getScores`` -- first finishers take the higher value.

        Pass ``viewer`` to score off the information that player is allowed to
        have; leave it ``None`` for ground truth (which is what final scoring
        uses).
        """
        out = [0] * self.config.players
        for slot, plan_id in enumerate(self.plan_ids):
            turns = self._plan_turns_view(slot, viewer)
            if not turns:
                continue
            first = min(turns.values())
            scores = PLANS[plan_id].scores
            for player, turn in turns.items():
                if player < 0:
                    continue
                out[player] += scores[0 if turn == first else 1]
        return out

    def temp_scores(self, viewer: Optional[int] = None) -> list[int]:
        """``Actions/Temp::getScore`` -- 7 / 4 / 1 by rank; a flat 7 in solo."""
        counts = [self._sheet_view(p, viewer).temps for p in range(self.config.players)]
        if self.config.single_player:
            return [TEMP_SOLO_SCORE if counts[0] >= TEMP_SOLO_THRESHOLD else 0]
        ranked = sorted({c for c in counts if c > 0}, reverse=True) + [-1, -1, -1]
        out = [0] * self.config.players
        for p, c in enumerate(counts):
            if c <= 0:
                continue
            for rank in range(3):
                if c == ranked[rank]:
                    out[p] = TEMP_RANK_SCORES[rank]
        return out

    def scores(self, viewer: Optional[int] = None) -> list[int]:
        plans = self.plan_scores(viewer)
        temps = self.temp_scores(viewer)
        out = []
        for p in range(self.config.players):
            base = self._sheet_view(p, viewer).local_score()
            out.append(
                SheetScore(
                    parks=base.parks,
                    pools=base.pools,
                    estates=base.estates,
                    plans=plans[p],
                    temp=temps[p],
                    bis=base.bis,
                    permits=base.permits,
                    roundabouts=base.roundabouts,
                ).total
            )
        return out

    def score_breakdown(self, player: int, viewer: Optional[int] = None) -> SheetScore:
        base = self._sheet_view(player, viewer).local_score()
        return SheetScore(
            parks=base.parks,
            pools=base.pools,
            estates=base.estates,
            plans=self.plan_scores(viewer)[player],
            temp=self.temp_scores(viewer)[player],
            bis=base.bis,
            permits=base.permits,
            roundabouts=base.roundabouts,
        )

    def ranking(self) -> list[int]:
        """Seats best-first, applying the estate tie-breaker."""
        scores = self.scores()
        return sorted(
            range(self.config.players),
            key=lambda p: (scores[p], self.sheets[p].tiebreak_key()),
            reverse=True,
        )

    def winners(self) -> list[int]:
        """Every seat sharing the best (score, tie-break) -- usually exactly one."""
        scores = self.scores()
        keys = [(scores[p], self.sheets[p].tiebreak_key()) for p in range(self.config.players)]
        best = max(keys)
        return [p for p, k in enumerate(keys) if k == best]

    def returns(self) -> list[float]:
        """Zero-sum-ish outcome in [-1, 1] for training: +1 win, 0 draw, -1 loss."""
        if not self.is_terminal:
            return [0.0] * self.config.players
        win = self.winners()
        if self.config.players == 1:
            return [0.0]
        share = 1.0 / len(win)
        return [
            (2.0 * share - 1.0) if p in win else -1.0
            for p in range(self.config.players)
        ]

    # ──────────────────────────────────────────────────────────────────
    # Debugging
    # ──────────────────────────────────────────────────────────────────
    def pretty(self) -> str:
        lines = [
            f"turn {self.turn}  phase {self.phase.name}  actor {self.actor}",
            "stacks: "
            + ", ".join(
                f"[{n if n is not None else '?'}|{e.name if e else '?'}]"
                for n, e in self.visible_cards()
            ),
            "plans: "
            + ", ".join(
                f"{PLANS[pid].kind.name}{PLANS[pid].params or ''}"
                f"={sorted(self.plan_turns[s].items())}"
                for s, pid in enumerate(self.plan_ids)
            ),
        ]
        for p, sheet in enumerate(self.sheets):
            lines.append(f"-- player {p}  score {self.scores()[p]}")
            lines.append(sheet.pretty())
        return "\n".join(lines)


#: ``ST_WRITE_NUMBER`` transition table (``states.inc.php``).
_EFFECT_PHASE: dict[Effect, Phase] = {
    Effect.SURVEYOR: Phase.ACTION_SURVEYOR,
    Effect.ESTATE: Phase.ACTION_ESTATE,
    Effect.PARK: Phase.ACTION_PARK,
    Effect.POOL: Phase.ACTION_POOL,
    Effect.TEMP: Phase.CHOOSE_PLAN,
    Effect.BIS: Phase.ACTION_BIS,
}


def play_random_game(
    seed: int = 0,
    config: Optional[GameConfig] = None,
    rng: Optional[random.Random] = None,
    max_steps: int = 20000,
) -> GameState:
    """Roll a game out with uniform random legal moves.  Handy as a smoke test."""
    rng = rng or random.Random(seed)
    state = GameState.new(seed=seed, config=config)
    for _ in range(max_steps):
        if state.is_terminal:
            return state
        state.apply(rng.choice(state.legal_actions()))
    raise RuntimeError("game did not terminate")
