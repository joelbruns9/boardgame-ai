//! The Welcome To engine — a mirror of `games/welcome_to/game.py`.
//!
//! ⚠ **Python is the oracle** (RUST_PORT_PLAN.md §3). `game.py` is what the BGA
//! differential harness validates; this is validated against `game.py`. If they
//! disagree, Python wins — including when this looks more sensible.
//!
//! The serialised-simultaneous-turn design, the information-set rules and the
//! three-part turn boundary are all documented at length in `game.py`; the
//! comments here name the Python method each block mirrors rather than
//! re-arguing the design.

use crate::codec;
use crate::constants::{
    card_effect, card_number, num_base_cards, solo_card_id, Effect, BIS_BOXES, MAX_NUMBER, MIN_NUMBER, PERMIT_BOXES, SOLO_DECK_MIDDLE, TEMP_BOXES, TEMP_DELTAS,
    TEMP_RANK_SCORES, TEMP_SOLO_SCORE, TEMP_SOLO_THRESHOLD,
};
use crate::plans::{
    available_plan_ids, can_be_scored, estates_matching_size, validation_cells, PLANS,
};
use crate::rng::Rng;
use crate::sheet::{Estate, Pos, Sheet, SheetScore};

/// The mock player id BGA uses when the solo card validates every plan.
pub const SOLO_MOCK_PLAYER: i32 = -1;

/// A card slot holding nothing. Python spells it `None`.
pub const NO_CARD: i32 = -1;

#[derive(Debug)]
pub enum EngineError {
    /// `game.IllegalAction`.
    Illegal(String),
    /// `RuntimeError` / `ValueError` at construction.
    Invalid(String),
}

pub type EngineResult<T> = Result<T, EngineError>;

fn illegal<T>(message: impl Into<String>) -> EngineResult<T> {
    Err(EngineError::Illegal(message.into()))
}

// ──────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Config {
    pub players: usize,
    /// Advanced variant: five extra City Plans in stacks 1 and 2, plus
    /// roundabouts.
    pub advanced: bool,
    /// Expert variant: each player gets their own three cards.
    pub expert: bool,
    /// Whether a one-player game uses the real solo rules.
    pub solo_rules: bool,
}

impl Config {
    /// Real solo mode, with the solo card and its scoring.
    pub fn solo(&self) -> bool {
        self.players == 1 && self.solo_rules
    }

    /// One seat, whether or not the solo rules are switched on.
    pub fn single_player(&self) -> bool {
        self.players == 1
    }

    /// `Globals::isStandard` — shared stacks of two cards each.
    pub fn standard(&self) -> bool {
        !self.expert && !self.solo()
    }

    /// How many independent sets of three stacks exist.
    pub fn stack_groups(&self) -> usize {
        if self.expert {
            self.players
        } else {
            1
        }
    }

    /// 3 shared stacks in standard mode, 6 ordered card pairs otherwise.
    pub fn choice_slots(&self) -> usize {
        if self.standard() {
            3
        } else {
            codec::EXPERT_PAIRS.len()
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Phases
// ──────────────────────────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Phase {
    ChooseCards = 0,
    RoundaboutPlace = 1,
    WriteNumber = 2,
    ActionSurveyor = 3,
    ActionEstate = 4,
    ActionPark = 5,
    ActionPool = 6,
    ActionBis = 7,
    ChoosePlan = 8,
    ValidatePlan = 9,
    AskReshuffle = 10,
    GameOver = 11,
}

impl Phase {
    pub fn as_i64(self) -> i64 {
        self as u8 as i64
    }

    /// The inverse, for a snapshot coming back in.
    pub fn from_i64(value: i64) -> Option<Phase> {
        Some(match value {
            0 => Phase::ChooseCards,
            1 => Phase::RoundaboutPlace,
            2 => Phase::WriteNumber,
            3 => Phase::ActionSurveyor,
            4 => Phase::ActionEstate,
            5 => Phase::ActionPark,
            6 => Phase::ActionPool,
            7 => Phase::ActionBis,
            8 => Phase::ChoosePlan,
            9 => Phase::ValidatePlan,
            10 => Phase::AskReshuffle,
            11 => Phase::GameOver,
            _ => return None,
        })
    }

    /// `_EFFECT_PHASE` — the `ST_WRITE_NUMBER` transition table.
    fn for_effect(effect: Effect) -> Phase {
        match effect {
            Effect::Surveyor => Phase::ActionSurveyor,
            Effect::Estate => Phase::ActionEstate,
            Effect::Park => Phase::ActionPark,
            Effect::Pool => Phase::ActionPool,
            Effect::Temp => Phase::ChoosePlan,
            Effect::Bis => Phase::ActionBis,
            Effect::Solo => panic!("the solo marker is not a playable effect"),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Turn scratch state
// ──────────────────────────────────────────────────────────────────────────
#[derive(Clone, Debug, Default)]
pub struct TurnCtx {
    pub slot: Option<usize>,
    pub number: Option<i32>,
    pub effect: Option<Effect>,
    pub last_house: Option<Pos>,
    pub built_roundabout: bool,
    /// Set once the player has declined the roundabout this turn — declining is
    /// sticky, which stops a bot oscillating between ST_ROUNDABOUT and
    /// ST_CHOOSE_CARDS forever. See `game.py` for the measurement.
    pub roundabout_declined: bool,
    pub refused: bool,
    pub plan_slot: Option<usize>,
    pub pending_sizes: Vec<usize>,
    pub chosen_estates: Vec<Estate>,
}

/// One immediate chance outcome at a turn boundary — `SEARCH_SPEC.md` §6.3.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryOutcome {
    /// The ordered sequence of **raw** draws the boundary made.
    pub draws: Vec<i32>,
    /// Whether the deck was reformed from the discard during this boundary.
    pub reformed: bool,
}

/// Where `_draw_playable` takes its cards from.
///
/// Python passes an optional `DrawFn` closure at the *raw* draw level; a
/// closure that also needs `&mut self` does not borrow-check here, so the three
/// call styles are an enum threaded alongside `self` instead. Same three cases,
/// same single reveal path.
enum DrawMode {
    /// `_draw` — off the top of the deck.
    Deck,
    /// `sample_boundary_outcome`'s recorder.
    Record { drawn: Vec<i32>, reformed: bool },
    /// `apply_boundary_outcome`'s replay: each draw is dictated.
    Replay { queue: Vec<i32>, at: usize },
}

// ──────────────────────────────────────────────────────────────────────────
// The state
// ──────────────────────────────────────────────────────────────────────────
#[derive(Clone, Debug)]
pub struct Game {
    pub config: Config,
    pub sheets: Vec<Sheet>,
    /// Snapshot of every sheet as it stood at the start of the current turn.
    /// This, not `sheets`, is what other players are allowed to see.
    pub public_sheets: Vec<Sheet>,
    pub deck: Vec<i32>,
    pub deck_pos: usize,
    pub discard: Vec<i32>,
    /// `stack_new[g][i]` — the card on top of the stack, showing its NUMBER.
    pub stack_new: Vec<[i32; 3]>,
    /// `stack_old[g][i]` — standard mode only: the card flipped aside, showing
    /// its EFFECT. Empty in expert and solo.
    pub stack_old: Vec<[i32; 3]>,
    /// Expert mode: the card the previous player passed to each player.
    pub expert_pending: Vec<i32>,
    pub plan_ids: [usize; 3],
    /// `plan_turns[slot]` — `(player, turn)` completions, in insertion order.
    /// The player may be `SOLO_MOCK_PLAYER`.
    pub plan_turns: [Vec<(i32, i32)>; 3],
    pub turn: i32,
    pub actor: usize,
    pub phase: Phase,
    pub ctx: TurnCtx,
    /// Per-player combination slot chosen this turn (`NO_CARD` for none).
    pub turn_choice: Vec<i32>,
    pub reshuffle_next_turn: bool,
    /// How each player voted at `ASK_RESHUFFLE` this turn, by seat. Private
    /// until the turn resolves — an information-set-safe reader asks this
    /// rather than the table-wide OR.
    pub reshuffle_votes: Vec<(usize, bool)>,
    pub rng: Rng,
    pub solo_card_drawn: bool,
    /// Whether `prepare_turn_boundary` has run and no reveal has followed.
    pub boundary_prepared: bool,
}

impl Game {
    // ──────────────────────────────────────────────────────────────────
    // Construction
    // ──────────────────────────────────────────────────────────────────
    /// `welcometo::setupNewGame`, on the portable RNG (M0-B).
    pub fn new(seed: u64, config: Config) -> EngineResult<Game> {
        if config.players < 1 {
            return Err(EngineError::Invalid("need at least one player".into()));
        }
        if config.expert && config.players < 2 {
            return Err(EngineError::Invalid(
                "expert rules need at least two players".into(),
            ));
        }
        let mut rng = Rng::new(seed);
        let mut deck: Vec<i32> = (0..num_base_cards() as i32).collect();
        rng.shuffle(&mut deck);

        let mut plan_ids = [0usize; 3];
        for (i, stack) in [1u8, 2, 3].into_iter().enumerate() {
            let choices = available_plan_ids(stack, config.advanced);
            plan_ids[i] = rng.choice(&choices);
        }

        let sheets: Vec<Sheet> = (0..config.players).map(|_| Sheet::new()).collect();
        let groups = config.stack_groups();
        let mut state = Game {
            config,
            public_sheets: sheets.clone(),
            sheets,
            deck,
            deck_pos: 0,
            discard: Vec::new(),
            stack_new: vec![[NO_CARD; 3]; groups],
            stack_old: if config.standard() {
                vec![[NO_CARD; 3]; groups]
            } else {
                Vec::new()
            },
            expert_pending: vec![NO_CARD; config.players],
            plan_ids,
            plan_turns: [Vec::new(), Vec::new(), Vec::new()],
            turn: 1,
            actor: 0,
            phase: Phase::ChooseCards,
            ctx: TurnCtx::default(),
            turn_choice: vec![NO_CARD; config.players],
            reshuffle_next_turn: false,
            reshuffle_votes: Vec::new(),
            rng,
            solo_card_drawn: false,
            boundary_prepared: false,
        };

        if config.solo() {
            state.solo_setup();
        }
        // `ConstructionCards::setupNewGame`: standard mode seeds one card in
        // each stack so the first `stNewTurn` has something to flip.
        let mut mode = DrawMode::Deck;
        if config.standard() {
            for i in 0..3 {
                state.stack_new[0][i] = state.draw_playable(&mut mode)?;
            }
        } else if config.expert {
            for p in 0..config.players {
                state.expert_pending[p] = state.draw_playable(&mut mode)?;
            }
        }

        state.begin_turn(&mut mode)?;
        Ok(state)
    }

    /// `ConstructionCards::soloSetupNewGame`.
    fn solo_setup(&mut self) {
        let solo = solo_card_id() as i32;
        self.deck.push(solo);
        self.rng.shuffle(&mut self.deck);
        let index = self.deck.iter().position(|&c| c == solo).expect("just pushed");
        // Python compares against a possibly-negative limit, which is simply
        // false for every index; a usize subtraction would panic instead, so
        // the saturating form is what preserves the comparison.
        let limit = (self.deck.len() - 1).saturating_sub(SOLO_DECK_MIDDLE);
        if self.deck.len() > SOLO_DECK_MIDDLE && index <= limit {
            self.deck.remove(index);
            let at = (index + SOLO_DECK_MIDDLE).min(self.deck.len());
            self.deck.insert(at, solo);
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // Deck
    // ──────────────────────────────────────────────────────────────────
    /// `Pieces::reformDeckFromDiscard`.
    fn reform_deck(&mut self) {
        let mut deck: Vec<i32> = self.deck[self.deck_pos..].to_vec();
        deck.extend(self.discard.drain(..));
        self.rng.shuffle(&mut deck);
        self.deck = deck;
        self.deck_pos = 0;
    }

    fn draw(&mut self) -> EngineResult<i32> {
        if self.deck_pos >= self.deck.len() {
            self.reform_deck();
        }
        if self.deck_pos >= self.deck.len() {
            return Err(EngineError::Invalid(
                "construction deck and discard are both empty".into(),
            ));
        }
        let card = self.deck[self.deck_pos];
        self.deck_pos += 1;
        Ok(card)
    }

    /// `_draw`, except *which* card comes off the top is dictated. The card is
    /// swapped to the top of the undrawn region, so the deck's **composition**
    /// stays exact while the order of what is left stays arbitrary.
    fn replay_draw(&mut self, card: i32) -> EngineResult<i32> {
        if self.deck_pos >= self.deck.len() {
            self.reform_deck();
        }
        let at = self.deck[self.deck_pos..]
            .iter()
            .position(|&c| c == card)
            .map(|i| i + self.deck_pos);
        let Some(at) = at else {
            return illegal(format!(
                "card {card} is not in the undrawn deck; this outcome does not \
                 belong to this boundary"
            ));
        };
        self.deck.swap(self.deck_pos, at);
        self.deck_pos += 1;
        Ok(card)
    }

    /// One raw draw, through whichever source this reveal is running on.
    fn take(&mut self, mode: &mut DrawMode) -> EngineResult<i32> {
        match mode {
            DrawMode::Deck => self.draw(),
            DrawMode::Record { drawn, reformed } => {
                if self.deck_pos >= self.deck.len() {
                    *reformed = true;
                }
                let card = self.draw()?;
                drawn.push(card);
                Ok(card)
            }
            DrawMode::Replay { queue, at } => {
                if *at >= queue.len() {
                    return illegal(
                        "the outcome ran out of cards; it does not belong to this boundary",
                    );
                }
                let card = queue[*at];
                *at += 1;
                self.replay_draw(card)
            }
        }
    }

    /// Draw, resolving the solo card if it turns up (`ConstructionCards::drawAux`).
    ///
    /// The replacement happens at the *raw* draw level, deliberately, so that a
    /// solo card and the extra draw it forces are recorded and replayed like any
    /// other card rather than being a special case a second implementation has
    /// to know about.
    fn draw_playable(&mut self, mode: &mut DrawMode) -> EngineResult<i32> {
        let mut card = self.take(mode)?;
        if card == solo_card_id() as i32 {
            self.on_solo_card();
            card = self.take(mode)?;
        }
        Ok(card)
    }

    /// `ConstructionCards::soloCardDrawn` — the ghost claims every plan, on the
    /// *previous* turn, so the solo player can never take a first-place value.
    fn on_solo_card(&mut self) {
        self.solo_card_drawn = true;
        let turn = self.turn - 1;
        for slot in 0..3 {
            if !self.plan_turns[slot]
                .iter()
                .any(|&(p, _)| p == SOLO_MOCK_PLAYER)
            {
                self.plan_turns[slot].push((SOLO_MOCK_PLAYER, turn));
            }
        }
    }

    pub fn deck_remaining(&self) -> usize {
        self.deck.len() - self.deck_pos
    }

    // ──────────────────────────────────────────────────────────────────
    // Turn boundaries
    // ──────────────────────────────────────────────────────────────────
    /// `stNewTurn`, kept whole for `setupNewGame`, which begins a turn without
    /// *ending* one: no turn increment and no end-of-game test.
    fn begin_turn(&mut self, mode: &mut DrawMode) -> EngineResult<()> {
        self.discard_step();
        self.reveal_step(mode)?;
        self.open_turn();
        Ok(())
    }

    /// The part of a boundary that reveals cards — all four cases of §6.3.
    ///
    /// ⚠ Which case fires is **not** chance: a queued reshuffle and an
    /// exact-empty reform are both settled by the prepared state before a card
    /// is seen. Only *which cards* is chance.
    fn reveal_step(&mut self, mode: &mut DrawMode) -> EngineResult<()> {
        if self.reshuffle_next_turn {
            self.reshuffle_decks(mode)?;
            self.reshuffle_next_turn = false;
        }
        self.draw_step(mode)
    }

    /// The deterministic tail: hand the turn to seat 0 and settle.
    fn open_turn(&mut self) {
        self.boundary_prepared = false;
        self.actor = 0;
        self.ctx = TurnCtx::default();
        self.reshuffle_votes.clear();
        self.turn_choice = vec![NO_CARD; self.config.players];
        self.phase = Phase::ChooseCards;
        self.public_sheets = self.sheets.clone();
        self.settle();
    }

    /// `ConstructionCards::discardAux`.
    fn discard_step(&mut self) {
        if self.config.standard() {
            for i in 0..3 {
                let old = self.stack_old[0][i];
                if old != NO_CARD {
                    self.discard.push(old);
                }
                self.stack_old[0][i] = self.stack_new[0][i];
                self.stack_new[0][i] = NO_CARD;
            }
        } else {
            for g in 0..self.stack_new.len() {
                for i in 0..3 {
                    if self.stack_new[g][i] != NO_CARD {
                        self.discard.push(self.stack_new[g][i]);
                    }
                    self.stack_new[g][i] = NO_CARD;
                }
            }
        }
    }

    /// `ConstructionCards::drawAux`.
    fn draw_step(&mut self, mode: &mut DrawMode) -> EngineResult<()> {
        if self.config.expert {
            for p in 0..self.config.players {
                for i in 0..3 {
                    if i == 0 && self.expert_pending[p] != NO_CARD {
                        self.stack_new[p][0] = self.expert_pending[p];
                        self.expert_pending[p] = NO_CARD;
                    } else {
                        self.stack_new[p][i] = self.draw_playable(mode)?;
                    }
                }
            }
        } else {
            for i in 0..3 {
                self.stack_new[0][i] = self.draw_playable(mode)?;
            }
        }
        Ok(())
    }

    /// `ConstructionCards::reshuffle` — offered once, after the first plan.
    ///
    /// In standard mode BGA reforms the deck and then burns a full draw/discard
    /// cycle, so the pair on show afterwards is two freshly drawn cards.
    fn reshuffle_decks(&mut self, mode: &mut DrawMode) -> EngineResult<()> {
        self.reform_deck();
        if self.config.standard() {
            for i in 0..3 {
                self.stack_new[0][i] = self.draw_playable(mode)?;
            }
            self.discard_step();
        } else if self.config.expert {
            for p in 0..self.config.players {
                if self.expert_pending[p] == NO_CARD {
                    self.expert_pending[p] = self.draw_playable(mode)?;
                }
            }
        }
        Ok(())
    }

    /// `stApplyTurn` — the three-part boundary, run straight through.
    fn end_turn(&mut self) -> EngineResult<()> {
        if !self.prepare_turn_boundary()? {
            return Ok(());
        }
        let mut mode = DrawMode::Deck;
        self.reveal_step(&mut mode)?;
        self.open_turn();
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────
    // The boundary, in three parts — SEARCH_SPEC.md §6.3
    // ──────────────────────────────────────────────────────────────────
    /// Everything a boundary does **before a card is revealed**, in place.
    ///
    /// Returns `true` if a reveal follows and `false` if the game ended here,
    /// which is the fourth case of §6.3 and the one a search that assumes "a
    /// boundary reveals cards" gets wrong.
    pub fn prepare_turn_boundary(&mut self) -> EngineResult<bool> {
        if self.boundary_prepared {
            return illegal(
                "this boundary is already prepared; preparing twice would \
                 increment the turn again and discard the pair just promoted",
            );
        }
        if self.config.expert {
            self.pass_unused_cards();
        }
        self.turn += 1;
        if self.is_end_of_game() {
            self.phase = Phase::GameOver;
            self.public_sheets = self.sheets.clone();
            return Ok(false);
        }
        self.discard_step();
        self.boundary_prepared = true;
        Ok(true)
    }

    fn is_boundary_afterstate(&self) -> bool {
        self.boundary_prepared && !self.is_terminal()
    }

    /// One immediate outcome of this boundary. **Does not modify `self`.**
    ///
    /// The probe is re-determinized first, so repeated calls on the *same*
    /// afterstate give independent outcomes — which is what a chance node needs
    /// and what reading the top of an already-determinized deck would not give.
    pub fn sample_boundary_outcome(&self, rng: &mut Rng) -> EngineResult<BoundaryOutcome> {
        if !self.is_boundary_afterstate() {
            return illegal(
                "sample_boundary_outcome needs a prepared boundary afterstate; \
                 call prepare_turn_boundary() first",
            );
        }
        let mut probe = self.redeterminize(rng);
        // `reform_deck` has exactly two call sites and both are visible from
        // here: `reshuffle_decks` always reforms, and `draw` reforms when the
        // undrawn region is empty as it is entered. Checking both is exact,
        // where inferring one from how `deck_pos` moved is not.
        let mut mode = DrawMode::Record {
            drawn: Vec::new(),
            reformed: probe.reshuffle_next_turn,
        };
        probe.reveal_step(&mut mode)?;
        match mode {
            DrawMode::Record { drawn, reformed } => Ok(BoundaryOutcome { draws: drawn, reformed }),
            _ => unreachable!(),
        }
    }

    /// Apply `outcome` to this afterstate, in place, and open the turn.
    ///
    /// ⚠ **Transactional**: the replay runs on a copy and is adopted only once
    /// the outcome has been validated *whole*. "Raises, and also destroys the
    /// state you called it on" is not a contract worth having.
    pub fn apply_boundary_outcome(&mut self, draws: &[i32]) -> EngineResult<()> {
        if !self.is_boundary_afterstate() {
            return illegal(
                "apply_boundary_outcome needs a prepared boundary afterstate; \
                 call prepare_turn_boundary() first",
            );
        }
        let mut staged = self.clone();
        let mut mode = DrawMode::Replay {
            queue: draws.to_vec(),
            at: 0,
        };
        staged.reveal_step(&mut mode)?;
        if let DrawMode::Replay { queue, at } = &mode {
            if *at < queue.len() {
                return illegal("the outcome has cards this boundary did not draw");
            }
        }
        staged.scramble_undrawn();
        staged.open_turn();
        *self = staged;
        Ok(())
    }

    /// Shuffle what is left of the deck, so its order carries no artefact of
    /// *which cards the outcome named*. The composition is exact either way and
    /// the order is hidden, but "nothing reads it" was a docstring rather than
    /// a property; this makes it one.
    fn scramble_undrawn(&mut self) {
        let mut tail: Vec<i32> = self.deck[self.deck_pos..].to_vec();
        self.rng.shuffle(&mut tail);
        self.deck.truncate(self.deck_pos);
        self.deck.extend(tail);
    }

    /// `Player::giveThirdCardToNextPlayer`. A player who took a permit refusal
    /// never chose a pair, so all three of their cards are simply discarded.
    fn pass_unused_cards(&mut self) {
        let mut moved: Vec<(usize, i32)> = Vec::new();
        for p in 0..self.config.players {
            let slot = self.turn_choice[p];
            if slot == NO_CARD {
                continue;
            }
            let (i, j) = codec::EXPERT_PAIRS[slot as usize];
            let spare = 3 - i - j; // the one of {0,1,2} the pair did not use
            let card = self.stack_new[p][spare];
            if card == NO_CARD {
                continue;
            }
            self.stack_new[p][spare] = NO_CARD;
            moved.push(((p + 1) % self.config.players, card));
        }
        for (target, card) in moved {
            self.expert_pending[target] = card;
        }
    }

    /// `EndOfGameTrait::isEndOfGame`.
    fn is_end_of_game(&self) -> bool {
        for (p, sheet) in self.sheets.iter().enumerate() {
            if !sheet.has_free_box() {
                return true;
            }
            if (0..3).all(|slot| self.plan_turn_of(slot, p as i32).is_some()) {
                return true;
            }
            if sheet.permits >= PERMIT_BOXES {
                return true;
            }
        }
        if self.config.solo() && self.deck_remaining() == 0 {
            return true;
        }
        false
    }

    /// Which `isEndOfGame` clause fired, for logging.
    pub fn end_of_game_reason(&self) -> Option<String> {
        for (p, sheet) in self.sheets.iter().enumerate() {
            if !sheet.has_free_box() {
                return Some(format!("player {p} filled every house"));
            }
            if (0..3).all(|slot| self.plan_turn_of(slot, p as i32).is_some()) {
                return Some(format!("player {p} completed all three plans"));
            }
            if sheet.permits >= PERMIT_BOXES {
                return Some(format!("player {p} took a third permit refusal"));
            }
        }
        if self.config.solo() && self.deck_remaining() == 0 {
            return Some("deck exhausted (solo)".into());
        }
        None
    }

    fn plan_turn_of(&self, slot: usize, player: i32) -> Option<i32> {
        self.plan_turns[slot]
            .iter()
            .find(|&&(p, _)| p == player)
            .map(|&(_, t)| t)
    }

    // ──────────────────────────────────────────────────────────────────
    // Card combinations
    // ──────────────────────────────────────────────────────────────────
    fn group_of(&self, player: usize) -> usize {
        if self.config.expert {
            player
        } else {
            0
        }
    }

    /// `ConstructionCards::getCombination` — the `(number, effect)` on offer.
    pub fn combination(&self, slot: usize, player: usize) -> EngineResult<(i32, Effect)> {
        let g = self.group_of(player);
        let (number_card, effect_card) = if self.config.standard() {
            // The number in play sits on top of the stack; the effect in play is
            // on the card flipped aside beside it last turn.
            (self.stack_new[g][slot], self.stack_old[g][slot])
        } else {
            let (i, j) = codec::EXPERT_PAIRS[slot];
            (self.stack_new[g][i], self.stack_new[g][j])
        };
        if number_card == NO_CARD || effect_card == NO_CARD {
            return Err(EngineError::Invalid("stacks are not populated yet".into()));
        }
        Ok((
            card_number(number_card as usize),
            card_effect(effect_card as usize),
        ))
    }

    /// The visible faces of stack `slot`, without the expert pairing.
    pub fn combination_faces(&self, slot: usize, player: usize) -> (Option<i32>, Option<Effect>) {
        let g = self.group_of(player);
        let top = self.stack_new[g][slot];
        if !self.config.standard() {
            if top == NO_CARD {
                return (None, None);
            }
            return (
                Some(card_number(top as usize)),
                Some(card_effect(top as usize)),
            );
        }
        let aside = self.stack_old[g][slot];
        (
            if top == NO_CARD {
                None
            } else {
                Some(card_number(top as usize))
            },
            if aside == NO_CARD {
                None
            } else {
                Some(card_effect(aside as usize))
            },
        )
    }

    /// The effect each stack will offer NEXT turn — known with certainty now,
    /// because a card prints the effect from its own back on its number face.
    /// `None` per slot in expert and solo mode, where nothing carries over.
    pub fn next_effects(&self, player: usize) -> [Option<Effect>; 3] {
        if !self.config.standard() {
            return [None, None, None];
        }
        let g = self.group_of(player);
        let mut out = [None, None, None];
        for i in 0..3 {
            let card = self.stack_new[g][i];
            out[i] = if card == NO_CARD {
                None
            } else {
                Some(card_effect(card as usize))
            };
        }
        out
    }

    /// Every card on the table. All of them are fully identified.
    pub fn table_cards(&self, player: usize) -> Vec<i32> {
        let g = self.group_of(player);
        let mut cards = self.stack_new[g].to_vec();
        if self.config.standard() {
            cards.extend_from_slice(&self.stack_old[g]);
        }
        cards
    }

    /// `Player::getAvailableNumbersOfCombination` — candidates in codec order.
    pub fn numbers_for(&self, number: i32, effect: Effect) -> Vec<i32> {
        if effect != Effect::Temp {
            return vec![number];
        }
        let mut out = vec![number];
        for &delta in TEMP_DELTAS[1..].iter() {
            let n = number + delta;
            if (MIN_NUMBER..=MAX_NUMBER).contains(&n) {
                out.push(n);
            }
        }
        out
    }

    fn writable(&self, number: i32, effect: Effect, sheet: &Sheet) -> bool {
        self.numbers_for(number, effect)
            .into_iter()
            .any(|n| !sheet.available_locations(Some(n)).is_empty())
    }

    /// `Player::getAvailableStacks` — combinations with somewhere to write.
    pub fn playable_slots(&self, player: usize) -> Vec<usize> {
        let sheet = &self.sheets[player];
        let mut out = Vec::new();
        for slot in 0..self.config.choice_slots() {
            if let Ok((number, effect)) = self.combination(slot, player) {
                if self.writable(number, effect, sheet) {
                    out.push(slot);
                }
            }
        }
        out
    }

    // ──────────────────────────────────────────────────────────────────
    // Legal actions
    // ──────────────────────────────────────────────────────────────────
    pub fn is_terminal(&self) -> bool {
        self.phase == Phase::GameOver
    }

    /// The legal action indices, **in the order `game.py` produces them**.
    ///
    /// ⚠ Raw order is load-bearing: PUCT's first-max tie-break depends on it, so
    /// the M1 gate compares ordered lists rather than sets.
    pub fn legal_actions(&self) -> Vec<usize> {
        let phase = self.phase;
        if phase == Phase::GameOver {
            return Vec::new();
        }
        let sheet = &self.sheets[self.actor];
        let ctx = &self.ctx;

        match phase {
            Phase::ChooseCards => {
                let mut actions: Vec<usize> = self
                    .playable_slots(self.actor)
                    .into_iter()
                    .map(codec::choose_stack)
                    .collect();
                if actions.is_empty() && sheet.can_take_permit() {
                    actions.push(codec::A_PERMIT_REFUSAL);
                }
                if self.config.advanced
                    && ctx.last_house.is_none()
                    && !ctx.roundabout_declined
                    && sheet.can_build_roundabout()
                    && sheet.has_free_box()
                {
                    actions.push(codec::A_ROUNDABOUT_OPEN);
                }
                actions
            }
            Phase::RoundaboutPlace => {
                let mut actions: Vec<usize> = sheet
                    .available_locations(None)
                    .into_iter()
                    .map(|(x, y)| codec::roundabout_pos(x, y))
                    .collect();
                actions.push(codec::A_PASS_ROUNDABOUT);
                actions
            }
            Phase::WriteNumber => {
                let number = ctx.number.expect("WRITE_NUMBER without a number");
                let effect = ctx.effect.expect("WRITE_NUMBER without an effect");
                let mut actions = Vec::new();
                for n in self.numbers_for(number, effect) {
                    let delta_slot = TEMP_DELTAS
                        .iter()
                        .position(|&d| d == n - number)
                        .expect("candidate is not a temp delta");
                    for (x, y) in sheet.available_locations(Some(n)) {
                        actions.push(codec::write(delta_slot, x, y));
                    }
                }
                // `argWriteNumber`: never force a player to spend the temp
                // agency just to have somewhere to write.
                if sheet.available_locations(Some(number)).is_empty() && sheet.can_take_permit() {
                    actions.push(codec::A_PERMIT_REFUSAL);
                }
                actions
            }
            Phase::ActionSurveyor => {
                let mut actions: Vec<usize> = sheet
                    .surveyor_zones()
                    .into_iter()
                    .map(|(x, j)| codec::surveyor_fence(x, j))
                    .collect();
                actions.push(codec::A_PASS_SURVEYOR);
                actions
            }
            Phase::ActionEstate => {
                let mut actions: Vec<usize> = sheet
                    .estate_rows()
                    .into_iter()
                    .map(codec::estate_row)
                    .collect();
                actions.push(codec::A_PASS_ESTATE);
                actions
            }
            Phase::ActionPark => {
                let mut actions: Vec<usize> = self
                    .park_streets()
                    .into_iter()
                    .map(codec::park_street)
                    .collect();
                actions.push(codec::A_PASS_PARK);
                actions
            }
            Phase::ActionPool => vec![codec::A_POOL_BUILD, codec::A_PASS_POOL],
            Phase::ActionBis => {
                let mut actions: Vec<usize> = sheet
                    .bis_candidates()
                    .into_iter()
                    .map(|(x, y, _, side)| codec::bis(x, y, side))
                    .collect();
                actions.push(codec::A_PASS_BIS);
                actions
            }
            Phase::ChoosePlan => {
                let mut actions: Vec<usize> = self
                    .scorable_plan_slots()
                    .into_iter()
                    .map(codec::choose_plan)
                    .collect();
                actions.push(codec::A_PASS_PLAN);
                actions
            }
            Phase::ValidatePlan => {
                let size = ctx.pending_sizes[0];
                estates_matching_size(sheet, size, &ctx.chosen_estates)
                    .into_iter()
                    .map(|(x, start, _)| codec::validate_estate(x, start))
                    .collect()
            }
            Phase::AskReshuffle => vec![codec::A_RESHUFFLE_YES, codec::A_RESHUFFLE_NO],
            Phase::GameOver => unreachable!("handled above"),
        }
    }

    /// `Actions/Park::getAvailableZones` — same street as the house just written.
    fn park_streets(&self) -> Vec<usize> {
        let Some((street, _)) = self.ctx.last_house else {
            return Vec::new();
        };
        self.sheets[self.actor]
            .park_streets()
            .into_iter()
            .filter(|&x| x == street)
            .collect()
    }

    fn pool_available(&self) -> bool {
        match self.ctx.last_house {
            None => false,
            Some(pos) => self.sheets[self.actor].can_build_pool_at(pos),
        }
    }

    /// `Player::getScorablePlans`.
    pub fn scorable_plan_slots(&self) -> Vec<usize> {
        let sheet = &self.sheets[self.actor];
        let mut out = Vec::new();
        for slot in 0..3 {
            if self.plan_turn_of(slot, self.actor as i32).is_some() {
                continue;
            }
            if can_be_scored(&PLANS[self.plan_ids[slot]], sheet) {
                out.push(slot);
            }
        }
        out
    }

    /// `stAskReshuffle` — only before any plan has been completed on an earlier
    /// turn. Real solo skips it, as BGA does.
    fn may_ask_reshuffle(&self) -> bool {
        if self.config.solo() {
            return false;
        }
        !self
            .plan_turns
            .iter()
            .any(|slot| slot.iter().any(|&(_, t)| t < self.turn))
    }

    // ──────────────────────────────────────────────────────────────────
    // Stepping
    // ──────────────────────────────────────────────────────────────────
    /// Apply `action` in place.
    pub fn apply(&mut self, action: usize) -> EngineResult<()> {
        if !self.legal_actions().contains(&action) {
            return illegal(format!(
                "action {action} is not legal in {:?} for player {}",
                self.phase, self.actor
            ));
        }
        self.dispatch(action)?;
        self.settle();
        Ok(())
    }

    fn dispatch(&mut self, action: usize) -> EngineResult<()> {
        let phase = self.phase;
        let actor = self.actor;
        let turn = self.turn;

        match phase {
            Phase::ChooseCards => {
                if action == codec::A_PERMIT_REFUSAL {
                    let sheet = &mut self.sheets[actor];
                    sheet.permits = (sheet.permits + 1).min(PERMIT_BOXES);
                    self.ctx.refused = true;
                    self.phase = Phase::ChoosePlan;
                    return Ok(());
                }
                if action == codec::A_ROUNDABOUT_OPEN {
                    self.phase = Phase::RoundaboutPlace;
                    return Ok(());
                }
                let slot = codec::decode_stack(action);
                let (number, effect) = self.combination(slot, actor)?;
                self.ctx.slot = Some(slot);
                self.ctx.number = Some(number);
                self.ctx.effect = Some(effect);
                self.turn_choice[actor] = slot as i32;
                self.phase = Phase::WriteNumber;
                Ok(())
            }
            Phase::RoundaboutPlace => {
                if action == codec::A_PASS_ROUNDABOUT {
                    self.ctx.roundabout_declined = true;
                } else {
                    let pos = codec::decode_roundabout_pos(action);
                    self.sheets[actor].build_roundabout(pos, turn);
                    self.ctx.built_roundabout = true;
                    self.ctx.last_house = Some(pos);
                }
                self.phase = Phase::ChooseCards;
                Ok(())
            }
            Phase::WriteNumber => {
                if action == codec::A_PERMIT_REFUSAL {
                    let sheet = &mut self.sheets[actor];
                    sheet.permits = (sheet.permits + 1).min(PERMIT_BOXES);
                    self.ctx.refused = true;
                    self.phase = Phase::ChoosePlan;
                    return Ok(());
                }
                let (delta_slot, x, y) = codec::decode_write(action);
                let base = self.ctx.number.expect("WRITE_NUMBER without a number");
                let effect = self.ctx.effect.expect("WRITE_NUMBER without an effect");
                let number = base + TEMP_DELTAS[delta_slot];
                self.sheets[actor].write(number, (x, y), turn, false);
                self.ctx.last_house = Some((x, y));
                self.phase = Phase::for_effect(effect);
                if effect == Effect::Temp {
                    // `stActionTemp` crosses a box off with no decision to make.
                    let sheet = &mut self.sheets[actor];
                    sheet.temps = (sheet.temps + 1).min(TEMP_BOXES);
                    self.phase = Phase::ChoosePlan;
                }
                Ok(())
            }
            Phase::ActionSurveyor => {
                if action != codec::A_PASS_SURVEYOR {
                    let (x, j) = codec::decode_surveyor_fence(action);
                    self.sheets[actor].fences[x][j] = true;
                }
                self.phase = Phase::ChoosePlan;
                Ok(())
            }
            Phase::ActionEstate => {
                if action != codec::A_PASS_ESTATE {
                    let row = codec::decode_estate_row(action);
                    self.sheets[actor].estate_marks[row] += 1;
                }
                self.phase = Phase::ChoosePlan;
                Ok(())
            }
            Phase::ActionPark => {
                if action != codec::A_PASS_PARK {
                    let x = codec::decode_park_street(action);
                    self.sheets[actor].parks[x] += 1;
                }
                self.phase = Phase::ChoosePlan;
                Ok(())
            }
            Phase::ActionPool => {
                if action == codec::A_POOL_BUILD {
                    let (x, _) = self.ctx.last_house.expect("pool without a house");
                    self.sheets[actor].pools[x] += 1;
                }
                self.phase = Phase::ChoosePlan;
                Ok(())
            }
            Phase::ActionBis => {
                if action != codec::A_PASS_BIS {
                    let (x, y, side) = codec::decode_bis(action);
                    let number = self.sheets[actor]
                        .bis_number_at(x, y, side)
                        .expect("legal bis has a number");
                    let sheet = &mut self.sheets[actor];
                    sheet.write(number, (x, y), turn, true);
                    sheet.bis_marks = (sheet.bis_marks + 1).min(BIS_BOXES);
                    self.ctx.last_house = Some((x, y));
                }
                self.phase = Phase::ChoosePlan;
                Ok(())
            }
            Phase::ChoosePlan => {
                if action == codec::A_PASS_PLAN {
                    return self.finish_player_turn();
                }
                let slot = codec::decode_plan(action);
                let plan = &PLANS[self.plan_ids[slot]];
                self.ctx.plan_slot = Some(slot);
                self.ctx.chosen_estates = Vec::new();
                if plan.is_automatic() {
                    self.validate_plan(slot);
                } else {
                    self.ctx.pending_sizes = plan.required_sizes();
                    self.phase = Phase::ValidatePlan;
                }
                Ok(())
            }
            Phase::ValidatePlan => {
                let (x, start) = codec::decode_validate_estate(action);
                let size = self.ctx.pending_sizes[0];
                self.ctx.chosen_estates.push((x, start, size));
                self.ctx.pending_sizes.remove(0);
                if self.ctx.pending_sizes.is_empty() {
                    let slot = self.ctx.plan_slot.expect("validating without a plan");
                    self.validate_plan(slot);
                }
                Ok(())
            }
            Phase::AskReshuffle => {
                let yes = action == codec::A_RESHUFFLE_YES;
                self.set_reshuffle_vote(actor, yes);
                if yes {
                    self.reshuffle_next_turn = true;
                }
                self.phase = Phase::ChoosePlan;
                Ok(())
            }
            Phase::GameOver => illegal("the game is over"),
        }
    }

    fn set_reshuffle_vote(&mut self, seat: usize, vote: bool) {
        match self.reshuffle_votes.iter_mut().find(|(s, _)| *s == seat) {
            Some(entry) => entry.1 = vote,
            None => self.reshuffle_votes.push((seat, vote)),
        }
    }

    /// `AbstractPlan::validate` — consume houses, record the turn.
    fn validate_plan(&mut self, slot: usize) {
        let plan = &PLANS[self.plan_ids[slot]];
        let cells = validation_cells(plan, &self.ctx.chosen_estates);
        self.sheets[self.actor].mark_top_fences(&cells);
        let actor = self.actor as i32;
        match self.plan_turns[slot].iter_mut().find(|(p, _)| *p == actor) {
            Some(entry) => entry.1 = self.turn,
            None => self.plan_turns[slot].push((actor, self.turn)),
        }
        self.ctx.plan_slot = None;
        self.ctx.pending_sizes = Vec::new();
        self.ctx.chosen_estates = Vec::new();
        self.phase = Phase::AskReshuffle;
    }

    /// `stConfirmTurn` / `stWaitOther` — hand over to the next seat.
    fn finish_player_turn(&mut self) -> EngineResult<()> {
        self.actor += 1;
        if self.actor < self.config.players {
            self.ctx = TurnCtx::default();
            self.phase = Phase::ChooseCards;
            Ok(())
        } else {
            self.end_turn()
        }
    }

    /// Run out every transition BGA resolves without asking the player.
    fn settle(&mut self) {
        for _ in 0..64 {
            if self.phase == Phase::ActionPark && self.park_streets().is_empty() {
                self.phase = Phase::ChoosePlan;
                continue;
            }
            if self.phase == Phase::ActionPool && !self.pool_available() {
                self.phase = Phase::ChoosePlan;
                continue;
            }
            if self.phase == Phase::ChoosePlan && self.scorable_plan_slots().is_empty() {
                // `_finish_player_turn` only errors out of `_end_turn`, and an
                // engine that reached a boundary with an empty deck has already
                // failed louder than a settle loop can report.
                self.finish_player_turn().expect("turn boundary failed while settling");
                continue;
            }
            if self.phase == Phase::AskReshuffle && !self.may_ask_reshuffle() {
                self.phase = Phase::ChoosePlan;
                continue;
            }
            return;
        }
        panic!("settle loop did not converge");
    }

    // ──────────────────────────────────────────────────────────────────
    // Information sets
    // ──────────────────────────────────────────────────────────────────
    /// The sheet `viewer` is allowed to see for `target` — your own is live,
    /// everyone else's is frozen as of the start of the turn.
    pub fn sheet_for(&self, viewer: usize, target: usize) -> &Sheet {
        if viewer == target {
            &self.sheets[target]
        } else {
            &self.public_sheets[target]
        }
    }

    /// Whether `viewer` knows a reshuffle is coming — i.e. voted for one. Not
    /// `reshuffle_next_turn`, which is the table-wide OR and is never
    /// legitimately public while it is true.
    pub fn reshuffle_vote_for(&self, viewer: usize) -> bool {
        self.reshuffle_votes
            .iter()
            .find(|&&(s, _)| s == viewer)
            .map(|&(_, v)| v)
            .unwrap_or(false)
    }

    /// Plan completions `viewer` may see (`AbstractPlan::getValidations`).
    pub fn plan_turns_for(&self, viewer: usize, slot: usize) -> Vec<(i32, i32)> {
        self.plan_turns[slot]
            .iter()
            .copied()
            .filter(|&(p, t)| p == viewer as i32 || t < self.turn)
            .collect()
    }

    /// Resample everything the acting player cannot see — a pure permutation of
    /// the undrawn deck, which never invents or destroys a card.
    ///
    /// The copy gets a **fresh generator of its own**, derived from `rng`;
    /// without that, two determinizations would carry identical RNG state into
    /// the rollout and any later reform would apply the same permutation across
    /// simulations that were supposed to be independent.
    pub fn redeterminize(&self, rng: &mut Rng) -> Game {
        let mut next = self.clone();
        let mut unseen: Vec<i32> = next.deck[next.deck_pos..].to_vec();
        rng.shuffle(&mut unseen);
        next.deck.truncate(next.deck_pos);
        next.deck.extend(unseen);
        next.rng = Rng::new(rng.getrandbits(64));
        next
    }

    // ──────────────────────────────────────────────────────────────────
    // Scoring
    // ──────────────────────────────────────────────────────────────────
    fn sheet_view(&self, player: usize, viewer: Option<usize>) -> &Sheet {
        match viewer {
            None => &self.sheets[player],
            Some(v) => self.sheet_for(v, player),
        }
    }

    fn plan_turns_view(&self, slot: usize, viewer: Option<usize>) -> Vec<(i32, i32)> {
        match viewer {
            None => self.plan_turns[slot].clone(),
            Some(v) => self.plan_turns_for(v, slot),
        }
    }

    /// `AbstractPlan::getScores` — first finishers take the higher value.
    pub fn plan_scores(&self, viewer: Option<usize>) -> Vec<i32> {
        let mut out = vec![0i32; self.config.players];
        for slot in 0..3 {
            let turns = self.plan_turns_view(slot, viewer);
            if turns.is_empty() {
                continue;
            }
            let first = turns.iter().map(|&(_, t)| t).min().expect("non-empty");
            let scores = PLANS[self.plan_ids[slot]].scores;
            for (player, turn) in turns {
                if player < 0 {
                    continue;
                }
                out[player as usize] += if turn == first { scores.0 } else { scores.1 };
            }
        }
        out
    }

    /// `Actions/Temp::getScore` — 7 / 4 / 1 by rank; a flat 7 in solo.
    pub fn temp_scores(&self, viewer: Option<usize>) -> Vec<i32> {
        let counts: Vec<i32> = (0..self.config.players)
            .map(|p| self.sheet_view(p, viewer).temps)
            .collect();
        if self.config.single_player() {
            return vec![if counts[0] >= TEMP_SOLO_THRESHOLD {
                TEMP_SOLO_SCORE
            } else {
                0
            }];
        }
        let mut distinct: Vec<i32> = counts.iter().copied().filter(|&c| c > 0).collect();
        distinct.sort_unstable();
        distinct.dedup();
        distinct.reverse();
        let mut ranked = distinct;
        ranked.extend([-1, -1, -1]);

        let mut out = vec![0i32; self.config.players];
        for (p, &c) in counts.iter().enumerate() {
            if c <= 0 {
                continue;
            }
            for rank in 0..3 {
                if c == ranked[rank] {
                    out[p] = TEMP_RANK_SCORES[rank];
                }
            }
        }
        out
    }

    pub fn scores(&self, viewer: Option<usize>) -> Vec<i32> {
        let plans = self.plan_scores(viewer);
        let temps = self.temp_scores(viewer);
        (0..self.config.players)
            .map(|p| {
                let base = self.sheet_view(p, viewer).local_score();
                SheetScore {
                    plans: plans[p],
                    temp: temps[p],
                    ..base
                }
                .total()
            })
            .collect()
    }

    pub fn score_breakdown(&self, player: usize, viewer: Option<usize>) -> SheetScore {
        let base = self.sheet_view(player, viewer).local_score();
        SheetScore {
            plans: self.plan_scores(viewer)[player],
            temp: self.temp_scores(viewer)[player],
            ..base
        }
    }

    /// Seats best-first, applying the estate tie-breaker.
    pub fn ranking(&self) -> Vec<usize> {
        let scores = self.scores(None);
        let mut order: Vec<usize> = (0..self.config.players).collect();
        // `sorted(..., reverse=True)` is stable, so equal keys keep ascending
        // seat order; sorting the reversed key list the same way reproduces it.
        order.sort_by(|&a, &b| {
            let ka = (scores[a], self.sheets[a].tiebreak_key());
            let kb = (scores[b], self.sheets[b].tiebreak_key());
            kb.partial_cmp(&ka).expect("total order")
        });
        order
    }

    /// Every seat sharing the best `(score, tie-break)` — usually exactly one.
    pub fn winners(&self) -> Vec<usize> {
        let scores = self.scores(None);
        let keys: Vec<(i32, [i32; 7])> = (0..self.config.players)
            .map(|p| (scores[p], self.sheets[p].tiebreak_key()))
            .collect();
        let best = keys.iter().max().expect("at least one seat");
        (0..self.config.players)
            .filter(|&p| keys[p] == *best)
            .collect()
    }

    /// Zero-sum-ish outcome in [-1, 1] for training: +1 win, 0 draw, -1 loss.
    pub fn returns(&self) -> Vec<f64> {
        if !self.is_terminal() {
            return vec![0.0; self.config.players];
        }
        if self.config.players == 1 {
            return vec![0.0];
        }
        let win = self.winners();
        let share = 1.0 / win.len() as f64;
        (0..self.config.players)
            .map(|p| {
                if win.contains(&p) {
                    2.0 * share - 1.0
                } else {
                    -1.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_player() -> Config {
        Config {
            players: 2,
            advanced: true,
            expert: false,
            solo_rules: false,
        }
    }

    #[test]
    fn a_new_game_opens_with_three_stacks_and_a_choice() {
        let state = Game::new(3, two_player()).expect("setup");
        assert_eq!(state.turn, 1);
        assert_eq!(state.actor, 0);
        assert_eq!(state.phase, Phase::ChooseCards);
        assert_eq!(state.stack_new.len(), 1);
        assert!(state.stack_new[0].iter().all(|&c| c != NO_CARD));
        assert!(state.stack_old[0].iter().all(|&c| c != NO_CARD));
        assert!(!state.legal_actions().is_empty());
    }

    #[test]
    fn a_random_game_terminates_and_scores() {
        let mut rng = Rng::new(99);
        let mut state = Game::new(11, two_player()).expect("setup");
        for _ in 0..20_000 {
            if state.is_terminal() {
                break;
            }
            let legal = state.legal_actions();
            let action = legal[rng.randrange(legal.len() as u64) as usize];
            state.apply(action).expect("legal action applies");
        }
        assert!(state.is_terminal());
        assert_eq!(state.scores(None).len(), 2);
        assert!(!state.winners().is_empty());
        assert!(state.end_of_game_reason().is_some());
    }
}
