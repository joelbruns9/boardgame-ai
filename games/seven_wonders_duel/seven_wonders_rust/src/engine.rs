//! Action resolution, victory conditions, and scoring — a port of `engine.py`.
//!
//! Chance resolution follows the `buffer.replay` simulator path (see
//! `state.rs`): reveals, age deals, and the wonder-group flip resolve from
//! locked state; only the Great Library draw consumes a recorded outcome.

use crate::data::{
    self, card, progress, progress_id, wonder, CardColor, Cost, EffectKind, Resource, ScienceSymbol,
};
use crate::rules::{discard_income, normal_trade_unit_cost};
use crate::state::{GameState, PendingChoice, PendingChoiceKind, Phase, VictoryType};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActionUse {
    DraftWonder,
    ConstructBuilding,
    DiscardForCoins,
    ConstructWonder,
    ResolvePendingChoice,
    ChooseNextStartPlayer,
}

/// One legal action. `slot` is a tableau slot index; `choice` is a card- or
/// progress-id per the pending kind; ids follow the codec's spaces.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Action {
    pub use_: ActionUse,
    pub slot: Option<usize>,
    pub wonder: Option<usize>,
    pub choice: Option<usize>,
    pub starting_player: Option<usize>,
}

impl Action {
    fn draft(wonder_id: usize) -> Action {
        Action {
            use_: ActionUse::DraftWonder,
            slot: None,
            wonder: Some(wonder_id),
            choice: None,
            starting_player: None,
        }
    }
    fn primary(use_: ActionUse, slot: usize, wonder: Option<usize>) -> Action {
        Action {
            use_,
            slot: Some(slot),
            wonder,
            choice: None,
            starting_player: None,
        }
    }
    fn pending(choice: usize) -> Action {
        Action {
            use_: ActionUse::ResolvePendingChoice,
            slot: None,
            wonder: None,
            choice: Some(choice),
            starting_player: None,
        }
    }
    fn next_start(player: usize) -> Action {
        Action {
            use_: ActionUse::ChooseNextStartPlayer,
            slot: None,
            wonder: None,
            choice: None,
            starting_player: Some(player),
        }
    }
}

/// Result of a payment search: only the total and trade portions affect state
/// (Economy rebate + Urbanism chain bonus); the purchase breakdown does not.
#[derive(Clone, Copy)]
pub(crate) struct Payment {
    pub(crate) total_coins: i32,
    pub(crate) trade_coins: i32,
    pub(crate) used_chain: bool,
}

/// Per-category victory-point breakdown (mirrors `engine.py::ScoreBreakdown`).
/// The encoder consumes every field; the endgame resolver needs only
/// `total`/`blue_buildings`.
#[derive(Clone, Copy)]
pub(crate) struct ScoreBreakdown {
    pub(crate) military: i32,
    pub(crate) buildings: i32, // all card VP incl. guilds
    pub(crate) guild: i32,     // guild-card VP alone (subset of buildings)
    pub(crate) wonders: i32,
    pub(crate) progress: i32,
    pub(crate) treasury: i32,
    pub(crate) total: i32,
    pub(crate) blue_buildings: i32,
}

// --- production / cost helpers ------------------------------------------------

fn count_color(g: &GameState, player: usize, color: CardColor) -> i32 {
    g.cities[player]
        .buildings
        .iter()
        .filter(|&&cid| card(cid).color == color)
        .count() as i32
}

fn has_token(g: &GameState, player: usize, name: &str) -> bool {
    g.cities[player]
        .progress_tokens
        .contains(&progress_id(name))
}

pub(crate) fn fixed_production(g: &GameState, player: usize) -> [i32; 5] {
    let mut out = [0i32; 5];
    for &cid in &g.cities[player].buildings {
        for &r in card(cid).fixed_production {
            out[r as usize] += 1;
        }
    }
    out
}

pub(crate) fn choice_producers(g: &GameState, player: usize) -> Vec<&'static [Resource]> {
    let mut out: Vec<&'static [Resource]> = Vec::new();
    for &cid in &g.cities[player].buildings {
        let cp = card(cid).choice_production;
        if !cp.is_empty() {
            out.push(cp);
        }
    }
    for &wid in &g.cities[player].built_wonders {
        let cp = wonder(wid).choice_production;
        if !cp.is_empty() {
            out.push(cp);
        }
    }
    out
}

pub(crate) fn opponent_trade_production(g: &GameState, player: usize) -> [i32; 5] {
    let mut out = [0i32; 5];
    for &cid in &g.cities[1 - player].buildings {
        let c = card(cid);
        if c.color == CardColor::Brown || c.color == CardColor::Grey {
            for &r in c.fixed_production {
                out[r as usize] += 1;
            }
        }
    }
    out
}

pub(crate) fn trade_discounts(g: &GameState, player: usize) -> [bool; 5] {
    let mut out = [false; 5];
    for &cid in &g.cities[player].buildings {
        for &r in card(cid).trade_discount {
            out[r as usize] = true;
        }
    }
    out
}

fn chain_is_free(g: &GameState, player: usize, c: &data::CardData) -> bool {
    match c.chain_from {
        None => false,
        Some(token) => g.cities[player]
            .buildings
            .iter()
            .any(|&cid| card(cid).chain_to == Some(token)),
    }
}

/// Enumerate rebate allocations (per-resource reductions summing to ≤ rebate,
/// each ≤ the cost's count) and, over each, the minimal trade cost across
/// flexible producers. Returns (total_coins, trade_coins, used_chain).
pub(crate) fn minimum_payment(
    g: &GameState,
    player: usize,
    cost: &Cost,
    card_opt: Option<&data::CardData>,
    is_wonder: bool,
) -> Payment {
    if let Some(c) = card_opt {
        if chain_is_free(g, player, c) {
            return Payment {
                total_coins: 0,
                trade_coins: 0,
                used_chain: true,
            };
        }
    }
    let mut rebate = 0;
    if is_wonder && has_token(g, player, "Architecture") {
        rebate = 2;
    } else if let Some(c) = card_opt {
        if c.color == CardColor::Blue && has_token(g, player, "Masonry") {
            rebate = 2;
        }
    }

    let fixed = fixed_production(g, player);
    let producers = choice_producers(g, player);
    let opponent = opponent_trade_production(g, player);
    let discounts = trade_discounts(g, player);

    let cost_counts: [i32; 5] = [cost.wood, cost.clay, cost.stone, cost.glass, cost.papyrus];

    // Precompute all producer assignments (cartesian product), as per-resource
    // added-production vectors.
    let mut assignments: Vec<[i32; 5]> = vec![[0i32; 5]];
    for prod in &producers {
        let mut next = Vec::with_capacity(assignments.len() * prod.len());
        for base in &assignments {
            for &r in *prod {
                let mut v = *base;
                v[r as usize] += 1;
                next.push(v);
            }
        }
        assignments = next;
    }

    let mut best_trade = i32::MAX;
    // Enumerate rebate allocations recursively over the five resources.
    let mut alloc = [0i32; 5];
    fn recurse(
        i: usize,
        remaining: i32,
        alloc: &mut [i32; 5],
        cost_counts: &[i32; 5],
        fixed: &[i32; 5],
        opponent: &[i32; 5],
        discounts: &[bool; 5],
        assignments: &[[i32; 5]],
        best_trade: &mut i32,
    ) {
        if i == 5 {
            for assign in assignments {
                let mut trade = 0;
                for r in 0..5 {
                    let requirement = cost_counts[r] - alloc[r];
                    let produced = fixed[r] + assign[r];
                    if requirement > produced {
                        let qty = requirement - produced;
                        let unit = if discounts[r] {
                            1
                        } else {
                            normal_trade_unit_cost(opponent[r])
                        };
                        trade += qty * unit;
                    }
                }
                if trade < *best_trade {
                    *best_trade = trade;
                }
            }
            return;
        }
        let max_here = cost_counts[i].min(remaining);
        for a in 0..=max_here {
            alloc[i] = a;
            recurse(
                i + 1,
                remaining - a,
                alloc,
                cost_counts,
                fixed,
                opponent,
                discounts,
                assignments,
                best_trade,
            );
        }
        alloc[i] = 0;
    }
    recurse(
        0,
        rebate,
        &mut alloc,
        &cost_counts,
        &fixed,
        &opponent,
        &discounts,
        &assignments,
        &mut best_trade,
    );

    let trade = best_trade;
    Payment {
        total_coins: cost.coins + trade,
        trade_coins: trade,
        used_chain: false,
    }
}

fn can_afford(g: &GameState, player: usize, p: &Payment) -> bool {
    g.cities[player].coins >= p.total_coins
}

/// Mirror of `engine.py::military_token_band`: the coin token whose BAND this
/// position sits in, keyed by the band's first space, or None between bands.
///
/// A military token is not on a single space. It is claimed on first entry to
/// a whole band and claimed only once -- BGA's `MilitaryTrack::getMilitaryToken`
/// buckets |position| into 3..5 and 6..8, then `takeMilitaryToken` zeroes the
/// token it returns.
fn military_token_band(position: i32) -> Option<i32> {
    let distance = position.abs();
    let band = if (3..=5).contains(&distance) {
        3
    } else if (6..=8).contains(&distance) {
        6
    } else {
        return None;
    };
    Some(if position > 0 { band } else { -band })
}

fn unbuilt_wonders(g: &GameState, player: usize) -> Vec<usize> {
    g.cities[player]
        .wonders
        .iter()
        .copied()
        .filter(|w| !g.cities[player].built_wonders.contains(w) && !g.retired_wonders.contains(w))
        .collect()
}

// --- legal actions ------------------------------------------------------------

impl GameState {
    pub fn legal_actions(&self) -> Vec<Action> {
        if let Some(p) = &self.pending_choice {
            return p.options.iter().map(|&o| Action::pending(o)).collect();
        }
        match self.phase {
            Phase::WonderDraft => self
                .legal_wonder_choices()
                .into_iter()
                .map(Action::draft)
                .collect(),
            Phase::ChooseNextStartPlayer => {
                vec![Action::next_start(0), Action::next_start(1)]
            }
            Phase::PlayAge => {
                let player = self.active_player;
                let mut actions = Vec::new();
                for slot in self.tableau.accessible_indices() {
                    let c = card(self.tableau.slots[slot].card_id);
                    let pay = minimum_payment(self, player, &c.cost, Some(c), false);
                    if can_afford(self, player, &pay) {
                        actions.push(Action::primary(ActionUse::ConstructBuilding, slot, None));
                    }
                    actions.push(Action::primary(ActionUse::DiscardForCoins, slot, None));
                    for wid in unbuilt_wonders(self, player) {
                        let w = wonder(wid);
                        let wc = w.cost.expect("wonder missing cost");
                        let wpay = minimum_payment(self, player, &wc, None, true);
                        if can_afford(self, player, &wpay) {
                            actions.push(Action::primary(
                                ActionUse::ConstructWonder,
                                slot,
                                Some(wid),
                            ));
                        }
                    }
                }
                actions
            }
            Phase::Complete => Vec::new(),
        }
    }

    // --- apply -----------------------------------------------------------------

    /// Record one heap change, when a journal is open. Free otherwise.
    #[inline]
    fn record(&mut self, delta: crate::state::Delta) {
        if let Some(journal) = self.journal.as_mut() {
            journal.push(delta);
        }
    }

    /// Apply `action`, returning what is needed to reverse it exactly.
    ///
    /// `None` means "not journaled, use snapshot/restore" and is never a
    /// failure: it is returned for anything outside the ordinary flow of a
    /// turn, where a move rewrites whole decks rather than nudging a few
    /// vectors. Refusing there is what keeps the journal honest -- an Age deal
    /// replaces the tableau and the deck wholesale, and no delta describes that
    /// cheaply.
    ///
    /// **Precondition:** `action` carries no chance signature. A chance action
    /// must go through `apply_with_chance` with explicit outcomes -- applying it
    /// here would resolve it from the locked deal, which is the hidden state a
    /// search must never read. The solver only journals `specs.is_empty()`
    /// edges; the assertion states that rather than leaving it to a reader.
    pub fn apply_journaled(&mut self, action: &Action) -> Option<crate::state::Undo> {
        debug_assert!(
            crate::chance::chance_signature(self, action).is_empty(),
            "apply_journaled on a chance action: use apply_with_chance"
        );
        if self.phase != Phase::PlayAge || self.journal.is_some() {
            return None;
        }
        match action.use_ {
            ActionUse::ConstructBuilding
            | ActionUse::ConstructWonder
            | ActionUse::DiscardForCoins
            | ActionUse::ResolvePendingChoice => {}
            _ => return None,
        }
        // The last card of an Age deals the next one (or ends the game), both
        // of which reach far beyond what the deltas cover.
        if crate::chance::exhausts_the_age(self, action) {
            return None;
        }
        let scalars = self.scalars();
        self.journal = Some(Vec::with_capacity(8));
        self.apply_action(action);
        let deltas = self.journal.take().unwrap_or_default();
        Some(crate::state::Undo { scalars, deltas })
    }

    fn scalars(&self) -> crate::state::Scalars {
        crate::state::Scalars {
            phase: self.phase,
            active_player: self.active_player,
            age: self.age,
            wonder_round: self.wonder_round,
            wonder_pick_index: self.wonder_pick_index,
            conflict_position: self.conflict_position,
            pending_extra_turn: self.pending_extra_turn,
            pending_shields: self.pending_shields,
            coins: [self.cities[0].coins, self.cities[1].coins],
            winner: self.winner,
            victory_type: self.victory_type,
            final_scores: self.final_scores,
            pending_choice: self.pending_choice.clone(),
        }
    }

    /// Reverse an `apply_journaled`, exactly. Deltas replay backwards so a move
    /// that removed and appended to the same list undoes in the right order.
    pub fn undo(&mut self, undo: crate::state::Undo) {
        use crate::state::{CityList, Delta};
        for delta in undo.deltas.into_iter().rev() {
            match delta {
                Delta::PushCity { seat, list } => {
                    match list {
                        CityList::Buildings => self.cities[seat].buildings.pop(),
                        CityList::BuiltWonders => self.cities[seat].built_wonders.pop(),
                        CityList::ProgressTokens => self.cities[seat].progress_tokens.pop(),
                        CityList::ClaimedSciencePairs => {
                            self.cities[seat].claimed_science_pairs.pop().map(|_| 0)
                        }
                    };
                }
                Delta::RemoveCity { seat, list, at, id } => match list {
                    CityList::Buildings => self.cities[seat].buildings.insert(at, id),
                    CityList::BuiltWonders => self.cities[seat].built_wonders.insert(at, id),
                    CityList::ProgressTokens => self.cities[seat].progress_tokens.insert(at, id),
                    CityList::ClaimedSciencePairs => {
                        unreachable!("science pairs are never removed")
                    }
                },
                Delta::PushDiscard => {
                    self.discard_pile.pop();
                }
                Delta::RemoveDiscard { at, id } => self.discard_pile.insert(at, id),
                Delta::PushBuried => {
                    self.buried_cards.pop();
                }
                Delta::PushBurial => {
                    self.wonder_burials.pop();
                }
                Delta::PushRetired => {
                    self.retired_wonders.pop();
                }
                Delta::RemoveMilitaryToken { at, token } => {
                    self.military_tokens_remaining.insert(at, token)
                }
                Delta::PopLibraryDraw { drawn } => self.library_draws.push_front(drawn),
                Delta::ProgressPools { available, unused } => {
                    self.available_progress_tokens = available;
                    self.unused_progress_tokens = unused;
                }
                Delta::Slot { index, before } => self.tableau.slots[index] = before,
            }
        }
        let s = undo.scalars;
        self.phase = s.phase;
        self.active_player = s.active_player;
        self.age = s.age;
        self.wonder_round = s.wonder_round;
        self.wonder_pick_index = s.wonder_pick_index;
        self.conflict_position = s.conflict_position;
        self.pending_extra_turn = s.pending_extra_turn;
        self.pending_shields = s.pending_shields;
        self.cities[0].coins = s.coins[0];
        self.cities[1].coins = s.coins[1];
        self.winner = s.winner;
        self.victory_type = s.victory_type;
        self.final_scores = s.final_scores;
        self.pending_choice = s.pending_choice;
    }

    pub fn apply_action(&mut self, action: &Action) {
        match action.use_ {
            ActionUse::DraftWonder => {
                let wid = action.wonder.expect("draft missing wonder");
                let _flipped = self.pick_wonder(wid);
                if self.phase == Phase::PlayAge {
                    // Eighth pick ends the draft: deal Age I from the locked deck.
                    let deck = self.age_decks[1].clone();
                    self.tableau = crate::state::TableauState::from_deck(1, &deck);
                }
            }
            ActionUse::ResolvePendingChoice => {
                let choice = action.choice.expect("pending missing choice");
                self.resolve_pending_choice(choice);
            }
            ActionUse::ChooseNextStartPlayer => {
                let sp = action.starting_player.expect("missing starting player");
                self.start_next_age(sp);
            }
            ActionUse::DiscardForCoins => {
                let slot = action.slot.expect("primary missing slot");
                let player = self.active_player;
                let card_id = self.tableau.slots[slot].card_id;
                self.take_and_reveal(slot);
                self.record(crate::state::Delta::PushDiscard);
                self.discard_pile.push(card_id);
                let yellow = count_color(self, player, CardColor::Yellow);
                self.cities[player].coins += discard_income(yellow);
                self.finish_turn(player, false);
            }
            ActionUse::ConstructBuilding => {
                let slot = action.slot.expect("primary missing slot");
                let player = self.active_player;
                let card_id = self.tableau.slots[slot].card_id;
                let c = card(card_id);
                let pay = minimum_payment(self, player, &c.cost, Some(c), false);
                self.pay(player, &pay);
                self.take_and_reveal(slot);
                self.record(crate::state::Delta::PushCity {
                    seat: player,
                    list: crate::state::CityList::Buildings,
                });
                self.cities[player].buildings.push(card_id);
                self.after_building_constructed(player, card_id);
                if pay.used_chain && has_token(self, player, "Urbanism") {
                    self.cities[player].coins += 4;
                }
                self.finish_turn(player, false);
            }
            ActionUse::ConstructWonder => {
                let slot = action.slot.expect("primary missing slot");
                let wid = action.wonder.expect("wonder action missing wonder");
                let player = self.active_player;
                let card_id = self.tableau.slots[slot].card_id;
                let w = wonder(wid);
                let wc = w.cost.expect("wonder missing cost");
                let pay = minimum_payment(self, player, &wc, None, true);
                self.pay(player, &pay);
                self.take_and_reveal(slot);
                self.record(crate::state::Delta::PushBuried);
                self.buried_cards.push(card_id);
                self.record(crate::state::Delta::PushBurial);
                self.wonder_burials.push((wid, card_id));
                self.record(crate::state::Delta::PushCity {
                    seat: player,
                    list: crate::state::CityList::BuiltWonders,
                });
                self.cities[player].built_wonders.push(wid);

                let total_built: usize = self.cities.iter().map(|c| c.built_wonders.len()).sum();
                if total_built == 7 {
                    let remaining: Vec<usize> = self
                        .cities
                        .iter()
                        .flat_map(|city| city.wonders.iter().copied())
                        .filter(|w| {
                            !self.cities[0].built_wonders.contains(w)
                                && !self.cities[1].built_wonders.contains(w)
                                && !self.retired_wonders.contains(w)
                        })
                        .collect();
                    assert_eq!(remaining.len(), 1, "seventh wonder must leave one unbuilt");
                    self.record(crate::state::Delta::PushRetired);
                    self.retired_wonders.push(remaining[0]);
                }

                let mut extra_turn = self.resolve_wonder_effects(player, wid);
                if has_token(self, player, "Theology") {
                    extra_turn = true;
                }
                if self.pending_choice.is_some() {
                    self.pending_shields = w.shields;
                } else if w.shields != 0 {
                    self.apply_military(player, w.shields);
                }
                self.finish_turn(player, extra_turn);
            }
        }
    }

    fn take_and_reveal(&mut self, slot: usize) {
        let before = self.tableau.slots[slot].clone();
        self.record(crate::state::Delta::Slot { index: slot, before });
        let (_card_id, newly) = self.tableau.take_accessible(slot);
        // Simulator reveal path: the locked card is already correct, so each
        // newly-accessible slot is simply revealed in (row, x) order.
        for j in newly {
            let before = self.tableau.slots[j].clone();
            self.record(crate::state::Delta::Slot { index: j, before });
            self.tableau.reveal(j);
        }
    }

    fn pay(&mut self, player: usize, p: &Payment) {
        assert!(can_afford(self, player, p), "cannot afford construction");
        self.cities[player].coins -= p.total_coins;
        if p.trade_coins != 0 && has_token(self, 1 - player, "Economy") {
            self.cities[1 - player].coins += p.trade_coins;
        }
    }

    fn apply_card_coin_effects(&mut self, player: usize, card_id: usize) {
        for effect in card(card_id).effects {
            match effect.kind {
                EffectKind::ImmediateCoins => self.cities[player].coins += effect.amount,
                EffectKind::CoinsPerOwnColor => {
                    let color = effect.color.expect("color-count effect missing color");
                    self.cities[player].coins += effect.amount * count_color(self, player, color);
                }
                EffectKind::CoinsPerOwnWonder => {
                    self.cities[player].coins +=
                        effect.amount * self.cities[player].built_wonders.len() as i32;
                }
                EffectKind::CoinsPerMostColor => {
                    let color = effect.color.expect("guild color effect missing color");
                    let best = count_color(self, 0, color).max(count_color(self, 1, color));
                    self.cities[player].coins += effect.amount * best;
                }
                EffectKind::CoinsPerMostBrownGrey => {
                    let best = (0..2)
                        .map(|p| {
                            count_color(self, p, CardColor::Brown)
                                + count_color(self, p, CardColor::Grey)
                        })
                        .max()
                        .unwrap();
                    self.cities[player].coins += effect.amount * best;
                }
                _ => {}
            }
        }
    }

    fn science_symbols(&self, player: usize) -> Vec<ScienceSymbol> {
        let mut out: Vec<ScienceSymbol> = Vec::new();
        for &cid in &self.cities[player].buildings {
            if let Some(s) = card(cid).science {
                if !out.contains(&s) {
                    out.push(s);
                }
            }
        }
        for &pid in &self.cities[player].progress_tokens {
            if let Some(s) = progress(pid).science {
                if !out.contains(&s) {
                    out.push(s);
                }
            }
        }
        out
    }

    fn declare_victory(&mut self, player: usize, vt: VictoryType) {
        self.winner = Some(player);
        self.victory_type = Some(vt);
        self.phase = Phase::Complete;
    }

    fn check_scientific_victory(&mut self, player: usize) -> bool {
        if self.science_symbols(player).len() >= 6 {
            self.declare_victory(player, VictoryType::Scientific);
            return true;
        }
        false
    }

    fn apply_science_building(&mut self, player: usize, card_id: usize) {
        let symbol = match card(card_id).science {
            None => return,
            Some(s) => s,
        };
        if self.check_scientific_victory(player) {
            return;
        }
        let copies = self.cities[player]
            .buildings
            .iter()
            .filter(|&&cid| card(cid).science == Some(symbol))
            .count();
        if copies >= 2 && !self.cities[player].claimed_science_pairs.contains(&symbol) {
            self.record(crate::state::Delta::PushCity {
                seat: player,
                list: crate::state::CityList::ClaimedSciencePairs,
            });
            self.cities[player].claimed_science_pairs.push(symbol);
            let options = self.available_progress_tokens.clone();
            self.set_pending_if_options(
                PendingChoiceKind::ChooseAvailableProgress,
                player,
                options,
                false,
            );
        }
    }

    fn apply_military(&mut self, player: usize, shields: i32) {
        let direction = if player == 0 { 1 } else { -1 };
        for _ in 0..shields {
            self.conflict_position += direction;
            let band = military_token_band(self.conflict_position);
            if let Some(pos) = band.and_then(|b| {
                self.military_tokens_remaining
                    .iter()
                    .position(|&(p, _)| p == b)
            }) {
                let token = self.military_tokens_remaining[pos];
                self.record(crate::state::Delta::RemoveMilitaryToken { at: pos, token });
                let (_, penalty) = self.military_tokens_remaining.remove(pos);
                let opp = &mut self.cities[1 - player];
                opp.coins = (opp.coins - penalty).max(0);
            }
            if self.conflict_position.abs() == 9 {
                self.declare_victory(player, VictoryType::Military);
                return;
            }
        }
    }

    fn after_building_constructed(&mut self, player: usize, card_id: usize) {
        self.apply_card_coin_effects(player, card_id);
        let c = card(card_id);
        let mut shields = c.shields;
        if c.color == CardColor::Red && has_token(self, player, "Strategy") {
            shields += 1;
        }
        if shields != 0 {
            self.apply_military(player, shields);
        }
        if self.phase != Phase::Complete && c.science.is_some() {
            self.apply_science_building(player, card_id);
        }
    }

    fn finish_turn(&mut self, player: usize, extra_turn: bool) {
        if self.phase == Phase::Complete {
            return;
        }
        if self.pending_choice.is_some() {
            self.pending_extra_turn = extra_turn;
            return;
        }
        if !self.tableau.accessible_indices().is_empty() {
            self.active_player = if extra_turn { player } else { 1 - player };
        } else if self.age == 3 {
            self.resolve_civilian_endgame();
        } else {
            // The next Age is laid out BEFORE the chooser is asked, as in the
            // physical game and on BGA: the choice is made looking at the
            // pyramid. Mirrors Python `_deal_next_age`.
            self.deal_next_age();
            self.phase = Phase::ChooseNextStartPlayer;
            self.active_player = if self.conflict_position > 0 {
                1
            } else if self.conflict_position < 0 {
                0
            } else {
                player
            };
        }
    }

    fn set_pending_if_options(
        &mut self,
        kind: PendingChoiceKind,
        player: usize,
        options: Vec<usize>,
        consume_all: bool,
    ) {
        if !options.is_empty() {
            self.pending_choice = Some(PendingChoice {
                kind,
                player,
                options,
                consume_all_options: consume_all,
            });
        }
    }

    fn resolve_wonder_effects(&mut self, player: usize, wonder_id: usize) -> bool {
        let mut extra_turn = false;
        for effect in wonder(wonder_id).effects {
            match effect.kind {
                EffectKind::ImmediateCoins => self.cities[player].coins += effect.amount,
                EffectKind::OpponentLosesCoins => {
                    let opp = &mut self.cities[1 - player];
                    opp.coins = (opp.coins - effect.amount).max(0);
                }
                EffectKind::PlayAgain => extra_turn = true,
                EffectKind::DestroyOpponentBrown => {
                    let options: Vec<usize> = self.cities[1 - player]
                        .buildings
                        .iter()
                        .copied()
                        .filter(|&cid| card(cid).color == CardColor::Brown)
                        .collect();
                    self.set_pending_if_options(
                        PendingChoiceKind::DestroyOpponentBrown,
                        player,
                        options,
                        false,
                    );
                }
                EffectKind::DestroyOpponentGrey => {
                    let options: Vec<usize> = self.cities[1 - player]
                        .buildings
                        .iter()
                        .copied()
                        .filter(|&cid| card(cid).color == CardColor::Grey)
                        .collect();
                    self.set_pending_if_options(
                        PendingChoiceKind::DestroyOpponentGrey,
                        player,
                        options,
                        false,
                    );
                }
                EffectKind::BuildFromDiscardFree => {
                    let options = self.discard_pile.clone();
                    self.set_pending_if_options(
                        PendingChoiceKind::BuildFromDiscardFree,
                        player,
                        options,
                        false,
                    );
                }
                EffectKind::ChooseUnusedProgress => {
                    let count = (effect.amount as usize).min(self.unused_progress_tokens.len());
                    if count > 0 {
                        let mut drawn = self
                            .library_draws
                            .pop_front()
                            .expect("great library draw outcome missing from chance log");
                        // Consumed, so undo has to hand it back to the front.
                        self.record(crate::state::Delta::PopLibraryDraw {
                            drawn: drawn.clone(),
                        });
                        assert_eq!(drawn.len(), count, "great library draw size mismatch");
                        drawn.sort_by_key(|&pid| pid);
                        self.set_pending_if_options(
                            PendingChoiceKind::ChooseUnusedProgress,
                            player,
                            drawn,
                            true,
                        );
                    }
                }
                _ => {}
            }
        }
        extra_turn
    }

    pub fn resolve_pending_choice(&mut self, choice: usize) {
        let pending = self
            .pending_choice
            .take()
            .expect("no pending choice to resolve");
        assert!(pending.options.contains(&choice), "invalid pending choice");
        let player = pending.player;
        let extra_turn = self.pending_extra_turn;
        self.pending_extra_turn = false;
        let pending_shields = self.pending_shields;
        self.pending_shields = 0;

        match pending.kind {
            PendingChoiceKind::DestroyOpponentBrown | PendingChoiceKind::DestroyOpponentGrey => {
                let pos = self.cities[1 - player]
                    .buildings
                    .iter()
                    .position(|&c| c == choice)
                    .expect("destroy target not present");
                self.record(crate::state::Delta::RemoveCity {
                    seat: 1 - player,
                    list: crate::state::CityList::Buildings,
                    at: pos,
                    id: choice,
                });
                self.cities[1 - player].buildings.remove(pos);
                self.record(crate::state::Delta::PushDiscard);
                self.discard_pile.push(choice);
            }
            PendingChoiceKind::BuildFromDiscardFree => {
                let pos = self
                    .discard_pile
                    .iter()
                    .position(|&c| c == choice)
                    .expect("revive target not in discard");
                self.record(crate::state::Delta::RemoveDiscard { at: pos, id: choice });
                self.discard_pile.remove(pos);
                self.record(crate::state::Delta::PushCity {
                    seat: player,
                    list: crate::state::CityList::Buildings,
                });
                self.cities[player].buildings.push(choice);
                self.after_building_constructed(player, choice);
            }
            PendingChoiceKind::ChooseUnusedProgress
            | PendingChoiceKind::ChooseAvailableProgress => {
                self.record(crate::state::Delta::PushCity {
                    seat: player,
                    list: crate::state::CityList::ProgressTokens,
                });
                self.cities[player].progress_tokens.push(choice);
                // The two pools are filtered rather than popped, and both are
                // tiny, so they are saved whole instead of described.
                self.record(crate::state::Delta::ProgressPools {
                    available: self.available_progress_tokens.clone(),
                    unused: self.unused_progress_tokens.clone(),
                });
                if pending.consume_all_options {
                    let consumed = pending.options.clone();
                    self.unused_progress_tokens
                        .retain(|t| !consumed.contains(t));
                } else {
                    self.available_progress_tokens.retain(|&t| t != choice);
                }
                self.apply_progress_immediate(player, choice);
                self.check_scientific_victory(player);
            }
        }

        if self.phase == Phase::Complete {
            return;
        }
        if pending_shields != 0 {
            self.apply_military(player, pending_shields);
        }
        if self.phase == Phase::Complete {
            return;
        }
        self.finish_turn(player, extra_turn);
    }

    fn apply_progress_immediate(&mut self, player: usize, token_id: usize) {
        for effect in progress(token_id).effects {
            if effect.kind == EffectKind::ImmediateCoins {
                self.cities[player].coins += effect.amount;
            }
        }
    }

    /// Lay out the next Age from the locked deck (the AGE_DEAL chance event).
    /// Fires when the current Age is exhausted, before the military chooser is
    /// asked who starts it. Port of Python `_deal_next_age`.
    fn deal_next_age(&mut self) {
        self.age += 1;
        let deck = self.age_decks[self.age as usize].clone();
        self.tableau = crate::state::TableauState::from_deck(self.age, &deck);
    }

    /// Resolve the military chooser's decision. Fires no chance event: the Age
    /// was dealt when the previous one ran out, so the chooser has already seen
    /// the layout.
    pub fn start_next_age(&mut self, starting_player: usize) {
        assert_eq!(
            self.phase,
            Phase::ChooseNextStartPlayer,
            "current age not complete"
        );
        assert!(starting_player < 2, "starting player must be 0 or 1");
        self.active_player = starting_player;
        self.phase = Phase::PlayAge;
    }

    // --- scoring / endgame -----------------------------------------------------

    fn military_victory_points(&self, player: usize) -> i32 {
        let position = self.conflict_position;
        if position == 0 || (position > 0) != (player == 0) {
            return 0;
        }
        // Bands 1-2 / 3-5 / 6-8, read from the CURRENT pawn position
        // (BGA MilitaryTrack::getVictoryPoints).
        let distance = position.abs();
        if distance <= 2 {
            2
        } else if distance <= 5 {
            5
        } else {
            10
        }
    }

    fn guild_victory_points(&self, player: usize) -> i32 {
        let mut points = 0;
        for &cid in &self.cities[player].buildings {
            for effect in card(cid).effects {
                match effect.kind {
                    EffectKind::VpPerMostColor => {
                        let color = effect.color.expect("guild VP effect missing color");
                        points += effect.amount
                            * count_color(self, 0, color).max(count_color(self, 1, color));
                    }
                    EffectKind::VpPerMostBrownGrey => {
                        let best = (0..2)
                            .map(|p| {
                                count_color(self, p, CardColor::Brown)
                                    + count_color(self, p, CardColor::Grey)
                            })
                            .max()
                            .unwrap();
                        points += effect.amount * best;
                    }
                    EffectKind::VpPerMostWonder => {
                        points += effect.amount
                            * self.cities[0]
                                .built_wonders
                                .len()
                                .max(self.cities[1].built_wonders.len())
                                as i32;
                    }
                    EffectKind::VpPerRichestCoinSet => {
                        let richest = self.cities[0].coins.max(self.cities[1].coins);
                        points += effect.amount * (richest / 3);
                    }
                    _ => {}
                }
            }
        }
        points
    }

    /// Full per-category victory-point breakdown (mirrors
    /// `engine.py::score_player`); the encoder consumes every field.
    pub(crate) fn score_player(&self, player: usize) -> ScoreBreakdown {
        let city = &self.cities[player];
        let military = self.military_victory_points(player);
        let guild = self.guild_victory_points(player);
        let buildings: i32 = city
            .buildings
            .iter()
            .map(|&c| card(c).victory_points)
            .sum::<i32>()
            + guild;
        let wonders: i32 = city
            .built_wonders
            .iter()
            .map(|&w| wonder(w).victory_points)
            .sum();
        let mut progress_vp: i32 = city
            .progress_tokens
            .iter()
            .map(|&p| progress(p).victory_points)
            .sum();
        for &pid in &city.progress_tokens {
            for effect in progress(pid).effects {
                if effect.kind == EffectKind::VpPerProgress {
                    progress_vp += effect.amount * city.progress_tokens.len() as i32;
                }
            }
        }
        let treasury = city.coins / 3;
        let blue: i32 = city
            .buildings
            .iter()
            .filter(|&&c| card(c).color == CardColor::Blue)
            .map(|&c| card(c).victory_points)
            .sum();
        let total = military + buildings + wonders + progress_vp + treasury;
        ScoreBreakdown {
            military,
            buildings,
            guild,
            wonders,
            progress: progress_vp,
            treasury,
            total,
            blue_buildings: blue,
        }
    }

    /// (total, blue_buildings) — the two quantities the civilian tiebreak needs.
    fn score_totals(&self, player: usize) -> (i32, i32) {
        let s = self.score_player(player);
        (s.total, s.blue_buildings)
    }

    fn resolve_civilian_endgame(&mut self) {
        let (t0, b0) = self.score_totals(0);
        let (t1, b1) = self.score_totals(1);
        self.final_scores = Some((t0, t1));
        self.phase = Phase::Complete;
        if t0 != t1 {
            self.winner = Some(if t0 > t1 { 0 } else { 1 });
            self.victory_type = Some(VictoryType::Civilian);
        } else if b0 != b1 {
            self.winner = Some(if b0 > b1 { 0 } else { 1 });
            self.victory_type = Some(VictoryType::Civilian);
        } else {
            self.winner = None;
            self.victory_type = Some(VictoryType::SharedCivilian);
        }
    }
}

/// Observable evidence that the engine fired exactly the predicted chance events.
///
/// `apply_with_chance` pre-installs each outcome and then applies the action
/// normally, so comparing `outcomes.len()` against `chance_signature` proves
/// nothing: the caller built those outcomes from that same signature, so a
/// wrong prediction agrees with itself. Python is protected structurally — its
/// chance context pops one supplied outcome per event as the event fires, and
/// raises on either an exhausted list or a leftover — and this is the Rust
/// equivalent, written against what each kind of event leaves behind.
///
/// Neither direction is harmless. An overpredicted AgeDeal has already
/// rewritten `age_decks` for an Age that was never dealt; an underpredicted one
/// resolves from the locked deal, which is exactly the hidden state the search
/// barrier exists to keep out of a search.
pub(crate) struct ChanceWitness {
    age: u8,
    phase: Phase,
    wonder_round: u8,
    library_draws: usize,
    revealed: usize,
}

impl ChanceWitness {
    pub(crate) fn of(state: &GameState) -> Self {
        Self {
            age: state.age,
            phase: state.phase,
            wonder_round: state.wonder_round,
            library_draws: state.library_draws.len(),
            revealed: Self::revealed(state),
        }
    }

    /// Monotone: `take_accessible` clears `present` but never `revealed`, so
    /// this only ever grows, by one per CARD_REVEAL.
    fn revealed(state: &GameState) -> usize {
        state.tableau.slots.iter().filter(|slot| slot.revealed).count()
    }

    pub(crate) fn check(
        &self,
        state: &GameState,
        specs: &[crate::chance::ChanceSpec],
    ) -> Result<(), String> {
        use crate::chance::ChanceKind;
        let predicted =
            |kind: ChanceKind| specs.iter().filter(|spec| spec.kind == kind).count();

        // A deal advances the Age -- except Age I's, which the 8th draft pick
        // fires while leaving `age` at 1. That one has NO state delta at all:
        // `from_setup` already laid the Age I tableau out, so an unoverridden
        // deal rebuilds an identical one. Leaving the draft is therefore the
        // only honest signal for it, and it is an exact one, since `pick_wonder`
        // enters PlayAge on the same pick that deals.
        let dealt = state.age != self.age
            || (self.phase == Phase::WonderDraft && state.phase != Phase::WonderDraft);
        if dealt != (predicted(ChanceKind::AgeDeal) == 1) {
            return Err(format!(
                "predicted {} AGE_DEAL event(s) but the engine {} one",
                predicted(ChanceKind::AgeDeal),
                if dealt { "fired" } else { "did not fire" }
            ));
        }

        let flipped = state.wonder_round != self.wonder_round;
        if flipped != (predicted(ChanceKind::WonderGroupReveal) == 1) {
            return Err(format!(
                "predicted {} WONDER_GROUP_REVEAL event(s) but the engine {} one",
                predicted(ChanceKind::WonderGroupReveal),
                if flipped { "fired" } else { "did not fire" }
            ));
        }

        // Measured from before pre-installation: one entry is pushed iff
        // predicted and one popped iff fired, so either way the queue must come
        // back to the length it started at.
        if state.library_draws.len() != self.library_draws {
            return Err(format!(
                "predicted {} GREAT_LIBRARY_DRAW event(s) but the draw queue went {} -> {}",
                predicted(ChanceKind::GreatLibraryDraw),
                self.library_draws,
                state.library_draws.len()
            ));
        }

        let reveals = predicted(ChanceKind::CardReveal);
        if dealt {
            // A deal replaces the tableau wholesale, so the count is not
            // comparable across it — but a deal means the take removed the last
            // card of the Age, which can expose nothing.
            if reveals != 0 {
                return Err(format!(
                    "predicted {reveals} CARD_REVEAL event(s) on the take that ends the Age"
                ));
            }
        } else if Self::revealed(state) != self.revealed + reveals {
            return Err(format!(
                "predicted {reveals} CARD_REVEAL event(s) but {} slot(s) were revealed",
                Self::revealed(state) - self.revealed
            ));
        }
        Ok(())
    }
}

// --- F3.1b: supplied-outcome apply (make_with_chance) -------------------------

impl GameState {
    /// Apply `action` with searcher-supplied chance `outcomes` (one id list per
    /// `chance_signature` spec, in order). Mirrors Python's
    /// `apply_action(chance_outcomes=...)`: each outcome is installed into the
    /// hidden state via a SWAP so the normal apply path resolves to it while the
    /// world stays a consistent whole. Pre-installing before `apply_action` is
    /// equivalent to Python's mid-apply overrides for the distinct outcomes the
    /// searcher produces — reveal targets are used-deduplicated, so no swap ever
    /// collides with an already-processed sibling slot.
    pub fn apply_with_chance(
        &mut self,
        action: &Action,
        outcomes: &[Vec<usize>],
    ) -> Result<(), String> {
        let specs = crate::chance::chance_signature(self, action);
        if specs.len() != outcomes.len() {
            return Err(format!(
                "expected {} chance outcome(s), got {}",
                specs.len(),
                outcomes.len()
            ));
        }
        // Validate every outcome against the pre-state BEFORE any mutation, so a
        // malformed supply cannot leave the state partially applied (the solver
        // make/unmake contract). Valid search outcomes always pass.
        self.validate_chance(&specs, outcomes)?;
        // BEFORE pre-installation: the Great Library outcome is pushed onto the
        // very queue the witness measures, so capturing after it would count
        // the push as if it were the engine's own work.
        let before = ChanceWitness::of(self);
        for (spec, outcome) in specs.iter().zip(outcomes) {
            match spec.kind {
                crate::chance::ChanceKind::CardReveal => {
                    let slot = self
                        .tableau
                        .slot_index_of(spec.context[0], spec.context[1])
                        .expect("reveal slot not found");
                    self.override_reveal(slot, outcome[0]);
                }
                crate::chance::ChanceKind::WonderGroupReveal => self.override_wonder_flip(outcome),
                crate::chance::ChanceKind::GreatLibraryDraw => {
                    self.library_draws.push_front(outcome.clone())
                }
                crate::chance::ChanceKind::AgeDeal => {
                    self.validated_age_deal(spec.context[0] as usize, outcome)
                }
            }
        }
        self.apply_action(action);
        before.check(self, &specs)
    }

    fn validate_chance(
        &self,
        specs: &[crate::chance::ChanceSpec],
        outcomes: &[Vec<usize>],
    ) -> Result<(), String> {
        use crate::chance::ChanceKind;
        use crate::data::{back_type_of, BackType, NUM_CARDS};
        let pool = crate::pool::unseen_pool(self);
        let distinct = |v: &[usize]| {
            let mut s = v.to_vec();
            s.sort_unstable();
            s.dedup();
            s.len() == v.len()
        };
        let mut revealed = Vec::new();
        for (spec, o) in specs.iter().zip(outcomes) {
            match spec.kind {
                ChanceKind::CardReveal => {
                    let back = spec.context[2] as usize;
                    if o.len() != 1 {
                        return Err("card reveal outcome must be one card".into());
                    }
                    let c = o[0];
                    if c >= NUM_CARDS || back_type_of(c) as usize != back {
                        return Err(format!("reveal {c} has the wrong back type"));
                    }
                    if revealed.contains(&c) || !pool.cards[back].contains(&c) {
                        return Err(format!("reveal {c} is not in the unseen pool"));
                    }
                    revealed.push(c);
                }
                ChanceKind::WonderGroupReveal => {
                    let mut flip_pool = self.wonder_groups[1].clone();
                    flip_pool.extend_from_slice(&self.unused_wonders);
                    if o.len() != 4 || !distinct(o) || o.iter().any(|w| !flip_pool.contains(w)) {
                        return Err("invalid wonder-flip outcome".into());
                    }
                }
                ChanceKind::GreatLibraryDraw => {
                    if o.len() != 3
                        || !distinct(o)
                        || o.iter().any(|p| !pool.offboard_progress.contains(p))
                    {
                        return Err("invalid Great Library draw".into());
                    }
                }
                ChanceKind::AgeDeal => {
                    let age = spec.context[0] as usize;
                    if o.len() != crate::data::layout(age as u8).len() || !distinct(o) {
                        return Err("age deal has the wrong size or a duplicate".into());
                    }
                    let mut visible = [false; NUM_CARDS];
                    for &c in &self.discard_pile {
                        visible[c] = true;
                    }
                    for &c in &self.buried_cards {
                        visible[c] = true;
                    }
                    for city in &self.cities {
                        for &c in &city.buildings {
                            visible[c] = true;
                        }
                    }
                    let mut guilds = 0;
                    for &c in o {
                        let back = back_type_of(c);
                        let ok_back = if age == 3 {
                            back == BackType::AgeIII || back == BackType::Guild
                        } else {
                            back as usize == age - 1
                        };
                        if !ok_back || visible[c] {
                            return Err(format!("card {c} cannot be in an age {age} deal"));
                        }
                        if back == BackType::Guild {
                            guilds += 1;
                        }
                    }
                    if age == 3 && guilds != 3 {
                        return Err("age III deal needs exactly 3 guilds".into());
                    }
                }
            }
        }
        Ok(())
    }

    /// Install `new_id` at `slot`, swapping the previously-locked card into the
    /// outcome card's hidden location (sibling face-down slot, then removed pile,
    /// then unused guilds) — a port of `_override_reveal`.
    fn override_reveal(&mut self, slot: usize, new_id: usize) {
        let old_id = self.tableau.slots[slot].card_id;
        if new_id == old_id {
            return;
        }
        for j in 0..self.tableau.slots.len() {
            let c = &self.tableau.slots[j];
            if c.present && !c.revealed && c.card_id == new_id {
                self.tableau.slots[j].card_id = old_id;
                self.tableau.slots[slot].card_id = new_id;
                return;
            }
        }
        let age = self.age as usize;
        if let Some(p) = self.removed_age_cards[age]
            .iter()
            .position(|&n| n == new_id)
        {
            self.removed_age_cards[age][p] = old_id;
            self.tableau.slots[slot].card_id = new_id;
            return;
        }
        if let Some(p) = self.unused_guilds.iter().position(|&n| n == new_id) {
            self.unused_guilds[p] = old_id;
            if let Some(sp) = self.selected_guilds.iter().position(|&n| n == old_id) {
                self.selected_guilds[sp] = new_id;
            }
            self.tableau.slots[slot].card_id = new_id;
            return;
        }
        panic!("reveal outcome {new_id} not in the unseen pool");
    }

    /// Set the flipped second wonder group; `wonder_offer` is left for
    /// `pick_wonder` to copy on the flip. Port of `_override_wonder_flip`.
    fn override_wonder_flip(&mut self, outcome: &[usize]) {
        let mut pool = self.wonder_groups[1].clone();
        pool.extend_from_slice(&self.unused_wonders);
        self.wonder_groups[1] = outcome.to_vec();
        self.unused_wonders = pool.into_iter().filter(|w| !outcome.contains(w)).collect();
    }

    /// Rearrange the setup records so `age`'s deal is `deal`, keeping removed
    /// cards / guild selection consistent — a port of `_validated_age_deal`.
    fn validated_age_deal(&mut self, age: usize, deal: &[usize]) {
        let mut visible = [false; crate::data::NUM_CARDS];
        for &c in &self.discard_pile {
            visible[c] = true;
        }
        for &c in &self.buried_cards {
            visible[c] = true;
        }
        for city in &self.cities {
            for &c in &city.buildings {
                visible[c] = true;
            }
        }
        let age_back = age - 1; // AgeI=0, AgeII=1, AgeIII=2
        self.removed_age_cards[age] = (0..crate::data::NUM_CARDS)
            .filter(|&c| {
                crate::data::back_type_of(c) as usize == age_back
                    && !deal.contains(&c)
                    && !visible[c]
            })
            .collect();
        if age == 3 {
            use crate::data::BackType;
            let guilds: Vec<usize> = deal
                .iter()
                .copied()
                .filter(|&c| crate::data::back_type_of(c) == BackType::Guild)
                .collect();
            self.selected_guilds = guilds.clone();
            self.unused_guilds = (0..crate::data::NUM_CARDS)
                .filter(|&c| {
                    crate::data::back_type_of(c) == BackType::Guild && !guilds.contains(&c)
                })
                .collect();
        }
        self.age_decks[age] = deal.to_vec();
    }
}

// --- F1b make/unmake audit ----------------------------------------------------

/// Exhaustive make/unmake audit from `state`, exploring every legal action to
/// `depth` plies as a nested LIFO stack. At each ply and for every sibling:
///   1. undo restores the *complete* prior state (`GameState: PartialEq`, so
///      fields excluded from the cross-language fingerprint — notably
///      `library_draws` — are checked too), and
///   2. a second application reproduces the same post-state (apply determinism;
///      also confirms a consumed Great Library draw was restored, since the
///      re-application must pop the same value).
/// Returns the first violation as a message. Used by `RustGame::roundtrip_all_ok`
/// (F1b) and by the crate's unit tests. Snapshot undo passes by construction;
/// the audit is written to stay load-bearing for a future journaled undo.
pub fn make_unmake_audit(state: &GameState, depth: usize) -> Result<(), String> {
    if depth == 0 {
        return Ok(());
    }
    let before = state.clone();
    for a in crate::codec::legal_action_indices(&before) {
        let mut g = before.clone();
        let undo = g.snapshot();
        g.apply_action(&crate::codec::decode_action(&g, a));
        let after = g.clone();
        make_unmake_audit(&g, depth - 1)?; // descend before undoing: nested LIFO
        g.restore(undo);
        if g != before {
            return Err(format!("undo did not restore full state before action {a}"));
        }
        g.apply_action(&crate::codec::decode_action(&g, a));
        if g != after {
            return Err(format!("re-applying action {a} was non-deterministic"));
        }
    }
    Ok(())
}

/// Exhaustive check that journaled undo restores a state *exactly*.
///
/// The whole risk of a journal is a mutation nobody recorded: the search then
/// runs on a state that is quietly wrong, with no crash and no wrong answer
/// until much later. So this compares the journaled path against the one it
/// replaces -- full-state equality after undo, over every legal action to
/// `depth`, and it walks the same tree the solver would.
///
/// Actions the journal declines are still exercised, through the snapshot path,
/// so the walk covers the real mix rather than only the journaled subset.
pub fn journal_undo_audit(state: &GameState, depth: usize) -> Result<(), String> {
    if depth == 0 {
        return Ok(());
    }
    let before = state.clone();
    for index in crate::codec::legal_action_indices(&before) {
        let action = crate::codec::decode_action(&before, index);
        let specs = crate::chance::chance_signature(&before, &action);
        if specs.iter().any(|s| s.kind == crate::chance::ChanceKind::AgeDeal) {
            continue; // sample-only: the solver refuses these outright
        }

        // Route exactly as the solver does. Applying a chance action through
        // plain `apply_action` would resolve it from the locked deal -- the
        // hidden state a search must not read -- and on an injected position,
        // which carries no preloaded Great Library draw, it panics outright.
        // That is how this audit used to crash on a legal Great Library build
        // rather than report on one.
        if !specs.is_empty() {
            let chains = crate::chance::enumerate_chains_unkeyed(&before, &specs);
            // One outcome per chance edge: enough to reach the states beyond it,
            // where the journaled applies this audit exists for actually happen,
            // without the branching factor making the walk unbounded.
            if let Some((outcomes, _)) = chains.first() {
                let mut g = before.clone();
                let snap = g.snapshot();
                if g.apply_with_chance(&action, outcomes).is_err() {
                    return Err(format!("chance apply of action {index} failed"));
                }
                journal_undo_audit(&g, depth - 1)?;
                g.restore(snap);
                if g != before {
                    return Err(format!("snapshot undo of chance action {index} did not restore"));
                }
            }
            continue;
        }

        let mut g = before.clone();
        // What the state should be afterwards, per the path being replaced.
        let mut reference = before.clone();
        reference.apply_action(&action);

        match g.apply_journaled(&action) {
            Some(undo) => {
                if g != reference {
                    return Err(format!(
                        "journaled apply of action {index} differs from apply_action"
                    ));
                }
                journal_undo_audit(&g, depth - 1)?;
                g.undo(undo);
                if g != before {
                    return Err(format!("journaled undo of action {index} did not restore"));
                }
            }
            None => {
                let snap = g.snapshot();
                g.apply_action(&action);
                journal_undo_audit(&g, depth - 1)?;
                g.restore(snap);
                if g != before {
                    return Err(format!("snapshot undo of action {index} did not restore"));
                }
            }
        }
    }
    Ok(())
}
