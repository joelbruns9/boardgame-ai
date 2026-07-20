//! Action resolution, victory conditions, and scoring — a port of `engine.py`.
//!
//! Chance resolution follows the `buffer.replay` simulator path (see
//! `state.rs`): reveals, age deals, and the wonder-group flip resolve from
//! locked state; only the Great Library draw consumes a recorded outcome.

use crate::data::{
    self, card, progress, progress_id, wonder, CardColor, Cost, EffectKind, Resource,
    ScienceSymbol,
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
struct Payment {
    total_coins: i32,
    trade_coins: i32,
    used_chain: bool,
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
    g.cities[player].progress_tokens.contains(&progress_id(name))
}

fn fixed_production(g: &GameState, player: usize) -> [i32; 5] {
    let mut out = [0i32; 5];
    for &cid in &g.cities[player].buildings {
        for &r in card(cid).fixed_production {
            out[r as usize] += 1;
        }
    }
    out
}

fn choice_producers(g: &GameState, player: usize) -> Vec<&'static [Resource]> {
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

fn opponent_trade_production(g: &GameState, player: usize) -> [i32; 5] {
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

fn trade_discounts(g: &GameState, player: usize) -> [bool; 5] {
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
fn minimum_payment(
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

fn unbuilt_wonders(g: &GameState, player: usize) -> Vec<usize> {
    g.cities[player]
        .wonders
        .iter()
        .copied()
        .filter(|w| {
            !g.cities[player].built_wonders.contains(w) && !g.retired_wonders.contains(w)
        })
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
                self.buried_cards.push(card_id);
                self.wonder_burials.push((wid, card_id));
                self.cities[player].built_wonders.push(wid);

                let total_built: usize =
                    self.cities.iter().map(|c| c.built_wonders.len()).sum();
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
        let (_card_id, newly) = self.tableau.take_accessible(slot);
        // Simulator reveal path: the locked card is already correct, so each
        // newly-accessible slot is simply revealed in (row, x) order.
        for j in newly {
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
            if let Some(pos) = self
                .military_tokens_remaining
                .iter()
                .position(|&(p, _)| p == self.conflict_position)
            {
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
                let opp = &mut self.cities[1 - player];
                let pos = opp
                    .buildings
                    .iter()
                    .position(|&c| c == choice)
                    .expect("destroy target not present");
                opp.buildings.remove(pos);
                self.discard_pile.push(choice);
            }
            PendingChoiceKind::BuildFromDiscardFree => {
                let pos = self
                    .discard_pile
                    .iter()
                    .position(|&c| c == choice)
                    .expect("revive target not in discard");
                self.discard_pile.remove(pos);
                self.cities[player].buildings.push(choice);
                self.after_building_constructed(player, choice);
            }
            PendingChoiceKind::ChooseUnusedProgress
            | PendingChoiceKind::ChooseAvailableProgress => {
                self.cities[player].progress_tokens.push(choice);
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

    pub fn start_next_age(&mut self, starting_player: usize) {
        assert_eq!(
            self.phase,
            Phase::ChooseNextStartPlayer,
            "current age not complete"
        );
        assert!(starting_player < 2, "starting player must be 0 or 1");
        self.age += 1;
        let deck = self.age_decks[self.age as usize].clone();
        self.tableau = crate::state::TableauState::from_deck(self.age, &deck);
        self.active_player = starting_player;
        self.phase = Phase::PlayAge;
    }

    // --- scoring / endgame -----------------------------------------------------

    fn military_victory_points(&self, player: usize) -> i32 {
        let position = self.conflict_position;
        if position == 0 || (position > 0) != (player == 0) {
            return 0;
        }
        let distance = position.abs();
        if distance <= 3 {
            2
        } else if distance <= 6 {
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

    /// (total, blue_buildings) — the two quantities the civilian tiebreak needs.
    fn score_totals(&self, player: usize) -> (i32, i32) {
        let city = &self.cities[player];
        let military = self.military_victory_points(player);
        let guild = self.guild_victory_points(player);
        let buildings: i32 =
            city.buildings.iter().map(|&c| card(c).victory_points).sum::<i32>() + guild;
        let wonders: i32 = city.built_wonders.iter().map(|&w| wonder(w).victory_points).sum();
        let mut progress_vp: i32 =
            city.progress_tokens.iter().map(|&p| progress(p).victory_points).sum();
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
        (total, blue)
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
