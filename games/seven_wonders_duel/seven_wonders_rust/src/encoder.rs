//! Observation encoder (CODEC_SPEC.md §5), F2.2 port of `encoder.py`.
//!
//! Produces the actor-relative typed entity-token sequence. Features are built
//! in **f64**, element-for-element in the same order as `encoder.py`, so the F2
//! gate can compare bit-for-bit. Economy/scoring come from the engine's own
//! helpers (`minimum_payment`, `fixed_production`, `score_player`, …) — the same
//! zero-duplication discipline as Python, run directly on the real `GameState`
//! (its public fields equal the observation's, so the Python "stub state" is
//! unnecessary).

use crate::data::{
    back_type_of, card, progress, progress_id, wonder, wonder_id, CardColor, EffectKind,
    ScienceSymbol, NUM_CARDS, NUM_WONDERS,
};
use crate::engine::{
    choice_producers, fixed_production, minimum_payment, opponent_trade_production,
    trade_discounts,
};
use crate::pool::{unseen_pool, UnseenPool};
use crate::rules::discard_income;
use crate::state::{GameState, PendingChoiceKind, Phase};

const NUM_SYMBOLS: usize = 7;
const NUM_COLORS: usize = 7;
const NUM_RESOURCES: usize = 5;

/// Pinned encoder schema signature — must equal `encoder.py::ENCODER_SIGNATURE`
/// (the F2 gate `test_encoder_signature_matches` asserts it). The F4 checkpoint
/// boundary compares a checkpoint's stored signature against this to reject a
/// net trained on a different feature schema.
pub const ENCODER_SIGNATURE: &str =
    "7d68ff20f280700f0c7a04d2411cded734c51b3e312a80578824d7dbb0098be2";

/// Feature-vector length per token type, in `TokenType` order. The encoder
/// asserts every emitted token matches (debug builds + `cargo test`); the
/// bit-exact gate enforces it in release via the value comparison.
pub const FEATURE_COUNTS: [usize; 9] = [130, 1, 26, 1, 8, 4, 1, 79, 14];

// TokenType indices, in `encoder.py::TokenType` declaration order.
const T_GLOBAL: usize = 0;
const T_DRAFT_OFFER: usize = 1;
const T_TABLEAU: usize = 2;
const T_CITY_CARD: usize = 3;
const T_WONDER: usize = 4;
const T_PROGRESS: usize = 5;
const T_DISCARD: usize = 6;
const T_POOL: usize = 7;
const T_POOL_WONDER: usize = 8;

const YELLOW: usize = CardColor::Yellow as usize;

/// One entity token: `(type_id, entity_id, aux_id, features)`, mirroring
/// `encoder.py::Token` (aux_id = -1 when unused).
pub struct Token {
    pub type_id: usize,
    pub entity_id: i32,
    pub aux_id: i32,
    pub features: Vec<f64>,
}

/// Per-encode derived cache + the state/actor/pool the builders share.
struct Enc<'a> {
    g: &'a GameState,
    actor: usize,
    pool: UnseenPool,
    symbols: [[bool; NUM_SYMBOLS]; 2],
    /// card ids obtainable by either side (revealed board cards + relevant-back
    /// pool), computed once.
    obtainable: Vec<usize>,
}

pub fn encode(g: &GameState) -> Vec<Token> {
    let actor = g
        .pending_choice
        .as_ref()
        .map_or(g.active_player, |p| p.player);
    let pool = unseen_pool(g);
    let obtainable = obtainable_cards(g, &pool);
    let e = Enc {
        g,
        actor,
        pool,
        symbols: [compute_symbols(g, 0), compute_symbols(g, 1)],
        obtainable,
    };
    let mut tokens = Vec::new();
    tokens.push(e.global_token());
    e.draft_offer_tokens(&mut tokens);
    e.tableau_tokens(&mut tokens);
    e.city_card_tokens(&mut tokens);
    e.wonder_tokens(&mut tokens);
    e.progress_tokens(&mut tokens);
    e.discard_tokens(&mut tokens);
    e.pool_tokens(&mut tokens);
    e.pool_wonder_token(&mut tokens);
    debug_assert!(
        tokens.iter().all(|t| t.features.len() == FEATURE_COUNTS[t.type_id]),
        "encoder token feature count disagrees with FEATURE_COUNTS"
    );
    tokens
}

// --- shared derived quantities ------------------------------------------------

fn compute_symbols(g: &GameState, seat: usize) -> [bool; NUM_SYMBOLS] {
    let mut s = [false; NUM_SYMBOLS];
    for &cid in &g.cities[seat].buildings {
        if let Some(sym) = card(cid).science {
            s[sym as usize] = true;
        }
    }
    for &pid in &g.cities[seat].progress_tokens {
        if let Some(sym) = progress(pid).science {
            s[sym as usize] = true;
        }
    }
    s
}

fn relevant_backs(g: &GameState) -> Vec<usize> {
    let mut v: Vec<usize> = ((g.age.max(1) as usize - 1)..=2).collect(); // age backs
    v.push(3); // Guild
    v
}

fn obtainable_cards(g: &GameState, pool: &UnseenPool) -> Vec<usize> {
    let mut names = Vec::new();
    if g.phase != Phase::WonderDraft {
        for slot in &g.tableau.slots {
            if slot.present && slot.revealed {
                names.push(slot.card_id);
            }
        }
    }
    for back in relevant_backs(g) {
        names.extend(pool.cards[back].iter().copied());
    }
    names
}

fn rel_position(g: &GameState, seat: usize) -> i32 {
    if seat == 0 {
        g.conflict_position
    } else {
        -g.conflict_position
    }
}

fn next_token(g: &GameState, seat: usize) -> (i32, i32) {
    let position = rel_position(g, seat);
    let mut best: Option<(i32, i32)> = None;
    for &(absolute, penalty) in &g.military_tokens_remaining {
        let relative = if seat == 0 { absolute } else { -absolute };
        if relative > position {
            let distance = relative - position;
            if best.map_or(true, |b| distance < b.0) {
                best = Some((distance, penalty));
            }
        }
    }
    best.unwrap_or((18, 0))
}

fn tokens_remaining(g: &GameState, seat: usize) -> (f64, f64) {
    let sign = if seat == 0 { 1 } else { -1 };
    let has = |p: i32| g.military_tokens_remaining.iter().any(|&(pos, _)| pos == p);
    (
        if has(sign * 4) { 1.0 } else { 0.0 },
        if has(sign * 7) { 1.0 } else { 0.0 },
    )
}

fn effective_shields(g: &GameState, seat: usize, cid: usize) -> i32 {
    let c = card(cid);
    let mut shields = c.shields;
    if c.color == CardColor::Red
        && g.cities[seat]
            .progress_tokens
            .contains(&progress_id("Strategy"))
    {
        shields += 1;
    }
    shields
}

fn progress_obtainable(g: &GameState, seat: usize, token_name: &str) -> bool {
    let tid = progress_id(token_name);
    if let Some(p) = &g.pending_choice {
        if p.kind == PendingChoiceKind::ChooseUnusedProgress
            && p.player == seat
            && p.options.contains(&tid)
        {
            return true;
        }
    }
    if g.available_progress_tokens.contains(&tid) {
        return true;
    }
    if g.cities[seat].progress_tokens.contains(&tid) {
        return true;
    }
    if g.cities[1 - seat].progress_tokens.contains(&tid) {
        return false;
    }
    let gl = wonder_id("The Great Library");
    g.cities[seat].wonders.contains(&gl)
        && !g.cities[seat].built_wonders.contains(&gl)
        && !g.retired_wonders.contains(&gl)
}

impl Enc<'_> {
    fn military_bound(&self, seat: usize) -> i32 {
        let g = self.g;
        let mut total: i32 = self.obtainable.iter().map(|&cid| card(cid).shields).sum();
        for &wid in &g.cities[seat].wonders {
            if !g.cities[seat].built_wonders.contains(&wid)
                && !g.retired_wonders.contains(&wid)
            {
                total += wonder(wid).shields;
            }
        }
        if progress_obtainable(g, seat, "Strategy") {
            total += self
                .obtainable
                .iter()
                .filter(|&&cid| card(cid).color == CardColor::Red)
                .count() as i32;
        }
        total
    }

    fn science_missing_obtainable(&self, seat: usize) -> i32 {
        let mut obtainable = [false; NUM_SYMBOLS];
        for &cid in &self.obtainable {
            if let Some(s) = card(cid).science {
                obtainable[s as usize] = true;
            }
        }
        if progress_obtainable(self.g, seat, "Law") {
            obtainable[ScienceSymbol::Law as usize] = true;
        }
        (0..NUM_SYMBOLS)
            .filter(|&i| obtainable[i] && !self.symbols[seat][i])
            .count() as i32
    }

    fn unbuilt_wonder_stats(&self, seat: usize) -> (i32, i32) {
        let g = self.g;
        let theology = g.cities[seat]
            .progress_tokens
            .contains(&progress_id("Theology"));
        let (mut unbuilt, mut extra) = (0, 0);
        for &wid in &g.cities[seat].wonders {
            if g.cities[seat].built_wonders.contains(&wid) || g.retired_wonders.contains(&wid)
            {
                continue;
            }
            unbuilt += 1;
            if theology
                || wonder(wid)
                    .effects
                    .iter()
                    .any(|e| e.kind == EffectKind::PlayAgain)
            {
                extra += 1;
            }
        }
        (unbuilt, extra)
    }

    // --- per-player global block ---------------------------------------------

    fn per_player_values(&self, seat: usize) -> Vec<f64> {
        let g = self.g;
        let city = &g.cities[seat];
        let have = &self.symbols[seat];
        let have_count = have.iter().filter(|&&b| b).count() as i32;
        let mut color_counts = [0i32; NUM_COLORS];
        for &cid in &city.buildings {
            color_counts[card(cid).color as usize] += 1;
        }
        let (unbuilt, extra_turn) = self.unbuilt_wonder_stats(seat);
        let fixed = fixed_production(g, seat);
        let choices = choice_producers(g, seat);
        let discounts = trade_discounts(g, seat);
        let opp = opponent_trade_production(g, seat);
        let score = g.score_player(seat);
        let mil_bound = self.military_bound(seat);
        let dist_win = 9 - rel_position(g, seat);
        let sci_missing = self.science_missing_obtainable(seat);
        let (tok2, tok5) = tokens_remaining(g, seat);

        let mut v = Vec::with_capacity(50);
        v.push(city.coins as f64);
        v.push(city.coins as f64 / 10.0);
        v.push(have_count as f64);
        v.push((6 - have_count).max(0) as f64);
        for i in 0..NUM_SYMBOLS {
            v.push(if have[i] { 1.0 } else { 0.0 });
        }
        for i in 0..NUM_COLORS {
            v.push(color_counts[i] as f64);
        }
        v.push(unbuilt as f64);
        v.push(extra_turn as f64);
        for r in 0..NUM_RESOURCES {
            v.push(fixed[r] as f64);
        }
        for r in 0..NUM_RESOURCES {
            let n = choices
                .iter()
                .filter(|grp| grp.iter().any(|&res| res as usize == r))
                .count();
            v.push(n as f64);
        }
        for r in 0..NUM_RESOURCES {
            v.push(if discounts[r] {
                1.0
            } else {
                (2 + opp[r]) as f64
            });
        }
        v.push(discard_income(color_counts[YELLOW]) as f64);
        v.push(score.military as f64);
        v.push(score.buildings as f64);
        v.push(score.guild as f64);
        v.push(score.wonders as f64);
        v.push(score.progress as f64);
        v.push(score.treasury as f64);
        v.push(score.total as f64);
        v.push(score.blue_buildings as f64);
        v.push(tok2);
        v.push(tok5);
        v.push(mil_bound as f64);
        v.push(if mil_bound >= dist_win { 1.0 } else { 0.0 });
        v.push(sci_missing as f64);
        v.push(if have_count + sci_missing >= 6 { 1.0 } else { 0.0 });
        v
    }

    fn global_token(&self) -> Token {
        let g = self.g;
        let decision = decision_index(g);
        let (present, face_down) = if g.phase == Phase::WonderDraft {
            (0i32, 0i32)
        } else {
            let mut present = 0;
            let mut face_down = 0;
            for slot in &g.tableau.slots {
                if slot.present {
                    present += 1;
                    if !slot.revealed {
                        face_down += 1;
                    }
                }
            }
            (present, face_down)
        };
        let military = rel_position(g, self.actor);
        let my_token = next_token(g, self.actor);
        let opp_token = next_token(g, 1 - self.actor);

        let mut v = Vec::new();
        for d in 0..9 {
            v.push(if d == decision { 1.0 } else { 0.0 });
        }
        for a in 1..=3 {
            v.push(if g.age as i32 == a { 1.0 } else { 0.0 });
        }
        v.push(present as f64);
        v.push(present as f64 / 20.0);
        v.push(face_down as f64);
        v.push(face_down as f64 / 10.0);
        v.push(military as f64);
        v.push(military as f64 / 9.0);
        v.push((9 - military) as f64);
        v.push((9 - military) as f64 / 18.0);
        v.push((9 + military) as f64);
        v.push((9 + military) as f64 / 18.0);
        v.push(my_token.0 as f64);
        v.push(my_token.0 as f64 / 18.0);
        v.push(my_token.1 as f64);
        v.push(opp_token.0 as f64);
        v.push(opp_token.0 as f64 / 18.0);
        v.push(opp_token.1 as f64);
        v.push(g.pending_shields as f64);
        v.push(if g.pending_extra_turn { 1.0 } else { 0.0 });
        v.extend(self.per_player_values(self.actor));
        v.extend(self.per_player_values(1 - self.actor));
        Token {
            type_id: T_GLOBAL,
            entity_id: 0,
            aux_id: -1,
            features: v,
        }
    }

    // --- tableau -------------------------------------------------------------

    fn tableau_card_per_player(&self, seat: usize, cid: usize) -> Vec<f64> {
        let g = self.g;
        let c = card(cid);
        let payment = minimum_payment(g, seat, &c.cost, Some(c), false);
        let cost = payment.total_coins;
        let affordable = g.cities[seat].coins >= cost;
        let have = &self.symbols[seat];
        let completes_pair = match c.science {
            Some(s) => {
                have[s as usize] && !g.cities[seat].claimed_science_pairs.contains(&s)
            }
            None => false,
        };
        let gives_sixth = match c.science {
            Some(s) => !have[s as usize] && self.symbols[seat].iter().filter(|&&b| b).count() + 1 >= 6,
            None => false,
        };
        let shields = effective_shields(g, seat, cid);
        let next_dist = next_token(g, seat).0;
        let dist_win = 9 - rel_position(g, seat);
        vec![
            if affordable { 1.0 } else { 0.0 },
            cost as f64,
            cost as f64 / 10.0,
            if payment.used_chain { 1.0 } else { 0.0 },
            if completes_pair { 1.0 } else { 0.0 },
            if gives_sixth { 1.0 } else { 0.0 },
            shields as f64,
            if shields >= next_dist { 1.0 } else { 0.0 },
            if shields >= dist_win { 1.0 } else { 0.0 },
        ]
    }

    fn tableau_tokens(&self, out: &mut Vec<Token>) {
        let g = self.g;
        if g.phase == Phase::WonderDraft {
            return; // observation tableau is empty during the draft
        }
        // Present slots as (row, x, slot index), sorted by (row, x).
        let mut present: Vec<(i32, i32, usize)> = g
            .tableau
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.present)
            .map(|(i, _)| {
                let (row, x) = g.tableau.slot_id(i);
                (row, x, i)
            })
            .collect();
        present.sort_by_key(|&(row, x, _)| (row, x));

        for &(row, x, i) in &present {
            let slot_card = &g.tableau.slots[i];
            let coverers = present
                .iter()
                .filter(|&&(orow, ox, _)| orow == row + 1 && (ox - x).abs() == 1)
                .count();
            let covers_hidden = present
                .iter()
                .filter(|&&(orow, ox, oi)| {
                    orow == row - 1 && (ox - x).abs() == 1 && !g.tableau.slots[oi].revealed
                })
                .count();
            let face_up = crate::data::layout(g.age)[i].face_up;
            let accessible = g.tableau.is_accessible(i);

            let mut v = vec![
                row as f64,
                row as f64 / 6.0,
                x as f64,
                x as f64 / 11.0,
                if face_up { 1.0 } else { 0.0 },
                if accessible { 1.0 } else { 0.0 },
                coverers as f64,
                covers_hidden as f64,
            ];
            let entity;
            if slot_card.revealed {
                entity = slot_card.card_id as i32;
                v.extend(self.tableau_card_per_player(self.actor, slot_card.card_id));
                v.extend(self.tableau_card_per_player(1 - self.actor, slot_card.card_id));
            } else {
                entity = 73 + back_type_of(slot_card.card_id) as i32;
                v.extend(std::iter::repeat(0.0).take(2 * 9));
            }
            out.push(Token {
                type_id: T_TABLEAU,
                entity_id: entity,
                aux_id: -1,
                features: v,
            });
        }
    }

    // --- remaining token types -----------------------------------------------

    fn draft_offer_tokens(&self, out: &mut Vec<Token>) {
        let g = self.g;
        if g.phase != Phase::WonderDraft {
            return;
        }
        let picked: usize = g.cities.iter().map(|c| c.wonders.len()).sum();
        let second_round = if picked >= 4 { 1.0 } else { 0.0 };
        let mut offer = g.wonder_offer.clone();
        offer.sort_unstable(); // wonder ids == WONDER_IDS order
        for wid in offer {
            out.push(Token {
                type_id: T_DRAFT_OFFER,
                entity_id: wid as i32,
                aux_id: -1,
                features: vec![second_round],
            });
        }
    }

    fn city_card_tokens(&self, out: &mut Vec<Token>) {
        for (mine, seat) in [(1.0, self.actor), (0.0, 1 - self.actor)] {
            for &cid in &self.g.cities[seat].buildings {
                out.push(Token {
                    type_id: T_CITY_CARD,
                    entity_id: cid as i32,
                    aux_id: -1,
                    features: vec![mine],
                });
            }
        }
    }

    fn wonder_tokens(&self, out: &mut Vec<Token>) {
        let g = self.g;
        for (mine, seat) in [(1.0, self.actor), (0.0, 1 - self.actor)] {
            let city = &g.cities[seat];
            let theology = city.progress_tokens.contains(&progress_id("Theology"));
            for &wid in &city.wonders {
                let w = wonder(wid);
                let built = city.built_wonders.contains(&wid);
                let retired = g.retired_wonders.contains(&wid);
                let (affordable, cost) = if built || retired {
                    (0.0, 0)
                } else {
                    let payment =
                        minimum_payment(g, seat, w.cost.as_ref().expect("wonder cost"), None, true);
                    (
                        if city.coins >= payment.total_coins { 1.0 } else { 0.0 },
                        payment.total_coins,
                    )
                };
                let grants_extra =
                    theology || w.effects.iter().any(|e| e.kind == EffectKind::PlayAgain);
                let aux_id = g
                    .wonder_burials
                    .iter()
                    .find(|&&(bw, _)| bw == wid)
                    .map_or(-1, |&(_, cid)| cid as i32);
                out.push(Token {
                    type_id: T_WONDER,
                    entity_id: wid as i32,
                    aux_id,
                    features: vec![
                        mine,
                        if built { 1.0 } else { 0.0 },
                        if retired { 1.0 } else { 0.0 },
                        affordable,
                        cost as f64,
                        cost as f64 / 10.0,
                        if grants_extra { 1.0 } else { 0.0 },
                        w.shields as f64,
                    ],
                });
            }
        }
    }

    fn progress_tokens(&self, out: &mut Vec<Token>) {
        let g = self.g;
        for &pid in &g.available_progress_tokens {
            out.push(Token {
                type_id: T_PROGRESS,
                entity_id: pid as i32,
                aux_id: -1,
                features: vec![1.0, 0.0, 0.0, 0.0],
            });
        }
        for (mine, seat) in [(1.0, self.actor), (0.0, 1 - self.actor)] {
            for &pid in &g.cities[seat].progress_tokens {
                out.push(Token {
                    type_id: T_PROGRESS,
                    entity_id: pid as i32,
                    aux_id: -1,
                    features: vec![0.0, mine, 1.0 - mine, 0.0],
                });
            }
        }
        if let Some(p) = &g.pending_choice {
            if p.kind == PendingChoiceKind::ChooseUnusedProgress {
                let mut candidates = p.options.clone();
                candidates.sort_unstable(); // progress ids == PROGRESS_IDS order
                for pid in candidates {
                    out.push(Token {
                        type_id: T_PROGRESS,
                        entity_id: pid as i32,
                        aux_id: -1,
                        features: vec![0.0, 0.0, 0.0, 1.0],
                    });
                }
            }
        }
    }

    fn discard_tokens(&self, out: &mut Vec<Token>) {
        let g = self.g;
        let mausoleum = g.pending_choice.as_ref().map_or(false, |p| {
            p.kind == PendingChoiceKind::BuildFromDiscardFree
        });
        for &cid in &g.discard_pile {
            out.push(Token {
                type_id: T_DISCARD,
                entity_id: cid as i32,
                aux_id: -1,
                features: vec![if mausoleum { 1.0 } else { 0.0 }],
            });
        }
    }

    fn pool_tokens(&self, out: &mut Vec<Token>) {
        for back in 0..4 {
            let members = &self.pool.cards[back];
            if members.is_empty() {
                continue;
            }
            let mut is_member = [false; NUM_CARDS];
            for &cid in members {
                is_member[cid] = true;
            }
            let my_costs: Vec<i32> = members
                .iter()
                .map(|&cid| {
                    minimum_payment(self.g, self.actor, &card(cid).cost, Some(card(cid)), false)
                        .total_coins
                })
                .collect();
            let opp_costs: Vec<i32> = members
                .iter()
                .map(|&cid| {
                    minimum_payment(
                        self.g,
                        1 - self.actor,
                        &card(cid).cost,
                        Some(card(cid)),
                        false,
                    )
                    .total_coins
                })
                .collect();
            let n = members.len();
            let sum_my: i32 = my_costs.iter().sum();
            let sum_opp: i32 = opp_costs.iter().sum();
            let mut v = vec![
                n as f64,
                n as f64 / 23.0,
                sum_my as f64 / n as f64,
                *my_costs.iter().min().unwrap() as f64,
                sum_opp as f64 / n as f64,
                *opp_costs.iter().min().unwrap() as f64,
            ];
            for cid in 0..NUM_CARDS {
                v.push(if is_member[cid] { 1.0 } else { 0.0 });
            }
            out.push(Token {
                type_id: T_POOL,
                entity_id: back as i32,
                aux_id: -1,
                features: v,
            });
        }
    }

    fn pool_wonder_token(&self, out: &mut Vec<Token>) {
        let g = self.g;
        if g.phase != Phase::WonderDraft {
            return;
        }
        let picked: usize = g.cities.iter().map(|c| c.wonders.len()).sum();
        if picked >= 4 {
            return; // second group is face-up; the unseen four never matter
        }
        let mut is_member = [false; NUM_WONDERS];
        for &wid in &self.pool.wonders {
            is_member[wid] = true;
        }
        let n = self.pool.wonders.len();
        let mut v = vec![n as f64, n as f64 / 12.0];
        for wid in 0..NUM_WONDERS {
            v.push(if is_member[wid] { 1.0 } else { 0.0 });
        }
        out.push(Token {
            type_id: T_POOL_WONDER,
            entity_id: 0,
            aux_id: -1,
            features: v,
        });
    }
}

fn decision_index(g: &GameState) -> usize {
    if let Some(p) = &g.pending_choice {
        return match p.kind {
            PendingChoiceKind::DestroyOpponentBrown => 2,
            PendingChoiceKind::DestroyOpponentGrey => 3,
            PendingChoiceKind::BuildFromDiscardFree => 4,
            PendingChoiceKind::ChooseUnusedProgress => 5,
            PendingChoiceKind::ChooseAvailableProgress => 6,
        };
    }
    match g.phase {
        Phase::WonderDraft => 0,
        Phase::ChooseNextStartPlayer => 7,
        Phase::Complete => 8,
        Phase::PlayAge => 1,
    }
}
