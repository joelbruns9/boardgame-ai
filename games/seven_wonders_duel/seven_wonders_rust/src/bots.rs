//! Rust ports of the Phase-D curriculum and anchor bots.

use crate::codec::encode_action;
use crate::data::{self, CardColor, EffectKind, Resource};
use crate::engine::{Action, ActionUse};
use crate::rng::Rng;
use crate::state::{GameState, PendingChoiceKind, Phase, VictoryType};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BotKind {
    Greedy,
    ScienceAggressive,
    ScienceEconomy,
    MilitaryAggressive,
    MilitaryEconomy,
}

impl BotKind {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "greedy" => Some(Self::Greedy),
            "science_aggressive/v1" => Some(Self::ScienceAggressive),
            "science_economy/v1" => Some(Self::ScienceEconomy),
            "military_aggressive/v1" => Some(Self::MilitaryAggressive),
            "military_economy/v1" => Some(Self::MilitaryEconomy),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::ScienceAggressive => "science_aggressive/v1",
            Self::ScienceEconomy => "science_economy/v1",
            Self::MilitaryAggressive => "military_aggressive/v1",
            Self::MilitaryEconomy => "military_economy/v1",
        }
    }
}

fn actor(g: &GameState) -> usize {
    g.pending_choice
        .as_ref()
        .map_or(g.active_player, |choice| choice.player)
}

fn science_symbols(g: &GameState, player: usize) -> [bool; 7] {
    let mut symbols = [false; 7];
    for &cid in &g.cities[player].buildings {
        if let Some(symbol) = data::card(cid).science {
            symbols[symbol as usize] = true;
        }
    }
    for &pid in &g.cities[player].progress_tokens {
        if let Some(symbol) = data::progress(pid).science {
            symbols[symbol as usize] = true;
        }
    }
    symbols
}

fn science_count(g: &GameState, player: usize) -> usize {
    science_symbols(g, player)
        .into_iter()
        .filter(|&x| x)
        .count()
}

fn economy_value(g: &GameState, player: usize) -> f64 {
    let mut value = 0.0;
    for &cid in &g.cities[player].buildings {
        let card = data::card(cid);
        value += 0.7 * card.fixed_production.len() as f64;
        if !card.choice_production.is_empty() {
            value += 0.8 + 0.2 * (card.choice_production.len() - 1) as f64;
        }
        value += 0.7 * card.trade_discount.len() as f64;
    }
    for &wid in &g.cities[player].built_wonders {
        let choices = data::wonder(wid).choice_production;
        if !choices.is_empty() {
            value += 0.8 + 0.2 * (choices.len() - 1) as f64;
        }
    }
    value
}

fn unbuilt_wonder_value(g: &GameState, player: usize) -> f64 {
    let city = &g.cities[player];
    let mut value = 0.0;
    for &wid in &city.wonders {
        if city.built_wonders.contains(&wid) || g.retired_wonders.contains(&wid) {
            continue;
        }
        let wonder = data::wonder(wid);
        value += 0.25 * wonder.victory_points as f64 + 0.75 * wonder.shields as f64;
        if !wonder.choice_production.is_empty() {
            value += 0.6;
        }
        for effect in wonder.effects {
            value += match effect.kind {
                EffectKind::PlayAgain => 1.2,
                EffectKind::DestroyOpponentBrown
                | EffectKind::DestroyOpponentGrey
                | EffectKind::BuildFromDiscardFree
                | EffectKind::ChooseUnusedProgress => 0.8,
                EffectKind::ImmediateCoins => 0.06 * effect.amount as f64,
                _ => 0.0,
            };
        }
    }
    value
}

fn evaluate_state(g: &GameState, player: usize) -> f64 {
    let opponent = 1 - player;
    if g.phase == Phase::Complete {
        return if g.winner == Some(player) {
            1_000_000.0
        } else if g.winner == Some(opponent) {
            -1_000_000.0
        } else {
            0.0
        };
    }
    let scores = (g.score_player(player).total, g.score_player(opponent).total);
    let military = if player == 0 {
        g.conflict_position
    } else {
        -g.conflict_position
    };
    let mut value = 8.0 * (scores.0 - scores.1) as f64;
    value += 0.2 * (g.cities[player].coins - g.cities[opponent].coins) as f64;
    value += 2.0 * military as f64;
    value += 4.0 * (science_count(g, player) as i32 - science_count(g, opponent) as i32) as f64;
    value += 1.2 * (economy_value(g, player) - economy_value(g, opponent));
    value += 0.5 * (unbuilt_wonder_value(g, player) - unbuilt_wonder_value(g, opponent));
    if g.active_player == player {
        value += 0.1;
    }
    value
}

fn use_rank(use_: ActionUse) -> u8 {
    match use_ {
        ActionUse::ChooseNextStartPlayer => 0,
        ActionUse::ConstructBuilding => 1,
        ActionUse::ConstructWonder => 2,
        ActionUse::DiscardForCoins => 3,
        ActionUse::DraftWonder => 4,
        ActionUse::ResolvePendingChoice => 5,
    }
}

fn choice_name(g: &GameState, action: &Action) -> &'static str {
    let Some(choice) = action.choice else {
        return "";
    };
    match g.pending_choice.as_ref().map(|pending| pending.kind) {
        Some(PendingChoiceKind::DestroyOpponentBrown)
        | Some(PendingChoiceKind::DestroyOpponentGrey)
        | Some(PendingChoiceKind::BuildFromDiscardFree) => data::card(choice).name,
        Some(PendingChoiceKind::ChooseUnusedProgress)
        | Some(PendingChoiceKind::ChooseAvailableProgress) => data::progress(choice).name,
        None => "",
    }
}

fn action_key(g: &GameState, action: &Action) -> (u8, (i32, i32), &'static str, &'static str, i32) {
    (
        use_rank(action.use_),
        action.slot.map_or((-1, -1), |slot| g.tableau.slot_id(slot)),
        action.wonder.map_or("", |wid| data::wonder(wid).name),
        choice_name(g, action),
        action.starting_player.map_or(-1, |seat| seat as i32),
    )
}

fn resource_weight(resource: Resource, science: bool) -> f64 {
    match (science, resource) {
        (true, Resource::Wood) => 1.1,
        (true, Resource::Clay) => 0.8,
        (true, Resource::Stone) => 1.0,
        (true, Resource::Glass | Resource::Papyrus) => 1.5,
        (false, Resource::Wood | Resource::Clay | Resource::Stone) => 1.4,
        (false, Resource::Glass) => 0.7,
        (false, Resource::Papyrus) => 0.8,
    }
}

fn resource_support_value(g: &GameState, player: usize, science: bool) -> f64 {
    let mut value = 0.0;
    for &cid in &g.cities[player].buildings {
        let card = data::card(cid);
        value += card
            .fixed_production
            .iter()
            .map(|&r| resource_weight(r, science))
            .sum::<f64>();
        if !card.choice_production.is_empty() {
            value += card
                .choice_production
                .iter()
                .map(|&r| resource_weight(r, science))
                .fold(f64::NEG_INFINITY, f64::max);
        }
        value += 0.8
            * card
                .trade_discount
                .iter()
                .map(|&r| resource_weight(r, science))
                .sum::<f64>();
    }
    for &wid in &g.cities[player].built_wonders {
        let choices = data::wonder(wid).choice_production;
        if !choices.is_empty() {
            value += choices
                .iter()
                .map(|&r| resource_weight(r, science))
                .fold(f64::NEG_INFINITY, f64::max);
        }
    }
    value
}

fn progress_is_off_board(g: &GameState, name: &str) -> bool {
    let pid = data::progress_id(name);
    !g.available_progress_tokens.contains(&pid)
        && !g
            .cities
            .iter()
            .any(|city| city.progress_tokens.contains(&pid))
}

fn wonder_value(name: &str, science: bool) -> f64 {
    match (science, name) {
        (true, "The Great Library") => 420.0,
        (true, "The Mausoleum") => 300.0,
        (true, "Piraeus") => 170.0,
        (true, "The Great Lighthouse") => 150.0,
        (true, "The Hanging Gardens" | "The Sphinx") => 130.0,
        (true, "The Temple of Artemis") => 120.0,
        (true, "The Appian Way") => 100.0,
        (false, "The Colossus") => 440.0,
        (false, "The Statue of Zeus") => 360.0,
        (false, "Circus Maximus") => 340.0,
        (false, "The Appian Way") => 160.0,
        (false, "The Hanging Gardens" | "The Sphinx") => 130.0,
        (false, "Piraeus") => 120.0,
        (false, "The Temple of Artemis") => 110.0,
        _ => 0.0,
    }
}

fn science_focus(
    g: &GameState,
    child: &GameState,
    action: &Action,
    player: usize,
    economy: bool,
) -> f64 {
    let before = science_symbols(g, player);
    let after = science_symbols(child, player);
    let before_count = before.into_iter().filter(|&x| x).count();
    let new_symbols = after
        .iter()
        .zip(before)
        .filter(|(after, before)| **after && !*before)
        .count();
    let pair_delta = child.cities[player].claimed_science_pairs.len() as i32
        - g.cities[player].claimed_science_pairs.len() as i32;
    let mut score = new_symbols as f64 * (700.0 + 140.0 * before_count as f64);
    if pair_delta != 0 {
        let law = data::progress_id("Law");
        let law_live = g.available_progress_tokens.contains(&law)
            && !g.cities[player].progress_tokens.contains(&law);
        score += pair_delta as f64 * if law_live { 900.0 } else { 260.0 };
    }
    if let Some(slot) = action.slot {
        let card = data::card(g.tableau.slots[slot].card_id);
        if card.color == CardColor::Green {
            if action.use_ == ActionUse::ConstructBuilding {
                score += 100.0;
                if card.age == 1 && card.science.is_some_and(|s| !before[s as usize]) {
                    score += 1_100.0;
                }
            } else if matches!(
                action.use_,
                ActionUse::DiscardForCoins | ActionUse::ConstructWonder
            ) {
                score -= if card.age == 1 { 900.0 } else { 350.0 };
            }
        }
    }
    if let Some(wid) = action.wonder {
        let name = data::wonder(wid).name;
        score += wonder_value(name, true);
        if name == "The Great Library" && progress_is_off_board(g, "Law") {
            score += 280.0;
        }
        if name == "The Mausoleum" {
            score += 80.0
                * g.discard_pile
                    .iter()
                    .filter(|&&cid| data::card(cid).color == CardColor::Green)
                    .count() as f64;
        }
        if economy && matches!(name, "Piraeus" | "The Great Lighthouse") {
            score += 130.0;
        }
    }
    if action.use_ == ActionUse::ResolvePendingChoice {
        let name = choice_name(g, action);
        score += match name {
            "Law" => 1_200.0,
            "Urbanism" | "Agriculture" => 180.0,
            "Economy" => 100.0,
            _ => 0.0,
        };
    }
    let support =
        resource_support_value(child, player, true) - resource_support_value(g, player, true);
    let coins = (child.cities[player].coins - g.cities[player].coins) as f64;
    score + if economy { 55.0 } else { 8.0 } * support + if economy { 1.8 } else { 0.4 } * coins
}

fn military_pressure(position: i32) -> f64 {
    25.0 * position as f64 + 8.0 * (position * position.abs()) as f64
}

fn military_focus(
    g: &GameState,
    child: &GameState,
    action: &Action,
    player: usize,
    economy: bool,
) -> f64 {
    let relative = |state: &GameState| {
        if player == 0 {
            state.conflict_position
        } else {
            -state.conflict_position
        }
    };
    let mut score = 3.0 * (military_pressure(relative(child)) - military_pressure(relative(g)));
    if let Some(slot) = action.slot {
        let card = data::card(g.tableau.slots[slot].card_id);
        if card.color == CardColor::Red {
            let amount = if card.age == 1 { 650.0 } else { 300.0 };
            score += if action.use_ == ActionUse::ConstructBuilding {
                amount
            } else if matches!(
                action.use_,
                ActionUse::DiscardForCoins | ActionUse::ConstructWonder
            ) {
                -amount
            } else {
                0.0
            };
        }
    }
    if let Some(wid) = action.wonder {
        let name = data::wonder(wid).name;
        score += wonder_value(name, false);
        if economy && matches!(name, "Piraeus" | "The Great Lighthouse") {
            score += 90.0;
        }
    }
    if action.use_ == ActionUse::ResolvePendingChoice {
        score += match choice_name(g, action) {
            "Strategy" => 1_200.0,
            "Urbanism" | "Agriculture" => 170.0,
            "Economy" => 100.0,
            _ => 0.0,
        };
    }
    let opponent = 1 - player;
    score += 8.0 * (g.cities[opponent].coins - child.cities[opponent].coins).max(0) as f64;
    let support =
        resource_support_value(child, player, false) - resource_support_value(g, player, false);
    let coins = (child.cities[player].coins - g.cities[player].coins) as f64;
    score + if economy { 55.0 } else { 8.0 } * support + if economy { 1.6 } else { 0.3 } * coins
}

fn rush_score(g: &GameState, action: &Action, kind: BotKind) -> f64 {
    let player = actor(g);
    let mut child = g.clone();
    child.apply_action(action);
    let science = matches!(kind, BotKind::ScienceAggressive | BotKind::ScienceEconomy);
    if child.phase == Phase::Complete {
        return if child.winner == Some(player) {
            if child.victory_type
                == Some(if science {
                    VictoryType::Scientific
                } else {
                    VictoryType::Military
                })
            {
                3_000_000.0
            } else {
                2_000_000.0
            }
        } else if child.winner.is_none() {
            0.0
        } else {
            -3_000_000.0
        };
    }
    let economy = matches!(kind, BotKind::ScienceEconomy | BotKind::MilitaryEconomy);
    let mut score = 0.02 * evaluate_state(&child, player);
    score += if science {
        science_focus(g, &child, action, player, economy)
    } else {
        military_focus(g, &child, action, player, economy)
    };
    if actor(&child) == player {
        score += 25.0;
    }
    if action.use_ == ActionUse::ChooseNextStartPlayer && action.starting_player == Some(player) {
        score += 15.0;
    }
    score
}

pub fn select_action(g: &GameState, kind: BotKind, rng: &mut Rng, exploration: f64) -> usize {
    let player = actor(g);
    let mut scored: Vec<(f64, _, Action)> = g
        .legal_actions()
        .into_iter()
        .map(|action| {
            let score = if kind == BotKind::Greedy {
                let mut child = g.clone();
                child.apply_action(&action);
                evaluate_state(&child, player)
            } else {
                rush_score(g, &action, kind)
            };
            (score, action_key(g, &action), action)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.1.cmp(&a.1))
    });
    let selected = if kind != BotKind::Greedy && exploration > 0.0 && rng.next_float() < exploration
    {
        let limit = scored.len().min(3);
        &scored[rng.randrange(limit as u64) as usize].2
    } else {
        &scored[0].2
    };
    encode_action(g, selected)
}
