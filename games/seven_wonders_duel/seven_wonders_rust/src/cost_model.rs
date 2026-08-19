//! Predicting what an exact endgame solve will cost, before paying for it.
//!
//! The trigger this replaces was `cards_left <= cap`, which cannot express the
//! thing that actually decides cost. Fitted on 1,955 strong-play endgames, the
//! largest term is `chance_fanout` (+1.04) and `cards_left` is only third
//! (+0.27): what makes a subtree expensive is how much of the board is still
//! face down, because that is what turns a minimax into an expectimax. Two
//! 10-card positions differ by an order of magnitude depending on it, and a card
//! cap is blind to the difference.
//!
//! Measured on held-out positions at a 4.5M budget, this attempts 20% of
//! 11-card positions and skips 4% of 8-card ones, buying the same 805 proofs as
//! `cap <= 10` for 44% of the nodes.
//!
//! **Every feature here must match `endgame_trigger_study.position_features`
//! exactly.** The model's coefficients were fit against those definitions, so a
//! divergence does not raise -- it silently prices positions with the wrong
//! weights. `test_cost_model_parity.py` checks all twenty against Python on real
//! positions, and that test is the reason this file can be trusted.

use crate::data::{back_type_of, wonder};
use crate::engine::minimum_payment;
use crate::pool::unseen_pool;
use crate::state::{GameState, Phase};

/// Feature order. Must equal `endgame_cost_model.json`'s `features`, which is
/// `validate_cost_trigger.RUST_FEATURES`.
pub const FEATURE_NAMES: [&str; 20] = [
    "cards_left",
    "unrevealed",
    "accessible",
    "legal",
    "unbuilt_wonders",
    "chance_wonders",
    "revive_wonders",
    "discard",
    "coins",
    "military",
    "vp_gap",
    "science_max",
    "science_threat",
    "military_to_win",
    "chance_fanout",
    "chance_fanout_max",
    "pending_options",
    "affordable_wonders",
    "cards_x_logleg",
    "discard_x_revive",
];

/// The fitted model, installed from Python so `endgame_cost_model.json` stays
/// the single source of truth. `None` leaves the card cap in charge.
#[derive(Clone, Debug)]
pub struct CostModel {
    pub intercept: f64,
    pub weights: [f64; 20],
    pub margin_decades: f64,
}

impl CostModel {
    /// Predicted `log10(nodes)` for a solve at this position.
    pub fn predict(&self, features: &[f64; 20]) -> f64 {
        self.intercept
            + features
                .iter()
                .zip(self.weights.iter())
                .map(|(x, w)| x * w)
                .sum::<f64>()
    }

    /// Is this position predicted to fit `budget`, with the safety margin?
    ///
    /// The margin is not decoration: the fit underpredicts roughly half of the
    /// positions that exhausted the study budget, because cost is long-tailed
    /// and the tail is censored.
    ///
    /// It is not set from the residual p90 (~0.8 decades), and should not be.
    /// The shipped 0.4 was chosen because it was MEASURED to buy the same 805
    /// proofs as `max_cards 10` for 44% of the nodes on held-out cloud
    /// endgames. That is defensible because the loss is bounded on both sides:
    /// an underprediction costs at most `budget` nodes and then declines, an
    /// overprediction costs one proof. A p90 margin would be the right choice
    /// only if an underprediction could run away -- which is exactly what a
    /// binding WALL CLOCK used to allow, and why the clock must stay slack
    /// against the node budget (`PRE_RETRAIN_PLAN.md` section 7).
    ///
    /// Only `budget` and `margin_decades` TOGETHER matter here, since the test
    /// is a comparison of their difference; the budget's separate job is
    /// deciding when an in-flight solve is abandoned.
    pub fn affordable(&self, features: &[f64; 20], budget: u64) -> bool {
        if budget == 0 {
            return false;
        }
        self.predict(features) + self.margin_decades <= (budget as f64).log10()
    }
}

fn unbuilt_named(state: &GameState, name: &str) -> usize {
    let target = crate::data::wonder_id(name);
    (0..2)
        .filter(|&seat| {
            let city = &state.cities[seat];
            city.wonders.contains(&target)
                && !city.built_wonders.contains(&target)
                && !state.retired_wonders.contains(&target)
        })
        .count()
}

/// The twenty features, in `FEATURE_NAMES` order.
///
/// O(board) by construction: this runs before every candidate solve, so it must
/// cost nothing next to the millions of nodes it is deciding about.
pub fn features(state: &GameState) -> [f64; 20] {
    let present: Vec<usize> = state
        .tableau
        .slots
        .iter()
        .enumerate()
        .filter(|(_, slot)| slot.present)
        .map(|(index, _)| index)
        .collect();
    let unrevealed = present
        .iter()
        .filter(|&&index| !state.tableau.slots[index].revealed)
        .count();
    let legal = crate::codec::legal_action_indices(state).len();
    let log_legal = (legal.max(1) as f64).log10();

    let unbuilt_wonders: usize = (0..2)
        .map(|seat| state.cities[seat].wonders.len() - state.cities[seat].built_wonders.len())
        .sum();
    let revive = unbuilt_named(state, "The Mausoleum");

    let military = state.conflict_position.abs();
    let vp_gap = (state.score_player(0).total - state.score_player(1).total).abs();
    let science_max = (0..2)
        .map(|seat| state.science_symbols(seat).len())
        .max()
        .unwrap_or(0);

    // The real expectimax fan-out: a CardReveal enumerates one outcome per
    // unseen card of that back, and a take that exposes several slots at once
    // multiplies them. Summing logs is the form that matches log(nodes) ~ sum of
    // log(branching), which is what the model fits.
    let pool = unseen_pool(state);
    let mut chance_fanout = 0.0_f64;
    let mut chance_fanout_max = 0_usize;
    for &index in &present {
        let slot = &state.tableau.slots[index];
        if slot.revealed {
            continue;
        }
        let size = pool.cards[back_type_of(slot.card_id) as usize].len();
        chance_fanout += (size.max(1) as f64).log10();
        chance_fanout_max = chance_fanout_max.max(size);
    }

    let pending_options = state
        .pending_choice
        .as_ref()
        .map(|choice| choice.options.len())
        .unwrap_or(0);

    let affordable_wonders = (0..2)
        .map(|seat| {
            let city = &state.cities[seat];
            city.wonders
                .iter()
                .filter(|&&wid| {
                    !city.built_wonders.contains(&wid)
                        && !state.retired_wonders.contains(&wid)
                        && minimum_payment(
                            state,
                            seat,
                            &wonder(wid).cost.expect("wonder missing cost"),
                            None,
                            true,
                        )
                        .total_coins
                            <= city.coins
                })
                .count()
        })
        .sum::<usize>();

    [
        present.len() as f64,
        unrevealed as f64,
        state.tableau.accessible_indices().len() as f64,
        legal as f64,
        unbuilt_wonders as f64,
        unbuilt_named(state, "The Great Library") as f64,
        revive as f64,
        state.discard_pile.len() as f64,
        (state.cities[0].coins + state.cities[1].coins) as f64,
        military as f64,
        vp_gap as f64,
        science_max as f64,
        // A forcing threat: one symbol from an instant win collapses the
        // opponent to "block or lose", a different search shape from merely
        // leading. Integer 0/1, exactly as Python casts it.
        f64::from(science_max >= 5),
        (9 - military) as f64,
        chance_fanout,
        chance_fanout_max as f64,
        pending_options as f64,
        affordable_wonders as f64,
        present.len() as f64 * log_legal,
        (state.discard_pile.len() * revive) as f64,
    ]
}

/// Is a solve even worth pricing here?
///
/// Kept separate from the cost model: this is the definitional gate (Age III,
/// mid-play), while the model answers the economic question. Both must pass.
pub fn eligible(state: &GameState) -> bool {
    state.phase == Phase::PlayAge && state.tableau.age == 3
}
