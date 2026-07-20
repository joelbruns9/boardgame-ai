//! Read-side of the unseen-card pool (encoder foundation; F2.1).
//!
//! Mirrors `pool.py::unseen_pool`/`visible_card_names`: the hidden-card
//! structure deduced from the *public* projection only (face-up tableau, both
//! cities, discard, burials). Hidden information is symmetric, so the pool does
//! not depend on the viewer. The search-side `resample_hidden`/`enumerate_*`
//! helpers arrive with F3; the encoder needs only these read helpers.

use crate::data::{back_type_of, NUM_CARDS, NUM_PROGRESS, NUM_WONDERS};
use crate::state::{GameState, Phase};

/// Unseen entities deducible from the public projection.
pub struct UnseenPool {
    /// Unseen card ids per back type, indexed by `BackType as usize`
    /// (AgeI=0, AgeII=1, AgeIII=2, Guild=3); each list ascending by id.
    pub cards: [Vec<usize>; 4],
    /// Wonders not yet seen in an offer or a city, ascending by id.
    pub wonders: Vec<usize>,
    /// Progress tokens neither on the board nor owned (the Great Library pool),
    /// ascending by id.
    pub offboard_progress: Vec<usize>,
}

/// `visible[id]` is true iff card `id` is visible somewhere public. The tableau
/// contributes only when not drafting (the age's cards are hidden until the
/// draft ends) and only its revealed, still-present slots.
fn visible_cards(state: &GameState) -> [bool; NUM_CARDS] {
    let mut visible = [false; NUM_CARDS];
    if state.phase != Phase::WonderDraft {
        for slot in &state.tableau.slots {
            if slot.present && slot.revealed {
                visible[slot.card_id] = true;
            }
        }
    }
    for city in &state.cities {
        for &b in &city.buildings {
            visible[b] = true;
        }
    }
    for &c in &state.discard_pile {
        visible[c] = true;
    }
    for &c in &state.buried_cards {
        visible[c] = true;
    }
    visible
}

pub fn unseen_pool(state: &GameState) -> UnseenPool {
    let visible = visible_cards(state);
    let mut cards: [Vec<usize>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for id in 0..NUM_CARDS {
        if !visible[id] {
            cards[back_type_of(id) as usize].push(id);
        }
    }

    let mut seen_wonder = [false; NUM_WONDERS];
    for &w in &state.wonder_offer {
        seen_wonder[w] = true;
    }
    for city in &state.cities {
        for &w in &city.wonders {
            seen_wonder[w] = true;
        }
    }
    let wonders = (0..NUM_WONDERS).filter(|&w| !seen_wonder[w]).collect();

    let mut owned_progress = [false; NUM_PROGRESS];
    for &p in &state.available_progress_tokens {
        owned_progress[p] = true;
    }
    for city in &state.cities {
        for &p in &city.progress_tokens {
            owned_progress[p] = true;
        }
    }
    let offboard_progress = (0..NUM_PROGRESS).filter(|&p| !owned_progress[p]).collect();

    UnseenPool {
        cards,
        wonders,
        offboard_progress,
    }
}
