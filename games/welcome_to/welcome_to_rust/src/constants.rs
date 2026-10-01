//! Static rule data — a transcription of `games/welcome_to/constants.py`, which
//! is itself a transcription of `BGA Files/welcometo`.
//!
//! ⚠ **Python is the oracle** (RUST_PORT_PLAN.md §3). If these tables and the
//! Python ones disagree, Python is right — and `tables::table_signature` is what
//! makes the disagreement loud rather than a legal-looking different game.

use std::sync::OnceLock;

// ──────────────────────────────────────────────────────────────────────────
// Effects
// ──────────────────────────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Effect {
    Surveyor = 1,
    Estate = 2,
    Park = 3,
    Pool = 4,
    Temp = 5,
    Bis = 6,
    /// The solo-mode marker card; never appears as a playable effect.
    Solo = 7,
}

impl Effect {
    pub fn as_i64(self) -> i64 {
        self as u8 as i64
    }

    /// The inverse, for a snapshot coming back in.
    pub fn from_i64(value: i64) -> Option<Effect> {
        Some(match value {
            1 => Effect::Surveyor,
            2 => Effect::Estate,
            3 => Effect::Park,
            4 => Effect::Pool,
            5 => Effect::Temp,
            6 => Effect::Bis,
            7 => Effect::Solo,
            _ => return None,
        })
    }
}

/// Column order of `DECK_COUNTS`, copied from `ConstructionCards::$actions`.
const DECK_EFFECT_ORDER: [Effect; 6] = [
    Effect::Surveyor,
    Effect::Pool,
    Effect::Temp,
    Effect::Bis,
    Effect::Park,
    Effect::Estate,
];

/// `ConstructionCards::$deck` — copies of each (number, effect) pair, in
/// ascending number order. Sums to 81 cards.
const DECK_COUNTS: [(i32, [u8; 6]); 15] = [
    (1, [1, 0, 0, 0, 1, 1]),
    (2, [1, 0, 0, 0, 1, 1]),
    (3, [1, 1, 1, 1, 0, 0]),
    (4, [0, 1, 1, 1, 1, 1]),
    (5, [2, 0, 0, 0, 2, 2]),
    (6, [2, 1, 1, 1, 1, 1]),
    (7, [1, 1, 1, 1, 2, 2]),
    (8, [2, 1, 1, 1, 2, 2]),
    (9, [1, 1, 1, 1, 2, 2]),
    (10, [2, 1, 1, 1, 1, 1]),
    (11, [2, 0, 0, 0, 2, 2]),
    (12, [0, 1, 1, 1, 1, 1]),
    (13, [1, 1, 1, 1, 0, 0]),
    (14, [1, 0, 0, 0, 1, 1]),
    (15, [1, 0, 0, 0, 1, 1]),
];

/// A card's number face, or `NO_NUMBER` for the solo marker.
pub const NO_NUMBER: i32 = -1;

/// Every construction card as `(number, effect)`. A "card id" anywhere in the
/// engine is an index into this table; the last entry is the solo card.
pub fn card_table() -> &'static Vec<(i32, Effect)> {
    static TABLE: OnceLock<Vec<(i32, Effect)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut cards: Vec<(i32, Effect)> = Vec::with_capacity(82);
        for (number, counts) in DECK_COUNTS.iter() {
            for (slot, effect) in DECK_EFFECT_ORDER.iter().enumerate() {
                for _ in 0..counts[slot] {
                    cards.push((*number, *effect));
                }
            }
        }
        cards.push((NO_NUMBER, Effect::Solo));
        cards
    })
}

pub fn card_number(card: usize) -> i32 {
    card_table()[card].0
}

pub fn card_effect(card: usize) -> Effect {
    card_table()[card].1
}

/// Index of the solo marker card; also the number of base cards below it.
pub fn solo_card_id() -> usize {
    card_table().len() - 1
}

pub fn num_base_cards() -> usize {
    solo_card_id()
}

/// `ConstructionCards::soloSetupNewGame` keeps the solo card out of the top
/// half of the deck by shifting it down by this many positions.
pub const SOLO_DECK_MIDDLE: usize = 42;

// ──────────────────────────────────────────────────────────────────────────
// Sheet geometry
// ──────────────────────────────────────────────────────────────────────────
pub const NUM_STREETS: usize = 3;
pub const STREET_SIZES: [usize; 3] = [10, 11, 12];
pub const MAX_STREET_LEN: usize = 12;
pub const BOX_OFFSET: [usize; 3] = [0, 10, 21];
pub const NUM_BOXES: usize = 33;

pub const FENCE_SIZES: [usize; 3] = [9, 10, 11];
pub const FENCE_OFFSET: [usize; 3] = [0, 9, 19];
pub const NUM_FENCES: usize = 30;

pub const MIN_NUMBER: i32 = 0;
pub const MAX_NUMBER: i32 = 17;

/// Sentinel written into a house box for a roundabout. Roundabouts break the
/// ascending-order chain and never belong to a housing estate.
pub const ROUNDABOUT: i32 = 100;

/// An empty house box. Python spells this `None`; a fixed-size array cannot,
/// and `-1` is outside the 0..17 range a house can hold.
pub const EMPTY: i32 = -1;

/// Temp agency modifiers, in the order used by the action codec.
pub const TEMP_DELTAS: [i32; 5] = [0, -2, -1, 1, 2];

pub fn box_index(x: usize, y: usize) -> usize {
    BOX_OFFSET[x] + y
}

pub fn box_coords(index: usize) -> (usize, usize) {
    for x in (0..NUM_STREETS).rev() {
        if index >= BOX_OFFSET[x] {
            return (x, index - BOX_OFFSET[x]);
        }
    }
    panic!("bad box index {index}")
}

pub fn fence_index(x: usize, j: usize) -> usize {
    FENCE_OFFSET[x] + j
}

pub fn fence_coords(index: usize) -> (usize, usize) {
    for x in (0..NUM_STREETS).rev() {
        if index >= FENCE_OFFSET[x] {
            return (x, index - FENCE_OFFSET[x]);
        }
    }
    panic!("bad fence index {index}")
}

// ──────────────────────────────────────────────────────────────────────────
// Score tracks
// ──────────────────────────────────────────────────────────────────────────
pub const PARK_BOXES: [i32; 3] = [3, 4, 5];
pub const PARK_SCORES: [&[i32]; 3] = [
    &[0, 2, 4, 10],
    &[0, 2, 4, 6, 14],
    &[0, 2, 4, 6, 8, 18],
];

pub const POOL_BOXES: i32 = 9;
pub const POOL_SCORES: [i32; 10] = [0, 3, 6, 9, 13, 17, 21, 26, 31, 36];

/// The nine printed pool locations, as `(street, box)`.
pub const POOL_POSITIONS: [(usize, usize); 9] = [
    (0, 2),
    (0, 6),
    (0, 7),
    (1, 0),
    (1, 3),
    (1, 7),
    (2, 1),
    (2, 6),
    (2, 10),
];

pub const BIS_BOXES: i32 = 9;
pub const BIS_SCORES: [i32; 10] = [0, 1, 3, 6, 9, 12, 16, 20, 24, 28];

pub const TEMP_BOXES: i32 = 11;
pub const TEMP_RANK_SCORES: [i32; 3] = [7, 4, 1];
pub const TEMP_SOLO_THRESHOLD: i32 = 6;
pub const TEMP_SOLO_SCORE: i32 = 7;

pub const MAX_ESTATE_SIZE: usize = 6;
pub const ESTATE_ROW_BOXES: [i32; 6] = [1, 2, 3, 4, 4, 4];
pub const ESTATE_ROW_SCORES: [&[i32]; 6] = [
    &[1, 3],
    &[2, 3, 4],
    &[3, 4, 5, 6],
    &[4, 5, 6, 7, 8],
    &[5, 6, 7, 8, 10],
    &[6, 7, 8, 10, 12],
];

pub const PERMIT_BOXES: i32 = 3;
pub const PERMIT_SCORES: [i32; 4] = [0, 0, 3, 5];

pub const ROUNDABOUT_BOXES: i32 = 2;
pub const ROUNDABOUT_SCORES: [i32; 3] = [0, 3, 8];

/// The six boxes checked by the "Extremities" advanced plan.
pub const EXTREMITY_POSITIONS: [(usize, usize); 6] = [
    (0, 0),
    (0, 9),
    (1, 0),
    (1, 10),
    (2, 0),
    (2, 11),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deck_is_81_cards_plus_the_solo_marker() {
        assert_eq!(num_base_cards(), 81);
        assert_eq!(card_table().len(), 82);
        assert_eq!(card_table()[0], (1, Effect::Surveyor));
        assert_eq!(card_effect(solo_card_id()), Effect::Solo);
    }

    #[test]
    fn box_and_fence_indices_round_trip() {
        for x in 0..NUM_STREETS {
            for y in 0..STREET_SIZES[x] {
                assert_eq!(box_coords(box_index(x, y)), (x, y));
            }
            for j in 0..FENCE_SIZES[x] {
                assert_eq!(fence_coords(fence_index(x, j)), (x, j));
            }
        }
    }
}
