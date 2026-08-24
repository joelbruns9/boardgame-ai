//! The static-table signature — a mirror of `games/welcome_to/tables.py`
//! (RUST_PORT_PLAN.md M0-D).
//!
//! Both engines hash the rule tables they were built from and compare at load.
//! A silent table divergence produces a legal-looking game that is a *different
//! game*, and it would pass every per-action gate: the two engines would agree
//! perfectly about a world neither shares with BGA.
//!
//! ⚠ The stream is integers, not text: nothing here depends on formatting,
//! encoding or float repr, all three of which Rust and Python may legitimately
//! disagree about.

use crate::codec;
use crate::constants::{
    card_table, solo_card_id, BIS_BOXES, BIS_SCORES, BOX_OFFSET, ESTATE_ROW_BOXES,
    ESTATE_ROW_SCORES, EXTREMITY_POSITIONS, FENCE_OFFSET, FENCE_SIZES, MAX_ESTATE_SIZE,
    MAX_NUMBER, MIN_NUMBER, NUM_BOXES, NUM_FENCES, NUM_STREETS, PARK_BOXES, PARK_SCORES,
    PERMIT_BOXES, PERMIT_SCORES, POOL_BOXES, POOL_POSITIONS, POOL_SCORES, ROUNDABOUT,
    ROUNDABOUT_BOXES, ROUNDABOUT_SCORES, SOLO_DECK_MIDDLE, STREET_SIZES, TEMP_BOXES,
    TEMP_DELTAS, TEMP_RANK_SCORES, TEMP_SOLO_SCORE, TEMP_SOLO_THRESHOLD,
};
use crate::plans::{Param, PLANS};

/// Bump when the *stream layout* changes (not when a table's values change).
pub const SIGNATURE_VERSION: i64 = 1;

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

/// Fixed codes for the plan parameters that are strings.
fn param_value(param: Param) -> i64 {
    match param {
        Param::Int(v) => v as i64,
        Param::Text(text) => match text {
            "park" => 1,
            "pool" => 2,
            "pool&park" => 3,
            "iceCream" => 4,
            "without" => 5,
            "with" => 6,
            "christmas" => 7,
            "easterEgg" => 8,
            other => panic!(
                "plan parameter {other:?} has no signature code; add it to both engines"
            ),
        },
    }
}

/// Every static table, flattened, in the order `tables.py::signature_stream`
/// yields them.
pub fn signature_stream() -> Vec<i64> {
    let mut out: Vec<i64> = Vec::with_capacity(768);
    out.push(SIGNATURE_VERSION);

    // -- cards ------------------------------------------------------------
    out.push(card_table().len() as i64);
    for &(number, effect) in card_table().iter() {
        out.push(number as i64);
        out.push(effect.as_i64());
    }
    out.push(solo_card_id() as i64);
    out.push(SOLO_DECK_MIDDLE as i64);

    // -- geometry ---------------------------------------------------------
    out.push(NUM_STREETS as i64);
    out.extend(STREET_SIZES.iter().map(|&v| v as i64));
    out.extend(FENCE_SIZES.iter().map(|&v| v as i64));
    out.extend(BOX_OFFSET.iter().map(|&v| v as i64));
    out.extend(FENCE_OFFSET.iter().map(|&v| v as i64));
    out.push(NUM_BOXES as i64);
    out.push(NUM_FENCES as i64);
    out.push(MIN_NUMBER as i64);
    out.push(MAX_NUMBER as i64);
    out.push(ROUNDABOUT as i64);
    out.extend(TEMP_DELTAS.iter().map(|&v| v as i64));

    // -- score tracks -----------------------------------------------------
    out.extend(PARK_BOXES.iter().map(|&v| v as i64));
    for row in PARK_SCORES.iter() {
        out.push(row.len() as i64);
        out.extend(row.iter().map(|&v| v as i64));
    }
    out.push(POOL_BOXES as i64);
    out.extend(POOL_SCORES.iter().map(|&v| v as i64));
    for &(x, y) in POOL_POSITIONS.iter() {
        out.push(x as i64);
        out.push(y as i64);
    }
    out.push(BIS_BOXES as i64);
    out.extend(BIS_SCORES.iter().map(|&v| v as i64));
    out.push(TEMP_BOXES as i64);
    out.extend(TEMP_RANK_SCORES.iter().map(|&v| v as i64));
    out.push(TEMP_SOLO_THRESHOLD as i64);
    out.push(TEMP_SOLO_SCORE as i64);
    out.push(MAX_ESTATE_SIZE as i64);
    out.extend(ESTATE_ROW_BOXES.iter().map(|&v| v as i64));
    for row in ESTATE_ROW_SCORES.iter() {
        out.push(row.len() as i64);
        out.extend(row.iter().map(|&v| v as i64));
    }
    out.push(PERMIT_BOXES as i64);
    out.extend(PERMIT_SCORES.iter().map(|&v| v as i64));
    out.push(ROUNDABOUT_BOXES as i64);
    out.extend(ROUNDABOUT_SCORES.iter().map(|&v| v as i64));
    for &(x, y) in EXTREMITY_POSITIONS.iter() {
        out.push(x as i64);
        out.push(y as i64);
    }

    // -- plans ------------------------------------------------------------
    out.push(PLANS.len() as i64);
    for plan in PLANS.iter() {
        out.push(plan.id as i64);
        out.push(plan.variant as u8 as i64);
        out.push(plan.stack as i64);
        out.push(plan.scores.0 as i64);
        out.push(plan.scores.1 as i64);
        out.push(plan.kind as u8 as i64);
        out.push(plan.params.len() as i64);
        for &param in plan.params.iter() {
            out.push(param_value(param));
        }
    }

    // -- codec layout -----------------------------------------------------
    out.push(codec::NUM_ACTIONS as i64);
    for offset in [
        codec::A_CHOOSE_STACK,
        codec::A_PERMIT_REFUSAL,
        codec::A_ROUNDABOUT_OPEN,
        codec::A_ROUNDABOUT_POS,
        codec::A_WRITE,
        codec::A_SURVEYOR_FENCE,
        codec::A_ESTATE_ROW,
        codec::A_PARK_STREET,
        codec::A_POOL_BUILD,
        codec::A_BIS,
        codec::A_CHOOSE_PLAN,
        codec::A_VALIDATE_ESTATE,
        codec::A_PASS_ROUNDABOUT,
        codec::A_PASS_SURVEYOR,
        codec::A_PASS_ESTATE,
        codec::A_PASS_PARK,
        codec::A_PASS_POOL,
        codec::A_PASS_BIS,
        codec::A_PASS_PLAN,
        codec::A_RESHUFFLE_YES,
        codec::A_RESHUFFLE_NO,
    ] {
        out.push(offset as i64);
    }
    for &(i, j) in codec::EXPERT_PAIRS.iter() {
        out.push(i as i64);
        out.push(j as i64);
    }

    out
}

/// FNV-1a 64 over `signature_stream`, each value as 8 little-endian bytes.
///
/// FNV rather than SHA-256 deliberately: the crate's dependency list starts at
/// pyo3 only (§6), and this hash guards against a transcription slip, not an
/// adversary.
pub fn table_signature() -> u64 {
    let mut h = FNV_OFFSET;
    for value in signature_stream() {
        let word = value as u64;
        for shift in (0..64).step_by(8) {
            h ^= (word >> shift) & 0xFF;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value `python -m games.welcome_to.tables` prints.
    ///
    /// This is the whole point of M0-D: if the tables here drift from the
    /// Python ones, this test fails before any game is played.
    #[test]
    fn signature_matches_python() {
        assert_eq!(signature_stream().len(), 697);
        assert_eq!(table_signature(), 0x2145_f101_3edc_fa99);
    }
}
