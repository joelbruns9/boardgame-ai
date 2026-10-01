//! The flat policy index space — a mirror of `games/welcome_to/action_codec.py`.
//!
//! The offsets are written out as constants rather than computed from a section
//! table, because they are an ABI: a checkpoint's policy head is indexed by
//! them. `tables::table_signature` hashes them so a shift cannot happen quietly.

use crate::constants::{box_coords, box_index, fence_coords, fence_index, NUM_BOXES};

/// Ordered (number-card, effect-card) stack pairs used by expert and solo mode,
/// in the order produced by `ConstructionCards::getPossibleCombinations`.
pub const EXPERT_PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)];

pub const NUM_BIS_SIDES: usize = 2;

pub const A_CHOOSE_STACK: usize = 0;
pub const A_PERMIT_REFUSAL: usize = 6;
pub const A_ROUNDABOUT_OPEN: usize = 7;
pub const A_ROUNDABOUT_POS: usize = 8;
pub const A_WRITE: usize = 41;
pub const A_SURVEYOR_FENCE: usize = 206;
pub const A_ESTATE_ROW: usize = 236;
pub const A_PARK_STREET: usize = 242;
pub const A_POOL_BUILD: usize = 245;
pub const A_BIS: usize = 246;
pub const A_CHOOSE_PLAN: usize = 312;
pub const A_VALIDATE_ESTATE: usize = 315;
pub const A_PASS_ROUNDABOUT: usize = 348;
pub const A_PASS_SURVEYOR: usize = 349;
pub const A_PASS_ESTATE: usize = 350;
pub const A_PASS_PARK: usize = 351;
pub const A_PASS_POOL: usize = 352;
pub const A_PASS_BIS: usize = 353;
pub const A_PASS_PLAN: usize = 354;
pub const A_RESHUFFLE_YES: usize = 355;
pub const A_RESHUFFLE_NO: usize = 356;
pub const NUM_ACTIONS: usize = 357;

// ── encoders ──────────────────────────────────────────────────────────────
pub fn choose_stack(slot: usize) -> usize {
    A_CHOOSE_STACK + slot
}

pub fn roundabout_pos(x: usize, y: usize) -> usize {
    A_ROUNDABOUT_POS + box_index(x, y)
}

/// `delta_slot` indexes `constants::TEMP_DELTAS`.
pub fn write(delta_slot: usize, x: usize, y: usize) -> usize {
    A_WRITE + delta_slot * NUM_BOXES + box_index(x, y)
}

pub fn surveyor_fence(x: usize, j: usize) -> usize {
    A_SURVEYOR_FENCE + fence_index(x, j)
}

pub fn estate_row(row: usize) -> usize {
    A_ESTATE_ROW + row
}

pub fn park_street(x: usize) -> usize {
    A_PARK_STREET + x
}

pub fn bis(x: usize, y: usize, side: usize) -> usize {
    A_BIS + box_index(x, y) * NUM_BIS_SIDES + side
}

pub fn choose_plan(slot: usize) -> usize {
    A_CHOOSE_PLAN + slot
}

pub fn validate_estate(x: usize, y_start: usize) -> usize {
    A_VALIDATE_ESTATE + box_index(x, y_start)
}

// ── decoders ──────────────────────────────────────────────────────────────
pub fn decode_stack(index: usize) -> usize {
    index - A_CHOOSE_STACK
}

pub fn decode_roundabout_pos(index: usize) -> (usize, usize) {
    box_coords(index - A_ROUNDABOUT_POS)
}

/// Returns `(delta_slot, x, y)`.
pub fn decode_write(index: usize) -> (usize, usize, usize) {
    let rel = index - A_WRITE;
    let (x, y) = box_coords(rel % NUM_BOXES);
    (rel / NUM_BOXES, x, y)
}

pub fn decode_surveyor_fence(index: usize) -> (usize, usize) {
    fence_coords(index - A_SURVEYOR_FENCE)
}

pub fn decode_estate_row(index: usize) -> usize {
    index - A_ESTATE_ROW
}

pub fn decode_park_street(index: usize) -> usize {
    index - A_PARK_STREET
}

/// Returns `(x, y, side)`.
pub fn decode_bis(index: usize) -> (usize, usize, usize) {
    let rel = index - A_BIS;
    let (x, y) = box_coords(rel / NUM_BIS_SIDES);
    (x, y, rel % NUM_BIS_SIDES)
}

pub fn decode_plan(index: usize) -> usize {
    index - A_CHOOSE_PLAN
}

pub fn decode_validate_estate(index: usize) -> (usize, usize) {
    box_coords(index - A_VALIDATE_ESTATE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{NUM_FENCES, NUM_STREETS, STREET_SIZES};

    #[test]
    fn layout_is_contiguous_and_full() {
        assert_eq!(A_WRITE + 5 * NUM_BOXES, A_SURVEYOR_FENCE);
        assert_eq!(A_SURVEYOR_FENCE + NUM_FENCES, A_ESTATE_ROW);
        assert_eq!(A_BIS + NUM_BOXES * NUM_BIS_SIDES, A_CHOOSE_PLAN);
        assert_eq!(A_VALIDATE_ESTATE + NUM_BOXES, A_PASS_ROUNDABOUT);
        assert_eq!(A_RESHUFFLE_NO + 1, NUM_ACTIONS);
    }

    #[test]
    fn write_and_bis_round_trip() {
        for x in 0..NUM_STREETS {
            for y in 0..STREET_SIZES[x] {
                for d in 0..5 {
                    assert_eq!(decode_write(write(d, x, y)), (d, x, y));
                }
                for side in 0..NUM_BIS_SIDES {
                    assert_eq!(decode_bis(bis(x, y, side)), (x, y, side));
                }
            }
        }
    }
}
