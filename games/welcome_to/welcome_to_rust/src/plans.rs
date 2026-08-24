//! City Plans — a mirror of `games/welcome_to/plans.py`.
//!
//! Plan ids are the array indices of `PlanCards::$plans`, so they compare
//! directly against a BGA game log. The seasonal boards are listed for id
//! fidelity and are never dealt; their predicates panic rather than guess.

use crate::constants::{
    EXTREMITY_POSITIONS, MAX_ESTATE_SIZE, NUM_STREETS, PARK_BOXES, STREET_SIZES,
};
use crate::sheet::{Estate, Pos, Sheet};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Variant {
    Basic = 1,
    Advanced = 2,
    IceCream = 3,
    Christmas = 4,
    Easter = 5,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum PlanKind {
    Estate = 0,
    FullStreet = 1,
    FiveBis = 2,
    SevenTemp = 3,
    Extremities = 4,
    Decorative = 5,
    CompleteStreet = 6,
    Unsupported = 7,
}

/// A plan parameter: estate sizes and street indices are integers, the
/// decorative plans name what they want.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Param {
    Int(i32),
    Text(&'static str),
}

impl Param {
    pub fn int(self) -> i32 {
        match self {
            Param::Int(v) => v,
            Param::Text(t) => panic!("plan parameter {t:?} is not an integer"),
        }
    }

    pub fn text(self) -> &'static str {
        match self {
            Param::Text(t) => t,
            Param::Int(v) => panic!("plan parameter {v} is not a string"),
        }
    }
}

pub struct Plan {
    pub id: usize,
    pub variant: Variant,
    /// 1, 2 or 3.
    pub stack: u8,
    /// `(first to finish, everyone later)`.
    pub scores: (i32, i32),
    pub kind: PlanKind,
    pub params: &'static [Param],
}

impl Plan {
    /// `AbstractPlan::$automatic` — only an estate plan asks the player which
    /// housing estates to spend.
    pub fn is_automatic(&self) -> bool {
        self.kind != PlanKind::Estate
    }

    pub fn required_sizes(&self) -> Vec<usize> {
        if self.kind == PlanKind::Estate {
            self.params.iter().map(|p| p.int() as usize).collect()
        } else {
            Vec::new()
        }
    }
}

use Param::{Int as I, Text as T};
use PlanKind::{
    CompleteStreet as K_CS, Decorative as K_DEC, Estate as K_E, Extremities as K_EX,
    FiveBis as K_5B, FullStreet as K_FS, SevenTemp as K_7T, Unsupported as K_U,
};
use Variant::{Advanced as V_A, Basic as V_B, Christmas as V_C, Easter as V_EA, IceCream as V_IC};

/// `PlanCards::$plans`, index for index.
pub static PLANS: [Plan; 37] = [
    Plan { id: 0, variant: V_B, stack: 1, scores: (8, 4), kind: K_E, params: &[I(1), I(1), I(1), I(1), I(1), I(1)] },
    Plan { id: 1, variant: V_B, stack: 1, scores: (8, 4), kind: K_E, params: &[I(2), I(2), I(2), I(2)] },
    Plan { id: 2, variant: V_B, stack: 1, scores: (8, 4), kind: K_E, params: &[I(3), I(3), I(3)] },
    Plan { id: 3, variant: V_B, stack: 1, scores: (6, 3), kind: K_E, params: &[I(4), I(4)] },
    Plan { id: 4, variant: V_B, stack: 1, scores: (8, 4), kind: K_E, params: &[I(5), I(5)] },
    Plan { id: 5, variant: V_B, stack: 1, scores: (10, 6), kind: K_E, params: &[I(6), I(6)] },
    Plan { id: 6, variant: V_B, stack: 2, scores: (11, 6), kind: K_E, params: &[I(1), I(1), I(1), I(6)] },
    Plan { id: 7, variant: V_B, stack: 2, scores: (10, 6), kind: K_E, params: &[I(5), I(2), I(2)] },
    Plan { id: 8, variant: V_B, stack: 2, scores: (12, 7), kind: K_E, params: &[I(3), I(3), I(4)] },
    Plan { id: 9, variant: V_B, stack: 2, scores: (8, 4), kind: K_E, params: &[I(3), I(6)] },
    Plan { id: 10, variant: V_B, stack: 2, scores: (9, 5), kind: K_E, params: &[I(4), I(5)] },
    Plan { id: 11, variant: V_B, stack: 2, scores: (9, 5), kind: K_E, params: &[I(4), I(1), I(1), I(1)] },
    Plan { id: 12, variant: V_B, stack: 3, scores: (12, 7), kind: K_E, params: &[I(1), I(2), I(6)] },
    Plan { id: 13, variant: V_B, stack: 3, scores: (13, 7), kind: K_E, params: &[I(1), I(4), I(5)] },
    Plan { id: 14, variant: V_B, stack: 3, scores: (7, 3), kind: K_E, params: &[I(3), I(4)] },
    Plan { id: 15, variant: V_B, stack: 3, scores: (7, 3), kind: K_E, params: &[I(2), I(5)] },
    Plan { id: 16, variant: V_B, stack: 3, scores: (11, 6), kind: K_E, params: &[I(1), I(2), I(2), I(3)] },
    Plan { id: 17, variant: V_B, stack: 3, scores: (13, 7), kind: K_E, params: &[I(2), I(3), I(5)] },
    Plan { id: 18, variant: V_A, stack: 1, scores: (8, 4), kind: K_FS, params: &[I(2)] },
    Plan { id: 19, variant: V_A, stack: 1, scores: (6, 3), kind: K_FS, params: &[I(0)] },
    Plan { id: 20, variant: V_A, stack: 1, scores: (8, 3), kind: K_5B, params: &[] },
    Plan { id: 21, variant: V_A, stack: 1, scores: (6, 3), kind: K_7T, params: &[] },
    Plan { id: 22, variant: V_A, stack: 1, scores: (7, 4), kind: K_EX, params: &[] },
    Plan { id: 23, variant: V_A, stack: 2, scores: (7, 4), kind: K_DEC, params: &[T("park")] },
    Plan { id: 24, variant: V_A, stack: 2, scores: (10, 5), kind: K_CS, params: &[] },
    Plan { id: 25, variant: V_A, stack: 2, scores: (7, 4), kind: K_DEC, params: &[T("pool")] },
    Plan { id: 26, variant: V_A, stack: 2, scores: (10, 5), kind: K_DEC, params: &[T("pool&park"), I(2)] },
    Plan { id: 27, variant: V_A, stack: 2, scores: (8, 3), kind: K_DEC, params: &[T("pool&park"), I(1)] },
    // -- seasonal boards, listed for id fidelity only, never dealt --
    Plan { id: 28, variant: V_IC, stack: 3, scores: (6, 4), kind: K_U, params: &[T("iceCream")] },
    Plan { id: 29, variant: V_IC, stack: 3, scores: (7, 3), kind: K_U, params: &[T("without"), I(3), I(4), I(5)] },
    Plan { id: 30, variant: V_IC, stack: 3, scores: (8, 4), kind: K_U, params: &[T("with"), I(4), I(4), I(4)] },
    Plan { id: 31, variant: V_C, stack: 3, scores: (10, 5), kind: K_U, params: &[T("with"), I(6), I(6)] },
    Plan { id: 32, variant: V_C, stack: 3, scores: (14, 7), kind: K_U, params: &[T("christmas")] },
    Plan { id: 33, variant: V_C, stack: 3, scores: (10, 5), kind: K_U, params: &[T("without"), I(3), I(3)] },
    Plan { id: 34, variant: V_EA, stack: 3, scores: (7, 3), kind: K_U, params: &[T("easterEgg")] },
    Plan { id: 35, variant: V_EA, stack: 3, scores: (10, 5), kind: K_U, params: &[T("without"), I(2), I(3), I(4)] },
    Plan { id: 36, variant: V_EA, stack: 3, scores: (8, 4), kind: K_U, params: &[T("with"), I(3), I(3), I(3)] },
];

fn is_supported(variant: Variant) -> bool {
    matches!(variant, Variant::Basic | Variant::Advanced)
}

/// `AbstractPlan::isAvailable` restricted to the base board, in id order.
pub fn available_plan_ids(stack: u8, advanced: bool) -> Vec<usize> {
    let mut out = Vec::new();
    for plan in PLANS.iter() {
        if plan.stack != stack || !is_supported(plan.variant) {
            continue;
        }
        if plan.variant == Variant::Advanced && !advanced {
            continue;
        }
        out.push(plan.id);
    }
    out
}

// ──────────────────────────────────────────────────────────────────────────
// Completion predicates
// ──────────────────────────────────────────────────────────────────────────

/// Per-size supply of free estates, indexed `size - 1`; sizes above
/// `MAX_ESTATE_SIZE` cannot be asked for and are counted nowhere.
fn free_estate_supply(sheet: &Sheet) -> [usize; MAX_ESTATE_SIZE] {
    let mut supply = [0usize; MAX_ESTATE_SIZE];
    for (_, _, size) in sheet.free_estates() {
        if size >= 1 && size <= MAX_ESTATE_SIZE {
            supply[size - 1] += 1;
        }
    }
    supply
}

/// `EstatePlan::canBeScored` — a multiset difference against the estates no
/// plan has consumed. Because every requirement asks for an *exact* size,
/// feasibility is pure counting.
fn estates_available_for(sizes: &[usize], sheet: &Sheet) -> bool {
    let supply = free_estate_supply(sheet);
    let mut need = [0usize; MAX_ESTATE_SIZE];
    for &size in sizes {
        if size < 1 || size > MAX_ESTATE_SIZE {
            return false;
        }
        need[size - 1] += 1;
    }
    (0..MAX_ESTATE_SIZE).all(|i| supply[i] >= need[i])
}

/// The per-sheet half of `AbstractPlan::canBeScored`; the caller owns the other
/// half (that this player has not already scored this plan), which lives in the
/// game rather than the sheet.
pub fn can_be_scored(plan: &Plan, sheet: &Sheet) -> bool {
    match plan.kind {
        PlanKind::Estate => estates_available_for(&plan.required_sizes(), sheet),
        PlanKind::FullStreet => {
            let x = plan.params[0].int() as usize;
            if (0..STREET_SIZES[x]).any(|y| sheet.top_fences[x][y]) {
                return false;
            }
            (0..STREET_SIZES[x]).all(|y| !sheet.is_empty(x, y))
        }
        PlanKind::FiveBis => sheet.bis_count_per_street().iter().any(|&c| c >= 5),
        PlanKind::SevenTemp => sheet.temps >= 7,
        PlanKind::Extremities => EXTREMITY_POSITIONS
            .iter()
            .all(|&(x, y)| !sheet.is_empty(x, y) && !sheet.top_fences[x][y]),
        PlanKind::CompleteStreet => {
            let parks = sheet.street_parks_complete();
            let pools = sheet.street_pools_complete();
            (0..NUM_STREETS)
                .any(|x| parks[x] && pools[x] && sheet.has_roundabout_in_street(x))
        }
        PlanKind::Decorative => match plan.params[0].text() {
            "park" => sheet.street_parks_complete().iter().filter(|&&c| c).count() >= 2,
            "pool" => sheet.street_pools_complete().iter().filter(|&&c| c).count() >= 2,
            "pool&park" => {
                let x = plan.params[1].int() as usize;
                sheet.street_parks_complete()[x] && sheet.street_pools_complete()[x]
            }
            other => panic!("seasonal decorative plan {other:?} is not supported"),
        },
        PlanKind::Unsupported => {
            panic!("plan {} belongs to an unsupported expansion", plan.id)
        }
    }
}

/// Free estates of exactly `size` that have not been picked yet this
/// validation, in `free_estates` order.
pub fn estates_matching_size(
    sheet: &Sheet,
    size: usize,
    already_chosen: &[Estate],
) -> Vec<Estate> {
    sheet
        .free_estates()
        .into_iter()
        .filter(|e| e.2 == size && !already_chosen.contains(e))
        .collect()
}

/// Houses this plan consumes, which get a top fence and cannot be reused.
pub fn validation_cells(plan: &Plan, chosen_estates: &[Estate]) -> Vec<Pos> {
    match plan.kind {
        PlanKind::Estate => {
            let mut cells = Vec::new();
            for &(x, start, size) in chosen_estates {
                for k in 0..size {
                    cells.push((x, start + k));
                }
            }
            cells
        }
        PlanKind::FullStreet => {
            let x = plan.params[0].int() as usize;
            (0..STREET_SIZES[x]).map(|y| (x, y)).collect()
        }
        PlanKind::Extremities => EXTREMITY_POSITIONS.to_vec(),
        _ => Vec::new(),
    }
}

/// Kept next to the tables it reads so the unused-import warning does not push
/// somebody into deleting the constant.
#[allow(dead_code)]
pub fn park_boxes(x: usize) -> i32 {
    PARK_BOXES[x]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_ids_are_their_own_indices() {
        for (i, plan) in PLANS.iter().enumerate() {
            assert_eq!(plan.id, i);
        }
    }

    #[test]
    fn stacks_match_python() {
        assert_eq!(available_plan_ids(1, false), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(
            available_plan_ids(1, true),
            vec![0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22]
        );
        assert_eq!(available_plan_ids(3, true), vec![12, 13, 14, 15, 16, 17]);
    }
}
