//! City Plans — a mirror of `games/welcome_to/plans.py`.
//!
//! Plan ids are the array indices of `PlanCards::$plans`, so they compare
//! directly against a BGA game log. The seasonal boards are listed for id
//! fidelity and are never dealt; their predicates panic rather than guess.

use crate::constants::{
    EMPTY, EXTREMITY_POSITIONS, FENCE_SIZES, MAX_ESTATE_SIZE, NUM_STREETS, PARK_BOXES,
    POOL_POSITIONS, STREET_SIZES, TEMP_BOXES,
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

/// Width of the dense advanced-superset plan identity used by the encoder.
pub const NUM_DEALT_PLANS: usize = 28;

/// Dense encoder position of a plan that standard/advanced play can deal.
pub fn dense_index(plan_id: usize) -> Option<usize> {
    let mut dense = 0usize;
    for plan in PLANS.iter() {
        if !is_supported(plan.variant) {
            continue;
        }
        if plan.id == plan_id {
            return Some(dense);
        }
        dense += 1;
    }
    None
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

/// Encoder distance to completion: `(fraction, marks_left)`.
pub fn progress(plan: &Plan, sheet: &Sheet) -> (f64, i32) {
    match plan.kind {
        PlanKind::Estate => {
            let required = plan.required_sizes();
            let mut supply = [0usize; MAX_ESTATE_SIZE];
            for (_, _, size) in sheet.free_estates() {
                if (1..=MAX_ESTATE_SIZE).contains(&size) {
                    supply[size - 1] += 1;
                }
            }
            let mut need = [0usize; MAX_ESTATE_SIZE];
            for &size in required.iter() {
                need[size - 1] += 1;
            }
            let matched: usize = (0..MAX_ESTATE_SIZE)
                .map(|i| need[i].min(supply[i]))
                .sum();
            let left = required.len() - matched;
            (matched as f64 / required.len() as f64, left as i32)
        }
        PlanKind::FullStreet => {
            let x = plan.params[0].int() as usize;
            let size = STREET_SIZES[x];
            if (0..size).any(|y| sheet.top_fences[x][y]) {
                return (0.0, size as i32);
            }
            let built = (0..size).filter(|&y| !sheet.is_empty(x, y)).count();
            (built as f64 / size as f64, (size - built) as i32)
        }
        PlanKind::FiveBis => {
            let best = *sheet.bis_count_per_street().iter().max().expect("three streets");
            (best.min(5) as f64 / 5.0, (5 - best).max(0))
        }
        PlanKind::SevenTemp => (
            sheet.temps.min(7) as f64 / 7.0,
            (7 - sheet.temps).max(0),
        ),
        PlanKind::Extremities => {
            let done = EXTREMITY_POSITIONS
                .iter()
                .filter(|&&(x, y)| !sheet.is_empty(x, y) && !sheet.top_fences[x][y])
                .count();
            (
                done as f64 / EXTREMITY_POSITIONS.len() as f64,
                (EXTREMITY_POSITIONS.len() - done) as i32,
            )
        }
        PlanKind::CompleteStreet => {
            let mut best_left: Option<i32> = None;
            let mut best_cap = 1i32;
            for x in 0..NUM_STREETS {
                let cap = PARK_BOXES[x] + 3 + 1;
                let left = (PARK_BOXES[x] - sheet.parks[x])
                    + (3 - sheet.pools[x])
                    + if sheet.has_roundabout_in_street(x) { 0 } else { 1 };
                if best_left.is_none() || left < best_left.expect("set") {
                    best_left = Some(left);
                    best_cap = cap;
                }
            }
            let left = best_left.unwrap_or(0);
            (1.0 - left as f64 / best_cap as f64, left)
        }
        PlanKind::Decorative => match plan.params[0].text() {
            "park" => {
                let mut needs = [0i32; NUM_STREETS];
                for x in 0..NUM_STREETS {
                    needs[x] = PARK_BOXES[x] - sheet.parks[x];
                }
                needs.sort_unstable();
                let left = needs[0] + needs[1];
                (
                    1.0 - left as f64 / (PARK_BOXES[0] + PARK_BOXES[1]) as f64,
                    left,
                )
            }
            "pool" => {
                let mut needs = [0i32; NUM_STREETS];
                for x in 0..NUM_STREETS {
                    needs[x] = 3 - sheet.pools[x];
                }
                needs.sort_unstable();
                let left = needs[0] + needs[1];
                (1.0 - left as f64 / 6.0, left)
            }
            "pool&park" => {
                let x = plan.params[1].int() as usize;
                let cap = PARK_BOXES[x] + 3;
                let left = (PARK_BOXES[x] - sheet.parks[x]) + (3 - sheet.pools[x]);
                (1.0 - left as f64 / cap as f64, left)
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

// ──────────────────────────────────────────────────────────────────────────
// Encoder v3: requirements, feasibility and a hard turn bound
//
// Mirrors plans.py (ENCODER_V3_SPEC §2, §3, §6.1, §6.2). Read §6.1 before
// touching `feasible`: a roundabout, a bis and a fence each put a house on the
// sheet without a drawn number, and forgetting any one made a death test unsound.
// ──────────────────────────────────────────────────────────────────────────

/// What a plan still wants from one sheet — "how much" AND "where".
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Requirements {
    pub temps_needed: i32,
    pub estate_steps_left: i32,
    pub estate_shortfall: [i32; MAX_ESTATE_SIZE],
    pub parks_needed: [i32; NUM_STREETS],
    pub pools_needed: [i32; NUM_STREETS],
    pub houses_needed: [i32; NUM_STREETS],
    pub bis_needed: [i32; NUM_STREETS],
    pub roundabout_needed: [i32; NUM_STREETS],
    pub street_serves: [i32; NUM_STREETS],
    pub target_boxes: Vec<Pos>,
}

/// Per-size counts of `required_sizes`, indexed `size - 1`.
fn need_counts(plan: &Plan) -> [i32; MAX_ESTATE_SIZE] {
    let mut need = [0i32; MAX_ESTATE_SIZE];
    for size in plan.required_sizes() {
        need[size - 1] += 1;
    }
    need
}

/// Loose upper bound on estates of each size 1..6 this sheet could still hold
/// (`plans.py::reachable_estate_counts`, spec §13.2). Counts boxes, not free
/// boxes: one fence re-partitions a built run while consuming nothing.
pub fn reachable_estate_counts(sheet: &Sheet) -> [i32; MAX_ESTATE_SIZE] {
    let mut counts = [0i32; MAX_ESTATE_SIZE];
    for x in 0..NUM_STREETS {
        let size = STREET_SIZES[x];
        let mut start = 0usize;
        for j in 0..size {
            if j == size - 1 || (j < FENCE_SIZES[x] && sheet.fences[x][j]) {
                let usable = (start..=j).filter(|&y| !sheet.top_fences[x][y]).count() as i32;
                for s in 1..=MAX_ESTATE_SIZE {
                    counts[s - 1] += usable / s as i32;
                }
                start = j + 1;
            }
        }
    }
    counts
}

/// Pool positions in street `x` that could still take a house.
///
/// ⚠ `span_if_roundabout(true)` regardless of the variant, exactly as
/// `plans.py::_pool_boxes_alive` calls it with its default.
fn pool_boxes_alive(sheet: &Sheet, x: usize) -> i32 {
    let spans = sheet.span_if_roundabout(true);
    let mut alive = 0;
    for &(px, py) in POOL_POSITIONS.iter() {
        if px != x || sheet.numbers[x][py] != EMPTY {
            continue;
        }
        if spans[x][py] > 0 || sheet.bis_reachable(x, py) {
            alive += 1;
        }
    }
    alive
}

/// `plans.plan_characteristics` (encoder v4): kind (7) · street (3) · needs
/// (6: pools, parks, roundabout, bis, temp, estates) · estate sizes (6, ÷6) ·
/// estate count (1, ÷6) · stack (3).
pub const NUM_PLAN_CHARACTERISTICS: usize = 26;

pub fn plan_characteristics(plan: &Plan) -> [f64; NUM_PLAN_CHARACTERISTICS] {
    let mut out = [0.0f64; NUM_PLAN_CHARACTERISTICS];
    let kinds = [
        PlanKind::Estate,
        PlanKind::FullStreet,
        PlanKind::FiveBis,
        PlanKind::SevenTemp,
        PlanKind::Extremities,
        PlanKind::Decorative,
        PlanKind::CompleteStreet,
    ];
    for (i, kind) in kinds.iter().enumerate() {
        out[i] = (plan.kind == *kind) as u8 as f64;
    }
    let what = if plan.kind == PlanKind::Decorative { Some(plan.params[0].text()) } else { None };
    if plan.kind == PlanKind::FullStreet {
        out[7 + plan.params[0].int() as usize] = 1.0;
    } else if what == Some("pool&park") {
        out[7 + plan.params[1].int() as usize] = 1.0;
    }
    let complete = plan.kind == PlanKind::CompleteStreet;
    out[10] = (complete || matches!(what, Some("pool") | Some("pool&park"))) as u8 as f64;
    out[11] = (complete || matches!(what, Some("park") | Some("pool&park"))) as u8 as f64;
    out[12] = complete as u8 as f64;
    out[13] = (plan.kind == PlanKind::FiveBis) as u8 as f64;
    out[14] = (plan.kind == PlanKind::SevenTemp) as u8 as f64;
    out[15] = (plan.kind == PlanKind::Estate) as u8 as f64;
    let sizes = plan.required_sizes();
    for size in 1..=6usize {
        out[15 + size] = sizes.iter().filter(|&&s| s == size).count() as f64 / 6.0;
    }
    out[22] = sizes.len() as f64 / 6.0;
    out[23 + (plan.stack as usize - 1)] = 1.0;
    out
}

/// `plans.pool_target_boxes` (encoder v4): the empty, still-usable pool boxes
/// a pool plan needs, in streets that can still finish their pools.
pub fn pool_target_boxes(plan: &Plan, sheet: &Sheet) -> Vec<(usize, usize)> {
    if !matches!(plan.kind, PlanKind::Decorative | PlanKind::CompleteStreet) || !feasible(plan, sheet) {
        return Vec::new(); // a dead plan needs no box
    }
    let candidates: Vec<usize> = match plan.kind {
        PlanKind::Decorative => match plan.params[0].text() {
            "pool" => (0..NUM_STREETS).collect(),
            "pool&park" => vec![plan.params[1].int() as usize],
            _ => return Vec::new(),
        },
        PlanKind::CompleteStreet => {
            let roundabout_left = sheet.can_build_roundabout();
            (0..NUM_STREETS)
                .filter(|&x| sheet.has_roundabout_in_street(x) || roundabout_left)
                .collect()
        }
        _ => return Vec::new(),
    };
    let streets: Vec<usize> = candidates
        .into_iter()
        .filter(|&x| sheet.pools[x] < 3 && sheet.pools[x] + pool_boxes_alive(sheet, x) >= 3)
        .collect();
    let spans = sheet.span_if_roundabout(true);
    POOL_POSITIONS
        .iter()
        .copied()
        .filter(|&(x, y)| {
            streets.contains(&x)
                && sheet.numbers[x][y] == EMPTY
                && (spans[x][y] > 0 || sheet.bis_reachable(x, y))
        })
        .collect()
}

/// Is `plan` still reachable on `sheet`? **Sound, not complete.**
pub fn feasible(plan: &Plan, sheet: &Sheet) -> bool {
    match plan.kind {
        PlanKind::Estate => {
            let supply = free_estate_supply(sheet);
            let reachable = reachable_estate_counts(sheet);
            let need = need_counts(plan);
            (0..MAX_ESTATE_SIZE).all(|i| need[i] <= supply[i] as i32 + reachable[i])
        }
        PlanKind::FullStreet => {
            let x = plan.params[0].int() as usize;
            !(0..STREET_SIZES[x]).any(|y| sheet.top_fences[x][y])
        }
        PlanKind::Extremities => {
            // ⚠ Review F1: dead only if no number, no bis AND no roundabout
            // placed in the box itself can reach it.
            let spans = sheet.span_if_roundabout(true);
            let roundabout_left = sheet.can_build_roundabout();
            for &(x, y) in EXTREMITY_POSITIONS.iter() {
                if sheet.top_fences[x][y] {
                    return false;
                }
                if !sheet.is_empty(x, y) {
                    continue;
                }
                if spans[x][y] == 0 && !sheet.bis_reachable(x, y) && !roundabout_left {
                    return false;
                }
            }
            true
        }
        PlanKind::FiveBis => {
            let reach = sheet.bis_reach();
            let counts = sheet.bis_count_per_street();
            (0..NUM_STREETS).any(|x| counts[x] + reach[x] >= 5)
        }
        PlanKind::SevenTemp => TEMP_BOXES >= 7,
        PlanKind::CompleteStreet => (0..NUM_STREETS).any(|x| {
            sheet.pools[x] + pool_boxes_alive(sheet, x) >= 3
                && (sheet.has_roundabout_in_street(x) || sheet.can_build_roundabout())
        }),
        PlanKind::Decorative => match plan.params[0].text() {
            "park" => true,
            "pool" => {
                (0..NUM_STREETS)
                    .filter(|&x| sheet.pools[x] + pool_boxes_alive(sheet, x) >= 3)
                    .count()
                    >= 2
            }
            "pool&park" => {
                let x = plan.params[1].int() as usize;
                sheet.pools[x] + pool_boxes_alive(sheet, x) >= 3
            }
            other => panic!("seasonal decorative plan {other:?} is not supported"),
        },
        PlanKind::Unsupported => {
            panic!("plan {} belongs to an unsupported expansion", plan.id)
        }
    }
}

/// What `plan` still wants from `sheet`, split by locus (spec §3).
#[cfg_attr(not(test), allow(dead_code))] // the Python-mirroring API; the encoder caches
pub fn requirements(plan: &Plan, sheet: &Sheet) -> Requirements {
    requirements_given(plan, sheet, feasible(plan, sheet))
}

/// `requirements` with `alive = feasible(plan, sheet)` already in hand.
pub fn requirements_given(plan: &Plan, sheet: &Sheet, alive: bool) -> Requirements {
    let mut r = Requirements {
        temps_needed: 0,
        estate_steps_left: 0,
        estate_shortfall: [0; MAX_ESTATE_SIZE],
        parks_needed: [0; NUM_STREETS],
        pools_needed: [0; NUM_STREETS],
        houses_needed: [0; NUM_STREETS],
        bis_needed: [0; NUM_STREETS],
        roundabout_needed: [0; NUM_STREETS],
        street_serves: [0; NUM_STREETS],
        target_boxes: Vec::new(),
    };
    let done = can_be_scored(plan, sheet);
    let alive_flag = alive as i32;

    match plan.kind {
        PlanKind::Estate => {
            let supply = free_estate_supply(sheet);
            let need = need_counts(plan);
            for i in 0..MAX_ESTATE_SIZE {
                if need[i] > 0 {
                    r.estate_shortfall[i] = (need[i] - supply[i] as i32).max(0);
                }
            }
            r.estate_steps_left = progress(plan, sheet).1;
            r.street_serves = [alive_flag; NUM_STREETS];
        }
        PlanKind::FullStreet => {
            let x = plan.params[0].int() as usize;
            r.street_serves[x] = alive_flag;
            for y in 0..STREET_SIZES[x] {
                if sheet.is_empty(x, y) {
                    r.houses_needed[x] += 1;
                    r.target_boxes.push((x, y));
                }
            }
        }
        PlanKind::Extremities => {
            for &(x, y) in EXTREMITY_POSITIONS.iter() {
                r.street_serves[x] = alive_flag;
                if sheet.is_empty(x, y) {
                    r.houses_needed[x] += 1;
                    r.target_boxes.push((x, y));
                }
            }
        }
        PlanKind::FiveBis => {
            let reach = sheet.bis_reach();
            let counts = sheet.bis_count_per_street();
            for x in 0..NUM_STREETS {
                r.bis_needed[x] = (5 - counts[x]).max(0);
                if counts[x] + reach[x] >= 5 {
                    r.street_serves[x] = 1;
                }
            }
        }
        PlanKind::SevenTemp => {
            r.temps_needed = (7 - sheet.temps).max(0);
        }
        PlanKind::CompleteStreet => {
            for x in 0..NUM_STREETS {
                r.parks_needed[x] = PARK_BOXES[x] - sheet.parks[x];
                r.pools_needed[x] = 3 - sheet.pools[x];
                r.roundabout_needed[x] = if sheet.has_roundabout_in_street(x) { 0 } else { 1 };
                if sheet.pools[x] + pool_boxes_alive(sheet, x) < 3 {
                    continue;
                }
                if !sheet.has_roundabout_in_street(x) && !sheet.can_build_roundabout() {
                    continue;
                }
                r.street_serves[x] = 1;
            }
        }
        PlanKind::Decorative => {
            let what = plan.params[0].text();
            let wants_pool = what == "pool" || what == "pool&park";
            let wants_park = what == "park" || what == "pool&park";
            let streets: Vec<usize> = if what == "pool&park" {
                vec![plan.params[1].int() as usize]
            } else {
                (0..NUM_STREETS).collect()
            };
            for x in streets {
                if wants_park {
                    r.parks_needed[x] = PARK_BOXES[x] - sheet.parks[x];
                }
                if wants_pool {
                    r.pools_needed[x] = 3 - sheet.pools[x];
                }
                if wants_pool && sheet.pools[x] + pool_boxes_alive(sheet, x) < 3 {
                    continue;
                }
                r.street_serves[x] = 1;
            }
        }
        PlanKind::Unsupported => {
            panic!("plan {} belongs to an unsupported expansion", plan.id)
        }
    }

    if done {
        // A completed plan wants nothing; its streets stay alive (see plans.py).
        r.temps_needed = 0;
        r.estate_steps_left = 0;
        r.estate_shortfall = [0; MAX_ESTATE_SIZE];
        r.parks_needed = [0; NUM_STREETS];
        r.pools_needed = [0; NUM_STREETS];
        r.houses_needed = [0; NUM_STREETS];
        r.bis_needed = [0; NUM_STREETS];
        r.roundabout_needed = [0; NUM_STREETS];
        r.target_boxes.clear();
    }
    if !alive {
        r.street_serves = [0; NUM_STREETS];
    }
    r
}

/// The most `progress()` steps one turn could close for `plan`
/// (`game.py::one_turn_ceiling`).
pub fn one_turn_ceiling(plan: &Plan) -> i32 {
    match plan.kind {
        PlanKind::Estate => plan.params.len() as i32,
        PlanKind::FullStreet | PlanKind::Extremities => 3,
        PlanKind::FiveBis | PlanKind::SevenTemp | PlanKind::Decorative => 1,
        PlanKind::CompleteStreet => 2,
        PlanKind::Unsupported => panic!("plan {} has no one-turn ceiling", plan.id),
    }
}

/// Fewest turns in which `plan` could still complete. A hard bound (§6.2).
#[cfg_attr(not(test), allow(dead_code))] // the Python-mirroring API; the encoder caches
pub fn turns_lower_bound(plan: &Plan, sheet: &Sheet) -> i32 {
    turns_lower_bound_given(plan, progress(plan, sheet).1, &requirements(plan, sheet))
}

/// `turns_lower_bound` from `progress(plan, sheet).1` and `requirements(plan, sheet)`.
pub fn turns_lower_bound_given(plan: &Plan, steps: i32, req: &Requirements) -> i32 {
    if steps == 0 {
        return 0;
    }
    let ceiling = one_turn_ceiling(plan).max(1);
    let step_term = (steps + ceiling - 1) / ceiling;
    let houses: i32 = req.houses_needed.iter().sum();
    let house_term = (houses + 2) / 3;
    step_term.max(house_term)
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
    fn cached_forms_match_the_plain_ones() {
        let mut sheet = Sheet::new();
        sheet.write(3, (0, 0), 1, false);
        sheet.write(9, (1, 4), 1, false);
        sheet.build_roundabout((2, 6), 1);
        for plan in PLANS.iter().filter(|p| p.kind != PlanKind::Unsupported) {
            let req = requirements(plan, &sheet);
            assert_eq!(req, requirements_given(plan, &sheet, feasible(plan, &sheet)));
            assert_eq!(
                turns_lower_bound(plan, &sheet),
                turns_lower_bound_given(plan, progress(plan, &sheet).1, &req)
            );
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
