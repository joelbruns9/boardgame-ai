//! Bit-exact mirror of `games/welcome_to/encoder.py` — encoder v3, ABI 2
//! (ENCODER_V3_SPEC.md §12 step 6).
//!
//! Python remains the oracle and §10.6 demands exact float equality. The rules
//! that make that reachable:
//!
//! * integer divisions happen in `f64` and are cast to `f32` once, at the write,
//!   matching Python-float -> NumPy-f32;
//! * the deck histogram is `f32`, as in NumPy, and its normalisations stay `f32`;
//! * every boundary-draw probability is an **integer** numerator sum over one
//!   denominator (`deck_knowledge.ordered_draw_counts`), so summation order is
//!   irrelevant — never a sum over a float joint, which NumPy sums pairwise;
//! * the one non-integer reduction (`eff_rate`) is a plain left-to-right loop
//!   on both sides.
//!
//! ⚠ §6.4's threat pair is DEMOTED (2026-09-25): a plan slot is 34 floats.
//! ⚠ The four `SPEC GAP` rules in `encoder.py` are mirrored here by name.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use crate::constants::{
    box_index, card_effect, card_number, num_base_cards, Effect, BIS_BOXES, EMPTY,
    ESTATE_ROW_BOXES, ESTATE_ROW_SCORES, EXTREMITY_POSITIONS, MAX_ESTATE_SIZE, MAX_NUMBER,
    MAX_STREET_LEN, MIN_NUMBER, NUM_BOXES, NUM_STREETS, PARK_BOXES, PERMIT_BOXES, POOL_BOXES,
    POOL_POSITIONS, ROUNDABOUT, ROUNDABOUT_BOXES, STREET_SIZES, TEMP_BOXES, TEMP_DELTAS,
};
use crate::game::{EngineError, EngineResult, Game, Phase, NO_CARD};
use crate::plans::{
    dense_index, feasible, progress, requirements_given, turns_lower_bound_given, Plan,
    PlanKind, Requirements, NUM_DEALT_PLANS, PLANS,
};
use crate::sheet::{Pos, Sheet};

pub const ENCODER_ABI_VERSION: usize = 3;
pub const MAX_SEATS: usize = 4;
pub const MAX_PLAYERS: usize = 6;
pub const SHEET_PLANES: usize = 22;
pub const NUM_SHEET_SCALAR: usize = 194;
pub const NUM_GLOBAL_SCALAR: usize = 367;
#[allow(dead_code)] // mirrors encoder.PLAN_SLOT_WIDTH; asserted in tests
pub const PLAN_SLOT_WIDTH: usize = 34;
pub const SHEET_PLANES_LEN: usize = MAX_SEATS * SHEET_PLANES * NUM_STREETS * MAX_STREET_LEN;
pub const SHEET_SCALARS_LEN: usize = MAX_SEATS * NUM_SHEET_SCALAR;
pub const VIEWER_PLANE_LEN: usize = NUM_STREETS * MAX_STREET_LEN;

const NUM_EFFECTS: usize = 6;
const NUM_NUMBERS: usize = 15;
const NUM_NUMBER_VALUES: usize = 18;
const TURN_SCALE: f64 = 30.0;
const SCORE_SCALE: f64 = 50.0;
const STEPS_SCALE: f64 = 12.0;
const TURNS_CAP: f64 = 12.0;
/// `constants.EPS` — the only guard constant, declared once in each language.
const EPS: f64 = 1e-6;

const LOW_SENTINEL: i32 = MIN_NUMBER - 1;
const HIGH_SENTINEL: i32 = MAX_NUMBER + 1;
const TOTAL_SPAN_SCALE: f64 = (NUM_BOXES * 18) as f64;
const ESTATE_DEMAND_SCALE: f64 = 66.0;
const FIT_RATE_BOX_SCALE: f64 = NUM_BOXES as f64;
/// ⚠ SPEC GAP 1: SURVEYOR has no track; 3 slots x MAX_ESTATE_SIZE steps.
const SURVEYOR_DEMAND_SCALE: f64 = 18.0;

const P_VALID: usize = 0;
const P_WRITTEN: usize = 1;
const P_NUMBER: usize = 2;
const P_BIS: usize = 3;
const P_ROUNDABOUT: usize = 4;
const P_TOP_FENCE: usize = 5;
const P_FENCE_RIGHT: usize = 6;
const P_POOL: usize = 7;
const P_WRITABLE: usize = 8;
const P_ESTATE_SIZE: usize = 9;
const P_SPAN: usize = 10;
const P_FIT: usize = 11;
const P_WRITABLE_TEMP: usize = 12;
const P_SPAN_ROUNDABOUT: usize = 13;
const P_FIT_DECK: usize = 14;
const P_FIT_TEMP: usize = 15;
const P_FIT_RESHUFFLE: usize = 16;
const P_FIT_ROUNDABOUT: usize = 17;
const P_FIT_NEXT_TURN: usize = 18;
const P_PLAN_TARGET: [usize; 3] = [19, 20, 21];

const E_SURVEYOR: usize = 0;
const E_POOL: usize = 1;
const E_TEMP: usize = 2;
const E_BIS: usize = 3;
const E_PARK: usize = 4;
const E_ESTATE: usize = 5;

pub(crate) type Matrix = [[f32; NUM_EFFECTS]; NUM_NUMBERS];

pub struct EncodedState {
    pub sheet_planes: Vec<f32>,
    pub sheet_scalars: Vec<f32>,
    pub viewer_plane: Vec<f32>,
    pub global_scalars: Vec<f32>,
}

fn ratio_i32(value: i32, scale: i32) -> f32 {
    (value as f64 / scale as f64) as f32
}

/// §9.4: every quotient is `num / max(den, EPS)`, clamped to [0, 1].
fn ratio(num: f64, den: f64) -> f64 {
    (num / den.max(EPS)).max(0.0).min(1.0)
}

/// §9.4: `min(t, TURNS_CAP) / TURNS_CAP`; a non-finite supply lands on 1.0.
fn turns(value: f64) -> f64 {
    if !value.is_finite() {
        return 1.0;
    }
    value.max(0.0).min(TURNS_CAP) / TURNS_CAP
}

fn effect_index(effect: Effect) -> Option<usize> {
    Some(match effect {
        Effect::Surveyor => E_SURVEYOR,
        Effect::Pool => E_POOL,
        Effect::Temp => E_TEMP,
        Effect::Bis => E_BIS,
        Effect::Park => E_PARK,
        Effect::Estate => E_ESTATE,
        Effect::Solo => return None,
    })
}

fn sheet_plane_index(seat: usize, plane: usize, x: usize, y: usize) -> usize {
    (((seat * SHEET_PLANES + plane) * NUM_STREETS + x) * MAX_STREET_LEN) + y
}

fn viewer_plane_index(x: usize, y: usize) -> usize {
    x * MAX_STREET_LEN + y
}

fn seat_order(game: &Game, viewer: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(game.config.players.min(MAX_SEATS));
    out.push(viewer);
    for k in 1..game.config.players {
        if out.len() == MAX_SEATS {
            break;
        }
        out.push((viewer + k) % game.config.players);
    }
    out
}

/// `game.py::_numbers_for` — the temp widening, in codec order.
fn numbers_for(number: i32, effect: Effect) -> Vec<i32> {
    if effect != Effect::Temp {
        return vec![number];
    }
    let mut out = vec![number];
    for &delta in TEMP_DELTAS[1..].iter() {
        let n = number + delta;
        if (MIN_NUMBER..=MAX_NUMBER).contains(&n) {
            out.push(n);
        }
    }
    out
}

fn banked(game: &Game, viewer: usize, seat: usize, slot: usize) -> bool {
    game.plan_turns_for(viewer, slot)
        .iter()
        .any(|&(player, _)| player == seat as i32)
}

/// `(number, effect)` per stack with both faces visible (`visible_cards`).
fn offers(game: &Game, viewer: usize) -> Vec<(i32, Effect)> {
    let mut out = Vec::new();
    for slot in 0..3 {
        if let (Some(number), Some(effect)) = game.combination_faces(slot, viewer) {
            out.push((number, effect));
        }
    }
    out
}

// ──────────────────────────────────────────────────────────────────────────
// Deck histograms
// ──────────────────────────────────────────────────────────────────────────
fn card_cell(card: i32) -> Option<(usize, usize)> {
    if card == NO_CARD || card < 0 {
        return None;
    }
    let card = card as usize;
    let number = card_number(card);
    if !(1..=15).contains(&number) {
        return None;
    }
    Some(((number - 1) as usize, effect_index(card_effect(card))?))
}

fn add_cards<'a>(out: &mut Matrix, cards: impl IntoIterator<Item = &'a i32>) {
    for &card in cards {
        if let Some((number, effect)) = card_cell(card) {
            out[number][effect] += 1.0;
        }
    }
}

fn histogram<'a>(cards: impl IntoIterator<Item = &'a i32>) -> Matrix {
    let mut out = [[0.0f32; NUM_EFFECTS]; NUM_NUMBERS];
    add_cards(&mut out, cards);
    out
}

/// The full 81-card deck as a `(number, effect)` histogram.
///
/// Memoized because `deck_composition` runs once per `information_key` — that
/// is once per search transition, not once per evaluated leaf.
fn base_deck_matrix() -> &'static Matrix {
    static BASE: OnceLock<Matrix> = OnceLock::new();
    BASE.get_or_init(|| {
        let cards: Vec<i32> = (0..num_base_cards()).map(|card| card as i32).collect();
        histogram(cards.iter())
    })
}

fn add_matrix(left: &Matrix, right: &Matrix) -> Matrix {
    let mut out = [[0.0f32; NUM_EFFECTS]; NUM_NUMBERS];
    for n in 0..NUM_NUMBERS {
        for e in 0..NUM_EFFECTS {
            out[n][e] = left[n][e] + right[n][e];
        }
    }
    out
}

pub(crate) fn deck_composition(game: &Game, viewer: usize) -> Matrix {
    // Counts are whole numbers held in `f32`, so summation order is exact.
    let mut seen = histogram(game.table_cards(viewer).iter());
    if !game.config.expert {
        add_cards(&mut seen, game.discard.iter());
    }
    let base = base_deck_matrix();
    let mut out = [[0.0f32; NUM_EFFECTS]; NUM_NUMBERS];
    for n in 0..NUM_NUMBERS {
        for e in 0..NUM_EFFECTS {
            out[n][e] = (base[n][e] - seen[n][e]).max(0.0);
        }
    }
    out
}

pub(crate) fn discard_composition(game: &Game) -> Matrix {
    if game.config.expert {
        [[0.0f32; NUM_EFFECTS]; NUM_NUMBERS]
    } else {
        histogram(game.discard.iter())
    }
}

fn aside_composition(game: &Game) -> Matrix {
    if !game.config.standard() {
        return [[0.0f32; NUM_EFFECTS]; NUM_NUMBERS];
    }
    histogram(game.stack_old[0].iter())
}

fn row_sums(matrix: &Matrix) -> [f32; NUM_NUMBERS] {
    let mut out = [0.0f32; NUM_NUMBERS];
    for n in 0..NUM_NUMBERS {
        for e in 0..NUM_EFFECTS {
            out[n] += matrix[n][e];
        }
    }
    out
}

fn column_sums(matrix: &Matrix) -> [f32; NUM_EFFECTS] {
    let mut out = [0.0f32; NUM_EFFECTS];
    for e in 0..NUM_EFFECTS {
        for row in matrix.iter().take(NUM_NUMBERS) {
            out[e] += row[e];
        }
    }
    out
}

fn normalize(values: &[f32], uniform_if_empty: bool) -> Vec<f32> {
    let total: f32 = values.iter().sum();
    if total <= 0.0 {
        if uniform_if_empty && !values.is_empty() {
            return vec![1.0f32 / values.len() as f32; values.len()];
        }
        return vec![0.0; values.len()];
    }
    values.iter().map(|&value| value / total).collect()
}

// ──────────────────────────────────────────────────────────────────────────
// Masked draw probabilities — integer inclusion-exclusion
// (`deck_knowledge.masked_draw_numerator` and friends, review 2026-09-25
// throughput #1). Exact integers, one division at the end, so the result is
// bit-identical to Python in any summation order.
// ──────────────────────────────────────────────────────────────────────────

/// `total * (total - 1) * ...` over `n` terms.
fn falling(total: i64, n: usize) -> i64 {
    let mut out = 1i64;
    for i in 0..n {
        out *= total - i as i64;
    }
    out
}

/// Ordered draws without replacement, draw `i` landing in `masks[i]`, as exact
/// `(numerator, denominator)`. Masks are 0/1.
fn masked_draw_numerator(counts: &[i64], masks: &[&[i64]]) -> (i64, i64) {
    let n = masks.len();
    let den = falling(counts.iter().sum(), n);
    if n == 0 {
        return (1, den);
    }
    let s: Vec<i64> = masks
        .iter()
        .map(|m| counts.iter().zip(m.iter()).map(|(c, x)| c * x).sum())
        .collect();
    if n == 1 {
        return (s[0], den);
    }
    let pair = |a: usize, b: usize| -> i64 {
        (0..counts.len()).map(|k| counts[k] * masks[a][k] * masks[b][k]).sum()
    };
    if n == 2 {
        return (s[0] * s[1] - pair(0, 1), den);
    }
    assert!(n == 3, "a boundary draws at most three cards, not {n}");
    let triple: i64 = (0..counts.len())
        .map(|k| counts[k] * masks[0][k] * masks[1][k] * masks[2][k])
        .sum();
    (
        s[0] * s[1] * s[2] - pair(0, 1) * s[2] - pair(0, 2) * s[1] - pair(1, 2) * s[0]
            + 2 * triple,
        den,
    )
}

/// `deck_knowledge._probability`.
fn probability(num: i64, den: i64) -> f64 {
    (num as f64 / (den as f64).max(EPS)).max(0.0).min(1.0)
}

/// `deck_knowledge.next_draw_probability`: the literal boundary draw, the
/// first `min(D, 3)` off the deck and the rest off the reform pool.
fn next_draw_probability(deck: &[i64], pool: &[i64], masks: [&[i64]; 3]) -> f64 {
    let split = (deck.iter().sum::<i64>() as usize).min(3);
    let (a, da) = masked_draw_numerator(deck, &masks[..split]);
    let (b, db) = masked_draw_numerator(pool, &masks[split..]);
    probability(a * b, da * db)
}

/// `deck_knowledge.two_triple_probability`: effects from one triple of cards,
/// numbers from another. ⚠ A stated approximation — the two triples are
/// independent of each other (see the Python docstring and its error test).
fn two_triple_probability(matrix: &Matrix, mask_non_temp: &[i64], mask_temp: &[i64]) -> f64 {
    let rows = counts_of(&row_sums(matrix));
    let cols = counts_of(&column_sums(matrix));
    let total: i64 = rows.iter().sum();
    let temp = cols[E_TEMP];
    let classes = [total - temp, temp];
    // Three sums give all eight sequences' numerators (see the Python).
    let single = [
        (0..NUM_NUMBERS).map(|k| rows[k] * mask_non_temp[k]).sum::<i64>(),
        (0..NUM_NUMBERS).map(|k| rows[k] * mask_temp[k]).sum::<i64>(),
    ];
    let both: i64 = (0..NUM_NUMBERS)
        .map(|k| rows[k] * mask_non_temp[k] * mask_temp[k])
        .sum();
    let pair = |a: usize, b: usize| if a == b { single[a] } else { both };
    let mut num = 0i64;
    for t0 in 0..2usize {
        for t1 in 0..2usize {
            for t2 in 0..2usize {
                let seq = [t0, t1, t2];
                let mut effect_num = 1i64;
                let mut used = [0i64; 2];
                for &t in &seq {
                    effect_num *= classes[t] - used[t];
                    used[t] += 1;
                }
                if effect_num <= 0 {
                    continue;
                }
                let all = if t0 == t1 && t1 == t2 { single[t0] } else { both };
                let number_num = single[t0] * single[t1] * single[t2]
                    - pair(t0, t1) * single[t2]
                    - pair(t0, t2) * single[t1]
                    - pair(t1, t2) * single[t0]
                    + 2 * all;
                num += effect_num * number_num;
            }
        }
    }
    probability(num, falling(total, 3) * falling(total, 3))
}

/// `deck_knowledge._as_counts`: whole-number `f32` counts as integers.
fn counts_of<const N: usize>(values: &[f32; N]) -> [i64; N] {
    let mut out = [0i64; N];
    for i in 0..N {
        out[i] = (values[i] as f64).round_ties_even() as i64;
    }
    out
}

/// `encoder.py::_interval_miss`: over printed 1..15, 1 where `n` is NOT in
/// `(low, high)`.
fn interval_miss(low: i32, high: i32) -> [i64; NUM_NUMBERS] {
    let mut out = [1i64; NUM_NUMBERS];
    for i in 0..NUM_NUMBERS {
        let n = i as i32 + 1;
        if low < n && n < high {
            out[i] = 0;
        }
    }
    out
}

/// `encoder.py::_gap_is_empty` — review F2: no integer strictly inside.
fn gap_is_empty(low: i32, high: i32) -> bool {
    high - low <= 1
}

/// `np.cumsum(np.rint(counts))` with a leading zero.
fn prefix(counts: &[f32; NUM_NUMBERS]) -> [i64; NUM_NUMBERS + 1] {
    let mut out = [0i64; NUM_NUMBERS + 1];
    for i in 0..NUM_NUMBERS {
        out[i + 1] = out[i] + (counts[i] as f64).round_ties_even() as i64;
    }
    out
}

/// Cards whose printed number `n` satisfies `low < n < high`.
fn count_in_open_interval(prefix: &[i64; NUM_NUMBERS + 1], low: i32, high: i32) -> i64 {
    let lo = low.max(0).min(NUM_NUMBERS as i32) as usize;
    let hi = (high - 1).max(0).min(NUM_NUMBERS as i32) as usize;
    if hi > lo {
        prefix[hi] - prefix[lo]
    } else {
        0
    }
}

/// One plan slot's per-sheet facts, computed once per seat and shared by the
/// planes, the plan block and `effect_demand` (Python recomputes them; the
/// values are identical).
struct SlotFacts {
    fraction: f64,
    steps: i32,
    alive: bool,
    turns_lower_bound: i32,
    req: Requirements,
}

fn slot_facts(game: &Game, sheet: &Sheet) -> [SlotFacts; 3] {
    std::array::from_fn(|slot| {
        let plan = &PLANS[game.plan_ids[slot]];
        let (fraction, steps) = progress(plan, sheet);
        let alive = feasible(plan, sheet);
        let req = requirements_given(plan, sheet, alive);
        SlotFacts {
            fraction,
            steps,
            alive,
            turns_lower_bound: turns_lower_bound_given(plan, steps, &req),
            req,
        }
    })
}

/// `encoder.py::_DeckView` — everything the whole state shares.
struct DeckView {
    deck_prefix: [i64; NUM_NUMBERS + 1],
    reshuffled_prefix: [i64; NUM_NUMBERS + 1],
    deck_total: f64,
    reshuffled_total: f64,
    deck_matrix: Matrix,
    reshuffled_matrix: Matrix,
    deck_numbers: [f32; NUM_NUMBERS],
    deck_effects: [f32; NUM_EFFECTS],
    reshuffled_numbers: [f32; NUM_NUMBERS],
    reshuffled_effects: [f32; NUM_EFFECTS],
    deck_counts: [i64; NUM_NUMBERS],
    reform_counts: [i64; NUM_NUMBERS],
    reshuffled_counts: [i64; NUM_NUMBERS],
    effect_rate: [f64; NUM_EFFECTS],
    next_effects: [Option<Effect>; 3],
    next_is_temp: [bool; 3],
    /// ⚠ Review F3: the viewer's OWN yes vote queues a reshuffle — only that
    /// vote is readable, never the table-wide aggregate.
    viewer_voted: bool,
}

impl DeckView {
    fn new(game: &Game, viewer: usize) -> DeckView {
        let deck_matrix = deck_composition(game, viewer);
        let pool_matrix = add_matrix(&discard_composition(game), &aside_composition(game));
        let reshuffled_matrix = add_matrix(&deck_matrix, &pool_matrix);
        let deck_numbers = row_sums(&deck_matrix);
        let reform_numbers = row_sums(&pool_matrix);
        let reshuffled_numbers = row_sums(&reshuffled_matrix);

        let deck_prefix = prefix(&deck_numbers);
        // `_prefix(deck + pool)`: the f32 sum of two row-sum vectors.
        let mut deck_plus_pool = [0.0f32; NUM_NUMBERS];
        for i in 0..NUM_NUMBERS {
            deck_plus_pool[i] = deck_numbers[i] + reform_numbers[i];
        }
        let reshuffled_prefix = prefix(&deck_plus_pool);

        // §9.3 effect_supply_rate, branching on the viewer's OWN vote.
        let viewer_voted = game.reshuffle_vote_for(viewer);
        let (rate_deck, rate_pool) = if viewer_voted {
            (counts_of(&column_sums(&reshuffled_matrix)), [0i64; NUM_EFFECTS])
        } else {
            (
                counts_of(&column_sums(&deck_matrix)),
                counts_of(&column_sums(&pool_matrix)),
            )
        };
        let mut effect_rate = [0.0f64; NUM_EFFECTS];
        for e in 0..NUM_EFFECTS {
            let mut miss = [1i64; NUM_EFFECTS];
            miss[e] = 0;
            let p = next_draw_probability(&rate_deck, &rate_pool, [&miss, &miss, &miss]);
            effect_rate[e] = (1.0 - p).max(0.0).min(1.0);
        }

        let next_effects = game.next_effects(viewer);
        let next_is_temp = [
            next_effects[0] == Some(Effect::Temp),
            next_effects[1] == Some(Effect::Temp),
            next_effects[2] == Some(Effect::Temp),
        ];

        DeckView {
            deck_total: deck_prefix[NUM_NUMBERS] as f64,
            reshuffled_total: reshuffled_prefix[NUM_NUMBERS] as f64,
            deck_prefix,
            reshuffled_prefix,
            deck_effects: column_sums(&deck_matrix),
            reshuffled_effects: column_sums(&reshuffled_matrix),
            deck_counts: counts_of(&deck_numbers),
            reform_counts: counts_of(&reform_numbers),
            reshuffled_counts: counts_of(&reshuffled_numbers),
            deck_matrix,
            reshuffled_matrix,
            deck_numbers,
            reshuffled_numbers,
            effect_rate,
            next_effects,
            next_is_temp,
            viewer_voted,
        }
    }

    /// P(all three of next turn's NUMBERS land in `mask`) — effects unused.
    fn next_numbers_all_in(&self, mask: &[i64; NUM_NUMBERS]) -> f64 {
        if self.viewer_voted {
            return next_draw_probability(
                &self.reshuffled_counts,
                &[0i64; NUM_NUMBERS],
                [mask, mask, mask],
            );
        }
        next_draw_probability(&self.deck_counts, &self.reform_counts, [mask, mask, mask])
    }

    /// P(every stack lands in its mask), the mask depending on its effect.
    fn next_stacks_all_in(
        &self,
        mask_non_temp: &[i64; NUM_NUMBERS],
        mask_temp: &[i64; NUM_NUMBERS],
    ) -> f64 {
        if self.viewer_voted {
            return two_triple_probability(&self.reshuffled_matrix, mask_non_temp, mask_temp);
        }
        let pick = |temp: bool| -> &[i64] { if temp { mask_temp } else { mask_non_temp } };
        next_draw_probability(
            &self.deck_counts,
            &self.reform_counts,
            [
                pick(self.next_is_temp[0]),
                pick(self.next_is_temp[1]),
                pick(self.next_is_temp[2]),
            ],
        )
    }

    fn fit_deck(&self, low: i32, high: i32) -> f64 {
        ratio(
            count_in_open_interval(&self.deck_prefix, low, high) as f64,
            self.deck_total,
        )
    }

    fn fit_reshuffled(&self, low: i32, high: i32) -> f64 {
        ratio(
            count_in_open_interval(&self.reshuffled_prefix, low, high) as f64,
            self.reshuffled_total,
        )
    }
}

/// §7.5: P(some stack next turn reveals a number fitting this gap). An empty
/// gap is 0 (review F2); the viewer's vote switches pool and effects (F3).
fn p_fit_next_turn(view: &DeckView, low: i32, high: i32) -> f64 {
    if gap_is_empty(low, high) {
        return 0.0;
    }
    let miss = view.next_stacks_all_in(&interval_miss(low, high), &interval_miss(low - 2, high + 2));
    (1.0 - miss).max(0.0).min(1.0)
}

// ──────────────────────────────────────────────────────────────────────────
// Spatial planes
// ──────────────────────────────────────────────────────────────────────────

/// `_roundabout_bounds`: mirrors `span_if_roundabout`'s tie-break exactly.
fn roundabout_bounds(sheet: &Sheet, x: usize, y: usize, available: bool) -> (i32, i32) {
    let (first, last, low, high) = sheet.gap_bounds(x, y).expect("empty box");
    if !available {
        return (low, high);
    }
    let mut best = (low, high);
    let mut best_span = (high - low - 1).max(0);
    if y > first {
        let span = (high - LOW_SENTINEL - 1).max(0);
        if span > best_span {
            best = (LOW_SENTINEL, high);
            best_span = span;
        }
    }
    if y < last {
        let span = (HIGH_SENTINEL - low - 1).max(0);
        if span > best_span {
            best = (low, HIGH_SENTINEL);
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn write_sheet_planes(
    game: &Game,
    viewer: usize,
    seat: usize,
    sheet: &Sheet,
    base_numbers: &[i32],
    all_numbers: &[i32],
    view: &DeckView,
    facts: &[SlotFacts; 3],
    axis: usize,
    out: &mut [f32],
) {
    let mut writable_base = [[false; MAX_STREET_LEN]; NUM_STREETS];
    for &n in base_numbers {
        for (x, y) in sheet.available_locations(Some(n)) {
            writable_base[x][y] = true;
        }
    }
    let mut writable_any = writable_base;
    for &n in all_numbers {
        for (x, y) in sheet.available_locations(Some(n)) {
            writable_any[x][y] = true;
        }
    }

    let spans = sheet.box_spans();
    let roundabout_open = game.config.advanced && sheet.can_build_roundabout();
    let spans_ra = sheet.span_if_roundabout(roundabout_open);

    let mut targets = [[[false; MAX_STREET_LEN]; NUM_STREETS]; 3];
    for slot in 0..3 {
        if banked(game, viewer, seat, slot) {
            continue;
        }
        for &(x, y) in &facts[slot].req.target_boxes {
            targets[slot][x][y] = true;
        }
    }

    let mut set = |plane: usize, x: usize, y: usize, value: f32| {
        out[sheet_plane_index(axis, plane, x, y)] = value;
    };

    for x in 0..NUM_STREETS {
        let size = STREET_SIZES[x];
        for y in 0..size {
            let n = sheet.numbers[x][y];
            set(P_VALID, x, y, 1.0);
            if n != EMPTY {
                set(P_WRITTEN, x, y, 1.0);
                if n == ROUNDABOUT {
                    set(P_ROUNDABOUT, x, y, 1.0);
                } else {
                    set(P_NUMBER, x, y, ratio_i32(n, 17));
                }
            }
            set(P_BIS, x, y, sheet.is_bis[x][y] as u8 as f32);
            set(P_TOP_FENCE, x, y, sheet.top_fences[x][y] as u8 as f32);
            if y + 1 < size {
                set(P_FENCE_RIGHT, x, y, sheet.fences[x][y] as u8 as f32);
            }
            set(P_POOL, x, y, POOL_POSITIONS.contains(&(x, y)) as u8 as f32);
            set(P_WRITABLE, x, y, writable_base[x][y] as u8 as f32);
            set(
                P_WRITABLE_TEMP,
                x,
                y,
                (writable_any[x][y] && !writable_base[x][y]) as u8 as f32,
            );
            set(P_SPAN, x, y, ratio_i32(spans[x][y], 18));
            set(P_SPAN_ROUNDABOUT, x, y, ratio_i32(spans_ra[x][y], 18));
            for k in 0..3 {
                set(P_PLAN_TARGET[k], x, y, targets[k][x][y] as u8 as f32);
            }

            if n != EMPTY {
                continue;
            }

            // ⚠ §5.2: positional_fit over the DELTA-0 numbers only.
            let mut best: Option<f64> = None;
            for &v in base_numbers {
                if let Some(fit) = sheet.positional_fit(v, x, y) {
                    best = Some(best.map_or(fit, |old: f64| old.max(fit)));
                }
            }
            if let Some(fit) = best {
                set(P_FIT, x, y, (1.0f64 / (1.0f64 - fit)) as f32);
            }

            let (_first, _last, low, high) = sheet.gap_bounds(x, y).expect("empty box");
            set(P_FIT_DECK, x, y, view.fit_deck(low, high) as f32);
            let fit_temp = if gap_is_empty(low, high) {
                0.0
            } else {
                view.fit_deck(low - 2, high + 2)
            };
            set(P_FIT_TEMP, x, y, fit_temp as f32);
            set(P_FIT_RESHUFFLE, x, y, view.fit_reshuffled(low, high) as f32);
            set(P_FIT_NEXT_TURN, x, y, p_fit_next_turn(view, low, high) as f32);

            let (ra_low, ra_high) = roundabout_bounds(sheet, x, y, roundabout_open);
            set(P_FIT_ROUNDABOUT, x, y, view.fit_deck(ra_low, ra_high) as f32);
        }
    }
    for (x, start, estate_size) in sheet.estates() {
        for y in start..start + estate_size {
            out[sheet_plane_index(axis, P_ESTATE_SIZE, x, y)] = estate_size as f32 / 6.0f32;
        }
    }
}

/// ⚠ `game.ctx` belongs to `game.actor`, so this reads it only when the viewer
/// *is* the actor. See `encoder.py::_viewer_plane` for the full argument.
fn write_viewer_plane(game: &Game, viewer: usize, out: &mut [f32]) {
    if viewer != game.actor {
        return;
    }
    let sheet = &game.sheets[viewer];
    let mut boxes = [[false; MAX_STREET_LEN]; NUM_STREETS];
    if game.phase == Phase::WriteNumber {
        if let (Some(number), Some(effect)) = (game.ctx.number, game.ctx.effect) {
            for candidate in game.numbers_for(number, effect) {
                for (x, y) in sheet.available_locations(Some(candidate)) {
                    boxes[x][y] = true;
                }
            }
        }
    } else if game.phase == Phase::RoundaboutPlace {
        for (x, y) in sheet.available_locations(None) {
            boxes[x][y] = true;
        }
    }
    for x in 0..NUM_STREETS {
        for y in 0..STREET_SIZES[x] {
            out[viewer_plane_index(x, y)] = boxes[x][y] as u8 as f32;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Flat features
// ──────────────────────────────────────────────────────────────────────────
struct Writer {
    buf: Vec<f32>,
    pos: usize,
}

impl Writer {
    fn new(size: usize) -> Writer {
        Writer {
            buf: vec![0.0; size],
            pos: 0,
        }
    }

    fn put(&mut self, value: f32) {
        self.buf[self.pos] = value;
        self.pos += 1;
    }

    fn put_f64(&mut self, value: f64) {
        self.put(value as f32);
    }

    fn put_array(&mut self, values: &[f32]) {
        self.buf[self.pos..self.pos + values.len()].copy_from_slice(values);
        self.pos += values.len();
    }

    fn one_hot(&mut self, index: Option<usize>, size: usize) {
        if let Some(index) = index {
            if index < size {
                self.buf[self.pos + index] = 1.0;
            }
        }
        self.pos += size;
    }

    fn skip(&mut self, count: usize) {
        self.pos += count;
    }
}

/// ⚠ SPEC GAP 4: marks of each effect this plan still needs, as ordered
/// `(effect index, marks)` pairs — the order is `encoder.py::_effect_needs`'s
/// dict order, which the float sum depends on.
fn effect_needs(plan: &Plan, req: &Requirements, steps_left: i32) -> Vec<(usize, i32)> {
    match plan.kind {
        PlanKind::SevenTemp => vec![(E_TEMP, req.temps_needed)],
        PlanKind::Estate => vec![(E_SURVEYOR, req.estate_steps_left)],
        PlanKind::FiveBis => (0..NUM_STREETS)
            .filter(|&x| req.street_serves[x] != 0)
            .map(|x| req.bis_needed[x])
            .min()
            .map(|m| vec![(E_BIS, m)])
            .unwrap_or_default(),
        PlanKind::FullStreet | PlanKind::Extremities => Vec::new(),
        PlanKind::CompleteStreet => (0..NUM_STREETS)
            .filter(|&x| req.street_serves[x] != 0)
            .min_by_key(|&x| {
                (
                    req.parks_needed[x] + req.pools_needed[x] + req.roundabout_needed[x],
                    x,
                )
            })
            .map(|best| {
                vec![
                    (E_PARK, req.parks_needed[best]),
                    (E_POOL, req.pools_needed[best]),
                ]
            })
            .unwrap_or_default(),
        PlanKind::Decorative => match plan.params[0].text() {
            "pool&park" => {
                let x = plan.params[1].int() as usize;
                vec![(E_PARK, req.parks_needed[x]), (E_POOL, req.pools_needed[x])]
            }
            "park" => vec![(E_PARK, steps_left)],
            _ => vec![(E_POOL, steps_left)],
        },
        PlanKind::Unsupported => Vec::new(),
    }
}

/// One plan slot's 34 floats (§3.4, without the demoted §6.4 pair).
#[allow(clippy::too_many_arguments)]
fn plan_block(
    game: &Game,
    viewer: usize,
    seat: usize,
    sheet: &Sheet,
    slot: usize,
    view: &DeckView,
    facts: &[SlotFacts; 3],
    w: &mut Writer,
) {
    let plan = &PLANS[game.plan_ids[slot]];
    let facts = &facts[slot];
    let (fraction, steps) = (facts.fraction, facts.steps);
    let is_banked = banked(game, viewer, seat, slot);
    let req = &facts.req;

    w.put_f64(fraction);
    w.put_f64((steps as f64).min(STEPS_SCALE) / STEPS_SCALE);
    w.put(is_banked as u8 as f32);

    w.put_f64(ratio(req.temps_needed as f64, 7.0));
    w.put_f64(if plan.kind == PlanKind::Estate {
        ratio(req.estate_steps_left as f64, 6.0)
    } else {
        0.0
    });

    for s in 0..MAX_ESTATE_SIZE {
        w.put_f64(ratio(req.estate_shortfall[s] as f64, 6.0));
    }

    for x in 0..NUM_STREETS {
        w.put_f64(ratio(req.parks_needed[x] as f64, PARK_BOXES[x] as f64));
        w.put_f64(ratio(req.pools_needed[x] as f64, 3.0));
        w.put_f64(ratio(req.houses_needed[x] as f64, STREET_SIZES[x] as f64));
        w.put_f64(ratio(req.bis_needed[x] as f64, 5.0));
        w.put_f64(req.roundabout_needed[x] as f64);
        w.put_f64(req.street_serves[x] as f64);
    }

    w.put(if facts.alive { 1.0 } else { 0.0 });
    w.put_f64(turns(facts.turns_lower_bound as f64));

    let mut effect_turns_raw = 0.0f64;
    for (effect, marks) in effect_needs(plan, req, steps) {
        if marks <= 0 {
            continue;
        }
        effect_turns_raw += marks as f64 / view.effect_rate[effect].max(EPS);
    }
    w.put_f64(turns(effect_turns_raw));

    // §6.3 number_rate_supply: the deck fraction supplying the target gaps.
    let houses_total: i32 = req.houses_needed.iter().sum();
    let mut supply_numbers = [false; NUM_NUMBERS];
    for &(x, y) in &req.target_boxes {
        if let Some((_f, _l, low, high)) = sheet.gap_bounds(x, y) {
            for n in 1..=NUM_NUMBERS as i32 {
                if low < n && n < high {
                    supply_numbers[(n - 1) as usize] = true;
                }
            }
        }
    }
    let mut supply = 0.0f64;
    for i in 0..NUM_NUMBERS {
        if supply_numbers[i] {
            supply += view.deck_numbers[i] as f64;
        }
    }
    let number_rate_supply = ratio(supply, view.deck_total);
    w.put_f64(number_rate_supply);

    let number_turns_raw = if houses_total != 0 {
        houses_total as f64 / number_rate_supply.max(EPS)
    } else {
        0.0
    };
    w.put_f64(turns(effect_turns_raw.max(number_turns_raw)));
}

/// §7.2: legal `(empty box, value)` pairs, per value 0..17.
fn number_demand(sheet: &Sheet) -> [f64; NUM_NUMBER_VALUES] {
    let mut demand = [0.0f64; NUM_NUMBER_VALUES];
    for x in 0..NUM_STREETS {
        let size = STREET_SIZES[x];
        let mut y = 0usize;
        while y < size {
            if sheet.numbers[x][y] != EMPTY {
                y += 1;
                continue;
            }
            let (first, last, low, high) = sheet.gap_bounds(x, y).expect("empty box");
            let length = (last - first + 1) as f64;
            let lo = (low + 1).max(MIN_NUMBER);
            let hi = (high - 1).min(MAX_NUMBER);
            if hi >= lo {
                for v in lo..=hi {
                    demand[v as usize] += length;
                }
            }
            y = last + 1;
        }
    }
    demand
}

/// §7.2: the scoring value of one more ESTATE mark, over ALL estates.
fn estate_demand(sheet: &Sheet) -> f64 {
    let counts = sheet.estate_size_counts();
    let mut total = 0.0f64;
    for i in 0..MAX_ESTATE_SIZE {
        let marks = sheet.estate_marks[i];
        if marks >= ESTATE_ROW_BOXES[i] {
            continue;
        }
        let row = ESTATE_ROW_SCORES[i];
        let delta = row[(marks + 1) as usize] - row[marks as usize];
        total += (counts[i] * delta) as f64;
    }
    total
}

fn effect_demand(
    game: &Game,
    sheet: &Sheet,
    viewer: usize,
    seat: usize,
    facts: &[SlotFacts; 3],
) -> [f64; NUM_EFFECTS] {
    let mut out = [0.0f64; NUM_EFFECTS];
    let park_total: i32 = PARK_BOXES.iter().sum();
    let parks: i32 = sheet.parks.iter().sum();
    out[E_PARK] = ratio(((park_total - parks) as f64).max(0.0), park_total as f64);
    out[E_POOL] = ratio(((POOL_BOXES - sheet.pool_count()) as f64).max(0.0), POOL_BOXES as f64);
    out[E_TEMP] = ratio(((TEMP_BOXES - sheet.temps) as f64).max(0.0), TEMP_BOXES as f64);
    out[E_BIS] = ratio(((BIS_BOXES - sheet.bis_marks) as f64).max(0.0), BIS_BOXES as f64);

    let mut fences = 0i32;
    for slot in 0..3 {
        if banked(game, viewer, seat, slot) {
            continue;
        }
        let plan = &PLANS[game.plan_ids[slot]];
        if plan.kind == PlanKind::Estate {
            fences += facts[slot].req.estate_steps_left;
        }
    }
    out[E_SURVEYOR] = ratio(fences as f64, SURVEYOR_DEMAND_SCALE);
    out[E_ESTATE] = ratio(estate_demand(sheet), ESTATE_DEMAND_SCALE);
    out
}

/// §7.3: per printed number 1..15, how many empty BOXES it could serve.
fn card_demand(sheet: &Sheet, with_temp: bool) -> [f64; NUM_NUMBERS] {
    let mut out = [0.0f64; NUM_NUMBERS];
    let deltas: &[i32] = if with_temp { &TEMP_DELTAS } else { &[0] };
    for x in 0..NUM_STREETS {
        let size = STREET_SIZES[x];
        let mut y = 0usize;
        while y < size {
            if sheet.numbers[x][y] != EMPTY {
                y += 1;
                continue;
            }
            let (first, last, low, high) = sheet.gap_bounds(x, y).expect("empty box");
            let length = (last - first + 1) as f64;
            for i in 0..NUM_NUMBERS {
                let n = i as i32 + 1;
                for &d in deltas {
                    let v = (n + d).max(MIN_NUMBER).min(MAX_NUMBER);
                    if low < v && v < high {
                        out[i] += length;
                        break;
                    }
                }
            }
            y = last + 1;
        }
    }
    out
}

/// §7.3's 8 floats. ⚠ SPEC GAP 2: the temp half's two effect entries equal the
/// no-temp half's by construction; the literal width is kept.
fn reshuffle_contraction(
    sheet: &Sheet,
    effect_demand: &[f64; NUM_EFFECTS],
    view: &DeckView,
    w: &mut Writer,
) {
    let fit_rate = |demand: &[f64; NUM_NUMBERS], supply: &[f32; NUM_NUMBERS]| -> f64 {
        // Integer-valued on both sides, so the sum is exact in any order.
        let total: f32 = supply.iter().sum();
        let mut num = 0.0f64;
        for i in 0..NUM_NUMBERS {
            num += demand[i] * supply[i] as f64;
        }
        ratio(num, FIT_RATE_BOX_SCALE * total as f64)
    };
    let eff_rate = |supply: &[f32; NUM_EFFECTS]| -> f64 {
        // ⚠ §10.6: left to right, as encoder.py does it.
        let mut demand_total = 0.0f64;
        let mut weighted = 0.0f64;
        for e in 0..NUM_EFFECTS {
            demand_total += effect_demand[e];
            weighted += effect_demand[e] * supply[e] as f64;
        }
        let supply_total: f32 = supply.iter().sum();
        ratio(weighted, supply_total as f64 * demand_total)
    };
    // SPEC GAP 2 as resolved by review: 6 floats, the two duplicates dropped.
    let plain = card_demand(sheet, false);
    w.put_f64(fit_rate(&plain, &view.deck_numbers));
    w.put_f64(eff_rate(&view.deck_effects));
    w.put_f64(fit_rate(&plain, &view.reshuffled_numbers));
    w.put_f64(eff_rate(&view.reshuffled_effects));
    let temp = card_demand(sheet, true);
    w.put_f64(fit_rate(&temp, &view.deck_numbers));
    w.put_f64(fit_rate(&temp, &view.reshuffled_numbers));
}

fn any_writable(writable: u32, number: i32, effect: Effect) -> bool {
    numbers_for(number, effect)
        .into_iter()
        .any(|v| writable >> v & 1 == 1)
}

type MissMasks = ([i64; NUM_NUMBERS], [i64; NUM_NUMBERS]);

/// `encoder.py::_miss_masks_from`: over printed 1..15, 1 where a card has NO
/// legal write — `(non-TEMP effect, TEMP effect)`.
fn miss_masks_from(writable: u32) -> MissMasks {
    let mut non_temp = [1i64; NUM_NUMBERS];
    let mut temp = [1i64; NUM_NUMBERS];
    for i in 0..NUM_NUMBERS {
        let n = i as i32 + 1;
        if writable >> n & 1 == 1 {
            non_temp[i] = 0;
        }
        if any_writable(writable, n, Effect::Temp) {
            temp[i] = 0;
        }
    }
    (non_temp, temp)
}

/// `encoder.py::_future_roundabout_masks`.
fn future_roundabout_masks(game: &Game, sheet: &Sheet) -> Vec<u32> {
    if !(game.config.advanced && sheet.can_build_roundabout() && sheet.has_free_box()) {
        return Vec::new();
    }
    sheet.roundabout_writable_masks()
}

/// `encoder.py::_roundabout_legal_now`: THIS turn, for this seat.
fn roundabout_legal_now(game: &Game, viewer: usize, seat: usize, sheet: &Sheet) -> bool {
    if !(game.config.advanced && sheet.can_build_roundabout() && sheet.has_free_box()) {
        return false;
    }
    if seat == viewer && viewer == game.actor {
        if game.phase == Phase::RoundaboutPlace {
            return true;
        }
        return game.phase == Phase::ChooseCards
            && game.ctx.last_house.is_none()
            && !game.ctx.roundabout_declined;
    }
    true
}

/// `encoder.py::_rescue_this_turn` — existential over placements.
fn rescue_this_turn(game: &Game, viewer: usize, seat: usize, sheet: &Sheet) -> bool {
    if !roundabout_legal_now(game, viewer, seat, sheet) {
        return false;
    }
    let writable = sheet.writable_mask();
    let blocked: Vec<(i32, Effect)> = offers(game, viewer)
        .into_iter()
        .filter(|&(n, e)| !any_writable(writable, n, e))
        .collect();
    if blocked.is_empty() {
        return false;
    }
    sheet
        .roundabout_writable_masks()
        .into_iter()
        .any(|after| blocked.iter().any(|&(n, e)| any_writable(after, n, e)))
}

/// §8's 5 floats — see `encoder.py::_refusal_block` for each definition.
/// Minima over placements are order-free, so the deduplicating set needs no
/// particular order to match Python.
fn refusal_block(
    game: &Game,
    viewer: usize,
    seat: usize,
    sheet: &Sheet,
    view: &DeckView,
    w: &mut Writer,
) {
    let (non_temp, temp) = miss_masks_from(sheet.writable_mask());
    let candidates: BTreeSet<MissMasks> = future_roundabout_masks(game, sheet)
        .into_iter()
        .collect::<BTreeSet<u32>>()
        .into_iter()
        .map(miss_masks_from)
        .collect();

    let p_now = view.next_stacks_all_in(&non_temp, &temp);
    let mut p_after = p_now;
    for (a, b) in &candidates {
        p_after = p_after.min(view.next_stacks_all_in(a, b));
    }
    w.put_f64(p_now);
    w.put_f64(p_after);

    let mut placeable = [0i64; NUM_NUMBERS];
    for i in 0..NUM_NUMBERS {
        placeable[i] = 1 - non_temp[i];
    }
    w.put_f64((1.0 - view.next_numbers_all_in(&placeable)).max(0.0).min(1.0));

    w.put(if rescue_this_turn(game, viewer, seat, sheet) { 1.0 } else { 0.0 });

    let mut population = &view.deck_matrix;
    if population.iter().flatten().sum::<f32>() < 6.0 {
        population = &view.reshuffled_matrix;
    }
    if population.iter().flatten().sum::<f32>() < 3.0 {
        w.put(0.0);
        return;
    }
    let mut steady = two_triple_probability(population, &non_temp, &temp);
    for (a, b) in &candidates {
        steady = steady.min(two_triple_probability(population, a, b));
    }
    w.put_f64(steady);
}

/// `game.py::max_houses_this_turn` — 0-3, the REMAINING turn for the acting
/// viewer (review F5), a hypothetical full turn for everyone else.
pub fn max_houses_this_turn(game: &Game, viewer: usize, seat: usize) -> i32 {
    let sheet = game.sheet_for(viewer, seat);
    let mut offers = offers(game, viewer);
    let mut roundabout_ok =
        game.config.advanced && sheet.can_build_roundabout() && sheet.has_free_box();
    if seat == viewer && viewer == game.actor {
        match game.phase {
            Phase::ChooseCards => {
                roundabout_ok = roundabout_ok
                    && game.ctx.last_house.is_none()
                    && !game.ctx.roundabout_declined;
            }
            Phase::RoundaboutPlace => {}
            Phase::WriteNumber => {
                let (Some(number), Some(effect)) = (game.ctx.number, game.ctx.effect) else {
                    return 0;
                };
                offers = vec![(number, effect)];
                roundabout_ok = false;
            }
            Phase::ActionBis => {
                return if sheet.bis_candidates().is_empty() { 0 } else { 1 };
            }
            _ => return 0,
        }
    }
    max_houses_from(sheet, &offers, roundabout_ok)
}

/// `game.py::_max_houses_from`: optional roundabout, write, bis.
fn max_houses_from(sheet: &Sheet, offers: &[(i32, Effect)], roundabout_ok: bool) -> i32 {
    let mut starts: Vec<(Sheet, i32)> = vec![(sheet.clone(), 0)];
    if roundabout_ok {
        for pos in sheet.available_locations(None) {
            let mut opened = sheet.clone();
            opened.build_roundabout(pos, 0);
            starts.push((opened, 1));
        }
    }
    // Exact shortcuts, mirrored in game.py.
    let bis_offered = offers.iter().any(|&(_, effect)| effect == Effect::Bis);
    let mut best = 0i32;
    for (start, placed) in &starts {
        best = best.max(*placed);
        if best >= placed + 1 + bis_offered as i32 {
            continue;
        }
        for &(number, effect) in offers {
            for value in numbers_for(number, effect) {
                let locations = start.available_locations(Some(value));
                if effect != Effect::Bis {
                    if !locations.is_empty() {
                        best = best.max(placed + 1);
                    }
                    continue;
                }
                for pos in locations {
                    let mut written = start.clone();
                    written.write(value, pos, 0, false);
                    let mut total = placed + 1;
                    if !written.bis_candidates().is_empty() {
                        total += 1;
                    }
                    best = best.max(total);
                    if best >= 3 {
                        return 3;
                    }
                }
            }
        }
    }
    best.min(3)
}

/// §9.2a's canonical satisfying selection, or `None` if none exists.
fn selected_estates(plan: &Plan, sheet: &Sheet) -> Option<Vec<Pos>> {
    let mut free = sheet.free_estates();
    free.sort();
    let mut sizes = plan.required_sizes();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let mut used = vec![false; free.len()];
    let mut taken = Vec::new();
    for size in sizes {
        let found = (0..free.len()).find(|&i| !used[i] && free[i].2 == size)?;
        used[found] = true;
        let (x, start, length) = free[found];
        for k in 0..length {
            taken.push((x, start + k));
        }
    }
    Some(taken)
}

/// `T(slot)` — the boxes completing this plan would consume (§9.2a).
fn target_boxes(game: &Game, viewer: usize, seat: usize, sheet: &Sheet, slot: usize) -> BTreeSet<Pos> {
    if banked(game, viewer, seat, slot) {
        return BTreeSet::new();
    }
    let plan = &PLANS[game.plan_ids[slot]];
    match plan.kind {
        PlanKind::FullStreet => {
            let x = plan.params[0].int() as usize;
            (0..STREET_SIZES[x]).map(|y| (x, y)).collect()
        }
        PlanKind::Extremities => EXTREMITY_POSITIONS.iter().copied().collect(),
        PlanKind::Estate => selected_estates(plan, sheet)
            .unwrap_or_default()
            .into_iter()
            .collect(),
        _ => BTreeSet::new(),
    }
}

/// §9.2a: 3 unordered overlaps + 6 directed kills.
fn plan_conflict_seat(game: &Game, viewer: usize, seat: usize, sheet: &Sheet, w: &mut Writer) {
    let sets: Vec<BTreeSet<Pos>> = (0..3)
        .map(|s| target_boxes(game, viewer, seat, sheet, s))
        .collect();

    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        if sets[a].is_empty() || sets[b].is_empty() {
            w.put(0.0);
            continue;
        }
        let inter = sets[a].intersection(&sets[b]).count();
        w.put_f64(ratio(inter as f64, sets[a].len().min(sets[b].len()) as f64));
    }

    for (a, b) in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)] {
        if sets[a].is_empty() || banked(game, viewer, seat, b) {
            w.put(0.0);
            continue;
        }
        let mut hypothetical = sheet.clone();
        for &(x, y) in &sets[a] {
            hypothetical.top_fences[x][y] = true;
        }
        let plan_b = &PLANS[game.plan_ids[b]];
        w.put(if feasible(plan_b, &hypothetical) { 0.0 } else { 1.0 });
    }
}

fn sheet_scalars(
    game: &Game,
    viewer: usize,
    seat: usize,
    view: &DeckView,
    facts: &[SlotFacts; 3],
) -> Vec<f32> {
    let sheet = game.sheet_for(viewer, seat);
    let mut w = Writer::new(NUM_SHEET_SCALAR);

    // tracks (26)
    for x in 0..NUM_STREETS {
        w.put(ratio_i32(sheet.parks[x], PARK_BOXES[x]));
    }
    w.put(ratio_i32(sheet.pool_count(), POOL_BOXES));
    w.put(ratio_i32(sheet.temps, TEMP_BOXES));
    w.put(ratio_i32(sheet.bis_marks, BIS_BOXES));
    w.put(ratio_i32(sheet.permits, PERMIT_BOXES));
    w.put(ratio_i32(sheet.roundabouts, ROUNDABOUT_BOXES));
    for i in 0..MAX_ESTATE_SIZE {
        w.put(ratio_i32(sheet.estate_marks[i], ESTATE_ROW_BOXES[i]));
    }
    for count in sheet.estate_size_counts() {
        w.put_f64(count as f64 / 4.0);
    }
    for count in sheet.free_estate_size_counts() {
        w.put_f64(count as f64 / 4.0);
    }

    // score components (9)
    let breakdown = game.score_breakdown(seat, Some(viewer));
    for value in [
        breakdown.parks,
        breakdown.pools,
        breakdown.estates,
        breakdown.plans,
        breakdown.temp,
        breakdown.bis,
        breakdown.permits,
        breakdown.roundabouts,
    ] {
        w.put_f64(value as f64 / SCORE_SCALE);
    }
    w.put_f64(breakdown.total() as f64 / 100.0);

    // placement capacity (4)
    let capacity = sheet.placement_capacity();
    for x in 0..NUM_STREETS {
        w.put(ratio_i32(capacity[x], STREET_SIZES[x] as i32));
    }
    w.put_f64(capacity.iter().sum::<i32>() as f64 / NUM_BOXES as f64);

    // §4: roundabout repair (3) and total span (1)
    let roundabout_open = game.config.advanced && sheet.can_build_roundabout();
    let repair = sheet.capacity_if_roundabout(roundabout_open);
    for x in 0..NUM_STREETS {
        w.put_f64(ratio(repair[x] as f64, STREET_SIZES[x] as f64));
    }
    w.put_f64(ratio(sheet.total_span() as f64, TOTAL_SPAN_SCALE));

    // THE RACE (102)
    for slot in 0..3 {
        plan_block(game, viewer, seat, sheet, slot, view, facts, &mut w);
    }

    // demand (24)
    for value in number_demand(sheet) {
        w.put_f64(value / NUM_BOXES as f64);
    }
    let demand = effect_demand(game, sheet, viewer, seat, facts);
    for value in demand {
        w.put_f64(value);
    }

    // reshuffle contraction (8)
    reshuffle_contraction(sheet, &demand, view, &mut w);

    // refusal (5)
    refusal_block(game, viewer, seat, sheet, view, &mut w);

    // houses this turn (2)
    w.put_f64(max_houses_this_turn(game, viewer, seat) as f64 / 3.0);
    w.put(if sheet.bis_candidates().is_empty() { 0.0 } else { 1.0 });

    // plan conflict (9)
    plan_conflict_seat(game, viewer, seat, sheet, &mut w);

    // free boxes, is_viewer, seat_valid (3)
    let written = (0..NUM_STREETS)
        .map(|x| (0..STREET_SIZES[x]).filter(|&y| sheet.numbers[x][y] != EMPTY).count())
        .sum::<usize>();
    w.put_f64((NUM_BOXES - written) as f64 / NUM_BOXES as f64);
    w.put((seat == viewer) as u8 as f32);
    w.put(1.0);
    debug_assert_eq!(w.pos, NUM_SHEET_SCALAR);
    w.buf
}

fn global_scalars(game: &Game, viewer: usize, view: &DeckView) -> EngineResult<Vec<f32>> {
    let mut w = Writer::new(NUM_GLOBAL_SCALAR);
    w.one_hot(Some(game.phase as usize), 12);
    w.put_f64(game.turn as f64 / TURN_SCALE);

    for slot in 0..3 {
        let (number, effect) = game.combination_faces(slot, viewer);
        w.one_hot(number.and_then(|n| usize::try_from(n).ok()), NUM_NUMBER_VALUES);
        w.one_hot(effect.and_then(effect_index), NUM_EFFECTS);
    }
    let playable = if viewer == game.actor {
        game.playable_slots(viewer)
    } else {
        Vec::new()
    };
    for slot in 0..6 {
        w.put(playable.contains(&slot) as u8 as f32);
    }

    let owns_ctx = viewer == game.actor;
    if owns_ctx && game.ctx.number.is_some() {
        w.one_hot(game.ctx.number.and_then(|n| usize::try_from(n).ok()), NUM_NUMBER_VALUES);
        w.one_hot(game.ctx.effect.and_then(effect_index), NUM_EFFECTS);
        w.put(1.0);
    } else {
        w.skip(NUM_NUMBER_VALUES + NUM_EFFECTS + 1);
    }

    match (owns_ctx, game.ctx.last_house) {
        (true, Some((x, y))) => {
            w.one_hot(Some(box_index(x, y)), NUM_BOXES);
            w.put(1.0);
        }
        _ => w.skip(NUM_BOXES + 1),
    }

    if owns_ctx && !game.ctx.pending_sizes.is_empty() {
        w.one_hot(Some(game.ctx.pending_sizes[0] - 1), 6);
        w.put(1.0);
    } else {
        w.skip(7);
    }

    for slot in 0..3 {
        let plan_id = game.plan_ids[slot];
        let dense = dense_index(plan_id).ok_or_else(|| {
            EngineError::Invalid(format!("plan {plan_id} is not in the encoder vocabulary"))
        })?;
        w.one_hot(Some(dense), NUM_DEALT_PLANS);
        let plan = &PLANS[plan_id];
        w.put_f64(plan.scores.0 as f64 / 20.0);
        w.put_f64(plan.scores.1 as f64 / 20.0);
        w.put(game.plan_turns_for(viewer, slot).is_empty() as u8 as f32);
    }

    w.put(game.may_ask_reshuffle() as u8 as f32);
    w.put(game.reshuffle_vote_for(viewer) as u8 as f32);

    // Blank after the viewer's own yes vote: the effects are redrawn (F3).
    if view.viewer_voted {
        w.skip(3 * NUM_EFFECTS);
    } else {
        for effect in view.next_effects {
            let mut row = [0.0f32; NUM_EFFECTS];
            if let Some(index) = effect.and_then(effect_index) {
                row[index] = 1.0;
            }
            w.put_array(&row);
        }
    }

    let discard = discard_composition(game);
    w.put_f64(game.deck_remaining() as f64 / num_base_cards() as f64);
    w.put_f64(game.discard.len() as f64 / num_base_cards() as f64);
    for value in view.deck_numbers {
        w.put_f64(value as f64 / 9.0);
    }
    for value in view.deck_effects {
        w.put_f64(value as f64 / 20.0);
    }
    for value in row_sums(&discard) {
        w.put_f64(value as f64 / 9.0);
    }
    for value in column_sums(&discard) {
        w.put_f64(value as f64 / 20.0);
    }
    w.put_array(&normalize(&view.deck_numbers, true));
    w.put_array(&normalize(&view.reshuffled_numbers, false));

    // §9.3 boundary-draw rates
    for value in view.effect_rate {
        w.put_f64(value);
    }
    w.put_f64(view.effect_rate[E_TEMP]);
    w.put_f64(view.effect_rate[E_BIS]);

    // §7.4 reveals_to_reform, an upper bound
    w.put_f64(turns((game.deck_remaining() / 3 + 1) as f64));

    w.put(game.config.advanced as u8 as f32);
    w.put(game.config.expert as u8 as f32);
    w.put(game.config.solo() as u8 as f32);
    w.put_f64(game.config.players as f64 / MAX_PLAYERS as f64);
    w.one_hot((viewer < MAX_PLAYERS).then_some(viewer), MAX_PLAYERS);
    let seats = seat_order(game, viewer).len();
    for k in 0..MAX_SEATS {
        w.put((k < seats) as u8 as f32);
    }
    debug_assert_eq!(w.pos, NUM_GLOBAL_SCALAR);
    Ok(w.buf)
}

/// §0.5: the 2+ player standard game only, and never a boundary afterstate.
fn require_scope(game: &Game) -> EngineResult<()> {
    if !game.config.standard() || game.config.players < 2 {
        return Err(EngineError::Invalid(
            "the v3 encoder is defined for the 2+ player standard game only, \
             not expert or one-seat play (ENCODER_V3_SPEC.md §0.5)"
                .into(),
        ));
    }
    if game.boundary_prepared {
        return Err(EngineError::Invalid(
            "the v3 encoder reads a mid-turn state; this is a prepared boundary \
             afterstate, whose discard step has already run"
                .into(),
        ));
    }
    Ok(())
}

pub fn encode_state(game: &Game, viewer: usize) -> EngineResult<EncodedState> {
    if viewer >= game.config.players {
        return Err(EngineError::Invalid(format!(
            "encoder viewer {viewer} is outside {} seats",
            game.config.players
        )));
    }
    require_scope(game)?;
    let mut sheet_planes = vec![0.0f32; SHEET_PLANES_LEN];
    let mut sheet_scalars_out = vec![0.0f32; SHEET_SCALARS_LEN];
    let mut viewer_plane = vec![0.0f32; VIEWER_PLANE_LEN];

    let view = DeckView::new(game, viewer);
    let offered = offers(game, viewer);
    let base_numbers: Vec<i32> = offered.iter().map(|&(n, _)| n).collect();
    let all_numbers: Vec<i32> = offered
        .iter()
        .flat_map(|&(n, e)| numbers_for(n, e))
        .collect();

    for (axis, seat) in seat_order(game, viewer).into_iter().enumerate() {
        let sheet = game.sheet_for(viewer, seat);
        let facts = slot_facts(game, sheet);
        write_sheet_planes(
            game,
            viewer,
            seat,
            sheet,
            &base_numbers,
            &all_numbers,
            &view,
            &facts,
            axis,
            &mut sheet_planes,
        );
        let scalars = sheet_scalars(game, viewer, seat, &view, &facts);
        let start = axis * NUM_SHEET_SCALAR;
        sheet_scalars_out[start..start + NUM_SHEET_SCALAR].copy_from_slice(&scalars);
    }
    write_viewer_plane(game, viewer, &mut viewer_plane);
    Ok(EncodedState {
        sheet_planes,
        sheet_scalars: sheet_scalars_out,
        viewer_plane,
        global_scalars: global_scalars(game, viewer, &view)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Config;

    #[test]
    fn encoder_layout_is_the_v3_shape() {
        assert_eq!(SHEET_PLANES_LEN, 4 * 22 * 36);
        assert_eq!(SHEET_SCALARS_LEN, 4 * 194);
        assert_eq!(VIEWER_PLANE_LEN, 36);
        assert_eq!(NUM_GLOBAL_SCALAR, 367);
        assert_eq!(NUM_DEALT_PLANS, 28);
        assert_eq!(3 * PLAN_SLOT_WIDTH, 102);
    }

    #[test]
    fn a_new_game_encodes_all_viewers() {
        let game = Game::new(
            3,
            Config {
                players: 4,
                advanced: true,
                expert: false,
                solo_rules: false,
            },
        )
        .expect("game");
        for viewer in 0..4 {
            let encoded = encode_state(&game, viewer).expect("encode");
            assert_eq!(encoded.sheet_planes.len(), SHEET_PLANES_LEN);
            assert_eq!(encoded.sheet_scalars.len(), SHEET_SCALARS_LEN);
            assert_eq!(encoded.viewer_plane.len(), VIEWER_PLANE_LEN);
            assert_eq!(encoded.global_scalars.len(), NUM_GLOBAL_SCALAR);
        }
    }

    #[test]
    fn expert_play_is_out_of_scope() {
        let game = Game::new(
            3,
            Config {
                players: 2,
                advanced: false,
                expert: true,
                solo_rules: false,
            },
        )
        .expect("game");
        assert!(encode_state(&game, 0).is_err());
    }
}
