//! Bit-exact mirror of `games/welcome_to/encoder.py` (RUST_PORT_PLAN M3).
//!
//! Python remains the oracle. Integer divisions deliberately happen in `f64`
//! and are cast to `f32` at the write, matching Python-float -> NumPy-f32. The
//! deck histogram is already `f32`, so its normalization stays in `f32`, as
//! NumPy does for those arrays.

use std::sync::OnceLock;

use crate::constants::{
    box_index, card_effect, card_number, num_base_cards, Effect, BIS_BOXES, EMPTY,
    ESTATE_ROW_BOXES, MAX_ESTATE_SIZE, MAX_STREET_LEN, NUM_BOXES, NUM_STREETS, PARK_BOXES,
    PERMIT_BOXES, POOL_BOXES, POOL_POSITIONS, ROUNDABOUT, ROUNDABOUT_BOXES, STREET_SIZES,
    TEMP_BOXES,
};
use crate::game::{EngineError, EngineResult, Game, Phase, NO_CARD};
use crate::plans::{dense_index, progress, NUM_DEALT_PLANS, PLANS};
use crate::sheet::Sheet;

pub const ENCODER_ABI_VERSION: usize = 1;
pub const MAX_SEATS: usize = 4;
pub const MAX_PLAYERS: usize = 6;
pub const SHEET_PLANES: usize = 12;
pub const NUM_SHEET_SCALAR: usize = 45;
pub const NUM_GLOBAL_SCALAR: usize = 358;
pub const SHEET_PLANES_LEN: usize = MAX_SEATS * SHEET_PLANES * NUM_STREETS * MAX_STREET_LEN;
pub const SHEET_SCALARS_LEN: usize = MAX_SEATS * NUM_SHEET_SCALAR;
pub const VIEWER_PLANE_LEN: usize = NUM_STREETS * MAX_STREET_LEN;

const NUM_EFFECTS: usize = 6;
const NUM_NUMBERS: usize = 15;
const NUM_NUMBER_VALUES: usize = 18;
const TURN_SCALE: f64 = 30.0;
const SCORE_SCALE: f64 = 50.0;
const STEPS_SCALE: f64 = 12.0;

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

fn effect_index(effect: Effect) -> Option<usize> {
    Some(match effect {
        Effect::Surveyor => 0,
        Effect::Pool => 1,
        Effect::Temp => 2,
        Effect::Bis => 3,
        Effect::Park => 4,
        Effect::Estate => 5,
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

fn offered_numbers(game: &Game, viewer: usize) -> Vec<i32> {
    let mut out = Vec::new();
    for slot in 0..3 {
        let (number, effect) = game.combination_faces(slot, viewer);
        if let (Some(number), Some(effect)) = (number, effect) {
            out.extend(game.numbers_for(number, effect));
        }
    }
    out
}

fn write_sheet_planes(sheet: &Sheet, offered: &[i32], seat: usize, out: &mut [f32]) {
    let mut writable = [[false; MAX_STREET_LEN]; NUM_STREETS];
    for &number in offered {
        for (x, y) in sheet.available_locations(Some(number)) {
            writable[x][y] = true;
        }
    }
    let spans = sheet.box_spans();
    for x in 0..NUM_STREETS {
        let size = STREET_SIZES[x];
        for y in 0..size {
            let number = sheet.numbers[x][y];
            out[sheet_plane_index(seat, P_VALID, x, y)] = 1.0;
            if number != EMPTY {
                out[sheet_plane_index(seat, P_WRITTEN, x, y)] = 1.0;
                if number == ROUNDABOUT {
                    out[sheet_plane_index(seat, P_ROUNDABOUT, x, y)] = 1.0;
                } else {
                    out[sheet_plane_index(seat, P_NUMBER, x, y)] = ratio_i32(number, 17);
                }
            }
            out[sheet_plane_index(seat, P_BIS, x, y)] = sheet.is_bis[x][y] as u8 as f32;
            out[sheet_plane_index(seat, P_TOP_FENCE, x, y)] = sheet.top_fences[x][y] as u8 as f32;
            if y + 1 < size {
                out[sheet_plane_index(seat, P_FENCE_RIGHT, x, y)] = sheet.fences[x][y] as u8 as f32;
            }
            out[sheet_plane_index(seat, P_POOL, x, y)] =
                POOL_POSITIONS.contains(&(x, y)) as u8 as f32;
            out[sheet_plane_index(seat, P_WRITABLE, x, y)] = writable[x][y] as u8 as f32;
            out[sheet_plane_index(seat, P_SPAN, x, y)] = ratio_i32(spans[x][y], 18);
            if number == EMPTY {
                let mut best: Option<f64> = None;
                for &candidate in offered {
                    if let Some(fit) = sheet.positional_fit(candidate, x, y) {
                        best = Some(best.map_or(fit, |old| old.max(fit)));
                    }
                }
                if let Some(fit) = best {
                    out[sheet_plane_index(seat, P_FIT, x, y)] = (1.0f64 / (1.0f64 - fit)) as f32;
                }
            }
        }
    }
    for (x, start, estate_size) in sheet.estates() {
        for y in start..start + estate_size {
            out[sheet_plane_index(seat, P_ESTATE_SIZE, x, y)] = estate_size as f32 / 6.0f32;
        }
    }
}

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

fn sheet_scalars(game: &Game, viewer: usize, seat: usize) -> Vec<f32> {
    let sheet = game.sheet_for(viewer, seat);
    let mut w = Writer::new(NUM_SHEET_SCALAR);

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

    let capacity = sheet.placement_capacity();
    for x in 0..NUM_STREETS {
        w.put(ratio_i32(capacity[x], STREET_SIZES[x] as i32));
    }
    w.put_f64(capacity.iter().sum::<i32>() as f64 / NUM_BOXES as f64);

    for slot in 0..3 {
        let (fraction, steps) = progress(&PLANS[game.plan_ids[slot]], sheet);
        let banked = game
            .plan_turns_for(viewer, slot)
            .iter()
            .any(|&(player, _)| player == seat as i32);
        w.put_f64(fraction);
        w.put_f64((steps as f64).min(STEPS_SCALE) / STEPS_SCALE);
        w.put(banked as u8 as f32);
    }

    let written = (0..NUM_STREETS)
        .map(|x| {
            (0..STREET_SIZES[x])
                .filter(|&y| sheet.numbers[x][y] != EMPTY)
                .count()
        })
        .sum::<usize>();
    w.put_f64((NUM_BOXES - written) as f64 / NUM_BOXES as f64);
    w.put((seat == viewer) as u8 as f32);
    w.put(1.0);
    debug_assert_eq!(w.pos, NUM_SHEET_SCALAR);
    w.buf
}

/// ⚠ `game.ctx` belongs to `game.actor`, so this reads it only when the viewer
/// *is* the actor. Ungated it would answer "where could the viewer write the
/// number the opponent just picked" — meaningless, a read of hidden mid-turn
/// state, and invisible to `information_key`, which carries `ctx` only on the
/// viewer's own turn. See `encoder.py::_viewer_plane` for the full argument.
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
/// is once per search transition, not once per evaluated leaf — and rebuilding
/// this allocated an 81-element `Vec` and walked the card table every time.
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
    // Accumulated in place rather than into a joined `Vec`: the previous
    // spelling cloned `discard` (up to 81 cards) and appended the table on
    // every call. Counts are whole numbers held in `f32`, so summation order
    // is exact and this stays bit-identical to the Python oracle.
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

fn global_scalars(game: &Game, viewer: usize) -> EngineResult<Vec<f32>> {
    let mut w = Writer::new(NUM_GLOBAL_SCALAR);
    w.one_hot(Some(game.phase as usize), 12);
    w.put_f64(game.turn as f64 / TURN_SCALE);

    for slot in 0..3 {
        let (number, effect) = game.combination_faces(slot, viewer);
        w.one_hot(
            number.and_then(|n| usize::try_from(n).ok()),
            NUM_NUMBER_VALUES,
        );
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
    if owns_ctx {
        if let Some(number) = game.ctx.number {
            w.one_hot(usize::try_from(number).ok(), NUM_NUMBER_VALUES);
            w.one_hot(game.ctx.effect.and_then(effect_index), NUM_EFFECTS);
            w.put(1.0);
        } else {
            w.skip(NUM_NUMBER_VALUES + NUM_EFFECTS + 1);
        }
    } else {
        w.skip(NUM_NUMBER_VALUES + NUM_EFFECTS + 1);
    }

    if owns_ctx {
        if let Some((x, y)) = game.ctx.last_house {
            w.one_hot(Some(box_index(x, y)), NUM_BOXES);
            w.put(1.0);
        } else {
            w.skip(NUM_BOXES + 1);
        }
    } else {
        w.skip(NUM_BOXES + 1);
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

    for effect in game.next_effects(viewer) {
        let mut row = [0.0f32; NUM_EFFECTS];
        if let Some(index) = effect.and_then(effect_index) {
            row[index] = 1.0;
        }
        w.put_array(&row);
    }

    let deck = deck_composition(game, viewer);
    let discard = discard_composition(game);
    let reshuffled = add_matrix(&add_matrix(&deck, &discard), &aside_composition(game));
    w.put_f64(game.deck_remaining() as f64 / num_base_cards() as f64);
    w.put_f64(game.discard.len() as f64 / num_base_cards() as f64);

    let deck_numbers = row_sums(&deck);
    let deck_effects = column_sums(&deck);
    let discard_numbers = row_sums(&discard);
    let discard_effects = column_sums(&discard);
    for value in deck_numbers {
        w.put_f64(value as f64 / 9.0);
    }
    for value in deck_effects {
        w.put_f64(value as f64 / 20.0);
    }
    for value in discard_numbers {
        w.put_f64(value as f64 / 9.0);
    }
    for value in discard_effects {
        w.put_f64(value as f64 / 20.0);
    }
    w.put_array(&normalize(&deck_numbers, true));
    w.put_array(&normalize(&row_sums(&reshuffled), false));

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

pub fn encode_state(game: &Game, viewer: usize) -> EngineResult<EncodedState> {
    if viewer >= game.config.players {
        return Err(EngineError::Invalid(format!(
            "encoder viewer {viewer} is outside {} seats",
            game.config.players
        )));
    }
    let mut sheet_planes = vec![0.0f32; SHEET_PLANES_LEN];
    let mut sheet_scalars_out = vec![0.0f32; SHEET_SCALARS_LEN];
    let mut viewer_plane = vec![0.0f32; VIEWER_PLANE_LEN];
    let offered = offered_numbers(game, viewer);
    for (axis, seat) in seat_order(game, viewer).into_iter().enumerate() {
        write_sheet_planes(
            game.sheet_for(viewer, seat),
            &offered,
            axis,
            &mut sheet_planes,
        );
        let scalars = sheet_scalars(game, viewer, seat);
        let start = axis * NUM_SHEET_SCALAR;
        sheet_scalars_out[start..start + NUM_SHEET_SCALAR].copy_from_slice(&scalars);
    }
    write_viewer_plane(game, viewer, &mut viewer_plane);
    Ok(EncodedState {
        sheet_planes,
        sheet_scalars: sheet_scalars_out,
        viewer_plane,
        global_scalars: global_scalars(game, viewer)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Config;

    #[test]
    fn encoder_layout_is_the_frozen_v2_shape() {
        assert_eq!(SHEET_PLANES_LEN, 1728);
        assert_eq!(SHEET_SCALARS_LEN, 180);
        assert_eq!(VIEWER_PLANE_LEN, 36);
        assert_eq!(NUM_GLOBAL_SCALAR, 358);
        assert_eq!(NUM_DEALT_PLANS, 28);
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
}
