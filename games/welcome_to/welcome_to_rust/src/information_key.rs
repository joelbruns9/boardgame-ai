//! Viewer information-state identity (`mcts.information_key`, M4).
//!
//! This is intentionally not a full game snapshot.  Future deck order,
//! opponents' live current-turn sheets, other seats' votes and context, and the
//! private table-wide reshuffle aggregate must not enter it.  The key is a
//! versioned, unambiguous byte sequence so M5 can use it directly in Rust hash
//! maps without constructing a Python tuple.

use crate::constants::{card_effect, card_number, Effect, EMPTY, FENCE_SIZES, STREET_SIZES};
use crate::encoder::{deck_composition, discard_composition, Matrix};
use crate::game::{EngineError, EngineResult, Game, TurnCtx, NO_CARD};
use crate::sheet::Sheet;

pub const INFORMATION_KEY_ABI_VERSION: u16 = 1;

/// Canonical bytes for one observation. Equality is the contract; the byte
/// layout is private except for its version and little-endian integer encoding.
pub type InformationKey = Vec<u8>;

struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        let mut out = Self {
            bytes: Vec::with_capacity(1024),
        };
        out.u16(INFORMATION_KEY_ABI_VERSION);
        out
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn bool(&mut self, value: bool) {
        self.u8(value as u8);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// Canonical unsigned LEB128. Most game values occupy one byte, while the
    /// encoding remains injective over the full u32 range. A fixed-width first
    /// draft made one 4-seat key 1.9 KiB, which is unacceptable once every
    /// retained chance child owns one in M5.
    fn u32(&mut self, value: usize) {
        let mut value = u32::try_from(value).expect("key value fits u32");
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            self.u8(byte | if value == 0 { 0 } else { 0x80 });
            if value == 0 {
                break;
            }
        }
    }

    fn i32(&mut self, value: i32) {
        let zigzag = ((value as u32) << 1) ^ ((value >> 31) as u32);
        self.u32(zigzag as usize);
    }

    fn option_i32(&mut self, value: Option<i32>) {
        self.bool(value.is_some());
        if let Some(value) = value {
            self.i32(value);
        }
    }

    fn option_usize(&mut self, value: Option<usize>) {
        self.bool(value.is_some());
        if let Some(value) = value {
            self.u32(value);
        }
    }

    fn option_effect(&mut self, value: Option<Effect>) {
        self.bool(value.is_some());
        if let Some(value) = value {
            self.u8(value as u8);
        }
    }

    fn option_pos(&mut self, value: Option<(usize, usize)>) {
        self.bool(value.is_some());
        if let Some((x, y)) = value {
            self.u32(x);
            self.u32(y);
        }
    }
}

fn write_sheet(w: &mut Writer, sheet: &Sheet) {
    // Exactly mcts._SHEET_FIELDS, in exactly that order. Rust's fixed-array
    // padding is representation, not part of Python's ragged Sheet value.
    for x in 0..STREET_SIZES.len() {
        for y in 0..STREET_SIZES[x] {
            w.option_i32((sheet.numbers[x][y] != EMPTY).then_some(sheet.numbers[x][y]));
        }
    }
    for x in 0..STREET_SIZES.len() {
        for y in 0..STREET_SIZES[x] {
            w.bool(sheet.is_bis[x][y]);
        }
    }
    for x in 0..STREET_SIZES.len() {
        for y in 0..STREET_SIZES[x] {
            w.i32(sheet.written_turn[x][y]);
        }
    }
    for x in 0..FENCE_SIZES.len() {
        for y in 0..FENCE_SIZES[x] {
            w.bool(sheet.fences[x][y]);
        }
    }
    for x in 0..STREET_SIZES.len() {
        for y in 0..STREET_SIZES[x] {
            w.bool(sheet.top_fences[x][y]);
        }
    }
    for value in sheet.parks {
        w.i32(value);
    }
    for value in sheet.pools {
        w.i32(value);
    }
    for value in sheet.estate_marks {
        w.i32(value);
    }
    w.i32(sheet.temps);
    w.i32(sheet.bis_marks);
    w.i32(sheet.permits);
    w.i32(sheet.roundabouts);
}

fn write_printed_card(w: &mut Writer, card: i32) {
    w.bool(card != NO_CARD);
    if card != NO_CARD {
        w.i32(card_number(card as usize));
        w.u8(card_effect(card as usize) as u8);
    }
}

fn write_composition(w: &mut Writer, counts: &Matrix) {
    for row in counts {
        for &count in row {
            debug_assert!(count >= 0.0 && count.fract() == 0.0);
            w.i32(count as i32);
        }
    }
}

fn write_ctx(w: &mut Writer, ctx: &TurnCtx) {
    w.option_usize(ctx.slot);
    w.option_i32(ctx.number);
    w.option_effect(ctx.effect);
    w.option_pos(ctx.last_house);
    w.bool(ctx.built_roundabout);
    w.bool(ctx.roundabout_declined);
    w.bool(ctx.refused);
    w.option_usize(ctx.plan_slot);
    w.u32(ctx.pending_sizes.len());
    for &size in &ctx.pending_sizes {
        w.u32(size);
    }
    w.u32(ctx.chosen_estates.len());
    for &(x, start, size) in &ctx.chosen_estates {
        w.u32(x);
        w.u32(start);
        w.u32(size);
    }
}

pub fn information_key(game: &Game, viewer: usize) -> EngineResult<InformationKey> {
    if viewer >= game.config.players {
        return Err(EngineError::Invalid(format!(
            "information-key viewer {viewer} is outside {} seats",
            game.config.players
        )));
    }

    let mut w = Writer::new();
    w.u32(viewer);
    w.i32(game.turn);
    w.u32(game.actor);
    w.u8(game.phase as u8);

    // The whole frozen GameConfig, not merely the seat count.
    w.u32(game.config.players);
    w.bool(game.config.advanced);
    w.bool(game.config.expert);
    w.bool(game.config.solo_rules);

    w.u32(game.discard.len());
    let table = game.table_cards(viewer);
    w.u32(table.len());
    for card in table {
        write_printed_card(&mut w, card);
    }

    w.u32(game.config.players);
    for player in 0..game.config.players {
        write_sheet(&mut w, game.sheet_for(viewer, player));
    }

    for plan_id in game.plan_ids {
        w.u32(plan_id);
    }
    for slot in 0..3 {
        let mut turns = game.plan_turns_for(viewer, slot);
        turns.sort_unstable();
        w.u32(turns.len());
        for (player, turn) in turns {
            w.i32(player);
            w.i32(turn);
        }
    }

    w.u32(game.deck_remaining());
    write_composition(&mut w, &deck_composition(game, viewer));
    write_composition(&mut w, &discard_composition(game));
    w.bool(game.solo_card_drawn);
    w.bool(game.reshuffle_vote_for(viewer));
    w.option_i32((game.turn_choice[viewer] != NO_CARD).then_some(game.turn_choice[viewer]));

    let own_turn = game.actor == viewer;
    w.bool(own_turn);
    if own_turn {
        write_ctx(&mut w, &game.ctx);
    }
    Ok(w.bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Config;
    use crate::rng::Rng;

    #[test]
    fn hidden_deck_order_does_not_move_the_key() {
        let game = Game::new(
            7,
            Config {
                players: 3,
                advanced: true,
                expert: false,
                solo_rules: false,
            },
        )
        .expect("game");
        let before = information_key(&game, 0).expect("key");
        let shuffled = game.redeterminize(&mut Rng::new(99));
        assert_eq!(information_key(&shuffled, 0).expect("key"), before);
    }

    #[test]
    fn viewer_is_part_of_the_key() {
        let game = Game::new(
            7,
            Config {
                players: 2,
                advanced: false,
                expert: false,
                solo_rules: false,
            },
        )
        .expect("game");
        assert_ne!(
            information_key(&game, 0).expect("key"),
            information_key(&game, 1).expect("key")
        );
    }

    #[test]
    fn varints_are_injective_at_every_width_boundary() {
        let unsigned = [
            0usize,
            1,
            127,
            128,
            255,
            256,
            16_383,
            16_384,
            u32::MAX as usize,
        ];
        let signed = [i32::MIN, -16_384, -129, -128, -1, 0, 1, 127, 128, i32::MAX];
        let unsigned_encodings: Vec<Vec<u8>> = unsigned
            .into_iter()
            .map(|value| {
                let mut w = Writer::new();
                w.u32(value);
                w.bytes
            })
            .collect();
        let signed_encodings: Vec<Vec<u8>> = signed
            .into_iter()
            .map(|value| {
                let mut w = Writer::new();
                w.i32(value);
                w.bytes
            })
            .collect();
        assert_eq!(
            unsigned_encodings
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            unsigned_encodings.len()
        );
        assert_eq!(
            signed_encodings
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            signed_encodings.len()
        );
    }
}
