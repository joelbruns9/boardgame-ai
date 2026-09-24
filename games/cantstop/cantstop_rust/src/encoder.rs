//! Value-net features — a mirror of `games/cantstop/encoder.py::encode_board`.
//!
//! Exists because M2 measured the board crossing as the new bottleneck:
//! shipping ~3.5k leaf snapshots to Python and rebuilding `GameState`s cost
//! ~33 ms per solve against ~12 ms of Rust solving. Encoding here hands
//! Python one float32 buffer instead.
//!
//! Bit-exact with Python on f32: every value is computed in f64 and cast to
//! f32 once, as numpy's float64 -> float32 array assignment does.

use crate::engine::{GameState, COLUMN_HEIGHTS, MAX_COL, MAX_PLAYERS, MIN_COL};

pub const NUM_COLUMNS: usize = MAX_COL - MIN_COL + 1;
pub const PER_SEAT: usize = 2 * NUM_COLUMNS + 2;
pub const NUM_GLOBAL: usize = 2;
pub const FEATURE_SIZE: usize = MAX_PLAYERS * PER_SEAT + NUM_GLOBAL;

/// Both scaled features divide by this constant, never by the rule set's
/// own threshold -- see `encoder.py` for why that matters.
pub const MAX_COLUMNS_TO_WIN: f64 = 5.0;

/// Encode one end-of-turn board (no runners) into `out`, which must hold
/// `FEATURE_SIZE` values. Callers guarantee there are no runners: the
/// solver only encodes post-stop / post-bust boards.
pub fn encode_board(state: &GameState, out: &mut [f32]) {
    debug_assert_eq!(out.len(), FEATURE_SIZE);
    debug_assert!(state.runners.iter().all(|&r| r == 0));
    out.fill(0.0);
    let n = state.rules.num_players as usize;
    let need = state.rules.columns_to_win as i32;
    for slot in 0..n {
        let seat = (state.active_player as usize + slot) % n;
        let base = slot * PER_SEAT;
        let mut claimed = 0i32;
        for (i, col) in (MIN_COL..=MAX_COL).enumerate() {
            if state.claimed_by[col] == seat as i8 {
                out[base + NUM_COLUMNS + i] = 1.0;
                claimed += 1;
            }
            let pos = state.progress[seat][col];
            if pos != 0 {
                out[base + i] = (pos as f64 / COLUMN_HEIGHTS[col] as f64) as f32;
            }
        }
        out[base + 2 * NUM_COLUMNS] = ((need - claimed).max(0) as f64 / MAX_COLUMNS_TO_WIN) as f32;
        out[base + 2 * NUM_COLUMNS + 1] = 1.0;
    }
    let g = MAX_PLAYERS * PER_SEAT;
    out[g] = if state.rules.blocking { 1.0 } else { 0.0 };
    out[g + 1] = (need as f64 / MAX_COLUMNS_TO_WIN) as f32;
}
