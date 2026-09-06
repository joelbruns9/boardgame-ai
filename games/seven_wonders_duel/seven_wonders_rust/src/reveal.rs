//! Per-slot reveal risk (port of `reveal_risk.py`).
//!
//! W3's control channels are chance-invariant by construction: `control_key_word`
//! is keyed on `(age, present-mask, who-moves, tempo)` with no card identities,
//! so it returns the same answer in all ten reveal-worlds of a position whose
//! value ranges 0.02% to 54%. These five channels are the part it cannot see --
//! what a removal *uncovers*.
//!
//! `reveal_n` counts the hidden slots a removal exposes; the four `*_sixth` /
//! `*_mil` channels multiply that by the fraction of the unseen pool that would
//! hand a seat a sixth science symbol or immediate military supremacy. They are
//! expected counts of decisive cards revealed, not probabilities: `reveal_n = 2`
//! against a 40%-decisive pool reads 0.8.
//!
//! No search and no joint enumeration: one pass over the unseen pool per seat
//! plus local geometry, which is the price class of the features already here.
//! Python's module docstring carries the full statement of what this is NOT --
//! a marginal risk per revealed slot, blind to affordability and turn order.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use crate::data::{back_type_of, card};
use crate::state::GameState;

/// Back count, and the fixed order the per-back sums below accumulate in --
/// `BackType as usize`, shared with Python's `_BACKS`, so the two languages add
/// the same floats in the same order.
const NUM_BACKS: usize = 4;

/// Channels per tableau token, in `REVEAL_FEATURES` order.
pub const WIDTH: usize = 5;

/// Input-off mode, exactly as control does it: the channels stay in the schema
/// and are emitted as zeros, so both arms share a width, a signature and an
/// architecture, and differ only in what the network is shown.
///
/// The default is read from `SWD_REVEAL_FEATURES` with Python's own parsing, so
/// the two languages agree without anybody having to remember to call the
/// setter. Hard-coding a default here instead is how control acquired a latent
/// disagreement: `encoder.py` reads `SWD_CONTROL_FEATURES` and Rust assumed on,
/// so setting the variable to 0 turned the channels off in one language only.
static ENABLED: OnceLock<AtomicBool> = OnceLock::new();

fn flag() -> &'static AtomicBool {
    ENABLED.get_or_init(|| AtomicBool::new(env_default("SWD_REVEAL_FEATURES", false)))
}

/// `not in ("0", "false", "no", "off")` after strip+lower -- Python's test.
pub(crate) fn env_default(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Err(_) => default,
        Ok(raw) => !matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
    }
}

pub fn set_enabled(enabled: bool) {
    flag().store(enabled, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    flag().load(Ordering::Relaxed)
}

/// `slot index -> how many hidden slots of each BACK removing it would reveal`.
///
/// A slot at row `r` is covered by slots at `r + 1`. Removing a coverer reveals
/// a covered slot only when it was the **last** present coverer -- the same
/// condition `take_accessible` reports as newly accessible.
///
/// The back is carried, not just the count: a card's back is public while its
/// face is not, and it says which pool the card comes from. A Guild-backed slot
/// cannot turn over an Age III card, so pooling the backs -- as this did until
/// 2026-09-07 -- priced a reveal with cards it could never produce.
fn newly_revealed_backs(
    g: &GameState,
    present: &[(i32, i32, usize)],
) -> Vec<[u32; NUM_BACKS]> {
    let mut counts = vec![[0u32; NUM_BACKS]; g.tableau.slots.len()];
    for &(row, x, i) in present {
        if g.tableau.slots[i].revealed {
            continue;
        }
        let mut coverers = present
            .iter()
            .filter(|&&(orow, ox, _)| orow == row + 1 && (ox - x).abs() == 1);
        // Exactly one coverer left: removing it uncovers this hidden card.
        if let (Some(&(_, _, only)), None) = (coverers.next(), coverers.next()) {
            counts[only][back_type_of(g.tableau.slots[i].card_id) as usize] += 1;
        }
    }
    counts
}

/// `(sixth-symbol fraction, immediate-military fraction)` of the unseen pool.
///
/// Mirrors the per-card tests the encoder already applies to face-up cards
/// (`gives_sixth`, `shields >= dist_win`), so a card is judged the same rule
/// whether it is visible or not.
fn decisive_fractions(
    g: &GameState,
    seat: usize,
    have: &[bool],
    names: &[usize],
    dist_win: i32,
    effective_shields: impl Fn(&GameState, usize, usize) -> i32,
) -> (f64, f64) {
    // `names` is ONE back's unseen pool, never the pooled relevant backs: see
    // `newly_revealed_backs`.
    if names.is_empty() {
        return (0.0, 0.0);
    }
    let one_away = have.iter().filter(|&&b| b).count() + 1 >= 6;
    let mut sixth = 0usize;
    let mut military = 0usize;
    for &cid in names {
        let c = card(cid);
        if one_away {
            if let Some(s) = c.science {
                if !have[s as usize] {
                    sixth += 1;
                }
            }
        }
        if effective_shields(g, seat, cid) >= dist_win {
            military += 1;
        }
    }
    let total = names.len() as f64;
    (sixth as f64 / total, military as f64 / total)
}

/// `slot index -> [reveal_n, my_sixth, opp_sixth, my_mil, opp_mil]`.
///
/// All zero when the channels are off, or when nothing hidden is left to
/// uncover -- which is the honest reading and not a missing value: a fully
/// revealed tableau carries no reveal risk.
#[allow(clippy::too_many_arguments)]
pub fn reveal_values(
    g: &GameState,
    present: &[(i32, i32, usize)],
    actor: usize,
    symbols: &[[bool; 7]; 2],
    unseen_by_back: &[Vec<usize>; NUM_BACKS],
    rel_position: impl Fn(&GameState, usize) -> i32,
    effective_shields: impl Fn(&GameState, usize, usize) -> i32 + Copy,
) -> Vec<[f64; WIDTH]> {
    let zeros = vec![[0.0; WIDTH]; g.tableau.slots.len()];
    if !enabled() || present.is_empty() {
        return zeros;
    }
    let counts = newly_revealed_backs(g, present);
    if counts.iter().all(|c| c.iter().all(|&n| n == 0)) {
        return zeros;
    }

    // One pass per (seat, back), not per slot, and only for the backs this
    // position can actually turn over -- in Age III that is two of the four.
    let mut occurs = [false; NUM_BACKS];
    for per_back in &counts {
        for back in 0..NUM_BACKS {
            occurs[back] |= per_back[back] > 0;
        }
    }
    let opponent = 1 - actor;
    let mut fractions = [[(0.0, 0.0); NUM_BACKS]; 2];
    for (index, seat) in [actor, opponent].into_iter().enumerate() {
        let dist_win = 9 - rel_position(g, seat);
        for back in 0..NUM_BACKS {
            if !occurs[back] {
                continue;
            }
            fractions[index][back] = decisive_fractions(
                g,
                seat,
                &symbols[seat],
                &unseen_by_back[back],
                dist_win,
                effective_shields,
            );
        }
    }

    let mut out = zeros;
    for &(_, _, i) in present {
        let per_back = &counts[i];
        let n: u32 = per_back.iter().sum();
        // Summed over backs in a FIXED order, so Python accumulates the same
        // floats in the same order.
        let mut totals = [0.0f64; 4];
        for back in 0..NUM_BACKS {
            let count = per_back[back];
            if count == 0 {
                continue;
            }
            let count = count as f64;
            totals[0] += count * fractions[0][back].0;
            totals[1] += count * fractions[1][back].0;
            totals[2] += count * fractions[0][back].1;
            totals[3] += count * fractions[1][back].1;
        }
        out[i] = [n as f64 / 2.0, totals[0], totals[1], totals[2], totals[3]];
    }
    out
}
