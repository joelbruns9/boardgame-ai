//! W3 positional-control key, emitted for the Python side to look up.
//!
//! Rust owns the encoding and the game state on the self-play path and in the
//! advisor's searcher, so it is the only place that can derive a control key --
//! and, now that control is an encoder INPUT, the only place that can read the
//! table for a search leaf.
//!
//! Rust does not load the table from disk. Python hands over the exact bytes it
//! is itself using (`control_table.table_blob`), so "both readers agree" is true
//! by construction rather than by a test. Keys are still emitted for the replay
//! path, where Python does the lookup.
//!
//! The packing must stay byte-identical to `control_table.pack_key`. One u64 per
//! example row; **zero means "no key"**, which Python treats as masked, never as
//! a default. A zero-filled control label would read as "the opponent gets
//! everything", which is a confident lie rather than a missing feature.
//!
//! Applicability follows `control_table.control_key`: outside `PlayAge`, or with
//! a pending choice, the current decision-maker is not the next tableau mover
//! (the engine defers `pending_extra_turn`), and resolving a progress token can
//! restate the tempo half of the key. A closed Wonder pool has no entry.

use crate::data::{progress_id, EffectKind, WONDERS};
use crate::state::{GameState, Phase};

const PACK_VALID: u64 = 1 << 63;

/// Pack `(age, mask, who_moves, tempo)`; mirrors `control_table.pack_key`.
fn pack(age: u8, mask: u32, who_moves: bool, tempo: [u8; 5]) -> u64 {
    PACK_VALID
        | (age as u64 & 0x3)
        | ((mask as u64 & 0xF_FFFF) << 2)
        | ((who_moves as u64) << 22)
        | ((tempo[0] as u64 & 0x7) << 23)
        | ((tempo[1] as u64 & 0x7) << 26)
        | ((tempo[2] as u64 & 0x7) << 29)
        | ((tempo[3] as u64 & 0x7) << 32)
        | ((tempo[4] as u64 & 0x7) << 35)
}

/// The control key for `state`, from the acting seat's point of view, or 0.
pub fn control_key_word(state: &GameState) -> u64 {
    if state.phase != Phase::PlayAge || state.pending_choice.is_some() {
        return 0;
    }
    let age = state.tableau.age;
    if !(1..=3).contains(&age) {
        return 0;
    }

    // Slot order is `layout(age)`, which is sorted by (row, x) and generated
    // from the same source as Python's `Layout.slots` -- so bit i means the same
    // slot on both sides. That equality is what a key-parity gate must check.
    let mut mask: u32 = 0;
    for (index, card) in state.tableau.slots.iter().enumerate() {
        if card.present {
            mask |= 1 << index;
        }
    }
    if mask == 0 {
        return 0;
    }

    let theology = progress_id("Theology");
    let attacker = state.active_player;
    let built_total: usize = state.cities.iter().map(|c| c.built_wonders.len()).sum();
    let mut counts = [0u8; 4];
    for (slot, player) in [(0usize, attacker), (1usize, 1 - attacker)] {
        let city = &state.cities[player];
        // Under Theology EVERY Wonder grants the extra turn, which is the whole
        // point of the token and what turned three ordinary Wonders into three
        // extra turns in the reference game.
        let all_grant = city.progress_tokens.contains(&theology);
        let (mut ordinary, mut extra) = (0u8, 0u8);
        for &wonder in &city.wonders {
            if city.built_wonders.contains(&wonder) || state.retired_wonders.contains(&wonder)
            {
                continue;
            }
            let grants = all_grant
                || WONDERS[wonder]
                    .effects
                    .iter()
                    .any(|effect| matches!(effect.kind, EffectKind::PlayAgain));
            if grants {
                extra += 1;
            } else {
                ordinary += 1;
            }
        }
        counts[slot * 2] = ordinary;
        counts[slot * 2 + 1] = extra;
    }

    // Seven Wonders are built across both cities and the eighth is retired, so
    // the pool is `7 - built`, and never more than what is actually unbuilt.
    let unbuilt: u8 = counts.iter().sum();
    let builds_left = (7usize.saturating_sub(built_total)).min(unbuilt as usize) as u8;
    if builds_left == 0 {
        return 0;
    }

    pack(
        age,
        mask,
        true, // a clean PlayAge row is encoded for the player on move
        [builds_left, counts[0], counts[1], counts[2], counts[3]],
    )
}


// ---------------------------------------------------------------------------
// The shipped table
// ---------------------------------------------------------------------------

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

pub const UNREACH: u8 = 254;
pub const ABSENT: u8 = 255;

/// Per-slot control, indexed exactly as Python indexes it.
pub struct ControlTable {
    pub digest: [u8; 32],
    tempo_index: HashMap<[u8; 5], usize>,
    n_tempo: usize,
    ages: HashMap<u8, AgePlane>,
}

struct AgePlane {
    width: usize,
    mask_index: HashMap<u32, usize>,
    cells: Vec<u8>, // [mask][who][tempo][slot]
}

static TABLE: OnceLock<ControlTable> = OnceLock::new();

/// Input-off mode. The channels stay in the schema and are emitted as zeros,
/// so both arms share a width, a signature and an architecture, and the only
/// difference is what the network is shown. Must track Python exactly: the two
/// languages disagreeing about what the model sees is worse than either arm.
static ENABLED: AtomicBool = AtomicBool::new(true);

pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Install the table Python is itself using.
///
/// Idempotent for the SAME table and an error for a different one. Silently
/// keeping the first install was worse than either: a partial table could be
/// installed, the correct one could then report success, and the process would
/// go on encoding against the partial one.
pub fn install(blob: &[u8]) -> Result<(), String> {
    let parsed = parse(blob)?;
    let digest = parsed.digest;
    if let Err(_rejected) = TABLE.set(parsed) {
        let installed = TABLE.get().expect("set failed, so a table is present");
        if installed.digest != digest {
            return Err(format!(
                "a different control table is already installed ({} != {});                  the process would keep encoding against the first one",
                hex(&installed.digest),
                hex(&digest)
            ));
        }
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Hex digest of the installed table, for Python to check against its manifest.
pub fn installed_digest() -> Option<String> {
    table().map(|t| hex(&t.digest))
}

pub fn table() -> Option<&'static ControlTable> {
    TABLE.get()
}

fn take<'a>(blob: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8], String> {
    let end = *at + n;
    if end > blob.len() {
        return Err(format!("control table truncated at {at} (+{n} of {})", blob.len()));
    }
    let slice = &blob[*at..end];
    *at = end;
    Ok(slice)
}

fn parse(blob: &[u8]) -> Result<ControlTable, String> {
    let mut at = 0usize;
    if take(blob, &mut at, 7)? != b"SWDCTL1" {
        return Err("control table blob has the wrong magic".to_string());
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(take(blob, &mut at, 32)?);

    let n_tempo = u32::from_le_bytes(take(blob, &mut at, 4)?.try_into().unwrap()) as usize;
    let mut tempo_index = HashMap::with_capacity(n_tempo);
    for i in 0..n_tempo {
        let raw = take(blob, &mut at, 5)?;
        let mut key = [0u8; 5];
        key.copy_from_slice(raw);
        tempo_index.insert(key, i);
    }

    let n_ages = u32::from_le_bytes(take(blob, &mut at, 4)?.try_into().unwrap()) as usize;
    let mut ages = HashMap::with_capacity(n_ages);
    let mut order: Vec<u8> = Vec::with_capacity(n_ages);
    for _ in 0..n_ages {
        let header = take(blob, &mut at, 6)?;
        let age = header[0];
        let width = header[1] as usize;
        let n_masks = u32::from_le_bytes(header[2..6].try_into().unwrap()) as usize;
        let mut mask_index = HashMap::with_capacity(n_masks);
        for i in 0..n_masks {
            let mask = u32::from_le_bytes(take(blob, &mut at, 4)?.try_into().unwrap());
            mask_index.insert(mask, i);
        }
        let cells = take(blob, &mut at, n_masks * 2 * n_tempo * width)?.to_vec();
        order.push(age);
        ages.insert(age, AgePlane { width, mask_index, cells });
    }
    if at != blob.len() {
        return Err(format!("control table blob has {} trailing bytes", blob.len() - at));
    }

    // Verify rather than trust. The digest is the artifact's identity, and a
    // copied-but-unchecked digest lets a truncated or wrong table install itself
    // under a name that says it is correct.
    let mut hasher = Sha256::new();
    for age in &order {
        hasher.update(&ages[age].cells);
    }
    let computed: [u8; 32] = hasher.finalize().into();
    if computed != digest {
        return Err(format!(
            "control table digest mismatch: blob declares {} but its contents hash to {}",
            hex(&digest),
            hex(&computed)
        ));
    }
    Ok(ControlTable { digest, tempo_index, n_tempo, ages })
}

impl ControlTable {
    /// Is this a tempo a legal game can reach? Decides APPLICABILITY, and is
    /// deliberately separate from whether the table covers a given age/mask.
    pub fn knows_tempo(&self, tempo: [u8; 5]) -> bool {
        self.tempo_index.contains_key(&tempo)
    }

    /// The 20 per-slot cells for one key, or None when the table does not cover
    /// it. None is a caller-visible miss, never zeros.
    pub fn lookup(&self, age: u8, mask: u32, who_moves: bool, tempo: [u8; 5]) -> Option<&[u8]> {
        let plane = self.ages.get(&age)?;
        let mask_i = *plane.mask_index.get(&mask)?;
        let tempo_i = *self.tempo_index.get(&tempo)?;
        let who = usize::from(!who_moves);
        let start = ((mask_i * 2 + who) * self.n_tempo + tempo_i) * plane.width;
        Some(&plane.cells[start..start + plane.width])
    }
}

/// Unpack a key word back into its fields; mirrors `control_table.unpack_key`.
pub fn unpack(word: u64) -> Option<(u8, u32, bool, [u8; 5])> {
    if word == 0 {
        return None;
    }
    Some((
        (word & 0x3) as u8,
        ((word >> 2) & 0xF_FFFF) as u32,
        (word >> 22) & 0x1 == 1,
        [
            ((word >> 23) & 0x7) as u8,
            ((word >> 26) & 0x7) as u8,
            ((word >> 29) & 0x7) as u8,
            ((word >> 32) & 0x7) as u8,
            ((word >> 35) & 0x7) as u8,
        ],
    ))
}

/// `_as_theology` for one player: fold ordinary unbuilt Wonders into extra-turn
/// ones. Mirrors `tableau_control._as_theology`.
pub fn as_theology(tempo: [u8; 5], player: usize) -> [u8; 5] {
    let mut out = tempo;
    let (ord, ext) = (1 + 2 * player, 2 + 2 * player);
    out[ext] += out[ord];
    out[ord] = 0;
    out
}


/// The three control maps the encoder reads, or None where the position has no
/// control answer.
///
/// Panics if a key exists but no table was installed. That is deliberate: the
/// alternative is emitting zeros, which the network reads as "the opponent
/// reaches every slot first" -- a confident lie, and indistinguishable from the
/// genuine all-unreachable position. A missing table is a deployment error and
/// must stop the run at the first encode, not degrade it silently.
pub fn control_maps(state: &GameState) -> Option<[&'static [u8]; 3]> {
    if !enabled() {
        return None;
    }
    let (age, mask, who, tempo) = unpack(control_key_word(state))?;
    let table = table().expect(
        "control table not installed: call seven_wonders_rust.set_control_table()          with control_table.table_blob() before encoding",
    );
    // Applicability is decided INDEPENDENTLY of coverage. A tempo outside the
    // enumeration is a position no legal game reaches -- eight Wonders are
    // drafted, so nine unbuilt cannot happen -- and is masked. Anything else
    // missing is a table gap, and masking it would silently disagree with
    // Python, which raises: an incomplete table would read as "no control
    // answer" on legal positions in one language only.
    if !table.knows_tempo(tempo) {
        return None;
    }
    let lookup = |t: [u8; 5]| {
        table.lookup(age, mask, who, t).unwrap_or_else(|| {
            panic!(
                "control table has no entry for the legal key age {age}                  mask {mask:#x} tempo {t:?}; the installed table is incomplete"
            )
        })
    };
    Some([
        lookup(tempo),
        lookup(as_theology(tempo, 0)),
        lookup(as_theology(tempo, 1)),
    ])
}
