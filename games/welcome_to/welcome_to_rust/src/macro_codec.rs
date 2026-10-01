//! The macro action vocabulary — a mirror of `games/welcome_to/macro_codec.py`
//! (RUST_PORT_PLAN.md M2). 684 indices, frozen in `ENCODER_V2_SPEC.md` §10.6.
//!
//! The whole `CHOOSE_CARDS -> WRITE_NUMBER` segment is one action. A correct
//! deterministic tree could model the split without averaging the placement
//! continuations; both nodes belong to the same player. The macro is still the
//! useful finite-budget representation because you pick a combination *for* a
//! placement: it matches the semantic action, shortens the horizon, improves
//! credit assignment, and removes the `WRITE_NUMBER` network evaluation
//! (`SEARCH_SPEC.md` §3 measured 28% fewer evaluations).
//!
//! ⚠ **Legality is enumerated, never intersected.** A macro index is legal iff
//! its full primitive sequence is legal end to end, so `legal_macros` steps into
//! each playable slot and reads *that child's* own `legal_actions()`.
//! Reconstructing it as a mask intersection ("slot s is legal" AND "writing 7 in
//! box 4 is legal") admits jointly-illegal pairs, because `WRITE` legality
//! depends on which slot was taken.
//!
//! ⚠ **`collapse` is deliberately absent.** Reading a primitive trajectory as
//! macro labels belongs to `datagen`, which stays in Python (§3). Porting it
//! would put a second implementation behind the training corpus for no gain.

use std::sync::OnceLock;

use crate::codec;
use crate::constants::{box_coords, box_index, NUM_BOXES, TEMP_DELTAS};
use crate::game::{EngineError, EngineResult, Game, Phase};

/// Choice slots a macro can name. Standard mode has exactly three.
pub const NUM_MACRO_SLOTS: usize = 3;
pub const NUM_TEMP_DELTAS: usize = TEMP_DELTAS.len();

pub const M_WRITE: usize = 0;
pub const M_REFUSE: usize = 495;
pub const M_DIRECT_REFUSE: usize = 498;
pub const M_ROUNDABOUT_OPEN: usize = 499;
pub const M_PRIMITIVE: usize = 500;
pub const NUM_MACRO_ACTIONS: usize = 684;

/// The primitive actions the macro layer *subsumes*: everything else keeps its
/// own index and its own phase.
fn is_subsumed(action: usize) -> bool {
    (codec::A_CHOOSE_STACK..codec::A_CHOOSE_STACK + 6).contains(&action)
        || (codec::A_WRITE..codec::A_WRITE + NUM_TEMP_DELTAS * NUM_BOXES).contains(&action)
        || action == codec::A_PERMIT_REFUSAL
        || action == codec::A_ROUNDABOUT_OPEN
}

/// Primitive codec indices that survive into the macro space, **in codec
/// order** — the order is the mapping, so it is computed the same way as the
/// Python rather than written out.
pub fn primitive_actions() -> &'static Vec<usize> {
    static ACTIONS: OnceLock<Vec<usize>> = OnceLock::new();
    ACTIONS.get_or_init(|| {
        (0..codec::NUM_ACTIONS)
            .filter(|&a| !is_subsumed(a))
            .collect()
    })
}

fn primitive_to_macro_table() -> &'static Vec<Option<usize>> {
    static TABLE: OnceLock<Vec<Option<usize>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = vec![None; codec::NUM_ACTIONS];
        for (i, &action) in primitive_actions().iter().enumerate() {
            table[action] = Some(M_PRIMITIVE + i);
        }
        table
    })
}

// ──────────────────────────────────────────────────────────────────────────
// Index arithmetic
// ──────────────────────────────────────────────────────────────────────────
/// `(slot, temp delta, box)` — take a combination and place it.
pub fn macro_write(slot: usize, delta_slot: usize, x: usize, y: usize) -> usize {
    assert!(
        slot < NUM_MACRO_SLOTS,
        "slot {slot} is outside standard mode's three stacks"
    );
    M_WRITE + (slot * NUM_TEMP_DELTAS + delta_slot) * NUM_BOXES + box_index(x, y)
}

/// `(slot, delta_slot, x, y)`.
pub fn decode_macro_write(index: usize) -> (usize, usize, usize, usize) {
    let offset = index - M_WRITE;
    let (x, y) = box_coords(offset % NUM_BOXES);
    let rest = offset / NUM_BOXES;
    (rest / NUM_TEMP_DELTAS, rest % NUM_TEMP_DELTAS, x, y)
}

/// Take `slot`, whose printed number has nowhere to go, and refuse.
///
/// ⚠ A *different action* from the direct refusal: this one is legal even when
/// the slot is playable, because the temp agency could have widened it and
/// `argWriteNumber` refuses to force a player to spend the agency merely to
/// have somewhere to write. Which slot you burn is a real decision — it is the
/// effect you forgo.
pub fn macro_refuse(slot: usize) -> usize {
    assert!(
        slot < NUM_MACRO_SLOTS,
        "slot {slot} is outside standard mode's three stacks"
    );
    M_REFUSE + slot
}

/// The macro index of a primitive the macro layer does **not** subsume.
///
/// `None` for a subsumed one rather than something plausible: a bare `WRITE`
/// has no macro meaning without the slot that preceded it, and quietly
/// inventing one would put two different decisions on one logit.
pub fn from_primitive(action: usize) -> Option<usize> {
    primitive_to_macro_table().get(action).copied().flatten()
}

/// The primitive behind a macro index, for the 184 that have exactly one.
pub fn to_primitive(index: usize) -> Option<usize> {
    if index < M_PRIMITIVE || index >= NUM_MACRO_ACTIONS {
        return None;
    }
    primitive_actions().get(index - M_PRIMITIVE).copied()
}

/// The primitive sequence a macro index stands for.
pub fn primitives_for(index: usize) -> EngineResult<Vec<usize>> {
    if index >= NUM_MACRO_ACTIONS {
        return Err(EngineError::Illegal(format!(
            "macro {index} is outside the {NUM_MACRO_ACTIONS}-index vocabulary"
        )));
    }
    if index < M_REFUSE {
        let (slot, delta_slot, x, y) = decode_macro_write(index);
        return Ok(vec![
            codec::choose_stack(slot),
            codec::write(delta_slot, x, y),
        ]);
    }
    if index < M_DIRECT_REFUSE {
        return Ok(vec![
            codec::choose_stack(index - M_REFUSE),
            codec::A_PERMIT_REFUSAL,
        ]);
    }
    if index < M_ROUNDABOUT_OPEN {
        return Ok(vec![codec::A_PERMIT_REFUSAL]);
    }
    if index < M_PRIMITIVE {
        return Ok(vec![codec::A_ROUNDABOUT_OPEN]);
    }
    Ok(vec![to_primitive(index).expect("checked in range")])
}

// ──────────────────────────────────────────────────────────────────────────
// Legality
// ──────────────────────────────────────────────────────────────────────────
/// Whether this is a state the macro layer makes a decision at — everything
/// except `WRITE_NUMBER`, which the macro layer swallows.
pub fn is_macro_root(state: &Game) -> bool {
    state.phase != Phase::WriteNumber
}

fn require_standard(state: &Game) -> EngineResult<()> {
    if !state.config.standard() {
        return Err(EngineError::Illegal(
            "the macro vocabulary covers standard mode only; expert and solo \
             have six ordered pairs and no macro representation"
                .into(),
        ));
    }
    Ok(())
}

/// Every macro index whose **whole primitive sequence** is legal here, in the
/// order `macro_codec.py` produces them.
pub fn legal_macros(state: &Game) -> EngineResult<Vec<usize>> {
    if state.phase == Phase::GameOver {
        return Ok(Vec::new());
    }
    if state.phase == Phase::WriteNumber {
        return Err(EngineError::Illegal(
            "WRITE_NUMBER is inside a macro; the macro layer never decides here".into(),
        ));
    }
    if state.phase != Phase::ChooseCards {
        return state
            .legal_actions()
            .into_iter()
            .map(|a| {
                from_primitive(a).ok_or_else(|| {
                    EngineError::Illegal(format!(
                        "primitive {a} is subsumed by the macro layer and has no \
                         standalone index"
                    ))
                })
            })
            .collect();
    }

    require_standard(state)?;
    let mut out: Vec<usize> = Vec::new();
    for action in state.legal_actions() {
        if action == codec::A_PERMIT_REFUSAL {
            out.push(M_DIRECT_REFUSE);
            continue;
        }
        if action == codec::A_ROUNDABOUT_OPEN {
            out.push(M_ROUNDABOUT_OPEN);
            continue;
        }
        let slot = codec::decode_stack(action);
        // Step into the slot and read the child's own legality — see the module
        // docstring on why this is not a mask intersection.
        let mut child = state.clone();
        child.apply(action)?;
        for follow in child.legal_actions() {
            if follow == codec::A_PERMIT_REFUSAL {
                out.push(macro_refuse(slot));
            } else {
                let (delta_slot, x, y) = codec::decode_write(follow);
                out.push(macro_write(slot, delta_slot, x, y));
            }
        }
    }
    Ok(out)
}

/// Passes the search never spends budget on, by the phase they are offered at.
///
/// Each is **provably** dominated, not merely unpromising: park, pool and estate
/// advance a strictly-increasing scoring track and consume no box, fence,
/// number, turn or resource, and plans are not auto-validated so advancing a
/// track cannot force an unwanted three-plan game end. Opening a roundabout and
/// passing reaches the same `CHOOSE_CARDS` state as never opening, minus the
/// option.
///
/// ⚠ `PASS_BIS` and `PASS_SURVEYOR` are deliberately absent — a bis fills a box
/// and takes a penalty, and a fence can destroy an `EstatePlan`'s required
/// sizes. Both are genuine decisions.
fn dominated_pass(phase: Phase) -> Option<usize> {
    let primitive = match phase {
        Phase::ActionPark => codec::A_PASS_PARK,
        Phase::ActionPool => codec::A_PASS_POOL,
        Phase::ActionEstate => codec::A_PASS_ESTATE,
        Phase::RoundaboutPlace => codec::A_PASS_ROUNDABOUT,
        _ => return None,
    };
    from_primitive(primitive)
}

/// `legal_macros` minus the dominated passes — **for the search only**.
///
/// ⚠ This is a search mask, not a rules change, and it must never move into
/// `legal_macros`: `datagen.replay` builds its training mask from the full
/// vocabulary, and the reference policy takes these passes 1,853 times in the
/// recorded corpus. A pass is dropped only when an alternative exists, which
/// `len(macros) > 1` *is*, the pass being a single index.
pub fn search_legal_macros(state: &Game, prune_roundabout_pass: bool) -> EngineResult<Vec<usize>> {
    Ok(prune_search_macros(state, legal_macros(state)?, prune_roundabout_pass))
}

/// The pruning half of [`search_legal_macros`], applied to a list the caller
/// already has.
///
/// Split out because the evaluator row carries the **full** vocabulary while the
/// tree searches the pruned one, so a leaf otherwise pays for `legal_macros`
/// twice — once to pack the request, once to expand the node the response
/// creates — and at `CHOOSE_CARDS` that enumeration clones the whole `Game`
/// once per playable slot. The output is identical to recomputing: this is the
/// same filter over the same list, in the same order.
pub fn prune_search_macros(
    state: &Game,
    macros: Vec<usize>,
    prune_roundabout_pass: bool,
) -> Vec<usize> {
    if !prune_roundabout_pass && state.phase == Phase::RoundaboutPlace {
        return macros;
    }
    let Some(pruned) = dominated_pass(state.phase) else {
        return macros;
    };
    if macros.len() < 2 {
        return macros;
    }
    macros.into_iter().filter(|&m| m != pruned).collect()
}

// ──────────────────────────────────────────────────────────────────────────
// Applying
// ──────────────────────────────────────────────────────────────────────────
/// Apply the whole sequence in place. Raises if any step is illegal.
pub fn apply_macro(state: &mut Game, index: usize) -> EngineResult<()> {
    for action in primitives_for(index)? {
        state.apply(action)?;
    }
    Ok(())
}

/// Apply the whole sequence to a copy and return it.
pub fn step_macro(state: &Game, index: usize) -> EngineResult<Game> {
    let mut next = state.clone();
    apply_macro(&mut next, index)?;
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Config;
    use crate::rng::Rng;

    fn two_player() -> Config {
        Config {
            players: 2,
            advanced: true,
            expert: false,
            solo_rules: false,
        }
    }

    #[test]
    fn the_layout_is_the_frozen_684() {
        assert_eq!(
            M_WRITE + NUM_MACRO_SLOTS * NUM_TEMP_DELTAS * NUM_BOXES,
            M_REFUSE
        );
        assert_eq!(M_REFUSE + NUM_MACRO_SLOTS, M_DIRECT_REFUSE);
        assert_eq!(M_PRIMITIVE + primitive_actions().len(), NUM_MACRO_ACTIONS);
        assert_eq!(primitive_actions().len(), 184);
    }

    #[test]
    fn write_indices_round_trip() {
        for slot in 0..NUM_MACRO_SLOTS {
            for delta in 0..NUM_TEMP_DELTAS {
                for box_i in 0..NUM_BOXES {
                    let (x, y) = box_coords(box_i);
                    let index = macro_write(slot, delta, x, y);
                    assert_eq!(decode_macro_write(index), (slot, delta, x, y));
                }
            }
        }
    }

    #[test]
    fn a_subsumed_primitive_has_no_standalone_index() {
        assert_eq!(from_primitive(codec::A_PERMIT_REFUSAL), None);
        assert_eq!(from_primitive(codec::A_WRITE), None);
        assert_eq!(from_primitive(codec::A_CHOOSE_STACK), None);
        assert!(from_primitive(codec::A_PASS_PLAN).is_some());
    }

    #[test]
    fn every_macro_a_root_offers_applies() {
        let mut rng = Rng::new(5);
        let mut state = Game::new(21, two_player()).expect("setup");
        while !state.is_terminal() {
            let macros = legal_macros(&state).expect("a macro root");
            assert!(!macros.is_empty());
            for &index in macros.iter() {
                step_macro(&state, index).expect("a legal macro applies");
            }
            let choice = macros[rng.randrange(macros.len() as u64) as usize];
            apply_macro(&mut state, choice).expect("legal");
        }
    }
}
