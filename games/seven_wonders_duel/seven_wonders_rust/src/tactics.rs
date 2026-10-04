//! G4 (`MODEL_GROWTH_PLAN.md`): exact immediate tactics for the searcher.
//!
//! A node whose mover has a GUARANTEED win this turn is proven: its value is
//! exact, so the searcher treats it like a finished game -- no network call, no
//! expansion, and no later averaging of network estimates into it. The
//! predicate is a port of `phase_e.guaranteed_win_now`, which is the Python
//! reference it is gated against (`test_exact_tactics.py`):
//!
//! * a military or scientific win reachable by one action, through the mover's
//!   OWN pending choices (Mausoleum retrieval, a science-pair token, Law) --
//!   the sign does not flip until the opponent moves;
//! * across EVERY consistent chance outcome of that action -- a win in one
//!   world is not a proof at the chance parent;
//! * extra-turn two-move wins and civilian last-card wins are out of scope, as
//!   in the reference.
//!
//! The prefilter (`threat_possible`) is a necessary condition, so it only costs
//! recall, never soundness: every positive is confirmed by applying actions.

use crate::chance::{self, ChanceKind};
use crate::codec::{decode_action, legal_action_indices};
use crate::data::card;
use crate::engine::ActionUse;
use crate::state::{GameState, Phase};
use std::sync::atomic::{AtomicBool, Ordering};

/// Max plausible one-action shield swing (3 shields + Strategy +1, with one
/// square of slack). `phase_e.MILITARY_REACH`.
const MILITARY_REACH: i32 = 5;
/// 4 distinct symbols can reach 6 in one action: a new-symbol green that also
/// completes a pair -> progress choice -> Law. `phase_e.SCIENCE_REACH`.
const SCIENCE_REACH: usize = 4;
/// `phase_e.PENDING_DEPTH`.
pub const PENDING_DEPTH: usize = 6;

static EXACT_TACTICS: AtomicBool = AtomicBool::new(false);

/// Process-wide switch, like the solver's: OFF by default until measured.
pub fn set_enabled(enabled: bool) {
    EXACT_TACTICS.store(enabled, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    EXACT_TACTICS.load(Ordering::Relaxed)
}

/// Layer 1b (proven LOSSES) under its own switch, ON by default but only
/// consulted when `enabled()`: it costs ~2x the win check per node.
static EXACT_LOSSES: AtomicBool = AtomicBool::new(true);

pub fn set_losses_enabled(enabled: bool) {
    EXACT_LOSSES.store(enabled, Ordering::Relaxed);
}

pub fn losses_enabled() -> bool {
    EXACT_LOSSES.load(Ordering::Relaxed)
}

fn actor(state: &GameState) -> usize {
    crate::tree::state_actor(state)
}

fn threat_possible(state: &GameState, player: usize) -> bool {
    let need = if player == 0 {
        9 - state.conflict_position
    } else {
        9 + state.conflict_position
    };
    need <= MILITARY_REACH || state.science_symbols(player).len() >= SCIENCE_REACH
}

/// `player` holds the pending choice: a terminal won by them that some option
/// chain reaches without the opponent moving, or None.
fn pending_forced_win(state: &GameState, player: usize, depth: usize) -> Option<GameState> {
    if state.phase == Phase::Complete {
        return (state.winner == Some(player)).then(|| state.clone());
    }
    let pending = state.pending_choice.as_ref()?;
    if pending.player != player || depth == 0 {
        return None;
    }
    for index in legal_action_indices(state) {
        let mut clone = state.clone();
        let action = decode_action(&clone, index);
        // A pending option whose outcome is hidden is not deterministic here;
        // the reference skips it (HiddenInformationError) and so does this.
        if clone.apply_with_chance(&action, &[]).is_err() {
            continue;
        }
        if let Some(won) = pending_forced_win(&clone, player, depth - 1) {
            return Some(won);
        }
    }
    None
}

/// Actions that could conceivably end the game for the actor this move.
fn winning_move_candidates(state: &GameState) -> Vec<usize> {
    legal_action_indices(state)
        .into_iter()
        .filter(|&index| {
            let action = decode_action(state, index);
            match action.use_ {
                ActionUse::ConstructWonder | ActionUse::ResolvePendingChoice => true,
                ActionUse::ConstructBuilding => {
                    let slot = action.slot.expect("construct names a slot");
                    let built = card(state.tableau.slots[slot].card_id);
                    built.shields > 0 || built.science.is_some()
                }
                _ => false,
            }
        })
        .collect()
}

/// A terminal state the actor of `state` can force THIS TURN, or None.
///
/// The returned witness is one won terminal; its victory type feeds the root
/// outlook. When several chance outcomes all win, they may win differently --
/// the value is exact, the witness's type is one of the forced ones.
pub fn guaranteed_win_now(state: &GameState, depth: usize) -> Option<GameState> {
    if state.phase == Phase::Complete {
        return None;
    }
    let player = actor(state);
    if !threat_possible(state, player) {
        return None;
    }
    if state.pending_choice.is_some() {
        return pending_forced_win(state, player, depth);
    }
    'actions: for index in winning_move_candidates(state) {
        let action = decode_action(state, index);
        let specs = chance::chance_signature(state, &action);
        if specs.iter().any(|spec| spec.kind == ChanceKind::AgeDeal) {
            continue;
        }
        let chains = if specs.is_empty() {
            vec![(Vec::new(), 1.0)]
        } else {
            chance::enumerate_chains_unkeyed(state, &specs)
        };
        let mut witness = None;
        for (outcomes, _probability) in chains {
            let mut clone = state.clone();
            if clone.apply_with_chance(&action, &outcomes).is_err() {
                continue 'actions;
            }
            let won = if clone.winner == Some(player) {
                Some(clone)
            } else {
                pending_forced_win(&clone, player, depth)
            };
            match won {
                Some(terminal) => {
                    witness.get_or_insert(terminal);
                }
                None => continue 'actions,
            }
        }
        if witness.is_some() {
            return witness;
        }
    }
    None
}

/// Every consistent child of one action, or None when the outcomes cannot be
/// enumerated (an Age deal) or applied. None means "cannot prove": a loss
/// proof must treat it as an escape. `phase_e._resolved_children`.
fn resolved_children(state: &GameState, index: usize) -> Option<Vec<GameState>> {
    let action = decode_action(state, index);
    let specs = chance::chance_signature(state, &action);
    if specs.iter().any(|spec| spec.kind == ChanceKind::AgeDeal) {
        return None;
    }
    let chains = if specs.is_empty() {
        vec![(Vec::new(), 1.0)]
    } else {
        chance::enumerate_chains_unkeyed(state, &specs)
    };
    let mut children = Vec::with_capacity(chains.len());
    for (outcomes, _probability) in chains {
        let mut clone = state.clone();
        clone.apply_with_chance(&action, &outcomes).ok()?;
        children.push(clone);
    }
    Some(children)
}

/// After `actor`'s action resolved: a terminal the opponent forces, or None
/// when `actor` is not certainly lost. `phase_e._outcome_lost`.
fn outcome_lost(child: &GameState, actor: usize, depth: usize) -> Option<GameState> {
    if child.phase == Phase::Complete {
        return match child.winner {
            Some(winner) if winner != actor => Some(child.clone()),
            _ => None,
        };
    }
    if crate::tree::state_actor(child) == actor {
        // Own pending choice: lost only if EVERY option is. An extra turn
        // keeps the initiative and is never lost here.
        if child.pending_choice.is_none() || depth == 0 {
            return None;
        }
        let mut witness = None;
        for index in legal_action_indices(child) {
            for grandchild in resolved_children(child, index)? {
                let lost = outcome_lost(&grandchild, actor, depth - 1)?;
                witness.get_or_insert(lost);
            }
        }
        return witness;
    }
    guaranteed_win_now(child, depth)
}

/// G4 layer 1b: EVERY action of the mover, under EVERY consistent chance
/// outcome, leaves the opponent a guaranteed win next move. Returns one of the
/// opponent's forced terminals as the witness. `phase_e.guaranteed_loss_now`.
pub fn guaranteed_loss_now(state: &GameState, depth: usize) -> Option<GameState> {
    if state.phase == Phase::Complete {
        return None;
    }
    let mover = actor(state);
    // The mover's action adds to the opponent's symbols or shields only by
    // revealing a card, which the screen already allows for.
    if !threat_possible(state, 1 - mover) {
        return None;
    }
    let legal = legal_action_indices(state);
    if legal.is_empty() {
        return None;
    }
    let mut witness = None;
    for index in legal {
        for child in resolved_children(state, index)? {
            let lost = outcome_lost(&child, mover, depth)?;
            witness.get_or_insert(lost);
        }
    }
    witness
}

/// `(value_p0, outlook)` for a node the switch says to check, or None: a
/// guaranteed win for the mover, else a guaranteed loss.
pub fn proven_value(state: &GameState) -> Option<(f64, crate::eval::Outlook)> {
    if !enabled() {
        return None;
    }
    let terminal = guaranteed_win_now(state, PENDING_DEPTH).or_else(|| {
        losses_enabled()
            .then(|| guaranteed_loss_now(state, PENDING_DEPTH))
            .flatten()
    })?;
    Some((
        crate::eval::terminal_value_p0(&terminal),
        crate::eval::terminal_outlook_p0(&terminal),
    ))
}
