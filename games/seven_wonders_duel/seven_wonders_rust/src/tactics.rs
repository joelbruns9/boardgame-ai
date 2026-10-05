//! G4 (`MODEL_GROWTH_PLAN.md`): exact immediate tactics for the searcher.
//!
//! A node whose outcome is FORCED is proven: its value is exact, so the
//! searcher treats it like a finished game -- no network call, no expansion,
//! and no later averaging of network estimates into it. Port of `tactics.py`,
//! the Python reference it is gated against (`test_exact_tactics.py`):
//!
//! * `guaranteed_win_now` -- the mover forces a win before the opponent moves
//!   again: military or science, through their own pending choices, through
//!   ONE extra turn, or on points by taking the last card of Age III; every
//!   chance outcome of every step must win;
//! * `guaranteed_loss_now` -- every action, under every outcome, leaves the
//!   opponent a forced win.
//!
//! Whatever cannot be seen through (an Age deal, an inapplicable outcome)
//! counts against the claim: a win proof skips it, a loss proof treats it as
//! an escape. The reach screens are necessary conditions, so they cost recall,
//! never soundness.

use crate::chance::{self, ChanceKind};
use crate::codec::{decode_action, legal_action_indices};
use crate::data::{card, progress_id, wonder, EffectKind};
use crate::engine::ActionUse;
use crate::state::{GameState, Phase};
use std::sync::atomic::{AtomicBool, Ordering};

/// One action's largest plausible shield swing. `tactics.MILITARY_REACH`.
const MILITARY_REACH: i32 = 5;
/// Steps of the mover's own pending choices followed. `tactics.PENDING_DEPTH`.
pub const PENDING_DEPTH: usize = 6;
/// Extra turns followed. `tactics.EXTRA_TURNS`.
pub const EXTRA_TURNS: usize = 1;

static EXACT_TACTICS: AtomicBool = AtomicBool::new(false);

/// Process-wide switch. OFF here; Phase D and the advisor host turn it on.
pub fn set_enabled(enabled: bool) {
    EXACT_TACTICS.store(enabled, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    EXACT_TACTICS.load(Ordering::Relaxed)
}

/// Proven LOSSES under their own switch, ON by default but only consulted
/// when `enabled()`.
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

fn present_cards(state: &GameState) -> usize {
    state.tableau.slots.iter().filter(|slot| slot.present).count()
}

/// Necessary condition for `player` to win within `turns` own actions.
/// `tactics._within_reach`.
fn within_reach(state: &GameState, player: usize, turns: usize) -> bool {
    let need = if player == 0 {
        9 - state.conflict_position
    } else {
        9 + state.conflict_position
    };
    if need <= MILITARY_REACH * turns as i32 {
        return true;
    }
    if state.science_symbols(player).len() + 2 * turns >= 6 {
        return true;
    }
    state.age == 3 && present_cards(state) <= turns
}

/// The chance outcomes of one action, or None when they cannot be enumerated
/// (an Age deal). `tactics._chains`.
fn chains(state: &GameState, index: usize) -> Option<Vec<Vec<Vec<usize>>>> {
    let action = decode_action(state, index);
    let specs = chance::chance_signature(state, &action);
    if specs.iter().any(|spec| spec.kind == ChanceKind::AgeDeal) {
        return None;
    }
    if specs.is_empty() {
        return Some(vec![Vec::new()]);
    }
    Some(
        chance::enumerate_chains_unkeyed(state, &specs)
            .into_iter()
            .map(|(outcomes, _probability)| outcomes)
            .collect(),
    )
}

/// `test` holds for EVERY consistent child of the action; returns the first
/// child's witness. Children are built one at a time and the walk stops at the
/// first failure -- most candidates fail in their first world, and building
/// every world first was ~50x the cost. `tactics._every_child`.
fn every_child<F>(state: &GameState, index: usize, mut test: F) -> Option<GameState>
where
    F: FnMut(&GameState) -> Option<GameState>,
{
    let outcomes_list = chains(state, index)?;
    let action = decode_action(state, index);
    let mut witness = None;
    for outcomes in outcomes_list {
        let mut child = state.clone();
        child.apply_with_chance(&action, &outcomes).ok()?;
        let terminal = test(&child)?;
        witness.get_or_insert(terminal);
    }
    witness
}

/// `player` still holds an unbuilt wonder that would grant an extra turn (a
/// play-again wonder, or any wonder with Theology). `tactics._can_replay`.
fn can_replay(state: &GameState, player: usize) -> bool {
    let city = &state.cities[player];
    let mut unbuilt = city.wonders.iter().filter(|w| !city.built_wonders.contains(w));
    if city.progress_tokens.contains(&progress_id("Theology")) {
        return unbuilt.next().is_some();
    }
    unbuilt.any(|&w| {
        wonder(w)
            .effects
            .iter()
            .any(|effect| effect.kind == EffectKind::PlayAgain)
    })
}

/// Necessary condition for a win that NEEDS the extra turn: a play-again
/// wonder brings no shields or symbols, so only Theology widens those reaches;
/// the civilian ending widens with each card. `tactics._replay_reach`.
fn replay_reach(state: &GameState, player: usize, extra: usize) -> bool {
    if extra == 0 || !can_replay(state, player) {
        return false;
    }
    if state.cities[player]
        .progress_tokens
        .contains(&progress_id("Theology"))
    {
        return within_reach(state, player, 1 + extra);
    }
    state.age == 3 && present_cards(state) <= 1 + extra
}

/// Wonder builds that hand the mover an extra turn: a play-again wonder, or
/// any wonder once they hold Theology. `tactics._replay_wonders`.
fn replay_wonders(state: &GameState) -> Vec<usize> {
    let player = actor(state);
    let theology = state.cities[player]
        .progress_tokens
        .contains(&progress_id("Theology"));
    legal_action_indices(state)
        .into_iter()
        .filter(|&index| {
            let action = decode_action(state, index);
            action.use_ == ActionUse::ConstructWonder
                && (theology
                    || wonder(action.wonder.expect("wonder build names a wonder"))
                        .effects
                        .iter()
                        .any(|effect| effect.kind == EffectKind::PlayAgain))
        })
        .collect()
}

/// `tactics._candidates`.
fn candidates(state: &GameState) -> Vec<usize> {
    let legal = legal_action_indices(state);
    if state.age == 3 && present_cards(state) <= 1 {
        return legal;
    }
    legal
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

/// After one of `player`'s actions resolved: a won terminal the line forces.
/// `tactics._won`.
fn won(child: &GameState, player: usize, depth: usize, extra: usize) -> Option<GameState> {
    if child.phase == Phase::Complete {
        return (child.winner == Some(player)).then(|| child.clone());
    }
    if actor(child) != player {
        return None;
    }
    if child.pending_choice.is_some() {
        return pending_win(child, player, depth, extra);
    }
    if extra == 0 {
        return None;
    }
    forced_win(child, depth, extra - 1)
}

fn pending_win(state: &GameState, player: usize, depth: usize, extra: usize) -> Option<GameState> {
    if depth == 0 {
        return None;
    }
    legal_action_indices(state).into_iter().find_map(|index| {
        every_child(state, index, |child| won(child, player, depth - 1, extra))
    })
}

fn forced_win(state: &GameState, depth: usize, extra: usize) -> Option<GameState> {
    if state.phase == Phase::Complete {
        return None;
    }
    let player = actor(state);
    let now = within_reach(state, player, 1);
    if !(now || replay_reach(state, player, extra)) {
        return None;
    }
    if state.pending_choice.is_some() {
        return pending_win(state, player, depth, extra);
    }
    // Out of reach this action: only an extra turn can get there.
    let candidates = if now {
        candidates(state)
    } else {
        replay_wonders(state)
    };
    candidates
        .into_iter()
        .find_map(|index| every_child(state, index, |child| won(child, player, depth, extra)))
}

/// A won terminal the mover can force before the opponent moves again, or
/// None. `tactics.forced_win`.
pub fn guaranteed_win_now(state: &GameState, depth: usize) -> Option<GameState> {
    forced_win(state, depth, EXTRA_TURNS)
}

/// After the mover's action resolved: a terminal the opponent forces, or None
/// when the mover is not certainly lost. `tactics._lost`.
fn lost(child: &GameState, mover: usize, depth: usize) -> Option<GameState> {
    if child.phase == Phase::Complete {
        return match child.winner {
            Some(winner) if winner != mover => Some(child.clone()),
            _ => None,
        };
    }
    if actor(child) == mover {
        // Own pending choice: lost only if EVERY option is. An extra turn
        // keeps the initiative and is never proven lost here.
        if child.pending_choice.is_none() || depth == 0 {
            return None;
        }
        let mut witness = None;
        for index in legal_action_indices(child) {
            let terminal = every_child(child, index, |g| lost(g, mover, depth - 1))?;
            witness.get_or_insert(terminal);
        }
        return witness;
    }
    guaranteed_win_now(child, depth)
}

/// EVERY action of the mover, under every outcome, leaves the opponent a
/// forced win; returns one of the opponent's terminals. `tactics.forced_loss`.
pub fn guaranteed_loss_now(state: &GameState, depth: usize) -> Option<GameState> {
    if state.phase == Phase::Complete {
        return None;
    }
    let mover = actor(state);
    let opponent = 1 - mover;
    // The mover's action brings the opponent no shields or symbols (bar a
    // reveal, which the reaches allow for), but it does take a card.
    // The mover's own action takes a card, so every civilian reach is one card
    // wider here.
    let present = present_cards(state);
    if !(within_reach(state, opponent, 1)
        || replay_reach(state, opponent, EXTRA_TURNS)
        || (state.age == 3 && present <= 2)
        || (state.age == 3 && present <= 2 + EXTRA_TURNS && can_replay(state, opponent)))
    {
        return None;
    }
    let mut witness = None;
    for index in legal_action_indices(state) {
        let terminal = every_child(state, index, |child| lost(child, mover, depth))?;
        witness.get_or_insert(terminal);
    }
    witness
}

/// G0: every legal action of the mover, labelled exactly: `+1` when it forces
/// a win before the opponent moves again, `-1` when every outcome leaves the
/// opponent a forced win, `0` when neither is proven. Aligned to
/// `legal_action_indices`. `tactics.classify_actions`.
pub fn classify_actions(state: &GameState) -> Vec<i8> {
    if state.phase == Phase::Complete {
        return Vec::new();
    }
    let mover = actor(state);
    let can_win = within_reach(state, mover, 1) || replay_reach(state, mover, EXTRA_TURNS);
    let present = present_cards(state);
    let can_lose = within_reach(state, 1 - mover, 1)
        || replay_reach(state, 1 - mover, EXTRA_TURNS)
        || (state.age == 3 && present <= 2)
        || (state.age == 3 && present <= 2 + EXTRA_TURNS && can_replay(state, 1 - mover));
    legal_action_indices(state)
        .into_iter()
        .map(|index| {
            if can_win
                && every_child(state, index, |child| {
                    won(child, mover, PENDING_DEPTH, EXTRA_TURNS)
                })
                .is_some()
            {
                1
            } else if can_lose
                && every_child(state, index, |child| lost(child, mover, PENDING_DEPTH)).is_some()
            {
                -1
            } else {
                0
            }
        })
        .collect()
}

/// G0 reveal traps: per legal action, `(losing mass, reveals)` -- the
/// probability, over the action's chance outcomes, that the opponent is then
/// left a forced win, and whether the action reveals a card -- or None when
/// the outcomes cannot be enumerated (an Age deal) or applied. Every outcome
/// is checked (no early exit: the mass is the quantity). `tactics.losing_mass`.
pub fn losing_mass(state: &GameState) -> Vec<Option<(f64, bool)>> {
    if state.phase == Phase::Complete {
        return Vec::new();
    }
    let mover = actor(state);
    let present = present_cards(state);
    let can_lose = within_reach(state, 1 - mover, 1)
        || replay_reach(state, 1 - mover, EXTRA_TURNS)
        || (state.age == 3 && present <= 2)
        || (state.age == 3 && present <= 2 + EXTRA_TURNS && can_replay(state, 1 - mover));
    legal_action_indices(state)
        .into_iter()
        .map(|index| {
            let action = decode_action(state, index);
            let specs = chance::chance_signature(state, &action);
            if specs.iter().any(|spec| spec.kind == ChanceKind::AgeDeal) {
                return None;
            }
            let reveals = specs.iter().any(|spec| spec.kind == ChanceKind::CardReveal);
            if !can_lose {
                return Some((0.0, reveals));
            }
            let chains = if specs.is_empty() {
                vec![(Vec::new(), 1.0)]
            } else {
                chance::enumerate_chains_unkeyed(state, &specs)
            };
            let mut mass = 0.0;
            for (outcomes, probability) in chains {
                let mut child = state.clone();
                child.apply_with_chance(&action, &outcomes).ok()?;
                if lost(&child, mover, PENDING_DEPTH).is_some() {
                    mass += probability;
                }
            }
            Some((mass, reveals))
        })
        .collect()
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
