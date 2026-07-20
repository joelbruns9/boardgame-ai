//! Fixed 1202-action identity-indexed codec — a port of `codec.py`.
//!
//! Index blocks (spec §3.1):
//!   WONDER_DRAFT 0..12, BUILD 12..85, DISCARD 85..158,
//!   CARD_TO_WONDER 158..1034, DESTROY 1034..1107, MAUSOLEUM 1107..1180,
//!   PROGRESS_BOARD 1180..1190, PROGRESS_LIBRARY 1190..1200, NEXT_AGE 1200..1202.

use crate::data::{NUM_CARDS, NUM_PROGRESS, NUM_WONDERS};
use crate::engine::{Action, ActionUse};
use crate::state::{GameState, PendingChoiceKind};

pub const WONDER_DRAFT_BASE: usize = 0;
pub const BUILD_BASE: usize = WONDER_DRAFT_BASE + NUM_WONDERS; // 12
pub const DISCARD_BASE: usize = BUILD_BASE + NUM_CARDS; // 85
pub const CARD_TO_WONDER_BASE: usize = DISCARD_BASE + NUM_CARDS; // 158
pub const DESTROY_BASE: usize = CARD_TO_WONDER_BASE + NUM_CARDS * NUM_WONDERS; // 1034
pub const MAUSOLEUM_BASE: usize = DESTROY_BASE + NUM_CARDS; // 1107
pub const PROGRESS_BOARD_BASE: usize = MAUSOLEUM_BASE + NUM_CARDS; // 1180
pub const PROGRESS_LIBRARY_BASE: usize = PROGRESS_BOARD_BASE + NUM_PROGRESS; // 1190
pub const NEXT_AGE_BASE: usize = PROGRESS_LIBRARY_BASE + NUM_PROGRESS; // 1200
pub const NUM_ACTIONS: usize = NEXT_AGE_BASE + 2; // 1202

fn card_at_slot(g: &GameState, slot: usize) -> usize {
    let c = &g.tableau.slots[slot];
    assert!(c.present && c.revealed, "slot holds no revealed card");
    c.card_id
}

fn pending_base(kind: PendingChoiceKind) -> usize {
    match kind {
        PendingChoiceKind::DestroyOpponentBrown | PendingChoiceKind::DestroyOpponentGrey => {
            DESTROY_BASE
        }
        PendingChoiceKind::BuildFromDiscardFree => MAUSOLEUM_BASE,
        PendingChoiceKind::ChooseAvailableProgress => PROGRESS_BOARD_BASE,
        PendingChoiceKind::ChooseUnusedProgress => PROGRESS_LIBRARY_BASE,
    }
}

pub fn encode_action(g: &GameState, action: &Action) -> usize {
    match action.use_ {
        ActionUse::DraftWonder => WONDER_DRAFT_BASE + action.wonder.unwrap(),
        ActionUse::ConstructBuilding => BUILD_BASE + card_at_slot(g, action.slot.unwrap()),
        ActionUse::DiscardForCoins => DISCARD_BASE + card_at_slot(g, action.slot.unwrap()),
        ActionUse::ConstructWonder => {
            let card_id = card_at_slot(g, action.slot.unwrap());
            CARD_TO_WONDER_BASE + card_id * NUM_WONDERS + action.wonder.unwrap()
        }
        ActionUse::ResolvePendingChoice => {
            let pending = g.pending_choice.as_ref().expect("no pending choice");
            pending_base(pending.kind) + action.choice.unwrap()
        }
        ActionUse::ChooseNextStartPlayer => {
            let sp = action.starting_player.unwrap();
            NEXT_AGE_BASE + if sp == g.active_player { 0 } else { 1 }
        }
    }
}

pub fn decode_action(g: &GameState, index: usize) -> Action {
    assert!(index < NUM_ACTIONS, "action index out of range: {index}");

    if index < BUILD_BASE {
        return Action {
            use_: ActionUse::DraftWonder,
            slot: None,
            wonder: Some(index),
            choice: None,
            starting_player: None,
        };
    }
    if index < DISCARD_BASE {
        let card_id = index - BUILD_BASE;
        let slot = g
            .tableau
            .accessible_slot_of(card_id)
            .expect("card not in an accessible slot");
        return Action {
            use_: ActionUse::ConstructBuilding,
            slot: Some(slot),
            wonder: None,
            choice: None,
            starting_player: None,
        };
    }
    if index < CARD_TO_WONDER_BASE {
        let card_id = index - DISCARD_BASE;
        let slot = g
            .tableau
            .accessible_slot_of(card_id)
            .expect("card not in an accessible slot");
        return Action {
            use_: ActionUse::DiscardForCoins,
            slot: Some(slot),
            wonder: None,
            choice: None,
            starting_player: None,
        };
    }
    if index < DESTROY_BASE {
        let offset = index - CARD_TO_WONDER_BASE;
        let card_id = offset / NUM_WONDERS;
        let wonder_id = offset % NUM_WONDERS;
        let slot = g
            .tableau
            .accessible_slot_of(card_id)
            .expect("card not in an accessible slot");
        return Action {
            use_: ActionUse::ConstructWonder,
            slot: Some(slot),
            wonder: Some(wonder_id),
            choice: None,
            starting_player: None,
        };
    }
    if index < NEXT_AGE_BASE {
        let (base, choice) = if index < MAUSOLEUM_BASE {
            (DESTROY_BASE, index - DESTROY_BASE)
        } else if index < PROGRESS_BOARD_BASE {
            (MAUSOLEUM_BASE, index - MAUSOLEUM_BASE)
        } else if index < PROGRESS_LIBRARY_BASE {
            (PROGRESS_BOARD_BASE, index - PROGRESS_BOARD_BASE)
        } else {
            (PROGRESS_LIBRARY_BASE, index - PROGRESS_LIBRARY_BASE)
        };
        let pending = g.pending_choice.as_ref().expect("no pending choice in state");
        assert_eq!(
            pending_base(pending.kind),
            base,
            "index {index} does not match the pending choice"
        );
        return Action {
            use_: ActionUse::ResolvePendingChoice,
            slot: None,
            wonder: None,
            choice: Some(choice),
            starting_player: None,
        };
    }
    // NEXT_AGE_BASE (self starts) or +1 (opponent starts), actor-relative.
    let starting_player = if index == NEXT_AGE_BASE {
        g.active_player
    } else {
        1 - g.active_player
    };
    Action {
        use_: ActionUse::ChooseNextStartPlayer,
        slot: None,
        wonder: None,
        choice: None,
        starting_player: Some(starting_player),
    }
}

pub fn legal_action_indices(g: &GameState) -> Vec<usize> {
    let mut idx: Vec<usize> = g
        .legal_actions()
        .iter()
        .map(|a| encode_action(g, a))
        .collect();
    idx.sort_unstable();
    idx
}
