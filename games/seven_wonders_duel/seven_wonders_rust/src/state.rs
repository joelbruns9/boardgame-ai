//! Complete simulator state and the cross-language logic fingerprint.
//!
//! Mirrors `game.py::GameState` field-for-field, with two deliberate departures
//! documented in `PHASE_F.md`:
//!
//! 1. No Python `random.Random` is modelled. The engine is constructed from a
//!    fully-locked setup (all decks, groups, guild selection, progress split),
//!    exactly as `GameState.new` produces it, so every chance event except the
//!    Great Library draw resolves deterministically from locked state — the
//!    same "simulator" path `buffer.replay` follows. The Great Library draw is
//!    the one RNG-at-play-time event; its outcomes are supplied from the
//!    recorded `chance_log` via `library_draws`.
//! 2. The fingerprint excludes Python's RNG internal state (unrepresentable in
//!    Rust) and uses numeric id sorts throughout; the Python gate helper mirrors
//!    this exact serialization, so equality is a genuine byte-for-byte gate over
//!    all game-logic state.

use crate::data::{layout, ScienceSymbol};
use std::collections::VecDeque;

pub type SlotId = (i32, i32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    WonderDraft = 0,
    PlayAge = 1,
    ChooseNextStartPlayer = 2,
    Complete = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VictoryType {
    Military = 0,
    Scientific = 1,
    Civilian = 2,
    SharedCivilian = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PendingChoiceKind {
    DestroyOpponentBrown = 0,
    DestroyOpponentGrey = 1,
    BuildFromDiscardFree = 2,
    ChooseUnusedProgress = 3,
    ChooseAvailableProgress = 4,
}

/// Option ids in a pending choice live in either the card id space (destroy /
/// mausoleum) or the progress id space (progress picks). Kind determines which.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingChoice {
    pub kind: PendingChoiceKind,
    pub player: usize,
    pub options: Vec<usize>,
    pub consume_all_options: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CityState {
    pub coins: i32,
    pub wonders: Vec<usize>,        // wonder ids, draft order
    pub built_wonders: Vec<usize>,  // wonder ids, build order
    pub buildings: Vec<usize>,      // card ids, build order
    pub progress_tokens: Vec<usize>, // progress ids, acquire order
    pub claimed_science_pairs: Vec<ScienceSymbol>, // set semantics
}

impl CityState {
    fn new() -> Self {
        CityState {
            coins: crate::rules::STARTING_COINS,
            ..Default::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableauCard {
    pub card_id: usize,
    pub revealed: bool,
    pub present: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableauState {
    pub age: u8,
    /// One entry per slot in `layout(age)`, in layout order.
    pub slots: Vec<TableauCard>,
}

impl TableauState {
    pub fn from_deck(age: u8, card_ids: &[usize]) -> Self {
        let lay = layout(age);
        assert_eq!(card_ids.len(), lay.len(), "age {age} tableau size mismatch");
        let slots = lay
            .iter()
            .zip(card_ids.iter())
            .map(|(slot, &cid)| TableauCard {
                card_id: cid,
                revealed: slot.face_up,
                present: true,
            })
            .collect();
        TableauState { age, slots }
    }

    #[inline]
    pub fn slot_id(&self, i: usize) -> SlotId {
        let s = &layout(self.age)[i];
        (s.row, s.x)
    }

    pub fn is_accessible(&self, i: usize) -> bool {
        let card = &self.slots[i];
        if !card.present {
            return false;
        }
        !coverers(self.age, i)
            .iter()
            .any(|&c| self.slots[c].present)
    }

    /// Slot indices of accessible cards, in layout order.
    pub fn accessible_indices(&self) -> Vec<usize> {
        (0..self.slots.len())
            .filter(|&i| self.is_accessible(i))
            .collect()
    }

    /// Remove the card at slot `i`; return (card_id, newly-accessible face-down
    /// slot indices sorted by (row, x)) — mirrors `TableauState.take_accessible`.
    pub fn take_accessible(&mut self, i: usize) -> (usize, Vec<usize>) {
        assert!(self.is_accessible(i), "slot not accessible: {i}");
        assert!(self.slots[i].revealed, "accessible card must be revealed");
        let card_id = self.slots[i].card_id;
        self.slots[i].present = false;
        let mut newly: Vec<usize> = (0..self.slots.len())
            .filter(|&j| {
                self.slots[j].present && !self.slots[j].revealed && self.is_accessible(j)
            })
            .collect();
        newly.sort_by_key(|&j| self.slot_id(j));
        (card_id, newly)
    }

    /// Slot index at layout position `(row, x)`, if any.
    pub fn slot_index_of(&self, row: i32, x: i32) -> Option<usize> {
        (0..self.slots.len()).find(|&i| self.slot_id(i) == (row, x))
    }

    pub fn reveal(&mut self, i: usize) {
        assert!(self.slots[i].present, "cannot reveal an absent card");
        self.slots[i].revealed = true;
    }

    /// Index of the unique accessible slot holding `card_id` (accessible ⇒
    /// revealed; no duplicate cards ⇒ bijective). Mirrors `_slot_for_card`.
    pub fn accessible_slot_of(&self, card_id: usize) -> Option<usize> {
        (0..self.slots.len())
            .find(|&i| self.is_accessible(i) && self.slots[i].card_id == card_id)
    }
}

/// Covering slots for `slot_i` in `age`: next-row slots whose x differs by 1.
/// Cached per age.
pub(crate) fn coverers(age: u8, slot_i: usize) -> &'static [usize] {
    use std::sync::OnceLock;
    static CACHE: OnceLock<[Vec<Vec<usize>>; 4]> = OnceLock::new();
    let table = CACHE.get_or_init(|| {
        let build = |age: u8| -> Vec<Vec<usize>> {
            let lay = layout(age);
            lay.iter()
                .map(|slot| {
                    lay.iter()
                        .enumerate()
                        .filter(|(_, cand)| {
                            cand.row == slot.row + 1 && (cand.x - slot.x).abs() == 1
                        })
                        .map(|(j, _)| j)
                        .collect()
                })
                .collect()
        };
        [Vec::new(), build(1), build(2), build(3)]
    });
    &table[age as usize][slot_i]
}

/// `PartialEq`/`Eq` cover *every* field — including `library_draws`, which the
/// cross-language `fingerprint` omits (Python has no equivalent remaining-draws
/// queue). The F1b make/unmake audit compares whole states via this, so a future
/// journaled undo that forgets any non-fingerprinted field is caught.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameState {
    pub first_player: usize,
    pub phase: Phase,
    pub active_player: usize,
    pub age: u8,
    pub cities: [CityState; 2],
    pub available_progress_tokens: Vec<usize>,
    pub unused_progress_tokens: Vec<usize>,
    pub wonder_groups: [Vec<usize>; 2],
    pub unused_wonders: Vec<usize>,
    pub wonder_offer: Vec<usize>,
    pub wonder_round: u8,
    pub wonder_pick_index: u8,
    pub age_decks: [Vec<usize>; 4], // index by age 1..=3
    pub removed_age_cards: [Vec<usize>; 4],
    pub selected_guilds: Vec<usize>,
    pub unused_guilds: Vec<usize>,
    pub tableau: TableauState,
    pub discard_pile: Vec<usize>,
    pub buried_cards: Vec<usize>,
    pub retired_wonders: Vec<usize>,
    pub wonder_burials: Vec<(usize, usize)>, // (wonder_id, card_id)
    pub pending_choice: Option<PendingChoice>,
    pub pending_extra_turn: bool,
    pub pending_shields: i32,
    pub conflict_position: i32,
    pub military_tokens_remaining: Vec<(i32, i32)>, // (position, penalty)
    pub winner: Option<usize>,
    pub victory_type: Option<VictoryType>,
    pub final_scores: Option<(i32, i32)>,
    /// Recorded Great Library draws (progress ids), consumed in order — the one
    /// play-time RNG event. Empty for positions that never reach a Great Library.
    pub library_draws: VecDeque<Vec<usize>>,
}

/// Fully-locked setup, extracted from a Python `GameState.new(seed)`.
pub struct Setup {
    pub first_player: usize,
    pub available_progress_tokens: Vec<usize>,
    pub unused_progress_tokens: Vec<usize>,
    pub wonder_groups: [Vec<usize>; 2],
    pub unused_wonders: Vec<usize>,
    pub age_decks: [Vec<usize>; 4],
    pub removed_age_cards: [Vec<usize>; 4],
    pub selected_guilds: Vec<usize>,
    pub unused_guilds: Vec<usize>,
}

impl GameState {
    pub fn from_setup(setup: Setup, library_draws: VecDeque<Vec<usize>>) -> Self {
        let tableau = TableauState::from_deck(1, &setup.age_decks[1]);
        GameState {
            first_player: setup.first_player,
            phase: Phase::WonderDraft,
            active_player: setup.first_player,
            age: 1,
            cities: [CityState::new(), CityState::new()],
            available_progress_tokens: setup.available_progress_tokens,
            unused_progress_tokens: setup.unused_progress_tokens,
            wonder_offer: setup.wonder_groups[0].clone(),
            wonder_groups: setup.wonder_groups,
            unused_wonders: setup.unused_wonders,
            wonder_round: 0,
            wonder_pick_index: 0,
            age_decks: setup.age_decks,
            removed_age_cards: setup.removed_age_cards,
            selected_guilds: setup.selected_guilds,
            unused_guilds: setup.unused_guilds,
            tableau,
            discard_pile: Vec::new(),
            buried_cards: Vec::new(),
            retired_wonders: Vec::new(),
            wonder_burials: Vec::new(),
            pending_choice: None,
            pending_extra_turn: false,
            pending_shields: 0,
            conflict_position: 0,
            military_tokens_remaining: vec![(-7, 5), (-4, 2), (4, 2), (7, 5)],
            winner: None,
            victory_type: None,
            final_scores: None,
            library_draws,
        }
    }

    // --- draft helpers (mirror game.py) ---

    pub fn draft_order(&self, round_index: u8) -> [usize; 4] {
        let first = if round_index == 0 {
            self.first_player
        } else {
            1 - self.first_player
        };
        let second = 1 - first;
        [first, second, second, first]
    }

    pub fn legal_wonder_choices(&self) -> Vec<usize> {
        if self.phase != Phase::WonderDraft {
            return Vec::new();
        }
        self.wonder_offer.clone()
    }

    /// Apply one draft pick; return true when it flipped the second group.
    pub fn pick_wonder(&mut self, wonder_id: usize) -> bool {
        assert_eq!(self.phase, Phase::WonderDraft, "draft complete");
        let pos = self
            .wonder_offer
            .iter()
            .position(|&w| w == wonder_id)
            .expect("wonder not in offer");
        let expected = self.draft_order(self.wonder_round)[self.wonder_pick_index as usize];
        assert_eq!(self.active_player, expected, "draft order mismatch");
        self.cities[self.active_player].wonders.push(wonder_id);
        self.wonder_offer.remove(pos);
        self.wonder_pick_index += 1;

        if self.wonder_pick_index < 4 {
            self.active_player =
                self.draft_order(self.wonder_round)[self.wonder_pick_index as usize];
            return false;
        }
        if self.wonder_round == 0 {
            self.wonder_round = 1;
            self.wonder_pick_index = 0;
            self.wonder_offer = self.wonder_groups[1].clone();
            self.active_player = self.draft_order(1)[0];
            return true;
        }
        self.wonder_offer.clear();
        self.phase = Phase::PlayAge;
        self.active_player = self.first_player;
        false
    }

    // --- make / unmake (F1b) ---
    //
    // Snapshot-based undo: correct by construction and enough for the Phase H
    // solver's API surface ("make/unmake from day one, not retrofitted"). A
    // journaled-delta undo is a documented F3 optimisation should search
    // profiling demand it; the F1b fingerprint round-trip gate validates either
    // implementation identically.

    pub fn snapshot(&self) -> GameState {
        self.clone()
    }

    pub fn restore(&mut self, snap: GameState) {
        *self = snap;
    }

    // --- fingerprint ---

    /// Canonical integer serialization of all game-logic state. The Python gate
    /// helper `logic_fingerprint` mirrors this exactly (numeric id sorts, same
    /// field order, same length prefixes). RNG state is deliberately excluded.
    pub fn fingerprint(&self) -> Vec<i32> {
        let mut out: Vec<i32> = Vec::with_capacity(256);
        let push_list = |out: &mut Vec<i32>, v: &[usize]| {
            out.push(v.len() as i32);
            out.extend(v.iter().map(|&x| x as i32));
        };

        out.push(self.phase as i32);
        out.push(self.first_player as i32);
        out.push(self.active_player as i32);
        out.push(self.age as i32);
        out.push(self.wonder_round as i32);
        out.push(self.wonder_pick_index as i32);

        for city in &self.cities {
            out.push(city.coins);
            push_list(&mut out, &city.wonders);
            push_list(&mut out, &city.built_wonders);
            push_list(&mut out, &city.buildings);
            push_list(&mut out, &city.progress_tokens);
            let mut pairs: Vec<i32> = city
                .claimed_science_pairs
                .iter()
                .map(|&s| s as i32)
                .collect();
            pairs.sort_unstable();
            out.push(pairs.len() as i32);
            out.extend(pairs);
        }

        push_list(&mut out, &self.available_progress_tokens);
        push_list(&mut out, &self.unused_progress_tokens);
        push_list(&mut out, &self.wonder_groups[0]);
        push_list(&mut out, &self.wonder_groups[1]);
        push_list(&mut out, &self.unused_wonders);
        push_list(&mut out, &self.wonder_offer);
        for age in 1..=3 {
            push_list(&mut out, &self.age_decks[age]);
        }
        for age in 1..=3 {
            push_list(&mut out, &self.removed_age_cards[age]);
        }
        push_list(&mut out, &self.selected_guilds);
        push_list(&mut out, &self.unused_guilds);

        // Tableau, sorted by (row, x).
        let mut order: Vec<usize> = (0..self.tableau.slots.len()).collect();
        order.sort_by_key(|&i| self.tableau.slot_id(i));
        out.push(order.len() as i32);
        for i in order {
            let (row, x) = self.tableau.slot_id(i);
            let c = &self.tableau.slots[i];
            out.push(row);
            out.push(x);
            out.push(c.card_id as i32);
            out.push(c.present as i32);
            out.push((c.present && c.revealed) as i32);
        }

        push_list(&mut out, &self.discard_pile);
        push_list(&mut out, &self.buried_cards);

        // Wonder burials, sorted by wonder id.
        let mut burials = self.wonder_burials.clone();
        burials.sort_by_key(|&(w, _)| w);
        out.push(burials.len() as i32);
        for (w, c) in burials {
            out.push(w as i32);
            out.push(c as i32);
        }

        let mut retired = self.retired_wonders.clone();
        retired.sort_unstable();
        push_list(&mut out, &retired);

        // Pending choice.
        match &self.pending_choice {
            None => out.push(-1),
            Some(p) => {
                out.push(p.kind as i32);
                out.push(p.player as i32);
                out.push(p.consume_all_options as i32);
                push_list(&mut out, &p.options);
            }
        }
        out.push(self.pending_extra_turn as i32);
        out.push(self.pending_shields);

        out.push(self.conflict_position);
        let mut mil = self.military_tokens_remaining.clone();
        mil.sort_by_key(|&(pos, _)| pos);
        out.push(mil.len() as i32);
        for (pos, pen) in mil {
            out.push(pos);
            out.push(pen);
        }

        out.push(match self.winner {
            None => -1,
            Some(p) => p as i32,
        });
        out.push(match self.victory_type {
            None => -1,
            Some(v) => v as i32,
        });
        match self.final_scores {
            None => out.push(-1),
            Some((a, b)) => {
                out.push(1);
                out.push(a);
                out.push(b);
            }
        }
        out
    }
}
