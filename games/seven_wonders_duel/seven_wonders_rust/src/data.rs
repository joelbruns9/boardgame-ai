//! Static component data for the 7 Wonders Duel base game.
//!
//! The enum ordinals and the id tables mirror `data.py` exactly: a card's id is
//! its index in `ALL_BUILDING_CARDS` (Age I ++ Age II ++ Age III ++ Guilds),
//! a wonder's id its index in `WONDERS`, a token's id its index in
//! `PROGRESS_TOKENS`. `data_gen.rs` (machine-generated from `data.py`) supplies
//! the concrete tables so the facts cannot drift by transcription; this module
//! owns the shapes and the lookups over them.

// A few accessors here (BackType, back_type_of*, RESOURCES, Cost::resource_count)
// are consumed by the F2 encoder/codec, not yet by the F1 engine. Kept alongside
// the generated tables they operate on rather than reintroduced later.
#![allow(dead_code)]

// Enum ordinals are load-bearing: they match Python's definition order and are
// used directly in the cross-language fingerprint. Do not reorder.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Resource {
    Wood = 0,
    Clay = 1,
    Stone = 2,
    Glass = 3,
    Papyrus = 4,
}
pub const RESOURCES: [Resource; 5] = [
    Resource::Wood,
    Resource::Clay,
    Resource::Stone,
    Resource::Glass,
    Resource::Papyrus,
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum CardColor {
    Brown = 0,
    Grey = 1,
    Blue = 2,
    Green = 3,
    Yellow = 4,
    Red = 5,
    Purple = 6,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum BackType {
    AgeI = 0,
    AgeII = 1,
    AgeIII = 2,
    Guild = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ScienceSymbol {
    ArmillarySphere = 0,
    Wheel = 1,
    Sundial = 2,
    MortarAndPestle = 3,
    SetSquare = 4,
    QuillAndInk = 5,
    Law = 6,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum EffectKind {
    ImmediateCoins = 0,
    OpponentLosesCoins = 1,
    PlayAgain = 2,
    CoinsPerOwnColor = 3,
    CoinsPerOwnWonder = 4,
    CoinsPerMostColor = 5,
    CoinsPerMostBrownGrey = 6,
    VpPerMostColor = 7,
    VpPerMostWonder = 8,
    VpPerRichestCoinSet = 9,
    VpPerMostBrownGrey = 10,
    DestroyOpponentBrown = 11,
    DestroyOpponentGrey = 12,
    BuildFromDiscardFree = 13,
    ChooseUnusedProgress = 14,
    FutureWonderResourceDiscount = 15,
    ReceiveOpponentTradeSpend = 16,
    FutureBlueResourceDiscount = 17,
    VpPerProgress = 18,
    FutureRedExtraShield = 19,
    FutureWonderPlayAgain = 20,
    CoinsPerChainBuild = 21,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Cost {
    pub coins: i32,
    pub wood: i32,
    pub clay: i32,
    pub stone: i32,
    pub glass: i32,
    pub papyrus: i32,
}

impl Cost {
    pub const ZERO: Cost = Cost {
        coins: 0,
        wood: 0,
        clay: 0,
        stone: 0,
        glass: 0,
        papyrus: 0,
    };

    #[inline]
    pub fn resource_count(&self, r: Resource) -> i32 {
        match r {
            Resource::Wood => self.wood,
            Resource::Clay => self.clay,
            Resource::Stone => self.stone,
            Resource::Glass => self.glass,
            Resource::Papyrus => self.papyrus,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Effect {
    pub kind: EffectKind,
    pub amount: i32,
    pub color: Option<CardColor>,
}

#[derive(Clone, Copy, Debug)]
pub struct CardData {
    pub name: &'static str,
    pub age: u8,
    pub color: CardColor,
    pub cost: Cost,
    pub victory_points: i32,
    pub shields: i32,
    pub fixed_production: &'static [Resource],
    pub choice_production: &'static [Resource],
    pub trade_discount: &'static [Resource],
    pub science: Option<ScienceSymbol>,
    /// Chain tokens are assigned small stable ids by the generator; a card is a
    /// free chain build when it `chain_from`s a token some built card `chain_to`s.
    pub chain_from: Option<u16>,
    pub chain_to: Option<u16>,
    pub effects: &'static [Effect],
}

#[derive(Clone, Copy, Debug)]
pub struct WonderData {
    pub name: &'static str,
    pub cost: Option<Cost>,
    pub victory_points: i32,
    pub shields: i32,
    pub choice_production: &'static [Resource],
    pub effects: &'static [Effect],
}

#[derive(Clone, Copy, Debug)]
pub struct ProgressData {
    pub name: &'static str,
    pub victory_points: i32,
    pub science: Option<ScienceSymbol>,
    pub effects: &'static [Effect],
}

#[derive(Clone, Copy, Debug)]
pub struct SlotDef {
    pub row: i32,
    pub x: i32,
    pub face_up: bool,
}

// Generated tables: CARDS, WONDERS, PROGRESS, LAYOUT_AGE_1/2/3.
include!("data_gen.rs");

pub const NUM_CARDS: usize = CARDS.len();
pub const NUM_WONDERS: usize = WONDERS.len();
pub const NUM_PROGRESS: usize = PROGRESS.len();

#[inline]
pub fn card(id: usize) -> &'static CardData {
    &CARDS[id]
}
#[inline]
pub fn wonder(id: usize) -> &'static WonderData {
    &WONDERS[id]
}
#[inline]
pub fn progress(id: usize) -> &'static ProgressData {
    &PROGRESS[id]
}

/// Tableau layout for an age (1..=3).
pub fn layout(age: u8) -> &'static [SlotDef] {
    match age {
        1 => &LAYOUT_AGE_1,
        2 => &LAYOUT_AGE_2,
        3 => &LAYOUT_AGE_3,
        _ => panic!("invalid age: {age}"),
    }
}

/// Back type of a card id (Purple ⇒ Guild, else by age), matching
/// `data.back_type_of`.
pub fn back_type_of(card_id: usize) -> BackType {
    let c = card(card_id);
    if c.color == CardColor::Purple {
        return BackType::Guild;
    }
    match c.age {
        1 => BackType::AgeI,
        2 => BackType::AgeII,
        3 => BackType::AgeIII,
        _ => panic!("invalid card age"),
    }
}

pub fn back_type_of_age(age: u8) -> BackType {
    match age {
        1 => BackType::AgeI,
        2 => BackType::AgeII,
        3 => BackType::AgeIII,
        _ => panic!("invalid age"),
    }
}

use std::collections::HashMap;
use std::sync::OnceLock;

fn name_index(names: impl Iterator<Item = &'static str>) -> HashMap<&'static str, usize> {
    names.enumerate().map(|(i, n)| (n, i)).collect()
}

pub fn card_id(name: &str) -> usize {
    static IDX: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
    let m = IDX.get_or_init(|| name_index(CARDS.iter().map(|c| c.name)));
    *m.get(name)
        .unwrap_or_else(|| panic!("unknown card: {name}"))
}

pub fn wonder_id(name: &str) -> usize {
    static IDX: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
    let m = IDX.get_or_init(|| name_index(WONDERS.iter().map(|w| w.name)));
    *m.get(name)
        .unwrap_or_else(|| panic!("unknown wonder: {name}"))
}

pub fn progress_id(name: &str) -> usize {
    static IDX: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
    let m = IDX.get_or_init(|| name_index(PROGRESS.iter().map(|p| p.name)));
    *m.get(name)
        .unwrap_or_else(|| panic!("unknown progress token: {name}"))
}
