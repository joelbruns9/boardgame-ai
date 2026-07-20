//! Rule primitives mirroring `rules.py`.

pub const STARTING_COINS: i32 = 7;

/// Bank price for one missing resource given the opponent's matching brown/grey
/// production. Callers apply commercial (one-coin) discounts separately.
#[inline]
pub fn normal_trade_unit_cost(opponent_brown_grey_production: i32) -> i32 {
    2 + opponent_brown_grey_production
}

/// Coins received for discarding an accessible Age card.
#[inline]
pub fn discard_income(yellow_buildings: i32) -> i32 {
    2 + yellow_buildings
}
