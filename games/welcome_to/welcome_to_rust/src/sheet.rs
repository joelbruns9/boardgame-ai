//! The player sheet — a mirror of `games/welcome_to/sheet.py`.
//!
//! Layout: fixed arrays, not `Vec`s (RUST_PORT_PLAN.md §6). Every street row is
//! `MAX_STREET_LEN` wide and the tail beyond `STREET_SIZES[x]` is never read;
//! every loop is bounded by `STREET_SIZES` / `FENCE_SIZES`, exactly as the
//! Python iterates its ragged lists.

use crate::constants::{
    BIS_BOXES, BIS_SCORES, EMPTY, ESTATE_ROW_BOXES, ESTATE_ROW_SCORES, FENCE_SIZES,
    MAX_ESTATE_SIZE, MAX_NUMBER, MAX_STREET_LEN, MIN_NUMBER, NUM_STREETS, PARK_BOXES,
    PARK_SCORES, PERMIT_BOXES, PERMIT_SCORES, POOL_BOXES, POOL_POSITIONS, POOL_SCORES,
    ROUNDABOUT, ROUNDABOUT_BOXES, ROUNDABOUT_SCORES, STREET_SIZES,
};

/// `(street, box)`.
pub type Pos = (usize, usize);
/// `(street, first box, size)` — a housing estate.
pub type Estate = (usize, usize, usize);

/// Everything a single sheet is worth. `plans` and `temp` are filled in by the
/// game, which is why `local_score` leaves them at zero.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct SheetScore {
    pub parks: i32,
    pub pools: i32,
    pub estates: i32,
    pub plans: i32,
    pub temp: i32,
    pub bis: i32,
    pub permits: i32,
    pub roundabouts: i32,
}

impl SheetScore {
    /// `Player::computeScore` — the three penalties are subtracted.
    pub fn total(&self) -> i32 {
        self.plans + self.parks + self.pools + self.temp + self.estates
            - self.bis
            - self.permits
            - self.roundabouts
    }
}

#[derive(Clone, Debug)]
pub struct Sheet {
    /// `EMPTY` for an empty box, otherwise the written number (0..17) or
    /// `ROUNDABOUT`.
    pub numbers: [[i32; MAX_STREET_LEN]; NUM_STREETS],
    pub is_bis: [[bool; MAX_STREET_LEN]; NUM_STREETS],
    pub written_turn: [[i32; MAX_STREET_LEN]; NUM_STREETS],
    /// The *estate* fence between boxes `j` and `j + 1`.
    pub fences: [[bool; MAX_STREET_LEN]; NUM_STREETS],
    /// Marks a house consumed by a City Plan: it cannot be reused by a later
    /// plan and no fence may be drawn through it.
    pub top_fences: [[bool; MAX_STREET_LEN]; NUM_STREETS],
    pub parks: [i32; NUM_STREETS],
    pub pools: [i32; NUM_STREETS],
    pub estate_marks: [i32; MAX_ESTATE_SIZE],
    pub temps: i32,
    pub bis_marks: i32,
    pub permits: i32,
    pub roundabouts: i32,
}

impl Sheet {
    pub fn new() -> Self {
        Sheet {
            numbers: [[EMPTY; MAX_STREET_LEN]; NUM_STREETS],
            is_bis: [[false; MAX_STREET_LEN]; NUM_STREETS],
            written_turn: [[-1; MAX_STREET_LEN]; NUM_STREETS],
            fences: [[false; MAX_STREET_LEN]; NUM_STREETS],
            top_fences: [[false; MAX_STREET_LEN]; NUM_STREETS],
            parks: [0; NUM_STREETS],
            pools: [0; NUM_STREETS],
            estate_marks: [0; MAX_ESTATE_SIZE],
            temps: 0,
            bis_marks: 0,
            permits: 0,
            roundabouts: 0,
        }
    }

    pub fn is_empty(&self, x: usize, y: usize) -> bool {
        self.numbers[x][y] == EMPTY
    }

    // ------------------------------------------------------------------
    // Writing numbers
    // ------------------------------------------------------------------
    /// `Houses::getAvailableLocations`.
    ///
    /// Numbers must increase strictly from left to right within a street. A
    /// roundabout resets the chain in both directions, so it acts as a divider
    /// rather than as a number. `None` asks for *every* empty box, which is how
    /// the engine finds roundabout sites and detects a full sheet.
    pub fn available_locations(&self, number: Option<i32>) -> Vec<Pos> {
        let mut result = Vec::new();
        for x in 0..NUM_STREETS {
            let size = STREET_SIZES[x];
            let row = &self.numbers[x];
            let mut ok = [false; MAX_STREET_LEN];

            let mut highest: i32 = -1;
            for y in 0..size {
                let n = row[y];
                if n == EMPTY {
                    ok[y] = match number {
                        None => true,
                        Some(v) => v > highest,
                    };
                } else {
                    ok[y] = false;
                    highest = if n == ROUNDABOUT { -1 } else { n };
                }
            }

            let mut lowest: i32 = 18;
            for y in (0..size).rev() {
                let n = row[y];
                if n == EMPTY {
                    ok[y] = match number {
                        None => true,
                        Some(v) => ok[y] && v < lowest,
                    };
                } else {
                    lowest = if n == ROUNDABOUT { 18 } else { n };
                }
            }

            for y in 0..size {
                if ok[y] {
                    result.push((x, y));
                }
            }
        }
        result
    }

    /// Cheap `available_locations(None)` emptiness test (end-of-game check).
    pub fn has_free_box(&self) -> bool {
        (0..NUM_STREETS).any(|x| (0..STREET_SIZES[x]).any(|y| self.is_empty(x, y)))
    }

    /// How many distinct numbers each empty box could still legally take.
    pub fn box_spans(&self) -> [[i32; MAX_STREET_LEN]; NUM_STREETS] {
        let mut spans = [[0i32; MAX_STREET_LEN]; NUM_STREETS];
        for x in 0..NUM_STREETS {
            let size = STREET_SIZES[x];
            let row = &self.numbers[x];
            for y in 0..size {
                if row[y] != EMPTY {
                    continue;
                }
                let mut low = MIN_NUMBER - 1;
                for k in (0..y).rev() {
                    if row[k] != EMPTY {
                        if row[k] != ROUNDABOUT {
                            low = row[k];
                        }
                        break;
                    }
                }
                let mut high = MAX_NUMBER + 1;
                for &value in row.iter().take(size).skip(y + 1) {
                    if value != EMPTY {
                        if value != ROUNDABOUT {
                            high = value;
                        }
                        break;
                    }
                }
                spans[x][y] = (high - low - 1).max(0);
            }
        }
        spans
    }

    /// Per street, the most houses that could still be written there.
    pub fn placement_capacity(&self) -> [i32; NUM_STREETS] {
        let mut out = [0i32; NUM_STREETS];
        for x in 0..NUM_STREETS {
            let size = STREET_SIZES[x];
            let row = &self.numbers[x];
            let mut y = 0usize;
            while y < size {
                if row[y] != EMPTY {
                    y += 1;
                    continue;
                }
                let start = y;
                while y < size && row[y] == EMPTY {
                    y += 1;
                }
                let run = (y - start) as i32;
                let mut low = MIN_NUMBER - 1;
                if start > 0 && row[start - 1] != EMPTY && row[start - 1] != ROUNDABOUT {
                    low = row[start - 1];
                }
                let mut high = MAX_NUMBER + 1;
                if y < size && row[y] != EMPTY && row[y] != ROUNDABOUT {
                    high = row[y];
                }
                out[x] += run.min((high - low - 1).max(0));
            }
        }
        out
    }

    /// `(first, last, low, high)` for the empty run containing `(x, y)`.
    fn gap_bounds(&self, x: usize, y: usize) -> Option<(usize, usize, i32, i32)> {
        let row = &self.numbers[x];
        let size = STREET_SIZES[x];
        if row[y] != EMPTY {
            return None;
        }
        let mut first = y;
        while first > 0 && row[first - 1] == EMPTY {
            first -= 1;
        }
        let mut last = y;
        while last + 1 < size && row[last + 1] == EMPTY {
            last += 1;
        }
        let mut low = MIN_NUMBER - 1;
        if first > 0 && row[first - 1] != EMPTY && row[first - 1] != ROUNDABOUT {
            low = row[first - 1];
        }
        let mut high = MAX_NUMBER + 1;
        if last + 1 < size && row[last + 1] != EMPTY && row[last + 1] != ROUNDABOUT {
            high = row[last + 1];
        }
        Some((first, last, low, high))
    }

    /// Negated distance from `number`'s ideal position; `0.0` is perfect.
    pub fn positional_fit(&self, number: i32, x: usize, y: usize) -> Option<f64> {
        let (first, last, low, high) = self.gap_bounds(x, y)?;
        if high - low <= 1 {
            return None;
        }
        let ideal = first as f64
            + (last - first) as f64 * (number - low) as f64 / (high - low) as f64;
        Some(-((y as f64) - ideal).abs())
    }

    /// `Houses::getAvailableLocationsForBis`.
    ///
    /// `(x, y, number, side)`, where `side` is 0 when the number is copied from
    /// the neighbour on the left and 1 from the right. Both sides can be legal
    /// at the same box with two different numbers, which is why `side` and not
    /// `number` is what the action codec stores.
    pub fn bis_candidates(&self) -> Vec<(usize, usize, i32, usize)> {
        let mut out = Vec::new();
        for x in 0..NUM_STREETS {
            let size = STREET_SIZES[x];
            let row = &self.numbers[x];
            let fence = &self.fences[x];
            for y in 0..size {
                if row[y] != EMPTY {
                    continue;
                }
                if y > 0 {
                    let left = row[y - 1];
                    if left != EMPTY && left != ROUNDABOUT && !fence[y - 1] {
                        out.push((x, y, left, 0));
                    }
                }
                if y < size - 1 {
                    let right = row[y + 1];
                    if right != EMPTY && right != ROUNDABOUT && !fence[y] {
                        out.push((x, y, right, 1));
                    }
                }
            }
        }
        out
    }

    /// The number a bis at `(x, y)` copied from `side` would be, else `None`.
    pub fn bis_number_at(&self, x: usize, y: usize, side: usize) -> Option<i32> {
        let row = &self.numbers[x];
        let fence = &self.fences[x];
        if row[y] != EMPTY {
            return None;
        }
        if side == 0 {
            if y == 0 || fence[y - 1] {
                return None;
            }
            let left = row[y - 1];
            return if left == EMPTY || left == ROUNDABOUT {
                None
            } else {
                Some(left)
            };
        }
        if y == STREET_SIZES[x] - 1 || fence[y] {
            return None;
        }
        let right = row[y + 1];
        if right == EMPTY || right == ROUNDABOUT {
            None
        } else {
            Some(right)
        }
    }

    /// `Houses::add` — unchecked; callers validate first.
    pub fn write(&mut self, number: i32, pos: Pos, turn: i32, is_bis: bool) {
        let (x, y) = pos;
        self.numbers[x][y] = number;
        self.is_bis[x][y] = is_bis;
        self.written_turn[x][y] = turn;
    }

    // ------------------------------------------------------------------
    // Effects
    // ------------------------------------------------------------------
    /// `Actions/Surveyor::getAvailableZones`.
    ///
    /// A fence may go in any empty fence slot, except that it may not split two
    /// houses already spent on the same City Plan, and it may not split a bis
    /// pair (two neighbours showing the same number).
    pub fn surveyor_zones(&self) -> Vec<Pos> {
        let mut out = Vec::new();
        for x in 0..NUM_STREETS {
            let row = &self.numbers[x];
            for j in 0..FENCE_SIZES[x] {
                if self.fences[x][j] {
                    continue;
                }
                if row[j] != EMPTY {
                    if self.top_fences[x][j] && self.top_fences[x][j + 1] {
                        continue;
                    }
                    if row[j + 1] != EMPTY && row[j] == row[j + 1] {
                        continue;
                    }
                }
                out.push((x, j));
            }
        }
        out
    }

    /// Streets that still have an unbuilt park (`Actions/Park`).
    pub fn park_streets(&self) -> Vec<usize> {
        (0..NUM_STREETS)
            .filter(|&x| self.parks[x] < PARK_BOXES[x])
            .collect()
    }

    /// Estate-value rows that still have a box to cross (`Actions/RealEstate`).
    pub fn estate_rows(&self) -> Vec<usize> {
        (0..MAX_ESTATE_SIZE)
            .filter(|&i| self.estate_marks[i] < ESTATE_ROW_BOXES[i])
            .collect()
    }

    /// `Actions/Pool::canBuild` — the house just written must be on a pool.
    pub fn can_build_pool_at(&self, pos: Pos) -> bool {
        POOL_POSITIONS.contains(&pos) && self.pool_count() < POOL_BOXES
    }

    pub fn pool_count(&self) -> i32 {
        self.pools.iter().sum()
    }

    pub fn can_take_permit(&self) -> bool {
        self.permits < PERMIT_BOXES
    }

    pub fn can_build_roundabout(&self) -> bool {
        self.roundabouts < ROUNDABOUT_BOXES
    }

    /// `WriteNumberTrait::buildRoundabout` — write the sentinel, fence both
    /// sides (silently skipping one that falls off the end of the street), and
    /// cross off a roundabout penalty box.
    pub fn build_roundabout(&mut self, pos: Pos, turn: i32) {
        let (x, y) = pos;
        self.write(ROUNDABOUT, pos, turn, false);
        for j in [y as isize - 1, y as isize] {
            if j >= 0 && (j as usize) < FENCE_SIZES[x] {
                self.fences[x][j as usize] = true;
            }
        }
        self.roundabouts = (self.roundabouts + 1).min(ROUNDABOUT_BOXES);
    }

    // ------------------------------------------------------------------
    // Housing estates
    // ------------------------------------------------------------------
    /// `Actions/RealEstate::getEstates` — a run of boxes bounded by fences (or
    /// the ends of the street) in which every box is built and none of them is
    /// a roundabout.
    pub fn estates(&self) -> Vec<Estate> {
        let mut out = Vec::new();
        for x in 0..NUM_STREETS {
            let size = STREET_SIZES[x];
            let row = &self.numbers[x];
            let mut start = 0usize;
            let mut full = true;
            for y in 0..size {
                let n = row[y];
                if n == EMPTY || n == ROUNDABOUT {
                    full = false;
                }
                if y == size - 1 || self.fences[x][y] {
                    if full {
                        out.push((x, start, y - start + 1));
                    }
                    full = true;
                    start = y + 1;
                }
            }
        }
        out
    }

    /// Estates no City Plan has consumed (`EstatePlan::getAvailableEstates`).
    pub fn free_estates(&self) -> Vec<Estate> {
        self.estates()
            .into_iter()
            .filter(|&(x, start, size)| (0..size).all(|k| !self.top_fences[x][start + k]))
            .collect()
    }

    /// `RealEstate::getAssocSizeNumber` — estates of size 1..6 (bigger ignored).
    pub fn estate_size_counts(&self) -> [i32; MAX_ESTATE_SIZE] {
        let mut mult = [0i32; MAX_ESTATE_SIZE];
        for (_, _, size) in self.estates() {
            if size <= MAX_ESTATE_SIZE {
                mult[size - 1] += 1;
            }
        }
        mult
    }

    pub fn mark_top_fences(&mut self, cells: &[Pos]) {
        for &(x, y) in cells {
            self.top_fences[x][y] = true;
        }
    }

    pub fn bis_count_per_street(&self) -> [i32; NUM_STREETS] {
        let mut counts = [0i32; NUM_STREETS];
        for x in 0..NUM_STREETS {
            counts[x] = (0..STREET_SIZES[x]).filter(|&y| self.is_bis[x][y]).count() as i32;
        }
        counts
    }

    pub fn has_roundabout_in_street(&self, x: usize) -> bool {
        (0..STREET_SIZES[x]).any(|y| self.numbers[x][y] == ROUNDABOUT)
    }

    /// `Actions/Pool::getCompleted` — all three pools of a street built.
    pub fn street_pools_complete(&self) -> [bool; NUM_STREETS] {
        let mut out = [false; NUM_STREETS];
        for x in 0..NUM_STREETS {
            out[x] = self.pools[x] == 3;
        }
        out
    }

    pub fn street_parks_complete(&self) -> [bool; NUM_STREETS] {
        let mut out = [false; NUM_STREETS];
        for x in 0..NUM_STREETS {
            out[x] = self.parks[x] == PARK_BOXES[x];
        }
        out
    }

    // ------------------------------------------------------------------
    // Scoring
    // ------------------------------------------------------------------
    pub fn park_score(&self) -> i32 {
        (0..NUM_STREETS)
            .map(|x| PARK_SCORES[x][self.parks[x] as usize])
            .sum()
    }

    pub fn pool_score(&self) -> i32 {
        POOL_SCORES[self.pool_count() as usize]
    }

    pub fn estate_score(&self) -> i32 {
        let mult = self.estate_size_counts();
        (0..MAX_ESTATE_SIZE)
            .map(|i| mult[i] * ESTATE_ROW_SCORES[i][self.estate_marks[i] as usize])
            .sum()
    }

    pub fn bis_penalty(&self) -> i32 {
        BIS_SCORES[self.bis_marks.min(BIS_BOXES) as usize]
    }

    pub fn permit_penalty(&self) -> i32 {
        PERMIT_SCORES[self.permits.min(PERMIT_BOXES) as usize]
    }

    pub fn roundabout_penalty(&self) -> i32 {
        ROUNDABOUT_SCORES[self.roundabouts.min(ROUNDABOUT_BOXES) as usize]
    }

    /// Everything except City Plans and the temp-agency rank.
    pub fn local_score(&self) -> SheetScore {
        SheetScore {
            parks: self.park_score(),
            pools: self.pool_score(),
            estates: self.estate_score(),
            plans: 0,
            temp: 0,
            bis: self.bis_penalty(),
            permits: self.permit_penalty(),
            roundabouts: self.roundabout_penalty(),
        }
    }

    /// `EndOfGameTrait::stComputeScores` — ties go to the most completed
    /// estates, then the most size-1 estates, then size-2, and so on. The
    /// larger tuple wins, so this is compared lexicographically.
    pub fn tiebreak_key(&self) -> [i32; MAX_ESTATE_SIZE + 1] {
        let estates = self.estates();
        let mut key = [0i32; MAX_ESTATE_SIZE + 1];
        key[0] = estates.len() as i32;
        for (_, _, size) in estates {
            if size >= 1 && size <= MAX_ESTATE_SIZE {
                key[size] += 1;
            }
        }
        key
    }
}

/// The bounds a number must respect are the same in both directions, so the
/// only sanity check worth having here is that the table constants line up.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_sheet_takes_any_number_anywhere() {
        let sheet = Sheet::new();
        assert_eq!(sheet.available_locations(None).len(), 33);
        assert_eq!(sheet.available_locations(Some(7)).len(), 33);
        assert!(sheet.has_free_box());
        assert_eq!(sheet.local_score().total(), 0);
        assert!(crate::constants::MIN_NUMBER == 0 && crate::constants::MAX_NUMBER == 17);
    }

    #[test]
    fn ascending_rule_bounds_a_written_street() {
        let mut sheet = Sheet::new();
        sheet.write(7, (0, 4), 1, false);
        let spots = sheet.available_locations(Some(7));
        // 7 is written at box 4, so no box may take another 7 in street 0.
        assert!(!spots.iter().any(|&(x, _)| x == 0));
        let low = sheet.available_locations(Some(3));
        assert!(low.contains(&(0, 0)) && !low.contains(&(0, 5)));
    }

    #[test]
    fn a_roundabout_divides_a_street() {
        let mut sheet = Sheet::new();
        sheet.build_roundabout((1, 5), 3);
        assert!(sheet.fences[1][4] && sheet.fences[1][5]);
        assert_eq!(sheet.roundabouts, 1);
        assert!(sheet.has_roundabout_in_street(1));
        // The chain resets, so a low number is legal again on the right.
        let spots = sheet.available_locations(Some(1));
        assert!(spots.contains(&(1, 6)));
    }
}
