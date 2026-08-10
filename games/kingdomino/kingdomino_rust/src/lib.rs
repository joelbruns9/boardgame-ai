use numpy::ndarray::{Array1, Array2, Array3, Array4, Axis};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use rand_distr::{Distribution, Gamma};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

mod deck0_draft_dp;
mod denial_tree;
mod nnue_features;
mod search;
mod sparse_nnue;

// Terrain constants matching Python's Terrain IntEnum
const EMPTY: u8 = 0;
const CASTLE: u8 = 1;
// WHEAT=2, FOREST=3, WATER=4, GRASS=5, SWAMP=6, MINE=7

// The four cardinal directions (dx, dy)
const DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

// Canvas is always 15; max board index = 14.
// Flat array index: idx(x, y) = y * 15 + x
const N: usize = 15;
const CELLS: usize = N * N; // 225

#[inline(always)]
fn idx(x: i8, y: i8) -> usize {
    y as usize * N + x as usize
}

#[inline(always)]
fn in_bounds(x: i8, y: i8) -> bool {
    (x as u8) < N as u8 && (y as u8) < N as u8
}

/// Kingdomino board implemented as flat arrays — no dict/set, no hashing.
/// All hot-path operations (is_empty, half_connects, is_legal_placement,
/// legal_placements) are plain array reads, which is why this is fast.
#[pyclass]
struct RustBoard {
    terrain: [u8; CELLS], // 0=empty 1=castle 2-7=terrain types
    crowns: [u8; CELLS],  // 0-3
    castle_x: i8,
    castle_y: i8,
    min_x: i8,
    max_x: i8,
    min_y: i8,
    max_y: i8,
    occupied: u8, // count of occupied cells (for harmony check)
}

#[pymethods]
impl RustBoard {
    /// Create a new board. castle_x and castle_y default to 7 (canvas centre).
    #[new]
    #[pyo3(signature = (castle_x=7, castle_y=7))]
    fn new(castle_x: i8, castle_y: i8) -> Self {
        let mut terrain = [EMPTY; CELLS];
        terrain[idx(castle_x, castle_y)] = CASTLE;
        RustBoard {
            terrain,
            crowns: [0u8; CELLS],
            castle_x,
            castle_y,
            min_x: castle_x,
            max_x: castle_x,
            min_y: castle_y,
            max_y: castle_y,
            occupied: 1,
        }
    }

    fn copy(&self) -> Self {
        RustBoard {
            terrain: self.terrain,
            crowns: self.crowns,
            castle_x: self.castle_x,
            castle_y: self.castle_y,
            min_x: self.min_x,
            max_x: self.max_x,
            min_y: self.min_y,
            max_y: self.max_y,
            occupied: self.occupied,
        }
    }

    #[staticmethod]
    fn from_flat_arrays(
        terrain_vec: Vec<u8>,
        crowns_vec: Vec<u8>,
        castle_x: i8,
        castle_y: i8,
    ) -> PyResult<Self> {
        RustBoard::from_flat_parts(terrain_vec, crowns_vec, castle_x, castle_y)
    }

    fn is_empty(&self, x: i8, y: i8) -> bool {
        in_bounds(x, y) && self.terrain[idx(x, y)] == EMPTY
    }

    /// Returns true if the half-tile at (x,y) with `terrain` connects to an
    /// adjacent occupied cell (castle or same terrain).
    fn half_connects(&self, x: i8, y: i8, terrain: u8) -> bool {
        for (dx, dy) in DIRS {
            let nx = x + dx;
            let ny = y + dy;
            if in_bounds(nx, ny) {
                let t = self.terrain[idx(nx, ny)];
                if t == CASTLE || t == terrain {
                    return true;
                }
            }
        }
        false
    }

    /// Returns true if placing domino (t1/c1 at (x1,y1), t2/c2 at (x2,y2),
    /// flipped flag) is legal. The domino halves are passed as raw terrain ints
    /// and crown counts — no Python object crossing required in the hot path.
    ///
    /// Arguments:
    ///   t_a, c_a: terrain/crowns of domino half A
    ///   t_b, c_b: terrain/crowns of domino half B
    ///   x1, y1, x2, y2: cell coordinates
    ///   flipped: if true, half B goes to (x1,y1) and half A to (x2,y2)
    fn is_legal_placement(
        &self,
        t_a: u8,
        _c_a: u8,
        t_b: u8,
        _c_b: u8,
        x1: i8,
        y1: i8,
        x2: i8,
        y2: i8,
        flipped: bool,
    ) -> bool {
        // Cells must be adjacent
        let dx = (x1 - x2).abs();
        let dy = (y1 - y2).abs();
        if dx + dy != 1 {
            return false;
        }
        // Both cells must be in bounds and empty
        if !in_bounds(x1, y1) || self.terrain[idx(x1, y1)] != EMPTY {
            return false;
        }
        if !in_bounds(x2, y2) || self.terrain[idx(x2, y2)] != EMPTY {
            return false;
        }
        // Bounding box must stay within 7×7 after adding both cells
        let mnx = self.min_x.min(x1).min(x2);
        let mxx = self.max_x.max(x1).max(x2);
        let mny = self.min_y.min(y1).min(y2);
        let mxy = self.max_y.max(y1).max(y2);
        if mxx - mnx >= 7 || mxy - mny >= 7 {
            return false;
        }
        // At least one half must connect
        let (t_h1, t_h2) = if flipped { (t_b, t_a) } else { (t_a, t_b) };
        self.half_connects(x1, y1, t_h1) || self.half_connects(x2, y2, t_h2)
    }

    /// Generate all legal, physically-distinct placements of a domino.
    ///
    /// Returns a list of (x1, y1, x2, y2, flipped) tuples — one per distinct
    /// placement. The Python wrapper converts these to Placement objects.
    ///
    /// Arguments:
    ///   t_a, c_a: terrain/crowns of half A
    ///   t_b, c_b: terrain/crowns of half B
    fn legal_placements(&self, t_a: u8, c_a: u8, t_b: u8, c_b: u8) -> Vec<(i8, i8, i8, i8, bool)> {
        // Collect frontier: empty in-bounds cells adjacent to any occupied cell.
        // We scan occupied cells directly from the terrain array within the
        // bounding box (plus one cell of padding for adjacency).
        let x0 = (self.min_x - 1).max(0);
        let x1 = (self.max_x + 2).min(N as i8);
        let y0 = (self.min_y - 1).max(0);
        let y1 = (self.max_y + 2).min(N as i8);

        // Build frontier as a small Vec of (x, y) — bounded by canvas size.
        let mut frontier: Vec<(i8, i8)> = Vec::with_capacity(64);
        for oy in y0..y1 {
            for ox in x0..x1 {
                if self.terrain[idx(ox, oy)] == EMPTY {
                    continue;
                }
                // This cell is occupied; check its empty neighbours.
                for (dx, dy) in DIRS {
                    let nx = ox + dx;
                    let ny = oy + dy;
                    if in_bounds(nx, ny) && self.terrain[idx(nx, ny)] == EMPTY {
                        // Add to frontier if not already there.
                        if !frontier.contains(&(nx, ny)) {
                            frontier.push((nx, ny));
                        }
                    }
                }
            }
        }

        // De-duplication key: ((x1,y1,t1,c1), (x2,y2,t2,c2)) sorted so
        // symmetric placements collapse. We use a small Vec of seen keys
        // (frontier is small so linear scan is fast and avoids hash overhead).
        let mut seen: Vec<((i8, i8, u8, u8), (i8, i8, u8, u8))> = Vec::with_capacity(64);
        let mut moves: Vec<(i8, i8, i8, i8, bool)> = Vec::with_capacity(32);

        for (fx, fy) in &frontier {
            let fx = *fx;
            let fy = *fy;
            for (dx, dy) in DIRS {
                let gx = fx + dx;
                let gy = fy + dy;
                // Second cell must be empty (not just in frontier).
                if !self.is_empty(gx, gy) {
                    continue;
                }
                for flipped in [false, true] {
                    if !self.is_legal_placement(t_a, c_a, t_b, c_b, fx, fy, gx, gy, flipped) {
                        continue;
                    }
                    let (t_h1, c_h1, t_h2, c_h2) = if flipped {
                        (t_b, c_b, t_a, c_a)
                    } else {
                        (t_a, c_a, t_b, c_b)
                    };
                    let k1 = (fx, fy, t_h1, c_h1);
                    let k2 = (gx, gy, t_h2, c_h2);
                    let key = if k1 <= k2 { (k1, k2) } else { (k2, k1) };
                    if seen.contains(&key) {
                        continue;
                    }
                    seen.push(key);
                    moves.push((fx, fy, gx, gy, flipped));
                }
            }
        }
        moves
    }

    /// Place a domino. Raises ValueError if the placement is illegal.
    fn place(
        &mut self,
        t_a: u8,
        c_a: u8,
        t_b: u8,
        c_b: u8,
        x1: i8,
        y1: i8,
        x2: i8,
        y2: i8,
        flipped: bool,
    ) -> PyResult<()> {
        if !self.is_legal_placement(t_a, c_a, t_b, c_b, x1, y1, x2, y2, flipped) {
            return Err(PyValueError::new_err("Illegal placement"));
        }
        let (t_h1, c_h1, t_h2, c_h2) = if flipped {
            (t_b, c_b, t_a, c_a)
        } else {
            (t_a, c_a, t_b, c_b)
        };
        for ((x, y), (t, c)) in [((x1, y1), (t_h1, c_h1)), ((x2, y2), (t_h2, c_h2))] {
            let i = idx(x, y);
            self.terrain[i] = t;
            self.crowns[i] = c;
            self.occupied += 1;
            if x < self.min_x {
                self.min_x = x;
            }
            if x > self.max_x {
                self.max_x = x;
            }
            if y < self.min_y {
                self.min_y = y;
            }
            if y > self.max_y {
                self.max_y = y;
            }
        }
        Ok(())
    }

    /// Score the board. Returns (territory_score, harmony_bonus, middle_kingdom_bonus).
    #[pyo3(signature = (harmony=true, middle_kingdom=true))]
    fn score(&self, harmony: bool, middle_kingdom: bool) -> (i32, i32, i32) {
        let mut visited = [false; CELLS];
        let mut territory_score: i32 = 0;

        for sy in self.min_y..=self.max_y {
            for sx in self.min_x..=self.max_x {
                let si = idx(sx, sy);
                let t = self.terrain[si];
                if visited[si] || t == EMPTY || t == CASTLE {
                    continue;
                }
                // BFS flood fill
                let mut stack: Vec<(i8, i8)> = Vec::with_capacity(49);
                stack.push((sx, sy));
                visited[si] = true;
                let mut area: i32 = 0;
                let mut crowns: i32 = 0;
                while let Some((cx, cy)) = stack.pop() {
                    area += 1;
                    crowns += self.crowns[idx(cx, cy)] as i32;
                    for (dx, dy) in DIRS {
                        let nx = cx + dx;
                        let ny = cy + dy;
                        if in_bounds(nx, ny) {
                            let ni = idx(nx, ny);
                            if !visited[ni] && self.terrain[ni] == t {
                                visited[ni] = true;
                                stack.push((nx, ny));
                            }
                        }
                    }
                }
                territory_score += area * crowns;
            }
        }

        let harmony_bonus = if harmony {
            let w = (self.max_x - self.min_x + 1) as i32;
            let h = (self.max_y - self.min_y + 1) as i32;
            if w == 7 && h == 7 && self.occupied == 49 {
                5
            } else {
                0
            }
        } else {
            0
        };

        let middle_bonus = if middle_kingdom {
            let w = (self.max_x - self.min_x + 1) as i32;
            let h = (self.max_y - self.min_y + 1) as i32;
            if w == 7
                && h == 7
                && self.castle_x == self.min_x + 3
                && self.castle_y == self.min_y + 3
            {
                10
            } else {
                0
            }
        } else {
            0
        };

        (territory_score, harmony_bonus, middle_bonus)
    }

    /// Read terrain at (x, y). Returns 0 (EMPTY) if out of bounds.
    fn get_terrain(&self, x: i8, y: i8) -> u8 {
        if in_bounds(x, y) {
            self.terrain[idx(x, y)]
        } else {
            EMPTY
        }
    }

    /// Read crowns at (x, y). Returns 0 if out of bounds.
    fn get_crowns(&self, x: i8, y: i8) -> u8 {
        if in_bounds(x, y) {
            self.crowns[idx(x, y)]
        } else {
            0
        }
    }

    /// Bounding box as (min_x, min_y, max_x, max_y).
    fn bbox(&self) -> (i8, i8, i8, i8) {
        (self.min_x, self.min_y, self.max_x, self.max_y)
    }

    fn castle_pos(&self) -> (i8, i8) {
        (self.castle_x, self.castle_y)
    }
}

impl RustBoard {
    fn from_flat_parts(
        terrain_vec: Vec<u8>,
        crowns_vec: Vec<u8>,
        castle_x: i8,
        castle_y: i8,
    ) -> PyResult<Self> {
        if terrain_vec.len() != CELLS || crowns_vec.len() != CELLS {
            return Err(PyValueError::new_err(format!(
                "RustBoard::from_flat_parts expected {} terrain/crown cells, got {}/{}",
                CELLS,
                terrain_vec.len(),
                crowns_vec.len()
            )));
        }
        if !in_bounds(castle_x, castle_y) {
            return Err(PyValueError::new_err("castle position outside board"));
        }

        let mut terrain = [EMPTY; CELLS];
        let mut crowns = [0u8; CELLS];
        let mut occupied: u8 = 0;
        let mut min_x = castle_x;
        let mut max_x = castle_x;
        let mut min_y = castle_y;
        let mut max_y = castle_y;

        for y in 0..N {
            for x in 0..N {
                let i = y * N + x;
                let t = terrain_vec[i];
                let c = crowns_vec[i];
                if t > 7 {
                    return Err(PyValueError::new_err(format!(
                        "terrain cell {i} has invalid terrain {t}"
                    )));
                }
                terrain[i] = t;
                crowns[i] = c;
                if t != EMPTY {
                    occupied = occupied.saturating_add(1);
                    let xi = x as i8;
                    let yi = y as i8;
                    min_x = min_x.min(xi);
                    max_x = max_x.max(xi);
                    min_y = min_y.min(yi);
                    max_y = max_y.max(yi);
                }
            }
        }

        let castle_i = idx(castle_x, castle_y);
        if terrain[castle_i] != CASTLE {
            return Err(PyValueError::new_err(format!(
                "castle cell ({castle_x},{castle_y}) has terrain {}, expected CASTLE",
                terrain[castle_i]
            )));
        }
        if occupied == 0 {
            return Err(PyValueError::new_err("board has no occupied cells"));
        }

        Ok(RustBoard {
            terrain,
            crowns,
            castle_x,
            castle_y,
            min_x,
            max_x,
            min_y,
            max_y,
            occupied,
        })
    }
}

// ─── Domino table ──────────────────────────────────────────────────────────
// Mirrors games/kingdomino/dominoes.py _RAW_DOMINOES, indexed by id-1.
// Each entry is (terrain_a, crowns_a, terrain_b, crowns_b).
// Terrain ints: WHEAT=2 FOREST=3 WATER=4 GRASS=5 SWAMP=6 MINE=7.
// This is fixed game data; test_rust_game_equiv verifies it against Python.
const DOMS: [(u8, u8, u8, u8); 48] = [
    (2, 0, 2, 0), // 1  WHEAT  WHEAT
    (2, 0, 2, 0), // 2  WHEAT  WHEAT
    (3, 0, 3, 0), // 3  FOREST FOREST
    (3, 0, 3, 0), // 4  FOREST FOREST
    (3, 0, 3, 0), // 5  FOREST FOREST
    (3, 0, 3, 0), // 6  FOREST FOREST
    (4, 0, 4, 0), // 7  WATER  WATER
    (4, 0, 4, 0), // 8  WATER  WATER
    (4, 0, 4, 0), // 9  WATER  WATER
    (5, 0, 5, 0), // 10 GRASS  GRASS
    (5, 0, 5, 0), // 11 GRASS  GRASS
    (6, 0, 6, 0), // 12 SWAMP  SWAMP
    (2, 0, 3, 0), // 13 WHEAT  FOREST
    (2, 0, 4, 0), // 14 WHEAT  WATER
    (2, 0, 5, 0), // 15 WHEAT  GRASS
    (2, 0, 6, 0), // 16 WHEAT  SWAMP
    (3, 0, 4, 0), // 17 FOREST WATER
    (3, 0, 5, 0), // 18 FOREST GRASS
    (2, 1, 3, 0), // 19 WHEAT+1 FOREST
    (2, 1, 4, 0), // 20 WHEAT+1 WATER
    (2, 1, 5, 0), // 21 WHEAT+1 GRASS
    (2, 1, 6, 0), // 22 WHEAT+1 SWAMP
    (2, 1, 7, 0), // 23 WHEAT+1 MINE
    (3, 1, 2, 0), // 24 FOREST+1 WHEAT
    (3, 1, 2, 0), // 25 FOREST+1 WHEAT
    (3, 1, 2, 0), // 26 FOREST+1 WHEAT
    (3, 1, 2, 0), // 27 FOREST+1 WHEAT
    (3, 1, 4, 0), // 28 FOREST+1 WATER
    (3, 1, 5, 0), // 29 FOREST+1 GRASS
    (4, 1, 2, 0), // 30 WATER+1 WHEAT
    (4, 1, 2, 0), // 31 WATER+1 WHEAT
    (4, 1, 3, 0), // 32 WATER+1 FOREST
    (4, 1, 3, 0), // 33 WATER+1 FOREST
    (4, 1, 3, 0), // 34 WATER+1 FOREST
    (4, 1, 3, 0), // 35 WATER+1 FOREST
    (2, 0, 5, 1), // 36 WHEAT  GRASS+1
    (4, 0, 5, 1), // 37 WATER  GRASS+1
    (2, 0, 6, 1), // 38 WHEAT  SWAMP+1
    (5, 0, 6, 1), // 39 GRASS  SWAMP+1
    (7, 1, 2, 0), // 40 MINE+1 WHEAT
    (2, 0, 5, 2), // 41 WHEAT  GRASS+2
    (4, 0, 5, 2), // 42 WATER  GRASS+2
    (2, 0, 6, 2), // 43 WHEAT  SWAMP+2
    (5, 0, 6, 2), // 44 GRASS  SWAMP+2
    (7, 2, 2, 0), // 45 MINE+2 WHEAT
    (6, 0, 7, 2), // 46 SWAMP  MINE+2
    (6, 0, 7, 2), // 47 SWAMP  MINE+2
    (2, 0, 7, 3), // 48 WHEAT  MINE+3
];

#[inline(always)]
fn dom(id: u16) -> (u8, u8, u8, u8) {
    DOMS[(id - 1) as usize]
}

// Phase codes — match Python's Phase IntEnum exactly.
const INITIAL_SELECTION: u8 = 0;
const PLACE_AND_SELECT: u8 = 1;
const FINAL_PLACEMENT: u8 = 2;
const GAME_OVER: u8 = 3;

// ─── Encoder ────────────────────────────────────────────────────────────────
// Mirrors games/kingdomino/encoder.py exactly.  Output is bit-for-bit identical
// to encode_state: all `int/int` divisions are done in f64 then cast to f32, to
// match numpy's float64→float32 array-assignment cast.
//
// Output canvas is 13×13 (castle-centred), distinct from the board's 15-canvas.
const OUT_N: usize = 13; // castle-centred canvas side
const OUT_CENTER: i8 = 6; // CASTLE_CENTER
const N_BOARD_CH: usize = 9;
const CH_CROWNS: usize = 6;
const CH_CASTLE: usize = 7;
const CH_OCCUPIED: usize = 8;

const TILE_FEAT: usize = 14; // a-terrain(6)+a-crowns(1)+b-terrain(6)+b-crowns(1)
const ROW_SLOT: usize = 15; // TILE_FEAT + present flag
const CLAIM_SLOT: usize = 16; // TILE_FEAT + is_mine + status
const PENDING_SUMMARY: usize = 18; // TILE_FEAT + present + turn_distance + active + remaining_count
const BOARD_SUMMARY: usize = 25;
const FLAT_SIZE: usize = 333;
const SCORE_SCALE: f32 = 160.0; // must match encoder.SCORE_SCALE / training score_scale
const MAX_BOARD_CELLS: f32 = 48.0;
const MAX_TOTAL_CROWNS: f32 = 24.0;
const MAX_LEGAL_PLACEMENTS: f32 = 64.0;

// Flat-vector field offsets (see encoder.FLAT_LAYOUT).
const OFF_MY_NEXT_PENDING: usize = 0;
const OFF_OPP_NEXT_PENDING: usize = 18;
const OFF_MY_BOARD_SUMMARY: usize = 36;
const OFF_OPP_BOARD_SUMMARY: usize = 61;
// Local offset of `width` within a board-summary block; `height` is the next
// slot (see write_board_summary). These two exchange under D4 rotations by an
// odd number of quarter-turns — see transform_flat.
const BS_WIDTH_LOCAL: usize = 20;
const OFF_CURRENT_ROW: usize = 86;
const OFF_PENDING: usize = 146;
const OFF_NEXT: usize = 210;
const OFF_BAG: usize = 274;
const OFF_PHASE: usize = 322;
const OFF_GAME_PROGRESS: usize = 325;
const OFF_MY_FILL: usize = 326;
const OFF_OPP_FILL: usize = 327;
const OFF_ACTOR_FLAG: usize = 328;
// Pick position features (4 scalars).
// Replaces OFF_MY_PICK_RANK / OFF_OPP_PICK_RANK (2 scalars).
const OFF_PICK_POS_0: usize = 329;
const OFF_PICK_POS_1: usize = 330;
const OFF_PICK_POS_2: usize = 331;
const OFF_PICK_POS_3: usize = 332;

/// Write a domino's 14-float tile features at `off`.  Layout:
/// [a-terrain one-hot(6), a-crowns/3, b-terrain one-hot(6), b-crowns/3].
#[inline]
fn write_tile(buf: &mut [f32], off: usize, domino_id: u16) {
    let (ta, ca, tb, cb) = dom(domino_id);
    buf[off + (ta - 2) as usize] = 1.0; // a terrain (WHEAT=2 → index 0)
    buf[off + 6] = (ca as f64 / 3.0) as f32; // a crowns / MAX_CROWNS
    buf[off + 7 + (tb - 2) as usize] = 1.0; // b terrain one-hot
    buf[off + 13] = (cb as f64 / 3.0) as f32; // b crowns / MAX_CROWNS
}

/// Row slot (15): tile features + present flag.  Empty slot stays zero.
#[inline]
fn write_row_slot(buf: &mut [f32], off: usize, domino: Option<u16>) {
    if let Some(d) = domino {
        write_tile(buf, off, d);
        buf[off + TILE_FEAT] = 1.0;
    }
}

/// Claim slot (16): tile features + is_mine flag + status flag.  Empty stays 0.
/// `player` is the perspective player (is_mine = claim.player == player).
#[inline]
fn write_claim_slot(
    buf: &mut [f32],
    off: usize,
    claim: Option<(u8, u16)>,
    player: u8,
    status: f32,
) {
    if let Some((cp, did)) = claim {
        write_tile(buf, off, did);
        buf[off + TILE_FEAT] = if cp == player { 1.0 } else { 0.0 };
        buf[off + TILE_FEAT + 1] = status;
    }
}

/// Pending summary (18): tile + present + turn_distance + active + remaining.
#[inline]
fn write_pending_summary(buf: &mut [f32], off: usize, summary: Option<(u16, usize, usize)>) {
    debug_assert!(off + PENDING_SUMMARY <= buf.len());
    if let Some((did, distance, remaining)) = summary {
        write_tile(buf, off, did);
        buf[off + TILE_FEAT] = 1.0;
        buf[off + TILE_FEAT + 1] = (distance.min(3) as f64 / 3.0) as f32;
        buf[off + TILE_FEAT + 2] = if distance == 0 { 1.0 } else { 0.0 };
        buf[off + TILE_FEAT + 3] = (remaining.min(2) as f64 / 2.0) as f32;
    }
}

/// Encode one board into a pre-allocated (9*13*13) flat slice, castle pinned to
/// the centre.  Channels: 0..5 terrain one-hot, 6 crowns/3, 7 castle, 8 occupied.
/// Zeroes `dst` first, so the caller need not pre-zero it.
fn encode_board_spatial_into(board: &RustBoard, dst: &mut [f32]) {
    dst.fill(0.0);
    let at = |c: usize, y: usize, x: usize| c * OUT_N * OUT_N + y * OUT_N + x;

    let (cx, cy) = (board.castle_x, board.castle_y);
    // Castle anchor — always the output centre.
    dst[at(CH_CASTLE, OUT_CENTER as usize, OUT_CENTER as usize)] = 1.0;
    dst[at(CH_OCCUPIED, OUT_CENTER as usize, OUT_CENTER as usize)] = 1.0;

    // Scan the occupied bounding box; any non-empty, non-castle cell is a placed
    // half (equivalent to Python's occupied_cells() minus the castle).
    for by in board.min_y..=board.max_y {
        for bx in board.min_x..=board.max_x {
            let i = by as usize * 15 + bx as usize;
            let t = board.terrain[i];
            if t == EMPTY || t == CASTLE {
                continue;
            }
            let out_x = bx - cx + OUT_CENTER;
            let out_y = by - cy + OUT_CENTER;
            if out_x < 0 || out_x >= OUT_N as i8 || out_y < 0 || out_y >= OUT_N as i8 {
                continue; // defensive; never happens on a legal board
            }
            let (ox, oy) = (out_x as usize, out_y as usize);
            let cr = board.crowns[i];
            dst[at((t - 2) as usize, oy, ox)] = 1.0; // terrain one-hot
            dst[at(CH_CROWNS, oy, ox)] = (cr as f64 / 3.0) as f32;
            dst[at(CH_OCCUPIED, oy, ox)] = 1.0;
        }
    }
}

/// Compactness: occupied cells (incl. castle) / bbox area.  Board is never
/// empty (castle), so the bbox is always valid and area ≥ 1.
fn fill_ratio(board: &RustBoard) -> f32 {
    let occ = board.occupied as i32;
    let w = (board.max_x - board.min_x + 1) as i32;
    let h = (board.max_y - board.min_y + 1) as i32;
    let area = w * h;
    if area == 0 {
        return 0.0;
    }
    (occ as f64 / area as f64) as f32
}

// ─── Action codec ────────────────────────────────────────────────────────────
// Mirrors games/kingdomino/action_codec.py.  Joint index = placement_idx *
// PICK_AXIS_SIZE + pick_idx over a 3390-action space.  Spatial placement index =
// direction * 169 + out_y * 13 + out_x in the 13×13 castle-centred frame.
fn board_component_facts(board: &RustBoard) -> ([f32; 6], [f32; 6], i32) {
    let mut visited = [false; CELLS];
    let mut score_by = [0.0f32; 6];
    let mut largest_by = [0.0f32; 6];
    let mut total_crowns: i32 = 0;

    for sy in board.min_y..=board.max_y {
        for sx in board.min_x..=board.max_x {
            let si = idx(sx, sy);
            let t = board.terrain[si];
            if visited[si] || t == EMPTY || t == CASTLE {
                continue;
            }
            let terrain_idx = (t - 2) as usize;
            let mut stack: Vec<(i8, i8)> = Vec::with_capacity(49);
            stack.push((sx, sy));
            visited[si] = true;
            let mut area: i32 = 0;
            let mut crowns: i32 = 0;
            while let Some((cx, cy)) = stack.pop() {
                area += 1;
                crowns += board.crowns[idx(cx, cy)] as i32;
                for (dx, dy) in DIRS {
                    let nx = cx + dx;
                    let ny = cy + dy;
                    if in_bounds(nx, ny) {
                        let ni = idx(nx, ny);
                        if !visited[ni] && board.terrain[ni] == t {
                            visited[ni] = true;
                            stack.push((nx, ny));
                        }
                    }
                }
            }
            score_by[terrain_idx] += (area * crowns) as f32;
            largest_by[terrain_idx] = largest_by[terrain_idx].max(area as f32);
            total_crowns += crowns;
        }
    }

    (score_by, largest_by, total_crowns)
}

/// Factual bonus states for harmony and middle kingdom (mirrors encoder.py
/// _bonus_state_features).  Layout per bonus: [awarded, still_possible,
/// impossible].  Harmony needs a full 7×7 (occupied == 49 → all 24 placed →
/// zero discards), so it is impossible the instant this player discards.
/// Middle kingdom needs a castle-centred 7×7 bbox (not a full fill), so it is
/// impossible once the bbox extends outside the castle-centred 7×7 target.
fn bonus_state_features(
    state: &RustGameState,
    board: &RustBoard,
    owner: u8,
) -> ([f32; 3], [f32; 3]) {
    let width = (board.max_x - board.min_x + 1) as i32;
    let height = (board.max_y - board.min_y + 1) as i32;
    let occupied = board.occupied as i32;

    let mut harmony = [0.0f32; 3];
    if state.harmony {
        let awarded = width == 7 && height == 7 && occupied == 49;
        let impossible = state.discards[owner as usize] > 0;
        if awarded {
            harmony[0] = 1.0;
        } else if impossible {
            harmony[2] = 1.0;
        } else {
            harmony[1] = 1.0;
        }
    }

    let mut middle = [0.0f32; 3];
    if state.middle_kingdom {
        let awarded = width == 7
            && height == 7
            && board.castle_x == board.min_x + 3
            && board.castle_y == board.min_y + 3;
        let outside_target = board.min_x < board.castle_x - 3
            || board.max_x > board.castle_x + 3
            || board.min_y < board.castle_y - 3
            || board.max_y > board.castle_y + 3;
        if awarded {
            middle[0] = 1.0;
        } else if outside_target {
            middle[2] = 1.0;
        } else {
            middle[1] = 1.0;
        }
    }

    (harmony, middle)
}

fn write_board_summary(buf: &mut [f32], off: usize, state: &RustGameState, player: u8) {
    debug_assert!(off + BOARD_SUMMARY <= buf.len());
    let board = &state.boards[player as usize];
    let (territory, harmony_bonus, middle_bonus) = board.score(state.harmony, state.middle_kingdom);
    let total_score = territory + harmony_bonus + middle_bonus;
    let (score_by, largest_by, total_crowns) = board_component_facts(board);
    let (harmony, middle) = bonus_state_features(state, board, player);
    let width = (board.max_x - board.min_x + 1) as f32;
    let height = (board.max_y - board.min_y + 1) as f32;
    let empty_remaining = (49i32 - board.occupied as i32).max(0) as f32;
    let next_summary = state.next_pending_summary(player);
    let mut legal_count = 0usize;
    if let Some((did, _, _)) = next_summary {
        let (ta, ca, tb, cb) = dom(did);
        legal_count = board.legal_placements(ta, ca, tb, cb).len();
    }

    let mut i = off;
    buf[i] = (total_score as f32).min(SCORE_SCALE) / SCORE_SCALE;
    i += 1;
    for v in score_by {
        buf[i] = v.min(SCORE_SCALE) / SCORE_SCALE;
        i += 1;
    }
    for v in largest_by {
        buf[i] = v.min(MAX_BOARD_CELLS) / MAX_BOARD_CELLS;
        i += 1;
    }
    buf[i] = (total_crowns as f32).min(MAX_TOTAL_CROWNS) / MAX_TOTAL_CROWNS;
    i += 1;
    for v in harmony {
        buf[i] = v;
        i += 1;
    }
    for v in middle {
        buf[i] = v;
        i += 1;
    }
    buf[i] = width.min(7.0) / 7.0;
    i += 1;
    buf[i] = height.min(7.0) / 7.0;
    i += 1;
    buf[i] = empty_remaining.min(MAX_BOARD_CELLS) / MAX_BOARD_CELLS;
    i += 1;
    buf[i] = (legal_count as f32).min(MAX_LEGAL_PLACEMENTS) / MAX_LEGAL_PLACEMENTS;
    i += 1;
    buf[i] = if legal_count == 0 && next_summary.is_some() {
        1.0
    } else {
        0.0
    };
}

const CODEC_CELLS: u16 = 169; // 13×13 castle-centred cells (NUM_CELLS)
const DISCARD_PLACEMENT_IDX: u16 = 676; // = NUM_SPATIAL_PLACEMENTS
const NO_PLACEMENT_IDX: u16 = 677;
const NO_PICK_IDX: u16 = 4; // = NUM_PICK_SLOTS
const PICK_AXIS_SIZE: u16 = 5;
// B-half offset from the A-half anchor, indexed by codec direction
// (0:right, 1:down, 2:left, 3:up).  NOTE: this is action_codec._DIRECTION_DELTAS
// order, deliberately distinct from board::DIRS — do not unify them.
const CODEC_DIRS: [(i8, i8); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

/// A 2-player Mighty-Duel Kingdomino game state, mirroring games/kingdomino/
/// game.py::GameState.  `step` is functional (returns a fresh state, leaving
/// the receiver untouched) so the MCTS can lazily set child states the same
/// way the Python search does.
///
/// A claim is stored as (player, domino_id), mirroring the Python Claim.
/// History is intentionally not tracked — the engine never reads it, and the
/// search/encoder don't need it.
#[pyclass]
struct RustGameState {
    boards: [RustBoard; 2],
    deck: Vec<u16>,
    current_row: Vec<u16>,
    pending_claims: Vec<(u8, u16)>,
    next_claims: Vec<(u8, u16)>,
    phase: u8,
    actor_index: usize,
    initial_pick_count: usize,
    start_player: u8,
    harmony: bool,
    middle_kingdom: bool,
    // Per-player forced-discard count (mirrors game.py GameState.discards).  A
    // discard permanently forfeits Harmony for that player (needs a full 7×7).
    discards: [u32; 2],
}

/// Compute next-round pick position features. Mirrors encoder.py
/// _pick_positions(). Returns [pos0, pos1, pos2, pos3] where:
///   +1.0 = encoded player acts at this position
///   -1.0 = opponent acts
///    0.0 = not yet committed, or no next round
///
/// INITIAL_SELECTION, FINAL_PLACEMENT, GAME_OVER → all 0.0.
/// Sorted by domino_id ascending (lower = earlier pick position).
fn pick_positions(state: &RustGameState, player: u8) -> [f32; 4] {
    if state.phase == INITIAL_SELECTION
        || state.phase == FINAL_PLACEMENT
        || state.phase == GAME_OVER
    {
        return [0.0; 4];
    }

    // Collect and sort next_claims by domino_id ascending.
    // next_claims items are (player, domino_id).
    let mut committed: Vec<(u16, u8)> =
        state.next_claims.iter().map(|&(p, did)| (did, p)).collect();
    committed.sort_by_key(|&(did, _)| did);

    let mut out = [0.0f32; 4];
    for (k, &(_, p)) in committed.iter().enumerate() {
        if k >= 4 {
            break;
        }
        out[k] = if p == player { 1.0 } else { -1.0 };
    }
    out
}

impl RustGameState {
    /// Deep copy — boards are cloned via RustBoard::copy, Vecs via clone.
    fn cloned(&self) -> RustGameState {
        RustGameState {
            boards: [self.boards[0].copy(), self.boards[1].copy()],
            deck: self.deck.clone(),
            current_row: self.current_row.clone(),
            pending_claims: self.pending_claims.clone(),
            next_claims: self.next_claims.clone(),
            phase: self.phase,
            actor_index: self.actor_index,
            initial_pick_count: self.initial_pick_count,
            start_player: self.start_player,
            harmony: self.harmony,
            middle_kingdom: self.middle_kingdom,
            discards: self.discards,
        }
    }

    /// Player about to act.  Errors after game over (matches Python).
    fn actor(&self) -> PyResult<u8> {
        match self.phase {
            INITIAL_SELECTION => {
                // Mighty Duel opening pick order: start, opp, opp, start.
                let s = self.start_player;
                let order = [s, 1 - s, 1 - s, s];
                Ok(order[self.initial_pick_count])
            }
            PLACE_AND_SELECT | FINAL_PLACEMENT => Ok(self.pending_claims[self.actor_index].0),
            _ => Err(PyValueError::new_err("No current actor after game over")),
        }
    }

    /// Deal the next four-tile row from the deck (sorted), advancing the deck.
    /// Caller guarantees the deck holds at least four tiles.
    fn deal_row(&mut self) {
        let mut row: Vec<u16> = self.deck[..4].to_vec();
        row.sort_unstable();
        self.current_row = row;
        self.deck.drain(..4);
    }

    /// End-of-round bookkeeping shared by the two turn phases: promote the
    /// next round's claims (sorted by domino id) to pending, reset the actor,
    /// then either deal a new row (PLACE_AND_SELECT) or enter FINAL_PLACEMENT.
    fn advance_round(&mut self) {
        self.next_claims.sort_by_key(|c| c.1);
        self.pending_claims = std::mem::take(&mut self.next_claims);
        self.actor_index = 0;
        // Rows are dealt four at a time, so the deck is always empty or holds a
        // whole number of future rows (a multiple of 4, ≥ 4 when non-empty).
        debug_assert!(
            self.deck.is_empty() || self.deck.len() >= 4,
            "advance_round: deck has {} tiles — expected 0 or >= 4",
            self.deck.len()
        );
        debug_assert_eq!(
            self.deck.len() % 4,
            0,
            "advance_round: deck length {} is not a multiple of 4",
            self.deck.len()
        );
        if !self.deck.is_empty() {
            self.deal_row();
            self.phase = PLACE_AND_SELECT;
        } else {
            self.current_row.clear();
            self.phase = FINAL_PLACEMENT;
        }
    }

    /// Domino the given player is currently placing, if any: only during a turn
    /// phase, only when the pending claim at actor_index belongs to `player`.
    fn domino_in_hand(&self, player: u8) -> Option<u16> {
        if self.phase != PLACE_AND_SELECT && self.phase != FINAL_PLACEMENT {
            return None;
        }
        let (cp, did) = *self.pending_claims.get(self.actor_index)?;
        if cp != player { None } else { Some(did) }
    }

    fn next_pending_summary(&self, owner: u8) -> Option<(u16, usize, usize)> {
        let mut current_remaining: Vec<(usize, u8, u16)> = Vec::new();
        if self.phase == PLACE_AND_SELECT || self.phase == FINAL_PLACEMENT {
            for (idx, &(claim_owner, did)) in self.pending_claims.iter().enumerate() {
                if idx >= self.actor_index {
                    current_remaining.push((idx, claim_owner, did));
                }
            }
            let mut first: Option<(u16, usize)> = None;
            let mut remaining: usize = 0;
            for &(idx, claim_owner, did) in &current_remaining {
                if claim_owner != owner {
                    continue;
                }
                remaining += 1;
                if first.is_none() {
                    first = Some((did, idx - self.actor_index));
                }
            }
            if let Some((did, distance)) = first {
                return Some((did, distance, remaining));
            }
        }

        if self.phase == INITIAL_SELECTION || self.phase == PLACE_AND_SELECT {
            let mut next_order = self.next_claims.clone();
            next_order.sort_by_key(|c| c.1);
            let mut first: Option<(u16, usize)> = None;
            let mut remaining: usize = 0;
            for (idx, &(claim_owner, did)) in next_order.iter().enumerate() {
                if claim_owner != owner {
                    continue;
                }
                remaining += 1;
                if first.is_none() {
                    first = Some((did, current_remaining.len() + idx));
                }
            }
            if let Some((did, distance)) = first {
                return Some((did, distance, remaining));
            }
        }

        None
    }

    /// Spatial placement index (mirrors action_codec._encode_placement): anchor =
    /// A-half cell, direction = where the B-half sits, in the 13×13 castle-centred
    /// frame.  None if the A-half maps outside the crop or the halves aren't an
    /// orthogonal step apart.  Uses the current actor's board for the castle.
    fn encode_placement(&self, p: (i8, i8, i8, i8, bool)) -> Option<u16> {
        let (x1, y1, x2, y2, flipped) = p;
        // Canonical form: A-half is the anchor; B-half's offset gives direction.
        let (ax, ay, bx, by) = if flipped {
            (x2, y2, x1, y1)
        } else {
            (x1, y1, x2, y2)
        };
        let actor = self.actor().ok()?;
        let board = &self.boards[actor as usize];
        let out_x = ax - board.castle_x + OUT_CENTER;
        let out_y = ay - board.castle_y + OUT_CENTER;
        if out_x < 0 || out_x >= OUT_N as i8 || out_y < 0 || out_y >= OUT_N as i8 {
            return None;
        }
        let delta = (bx - ax, by - ay);
        let direction = CODEC_DIRS.iter().position(|&d| d == delta)? as u16;
        Some(direction * CODEC_CELLS + out_y as u16 * OUT_N as u16 + out_x as u16)
    }

    /// Placement index with symmetric-domino canonicalization (mirrors the
    /// Python encode_action fix): when the domino being placed has two identical
    /// halves (ids 1..12), the same physical move can anchor at either cell, so
    /// collapse to the smaller of the two anchor encodings.  Representation-
    /// invariant, so it agrees regardless of which representative legal_placements
    /// returned.
    fn encode_placement_canonical(&self, p: (i8, i8, i8, i8, bool)) -> Option<u16> {
        let idx = self.encode_placement(p)?;
        let domino_id = self.pending_claims.get(self.actor_index)?.1;
        let (ta, ca, tb, cb) = dom(domino_id);
        if ta == tb && ca == cb {
            let (x1, y1, x2, y2, flipped) = p;
            let alt = self.encode_placement((x2, y2, x1, y1, flipped))?;
            Some(idx.min(alt))
        } else {
            Some(idx)
        }
    }

    /// Joint action index for a (placement, pick) action (mirrors
    /// action_codec.encode_action), dispatched by phase.  None if it doesn't
    /// encode (placement out of crop, or pick not in current_row) — which never
    /// happens for a legal action.
    fn enc_action(
        &self,
        placement: Option<(i8, i8, i8, i8, bool)>,
        pick: Option<u16>,
    ) -> Option<u16> {
        // Phase-strict shape: INITIAL_SELECTION is pick-only (no placement);
        // FINAL_PLACEMENT is placement-only (no pick).  A mismatch means a caller
        // built an action for the wrong phase.
        if self.phase == INITIAL_SELECTION {
            debug_assert!(
                placement.is_none(),
                "enc_action: placement must be None in INITIAL_SELECTION"
            );
        }
        if self.phase == FINAL_PLACEMENT {
            debug_assert!(
                pick.is_none(),
                "enc_action: pick must be None in FINAL_PLACEMENT, got {:?}",
                pick
            );
        }
        match self.phase {
            INITIAL_SELECTION => {
                let d = pick?;
                let pi = self.current_row.iter().position(|&x| x == d)? as u16;
                Some(NO_PLACEMENT_IDX * PICK_AXIS_SIZE + pi)
            }
            PLACE_AND_SELECT | FINAL_PLACEMENT => {
                let placement_idx = match placement {
                    None => DISCARD_PLACEMENT_IDX,
                    Some(p) => self.encode_placement_canonical(p)?,
                };
                let pick_idx = if self.phase == FINAL_PLACEMENT {
                    NO_PICK_IDX
                } else {
                    let pk = pick?;
                    self.current_row.iter().position(|&x| x == pk)? as u16
                };
                Some(placement_idx * PICK_AXIS_SIZE + pick_idx)
            }
            _ => None,
        }
    }

    /// Legal actions as (placement, pick) tuples in raw enumeration order
    /// (placements in board order × picks in current_row order).  Set-equivalent
    /// to Python's legal_actions; ordering is canonicalised by callers.
    fn legal_actions_raw(&self) -> Vec<(Option<(i8, i8, i8, i8, bool)>, Option<u16>)> {
        match self.phase {
            INITIAL_SELECTION => self.current_row.iter().map(|&d| (None, Some(d))).collect(),
            PLACE_AND_SELECT | FINAL_PLACEMENT => {
                let (player, domino_id) = self.pending_claims[self.actor_index];
                let (ta, ca, tb, cb) = dom(domino_id);
                let placements = self.boards[player as usize].legal_placements(ta, ca, tb, cb);
                // Kingdomino forces a discard only when nothing can be placed.
                let placement_options: Vec<Option<(i8, i8, i8, i8, bool)>> =
                    if placements.is_empty() {
                        vec![None]
                    } else {
                        placements.into_iter().map(Some).collect()
                    };
                if self.phase == FINAL_PLACEMENT {
                    placement_options.into_iter().map(|p| (p, None)).collect()
                } else {
                    let mut out =
                        Vec::with_capacity(placement_options.len() * self.current_row.len());
                    for p in &placement_options {
                        for &pick in &self.current_row {
                            out.push((*p, Some(pick)));
                        }
                    }
                    out
                }
            }
            _ => Vec::new(),
        }
    }

    /// Legal actions paired with their joint index, sorted ascending by index —
    /// the canonical ordering shared with the Python engine (legal joint indices
    /// are unique, so the sort is total and deterministic).
    fn legal_actions_indexed(&self) -> Vec<(u16, Option<(i8, i8, i8, i8, bool)>, Option<u16>)> {
        let mut v: Vec<(u16, Option<(i8, i8, i8, i8, bool)>, Option<u16>)> = self
            .legal_actions_raw()
            .into_iter()
            .map(|(p, pk)| {
                (
                    self.enc_action(p, pk).expect("legal action must encode"),
                    p,
                    pk,
                )
            })
            .collect();
        v.sort_by_key(|t| t.0);
        // Joint indices are unique per legal action; a duplicate means the codec
        // mapped two distinct legal actions to the same index (a codec bug that
        // would silently corrupt masks/priors).  Strictly ascending after sort.
        debug_assert!(
            v.windows(2).all(|w| w[0].0 != w[1].0),
            "legal_actions_indexed: duplicate joint index detected — codec bug"
        );
        v
    }

    /// Build the encoder outputs as ndarray Arrays (the core of `encode`, with no
    /// Python/numpy objects).  Mirrors encoder.encode_state exactly; the `encode`
    /// pymethod and the in-process MCTS leaf evaluation both go through this so
    /// they cannot drift.
    /// Core encoder: writes the (9,13,13) my/opp board planes and the flat
    /// vector directly into the provided slices (each exactly one example wide).
    /// Zeroes all three first, so callers may pass reused buffers.  Both the
    /// allocating `encode_arrays` and the batch-buffer `encode_arrays_into` go
    /// through this, so they cannot drift from encoder.encode_state.
    fn encode_into_slices(
        &self,
        player: u8,
        mb: &mut [f32],
        ob: &mut [f32],
        flat: &mut [f32],
    ) -> PyResult<()> {
        if self.phase == GAME_OVER {
            return Err(PyValueError::new_err(
                "encode is not defined for terminal states",
            ));
        }
        if player >= 2 {
            return Err(PyValueError::new_err("Invalid player index"));
        }
        let opp = 1 - player;

        encode_board_spatial_into(&self.boards[player as usize], mb);
        encode_board_spatial_into(&self.boards[opp as usize], ob);

        flat.fill(0.0);

        // 1. Symmetric pending-placement summaries for both sides.
        write_pending_summary(flat, OFF_MY_NEXT_PENDING, self.next_pending_summary(player));
        write_pending_summary(flat, OFF_OPP_NEXT_PENDING, self.next_pending_summary(opp));
        // 1b. Rule-derived board summaries for both sides.
        write_board_summary(flat, OFF_MY_BOARD_SUMMARY, self, player);
        write_board_summary(flat, OFF_OPP_BOARD_SUMMARY, self, opp);
        // 2. Current row (up to 4 slots).
        for i in 0..4 {
            write_row_slot(
                flat,
                OFF_CURRENT_ROW + i * ROW_SLOT,
                self.current_row.get(i).copied(),
            );
        }
        // 3. Pending claims — status flag marks claims already resolved (placed).
        for i in 0..4 {
            let claim = self.pending_claims.get(i).copied();
            let already_placed = if claim.is_some() && i < self.actor_index {
                1.0
            } else {
                0.0
            };
            write_claim_slot(
                flat,
                OFF_PENDING + i * CLAIM_SLOT,
                claim,
                player,
                already_placed,
            );
        }
        // 4. Next claims — status flag just marks the slot as filled.
        for i in 0..4 {
            let claim = self.next_claims.get(i).copied();
            let slot_filled = if claim.is_some() { 1.0 } else { 0.0 };
            write_claim_slot(flat, OFF_NEXT + i * CLAIM_SLOT, claim, player, slot_filled);
        }
        // 5. Bag — derived from deck membership (the deck is exactly the set of
        //    unrevealed tiles, i.e. the complement of row ∪ claims ∪ placed).
        for &did in &self.deck {
            flat[OFF_BAG + (did - 1) as usize] = 1.0;
        }
        // 6. Phase one-hot (GAME_OVER excluded above; phase ∈ {0,1,2}).
        flat[OFF_PHASE + self.phase as usize] = 1.0;
        // 7. Game progress: placed cells (excluding castles) / 96.
        let placed = (self.boards[0].occupied as i32 - 1) + (self.boards[1].occupied as i32 - 1);
        flat[OFF_GAME_PROGRESS] = (placed as f64 / 96.0) as f32;
        // 8. Per-board fill ratios.
        flat[OFF_MY_FILL] = fill_ratio(&self.boards[player as usize]);
        flat[OFF_OPP_FILL] = fill_ratio(&self.boards[opp as usize]);
        // 9. Actor flag: is the encoded player the one about to act?
        flat[OFF_ACTOR_FLAG] = if self.actor()? == player { 1.0 } else { 0.0 };
        // 10. Next-round pick positions (full interleaving).
        //     +1 = encoded player acts here, -1 = opponent, 0 = unknown/no round.
        let pos = pick_positions(self, player);
        flat[OFF_PICK_POS_0] = pos[0];
        flat[OFF_PICK_POS_1] = pos[1];
        flat[OFF_PICK_POS_2] = pos[2];
        flat[OFF_PICK_POS_3] = pos[3];

        Ok(())
    }

    /// Allocating encoder: (my_board (9,13,13), opp_board (9,13,13), flat).
    /// Thin wrapper over encode_into_slices (which holds the canonical logic).
    fn encode_arrays(&self, player: u8) -> PyResult<(Array3<f32>, Array3<f32>, Array1<f32>)> {
        let mut mb = vec![0f32; N_BOARD_CH * OUT_N * OUT_N];
        let mut ob = vec![0f32; N_BOARD_CH * OUT_N * OUT_N];
        let mut flat = vec![0f32; FLAT_SIZE];
        self.encode_into_slices(player, &mut mb, &mut ob, &mut flat)?;
        Ok((
            Array3::from_shape_vec((N_BOARD_CH, OUT_N, OUT_N), mb).expect("mb shape"),
            Array3::from_shape_vec((N_BOARD_CH, OUT_N, OUT_N), ob).expect("ob shape"),
            Array1::from_vec(flat),
        ))
    }

    /// Encode directly into pre-allocated batch buffers at row `row`, avoiding the
    /// intermediate Array3/Array1 allocation + copy that encode_arrays incurs.
    /// `mb_data`/`ob_data` are (rows, 9, 13, 13) flat; `flat_data` is (rows, FLAT_SIZE)
    /// flat.  Each row's region is written (and zeroed) by encode_into_slices.
    fn encode_arrays_into(
        &self,
        player: u8,
        mb_data: &mut [f32],
        ob_data: &mut [f32],
        flat_data: &mut [f32],
        row: usize,
    ) -> PyResult<()> {
        let board_sz = N_BOARD_CH * OUT_N * OUT_N;
        let mb_off = row * board_sz;
        let fl_off = row * FLAT_SIZE;
        self.encode_into_slices(
            player,
            &mut mb_data[mb_off..mb_off + board_sz],
            &mut ob_data[mb_off..mb_off + board_sz],
            &mut flat_data[fl_off..fl_off + FLAT_SIZE],
        )
    }
}

// ─── Make / unmake: mutable, reversible engine for deep search ───────────────
//
// The NNUE searcher walks ONE `RustGameState` down and back up the tree instead
// of cloning per node (the functional `step` clones two 225-cell boards every
// call — the measured bottleneck).  `make` applies an action in place and returns
// an `UndoRecord`; `unmake` replays it in reverse to the byte-identical prior
// state.  `step` is retained unchanged as the correctness oracle (see the
// differential tests in `make_unmake_tests`).
//
// Reversibility notes (verified against `place` / `advance_round`):
//   * `place` writes 2 cells + `occupied += 2` and updates the bounding box
//     MONOTONICALLY, so the bbox cannot be recomputed on undo — `PlaceUndo`
//     snapshots the pre-move bbox and restores it verbatim.  This is the one
//     spot a naive "clear the two cells" undo would silently corrupt scoring.
//   * `current_row.remove(pos)` shifts the tail → undo re-inserts at `pos`.
//   * `next_claims.push` → undo `pop` (the current actor's claim is always last).
//   * Round boundaries (`deal_row` / `advance_round`, and the 4th-initial-pick
//     promotion) rewrite deck / row / claim vectors wholesale.  Rather than
//     invert the deal, `RoundSnapshot` captures the six mutated vectors/scalars
//     just before the transform and restores them.  These vecs are tiny and the
//     boundary fires ~12×/game-line, not per node.

/// Undo payload for a single board placement.
struct PlaceUndo {
    i1: usize,
    i2: usize,
    /// Pre-placement (min_x, min_y, max_x, max_y).
    bbox: (i8, i8, i8, i8),
}

/// Snapshot of every field a round-boundary transform rewrites, captured at the
/// instant AFTER the per-move mutations but BEFORE the boundary transform.
struct RoundSnapshot {
    deck: Vec<u16>,
    current_row: Vec<u16>,
    pending_claims: Vec<(u8, u16)>,
    next_claims: Vec<(u8, u16)>,
    actor_index: usize,
    phase: u8,
}

impl RoundSnapshot {
    fn capture(s: &RustGameState) -> Self {
        RoundSnapshot {
            deck: s.deck.clone(),
            current_row: s.current_row.clone(),
            pending_claims: s.pending_claims.clone(),
            next_claims: s.next_claims.clone(),
            actor_index: s.actor_index,
            phase: s.phase,
        }
    }
    fn restore(self, s: &mut RustGameState) {
        s.deck = self.deck;
        s.current_row = self.current_row;
        s.pending_claims = self.pending_claims;
        s.next_claims = self.next_claims;
        s.actor_index = self.actor_index;
        s.phase = self.phase;
    }
}

/// Everything needed to reverse one `make`.
enum UndoRecord {
    InitialPick {
        removed_pos: usize,
        domino_id: u16,
        boundary: Option<RoundSnapshot>,
    },
    Move {
        player: u8,
        /// `Some` for a placement, `None` for a forced discard.
        place: Option<PlaceUndo>,
        /// `(pos, pick_id)` in PLACE_AND_SELECT; `None` in FINAL_PLACEMENT.
        pick: Option<(usize, u16)>,
        boundary: Option<RoundSnapshot>,
    },
}

impl RustGameState {
    /// Apply `action` in place, returning the record needed to reverse it.
    /// Mirrors `step`'s branches exactly but mutates `self`.  Atomic on any legal
    /// action: all fallible lookups run before any mutation, and `place` is itself
    /// atomic, so an error leaves `self` unchanged.
    fn make(
        &mut self,
        placement: Option<(i8, i8, i8, i8, bool)>,
        pick_domino_id: Option<u16>,
    ) -> PyResult<UndoRecord> {
        match self.phase {
            INITIAL_SELECTION => {
                if placement.is_some() {
                    return Err(PyValueError::new_err(
                        "INITIAL_SELECTION takes a pick only, no placement",
                    ));
                }
                let d = pick_domino_id
                    .ok_or_else(|| PyValueError::new_err("INITIAL_SELECTION requires a pick"))?;
                let pos = self
                    .current_row
                    .iter()
                    .position(|&x| x == d)
                    .ok_or_else(|| PyValueError::new_err("Picked domino not available"))?;
                let actor = self.actor()?; // attributed before the count increments
                // --- mutate ---
                self.current_row.remove(pos);
                self.next_claims.push((actor, d));
                self.initial_pick_count += 1;
                let boundary = if self.initial_pick_count == 4 {
                    let snap = RoundSnapshot::capture(self);
                    self.next_claims.sort_by_key(|c| c.1);
                    self.pending_claims = std::mem::take(&mut self.next_claims);
                    self.deal_row();
                    self.actor_index = 0;
                    self.phase = PLACE_AND_SELECT;
                    Some(snap)
                } else {
                    None
                };
                Ok(UndoRecord::InitialPick {
                    removed_pos: pos,
                    domino_id: d,
                    boundary,
                })
            }
            PLACE_AND_SELECT | FINAL_PLACEMENT => {
                let (player, domino_id) = self.pending_claims[self.actor_index];
                // Resolve the fallible pick lookup BEFORE mutating so `make` stays
                // atomic (PLACE_AND_SELECT only; FINAL_PLACEMENT has no pick).
                let pick_undo = if self.phase == PLACE_AND_SELECT {
                    let pk = pick_domino_id
                        .ok_or_else(|| PyValueError::new_err("PLACE_AND_SELECT requires a pick"))?;
                    let pos = self
                        .current_row
                        .iter()
                        .position(|&x| x == pk)
                        .ok_or_else(|| PyValueError::new_err("Picked domino not available"))?;
                    Some((pos, pk))
                } else {
                    None
                };
                // --- placement / discard (place is atomic on error) ---
                let place_undo = if let Some((x1, y1, x2, y2, flipped)) = placement {
                    let (ta, ca, tb, cb) = dom(domino_id);
                    let b = &self.boards[player as usize];
                    let bbox = (b.min_x, b.min_y, b.max_x, b.max_y);
                    self.boards[player as usize].place(ta, ca, tb, cb, x1, y1, x2, y2, flipped)?;
                    Some(PlaceUndo {
                        i1: idx(x1, y1),
                        i2: idx(x2, y2),
                        bbox,
                    })
                } else {
                    self.discards[player as usize] += 1;
                    None
                };
                // --- pick apply ---
                if let Some((pos, pk)) = pick_undo {
                    self.current_row.remove(pos);
                    self.next_claims.push((player, pk));
                }
                self.actor_index += 1;
                let boundary = if self.actor_index >= self.pending_claims.len() {
                    let snap = RoundSnapshot::capture(self);
                    if self.phase == FINAL_PLACEMENT {
                        self.phase = GAME_OVER;
                    } else {
                        self.advance_round();
                    }
                    Some(snap)
                } else {
                    None
                };
                Ok(UndoRecord::Move {
                    player,
                    place: place_undo,
                    pick: pick_undo,
                    boundary,
                })
            }
            _ => Err(PyValueError::new_err("Cannot step a terminal state")),
        }
    }

    /// Reverse a `make`, restoring the exact prior state.  Undo the boundary
    /// (if any) first, then the per-move mutations in reverse.
    fn unmake(&mut self, record: UndoRecord) {
        match record {
            UndoRecord::InitialPick {
                removed_pos,
                domino_id,
                boundary,
            } => {
                if let Some(snap) = boundary {
                    snap.restore(self);
                }
                self.initial_pick_count -= 1;
                self.next_claims.pop();
                self.current_row.insert(removed_pos, domino_id);
            }
            UndoRecord::Move {
                player,
                place,
                pick,
                boundary,
            } => {
                if let Some(snap) = boundary {
                    snap.restore(self);
                }
                self.actor_index -= 1;
                if let Some((pos, pk)) = pick {
                    self.next_claims.pop();
                    self.current_row.insert(pos, pk);
                }
                if let Some(pu) = place {
                    let b = &mut self.boards[player as usize];
                    b.terrain[pu.i1] = EMPTY;
                    b.crowns[pu.i1] = 0;
                    b.terrain[pu.i2] = EMPTY;
                    b.crowns[pu.i2] = 0;
                    b.occupied -= 2;
                    b.min_x = pu.bbox.0;
                    b.min_y = pu.bbox.1;
                    b.max_x = pu.bbox.2;
                    b.max_y = pu.bbox.3;
                } else {
                    self.discards[player as usize] -= 1;
                }
            }
        }
    }

    /// Official outcome in player-0 frame: +1 (P0 wins), -1 (P1 wins), 0 (draw).
    /// Mirrors game.py `determine_winner`'s cascade — meaningful only at a
    /// terminal state (it scores the current boards directly).
    fn official_outcome_i8(&self) -> i8 {
        let k0 = self.boards[0].cascade_key(self.harmony, self.middle_kingdom);
        let k1 = self.boards[1].cascade_key(self.harmony, self.middle_kingdom);
        // Rust tuple comparison is lexicographic, matching Python's key tuples.
        if k0 > k1 {
            1
        } else if k1 > k0 {
            -1
        } else {
            0
        }
    }
}

impl RustBoard {
    /// Official tiebreak key `(total, largest_single_territory_size, total_crowns)`
    /// — the per-board key `determine_winner` compares.  The largest-territory and
    /// crown terms reuse the exact same-terrain flood fill as `score`.
    fn cascade_key(&self, harmony: bool, middle_kingdom: bool) -> (i32, i32, i32) {
        let (territory, hb, mb) = self.score(harmony, middle_kingdom);
        let total = territory + hb + mb;
        let mut visited = [false; CELLS];
        let mut largest: i32 = 0;
        for sy in self.min_y..=self.max_y {
            for sx in self.min_x..=self.max_x {
                let si = idx(sx, sy);
                let t = self.terrain[si];
                if visited[si] || t == EMPTY || t == CASTLE {
                    continue;
                }
                let mut stack: Vec<(i8, i8)> = Vec::with_capacity(49);
                stack.push((sx, sy));
                visited[si] = true;
                let mut area: i32 = 0;
                while let Some((cx, cy)) = stack.pop() {
                    area += 1;
                    for (dx, dy) in DIRS {
                        let nx = cx + dx;
                        let ny = cy + dy;
                        if in_bounds(nx, ny) {
                            let ni = idx(nx, ny);
                            if !visited[ni] && self.terrain[ni] == t {
                                visited[ni] = true;
                                stack.push((nx, ny));
                            }
                        }
                    }
                }
                if area > largest {
                    largest = area;
                }
            }
        }
        let total_crowns: i32 = self.crowns.iter().map(|&c| c as i32).sum();
        (total, largest, total_crowns)
    }
}

/// A mutable search driver over one `RustGameState`: walks the state via
/// `make`/`unmake` with an internal undo stack, so a Python-hosted expectiminimax
/// never clones per node.  Chance children are produced with `make_with_row`,
/// which installs a specified drawn row (the enumerated/sampled future) instead
/// of the hidden actual deck order.  The functional `RustGameState::step` remains
/// the oracle; `make`/`unmake` were byte-exact validated against it (see
/// `make_unmake_tests`).
#[pyclass]
struct SearchEngine {
    state: RustGameState,
    undo: Vec<UndoRecord>,
}

#[pymethods]
impl SearchEngine {
    #[new]
    fn new(state: &RustGameState) -> Self {
        SearchEngine {
            state: state.cloned(),
            undo: Vec::new(),
        }
    }

    /// Apply an action, following the ACTUAL deck order on a deal.  Use for
    /// deterministic moves and for a straight (non-search) playout.
    fn make(
        &mut self,
        placement: Option<(i8, i8, i8, i8, bool)>,
        pick: Option<u16>,
    ) -> PyResult<()> {
        let rec = self.state.make(placement, pick)?;
        self.undo.push(rec);
        Ok(())
    }

    /// Apply an action; if it triggers a deal, install `row` (a 4-tile subset of
    /// the pre-deal bag) as the new current row instead of the actual deck order.
    /// This is the chance-child expansion: `deck` becomes the pre-deal bag minus
    /// `row`, both sorted (hidden order carries no information).  `unmake` reverses
    /// it via the boundary snapshot exactly as for a plain `make`.
    fn make_with_row(
        &mut self,
        placement: Option<(i8, i8, i8, i8, bool)>,
        pick: Option<u16>,
        row: Vec<u16>,
    ) -> PyResult<()> {
        // Validate the chosen row against the PRE-deal bag BEFORE mutating, so an
        // invalid request errors with the engine and undo depth untouched.  A
        // silently-accepted bad row (wrong length, dups, alien/absent tiles, or an
        // empty row that preserves the hidden actual deal) would corrupt the state
        // or reintroduce clairvoyance — this is foundational infra, so it is strict.
        if row.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "make_with_row: row must have exactly 4 tiles, got {}",
                row.len()
            )));
        }
        let pre_deck = self.state.deck.clone();
        let mut sorted_row = row;
        sorted_row.sort_unstable();
        for w in sorted_row.windows(2) {
            if w[0] == w[1] {
                return Err(PyValueError::new_err(format!(
                    "make_with_row: row has a duplicate tile ({})",
                    w[0]
                )));
            }
        }
        for r in &sorted_row {
            if !pre_deck.contains(r) {
                return Err(PyValueError::new_err(format!(
                    "make_with_row: tile {} is not in the pre-deal bag",
                    r
                )));
            }
        }
        // Apply, then require that this action actually deals a new row.
        let rec = self.state.make(placement, pick)?;
        let dealt = matches!(
            &rec,
            UndoRecord::InitialPick {
                boundary: Some(_),
                ..
            } | UndoRecord::Move {
                boundary: Some(_),
                ..
            }
        ) && !self.state.current_row.is_empty();
        if !dealt {
            // A row was supplied for an action that does not deal — reverse and
            // reject rather than silently keep the actual (hidden-order) row.
            self.state.unmake(rec);
            return Err(PyValueError::new_err(
                "make_with_row: action does not reveal a new row (not a chance node)",
            ));
        }
        // Install the validated chosen row; deck = pre-deal bag minus row (sorted).
        // Every tile is validated present-and-distinct, so each removal succeeds.
        let mut remaining = pre_deck;
        for r in &sorted_row {
            if let Some(pos) = remaining.iter().position(|x| x == r) {
                remaining.remove(pos);
            }
        }
        remaining.sort_unstable();
        self.state.current_row = sorted_row;
        self.state.deck = remaining;
        self.undo.push(rec);
        Ok(())
    }

    /// Reverse the most recent make.  Errors on an empty stack.
    fn unmake(&mut self) -> PyResult<()> {
        let rec = self
            .undo
            .pop()
            .ok_or_else(|| PyValueError::new_err("unmake called with empty undo stack"))?;
        self.state.unmake(rec);
        Ok(())
    }

    /// Number of un-reversed makes on the stack (search ply depth).
    fn depth(&self) -> usize {
        self.undo.len()
    }

    // ── read delegates for the search + eval ────────────────────────────────
    #[getter]
    fn phase(&self) -> u8 {
        self.state.phase
    }
    fn current_actor(&self) -> PyResult<u8> {
        self.state.actor()
    }
    #[getter]
    fn actor_index(&self) -> usize {
        self.state.actor_index
    }
    #[getter]
    fn initial_pick_count(&self) -> usize {
        self.state.initial_pick_count
    }
    fn legal_actions(&self) -> Vec<(Option<(i8, i8, i8, i8, bool)>, Option<u16>)> {
        self.state.legal_actions()
    }
    fn deck(&self) -> Vec<u16> {
        self.state.deck.clone()
    }
    fn current_row(&self) -> Vec<u16> {
        self.state.current_row.clone()
    }
    fn pending_claims(&self) -> Vec<(u8, u16)> {
        self.state.pending_claims.clone()
    }
    fn next_claims(&self) -> Vec<(u8, u16)> {
        self.state.next_claims.clone()
    }
    /// Board totals (territory + harmony + middle) for (player0, player1).
    fn scores(&self) -> (i32, i32) {
        let a = self.state.boards[0].score(self.state.harmony, self.state.middle_kingdom);
        let b = self.state.boards[1].score(self.state.harmony, self.state.middle_kingdom);
        (a.0 + a.1 + a.2, b.0 + b.1 + b.2)
    }
    /// Official outcome, player-0 frame: +1 (P0), -1 (P1), 0 (draw).
    fn official_outcome(&self) -> i8 {
        self.state.official_outcome_i8()
    }
    /// Deep copy of the current underlying state.
    fn snapshot(&self) -> RustGameState {
        self.state.cloned()
    }
}

// ── Kingdomino as the first `search::Game` implementation ────────────────────
//
// The depth-limited expectiminimax recursion now lives in the game-agnostic
// `search` module; this section is the KINGDOMINO-SPECIFIC glue that plugs into
// it: the `Game` trait impl (rules/chance) + `Eval` impls (leaf value) + the
// `RustSearch` pyclass (Python entry point). The search walks one mutable
// `RustGameState` via make/unmake (no per-node FFI), and — with every in-horizon
// chance node enumerated (deterministic) — returns search VALUES numerically
// identical (within 1e-9) to the Python-hosted `RustExpectiminimax` /
// `ExpectiminimaxBot`. See `test_rust_search_equiv.py`. Sampled (wide) chance uses
// its own reproducible SplitMix64 RNG — a Monte-Carlo estimate of the same
// expectiminimax value, validated by the enumerated core plus convergence.
//
// The trait boundary (search::Game / search::Eval) is what a SECOND game would
// implement; `search` is written to be extractable to a standalone crate at that
// point ("build one, extract at two"). Constants/evals below are Kingdomino-only.

const EMM_SCORE_SCALE: f64 = 40.0; // must match rust_expectiminimax.SCORE_SCALE
const EMM_CROWN_WEIGHT: f64 = 4.0; // must match rust_expectiminimax.pick_aware

#[derive(Clone, Copy, PartialEq)]
enum EmmEval {
    PickBlind,           // tanh(margin / scale)
    PickAware,           // tanh((margin + w*(claimed_crowns0 - claimed_crowns1)) / scale)
    Nnue,                // trained dense net (weights held in RustSearch.nnue)
    SparseNnueRef,       // Step-3 sparse net, stateless full accumulator rebuild
    SparseNnue,          // Step-3 sparse net, reversible dual accumulators
    QuantizedSparseNnue, // v3.2 guarded int16/int8 incremental inference
}

/// C(n, 4) as a u64 (0 for n < 4).
fn emm_comb4(n: usize) -> u64 {
    if n < 4 {
        return 0;
    }
    let n = n as u64;
    n * (n - 1) * (n - 2) * (n - 3) / 24
}

fn emm_margin(s: &RustGameState) -> f64 {
    let (s0, s1) = s.scores();
    (s0 - s1) as f64
}

fn emm_tanh_margin(s: &RustGameState) -> f64 {
    (emm_margin(s) / EMM_SCORE_SCALE).tanh()
}

/// Crowns on `player`'s claimed-but-unplaced dominoes.  Mirrors
/// rust_expectiminimax._claimed_crowns: next_claims (all) + pending_claims from
/// the current actor_index onward (only while the game is live).
fn emm_claimed_crowns(s: &RustGameState, player: u8) -> i32 {
    let mut crowns = 0i32;
    for &(pl, did) in &s.next_claims {
        if pl == player {
            let (_, ca, _, cb) = dom(did);
            crowns += (ca + cb) as i32;
        }
    }
    if s.phase != GAME_OVER {
        for &(pl, did) in &s.pending_claims[s.actor_index..] {
            if pl == player {
                let (_, ca, _, cb) = dom(did);
                crowns += (ca + cb) as i32;
            }
        }
    }
    crowns
}

fn emm_pick_aware(s: &RustGameState) -> f64 {
    let pot = EMM_CROWN_WEIGHT * (emm_claimed_crowns(s, 0) - emm_claimed_crowns(s, 1)) as f64;
    ((emm_margin(s) + pot) / EMM_SCORE_SCALE).tanh()
}

/// Player-0-frame leaf evals for Kingdomino, plugged into the generic searcher.
struct PickBlindEval; // tanh(margin / scale)
struct PickAwareEval; // + claimed-domino crown potential

impl search::Eval<Kingdomino> for PickBlindEval {
    fn eval(&self, s: &RustGameState) -> f64 {
        emm_tanh_margin(s)
    }
}

impl search::Eval<Kingdomino> for PickAwareEval {
    fn eval(&self, s: &RustGameState) -> f64 {
        emm_pick_aware(s)
    }
}

/// Kingdomino as an implementation of the generic `search::Game` trait (impl #1).
/// A zero-sized type used only as a type-level namespace; all state lives in the
/// `RustGameState` passed to each method.
struct Kingdomino;

impl search::Game for Kingdomino {
    type State = RustGameState;
    type Action = (Option<(i8, i8, i8, i8, bool)>, Option<u16>);
    type Chance = Vec<u16>;
    type Undo = UndoRecord;

    fn to_move(s: &RustGameState) -> PyResult<search::Turn> {
        Ok(if s.actor()? == 0 {
            search::Turn::P0
        } else {
            search::Turn::P1
        })
    }

    fn is_terminal(s: &RustGameState) -> bool {
        s.phase == GAME_OVER
    }

    /// Official outcome, player-0 frame: +1 / 0 / -1 (the determine_winner cascade).
    fn terminal_value_p0(s: &RustGameState) -> f64 {
        s.official_outcome_i8() as f64
    }

    fn bounded_margin(s: &RustGameState) -> f64 {
        emm_tanh_margin(s)
    }

    fn legal_actions(s: &RustGameState, out: &mut Vec<Self::Action>) {
        out.extend(s.legal_actions());
    }

    fn action_order_score(s: &RustGameState, a: Self::Action) -> f64 {
        let Ok(actor) = s.actor() else {
            return 0.0;
        };
        let mut score = 0i32;
        if let Some((x1, y1, x2, y2, flipped)) = a.0 {
            if let Some(&(_, domino_id)) = s.pending_claims.get(s.actor_index) {
                let (ta, ca, tb, cb) = dom(domino_id);
                let (t1, c1, t2, c2) = if flipped {
                    (tb, cb, ta, ca)
                } else {
                    (ta, ca, tb, cb)
                };
                score += 8 * placement_score_delta(
                    &s.boards[actor as usize],
                    t1,
                    c1,
                    x1,
                    y1,
                    t2,
                    c2,
                    x2,
                    y2,
                );
            }
        }
        if let Some(domino_id) = a.1 {
            let mine = terrain_counts(&s.boards[actor as usize]);
            let theirs = terrain_counts(&s.boards[1 - actor as usize]);
            let (_, ca, _, cb) = dom(domino_id);
            score += 6 * (ca + cb) as i32;
            score += 2 * pick_order_score(domino_id, &mine);
            score += opponent_denial_score(domino_id, &theirs);
        }
        if actor == 0 {
            score as f64
        } else {
            -(score as f64)
        }
    }

    /// Mirrors `rust_expectiminimax.RustExpectiminimax._deals`: the last mover of a
    /// round triggers the next deal (a chance node). In Kingdomino the deal is
    /// independent of which placement/pick the mover chose, so `_a` is ignored.
    fn is_stochastic(s: &RustGameState, _a: Self::Action) -> bool {
        match s.phase {
            INITIAL_SELECTION => s.initial_pick_count == 3,
            PLACE_AND_SELECT => s.deck.len() >= 4 && s.actor_index + 1 == s.pending_claims.len(),
            _ => false,
        }
    }

    /// (row, weight) pairs for the pending deal: enumerated in lexicographic order
    /// over the sorted deck when C(n,4) <= enum_cap (matching itertools.combinations),
    /// else Monte-Carlo sampled with SplitMix64. Weights sum to 1. The deal does not
    /// depend on the action taken, so `_a` is ignored.
    fn chance_children(
        s: &RustGameState,
        _a: Self::Action,
        cfg: &search::SearchConfig,
    ) -> Vec<(Vec<u16>, f64)> {
        let mut deck = s.deck.clone();
        deck.sort_unstable();
        let n = deck.len();
        let n_rows = emm_comb4(n);
        if n_rows <= cfg.enum_cap {
            let p = 1.0 / n_rows as f64;
            let mut out = Vec::with_capacity(n_rows as usize);
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        for l in (k + 1)..n {
                            out.push((vec![deck[i], deck[j], deck[k], deck[l]], p));
                        }
                    }
                }
            }
            out
        } else {
            // Reproducible per-node seed from the config seed folded with the sorted
            // bag, so a given position always samples the same rows.
            let mut rng = cfg.seed ^ 0x1234_5678_9ABC_DEF0;
            for &d in &deck {
                rng = rng.wrapping_mul(0x0100_0000_01B3) ^ d as u64;
            }
            let w = 1.0 / cfg.chance_samples as f64;
            (0..cfg.chance_samples)
                .map(|_| {
                    // partial Fisher-Yates over indices → 4 distinct tiles, then sort.
                    let mut idx: Vec<usize> = (0..n).collect();
                    for i in 0..4 {
                        let j = i + (search::splitmix64(&mut rng) as usize) % (n - i);
                        idx.swap(i, j);
                    }
                    let mut row: Vec<u16> = idx[..4].iter().map(|&i| deck[i]).collect();
                    row.sort_unstable();
                    (row, w)
                })
                .collect()
        }
    }

    fn make(s: &mut RustGameState, a: Self::Action) -> PyResult<UndoRecord> {
        s.make(a.0, a.1)
    }

    fn make_with_chance(
        s: &mut RustGameState,
        a: Self::Action,
        c: &Self::Chance,
    ) -> PyResult<UndoRecord> {
        s.make_with_row_core(a.0, a.1, c)
    }

    fn unmake(s: &mut RustGameState, u: UndoRecord) {
        s.unmake(u);
    }

    fn exact_remaining_plies(s: &RustGameState) -> Option<u32> {
        match s.phase {
            FINAL_PLACEMENT if s.deck.is_empty() => {
                Some((s.pending_claims.len() - s.actor_index) as u32)
            }
            PLACE_AND_SELECT if s.deck.is_empty() || s.deck.len() == 4 => {
                let current = (s.pending_claims.len() - s.actor_index) as u32;
                // Every current action claims one domino for the next round.
                // With an empty deck that next round is four final placements;
                // with one row left it is four place-and-select moves followed
                // by four final placements. No hidden draw remains in either case.
                Some(current + if s.deck.is_empty() { 4 } else { 8 })
            }
            _ => None,
        }
    }

    fn position_key(s: &RustGameState, scratch: &mut Vec<u8>) -> Option<u128> {
        // Kingdomino's measured re-entry comes from pick-order permutations
        // collapsing when a round is promoted/sorted. Hash those canonical
        // round roots; hashing every interior placement node costs more than it
        // saves and those states rarely transpose.
        if s.phase == INITIAL_SELECTION || s.actor_index != 0 {
            return None;
        }
        solver_state_bytes(s, scratch);
        // The legacy exact-endgame key intentionally omits fields that cannot
        // affect future rules. A depth-limited NNUE TT must additionally retain
        // every field visible to the evaluator, especially discard history, and
        // opening actor identity before actor_index becomes meaningful.
        scratch.extend_from_slice(&(s.initial_pick_count as u64).to_le_bytes());
        scratch.push(s.start_player);
        scratch.push(s.harmony as u8);
        scratch.push(s.middle_kingdom as u8);
        scratch.extend_from_slice(&s.discards[0].to_le_bytes());
        scratch.extend_from_slice(&s.discards[1].to_le_bytes());
        Some(xxhash_rust::xxh3::xxh3_128(scratch))
    }
}

impl RustGameState {
    /// Chance-child expansion for the Rust-hosted search: apply `(placement,
    /// pick)` (which MUST deal), then overwrite the dealt row with `sorted_row`
    /// (deck = pre-deal bag minus row, both sorted).  Rows here are generated
    /// internally from the current bag, so — unlike `SearchEngine::make_with_row`
    /// — no Python-facing validation is done.  `unmake` reverses it via the
    /// boundary snapshot exactly as for a plain `make`.
    fn make_with_row_core(
        &mut self,
        placement: Option<(i8, i8, i8, i8, bool)>,
        pick: Option<u16>,
        sorted_row: &[u16],
    ) -> PyResult<UndoRecord> {
        let mut remaining = self.deck.clone();
        let rec = self.make(placement, pick)?;
        for r in sorted_row {
            if let Some(pos) = remaining.iter().position(|x| x == r) {
                remaining.remove(pos);
            }
        }
        remaining.sort_unstable();
        self.current_row = sorted_row.to_vec();
        self.deck = remaining;
        Ok(rec)
    }
}

// ── NNUE dense evaluator (Step 2b) ───────────────────────────────────────────
// Loads a `.knnue` export (see nnue/export.py) and runs the trained two-head net
// as a leaf `Eval<Kingdomino>`: encode the state actor-relative (the ported
// bit-exact encoder) -> forward -> convert the actor-frame EXPECTED SCORE to a
// player-0-frame value via 2*sigmoid-1, sign-flipped when the actor is not P0.
// Dense (non-incremental) forward — Step 3's accumulator makes the first layer
// fast; this is correctness + integration, not speed.

const KNNUE_MAGIC: &[u8; 4] = b"KNNU"; // exact header[0..4] bytes
const KNNUE_FORMAT_VERSION: u32 = 2;
const KNNUE_HEADER_SIZE: usize = 40;
// Encoder-contract signature this build expects (encoder_signature() in
// nnue/export.py: board dims + channel ordering + flat layout + checkpoint_version).
// Bump BOTH sides together whenever the encoder contract changes — a stale export
// then fails to load loudly instead of feeding the net mislaid-out features.
const KNNUE_EXPECTED_ENCODER_SIG: u64 = 0x0320_d99b_a270_0657;
// Sanity bounds on advertised dimensions (guard against absurd allocations from a
// corrupt header).
const KNNUE_MAX_DIM: usize = 1 << 24;

struct NnueWeights {
    input_dim: usize,
    acc_width: usize,
    tail_hidden: usize,
    margin_scale: f32,
    acc_w: Vec<f32>, // (acc_width, input_dim) row-major
    acc_b: Vec<f32>,
    t0_w: Vec<f32>, // (tail_hidden, acc_width)
    t0_b: Vec<f32>,
    t1_w: Vec<f32>, // (tail_hidden, tail_hidden)
    t1_b: Vec<f32>,
    out_w: Vec<f32>, // (1, tail_hidden)
    out_b: f32,
    mgn_w: Vec<f32>, // (1, tail_hidden)
    mgn_b: f32,
}

impl NnueWeights {
    fn load(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| PyValueError::new_err(format!("nnue load '{path}': {e}")))?;
        if bytes.len() < KNNUE_HEADER_SIZE {
            return Err(PyValueError::new_err(format!(
                "nnue: file too small for the {KNNUE_HEADER_SIZE}-byte header"
            )));
        }
        if &bytes[0..4] != KNNUE_MAGIC {
            return Err(PyValueError::new_err("nnue: bad magic (not a .knnue file)"));
        }
        let u = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        if u(4) != KNNUE_FORMAT_VERSION {
            return Err(PyValueError::new_err(format!(
                "nnue: unsupported format version {} (expected {KNNUE_FORMAT_VERSION})",
                u(4)
            )));
        }
        let input_dim = u(8) as usize;
        let acc_width = u(12) as usize;
        let tail_hidden = u(16) as usize;
        let board_size = u(20) as usize;
        let flat_size = u(24) as usize;
        let margin_scale = f32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
        let encoder_sig = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
        // Encoder-contract guard: the export must have been built against the same
        // encoder this crate expects, else the net is fed mislaid-out features
        // (silent garbage). The signature covers dims + channel order + flat layout
        // + checkpoint_version (see nnue/export.encoder_signature).
        if encoder_sig != KNNUE_EXPECTED_ENCODER_SIG {
            return Err(PyValueError::new_err(format!(
                "nnue: encoder signature mismatch (file 0x{encoder_sig:016x}, \
                 crate expects 0x{KNNUE_EXPECTED_ENCODER_SIG:016x}); re-export against \
                 the current encoder"
            )));
        }
        // Dimension sanity: reject zero/absurd dims and any encoder-layout drift the
        // signature somehow missed, before allocating anything.
        let want_board = N_BOARD_CH * OUT_N * OUT_N;
        if acc_width == 0
            || acc_width > KNNUE_MAX_DIM
            || tail_hidden == 0
            || tail_hidden > KNNUE_MAX_DIM
            || input_dim == 0
            || input_dim > KNNUE_MAX_DIM
        {
            return Err(PyValueError::new_err("nnue: dimension out of range"));
        }
        if board_size != want_board
            || flat_size != FLAT_SIZE
            || input_dim != 2 * board_size + flat_size
        {
            return Err(PyValueError::new_err(format!(
                "nnue: encoder layout mismatch (file board={board_size} flat={flat_size} \
                 input={input_dim}; crate board={want_board} flat={FLAT_SIZE})"
            )));
        }
        if !(margin_scale.is_finite() && margin_scale > 0.0) {
            return Err(PyValueError::new_err(format!(
                "nnue: margin_scale must be finite and positive, got {margin_scale}"
            )));
        }
        let mut off = KNNUE_HEADER_SIZE;
        let read = |off: &mut usize, n: usize| -> PyResult<Vec<f32>> {
            let need = n * 4;
            if *off + need > bytes.len() {
                return Err(PyValueError::new_err("nnue: truncated tensor data"));
            }
            let v = bytes[*off..*off + need]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            *off += need;
            Ok(v)
        };
        let acc_w = read(&mut off, acc_width * input_dim)?;
        let acc_b = read(&mut off, acc_width)?;
        let t0_w = read(&mut off, tail_hidden * acc_width)?;
        let t0_b = read(&mut off, tail_hidden)?;
        let t1_w = read(&mut off, tail_hidden * tail_hidden)?;
        let t1_b = read(&mut off, tail_hidden)?;
        let out_w = read(&mut off, tail_hidden)?;
        let out_b = read(&mut off, 1)?[0];
        let mgn_w = read(&mut off, tail_hidden)?;
        let mgn_b = read(&mut off, 1)?[0];
        if off != bytes.len() {
            return Err(PyValueError::new_err("nnue: trailing bytes after tensors"));
        }
        // Non-finite weights poison the forward pass (NaN value -> the searcher's
        // choose_action finds no best move). Reject at load, not mid-search.
        let all_finite = [&acc_w, &acc_b, &t0_w, &t0_b, &t1_w, &t1_b, &out_w, &mgn_w]
            .iter()
            .all(|v| v.iter().all(|f| f.is_finite()))
            && out_b.is_finite()
            && mgn_b.is_finite();
        if !all_finite {
            return Err(PyValueError::new_err("nnue: non-finite weight/bias value"));
        }
        Ok(NnueWeights {
            input_dim,
            acc_width,
            tail_hidden,
            margin_scale,
            acc_w,
            acc_b,
            t0_w,
            t0_b,
            t1_w,
            t1_b,
            out_w,
            out_b,
            mgn_w,
            mgn_b,
        })
    }

    /// Forward from feature vector `x` (len input_dim) -> (expected_score in [0,1],
    /// margin_normalized). Plain (non-clipped) ReLU on accumulator + tail — matches
    /// training. PyTorch Linear is y = x @ W^T + b with W row-major (out, in).
    fn forward(&self, x: &[f32]) -> (f32, f32) {
        let relu_layer = |w: &[f32], b: &[f32], inp: &[f32], n_out: usize, n_in: usize| {
            let mut out = vec![0f32; n_out];
            for o in 0..n_out {
                let row = &w[o * n_in..(o + 1) * n_in];
                let mut s = b[o];
                for i in 0..n_in {
                    s += row[i] * inp[i];
                }
                out[o] = s.max(0.0);
            }
            out
        };
        let a = relu_layer(&self.acc_w, &self.acc_b, x, self.acc_width, self.input_dim);
        let h0 = relu_layer(&self.t0_w, &self.t0_b, &a, self.tail_hidden, self.acc_width);
        let h1 = relu_layer(
            &self.t1_w,
            &self.t1_b,
            &h0,
            self.tail_hidden,
            self.tail_hidden,
        );
        let mut logit = self.out_b;
        let mut margin = self.mgn_b;
        for i in 0..self.tail_hidden {
            logit += self.out_w[i] * h1[i];
            margin += self.mgn_w[i] * h1[i];
        }
        let expected = 1.0 / (1.0 + (-logit).exp());
        (expected, margin)
    }
}

/// A leaf `Eval<Kingdomino>` backed by the trained dense net.
struct NnueEval {
    w: NnueWeights,
}

impl NnueEval {
    fn features(&self, s: &RustGameState, actor: u8) -> PyResult<Vec<f32>> {
        let board = N_BOARD_CH * OUT_N * OUT_N;
        let mut x = vec![0f32; 2 * board + FLAT_SIZE];
        let (mb, rest) = x.split_at_mut(board);
        let (ob, flat) = rest.split_at_mut(board);
        s.encode_into_slices(actor, mb, ob, flat)?;
        Ok(x)
    }

    /// (p0_value in [-1,1], expected_score in [0,1], margin_points). Actor is read
    /// from the state (valid at any non-terminal — where the searcher calls eval).
    fn value_and_aux(&self, s: &RustGameState) -> PyResult<(f64, f32, f32)> {
        let actor = s.actor()?;
        let x = self.features(s, actor)?;
        let (expected, margin_norm) = self.w.forward(&x);
        let actor_value = 2.0 * expected - 1.0; // expected score [0,1] -> [-1,1]
        let p0 = if actor == 0 {
            actor_value
        } else {
            -actor_value
        };
        Ok((p0 as f64, expected, margin_norm * self.w.margin_scale))
    }
}

impl search::Eval<Kingdomino> for NnueEval {
    fn eval(&self, s: &RustGameState) -> f64 {
        // Only called at non-terminal leaves, so actor()/encode succeed; a failure
        // is a real bug, surfaced as a panic at the FFI boundary.
        self.value_and_aux(s)
            .map(|(p0, _, _)| p0)
            .expect("NnueEval: encode/forward failed at a search leaf")
    }
}

/// Standalone Python handle for the dense NNUE eval — construct from a `.knnue`
/// export and evaluate a state. Used by the PyTorch-equivalence test and shared
/// with `RustSearch` when `eval="nnue"`.
#[pyclass]
struct NnueEvaluator {
    inner: NnueEval,
}

#[pymethods]
impl NnueEvaluator {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Ok(NnueEvaluator {
            inner: NnueEval {
                w: NnueWeights::load(path)?,
            },
        })
    }

    /// (p0_value, expected_score, margin_points) for `state`.
    fn evaluate(&self, state: &RustGameState) -> PyResult<(f64, f64, f64)> {
        let (p0, exp, mgn) = self.inner.value_and_aux(state)?;
        Ok((p0, exp as f64, mgn as f64))
    }
}

/// Rust-hosted depth-limited expectiminimax searcher for Kingdomino (make/unmake,
/// no per-node FFI). A thin Python-facing wrapper over the generic `search`
/// module: it holds a `SearchConfig` + a Kingdomino leaf eval selector and
/// dispatches into `search::value` / `search::choose_action`.
#[pyclass]
struct RustSearch {
    cfg: search::SearchConfig,
    eval: EmmEval,
    nnue: Option<NnueEval>, // Some iff eval == Nnue
    sparse_nnue: Option<Arc<sparse_nnue::SparseNnueWeights>>,
    quantized_sparse_nnue: Option<Arc<sparse_nnue::QuantizedSparseWeights>>,
    #[pyo3(get)]
    nodes: u64,
}

/// Telemetry for one deadline-safe iterative-deepening move. `action` always
/// contains a legal fallback; `value` is absent only when no depth completed
/// (or the move was forced and therefore required no search).
#[pyclass]
struct OperationalSearchReport {
    #[pyo3(get)]
    action: (Option<(i8, i8, i8, i8, bool)>, Option<u16>),
    #[pyo3(get)]
    value: Option<f64>,
    #[pyo3(get)]
    completed_depth: u32,
    #[pyo3(get)]
    timed_out: bool,
    #[pyo3(get)]
    elapsed_secs: f64,
    #[pyo3(get)]
    nodes: u64,
    #[pyo3(get)]
    chance_nodes: u64,
    #[pyo3(get)]
    aspiration_researches: u32,
    #[pyo3(get)]
    star_cutoffs: u64,
    #[pyo3(get)]
    exact_extensions: u64,
    #[pyo3(get)]
    tt_hits: u64,
    #[pyo3(get)]
    tt_cutoffs: u64,
    #[pyo3(get)]
    last_iteration_nodes: u64,
    #[pyo3(get)]
    ordering_evals: u64,
    #[pyo3(get)]
    ordering_actions: u64,
    #[pyo3(get)]
    full_width_ordering: bool,
    #[pyo3(get)]
    selective_pruned: u64,
    #[pyo3(get)]
    selective: bool,
    #[pyo3(get)]
    selective_width: Option<usize>,
    #[pyo3(get)]
    selective_root_width: Option<usize>,
    #[pyo3(get)]
    selective_min_depth: u32,
}

#[pymethods]
impl RustSearch {
    #[new]
    #[pyo3(signature = (depth=4, chance_samples=16, enum_cap=128, eval="pick_blind".to_string(), margin_weight=0.0, seed=0, nnue_path=None))]
    fn new(
        depth: u32,
        chance_samples: usize,
        enum_cap: u64,
        eval: String,
        margin_weight: f64,
        seed: u64,
        nnue_path: Option<String>,
    ) -> PyResult<Self> {
        if depth < 1 || chance_samples < 1 || enum_cap < 1 {
            return Err(PyValueError::new_err(
                "depth, chance_samples, enum_cap must all be >= 1",
            ));
        }
        // A non-finite margin_weight makes every terminal blend NaN, so no action
        // score compares greater than the initial -inf best → the best-action
        // vector stays empty → choose_action panics on index. Reject up front.
        if !margin_weight.is_finite() {
            return Err(PyValueError::new_err("margin_weight must be finite"));
        }
        let eval = match eval.as_str() {
            "pick_blind" | "tanh_margin" => EmmEval::PickBlind,
            "pick_aware" => EmmEval::PickAware,
            "nnue" => EmmEval::Nnue,
            "sparse_nnue_ref" => EmmEval::SparseNnueRef,
            "sparse_nnue" => EmmEval::SparseNnue,
            "sparse_nnue_q" => EmmEval::QuantizedSparseNnue,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown eval '{}' (expected 'pick_blind', 'pick_aware', 'nnue', \
                     'sparse_nnue_ref', 'sparse_nnue', or 'sparse_nnue_q')",
                    other
                )));
            }
        };
        let nnue = match (eval, nnue_path.as_ref()) {
            (EmmEval::Nnue, Some(p)) => Some(NnueEval {
                w: NnueWeights::load(p)?,
            }),
            (EmmEval::Nnue, None) => {
                return Err(PyValueError::new_err("eval='nnue' requires nnue_path"));
            }
            (_, None) => None,
            (_, Some(_)) => None,
        };
        let sparse_nnue = match (eval, nnue_path.as_ref()) {
            (EmmEval::SparseNnue | EmmEval::SparseNnueRef, Some(p)) => {
                Some(Arc::new(sparse_nnue::SparseNnueWeights::load(p)?))
            }
            (EmmEval::SparseNnue | EmmEval::SparseNnueRef, None) => {
                return Err(PyValueError::new_err("sparse NNUE eval requires nnue_path"));
            }
            (EmmEval::Nnue | EmmEval::QuantizedSparseNnue, _) => None,
            (_, Some(_)) => {
                return Err(PyValueError::new_err(
                    "nnue_path given but eval is not an NNUE evaluator",
                ));
            }
            (_, None) => None,
        };
        let quantized_sparse_nnue = match (eval, nnue_path.as_ref()) {
            (EmmEval::QuantizedSparseNnue, Some(p)) => {
                Some(Arc::new(sparse_nnue::QuantizedSparseWeights::load(p)?))
            }
            (EmmEval::QuantizedSparseNnue, None) => {
                return Err(PyValueError::new_err(
                    "quantized sparse NNUE eval requires nnue_path",
                ));
            }
            _ => None,
        };
        Ok(RustSearch {
            cfg: search::SearchConfig {
                depth,
                chance_samples,
                enum_cap,
                margin_weight,
                seed,
            },
            eval,
            nnue,
            sparse_nnue,
            quantized_sparse_nnue,
            nodes: 0,
        })
    }

    /// Root player-0-frame value of `state` at `depth` (default the configured
    /// depth).  Searches a fresh clone; `self.nodes` is set to the node count.
    #[pyo3(signature = (state, depth=None))]
    fn value(&mut self, state: &RustGameState, depth: Option<i32>) -> PyResult<f64> {
        let mut s = state.cloned();
        let d = depth.unwrap_or(self.cfg.depth as i32);
        let mut nodes = 0u64;
        let (a, b) = (f64::NEG_INFINITY, f64::INFINITY);
        let v = match self.eval {
            EmmEval::PickBlind => search::value::<Kingdomino, _>(
                &mut s,
                d,
                a,
                b,
                &PickBlindEval,
                &self.cfg,
                &mut nodes,
            )?,
            EmmEval::PickAware => search::value::<Kingdomino, _>(
                &mut s,
                d,
                a,
                b,
                &PickAwareEval,
                &self.cfg,
                &mut nodes,
            )?,
            EmmEval::Nnue => {
                let e = self.nnue.as_ref().expect("nnue eval without weights");
                search::value::<Kingdomino, _>(&mut s, d, a, b, e, &self.cfg, &mut nodes)?
            }
            EmmEval::SparseNnueRef => {
                let e = sparse_nnue::SparseStatelessEval {
                    weights: Arc::clone(
                        self.sparse_nnue
                            .as_ref()
                            .expect("sparse nnue without weights"),
                    ),
                };
                search::value::<Kingdomino, _>(&mut s, d, a, b, &e, &self.cfg, &mut nodes)?
            }
            EmmEval::SparseNnue => {
                let weights = Arc::clone(
                    self.sparse_nnue
                        .as_ref()
                        .expect("sparse nnue without weights"),
                );
                let mut state = sparse_nnue::SparseSearchState::new(s, weights)?;
                search::value::<sparse_nnue::SparseKingdomino, _>(
                    &mut state,
                    d,
                    a,
                    b,
                    &sparse_nnue::SparseIncrementalEval,
                    &self.cfg,
                    &mut nodes,
                )?
            }
            EmmEval::QuantizedSparseNnue => {
                let weights = Arc::clone(
                    self.quantized_sparse_nnue
                        .as_ref()
                        .expect("quantized sparse nnue without weights"),
                );
                let mut state = sparse_nnue::QuantizedSparseSearchState::new(s, weights)?;
                search::value::<sparse_nnue::QuantizedSparseKingdomino, _>(
                    &mut state,
                    d,
                    a,
                    b,
                    &sparse_nnue::QuantizedSparseEval,
                    &self.cfg,
                    &mut nodes,
                )?
            }
        };
        self.nodes = nodes;
        Ok(v)
    }

    /// Best action for the side to move (player 0 maximizes the value, player 1
    /// minimizes it), searched at the configured depth.  Each root child gets a
    /// full (-inf, inf) window (no root-sibling pruning).  Ties broken by `seed`
    /// (deterministic first-best when `seed` is None).  `self.nodes` is set to the
    /// node count (0 for a forced single-action move — no search performed).
    #[pyo3(signature = (state, seed=None))]
    fn choose_action(
        &mut self,
        state: &RustGameState,
        seed: Option<u64>,
    ) -> PyResult<(Option<(i8, i8, i8, i8, bool)>, Option<u16>)> {
        let mut s = state.cloned();
        let mut nodes = 0u64;
        let chosen = match self.eval {
            EmmEval::PickBlind => search::choose_action::<Kingdomino, _>(
                &mut s,
                &PickBlindEval,
                &self.cfg,
                seed,
                &mut nodes,
            )?,
            EmmEval::PickAware => search::choose_action::<Kingdomino, _>(
                &mut s,
                &PickAwareEval,
                &self.cfg,
                seed,
                &mut nodes,
            )?,
            EmmEval::Nnue => {
                let e = self.nnue.as_ref().expect("nnue eval without weights");
                search::choose_action::<Kingdomino, _>(&mut s, e, &self.cfg, seed, &mut nodes)?
            }
            EmmEval::SparseNnueRef => {
                let e = sparse_nnue::SparseStatelessEval {
                    weights: Arc::clone(
                        self.sparse_nnue
                            .as_ref()
                            .expect("sparse nnue without weights"),
                    ),
                };
                search::choose_action::<Kingdomino, _>(&mut s, &e, &self.cfg, seed, &mut nodes)?
            }
            EmmEval::SparseNnue => {
                let weights = Arc::clone(
                    self.sparse_nnue
                        .as_ref()
                        .expect("sparse nnue without weights"),
                );
                let mut state = sparse_nnue::SparseSearchState::new(s, weights)?;
                search::choose_action::<sparse_nnue::SparseKingdomino, _>(
                    &mut state,
                    &sparse_nnue::SparseIncrementalEval,
                    &self.cfg,
                    seed,
                    &mut nodes,
                )?
            }
            EmmEval::QuantizedSparseNnue => {
                let weights = Arc::clone(
                    self.quantized_sparse_nnue
                        .as_ref()
                        .expect("quantized sparse nnue without weights"),
                );
                let mut state = sparse_nnue::QuantizedSparseSearchState::new(s, weights)?;
                search::choose_action::<sparse_nnue::QuantizedSparseKingdomino, _>(
                    &mut state,
                    &sparse_nnue::QuantizedSparseEval,
                    &self.cfg,
                    seed,
                    &mut nodes,
                )?
            }
        };
        self.nodes = nodes;
        chosen
            .ok_or_else(|| PyValueError::new_err("choose_action on a state with no legal actions"))
    }

    /// Operational bot path: iterative deepening under one shared deadline.
    /// Root/PV-first ordering, root-sibling alpha/beta reuse, aspiration windows,
    /// bounded Star1 chance pruning, and deterministic exact-tail extensions are
    /// all enabled. On timeout the last fully completed depth is returned.
    #[pyo3(signature = (state, max_secs=1.0, max_depth=None, aspiration_window=0.25, max_nodes=None, full_width_ordering=false, selective_width=None, selective_root_width=None, selective_min_depth=4))]
    fn choose_action_timed(
        &mut self,
        state: &RustGameState,
        max_secs: f64,
        max_depth: Option<u32>,
        aspiration_window: f64,
        max_nodes: Option<u64>,
        full_width_ordering: bool,
        selective_width: Option<usize>,
        selective_root_width: Option<usize>,
        selective_min_depth: u32,
    ) -> PyResult<OperationalSearchReport> {
        if !max_secs.is_finite() || max_secs <= 0.0 {
            return Err(PyValueError::new_err("max_secs must be finite and > 0"));
        }
        let max_depth = max_depth.unwrap_or(self.cfg.depth);
        if max_depth < 1 {
            return Err(PyValueError::new_err("max_depth must be >= 1"));
        }
        if !aspiration_window.is_finite() || aspiration_window <= 0.0 {
            return Err(PyValueError::new_err(
                "aspiration_window must be finite and > 0",
            ));
        }
        if max_nodes == Some(0) {
            return Err(PyValueError::new_err("max_nodes must be >= 1"));
        }
        if selective_width == Some(0) {
            return Err(PyValueError::new_err("selective_width must be >= 1"));
        }
        if selective_root_width == Some(0) {
            return Err(PyValueError::new_err("selective_root_width must be >= 1"));
        }
        if selective_root_width.is_some() && selective_width.is_none() {
            return Err(PyValueError::new_err(
                "selective_root_width requires selective_width",
            ));
        }
        if selective_min_depth < 1 {
            return Err(PyValueError::new_err("selective_min_depth must be >= 1"));
        }

        let start = std::time::Instant::now();
        let limits = search::OperationalLimits {
            max_depth,
            deadline: start + std::time::Duration::from_secs_f64(max_secs),
            aspiration_window,
            node_limit: max_nodes,
            value_bound: 1.0 + self.cfg.margin_weight.abs(),
            full_width_ordering,
            selective_width,
            selective_root_width,
            selective_min_depth,
        };
        let result = match self.eval {
            EmmEval::PickBlind => {
                let mut s = state.cloned();
                search::choose_action_operational::<Kingdomino, _>(
                    &mut s,
                    &PickBlindEval,
                    &self.cfg,
                    &limits,
                )?
            }
            EmmEval::PickAware => {
                let mut s = state.cloned();
                search::choose_action_operational::<Kingdomino, _>(
                    &mut s,
                    &PickAwareEval,
                    &self.cfg,
                    &limits,
                )?
            }
            EmmEval::Nnue => {
                let mut s = state.cloned();
                let eval = self.nnue.as_ref().expect("nnue eval without weights");
                search::choose_action_operational::<Kingdomino, _>(
                    &mut s, eval, &self.cfg, &limits,
                )?
            }
            EmmEval::SparseNnueRef => {
                let mut s = state.cloned();
                let eval = sparse_nnue::SparseStatelessEval {
                    weights: Arc::clone(
                        self.sparse_nnue
                            .as_ref()
                            .expect("sparse nnue without weights"),
                    ),
                };
                search::choose_action_operational::<Kingdomino, _>(
                    &mut s, &eval, &self.cfg, &limits,
                )?
            }
            EmmEval::SparseNnue => {
                let weights = Arc::clone(
                    self.sparse_nnue
                        .as_ref()
                        .expect("sparse nnue without weights"),
                );
                let mut s = sparse_nnue::SparseSearchState::new(state.cloned(), weights)?;
                search::choose_action_operational::<sparse_nnue::SparseKingdomino, _>(
                    &mut s,
                    &sparse_nnue::SparseIncrementalEval,
                    &self.cfg,
                    &limits,
                )?
            }
            EmmEval::QuantizedSparseNnue => {
                let weights = Arc::clone(
                    self.quantized_sparse_nnue
                        .as_ref()
                        .expect("quantized sparse nnue without weights"),
                );
                let mut s = sparse_nnue::QuantizedSparseSearchState::new(state.cloned(), weights)?;
                search::choose_action_operational::<sparse_nnue::QuantizedSparseKingdomino, _>(
                    &mut s,
                    &sparse_nnue::QuantizedSparseEval,
                    &self.cfg,
                    &limits,
                )?
            }
        }
        .ok_or_else(|| {
            PyValueError::new_err("choose_action_timed on a state with no legal actions")
        })?;
        self.nodes = result.nodes;
        Ok(OperationalSearchReport {
            action: result.action,
            value: result.value,
            completed_depth: result.completed_depth,
            timed_out: result.timed_out,
            elapsed_secs: start.elapsed().as_secs_f64(),
            nodes: result.nodes,
            chance_nodes: result.chance_nodes,
            aspiration_researches: result.aspiration_researches,
            star_cutoffs: result.star_cutoffs,
            exact_extensions: result.exact_extensions,
            tt_hits: result.tt_hits,
            tt_cutoffs: result.tt_cutoffs,
            last_iteration_nodes: result.last_iteration_nodes,
            ordering_evals: result.ordering_evals,
            ordering_actions: result.ordering_actions,
            full_width_ordering,
            selective_pruned: result.selective_pruned,
            selective: result.selective,
            selective_width,
            selective_root_width,
            selective_min_depth,
        })
    }
}

#[pymethods]
impl RustGameState {
    /// Build a fresh INITIAL_SELECTION state.  `deck` and `current_row` are the
    /// post-deal lists (deck already missing the four row tiles), so a caller
    /// mirrors a Python GameState by passing state.deck and state.current_row.
    #[new]
    #[pyo3(signature = (start_player, deck, current_row, harmony=true, middle_kingdom=true))]
    fn new(
        start_player: u8,
        deck: Vec<u16>,
        current_row: Vec<u16>,
        harmony: bool,
        middle_kingdom: bool,
    ) -> Self {
        RustGameState {
            boards: [RustBoard::new(7, 7), RustBoard::new(7, 7)],
            deck,
            current_row,
            pending_claims: Vec::new(),
            next_claims: Vec::new(),
            phase: INITIAL_SELECTION,
            actor_index: 0,
            initial_pick_count: 0,
            start_player,
            harmony,
            middle_kingdom,
            discards: [0, 0],
        }
    }

    /// Build a RustGameState from an arbitrary Python GameState snapshot.
    /// Board arrays are flat row-major 15x15 terrain/crown vectors.
    #[staticmethod]
    #[pyo3(signature = (
        deck,
        current_row,
        pending_claims,
        next_claims,
        phase,
        actor_index,
        initial_pick_count,
        start_player,
        board0_terrain,
        board0_crowns,
        board1_terrain,
        board1_crowns,
        harmony=true,
        middle_kingdom=true,
        castle_x=7,
        castle_y=7,
        discards=(0, 0)
    ))]
    fn from_parts(
        deck: Vec<u16>,
        current_row: Vec<u16>,
        pending_claims: Vec<(u8, u16)>,
        next_claims: Vec<(u8, u16)>,
        phase: u8,
        actor_index: usize,
        initial_pick_count: usize,
        start_player: u8,
        board0_terrain: Vec<u8>,
        board0_crowns: Vec<u8>,
        board1_terrain: Vec<u8>,
        board1_crowns: Vec<u8>,
        harmony: bool,
        middle_kingdom: bool,
        castle_x: i8,
        castle_y: i8,
        discards: (u32, u32),
    ) -> PyResult<Self> {
        if phase > GAME_OVER {
            return Err(PyValueError::new_err(format!("invalid phase {phase}")));
        }
        if start_player > 1 {
            return Err(PyValueError::new_err(format!(
                "invalid start_player {start_player}"
            )));
        }
        if phase == INITIAL_SELECTION && initial_pick_count >= 4 {
            return Err(PyValueError::new_err(format!(
                "INITIAL_SELECTION initial_pick_count must be < 4, got {initial_pick_count}"
            )));
        }
        if (phase == PLACE_AND_SELECT || phase == FINAL_PLACEMENT)
            && actor_index >= pending_claims.len()
        {
            return Err(PyValueError::new_err(format!(
                "actor_index {actor_index} outside pending_claims length {}",
                pending_claims.len()
            )));
        }

        Ok(RustGameState {
            boards: [
                RustBoard::from_flat_parts(board0_terrain, board0_crowns, castle_x, castle_y)?,
                RustBoard::from_flat_parts(board1_terrain, board1_crowns, castle_x, castle_y)?,
            ],
            deck,
            current_row,
            pending_claims,
            next_claims,
            phase,
            actor_index,
            initial_pick_count,
            start_player,
            harmony,
            middle_kingdom,
            discards: [discards.0, discards.1],
        })
    }

    /// Apply one action, returning a new state (the receiver is unchanged).
    ///
    /// Action encoding mirrors `step`'s two arguments, disambiguated by phase:
    ///   INITIAL_SELECTION : placement=None, pick=Some(claimed domino id)
    ///   PLACE_AND_SELECT  : placement=Some(p)|None(discard), pick=Some(next id)
    ///   FINAL_PLACEMENT   : placement=Some(p)|None(discard), pick=None
    /// A placement tuple is (x1, y1, x2, y2, flipped) in canvas coords.
    fn step(
        &self,
        placement: Option<(i8, i8, i8, i8, bool)>,
        pick_domino_id: Option<u16>,
    ) -> PyResult<RustGameState> {
        let mut s = self.cloned();
        match s.phase {
            INITIAL_SELECTION => {
                if placement.is_some() {
                    return Err(PyValueError::new_err(
                        "INITIAL_SELECTION takes a pick only, no placement",
                    ));
                }
                let d = pick_domino_id
                    .ok_or_else(|| PyValueError::new_err("INITIAL_SELECTION requires a pick"))?;
                let pos = s
                    .current_row
                    .iter()
                    .position(|&x| x == d)
                    .ok_or_else(|| PyValueError::new_err("Picked domino not available"))?;
                let actor = s.actor()?; // attributed before the count increments
                s.current_row.remove(pos);
                s.next_claims.push((actor, d));
                s.initial_pick_count += 1;
                if s.initial_pick_count == 4 {
                    s.next_claims.sort_by_key(|c| c.1);
                    s.pending_claims = std::mem::take(&mut s.next_claims);
                    s.deal_row();
                    s.actor_index = 0;
                    s.phase = PLACE_AND_SELECT;
                }
                Ok(s)
            }
            PLACE_AND_SELECT | FINAL_PLACEMENT => {
                let (player, domino_id) = s.pending_claims[s.actor_index];
                if let Some((x1, y1, x2, y2, flipped)) = placement {
                    let (ta, ca, tb, cb) = dom(domino_id);
                    s.boards[player as usize].place(ta, ca, tb, cb, x1, y1, x2, y2, flipped)?;
                } else {
                    // Forced discard: the claimed tile had no legal placement.
                    s.discards[player as usize] += 1;
                }
                if s.phase == PLACE_AND_SELECT {
                    let pick = pick_domino_id
                        .ok_or_else(|| PyValueError::new_err("PLACE_AND_SELECT requires a pick"))?;
                    let pos = s
                        .current_row
                        .iter()
                        .position(|&x| x == pick)
                        .ok_or_else(|| PyValueError::new_err("Picked domino not available"))?;
                    s.current_row.remove(pos);
                    s.next_claims.push((player, pick));
                }
                s.actor_index += 1;
                if s.actor_index >= s.pending_claims.len() {
                    if s.phase == FINAL_PLACEMENT {
                        s.phase = GAME_OVER;
                    } else {
                        s.advance_round();
                    }
                }
                Ok(s)
            }
            _ => Err(PyValueError::new_err("Cannot step a terminal state")),
        }
    }

    /// Legal actions as (placement, pick) tuples (same encoding as `step`), in
    /// canonical ascending joint-index order — identical to the Python engine's
    /// ordering, so the search tree's child iteration is deterministic.
    fn legal_actions(&self) -> Vec<(Option<(i8, i8, i8, i8, bool)>, Option<u16>)> {
        self.legal_actions_indexed()
            .into_iter()
            .map(|(_, p, pk)| (p, pk))
            .collect()
    }

    /// Joint indices of all legal actions, ascending (the canonical order).
    fn legal_action_indices(&self) -> Vec<u16> {
        self.legal_actions_indexed()
            .into_iter()
            .map(|(i, _, _)| i)
            .collect()
    }

    /// Boolean legal-action mask as a (3390,) bool numpy array.
    /// Equivalent to action_codec.legal_mask(state) but avoids the Python
    /// encode_action loop — uses the already-computed joint indices directly.
    fn legal_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let mut mask = vec![false; POLICY_SIZE];
        if self.phase == GAME_OVER {
            return Ok(Array1::from_vec(mask).into_pyarray(py));
        }
        for idx in self.legal_action_indices() {
            if mask[idx as usize] {
                return Err(PyValueError::new_err(format!(
                    "Action collision at joint idx {idx} — indexing bug in codec"
                )));
            }
            mask[idx as usize] = true;
        }
        Ok(Array1::from_vec(mask).into_pyarray(py))
    }

    /// Joint action index for one (placement, pick) action, mirroring
    /// action_codec.encode_action.  Errors if the action doesn't encode.
    fn encode_action(
        &self,
        placement: Option<(i8, i8, i8, i8, bool)>,
        pick: Option<u16>,
    ) -> PyResult<u16> {
        self.enc_action(placement, pick).ok_or_else(|| {
            PyValueError::new_err(
                "action does not encode (placement out of crop, or pick not in current_row)",
            )
        })
    }

    /// Encode this state from `player`'s perspective, mirroring
    /// encoder.encode_state.  Returns (my_board, opp_board, flat) as numpy
    /// arrays: two (9, 13, 13) float32 plane stacks and a flat float32 vector.
    /// Errors on a terminal state (matches Python).
    fn encode<'py>(
        &self,
        py: Python<'py>,
        player: u8,
    ) -> PyResult<(
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        let (my_board, opp_board, flat) = self.encode_arrays(player)?;
        Ok((
            my_board.into_pyarray(py),
            opp_board.into_pyarray(py),
            flat.into_pyarray(py),
        ))
    }

    /// Frozen Step-3 NNUE reference features for this state and perspective.
    ///
    /// Returns `(sparse_indices, summary)`: sorted unique int32 indices in the
    /// 5,710-feature public-state core plus the 171-value float32 summary.  This
    /// stateless path is the correctness oracle for the later incremental
    /// accumulator and is intentionally defined for terminal states too.
    fn nnue_features<'py>(
        &self,
        py: Python<'py>,
        player: u8,
    ) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f32>>)> {
        let sparse = nnue_features::sparse_indices(self, player).map_err(PyValueError::new_err)?;
        let summary = nnue_features::summary(self, player).map_err(PyValueError::new_err)?;
        Ok((
            Array1::from_vec(sparse).into_pyarray(py),
            Array1::from_vec(summary).into_pyarray(py),
        ))
    }

    // ── read-only accessors (used by the lockstep equivalence test) ──
    #[getter]
    fn phase(&self) -> u8 {
        self.phase
    }

    fn current_actor(&self) -> PyResult<u8> {
        self.actor()
    }

    #[getter]
    fn actor_index(&self) -> usize {
        self.actor_index
    }

    #[getter]
    fn initial_pick_count(&self) -> usize {
        self.initial_pick_count
    }

    #[getter]
    fn start_player(&self) -> u8 {
        self.start_player
    }

    fn current_row(&self) -> Vec<u16> {
        self.current_row.clone()
    }

    fn deck(&self) -> Vec<u16> {
        self.deck.clone()
    }

    /// Versioned canonical public-state identity used by the chance-correct
    /// Python/Rust parity gate. The bytes deliberately exclude hidden deck order
    /// and Python's display-only board domino-id history. See
    /// `denial_search.chance_public_state_key_v1` for the mirrored layout.
    fn chance_public_state_key_v1(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1024);
        chance_public_state_key_v1_bytes(self, &mut out);
        out
    }

    /// Expose the production expectiminimax chance distribution for parity
    /// tests and the forthcoming observation-split search. Rows are enumerated
    /// exactly when C(n,4) <= enum_cap, otherwise IID sampled; all returned
    /// weights are the actual backup probabilities.
    #[pyo3(signature = (enum_cap=4096, chance_samples=16, seed=0))]
    fn chance_outcomes(
        &self,
        enum_cap: u64,
        chance_samples: usize,
        seed: u64,
    ) -> PyResult<Vec<(Vec<u16>, f64)>> {
        if self.deck.len() < 4 || self.deck.len() % 4 != 0 {
            return Err(PyValueError::new_err(format!(
                "chance_outcomes requires a non-empty pre-reveal bag divisible by four, got {}",
                self.deck.len()
            )));
        }
        if enum_cap == 0 || chance_samples == 0 {
            return Err(PyValueError::new_err(
                "chance_outcomes requires enum_cap > 0 and chance_samples > 0",
            ));
        }
        let cfg = search::SearchConfig {
            depth: 1,
            chance_samples,
            enum_cap,
            margin_weight: 0.0,
            seed,
        };
        let outcomes = <Kingdomino as search::Game>::chance_children(self, (None, None), &cfg);
        let total = outcomes.iter().map(|(_, weight)| *weight).sum::<f64>();
        if outcomes.is_empty()
            || outcomes
                .iter()
                .any(|(row, weight)| row.len() != 4 || !weight.is_finite() || *weight <= 0.0)
            || (total - 1.0).abs() > 1e-12
        {
            return Err(PyValueError::new_err(
                "chance_outcomes produced an invalid probability distribution",
            ));
        }
        Ok(outcomes)
    }

    /// Per-player forced-discard counts (player0, player1).
    fn discards(&self) -> (u32, u32) {
        (self.discards[0], self.discards[1])
    }

    fn pending_claims(&self) -> Vec<(u8, u16)> {
        self.pending_claims.clone()
    }

    fn next_claims(&self) -> Vec<(u8, u16)> {
        self.next_claims.clone()
    }

    /// Return a copy with the hidden deck reshuffled (mirrors
    /// encoder.redeterminize): public information is unchanged (boards, row,
    /// claims, phase), only the order of future tile reveals changes.  Call at
    /// the root of each search to close the information-set loop.  `seed` makes
    /// the reshuffle reproducible (None = entropy).
    #[pyo3(signature = (seed=None))]
    fn redeterminize(&self, seed: Option<u64>) -> RustGameState {
        let mut s = self.cloned();
        let mut rng = match seed {
            Some(x) => StdRng::seed_from_u64(x),
            None => StdRng::from_entropy(),
        };
        s.deck.shuffle(&mut rng);
        s
    }

    /// Totals (territory + harmony + middle-kingdom) for (player0, player1).
    fn scores(&self) -> (i32, i32) {
        let a = self.boards[0].score(self.harmony, self.middle_kingdom);
        let b = self.boards[1].score(self.harmony, self.middle_kingdom);
        (a.0 + a.1 + a.2, b.0 + b.1 + b.2)
    }

    /// Final-score components for auxiliary NNUE training targets.
    ///
    /// Each player tuple is `(total, territory_score, largest_territory_size,
    /// total_crowns, harmony_bonus, middle_kingdom_bonus)`.  This is valid for
    /// any state, although the replay dataset calls it only after game-over.
    fn score_breakdowns(
        &self,
    ) -> (
        (i32, i32, i32, i32, i32, i32),
        (i32, i32, i32, i32, i32, i32),
    ) {
        let one = |board: &RustBoard| {
            let (territory, harmony_bonus, middle_bonus) =
                board.score(self.harmony, self.middle_kingdom);
            let (total, largest, total_crowns) =
                board.cascade_key(self.harmony, self.middle_kingdom);
            (
                total,
                territory,
                largest,
                total_crowns,
                harmony_bonus,
                middle_bonus,
            )
        };
        (one(&self.boards[0]), one(&self.boards[1]))
    }

    /// Flat 225-cell terrain map (idx = y*15 + x) for one player's board.
    fn board_terrain(&self, player: usize) -> Vec<u8> {
        self.boards[player].terrain.to_vec()
    }

    /// Flat 225-cell crown map for one player's board.
    fn board_crowns(&self, player: usize) -> Vec<u8> {
        self.boards[player].crowns.to_vec()
    }

    /// Benchmark-only: run the alpha-beta solver to completion (or until
    /// `max_secs` wall-clock elapses) and return (value, fully_solved,
    /// elapsed_secs).
    ///
    /// Unlike `exact_endgame_value_no_chance`, this is intended for measuring the
    /// real solve-time distribution. `max_secs` should be set high (e.g. 60.0) as
    /// a safety ceiling, not a routine budget. `fully_solved` is False only if the
    /// deadline was hit.
    ///
    /// Uses the SAME solver (alpha-beta + move ordering) as production, so timings
    /// reflect production pruning behavior. `alpha` defaults to 0.5 (the training
    /// frame): alpha-beta cutoffs depend on leaf values, so pruning — and
    /// therefore solve time — can vary with alpha. Measure at the alpha training
    /// actually uses.
    ///
    /// `parallel=True` (default) uses the YBW parallel solver
    /// (`solve_endgame_ab_parallel`) to measure wall-clock; `parallel=False` uses
    /// the serial solver — use that to compare single-core solve times.
    #[pyo3(signature = (max_secs=60.0, score_scale=160.0, margin_gain=2.0, alpha=0.5, parallel=true, ordering="lookahead2_clustered"))]
    fn measure_endgame_tree(
        &self,
        max_secs: f64,
        score_scale: f64,
        margin_gain: f64,
        alpha: f64,
        parallel: bool,
        ordering: &str,
    ) -> PyResult<(f64, bool, f64)> {
        if self.phase == GAME_OVER {
            return Err(PyValueError::new_err("Cannot measure a terminal state"));
        }
        if self.deck.len() > 4 {
            return Err(PyValueError::new_err(format!(
                "deck.len()={} > 4; measure only supports no-chance endgames (deck <= 4)",
                self.deck.len()
            )));
        }
        if !is_no_chance_endgame_state(self) {
            return Err(PyValueError::new_err(
                "measure_endgame_tree requires a no-chance endgame state (deck in {0,4})",
            ));
        }
        let mode = SolverOrderMode::from_str(ordering)?;
        let start = std::time::Instant::now();
        let deadline = start + std::time::Duration::from_secs_f64(max_secs);
        let raw = if parallel {
            solve_endgame_ab_parallel(self, deadline, mode)?
        } else {
            solve_endgame_ab(self, deadline, MARGIN_LO, MARGIN_HI, mode, 0)?
        };
        let (value, solved) = match raw {
            Some(solver_utility) => (
                solver_utility_to_training_value(solver_utility, score_scale, margin_gain, alpha),
                true,
            ),
            None => (0.0, false),
        };
        let elapsed_secs = start.elapsed().as_secs_f64();
        Ok((value, solved, elapsed_secs))
    }
}

// ─── AlphaZero MCTS (arena tree) ─────────────────────────────────────────────
// Ports mcts_az.py.  Fixed player-0 value frame throughout: every value_sum is
// from player 0's perspective regardless of who is acting; selection re-frames
// to the acting player's view.  f64 is used for value_sum/prior/PUCT (NOT the
// f32 sketched in the spec) so accumulation matches Python's float64 exactly —
// required for the bit-identical mock-evaluator gate.
//
// The only Python boundary during search is leaf evaluation: Rust hands the
// encoded leaf (mb/ob/flat numpy arrays + legal joint indices) to a Python
// evaluator and gets back (values, gathered_logits).  States, actions, and all
// per-node data stay in the Rust arena.

/// One search-tree node in the arena.  Edge stats live in the child node.
struct Node {
    prior: f64,
    visit_count: i32,
    value_sum: f64,                                        // PLAYER-0 frame
    virtual_loss: i32, // unused by serial _simulate; for the batched path later
    children: Vec<(u16, u32)>, // (joint_index, child node id), ascending index
    state: Option<RustGameState>, // set lazily on first descent / at root
    action: (Option<(i8, i8, i8, i8, bool)>, Option<u16>), // move from parent
    is_expanded: bool,
}

impl Node {
    fn new(prior: f64, action: (Option<(i8, i8, i8, i8, bool)>, Option<u16>)) -> Self {
        Node {
            prior,
            visit_count: 0,
            value_sum: 0.0,
            virtual_loss: 0,
            children: Vec::new(),
            state: None,
            action,
            is_expanded: false,
        }
    }
}

/// Terminal backup value in player-0 frame, using the SAME win-gated formula as
/// non-terminal leaf values (Fix 1).  Replaces mcts_compute_target_z (whose
/// tanh(margin/30) scale was inconsistent with the non-terminal estimates).
///
///     win_value  = +1 win / 0 draw / -1 loss   (official outcome cascade)
///     win_gate   = win_value^4  (= 1 for a decided game, 0 for a draw)
///     value      = (1 - alpha) * win_value + alpha * win_gate * margin_value
///
/// alpha is the reserved margin band B.  win_gate is computed as (w*w)*(w*w) so
/// the f64 op order matches Python's, keeping Rust and Python terminal values
/// bit-identical for the same terminal state.  score_scale / margin_gain / alpha
/// must match the Python config (SCORE_SCALE=160.0, MARGIN_GAIN=2.0, ALPHA=0.5).
fn terminal_search_value(
    state: &RustGameState,
    score_scale: f64,
    margin_gain: f64,
    alpha: f64,
) -> f64 {
    let (s0, s1) = state.scores();
    let own_norm = s0 as f64 / score_scale;
    let opp_norm = s1 as f64 / score_scale;
    let margin_value = ((own_norm - opp_norm) * margin_gain).tanh();
    let win_value = state.official_outcome_i8() as f64;
    let win_gate = win_value * win_value;
    let win_gate = win_gate * win_gate; // win_value^4 (n=4 win-certainty gate)
    (1.0 - alpha) * win_value + alpha * win_gate * margin_value
}

/// Convert a raw score margin (s0 - s1, player-0 frame) into the training value.
/// Kept as the score-margin formula helper and Python test export. Exact solving
/// uses `solver_utility_to_training_value`, which additionally preserves the
/// official tiebreak outcome at a zero raw margin.
fn margin_to_training_value(margin: f64, score_scale: f64, margin_gain: f64, alpha: f64) -> f64 {
    let win_value = if margin > 0.0 {
        1.0
    } else if margin < 0.0 {
        -1.0
    } else {
        0.0
    };
    let margin_value = (margin / score_scale * margin_gain).tanh();
    let win_gate = win_value * win_value;
    let win_gate = win_gate * win_gate; // win_value^4 (n=4 win-certainty gate)
    (1.0 - alpha) * win_value + alpha * win_gate * margin_value
}

/// Solver-only scalar that preserves the official lexicographic outcome.
///
/// Non-zero integer score margins retain their ordinary value. On an equal raw
/// score, +/-0.25 orders an official tiebreak win/loss around a true draw while
/// remaining strictly between the adjacent integer margins (-1 and +1). This
/// lets the existing scalar alpha-beta and transposition machinery optimize the
/// official cascade without pretending that the tiebreak is a score margin.
const OFFICIAL_TIEBREAK_UTILITY: f64 = 0.25;

fn terminal_solver_utility(state: &RustGameState) -> f64 {
    let (s0, s1) = state.scores();
    let margin = s0 - s1;
    if margin != 0 {
        margin as f64
    } else {
        state.official_outcome_i8() as f64 * OFFICIAL_TIEBREAK_UTILITY
    }
}

/// Convert the solver ordering scalar back to the training value. A +/-0.25
/// tiebreak result has a decisive official outcome but a zero raw-score margin.
fn solver_utility_to_training_value(
    utility: f64,
    score_scale: f64,
    margin_gain: f64,
    alpha: f64,
) -> f64 {
    let win_value = if utility > 0.0 {
        1.0
    } else if utility < 0.0 {
        -1.0
    } else {
        0.0
    };
    let raw_margin = if utility.abs() < 0.5 { 0.0 } else { utility };
    let margin_value = (raw_margin / score_scale * margin_gain).tanh();
    let win_gate = win_value * win_value;
    let win_gate = win_gate * win_gate;
    (1.0 - alpha) * win_value + alpha * win_gate * margin_value
}

/// Full-window sentinel for the solver utility. Raw margins live in ~[-80, 80]
/// and the tie sentinels in [-0.25, 0.25], so ±200 brackets every value.
const MARGIN_LO: f64 = -200.0;
const MARGIN_HI: f64 = 200.0;

fn is_no_chance_endgame_state(state: &RustGameState) -> bool {
    match state.phase {
        GAME_OVER => true,
        PLACE_AND_SELECT => state.deck.is_empty() || state.deck.len() == 4,
        FINAL_PLACEMENT => state.deck.is_empty(),
        _ => false,
    }
}

fn exact_count_no_chance_bounded(state: &RustGameState, cap: u64) -> PyResult<u64> {
    if state.phase == GAME_OVER {
        return Ok(0);
    }
    if !is_no_chance_endgame_state(state) {
        return Ok(cap.saturating_add(1));
    }

    let legal = state.legal_actions_indexed();
    let mut total = 1u64;
    for &(_idx, placement, pick) in &legal {
        let child = state.step(placement, pick)?;
        let child_count = exact_count_no_chance_bounded(&child, cap.saturating_sub(total))?;
        total = total.saturating_add(child_count.max(1));
        if total > cap {
            return Ok(total);
        }
    }
    Ok(total)
}

/// Plain (unpruned, unbudgeted) minimax reference solver.  Superseded in the hot
/// path by `solve_endgame_ab`, but kept as the simplest correct implementation:
/// the Python expectiminimax in `endgame_solver.py` is equivalence-tested against
/// the alpha-beta solver, and this mirrors that reference shape in Rust.
#[allow(dead_code)]
fn exact_solve_no_chance(
    state: &RustGameState,
    score_scale: f64,
    margin_gain: f64,
    alpha: f64,
) -> PyResult<f64> {
    if state.phase == GAME_OVER {
        return Ok(terminal_search_value(
            state,
            score_scale,
            margin_gain,
            alpha,
        ));
    }
    if !is_no_chance_endgame_state(state) {
        return Err(PyValueError::new_err(
            "exact_solve_no_chance requires PLACE_AND_SELECT with deck len 0 or 4, or FINAL_PLACEMENT with deck len 0",
        ));
    }

    let actor = state.actor()?;
    let legal = state.legal_actions_indexed();
    if legal.is_empty() {
        return Err(PyValueError::new_err(format!(
            "non-terminal state has no legal actions (phase={})",
            state.phase
        )));
    }

    let mut best = if actor == 0 {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    for &(_idx, placement, pick) in &legal {
        let child = state.step(placement, pick)?;
        let v = exact_solve_no_chance(&child, score_scale, margin_gain, alpha)?;
        if actor == 0 {
            best = best.max(v);
        } else {
            best = best.min(v);
        }
    }
    Ok(best)
}

/// Flood one connected same-terrain region from (sx, sy), marking `visited`,
/// returning (area, crowns).  Scoped helper for `placement_score_delta`.
fn bfs_region(board: &RustBoard, sx: i8, sy: i8, t: u8, visited: &mut [bool; CELLS]) -> (i32, i32) {
    let mut stack: Vec<(i8, i8)> = vec![(sx, sy)];
    visited[idx(sx, sy)] = true;
    let mut area = 0i32;
    let mut crowns = 0i32;
    while let Some((cx, cy)) = stack.pop() {
        area += 1;
        crowns += board.crowns[idx(cx, cy)] as i32;
        for (dx, dy) in DIRS {
            let nx = cx + dx;
            let ny = cy + dy;
            if in_bounds(nx, ny) {
                let ni = idx(nx, ny);
                if !visited[ni] && board.terrain[ni] == t {
                    visited[ni] = true;
                    stack.push((nx, ny));
                }
            }
        }
    }
    (area, crowns)
}

/// Territory-score delta of one terrain group's new cells (`seeds` = list of
/// (x, y, crowns)) merging with the existing same-terrain regions they touch.
/// `visited` is shared across the two halves of a placement so a region adjacent
/// to both isn't double-counted.  Returns new_contribution − old_contribution.
fn terrain_group_delta(
    board: &RustBoard,
    t: u8,
    seeds: &[(i8, i8, i32)],
    visited: &mut [bool; CELLS],
) -> i32 {
    let nc = seeds.len() as i32;
    let sc: i32 = seeds.iter().map(|s| s.2).sum();
    let mut old_contrib = 0i32;
    let mut tot_area = 0i32;
    let mut tot_crowns = 0i32;
    for &(sx, sy, _) in seeds {
        for (dx, dy) in DIRS {
            let nx = sx + dx;
            let ny = sy + dy;
            if in_bounds(nx, ny) {
                let ni = idx(nx, ny);
                // New cells are still EMPTY on the board, so they never match `t`
                // here — only existing same-terrain neighbours are flooded.
                if !visited[ni] && board.terrain[ni] == t {
                    let (a, cr) = bfs_region(board, nx, ny, t, visited);
                    old_contrib += a * cr;
                    tot_area += a;
                    tot_crowns += cr;
                }
            }
        }
    }
    let merged_area = nc + tot_area;
    let merged_crowns = sc + tot_crowns;
    merged_area * merged_crowns - old_contrib
}

/// Move-ordering heuristic (OPT-4b): the exact immediate territory-score delta of
/// a placement — the increase in Σ(region_area × region_crowns) caused by adding
/// the two cells (terrains/crowns `t_a/c_a` at (x1,y1), `t_b/c_b` at (x2,y2)).
/// Harmony/middle-kingdom bonuses are end-state properties and are intentionally
/// excluded.  Advisory only: never changes the minimax value, only pruning order.
#[allow(clippy::too_many_arguments)]
fn placement_score_delta(
    board: &RustBoard,
    t_a: u8,
    c_a: u8,
    x1: i8,
    y1: i8,
    t_b: u8,
    c_b: u8,
    x2: i8,
    y2: i8,
) -> i32 {
    let mut visited = [false; CELLS];
    if t_a == t_b {
        // Same terrain: the two halves form one connected unit and merge with all
        // same-terrain regions adjacent to either.
        terrain_group_delta(
            board,
            t_a,
            &[(x1, y1, c_a as i32), (x2, y2, c_b as i32)],
            &mut visited,
        )
    } else {
        terrain_group_delta(board, t_a, &[(x1, y1, c_a as i32)], &mut visited)
            + terrain_group_delta(board, t_b, &[(x2, y2, c_b as i32)], &mut visited)
    }
}

/// Count occupied cells per terrain type (indices 2..=7) on a board.
fn terrain_counts(board: &RustBoard) -> [u8; 8] {
    let mut counts = [0u8; 8];
    for y in board.min_y..=board.max_y {
        for x in board.min_x..=board.max_x {
            let t = board.terrain[idx(x, y)] as usize;
            if (2..=7).contains(&t) {
                counts[t] += 1;
            }
        }
    }
    counts
}

/// Pick-ordering heuristic (OPT-4b): value of claiming `domino_id` for `player` —
/// each half's crowns weighted by how many cells of that terrain the player
/// already owns (a tile that extends an established terrain is worth more).
fn pick_order_score(domino_id: u16, terrain_counts: &[u8; 8]) -> i32 {
    let (t_a, c_a, t_b, c_b) = dom(domino_id);
    (c_a as i32) * (terrain_counts[t_a as usize] as i32)
        + (c_b as i32) * (terrain_counts[t_b as usize] as i32)
}

/// Estimate how valuable the picked tile would be to the opponent if they had
/// taken it instead. Used to score the denial value of a pick.
fn opponent_denial_score(domino_id: u16, opponent_terrain_counts: &[u8; 8]) -> i32 {
    pick_order_score(domino_id, opponent_terrain_counts)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SolverOrderMode {
    Baseline,
    Denial,
    Lookahead,
    Lookahead2,
    Lookahead2Adaptive8,
    Lookahead2Adaptive,
    Lookahead2Adaptive16,
    Lookahead2Adaptive20,
    Lookahead2Clustered,
    Lookahead1Clustered,
    Combined,
}

const ADAPTIVE_LOOKAHEAD_MIN_LEGAL: usize = 12;
const CLUSTERED_LOOKAHEAD_MIN_LEGAL: usize = 8;
const CLUSTERED_LOOKAHEAD_DELTA: i32 = 4;
const CLUSTERED_LOOKAHEAD_MIN_TOP_BAND: usize = 4;

impl SolverOrderMode {
    fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "baseline" => Ok(Self::Baseline),
            "denial" | "option_a" => Ok(Self::Denial),
            "lookahead" | "option_b" => Ok(Self::Lookahead),
            "lookahead2" | "recursive_lookahead2" => Ok(Self::Lookahead2),
            "lookahead2_adaptive8" => Ok(Self::Lookahead2Adaptive8),
            "lookahead2_adaptive" | "adaptive_lookahead2" | "lookahead2_adaptive12" => {
                Ok(Self::Lookahead2Adaptive)
            }
            "lookahead2_adaptive16" => Ok(Self::Lookahead2Adaptive16),
            "lookahead2_adaptive20" => Ok(Self::Lookahead2Adaptive20),
            "lookahead2_clustered" | "clustered_lookahead2" => Ok(Self::Lookahead2Clustered),
            "lookahead1_clustered" | "clustered_lookahead1" => Ok(Self::Lookahead1Clustered),
            "combined" | "option_c" => Ok(Self::Combined),
            _ => Err(PyValueError::new_err(format!(
                "unknown solver ordering '{s}' (expected baseline, denial, lookahead, lookahead2, lookahead2_adaptive8, lookahead2_adaptive, lookahead2_adaptive16, lookahead2_adaptive20, lookahead2_clustered, lookahead1_clustered, combined)"
            ))),
        }
    }

    fn uses_denial(self) -> bool {
        matches!(self, Self::Denial | Self::Combined)
    }

    fn uses_lookahead_at_depth(self, depth: u32) -> bool {
        match self {
            Self::Lookahead | Self::Combined => depth == 0,
            Self::Lookahead2 => depth <= 2,
            Self::Lookahead2Adaptive8
            | Self::Lookahead2Adaptive
            | Self::Lookahead2Adaptive16
            | Self::Lookahead2Adaptive20
            | Self::Lookahead2Clustered
            | Self::Lookahead1Clustered => depth == 0,
            _ => false,
        }
    }

    fn adaptive_lookahead_min_legal(self) -> Option<usize> {
        match self {
            Self::Lookahead2Adaptive8 => Some(8),
            Self::Lookahead2Adaptive => Some(ADAPTIVE_LOOKAHEAD_MIN_LEGAL),
            Self::Lookahead2Adaptive16 => Some(16),
            Self::Lookahead2Adaptive20 => Some(20),
            _ => None,
        }
    }

    fn uses_adaptive_lookahead(self, depth: u32, legal_len: usize) -> bool {
        self.adaptive_lookahead_min_legal()
            .is_some_and(|min_legal| (1..=2).contains(&depth) && legal_len >= min_legal)
    }
}

fn cheap_order_score_for_solver(
    board: &RustBoard,
    halves: Option<(u8, u8, u8, u8)>,
    tc: &[u8; 8],
    opp_tc: Option<&[u8; 8]>,
    p: Option<(i8, i8, i8, i8, bool)>,
    pk: Option<u16>,
) -> i32 {
    let key_placement = match (p, halves) {
        (Some((x1, y1, x2, y2, flipped)), Some((t_a, c_a, t_b, c_b))) => {
            let (th1, ch1, th2, ch2) = if flipped {
                (t_b, c_b, t_a, c_a)
            } else {
                (t_a, c_a, t_b, c_b)
            };
            placement_score_delta(board, th1, ch1, x1, y1, th2, ch2, x2, y2)
        }
        _ => 0,
    };
    let key_pick = match pk {
        Some(pid) => pick_order_score(pid, tc),
        None => 0,
    };
    let key_denial = match (pk, opp_tc) {
        (Some(pid), Some(counts)) => opponent_denial_score(pid, counts),
        _ => 0,
    };
    key_placement + key_pick + key_denial
}

/// Sort legal actions in place by descending move-ordering heuristic, breaking
/// ties by ascending joint index for determinism.  Primary key: placement score
/// delta; secondary: pick value; both descending for the mover (the same sort
/// serves max and min nodes — each tries its locally strongest moves first).
fn order_legal_for_solver(
    state: &RustGameState,
    legal: &mut [(u16, Option<(i8, i8, i8, i8, bool)>, Option<u16>)],
    mode: SolverOrderMode,
) {
    if legal.len() < 2 {
        return;
    }
    let actor = match state.actor() {
        Ok(a) => a,
        Err(_) => return,
    };
    let board = &state.boards[actor as usize];
    let halves = state.domino_in_hand(actor).map(dom);
    // terrain_counts is board-wide and constant across this node's actions, so
    // compute it once rather than per-action.
    let tc = terrain_counts(board);
    let opp_tc = if mode.uses_denial() {
        Some(terrain_counts(&state.boards[(1 - actor) as usize]))
    } else {
        None
    };
    legal.sort_by_cached_key(|&(idx_key, p, pk)| {
        let score = cheap_order_score_for_solver(board, halves, &tc, opp_tc.as_ref(), p, pk);
        // Negate so the natural ascending sort yields descending benefit;
        // (score, idx) lexicographic with idx as the stable tiebreaker.
        (-score, idx_key)
    });
}

fn cheap_scores_clustered_for_solver(
    state: &RustGameState,
    legal: &[(u16, Option<(i8, i8, i8, i8, bool)>, Option<u16>)],
    mode: SolverOrderMode,
) -> bool {
    if legal.len() < CLUSTERED_LOOKAHEAD_MIN_LEGAL {
        return false;
    }
    let actor = match state.actor() {
        Ok(a) => a,
        Err(_) => return false,
    };
    let board = &state.boards[actor as usize];
    let halves = state.domino_in_hand(actor).map(dom);
    let tc = terrain_counts(board);
    let opp_tc = if mode.uses_denial() {
        Some(terrain_counts(&state.boards[(1 - actor) as usize]))
    } else {
        None
    };
    let mut best = i32::MIN;
    let mut scores = Vec::with_capacity(legal.len());
    for &(_idx_key, p, pk) in legal {
        let score = cheap_order_score_for_solver(board, halves, &tc, opp_tc.as_ref(), p, pk);
        best = best.max(score);
        scores.push(score);
    }
    scores
        .into_iter()
        .filter(|&score| best - score <= CLUSTERED_LOOKAHEAD_DELTA)
        .take(CLUSTERED_LOOKAHEAD_MIN_TOP_BAND)
        .count()
        >= CLUSTERED_LOOKAHEAD_MIN_TOP_BAND
}

/// Compute the raw margin (s0 - s1) after applying `action` to `state`.
/// Used for 1-ply look-ahead move ordering at root nodes.
fn one_ply_margin(
    state: &RustGameState,
    placement: Option<(i8, i8, i8, i8, bool)>,
    pick: Option<u16>,
) -> PyResult<i32> {
    let next = state.step(placement, pick)?;
    let (s0, s1) = next.scores();
    Ok(s0 - s1)
}

/// Order legal actions using 1-ply look-ahead margin evaluation.
fn order_legal_for_solver_lookahead(
    state: &RustGameState,
    legal: &mut [(u16, Option<(i8, i8, i8, i8, bool)>, Option<u16>)],
    mode: SolverOrderMode,
) -> PyResult<()> {
    if legal.len() < 2 {
        return Ok(());
    }
    let actor = state.actor()?;
    let opp_tc = if mode == SolverOrderMode::Combined {
        Some(terrain_counts(&state.boards[(1 - actor) as usize]))
    } else {
        None
    };
    let mut keyed: Vec<_> = legal
        .iter()
        .map(|&(idx_key, p, pk)| {
            let margin = one_ply_margin(state, p, pk)?;
            let denial = match (pk, opp_tc.as_ref()) {
                (Some(pid), Some(counts)) => opponent_denial_score(pid, counts),
                _ => 0,
            };
            let key = if actor == 0 {
                (-margin, -denial, idx_key)
            } else {
                (margin, -denial, idx_key)
            };
            Ok((key, (idx_key, p, pk)))
        })
        .collect::<PyResult<Vec<_>>>()?;
    keyed.sort_unstable_by_key(|(key, _action)| *key);
    for (dst, (_key, action)) in legal.iter_mut().zip(keyed.into_iter()) {
        *dst = action;
    }
    Ok(())
}

fn order_legal_for_solver_at_depth(
    state: &RustGameState,
    legal: &mut [(u16, Option<(i8, i8, i8, i8, bool)>, Option<u16>)],
    mode: SolverOrderMode,
    depth: u32,
) -> PyResult<()> {
    let clustered_depth = match mode {
        SolverOrderMode::Lookahead2Clustered => (1..=2).contains(&depth),
        SolverOrderMode::Lookahead1Clustered => depth == 1,
        _ => false,
    };
    let clustered = clustered_depth && cheap_scores_clustered_for_solver(state, legal, mode);
    if mode.uses_lookahead_at_depth(depth)
        || mode.uses_adaptive_lookahead(depth, legal.len())
        || clustered
    {
        order_legal_for_solver_lookahead(state, legal, mode)
    } else {
        order_legal_for_solver(state, legal, mode);
        Ok(())
    }
}

/// Single-pass, budgeted, alpha-beta minimax over a no-chance endgame (OPT-2 +
/// OPT-3 + OPT-4).  Returns `Ok(Some(value))` in the player-0 frame, or
/// `Ok(None)` when the node budget was exhausted mid-traversal.  Unlike the
/// count-then-solve pair it replaces, it never traverses the tree twice and
/// prunes subtrees that cannot affect the result.
///
/// Correctness of pruning: terminal solver utility is a total ordering over the
/// official result: integer raw margins, with -0.25/0/+0.25 representing a
/// score-tied tiebreak loss/true draw/tiebreak win. The position value remains a
/// standard scalar min/max, so alpha-beta applies without modification. The
/// max/min layers need not strictly alternate — each node is typed by `actor()`
/// and the (alpha, beta) window stays valid through any sequence of nodes.
///
/// Caller guarantees `state` is a no-chance endgame state (deck ∈ {0, 4} in a
/// turn phase, or GAME_OVER); descendants of such states are likewise no-chance,
/// so the property is not re-checked in the hot recursion.
#[allow(clippy::too_many_arguments)]
fn solve_endgame_ab(
    state: &RustGameState,
    deadline: std::time::Instant,
    mut alpha: f64,
    mut beta: f64,
    mode: SolverOrderMode,
    depth: u32,
) -> PyResult<Option<f64>> {
    // The search runs on a player-0-frame ordering scalar: the raw integer score
    // margin except at score ties, where +/-0.25 encodes the official territory/
    // crowns tiebreak around a true draw. It contains no training hyperparameters.
    // The training value is reconstructed only AFTER the solve.
    //
    // GAME_OVER returns a value with zero further work, so resolve it before the
    // deadline check — a timed-out search should still return exact terminal leaves
    // it has already reached rather than abort on them.
    if state.phase == GAME_OVER {
        return Ok(Some(terminal_solver_utility(state)));
    }
    // Wall-clock budget (replaces the old node-count budget). Per-node
    // Instant::now() is ~5ns — negligible against the ~1μs+ of step()+ordering
    // work each interior node does. Ok(None) == deadline exceeded (caller falls
    // back), matching the previous budget-exceeded sentinel.
    if std::time::Instant::now() >= deadline {
        return Ok(None);
    }

    let actor = state.actor()?;
    let mut legal = state.legal_actions_indexed();
    if legal.is_empty() {
        return Err(PyValueError::new_err(format!(
            "non-terminal state has no legal actions (phase={})",
            state.phase
        )));
    }
    order_legal_for_solver_at_depth(state, &mut legal, mode, depth)?;

    if actor == 0 {
        let mut best = f64::NEG_INFINITY;
        for &(_idx, p, pk) in &legal {
            let child = state.step(p, pk)?;
            match solve_endgame_ab(&child, deadline, alpha, beta, mode, depth + 1)? {
                None => return Ok(None),
                Some(v) => {
                    if v > best {
                        best = v;
                    }
                    if best > alpha {
                        alpha = best;
                    }
                    if alpha >= beta {
                        break; // beta cutoff
                    }
                }
            }
        }
        Ok(Some(best))
    } else {
        let mut best = f64::INFINITY;
        for &(_idx, p, pk) in &legal {
            let child = state.step(p, pk)?;
            match solve_endgame_ab(&child, deadline, alpha, beta, mode, depth + 1)? {
                None => return Ok(None),
                Some(v) => {
                    if v < best {
                        best = v;
                    }
                    if best < beta {
                        beta = best;
                    }
                    if beta <= alpha {
                        break; // alpha cutoff
                    }
                }
            }
        }
        Ok(Some(best))
    }
}

/// Young Brothers Wait (YBW) parallel alpha-beta solver (OPT-6).
///
/// Solves the first (best-ordered) root child serially to establish an alpha/beta
/// bound, then solves the remaining root children in parallel via Rayon, each
/// seeded with that bound. Returns `Ok(Some(value))` when every subtree completed
/// before the deadline, `Ok(None)` when some subtree hit it, or `Err` on an
/// internal error.
///
/// **Call ONLY at the root.** The recursive calls use the serial `solve_endgame_ab`,
/// so there is exactly one fan-out — no nested Rayon / thread explosion.
///
/// Budget semantics: every child (the serial first child and all parallel
/// siblings) shares the SAME wall-clock `deadline`, so total solve time for the
/// position is bounded by the deadline regardless of fan-out. YBW seeds siblings
/// with only the first child's bound (not the progressively tightened serial
/// bound), so parallel subtrees visit ≥ as many nodes as serial: wall-clock drops
/// via parallelism, node counts do not. Returns `Ok(Some(value))` when every
/// subtree completed before the deadline, `Ok(None)` when any subtree hit it.
fn solve_endgame_ab_parallel(
    state: &RustGameState,
    deadline: std::time::Instant,
    mode: SolverOrderMode,
) -> PyResult<Option<f64>> {
    // Returns the official-cascade solver utility; callers convert it to the
    // training value via `solver_utility_to_training_value`.
    if state.phase == GAME_OVER {
        return Ok(Some(terminal_solver_utility(state)));
    }
    let actor = state.actor()?;
    let mut legal = state.legal_actions_indexed();
    if legal.is_empty() {
        return Err(PyValueError::new_err(format!(
            "non-terminal state has no legal actions (phase={})",
            state.phase
        )));
    }
    order_legal_for_solver_at_depth(state, &mut legal, mode, 0)?;

    // One transposition table shared by the serial first child and all
    // parallel siblings (same rationale + correctness contract as
    // solve_root_exact: 62-86% duplicate interior visits measured on the real
    // fallback corpus; full-window root value unchanged). This is the advisor/
    // value-only path's share of the TT win — paired corpus measurement showed
    // ~3.4x median wall-clock.
    let tt = TranspositionTable::new();

    // Step 1: solve the first (best-ordered) child serially to establish a bound.
    let (_i0, p0, pk0) = legal[0];
    let first_next = state.step(p0, pk0)?;
    let mut buf0 = Vec::with_capacity(1024);
    let first_val = match solve_endgame_ab_tt(
        &first_next,
        deadline,
        MARGIN_LO,
        MARGIN_HI,
        mode,
        1,
        &tt,
        &mut buf0,
    )? {
        Some(v) => v,
        None => return Ok(None),
    };

    let mut best_val = first_val;
    let mut alpha = MARGIN_LO;
    let mut beta = MARGIN_HI;
    if actor == 0 {
        alpha = alpha.max(first_val);
    } else {
        beta = beta.min(first_val);
    }
    // First child alone caused a cutoff, or it was the only child.
    if alpha >= beta {
        return Ok(Some(best_val));
    }
    let remaining = &legal[1..];
    if remaining.is_empty() {
        return Ok(Some(best_val));
    }
    let (captured_alpha, captured_beta) = (alpha, beta);

    // Step 2: solve the remaining children in parallel, all sharing the deadline
    // and the first child's bound.
    let results: Vec<PyResult<Option<f64>>> = remaining
        .par_iter()
        .map(|&(_idx, p, pk)| -> PyResult<Option<f64>> {
            let next = state.step(p, pk)?;
            let mut buf = Vec::with_capacity(1024);
            solve_endgame_ab_tt(
                &next,
                deadline,
                captured_alpha,
                captured_beta,
                mode,
                1,
                &tt,
                &mut buf,
            )
        })
        .collect();

    // Step 3: combine. Any subtree that hit the deadline fails the whole solve.
    for r in results {
        match r? {
            None => return Ok(None),
            Some(val) => {
                if actor == 0 {
                    if val > best_val {
                        best_val = val;
                    }
                } else if val < best_val {
                    best_val = val;
                }
            }
        }
    }
    Ok(Some(best_val))
}

// ─── Solver-state hashing + transposition diagnostics ───────────────────────
//
// State identity for the solver mirrors `EndgameKey`: boards (terrain+crowns),
// phase, actor_index, sorted deck, current_row, pending_claims, next_claims.
// `discards` is deliberately excluded — it feeds encoder features only, never
// scores or legal actions, so two states differing only in discards are
// solver-identical. harmony/middle_kingdom/castle are constant within a game
// and every hash consumer is scoped to a single game's solve.

/// Serialize the solver-relevant state into `buf` (cleared first). Length
/// prefixes guard against aliasing across the variable-length sections.
fn solver_state_bytes(state: &RustGameState, buf: &mut Vec<u8>) {
    buf.clear();
    buf.push(state.phase);
    buf.push(state.actor_index as u8);
    let mut deck = state.deck.clone();
    deck.sort_unstable();
    buf.push(deck.len() as u8);
    for d in &deck {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.push(state.current_row.len() as u8);
    for d in &state.current_row {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.push(state.pending_claims.len() as u8);
    for &(p, d) in &state.pending_claims {
        buf.push(p);
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.push(state.next_claims.len() as u8);
    for &(p, d) in &state.next_claims {
        buf.push(p);
        buf.extend_from_slice(&d.to_le_bytes());
    }
    for b in &state.boards {
        buf.extend_from_slice(&b.terrain);
        buf.extend_from_slice(&b.crowns);
    }
}

/// 128-bit xxh3 hash of the solver-relevant state. At the scale of one root
/// solve (≤ tens of millions of distinct states) the collision probability of a
/// 128-bit hash is negligible, which is what lets the TT / diagnostics compare
/// hashes instead of full 1KB `EndgameKey`s.
fn solver_state_hash128(state: &RustGameState, buf: &mut Vec<u8>) -> u128 {
    solver_state_bytes(state, buf);
    xxhash_rust::xxh3::xxh3_128(buf)
}

// ─── Within-single-solve transposition table ────────────────────────────────
//
// The endgame solver always searches to GAME_OVER, so a stored EXACT value is
// the true minimax value of the state — valid on every re-visit regardless of
// path or remaining depth (no depth field needed, unlike depth-limited chess
// TTs). LOWER/UPPER entries are fail-soft bounds produced by window cutoffs.
//
// Scope: ONE root solve (all children of one root share a table; the plan
// cascade reuses it across the endgame's successive roots). Never persisted
// across games. Diagnosis on the real fallback corpus measured 62-86% of
// interior visits re-entering already-seen states (pick-order permutations
// collapse 4x per selection round via advance_round's next_claims sort), which
// is what justifies the probe/insert cost.

/// TT value classification. For the mover-agnostic player-0-frame margin value:
/// Exact = true minimax value; Lower = value >= stored (fail high);
/// Upper = value <= stored (fail low).
#[derive(Clone, Copy, PartialEq)]
enum TTFlag {
    Exact,
    Lower,
    Upper,
}

/// Sharded transposition table, safe for concurrent use by parallel sibling
/// solves (rayon). 64 mutex shards keep contention negligible against the
/// ~1µs/node solve work. Inserts stop at `cap` entries (probes continue), so
/// one pathological solve cannot grow memory unboundedly.
struct TranspositionTable {
    shards: Vec<std::sync::Mutex<HashMap<u128, (f64, TTFlag)>>>,
    cap_per_shard: usize,
}

const TT_SHARDS: usize = 64;
/// ~4M entries total ≈ 130MB worst case ((16+8+1+pad)*4M + HashMap overhead) —
/// bounded and short-lived (freed when the root solve returns).
const TT_CAP_TOTAL: usize = 4_000_000;

impl TranspositionTable {
    fn new() -> Self {
        TranspositionTable {
            shards: (0..TT_SHARDS)
                .map(|_| std::sync::Mutex::new(HashMap::new()))
                .collect(),
            cap_per_shard: TT_CAP_TOTAL / TT_SHARDS,
        }
    }

    #[inline]
    fn shard(&self, key: u128) -> &std::sync::Mutex<HashMap<u128, (f64, TTFlag)>> {
        &self.shards[(key as usize) & (TT_SHARDS - 1)]
    }

    #[inline]
    fn probe(&self, key: u128) -> Option<(f64, TTFlag)> {
        self.shard(key).lock().unwrap().get(&key).copied()
    }

    #[inline]
    fn store(&self, key: u128, value: f64, flag: TTFlag) {
        let mut m = self.shard(key).lock().unwrap();
        if let Some(slot) = m.get_mut(&key) {
            // Never let a bound clobber an Exact entry (parallel siblings can
            // finish the same state with different windows); among bounds of
            // the same kind keep the tighter one.
            let keep = match (slot.1, flag) {
                (TTFlag::Exact, _) => true,
                (_, TTFlag::Exact) => false,
                (TTFlag::Lower, TTFlag::Lower) => slot.0 >= value,
                (TTFlag::Upper, TTFlag::Upper) => slot.0 <= value,
                _ => true, // mixed bounds: keep the incumbent (either is valid)
            };
            if !keep {
                *slot = (value, flag);
            }
        } else if m.len() < self.cap_per_shard {
            m.insert(key, (value, flag));
        }
    }
}

/// `solve_endgame_ab` + transposition table. Traversal, ordering, deadline and
/// fail-soft window semantics are identical; the TT adds:
///   probe — Exact hit returns immediately; Lower/Upper hits tighten the
///   window (standard TT window narrowing) and may cut off;
///   store — the returned best is classified against the ORIGINAL window:
///   fail-low (best <= alpha_in) → Upper, fail-high (best >= beta) → Lower,
///   else Exact.
/// Correctness contract (weaker than bit-identical traversal, sufficient for
/// every caller): the returned value is a valid fail-soft alpha-beta result
/// for the caller's window — in particular a FULL-window call returns the
/// true minimax value, identical to the untabled solver. Interior fail-high/
/// fail-low returns may differ from the untabled search's fail-soft values
/// (a TT bound hit returns the stored bound), which alpha-beta treats
/// equivalently. States never repeat within a game (tiles only accumulate),
/// so there is no graph-history/path-dependence hazard.
#[allow(clippy::too_many_arguments)]
fn solve_endgame_ab_tt(
    state: &RustGameState,
    deadline: std::time::Instant,
    mut alpha: f64,
    mut beta: f64,
    mode: SolverOrderMode,
    depth: u32,
    tt: &TranspositionTable,
    buf: &mut Vec<u8>,
) -> PyResult<Option<f64>> {
    if state.phase == GAME_OVER {
        return Ok(Some(terminal_solver_utility(state)));
    }
    if std::time::Instant::now() >= deadline {
        return Ok(None);
    }

    let key = solver_state_hash128(state, buf);
    if let Some((v, flag)) = tt.probe(key) {
        match flag {
            TTFlag::Exact => return Ok(Some(v)),
            TTFlag::Lower => {
                if v >= beta {
                    return Ok(Some(v));
                }
                if v > alpha {
                    alpha = v;
                }
            }
            TTFlag::Upper => {
                if v <= alpha {
                    return Ok(Some(v));
                }
                if v < beta {
                    beta = v;
                }
            }
        }
    }

    let actor = state.actor()?;
    let mut legal = state.legal_actions_indexed();
    if legal.is_empty() {
        return Err(PyValueError::new_err(format!(
            "non-terminal state has no legal actions (phase={})",
            state.phase
        )));
    }
    order_legal_for_solver_at_depth(state, &mut legal, mode, depth)?;

    // Fail-soft classification must use the window the search actually ran
    // with — INCLUDING any TT probe tightening above — or a fail-low against
    // a probe-raised alpha would be mis-stored as Exact.
    let alpha_in = alpha;
    let beta_in = beta;
    let best = if actor == 0 {
        let mut best = f64::NEG_INFINITY;
        for &(_idx, p, pk) in &legal {
            let child = state.step(p, pk)?;
            match solve_endgame_ab_tt(&child, deadline, alpha, beta, mode, depth + 1, tt, buf)? {
                None => return Ok(None),
                Some(v) => {
                    if v > best {
                        best = v;
                    }
                    if best > alpha {
                        alpha = best;
                    }
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        best
    } else {
        let mut best = f64::INFINITY;
        for &(_idx, p, pk) in &legal {
            let child = state.step(p, pk)?;
            match solve_endgame_ab_tt(&child, deadline, alpha, beta, mode, depth + 1, tt, buf)? {
                None => return Ok(None),
                Some(v) => {
                    if v < best {
                        best = v;
                    }
                    if best < beta {
                        beta = best;
                    }
                    if beta <= alpha {
                        break;
                    }
                }
            }
        }
        best
    };

    let flag = if best <= alpha_in {
        TTFlag::Upper
    } else if best >= beta_in {
        TTFlag::Lower
    } else {
        TTFlag::Exact
    };
    tt.store(key, best, flag);
    Ok(Some(best))
}

/// Counters for `solve_endgame_ab_transpo`.
#[derive(Default)]
struct TranspoStats {
    interior: u64,   // interior (non-terminal) nodes visited
    terminals: u64,  // terminal leaves visited
    dup_visits: u64, // interior visits whose state was already seen
    seen: HashSet<u128>,
}

/// Instrumented copy of `solve_endgame_ab`: identical traversal and value, plus
/// per-interior-node state hashing into `stats.seen`. `dup_visits / interior`
/// is the fraction of pruned-search work re-entering an already-visited state —
/// the first-order estimate of what an EXACT-hit transposition table would skip
/// (an underestimate of nothing: every node inside a re-entered subtree also
/// counts as a duplicate). Diagnostic only; not on any hot path.
#[allow(clippy::too_many_arguments)]
fn solve_endgame_ab_transpo(
    state: &RustGameState,
    deadline: std::time::Instant,
    mut alpha: f64,
    mut beta: f64,
    mode: SolverOrderMode,
    depth: u32,
    stats: &mut TranspoStats,
    buf: &mut Vec<u8>,
) -> PyResult<Option<f64>> {
    if state.phase == GAME_OVER {
        stats.terminals += 1;
        return Ok(Some(terminal_solver_utility(state)));
    }
    if std::time::Instant::now() >= deadline {
        return Ok(None);
    }
    stats.interior += 1;
    let h = solver_state_hash128(state, buf);
    if !stats.seen.insert(h) {
        stats.dup_visits += 1;
    }

    let actor = state.actor()?;
    let mut legal = state.legal_actions_indexed();
    if legal.is_empty() {
        return Err(PyValueError::new_err(format!(
            "non-terminal state has no legal actions (phase={})",
            state.phase
        )));
    }
    order_legal_for_solver_at_depth(state, &mut legal, mode, depth)?;

    if actor == 0 {
        let mut best = f64::NEG_INFINITY;
        for &(_idx, p, pk) in &legal {
            let child = state.step(p, pk)?;
            match solve_endgame_ab_transpo(
                &child,
                deadline,
                alpha,
                beta,
                mode,
                depth + 1,
                stats,
                buf,
            )? {
                None => return Ok(None),
                Some(v) => {
                    if v > best {
                        best = v;
                    }
                    if best > alpha {
                        alpha = best;
                    }
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        Ok(Some(best))
    } else {
        let mut best = f64::INFINITY;
        for &(_idx, p, pk) in &legal {
            let child = state.step(p, pk)?;
            match solve_endgame_ab_transpo(
                &child,
                deadline,
                alpha,
                beta,
                mode,
                depth + 1,
                stats,
                buf,
            )? {
                None => return Ok(None),
                Some(v) => {
                    if v < best {
                        best = v;
                    }
                    if best < beta {
                        beta = best;
                    }
                    if beta <= alpha {
                        break;
                    }
                }
            }
        }
        Ok(Some(best))
    }
}

/// Stable softmax over legal logits, matching encoder/mcts `_postprocess`
/// (subtract max, exp, normalise) in f64.
fn softmax_f64(logits: &[f64]) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - m).exp()).collect();
    let s: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / s).collect()
}

/// PUCT child selection (mirrors `_select_child`): argmax over children of
/// Q + cpuct·prior·√N_parent / (1 + N_child), Q in the acting player's frame,
/// FPU for unvisited children.  First child achieving the max wins; children are
/// in ascending joint-index order, so tie-breaking matches the Python engine.
fn select_child(arena: &[Node], node_id: u32, fpu: f64, cpuct: f64) -> u32 {
    let node = &arena[node_id as usize];
    let actor = node
        .state
        .as_ref()
        .unwrap()
        .actor()
        .expect("non-terminal node has an actor");
    let sqrt_n = (node.visit_count as f64).sqrt();
    let mut best_score = f64::NEG_INFINITY;
    let mut best = u32::MAX;
    for &(_idx, cid) in &node.children {
        let child = &arena[cid as usize];
        let q = if child.visit_count > 0 {
            let q0 = child.value_sum / child.visit_count as f64; // player-0 frame
            if actor == 0 { q0 } else { -q0 } // → acting player's frame
        } else {
            fpu
        };
        let u = cpuct * child.prior * sqrt_n / (1.0 + child.visit_count as f64);
        let score = q + u;
        if score > best_score {
            best_score = score;
            best = cid;
        }
    }
    best
}

/// Expand a leaf: evaluate the network at the leaf via the Python evaluator,
/// create child edges with softmaxed priors, return the leaf value in player-0
/// frame.  Mirrors `_expand` + `_evaluate` + `_postprocess`.
fn expand(arena: &mut Vec<Node>, node_id: u32, ev: &Py<PyAny>) -> PyResult<f64> {
    // No-GIL phase: encode the leaf + collect legal actions (pure Rust).  Runs
    // inside RustMCTS.search's py.allow_threads, so the GIL is released here.
    let (actor, legal, my, opp, flat) = {
        let state = arena[node_id as usize].state.as_ref().unwrap();
        let actor = state.actor()?;
        let legal = state.legal_actions_indexed();
        let (my, opp, flat) = state.encode_arrays(actor)?;
        (actor, legal, my, opp, flat)
    };
    let idxs: Vec<i64> = legal.iter().map(|t| t.0 as i64).collect();

    // GIL ONLY here: build numpy arrays, call the Python evaluator, read results.
    // numpy/PyList construction and the call all require the GIL, so they are the
    // single re-acquisition point at the leaf-evaluation boundary.
    let (value, gathered) = Python::attach(|py| -> PyResult<(f64, Vec<f64>)> {
        let mb_py = my.insert_axis(Axis(0)).into_pyarray(py); // (1,9,13,13)
        let ob_py = opp.insert_axis(Axis(0)).into_pyarray(py);
        let flat_py = flat.insert_axis(Axis(0)).into_pyarray(py);
        let idxs_py = idxs.into_pyarray(py); // (n,) int64
        let idxs_list = PyList::new(py, [idxs_py])?;
        let result = ev.bind(py).call1((mb_py, ob_py, flat_py, idxs_list))?;
        let tuple = result.downcast::<PyTuple>()?;
        // Python passes f32 (logits/values are .float()) to halve D2H transfer;
        // cast to f64 here for the tree's internal accumulation.
        let value = {
            let arr = tuple.get_item(0)?;
            let arr = arr.downcast::<PyArray1<f32>>()?;
            arr.readonly().as_slice()?[0] as f64
        };
        let gathered = {
            let list = tuple.get_item(1)?;
            let list = list.downcast::<PyList>()?;
            let g0 = list.get_item(0)?;
            let arr = g0.downcast::<PyArray1<f32>>()?;
            arr.readonly()
                .as_slice()?
                .iter()
                .map(|&x| x as f64)
                .collect()
        };
        Ok((value, gathered))
    })?;

    // No-GIL phase: softmax + child creation.
    let priors = softmax_f64(&gathered);
    let value0 = if actor == 0 { value } else { -value };
    for (i, &(idx, placement, pick)) in legal.iter().enumerate() {
        let child_id = arena.len() as u32;
        arena.push(Node::new(priors[i], (placement, pick)));
        arena[node_id as usize].children.push((idx, child_id));
    }
    arena[node_id as usize].is_expanded = true;
    Ok(value0)
}

/// One serial simulation (mirrors `_simulate`): descend by PUCT to an unexpanded
/// or terminal leaf, evaluate/expand it, back the value up the path (player-0
/// frame, no sign flips).
fn simulate(
    arena: &mut Vec<Node>,
    root_id: u32,
    ev: &Py<PyAny>,
    fpu: f64,
    cpuct: f64,
    score_scale: f64,
    margin_gain: f64,
    alpha: f64,
) -> PyResult<()> {
    let mut path: Vec<u32> = vec![root_id];
    let mut node_id = root_id;
    loop {
        let (expanded, terminal) = {
            let n = &arena[node_id as usize];
            (n.is_expanded, n.state.as_ref().unwrap().phase == GAME_OVER)
        };
        if !expanded || terminal {
            break;
        }
        let child_id = select_child(arena, node_id, fpu, cpuct);
        if arena[child_id as usize].state.is_none() {
            let (placement, pick) = arena[child_id as usize].action;
            let child_state = {
                let parent = arena[node_id as usize].state.as_ref().unwrap();
                parent.step(placement, pick)?
            };
            arena[child_id as usize].state = Some(child_state);
        }
        path.push(child_id);
        node_id = child_id;
    }

    let v0 = {
        let terminal = arena[node_id as usize].state.as_ref().unwrap().phase == GAME_OVER;
        if terminal {
            terminal_search_value(
                arena[node_id as usize].state.as_ref().unwrap(),
                score_scale,
                margin_gain,
                alpha,
            )
        } else {
            expand(arena, node_id, ev)?
        }
    };

    for &n in &path {
        arena[n as usize].visit_count += 1;
        arena[n as usize].value_sum += v0;
    }
    Ok(())
}

/// Descend by PUCT to an unexpanded or terminal leaf, setting child states
/// lazily (mirrors `_descend`).  No virtual loss applied here — it is applied by
/// the caller AFTER the descent, so VL from earlier descents in the batch
/// affects the PUCT scores read here but not the descent code itself.
fn descend(arena: &mut Vec<Node>, root_id: u32, fpu: f64, cpuct: f64) -> PyResult<Vec<u32>> {
    let mut path: Vec<u32> = vec![root_id];
    let mut node_id = root_id;
    loop {
        let (expanded, terminal) = {
            let n = &arena[node_id as usize];
            (n.is_expanded, n.state.as_ref().unwrap().phase == GAME_OVER)
        };
        if !expanded || terminal {
            break;
        }
        let child_id = select_child(arena, node_id, fpu, cpuct);
        if arena[child_id as usize].state.is_none() {
            let (placement, pick) = arena[child_id as usize].action;
            let child_state = {
                let parent = arena[node_id as usize].state.as_ref().unwrap();
                parent.step(placement, pick)?
            };
            arena[child_id as usize].state = Some(child_state);
        }
        path.push(child_id);
        node_id = child_id;
    }
    Ok(path)
}

/// Apply (sign=+1) or remove (sign=-1) virtual loss along a path, mirroring
/// `_apply_virtual_loss` in the fixed player-0 frame.  Every node gets a
/// visit-count bump; non-root nodes also get value_sum nudged DOWN if their
/// chooser is player 0, UP if player 1 (vl_value0 = -1 if chooser==0 else +1),
/// so the just-collected path looks pessimistic to whoever chose it.  Removal
/// (-1) over the same path is the exact additive inverse.
fn apply_virtual_loss(arena: &mut [Node], path: &[u32], sign: i32, n_vl: i32) {
    if n_vl <= 0 {
        return;
    }
    for i in 0..path.len() {
        arena[path[i] as usize].visit_count += sign * n_vl;
        if i > 0 {
            let chooser = arena[path[i - 1] as usize]
                .state
                .as_ref()
                .unwrap()
                .actor()
                .expect("non-terminal chooser has an actor");
            let vl_value0 = if chooser == 0 { -1.0 } else { 1.0 };
            arena[path[i] as usize].value_sum += (sign * n_vl) as f64 * vl_value0;
        }
    }
}

/// Evaluate K leaves in ONE batched call to the Python evaluator.  Returns, per
/// leaf: value (network frame), gathered legal logits, the acting player, and
/// the indexed legal actions (for expansion).  mb/ob/flat are stacked to
/// (K,9,13,13)/(K,9,13,13)/(K,FLAT_SIZE); idxs are passed as a list of K int arrays.
fn evaluate_batch(
    arena: &Vec<Node>,
    leaves: &[u32],
    ev: &Py<PyAny>,
) -> PyResult<(
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<u8>,
    Vec<Vec<(u16, Option<(i8, i8, i8, i8, bool)>, Option<u16>)>>,
)> {
    let k = leaves.len();
    // No-GIL phase: encode all leaves into flat buffers (pure Rust).
    let mut mb_data: Vec<f32> = Vec::with_capacity(k * N_BOARD_CH * OUT_N * OUT_N);
    let mut ob_data: Vec<f32> = Vec::with_capacity(k * N_BOARD_CH * OUT_N * OUT_N);
    let mut flat_data: Vec<f32> = Vec::with_capacity(k * FLAT_SIZE);
    let mut actors: Vec<u8> = Vec::with_capacity(k);
    let mut legals: Vec<Vec<(u16, Option<(i8, i8, i8, i8, bool)>, Option<u16>)>> =
        Vec::with_capacity(k);
    let mut idxs_per: Vec<Vec<i64>> = Vec::with_capacity(k);

    for &leaf in leaves {
        let state = arena[leaf as usize].state.as_ref().unwrap();
        let actor = state.actor()?;
        let legal = state.legal_actions_indexed();
        let (my, opp, flat) = state.encode_arrays(actor)?;
        mb_data.extend_from_slice(my.as_slice().expect("contiguous"));
        ob_data.extend_from_slice(opp.as_slice().expect("contiguous"));
        flat_data.extend_from_slice(flat.as_slice().expect("contiguous"));
        idxs_per.push(legal.iter().map(|t| t.0 as i64).collect());
        actors.push(actor);
        legals.push(legal);
    }

    // GIL ONLY here: stack into numpy, call the evaluator, read results.
    let (values, gathered) = Python::attach(|py| -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        let mb_py = Array4::from_shape_vec((k, N_BOARD_CH, OUT_N, OUT_N), mb_data)
            .expect("mb shape")
            .into_pyarray(py);
        let ob_py = Array4::from_shape_vec((k, N_BOARD_CH, OUT_N, OUT_N), ob_data)
            .expect("ob shape")
            .into_pyarray(py);
        let flat_py = Array2::from_shape_vec((k, FLAT_SIZE), flat_data)
            .expect("flat shape")
            .into_pyarray(py);
        let idxs_items: Vec<_> = idxs_per.into_iter().map(|v| v.into_pyarray(py)).collect();
        let idxs_list = PyList::new(py, idxs_items)?;

        let result = ev.bind(py).call1((mb_py, ob_py, flat_py, idxs_list))?;
        let tuple = result.downcast::<PyTuple>()?;
        // Python passes f32 (.float()) to halve D2H transfer; cast to f64 here.
        let values: Vec<f64> = {
            let arr = tuple.get_item(0)?;
            let arr = arr.downcast::<PyArray1<f32>>()?;
            arr.readonly()
                .as_slice()?
                .iter()
                .map(|&x| x as f64)
                .collect()
        };
        let gathered: Vec<Vec<f64>> = {
            let list = tuple.get_item(1)?;
            let list = list.downcast::<PyList>()?;
            let mut out = Vec::with_capacity(k);
            for i in 0..k {
                let g = list.get_item(i)?;
                let arr = g.downcast::<PyArray1<f32>>()?;
                out.push(
                    arr.readonly()
                        .as_slice()?
                        .iter()
                        .map(|&x| x as f64)
                        .collect(),
                );
            }
            out
        };
        Ok((values, gathered))
    })?;

    Ok((values, gathered, actors, legals))
}

/// One leaf-parallel simulation step (mirrors `_simulate_batch`): collect
/// `batch_size` leaves with virtual loss, evaluate the unique non-terminal ones
/// in one batched call, expand them, remove VL, then back up real values.  A
/// collision (two descents reaching the same leaf) backs that leaf up twice,
/// exactly as two simulations would.
fn simulate_batch(
    arena: &mut Vec<Node>,
    root_id: u32,
    ev: &Py<PyAny>,
    fpu: f64,
    cpuct: f64,
    batch_size: usize,
    virtual_loss: i32,
    score_scale: f64,
    margin_gain: f64,
    alpha: f64,
) -> PyResult<()> {
    let mut paths: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let path = descend(arena, root_id, fpu, cpuct)?;
        apply_virtual_loss(arena, &path, 1, virtual_loss);
        paths.push(path);
    }

    // Unique non-terminal leaves needing evaluation (first-occurrence order, so
    // a second collision does not re-expand and overwrite fresh child stats).
    let mut unique: Vec<u32> = Vec::new();
    let mut seen: HashSet<u32> = HashSet::new();
    for path in &paths {
        let leaf = *path.last().unwrap();
        if arena[leaf as usize].state.as_ref().unwrap().phase == GAME_OVER {
            continue;
        }
        if seen.insert(leaf) {
            unique.push(leaf);
        }
    }

    let mut leaf_v0: HashMap<u32, f64> = HashMap::new();
    if !unique.is_empty() {
        let (values, gathered, actors, legals) = evaluate_batch(arena, &unique, ev)?;
        for (k, &leaf) in unique.iter().enumerate() {
            let priors = softmax_f64(&gathered[k]);
            let value0 = if actors[k] == 0 {
                values[k]
            } else {
                -values[k]
            };
            if !arena[leaf as usize].is_expanded {
                for (i, &(idx, placement, pick)) in legals[k].iter().enumerate() {
                    let child_id = arena.len() as u32;
                    arena.push(Node::new(priors[i], (placement, pick)));
                    arena[leaf as usize].children.push((idx, child_id));
                }
                arena[leaf as usize].is_expanded = true;
            }
            leaf_v0.insert(leaf, value0);
        }
    }

    // Remove VL over the exact same paths (exact additive inverse).
    for path in &paths {
        apply_virtual_loss(arena, path, -1, virtual_loss);
    }

    // Real backup — player-0 frame, no sign flips.
    for path in &paths {
        let leaf = *path.last().unwrap();
        let v0 = if arena[leaf as usize].state.as_ref().unwrap().phase == GAME_OVER {
            terminal_search_value(
                arena[leaf as usize].state.as_ref().unwrap(),
                score_scale,
                margin_gain,
                alpha,
            )
        } else {
            leaf_v0[&leaf]
        };
        for &n in path {
            arena[n as usize].visit_count += 1;
            arena[n as usize].value_sum += v0;
        }
    }
    Ok(())
}

/// Add Dirichlet noise to the root children's priors (mirrors
/// `_add_dirichlet_noise`): prior ← (1-eps)·prior + eps·noise.  Noise is sampled
/// via Gamma(alpha,1)/sum.  NOTE: the noise VALUES cannot match Python's numpy
/// RNG, so noise-on search is not bit-comparable — the equivalence gate uses
/// eps=0 (this is never called).
fn add_dirichlet_noise(arena: &mut [Node], root_id: u32, alpha: f64, eps: f64, seed: Option<u64>) {
    let child_ids: Vec<u32> = arena[root_id as usize]
        .children
        .iter()
        .map(|&(_, c)| c)
        .collect();
    let n = child_ids.len();
    if n == 0 {
        return;
    }
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let gamma = Gamma::new(alpha, 1.0).expect("alpha > 0");
    let samples: Vec<f64> = (0..n).map(|_| gamma.sample(&mut rng)).collect();
    let s: f64 = samples.iter().sum();
    for (i, &cid) in child_ids.iter().enumerate() {
        let noise = samples[i] / s;
        let c = &mut arena[cid as usize];
        c.prior = (1.0 - eps) * c.prior + eps * noise;
    }
}

// ─── Open-loop MCTS support (stateless nodes, per-simulation determinization) ──
// Port of Python's OpenLoopMCTS.  The tree is keyed on action sequences and
// stores NO concrete state in nodes; each simulation reconstructs its concrete
// state by replaying the action path on a freshly resampled deck order.  These
// helpers are the open-loop analogues of select_child / descend /
// apply_virtual_loss / add_dirichlet_noise / select_from_visits; expansion and
// backup are done inline in BatchedMCTS::update (no Rust-side evaluator call —
// evaluation goes over the external step/update batch boundary, same as the
// closed-loop path).

/// Open-loop search-tree node. Stateless: no concrete GameState is stored.
/// children: Vec<(joint_index: u16, child_id: u32)> ascending by index.
/// value_sum / visit_count in PLAYER-0 frame, same convention as Node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OLChanceBackup {
    /// Unbiased Monte Carlo backup of the row actually realized this visit.
    Sampled,
    /// Registered-probability mean over observation rows with real visits.
    Hajek,
    /// Direct probability mean over a fully bootstrapped A1c active panel.
    /// Bootstrap evaluations affect Q but never visit or policy-target counts.
    PanelMean,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OLChanceTraversal {
    /// Draw independently from the registered row probabilities each visit.
    Iid,
    /// Visit a shuffled multiplicity-expanded panel once before repeating it.
    Balanced,
}

/// How A1c constructs each fixed-width cycle of candidate public reveals.
/// This is deliberately separate from traversal: panel design controls which
/// counterfactual worlds are represented, while traversal controls which
/// already-active world receives the next search visit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum A1cPanelSampling {
    /// One shuffled bag partition per cycle. Every hidden tile appears exactly
    /// once among that cycle's hypothetical first reveals.
    Balanced,
    /// The same number of uniform 4-of-n rows, sampled independently with
    /// replacement. This is the matched-width ablation required by the deck-8
    /// oracle result; it is not the existing IID traversal over a balanced set.
    Iid,
}

#[derive(Clone, Copy)]
struct OLChanceConfig<'a> {
    panel: &'a [OLChancePanelRow],
    backup: OLChanceBackup,
    traversal: OLChanceTraversal,
    schedule_seed: u64,
    draw_seed: u64,
    a1c: Option<A1cChanceConfig<'a>>,
    // Production BatchedMCTS can ask the caller to evaluate the complete fixed
    // panel over the normal step()/update() inference boundary.  The advisor
    // performs its own synchronous admission and therefore leaves this false.
    bootstrap_full_panel: bool,
}

#[derive(Clone, Copy)]
struct A1cChanceConfig<'a> {
    cycles: &'a [Vec<[u16; 4]>],
    n_init: usize,
    widening_c: f64,
    max_initialization_fraction: f64,
}

#[derive(Clone, Copy)]
struct A1cSearchOptions {
    sampling: A1cPanelSampling,
    n_init: usize,
    widening_c: f64,
    max_initialization_fraction: f64,
}

struct A1cRuntime<'a> {
    evaluator: &'a Py<PyAny>,
    initialization_nn_evals: &'a mut usize,
    total_nn_evals: &'a mut usize,
    initialization_blocked_cycles: &'a mut usize,
    nn_eval_budget: Option<usize>,
    nn_budget_blocked_cycles: &'a mut usize,
    nn_budget_blocked_rows: &'a mut usize,
    evaluator_calls: &'a mut usize,
    max_batch_size: &'a mut usize,
    initialization_evaluator_calls: &'a mut usize,
    initialization_max_batch_size: &'a mut usize,
}

struct A1cAdmissionRequest {
    chance_node_id: u32,
    pre_reveal_state: RustGameState,
    placement: Option<(i8, i8, i8, i8, bool)>,
    pick: Option<u16>,
    draw_seed: u64,
}

struct OLNode {
    prior: f64,
    visit_count: i32,
    value_sum: f64,
    // Reversible in-flight accounting. Decision nodes use the mutations to
    // visit_count/value_sum directly. Chance nodes reconstruct their current
    // probability-weighted Q from observation children, so they need these
    // separate fields to retain a full-strength virtual-loss penalty.
    virtual_visit_count: i32,
    virtual_value_sum: f64,
    // O(1) Hájek estimate over observation children with at least one real
    // visit. Updated incrementally when an observation mean changes.
    chance_visited_mass: f64,
    chance_weighted_value: f64,
    chance_post_actor: Option<u8>,
    chance_chooser_actor: Option<u8>,
    chance_backup: OLChanceBackup,
    chance_traversal: OLChanceTraversal,
    // Balanced traversal expands coalesced row multiplicities into one local
    // cycle. The schedule is shuffled at each cycle boundary and consumed when
    // the chance node is reached, after its parent action has been selected.
    chance_schedule: Vec<usize>,
    chance_route_count: u64,
    chance_balanced_start_visits: u64,
    chance_schedule_seed: u64,
    // A1c admits complete cycles atomically. Zero means the node is still in
    // unbiased sampled-backup mode below its initialization threshold.
    chance_initialized_cycles: usize,
    // Real sampled-backup visits already propagated before the first complete
    // panel was admitted. These remain in ancestors' running means, so report
    // them explicitly as a finite-budget dilution diagnostic.
    chance_preinit_visits: usize,
    children: Vec<(u16, u32)>,
    action: (Option<(i8, i8, i8, i8, bool)>, Option<u16>),
    is_expanded: bool,
    // A non-empty support makes this node the afterstate/chance node for the
    // action stored in it.  The child identity is the revealed PUBLIC state;
    // probabilities are fixed when the support is created and never inferred
    // from downstream visit allocation.
    chance_children: Vec<OLChanceChild>,
    // Network estimate installed without a real MCTS visit when A1c completes
    // a panel. It supplies the conditional row value until that public subtree
    // receives search visits, and it must never enter visit-based targets.
    bootstrap_value0: Option<f64>,
}

struct OLChanceChild {
    row: [u16; 4],
    probability: f64,
    multiplicity: usize,
    // Observation subtrees are allocated only when their row is sampled.
    // Registering a fixed support must not clone+step every counterfactual
    // state, especially for 70-row deck=8 chance nodes.
    node_id: Option<u32>,
    #[cfg(debug_assertions)]
    public_key: Option<Vec<u8>>,
}

#[derive(Clone)]
struct OLChancePanelRow {
    row: Vec<u16>,
    probability: f64,
    multiplicity: usize,
}

struct AdvisorOpenLoopOutput {
    children: Vec<(u16, i32, f64, f64)>,
    root_value0: f64,
    diagnostics: HashMap<String, f64>,
}

impl OLNode {
    fn new(prior: f64, action: (Option<(i8, i8, i8, i8, bool)>, Option<u16>)) -> Self {
        OLNode {
            prior,
            visit_count: 0,
            value_sum: 0.0,
            virtual_visit_count: 0,
            virtual_value_sum: 0.0,
            chance_visited_mass: 0.0,
            chance_weighted_value: 0.0,
            chance_post_actor: None,
            chance_chooser_actor: None,
            chance_backup: OLChanceBackup::Hajek,
            chance_traversal: OLChanceTraversal::Iid,
            chance_schedule: Vec::new(),
            chance_route_count: 0,
            chance_balanced_start_visits: 0,
            chance_schedule_seed: 0,
            chance_initialized_cycles: 0,
            chance_preinit_visits: 0,
            children: Vec::new(),
            action,
            is_expanded: false,
            chance_children: Vec::new(),
            bootstrap_value0: None,
        }
    }
}

fn ol_parse_chance_backup(value: &str) -> PyResult<OLChanceBackup> {
    match value {
        "sampled" => Ok(OLChanceBackup::Sampled),
        "hajek" | "renormalized" => Ok(OLChanceBackup::Hajek),
        _ => Err(PyValueError::new_err(format!(
            "chance_backup must be 'sampled' or 'hajek', got {value:?}"
        ))),
    }
}

fn ol_parse_chance_traversal(value: &str) -> PyResult<OLChanceTraversal> {
    match value {
        "iid" => Ok(OLChanceTraversal::Iid),
        "balanced" => Ok(OLChanceTraversal::Balanced),
        _ => Err(PyValueError::new_err(format!(
            "chance_traversal must be 'iid' or 'balanced', got {value:?}"
        ))),
    }
}

fn a1c_parse_panel_sampling(value: &str) -> PyResult<A1cPanelSampling> {
    match value {
        "balanced" => Ok(A1cPanelSampling::Balanced),
        "iid" => Ok(A1cPanelSampling::Iid),
        _ => Err(PyValueError::new_err(format!(
            "A1c panel sampling must be 'balanced' or 'iid', got {value:?}"
        ))),
    }
}

/// Preserve cycle identity for A1c. The lazy A1/A1b support builder below
/// intentionally coalesces duplicate rows, but A1c needs whole-cycle admission
/// so it can widen atomically and cluster diagnostics by cycle.
fn a1c_one_reveal_cycles(
    deck: &[u16],
    max_cycles: usize,
    sampling: A1cPanelSampling,
    seed: u64,
) -> PyResult<Vec<Vec<[u16; 4]>>> {
    if max_cycles == 0 {
        return Ok(Vec::new());
    }
    if deck.is_empty() || deck.len() < 4 || deck.len() % 4 != 0 {
        return Err(PyValueError::new_err(format!(
            "A1c one-reveal cycles require a non-empty bag divisible by four, got {}",
            deck.len()
        )));
    }
    let mut bag = deck.to_vec();
    bag.sort_unstable();
    if bag.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(PyValueError::new_err(
            "A1c one-reveal cycles require distinct domino ids",
        ));
    }
    let rows_per_cycle = bag.len() / 4;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cycles = Vec::with_capacity(max_cycles);
    for _ in 0..max_cycles {
        let mut cycle = Vec::with_capacity(rows_per_cycle);
        match sampling {
            A1cPanelSampling::Balanced => {
                let mut permutation = bag.clone();
                permutation.shuffle(&mut rng);
                for chunk in permutation.chunks_exact(4) {
                    let mut row: [u16; 4] = chunk
                        .try_into()
                        .expect("chunks_exact(4) always yields four tiles");
                    row.sort_unstable();
                    cycle.push(row);
                }
            }
            A1cPanelSampling::Iid => {
                for _ in 0..rows_per_cycle {
                    let mut permutation = bag.clone();
                    permutation.shuffle(&mut rng);
                    let mut row = [
                        permutation[0],
                        permutation[1],
                        permutation[2],
                        permutation[3],
                    ];
                    row.sort_unstable();
                    cycle.push(row);
                }
            }
        }
        cycles.push(cycle);
    }
    Ok(cycles)
}

/// Visit-controlled A1c width. `n_init` gates the first complete cycle; after
/// that, the preregistered sqrt schedule may request additional whole cycles.
fn a1c_target_cycles(visits: usize, n_init: usize, max_cycles: usize, c: f64) -> usize {
    if visits < n_init || max_cycles == 0 || !(c > 0.0 && c.is_finite()) {
        return 0;
    }
    let scheduled = (c * (visits as f64).sqrt()).ceil() as usize;
    scheduled.max(1).min(max_cycles)
}

/// Stable seeded tie-break for admission requests with equal real visits.
/// Node ids alone track root action insertion order, so using them directly
/// would spend a tight initialization budget preferentially on low action ids.
fn a1c_admission_tiebreak(seed: u64, wave: usize, chance_node_id: u32) -> u64 {
    let mut value = seed
        ^ (wave as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (chance_node_id as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Admit a complete cycle only if doing all of its missing network rows keeps
/// cumulative initialization work within the preregistered fraction. `total`
/// is the number of NN rows already evaluated, including prior initialization.
fn a1c_initialization_within_budget(
    initialization_nn_evals: usize,
    total_nn_evals: usize,
    additional_nn_evals: usize,
    max_fraction: f64,
) -> bool {
    if additional_nn_evals == 0 {
        return true;
    }
    if initialization_nn_evals > total_nn_evals
        || !(max_fraction > 0.0 && max_fraction <= 1.0 && max_fraction.is_finite())
    {
        return false;
    }
    let next_initialization = initialization_nn_evals.saturating_add(additional_nn_evals);
    let next_total = total_nn_evals.saturating_add(additional_nn_evals);
    (next_initialization as f64) <= max_fraction * next_total as f64 + 1e-12
}

/// Player-0 value used by selection for a one-reveal chance node.
///
/// Sampled mode is the ordinary Monte Carlo running mean. Hájek mode omits
/// unvisited outcomes and renormalizes registered probability over visited
/// mass. Its load-bearing assumption is that inclusion in the visited set is
/// value-independent: the reveal is chosen only after the pre-chance action and
/// is never PUCT-selected. If chance outcomes ever become value-selected, the
/// Hájek estimator is invalid. Once every outcome has a real visit it is the
/// exact fixed-panel expectation.
/// A separate chance-node virtual-loss overlay keeps batched traversal from
/// repeatedly selecting the same pre-reveal action while evaluations are in
/// flight; observation-child virtual loss is excluded from the base estimate.
fn ol_chance_value_p0(arena: &[OLNode], node_id: u32) -> f64 {
    let node = &arena[node_id as usize];
    debug_assert!(!node.chance_children.is_empty());
    if node.chance_backup == OLChanceBackup::Sampled {
        return if node.visit_count > 0 {
            node.value_sum / node.visit_count as f64
        } else {
            0.0
        };
    }
    let real_visits = node.visit_count - node.virtual_visit_count;
    debug_assert!(
        real_visits <= 0 || node.chance_visited_mass > 0.0,
        "a real chance-node visit must have a real observation visit"
    );
    let base_value0 = if node.chance_visited_mass > 0.0 {
        node.chance_weighted_value / node.chance_visited_mass
    } else {
        0.0
    };
    let virtual_visits = node.virtual_visit_count.max(0);
    if virtual_visits == 0 {
        return base_value0;
    }
    let real_visits = real_visits.max(0);
    let total = real_visits + virtual_visits;
    if total == 0 {
        base_value0
    } else {
        (base_value0 * real_visits as f64 + node.virtual_value_sum) / total as f64
    }
}

/// Probability-weighted conditional estimate for a fully materialized panel.
/// Real searched observations take precedence over their network bootstraps.
fn ol_panel_weighted_value(arena: &[OLNode], chance_node_id: u32) -> f64 {
    arena[chance_node_id as usize]
        .chance_children
        .iter()
        .filter(|outcome| outcome.probability > 0.0)
        .map(|outcome| {
            let observation_id = outcome
                .node_id
                .expect("every active panel outcome must have an observation node");
            let observation = &arena[observation_id as usize];
            let estimate = if observation.visit_count > 0 {
                observation.value_sum / observation.visit_count as f64
            } else {
                observation
                    .bootstrap_value0
                    .expect("every unvisited active panel outcome must be bootstrapped")
            };
            outcome.probability * estimate
        })
        .sum()
}

fn ol_rank_values(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let rank = (start + end - 1) as f64 / 2.0;
        for &index in &order[start..end] {
            ranks[index] = rank;
        }
        start = end;
    }
    ranks
}

fn ol_pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    // Called on rank vectors above; Pearson(rank(x), rank(y)) is Spearman rho.
    debug_assert_eq!(left.len(), right.len());
    if left.len() < 2 {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut covariance = 0.0;
    let mut left_ss = 0.0;
    let mut right_ss = 0.0;
    for (&x, &y) in left.iter().zip(right) {
        let dx = x - left_mean;
        let dy = y - right_mean;
        covariance += dx * dy;
        left_ss += dx * dx;
        right_ss += dy * dy;
    }
    if left_ss == 0.0 || right_ss == 0.0 {
        None
    } else {
        Some(covariance / (left_ss * right_ss).sqrt())
    }
}

fn ol_chance_diagnostics(arena: &[OLNode]) -> HashMap<String, f64> {
    let chance_nodes: Vec<&OLNode> = arena
        .iter()
        .filter(|node| !node.chance_children.is_empty())
        .collect();
    let mut visits = Vec::<i32>::new();
    let mut visited_outcomes = 0usize;
    let mut allocated_outcomes = 0usize;
    let mut observation_visits = 0i64;
    let mut visited_mass_sum = 0.0;
    let mut visit_weighted_visited_mass_sum = 0.0;
    let mut chance_node_visits = 0i64;
    let mut fully_visited_chance_nodes = 0usize;
    let mut outcome_count = 0usize;
    let mut balanced_route_count = 0u64;
    let mut balanced_completed_cycles = 0u64;
    for node in &chance_nodes {
        let node_visits = node.visit_count.max(0) as i64;
        chance_node_visits += node_visits;
        if node.chance_traversal == OLChanceTraversal::Balanced {
            debug_assert_eq!(
                node.chance_route_count + node.chance_balanced_start_visits,
                node_visits as u64
            );
            balanced_route_count += node.chance_route_count;
            if !node.chance_schedule.is_empty() {
                balanced_completed_cycles +=
                    node.chance_route_count / node.chance_schedule.len() as u64;
            }
        }
        let mut node_visited_mass = 0.0;
        let mut node_weighted_value = 0.0;
        let mut node_fully_visited = true;
        for outcome in &node.chance_children {
            if outcome.node_id.is_some() {
                allocated_outcomes += 1;
            }
            let count = outcome
                .node_id
                .map(|node_id| arena[node_id as usize].visit_count.max(0))
                .unwrap_or(0);
            outcome_count += 1;
            observation_visits += count as i64;
            visits.push(count);
            if count > 0 {
                visited_outcomes += 1;
                visited_mass_sum += outcome.probability;
                node_visited_mass += outcome.probability;
                let child_id = outcome
                    .node_id
                    .expect("a visited chance outcome must have an observation node");
                let child = &arena[child_id as usize];
                node_weighted_value +=
                    outcome.probability * child.value_sum / child.visit_count as f64;
            } else {
                node_fully_visited = false;
            }
        }
        if node.chance_backup == OLChanceBackup::PanelMean {
            let mut panel_weighted_value = 0.0;
            for outcome in &node.chance_children {
                if outcome.probability == 0.0 {
                    continue;
                }
                let child_id = outcome
                    .node_id
                    .expect("an active A1c outcome must have an observation node");
                let child = &arena[child_id as usize];
                let estimate = if child.visit_count > 0 {
                    child.value_sum / child.visit_count as f64
                } else {
                    child
                        .bootstrap_value0
                        .expect("an unvisited active A1c outcome must be bootstrapped")
                };
                panel_weighted_value += outcome.probability * estimate;
            }
            debug_assert!((node.chance_visited_mass - 1.0).abs() < 1e-9);
            debug_assert!((node.chance_weighted_value - panel_weighted_value).abs() < 1e-9);
        } else {
            debug_assert!((node.chance_visited_mass - node_visited_mass).abs() < 1e-9);
            debug_assert!((node.chance_weighted_value - node_weighted_value).abs() < 1e-9);
        }
        visit_weighted_visited_mass_sum += node_visits as f64 * node_visited_mass;
        if node_fully_visited {
            fully_visited_chance_nodes += 1;
        }
    }
    visits.sort_unstable();
    let percentile = |q: f64| -> f64 {
        if visits.is_empty() {
            return 0.0;
        }
        let index = ((visits.len() - 1) as f64 * q).ceil() as usize;
        visits[index] as f64
    };
    let n_chance = chance_nodes.len();
    let mut out = HashMap::new();
    out.insert("chance_nodes".to_string(), n_chance as f64);
    out.insert("support_outcomes".to_string(), outcome_count as f64);
    out.insert("visited_outcomes".to_string(), visited_outcomes as f64);
    out.insert("allocated_outcomes".to_string(), allocated_outcomes as f64);
    out.insert(
        "avoided_observation_allocations".to_string(),
        outcome_count.saturating_sub(allocated_outcomes) as f64,
    );
    out.insert("observation_visits".to_string(), observation_visits as f64);
    out.insert("chance_node_visits".to_string(), chance_node_visits as f64);
    out.insert(
        "balanced_route_count".to_string(),
        balanced_route_count as f64,
    );
    out.insert(
        "balanced_completed_cycles".to_string(),
        balanced_completed_cycles as f64,
    );
    out.insert(
        "fully_visited_chance_nodes".to_string(),
        fully_visited_chance_nodes as f64,
    );
    out.insert(
        "fully_visited_chance_node_fraction".to_string(),
        if n_chance == 0 {
            0.0
        } else {
            fully_visited_chance_nodes as f64 / n_chance as f64
        },
    );
    out.insert(
        "mean_support_outcomes_per_chance_node".to_string(),
        if n_chance == 0 {
            0.0
        } else {
            outcome_count as f64 / n_chance as f64
        },
    );
    out.insert(
        "mean_visited_probability_mass".to_string(),
        if n_chance == 0 {
            0.0
        } else {
            visited_mass_sum / n_chance as f64
        },
    );
    out.insert(
        "mean_unvisited_probability_mass".to_string(),
        if n_chance == 0 {
            0.0
        } else {
            1.0 - visited_mass_sum / n_chance as f64
        },
    );
    out.insert(
        "visit_weighted_visited_probability_mass".to_string(),
        if chance_node_visits == 0 {
            0.0
        } else {
            visit_weighted_visited_mass_sum / chance_node_visits as f64
        },
    );
    out.insert(
        "visit_weighted_unvisited_probability_mass".to_string(),
        if chance_node_visits == 0 {
            0.0
        } else {
            1.0 - visit_weighted_visited_mass_sum / chance_node_visits as f64
        },
    );
    out.insert("min_visits_per_outcome".to_string(), percentile(0.0));
    out.insert("median_visits_per_outcome".to_string(), percentile(0.5));
    out.insert("p90_visits_per_outcome".to_string(), percentile(0.9));
    out.insert("max_visits_per_outcome".to_string(), percentile(1.0));
    let mut visit_q_rank_correlations = Vec::new();
    let mut ranked_chance_actions = 0usize;
    for parent in arena {
        let mut action_visits = Vec::new();
        let mut action_q = Vec::new();
        for &(_, child_id) in &parent.children {
            let child = &arena[child_id as usize];
            if child.chance_children.is_empty() || child.visit_count <= 0 {
                continue;
            }
            let chooser = child
                .chance_chooser_actor
                .expect("registered chance node must retain its chooser actor");
            let q0 = ol_chance_value_p0(arena, child_id);
            action_visits.push(child.visit_count as f64);
            action_q.push(if chooser == 0 { q0 } else { -q0 });
        }
        // Two-action Spearman values are necessarily ±1 and turn the aggregate
        // into a sign vote. Require at least three siblings for a useful shape.
        if action_visits.len() < 3 {
            continue;
        }
        let visit_ranks = ol_rank_values(&action_visits);
        let q_ranks = ol_rank_values(&action_q);
        if let Some(correlation) = ol_pearson(&visit_ranks, &q_ranks) {
            ranked_chance_actions += action_visits.len();
            visit_q_rank_correlations.push(correlation);
        }
    }
    let spearman_mean = if visit_q_rank_correlations.is_empty() {
        0.0
    } else {
        // Fisher-z averaging avoids biasing the mean correlation toward zero.
        let mean_z = visit_q_rank_correlations
            .iter()
            .map(|&value| value.clamp(-1.0 + 1e-12, 1.0 - 1e-12).atanh())
            .sum::<f64>()
            / visit_q_rank_correlations.len() as f64;
        mean_z.tanh()
    };
    let mut sorted_correlations = visit_q_rank_correlations.clone();
    sorted_correlations.sort_by(f64::total_cmp);
    let correlation_percentile = |q: f64| -> f64 {
        if sorted_correlations.is_empty() {
            return 0.0;
        }
        let index = ((sorted_correlations.len() - 1) as f64 * q).ceil() as usize;
        sorted_correlations[index]
    };
    out.insert(
        "chance_action_visit_q_rank_spearman_mean".to_string(),
        spearman_mean,
    );
    out.insert(
        "chance_action_visit_q_rank_spearman_p10".to_string(),
        correlation_percentile(0.1),
    );
    out.insert(
        "chance_action_visit_q_rank_spearman_median".to_string(),
        correlation_percentile(0.5),
    );
    out.insert(
        "chance_action_visit_q_rank_spearman_p90".to_string(),
        correlation_percentile(0.9),
    );
    out.insert(
        "chance_action_visit_q_rank_parent_groups".to_string(),
        visit_q_rank_correlations.len() as f64,
    );
    out.insert(
        "chance_action_visit_q_rank_actions".to_string(),
        ranked_chance_actions as f64,
    );
    out
}

fn ol_reported_value_sum(arena: &[OLNode], node_id: u32) -> f64 {
    let node = &arena[node_id as usize];
    if node.chance_children.is_empty() {
        node.value_sum
    } else {
        ol_chance_value_p0(arena, node_id) * node.visit_count as f64
    }
}

fn comb4(n: usize) -> u64 {
    if n < 4 {
        0
    } else {
        (n as u64 * (n - 1) as u64 * (n - 2) as u64 * (n - 3) as u64) / 24
    }
}

/// Fixed public-row support for the A1 one-reveal probe. Small bags are
/// exhaustive; larger bags use X shuffled permutation cycles, so every tile is
/// exposed exactly X times. Duplicate rows across cycles are coalesced while
/// retaining their total probability mass.
fn ol_one_reveal_panel(
    deck: &[u16],
    exposure: usize,
    enum_max_rows: u64,
    seed: u64,
) -> PyResult<Vec<OLChancePanelRow>> {
    if exposure == 0 {
        return Ok(Vec::new());
    }
    if deck.is_empty() {
        // No future reveal exists. Keeping the treatment a no-op here makes the
        // probe safe on terminal-adjacent roots that bypass the exact router.
        return Ok(Vec::new());
    }
    if deck.len() < 4 || deck.len() % 4 != 0 {
        return Err(PyValueError::new_err(format!(
            "one-reveal support requires a non-empty bag divisible by four, got {}",
            deck.len()
        )));
    }
    let mut bag = deck.to_vec();
    bag.sort_unstable();
    let rows = comb4(bag.len());
    let raw: Vec<Vec<u16>> = if rows <= enum_max_rows {
        let mut out = Vec::with_capacity(rows as usize);
        for i in 0..bag.len() {
            for j in (i + 1)..bag.len() {
                for k in (j + 1)..bag.len() {
                    for l in (k + 1)..bag.len() {
                        out.push(vec![bag[i], bag[j], bag[k], bag[l]]);
                    }
                }
            }
        }
        out
    } else {
        let mut out = Vec::with_capacity(deck.len() / 4 * exposure);
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..exposure {
            let mut permutation = bag.clone();
            permutation.shuffle(&mut rng);
            for chunk in permutation.chunks_exact(4) {
                let mut row = chunk.to_vec();
                row.sort_unstable();
                out.push(row);
            }
        }
        out
    };
    let total = raw.len();
    let mut counts: HashMap<Vec<u16>, usize> = HashMap::new();
    for row in raw {
        *counts.entry(row).or_insert(0) += 1;
    }
    let mut support: Vec<OLChancePanelRow> = counts
        .into_iter()
        .map(|(row, count)| OLChancePanelRow {
            row,
            probability: count as f64 / total as f64,
            multiplicity: count,
        })
        .collect();
    support.sort_by(|a, b| a.row.cmp(&b.row));
    let mass: f64 = support.iter().map(|x| x.probability).sum();
    debug_assert!((mass - 1.0).abs() < 1e-12);
    Ok(support)
}

/// Redeterminize while pinning only the next public reveal. The row is sampled
/// outside the action-selection path; remaining hidden order is independently
/// shuffled and therefore stays unavailable to the tree/network.
fn redeterminize_with_first_row(
    state: &RustGameState,
    row: &[u16],
    seed: u64,
) -> PyResult<RustGameState> {
    if row.len() != 4 {
        return Err(PyValueError::new_err(
            "a Kingdomino reveal row must contain four tiles",
        ));
    }
    let mut remaining = state.deck.clone();
    for &tile in row {
        let Some(index) = remaining.iter().position(|&candidate| candidate == tile) else {
            return Err(PyValueError::new_err(format!(
                "forced reveal tile {tile} is not in the public bag"
            )));
        };
        remaining.swap_remove(index);
    }
    remaining.shuffle(&mut StdRng::seed_from_u64(seed));
    let mut out = state.cloned();
    out.deck = row.to_vec();
    out.deck.extend(remaining);
    Ok(out)
}

fn ol_register_chance_support(
    arena: &mut Vec<OLNode>,
    chance_node_id: u32,
    config: OLChanceConfig<'_>,
    chooser_actor: u8,
) -> PyResult<()> {
    if !arena[chance_node_id as usize].chance_children.is_empty() {
        let node = &arena[chance_node_id as usize];
        debug_assert_eq!(node.chance_backup, config.backup);
        debug_assert_eq!(node.chance_traversal, config.traversal);
        debug_assert_eq!(node.chance_chooser_actor, Some(chooser_actor));
        return Ok(());
    }
    let children: Vec<OLChanceChild> = config
        .panel
        .iter()
        .map(|outcome| {
            let row: [u16; 4] = outcome
                .row
                .as_slice()
                .try_into()
                .expect("one-reveal panel rows always contain four tiles");
            OLChanceChild {
                row,
                probability: outcome.probability,
                multiplicity: outcome.multiplicity,
                node_id: None,
                #[cfg(debug_assertions)]
                public_key: None,
            }
        })
        .collect();
    debug_assert!(
        children.windows(2).all(|pair| pair[0].row < pair[1].row),
        "one-reveal chance support must be strictly sorted by row"
    );
    let mass: f64 = children.iter().map(|x| x.probability).sum();
    if children.is_empty() || (mass - 1.0).abs() > 1e-9 {
        return Err(PyValueError::new_err(format!(
            "one-reveal fixed support has probability mass {mass}, expected 1"
        )));
    }
    let mut schedule = Vec::new();
    if config.traversal == OLChanceTraversal::Balanced {
        schedule.reserve(children.iter().map(|child| child.multiplicity).sum());
        for (index, child) in children.iter().enumerate() {
            schedule.extend(std::iter::repeat_n(index, child.multiplicity));
        }
        debug_assert!(!schedule.is_empty());
    }
    let node = &mut arena[chance_node_id as usize];
    node.chance_children = children;
    node.chance_chooser_actor = Some(chooser_actor);
    node.chance_backup = config.backup;
    node.chance_traversal = config.traversal;
    node.chance_schedule = schedule;
    node.chance_route_count = 0;
    node.chance_balanced_start_visits = 0;
    node.chance_schedule_seed =
        config.schedule_seed ^ (chance_node_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    Ok(())
}

/// Select the row only after PUCT has committed to the pre-reveal action.
/// Consequently neither IID nor locally balanced routing leaks the row into
/// that action choice. Balanced mode consumes a shuffled multiplicity-expanded
/// cycle local to this chance node before reshuffling for the next cycle.
/// Conditional on the number of times the node is reached, that prefix is a
/// value-independent uniform sample of the expanded panel: PUCT controls only
/// prefix length, never schedule order. This is the load-bearing condition that
/// keeps the Hájek visited-set estimator valid under balanced traversal.
fn ol_select_chance_row(arena: &mut [OLNode], chance_node_id: u32, iid_seed: u64) -> [u16; 4] {
    let node = &mut arena[chance_node_id as usize];
    match node.chance_traversal {
        OLChanceTraversal::Iid => {
            let mut rng = StdRng::seed_from_u64(iid_seed ^ 0xC0A1_5EED_5EED_C0A1);
            let target = rng.r#gen::<f64>();
            let mut cumulative = 0.0;
            node.chance_children
                .iter()
                .find(|outcome| {
                    cumulative += outcome.probability;
                    target < cumulative
                })
                .unwrap_or_else(|| node.chance_children.last().unwrap())
                .row
        }
        OLChanceTraversal::Balanced => {
            let cycle_len = node.chance_schedule.len();
            debug_assert!(cycle_len > 0);
            let offset = node.chance_route_count as usize % cycle_len;
            if offset == 0 {
                let cycle = node.chance_route_count / cycle_len as u64;
                node.chance_schedule.shuffle(&mut StdRng::seed_from_u64(
                    node.chance_schedule_seed ^ cycle.wrapping_mul(0xD1B5_4A32_D192_ED03),
                ));
            }
            let outcome_index = node.chance_schedule[offset];
            node.chance_route_count += 1;
            node.chance_children[outcome_index].row
        }
    }
}

fn ol_route_chance_observation(
    arena: &mut Vec<OLNode>,
    chance_node_id: u32,
    revealed_state: &RustGameState,
    post_actor: u8,
) -> PyResult<(u32, f64)> {
    let mut row: [u16; 4] = revealed_state
        .current_row
        .as_slice()
        .try_into()
        .map_err(|_| {
            PyValueError::new_err("a routed Kingdomino reveal row must contain four tiles")
        })?;
    row.sort_unstable();
    match arena[chance_node_id as usize].chance_post_actor {
        Some(expected) if expected != post_actor => {
            return Err(PyValueError::new_err(format!(
                "chance panel changed post-reveal actor from {expected} to {post_actor}"
            )));
        }
        None => arena[chance_node_id as usize].chance_post_actor = Some(post_actor),
        _ => {}
    }
    let outcome_index = arena[chance_node_id as usize]
        .chance_children
        .binary_search_by(|outcome| outcome.row.cmp(&row))
        .map_err(|_| {
            PyValueError::new_err("sampled reveal is absent from its closed one-reveal support")
        })?;
    #[cfg(debug_assertions)]
    let public_key = {
        let mut key = Vec::with_capacity(1024);
        chance_public_state_key_v1_bytes(revealed_state, &mut key);
        key
    };
    let probability = arena[chance_node_id as usize].chance_children[outcome_index].probability;
    if let Some(node_id) = arena[chance_node_id as usize].chance_children[outcome_index].node_id {
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            arena[chance_node_id as usize].chance_children[outcome_index]
                .public_key
                .as_ref(),
            Some(&public_key),
            "the same chance node and row must route to one public information state"
        );
        return Ok((node_id, probability));
    }
    let node_id = arena.len() as u32;
    arena.push(OLNode::new(1.0, (None, None)));
    arena[chance_node_id as usize].chance_children[outcome_index].node_id = Some(node_id);
    #[cfg(debug_assertions)]
    {
        arena[chance_node_id as usize].chance_children[outcome_index].public_key = Some(public_key);
    }
    Ok((node_id, probability))
}

fn a1c_uniform_row(deck: &[u16], seed: u64) -> [u16; 4] {
    debug_assert!(deck.len() >= 4);
    let mut permutation = deck.to_vec();
    permutation.shuffle(&mut StdRng::seed_from_u64(seed ^ 0xA1C1_1D5A_7EED_0001));
    let mut row = [
        permutation[0],
        permutation[1],
        permutation[2],
        permutation[3],
    ];
    row.sort_unstable();
    row
}

fn a1c_ensure_outcome(arena: &mut Vec<OLNode>, chance_node_id: u32, row: [u16; 4]) -> usize {
    match arena[chance_node_id as usize]
        .chance_children
        .binary_search_by(|outcome| outcome.row.cmp(&row))
    {
        Ok(index) => index,
        Err(index) => {
            arena[chance_node_id as usize].chance_children.insert(
                index,
                OLChanceChild {
                    row,
                    probability: 0.0,
                    multiplicity: 0,
                    node_id: None,
                    #[cfg(debug_assertions)]
                    public_key: None,
                },
            );
            index
        }
    }
}

fn a1c_prepare_sampled_node(
    arena: &mut Vec<OLNode>,
    chance_node_id: u32,
    chooser_actor: u8,
    schedule_seed: u64,
) {
    let node = &mut arena[chance_node_id as usize];
    if node.chance_chooser_actor.is_none() {
        node.chance_chooser_actor = Some(chooser_actor);
        node.chance_backup = OLChanceBackup::Sampled;
        node.chance_traversal = OLChanceTraversal::Iid;
        node.chance_schedule_seed =
            schedule_seed ^ (chance_node_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    } else {
        debug_assert_eq!(node.chance_chooser_actor, Some(chooser_actor));
    }
}

struct A1cBootstrapRequest {
    row: [u16; 4],
    state: RustGameState,
    actor: u8,
    legal: Vec<(u16, Option<(i8, i8, i8, i8, bool)>, Option<u16>)>,
    my: Array3<f32>,
    opp: Array3<f32>,
    flat: Array1<f32>,
}

fn a1c_evaluate_bootstraps(
    ev: &Py<PyAny>,
    requests: &[A1cBootstrapRequest],
) -> PyResult<Vec<(f64, Vec<f64>)>> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    let idxs_per: Vec<Vec<i64>> = requests
        .iter()
        .map(|request| request.legal.iter().map(|action| action.0 as i64).collect())
        .collect();
    Python::attach(|py| -> PyResult<Vec<(f64, Vec<f64>)>> {
        let my_views: Vec<_> = requests.iter().map(|request| request.my.view()).collect();
        let opp_views: Vec<_> = requests.iter().map(|request| request.opp.view()).collect();
        let flat_views: Vec<_> = requests.iter().map(|request| request.flat.view()).collect();
        let my = numpy::ndarray::stack(Axis(0), &my_views)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let opp = numpy::ndarray::stack(Axis(0), &opp_views)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let flat = numpy::ndarray::stack(Axis(0), &flat_views)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let index_arrays: Vec<_> = idxs_per
            .iter()
            .map(|indices| indices.clone().into_pyarray(py))
            .collect();
        let index_list = PyList::new(py, index_arrays)?;
        let result = ev.bind(py).call1((
            my.into_pyarray(py),
            opp.into_pyarray(py),
            flat.into_pyarray(py),
            index_list,
        ))?;
        let tuple = result.downcast::<PyTuple>()?;
        let values: Vec<f32> = tuple
            .get_item(0)?
            .downcast::<PyArray1<f32>>()?
            .readonly()
            .as_slice()?
            .to_vec();
        let logits_item = tuple.get_item(1)?;
        let logits = logits_item.downcast::<PyList>()?;
        if values.len() != requests.len() || logits.len() != requests.len() {
            return Err(PyValueError::new_err(
                "A1c bootstrap evaluator returned the wrong batch length",
            ));
        }
        let mut output = Vec::with_capacity(requests.len());
        for index in 0..requests.len() {
            let gathered: Vec<f64> = logits
                .get_item(index)?
                .downcast::<PyArray1<f32>>()?
                .readonly()
                .as_slice()?
                .iter()
                .map(|&value| value as f64)
                .collect();
            if gathered.len() != requests[index].legal.len() {
                return Err(PyValueError::new_err(
                    "A1c bootstrap evaluator returned the wrong policy length",
                ));
            }
            let value0 = if requests[index].actor == 0 {
                values[index] as f64
            } else {
                -(values[index] as f64)
            };
            output.push((value0, softmax_f64(&gathered)));
        }
        Ok(output)
    })
}

#[allow(clippy::too_many_arguments)]
fn a1c_admit_next_cycle(
    arena: &mut Vec<OLNode>,
    chance_node_id: u32,
    pre_reveal_state: &RustGameState,
    placement: Option<(i8, i8, i8, i8, bool)>,
    pick: Option<u16>,
    draw_seed: u64,
    a1c: A1cChanceConfig<'_>,
    runtime: &mut A1cRuntime<'_>,
) -> PyResult<bool> {
    let current_cycles = arena[chance_node_id as usize].chance_initialized_cycles;
    if current_cycles >= a1c.cycles.len() {
        return Ok(false);
    }
    let next_cycles = current_cycles + 1;
    let mut counts = HashMap::<[u16; 4], usize>::new();
    for cycle in &a1c.cycles[..next_cycles] {
        for &row in cycle {
            *counts.entry(row).or_insert(0) += 1;
        }
    }
    let mut rows: Vec<[u16; 4]> = counts.keys().copied().collect();
    rows.sort_unstable();
    let mut requests = Vec::new();
    for &row in &rows {
        let existing_node = arena[chance_node_id as usize]
            .chance_children
            .binary_search_by(|outcome| outcome.row.cmp(&row))
            .ok()
            .and_then(|index| arena[chance_node_id as usize].chance_children[index].node_id);
        let already_estimated = existing_node.is_some_and(|node_id| {
            let node = &arena[node_id as usize];
            node.visit_count > 0 || node.bootstrap_value0.is_some()
        });
        if already_estimated {
            continue;
        }
        let row_seed = draw_seed
            ^ row.iter().fold(0xA1C1_C1E0_0000_0001u64, |hash, &tile| {
                hash.rotate_left(11) ^ tile as u64
            });
        let forced = redeterminize_with_first_row(pre_reveal_state, &row, row_seed)?;
        let revealed = forced.step(placement, pick)?;
        if revealed.phase == GAME_OVER {
            return Err(PyValueError::new_err(
                "A1c bootstrap unexpectedly reached a terminal state",
            ));
        }
        let actor = revealed.actor()?;
        let legal = revealed.legal_actions_indexed();
        let (my, opp, flat) = revealed.encode_arrays(actor)?;
        requests.push(A1cBootstrapRequest {
            row,
            state: revealed,
            actor,
            legal,
            my,
            opp,
            flat,
        });
    }
    if runtime
        .nn_eval_budget
        .is_some_and(|budget| runtime.total_nn_evals.saturating_add(requests.len()) > budget)
    {
        *runtime.nn_budget_blocked_cycles += 1;
        *runtime.nn_budget_blocked_rows += requests.len();
        return Ok(false);
    }
    if !a1c_initialization_within_budget(
        *runtime.initialization_nn_evals,
        *runtime.total_nn_evals,
        requests.len(),
        a1c.max_initialization_fraction,
    ) {
        *runtime.initialization_blocked_cycles += 1;
        return Ok(false);
    }
    let evaluated = a1c_evaluate_bootstraps(runtime.evaluator, &requests)?;
    if !requests.is_empty() {
        *runtime.evaluator_calls += 1;
        *runtime.initialization_evaluator_calls += 1;
        *runtime.max_batch_size = (*runtime.max_batch_size).max(requests.len());
        *runtime.initialization_max_batch_size =
            (*runtime.initialization_max_batch_size).max(requests.len());
    }

    // Commit only after the whole evaluator batch succeeds. This keeps cycle
    // admission atomic even if Python raises or returns a malformed batch.
    for (request, (value0, priors)) in requests.iter().zip(evaluated) {
        a1c_ensure_outcome(arena, chance_node_id, request.row);
        let (observation_id, _) =
            ol_route_chance_observation(arena, chance_node_id, &request.state, request.actor)?;
        if arena[observation_id as usize].is_expanded {
            ol_add_missing_children(arena, observation_id, &request.legal, &priors);
        } else {
            for (index, &(action_index, placement, pick)) in request.legal.iter().enumerate() {
                let child_id = arena.len() as u32;
                arena.push(OLNode::new(priors[index], (placement, pick)));
                arena[observation_id as usize]
                    .children
                    .push((action_index, child_id));
            }
            arena[observation_id as usize].is_expanded = true;
        }
        arena[observation_id as usize].bootstrap_value0 = Some(value0);
    }

    let raw_rows: usize = counts.values().sum();
    let mut schedule = Vec::with_capacity(raw_rows);
    for outcome in &mut arena[chance_node_id as usize].chance_children {
        let multiplicity = counts.get(&outcome.row).copied().unwrap_or(0);
        outcome.multiplicity = multiplicity;
        outcome.probability = multiplicity as f64 / raw_rows as f64;
    }
    for (index, outcome) in arena[chance_node_id as usize]
        .chance_children
        .iter()
        .enumerate()
    {
        schedule.extend(std::iter::repeat_n(index, outcome.multiplicity));
    }
    let weighted = ol_panel_weighted_value(arena, chance_node_id);
    let chance_node = &mut arena[chance_node_id as usize];
    if current_cycles == 0 {
        chance_node.chance_preinit_visits =
            (chance_node.visit_count - chance_node.virtual_visit_count).max(0) as usize;
    }
    chance_node.chance_schedule = schedule;
    chance_node.chance_route_count = 0;
    chance_node.chance_balanced_start_visits =
        (chance_node.visit_count - chance_node.virtual_visit_count).max(0) as u64;
    chance_node.chance_traversal = OLChanceTraversal::Balanced;
    chance_node.chance_backup = OLChanceBackup::PanelMean;
    chance_node.chance_visited_mass = 1.0;
    chance_node.chance_weighted_value = weighted;
    chance_node.chance_initialized_cycles = next_cycles;
    *runtime.initialization_nn_evals += requests.len();
    *runtime.total_nn_evals += requests.len();
    Ok(true)
}

/// PUCT child selection for the open-loop tree.  Considers only children whose
/// joint index is legal in THIS simulation's concrete state (at deep nodes the
/// concrete current_row differs across determinizations).  Returns the chosen
/// child's id plus the action DECODED against this state; None (a counted
/// dead-end) when no child is legal here, which stops the descent.  Actor comes
/// from the concrete state, not the (stateless) node.
/// Run10 pick-group visit floors: at shallow non-root nodes, guarantee every
/// PICK-GROUP (joint_idx % 5 — the codec is placement*5 + pick, and picks are
/// the only interactive dimension in 2p Kingdomino) a minimum visit share.
/// Prior-guided PUCT compounds starvation with depth (measured: a game-losing
/// opponent reply at 4.7% prior received 0.6% of 3200 sims); the floor gives
/// each pick branch enough visits to reveal its value, after which normal
/// PUCT takes over on merit. Applied at node depths min_depth..=max_depth.
///
/// TRAINING (BatchedMCTS) always sets min_depth=1, i.e. NEVER the root: root
/// visit counts BECOME the policy target, so forcing there would teach the
/// policy that every forced pick is good (root pick-groups are also not
/// starved at training sim counts, and the root heals itself once child Q
/// values are accurate — so policy targets need no forced-visit subtraction).
///
/// The ADVISOR path may set min_depth=0 to include the root. It records no
/// training targets, so the objection above does not apply, and the Phase A
/// allocation study needs a root-level arm to localize whether secondary-pick
/// overvaluation comes from starvation at the root or inside the reply nodes.
#[derive(Clone, Copy)]
struct PickFloor {
    frac: f64,
    min_depth: usize,
    max_depth: usize,
    min_visits: i32, // don't force below this node visit count (too noisy)
}

fn ol_select_child(
    arena: &[OLNode],
    node_id: u32,
    state: &RustGameState,
    fpu: f64,
    cpuct: f64,
    fallback_count: &mut u32,
    missing_child_count: &mut u32,
    pick_filter: Option<u16>,
) -> Option<(u32, Option<(i8, i8, i8, i8, bool)>, Option<u16>)> {
    let node = &arena[node_id as usize];
    // Both lists are sorted ascending by joint index — legal_actions_indexed()
    // sorts, and node.children is kept ascending (ol_add_missing_children inserts
    // in order).  A two-pointer merge then finds matches in O(n+m) with NO
    // allocation, replacing the per-call HashMap.  The merge is correct ONLY if
    // both are strictly ascending; assert that in debug builds (free in release).
    let legal = state.legal_actions_indexed();
    let children = &node.children;
    debug_assert!(
        legal.windows(2).all(|w| w[0].0 < w[1].0),
        "legal_actions_indexed() is not strictly ascending"
    );
    debug_assert!(
        children.windows(2).all(|w| w[0].0 < w[1].0),
        "OLNode children are not strictly ascending — invariant violated"
    );

    let actor = state.actor().expect("non-terminal node has an actor");
    // A bootstrapped A1c observation has policy priors but intentionally has no
    // real visit. Give those priors one unit of selection scale without
    // fabricating a visit that could leak into a training target.
    let selection_visits = if node.visit_count == 0 && node.bootstrap_value0.is_some() {
        1
    } else {
        node.visit_count
    };
    let sqrt_n = (selection_visits.max(0) as f64).sqrt();

    let mut best_score = f64::NEG_INFINITY;
    let mut best_cid: Option<u32> = None;
    let mut best_action: Option<(Option<(i8, i8, i8, i8, bool)>, Option<u16>)> = None;
    let mut has_missing = false;

    // ci advances MONOTONICALLY across the whole legal sweep (never reset) — that
    // is what makes the merge O(n+m); resetting it would be O(n*m) and wrong.
    let mut ci = 0usize;
    for &(legal_idx, placement, pick) in &legal {
        while ci < children.len() && children[ci].0 < legal_idx {
            ci += 1;
        }
        if ci < children.len() && children[ci].0 == legal_idx {
            // Pick-floor restriction: only actions in the forced pick-group are
            // candidates.  The merge still walks EVERY legal action so that
            // has_missing stays a global property of the node — a missing child
            // outside the group must still halt the descent for expansion.
            if let Some(pf) = pick_filter {
                if legal_idx % 5 != pf {
                    continue;
                }
            }
            // This legal action has a stored child — score it.  Scoring order is
            // ascending by joint index (same as the old children-order loop), so
            // the strict-`>` tie-break selects the SAME child: bit-identical.
            let cid = children[ci].1;
            let child = &arena[cid as usize];
            let q0 = if !child.chance_children.is_empty() {
                ol_chance_value_p0(arena, cid)
            } else if child.visit_count > 0 {
                child.value_sum / child.visit_count as f64
            } else {
                if actor == 0 { fpu } else { -fpu }
            };
            let q = if actor == 0 { q0 } else { -q0 };
            let u = cpuct * child.prior * sqrt_n / (1.0 + child.visit_count as f64);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best_cid = Some(cid);
                best_action = Some((placement, pick));
            }
        } else {
            // legal_idx not present in children — a missing child (Issue 2).
            has_missing = true;
        }
    }

    // Issue 2: if ANY legal action lacks a stored child, stop the descent so
    // update() adds the missing children — even if some matched.  Selecting among
    // only the present children would permanently exclude the missing ones and
    // bias PUCT.  We finish the full merge before deciding (rather than R1's
    // first-missing early return); the result is identical (None), and this leaves
    // room to later return the best present child while queuing the missing for
    // expansion, should the missing-child rate ever become non-negligible.
    if has_missing {
        *missing_child_count += 1;
        return None;
    }

    match best_cid {
        Some(cid) => {
            let (placement, pick) = best_action.unwrap();
            Some((cid, placement, pick))
        }
        None => {
            // With a pick_filter this just means the forced group has no legal
            // action under THIS determinization (open-loop legality varies per
            // det) — an expected miss, not a dead-end: the caller retries
            // unfiltered.  Without a filter it is the Issue-2-era defensive
            // dead-end, kept counted.
            if pick_filter.is_none() {
                *fallback_count += 1;
            }
            None
        }
    }
}

/// Descend by PUCT to an unexpanded or terminal leaf, stepping a concrete
/// simulation state forward with each selected action.  No lazy state storage —
/// the concrete state is threaded as a local.  Returns (path of node ids, the
/// actor at each NON-leaf node on the path [for VL framing], leaf concrete state).
/// If `node_id` (at an eligible depth) has a pick-group whose visit share is
/// below the floor, return the MOST-deficient group's pick index (idx % 5).
/// Groups are computed over stored children (the union of legalities seen so
/// far) — per-det legality is handled by the caller's unfiltered retry.
fn ol_pick_floor_group(arena: &[OLNode], node_id: u32, floor: &PickFloor) -> Option<u16> {
    let node = &arena[node_id as usize];
    if node.visit_count < floor.min_visits {
        return None;
    }
    let mut group_visits = [0i64; 5];
    let mut group_present = [false; 5];
    for &(idx, cid) in &node.children {
        let g = (idx % 5) as usize;
        group_present[g] = true;
        group_visits[g] += arena[cid as usize].visit_count as i64;
    }
    let total: i64 = group_visits.iter().sum();
    if total <= 0 || group_present.iter().filter(|&&p| p).count() < 2 {
        return None; // single pick-group (or nothing visited yet): floor is moot
    }
    let mut best_g: Option<u16> = None;
    let mut best_deficit = 0.0f64;
    for g in 0..5 {
        if !group_present[g] {
            continue;
        }
        let share = group_visits[g] as f64 / total as f64;
        let deficit = floor.frac - share;
        if deficit > best_deficit {
            best_deficit = deficit;
            best_g = Some(g as u16);
        }
    }
    best_g
}

fn ol_descend(
    arena: &mut Vec<OLNode>,
    root_id: u32,
    mut state: RustGameState, // owned: the caller's `det` is moved in, not cloned
    fpu: f64,
    cpuct: f64,
    fallback_count: &mut u32,
    missing_child_count: &mut u32,
    pick_floor: Option<PickFloor>,
    chance_config: Option<OLChanceConfig<'_>>,
) -> PyResult<(
    Vec<u32>,
    Vec<u8>,
    RustGameState,
    Option<(usize, f64)>,
    Option<A1cAdmissionRequest>,
)> {
    let mut path: Vec<u32> = vec![root_id];
    let mut actors: Vec<u8> = Vec::new();
    let mut node_id = root_id;
    let mut chance_step: Option<(usize, f64)> = None;
    let mut a1c_admission_request: Option<A1cAdmissionRequest> = None;
    // `state` is already owned (moved in); no clone needed — it is stepped in
    // place as we descend and returned as the leaf state.
    loop {
        let expanded = arena[node_id as usize].is_expanded;
        let terminal = state.phase == GAME_OVER;
        if !expanded || terminal {
            break;
        }
        let actor = state.actor()?;
        // Pick-group visit floor: at node depths min_depth..=max_depth (the
        // root is path.len()-1 == 0, included only when min_depth == 0 — see
        // PickFloor), if a pick-group is starved below its floor share,
        // restrict this selection to that group (best child within it still
        // chosen by PUCT).  If the forced group has no legal action under this
        // determinization, fall through to normal unfiltered selection.
        let depth = path.len() - 1;
        let mut selected: Option<Option<(u32, Option<(i8, i8, i8, i8, bool)>, Option<u16>)>> = None;
        if let Some(pf) = &pick_floor {
            if depth >= pf.min_depth && depth <= pf.max_depth {
                if let Some(g) = ol_pick_floor_group(arena, node_id, pf) {
                    let missing_before = *missing_child_count;
                    let r = ol_select_child(
                        arena,
                        node_id,
                        &state,
                        fpu,
                        cpuct,
                        fallback_count,
                        missing_child_count,
                        Some(g),
                    );
                    if r.is_some() || *missing_child_count > missing_before {
                        // Either a forced-group child was selected, or the node
                        // has missing children (a GLOBAL stop — must expand).
                        selected = Some(r);
                    }
                    // else: group det-illegal here — retry unfiltered below.
                }
            }
        }
        let step = match selected {
            Some(r) => r,
            None => ol_select_child(
                arena,
                node_id,
                &state,
                fpu,
                cpuct,
                fallback_count,
                missing_child_count,
                None,
            ),
        };
        match step {
            None => break, // dead-end / missing children: re-evaluate this node as the leaf
            Some((child_id, placement, pick)) => {
                let split_here = chance_step.is_none()
                    && chance_config.is_some()
                    && <Kingdomino as search::Game>::is_stochastic(&state, (placement, pick));
                if split_here {
                    let config = chance_config.expect("split_here requires chance config");
                    let row = if let Some(a1c) = config.a1c {
                        a1c_prepare_sampled_node(arena, child_id, actor, config.schedule_seed);
                        if arena[child_id as usize].chance_initialized_cycles < a1c.cycles.len() {
                            // Admission is deliberately deferred until every
                            // path in this leaf-parallel wave has removed VL and
                            // backed up under the estimator it traversed.
                            a1c_admission_request = Some(A1cAdmissionRequest {
                                chance_node_id: child_id,
                                pre_reveal_state: state.cloned(),
                                placement,
                                pick,
                                draw_seed: config.draw_seed,
                            });
                        }
                        if arena[child_id as usize].chance_initialized_cycles > 0 {
                            ol_select_chance_row(arena, child_id, config.draw_seed)
                        } else {
                            let sampled = a1c_uniform_row(&state.deck, config.draw_seed);
                            let outcome = a1c_ensure_outcome(arena, child_id, sampled);
                            arena[child_id as usize].chance_children[outcome].probability =
                                1.0 / comb4(state.deck.len()) as f64;
                            sampled
                        }
                    } else {
                        ol_register_chance_support(arena, child_id, config, actor)?;
                        if config.bootstrap_full_panel
                            && arena[child_id as usize].chance_backup != OLChanceBackup::PanelMean
                        {
                            a1c_admission_request = Some(A1cAdmissionRequest {
                                chance_node_id: child_id,
                                pre_reveal_state: state.cloned(),
                                placement,
                                pick,
                                draw_seed: config.draw_seed,
                            });
                        }
                        ol_select_chance_row(arena, child_id, config.draw_seed)
                    };
                    // PUCT has already committed to child_id. Pinning the row at
                    // this point prevents panel construction or traversal from
                    // changing the preceding action's legality or score.
                    state = redeterminize_with_first_row(&state, &row, config.draw_seed)?;
                }
                actors.push(actor);
                state = state.step(placement, pick)?;
                path.push(child_id);
                node_id = child_id;
                if split_here {
                    let post_actor = state.actor()?;
                    let (observation_id, outcome_probability) =
                        ol_route_chance_observation(arena, child_id, &state, post_actor)?;
                    let chance_index = path.len() - 1;
                    // This is a stochastic edge, not a player choice. The actor is
                    // used only to frame reversible virtual loss on the observation
                    // node; final backup remains in the player-0 frame.
                    actors.push(post_actor);
                    path.push(observation_id);
                    node_id = observation_id;
                    chance_step = Some((chance_index, outcome_probability));
                }
            }
        }
    }
    Ok((path, actors, state, chance_step, a1c_admission_request))
}

/// Issue 2: add to `node_id` any child whose legal joint index is not already
/// present (a later determinization can reach an expanded node with legal actions
/// the original expansion never saw — the domino-in-hand differs across decks).
/// New children take their priors from THIS determinization's view.  Children are
/// re-sorted ascending by joint index, preserving the invariant ol_select_child's
/// binary search relies on.  Returns the number of children added.
fn ol_add_missing_children(
    arena: &mut Vec<OLNode>,
    node_id: u32,
    legal: &[(u16, Option<(i8, i8, i8, i8, bool)>, Option<u16>)],
    priors: &[f64],
) -> usize {
    let mut added = 0usize;
    for (i, &(idx, placement, pick)) in legal.iter().enumerate() {
        // children stays ascending, so binary_search both detects presence and
        // gives the in-order insertion point — no post-insert re-sort needed.
        match arena[node_id as usize]
            .children
            .binary_search_by_key(&idx, |&(c, _)| c)
        {
            Ok(_) => {} // already present
            Err(insert_at) => {
                let cid = arena.len() as u32;
                arena.push(OLNode::new(priors[i], (placement, pick)));
                arena[node_id as usize]
                    .children
                    .insert(insert_at, (idx, cid));
                added += 1;
            }
        }
    }
    debug_assert!(
        arena[node_id as usize]
            .children
            .windows(2)
            .all(|w| w[0].0 < w[1].0),
        "ol_add_missing_children: children not strictly ascending after insert"
    );
    added
}

fn ol_expand_uniform(arena: &mut Vec<OLNode>, node_id: u32, state: &RustGameState) -> PyResult<()> {
    let legal = state.legal_actions_indexed();
    if legal.is_empty() {
        return Ok(());
    }
    let priors = vec![1.0 / legal.len() as f64; legal.len()];
    if arena[node_id as usize].is_expanded {
        ol_add_missing_children(arena, node_id, &legal, &priors);
    } else {
        for (i, &(idx, placement, pick)) in legal.iter().enumerate() {
            let cid = arena.len() as u32;
            arena.push(OLNode::new(priors[i], (placement, pick)));
            arena[node_id as usize].children.push((idx, cid));
        }
        arena[node_id as usize].is_expanded = true;
    }
    Ok(())
}

/// Apply (sign=+1) / remove (sign=-1) virtual loss along an open-loop path.
/// Mirrors apply_virtual_loss but takes the per-node actor explicitly (nodes are
/// stateless): non-root node path[i] nudged by vl_value0 = -1 if its chooser
/// (actors[i-1]) is player 0 else +1.  Removal is the exact additive inverse.
fn ol_apply_virtual_loss(arena: &mut [OLNode], path: &[u32], actors: &[u8], sign: i32, n_vl: i32) {
    if n_vl <= 0 {
        return;
    }
    for i in 0..path.len() {
        arena[path[i] as usize].visit_count += sign * n_vl;
        if i > 0 {
            let chooser = actors[i - 1];
            let vl_value0 = if chooser == 0 { -1.0 } else { 1.0 };
            arena[path[i] as usize].value_sum += (sign * n_vl) as f64 * vl_value0;
            // Only chance nodes reconstruct Q independently of their own
            // value_sum. Ordinary decision nodes already consume the standard
            // visit/value mutations above, so extra bookkeeping there is dead
            // hot-path work (and would penalize the disabled path).
            if !arena[path[i] as usize].chance_children.is_empty() {
                if sign < 0 {
                    debug_assert!(
                        arena[path[i] as usize].virtual_visit_count >= n_vl,
                        "chance-node status and VL bookkeeping must survive apply/revert"
                    );
                }
                arena[path[i] as usize].virtual_visit_count += sign * n_vl;
                arena[path[i] as usize].virtual_value_sum += (sign * n_vl) as f64 * vl_value0;
            }
        }
    }
}

fn ol_backup_path(
    arena: &mut [OLNode],
    path: &[u32],
    sampled_value0: f64,
    chance_step: Option<(usize, f64)>,
) {
    let Some((chance_index, outcome_probability)) = chance_step else {
        for &node_id in path {
            let node = &mut arena[node_id as usize];
            node.visit_count += 1;
            node.value_sum += sampled_value0;
        }
        return;
    };

    let observation_id = path[chance_index + 1];
    let old_observation_visits = arena[observation_id as usize].visit_count;
    let old_observation_mean = if old_observation_visits > 0 {
        arena[observation_id as usize].value_sum / old_observation_visits as f64
    } else {
        arena[observation_id as usize]
            .bootstrap_value0
            .unwrap_or(0.0)
    };
    // First update the conditional observation subtree with this row's sample.
    for &node_id in &path[(chance_index + 1)..] {
        let node = &mut arena[node_id as usize];
        node.visit_count += 1;
        node.value_sum += sampled_value0;
    }
    let observation = &arena[observation_id as usize];
    let new_observation_mean = observation.value_sum / observation.visit_count as f64;
    let chance_node = &mut arena[path[chance_index] as usize];
    if old_observation_visits == 0 && chance_node.chance_backup != OLChanceBackup::PanelMean {
        chance_node.chance_visited_mass += outcome_probability;
    }
    chance_node.chance_weighted_value +=
        outcome_probability * (new_observation_mean - old_observation_mean);
    let backup = chance_node.chance_backup;
    // Both arms retain separate public observation subtrees. Sampled backup
    // propagates this visit's realized return; Hájek additionally replaces it
    // with the registered-probability estimate over rows evaluated so far.
    let propagated_value0 = match backup {
        OLChanceBackup::Sampled => sampled_value0,
        OLChanceBackup::Hajek => ol_chance_value_p0(arena, path[chance_index]),
        OLChanceBackup::PanelMean => ol_chance_value_p0(arena, path[chance_index]),
    };
    for &node_id in &path[..=chance_index] {
        let node = &mut arena[node_id as usize];
        node.visit_count += 1;
        node.value_sum += propagated_value0;
    }
}

/// Dirichlet noise on the open-loop root's child priors (OLNode analogue of
/// add_dirichlet_noise).  Noise-on search is not bit-comparable to Python, so
/// the equivalence gate runs with eps=0 (this is never called there).
fn ol_add_dirichlet_noise(
    arena: &mut [OLNode],
    root_id: u32,
    alpha: f64,
    eps: f64,
    seed: Option<u64>,
) {
    let child_ids: Vec<u32> = arena[root_id as usize]
        .children
        .iter()
        .map(|&(_, c)| c)
        .collect();
    let n = child_ids.len();
    if n == 0 {
        return;
    }
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let gamma = Gamma::new(alpha, 1.0).expect("alpha > 0");
    let samples: Vec<f64> = (0..n).map(|_| gamma.sample(&mut rng)).collect();
    let s: f64 = samples.iter().sum();
    for (i, &cid) in child_ids.iter().enumerate() {
        let noise = samples[i] / s;
        let c = &mut arena[cid as usize];
        c.prior = (1.0 - eps) * c.prior + eps * noise;
    }
}

/// Pick an open-loop root child by visit count (OLNode analogue of
/// select_from_visits).  τ=0 → argmax (ties → lowest joint index).
fn ol_select_from_visits(arena: &[OLNode], temp: f64, rng: &mut StdRng) -> u16 {
    let children = &arena[0].children;
    if temp <= 1e-6 {
        let mut best_v = -1i32;
        let mut best_idx = 0u16;
        for &(idx, cid) in children {
            let v = arena[cid as usize].visit_count;
            if v > best_v {
                best_v = v;
                best_idx = idx;
            }
        }
        best_idx
    } else {
        let weights: Vec<f64> = children
            .iter()
            .map(|&(_, cid)| (arena[cid as usize].visit_count as f64).powf(1.0 / temp))
            .collect();
        let sum: f64 = weights.iter().sum();
        if sum <= 0.0 {
            // Degenerate: every child unvisited / zero weight.  Should not occur
            // after n_sims > 0; fall back to the first child by prior order.
            debug_assert!(false, "ol_select_from_visits: sum of weights is zero");
            return children[0].0;
        }
        let mut r = rng.r#gen::<f64>() * sum;
        for (k, &(idx, _)) in children.iter().enumerate() {
            r -= weights[k];
            if r <= 0.0 {
                return idx;
            }
        }
        children.last().map(|t| t.0).unwrap_or(0)
    }
}

// ─── Advisor open-loop single-root search ───────────────────────────────────
// Rust port of the advisor's OpenLoopMCTS search loop (web_app._choose_nn_action):
// one root, per-simulation deck redeterminization, network leaf evaluation via
// the BatchedEvaluator callback contract, leaf-parallel waves with virtual loss.
// Exists because the Python OL engine drives batch-1 GPU forwards from Python
// (~1.7k leaves/s locally) while this path runs Rust tree ops + K-row eval
// batches — the advisor's sims/latency budget, not training throughput.
//
// Exact endgames are NOT hooked here: the advisor routes terminal-adjacent
// ROOTS (deck <= 4) to the exact per-child solver in web_app before searching,
// and interior deck<=4 nodes under a deck>=8 root are determinization-dependent
// (the same correctness gate as training — see OPT-1 notes).

/// Expand an OLNode with evaluator priors; returns the leaf value in the
/// player-0 frame. OL analogue of `expand` (closed-loop): stateless node, so
/// the caller passes the concrete `leaf_state`; an already-expanded node gets
/// only its missing children added (Issue 2 semantics).
fn ol_expand_with_evaluator(
    arena: &mut Vec<OLNode>,
    node_id: u32,
    leaf_state: &RustGameState,
    ev: &Py<PyAny>,
) -> PyResult<f64> {
    let actor = leaf_state.actor()?;
    let legal = leaf_state.legal_actions_indexed();
    let (my, opp, flat) = leaf_state.encode_arrays(actor)?;
    let idxs: Vec<i64> = legal.iter().map(|t| t.0 as i64).collect();

    let (value, gathered) = Python::attach(|py| -> PyResult<(f64, Vec<f64>)> {
        let mb_py = my.insert_axis(Axis(0)).into_pyarray(py);
        let ob_py = opp.insert_axis(Axis(0)).into_pyarray(py);
        let flat_py = flat.insert_axis(Axis(0)).into_pyarray(py);
        let idxs_py = idxs.into_pyarray(py);
        let idxs_list = PyList::new(py, [idxs_py])?;
        let result = ev.bind(py).call1((mb_py, ob_py, flat_py, idxs_list))?;
        let tuple = result.downcast::<PyTuple>()?;
        let value = {
            let arr = tuple.get_item(0)?;
            let arr = arr.downcast::<PyArray1<f32>>()?;
            arr.readonly().as_slice()?[0] as f64
        };
        let gathered = {
            let list = tuple.get_item(1)?;
            let list = list.downcast::<PyList>()?;
            let g0 = list.get_item(0)?;
            let arr = g0.downcast::<PyArray1<f32>>()?;
            arr.readonly()
                .as_slice()?
                .iter()
                .map(|&x| x as f64)
                .collect()
        };
        Ok((value, gathered))
    })?;

    let priors = softmax_f64(&gathered);
    let value0 = if actor == 0 { value } else { -value };
    if arena[node_id as usize].is_expanded {
        ol_add_missing_children(arena, node_id, &legal, &priors);
    } else {
        for (i, &(idx, placement, pick)) in legal.iter().enumerate() {
            let cid = arena.len() as u32;
            arena.push(OLNode::new(priors[i], (placement, pick)));
            arena[node_id as usize].children.push((idx, cid));
        }
        arena[node_id as usize].is_expanded = true;
    }
    Ok(value0)
}

/// One wave's pending network evaluation: the path that produced it and the
/// concrete leaf state that will be evaluated/expanded.
struct AdvisorPendingEval {
    path_idx: usize,
    leaf: u32,
    actor: u8,
    legal: Vec<(u16, Option<(i8, i8, i8, i8, bool)>, Option<u16>)>,
    my: Array3<f32>,
    opp: Array3<f32>,
    flat: Array1<f32>,
}

#[allow(clippy::too_many_arguments)]
fn advisor_open_loop_search_impl(
    root_state: &RustGameState,
    ev: &Py<PyAny>,
    n_sims: usize,
    dirichlet_alpha: f64,
    dirichlet_eps: f64,
    fpu: f64,
    cpuct: f64,
    seed: u64,
    leaf_batch: usize,
    virtual_loss: i32,
    score_scale: f64,
    margin_gain: f64,
    alpha_param: f64,
    pick_floor: Option<PickFloor>,
    chance_exposure: usize,
    chance_enum_max_rows: u64,
    chance_backup: OLChanceBackup,
    chance_traversal: OLChanceTraversal,
    a1c_options: Option<A1cSearchOptions>,
    nn_eval_budget: Option<usize>,
) -> PyResult<AdvisorOpenLoopOutput> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut arena: Vec<OLNode> = vec![OLNode::new(1.0, (None, None))];
    let mut fallback_count = 0u32;
    let mut missing_child_count = 0u32;
    let one_reveal_panel = if a1c_options.is_none() {
        ol_one_reveal_panel(
            &root_state.deck,
            chance_exposure,
            chance_enum_max_rows,
            seed ^ 0xA1C0_77EC_7A11_5EED,
        )?
    } else {
        Vec::new()
    };
    let a1c_cycles = if let Some(options) = a1c_options {
        a1c_one_reveal_cycles(
            &root_state.deck,
            chance_exposure,
            options.sampling,
            seed ^ 0xA1C0_77EC_7A11_5EED,
        )?
    } else {
        Vec::new()
    };
    // A one-row panel (deck=4) has no uncertainty or strategy-fusion risk.
    // Keep its support metadata, but avoid a pointless chance/observation layer.
    let chance_split_enabled = if a1c_options.is_some() {
        root_state.deck.len() > 4 && !a1c_cycles.is_empty()
    } else {
        one_reveal_panel.len() > 1
    };

    // Root: expand on the REAL state — the root's public information (boards,
    // row, claims) is determinization-independent; only the deck order below
    // it varies per simulation.
    let root_v0 = ol_expand_with_evaluator(&mut arena, 0, root_state, ev)?;
    let mut total_nn_evals = 1usize;
    let mut initialization_nn_evals = 0usize;
    let mut initialization_blocked_cycles = 0usize;
    let mut initialization_nn_budget_blocked_cycles = 0usize;
    let mut initialization_nn_budget_blocked_rows = 0usize;
    let mut nn_evaluator_calls = 1usize;
    let mut nn_max_batch_size = 1usize;
    let mut initialization_evaluator_calls = 0usize;
    let mut initialization_max_batch_size = 0usize;
    let mut a1c_admission_requested_paths = 0usize;
    let mut a1c_admission_unique_nodes = 0usize;
    let mut a1c_admission_committed_cycles = 0usize;
    let mut a1c_admission_waves = 0usize;
    arena[0].visit_count = 1;
    arena[0].value_sum = root_v0;
    if dirichlet_eps > 0.0 {
        ol_add_dirichlet_noise(&mut arena, 0, dirichlet_alpha, dirichlet_eps, Some(seed));
    }

    let wave_cap = leaf_batch.max(1);
    let vl = virtual_loss.max(0);
    let mut sims_done = 0usize;
    let mut search_waves = 0usize;
    while sims_done < n_sims && nn_eval_budget.is_none_or(|budget| total_nn_evals < budget) {
        let remaining_nn_budget = nn_eval_budget
            .map(|budget| budget.saturating_sub(total_nn_evals))
            .unwrap_or(usize::MAX);
        // A simulation produces at most one ordinary leaf evaluation. Limiting
        // the wave by the remaining budget prevents an evaluator batch from
        // overshooting the hard cap; terminal paths simply leave capacity for
        // a later wave.
        let wave = wave_cap.min(n_sims - sims_done).min(remaining_nn_budget);
        debug_assert!(wave > 0);
        let mut paths: Vec<Vec<u32>> = Vec::with_capacity(wave);
        let mut path_actors: Vec<Vec<u8>> = Vec::with_capacity(wave);
        let mut leaf_states: Vec<RustGameState> = Vec::with_capacity(wave);
        let mut chance_steps: Vec<Option<(usize, f64)>> = Vec::with_capacity(wave);
        let mut a1c_admission_requests: Vec<A1cAdmissionRequest> = Vec::new();
        let mut evals: Vec<AdvisorPendingEval> = Vec::new();

        for _ in 0..wave {
            let det_seed = rng.r#gen::<u64>();
            let det = if !chance_split_enabled {
                root_state.redeterminize(Some(det_seed))
            } else {
                // The first reveal is pinned only after PUCT chooses the
                // pre-reveal action. No operation before that split reads deck
                // order, so cloning the root here is information-set safe.
                root_state.cloned()
            };
            let chance_config = if !chance_split_enabled {
                None
            } else {
                Some(OLChanceConfig {
                    panel: one_reveal_panel.as_slice(),
                    backup: chance_backup,
                    traversal: chance_traversal,
                    schedule_seed: seed ^ 0xBA1A_4CED_C1C1_E5E5,
                    draw_seed: det_seed,
                    a1c: a1c_options.map(|options| A1cChanceConfig {
                        cycles: a1c_cycles.as_slice(),
                        n_init: options.n_init,
                        widening_c: options.widening_c,
                        max_initialization_fraction: options.max_initialization_fraction,
                    }),
                    bootstrap_full_panel: false,
                })
            };
            let (path, actors, leaf_state, chance_step, a1c_admission_request) = ol_descend(
                &mut arena,
                0,
                det,
                fpu,
                cpuct,
                &mut fallback_count,
                &mut missing_child_count,
                // Normally None: the advisor is pure PUCT.  The Phase A
                // allocation study passes a floor to test whether forced
                // pick-group coverage closes the secondary-pick fragility gap
                // at equal budget.  Off by default (frac=0 => None).
                pick_floor,
                chance_config,
            )?;
            if let Some(request) = a1c_admission_request {
                a1c_admission_requests.push(request);
            }
            ol_apply_virtual_loss(&mut arena, &path, &actors, 1, vl);
            let leaf = *path.last().unwrap();
            if leaf_state.phase != GAME_OVER {
                let actor = leaf_state.actor()?;
                let legal = leaf_state.legal_actions_indexed();
                let (my, opp, flat) = leaf_state.encode_arrays(actor)?;
                evals.push(AdvisorPendingEval {
                    path_idx: paths.len(),
                    leaf,
                    actor,
                    legal,
                    my,
                    opp,
                    flat,
                });
            }
            paths.push(path);
            path_actors.push(actors);
            leaf_states.push(leaf_state);
            chance_steps.push(chance_step);
        }

        // One evaluator call for the whole wave's non-terminal leaves.
        let mut path_v0: Vec<Option<f64>> = vec![None; paths.len()];
        if !evals.is_empty() {
            total_nn_evals += evals.len();
            nn_evaluator_calls += 1;
            nn_max_batch_size = nn_max_batch_size.max(evals.len());
            let idxs_per: Vec<Vec<i64>> = evals
                .iter()
                .map(|e| e.legal.iter().map(|t| t.0 as i64).collect())
                .collect();
            let (vals, gvecs) = Python::attach(|py| -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
                let mb_views: Vec<_> = evals.iter().map(|e| e.my.view()).collect();
                let ob_views: Vec<_> = evals.iter().map(|e| e.opp.view()).collect();
                let flat_views: Vec<_> = evals.iter().map(|e| e.flat.view()).collect();
                let mb = numpy::ndarray::stack(Axis(0), &mb_views)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let ob = numpy::ndarray::stack(Axis(0), &ob_views)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let flat = numpy::ndarray::stack(Axis(0), &flat_views)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let mb_py = mb.into_pyarray(py);
                let ob_py = ob.into_pyarray(py);
                let flat_py = flat.into_pyarray(py);
                let idx_arrays: Vec<_> = idxs_per
                    .iter()
                    .map(|v| v.clone().into_pyarray(py))
                    .collect();
                let idxs_list = PyList::new(py, idx_arrays)?;
                let result = ev.bind(py).call1((mb_py, ob_py, flat_py, idxs_list))?;
                let tuple = result.downcast::<PyTuple>()?;
                let vals: Vec<f64> = {
                    let arr = tuple.get_item(0)?;
                    let arr = arr.downcast::<PyArray1<f32>>()?;
                    arr.readonly()
                        .as_slice()?
                        .iter()
                        .map(|&x| x as f64)
                        .collect()
                };
                let gvecs: Vec<Vec<f64>> = {
                    let list = tuple.get_item(1)?;
                    let list = list.downcast::<PyList>()?;
                    let mut out = Vec::with_capacity(list.len());
                    for item in list.iter() {
                        let arr = item.downcast::<PyArray1<f32>>()?;
                        out.push(
                            arr.readonly()
                                .as_slice()?
                                .iter()
                                .map(|&x| x as f64)
                                .collect(),
                        );
                    }
                    out
                };
                Ok((vals, gvecs))
            })?;
            for (row, e) in evals.iter().enumerate() {
                let priors = softmax_f64(&gvecs[row]);
                let value0 = if e.actor == 0 { vals[row] } else { -vals[row] };
                if arena[e.leaf as usize].is_expanded {
                    ol_add_missing_children(&mut arena, e.leaf, &e.legal, &priors);
                } else {
                    for (i, &(idx, placement, pick)) in e.legal.iter().enumerate() {
                        let cid = arena.len() as u32;
                        arena.push(OLNode::new(priors[i], (placement, pick)));
                        arena[e.leaf as usize].children.push((idx, cid));
                    }
                    arena[e.leaf as usize].is_expanded = true;
                }
                path_v0[e.path_idx] = Some(value0);
            }
        }

        // Remove VL, then back up each path's own value (terminal leaves take
        // their concrete state's terminal value — deck-dependent by design).
        for (pi, path) in paths.iter().enumerate() {
            ol_apply_virtual_loss(&mut arena, path, &path_actors[pi], -1, vl);
        }
        for (pi, path) in paths.iter().enumerate() {
            let v0 = if leaf_states[pi].phase == GAME_OVER {
                terminal_search_value(&leaf_states[pi], score_scale, margin_gain, alpha_param)
            } else {
                path_v0[pi].expect("non-terminal path must have an eval value")
            };
            ol_backup_path(&mut arena, path, v0, chance_steps[pi]);
        }

        // Only now may a chance node change estimators. Every path in the wave
        // has removed virtual loss and backed up under the semantics it saw
        // during descent. Deduplicate nodes so at most one whole cycle is
        // admitted per node and wave.
        if let Some(options) = a1c_options {
            a1c_admission_requested_paths += a1c_admission_requests.len();
            a1c_admission_requests.sort_by_key(|request| request.chance_node_id);
            a1c_admission_requests.dedup_by_key(|request| request.chance_node_id);
            a1c_admission_unique_nodes += a1c_admission_requests.len();
            // Spend constrained initialization work on the most-visited chance
            // nodes first. Equal-visit requests use a seeded permutation rather
            // than node/action insertion order, avoiding a low-action-id bias.
            a1c_admission_requests.sort_by(|left, right| {
                let left_visits = arena[left.chance_node_id as usize].visit_count.max(0);
                let right_visits = arena[right.chance_node_id as usize].visit_count.max(0);
                right_visits.cmp(&left_visits).then_with(|| {
                    a1c_admission_tiebreak(seed, search_waves, left.chance_node_id).cmp(
                        &a1c_admission_tiebreak(seed, search_waves, right.chance_node_id),
                    )
                })
            });
            let mut committed_this_wave = 0usize;
            let a1c = A1cChanceConfig {
                cycles: a1c_cycles.as_slice(),
                n_init: options.n_init,
                widening_c: options.widening_c,
                max_initialization_fraction: options.max_initialization_fraction,
            };
            for request in a1c_admission_requests {
                let node = &arena[request.chance_node_id as usize];
                let current_cycles = node.chance_initialized_cycles;
                let real_visits = node.visit_count.max(0) as usize;
                let scheduled_target =
                    a1c_target_cycles(real_visits, a1c.n_init, a1c.cycles.len(), a1c.widening_c);
                // A node can cross multiple schedule thresholds in one wave,
                // but admission remains one atomic cycle per wave.
                let target = scheduled_target.min(current_cycles.saturating_add(1));
                if current_cycles >= target {
                    continue;
                }
                let mut runtime = A1cRuntime {
                    evaluator: ev,
                    initialization_nn_evals: &mut initialization_nn_evals,
                    total_nn_evals: &mut total_nn_evals,
                    initialization_blocked_cycles: &mut initialization_blocked_cycles,
                    nn_eval_budget,
                    nn_budget_blocked_cycles: &mut initialization_nn_budget_blocked_cycles,
                    nn_budget_blocked_rows: &mut initialization_nn_budget_blocked_rows,
                    evaluator_calls: &mut nn_evaluator_calls,
                    max_batch_size: &mut nn_max_batch_size,
                    initialization_evaluator_calls: &mut initialization_evaluator_calls,
                    initialization_max_batch_size: &mut initialization_max_batch_size,
                };
                if a1c_admit_next_cycle(
                    &mut arena,
                    request.chance_node_id,
                    &request.pre_reveal_state,
                    request.placement,
                    request.pick,
                    request.draw_seed,
                    a1c,
                    &mut runtime,
                )? {
                    committed_this_wave += 1;
                }
            }
            if committed_this_wave > 0 {
                a1c_admission_waves += 1;
                a1c_admission_committed_cycles += committed_this_wave;
            }
        }
        sims_done += paths.len();
        search_waves += 1;
    }

    let incumbent_root_value0 = arena[0].value_sum / arena[0].visit_count.max(1) as f64;
    let children: Vec<(u16, i32, f64, f64)> = arena[0]
        .children
        .iter()
        .map(|&(idx, cid)| {
            let c = &arena[cid as usize];
            let reported_value_sum = ol_reported_value_sum(&arena, cid);
            (idx, c.visit_count, reported_value_sum, c.prior)
        })
        .collect();
    let root_child_visits: i32 = children.iter().map(|row| row.1).sum();
    let current_children_root_value0 = if root_child_visits > 0 {
        children.iter().map(|row| row.2).sum::<f64>() / root_child_visits as f64
    } else {
        root_v0
    };
    let mut diagnostics = ol_chance_diagnostics(&arena);
    let exhaustive_panel =
        !one_reveal_panel.is_empty() && comb4(root_state.deck.len()) <= chance_enum_max_rows;
    let a1c_raw_panel_rows: usize = a1c_cycles.iter().map(Vec::len).sum();
    let a1c_unique_panel_rows: HashSet<[u16; 4]> = a1c_cycles.iter().flatten().copied().collect();
    let a1c_reached_nodes: Vec<(usize, &OLNode)> = if a1c_options.is_some() {
        arena
            .iter()
            .enumerate()
            .filter(|(_, node)| node.chance_chooser_actor.is_some() && node.visit_count > 0)
            .collect()
    } else {
        Vec::new()
    };
    let a1c_initialized_nodes: Vec<(usize, &OLNode)> = a1c_reached_nodes
        .iter()
        .copied()
        .filter(|(_, node)| node.chance_initialized_cycles > 0)
        .collect();
    let a1c_initialized_chance_nodes = a1c_initialized_nodes.len();
    let a1c_preinit_visits = a1c_initialized_nodes
        .iter()
        .map(|(_, node)| node.chance_preinit_visits)
        .sum::<usize>();
    let a1c_initialized_node_visits = a1c_initialized_nodes
        .iter()
        .map(|(_, node)| node.visit_count.max(0) as usize)
        .sum::<usize>();
    let a1c_uninitialized_chance_nodes = a1c_reached_nodes
        .iter()
        .filter(|(_, node)| node.chance_initialized_cycles == 0)
        .count();
    let a1c_uninitialized_node_visits = a1c_reached_nodes
        .iter()
        .filter(|(_, node)| node.chance_initialized_cycles == 0)
        .map(|(_, node)| node.visit_count.max(0) as usize)
        .sum::<usize>();
    diagnostics.insert(
        "chance_panel_rows".to_string(),
        if a1c_options.is_some() {
            a1c_unique_panel_rows.len() as f64
        } else {
            one_reveal_panel.len() as f64
        },
    );
    diagnostics.insert(
        "chance_panel_exhaustive".to_string(),
        if exhaustive_panel { 1.0 } else { 0.0 },
    );
    diagnostics.insert(
        "a1c_enabled".to_string(),
        if a1c_options.is_some() { 1.0 } else { 0.0 },
    );
    diagnostics.insert("a1c_max_cycles".to_string(), a1c_cycles.len() as f64);
    diagnostics.insert(
        "a1c_raw_planned_rows".to_string(),
        a1c_raw_panel_rows as f64,
    );
    diagnostics.insert(
        "a1c_unique_planned_rows".to_string(),
        a1c_unique_panel_rows.len() as f64,
    );
    diagnostics.insert("arena_nodes".to_string(), arena.len() as f64);
    diagnostics.insert("nn_evaluations".to_string(), total_nn_evals as f64);
    diagnostics.insert(
        "ordinary_nn_evaluations".to_string(),
        total_nn_evals.saturating_sub(initialization_nn_evals) as f64,
    );
    diagnostics.insert("nn_evaluator_calls".to_string(), nn_evaluator_calls as f64);
    diagnostics.insert(
        "nn_mean_batch_size".to_string(),
        total_nn_evals as f64 / nn_evaluator_calls.max(1) as f64,
    );
    diagnostics.insert("nn_max_batch_size".to_string(), nn_max_batch_size as f64);
    diagnostics.insert(
        "nn_eval_budget_enabled".to_string(),
        if nn_eval_budget.is_some() { 1.0 } else { 0.0 },
    );
    diagnostics.insert(
        "nn_eval_budget".to_string(),
        nn_eval_budget.unwrap_or(0) as f64,
    );
    diagnostics.insert(
        "nn_eval_budget_hit".to_string(),
        if nn_eval_budget.is_some_and(|budget| total_nn_evals >= budget) {
            1.0
        } else {
            0.0
        },
    );
    diagnostics.insert(
        "nn_eval_budget_unused".to_string(),
        nn_eval_budget
            .map(|budget| budget.saturating_sub(total_nn_evals))
            .unwrap_or(0) as f64,
    );
    diagnostics.insert("simulations_requested".to_string(), n_sims as f64);
    diagnostics.insert("simulations_completed".to_string(), sims_done as f64);
    diagnostics.insert("search_waves".to_string(), search_waves as f64);
    diagnostics.insert(
        "simulation_limit_hit".to_string(),
        if sims_done >= n_sims { 1.0 } else { 0.0 },
    );
    diagnostics.insert(
        "initialization_nn_evaluations".to_string(),
        initialization_nn_evals as f64,
    );
    diagnostics.insert(
        "initialization_nn_fraction".to_string(),
        initialization_nn_evals as f64 / total_nn_evals.max(1) as f64,
    );
    diagnostics.insert(
        "initialization_blocked_cycles".to_string(),
        initialization_blocked_cycles as f64,
    );
    diagnostics.insert(
        "initialization_nn_budget_blocked_cycles".to_string(),
        initialization_nn_budget_blocked_cycles as f64,
    );
    diagnostics.insert(
        "initialization_nn_budget_blocked_rows".to_string(),
        initialization_nn_budget_blocked_rows as f64,
    );
    diagnostics.insert(
        "initialization_evaluator_calls".to_string(),
        initialization_evaluator_calls as f64,
    );
    diagnostics.insert(
        "initialization_max_batch_size".to_string(),
        initialization_max_batch_size as f64,
    );
    diagnostics.insert(
        "a1c_admission_requested_paths".to_string(),
        a1c_admission_requested_paths as f64,
    );
    diagnostics.insert(
        "a1c_admission_unique_nodes".to_string(),
        a1c_admission_unique_nodes as f64,
    );
    diagnostics.insert(
        "a1c_admission_committed_cycles".to_string(),
        a1c_admission_committed_cycles as f64,
    );
    diagnostics.insert(
        "a1c_admission_waves".to_string(),
        a1c_admission_waves as f64,
    );
    diagnostics.insert(
        "a1c_wave_safe_admission".to_string(),
        if a1c_options.is_some() { 1.0 } else { 0.0 },
    );
    diagnostics.insert(
        "a1c_visit_prioritized_admission".to_string(),
        if a1c_options.is_some() { 1.0 } else { 0.0 },
    );
    diagnostics.insert(
        "a1c_initialized_chance_nodes".to_string(),
        a1c_initialized_chance_nodes as f64,
    );
    diagnostics.insert(
        "a1c_reached_chance_nodes".to_string(),
        a1c_reached_nodes.len() as f64,
    );
    diagnostics.insert(
        "a1c_uninitialized_chance_nodes".to_string(),
        a1c_uninitialized_chance_nodes as f64,
    );
    diagnostics.insert(
        "a1c_uninitialized_node_visits".to_string(),
        a1c_uninitialized_node_visits as f64,
    );
    diagnostics.insert(
        "a1c_initialized_cycles".to_string(),
        arena
            .iter()
            .map(|node| node.chance_initialized_cycles)
            .sum::<usize>() as f64,
    );
    diagnostics.insert("a1c_preinit_visits".to_string(), a1c_preinit_visits as f64);
    diagnostics.insert(
        "a1c_initialized_node_visits".to_string(),
        a1c_initialized_node_visits as f64,
    );
    diagnostics.insert(
        "a1c_preinit_visit_fraction".to_string(),
        if a1c_initialized_node_visits == 0 {
            0.0
        } else {
            a1c_preinit_visits as f64 / a1c_initialized_node_visits as f64
        },
    );
    // Preserve the per-node transition evidence in the flat diagnostic map.
    // The Python artifact layer reconstructs these keys into structured rows so
    // a null result can distinguish late or absent panel admission by node.
    for (node_id, node) in &a1c_reached_nodes {
        diagnostics.insert(
            format!("a1c_node_{node_id}_visits"),
            node.visit_count.max(0) as f64,
        );
        diagnostics.insert(
            format!("a1c_node_{node_id}_preinit_visits"),
            node.chance_preinit_visits as f64,
        );
        diagnostics.insert(
            format!("a1c_node_{node_id}_initialized_cycles"),
            node.chance_initialized_cycles as f64,
        );
    }
    diagnostics.insert(
        "root_value_running_mean_player0".to_string(),
        incumbent_root_value0,
    );
    diagnostics.insert(
        "root_value_current_children_player0".to_string(),
        current_children_root_value0,
    );
    Ok(AdvisorOpenLoopOutput {
        children,
        // Keep one stable legacy meaning for the 2-tuple advisor API. The
        // diagnostic API exposes both estimators under explicit names.
        root_value0: incumbent_root_value0,
        diagnostics,
    })
}

/// Hash one advisor information-set state. Hidden deck order is canonicalized;
/// all revealed/public fields remain exact.
fn advisor_public_signature(state: &RustGameState) -> u128 {
    let mut buf = Vec::with_capacity(2 * CELLS * 2 + 160);
    for board in &state.boards {
        buf.extend_from_slice(&board.terrain);
        buf.extend_from_slice(&board.crowns);
        buf.extend_from_slice(&[
            board.castle_x as u8,
            board.castle_y as u8,
            board.min_x as u8,
            board.max_x as u8,
            board.min_y as u8,
            board.max_y as u8,
            board.occupied,
        ]);
    }
    let mut bag = state.deck.clone();
    bag.sort_unstable();
    for value in bag {
        buf.extend_from_slice(&value.to_le_bytes());
    }
    buf.push(0xff);
    for &value in &state.current_row {
        buf.extend_from_slice(&value.to_le_bytes());
    }
    buf.push(0xfe);
    for &(player, value) in &state.pending_claims {
        buf.push(player);
        buf.extend_from_slice(&value.to_le_bytes());
    }
    buf.push(0xfd);
    for &(player, value) in &state.next_claims {
        buf.push(player);
        buf.extend_from_slice(&value.to_le_bytes());
    }
    buf.extend_from_slice(&[
        state.phase,
        state.actor_index as u8,
        state.initial_pick_count as u8,
        state.start_player,
        state.harmony as u8,
        state.middle_kingdom as u8,
    ]);
    buf.extend_from_slice(&state.discards[0].to_le_bytes());
    buf.extend_from_slice(&state.discards[1].to_le_bytes());
    xxhash_rust::xxh3::xxh3_128(&buf)
}

fn advisor_link_children(
    arena: &mut Vec<OLNode>,
    tt: &mut HashMap<u128, u32>,
    node_id: u32,
    leaf_state: &RustGameState,
    legal: &[(u16, Option<(i8, i8, i8, i8, bool)>, Option<u16>)],
    priors: &[f64],
) -> PyResult<()> {
    for (i, &(idx, placement, pick)) in legal.iter().enumerate() {
        if arena[node_id as usize]
            .children
            .iter()
            .any(|&(old, _)| old == idx)
        {
            continue;
        }
        let child_state = leaf_state.step(placement, pick)?;
        let signature = advisor_public_signature(&child_state);
        let cid = if let Some(&existing) = tt.get(&signature) {
            existing
        } else {
            let fresh = arena.len() as u32;
            arena.push(OLNode::new(priors[i], (placement, pick)));
            tt.insert(signature, fresh);
            fresh
        };
        arena[node_id as usize].children.push((idx, cid));
    }
    arena[node_id as usize]
        .children
        .sort_unstable_by_key(|&(idx, _)| idx);
    arena[node_id as usize].is_expanded = true;
    Ok(())
}

/// Persistent main-root advisor tree. Never reused across requests or by the
/// independent draft-matrix mini-searches.
#[pyclass]
struct AdvisorSearchHandle {
    root_state: RustGameState,
    ev: Py<PyAny>,
    rng: StdRng,
    arena: Vec<OLNode>,
    tt: HashMap<u128, u32>,
    fallback_count: u32,
    missing_child_count: u32,
    fpu: f64,
    cpuct: f64,
    leaf_batch: usize,
    virtual_loss: i32,
    score_scale: f64,
    margin_gain: f64,
    alpha_param: f64,
    sims_done: usize,
}

impl AdvisorSearchHandle {
    fn snapshot_inner(&self) -> (Vec<(u16, i32, f64, f64)>, f64) {
        let root = &self.arena[0];
        let value0 = root.value_sum / root.visit_count.max(1) as f64;
        let children = root
            .children
            .iter()
            .map(|&(idx, cid)| {
                let child = &self.arena[cid as usize];
                (idx, child.visit_count, child.value_sum, child.prior)
            })
            .collect();
        (children, value0)
    }

    fn advance_inner(&mut self, n_sims: usize) -> PyResult<(Vec<(u16, i32, f64, f64)>, f64)> {
        let wave_cap = self.leaf_batch.max(1);
        let vl = self.virtual_loss.max(0);
        let mut advanced = 0usize;
        while advanced < n_sims {
            let wave = wave_cap.min(n_sims - advanced);
            let mut paths: Vec<Vec<u32>> = Vec::with_capacity(wave);
            let mut path_actors: Vec<Vec<u8>> = Vec::with_capacity(wave);
            let mut leaf_states: Vec<RustGameState> = Vec::with_capacity(wave);
            let mut evals: Vec<AdvisorPendingEval> = Vec::new();
            for _ in 0..wave {
                let det = self.root_state.redeterminize(Some(self.rng.r#gen::<u64>()));
                let (path, actors, leaf_state, chance_step, a1c_admission_request) = ol_descend(
                    &mut self.arena,
                    0,
                    det,
                    self.fpu,
                    self.cpuct,
                    &mut self.fallback_count,
                    &mut self.missing_child_count,
                    None,
                    None,
                )?;
                debug_assert!(chance_step.is_none());
                debug_assert!(a1c_admission_request.is_none());
                ol_apply_virtual_loss(&mut self.arena, &path, &actors, 1, vl);
                let leaf = *path.last().unwrap();
                if leaf_state.phase != GAME_OVER {
                    let actor = leaf_state.actor()?;
                    let legal = leaf_state.legal_actions_indexed();
                    let (my, opp, flat) = leaf_state.encode_arrays(actor)?;
                    evals.push(AdvisorPendingEval {
                        path_idx: paths.len(),
                        leaf,
                        actor,
                        legal,
                        my,
                        opp,
                        flat,
                    });
                }
                paths.push(path);
                path_actors.push(actors);
                leaf_states.push(leaf_state);
            }
            let mut path_v0: Vec<Option<f64>> = vec![None; paths.len()];
            if !evals.is_empty() {
                let idxs_per: Vec<Vec<i64>> = evals
                    .iter()
                    .map(|e| e.legal.iter().map(|t| t.0 as i64).collect())
                    .collect();
                let (vals, gvecs) = Python::attach(|py| -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
                    let mb_views: Vec<_> = evals.iter().map(|e| e.my.view()).collect();
                    let ob_views: Vec<_> = evals.iter().map(|e| e.opp.view()).collect();
                    let flat_views: Vec<_> = evals.iter().map(|e| e.flat.view()).collect();
                    let mb = numpy::ndarray::stack(Axis(0), &mb_views)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let ob = numpy::ndarray::stack(Axis(0), &ob_views)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let flat = numpy::ndarray::stack(Axis(0), &flat_views)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let idx_arrays: Vec<_> = idxs_per
                        .iter()
                        .map(|v| v.clone().into_pyarray(py))
                        .collect();
                    let result = self.ev.bind(py).call1((
                        mb.into_pyarray(py),
                        ob.into_pyarray(py),
                        flat.into_pyarray(py),
                        PyList::new(py, idx_arrays)?,
                    ))?;
                    let tuple = result.downcast::<PyTuple>()?;
                    let vals = tuple
                        .get_item(0)?
                        .downcast::<PyArray1<f32>>()?
                        .readonly()
                        .as_slice()?
                        .iter()
                        .map(|&x| x as f64)
                        .collect();
                    let list_item = tuple.get_item(1)?;
                    let list = list_item.downcast::<PyList>()?;
                    let mut gvecs = Vec::with_capacity(list.len());
                    for item in list.iter() {
                        gvecs.push(
                            item.downcast::<PyArray1<f32>>()?
                                .readonly()
                                .as_slice()?
                                .iter()
                                .map(|&x| x as f64)
                                .collect(),
                        );
                    }
                    Ok((vals, gvecs))
                })?;
                for (row, eval) in evals.iter().enumerate() {
                    let priors = softmax_f64(&gvecs[row]);
                    let value0 = if eval.actor == 0 {
                        vals[row]
                    } else {
                        -vals[row]
                    };
                    advisor_link_children(
                        &mut self.arena,
                        &mut self.tt,
                        eval.leaf,
                        &leaf_states[eval.path_idx],
                        &eval.legal,
                        &priors,
                    )?;
                    path_v0[eval.path_idx] = Some(value0);
                }
            }
            for (pi, path) in paths.iter().enumerate() {
                ol_apply_virtual_loss(&mut self.arena, path, &path_actors[pi], -1, vl);
            }
            for (pi, path) in paths.iter().enumerate() {
                let value0 = if leaf_states[pi].phase == GAME_OVER {
                    terminal_search_value(
                        &leaf_states[pi],
                        self.score_scale,
                        self.margin_gain,
                        self.alpha_param,
                    )
                } else {
                    path_v0[pi].expect("non-terminal path must have an eval value")
                };
                for &node_id in path {
                    let node = &mut self.arena[node_id as usize];
                    node.visit_count += 1;
                    node.value_sum += value0;
                }
            }
            advanced += paths.len();
            self.sims_done += paths.len();
        }
        Ok(self.snapshot_inner())
    }
}

#[pymethods]
impl AdvisorSearchHandle {
    #[new]
    #[pyo3(signature = (state, evaluator, dirichlet_alpha=0.3, dirichlet_eps=0.0,
                        fpu=0.0, cpuct=1.5, seed=0, leaf_batch=8, virtual_loss=1,
                        score_scale=160.0, margin_gain=2.0, alpha=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        state: &RustGameState,
        evaluator: Bound<'_, PyAny>,
        dirichlet_alpha: f64,
        dirichlet_eps: f64,
        fpu: f64,
        cpuct: f64,
        seed: u64,
        leaf_batch: usize,
        virtual_loss: i32,
        score_scale: f64,
        margin_gain: f64,
        alpha: f64,
    ) -> PyResult<Self> {
        if state.phase == GAME_OVER {
            return Err(PyValueError::new_err("Cannot search from a terminal state"));
        }
        let root_state = state.cloned();
        let ev = evaluator.unbind();
        py.detach(move || {
            let mut arena = vec![OLNode::new(1.0, (None, None))];
            let root_v0 = ol_expand_with_evaluator(&mut arena, 0, &root_state, &ev)?;
            arena[0].visit_count = 1;
            arena[0].value_sum = root_v0;
            if dirichlet_eps > 0.0 {
                ol_add_dirichlet_noise(&mut arena, 0, dirichlet_alpha, dirichlet_eps, Some(seed));
            }
            let mut tt = HashMap::new();
            tt.insert(advisor_public_signature(&root_state), 0);
            for &(idx, placement, pick) in &root_state.legal_actions_indexed() {
                if let Some(&(_, cid)) = arena[0].children.iter().find(|&&(old, _)| old == idx) {
                    let child = root_state.step(placement, pick)?;
                    tt.insert(advisor_public_signature(&child), cid);
                }
            }
            Ok(AdvisorSearchHandle {
                root_state,
                ev,
                rng: StdRng::seed_from_u64(seed),
                arena,
                tt,
                fallback_count: 0,
                missing_child_count: 0,
                fpu,
                cpuct,
                leaf_batch,
                virtual_loss,
                score_scale,
                margin_gain,
                alpha_param: alpha,
                sims_done: 0,
            })
        })
    }

    fn advance(
        &mut self,
        py: Python<'_>,
        n_sims: usize,
    ) -> PyResult<(Vec<(u16, i32, f64, f64)>, f64)> {
        py.detach(|| self.advance_inner(n_sims))
    }
    fn snapshot(&self) -> (Vec<(u16, i32, f64, f64)>, f64) {
        self.snapshot_inner()
    }
    #[getter]
    fn sims_done(&self) -> usize {
        self.sims_done
    }
    #[getter]
    fn transpositions(&self) -> usize {
        self.tt.len()
    }
}

// ─── Batched MCTS (N games, synchronized ticks, one GPU forward per tick) ─────
// Drives N independent search trees ("slots") in lockstep: every tick each slot
// descends to its leaves (pure Rust), ALL N×leaf_batch leaves are stacked into
// one batch, Python runs ONE forward, results scatter back and back up.  Per
// slot the math is exactly `simulate_batch`, so N=1 is bit-identical to RustMCTS.
// Single-threaded driver — no GIL contention, no coalescing service.

/// Deterministic per-move redeterminize seed, a pure function of (game_seed,
/// move_num).  Exposed as `batched_det_seed` so the M6 reference can replay the
/// EXACT redeterminization BatchedMCTS used (splitmix64 mixing).
fn det_seed(game_seed: u64, move_num: usize) -> u64 {
    let mut z = game_seed.wrapping_add((move_num as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Build a fresh INITIAL_SELECTION game deterministically from a seed (Rust-side
/// setup — the batched engine recycles games without calling Python).  NOT a
/// reproduction of Python's RNG; it's this engine's own deterministic setup.
fn new_game(seed: u64, harmony: bool, middle_kingdom: bool) -> RustGameState {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut deck: Vec<u16> = (1..=48u16).collect();
    deck.shuffle(&mut rng);
    let mut row: Vec<u16> = deck[..4].to_vec();
    row.sort_unstable();
    deck.drain(..4);
    let start_player: u8 = rng.gen_range(0..2);
    RustGameState {
        boards: [RustBoard::new(7, 7), RustBoard::new(7, 7)],
        deck,
        current_row: row,
        pending_claims: Vec::new(),
        next_claims: Vec::new(),
        phase: INITIAL_SELECTION,
        actor_index: 0,
        initial_pick_count: 0,
        start_player,
        harmony,
        middle_kingdom,
        discards: [0, 0],
    }
}

/// AlphaZero PUCT search over a Rust-owned arena tree.
#[pyclass]
struct RustMCTS {}

#[pymethods]
impl RustMCTS {
    #[new]
    fn new() -> Self {
        RustMCTS {}
    }

    /// Run `n_sims` PUCT simulations from `state` and return root edge visit
    /// counts as (joint_index, visit_count) pairs in ascending-index order.
    ///
    /// `evaluator` is a Python callable with the BatchedEvaluator contract:
    ///   (mb (K,9,13,13) f32, ob (K,9,13,13) f32, flat (K,FLAT_SIZE) f32, idxs_list)
    ///     -> (values (K,) f64, [gathered_logits_i (n_i,) f64])
    /// Serial search calls it with K=1.  `seed` only affects Dirichlet noise.
    #[pyo3(signature = (state, evaluator, n_sims, dirichlet_alpha=0.3, dirichlet_eps=0.0, fpu=0.0, cpuct=1.5, seed=None, leaf_batch=1, virtual_loss=1, score_scale=160.0, margin_gain=2.0, alpha=0.5))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        state: &RustGameState,
        evaluator: Bound<'py, PyAny>,
        n_sims: usize,
        dirichlet_alpha: f64,
        dirichlet_eps: f64,
        fpu: f64,
        cpuct: f64,
        seed: Option<u64>,
        leaf_batch: usize,
        virtual_loss: i32,
        score_scale: f64,
        margin_gain: f64,
        alpha: f64,
    ) -> PyResult<Vec<(u16, i32)>> {
        if state.phase == GAME_OVER {
            return Err(PyValueError::new_err("Cannot search from a terminal state"));
        }

        // Set up under the GIL: own the root state (pure-Rust clone) and make the
        // evaluator GIL-independent so both can move into the GIL-released closure.
        let root_state = state.cloned();
        let ev: Py<PyAny> = evaluator.unbind();

        // Release the GIL for ALL tree work; expand/evaluate_batch re-acquire it
        // only at the leaf-evaluation callback (Python::with_gil).  This lets many
        // game threads (each its own RustMCTS) overlap tree work, and — while one
        // is blocked in the evaluator — lets the others submit leaves so an
        // in-process coalescing evaluator can batch across games.
        py.detach(move || -> PyResult<Vec<(u16, i32)>> {
            let mut arena: Vec<Node> = Vec::new();
            arena.push(Node::new(1.0, (None, None)));
            let root_id = 0u32;
            arena[0].state = Some(root_state);

            // Expand the root once, seed its stats.
            let root_v0 = expand(&mut arena, root_id, &ev)?;
            arena[0].visit_count = 1;
            arena[0].value_sum = root_v0;

            if dirichlet_eps > 0.0 {
                add_dirichlet_noise(&mut arena, root_id, dirichlet_alpha, dirichlet_eps, seed);
            }

            if leaf_batch <= 1 {
                // Serial path — bit-identical to the pre-leaf-parallel search.
                for _ in 0..n_sims {
                    simulate(
                        &mut arena,
                        root_id,
                        &ev,
                        fpu,
                        cpuct,
                        score_scale,
                        margin_gain,
                        alpha,
                    )?;
                }
            } else {
                // Leaf-parallel path (virtual loss); total leaf budget stays n_sims.
                let mut remaining = n_sims;
                while remaining > 0 {
                    let b = remaining.min(leaf_batch);
                    simulate_batch(
                        &mut arena,
                        root_id,
                        &ev,
                        fpu,
                        cpuct,
                        b,
                        virtual_loss,
                        score_scale,
                        margin_gain,
                        alpha,
                    )?;
                    remaining -= b;
                }
            }

            let root = &arena[0];
            Ok(root
                .children
                .iter()
                .map(|&(idx, cid)| (idx, arena[cid as usize].visit_count))
                .collect())
        })
    }
}

// One finished move's training data (pure-Rust ndarrays; converted to numpy only
// when the game finishes and examples are returned to Python).
#[derive(Clone)]
struct MoveRecord {
    my: Array3<f32>,
    opp: Array3<f32>,
    flat: Array1<f32>,
    policy_idx: Vec<i32>,
    policy_val: Vec<f32>,
    legal_idx: Vec<i32>,
    root_prior_idx: Vec<i32>,
    root_prior_val: Vec<f32>,
    root_visit_count: Vec<i32>,
    actor: u8,
    own_score: f32,  // raw own final score (filled at game end in finalize_move)
    opp_score: f32,  // raw opponent final score (filled at game end)
    win_target: f32, // 1.0 win / 0.5 draw / 0.0 loss, actor frame (filled at end)
}

/// Canonical public-state bytes shared with Python chance-correct search.
/// Keep this layout byte-for-byte aligned with
/// `denial_search.chance_public_state_key_v1`; changing it requires a version
/// bump rather than an in-place edit.
fn chance_public_state_key_v1_bytes(state: &RustGameState, buf: &mut Vec<u8>) {
    buf.clear();
    buf.extend_from_slice(b"KD-PUBLIC\x01");
    buf.push(state.phase);
    buf.push(state.actor_index as u8);
    buf.push(state.initial_pick_count as u8);
    buf.push(state.start_player);
    buf.push(state.harmony as u8);
    buf.push(state.middle_kingdom as u8);
    buf.extend_from_slice(&state.discards[0].to_le_bytes());
    buf.extend_from_slice(&state.discards[1].to_le_bytes());

    let mut deck = state.deck.clone();
    deck.sort_unstable();
    buf.push(deck.len() as u8);
    for value in deck {
        buf.extend_from_slice(&value.to_le_bytes());
    }
    let mut row = state.current_row.clone();
    row.sort_unstable();
    buf.push(row.len() as u8);
    for value in row {
        buf.extend_from_slice(&value.to_le_bytes());
    }
    buf.push(state.pending_claims.len() as u8);
    for &(player, domino_id) in &state.pending_claims {
        buf.push(player);
        buf.extend_from_slice(&domino_id.to_le_bytes());
    }
    buf.push(state.next_claims.len() as u8);
    for &(player, domino_id) in &state.next_claims {
        buf.push(player);
        buf.extend_from_slice(&domino_id.to_le_bytes());
    }
    for board in &state.boards {
        buf.extend_from_slice(&board.terrain);
        buf.extend_from_slice(&board.crowns);
    }
}

/// Completed game payload kept inside Rust until conversion to Python.
/// The final i8 is the official player-0 outcome (+1/0/-1).
type FinishedGame = (u64, Vec<MoveRecord>, (i32, i32), i8);

#[derive(Clone, Copy)]
struct MoveSearchProfile {
    target_sims: usize,
    record_example: bool,
    is_full_search: bool,
    dirichlet_eps: f64,
    temp_moves: usize,
}

impl MoveSearchProfile {
    fn full(n_sims: usize, dirichlet_eps: f64, temp_moves: usize) -> Self {
        Self {
            target_sims: n_sims.max(1),
            record_example: true,
            is_full_search: true,
            dirichlet_eps,
            temp_moves,
        }
    }
}

/// Two-net asymmetric self-play (HOF diversity games): the override seat — the
/// frozen HOF opponent — always searches a fixed shallow budget with its own
/// noise/temperature settings and NEVER records training examples.  The other
/// seat (the learner) gets the normal move profile, including playout-cap
/// randomization, so its recorded targets are full-strength searches.
#[derive(Clone, Copy)]
struct SeatSearchOverride {
    seat: u8,
    sims: usize,
    dirichlet_eps: f64,
    temp_moves: usize,
}

#[derive(PartialEq, Clone, Copy)]
enum SlotState {
    NeedsRootEval, // root set but unexpanded; contributes the root as 1 leaf
    ExactSolving,  // root is a terminal-adjacent endgame; awaiting exact solve
    // Async (Step 1.5): the endgame was dispatched to the background solver. The
    // slot KEEPS its game (real_state/records/move_num) so a timed-out solve
    // resumes MCTS in place; a solved one rejoins as a finished game on harvest.
    SolvingInBackground,
    Searching, // contributes up to leaf_batch descended leaves per tick
    Idle,      // no game (quota met); contributes nothing
}

/// Cached result of an exact endgame solve at the current move's root, present
/// only while a slot is resolving a terminal-adjacent (deck ∈ {0,4}) endgame.
/// `finalize_move` uses `child_values` to build the policy target and pick the
/// minimax-optimal move. own/opp/win and the value target `z` are NOT taken from
/// here — the game plays out to GAME_OVER under exact-optimal moves, so they are
/// filled from the real terminal scores at game end, exactly as for MCTS moves.
/// How the exact root solve prices non-best children for the POLICY target.
/// The root VALUE and the chosen (minimax-best) MOVE are exact in every mode —
/// modes only trade per-child value precision for solve cost:
///
/// - `Exact`: every child solved full-window (the historical behavior). The
///   policy label is the advantage-weighted softmax over exact child values.
/// - `SoftClamp`: children within `clamp_delta` raw points of the best are
///   solved exactly; the rest are only PROVEN at least `clamp_delta` worse
///   (cheap fail-low search) and recorded at the clamp value `best ∓ delta`.
///   Since the clamp can only raise a bad child's value, the label error is
///   one-sided (dominated moves slightly overweighted) and bounded by the
///   softmax weight at delta (~e^-3 relative when delta spans the range).
/// - `ArgmaxTies`: quarter-point-separated solver utilities let a 0.25 window
///   prove exact ties with the best; the label is uniform over tied-best children
///   elsewhere. Cheapest mode — and the PRODUCTION DEFAULT since the
///   2026-07-05 label-shape ablation, where it beat soft_clamp 231-162-7 in a
///   400-game head-to-head between matched training runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExactPolicyMode {
    Exact,
    SoftClamp,
    ArgmaxTies,
}

impl ExactPolicyMode {
    fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "exact" => Ok(Self::Exact),
            "soft_clamp" => Ok(Self::SoftClamp),
            "argmax_ties" => Ok(Self::ArgmaxTies),
            _ => Err(PyValueError::new_err(format!(
                "unknown exact_policy_mode '{s}' (expected exact, soft_clamp, argmax_ties)"
            ))),
        }
    }
}

#[derive(Clone)]
struct ExactSolveResult {
    /// (joint_index, value player-0 frame, training-value space) for ALL legal
    /// root actions. Exact for the best child always; under SoftClamp/ArgmaxTies
    /// dominated children hold their clamp value (an upper bound on how good
    /// they are, from the mover's perspective).
    child_values: Vec<(u16, f64)>,
    /// Label mode this result was solved under (drives `policy_target`).
    policy_mode: ExactPolicyMode,
    /// Children recorded at the clamp value instead of an exact value.
    n_clamped: u16,
}

impl ExactSolveResult {
    /// Build the policy training target for this solved root.
    ///
    /// Exact / SoftClamp use the advantage-weighted softmax
    /// (`exact_policy_target`); ArgmaxTies is uniform over the proven-tied-best
    /// children. SoftClamp label semantics: with the range compressed to the
    /// clamp distance, the softmax temperature anchors to Δ instead of the
    /// true (unsolved) range — near-best discrimination is Δ-scaled ("3 points
    /// worse" always means the same weight ratio), whereas the Exact label's
    /// sharpness varies with how bad the worst legal move happens to be. All
    /// clamped children collapse to the recorded worst and share one weight.
    fn policy_target(&self, actor: u8) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
        match self.policy_mode {
            ExactPolicyMode::ArgmaxTies => exact_policy_target_ties(&self.child_values, actor),
            _ => exact_policy_target(&self.child_values, actor),
        }
    }
}

struct ExactPlanItem {
    result: ExactSolveResult,
}

#[derive(Hash, Eq, PartialEq, Clone)]
struct EndgameKey {
    phase: u8,
    actor_index: usize,
    deck: Vec<u16>,
    current_row: Vec<u16>,
    pending_claims: Vec<(u8, u16)>,
    next_claims: Vec<(u8, u16)>,
    board0_terrain: [u8; CELLS],
    board0_crowns: [u8; CELLS],
    board1_terrain: [u8; CELLS],
    board1_crowns: [u8; CELLS],
}

fn deck4_remaining_current_claims(state: &RustGameState) -> usize {
    state.pending_claims.len().saturating_sub(state.actor_index)
}

fn should_enter_exact_solving(exact_enabled: bool, slot: &SearchSlot) -> bool {
    if !exact_enabled || !is_no_chance_endgame_state(&slot.real_state) {
        return false;
    }
    if !slot.exact_unsolvable {
        return true;
    }
    if slot.real_state.deck.is_empty() {
        return true;
    }
    slot.real_state.deck.len() == 4
        && !slot.exact_deck4_after_two_retry_used
        && deck4_remaining_current_claims(&slot.real_state) <= 2
}

fn mark_exact_timeout(slot: &mut SearchSlot) {
    if slot.exact_unsolvable
        && slot.real_state.deck.len() == 4
        && deck4_remaining_current_claims(&slot.real_state) <= 2
    {
        slot.exact_deck4_after_two_retry_used = true;
    }
    slot.exact_unsolvable = true;
}

#[derive(Copy, Clone)]
enum ExactAttemptKind {
    Deck4Initial,
    Deck4Retry,
    Deck0,
}

fn exact_attempt_kind_label(kind: ExactAttemptKind) -> &'static str {
    match kind {
        ExactAttemptKind::Deck4Initial => "deck4_initial",
        ExactAttemptKind::Deck4Retry => "deck4_retry",
        ExactAttemptKind::Deck0 => "deck0",
    }
}

fn exact_attempt_kind(slot: &SearchSlot) -> ExactAttemptKind {
    if slot.real_state.deck.is_empty() {
        ExactAttemptKind::Deck0
    } else if slot.exact_unsolvable {
        ExactAttemptKind::Deck4Retry
    } else {
        ExactAttemptKind::Deck4Initial
    }
}

/// One game's search tree + real state.  Per-tick the slot runs exactly one
/// `simulate_batch` chunk; root handling mirrors `RustMCTS::search`.
struct SearchSlot {
    state: SlotState,
    arena: Vec<Node>,      // closed-loop tree (empty when open_loop)
    ol_arena: Vec<OLNode>, // open-loop tree (empty when !open_loop)
    real_state: RustGameState,
    sims_done: usize,
    move_num: usize,
    move_profile: MoveSearchProfile,
    game_seed: u64,
    rng: StdRng, // Dirichlet noise + move selection + per-sim determinization
    records: Vec<MoveRecord>,
    fallback_count: u32, // open-loop: deep-node legal-filter fallbacks (diagnostic)
    missing_child_count: u32, // open-loop: descents stopped to add newly-legal children (diagnostic)
    deck8_chance_panel_count: u32,
    deck8_chance_bootstrap_rows: u32,
    deck8_chance_budget_blocked_count: u32,
    // Per-tree admission guard. Reset after the real move is played, so at
    // most one sibling action receives an exhaustive panel in each search.
    deck8_chance_panel_admitted_this_move: bool,
    exact_result: Option<ExactSolveResult>, // Some only while state == ExactSolving
    exact_plan: Vec<ExactPlanItem>,         // chosen-line plan for the deterministic endgame
    // Set once the exact solver times out on this game's deck=4 endgame. This
    // suppresses retrying the same expensive full-row root, but deck=0 remains
    // eligible and one deck=4 retry is allowed after the current row has shrunk
    // to two remaining decisions. Reset when the slot starts a new game.
    exact_unsolvable: bool,
    exact_deck4_after_two_retry_used: bool,
    fast_move_count: u32,
    full_move_count: u32,
    recorded_fast_move_count: u32,
    recorded_full_move_count: u32,
    exact_recorded_move_count: u32,
    // Pick-floor diagnostic: at each full-search finalize, the minimum
    // pick-group visit share at the most-visited root CHILD (a depth-1 node,
    // where the floor acts). sum/count → mean min-share per iteration; with
    // floors off this is the starvation baseline, with floors on it should
    // sit at ≈ pick_floor_frac.
    pf_minshare_sum: f64,
    pf_minshare_count: u32,
}

/// Random-opening diversification (run9): with probability `fraction`, a new
/// game starts with k ~ Uniform[min_plies, max_plies] uniformly-random plies
/// played out BEFORE the first search. The plies are never recorded (they
/// happen before any MoveRecord exists), so target quality is untouched;
/// their purpose is stretching the position distribution the searched game
/// then explores. Early Kingdomino plies are mostly picks plus low-stakes
/// placements on a near-empty board, so k <= 8 perturbs without ruining.
#[derive(Clone, Copy)]
struct RandomOpening {
    fraction: f64,
    min_plies: usize,
    max_plies: usize,
}

impl SearchSlot {
    /// Start a fresh game in this slot: real state + a redeterminized root,
    /// ready for its root evaluation on the next tick.
    fn new_for_game(
        mut real_state: RustGameState,
        game_seed: u64,
        open_loop: bool,
        opening: Option<RandomOpening>,
    ) -> SearchSlot {
        let mut rng = StdRng::seed_from_u64(game_seed);
        let mut move_num = 0usize;
        if let Some(o) = opening {
            if o.fraction > 0.0 && rng.r#gen::<f64>() < o.fraction {
                let k = if o.max_plies > o.min_plies {
                    rng.gen_range(o.min_plies..=o.max_plies)
                } else {
                    o.min_plies
                };
                for _ in 0..k {
                    if real_state.phase == GAME_OVER {
                        break;
                    }
                    let acts = real_state.legal_actions_indexed();
                    if acts.is_empty() {
                        break;
                    }
                    let (_, placement, pick) = acts[rng.gen_range(0..acts.len())];
                    match real_state.step(placement, pick) {
                        Ok(next) => real_state = next,
                        Err(_) => break, // defensive: keep the pre-step state
                    }
                    move_num += 1;
                }
            }
        }
        // Closed-loop stores a redeterminized root state in arena[0]; open-loop
        // is stateless and evaluates its root from the public real_state, so it
        // only needs a bare ol_arena root.
        let (arena, ol_arena) = if open_loop {
            (Vec::new(), vec![OLNode::new(1.0, (None, None))])
        } else {
            let root_state = real_state.redeterminize(Some(det_seed(game_seed, move_num)));
            let mut arena = vec![Node::new(1.0, (None, None))];
            arena[0].state = Some(root_state);
            (arena, Vec::new())
        };
        // A fresh game starts at INITIAL_SELECTION with a full deck, so this is
        // virtually never an endgame — but check anyway so the trigger is uniform.
        let state = if is_no_chance_endgame_state(&real_state) {
            SlotState::ExactSolving
        } else {
            SlotState::NeedsRootEval
        };
        SearchSlot {
            state,
            arena,
            ol_arena,
            real_state,
            sims_done: 0,
            move_num,
            move_profile: MoveSearchProfile::full(1, 0.0, 0),
            game_seed,
            rng,
            records: Vec::new(),
            fallback_count: 0,
            missing_child_count: 0,
            deck8_chance_panel_count: 0,
            deck8_chance_bootstrap_rows: 0,
            deck8_chance_budget_blocked_count: 0,
            deck8_chance_panel_admitted_this_move: false,
            exact_result: None,
            exact_plan: Vec::new(),
            exact_unsolvable: false,
            exact_deck4_after_two_retry_used: false,
            fast_move_count: 0,
            full_move_count: 0,
            recorded_fast_move_count: 0,
            recorded_full_move_count: 0,
            exact_recorded_move_count: 0,
            pf_minshare_sum: 0.0,
            pf_minshare_count: 0,
        }
    }

    /// A slot with no game (used when n_games < n_slots, or after the quota is
    /// met).  Holds a throwaway state that is never searched.
    fn idle(harmony: bool, middle_kingdom: bool) -> SearchSlot {
        SearchSlot {
            state: SlotState::Idle,
            arena: Vec::new(),
            ol_arena: Vec::new(),
            real_state: new_game(0, harmony, middle_kingdom),
            sims_done: 0,
            move_num: 0,
            move_profile: MoveSearchProfile::full(1, 0.0, 0),
            game_seed: 0,
            rng: StdRng::seed_from_u64(0),
            records: Vec::new(),
            fallback_count: 0,
            missing_child_count: 0,
            deck8_chance_panel_count: 0,
            deck8_chance_bootstrap_rows: 0,
            deck8_chance_budget_blocked_count: 0,
            deck8_chance_panel_admitted_this_move: false,
            exact_result: None,
            exact_plan: Vec::new(),
            exact_unsolvable: false,
            exact_deck4_after_two_retry_used: false,
            fast_move_count: 0,
            full_move_count: 0,
            recorded_fast_move_count: 0,
            recorded_full_move_count: 0,
            exact_recorded_move_count: 0,
            pf_minshare_sum: 0.0,
            pf_minshare_count: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn choose_move_profile(
        &mut self,
        playout_cap_randomization: bool,
        full_search_fraction: f64,
        n_sims: usize,
        fast_move_sims: usize,
        record_fast_moves: bool,
        dirichlet_eps: f64,
        fast_move_dirichlet_eps: f64,
        temp_moves: usize,
        fast_move_temp_moves: usize,
        seat_override: Option<SeatSearchOverride>,
    ) {
        // Asymmetric two-net games: the override seat (frozen HOF opponent) is
        // pinned to its own shallow profile and records nothing; every other
        // parameter below applies to the learner seat only.
        if let Some(o) = seat_override {
            if self
                .real_state
                .actor()
                .map(|a| a == o.seat)
                .unwrap_or(false)
            {
                self.move_profile = MoveSearchProfile {
                    target_sims: o.sims.max(1),
                    record_example: false,
                    is_full_search: false,
                    dirichlet_eps: o.dirichlet_eps,
                    temp_moves: o.temp_moves,
                };
                return;
            }
        }
        if !playout_cap_randomization {
            self.move_profile = MoveSearchProfile::full(n_sims, dirichlet_eps, temp_moves);
            return;
        }
        let p = full_search_fraction.clamp(0.0, 1.0);
        let is_full = self.rng.r#gen::<f64>() < p;
        self.move_profile = if is_full {
            MoveSearchProfile::full(n_sims, dirichlet_eps, temp_moves)
        } else {
            MoveSearchProfile {
                target_sims: fast_move_sims.max(1),
                record_example: record_fast_moves,
                is_full_search: false,
                dirichlet_eps: fast_move_dirichlet_eps,
                temp_moves: fast_move_temp_moves,
            }
        };
    }

    /// Finalize the current move (sims complete): record the training example,
    /// select + apply a move to the REAL state, then either start the next move
    /// (→ NeedsRootEval) or, if the game is over, return its finished records.
    fn finalize_move(
        &mut self,
        open_loop: bool,
        exact_enabled: bool,
        playout_cap_randomization: bool,
        full_search_fraction: f64,
        n_sims: usize,
        fast_move_sims: usize,
        record_fast_moves: bool,
        dirichlet_eps: f64,
        fast_move_dirichlet_eps: f64,
        temp_moves: usize,
        fast_move_temp_moves: usize,
        seat_override: Option<SeatSearchOverride>,
    ) -> PyResult<Option<FinishedGame>> {
        // Training record: encode the REAL (public) state + policy target.
        let actor = self.real_state.actor()?;

        // Take any exact-solve result for this move (clears it so the next move,
        // if MCTS-driven, never sees a stale value).
        let exact = self.exact_result.take();
        let is_exact = exact.is_some();

        let (
            policy_idx,
            policy_val,
            legal_idx,
            root_prior_idx,
            root_prior_val,
            root_visit_count,
            chosen,
        ) = if let Some(exact) = exact {
            // ── Exact endgame path: policy + move from minimax child values ──
            let (policy_idx, policy_val, legal_idx) = exact.policy_target(actor);
            // The optimal move is unambiguous; always play the minimax-best child
            // (temperature does not apply — there is a single correct answer).
            let best = if actor == 0 {
                exact
                    .child_values
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            } else {
                exact
                    .child_values
                    .iter()
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            };
            let chosen = best.map(|&(idx, _)| idx).unwrap_or(legal_idx[0] as u16);
            (
                policy_idx,
                policy_val,
                legal_idx,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                chosen,
            )
        } else {
            // ── Normal MCTS path: visit-count policy + visit-count selection ──
            let root_children: Vec<(u16, i32, f64)> = if open_loop {
                self.ol_arena[0]
                    .children
                    .iter()
                    .map(|&(idx, c)| {
                        let child = &self.ol_arena[c as usize];
                        (idx, child.visit_count, child.prior)
                    })
                    .collect()
            } else {
                self.arena[0]
                    .children
                    .iter()
                    .map(|&(idx, c)| {
                        let child = &self.arena[c as usize];
                        (idx, child.visit_count, child.prior)
                    })
                    .collect()
            };
            // Pick-floor diagnostic: min pick-group visit share at the
            // most-visited root child (a depth-1 node — where the floor acts).
            if open_loop {
                let best_child = self.ol_arena[0]
                    .children
                    .iter()
                    .max_by_key(|&&(_, c)| self.ol_arena[c as usize].visit_count)
                    .map(|&(_, c)| c);
                if let Some(cid) = best_child {
                    let node = &self.ol_arena[cid as usize];
                    if node.visit_count >= 64 {
                        let mut gv = [0i64; 5];
                        let mut gp = [false; 5];
                        for &(idx, c) in &node.children {
                            let g = (idx % 5) as usize;
                            gp[g] = true;
                            gv[g] += self.ol_arena[c as usize].visit_count as i64;
                        }
                        let gtotal: i64 = gv.iter().sum();
                        let n_groups = gp.iter().filter(|&&p| p).count();
                        if gtotal > 0 && n_groups >= 2 {
                            let min_share = (0..5)
                                .filter(|&g| gp[g])
                                .map(|g| gv[g] as f64 / gtotal as f64)
                                .fold(f64::INFINITY, f64::min);
                            self.pf_minshare_sum += min_share;
                            self.pf_minshare_count += 1;
                        }
                    }
                }
            }
            let total: i32 = root_children.iter().map(|&(_, vc, _)| vc).sum();
            let mut policy_idx = Vec::new();
            let mut policy_val = Vec::new();
            let mut legal_idx = Vec::new();
            let mut root_prior_idx = Vec::new();
            let mut root_prior_val = Vec::new();
            let mut root_visit_count = Vec::new();
            for &(idx, vc, prior) in &root_children {
                legal_idx.push(idx as i32);
                root_prior_idx.push(idx as i32);
                root_prior_val.push(prior as f32);
                root_visit_count.push(vc);
                if vc > 0 {
                    policy_idx.push(idx as i32);
                    policy_val.push(vc as f32 / total as f32);
                }
            }
            let temp = if self.move_num < self.move_profile.temp_moves {
                1.0
            } else {
                0.0
            };
            let chosen = if open_loop {
                ol_select_from_visits(&self.ol_arena, temp, &mut self.rng)
            } else {
                select_from_visits(&self.arena, temp, &mut self.rng)
            };
            (
                policy_idx,
                policy_val,
                legal_idx,
                root_prior_idx,
                root_prior_val,
                root_visit_count,
                chosen,
            )
        };

        let should_record = is_exact || self.move_profile.record_example;
        if is_exact {
            self.exact_recorded_move_count += 1;
        } else if self.move_profile.is_full_search {
            self.full_move_count += 1;
            if should_record {
                self.recorded_full_move_count += 1;
            }
        } else {
            self.fast_move_count += 1;
            if should_record {
                self.recorded_fast_move_count += 1;
            }
        }

        if should_record {
            let (my, opp, flat) = self.real_state.encode_arrays(actor)?;
            self.records.push(MoveRecord {
                my,
                opp,
                flat,
                policy_idx,
                policy_val,
                legal_idx,
                root_prior_idx,
                root_prior_val,
                root_visit_count,
                actor,
                own_score: 0.0,
                opp_score: 0.0,
                win_target: 0.5,
            });
        }
        let (placement, pick) = self
            .real_state
            .legal_actions_indexed()
            .into_iter()
            .find(|t| t.0 == chosen)
            .map(|t| (t.1, t.2))
            .ok_or_else(|| PyValueError::new_err("selected index not legal in real state"))?;
        self.real_state = self.real_state.step(placement, pick)?;
        self.move_num += 1;

        if self.real_state.phase == GAME_OVER {
            let (s0, s1) = self.real_state.scores();
            let outcome0 = self.real_state.official_outcome_i8();
            // Fill the per-move targets now that the game is over.  win_target
            // uses the OFFICIAL outcome cascade (total score → largest single
            // territory → total crowns); score-ties resolved by the cascade are
            // decisive, and only a genuine tie through all levels stays 0.5.
            // (own_score/opp_score below remain the raw scores for the margin
            // head.)  actor attribution uses the recorded rec.actor, not ply
            // parity — Kingdomino does not alternate seats reliably.
            let win0: f32 = match outcome0 {
                1 => 1.0,
                -1 => 0.0,
                _ => 0.5,
            };
            for rec in &mut self.records {
                let (own_s, opp_s, win_t) = if rec.actor == 0 {
                    (s0 as f32, s1 as f32, win0)
                } else {
                    (s1 as f32, s0 as f32, 1.0 - win0)
                };
                rec.own_score = own_s;
                rec.opp_score = opp_s;
                rec.win_target = win_t;
            }
            Ok(Some((
                self.game_seed,
                std::mem::take(&mut self.records),
                (s0, s1),
                outcome0,
            )))
        } else {
            // Next move: reset the active tree to a bare root.  Closed-loop
            // re-stores a redeterminized root state; open-loop is stateless and
            // re-evaluates from real_state on the next tick.
            if open_loop {
                self.ol_arena.clear();
                self.ol_arena.push(OLNode::new(1.0, (None, None)));
            } else {
                let root_state = self
                    .real_state
                    .redeterminize(Some(det_seed(self.game_seed, self.move_num)));
                self.arena.clear();
                self.arena.push(Node::new(1.0, (None, None)));
                self.arena[0].state = Some(root_state);
            }
            self.sims_done = 0;
            self.deck8_chance_panel_admitted_this_move = false;
            self.choose_move_profile(
                playout_cap_randomization,
                full_search_fraction,
                n_sims,
                fast_move_sims,
                record_fast_moves,
                dirichlet_eps,
                fast_move_dirichlet_eps,
                temp_moves,
                fast_move_temp_moves,
                seat_override,
            );
            // If the solver is enabled and the new root is terminal-adjacent
            // (deck ∈ {0,4}), hand it to the exact solver instead of GPU-backed
            // MCTS. The deck only shrinks, so once a game enters ExactSolving it
            // stays there until GAME_OVER — resolve_exact_slots cascades the whole
            // endgame with zero forwards. When disabled (budget 0), endgames go
            // through normal MCTS.
            self.state = if should_enter_exact_solving(exact_enabled, self) {
                SlotState::ExactSolving
            } else {
                SlotState::NeedsRootEval
            };
            Ok(None)
        }
    }
}

/// Derive a policy-target distribution from exact minimax child values via an
/// advantage-weighted softmax with a self-calibrating temperature.
///
/// `advantage_i = |v_i - v_worst|` (0 at the worst move, `range` at the best);
/// `T = range / 3`, so the best move gets ~95% of the mass when the value range
/// is large (a clear best move) and the distribution is flatter when moves are
/// close (a genuinely ambiguous endgame). No fixed hyperparameter. If all moves
/// tie (`range ≈ 0`), fall back to uniform.
///
/// Returns (policy_idx, policy_val, legal_idx) in MoveRecord format: legal_idx
/// lists every legal action; policy_idx/policy_val carry the non-negligible mass.
fn exact_policy_target(child_values: &[(u16, f64)], actor: u8) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
    let legal_idx: Vec<i32> = child_values.iter().map(|&(idx, _)| idx as i32).collect();

    let (v_best, v_worst) = if actor == 0 {
        (
            child_values
                .iter()
                .map(|&(_, v)| v)
                .fold(f64::NEG_INFINITY, f64::max),
            child_values
                .iter()
                .map(|&(_, v)| v)
                .fold(f64::INFINITY, f64::min),
        )
    } else {
        // Minimising player: "best" is the smallest value.
        (
            child_values
                .iter()
                .map(|&(_, v)| v)
                .fold(f64::INFINITY, f64::min),
            child_values
                .iter()
                .map(|&(_, v)| v)
                .fold(f64::NEG_INFINITY, f64::max),
        )
    };
    let range = (v_best - v_worst).abs();

    let weights: Vec<f64> = if range < 1e-9 {
        vec![1.0 / child_values.len() as f64; child_values.len()]
    } else {
        let temperature = range / 3.0;
        // advantage = |v - v_worst| / T  ∈ [0, 3]; softmax with max-shift for stability.
        let adv: Vec<f64> = child_values
            .iter()
            .map(|&(_, v)| (v - v_worst).abs() / temperature)
            .collect();
        let max_adv = adv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = adv.iter().map(|&a| (a - max_adv).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    };

    let mut policy_idx = Vec::new();
    let mut policy_val = Vec::new();
    for (i, &(idx, _)) in child_values.iter().enumerate() {
        let w = weights[i] as f32;
        if w > 1e-7 {
            policy_idx.push(idx as i32);
            policy_val.push(w);
        }
    }
    (policy_idx, policy_val, legal_idx)
}

/// ArgmaxTies policy target: uniform over the children whose value exactly
/// equals the best, zero elsewhere. Exact f64 equality is sound here because
/// solver utilities are integer score margins or exact quarter-point tiebreak
/// sentinels, and identical inputs map to identical training-value bits.
fn exact_policy_target_ties(
    child_values: &[(u16, f64)],
    actor: u8,
) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
    let legal_idx: Vec<i32> = child_values.iter().map(|&(idx, _)| idx as i32).collect();
    let v_best = if actor == 0 {
        child_values
            .iter()
            .map(|&(_, v)| v)
            .fold(f64::NEG_INFINITY, f64::max)
    } else {
        child_values
            .iter()
            .map(|&(_, v)| v)
            .fold(f64::INFINITY, f64::min)
    };
    let ties: Vec<u16> = child_values
        .iter()
        .filter(|&&(_, v)| v == v_best)
        .map(|&(idx, _)| idx)
        .collect();
    let w = 1.0f32 / ties.len() as f32;
    let policy_idx: Vec<i32> = ties.iter().map(|&idx| idx as i32).collect();
    let policy_val: Vec<f32> = vec![w; ties.len()];
    (policy_idx, policy_val, legal_idx)
}

fn endgame_key(state: &RustGameState) -> EndgameKey {
    let mut deck = state.deck.clone();
    deck.sort_unstable();
    EndgameKey {
        phase: state.phase,
        actor_index: state.actor_index,
        deck,
        current_row: state.current_row.clone(),
        pending_claims: state.pending_claims.clone(),
        next_claims: state.next_claims.clone(),
        board0_terrain: state.boards[0].terrain,
        board0_crowns: state.boards[0].crowns,
        board1_terrain: state.boards[1].terrain,
        board1_crowns: state.boards[1].crowns,
    }
}

fn best_exact_joint(result: &ExactSolveResult, actor: u8) -> Option<u16> {
    let best = if actor == 0 {
        result
            .child_values
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    } else {
        result
            .child_values
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    };
    best.map(|&(idx, _)| idx)
}

#[allow(clippy::too_many_arguments)]
fn solve_endgame_ab_value_cached(
    state: &RustGameState,
    deadline: std::time::Instant,
    alpha: f64,
    beta: f64,
    mode: SolverOrderMode,
    value_cache: &mut HashMap<EndgameKey, f64>,
) -> PyResult<Option<f64>> {
    // Cache only full-window solves (exact solver utilities). ±200 brackets
    // every raw margin and tiebreak sentinel, so such a window is exact.
    let full_window = alpha <= MARGIN_LO && beta >= MARGIN_HI;
    let key = if full_window {
        Some(endgame_key(state))
    } else {
        None
    };
    if let Some(k) = key.as_ref() {
        if let Some(&v) = value_cache.get(k) {
            return Ok(Some(v));
        }
    }

    let v = solve_endgame_ab(state, deadline, alpha, beta, mode, 0)?;
    if let (Some(k), Some(value)) = (key, v) {
        value_cache.insert(k, value);
    }
    Ok(v)
}

/// Solve a terminal-adjacent root exactly, returning per-child minimax values for
/// ALL legal root actions (player-0 frame). Each child is solved with a full
/// (-∞, +∞) window so its value is exact (needed for the policy target), sharing
/// one wall-clock `deadline` across children. Returns `Ok(None)` if the deadline
/// is exceeded (caller falls back to MCTS).
fn solve_root_exact_cached(
    state: &RustGameState,
    deadline: std::time::Instant,
    score_scale: f64,
    margin_gain: f64,
    alpha_param: f64,
    policy_mode: ExactPolicyMode,
    clamp_delta: f64,
    result_cache: &mut HashMap<EndgameKey, ExactSolveResult>,
) -> PyResult<Option<ExactSolveResult>> {
    let key = endgame_key(state);
    if let Some(result) = result_cache.get(&key) {
        return Ok(Some(result.clone()));
    }
    match solve_root_exact(
        state,
        deadline,
        score_scale,
        margin_gain,
        alpha_param,
        SolverOrderMode::Lookahead2Clustered,
        policy_mode,
        clamp_delta,
    )? {
        Some(result) => {
            result_cache.insert(key, result.clone());
            Ok(Some(result))
        }
        None => Ok(None),
    }
}

/// Solve one exact root under `policy_mode` (see `ExactPolicyMode`).
///
/// - `Exact`: every child solved with a full window IN PARALLEL — verbatim the
///   historical behavior (bit-identical child values and policy labels).
/// - `SoftClamp` / `ArgmaxTies`: the first (best-ordered) child is solved
///   full-window serially to establish the bound `b0`, then the remaining
///   children are solved in parallel with the one-sided window
///   `(b0 - delta, +inf)` (mover-maximising frame; mirrored for the minimiser).
///   Fail-soft: a sibling whose true value lies inside the window returns it
///   exactly (including any that BEAT b0 — the final best is exact in every
///   case); one outside fails cheaply and is recorded at the clamp value
///   `best_final ∓ delta`, a valid upper bound on its worth to the mover.
///
/// The root value (best child) and minimax move are exact in all modes; the
/// deadline is shared across all children, `Ok(None)` = budget exceeded.
#[allow(clippy::too_many_arguments)]
fn solve_root_exact(
    state: &RustGameState,
    deadline: std::time::Instant,
    score_scale: f64,
    margin_gain: f64,
    alpha_param: f64,
    order_mode: SolverOrderMode,
    policy_mode: ExactPolicyMode,
    clamp_delta: f64,
) -> PyResult<Option<ExactSolveResult>> {
    let mut legal = state.legal_actions_indexed();
    if legal.is_empty() {
        return Ok(None); // not GAME_OVER but no actions — fall back defensively
    }
    order_legal_for_solver_at_depth(state, &mut legal, order_mode, 0)?;

    if std::time::Instant::now() >= deadline {
        return Ok(None);
    }
    let ttv = |utility: f64| {
        solver_utility_to_training_value(utility, score_scale, margin_gain, alpha_param)
    };
    // One transposition table for the WHOLE root solve: sibling subtrees
    // overlap heavily (62-86% duplicate interior visits measured on the real
    // fallback corpus), so sharing it across children attacks the same
    // redundancy that made per-child solving ~11x a value-only solve.
    let tt = TranspositionTable::new();

    if policy_mode == ExactPolicyMode::Exact || legal.len() == 1 {
        // Historical path: every child full-window, all in parallel.
        let child_results: Vec<PyResult<Option<(u16, f64)>>> = legal
            .par_iter()
            .map(
                |&(joint_idx, placement, pick)| -> PyResult<Option<(u16, f64)>> {
                    let next = state.step(placement, pick)?;
                    let mut buf = Vec::with_capacity(1024);
                    match solve_endgame_ab_tt(
                        &next, deadline, MARGIN_LO, MARGIN_HI, order_mode, 0, &tt, &mut buf,
                    )? {
                        Some(solver_utility) => Ok(Some((joint_idx, ttv(solver_utility)))),
                        None => Ok(None),
                    }
                },
            )
            .collect();
        let mut child_values: Vec<(u16, f64)> = Vec::with_capacity(legal.len());
        for r in child_results {
            match r? {
                Some(cv) => child_values.push(cv),
                None => return Ok(None), // a child hit the deadline → whole solve fails
            }
        }
        return Ok(Some(ExactSolveResult {
            child_values,
            policy_mode,
            n_clamped: 0,
        }));
    }

    let actor = state.actor()?;
    let delta = if policy_mode == ExactPolicyMode::ArgmaxTies {
        // Solver utilities differ by at least one quarter point: raw margins
        // are integers and equal-score official outcomes are -0.25/0/+0.25.
        OFFICIAL_TIEBREAK_UTILITY
    } else {
        clamp_delta
    };

    // First (best-ordered) child: serial, full window → exact bound b0.
    let (_i0, p0, pk0) = legal[0];
    let first = state.step(p0, pk0)?;
    let mut buf0 = Vec::with_capacity(1024);
    let Some(v0) = solve_endgame_ab_tt(
        &first, deadline, MARGIN_LO, MARGIN_HI, order_mode, 0, &tt, &mut buf0,
    )?
    else {
        return Ok(None);
    };

    // Siblings: parallel, one-sided window `delta` beyond b0 on the mover's
    // losing side. Fail-soft classification below distinguishes exact returns
    // from bound returns.
    let (w_lo, w_hi) = if actor == 0 {
        ((v0 - delta).max(MARGIN_LO), MARGIN_HI)
    } else {
        (MARGIN_LO, (v0 + delta).min(MARGIN_HI))
    };
    let sibling_results: Vec<PyResult<Option<(u16, f64)>>> = legal[1..]
        .par_iter()
        .map(
            |&(joint_idx, placement, pick)| -> PyResult<Option<(u16, f64)>> {
                let next = state.step(placement, pick)?;
                let mut buf = Vec::with_capacity(1024);
                Ok(
                    solve_endgame_ab_tt(&next, deadline, w_lo, w_hi, order_mode, 0, &tt, &mut buf)?
                        .map(|raw| (joint_idx, raw)),
                )
            },
        )
        .collect();
    let mut raw_values: Vec<(u16, f64)> = Vec::with_capacity(legal.len());
    raw_values.push((legal[0].0, v0));
    for r in sibling_results {
        match r? {
            Some(rv) => raw_values.push(rv),
            None => return Ok(None),
        }
    }

    // Fail-soft: exact iff the return lies strictly inside the window on the
    // mover's losing side (the winning side is unbounded, so any improvement
    // over b0 — including the final best — is exact).
    let is_exact = |r: f64| if actor == 0 { r > w_lo } else { r < w_hi };
    let b_final = raw_values
        .iter()
        .filter(|&&(_, r)| is_exact(r))
        .map(|&(_, r)| r)
        .fold(
            if actor == 0 {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            },
            |acc, r| {
                if actor == 0 { acc.max(r) } else { acc.min(r) }
            },
        );
    let clamp_raw = if actor == 0 {
        b_final - delta
    } else {
        b_final + delta
    };
    let mut n_clamped: u16 = 0;
    let child_values: Vec<(u16, f64)> = raw_values
        .iter()
        .map(|&(idx, r)| {
            if is_exact(r) {
                (idx, ttv(r))
            } else {
                n_clamped += 1;
                (idx, ttv(clamp_raw))
            }
        })
        .collect();
    Ok(Some(ExactSolveResult {
        child_values,
        policy_mode,
        n_clamped,
    }))
}

/// A dispatched endgame solve (Step 1.5 async path): an owned snapshot of the
/// slot's game. Sent to the background solver thread; the slot keeps its own copy.
struct SolveJob {
    slot_idx: usize,
    kind: ExactAttemptKind,
    state: RustGameState,
    records: Vec<MoveRecord>,
    game_seed: u64,
    move_num: usize,
}

enum SolveResult {
    /// The endgame solved to GAME_OVER: full game plus official P0 outcome.
    Finished(FinishedGame),
    /// Deadline exceeded (or solve error): the slot must resume MCTS in place.
    Fallback,
}

/// Result of one background solve, harvested by the main thread on the next step().
struct SolveOutcome {
    slot_idx: usize,
    kind: ExactAttemptKind,
    result: SolveResult,
    n_solved: u64, // plan length (for counters); 0 on fallback
    solve_secs: f64,
    fallback_state: Option<RustGameState>,
    game_seed: u64,
    move_num: usize,
}

/// Spawn the single background solver thread (concurrency 1, so each solve keeps
/// the whole machine via the within-solve YBW `par_iter`). It pulls jobs, solves
/// each endgame to completion (or fails to Fallback), and returns outcomes. Pure
/// Rust — no GIL touched (solve_exact_plan / play_out_exact_endgame only build a
/// `PyErr` lazily on the error path, which is discarded here). Exits when the job
/// channel closes (BatchedMCTS dropped).
fn spawn_endgame_solver(
    job_rx: std::sync::mpsc::Receiver<SolveJob>,
    out_tx: std::sync::mpsc::Sender<SolveOutcome>,
    max_secs: f64,
    score_scale: f64,
    margin_gain: f64,
    alpha: f64,
    policy_mode: ExactPolicyMode,
    clamp_delta: f64,
    solver_pool: rayon::ThreadPool,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        while let Ok(job) = job_rx.recv() {
            let SolveJob {
                slot_idx,
                kind,
                state,
                records,
                game_seed,
                move_num,
            } = job;
            let t0 = std::time::Instant::now();
            let state_for_fallback = state.cloned();
            // Confine the within-solve YBW par_iter to the DEDICATED solver pool so
            // its long, non-preemptible subtree-solves never head-of-line-block the
            // GLOBAL pool that step()/update() use to feed the GPU. The pool's thread
            // count is the gen/solver core split (solver_cpus); generation gets the
            // rest via the global pool.
            let (result, n_solved) = solver_pool.install(move || {
                match solve_exact_plan(
                    &state,
                    max_secs,
                    score_scale,
                    margin_gain,
                    alpha,
                    policy_mode,
                    clamp_delta,
                ) {
                    Ok(Some(plan)) if !plan.is_empty() => {
                        let n = plan.len() as u64;
                        match play_out_exact_endgame(state, records, game_seed, plan) {
                            Ok(fg) => (SolveResult::Finished(fg), n),
                            Err(_) => (SolveResult::Fallback, 0),
                        }
                    }
                    _ => (SolveResult::Fallback, 0),
                }
            });
            let is_fallback = matches!(&result, SolveResult::Fallback);
            let outcome = SolveOutcome {
                slot_idx,
                kind,
                fallback_state: if is_fallback {
                    Some(state_for_fallback)
                } else {
                    None
                },
                result,
                n_solved,
                solve_secs: t0.elapsed().as_secs_f64(),
                game_seed,
                move_num,
            };
            if out_tx.send(outcome).is_err() {
                break; // main side dropped the receiver
            }
        }
    })
}

/// Play a solved endgame to completion on OWNED data — no slot, no MCTS tree.
/// `plan` is the exact continuation from `state` (item i's child_values are for
/// the position after i moves); `records` already holds the game's pre-endgame
/// MCTS moves and gets one MoveRecord appended per endgame move. The plan always
/// reaches GAME_OVER (solve_exact_plan only returns a full plan), so this finishes
/// the game and fills every record's final-score targets. This is the standalone
/// unit shared by the synchronous solver and the async background solver (Step 1.5).
fn play_out_exact_endgame(
    mut state: RustGameState,
    mut records: Vec<MoveRecord>,
    game_seed: u64,
    plan: Vec<ExactPlanItem>,
) -> PyResult<FinishedGame> {
    for item in plan {
        let exact = item.result;
        let actor = state.actor()?;
        let (my, opp, flat) = state.encode_arrays(actor)?;
        let (policy_idx, policy_val, legal_idx) = exact.policy_target(actor);
        // Optimal move is unambiguous: minimax-best child (no temperature).
        let best = if actor == 0 {
            exact
                .child_values
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        } else {
            exact
                .child_values
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        };
        let chosen = best.map(|&(idx, _)| idx).unwrap_or(legal_idx[0] as u16);
        records.push(MoveRecord {
            my,
            opp,
            flat,
            policy_idx,
            policy_val,
            legal_idx,
            root_prior_idx: Vec::new(),
            root_prior_val: Vec::new(),
            root_visit_count: Vec::new(),
            actor,
            own_score: 0.0,
            opp_score: 0.0,
            win_target: 0.5,
        });
        let (placement, pick) = state
            .legal_actions_indexed()
            .into_iter()
            .find(|t| t.0 == chosen)
            .map(|t| (t.1, t.2))
            .ok_or_else(|| PyValueError::new_err("exact endgame: selected index not legal"))?;
        state = state.step(placement, pick)?;
    }
    // Plan plays to GAME_OVER; fill per-move targets. win_target uses the
    // OFFICIAL outcome cascade (matching finalize_move); own_score/opp_score
    // stay raw for the margin head; actor attribution uses recorded rec.actor.
    let (s0, s1) = state.scores();
    let outcome0 = state.official_outcome_i8();
    let win0: f32 = match outcome0 {
        1 => 1.0,
        -1 => 0.0,
        _ => 0.5,
    };
    for rec in &mut records {
        let (own_s, opp_s, win_t) = if rec.actor == 0 {
            (s0 as f32, s1 as f32, win0)
        } else {
            (s1 as f32, s0 as f32, 1.0 - win0)
        };
        rec.own_score = own_s;
        rec.opp_score = opp_s;
        rec.win_target = win_t;
    }
    Ok((game_seed, records, (s0, s1), outcome0))
}

fn solve_exact_plan(
    state: &RustGameState,
    max_secs: f64,
    score_scale: f64,
    margin_gain: f64,
    alpha_param: f64,
    policy_mode: ExactPolicyMode,
    clamp_delta: f64,
) -> PyResult<Option<Vec<ExactPlanItem>>> {
    let mut cur = state.cloned();
    // One shared deadline for the whole endgame cascade from this root. The plan
    // is built once and reused (cache hits) for the deterministic continuation,
    // so this bounds the once-per-game expensive solve, not each move.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(max_secs);
    let mut result_cache: HashMap<EndgameKey, ExactSolveResult> = HashMap::new();
    let mut plan = Vec::new();

    while cur.phase != GAME_OVER {
        if !is_no_chance_endgame_state(&cur) || std::time::Instant::now() >= deadline {
            return Ok(None);
        }
        let result = match solve_root_exact_cached(
            &cur,
            deadline,
            score_scale,
            margin_gain,
            alpha_param,
            policy_mode,
            clamp_delta,
            &mut result_cache,
        )? {
            Some(r) => r,
            None => return Ok(None),
        };
        let actor = cur.actor()?;
        let chosen = best_exact_joint(&result, actor)
            .ok_or_else(|| PyValueError::new_err("exact plan state has no best action"))?;
        let (placement, pick) = cur
            .legal_actions_indexed()
            .into_iter()
            .find(|t| t.0 == chosen)
            .map(|t| (t.1, t.2))
            .ok_or_else(|| PyValueError::new_err("exact plan selected illegal action"))?;
        plan.push(ExactPlanItem { result });
        cur = cur.step(placement, pick)?;
    }
    Ok(Some(plan))
}

#[cfg(test)]
mod terminal_tiebreak_tests {
    use super::*;

    #[test]
    fn training_conversion_keeps_tiebreak_outcome_but_zero_raw_margin() {
        let alpha = 0.8;
        let expected = 1.0 - alpha;
        assert_eq!(
            solver_utility_to_training_value(OFFICIAL_TIEBREAK_UTILITY, 100.0, 2.0, alpha,),
            expected,
        );
        assert_eq!(
            solver_utility_to_training_value(-OFFICIAL_TIEBREAK_UTILITY, 100.0, 2.0, alpha,),
            -expected,
        );
        assert_eq!(
            solver_utility_to_training_value(0.0, 100.0, 2.0, alpha),
            0.0
        );
    }

    #[test]
    fn terminal_backup_and_solver_utility_use_official_tiebreak() -> PyResult<()> {
        let mut found = None;
        for seed in 0..600u64 {
            let mut state = new_game(seed, true, true);
            let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(7).wrapping_add(3));
            while state.phase != GAME_OVER {
                let legal = state.legal_actions_indexed();
                let &(_idx, placement, pick) = legal
                    .choose(&mut rng)
                    .ok_or_else(|| PyValueError::new_err("non-terminal state has no action"))?;
                state = state.step(placement, pick)?;
            }
            let (s0, s1) = state.scores();
            let outcome = state.official_outcome_i8();
            if s0 == s1 && outcome != 0 {
                found = Some((state, outcome));
                break;
            }
        }
        let (state, outcome) = found.expect("no tiebreak-decided raw-score tie in 600 games");
        assert_eq!(
            terminal_solver_utility(&state),
            outcome as f64 * OFFICIAL_TIEBREAK_UTILITY,
        );
        assert!(
            (terminal_search_value(&state, 100.0, 2.0, 0.8) - outcome as f64 * 0.2).abs() < 1e-12,
        );
        Ok(())
    }
}

#[cfg(test)]
mod make_unmake_tests {
    //! Differential gate for the mutable make/unmake engine.  The functional
    //! `step` is the oracle: `make` must reproduce `step` byte-for-byte, and
    //! `make` + `unmake` must round-trip to the exact prior state.  `fingerprint`
    //! serializes EVERY field — including the board bounding box, `occupied`, and
    //! the deck in ORDER — so the bbox-restore hazard and any deck-order leak are
    //! caught, not masked (`solver_state_bytes` sorts the deck and omits the bbox,
    //! so it is deliberately NOT reused here).
    use super::*;

    /// Full structural fingerprint of a state (all fields, deck in order).
    fn fingerprint(s: &RustGameState) -> Vec<u8> {
        let mut b: Vec<u8> = Vec::new();
        b.push(s.phase);
        b.extend_from_slice(&(s.actor_index as u64).to_le_bytes());
        b.extend_from_slice(&(s.initial_pick_count as u64).to_le_bytes());
        b.push(s.start_player);
        b.push(s.harmony as u8);
        b.push(s.middle_kingdom as u8);
        b.extend_from_slice(&s.discards[0].to_le_bytes());
        b.extend_from_slice(&s.discards[1].to_le_bytes());
        for &d in &s.deck {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.push(0xFF); // section separator (deck len is variable)
        for &d in &s.current_row {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.push(0xFF);
        for &(p, d) in &s.pending_claims {
            b.push(p);
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.push(0xFF);
        for &(p, d) in &s.next_claims {
            b.push(p);
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.push(0xFF);
        for brd in &s.boards {
            b.extend_from_slice(&brd.terrain);
            b.extend_from_slice(&brd.crowns);
            b.extend_from_slice(&[
                brd.castle_x as u8,
                brd.castle_y as u8,
                brd.min_x as u8,
                brd.max_x as u8,
                brd.min_y as u8,
                brd.max_y as u8,
                brd.occupied,
            ]);
        }
        b
    }

    /// At every ply of many random games: (1) `make` on a clone reproduces `step`
    /// byte-for-byte; (2) `make` then `unmake` round-trips to the prior state.
    /// Random play naturally covers all phases, forced discards, round
    /// boundaries, and the terminal transition.
    #[test]
    fn make_matches_step_and_round_trips() -> PyResult<()> {
        let mut games = 0;
        let mut plies = 0;
        let mut discards_seen = 0;
        let mut boundaries_seen = 0;
        for seed in 0..200u64 {
            let mut rng = StdRng::seed_from_u64(seed ^ 0x5A17);
            let mut state = new_game(seed, true, true);
            games += 1;
            for _ in 0..400 {
                if state.phase == GAME_OVER {
                    break;
                }
                let legal = state.legal_actions();
                let &(p, pk) = legal.choose(&mut rng).expect("nonempty legal actions");
                // A forced discard is a placement-PHASE move with no placement.
                // INITIAL_SELECTION picks also carry placement=None, so they must
                // be excluded or the coverage assertion is a false positive.
                if state.phase != INITIAL_SELECTION && p.is_none() {
                    discards_seen += 1;
                }

                // (1) make == step
                let stepped = state.step(p, pk)?;
                let mut m = state.cloned();
                let _ = m.make(p, pk)?;
                assert_eq!(
                    fingerprint(&m),
                    fingerprint(&stepped),
                    "seed {seed} ply {plies}: make diverged from step"
                );

                // (2) round-trip
                let before = fingerprint(&state);
                let mut r = state.cloned();
                let rec = r.make(p, pk)?;
                if matches!(
                    rec,
                    UndoRecord::InitialPick {
                        boundary: Some(_),
                        ..
                    } | UndoRecord::Move {
                        boundary: Some(_),
                        ..
                    }
                ) {
                    boundaries_seen += 1;
                }
                r.unmake(rec);
                assert_eq!(
                    fingerprint(&r),
                    before,
                    "seed {seed} ply {plies}: unmake did not restore prior state"
                );

                state = stepped;
                plies += 1;
            }
        }
        // Guard the test actually exercised the interesting transitions.
        assert!(games >= 100, "too few games ({games})");
        assert!(discards_seen >= 1, "no forced discards exercised");
        assert!(
            boundaries_seen >= 50,
            "too few round boundaries ({boundaries_seen})"
        );
        Ok(())
    }

    /// A single mutable state, walked to game over via `make` (records pushed on a
    /// stack) and then fully unwound via `unmake`, must return byte-identical to
    /// the start — the real search-walk discipline, not just isolated plies.
    #[test]
    fn full_playout_unwinds_to_start() -> PyResult<()> {
        for seed in 0..60u64 {
            let mut rng = StdRng::seed_from_u64(seed ^ 0xB00C);
            let mut state = new_game(seed, true, true);
            let start = fingerprint(&state);
            let mut stack: Vec<UndoRecord> = Vec::new();
            // Parallel functional reference advanced by `step`.
            let mut reference = state.cloned();
            for _ in 0..400 {
                if state.phase == GAME_OVER {
                    break;
                }
                let legal = state.legal_actions();
                let &(p, pk) = legal.choose(&mut rng).expect("nonempty legal actions");
                let rec = state.make(p, pk)?;
                reference = reference.step(p, pk)?;
                assert_eq!(
                    fingerprint(&state),
                    fingerprint(&reference),
                    "seed {seed}: running make state diverged from step"
                );
                stack.push(rec);
            }
            assert_eq!(
                state.phase, GAME_OVER,
                "seed {seed}: game did not reach GAME_OVER within 400 plies — \
                 the 'walked to game over' property was not actually exercised"
            );
            assert!(!stack.is_empty(), "seed {seed}: game produced no moves");
            while let Some(rec) = stack.pop() {
                state.unmake(rec);
            }
            assert_eq!(
                fingerprint(&state),
                start,
                "seed {seed}: unwinding the full game did not restore the start"
            );
        }
        Ok(())
    }

    /// Atomicity: an illegal action must return `Err` and leave the state
    /// byte-identical (the documented guarantee that fallible lookups run before
    /// any mutation and `place` validates before writing). Covers all four paths:
    /// invalid opening pick, placement supplied in INITIAL_SELECTION, invalid pick
    /// with a legal placement, illegal placement with a valid pick, and `make` on
    /// a terminal state.
    #[test]
    fn make_rejects_illegal_actions_without_mutating() -> PyResult<()> {
        const BAD_ID: u16 = 9999;

        // --- INITIAL_SELECTION ---
        let mut s0 = new_game(1, true, true);
        assert_eq!(s0.phase, INITIAL_SELECTION);
        let fp0 = fingerprint(&s0);
        assert!(s0.make(None, Some(BAD_ID)).is_err(), "invalid opening pick");
        assert_eq!(fingerprint(&s0), fp0, "errored opening pick mutated state");
        assert!(
            s0.make(Some((7, 7, 7, 8, false)), Some(0)).is_err(),
            "placement in INITIAL_SELECTION must be rejected"
        );
        assert_eq!(
            fingerprint(&s0),
            fp0,
            "errored initial placement mutated state"
        );

        // --- PLACE_AND_SELECT with a real placement action available ---
        let mut rng = StdRng::seed_from_u64(0xA704);
        let mut s = new_game(2, true, true);
        let mut found: Option<(RustGameState, Option<(i8, i8, i8, i8, bool)>, Option<u16>)> = None;
        for _ in 0..400 {
            if s.phase == GAME_OVER {
                break;
            }
            if s.phase == PLACE_AND_SELECT {
                if let Some(&(p, pk)) = s.legal_actions().iter().find(|(p, _)| p.is_some()) {
                    found = Some((s.cloned(), p, pk));
                    break;
                }
            }
            let legal = s.legal_actions();
            let &(p, pk) = legal.choose(&mut rng).expect("nonempty legal actions");
            s = s.step(p, pk)?;
        }
        let (mut sp, p, pk) = found.expect("no PLACE_AND_SELECT placement action found");
        let fpp = fingerprint(&sp);
        // Invalid pick + legal placement: pick is resolved before placement, so it
        // errors with the board untouched.
        assert!(
            sp.make(p, Some(BAD_ID)).is_err(),
            "invalid pick should error"
        );
        assert_eq!(fingerprint(&sp), fpp, "errored invalid pick mutated state");
        // Illegal placement + valid pick: `place` validates before writing, so it
        // errors with no pick mutation applied.
        let illegal = Some((0i8, 0i8, 0i8, 1i8, false)); // far from castle -> illegal
        assert!(
            sp.make(illegal, pk).is_err(),
            "illegal placement should error"
        );
        assert_eq!(
            fingerprint(&sp),
            fpp,
            "errored illegal placement mutated state"
        );

        // --- terminal ---
        let mut t = new_game(3, true, true);
        let mut rng2 = StdRng::seed_from_u64(0xDEAD);
        for _ in 0..400 {
            if t.phase == GAME_OVER {
                break;
            }
            let legal = t.legal_actions();
            let &(p, pk) = legal.choose(&mut rng2).expect("nonempty legal actions");
            t = t.step(p, pk)?;
        }
        assert_eq!(t.phase, GAME_OVER);
        let fpt = fingerprint(&t);
        assert!(t.make(None, None).is_err(), "make on terminal should error");
        assert_eq!(fingerprint(&t), fpt, "errored terminal make mutated state");
        Ok(())
    }

    /// The operational search may extend a horizon only when the game reports an
    /// exact deterministic distance to GAME_OVER. Prove that Kingdomino's count
    /// is exact on every tail state encountered across random full games.
    #[test]
    fn operational_exact_tail_count_is_exact() -> PyResult<()> {
        let mut checked = 0usize;
        let mut saw_deck4 = false;
        let mut saw_deck0 = false;
        let mut saw_final = false;
        for seed in 0..40u64 {
            let mut rng = StdRng::seed_from_u64(seed ^ 0x0EAC_7A11);
            let mut state = new_game(seed, true, true);
            while state.phase != GAME_OVER {
                if let Some(predicted) = <Kingdomino as search::Game>::exact_remaining_plies(&state)
                {
                    saw_deck4 |= state.phase == PLACE_AND_SELECT && state.deck.len() == 4;
                    saw_deck0 |= state.phase == PLACE_AND_SELECT && state.deck.is_empty();
                    saw_final |= state.phase == FINAL_PLACEMENT;
                    let mut tail = state.cloned();
                    let mut actual = 0u32;
                    while tail.phase != GAME_OVER {
                        let legal = tail.legal_actions();
                        let &(placement, pick) = legal
                            .choose(&mut rng)
                            .expect("deterministic tail must expose an action");
                        tail = tail.step(placement, pick)?;
                        actual += 1;
                    }
                    assert_eq!(
                        predicted,
                        actual,
                        "seed {seed}: exact tail count wrong at phase {} deck {} actor {}/{}",
                        state.phase,
                        state.deck.len(),
                        state.actor_index,
                        state.pending_claims.len()
                    );
                    checked += 1;
                }
                let legal = state.legal_actions();
                let &(placement, pick) = legal.choose(&mut rng).expect("nonempty legal actions");
                state = state.step(placement, pick)?;
            }
        }
        assert!(
            checked >= 100,
            "too few exact-tail states checked ({checked})"
        );
        assert!(
            saw_deck4 && saw_deck0 && saw_final,
            "tail phase coverage incomplete"
        );
        Ok(())
    }

    /// Real Kingdomino sampled-chance gate: the operational path must select the
    /// fixed-depth oracle's move, restore the mutable state, and recover enough
    /// pruning to pay for its shallower iterative-deepening passes.
    #[test]
    fn operational_real_tree_matches_fixed_and_reduces_nodes() -> PyResult<()> {
        let mut root = None;
        for seed in 0..80u64 {
            let mut rng = StdRng::seed_from_u64(seed ^ 0x0F45_7B0A);
            let mut state = new_game(seed, true, true);
            while state.phase != GAME_OVER {
                if state.phase == PLACE_AND_SELECT
                    && state.deck.len() >= 12
                    && state.actor_index + 1 == state.pending_claims.len()
                    && state.legal_actions().len() >= 2
                {
                    root = Some(state);
                    break;
                }
                let legal = state.legal_actions();
                let &(placement, pick) = legal.choose(&mut rng).expect("nonempty legal actions");
                state = state.step(placement, pick)?;
            }
            if root.is_some() {
                break;
            }
        }
        let root = root.expect("no sampled-chance operational fixture found");
        let before = fingerprint(&root);
        let cfg = search::SearchConfig {
            depth: 4,
            chance_samples: 8,
            enum_cap: 1,
            margin_weight: 0.0,
            seed: 17,
        };

        let mut fixed_state = root.cloned();
        let mut fixed_nodes = 0u64;
        let fixed_start = std::time::Instant::now();
        let fixed = search::choose_action::<Kingdomino, _>(
            &mut fixed_state,
            &PickAwareEval,
            &cfg,
            None,
            &mut fixed_nodes,
        )?
        .expect("fixed search action");
        let fixed_secs = fixed_start.elapsed().as_secs_f64();

        let mut operational_state = root.cloned();
        let operational_start = std::time::Instant::now();
        let result = search::choose_action_operational::<Kingdomino, _>(
            &mut operational_state,
            &PickAwareEval,
            &cfg,
            &search::OperationalLimits {
                max_depth: 4,
                deadline: std::time::Instant::now() + std::time::Duration::from_secs(30),
                aspiration_window: 0.25,
                node_limit: None,
                value_bound: 1.0,
                full_width_ordering: false,
                selective_width: None,
                selective_root_width: None,
                selective_min_depth: 4,
            },
        )?
        .expect("operational search action");
        let operational_secs = operational_start.elapsed().as_secs_f64();
        assert_eq!(result.completed_depth, 4);
        assert!(!result.timed_out);
        assert!(result.chance_nodes > 0, "live deal node was not counted");
        assert_eq!(result.action, fixed);
        assert_eq!(
            fingerprint(&operational_state),
            before,
            "search did not unwind root"
        );
        eprintln!(
            "operational depth4: {:.3}s, {} total / {} final-iteration nodes vs fixed {:.3}s / {} nodes ({:.2}x final ratio), star cutoffs {}, aspiration re-searches {}, TT hits {}, TT cutoffs {}",
            operational_secs,
            result.nodes,
            result.last_iteration_nodes,
            fixed_secs,
            fixed_nodes,
            fixed_nodes as f64 / result.last_iteration_nodes as f64,
            result.star_cutoffs,
            result.aspiration_researches,
            result.tt_hits,
            result.tt_cutoffs,
        );
        assert!(
            result.last_iteration_nodes <= fixed_nodes,
            "operational final iteration regressed: {} vs fixed {} nodes",
            result.last_iteration_nodes,
            fixed_nodes
        );
        Ok(())
    }

    /// The public-state operational counter records explicit deal layers only:
    /// a live sampled deal must be non-zero, while a deck-in-{0,4} exact tail
    /// must never invent a chance node even when the search is node-limited.
    #[test]
    fn operational_chance_nodes_live_vs_no_chance_endgame() -> PyResult<()> {
        let cfg = search::SearchConfig {
            depth: 4,
            chance_samples: 2,
            enum_cap: 1,
            margin_weight: 0.0,
            seed: 23,
        };
        let limits = |max_depth, node_limit| search::OperationalLimits {
            max_depth,
            deadline: std::time::Instant::now() + std::time::Duration::from_secs(10),
            aspiration_window: 0.25,
            node_limit,
            value_bound: 1.0,
            full_width_ordering: true,
            selective_width: None,
            selective_root_width: None,
            selective_min_depth: 4,
        };

        let mut live = new_game(101, true, true);
        let live_result = search::choose_action_operational::<Kingdomino, _>(
            &mut live,
            &PickAwareEval,
            &cfg,
            &limits(4, Some(50_000)),
        )?
        .expect("live-chance search action");
        assert!(
            live_result.chance_nodes > 0,
            "opening deal layer was not counted"
        );

        let mut rng = StdRng::seed_from_u64(0xC11A_CE00);
        let mut tail = new_game(102, true, true);
        while !is_no_chance_endgame_state(&tail) || tail.legal_actions().len() < 2 {
            let legal = tail.legal_actions();
            let &(placement, pick) = legal.choose(&mut rng).expect("nonempty legal actions");
            tail = tail.step(placement, pick)?;
        }
        let tail_result = search::choose_action_operational::<Kingdomino, _>(
            &mut tail,
            &PickAwareEval,
            &cfg,
            &limits(1, Some(200)),
        )?
        .expect("no-chance tail search action");
        assert_eq!(tail_result.chance_nodes, 0);
        Ok(())
    }
}

#[cfg(test)]
mod solver_restructure_tests {
    use super::*;

    /// Play a random game to its first no-chance endgame root (deck ∈ {0,4}).
    fn first_endgame_root(seed: u64) -> Option<RustGameState> {
        let mut rng = StdRng::seed_from_u64(seed ^ 0xE17D);
        let mut state = new_game(seed, true, true);
        for _ in 0..200 {
            if state.phase == GAME_OVER {
                return None;
            }
            if is_no_chance_endgame_state(&state) && state.legal_actions_indexed().len() >= 2 {
                return Some(state);
            }
            let legal = state.legal_actions_indexed();
            let &(_i, p, pk) = legal.choose(&mut rng)?;
            state = state.step(p, pk).ok()?;
        }
        None
    }

    fn far_deadline() -> std::time::Instant {
        std::time::Instant::now() + std::time::Duration::from_secs(60)
    }

    /// TT solver == plain solver on full-window solves (true minimax value).
    #[test]
    fn tt_solver_matches_plain_full_window() -> PyResult<()> {
        let mode = SolverOrderMode::Lookahead2Clustered;
        let mut checked = 0;
        for seed in 0..24u64 {
            let Some(root) = first_endgame_root(seed) else {
                continue;
            };
            let plain = solve_endgame_ab(&root, far_deadline(), MARGIN_LO, MARGIN_HI, mode, 0)?
                .expect("plain solve");
            let tt = TranspositionTable::new();
            let mut buf = Vec::new();
            let tabled = solve_endgame_ab_tt(
                &root,
                far_deadline(),
                MARGIN_LO,
                MARGIN_HI,
                mode,
                0,
                &tt,
                &mut buf,
            )?
            .expect("tt solve");
            assert_eq!(plain, tabled, "seed {seed}: TT value diverged");
            checked += 1;
        }
        assert!(checked >= 5, "too few endgame roots reached ({checked})");
        Ok(())
    }

    /// Advisor/value path: YBW parallel solver WITH the shared TT matches the
    /// untabled serial solver's root value on real endgame roots.
    #[test]
    fn parallel_tt_matches_serial_plain() -> PyResult<()> {
        let mode = SolverOrderMode::Lookahead2Clustered;
        let mut checked = 0;
        for seed in 0..24u64 {
            let Some(root) = first_endgame_root(seed) else {
                continue;
            };
            let serial = solve_endgame_ab(&root, far_deadline(), MARGIN_LO, MARGIN_HI, mode, 0)?
                .expect("serial solve");
            let parallel =
                solve_endgame_ab_parallel(&root, far_deadline(), mode)?.expect("parallel TT solve");
            assert_eq!(serial, parallel, "seed {seed}: parallel+TT value diverged");
            checked += 1;
        }
        assert!(checked >= 5, "too few endgame roots reached ({checked})");
        Ok(())
    }

    /// All three policy modes agree on the root value and the minimax-best
    /// child set; soft_clamp only ever RAISES dominated children's values.
    #[test]
    fn policy_modes_agree_on_value_and_move() -> PyResult<()> {
        let mode = SolverOrderMode::Lookahead2Clustered;
        let mut checked = 0;
        for seed in 0..24u64 {
            let Some(root) = first_endgame_root(seed) else {
                continue;
            };
            let actor = root.actor()?;
            let best_of = |r: &ExactSolveResult| -> f64 {
                r.child_values.iter().map(|&(_, v)| v).fold(
                    if actor == 0 {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    },
                    |a, v| {
                        if actor == 0 { a.max(v) } else { a.min(v) }
                    },
                )
            };
            let solve = |pm: ExactPolicyMode| -> PyResult<ExactSolveResult> {
                Ok(
                    solve_root_exact(&root, far_deadline(), 160.0, 2.0, 0.5, mode, pm, 10.0)?
                        .expect("root solve"),
                )
            };
            let exact = solve(ExactPolicyMode::Exact)?;
            let clamp = solve(ExactPolicyMode::SoftClamp)?;
            let ties = solve(ExactPolicyMode::ArgmaxTies)?;
            let (vb_e, vb_c, vb_t) = (best_of(&exact), best_of(&clamp), best_of(&ties));
            assert_eq!(vb_e, vb_c, "seed {seed}: soft_clamp root value diverged");
            assert_eq!(vb_e, vb_t, "seed {seed}: argmax_ties root value diverged");
            // Same tied-best child set in every mode.
            let bestset = |r: &ExactSolveResult, vb: f64| -> Vec<u16> {
                let mut v: Vec<u16> = r
                    .child_values
                    .iter()
                    .filter(|&&(_, cv)| cv == vb)
                    .map(|&(i, _)| i)
                    .collect();
                v.sort_unstable();
                v
            };
            assert_eq!(bestset(&exact, vb_e), bestset(&clamp, vb_c), "seed {seed}");
            assert_eq!(bestset(&exact, vb_e), bestset(&ties, vb_t), "seed {seed}");
            // Clamp error is one-sided: recorded >= exact for the maximiser's
            // dominated children (mirrored for the minimiser).
            let exact_map: HashMap<u16, f64> = exact.child_values.iter().copied().collect();
            for &(idx, v) in &clamp.child_values {
                let ev = exact_map[&idx];
                if actor == 0 {
                    assert!(v >= ev - 1e-12, "seed {seed}: clamp lowered child {idx}");
                } else {
                    assert!(v <= ev + 1e-12, "seed {seed}: clamp lowered child {idx}");
                }
            }
            // ArgmaxTies label: uniform over ties, nothing else.
            let (pidx, pval, _legal) = ties.policy_target(actor);
            let nb = bestset(&ties, vb_t).len();
            assert_eq!(pidx.len(), nb, "seed {seed}: ties label size");
            for &w in &pval {
                assert!(
                    (w - 1.0 / nb as f32).abs() < 1e-6,
                    "seed {seed}: not uniform"
                );
            }
            checked += 1;
        }
        assert!(checked >= 5, "too few endgame roots reached ({checked})");
        Ok(())
    }

    /// SoftClamp label guarantees: same argmax move as the Exact label; all
    /// clamped children collapse to one shared weight that never exceeds any
    /// exact-band child's weight.
    #[test]
    fn soft_clamp_label_argmax_and_tail_shape() -> PyResult<()> {
        let mode = SolverOrderMode::Lookahead2Clustered;
        let mut checked_with_clamp = 0;
        for seed in 0..24u64 {
            let Some(root) = first_endgame_root(seed) else {
                continue;
            };
            let actor = root.actor()?;
            let solve = |pm: ExactPolicyMode| -> PyResult<ExactSolveResult> {
                Ok(
                    solve_root_exact(&root, far_deadline(), 160.0, 2.0, 0.5, mode, pm, 10.0)?
                        .expect("root solve"),
                )
            };
            let clamp = solve(ExactPolicyMode::SoftClamp)?;
            if clamp.n_clamped == 0 {
                continue;
            }
            checked_with_clamp += 1;
            let exact = solve(ExactPolicyMode::Exact)?;
            let (ci, cv, _) = clamp.policy_target(actor);
            let (ei, ev, _) = exact.policy_target(actor);
            let argmax = |idx: &[i32], val: &[f32]| -> i32 {
                idx.iter()
                    .zip(val)
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(&i, _)| i)
                    .unwrap()
            };
            assert_eq!(
                argmax(&ci, &cv),
                argmax(&ei, &ev),
                "seed {seed}: clamped label argmax diverged from exact"
            );
            // Clamped children all sit at the recorded worst → equal weights,
            // and no clamped child outweighs the best move.
            let exact_map: HashMap<u16, f64> = exact.child_values.iter().copied().collect();
            let clamp_map: HashMap<u16, f64> = clamp.child_values.iter().copied().collect();
            // Children whose recorded value differs from the exact one are
            // necessarily clamped; the converse can fail (a fail-low child
            // whose true value happens to EQUAL the clamp value is counted in
            // n_clamped but invisible here), so subset — not equality.
            let clamped_ids: Vec<u16> = clamp
                .child_values
                .iter()
                .filter(|&&(idx, v)| v != exact_map[&idx])
                .map(|&(idx, _)| idx)
                .collect();
            assert!(
                clamped_ids.len() <= clamp.n_clamped as usize,
                "seed {seed}: {} value-diffs > n_clamped {}",
                clamped_ids.len(),
                clamp.n_clamped
            );
            let cw_map: HashMap<i32, f32> = ci.iter().copied().zip(cv.iter().copied()).collect();
            let max_w = cv.iter().cloned().fold(f32::MIN, f32::max);
            let mut clamped_w: Option<f32> = None;
            for &cid in &clamped_ids {
                // All clamped children share one recorded value → one weight.
                let w = cw_map.get(&(cid as i32)).copied().unwrap_or(0.0);
                if let Some(prev) = clamped_w {
                    assert!(
                        (w - prev).abs() < 1e-6,
                        "seed {seed}: clamped weights differ"
                    );
                } else {
                    clamped_w = Some(w);
                }
                assert!(
                    w <= max_w + 1e-6,
                    "seed {seed}: clamped child outweighs best"
                );
                let same_val = clamp_map[&cid];
                let _ = same_val;
            }
        }
        assert!(
            checked_with_clamp >= 2,
            "too few clamped roots exercised ({checked_with_clamp})"
        );
        Ok(())
    }
}

#[cfg(test)]
mod exact_policy_tests {
    use super::exact_policy_target;

    fn argmax(policy_idx: &[i32], policy_val: &[f32]) -> i32 {
        let mut best_i = 0usize;
        for i in 1..policy_val.len() {
            if policy_val[i] > policy_val[best_i] {
                best_i = i;
            }
        }
        policy_idx[best_i]
    }

    #[test]
    fn peaks_on_best_for_maximiser() {
        // actor 0 maximises: best child is the highest value (idx 10).
        let cv = vec![(10u16, 0.9f64), (20, 0.1), (30, -0.5)];
        let (pidx, pval, lidx) = exact_policy_target(&cv, 0);
        assert_eq!(lidx, vec![10, 20, 30]);
        let sum: f32 = pval.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "policy must sum to 1, got {sum}");
        assert_eq!(argmax(&pidx, &pval), 10);
    }

    #[test]
    fn peaks_on_best_for_minimiser() {
        // actor 1 minimises: best child is the lowest value (idx 30).
        let cv = vec![(10u16, 0.9f64), (20, 0.1), (30, -0.5)];
        let (pidx, pval, _lidx) = exact_policy_target(&cv, 1);
        let sum: f32 = pval.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert_eq!(argmax(&pidx, &pval), 30);
    }

    #[test]
    fn uniform_on_ties() {
        let cv = vec![(10u16, 0.3f64), (20, 0.3), (30, 0.3)];
        let (_pidx, pval, _lidx) = exact_policy_target(&cv, 0);
        let sum: f32 = pval.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        for &w in &pval {
            assert!((w - 1.0 / 3.0).abs() < 1e-5, "expected uniform, got {w}");
        }
    }
}

/// Pick a root child by visit count.  τ=0 → argmax (ties → lowest joint index,
/// since children are ascending), matching the Python greedy select; τ>0 →
/// sample ∝ visit^(1/τ) using the slot RNG.
#[cfg(test)]
mod exact_retry_tests {
    use super::*;

    fn deck4_slot(actor_index: usize) -> SearchSlot {
        let mut state = new_game(0, true, true);
        state.phase = PLACE_AND_SELECT;
        state.deck = vec![41, 42, 43, 44];
        state.current_row = vec![1, 2, 3, 4];
        state.pending_claims = vec![(0, 5), (1, 6), (0, 7), (1, 8)];
        state.next_claims.clear();
        state.actor_index = actor_index;
        SearchSlot::new_for_game(state, 0, true, None)
    }

    #[test]
    fn deck4_timeout_allows_one_after_two_moves_retry_and_deck0() {
        let mut slot = deck4_slot(0);
        assert!(should_enter_exact_solving(true, &slot));

        mark_exact_timeout(&mut slot);
        assert!(!should_enter_exact_solving(true, &slot));

        slot.real_state.actor_index = 2;
        assert!(should_enter_exact_solving(true, &slot));

        mark_exact_timeout(&mut slot);
        assert!(slot.exact_deck4_after_two_retry_used);
        assert!(!should_enter_exact_solving(true, &slot));

        slot.real_state.deck.clear();
        slot.real_state.actor_index = 0;
        assert!(should_enter_exact_solving(true, &slot));
    }
}

fn select_from_visits(arena: &[Node], temp: f64, rng: &mut StdRng) -> u16 {
    let children = &arena[0].children;
    if temp <= 1e-6 {
        let mut best_v = -1i32;
        let mut best_idx = 0u16;
        for &(idx, cid) in children {
            let v = arena[cid as usize].visit_count;
            if v > best_v {
                best_v = v;
                best_idx = idx;
            }
        }
        best_idx
    } else {
        let weights: Vec<f64> = children
            .iter()
            .map(|&(_, cid)| (arena[cid as usize].visit_count as f64).powf(1.0 / temp))
            .collect();
        let sum: f64 = weights.iter().sum();
        if sum <= 0.0 {
            // Degenerate: every child unvisited / zero weight.  Should not occur
            // after n_sims > 0; fall back to the first child by prior order.
            debug_assert!(false, "select_from_visits: sum of weights is zero");
            return children[0].0;
        }
        let mut r = rng.r#gen::<f64>() * sum;
        for (k, &(idx, _)) in children.iter().enumerate() {
            r -= weights[k];
            if r <= 0.0 {
                return idx;
            }
        }
        children.last().map(|t| t.0).unwrap_or(0)
    }
}

/// A single eval leaf in this tick's batch: its node, its batch row, the acting
/// player (for value framing), and its legal actions (for expansion).
struct EvalLeaf {
    leaf: u32,
    row: usize,
    actor: u8,
    legal: Vec<(u16, Option<(i8, i8, i8, i8, bool)>, Option<u16>)>,
    // Present only for one row in an atomic deck=8 exhaustive chance panel.
    // `leaf` is then the public observation node for this reveal.
    chance_bootstrap: Option<ChanceBootstrapEval>,
}

#[derive(Clone, Copy)]
struct ChanceBootstrapEval {
    chance_node: u32,
    probability: f64,
}

/// Per-slot bookkeeping for one tick, set by step() and consumed by update().
struct SlotTick {
    slot: usize,
    is_root: bool,
    paths: Vec<Vec<u32>>, // Searching: the chunk's descents (Root: empty)
    evals: Vec<EvalLeaf>, // unique non-terminal leaves contributed to the batch
    // Open-loop only (empty for closed-loop): per-path actor lists (for VL undo,
    // since nodes are stateless) and per-path leaf concrete states (for terminal
    // value + terminal detection in update, since the leaf node stores no state).
    ol_actors: Vec<Vec<u8>>,
    ol_leaf_states: Vec<RustGameState>,
    // Open-loop Searching only: eval_path_indices[ei] = pi means evals[ei] is the
    // leaf of paths[pi].  With Issue-1 de-dup removed, multiple evals may share an
    // OLNode id; this lets update() back up each path with its OWN eval value.
    // Empty for closed-loop and root-eval ticks.
    eval_path_indices: Vec<usize>,
    // One entry per simulation path.  Some((path index of the chance node,
    // outcome probability)) activates probability-aware chance backup.
    chance_steps: Vec<Option<(usize, f64)>>,
    // Exhaustive bootstrap rows admitted in this tick.  These are charged to
    // the move budget in addition to ordinary simulation paths.
    chance_bootstrap_rows: usize,
}

struct SlotStepOutput {
    tick: SlotTick,
    mb_data: Vec<f32>,
    ob_data: Vec<f32>,
    flat_data: Vec<f32>,
    idxs_per: Vec<Vec<i64>>,
}

/// Batched MCTS: N synchronized games, one GPU forward per tick.
#[pyclass]
struct BatchedMCTS {
    slots: Vec<SearchSlot>,
    n_sims: usize,
    leaf_batch: usize,
    virtual_loss: i32,
    cpuct: f64,
    fpu: f64,
    dirichlet_alpha: f64,
    dirichlet_eps: f64,
    temp_moves: usize,
    playout_cap_randomization: bool,
    full_search_fraction: f64,
    fast_move_sims: usize,
    record_fast_moves: bool,
    fast_move_dirichlet_eps: f64,
    fast_move_temp_moves: usize,
    // Asymmetric two-net (HOF) games: shallow no-record profile for one seat.
    seat_override: Option<SeatSearchOverride>,
    // Run9 diversity: uniformly-random unrecorded opening plies (see
    // RandomOpening). None = every game starts from the standard deal.
    random_opening: Option<RandomOpening>,
    // Run10: pick-group visit floors at shallow non-root nodes (see PickFloor).
    pick_floor: Option<PickFloor>,
    // Terminal-value formula params (Fix 1): GAME_OVER backup uses
    // terminal_search_value with these, matching the non-terminal leaf scale.
    score_scale: f64,
    margin_gain: f64,
    alpha: f64,
    harmony: bool,
    middle_kingdom: bool,
    next_seed: u64,
    games_started: usize,
    games_target: usize,
    pending: Vec<SlotTick>,
    open_loop: bool,
    // Opt-in production vertical slice: at roots with exactly eight hidden
    // dominoes, enumerate all C(8,4)=70 next public rows at each admitted
    // stochastic action node and use their exact probability mean for Q.
    deck8_chance_enumeration: bool,
    // Evaluation-only seat filter: None applies the mode to both seats; Some
    // applies it only when that player owns the current root search.
    deck8_chance_enumeration_seat: Option<u8>,
    cum_deck8_chance_panel_count: u64,
    cum_deck8_chance_bootstrap_rows: u64,
    cum_deck8_chance_budget_blocked_count: u64,
    // Cumulative open-loop diagnostics rolled in from finished slots (so the
    // Python-readable getters survive games being reset in their slots).
    cum_fallback_count: u64,
    cum_missing_child_count: u64,
    // Pick-floor diagnostic accumulators (see SearchSlot::pf_minshare_*).
    cum_pf_minshare_sum: f64,
    cum_pf_minshare_count: u64,
    // Exact endgame solver (deck ∈ {0,4} roots). Per-position wall-clock budget
    // in seconds; <= 0.0 disables it (ablation).
    exact_endgame_max_secs: f64,
    // How exact roots price dominated children for the policy label (see
    // ExactPolicyMode). Root value + chosen move are exact in every mode.
    exact_policy_mode: ExactPolicyMode,
    // SoftClamp threshold in solver-utility points (raw margin except at an
    // official score tiebreak); dominated children are recorded at the clamp.
    exact_clamp_delta: f64,
    cum_exact_solve_count: u64,      // root moves solved exactly
    cum_exact_tree_solve_count: u64, // expensive exact continuation plans built
    cum_exact_cache_hit_count: u64,  // exact moves served from a precomputed plan
    cum_exact_fallback_count: u64,   // budget exceeded → fell back to MCTS (≈0)
    cum_exact_attempt_deck4_initial_count: u64,
    cum_exact_attempt_deck4_retry_count: u64,
    cum_exact_attempt_deck0_count: u64,
    cum_exact_fallback_deck4_initial_count: u64,
    cum_exact_fallback_deck4_retry_count: u64,
    cum_exact_fallback_deck0_count: u64,
    cum_exact_solver_secs: f64, // wall time spent in resolve_exact_slots (solve + finalize)
    exact_fallback_records: Vec<ExactFallbackRecord>,
    cum_fast_move_count: u64,
    cum_full_move_count: u64,
    cum_recorded_fast_move_count: u64,
    cum_recorded_full_move_count: u64,
    cum_exact_recorded_move_count: u64,
    // Games finished entirely inside resolve_exact_slots during step(); drained
    // by update() into the finished-games list it returns.
    pending_exact: Vec<FinishedGame>,
    // Async endgame solver (Step 1.5). When async_solve, step() dispatches
    // ExactSolving slots to the background thread and harvests results instead of
    // solving synchronously; inflight_solves counts dispatched-not-yet-harvested.
    async_solve: bool,
    inflight_solves: usize,
    // Mutex-wrapped so BatchedMCTS stays Sync (pyclass requirement); access is
    // GIL-serialized on the main thread, so the lock is always uncontended.
    job_tx: std::sync::Mutex<std::sync::mpsc::Sender<SolveJob>>,
    out_rx: std::sync::Mutex<std::sync::mpsc::Receiver<SolveOutcome>>,
    _solver_handle: std::thread::JoinHandle<()>,
}

struct ExactFallbackRecord {
    kind: ExactAttemptKind,
    game_seed: u64,
    move_num: usize,
    max_secs: f64,
    solve_secs: f64,
    state: RustGameState,
}

impl BatchedMCTS {
    fn note_exact_attempt(&mut self, kind: ExactAttemptKind) {
        match kind {
            ExactAttemptKind::Deck4Initial => {
                self.cum_exact_attempt_deck4_initial_count += 1;
            }
            ExactAttemptKind::Deck4Retry => {
                self.cum_exact_attempt_deck4_retry_count += 1;
            }
            ExactAttemptKind::Deck0 => {
                self.cum_exact_attempt_deck0_count += 1;
            }
        }
    }

    fn note_exact_fallback(&mut self, kind: ExactAttemptKind) {
        self.cum_exact_fallback_count += 1;
        match kind {
            ExactAttemptKind::Deck4Initial => {
                self.cum_exact_fallback_deck4_initial_count += 1;
            }
            ExactAttemptKind::Deck4Retry => {
                self.cum_exact_fallback_deck4_retry_count += 1;
            }
            ExactAttemptKind::Deck0 => {
                self.cum_exact_fallback_deck0_count += 1;
            }
        }
    }

    fn record_exact_fallback(
        &mut self,
        kind: ExactAttemptKind,
        state: RustGameState,
        game_seed: u64,
        move_num: usize,
        solve_secs: f64,
    ) {
        self.exact_fallback_records.push(ExactFallbackRecord {
            kind,
            game_seed,
            move_num,
            max_secs: self.exact_endgame_max_secs,
            solve_secs,
            state,
        });
    }

    fn absorb_slot_move_counts(&mut self, si: usize) {
        self.cum_fast_move_count += self.slots[si].fast_move_count as u64;
        self.cum_full_move_count += self.slots[si].full_move_count as u64;
        self.cum_recorded_fast_move_count += self.slots[si].recorded_fast_move_count as u64;
        self.cum_recorded_full_move_count += self.slots[si].recorded_full_move_count as u64;
        self.cum_exact_recorded_move_count += self.slots[si].exact_recorded_move_count as u64;
        self.cum_pf_minshare_sum += self.slots[si].pf_minshare_sum;
        self.cum_pf_minshare_count += self.slots[si].pf_minshare_count as u64;
        self.cum_deck8_chance_panel_count += self.slots[si].deck8_chance_panel_count as u64;
        self.cum_deck8_chance_bootstrap_rows += self.slots[si].deck8_chance_bootstrap_rows as u64;
        self.cum_deck8_chance_budget_blocked_count +=
            self.slots[si].deck8_chance_budget_blocked_count as u64;
    }

    /// Async path (Step 1.5): drain every completed background solve and apply it.
    fn harvest_solves(&mut self) {
        loop {
            // Scoped so the lock guard drops before apply_outcome (needs &mut self).
            let outcome = match self.out_rx.lock().unwrap().try_recv() {
                Ok(o) => o,
                Err(_) => break,
            };
            self.apply_outcome(outcome);
        }
    }

    /// Apply one harvested outcome. Finished → record the game + recycle the slot
    /// (overbooking: a freed slot starts the next game). Fallback → resume MCTS in
    /// the SAME slot, which kept its game state while the solve was in flight.
    fn apply_outcome(&mut self, outcome: SolveOutcome) {
        let si = outcome.slot_idx;
        self.inflight_solves = self.inflight_solves.saturating_sub(1);
        let solve_secs = outcome.solve_secs;
        let fallback_state = outcome.fallback_state;
        let game_seed = outcome.game_seed;
        let move_num = outcome.move_num;
        self.cum_exact_solver_secs += solve_secs;
        let open_loop = self.open_loop;
        self.note_exact_attempt(outcome.kind);
        match outcome.result {
            SolveResult::Finished(fg) => {
                self.cum_exact_tree_solve_count += 1;
                self.cum_exact_solve_count += outcome.n_solved;
                self.cum_exact_cache_hit_count += outcome.n_solved.saturating_sub(1);
                self.cum_exact_recorded_move_count += outcome.n_solved;
                self.cum_fallback_count += self.slots[si].fallback_count as u64;
                self.cum_missing_child_count += self.slots[si].missing_child_count as u64;
                self.absorb_slot_move_counts(si);
                self.pending_exact.push(fg);
                // Recycle the slot: next game if quota remains, else Idle.
                if self.games_started < self.games_target {
                    let ns = self.next_seed;
                    self.next_seed += 1;
                    self.games_started += 1;
                    let mut slot = SearchSlot::new_for_game(
                        new_game(ns, self.harmony, self.middle_kingdom),
                        ns,
                        open_loop,
                        self.random_opening,
                    );
                    slot.choose_move_profile(
                        self.playout_cap_randomization,
                        self.full_search_fraction,
                        self.n_sims,
                        self.fast_move_sims,
                        self.record_fast_moves,
                        self.dirichlet_eps,
                        self.fast_move_dirichlet_eps,
                        self.temp_moves,
                        self.fast_move_temp_moves,
                        self.seat_override,
                    );
                    self.slots[si] = slot;
                } else {
                    self.slots[si].state = SlotState::Idle;
                    self.slots[si].fallback_count = 0;
                    self.slots[si].missing_child_count = 0;
                    self.slots[si].deck8_chance_panel_count = 0;
                    self.slots[si].deck8_chance_bootstrap_rows = 0;
                    self.slots[si].deck8_chance_budget_blocked_count = 0;
                }
            }
            SolveResult::Fallback => {
                self.note_exact_fallback(outcome.kind);
                if let Some(state) = fallback_state {
                    self.record_exact_fallback(
                        outcome.kind,
                        state,
                        game_seed,
                        move_num,
                        solve_secs,
                    );
                }
                let slot = &mut self.slots[si];
                mark_exact_timeout(slot);
                slot.exact_result = None;
                slot.exact_plan.clear();
                slot.state = SlotState::NeedsRootEval;
                slot.sims_done = 0;
                slot.deck8_chance_panel_admitted_this_move = false;
                if open_loop {
                    slot.ol_arena.clear();
                    slot.ol_arena.push(OLNode::new(1.0, (None, None)));
                } else {
                    let root_state = slot
                        .real_state
                        .redeterminize(Some(det_seed(slot.game_seed, slot.move_num)));
                    slot.arena.clear();
                    slot.arena.push(Node::new(1.0, (None, None)));
                    slot.arena[0].state = Some(root_state);
                }
            }
        }
    }

    /// Async path: dispatch every `ExactSolving` slot to the background solver,
    /// cloning a snapshot so the slot keeps its game (for in-place fallback resume).
    fn dispatch_solves(&mut self) {
        for si in 0..self.slots.len() {
            if self.slots[si].state == SlotState::ExactSolving {
                let job = SolveJob {
                    slot_idx: si,
                    kind: exact_attempt_kind(&self.slots[si]),
                    state: self.slots[si].real_state.cloned(),
                    records: self.slots[si].records.clone(),
                    game_seed: self.slots[si].game_seed,
                    move_num: self.slots[si].move_num,
                };
                self.slots[si].state = SlotState::SolvingInBackground;
                self.inflight_solves += 1;
                let _ = self.job_tx.lock().unwrap().send(job);
            }
        }
    }

    /// Async path top-of-step: harvest, dispatch, and — if nothing is searchable
    /// but solves are in flight — block for outcomes so the loop drains in-flight
    /// solves at end-of-iteration instead of spinning on empty batches.
    fn pump_async_solves(&mut self) {
        self.harvest_solves();
        self.dispatch_solves();
        while self.inflight_solves > 0
            && !self
                .slots
                .iter()
                .any(|s| matches!(s.state, SlotState::Searching | SlotState::NeedsRootEval))
        {
            let outcome = match self.out_rx.lock().unwrap().recv() {
                Ok(o) => o,
                Err(_) => break,
            };
            self.apply_outcome(outcome);
            self.dispatch_solves();
        }
    }
}

#[pymethods]
impl BatchedMCTS {
    #[new]
    #[pyo3(signature = (n_slots, n_games, base_seed, n_sims, leaf_batch=6,
                        virtual_loss=1, cpuct=1.5, fpu=0.0, dirichlet_alpha=0.3,
                        dirichlet_eps=0.25, temp_moves=20, harmony=true,
                        middle_kingdom=true, open_loop=false,
                        score_scale=160.0, margin_gain=2.0, alpha=0.5,
                        exact_endgame_max_secs=3.0, async_solve=false, solver_cpus=0,
                        playout_cap_randomization=false, full_search_fraction=0.25,
                        fast_move_sims=100, record_fast_moves=false,
                        fast_move_dirichlet_eps=0.0, fast_move_temp_moves=0,
                        exact_policy_mode="argmax_ties", exact_clamp_delta=10.0,
                        hof_opponent_seat=-1, hof_opponent_sims=0,
                        hof_opponent_dirichlet_eps=0.0, hof_opponent_temp_moves=0,
                        random_opening_fraction=0.0, random_opening_plies_min=0,
                        random_opening_plies_max=0,
                        pick_floor_frac=0.0, pick_floor_depth=2,
                        deck8_chance_enumeration=false,
                        deck8_chance_enumeration_seat=-1))]
    fn new(
        n_slots: usize,
        n_games: usize,
        base_seed: u64,
        n_sims: usize,
        leaf_batch: usize,
        virtual_loss: i32,
        cpuct: f64,
        fpu: f64,
        dirichlet_alpha: f64,
        dirichlet_eps: f64,
        temp_moves: usize,
        harmony: bool,
        middle_kingdom: bool,
        open_loop: bool,
        score_scale: f64,
        margin_gain: f64,
        alpha: f64,
        exact_endgame_max_secs: f64,
        async_solve: bool,
        solver_cpus: usize,
        playout_cap_randomization: bool,
        full_search_fraction: f64,
        fast_move_sims: usize,
        record_fast_moves: bool,
        fast_move_dirichlet_eps: f64,
        fast_move_temp_moves: usize,
        exact_policy_mode: &str,
        exact_clamp_delta: f64,
        hof_opponent_seat: i64,
        hof_opponent_sims: usize,
        hof_opponent_dirichlet_eps: f64,
        hof_opponent_temp_moves: usize,
        random_opening_fraction: f64,
        random_opening_plies_min: usize,
        random_opening_plies_max: usize,
        pick_floor_frac: f64,
        pick_floor_depth: usize,
        deck8_chance_enumeration: bool,
        deck8_chance_enumeration_seat: i64,
    ) -> Self {
        let exact_policy_mode = ExactPolicyMode::from_str(exact_policy_mode)
            .expect("BatchedMCTS: invalid exact_policy_mode");
        // Asymmetric two-net games: seat >= 0 pins that seat (the frozen HOF
        // opponent) to a fixed shallow search that records no examples, while
        // the other seat keeps the full profile above (incl. playout cap).
        assert!(
            hof_opponent_seat < 2,
            "BatchedMCTS: hof_opponent_seat must be -1 (off), 0, or 1, got {}",
            hof_opponent_seat
        );
        let seat_override = if hof_opponent_seat >= 0 {
            Some(SeatSearchOverride {
                seat: hof_opponent_seat as u8,
                sims: if hof_opponent_sims > 0 {
                    hof_opponent_sims
                } else {
                    n_sims
                },
                dirichlet_eps: hof_opponent_dirichlet_eps,
                temp_moves: hof_opponent_temp_moves,
            })
        } else {
            None
        };
        assert!(
            (0.0..=1.0).contains(&random_opening_fraction),
            "BatchedMCTS: random_opening_fraction must be in [0, 1], got {}",
            random_opening_fraction
        );
        assert!(
            random_opening_plies_min <= random_opening_plies_max,
            "BatchedMCTS: random_opening_plies_min {} > max {}",
            random_opening_plies_min,
            random_opening_plies_max
        );
        let random_opening = if random_opening_fraction > 0.0 && random_opening_plies_max > 0 {
            Some(RandomOpening {
                fraction: random_opening_fraction,
                min_plies: random_opening_plies_min,
                max_plies: random_opening_plies_max,
            })
        } else {
            None
        };
        // A floor of 0.2+ per group with 5 groups would force uniform picks;
        // stay well under 1/5 so PUCT retains most of the budget on merit.
        assert!(
            (0.0..0.2).contains(&pick_floor_frac),
            "BatchedMCTS: pick_floor_frac must be in [0, 0.2), got {}",
            pick_floor_frac
        );
        let pick_floor = if pick_floor_frac > 0.0 && pick_floor_depth > 0 {
            Some(PickFloor {
                frac: pick_floor_frac,
                // Training NEVER floors the root: root visits become the policy
                // target.  Pinned explicitly so the advisor's min_depth=0 option
                // cannot leak into training via a shared default.
                min_depth: 1,
                max_depth: pick_floor_depth,
                min_visits: 16,
            })
        } else {
            None
        };
        assert!(
            exact_clamp_delta > 0.0,
            "BatchedMCTS: exact_clamp_delta must be > 0, got {}",
            exact_clamp_delta
        );
        // Misconfigured callers: cheap one-time hard checks (assert!, not
        // debug_assert!) at construction so a bad config fails loudly up front
        // rather than producing degenerate searches or div-by-zero later.
        assert!(
            n_sims > 0,
            "BatchedMCTS: n_sims must be > 0, got {}",
            n_sims
        );
        assert!(
            n_slots > 0,
            "BatchedMCTS: n_slots must be > 0, got {}",
            n_slots
        );
        assert!(
            leaf_batch > 0,
            "BatchedMCTS: leaf_batch must be > 0, got {}",
            leaf_batch
        );
        assert!(
            !deck8_chance_enumeration || open_loop,
            "BatchedMCTS: deck8_chance_enumeration requires open_loop=true"
        );
        assert!(
            (-1..=1).contains(&deck8_chance_enumeration_seat),
            "BatchedMCTS: deck8_chance_enumeration_seat must be -1 (both), 0, or 1, got {}",
            deck8_chance_enumeration_seat
        );
        let deck8_chance_enumeration_seat =
            (deck8_chance_enumeration_seat >= 0).then_some(deck8_chance_enumeration_seat as u8);
        let mut slots = Vec::with_capacity(n_slots);
        let mut games_started = 0usize;
        for _ in 0..n_slots {
            if games_started < n_games {
                let seed = base_seed + games_started as u64;
                let mut slot = SearchSlot::new_for_game(
                    new_game(seed, harmony, middle_kingdom),
                    seed,
                    open_loop,
                    random_opening,
                );
                slot.choose_move_profile(
                    playout_cap_randomization,
                    full_search_fraction,
                    n_sims,
                    fast_move_sims,
                    record_fast_moves,
                    dirichlet_eps,
                    fast_move_dirichlet_eps,
                    temp_moves,
                    fast_move_temp_moves,
                    seat_override,
                );
                slots.push(slot);
                games_started += 1;
            } else {
                slots.push(SearchSlot::idle(harmony, middle_kingdom));
            }
        }
        // Background endgame solver (used only when async_solve). Always spawned —
        // it just blocks on an empty channel otherwise — and exits when this
        // BatchedMCTS is dropped (job_tx closes).
        let (job_tx, job_rx) = std::sync::mpsc::channel::<SolveJob>();
        let (out_tx, out_rx) = std::sync::mpsc::channel::<SolveOutcome>();
        // Dedicated solver pool. solver_cpus = threads for solving; generation gets
        // the rest via the global Rayon pool. 0 => auto (half of available threads).
        let solver_threads = if solver_cpus > 0 {
            solver_cpus
        } else {
            (std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                / 2)
            .max(1)
        };
        let solver_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(solver_threads)
            .thread_name(|i| format!("endgame-solver-{i}"))
            .build()
            .expect("failed to build the dedicated endgame solver pool");
        let solver_handle = spawn_endgame_solver(
            job_rx,
            out_tx,
            exact_endgame_max_secs,
            score_scale,
            margin_gain,
            alpha,
            exact_policy_mode,
            exact_clamp_delta,
            solver_pool,
        );
        BatchedMCTS {
            slots,
            n_sims,
            leaf_batch: leaf_batch.max(1),
            virtual_loss,
            cpuct,
            fpu,
            dirichlet_alpha,
            dirichlet_eps,
            temp_moves,
            playout_cap_randomization,
            full_search_fraction,
            fast_move_sims: fast_move_sims.max(1),
            record_fast_moves,
            fast_move_dirichlet_eps,
            fast_move_temp_moves,
            seat_override,
            random_opening,
            pick_floor,
            score_scale,
            margin_gain,
            alpha,
            harmony,
            middle_kingdom,
            next_seed: base_seed + games_started as u64,
            games_started,
            games_target: n_games,
            pending: Vec::new(),
            open_loop,
            deck8_chance_enumeration,
            deck8_chance_enumeration_seat,
            cum_deck8_chance_panel_count: 0,
            cum_deck8_chance_bootstrap_rows: 0,
            cum_deck8_chance_budget_blocked_count: 0,
            cum_fallback_count: 0,
            cum_missing_child_count: 0,
            cum_pf_minshare_sum: 0.0,
            cum_pf_minshare_count: 0,
            exact_endgame_max_secs,
            exact_policy_mode,
            exact_clamp_delta,
            cum_exact_solve_count: 0,
            cum_exact_tree_solve_count: 0,
            cum_exact_cache_hit_count: 0,
            cum_exact_fallback_count: 0,
            cum_exact_attempt_deck4_initial_count: 0,
            cum_exact_attempt_deck4_retry_count: 0,
            cum_exact_attempt_deck0_count: 0,
            cum_exact_fallback_deck4_initial_count: 0,
            cum_exact_fallback_deck4_retry_count: 0,
            cum_exact_fallback_deck0_count: 0,
            cum_exact_solver_secs: 0.0,
            exact_fallback_records: Vec::new(),
            cum_fast_move_count: 0,
            cum_full_move_count: 0,
            cum_recorded_fast_move_count: 0,
            cum_recorded_full_move_count: 0,
            cum_exact_recorded_move_count: 0,
            pending_exact: Vec::new(),
            async_solve,
            inflight_solves: 0,
            job_tx: std::sync::Mutex::new(job_tx),
            out_rx: std::sync::Mutex::new(out_rx),
            _solver_handle: solver_handle,
        }
    }

    /// Diagnostic: total root moves solved exactly by the endgame solver across
    /// the whole run (deck ∈ {0,4} positions resolved without GPU forwards).
    #[getter]
    fn exact_solve_count(&self) -> u64 {
        self.cum_exact_solve_count
    }

    /// Diagnostic: expensive exact continuation plans built. With plan reuse,
    /// this should be roughly one per game endgame rather than one per move.
    #[getter]
    fn exact_tree_solve_count(&self) -> u64 {
        self.cum_exact_tree_solve_count
    }

    /// Diagnostic: cumulative wall-clock seconds spent inside resolve_exact_slots
    /// (parallel plan build + serial finalize) across the whole run. The Step-1
    /// parallelism target metric — log the per-iteration delta to see solver time
    /// drop after the par_iter change.
    #[getter]
    fn exact_solver_secs(&self) -> f64 {
        self.cum_exact_solver_secs
    }

    /// Diagnostic: exact moves served from an already-built continuation plan.
    #[getter]
    fn exact_cache_hit_count(&self) -> u64 {
        self.cum_exact_cache_hit_count
    }

    /// Diagnostic: endgame roots where the node budget was exceeded and the slot
    /// fell back to GPU-backed MCTS. Expected to stay 0 at the default 15M budget.
    #[getter]
    fn exact_fallback_count(&self) -> u64 {
        self.cum_exact_fallback_count
    }

    #[getter]
    fn exact_attempt_deck4_initial_count(&self) -> u64 {
        self.cum_exact_attempt_deck4_initial_count
    }

    #[getter]
    fn exact_attempt_deck4_retry_count(&self) -> u64 {
        self.cum_exact_attempt_deck4_retry_count
    }

    #[getter]
    fn exact_attempt_deck0_count(&self) -> u64 {
        self.cum_exact_attempt_deck0_count
    }

    #[getter]
    fn exact_fallback_deck4_initial_count(&self) -> u64 {
        self.cum_exact_fallback_deck4_initial_count
    }

    #[getter]
    fn exact_fallback_deck4_retry_count(&self) -> u64 {
        self.cum_exact_fallback_deck4_retry_count
    }

    #[getter]
    fn exact_fallback_deck0_count(&self) -> u64 {
        self.cum_exact_fallback_deck0_count
    }

    /// Diagnostic: exact-solver fallback roots buffered for optional sidecar
    /// logging. These are not replay examples and are drained by Python after
    /// each self-play generation block.
    #[getter]
    fn exact_fallback_record_count(&self) -> usize {
        self.exact_fallback_records.len()
    }

    fn drain_exact_fallback_records(&mut self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let mut out = Vec::with_capacity(self.exact_fallback_records.len());
        for rec in self.exact_fallback_records.drain(..) {
            let state = rec.state;
            let dict = PyDict::new(py);
            dict.set_item("kind", exact_attempt_kind_label(rec.kind))?;
            dict.set_item("game_seed", rec.game_seed)?;
            dict.set_item("move_num", rec.move_num)?;
            dict.set_item("max_secs", rec.max_secs)?;
            dict.set_item("solve_secs", rec.solve_secs)?;
            dict.set_item("deck_len", state.deck.len())?;
            dict.set_item(
                "remaining_current_claims",
                deck4_remaining_current_claims(&state),
            )?;
            dict.set_item("phase", state.phase)?;
            dict.set_item("actor_index", state.actor_index)?;
            dict.set_item("initial_pick_count", state.initial_pick_count)?;
            dict.set_item("start_player", state.start_player)?;
            dict.set_item("harmony", state.harmony)?;
            dict.set_item("middle_kingdom", state.middle_kingdom)?;
            dict.set_item("deck", state.deck)?;
            dict.set_item("current_row", state.current_row)?;
            dict.set_item("pending_claims", state.pending_claims)?;
            dict.set_item("next_claims", state.next_claims)?;
            dict.set_item("board0_terrain", state.boards[0].terrain.to_vec())?;
            dict.set_item("board0_crowns", state.boards[0].crowns.to_vec())?;
            dict.set_item("board1_terrain", state.boards[1].terrain.to_vec())?;
            dict.set_item("board1_crowns", state.boards[1].crowns.to_vec())?;
            dict.set_item("castle_x", state.boards[0].castle_x)?;
            dict.set_item("castle_y", state.boards[0].castle_y)?;
            out.push(dict.into_any().unbind());
        }
        Ok(out)
    }

    /// Diagnostic: non-exact moves searched with the fast per-move sim cap.
    #[getter]
    fn fast_move_count(&self) -> u64 {
        self.cum_fast_move_count
            + self
                .slots
                .iter()
                .map(|s| s.fast_move_count as u64)
                .sum::<u64>()
    }

    /// Diagnostic: non-exact moves searched with the full per-move sim cap.
    #[getter]
    fn full_move_count(&self) -> u64 {
        self.cum_full_move_count
            + self
                .slots
                .iter()
                .map(|s| s.full_move_count as u64)
                .sum::<u64>()
    }

    /// Diagnostic: fast-search moves that were stored as training examples.
    #[getter]
    fn recorded_fast_move_count(&self) -> u64 {
        self.cum_recorded_fast_move_count
            + self
                .slots
                .iter()
                .map(|s| s.recorded_fast_move_count as u64)
                .sum::<u64>()
    }

    /// Diagnostic: full-search moves that were stored as training examples.
    #[getter]
    fn recorded_full_move_count(&self) -> u64 {
        self.cum_recorded_full_move_count
            + self
                .slots
                .iter()
                .map(|s| s.recorded_full_move_count as u64)
                .sum::<u64>()
    }

    /// Diagnostic: exact-solved moves stored as training examples.
    #[getter]
    fn exact_recorded_move_count(&self) -> u64 {
        self.cum_exact_recorded_move_count
            + self
                .slots
                .iter()
                .map(|s| s.exact_recorded_move_count as u64)
                .sum::<u64>()
    }

    /// Resolve every `ExactSolving` slot: solve the terminal-adjacent root exactly
    /// and finalize the move without any GPU forward, cascading through the whole
    /// endgame (deck only shrinks, so a slot stays `ExactSolving` until GAME_OVER).
    /// Finished games are stashed in `pending_exact` for `update()` to return.
    /// On budget exhaustion the slot falls back to `NeedsRootEval`.
    ///
    /// Slots are processed SERIALLY here on purpose: each solve uses the whole
    /// machine via within-solve (YBW) parallelism over root children in
    /// `solve_root_exact_cached`. Running internally-parallel solves concurrently
    /// would oversubscribe the cores and inflate every solve's wall time past the
    /// per-solve deadline (the cause of the high-fallback regression).
    fn resolve_exact_slots(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.exact_endgame_max_secs <= 0.0 {
            return Ok(());
        }
        let solver_t0 = std::time::Instant::now();
        // Solve with the GIL RELEASED. The whole inner body is pure Rust
        // (solve_exact_plan; finalize_move/encode_arrays return ndarray, not
        // PyArray; counters + recycle touch only Rust fields), so releasing the GIL
        // is sound and lets the OTHER double-buffer instance drive the GPU forward
        // concurrently while this one solves — the Step 1.5 overlap. Each solve still
        // uses the whole machine via the within-solve par_iter; slots are serial here.
        let result = py.detach(|| self.resolve_exact_slots_inner());
        self.cum_exact_solver_secs += solver_t0.elapsed().as_secs_f64();
        result
    }

    /// Pure-Rust body of `resolve_exact_slots`, run with the GIL released.
    fn resolve_exact_slots_inner(&mut self) -> PyResult<()> {
        let (score_scale, margin_gain, val_alpha) =
            (self.score_scale, self.margin_gain, self.alpha);
        let max_secs = self.exact_endgame_max_secs;
        let open_loop = self.open_loop;

        // Solve each ExactSolving slot's whole endgame, then play it out on the
        // slot's OWNED data via play_out_exact_endgame (the standalone unit the
        // async solver will also use). solve_exact_plan returns Some only when it
        // solves to GAME_OVER, so the playout always finishes the game. Finished
        // games are collected with their slot index so recycling (which needs
        // &mut self) happens after, like update() does.
        let mut finished: Vec<(usize, FinishedGame)> = Vec::new();
        for si in 0..self.slots.len() {
            if self.slots[si].state != SlotState::ExactSolving {
                continue;
            }
            let attempt_kind = exact_attempt_kind(&self.slots[si]);
            self.note_exact_attempt(attempt_kind);
            let solve_t0 = std::time::Instant::now();
            match solve_exact_plan(
                &self.slots[si].real_state,
                max_secs,
                score_scale,
                margin_gain,
                val_alpha,
                self.exact_policy_mode,
                self.exact_clamp_delta,
            )? {
                Some(plan) if !plan.is_empty() => {
                    // Accounting matches the old per-move loop: one tree solve per
                    // endgame, n total solved moves, n-1 served from the built plan.
                    let n = plan.len() as u64;
                    self.cum_exact_tree_solve_count += 1;
                    self.cum_exact_solve_count += n;
                    self.cum_exact_cache_hit_count += n.saturating_sub(1);
                    self.cum_exact_recorded_move_count += n;
                    let fg = play_out_exact_endgame(
                        self.slots[si].real_state.cloned(),
                        std::mem::take(&mut self.slots[si].records),
                        self.slots[si].game_seed,
                        plan,
                    )?;
                    finished.push((si, fg));
                }
                _ => {
                    // Deadline exceeded (or degenerate): suppress retrying the
                    // same full deck=4 root, but allow the cheap deck=0 solve
                    // and one later deck=4 retry once two current-row decisions
                    // have been fixed.
                    let solve_secs = solve_t0.elapsed().as_secs_f64();
                    self.note_exact_fallback(attempt_kind);
                    self.record_exact_fallback(
                        attempt_kind,
                        self.slots[si].real_state.cloned(),
                        self.slots[si].game_seed,
                        self.slots[si].move_num,
                        solve_secs,
                    );
                    mark_exact_timeout(&mut self.slots[si]);
                    self.slots[si].exact_result = None;
                    self.slots[si].exact_plan.clear();
                    self.slots[si].state = SlotState::NeedsRootEval;
                    self.slots[si].sims_done = 0;
                    self.slots[si].deck8_chance_panel_admitted_this_move = false;
                    if open_loop {
                        self.slots[si].ol_arena.clear();
                        self.slots[si].ol_arena.push(OLNode::new(1.0, (None, None)));
                    } else {
                        let root_state = self.slots[si].real_state.redeterminize(Some(det_seed(
                            self.slots[si].game_seed,
                            self.slots[si].move_num,
                        )));
                        self.slots[si].arena.clear();
                        self.slots[si].arena.push(Node::new(1.0, (None, None)));
                        self.slots[si].arena[0].state = Some(root_state);
                    }
                }
            }
        }

        // Phase 2: recycle finished slots (mirrors update()'s tail).
        for (si, fg) in finished {
            self.cum_fallback_count += self.slots[si].fallback_count as u64;
            self.cum_missing_child_count += self.slots[si].missing_child_count as u64;
            self.absorb_slot_move_counts(si);
            self.pending_exact.push(fg);
            if self.games_started < self.games_target {
                let ns = self.next_seed;
                self.next_seed += 1;
                self.games_started += 1;
                let mut slot = SearchSlot::new_for_game(
                    new_game(ns, self.harmony, self.middle_kingdom),
                    ns,
                    self.open_loop,
                    self.random_opening,
                );
                slot.choose_move_profile(
                    self.playout_cap_randomization,
                    self.full_search_fraction,
                    self.n_sims,
                    self.fast_move_sims,
                    self.record_fast_moves,
                    self.dirichlet_eps,
                    self.fast_move_dirichlet_eps,
                    self.temp_moves,
                    self.fast_move_temp_moves,
                    self.seat_override,
                );
                self.slots[si] = slot;
            } else {
                self.slots[si].state = SlotState::Idle;
                self.slots[si].fallback_count = 0;
                self.slots[si].missing_child_count = 0;
                self.slots[si].deck8_chance_panel_count = 0;
                self.slots[si].deck8_chance_bootstrap_rows = 0;
                self.slots[si].deck8_chance_budget_blocked_count = 0;
            }
        }
        Ok(())
    }

    /// Open-loop diagnostic: total deep-node fallbacks (a determinization reached
    /// an expanded node with NO stored child legal here) across the whole run —
    /// cumulative over finished games plus the in-progress slots.
    #[getter]
    fn fallback_count(&self) -> u64 {
        self.cum_fallback_count
            + self
                .slots
                .iter()
                .map(|s| s.fallback_count as u64)
                .sum::<u64>()
    }

    /// Open-loop diagnostic: total descents stopped to add newly-legal children
    /// (Issue 2: a later determinization had legal actions the original expansion
    /// never saw) across the whole run.  Cumulative, same accounting as above.
    #[getter]
    fn missing_child_count(&self) -> u64 {
        self.cum_missing_child_count
            + self
                .slots
                .iter()
                .map(|s| s.missing_child_count as u64)
                .sum::<u64>()
    }

    /// Complete C(8,4) panels atomically admitted into production searches.
    #[getter]
    fn deck8_chance_panel_count(&self) -> u64 {
        self.cum_deck8_chance_panel_count
            + self
                .slots
                .iter()
                .map(|slot| slot.deck8_chance_panel_count as u64)
                .sum::<u64>()
    }

    /// NN rows spent on admitted exhaustive deck=8 chance panels.
    #[getter]
    fn deck8_chance_bootstrap_rows(&self) -> u64 {
        self.cum_deck8_chance_bootstrap_rows
            + self
                .slots
                .iter()
                .map(|slot| slot.deck8_chance_bootstrap_rows as u64)
                .sum::<u64>()
    }

    /// Atomic-panel admission attempts blocked because fewer than 70 move-budget
    /// units remained. A sampled node can be counted again on a later wave.
    #[getter]
    fn deck8_chance_budget_blocked_count(&self) -> u64 {
        self.cum_deck8_chance_budget_blocked_count
            + self
                .slots
                .iter()
                .map(|slot| slot.deck8_chance_budget_blocked_count as u64)
                .sum::<u64>()
    }

    /// Pick-floor diagnostic: mean of the minimum pick-group visit share at the
    /// most-visited root child, over all full-search finalized moves so far.
    /// Baseline (floors off) shows starvation depth; with floors on it should
    /// approach pick_floor_frac. NaN-free: returns 0.0 before any sample.
    #[getter]
    fn pick_floor_min_share_mean(&self) -> f64 {
        let sum: f64 =
            self.cum_pf_minshare_sum + self.slots.iter().map(|s| s.pf_minshare_sum).sum::<f64>();
        let count: u64 = self.cum_pf_minshare_count
            + self
                .slots
                .iter()
                .map(|s| s.pf_minshare_count as u64)
                .sum::<u64>();
        if count == 0 { 0.0 } else { sum / count as f64 }
    }

    /// Sample count behind pick_floor_min_share_mean.
    #[getter]
    fn pick_floor_min_share_count(&self) -> u64 {
        self.cum_pf_minshare_count
            + self
                .slots
                .iter()
                .map(|s| s.pf_minshare_count as u64)
                .sum::<u64>()
    }

    /// True once every game is finished and all slots are Idle.
    fn done(&self) -> bool {
        self.slots.iter().all(|s| s.state == SlotState::Idle)
    }

    /// Diagnostic (open-loop): (visit_count, value_sum) of an OLNode in a slot's
    /// tree.  Used by the Issue-1 regression test to confirm two simulations that
    /// collide on the same node under different determinizations contribute their
    /// OWN distinct values to the backup (value_sum = v1+v2, not 2*v1).
    fn debug_ol_node(&self, slot: usize, node_id: usize) -> PyResult<(i32, f64)> {
        let s = self
            .slots
            .get(slot)
            .ok_or_else(|| PyValueError::new_err("slot out of range"))?;
        let n = s
            .ol_arena
            .get(node_id)
            .ok_or_else(|| PyValueError::new_err("node_id out of range"))?;
        Ok((n.visit_count, n.value_sum))
    }

    /// Diagnostic (open-loop): number of children of an OLNode in a slot's tree.
    fn debug_ol_n_children(&self, slot: usize, node_id: usize) -> PyResult<usize> {
        let s = self
            .slots
            .get(slot)
            .ok_or_else(|| PyValueError::new_err("slot out of range"))?;
        let n = s
            .ol_arena
            .get(node_id)
            .ok_or_else(|| PyValueError::new_err("node_id out of range"))?;
        Ok(n.children.len())
    }

    /// Value-sensitive production test hook: committed exhaustive panel nodes
    /// in one slot as (node_id, player-0 Q, real/virtual visit count, support).
    fn debug_ol_panel_values(&self, slot: usize) -> PyResult<Vec<(u32, f64, i32, usize)>> {
        let search_slot = self
            .slots
            .get(slot)
            .ok_or_else(|| PyValueError::new_err("slot out of range"))?;
        Ok(search_slot
            .ol_arena
            .iter()
            .enumerate()
            .filter(|(_, node)| {
                node.chance_backup == OLChanceBackup::PanelMean && !node.chance_children.is_empty()
            })
            .map(|(node_id, node)| {
                (
                    node_id as u32,
                    ol_chance_value_p0(&search_slot.ol_arena, node_id as u32),
                    node.visit_count,
                    node.chance_children.len(),
                )
            })
            .collect())
    }

    /// Number of slots still working on a game.
    fn n_active(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.state != SlotState::Idle)
            .count()
    }

    /// Actor id for each row returned by the most recent step().
    ///
    /// This is intentionally separate from step() so the original self-play API
    /// remains stable. Evaluation code with two different networks can call
    /// step(), then row_actors(), route rows by actor, and pass update() results
    /// back in the original row order.
    fn row_actors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let n_rows = self
            .pending
            .iter()
            .flat_map(|tick| tick.evals.iter())
            .map(|ev| ev.row)
            .max()
            .map(|row| row + 1)
            .unwrap_or(0);
        let mut actors = vec![0i64; n_rows];
        for tick in &self.pending {
            for ev in &tick.evals {
                actors[ev.row] = ev.actor as i64;
            }
        }
        actors.into_pyarray(py)
    }

    /// Actor id of the SEARCHER (root) for each row returned by the most recent
    /// step() — the player whose move the slot's search is deciding, i.e.
    /// `self.slots[tick.slot].real_state.actor()`.
    ///
    /// This differs from row_actors(), which reports the LEAF actor (the player
    /// to move at the evaluated leaf state — which alternates with tree depth).
    /// Two-network rating must route on THIS value (searcher-owns-network): when
    /// it is player 0's turn, player 0's net drives the *entire* MCTS search,
    /// evaluating every leaf (including player-1 nodes) — exactly the agent
    /// definition used in benchmark_vs_rust and in deployment. All leaves of a
    /// given slot share one searcher, so every row a slot contributes gets the
    /// same actor here.
    fn row_search_actors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let n_rows = self
            .pending
            .iter()
            .flat_map(|tick| tick.evals.iter())
            .map(|ev| ev.row)
            .max()
            .map(|row| row + 1)
            .unwrap_or(0);
        let mut actors = vec![0i64; n_rows];
        for tick in &self.pending {
            // real_state.actor() is deterministic on the public game state; on the
            // (unexpected) error path fall back to 0 so this getter stays infallible
            // like row_actors().
            let root_actor = self.slots[tick.slot].real_state.actor().unwrap_or(0) as i64;
            for ev in &tick.evals {
                actors[ev.row] = root_actor;
            }
        }
        actors.into_pyarray(py)
    }

    /// Phase 1 of a tick: descend every active slot, stack all non-terminal
    /// unique leaves into one batch, stash per-slot bookkeeping for update().
    /// Returns (mb (B,9,13,13), ob (B,9,13,13), flat (B,FLAT_SIZE), idxs_list[B]).
    fn step<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyList>,
    )> {
        // Resolve any terminal-adjacent endgame slots exactly first — they
        // contribute nothing to the GPU batch and may finish games (stashed in
        // pending_exact for update()). After this, no slot is ExactSolving.
        if self.async_solve {
            // Step 1.5: harvest completed background solves + dispatch new ones.
            // The solve itself overlaps the GPU eval that follows this step().
            self.pump_async_solves();
        } else {
            self.resolve_exact_slots(py)?;
        }

        let (
            fpu,
            cpuct,
            leaf_batch,
            vl,
            open_loop,
            pick_floor,
            deck8_chance_enumeration,
            deck8_chance_enumeration_seat,
        ) = (
            self.fpu,
            self.cpuct,
            self.leaf_batch,
            self.virtual_loss,
            self.open_loop,
            self.pick_floor,
            self.deck8_chance_enumeration,
            self.deck8_chance_enumeration_seat,
        );

        let slot_outputs: PyResult<Vec<SlotStepOutput>> = self
            .slots
            .par_iter_mut()
            .enumerate()
            .filter_map(|(si, slot)| match slot.state {
                // Idle never contributes; ExactSolving is resolved/dispatched above;
                // SolvingInBackground is out on the async solver (skip defensively).
                SlotState::Idle | SlotState::ExactSolving | SlotState::SolvingInBackground => None,
                _ => Some((si, slot)),
            })
            .map(|(si, slot)| -> PyResult<SlotStepOutput> {
                let mut mb_data: Vec<f32> = Vec::new();
                let mut ob_data: Vec<f32> = Vec::new();
                let mut flat_data: Vec<f32> = Vec::new();
                let mut idxs_per: Vec<Vec<i64>> = Vec::new();
                let mut row: usize = 0;

                let tick = if open_loop {
                    // ── Open-loop: stateless tree, fresh determinization per
                    // descent.  Encode leaf CONCRETE states (carried to update via
                    // SlotTick), not node-stored states.
                    match slot.state {
                        SlotState::Idle => unreachable!("idle slots were filtered"),
                        SlotState::ExactSolving | SlotState::SolvingInBackground => {
                            unreachable!(
                                "ExactSolving/SolvingInBackground filtered before the batch"
                            )
                        }
                        SlotState::NeedsRootEval => {
                            // Root is evaluated from the PUBLIC real_state (its
                            // legal joint indices are determinization-independent).
                            let state = &slot.real_state;
                            let actor = state.actor()?;
                            let legal = state.legal_actions_indexed();
                            // Encode the single root leaf directly into the (pre-
                            // sized) batch buffers at row 0 — no Array3/Array1
                            // intermediate.
                            let board_sz = N_BOARD_CH * OUT_N * OUT_N;
                            mb_data.resize(board_sz, 0.0);
                            ob_data.resize(board_sz, 0.0);
                            flat_data.resize(FLAT_SIZE, 0.0);
                            state.encode_arrays_into(
                                actor,
                                &mut mb_data,
                                &mut ob_data,
                                &mut flat_data,
                                row,
                            )?;
                            idxs_per.push(legal.iter().map(|t| t.0 as i64).collect());
                            // Single root eval leaf at row 0 (row is not reused).
                            let ev = EvalLeaf {
                                leaf: 0,
                                row,
                                actor,
                                legal,
                                chance_bootstrap: None,
                            };
                            SlotTick {
                                slot: si,
                                is_root: true,
                                paths: Vec::new(),
                                evals: vec![ev],
                                ol_actors: Vec::new(),
                                ol_leaf_states: Vec::new(),
                                eval_path_indices: Vec::new(),
                                chance_steps: Vec::new(),
                                chance_bootstrap_rows: 0,
                            }
                        }
                        SlotState::Searching => {
                            let remaining = slot.move_profile.target_sims - slot.sims_done;
                            let path_cap = leaf_batch.min(remaining);
                            let mut paths: Vec<Vec<u32>> = Vec::with_capacity(path_cap);
                            let mut ol_actors: Vec<Vec<u8>> = Vec::with_capacity(path_cap);
                            let mut ol_leaf_states: Vec<RustGameState> =
                                Vec::with_capacity(path_cap);
                            let mut chance_steps: Vec<Option<(usize, f64)>> =
                                Vec::with_capacity(path_cap);
                            let root_actor = slot.real_state.actor()?;
                            let chance_enabled_for_seat = deck8_chance_enumeration
                                && deck8_chance_enumeration_seat
                                    .map_or(true, |seat| seat == root_actor);
                            let panel =
                                if chance_enabled_for_seat && slot.real_state.deck.len() == 8 {
                                    ol_one_reveal_panel(
                                        &slot.real_state.deck,
                                        1,
                                        70,
                                        slot.game_seed
                                            ^ (slot.move_num as u64)
                                                .wrapping_mul(0xA1C0_77EC_7A11_5EED),
                                    )?
                                } else {
                                    Vec::new()
                                };
                            debug_assert!(panel.is_empty() || panel.len() == 70);
                            let chance_enabled = panel.len() == 70;
                            let mut bootstrap_requests: Vec<A1cAdmissionRequest> = Vec::new();
                            let mut considered_chance_nodes: HashSet<u32> = HashSet::new();
                            let mut reserved_bootstrap_rows = 0usize;
                            while paths.len() < path_cap
                                && slot.sims_done + paths.len() + reserved_bootstrap_rows
                                    < slot.move_profile.target_sims
                            {
                                let seed = slot.rng.r#gen::<u64>();
                                let det = if chance_enabled {
                                    // Hidden order remains unread until PUCT has
                                    // selected the action that triggers the reveal.
                                    slot.real_state.cloned()
                                } else {
                                    slot.real_state.redeterminize(Some(seed))
                                };
                                let chance_config = chance_enabled.then_some(OLChanceConfig {
                                    panel: panel.as_slice(),
                                    backup: OLChanceBackup::Sampled,
                                    traversal: OLChanceTraversal::Balanced,
                                    schedule_seed: slot.game_seed
                                        ^ (slot.move_num as u64)
                                            .wrapping_mul(0xBA1A_4CED_C1C1_E5E5),
                                    draw_seed: seed,
                                    a1c: None,
                                    bootstrap_full_panel: true,
                                });
                                let (path, actors, leaf_state, chance_step, a1c_admission_request) =
                                    ol_descend(
                                        &mut slot.ol_arena,
                                        0,
                                        det,
                                        fpu,
                                        cpuct, // det moved in (no clone)
                                        &mut slot.fallback_count,
                                        &mut slot.missing_child_count,
                                        pick_floor,
                                        chance_config,
                                    )?;
                                if let Some(request) = a1c_admission_request {
                                    if considered_chance_nodes.insert(request.chance_node_id) {
                                        // The first admitted action gets the exact
                                        // panel; siblings remain on unbiased sampled
                                        // backup. This bounds the midgame tax at 70
                                        // rows and makes the A/B attributable.
                                        if !slot.deck8_chance_panel_admitted_this_move
                                            && bootstrap_requests.is_empty()
                                        {
                                            let projected = slot.sims_done
                                                + paths.len()
                                                + 1
                                                + reserved_bootstrap_rows
                                                + panel.len();
                                            if projected <= slot.move_profile.target_sims {
                                                reserved_bootstrap_rows += panel.len();
                                                slot.deck8_chance_panel_admitted_this_move = true;
                                                bootstrap_requests.push(request);
                                            } else {
                                                slot.deck8_chance_budget_blocked_count += 1;
                                            }
                                        }
                                    }
                                }
                                ol_apply_virtual_loss(&mut slot.ol_arena, &path, &actors, 1, vl);
                                paths.push(path);
                                ol_actors.push(actors);
                                ol_leaf_states.push(leaf_state);
                                chance_steps.push(chance_step);
                            }
                            // Issue 1 fix: NO de-dup by OLNode id.  The same node
                            // can be reached by different determinizations whose
                            // concrete states (current_row, domino-in-hand, board)
                            // differ, so each non-terminal simulation gets its OWN
                            // eval row and its own concrete-state evaluation.
                            // eval_path_indices[ei] = pi maps each eval back to the
                            // simulation path it belongs to, so update() backs up
                            // each path with ITS value (not a colliding sim's).
                            let mut evals: Vec<EvalLeaf> = Vec::new();
                            let mut eval_path_indices: Vec<usize> = Vec::new();
                            // Pre-size the batch buffers to the non-terminal leaf
                            // count, then write each leaf's encoding DIRECTLY into
                            // its row — no per-leaf Array3/Array1 alloc + copy.
                            let board_sz = N_BOARD_CH * OUT_N * OUT_N;
                            let n_nonterm = ol_leaf_states
                                .iter()
                                .filter(|ls| ls.phase != GAME_OVER)
                                .count();
                            mb_data.resize(n_nonterm * board_sz, 0.0);
                            ob_data.resize(n_nonterm * board_sz, 0.0);
                            flat_data.resize(n_nonterm * FLAT_SIZE, 0.0);
                            for (pi, path) in paths.iter().enumerate() {
                                let ls = &ol_leaf_states[pi];
                                if ls.phase == GAME_OVER {
                                    continue;
                                }
                                let leaf = *path.last().unwrap();
                                let actor = ls.actor()?;
                                let legal = ls.legal_actions_indexed();
                                ls.encode_arrays_into(
                                    actor,
                                    &mut mb_data,
                                    &mut ob_data,
                                    &mut flat_data,
                                    row,
                                )?;
                                idxs_per.push(legal.iter().map(|t| t.0 as i64).collect());
                                evals.push(EvalLeaf {
                                    leaf,
                                    row,
                                    actor,
                                    legal,
                                    chance_bootstrap: None,
                                });
                                eval_path_indices.push(pi);
                                row += 1;
                            }
                            // Cross-node and cross-slot batching happens naturally:
                            // each accepted panel is appended to this SlotStepOutput,
                            // then the outer gather concatenates every slot before the
                            // single Python/GPU evaluator call.
                            for request in &bootstrap_requests {
                                for outcome in &panel {
                                    let row_tiles: [u16; 4] = outcome
                                        .row
                                        .as_slice()
                                        .try_into()
                                        .expect("deck=8 panel rows contain four tiles");
                                    let row_seed = request.draw_seed
                                        ^ row_tiles
                                            .iter()
                                            .fold(0xD8C8_C1E0_0000_0001u64, |hash, &tile| {
                                                hash.rotate_left(11) ^ tile as u64
                                            });
                                    let forced = redeterminize_with_first_row(
                                        &request.pre_reveal_state,
                                        &row_tiles,
                                        row_seed,
                                    )?;
                                    let revealed = forced.step(request.placement, request.pick)?;
                                    let actor = revealed.actor()?;
                                    let legal = revealed.legal_actions_indexed();
                                    let (observation_id, probability) =
                                        ol_route_chance_observation(
                                            &mut slot.ol_arena,
                                            request.chance_node_id,
                                            &revealed,
                                            actor,
                                        )?;
                                    mb_data.resize((row + 1) * board_sz, 0.0);
                                    ob_data.resize((row + 1) * board_sz, 0.0);
                                    flat_data.resize((row + 1) * FLAT_SIZE, 0.0);
                                    revealed.encode_arrays_into(
                                        actor,
                                        &mut mb_data,
                                        &mut ob_data,
                                        &mut flat_data,
                                        row,
                                    )?;
                                    idxs_per
                                        .push(legal.iter().map(|action| action.0 as i64).collect());
                                    evals.push(EvalLeaf {
                                        leaf: observation_id,
                                        row,
                                        actor,
                                        legal,
                                        chance_bootstrap: Some(ChanceBootstrapEval {
                                            chance_node: request.chance_node_id,
                                            probability,
                                        }),
                                    });
                                    row += 1;
                                }
                            }
                            SlotTick {
                                slot: si,
                                is_root: false,
                                paths,
                                evals,
                                ol_actors,
                                ol_leaf_states,
                                eval_path_indices,
                                chance_steps,
                                chance_bootstrap_rows: reserved_bootstrap_rows,
                            }
                        }
                    }
                } else {
                    let mut push_leaf = |arena: &Vec<Node>,
                                         leaf: u32,
                                         mb: &mut Vec<f32>,
                                         ob: &mut Vec<f32>,
                                         fl: &mut Vec<f32>,
                                         ix: &mut Vec<Vec<i64>>|
                     -> PyResult<EvalLeaf> {
                        let state = arena[leaf as usize].state.as_ref().unwrap();
                        let actor = state.actor()?;
                        let legal = state.legal_actions_indexed();
                        let (my, opp, flat) = state.encode_arrays(actor)?;
                        mb.extend_from_slice(my.as_slice().expect("contig"));
                        ob.extend_from_slice(opp.as_slice().expect("contig"));
                        fl.extend_from_slice(flat.as_slice().expect("contig"));
                        ix.push(legal.iter().map(|t| t.0 as i64).collect());
                        let ev = EvalLeaf {
                            leaf,
                            row,
                            actor,
                            legal,
                            chance_bootstrap: None,
                        };
                        row += 1;
                        Ok(ev)
                    };

                    match slot.state {
                        SlotState::Idle => unreachable!("idle slots were filtered"),
                        SlotState::ExactSolving | SlotState::SolvingInBackground => {
                            unreachable!(
                                "ExactSolving/SolvingInBackground filtered before the batch"
                            )
                        }
                        SlotState::NeedsRootEval => {
                            let ev = push_leaf(
                                &slot.arena,
                                0,
                                &mut mb_data,
                                &mut ob_data,
                                &mut flat_data,
                                &mut idxs_per,
                            )?;
                            SlotTick {
                                slot: si,
                                is_root: true,
                                paths: Vec::new(),
                                evals: vec![ev],
                                ol_actors: Vec::new(),
                                ol_leaf_states: Vec::new(),
                                eval_path_indices: Vec::new(),
                                chance_steps: Vec::new(),
                                chance_bootstrap_rows: 0,
                            }
                        }
                        SlotState::Searching => {
                            let chunk =
                                leaf_batch.min(slot.move_profile.target_sims - slot.sims_done);
                            let mut paths: Vec<Vec<u32>> = Vec::with_capacity(chunk);
                            for _ in 0..chunk {
                                let path = descend(&mut slot.arena, 0, fpu, cpuct)?;
                                apply_virtual_loss(&mut slot.arena, &path, 1, vl);
                                paths.push(path);
                            }
                            let mut evals: Vec<EvalLeaf> = Vec::new();
                            let mut seen: HashSet<u32> = HashSet::new();
                            for path in &paths {
                                let leaf = *path.last().unwrap();
                                if slot.arena[leaf as usize].state.as_ref().unwrap().phase
                                    == GAME_OVER
                                {
                                    continue;
                                }
                                if seen.insert(leaf) {
                                    let ev = push_leaf(
                                        &slot.arena,
                                        leaf,
                                        &mut mb_data,
                                        &mut ob_data,
                                        &mut flat_data,
                                        &mut idxs_per,
                                    )?;
                                    evals.push(ev);
                                }
                            }
                            SlotTick {
                                slot: si,
                                is_root: false,
                                paths,
                                evals,
                                ol_actors: Vec::new(),
                                ol_leaf_states: Vec::new(),
                                eval_path_indices: Vec::new(),
                                chance_steps: Vec::new(),
                                chance_bootstrap_rows: 0,
                            }
                        }
                    }
                };

                Ok(SlotStepOutput {
                    tick,
                    mb_data,
                    ob_data,
                    flat_data,
                    idxs_per,
                })
            })
            .collect();

        let mut slot_outputs = slot_outputs?;
        slot_outputs.sort_unstable_by_key(|out| out.tick.slot);

        let total_rows: usize = slot_outputs.iter().map(|out| out.idxs_per.len()).sum();
        let mut mb_data: Vec<f32> = Vec::with_capacity(total_rows * N_BOARD_CH * OUT_N * OUT_N);
        let mut ob_data: Vec<f32> = Vec::with_capacity(total_rows * N_BOARD_CH * OUT_N * OUT_N);
        let mut flat_data: Vec<f32> = Vec::with_capacity(total_rows * FLAT_SIZE);
        let mut idxs_per: Vec<Vec<i64>> = Vec::with_capacity(total_rows);
        let mut pending: Vec<SlotTick> = Vec::with_capacity(slot_outputs.len());
        let mut row: usize = 0;

        for mut out in slot_outputs {
            let offset = row;
            for ev in &mut out.tick.evals {
                ev.row += offset;
            }
            row += out.idxs_per.len();
            mb_data.extend(out.mb_data);
            ob_data.extend(out.ob_data);
            flat_data.extend(out.flat_data);
            idxs_per.extend(out.idxs_per);
            pending.push(out.tick);
        }

        self.pending = pending;
        let b = row;
        let mb = Array4::from_shape_vec((b, N_BOARD_CH, OUT_N, OUT_N), mb_data)
            .expect("mb shape")
            .into_pyarray(py);
        let ob = Array4::from_shape_vec((b, N_BOARD_CH, OUT_N, OUT_N), ob_data)
            .expect("ob shape")
            .into_pyarray(py);
        let flat = Array2::from_shape_vec((b, FLAT_SIZE), flat_data)
            .expect("flat shape")
            .into_pyarray(py);
        let idxs_items: Vec<_> = idxs_per.into_iter().map(|v| v.into_pyarray(py)).collect();
        let idxs_list = PyList::new(py, idxs_items)?;
        Ok((mb, ob, flat, idxs_list))
    }

    /// Phase 2 of a tick: scatter forward results, expand + back up every slot,
    /// advance state machines, recycle finished games.  Returns finished games as
    /// [(game_seed,
    ///   [(mb,ob,flat,pidx,pval,lidx,root_stats,z,own_score,opp_score,win_target,actor)],
    ///  (score0,score1,official_outcome0))]. `actor` (0/1) is the seat that made
    /// the move; used by the two-net HOF path to retain learner-searched moves.
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f32>,
        gathered: Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyList>> {
        let (
            n_sims,
            vl,
            temp_moves,
            alpha,
            dirichlet_eps,
            harmony,
            mk,
            open_loop,
            playout_cap_randomization,
            full_search_fraction,
            fast_move_sims,
            record_fast_moves,
            fast_move_dirichlet_eps,
            fast_move_temp_moves,
            seat_override,
            random_opening,
        ) = (
            self.n_sims,
            self.virtual_loss,
            self.temp_moves,
            self.dirichlet_alpha,
            self.dirichlet_eps,
            self.harmony,
            self.middle_kingdom,
            self.open_loop,
            self.playout_cap_randomization,
            self.full_search_fraction,
            self.fast_move_sims,
            self.record_fast_moves,
            self.fast_move_dirichlet_eps,
            self.fast_move_temp_moves,
            self.seat_override,
            self.random_opening,
        );
        // Terminal-value formula params (Fix 1).  `alpha` above is the DIRICHLET
        // alpha; the value-formula weight is `val_alpha`.  Copied to locals so the
        // par_iter_mut closure captures them, not &self.
        let (score_scale, margin_gain, val_alpha) =
            (self.score_scale, self.margin_gain, self.alpha);
        let exact_enabled = self.exact_endgame_max_secs > 0.0;
        // Python passes f32 (values/logits are .float()) to halve D2H transfer;
        // cast to f64 here for the tree's internal accumulation (unchanged).
        let vals: Vec<f64> = values.as_slice()?.iter().map(|&v| v as f64).collect();
        let n_rows = gathered.len();
        let mut gvecs: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let a = gathered.get_item(i)?;
            let a = a.downcast::<PyArray1<f32>>()?;
            gvecs.push(a.readonly().as_slice()?.iter().map(|&x| x as f64).collect());
        }

        let pending = std::mem::take(&mut self.pending);
        let mut pending_by_slot: Vec<Option<SlotTick>> =
            (0..self.slots.len()).map(|_| None).collect();
        for tick in pending {
            let si = tick.slot;
            pending_by_slot[si] = Some(tick);
        }

        // Keep this closure's established indentation stable; rustfmt otherwise
        // rewrites the entire ~200-line hot path when only its payload alias changes.
        #[rustfmt::skip]
        let finished_by_slot: PyResult<
            Vec<(usize, Option<(u64, Vec<MoveRecord>, (i32, i32), i8)>)>,
        > =
            self.slots
                .par_iter_mut()
                .zip(pending_by_slot.into_par_iter())
                .enumerate()
                .map(|(si, (slot, tick))| -> PyResult<_> {
                    let Some(tick) = tick else {
                        return Ok((si, None));
                    };

                    if tick.is_root {
                        let ev = &tick.evals[0];
                        let priors = softmax_f64(&gvecs[ev.row]);
                        let value0 = if ev.actor == 0 {
                            vals[ev.row]
                        } else {
                            -vals[ev.row]
                        };
                        if open_loop {
                            for (i, &(idx, placement, pick)) in ev.legal.iter().enumerate() {
                                let cid = slot.ol_arena.len() as u32;
                                slot.ol_arena
                                    .push(OLNode::new(priors[i], (placement, pick)));
                                slot.ol_arena[0].children.push((idx, cid));
                            }
                            slot.ol_arena[0].is_expanded = true;
                            slot.ol_arena[0].visit_count = 1;
                            slot.ol_arena[0].value_sum = value0;
                            if slot.move_profile.dirichlet_eps > 0.0 {
                                let dseed = slot.rng.r#gen::<u64>();
                                ol_add_dirichlet_noise(
                                    &mut slot.ol_arena,
                                    0,
                                    alpha,
                                    slot.move_profile.dirichlet_eps,
                                    Some(dseed),
                                );
                            }
                        } else {
                            for (i, &(idx, placement, pick)) in ev.legal.iter().enumerate() {
                                let cid = slot.arena.len() as u32;
                                slot.arena.push(Node::new(priors[i], (placement, pick)));
                                slot.arena[0].children.push((idx, cid));
                            }
                            slot.arena[0].is_expanded = true;
                            slot.arena[0].visit_count = 1;
                            slot.arena[0].value_sum = value0;
                            if slot.move_profile.dirichlet_eps > 0.0 {
                                let dseed = slot.rng.r#gen::<u64>();
                                add_dirichlet_noise(
                                    &mut slot.arena,
                                    0,
                                    alpha,
                                    slot.move_profile.dirichlet_eps,
                                    Some(dseed),
                                );
                            }
                        }
                        slot.sims_done = 0;
                        slot.state = SlotState::Searching;
                        return Ok((si, None));
                    }

                    // Searching tick: expand eval leaves, remove VL, back up, advance.
                    if open_loop {
                        // Validate the complete inference result for every admitted
                        // deck=8 panel before mutating any panel Q.  Each group must
                        // be the closed C(8,4)=70 support with unit probability mass.
                        let mut bootstrap_groups: HashMap<
                            u32,
                            Vec<(usize, f64, Vec<f64>, f64)>,
                        > =
                            HashMap::new();
                        for ev in tick
                            .evals
                            .iter()
                            .filter(|ev| ev.chance_bootstrap.is_some())
                        {
                            let meta = ev.chance_bootstrap.unwrap();
                            let priors = softmax_f64(&gvecs[ev.row]);
                            if priors.len() != ev.legal.len() {
                                return Err(PyValueError::new_err(
                                    "deck=8 bootstrap policy length mismatch",
                                ));
                            }
                            let value0 = if ev.actor == 0 {
                                vals[ev.row]
                            } else {
                                -vals[ev.row]
                            };
                            bootstrap_groups
                                .entry(meta.chance_node)
                                .or_default()
                                .push((ev.row, value0, priors, meta.probability));
                        }
                        for (&chance_node, results) in &bootstrap_groups {
                            let support = &slot.ol_arena[chance_node as usize].chance_children;
                            let mass: f64 = support.iter().map(|outcome| outcome.probability).sum();
                            let result_mass: f64 =
                                results.iter().map(|(_, _, _, probability)| probability).sum();
                            if results.len() != 70
                                || support.len() != 70
                                || (mass - 1.0).abs() > 1e-9
                                || (result_mass - 1.0).abs() > 1e-9
                            {
                                return Err(PyValueError::new_err(format!(
                                    "deck=8 chance panel must commit atomically as 70 rows with unit mass; got results={}, support={}, support_mass={mass}, result_mass={result_mass}",
                                    results.len(),
                                    support.len(),
                                )));
                            }
                        }

                        // All groups validated: install conditional policy/value
                        // bootstraps. The panel mean is published only after current
                        // virtual loss is removed, so any earlier real observation
                        // visits can safely supersede their bootstrap values.
                        for ev in tick
                            .evals
                            .iter()
                            .filter(|ev| ev.chance_bootstrap.is_some())
                        {
                            let meta = ev.chance_bootstrap.unwrap();
                            let results = &bootstrap_groups[&meta.chance_node];
                            let (_, value0, priors, _) = results
                                .iter()
                                .find(|(result_row, _, _, _)| *result_row == ev.row)
                                .expect("validated bootstrap result row");
                            if !slot.ol_arena[ev.leaf as usize].is_expanded {
                                for (i, &(idx, placement, pick)) in ev.legal.iter().enumerate() {
                                    let cid = slot.ol_arena.len() as u32;
                                    slot.ol_arena.push(OLNode::new(
                                        priors[i],
                                        (placement, pick),
                                    ));
                                    slot.ol_arena[ev.leaf as usize].children.push((idx, cid));
                                }
                                slot.ol_arena[ev.leaf as usize].is_expanded = true;
                            } else {
                                ol_add_missing_children(
                                    &mut slot.ol_arena,
                                    ev.leaf,
                                    &ev.legal,
                                    priors,
                                );
                            }
                            slot.ol_arena[ev.leaf as usize].bootstrap_value0 = Some(*value0);
                        }

                        // Issue 1 fix: per-PATH value (not per-node).  evals are now
                        // one-per-non-terminal-simulation (no de-dup), so each path
                        // backs up its OWN concrete eval.  path_v0[pi] is the value
                        // for paths[pi]; terminal paths fill it from their leaf state.
                        let mut path_v0: Vec<Option<f64>> = vec![None; tick.paths.len()];
                        let mut search_eval_index = 0usize;
                        for ev in tick
                            .evals
                            .iter()
                            .filter(|ev| ev.chance_bootstrap.is_none())
                        {
                            let priors = softmax_f64(&gvecs[ev.row]);
                            let value0 = if ev.actor == 0 {
                                vals[ev.row]
                            } else {
                                -vals[ev.row]
                            };
                            if !slot.ol_arena[ev.leaf as usize].is_expanded {
                                // First expansion of this OLNode (this tick or ever):
                                // create children from this concrete state's legal
                                // actions + priors.  Only the FIRST eval for a given
                                // node id expands it; later evals for the same node
                                // (Issue 1: collisions are now possible) fall to the
                                // missing-child branch below.
                                for (i, &(idx, placement, pick)) in ev.legal.iter().enumerate() {
                                    let cid = slot.ol_arena.len() as u32;
                                    slot.ol_arena
                                        .push(OLNode::new(priors[i], (placement, pick)));
                                    slot.ol_arena[ev.leaf as usize].children.push((idx, cid));
                                }
                                slot.ol_arena[ev.leaf as usize].is_expanded = true;
                            } else {
                                // Issue 2 fix: node already expanded (earlier this tick
                                // or a previous tick) but THIS determinization may have
                                // legal actions absent from its children — add them.
                                // LIMITATION: the added children take their priors from
                                // the determinization that first encounters them, not an
                                // average across determinizations.  Correct treatment
                                // (running-average priors or deferred expansion) is a
                                // known open-loop approximation, deferred.
                                ol_add_missing_children(
                                    &mut slot.ol_arena,
                                    ev.leaf,
                                    &ev.legal,
                                    &priors,
                                );
                            }
                            // Always record THIS eval's value for its own path.
                            path_v0[tick.eval_path_indices[search_eval_index]] = Some(value0);
                            search_eval_index += 1;
                        }
                        debug_assert_eq!(
                            search_eval_index,
                            tick.eval_path_indices.len(),
                            "every open-loop search eval must map to exactly one path",
                        );
                        // Remove VL using each path's recorded per-node actors.
                        for (pi, path) in tick.paths.iter().enumerate() {
                            ol_apply_virtual_loss(
                                &mut slot.ol_arena,
                                path,
                                &tick.ol_actors[pi],
                                -1,
                                vl,
                            );
                        }
                        // Match A1c admission semantics: an observation with real
                        // visits contributes its searched conditional mean; only an
                        // unvisited row contributes the new network bootstrap.
                        for &chance_node in bootstrap_groups.keys() {
                            let weighted = ol_panel_weighted_value(&slot.ol_arena, chance_node);
                            let node = &mut slot.ol_arena[chance_node as usize];
                            node.chance_backup = OLChanceBackup::PanelMean;
                            node.chance_visited_mass = 1.0;
                            node.chance_weighted_value = weighted;
                            slot.deck8_chance_panel_count += 1;
                            slot.deck8_chance_bootstrap_rows += 70;
                        }
                        // Backup: terminal value from each path's own concrete leaf
                        // state (deck-dependent), else this path's own eval value.
                        for (pi, path) in tick.paths.iter().enumerate() {
                            let ls = &tick.ol_leaf_states[pi];
                            let v0 = if ls.phase == GAME_OVER {
                                terminal_search_value(ls, score_scale, margin_gain, val_alpha)
                            } else {
                                path_v0[pi].expect("non-terminal path must have an eval value")
                            };
                            ol_backup_path(
                                &mut slot.ol_arena,
                                path,
                                v0,
                                tick.chance_steps[pi],
                            );
                        }
                        slot.sims_done += tick.paths.len() + tick.chance_bootstrap_rows;
                    } else {
                        let mut leaf_v0: HashMap<u32, f64> = HashMap::new();
                        for ev in &tick.evals {
                            let priors = softmax_f64(&gvecs[ev.row]);
                            let value0 = if ev.actor == 0 {
                                vals[ev.row]
                            } else {
                                -vals[ev.row]
                            };
                            if !slot.arena[ev.leaf as usize].is_expanded {
                                for (i, &(idx, placement, pick)) in ev.legal.iter().enumerate() {
                                    let cid = slot.arena.len() as u32;
                                    slot.arena.push(Node::new(priors[i], (placement, pick)));
                                    slot.arena[ev.leaf as usize].children.push((idx, cid));
                                }
                                slot.arena[ev.leaf as usize].is_expanded = true;
                            }
                            leaf_v0.insert(ev.leaf, value0);
                        }
                        for path in &tick.paths {
                            apply_virtual_loss(&mut slot.arena, path, -1, vl);
                        }
                        for path in &tick.paths {
                            let leaf = *path.last().unwrap();
                            let v0 = if slot.arena[leaf as usize].state.as_ref().unwrap().phase
                                == GAME_OVER
                            {
                                terminal_search_value(
                                    slot.arena[leaf as usize].state.as_ref().unwrap(),
                                    score_scale,
                                    margin_gain,
                                    val_alpha,
                                )
                            } else {
                                leaf_v0[&leaf]
                            };
                            for &n in path {
                                arena_backup(&mut slot.arena, n, v0);
                            }
                        }
                        slot.sims_done += tick.paths.len();
                    }
                    let finished_game = if slot.sims_done >= slot.move_profile.target_sims {
                        slot.finalize_move(
                            open_loop,
                            exact_enabled,
                            playout_cap_randomization,
                            full_search_fraction,
                            n_sims,
                            fast_move_sims,
                            record_fast_moves,
                            dirichlet_eps,
                            fast_move_dirichlet_eps,
                            temp_moves,
                            fast_move_temp_moves,
                            seat_override,
                        )?
                    } else {
                        None
                    };
                    Ok((si, finished_game))
                })
                .collect();

        let mut finished_by_slot = finished_by_slot?;
        finished_by_slot.sort_unstable_by_key(|(si, _)| *si);
        // Games finished by the exact endgame solver during step() (their slots
        // were already recycled there) are returned alongside the MCTS-finished
        // games of this tick.
        let mut finished_rust: Vec<FinishedGame> = std::mem::take(&mut self.pending_exact);
        for (si, finished_game) in finished_by_slot {
            if let Some(fg) = finished_game {
                finished_rust.push(fg);
                // Roll this finished slot's diagnostics into the cumulative totals
                // before it is reset/idled, so the getters survive across games.
                self.cum_fallback_count += self.slots[si].fallback_count as u64;
                self.cum_missing_child_count += self.slots[si].missing_child_count as u64;
                self.absorb_slot_move_counts(si);
                if self.games_started < self.games_target {
                    let ns = self.next_seed;
                    self.next_seed += 1;
                    self.games_started += 1;
                    let mut slot = SearchSlot::new_for_game(
                        new_game(ns, harmony, mk),
                        ns,
                        open_loop,
                        random_opening,
                    );
                    slot.choose_move_profile(
                        playout_cap_randomization,
                        full_search_fraction,
                        n_sims,
                        fast_move_sims,
                        record_fast_moves,
                        dirichlet_eps,
                        fast_move_dirichlet_eps,
                        temp_moves,
                        fast_move_temp_moves,
                        seat_override,
                    );
                    self.slots[si] = slot;
                } else {
                    self.slots[si].state = SlotState::Idle;
                    self.slots[si].fallback_count = 0;
                    self.slots[si].missing_child_count = 0;
                    self.slots[si].deck8_chance_panel_count = 0;
                    self.slots[si].deck8_chance_bootstrap_rows = 0;
                    self.slots[si].deck8_chance_budget_blocked_count = 0;
                }
            }
        }

        // Convert finished games' records to numpy + return.
        let out = PyList::empty(py);
        for (seed, records, (s0, s1), outcome0) in finished_rust {
            let z0 = ((s0 - s1) as f64 / 30.0).tanh();
            let examples = PyList::empty(py);
            for r in records {
                let z = if r.actor == 0 { z0 } else { -z0 };
                let root_stats = (
                    r.root_prior_idx.into_pyarray(py),
                    r.root_prior_val.into_pyarray(py),
                    r.root_visit_count.into_pyarray(py),
                );
                let actor = r.actor;
                let tup = (
                    r.my.into_pyarray(py),
                    r.opp.into_pyarray(py),
                    r.flat.into_pyarray(py),
                    r.policy_idx.into_pyarray(py),
                    r.policy_val.into_pyarray(py),
                    r.legal_idx.into_pyarray(py),
                    root_stats,
                    z,
                    r.own_score,
                    r.opp_score,
                    r.win_target,
                    actor,
                );
                examples.append(tup)?;
            }
            out.append((seed, examples, (s0, s1, outcome0)))?;
        }
        Ok(out)
    }
}

#[inline]
fn arena_backup(arena: &mut [Node], node_id: u32, v0: f64) {
    let n = &mut arena[node_id as usize];
    n.visit_count += 1;
    n.value_sum += v0;
}

// ─── D4 augmentation ─────────────────────────────────────────────────────
const NUM_D4: usize = 8;
const N_BOARD_CH_AUG: usize = 9; // same as N_BOARD_CH — alias for clarity
const POLICY_SIZE: usize = 3390; // NUM_JOINT_ACTIONS
const PLACEMENT_AXIS: usize = 678; // PLACEMENT_AXIS_SIZE
const PICK_AXIS: usize = 5; // PICK_AXIS_SIZE
const NUM_SPATIAL: usize = 676; // NUM_SPATIAL_PLACEMENTS = 4 * 169
const NUM_DIRS: usize = 4;
const CANVAS: usize = 13; // CANVAS_SIZE

// Each D4 element: (ccw_rotations, h_flip, direction_permutation).
// Direction permutation: new_dir_channel[d] = old[perm[d]].
// Mirrors augmentation.py _D4_ELEMENTS exactly.
const D4_ELEMENTS: [(u8, bool, [usize; 4]); 8] = [
    (0, false, [0, 1, 2, 3]), // 0: IDENTITY
    (1, false, [1, 2, 3, 0]), // 1: ROT90 CCW
    (2, false, [2, 3, 0, 1]), // 2: ROT180
    (3, false, [3, 0, 1, 2]), // 3: ROT270 CCW
    (0, true, [2, 1, 0, 3]),  // 4: FLIP_H
    (1, true, [3, 2, 1, 0]),  // 5: ROT90 + FLIP_H
    (2, true, [0, 3, 2, 1]),  // 6: ROT180 + FLIP_H (= FLIP_V)
    (3, true, [1, 0, 3, 2]),  // 7: ROT270 + FLIP_H
];

const INVERSE_D4: [usize; 8] = [0, 3, 2, 1, 4, 5, 6, 7];

/// Apply k CCW 90° rotations then optional h-flip to a (C, H, W) array
/// stored as a flat Vec<f32> with C=channels, H=W=CANVAS (13).
/// Returns a new contiguous Vec<f32> in the same (C, H, W) layout.
fn transform_spatial(src: &[f32], channels: usize, k: u8, flip: bool) -> Vec<f32> {
    let n = CANVAS;
    let ch_stride = n * n;
    let mut out = vec![0f32; channels * ch_stride];

    for c in 0..channels {
        for y in 0..n {
            for x in 0..n {
                // Apply k CCW rotations: one CCW rotation maps (y,x) → (n-1-x, y).
                let (mut ry, mut rx) = (y, x);
                for _ in 0..k {
                    // np.rot90 CCW: (y,x) → (n-1-x, y).
                    let tmp = ry;
                    ry = n - 1 - rx;
                    rx = tmp;
                }
                // H-flip: flip the x axis.
                if flip {
                    rx = n - 1 - rx;
                }
                let src_idx = c * ch_stride + y * n + x;
                let dst_idx = c * ch_stride + ry * n + rx;
                out[dst_idx] = src[src_idx];
            }
        }
    }
    out
}

/// Apply a D4 transform to a flat policy vector of length POLICY_SIZE=3390.
/// Layout: joint_idx = placement_idx * PICK_AXIS + pick_idx.
/// Spatial placements: placement_idx = dir * 169 + y * 13 + x  (indices 0..676).
/// Non-spatial (DISCARD=676, NO_PLACEMENT=677) and pick axis are invariant.
fn transform_policy(src: &[f32], k: u8, flip: bool, dir_perm: &[usize; 4]) -> Vec<f32> {
    let n = CANVAS;
    let mut out = vec![0f32; POLICY_SIZE];

    // Copy invariant non-spatial slice (DISCARD + NO_PLACEMENT rows, all picks).
    for p in NUM_SPATIAL..PLACEMENT_AXIS {
        for pk in 0..PICK_AXIS {
            let idx = p * PICK_AXIS + pk;
            out[idx] = src[idx];
        }
    }

    // Transform spatial slice:
    // src layout: [dir][y][x][pick], flattened as placement_idx * PICK_AXIS + pick
    // where placement_idx = dir * 169 + y * 13 + x.
    for src_dir in 0..NUM_DIRS {
        for y in 0..n {
            for x in 0..n {
                // Rotate (y, x): same logic as transform_spatial.
                let (mut ry, mut rx) = (y, x);
                for _ in 0..k {
                    // np.rot90 CCW: (y,x) → (n-1-x, y).
                    let tmp = ry;
                    ry = n - 1 - rx;
                    rx = tmp;
                }
                if flip {
                    rx = n - 1 - rx;
                }

                // Permute direction: new direction d gets old direction perm[d].
                // We are writing src_dir's data into dst_dir = perm^{-1}[src_dir].
                // Equivalently: for each dst_dir d, dst[d] = src[perm[d]].
                // Find dst_dir such that dir_perm[dst_dir] == src_dir.
                let dst_dir = dir_perm.iter().position(|&p| p == src_dir).unwrap();

                let dst_placement = dst_dir * n * n + ry * n + rx;
                let src_placement = src_dir * n * n + y * n + x;

                for pk in 0..PICK_AXIS {
                    out[dst_placement * PICK_AXIS + pk] = src[src_placement * PICK_AXIS + pk];
                }
            }
        }
    }
    out
}

/// Flat-vector D4 transform. Every flat feature is orientation-invariant EXCEPT
/// the per-player bounding-box (width, height): a rotation by an odd number of
/// quarter-turns exchanges the x and y extents. 180° rotations and pure
/// reflections (h-flip) preserve both, so the swap depends only on `k` parity,
/// not `flip`. Keep bit-identical to augmentation._transform_flat.
fn transform_flat(src: &[f32], k: u8) -> Vec<f32> {
    let mut out = src.to_vec();
    if k % 2 == 1 {
        for off in [OFF_MY_BOARD_SUMMARY, OFF_OPP_BOARD_SUMMARY] {
            out.swap(off + BS_WIDTH_LOCAL, off + BS_WIDTH_LOCAL + 1);
        }
    }
    out
}

/// Apply one of the 8 D4 transforms to a Kingdomino training tuple.
///
/// Arguments:
///   my_board    : (9, 13, 13) f32 array, C-contiguous
///   opp_board   : (9, 13, 13) f32 array, C-contiguous
///   flat        : flat f32 array — width/height swap under odd rotations,
///                 all other fields orientation-invariant (see transform_flat)
///   policy      : (3390,) f32 array
///   transform_id: int in [0, 8)
///
/// Returns (my_board_t, opp_board_t, flat_copy, policy_t) as numpy arrays.
/// Scalars (z, own_score, opp_score, win_target) are invariant; callers
/// pass them through unchanged — not included in the return value to keep
/// the boundary minimal.
///
/// Bit-identical to augmentation.py augment() for the array components.
#[pyfunction]
fn d4_augment<'py>(
    py: Python<'py>,
    my_board: PyReadonlyArray3<'py, f32>,
    opp_board: PyReadonlyArray3<'py, f32>,
    flat: PyReadonlyArray1<'py, f32>,
    policy: PyReadonlyArray1<'py, f32>,
    transform_id: usize,
) -> PyResult<(
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    if transform_id >= NUM_D4 {
        return Err(PyValueError::new_err(format!(
            "transform_id must be in [0, {NUM_D4}); got {transform_id}"
        )));
    }
    let (k, flip, dir_perm) = D4_ELEMENTS[transform_id];
    let mb_sl = my_board.as_slice()?;
    let ob_sl = opp_board.as_slice()?;
    let fl_sl = flat.as_slice()?;
    let pol_sl = policy.as_slice()?;

    // Validate lengths up front so malformed direct callers get a ValueError
    // instead of an out-of-bounds panic (transform_flat/spatial index by fixed
    // offsets). Normal callers validate upstream; this hardens the public entry.
    let board_len = N_BOARD_CH_AUG * CANVAS * CANVAS;
    if mb_sl.len() != board_len || ob_sl.len() != board_len {
        return Err(PyValueError::new_err(format!(
            "board must have {board_len} elements ({N_BOARD_CH_AUG}x{CANVAS}x{CANVAS})"
        )));
    }
    if fl_sl.len() != FLAT_SIZE {
        return Err(PyValueError::new_err(format!(
            "flat must have {FLAT_SIZE} elements; got {}",
            fl_sl.len()
        )));
    }
    if pol_sl.len() != POLICY_SIZE {
        return Err(PyValueError::new_err(format!(
            "policy must have {POLICY_SIZE} elements; got {}",
            pol_sl.len()
        )));
    }

    let mb_t = transform_spatial(mb_sl, N_BOARD_CH_AUG, k, flip);
    let ob_t = transform_spatial(ob_sl, N_BOARD_CH_AUG, k, flip);
    let fl_cp = transform_flat(fl_sl, k);
    let pol_t = transform_policy(pol_sl, k, flip, &dir_perm);

    Ok((
        Array3::from_shape_vec((N_BOARD_CH_AUG, CANVAS, CANVAS), mb_t)
            .expect("board shape")
            .into_pyarray(py),
        Array3::from_shape_vec((N_BOARD_CH_AUG, CANVAS, CANVAS), ob_t)
            .expect("board shape")
            .into_pyarray(py),
        Array1::from_vec(fl_cp).into_pyarray(py),
        Array1::from_vec(pol_t).into_pyarray(py),
    ))
}

/// Return the D4 transform_id that undoes `t` (mirrors
/// augmentation._INVERSE_TRANSFORM).
#[pyfunction]
fn d4_inverse_transform_id(t: usize) -> PyResult<usize> {
    if t >= NUM_D4 {
        return Err(PyValueError::new_err(format!(
            "transform_id must be in [0, {NUM_D4}); got {t}"
        )));
    }
    Ok(INVERSE_D4[t])
}

/// Apply a D4 transform to a flat bool legal-mask of length POLICY_SIZE.
/// Identical spatial-rotate + direction-permute as transform_policy (the mask
/// must transform by the SAME element as the policy), with the non-spatial
/// (DISCARD / NO_PLACEMENT) rows and the pick axis left invariant.
fn transform_mask(src: &[bool], k: u8, flip: bool, dir_perm: &[usize; 4]) -> Vec<bool> {
    let n = CANVAS;
    let mut out = vec![false; POLICY_SIZE];
    for p in NUM_SPATIAL..PLACEMENT_AXIS {
        for pk in 0..PICK_AXIS {
            let idx = p * PICK_AXIS + pk;
            out[idx] = src[idx];
        }
    }
    for src_dir in 0..NUM_DIRS {
        for y in 0..n {
            for x in 0..n {
                let (mut ry, mut rx) = (y, x);
                for _ in 0..k {
                    // np.rot90 CCW: (y,x) → (n-1-x, y).
                    let tmp = ry;
                    ry = n - 1 - rx;
                    rx = tmp;
                }
                if flip {
                    rx = n - 1 - rx;
                }
                let dst_dir = dir_perm.iter().position(|&p| p == src_dir).unwrap();
                let dst_placement = dst_dir * n * n + ry * n + rx;
                let src_placement = src_dir * n * n + y * n + x;
                for pk in 0..PICK_AXIS {
                    out[dst_placement * PICK_AXIS + pk] = src[src_placement * PICK_AXIS + pk];
                }
            }
        }
    }
    out
}

/// Apply one of the 8 D4 transforms to a flat bool legal-mask (3390,).
/// Releases the GIL during the transform (the loop is pure Rust), so callers in
/// a thread pool actually run in parallel.  Bit-identical to
/// augmentation.augment_mask().
#[pyfunction]
fn d4_augment_mask<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray1<'py, bool>,
    transform_id: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    if transform_id >= NUM_D4 {
        return Err(PyValueError::new_err(format!(
            "transform_id must be in [0, {NUM_D4}); got {transform_id}"
        )));
    }
    let (k, flip, dir_perm) = D4_ELEMENTS[transform_id];
    let m = mask.as_slice()?;
    let out = transform_mask(m, k, flip, &dir_perm);
    Ok(Array1::from_vec(out).into_pyarray(py))
}

#[cfg(test)]
mod augment_tests {
    use super::*;

    #[test]
    fn sim_seeds_match_sequential_gen() {
        // Pre-generating the per-simulation seeds in a batch must yield the exact
        // same sequence as calling rng.gen() once per simulation inside the loop.
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);
        let n = 6usize;
        let seeds_batch: Vec<u64> = (0..n).map(|_| rng_a.r#gen::<u64>()).collect();
        let seeds_seq: Vec<u64> = (0..n).map(|_| rng_b.r#gen::<u64>()).collect();
        assert_eq!(
            seeds_batch, seeds_seq,
            "pre-generated seeds must match sequential gen()"
        );
    }

    #[test]
    fn mask_transform_matches_policy_transform() {
        // transform_mask must agree with transform_policy on a bool mask cast to
        // f32 (the mask must transform by the same element as the policy).
        let mut mask = vec![false; POLICY_SIZE];
        for i in (0..POLICY_SIZE).step_by(7) {
            mask[i] = true;
        }
        let polf: Vec<f32> = mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        for t in 0..8 {
            let (k, flip, dp) = D4_ELEMENTS[t];
            let mt = transform_mask(&mask, k, flip, &dp);
            let pt = transform_policy(&polf, k, flip, &dp);
            let mt_f: Vec<f32> = mt.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
            assert_eq!(mt_f, pt, "mask vs policy transform mismatch at t={t}");
        }
    }

    #[test]
    fn identity_is_noop() {
        // transform_id=0 (identity): output == input for both board and policy.
        let board: Vec<f32> = (0..9 * 13 * 13).map(|i| i as f32).collect();
        let out = transform_spatial(&board, 9, 0, false);
        assert_eq!(board, out);

        let policy: Vec<f32> = (0..3390).map(|i| i as f32).collect();
        let pout = transform_policy(&policy, 0, false, &[0, 1, 2, 3]);
        assert_eq!(policy, pout);
    }

    #[test]
    fn castle_centre_invariant() {
        // Castle at (6,6) survives all 8 transforms on the CASTLE channel (ch 7).
        let mut board = vec![0f32; 9 * 13 * 13];
        let castle_ch = 7usize;
        board[castle_ch * 13 * 13 + 6 * 13 + 6] = 1.0;
        for &(k, flip, _) in &D4_ELEMENTS {
            let out = transform_spatial(&board, 9, k, flip);
            assert_eq!(
                out[castle_ch * 13 * 13 + 6 * 13 + 6],
                1.0,
                "castle moved under k={k} flip={flip}"
            );
        }
    }

    #[test]
    fn four_rotations_return_to_identity() {
        // Applying ROT90 four times gives back the original.
        let board: Vec<f32> = (0..9 * 13 * 13).map(|i| i as f32).collect();
        let mut cur = board.clone();
        for _ in 0..4 {
            cur = transform_spatial(&cur, 9, 1, false);
        }
        assert_eq!(board, cur);
    }

    #[test]
    fn inverse_undoes_transform() {
        // augment(inverse(t), augment(t, x)) == x for all t.
        // Test on the policy vector (most sensitive to direction permutation).
        let policy: Vec<f32> = (0..3390).map(|i| i as f32).collect();
        for t in 0..8 {
            let (k, flip, dir_perm) = D4_ELEMENTS[t];
            let inv_t = INVERSE_D4[t];
            let (ki, fi, dpi) = D4_ELEMENTS[inv_t];
            let mid = transform_policy(&policy, k, flip, &dir_perm);
            let back = transform_policy(&mid, ki, fi, &dpi);
            assert_eq!(policy, back, "inverse failed for t={t}");
        }
    }
}

#[cfg(test)]
mod ol_tests {
    use super::*;

    fn test_chance_child(row: [u16; 4], probability: f64, node_id: u32) -> OLChanceChild {
        OLChanceChild {
            row,
            probability,
            multiplicity: 1,
            node_id: Some(node_id),
            #[cfg(debug_assertions)]
            public_key: None,
        }
    }

    fn refresh_test_chance_cache(arena: &mut [OLNode], node_id: u32) {
        let mut mass = 0.0;
        let mut weighted = 0.0;
        for outcome in &arena[node_id as usize].chance_children {
            let Some(child_id) = outcome.node_id else {
                continue;
            };
            let child = &arena[child_id as usize];
            if child.visit_count > 0 {
                mass += outcome.probability;
                weighted += outcome.probability * child.value_sum / child.visit_count as f64;
            }
        }
        arena[node_id as usize].chance_visited_mass = mass;
        arena[node_id as usize].chance_weighted_value = weighted;
    }

    #[test]
    fn issue1_per_path_value_mapping() {
        // Issue 1: with de-dup removed, two simulations can end at the SAME OLNode
        // (same leaf id) under different determinizations.  eval_path_indices must
        // route each eval's value to ITS OWN path, so path_v0 holds the two
        // DISTINCT values — not the first one duplicated (the old leaf_v0 bug).
        let tick_eval_leaf = [5u32, 5u32]; // both evals collide on node 5
        let eval_values = [0.3f64, 0.7f64];
        let eval_path_indices = [0usize, 1usize];
        let mut path_v0: Vec<Option<f64>> = vec![None; 2];
        for ei in 0..2 {
            path_v0[eval_path_indices[ei]] = Some(eval_values[ei]);
        }
        assert_eq!(
            path_v0,
            vec![Some(0.3), Some(0.7)],
            "colliding-leaf sims must keep distinct per-path values"
        );
        assert_eq!(tick_eval_leaf[0], tick_eval_leaf[1]); // they really do collide

        // Backup both paths (each = root→child) and confirm the shared child's
        // value_sum is v1+v2 (not 2*v1, which the de-dup bug produced).
        let mut arena = vec![
            OLNode::new(1.0, (None, None)),    // 0: root
            OLNode::new(0.5, (None, Some(0))), // 1: shared child
        ];
        arena[0].children.push((0, 1));
        arena[0].is_expanded = true;
        let paths = [vec![0u32, 1u32], vec![0u32, 1u32]];
        for (pi, path) in paths.iter().enumerate() {
            let v0 = path_v0[pi].unwrap();
            for &n in path {
                arena[n as usize].visit_count += 1;
                arena[n as usize].value_sum += v0;
            }
        }
        assert!(
            (arena[1].value_sum - 1.0).abs() < 1e-12,
            "shared child value_sum should be 0.3+0.7=1.0, got {}",
            arena[1].value_sum
        );
        assert_eq!(arena[1].visit_count, 2);
    }

    #[test]
    fn issue2_missing_child_insert() {
        // Issue 2: a node expanded with children {1,3} is later reached by a
        // determinization whose legal set is {1,2,3,5}.  ol_add_missing_children
        // must add 2 and 5 (with this det's priors) and keep children ascending.
        let mut arena = vec![
            OLNode::new(1.0, (None, None)), // 0: node under test
            OLNode::new(0.5, (None, Some(0))),
            OLNode::new(0.5, (None, Some(1))),
        ];
        arena[0].children.push((1, 1));
        arena[0].children.push((3, 2));
        arena[0].is_expanded = true;
        let legal: Vec<(u16, Option<(i8, i8, i8, i8, bool)>, Option<u16>)> = vec![
            (1, None, Some(0)),
            (2, None, Some(1)),
            (3, None, Some(2)),
            (5, None, Some(3)),
        ];
        let priors = vec![0.25f64, 0.25, 0.25, 0.25];
        let added = ol_add_missing_children(&mut arena, 0, &legal, &priors);
        assert_eq!(
            added, 2,
            "should add exactly the two missing children (2 and 5)"
        );
        let idxs: Vec<u16> = arena[0].children.iter().map(|&(i, _)| i).collect();
        assert_eq!(
            idxs,
            vec![1, 2, 3, 5],
            "children must be the union, ascending"
        );
        // Sorted ascending → binary search (ol_select_child) finds every index.
        for q in [1u16, 2, 3, 5] {
            assert!(
                arena[0]
                    .children
                    .binary_search_by_key(&q, |&(i, _)| i)
                    .is_ok()
            );
        }
        // Idempotent: re-adding the same legal set adds nothing.
        let again = ol_add_missing_children(&mut arena, 0, &legal, &priors);
        assert_eq!(again, 0, "no new children on a second pass");
    }

    #[test]
    fn one_reveal_balanced_panel_exposes_every_tile_x_times() -> PyResult<()> {
        let deck: Vec<u16> = (1..=12).collect();
        let exposure = 4usize;
        let panel = ol_one_reveal_panel(&deck, exposure, 1, 20260808)?;
        let mut counts = HashMap::<u16, usize>::new();
        let raw_rows: usize = panel.iter().map(|x| x.multiplicity).sum();
        assert_eq!(raw_rows, deck.len() / 4 * exposure);
        for outcome in &panel {
            for &tile in &outcome.row {
                *counts.entry(tile).or_insert(0) += outcome.multiplicity;
            }
        }
        for tile in deck {
            assert_eq!(counts.get(&tile), Some(&exposure));
        }
        let mass: f64 = panel.iter().map(|x| x.probability).sum();
        assert!((mass - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn a1c_balanced_plan_preserves_complete_cycle_identity() -> PyResult<()> {
        let deck: Vec<u16> = (1..=28).collect();
        let cycles = a1c_one_reveal_cycles(&deck, 4, A1cPanelSampling::Balanced, 20260809)?;
        assert_eq!(cycles.len(), 4);
        for cycle in cycles {
            assert_eq!(cycle.len(), 7);
            let mut exposed: Vec<u16> = cycle.into_iter().flatten().collect();
            exposed.sort_unstable();
            assert_eq!(exposed, deck);
        }
        Ok(())
    }

    #[test]
    fn a1c_iid_plan_is_matched_width_without_balance_promise() -> PyResult<()> {
        let deck: Vec<u16> = (1..=28).collect();
        let cycles = a1c_one_reveal_cycles(&deck, 4, A1cPanelSampling::Iid, 20260809)?;
        assert_eq!(cycles.len(), 4);
        assert!(cycles.iter().all(|cycle| cycle.len() == 7));
        for row in cycles.iter().flatten() {
            assert!(row.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(row.iter().all(|tile| deck.contains(tile)));
        }
        // This fixed seed is intentionally not tile-balanced. The assertion
        // catches an accidental implementation that merely relabels balanced
        // partitions as IID while still matching its row count.
        let first_cycle: Vec<u16> = cycles[0].iter().flatten().copied().collect();
        let unique: std::collections::HashSet<u16> = first_cycle.iter().copied().collect();
        assert!(unique.len() < deck.len());
        Ok(())
    }

    #[test]
    fn a1c_width_waits_for_init_and_adds_whole_cycles() {
        assert_eq!(a1c_target_cycles(15, 16, 8, 0.25), 0);
        assert_eq!(a1c_target_cycles(16, 16, 8, 0.25), 1);
        assert_eq!(a1c_target_cycles(64, 16, 8, 0.25), 2);
        assert_eq!(a1c_target_cycles(1024, 16, 4, 0.25), 4);
        assert_eq!(a1c_target_cycles(1024, 16, 0, 0.25), 0);
    }

    #[test]
    fn a1c_initialization_guard_charges_the_proposed_batch() {
        // 11 new rows after 33 ordinary rows is exactly 25% of the resulting
        // 44 NN rows and is admissible. One fewer ordinary row is not.
        assert!(a1c_initialization_within_budget(0, 33, 11, 0.25));
        assert!(!a1c_initialization_within_budget(0, 32, 11, 0.25));
        // Cumulative accounting includes prior initialization work.
        assert!(a1c_initialization_within_budget(11, 55, 7, 0.30));
        assert!(!a1c_initialization_within_budget(11, 55, 7, 0.25));
        assert!(a1c_initialization_within_budget(11, 55, 0, 0.25));
        assert!(!a1c_initialization_within_budget(56, 55, 1, 0.25));
    }

    #[test]
    fn one_reveal_deck8_support_is_exact_and_closed() -> PyResult<()> {
        let deck: Vec<u16> = (1..=8).collect();
        let panel = ol_one_reveal_panel(&deck, 1, 70, 9)?;
        assert_eq!(panel.len(), 70);
        assert!(panel.iter().all(|x| x.multiplicity == 1));
        assert!(
            panel
                .iter()
                .all(|x| (x.probability - 1.0 / 70.0).abs() < 1e-12)
        );
        Ok(())
    }

    #[test]
    fn one_reveal_chance_value_uses_probability_not_visit_share() {
        let mut arena = vec![
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
        ];
        arena[1].visit_count = 100;
        arena[1].value_sum = 100.0;
        arena[2].visit_count = 1;
        arena[2].value_sum = -1.0;
        arena[0].chance_children = vec![
            test_chance_child([1, 2, 3, 4], 0.25, 1),
            test_chance_child([5, 6, 7, 8], 0.75, 2),
        ];
        refresh_test_chance_cache(&mut arena, 0);
        assert!((ol_chance_value_p0(&arena, 0) + 0.5).abs() < 1e-12);
        // The empirical visit-weighted mean would be strongly positive; pin the
        // discriminator so a future refactor cannot silently revert to it.
        assert!((arena[0].chance_children[0].probability - 0.25).abs() < 1e-12);
    }

    #[test]
    fn one_reveal_chance_value_renormalizes_visited_mass_without_fpu() {
        let mut arena = vec![
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
        ];
        arena[1].visit_count = 2;
        arena[1].value_sum = 1.0;
        arena[0].visit_count = 2;
        arena[0].chance_children = vec![
            test_chance_child([1, 2, 3, 4], 0.01, 1),
            test_chance_child([5, 6, 7, 8], 0.99, 2),
        ];
        refresh_test_chance_cache(&mut arena, 0);
        // The unseen 99% contributes no arbitrary constant. The only observed
        // conditional estimate is +0.5, irrespective of its chance mass.
        assert!((ol_chance_value_p0(&arena, 0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn one_reveal_chance_node_keeps_full_strength_virtual_loss() {
        let mut arena = vec![
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
        ];
        arena[1].visit_count = 100;
        arena[1].value_sum = 50.0;
        arena[0].visit_count = 100;
        arena[0].chance_children = vec![test_chance_child([1, 2, 3, 4], 0.01, 1)];
        refresh_test_chance_cache(&mut arena, 0);
        assert!((ol_chance_value_p0(&arena, 0) - 0.5).abs() < 1e-12);

        // A player-0 in-flight traversal applies a -1 virtual result to the
        // chance node as a whole, not a probability-scaled -0.01 penalty.
        ol_apply_virtual_loss(&mut arena, &[2, 0], &[0], 1, 1);
        let expected = (0.5 * 100.0 - 1.0) / 101.0;
        assert!((ol_chance_value_p0(&arena, 0) - expected).abs() < 1e-12);
        ol_apply_virtual_loss(&mut arena, &[2, 0], &[0], -1, 1);
        assert!((ol_chance_value_p0(&arena, 0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn one_reveal_reported_q_uses_current_chance_expectation() {
        let mut arena = vec![
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
        ];
        arena[0].visit_count = 4;
        arena[0].value_sum = -40.0; // deliberately stale expectation snapshots
        arena[1].visit_count = 2;
        arena[1].value_sum = 1.0;
        arena[0].chance_children = vec![test_chance_child([1, 2, 3, 4], 1.0, 1)];
        refresh_test_chance_cache(&mut arena, 0);
        assert!((ol_chance_value_p0(&arena, 0) - 0.5).abs() < 1e-12);
        assert!((ol_reported_value_sum(&arena, 0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn one_reveal_backup_updates_cached_hajek_estimate_incrementally() {
        let mut arena = vec![
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
        ];
        arena[0].chance_children = vec![
            test_chance_child([1, 2, 3, 4], 0.25, 1),
            OLChanceChild {
                row: [5, 6, 7, 8],
                probability: 0.75,
                multiplicity: 1,
                node_id: None,
                #[cfg(debug_assertions)]
                public_key: None,
            },
        ];
        ol_backup_path(&mut arena, &[0, 1], 0.8, Some((0, 0.25)));
        assert!((arena[0].chance_visited_mass - 0.25).abs() < 1e-12);
        assert!((arena[0].chance_weighted_value - 0.2).abs() < 1e-12);
        assert!((ol_chance_value_p0(&arena, 0) - 0.8).abs() < 1e-12);

        ol_backup_path(&mut arena, &[0, 1], -0.2, Some((0, 0.25)));
        assert!((arena[0].chance_visited_mass - 0.25).abs() < 1e-12);
        assert!((arena[0].chance_weighted_value - 0.075).abs() < 1e-12);
        assert!((ol_chance_value_p0(&arena, 0) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn one_reveal_sampled_backup_is_the_realized_monte_carlo_mean() {
        let mut arena = vec![
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
        ];
        arena[0].chance_backup = OLChanceBackup::Sampled;
        arena[0].chance_children = vec![
            test_chance_child([1, 2, 3, 4], 0.99, 1),
            test_chance_child([5, 6, 7, 8], 0.01, 2),
        ];
        ol_backup_path(&mut arena, &[0, 1], 0.0, Some((0, 0.99)));
        ol_backup_path(&mut arena, &[0, 2], 1.0, Some((0, 0.01)));

        assert_eq!(arena[0].visit_count, 2);
        assert!((arena[0].value_sum - 1.0).abs() < 1e-12);
        assert!((ol_chance_value_p0(&arena, 0) - 0.5).abs() < 1e-12);
        // The cache still records the counterfactual Hájek quantity for
        // diagnostics, but sampled selection/backup deliberately does not use it.
        assert!((arena[0].chance_visited_mass - 1.0).abs() < 1e-12);
        assert!((arena[0].chance_weighted_value - 0.01).abs() < 1e-12);
    }

    #[test]
    fn one_reveal_balanced_routing_completes_local_multiplicity_cycles() -> PyResult<()> {
        let panel = vec![
            OLChancePanelRow {
                row: vec![1, 2, 3, 4],
                probability: 2.0 / 3.0,
                multiplicity: 2,
            },
            OLChancePanelRow {
                row: vec![5, 6, 7, 8],
                probability: 1.0 / 3.0,
                multiplicity: 1,
            },
        ];
        let mut arena = vec![OLNode::new(1.0, (None, None))];
        ol_register_chance_support(
            &mut arena,
            0,
            OLChanceConfig {
                panel: &panel,
                backup: OLChanceBackup::Sampled,
                traversal: OLChanceTraversal::Balanced,
                schedule_seed: 20260808,
                draw_seed: 7,
                a1c: None,
                bootstrap_full_panel: false,
            },
            0,
        )?;

        let first_cycle: Vec<[u16; 4]> = (0..3)
            .map(|seed| ol_select_chance_row(&mut arena, 0, seed))
            .collect();
        assert_eq!(
            first_cycle
                .iter()
                .filter(|&&row| row == [1, 2, 3, 4])
                .count(),
            2
        );
        assert_eq!(
            first_cycle
                .iter()
                .filter(|&&row| row == [5, 6, 7, 8])
                .count(),
            1
        );
        assert_eq!(arena[0].chance_route_count, 3);
        let _first_of_next_cycle = ol_select_chance_row(&mut arena, 0, 99);
        assert_eq!(arena[0].chance_route_count, 4);
        Ok(())
    }

    #[test]
    fn one_reveal_diagnostics_report_unvisited_probability_mass() {
        let mut arena = vec![
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
        ];
        arena[0].visit_count = 3;
        arena[2].visit_count = 3;
        arena[0].chance_children = vec![
            test_chance_child([1, 2, 3, 4], 0.25, 1),
            test_chance_child([5, 6, 7, 8], 0.75, 2),
        ];
        refresh_test_chance_cache(&mut arena, 0);
        let diagnostics = ol_chance_diagnostics(&arena);
        assert_eq!(diagnostics["chance_nodes"], 1.0);
        assert_eq!(diagnostics["support_outcomes"], 2.0);
        assert_eq!(diagnostics["visited_outcomes"], 1.0);
        assert_eq!(diagnostics["observation_visits"], 3.0);
        assert_eq!(diagnostics["chance_node_visits"], 3.0);
        assert_eq!(diagnostics["fully_visited_chance_nodes"], 0.0);
        assert!((diagnostics["mean_visited_probability_mass"] - 0.75).abs() < 1e-12);
        assert!((diagnostics["mean_unvisited_probability_mass"] - 0.25).abs() < 1e-12);
        assert!((diagnostics["visit_weighted_visited_probability_mass"] - 0.75).abs() < 1e-12);
        assert!((diagnostics["visit_weighted_unvisited_probability_mass"] - 0.25).abs() < 1e-12);
        assert_eq!(diagnostics["min_visits_per_outcome"], 0.0);
        assert_eq!(diagnostics["max_visits_per_outcome"], 3.0);
    }

    #[test]
    fn one_reveal_visit_q_rank_diagnostic_uses_chooser_frame() {
        let mut arena = vec![
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
            OLNode::new(1.0, (None, None)),
        ];
        arena[0].children = vec![(1, 1), (2, 2), (3, 3)];
        arena[1].visit_count = 10;
        arena[1].chance_chooser_actor = Some(1);
        arena[1].chance_children = vec![test_chance_child([1, 2, 3, 4], 1.0, 4)];
        arena[4].visit_count = 10;
        arena[4].value_sum = -10.0;
        refresh_test_chance_cache(&mut arena, 1);
        arena[2].visit_count = 5;
        arena[2].chance_chooser_actor = Some(1);
        arena[2].chance_children = vec![test_chance_child([5, 6, 7, 8], 1.0, 5)];
        arena[5].visit_count = 5;
        arena[5].value_sum = -2.5;
        refresh_test_chance_cache(&mut arena, 2);
        arena[3].visit_count = 1;
        arena[3].chance_chooser_actor = Some(1);
        arena[3].chance_children = vec![test_chance_child([9, 10, 11, 12], 1.0, 6)];
        arena[6].visit_count = 1;
        arena[6].value_sum = 0.0;
        refresh_test_chance_cache(&mut arena, 3);

        let diagnostics = ol_chance_diagnostics(&arena);
        assert_eq!(diagnostics["chance_action_visit_q_rank_parent_groups"], 1.0);
        assert_eq!(diagnostics["chance_action_visit_q_rank_actions"], 3.0);
        assert!((diagnostics["chance_action_visit_q_rank_spearman_mean"] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn one_reveal_encoder_is_bit_identical_under_hidden_deck_permutation() -> PyResult<()> {
        let state = new_game(20260808, true, true);
        let actor = state.actor()?;
        let (my_before, opp_before, flat_before) = state.encode_arrays(actor)?;
        let mut permuted = state.cloned();
        permuted.deck.reverse();
        assert_ne!(state.deck, permuted.deck);
        let (my_after, opp_after, flat_after) = permuted.encode_arrays(actor)?;
        assert_eq!(my_before.as_slice(), my_after.as_slice());
        assert_eq!(opp_before.as_slice(), opp_after.as_slice());
        assert_eq!(flat_before.as_slice(), flat_after.as_slice());
        Ok(())
    }

    #[test]
    fn one_reveal_materialization_uses_distinct_public_observation_nodes() -> PyResult<()> {
        let mut state = new_game(17, true, true);
        for _ in 0..3 {
            let (_idx, placement, pick) = state.legal_actions_indexed()[0];
            state = state.step(placement, pick)?;
        }
        let action = {
            let (_idx, placement, pick) = state.legal_actions_indexed()[0];
            (placement, pick)
        };
        assert!(<Kingdomino as search::Game>::is_stochastic(&state, action));
        let panel = ol_one_reveal_panel(&state.deck, 1, 1, 123)?;
        assert_eq!(panel.iter().map(|x| x.multiplicity).sum::<usize>(), 11);
        let mut arena = vec![OLNode::new(1.0, action)];
        ol_register_chance_support(
            &mut arena,
            0,
            OLChanceConfig {
                panel: &panel,
                backup: OLChanceBackup::Hajek,
                traversal: OLChanceTraversal::Iid,
                schedule_seed: 9,
                draw_seed: 10,
                a1c: None,
                bootstrap_full_panel: false,
            },
            state.actor()?,
        )?;
        assert_eq!(arena[0].chance_children.len(), panel.len());
        assert_eq!(
            arena.len(),
            1,
            "support registration must allocate no subtrees"
        );
        let rows: HashSet<[u16; 4]> = arena[0].chance_children.iter().map(|x| x.row).collect();
        assert_eq!(rows.len(), panel.len());
        let first_state =
            redeterminize_with_first_row(&state, &panel[0].row, 1)?.step(action.0, action.1)?;
        let first_actor = first_state.actor()?;
        let (first_id, first_probability) =
            ol_route_chance_observation(&mut arena, 0, &first_state, first_actor)?;
        assert!((first_probability - panel[0].probability).abs() < 1e-12);
        assert_eq!(arena.len(), 2);
        assert_eq!(
            ol_route_chance_observation(&mut arena, 0, &first_state, first_actor)?.0,
            first_id,
            "the same public row must reuse its observation subtree"
        );
        let second_state =
            redeterminize_with_first_row(&state, &panel[1].row, 2)?.step(action.0, action.1)?;
        let second_actor = second_state.actor()?;
        let (second_id, _) =
            ol_route_chance_observation(&mut arena, 0, &second_state, second_actor)?;
        assert_ne!(first_id, second_id);
        assert_eq!(arena.len(), 3);
        assert!(
            (arena[0]
                .chance_children
                .iter()
                .map(|x| x.probability)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-12
        );
        Ok(())
    }

    #[test]
    fn deck8_exhaustive_panel_is_closed_and_bootstraps_without_fake_visits() -> PyResult<()> {
        // Drive a legal game to the last action before the deck=8 reveal. Using
        // legal_actions_indexed keeps this independent of hand-written board data.
        let mut state = new_game(20260810, true, true);
        let action = loop {
            let actions = state.legal_actions_indexed();
            assert!(!actions.is_empty());
            if state.deck.len() == 8
                && <Kingdomino as search::Game>::is_stochastic(&state, (actions[0].1, actions[0].2))
            {
                break (actions[0].1, actions[0].2);
            }
            state = state.step(actions[0].1, actions[0].2)?;
        };

        let panel = ol_one_reveal_panel(&state.deck, 1, 70, 99)?;
        assert_eq!(panel.len(), 70);
        assert!(
            panel
                .iter()
                .all(|outcome| (outcome.probability - 1.0 / 70.0).abs() < 1e-12)
        );

        let mut arena = vec![OLNode::new(1.0, (None, None))];
        arena[0].is_expanded = true;
        arena[0].visit_count = 1;
        let mut action_node = 0;
        for (i, &(action_index, placement, pick)) in
            state.legal_actions_indexed().iter().enumerate()
        {
            let child_id = arena.len() as u32;
            arena.push(OLNode::new(1.0, (placement, pick)));
            arena[0].children.push((action_index, child_id));
            if i == 0 {
                action_node = child_id;
            }
        }

        let config = OLChanceConfig {
            panel: &panel,
            backup: OLChanceBackup::Sampled,
            traversal: OLChanceTraversal::Balanced,
            schedule_seed: 123,
            draw_seed: 456,
            a1c: None,
            bootstrap_full_panel: true,
        };
        let mut fallback_count = 0;
        let mut missing_child_count = 0;
        let (path, _actors, _leaf, chance_step, request) = ol_descend(
            &mut arena,
            0,
            state.cloned(),
            0.0,
            1.5,
            &mut fallback_count,
            &mut missing_child_count,
            None,
            Some(config),
        )?;
        let request = request.expect("production descent must request panel admission");
        assert_eq!(request.chance_node_id, action_node);
        assert_eq!(arena[action_node as usize].chance_children.len(), 70);
        assert_eq!(arena[action_node as usize].visit_count, 0);

        // Install synthetic conditional bootstrap values in row order. This is
        // the same probability mean BatchedMCTS::update publishes after it has
        // validated all 70 inference results.
        let mut expected = 0.0;
        for (i, outcome) in panel.iter().enumerate() {
            let revealed = redeterminize_with_first_row(&state, &outcome.row, i as u64 + 1)?
                .step(action.0, action.1)?;
            let actor = revealed.actor()?;
            let (observation_id, probability) =
                ol_route_chance_observation(&mut arena, action_node, &revealed, actor)?;
            let value = i as f64 / 69.0;
            arena[observation_id as usize].bootstrap_value0 = Some(value);
            expected += probability * value;
        }
        {
            let chance = &mut arena[action_node as usize];
            chance.chance_backup = OLChanceBackup::PanelMean;
            chance.chance_visited_mass = 1.0;
            chance.chance_weighted_value = expected;
        }
        assert!((ol_chance_value_p0(&arena, action_node) - 0.5).abs() < 1e-12);
        assert_eq!(arena[action_node as usize].visit_count, 0);
        assert!(
            arena[action_node as usize]
                .chance_children
                .iter()
                .all(|outcome| arena[outcome.node_id.unwrap() as usize].visit_count == 0)
        );

        // The shared A1c/production commit helper must preserve an observation
        // that was already searched instead of overwriting it with bootstrap.
        let path_observation = path.last().copied().unwrap();
        let (prior_observation, prior_probability) = arena[action_node as usize]
            .chance_children
            .iter()
            .map(|outcome| (outcome.node_id.unwrap(), outcome.probability))
            .find(|(node_id, _)| *node_id != path_observation)
            .expect("the 70-row panel has an observation outside the test path");
        let prior_bootstrap = arena[prior_observation as usize].bootstrap_value0.unwrap();
        arena[prior_observation as usize].visit_count = 2;
        arena[prior_observation as usize].value_sum = -0.5;
        let expected_with_prior = expected + prior_probability * (-0.25 - prior_bootstrap);
        assert!((ol_panel_weighted_value(&arena, action_node) - expected_with_prior).abs() < 1e-12);
        arena[prior_observation as usize].visit_count = 0;
        arena[prior_observation as usize].value_sum = 0.0;
        assert!((ol_panel_weighted_value(&arena, action_node) - expected).abs() < 1e-12);

        // A real simulation backup creates exactly one visit and replaces only
        // its realized row's bootstrap estimate inside the exact panel mean.
        ol_backup_path(&mut arena, &path, -0.25, chance_step);
        assert_eq!(arena[action_node as usize].visit_count, 1);
        let visited = path.last().copied().unwrap();
        assert_eq!(arena[visited as usize].visit_count, 1);
        assert_eq!(arena[0].visit_count, 2);
        Ok(())
    }
}

#[cfg(test)]
mod pick_pos_tests {
    use super::*;

    /// Minimal RustGameState carrying only the fields pick_positions reads
    /// (phase + next_claims); the rest are placeholder.
    fn mk_state(phase: u8, next_claims: Vec<(u8, u16)>) -> RustGameState {
        RustGameState {
            boards: [RustBoard::new(7, 7), RustBoard::new(7, 7)],
            deck: Vec::new(),
            current_row: Vec::new(),
            pending_claims: Vec::new(),
            next_claims,
            phase,
            actor_index: 0,
            initial_pick_count: 0,
            start_player: 0,
            harmony: true,
            middle_kingdom: true,
            discards: [0, 0],
        }
    }

    #[test]
    fn pick_positions_initial_selection_all_zero() {
        // INITIAL_SELECTION returns all 0.0 even with committed claims (opening
        // claims are not next-round tempo signals).
        let s = mk_state(INITIAL_SELECTION, vec![(0, 12)]);
        assert_eq!(pick_positions(&s, 0), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(pick_positions(&s, 1), [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn pick_positions_two_committed() {
        // P0 committed domino 10, P1 committed domino 20.
        // From P0's view: pos0=+1 (did 10 is P0), pos1=-1 (did 20 is P1).
        let s = mk_state(PLACE_AND_SELECT, vec![(0, 10), (1, 20)]);
        assert_eq!(pick_positions(&s, 0), [1.0, -1.0, 0.0, 0.0]);
        // Perspective flip for P1.
        assert_eq!(pick_positions(&s, 1), [-1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn pick_positions_fully_committed_sums_to_zero() {
        // Four claims (append order deliberately scrambled); pick_positions
        // sorts by domino_id ascending: 5(P0),20(P1),40(P0),45(P1).
        let s = mk_state(PLACE_AND_SELECT, vec![(1, 45), (0, 5), (1, 20), (0, 40)]);
        let p0 = pick_positions(&s, 0);
        assert_eq!(p0, [1.0, -1.0, 1.0, -1.0]);
        assert!(
            (p0.iter().sum::<f32>()).abs() < 1e-6,
            "fully committed sums to 0"
        );
        // Perspective flip: P1 is the exact negation.
        let p1 = pick_positions(&s, 1);
        for k in 0..4 {
            assert_eq!(p1[k], -p0[k]);
        }
    }
}

#[pymodule]
mod kingdomino_rust {
    use super::*;

    #[pymodule_export]
    use super::d4_augment;

    #[pymodule_export]
    use super::d4_augment_mask;

    #[pymodule_export]
    use super::d4_inverse_transform_id;

    #[pymodule_export]
    use super::denial_tree::denial_forced_tree;

    #[pymodule_export]
    use super::RustBoard;

    #[pymodule_export]
    use super::RustGameState;

    #[pymodule_export]
    use super::AdvisorSearchHandle;

    #[pymodule_export]
    use super::SearchEngine;

    #[pymodule_export]
    use super::RustSearch;

    #[pymodule_export]
    use super::OperationalSearchReport;

    #[pymodule_export]
    use super::NnueEvaluator;

    #[pymodule_export]
    use super::sparse_nnue::SparseNnueEvaluator;

    #[pymodule_export]
    use super::sparse_nnue::QuantizedSparseNnueEvaluator;

    #[pymodule_export]
    use super::RustMCTS;

    #[pymodule_export]
    use super::BatchedMCTS;

    /// (terrain_a, crowns_a, terrain_b, crowns_b) for a domino id (1..=48).
    /// Exposed so the equivalence test can verify the Rust table against
    /// Python's DOMINOES directly.
    #[pyfunction]
    fn domino_halves(id: u16) -> (u8, u8, u8, u8) {
        super::dom(id)
    }

    /// Frozen NNUE feature schema identifiers shared with the Python reference.
    /// Derived-feature artifacts and exported models persist both hashes.
    #[pyfunction]
    fn nnue_schema_info() -> (usize, usize, &'static str, &'static str) {
        (
            super::nnue_features::CORE_SIZE,
            super::nnue_features::SUMMARY_SIZE,
            super::nnue_features::CORE_SCHEMA_HASH,
            super::nnue_features::SUMMARY_SCHEMA_HASH,
        )
    }

    /// Deterministic redeterminization seed used by BatchedMCTS for a move.
    #[pyfunction]
    fn batched_det_seed(game_seed: u64, move_num: usize) -> u64 {
        super::det_seed(game_seed, move_num)
    }

    /// Deterministic Rust-side game constructor used by BatchedMCTS.
    #[pyfunction]
    #[pyo3(signature = (seed, harmony=true, middle_kingdom=true))]
    fn batched_new_game(seed: u64, harmony: bool, middle_kingdom: bool) -> RustGameState {
        super::new_game(seed, harmony, middle_kingdom)
    }

    /// Convert a raw score margin (s0 - s1) to the training value formula.
    /// Exposed for formula tests; exact solving uses the tiebreak-aware converter.
    #[pyfunction]
    #[pyo3(signature = (margin, score_scale=160.0, margin_gain=2.0, alpha=0.5))]
    fn margin_to_training_value(
        margin: f64,
        score_scale: f64,
        margin_gain: f64,
        alpha: f64,
    ) -> f64 {
        super::margin_to_training_value(margin, score_scale, margin_gain, alpha)
    }

    /// Exact minimax endgame solve for states with no chance branching:
    /// PLACE_AND_SELECT with deck length 0 or 4, or FINAL_PLACEMENT with deck
    /// length 0. When deck length is 4, the next row is forced to be exactly
    /// those four tiles, so no public bag expectation is needed. Returns
    /// (value_player0, solved_exactly, elapsed_secs). Falls back with solved=false
    /// if the state still has chance branching or the per-position wall-clock
    /// budget `max_secs` is exceeded.
    #[pyfunction]
    #[pyo3(signature = (state, max_secs=3.0, score_scale=160.0, margin_gain=2.0, alpha=0.5))]
    fn exact_endgame_value_no_chance(
        state: &RustGameState,
        max_secs: f64,
        score_scale: f64,
        margin_gain: f64,
        alpha: f64,
    ) -> PyResult<(f64, bool, f64)> {
        if state.phase == GAME_OVER {
            return Ok((
                super::terminal_search_value(state, score_scale, margin_gain, alpha),
                true,
                0.0,
            ));
        }
        if !super::is_no_chance_endgame_state(state) {
            return Ok((0.0, false, 0.0));
        }
        // YBW parallel alpha-beta (OPT-2/3/4/6): first child serial to set a
        // bound, remaining children in parallel, all sharing one wall-clock
        // deadline. solved=false only if the deadline was hit. See
        // solve_endgame_ab_parallel for budget semantics.
        let start = std::time::Instant::now();
        let deadline = start + std::time::Duration::from_secs_f64(max_secs);
        match super::solve_endgame_ab_parallel(
            state,
            deadline,
            super::SolverOrderMode::Lookahead2Clustered,
        )? {
            Some(solver_utility) => {
                let value = super::solver_utility_to_training_value(
                    solver_utility,
                    score_scale,
                    margin_gain,
                    alpha,
                );
                Ok((value, true, start.elapsed().as_secs_f64()))
            }
            None => Ok((0.0, false, start.elapsed().as_secs_f64())),
        }
    }

    /// Transposition-rate diagnostic for a no-chance endgame root.
    ///
    /// Runs the SAME alpha-beta traversal as the production solver while
    /// hashing every interior state, and reports how much of the pruned
    /// search re-enters already-visited states — the evidence base for a
    /// within-solve transposition table.
    ///
    /// `per_child_full_window=True` mirrors the production training-path root
    /// solve (`solve_root_exact_cached`): every root child is solved with a
    /// full window, sharing ONE seen-set, so cross-child state reuse counts as
    /// duplicates. `False` measures a single serial full-window root solve.
    ///
    /// Returns (interior_nodes, terminal_nodes, distinct_states, dup_visits,
    /// completed). On deadline the partial counts are still returned with
    /// completed=False (dup shares of a partial traversal remain meaningful).
    #[pyfunction]
    #[pyo3(signature = (state, max_secs=10.0, ordering="lookahead2_clustered", per_child_full_window=true))]
    fn measure_endgame_transpositions(
        state: &RustGameState,
        max_secs: f64,
        ordering: &str,
        per_child_full_window: bool,
    ) -> PyResult<(u64, u64, u64, u64, bool)> {
        if state.phase == GAME_OVER || !super::is_no_chance_endgame_state(state) {
            return Err(PyValueError::new_err(
                "requires a non-terminal no-chance endgame state (deck in {0,4})",
            ));
        }
        let mode = super::SolverOrderMode::from_str(ordering)?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(max_secs);
        let mut stats = super::TranspoStats::default();
        let mut buf: Vec<u8> = Vec::with_capacity(1024);
        let completed = if per_child_full_window {
            let mut legal = state.legal_actions_indexed();
            super::order_legal_for_solver_at_depth(state, &mut legal, mode, 0)?;
            let mut done = true;
            for &(_idx, p, pk) in &legal {
                let child = state.step(p, pk)?;
                if super::solve_endgame_ab_transpo(
                    &child,
                    deadline,
                    super::MARGIN_LO,
                    super::MARGIN_HI,
                    mode,
                    0,
                    &mut stats,
                    &mut buf,
                )?
                .is_none()
                {
                    done = false;
                    break;
                }
            }
            done
        } else {
            super::solve_endgame_ab_transpo(
                state,
                deadline,
                super::MARGIN_LO,
                super::MARGIN_HI,
                mode,
                0,
                &mut stats,
                &mut buf,
            )?
            .is_some()
        };
        Ok((
            stats.interior,
            stats.terminals,
            stats.seen.len() as u64,
            stats.dup_visits,
            completed,
        ))
    }

    /// Benchmark the PRODUCTION root solve (`solve_root_exact`) under a given
    /// policy-label mode. Unlike `measure_endgame_tree` (value-only YBW solve),
    /// this measures the training path: per-child solves for policy targets.
    ///
    /// Returns (root_value_training_frame, solved, elapsed_secs, n_children,
    /// n_clamped). `n_clamped` is 0 for mode="exact"; for "soft_clamp" /
    /// "argmax_ties" it is the number of children priced at the clamp value
    /// instead of solved exactly.
    #[pyfunction]
    #[pyo3(signature = (state, max_secs=10.0, score_scale=160.0, margin_gain=2.0, alpha=0.5,
                        ordering="lookahead2_clustered", policy_mode="soft_clamp", clamp_delta=10.0))]
    #[allow(clippy::too_many_arguments)]
    fn measure_root_exact(
        state: &RustGameState,
        max_secs: f64,
        score_scale: f64,
        margin_gain: f64,
        alpha: f64,
        ordering: &str,
        policy_mode: &str,
        clamp_delta: f64,
    ) -> PyResult<(f64, bool, f64, u32, u32)> {
        if state.phase == GAME_OVER || !super::is_no_chance_endgame_state(state) {
            return Err(PyValueError::new_err(
                "requires a non-terminal no-chance endgame state (deck in {0,4})",
            ));
        }
        let order_mode = super::SolverOrderMode::from_str(ordering)?;
        let pmode = super::ExactPolicyMode::from_str(policy_mode)?;
        let start = std::time::Instant::now();
        let deadline = start + std::time::Duration::from_secs_f64(max_secs);
        match super::solve_root_exact(
            state,
            deadline,
            score_scale,
            margin_gain,
            alpha,
            order_mode,
            pmode,
            clamp_delta,
        )? {
            Some(result) => {
                let actor = state.actor()?;
                let root_value = if actor == 0 {
                    result
                        .child_values
                        .iter()
                        .map(|&(_, v)| v)
                        .fold(f64::NEG_INFINITY, f64::max)
                } else {
                    result
                        .child_values
                        .iter()
                        .map(|&(_, v)| v)
                        .fold(f64::INFINITY, f64::min)
                };
                Ok((
                    root_value,
                    true,
                    start.elapsed().as_secs_f64(),
                    result.child_values.len() as u32,
                    result.n_clamped as u32,
                ))
            }
            None => Ok((0.0, false, start.elapsed().as_secs_f64(), 0, 0)),
        }
    }

    /// Advisor open-loop single-root search (Rust port of the web advisor's
    /// OpenLoopMCTS loop). Runs `n_sims` simulations from `state` with a fresh
    /// deck redeterminization per simulation, evaluating leaves through the
    /// BatchedEvaluator-contract callable in `leaf_batch`-sized waves (virtual
    /// loss). Returns (children, root_value_player0) where children is
    /// [(joint_index, visit_count, value_sum_player0, prior)] for every root
    /// child, ascending by joint index.
    ///
    /// No exact-endgame hook: route terminal-adjacent ROOTS (deck <= 4) to the
    /// exact solver before calling this (interior deck<=4 nodes under a deeper
    /// root are determinization-dependent — same gate as training).
    #[pyfunction]
    #[pyo3(signature = (state, evaluator, n_sims, dirichlet_alpha=0.3, dirichlet_eps=0.0,
                        fpu=0.0, cpuct=1.5, seed=0, leaf_batch=8, virtual_loss=1,
                        score_scale=160.0, margin_gain=2.0, alpha=0.0,
                        pick_floor_frac=0.0, pick_floor_min_depth=0,
                        pick_floor_max_depth=0, pick_floor_min_visits=16,
                        chance_exposure=0, chance_enum_max_rows=70,
                        chance_backup="hajek", chance_traversal="iid"))]
    #[allow(clippy::too_many_arguments)]
    fn advisor_open_loop_search<'py>(
        py: Python<'py>,
        state: &RustGameState,
        evaluator: Bound<'py, PyAny>,
        n_sims: usize,
        dirichlet_alpha: f64,
        dirichlet_eps: f64,
        fpu: f64,
        cpuct: f64,
        seed: u64,
        leaf_batch: usize,
        virtual_loss: i32,
        score_scale: f64,
        margin_gain: f64,
        alpha: f64,
        pick_floor_frac: f64,
        pick_floor_min_depth: usize,
        pick_floor_max_depth: usize,
        pick_floor_min_visits: i32,
        chance_exposure: usize,
        chance_enum_max_rows: u64,
        chance_backup: &str,
        chance_traversal: &str,
    ) -> PyResult<(Vec<(u16, i32, f64, f64)>, f64)> {
        if state.phase == GAME_OVER {
            return Err(PyValueError::new_err("Cannot search from a terminal state"));
        }
        // Wider than the BatchedMCTS bound (which stops well under 1/5 to keep
        // policy targets meritocratic).  The advisor records no targets, so the
        // Phase A study is free to run a near-uniform root arm; 0.25 still
        // guarantees a group cannot be floored above an even 4-way split.
        if !(0.0..=0.25).contains(&pick_floor_frac) {
            return Err(PyValueError::new_err(format!(
                "pick_floor_frac must be in [0, 0.25], got {}",
                pick_floor_frac
            )));
        }
        if pick_floor_frac > 0.0 && pick_floor_min_depth > pick_floor_max_depth {
            return Err(PyValueError::new_err(format!(
                "pick_floor_min_depth ({}) must be <= pick_floor_max_depth ({})",
                pick_floor_min_depth, pick_floor_max_depth
            )));
        }
        if chance_exposure > 0 && chance_enum_max_rows == 0 {
            return Err(PyValueError::new_err(
                "chance_enum_max_rows must be > 0 when chance_exposure is enabled",
            ));
        }
        let chance_backup = super::ol_parse_chance_backup(chance_backup)?;
        let chance_traversal = super::ol_parse_chance_traversal(chance_traversal)?;
        // frac == 0 => None => byte-identical to the pre-Phase-A path.
        let pick_floor = if pick_floor_frac > 0.0 {
            Some(super::PickFloor {
                frac: pick_floor_frac,
                min_depth: pick_floor_min_depth,
                max_depth: pick_floor_max_depth,
                min_visits: pick_floor_min_visits,
            })
        } else {
            None
        };
        let root_state = state.cloned();
        let ev: Py<PyAny> = evaluator.unbind();
        let output = py.detach(move || {
            super::advisor_open_loop_search_impl(
                &root_state,
                &ev,
                n_sims,
                dirichlet_alpha,
                dirichlet_eps,
                fpu,
                cpuct,
                seed,
                leaf_batch,
                virtual_loss,
                score_scale,
                margin_gain,
                alpha,
                pick_floor,
                chance_exposure,
                chance_enum_max_rows,
                chance_backup,
                chance_traversal,
                None,
                None,
            )
        })?;
        Ok((output.children, output.root_value0))
    }

    /// Diagnostic A1 entry point. It uses the same advisor open-loop engine but
    /// also reports realized fixed-support coverage. The incumbent is exposure
    /// zero; positive exposure splits only the first future reveal.
    #[pyfunction]
    #[pyo3(signature = (state, evaluator, n_sims, chance_exposure=0,
                        chance_enum_max_rows=70, fpu=0.0, cpuct=1.5, seed=0,
                        leaf_batch=8, virtual_loss=1, score_scale=160.0,
                        margin_gain=2.0, alpha=0.0, chance_backup="hajek",
                        chance_traversal="iid", chance_panel_mode="lazy",
                         chance_panel_sampling="balanced", chance_init_visits=32,
                         chance_widening_c=0.25,
                         chance_init_max_fraction=0.25, nn_eval_budget=0))]
    #[allow(clippy::too_many_arguments)]
    fn advisor_one_reveal_search<'py>(
        py: Python<'py>,
        state: &RustGameState,
        evaluator: Bound<'py, PyAny>,
        n_sims: usize,
        chance_exposure: usize,
        chance_enum_max_rows: u64,
        fpu: f64,
        cpuct: f64,
        seed: u64,
        leaf_batch: usize,
        virtual_loss: i32,
        score_scale: f64,
        margin_gain: f64,
        alpha: f64,
        chance_backup: &str,
        chance_traversal: &str,
        chance_panel_mode: &str,
        chance_panel_sampling: &str,
        chance_init_visits: usize,
        chance_widening_c: f64,
        chance_init_max_fraction: f64,
        nn_eval_budget: usize,
    ) -> PyResult<(Vec<(u16, i32, f64, f64)>, f64, HashMap<String, f64>)> {
        if state.phase == GAME_OVER {
            return Err(PyValueError::new_err("Cannot search from a terminal state"));
        }
        if chance_exposure > 0 && chance_enum_max_rows == 0 {
            return Err(PyValueError::new_err(
                "chance_enum_max_rows must be > 0 when chance_exposure is enabled",
            ));
        }
        let chance_backup = super::ol_parse_chance_backup(chance_backup)?;
        let chance_traversal = super::ol_parse_chance_traversal(chance_traversal)?;
        let a1c_options = match chance_panel_mode {
            "lazy" => None,
            "a1c" => {
                if chance_exposure == 0 {
                    return Err(PyValueError::new_err(
                        "A1c panel mode requires chance_exposure > 0",
                    ));
                }
                if !(chance_widening_c > 0.0 && chance_widening_c.is_finite()) {
                    return Err(PyValueError::new_err(
                        "chance_widening_c must be finite and greater than zero",
                    ));
                }
                if !(chance_init_max_fraction > 0.0
                    && chance_init_max_fraction <= 1.0
                    && chance_init_max_fraction.is_finite())
                {
                    return Err(PyValueError::new_err(
                        "chance_init_max_fraction must be finite and in (0, 1]",
                    ));
                }
                Some(super::A1cSearchOptions {
                    sampling: super::a1c_parse_panel_sampling(chance_panel_sampling)?,
                    n_init: chance_init_visits,
                    widening_c: chance_widening_c,
                    max_initialization_fraction: chance_init_max_fraction,
                })
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "chance_panel_mode must be 'lazy' or 'a1c', got {chance_panel_mode:?}"
                )));
            }
        };
        let root_state = state.cloned();
        let ev: Py<PyAny> = evaluator.unbind();
        let output = py.detach(move || {
            super::advisor_open_loop_search_impl(
                &root_state,
                &ev,
                n_sims,
                0.3,
                0.0,
                fpu,
                cpuct,
                seed,
                leaf_batch,
                virtual_loss,
                score_scale,
                margin_gain,
                alpha,
                None,
                chance_exposure,
                chance_enum_max_rows,
                chance_backup,
                chance_traversal,
                a1c_options,
                (nn_eval_budget > 0).then_some(nn_eval_budget),
            )
        })?;
        Ok((output.children, output.root_value0, output.diagnostics))
    }

    /// Test seam for the A1c panel planner. Production search integration uses
    /// the same helper, but keeping this deterministic export makes cycle
    /// balance, IID width matching and seed stability reviewable from Python.
    #[pyfunction]
    #[pyo3(signature = (deck, max_cycles, sampling="balanced", seed=0))]
    fn debug_a1c_panel_cycles(
        deck: Vec<u16>,
        max_cycles: usize,
        sampling: &str,
        seed: u64,
    ) -> PyResult<Vec<Vec<Vec<u16>>>> {
        let sampling = super::a1c_parse_panel_sampling(sampling)?;
        Ok(
            super::a1c_one_reveal_cycles(&deck, max_cycles, sampling, seed)?
                .into_iter()
                .map(|cycle| cycle.into_iter().map(|row| row.to_vec()).collect())
                .collect(),
        )
    }

    /// Test seam for A1c's visit-width schedule and cumulative NN-work guard.
    /// The returned decision charges the entire proposed batch before testing
    /// the cap, matching the atomic admission rule used by the search design.
    #[pyfunction]
    #[pyo3(signature = (
        visits,
        n_init,
        max_cycles,
        widening_c,
        initialization_nn_evals,
        total_nn_evals,
        additional_nn_evals,
        max_initialization_fraction=0.25
    ))]
    #[allow(clippy::too_many_arguments)]
    fn debug_a1c_admission(
        visits: usize,
        n_init: usize,
        max_cycles: usize,
        widening_c: f64,
        initialization_nn_evals: usize,
        total_nn_evals: usize,
        additional_nn_evals: usize,
        max_initialization_fraction: f64,
    ) -> PyResult<(usize, bool)> {
        if !(widening_c > 0.0 && widening_c.is_finite()) {
            return Err(PyValueError::new_err(
                "widening_c must be finite and greater than zero",
            ));
        }
        if !(max_initialization_fraction > 0.0
            && max_initialization_fraction <= 1.0
            && max_initialization_fraction.is_finite())
        {
            return Err(PyValueError::new_err(
                "max_initialization_fraction must be finite and in (0, 1]",
            ));
        }
        if initialization_nn_evals > total_nn_evals {
            return Err(PyValueError::new_err(
                "initialization_nn_evals cannot exceed total_nn_evals",
            ));
        }
        Ok((
            super::a1c_target_cycles(visits, n_init, max_cycles, widening_c),
            super::a1c_initialization_within_budget(
                initialization_nn_evals,
                total_nn_evals,
                additional_nn_evals,
                max_initialization_fraction,
            ),
        ))
    }

    /// Count exact minimax nodes for a no-chance endgame, with the same
    /// conservative max_nodes cap as exact_endgame_value_no_chance.
    #[pyfunction]
    #[pyo3(signature = (state, max_nodes=50_000))]
    fn count_endgame_nodes_no_chance(state: &RustGameState, max_nodes: u64) -> PyResult<u64> {
        if state.phase == GAME_OVER {
            return Ok(0);
        }
        if !super::is_no_chance_endgame_state(state) {
            return Ok(max_nodes.saturating_add(1));
        }
        super::exact_count_no_chance_bounded(state, max_nodes)
    }

    /// Compatibility alias for the original deck-empty export. It now also
    /// accepts deck length 4, because that is likewise no-chance: all four
    /// hidden tiles form the next row.
    #[pyfunction]
    #[pyo3(signature = (state, max_secs=3.0, score_scale=160.0, margin_gain=2.0, alpha=0.5))]
    fn exact_endgame_value_deck_empty(
        state: &RustGameState,
        max_secs: f64,
        score_scale: f64,
        margin_gain: f64,
        alpha: f64,
    ) -> PyResult<(f64, bool, f64)> {
        exact_endgame_value_no_chance(state, max_secs, score_scale, margin_gain, alpha)
    }

    /// Compatibility alias for the original deck-empty count export.
    #[pyfunction]
    #[pyo3(signature = (state, max_nodes=50_000))]
    fn count_endgame_nodes_deck_empty(state: &RustGameState, max_nodes: u64) -> PyResult<u64> {
        count_endgame_nodes_no_chance(state, max_nodes)
    }

    #[pyfunction]
    #[pyo3(signature = (state, domino_id, actor))]
    fn debug_opponent_denial_score(
        state: &RustGameState,
        domino_id: u16,
        actor: u8,
    ) -> PyResult<i32> {
        if actor > 1 {
            return Err(PyValueError::new_err("actor must be 0 or 1"));
        }
        let opponent = (1 - actor) as usize;
        Ok(super::opponent_denial_score(
            domino_id,
            &super::terrain_counts(&state.boards[opponent]),
        ))
    }

    #[pyfunction]
    #[pyo3(signature = (state, ordering="combined"))]
    fn debug_ordered_legal_indices(state: &RustGameState, ordering: &str) -> PyResult<Vec<u16>> {
        let mode = super::SolverOrderMode::from_str(ordering)?;
        let mut legal = state.legal_actions_indexed();
        super::order_legal_for_solver_at_depth(state, &mut legal, mode, 0)?;
        Ok(legal.into_iter().map(|(idx, _p, _pk)| idx).collect())
    }

    #[pyfunction]
    #[pyo3(signature = (state, max_secs=3.0, score_scale=160.0, margin_gain=2.0, alpha=0.5, ordering="combined", parallel=true))]
    fn exact_endgame_value_no_chance_ordered(
        state: &RustGameState,
        max_secs: f64,
        score_scale: f64,
        margin_gain: f64,
        alpha: f64,
        ordering: &str,
        parallel: bool,
    ) -> PyResult<(f64, bool, f64)> {
        if state.phase == GAME_OVER {
            return Ok((
                super::terminal_search_value(state, score_scale, margin_gain, alpha),
                true,
                0.0,
            ));
        }
        if !super::is_no_chance_endgame_state(state) {
            return Ok((0.0, false, 0.0));
        }
        let mode = super::SolverOrderMode::from_str(ordering)?;
        let start = std::time::Instant::now();
        let deadline = start + std::time::Duration::from_secs_f64(max_secs);
        let raw = if parallel {
            super::solve_endgame_ab_parallel(state, deadline, mode)?
        } else {
            super::solve_endgame_ab(state, deadline, super::MARGIN_LO, super::MARGIN_HI, mode, 0)?
        };
        match raw {
            Some(solver_utility) => {
                let value = super::solver_utility_to_training_value(
                    solver_utility,
                    score_scale,
                    margin_gain,
                    alpha,
                );
                Ok((value, true, start.elapsed().as_secs_f64()))
            }
            None => Ok((0.0, false, start.elapsed().as_secs_f64())),
        }
    }

    /// Exact deck=0 FINAL_PLACEMENT value via independent per-player board
    /// maximization. Returns solved=false for PLACE_AND_SELECT deck=0 because
    /// final-row picks are still interactive there.
    #[pyfunction]
    #[pyo3(signature = (state, score_scale=160.0, margin_gain=2.0, alpha=0.5))]
    fn exact_deck0_final_value_separable(
        state: &RustGameState,
        score_scale: f64,
        margin_gain: f64,
        alpha: f64,
    ) -> PyResult<(f64, bool)> {
        match super::deck0_draft_dp::solve_deck0_final_placement_separable_utility(state)? {
            Some(solver_utility) => Ok((
                super::solver_utility_to_training_value(
                    solver_utility,
                    score_scale,
                    margin_gain,
                    alpha,
                ),
                true,
            )),
            None => Ok((0.0, false)),
        }
    }

    /// Exact deck=0 PLACE_AND_SELECT/FINAL_PLACEMENT value using a memoized
    /// draft DP with final-placement cutoff into independent board maximizers.
    /// `max_current_row_len=0` effectively means final-placement only.
    #[pyfunction]
    #[pyo3(signature = (
        state,
        max_current_row_len=4,
        max_secs=1.0,
        score_scale=160.0,
        margin_gain=2.0,
        alpha=0.5
    ))]
    fn exact_deck0_draft_value_dp(
        state: &RustGameState,
        max_current_row_len: usize,
        max_secs: f64,
        score_scale: f64,
        margin_gain: f64,
        alpha: f64,
    ) -> PyResult<(f64, bool, f64, u64, u64, u64, u64)> {
        let start = std::time::Instant::now();
        let deadline = start + std::time::Duration::from_secs_f64(max_secs);
        let mut cache: HashMap<EndgameKey, Option<f64>> = HashMap::new();
        let mut stats = super::deck0_draft_dp::Deck0DraftDpStats::default();
        match super::deck0_draft_dp::solve_deck0_draft_dp_utility(
            state,
            max_current_row_len,
            deadline,
            &mut cache,
            &mut stats,
        )? {
            Some(solver_utility) => Ok((
                super::solver_utility_to_training_value(
                    solver_utility,
                    score_scale,
                    margin_gain,
                    alpha,
                ),
                true,
                start.elapsed().as_secs_f64(),
                stats.nodes,
                stats.cache_hits,
                stats.final_cutoffs,
                stats.deadline_hits,
            )),
            None => Ok((
                0.0,
                false,
                start.elapsed().as_secs_f64(),
                stats.nodes,
                stats.cache_hits,
                stats.final_cutoffs,
                stats.deadline_hits,
            )),
        }
    }

    /// Benchmark-only probe: run open-loop MCTS from `state` with uniform priors
    /// and value-zero nonterminal leaves, optionally replacing deck=0 leaf values
    /// with exact minimax solves. This does not affect production training.
    #[pyfunction]
    #[pyo3(signature = (
        state,
        n_sims=1600,
        exact_deck0=false,
        leaf_max_secs=0.25,
        final_only=false,
        draft_k=-1,
        fpu=-0.2,
        cpuct=1.5,
        seed=0,
        score_scale=160.0,
        margin_gain=2.0,
        alpha=0.5,
        ordering="lookahead2_clustered",
        parallel=true
    ))]
    fn bench_open_loop_deck0_leaf_exact(
        state: &RustGameState,
        n_sims: usize,
        exact_deck0: bool,
        leaf_max_secs: f64,
        final_only: bool,
        draft_k: i32,
        fpu: f64,
        cpuct: f64,
        seed: u64,
        score_scale: f64,
        margin_gain: f64,
        alpha: f64,
        ordering: &str,
        parallel: bool,
    ) -> PyResult<Vec<f64>> {
        let mode = super::SolverOrderMode::from_str(ordering)?;
        let start = std::time::Instant::now();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut arena: Vec<OLNode> = vec![OLNode::new(1.0, (None, None))];
        super::ol_expand_uniform(&mut arena, 0, state)?;
        arena[0].visit_count = 1;

        let mut exact_cache: HashMap<EndgameKey, Option<f64>> = HashMap::new();
        let mut deck0_leaf_hits: u64 = 0;
        let mut deck0_unique_solves: u64 = 0;
        let mut deck0_cache_hits: u64 = 0;
        let mut deck0_timeouts: u64 = 0;
        let mut exact_solve_secs = 0.0;
        let mut terminal_leaf_hits: u64 = 0;
        let mut network_leaf_evals: u64 = 0;
        let mut fallback_count: u32 = 0;
        let mut missing_child_count: u32 = 0;
        let mut dp_stats_total = super::deck0_draft_dp::Deck0DraftDpStats::default();

        for _ in 0..n_sims {
            let det = state.redeterminize(Some(rng.r#gen::<u64>()));
            let (path, _actors, leaf_state, chance_step, a1c_admission_request) =
                super::ol_descend(
                    &mut arena,
                    0,
                    det,
                    fpu,
                    cpuct,
                    &mut fallback_count,
                    &mut missing_child_count,
                    None,
                    None,
                )?;
            debug_assert!(chance_step.is_none());
            debug_assert!(a1c_admission_request.is_none());
            let leaf = *path.last().unwrap();

            let v0 = if leaf_state.phase == GAME_OVER {
                terminal_leaf_hits += 1;
                super::terminal_search_value(&leaf_state, score_scale, margin_gain, alpha)
            } else if leaf_state.deck.is_empty()
                && (leaf_state.phase == PLACE_AND_SELECT || leaf_state.phase == FINAL_PLACEMENT)
            {
                deck0_leaf_hits += 1;
                let dp_enabled = draft_k >= 0;
                let dp_eligible = dp_enabled
                    && (leaf_state.phase == FINAL_PLACEMENT
                        || (leaf_state.phase == PLACE_AND_SELECT
                            && leaf_state.current_row.len() <= draft_k as usize));
                let old_eligible =
                    !dp_enabled && (!final_only || leaf_state.phase == FINAL_PLACEMENT);
                if exact_deck0 && (dp_eligible || old_eligible) {
                    let key = super::endgame_key(&leaf_state);
                    if let Some(cached) = exact_cache.get(&key) {
                        deck0_cache_hits += 1;
                        if let Some(solver_utility) = cached {
                            super::solver_utility_to_training_value(
                                *solver_utility,
                                score_scale,
                                margin_gain,
                                alpha,
                            )
                        } else {
                            network_leaf_evals += 1;
                            super::ol_expand_uniform(&mut arena, leaf, &leaf_state)?;
                            0.0
                        }
                    } else {
                        deck0_unique_solves += 1;
                        let solve_start = std::time::Instant::now();
                        let raw = if dp_enabled {
                            let deadline =
                                solve_start + std::time::Duration::from_secs_f64(leaf_max_secs);
                            let mut dp_stats = super::deck0_draft_dp::Deck0DraftDpStats::default();
                            let result = super::deck0_draft_dp::solve_deck0_draft_dp_utility(
                                &leaf_state,
                                draft_k.max(0) as usize,
                                deadline,
                                &mut exact_cache,
                                &mut dp_stats,
                            )?;
                            dp_stats_total.nodes += dp_stats.nodes;
                            dp_stats_total.cache_hits += dp_stats.cache_hits;
                            dp_stats_total.final_cutoffs += dp_stats.final_cutoffs;
                            dp_stats_total.deadline_hits += dp_stats.deadline_hits;
                            result
                        } else {
                            if let Some(solver_utility) =
                                super::deck0_draft_dp::solve_deck0_final_placement_separable_utility(&leaf_state)?
                            {
                                Some(solver_utility)
                            } else {
                                let deadline = solve_start
                                    + std::time::Duration::from_secs_f64(leaf_max_secs);
                                if parallel {
                                    super::solve_endgame_ab_parallel(&leaf_state, deadline, mode)?
                                } else {
                                    super::solve_endgame_ab(
                                        &leaf_state,
                                        deadline,
                                        super::MARGIN_LO,
                                        super::MARGIN_HI,
                                        mode,
                                        0,
                                    )?
                                }
                            }
                        };
                        exact_solve_secs += solve_start.elapsed().as_secs_f64();
                        if !dp_enabled {
                            exact_cache.insert(key, raw);
                        }
                        if let Some(solver_utility) = raw {
                            super::solver_utility_to_training_value(
                                solver_utility,
                                score_scale,
                                margin_gain,
                                alpha,
                            )
                        } else {
                            deck0_timeouts += 1;
                            network_leaf_evals += 1;
                            super::ol_expand_uniform(&mut arena, leaf, &leaf_state)?;
                            0.0
                        }
                    }
                } else {
                    network_leaf_evals += 1;
                    super::ol_expand_uniform(&mut arena, leaf, &leaf_state)?;
                    0.0
                }
            } else {
                network_leaf_evals += 1;
                super::ol_expand_uniform(&mut arena, leaf, &leaf_state)?;
                0.0
            };

            for &n in &path {
                let node = &mut arena[n as usize];
                node.visit_count += 1;
                node.value_sum += v0;
            }
        }

        Ok(vec![
            start.elapsed().as_secs_f64(),
            deck0_leaf_hits as f64,
            deck0_unique_solves as f64,
            deck0_cache_hits as f64,
            deck0_timeouts as f64,
            terminal_leaf_hits as f64,
            exact_solve_secs,
            network_leaf_evals as f64,
            fallback_count as f64,
            arena.len() as f64,
            dp_stats_total.nodes as f64,
            dp_stats_total.cache_hits as f64,
            dp_stats_total.final_cutoffs as f64,
            dp_stats_total.deadline_hits as f64,
        ])
    }
}
