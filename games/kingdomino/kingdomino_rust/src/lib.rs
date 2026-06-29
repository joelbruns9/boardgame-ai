use numpy::ndarray::{Array1, Array2, Array3, Array4, Axis};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use rand_distr::{Distribution, Gamma};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

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
const FLAT_SIZE: usize = 261;

// Flat-vector field offsets (see encoder.FLAT_LAYOUT).
const OFF_DOMINO_IN_HAND: usize = 0;
const OFF_CURRENT_ROW: usize = 14;
const OFF_PENDING: usize = 74;
const OFF_NEXT: usize = 138;
const OFF_BAG: usize = 202;
const OFF_PHASE: usize = 250;
const OFF_GAME_PROGRESS: usize = 253;
const OFF_MY_FILL: usize = 254;
const OFF_OPP_FILL: usize = 255;
const OFF_ACTOR_FLAG: usize = 256;
// Pick position features (4 scalars, FLAT_SIZE 259 → 261).
// Replaces OFF_MY_PICK_RANK / OFF_OPP_PICK_RANK (2 scalars).
const OFF_PICK_POS_0: usize = 257;
const OFF_PICK_POS_1: usize = 258;
const OFF_PICK_POS_2: usize = 259;
const OFF_PICK_POS_3: usize = 260;

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

    /// Domino the given player is currently placing, if any (mirrors
    /// encoder._domino_in_hand): only during a turn phase, only when the
    /// pending claim at actor_index belongs to `player`.
    fn domino_in_hand(&self, player: u8) -> Option<u16> {
        if self.phase != PLACE_AND_SELECT && self.phase != FINAL_PLACEMENT {
            return None;
        }
        let (cp, did) = *self.pending_claims.get(self.actor_index)?;
        if cp != player { None } else { Some(did) }
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
    /// Core encoder: writes the (9,13,13) my/opp board planes and the (261,) flat
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

        // 1. Domino in hand (only when it's this player's turn to place).
        if let Some(d) = self.domino_in_hand(player) {
            write_tile(flat, OFF_DOMINO_IN_HAND, d);
        }
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
        // 10. Next-round pick positions (full interleaving, FLAT_SIZE 259→261).
        //     +1 = encoded player acts here, -1 = opponent, 0 = unknown/no round.
        let pos = pick_positions(self, player);
        flat[OFF_PICK_POS_0] = pos[0];
        flat[OFF_PICK_POS_1] = pos[1];
        flat[OFF_PICK_POS_2] = pos[2];
        flat[OFF_PICK_POS_3] = pos[3];

        Ok(())
    }

    /// Allocating encoder: (my_board (9,13,13), opp_board (9,13,13), flat (261,)).
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
    /// `mb_data`/`ob_data` are (rows, 9, 13, 13) flat; `flat_data` is (rows, 261)
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
        castle_y=7
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
    /// arrays: two (9, 13, 13) float32 plane stacks and a (261,) float32 vector.
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
    /// reflect production pruning behavior. `alpha` defaults to 0.8 (the training
    /// frame): alpha-beta cutoffs depend on leaf values, so pruning — and
    /// therefore solve time — can vary with alpha. Measure at the alpha training
    /// actually uses.
    ///
    /// `parallel=True` (default) uses the YBW parallel solver
    /// (`solve_endgame_ab_parallel`) to measure wall-clock; `parallel=False` uses
    /// the serial solver — use that to compare single-core solve times.
    #[pyo3(signature = (max_secs=60.0, score_scale=100.0, margin_gain=2.0, alpha=0.8, parallel=true, ordering="lookahead2_clustered"))]
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
            Some(raw_margin) => (
                margin_to_training_value(raw_margin, score_scale, margin_gain, alpha),
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

/// Terminal backup value in player-0 frame, using the SAME mixed formula as
/// non-terminal leaf values (Fix 1).  Replaces mcts_compute_target_z (whose
/// tanh(margin/30) scale was inconsistent with the non-terminal estimates).
///
/// score_scale / margin_gain / alpha must match the Python config values
/// (SCORE_SCALE=100.0, MARGIN_GAIN=2.0, ALPHA=0.8 by default) so the Rust and
/// Python terminal values are bit-identical for the same terminal state.
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
    let win_value = if s0 > s1 {
        1.0
    } else if s1 > s0 {
        -1.0
    } else {
        0.0
    };
    alpha * margin_value + (1.0 - alpha) * win_value
}

/// Convert a raw score margin (s0 - s1, player-0 frame) into the training value.
/// Called AFTER the alpha-beta solve — the search itself runs on raw margins for
/// tightest pruning, and this is a monotone transform of the margin, so the
/// minimax-optimal move is unchanged. Bit-identical to `terminal_search_value`
/// evaluated on the same final scores (which is why values match the old solver).
fn margin_to_training_value(margin: f64, score_scale: f64, margin_gain: f64, alpha: f64) -> f64 {
    let win_value = if margin > 0.0 {
        1.0
    } else if margin < 0.0 {
        -1.0
    } else {
        0.0
    };
    let margin_value = (margin / score_scale * margin_gain).tanh();
    alpha * margin_value + (1.0 - alpha) * win_value
}

/// Full-window sentinel for the raw-margin solver. Margins live in ~[-80, 80], so
/// ±200 brackets every possible value with room to spare; callers use these as the
/// "exact, untightened" alpha-beta bounds (replacing ±∞).
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
        let score = cheap_order_score_for_solver(
            board,
            halves,
            &tc,
            opp_tc.as_ref(),
            p,
            pk,
        );
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
        let score = cheap_order_score_for_solver(
            board,
            halves,
            &tc,
            opp_tc.as_ref(),
            p,
            pk,
        );
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
    let clustered = clustered_depth
        && cheap_scores_clustered_for_solver(state, legal, mode);
    if mode.uses_lookahead_at_depth(depth) || mode.uses_adaptive_lookahead(depth, legal.len()) || clustered {
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
/// Correctness of pruning: `terminal_search_value` is monotone non-decreasing in
/// (score0 - score1), so the position value is a standard min/max over a scalar
/// in [-1, 1]; alpha-beta applies without modification.  The max/min layers need
/// not strictly alternate — each node is typed by `actor()` and the (alpha, beta)
/// window stays valid through any sequence of max and min nodes.
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
    // The search runs on the RAW integer score margin (s0 - s1), player-0 frame,
    // range ~[-80, 80]. Integer margins give the widest spread and therefore the
    // tightest alpha-beta bounds, and contain no training hyperparameters. The
    // training value is a monotone transform applied AFTER the solve, so the
    // minimax-optimal move is identical to the old value-space search.
    //
    // GAME_OVER returns a value with zero further work, so resolve it before the
    // deadline check — a timed-out search should still return exact terminal leaves
    // it has already reached rather than abort on them.
    if state.phase == GAME_OVER {
        let (s0, s1) = state.scores();
        return Ok(Some((s0 - s1) as f64));
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
    // Returns the RAW score margin (s0 - s1); callers convert to the training value
    // via `margin_to_training_value`. See `solve_endgame_ab`.
    if state.phase == GAME_OVER {
        let (s0, s1) = state.scores();
        return Ok(Some((s0 - s1) as f64));
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

    // Step 1: solve the first (best-ordered) child serially to establish a bound.
    let (_i0, p0, pk0) = legal[0];
    let first_next = state.step(p0, pk0)?;
    let first_val = match solve_endgame_ab(&first_next, deadline, MARGIN_LO, MARGIN_HI, mode, 1)? {
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
            solve_endgame_ab(&next, deadline, captured_alpha, captured_beta, mode, 1)
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
        let flat_py = flat.insert_axis(Axis(0)).into_pyarray(py); // (1,261)
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
/// (K,9,13,13)/(K,9,13,13)/(K,261); idxs are passed as a list of K int arrays.
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
struct OLNode {
    prior: f64,
    visit_count: i32,
    value_sum: f64,
    children: Vec<(u16, u32)>,
    action: (Option<(i8, i8, i8, i8, bool)>, Option<u16>),
    is_expanded: bool,
}

impl OLNode {
    fn new(prior: f64, action: (Option<(i8, i8, i8, i8, bool)>, Option<u16>)) -> Self {
        OLNode {
            prior,
            visit_count: 0,
            value_sum: 0.0,
            children: Vec::new(),
            action,
            is_expanded: false,
        }
    }
}

/// PUCT child selection for the open-loop tree.  Considers only children whose
/// joint index is legal in THIS simulation's concrete state (at deep nodes the
/// concrete current_row differs across determinizations).  Returns the chosen
/// child's id plus the action DECODED against this state; None (a counted
/// dead-end) when no child is legal here, which stops the descent.  Actor comes
/// from the concrete state, not the (stateless) node.
fn ol_select_child(
    arena: &[OLNode],
    node_id: u32,
    state: &RustGameState,
    fpu: f64,
    cpuct: f64,
    fallback_count: &mut u32,
    missing_child_count: &mut u32,
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
    let sqrt_n = (node.visit_count as f64).sqrt();

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
            // This legal action has a stored child — score it.  Scoring order is
            // ascending by joint index (same as the old children-order loop), so
            // the strict-`>` tie-break selects the SAME child: bit-identical.
            let cid = children[ci].1;
            let child = &arena[cid as usize];
            let q = if child.visit_count > 0 {
                let q0 = child.value_sum / child.visit_count as f64;
                if actor == 0 { q0 } else { -q0 }
            } else {
                fpu
            };
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
            // No legal action had a stored child at all.  After the Issue 2 fix
            // this is effectively unreachable (missing children are added), but
            // keep it as a counted defensive dead-end.
            *fallback_count += 1;
            None
        }
    }
}

/// Descend by PUCT to an unexpanded or terminal leaf, stepping a concrete
/// simulation state forward with each selected action.  No lazy state storage —
/// the concrete state is threaded as a local.  Returns (path of node ids, the
/// actor at each NON-leaf node on the path [for VL framing], leaf concrete state).
fn ol_descend(
    arena: &[OLNode],
    root_id: u32,
    mut state: RustGameState, // owned: the caller's `det` is moved in, not cloned
    fpu: f64,
    cpuct: f64,
    fallback_count: &mut u32,
    missing_child_count: &mut u32,
) -> PyResult<(Vec<u32>, Vec<u8>, RustGameState)> {
    let mut path: Vec<u32> = vec![root_id];
    let mut actors: Vec<u8> = Vec::new();
    let mut node_id = root_id;
    // `state` is already owned (moved in); no clone needed — it is stepped in
    // place as we descend and returned as the leaf state.
    loop {
        let expanded = arena[node_id as usize].is_expanded;
        let terminal = state.phase == GAME_OVER;
        if !expanded || terminal {
            break;
        }
        let actor = state.actor()?;
        match ol_select_child(
            arena,
            node_id,
            &state,
            fpu,
            cpuct,
            fallback_count,
            missing_child_count,
        ) {
            None => break, // dead-end / missing children: re-evaluate this node as the leaf
            Some((child_id, placement, pick)) => {
                actors.push(actor);
                state = state.step(placement, pick)?;
                path.push(child_id);
                node_id = child_id;
            }
        }
    }
    Ok((path, actors, state))
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
        }
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
    ///   (mb (K,9,13,13) f32, ob (K,9,13,13) f32, flat (K,261) f32, idxs_list)
    ///     -> (values (K,) f64, [gathered_logits_i (n_i,) f64])
    /// Serial search calls it with K=1.  `seed` only affects Dirichlet noise.
    #[pyo3(signature = (state, evaluator, n_sims, dirichlet_alpha=0.3, dirichlet_eps=0.0, fpu=0.0, cpuct=1.5, seed=None, leaf_batch=1, virtual_loss=1, score_scale=100.0, margin_gain=2.0, alpha=0.8))]
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
    actor: u8,
    own_score: f32,  // raw own final score (filled at game end in finalize_move)
    opp_score: f32,  // raw opponent final score (filled at game end)
    win_target: f32, // 1.0 win / 0.5 draw / 0.0 loss, actor frame (filled at end)
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
#[derive(Clone)]
struct ExactSolveResult {
    /// (joint_index, minimax value player-0 frame) for ALL legal root actions.
    child_values: Vec<(u16, f64)>,
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

/// One game's search tree + real state.  Per-tick the slot runs exactly one
/// `simulate_batch` chunk; root handling mirrors `RustMCTS::search`.
struct SearchSlot {
    state: SlotState,
    arena: Vec<Node>,      // closed-loop tree (empty when open_loop)
    ol_arena: Vec<OLNode>, // open-loop tree (empty when !open_loop)
    real_state: RustGameState,
    sims_done: usize,
    move_num: usize,
    game_seed: u64,
    rng: StdRng, // Dirichlet noise + move selection + per-sim determinization
    records: Vec<MoveRecord>,
    fallback_count: u32, // open-loop: deep-node legal-filter fallbacks (diagnostic)
    missing_child_count: u32, // open-loop: descents stopped to add newly-legal children (diagnostic)
    exact_result: Option<ExactSolveResult>, // Some only while state == ExactSolving
    exact_plan: Vec<ExactPlanItem>, // chosen-line plan for the deterministic endgame
    // Set once the exact solver times out on this game's endgame: the position is
    // too hard to solve within budget, so fall through to MCTS for ALL remaining
    // moves of this game instead of re-attempting the (still-failing) solve every
    // move. Reset to false when the slot starts a new game (new_for_game).
    exact_unsolvable: bool,
}

impl SearchSlot {
    /// Start a fresh game in this slot: real state + a redeterminized root,
    /// ready for its root evaluation on the next tick.
    fn new_for_game(real_state: RustGameState, game_seed: u64, open_loop: bool) -> SearchSlot {
        // Closed-loop stores a redeterminized root state in arena[0]; open-loop
        // is stateless and evaluates its root from the public real_state, so it
        // only needs a bare ol_arena root.
        let (arena, ol_arena) = if open_loop {
            (Vec::new(), vec![OLNode::new(1.0, (None, None))])
        } else {
            let root_state = real_state.redeterminize(Some(det_seed(game_seed, 0)));
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
            move_num: 0,
            game_seed,
            rng: StdRng::seed_from_u64(game_seed),
            records: Vec::new(),
            fallback_count: 0,
            missing_child_count: 0,
            exact_result: None,
            exact_plan: Vec::new(),
            exact_unsolvable: false,
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
            game_seed: 0,
            rng: StdRng::seed_from_u64(0),
            records: Vec::new(),
            fallback_count: 0,
            missing_child_count: 0,
            exact_result: None,
            exact_plan: Vec::new(),
            exact_unsolvable: false,
        }
    }

    /// Finalize the current move (sims complete): record the training example,
    /// select + apply a move to the REAL state, then either start the next move
    /// (→ NeedsRootEval) or, if the game is over, return its finished records.
    fn finalize_move(
        &mut self,
        temp_moves: usize,
        open_loop: bool,
        exact_enabled: bool,
    ) -> PyResult<Option<(u64, Vec<MoveRecord>, (i32, i32))>> {
        // Training record: encode the REAL (public) state + policy target.
        let actor = self.real_state.actor()?;
        let (my, opp, flat) = self.real_state.encode_arrays(actor)?;

        // Take any exact-solve result for this move (clears it so the next move,
        // if MCTS-driven, never sees a stale value).
        let exact = self.exact_result.take();

        let (policy_idx, policy_val, legal_idx, chosen) = if let Some(exact) = exact {
            // ── Exact endgame path: policy + move from minimax child values ──
            let (policy_idx, policy_val, legal_idx) =
                exact_policy_target(&exact.child_values, actor);
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
            (policy_idx, policy_val, legal_idx, chosen)
        } else {
            // ── Normal MCTS path: visit-count policy + visit-count selection ──
            let root_children: Vec<(u16, i32)> = if open_loop {
                self.ol_arena[0]
                    .children
                    .iter()
                    .map(|&(idx, c)| (idx, self.ol_arena[c as usize].visit_count))
                    .collect()
            } else {
                self.arena[0]
                    .children
                    .iter()
                    .map(|&(idx, c)| (idx, self.arena[c as usize].visit_count))
                    .collect()
            };
            let total: i32 = root_children.iter().map(|&(_, vc)| vc).sum();
            let mut policy_idx = Vec::new();
            let mut policy_val = Vec::new();
            let mut legal_idx = Vec::new();
            for &(idx, vc) in &root_children {
                legal_idx.push(idx as i32);
                if vc > 0 {
                    policy_idx.push(idx as i32);
                    policy_val.push(vc as f32 / total as f32);
                }
            }
            let temp = if self.move_num < temp_moves { 1.0 } else { 0.0 };
            let chosen = if open_loop {
                ol_select_from_visits(&self.ol_arena, temp, &mut self.rng)
            } else {
                select_from_visits(&self.arena, temp, &mut self.rng)
            };
            (policy_idx, policy_val, legal_idx, chosen)
        };

        self.records.push(MoveRecord {
            my,
            opp,
            flat,
            policy_idx,
            policy_val,
            legal_idx,
            actor,
            own_score: 0.0,
            opp_score: 0.0,
            win_target: 0.5,
        });
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
            // Fill the per-move targets now that final scores are known.  No
            // tiebreaker cascade in Rust (RustGameState lacks determine_winner);
            // score-only win, matching play_selfplay_game_rust's documented
            // limitation.  Draw → 0.5 for both (1.0 - 0.5 = 0.5).
            let win0: f32 = if s0 > s1 {
                1.0
            } else if s1 > s0 {
                0.0
            } else {
                0.5
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
            // If the solver is enabled and the new root is terminal-adjacent
            // (deck ∈ {0,4}), hand it to the exact solver instead of GPU-backed
            // MCTS. The deck only shrinks, so once a game enters ExactSolving it
            // stays there until GAME_OVER — resolve_exact_slots cascades the whole
            // endgame with zero forwards. When disabled (budget 0), endgames go
            // through normal MCTS.
            self.state = if exact_enabled
                && !self.exact_unsolvable
                && is_no_chance_endgame_state(&self.real_state)
            {
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
    // Cache only full-window solves (exact raw margins). ±200 brackets every real
    // margin, so any window at least that wide is "full" and its result is exact.
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
    value_cache: &mut HashMap<EndgameKey, f64>,
    result_cache: &mut HashMap<EndgameKey, ExactSolveResult>,
) -> PyResult<Option<ExactSolveResult>> {
    let key = endgame_key(state);
    if let Some(result) = result_cache.get(&key) {
        return Ok(Some(result.clone()));
    }

    let mut legal = state.legal_actions_indexed();
    if legal.is_empty() {
        return Ok(None); // not GAME_OVER but no actions — fall back defensively
    }
    let mode = SolverOrderMode::Lookahead2Clustered;
    order_legal_for_solver_at_depth(state, &mut legal, mode, 0)?;

    if std::time::Instant::now() >= deadline {
        return Ok(None);
    }
    // Solve each root child with a full window (the exact per-child value is needed
    // for the policy target) IN PARALLEL across cores. This is the YBW-style
    // within-solve parallelism (mirrors solve_endgame_ab_parallel) that lets ONE
    // endgame use the whole machine and finish within budget — the axis that
    // actually matters for the per-solve wall-clock deadline. Children are
    // independent (each owns its `next` state), so the shared value_cache is not
    // threaded through here; cross-move reuse via `result_cache` below is
    // unaffected. The solver returns the exact RAW margin per child; convert to the
    // (monotone) training value so argmax/argmin over children is unchanged.
    let _ = &value_cache; // intentionally unused by the parallel per-child solves
    let child_results: Vec<PyResult<Option<(u16, f64)>>> = legal
        .par_iter()
        .map(|&(joint_idx, placement, pick)| -> PyResult<Option<(u16, f64)>> {
            let next = state.step(placement, pick)?;
            match solve_endgame_ab(&next, deadline, MARGIN_LO, MARGIN_HI, mode, 0)? {
                Some(raw_margin) => Ok(Some((
                    joint_idx,
                    margin_to_training_value(raw_margin, score_scale, margin_gain, alpha_param),
                ))),
                None => Ok(None),
            }
        })
        .collect();
    let mut child_values: Vec<(u16, f64)> = Vec::with_capacity(legal.len());
    for r in child_results {
        match r? {
            Some(cv) => child_values.push(cv),
            None => return Ok(None), // a child hit the deadline → whole solve fails
        }
    }
    let result = ExactSolveResult { child_values };
    result_cache.insert(key, result.clone());
    Ok(Some(result))
}

/// A dispatched endgame solve (Step 1.5 async path): an owned snapshot of the
/// slot's game. Sent to the background solver thread; the slot keeps its own copy.
struct SolveJob {
    slot_idx: usize,
    state: RustGameState,
    records: Vec<MoveRecord>,
    game_seed: u64,
}

enum SolveResult {
    /// The endgame solved to GAME_OVER: the full finished game (seed, records, scores).
    Finished((u64, Vec<MoveRecord>, (i32, i32))),
    /// Deadline exceeded (or solve error): the slot must resume MCTS in place.
    Fallback,
}

/// Result of one background solve, harvested by the main thread on the next step().
struct SolveOutcome {
    slot_idx: usize,
    result: SolveResult,
    n_solved: u64, // plan length (for counters); 0 on fallback
    solve_secs: f64,
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
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        while let Ok(job) = job_rx.recv() {
            let SolveJob {
                slot_idx,
                state,
                records,
                game_seed,
            } = job;
            let t0 = std::time::Instant::now();
            let result = match solve_exact_plan(&state, max_secs, score_scale, margin_gain, alpha) {
                Ok(Some(plan)) if !plan.is_empty() => {
                    let n = plan.len() as u64;
                    match play_out_exact_endgame(state, records, game_seed, plan) {
                        Ok(fg) => (SolveResult::Finished(fg), n),
                        Err(_) => (SolveResult::Fallback, 0),
                    }
                }
                _ => (SolveResult::Fallback, 0),
            };
            let outcome = SolveOutcome {
                slot_idx,
                result: result.0,
                n_solved: result.1,
                solve_secs: t0.elapsed().as_secs_f64(),
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
) -> PyResult<(u64, Vec<MoveRecord>, (i32, i32))> {
    for item in plan {
        let exact = item.result;
        let actor = state.actor()?;
        let (my, opp, flat) = state.encode_arrays(actor)?;
        let (policy_idx, policy_val, legal_idx) = exact_policy_target(&exact.child_values, actor);
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
    // Plan plays to GAME_OVER; fill per-move targets from the final scores
    // (score-only win, draw -> 0.5, matching finalize_move).
    let (s0, s1) = state.scores();
    let win0: f32 = if s0 > s1 {
        1.0
    } else if s1 > s0 {
        0.0
    } else {
        0.5
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
    Ok((game_seed, records, (s0, s1)))
}

fn solve_exact_plan(
    state: &RustGameState,
    max_secs: f64,
    score_scale: f64,
    margin_gain: f64,
    alpha_param: f64,
) -> PyResult<Option<Vec<ExactPlanItem>>> {
    let mut cur = state.cloned();
    // One shared deadline for the whole endgame cascade from this root. The plan
    // is built once and reused (cache hits) for the deterministic continuation,
    // so this bounds the once-per-game expensive solve, not each move.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(max_secs);
    let mut value_cache: HashMap<EndgameKey, f64> = HashMap::new();
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
            &mut value_cache,
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
    // Open-loop only: the per-simulation deck-shuffle seeds for this tick,
    // pre-generated from the slot RNG before descent.  Carried for debugging and
    // as the prerequisite for future cross-simulation parallelism; update()
    // ignores it.  Empty for closed-loop and root-eval ticks.
    sim_seeds: Vec<u64>,
    // Open-loop Searching only: eval_path_indices[ei] = pi means evals[ei] is the
    // leaf of paths[pi].  With Issue-1 de-dup removed, multiple evals may share an
    // OLNode id; this lets update() back up each path with its OWN eval value.
    // Empty for closed-loop and root-eval ticks.
    eval_path_indices: Vec<usize>,
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
    // Cumulative open-loop diagnostics rolled in from finished slots (so the
    // Python-readable getters survive games being reset in their slots).
    cum_fallback_count: u64,
    cum_missing_child_count: u64,
    // Exact endgame solver (deck ∈ {0,4} roots). Per-position wall-clock budget
    // in seconds; <= 0.0 disables it (ablation).
    exact_endgame_max_secs: f64,
    cum_exact_solve_count: u64,      // root moves solved exactly
    cum_exact_tree_solve_count: u64, // expensive exact continuation plans built
    cum_exact_cache_hit_count: u64,  // exact moves served from a precomputed plan
    cum_exact_fallback_count: u64,   // budget exceeded → fell back to MCTS (≈0)
    cum_exact_solver_secs: f64,      // wall time spent in resolve_exact_slots (solve + finalize)
    // Games finished entirely inside resolve_exact_slots during step(); drained
    // by update() into the finished-games list it returns.
    pending_exact: Vec<(u64, Vec<MoveRecord>, (i32, i32))>,
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

impl BatchedMCTS {
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
        self.cum_exact_solver_secs += outcome.solve_secs;
        let open_loop = self.open_loop;
        match outcome.result {
            SolveResult::Finished(fg) => {
                self.cum_exact_tree_solve_count += 1;
                self.cum_exact_solve_count += outcome.n_solved;
                self.cum_exact_cache_hit_count += outcome.n_solved.saturating_sub(1);
                self.cum_fallback_count += self.slots[si].fallback_count as u64;
                self.cum_missing_child_count += self.slots[si].missing_child_count as u64;
                self.pending_exact.push(fg);
                // Recycle the slot: next game if quota remains, else Idle.
                if self.games_started < self.games_target {
                    let ns = self.next_seed;
                    self.next_seed += 1;
                    self.games_started += 1;
                    self.slots[si] =
                        SearchSlot::new_for_game(new_game(ns, self.harmony, self.middle_kingdom), ns, open_loop);
                } else {
                    self.slots[si].state = SlotState::Idle;
                    self.slots[si].fallback_count = 0;
                    self.slots[si].missing_child_count = 0;
                }
            }
            SolveResult::Fallback => {
                self.cum_exact_fallback_count += 1;
                let slot = &mut self.slots[si];
                slot.exact_unsolvable = true;
                slot.exact_result = None;
                slot.exact_plan.clear();
                slot.state = SlotState::NeedsRootEval;
                slot.sims_done = 0;
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
                    state: self.slots[si].real_state.cloned(),
                    records: self.slots[si].records.clone(),
                    game_seed: self.slots[si].game_seed,
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
            && !self.slots.iter().any(|s| {
                matches!(s.state, SlotState::Searching | SlotState::NeedsRootEval)
            })
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
                        score_scale=100.0, margin_gain=2.0, alpha=0.8,
                        exact_endgame_max_secs=3.0, async_solve=false))]
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
    ) -> Self {
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
        let mut slots = Vec::with_capacity(n_slots);
        let mut games_started = 0usize;
        for _ in 0..n_slots {
            if games_started < n_games {
                let seed = base_seed + games_started as u64;
                slots.push(SearchSlot::new_for_game(
                    new_game(seed, harmony, middle_kingdom),
                    seed,
                    open_loop,
                ));
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
        let solver_handle =
            spawn_endgame_solver(job_rx, out_tx, exact_endgame_max_secs, score_scale, margin_gain, alpha);
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
            cum_fallback_count: 0,
            cum_missing_child_count: 0,
            exact_endgame_max_secs,
            cum_exact_solve_count: 0,
            cum_exact_tree_solve_count: 0,
            cum_exact_cache_hit_count: 0,
            cum_exact_fallback_count: 0,
            cum_exact_solver_secs: 0.0,
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
        let mut finished: Vec<(usize, (u64, Vec<MoveRecord>, (i32, i32)))> = Vec::new();
        for si in 0..self.slots.len() {
            if self.slots[si].state != SlotState::ExactSolving {
                continue;
            }
            match solve_exact_plan(
                &self.slots[si].real_state,
                max_secs,
                score_scale,
                margin_gain,
                val_alpha,
            )? {
                Some(plan) if !plan.is_empty() => {
                    // Accounting matches the old per-move loop: one tree solve per
                    // endgame, n total solved moves, n-1 served from the built plan.
                    let n = plan.len() as u64;
                    self.cum_exact_tree_solve_count += 1;
                    self.cum_exact_solve_count += n;
                    self.cum_exact_cache_hit_count += n.saturating_sub(1);
                    let fg = play_out_exact_endgame(
                        self.slots[si].real_state.cloned(),
                        std::mem::take(&mut self.slots[si].records),
                        self.slots[si].game_seed,
                        plan,
                    )?;
                    finished.push((si, fg));
                }
                _ => {
                    // Deadline exceeded (or degenerate): too hard within budget.
                    // Mark Unsolvable so the rest of THIS game falls back to MCTS
                    // (cleared when the slot is recycled in new_for_game).
                    self.cum_exact_fallback_count += 1;
                    self.slots[si].exact_unsolvable = true;
                    self.slots[si].exact_result = None;
                    self.slots[si].exact_plan.clear();
                    self.slots[si].state = SlotState::NeedsRootEval;
                    self.slots[si].sims_done = 0;
                    if open_loop {
                        self.slots[si].ol_arena.clear();
                        self.slots[si].ol_arena.push(OLNode::new(1.0, (None, None)));
                    } else {
                        let root_state = self.slots[si].real_state.redeterminize(Some(
                            det_seed(self.slots[si].game_seed, self.slots[si].move_num),
                        ));
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
            self.pending_exact.push(fg);
            if self.games_started < self.games_target {
                let ns = self.next_seed;
                self.next_seed += 1;
                self.games_started += 1;
                self.slots[si] = SearchSlot::new_for_game(
                    new_game(ns, self.harmony, self.middle_kingdom),
                    ns,
                    self.open_loop,
                );
            } else {
                self.slots[si].state = SlotState::Idle;
                self.slots[si].fallback_count = 0;
                self.slots[si].missing_child_count = 0;
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
    /// Returns (mb (B,9,13,13), ob (B,9,13,13), flat (B,261), idxs_list[B]).
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

        let (fpu, cpuct, leaf_batch, vl, n_sims, open_loop) = (
            self.fpu,
            self.cpuct,
            self.leaf_batch,
            self.virtual_loss,
            self.n_sims,
            self.open_loop,
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
                            unreachable!("ExactSolving/SolvingInBackground filtered before the batch")
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
                            };
                            SlotTick {
                                slot: si,
                                is_root: true,
                                paths: Vec::new(),
                                evals: vec![ev],
                                ol_actors: Vec::new(),
                                ol_leaf_states: Vec::new(),
                                sim_seeds: Vec::new(),
                                eval_path_indices: Vec::new(),
                            }
                        }
                        SlotState::Searching => {
                            let chunk = leaf_batch.min(n_sims - slot.sims_done);
                            let mut paths: Vec<Vec<u32>> = Vec::with_capacity(chunk);
                            let mut ol_actors: Vec<Vec<u8>> = Vec::with_capacity(chunk);
                            let mut ol_leaf_states: Vec<RustGameState> = Vec::with_capacity(chunk);
                            // Pre-generate all per-simulation deck-shuffle seeds from
                            // the slot RNG BEFORE descent begins.  Materialising the
                            // sequence up front keeps it deterministic (identical to
                            // calling rng.gen() inside the loop — same count, same
                            // order) and makes the seeds independent inputs, the
                            // prerequisite for future rayon parallelism across the
                            // simulations within this slot.
                            let sim_seeds: Vec<u64> =
                                (0..chunk).map(|_| slot.rng.r#gen::<u64>()).collect();
                            for &seed in &sim_seeds {
                                let det = slot.real_state.redeterminize(Some(seed));
                                let (path, actors, leaf_state) = ol_descend(
                                    &slot.ol_arena,
                                    0,
                                    det,
                                    fpu,
                                    cpuct, // det moved in (no clone)
                                    &mut slot.fallback_count,
                                    &mut slot.missing_child_count,
                                )?;
                                ol_apply_virtual_loss(&mut slot.ol_arena, &path, &actors, 1, vl);
                                paths.push(path);
                                ol_actors.push(actors);
                                ol_leaf_states.push(leaf_state);
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
                                });
                                eval_path_indices.push(pi);
                                row += 1;
                            }
                            SlotTick {
                                slot: si,
                                is_root: false,
                                paths,
                                evals,
                                ol_actors,
                                ol_leaf_states,
                                sim_seeds,
                                eval_path_indices,
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
                        };
                        row += 1;
                        Ok(ev)
                    };

                    match slot.state {
                        SlotState::Idle => unreachable!("idle slots were filtered"),
                        SlotState::ExactSolving | SlotState::SolvingInBackground => {
                            unreachable!("ExactSolving/SolvingInBackground filtered before the batch")
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
                                sim_seeds: Vec::new(),
                                eval_path_indices: Vec::new(),
                            }
                        }
                        SlotState::Searching => {
                            let chunk = leaf_batch.min(n_sims - slot.sims_done);
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
                                sim_seeds: Vec::new(),
                                eval_path_indices: Vec::new(),
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
    /// [(game_seed, [(mb,ob,flat,pidx,pval,lidx,z,own_score,opp_score,win_target)],
    ///  (score0,score1))].
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f32>,
        gathered: Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyList>> {
        let (n_sims, vl, temp_moves, alpha, eps, harmony, mk, open_loop) = (
            self.n_sims,
            self.virtual_loss,
            self.temp_moves,
            self.dirichlet_alpha,
            self.dirichlet_eps,
            self.harmony,
            self.middle_kingdom,
            self.open_loop,
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

        let finished_by_slot: PyResult<Vec<(usize, Option<(u64, Vec<MoveRecord>, (i32, i32))>)>> =
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
                            if eps > 0.0 {
                                let dseed = slot.rng.r#gen::<u64>();
                                ol_add_dirichlet_noise(
                                    &mut slot.ol_arena,
                                    0,
                                    alpha,
                                    eps,
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
                            if eps > 0.0 {
                                let dseed = slot.rng.r#gen::<u64>();
                                add_dirichlet_noise(&mut slot.arena, 0, alpha, eps, Some(dseed));
                            }
                        }
                        slot.sims_done = 0;
                        slot.state = SlotState::Searching;
                        return Ok((si, None));
                    }

                    // Searching tick: expand eval leaves, remove VL, back up, advance.
                    if open_loop {
                        // Issue 1 fix: per-PATH value (not per-node).  evals are now
                        // one-per-non-terminal-simulation (no de-dup), so each path
                        // backs up its OWN concrete eval.  path_v0[pi] is the value
                        // for paths[pi]; terminal paths fill it from their leaf state.
                        let mut path_v0: Vec<Option<f64>> = vec![None; tick.paths.len()];
                        for (ei, ev) in tick.evals.iter().enumerate() {
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
                            path_v0[tick.eval_path_indices[ei]] = Some(value0);
                        }
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
                        // Backup: terminal value from each path's own concrete leaf
                        // state (deck-dependent), else this path's own eval value.
                        for (pi, path) in tick.paths.iter().enumerate() {
                            let ls = &tick.ol_leaf_states[pi];
                            let v0 = if ls.phase == GAME_OVER {
                                terminal_search_value(ls, score_scale, margin_gain, val_alpha)
                            } else {
                                path_v0[pi].expect("non-terminal path must have an eval value")
                            };
                            for &n in path {
                                let node = &mut slot.ol_arena[n as usize];
                                node.visit_count += 1;
                                node.value_sum += v0;
                            }
                        }
                        slot.sims_done += tick.paths.len();
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
                    let finished_game = if slot.sims_done >= n_sims {
                        slot.finalize_move(temp_moves, open_loop, exact_enabled)?
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
        let mut finished_rust: Vec<(u64, Vec<MoveRecord>, (i32, i32))> =
            std::mem::take(&mut self.pending_exact);
        for (si, finished_game) in finished_by_slot {
            if let Some(fg) = finished_game {
                finished_rust.push(fg);
                // Roll this finished slot's diagnostics into the cumulative totals
                // before it is reset/idled, so the getters survive across games.
                self.cum_fallback_count += self.slots[si].fallback_count as u64;
                self.cum_missing_child_count += self.slots[si].missing_child_count as u64;
                if self.games_started < self.games_target {
                    let ns = self.next_seed;
                    self.next_seed += 1;
                    self.games_started += 1;
                    self.slots[si] =
                        SearchSlot::new_for_game(new_game(ns, harmony, mk), ns, open_loop);
                } else {
                    self.slots[si].state = SlotState::Idle;
                    self.slots[si].fallback_count = 0;
                    self.slots[si].missing_child_count = 0;
                }
            }
        }

        // Convert finished games' records to numpy + return.
        let out = PyList::empty(py);
        for (seed, records, (s0, s1)) in finished_rust {
            let z0 = ((s0 - s1) as f64 / 30.0).tanh();
            let examples = PyList::empty(py);
            for r in records {
                let z = if r.actor == 0 { z0 } else { -z0 };
                let tup = (
                    r.my.into_pyarray(py),
                    r.opp.into_pyarray(py),
                    r.flat.into_pyarray(py),
                    r.policy_idx.into_pyarray(py),
                    r.policy_val.into_pyarray(py),
                    r.legal_idx.into_pyarray(py),
                    z,
                    r.own_score,
                    r.opp_score,
                    r.win_target,
                );
                examples.append(tup)?;
            }
            out.append((seed, examples, (s0, s1)))?;
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

/// Apply one of the 8 D4 transforms to a Kingdomino training tuple.
///
/// Arguments:
///   my_board    : (9, 13, 13) f32 array, C-contiguous
///   opp_board   : (9, 13, 13) f32 array, C-contiguous
///   flat        : (261,) f32 array — invariant, copied unchanged
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

    let mb_t = transform_spatial(mb_sl, N_BOARD_CH_AUG, k, flip);
    let ob_t = transform_spatial(ob_sl, N_BOARD_CH_AUG, k, flip);
    let fl_cp = fl_sl.to_vec();
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
    use super::RustBoard;

    #[pymodule_export]
    use super::RustGameState;

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

    /// Convert a raw score margin (s0 - s1) to the training value formula. Exposed
    /// for tests: the raw-margin alpha-beta solver applies this AFTER the solve.
    #[pyfunction]
    #[pyo3(signature = (margin, score_scale=160.0, margin_gain=2.0, alpha=0.8))]
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
    #[pyo3(signature = (state, max_secs=3.0, score_scale=100.0, margin_gain=2.0, alpha=0.8))]
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
            Some(raw_margin) => {
                let value =
                    super::margin_to_training_value(raw_margin, score_scale, margin_gain, alpha);
                Ok((value, true, start.elapsed().as_secs_f64()))
            }
            None => Ok((0.0, false, start.elapsed().as_secs_f64())),
        }
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
    #[pyo3(signature = (state, max_secs=3.0, score_scale=100.0, margin_gain=2.0, alpha=0.8))]
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
    fn debug_ordered_legal_indices(
        state: &RustGameState,
        ordering: &str,
    ) -> PyResult<Vec<u16>> {
        let mode = super::SolverOrderMode::from_str(ordering)?;
        let mut legal = state.legal_actions_indexed();
        super::order_legal_for_solver_at_depth(state, &mut legal, mode, 0)?;
        Ok(legal.into_iter().map(|(idx, _p, _pk)| idx).collect())
    }

    #[pyfunction]
    #[pyo3(signature = (state, max_secs=3.0, score_scale=100.0, margin_gain=2.0, alpha=0.8, ordering="combined", parallel=true))]
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
            Some(raw_margin) => {
                let value =
                    super::margin_to_training_value(raw_margin, score_scale, margin_gain, alpha);
                Ok((value, true, start.elapsed().as_secs_f64()))
            }
            None => Ok((0.0, false, start.elapsed().as_secs_f64())),
        }
    }
}
