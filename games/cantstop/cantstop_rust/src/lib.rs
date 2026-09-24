//! Can't Stop in Rust: portable RNG (M0), engine (M1), turn solver (M2).
//!
//! Boundary, per VARIANT_SOLVER_PLAN.md Phase 3: Rust owns the engine, turn
//! enumeration and backward induction. Python keeps the training loop, the
//! replay buffer and torch. Only the per-turn leaf batch crosses back.

pub mod encoder;
pub mod engine;
pub mod rng;
pub mod selfplay;
pub mod solver;

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes};

use engine::{EngineError, GameState, Move, Phase, RuleSet, COLUMN_HEIGHTS, MAX_COL, MIN_COL};
use solver::{RKey, SolveError, TurnSolver};

/// Python-visible handle on the portable stream, so the equivalence gate can
/// drive both generators in lockstep and compare states after every draw.
#[pyclass(name = "Rng")]
pub struct PyRng {
    inner: rng::Rng,
}

#[pymethods]
impl PyRng {
    #[new]
    fn new(seed: u64) -> Self {
        PyRng {
            inner: rng::Rng::new(seed),
        }
    }

    #[getter]
    fn state(&self) -> u64 {
        self.inner.state()
    }

    #[setter]
    fn set_state(&mut self, state: u64) {
        self.inner.set_state(state);
    }

    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    fn next_float(&mut self) -> f64 {
        self.inner.next_float()
    }

    fn randrange(&mut self, n: u64) -> u64 {
        self.inner.randrange(n)
    }

    fn randint(&mut self, a: u64, b: u64) -> u64 {
        self.inner.randint(a, b)
    }

    /// Four dice as `engine.random_dice` draws them, unsorted.
    fn roll_dice(&mut self) -> Vec<u8> {
        // Note: `Vec<u8>` crosses to Python as `bytes`, not `list`. The gate
        // wraps this in `list(...)` before comparing -- a bare
        // `list_of_ints != bytes` is always True even when element-wise equal.
        rng::roll_dice(&mut self.inner).to_vec()
    }

    /// Fisher-Yates over a list of ints, returned in shuffled order.
    fn shuffle(&mut self, mut seq: Vec<i64>) -> Vec<i64> {
        self.inner.shuffle(&mut seq);
        seq
    }
}

impl From<EngineError> for PyErr {
    fn from(e: EngineError) -> PyErr {
        PyValueError::new_err(e.0)
    }
}

/// The snapshot, field for field as `games/cantstop/snapshot.py::snapshot`
/// builds it. ⚠ Every sequence is `Vec<i64>`, never `Vec<u8>`: pyo3 turns
/// `Vec<u8>` into `bytes`, and `bytes` never compares equal to a list.
type Snapshot = (
    (i64, i64, bool),
    i64,
    Vec<Vec<i64>>,
    Vec<i64>,
    Vec<(i64, i64)>,
    Option<Vec<i64>>,
    i64,
    i64,
);

fn to_dice(dice: Vec<i64>) -> Result<[u8; 4], EngineError> {
    if dice.len() != 4 || dice.iter().any(|&d| !(1..=6).contains(&d)) {
        return Err(EngineError(format!("dice must be four values in 1..6, got {dice:?}")));
    }
    Ok([dice[0] as u8, dice[1] as u8, dice[2] as u8, dice[3] as u8])
}

/// Any column outside 2..12 cannot be legal, so it is refused as an illegal
/// move -- the same outcome Python reaches by failing the membership test.
fn to_move(cols: &[i64]) -> Option<Move> {
    if cols.iter().any(|&c| !(MIN_COL as i64..=MAX_COL as i64).contains(&c)) {
        return None;
    }
    match cols.len() {
        1 => Some(Move::single(cols[0] as u8)),
        2 => Some(Move::pair(cols[0] as u8, cols[1] as u8)),
        _ => None,
    }
}

fn moves_out(moves: Vec<Move>) -> Vec<Vec<i64>> {
    moves
        .iter()
        .map(|m| m.as_slice().iter().map(|&c| c as i64).collect())
        .collect()
}

/// Python-visible engine state. The equivalence gate drives this and
/// `engine.GameState` in lockstep and compares `snapshot()` after every
/// transition.
#[pyclass(name = "GameState", skip_from_py_object)]
#[derive(Clone)]
pub struct PyGameState {
    inner: GameState,
}

#[pymethods]
impl PyGameState {
    #[new]
    #[pyo3(signature = (num_players, columns_to_win, blocking))]
    fn new(num_players: u8, columns_to_win: u8, blocking: bool) -> PyResult<Self> {
        let rules = RuleSet::new(num_players, columns_to_win, blocking)?;
        Ok(PyGameState { inner: GameState::new(rules) })
    }

    /// Rebuild from a snapshot, for constructed positions. Validates shape
    /// and ranges, not reachability -- a constructed position may be one no
    /// game would reach, as long as both engines load it.
    #[staticmethod]
    fn from_snapshot(snap: Snapshot) -> PyResult<Self> {
        let ((n, cols, blocking), active, progress, claimed, runners, dice, phase, winner) = snap;
        let err = |m: &str| PyValueError::new_err(format!("bad snapshot: {m}"));
        if !(2..=4).contains(&n) || !(1..=11).contains(&cols) {
            return Err(err("rules"));
        }
        let rules = RuleSet::new(n as u8, cols as u8, blocking)?;
        let ncols = MAX_COL - MIN_COL + 1;
        if !(0..n).contains(&active) {
            return Err(err("active_player"));
        }
        if progress.len() != n as usize || progress.iter().any(|r| r.len() != ncols) {
            return Err(err("progress shape"));
        }
        if claimed.len() != ncols {
            return Err(err("claimed_by shape"));
        }
        let mut s = GameState::new(rules);
        s.active_player = active as u8;
        for (p, row) in progress.iter().enumerate() {
            for (i, &v) in row.iter().enumerate() {
                let col = MIN_COL + i;
                if !(0..=COLUMN_HEIGHTS[col] as i64).contains(&v) {
                    return Err(err("progress value"));
                }
                s.progress[p][col] = v as u8;
            }
        }
        for (i, &c) in claimed.iter().enumerate() {
            if !(-1..n).contains(&c) {
                return Err(err("claimed_by value"));
            }
            s.claimed_by[MIN_COL + i] = c as i8;
        }
        for (col, pos) in runners {
            if !(MIN_COL as i64..=MAX_COL as i64).contains(&col)
                || !(1..=COLUMN_HEIGHTS[col as usize] as i64).contains(&pos)
            {
                return Err(err("runner"));
            }
            s.runners[col as usize] = pos as u8;
        }
        if s.num_runners() > engine::MAX_RUNNERS {
            return Err(err("more than three runners"));
        }
        s.dice = match dice {
            None => None,
            Some(d) => {
                let mut d = to_dice(d)?;
                d.sort_unstable();
                Some(d)
            }
        };
        if !(0..=3).contains(&phase) {
            return Err(err("phase"));
        }
        s.phase = Phase::from_u8(phase as u8).ok_or_else(|| err("phase"))?;
        if (s.phase == Phase::AwaitMove) != s.dice.is_some() {
            return Err(err("dice are held exactly while AWAIT_MOVE"));
        }
        s.winner = match winner {
            -1 => None,
            w if (0..n).contains(&w) => Some(w as u8),
            _ => return Err(err("winner")),
        };
        Ok(PyGameState { inner: s })
    }

    fn snapshot(&self) -> Snapshot {
        let s = &self.inner;
        let n = s.rules.num_players as usize;
        (
            (s.rules.num_players as i64, s.rules.columns_to_win as i64, s.rules.blocking),
            s.active_player as i64,
            (0..n)
                .map(|p| (MIN_COL..=MAX_COL).map(|c| s.progress[p][c] as i64).collect())
                .collect(),
            (MIN_COL..=MAX_COL).map(|c| s.claimed_by[c] as i64).collect(),
            (MIN_COL..=MAX_COL)
                .filter(|&c| s.runners[c] != 0)
                .map(|c| (c as i64, s.runners[c] as i64))
                .collect(),
            s.dice.map(|d| d.iter().map(|&v| v as i64).collect()),
            s.phase as i64,
            s.winner.map_or(-1, |w| w as i64),
        )
    }

    fn clone(&self) -> Self {
        Clone::clone(self)
    }

    #[getter]
    fn active_player(&self) -> u8 {
        self.inner.active_player
    }

    #[getter]
    fn phase(&self) -> u8 {
        self.inner.phase as u8
    }

    #[getter]
    fn winner(&self) -> Option<u8> {
        self.inner.winner
    }

    #[getter]
    fn game_over(&self) -> bool {
        self.inner.game_over()
    }

    fn legal_moves(&self, dice: Vec<i64>) -> PyResult<Vec<Vec<i64>>> {
        Ok(moves_out(self.inner.legal_moves(to_dice(dice)?)))
    }

    fn roll(&mut self, dice: Vec<i64>) -> PyResult<Vec<Vec<i64>>> {
        Ok(moves_out(self.inner.roll(to_dice(dice)?)?))
    }

    fn apply_move(&mut self, mv: Vec<i64>) -> PyResult<()> {
        match to_move(&mv) {
            Some(m) => Ok(self.inner.apply_move(m)?),
            // Python checks the phase before legality; keep that order so a
            // malformed move in the wrong phase names the phase on both sides.
            None if self.inner.phase != Phase::AwaitMove => Err(PyValueError::new_err(format!(
                "illegal in phase {}",
                self.inner.phase.name()
            ))),
            None => Err(PyValueError::new_err(format!(
                "illegal move {mv:?} for dice {:?}",
                self.inner.dice
            ))),
        }
    }

    fn stop_blocked(&self) -> bool {
        self.inner.stop_blocked()
    }

    fn can_stop(&self) -> bool {
        self.inner.can_stop()
    }

    fn stop(&mut self) -> PyResult<()> {
        Ok(self.inner.stop()?)
    }

    fn bust(&mut self) -> PyResult<()> {
        Ok(self.inner.bust()?)
    }
}

fn state_from_snapshot(snap: Snapshot) -> PyResult<GameState> {
    Ok(PyGameState::from_snapshot(snap)?.inner)
}

fn snapshot_of(state: &GameState) -> Snapshot {
    PyGameState { inner: state.clone() }.snapshot()
}

type PyKey = Vec<(i64, i64)>;

fn key_out(key: &RKey) -> PyKey {
    key.as_slice().iter().map(|&(c, p)| (c as i64, p as i64)).collect()
}

fn key_in(runners: PyKey) -> PyResult<RKey> {
    let mut r = [0u8; 13];
    for (col, pos) in runners {
        if !(MIN_COL as i64..=MAX_COL as i64).contains(&col)
            || !(1..=COLUMN_HEIGHTS[col as usize] as i64).contains(&pos)
        {
            return Err(PyValueError::new_err(format!("bad runner ({col}, {pos})")));
        }
        r[col as usize] = pos as u8;
    }
    Ok(RKey::from_runners(&r))
}

impl From<SolveError> for PyErr {
    fn from(e: SolveError) -> PyErr {
        match e {
            SolveError::Unreachable(k) => PyKeyError::new_err(format!(
                "runners {:?} not reachable from this solve",
                key_out(&k)
            )),
            SolveError::GameOver => PyValueError::new_err("game is over"),
            SolveError::NoLegalMove => PyValueError::new_err("roll has no legal move"),
            SolveError::NotSolved => {
                PyRuntimeError::new_err("call set_leaf_values before querying")
            }
            SolveError::AlreadySolved => PyRuntimeError::new_err("leaf values already set"),
            SolveError::BadValues(m) => PyValueError::new_err(m),
        }
    }
}

type TableRow = (
    PyKey,
    bool,
    bool,
    f64,
    Vec<(Vec<PyKey>, f64)>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Vec<f64>,
    bool,
);

/// Python-visible turn solver, in two phases: construct (enumerates), read
/// `leaf_snapshots()`, evaluate them however you like, `set_leaf_values`
/// (backs up), then query. `games/cantstop/rust_solver.py` wraps this in
/// `solver.TurnSolver`'s interface.
#[pyclass(name = "TurnSolver", unsendable)]
pub struct PyTurnSolver {
    inner: TurnSolver,
}

#[pymethods]
impl PyTurnSolver {
    #[new]
    fn new(snap: Snapshot) -> PyResult<Self> {
        let state = state_from_snapshot(snap)?;
        let inner = TurnSolver::new_cached(&state)?;
        Ok(PyTurnSolver { inner })
    }

    #[getter]
    fn num_positions(&self) -> usize {
        self.inner.num_positions()
    }

    #[getter]
    fn num_leaves(&self) -> usize {
        self.inner.num_leaves()
    }

    #[getter]
    fn roots(&self) -> Vec<PyKey> {
        self.inner.roots.iter().map(key_out).collect()
    }

    /// End-of-turn boards to evaluate, the bust board first.
    fn leaf_snapshots(&self) -> Vec<Snapshot> {
        self.inner.leaf_boards().iter().map(snapshot_of).collect()
    }

    /// The leaf boards encoded for the value net: little-endian float32
    /// bytes, `num_leaves x FEATURE_SIZE`, for `np.frombuffer`.
    fn leaf_features<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        f32_bytes(py, &self.inner.leaf_features())
    }

    #[getter]
    fn leaf_active_player(&self) -> u8 {
        self.inner.leaf_active_player()
    }

    fn set_leaf_values(&mut self, values: Vec<Vec<f64>>) -> PyResult<()> {
        let n = self.inner.num_players;
        if values.iter().any(|r| r.len() != n) {
            return Err(SolveError::BadValues(format!(
                "expected {} rows of {} values",
                self.inner.num_leaves(),
                n
            ))
            .into());
        }
        let flat: Vec<f64> = values.into_iter().flatten().collect();
        Ok(self.inner.set_leaf_values(&flat)?)
    }

    /// The same from little-endian float64 bytes, row-major
    /// `num_leaves x num_players` (`values.astype("<f8").tobytes()`): one
    /// buffer instead of a list of lists, which cost more than the backup.
    fn set_leaf_values_bytes(&mut self, raw: &[u8]) -> PyResult<()> {
        if raw.len() % 8 != 0 {
            return Err(PyValueError::new_err("byte length is not a multiple of 8"));
        }
        let flat: Vec<f64> = raw
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().expect("8 bytes")))
            .collect();
        Ok(self.inner.set_leaf_values(&flat)?)
    }

    fn table(&self) -> Vec<TableRow> {
        self.inner
            .table()
            .into_iter()
            .map(|(k, st, w, bp, menu, sv, rv, dv, stops)| {
                (
                    key_out(&k),
                    st,
                    w,
                    bp,
                    menu.into_iter()
                        .map(|(kids, p)| (kids.iter().map(key_out).collect(), p))
                        .collect(),
                    sv,
                    rv,
                    dv,
                    stops,
                )
            })
            .collect()
    }

    fn choose_move(&self, runners: PyKey, dice: Vec<i64>) -> PyResult<Vec<i64>> {
        let mv = self.inner.choose_move(key_in(runners)?, to_dice(dice)?)?;
        Ok(mv.as_slice().iter().map(|&c| c as i64).collect())
    }

    fn should_stop(&self, runners: PyKey) -> PyResult<bool> {
        Ok(self.inner.should_stop(key_in(runners)?)?)
    }

    #[pyo3(signature = (runners, phase, dice=None))]
    fn value(&self, runners: PyKey, phase: u8, dice: Option<Vec<i64>>) -> PyResult<Vec<f64>> {
        let phase = Phase::from_u8(phase).ok_or_else(|| PyValueError::new_err("bad phase"))?;
        let dice = dice.map(to_dice).transpose()?;
        Ok(self.inner.value(key_in(runners)?, phase, dice)?)
    }
}

fn f32_bytes<'py>(py: Python<'py>, values: &[f32]) -> Bound<'py, PyBytes> {
    let mut raw = Vec::with_capacity(values.len() * 4);
    for v in values {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    PyBytes::new(py, &raw)
}

/// Encode arbitrary end-of-turn boards, for gating the encoder directly.
#[pyfunction]
fn encode_snapshots<'py>(py: Python<'py>, snaps: Vec<Snapshot>) -> PyResult<Bound<'py, PyBytes>> {
    let mut out = vec![0.0f32; snaps.len() * encoder::FEATURE_SIZE];
    for (snap, row) in snaps.into_iter().zip(out.chunks_exact_mut(encoder::FEATURE_SIZE)) {
        let state = state_from_snapshot(snap)?;
        if state.num_runners() != 0 {
            return Err(PyValueError::new_err(
                "encode expects an end-of-turn board (no runners)",
            ));
        }
        encoder::encode_board(&state, row);
    }
    Ok(f32_bytes(py, &out))
}

/// Let `fill` write `floats` f32s into `bytes` as little-endian. Writes in
/// place when the buffer is f32-aligned (CPython's allocator gives at least
/// 8-byte alignment, so in practice always), else through a temporary.
fn write_f32_le(bytes: &mut [u8], floats: usize, fill: impl FnOnce(&mut [f32])) {
    #[cfg(not(target_endian = "little"))]
    compile_error!("the feature buffer is handed to numpy as little-endian f32");
    // SAFETY: every bit pattern is a valid f32 and a valid u8.
    let (pre, mid, post) = unsafe { bytes.align_to_mut::<f32>() };
    if pre.is_empty() && post.is_empty() && mid.len() == floats {
        fill(mid);
    } else {
        let mut tmp = vec![0.0f32; floats];
        fill(&mut tmp);
        for (dst, v) in bytes.chunks_exact_mut(4).zip(&tmp) {
            dst.copy_from_slice(&v.to_le_bytes());
        }
    }
}

fn f64_from_bytes(raw: &[u8]) -> PyResult<Vec<f64>> {
    if raw.len() % 8 != 0 {
        return Err(PyValueError::new_err("byte length is not a multiple of 8"));
    }
    Ok(raw
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().expect("8 bytes")))
        .collect())
}

type BlockOut = (usize, usize, u8, u8, u8);
type GameOut = (u64, i64, i64, Vec<i64>, u32, u32, u64, Vec<u32>);

/// Many self-play or arena games advanced together (M3). Python's loop:
///
///     pool.advance()
///     while pool.running:
///         feats, blocks = pool.pending()
///         pool.resume(values_for(feats, blocks))
///
/// The Python driver lives in `games/cantstop/rust_pool.py`.
/// The GIL is released for every call that does real work.
#[pyclass(name = "SelfPlayPool")]
pub struct PySelfPlayPool {
    inner: selfplay::Pool,
    features_taken: bool,
}

#[pymethods]
impl PySelfPlayPool {
    /// `games`: one `(snapshot, seed, evaluator_of_seat)` per game, where the
    /// snapshot is the starting position (normally a fresh game) and
    /// `evaluator_of_seat` has one small int per seat. `threads == 0` means
    /// one per logical core.
    #[new]
    #[pyo3(signature = (games, max_turns, threads=0, in_flight=0))]
    fn new(
        games: Vec<(Snapshot, u64, Vec<u8>)>,
        max_turns: u32,
        threads: usize,
        in_flight: usize,
    ) -> PyResult<Self> {
        let mut out = Vec::with_capacity(games.len());
        for (i, (snap, seed, seats)) in games.into_iter().enumerate() {
            let state = state_from_snapshot(snap)?;
            if seats.len() != state.rules.num_players as usize {
                return Err(PyValueError::new_err(format!(
                    "game {i}: {} evaluator seats for {} players",
                    seats.len(),
                    state.rules.num_players
                )));
            }
            let mut e = [0u8; 4];
            e[..seats.len()].copy_from_slice(&seats);
            out.push(selfplay::Game::new(state, seed, e, i as u64, max_turns));
        }
        Ok(PySelfPlayPool { inner: selfplay::Pool::new(out, threads, in_flight), features_taken: false })
    }

    fn advance(&mut self, py: Python<'_>) {
        let inner = &mut self.inner;
        py.detach(|| inner.advance());
    }

    #[getter]
    fn running(&self) -> bool {
        self.inner.running()
    }

    #[getter]
    fn failure(&self) -> Option<String> {
        self.inner.failure()
    }

    #[getter]
    fn num_games(&self) -> usize {
        self.inner.games.len()
    }

    /// `(features, blocks)`: a writable `bytearray` of LE float32,
    /// `sum(rows) x FEATURE_SIZE` (for `np.frombuffer`, no copy), and one
    /// `(game, rows, evaluator, num_players, leaf_active)` per waiting game,
    /// in buffer order. Rust encodes straight into the bytearray, in
    /// parallel, with the GIL released.
    fn pending<'py>(&mut self, py: Python<'py>) -> PyResult<(Bound<'py, PyByteArray>, Vec<BlockOut>)> {
        let blocks: Vec<BlockOut> = self
            .inner
            .pending_blocks()
            .iter()
            .map(|b| (b.game, b.rows, b.evaluator, b.num_players, b.leaf_active))
            .collect();
        let floats = self.inner.pending_rows() * encoder::FEATURE_SIZE;
        let inner = &self.inner;
        let buf = PyByteArray::new_with(py, floats * 4, |bytes| {
            py.detach(|| write_f32_le(bytes, floats, |out| inner.write_pending(out)));
            Ok(())
        })?;
        self.features_taken = true;
        Ok((buf, blocks))
    }

    /// LE float64 bytes: for each block in order, `rows x num_players`
    /// absolute-seat values.
    fn resume(&mut self, py: Python<'_>, values: &[u8]) -> PyResult<()> {
        if !self.features_taken {
            return Err(PyRuntimeError::new_err("call pending() before resume()"));
        }
        let values = f64_from_bytes(values)?;
        let inner = &mut self.inner;
        py.detach(|| inner.resume(&values)).map_err(PyValueError::new_err)?;
        self.features_taken = false;
        Ok(())
    }

    /// The rotation `resume_relative` applies, without resuming: LE f64
    /// bytes. For gating it against `encoder.to_absolute`.
    fn absolute_from_relative<'py>(&self, py: Python<'py>, relative: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let rel: Vec<f32> = relative
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
            .collect();
        let v = self.inner.absolute_from_relative(&rel).map_err(PyValueError::new_err)?;
        let mut raw = Vec::with_capacity(v.len() * 8);
        for x in v {
            raw.extend_from_slice(&x.to_le_bytes());
        }
        Ok(PyBytes::new(py, &raw))
    }

    /// `resume` from the net's seat-relative output instead: LE float32,
    /// `pending_rows x 4`, slot 0 = the leaf boards' seat to move. Rust does
    /// `to_absolute` per block.
    fn resume_relative(&mut self, py: Python<'_>, relative: &[u8]) -> PyResult<()> {
        if !self.features_taken {
            return Err(PyRuntimeError::new_err("call pending() before resume()"));
        }
        if relative.len() % 4 != 0 {
            return Err(PyValueError::new_err("byte length is not a multiple of 4"));
        }
        let rel: Vec<f32> = relative
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
            .collect();
        let inner = &mut self.inner;
        py.detach(|| {
            let values = inner.absolute_from_relative(&rel)?;
            inner.resume(&values)
        })
        .map_err(PyValueError::new_err)?;
        self.features_taken = false;
        Ok(())
    }

    /// One tuple per game, in game order: `(id, winner, rows, winner_slots,
    /// turns, solves, evaluator_rows, turn_lengths)`, with the features of
    /// every game concatenated in `features()`. Winner is -1 for a failed
    /// game.
    fn results(&self) -> PyResult<Vec<GameOut>> {
        if self.inner.running() {
            return Err(PyRuntimeError::new_err("games are still running"));
        }
        Ok(self
            .inner
            .games
            .iter()
            .map(|g| {
                let rows = (g.features.len() / encoder::FEATURE_SIZE) as i64;
                let (winner, slots) = match g.winner() {
                    Some(w) => (w as i64, g.winner_slots().iter().map(|&s| s as i64).collect()),
                    None => (-1, Vec::new()),
                };
                (g.id, winner, rows, slots, g.turns, g.solves, g.evaluator_rows, g.turn_lengths.clone())
            })
            .collect())
    }

    /// Every game's recorded boards, encoded, concatenated in game order.
    fn features<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let all: Vec<f32> = self.inner.games.iter().flat_map(|g| g.features.iter().copied()).collect();
        f32_bytes(py, &all)
    }
}

/// `(menus built, approx bytes)` summed over every thread's menu cache since
/// the process started (never decremented).
#[pyfunction]
fn menu_cache_stats() -> (usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    (solver::MENUS_BUILT.load(Relaxed), solver::MENU_BYTES.load(Relaxed))
}

/// Cap every thread's menu cache at `entries` (0 = unbounded; default
/// 20,000 -- see `solver::MENU_CACHE_CAP` for the measurement).
#[pyfunction]
fn set_menu_cache_cap(entries: usize) {
    solver::MENU_CACHE_CAP.store(entries, std::sync::atomic::Ordering::Relaxed);
}

#[pymodule]
fn cantstop_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRng>()?;
    m.add_class::<PyGameState>()?;
    m.add_class::<PyTurnSolver>()?;
    m.add_class::<PySelfPlayPool>()?;
    m.add_function(wrap_pyfunction!(encode_snapshots, m)?)?;
    m.add_function(wrap_pyfunction!(menu_cache_stats, m)?)?;
    m.add_function(wrap_pyfunction!(set_menu_cache_cap, m)?)?;
    m.add("FEATURE_SIZE", encoder::FEATURE_SIZE)?;
    Ok(())
}
