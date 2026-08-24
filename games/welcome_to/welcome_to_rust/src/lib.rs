//! M1 pyo3 bindings: the Welcome To engine in Rust, behind an opaque handle.
//!
//! Path B, as Kingdomino settled it (RUST_PORT_PLAN.md §3): Rust owns the game
//! state; Python keeps the loop, the replay buffer and training. M1 is the
//! engine alone — the search, the encoder and the scheduler arrive at M3/M5/M6.
//!
//! ⚠ **Python is the oracle.** `games/welcome_to/game.py` is validated against
//! BGA by the differential harness; this is validated against `game.py` by
//! `tests/test_rust_engine_equiv.py`. If they disagree, Python wins.

mod codec;
mod constants;
mod encoder;
mod game;
mod information_key;
mod macro_codec;
mod plans;
mod rng;
mod sheet;
mod snapshot;
mod tables;

use pyo3::exceptions::{PyImportError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use game::{Config, EngineError, Game};
use rng::Rng;

fn to_py(err: EngineError) -> PyErr {
    match err {
        // `game.IllegalAction` is a `ValueError`, so an illegal action raises
        // the same class on both sides and a caller's `except ValueError`
        // behaves identically against either backend.
        EngineError::Illegal(message) => PyValueError::new_err(message),
        EngineError::Invalid(message) => PyRuntimeError::new_err(message),
    }
}

/// M0-A: the supported configuration matrix, enforced where a configuration
/// enters, and **loudly** — a silently-ignored flag is how a Rust self-play run
/// stops being equivalent without anybody noticing.
///
/// The expert and solo *rules* are ported (they share the boundary code with
/// the standard path, so leaving them out would have meant writing a different
/// engine, not a smaller one). They are refused as a **configuration**: they are
/// out of training scope, so nothing has ever gated them against Python.
fn check_supported(config: &Config) -> PyResult<()> {
    if config.expert {
        return Err(PyValueError::new_err(
            "expert mode is out of training scope and the Rust backend refuses \
             it (RUST_PORT_PLAN.md M0-A). Use the Python engine.",
        ));
    }
    if config.single_player() {
        return Err(PyValueError::new_err(
            "one-seat games change the scoring rules (TEMP_SOLO_SCORE, every \
             plan pays first place), so they are a different game and out of \
             training scope; the Rust backend refuses them (M0-A). Use the \
             Python engine.",
        ));
    }
    if config.players > 4 {
        return Err(PyValueError::new_err(format!(
            "{} seats is outside the supported 2-4 (M0-A)",
            config.players
        )));
    }
    Ok(())
}

/// The Rust engine, mirroring `games.welcome_to.game.GameState`.
#[pyclass(module = "welcome_to_rust")]
pub struct RustGameState {
    inner: Game,
}

#[pymethods]
impl RustGameState {
    /// `GameState.new(seed=..., config=...)` on the portable RNG (M0-B).
    #[new]
    #[pyo3(signature = (seed, players=2, advanced=false, expert=false, solo_rules=true))]
    fn new(
        seed: u64,
        players: usize,
        advanced: bool,
        expert: bool,
        solo_rules: bool,
    ) -> PyResult<Self> {
        let config = Config {
            players,
            advanced,
            expert,
            solo_rules,
        };
        check_supported(&config)?;
        Ok(RustGameState {
            inner: Game::new(seed, config).map_err(to_py)?,
        })
    }

    // ── the state, as the search and the tests read it ────────────────
    #[getter]
    fn turn(&self) -> i32 {
        self.inner.turn
    }

    #[getter]
    fn actor(&self) -> usize {
        self.inner.actor
    }

    /// The `Phase` value as an int, comparable with `int(state.phase)`.
    #[getter]
    fn phase(&self) -> i64 {
        self.inner.phase.as_i64()
    }

    #[getter]
    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    #[getter]
    fn current_player(&self) -> usize {
        self.inner.actor
    }

    #[getter]
    fn deck_pos(&self) -> usize {
        self.inner.deck_pos
    }

    #[getter]
    fn deck_remaining(&self) -> usize {
        self.inner.deck_remaining()
    }

    #[getter]
    fn plan_ids(&self) -> Vec<usize> {
        self.inner.plan_ids.to_vec()
    }

    #[getter]
    fn boundary_prepared(&self) -> bool {
        self.inner.boundary_prepared
    }

    #[getter]
    fn reshuffle_next_turn(&self) -> bool {
        self.inner.reshuffle_next_turn
    }

    #[getter]
    fn solo_card_drawn(&self) -> bool {
        self.inner.solo_card_drawn
    }

    /// The generator's whole state — one u64, which is what makes the M1 gate
    /// able to catch a divergence in the *number of draws* on the step it
    /// happens rather than several boundaries later.
    #[getter]
    fn rng_state(&self) -> u64 {
        self.inner.rng.state()
    }

    // ── stepping ──────────────────────────────────────────────────────
    /// Legal action indices, **in `game.py`'s order**: PUCT's first-max
    /// tie-break depends on it.
    fn legal_actions(&self) -> Vec<usize> {
        self.inner.legal_actions()
    }

    /// Apply `action` in place.
    fn apply(&mut self, action: usize) -> PyResult<()> {
        self.inner.apply(action).map_err(to_py)
    }

    /// Apply `action` to a copy and return it (states are treated as values).
    fn step(&self, action: usize) -> PyResult<RustGameState> {
        let mut next = self.inner.clone();
        next.apply(action).map_err(to_py)?;
        Ok(RustGameState { inner: next })
    }

    fn copy(&self) -> RustGameState {
        RustGameState {
            inner: self.inner.clone(),
        }
    }

    // ── the turn boundary, in three parts (SEARCH_SPEC.md §6.3) ───────
    /// Everything a boundary does before a card is revealed. `False` means the
    /// game ended here and no reveal follows.
    fn prepare_turn_boundary(&mut self) -> PyResult<bool> {
        self.inner.prepare_turn_boundary().map_err(to_py)
    }

    /// One immediate outcome of this boundary, **without modifying self**.
    ///
    /// Takes and returns the search generator's state rather than an object:
    /// the caller threads it (`PortableRng.state`), so the draws stay
    /// reproducible across the boundary without a Python generator having to
    /// live inside a Rust state.
    ///
    /// Returns `(draws, reformed, rng_state)`.
    fn sample_boundary_outcome(&self, rng_state: u64) -> PyResult<(Vec<i32>, bool, u64)> {
        let mut rng = Rng::new(rng_state);
        let outcome = self.inner.sample_boundary_outcome(&mut rng).map_err(to_py)?;
        Ok((outcome.draws, outcome.reformed, rng.state()))
    }

    /// Apply an outcome to this afterstate and open the turn. Transactional:
    /// a rejected outcome leaves the receiver untouched.
    fn apply_boundary_outcome(&mut self, draws: Vec<i32>) -> PyResult<()> {
        self.inner.apply_boundary_outcome(&draws).map_err(to_py)
    }

    /// Resample everything the acting player cannot see — a pure permutation of
    /// the undrawn deck. Returns `(state, rng_state)`.
    fn redeterminize(&self, rng_state: u64) -> (RustGameState, u64) {
        let mut rng = Rng::new(rng_state);
        let next = self.inner.redeterminize(&mut rng);
        (RustGameState { inner: next }, rng.state())
    }

    // ── information sets ──────────────────────────────────────────────
    fn reshuffle_vote_for(&self, viewer: usize) -> bool {
        self.inner.reshuffle_vote_for(viewer)
    }

    fn plan_turns_for(&self, viewer: usize, slot: usize) -> Vec<(i32, i32)> {
        self.inner.plan_turns_for(viewer, slot)
    }

    // ── the table ─────────────────────────────────────────────────────
    /// `(number, effect)` on offer in `slot`, as ints.
    #[pyo3(signature = (slot, player=None))]
    fn combination(&self, slot: usize, player: Option<usize>) -> PyResult<(i32, i64)> {
        let player = player.unwrap_or(self.inner.actor);
        let (number, effect) = self.inner.combination(slot, player).map_err(to_py)?;
        Ok((number, effect.as_i64()))
    }

    /// The visible `(number, effect)` faces of each stack, effects as ints.
    #[pyo3(signature = (player=None))]
    fn visible_cards(&self, player: Option<usize>) -> Vec<(Option<i32>, Option<i64>)> {
        let player = player.unwrap_or(self.inner.actor);
        (0..3)
            .map(|slot| {
                let (number, effect) = self.inner.combination_faces(slot, player);
                (number, effect.map(|e| e.as_i64()))
            })
            .collect()
    }

    /// The effect each stack will offer NEXT turn — a certainty, not a
    /// posterior (`game.py::next_effects`).
    #[pyo3(signature = (player=None))]
    fn next_effects(&self, player: Option<usize>) -> Vec<Option<i64>> {
        let player = player.unwrap_or(self.inner.actor);
        self.inner
            .next_effects(player)
            .iter()
            .map(|e| e.map(|v| v.as_i64()))
            .collect()
    }

    /// Every card on the table; all of them are fully identified.
    #[pyo3(signature = (player=None))]
    fn table_cards(&self, player: Option<usize>) -> Vec<Option<i32>> {
        let player = player.unwrap_or(self.inner.actor);
        self.inner
            .table_cards(player)
            .into_iter()
            .map(|c| if c == game::NO_CARD { None } else { Some(c) })
            .collect()
    }

    #[pyo3(signature = (player=None))]
    fn playable_slots(&self, player: Option<usize>) -> Vec<usize> {
        let player = player.unwrap_or(self.inner.actor);
        self.inner.playable_slots(player)
    }

    fn scorable_plan_slots(&self) -> Vec<usize> {
        self.inner.scorable_plan_slots()
    }

    // ── scoring ───────────────────────────────────────────────────────
    #[pyo3(signature = (viewer=None))]
    fn scores(&self, viewer: Option<usize>) -> Vec<i32> {
        self.inner.scores(viewer)
    }

    #[pyo3(signature = (viewer=None))]
    fn plan_scores(&self, viewer: Option<usize>) -> Vec<i32> {
        self.inner.plan_scores(viewer)
    }

    #[pyo3(signature = (viewer=None))]
    fn temp_scores(&self, viewer: Option<usize>) -> Vec<i32> {
        self.inner.temp_scores(viewer)
    }

    #[pyo3(signature = (player, viewer=None))]
    fn score_breakdown<'py>(
        &self,
        py: Python<'py>,
        player: usize,
        viewer: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let score = self.inner.score_breakdown(player, viewer);
        let out = PyDict::new(py);
        out.set_item("parks", score.parks)?;
        out.set_item("pools", score.pools)?;
        out.set_item("estates", score.estates)?;
        out.set_item("plans", score.plans)?;
        out.set_item("temp", score.temp)?;
        out.set_item("bis", score.bis)?;
        out.set_item("permits", score.permits)?;
        out.set_item("roundabouts", score.roundabouts)?;
        out.set_item("total", score.total())?;
        Ok(out)
    }

    fn ranking(&self) -> Vec<usize> {
        self.inner.ranking()
    }

    fn winners(&self) -> Vec<usize> {
        self.inner.winners()
    }

    fn returns(&self) -> Vec<f64> {
        self.inner.returns()
    }

    fn end_of_game_reason(&self) -> Option<String> {
        self.inner.end_of_game_reason()
    }

    // ── the macro vocabulary (M2) ─────────────────────────────────────
    /// Every macro index whose **whole primitive sequence** is legal here, in
    /// `macro_codec.py`'s order.
    fn legal_macros(&self) -> PyResult<Vec<usize>> {
        macro_codec::legal_macros(&self.inner).map_err(to_py)
    }

    /// `legal_macros` minus the provably dominated passes — the search's action
    /// set, and **not** a rules change (SEARCH_SPEC.md §5.1).
    #[pyo3(signature = (prune_roundabout_pass=true))]
    fn search_legal_macros(&self, prune_roundabout_pass: bool) -> PyResult<Vec<usize>> {
        macro_codec::search_legal_macros(&self.inner, prune_roundabout_pass).map_err(to_py)
    }

    /// Whether the macro layer decides here — everything except `WRITE_NUMBER`,
    /// which it swallows.
    #[getter]
    fn is_macro_root(&self) -> bool {
        macro_codec::is_macro_root(&self.inner)
    }

    /// Apply a macro's whole primitive sequence in place.
    fn apply_macro(&mut self, index: usize) -> PyResult<()> {
        macro_codec::apply_macro(&mut self.inner, index).map_err(to_py)
    }

    /// Apply a macro's whole primitive sequence to a copy.
    fn step_macro(&self, index: usize) -> PyResult<RustGameState> {
        Ok(RustGameState {
            inner: macro_codec::step_macro(&self.inner, index).map_err(to_py)?,
        })
    }

    /// M3 encoder output as four packed little-endian float32 buffers.
    /// Python exposes these as zero-copy, read-only NumPy arrays.
    #[pyo3(signature = (player=None))]
    fn encode_state<'py>(
        &self,
        py: Python<'py>,
        player: Option<usize>,
    ) -> PyResult<(
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
    )> {
        let viewer = player.unwrap_or(self.inner.actor);
        let encoded = encoder::encode_state(&self.inner, viewer).map_err(to_py)?;
        Ok((
            PyBytes::new(py, &f32_le_bytes(&encoded.sheet_planes)),
            PyBytes::new(py, &f32_le_bytes(&encoded.sheet_scalars)),
            PyBytes::new(py, &f32_le_bytes(&encoded.viewer_plane)),
            PyBytes::new(py, &f32_le_bytes(&encoded.global_scalars)),
        ))
    }

    /// M4 viewer information-state key as canonical little-endian bytes.
    #[pyo3(signature = (viewer=None))]
    fn information_key<'py>(
        &self,
        py: Python<'py>,
        viewer: Option<usize>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let viewer = viewer.unwrap_or(self.inner.actor);
        let key = information_key::information_key(&self.inner, viewer).map_err(to_py)?;
        Ok(PyBytes::new(py, &key))
    }

    // ── the M0-C snapshot ─────────────────────────────────────────────
    /// The whole state, in the shape `snapshot.to_snapshot` produces.
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        snapshot::snapshot(py, &self.inner)
    }

    /// Adopt a state written by `snapshot.to_snapshot` (M0-C).
    ///
    /// This is the direction that lets a test hand the Rust engine a
    /// *constructed* position — a filled sheet, three completed plans, a queued
    /// reshuffle — instead of only the ones play reaches.
    #[staticmethod]
    fn from_snapshot(raw: &Bound<'_, PyDict>) -> PyResult<RustGameState> {
        let inner = snapshot::from_snapshot(raw)?;
        check_supported(&inner.config)?;
        Ok(RustGameState { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "RustGameState(turn={}, actor={}, phase={:?}, deck_remaining={})",
            self.inner.turn,
            self.inner.actor,
            self.inner.phase,
            self.inner.deck_remaining()
        )
    }
}

/// The primitive sequence a macro index stands for.
#[pyfunction]
fn macro_primitives(index: usize) -> PyResult<Vec<usize>> {
    macro_codec::primitives_for(index).map_err(to_py)
}

/// `(slot, temp delta, box)` -> macro index, and its inverse.
#[pyfunction]
fn macro_write(slot: usize, delta_slot: usize, x: usize, y: usize) -> usize {
    macro_codec::macro_write(slot, delta_slot, x, y)
}

#[pyfunction]
fn decode_macro_write(index: usize) -> (usize, usize, usize, usize) {
    macro_codec::decode_macro_write(index)
}

#[pyfunction]
fn macro_refuse(slot: usize) -> usize {
    macro_codec::macro_refuse(slot)
}

/// The macro index of a primitive the macro layer does not subsume, or `None`
/// for one it does — a bare `WRITE` has no macro meaning without its slot.
#[pyfunction]
fn macro_from_primitive(action: usize) -> Option<usize> {
    macro_codec::from_primitive(action)
}

#[pyfunction]
fn macro_to_primitive(index: usize) -> Option<usize> {
    macro_codec::to_primitive(index)
}

/// M0-D: the static-table signature. Compare against
/// `games.welcome_to.tables.table_signature()` at load.
#[pyfunction]
fn table_signature() -> u64 {
    tables::table_signature()
}

/// The snapshot schema version this build emits (M0-C).
#[pyfunction]
fn snapshot_version() -> i64 {
    snapshot::SNAPSHOT_VERSION
}

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// M0-C/D are load-time ABI contracts, not merely test assertions. A stale
/// wheel must refuse to import before a self-play process can use mismatched
/// rule tables or hand it a snapshot with a different shape.
fn check_python_compatibility(py: Python<'_>) -> PyResult<()> {
    let python_tables = py.import("games.welcome_to.tables")?;
    let python_signature: u64 = python_tables
        .call_method0("table_signature")?
        .extract()?;
    let rust_signature = tables::table_signature();
    if python_signature != rust_signature {
        return Err(PyImportError::new_err(format!(
            "welcome_to_rust table signature 0x{rust_signature:016x} != Python \
             0x{python_signature:016x}; rebuild the Rust extension against the \
             current rule tables"
        )));
    }

    let python_snapshot = py.import("games.welcome_to.snapshot")?;
    let python_version: i64 = python_snapshot.getattr("SNAPSHOT_VERSION")?.extract()?;
    if python_version != snapshot::SNAPSHOT_VERSION {
        return Err(PyImportError::new_err(format!(
            "welcome_to_rust snapshot version {} != Python {python_version}; \
             rebuild the Rust extension against the current schema",
            snapshot::SNAPSHOT_VERSION
        )));
    }

    let python_encoder = py.import("games.welcome_to.encoder")?;
    let python_contract = (
        python_encoder
            .getattr("ENCODER_ABI_VERSION")?
            .extract::<usize>()?,
        python_encoder.getattr("SHEET_PLANES")?.extract::<usize>()?,
        python_encoder
            .getattr("NUM_SHEET_SCALAR")?
            .extract::<usize>()?,
        python_encoder
            .getattr("NUM_GLOBAL_SCALAR")?
            .extract::<usize>()?,
        python_encoder.getattr("MAX_SEATS")?.extract::<usize>()?,
    );
    let rust_contract = (
        encoder::ENCODER_ABI_VERSION,
        encoder::SHEET_PLANES,
        encoder::NUM_SHEET_SCALAR,
        encoder::NUM_GLOBAL_SCALAR,
        encoder::MAX_SEATS,
    );
    if python_contract != rust_contract {
        return Err(PyImportError::new_err(format!(
            "welcome_to_rust encoder contract {rust_contract:?} != Python \
             {python_contract:?}; rebuild the Rust extension against the \
             current encoder"
        )));
    }
    Ok(())
}

/// `n` draws of the portable stream from `seed` — how the RNG parity test
/// reaches the Rust generator without a game in the way (M0-B).
#[pyfunction]
fn portable_rng_stream(seed: u64, n: usize) -> Vec<u64> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.next_u64()).collect()
}

/// `PortableRng(seed).shuffle(list(range(n)))`, for the same reason.
#[pyfunction]
fn portable_rng_shuffle(seed: u64, n: usize) -> (Vec<usize>, u64) {
    let mut rng = Rng::new(seed);
    let mut seq: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut seq);
    (seq, rng.state())
}

#[pymodule]
fn welcome_to_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    check_python_compatibility(module.py())?;
    module.add_class::<RustGameState>()?;
    module.add_function(wrap_pyfunction!(table_signature, module)?)?;
    module.add_function(wrap_pyfunction!(snapshot_version, module)?)?;
    module.add_function(wrap_pyfunction!(portable_rng_stream, module)?)?;
    module.add_function(wrap_pyfunction!(portable_rng_shuffle, module)?)?;
    module.add_function(wrap_pyfunction!(macro_primitives, module)?)?;
    module.add_function(wrap_pyfunction!(macro_write, module)?)?;
    module.add_function(wrap_pyfunction!(decode_macro_write, module)?)?;
    module.add_function(wrap_pyfunction!(macro_refuse, module)?)?;
    module.add_function(wrap_pyfunction!(macro_from_primitive, module)?)?;
    module.add_function(wrap_pyfunction!(macro_to_primitive, module)?)?;
    module.add("NUM_ACTIONS", codec::NUM_ACTIONS)?;
    module.add("NUM_MACRO_ACTIONS", macro_codec::NUM_MACRO_ACTIONS)?;
    module.add("PRIMITIVE_ACTIONS", macro_codec::primitive_actions().clone())?;
    module.add("ENCODER_ABI_VERSION", encoder::ENCODER_ABI_VERSION)?;
    module.add("SHEET_PLANES_LEN", encoder::SHEET_PLANES_LEN)?;
    module.add("SHEET_SCALARS_LEN", encoder::SHEET_SCALARS_LEN)?;
    module.add("VIEWER_PLANE_LEN", encoder::VIEWER_PLANE_LEN)?;
    module.add("GLOBAL_SCALARS_LEN", encoder::NUM_GLOBAL_SCALAR)?;
    module.add(
        "INFORMATION_KEY_ABI_VERSION",
        information_key::INFORMATION_KEY_ABI_VERSION,
    )?;
    Ok(())
}
