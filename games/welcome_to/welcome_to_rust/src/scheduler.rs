//! M6 cross-search scheduler and packed evaluator coalescer.
//!
//! Each search still has exactly one outstanding evaluator request. Independent
//! searches run on Rust worker threads and stop at the unchanged M5 callback;
//! this coordinator forms one deterministic wave from those requests, packs the
//! frozen M0-E V2 ABI, calls Python once per chunk, and routes rows by request
//! id. There is deliberately no virtual loss or within-search leaf batching.

use std::collections::{HashMap, HashSet};
use std::mem::{size_of, size_of_val};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Mutex};
use std::time::Instant;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};

use crate::encoder::{self, EncodedState};
use crate::game::{EngineError, EngineResult, Game};
use crate::macro_codec;
use crate::search::{
    EvalResponse, PlaySession, RequestKind, Search, SearchConfig, SearchEvent, SearchOutput,
};
use crate::{search_output, to_py, RustGameState};

const EVALUATOR_ABI_VERSION: u16 = 2;

struct EvalRow {
    input: usize,
    kind: RequestKind,
    encoded: EncodedState,
    legal: Vec<usize>,
    seats: usize,
    abi_id: u32,
}

struct EvalRequest {
    row: EvalRow,
    reply: mpsc::Sender<Result<EvalResponse, String>>,
}

struct Finished {
    input: usize,
    slot: usize,
    search: Search,
    result: NativeResult,
}

type NativeResult = EngineResult<(Option<usize>, SearchOutput)>;

enum WorkerMessage {
    Request(EvalRequest),
    Finished(Finished),
}

#[derive(Default)]
struct PackedBatch {
    sheet_planes: Vec<u8>,
    sheet_scalars: Vec<u8>,
    viewer_plane: Vec<u8>,
    global_scalars: Vec<u8>,
    legal_indices: Vec<u8>,
    legal_offsets: Vec<u8>,
    kind: Vec<u8>,
    seats: Vec<u8>,
    request_id: Vec<u8>,
}

fn extend_f32(out: &mut Vec<u8>, values: &[f32]) {
    out.reserve(size_of_val(values));
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

impl PackedBatch {
    fn clear(&mut self) {
        self.sheet_planes.clear();
        self.sheet_scalars.clear();
        self.viewer_plane.clear();
        self.global_scalars.clear();
        self.legal_indices.clear();
        self.legal_offsets.clear();
        self.kind.clear();
        self.seats.clear();
        self.request_id.clear();
    }

    fn pack(&mut self, requests: &[&EvalRow]) -> PyResult<()> {
        self.clear();
        push_u32(&mut self.legal_offsets, 0);
        let mut legal_total = 0usize;
        for request in requests {
            extend_f32(&mut self.sheet_planes, &request.encoded.sheet_planes);
            extend_f32(&mut self.sheet_scalars, &request.encoded.sheet_scalars);
            extend_f32(&mut self.viewer_plane, &request.encoded.viewer_plane);
            extend_f32(&mut self.global_scalars, &request.encoded.global_scalars);
            for &action in &request.legal {
                let action = u16::try_from(action).map_err(|_| {
                    PyRuntimeError::new_err("macro index does not fit evaluator ABI u16")
                })?;
                self.legal_indices.extend_from_slice(&action.to_le_bytes());
            }
            legal_total += request.legal.len();
            push_u32(
                &mut self.legal_offsets,
                u32::try_from(legal_total).map_err(|_| {
                    PyRuntimeError::new_err("evaluator legal-index buffer exceeds u32")
                })?,
            );
            self.kind.push(request.kind as u8);
            self.seats.push(request.seats as u8);
            push_u32(&mut self.request_id, request.abi_id);
        }
        Ok(())
    }

    fn payload<'py>(&self, py: Python<'py>, rows: usize) -> PyResult<Bound<'py, PyDict>> {
        let payload = PyDict::new(py);
        payload.set_item("version", EVALUATOR_ABI_VERSION)?;
        payload.set_item("rows", rows)?;
        payload.set_item("sheet_planes", PyBytes::new(py, &self.sheet_planes))?;
        payload.set_item("sheet_scalars", PyBytes::new(py, &self.sheet_scalars))?;
        payload.set_item("viewer_plane", PyBytes::new(py, &self.viewer_plane))?;
        payload.set_item("global_scalars", PyBytes::new(py, &self.global_scalars))?;
        payload.set_item("legal_indices", PyBytes::new(py, &self.legal_indices))?;
        payload.set_item("legal_offsets", PyBytes::new(py, &self.legal_offsets))?;
        payload.set_item("kind", PyBytes::new(py, &self.kind))?;
        payload.set_item("seats", PyBytes::new(py, &self.seats))?;
        payload.set_item("request_id", PyBytes::new(py, &self.request_id))?;
        Ok(payload)
    }
}

fn dict_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("evaluator response omitted {key}")))
}

fn response_bytes(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<u8>> {
    Ok(dict_item(dict, key)?.cast::<PyBytes>()?.as_bytes().to_vec())
}

fn evaluate_chunk(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    batch: &mut PackedBatch,
    requests: &[&EvalRow],
) -> PyResult<(Vec<EvalResponse>, ChunkProfile)> {
    let pack_started = Instant::now();
    batch.pack(requests)?;
    let payload = batch.payload(py, requests.len())?;
    let pack_ns = pack_started.elapsed().as_nanos();
    let python_started = Instant::now();
    let raw = evaluator.call_method1("forward", (payload,))?;
    let python_eval_ns = python_started.elapsed().as_nanos();
    let decode_started = Instant::now();
    let response = raw.cast::<PyDict>()?;
    let version: u16 = dict_item(response, "version")?.extract()?;
    let rows: usize = dict_item(response, "rows")?.extract()?;
    if version != EVALUATOR_ABI_VERSION {
        return Err(PyValueError::new_err(format!(
            "evaluator response ABI {version}, expected {EVALUATOR_ABI_VERSION}"
        )));
    }
    if rows != requests.len() {
        return Err(PyValueError::new_err(format!(
            "evaluator returned {rows} rows for {} requests",
            requests.len()
        )));
    }

    let ids_raw = response_bytes(response, "request_id")?;
    let priors_raw = response_bytes(response, "priors")?;
    let values_raw = response_bytes(response, "values")?;
    let expected_priors = rows * macro_codec::NUM_MACRO_ACTIONS * size_of::<f32>();
    if ids_raw.len() != rows * size_of::<u32>()
        || priors_raw.len() != expected_priors
        || values_raw.len() != rows * size_of::<f64>()
    {
        return Err(PyValueError::new_err(format!(
            "malformed evaluator response buffers: ids={}, priors={}, values={}, rows={rows}",
            ids_raw.len(),
            priors_raw.len(),
            values_raw.len()
        )));
    }

    let expected: HashMap<u32, usize> = requests
        .iter()
        .enumerate()
        .map(|(row, request)| (request.abi_id, row))
        .collect();
    let mut output: Vec<Option<EvalResponse>> = (0..rows).map(|_| None).collect();
    let mut seen = HashSet::with_capacity(rows);
    for response_row in 0..rows {
        let id_start = response_row * 4;
        let id = u32::from_le_bytes(ids_raw[id_start..id_start + 4].try_into().unwrap());
        let request_row = *expected.get(&id).ok_or_else(|| {
            PyValueError::new_err(format!("evaluator returned unknown request_id {id}"))
        })?;
        if !seen.insert(id) {
            return Err(PyValueError::new_err(format!(
                "evaluator returned duplicate request_id {id}"
            )));
        }
        let prior_start = response_row * macro_codec::NUM_MACRO_ACTIONS * 4;
        let prior_end = prior_start + macro_codec::NUM_MACRO_ACTIONS * 4;
        let priors: Vec<f32> = priors_raw[prior_start..prior_end]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let mut legal = vec![false; macro_codec::NUM_MACRO_ACTIONS];
        for &action in &requests[request_row].legal {
            legal[action] = true;
        }
        if priors
            .iter()
            .enumerate()
            .any(|(action, &prior)| prior != 0.0 && !legal[action])
        {
            return Err(PyValueError::new_err(format!(
                "evaluator row for request_id {id} assigned probability to an illegal macro"
            )));
        }
        let value_start = response_row * 8;
        let raw_value =
            f64::from_le_bytes(values_raw[value_start..value_start + 8].try_into().unwrap());
        let value = (requests[request_row].kind == RequestKind::Leaf).then_some(raw_value);
        output[request_row] = Some(EvalResponse { priors, value });
    }
    if seen.len() != rows {
        return Err(PyValueError::new_err(
            "evaluator response omitted a request_id",
        ));
    }
    let output = output.into_iter().map(Option::unwrap).collect();
    Ok((
        output,
        ChunkProfile {
            pack_ns,
            python_eval_ns,
            decode_ns: decode_started.elapsed().as_nanos(),
        },
    ))
}

#[derive(Default)]
struct ChunkProfile {
    pack_ns: u128,
    python_eval_ns: u128,
    decode_ns: u128,
}

#[allow(clippy::too_many_arguments)]
fn config(
    simulations: usize,
    c_puct: f64,
    alpha: f64,
    margin_gain: f64,
    confidence_power: f64,
    prune_roundabout_pass: bool,
    chance_widening: Option<f64>,
    chance_widening_alpha: f64,
    max_particles: usize,
    noise_fresh_fraction: f64,
    dirichlet_weight: f64,
    temperature: f64,
    noise_required: bool,
) -> SearchConfig {
    SearchConfig {
        simulations,
        c_puct,
        alpha,
        margin_gain,
        confidence_power,
        prune_roundabout_pass,
        chance_widening,
        chance_widening_alpha,
        max_particles,
        noise_fresh_fraction,
        dirichlet_weight,
        temperature,
        noise_required,
    }
}

fn search_stats<'py>(
    py: Python<'py>,
    searches: &[Option<Search>],
    slot: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let search = searches
        .get(slot)
        .and_then(Option::as_ref)
        .ok_or_else(|| PyValueError::new_err("scheduler slot is out of range or busy"))?;
    let out = PyDict::new(py);
    out.set_item("simulations_run", search.simulations_run)?;
    out.set_item("simulations_reused", search.simulations_reused)?;
    out.set_item("reroots", search.reroots)?;
    out.set_item("terminal_leaves", search.terminal_leaves)?;
    out.set_item(
        "particle_slots_allocated",
        search.particle_slots_allocated(),
    )?;
    out.set_item(
        "particle_states_allocated",
        search.particle_states_allocated(),
    )?;
    Ok(out)
}

/// A fixed set of persistent search slots feeding one packed global coalescer.
/// Slots may be reused across calls (and reassigned after `reset`) so a Python
/// game driver can continuously replace completed games without rebuilding the
/// scheduler or losing within-turn retained subtrees for games still in flight.
#[pyclass(module = "welcome_to_rust")]
pub struct RustScheduler {
    searches: Vec<Option<Search>>,
    batch: PackedBatch,
    next_request_id: u32,
}

#[pymethods]
impl RustScheduler {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        capacity=32,
        simulations=128,
        c_puct=1.5,
        alpha=0.5,
        margin_gain=2.0,
        confidence_power=1.0,
        prune_roundabout_pass=true,
        chance_widening=Some(1.0),
        chance_widening_alpha=0.5,
        max_particles=4,
        noise_fresh_fraction=1.0,
        dirichlet_weight=0.25,
        temperature=0.0,
        noise_required=false
    ))]
    fn new(
        capacity: usize,
        simulations: usize,
        c_puct: f64,
        alpha: f64,
        margin_gain: f64,
        confidence_power: f64,
        prune_roundabout_pass: bool,
        chance_widening: Option<f64>,
        chance_widening_alpha: f64,
        max_particles: usize,
        noise_fresh_fraction: f64,
        dirichlet_weight: f64,
        temperature: f64,
        noise_required: bool,
    ) -> PyResult<Self> {
        if capacity == 0 {
            return Err(PyValueError::new_err("scheduler capacity must be positive"));
        }
        let config = config(
            simulations,
            c_puct,
            alpha,
            margin_gain,
            confidence_power,
            prune_roundabout_pass,
            chance_widening,
            chance_widening_alpha,
            max_particles,
            noise_fresh_fraction,
            dirichlet_weight,
            temperature,
            noise_required,
        );
        let mut searches = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            searches.push(Some(Search::new(config.clone()).map_err(to_py)?));
        }
        Ok(Self {
            searches,
            batch: PackedBatch::default(),
            next_request_id: 0,
        })
    }

    #[pyo3(signature = (states, evaluator, seeds, roots=None, noises=None, slots=None, max_batch=32))]
    #[allow(clippy::too_many_arguments)]
    fn search<'py>(
        &mut self,
        py: Python<'py>,
        states: Vec<Py<RustGameState>>,
        evaluator: &Bound<'py, PyAny>,
        seeds: Vec<u64>,
        roots: Option<Vec<usize>>,
        noises: Option<Vec<Option<Vec<f64>>>>,
        slots: Option<Vec<usize>>,
        max_batch: usize,
    ) -> PyResult<Vec<Py<PyDict>>> {
        self.run(
            py, states, evaluator, seeds, roots, noises, None, slots, max_batch, false,
        )
    }

    #[pyo3(signature = (states, evaluator, seeds, roots=None, noises=None, temperatures=None, slots=None, max_batch=32))]
    #[allow(clippy::too_many_arguments)]
    fn play<'py>(
        &mut self,
        py: Python<'py>,
        states: Vec<Py<RustGameState>>,
        evaluator: &Bound<'py, PyAny>,
        seeds: Vec<u64>,
        roots: Option<Vec<usize>>,
        noises: Option<Vec<Option<Vec<f64>>>>,
        temperatures: Option<Vec<f64>>,
        slots: Option<Vec<usize>>,
        max_batch: usize,
    ) -> PyResult<Vec<Py<PyDict>>> {
        self.run(
            py,
            states,
            evaluator,
            seeds,
            roots,
            noises,
            temperatures,
            slots,
            max_batch,
            true,
        )
    }

    #[pyo3(signature = (slot=None))]
    fn reset(&mut self, slot: Option<usize>) -> PyResult<()> {
        match slot {
            Some(slot) => self
                .searches
                .get_mut(slot)
                .ok_or_else(|| PyValueError::new_err("scheduler slot is out of range"))?
                .as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("scheduler slot is busy"))?
                .reset(),
            None => {
                for search in &mut self.searches {
                    search
                        .as_mut()
                        .ok_or_else(|| PyRuntimeError::new_err("scheduler slot is busy"))?
                        .reset();
                }
            }
        }
        Ok(())
    }

    fn stats<'py>(&self, py: Python<'py>, slot: usize) -> PyResult<Bound<'py, PyDict>> {
        let search = self
            .searches
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| PyValueError::new_err("scheduler slot is out of range or busy"))?;
        let out = PyDict::new(py);
        out.set_item("simulations_run", search.simulations_run)?;
        out.set_item("simulations_reused", search.simulations_reused)?;
        out.set_item("reroots", search.reroots)?;
        out.set_item("terminal_leaves", search.terminal_leaves)?;
        out.set_item(
            "particle_slots_allocated",
            search.particle_slots_allocated(),
        )?;
        out.set_item(
            "particle_states_allocated",
            search.particle_states_allocated(),
        )?;
        Ok(out)
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.searches.len()
    }
}

impl RustCloudScheduler {
    #[allow(clippy::too_many_arguments)]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        states: Vec<Py<RustGameState>>,
        evaluator: &Bound<'py, PyAny>,
        seeds: Vec<u64>,
        roots: Option<Vec<usize>>,
        noises: Option<Vec<Option<Vec<f64>>>>,
        temperatures: Option<Vec<f64>>,
        slots: Option<Vec<usize>>,
        max_batch: usize,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let wall_started = Instant::now();
        let rows = states.len();
        if rows == 0 {
            return Ok(Vec::new());
        }
        if max_batch == 0 {
            return Err(PyValueError::new_err("max_batch must be positive"));
        }
        if seeds.len() != rows {
            return Err(PyValueError::new_err(
                "states and seeds are not row-aligned",
            ));
        }
        let slots = slots.unwrap_or_else(|| (0..rows).collect());
        if slots.len() != rows || slots.iter().any(|&slot| slot >= self.searches.len()) {
            return Err(PyValueError::new_err(
                "scheduler slots are not valid and row-aligned",
            ));
        }
        if slots.iter().copied().collect::<HashSet<_>>().len() != rows {
            return Err(PyValueError::new_err(
                "scheduler slots must be unique within a wave",
            ));
        }
        let roots = match roots {
            Some(roots) if roots.len() == rows => roots,
            Some(_) => return Err(PyValueError::new_err("roots are not row-aligned")),
            None => states
                .iter()
                .map(|state| state.bind(py).borrow().inner.actor)
                .collect(),
        };
        let noises = match noises {
            Some(noises) if noises.len() == rows => noises,
            Some(_) => return Err(PyValueError::new_err("noises are not row-aligned")),
            None => (0..rows).map(|_| None).collect(),
        };
        let temperatures = match temperatures {
            Some(temperatures) if temperatures.len() == rows => temperatures
                .into_iter()
                .map(Some)
                .collect::<Vec<Option<f64>>>(),
            Some(_) => return Err(PyValueError::new_err("temperatures are not row-aligned")),
            None => (0..rows).map(|_| None).collect(),
        };
        let games: Vec<Game> = states
            .iter()
            .map(|state| state.bind(py).borrow().inner.clone())
            .collect();
        for input in 0..rows {
            if roots[input] != games[input].actor {
                return Err(PyValueError::new_err(format!(
                    "play root {} is not actor {} for row {input}",
                    roots[input], games[input].actor
                )));
            }
            if let Some(temperature) = temperatures[input] {
                if !temperature.is_finite() || temperature < 0.0 {
                    return Err(PyValueError::new_err(
                        "temperature overrides must be finite and non-negative",
                    ));
                }
            }
        }
        if slots.iter().any(|&slot| self.searches[slot].is_none()) {
            return Err(PyRuntimeError::new_err("scheduler slot is already busy"));
        }

        let worker_count = self.workers.min(rows);
        let mut buckets: Vec<Vec<CloudTask>> = (0..worker_count).map(|_| Vec::new()).collect();
        for input in 0..rows {
            let search = self.searches[slots[input]]
                .take()
                .expect("slots validated idle");
            let temperature = temperatures[input].unwrap_or(search.config.temperature);
            let session = PlaySession::new(
                search,
                games[input].clone(),
                roots[input],
                seeds[input],
                noises[input].clone(),
                temperature,
            )
            .map_err(to_py)?;
            buckets[input % worker_count].push(CloudTask {
                input,
                slot: slots[input],
                session: Some(session),
            });
        }

        let mut finished = Vec::with_capacity(rows);
        let mut python_error: Option<PyErr> = None;
        let mut active: HashSet<usize> = (0..worker_count).collect();
        for (worker, tasks) in buckets.into_iter().enumerate() {
            self.commands[worker]
                .send(CloudCommand::Start(tasks))
                .map_err(|_| PyRuntimeError::new_err("cloud worker pool is not running"))?;
        }

        while !active.is_empty() {
            let wave_target = active.clone();
            let mut arrived = HashSet::with_capacity(wave_target.len());
            let mut pending = Vec::new();
            let wait_started = Instant::now();
            while arrived.len() < wave_target.len() {
                let message = py.detach(|| {
                    self.messages
                        .lock()
                        .expect("cloud scheduler receiver lock poisoned")
                        .recv()
                });
                match message {
                    Ok(CloudWorkerMessage::Request(request)) => pending.push(request),
                    Ok(CloudWorkerMessage::Finished(done)) => finished.push(done),
                    Ok(CloudWorkerMessage::Barrier(worker)) => {
                        if wave_target.contains(&worker) {
                            arrived.insert(worker);
                        }
                    }
                    Ok(CloudWorkerMessage::Done(worker, worker_profile)) => {
                        self.profile.add_worker(worker_profile);
                        active.remove(&worker);
                        if wave_target.contains(&worker) {
                            arrived.insert(worker);
                        }
                    }
                    Ok(CloudWorkerMessage::Panicked(worker)) => {
                        python_error.get_or_insert_with(|| {
                            PyRuntimeError::new_err(format!(
                                "cloud search worker {worker} panicked"
                            ))
                        });
                        active.remove(&worker);
                        if wave_target.contains(&worker) {
                            arrived.insert(worker);
                        }
                    }
                    Err(_) => {
                        python_error.get_or_insert_with(|| {
                            PyRuntimeError::new_err(
                                "cloud search workers disconnected before completion",
                            )
                        });
                        for worker in &wave_target {
                            arrived.insert(*worker);
                        }
                        active.clear();
                    }
                }
            }
            self.profile.coordinator_wait_ns += wait_started.elapsed().as_nanos();
            self.profile.waves += 1;

            pending.sort_by_key(|request| request.row.input);
            let mut responses: Vec<Vec<(usize, EvalResponse)>> =
                (0..worker_count).map(|_| Vec::new()).collect();
            if python_error.is_none() {
                for chunk in pending.chunks_mut(max_batch) {
                    for request in chunk.iter_mut() {
                        request.row.abi_id = self.next_request_id;
                        self.next_request_id = self.next_request_id.wrapping_add(1);
                    }
                    let eval_rows: Vec<&EvalRow> =
                        chunk.iter().map(|request| &request.row).collect();
                    match evaluate_chunk(py, evaluator, &mut self.batch, &eval_rows) {
                        Ok((chunk_responses, chunk_profile)) => {
                            self.profile.add_chunk(chunk.len(), chunk_profile);
                            for (request, response) in chunk.iter().zip(chunk_responses) {
                                responses[request.worker].push((request.row.input, response));
                            }
                        }
                        Err(error) => {
                            python_error = Some(error);
                            break;
                        }
                    }
                }
            }

            if python_error.is_some() {
                for &worker in &active {
                    let _ = self.commands[worker].send(CloudCommand::Cancel);
                }
            } else {
                for &worker in &active {
                    let rows = std::mem::take(&mut responses[worker]);
                    let _ = self.commands[worker].send(CloudCommand::Responses(rows));
                }
            }
        }

        self.profile.wall_ns += wall_started.elapsed().as_nanos();
        self.profile.plays += 1;
        finished.sort_by_key(|done| done.input);
        let mut results: Vec<Option<NativeResult>> = (0..rows).map(|_| None).collect();
        for done in finished {
            results[done.input] = Some(done.result);
            self.searches[done.slot] = Some(done.search);
        }
        // A caught worker panic cannot return the in-progress sessions it
        // owned. Restore those slots with clean trees so the Python object is
        // never left permanently "busy" after reporting the infrastructure
        // failure. Normal evaluator errors preserve the original trees above.
        if python_error.is_some() || results.iter().any(Option::is_none) {
            for &slot in &slots {
                if self.searches[slot].is_none() {
                    self.searches[slot] =
                        Some(Search::new(self.config.clone()).expect("validated config"));
                }
            }
        }
        if let Some(error) = python_error {
            return Err(error);
        }
        if results.iter().any(Option::is_none) {
            return Err(PyRuntimeError::new_err(format!(
                "cloud scheduler completed {} of {rows} searches",
                results.iter().filter(|result| result.is_some()).count()
            )));
        }
        let mut output = Vec::with_capacity(rows);
        for result in results {
            let (choice, search) = result.expect("checked").map_err(to_py)?;
            output.push(search_output(py, search, choice)?.unbind());
        }
        Ok(output)
    }
}

// ── Cloud scheduler: resumable searches on a fixed worker pool ──────────

struct CloudTask {
    input: usize,
    slot: usize,
    session: Option<PlaySession>,
}

struct CloudRequest {
    worker: usize,
    row: EvalRow,
}

enum CloudWorkerMessage {
    Request(CloudRequest),
    Finished(Finished),
    Barrier(usize),
    Done(usize, WorkerProfile),
    Panicked(usize),
}

enum CloudCommand {
    Start(Vec<CloudTask>),
    Responses(Vec<(usize, EvalResponse)>),
    Cancel,
    Shutdown,
}

#[derive(Default)]
struct WorkerProfile {
    search_ns: u128,
    encode_ns: u128,
    requests: u64,
}

#[derive(Default)]
struct SchedulerProfile {
    search_ns: u128,
    encode_ns: u128,
    coordinator_wait_ns: u128,
    pack_ns: u128,
    python_eval_ns: u128,
    decode_ns: u128,
    wall_ns: u128,
    requests: u64,
    calls: u64,
    waves: u64,
    plays: u64,
    batch_widths: HashMap<usize, u64>,
}

impl SchedulerProfile {
    fn add_worker(&mut self, worker: WorkerProfile) {
        self.search_ns += worker.search_ns;
        self.encode_ns += worker.encode_ns;
        self.requests += worker.requests;
    }

    fn add_chunk(&mut self, rows: usize, chunk: ChunkProfile) {
        self.pack_ns += chunk.pack_ns;
        self.python_eval_ns += chunk.python_eval_ns;
        self.decode_ns += chunk.decode_ns;
        self.calls += 1;
        *self.batch_widths.entry(rows).or_default() += 1;
    }
}

fn emit_cloud_event(
    task: &mut CloudTask,
    response: Option<EvalResponse>,
    worker: usize,
    messages: &mpsc::Sender<CloudWorkerMessage>,
    profile: &mut WorkerProfile,
) -> bool {
    let started = Instant::now();
    let event = match response {
        Some(response) => task
            .session
            .as_mut()
            .expect("live session")
            .resume(response),
        None => task.session.as_mut().expect("live session").next_event(),
    };
    profile.search_ns += started.elapsed().as_nanos();
    match event {
        Ok(SearchEvent::Evaluation(request)) => {
            let encode_started = Instant::now();
            let encoded = encoder::encode_state(&request.state, request.viewer);
            let legal = macro_codec::legal_macros(&request.state);
            profile.encode_ns += encode_started.elapsed().as_nanos();
            match (encoded, legal) {
                (Ok(encoded), Ok(legal)) => {
                    profile.requests += 1;
                    messages
                        .send(CloudWorkerMessage::Request(CloudRequest {
                            worker,
                            row: EvalRow {
                                input: task.input,
                                kind: request.kind,
                                encoded,
                                legal,
                                seats: request.state.config.players,
                                abi_id: request.request_id,
                            },
                        }))
                        .is_ok()
                }
                (Err(error), _) | (_, Err(error)) => {
                    let input = task.input;
                    let slot = task.slot;
                    let search = task.session.take().expect("live session").into_search();
                    let _ = messages.send(CloudWorkerMessage::Finished(Finished {
                        input,
                        slot,
                        search,
                        result: Err(error),
                    }));
                    false
                }
            }
        }
        Ok(SearchEvent::Complete) => {
            let input = task.input;
            let slot = task.slot;
            let session = task.session.take().expect("live session");
            let (search, choice, output) = session
                .into_parts()
                .expect("SearchEvent::Complete carries a result");
            let _ = messages.send(CloudWorkerMessage::Finished(Finished {
                input,
                slot,
                search,
                result: Ok((Some(choice), output)),
            }));
            false
        }
        Err(error) => {
            let input = task.input;
            let slot = task.slot;
            let search = task.session.take().expect("live session").into_search();
            let _ = messages.send(CloudWorkerMessage::Finished(Finished {
                input,
                slot,
                search,
                result: Err(error),
            }));
            false
        }
    }
}

fn run_cloud_worker(
    worker: usize,
    messages: mpsc::Sender<CloudWorkerMessage>,
    commands: mpsc::Receiver<CloudCommand>,
) {
    loop {
        let mut tasks = match commands.recv() {
            Ok(CloudCommand::Start(tasks)) => tasks,
            Ok(CloudCommand::Shutdown) | Err(_) => return,
            Ok(CloudCommand::Responses(_)) | Ok(CloudCommand::Cancel) => {
                panic!("idle cloud worker received a non-start command")
            }
        };
        let mut profile = WorkerProfile::default();
        tasks.retain_mut(|task| emit_cloud_event(task, None, worker, &messages, &mut profile));
        loop {
            if tasks.is_empty() {
                let _ = messages.send(CloudWorkerMessage::Done(worker, profile));
                break;
            }
            if messages.send(CloudWorkerMessage::Barrier(worker)).is_err() {
                return;
            }
            match commands.recv() {
                Ok(CloudCommand::Responses(responses)) => {
                    let mut by_input: HashMap<usize, EvalResponse> =
                        responses.into_iter().collect();
                    tasks.retain_mut(|task| {
                        let response = by_input
                            .remove(&task.input)
                            .expect("coordinator returned one response per suspended search");
                        emit_cloud_event(task, Some(response), worker, &messages, &mut profile)
                    });
                    debug_assert!(by_input.is_empty());
                }
                Ok(CloudCommand::Cancel) => {
                    for task in tasks.drain(..) {
                        let _ = messages.send(CloudWorkerMessage::Finished(Finished {
                            input: task.input,
                            slot: task.slot,
                            search: task.session.expect("live session").into_search(),
                            result: Err(EngineError::Invalid("global evaluator cancelled".into())),
                        }));
                    }
                }
                Ok(CloudCommand::Shutdown) | Err(_) => return,
                Ok(CloudCommand::Start(_)) => {
                    panic!("busy cloud worker received a second start command")
                }
            }
        }
    }
}

/// Cloud-sized scheduler: many resumable searches, a fixed Rust worker pool,
/// and one deterministic global inference barrier. Unlike `RustScheduler`, the
/// worker count is independent of inflight games.
#[pyclass(module = "welcome_to_rust")]
pub struct RustCloudScheduler {
    searches: Vec<Option<Search>>,
    config: SearchConfig,
    workers: usize,
    batch: PackedBatch,
    next_request_id: u32,
    profile: SchedulerProfile,
    commands: Vec<mpsc::Sender<CloudCommand>>,
    messages: Mutex<mpsc::Receiver<CloudWorkerMessage>>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Drop for RustCloudScheduler {
    fn drop(&mut self) {
        for command in &self.commands {
            let _ = command.send(CloudCommand::Shutdown);
        }
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

#[pymethods]
impl RustCloudScheduler {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        capacity=256,
        workers=8,
        simulations=128,
        c_puct=1.5,
        alpha=0.5,
        margin_gain=2.0,
        confidence_power=1.0,
        prune_roundabout_pass=true,
        chance_widening=Some(1.0),
        chance_widening_alpha=0.5,
        max_particles=4,
        noise_fresh_fraction=1.0,
        dirichlet_weight=0.25,
        temperature=0.0,
        noise_required=false
    ))]
    fn new(
        capacity: usize,
        workers: usize,
        simulations: usize,
        c_puct: f64,
        alpha: f64,
        margin_gain: f64,
        confidence_power: f64,
        prune_roundabout_pass: bool,
        chance_widening: Option<f64>,
        chance_widening_alpha: f64,
        max_particles: usize,
        noise_fresh_fraction: f64,
        dirichlet_weight: f64,
        temperature: f64,
        noise_required: bool,
    ) -> PyResult<Self> {
        if capacity == 0 || workers == 0 {
            return Err(PyValueError::new_err(
                "scheduler capacity and worker count must be positive",
            ));
        }
        let config = config(
            simulations,
            c_puct,
            alpha,
            margin_gain,
            confidence_power,
            prune_roundabout_pass,
            chance_widening,
            chance_widening_alpha,
            max_particles,
            noise_fresh_fraction,
            dirichlet_weight,
            temperature,
            noise_required,
        );
        let mut searches = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            searches.push(Some(Search::new(config.clone()).map_err(to_py)?));
        }
        let (message_tx, message_rx) = mpsc::channel();
        let mut commands = Vec::with_capacity(workers);
        let mut threads = Vec::with_capacity(workers);
        for worker in 0..workers {
            let (command_tx, command_rx) = mpsc::channel();
            commands.push(command_tx);
            let messages = message_tx.clone();
            let panic_messages = messages.clone();
            threads.push(std::thread::spawn(move || {
                if catch_unwind(AssertUnwindSafe(|| {
                    run_cloud_worker(worker, messages, command_rx)
                }))
                .is_err()
                {
                    let _ = panic_messages.send(CloudWorkerMessage::Panicked(worker));
                }
            }));
        }
        drop(message_tx);
        Ok(Self {
            searches,
            config,
            workers,
            batch: PackedBatch::default(),
            next_request_id: 0,
            profile: SchedulerProfile::default(),
            commands,
            messages: Mutex::new(message_rx),
            threads,
        })
    }

    #[pyo3(signature = (states, evaluator, seeds, roots=None, noises=None, temperatures=None, slots=None, max_batch=256))]
    #[allow(clippy::too_many_arguments)]
    fn play<'py>(
        &mut self,
        py: Python<'py>,
        states: Vec<Py<RustGameState>>,
        evaluator: &Bound<'py, PyAny>,
        seeds: Vec<u64>,
        roots: Option<Vec<usize>>,
        noises: Option<Vec<Option<Vec<f64>>>>,
        temperatures: Option<Vec<f64>>,
        slots: Option<Vec<usize>>,
        max_batch: usize,
    ) -> PyResult<Vec<Py<PyDict>>> {
        self.run(
            py,
            states,
            evaluator,
            seeds,
            roots,
            noises,
            temperatures,
            slots,
            max_batch,
        )
    }

    #[pyo3(signature = (slot=None))]
    fn reset(&mut self, slot: Option<usize>) -> PyResult<()> {
        match slot {
            Some(slot) => self
                .searches
                .get_mut(slot)
                .ok_or_else(|| PyValueError::new_err("scheduler slot is out of range"))?
                .as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("scheduler slot is busy"))?
                .reset(),
            None => {
                for search in &mut self.searches {
                    search
                        .as_mut()
                        .ok_or_else(|| PyRuntimeError::new_err("scheduler slot is busy"))?
                        .reset();
                }
            }
        }
        Ok(())
    }

    fn stats<'py>(&self, py: Python<'py>, slot: usize) -> PyResult<Bound<'py, PyDict>> {
        search_stats(py, &self.searches, slot)
    }

    fn profile<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("workers", self.workers)?;
        out.set_item("plays", self.profile.plays)?;
        out.set_item("waves", self.profile.waves)?;
        out.set_item("requests", self.profile.requests)?;
        out.set_item("calls", self.profile.calls)?;
        out.set_item("search_ms", self.profile.search_ns as f64 / 1e6)?;
        out.set_item("encode_ms", self.profile.encode_ns as f64 / 1e6)?;
        out.set_item(
            "coordinator_wait_ms",
            self.profile.coordinator_wait_ns as f64 / 1e6,
        )?;
        out.set_item("pack_ms", self.profile.pack_ns as f64 / 1e6)?;
        out.set_item("python_eval_ms", self.profile.python_eval_ns as f64 / 1e6)?;
        out.set_item("decode_ms", self.profile.decode_ns as f64 / 1e6)?;
        out.set_item("wall_ms", self.profile.wall_ns as f64 / 1e6)?;
        let widths = PyDict::new(py);
        let mut entries: Vec<_> = self.profile.batch_widths.iter().collect();
        entries.sort_by_key(|(width, _)| **width);
        for (width, calls) in entries {
            widths.set_item(width, calls)?;
        }
        out.set_item("batch_widths", widths)?;
        Ok(out)
    }

    fn reset_profile(&mut self) {
        self.profile = SchedulerProfile::default();
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.searches.len()
    }

    #[getter]
    fn workers(&self) -> usize {
        self.workers
    }
}

impl RustScheduler {
    #[allow(clippy::too_many_arguments)]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        states: Vec<Py<RustGameState>>,
        evaluator: &Bound<'py, PyAny>,
        seeds: Vec<u64>,
        roots: Option<Vec<usize>>,
        noises: Option<Vec<Option<Vec<f64>>>>,
        temperatures: Option<Vec<f64>>,
        slots: Option<Vec<usize>>,
        max_batch: usize,
        play: bool,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let rows = states.len();
        if rows == 0 {
            return Ok(Vec::new());
        }
        if max_batch == 0 {
            return Err(PyValueError::new_err("max_batch must be positive"));
        }
        if seeds.len() != rows {
            return Err(PyValueError::new_err(
                "states and seeds are not row-aligned",
            ));
        }
        let slots = slots.unwrap_or_else(|| (0..rows).collect());
        if slots.len() != rows || slots.iter().any(|&slot| slot >= self.searches.len()) {
            return Err(PyValueError::new_err(
                "scheduler slots are not valid and row-aligned",
            ));
        }
        if slots.iter().copied().collect::<HashSet<_>>().len() != rows {
            return Err(PyValueError::new_err(
                "scheduler slots must be unique within a wave",
            ));
        }
        let roots = match roots {
            Some(roots) if roots.len() == rows => roots,
            Some(_) => return Err(PyValueError::new_err("roots are not row-aligned")),
            None => states
                .iter()
                .map(|state| state.bind(py).borrow().inner.actor)
                .collect(),
        };
        let noises = match noises {
            Some(noises) if noises.len() == rows => noises,
            Some(_) => return Err(PyValueError::new_err("noises are not row-aligned")),
            None => (0..rows).map(|_| None).collect(),
        };
        let temperatures = match temperatures {
            Some(_) if !play => {
                return Err(PyValueError::new_err(
                    "temperature overrides apply only to play, not search",
                ))
            }
            Some(temperatures) if temperatures.len() == rows => temperatures
                .into_iter()
                .map(Some)
                .collect::<Vec<Option<f64>>>(),
            Some(_) => return Err(PyValueError::new_err("temperatures are not row-aligned")),
            None => (0..rows).map(|_| None).collect(),
        };

        let games: Vec<Game> = states
            .iter()
            .map(|state| state.bind(py).borrow().inner.clone())
            .collect();
        if slots.iter().any(|&slot| self.searches[slot].is_none()) {
            return Err(PyRuntimeError::new_err("scheduler slot is already busy"));
        }
        let mut tasks = Vec::with_capacity(rows);
        for input in 0..rows {
            let search = self.searches[slots[input]]
                .take()
                .ok_or_else(|| PyRuntimeError::new_err("scheduler slot is already busy"))?;
            tasks.push((
                input,
                slots[input],
                search,
                games[input].clone(),
                roots[input],
                seeds[input],
                noises[input].clone(),
                temperatures[input],
            ));
        }

        let (message_tx, message_rx) = mpsc::channel::<WorkerMessage>();
        // std::mpsc::Receiver is Send but not Sync. The mutex is uncontended
        // (only this coordinator receives) and lets `py.detach` prove the
        // blocking closure is Ungil rather than retaining the GIL at a barrier.
        let message_rx = Mutex::new(message_rx);
        let mut finished = Vec::with_capacity(rows);
        let mut python_error: Option<PyErr> = None;
        let mut live = rows;

        std::thread::scope(|scope| {
            for (input, slot, mut search, state, root, seed, noise, temperature) in tasks {
                let messages = message_tx.clone();
                scope.spawn(move || {
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        let mut adapter =
                            |kind: RequestKind, position: &Game, viewer: usize, _local_id: u32| {
                                let encoded = encoder::encode_state(position, viewer)?;
                                let legal = macro_codec::legal_macros(position)?;
                                let (reply_tx, reply_rx) = mpsc::channel();
                                messages
                                    .send(WorkerMessage::Request(EvalRequest {
                                        row: EvalRow {
                                            input,
                                            kind,
                                            encoded,
                                            legal,
                                            seats: position.config.players,
                                            abi_id: 0,
                                        },
                                        reply: reply_tx,
                                    }))
                                    .map_err(|_| {
                                        EngineError::Invalid(
                                            "global evaluator coordinator stopped".into(),
                                        )
                                    })?;
                                reply_rx
                                    .recv()
                                    .map_err(|_| {
                                        EngineError::Invalid(
                                            "global evaluator dropped its response".into(),
                                        )
                                    })?
                                    .map_err(EngineError::Invalid)
                            };
                        if play {
                            let result = match temperature {
                                Some(temperature) => search.play_with_temperature(
                                    &state,
                                    root,
                                    seed,
                                    &mut adapter,
                                    noise.as_deref(),
                                    temperature,
                                ),
                                None => {
                                    search.play(&state, root, seed, &mut adapter, noise.as_deref())
                                }
                            };
                            result.map(|(choice, output)| (Some(choice), output))
                        } else {
                            search
                                .search(&state, root, seed, &mut adapter, noise.as_deref())
                                .map(|output| (None, output))
                        }
                    }));
                    let result = match result {
                        Ok(result) => result,
                        Err(_) => Err(EngineError::Invalid("search worker panicked".into())),
                    };
                    let _ = messages.send(WorkerMessage::Finished(Finished {
                        input,
                        slot,
                        search,
                        result,
                    }));
                });
            }
            drop(message_tx);

            while live > 0 {
                let mut pending = Vec::with_capacity(live);
                while pending.len() < live {
                    let message = py.detach(|| {
                        message_rx
                            .lock()
                            .expect("scheduler receiver lock poisoned")
                            .recv()
                    });
                    match message {
                        Ok(WorkerMessage::Request(request)) => pending.push(request),
                        Ok(WorkerMessage::Finished(done)) => {
                            live -= 1;
                            finished.push(done);
                        }
                        Err(_) => {
                            python_error.get_or_insert_with(|| {
                                PyRuntimeError::new_err(
                                    "global search workers disconnected before completion",
                                )
                            });
                            live = 0;
                            break;
                        }
                    }
                }
                if pending.is_empty() {
                    continue;
                }
                // Thread arrival order is timing, not game state. Stable input
                // order makes batch geometry and response tapes reproducible.
                pending.sort_by_key(|request| request.row.input);
                for chunk in pending.chunks_mut(max_batch) {
                    for request in chunk.iter_mut() {
                        request.row.abi_id = self.next_request_id;
                        self.next_request_id = self.next_request_id.wrapping_add(1);
                    }
                    if python_error.is_none() {
                        let rows: Vec<&EvalRow> =
                            chunk.iter().map(|request| &request.row).collect();
                        match evaluate_chunk(py, evaluator, &mut self.batch, &rows) {
                            Ok((responses, _profile)) => {
                                for (request, response) in chunk.iter().zip(responses) {
                                    let _ = request.reply.send(Ok(response));
                                }
                            }
                            Err(error) => {
                                python_error = Some(error);
                                for request in chunk.iter() {
                                    let _ = request
                                        .reply
                                        .send(Err("Python batch evaluator failed".into()));
                                }
                            }
                        }
                    } else {
                        for request in chunk.iter() {
                            let _ = request.reply.send(Err("global evaluator cancelled".into()));
                        }
                    }
                }
            }
        });

        finished.sort_by_key(|done| done.input);
        let mut results: Vec<Option<NativeResult>> = (0..rows).map(|_| None).collect();
        for done in finished {
            results[done.input] = Some(done.result);
            self.searches[done.slot] = Some(done.search);
        }
        if let Some(error) = python_error {
            return Err(error);
        }
        if results.iter().any(Option::is_none) {
            return Err(PyRuntimeError::new_err(format!(
                "scheduler completed {} of {rows} searches",
                results.iter().filter(|result| result.is_some()).count()
            )));
        }
        let mut output = Vec::with_capacity(rows);
        for result in results {
            let (choice, search) = result.expect("checked").map_err(to_py)?;
            output.push(search_output(py, search, choice)?.unbind());
        }
        Ok(output)
    }
}
