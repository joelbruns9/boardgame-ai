//! Leaf evaluator abstraction for the closed searcher (F3.2+).
//!
//! The tree is generic over `Eval`. `MockEval` is a deterministic
//! fingerprint-derived oracle used by the F3.2/F3.3 tree-equivalence gates:
//! value and priors are pure functions of the state fingerprint, so Python and
//! Rust — sharing the fingerprint and the same splitmix mixing — evaluate every
//! state identically without a neural net. The real batched-NN evaluator arrives
//! in F3.4 as another `Eval` impl.

use crate::codec::legal_action_indices;
use crate::state::{GameState, Phase};
use pyo3::exceptions::{PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyDict};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// `(value_p0, priors)` where `priors` is aligned to `legal_action_indices`.
/// Terminal states return the game value and empty priors. Fallible so a real
/// evaluator can surface operational errors (CUDA OOM, a bad checkpoint, a
/// contract violation) as a `PyErr` through the search rather than panicking.
pub trait Eval {
    fn evaluate(&self, state: &GameState) -> PyResult<(f64, Vec<f64>)>;

    /// F4.2 local batching boundary. Implementations may override this with a
    /// true vectorized evaluator; the default preserves alignment and error
    /// propagation by evaluating the supplied states in order. F4.4/F4.5 replace
    /// `PyEval`'s scalar fallback with the global Torch bridge.
    fn evaluate_batch(&self, states: &[&GameState]) -> PyResult<Vec<(f64, Vec<f64>)>> {
        states.iter().map(|state| self.evaluate(state)).collect()
    }

    /// F4.5 metadata-aware boundary. Search nodes already cache actor and legal
    /// actions, so production evaluators can avoid deriving them again while
    /// packing a global batch. Legacy/scalar evaluators retain the old behavior.
    fn evaluate_batch_prepared(
        &self,
        states: &[&GameState],
        actors: &[usize],
        legals: &[Vec<usize>],
    ) -> PyResult<Vec<(f64, Vec<f64>)>> {
        if states.len() != actors.len() || states.len() != legals.len() {
            return Err(PyValueError::new_err(
                "prepared evaluator metadata is not row-aligned",
            ));
        }
        self.evaluate_batch(states)
    }
}

pub fn terminal_value_p0(state: &GameState) -> f64 {
    match state.winner {
        None => 0.0,
        Some(0) => 1.0,
        Some(_) => -1.0,
    }
}

pub struct MockEval;

fn mix(mut h: u64) -> u64 {
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

fn fold_fingerprint(fp: &[i32]) -> u64 {
    let mut h = 0x9E37_79B9_7F4A_7C15_u64;
    for &x in fp {
        h ^= x as u64; // i32 sign-extends to u64, matching Python's x & MASK64
        h = mix(h);
    }
    h
}

fn to_unit(h: u64) -> f64 {
    (h >> 11) as f64 / 9_007_199_254_740_992.0
}

impl MockEval {
    /// Standalone value+priors for one state — mirrors the Python `mock_eval`
    /// reference. `priors` are raw (unnormalized) per-action weights aligned to
    /// the sorted legal indices (empty at terminals); see `evaluate` for why.
    pub fn eval_state(state: &GameState) -> (f64, Vec<f64>) {
        let fp = state.fingerprint();
        let h = fold_fingerprint(&fp);
        let value_p0 = to_unit(h) * 2.0 - 1.0;
        if state.phase == Phase::Complete {
            return (terminal_value_p0(state), Vec::new());
        }
        // Raw per-action weights in [0,1) — deliberately NOT normalized.
        // Normalizing needs a cross-language sum that diverges in the last ULP;
        // leaving them raw keeps the oracle bit-identical on both sides, which is
        // all the equivalence gate needs (Python and Rust consume the SAME priors
        // and so build the SAME tree). NOTE: raw weights are not a probability
        // distribution, so this oracle does NOT reproduce a normalized
        // evaluator's PUCT exploration — `Q + c_puct*prior*...` is not
        // scale-invariant. F3.4 must gate against production-shaped normalized
        // priors.
        let legal = legal_action_indices(state);
        let priors = legal
            .iter()
            .map(|&a| to_unit(mix(h ^ (a as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))))
            .collect();
        (value_p0, priors)
    }
}

impl Eval for MockEval {
    fn evaluate(&self, state: &GameState) -> PyResult<(f64, Vec<f64>)> {
        Ok(MockEval::eval_state(state))
    }
}

/// F3.4: real-net evaluator. Encodes with the Rust F2 encoder and calls a Python
/// adapter `(tokens, actor, legal) -> (value_actor, priors)` that runs the net —
/// so the Rust searcher uses identical net inputs/outputs to Python's reference.
/// This is a *scalar* per-leaf bridge for correctness; F4 replaces it with leaf
/// coalescing + GIL release for throughput (do NOT make this the production
/// batching boundary).
pub struct PyEval {
    adapter: Py<PyAny>,
}

impl PyEval {
    pub fn new(adapter: Py<PyAny>) -> Self {
        PyEval { adapter }
    }
}

impl Eval for PyEval {
    fn evaluate(&self, state: &GameState) -> PyResult<(f64, Vec<f64>)> {
        if state.phase == Phase::Complete {
            return Ok((terminal_value_p0(state), Vec::new()));
        }
        let actor = crate::tree::state_actor(state);
        let tokens: Vec<(usize, i32, i32, Vec<f64>)> = crate::encoder::encode(state)
            .into_iter()
            .map(|t| (t.type_id, t.entity_id, t.aux_id, t.features))
            .collect();
        let legal = legal_action_indices(state);
        let n = legal.len();
        Python::attach(|py| {
            // Propagate the adapter's own PyErr (OOM, bad checkpoint, ...).
            let out = self.adapter.bind(py).call1((tokens, actor, legal))?;
            let (value_actor, priors): (f64, Vec<f64>) = out.extract()?;
            // Validate the evaluator contract before the search trusts it.
            if !value_actor.is_finite() {
                return Err(PyValueError::new_err("net returned a non-finite value"));
            }
            if priors.len() != n {
                return Err(PyValueError::new_err(format!(
                    "net returned {} priors for {n} legal actions",
                    priors.len()
                )));
            }
            let mut mass = 0.0;
            for &p in &priors {
                if !p.is_finite() || p < 0.0 {
                    return Err(PyValueError::new_err(
                        "net returned a non-finite or negative prior",
                    ));
                }
                mass += p;
            }
            if mass <= 0.0 {
                return Err(PyValueError::new_err("net returned a zero-mass policy"));
            }
            let value_p0 = if actor == 0 {
                value_actor
            } else {
                -value_actor
            };
            Ok((value_p0, priors))
        })
    }
}

/// F4.4 global coalescer boundary. The adapter is called once with an ordered
/// list of `(tokens, actor, legal)` rows and must return the same number of
/// `(value_actor, legal_priors)` rows. Ownership/order validation happens before
/// any search session receives a result.
pub struct PyBatchEval {
    adapter: Py<PyAny>,
}

impl PyBatchEval {
    pub fn new(adapter: Py<PyAny>) -> Self {
        Self { adapter }
    }
}

impl Eval for PyBatchEval {
    fn evaluate(&self, state: &GameState) -> PyResult<(f64, Vec<f64>)> {
        let mut rows = self.evaluate_batch(&[state])?;
        Ok(rows.remove(0))
    }

    fn evaluate_batch(&self, states: &[&GameState]) -> PyResult<Vec<(f64, Vec<f64>)>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let mut actors = Vec::with_capacity(states.len());
        let mut legal_counts = Vec::with_capacity(states.len());
        let rows: Vec<_> = states
            .iter()
            .map(|state| {
                let actor = crate::tree::state_actor(state);
                let tokens: Vec<(usize, i32, i32, Vec<f64>)> = crate::encoder::encode(state)
                    .into_iter()
                    .map(|t| (t.type_id, t.entity_id, t.aux_id, t.features))
                    .collect();
                let legal = legal_action_indices(state);
                actors.push(actor);
                legal_counts.push(legal.len());
                (tokens, actor, legal)
            })
            .collect();
        Python::attach(|py| {
            let out = self.adapter.bind(py).call1((rows,))?;
            let raw: Vec<(f64, Vec<f64>)> = out.extract()?;
            if raw.len() != states.len() {
                return Err(PyValueError::new_err(format!(
                    "batch net returned {} rows for {} states",
                    raw.len(),
                    states.len()
                )));
            }
            raw.into_iter()
                .enumerate()
                .map(|(row, (value_actor, priors))| {
                    if states[row].phase == Phase::Complete {
                        if !priors.is_empty() {
                            return Err(PyValueError::new_err(format!(
                                "batch net terminal row {row} returned policy priors"
                            )));
                        }
                        return Ok((terminal_value_p0(states[row]), Vec::new()));
                    }
                    if !value_actor.is_finite() {
                        return Err(PyValueError::new_err(format!(
                            "batch net row {row} returned a non-finite value"
                        )));
                    }
                    if priors.len() != legal_counts[row] {
                        return Err(PyValueError::new_err(format!(
                            "batch net row {row} returned {} priors for {} legal actions",
                            priors.len(),
                            legal_counts[row]
                        )));
                    }
                    let mut mass = 0.0;
                    for &prior in &priors {
                        if !prior.is_finite() || prior < 0.0 {
                            return Err(PyValueError::new_err(format!(
                                "batch net row {row} returned a non-finite or negative prior"
                            )));
                        }
                        mass += prior;
                    }
                    if mass <= 0.0 {
                        return Err(PyValueError::new_err(format!(
                            "batch net row {row} returned a zero-mass policy"
                        )));
                    }
                    let value_p0 = if actors[row] == 0 {
                        value_actor
                    } else {
                        -value_actor
                    };
                    Ok((value_p0, priors))
                })
                .collect()
        })
    }
}

const FLAT_FEATURE_WIDTH: usize = 130;

#[derive(Clone, Debug, Default)]
pub struct BoundaryMetrics {
    pub batches: usize,
    pub rows: usize,
    pub tokens: usize,
    pub padded_tokens: usize,
    pub max_tokens: usize,
    pub encode_pack_ns: u64,
    pub queue_wait_ns: u64,
    pub py_call_ns: u64,
    pub extract_ns: u64,
}

#[derive(Default)]
struct FlatBatchBuilder {
    token_offsets: Vec<u8>,
    type_ids: Vec<u8>,
    entity_ids: Vec<u8>,
    aux_ids: Vec<u8>,
    features: Vec<u8>,
    actors: Vec<u8>,
    legal_offsets: Vec<u8>,
    legal_actions: Vec<u8>,
    rows: usize,
    tokens: usize,
    max_tokens: usize,
}

fn push_u32(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u32).to_le_bytes());
}

impl FlatBatchBuilder {
    fn clear(&mut self) {
        self.token_offsets.clear();
        self.type_ids.clear();
        self.entity_ids.clear();
        self.aux_ids.clear();
        self.features.clear();
        self.actors.clear();
        self.legal_offsets.clear();
        self.legal_actions.clear();
        self.rows = 0;
        self.tokens = 0;
        self.max_tokens = 0;
    }

    fn pack(&mut self, states: &[&GameState], actors: &[usize], legals: &[Vec<usize>]) {
        self.clear();
        push_u32(&mut self.token_offsets, 0);
        push_u32(&mut self.legal_offsets, 0);
        for ((state, &actor), legal) in states.iter().zip(actors).zip(legals) {
            let tokens = crate::encoder::encode(state);
            self.actors.push(actor as u8);
            self.max_tokens = self.max_tokens.max(tokens.len());
            for token in tokens {
                self.type_ids.push(token.type_id as u8);
                self.entity_ids
                    .extend_from_slice(&(token.entity_id as i16).to_le_bytes());
                self.aux_ids
                    .extend_from_slice(&((token.aux_id + 1) as i16).to_le_bytes());
                for index in 0..FLAT_FEATURE_WIDTH {
                    // The flat Python/Torch boundary is explicitly f32; keeping
                    // this cast here permits a zero-copy tensor view of the bytes.
                    let value = token.features.get(index).copied().unwrap_or(0.0) as f32;
                    self.features.extend_from_slice(&value.to_le_bytes());
                }
                self.tokens += 1;
            }
            push_u32(&mut self.token_offsets, self.tokens);
            for &action in legal {
                self.legal_actions
                    .extend_from_slice(&(action as u16).to_le_bytes());
            }
            let legal_total = self.legal_actions.len() / 2;
            push_u32(&mut self.legal_offsets, legal_total);
        }
        self.rows = states.len();
    }
}

pub struct PyFlatBatchEval {
    adapter: Py<PyAny>,
    scratch: Mutex<FlatBatchBuilder>,
    metrics: Arc<Mutex<BoundaryMetrics>>,
}

impl PyFlatBatchEval {
    pub fn new(adapter: Py<PyAny>, metrics: Arc<Mutex<BoundaryMetrics>>) -> Self {
        Self {
            adapter,
            scratch: Mutex::new(FlatBatchBuilder::default()),
            metrics,
        }
    }
}

impl Eval for PyFlatBatchEval {
    fn evaluate(&self, state: &GameState) -> PyResult<(f64, Vec<f64>)> {
        let mut rows = self.evaluate_batch(&[state])?;
        Ok(rows.remove(0))
    }

    fn evaluate_batch(&self, states: &[&GameState]) -> PyResult<Vec<(f64, Vec<f64>)>> {
        let actors: Vec<_> = states
            .iter()
            .map(|state| crate::tree::state_actor(state))
            .collect();
        let legals: Vec<_> = states
            .iter()
            .map(|state| legal_action_indices(state))
            .collect();
        self.evaluate_batch_prepared(states, &actors, &legals)
    }

    fn evaluate_batch_prepared(
        &self,
        states: &[&GameState],
        actors: &[usize],
        legals: &[Vec<usize>],
    ) -> PyResult<Vec<(f64, Vec<f64>)>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        if states.len() != actors.len() || states.len() != legals.len() {
            return Err(PyValueError::new_err(
                "flat evaluator metadata is not row-aligned",
            ));
        }
        let pack_start = Instant::now();
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| PyValueError::new_err("flat batch scratch lock poisoned"))?;
        scratch.pack(states, actors, legals);
        let legal_counts: Vec<_> = legals.iter().map(Vec::len).collect();
        let pack_ns = pack_start.elapsed().as_nanos() as u64;
        let rows = scratch.rows;
        let tokens = scratch.tokens;
        let max_tokens = scratch.max_tokens;
        let (raw, call_ns, extract_ns) = Python::attach(|py| {
            let payload = PyDict::new(py);
            payload.set_item("rows", rows)?;
            payload.set_item("tokens", tokens)?;
            payload.set_item("max_tokens", max_tokens)?;
            payload.set_item("feature_width", FLAT_FEATURE_WIDTH)?;
            // Writable buffers let Torch create zero-copy CPU tensor views
            // without warning that an immutable Python `bytes` object could be
            // mutated through the view. The adapter treats them as read-only.
            payload.set_item(
                "token_offsets",
                PyByteArray::new(py, &scratch.token_offsets),
            )?;
            payload.set_item("type_ids", PyByteArray::new(py, &scratch.type_ids))?;
            payload.set_item("entity_ids", PyByteArray::new(py, &scratch.entity_ids))?;
            payload.set_item("aux_ids", PyByteArray::new(py, &scratch.aux_ids))?;
            payload.set_item("features", PyByteArray::new(py, &scratch.features))?;
            payload.set_item("actors", PyByteArray::new(py, &scratch.actors))?;
            payload.set_item(
                "legal_offsets",
                PyByteArray::new(py, &scratch.legal_offsets),
            )?;
            payload.set_item(
                "legal_actions",
                PyByteArray::new(py, &scratch.legal_actions),
            )?;
            let call_start = Instant::now();
            let out = self.adapter.bind(py).call1((payload,))?;
            let call_ns = call_start.elapsed().as_nanos() as u64;
            let extract_start = Instant::now();
            let raw: Vec<(f64, Vec<f64>)> = out.extract()?;
            let extract_ns = extract_start.elapsed().as_nanos() as u64;
            Ok::<_, PyErr>((raw, call_ns, extract_ns))
        })?;
        drop(scratch);
        if raw.len() != states.len() {
            return Err(PyValueError::new_err(format!(
                "flat net returned {} rows for {} states",
                raw.len(),
                states.len()
            )));
        }
        let mut validated = Vec::with_capacity(raw.len());
        for (row, (value_actor, priors)) in raw.into_iter().enumerate() {
            if states[row].phase == Phase::Complete {
                if !priors.is_empty() {
                    return Err(PyValueError::new_err(format!(
                        "flat net terminal row {row} returned policy priors"
                    )));
                }
                validated.push((terminal_value_p0(states[row]), Vec::new()));
                continue;
            }
            if !value_actor.is_finite() || priors.len() != legal_counts[row] {
                return Err(PyValueError::new_err(format!(
                    "flat net row {row} violates value/prior alignment"
                )));
            }
            let mass: f64 = priors.iter().sum();
            if priors.iter().any(|p| !p.is_finite() || *p < 0.0) || mass <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "flat net row {row} returned an invalid policy"
                )));
            }
            validated.push((
                if actors[row] == 0 {
                    value_actor
                } else {
                    -value_actor
                },
                priors,
            ));
        }
        let mut metrics = self
            .metrics
            .lock()
            .map_err(|_| PyValueError::new_err("boundary metrics lock poisoned"))?;
        metrics.batches += 1;
        metrics.rows += rows;
        metrics.tokens += tokens;
        metrics.padded_tokens += rows * max_tokens;
        metrics.max_tokens = metrics.max_tokens.max(max_tokens);
        metrics.encode_pack_ns += pack_ns;
        metrics.py_call_ns += call_ns;
        metrics.extract_ns += extract_ns;
        Ok(validated)
    }
}

type WorkerResponse = PyResult<Vec<(f64, Vec<f64>)>>;
struct WorkerRequest {
    states: Vec<GameState>,
    actors: Vec<usize>,
    legals: Vec<Vec<usize>>,
    reply: mpsc::Sender<WorkerResponse>,
    enqueued: Instant,
}

/// Scheduler-side handle for the dedicated F4.4 Python inference thread. Rust
/// scheduling runs with the caller's GIL detached; only the worker attaches to
/// Python and invokes the batch adapter.
pub struct EvalWorker {
    sender: mpsc::Sender<WorkerRequest>,
    timeout: Option<Duration>,
    timed_out: Arc<AtomicBool>,
    max_rows: usize,
}

pub struct EvalTicket {
    receiver: mpsc::Receiver<WorkerResponse>,
    timeout: Option<Duration>,
    timed_out: Arc<AtomicBool>,
}

impl EvalTicket {
    pub fn wait(self) -> WorkerResponse {
        match self.timeout {
            Some(timeout) => match self.receiver.recv_timeout(timeout) {
                Ok(result) => result,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    self.timed_out.store(true, Ordering::Release);
                    Err(PyTimeoutError::new_err(format!(
                        "global inference batch timed out after {:.3} ms",
                        timeout.as_secs_f64() * 1000.0
                    )))
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => Err(PyValueError::new_err(
                    "global inference worker dropped its response",
                )),
            },
            None => self.receiver.recv().map_err(|_| {
                PyValueError::new_err("global inference worker dropped its response")
            })?,
        }
    }
}

impl EvalWorker {
    pub fn submit_prepared(
        &self,
        states: Vec<GameState>,
        actors: Vec<usize>,
        legals: Vec<Vec<usize>>,
    ) -> PyResult<EvalTicket> {
        if states.is_empty() || states.len() > self.max_rows {
            return Err(PyValueError::new_err(format!(
                "inference request has {} rows outside cap {}",
                states.len(),
                self.max_rows
            )));
        }
        if states.len() != actors.len() || states.len() != legals.len() {
            return Err(PyValueError::new_err(
                "inference request metadata is not row-aligned",
            ));
        }
        let (reply_tx, reply_rx) = mpsc::channel();
        self.sender
            .send(WorkerRequest {
                states,
                actors,
                legals,
                reply: reply_tx,
                enqueued: Instant::now(),
            })
            .map_err(|_| PyValueError::new_err("global inference worker is not running"))?;
        Ok(EvalTicket {
            receiver: reply_rx,
            timeout: self.timeout,
            timed_out: Arc::clone(&self.timed_out),
        })
    }
}

impl Eval for EvalWorker {
    fn evaluate(&self, state: &GameState) -> PyResult<(f64, Vec<f64>)> {
        let mut rows = self.evaluate_batch(&[state])?;
        Ok(rows.remove(0))
    }

    fn evaluate_batch(&self, states: &[&GameState]) -> PyResult<Vec<(f64, Vec<f64>)>> {
        let actors = states
            .iter()
            .map(|state| crate::tree::state_actor(state))
            .collect::<Vec<_>>();
        let legals = states
            .iter()
            .map(|state| legal_action_indices(state))
            .collect::<Vec<_>>();
        self.evaluate_batch_prepared(states, &actors, &legals)
    }

    fn evaluate_batch_prepared(
        &self,
        states: &[&GameState],
        actors: &[usize],
        legals: &[Vec<usize>],
    ) -> PyResult<Vec<(f64, Vec<f64>)>> {
        if states.len() != actors.len() || states.len() != legals.len() {
            return Err(PyValueError::new_err(
                "worker evaluator metadata is not row-aligned",
            ));
        }
        let mut output = Vec::with_capacity(states.len());
        for start in (0..states.len()).step_by(self.max_rows) {
            let end = (start + self.max_rows).min(states.len());
            let owned = states[start..end]
                .iter()
                .map(|state| (*state).clone())
                .collect();
            output.extend(
                self.submit_prepared(
                    owned,
                    actors[start..end].to_vec(),
                    legals[start..end].to_vec(),
                )?
                .wait()?,
            );
        }
        Ok(output)
    }
}

pub fn spawn_py_batch_worker(
    adapter: Py<PyAny>,
    timeout_ms: f64,
    max_rows: usize,
) -> PyResult<(EvalWorker, Arc<AtomicBool>, thread::JoinHandle<()>)> {
    if !timeout_ms.is_finite() || timeout_ms < 0.0 {
        return Err(PyValueError::new_err(
            "inference_timeout_ms must be finite and non-negative",
        ));
    }
    if max_rows == 0 {
        return Err(PyValueError::new_err(
            "inference worker max_rows must be positive",
        ));
    }
    let timeout = if timeout_ms == 0.0 {
        None
    } else {
        Some(Duration::from_secs_f64(timeout_ms / 1000.0))
    };
    let (request_tx, request_rx) = mpsc::channel::<WorkerRequest>();
    let timed_out = Arc::new(AtomicBool::new(false));
    let handle = thread::spawn(move || {
        let evaluator = PyBatchEval::new(adapter);
        while let Ok(request) = request_rx.recv() {
            let refs: Vec<&GameState> = request.states.iter().collect();
            let result = evaluator.evaluate_batch_prepared(&refs, &request.actors, &request.legals);
            let failed = result.is_err();
            if request.reply.send(result).is_err() || failed {
                break;
            }
        }
    });
    Ok((
        EvalWorker {
            sender: request_tx,
            timeout,
            timed_out: Arc::clone(&timed_out),
            max_rows,
        },
        timed_out,
        handle,
    ))
}

pub fn spawn_py_flat_worker(
    adapter: Py<PyAny>,
    timeout_ms: f64,
    max_rows: usize,
) -> PyResult<(
    EvalWorker,
    Arc<AtomicBool>,
    Arc<Mutex<BoundaryMetrics>>,
    thread::JoinHandle<()>,
)> {
    if !timeout_ms.is_finite() || timeout_ms < 0.0 {
        return Err(PyValueError::new_err(
            "inference_timeout_ms must be finite and non-negative",
        ));
    }
    if max_rows == 0 {
        return Err(PyValueError::new_err(
            "inference worker max_rows must be positive",
        ));
    }
    let timeout = if timeout_ms == 0.0 {
        None
    } else {
        Some(Duration::from_secs_f64(timeout_ms / 1000.0))
    };
    let (request_tx, request_rx) = mpsc::channel::<WorkerRequest>();
    let timed_out = Arc::new(AtomicBool::new(false));
    let metrics = Arc::new(Mutex::new(BoundaryMetrics::default()));
    let worker_metrics = Arc::clone(&metrics);
    let handle = thread::spawn(move || {
        let evaluator = PyFlatBatchEval::new(adapter, Arc::clone(&worker_metrics));
        while let Ok(request) = request_rx.recv() {
            if let Ok(mut counters) = worker_metrics.lock() {
                counters.queue_wait_ns += request.enqueued.elapsed().as_nanos() as u64;
            }
            let refs: Vec<&GameState> = request.states.iter().collect();
            let result = evaluator.evaluate_batch_prepared(&refs, &request.actors, &request.legals);
            let failed = result.is_err();
            if request.reply.send(result).is_err() || failed {
                break;
            }
        }
    });
    Ok((
        EvalWorker {
            sender: request_tx,
            timeout,
            timed_out: Arc::clone(&timed_out),
            max_rows,
        },
        timed_out,
        metrics,
        handle,
    ))
}
