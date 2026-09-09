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
use rayon::prelude::*;
use pyo3::types::{PyByteArray, PyDict};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

fn checked_bytearray<'py>(py: Python<'py>, src: &[u8]) -> PyResult<Bound<'py, PyByteArray>> {
    PyByteArray::new_with(py, src.len(), |destination| {
        destination.copy_from_slice(src);
        Ok(())
    })
}

/// `(value_p0, priors)` where `priors` is aligned to `legal_action_indices`.
/// Terminal states return the game value and empty priors. Fallible so a real
/// evaluator can surface operational errors (CUDA OOM, a bad checkpoint, a
/// contract violation) as a `PyErr` through the search rather than panicking.
/// W4: the seven-way winner x victory-type distribution for one leaf,
/// canonicalised to player 0 exactly as `value_p0` is.
///
/// Class order is `dataset.JOINT7_CLASSES`: p0 civil/science/military, then
/// p1's three, then draw. Canonicalising means SWAPPING the two triples when
/// the evaluated actor is player 1 -- the same operation as negating the
/// scalar, and for the same reason: sums taken across leaves with different
/// actors are otherwise adding two different questions together.
pub type Outlook = [f64; 7];

/// Actor-relative to player-0 terms. The draw class is its own mirror.
pub fn outlook_to_p0(outlook: Outlook, actor: usize) -> Outlook {
    if actor == 0 {
        return outlook;
    }
    [
        outlook[3], outlook[4], outlook[5],
        outlook[0], outlook[1], outlook[2],
        outlook[6],
    ]
}

/// One leaf's evaluation.
///
/// `outlook_p0` is `None` for every evaluator that does not produce one --
/// mock evaluators, the solver's boundary, and any net without W4's head -- so
/// carrying it costs nothing where it does not exist.
#[derive(Clone, Debug)]
pub struct LeafOut {
    pub value_p0: f64,
    pub priors: Vec<f64>,
    pub outlook_p0: Option<Outlook>,
}

impl LeafOut {
    /// The historical `(value, priors)` shape, with no outlook.
    pub fn scalar(value_p0: f64, priors: Vec<f64>) -> Self {
        Self { value_p0, priors, outlook_p0: None }
    }
}

impl From<(f64, Vec<f64>)> for LeafOut {
    fn from(pair: (f64, Vec<f64>)) -> Self {
        LeafOut::scalar(pair.0, pair.1)
    }
}

pub trait Eval {
    fn evaluate(&self, state: &GameState) -> PyResult<LeafOut>;

    /// F4.2 local batching boundary. Implementations may override this with a
    /// true vectorized evaluator; the default preserves alignment and error
    /// propagation by evaluating the supplied states in order. F4.4/F4.5 replace
    /// `PyEval`'s scalar fallback with the global Torch bridge.
    fn evaluate_batch(&self, states: &[&GameState]) -> PyResult<Vec<LeafOut>> {
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
    ) -> PyResult<Vec<LeafOut>> {
        if states.len() != actors.len() || states.len() != legals.len() {
            return Err(PyValueError::new_err(
                "prepared evaluator metadata is not row-aligned",
            ));
        }
        self.evaluate_batch(states)
    }

    /// W1.3 league boundary: as `evaluate_batch_prepared`, plus one **network
    /// id** per row so a single batch can mix games played by different
    /// checkpoints.
    ///
    /// The id is resolved per row from the **searcher** (the slot whose search
    /// produced the row), never from the leaf actor. When it is player 0's turn,
    /// player 0's network drives the *entire* search and evaluates every leaf,
    /// including the leaves where player 1 is to move -- that is what an "agent"
    /// means here and in deployment. Routing on the leaf actor instead would let
    /// the opponent's network evaluate the interior of my own tree, which is a
    /// different (and wrong) player. Kingdomino's `row_search_actors` documents
    /// the same distinction for its two-net rating path.
    ///
    /// The default ignores the ids, so every existing evaluator keeps its exact
    /// behaviour and a caller that supplies no routing is byte-identical to
    /// before.
    fn evaluate_batch_prepared_routed(
        &self,
        states: &[&GameState],
        actors: &[usize],
        legals: &[Vec<usize>],
        net_ids: &[u8],
    ) -> PyResult<Vec<LeafOut>> {
        // Empty means "one network for every row" -- the ordinary self-play
        // case, which the scheduler signals by sending no ids at all. Only a
        // non-empty, wrongly-sized slice is a caller error.
        if !net_ids.is_empty() && net_ids.len() != states.len() {
            return Err(PyValueError::new_err(
                "routed evaluator net ids are not row-aligned",
            ));
        }
        self.evaluate_batch_prepared(states, actors, legals)
    }
}

/// The EXACT seven-way outlook of a finished game, in player-0 terms.
///
/// Ground truth, not an estimate: a terminal state knows both who won and how.
/// Deep searches reach many terminals, so this is the part of a backed-up
/// distribution that carries no model error at all -- which matters, because a
/// searched target is otherwise only as good as the net that produced it.
///
/// Mirrors `dataset._joint7_class`: no winner is a draw, and a SHARED civilian
/// finish is a draw too -- it is the tie-break class, not a civilian win.
/// One adapter row: the historical `(value, priors)` pair, or that pair plus a
/// seven-way outlook.
///
/// Both shapes are accepted so every existing Python adapter -- and every test
/// stub -- keeps working untouched. A net without W4's head has nothing to send
/// and says so by sending the short form, which is not an error.
type RawRow = (f64, Vec<f64>, Option<Vec<f64>>);

fn extract_rows(out: &Bound<'_, PyAny>) -> PyResult<Vec<RawRow>> {
    if let Ok(rows) = out.extract::<Vec<RawRow>>() {
        return Ok(rows);
    }
    let short: Vec<(f64, Vec<f64>)> = out.extract()?;
    Ok(short
        .into_iter()
        .map(|(value, priors)| (value, priors, None))
        .collect())
}

/// Validate and canonicalise one adapter-supplied outlook.
fn adapter_outlook(row: usize, raw: Option<Vec<f64>>, actor: usize) -> PyResult<Option<Outlook>> {
    let Some(values) = raw else { return Ok(None) };
    if values.len() != 7 {
        return Err(PyValueError::new_err(format!(
            "net row {row} returned {} outlook classes, expected 7",
            values.len()
        )));
    }
    let mut outlook = [0.0f64; 7];
    let mut mass = 0.0;
    for (k, &value) in values.iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(PyValueError::new_err(format!(
                "net row {row} returned a non-finite or negative outlook class"
            )));
        }
        outlook[k] = value;
        mass += value;
    }
    // A distribution, not logits. Loud, because a caller that sent logits would
    // otherwise poison every backed-up sum with plausible-looking numbers.
    if (mass - 1.0).abs() > 1e-3 {
        return Err(PyValueError::new_err(format!(
            "net row {row} returned an outlook summing to {mass}, not 1"
        )));
    }
    Ok(Some(outlook_to_p0(outlook, actor)))
}

pub fn terminal_outlook_p0(state: &GameState) -> Outlook {
    let mut out = [0.0; 7];
    let offset = match state.victory_type {
        Some(crate::state::VictoryType::Civilian) => 0,
        Some(crate::state::VictoryType::Scientific) => 1,
        Some(crate::state::VictoryType::Military) => 2,
        _ => {
            out[6] = 1.0;
            return out;
        }
    };
    match state.winner {
        Some(0) => out[offset] = 1.0,
        Some(_) => out[3 + offset] = 1.0,
        None => out[6] = 1.0,
    }
    out
}


/// Which victory type a specialist is biased toward.
///
/// The names are `dataset.JOINT7_CLASSES`' three victory types, and `offset`
/// is that class's position inside a player's triple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VictoryClass {
    Civilian,
    Scientific,
    Military,
}

impl VictoryClass {
    pub fn offset(self) -> usize {
        match self {
            VictoryClass::Civilian => 0,
            VictoryClass::Scientific => 1,
            VictoryClass::Military => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            VictoryClass::Civilian => "civilian",
            VictoryClass::Scientific => "scientific",
            VictoryClass::Military => "military",
        }
    }

    pub fn from_name(name: &str) -> PyResult<Self> {
        match name {
            "civilian" => Ok(VictoryClass::Civilian),
            "scientific" | "science" => Ok(VictoryClass::Scientific),
            "military" => Ok(VictoryClass::Military),
            other => Err(PyValueError::new_err(format!(
                "unknown specialist victory type {other:?}"
            ))),
        }
    }
}

/// W7 S0: the specialist's leaf-utility bias.
///
/// A specialist is an ordinary agent whose SEARCH values its own victory type
/// above the win probability alone. The bias lives at the leaf, not in the
/// stored labels: the policy learns the visit distribution, so biasing the leaf
/// moves the visits, while reweighting a stored label moves nothing (that
/// target already points where the unbiased search pointed). Every value head
/// therefore stays a calibrated win probability.
///
/// # The sign convention, which is the whole of the difficulty
///
/// `Outlook` and the search utility are both **player-0 relative**
/// (`outlook_to_p0`), but "my victory type" is **specialist relative**. Writing
/// `value_p0 + lambda * outlook_p0[my_class]` with a specialist-relative index
/// rewards the OPPONENT's win whenever the specialist sits on seat 1. Hence the
/// two explicit forms:
///
/// ```text
/// seat 0:  utility_p0 = value_p0 + lambda * outlook_p0[p0_<type>_win]
/// seat 1:  utility_p0 = value_p0 - lambda * outlook_p0[p1_<type>_win]
/// ```
///
/// `symmetric` is the other agent, kept behind a flag and off by default: it
/// adds `-lambda * outlook_p0[opponent_<type>_win]` as well, which rewards
/// pursuing the type *and* penalises conceding it -- a mirror-player rather than
/// an attacker, and it moves defensive behaviour in exactly the dimension S0
/// exists to measure. Note it is seat-INDEPENDENT in p0 terms, which is a
/// property worth testing: both seats optimise the same scalar.
///
/// # Range
///
/// `value_p0` lies in [-1, 1]; the bonus lies in [0, lambda] (own-win) or
/// [-lambda, lambda] (symmetric), so the utility scale widens to
/// [-1-lambda, 1+lambda]. PUCT's `Q + c_puct * P * sqrt(N)/(1+n)` is NOT scale
/// invariant, so a nonzero lambda makes exploration relatively cheaper.
///
/// This applies to a GUMBEL search too, which an earlier version of this
/// comment denied. `sigma_vector` min-max rescales completed Q at the ROOT, so
/// the root's own halving is scale free -- but every interior node of a Gumbel
/// search still selects by PUCT on the raw utility, so the tree underneath is
/// affected exactly as a PUCT root's is.
///
/// Left explicit rather than silently compensated: rescaling `c_puct` with
/// lambda would fold two changes into one flag and make a training A/B
/// uninterpretable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeafBias {
    pub lambda: f64,
    pub victory: VictoryClass,
    /// The SPECIALIST's seat -- the searcher's, never the leaf actor's.
    pub seat: usize,
    pub symmetric: bool,
}

impl LeafBias {
    pub const NONE: LeafBias = LeafBias {
        lambda: 0.0,
        victory: VictoryClass::Scientific,
        seat: 0,
        symmetric: false,
    };

    pub fn new(lambda: f64, victory: VictoryClass, seat: usize, symmetric: bool) -> PyResult<Self> {
        let bias = LeafBias { lambda, victory, seat, symmetric };
        bias.validate()?;
        Ok(bias)
    }

    pub fn validate(&self) -> PyResult<()> {
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(PyValueError::new_err(
                "specialist lambda must be finite and non-negative",
            ));
        }
        if self.seat > 1 {
            return Err(PyValueError::new_err("specialist seat must be 0 or 1"));
        }
        Ok(())
    }

    pub fn is_active(&self) -> bool {
        self.lambda > 0.0
    }

    /// The bonus in SPECIALIST terms: positive is good for the specialist.
    fn bonus(&self, outlook: &Outlook) -> f64 {
        let own = self.seat * 3 + self.victory.offset();
        let other = (1 - self.seat) * 3 + self.victory.offset();
        if self.symmetric {
            self.lambda * (outlook[own] - outlook[other])
        } else {
            self.lambda * outlook[own]
        }
    }

    /// Shape one leaf's player-0 value into the searcher's utility.
    ///
    /// A missing outlook under a live lambda is a HARD ERROR, never a silent
    /// zero bias: several `LeafOut` branches construct `None` (mock evaluators,
    /// nets without W4's head, the solver boundary), and a treatment that
    /// reaches some leaves and not others measures nothing at all.
    pub fn shape(&self, value_p0: f64, outlook: Option<Outlook>) -> PyResult<f64> {
        if !self.is_active() {
            return Ok(value_p0);
        }
        let Some(outlook) = outlook else {
            return Err(PyValueError::new_err(
                "a specialist search (lambda > 0) reached a leaf with no outlook; \
                 a biased search requires an evaluator with a W4 outlook head",
            ));
        };
        let bonus = self.bonus(&outlook);
        Ok(if self.seat == 0 {
            value_p0 + bonus
        } else {
            value_p0 - bonus
        })
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
    fn evaluate(&self, state: &GameState) -> PyResult<LeafOut> {
        Ok(MockEval::eval_state(state).into())
    }
}

/// `MockEval` plus a deterministic seven-way outlook.
///
/// The lambda-zero equivalence gate runs on `MockEval`, which supplies no
/// outlook -- correctly, since nothing biased consumes one. A lambda > 0 gate
/// needs an oracle that does, and it must be reproducible in Python to the last
/// bit, so the distribution is folded from the same fingerprint hash and
/// normalised by an explicit left fold (Python's `sum` and Rust's
/// `fold(0.0, +)` associate identically).
///
/// Terminal states keep the EXACT outlook: a finished game knows how it ended,
/// and a mock that invented one there would hide sign errors at precisely the
/// leaves where the bias is unambiguous.
pub struct MockOutlookEval;

const OUTLOOK_SALT: u64 = 0x2545_F491_4F6C_DD1D;

impl MockOutlookEval {
    pub fn outlook_of(state: &GameState) -> Outlook {
        if state.phase == Phase::Complete {
            return terminal_outlook_p0(state);
        }
        let h = fold_fingerprint(&state.fingerprint());
        let mut raw = [0.0f64; 7];
        for (k, slot) in raw.iter_mut().enumerate() {
            *slot = to_unit(mix(h ^ OUTLOOK_SALT.wrapping_mul(k as u64 + 1)));
        }
        let mass = raw.iter().fold(0.0_f64, |a, &b| a + b);
        let mut out = [0.0f64; 7];
        for k in 0..7 {
            out[k] = raw[k] / mass;
        }
        out
    }
}

impl Eval for MockOutlookEval {
    fn evaluate(&self, state: &GameState) -> PyResult<LeafOut> {
        let (value_p0, priors) = MockEval::eval_state(state);
        Ok(LeafOut {
            value_p0,
            priors,
            outlook_p0: Some(MockOutlookEval::outlook_of(state)),
        })
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
    fn evaluate(&self, state: &GameState) -> PyResult<LeafOut> {
        if state.phase == Phase::Complete {
            return Ok(LeafOut {
                value_p0: terminal_value_p0(state),
                priors: Vec::new(),
                outlook_p0: Some(terminal_outlook_p0(state)),
            });
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
            Ok(LeafOut::scalar(value_p0, priors))
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
    fn evaluate(&self, state: &GameState) -> PyResult<LeafOut> {
        let mut rows = self.evaluate_batch(&[state])?;
        Ok(rows.remove(0))
    }

    fn evaluate_batch(&self, states: &[&GameState]) -> PyResult<Vec<LeafOut>> {
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
            let raw = extract_rows(&out)?;
            if raw.len() != states.len() {
                return Err(PyValueError::new_err(format!(
                    "batch net returned {} rows for {} states",
                    raw.len(),
                    states.len()
                )));
            }
            raw.into_iter()
                .enumerate()
                .map(|(row, (value_actor, priors, raw_outlook))| {
                    if states[row].phase == Phase::Complete {
                        if !priors.is_empty() {
                            return Err(PyValueError::new_err(format!(
                                "batch net terminal row {row} returned policy priors"
                            )));
                        }
                        return Ok(LeafOut {
                            value_p0: terminal_value_p0(states[row]),
                            priors: Vec::new(),
                            outlook_p0: Some(terminal_outlook_p0(states[row])),
                        });
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
                    Ok(LeafOut {
                        value_p0,
                        priors,
                        outlook_p0: adapter_outlook(row, raw_outlook, actors[row])?,
                    })
                })
                .collect()
        })
    }
}

const FLAT_FEATURE_WIDTH: usize = crate::encoder::MAX_FEATURES;

#[derive(Clone, Debug, Default)]
pub struct BoundaryMetrics {
    pub batches: usize,
    pub rows: usize,
    pub tokens: usize,
    pub padded_tokens: usize,
    pub max_tokens: usize,
    /// Sequence-length padding, quadratic form. The network's attention cost is
    /// quadratic in sequence length while `padded_tokens` is linear, so the
    /// linear ratio understates the compute actually wasted.
    /// `1 - tokens_sq / padded_tokens_sq` is the quadratic waste.
    pub tokens_sq: usize,
    pub padded_tokens_sq: usize,
    /// **Feature-width** padding, a second and previously untracked dimension.
    /// Every token is written at `FLAT_FEATURE_WIDTH` floats, but `net.py`
    /// projects each type with `nn.Linear(FEATURE_COUNTS[type], d_model)` and
    /// reads no further. FEATURE_COUNTS is [130, 1, 26, 1, 8, 4, 1, 79, 14], so
    /// most tokens carry mostly zeros -- through the pack, the bytearray copy,
    /// H2D, and the tensor build.
    pub feature_values_used: usize,
    pub feature_values_written: usize,
    pub encode_pack_ns: u64,
    pub queue_wait_ns: u64,
    pub py_call_ns: u64,
    pub extract_ns: u64,
    // --- Worker sub-partition (CORE_UTILIZATION_PLAN.md build-order step 1) ---
    // `encode_pack_ns`, `py_call_ns` and `extract_ns` do not tile this function,
    // so `sched_wait_ns - pack - call` left a 14.76 s residual that an earlier
    // revision of the plan mislabelled "extract + validation" and sized a prize
    // from. The real cloud run says extraction cannot be it: 25.95 s of
    // `extract_ns` against 6,404 s of `encode_pack_ns`. These four close the gap.
    /// Acquiring the GIL (`Python::attach`).
    pub attach_ns: u64,
    /// Building the payload dict: eleven `PyByteArray` allocations and copies,
    /// all of it *before* `py_call_ns` starts. The leading suspect.
    pub payload_ns: u64,
    /// Validating every returned prior is finite, non-negative and positive-mass.
    pub validate_ns: u64,
    /// Taking the metrics mutex and updating it.
    pub metrics_ns: u64,
    // --- Coalescing (COALESCER_BUILD_PLAN.md §3) ----------------------------
    /// Requests the worker RECEIVED, against `batches` forwards it ISSUED. The
    /// ratio is the coalescing engagement, and it is exactly 1.00 when nothing
    /// merged -- which is what makes a silent revert to one-request-per-forward
    /// visible instead of merely slow.
    ///
    /// The scheduler-side `global_batches` cannot answer this. It is
    /// incremented inside the shard, before anything could merge, so it counts
    /// requests no matter what the worker does with them.
    pub worker_requests: usize,
    /// Time the drain loop spent WAITING for more work after the first request
    /// of a batch arrived. Zero under the `try_recv`-only default; a positive
    /// `inference_wait_ms` buys batch width with exactly this.
    pub coalesce_wait_ns: u64,
    /// Batches closed early because admitting the next request would have
    /// exceeded `max_rows`. The held request heads the following batch. A large
    /// count means the CAP is what limits width, not the arrival rate -- a
    /// different lever from the wait.
    pub coalesce_carried: usize,
}

#[derive(Default)]
struct FlatBatchBuilder {
    token_offsets: Vec<u8>,
    type_ids: Vec<u8>,
    entity_ids: Vec<u8>,
    aux_ids: Vec<u8>,
    features: Vec<u8>,
    actors: Vec<u8>,
    /// One network id per row, resolved from the searcher (see
    /// `Eval::evaluate_batch_prepared_routed`). All zeros when the caller
    /// supplied no routing, which is the single-network case.
    net_ids: Vec<u8>,
    legal_offsets: Vec<u8>,
    legal_actions: Vec<u8>,
    rows: usize,
    tokens: usize,
    max_tokens: usize,
    tokens_sq: usize,
    feature_values_used: usize,
    /// Per-row scratch, retained across batches and grown to the widest batch
    /// seen, so a steady-state run allocates nothing here either.
    row_scratch: Vec<RowPack>,
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
        self.net_ids.clear();
        self.legal_offsets.clear();
        self.legal_actions.clear();
        self.rows = 0;
        self.tokens = 0;
        self.max_tokens = 0;
        self.tokens_sq = 0;
        self.feature_values_used = 0;
    }

    /// `net_ids` may be empty, meaning "one network for every row". Anything
    /// else must be row-aligned; the caller checks that before getting here.
    fn pack_routed(
        &mut self,
        states: &[&GameState],
        actors: &[usize],
        legals: &[Vec<usize>],
        net_ids: &[u8],
    ) {
        self.clear();
        push_u32(&mut self.token_offsets, 0);
        push_u32(&mut self.legal_offsets, 0);
        let n = states.len();
        // Moved out so the ordered copy below can borrow the output buffers
        // mutably while reading the scratch.
        let mut scratch = std::mem::take(&mut self.row_scratch);
        if scratch.len() < n {
            scratch.resize_with(n, RowPack::default);
        }
        // Rows are independent: `encode` reads only its own `GameState`.
        //
        // Measured: rayon's dispatch costs ~13% of pack time, so a one-thread
        // pool is 0.971x the serial loop -- a silent regression for anyone who
        // sets RAYON_NUM_THREADS=1. Take the serial path when there is nothing
        // to parallelise.
        let encode_rows = |scratch: &mut [RowPack]| {
            scratch
                .par_iter_mut()
                .zip(states.par_iter())
                .for_each(|(row_pack, state)| row_pack.fill(state));
        };
        if pack_threads() <= 1 {
            for (row_pack, state) in scratch[..n].iter_mut().zip(states.iter()) {
                row_pack.fill(state);
            }
        } else if let Some(pool) = PACK_POOL.get() {
            pool.install(|| encode_rows(&mut scratch[..n]));
        } else {
            encode_rows(&mut scratch[..n]);
        }
        for (row, ((row_pack, &actor), legal)) in
            scratch[..n].iter().zip(actors).zip(legals).enumerate()
        {
            self.actors.push(actor as u8);
            self.net_ids
                .push(net_ids.get(row).copied().unwrap_or(0));
            self.max_tokens = self.max_tokens.max(row_pack.tokens);
            self.type_ids.extend_from_slice(&row_pack.type_ids);
            self.entity_ids.extend_from_slice(&row_pack.entity_ids);
            self.aux_ids.extend_from_slice(&row_pack.aux_ids);
            self.features.extend_from_slice(&row_pack.features);
            self.tokens += row_pack.tokens;
            self.tokens_sq += row_pack.tokens * row_pack.tokens;
            self.feature_values_used += row_pack.feature_values_used;
            push_u32(&mut self.token_offsets, self.tokens);
            for &action in legal {
                self.legal_actions
                    .extend_from_slice(&(action as u16).to_le_bytes());
            }
            let legal_total = self.legal_actions.len() / 2;
            push_u32(&mut self.legal_offsets, legal_total);
        }
        self.row_scratch = scratch;
        self.rows = n;
    }
}


/// One row's packed bytes, encoded independently so rows can run in parallel.
///
/// Rows are concatenated afterwards **in row order**, so the flat buffers are
/// byte-identical to the serial build regardless of thread count or completion
/// order. Token counts are only known after encoding, which is why this is
/// row-local buffers plus an ordered copy rather than rows writing into
/// pre-sized global slices.
#[derive(Default)]
struct RowPack {
    token_buf: crate::encoder::TokenBuf,
    type_ids: Vec<u8>,
    entity_ids: Vec<u8>,
    aux_ids: Vec<u8>,
    features: Vec<u8>,
    tokens: usize,
    feature_values_used: usize,
}

impl RowPack {
    fn fill(&mut self, state: &GameState) {
        let Self {
            token_buf,
            type_ids,
            entity_ids,
            aux_ids,
            features,
            tokens,
            feature_values_used,
        } = self;
        type_ids.clear();
        entity_ids.clear();
        aux_ids.clear();
        features.clear();
        crate::encoder::encode_into(state, token_buf);
        let encoded = token_buf.tokens();
        *tokens = encoded.len();
        *feature_values_used = 0;
        for token in encoded {
            *feature_values_used += crate::encoder::FEATURE_COUNTS[token.type_id];
            type_ids.push(token.type_id as u8);
            entity_ids.extend_from_slice(&(token.entity_id as i16).to_le_bytes());
            aux_ids.extend_from_slice(&((token.aux_id + 1) as i16).to_le_bytes());
            for index in 0..FLAT_FEATURE_WIDTH {
                // The flat Python/Torch boundary is explicitly f32; keeping this
                // cast here permits a zero-copy tensor view of the bytes.
                let value = token.features.get(index).copied().unwrap_or(0.0) as f32;
                features.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
}

/// The pack thread pool, sized explicitly rather than inherited.
///
/// Packing used rayon's **global** pool, which defaults to every CPU the process
/// can see. On a rented slice that is the *host's* count, not the quota sold --
/// so a 192-core host selling 12 cores would spawn 192 packing threads and
/// oversubscribe badly, turning a measured win into a loss. It also meant
/// `f4_pack_sweep`'s recommendation controlled nothing: the sweep installs a
/// scoped pool, production read the global one, and the two were connected only
/// by the laptop coincidentally exposing 16 CPUs.
///
/// Python owns the detection (cgroup quota, cpuset and affinity -- see
/// `cloud_preflight.effective_cpu_count`) and calls `set_pack_threads`.
/// Unset, packing falls back to the global pool exactly as before.
static PACK_POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();

/// Size the pack pool. First call wins; returns the pool's actual thread count
/// so callers can record what they *got* rather than what they asked for.
pub fn set_pack_threads(threads: usize) -> PyResult<usize> {
    let threads = threads.max(1);
    if PACK_POOL.get().is_none() {
        let built = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("swd-pack-{i}"))
            .build()
            .map_err(|e| PyValueError::new_err(format!("pack pool: {e}")))?;
        let _ = PACK_POOL.set(built);
    }
    Ok(pack_threads())
}

/// Actual pack parallelism: the dedicated pool's size, or the global pool's.
pub fn pack_threads() -> usize {
    PACK_POOL
        .get()
        .map(|pool| pool.current_num_threads())
        .unwrap_or_else(rayon::current_num_threads)
}

/// Run CPU packing/encoding work on the same explicitly sized pool used by the
/// search boundary. Derivation is a separate phase, so sharing the pool avoids
/// another host-sized Rayon pool and cannot contend with live inference.
pub(crate) fn with_pack_pool<F, R>(operation: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    if let Some(pool) = PACK_POOL.get() {
        pool.install(operation)
    } else {
        operation()
    }
}

/// Packing-only benchmark for CORE_UTILIZATION_PLAN.md step 2b.
///
/// Times `pack_routed` with no GPU, no search and no Python round trip, so a
/// threads x rows surface costs seconds instead of minutes of generation. The
/// thread count is a **scoped** pool rather than the global one, so a single
/// process can sweep every rung -- `RAYON_NUM_THREADS` is read once at first
/// use and could not.
///
/// Returns seconds for `iterations` packs of the supplied rows, excluding a
/// warmup pack that fills the retained buffers.
pub fn bench_pack_routed(states: &[&GameState], iterations: usize, threads: usize) -> PyResult<f64> {
    let actors: Vec<usize> = states.iter().map(|s| crate::tree::state_actor(s)).collect();
    let legals: Vec<Vec<usize>> = states
        .iter()
        .map(|s| crate::codec::legal_action_indices(s))
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| PyValueError::new_err(format!("rayon pool: {e}")))?;
    let mut builder = FlatBatchBuilder::default();
    pool.install(|| {
        builder.pack_routed(states, &actors, &legals, &[]);
        let started = Instant::now();
        for _ in 0..iterations {
            builder.pack_routed(states, &actors, &legals, &[]);
        }
        Ok(started.elapsed().as_secs_f64())
    })
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
    fn evaluate(&self, state: &GameState) -> PyResult<LeafOut> {
        let mut rows = self.evaluate_batch(&[state])?;
        Ok(rows.remove(0))
    }

    fn evaluate_batch(&self, states: &[&GameState]) -> PyResult<Vec<LeafOut>> {
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
    ) -> PyResult<Vec<LeafOut>> {
        self.evaluate_batch_prepared_routed(states, actors, legals, &[])
    }

    fn evaluate_batch_prepared_routed(
        &self,
        states: &[&GameState],
        actors: &[usize],
        legals: &[Vec<usize>],
        net_ids: &[u8],
    ) -> PyResult<Vec<LeafOut>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        if states.len() != actors.len() || states.len() != legals.len() {
            return Err(PyValueError::new_err(
                "flat evaluator metadata is not row-aligned",
            ));
        }
        if !net_ids.is_empty() && net_ids.len() != states.len() {
            return Err(PyValueError::new_err(
                "flat evaluator net ids are not row-aligned",
            ));
        }
        let pack_start = Instant::now();
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| PyValueError::new_err("flat batch scratch lock poisoned"))?;
        scratch.pack_routed(states, actors, legals, net_ids);
        let legal_counts: Vec<_> = legals.iter().map(Vec::len).collect();
        let pack_ns = pack_start.elapsed().as_nanos() as u64;
        let rows = scratch.rows;
        let tokens = scratch.tokens;
        let max_tokens = scratch.max_tokens;
        let tokens_sq = scratch.tokens_sq;
        let feature_values_used = scratch.feature_values_used;
        let attach_start = Instant::now();
        let (raw, call_ns, extract_ns, attach_ns, payload_ns) = Python::attach(|py| {
            let attach_ns = attach_start.elapsed().as_nanos() as u64;
            let payload_start = Instant::now();
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
                checked_bytearray(py, &scratch.token_offsets)?,
            )?;
            payload.set_item("type_ids", checked_bytearray(py, &scratch.type_ids)?)?;
            payload.set_item("entity_ids", checked_bytearray(py, &scratch.entity_ids)?)?;
            payload.set_item("aux_ids", checked_bytearray(py, &scratch.aux_ids)?)?;
            payload.set_item("features", checked_bytearray(py, &scratch.features)?)?;
            payload.set_item("actors", checked_bytearray(py, &scratch.actors)?)?;
            // Always present, so the adapter never has to branch on its
            // absence; all zeros in the single-network case.
            payload.set_item("net_ids", checked_bytearray(py, &scratch.net_ids)?)?;
            payload.set_item(
                "legal_offsets",
                checked_bytearray(py, &scratch.legal_offsets)?,
            )?;
            payload.set_item(
                "legal_actions",
                checked_bytearray(py, &scratch.legal_actions)?,
            )?;
            let payload_ns = payload_start.elapsed().as_nanos() as u64;
            let call_start = Instant::now();
            let out = self.adapter.bind(py).call1((payload,))?;
            let call_ns = call_start.elapsed().as_nanos() as u64;
            let extract_start = Instant::now();
            let raw = extract_rows(&out)?;
            let extract_ns = extract_start.elapsed().as_nanos() as u64;
            Ok::<_, PyErr>((raw, call_ns, extract_ns, attach_ns, payload_ns))
        })?;
        drop(scratch);
        if raw.len() != states.len() {
            return Err(PyValueError::new_err(format!(
                "flat net returned {} rows for {} states",
                raw.len(),
                states.len()
            )));
        }
        let validate_start = Instant::now();
        let mut validated = Vec::with_capacity(raw.len());
        for (row, (value_actor, priors, raw_outlook)) in raw.into_iter().enumerate() {
            if states[row].phase == Phase::Complete {
                if !priors.is_empty() {
                    return Err(PyValueError::new_err(format!(
                        "flat net terminal row {row} returned policy priors"
                    )));
                }
                validated.push(LeafOut {
                    value_p0: terminal_value_p0(states[row]),
                    priors: Vec::new(),
                    outlook_p0: Some(terminal_outlook_p0(states[row])),
                });
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
            validated.push(LeafOut {
                value_p0: if actors[row] == 0 {
                    value_actor
                } else {
                    -value_actor
                },
                priors,
                outlook_p0: adapter_outlook(row, raw_outlook, actors[row])?,
            });
        }
        let validate_ns = validate_start.elapsed().as_nanos() as u64;
        let metrics_start = Instant::now();
        let mut metrics = self
            .metrics
            .lock()
            .map_err(|_| PyValueError::new_err("boundary metrics lock poisoned"))?;
        metrics.batches += 1;
        metrics.rows += rows;
        metrics.tokens += tokens;
        metrics.padded_tokens += rows * max_tokens;
        metrics.tokens_sq += tokens_sq;
        metrics.padded_tokens_sq += rows * max_tokens * max_tokens;
        metrics.feature_values_used += feature_values_used;
        metrics.feature_values_written += tokens * FLAT_FEATURE_WIDTH;
        metrics.max_tokens = metrics.max_tokens.max(max_tokens);
        metrics.encode_pack_ns += pack_ns;
        metrics.py_call_ns += call_ns;
        metrics.extract_ns += extract_ns;
        metrics.attach_ns += attach_ns;
        metrics.payload_ns += payload_ns;
        metrics.validate_ns += validate_ns;
        // Excludes its own store, which is a handful of adds under a lock this
        // thread is the only writer of.
        metrics.metrics_ns += metrics_start.elapsed().as_nanos() as u64;
        Ok(validated)
    }
}

type WorkerResponse = PyResult<Vec<LeafOut>>;

/// Several `WorkerRequest`s concatenated into ONE forward, with enough
/// bookkeeping to take the single answer apart again.
///
/// The evaluator boundary is what binds generation: on the iter85 run
/// `py_call_ns` was 91.8% of scheduler wall at **46.9 rows against a 2,048-row
/// cap**, because every shard's request crossed it alone. Merging is worth
/// having precisely because the cost is nearly all fixed per call.
#[derive(Default)]
struct CoalescedBatch {
    states: Vec<GameState>,
    actors: Vec<usize>,
    legals: Vec<Vec<usize>>,
    /// Empty only while EVERY member so far was unrouted, which the packer
    /// reads as "all rows on network 0". As soon as one member routes, the
    /// unrouted members are expanded to explicit zeros: concatenating a short
    /// `net_ids` onto a long batch would silently shift rows onto the wrong
    /// network, and a league game evaluated by its opponent's net is a bug that
    /// produces plausible numbers.
    net_ids: Vec<u8>,
    routed: bool,
    /// Reply channel and row count per member, IN RECEIPT ORDER -- so a shard
    /// running `max_inflight_batches > 1` keeps its own sequencing.
    members: Vec<(mpsc::Sender<WorkerResponse>, usize)>,
}

impl CoalescedBatch {
    fn rows(&self) -> usize {
        self.states.len()
    }

    fn push(&mut self, request: WorkerRequest) {
        let rows = request.states.len();
        if request.net_ids.is_empty() {
            // Unrouted means network 0. Only materialise that once some other
            // member has forced the batch to carry ids at all.
            if self.routed {
                self.net_ids.resize(self.states.len() + rows, 0);
            }
        } else {
            if !self.routed {
                // Every row admitted so far was unrouted, i.e. network 0.
                self.net_ids = vec![0u8; self.states.len()];
                self.routed = true;
            }
            self.net_ids.extend_from_slice(&request.net_ids);
        }
        self.states.extend(request.states);
        self.actors.extend(request.actors);
        self.legals.extend(request.legals);
        self.members.push((request.reply, rows));
    }

    /// Slice the forward's rows back to their submitters by running offset.
    fn scatter(self, rows: Vec<LeafOut>) {
        let mut rows = rows.into_iter();
        for (reply, count) in self.members {
            let slice: Vec<LeafOut> = rows.by_ref().take(count).collect();
            // A dropped receiver is an ABANDONED TICKET -- its owner timed out
            // and walked away. Before coalescing, the worker broke its loop on
            // this. It must not now, because the other members of this batch
            // are still waiting on a forward that already succeeded.
            let _ = reply.send(Ok(slice));
        }
    }

    /// Every member of a merged batch gets the failure. One request's error is
    /// all of their errors: they shared the forward that raised it.
    fn fan_error(&self, error: &PyErr) {
        Python::attach(|py| {
            for (reply, _) in &self.members {
                let _ = reply.send(Err(error.clone_ref(py)));
            }
        });
    }
}

struct WorkerRequest {
    states: Vec<GameState>,
    actors: Vec<usize>,
    legals: Vec<Vec<usize>>,
    /// W1.3 routing, carried across the worker boundary. Empty means one
    /// network. Dropping it here would silently un-route league games whenever
    /// `max_inflight_batches > 1`, which is the production configuration.
    net_ids: Vec<u8>,
    reply: mpsc::Sender<WorkerResponse>,
    enqueued: Instant,
}

// --- the inference thread pool ----------------------------------------------
//
// Every worker below used to be `thread::spawn`ed per call and joined at the
// end of it. That is correct and it leaks: Torch attaches per-thread state to
// any thread that runs a forward and never gives it back when the thread exits.
// Measured on the arena's boundary at ~16 MB and ~7 OS threads per call, which
// a per-ply caller turns into +3.2 GB per match and, once, an out-of-memory
// shutdown mid-run.
//
// So the threads are pooled instead. What has to stay bounded is the number of
// DISTINCT threads that ever call the adapter, not the number of live ones, and
// reuse bounds it by peak concurrency rather than by total calls.
//
// Concurrency is preserved deliberately: a single shared thread would deadlock
// the moment two searches ran at once (the advisor host runs several), because
// a worker loop only ends when its `EvalWorker` is dropped, and a second job
// queued behind it would never start. A thread is only offered back to the pool
// once its job has actually returned, so a worker abandoned on timeout keeps
// its thread out of circulation until the Python call it is stuck in finishes,
// rather than handing a busy thread to the next caller.

type PoolJob = Box<dyn FnOnce() + Send + 'static>;

/// Idle threads, each addressed by the sender to its own job channel.
static IDLE_WORKERS: Mutex<Vec<mpsc::Sender<PoolJob>>> = Mutex::new(Vec::new());

fn release_worker(slot: mpsc::Sender<PoolJob>) {
    if let Ok(mut idle) = IDLE_WORKERS.lock() {
        idle.push(slot);
    }
    // A poisoned lock drops the sender, retiring that thread. The pool is an
    // optimisation; losing one thread from it is not worth failing a search.
}

/// Handle for one pooled worker loop: `join` waits for the loop, `drop` detaches.
pub struct WorkerHandle {
    done: mpsc::Receiver<bool>,
}

impl WorkerHandle {
    /// Wait for the worker loop to finish. `Err(())` means it panicked, which
    /// keeps the same shape the callers' `JoinHandle::join` had.
    pub fn join(self) -> Result<(), ()> {
        match self.done.recv() {
            // The thread died without reporting, which only happens if the
            // pooled thread itself was torn down mid-job.
            Ok(true) => Ok(()),
            Ok(false) | Err(_) => Err(()),
        }
    }
}

/// Run `body` on a pooled thread, returning a handle that waits for it.
fn run_pooled<F>(body: F) -> WorkerHandle
where
    F: FnOnce() + Send + 'static,
{
    let (done_tx, done_rx) = mpsc::channel::<bool>();
    let slot = IDLE_WORKERS.lock().ok().and_then(|mut idle| idle.pop());
    let slot = match slot {
        Some(slot) => slot,
        None => spawn_pool_thread(),
    };
    let mine = slot.clone();
    let job: PoolJob = Box::new(move || {
        // The worker loops below call into Python; a panic there must reach the
        // caller as a failed join rather than unwinding the pooled thread.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
        let _ = done_tx.send(outcome.is_ok());
        // Only now is this thread genuinely free: a job abandoned by a
        // `drop(handle)` on timeout is still inside its adapter call above.
        release_worker(mine);
    });
    if let Err(mpsc::SendError(job)) = slot.send(job) {
        // The pooled thread died between being parked and being handed work.
        // The job comes back with the error, so it runs on a fresh thread
        // rather than being silently dropped -- a caller waiting on the handle
        // of a job that never ran would block forever.
        let replacement = spawn_pool_thread();
        if replacement.send(job).is_err() {
            return WorkerHandle { done: done_rx };
        }
    }
    WorkerHandle { done: done_rx }
}

fn spawn_pool_thread() -> mpsc::Sender<PoolJob> {
    let (tx, rx) = mpsc::channel::<PoolJob>();
    thread::spawn(move || {
        // Parks between jobs and never exits, which is the point: the Torch
        // state attached to this thread is paid for once and reused.
        while let Ok(job) = rx.recv() {
            job();
        }
    });
    tx
}

/// Scheduler-side handle for the dedicated F4.4 Python inference thread. Rust
/// scheduling runs with the caller's GIL detached; only the worker attaches to
/// Python and invokes the batch adapter.
pub struct EvalWorker {
    sender: mpsc::Sender<WorkerRequest>,
    timeout: Option<Duration>,
    timed_out: Arc<AtomicBool>,
    terminal_error: Arc<Mutex<Option<PyErr>>>,
    max_rows: usize,
}

pub struct EvalTicket {
    receiver: mpsc::Receiver<WorkerResponse>,
    timeout: Option<Duration>,
    timed_out: Arc<AtomicBool>,
    terminal_error: Arc<Mutex<Option<PyErr>>>,
}

fn terminal_worker_error(terminal_error: &Mutex<Option<PyErr>>, fallback: &str) -> PyErr {
    match terminal_error.lock() {
        Ok(error) => match error.as_ref() {
            Some(error) => Python::attach(|py| error.clone_ref(py)),
            None => PyValueError::new_err(fallback.to_owned()),
        },
        Err(_) => PyValueError::new_err(format!(
            "{fallback}; terminal inference error lock poisoned"
        )),
    }
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
                Err(mpsc::RecvTimeoutError::Disconnected) => Err(terminal_worker_error(
                    &self.terminal_error,
                    "global inference worker dropped its response",
                )),
            },
            None => self.receiver.recv().map_err(|_| {
                terminal_worker_error(
                    &self.terminal_error,
                    "global inference worker dropped its response",
                )
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
        self.submit_prepared_routed(states, actors, legals, Vec::new())
    }

    pub fn submit_prepared_routed(
        &self,
        states: Vec<GameState>,
        actors: Vec<usize>,
        legals: Vec<Vec<usize>>,
        net_ids: Vec<u8>,
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
        if !net_ids.is_empty() && net_ids.len() != states.len() {
            return Err(PyValueError::new_err(
                "inference request net ids are not row-aligned",
            ));
        }
        let (reply_tx, reply_rx) = mpsc::channel();
        if self
            .sender
            .send(WorkerRequest {
                states,
                actors,
                legals,
                net_ids,
                reply: reply_tx,
                enqueued: Instant::now(),
            })
            .is_err()
        {
            return Err(terminal_worker_error(
                &self.terminal_error,
                "global inference worker is not running",
            ));
        }
        Ok(EvalTicket {
            receiver: reply_rx,
            timeout: self.timeout,
            timed_out: Arc::clone(&self.timed_out),
            terminal_error: Arc::clone(&self.terminal_error),
        })
    }
}

impl Eval for EvalWorker {
    fn evaluate_batch_prepared_routed(
        &self,
        states: &[&GameState],
        actors: &[usize],
        legals: &[Vec<usize>],
        net_ids: &[u8],
    ) -> PyResult<Vec<LeafOut>> {
        let ticket = self.submit_prepared_routed(
            states.iter().map(|state| (*state).clone()).collect(),
            actors.to_vec(),
            legals.to_vec(),
            net_ids.to_vec(),
        )?;
        ticket.wait()
    }

    fn evaluate(&self, state: &GameState) -> PyResult<LeafOut> {
        let mut rows = self.evaluate_batch(&[state])?;
        Ok(rows.remove(0))
    }

    fn evaluate_batch(&self, states: &[&GameState]) -> PyResult<Vec<LeafOut>> {
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
    ) -> PyResult<Vec<LeafOut>> {
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
) -> PyResult<(EvalWorker, Arc<AtomicBool>, WorkerHandle)> {
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
    let terminal_error = Arc::new(Mutex::new(None));
    let worker_terminal_error = Arc::clone(&terminal_error);
    let handle = run_pooled(move || {
        let evaluator = PyBatchEval::new(adapter);
        while let Ok(request) = request_rx.recv() {
            let refs: Vec<&GameState> = request.states.iter().collect();
            let result = evaluator.evaluate_batch_prepared_routed(
                &refs,
                &request.actors,
                &request.legals,
                &request.net_ids,
            );
            match result {
                Ok(rows) => {
                    if request.reply.send(Ok(rows)).is_err() {
                        break;
                    }
                }
                Err(error) => {
                    let reply_error = Python::attach(|py| error.clone_ref(py));
                    if let Ok(mut terminal) = worker_terminal_error.lock() {
                        *terminal = Some(error);
                    }
                    let _ = request.reply.send(Err(reply_error));
                    break;
                }
            }
        }
    });
    Ok((
        EvalWorker {
            sender: request_tx,
            timeout,
            timed_out: Arc::clone(&timed_out),
            terminal_error,
            max_rows,
        },
        timed_out,
        handle,
    ))
}

/// The flat inference worker, with cross-shard coalescing.
///
/// `wait_ms` is the drain policy, and 0 is the intended default rather than a
/// disabled feature: the loop still drains everything ALREADY queued into one
/// forward, it just never blocks to grow a batch further. That is the whole
/// mechanism at a queue measured ~3 deep. A positive wait trades latency for
/// width and is an axis to sweep, not a number to assume.
///
/// `coalesce = false` restores the pre-coalescer behaviour: one request per
/// forward. It exists so a rented box can A/B this change against itself under
/// identical conditions -- same machine, same checkpoint, same seeds -- rather
/// than against cloud2's numbers at a different geometry, which cannot
/// attribute a throughput change to this implementation.
///
/// It is one branch inside the same loop, not a second path: the batch, the
/// scatter and the error fan-out are the same code, just never given a second
/// member.
pub fn spawn_py_flat_worker(
    adapter: Py<PyAny>,
    timeout_ms: f64,
    max_rows: usize,
    wait_ms: f64,
    coalesce: bool,
) -> PyResult<(
    EvalWorker,
    Arc<AtomicBool>,
    Arc<Mutex<BoundaryMetrics>>,
    WorkerHandle,
)> {
    if !timeout_ms.is_finite() || timeout_ms < 0.0 {
        return Err(PyValueError::new_err(
            "inference_timeout_ms must be finite and non-negative",
        ));
    }
    if !wait_ms.is_finite() || wait_ms < 0.0 {
        return Err(PyValueError::new_err(
            "inference_wait_ms must be finite and non-negative",
        ));
    }
    if max_rows == 0 {
        return Err(PyValueError::new_err(
            "inference worker max_rows must be positive",
        ));
    }
    if !coalesce && wait_ms > 0.0 {
        // A wait with nothing to wait FOR is pure added latency on every batch.
        return Err(PyValueError::new_err(
            "inference_wait_ms > 0 with coalescing disabled would add latency to              every forward and widen nothing",
        ));
    }
    if timeout_ms > 0.0 && wait_ms >= timeout_ms {
        // The wait is spent INSIDE each ticket's deadline, so a wait at or
        // above it guarantees every ticket expires before its forward is even
        // issued. The run then fails slowly, with a timeout message that points
        // at the network.
        return Err(PyValueError::new_err(format!(
            "inference_wait_ms={wait_ms} must be below \
             inference_timeout_ms={timeout_ms}; the coalescing wait is spent \
             inside each ticket's deadline"
        )));
    }
    let timeout = if timeout_ms == 0.0 {
        None
    } else {
        Some(Duration::from_secs_f64(timeout_ms / 1000.0))
    };
    let wait = if wait_ms == 0.0 {
        None
    } else {
        Some(Duration::from_secs_f64(wait_ms / 1000.0))
    };
    let (request_tx, request_rx) = mpsc::channel::<WorkerRequest>();
    let timed_out = Arc::new(AtomicBool::new(false));
    let terminal_error = Arc::new(Mutex::new(None));
    let worker_terminal_error = Arc::clone(&terminal_error);
    let metrics = Arc::new(Mutex::new(BoundaryMetrics::default()));
    let worker_metrics = Arc::clone(&metrics);
    let handle = run_pooled(move || {
        let evaluator = PyFlatBatchEval::new(adapter, Arc::clone(&worker_metrics));
        // A request that would overflow `max_rows` is HELD, never dropped: an
        // `mpsc::Receiver` cannot un-receive, and the submitter is blocked on a
        // ticket nothing else will complete. It heads the next batch.
        let mut carried: Option<WorkerRequest> = None;
        loop {
            let first = match carried.take() {
                Some(request) => request,
                None => match request_rx.recv() {
                    Ok(request) => request,
                    // Every scheduler has gone. `recv` yields what is still
                    // buffered before reporting this, so nothing is stranded.
                    Err(_) => break,
                },
            };
            let mut queue_wait_ns = first.enqueued.elapsed().as_nanos() as u64;
            let mut waited_ns = 0u64;
            let mut batch = CoalescedBatch::default();
            batch.push(first);
            let deadline = wait.map(|wait| Instant::now() + wait);
            loop {
                // `!coalesce` is the A/B arm: one request per forward, which is
                // exactly what this worker did before the drain loop existed.
                if !coalesce || batch.rows() >= max_rows {
                    break;
                }
                let next = match deadline {
                    Some(deadline) => match deadline.checked_duration_since(Instant::now()) {
                        Some(left) => {
                            let started = Instant::now();
                            let next = request_rx.recv_timeout(left).ok();
                            waited_ns += started.elapsed().as_nanos() as u64;
                            next
                        }
                        // Deadline passed; still take whatever is already
                        // sitting there rather than issuing a narrow batch
                        // beside a full queue.
                        None => request_rx.try_recv().ok(),
                    },
                    None => request_rx.try_recv().ok(),
                };
                let Some(next) = next else { break };
                if batch.rows() + next.states.len() > max_rows {
                    carried = Some(next);
                    break;
                }
                queue_wait_ns += next.enqueued.elapsed().as_nanos() as u64;
                batch.push(next);
            }
            if let Ok(mut counters) = worker_metrics.lock() {
                counters.queue_wait_ns += queue_wait_ns;
                counters.worker_requests += batch.members.len();
                counters.coalesce_wait_ns += waited_ns;
                if carried.is_some() {
                    counters.coalesce_carried += 1;
                }
            }
            // Scoped so the borrow of `batch.states` ends before `scatter`
            // consumes the batch.
            let result = {
                let refs: Vec<&GameState> = batch.states.iter().collect();
                evaluator.evaluate_batch_prepared_routed(
                    &refs,
                    &batch.actors,
                    &batch.legals,
                    &batch.net_ids,
                )
            };
            match result {
                Ok(rows) => batch.scatter(rows),
                Err(error) => {
                    batch.fan_error(&error);
                    // The held-over request shares this worker's fate and is
                    // waiting on a ticket nothing else will ever answer.
                    if let Some(request) = carried.take() {
                        let _ = request
                            .reply
                            .send(Err(Python::attach(|py| error.clone_ref(py))));
                    }
                    if let Ok(mut terminal) = worker_terminal_error.lock() {
                        *terminal = Some(error);
                    }
                    break;
                }
            }
        }
    });
    Ok((
        EvalWorker {
            sender: request_tx,
            timeout,
            timed_out: Arc::clone(&timed_out),
            terminal_error,
            max_rows,
        },
        timed_out,
        metrics,
        handle,
    ))
}
