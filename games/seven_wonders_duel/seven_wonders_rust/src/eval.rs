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
use pyo3::prelude::*;

/// `(value_p0, priors)` where `priors` is aligned to `legal_action_indices`.
/// Terminal states return the game value and empty priors.
#[allow(dead_code)] // consumed by the F3.2 closed tree
pub trait Eval {
    fn evaluate(&self, state: &GameState) -> (f64, Vec<f64>);
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
    fn evaluate(&self, state: &GameState) -> (f64, Vec<f64>) {
        MockEval::eval_state(state)
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
    fn evaluate(&self, state: &GameState) -> (f64, Vec<f64>) {
        if state.phase == Phase::Complete {
            return (terminal_value_p0(state), Vec::new());
        }
        let actor = crate::tree::state_actor(state);
        let tokens: Vec<(usize, i32, i32, Vec<f64>)> = crate::encoder::encode(state)
            .into_iter()
            .map(|t| (t.type_id, t.entity_id, t.aux_id, t.features))
            .collect();
        let legal = legal_action_indices(state);
        Python::attach(|py| {
            let out = self
                .adapter
                .bind(py)
                .call1((tokens, actor, legal))
                .expect("net adapter call failed");
            let (value_actor, priors): (f64, Vec<f64>) =
                out.extract().expect("net adapter returned an unexpected shape");
            let value_p0 = if actor == 0 { value_actor } else { -value_actor };
            (value_p0, priors)
        })
    }
}
