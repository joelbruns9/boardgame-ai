//! Can't Stop in Rust: portable RNG (M0), engine (M1), turn solver (M2).
//!
//! Boundary, per VARIANT_SOLVER_PLAN.md Phase 3: Rust owns the engine, turn
//! enumeration and backward induction. Python keeps the training loop, the
//! replay buffer and torch. Only the per-turn leaf batch crosses back.

pub mod rng;

use pyo3::prelude::*;

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

#[pymodule]
fn cantstop_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRng>()?;
    Ok(())
}
