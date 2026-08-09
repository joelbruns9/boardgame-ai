//! Portable SplitMix64 — a bit-for-bit mirror of `portable_rng.py::PortableRng`
//! (PHASE_F.md F3.0/F3.1), so the Rust searcher reproduces the Python reference
//! stream. Same constants as `kingdomino_rust::search::splitmix64`.

pub struct Rng {
    state: u64,
}

#[allow(dead_code)] // gumbel is consumed by the F3.3 Gumbel root
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) from the top 53 bits (exactly matches Python's
    /// `(next_u64() >> 11) / 2**53`).
    pub fn next_float(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }

    /// Gumbel(0,1) key via `-log(-log(1 - U))` with the same clamps as Python.
    pub fn gumbel(&mut self) -> f64 {
        let gamma = -((1.0_f64 - self.next_float()).max(1e-12)).ln();
        -(gamma.max(1e-12)).ln()
    }

    /// Integer in [0, n) by modulo (matches Python `next_u64() % n`).
    pub fn randrange(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// In-place Fisher–Yates (Durstenfeld), high index to low.
    pub fn shuffle<T>(&mut self, seq: &mut [T]) {
        for i in (1..seq.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            seq.swap(i, j);
        }
    }

    /// Standard normal by the Marsaglia POLAR method, mirroring
    /// `PortableRng.normal`.
    ///
    /// Polar rather than Box-Muller because Box-Muller needs `cos`, which has no
    /// correct-rounding guarantee and is where a cross-runtime last-bit
    /// disagreement would hide. This path uses only multiplication, `ln` (already
    /// proven across the boundary by `gumbel_golden_matches_python`) and `sqrt`
    /// (IEEE-754 requires correct rounding). The second variate is discarded
    /// rather than cached, so the state stays a single u64.
    pub fn normal(&mut self) -> f64 {
        loop {
            let u = 2.0 * self.next_float() - 1.0;
            let v = 2.0 * self.next_float() - 1.0;
            let s = u * u + v * v;
            if s >= 1.0 || s == 0.0 {
                continue;
            }
            return u * (-2.0 * s.ln() / s).sqrt();
        }
    }

    /// Gamma(alpha, 1) by Marsaglia-Tsang, mirroring `PortableRng.gamma`.
    ///
    /// Every arithmetic expression is written in the same association order as
    /// the Python so the two round identically. `alpha >= 1` is the golden-tested
    /// path and the only one production uses; below 1 the boost introduces `powf`,
    /// which carries no correct-rounding guarantee.
    pub fn gamma(&mut self, alpha: f64) -> f64 {
        assert!(alpha.is_finite() && alpha > 0.0, "gamma requires a finite alpha > 0");
        if alpha < 1.0 {
            let boosted = self.gamma(alpha + 1.0);
            return boosted * self.next_float().max(1e-12).powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.next_float();
            // Squeeze: pure arithmetic, short-circuits most draws.
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return d * v;
            }
            if u.max(1e-12).ln() < 0.5 * x * x + d * (1.0 - v + v.max(1e-12).ln()) {
                return d * v;
            }
        }
    }

    /// Symmetric Dirichlet over `n` categories, mirroring `PortableRng.dirichlet`.
    /// Falls back to uniform if every draw underflows, so the caller's convex
    /// blend stays well-formed.
    pub fn dirichlet(&mut self, alpha: f64, n: usize) -> Vec<f64> {
        assert!(n > 0, "dirichlet requires n > 0");
        let draws: Vec<f64> = (0..n).map(|_| self.gamma(alpha)).collect();
        let total: f64 = draws.iter().sum();
        if total <= 0.0 {
            return vec![1.0 / n as f64; n];
        }
        draws.into_iter().map(|draw| draw / total).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn splitmix64_golden_matches_python() {
        // Canonical SplitMix64(0) first output; then the pinned Python stream.
        let mut r = Rng::new(0);
        assert_eq!(r.next_u64(), 0xE220A8397B1DCDAF);
        let mut r = Rng::new(0);
        let seq: Vec<u64> = (0..5).map(|_| r.next_u64()).collect();
        assert_eq!(
            seq,
            [
                16294208416658607535,
                7960286522194355700,
                487617019471545679,
                17909611376780542444,
                1961750202426094747,
            ]
        );
        let mut r = Rng::new(0);
        assert_eq!(r.next_float(), 0.8833108082136426);
        let mut r = Rng::new(42);
        let mut seq: Vec<u32> = (0..8).collect();
        r.shuffle(&mut seq);
        assert_eq!(seq, [3, 1, 6, 2, 4, 0, 7, 5]);
    }

    #[test]
    fn normal_golden_matches_python() {
        // Pinned `PortableRng(7).normal()` stream. Equality, not approx: the
        // polar method was chosen over Box-Muller precisely so this can be exact.
        let mut r = Rng::new(7);
        let seq: Vec<f64> = (0..3).map(|_| r.normal()).collect();
        assert_eq!(
            seq,
            [
                -0.04174152338145233,
                0.8764814690994567,
                -0.3059911682027957
            ]
        );
    }

    #[test]
    fn gamma_golden_matches_python() {
        // alpha >= 1 is the production path (7WD's branching puts alpha ~1.8).
        let mut r = Rng::new(7);
        let seq: Vec<f64> = (0..3).map(|_| r.gamma(1.8)).collect();
        assert_eq!(
            seq,
            [1.4166937321647761, 6.169133592343606, 1.0571680637635033]
        );
        // The sub-1 boost path uses powf and is not guaranteed correctly rounded;
        // pinned anyway so a divergence is caught rather than discovered later.
        let mut r = Rng::new(11);
        let seq: Vec<f64> = (0..2).map(|_| r.gamma(0.3)).collect();
        assert_eq!(seq, [0.03449253441949256, 0.0832407704142125]);
    }

    #[test]
    fn dirichlet_golden_matches_python() {
        // The vector self-play actually consumes as root noise.
        let mut r = Rng::new(42);
        assert_eq!(
            r.dirichlet(1.8, 5),
            [
                0.18213062727984705,
                0.11420377943148874,
                0.07221738027724044,
                0.1832397608630785,
                0.44820845214834526
            ]
        );
    }

    #[test]
    fn gumbel_golden_matches_python() {
        // Pinned Python PortableRng(7).gumbel() stream — F3.3 root selection
        // needs these bit-identical (cross-runtime ln parity).
        let mut r = Rng::new(7);
        let seq: Vec<f64> = (0..3).map(|_| r.gumbel()).collect();
        assert_eq!(
            seq,
            [0.7051848236225707, 4.0786199258627525, -0.8373431815918142]
        );
    }
}
