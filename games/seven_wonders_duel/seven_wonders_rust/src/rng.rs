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
}
