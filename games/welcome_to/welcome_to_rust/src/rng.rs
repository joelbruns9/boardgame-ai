//! Portable SplitMix64 — a bit-for-bit mirror of
//! `games/welcome_to/portable_rng.py::PortableRng` (RUST_PORT_PLAN.md M0-B).
//!
//! The Mersenne Twister cannot be reproduced here, so "same seed" would give
//! two different games — and a Rust-generated trajectory could not be replayed
//! by the Python trainer at all, because `datagen.replay` rebuilds the deal
//! from the seed. This is the shared stream that makes both possible.
//!
//! Same constants as `seven_wonders_rust::rng` and
//! `kingdomino_rust::search::splitmix64`.

#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    /// The whole of the state, which is what makes `clone` exact and lets the
    /// M1 gate compare the two engines' generators directly.
    pub fn state(&self) -> u64 {
        self.state
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
    #[allow(dead_code)] // the engine draws integers; search (M5) wants this
    pub fn next_float(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }

    /// Integer in [0, n) by modulo, matching Python `next_u64() % n`.
    #[allow(dead_code)] // the engine draws by shuffle/choice; search (M5) wants this
    pub fn randrange(&mut self, n: u64) -> u64 {
        assert!(n > 0, "randrange requires n > 0");
        self.next_u64() % n
    }

    /// Low `k` bits (k <= 64). Used to reseed a determinized copy.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        assert!(k > 0 && k <= 64, "getrandbits supports 1..64 bits");
        let value = self.next_u64();
        if k == 64 {
            value
        } else {
            value & ((1u64 << k) - 1)
        }
    }

    /// In-place Fisher–Yates (Durstenfeld), high index to low, so the
    /// permutation matches `PortableRng.shuffle`.
    pub fn shuffle<T>(&mut self, seq: &mut [T]) {
        for i in (1..seq.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            seq.swap(i, j);
        }
    }

    /// One element, uniformly.
    ///
    /// ⚠ **Not** `random.Random.choice`, which draws through `_randbelow` and
    /// its rejection loop; this is a plain modulo of one `next_u64`, matching
    /// `PortableRng.choice`.
    pub fn choice<T: Copy>(&mut self, seq: &[T]) -> T {
        assert!(!seq.is_empty(), "cannot choose from an empty sequence");
        seq[(self.next_u64() % seq.len() as u64) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden values taken from the Python reference:
    /// `[PortableRng(12345).next_u64() for _ in range(4)]`.
    #[test]
    fn matches_python_stream() {
        let mut rng = Rng::new(12345);
        let got: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
        assert_eq!(
            got,
            vec![
                2_454_886_589_211_414_944,
                3_778_200_017_661_327_597,
                2_205_171_434_679_333_405,
                3_248_800_117_070_709_450,
            ]
        );
    }

    /// `r = PortableRng(7); s = list(range(8)); r.shuffle(s)` in Python leaves
    /// this permutation and this state.
    #[test]
    fn shuffle_matches_python() {
        let mut rng = Rng::new(7);
        let mut seq: Vec<u32> = (0..8).collect();
        rng.shuffle(&mut seq);
        assert_eq!(seq, vec![1, 4, 5, 2, 6, 0, 3, 7]);
        assert_eq!(rng.state(), 6_018_027_440_424_182_938);
    }
}
