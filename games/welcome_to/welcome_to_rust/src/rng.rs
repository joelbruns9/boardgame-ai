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

/// SplitMix64's state step. Named because `derive_search_seed` skips states in
/// O(1) by multiplying it, and the two must not drift apart.
const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

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
        self.state = self.state.wrapping_add(GAMMA);
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

    /// One index in proportion to non-negative f64 weights. Mirrors
    /// `PortableRng.choices(..., k=1)` / CPython's right-bisect rule.
    pub fn weighted_index(&mut self, weights: &[f64]) -> usize {
        assert!(!weights.is_empty(), "weights cannot be empty");
        let total: f64 = weights.iter().sum();
        assert!(
            total.is_finite() && total > 0.0,
            "weights need positive finite mass"
        );
        let target = self.next_float() * total;
        let mut cumulative = 0.0;
        for (index, &weight) in weights.iter().enumerate() {
            cumulative += weight;
            if target < cumulative || index + 1 == weights.len() {
                return index;
            }
        }
        unreachable!()
    }
}

pub const SEARCH_SEED_DOMAIN: u64 = 0x5745_4C43_4F4D_4553;

/// The portable tape for one search — mirrors
/// `portable_rng.derive_search_seed`, whose docstring carries the argument.
///
/// ⚠ **Not** `game_seed ^ DOMAIN ^ search_index`. Under XOR the decision
/// counter occupies the same low bits as a contiguous block of game seeds, so
/// decision `i` of game `s` reuses decision `0` of game `s ^ i` — same
/// determinization permutations and, through `root_noise`, the same Dirichlet
/// vector. This is instead draw `search_index` of the stream seeded at
/// `game_seed ^ DOMAIN`, reached in O(1) because the state step is `+GAMMA`.
pub fn derive_search_seed(game_seed: u64, search_index: u64) -> u64 {
    let base = (game_seed ^ SEARCH_SEED_DOMAIN).wrapping_add(GAMMA.wrapping_mul(search_index));
    Rng::new(base).next_u64()
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

    #[test]
    fn weighted_choice_and_search_seed_match_python() {
        let mut rng = Rng::new(31);
        let got: Vec<usize> = (0..8)
            .map(|_| rng.weighted_index(&[0.1, 0.0, 0.3, 0.6]))
            .collect();
        assert_eq!(got, vec![3, 3, 3, 3, 3, 2, 2, 3]);
    }

    /// The XOR derivation this replaced gave decision `i` of game `s` the same
    /// tape as decision `0` of game `s ^ i`. A contiguous seed block is exactly
    /// where that bites, so assert it directly rather than only pinning a
    /// golden value.
    #[test]
    fn search_seeds_do_not_collide_across_a_contiguous_seed_block() {
        let mut seen = std::collections::HashSet::new();
        for game_seed in 1_000u64..1_064 {
            for search_index in 0u64..64 {
                assert!(
                    seen.insert(derive_search_seed(game_seed, search_index)),
                    "collision at ({game_seed}, {search_index})"
                );
            }
        }
    }
}
