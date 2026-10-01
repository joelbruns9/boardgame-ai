//! Portable SplitMix64 — a bit-for-bit mirror of
//! `games/cantstop/portable_rng.py::PortableRng` (VARIANT_SOLVER_PLAN.md M0).
//!
//! Same constants as `seven_wonders_rust::rng` and `welcome_to_rust::rng`.
//! The Mersenne Twister cannot be reproduced here, so without this a seed
//! would mean two different games and the M1 gate would have nothing to
//! compare.

/// SplitMix64's state step.
const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX1: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX2: u64 = 0x94D0_49BB_1331_11EB;

#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    /// The whole of the state, which is what lets the gate compare the two
    /// engines' generators directly and makes a divergence in the *number of
    /// draws* fail on the step it happens.
    pub fn state(&self) -> u64 {
        self.state
    }

    pub fn set_state(&mut self, state: u64) {
        self.state = state;
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(MIX1);
        z = (z ^ (z >> 27)).wrapping_mul(MIX2);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) from the top 53 bits, matching Python's
    /// `(next_u64() >> 11) / 2**53`.
    pub fn next_float(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }

    /// Integer in [0, n) by plain modulo.
    ///
    /// ⚠ Modulo, not rejection sampling. A rejection loop would make the
    /// number of draws depend on the values drawn, which is exactly what
    /// makes `random.Random` unportable.
    pub fn randrange(&mut self, n: u64) -> u64 {
        assert!(n > 0, "randrange requires n > 0");
        self.next_u64() % n
    }

    /// Inclusive [a, b], matching `PortableRng.randint`.
    pub fn randint(&mut self, a: u64, b: u64) -> u64 {
        assert!(a <= b, "randint requires a <= b");
        a + self.randrange(b - a + 1)
    }

    /// In-place Fisher–Yates (Durstenfeld), high index to low, so the
    /// permutation matches `PortableRng.shuffle`.
    pub fn shuffle<T>(&mut self, seq: &mut [T]) {
        for i in (1..seq.len()).rev() {
            let j = self.randrange(i as u64 + 1) as usize;
            seq.swap(i, j);
        }
    }
}

/// Four dice, drawn exactly as `engine.random_dice` draws them: four
/// independent `randint(1, 6)` calls, in order, *unsorted*.
///
/// Sorting happens in `roll`, not here. Drawing and then sorting in one step
/// would silently change how many values the stream consumed if the two sides
/// ever disagreed about the order.
pub fn roll_dice(rng: &mut Rng) -> [u8; 4] {
    [
        rng.randint(1, 6) as u8,
        rng.randint(1, 6) as u8,
        rng.randint(1, 6) as u8,
        rng.randint(1, 6) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_advances_by_gamma() {
        let mut rng = Rng::new(0);
        rng.next_u64();
        assert_eq!(rng.state(), GAMMA);
    }

    #[test]
    fn randint_stays_in_range() {
        let mut rng = Rng::new(12345);
        for _ in 0..1000 {
            let v = rng.randint(1, 6);
            assert!((1..=6).contains(&v));
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = Rng::new(7);
        let mut seq: Vec<u32> = (0..32).collect();
        rng.shuffle(&mut seq);
        seq.sort();
        assert_eq!(seq, (0..32).collect::<Vec<u32>>());
    }
}
