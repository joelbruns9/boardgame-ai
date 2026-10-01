//! Audit-only dice, indexed by completed turn and roll. Legacy game RNG stays unchanged.
use sha2::{Digest, Sha256};
use crate::rng::Rng;

fn hash_seed(payload: &str) -> u64 {
    let digest = Sha256::digest(payload.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("eight bytes"))
}

pub fn shared_dice(seed: u64, turn: u32, roll: u32) -> [u8; 4] {
    let seed = hash_seed(&format!("cantstop-shared-dice-v1|{seed}|{turn}|{roll}"));
    let mut rng = Rng::new(hash_seed(&format!("cantstop-decision-rng-v1|{seed}|search|0")));
    let mut dice = [0; 4];
    for die in &mut dice {
        loop {
            let x = rng.next_u64();
            // Python rejects x >= 2**64 - (2**64 % 6).
            if x < u64::MAX - 3 {
                *die = (x % 6 + 1) as u8;
                break;
            }
        }
    }
    dice
}

pub struct AuditStream {
    pub seed: u64,
    pub rolls: u32,
    pub roll_index: u32,
    pub max_rolls: u32,
    pub dice_luck: bool,
    pub correction: Vec<f64>,
}
