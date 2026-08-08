//! Language-neutral durable buffer digests.

use crate::state::GameState;
use sha2::{Digest, Sha256};

pub const VERSION: &str = "logic-sha256-v1";

fn update_state(hasher: &mut Sha256, state: &GameState) {
    let fingerprint = state.fingerprint();
    hasher.update((fingerprint.len() as u32).to_le_bytes());
    for value in fingerprint {
        hasher.update(value.to_le_bytes());
    }
}

fn finish(hasher: Sha256) -> String {
    let bytes = hasher.finalize();
    let mut encoded = String::with_capacity(7 + bytes.len() * 2);
    encoded.push_str("sha256:");
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

pub fn state_digest(state: &GameState) -> String {
    let mut hasher = Sha256::new();
    update_state(&mut hasher, state);
    finish(hasher)
}

pub struct TrajectoryDigest(Sha256);

impl TrajectoryDigest {
    pub fn new() -> Self {
        Self(Sha256::new())
    }

    pub fn update(&mut self, state: &GameState) {
        update_state(&mut self.0, state);
    }

    pub fn finish(self) -> String {
        finish(self.0)
    }
}
