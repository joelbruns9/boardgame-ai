//! Phase F1 pyo3 bindings: a Rust 7 Wonders Duel engine exposed to Python for
//! the byte-exact replay gate (F1a) and the make/unmake round-trip gate (F1b).
//!
//! The engine is constructed from a fully-locked setup (extracted from a Python
//! `GameState.new`) plus the recorded Great Library draws, and replays action
//! indices. See `state.rs` for why no Python RNG is modelled and why the
//! fingerprint is the equivalence surface.

mod chance;
mod codec;
mod data;
mod encoder;
mod engine;
mod eval;
mod pool;
mod rng;
mod rules;
mod state;
mod tree;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::VecDeque;

use state::{GameState, Setup};

fn card_ids(names: &[String]) -> Vec<usize> {
    names.iter().map(|n| data::card_id(n)).collect()
}
fn wonder_ids(names: &[String]) -> Vec<usize> {
    names.iter().map(|n| data::wonder_id(n)).collect()
}
fn progress_ids(names: &[String]) -> Vec<usize> {
    names.iter().map(|n| data::progress_id(n)).collect()
}

/// A 7WD game state driven from Python by codec action index.
#[pyclass]
struct RustGame {
    state: GameState,
}

#[pymethods]
impl RustGame {
    /// Construct from a fully-locked setup. Lists carry component *names*; Rust
    /// maps them to the same ids Python's `CARD_IDS`/`WONDER_IDS`/`PROGRESS_IDS`
    /// assign (both index into the identical `data.py` tables).
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        first_player, available_progress, unused_progress, wonder_group0,
        wonder_group1, unused_wonders, age1, age2, age3, removed1, removed2,
        removed3, selected_guilds, unused_guilds, library_draws
    ))]
    fn new(
        first_player: usize,
        available_progress: Vec<String>,
        unused_progress: Vec<String>,
        wonder_group0: Vec<String>,
        wonder_group1: Vec<String>,
        unused_wonders: Vec<String>,
        age1: Vec<String>,
        age2: Vec<String>,
        age3: Vec<String>,
        removed1: Vec<String>,
        removed2: Vec<String>,
        removed3: Vec<String>,
        selected_guilds: Vec<String>,
        unused_guilds: Vec<String>,
        library_draws: Vec<Vec<String>>,
    ) -> PyResult<Self> {
        if first_player > 1 {
            return Err(PyValueError::new_err("first_player must be 0 or 1"));
        }
        let setup = Setup {
            first_player,
            available_progress_tokens: progress_ids(&available_progress),
            unused_progress_tokens: progress_ids(&unused_progress),
            wonder_groups: [wonder_ids(&wonder_group0), wonder_ids(&wonder_group1)],
            unused_wonders: wonder_ids(&unused_wonders),
            age_decks: [
                Vec::new(),
                card_ids(&age1),
                card_ids(&age2),
                card_ids(&age3),
            ],
            removed_age_cards: [
                Vec::new(),
                card_ids(&removed1),
                card_ids(&removed2),
                card_ids(&removed3),
            ],
            selected_guilds: card_ids(&selected_guilds),
            unused_guilds: card_ids(&unused_guilds),
        };
        let draws: VecDeque<Vec<usize>> =
            library_draws.iter().map(|d| progress_ids(d)).collect();
        Ok(RustGame {
            state: GameState::from_setup(setup, draws),
        })
    }

    /// Sorted codec indices of exactly the engine's legal actions.
    fn legal_action_indices(&self) -> Vec<usize> {
        codec::legal_action_indices(&self.state)
    }

    /// Canonical integer fingerprint of all game-logic state (RNG excluded).
    fn fingerprint(&self) -> Vec<i32> {
        self.state.fingerprint()
    }

    /// Decode `index` in the current state and apply it (advances the game).
    /// Rejects any index that is not a currently-legal action: this is the
    /// public boundary, and the decoder alone does not verify wonder
    /// ownership/retirement or affordability, so an unchecked index could
    /// otherwise mutate state illegally.
    fn apply_index(&mut self, index: usize) -> PyResult<()> {
        if !codec::legal_action_indices(&self.state).contains(&index) {
            return Err(PyValueError::new_err(format!(
                "illegal action index {index} for the current state"
            )));
        }
        let action = codec::decode_action(&self.state, index);
        self.state.apply_action(&action);
        Ok(())
    }

    /// F1b: apply `index` then unmake, returning whether the *complete* state is
    /// restored (`GameState: PartialEq`, not just the cross-language
    /// fingerprint). Leaves the game unchanged (snapshot-based undo).
    fn roundtrip_ok(&mut self, index: usize) -> PyResult<bool> {
        let before = self.state.clone();
        let undo = self.state.snapshot();
        let action = codec::decode_action(&self.state, index);
        self.state.apply_action(&action);
        self.state.restore(undo);
        Ok(self.state == before)
    }

    /// F1b (strengthened): exhaustive make/unmake audit from the current state —
    /// every legal action to `depth` plies (nested LIFO), full-state undo, and
    /// apply determinism. Non-destructive (operates on clones). Run on sampled
    /// states in the gate; `depth=2` proves nesting without an O(branch^3) cost.
    #[pyo3(signature = (depth=2))]
    fn roundtrip_all_ok(&self, depth: usize) -> bool {
        engine::make_unmake_audit(&self.state, depth).is_ok()
    }

    /// F2.1 foundation: the unseen-card pool read from the public projection.
    /// Returns `(age1, age2, age3, guild, wonders, offboard_progress)`, each a
    /// sorted id list — the encoder's hidden-structure inputs. Viewer-independent
    /// (hidden info is symmetric).
    fn unseen_pool(&self) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
        let p = pool::unseen_pool(&self.state);
        let [age1, age2, age3, guild] = p.cards;
        (age1, age2, age3, guild, p.wonders, p.offboard_progress)
    }

    /// F2.2: the actor-relative encoder token sequence. Each token is
    /// `(type_id, entity_id, aux_id, features)` with `type_id` in `TokenType`
    /// declaration order (GLOBAL=0 … POOL_WONDER=8). Actor-relative: derived
    /// from the pending choice's player, else the active player.
    fn encode(&self) -> Vec<(usize, i32, i32, Vec<f64>)> {
        encoder::encode(&self.state)
            .into_iter()
            .map(|t| (t.type_id, t.entity_id, t.aux_id, t.features))
            .collect()
    }

    /// F3.1a: predicted chance events for action `index`, as
    /// `(kind_id, context)` — kind_id in `ChanceKind` order (CardReveal=0 …
    /// AgeDeal=3), context flattened (CardReveal `[row, x, back]`, AgeDeal
    /// `[age]`, else `[]`).
    fn chance_signature(&self, index: usize) -> Vec<(u8, Vec<i32>)> {
        let action = codec::decode_action(&self.state, index);
        chance::chance_signature(&self.state, &action)
            .into_iter()
            .map(|s| (s.kind as u8, s.context))
            .collect()
    }

    /// F3.1a: all `(outcomes, probability, observable_key)` chains for action
    /// `index`'s enumerable chance specs. Each chain's `outcomes` is one id list
    /// per spec (CardReveal `[card_id]`, GreatLibraryDraw `[p,p,p]`,
    /// WonderGroupReveal `[w,w,w,w]`); `key` equals the outcomes off AGE_DEAL.
    /// Errors on AgeDeal (sample-only).
    fn enumerate_chains(
        &self,
        index: usize,
    ) -> PyResult<Vec<(Vec<Vec<usize>>, f64, Vec<Vec<i32>>)>> {
        let action = codec::decode_action(&self.state, index);
        let specs = chance::chance_signature(&self.state, &action);
        if specs
            .iter()
            .any(|s| s.kind == chance::ChanceKind::AgeDeal)
        {
            return Err(PyValueError::new_err("cannot enumerate AGE_DEAL chains"));
        }
        Ok(chance::enumerate_chains(&self.state, &specs))
    }

    /// F3.1b: sample one chance chain for action `index` from a fresh
    /// `Rng(seed)` — the standalone-seed form the gate compares against Python's
    /// `sample_outcomes(..., PortableRng(seed))`. Returns `(outcomes, prob)` with
    /// `prob` absent when a spec is sample-only (AGE_DEAL).
    fn sample_outcomes(
        &self,
        index: usize,
        seed: u64,
    ) -> (Vec<Vec<usize>>, Option<f64>, Vec<Vec<i32>>) {
        let action = codec::decode_action(&self.state, index);
        let specs = chance::chance_signature(&self.state, &action);
        let mut rng = rng::Rng::new(seed);
        chance::sample_outcomes(&self.state, &specs, &mut rng)
    }

    /// F3.1b: fingerprint of the state after applying action `index` with
    /// supplied chance `outcomes` (one id list per spec). Non-destructive
    /// (snapshot/restore) so the gate can probe many outcomes from one state.
    fn fingerprint_after_chance(
        &mut self,
        index: usize,
        outcomes: Vec<Vec<usize>>,
    ) -> PyResult<Vec<i32>> {
        let undo = self.state.snapshot();
        let action = codec::decode_action(&self.state, index);
        let result = self.state.apply_with_chance(&action, &outcomes);
        match result {
            Ok(()) => {
                let fp = self.state.fingerprint();
                self.state.restore(undo);
                Ok(fp)
            }
            Err(e) => {
                self.state.restore(undo);
                Err(PyValueError::new_err(e))
            }
        }
    }

    /// F3.2: deterministic mock leaf evaluation of the current state
    /// `(value_p0, priors aligned to legal_action_indices)` — the shared oracle
    /// for the tree-equivalence gate.
    fn mock_eval(&self) -> (f64, Vec<f64>) {
        eval::MockEval::eval_state(&self.state)
    }

    /// F3.2: build the closed tree from the current state with `MockEval`, a
    /// fixed round-robin root schedule, `sims` simulations and RNG `seed`, and
    /// return its canonical digest for the 1e-6 equivalence gate.
    #[pyo3(signature = (sims, seed, c_puct=1.5))]
    fn closed_tree_digest(&self, sims: usize, seed: u64, c_puct: f64) -> Vec<f64> {
        let root = tree::closed_tree_fixed(&self.state, sims, &eval::MockEval, seed, c_puct);
        let mut out = Vec::new();
        tree::digest(&root, &mut out);
        out
    }

    /// F3.3: full closed search (Gumbel root + sequential halving) from the
    /// current state under `MockEval`. Returns `(action_index, action_value,
    /// root_value, visits, policy_target, gumbel_topk, sims, tree_digest)` with
    /// `visits`/`policy_target` aligned to `legal_action_indices`.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (sims, top_k, seed, c_puct=1.5, c_visit=50.0, c_scale=1.0, force=false))]
    fn closed_search(
        &self,
        sims: usize,
        top_k: usize,
        seed: u64,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
    ) -> (usize, f64, f64, Vec<u32>, Vec<f64>, Vec<usize>, usize, Vec<f64>) {
        let cfg = tree::SearchConfig {
            sims,
            top_k,
            c_puct,
            c_visit,
            c_scale,
            seed,
            force_expand_root_chance: force,
        };
        let (res, root) = tree::search_closed(&self.state, &eval::MockEval, &cfg);
        let mut dig = Vec::new();
        tree::digest(&root, &mut dig);
        (
            res.action_index,
            res.action_value,
            res.root_value,
            res.visits,
            res.policy_target,
            res.gumbel_topk,
            res.sims,
            dig,
        )
    }

    fn is_complete(&self) -> bool {
        self.state.phase == state::Phase::Complete
    }

    #[getter]
    fn active_player(&self) -> usize {
        self.state.active_player
    }
}

/// Total size of the fixed action space (1202).
#[pyfunction]
fn num_actions() -> usize {
    codec::NUM_ACTIONS
}

/// The encoder schema signature this build produces (must equal Python's
/// `ENCODER_SIGNATURE`). F4 uses it to reject checkpoints trained on a different
/// feature schema.
#[pyfunction]
fn encoder_signature() -> &'static str {
    encoder::ENCODER_SIGNATURE
}

/// `n` consecutive `gumbel()` draws from `Rng(seed)` — lets the gate check
/// cross-runtime `ln` parity in bulk (F3.3 needs bit-identical Gumbel keys).
#[pyfunction]
fn gumbel_stream(seed: u64, n: usize) -> Vec<f64> {
    let mut r = rng::Rng::new(seed);
    (0..n).map(|_| r.gumbel()).collect()
}

#[pymodule]
mod seven_wonders_rust {
    #[pymodule_export]
    use super::RustGame;

    #[pymodule_export]
    use super::num_actions;

    #[pymodule_export]
    use super::encoder_signature;

    #[pymodule_export]
    use super::gumbel_stream;
}

#[cfg(test)]
mod tests {
    //! Rust-side smoke coverage so `cargo test` is load-bearing independent of
    //! the Python gate. The exhaustive cross-language replay lives in
    //! `test_rust_engine_equiv.py`; these lock crate invariants and the F1b
    //! make/unmake audit on a self-contained setup (valid-sized, not shuffled;
    //! enough to drive the draft and Age I, which is where undo is exercised).

    use crate::codec;
    use crate::engine::make_unmake_audit;
    use crate::state::{GameState, Phase, Setup};
    use std::collections::VecDeque;

    fn sample_setup() -> Setup {
        Setup {
            first_player: 0,
            available_progress_tokens: vec![0, 1, 2, 3, 4],
            unused_progress_tokens: vec![5, 6, 7, 8, 9],
            wonder_groups: [vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
            unused_wonders: vec![8, 9, 10, 11],
            age_decks: [
                Vec::new(),
                (0..20).collect(),
                (0..20).collect(),
                (0..20).collect(),
            ],
            removed_age_cards: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            selected_guilds: Vec::new(),
            unused_guilds: Vec::new(),
        }
    }

    #[test]
    fn action_space_is_1202() {
        assert_eq!(codec::NUM_ACTIONS, 1202);
    }

    #[test]
    fn fingerprint_deterministic_and_clone_equal() {
        let g = GameState::from_setup(sample_setup(), VecDeque::new());
        assert_eq!(g.fingerprint(), g.fingerprint());
        assert!(g.clone() == g);
    }

    #[test]
    fn encoder_feature_counts_match_schema() {
        use crate::encoder::{encode, FEATURE_COUNTS};
        let mut g = GameState::from_setup(sample_setup(), VecDeque::new());
        let mut steps = 0;
        while g.phase != Phase::Complete && steps < 14 {
            for t in encode(&g) {
                assert_eq!(
                    t.features.len(),
                    FEATURE_COUNTS[t.type_id],
                    "token type {} feature count",
                    t.type_id
                );
            }
            let legal = codec::legal_action_indices(&g);
            g.apply_action(&codec::decode_action(&g, legal[0]));
            steps += 1;
        }
        assert!(steps > 8);
    }

    #[test]
    fn make_unmake_audit_holds_through_age_one() {
        let mut g = GameState::from_setup(sample_setup(), VecDeque::new());
        // Draft branch is <= 4 wide, so a depth-2 (nested) audit is cheap here.
        make_unmake_audit(&g, 2).expect("draft make/unmake");
        // Drive the 8 draft picks and a few Age I decisions, auditing the full
        // legal fan-out (depth 1) at each live state.
        let mut steps = 0;
        while g.phase != Phase::Complete && steps < 14 {
            make_unmake_audit(&g, 1).expect("live-state make/unmake");
            let legal = codec::legal_action_indices(&g);
            g.apply_action(&codec::decode_action(&g, legal[0]));
            steps += 1;
        }
        assert!(steps > 8, "test should reach Age I play, got {steps} steps");
    }
}
