//! Phase F1 pyo3 bindings: a Rust 7 Wonders Duel engine exposed to Python for
//! the byte-exact replay gate (F1a) and the make/unmake round-trip gate (F1b).
//!
//! The engine is constructed from a fully-locked setup (extracted from a Python
//! `GameState.new`) plus the recorded Great Library draws, and replays action
//! indices. See `state.rs` for why no Python RNG is modelled and why the
//! fingerprint is the equivalence surface.

mod bots;
mod chance;
mod codec;
mod data;
mod encoder;
mod engine;
mod eval;
mod pool;
mod rng;
mod rules;
mod self_play;
mod state;
mod tree;
mod tree_resumable;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::VecDeque;

use eval::Eval;
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

fn self_play_record_to_py(py: Python<'_>, record: self_play::GameRecord) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("schema", 1)?;
    out.set_item("spec_version", "codec-1")?;
    out.set_item("seed", record.seed)?;
    out.set_item("first_player", record.first_player)?;
    out.set_item("iteration", record.iteration)?;
    out.set_item("winner", record.winner)?;
    out.set_item(
        "victory_type",
        record.victory_type.map(|kind| match kind {
            state::VictoryType::Military => "military",
            state::VictoryType::Scientific => "scientific",
            state::VictoryType::Civilian => "civilian",
            state::VictoryType::SharedCivilian => "shared_civilian",
        }),
    )?;
    out.set_item("scores", record.scores)?;
    let kind = if record.agent_names.iter().any(|name| name != "network") {
        "mixed"
    } else {
        "self_play"
    };
    out.set_item(
        "agents",
        [
            ("p0", record.agent_names[0].as_str()),
            ("p1", record.agent_names[1].as_str()),
            ("kind", kind),
        ]
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>(),
    )?;

    let moves = PyList::empty(py);
    for row in record.moves {
        let item = PyDict::new(py);
        item.set_item("i", row.i)?;
        item.set_item("actor", row.actor)?;
        item.set_item("action", row.action)?;
        item.set_item("legal", row.legal)?;
        item.set_item("visits", row.visits)?;
        item.set_item(
            "policy_target",
            if row.is_bot {
                None
            } else {
                Some(row.policy_target)
            },
        )?;
        item.set_item(
            "root_value",
            if row.is_bot {
                None
            } else {
                Some(row.root_value)
            },
        )?;
        item.set_item("sims", row.sims)?;
        item.set_item("mode", if row.is_bot { "bot" } else { "closed" })?;
        item.set_item(
            "gumbel_topk",
            if row.is_bot {
                None
            } else {
                Some(row.gumbel_topk)
            },
        )?;
        item.set_item("policy_excluded", row.policy_excluded)?;
        item.set_item("full_search", row.full_search)?;
        item.set_item("search_seed", row.search_seed)?;
        moves.append(item)?;
    }
    out.set_item("moves", moves)?;

    let chance_log = PyList::empty(py);
    for event in record.chance_log {
        let item = PyDict::new(py);
        item.set_item("move_index", event.move_index)?;
        item.set_item("kind_id", event.kind as u8)?;
        item.set_item("outcome_ids", event.outcome.clone())?;
        item.set_item(
            "outcome",
            event
                .outcome
                .into_iter()
                .map(|id| self_play::component_name(event.kind, id))
                .collect::<Vec<_>>(),
        )?;
        chance_log.append(item)?;
    }
    out.set_item("chance_log", chance_log)?;
    out.set_item("final_fingerprint", record.final_fingerprint)?;
    Ok(out.unbind())
}

#[allow(clippy::too_many_arguments)]
fn make_self_play_config(
    game_seed: u64,
    iteration: Option<i64>,
    leaf_batch: usize,
    cheap_sims_min: usize,
    cheap_sims_max: usize,
    full_sims_min: usize,
    full_sims_max: usize,
    full_search_fraction: f64,
    top_k: usize,
    draft_prior: f64,
    c_puct: f64,
    c_visit: f64,
    c_scale: f64,
    force: bool,
    age_deal_samples: usize,
    max_moves: usize,
) -> self_play::SelfPlayConfig {
    self_play::SelfPlayConfig {
        game_seed,
        iteration,
        leaf_batch,
        leaf_batch_by_player: None,
        deterministic_actions: false,
        cheap_sims_min,
        cheap_sims_max,
        full_sims_min,
        full_sims_max,
        full_search_fraction,
        top_k,
        draft_prior,
        c_puct,
        c_visit,
        c_scale,
        force_expand_root_chance: force,
        age_deal_samples,
        age_deal_samples_by_player: None,
        bot_by_player: [None, None],
        bot_exploration: 0.0,
        bot_policy_iterations: 10,
        max_moves,
    }
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
        let draws: VecDeque<Vec<usize>> = library_draws.iter().map(|d| progress_ids(d)).collect();
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

    #[pyo3(signature = (kind, seed=0, exploration=0.0))]
    fn bot_action(&self, kind: &str, seed: u64, exploration: f64) -> PyResult<usize> {
        let kind = bots::BotKind::parse(kind)
            .ok_or_else(|| PyValueError::new_err(format!("unknown Rust bot: {kind}")))?;
        if !(0.0..=1.0).contains(&exploration) {
            return Err(PyValueError::new_err("exploration must be in [0, 1]"));
        }
        Ok(bots::select_action(
            &self.state,
            kind,
            &mut rng::Rng::new(seed),
            exploration,
        ))
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
    fn unseen_pool(
        &self,
    ) -> (
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
    ) {
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
        if specs.iter().any(|s| s.kind == chance::ChanceKind::AgeDeal) {
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
    fn closed_tree_digest(&self, sims: usize, seed: u64, c_puct: f64) -> PyResult<Vec<f64>> {
        let root = tree::closed_tree_fixed(&self.state, sims, &eval::MockEval, seed, c_puct)?;
        let mut out = Vec::new();
        tree::digest(&root, &mut out);
        Ok(out)
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
    ) -> PyResult<(
        usize,
        f64,
        f64,
        Vec<u32>,
        Vec<f64>,
        Vec<usize>,
        usize,
        Vec<f64>,
    )> {
        let cfg = tree::SearchConfig {
            sims,
            top_k,
            c_puct,
            c_visit,
            c_scale,
            seed,
            force_expand_root_chance: force,
            age_deal_samples: 0,
        };
        let (res, root) = tree::search_closed(&self.state, &eval::MockEval, &cfg)?;
        let mut dig = Vec::new();
        tree::digest(&root, &mut dig);
        Ok((
            res.action_index,
            res.action_value,
            res.root_value,
            res.visits,
            res.policy_target,
            res.gumbel_topk,
            res.sims,
            dig,
        ))
    }

    /// F4.1: arena-backed, phase-split closed search.  `leaf_batch=1` is the
    /// exact refactor gate: selection/materialization yields an evaluation
    /// request, evaluation is applied separately, and stable arena paths are
    /// backed up before the next simulation is selected.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (sims, top_k, seed, c_puct=1.5, c_visit=50.0, c_scale=1.0, force=false))]
    fn closed_search_resumable(
        &self,
        sims: usize,
        top_k: usize,
        seed: u64,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
    ) -> PyResult<(
        usize,
        f64,
        f64,
        Vec<u32>,
        Vec<f64>,
        Vec<usize>,
        usize,
        Vec<f64>,
    )> {
        let cfg = tree::SearchConfig {
            sims,
            top_k,
            c_puct,
            c_visit,
            c_scale,
            seed,
            force_expand_root_chance: force,
            age_deal_samples: 0,
        };
        let (res, arena) = tree_resumable::search_closed(&self.state, &eval::MockEval, &cfg)?;
        let mut dig = Vec::new();
        tree_resumable::digest(&arena, &mut dig);
        Ok((
            res.action_index,
            res.action_value,
            res.root_value,
            res.visits,
            res.policy_target,
            res.gumbel_topk,
            res.sims,
            dig,
        ))
    }

    /// F4.1 real-evaluator counterpart to `closed_search_resumable`. This is
    /// still the scalar F3.4 Python adapter; F4.4/F4.5 replace the boundary,
    /// while this method keeps the phase split independently gateable today.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (adapter, sims, top_k, seed, c_puct=1.5, c_visit=50.0, c_scale=1.0, force=false))]
    fn closed_search_resumable_net(
        &self,
        adapter: Py<PyAny>,
        sims: usize,
        top_k: usize,
        seed: u64,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
    ) -> PyResult<(
        usize,
        f64,
        f64,
        Vec<u32>,
        Vec<f64>,
        Vec<usize>,
        usize,
        Vec<f64>,
    )> {
        let cfg = tree::SearchConfig {
            sims,
            top_k,
            c_puct,
            c_visit,
            c_scale,
            seed,
            force_expand_root_chance: force,
            age_deal_samples: 0,
        };
        let evaluator = eval::PyEval::new(adapter);
        let (res, arena) = tree_resumable::search_closed(&self.state, &evaluator, &cfg)?;
        let mut dig = Vec::new();
        tree_resumable::digest(&arena, &mut dig);
        Ok((
            res.action_index,
            res.action_value,
            res.root_value,
            res.visits,
            res.policy_target,
            res.gumbel_topk,
            res.sims,
            dig,
        ))
    }

    /// F4.2 WU-PUCT leaf waves under the deterministic mock evaluator. Returns
    /// the normal search tuple plus `(scheduled, requested, unique, terminal,
    /// collisions, waves, max_wave_paths, max_wave_unique)`, completed-Q aligned
    /// to legal actions, and the tree digest.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (leaf_batch, sims, top_k, seed, c_puct=1.5, c_visit=50.0, c_scale=1.0, force=false))]
    fn closed_search_batched(
        &self,
        leaf_batch: usize,
        sims: usize,
        top_k: usize,
        seed: u64,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
    ) -> PyResult<(
        usize,
        f64,
        f64,
        Vec<u32>,
        Vec<f64>,
        Vec<usize>,
        usize,
        (usize, usize, usize, usize, usize, usize, usize, usize),
        Vec<f64>,
        Vec<f64>,
    )> {
        let cfg = tree::SearchConfig {
            sims,
            top_k,
            c_puct,
            c_visit,
            c_scale,
            seed,
            force_expand_root_chance: force,
            age_deal_samples: 0,
        };
        let (res, arena, metrics) =
            tree_resumable::search_closed_batched(&self.state, &eval::MockEval, &cfg, leaf_batch)?;
        let mut dig = Vec::new();
        tree_resumable::digest(&arena, &mut dig);
        Ok((
            res.action_index,
            res.action_value,
            res.root_value,
            res.visits,
            res.policy_target,
            res.gumbel_topk,
            res.sims,
            (
                metrics.scheduled_simulations,
                metrics.requested_nn_leaves,
                metrics.unique_nn_leaves,
                metrics.terminal_leaves,
                metrics.collisions,
                metrics.leaf_waves,
                metrics.max_wave_paths,
                metrics.max_wave_unique,
            ),
            metrics.root_completed_q,
            dig,
        ))
    }

    /// F4.2 WU leaf-wave path through the scalar correctness adapter. The
    /// adapter remains one Python call per unique leaf until the F4.4 global
    /// coalescer; this surface gates batched search semantics with a real net.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (adapter, leaf_batch, sims, top_k, seed, c_puct=1.5, c_visit=50.0, c_scale=1.0, force=false))]
    fn closed_search_batched_net(
        &self,
        adapter: Py<PyAny>,
        leaf_batch: usize,
        sims: usize,
        top_k: usize,
        seed: u64,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
    ) -> PyResult<(
        usize,
        f64,
        f64,
        Vec<u32>,
        Vec<f64>,
        Vec<usize>,
        usize,
        (usize, usize, usize, usize, usize, usize, usize, usize),
        Vec<f64>,
        Vec<f64>,
    )> {
        let cfg = tree::SearchConfig {
            sims,
            top_k,
            c_puct,
            c_visit,
            c_scale,
            seed,
            force_expand_root_chance: force,
            age_deal_samples: 0,
        };
        let evaluator = eval::PyEval::new(adapter);
        let (res, arena, metrics) =
            tree_resumable::search_closed_batched(&self.state, &evaluator, &cfg, leaf_batch)?;
        let mut dig = Vec::new();
        tree_resumable::digest(&arena, &mut dig);
        Ok((
            res.action_index,
            res.action_value,
            res.root_value,
            res.visits,
            res.policy_target,
            res.gumbel_topk,
            res.sims,
            (
                metrics.scheduled_simulations,
                metrics.requested_nn_leaves,
                metrics.unique_nn_leaves,
                metrics.terminal_leaves,
                metrics.collisions,
                metrics.leaf_waves,
                metrics.max_wave_paths,
                metrics.max_wave_unique,
            ),
            metrics.root_completed_q,
            dig,
        ))
    }

    /// F3.4: like `closed_search` but with the real net. `adapter` is a Python
    /// callable `(tokens, actor, legal) -> (value_actor, priors)`; the Rust
    /// encoder (F2) feeds it, so results match Python's searcher on the same net.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (adapter, sims, top_k, seed, c_puct=1.5, c_visit=50.0, c_scale=1.0, force=false))]
    fn closed_search_net(
        &self,
        adapter: Py<PyAny>,
        sims: usize,
        top_k: usize,
        seed: u64,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
    ) -> PyResult<(
        usize,
        f64,
        f64,
        Vec<u32>,
        Vec<f64>,
        Vec<usize>,
        usize,
        Vec<f64>,
    )> {
        let cfg = tree::SearchConfig {
            sims,
            top_k,
            c_puct,
            c_visit,
            c_scale,
            seed,
            force_expand_root_chance: force,
            age_deal_samples: 0,
        };
        let evaluator = eval::PyEval::new(adapter);
        let (res, root) = tree::search_closed(&self.state, &evaluator, &cfg)?;
        let mut dig = Vec::new();
        tree::digest(&root, &mut dig);
        Ok((
            res.action_index,
            res.action_value,
            res.root_value,
            res.visits,
            res.policy_target,
            res.gumbel_topk,
            res.sims,
            dig,
        ))
    }

    /// F4.3: run one complete network-vs-network self-play game in Rust under
    /// the deterministic Phase-D schedule.  Python is called only for neural
    /// leaf evaluations and receives one completed raw record at the end.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        adapter, game_seed, leaf_batch, cheap_sims_min, cheap_sims_max,
        full_sims_min, full_sims_max, full_search_fraction, top_k, draft_prior,
        iteration=None, c_puct=1.5, c_visit=50.0, c_scale=1.0, force=false,
        age_deal_samples=0, max_moves=256
    ))]
    fn self_play_net(
        &self,
        adapter: Py<PyAny>,
        game_seed: u64,
        leaf_batch: usize,
        cheap_sims_min: usize,
        cheap_sims_max: usize,
        full_sims_min: usize,
        full_sims_max: usize,
        full_search_fraction: f64,
        top_k: usize,
        draft_prior: f64,
        iteration: Option<i64>,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
        age_deal_samples: usize,
        max_moves: usize,
    ) -> PyResult<Py<PyDict>> {
        let cfg = make_self_play_config(
            game_seed,
            iteration,
            leaf_batch,
            cheap_sims_min,
            cheap_sims_max,
            full_sims_min,
            full_sims_max,
            full_search_fraction,
            top_k,
            draft_prior,
            c_puct,
            c_visit,
            c_scale,
            force,
            age_deal_samples,
            max_moves,
        );
        let evaluator = eval::PyEval::new(adapter);
        let record = self_play::run(&self.state, &evaluator, &cfg)?;
        Python::attach(|py| self_play_record_to_py(py, record))
    }

    /// Deterministic mock-evaluator counterpart used by the F4.3 full-game
    /// oracle and replay/schema gates.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        game_seed, leaf_batch, cheap_sims_min, cheap_sims_max, full_sims_min,
        full_sims_max, full_search_fraction, top_k, draft_prior,
        iteration=None, c_puct=1.5, c_visit=50.0, c_scale=1.0, force=false,
        age_deal_samples=0, max_moves=256
    ))]
    fn self_play_mock(
        &self,
        game_seed: u64,
        leaf_batch: usize,
        cheap_sims_min: usize,
        cheap_sims_max: usize,
        full_sims_min: usize,
        full_sims_max: usize,
        full_search_fraction: f64,
        top_k: usize,
        draft_prior: f64,
        iteration: Option<i64>,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
        age_deal_samples: usize,
        max_moves: usize,
    ) -> PyResult<Py<PyDict>> {
        let cfg = make_self_play_config(
            game_seed,
            iteration,
            leaf_batch,
            cheap_sims_min,
            cheap_sims_max,
            full_sims_min,
            full_sims_max,
            full_search_fraction,
            top_k,
            draft_prior,
            c_puct,
            c_visit,
            c_scale,
            force,
            age_deal_samples,
            max_moves,
        );
        let record = self_play::run(&self.state, &eval::MockEval, &cfg)?;
        Python::attach(|py| self_play_record_to_py(py, record))
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

/// `x.ln()` for each input — the gate uses it to confirm cross-runtime `ln`
/// parity over the range `log_prior = ln(max(prior, 1e-12))` covers.
#[pyfunction]
fn ln_values(xs: Vec<f64>) -> Vec<f64> {
    xs.iter().map(|&x| x.ln()).collect()
}

fn scheduler_result_to_py(
    py: Python<'_>,
    result: self_play::SchedulerResult,
) -> PyResult<(Vec<Py<PyDict>>, Py<PyDict>)> {
    let records = result
        .records
        .into_iter()
        .map(|record| self_play_record_to_py(py, record))
        .collect::<PyResult<_>>()?;
    let metrics = PyDict::new(py);
    let m = result.metrics;
    metrics.set_item("games", m.games)?;
    metrics.set_item("moves", m.moves)?;
    metrics.set_item("simulations", m.simulations)?;
    metrics.set_item("requested_nn_leaves", m.requested_nn_leaves)?;
    metrics.set_item("unique_nn_leaves", m.unique_nn_leaves)?;
    metrics.set_item("terminal_leaves", m.terminal_leaves)?;
    metrics.set_item("collisions", m.collisions)?;
    metrics.set_item("global_batches", m.global_batches)?;
    metrics.set_item("global_rows", m.global_rows)?;
    metrics.set_item("root_rows", m.root_rows)?;
    metrics.set_item("leaf_rows", m.leaf_rows)?;
    metrics.set_item("forced_rows", m.forced_rows)?;
    metrics.set_item("forced_card_reveal_rows", m.forced_rows_by_kind[0])?;
    metrics.set_item("forced_great_library_rows", m.forced_rows_by_kind[1])?;
    metrics.set_item("forced_wonder_group_rows", m.forced_rows_by_kind[2])?;
    metrics.set_item("forced_age_deal_rows", m.forced_rows_by_kind[3])?;
    metrics.set_item("ordinary_leaf_rows", m.ordinary_leaf_rows)?;
    metrics.set_item("forced_cache_hits", m.forced_cache_hits)?;
    metrics.set_item("forced_rows_per_search", m.forced_rows_per_search.clone())?;
    metrics.set_item("max_batch_rows", m.max_batch_rows)?;
    metrics.set_item("scheduler_cycles", m.scheduler_cycles)?;
    metrics.set_item("scheduler_workers", m.scheduler_workers)?;
    metrics.set_item("max_inflight_batches", m.max_inflight_batches)?;
    metrics.set_item("boundary_tokens", m.boundary_tokens)?;
    metrics.set_item("boundary_padded_tokens", m.boundary_padded_tokens)?;
    metrics.set_item("boundary_max_tokens", m.boundary_max_tokens)?;
    metrics.set_item("encode_pack_ns", m.encode_pack_ns)?;
    metrics.set_item("queue_wait_ns", m.queue_wait_ns)?;
    metrics.set_item("py_call_ns", m.py_call_ns)?;
    metrics.set_item("extract_ns", m.extract_ns)?;
    metrics.set_item("rust_tree_ns", m.rust_tree_ns)?;
    metrics.set_item("rust_chance_ns", m.rust_chance_ns)?;
    metrics.set_item("rust_record_ns", m.rust_record_ns)?;
    metrics.set_item("scatter_ns", m.scatter_ns)?;
    metrics.set_item("scheduler_ready_slot_cycles", m.scheduler_ready_slot_cycles)?;
    metrics.set_item(
        "scheduler_waiting_slot_cycles",
        m.scheduler_waiting_slot_cycles,
    )?;
    metrics.set_item("scheduler_idle_slot_cycles", m.scheduler_idle_slot_cycles)?;
    metrics.set_item(
        "padding_ratio",
        if m.boundary_padded_tokens == 0 {
            0.0
        } else {
            1.0 - m.boundary_tokens as f64 / m.boundary_padded_tokens as f64
        },
    )?;
    metrics.set_item("batch_rows", m.batch_rows.clone())?;
    metrics.set_item(
        "mean_batch_rows",
        if m.batch_rows.is_empty() {
            0.0
        } else {
            m.batch_rows.iter().sum::<usize>() as f64 / m.batch_rows.len() as f64
        },
    )?;
    Ok((records, metrics.unbind()))
}

fn search_result_to_py(
    py: Python<'_>,
    result: tree::SearchResult,
    metrics: tree_resumable::SearchMetrics,
    digest: Vec<f64>,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("action", result.action_index)?;
    out.set_item("action_value", result.action_value)?;
    out.set_item("root_value", result.root_value)?;
    out.set_item("visits", result.visits)?;
    out.set_item("policy", result.policy_target)?;
    out.set_item("topk", result.gumbel_topk)?;
    out.set_item("sims", result.sims)?;
    out.set_item("completed_q", metrics.root_completed_q)?;
    out.set_item("digest", digest)?;
    let counters = PyDict::new(py);
    counters.set_item("scheduled", metrics.scheduled_simulations)?;
    counters.set_item("requested", metrics.requested_nn_leaves)?;
    counters.set_item("unique", metrics.unique_nn_leaves)?;
    counters.set_item("terminal", metrics.terminal_leaves)?;
    counters.set_item("collisions", metrics.collisions)?;
    counters.set_item("waves", metrics.leaf_waves)?;
    counters.set_item("max_wave_paths", metrics.max_wave_paths)?;
    counters.set_item("max_wave_unique", metrics.max_wave_unique)?;
    out.set_item("metrics", counters)?;
    let nn_work = PyDict::new(py);
    nn_work.set_item("forced_rows", metrics.forced_outcome_rows)?;
    nn_work.set_item("forced_cache_hits", metrics.cached_forced_leaves)?;
    out.set_item("nn_work", nn_work)?;
    Ok(out.unbind())
}

/// F4.6 position-calibration primitive: run many independent root searches
/// through one flat global evaluator instead of paying a scalar Python hop for
/// every leaf. Results remain in input order and use the same resumable search.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    adapter, games, search_seeds, global_batch_cap, leaf_batch, sims, top_k,
    c_puct=1.5, c_visit=50.0, c_scale=1.0, force=false,
    age_deal_samples=0, inference_timeout_ms=0.0
))]
fn search_many_flat_net(
    py: Python<'_>,
    adapter: Py<PyAny>,
    games: Vec<Py<RustGame>>,
    search_seeds: Vec<u64>,
    global_batch_cap: usize,
    leaf_batch: usize,
    sims: usize,
    top_k: usize,
    c_puct: f64,
    c_visit: f64,
    c_scale: f64,
    force: bool,
    age_deal_samples: usize,
    inference_timeout_ms: f64,
) -> PyResult<Vec<Py<PyDict>>> {
    if games.is_empty() || games.len() != search_seeds.len() {
        return Err(PyValueError::new_err(
            "games and search_seeds must be non-empty and aligned",
        ));
    }
    if global_batch_cap == 0 || leaf_batch == 0 || sims == 0 || top_k == 0 {
        return Err(PyValueError::new_err(
            "global_batch_cap, leaf_batch, sims, and top_k must be positive",
        ));
    }
    if leaf_batch > global_batch_cap {
        return Err(PyValueError::new_err(format!(
            "leaf_batch={leaf_batch} exceeds global_batch_cap={global_batch_cap}"
        )));
    }
    let states: Vec<GameState> = games
        .iter()
        .map(|game| game.borrow(py).state.clone())
        .collect();
    let (worker, timed_out, _boundary_metrics, worker_handle) =
        eval::spawn_py_flat_worker(adapter, inference_timeout_ms, global_batch_cap)?;
    let outputs = py.detach(move || {
        let state_refs: Vec<&GameState> = states.iter().collect();
        let actors: Vec<_> = states.iter().map(tree::state_actor).collect();
        let legals: Vec<_> = states.iter().map(codec::legal_action_indices).collect();
        let roots = worker.evaluate_batch_prepared(&state_refs, &actors, &legals)?;
        let mut sessions = Vec::with_capacity(states.len());
        for ((state, seed), evaluation) in states.iter().zip(search_seeds).zip(roots) {
            let cfg = tree::SearchConfig {
                sims,
                top_k,
                c_puct,
                c_visit,
                c_scale,
                seed,
                force_expand_root_chance: force,
                age_deal_samples,
            };
            let session = if force {
                tree_resumable::begin_search_from_root_forced(state, &cfg, leaf_batch, evaluation)?
            } else {
                tree_resumable::begin_search_from_root(state, &cfg, leaf_batch, evaluation)?
            };
            sessions.push(Some(session));
        }

        struct Group {
            slot: usize,
            request: tree_resumable::EvalBatchRequest,
            states: Vec<GameState>,
            actors: Vec<usize>,
            legals: Vec<Vec<usize>>,
        }

        let mut completed: Vec<
            Option<(tree::SearchResult, tree_resumable::SearchMetrics, Vec<f64>)>,
        > = (0..sessions.len()).map(|_| None).collect();
        while completed.iter().any(Option::is_none) {
            let mut groups = VecDeque::new();
            let live = sessions
                .iter()
                .filter(|session| session.is_some())
                .count()
                .max(1);
            let forced_row_limit = (global_batch_cap / live).max(1);
            for slot in 0..sessions.len() {
                let Some(session) = sessions[slot].as_mut() else {
                    continue;
                };
                match session.next_event_with_limit(forced_row_limit) {
                    Ok(tree_resumable::SearchEvent::Evaluation(request)) => {
                        let states = match session.evaluation_states(&request) {
                            Ok(rows) => rows.into_iter().cloned().collect(),
                            Err(err) => {
                                session.cancel_pending();
                                return Err(err);
                            }
                        };
                        groups.push_back(Group {
                            slot,
                            actors: request.leaves.iter().map(|leaf| leaf.actor).collect(),
                            legals: request
                                .leaves
                                .iter()
                                .map(|leaf| leaf.legal.clone())
                                .collect(),
                            request,
                            states,
                        });
                    }
                    Ok(tree_resumable::SearchEvent::Complete) => {
                        let session = sessions[slot].take().expect("session must exist");
                        let (result, arena, metrics) = session.into_result()?;
                        let mut digest = Vec::new();
                        tree_resumable::digest(&arena, &mut digest);
                        completed[slot] = Some((result, metrics, digest));
                    }
                    Err(err) => {
                        session.cancel_pending();
                        return Err(err);
                    }
                }
            }
            while !groups.is_empty() {
                let mut batch = Vec::new();
                let mut rows = 0;
                while let Some(group) = groups.front() {
                    let count = group.states.len();
                    if count > global_batch_cap {
                        for session in sessions.iter_mut().flatten() {
                            session.cancel_pending();
                        }
                        return Err(PyValueError::new_err(format!(
                            "search leaf wave has {count} rows above global cap {global_batch_cap}"
                        )));
                    }
                    if !batch.is_empty() && rows + count > global_batch_cap {
                        break;
                    }
                    rows += count;
                    batch.push(groups.pop_front().expect("front group must exist"));
                }
                let owned: Vec<_> = batch
                    .iter()
                    .flat_map(|group| group.states.iter().cloned())
                    .collect();
                let actors: Vec<_> = batch
                    .iter()
                    .flat_map(|group| group.actors.iter().copied())
                    .collect();
                let legals: Vec<_> = batch
                    .iter()
                    .flat_map(|group| group.legals.iter().cloned())
                    .collect();
                let evaluations = match worker
                    .submit_prepared(owned, actors, legals)
                    .and_then(|ticket| ticket.wait())
                {
                    Ok(rows) => rows,
                    Err(err) => {
                        for session in sessions.iter_mut().flatten() {
                            session.cancel_pending();
                        }
                        return Err(err);
                    }
                };
                let mut cursor = 0;
                for group in batch {
                    let count = group.states.len();
                    let result = sessions[group.slot]
                        .as_mut()
                        .expect("search session must exist")
                        .apply_evaluations(
                            group.request.request_id,
                            evaluations[cursor..cursor + count].to_vec(),
                        );
                    cursor += count;
                    if let Err(err) = result {
                        for session in sessions.iter_mut().flatten() {
                            session.cancel_pending();
                        }
                        return Err(err);
                    }
                }
            }
        }
        drop(worker);
        if timed_out.load(std::sync::atomic::Ordering::Acquire) {
            drop(worker_handle);
        } else if worker_handle.join().is_err() {
            return Err(PyRuntimeError::new_err(
                "flat search inference worker panicked during shutdown",
            ));
        }
        Ok(completed
            .into_iter()
            .map(|row| row.expect("all searches must complete"))
            .collect::<Vec<_>>())
    })?;
    outputs
        .into_iter()
        .map(|(result, metrics, digest)| search_result_to_py(py, result, metrics, digest))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn cooperative_jobs(
    py: Python<'_>,
    games: &[Py<RustGame>],
    game_seeds: &[u64],
    iteration: Option<i64>,
    leaf_batch: usize,
    cheap_sims_min: usize,
    cheap_sims_max: usize,
    full_sims_min: usize,
    full_sims_max: usize,
    full_search_fraction: f64,
    top_k: usize,
    draft_prior: f64,
    c_puct: f64,
    c_visit: f64,
    c_scale: f64,
    force: bool,
    age_deal_samples: usize,
    max_moves: usize,
) -> PyResult<Vec<(GameState, self_play::SelfPlayConfig)>> {
    if games.len() != game_seeds.len() {
        return Err(PyValueError::new_err(format!(
            "received {} games but {} game seeds",
            games.len(),
            game_seeds.len()
        )));
    }
    games
        .iter()
        .zip(game_seeds)
        .map(|(game, &game_seed)| {
            let state = game.borrow(py).state.clone();
            let cfg = make_self_play_config(
                game_seed,
                iteration,
                leaf_batch,
                cheap_sims_min,
                cheap_sims_max,
                full_sims_min,
                full_sims_max,
                full_search_fraction,
                top_k,
                draft_prior,
                c_puct,
                c_visit,
                c_scale,
                force,
                age_deal_samples,
                max_moves,
            );
            Ok((state, cfg))
        })
        .collect()
}

/// F4.4 deterministic cooperative scheduler under the mock evaluator.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    games, game_seeds, global_batch_cap, leaf_batch, cheap_sims_min,
    cheap_sims_max, full_sims_min, full_sims_max, full_search_fraction, top_k,
    draft_prior, iteration=None, c_puct=1.5, c_visit=50.0, c_scale=1.0,
    force=false, age_deal_samples=0, max_moves=256
))]
fn self_play_many_mock(
    py: Python<'_>,
    games: Vec<Py<RustGame>>,
    game_seeds: Vec<u64>,
    global_batch_cap: usize,
    leaf_batch: usize,
    cheap_sims_min: usize,
    cheap_sims_max: usize,
    full_sims_min: usize,
    full_sims_max: usize,
    full_search_fraction: f64,
    top_k: usize,
    draft_prior: f64,
    iteration: Option<i64>,
    c_puct: f64,
    c_visit: f64,
    c_scale: f64,
    force: bool,
    age_deal_samples: usize,
    max_moves: usize,
) -> PyResult<(Vec<Py<PyDict>>, Py<PyDict>)> {
    let jobs = cooperative_jobs(
        py,
        &games,
        &game_seeds,
        iteration,
        leaf_batch,
        cheap_sims_min,
        cheap_sims_max,
        full_sims_min,
        full_sims_max,
        full_search_fraction,
        top_k,
        draft_prior,
        c_puct,
        c_visit,
        c_scale,
        force,
        age_deal_samples,
        max_moves,
    )?;
    let result = py.detach(move || self_play::run_many(jobs, &eval::MockEval, global_batch_cap))?;
    scheduler_result_to_py(py, result)
}

/// F4.4 global Python evaluator boundary. `adapter(rows)` is called once per
/// global batch; each row is `(tokens, actor, legal)` and results stay aligned.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    adapter, games, game_seeds, global_batch_cap, leaf_batch, cheap_sims_min,
    cheap_sims_max, full_sims_min, full_sims_max, full_search_fraction, top_k,
    draft_prior, iteration=None, c_puct=1.5, c_visit=50.0, c_scale=1.0,
    force=false, age_deal_samples=0, max_moves=256, inference_timeout_ms=0.0,
    max_inflight_batches=2, scheduler_workers=1, leaf_batch_p0=None, leaf_batch_p1=None,
    age_deal_samples_p0=None, age_deal_samples_p1=None, deterministic_actions=false
))]
fn self_play_many_net(
    py: Python<'_>,
    adapter: Py<PyAny>,
    games: Vec<Py<RustGame>>,
    game_seeds: Vec<u64>,
    global_batch_cap: usize,
    leaf_batch: usize,
    cheap_sims_min: usize,
    cheap_sims_max: usize,
    full_sims_min: usize,
    full_sims_max: usize,
    full_search_fraction: f64,
    top_k: usize,
    draft_prior: f64,
    iteration: Option<i64>,
    c_puct: f64,
    c_visit: f64,
    c_scale: f64,
    force: bool,
    age_deal_samples: usize,
    max_moves: usize,
    inference_timeout_ms: f64,
    max_inflight_batches: usize,
    scheduler_workers: usize,
    leaf_batch_p0: Option<usize>,
    leaf_batch_p1: Option<usize>,
    age_deal_samples_p0: Option<usize>,
    age_deal_samples_p1: Option<usize>,
    deterministic_actions: bool,
) -> PyResult<(Vec<Py<PyDict>>, Py<PyDict>)> {
    let mut jobs = cooperative_jobs(
        py,
        &games,
        &game_seeds,
        iteration,
        leaf_batch,
        cheap_sims_min,
        cheap_sims_max,
        full_sims_min,
        full_sims_max,
        full_search_fraction,
        top_k,
        draft_prior,
        c_puct,
        c_visit,
        c_scale,
        force,
        age_deal_samples,
        max_moves,
    )?;
    match (leaf_batch_p0, leaf_batch_p1) {
        (None, None) => {}
        (Some(p0), Some(p1)) if p0 > 0 && p1 > 0 => {
            for (_, cfg) in &mut jobs {
                cfg.leaf_batch_by_player = Some([p0, p1]);
                cfg.deterministic_actions = deterministic_actions;
            }
        }
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "leaf_batch_p0 and leaf_batch_p1 must be positive",
            ));
        }
        _ => {
            return Err(PyValueError::new_err(
                "leaf_batch_p0 and leaf_batch_p1 must be supplied together",
            ));
        }
    }
    if leaf_batch_p0.is_none() {
        for (_, cfg) in &mut jobs {
            cfg.deterministic_actions = deterministic_actions;
        }
    }
    match (age_deal_samples_p0, age_deal_samples_p1) {
        (None, None) => {}
        (Some(p0), Some(p1)) if p0 <= 32 && p1 <= 32 => {
            for (_, cfg) in &mut jobs {
                cfg.age_deal_samples_by_player = Some([p0, p1]);
            }
        }
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "age_deal_samples_p0 and age_deal_samples_p1 cannot exceed 32",
            ));
        }
        _ => {
            return Err(PyValueError::new_err(
                "age_deal_samples_p0 and age_deal_samples_p1 must be supplied together",
            ));
        }
    }
    let (worker, timed_out, worker_handle) =
        eval::spawn_py_batch_worker(adapter, inference_timeout_ms, global_batch_cap)?;
    let result = py.detach(move || {
        let result = self_play::run_many_pipelined_sharded(
            jobs,
            &worker,
            global_batch_cap,
            max_inflight_batches,
            scheduler_workers,
        );
        drop(worker);
        if timed_out.load(std::sync::atomic::Ordering::Acquire) {
            // Rust cannot safely kill a Python/Torch call. Detach the timed-out
            // worker so every scheduler slot wakes immediately; the worker owns
            // no slot/search state and exits when the adapter call returns.
            drop(worker_handle);
            return result;
        }
        if worker_handle.join().is_err() {
            return Err(PyRuntimeError::new_err(
                "global inference worker panicked during shutdown",
            ));
        }
        result
    })?;
    scheduler_result_to_py(py, result)
}

/// F4.5 production-shaped flat transformer boundary. The adapter receives one
/// dictionary of packed byte buffers and returns only actor value plus priors
/// aligned to the packed legal-action rows.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    adapter, games, game_seeds, global_batch_cap, leaf_batch, cheap_sims_min,
    cheap_sims_max, full_sims_min, full_sims_max, full_search_fraction, top_k,
    draft_prior, iteration=None, c_puct=1.5, c_visit=50.0, c_scale=1.0,
    force=false, age_deal_samples=0, max_moves=256, inference_timeout_ms=0.0,
    max_inflight_batches=2, scheduler_workers=1, leaf_batch_p0=None, leaf_batch_p1=None,
    age_deal_samples_p0=None, age_deal_samples_p1=None, deterministic_actions=false,
    bot_p0=None, bot_p1=None, bot_exploration=0.0, bot_policy_iterations=10
))]
fn self_play_many_flat_net(
    py: Python<'_>,
    adapter: Py<PyAny>,
    games: Vec<Py<RustGame>>,
    game_seeds: Vec<u64>,
    global_batch_cap: usize,
    leaf_batch: usize,
    cheap_sims_min: usize,
    cheap_sims_max: usize,
    full_sims_min: usize,
    full_sims_max: usize,
    full_search_fraction: f64,
    top_k: usize,
    draft_prior: f64,
    iteration: Option<i64>,
    c_puct: f64,
    c_visit: f64,
    c_scale: f64,
    force: bool,
    age_deal_samples: usize,
    max_moves: usize,
    inference_timeout_ms: f64,
    max_inflight_batches: usize,
    scheduler_workers: usize,
    leaf_batch_p0: Option<usize>,
    leaf_batch_p1: Option<usize>,
    age_deal_samples_p0: Option<usize>,
    age_deal_samples_p1: Option<usize>,
    deterministic_actions: bool,
    bot_p0: Option<String>,
    bot_p1: Option<String>,
    bot_exploration: f64,
    bot_policy_iterations: i64,
) -> PyResult<(Vec<Py<PyDict>>, Py<PyDict>)> {
    let mut jobs = cooperative_jobs(
        py,
        &games,
        &game_seeds,
        iteration,
        leaf_batch,
        cheap_sims_min,
        cheap_sims_max,
        full_sims_min,
        full_sims_max,
        full_search_fraction,
        top_k,
        draft_prior,
        c_puct,
        c_visit,
        c_scale,
        force,
        age_deal_samples,
        max_moves,
    )?;
    match (leaf_batch_p0, leaf_batch_p1) {
        (None, None) => {}
        (Some(p0), Some(p1)) if p0 > 0 && p1 > 0 => {
            for (_, cfg) in &mut jobs {
                cfg.leaf_batch_by_player = Some([p0, p1]);
            }
        }
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "leaf_batch_p0 and leaf_batch_p1 must be positive",
            ));
        }
        _ => {
            return Err(PyValueError::new_err(
                "leaf_batch_p0 and leaf_batch_p1 must be supplied together",
            ));
        }
    }
    let parse_bot = |name: Option<&str>| -> PyResult<Option<bots::BotKind>> {
        name.map(|value| {
            bots::BotKind::parse(value)
                .ok_or_else(|| PyValueError::new_err(format!("unknown Rust bot: {value}")))
        })
        .transpose()
    };
    let bot_by_player = [parse_bot(bot_p0.as_deref())?, parse_bot(bot_p1.as_deref())?];
    for (_, cfg) in &mut jobs {
        cfg.deterministic_actions = deterministic_actions;
        cfg.bot_by_player = bot_by_player;
        cfg.bot_exploration = bot_exploration;
        cfg.bot_policy_iterations = bot_policy_iterations;
    }
    match (age_deal_samples_p0, age_deal_samples_p1) {
        (None, None) => {}
        (Some(p0), Some(p1)) if p0 <= 32 && p1 <= 32 => {
            for (_, cfg) in &mut jobs {
                cfg.age_deal_samples_by_player = Some([p0, p1]);
            }
        }
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "age_deal_samples_p0 and age_deal_samples_p1 cannot exceed 32",
            ));
        }
        _ => {
            return Err(PyValueError::new_err(
                "age_deal_samples_p0 and age_deal_samples_p1 must be supplied together",
            ));
        }
    }
    let (worker, timed_out, boundary_metrics, worker_handle) =
        eval::spawn_py_flat_worker(adapter, inference_timeout_ms, global_batch_cap)?;
    let result = py.detach(move || {
        let mut result = self_play::run_many_pipelined_sharded(
            jobs,
            &worker,
            global_batch_cap,
            max_inflight_batches,
            scheduler_workers,
        );
        drop(worker);
        if timed_out.load(std::sync::atomic::Ordering::Acquire) {
            drop(worker_handle);
            return result;
        }
        if worker_handle.join().is_err() {
            return Err(PyRuntimeError::new_err(
                "flat inference worker panicked during shutdown",
            ));
        }
        if let Ok(output) = &mut result {
            let counters = boundary_metrics
                .lock()
                .map_err(|_| PyRuntimeError::new_err("boundary metrics lock poisoned"))?
                .clone();
            output.metrics.boundary_tokens = counters.tokens;
            output.metrics.boundary_padded_tokens = counters.padded_tokens;
            output.metrics.boundary_max_tokens = counters.max_tokens;
            output.metrics.encode_pack_ns = counters.encode_pack_ns;
            output.metrics.queue_wait_ns = counters.queue_wait_ns;
            output.metrics.py_call_ns = counters.py_call_ns;
            output.metrics.extract_ns = counters.extract_ns;
        }
        result
    })?;
    scheduler_result_to_py(py, result)
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

    #[pymodule_export]
    use super::ln_values;

    #[pymodule_export]
    use super::self_play_many_mock;

    #[pymodule_export]
    use super::self_play_many_net;

    #[pymodule_export]
    use super::self_play_many_flat_net;

    #[pymodule_export]
    use super::search_many_flat_net;
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
