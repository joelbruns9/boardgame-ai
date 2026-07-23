//! F4.3 single-slot full-game self-play.
//!
//! The complete move loop lives here: search-size scheduling, search seeds,
//! draft-prior blending, temperature sampling, action/chance application, and
//! move/result recording.  Python is entered only by `PyEval` while a search
//! needs neural evaluations; it does not regain control between moves.

use crate::bots::{self, BotKind};
use crate::chance::{self, ChanceKind};
use crate::codec::{decode_action, legal_action_indices};
use crate::data::{self, wonder};
use crate::eval::{Eval, EvalTicket, EvalWorker};
use crate::rng::Rng;
use crate::state::{GameState, Phase, VictoryType};
use crate::tree::SearchConfig;
use crate::tree_resumable;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::time::Instant;

const GAME_RNG_XOR: u64 = 0xC6BC_2796_92B5_CC83;

#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    pub game_seed: u64,
    pub iteration: Option<i64>,
    pub leaf_batch: usize,
    pub leaf_batch_by_player: Option<[usize; 2]>,
    pub deterministic_actions: bool,
    pub cheap_sims_min: usize,
    pub cheap_sims_max: usize,
    pub full_sims_min: usize,
    pub full_sims_max: usize,
    pub full_search_fraction: f64,
    pub top_k: usize,
    pub draft_prior: f64,
    pub c_puct: f64,
    pub c_visit: f64,
    pub c_scale: f64,
    pub force_expand_root_chance: bool,
    pub age_deal_samples: usize,
    pub age_deal_samples_by_player: Option<[usize; 2]>,
    pub bot_by_player: [Option<BotKind>; 2],
    pub bot_exploration: f64,
    pub bot_policy_iterations: i64,
    pub max_moves: usize,
}

impl SelfPlayConfig {
    pub fn validate(&self) -> PyResult<()> {
        if self.leaf_batch == 0
            || self
                .leaf_batch_by_player
                .is_some_and(|batches| batches.contains(&0))
            || self.top_k == 0
            || self.max_moves == 0
        {
            return Err(PyValueError::new_err(
                "leaf_batch, top_k, and max_moves must be positive",
            ));
        }
        if self.age_deal_samples > 32
            || self
                .age_deal_samples_by_player
                .is_some_and(|samples| samples.iter().any(|&count| count > 32))
        {
            return Err(PyValueError::new_err(
                "AgeDeal samples cannot exceed the paired-32 diagnostic reference",
            ));
        }
        if self.cheap_sims_min == 0
            || self.cheap_sims_min > self.cheap_sims_max
            || self.full_sims_min == 0
            || self.full_sims_min > self.full_sims_max
        {
            return Err(PyValueError::new_err("invalid self-play simulation range"));
        }
        if !(0.0..=1.0).contains(&self.full_search_fraction)
            || !(0.0..=1.0).contains(&self.draft_prior)
            || !(0.0..=1.0).contains(&self.bot_exploration)
        {
            return Err(PyValueError::new_err(
                "full_search_fraction, draft_prior, and bot_exploration must be in [0, 1]",
            ));
        }
        for (name, value) in [
            ("c_puct", self.c_puct),
            ("c_visit", self.c_visit),
            ("c_scale", self.c_scale),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct MoveRecord {
    pub i: usize,
    pub actor: usize,
    pub action: usize,
    pub legal: Vec<usize>,
    pub visits: Vec<u32>,
    pub policy_target: Vec<f64>,
    pub root_value: f64,
    pub sims: usize,
    pub gumbel_topk: Vec<usize>,
    pub policy_excluded: bool,
    pub full_search: bool,
    pub search_seed: u64,
    pub is_bot: bool,
}

#[derive(Clone, Debug)]
pub struct ChanceRecord {
    pub move_index: usize,
    pub kind: ChanceKind,
    pub outcome: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct GameRecord {
    pub seed: u64,
    pub first_player: usize,
    pub iteration: Option<i64>,
    pub winner: Option<usize>,
    pub victory_type: Option<VictoryType>,
    pub scores: Option<(i32, i32)>,
    pub moves: Vec<MoveRecord>,
    pub chance_log: Vec<ChanceRecord>,
    pub final_fingerprint: Vec<i32>,
    pub agent_names: [String; 2],
}

fn agent_names(cfg: &SelfPlayConfig) -> [String; 2] {
    std::array::from_fn(|seat| {
        cfg.bot_by_player[seat].map_or_else(|| "network".to_owned(), |bot| bot.name().to_owned())
    })
}

fn normalize(mut weights: Vec<f64>) -> Vec<f64> {
    if weights.is_empty() {
        return weights;
    }
    for value in &mut weights {
        *value = value.max(0.0);
    }
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        let uniform = 1.0 / weights.len() as f64;
        return vec![uniform; weights.len()];
    }
    weights.into_iter().map(|value| value / total).collect()
}

fn wonder_draft_tier(wonder_id: usize) -> f64 {
    match wonder(wonder_id).name {
        "The Temple of Artemis"
        | "Piraeus"
        | "The Hanging Gardens"
        | "The Appian Way"
        | "The Sphinx" => 1.0,
        "The Statue of Zeus" | "The Great Library" => 0.8,
        "The Mausoleum" | "Circus Maximus" | "The Colossus" => 0.6,
        "The Great Lighthouse" => 0.4,
        "The Pyramids" => 0.0,
        name => panic!("missing Phase-D Wonder draft tier for {name}"),
    }
}

pub(crate) fn blend_priors(state: &GameState, priors: Vec<f64>, amount: f64) -> Vec<f64> {
    let neural = normalize(priors);
    if state.phase != Phase::WonderDraft || amount <= 0.0 {
        return neural;
    }
    let legal = legal_action_indices(state);
    let logits: Vec<f64> = legal
        .iter()
        .map(|&index| {
            let action = decode_action(state, index);
            wonder_draft_tier(action.wonder.expect("draft action missing wonder"))
        })
        .collect();
    let peak = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let tier = normalize(logits.into_iter().map(|x| (x - peak).exp()).collect());
    normalize(
        neural
            .iter()
            .zip(tier)
            .map(|(&n, t)| (1.0 - amount) * n + amount * t)
            .collect(),
    )
}

struct DraftPriorEval<'a, E> {
    base: &'a E,
    amount: f64,
}

impl<E: Eval> Eval for DraftPriorEval<'_, E> {
    fn evaluate(&self, state: &GameState) -> PyResult<(f64, Vec<f64>)> {
        let (value, priors) = self.base.evaluate(state)?;
        Ok((value, blend_priors(state, priors, self.amount)))
    }

    fn evaluate_batch(&self, states: &[&GameState]) -> PyResult<Vec<(f64, Vec<f64>)>> {
        let rows = self.base.evaluate_batch(states)?;
        if rows.len() != states.len() {
            return Err(PyValueError::new_err(format!(
                "evaluator returned {} rows for {} states",
                rows.len(),
                states.len()
            )));
        }
        Ok(rows
            .into_iter()
            .zip(states)
            .map(|((value, priors), state)| (value, blend_priors(state, priors, self.amount)))
            .collect())
    }

    fn evaluate_batch_prepared(
        &self,
        states: &[&GameState],
        actors: &[usize],
        legals: &[Vec<usize>],
    ) -> PyResult<Vec<(f64, Vec<f64>)>> {
        let rows = self.base.evaluate_batch_prepared(states, actors, legals)?;
        if rows.len() != states.len() {
            return Err(PyValueError::new_err(format!(
                "evaluator returned {} rows for {} states",
                rows.len(),
                states.len()
            )));
        }
        Ok(rows
            .into_iter()
            .zip(states)
            .map(|((value, priors), state)| (value, blend_priors(state, priors, self.amount)))
            .collect())
    }
}

fn actual_chance_outcomes(
    state: &GameState,
    action_index: usize,
    move_index: usize,
) -> PyResult<Vec<ChanceRecord>> {
    let action = decode_action(state, action_index);
    chance::chance_signature(state, &action)
        .into_iter()
        .map(|spec| {
            let outcome = match spec.kind {
                ChanceKind::CardReveal => {
                    let slot = state
                        .tableau
                        .slot_index_of(spec.context[0], spec.context[1])
                        .ok_or_else(|| PyRuntimeError::new_err("chance reveal slot missing"))?;
                    vec![state.tableau.slots[slot].card_id]
                }
                ChanceKind::GreatLibraryDraw => {
                    let mut outcome = state.library_draws.front().cloned().ok_or_else(|| {
                        PyValueError::new_err("Great Library requires a pre-locked simulator draw")
                    })?;
                    outcome.sort_unstable();
                    outcome
                }
                ChanceKind::WonderGroupReveal => {
                    let mut outcome = state.wonder_groups[1].clone();
                    outcome.sort_unstable();
                    outcome
                }
                ChanceKind::AgeDeal => state.age_decks[spec.context[0] as usize].clone(),
            };
            Ok(ChanceRecord {
                move_index,
                kind: spec.kind,
                outcome,
            })
        })
        .collect()
}

fn temperature(move_index: usize) -> f64 {
    let progress = (move_index as f64 / 20.0).min(1.0);
    1.0 + (0.25 - 1.0) * progress
}

fn sample_policy(legal: &[usize], policy: &[f64], temp: f64, rng: &mut Rng) -> usize {
    assert_eq!(legal.len(), policy.len());
    let power = 1.0 / temp;
    let weights: Vec<f64> = policy.iter().map(|&p| p.max(1e-12).powf(power)).collect();
    let total: f64 = weights.iter().sum();
    let target = rng.next_float() * total;
    let mut cumulative = 0.0;
    for (&action, weight) in legal.iter().zip(weights) {
        cumulative += weight;
        if target < cumulative {
            return action;
        }
    }
    *legal
        .last()
        .expect("search root cannot have no legal actions")
}

fn random_sims(rng: &mut Rng, min: usize, max: usize) -> usize {
    min + rng.randrange((max - min + 1) as u64) as usize
}

pub fn run<E: Eval>(
    initial: &GameState,
    evaluator: &E,
    cfg: &SelfPlayConfig,
) -> PyResult<GameRecord> {
    cfg.validate()?;
    if initial.phase == Phase::Complete {
        return Err(PyValueError::new_err("cannot self-play a completed game"));
    }
    let mut state = initial.clone();
    let mut rng = Rng::new(cfg.game_seed ^ GAME_RNG_XOR);
    let eval = DraftPriorEval {
        base: evaluator,
        amount: cfg.draft_prior,
    };
    let mut moves = Vec::new();
    let mut chance_log = Vec::new();

    while state.phase != Phase::Complete {
        let i = moves.len();
        if i >= cfg.max_moves {
            return Err(PyRuntimeError::new_err(format!(
                "self-play exceeded max_moves={} without completing",
                cfg.max_moves
            )));
        }
        let actor = crate::tree::state_actor(&state);
        let legal = legal_action_indices(&state);
        let full = rng.next_float() < cfg.full_search_fraction;
        let sims = if full {
            random_sims(&mut rng, cfg.full_sims_min, cfg.full_sims_max)
        } else {
            random_sims(&mut rng, cfg.cheap_sims_min, cfg.cheap_sims_max)
        };
        let search_seed = rng.next_u64() & ((1_u64 << 63) - 1);
        let search_cfg = SearchConfig {
            sims,
            top_k: cfg.top_k,
            c_puct: cfg.c_puct,
            c_visit: cfg.c_visit,
            c_scale: cfg.c_scale,
            seed: search_seed,
            force_expand_root_chance: cfg.force_expand_root_chance,
            age_deal_samples: cfg.age_deal_samples,
        };
        let leaf_batch = cfg
            .leaf_batch_by_player
            .map_or(cfg.leaf_batch, |batches| batches[actor]);
        let (result, _, _) =
            tree_resumable::search_closed_batched(&state, &eval, &search_cfg, leaf_batch)?;
        let action = if cfg.deterministic_actions {
            result.action_index
        } else {
            sample_policy(&legal, &result.policy_target, temperature(i), &mut rng)
        };
        chance_log.extend(actual_chance_outcomes(&state, action, i)?);
        state.apply_action(&decode_action(&state, action));
        moves.push(MoveRecord {
            i,
            actor,
            action,
            legal,
            visits: result.visits,
            policy_target: result.policy_target,
            root_value: result.root_value,
            sims: result.sims,
            gumbel_topk: result.gumbel_topk,
            policy_excluded: !full,
            full_search: full,
            search_seed,
            is_bot: false,
        });
    }

    Ok(GameRecord {
        seed: cfg.game_seed,
        first_player: initial.first_player,
        iteration: cfg.iteration,
        winner: state.winner,
        victory_type: state.victory_type,
        scores: state.final_scores,
        moves,
        chance_log,
        final_fingerprint: state.fingerprint(),
        agent_names: agent_names(&cfg),
    })
}

#[derive(Clone, Debug, Default)]
pub struct SchedulerMetrics {
    pub games: usize,
    pub moves: usize,
    pub simulations: usize,
    pub requested_nn_leaves: usize,
    pub unique_nn_leaves: usize,
    pub terminal_leaves: usize,
    pub collisions: usize,
    pub global_batches: usize,
    pub global_rows: usize,
    pub root_rows: usize,
    /// All non-root rows (forced + ordinary), retained for schema compatibility.
    pub leaf_rows: usize,
    pub forced_rows: usize,
    pub forced_rows_by_kind: [usize; 4],
    pub ordinary_leaf_rows: usize,
    pub forced_cache_hits: usize,
    pub forced_rows_per_search: Vec<usize>,
    pub max_batch_rows: usize,
    pub scheduler_cycles: usize,
    pub scheduler_workers: usize,
    pub max_inflight_batches: usize,
    pub batch_rows: Vec<usize>,
    pub boundary_tokens: usize,
    pub boundary_padded_tokens: usize,
    pub boundary_max_tokens: usize,
    pub encode_pack_ns: u64,
    pub queue_wait_ns: u64,
    pub py_call_ns: u64,
    pub extract_ns: u64,
    pub rust_tree_ns: u64,
    pub rust_chance_ns: u64,
    pub rust_record_ns: u64,
    pub scatter_ns: u64,
    pub scheduler_ready_slot_cycles: u64,
    pub scheduler_waiting_slot_cycles: u64,
    pub scheduler_idle_slot_cycles: u64,
}

pub struct SchedulerResult {
    pub records: Vec<GameRecord>,
    pub metrics: SchedulerMetrics,
}

impl SchedulerMetrics {
    fn merge(&mut self, other: SchedulerMetrics) {
        self.games += other.games;
        self.moves += other.moves;
        self.simulations += other.simulations;
        self.requested_nn_leaves += other.requested_nn_leaves;
        self.unique_nn_leaves += other.unique_nn_leaves;
        self.terminal_leaves += other.terminal_leaves;
        self.collisions += other.collisions;
        self.global_batches += other.global_batches;
        self.global_rows += other.global_rows;
        self.root_rows += other.root_rows;
        self.leaf_rows += other.leaf_rows;
        self.forced_rows += other.forced_rows;
        self.ordinary_leaf_rows += other.ordinary_leaf_rows;
        self.forced_cache_hits += other.forced_cache_hits;
        for kind in 0..4 {
            self.forced_rows_by_kind[kind] += other.forced_rows_by_kind[kind];
        }
        self.forced_rows_per_search
            .extend(other.forced_rows_per_search);
        self.max_batch_rows = self.max_batch_rows.max(other.max_batch_rows);
        self.scheduler_cycles += other.scheduler_cycles;
        self.scheduler_workers += other.scheduler_workers;
        self.max_inflight_batches = self.max_inflight_batches.max(other.max_inflight_batches);
        self.batch_rows.extend(other.batch_rows);
        self.boundary_tokens += other.boundary_tokens;
        self.boundary_padded_tokens += other.boundary_padded_tokens;
        self.boundary_max_tokens = self.boundary_max_tokens.max(other.boundary_max_tokens);
        self.encode_pack_ns += other.encode_pack_ns;
        self.queue_wait_ns += other.queue_wait_ns;
        self.py_call_ns += other.py_call_ns;
        self.extract_ns += other.extract_ns;
        self.rust_tree_ns += other.rust_tree_ns;
        self.rust_chance_ns += other.rust_chance_ns;
        self.rust_record_ns += other.rust_record_ns;
        self.scatter_ns += other.scatter_ns;
        self.scheduler_ready_slot_cycles += other.scheduler_ready_slot_cycles;
        self.scheduler_waiting_slot_cycles += other.scheduler_waiting_slot_cycles;
        self.scheduler_idle_slot_cycles += other.scheduler_idle_slot_cycles;
    }
}

struct SearchMeta {
    actor: usize,
    legal: Vec<usize>,
    leaf_batch: usize,
    full: bool,
    search_seed: u64,
    search_cfg: SearchConfig,
}

enum SlotStage {
    NeedRoot(SearchMeta),
    Searching {
        meta: SearchMeta,
        session: tree_resumable::SearchSession,
    },
    Complete,
}

struct GameSlot {
    state: GameState,
    rng: Rng,
    cfg: SelfPlayConfig,
    moves: Vec<MoveRecord>,
    chance_log: Vec<ChanceRecord>,
    stage: SlotStage,
    simulations: usize,
    requested_nn_leaves: usize,
    unique_nn_leaves: usize,
    terminal_leaves: usize,
    collisions: usize,
    forced_rows: usize,
    forced_rows_by_kind: [usize; 4],
    forced_cache_hits: usize,
    forced_rows_per_search: Vec<usize>,
    tree_ns: u64,
    chance_ns: u64,
    record_ns: u64,
    scatter_ns: u64,
    bot_rngs: [Rng; 2],
}

enum EvalGroupKind {
    Root,
    Forced(tree_resumable::EvalBatchRequest),
    Wave(tree_resumable::EvalBatchRequest),
}

struct EvalGroup {
    slot: usize,
    states: Vec<GameState>,
    actors: Vec<usize>,
    legals: Vec<Vec<usize>>,
    kind: EvalGroupKind,
}

impl GameSlot {
    fn new(initial: GameState, cfg: SelfPlayConfig) -> PyResult<Self> {
        cfg.validate()?;
        if initial.phase == Phase::Complete {
            return Err(PyValueError::new_err("cannot self-play a completed game"));
        }
        let bot_seed = cfg.game_seed ^ 0x51ED;
        let mut slot = Self {
            state: initial,
            rng: Rng::new(cfg.game_seed ^ GAME_RNG_XOR),
            cfg,
            moves: Vec::new(),
            chance_log: Vec::new(),
            stage: SlotStage::Complete,
            simulations: 0,
            requested_nn_leaves: 0,
            unique_nn_leaves: 0,
            terminal_leaves: 0,
            collisions: 0,
            forced_rows: 0,
            forced_rows_by_kind: [0; 4],
            forced_cache_hits: 0,
            forced_rows_per_search: Vec::new(),
            tree_ns: 0,
            chance_ns: 0,
            record_ns: 0,
            scatter_ns: 0,
            bot_rngs: [Rng::new(bot_seed), Rng::new(bot_seed)],
        };
        slot.stage = SlotStage::NeedRoot(slot.make_search_meta()?);
        Ok(slot)
    }

    fn make_search_meta(&mut self) -> PyResult<SearchMeta> {
        let move_index = self.moves.len();
        if move_index >= self.cfg.max_moves {
            return Err(PyRuntimeError::new_err(format!(
                "self-play exceeded max_moves={} without completing",
                self.cfg.max_moves
            )));
        }
        let actor = crate::tree::state_actor(&self.state);
        let legal = legal_action_indices(&self.state);
        let full = self.rng.next_float() < self.cfg.full_search_fraction;
        let sims = if full {
            random_sims(
                &mut self.rng,
                self.cfg.full_sims_min,
                self.cfg.full_sims_max,
            )
        } else {
            random_sims(
                &mut self.rng,
                self.cfg.cheap_sims_min,
                self.cfg.cheap_sims_max,
            )
        };
        let search_seed = self.rng.next_u64() & ((1_u64 << 63) - 1);
        Ok(SearchMeta {
            actor,
            legal,
            leaf_batch: self
                .cfg
                .leaf_batch_by_player
                .map_or(self.cfg.leaf_batch, |batches| batches[actor]),
            full,
            search_seed,
            search_cfg: SearchConfig {
                sims,
                top_k: self.cfg.top_k,
                c_puct: self.cfg.c_puct,
                c_visit: self.cfg.c_visit,
                c_scale: self.cfg.c_scale,
                seed: search_seed,
                force_expand_root_chance: self.cfg.force_expand_root_chance,
                age_deal_samples: self
                    .cfg
                    .age_deal_samples_by_player
                    .map_or(self.cfg.age_deal_samples, |samples| samples[actor]),
            },
        })
    }

    fn next_eval_group(
        &mut self,
        slot_index: usize,
        forced_row_limit: usize,
    ) -> PyResult<Option<EvalGroup>> {
        loop {
            if let SlotStage::NeedRoot(meta) = &self.stage {
                if let Some(kind) = self.cfg.bot_by_player[meta.actor] {
                    let old_stage = std::mem::replace(&mut self.stage, SlotStage::Complete);
                    let SlotStage::NeedRoot(meta) = old_stage else {
                        unreachable!()
                    };
                    self.finish_bot_move(meta, kind)?;
                    continue;
                }
            }
            match &mut self.stage {
                SlotStage::NeedRoot(meta) => {
                    return Ok(Some(EvalGroup {
                        slot: slot_index,
                        states: vec![self.state.clone()],
                        actors: vec![meta.actor],
                        legals: vec![meta.legal.clone()],
                        kind: EvalGroupKind::Root,
                    }));
                }
                SlotStage::Complete => return Ok(None),
                SlotStage::Searching { session, .. } => {
                    let started = Instant::now();
                    let event = session.next_event_with_limit(forced_row_limit);
                    self.tree_ns += started.elapsed().as_nanos() as u64;
                    match event? {
                        tree_resumable::SearchEvent::Evaluation(request) => {
                            let states = session
                                .evaluation_states(&request)?
                                .into_iter()
                                .cloned()
                                .collect();
                            return Ok(Some(EvalGroup {
                                slot: slot_index,
                                states,
                                actors: request.leaves.iter().map(|leaf| leaf.actor).collect(),
                                legals: request
                                    .leaves
                                    .iter()
                                    .map(|leaf| leaf.legal.clone())
                                    .collect(),
                                kind: if request.forced {
                                    EvalGroupKind::Forced(request)
                                } else {
                                    EvalGroupKind::Wave(request)
                                },
                            }));
                        }
                        tree_resumable::SearchEvent::Complete => {}
                    }
                }
            }

            let old_stage = std::mem::replace(&mut self.stage, SlotStage::Complete);
            let SlotStage::Searching { meta, session } = old_stage else {
                unreachable!("only a completed search reaches this branch")
            };
            let (result, _, metrics) = session.into_result()?;
            self.simulations += result.sims;
            self.requested_nn_leaves += metrics.requested_nn_leaves;
            self.unique_nn_leaves += metrics.unique_nn_leaves;
            self.terminal_leaves += metrics.terminal_leaves;
            self.collisions += metrics.collisions;
            self.forced_rows += metrics.forced_outcome_rows;
            for (target, value) in self
                .forced_rows_by_kind
                .iter_mut()
                .zip(metrics.forced_rows_by_kind)
            {
                *target += value;
            }
            self.forced_cache_hits += metrics.cached_forced_leaves;
            self.forced_rows_per_search
                .push(metrics.forced_outcome_rows);
            self.finish_move(meta, result)?;
        }
    }

    fn apply_root(&mut self, evaluation: (f64, Vec<f64>)) -> PyResult<()> {
        let timer = Instant::now();
        let old_stage = std::mem::replace(&mut self.stage, SlotStage::Complete);
        let SlotStage::NeedRoot(meta) = old_stage else {
            return Err(PyRuntimeError::new_err(
                "root evaluation delivered to a slot not waiting for its root",
            ));
        };
        let evaluation = (
            evaluation.0,
            blend_priors(&self.state, evaluation.1, self.cfg.draft_prior),
        );
        let started = if meta.search_cfg.force_expand_root_chance {
            tree_resumable::begin_search_from_root_forced(
                &self.state,
                &meta.search_cfg,
                meta.leaf_batch,
                evaluation,
            )
        } else {
            tree_resumable::begin_search_from_root(
                &self.state,
                &meta.search_cfg,
                meta.leaf_batch,
                evaluation,
            )
        };
        let result = match started {
            Ok(session) => {
                self.stage = SlotStage::Searching { meta, session };
                Ok(())
            }
            Err(err) => Err(err),
        };
        self.tree_ns += timer.elapsed().as_nanos() as u64;
        result
    }

    fn apply_wave(
        &mut self,
        request: tree_resumable::EvalBatchRequest,
        states: &[GameState],
        evaluations: Vec<(f64, Vec<f64>)>,
    ) -> PyResult<()> {
        let started = Instant::now();
        let SlotStage::Searching { session, .. } = &mut self.stage else {
            return Err(PyRuntimeError::new_err(
                "leaf evaluation delivered to a slot not waiting for leaves",
            ));
        };
        if states.len() != evaluations.len() {
            session.cancel_pending();
            return Err(PyValueError::new_err(
                "leaf evaluation/state alignment mismatch",
            ));
        }
        let rows = evaluations
            .into_iter()
            .zip(states)
            .map(|((value, priors), state)| {
                (value, blend_priors(state, priors, self.cfg.draft_prior))
            })
            .collect();
        let result = session.apply_evaluations(request.request_id, rows);
        self.scatter_ns += started.elapsed().as_nanos() as u64;
        result
    }

    fn finish_move(&mut self, meta: SearchMeta, result: crate::tree::SearchResult) -> PyResult<()> {
        let i = self.moves.len();
        let action = if self.cfg.deterministic_actions {
            result.action_index
        } else {
            sample_policy(
                &meta.legal,
                &result.policy_target,
                temperature(i),
                &mut self.rng,
            )
        };
        let chance_started = Instant::now();
        self.chance_log
            .extend(actual_chance_outcomes(&self.state, action, i)?);
        self.state.apply_action(&decode_action(&self.state, action));
        self.chance_ns += chance_started.elapsed().as_nanos() as u64;
        let record_started = Instant::now();
        self.moves.push(MoveRecord {
            i,
            actor: meta.actor,
            action,
            legal: meta.legal,
            visits: result.visits,
            policy_target: result.policy_target,
            root_value: result.root_value,
            sims: result.sims,
            gumbel_topk: result.gumbel_topk,
            policy_excluded: !meta.full,
            full_search: meta.full,
            search_seed: meta.search_seed,
            is_bot: false,
        });
        self.stage = if self.state.phase == Phase::Complete {
            SlotStage::Complete
        } else {
            SlotStage::NeedRoot(self.make_search_meta()?)
        };
        self.record_ns += record_started.elapsed().as_nanos() as u64;
        Ok(())
    }

    fn finish_bot_move(&mut self, meta: SearchMeta, kind: BotKind) -> PyResult<()> {
        let i = self.moves.len();
        let action = bots::select_action(
            &self.state,
            kind,
            &mut self.bot_rngs[meta.actor],
            self.cfg.bot_exploration,
        );
        let chance_started = Instant::now();
        self.chance_log
            .extend(actual_chance_outcomes(&self.state, action, i)?);
        self.state.apply_action(&decode_action(&self.state, action));
        self.chance_ns += chance_started.elapsed().as_nanos() as u64;
        self.moves.push(MoveRecord {
            i,
            actor: meta.actor,
            action,
            legal: meta.legal,
            visits: Vec::new(),
            policy_target: Vec::new(),
            root_value: 0.0,
            sims: 0,
            gumbel_topk: Vec::new(),
            policy_excluded: self
                .cfg
                .iteration
                .is_some_and(|iteration| iteration >= self.cfg.bot_policy_iterations),
            full_search: false,
            search_seed: 0,
            is_bot: true,
        });
        self.stage = if self.state.phase == Phase::Complete {
            SlotStage::Complete
        } else {
            SlotStage::NeedRoot(self.make_search_meta()?)
        };
        Ok(())
    }

    fn cancel_pending(&mut self) {
        if let SlotStage::Searching { session, .. } = &mut self.stage {
            session.cancel_pending();
        }
    }

    fn into_record(self) -> PyResult<GameRecord> {
        if self.state.phase != Phase::Complete || !matches!(self.stage, SlotStage::Complete) {
            return Err(PyRuntimeError::new_err(
                "scheduler attempted to emit an incomplete game",
            ));
        }
        Ok(GameRecord {
            seed: self.cfg.game_seed,
            first_player: self.state.first_player,
            iteration: self.cfg.iteration,
            winner: self.state.winner,
            victory_type: self.state.victory_type,
            scores: self.state.final_scores,
            moves: self.moves,
            chance_log: self.chance_log,
            final_fingerprint: self.state.fingerprint(),
            agent_names: agent_names(&self.cfg),
        })
    }
}

fn cancel_all(slots: &mut [GameSlot]) {
    for slot in slots {
        slot.cancel_pending();
    }
}

fn validate_leaf_batches_fit(
    jobs: &[(GameState, SelfPlayConfig)],
    global_batch_cap: usize,
) -> PyResult<()> {
    for (job_index, (_, cfg)) in jobs.iter().enumerate() {
        if cfg.leaf_batch > global_batch_cap {
            return Err(PyValueError::new_err(format!(
                "job {job_index} leaf_batch={} exceeds global_batch_cap={global_batch_cap}",
                cfg.leaf_batch
            )));
        }
        if let Some(batches) = cfg.leaf_batch_by_player {
            for (player, leaf_batch) in batches.into_iter().enumerate() {
                if leaf_batch > global_batch_cap {
                    return Err(PyValueError::new_err(format!(
                        "job {job_index} leaf_batch_p{player}={leaf_batch} exceeds global_batch_cap={global_batch_cap}"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// F4.4 deterministic cooperative scheduler. Every slot advances until it
/// yields one indivisible root/leaf evaluation group; groups are packed in slot
/// order up to `global_batch_cap`, evaluated once, and scattered before the next
/// scheduler cycle. Records are always returned in input order, independent of
/// the order in which games reach terminal states.
pub fn run_many<E: Eval>(
    jobs: Vec<(GameState, SelfPlayConfig)>,
    evaluator: &E,
    global_batch_cap: usize,
) -> PyResult<SchedulerResult> {
    if jobs.is_empty() {
        return Err(PyValueError::new_err(
            "cooperative self-play needs at least one game",
        ));
    }
    if global_batch_cap == 0 {
        return Err(PyValueError::new_err("global_batch_cap must be positive"));
    }
    validate_leaf_batches_fit(&jobs, global_batch_cap)?;
    let mut slots: Vec<GameSlot> = jobs
        .into_iter()
        .map(|(state, cfg)| GameSlot::new(state, cfg))
        .collect::<PyResult<_>>()?;
    let mut metrics = SchedulerMetrics {
        games: slots.len(),
        scheduler_workers: 1,
        ..SchedulerMetrics::default()
    };

    while slots
        .iter()
        .any(|slot| !matches!(slot.stage, SlotStage::Complete))
    {
        metrics.scheduler_cycles += 1;
        for slot in &slots {
            if matches!(slot.stage, SlotStage::Complete) {
                metrics.scheduler_idle_slot_cycles += 1;
            } else {
                metrics.scheduler_ready_slot_cycles += 1;
            }
        }
        let mut groups = Vec::new();
        let live = slots
            .iter()
            .filter(|slot| !matches!(slot.stage, SlotStage::Complete))
            .count()
            .max(1);
        let forced_row_limit = (global_batch_cap / live).max(1);
        for (slot_index, slot) in slots.iter_mut().enumerate() {
            match slot.next_eval_group(slot_index, forced_row_limit) {
                Ok(Some(group)) => groups.push(group),
                Ok(None) => {}
                Err(err) => {
                    cancel_all(&mut slots);
                    return Err(err);
                }
            }
        }
        if groups.is_empty() {
            if slots
                .iter()
                .all(|slot| matches!(slot.stage, SlotStage::Complete))
            {
                break;
            }
            cancel_all(&mut slots);
            return Err(PyRuntimeError::new_err(
                "cooperative scheduler made no progress with live slots",
            ));
        }

        let mut pending = std::collections::VecDeque::from(groups);
        while !pending.is_empty() {
            let mut batch = Vec::new();
            let mut row_count = 0;
            while let Some(group) = pending.front() {
                let group_rows = group.states.len();
                if group_rows > global_batch_cap {
                    cancel_all(&mut slots);
                    return Err(PyValueError::new_err(format!(
                        "evaluation group has {group_rows} rows, exceeding global_batch_cap={global_batch_cap}"
                    )));
                }
                if !batch.is_empty() && row_count + group_rows > global_batch_cap {
                    break;
                }
                row_count += group_rows;
                batch.push(pending.pop_front().expect("front group must exist"));
            }

            let owned_states: Vec<GameState> = batch
                .iter()
                .flat_map(|group| group.states.iter().cloned())
                .collect();
            let state_refs: Vec<&GameState> = owned_states.iter().collect();
            let actors: Vec<_> = batch
                .iter()
                .flat_map(|group| group.actors.iter().copied())
                .collect();
            let legals: Vec<_> = batch
                .iter()
                .flat_map(|group| group.legals.iter().cloned())
                .collect();
            let evaluations = match evaluator.evaluate_batch_prepared(&state_refs, &actors, &legals)
            {
                Ok(rows) => rows,
                Err(err) => {
                    cancel_all(&mut slots);
                    return Err(err);
                }
            };
            if evaluations.len() != row_count {
                cancel_all(&mut slots);
                return Err(PyValueError::new_err(format!(
                    "global evaluator returned {} rows for {row_count} states",
                    evaluations.len()
                )));
            }
            metrics.global_batches += 1;
            metrics.global_rows += row_count;
            metrics.max_batch_rows = metrics.max_batch_rows.max(row_count);
            metrics.batch_rows.push(row_count);

            let mut cursor = 0;
            for group in batch {
                let count = group.states.len();
                let rows = evaluations[cursor..cursor + count].to_vec();
                cursor += count;
                let result = match group.kind {
                    EvalGroupKind::Root => {
                        metrics.root_rows += count;
                        slots[group.slot].apply_root(rows.into_iter().next().expect("root row"))
                    }
                    EvalGroupKind::Forced(request) => {
                        metrics.leaf_rows += count;
                        slots[group.slot].apply_wave(request, &group.states, rows)
                    }
                    EvalGroupKind::Wave(request) => {
                        metrics.leaf_rows += count;
                        metrics.ordinary_leaf_rows += count;
                        slots[group.slot].apply_wave(request, &group.states, rows)
                    }
                };
                if let Err(err) = result {
                    cancel_all(&mut slots);
                    return Err(err);
                }
            }
        }
    }

    metrics.moves = slots.iter().map(|slot| slot.moves.len()).sum();
    metrics.simulations = slots.iter().map(|slot| slot.simulations).sum();
    metrics.requested_nn_leaves = slots.iter().map(|slot| slot.requested_nn_leaves).sum();
    metrics.unique_nn_leaves = slots.iter().map(|slot| slot.unique_nn_leaves).sum();
    metrics.terminal_leaves = slots.iter().map(|slot| slot.terminal_leaves).sum();
    metrics.collisions = slots.iter().map(|slot| slot.collisions).sum();
    metrics.forced_rows = slots.iter().map(|slot| slot.forced_rows).sum();
    for kind in 0..4 {
        metrics.forced_rows_by_kind[kind] = slots
            .iter()
            .map(|slot| slot.forced_rows_by_kind[kind])
            .sum();
    }
    metrics.forced_cache_hits = slots.iter().map(|slot| slot.forced_cache_hits).sum();
    metrics.forced_rows_per_search = slots
        .iter()
        .flat_map(|slot| slot.forced_rows_per_search.iter().copied())
        .collect();
    metrics.rust_tree_ns = slots.iter().map(|slot| slot.tree_ns).sum();
    metrics.rust_chance_ns = slots.iter().map(|slot| slot.chance_ns).sum();
    metrics.rust_record_ns = slots.iter().map(|slot| slot.record_ns).sum();
    metrics.scatter_ns = slots.iter().map(|slot| slot.scatter_ns).sum();
    let records = slots
        .into_iter()
        .map(GameSlot::into_record)
        .collect::<PyResult<_>>()?;
    Ok(SchedulerResult { records, metrics })
}

struct InflightBatch {
    groups: Vec<EvalGroup>,
    row_count: usize,
    ticket: EvalTicket,
}

fn collect_ready_groups(
    slots: &mut [GameSlot],
    outstanding: &mut [bool],
    pending: &mut std::collections::VecDeque<EvalGroup>,
    global_batch_cap: usize,
) -> PyResult<usize> {
    let mut collected = 0;
    let ready = slots
        .iter()
        .enumerate()
        .filter(|(slot, game)| !outstanding[*slot] && !matches!(game.stage, SlotStage::Complete))
        .count()
        .max(1);
    let forced_row_limit = (global_batch_cap / ready).max(1);
    for (slot_index, slot) in slots.iter_mut().enumerate() {
        if outstanding[slot_index] {
            continue;
        }
        if let Some(group) = slot.next_eval_group(slot_index, forced_row_limit)? {
            outstanding[slot_index] = true;
            pending.push_back(group);
            collected += 1;
        }
    }
    Ok(collected)
}

fn take_global_batch(
    pending: &mut std::collections::VecDeque<EvalGroup>,
    global_batch_cap: usize,
) -> PyResult<(Vec<EvalGroup>, usize)> {
    let mut batch = Vec::new();
    let mut row_count = 0;
    while let Some(group) = pending.front() {
        let group_rows = group.states.len();
        if group_rows > global_batch_cap {
            return Err(PyValueError::new_err(format!(
                "evaluation group has {group_rows} rows, exceeding global_batch_cap={global_batch_cap}"
            )));
        }
        if !batch.is_empty() && row_count + group_rows > global_batch_cap {
            break;
        }
        row_count += group_rows;
        batch.push(pending.pop_front().expect("front group must exist"));
    }
    Ok((batch, row_count))
}

/// F4.4 double-buffered network scheduler. At most `max_inflight_batches`
/// requests are owned by the dedicated inference worker. While batch N+1 is
/// executing, completed batch N is scattered and its newly-ready slots advance
/// on the Rust scheduler thread.
pub fn run_many_pipelined(
    jobs: Vec<(GameState, SelfPlayConfig)>,
    worker: &EvalWorker,
    global_batch_cap: usize,
    max_inflight_batches: usize,
) -> PyResult<SchedulerResult> {
    if jobs.is_empty() {
        return Err(PyValueError::new_err(
            "cooperative self-play needs at least one game",
        ));
    }
    if global_batch_cap == 0 || max_inflight_batches == 0 {
        return Err(PyValueError::new_err(
            "global_batch_cap and max_inflight_batches must be positive",
        ));
    }
    validate_leaf_batches_fit(&jobs, global_batch_cap)?;
    let mut slots: Vec<GameSlot> = jobs
        .into_iter()
        .map(|(state, cfg)| GameSlot::new(state, cfg))
        .collect::<PyResult<_>>()?;
    let mut outstanding = vec![false; slots.len()];
    let mut pending = std::collections::VecDeque::new();
    let mut inflight = std::collections::VecDeque::<InflightBatch>::new();
    let mut metrics = SchedulerMetrics {
        games: slots.len(),
        scheduler_workers: 1,
        ..SchedulerMetrics::default()
    };

    loop {
        metrics.scheduler_cycles += 1;
        for (slot_index, slot) in slots.iter().enumerate() {
            if matches!(slot.stage, SlotStage::Complete) {
                metrics.scheduler_idle_slot_cycles += 1;
            } else if outstanding[slot_index] {
                metrics.scheduler_waiting_slot_cycles += 1;
            } else {
                metrics.scheduler_ready_slot_cycles += 1;
            }
        }
        if let Err(err) =
            collect_ready_groups(&mut slots, &mut outstanding, &mut pending, global_batch_cap)
        {
            cancel_all(&mut slots);
            return Err(err);
        }

        while inflight.len() < max_inflight_batches && !pending.is_empty() {
            let (groups, row_count) = match take_global_batch(&mut pending, global_batch_cap) {
                Ok(batch) => batch,
                Err(err) => {
                    cancel_all(&mut slots);
                    return Err(err);
                }
            };
            let owned_states = groups
                .iter()
                .flat_map(|group| group.states.iter().cloned())
                .collect();
            let actors = groups
                .iter()
                .flat_map(|group| group.actors.iter().copied())
                .collect();
            let legals = groups
                .iter()
                .flat_map(|group| group.legals.iter().cloned())
                .collect();
            let ticket = match worker.submit_prepared(owned_states, actors, legals) {
                Ok(ticket) => ticket,
                Err(err) => {
                    cancel_all(&mut slots);
                    return Err(err);
                }
            };
            metrics.global_batches += 1;
            metrics.global_rows += row_count;
            metrics.max_batch_rows = metrics.max_batch_rows.max(row_count);
            metrics.batch_rows.push(row_count);
            inflight.push_back(InflightBatch {
                groups,
                row_count,
                ticket,
            });
            metrics.max_inflight_batches = metrics.max_inflight_batches.max(inflight.len());
        }

        let Some(flight) = inflight.pop_front() else {
            if slots
                .iter()
                .all(|slot| matches!(slot.stage, SlotStage::Complete))
            {
                break;
            }
            cancel_all(&mut slots);
            return Err(PyRuntimeError::new_err(
                "pipelined scheduler made no progress with live slots",
            ));
        };
        let evaluations = match flight.ticket.wait() {
            Ok(rows) => rows,
            Err(err) => {
                cancel_all(&mut slots);
                return Err(err);
            }
        };
        if evaluations.len() != flight.row_count {
            cancel_all(&mut slots);
            return Err(PyValueError::new_err(format!(
                "global evaluator returned {} rows for {} states",
                evaluations.len(),
                flight.row_count
            )));
        }

        let mut cursor = 0;
        for group in flight.groups {
            let count = group.states.len();
            let rows = evaluations[cursor..cursor + count].to_vec();
            cursor += count;
            let result = match group.kind {
                EvalGroupKind::Root => {
                    metrics.root_rows += count;
                    slots[group.slot].apply_root(rows.into_iter().next().expect("root row"))
                }
                EvalGroupKind::Forced(request) => {
                    metrics.leaf_rows += count;
                    slots[group.slot].apply_wave(request, &group.states, rows)
                }
                EvalGroupKind::Wave(request) => {
                    metrics.leaf_rows += count;
                    metrics.ordinary_leaf_rows += count;
                    slots[group.slot].apply_wave(request, &group.states, rows)
                }
            };
            outstanding[group.slot] = false;
            if let Err(err) = result {
                cancel_all(&mut slots);
                return Err(err);
            }
        }
    }

    metrics.moves = slots.iter().map(|slot| slot.moves.len()).sum();
    metrics.simulations = slots.iter().map(|slot| slot.simulations).sum();
    metrics.requested_nn_leaves = slots.iter().map(|slot| slot.requested_nn_leaves).sum();
    metrics.unique_nn_leaves = slots.iter().map(|slot| slot.unique_nn_leaves).sum();
    metrics.terminal_leaves = slots.iter().map(|slot| slot.terminal_leaves).sum();
    metrics.collisions = slots.iter().map(|slot| slot.collisions).sum();
    metrics.forced_rows = slots.iter().map(|slot| slot.forced_rows).sum();
    for kind in 0..4 {
        metrics.forced_rows_by_kind[kind] = slots
            .iter()
            .map(|slot| slot.forced_rows_by_kind[kind])
            .sum();
    }
    metrics.forced_cache_hits = slots.iter().map(|slot| slot.forced_cache_hits).sum();
    metrics.forced_rows_per_search = slots
        .iter()
        .flat_map(|slot| slot.forced_rows_per_search.iter().copied())
        .collect();
    metrics.rust_tree_ns = slots.iter().map(|slot| slot.tree_ns).sum();
    metrics.rust_chance_ns = slots.iter().map(|slot| slot.chance_ns).sum();
    metrics.rust_record_ns = slots.iter().map(|slot| slot.record_ns).sum();
    metrics.scatter_ns = slots.iter().map(|slot| slot.scatter_ns).sum();
    let records = slots
        .into_iter()
        .map(GameSlot::into_record)
        .collect::<PyResult<_>>()?;
    Ok(SchedulerResult { records, metrics })
}

/// Coarse persistent scheduler shards. Each shard owns a contiguous set of
/// logical games and submits prepared batches to the same inference worker.
/// Records are joined in shard/input order, independent of completion order.
/// Search choices can still differ across shard counts when the evaluator is
/// sensitive to batch shape (notably through CUDA floating-point ties); only
/// result ordering, not cross-shard bit identity, is guaranteed.
pub fn run_many_pipelined_sharded(
    jobs: Vec<(GameState, SelfPlayConfig)>,
    worker: &EvalWorker,
    global_batch_cap: usize,
    max_inflight_batches: usize,
    scheduler_workers: usize,
) -> PyResult<SchedulerResult> {
    if scheduler_workers == 0 {
        return Err(PyValueError::new_err("scheduler_workers must be positive"));
    }
    if jobs.is_empty() {
        return Err(PyValueError::new_err(
            "cooperative self-play needs at least one game",
        ));
    }
    if global_batch_cap == 0 || max_inflight_batches == 0 {
        return Err(PyValueError::new_err(
            "global_batch_cap and max_inflight_batches must be positive",
        ));
    }
    validate_leaf_batches_fit(&jobs, global_batch_cap)?;
    if scheduler_workers == 1 || jobs.len() == 1 {
        return run_many_pipelined(jobs, worker, global_batch_cap, max_inflight_batches);
    }
    let shard_count = scheduler_workers.min(jobs.len());
    let chunk_size = (jobs.len() + shard_count - 1) / shard_count;
    let mut shards = Vec::new();
    let mut remaining = jobs.into_iter();
    loop {
        let chunk: Vec<_> = remaining.by_ref().take(chunk_size).collect();
        if chunk.is_empty() {
            break;
        }
        shards.push(chunk);
    }
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = shards
            .into_iter()
            .map(|shard| {
                scope.spawn(move || {
                    run_many_pipelined(shard, worker, global_batch_cap, max_inflight_batches)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| PyRuntimeError::new_err("scheduler worker shard panicked"))?
            })
            .collect::<PyResult<Vec<_>>>()
    })?;
    let mut records = Vec::new();
    let mut metrics = SchedulerMetrics::default();
    for result in results {
        records.extend(result.records);
        metrics.merge(result.metrics);
    }
    metrics.scheduler_workers = shard_count;
    Ok(SchedulerResult { records, metrics })
}

pub fn component_name(kind: ChanceKind, id: usize) -> &'static str {
    match kind {
        ChanceKind::CardReveal | ChanceKind::AgeDeal => data::card(id).name,
        ChanceKind::GreatLibraryDraw => data::progress(id).name,
        ChanceKind::WonderGroupReveal => data::wonder(id).name,
    }
}
