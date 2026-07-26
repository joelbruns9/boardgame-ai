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

/// Chance capping is a cheap-move economy: a full-search move keeps the exact
/// root expectation, so its policy target is bit-identical to an uncapped run's
/// at the same state and seed.
fn cheap_offsets(configured: usize, full: bool) -> usize {
    if full {
        0
    } else {
        configured
    }
}

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
    /// PUCT root selection; evaluation only. Self-play must stay Gumbel.
    pub puct_root: bool,
    pub age_deal_samples: usize,
    pub age_deal_samples_by_player: Option<[usize; 2]>,
    /// Offsets per first-reveal stratum for the balanced double-reveal support,
    /// applied to CHEAP moves only. Zero keeps forced expansion exhaustive
    /// everywhere. Full-search moves always pass 0, so their policy targets --
    /// the training targets whose comparability across runs matters -- stay
    /// exactly what an uncapped run would produce (CHANCE_ENUMERATION_PLAN.md
    /// Step 3). Values large enough to retain the whole outcome space are
    /// no-ops, resolved per edge.
    pub cheap_double_reveal_offsets: usize,
    /// Per-seat override, for the seat-mirrored search-strength arena: it must
    /// be possible to play capped against exhaustive with one shared net.
    pub cheap_double_reveal_offsets_by_player: Option<[usize; 2]>,
    pub bot_by_player: [Option<BotKind>; 2],
    pub bot_exploration: f64,
    pub bot_policy_iterations: i64,
    pub max_moves: usize,
    /// Phase 2: hold the conflict-free wave invariant, making `leaf_batch > 1`
    /// an exact batching of `leaf_batch = 1` rather than an approximation.
    pub conflict_free_waves: bool,
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

/// The action to PLAY in a competitive game: argmax of the improved policy.
///
/// `SearchResult::action_index` is the Gumbel-perturbed selection -- correct as
/// self-play's exploration action (it is what gives the sampled move its
/// policy-improvement property), but it carries exploration noise that has no
/// place in an arena or advisor game. `policy_target` is built from the same
/// logits WITHOUT the Gumbel keys, so its argmax is the noise-free choice.
///
/// First-max tie-break over `legal` order, matching Python's `max(dict, key=)`.
fn best_policy_action(legal: &[usize], policy: &[f64]) -> usize {
    assert_eq!(legal.len(), policy.len());
    let mut best = *legal
        .first()
        .expect("search root cannot have no legal actions");
    let mut best_weight = f64::NEG_INFINITY;
    for (&action, &weight) in legal.iter().zip(policy) {
        if weight > best_weight {
            best_weight = weight;
            best = action;
        }
    }
    best
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
            puct_root: cfg.puct_root,
            age_deal_samples: cfg.age_deal_samples,
            double_reveal_offsets: cheap_offsets(
                cfg.cheap_double_reveal_offsets_by_player
                    .map_or(cfg.cheap_double_reveal_offsets, |per_seat| per_seat[actor]),
                full,
            ),
            conflict_free_waves: cfg.conflict_free_waves,
        };
        let leaf_batch = cfg
            .leaf_batch_by_player
            .map_or(cfg.leaf_batch, |batches| batches[actor]);
        let (result, _, _) =
            tree_resumable::search_closed_batched(&state, &eval, &search_cfg, leaf_batch)?;
        let action = if cfg.deterministic_actions {
            best_policy_action(&legal, &result.policy_target)
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
    /// Root edges closed over an APPROXIMATE support. Exact-chance runs (gates,
    /// evaluation) assert zero.
    pub fixed_support_edges: usize,
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
    // --- Phase 0 telemetry ---
    // The `*_slot_cycles` counters above count *loop iterations*, so they say
    // nothing about how long slots were live. These integrate the slot census
    // over wall time instead, and are the only occupancy numbers a throughput
    // decision may be based on.
    /// Wall time spent inside the scheduler loop.
    pub scheduler_wall_ns: u64,
    /// ∫ live slots dt. Divided by `scheduler_wall_ns` this is time-weighted
    /// occupancy: the mean number of games actually in flight.
    pub live_slot_ns: u64,
    pub ready_slot_ns: u64,
    pub waiting_slot_ns: u64,
    pub idle_slot_ns: u64,
    pub max_live_slots: usize,
    /// The activation ceiling this run was given (globally, across shards).
    pub max_active_slots: usize,
    /// Live slots at the moment each global batch was submitted, and that
    /// submission's offset from the start of the scheduler loop. Together these
    /// let the caller separate the steady-state window from the drain tail
    /// instead of averaging over both. Offsets are loop-relative, so they are
    /// only comparable across shards when `scheduler_workers == 1`.
    pub batch_live_slots: Vec<usize>,
    pub batch_submit_ns: Vec<u64>,
    /// Peak of the summed arena node count over live slots, the peak for any
    /// single slot, and a sampled deep-byte measurement of one slot's arena.
    /// This is the term that bounds `max_active_slots`.
    pub arena_nodes_live_peak: usize,
    pub arena_nodes_slot_peak: usize,
    pub arena_deep_bytes_slot_peak: usize,
    pub arena_node_struct_bytes: usize,
    /// Phase 2: realized leaf-wave widths, and how many waves the conflict-free
    /// invariant cut short. The plan requires the realized distribution; width
    /// cannot be inferred from `top_k` or `leaf_batch`.
    pub wave_width_histogram: [usize; tree_resumable::WAVE_WIDTH_BUCKETS],
    pub conflict_cuts: usize,
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
        self.fixed_support_edges += other.fixed_support_edges;
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
        // Shards run concurrently: wall time is the envelope, slot-seconds add,
        // and the memory peaks are summed because a bound must hold even if the
        // shard peaks coincide.
        self.scheduler_wall_ns = self.scheduler_wall_ns.max(other.scheduler_wall_ns);
        self.live_slot_ns += other.live_slot_ns;
        self.ready_slot_ns += other.ready_slot_ns;
        self.waiting_slot_ns += other.waiting_slot_ns;
        self.idle_slot_ns += other.idle_slot_ns;
        // Both are global figures every shard already reports identically:
        // taking the max keeps them global rather than summing shard peaks that
        // need not have coincided.
        self.max_live_slots = self.max_live_slots.max(other.max_live_slots);
        self.max_active_slots = other.max_active_slots;
        self.batch_live_slots.extend(other.batch_live_slots);
        self.batch_submit_ns.extend(other.batch_submit_ns);
        self.arena_nodes_live_peak += other.arena_nodes_live_peak;
        self.arena_nodes_slot_peak = self.arena_nodes_slot_peak.max(other.arena_nodes_slot_peak);
        self.arena_deep_bytes_slot_peak = self
            .arena_deep_bytes_slot_peak
            .max(other.arena_deep_bytes_slot_peak);
        self.arena_node_struct_bytes = other.arena_node_struct_bytes;
        for (bucket, count) in self
            .wave_width_histogram
            .iter_mut()
            .zip(other.wave_width_histogram)
        {
            *bucket += count;
        }
        self.conflict_cuts += other.conflict_cuts;
    }
}

/// Integrates the slot census over wall time and samples arena occupancy.
///
/// Charging works one cycle in arrears: each tick attributes the time since the
/// previous tick to the census that was true over that interval, then records
/// the new census. That is what makes `live_slot_ns / scheduler_wall_ns` an
/// honest time-weighted occupancy rather than a per-iteration count.
struct Occupancy {
    start: Instant,
    last: Instant,
    live: usize,
    ready: usize,
    waiting: usize,
    idle: usize,
    probe: usize,
}

impl Occupancy {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last: now,
            live: 0,
            ready: 0,
            waiting: 0,
            idle: 0,
            probe: 0,
        }
    }

    fn elapsed_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    /// Charge elapsed time to the previous census, then adopt the pool's current
    /// one. `outstanding` is empty for the non-pipelined scheduler, where no
    /// slot can be waiting on an in-flight batch.
    ///
    /// "Idle" is unused *capacity* — activations the budget allows but the queue
    /// cannot fill — not finished games, since finished games leave the pool.
    fn tick(
        &mut self,
        metrics: &mut SchedulerMetrics,
        pool: &SlotPool,
        outstanding: &[bool],
        capacity: usize,
    ) {
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_nanos() as u64;
        self.last = now;
        metrics.live_slot_ns += dt * self.live as u64;
        metrics.ready_slot_ns += dt * self.ready as u64;
        metrics.waiting_slot_ns += dt * self.waiting as u64;
        metrics.idle_slot_ns += dt * self.idle as u64;

        let (mut live, mut ready, mut waiting) = (0, 0, 0);
        let mut live_nodes = 0;
        for (index, entry) in pool.entries.iter().enumerate() {
            let SlotEntry::Active(slot) = entry else {
                continue;
            };
            live += 1;
            live_nodes += slot.arena_nodes();
            if outstanding.get(index).copied().unwrap_or(false) {
                waiting += 1;
            } else {
                ready += 1;
            }
        }
        self.live = live;
        self.ready = ready;
        self.waiting = waiting;
        self.idle = capacity.saturating_sub(live);
        metrics.max_live_slots = metrics.max_live_slots.max(live);
        metrics.arena_nodes_live_peak = metrics.arena_nodes_live_peak.max(live_nodes);

        // Deep byte accounting walks a whole arena, so sample one entry per
        // cycle round-robin; the cost is negligible beside a global batch.
        if !pool.entries.is_empty() {
            self.probe = (self.probe + 1) % pool.entries.len();
            if let SlotEntry::Active(slot) = &pool.entries[self.probe] {
                metrics.arena_nodes_slot_peak =
                    metrics.arena_nodes_slot_peak.max(slot.arena_nodes());
                metrics.arena_deep_bytes_slot_peak = metrics
                    .arena_deep_bytes_slot_peak
                    .max(slot.arena_deep_bytes());
            }
        }
    }
}

/// Activation budget shared by every scheduler shard.
///
/// Phase 1 requires the slot budget to be **global**: sharding the scheduler
/// must not multiply the number of games resident in memory. Each shard keeps
/// one reserved activation — without it a shard could be permanently starved and
/// the run would deadlock — and competes for the remainder through `spare`.
pub struct SlotBudget {
    spare: std::sync::atomic::AtomicUsize,
    /// Games active across *all* shards, and its high-water mark. Summing each
    /// shard's own peak would overstate concurrency, because shard peaks need
    /// not coincide; this counter is the real thing.
    live: std::sync::atomic::AtomicUsize,
    peak_live: std::sync::atomic::AtomicUsize,
    total: usize,
    shards: usize,
}

impl SlotBudget {
    pub fn new(max_active_slots: usize, shards: usize) -> PyResult<Self> {
        let shards = shards.max(1);
        if max_active_slots < shards {
            return Err(PyValueError::new_err(format!(
                "max_active_slots={max_active_slots} is below scheduler_workers={shards}: \
                 every shard needs at least one active slot to make progress"
            )));
        }
        Ok(Self {
            spare: std::sync::atomic::AtomicUsize::new(max_active_slots - shards),
            live: std::sync::atomic::AtomicUsize::new(0),
            peak_live: std::sync::atomic::AtomicUsize::new(0),
            total: max_active_slots,
            shards,
        })
    }

    /// Activations happen once per game, so exact global accounting here is
    /// cheap enough to prefer over sampling.
    fn on_activate(&self) {
        use std::sync::atomic::Ordering;
        let live = self.live.fetch_add(1, Ordering::AcqRel) + 1;
        self.peak_live.fetch_max(live, Ordering::AcqRel);
    }

    fn on_retire(&self) {
        self.live
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }

    pub fn peak_live(&self) -> usize {
        self.peak_live.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Budget for a single-shard run, where every activation is local.
    pub fn single(max_active_slots: usize) -> PyResult<Self> {
        Self::new(max_active_slots.max(1), 1)
    }

    fn try_take(&self) -> bool {
        use std::sync::atomic::Ordering;
        self.spare
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_sub(1)
            })
            .is_ok()
    }

    fn give_back(&self) {
        self.spare
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn shards(&self) -> usize {
        self.shards
    }
}

/// One job's position in the rolling pool.
///
/// Queued jobs hold only their starting state and config — no tree, no search
/// session — because the arena a `GameSlot` accumulates is what actually bounds
/// concurrency (Phase 0 measured several MB per active game against ~1 KB for a
/// `GameState`). Finished games are retired to records immediately so a
/// completed game stops costing memory the moment it ends.
enum SlotEntry {
    Queued(Box<(GameState, SelfPlayConfig)>),
    Active(Box<GameSlot>),
    Finished(Box<GameRecord>),
    /// Transient hole while an entry is moved between the states above.
    Taken,
}

/// Bounded rolling pool of active games over a longer queue of jobs.
///
/// Entries are indexed by **job index** throughout, so evaluation groups,
/// scatter and the returned records all stay in input order regardless of the
/// order in which games activate or finish.
struct SlotPool {
    entries: Vec<SlotEntry>,
    next_queued: usize,
    active_count: usize,
    holds_reserved: bool,
    budget_held: usize,
    finished: usize,
}

impl SlotPool {
    fn new(jobs: Vec<(GameState, SelfPlayConfig)>) -> Self {
        Self {
            entries: jobs
                .into_iter()
                .map(|job| SlotEntry::Queued(Box::new(job)))
                .collect(),
            next_queued: 0,
            active_count: 0,
            holds_reserved: false,
            budget_held: 0,
            finished: 0,
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn active_count(&self) -> usize {
        self.active_count
    }

    /// Job indices of currently active games, ascending. Ascending order is what
    /// makes batch packing deterministic.
    fn active_indices(&self) -> Vec<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| matches!(entry, SlotEntry::Active(_)))
            .map(|(index, _)| index)
            .collect()
    }

    fn slot_mut(&mut self, index: usize) -> PyResult<&mut GameSlot> {
        match &mut self.entries[index] {
            SlotEntry::Active(slot) => Ok(slot),
            _ => Err(PyRuntimeError::new_err(format!(
                "evaluation delivered to job {index}, which is not an active slot"
            ))),
        }
    }

    fn work_remaining(&self) -> bool {
        self.active_count > 0 || self.next_queued < self.entries.len()
    }

    /// Activate queued jobs until the budget or the queue runs out.
    fn refill(&mut self, budget: &SlotBudget) -> PyResult<usize> {
        let mut activated = 0;
        while self.next_queued < self.entries.len() {
            // Every shard keeps one activation of its own; further concurrency
            // is drawn from the shared pool.
            if !self.holds_reserved {
                self.holds_reserved = true;
            } else if budget.try_take() {
                self.budget_held += 1;
            } else {
                break;
            }
            let index = self.next_queued;
            self.next_queued += 1;
            let entry = std::mem::replace(&mut self.entries[index], SlotEntry::Taken);
            let SlotEntry::Queued(job) = entry else {
                return Err(PyRuntimeError::new_err(
                    "pool queue pointer reached a job that is not queued",
                ));
            };
            let (state, cfg) = *job;
            match GameSlot::new(state, cfg) {
                Ok(slot) => self.entries[index] = SlotEntry::Active(Box::new(slot)),
                Err(err) => {
                    self.release_activation();
                    return Err(err);
                }
            }
            self.active_count += 1;
            budget.on_activate();
            activated += 1;
        }
        Ok(activated)
    }

    fn release_activation(&mut self) {
        if self.budget_held > 0 {
            self.budget_held -= 1;
        } else {
            self.holds_reserved = false;
        }
    }

    /// Turn a completed slot into its record, freeing its arena and its budget.
    fn retire(
        &mut self,
        index: usize,
        metrics: &mut SchedulerMetrics,
        budget: &SlotBudget,
    ) -> PyResult<()> {
        let entry = std::mem::replace(&mut self.entries[index], SlotEntry::Taken);
        let SlotEntry::Active(slot) = entry else {
            self.entries[index] = entry;
            return Err(PyRuntimeError::new_err(format!(
                "cannot retire job {index}, which is not active"
            )));
        };
        absorb_slot_metrics(metrics, &slot);
        let record = slot.into_record()?;
        self.entries[index] = SlotEntry::Finished(Box::new(record));
        self.active_count -= 1;
        self.finished += 1;
        budget.on_retire();
        if self.budget_held > 0 {
            self.budget_held -= 1;
            budget.give_back();
        } else {
            self.holds_reserved = false;
        }
        Ok(())
    }

    fn cancel_all(&mut self) {
        for entry in &mut self.entries {
            if let SlotEntry::Active(slot) = entry {
                slot.cancel_pending();
            }
        }
    }

    /// Records in **input order**, independent of completion order.
    fn into_records(self) -> PyResult<Vec<GameRecord>> {
        self.entries
            .into_iter()
            .map(|entry| match entry {
                SlotEntry::Finished(record) => Ok(*record),
                _ => Err(PyRuntimeError::new_err(
                    "scheduler attempted to emit a game that never completed",
                )),
            })
            .collect()
    }
}

/// Fold one finished slot's counters into the run metrics.
///
/// Done at retirement rather than at the end, because the pool drops each slot
/// as soon as its game ends — that is the whole point of the pool.
fn absorb_slot_metrics(metrics: &mut SchedulerMetrics, slot: &GameSlot) {
    metrics.moves += slot.moves.len();
    metrics.simulations += slot.simulations;
    metrics.requested_nn_leaves += slot.requested_nn_leaves;
    metrics.unique_nn_leaves += slot.unique_nn_leaves;
    metrics.terminal_leaves += slot.terminal_leaves;
    metrics.collisions += slot.collisions;
    metrics.forced_rows += slot.forced_rows;
    metrics.fixed_support_edges += slot.fixed_support_edges;
    for kind in 0..4 {
        metrics.forced_rows_by_kind[kind] += slot.forced_rows_by_kind[kind];
    }
    metrics.forced_cache_hits += slot.forced_cache_hits;
    metrics
        .forced_rows_per_search
        .extend(slot.forced_rows_per_search.iter().copied());
    metrics.rust_tree_ns += slot.tree_ns;
    metrics.rust_chance_ns += slot.chance_ns;
    metrics.rust_record_ns += slot.record_ns;
    metrics.scatter_ns += slot.scatter_ns;
    for (bucket, count) in metrics
        .wave_width_histogram
        .iter_mut()
        .zip(slot.wave_width_histogram)
    {
        *bucket += count;
    }
    metrics.conflict_cuts += slot.conflict_cuts;
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
    fixed_support_edges: usize,
    wave_width_histogram: [usize; tree_resumable::WAVE_WIDTH_BUCKETS],
    conflict_cuts: usize,
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
            fixed_support_edges: 0,
            wave_width_histogram: [0; tree_resumable::WAVE_WIDTH_BUCKETS],
            conflict_cuts: 0,
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
                puct_root: self.cfg.puct_root,
                age_deal_samples: self
                    .cfg
                    .age_deal_samples_by_player
                    .map_or(self.cfg.age_deal_samples, |samples| samples[actor]),
                double_reveal_offsets: cheap_offsets(
                    self.cfg
                        .cheap_double_reveal_offsets_by_player
                        .map_or(self.cfg.cheap_double_reveal_offsets, |per_seat| {
                            per_seat[actor]
                        }),
                    full,
                ),
                conflict_free_waves: self.cfg.conflict_free_waves,
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
            self.fixed_support_edges += metrics.fixed_support_edges;
            for (target, value) in self
                .forced_rows_by_kind
                .iter_mut()
                .zip(metrics.forced_rows_by_kind)
            {
                *target += value;
            }
            self.forced_cache_hits += metrics.cached_forced_leaves;
            for (bucket, count) in self
                .wave_width_histogram
                .iter_mut()
                .zip(metrics.wave_width_histogram)
            {
                *bucket += count;
            }
            self.conflict_cuts += metrics.conflict_cuts;
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
            best_policy_action(&meta.legal, &result.policy_target)
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

    /// Arena nodes this slot currently holds; zero between searches, when the
    /// tree has been dropped.
    fn arena_nodes(&self) -> usize {
        match &self.stage {
            SlotStage::Searching { session, .. } => session.arena_nodes(),
            _ => 0,
        }
    }

    fn arena_deep_bytes(&self) -> usize {
        match &self.stage {
            SlotStage::Searching { session, .. } => session.arena_deep_bytes(),
            _ => 0,
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

/// `0` means "no pool": every job is active at once, which is what every
/// pre-Phase-1 caller expected.
fn resolve_max_active(max_active_slots: usize, jobs: usize) -> usize {
    if max_active_slots == 0 {
        jobs
    } else {
        max_active_slots.min(jobs)
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

/// F4.4 deterministic cooperative scheduler, F4/Phase-1 rolling pool.
///
/// Every active slot advances until it yields one indivisible root/leaf
/// evaluation group; groups are packed in job-index order up to
/// `global_batch_cap`, evaluated once, and scattered before the next scheduler
/// cycle. `jobs` may greatly exceed `max_active_slots`: finished games are
/// retired and queued ones activated in their place, so concurrency holds at
/// `min(max_active_slots, jobs remaining)` instead of decaying to one.
///
/// Records are always returned in input order, independent of the order in
/// which games activate or reach terminal states.
pub fn run_many<E: Eval>(
    jobs: Vec<(GameState, SelfPlayConfig)>,
    evaluator: &E,
    global_batch_cap: usize,
    max_active_slots: usize,
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
    let budget = SlotBudget::single(resolve_max_active(max_active_slots, jobs.len()))?;
    let mut pool = SlotPool::new(jobs);
    let mut metrics = SchedulerMetrics {
        games: pool.len(),
        scheduler_workers: 1,
        arena_node_struct_bytes: std::mem::size_of::<tree_resumable::Node>(),
        max_active_slots: budget.total(),
        ..SchedulerMetrics::default()
    };
    let mut occupancy = Occupancy::new();

    while pool.work_remaining() {
        if let Err(err) = pool.refill(&budget) {
            pool.cancel_all();
            return Err(err);
        }
        occupancy.tick(&mut metrics, &pool, &[], budget.total());
        metrics.scheduler_cycles += 1;
        metrics.scheduler_ready_slot_cycles += pool.active_count() as u64;
        metrics.scheduler_idle_slot_cycles +=
            budget.total().saturating_sub(pool.active_count()) as u64;

        let mut groups = Vec::new();
        let mut retire = Vec::new();
        let forced_row_limit = (global_batch_cap / pool.active_count().max(1)).max(1);
        for slot_index in pool.active_indices() {
            let outcome = pool
                .slot_mut(slot_index)
                .and_then(|slot| slot.next_eval_group(slot_index, forced_row_limit));
            match outcome {
                Ok(Some(group)) => groups.push(group),
                // A slot that yields no group has finished its game.
                Ok(None) => retire.push(slot_index),
                Err(err) => {
                    pool.cancel_all();
                    return Err(err);
                }
            }
        }
        for slot_index in retire {
            if let Err(err) = pool.retire(slot_index, &mut metrics, &budget) {
                pool.cancel_all();
                return Err(err);
            }
        }
        if groups.is_empty() {
            if !pool.work_remaining() {
                break;
            }
            // Retiring freed budget, so the next cycle can activate more work.
            if pool.active_count() == 0 {
                continue;
            }
            pool.cancel_all();
            return Err(PyRuntimeError::new_err(
                "cooperative scheduler made no progress with live slots",
            ));
        }

        let mut pending = std::collections::VecDeque::from(groups);
        while !pending.is_empty() {
            let (batch, row_count) = match take_global_batch(&mut pending, global_batch_cap) {
                Ok(taken) => taken,
                Err(err) => {
                    pool.cancel_all();
                    return Err(err);
                }
            };

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
            metrics.batch_live_slots.push(pool.active_count());
            metrics.batch_submit_ns.push(occupancy.elapsed_ns());
            let evaluations = match evaluator.evaluate_batch_prepared(&state_refs, &actors, &legals)
            {
                Ok(rows) => rows,
                Err(err) => {
                    pool.cancel_all();
                    return Err(err);
                }
            };
            if evaluations.len() != row_count {
                pool.cancel_all();
                return Err(PyValueError::new_err(format!(
                    "global evaluator returned {} rows for {row_count} states",
                    evaluations.len()
                )));
            }
            metrics.global_batches += 1;
            metrics.global_rows += row_count;
            metrics.max_batch_rows = metrics.max_batch_rows.max(row_count);
            metrics.batch_rows.push(row_count);

            if let Err(err) = scatter_batch(&mut pool, &mut metrics, batch, evaluations) {
                pool.cancel_all();
                return Err(err);
            }
        }
    }

    occupancy.tick(&mut metrics, &pool, &[], budget.total());
    metrics.scheduler_wall_ns = occupancy.elapsed_ns();
    // Exact, from the activation counter, rather than the cycle-boundary sample.
    metrics.max_live_slots = budget.peak_live();
    let records = pool.into_records()?;
    Ok(SchedulerResult { records, metrics })
}

/// Deliver one evaluated batch back to the slots that produced it.
fn scatter_batch(
    pool: &mut SlotPool,
    metrics: &mut SchedulerMetrics,
    batch: Vec<EvalGroup>,
    evaluations: Vec<(f64, Vec<f64>)>,
) -> PyResult<()> {
    let mut cursor = 0;
    for group in batch {
        let count = group.states.len();
        let rows = evaluations[cursor..cursor + count].to_vec();
        cursor += count;
        let slot = pool.slot_mut(group.slot)?;
        match group.kind {
            EvalGroupKind::Root => {
                metrics.root_rows += count;
                slot.apply_root(rows.into_iter().next().expect("root row"))?;
            }
            EvalGroupKind::Forced(request) => {
                metrics.leaf_rows += count;
                slot.apply_wave(request, &group.states, rows)?;
            }
            EvalGroupKind::Wave(request) => {
                metrics.leaf_rows += count;
                metrics.ordinary_leaf_rows += count;
                slot.apply_wave(request, &group.states, rows)?;
            }
        }
    }
    Ok(())
}

struct InflightBatch {
    groups: Vec<EvalGroup>,
    row_count: usize,
    ticket: EvalTicket,
}

/// Advance every active slot that is not already waiting on an in-flight batch,
/// and report which slots finished their game so the caller can retire them.
fn collect_ready_groups(
    pool: &mut SlotPool,
    outstanding: &mut [bool],
    pending: &mut std::collections::VecDeque<EvalGroup>,
    global_batch_cap: usize,
    finished: &mut Vec<usize>,
) -> PyResult<usize> {
    let mut collected = 0;
    let active = pool.active_indices();
    let ready = active
        .iter()
        .filter(|index| !outstanding[**index])
        .count()
        .max(1);
    let forced_row_limit = (global_batch_cap / ready).max(1);
    for slot_index in active {
        if outstanding[slot_index] {
            continue;
        }
        match pool
            .slot_mut(slot_index)?
            .next_eval_group(slot_index, forced_row_limit)?
        {
            Some(group) => {
                outstanding[slot_index] = true;
                pending.push_back(group);
                collected += 1;
            }
            None => finished.push(slot_index),
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
    budget: &SlotBudget,
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
    let mut pool = SlotPool::new(jobs);
    let mut outstanding = vec![false; pool.len()];
    let mut pending = std::collections::VecDeque::new();
    let mut inflight = std::collections::VecDeque::<InflightBatch>::new();
    let mut metrics = SchedulerMetrics {
        games: pool.len(),
        scheduler_workers: 1,
        arena_node_struct_bytes: std::mem::size_of::<tree_resumable::Node>(),
        max_active_slots: budget.total(),
        ..SchedulerMetrics::default()
    };
    let mut occupancy = Occupancy::new();
    // The shared budget bounds every shard together, so a single shard's own
    // ceiling is only known once it has actually acquired activations.
    let local_capacity = budget.total() / budget.shards().max(1);

    loop {
        if let Err(err) = pool.refill(budget) {
            pool.cancel_all();
            return Err(err);
        }
        occupancy.tick(&mut metrics, &pool, &outstanding, local_capacity);
        metrics.scheduler_cycles += 1;
        for slot_index in pool.active_indices() {
            if outstanding[slot_index] {
                metrics.scheduler_waiting_slot_cycles += 1;
            } else {
                metrics.scheduler_ready_slot_cycles += 1;
            }
        }
        metrics.scheduler_idle_slot_cycles +=
            local_capacity.saturating_sub(pool.active_count()) as u64;
        let mut finished = Vec::new();
        if let Err(err) = collect_ready_groups(
            &mut pool,
            &mut outstanding,
            &mut pending,
            global_batch_cap,
            &mut finished,
        ) {
            pool.cancel_all();
            return Err(err);
        }
        for slot_index in finished {
            if let Err(err) = pool.retire(slot_index, &mut metrics, budget) {
                pool.cancel_all();
                return Err(err);
            }
        }

        while inflight.len() < max_inflight_batches && !pending.is_empty() {
            let (groups, row_count) = match take_global_batch(&mut pending, global_batch_cap) {
                Ok(batch) => batch,
                Err(err) => {
                    pool.cancel_all();
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
                    pool.cancel_all();
                    return Err(err);
                }
            };
            metrics.global_batches += 1;
            metrics.global_rows += row_count;
            metrics.max_batch_rows = metrics.max_batch_rows.max(row_count);
            metrics.batch_rows.push(row_count);
            metrics.batch_live_slots.push(pool.active_count());
            metrics.batch_submit_ns.push(occupancy.elapsed_ns());
            inflight.push_back(InflightBatch {
                groups,
                row_count,
                ticket,
            });
            metrics.max_inflight_batches = metrics.max_inflight_batches.max(inflight.len());
        }

        let Some(flight) = inflight.pop_front() else {
            if !pool.work_remaining() {
                break;
            }
            // Nothing in flight and nothing ready: the only legitimate reason is
            // that retiring freed budget which the next refill will spend.
            if pool.active_count() == 0 {
                continue;
            }
            pool.cancel_all();
            return Err(PyRuntimeError::new_err(
                "pipelined scheduler made no progress with live slots",
            ));
        };
        let evaluations = match flight.ticket.wait() {
            Ok(rows) => rows,
            Err(err) => {
                pool.cancel_all();
                return Err(err);
            }
        };
        if evaluations.len() != flight.row_count {
            pool.cancel_all();
            return Err(PyValueError::new_err(format!(
                "global evaluator returned {} rows for {} states",
                evaluations.len(),
                flight.row_count
            )));
        }

        for group in &flight.groups {
            outstanding[group.slot] = false;
        }
        if let Err(err) = scatter_batch(&mut pool, &mut metrics, flight.groups, evaluations) {
            pool.cancel_all();
            return Err(err);
        }
    }

    occupancy.tick(&mut metrics, &pool, &outstanding, local_capacity);
    metrics.scheduler_wall_ns = occupancy.elapsed_ns();
    // Exact and global: with shards this is concurrency across all of them, not
    // this shard's own peak.
    metrics.max_live_slots = budget.peak_live();
    let records = pool.into_records()?;
    Ok(SchedulerResult { records, metrics })
}

/// Coarse persistent scheduler shards. Each shard owns a contiguous set of
/// logical games and submits prepared batches to the same inference worker.
/// Records are joined in shard/input order, independent of completion order.
/// Search choices can still differ across shard counts when the evaluator is
/// sensitive to batch shape (notably through CUDA floating-point ties); only
/// result ordering, not cross-shard bit identity, is guaranteed.
/// The activation budget is **global**: sharding the scheduler must not
/// multiply the number of games resident in memory, so all shards draw from one
/// `SlotBudget` rather than each receiving `max_active_slots` of their own.
pub fn run_many_pipelined_sharded(
    jobs: Vec<(GameState, SelfPlayConfig)>,
    worker: &EvalWorker,
    global_batch_cap: usize,
    max_inflight_batches: usize,
    scheduler_workers: usize,
    max_active_slots: usize,
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
    let active_cap = resolve_max_active(max_active_slots, jobs.len());
    if scheduler_workers == 1 || jobs.len() == 1 {
        let budget = SlotBudget::single(active_cap)?;
        return run_many_pipelined(
            jobs,
            worker,
            global_batch_cap,
            max_inflight_batches,
            &budget,
        );
    }
    let shard_count = scheduler_workers.min(jobs.len());
    let chunk_size = (jobs.len() + shard_count - 1) / shard_count;
    // Deliberately not widened to fit: a budget below the shard count is a
    // configuration error, and silently raising it would defeat the ceiling.
    let budget = SlotBudget::new(active_cap, shard_count)?;
    let mut shards = Vec::new();
    let mut remaining = jobs.into_iter();
    loop {
        let chunk: Vec<_> = remaining.by_ref().take(chunk_size).collect();
        if chunk.is_empty() {
            break;
        }
        shards.push(chunk);
    }
    let budget = &budget;
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = shards
            .into_iter()
            .map(|shard| {
                scope.spawn(move || {
                    run_many_pipelined(
                        shard,
                        worker,
                        global_batch_cap,
                        max_inflight_batches,
                        budget,
                    )
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
