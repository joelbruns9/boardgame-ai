//! F4.3 single-slot full-game self-play.
//!
//! The complete move loop lives here: search-size scheduling, search seeds,
//! draft-prior blending, temperature sampling, action/chance application, and
//! move/result recording.  Python is entered only by `PyEval` while a search
//! needs neural evaluations; it does not regain control between moves.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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
/// Root candidate width, by search budget.
///
/// Sequential halving spends its first round giving every candidate the same
/// allocation, so `m` and `n` are not independent: at `n = 20` with `m = 10`
/// (7WD's typical legal count, which is what `top_k = 16` clips to) every
/// candidate gets exactly one simulation, and a one-simulation Q is the value
/// head's static opinion of the position after the move -- no opponent reply.
/// In 7WD that is the whole game: what you take determines what you expose.
///
/// Halving `m` doubles the floor. `cheap_top_k = 0` keeps the single shared
/// width, which is the pre-2026-08-06 behaviour.
static CHEAP_TOP_K: AtomicUsize = AtomicUsize::new(0);

/// Set the cheap-move root width, process-wide. Global for the same reason the
/// temperature schedule is: only self-play generation reads it -- gates, arenas
/// and anchors run full-search width -- so no two callers need different values
/// at once, and threading it through would touch ten pyo3 signatures.
pub fn set_cheap_top_k(width: usize) {
    CHEAP_TOP_K.store(width, Ordering::Relaxed);
}

pub fn cheap_top_k() -> usize {
    CHEAP_TOP_K.load(Ordering::Relaxed)
}

fn search_top_k(top_k: usize, cheap_top_k: usize, full: bool) -> usize {
    if full || cheap_top_k == 0 {
        top_k
    } else {
        cheap_top_k
    }
}

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
    /// Root selection for FULL-budget moves.
    pub puct_root: bool,
    /// Whether this call may run the exact endgame solver at all.
    ///
    /// Per-call, and default false, for the same reason `cheap_puct_root` is:
    /// gates invoke the same entry points in the same process, so a
    /// process-global would be inherited by every promotion gate. That is worse
    /// than wasted time -- masking makes BOTH sides play the endgame provably
    /// optimally, which erases endgame skill from the comparison. A candidate
    /// better at endgames would get no credit and one worse no penalty, exactly
    /// when the open problem is that nothing promotes.
    ///
    /// Note gating on `net_by_player[actor] == 0` instead would be actively
    /// worse: a gate runs different nets per seat, so the solver would mask one
    /// side's moves and not the other's. Asymmetric masking is a bias, not a
    /// handicap.
    pub solve_endgames: bool,
    /// Root selection for CHEAP moves; `None` means "whatever `puct_root` says".
    ///
    /// This is the hybrid: PUCT on the moves that produce training targets,
    /// Gumbel on the cheap ones, whose guarantee is designed for exactly the
    /// tiny budgets those run at. Cheap moves emit no policy target, so mixing
    /// root selection there costs no label consistency.
    ///
    /// Per-call and defaulting to `puct_root` on purpose. The obvious spelling
    /// -- keying the hybrid off `full` -- is wrong, because `full` means "this
    /// move drew the full simulation budget", NOT "this move is recorded". Gate
    /// games set `full_search_fraction = 0.0` and take their strength from the
    /// cheap path with `cheap_sims = full_sims = gate_sims`, so `puct_root &&
    /// full` would run every promotion gate under Gumbel while
    /// `--eval-search-mode puct` reported success. A process-global would fail
    /// the same way, since gates call the same entry points; only an explicit
    /// per-call value a gate never sets is safe.
    pub cheap_puct_root: Option<bool>,
    /// Root exploration noise for the PUCT root; inert under Gumbel.
    pub dirichlet_epsilon: f64,
    pub dirichlet_alpha: f64,
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
    /// W1.3 league play: which network evaluates each seat's own searches.
    ///
    /// `[0, 0]` -- the default -- is ordinary self-play: one network, and the
    /// packed `net_ids` come out all zeros, so the evaluation boundary is
    /// byte-identical to a build without this field. `[0, 1]` puts the archived
    /// opponent on seat 1: every leaf of *seat 1's* searches routes to network 1,
    /// including the leaves where seat 0 is to move, because the searcher owns
    /// the network.
    pub net_by_player: [u8; 2],
    pub bot_exploration: f64,
    pub bot_policy_iterations: i64,
    pub max_moves: usize,
    /// Phase 2: hold the conflict-free wave invariant, making `leaf_batch > 1`
    /// an exact batching of `leaf_batch = 1` rather than an approximation.
    pub conflict_free_waves: bool,
    /// Phase 2 follow-up: interleave each halving round instead of blocking it,
    /// so consecutive simulations come from different candidates and waves can
    /// actually widen. Changes search outputs — see `tree::SearchConfig`.
    pub round_robin_candidates: bool,
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
        if !(0.0..=1.0).contains(&self.dirichlet_epsilon)
        {
            return Err(PyValueError::new_err(
                "dirichlet_epsilon must lie in [0, 1]",
            ));
        }
        // `NaN <= 0.0` is false, so a bare positivity test admits NaN, which
        // then panics in `Rng::gamma`. Infinity is quieter and worse: every
        // draw returns inf, inf/inf makes the Dirichlet vector NaN, and NaN
        // comparisons pin PUCT selection to the first edge with no error.
        if !self.dirichlet_alpha.is_finite() || self.dirichlet_alpha <= 0.0 {
            return Err(PyValueError::new_err(
                "dirichlet_alpha must be finite and positive",
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
    /// Root prior over `legal`. Paired with `policy_target` it makes the search's
    /// actual policy improvement measurable per row.
    pub prior: Vec<f64>,
    pub root_value: f64,
    /// Completed Q of the action the search SELECTED, actor-relative.
    ///
    /// Distinct from `root_value`, which is the visit-weighted mean over
    /// everything the search explored -- including the inferior actions it
    /// deliberately sampled. Comparing `root_value` against a proven value
    /// therefore penalises a searcher for exploring, and penalises PUCT more
    /// than Gumbel because Dirichlet noise puts visits on moves it already
    /// believes are bad. This is the search's estimate of the position under
    /// its own best play, which is the quantity a proof is comparable to.
    pub action_value: f64,
    /// The network's raw evaluation of this root, actor-relative, before the
    /// search backed anything up. Paired with `root_value` and `solver_value`
    /// it separates a value head that is wrong from a search that is.
    pub net_root_value: f64,
    pub sims: usize,
    pub gumbel_topk: Vec<usize>,
    pub policy_excluded: bool,
    pub full_search: bool,
    pub search_seed: u64,
    pub is_bot: bool,
    /// Exact endgame value of the PRE-move position, actor-relative, when the
    /// solver reached one. Distinct from `root_value`, which stays the search's
    /// own estimate: the two are the sampled and the proven answer to the same
    /// question, and comparing them is how the solver's contribution gets
    /// measured. `None` on every move the solver did not solve.
    pub solver_value: Option<f64>,
    pub solver_regime: Option<&'static str>,
    /// Whether a solve was attempted at all. Separates the three populations a
    /// missing `solver_value` would otherwise conflate: the solver was off, the
    /// trigger did not select this move, or the solve ran and declined.
    pub solver_attempted: bool,
    /// Why an attempted solve declined -- `"unsolvable"` or `"budget"` -- and
    /// `None` when it succeeded.
    pub solver_stop: Option<&'static str>,
    /// Nodes visited, INCLUDING by a solve that then declined. A decline is not
    /// free: the budget was spent synchronously before it was reached.
    pub solver_nodes: u64,
    pub solver_masked: bool,
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
    pub digest_version: &'static str,
    pub final_digest: String,
    pub trajectory_digest: String,
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

    fn evaluate_batch_prepared_routed(
        &self,
        states: &[&GameState],
        actors: &[usize],
        legals: &[Vec<usize>],
        net_ids: &[u8],
    ) -> PyResult<Vec<(f64, Vec<f64>)>> {
        let rows = self
            .base
            .evaluate_batch_prepared_routed(states, actors, legals, net_ids)?;
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

pub(crate) fn actual_chance_outcomes(
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

// ---------------------------------------------------------------------------
// Exact endgame solver overlay (SOLVER_SELF_PLAY_PLAN.md)
// ---------------------------------------------------------------------------

/// Node budget for one endgame solve. **Zero disables the solver entirely**,
/// which is the default: with it unset every state, digest and target this
/// module produces is byte-identical to a build without this feature, so the
/// existing gates keep gating the thing they were written for.
///
/// The node budget, not the deadline, is the knob to set. A solve that stops on
/// wall-clock time makes self-play irreproducible from `(seed, net)`: the same
/// position solves on an idle machine and times out on a loaded one, the mask
/// appears or does not, and a different move is sampled. Nodes are deterministic.
/// `SOLVER_MAX_SECS` exists as a safety net against a pathological position
/// holding a scheduler slot, and should be set generously enough not to bind.
static SOLVER_MAX_NODES: AtomicU64 = AtomicU64::new(0);
static SOLVER_MAX_SECS_BITS: AtomicU64 = AtomicU64::new(0);
/// Cards still on the board at or below which a solve is attempted. Measured on
/// real (human) endgames, not bot ones, which are 3x cheaper and far narrower:
/// <=6 cards is milliseconds, 8 is 0.05-0.31s, 10 is 3.4-4.1s, 12 is ~60s or
/// unsolved. Age I and II are never solvable at any budget -- the next age's
/// deal is a sample-only chance edge -- so the trigger also requires Age III.
static SOLVER_MAX_CARDS: AtomicUsize = AtomicUsize::new(0);
/// Whether a successful solve also masks the policy target. Independent of the
/// value target on purpose: the two have different risk profiles and want to be
/// A/B-able separately.
static SOLVER_MASK_POLICY: AtomicBool = AtomicBool::new(false);

/// KataGo forced playouts at the PUCT root, process-wide; 0.0 is off.
///
/// Global for the same reason `cheap_top_k` and the temperature schedule are:
/// `SearchConfig` is constructed at twelve sites and no two concurrent callers
/// need different values. The SEARCH still takes it as an ordinary config field
/// -- `search.py` has it as one too, and an asymmetry there (field in Python,
/// global in Rust) is exactly the kind of thing the equivalence gate would trip
/// over on a boundary case nobody looks at.
static FORCED_PLAYOUT_K_BITS: AtomicU64 = AtomicU64::new(0);

pub fn set_forced_playout_k(k: f64) {
    FORCED_PLAYOUT_K_BITS.store(k.to_bits(), Ordering::Relaxed);
}

pub fn forced_playout_k() -> f64 {
    f64::from_bits(FORCED_PLAYOUT_K_BITS.load(Ordering::Relaxed))
}

/// Background solver threads; 0 (the default) solves inline on the scheduler
/// thread. Process-wide like the solver's other parameters -- and safe as one
/// for the same reason, since `SelfPlayConfig::solve_endgames` is what actually
/// grants permission to solve, and a gate never sets it.
static SOLVER_THREADS: AtomicUsize = AtomicUsize::new(0);

pub fn set_solver_threads(threads: usize) {
    SOLVER_THREADS.store(threads, Ordering::Relaxed);
}

pub fn solver_threads() -> usize {
    SOLVER_THREADS.load(Ordering::Relaxed)
}

/// Configure the endgame solver overlay, process-wide.
///
/// Global for the same reason `set_cheap_top_k` and the temperature schedule
/// are: only self-play generation reads it, and threading four more fields
/// through `SelfPlayConfig` would touch ten pyo3 signatures to express something
/// no two concurrent callers vary.
pub fn set_endgame_solver(max_nodes: u64, max_secs: f64, max_cards: usize, mask_policy: bool) {
    SOLVER_MAX_NODES.store(max_nodes, Ordering::Relaxed);
    SOLVER_MAX_SECS_BITS.store(max_secs.to_bits(), Ordering::Relaxed);
    SOLVER_MAX_CARDS.store(max_cards, Ordering::Relaxed);
    SOLVER_MASK_POLICY.store(mask_policy, Ordering::Relaxed);
}

/// `(max_nodes, max_secs, max_cards, mask_policy)` in force, for run manifests.
pub fn endgame_solver() -> (u64, f64, usize, bool) {
    (
        SOLVER_MAX_NODES.load(Ordering::Relaxed),
        f64::from_bits(SOLVER_MAX_SECS_BITS.load(Ordering::Relaxed)),
        SOLVER_MAX_CARDS.load(Ordering::Relaxed),
        SOLVER_MASK_POLICY.load(Ordering::Relaxed),
    )
}

/// What an ATTEMPTED solve contributes to the move about to be recorded.
///
/// Produced whenever the trigger fired, including when the solve then declined.
/// A refusal that left no trace would be indistinguishable from a move the
/// trigger never selected and from a run with the solver off, which makes the
/// declined positions unfindable in the buffer and hides what the failed
/// attempts cost -- and they are not free, since the solve is synchronous.
#[derive(Clone, Debug)]
pub struct SolverOverlay {
    /// Exact value of the position, ACTOR-relative, in [-1, 1]. `None` when the
    /// solve declined: the caller must treat that as "no answer", never as 0.
    pub value: Option<f64>,
    /// `"exact"` (no chance edge was crossed) or `"exact_expectimax"`. Only the
    /// first yields a value target: a chance-free value is a min/max over
    /// terminals, so it is exactly -1, 0 or +1 and maps onto a W/D/L class,
    /// while an expectimax scalar is `P(win) - P(loss)` and does not determine
    /// one -- see `dataset.solver_value_distribution`. `None` when declined.
    pub regime: Option<&'static str>,
    /// `"unsolvable"` (a sample-only Age deal: no budget would help) or
    /// `"budget"` (nodes or deadline). `None` when the solve succeeded.
    pub stop: Option<&'static str>,
    pub nodes: u64,
    /// Whether the policy target in this record was masked and renormalised.
    /// Per-move rather than per-run because it is exactly the set of rows whose
    /// `policy_target` follows the new definition; a global `TARGET_VERSION`
    /// bump would say less and invalidate every existing buffer to say it.
    pub masked: bool,
    /// The proven-optimal set, aligned to `legal`, or `None` when the solve
    /// declined or the mask is switched off.
    ///
    /// Returned rather than applied because a solved position has TWO
    /// distributions to correct and only one solve to pay for: the distribution
    /// the move is sampled from, and the distribution recorded as the label.
    /// They diverge under policy-target pruning, which cleans the label while
    /// deliberately leaving forced exploration in the trajectory.
    pub keep: Option<Vec<bool>>,
}

fn cards_left(state: &GameState) -> usize {
    state.tableau.slots.iter().filter(|card| card.present).count()
}

/// Would the solver take this position at all? The same predicate
/// `endgame_overlay` applies, exposed so the async path can decline to dispatch
/// rather than pay a channel round trip to be told no.
pub fn solver_wants(state: &GameState) -> bool {
    let (max_nodes, _secs, max_cards, _mask) = endgame_solver();
    max_nodes != 0 && solver_eligible(state, max_cards)
}

/// Cheap pre-filter, evaluated before any search work is committed to a solve.
fn solver_eligible(state: &GameState, max_cards: usize) -> bool {
    state.phase == Phase::PlayAge && state.tableau.age == 3 && cards_left(state) <= max_cards
}

/// Zero the losing moves and renormalise the survivors, in place.
///
/// Split out from the solve so the arithmetic is testable without one. The
/// fallback matters: `keep` always holds at least one move (the maximum attains
/// its own tolerance), but a search that put all of its mass on provably-losing
/// moves would leave nothing to renormalise, and dividing by that zero would
/// emit a NaN policy label.
fn mask_and_renormalise(policy: &mut [f64], keep: &[bool]) {
    // Runtime, not `debug_assert`: on a length mismatch `zip` silently truncates
    // and leaves an unnormalised label behind, which trains as a quiet poison
    // rather than a crash. Release builds are where that would happen.
    assert_eq!(
        policy.len(),
        keep.len(),
        "policy/keep length mismatch would silently truncate the mask"
    );
    let total: f64 = policy
        .iter()
        .zip(keep)
        .filter(|(_, &alive)| alive)
        .map(|(&p, _)| p)
        .sum();
    if total > 0.0 {
        for (p, &alive) in policy.iter_mut().zip(keep) {
            *p = if alive { *p / total } else { 0.0 };
        }
        return;
    }
    let survivors = keep.iter().filter(|&&alive| alive).count().max(1) as f64;
    for (p, &alive) in policy.iter_mut().zip(keep) {
        *p = if alive { 1.0 / survivors } else { 0.0 };
    }
}

/// Solve `state` if it is an endgame the budget can reach, and mask
/// `policy_target` with the answer.
///
/// The policy target masked here is the SEARCH's, not the net's raw prior, and
/// that is the load-bearing choice. 77-88% of legal moves at these positions are
/// proven equally optimal, so the solver says almost nothing about which
/// survivor is better; the search is the only thing in the system that ranks
/// them. Masking a prior would hand the net back a mask over its own opinion.
///
/// Returns `None` only when no solve was ATTEMPTED -- the solver is off, or the
/// position is out of range. A solve that was attempted and declined returns an
/// overlay with no value, so the decline is visible in the record along with
/// what it cost; a caller must treat a missing value as "no answer", never as 0.
pub fn endgame_overlay(
    state: &GameState,
    legal: &[usize],
) -> Option<SolverOverlay> {
    let (max_nodes, max_secs, max_cards, mask_policy) = endgame_solver();
    if max_nodes == 0 || !solver_eligible(state, max_cards) {
        return None;
    }
    let limits = crate::solver::Limits {
        max_nodes,
        deadline: Instant::now() + std::time::Duration::from_secs_f64(max_secs.max(0.0)),
    };
    // `Exact` is needed ONLY for the mask: it prices every root action on a full
    // window, while `ValueOnly` narrows the window as better actions are found,
    // so its non-best entries are bounds and the ties -- the entire signal the
    // mask reads -- are hidden. When the mask is off nothing but `root_value` is
    // consumed, and `ValueOnly` is strictly cheaper: 1.17x fewer nodes for the
    // mode itself, and star1 also bites far harder against a narrow root window
    // (0.77x the corpus nodes against 0.96x). The solve is synchronous, so that
    // is generation latency, not just CPU.
    let mode = if mask_policy {
        crate::solver::PolicyMode::Exact
    } else {
        crate::solver::PolicyMode::ValueOnly
    };
    let (outcome, nodes) = crate::solver::solve_root_counted(
        state,
        &limits,
        mode,
        crate::solver::ChancePruning::Star1,
    );
    let solved = match outcome {
        Ok(solved) => solved,
        Err(stop) => {
            return Some(SolverOverlay {
                value: None,
                regime: None,
                stop: Some(match stop {
                    crate::solver::SolveStop::Unsolvable => "unsolvable",
                    crate::solver::SolveStop::Budget => "budget",
                }),
                nodes,
                masked: false,
                keep: None,
            })
        }
    };

    let mut keep_set: Option<Vec<bool>> = None;
    if mask_policy {
        // `per_action` comes back in the solver's own move ordering, so align it
        // to `legal` before comparing anything positionally.
        let mut values = vec![f64::NAN; legal.len()];
        for &(index, value) in &solved.per_action {
            if let Some(position) = legal.iter().position(|&action| action == index) {
                values[position] = value;
            }
        }
        // Every legal action is priced in `Exact` mode. If that is somehow not
        // true the sets disagree, and masking against a partial pricing could
        // zero a move that was never evaluated -- so decline the mask and keep
        // the value, rather than emit a label built on a guess.
        if values.iter().all(|value| value.is_finite()) {
            let best = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let keep: Vec<bool> = values
                .iter()
                .map(|&value| value >= best - crate::solver::TIE_EPSILON)
                .collect();
            keep_set = Some(keep);
        }
    }

    Some(SolverOverlay {
        value: Some(solved.root_value),
        // `saw_chance` is set by any chance edge the search actually crossed, so
        // it can only ever over-report -- and that direction is the safe one.
        // False means every value in the explored tree was a min/max over
        // terminals, which makes the root value exactly -1, 0 or +1 and the
        // W/D/L class sound. True may occasionally mark a position whose value
        // did not really depend on chance, costing a value target rather than
        // inventing one. That asymmetry is why this is read off the search
        // rather than from a static look at the position, and it holds in both
        // policy modes.
        regime: Some(if solved.saw_chance {
            "exact_expectimax"
        } else {
            "exact"
        }),
        stop: None,
        nodes: solved.nodes,
        masked: keep_set.is_some(),
        keep: keep_set,
    })
}

/// Defaults, matching `phase_d.temperature_for_move`. Every run before
/// 2026-08-05 hard-coded these, so leaving them unset reproduces it exactly.
pub const DEFAULT_TEMPERATURE_FLOOR: f64 = 0.25;
pub const DEFAULT_TEMPERATURE_ANNEAL_MOVES: f64 = 20.0;

static TEMPERATURE_CONFIGURED: AtomicBool = AtomicBool::new(false);
static TEMPERATURE_FLOOR_BITS: AtomicU64 = AtomicU64::new(0);
static TEMPERATURE_ANNEAL_BITS: AtomicU64 = AtomicU64::new(0);

/// Set the self-play move-selection temperature schedule, process-wide.
///
/// Global rather than per-`SelfPlayConfig` deliberately: temperature applies
/// only to self-play move *selection*, and every evaluation path -- gates,
/// arenas, anchors -- sets `deterministic_actions` and takes the argmax
/// instead, so there is no caller that needs a different value at the same
/// time. Threading it through every construction site would touch a dozen
/// signatures to express something no caller varies.
///
/// The floor is the strongest diversity lever in self-play: at the historical
/// 0.25 the selection probability is proportional to visits^4, so a converged
/// policy plays a ~94% favourite as good as deterministically, and roughly 70%
/// of a ~70-move game is decided that way.
pub fn set_temperature_schedule(floor: f64, anneal_moves: f64) {
    TEMPERATURE_FLOOR_BITS.store(floor.to_bits(), Ordering::Relaxed);
    TEMPERATURE_ANNEAL_BITS.store(anneal_moves.to_bits(), Ordering::Relaxed);
    TEMPERATURE_CONFIGURED.store(true, Ordering::Relaxed);
}

/// `(floor, anneal_moves)` in force, for recording in run manifests.
pub fn temperature_schedule() -> (f64, f64) {
    if !TEMPERATURE_CONFIGURED.load(Ordering::Relaxed) {
        return (DEFAULT_TEMPERATURE_FLOOR, DEFAULT_TEMPERATURE_ANNEAL_MOVES);
    }
    (
        f64::from_bits(TEMPERATURE_FLOOR_BITS.load(Ordering::Relaxed)),
        f64::from_bits(TEMPERATURE_ANNEAL_BITS.load(Ordering::Relaxed)),
    )
}

fn temperature(move_index: usize) -> f64 {
    let (floor, anneal) = temperature_schedule();
    let progress = (move_index as f64 / anneal).min(1.0);
    1.0 + (floor - 1.0) * progress
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
    let mut trajectory = crate::digest::TrajectoryDigest::new();

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
            forced_playout_k: if full && cfg.net_by_player[actor] == 0 {
                // Gated exactly like `dirichlet_epsilon` below, and for the same
                // reasons. Forcing spends simulations on children PUCT declined,
                // which is a handicap: on an archived opponent it inflates the
                // learner's league win rate and skews those games' value labels
                // optimistic, and on a cheap move it degrades the trajectory for
                // no label benefit, since cheap moves emit no policy target.
                //
                // It also makes GATES immune for free. A gate runs
                // `full_search_fraction = 0.0`, so `full` is always false there
                // -- which matters because the advisor's `RustPuctSearch` takes
                // this as an explicit parameter defaulting to 0.0 and never
                // reads the global. Ungated, a gate would search with forcing
                // while the advisor did not, reintroducing the very
                // gate/advisor divergence the PUCT switch exists to close.
                forced_playout_k()
            } else {
                0.0
            },
            sims,
            top_k: search_top_k(cfg.top_k, cheap_top_k(), full),
            c_puct: cfg.c_puct,
            c_visit: cfg.c_visit,
            c_scale: cfg.c_scale,
            seed: search_seed,
            force_expand_root_chance: cfg.force_expand_root_chance,
            puct_root: if full {
                cfg.puct_root
            } else {
                cfg.cheap_puct_root.unwrap_or(cfg.puct_root)
            },
            // Learner-only and full-search-only; see make_search_meta.
            dirichlet_epsilon: if full && cfg.net_by_player[actor] == 0 {
                cfg.dirichlet_epsilon
            } else {
                0.0
            },
            dirichlet_alpha: cfg.dirichlet_alpha,
            age_deal_samples: cfg.age_deal_samples,
            double_reveal_offsets: cheap_offsets(
                cfg.cheap_double_reveal_offsets_by_player
                    .map_or(cfg.cheap_double_reveal_offsets, |per_seat| per_seat[actor]),
                full,
            ),
            conflict_free_waves: cfg.conflict_free_waves,
            round_robin_candidates: cfg.round_robin_candidates,
        };
        let leaf_batch = cfg
            .leaf_batch_by_player
            .map_or(cfg.leaf_batch, |batches| batches[actor]);
        let (result, _, _) =
            tree_resumable::search_closed_batched(&state, &eval, &search_cfg, leaf_batch)?;
        // The solve runs on the PRE-move state, and before the action is chosen:
        // the mask is what makes a provably-losing move unplayable, not merely
        // unlabelled. EVERY eligible ply is solved, cheap ones included -- the
        // mask is what keeps both sides playing the endgame provably optimally,
        // which is what makes the game's realised result equal the proven value.
        // Cheap plies still emit no training example; the label is a separate
        // concern from the mask.
        // TWO distributions from one search, and they are not the same object.
        //
        // `selection` decides the move actually played. `training` is what the
        // buffer records. They start identical and diverge under policy-target
        // pruning, whose entire purpose is to clean the LABEL while leaving the
        // forced exploration in the trajectory -- pruning `selection` would
        // delete that exploration from the games themselves.
        //
        // A proof, by contrast, belongs in both: a provably-losing move should
        // be neither played nor taught. So `endgame_overlay` hands back the
        // proven-optimal set and both distributions are masked from the one
        // solve, in the order PRUNE -> MASK -> RENORMALISE.
        let mut selection = result.policy_target.clone();
        // Pruned when the search forced playouts, identical otherwise. The
        // search owns this: it is the only place that holds the noised priors
        // and the per-action Q the rule needs.
        let mut training = result.training_policy.unwrap_or(result.policy_target);
        let overlay = if cfg.solve_endgames {
            endgame_overlay(&state, &legal)
        } else {
            None
        };
        if let Some(keep) = overlay.as_ref().and_then(|o| o.keep.as_ref()) {
            mask_and_renormalise(&mut selection, keep);
            mask_and_renormalise(&mut training, keep);
        }
        let action = if cfg.deterministic_actions {
            best_policy_action(&legal, &selection)
        } else {
            sample_policy(&legal, &selection, temperature(i), &mut rng)
        };
        trajectory.update(&state);
        chance_log.extend(actual_chance_outcomes(&state, action, i)?);
        state.apply_action(&decode_action(&state, action));
        moves.push(MoveRecord {
            i,
            actor,
            action,
            legal,
            // Visits stay the raw search evidence, unmasked: reanalyze and the
            // target diagnostics read them to reconstruct what the search
            // actually did, which the masked distribution no longer shows.
            visits: result.visits,
            policy_target: training,
            prior: result.prior,
            root_value: result.root_value,
            action_value: result.action_value,
            net_root_value: result.net_root_value,
            sims: result.sims,
            gumbel_topk: result.gumbel_topk,
            policy_excluded: !full,
            full_search: full,
            search_seed,
            is_bot: false,
            solver_value: overlay.as_ref().and_then(|o| o.value),
            solver_regime: overlay.as_ref().and_then(|o| o.regime),
            solver_attempted: overlay.is_some(),
            solver_stop: overlay.as_ref().and_then(|o| o.stop),
            solver_nodes: overlay.as_ref().map_or(0, |o| o.nodes),
            solver_masked: overlay.is_some_and(|o| o.masked),
        });
    }

    let final_digest = crate::digest::state_digest(&state);
    trajectory.update(&state);
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
        digest_version: crate::digest::VERSION,
        final_digest,
        trajectory_digest: trajectory.finish(),
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
    pub boundary_tokens_sq: usize,
    pub boundary_padded_tokens_sq: usize,
    pub boundary_feature_values_used: usize,
    pub boundary_feature_values_written: usize,
    pub encode_pack_ns: u64,
    pub queue_wait_ns: u64,
    pub py_call_ns: u64,
    pub extract_ns: u64,
    /// Worker sub-partition, mirrored from `eval::BoundaryMetrics`.
    pub attach_ns: u64,
    pub payload_ns: u64,
    pub validate_ns: u64,
    pub metrics_ns: u64,
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
    // --- Phase 1a: partition the scheduler thread ---
    // `CORE_UTILIZATION_PLAN.md` §11/§13/§14 measured a residual of 18-25% of
    // non-device time, stable across two machines, three widths and two
    // precisions. It was computed as `wall - (rust terms + py_call_ns)`, which
    // is not a partition of anything: `py_call_ns` is measured on the *worker*
    // thread while the rest are the scheduler's. These six are all taken on the
    // scheduler thread and together with `scatter_ns` they tile the loop, so the
    // leftover is real bookkeeping rather than an artifact of mixing threads.
    //
    // The 2.01x/3.0x Phase 1+2 estimates assume the residual parallelises with
    // the rest of the thread. That is what these settle.
    /// `pool.refill`: activating queued games, i.e. game construction.
    pub sched_refill_ns: u64,
    /// `collect_ready_groups`. **Inclusive** of the per-slot tree, chance and
    /// encode_pack work already counted separately -- subtract those to get the
    /// collection overhead itself.
    pub sched_collect_ns: u64,
    /// `pool.retire`: finishing games and building records.
    pub sched_retire_ns: u64,
    /// Wall time the scheduler spent blocked on the solver pool -- i.e. the
    /// stall async did NOT remove. The number that says whether more solver
    /// threads would help.
    pub sched_solve_wait_ns: u64,
    /// Cloning every row's states/actors/legals/net_ids into the batch vectors.
    /// Untimed until now and a prime suspect: it copies the whole batch.
    pub sched_assemble_ns: u64,
    /// Handing the assembled batch to the worker.
    pub sched_submit_ns: u64,
    /// `ticket.wait()`: what the scheduler thread actually spends blocked on the
    /// forward. This, not `py_call_ns`, is the serialisation Phase 1 removes.
    pub sched_wait_ns: u64,
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
        self.boundary_tokens_sq += other.boundary_tokens_sq;
        self.boundary_padded_tokens_sq += other.boundary_padded_tokens_sq;
        self.boundary_feature_values_used += other.boundary_feature_values_used;
        self.boundary_feature_values_written += other.boundary_feature_values_written;
        self.encode_pack_ns += other.encode_pack_ns;
        self.queue_wait_ns += other.queue_wait_ns;
        self.py_call_ns += other.py_call_ns;
        self.extract_ns += other.extract_ns;
        self.attach_ns += other.attach_ns;
        self.payload_ns += other.payload_ns;
        self.validate_ns += other.validate_ns;
        self.metrics_ns += other.metrics_ns;
        self.rust_tree_ns += other.rust_tree_ns;
        self.rust_chance_ns += other.rust_chance_ns;
        self.rust_record_ns += other.rust_record_ns;
        self.scatter_ns += other.scatter_ns;
        // Per-thread durations, so they sum across shards exactly like the other
        // `*_ns` work counters. Only `scheduler_wall_ns` is an envelope.
        self.sched_refill_ns += other.sched_refill_ns;
        self.sched_collect_ns += other.sched_collect_ns;
        self.sched_retire_ns += other.sched_retire_ns;
        self.sched_assemble_ns += other.sched_assemble_ns;
        self.sched_submit_ns += other.sched_submit_ns;
        self.sched_wait_ns += other.sched_wait_ns;
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

    /// Activations currently available beyond each shard's reservation. Test
    /// surface: the accounting invariants are not otherwise observable.
    pub fn spare_for_test(&self) -> usize {
        self.spare.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Games currently active across all shards. `peak_live` is the shipped
    /// metric; the instantaneous count is what abort/retire accounting is
    /// asserted against.
    pub fn live_for_test(&self) -> usize {
        self.live.load(std::sync::atomic::Ordering::Acquire)
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
    /// A finished shard's reservation is given back exactly once.
    donated_reservation: bool,
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
            donated_reservation: false,
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
                    // The activation was already accounted for; hand it back, or
                    // a failed construction strands a token for the whole run.
                    self.release_activation(budget);
                    return Err(err);
                }
            }
            self.active_count += 1;
            budget.on_activate();
            activated += 1;
        }
        Ok(activated)
    }

    /// Give back one activation: the shared token if this slot borrowed one,
    /// otherwise the shard's own reservation.
    fn release_activation(&mut self, budget: &SlotBudget) {
        if self.budget_held > 0 {
            self.budget_held -= 1;
            budget.give_back();
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
        self.release_activation(budget);
        self.donate_reservation_if_finished(budget);
        Ok(())
    }

    /// Once a shard can never activate again, hand its reserved activation to
    /// the shards still running.
    ///
    /// Each shard keeps one reservation so it cannot be starved into a deadlock,
    /// but that reservation is dead weight after its queue empties: without this,
    /// total usable concurrency shrinks by one for every shard that finishes.
    fn donate_reservation_if_finished(&mut self, budget: &SlotBudget) {
        if self.donated_reservation
            || self.holds_reserved
            || self.active_count > 0
            || self.next_queued < self.entries.len()
        {
            return;
        }
        self.donated_reservation = true;
        budget.give_back();
    }

    /// Abandon this shard's work and return everything it holds to the budget.
    ///
    /// Cancelling pending evaluations is not enough on an error path: the shard
    /// still holds one activation per active slot — borrowed tokens, its own
    /// reservation, and the global `live` count. Shards run concurrently under
    /// `std::thread::scope` and a failing one is only joined at the end, so
    /// anything it keeps starves its siblings for the rest of the run.
    ///
    /// Idempotent: retired entries are replaced, so a second call finds nothing
    /// to release.
    fn abort(&mut self, budget: &SlotBudget) {
        for index in 0..self.entries.len() {
            let SlotEntry::Active(slot) = &mut self.entries[index] else {
                continue;
            };
            slot.cancel_pending();
            self.entries[index] = SlotEntry::Taken;
            self.active_count -= 1;
            // Exactly the pair `retire` performs, minus the record: one global
            // live count and one activation (borrowed token or reservation).
            budget.on_retire();
            self.release_activation(budget);
        }
        // An activation taken but never turned into an active slot: `refill`
        // reserves before it constructs, and the queue-pointer invariant check
        // returns in between. Rare, but this is the path that is supposed to
        // hold under errors, so it drains rather than assuming.
        while self.budget_held > 0 {
            self.release_activation(budget);
        }
        // The queue can never be drained now, so the reservation is dead weight
        // for the same reason a finished shard's is.
        self.holds_reserved = false;
        self.next_queued = self.entries.len();
        self.donate_reservation_if_finished(budget);
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
    /// The search is done and a solve is out with the pool. The slot cannot
    /// proceed -- the mask decides which move is played -- but the scheduler
    /// thread can, which is the entire point of the async path.
    SolvePending {
        meta: SearchMeta,
        result: crate::tree::SearchResult,
    },
    Complete,
}

struct GameSlot {
    state: GameState,
    rng: Rng,
    cfg: SelfPlayConfig,
    moves: Vec<MoveRecord>,
    chance_log: Vec<ChanceRecord>,
    trajectory: crate::digest::TrajectoryDigest,
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
    /// Network id for every row in this group. A group comes from one slot and
    /// one search, so all its rows share a searcher and therefore one id --
    /// which is why this can be a single value fanned out rather than derived
    /// per leaf.
    net_id: u8,
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
            trajectory: crate::digest::TrajectoryDigest::new(),
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
                forced_playout_k: if full && self.cfg.net_by_player[actor] == 0 {
                    // Gated exactly like `dirichlet_epsilon` below, and for the same
                    // reasons. Forcing spends simulations on children PUCT declined,
                    // which is a handicap: on an archived opponent it inflates the
                    // learner's league win rate and skews those games' value labels
                    // optimistic, and on a cheap move it degrades the trajectory for
                    // no label benefit, since cheap moves emit no policy target.
                    //
                    // It also makes GATES immune for free. A gate runs
                    // `full_search_fraction = 0.0`, so `full` is always false there
                    // -- which matters because the advisor's `RustPuctSearch` takes
                    // this as an explicit parameter defaulting to 0.0 and never
                    // reads the global. Ungated, a gate would search with forcing
                    // while the advisor did not, reintroducing the very
                    // gate/advisor divergence the PUCT switch exists to close.
                    forced_playout_k()
                } else {
                    0.0
                },
                sims,
                top_k: search_top_k(self.cfg.top_k, cheap_top_k(), full),
                c_puct: self.cfg.c_puct,
                c_visit: self.cfg.c_visit,
                c_scale: self.cfg.c_scale,
                seed: search_seed,
                force_expand_root_chance: self.cfg.force_expand_root_chance,
                puct_root: if full {
                    self.cfg.puct_root
                } else {
                    self.cfg.cheap_puct_root.unwrap_or(self.cfg.puct_root)
                },
                // Only network 0 -- the learner -- explores. An archived
                // opponent must play noise-free: handicapping it inflates the
                // learner's league win rate, and league games are ~15% of the
                // training data, so their value labels would skew optimistic.
                // Same predicate the buffer already uses to drop an archived
                // net's policy targets (`policy_excluded`, below).
                // Learner-only AND full-search-only. An archived opponent must
                // play clean (league games are ~15% of training data, so a
                // handicapped opponent skews their value labels optimistic), and
                // cheap moves emit no targets at all -- noise there buys no
                // exploration of the label space and only degrades the
                // trajectory. Kingdomino settles both the same way
                // (`hof_dirichlet_epsilon` / `fast_move_dirichlet_epsilon` = 0).
                dirichlet_epsilon: if full && self.cfg.net_by_player[actor] == 0 {
                    self.cfg.dirichlet_epsilon
                } else {
                    0.0
                },
                dirichlet_alpha: self.cfg.dirichlet_alpha,
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
                round_robin_candidates: self.cfg.round_robin_candidates,
            },
        })
    }

    fn next_eval_group(
        &mut self,
        slot_index: usize,
        forced_row_limit: usize,
        mut pool: Option<&mut SolverPool>,
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
                        // The root row's actor IS the searcher.
                        net_id: self.cfg.net_by_player[meta.actor],
                        kind: EvalGroupKind::Root,
                    }));
                }
                SlotStage::Complete => return Ok(None),
                // Waiting on the pool; nothing to evaluate from here.
                SlotStage::SolvePending { .. } => return Ok(None),
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
                                // Deliberately the SEARCHER, not the leaf actor:
                                // `self.state` is the real game state this search
                                // is deciding a move for, so its actor owns every
                                // leaf in the tree regardless of tree depth.
                                net_id: self.cfg.net_by_player
                                    [crate::tree::state_actor(&self.state)],
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
            self.finish_move(meta, result, slot_index, pool.as_deref_mut())?;
            if matches!(self.stage, SlotStage::SolvePending { .. }) {
                // Parked on a solve: this slot has nothing to evaluate until the
                // answer lands, and the caller must not treat that as finished.
                return Ok(None);
            }
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

    /// Solve if this slot wants one, then complete the move.
    ///
    /// Split from `complete_move` so the solve can move off the scheduler
    /// thread: everything before the overlay is search bookkeeping, everything
    /// after depends on the answer. Synchronous here; the pool path parks the
    /// slot between the two halves instead.
    fn finish_move(
        &mut self,
        meta: SearchMeta,
        result: crate::tree::SearchResult,
        slot_index: usize,
        pool: Option<&mut SolverPool>,
    ) -> PyResult<()> {
        if !self.cfg.solve_endgames {
            return self.complete_move(meta, result, None);
        }
        if let Some(pool) = pool {
            // Only park for a solve the trigger would actually take. Dispatching
            // an ineligible position would cost a channel round trip per move to
            // learn what `solver_eligible` answers in nanoseconds.
            if solver_wants(&self.state) {
                let dispatched = pool.dispatch(SolveJob {
                    slot: slot_index,
                    state: self.state.clone(),
                    legal: meta.legal.clone(),
                });
                if dispatched {
                    self.stage = SlotStage::SolvePending { meta, result };
                    return Ok(());
                }
                // The pool is gone; fall through and solve inline rather than
                // silently drop the mask, which would change the targets.
            }
        }
        let overlay = endgame_overlay(&self.state, &meta.legal);
        self.complete_move(meta, result, overlay)
    }

    /// Apply a solve that came back from the pool.
    fn resume_after_solve(&mut self, overlay: Option<SolverOverlay>) -> PyResult<()> {
        let stage = std::mem::replace(&mut self.stage, SlotStage::Complete);
        let SlotStage::SolvePending { meta, result } = stage else {
            return Err(PyRuntimeError::new_err(
                "solve outcome delivered to a slot that was not waiting for one",
            ));
        };
        self.complete_move(meta, result, overlay)
    }

    /// The half that needs the solver's answer. `overlay` is whatever the solve
    /// produced -- synchronously or from the pool; the two must be
    /// indistinguishable in the record, which is the property the async port is
    /// gated on.
    fn complete_move(
        &mut self,
        meta: SearchMeta,
        result: crate::tree::SearchResult,
        overlay: Option<SolverOverlay>,
    ) -> PyResult<()> {
        let i = self.moves.len();
        // See `run`: pre-move state, before the action is chosen, EVERY eligible
        // ply. The solve is synchronous, so it holds this scheduler slot (and
        // the shard's thread) for its duration -- which is why the budget is a
        // node count and why the card trigger is set from the measured table
        // rather than optimistically.
        // See `run` for why selection and training are separate objects.
        let mut selection = result.policy_target.clone();
        let mut training = result.training_policy.unwrap_or(result.policy_target);
        if let Some(keep) = overlay.as_ref().and_then(|o| o.keep.as_ref()) {
            mask_and_renormalise(&mut selection, keep);
            mask_and_renormalise(&mut training, keep);
        }
        let action = if self.cfg.deterministic_actions {
            best_policy_action(&meta.legal, &selection)
        } else {
            sample_policy(&meta.legal, &selection, temperature(i), &mut self.rng)
        };
        let digest_started = Instant::now();
        self.trajectory.update(&self.state);
        self.record_ns += digest_started.elapsed().as_nanos() as u64;
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
            policy_target: training,
            prior: result.prior,
            root_value: result.root_value,
            action_value: result.action_value,
            net_root_value: result.net_root_value,
            sims: result.sims,
            gumbel_topk: result.gumbel_topk,
            // Cheap searches are excluded as always, and so is anything the
            // archived opponent searched: network 0 is by definition the
            // learner, so a policy target produced by network 1 would train the
            // learner to imitate an older, weaker net. The move still records
            // its visits and the game still supplies a value target -- only the
            // policy label is withheld, exactly as for curriculum-bot moves.
            // Kingdomino's `play_current_vs_hof_game` keeps "only current-owned
            // labels" the same way.
            policy_excluded: !meta.full || self.cfg.net_by_player[meta.actor] != 0,
            full_search: meta.full,
            search_seed: meta.search_seed,
            is_bot: false,
            solver_value: overlay.as_ref().and_then(|o| o.value),
            solver_regime: overlay.as_ref().and_then(|o| o.regime),
            solver_attempted: overlay.is_some(),
            solver_stop: overlay.as_ref().and_then(|o| o.stop),
            solver_nodes: overlay.as_ref().map_or(0, |o| o.nodes),
            solver_masked: overlay.is_some_and(|o| o.masked),
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
        let digest_started = Instant::now();
        self.trajectory.update(&self.state);
        self.record_ns += digest_started.elapsed().as_nanos() as u64;
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
            prior: Vec::new(),
            root_value: 0.0,
            action_value: 0.0,
            net_root_value: 0.0,
            sims: 0,
            gumbel_topk: Vec::new(),
            policy_excluded: self
                .cfg
                .iteration
                .is_some_and(|iteration| iteration >= self.cfg.bot_policy_iterations),
            full_search: false,
            search_seed: 0,
            is_bot: true,
            // A curriculum bot's move carries no solve: its policy label is
            // "imitate the bot", which a mask would contradict rather than
            // sharpen, and there is no search whose ranking survives masking.
            solver_value: None,
            solver_regime: None,
            solver_attempted: false,
            solver_stop: None,
            solver_nodes: 0,
            solver_masked: false,
        });
        self.stage = if self.state.phase == Phase::Complete {
            SlotStage::Complete
        } else {
            SlotStage::NeedRoot(self.make_search_meta()?)
        };
        Ok(())
    }

    /// Parked waiting on the solver pool. Distinct from finished: a slot that
    /// yields no evaluation group is normally done with its game, and retiring
    /// one that is merely waiting would discard a game mid-solve.
    fn is_parked(&self) -> bool {
        matches!(self.stage, SlotStage::SolvePending { .. })
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

    fn into_record(mut self) -> PyResult<GameRecord> {
        if self.state.phase != Phase::Complete || !matches!(self.stage, SlotStage::Complete) {
            return Err(PyRuntimeError::new_err(
                "scheduler attempted to emit an incomplete game",
            ));
        }
        let final_digest = crate::digest::state_digest(&self.state);
        self.trajectory.update(&self.state);
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
            digest_version: crate::digest::VERSION,
            final_digest,
            trajectory_digest: self.trajectory.finish(),
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
            pool.abort(&budget);
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
                .and_then(|slot| slot.next_eval_group(slot_index, forced_row_limit, None));
            let parked = pool.slot_mut(slot_index).map(|s| s.is_parked()).unwrap_or(false);
            match outcome {
                Ok(Some(group)) => groups.push(group),
                // No group means the game is over -- UNLESS the slot is parked on
                // a solve, which yields nothing and is emphatically not finished.
                Ok(None) if parked => {}
                Ok(None) => retire.push(slot_index),
                Err(err) => {
                    pool.abort(&budget);
                    return Err(err);
                }
            }
        }
        for slot_index in retire {
            if let Err(err) = pool.retire(slot_index, &mut metrics, &budget) {
                pool.abort(&budget);
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
            pool.abort(&budget);
            return Err(PyRuntimeError::new_err(
                "cooperative scheduler made no progress with live slots",
            ));
        }

        let mut pending = std::collections::VecDeque::from(groups);
        while !pending.is_empty() {
            let (batch, row_count) = match take_global_batch(&mut pending, global_batch_cap) {
                Ok(taken) => taken,
                Err(err) => {
                    pool.abort(&budget);
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
            // One id per row, fanned out from each group's searcher. Left empty
            // when every group is on network 0, so ordinary self-play submits the
            // identical payload it always did.
            let net_ids: Vec<u8> = if batch.iter().any(|group| group.net_id != 0) {
                batch
                    .iter()
                    .flat_map(|group| std::iter::repeat(group.net_id).take(group.states.len()))
                    .collect()
            } else {
                Vec::new()
            };
            metrics.batch_live_slots.push(pool.active_count());
            metrics.batch_submit_ns.push(occupancy.elapsed_ns());
            let evaluations = match evaluator
                .evaluate_batch_prepared_routed(&state_refs, &actors, &legals, &net_ids)
            {
                Ok(rows) => rows,
                Err(err) => {
                    pool.abort(&budget);
                    return Err(err);
                }
            };
            if evaluations.len() != row_count {
                pool.abort(&budget);
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
                pool.abort(&budget);
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
    mut solver: Option<&mut SolverPool>,
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
            .next_eval_group(slot_index, forced_row_limit, solver.as_deref_mut())?
        {
            Some(group) => {
                outstanding[slot_index] = true;
                pending.push_back(group);
                collected += 1;
            }
            // Parked on a solve yields no group and is not a finished game.
            None if pool.slot_mut(slot_index)?.is_parked() => {}
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
    // One pool per scheduler loop, so shards do not contend on a shared queue.
    // Zero threads means synchronous solving, which is the default.
    let solver_threads = solver_threads();
    let mut solver = (solver_threads > 0).then(|| SolverPool::new(solver_threads));
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
        let refill_started = Instant::now();
        if let Err(err) = pool.refill(budget) {
            pool.abort(&budget);
            return Err(err);
        }
        metrics.sched_refill_ns += refill_started.elapsed().as_nanos() as u64;
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
        let collect_started = Instant::now();
        if let Err(err) = collect_ready_groups(
            &mut pool,
            &mut outstanding,
            &mut pending,
            global_batch_cap,
            &mut finished,
            solver.as_mut(),
        ) {
            pool.abort(&budget);
            return Err(err);
        }
        metrics.sched_collect_ns += collect_started.elapsed().as_nanos() as u64;
        let retire_started = Instant::now();
        for slot_index in finished {
            if let Err(err) = pool.retire(slot_index, &mut metrics, budget) {
                pool.abort(&budget);
                return Err(err);
            }
        }
        metrics.sched_retire_ns += retire_started.elapsed().as_nanos() as u64;

        // Pump the solver pool. Port of KD's `pump_async_solves`: harvest what
        // has already landed, then block for an outcome ONLY when nothing else
        // can move -- no group waiting to be batched and no batch in flight.
        // Waiting on a solve while a leaf evaluation could be gathered instead
        // is precisely the stall this path exists to remove.
        if let Some(active_solver) = solver.as_mut() {
            let solve_started = Instant::now();
            loop {
                for outcome in active_solver.harvest() {
                    let slot = match pool.slot_mut(outcome.slot) {
                        Ok(slot) => slot,
                        Err(err) => {
                            pool.abort(&budget);
                            return Err(err);
                        }
                    };
                    if let Err(err) = slot.resume_after_solve(outcome.overlay) {
                        pool.abort(&budget);
                        return Err(err);
                    }
                }
                let stuck = active_solver.inflight() > 0
                    && pending.is_empty()
                    && inflight.is_empty();
                if !stuck {
                    break;
                }
                let Some(outcome) = active_solver.wait_one() else {
                    break;
                };
                let slot = match pool.slot_mut(outcome.slot) {
                    Ok(slot) => slot,
                    Err(err) => {
                        pool.abort(&budget);
                        return Err(err);
                    }
                };
                if let Err(err) = slot.resume_after_solve(outcome.overlay) {
                    pool.abort(&budget);
                    return Err(err);
                }
                // A resumed slot may now have work, so re-collect before
                // deciding to block again.
                let mut more = Vec::new();
                if let Err(err) = collect_ready_groups(
                    &mut pool,
                    &mut outstanding,
                    &mut pending,
                    global_batch_cap,
                    &mut more,
                    None,
                ) {
                    pool.abort(&budget);
                    return Err(err);
                }
                for slot_index in more {
                    if let Err(err) = pool.retire(slot_index, &mut metrics, budget) {
                        pool.abort(&budget);
                        return Err(err);
                    }
                }
            }
            metrics.sched_solve_wait_ns += solve_started.elapsed().as_nanos() as u64;
        }

        while inflight.len() < max_inflight_batches && !pending.is_empty() {
            let (groups, row_count) = match take_global_batch(&mut pending, global_batch_cap) {
                Ok(batch) => batch,
                Err(err) => {
                    pool.abort(&budget);
                    return Err(err);
                }
            };
            let assemble_started = Instant::now();
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
            // Same fan-out as `run_many`: one id per row from each group's
            // searcher, empty when nothing is routed. This path is the
            // production one (`max_inflight_batches > 1`), so omitting it here
            // would leave league routing working only in tests.
            let net_ids: Vec<u8> = if groups.iter().any(|group| group.net_id != 0) {
                groups
                    .iter()
                    .flat_map(|group| std::iter::repeat(group.net_id).take(group.states.len()))
                    .collect()
            } else {
                Vec::new()
            };
            metrics.sched_assemble_ns += assemble_started.elapsed().as_nanos() as u64;
            let submit_started = Instant::now();
            let ticket = match worker.submit_prepared_routed(
                owned_states,
                actors,
                legals,
                net_ids,
            ) {
                Ok(ticket) => ticket,
                Err(err) => {
                    pool.abort(&budget);
                    return Err(err);
                }
            };
            metrics.sched_submit_ns += submit_started.elapsed().as_nanos() as u64;
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
            pool.abort(&budget);
            return Err(PyRuntimeError::new_err(
                "pipelined scheduler made no progress with live slots",
            ));
        };
        let wait_started = Instant::now();
        let evaluations = match flight.ticket.wait() {
            Ok(rows) => rows,
            Err(err) => {
                pool.abort(&budget);
                return Err(err);
            }
        };
        metrics.sched_wait_ns += wait_started.elapsed().as_nanos() as u64;
        if evaluations.len() != flight.row_count {
            pool.abort(&budget);
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
            pool.abort(&budget);
            return Err(err);
        }
    }

    occupancy.tick(&mut metrics, &pool, &outstanding, local_capacity);
    metrics.scheduler_wall_ns = occupancy.elapsed_ns();
    // Exact and global: with shards this is concurrency across all of them, not
    // this shard's own peak.
    metrics.max_live_slots = budget.peak_live();
    let records = pool.into_records()?;
    metrics.games = records.len();
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

#[cfg(test)]
mod budget_tests {
    use super::*;

    fn pool_with(jobs: usize) -> SlotPool {
        // Entries are never activated in these tests; only the accounting is
        // under test, so empty queued jobs would need a GameState. Build the
        // pool directly with the queue pointer pre-positioned instead.
        let mut pool = SlotPool::new(Vec::new());
        pool.entries = Vec::new();
        pool.next_queued = 0;
        let _ = jobs;
        pool
    }

    #[test]
    fn spare_starts_at_total_minus_one_reservation_per_shard() {
        let budget = SlotBudget::new(8, 3).expect("valid budget");
        assert_eq!(budget.spare_for_test(), 5);
        assert_eq!(budget.total(), 8);
    }

    #[test]
    fn budget_below_shard_count_is_rejected() {
        assert!(SlotBudget::new(2, 4).is_err());
        assert!(SlotBudget::new(4, 4).is_ok());
    }

    #[test]
    fn releasing_a_borrowed_activation_returns_the_token() {
        // The bug this guards: a failed GameSlot::new released the local counter
        // without giving the shared token back, stranding capacity for the run.
        let budget = SlotBudget::single(4).expect("valid budget");
        let mut pool = pool_with(0);
        pool.holds_reserved = true;
        assert!(budget.try_take());
        pool.budget_held = 1;
        assert_eq!(budget.spare_for_test(), 2);

        pool.release_activation(&budget);
        assert_eq!(pool.budget_held, 0);
        assert_eq!(budget.spare_for_test(), 3, "borrowed token must come back");

        // The reservation itself is local and must NOT inflate the shared spare.
        pool.release_activation(&budget);
        assert!(!pool.holds_reserved);
        assert_eq!(budget.spare_for_test(), 3);
    }

    #[test]
    fn a_finished_shard_donates_its_reservation() {
        // Without this, usable concurrency shrinks by one per finished shard.
        let budget = SlotBudget::new(4, 2).expect("valid budget");
        assert_eq!(budget.spare_for_test(), 2);
        let mut pool = pool_with(0);
        pool.holds_reserved = false;
        pool.active_count = 0;
        pool.next_queued = 0; // entries is empty: the queue is exhausted

        pool.donate_reservation_if_finished(&budget);
        assert_eq!(budget.spare_for_test(), 3, "reservation donated once");
        pool.donate_reservation_if_finished(&budget);
        assert_eq!(budget.spare_for_test(), 3, "and only once");
    }

    #[test]
    fn a_shard_still_holding_work_keeps_its_reservation() {
        let budget = SlotBudget::new(4, 2).expect("valid budget");
        let mut pool = pool_with(0);
        pool.holds_reserved = true;
        pool.donate_reservation_if_finished(&budget);
        assert_eq!(budget.spare_for_test(), 2, "still active: no donation");
    }

    fn sample_setup() -> crate::state::Setup {
        crate::state::Setup {
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

    fn sample_config(game_seed: u64) -> SelfPlayConfig {
        SelfPlayConfig {
            solve_endgames: true,
            cheap_puct_root: None,
            game_seed,
            iteration: None,
            leaf_batch: 1,
            leaf_batch_by_player: None,
            deterministic_actions: false,
            cheap_sims_min: 2,
            cheap_sims_max: 2,
            full_sims_min: 2,
            full_sims_max: 2,
            full_search_fraction: 0.0,
            top_k: 2,
            draft_prior: 0.0,
            c_puct: 1.25,
            c_visit: 50.0,
            c_scale: 1.0,
            force_expand_root_chance: false,
            puct_root: false,
            dirichlet_epsilon: 0.0,
            dirichlet_alpha: 1.8,
            age_deal_samples: 0,
            age_deal_samples_by_player: None,
            cheap_double_reveal_offsets: 0,
            cheap_double_reveal_offsets_by_player: None,
            bot_by_player: [None, None],
            net_by_player: [0, 0],
            bot_exploration: 0.0,
            bot_policy_iterations: 1,
            max_moves: 512,
            conflict_free_waves: false,
            round_robin_candidates: false,
        }
    }

    fn queued_jobs(count: usize) -> Vec<(GameState, SelfPlayConfig)> {
        (0..count)
            .map(|index| {
                (
                    GameState::from_setup(sample_setup(), std::collections::VecDeque::new()),
                    sample_config(index as u64 + 1),
                )
            })
            .collect()
    }

    #[test]
    fn aborting_a_shard_returns_every_activation_it_holds() {
        // The bug this guards: on a mid-run error `cancel_all` stopped the
        // pending work but kept the shard's borrowed tokens, its reservation and
        // its share of the global live count. Shards are joined only at the end
        // of the scope, so the surviving ones ran the rest of the job at reduced
        // concurrency.
        let budget = SlotBudget::new(6, 2).expect("valid budget");
        assert_eq!(budget.spare_for_test(), 4, "6 total minus one per shard");

        let mut pool = SlotPool::new(queued_jobs(3));
        let activated = pool.refill(&budget).expect("activation succeeds");
        assert_eq!(activated, 3);
        assert_eq!(pool.active_count(), 3);
        assert_eq!(budget.live_for_test(), 3);
        assert_eq!(budget.spare_for_test(), 2, "one reserved + two borrowed");

        pool.abort(&budget);

        assert_eq!(pool.active_count(), 0);
        assert_eq!(budget.live_for_test(), 0, "global live count restored");
        assert_eq!(
            budget.spare_for_test(),
            5,
            "both borrowed tokens back, plus the reservation donated because \
             an aborted shard can never activate again"
        );

        // Idempotent: every call site aborts then returns, but a second call
        // must not manufacture budget.
        pool.abort(&budget);
        assert_eq!(budget.spare_for_test(), 5);
        assert_eq!(budget.live_for_test(), 0);
    }

    #[test]
    fn aborting_releases_an_activation_taken_but_never_activated() {
        // `refill` takes the activation before constructing the slot, and the
        // queue-pointer invariant check can return in between. That token
        // belongs to no active slot, so the per-slot loop alone would miss it.
        let budget = SlotBudget::new(4, 2).expect("valid budget");
        let mut pool = SlotPool::new(queued_jobs(2));
        // Corrupt the queue exactly as the invariant error describes.
        pool.entries[0] = SlotEntry::Taken;

        assert!(pool.refill(&budget).is_err(), "the invariant must be caught");
        pool.abort(&budget);

        assert_eq!(budget.live_for_test(), 0);
        assert_eq!(
            budget.spare_for_test(),
            3,
            "the reservation is donated and no borrowed token is stranded"
        );
        assert_eq!(pool.budget_held, 0);
    }

    #[test]
    fn aborting_leaves_a_sibling_shard_its_full_share() {
        // The consequence the fix exists for, stated as an invariant: after one
        // shard aborts, everything it held is available to the other.
        let budget = SlotBudget::new(4, 2).expect("valid budget");
        let mut failing = SlotPool::new(queued_jobs(2));
        failing.refill(&budget).expect("activation succeeds");
        assert_eq!(budget.spare_for_test(), 1);

        failing.abort(&budget);

        let mut survivor = SlotPool::new(queued_jobs(3));
        let activated = survivor.refill(&budget).expect("activation succeeds");
        assert_eq!(
            activated, 3,
            "the survivor must reach the full budget, not a shrunken one"
        );
        assert_eq!(budget.live_for_test(), 3);
    }
}

pub fn component_name(kind: ChanceKind, id: usize) -> &'static str {
    match kind {
        ChanceKind::CardReveal | ChanceKind::AgeDeal => data::card(id).name,
        ChanceKind::GreatLibraryDraw => data::progress(id).name,
        ChanceKind::WonderGroupReveal => data::wonder(id).name,
    }
}

#[cfg(test)]
mod solver_overlay_tests {
    //! The mask's arithmetic, separated from the solve that produces it.
    //! Whether the solver is *right* is `endgame_corpus.py`'s job; this is the
    //! part that turns a right answer into a training label.

    use super::mask_and_renormalise;

    #[test]
    fn losing_moves_lose_their_mass_and_the_survivors_keep_their_ranking() {
        let mut policy = vec![0.5, 0.3, 0.2];
        mask_and_renormalise(&mut policy, &[false, true, true]);
        assert_eq!(policy[0], 0.0);
        // 0.3 : 0.2 before, 0.6 : 0.4 after -- the search's discrimination among
        // proven-equal moves is exactly what the mask must not flatten.
        assert!((policy[1] - 0.6).abs() < 1e-12);
        assert!((policy[2] - 0.4).abs() < 1e-12);
        assert!((policy.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn an_all_optimal_position_is_left_alone() {
        let mut policy = vec![0.25, 0.25, 0.5];
        mask_and_renormalise(&mut policy, &[true, true, true]);
        assert_eq!(policy, vec![0.25, 0.25, 0.5]);
    }

    #[test]
    fn a_search_that_backed_only_losing_moves_falls_back_to_uniform() {
        // Not defensive: dividing the survivors' zero mass by itself would put
        // NaN into a policy label, which trains as a silent poison rather than
        // an error.
        let mut policy = vec![1.0, 0.0, 0.0];
        mask_and_renormalise(&mut policy, &[false, true, true]);
        assert_eq!(policy, vec![0.0, 0.5, 0.5]);
    }
}

// ---------------------------------------------------------------------------
// Async endgame solving — ported from Kingdomino's `BatchedMCTS` (kingdomino_rust
// `lib.rs`, `async_solve` / `pump_async_solves`).
// ---------------------------------------------------------------------------

/// One position handed to the background solver.
pub struct SolveJob {
    pub slot: usize,
    pub state: GameState,
    pub legal: Vec<usize>,
}

/// The answer, tagged with the slot that asked. `overlay` is `None` when the
/// trigger declined before any solving — the same meaning `endgame_overlay`
/// gives it.
pub struct SolveOutcome {
    pub slot: usize,
    pub overlay: Option<SolverOverlay>,
}

/// A background solver thread plus its channels.
///
/// The solve must finish before the move is chosen -- the mask decides what is
/// played -- so the SLOT genuinely cannot proceed. What this buys is that the
/// other slots do: today a multi-second solve holds the scheduler loop, and
/// while it runs no leaf evaluations are gathered from any of that worker's
/// slots, so the evaluation boundary starves too.
///
/// Structure follows KD's, including the parts that look odd:
///
/// * the thread is spawned **unconditionally**, even when async is off, so the
///   disabled path has no conditional-lifetime field and stays structurally
///   identical;
/// * `inflight` counts dispatched-but-unharvested jobs, which is what tells the
///   pump whether blocking is even necessary.
pub struct SolverPool {
    job_tx: Option<std::sync::mpsc::Sender<SolveJob>>,
    out_rx: std::sync::mpsc::Receiver<SolveOutcome>,
    handle: Option<std::thread::JoinHandle<()>>,
    inflight: usize,
}

impl SolverPool {
    pub fn new(threads: usize) -> Self {
        let (job_tx, job_rx) = std::sync::mpsc::channel::<SolveJob>();
        let (out_tx, out_rx) = std::sync::mpsc::channel::<SolveOutcome>();
        let job_rx = std::sync::Arc::new(std::sync::Mutex::new(job_rx));
        // One receiver shared by N workers: solves are independent and
        // wildly uneven in length (milliseconds to seconds), so pulling from a
        // common queue balances them without any scheduling logic.
        let workers = threads.max(1);
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let job_rx = std::sync::Arc::clone(&job_rx);
            let out_tx = out_tx.clone();
            handles.push(std::thread::spawn(move || loop {
                let job = {
                    let guard = job_rx.lock().expect("solver queue poisoned");
                    guard.recv()
                };
                let Ok(job) = job else { break }; // senders dropped: shut down
                let overlay = endgame_overlay(&job.state, &job.legal);
                if out_tx.send(SolveOutcome { slot: job.slot, overlay }).is_err() {
                    break; // the scheduler is gone
                }
            }));
        }
        // Join only the first; the rest exit on the same channel close. Kept as
        // one handle because the pool's Drop simply closes the queue.
        let handle = handles.into_iter().next();
        Self { job_tx: Some(job_tx), out_rx, handle, inflight: 0 }
    }

    pub fn dispatch(&mut self, job: SolveJob) -> bool {
        match self.job_tx.as_ref().map(|tx| tx.send(job)) {
            Some(Ok(())) => {
                self.inflight += 1;
                true
            }
            _ => false,
        }
    }

    /// Outcomes that have already arrived. Never blocks.
    pub fn harvest(&mut self) -> Vec<SolveOutcome> {
        let mut out = Vec::new();
        while let Ok(outcome) = self.out_rx.try_recv() {
            self.inflight -= 1;
            out.push(outcome);
        }
        out
    }

    /// Block for one outcome. Only correct to call when nothing else can make
    /// progress -- see `pump` in the scheduler loop.
    pub fn wait_one(&mut self) -> Option<SolveOutcome> {
        if self.inflight == 0 {
            return None;
        }
        match self.out_rx.recv() {
            Ok(outcome) => {
                self.inflight -= 1;
                Some(outcome)
            }
            Err(_) => None,
        }
    }

    pub fn inflight(&self) -> usize {
        self.inflight
    }
}

impl Drop for SolverPool {
    fn drop(&mut self) {
        // Closing the queue is what stops the workers; joining before that
        // would deadlock on their blocking `recv`.
        self.job_tx = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod solver_pool_tests {
    use super::*;

    #[test]
    fn a_pool_with_no_work_reports_nothing_inflight_and_never_blocks() {
        let mut pool = SolverPool::new(2);
        assert_eq!(pool.inflight(), 0);
        assert!(pool.harvest().is_empty());
        assert!(pool.wait_one().is_none(), "wait_one blocked with no work");
    }

    #[test]
    fn dropping_the_pool_shuts_the_workers_down() {
        // The workers block on `recv`, so shutdown has to come from closing the
        // queue rather than from a flag they would never look at.
        let pool = SolverPool::new(4);
        drop(pool); // must not hang
    }
}
