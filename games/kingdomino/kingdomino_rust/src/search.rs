//! Generic depth-limited expectiminimax search — GAME-AGNOSTIC.
//!
//! The search algorithm (alpha-beta on decision layers, chance nodes handled by
//! enumerate-or-sample, expected-outcome terminals with an optional bounded-margin
//! blend) knows nothing about any specific game. A game plugs in by implementing
//! [`Game`] (rules: make/unmake, legality, chance, terminal/outcome) and supplying
//! an [`Eval`] (leaf value). Kingdomino is impl #1 (see `lib.rs`).
//!
//! CRATE-SPLIT NOTE: this module is deliberately free of game-specific types so it
//! can move to a standalone `nnue_search` crate when a second game arrives and
//! validates the trait boundary ("build one, extract at two"). The only current
//! coupling is `pyo3::PyResult` on the fallible rules methods; at extraction that
//! becomes a game-agnostic error type.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

/// Which player decides at a (non-chance) decision node, player-0 frame.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Turn {
    P0,
    P1,
}

/// Knobs shared by every search; all game-agnostic. The leaf evaluator is passed
/// separately (as an [`Eval`]) rather than named here, so a game can swap evals
/// (hand-crafted, NNUE, …) without touching this struct.
#[derive(Clone)]
pub(crate) struct SearchConfig {
    pub depth: u32,
    pub chance_samples: usize,
    pub enum_cap: u64,
    pub margin_weight: f64,
    pub seed: u64,
}

/// A two-player, perfect-information-modulo-chance game the searcher can walk via
/// make/unmake.
///
/// **Chance is per-ACTION.** A given action may resolve hidden randomness while a
/// sibling action in the same state does not, and different actions may draw from
/// different distributions: [`is_stochastic`](Game::is_stochastic) and
/// [`chance_children`](Game::chance_children) both receive the action so the game
/// can express `P(outcome | state, action)`. When `is_stochastic(s, a)` is true the
/// searcher expands `chance_children(s, a, cfg)` (probability-weighted, no pruning
/// across chance) via [`make_with_chance`](Game::make_with_chance); otherwise it
/// applies the single deterministic child via [`make`](Game::make).
///
/// **Depth convention (read before modelling a dice game).** Every `make` /
/// `make_with_chance` consumes one search ply (the recursion always goes
/// `depth - 1`). An action that *also* resolves chance therefore costs exactly one
/// ply, which is correct for a game like Kingdomino where the deal is fused into a
/// decision. A "roll, then decide" game modelled as a trivial action carrying the
/// roll would spend a decision ply on the roll — i.e. depth-free (pure) chance
/// nodes are NOT supported yet. Adding them is a deliberate interface extension to
/// make once a second game needs it (not guessed from Kingdomino alone).
///
/// **Contracts the searcher relies on:**
/// - `make` / `make_with_chance` MUST be atomic on error: on `Err` they leave `s`
///   unchanged, because the searcher holds no undo record for a failed make and
///   cannot restore partial mutation.
/// - Every non-terminal state MUST expose ≥1 legal action (model a forced move as
///   an explicit pass). The searcher returns an error, not a silent ±∞, otherwise.
/// - `terminal_value_p0` and `bounded_margin` MUST be finite; chance weights MUST
///   be finite and sum to 1.
pub(crate) trait Game {
    type State;
    type Action: Copy + PartialEq;
    type Chance;
    type Undo;

    /// The deciding player at a non-terminal decision node (player-0 frame).
    fn to_move(s: &Self::State) -> PyResult<Turn>;
    fn is_terminal(s: &Self::State) -> bool;
    /// Official outcome at a terminal state, player-0 frame: +1 / 0 / -1.
    fn terminal_value_p0(s: &Self::State) -> f64;
    /// Bounded (|·| < 1) margin proxy, player-0 frame. Used both for optional
    /// terminal shaping and as the trivial "pick-blind" leaf eval.
    fn bounded_margin(s: &Self::State) -> f64;
    /// Append the legal actions at `s`. Order is irrelevant to the returned value
    /// (max/min is order-independent); it only affects alpha-beta cutoff timing.
    fn legal_actions(s: &Self::State, out: &mut Vec<Self::Action>);

    /// Does applying `a` at `s` resolve hidden randomness? When true, the searcher
    /// expands [`chance_children`] for this action instead of a single child.
    fn is_stochastic(s: &Self::State, a: Self::Action) -> bool;
    /// (outcome, weight) children for the stochastic action `a`; weights sum to 1.
    /// The game decides enumerate-vs-sample using `cfg` (enum_cap/chance_samples/seed).
    fn chance_children(
        s: &Self::State,
        a: Self::Action,
        cfg: &SearchConfig,
    ) -> Vec<(Self::Chance, f64)>;

    /// Apply a deterministic action in place, returning its undo record. Atomic on
    /// error (see trait contracts).
    fn make(s: &mut Self::State, a: Self::Action) -> PyResult<Self::Undo>;
    /// Apply an action together with a resolved chance outcome, in place. Atomic on
    /// error.
    fn make_with_chance(
        s: &mut Self::State,
        a: Self::Action,
        c: &Self::Chance,
    ) -> PyResult<Self::Undo>;
    /// Reverse the most recent `make` / `make_with_chance`.
    fn unmake(s: &mut Self::State, u: Self::Undo);

    /// If this position begins a deterministic tail, return the exact number of
    /// plies remaining to GAME_OVER. The operational search uses this to extend a
    /// horizon through a solved endgame under the same deadline. The default is
    /// no extension; a game must opt in with an exact (not estimated) count.
    fn exact_remaining_plies(_s: &Self::State) -> Option<u32> {
        None
    }

    /// Stable public-state key for a within-move transposition table. The scratch
    /// buffer is owned by the search and may be cleared/reused. Returning `None`
    /// disables TT value reuse and hash-move ordering for this game.
    fn position_key(_s: &Self::State, _scratch: &mut Vec<u8>) -> Option<u128> {
        None
    }
}

/// A leaf evaluator for game `G`, returning a value in the player-0 frame. The
/// searcher only calls it at the horizon; terminals use `terminal_value_p0`.
///
/// MUST return a finite value: a non-finite (NaN/±∞) eval poisons the alpha-beta
/// comparisons — no score compares greater than the initial −∞ — which would leave
/// `choose_action` with no best move. The searcher turns that into an error rather
/// than a panic, but a correct evaluator should never produce it.
pub(crate) trait Eval<G: Game> {
    fn eval(&self, s: &G::State) -> f64;
}

/// Wall-clock and quality controls for deadline-safe iterative deepening.
pub(crate) struct OperationalLimits {
    pub max_depth: u32,
    pub deadline: Instant,
    pub aspiration_window: f64,
    pub node_limit: Option<u64>,
    /// Symmetric absolute bound on every leaf and terminal value. This enables
    /// safe Star1 chance pruning. Use infinity to disable chance pruning.
    pub value_bound: f64,
}

/// A move from the last fully completed iteration. Partial iterations never
/// overwrite these fields.
pub(crate) struct OperationalResult<A> {
    pub action: A,
    pub value: Option<f64>,
    pub completed_depth: u32,
    pub timed_out: bool,
    pub nodes: u64,
    pub aspiration_researches: u32,
    pub star_cutoffs: u64,
    pub exact_extensions: u64,
    pub tt_hits: u64,
    pub tt_cutoffs: u64,
    pub last_iteration_nodes: u64,
}

#[derive(Clone, Copy)]
enum TtBound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TtEntry<A: Copy> {
    depth: i32,
    value: f64,
    bound: TtBound,
    best_action: A,
}

struct OperationalControl<A: Copy> {
    deadline: Instant,
    node_limit: Option<u64>,
    nodes: u64,
    star_cutoffs: u64,
    exact_extensions: u64,
    tt_hits: u64,
    tt_cutoffs: u64,
    tt: HashMap<u128, TtEntry<A>>,
    hash_scratch: Vec<u8>,
}

impl<A: Copy> OperationalControl<A> {
    #[inline]
    fn enter_node(&mut self) -> bool {
        if self.node_limit.is_some_and(|limit| self.nodes >= limit)
            || Instant::now() >= self.deadline
        {
            return false;
        }
        self.nodes += 1;
        true
    }

    #[inline]
    fn expired(&self) -> bool {
        self.node_limit.is_some_and(|limit| self.nodes >= limit)
            || Instant::now() >= self.deadline
    }
}

/// SplitMix64 — a tiny reproducible PRNG (chance sampling + tie-breaking).
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Value of applying `a` then searching one ply deeper. At a stochastic move it
/// probability-weights child values over the chance outcomes (fresh window per
/// child — no pruning across chance). Always unmakes, even when a child errors.
#[allow(clippy::too_many_arguments)]
fn action_value<G: Game, E: Eval<G>>(
    s: &mut G::State,
    a: G::Action,
    depth: i32,
    alpha: f64,
    beta: f64,
    eval: &E,
    cfg: &SearchConfig,
    nodes: &mut u64,
) -> PyResult<f64> {
    if !G::is_stochastic(s, a) {
        let u = G::make(s, a)?;
        let r = value::<G, E>(s, depth - 1, alpha, beta, eval, cfg, nodes);
        G::unmake(s, u);
        return r;
    }
    let mut expected = 0.0;
    for (c, w) in G::chance_children(s, a, cfg) {
        let u = G::make_with_chance(s, a, &c)?;
        let r = value::<G, E>(s, depth - 1, f64::NEG_INFINITY, f64::INFINITY, eval, cfg, nodes);
        G::unmake(s, u);
        expected += w * r?;
    }
    Ok(expected)
}

/// Depth-limited expectiminimax value, player-0 frame. Alpha-beta on the decision
/// layers; player 0 maximizes, player 1 minimizes.
pub(crate) fn value<G: Game, E: Eval<G>>(
    s: &mut G::State,
    depth: i32,
    mut alpha: f64,
    mut beta: f64,
    eval: &E,
    cfg: &SearchConfig,
    nodes: &mut u64,
) -> PyResult<f64> {
    *nodes += 1;
    if G::is_terminal(s) {
        let mut v = G::terminal_value_p0(s);
        if cfg.margin_weight != 0.0 {
            v += cfg.margin_weight * G::bounded_margin(s);
        }
        return Ok(v);
    }
    if depth <= 0 {
        return Ok(eval.eval(s));
    }
    let mut actions = Vec::new();
    G::legal_actions(s, &mut actions);
    if actions.is_empty() {
        // Contract: every non-terminal state must expose ≥1 action. Erroring here
        // beats silently returning ±∞ (which would corrupt the parent's minimax).
        return Err(PyValueError::new_err(
            "non-terminal state has no legal actions (model a forced move as a pass)",
        ));
    }
    if matches!(G::to_move(s)?, Turn::P0) {
        let mut v = f64::NEG_INFINITY;
        for a in actions {
            let av = action_value::<G, E>(s, a, depth, alpha, beta, eval, cfg, nodes)?;
            if av > v {
                v = av;
            }
            if v > alpha {
                alpha = v;
            }
            if alpha >= beta {
                break; // beta cutoff
            }
        }
        Ok(v)
    } else {
        let mut v = f64::INFINITY;
        for a in actions {
            let av = action_value::<G, E>(s, a, depth, alpha, beta, eval, cfg, nodes)?;
            if av < v {
                v = av;
            }
            if v < beta {
                beta = v;
            }
            if beta <= alpha {
                break; // alpha cutoff
            }
        }
        Ok(v)
    }
}

/// Best action for the side to move (player 0 maximizes the value, player 1
/// minimizes it), searched at `cfg.depth`. Each root child gets a full
/// (-inf, inf) window (no root-sibling pruning), so every root child value is
/// exact. Ties broken by `seed` (deterministic first-best when `seed` is None).
/// Returns `None` only when `s` has no legal actions (caller decides how to
/// surface that); a single legal action is returned without searching (so
/// `*nodes` stays 0 — the caller should treat that as "no search performed").
pub(crate) fn choose_action<G: Game, E: Eval<G>>(
    s: &mut G::State,
    eval: &E,
    cfg: &SearchConfig,
    seed: Option<u64>,
    nodes: &mut u64,
) -> PyResult<Option<G::Action>> {
    let mut actions = Vec::new();
    G::legal_actions(s, &mut actions);
    if actions.is_empty() {
        return Ok(None);
    }
    if actions.len() == 1 {
        return Ok(Some(actions[0]));
    }
    let maximizing = matches!(G::to_move(s)?, Turn::P0);
    let mut best_score = f64::NEG_INFINITY;
    let mut best: Vec<G::Action> = Vec::new();
    for a in actions {
        let av = action_value::<G, E>(
            s,
            a,
            cfg.depth as i32,
            f64::NEG_INFINITY,
            f64::INFINITY,
            eval,
            cfg,
            nodes,
        )?;
        let score = if maximizing { av } else { -av };
        if score > best_score {
            best_score = score;
            best.clear();
            best.push(a);
        } else if score == best_score {
            best.push(a);
        }
    }
    if best.is_empty() {
        // Reached only when every action scored non-finite (e.g. a NaN-producing
        // evaluator): nothing compared greater than the initial −∞. Return an error
        // rather than index an empty vector (the panic the reviewer flagged).
        return Err(PyValueError::new_err(
            "choose_action: no action produced a finite score (evaluator returned non-finite)",
        ));
    }
    let choice = match seed {
        Some(sd) if best.len() > 1 => {
            let mut rng = sd ^ *nodes;
            (splitmix64(&mut rng) as usize) % best.len()
        }
        _ => 0,
    };
    Ok(Some(best[choice]))
}

/// Deadline-aware counterpart of `action_value`. `Ok(None)` means the shared
/// budget expired. Every successful make is unmade before timeout/error is
/// propagated, which is the core safety contract for a mutable search state.
#[allow(clippy::too_many_arguments)]
fn operational_action_value<G: Game, E: Eval<G>>(
    s: &mut G::State,
    a: G::Action,
    depth: i32,
    alpha: f64,
    beta: f64,
    eval: &E,
    cfg: &SearchConfig,
    control: &mut OperationalControl<G::Action>,
    value_bound: f64,
) -> PyResult<Option<f64>> {
    if !G::is_stochastic(s, a) {
        let u = G::make(s, a)?;
        let result = operational_value::<G, E>(
            s,
            depth - 1,
            alpha,
            beta,
            eval,
            cfg,
            control,
            value_bound,
        );
        G::unmake(s, u);
        return result;
    }

    let children = G::chance_children(s, a, cfg);
    if children.is_empty() {
        return Err(PyValueError::new_err(
            "stochastic action produced no chance children",
        ));
    }
    let weight_sum: f64 = children.iter().map(|(_, w)| *w).sum();
    if children.iter().any(|(_, w)| !w.is_finite() || *w <= 0.0)
        || !weight_sum.is_finite()
        || (weight_sum - 1.0).abs() > 1e-9
    {
        return Err(PyValueError::new_err(
            "chance weights must be positive, finite, and sum to 1",
        ));
    }

    let bounded = value_bound.is_finite() && value_bound > 0.0;
    let (lower, upper) = (-value_bound, value_bound);
    let mut expected = 0.0;
    let mut consumed_weight = 0.0;
    for (chance, weight) in children {
        if control.expired() {
            return Ok(None);
        }
        let remaining_after = (1.0 - consumed_weight - weight).max(0.0);
        let (child_alpha, child_beta, alpha_is_cut, beta_is_cut) = if bounded {
            // Star1 window transformation. If this child fails outside the
            // transformed window, even the most favorable remaining outcomes
            // cannot bring the chance expectation back inside [alpha, beta].
            let raw_alpha = (alpha - expected - remaining_after * upper) / weight;
            let raw_beta = (beta - expected - remaining_after * lower) / weight;
            (
                raw_alpha.max(lower),
                raw_beta.min(upper),
                raw_alpha > lower,
                raw_beta < upper,
            )
        } else {
            (f64::NEG_INFINITY, f64::INFINITY, false, false)
        };

        let u = G::make_with_chance(s, a, &chance)?;
        let result = operational_value::<G, E>(
            s,
            depth - 1,
            child_alpha,
            child_beta,
            eval,
            cfg,
            control,
            value_bound,
        );
        G::unmake(s, u);
        let Some(child_value) = result? else {
            return Ok(None);
        };
        if alpha_is_cut && child_value <= child_alpha {
            control.star_cutoffs += 1;
            return Ok(Some(alpha));
        }
        if beta_is_cut && child_value >= child_beta {
            control.star_cutoffs += 1;
            return Ok(Some(beta));
        }
        expected += weight * child_value;
        consumed_weight += weight;
    }
    Ok(Some(expected))
}

/// Deadline-aware alpha-beta value. A deterministic exact tail may extend past
/// the nominal horizon, but it remains governed by the same wall-clock/node
/// budget and therefore cannot make a timed move overrun indefinitely.
#[allow(clippy::too_many_arguments)]
fn operational_value<G: Game, E: Eval<G>>(
    s: &mut G::State,
    mut depth: i32,
    mut alpha: f64,
    mut beta: f64,
    eval: &E,
    cfg: &SearchConfig,
    control: &mut OperationalControl<G::Action>,
    value_bound: f64,
) -> PyResult<Option<f64>> {
    if !control.enter_node() {
        return Ok(None);
    }
    if G::is_terminal(s) {
        let mut value = G::terminal_value_p0(s);
        if cfg.margin_weight != 0.0 {
            value += cfg.margin_weight * G::bounded_margin(s);
        }
        return Ok(Some(value));
    }
    if depth <= 0 {
        if let Some(remaining) = G::exact_remaining_plies(s).filter(|n| *n > 0) {
            depth = remaining as i32;
            control.exact_extensions += 1;
        } else {
            let value = eval.eval(s);
            if !value.is_finite() {
                return Err(PyValueError::new_err(
                    "operational search evaluator returned a non-finite value",
                ));
            }
            return Ok(Some(value));
        }
    }

    let key = G::position_key(s, &mut control.hash_scratch);
    let mut hash_move = None;
    if let Some(entry) = key.and_then(|k| control.tt.get(&k).copied()) {
        hash_move = Some(entry.best_action);
        if entry.depth >= depth {
            control.tt_hits += 1;
            match entry.bound {
                TtBound::Exact => return Ok(Some(entry.value)),
                TtBound::Lower => alpha = alpha.max(entry.value),
                TtBound::Upper => beta = beta.min(entry.value),
            }
            if alpha >= beta {
                control.tt_cutoffs += 1;
                return Ok(Some(entry.value));
            }
        }
    }
    // Classify the new entry against the window actually searched, after any
    // older TT bound tightened it. Using the caller's wider window here could
    // mislabel a cutoff against the tightened bound as Exact.
    let alpha_original = alpha;
    let beta_original = beta;

    let mut actions = Vec::new();
    G::legal_actions(s, &mut actions);
    if actions.is_empty() {
        return Err(PyValueError::new_err(
            "non-terminal state has no legal actions (model a forced move as a pass)",
        ));
    }
    if let Some(best) = hash_move {
        if let Some(index) = actions.iter().position(|action| *action == best) {
            actions.swap(0, index);
        }
    }
    let maximizing = matches!(G::to_move(s)?, Turn::P0);
    let mut best_action = actions[0];
    let value = if maximizing {
        let mut value = f64::NEG_INFINITY;
        for action in actions {
            let Some(action_value) = operational_action_value::<G, E>(
                s, action, depth, alpha, beta, eval, cfg, control, value_bound,
            )? else {
                return Ok(None);
            };
            if action_value > value {
                value = action_value;
                best_action = action;
            }
            alpha = alpha.max(value);
            if alpha >= beta {
                break;
            }
        }
        value
    } else {
        let mut value = f64::INFINITY;
        for action in actions {
            let Some(action_value) = operational_action_value::<G, E>(
                s, action, depth, alpha, beta, eval, cfg, control, value_bound,
            )? else {
                return Ok(None);
            };
            if action_value < value {
                value = action_value;
                best_action = action;
            }
            beta = beta.min(value);
            if beta <= alpha {
                break;
            }
        }
        value
    };

    if let Some(key) = key {
        let bound = if value <= alpha_original {
            TtBound::Upper
        } else if value >= beta_original {
            TtBound::Lower
        } else {
            TtBound::Exact
        };
        let replace = control.tt.get(&key).is_none_or(|entry| depth >= entry.depth);
        if replace {
            control.tt.insert(
                key,
                TtEntry {
                    depth,
                    value,
                    bound,
                    best_action,
                },
            );
        }
    }
    Ok(Some(value))
}

struct RootIteration<A> {
    action: A,
    value: f64,
}

/// One root iteration with root-sibling window reuse. The previous iteration's
/// best action is searched first; subsequent siblings only prove whether they
/// beat the current bound, avoiding the fixed searcher's full-window root tax.
#[allow(clippy::too_many_arguments)]
fn operational_root_iteration<G: Game, E: Eval<G>>(
    s: &mut G::State,
    eval: &E,
    cfg: &SearchConfig,
    depth: u32,
    previous_best: Option<G::Action>,
    mut alpha: f64,
    mut beta: f64,
    control: &mut OperationalControl<G::Action>,
    value_bound: f64,
) -> PyResult<Option<RootIteration<G::Action>>> {
    let mut actions = Vec::new();
    G::legal_actions(s, &mut actions);
    if let Some(best) = previous_best {
        if let Some(index) = actions.iter().position(|action| *action == best) {
            actions.swap(0, index);
        }
    }
    let maximizing = matches!(G::to_move(s)?, Turn::P0);
    let mut best_action = None;
    let mut best_value = if maximizing {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    for action in actions {
        let Some(action_value) = operational_action_value::<G, E>(
            s,
            action,
            depth as i32,
            alpha,
            beta,
            eval,
            cfg,
            control,
            value_bound,
        )? else {
            return Ok(None);
        };
        if best_action.is_none()
            || (maximizing && action_value > best_value)
            || (!maximizing && action_value < best_value)
        {
            best_action = Some(action);
            best_value = action_value;
        }
        if maximizing {
            alpha = alpha.max(best_value);
        } else {
            beta = beta.min(best_value);
        }
        if alpha >= beta {
            break;
        }
    }
    let action = best_action.ok_or_else(|| {
        PyValueError::new_err("operational search: no action produced a finite score")
    })?;
    Ok(Some(RootIteration {
        action,
        value: best_value,
    }))
}

/// Deadline-safe iterative deepening. If an iteration or aspiration re-search
/// times out, the result from the previous fully completed depth is returned.
/// With no completed iteration, the stable first legal action is the fallback.
pub(crate) fn choose_action_operational<G: Game, E: Eval<G>>(
    s: &mut G::State,
    eval: &E,
    cfg: &SearchConfig,
    limits: &OperationalLimits,
) -> PyResult<Option<OperationalResult<G::Action>>> {
    let mut legal = Vec::new();
    G::legal_actions(s, &mut legal);
    if legal.is_empty() {
        return Ok(None);
    }
    let fallback = legal[0];
    let root_is_exact_tail = G::exact_remaining_plies(s).is_some();
    if legal.len() == 1 {
        return Ok(Some(OperationalResult {
            action: fallback,
            value: None,
            completed_depth: 0,
            timed_out: false,
            nodes: 0,
            aspiration_researches: 0,
            star_cutoffs: 0,
            exact_extensions: 0,
            tt_hits: 0,
            tt_cutoffs: 0,
            last_iteration_nodes: 0,
        }));
    }

    let mut control = OperationalControl {
        deadline: limits.deadline,
        node_limit: limits.node_limit,
        nodes: 0,
        star_cutoffs: 0,
        exact_extensions: 0,
        tt_hits: 0,
        tt_cutoffs: 0,
        tt: HashMap::new(),
        hash_scratch: Vec::with_capacity(1024),
    };
    let mut action = fallback;
    let mut value = None;
    let mut completed_depth = 0;
    let mut timed_out = false;
    let mut aspiration_researches = 0;
    let mut last_iteration_nodes = 0;

    for depth in 1..=limits.max_depth {
        let iteration_start_nodes = control.nodes;
        let aspiration = value.map(|center| {
            (
                center - limits.aspiration_window,
                center + limits.aspiration_window,
            )
        });
        let (alpha, beta) = aspiration.unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
        let Some(mut iteration) = operational_root_iteration::<G, E>(
            s,
            eval,
            cfg,
            depth,
            Some(action),
            alpha,
            beta,
            &mut control,
            limits.value_bound,
        )? else {
            timed_out = true;
            break;
        };

        if aspiration.is_some() && (iteration.value <= alpha || iteration.value >= beta) {
            aspiration_researches += 1;
            let Some(full) = operational_root_iteration::<G, E>(
                s,
                eval,
                cfg,
                depth,
                Some(iteration.action),
                f64::NEG_INFINITY,
                f64::INFINITY,
                &mut control,
                limits.value_bound,
            )? else {
                timed_out = true;
                break;
            };
            iteration = full;
        }
        action = iteration.action;
        value = Some(iteration.value);
        completed_depth = depth;
        last_iteration_nodes = control.nodes - iteration_start_nodes;
        if root_is_exact_tail {
            break;
        }
    }

    Ok(Some(OperationalResult {
        action,
        value,
        completed_depth,
        timed_out,
        nodes: control.nodes,
        aspiration_researches,
        star_cutoffs: control.star_cutoffs,
        exact_extensions: control.exact_extensions,
        tt_hits: control.tt_hits,
        tt_cutoffs: control.tt_cutoffs,
        last_iteration_nodes,
    }))
}

#[cfg(test)]
mod tests {
    //! Contract tests for the generic search, using a tiny explicit-tree TOY GAME
    //! (unrelated to Kingdomino). These validate the trait boundary itself — the
    //! paths Kingdomino never exercises: action-dependent chance, a standalone
    //! chance node, consecutive turns by the same player, make-error unwind,
    //! no-legal-action handling, non-finite evaluation, and tie-breaking.
    use super::*;
    use std::rc::Rc;

    #[derive(Clone)]
    enum Edge {
        Det(usize),                // deterministic -> child node id
        Chance(Vec<(usize, f64)>), // action-dependent chance -> (child, weight) pairs
        Err,                       // make returns Err (atomicity / unwind test)
    }

    #[derive(Clone)]
    enum Node {
        Decision { player: u8, edges: Vec<Edge> }, // player 0 maximizes, 1 minimizes
        Terminal(f64),                             // player-0-frame outcome
    }

    /// State is a cursor (node id) into a shared node table, so make/unmake are
    /// trivial (set / restore the id).
    #[derive(Clone)]
    struct ToyState {
        tree: Rc<Vec<Node>>,
        node: usize,
    }

    struct Toy;

    impl Game for Toy {
        type State = ToyState;
        type Action = usize; // index into the current Decision's edges
        type Chance = usize; // the chosen child node id
        type Undo = usize; // the previous node id

        fn to_move(s: &ToyState) -> PyResult<Turn> {
            match &s.tree[s.node] {
                Node::Decision { player, .. } => {
                    Ok(if *player == 0 { Turn::P0 } else { Turn::P1 })
                }
                Node::Terminal(_) => Err(PyValueError::new_err("to_move on terminal")),
            }
        }
        fn is_terminal(s: &ToyState) -> bool {
            matches!(s.tree[s.node], Node::Terminal(_))
        }
        fn terminal_value_p0(s: &ToyState) -> f64 {
            match s.tree[s.node] {
                Node::Terminal(v) => v,
                _ => panic!("terminal_value_p0 on non-terminal"),
            }
        }
        fn bounded_margin(_s: &ToyState) -> f64 {
            0.0
        }
        fn legal_actions(s: &ToyState, out: &mut Vec<usize>) {
            if let Node::Decision { edges, .. } = &s.tree[s.node] {
                out.extend(0..edges.len());
            }
        }
        fn is_stochastic(s: &ToyState, a: usize) -> bool {
            matches!(&s.tree[s.node],
                Node::Decision { edges, .. } if matches!(edges[a], Edge::Chance(_)))
        }
        fn chance_children(s: &ToyState, a: usize, _cfg: &SearchConfig) -> Vec<(usize, f64)> {
            if let Node::Decision { edges, .. } = &s.tree[s.node] {
                if let Edge::Chance(outs) = &edges[a] {
                    return outs.clone();
                }
            }
            Vec::new()
        }
        fn make(s: &mut ToyState, a: usize) -> PyResult<usize> {
            let prev = s.node;
            match &s.tree[s.node] {
                Node::Decision { edges, .. } => match &edges[a] {
                    Edge::Det(child) => {
                        s.node = *child; // only mutation happens on success -> atomic
                        Ok(prev)
                    }
                    Edge::Err => Err(PyValueError::new_err("toy: erroring edge")),
                    Edge::Chance(_) => Err(PyValueError::new_err("toy: make on chance edge")),
                },
                Node::Terminal(_) => Err(PyValueError::new_err("toy: make on terminal")),
            }
        }
        fn make_with_chance(s: &mut ToyState, _a: usize, c: &usize) -> PyResult<usize> {
            let prev = s.node;
            s.node = *c;
            Ok(prev)
        }
        fn unmake(s: &mut ToyState, u: usize) {
            s.node = u;
        }
        fn position_key(s: &ToyState, _scratch: &mut Vec<u8>) -> Option<u128> {
            Some(s.node as u128)
        }
    }

    struct ZeroEval;
    impl<G: Game> Eval<G> for ZeroEval {
        fn eval(&self, _s: &G::State) -> f64 {
            0.0
        }
    }
    struct NanEval;
    impl<G: Game> Eval<G> for NanEval {
        fn eval(&self, _s: &G::State) -> f64 {
            f64::NAN
        }
    }

    fn cfg() -> SearchConfig {
        SearchConfig {
            depth: 8,
            chance_samples: 4,
            enum_cap: 1_000_000,
            margin_weight: 0.0,
            seed: 0,
        }
    }
    fn st(tree: Vec<Node>) -> ToyState {
        ToyState { tree: Rc::new(tree), node: 0 }
    }
    fn deep_value(s: &mut ToyState, eval: &impl Eval<Toy>) -> PyResult<f64> {
        let mut nodes = 0u64;
        value::<Toy, _>(s, 8, f64::NEG_INFINITY, f64::INFINITY, eval, &cfg(), &mut nodes)
    }
    fn operational_limits(
        max_depth: u32,
        node_limit: Option<u64>,
        aspiration_window: f64,
    ) -> OperationalLimits {
        OperationalLimits {
            max_depth,
            deadline: Instant::now() + std::time::Duration::from_secs(60),
            aspiration_window,
            node_limit,
            value_bound: 1.0,
        }
    }

    #[test]
    fn minimax_with_consecutive_same_player() {
        // 0:P0 -> {1, 2};  1:P1 -> min(5, -1) = -1;  2:P0(again) -> max(2, 8) = 8.
        // root = max(-1, 8) = 8, and the best action leads to node 2.
        let tree = vec![
            Node::Decision { player: 0, edges: vec![Edge::Det(1), Edge::Det(2)] },
            Node::Decision { player: 1, edges: vec![Edge::Det(3), Edge::Det(4)] },
            Node::Decision { player: 0, edges: vec![Edge::Det(5), Edge::Det(6)] },
            Node::Terminal(5.0),
            Node::Terminal(-1.0),
            Node::Terminal(2.0),
            Node::Terminal(8.0),
        ];
        let mut s = st(tree);
        assert_eq!(deep_value(&mut s, &ZeroEval).unwrap(), 8.0);
        assert_eq!(s.node, 0, "cursor restored after search");
        let mut n = 0;
        let a = choose_action::<Toy, _>(&mut s, &ZeroEval, &cfg(), None, &mut n).unwrap();
        assert_eq!(a, Some(1));
    }

    #[test]
    fn action_dependent_chance() {
        // root action 0 stochastic (0.5*10 + 0.5*0 = 5), action 1 deterministic (4).
        // P0 max(5, 4) = 5 — a stochastic AND a deterministic action in ONE state.
        let tree = vec![
            Node::Decision {
                player: 0,
                edges: vec![Edge::Chance(vec![(1, 0.5), (2, 0.5)]), Edge::Det(3)],
            },
            Node::Terminal(10.0),
            Node::Terminal(0.0),
            Node::Terminal(4.0),
        ];
        let mut s = st(tree);
        assert_eq!(deep_value(&mut s, &ZeroEval).unwrap(), 5.0);
    }

    #[test]
    fn standalone_chance_node() {
        // A state whose only action is stochastic: 0.5*2 + 0.5*(-2) = 0.
        let tree = vec![
            Node::Decision { player: 0, edges: vec![Edge::Chance(vec![(1, 0.5), (2, 0.5)])] },
            Node::Terminal(2.0),
            Node::Terminal(-2.0),
        ];
        let mut s = st(tree);
        assert_eq!(deep_value(&mut s, &ZeroEval).unwrap(), 0.0);
    }

    #[test]
    fn nonterminal_without_actions_errors() {
        // value() must error (not silently return +-inf); choose_action returns None
        // (the pyclass wrapper turns that into an error).
        let mut s = st(vec![Node::Decision { player: 0, edges: vec![] }]);
        assert!(deep_value(&mut s, &ZeroEval).is_err());
        let mut n = 0;
        let r = choose_action::<Toy, _>(&mut s, &ZeroEval, &cfg(), None, &mut n).unwrap();
        assert_eq!(r, None);
    }

    #[test]
    fn nonfinite_eval_errors_not_panics() {
        // depth 1 so the (non-terminal) children are scored by the NaN evaluator:
        // every root action scores NaN -> best stays empty -> error, not a panic.
        let tree = vec![
            Node::Decision { player: 0, edges: vec![Edge::Det(1), Edge::Det(2)] },
            Node::Decision { player: 1, edges: vec![Edge::Det(3), Edge::Det(4)] },
            Node::Decision { player: 0, edges: vec![Edge::Det(5), Edge::Det(6)] },
            Node::Terminal(5.0),
            Node::Terminal(-1.0),
            Node::Terminal(2.0),
            Node::Terminal(8.0),
        ];
        let mut s = st(tree);
        let c = SearchConfig { depth: 1, ..cfg() };
        let mut n = 0;
        assert!(choose_action::<Toy, _>(&mut s, &NanEval, &c, None, &mut n).is_err());
    }

    #[test]
    fn make_error_unwinds_to_start() {
        // A make() that errors two plies deep must leave the cursor restored to the
        // root (action_value unmakes before propagating the error).
        let tree = vec![
            Node::Decision { player: 0, edges: vec![Edge::Det(1)] },
            Node::Decision { player: 1, edges: vec![Edge::Err] },
        ];
        let mut s = st(tree);
        assert!(deep_value(&mut s, &ZeroEval).is_err());
        assert_eq!(s.node, 0, "cursor must be restored after a mid-search make error");
    }

    #[test]
    fn tie_break_deterministic_and_seeded() {
        // Two equal-value actions. No seed -> first best; a fixed seed -> reproducible
        // (fresh node counter per call, as the pyclass wrapper does).
        let tree = vec![
            Node::Decision { player: 0, edges: vec![Edge::Det(1), Edge::Det(2)] },
            Node::Terminal(5.0),
            Node::Terminal(5.0),
        ];
        let mut s = st(tree);
        let mut n0 = 0;
        assert_eq!(
            choose_action::<Toy, _>(&mut s, &ZeroEval, &cfg(), None, &mut n0).unwrap(),
            Some(0)
        );
        let mut na = 0;
        let a1 = choose_action::<Toy, _>(&mut s, &ZeroEval, &cfg(), Some(42), &mut na).unwrap();
        let mut nb = 0;
        let a2 = choose_action::<Toy, _>(&mut s, &ZeroEval, &cfg(), Some(42), &mut nb).unwrap();
        assert_eq!(a1, a2, "seeded tie-break must be reproducible");
        assert!(matches!(a1, Some(0) | Some(1)));
    }

    #[test]
    fn operational_matches_fixed_and_restores() {
        let tree = vec![
            Node::Decision { player: 0, edges: vec![Edge::Det(1), Edge::Det(2)] },
            Node::Decision { player: 1, edges: vec![Edge::Det(3), Edge::Det(4)] },
            Node::Decision { player: 0, edges: vec![Edge::Det(5), Edge::Det(6)] },
            Node::Terminal(0.5),
            Node::Terminal(-0.1),
            Node::Terminal(0.2),
            Node::Terminal(0.8),
        ];
        let mut s = st(tree);
        let result = choose_action_operational::<Toy, _>(
            &mut s,
            &ZeroEval,
            &cfg(),
            &operational_limits(3, None, 0.25),
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.action, 1);
        assert_eq!(result.value, Some(0.8));
        assert_eq!(result.completed_depth, 3);
        assert!(!result.timed_out);
        assert_eq!(s.node, 0, "operational search must unwind to root");
    }

    #[test]
    fn operational_timeout_keeps_last_complete_iteration() {
        // Depth 1 sees two equal zero-valued leaves and keeps action 0. Depth 2
        // discovers action 1 is better. Give the second run exactly enough nodes
        // to complete depth 1 and begin (but not finish) depth 2.
        let tree = vec![
            Node::Decision { player: 0, edges: vec![Edge::Det(1), Edge::Det(2)] },
            Node::Decision { player: 1, edges: vec![Edge::Det(3), Edge::Det(4)] },
            Node::Decision { player: 1, edges: vec![Edge::Det(5), Edge::Det(6)] },
            Node::Terminal(-1.0),
            Node::Terminal(-0.5),
            Node::Terminal(0.5),
            Node::Terminal(0.8),
        ];
        let mut baseline = st(tree.clone());
        let depth_one = choose_action_operational::<Toy, _>(
            &mut baseline,
            &ZeroEval,
            &cfg(),
            &operational_limits(1, None, 0.25),
        )
        .unwrap()
        .unwrap();
        assert_eq!(depth_one.action, 0);

        let mut limited = st(tree);
        let result = choose_action_operational::<Toy, _>(
            &mut limited,
            &ZeroEval,
            &cfg(),
            &operational_limits(2, Some(depth_one.nodes + 1), 0.25),
        )
        .unwrap()
        .unwrap();
        assert!(result.timed_out);
        assert_eq!(result.completed_depth, 1);
        assert_eq!(result.action, depth_one.action);
        assert_eq!(result.value, depth_one.value);
        assert_eq!(limited.node, 0, "timeout must unwind the partial iteration");
    }

    #[test]
    fn aspiration_failure_researches_full_window() {
        let tree = vec![
            Node::Decision { player: 0, edges: vec![Edge::Det(1), Edge::Det(2)] },
            Node::Decision { player: 1, edges: vec![Edge::Det(3), Edge::Det(4)] },
            Node::Decision { player: 1, edges: vec![Edge::Det(5), Edge::Det(6)] },
            Node::Terminal(-1.0),
            Node::Terminal(-0.5),
            Node::Terminal(0.5),
            Node::Terminal(0.8),
        ];
        let mut s = st(tree);
        let result = choose_action_operational::<Toy, _>(
            &mut s,
            &ZeroEval,
            &cfg(),
            &operational_limits(2, None, 0.1),
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.action, 1);
        assert_eq!(result.value, Some(0.5));
        assert_eq!(result.completed_depth, 2);
        assert_eq!(result.aspiration_researches, 1);
        assert_eq!(s.node, 0);
    }

    #[test]
    fn star1_prunes_chance_action_without_changing_best_move() {
        let tree = vec![
            Node::Decision {
                player: 0,
                edges: vec![Edge::Det(1), Edge::Chance(vec![(2, 0.5), (3, 0.5)])],
            },
            Node::Terminal(0.8),
            Node::Terminal(-1.0),
            Node::Terminal(-1.0),
        ];
        let mut s = st(tree);
        let result = choose_action_operational::<Toy, _>(
            &mut s,
            &ZeroEval,
            &cfg(),
            &operational_limits(1, None, 0.25),
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.action, 0);
        assert_eq!(result.value, Some(0.8));
        assert!(result.star_cutoffs >= 1);
        assert_eq!(s.node, 0);
    }

    #[test]
    fn transposition_table_reuses_completed_subtree() {
        // Both root actions reach the same decision node. The first search stores
        // it; the second must reuse the exact value without walking its children.
        let tree = vec![
            Node::Decision { player: 0, edges: vec![Edge::Det(1), Edge::Det(1)] },
            Node::Decision { player: 1, edges: vec![Edge::Det(2), Edge::Det(3)] },
            Node::Terminal(-0.5),
            Node::Terminal(0.5),
        ];
        let mut s = st(tree);
        let result = choose_action_operational::<Toy, _>(
            &mut s,
            &ZeroEval,
            &cfg(),
            &operational_limits(2, None, 0.25),
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.action, 0);
        assert_eq!(result.value, Some(-0.5));
        assert!(result.tt_hits >= 1);
        assert_eq!(s.node, 0);
    }
}
