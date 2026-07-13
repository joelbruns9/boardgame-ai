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

use pyo3::prelude::*;

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
/// make/unmake. Chance is modelled as a move that *resolves* hidden randomness:
/// [`is_stochastic`](Game::is_stochastic) flags such a state, [`chance_children`]
/// (Game::chance_children) enumerates/samples the outcomes, and
/// [`make_with_chance`](Game::make_with_chance) applies the move together with a
/// chosen outcome. A pure chance node (roll then decide) is just a state whose
/// only legal action is trivial and `is_stochastic` is true.
pub(crate) trait Game {
    type State;
    type Action: Copy;
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

    /// Does the next move from `s` resolve hidden randomness (i.e. is this a chance
    /// boundary)? When true, the searcher expands [`chance_children`] instead of a
    /// single deterministic child.
    fn is_stochastic(s: &Self::State) -> bool;
    /// (outcome, weight) children for a stochastic move; weights sum to 1. The game
    /// decides enumerate-vs-sample using `cfg` (enum_cap / chance_samples / seed).
    fn chance_children(s: &Self::State, cfg: &SearchConfig) -> Vec<(Self::Chance, f64)>;

    /// Apply a deterministic action in place, returning its undo record.
    fn make(s: &mut Self::State, a: Self::Action) -> PyResult<Self::Undo>;
    /// Apply an action together with a resolved chance outcome, in place.
    fn make_with_chance(
        s: &mut Self::State,
        a: Self::Action,
        c: &Self::Chance,
    ) -> PyResult<Self::Undo>;
    /// Reverse the most recent `make` / `make_with_chance`.
    fn unmake(s: &mut Self::State, u: Self::Undo);
}

/// A leaf evaluator for game `G`, returning a value in the player-0 frame. The
/// searcher only calls it at the horizon; terminals use `terminal_value_p0`.
pub(crate) trait Eval<G: Game> {
    fn eval(&self, s: &G::State) -> f64;
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
    if !G::is_stochastic(s) {
        let u = G::make(s, a)?;
        let r = value::<G, E>(s, depth - 1, alpha, beta, eval, cfg, nodes);
        G::unmake(s, u);
        return r;
    }
    let mut expected = 0.0;
    for (c, w) in G::chance_children(s, cfg) {
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
    let choice = match seed {
        Some(sd) if best.len() > 1 => {
            let mut rng = sd ^ *nodes;
            (splitmix64(&mut rng) as usize) % best.len()
        }
        _ => 0,
    };
    Ok(Some(best[choice]))
}
