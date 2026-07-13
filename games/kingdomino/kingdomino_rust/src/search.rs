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
}
