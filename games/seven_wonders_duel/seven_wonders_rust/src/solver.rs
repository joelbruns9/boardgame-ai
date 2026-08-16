//! Exact endgame solver: minimax over decisions, expectimax over chance.
//!
//! The Python reference is `advisor_endgame.py`, and this must agree with it
//! exactly — its answers become training labels, and a solver that is quietly
//! wrong is worse than no solver at all. `endgame_corpus.py` is the gate.
//!
//! Three things this does that the reference does not, all of which preserve
//! the answer:
//!
//! * **Alpha-beta at decision nodes.** The reference is full-width: no cutoffs
//!   at all. Pruning is where most of the speedup lives, since node counts
//!   roughly double per extra card on the board.
//! * **Move ordering.** The strongest measured correlate of a position's node
//!   count was its legal-action count (+0.65 rank correlation), which is what
//!   ordering attacks: try the moves most likely to be best first and the
//!   window closes sooner.
//! What it does NOT do is avoid copying the state: `snapshot`/`restore` is one
//! full `GameState` clone per child edge, because there is no journaled undo.
//! Measured, that copy is 280-320ns against ~900ns per node, so about a third
//! of the search -- but it is the *copy* that costs, not the allocation, and
//! the fix for it is journaled undo or a smaller state, not a cleverer buffer:
//!
//! * reusing one buffer per depth (`clone_from` + swap, so the allocator is
//!   never asked for anything) measured 1,107k vs 1,100k nodes/s on the corpus
//!   and 605k vs 614k on deep positions, i.e. nothing. The crate allocates
//!   through mimalloc, so the malloc that trick removes was already cheap;
//! * likewise dropping the chance-chain key vectors: 1,133k vs 1,100k.
//!
//! And the ceiling on all of it is low. Node counts grow ~2.5x per extra card
//! (4.5M at 8 cards, 11.0M at 9, 29.6M at 10), so removing the clone entirely
//! would be ~1.4x, worth about half a card of depth.
//!
//! **Threads.** One solve is single-threaded: plain recursion, no rayon. It
//! parallelises across *positions* instead, which is what a self-play solver
//! pool wants -- the binding releases the GIL, so N Python threads run N solves
//! at once with nothing shared. Measured on 16 logical CPUs: 2.56x at 4
//! threads, 3.76x at 8, 4.42x at 16.
//!
//! That ceiling is the state copy again, and this time it is provable rather
//! than inferred: cloning alone, with no search at all, scales the same way
//! (2.80x at 4, 3.74x at 8) and saturates at ~10.4M clones/s, i.e. ~17.7 GB/s
//! of memcpy. The pool runs out of memory bandwidth, not of cores. So a
//! journaled undo would buy more than its single-threaded ~1.4x suggests: it
//! would raise the per-core rate AND the number of cores worth giving the pool.
//!
//! **Star1 and star2 were tried and mostly did not pay.** 94% of corpus nodes
//! and ~100% of deep-position nodes sit below a chance edge, where the
//! inherited window used to be discarded -- which looked like the biggest lever
//! in the solver. Measured: star1 gives 0.87-0.97x the nodes, star2 costs
//! 1.17-1.86x. The mechanism is the shape of 7WD chance: an edge has 5-20
//! outcomes carrying 0.05-0.2 probability each, so one resolved outcome barely
//! constrains an average the other 80-95% of the mass can still move by +-1.
//! A useful bound arrives only once most of the mass is already resolved. Star1
//! is on anyway (identical values, a little cheaper); star2 is not.
//!
//! **Chance nodes are not pruned.** A chance node's value is an average, so a
//! partial sum bounds nothing until every child is in — pruning there needs
//! star1/star2, which is a separate change and a separate gate. Children of a
//! chance node are therefore evaluated on the full window. This matters more
//! for 7WD than it did for Kingdomino: measured on the corpus, 19 of 28 real
//! endgame positions still contain chance, so a chance-free-only solver would
//! decline the majority of them.
//!
//! Values are in player-0 terms throughout, like the reference, and are only
//! flipped into actor terms at the root.

use std::time::Instant;

use crate::chance::{self, ChanceKind};
use crate::codec;
use crate::engine::Action;
use crate::state::{GameState, Phase};

/// Why a solve stopped without an answer. Both mean "the net estimate stands":
/// the caller must not treat either as a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStop {
    /// A sample-only chance edge (an Age deal) — not enumerable at any budget.
    Unsolvable,
    /// Node budget or deadline reached.
    Budget,
}

/// Pruning at chance nodes. Switchable because its value is an empirical
/// question, and because every setting must return identical values -- only the
/// node count may differ, which is what `endgame_corpus.py` checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChancePruning {
    /// Every outcome solved on the full window (what the Python reference does).
    None,
    /// Derive each child's window from the mass still unexamined. The default:
    /// measured 0.87-0.97x the nodes of `None` on deep positions, 0.96x on the
    /// corpus. Real, but far smaller than "94% of nodes sit under a chance
    /// edge" suggests -- see the module note on why.
    Star1,
    /// Star1, plus a probing pass over one move of each child first. **Loses
    /// here**: 1.17x, 1.61x and 1.86x the nodes of `None` at 8, 9 and 10 cards,
    /// worsening with depth. Kept switchable because the reason is specific and
    /// might not survive a change of design: star2 assumes a probe is cheap,
    /// which holds when a probe is a depth-limited search ending in a heuristic
    /// evaluation. This solver has no evaluator -- it runs to terminal -- so
    /// every probe is a full search of one move, and probing all children costs
    /// about as much as solving one of them outright.
    Star2,
}

/// How the root prices actions other than the best one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyMode {
    /// Root value and best action only; alternatives get alpha-beta bounds, so
    /// their reported values are NOT exact. Cheapest, and all self-play needs
    /// when the policy label is "the proven-best move".
    ValueOnly,
    /// Every root action solved on a full window: exact values for all of them.
    /// This is what the reference does, and the mode the equivalence gate uses.
    Exact,
}

pub struct Limits {
    pub max_nodes: u64,
    pub deadline: Instant,
}

pub struct Solve {
    pub root_value: f64,
    pub best_index: usize,
    /// `(action index, value in ACTOR terms)`. Under `ValueOnly` the non-best
    /// entries are bounds; under `Exact` every entry is exact.
    pub per_action: Vec<(usize, f64)>,
    pub saw_chance: bool,
    pub nodes: u64,
    pub nodes_under_chance: u64,
    pub exact_per_action: bool,
}

struct Ctx {
    nodes: u64,
    max_nodes: u64,
    deadline: Instant,
    saw_chance: bool,
    /// Checking the clock every node costs more than it saves, so it is sampled
    /// between batches -- but the FIRST tick always checks. Starting the counter
    /// at zero made the deadline soft: a position finishable in under 1,024
    /// nodes ignored it completely, and `max_secs = 0` still returned complete
    /// answers. The budget is a contract with the caller, not a hint.
    since_clock_check: u32,
    /// Depth in chance subtrees, and the nodes visited inside them. Reported so
    /// the value of pruning at chance nodes can be judged from data rather than
    /// from a feeling about how often 7WD endgames contain chance.
    chance_depth: u32,
    nodes_under_chance: u64,
    pruning: ChancePruning,
}

const CLOCK_CHECK_EVERY: u32 = 1024;

/// The range every game value lives in. Star1/star2 exist because of it: the
/// mass not yet examined at a chance node can move the average by at most that
/// mass times this range.
const VALUE_MIN: f64 = -1.0;
const VALUE_MAX: f64 = 1.0;

impl Ctx {
    fn tick(&mut self) -> Result<(), SolveStop> {
        self.nodes += 1;
        if self.chance_depth > 0 {
            self.nodes_under_chance += 1;
        }
        if self.nodes > self.max_nodes {
            return Err(SolveStop::Budget);
        }
        self.since_clock_check += 1;
        if self.nodes == 1 || self.since_clock_check >= CLOCK_CHECK_EVERY {
            self.since_clock_check = 0;
            if Instant::now() > self.deadline {
                return Err(SolveStop::Budget);
            }
        }
        Ok(())
    }
}

fn actor_of(state: &GameState) -> usize {
    match &state.pending_choice {
        Some(choice) => choice.player as usize,
        None => state.active_player as usize,
    }
}

fn terminal_p0(state: &GameState) -> f64 {
    match state.winner {
        None => 0.0,
        Some(0) => 1.0,
        Some(_) => -1.0,
    }
}

/// Cheap ordering hint, in player-0 terms: try what is most likely to be best.
///
/// Only the sign matters per actor, so this is a single heuristic read through
/// the moving player's eyes. Deliberately shallow — the point is to close the
/// window sooner, and anything expensive here is paid at every node.
fn order_key(state: &GameState, action: &Action, actor: usize) -> i64 {
    let mut score: i64 = 0;
    match action.use_ {
        crate::engine::ActionUse::ConstructWonder => score += 40,
        crate::engine::ActionUse::ConstructBuilding => {
            if let Some(slot) = action.slot {
                let card = crate::data::card(state.tableau.slots[slot].card_id);
                // Points on the board and shields that may end the game are the
                // two things that decide a 7WD endgame.
                score += (card.victory_points as i64) * 6;
                score += (card.shields as i64) * 10;
            }
        }
        crate::engine::ActionUse::DiscardForCoins => score += 2,
        _ => score += 1,
    }
    let _ = actor;
    score
}

/// Exact value of `state` in player-0 terms, within the window `[alpha, beta]`.
fn solve_p0(
    state: &mut GameState,
    ctx: &mut Ctx,
    mut alpha: f64,
    mut beta: f64,
) -> Result<f64, SolveStop> {
    ctx.tick()?;
    if state.phase == Phase::Complete {
        return Ok(terminal_p0(state));
    }
    let actor = actor_of(state);
    let indices = codec::legal_action_indices(state);
    if indices.is_empty() {
        return Ok(terminal_p0(state));
    }

    let mut ordered: Vec<(i64, usize)> = indices
        .iter()
        .map(|&index| {
            let action = codec::decode_action(state, index);
            (order_key(state, &action, actor), index)
        })
        .collect();
    ordered.sort_by(|a, b| b.0.cmp(&a.0));

    let maximizing = actor == 0;
    let mut best = if maximizing { f64::NEG_INFINITY } else { f64::INFINITY };
    for (_, index) in ordered {
        let value = edge_value_p0(state, index, ctx, alpha, beta)?;
        if maximizing {
            if value > best {
                best = value;
            }
            if best > alpha {
                alpha = best;
            }
        } else {
            if value < best {
                best = value;
            }
            if best < beta {
                beta = best;
            }
        }
        if alpha >= beta {
            break; // the other side already has something at least this good
        }
    }
    Ok(best)
}

/// Value of one action from `state`, integrating enumerable chance.
fn edge_value_p0(
    state: &mut GameState,
    index: usize,
    ctx: &mut Ctx,
    alpha: f64,
    beta: f64,
) -> Result<f64, SolveStop> {
    let action = codec::decode_action(state, index);
    let specs = chance::chance_signature(state, &action);
    if specs.iter().any(|s| s.kind == ChanceKind::AgeDeal) {
        return Err(SolveStop::Unsolvable);
    }
    if specs.is_empty() {
        let undo = state.snapshot();
        state.apply_action(&action);
        let value = solve_p0(state, ctx, alpha, beta);
        state.restore(undo);
        return value;
    }

    ctx.saw_chance = true;
    chance_edge_value_p0(state, &action, &specs, ctx, alpha, beta)
}

/// Expectimax over one action's chance outcomes, pruned per `ctx.pruning`.
///
/// The value is `sum(p_i * v_i)`, which is usually where pruning stops -- a
/// partial sum bounds nothing. It bounds plenty here, because every `v_i` is a
/// game value in `[-1, 1]`: whatever mass is still unexamined can move the
/// average by at most that mass. Star1 turns that into a window per child;
/// star2 first buys a tighter running bound cheaply, by looking at one move of
/// each child before committing to any full search.
fn chance_edge_value_p0(
    state: &mut GameState,
    action: &Action,
    specs: &[chance::ChanceSpec],
    ctx: &mut Ctx,
    alpha: f64,
    beta: f64,
) -> Result<f64, SolveStop> {
    let chains = chance::enumerate_chains_unkeyed(state, specs);
    let mass: f64 = chains.iter().map(|(_, p)| *p).sum();
    if (mass - 1.0).abs() > 1e-6 {
        return Err(SolveStop::Unsolvable);
    }

    // Mass still unexamined after each child, so the pruning bounds know how
    // much the remaining outcomes could still swing the average.
    let mut suffix = vec![0.0; chains.len() + 1];
    for index in (0..chains.len()).rev() {
        suffix[index] = suffix[index + 1] + chains[index].1;
    }

    if ctx.pruning == ChancePruning::Star2 {
        if let Some(bound) = star2_probe(state, action, &chains, &suffix, ctx, alpha, beta)? {
            return Ok(bound);
        }
    }

    let mut exact_so_far = 0.0;
    for (index, (outcomes, probability)) in chains.iter().enumerate() {
        let remaining = suffix[index + 1];
        // Invert "could the finished average still land inside [alpha, beta]?"
        // into a window for this child. On the full window this collapses to
        // [-1, 1], so an exact root solve prunes nothing and stays exact.
        let (child_alpha, child_beta) = match ctx.pruning {
            ChancePruning::None => (VALUE_MIN, VALUE_MAX),
            _ => (
                ((alpha - exact_so_far - remaining) / probability).max(VALUE_MIN),
                ((beta - exact_so_far + remaining) / probability).min(VALUE_MAX),
            ),
        };

        let value = solve_chance_child(state, action, outcomes, ctx, child_alpha, child_beta)?;
        if ctx.pruning != ChancePruning::None {
            // Outside its window this child has already decided the node: no
            // remaining outcome can pull the average back into [alpha, beta].
            // Fail soft -- return the bound, which the caller reads as "at most"
            // or "at least", exactly as it would from alpha-beta.
            //
            // The `> VALUE_MIN` / `< VALUE_MAX` guards are load-bearing, not
            // defensive. A window clamped to the full value range is not a
            // window at all, and a child that comes back at exactly -1 or +1
            // through one is reporting the true value, not failing against a
            // bound -- and decided endgames are full of exact -1s and +1s.
            // Without the guards the root reported bounds as values, which the
            // corpus caught on the first run.
            if child_alpha > VALUE_MIN && value <= child_alpha {
                return Ok(exact_so_far + probability * value + remaining);
            }
            if child_beta < VALUE_MAX && value >= child_beta {
                return Ok(exact_so_far + probability * value - remaining);
            }
        }
        exact_so_far += probability * value;
    }
    Ok(exact_so_far)
}

/// Apply one chance outcome and solve the resulting position.
fn solve_chance_child(
    state: &mut GameState,
    action: &Action,
    outcomes: &[Vec<usize>],
    ctx: &mut Ctx,
    alpha: f64,
    beta: f64,
) -> Result<f64, SolveStop> {
    let undo = state.snapshot();
    let applied = state.apply_with_chance(action, outcomes);
    ctx.chance_depth += 1;
    let value = match applied {
        Ok(()) => solve_p0(state, ctx, alpha, beta),
        Err(_) => Err(SolveStop::Unsolvable),
    };
    ctx.chance_depth -= 1;
    state.restore(undo);
    value
}

/// Star2's probing pass: one move of each child, then ask whether the node is
/// already decided.
///
/// A decision node is worth at least any single move to the player to move, and
/// at most any single move when it is the opponent's turn. So one move per
/// child costs a fraction of a full search and still yields a real bound on the
/// whole chance node. Every outcome of one action leaves the same player to
/// move, so the direction of those bounds is settled once, not per child.
///
/// Returns `Some(bound)` when the probes alone prove the node lies outside
/// `[alpha, beta]`, and `None` when the full search still has to happen.
fn star2_probe(
    state: &mut GameState,
    action: &Action,
    chains: &[(Vec<Vec<usize>>, f64)],
    suffix: &[f64],
    ctx: &mut Ctx,
    alpha: f64,
    beta: f64,
) -> Result<Option<f64>, SolveStop> {
    let mut probed = 0.0;
    let mut maximizing = None;
    for (index, (outcomes, probability)) in chains.iter().enumerate() {
        let remaining = suffix[index + 1];
        let child_alpha = ((alpha - probed - remaining) / probability).max(VALUE_MIN);
        let child_beta = ((beta - probed + remaining) / probability).min(VALUE_MAX);

        let undo = state.snapshot();
        let applied = state.apply_with_chance(action, outcomes);
        ctx.chance_depth += 1;
        let probe = match applied {
            Ok(()) => probe_first_move(state, ctx, child_alpha, child_beta),
            Err(_) => Err(SolveStop::Unsolvable),
        };
        ctx.chance_depth -= 1;
        state.restore(undo);
        let (bound, child_maximizing) = match probe? {
            Some(row) => row,
            None => return Ok(None), // nothing to probe: fall through to the full search
        };
        maximizing = Some(child_maximizing);
        probed += probability * bound;
    }

    // The probes bound the node in one direction only: from below when the
    // player to move maximises (each probe under-states its child), from above
    // when the opponent moves. Only a cut in that direction is available.
    // Same guard as the main loop: a "cut" against the full value range is not
    // a cut, it is the answer, and returning it as a bound loses exactness.
    match maximizing {
        Some(true) if beta < VALUE_MAX && probed >= beta => Ok(Some(probed)),
        Some(false) if alpha > VALUE_MIN && probed <= alpha => Ok(Some(probed)),
        _ => Ok(None),
    }
}

/// Value of the first (best-ordered) move of `state`, as a bound on the
/// position. `None` when there is no move to probe.
fn probe_first_move(
    state: &mut GameState,
    ctx: &mut Ctx,
    alpha: f64,
    beta: f64,
) -> Result<Option<(f64, bool)>, SolveStop> {
    if state.phase == Phase::Complete {
        return Ok(Some((terminal_p0(state), actor_of(state) == 0)));
    }
    let actor = actor_of(state);
    let indices = codec::legal_action_indices(state);
    if indices.is_empty() {
        return Ok(None);
    }
    let best = indices
        .iter()
        .map(|&index| {
            let decoded = codec::decode_action(state, index);
            (order_key(state, &decoded, actor), index)
        })
        .max_by_key(|row| row.0)
        .map(|row| row.1)
        .unwrap_or(indices[0]);
    let value = edge_value_p0(state, best, ctx, alpha, beta)?;
    Ok(Some((value, actor == 0)))
}

/// Solve every legal action at the root, in actor terms.
pub fn solve_root(
    state: &GameState,
    limits: &Limits,
    mode: PolicyMode,
    pruning: ChancePruning,
) -> Result<Solve, SolveStop> {
    if Instant::now() > limits.deadline {
        return Err(SolveStop::Budget); // already out of time before any work
    }
    let mut ctx = Ctx {
        nodes: 0,
        max_nodes: limits.max_nodes,
        deadline: limits.deadline,
        saw_chance: false,
        since_clock_check: 0,
        chance_depth: 0,
        nodes_under_chance: 0,
        pruning,
    };
    let actor = actor_of(state);
    let sign = if actor == 0 { 1.0 } else { -1.0 };
    let mut work = state.clone();
    let indices = codec::legal_action_indices(&work);
    if indices.is_empty() {
        return Err(SolveStop::Unsolvable);
    }

    let mut per_action: Vec<(usize, f64)> = Vec::with_capacity(indices.len());
    let mut best_actor_value = f64::NEG_INFINITY;
    let mut best_index = indices[0];
    for index in indices {
        // Exact mode prices every action on the full window, which is what
        // makes the values comparable with the reference; value-only mode lets
        // the root window tighten as better actions are found, which is
        // cheaper but leaves the alternatives as bounds.
        let (alpha, beta) = match mode {
            PolicyMode::Exact => (-1.0, 1.0),
            PolicyMode::ValueOnly => {
                if actor == 0 {
                    (best_actor_value.max(-1.0), 1.0)
                } else {
                    (-1.0, (-best_actor_value).min(1.0))
                }
            }
        };
        let value_p0 = edge_value_p0(&mut work, index, &mut ctx, alpha, beta)?;
        let actor_value = (sign * value_p0).clamp(-1.0, 1.0);
        per_action.push((index, actor_value));
        if actor_value > best_actor_value {
            best_actor_value = actor_value;
            best_index = index;
        }
    }

    Ok(Solve {
        root_value: best_actor_value,
        best_index,
        per_action,
        saw_chance: ctx.saw_chance,
        nodes: ctx.nodes,
        nodes_under_chance: ctx.nodes_under_chance,
        exact_per_action: mode == PolicyMode::Exact,
    })
}

#[cfg(test)]
mod tests {
    //! Budget and refusal semantics, which the Python equivalence gate cannot
    //! reach: it only ever compares *answers*, so a solver that ignored its
    //! limits or returned a value where it should have declined would still
    //! pass every position in the corpus.

    use super::*;
    use crate::state::Setup;
    use std::collections::VecDeque;
    use std::time::Duration;

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

    fn age_one_position() -> GameState {
        let mut g = GameState::from_setup(sample_setup(), VecDeque::new());
        while g.phase != Phase::PlayAge {
            let legal = codec::legal_action_indices(&g);
            g.apply_action(&codec::decode_action(&g, legal[0]));
        }
        g
    }

    fn limits(max_nodes: u64, secs: f64) -> Limits {
        Limits {
            max_nodes,
            deadline: Instant::now() + Duration::from_secs_f64(secs),
        }
    }

    #[test]
    fn an_age_one_position_is_refused_rather_than_guessed() {
        // Age I and II always reach the next Age's deal, which is sample-only.
        // Returning any value here would be inventing one.
        let state = age_one_position();
        let stop = solve_root(&state, &limits(u64::MAX, 30.0), PolicyMode::Exact, ChancePruning::Star2);
        assert_eq!(stop.err(), Some(SolveStop::Unsolvable));
    }

    #[test]
    fn the_node_budget_is_enforced() {
        let state = age_one_position();
        let stop = solve_root(&state, &limits(1, 30.0), PolicyMode::Exact, ChancePruning::Star2);
        assert_eq!(stop.err(), Some(SolveStop::Budget));
    }

    #[test]
    fn an_expired_deadline_stops_before_any_work() {
        //! The clock is sampled every N nodes for throughput, which once made
        //! the deadline unenforceable for anything finishing inside one window.
        let state = age_one_position();
        let expired = Limits {
            max_nodes: u64::MAX,
            deadline: Instant::now() - Duration::from_secs(1),
        };
        let stop = solve_root(&state, &expired, PolicyMode::Exact, ChancePruning::Star2);
        assert_eq!(stop.err(), Some(SolveStop::Budget));
    }
}

