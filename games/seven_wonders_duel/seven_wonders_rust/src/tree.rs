//! F3.2 closed-node MCTS tree — a fresh port of `search.py`'s closed searcher
//! (nodes/edges/children, PUCT descent, outcome-keyed child materialization),
//! generic over `Eval`. The Gumbel root is added in F3.3; here a fixed
//! round-robin root schedule exercises the tree machinery for the 1e-6
//! equivalence gate under `MockEval`.
//!
//! Children are stored **insertion-ordered** (a `Vec`, not a map) so the
//! probability-weighted `q_p0` sum and value backprop fold in the same order as
//! Python's dict — cross-language f64 sums are order-sensitive (see F3.2 note).

use crate::chance::{self, ChanceSpec};
use crate::codec::{decode_action, legal_action_indices};
use crate::eval::{terminal_value_p0, Eval};
use crate::rng::Rng;
use crate::state::{GameState, Phase};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub struct Child {
    pub probability: Option<f64>,
    pub node: Box<Node>,
    pub samples: u32,
}

pub struct Edge {
    pub action_index: usize,
    pub prior: f64,
    pub specs: Vec<ChanceSpec>,
    pub children: Vec<(Vec<Vec<i32>>, Child)>, // observable key -> child, insertion order
    pub visits: u32,
    pub value_sum_p0: f64,
    pub probability_weighted: bool,
    /// The children are an APPROXIMATE, re-normalised subset of the outcome
    /// space and the edge is closed against growth (see `fixed_support_index`
    /// and `search.py::_Edge.fixed_support`). Ordinary descent samples from the
    /// COMPLETE distribution and appends whatever key it cannot find, which on
    /// a truncated edge would push the mass past 1 with nothing raising; a
    /// closed edge instead draws only among what it already holds.
    pub fixed_support: bool,
}

/// Index of the child a fixed-support draw lands on: the first whose cumulative
/// weight exceeds `target`. `None` means the draw fell past the final
/// cumulative sum (float residue, ~1e-16 for unit mass) — callers take the last
/// child. `search.py::fixed_support_index` folds in the same order with the
/// same strict `<`, so both languages pick the same child from the same draw.
pub(crate) fn fixed_support_index<I: IntoIterator<Item = f64>>(
    weights: I,
    target: f64,
) -> Option<usize> {
    let mut cumulative = 0.0;
    for (index, weight) in weights.into_iter().enumerate() {
        cumulative += weight;
        if target < cumulative {
            return Some(index);
        }
    }
    None
}

impl Edge {
    pub fn q_p0(&self) -> f64 {
        if self.probability_weighted {
            self.children
                .iter()
                .map(|(_, c)| {
                    c.probability.expect("weighted child needs probability") * c.node.value_p0()
                })
                .sum()
        } else if self.visits > 0 {
            self.value_sum_p0 / self.visits as f64
        } else {
            0.0
        }
    }
}

pub struct Node {
    pub state: GameState,
    pub actor: usize,
    pub terminal: bool,
    pub edges: Vec<Edge>,
    pub legal: Vec<usize>,
    pub visits: u32,
    pub value_sum_p0: f64,
}

pub(crate) fn state_actor(state: &GameState) -> usize {
    state
        .pending_choice
        .as_ref()
        .map_or(state.active_player, |p| p.player)
}

impl Node {
    pub fn make(state: GameState) -> Node {
        let terminal = state.phase == Phase::Complete;
        let actor = if terminal { 0 } else { state_actor(&state) };
        let legal = if terminal {
            Vec::new()
        } else {
            legal_action_indices(&state)
        };
        Node {
            state,
            actor,
            terminal,
            edges: Vec::new(),
            legal,
            visits: 0,
            value_sum_p0: 0.0,
        }
    }

    pub fn value_p0(&self) -> f64 {
        if self.visits > 0 {
            self.value_sum_p0 / self.visits as f64
        } else {
            0.0
        }
    }

    fn expand<E: Eval>(&mut self, eval: &E) -> PyResult<f64> {
        let (value_p0, priors) = eval.evaluate(&self.state)?;
        if !self.terminal {
            self.edges = self
                .legal
                .iter()
                .enumerate()
                .map(|(j, &index)| {
                    let action = decode_action(&self.state, index);
                    let specs = chance::chance_signature(&self.state, &action);
                    Edge {
                        action_index: index,
                        prior: priors[j],
                        specs,
                        children: Vec::new(),
                        visits: 0,
                        value_sum_p0: 0.0,
                        probability_weighted: false,
                        fixed_support: false,
                    }
                })
                .collect();
        }
        Ok(value_p0)
    }

    fn select(&self, c_puct: f64) -> usize {
        let sign = if self.actor == 0 { 1.0 } else { -1.0 };
        let total = (self.visits.max(1) as f64).sqrt();
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, edge) in self.edges.iter().enumerate() {
            let q = sign * edge.q_p0();
            let score = q + c_puct * edge.prior * total / (1.0 + edge.visits as f64);
            if score > best_score {
                best = i;
                best_score = score;
            }
        }
        best
    }
}

/// Descend one edge: sample its chance chain and materialize/reuse the child,
/// keyed by the observable key. Returns the child's index in `edge.children`.
fn closed_child(node: &mut Node, edge_idx: usize, rng: &mut Rng) -> usize {
    if node.edges[edge_idx].fixed_support {
        let children = &mut node.edges[edge_idx].children;
        assert!(!children.is_empty(), "fixed-support edge has no children");
        let target = rng.next_float();
        let idx = fixed_support_index(
            children.iter().map(|(_, child)| {
                child
                    .probability
                    .expect("fixed-support child needs a re-normalised weight")
            }),
            target,
        )
        .unwrap_or(children.len() - 1);
        children[idx].1.samples += 1;
        return idx;
    }
    let (outcomes, probability, key) = if node.edges[edge_idx].specs.is_empty() {
        (Vec::new(), Some(1.0), Vec::new())
    } else {
        let specs = &node.edges[edge_idx].specs;
        chance::sample_outcomes(&node.state, specs, rng)
    };
    if let Some(idx) = node.edges[edge_idx]
        .children
        .iter()
        .position(|(k, _)| *k == key)
    {
        node.edges[edge_idx].children[idx].1.samples += 1;
        return idx;
    }
    let action = decode_action(&node.state, node.edges[edge_idx].action_index);
    let mut child_state = node.state.clone();
    child_state
        .apply_with_chance(&action, &outcomes)
        .expect("sampled chance outcome must be valid");
    let child = Child {
        probability,
        node: Box::new(Node::make(child_state)),
        samples: 1,
    };
    node.edges[edge_idx].children.push((key, child));
    node.edges[edge_idx].children.len() - 1
}

/// One simulation from `node` (player-0-relative leaf value). `forced` picks the
/// edge at this level (used at the root); deeper levels select via PUCT.
fn descend<E: Eval>(
    node: &mut Node,
    forced: Option<usize>,
    eval: &E,
    rng: &mut Rng,
    c_puct: f64,
) -> PyResult<f64> {
    if node.terminal {
        let v = terminal_value_p0(&node.state);
        node.visits += 1;
        node.value_sum_p0 += v;
        return Ok(v);
    }
    if node.edges.is_empty() {
        let v = node.expand(eval)?;
        node.visits += 1;
        node.value_sum_p0 += v;
        return Ok(v);
    }
    let edge_idx = forced.unwrap_or_else(|| node.select(c_puct));
    let child_idx = closed_child(node, edge_idx, rng);
    let v = {
        let child = &mut *node.edges[edge_idx].children[child_idx].1.node;
        descend(child, None, eval, rng, c_puct)?
    };
    node.edges[edge_idx].visits += 1;
    node.edges[edge_idx].value_sum_p0 += v;
    node.visits += 1;
    node.value_sum_p0 += v;
    Ok(v)
}

#[cfg(test)]
mod fixed_support_tests {
    use super::fixed_support_index;

    /// The golden table is pinned identically in
    /// `test_search.py::test_fixed_support_index_matches_the_rust_golden_table`;
    /// both languages must map the same (weights, draw) to the same child.
    #[test]
    fn golden_table_matches_python() {
        let uniform = [0.25, 0.25, 0.25, 0.25];
        let skewed = [0.1, 0.7, 0.2];
        let cases: &[(&[f64], f64, usize)] = &[
            (&uniform, 0.0, 0),
            (&uniform, 0.249_999, 0),
            (&uniform, 0.25, 1),
            (&uniform, 0.5, 2),
            (&uniform, 0.999_999, 3),
            (&skewed, 0.09, 0),
            (&skewed, 0.1, 1),
            (&skewed, 0.799_999, 1),
            (&skewed, 0.8, 2),
            (&[1.0], 0.0, 0),
            (&[1.0], 0.999_999, 0),
        ];
        for &(weights, target, expected) in cases {
            assert_eq!(
                fixed_support_index(weights.iter().copied(), target),
                Some(expected),
                "weights {weights:?} target {target}"
            );
        }
    }

    #[test]
    fn draw_past_the_last_cumulative_sum_falls_through_to_the_caller() {
        // Float residue: the caller takes the last child rather than nothing.
        assert_eq!(fixed_support_index([0.5, 0.5], 1.0), None);
        assert_eq!(fixed_support_index([], 0.0), None);
    }
}

/// Build a closed tree from `state` with a fixed round-robin root-edge schedule
/// (the F3.2 stand-in for the F3.3 Gumbel root). Root is expanded and counted
/// once, then `sims` descents cycle through the root edges.
pub fn closed_tree_fixed<E: Eval>(
    state: &GameState,
    sims: usize,
    eval: &E,
    seed: u64,
    c_puct: f64,
) -> PyResult<Node> {
    let root_state = state.clone();
    let mut root = Node::make(root_state);
    let v = root.expand(eval)?;
    root.visits += 1;
    root.value_sum_p0 += v;
    let mut rng = Rng::new(seed);
    let n_edges = root.edges.len().max(1);
    for i in 0..sims {
        descend(&mut root, Some(i % n_edges), eval, &mut rng, c_puct)?;
    }
    Ok(root)
}

// --- F3.3: force-expansion + Gumbel root --------------------------------------

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub sims: usize,
    pub top_k: usize,
    pub c_puct: f64,
    pub c_visit: f64,
    pub c_scale: f64,
    pub seed: u64,
    pub force_expand_root_chance: bool,
    /// Root action selection: false = Gumbel top-k + sequential halving (the
    /// training-target generator), true = plain PUCT at the root with argmax
    /// visits (what the advisor runs, and what evaluation should measure).
    pub puct_root: bool,
    /// Root exploration noise, applied ONLY when `puct_root` is set
    /// (`search.py::SearchConfig.dirichlet_epsilon`). Zero is off. The Gumbel
    /// root carries exploration in its keys, so noise there would double up;
    /// PUCT has no other source and without it self-play collapses toward
    /// deterministic lines.
    pub dirichlet_epsilon: f64,
    /// Dirichlet concentration. Convention is `alpha ~ 10 / branching`; 7WD's
    /// median 4 / mean 5.6 legal actions put it near 1.8, six times
    /// Kingdomino's 0.3. See `search.py::SearchConfig.dirichlet_alpha`.
    pub dirichlet_alpha: f64,
    /// Common root AgeDeal samples per legal action. Zero preserves the legacy
    /// independently-sampled behavior.
    pub age_deal_samples: usize,
    /// Offsets per first-reveal stratum on a PURE double card-reveal root edge
    /// (`search.py::SearchConfig.double_reveal_offsets`). Zero keeps forced
    /// expansion exhaustive; a positive `X` keeps the balanced `n * X` support
    /// and CLOSES the edge.
    pub double_reveal_offsets: usize,
    /// Phase 2 follow-up: interleave the sequential-halving round instead of
    /// blocking it. Sequential halving fixes only *how many* simulations each
    /// surviving candidate gets per round, not their order, so visiting
    /// `c0, c1, .., ck, c0, c1, ..` is as faithful as `c0 x per_action, c1 x
    /// per_action, ..`. The blocked order came from the Python reference
    /// (`search.py`), and it is what holds realized wave width at 1.19: with
    /// `per_action >= 2` the next simulation always repeats the current
    /// candidate, so the conflict-free rule cuts every wave to width 1.
    ///
    /// This changes which leaves a round visits, hence every search output — it
    /// is a different, equally valid sample, not a refactor.
    pub round_robin_candidates: bool,
    /// Phase 2: forbid two in-flight simulations in the same root candidate's
    /// subtree. A wave is cut short rather than admitting the second one, so
    /// every simulation in a wave descends a subtree no other member touches —
    /// which makes `leaf_batch > 1` an exact batching of `leaf_batch = 1`
    /// instead of a virtual-loss approximation of it. The taper is a consequence
    /// of the invariant, not a configured schedule.
    pub conflict_free_waves: bool,
}

/// Root prior over the legal set, renormalised. Edge priors are already
/// restricted to legal actions, but they are not guaranteed to sum to one.
pub fn root_prior_from(priors: impl Iterator<Item = f64>) -> Vec<f64> {
    let priors: Vec<f64> = priors.collect();
    let mass: f64 = priors.iter().sum();
    if mass > 0.0 {
        priors.iter().map(|p| p / mass).collect()
    } else {
        let uniform = 1.0 / (priors.len().max(1) as f64);
        vec![uniform; priors.len()]
    }
}

pub struct SearchResult {
    pub action_index: usize,
    pub action_value: f64,
    pub root_value: f64,
    pub visits: Vec<u32>,        // aligned to root.legal
    pub policy_target: Vec<f64>, // aligned to root.legal
    /// The network's root policy over `root.legal`, renormalised over the legal
    /// set. Recorded so the improvement the search actually produced --
    /// KL(policy_target || prior) -- is measurable rather than assumed: at a
    /// small budget the Gumbel guarantee bounds that quantity below by zero and
    /// says nothing about its size.
    pub prior: Vec<f64>,
    pub gumbel_topk: Vec<usize>, // action indices
    pub sims: usize,
}

/// Gumbel-AlphaZero sigma over MIN-MAX NORMALISED completed Q values.
///
/// Port of Python's `GumbelMCTS._sigma`. The paper (and mctx's
/// `qtransform_completed_by_mix_value`) applies the (c_visit + max_visits) *
/// c_scale factor to a Q rescaled to [0, 1] across the root's actions; applied
/// to a raw actor-relative q in [-1, 1] it swamped the log-prior entirely.
///
/// Operation order is load-bearing: `scale * (q - low) / span` must group as
/// `(scale * (q - low)) / span` to stay bit-identical with Python's
/// left-associative evaluation.
pub fn sigma_vector(cfg: &SearchConfig, completed: &[f64], max_visits: u32) -> Vec<f64> {
    let low = completed.iter().copied().fold(f64::INFINITY, f64::min);
    let high = completed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (high - low).max(1e-8);
    let scale = (cfg.c_visit + max_visits as f64) * cfg.c_scale;
    completed
        .iter()
        .map(|&q| scale * (q - low) / span)
        .collect()
}

/// Materialize + evaluate every enumerable chance child of each root edge (AGE_DEAL
/// stays sampled), marking those edges probability-weighted — the closed-mode
/// catastrophe-coverage toggle (port of `_force_expand_root`).
///
/// With `cfg.double_reveal_offsets` set, a pure double card-reveal edge keeps
/// the balanced `n * X` support instead and is CLOSED against later growth.
fn force_expand_root<E: Eval>(root: &mut Node, eval: &E, cfg: &SearchConfig) -> PyResult<()> {
    for edge in &mut root.edges {
        if edge.specs.is_empty()
            || edge
                .specs
                .iter()
                .any(|s| s.kind == crate::chance::ChanceKind::AgeDeal)
        {
            continue;
        }
        let action = decode_action(&root.state, edge.action_index);
        let balanced = chance::balanced_double_reveal_chains(
            &root.state,
            &edge.specs,
            cfg.double_reveal_offsets,
            cfg.seed,
        );
        let capped = balanced.is_some();
        let chains =
            balanced.unwrap_or_else(|| chance::enumerate_chains(&root.state, &edge.specs));
        for (outcomes, probability, key) in chains {
            if edge.children.iter().any(|(k, _)| *k == key) {
                continue;
            }
            let mut child_state = root.state.clone();
            child_state
                .apply_with_chance(&action, &outcomes)
                .expect("enumerated outcome must be valid");
            let mut child_node = Node::make(child_state);
            let (value_p0, _) = eval.evaluate(&child_node.state)?;
            child_node.visits = 1;
            child_node.value_sum_p0 = value_p0;
            edge.children.push((
                key,
                Child {
                    probability: Some(probability),
                    node: Box::new(child_node),
                    samples: 0,
                },
            ));
        }
        // The retained children must carry the full (re-normalised) chance mass
        // before the edge trusts the invariant in `q_p0` (port of Python's
        // check). A capped edge is additionally closed against growth.
        let mass: f64 = edge
            .children
            .iter()
            .map(|(_, c)| c.probability.unwrap_or(0.0))
            .sum();
        if (mass - 1.0).abs() > 1e-9 {
            return Err(PyValueError::new_err(format!(
                "force-expanded edge {} holds probability mass {mass} != 1",
                edge.action_index
            )));
        }
        edge.probability_weighted = true;
        edge.fixed_support = capped;
    }
    Ok(())
}

/// Plain PUCT from the root: a port of `GumbelMCTS._puct_root`.
///
/// The Gumbel root exists to make a small fixed budget produce an unbiased
/// policy-improvement TARGET. Evaluation is not building a target, and the
/// Gumbel keys perturb which candidates get searched at all, so competitive
/// play selects at the root by PUCT like every node below it and plays argmax
/// visits.
/// Blend Dirichlet noise into the root edges' priors, in place.
///
/// Mirrors `search.py::GumbelMCTS._add_dirichlet_noise`, and draws from the same
/// `PortableRng` stream so a noise-on search stays bit-comparable with Python --
/// the property Kingdomino gave up by drawing from numpy, which forced its
/// equivalence gate to run at eps=0.
///
/// The unit-sum noise is scaled to the priors' existing mass so `eps` means
/// "fraction of prior mass replaced" whatever that mass is, and so the total
/// PUCT sees is unchanged.
///
/// `tree` and `tree_resumable` define separate `Node`/`Edge` types, so the blend
/// itself lives here over a bare prior slice and each searcher passes its own
/// edges in. One implementation, two thin adapters -- duplicating the arithmetic
/// is exactly how the two paths would drift apart.
pub(crate) fn blend_dirichlet(priors: &mut [f64], cfg: &SearchConfig, rng: &mut Rng) {
    if !cfg.puct_root || cfg.dirichlet_epsilon <= 0.0 || priors.is_empty() {
        return;
    }
    // Standard AlphaZero blend. Priors are expected to be a distribution:
    // production self-play normalises them in `blend_priors` before they ever
    // reach a root. An earlier version scaled the noise by the observed prior
    // mass so `eps` would keep its meaning on unnormalised input -- that was a
    // mistake. It bought nothing in production and, on a mock evaluator whose
    // priors summed to 2.53, it preserved a 2.53x inflated `c_puct * prior`
    // exploration term, concealing the contract violation instead of exposing
    // it.
    let eps = cfg.dirichlet_epsilon;
    let noise = rng.dirichlet(cfg.dirichlet_alpha, priors.len());
    for (prior, sample) in priors.iter_mut().zip(noise) {
        *prior = (1.0 - eps) * *prior + eps * sample;
    }
}

fn add_dirichlet_noise(root: &mut Node, cfg: &SearchConfig, rng: &mut Rng) {
    let mut priors: Vec<f64> = root.edges.iter().map(|e| e.prior).collect();
    blend_dirichlet(&mut priors, cfg, rng);
    for (edge, prior) in root.edges.iter_mut().zip(priors) {
        edge.prior = prior;
    }
}

fn puct_root<E: Eval>(
    mut root: Node,
    eval: &E,
    cfg: &SearchConfig,
    sign: f64,
    root_value: f64,
    legal: Vec<usize>,
) -> PyResult<(SearchResult, Node)> {
    let n = root.edges.len();
    let mut rng = Rng::new(cfg.seed);
    // Snapshot BEFORE noise. `prior` is recorded as the network's opinion and is
    // what every KL diagnostic scores against; blending noise into what gets
    // reported would silently redefine what a buffer row means. The terminal
    // policy-target fallback below uses the same clean copy.
    let clean_priors: Vec<f64> = root.edges.iter().map(|e| e.prior).collect();
    add_dirichlet_noise(&mut root, cfg, &mut rng);
    for _ in 0..cfg.sims {
        descend(&mut root, None, eval, &mut rng, cfg.c_puct)?;
    }
    let visits: Vec<u32> = root.edges.iter().map(|e| e.visits).collect();
    let completed: Vec<f64> = root
        .edges
        .iter()
        .map(|e| {
            if e.visits > 0 || e.probability_weighted {
                sign * e.q_p0()
            } else {
                root_value
            }
        })
        .collect();
    // Left-fold in legal order to match Python's `sum`.
    let total: f64 = visits.iter().fold(0.0_f64, |a, &b| a + b as f64);
    let policy_target: Vec<f64> = if total > 0.0 {
        visits.iter().map(|&v| v as f64 / total).collect()
    } else {
        // Every simulation hit a terminal root edge; fall back to the prior --
        // the CLEAN one, so a rare fallback never emits a noise-shaped label.
        let mass: f64 = clean_priors.iter().fold(0.0_f64, |a, &p| a + p);
        let mass = if mass > 0.0 { mass } else { 1.0 };
        clean_priors.iter().map(|p| p / mass).collect()
    };
    // Python's `max` returns the FIRST maximum in legal order; a strict `>`
    // over the same order agrees on ties.
    let mut best = 0usize;
    let mut best_visits = 0u32;
    for j in 0..n {
        if visits[j] > best_visits {
            best_visits = visits[j];
            best = j;
        }
    }
    let result = SearchResult {
        action_index: legal[best],
        action_value: completed[best],
        root_value: sign * root.value_p0(),
        visits,
        policy_target,
        prior: root_prior_from(clean_priors.iter().copied()),
        // No Gumbel top-k exists here; an invented one would let a buffer row
        // claim a candidate set that never happened.
        gumbel_topk: Vec::new(),
        sims: cfg.sims,
    };
    Ok((result, root))
}

/// Full closed search with a Gumbel root (top-k + sequential halving +
/// completed-Q policy target), a port of `_gumbel_root` + `_search_closed`.
/// Returns the result and the built tree (for the digest gate).
pub fn search_closed<E: Eval>(
    state: &GameState,
    eval: &E,
    cfg: &SearchConfig,
) -> PyResult<(SearchResult, Node)> {
    if cfg.sims < 1 || cfg.top_k < 1 {
        return Err(PyValueError::new_err("sims and top_k must be positive"));
    }
    let mut root = Node::make(state.clone());
    if root.terminal || root.legal.is_empty() {
        return Err(PyValueError::new_err(
            "cannot search a terminal or action-less root",
        ));
    }
    let root_value_p0 = root.expand(eval)?;
    root.visits += 1;
    root.value_sum_p0 += root_value_p0;
    if cfg.force_expand_root_chance {
        force_expand_root(&mut root, eval, cfg)?;
    }
    let sign = if root.actor == 0 { 1.0 } else { -1.0 };
    let root_value = sign * root_value_p0;
    let n = root.edges.len();
    let legal: Vec<usize> = root.legal.clone();

    if cfg.puct_root {
        return puct_root(root, eval, cfg, sign, root_value, legal);
    }

    // Gumbel keys (one per legal action, in sorted order) then the per-edge
    // priors, log-priors, and any forced (probability-weighted) initial Q.
    let mut rng = Rng::new(cfg.seed);
    let log_prior: Vec<f64> = root.edges.iter().map(|e| e.prior.max(1e-12).ln()).collect();
    let gumbel: Vec<f64> = (0..n).map(|_| rng.gumbel()).collect();
    let initial_q: Vec<Option<f64>> = root
        .edges
        .iter()
        .map(|e| {
            if e.probability_weighted {
                Some(sign * e.q_p0())
            } else {
                None
            }
        })
        .collect();

    let mut q_hat: Vec<Option<f64>> = vec![None; n];
    let mut visits: Vec<u32> = vec![0; n];
    let completed_q = |j: usize, q_hat: &[Option<f64>]| -> f64 {
        q_hat[j].or(initial_q[j]).unwrap_or(root_value)
    };

    let mut candidates: Vec<usize> = (0..n).collect();
    candidates.sort_by(|&a, &b| {
        (gumbel[b] + log_prior[b])
            .partial_cmp(&(gumbel[a] + log_prior[a]))
            .unwrap()
    });
    candidates.truncate(cfg.top_k.min(n).max(0));
    if candidates.is_empty() {
        candidates.push(0);
    }
    let topk: Vec<usize> = candidates.iter().map(|&j| legal[j]).collect();

    let budget = cfg.sims;
    let mut sims_used = 0usize;
    let rounds_total = ((candidates.len().max(2) as f64).log2().ceil() as usize).max(1);
    let mut round_index = 0usize;
    while sims_used < budget {
        let rounds_remaining = rounds_total.saturating_sub(round_index).max(1);
        let per_action = ((budget - sims_used) / (rounds_remaining * candidates.len())).max(1);
        'outer: for idx in 0..candidates.len() {
            let j = candidates[idx];
            for _ in 0..per_action {
                if sims_used >= budget {
                    break 'outer;
                }
                descend(&mut root, Some(j), eval, &mut rng, cfg.c_puct)?;
                q_hat[j] = Some(sign * root.edges[j].q_p0());
                visits[j] = root.edges[j].visits;
                sims_used += 1;
            }
        }
        if candidates.len() > 1 {
            let max_visits = visits.iter().copied().max().unwrap_or(0);
            let completed: Vec<f64> = (0..n).map(|j| completed_q(j, &q_hat)).collect();
            let sig = sigma_vector(cfg, &completed, max_visits);
            candidates.sort_by(|&a, &b| {
                let ka = gumbel[a] + log_prior[a] + sig[a];
                let kb = gumbel[b] + log_prior[b] + sig[b];
                kb.partial_cmp(&ka).unwrap()
            });
            candidates.truncate((candidates.len() / 2).max(1));
        }
        round_index += 1;
    }

    let max_visits = visits.iter().copied().max().unwrap_or(0);
    // Improved policy over ALL legal actions (completed Q); sigma normalises
    // over this same full-legal window so the halving key, the played action
    // and the target share one scale.
    let completed: Vec<f64> = (0..n).map(|j| completed_q(j, &q_hat)).collect();
    let sig = sigma_vector(cfg, &completed, max_visits);
    // best = first argmax over the surviving candidates.
    let mut best = candidates[0];
    let mut best_score = f64::NEG_INFINITY;
    for &j in &candidates {
        let s = gumbel[j] + log_prior[j] + sig[j];
        if s > best_score {
            best_score = s;
            best = j;
        }
    }

    // Left-fold the normalizer in legal order to match Python's sum.
    let logits: Vec<f64> = (0..n).map(|j| log_prior[j] + sig[j]).collect();
    let peak = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits.iter().map(|&v| (v - peak).exp()).collect();
    let total = weights.iter().fold(0.0_f64, |a, &b| a + b);
    let policy_target: Vec<f64> = weights.iter().map(|&w| w / total).collect();

    let result = SearchResult {
        action_index: legal[best],
        action_value: completed_q(best, &q_hat),
        root_value: sign * root.value_p0(),
        visits,
        policy_target,
        prior: root_prior_from(root.edges.iter().map(|e| e.prior)),
        gumbel_topk: topk,
        sims: sims_used,
    };
    Ok((result, root))
}

/// Canonical depth-first serialization for the equivalence gate. Includes the
/// node actor/terminal flag and the full state fingerprint (so equal digests
/// imply equal states), plus edge stats and child keys serialized with explicit
/// part counts and per-part lengths ([[1],[2]] and [[1,2]] must not collide).
pub fn digest(node: &Node, out: &mut Vec<f64>) {
    out.push(node.visits as f64);
    out.push(node.value_sum_p0);
    out.push(node.actor as f64);
    out.push(if node.terminal { 1.0 } else { 0.0 });
    let fp = node.state.fingerprint();
    out.push(fp.len() as f64);
    out.extend(fp.iter().map(|&x| x as f64));
    out.push(node.edges.len() as f64);
    for edge in &node.edges {
        out.push(edge.action_index as f64);
        out.push(edge.visits as f64);
        out.push(edge.value_sum_p0);
        out.push(edge.prior);
        out.push(if edge.probability_weighted { 1.0 } else { 0.0 });
        out.push(edge.children.len() as f64);
        for (key, child) in &edge.children {
            out.push(key.len() as f64); // number of parts
            for part in key {
                out.push(part.len() as f64); // length of this part
                out.extend(part.iter().map(|&k| k as f64));
            }
            out.push(child.samples as f64);
            out.push(child.probability.unwrap_or(f64::NAN));
            digest(&child.node, out);
        }
    }
}
