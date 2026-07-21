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
}

impl Edge {
    pub fn q_p0(&self) -> f64 {
        if self.probability_weighted {
            self.children
                .iter()
                .map(|(_, c)| c.probability.expect("weighted child needs probability") * c.node.value_p0())
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

pub struct SearchConfig {
    pub sims: usize,
    pub top_k: usize,
    pub c_puct: f64,
    pub c_visit: f64,
    pub c_scale: f64,
    pub seed: u64,
    pub force_expand_root_chance: bool,
}

pub struct SearchResult {
    pub action_index: usize,
    pub action_value: f64,
    pub root_value: f64,
    pub visits: Vec<u32>,        // aligned to root.legal
    pub policy_target: Vec<f64>, // aligned to root.legal
    pub gumbel_topk: Vec<usize>, // action indices
    pub sims: usize,
}

fn sigma(cfg: &SearchConfig, q: f64, max_visits: u32) -> f64 {
    (cfg.c_visit + max_visits as f64) * cfg.c_scale * q
}

/// Materialize + evaluate every enumerable chance child of each root edge (AGE_DEAL
/// stays sampled), marking those edges probability-weighted — the closed-mode
/// catastrophe-coverage toggle (port of `_force_expand_root`).
fn force_expand_root<E: Eval>(root: &mut Node, eval: &E) -> PyResult<()> {
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
        for (outcomes, probability, key) in chance::enumerate_chains(&root.state, &edge.specs) {
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
        // The enumerated children must carry the full chance mass before the edge
        // trusts the invariant in `q_p0` (port of Python's check).
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
    }
    Ok(())
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
        force_expand_root(&mut root, eval)?;
    }
    let sign = if root.actor == 0 { 1.0 } else { -1.0 };
    let root_value = sign * root_value_p0;
    let n = root.edges.len();
    let legal: Vec<usize> = root.legal.clone();

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
            candidates.sort_by(|&a, &b| {
                let ka = gumbel[a] + log_prior[a] + sigma(cfg, completed_q(a, &q_hat), max_visits);
                let kb = gumbel[b] + log_prior[b] + sigma(cfg, completed_q(b, &q_hat), max_visits);
                kb.partial_cmp(&ka).unwrap()
            });
            candidates.truncate((candidates.len() / 2).max(1));
        }
        round_index += 1;
    }

    let max_visits = visits.iter().copied().max().unwrap_or(0);
    // best = first argmax over the surviving candidates.
    let mut best = candidates[0];
    let mut best_score = f64::NEG_INFINITY;
    for &j in &candidates {
        let s = gumbel[j] + log_prior[j] + sigma(cfg, completed_q(j, &q_hat), max_visits);
        if s > best_score {
            best_score = s;
            best = j;
        }
    }

    // Improved policy over ALL legal actions (completed Q); left-fold the
    // normalizer in legal order to match Python's sum.
    let logits: Vec<f64> = (0..n)
        .map(|j| log_prior[j] + sigma(cfg, completed_q(j, &q_hat), max_visits))
        .collect();
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
