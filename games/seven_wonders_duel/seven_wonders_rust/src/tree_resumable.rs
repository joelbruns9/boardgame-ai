//! F4.1 arena-backed, resumable closed search.
//!
//! The F3.3 recursive search in `tree.rs` remains the permanent sequential
//! oracle.  This module deliberately duplicates the small amount of tree math
//! needed for an independently gateable refactor: nodes have stable arena IDs,
//! selection/materialization records stable path steps, evaluation is a
//! separate phase, and backup consumes the recorded path.  At `leaf_batch=1`
//! no pending-count approximation is active, so the result and canonical tree
//! digest must match F3.3 exactly.

use crate::chance::{self, ChanceKind, ChanceSpec};
use crate::codec::{decode_action, legal_action_indices};
use crate::eval::{terminal_value_p0, Eval};
use crate::rng::Rng;
use crate::state::{GameState, Phase};
use crate::tree::{SearchConfig, SearchResult};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

pub type NodeId = usize;

pub struct Child {
    pub probability: Option<f64>,
    pub node_id: NodeId,
    pub samples: u32,
}

pub struct Edge {
    pub action_index: usize,
    pub prior: f64,
    pub specs: Vec<ChanceSpec>,
    pub children: Vec<(Vec<Vec<i32>>, Child)>,
    pub visits: u32,
    pub value_sum_p0: f64,
    pub incomplete: u32,
    pub probability_weighted: bool,
    pub paired_sampled: bool,
}

pub struct Node {
    pub state: GameState,
    pub actor: usize,
    pub terminal: bool,
    pub edges: Vec<Edge>,
    pub legal: Vec<usize>,
    pub visits: u32,
    pub value_sum_p0: f64,
    pub incomplete: u32,
    /// F4.5 forced-child cache. Force expansion seeds the child's value/visit
    /// exactly as before but retains priors for the first ordinary visit.
    cached_evaluation: Option<(f64, Vec<f64>)>,
}

impl Node {
    fn make(state: GameState) -> Self {
        let terminal = state.phase == Phase::Complete;
        let actor = if terminal {
            0
        } else {
            crate::tree::state_actor(&state)
        };
        let legal = if terminal {
            Vec::new()
        } else {
            legal_action_indices(&state)
        };
        Self {
            state,
            actor,
            terminal,
            edges: Vec::new(),
            legal,
            visits: 0,
            value_sum_p0: 0.0,
            incomplete: 0,
            cached_evaluation: None,
        }
    }

    fn value_p0(&self) -> f64 {
        if self.visits > 0 {
            self.value_sum_p0 / self.visits as f64
        } else {
            0.0
        }
    }
}

/// Stable storage for every node materialized by one search.
pub struct Arena {
    nodes: Vec<Node>,
    root_id: NodeId,
}

impl Arena {
    fn new(root: Node) -> Self {
        Self {
            nodes: vec![root],
            root_id: 0,
        }
    }

    fn push(&mut self, node: Node) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    fn edge_q_p0(&self, node_id: NodeId, edge_idx: usize) -> f64 {
        let edge = &self.nodes[node_id].edges[edge_idx];
        if edge.probability_weighted {
            edge.children
                .iter()
                .map(|(_, child)| {
                    child.probability.expect("weighted child needs probability")
                        * self.nodes[child.node_id].value_p0()
                })
                .sum()
        } else if edge.visits > 0 {
            edge.value_sum_p0 / edge.visits as f64
        } else {
            0.0
        }
    }

    fn select(&self, node_id: NodeId, c_puct: f64) -> usize {
        let node = &self.nodes[node_id];
        let sign = if node.actor == 0 { 1.0 } else { -1.0 };
        let total = ((node.visits + node.incomplete).max(1) as f64).sqrt();
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, edge) in node.edges.iter().enumerate() {
            let q = sign * self.edge_q_p0(node_id, i);
            let score = q + c_puct * edge.prior * total
                / (1.0 + edge.visits as f64 + edge.incomplete as f64);
            if score > best_score {
                best = i;
                best_score = score;
            }
        }
        best
    }

    fn incomplete_total(&self) -> u64 {
        self.nodes
            .iter()
            .map(|node| {
                node.incomplete as u64
                    + node
                        .edges
                        .iter()
                        .map(|edge| edge.incomplete as u64)
                        .sum::<u64>()
            })
            .sum()
    }
}

fn expand(arena: &mut Arena, node_id: NodeId, priors: Vec<f64>) -> PyResult<()> {
    let legal_len = arena.nodes[node_id].legal.len();
    if priors.len() != legal_len {
        return Err(PyValueError::new_err(format!(
            "evaluator returned {} priors for {legal_len} legal actions",
            priors.len()
        )));
    }
    if arena.nodes[node_id].terminal {
        return Ok(());
    }
    let state = arena.nodes[node_id].state.clone();
    let legal = arena.nodes[node_id].legal.clone();
    arena.nodes[node_id].edges = legal
        .iter()
        .enumerate()
        .map(|(j, &index)| {
            let action = decode_action(&state, index);
            Edge {
                action_index: index,
                prior: priors[j],
                specs: chance::chance_signature(&state, &action),
                children: Vec::new(),
                visits: 0,
                value_sum_p0: 0.0,
                incomplete: 0,
                probability_weighted: false,
                paired_sampled: false,
            }
        })
        .collect();
    Ok(())
}

/// One stable step in a selected pending path. `chance_child_id` is the arena
/// node reached through `edge_id`; it remains valid if other children/nodes are
/// materialized while the request is outstanding.
#[derive(Clone, Copy, Debug)]
pub struct PathStep {
    pub node_id: NodeId,
    pub edge_id: usize,
    pub chance_child_id: NodeId,
}

#[derive(Debug)]
struct PendingSimulation {
    path: Vec<PathStep>,
    leaf_id: NodeId,
    root_edge: usize,
    has_incomplete: bool,
}

/// Metadata needed to align an evaluator response to its pending simulation.
/// F4.2 will collect several of these into a leaf wave; F4.1 issues exactly one.
pub struct EvalLeafRequest {
    pub leaf_id: NodeId,
    pub actor: usize,
    pub legal: Vec<usize>,
}

pub struct EvalBatchRequest {
    pub request_id: u64,
    pub leaves: Vec<EvalLeafRequest>,
    pub forced: bool,
}

pub enum SearchEvent {
    Evaluation(EvalBatchRequest),
    Complete,
}

fn closed_child(arena: &mut Arena, node_id: NodeId, edge_idx: usize, rng: &mut Rng) -> NodeId {
    if arena.nodes[node_id].edges[edge_idx].paired_sampled {
        let target = rng.next_float();
        let children = &mut arena.nodes[node_id].edges[edge_idx].children;
        let mut cumulative = 0.0;
        let mut selected = children.len().saturating_sub(1);
        for (index, (_, child)) in children.iter().enumerate() {
            cumulative += child
                .probability
                .expect("paired sampled child needs an empirical probability");
            if target < cumulative {
                selected = index;
                break;
            }
        }
        children[selected].1.samples += 1;
        return children[selected].1.node_id;
    }
    let (outcomes, probability, key) = {
        let node = &arena.nodes[node_id];
        let edge = &node.edges[edge_idx];
        if edge.specs.is_empty() {
            (Vec::new(), Some(1.0), Vec::new())
        } else {
            chance::sample_outcomes(&node.state, &edge.specs, rng)
        }
    };
    if let Some(child_idx) = arena.nodes[node_id].edges[edge_idx]
        .children
        .iter()
        .position(|(candidate, _)| *candidate == key)
    {
        let child = &mut arena.nodes[node_id].edges[edge_idx].children[child_idx].1;
        child.samples += 1;
        return child.node_id;
    }

    let (mut child_state, action_index) = {
        let node = &arena.nodes[node_id];
        (node.state.clone(), node.edges[edge_idx].action_index)
    };
    let action = decode_action(&child_state, action_index);
    child_state
        .apply_with_chance(&action, &outcomes)
        .expect("sampled chance outcome must be valid");
    let child_id = arena.push(Node::make(child_state));
    arena.nodes[node_id].edges[edge_idx].children.push((
        key,
        Child {
            probability,
            node_id: child_id,
            samples: 1,
        },
    ));
    child_id
}

fn select_leaf(
    arena: &mut Arena,
    root_id: NodeId,
    forced_edge: usize,
    rng: &mut Rng,
    c_puct: f64,
) -> (PendingSimulation, Option<(f64, Option<Vec<f64>>)>) {
    let mut path = Vec::new();
    let mut node_id = root_id;
    let mut forced = Some(forced_edge);
    loop {
        let node = &arena.nodes[node_id];
        if node.terminal {
            let value = terminal_value_p0(&node.state);
            return (
                PendingSimulation {
                    path,
                    leaf_id: node_id,
                    root_edge: forced_edge,
                    has_incomplete: false,
                },
                Some((value, None)),
            );
        }
        if node.edges.is_empty() {
            let cached = arena.nodes[node_id].cached_evaluation.take();
            return (
                PendingSimulation {
                    path,
                    leaf_id: node_id,
                    root_edge: forced_edge,
                    has_incomplete: false,
                },
                cached.map(|(value, priors)| (value, Some(priors))),
            );
        }
        let edge_idx = forced
            .take()
            .unwrap_or_else(|| arena.select(node_id, c_puct));
        let child_id = closed_child(arena, node_id, edge_idx, rng);
        path.push(PathStep {
            node_id,
            edge_id: edge_idx,
            chance_child_id: child_id,
        });
        node_id = child_id;
    }
}

fn mark_incomplete(arena: &mut Arena, pending: &mut PendingSimulation) {
    debug_assert!(!pending.has_incomplete);
    arena.nodes[pending.leaf_id].incomplete += 1;
    for step in &pending.path {
        let node = &mut arena.nodes[step.node_id];
        node.incomplete += 1;
        node.edges[step.edge_id].incomplete += 1;
    }
    pending.has_incomplete = true;
}

fn clear_incomplete(arena: &mut Arena, pending: &mut PendingSimulation) {
    if !pending.has_incomplete {
        return;
    }
    let leaf = &mut arena.nodes[pending.leaf_id];
    debug_assert!(leaf.incomplete > 0);
    leaf.incomplete -= 1;
    for step in &pending.path {
        let node = &mut arena.nodes[step.node_id];
        debug_assert!(node.incomplete > 0);
        debug_assert!(node.edges[step.edge_id].incomplete > 0);
        node.incomplete -= 1;
        node.edges[step.edge_id].incomplete -= 1;
    }
    pending.has_incomplete = false;
}

fn backup(arena: &mut Arena, pending: PendingSimulation, value_p0: f64) {
    let leaf = &mut arena.nodes[pending.leaf_id];
    leaf.visits += 1;
    leaf.value_sum_p0 += value_p0;
    for step in pending.path.iter().rev() {
        debug_assert!(arena.nodes[step.node_id].edges[step.edge_id]
            .children
            .iter()
            .any(|(_, child)| child.node_id == step.chance_child_id));
        let node = &mut arena.nodes[step.node_id];
        node.edges[step.edge_id].visits += 1;
        node.edges[step.edge_id].value_sum_p0 += value_p0;
        node.visits += 1;
        node.value_sum_p0 += value_p0;
    }
}

/// F4.5 semantic-safe forced-child expansion: evaluate all materialized children
/// in one prepared batch, seed the legacy value/visit, and retain priors for the
/// first ordinary visit without descending an extra ply.
const MAX_FORCED_ROWS_PER_ROOT: usize = 45_540;

fn forced_chain_bound(specs: &[ChanceSpec]) -> PyResult<usize> {
    let card_reveals = specs
        .iter()
        .filter(|spec| spec.kind == ChanceKind::CardReveal)
        .count();
    if card_reveals > 2 {
        return Err(PyValueError::new_err(format!(
            "unregistered tractable chance chain has {card_reveals} card reveals"
        )));
    }
    let mut bound = match card_reveals {
        0 => 1,
        1 => 23,
        2 => 23 * 22,
        _ => unreachable!(),
    };
    for spec in specs {
        match spec.kind {
            ChanceKind::CardReveal => {}
            ChanceKind::GreatLibraryDraw => bound *= 10,
            ChanceKind::WonderGroupReveal => bound *= 70,
            ChanceKind::AgeDeal => {
                return Err(PyValueError::new_err(
                    "AgeDeal is sample-only and cannot be force expanded",
                ));
            }
        }
    }
    Ok(bound)
}

type EnumeratedChain = (Vec<Vec<usize>>, f64, Vec<Vec<i32>>);

#[derive(Default)]
struct ForcedRows {
    nodes: Vec<NodeId>,
    by_kind: [usize; 4],
}

impl ForcedRows {
    fn extend(&mut self, other: ForcedRows) {
        self.nodes.extend(other.nodes);
        for (target, value) in self.by_kind.iter_mut().zip(other.by_kind) {
            *target += value;
        }
    }
}

/// Materialize every tractable immediate root outcome without evaluating it.
/// The returned stable arena IDs are consumed by the resumable forced phase, so
/// the global scheduler can coalesce rows from independent games.
fn materialize_forced_root(arena: &mut Arena) -> PyResult<ForcedRows> {
    let root_id = arena.root_id;
    let edge_count = arena.nodes[root_id].edges.len();
    let mut forced_nodes = Vec::new();
    let mut by_kind = [0_usize; 4];
    let mut enumeration_cache: HashMap<Vec<(u8, Vec<i32>)>, Vec<EnumeratedChain>> = HashMap::new();
    for edge_idx in 0..edge_count {
        let skip = {
            let edge = &arena.nodes[root_id].edges[edge_idx];
            edge.specs.is_empty() || edge.specs.iter().any(|s| s.kind == ChanceKind::AgeDeal)
        };
        if skip {
            continue;
        }
        let (state, action_index, specs) = {
            let root = &arena.nodes[root_id];
            (
                root.state.clone(),
                root.edges[edge_idx].action_index,
                root.edges[edge_idx].specs.clone(),
            )
        };
        let bound = forced_chain_bound(&specs)?;
        let signature: Vec<_> = specs
            .iter()
            .map(|spec| (spec.kind as u8, spec.context.clone()))
            .collect();
        let enumerated = enumeration_cache
            .entry(signature)
            .or_insert_with(|| chance::enumerate_chains(&state, &specs));
        if enumerated.len() > bound {
            return Err(PyValueError::new_err(format!(
                "force-expanded edge {action_index} has {} outcomes above registered bound {bound}",
                enumerated.len()
            )));
        }
        let metric_kind = if specs
            .iter()
            .any(|spec| spec.kind == ChanceKind::GreatLibraryDraw)
        {
            ChanceKind::GreatLibraryDraw
        } else if specs
            .iter()
            .any(|spec| spec.kind == ChanceKind::WonderGroupReveal)
        {
            ChanceKind::WonderGroupReveal
        } else {
            ChanceKind::CardReveal
        };
        by_kind[metric_kind as usize] += enumerated.len();
        let action = decode_action(&state, action_index);
        for (outcomes, probability, key) in enumerated.iter().cloned() {
            if arena.nodes[root_id].edges[edge_idx]
                .children
                .iter()
                .any(|(candidate, _)| *candidate == key)
            {
                continue;
            }
            let mut child_state = state.clone();
            child_state
                .apply_with_chance(&action, &outcomes)
                .expect("enumerated outcome must be valid");
            let child_id = arena.push(Node::make(child_state));
            forced_nodes.push(child_id);
            arena.nodes[root_id].edges[edge_idx].children.push((
                key,
                Child {
                    probability: Some(probability),
                    node_id: child_id,
                    samples: 0,
                },
            ));
        }
        let mass: f64 = arena.nodes[root_id].edges[edge_idx]
            .children
            .iter()
            .map(|(_, child)| child.probability.unwrap_or(0.0))
            .sum();
        if (mass - 1.0).abs() > 1e-9 {
            return Err(PyValueError::new_err(format!(
                "force-expanded edge {action_index} holds probability mass {mass} != 1"
            )));
        }
        arena.nodes[root_id].edges[edge_idx].probability_weighted = true;
    }
    if forced_nodes.len() > MAX_FORCED_ROWS_PER_ROOT {
        return Err(PyValueError::new_err(format!(
            "root has {} forced outcomes above registered bound {MAX_FORCED_ROWS_PER_ROOT}",
            forced_nodes.len()
        )));
    }
    Ok(ForcedRows {
        nodes: forced_nodes,
        by_kind,
    })
}

/// Materialize a common Monte-Carlo AgeDeal sample for every affected root
/// action. Each action gets its own child states/evaluations, but all actions
/// see the same sampled deals and empirical weights.
fn materialize_paired_age_deals(
    arena: &mut Arena,
    sample_count: usize,
    seed: u64,
) -> PyResult<ForcedRows> {
    if sample_count == 0 {
        return Ok(ForcedRows::default());
    }
    if sample_count > 32 {
        return Err(PyValueError::new_err(
            "AgeDeal sample count exceeds registered diagnostic maximum 32",
        ));
    }
    let root_id = arena.root_id;
    // The paired sampler exists to compare the two choices for who starts the
    // next age.  The eighth wonder-draft pick also carries an AgeDeal chance
    // spec, but Age I's starter is fixed and that root has no starter choice.
    // Treat it like ordinary sampled chance instead of evaluating a pointless
    // common-deal set for its sole action.
    if arena.nodes[root_id].state.phase != Phase::ChooseNextStartPlayer {
        return Ok(ForcedRows::default());
    }
    let age_edges: Vec<_> = arena.nodes[root_id]
        .edges
        .iter()
        .enumerate()
        .filter(|(_, edge)| {
            edge.specs
                .iter()
                .any(|spec| spec.kind == ChanceKind::AgeDeal)
        })
        .map(|(index, _)| index)
        .collect();
    if age_edges.is_empty() {
        return Ok(ForcedRows::default());
    }
    let signature = |specs: &[ChanceSpec]| -> Vec<_> {
        specs
            .iter()
            .map(|spec| (spec.kind as u8, spec.context.clone()))
            .collect()
    };
    let reference_specs = arena.nodes[root_id].edges[age_edges[0]].specs.clone();
    let reference_signature = signature(&reference_specs);
    if age_edges
        .iter()
        .any(|&edge| signature(&arena.nodes[root_id].edges[edge].specs) != reference_signature)
    {
        return Err(PyValueError::new_err(
            "root AgeDeal actions do not share one chance signature",
        ));
    }

    let root_state = arena.nodes[root_id].state.clone();
    let mut rng = Rng::new(seed ^ 0xA6E1_D3A1_5EED_0002);
    let mut samples: Vec<(Vec<Vec<usize>>, Vec<Vec<i32>>, usize)> = Vec::new();
    for _ in 0..sample_count {
        let (outcomes, _, key) = chance::sample_outcomes(&root_state, &reference_specs, &mut rng);
        if let Some((_, _, frequency)) = samples
            .iter_mut()
            .find(|(_, candidate, _)| *candidate == key)
        {
            *frequency += 1;
        } else {
            samples.push((outcomes, key, 1));
        }
    }

    let mut nodes = Vec::new();
    for edge_idx in age_edges {
        let action_index = arena.nodes[root_id].edges[edge_idx].action_index;
        let action = decode_action(&root_state, action_index);
        for (outcomes, key, frequency) in &samples {
            let mut child_state = root_state.clone();
            child_state
                .apply_with_chance(&action, outcomes)
                .expect("paired AgeDeal outcome must be valid for every root action");
            let child_id = arena.push(Node::make(child_state));
            nodes.push(child_id);
            arena.nodes[root_id].edges[edge_idx].children.push((
                key.clone(),
                Child {
                    probability: Some(*frequency as f64 / sample_count as f64),
                    node_id: child_id,
                    samples: 0,
                },
            ));
        }
        let mass: f64 = arena.nodes[root_id].edges[edge_idx]
            .children
            .iter()
            .map(|(_, child)| child.probability.unwrap_or(0.0))
            .sum();
        if (mass - 1.0).abs() > 1e-9 {
            return Err(PyValueError::new_err(format!(
                "paired AgeDeal edge {action_index} holds probability mass {mass} != 1"
            )));
        }
        arena.nodes[root_id].edges[edge_idx].probability_weighted = true;
        arena.nodes[root_id].edges[edge_idx].paired_sampled = true;
    }
    let mut by_kind = [0_usize; 4];
    by_kind[ChanceKind::AgeDeal as usize] = nodes.len();
    Ok(ForcedRows { nodes, by_kind })
}

fn sigma(cfg: &SearchConfig, q: f64, max_visits: u32) -> f64 {
    (cfg.c_visit + max_visits as f64) * cfg.c_scale * q
}

#[derive(Clone, Debug, Default)]
pub struct SearchMetrics {
    pub scheduled_simulations: usize,
    pub requested_nn_leaves: usize,
    pub unique_nn_leaves: usize,
    pub terminal_leaves: usize,
    pub collisions: usize,
    pub leaf_waves: usize,
    pub max_wave_paths: usize,
    pub max_wave_unique: usize,
    pub cached_forced_leaves: usize,
    pub forced_outcome_rows: usize,
    pub forced_rows_by_kind: [usize; 4],
    pub root_completed_q: Vec<f64>,
}

struct PendingWave {
    request_id: u64,
    simulations: Vec<PendingSimulation>,
    unique_leaf_ids: Vec<NodeId>,
}

enum PendingEvaluation {
    Forced {
        request_id: u64,
        node_ids: Vec<NodeId>,
    },
    Wave(PendingWave),
}

/// Resumable sequential-halving driver. Selection launches paths in scalar
/// schedule/RNG order, WU incomplete counts influence only deeper PUCT, and a
/// leaf wave is always drained before a halving-round reduction.
pub struct SearchSession {
    arena: Arena,
    cfg: SearchConfig,
    leaf_batch: usize,
    rng: Rng,
    sign: f64,
    root_value: f64,
    legal: Vec<usize>,
    log_prior: Vec<f64>,
    gumbel: Vec<f64>,
    initial_q: Vec<Option<f64>>,
    q_hat: Vec<Option<f64>>,
    visits: Vec<u32>,
    candidates: Vec<usize>,
    topk: Vec<usize>,
    rounds_total: usize,
    round_index: usize,
    per_action: usize,
    candidate_pos: usize,
    repeat_pos: usize,
    sims_launched: usize,
    sims_completed: usize,
    request_seq: u64,
    waiting: Option<PendingEvaluation>,
    forced_nodes: Vec<NodeId>,
    forced_cursor: usize,
    forced_finalized: bool,
    metrics: SearchMetrics,
}

impl SearchSession {
    fn new(arena: Arena, cfg: &SearchConfig, leaf_batch: usize, forced_nodes: Vec<NodeId>) -> Self {
        let root = &arena.nodes[arena.root_id];
        let sign = if root.actor == 0 { 1.0 } else { -1.0 };
        let root_value = sign * root.value_p0();
        let legal = root.legal.clone();
        let n = root.edges.len();
        let log_prior: Vec<f64> = root.edges.iter().map(|e| e.prior.max(1e-12).ln()).collect();
        let initial_q: Vec<Option<f64>> = root
            .edges
            .iter()
            .enumerate()
            .map(|(j, edge)| {
                if edge.probability_weighted {
                    Some(sign * arena.edge_q_p0(arena.root_id, j))
                } else {
                    None
                }
            })
            .collect();
        let mut rng = Rng::new(cfg.seed);
        let gumbel: Vec<f64> = (0..n).map(|_| rng.gumbel()).collect();
        let mut candidates: Vec<usize> = (0..n).collect();
        candidates.sort_by(|&a, &b| {
            (gumbel[b] + log_prior[b])
                .partial_cmp(&(gumbel[a] + log_prior[a]))
                .unwrap()
        });
        candidates.truncate(cfg.top_k.min(n));
        if candidates.is_empty() {
            candidates.push(0);
        }
        let topk = candidates.iter().map(|&j| legal[j]).collect();
        let rounds_total = ((candidates.len().max(2) as f64).log2().ceil() as usize).max(1);
        let per_action = (cfg.sims / (rounds_total * candidates.len())).max(1);
        Self {
            arena,
            cfg: SearchConfig {
                sims: cfg.sims,
                top_k: cfg.top_k,
                c_puct: cfg.c_puct,
                c_visit: cfg.c_visit,
                c_scale: cfg.c_scale,
                seed: cfg.seed,
                force_expand_root_chance: cfg.force_expand_root_chance,
                age_deal_samples: cfg.age_deal_samples,
            },
            leaf_batch,
            rng,
            sign,
            root_value,
            legal,
            log_prior,
            gumbel,
            initial_q,
            q_hat: vec![None; n],
            visits: vec![0; n],
            candidates,
            topk,
            rounds_total,
            round_index: 0,
            per_action,
            candidate_pos: 0,
            repeat_pos: 0,
            sims_launched: 0,
            sims_completed: 0,
            request_seq: 0,
            waiting: None,
            forced_nodes,
            forced_cursor: 0,
            forced_finalized: false,
            metrics: SearchMetrics::default(),
        }
    }

    fn finalize_forced(&mut self) -> PyResult<()> {
        if self.forced_finalized {
            return Ok(());
        }
        if self.forced_cursor != self.forced_nodes.len() || self.waiting.is_some() {
            return Err(PyRuntimeError::new_err(
                "forced root expansion finalized with pending rows",
            ));
        }
        if self
            .forced_nodes
            .iter()
            .any(|&node_id| self.arena.nodes[node_id].cached_evaluation.is_none())
        {
            return Err(PyRuntimeError::new_err(
                "forced root expansion is missing an evaluated child",
            ));
        }
        let root_id = self.arena.root_id;
        self.initial_q = self.arena.nodes[root_id]
            .edges
            .iter()
            .enumerate()
            .map(|(edge, item)| {
                if item.probability_weighted {
                    Some(self.sign * self.arena.edge_q_p0(root_id, edge))
                } else {
                    None
                }
            })
            .collect();
        self.forced_finalized = true;
        Ok(())
    }

    fn set_forced_rows(&mut self, forced: ForcedRows) {
        self.forced_nodes = forced.nodes;
        self.metrics.forced_rows_by_kind = forced.by_kind;
        self.forced_finalized = false;
    }

    fn completed_q(&self, j: usize) -> f64 {
        self.q_hat[j]
            .or(self.initial_q[j])
            .unwrap_or(self.root_value)
    }

    fn finish_round(&mut self) -> PyResult<()> {
        if self.waiting.is_some() || self.sims_launched != self.sims_completed {
            return Err(PyRuntimeError::new_err(
                "halving reduction attempted with pending simulations",
            ));
        }
        if self.arena.incomplete_total() != 0 {
            return Err(PyRuntimeError::new_err(
                "halving reduction attempted with nonzero WU counts",
            ));
        }
        if self.candidates.len() > 1 {
            let max_visits = self.visits.iter().copied().max().unwrap_or(0);
            let cfg = &self.cfg;
            let q_hat = &self.q_hat;
            let initial_q = &self.initial_q;
            let root_value = self.root_value;
            self.candidates.sort_by(|&a, &b| {
                let completed = |j: usize| q_hat[j].or(initial_q[j]).unwrap_or(root_value);
                let ka = self.gumbel[a] + self.log_prior[a] + sigma(cfg, completed(a), max_visits);
                let kb = self.gumbel[b] + self.log_prior[b] + sigma(cfg, completed(b), max_visits);
                kb.partial_cmp(&ka).unwrap()
            });
            self.candidates.truncate((self.candidates.len() / 2).max(1));
        }
        self.round_index += 1;
        self.candidate_pos = 0;
        self.repeat_pos = 0;
        let rounds_remaining = self.rounds_total.saturating_sub(self.round_index).max(1);
        self.per_action = ((self.cfg.sims - self.sims_completed)
            / (rounds_remaining * self.candidates.len()))
        .max(1);
        Ok(())
    }

    fn launch_simulation(&mut self) {
        self.sims_launched += 1;
        self.repeat_pos += 1;
        if self.repeat_pos >= self.per_action {
            self.repeat_pos = 0;
            self.candidate_pos += 1;
        }
    }

    fn complete_simulation(&mut self, root_edge: usize) {
        self.q_hat[root_edge] =
            Some(self.sign * self.arena.edge_q_p0(self.arena.root_id, root_edge));
        self.visits[root_edge] = self.arena.nodes[self.arena.root_id].edges[root_edge].visits;
        self.sims_completed += 1;
        self.metrics.scheduled_simulations += 1;
    }

    fn make_request(&mut self, wave: PendingWave) -> SearchEvent {
        let leaves = wave
            .unique_leaf_ids
            .iter()
            .map(|&leaf_id| {
                let leaf = &self.arena.nodes[leaf_id];
                EvalLeafRequest {
                    leaf_id,
                    actor: leaf.actor,
                    legal: leaf.legal.clone(),
                }
            })
            .collect();
        let request_id = wave.request_id;
        self.metrics.leaf_waves += 1;
        self.metrics.max_wave_paths = self.metrics.max_wave_paths.max(wave.simulations.len());
        self.metrics.max_wave_unique = self.metrics.max_wave_unique.max(wave.unique_leaf_ids.len());
        self.waiting = Some(PendingEvaluation::Wave(wave));
        SearchEvent::Evaluation(EvalBatchRequest {
            request_id,
            leaves,
            forced: false,
        })
    }

    pub fn next_event(&mut self) -> PyResult<SearchEvent> {
        self.next_event_with_limit(usize::MAX)
    }

    /// Advance the search, limiting only a forced-root request. Ordinary leaf
    /// waves remain indivisible because every path in the wave owns WU state.
    pub fn next_event_with_limit(&mut self, forced_row_limit: usize) -> PyResult<SearchEvent> {
        if forced_row_limit == 0 {
            return Err(PyValueError::new_err(
                "forced evaluation row limit must be positive",
            ));
        }
        if self.waiting.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot select while an evaluation request is outstanding",
            ));
        }
        if self.forced_cursor < self.forced_nodes.len() {
            let end = self
                .forced_nodes
                .len()
                .min(self.forced_cursor.saturating_add(forced_row_limit));
            let node_ids = self.forced_nodes[self.forced_cursor..end].to_vec();
            self.forced_cursor = end;
            let leaves = node_ids
                .iter()
                .map(|&leaf_id| {
                    let leaf = &self.arena.nodes[leaf_id];
                    EvalLeafRequest {
                        leaf_id,
                        actor: leaf.actor,
                        legal: leaf.legal.clone(),
                    }
                })
                .collect();
            let request_id = self.request_seq;
            self.request_seq += 1;
            self.waiting = Some(PendingEvaluation::Forced {
                request_id,
                node_ids,
            });
            return Ok(SearchEvent::Evaluation(EvalBatchRequest {
                request_id,
                leaves,
                forced: true,
            }));
        }
        if !self.forced_finalized {
            self.finalize_forced()?;
        }
        let request_id = self.request_seq;
        let mut wave = PendingWave {
            request_id,
            simulations: Vec::new(),
            unique_leaf_ids: Vec::new(),
        };
        loop {
            if self.sims_launched >= self.cfg.sims {
                if !wave.simulations.is_empty() {
                    self.request_seq += 1;
                    return Ok(self.make_request(wave));
                }
                if self.sims_completed != self.cfg.sims || self.arena.incomplete_total() != 0 {
                    return Err(PyRuntimeError::new_err(
                        "search exhausted its launch budget with pending work",
                    ));
                }
                return Ok(SearchEvent::Complete);
            }
            if self.candidate_pos >= self.candidates.len() {
                if !wave.simulations.is_empty() {
                    self.request_seq += 1;
                    return Ok(self.make_request(wave));
                }
                self.finish_round()?;
                continue;
            }
            let root_edge = self.candidates[self.candidate_pos];
            let root_id = self.arena.root_id;
            let (mut pending, immediate) = select_leaf(
                &mut self.arena,
                root_id,
                root_edge,
                &mut self.rng,
                self.cfg.c_puct,
            );
            self.launch_simulation();
            if let Some((value_p0, cached_priors)) = immediate {
                if let Some(priors) = cached_priors {
                    expand(&mut self.arena, pending.leaf_id, priors)?;
                    self.metrics.cached_forced_leaves += 1;
                } else {
                    self.metrics.terminal_leaves += 1;
                }
                backup(&mut self.arena, pending, value_p0);
                self.complete_simulation(root_edge);
                continue;
            }
            if self.leaf_batch > 1 {
                mark_incomplete(&mut self.arena, &mut pending);
            }
            self.metrics.requested_nn_leaves += 1;
            if !wave.unique_leaf_ids.contains(&pending.leaf_id) {
                wave.unique_leaf_ids.push(pending.leaf_id);
                self.metrics.unique_nn_leaves += 1;
            } else {
                self.metrics.collisions += 1;
            }
            wave.simulations.push(pending);
            if wave.simulations.len() >= self.leaf_batch {
                self.request_seq += 1;
                return Ok(self.make_request(wave));
            }
        }
    }

    pub fn evaluation_states<'a>(
        &'a self,
        request: &EvalBatchRequest,
    ) -> PyResult<Vec<&'a GameState>> {
        let Some(pending) = &self.waiting else {
            return Err(PyRuntimeError::new_err(
                "no evaluation request is outstanding",
            ));
        };
        let node_ids: &[NodeId] = match pending {
            PendingEvaluation::Forced {
                request_id,
                node_ids,
            } => {
                if *request_id != request.request_id {
                    return Err(PyValueError::new_err(
                        "evaluation batch request id does not match forced rows",
                    ));
                }
                node_ids
            }
            PendingEvaluation::Wave(wave) => {
                if wave.request_id != request.request_id {
                    return Err(PyValueError::new_err(
                        "evaluation batch request id does not match pending leaves",
                    ));
                }
                &wave.unique_leaf_ids
            }
        };
        if node_ids.len() != request.leaves.len()
            || node_ids
                .iter()
                .zip(&request.leaves)
                .any(|(&leaf_id, request)| leaf_id != request.leaf_id)
        {
            return Err(PyValueError::new_err(
                "evaluation batch does not match pending leaves",
            ));
        }
        Ok(node_ids
            .iter()
            .map(|&leaf_id| &self.arena.nodes[leaf_id].state)
            .collect())
    }

    fn clear_wave(&mut self, wave: &mut PendingWave) {
        for pending in &mut wave.simulations {
            clear_incomplete(&mut self.arena, pending);
        }
    }

    pub fn cancel_pending(&mut self) {
        if let Some(pending) = self.waiting.take() {
            if let PendingEvaluation::Wave(mut wave) = pending {
                self.clear_wave(&mut wave);
            }
        }
    }

    pub fn apply_evaluations(
        &mut self,
        request_id: u64,
        evaluations: Vec<(f64, Vec<f64>)>,
    ) -> PyResult<()> {
        let Some(pending) = self.waiting.take() else {
            return Err(PyRuntimeError::new_err(
                "no evaluation request is outstanding",
            ));
        };
        if let PendingEvaluation::Forced {
            request_id: expected,
            node_ids,
        } = pending
        {
            if request_id != expected || evaluations.len() != node_ids.len() {
                return Err(PyValueError::new_err(format!(
                    "forced evaluation alignment mismatch: request {request_id}, expected {expected}, rows {}, expected {}",
                    evaluations.len(),
                    node_ids.len()
                )));
            }
            for (&node_id, (value_p0, priors)) in node_ids.iter().zip(evaluations) {
                if priors.len() != self.arena.nodes[node_id].legal.len() {
                    return Err(PyValueError::new_err(format!(
                        "forced child {node_id} returned {} priors for {} legal actions",
                        priors.len(),
                        self.arena.nodes[node_id].legal.len()
                    )));
                }
                self.arena.nodes[node_id].visits = 1;
                self.arena.nodes[node_id].value_sum_p0 = value_p0;
                self.arena.nodes[node_id].cached_evaluation = Some((value_p0, priors));
                self.metrics.forced_outcome_rows += 1;
            }
            return Ok(());
        }
        let PendingEvaluation::Wave(mut wave) = pending else {
            unreachable!()
        };
        if request_id != wave.request_id || evaluations.len() != wave.unique_leaf_ids.len() {
            self.clear_wave(&mut wave);
            return Err(PyValueError::new_err(format!(
                "evaluation batch alignment mismatch: request {request_id}, expected {}, rows {}, expected {}",
                wave.request_id,
                evaluations.len(),
                wave.unique_leaf_ids.len()
            )));
        }
        for (&leaf_id, (_, priors)) in wave.unique_leaf_ids.iter().zip(&evaluations) {
            if let Err(err) = expand(&mut self.arena, leaf_id, priors.clone()) {
                self.clear_wave(&mut wave);
                return Err(err);
            }
        }
        for mut pending in wave.simulations {
            let row = wave
                .unique_leaf_ids
                .iter()
                .position(|&leaf_id| leaf_id == pending.leaf_id)
                .expect("pending leaf must have an evaluation row");
            let value_p0 = evaluations[row].0;
            clear_incomplete(&mut self.arena, &mut pending);
            let root_edge = pending.root_edge;
            backup(&mut self.arena, pending, value_p0);
            self.complete_simulation(root_edge);
        }
        Ok(())
    }

    pub fn into_result(mut self) -> PyResult<(SearchResult, Arena, SearchMetrics)> {
        if self.sims_launched != self.cfg.sims
            || self.sims_completed != self.cfg.sims
            || self.waiting.is_some()
            || self.arena.incomplete_total() != 0
        {
            return Err(PyRuntimeError::new_err("search session is not complete"));
        }
        let max_visits = self.visits.iter().copied().max().unwrap_or(0);
        let mut best = self.candidates[0];
        let mut best_score = f64::NEG_INFINITY;
        for &j in &self.candidates {
            let score = self.gumbel[j]
                + self.log_prior[j]
                + sigma(&self.cfg, self.completed_q(j), max_visits);
            if score > best_score {
                best_score = score;
                best = j;
            }
        }
        let logits: Vec<f64> = (0..self.legal.len())
            .map(|j| self.log_prior[j] + sigma(&self.cfg, self.completed_q(j), max_visits))
            .collect();
        let peak = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = logits.iter().map(|&value| (value - peak).exp()).collect();
        let total = weights.iter().fold(0.0_f64, |sum, &value| sum + value);
        let policy_target = weights.iter().map(|&weight| weight / total).collect();
        let root_value = self.sign * self.arena.nodes[self.arena.root_id].value_p0();
        let root_completed_q = (0..self.legal.len()).map(|j| self.completed_q(j)).collect();
        self.metrics.root_completed_q = root_completed_q;
        let result = SearchResult {
            action_index: self.legal[best],
            action_value: self.completed_q(best),
            root_value,
            visits: self.visits,
            policy_target,
            gumbel_topk: self.topk,
            sims: self.sims_completed,
        };
        Ok((result, self.arena, self.metrics))
    }
}

/// F4.4 scheduler boundary: initialize a search from a globally-coalesced root
/// evaluation, then let `SearchSession::next_event` yield its leaf waves. Forced
/// root-child evaluation remains on the scalar path until F4.5's semantic-safe
/// cached expansion is available.
pub fn begin_search_from_root(
    state: &GameState,
    cfg: &SearchConfig,
    leaf_batch: usize,
    root_evaluation: (f64, Vec<f64>),
) -> PyResult<SearchSession> {
    if cfg.sims < 1 || cfg.top_k < 1 || leaf_batch < 1 {
        return Err(PyValueError::new_err(
            "sims, top_k, and leaf_batch must be positive",
        ));
    }
    if cfg.force_expand_root_chance {
        return Err(PyValueError::new_err(
            "cooperative force expansion requires the F4.5 forced-child cache",
        ));
    }
    let root = Node::make(state.clone());
    if root.terminal || root.legal.is_empty() {
        return Err(PyValueError::new_err(
            "cannot search a terminal or action-less root",
        ));
    }
    let mut arena = Arena::new(root);
    let root_id = arena.root_id;
    let (root_value_p0, root_priors) = root_evaluation;
    expand(&mut arena, root_id, root_priors)?;
    arena.nodes[root_id].visits += 1;
    arena.nodes[root_id].value_sum_p0 += root_value_p0;
    Ok(SearchSession::new(arena, cfg, leaf_batch, Vec::new()))
}

/// F4-R1 force-enabled scheduler boundary. Forced children are materialized
/// now, then yielded by `next_event_with_limit` to the global coalescer.
pub fn begin_search_from_root_forced(
    state: &GameState,
    cfg: &SearchConfig,
    leaf_batch: usize,
    root_evaluation: (f64, Vec<f64>),
) -> PyResult<SearchSession> {
    let force = cfg.force_expand_root_chance;
    let mut base_cfg = cfg.clone();
    base_cfg.force_expand_root_chance = false;
    let mut session = begin_search_from_root(state, &base_cfg, leaf_batch, root_evaluation)?;
    session.cfg.force_expand_root_chance = force;
    if force {
        let mut forced = materialize_forced_root(&mut session.arena)?;
        forced.extend(materialize_paired_age_deals(
            &mut session.arena,
            cfg.age_deal_samples,
            cfg.seed,
        )?);
        session.set_forced_rows(forced);
    }
    Ok(session)
}

pub fn search_closed_batched<E: Eval>(
    state: &GameState,
    eval: &E,
    cfg: &SearchConfig,
    leaf_batch: usize,
) -> PyResult<(SearchResult, Arena, SearchMetrics)> {
    if cfg.sims < 1 || cfg.top_k < 1 || leaf_batch < 1 {
        return Err(PyValueError::new_err(
            "sims, top_k, and leaf_batch must be positive",
        ));
    }
    let root = Node::make(state.clone());
    if root.terminal || root.legal.is_empty() {
        return Err(PyValueError::new_err(
            "cannot search a terminal or action-less root",
        ));
    }
    let mut arena = Arena::new(root);
    let root_id = arena.root_id;
    let (root_value_p0, root_priors) = eval.evaluate(&arena.nodes[root_id].state)?;
    expand(&mut arena, root_id, root_priors)?;
    arena.nodes[root_id].visits += 1;
    arena.nodes[root_id].value_sum_p0 += root_value_p0;
    let forced_nodes = if cfg.force_expand_root_chance {
        let mut forced = materialize_forced_root(&mut arena)?;
        forced.extend(materialize_paired_age_deals(
            &mut arena,
            cfg.age_deal_samples,
            cfg.seed,
        )?);
        forced
    } else {
        ForcedRows::default()
    };
    let mut session = SearchSession::new(arena, cfg, leaf_batch, forced_nodes.nodes);
    session.metrics.forced_rows_by_kind = forced_nodes.by_kind;
    loop {
        match session.next_event()? {
            SearchEvent::Complete => break,
            SearchEvent::Evaluation(request) => {
                for leaf in &request.leaves {
                    let state = &session.arena.nodes[leaf.leaf_id].state;
                    debug_assert_eq!(leaf.actor, crate::tree::state_actor(state));
                    debug_assert_eq!(leaf.legal, legal_action_indices(state));
                }
                let evaluations = {
                    let states = session.evaluation_states(&request)?;
                    let actors: Vec<_> = request.leaves.iter().map(|leaf| leaf.actor).collect();
                    let legals: Vec<_> = request
                        .leaves
                        .iter()
                        .map(|leaf| leaf.legal.clone())
                        .collect();
                    eval.evaluate_batch_prepared(&states, &actors, &legals)
                };
                let evaluations = match evaluations {
                    Ok(rows) => rows,
                    Err(err) => {
                        session.cancel_pending();
                        return Err(err);
                    }
                };
                session.apply_evaluations(request.request_id, evaluations)?;
            }
        }
    }
    session.into_result()
}

pub fn search_closed<E: Eval>(
    state: &GameState,
    eval: &E,
    cfg: &SearchConfig,
) -> PyResult<(SearchResult, Arena)> {
    let (result, arena, _) = search_closed_batched(state, eval, cfg, 1)?;
    Ok((result, arena))
}

pub fn digest(arena: &Arena, out: &mut Vec<f64>) {
    fn visit(arena: &Arena, node_id: NodeId, out: &mut Vec<f64>) {
        let node = &arena.nodes[node_id];
        out.push(node.visits as f64);
        out.push(node.value_sum_p0);
        out.push(node.actor as f64);
        out.push(if node.terminal { 1.0 } else { 0.0 });
        let fp = node.state.fingerprint();
        out.push(fp.len() as f64);
        out.extend(fp.iter().map(|&value| value as f64));
        out.push(node.edges.len() as f64);
        for edge in &node.edges {
            out.push(edge.action_index as f64);
            out.push(edge.visits as f64);
            out.push(edge.value_sum_p0);
            out.push(edge.prior);
            out.push(if edge.probability_weighted { 1.0 } else { 0.0 });
            out.push(edge.children.len() as f64);
            for (key, child) in &edge.children {
                out.push(key.len() as f64);
                for part in key {
                    out.push(part.len() as f64);
                    out.extend(part.iter().map(|&value| value as f64));
                }
                out.push(child.samples as f64);
                out.push(child.probability.unwrap_or(f64::NAN));
                visit(arena, child.node_id, out);
            }
        }
    }
    visit(arena, arena.root_id, out);
}
