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

fn state_actor(state: &GameState) -> usize {
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

    fn expand<E: Eval>(&mut self, eval: &E) -> f64 {
        let (value_p0, priors) = eval.evaluate(&self.state);
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
        value_p0
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
) -> f64 {
    if node.terminal {
        let v = terminal_value_p0(&node.state);
        node.visits += 1;
        node.value_sum_p0 += v;
        return v;
    }
    if node.edges.is_empty() {
        let v = node.expand(eval);
        node.visits += 1;
        node.value_sum_p0 += v;
        return v;
    }
    let edge_idx = forced.unwrap_or_else(|| node.select(c_puct));
    let child_idx = closed_child(node, edge_idx, rng);
    let v = {
        let child = &mut *node.edges[edge_idx].children[child_idx].1.node;
        descend(child, None, eval, rng, c_puct)
    };
    node.edges[edge_idx].visits += 1;
    node.edges[edge_idx].value_sum_p0 += v;
    node.visits += 1;
    node.value_sum_p0 += v;
    v
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
) -> Node {
    let root_state = state.clone();
    let mut root = Node::make(root_state);
    let v = root.expand(eval);
    root.visits += 1;
    root.value_sum_p0 += v;
    let mut rng = Rng::new(seed);
    let n_edges = root.edges.len().max(1);
    for i in 0..sims {
        descend(&mut root, Some(i % n_edges), eval, &mut rng, c_puct);
    }
    root
}

/// Canonical depth-first serialization of the tree (visits, values, edge stats,
/// child keys/samples/probabilities) for the equivalence gate.
pub fn digest(node: &Node, out: &mut Vec<f64>) {
    out.push(node.visits as f64);
    out.push(node.value_sum_p0);
    out.push(node.edges.len() as f64);
    for edge in &node.edges {
        out.push(edge.action_index as f64);
        out.push(edge.visits as f64);
        out.push(edge.value_sum_p0);
        out.push(edge.prior);
        out.push(if edge.probability_weighted { 1.0 } else { 0.0 });
        out.push(edge.children.len() as f64);
        for (key, child) in &edge.children {
            out.push(key.iter().map(|k| k.len()).sum::<usize>() as f64);
            for part in key {
                for &k in part {
                    out.push(k as f64);
                }
            }
            out.push(child.samples as f64);
            out.push(child.probability.unwrap_or(f64::NAN));
            digest(&child.node, out);
        }
    }
}
