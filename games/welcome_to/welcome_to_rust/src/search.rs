//! M5 single-search descent, including the optional progressive-widening
//! chance edge from SEARCH_SPEC §7.1a.
//!
//! Rust owns the tree and every game transition. Python is called only for the
//! two evaluator requests: a root-player LEAF or an opponent POLICY. M6 will
//! replace the blocking evaluator with the global coalescer; this descent and
//! its request order stay unchanged.

use std::collections::{HashMap, VecDeque};

use crate::game::{EngineError, EngineResult, Game};
use crate::information_key::{information_key, position_key, InformationKey};
use crate::macro_codec;
use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RequestKind {
    Leaf = 0,
    Policy = 1,
}

pub struct EvalResponse {
    pub priors: Vec<f32>,
    pub value: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub simulations: usize,
    pub c_puct: f64,
    pub alpha: f64,
    pub margin_gain: f64,
    pub confidence_power: f64,
    pub prune_roundabout_pass: bool,
    pub chance_widening: Option<f64>,
    pub chance_widening_alpha: f64,
    pub max_particles: usize,
    pub noise_fresh_fraction: f64,
    pub dirichlet_weight: f64,
    pub temperature: f64,
    pub noise_required: bool,
}

#[derive(Debug)]
struct Outcome {
    /// Fresh transition draws which landed on this viewer information state.
    /// Reusing a retained outcome must never increment this counter.
    count: u64,
    child: Option<usize>,
    terminal_value: Option<f64>,
    /// Concrete hidden states representing this information-state outcome.
    /// None in the no-widening control arm and for terminal or deterministic
    /// outcomes; exact edges preserve the incoming state instead.
    particle_slot: Option<usize>,
    turn_changed: bool,
    ordinal: u32,
}

struct Transition {
    observation: InformationKey,
    turn_changed: bool,
}

#[allow(dead_code)]
struct OpenLoopOutcomeLayout {
    child: Option<usize>,
    ordinal: u32,
    turn_changed: bool,
}

/// Current-target layout measurement for §8.1's widening hedge. The HashMap
/// and observation key exist in both layouts and are deliberately excluded.
pub fn outcome_layout_bytes() -> (usize, usize, usize) {
    (
        std::mem::size_of::<Outcome>(),
        std::mem::size_of::<OpenLoopOutcomeLayout>(),
        std::mem::size_of::<Vec<Vec<Game>>>(),
    )
}

#[derive(Debug, Default)]
struct ActionEdge {
    outcomes: HashMap<InformationKey, Outcome>,
    next_ordinal: u32,
    /// All traversals of this edge. This drives K(n), independently of the
    /// fresh-draw counts above.
    visits: u64,
    /// A within-turn transition consumed no randomness, so support is exactly
    /// one and this edge may stay closed even after K(n) grows past one.
    exact: bool,
}

#[derive(Debug)]
struct Node {
    actions: Vec<usize>,
    prior: Vec<f64>,
    visits: Vec<f64>,
    total: Vec<f64>,
    edges: Vec<ActionEdge>,
    noised: bool,
}

impl Node {
    fn new(actions: Vec<usize>, prior: Vec<f64>) -> Self {
        let width = actions.len();
        Self {
            actions,
            prior,
            visits: vec![0.0; width],
            total: vec![0.0; width],
            edges: (0..width).map(|_| ActionEdge::default()).collect(),
            noised: false,
        }
    }

    fn select(&self, c_puct: f64) -> usize {
        let total_visits: f64 = self.visits.iter().sum();
        let scale = total_visits.max(1.0).sqrt();
        let mut best = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for index in 0..self.actions.len() {
            let q = if self.visits[index] > 0.0 {
                self.total[index] / self.visits[index]
            } else {
                0.0
            };
            let exploration = c_puct * self.prior[index] * scale / (1.0 + self.visits[index]);
            let score = q + exploration;
            if score > best_score {
                best = index;
                best_score = score;
            }
        }
        best
    }
}

#[derive(Debug)]
struct Retained {
    node: usize,
    root: usize,
    key: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct SearchOutput {
    pub actions: Vec<usize>,
    pub visits: Vec<f64>,
    pub total: Vec<f64>,
    pub root_node: usize,
    pub rng_state: u64,
}

#[derive(Clone, Debug)]
pub struct DebugOutcome {
    pub action: usize,
    pub ordinal: u32,
    pub key: Vec<u8>,
    pub child: Option<usize>,
    pub terminal_value: Option<f64>,
    pub count: u64,
    pub particle_slot: Option<usize>,
    pub particle_count: usize,
    pub turn_changed: bool,
}

#[derive(Clone, Debug)]
pub struct DebugNode {
    pub id: usize,
    pub actions: Vec<usize>,
    pub prior: Vec<f64>,
    pub visits: Vec<f64>,
    pub total: Vec<f64>,
    pub noised: bool,
    pub edge_visits: Vec<u64>,
    pub edge_exact: Vec<bool>,
    pub outcomes: Vec<DebugOutcome>,
}

pub struct Search {
    pub config: SearchConfig,
    arena: Vec<Node>,
    /// Concrete states retained only when progressive widening is enabled.
    /// The no-widening control arm allocates no entries.
    particle_arena: Vec<Vec<Game>>,
    retained: Option<Retained>,
    next_request_id: u32,
    pub simulations_run: usize,
    pub simulations_reused: usize,
    pub reroots: usize,
    pub terminal_leaves: usize,
}

impl Search {
    pub fn new(config: SearchConfig) -> EngineResult<Self> {
        if config.simulations == 0 {
            return Err(EngineError::Invalid("simulations must be positive".into()));
        }
        if !config.c_puct.is_finite() || config.c_puct < 0.0 {
            return Err(EngineError::Invalid(
                "c_puct must be finite and non-negative".into(),
            ));
        }
        if !config.temperature.is_finite() || config.temperature < 0.0 {
            return Err(EngineError::Invalid(
                "temperature must be finite and non-negative".into(),
            ));
        }
        if !config.alpha.is_finite() || !(0.0..=1.0).contains(&config.alpha) {
            return Err(EngineError::Invalid("alpha must be in [0, 1]".into()));
        }
        if !config.margin_gain.is_finite()
            || !config.confidence_power.is_finite()
            || config.confidence_power < 0.0
        {
            return Err(EngineError::Invalid(
                "value-blend parameters must be finite and confidence_power non-negative".into(),
            ));
        }
        if !config.noise_fresh_fraction.is_finite()
            || !config.dirichlet_weight.is_finite()
            || !(0.0..=1.0).contains(&config.dirichlet_weight)
        {
            return Err(EngineError::Invalid(
                "noise_fresh_fraction must be finite and dirichlet_weight in [0, 1]".into(),
            ));
        }
        if let Some(c) = config.chance_widening {
            if !c.is_finite() || c <= 0.0 {
                return Err(EngineError::Invalid(
                    "chance_widening must be finite and positive".into(),
                ));
            }
            if !config.chance_widening_alpha.is_finite()
                || !(0.0..=1.0).contains(&config.chance_widening_alpha)
            {
                return Err(EngineError::Invalid(
                    "chance_widening_alpha must be in [0, 1]".into(),
                ));
            }
            if config.max_particles == 0 {
                return Err(EngineError::Invalid(
                    "max_particles must be positive when chance widening is enabled".into(),
                ));
            }
        }
        Ok(Self {
            config,
            arena: Vec::new(),
            particle_arena: Vec::new(),
            retained: None,
            next_request_id: 0,
            simulations_run: 0,
            simulations_reused: 0,
            reroots: 0,
            terminal_leaves: 0,
        })
    }

    pub fn reset(&mut self) {
        self.arena.clear();
        self.particle_arena.clear();
        self.retained = None;
    }

    pub fn particle_slots_allocated(&self) -> usize {
        self.particle_arena.len()
    }

    pub fn particle_states_allocated(&self) -> usize {
        self.particle_arena.iter().map(Vec::len).sum()
    }

    fn search_actions(&self, state: &Game) -> EngineResult<Vec<usize>> {
        macro_codec::search_legal_macros(state, self.config.prune_roundabout_pass)
    }

    fn request<E>(
        &mut self,
        kind: RequestKind,
        state: &Game,
        viewer: usize,
        evaluator: &mut E,
    ) -> EngineResult<EvalResponse>
    where
        E: FnMut(RequestKind, &Game, usize, u32) -> EngineResult<EvalResponse>,
    {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1);
        let response = evaluator(kind, state, viewer, request_id)?;
        if response.priors.len() != macro_codec::NUM_MACRO_ACTIONS {
            return Err(EngineError::Invalid(format!(
                "evaluator returned {} priors, expected {}",
                response.priors.len(),
                macro_codec::NUM_MACRO_ACTIONS
            )));
        }
        if response.priors.iter().any(|p| !p.is_finite() || *p < 0.0) {
            return Err(EngineError::Invalid(
                "evaluator priors must be finite and non-negative".into(),
            ));
        }
        let mass: f64 = response.priors.iter().map(|&p| f64::from(p)).sum();
        if !mass.is_finite() || mass <= 0.0 {
            return Err(EngineError::Invalid(
                "evaluator priors need positive finite mass".into(),
            ));
        }
        if kind == RequestKind::Leaf && response.value.is_none() {
            return Err(EngineError::Invalid(
                "LEAF response omitted its value".into(),
            ));
        }
        if response.value.is_some_and(|value| !value.is_finite()) {
            return Err(EngineError::Invalid(
                "evaluator value must be finite".into(),
            ));
        }
        Ok(response)
    }

    fn expand_leaf<E>(
        &mut self,
        state: &Game,
        root: usize,
        evaluator: &mut E,
    ) -> EngineResult<(Option<usize>, f64)>
    where
        E: FnMut(RequestKind, &Game, usize, u32) -> EngineResult<EvalResponse>,
    {
        if state.is_terminal() {
            self.terminal_leaves += 1;
            return Ok((None, terminal_value(state, root, &self.config)));
        }
        let response = self.request(RequestKind::Leaf, state, root, evaluator)?;
        let actions = self.search_actions(state)?;
        if actions.is_empty() {
            return Err(EngineError::Invalid(
                "a live search leaf has no actions".into(),
            ));
        }
        // Mirror NumPy's f32 gather / sum / division, then widen to f64.
        let mut gathered: Vec<f32> = actions.iter().map(|&a| response.priors[a]).collect();
        let total: f32 = gathered.iter().sum();
        if total > 0.0 {
            for value in &mut gathered {
                *value /= total;
            }
        } else {
            gathered.fill(1.0 / actions.len() as f32);
        }
        let prior = gathered.into_iter().map(f64::from).collect();
        let id = self.arena.len();
        self.arena.push(Node::new(actions, prior));
        Ok((Some(id), response.value.expect("checked")))
    }

    fn advance<E>(
        &mut self,
        state: &mut Game,
        root: usize,
        rng: &mut Rng,
        origin_turn: i32,
        evaluator: &mut E,
    ) -> EngineResult<Transition>
    where
        E: FnMut(RequestKind, &Game, usize, u32) -> EngineResult<EvalResponse>,
    {
        let mut guard = 0usize;
        while !state.is_terminal() && state.actor != root {
            let seat = state.actor;
            let response = self.request(RequestKind::Policy, state, seat, evaluator)?;
            let weights: Vec<f64> = response.priors.iter().map(|&p| f64::from(p)).collect();
            let action = rng.weighted_index(&weights);
            macro_codec::apply_macro(state, action)?;
            guard += 1;
            if guard > 5000 {
                return Err(EngineError::Invalid(
                    "opponents did not yield the turn".into(),
                ));
            }
        }
        Ok(Transition {
            observation: information_key(state, root)?,
            turn_changed: state.turn != origin_turn,
        })
    }

    fn collapse_forced<E>(
        &mut self,
        state: &mut Game,
        root: usize,
        rng: &mut Rng,
        turn: i32,
        mut transition: Transition,
        evaluator: &mut E,
    ) -> EngineResult<Transition>
    where
        E: FnMut(RequestKind, &Game, usize, u32) -> EngineResult<EvalResponse>,
    {
        while state.turn == turn
            && !state.is_terminal()
            && state.actor == root
            && macro_codec::is_macro_root(state)
        {
            let forced = self.search_actions(state)?;
            if forced.len() != 1 {
                break;
            }
            macro_codec::apply_macro(state, forced[0])?;
            transition = self.advance(state, root, rng, turn, evaluator)?;
        }
        Ok(transition)
    }

    /// Whether this action edge has reached its current progressive-widening
    /// allowance. A closed edge reuses a retained outcome instead of sampling
    /// another root-to-root transition.
    fn edge_is_closed(&self, node_id: usize, index: usize) -> bool {
        let Some(c) = self.config.chance_widening else {
            return false;
        };
        let edge = &self.arena[node_id].edges[index];
        if edge.outcomes.is_empty() {
            return false;
        }
        if edge.exact {
            return true;
        }
        let allowed = (c * (edge.visits as f64).powf(self.config.chance_widening_alpha))
            .ceil()
            .max(1.0) as usize;
        edge.outcomes.len() >= allowed
    }

    /// Reuse one retained outcome in empirical fresh-draw proportion. Outcome
    /// ordinal, rather than HashMap iteration order, preserves Python's
    /// insertion-ordered `dict` tape exactly.
    fn reuse_outcome(
        &mut self,
        node_id: usize,
        index: usize,
        rng: &mut Rng,
    ) -> EngineResult<(Option<Game>, Option<usize>, Option<f64>)> {
        let (particle_slot, child, terminal_value) = {
            let edge = &mut self.arena[node_id].edges[index];
            let mut outcomes: Vec<&Outcome> = edge.outcomes.values().collect();
            outcomes.sort_by_key(|outcome| outcome.ordinal);
            let weights: Vec<f64> = outcomes.iter().map(|outcome| outcome.count as f64).collect();
            let chosen = outcomes[rng.weighted_index(&weights)];
            let result = (chosen.particle_slot, chosen.child, chosen.terminal_value);
            edge.visits = edge
                .visits
                .checked_add(1)
                .ok_or_else(|| EngineError::Invalid("chance edge visit counter overflow".into()))?;
            result
        };
        if let Some(value) = terminal_value {
            return Ok((None, None, Some(value)));
        }
        let slot = particle_slot.ok_or_else(|| {
            EngineError::Invalid("live widened outcome has no particle collection".into())
        })?;
        let particles = self.particle_arena.get(slot).ok_or_else(|| {
            EngineError::Invalid("widened outcome points outside the particle arena".into())
        })?;
        if particles.is_empty() {
            return Err(EngineError::Invalid(
                "live widened outcome has an empty particle collection".into(),
            ));
        }
        let particle = particles[rng.randrange(particles.len() as u64) as usize].clone();
        let child = child.ok_or_else(|| {
            EngineError::Invalid("live widened outcome has no decision child".into())
        })?;
        Ok((Some(particle), Some(child), None))
    }

    /// Merge one freshly sampled transition into an information-state outcome
    /// and reservoir-sample its concrete hidden state.
    fn record_outcome(
        &mut self,
        node_id: usize,
        index: usize,
        observation: &InformationKey,
        state: &Game,
        rng: &mut Rng,
        exact: bool,
        turn_changed: bool,
    ) -> EngineResult<()> {
        if self.config.chance_widening.is_none() {
            return Ok(());
        }

        let is_terminal = state.is_terminal();
        let needs_particle_slot = !is_terminal
            && !exact
            && self.arena[node_id].edges[index]
                .outcomes
                .get(observation)
                .and_then(|outcome| outcome.particle_slot)
                .is_none();
        let new_particle_slot = if needs_particle_slot {
            let slot = self.particle_arena.len();
            self.particle_arena.push(Vec::new());
            Some(slot)
        } else {
            None
        };

        let (particle_slot, count) = {
            let edge = &mut self.arena[node_id].edges[index];
            edge.visits = edge
                .visits
                .checked_add(1)
                .ok_or_else(|| EngineError::Invalid("chance edge visit counter overflow".into()))?;
            if exact && !is_terminal {
                edge.exact = true;
            }
            if !edge.outcomes.contains_key(observation) {
                let ordinal = edge.next_ordinal;
                edge.next_ordinal = edge.next_ordinal.checked_add(1).ok_or_else(|| {
                    EngineError::Invalid("chance outcome ordinal overflow".into())
                })?;
                edge.outcomes.insert(
                    observation.clone(),
                    Outcome {
                        count: 0,
                        child: None,
                        terminal_value: None,
                        particle_slot: new_particle_slot,
                        turn_changed,
                        ordinal,
                    },
                );
            }
            let outcome = edge
                .outcomes
                .get_mut(observation)
                .expect("inserted above");
            if !is_terminal && !exact && outcome.particle_slot.is_none() {
                outcome.particle_slot = new_particle_slot;
            }
            if outcome.turn_changed != turn_changed {
                return Err(EngineError::Invalid(
                    "one information-state outcome disagrees on whether the turn changed".into(),
                ));
            }
            outcome.count = outcome.count.checked_add(1).ok_or_else(|| {
                EngineError::Invalid("chance outcome fresh-draw counter overflow".into())
            })?;
            (outcome.particle_slot, outcome.count)
        };

        if !is_terminal && !exact {
            let slot = particle_slot.ok_or_else(|| {
                EngineError::Invalid("live widened outcome has no particle slot".into())
            })?;
            let particles = &mut self.particle_arena[slot];
            if particles.len() < self.config.max_particles {
                particles.push(state.clone());
            } else {
                // The k-th fresh sample replaces a uniform retained slot with
                // probability max_particles/k. This keeps the conditional
                // belief live rather than freezing the first few particles.
                let replacement = rng.randrange(count);
                if replacement < self.config.max_particles as u64 {
                    particles[replacement as usize] = state.clone();
                }
            }
        }
        Ok(())
    }

    fn simulate<E>(
        &mut self,
        mut state: Game,
        mut node_id: usize,
        root: usize,
        rng: &mut Rng,
        evaluator: &mut E,
    ) -> EngineResult<f64>
    where
        E: FnMut(RequestKind, &Game, usize, u32) -> EngineResult<EvalResponse>,
    {
        let mut path: Vec<(usize, usize)> = Vec::new();
        let value = loop {
            let index = self.arena[node_id].select(self.config.c_puct);
            let action = self.arena[node_id].actions[index];
            path.push((node_id, index));

            if self.arena[node_id].edges[index].exact {
                // The visible outcome is unique, but the incoming hidden deck
                // is a fresh root determinization. Reusing the first particle
                // would bind this action branch to that deck even though this
                // transition consumed no randomness. Replay the deterministic
                // macros on the current state and reuse only the child node.
                let turn = state.turn;
                macro_codec::apply_macro(&mut state, action)?;
                let transition = self.advance(&mut state, root, rng, turn, evaluator)?;
                let transition =
                    self.collapse_forced(&mut state, root, rng, turn, transition, evaluator)?;
                if state.turn != turn || state.is_terminal() || transition.turn_changed {
                    return Err(EngineError::Invalid(
                        "an exact chance edge consumed randomness".into(),
                    ));
                }
                let child = self.arena[node_id].edges[index]
                    .outcomes
                    .get(&transition.observation)
                    .and_then(|outcome| outcome.child)
                    .ok_or_else(|| {
                        EngineError::Invalid(
                            "an exact chance edge has no matching decision child".into(),
                        )
                    })?;
                self.arena[node_id].edges[index].visits = self.arena[node_id].edges[index]
                    .visits
                    .checked_add(1)
                    .ok_or_else(|| {
                        EngineError::Invalid("chance edge visit counter overflow".into())
                    })?;
                node_id = child;
                continue;
            }

            if self.edge_is_closed(node_id, index) {
                let (particle, child, terminal_value) =
                    self.reuse_outcome(node_id, index, rng)?;
                if let Some(value) = terminal_value {
                    break value;
                }
                state = particle.expect("live reuse checked by reuse_outcome");
                node_id = child.expect("live reuse checked by reuse_outcome");
                continue;
            }

            let turn = state.turn;
            macro_codec::apply_macro(&mut state, action)?;
            let transition = self.advance(&mut state, root, rng, turn, evaluator)?;
            let transition =
                self.collapse_forced(&mut state, root, rng, turn, transition, evaluator)?;
            let observation = transition.observation;

            self.record_outcome(
                node_id,
                index,
                &observation,
                &state,
                rng,
                state.turn == turn && !state.is_terminal(),
                transition.turn_changed,
            )?;

            let existing = self.arena[node_id].edges[index]
                .outcomes
                .get(&observation)
                .and_then(|outcome| outcome.child);
            if let Some(child) = existing {
                node_id = child;
                continue;
            }

            let (child, leaf_value) = self.expand_leaf(&state, root, evaluator)?;
            if self.config.chance_widening.is_some() {
                let edge = &mut self.arena[node_id].edges[index];
                let outcome = edge
                    .outcomes
                    .get_mut(&observation)
                    .expect("record_outcome created widening outcome");
                outcome.child = child;
                if child.is_none() {
                    outcome.terminal_value = Some(leaf_value);
                }
            } else if let Some(child) = child {
                // The control arm stores live observation-keyed children but no
                // counters, particles, or terminal cache.
                let edge = &mut self.arena[node_id].edges[index];
                let ordinal = edge.next_ordinal;
                edge.next_ordinal = edge.next_ordinal.checked_add(1).ok_or_else(|| {
                    EngineError::Invalid("chance outcome ordinal overflow".into())
                })?;
                edge.outcomes.insert(
                    observation,
                    Outcome {
                        count: 0,
                        child: Some(child),
                        terminal_value: None,
                        particle_slot: None,
                        turn_changed: transition.turn_changed,
                        ordinal,
                    },
                );
            }
            break leaf_value;
        };
        for (parent, index) in path {
            self.arena[parent].visits[index] += 1.0;
            self.arena[parent].total[index] += value;
        }
        Ok(value)
    }

    fn apply_root_noise(&mut self, node: usize, noise: Option<&[f64]>) -> EngineResult<bool> {
        if self.arena[node].noised || self.arena[node].actions.len() < 2 {
            return Ok(false);
        }
        if self.config.noise_required && noise.is_none() {
            return Err(EngineError::Invalid(
                "root noise is configured but no Python-generated vector was supplied".into(),
            ));
        }
        let Some(noise) = noise else {
            return Ok(false);
        };
        if noise.len() != self.arena[node].actions.len()
            || noise.iter().any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(EngineError::Invalid(
                "root noise must be finite, non-negative and match root width".into(),
            ));
        }
        let weight = self.config.dirichlet_weight;
        for (prior, &sample) in self.arena[node].prior.iter_mut().zip(noise) {
            *prior = (1.0 - weight) * *prior + weight * sample;
        }
        self.arena[node].noised = true;
        Ok(true)
    }

    fn run_from_node<E>(
        &mut self,
        state: &Game,
        root: usize,
        node: usize,
        rng: &mut Rng,
        evaluator: &mut E,
        noise: Option<&[f64]>,
    ) -> EngineResult<SearchOutput>
    where
        E: FnMut(RequestKind, &Game, usize, u32) -> EngineResult<EvalResponse>,
    {
        let noised = self.apply_root_noise(node, noise)?;
        let reused = self.arena[node].visits.iter().sum::<f64>() as usize;
        let mut budget = self.config.simulations.saturating_sub(reused);
        if noised {
            let fraction = self.config.noise_fresh_fraction.clamp(0.0, 1.0);
            budget = budget.max((self.config.simulations as f64 * fraction).ceil() as usize);
        }
        self.simulations_reused += reused;
        self.simulations_run += budget;
        for _ in 0..budget {
            let determinized = state.redeterminize(rng);
            self.simulate(determinized, node, root, rng, evaluator)?;
        }
        Ok(SearchOutput {
            actions: self.arena[node].actions.clone(),
            visits: self.arena[node].visits.clone(),
            total: self.arena[node].total.clone(),
            root_node: node,
            rng_state: rng.state(),
        })
    }

    pub fn search<E>(
        &mut self,
        state: &Game,
        root: usize,
        seed: u64,
        evaluator: &mut E,
        noise: Option<&[f64]>,
    ) -> EngineResult<SearchOutput>
    where
        E: FnMut(RequestKind, &Game, usize, u32) -> EngineResult<EvalResponse>,
    {
        if state.actor != root {
            return Err(EngineError::Invalid(
                "search starts at a state the root player is to act in".into(),
            ));
        }
        self.reset();
        let (node, _) = self.expand_leaf(state, root, evaluator)?;
        let node =
            node.ok_or_else(|| EngineError::Invalid("cannot search a finished game".into()))?;
        self.run_from_node(state, root, node, &mut Rng::new(seed), evaluator, noise)
    }

    fn take_retained(&mut self, state: &Game, root: usize) -> EngineResult<Option<usize>> {
        let retained = self.retained.take();
        let Some(retained) = retained else {
            self.arena.clear();
            self.particle_arena.clear();
            return Ok(None);
        };
        if retained.root != root || retained.key != position_key(state, root)? {
            self.arena.clear();
            self.particle_arena.clear();
            return Ok(None);
        }
        self.reroots += 1;
        Ok(Some(retained.node))
    }

    fn within_turn_successor(
        &self,
        state: &Game,
        root: usize,
        choice: usize,
    ) -> EngineResult<Option<Game>> {
        let turn = state.turn;
        let mut next = macro_codec::step_macro(state, choice)?;
        while next.turn == turn
            && !next.is_terminal()
            && next.actor == root
            && macro_codec::is_macro_root(&next)
        {
            let forced = self.search_actions(&next)?;
            if forced.len() != 1 {
                break;
            }
            macro_codec::apply_macro(&mut next, forced[0])?;
        }
        if next.is_terminal()
            || next.turn != turn
            || next.actor != root
            || !macro_codec::is_macro_root(&next)
        {
            return Ok(None);
        }
        Ok(Some(next))
    }

    fn retain(
        &mut self,
        state: &Game,
        root: usize,
        node: usize,
        choice: usize,
    ) -> EngineResult<()> {
        let Some(successor) = self.within_turn_successor(state, root, choice)? else {
            self.retained = None;
            return Ok(());
        };
        let observation = information_key(&successor, root)?;
        let Some(index) = self.arena[node].actions.iter().position(|&a| a == choice) else {
            self.retained = None;
            return Ok(());
        };
        let child = self.arena[node].edges[index]
            .outcomes
            .get(&observation)
            .and_then(|outcome| outcome.child);
        self.retained = child.map(|child| Retained {
            node: child,
            root,
            key: position_key(&successor, root).expect("validated successor"),
        });
        Ok(())
    }

    pub fn play<E>(
        &mut self,
        state: &Game,
        root: usize,
        seed: u64,
        evaluator: &mut E,
        noise: Option<&[f64]>,
    ) -> EngineResult<(usize, SearchOutput)>
    where
        E: FnMut(RequestKind, &Game, usize, u32) -> EngineResult<EvalResponse>,
    {
        if state.actor != root {
            return Err(EngineError::Invalid("play root is not the actor".into()));
        }
        let legal = self.search_actions(state)?;
        if legal.len() == 1 {
            return Ok((
                legal[0],
                SearchOutput {
                    actions: legal,
                    visits: vec![0.0],
                    total: vec![0.0],
                    root_node: usize::MAX,
                    rng_state: seed,
                },
            ));
        }
        let mut rng = Rng::new(seed);
        let node = match self.take_retained(state, root)? {
            Some(node) => node,
            None => self
                .expand_leaf(state, root, evaluator)?
                .0
                .ok_or_else(|| EngineError::Invalid("cannot play a finished game".into()))?,
        };
        let output = self.run_from_node(state, root, node, &mut rng, evaluator, noise)?;
        let choice_index = if self.config.temperature <= 0.0 {
            output
                .visits
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    a.1.partial_cmp(b.1)
                        .expect("finite visits")
                        .then_with(|| b.0.cmp(&a.0))
                })
                .map(|(index, _)| index)
                .expect("non-empty root")
        } else {
            let weights: Vec<f64> = output
                .visits
                .iter()
                .map(|visits| visits.powf(1.0 / self.config.temperature))
                .collect();
            rng.weighted_index(&weights)
        };
        let choice = output.actions[choice_index];
        self.retain(state, root, node, choice)?;
        let mut output = output;
        output.rng_state = rng.state();
        Ok((choice, output))
    }

    pub fn debug_tree(&self, root: usize) -> Vec<DebugNode> {
        if root >= self.arena.len() {
            return Vec::new();
        }
        let mut queue = VecDeque::from([root]);
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        while let Some(id) = queue.pop_front() {
            if !seen.insert(id) {
                continue;
            }
            let node = &self.arena[id];
            let mut outcomes = Vec::new();
            for (index, edge) in node.edges.iter().enumerate() {
                for (key, outcome) in &edge.outcomes {
                    if let Some(child) = outcome.child {
                        queue.push_back(child);
                    }
                    outcomes.push(DebugOutcome {
                        action: node.actions[index],
                        ordinal: outcome.ordinal,
                        key: key.clone(),
                        child: outcome.child,
                        terminal_value: outcome.terminal_value,
                        count: outcome.count,
                        particle_slot: outcome.particle_slot,
                        particle_count: outcome
                            .particle_slot
                            .and_then(|slot| self.particle_arena.get(slot))
                            .map_or(0, Vec::len),
                        turn_changed: outcome.turn_changed,
                    });
                }
            }
            outcomes.sort_by_key(|outcome| (outcome.action, outcome.ordinal));
            out.push(DebugNode {
                id,
                actions: node.actions.clone(),
                prior: node.prior.clone(),
                visits: node.visits.clone(),
                total: node.total.clone(),
                noised: node.noised,
                edge_visits: node.edges.iter().map(|edge| edge.visits).collect(),
                edge_exact: node.edges.iter().map(|edge| edge.exact).collect(),
                outcomes,
            });
        }
        out.sort_by_key(|node| node.id);
        out
    }
}

fn confidence_floor(num_seats: usize) -> f64 {
    if num_seats < 2 {
        0.0
    } else {
        1.0 - (num_seats + 1) as f64 / (3.0 * (num_seats - 1) as f64)
    }
}

fn blend_value(rank_probs: &[f64], scores: &[f64], seats: usize, config: &SearchConfig) -> f64 {
    let mut mean = 0.0;
    let mut second = 0.0;
    for (rank, &probability) in rank_probs.iter().take(seats).enumerate() {
        let utility = if seats < 2 {
            1.0
        } else {
            (seats - 1 - rank) as f64 / (seats - 1) as f64
        };
        mean += probability * utility;
        second += probability * utility * utility;
    }
    let variance = second - mean * mean;
    let rank_value = 2.0 * mean - 1.0;
    let margin = if seats < 2 {
        0.0
    } else {
        ((scores[0]
            - scores[1..seats]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max))
            * config.margin_gain)
            .tanh()
    };
    let spread = 1.0 - 4.0 * variance;
    let floor = confidence_floor(seats);
    let confidence = ((spread - floor) / (1.0 - floor))
        .max(0.0)
        .powf(config.confidence_power);
    (1.0 - config.alpha) * rank_value + config.alpha * confidence * margin
}

fn terminal_value(state: &Game, root: usize, config: &SearchConfig) -> f64 {
    let seats = state.config.players.min(4);
    let scores = state.scores(None);
    let keys: Vec<(i32, [i32; 7])> = (0..state.config.players)
        .map(|seat| (scores[seat], state.sheets[seat].tiebreak_key()))
        .collect();
    let mut order: Vec<usize> = (0..state.config.players).collect();
    order.sort_by(|&left, &right| keys[right].cmp(&keys[left]).then_with(|| left.cmp(&right)));
    let mut distribution = vec![0.0f64; 4];
    let mut lo = 0usize;
    while lo < order.len() {
        let mut hi = lo + 1;
        while hi < order.len() && keys[order[hi]] == keys[order[lo]] {
            hi += 1;
        }
        if order[lo..hi].contains(&root) {
            let share = 1.0 / (hi - lo) as f64;
            distribution[lo..hi].fill(share);
            break;
        }
        lo = hi;
    }
    let seat_order: Vec<usize> = (0..state.config.players)
        .map(|offset| (root + offset) % state.config.players)
        .take(4)
        .collect();
    let ordered_scores: Vec<f64> = seat_order
        .iter()
        .map(|&seat| scores[seat] as f64 / 80.0)
        .collect();
    blend_value(&distribution, &ordered_scores, seats, config)
}
