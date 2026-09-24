//! Exact turn solver — a mirror of `games/cantstop/solver.py::TurnSolver` (M2).
//!
//! Split into two phases so the evaluator never has to be called from Rust:
//!
//! 1. `TurnSolver::new` enumerates every runner configuration the turn can
//!    reach and builds the end-of-turn boards that need a value (the shared
//!    bust board first, then every stoppable, non-winning configuration);
//! 2. `set_leaf_values` takes one value row per leaf board and runs the
//!    backward induction.
//!
//! That split is what M3 needs: many games' leaf batches can be coalesced
//! into one torch forward between the two phases.
//!
//! Bit-identity with Python, not closeness, is the M2 gate, so everything
//! that touches a float follows Python's order exactly: roll classes in
//! `combinations_with_replacement` order, menus grouped in first-occurrence
//! order with probabilities summed in that order, and backup as
//! `roll = bust_p * bust; roll = roll + p * child` element by element (no
//! fused multiply-add). f64 throughout -- Python accumulates in float64.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::{Arc, OnceLock};

use crate::encoder::{encode_board, FEATURE_SIZE};
use crate::engine::{dice_pairings, GameState, Move, Phase, COLUMN_HEIGHTS, MAX_COL, MIN_COL, MAX_RUNNERS};

/// A runner configuration: up to three (column, position) pairs sorted by
/// column. Ordering is Python's tuple-of-tuples ordering, which is exactly
/// slice ordering over the live entries -- tie-breaks depend on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct RKey {
    entries: [(u8, u8); MAX_RUNNERS],
    len: u8,
}

impl RKey {
    pub fn from_runners(runners: &[u8; 13]) -> RKey {
        let mut k = RKey::default();
        for col in MIN_COL..=MAX_COL {
            if runners[col] != 0 {
                k.entries[k.len as usize] = (col as u8, runners[col]);
                k.len += 1;
            }
        }
        k
    }

    pub fn as_slice(&self) -> &[(u8, u8)] {
        &self.entries[..self.len as usize]
    }

    pub fn to_runners(&self) -> [u8; 13] {
        let mut r = [0u8; 13];
        for &(c, p) in self.as_slice() {
            r[c as usize] = p;
        }
        r
    }

    fn height_sum(&self) -> u32 {
        self.as_slice().iter().map(|&(_, p)| p as u32).sum()
    }

    fn tops(&self) -> u8 {
        self.as_slice()
            .iter()
            .filter(|&&(c, p)| p >= COLUMN_HEIGHTS[c as usize])
            .count() as u8
    }
}

impl Ord for RKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl PartialOrd for RKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---- roll classes ----

/// One representative roll per distinct set of sum pairings, with its
/// probability, in the order `solver._roll_classes` produces them.
pub fn roll_classes() -> &'static [([u8; 4], f64)] {
    static CLASSES: OnceLock<Vec<([u8; 4], f64)>> = OnceLock::new();
    CLASSES.get_or_init(|| {
        let mut classes: Vec<(Vec<(u8, u8)>, [u8; 4], u32)> = Vec::new();
        for a in 1..=6u8 {
            for b in a..=6 {
                for c in b..=6 {
                    for d in c..=6 {
                        let dice = [a, b, c, d];
                        let mut key = dice_pairings(dice);
                        key.sort();
                        let w = multiset_weight(dice);
                        match classes.iter_mut().find(|(k, _, _)| *k == key) {
                            Some(entry) => entry.2 += w,
                            None => classes.push((key, dice, w)),
                        }
                    }
                }
            }
        }
        // Python divides the integer weight once: w / 1296.
        classes.into_iter().map(|(_, dice, w)| (dice, w as f64 / 1296.0)).collect()
    })
}

fn multiset_weight(dice: [u8; 4]) -> u32 {
    let mut counts = [0u32; 7];
    for d in dice {
        counts[d as usize] += 1;
    }
    let fact = |n: u32| (1..=n).product::<u32>();
    let mut w = fact(4);
    for &c in &counts {
        w /= fact(c);
    }
    w
}

// ---- roll menus ----

/// `(bust prob, distinct moves, [(move indices, prob)])`, as `_build_menu`.
#[derive(Clone, Debug)]
pub struct Menu {
    pub bust_p: f64,
    pub moves: Vec<Move>,
    pub groups: Vec<(Vec<usize>, f64)>,
}

type Signature = [u8; 11];

/// `solver.column_signature`: the moves a roll offers depend on each column
/// only through this code, so menus are shared by every configuration (and
/// every game) with the same signature.
fn column_signature(state: &GameState, saved: &[u8; 13], runners: &[u8; 13]) -> Signature {
    let mut sig = [0u8; 11];
    for col in MIN_COL..=MAX_COL {
        let in_runners = runners[col] != 0;
        let pos = if in_runners { runners[col] } else { saved[col] };
        let room = COLUMN_HEIGHTS[col] as i32 - pos as i32;
        sig[col - MIN_COL] = if state.claimed_by[col] != -1 {
            0
        } else if room <= 0 {
            if in_runners { 5 } else { 0 }
        } else if in_runners {
            if room == 1 { 1 } else { 2 }
        } else if room == 1 {
            3
        } else {
            4
        };
    }
    sig
}

fn build_menu(state: &GameState) -> Menu {
    let mut bust_p = 0.0f64;
    let mut moves: Vec<Move> = Vec::new();
    let mut groups: Vec<(Vec<usize>, f64)> = Vec::new();
    for &(dice, p) in roll_classes() {
        let legal = state.legal_moves(dice);
        if legal.is_empty() {
            bust_p += p;
            continue;
        }
        let mut idx: Vec<usize> = legal
            .iter()
            .map(|m| match moves.iter().position(|x| x == m) {
                Some(i) => i,
                None => {
                    moves.push(*m);
                    moves.len() - 1
                }
            })
            .collect();
        idx.sort_unstable();
        match groups.iter_mut().find(|(k, _)| *k == idx) {
            Some(g) => g.1 += p,
            None => groups.push((idx, p)),
        }
    }
    Menu { bust_p, moves, groups }
}

/// Menus keyed by column signature. Valid across solves and games (a menu
/// depends on the signature alone), so callers keep one per thread for the
/// life of the process, as Python keeps its module-level cache.
#[derive(Default)]
pub struct MenuCache {
    menus: HashMap<Signature, Arc<Menu>, FxBuild>,
}


// ---- a fast hasher for small fixed-size keys ----
//
// std's SipHash is DoS-resistant and slow; these keys are a few bytes of
// runner positions or column codes. FxHash's multiply-rotate, inlined here
// to keep the crate dependency-free.

#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

impl std::hash::Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

pub type FxBuild = BuildHasherDefault<FxHasher>;

// ---- the solver ----
//
// Storage is flat: node fields in one Vec, every node's children in one
// shared `kids` pool (one entry per move of the node's cached menu), and
// values as `len * num_players` f64 slabs. The first version kept a
// Vec<Vec<usize>> per node for its menu groups, sorted per group -- ~150k
// small allocations per solve, and most of the enumeration time.

const NONE: u32 = u32::MAX;

struct Node {
    key: RKey,
    stoppable: bool,
    winning: bool,
    /// Index into `kids` of this node's first child (one per menu move), or
    /// NONE for a winning node, which is never expanded.
    kids_start: u32,
    menu: Option<Arc<Menu>>,
}

pub struct TurnSolver {
    pub base: GameState,
    pub active: u8,
    pub num_players: usize,
    nodes: Vec<Node>,
    kids: Vec<u32>,
    index: HashMap<RKey, u32, FxBuild>,
    /// Node indices, highest total runner height first (backup order).
    order: Vec<u32>,
    /// Nodes whose stop needs a value, in leaf order after the bust board.
    leaf_nodes: Vec<u32>,
    pub roots: Vec<RKey>,
    // Filled by `set_leaf_values`, `len * num_players` each.
    stop_values: Vec<f64>,
    has_stop: Vec<bool>,
    decision_values: Vec<f64>,
    roll_values: Vec<f64>,
    has_roll: Vec<bool>,
    stops: Vec<bool>,
    solved: bool,
}

#[derive(Debug)]
pub enum SolveError {
    GameOver,
    NotSolved,
    AlreadySolved,
    BadValues(String),
    Unreachable(RKey),
    NoLegalMove,
}

impl TurnSolver {
    /// Enumerate the turn from `state` (any in-turn phase). In AWAIT_MOVE
    /// the table is rooted at the rolled options.
    pub fn new(state: &GameState, cache: &mut MenuCache) -> Result<TurnSolver, SolveError> {
        if state.game_over() {
            return Err(SolveError::GameOver);
        }
        let mut base = state.clone();
        base.dice = None;
        let root = RKey::from_runners(&state.runners);
        let mut s = TurnSolver {
            active: state.active_player,
            num_players: state.rules.num_players as usize,
            base,
            nodes: Vec::new(),
            kids: Vec::new(),
            index: HashMap::default(),
            order: Vec::new(),
            leaf_nodes: Vec::new(),
            roots: Vec::new(),
            stop_values: Vec::new(),
            has_stop: Vec::new(),
            decision_values: Vec::new(),
            roll_values: Vec::new(),
            has_roll: Vec::new(),
            stops: Vec::new(),
            solved: false,
        };
        s.roots = if state.phase == Phase::AwaitMove {
            s.child_keys(root, state.dice.expect("AWAIT_MOVE holds dice"))
                .into_iter()
                .map(|(k, _)| k)
                .collect()
        } else {
            vec![root]
        };
        s.enumerate(cache);
        Ok(s)
    }

    fn saved(&self) -> &[u8; 13] {
        &self.base.progress[self.active as usize]
    }

    #[inline]
    fn step(saved: &[u8; 13], runners: &[u8; 13], mv: &Move) -> RKey {
        let mut child = *runners;
        for &col in mv.as_slice() {
            let c = col as usize;
            child[c] = if child[c] != 0 { child[c] } else { saved[c] } + 1;
        }
        RKey::from_runners(&child)
    }

    fn intern(&mut self, key: RKey, stack: &mut Vec<u32>) -> u32 {
        if let Some(&i) = self.index.get(&key) {
            return i;
        }
        let i = self.nodes.len() as u32;
        self.nodes.push(Node {
            key,
            stoppable: false,
            winning: false,
            kids_start: NONE,
            menu: None,
        });
        self.index.insert(key, i);
        stack.push(i);
        i
    }

    fn enumerate(&mut self, cache: &mut MenuCache) {
        let mut scratch = self.base.clone();
        let saved = *self.saved();
        let already = self.base.claimed_count(self.active);
        let cols_to_win = self.base.rules.columns_to_win;
        let mut stack: Vec<u32> = Vec::new();
        for root in self.roots.clone() {
            self.intern(root, &mut stack);
        }
        while let Some(i) = stack.pop() {
            let key = self.nodes[i as usize].key;
            let runners = key.to_runners();
            scratch.runners = runners;
            let can = key.len > 0 && !scratch.stop_blocked();
            let wins = can && already + key.tops() >= cols_to_win;
            let node = &mut self.nodes[i as usize];
            node.stoppable = can;
            node.winning = wins;
            if wins {
                continue;
            }
            let sig = column_signature(&scratch, &saved, &runners);
            let menu = cache
                .menus
                .entry(sig)
                .or_insert_with(|| Arc::new(build_menu(&scratch)))
                .clone();
            let start = self.kids.len() as u32;
            // Reserve the node's slots first: interning children below may
            // push to `nodes` but never to `kids` out of order.
            self.kids.resize(self.kids.len() + menu.moves.len(), NONE);
            for (j, mv) in menu.moves.iter().enumerate() {
                let ck = Self::step(&saved, &runners, mv);
                let c = self.intern(ck, &mut stack);
                self.kids[start as usize + j] = c;
            }
            let node = &mut self.nodes[i as usize];
            node.kids_start = start;
            node.menu = Some(menu);
        }
        let mut order: Vec<u32> = (0..self.nodes.len() as u32).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(self.nodes[i as usize].key.height_sum()));
        self.leaf_nodes = order
            .iter()
            .copied()
            .filter(|&i| {
                let n = &self.nodes[i as usize];
                n.stoppable && !n.winning
            })
            .collect();
        self.order = order;
    }

    pub fn num_positions(&self) -> usize {
        self.nodes.len()
    }

    /// Number of boards `set_leaf_values` expects: the bust board plus one
    /// per stoppable, non-winning configuration.
    pub fn num_leaves(&self) -> usize {
        1 + self.leaf_nodes.len()
    }

    /// Call `f` on each end-of-turn board to evaluate, bust board first, as
    /// `solver.py`'s `_board_after` builds them, reusing one scratch state.
    fn for_each_leaf_board(&self, mut f: impl FnMut(&GameState)) {
        let mut s = self.base.clone();
        s.runners = [0; 13];
        s.phase = Phase::AwaitDecision;
        s.bust().expect("AWAIT_DECISION can bust");
        f(&s);
        for &i in &self.leaf_nodes {
            s.clone_from(&self.base);
            s.runners = self.nodes[i as usize].key.to_runners();
            s.phase = Phase::AwaitDecision;
            s.stop().expect("leaf configurations are stoppable");
            f(&s);
        }
    }

    pub fn leaf_boards(&self) -> Vec<GameState> {
        let mut out = Vec::with_capacity(self.num_leaves());
        self.for_each_leaf_board(|s| out.push(s.clone()));
        out
    }

    /// The leaf boards encoded for the value net, row-major
    /// `num_leaves x FEATURE_SIZE`, in `leaf_boards` order.
    pub fn leaf_features(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.num_leaves() * FEATURE_SIZE];
        let mut rows = out.chunks_exact_mut(FEATURE_SIZE);
        self.for_each_leaf_board(|s| encode_board(s, rows.next().expect("row per leaf")));
        out
    }

    /// The seat to move on every leaf board (bust and stop alike pass the
    /// turn), which is what rotates the net's seat-relative output back.
    pub fn leaf_active_player(&self) -> u8 {
        (self.active + 1) % self.base.rules.num_players
    }

    /// Backward induction, given `num_leaves * num_players` values, row-major
    /// in `leaf_boards` order. Rows are absolute-seat win probabilities.
    pub fn set_leaf_values(&mut self, values: &[f64]) -> Result<(), SolveError> {
        if self.solved {
            return Err(SolveError::AlreadySolved);
        }
        let n = self.num_players;
        if values.len() != self.num_leaves() * n {
            return Err(SolveError::BadValues(format!(
                "expected {} rows of {} values",
                self.num_leaves(),
                n
            )));
        }
        let len = self.nodes.len();
        let a = self.active as usize;
        self.stop_values = vec![0.0; len * n];
        self.has_stop = vec![false; len];
        self.decision_values = vec![0.0; len * n];
        self.roll_values = vec![0.0; len * n];
        self.has_roll = vec![false; len];
        self.stops = vec![false; len];

        let bust_value = &values[..n];
        for (row, &i) in values[n..].chunks_exact(n).zip(&self.leaf_nodes) {
            let i = i as usize;
            self.stop_values[i * n..(i + 1) * n].copy_from_slice(row);
            self.has_stop[i] = true;
        }
        for i in 0..len {
            if self.nodes[i].winning {
                self.stop_values[i * n + a] = 1.0;
                self.has_stop[i] = true;
            }
        }

        let mut roll = [0.0f64; 4];
        for &i in &self.order {
            let i = i as usize;
            let node = &self.nodes[i];
            if node.winning {
                self.decision_values[i * n + a] = 1.0;
                self.stops[i] = true;
                continue;
            }
            let menu = node.menu.as_ref().expect("expanded node");
            let kids = &self.kids[node.kids_start as usize..];
            for s in 0..n {
                roll[s] = menu.bust_p * bust_value[s];
            }
            for (idx, p) in &menu.groups {
                // Python takes the first maximum over the children sorted by
                // key; the same thing without sorting is "larger value, or
                // equal value and smaller key".
                let mut best = kids[idx[0]] as usize;
                let mut best_v = self.decision_values[best * n + a];
                for &j in &idx[1..] {
                    let k = kids[j] as usize;
                    let v = self.decision_values[k * n + a];
                    if v > best_v || (v == best_v && self.nodes[k].key < self.nodes[best].key) {
                        best = k;
                        best_v = v;
                    }
                }
                let dv = &self.decision_values[best * n..(best + 1) * n];
                for s in 0..n {
                    roll[s] = roll[s] + p * dv[s];
                }
            }
            self.roll_values[i * n..(i + 1) * n].copy_from_slice(&roll[..n]);
            self.has_roll[i] = true;
            let stop_here = node.stoppable && self.stop_values[i * n + a] >= roll[a];
            let src = if stop_here { &self.stop_values } else { &self.roll_values };
            let row: [f64; 4] = {
                let mut r = [0.0; 4];
                r[..n].copy_from_slice(&src[i * n..(i + 1) * n]);
                r
            };
            self.decision_values[i * n..(i + 1) * n].copy_from_slice(&row[..n]);
            self.stops[i] = stop_here;
        }
        self.solved = true;
        Ok(())
    }

    // ---- queries ----

    /// Distinct child configurations a roll leads to from `key`, sorted,
    /// each with the first move (in move order) reaching it.
    pub fn child_keys(&self, key: RKey, dice: [u8; 4]) -> Vec<(RKey, Move)> {
        let mut scratch = self.base.clone();
        let runners = key.to_runners();
        scratch.runners = runners;
        let saved = self.saved();
        let mut out: Vec<(RKey, Move)> = Vec::new();
        let mut d = dice;
        d.sort_unstable();
        for mv in scratch.legal_moves(d) {
            let ck = Self::step(saved, &runners, &mv);
            if !out.iter().any(|(k, _)| *k == ck) {
                out.push((ck, mv));
            }
        }
        out.sort_by(|x, y| x.0.cmp(&y.0));
        out
    }

    fn node(&self, key: RKey) -> Result<usize, SolveError> {
        if !self.solved {
            return Err(SolveError::NotSolved);
        }
        self.index.get(&key).map(|&i| i as usize).ok_or(SolveError::Unreachable(key))
    }

    fn row(slab: &[f64], i: usize, n: usize) -> Vec<f64> {
        slab[i * n..(i + 1) * n].to_vec()
    }

    /// Best child configuration and the move reaching it.
    pub fn best_child(&self, key: RKey, dice: [u8; 4]) -> Result<(RKey, Move), SolveError> {
        let kids = self.child_keys(key, dice);
        if kids.is_empty() {
            return Err(SolveError::NoLegalMove);
        }
        let (a, n) = (self.active as usize, self.num_players);
        let mut best = (kids[0], self.decision_values[self.node(kids[0].0)? * n + a]);
        for &kid in &kids[1..] {
            let v = self.decision_values[self.node(kid.0)? * n + a];
            if v > best.1 {
                best = (kid, v);
            }
        }
        Ok(best.0)
    }

    pub fn choose_move(&self, key: RKey, dice: [u8; 4]) -> Result<Move, SolveError> {
        Ok(self.best_child(key, dice)?.1)
    }

    pub fn should_stop(&self, key: RKey) -> Result<bool, SolveError> {
        let i = self.node(key)?;
        Ok(self.nodes[i].stoppable && self.stops[i])
    }

    /// Per-seat values under best play, as `TurnSolver.value`.
    pub fn value(&self, key: RKey, phase: Phase, dice: Option<[u8; 4]>) -> Result<Vec<f64>, SolveError> {
        let n = self.num_players;
        match phase {
            Phase::AwaitMove => {
                let (child, _) = self.best_child(key, dice.expect("AWAIT_MOVE holds dice"))?;
                Ok(Self::row(&self.decision_values, self.node(child)?, n))
            }
            Phase::AwaitRoll => {
                let i = self.node(key)?;
                if !self.has_roll[i] {
                    return Err(SolveError::Unreachable(key));
                }
                Ok(Self::row(&self.roll_values, i, n))
            }
            _ => Ok(Self::row(&self.decision_values, self.node(key)?, n)),
        }
    }

    /// Every configuration with everything the Python solver records about
    /// it, for the M2 gate: (key, stoppable, winning, bust_p, menu as
    /// (child keys sorted, p), stop value, roll value, decision value, stops).
    #[allow(clippy::type_complexity)]
    pub fn table(
        &self,
    ) -> Vec<(RKey, bool, bool, f64, Vec<(Vec<RKey>, f64)>, Option<Vec<f64>>, Option<Vec<f64>>, Vec<f64>, bool)> {
        let n = self.num_players;
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, nd)| {
                let (bust_p, menu) = match &nd.menu {
                    None => (0.0, Vec::new()),
                    Some(m) => {
                        let kids = &self.kids[nd.kids_start as usize..];
                        let groups = m
                            .groups
                            .iter()
                            .map(|(idx, p)| {
                                let mut ks: Vec<RKey> =
                                    idx.iter().map(|&j| self.nodes[kids[j] as usize].key).collect();
                                ks.sort();
                                (ks, *p)
                            })
                            .collect();
                        (m.bust_p, groups)
                    }
                };
                let solved = self.solved;
                (
                    nd.key,
                    nd.stoppable,
                    nd.winning,
                    bust_p,
                    menu,
                    (solved && self.has_stop[i]).then(|| Self::row(&self.stop_values, i, n)),
                    (solved && self.has_roll[i]).then(|| Self::row(&self.roll_values, i, n)),
                    if solved { Self::row(&self.decision_values, i, n) } else { Vec::new() },
                    solved && self.stops[i],
                )
            })
            .collect()
    }
}
