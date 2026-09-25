//! Self-play and arena games in Rust, many at once (M3).
//!
//! A `Game` mirrors `self_play.play_game` / `arena.play_match_game` turn for
//! turn: roll; bust on the opening roll without solving; otherwise solve
//! once, then play the whole turn from that one table, rolling again until
//! the solver stops or a roll busts. Dice come from the game's own portable
//! RNG, in the order `play_turn` draws them.
//!
//! Each turn needs the value net exactly once, between enumeration and
//! backup. A game therefore runs until it `NeedsValues` (or is done), the
//! `Pool` gathers every waiting game's leaf features into one buffer for one
//! forward, and `resume` hands each game its rows back. Games advance in
//! parallel on a rayon pool; a single turn solve never is.
//!
//! Games are independent -- own state, own RNG, and the menu cache only
//! caches -- so results do not depend on thread count or on how many games
//! are in flight. The M3 gate checks exactly that.

use rayon::prelude::*;

use crate::encoder::{encode_board, FEATURE_SIZE};
use crate::engine::{GameState, Phase};
use crate::rng::{roll_dice, Rng};
use crate::solver::{RKey, SolveError, TurnSolver};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// Scheduled, waiting for an in-flight slot.
    Queued,
    /// Not yet advanced, or values just supplied.
    Ready,
    /// A turn solve (or its lookahead refinements) is waiting for values.
    NeedsValues,
    Done,
    /// Ran past `max_turns` with no winner -- a defect, as in Python.
    Failed(String),
}

/// How a seat searches its turns (`self_play.Search`). Per seat, like the
/// evaluator, so an arena game can pit depth 2 against depth 1.
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchConfig {
    /// Solve before the opening roll, so the recorded turn value is the
    /// exact expectation over every roll (the exact TD backup). Decisions
    /// are unchanged: the same table answers every one of them.
    pub exact_root: bool,
    /// Selective 2-turn lookahead: leaves refined per turn (0 = off).
    pub lookahead_k: usize,
    /// Shift unrefined leaves by the mean refinement (`lookahead::combine`).
    pub lookahead_offset: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Main,
    Refine,
}

pub struct Game {
    pub id: u64,
    pub state: GameState,
    rng: Rng,
    /// Which evaluator scores the leaves of each seat's turns. All zero in
    /// self-play; the seating in an arena game. A lookahead's refinement
    /// solves use the MOVER's evaluator: it is the mover's search.
    pub evaluator_of_seat: [u8; 4],
    search_of_seat: [SearchConfig; 4],
    solver: Option<TurnSolver>,
    stage: Stage,
    /// Lookahead state between the main solve and its refinements.
    v1: Vec<f64>,
    refine_leaves: Vec<usize>,
    refine_weights: Vec<f64>,
    refiners: Vec<TurnSolver>,
    decisions: u32,
    max_turns: u32,
    pub status: Status,
    // Outputs, as `self_play.GameResult`.
    pub turns: u32,
    pub solves: u32,
    pub evaluator_rows: u64,
    pub turn_lengths: Vec<u32>,
    /// Encoded end-of-turn boards (non-terminal), row-major.
    pub features: Vec<f32>,
    /// The seat to move on each recorded board, for the winner's slot.
    board_active: Vec<u8>,
    /// Per turn: the solver's value (absolute seats) at its root -- right
    /// after the opening roll, or before it with `exact_root` -- or None when
    /// the opening roll busted with no solve. See `self_play.td_targets`.
    pub turn_values: Vec<Option<Vec<f64>>>,
    pending_value: Option<Vec<f64>>,
}

impl Game {
    pub fn new(
        state: GameState,
        seed: u64,
        evaluator_of_seat: [u8; 4],
        id: u64,
        max_turns: u32,
        search_of_seat: [SearchConfig; 4],
    ) -> Game {
        Game {
            id,
            state,
            rng: Rng::new(seed),
            evaluator_of_seat,
            search_of_seat,
            solver: None,
            stage: Stage::Main,
            v1: Vec::new(),
            refine_leaves: Vec::new(),
            refine_weights: Vec::new(),
            refiners: Vec::new(),
            decisions: 0,
            max_turns,
            status: Status::Ready,
            turns: 0,
            solves: 0,
            evaluator_rows: 0,
            turn_lengths: Vec::new(),
            features: Vec::new(),
            board_active: Vec::new(),
            turn_values: Vec::new(),
            pending_value: None,
        }
    }

    pub fn winner(&self) -> Option<u8> {
        self.state.winner
    }

    /// Encoding slot of the winner on each recorded board, as
    /// `encoder.seat_to_slot`.
    pub fn winner_slots(&self) -> Vec<u8> {
        let n = self.state.rules.num_players;
        let w = self.state.winner.expect("finished game");
        self.board_active.iter().map(|&a| (w + n - a) % n).collect()
    }

    /// Solves waiting for leaf values: the turn's main solve, or its
    /// lookahead refinements (one block each).
    pub fn pending_solvers(&self) -> Vec<&TurnSolver> {
        if self.status != Status::NeedsValues {
            return Vec::new();
        }
        match self.stage {
            Stage::Main => self.solver.iter().collect(),
            Stage::Refine => self.refiners.iter().collect(),
        }
    }

    /// Which evaluator the pending solves need (the mover's).
    pub fn pending_evaluator(&self) -> u8 {
        self.evaluator_of_seat[self.state.active_player as usize]
    }

    /// The mover's search settings.
    fn config(&self) -> SearchConfig {
        self.search_of_seat[self.state.active_player as usize]
    }

    fn end_turn(&mut self) {
        self.turns += 1;
        self.turn_lengths.push(self.decisions);
        self.turn_values.push(self.pending_value.take());
        self.decisions = 0;
        self.solver = None;
        if !self.state.game_over() {
            let row = self.features.len();
            self.features.resize(row + FEATURE_SIZE, 0.0);
            encode_board(&self.state, &mut self.features[row..]);
            self.board_active.push(self.state.active_player);
        }
    }

    /// Supply values for `pending_solvers`, one slice per solver in order
    /// (`num_leaves * num_players`, row-major, absolute seats).
    pub fn supply(&mut self, values: &[&[f64]]) -> Result<(), SolveError> {
        match self.stage {
            Stage::Main => {
                let config = self.config();
                let v = values[0];
                let main = self.solver.as_mut().expect("a pending solve");
                main.set_leaf_values(v)?;
                self.evaluator_rows += main.num_leaves() as u64;
                if config.lookahead_k > 0 {
                    let (leaves, weights) = crate::lookahead::choose(main, config.lookahead_k)?;
                    self.refiners = leaves
                        .iter()
                        .map(|&i| TurnSolver::new_cached(&main.leaf_board(i)))
                        .collect::<Result<_, _>>()?;
                    self.refine_leaves = leaves;
                    self.refine_weights = weights;
                    self.v1 = v.to_vec();
                    self.stage = Stage::Refine;
                    self.status = Status::NeedsValues;
                    return Ok(());
                }
            }
            Stage::Refine => {
                let mut refined = Vec::with_capacity(self.refiners.len());
                for (r, v) in self.refiners.iter_mut().zip(values) {
                    r.set_leaf_values(v)?;
                    self.evaluator_rows += r.num_leaves() as u64;
                    // The refined board is a turn start: its root roll value.
                    refined.push(r.value(RKey::default(), Phase::AwaitRoll, None)?);
                }
                let n = self.state.rules.num_players as usize;
                let new = crate::lookahead::combine(
                    &self.v1,
                    n,
                    &self.refine_leaves,
                    &self.refine_weights,
                    &refined,
                    self.config().lookahead_offset,
                );
                self.solver.as_mut().expect("main solve").rebackup(&new)?;
                self.refiners.clear();
                self.stage = Stage::Main;
            }
        }
        // `TurnSolver.value(state)` at the solve's root, as play_turn.
        let solver = self.solver.as_ref().expect("main solve");
        let key = RKey::from_runners(&self.state.runners);
        self.pending_value = Some(solver.value(key, self.state.phase, self.state.dice)?);
        self.status = Status::Ready;
        Ok(())
    }

    /// Play until the next solve needs values, or the game ends.
    pub fn advance(&mut self) {
        if self.status != Status::Ready {
            return;
        }
        loop {
            if let Some(solver) = self.solver.as_ref() {
                if self.state.phase == Phase::AwaitRoll {
                    // exact_root: solved before the opening roll; roll now.
                    let moves = self.state.roll(roll_dice(&mut self.rng)).expect("rollable");
                    if moves.is_empty() {
                        self.end_turn();
                    }
                    continue;
                }
                // Finish the turn from the solved table: play_turn's loop.
                let dice = self.state.dice.expect("AWAIT_MOVE holds dice");
                let key = RKey::from_runners(&self.state.runners);
                let mv = solver.choose_move(key, dice).expect("solved turn has a move");
                self.state.apply_move(mv).expect("solver move is legal");
                self.decisions += 1;
                let key = RKey::from_runners(&self.state.runners);
                if self.state.can_stop() && solver.should_stop(key).expect("reachable") {
                    self.state.stop().expect("can_stop");
                    self.end_turn();
                } else {
                    let moves = self.state.roll(roll_dice(&mut self.rng)).expect("rollable");
                    if moves.is_empty() {
                        self.end_turn();
                    }
                }
                continue;
            }
            if self.state.game_over() {
                self.status = Status::Done;
                return;
            }
            if self.turns >= self.max_turns {
                self.status = Status::Failed(format!(
                    "{:?} reached {} turns with no winner; the policy is likely never banking progress",
                    self.state.rules, self.max_turns
                ));
                return;
            }
            if !self.config().exact_root {
                // Start a turn: an opening bust needs no solve.
                let moves = self.state.roll(roll_dice(&mut self.rng)).expect("rollable");
                if moves.is_empty() {
                    self.end_turn();
                    continue;
                }
            }
            self.solver = Some(TurnSolver::new_cached(&self.state).expect("game is live"));
            self.solves += 1;
            self.status = Status::NeedsValues;
            return;
        }
    }
}

/// Where one waiting game's rows sit in a pending buffer.
#[derive(Clone, Debug)]
pub struct Block {
    pub game: usize,
    /// Which of the game's `pending_solvers` (0 unless a lookahead is
    /// refining several boards).
    pub sub: usize,
    pub rows: usize,
    pub evaluator: u8,
    pub num_players: u8,
    /// Seat to move on the leaf boards: rotates the net's output back.
    pub leaf_active: u8,
}

pub struct Pool {
    pub games: Vec<Game>,
    threads: rayon::ThreadPool,
    /// At most this many games are live (Ready or NeedsValues) at once;
    /// finished games are replaced from the queue in schedule order. The
    /// first version ran every game from the start: long games then trailed
    /// on alone, and the median round had 6 of 64 games waiting.
    in_flight: usize,
    /// At most this many leaf rows per round; waiting games beyond it wait
    /// for the next round (a game's blocks always go together). Needed once
    /// lookahead turned each waiting game into K+1 turn-start solves: 64
    /// games in flight at K=4 asked one forward for ~16M rows (16.8 GB).
    /// 0 = unlimited.
    max_rows: usize,
    /// Blocks of the last `pending` call, in buffer order.
    blocks: Vec<Block>,
}

impl Pool {
    /// `threads == 0` uses rayon's default (one per logical core);
    /// `in_flight == 0` puts every game in flight from the start.
    pub fn new(mut games: Vec<Game>, threads: usize, in_flight: usize, max_rows: usize) -> Pool {
        let threads = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon thread pool");
        let in_flight = if in_flight == 0 { games.len() } else { in_flight };
        for g in games.iter_mut().skip(in_flight) {
            g.status = Status::Queued;
        }
        Pool { games, threads, in_flight, max_rows, blocks: Vec::new() }
    }

    fn live(g: &Game) -> bool {
        g.status == Status::NeedsValues || g.status == Status::Ready
    }

    /// Fill free in-flight slots from the queue. True if any game started.
    fn promote(&mut self) -> bool {
        let live = self.games.iter().filter(|g| Self::live(g)).count();
        let mut free = self.in_flight.saturating_sub(live);
        let mut any = false;
        for g in self.games.iter_mut() {
            if free == 0 {
                break;
            }
            if g.status == Status::Queued {
                g.status = Status::Ready;
                free -= 1;
                any = true;
            }
        }
        any
    }

    /// Advance every Ready game to its next wait, refilling finished slots
    /// until the in-flight set is all waiting games (or the queue is empty).
    pub fn advance(&mut self) {
        loop {
            let games = &mut self.games;
            self.threads.install(|| games.par_iter_mut().for_each(|g| g.advance()));
            if !self.promote() {
                break;
            }
        }
    }

    pub fn running(&self) -> bool {
        self.games.iter().any(|g| Self::live(g) || g.status == Status::Queued)
    }

    pub fn failure(&self) -> Option<String> {
        self.games.iter().find_map(|g| match &g.status {
            Status::Failed(m) => Some(m.clone()),
            _ => None,
        })
    }

    /// Where each waiting game's leaf rows will sit in the pending buffer,
    /// in ascending game order. `write_pending` fills the buffer.
    pub fn pending_blocks(&mut self) -> &[Block] {
        let mut blocks = Vec::new();
        let mut rows = 0usize;
        for (i, g) in self.games.iter().enumerate() {
            let solvers = g.pending_solvers();
            if solvers.is_empty() {
                continue;
            }
            let game_rows: usize = solvers.iter().map(|s| s.num_leaves()).sum();
            // Always admit at least one game, however large.
            if self.max_rows != 0 && rows != 0 && rows + game_rows > self.max_rows {
                continue;
            }
            rows += game_rows;
            for (sub, s) in solvers.into_iter().enumerate() {
                blocks.push(Block {
                    game: i,
                    sub,
                    rows: s.num_leaves(),
                    evaluator: g.pending_evaluator(),
                    num_players: g.state.rules.num_players,
                    leaf_active: s.leaf_active_player(),
                });
            }
        }
        self.blocks = blocks;
        &self.blocks
    }

    pub fn pending_rows(&self) -> usize {
        self.blocks.iter().map(|b| b.rows).sum()
    }

    /// Encode every waiting game's leaves straight into `out`
    /// (`pending_rows * FEATURE_SIZE`), one disjoint slice per game, in
    /// parallel.
    pub fn write_pending(&self, out: &mut [f32]) {
        let mut slices: Vec<(usize, usize, &mut [f32])> = Vec::with_capacity(self.blocks.len());
        let mut rest = out;
        for b in &self.blocks {
            let (head, tail) = rest.split_at_mut(b.rows * FEATURE_SIZE);
            slices.push((b.game, b.sub, head));
            rest = tail;
        }
        let games = &self.games;
        self.threads.install(|| {
            slices.into_par_iter().for_each(|(i, sub, slice)| {
                games[i].pending_solvers()[sub].encode_leaves_into(slice);
            })
        });
    }

    /// Values for the last `pending` buffer, row-major, `num_players` per
    /// row (which varies by block when rule sets are mixed). Supplies each
    /// game and advances it to its next wait, all in parallel.
    /// Absolute-seat values from the net's seat-relative output
    /// (`pending_rows x 4` f32, slot 0 = seat to move on the leaf boards):
    /// `encoder.to_absolute` per block, done here because Python paid ~28 ms
    /// a round for it. Same arithmetic: widen to f64, place slot k at seat
    /// `(leaf_active + k) % n`, divide by the row sum taken in seat order.
    pub fn absolute_from_relative(&self, rel: &[f32]) -> Result<Vec<f64>, String> {
        const SLOTS: usize = 4;
        let rows: usize = self.blocks.iter().map(|b| b.rows).sum();
        if rel.len() != rows * SLOTS {
            return Err(format!("expected {} relative values, got {}", rows * SLOTS, rel.len()));
        }
        let total: usize = self.blocks.iter().map(|b| b.rows * b.num_players as usize).sum();
        let mut out = Vec::with_capacity(total);
        let mut src = rel.chunks_exact(SLOTS);
        for b in &self.blocks {
            let n = b.num_players as usize;
            for _ in 0..b.rows {
                let r = src.next().expect("row per leaf");
                let mut v = [0.0f64; 4];
                for (slot, &p) in r.iter().take(n).enumerate() {
                    v[(b.leaf_active as usize + slot) % n] = p as f64;
                }
                let mut sum = 0.0f64;
                for &x in &v[..n] {
                    sum += x;
                }
                if !(sum > 0.0) {
                    return Err("live seats carry no probability mass".to_string());
                }
                out.extend(v[..n].iter().map(|&x| x / sum));
            }
        }
        Ok(out)
    }

    pub fn resume(&mut self, values: &[f64]) -> Result<(), String> {
        let expected: usize = self.blocks.iter().map(|b| b.rows * b.num_players as usize).sum();
        if values.len() != expected {
            return Err(format!("expected {expected} values for the pending blocks, got {}", values.len()));
        }
        // Each game's slices, in its `pending_solvers` order.
        let mut by_game: Vec<Vec<&[f64]>> = vec![Vec::new(); self.games.len()];
        let mut at = 0;
        for b in &self.blocks {
            let len = b.rows * b.num_players as usize;
            debug_assert_eq!(by_game[b.game].len(), b.sub);
            by_game[b.game].push(&values[at..at + len]);
            at += len;
        }
        let games = &mut self.games;
        let errors: Vec<String> = self.threads.install(|| {
            games
                .par_iter_mut()
                .zip(by_game.par_iter())
                .filter_map(|(g, v)| {
                    if v.is_empty() {
                        return None;
                    }
                    if let Err(e) = g.supply(v) {
                        return Some(format!("game {}: {:?}", g.id, e));
                    }
                    g.advance();
                    None
                })
                .collect()
        });
        self.blocks.clear();
        if let Some(e) = errors.into_iter().next() {
            return Err(e);
        }
        // Games that finished this round free slots for queued ones.
        if self.promote() {
            self.advance();
        }
        Ok(())
    }
}
