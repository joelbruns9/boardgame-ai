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
use crate::engine::GameState;
use crate::rng::{roll_dice, Rng};
use crate::solver::{RKey, SolveError, TurnSolver};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// Scheduled, waiting for an in-flight slot.
    Queued,
    /// Not yet advanced, or values just supplied.
    Ready,
    /// A turn solve is waiting for leaf values.
    NeedsValues,
    Done,
    /// Ran past `max_turns` with no winner -- a defect, as in Python.
    Failed(String),
}

pub struct Game {
    pub id: u64,
    pub state: GameState,
    rng: Rng,
    /// Which evaluator scores the leaves of each seat's turns. All zero in
    /// self-play; the seating in an arena game.
    pub evaluator_of_seat: [u8; 4],
    solver: Option<TurnSolver>,
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
    /// Per turn: the solver's value (absolute seats) right after the
    /// opening roll, or None when the opening roll busted (no solve). The
    /// TD target's bootstrap; see `self_play.td_targets`.
    pub turn_values: Vec<Option<Vec<f64>>>,
    pending_value: Option<Vec<f64>>,
}

impl Game {
    pub fn new(state: GameState, seed: u64, evaluator_of_seat: [u8; 4], id: u64, max_turns: u32) -> Game {
        Game {
            id,
            state,
            rng: Rng::new(seed),
            evaluator_of_seat,
            solver: None,
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

    pub fn pending(&self) -> Option<&TurnSolver> {
        match self.status {
            Status::NeedsValues => self.solver.as_ref(),
            _ => None,
        }
    }

    /// Which evaluator the pending solve needs.
    pub fn pending_evaluator(&self) -> u8 {
        self.evaluator_of_seat[self.state.active_player as usize]
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

    /// Supply the pending solve's leaf values (`num_leaves * num_players`,
    /// row-major, absolute seats).
    pub fn supply(&mut self, values: &[f64]) -> Result<(), SolveError> {
        let solver = self.solver.as_mut().expect("a pending solve");
        solver.set_leaf_values(values)?;
        self.evaluator_rows += solver.num_leaves() as u64;
        // The state is still at the opening roll (AWAIT_MOVE): this is
        // `TurnSolver.value(state)` in play_turn.
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
            // Start a turn: an opening bust needs no solve.
            let moves = self.state.roll(roll_dice(&mut self.rng)).expect("rollable");
            if moves.is_empty() {
                self.end_turn();
                continue;
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
    /// Blocks of the last `pending` call, in buffer order.
    blocks: Vec<Block>,
}

impl Pool {
    /// `threads == 0` uses rayon's default (one per logical core);
    /// `in_flight == 0` puts every game in flight from the start.
    pub fn new(mut games: Vec<Game>, threads: usize, in_flight: usize) -> Pool {
        let threads = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon thread pool");
        let in_flight = if in_flight == 0 { games.len() } else { in_flight };
        for g in games.iter_mut().skip(in_flight) {
            g.status = Status::Queued;
        }
        Pool { games, threads, in_flight, blocks: Vec::new() }
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
        self.blocks = self
            .games
            .iter()
            .enumerate()
            .filter_map(|(i, g)| {
                let s = g.pending()?;
                Some(Block {
                    game: i,
                    rows: s.num_leaves(),
                    evaluator: g.pending_evaluator(),
                    num_players: g.state.rules.num_players,
                    leaf_active: s.leaf_active_player(),
                })
            })
            .collect();
        &self.blocks
    }

    pub fn pending_rows(&self) -> usize {
        self.blocks.iter().map(|b| b.rows).sum()
    }

    /// Encode every waiting game's leaves straight into `out`
    /// (`pending_rows * FEATURE_SIZE`), one disjoint slice per game, in
    /// parallel.
    pub fn write_pending(&self, out: &mut [f32]) {
        let mut slices: Vec<(usize, &mut [f32])> = Vec::with_capacity(self.blocks.len());
        let mut rest = out;
        for b in &self.blocks {
            let (head, tail) = rest.split_at_mut(b.rows * FEATURE_SIZE);
            slices.push((b.game, head));
            rest = tail;
        }
        let games = &self.games;
        self.threads.install(|| {
            slices.into_par_iter().for_each(|(i, slice)| {
                games[i].pending().expect("waiting").encode_leaves_into(slice);
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
        let mut slices: Vec<(usize, &[f64])> = Vec::with_capacity(self.blocks.len());
        let mut at = 0;
        for b in &self.blocks {
            let len = b.rows * b.num_players as usize;
            slices.push((b.game, &values[at..at + len]));
            at += len;
        }
        // Pair each waiting game with its slice without aliasing: walk the
        // games in order, since blocks are in ascending game order.
        let mut by_game: Vec<Option<&[f64]>> = vec![None; self.games.len()];
        for (g, s) in slices {
            by_game[g] = Some(s);
        }
        let games = &mut self.games;
        let errors: Vec<String> = self.threads.install(|| {
            games
                .par_iter_mut()
                .zip(by_game.par_iter())
                .filter_map(|(g, v)| {
                    let v = (*v)?;
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
