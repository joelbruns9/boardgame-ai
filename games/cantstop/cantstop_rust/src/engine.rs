//! Can't Stop rules engine — a mirror of `games/cantstop/engine.py` (M1).
//!
//! Python is the reference. Every rule here is transcribed from it, and the
//! equivalence gate (`games/cantstop/rust_equiv.py`) replays seeded games in
//! lockstep and compares the full state after every transition. Where the
//! representation differs, the difference is noted at the field.
//!
//! Columns are indexed by their dice sum, 2..=12, in arrays of length 13 so
//! no index arithmetic can drift between the two sides; slots 0 and 1 are
//! never used.

pub const MIN_COL: usize = 2;
pub const MAX_COL: usize = 12;
pub const COLUMN_HEIGHTS: [u8; 13] = [0, 0, 3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3];
pub const MAX_RUNNERS: usize = 3;
pub const MAX_PLAYERS: usize = 4;

pub const BASE_COLUMNS_TO_WIN: u8 = 3;

/// Extended-columns target for n players, as `EXTENDED_COLUMNS_TO_WIN`.
pub fn extended_columns_to_win(num_players: u8) -> Option<u8> {
    match num_players {
        2 => Some(5),
        3 => Some(4),
        4 => Some(3),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuleSet {
    pub num_players: u8,
    pub columns_to_win: u8,
    pub blocking: bool,
}

impl RuleSet {
    /// Validates exactly as `RuleSet.__post_init__` does.
    pub fn new(num_players: u8, columns_to_win: u8, blocking: bool) -> Result<Self, EngineError> {
        let ext = extended_columns_to_win(num_players).ok_or_else(|| {
            EngineError(format!("num_players must be 2-4, got {num_players}"))
        })?;
        if columns_to_win != BASE_COLUMNS_TO_WIN && columns_to_win != ext {
            return Err(EngineError(format!(
                "{num_players} players cannot play to {columns_to_win} columns"
            )));
        }
        Ok(RuleSet { num_players, columns_to_win, blocking })
    }
}

/// Numbering matches Python's `Phase(IntEnum)`; the snapshot carries it as
/// an integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    AwaitRoll = 0,
    AwaitMove = 1,
    AwaitDecision = 2,
    GameOver = 3,
}

impl Phase {
    pub fn from_u8(v: u8) -> Option<Phase> {
        match v {
            0 => Some(Phase::AwaitRoll),
            1 => Some(Phase::AwaitMove),
            2 => Some(Phase::AwaitDecision),
            3 => Some(Phase::GameOver),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Phase::AwaitRoll => "AWAIT_ROLL",
            Phase::AwaitMove => "AWAIT_MOVE",
            Phase::AwaitDecision => "AWAIT_DECISION",
            Phase::GameOver => "GAME_OVER",
        }
    }
}

/// Every refusal Python raises as `ValueError`. The pyo3 layer maps this to
/// `ValueError` so callers catch the same exception on both sides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineError(pub String);

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A move: the columns advanced, one entry per step, sorted. `len` is 1 or
/// 2; `cols[1]` is meaningless when `len == 1`. Ordering (derived, field
/// order `len` last) is NOT Python's tuple ordering -- use `sort_key`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Move {
    pub cols: [u8; 2],
    pub len: u8,
}

impl Move {
    pub fn single(a: u8) -> Move {
        Move { cols: [a, 0], len: 1 }
    }

    pub fn pair(a: u8, b: u8) -> Move {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        Move { cols: [lo, hi], len: 2 }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.cols[..self.len as usize]
    }

    /// Python sorts moves as tuples: `(6,) < (6, 8) < (7,)`. A shorter tuple
    /// that is a prefix sorts first, which is exactly slice ordering.
    pub fn sort_key(&self) -> &[u8] {
        self.as_slice()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameState {
    pub rules: RuleSet,
    pub active_player: u8,
    /// progress[p][col]; zero on claimed columns. Rows beyond
    /// `num_players` stay zero.
    pub progress: [[u8; 13]; MAX_PLAYERS],
    /// claimed_by[col] = player, or -1 for Python's `None`.
    pub claimed_by: [i8; 13],
    /// runners[col] = absolute position, 0 = no runner. Python keeps a dict;
    /// a runner is always at position >= 1, so 0 is free as the sentinel.
    /// Python iterates the dict in insertion order, but no rule depends on
    /// that order (every column is handled independently), so none is kept.
    pub runners: [u8; 13],
    /// Sorted, while AWAIT_MOVE.
    pub dice: Option<[u8; 4]>,
    pub phase: Phase,
    pub winner: Option<u8>,
}

impl GameState {
    pub fn new(rules: RuleSet) -> GameState {
        GameState {
            rules,
            active_player: 0,
            progress: [[0; 13]; MAX_PLAYERS],
            claimed_by: [-1; 13],
            runners: [0; 13],
            dice: None,
            phase: Phase::AwaitRoll,
            winner: None,
        }
    }

    pub fn game_over(&self) -> bool {
        self.phase == Phase::GameOver
    }

    pub fn num_runners(&self) -> usize {
        self.runners.iter().filter(|&&r| r != 0).count()
    }

    pub fn claimed_count(&self, player: u8) -> u8 {
        self.claimed_by
            .iter()
            .filter(|&&c| c == player as i8)
            .count() as u8
    }

    /// Active player's current position on col, runner included.
    pub fn position(&self, col: u8) -> u8 {
        let r = self.runners[col as usize];
        if r != 0 {
            r
        } else {
            self.progress[self.active_player as usize][col as usize]
        }
    }

    fn has_runner(&self, col: u8) -> bool {
        self.runners[col as usize] != 0
    }

    /// Active player may place/advance a runner on col (ignoring the cap).
    fn column_open(&self, col: u8) -> bool {
        self.claimed_by[col as usize] == -1 && self.position(col) < COLUMN_HEIGHTS[col as usize]
    }

    /// Legal moves for the active player given (sorted or unsorted) dice,
    /// sorted as Python sorts its tuples. Empty = bust.
    pub fn legal_moves(&self, dice: [u8; 4]) -> Vec<Move> {
        let free_slots = MAX_RUNNERS - self.num_runners();
        let mut moves: Vec<Move> = Vec::with_capacity(6);
        let add = |m: Move, moves: &mut Vec<Move>| {
            if !moves.contains(&m) {
                moves.push(m);
            }
        };
        for (a, b) in dice_pairings(dice) {
            if a == b {
                if !self.column_open(a) {
                    continue;
                }
                if !self.has_runner(a) && free_slots == 0 {
                    continue;
                }
                let room = COLUMN_HEIGHTS[a as usize] - self.position(a);
                add(if room >= 2 { Move::pair(a, a) } else { Move::single(a) }, &mut moves);
                continue;
            }
            let ok_a = self.column_open(a) && (self.has_runner(a) || free_slots > 0);
            let ok_b = self.column_open(b) && (self.has_runner(b) || free_slots > 0);
            let new_needed = (!self.has_runner(a)) as usize + (!self.has_runner(b)) as usize;
            if ok_a && ok_b && new_needed <= free_slots {
                add(Move::pair(a, b), &mut moves);
            } else {
                if ok_a {
                    add(Move::single(a), &mut moves);
                }
                if ok_b {
                    add(Move::single(b), &mut moves);
                }
            }
        }
        moves.sort_by(|x, y| x.sort_key().cmp(y.sort_key()));
        moves
    }

    fn require(&self, phases: &[Phase]) -> Result<(), EngineError> {
        if phases.contains(&self.phase) {
            Ok(())
        } else {
            Err(EngineError(format!("illegal in phase {}", self.phase.name())))
        }
    }

    /// Roll for the active player. Busts (and passes the turn) if no legal
    /// move exists. Returns the legal moves; empty means the roll busted.
    pub fn roll(&mut self, dice: [u8; 4]) -> Result<Vec<Move>, EngineError> {
        self.require(&[Phase::AwaitRoll, Phase::AwaitDecision])?;
        let mut dice = dice;
        dice.sort_unstable();
        let moves = self.legal_moves(dice);
        if moves.is_empty() {
            self.bust()?;
            return Ok(moves);
        }
        self.dice = Some(dice);
        self.phase = Phase::AwaitMove;
        Ok(moves)
    }

    pub fn apply_move(&mut self, mv: Move) -> Result<(), EngineError> {
        self.require(&[Phase::AwaitMove])?;
        let dice = self.dice.expect("AWAIT_MOVE always holds dice");
        let mv = if mv.len == 2 { Move::pair(mv.cols[0], mv.cols[1]) } else { mv };
        if !self.legal_moves(dice).contains(&mv) {
            return Err(EngineError(format!(
                "illegal move {:?} for dice {:?}",
                mv.as_slice(),
                dice
            )));
        }
        for &col in mv.as_slice() {
            // Sequential, exactly like Python's loop: the second step of a
            // double reads the runner the first step just placed.
            self.runners[col as usize] = self.position(col) + 1;
        }
        self.dice = None;
        self.phase = Phase::AwaitDecision;
        Ok(())
    }

    /// Blocking variant: a runner shares a space with another player's saved
    /// marker, so the turn cannot end by stopping.
    pub fn stop_blocked(&self) -> bool {
        if !self.rules.blocking {
            return false;
        }
        let me = self.active_player as usize;
        for col in MIN_COL..=MAX_COL {
            let pos = self.runners[col];
            if pos == 0 {
                continue;
            }
            for p in 0..self.rules.num_players as usize {
                if p != me && self.progress[p][col] == pos {
                    return true;
                }
            }
        }
        false
    }

    pub fn can_stop(&self) -> bool {
        self.phase == Phase::AwaitDecision && !self.stop_blocked()
    }

    /// Bank runners, claim finished columns, check for a win, pass the turn.
    pub fn stop(&mut self) -> Result<(), EngineError> {
        self.require(&[Phase::AwaitDecision])?;
        if self.stop_blocked() {
            return Err(EngineError(
                "cannot stop: a runner is on another player's marker".to_string(),
            ));
        }
        let me = self.active_player;
        for col in MIN_COL..=MAX_COL {
            let pos = self.runners[col];
            if pos == 0 {
                continue;
            }
            if pos >= COLUMN_HEIGHTS[col] {
                self.claimed_by[col] = me as i8;
                for p in 0..self.rules.num_players as usize {
                    self.progress[p][col] = 0;
                }
            } else {
                self.progress[me as usize][col] = pos;
            }
        }
        self.runners = [0; 13];
        if self.claimed_count(me) >= self.rules.columns_to_win {
            self.winner = Some(me);
            self.phase = Phase::GameOver;
            return Ok(());
        }
        self.next_player();
        Ok(())
    }

    /// Lose the runners and pass the turn.
    pub fn bust(&mut self) -> Result<(), EngineError> {
        self.require(&[Phase::AwaitRoll, Phase::AwaitDecision])?;
        self.runners = [0; 13];
        self.next_player();
        Ok(())
    }

    fn next_player(&mut self) {
        self.active_player = (self.active_player + 1) % self.rules.num_players;
        self.dice = None;
        self.phase = Phase::AwaitRoll;
    }
}

/// Distinct (low, high) sum pairs from the three ways to split 4 dice, in
/// Python's order: (d0+d1, d2+d3), (d0+d2, d1+d3), (d0+d3, d1+d2).
pub fn dice_pairings(dice: [u8; 4]) -> Vec<(u8, u8)> {
    let [d0, d1, d2, d3] = dice;
    let mut out: Vec<(u8, u8)> = Vec::with_capacity(3);
    for (a, b) in [(d0 + d1, d2 + d3), (d0 + d2, d1 + d3), (d0 + d3, d1 + d2)] {
        let pair = if a <= b { (a, b) } else { (b, a) };
        if !out.contains(&pair) {
            out.push(pair);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules(n: u8) -> RuleSet {
        RuleSet::new(n, 3, false).unwrap()
    }

    #[test]
    fn move_order_matches_python_tuples() {
        let mut v = vec![Move::single(7), Move::pair(6, 8), Move::single(6)];
        v.sort_by(|x, y| x.sort_key().cmp(y.sort_key()));
        assert_eq!(v, vec![Move::single(6), Move::pair(6, 8), Move::single(7)]);
    }

    #[test]
    fn rules_validate_like_python() {
        assert!(RuleSet::new(2, 5, true).is_ok());
        assert!(RuleSet::new(2, 4, false).is_err());
        assert!(RuleSet::new(5, 3, false).is_err());
    }

    #[test]
    fn double_with_one_space_left_is_a_single() {
        let mut s = GameState::new(rules(2));
        s.progress[0][2] = 2; // column 2 has height 3
        assert_eq!(s.legal_moves([1, 1, 1, 1]), vec![Move::single(2)]);
    }

    #[test]
    fn stop_claims_and_clears_every_marker() {
        let mut s = GameState::new(rules(2));
        s.progress[1][2] = 2;
        s.runners[2] = 3;
        s.phase = Phase::AwaitDecision;
        s.stop().unwrap();
        assert_eq!(s.claimed_by[2], 0);
        assert_eq!(s.progress[1][2], 0);
        assert_eq!(s.active_player, 1);
    }
}
