//! Frozen Kingdomino NNUE reference feature derivation.
//!
//! This is deliberately a stateless, recomputing implementation.  It is the
//! Rust correctness oracle for the later incremental accumulator: first prove
//! that Rust derives exactly the same 5,710 sparse indices and 171-value summary
//! as the frozen Python schema, then optimize updates without changing meaning.

use std::collections::HashSet;

use super::{
    dom, idx, write_board_summary, RustBoard, RustGameState, BOARD_SUMMARY, CASTLE, CELLS, DIRS,
    EMPTY, FINAL_PLACEMENT, GAME_OVER, INITIAL_SELECTION, MAX_LEGAL_PLACEMENTS, N,
    PLACE_AND_SELECT,
};

pub(super) const CORE_SCHEMA_HASH: &str = "f4a681bf7fa8950c";
pub(super) const SUMMARY_SCHEMA_HASH: &str = "0eca00b192111097";

const NUM_DOMINOES: usize = 48;
const NUM_HALF: usize = 16;
const NUM_TERRAINS: usize = 6;
const CELL_SIDE: usize = 13;
const NUM_CELLS: usize = CELL_SIDE * CELL_SIDE;
const CASTLE_COORD: i8 = 7;
const MIN_CELL: i8 = 1;
const MAX_CELL: i8 = 13;

const BOARD_OFF: usize = 0;
const BOARD_SIZE: usize = 2 * NUM_CELLS * NUM_HALF;
const ROW_OFF: usize = BOARD_OFF + BOARD_SIZE;
pub(super) const BOARD_FEATURE_END: i32 = ROW_OFF as i32;
const PENDING_OFF: usize = ROW_OFF + NUM_DOMINOES;
const NEXT_OFF: usize = PENDING_OFF + 2 * NUM_DOMINOES;
const BAG_OFF: usize = NEXT_OFF + 2 * NUM_DOMINOES;
const PHASE_OFF: usize = BAG_OFF + NUM_DOMINOES;
const ACTOR_OFF: usize = PHASE_OFF + 4;
const SLOT_OFF: usize = ACTOR_OFF + 2;
const DISC_OFF: usize = SLOT_OFF + 4;
const RULES_OFF: usize = DISC_OFF + 2;
pub(super) const CORE_SIZE: usize = RULES_OFF + 2;

const BASE_SIZE: usize = 50;
const EXT_PER: usize = 39;
const GLOBAL_SIZE: usize = 43;
pub(super) const SUMMARY_SIZE: usize = BASE_SIZE + 2 * EXT_PER + GLOBAL_SIZE;

const MAX_CROWNS_PER_TERRAIN: [f64; NUM_TERRAINS] = [5.0, 6.0, 6.0, 6.0, 6.0, 10.0];
const MAX_HALVES_PER_TERRAIN: [f64; NUM_TERRAINS] = [26.0, 22.0, 18.0, 14.0, 10.0, 6.0];
const TOTAL_CATALOG_CROWNS: f64 = 39.0;

#[inline]
fn checked_domino_id(did: u16, field: &str) -> Result<usize, String> {
    if (1..=NUM_DOMINOES as u16).contains(&did) {
        Ok((did - 1) as usize)
    } else {
        Err(format!("{field}: domino id {did} outside 1..=48"))
    }
}

#[inline]
fn half_type_index(terrain: u8, crowns: u8) -> Option<usize> {
    match (terrain, crowns) {
        (2, 0) => Some(0),
        (2, 1) => Some(1),
        (3, 0) => Some(2),
        (3, 1) => Some(3),
        (4, 0) => Some(4),
        (4, 1) => Some(5),
        (5, 0) => Some(6),
        (5, 1) => Some(7),
        (5, 2) => Some(8),
        (6, 0) => Some(9),
        (6, 1) => Some(10),
        (6, 2) => Some(11),
        (7, 0) => Some(12),
        (7, 1) => Some(13),
        (7, 2) => Some(14),
        (7, 3) => Some(15),
        _ => None,
    }
}

#[inline]
fn cell_index(x: i8, y: i8) -> Result<usize, String> {
    if !(MIN_CELL..=MAX_CELL).contains(&x) || !(MIN_CELL..=MAX_CELL).contains(&y) {
        return Err(format!(
            "occupied cell ({x},{y}) outside the reachable 13x13 NNUE region"
        ));
    }
    Ok((y - MIN_CELL) as usize * CELL_SIDE + (x - MIN_CELL) as usize)
}

fn validate_board(board: &RustBoard, player: usize) -> Result<(), String> {
    if board.castle_x != CASTLE_COORD || board.castle_y != CASTLE_COORD {
        return Err(format!(
            "board {player}: NNUE requires castle at (7,7), got ({},{})",
            board.castle_x, board.castle_y
        ));
    }
    for y in 0..N as i8 {
        for x in 0..N as i8 {
            let i = idx(x, y);
            let terrain = board.terrain[i];
            if terrain == EMPTY {
                continue;
            }
            if terrain == CASTLE {
                if x != CASTLE_COORD || y != CASTLE_COORD {
                    return Err(format!("board {player}: castle terrain at ({x},{y})"));
                }
                continue;
            }
            cell_index(x, y)?;
            half_type_index(terrain, board.crowns[i]).ok_or_else(|| {
                format!(
                    "board {player}: invalid half type ({terrain},{}) at ({x},{y})",
                    board.crowns[i]
                )
            })?;
        }
    }
    Ok(())
}

fn validate_state(state: &RustGameState, perspective: u8) -> Result<(), String> {
    if perspective >= 2 {
        return Err(format!("perspective must be 0 or 1, got {perspective}"));
    }
    if state.phase > GAME_OVER {
        return Err(format!("invalid phase {}", state.phase));
    }
    validate_board(&state.boards[0], 0)?;
    validate_board(&state.boards[1], 1)?;
    for &did in &state.current_row {
        checked_domino_id(did, "current_row")?;
    }
    for &did in &state.deck {
        checked_domino_id(did, "deck")?;
    }
    for &(owner, did) in &state.pending_claims {
        if owner >= 2 {
            return Err(format!("pending_claims: invalid owner {owner}"));
        }
        checked_domino_id(did, "pending_claims")?;
    }
    for &(owner, did) in &state.next_claims {
        if owner >= 2 {
            return Err(format!("next_claims: invalid owner {owner}"));
        }
        checked_domino_id(did, "next_claims")?;
    }
    if matches!(state.phase, PLACE_AND_SELECT | FINAL_PLACEMENT)
        && state.actor_index >= state.pending_claims.len()
    {
        return Err(format!(
            "actor_index {} outside pending_claims length {}",
            state.actor_index,
            state.pending_claims.len()
        ));
    }
    Ok(())
}

fn current_actor(state: &RustGameState) -> Result<u8, String> {
    match state.phase {
        INITIAL_SELECTION => {
            if state.initial_pick_count >= 4 {
                return Err(format!(
                    "initial_pick_count {} outside 0..4",
                    state.initial_pick_count
                ));
            }
            let s = state.start_player;
            Ok([s, 1 - s, 1 - s, s][state.initial_pick_count])
        }
        PLACE_AND_SELECT | FINAL_PLACEMENT => Ok(state.pending_claims[state.actor_index].0),
        _ => Err("terminal state has no current actor".to_owned()),
    }
}

fn turn_slot(state: &RustGameState) -> usize {
    match state.phase {
        INITIAL_SELECTION => state.initial_pick_count.min(3),
        PLACE_AND_SELECT | FINAL_PLACEMENT => state.actor_index.min(3),
        _ => 0,
    }
}

pub(super) fn sparse_indices(state: &RustGameState, perspective: u8) -> Result<Vec<i32>, String> {
    validate_state(state, perspective)?;
    let opponent = 1 - perspective;
    let mut active: Vec<i32> = Vec::with_capacity(160);

    for (role, player) in [(0usize, perspective), (1usize, opponent)] {
        let board = &state.boards[player as usize];
        for y in board.min_y..=board.max_y {
            for x in board.min_x..=board.max_x {
                let i = idx(x, y);
                let terrain = board.terrain[i];
                if terrain == EMPTY || terrain == CASTLE {
                    continue;
                }
                let cell = cell_index(x, y)?;
                let half = half_type_index(terrain, board.crowns[i])
                    .ok_or_else(|| format!("invalid half type ({terrain},{})", board.crowns[i]))?;
                active.push((BOARD_OFF + (role * NUM_CELLS + cell) * NUM_HALF + half) as i32);
            }
        }
    }

    active.extend(non_board_indices(state, perspective)?);

    active.sort_unstable();
    if active.windows(2).any(|w| w[0] == w[1]) {
        return Err("duplicate active NNUE feature index".to_owned());
    }
    if active.iter().any(|&i| i < 0 || i as usize >= CORE_SIZE) {
        return Err("NNUE feature index outside frozen core schema".to_owned());
    }
    Ok(active)
}

/// All sparse banks except board cells. Used by the incremental transition path:
/// board additions come directly from the placement undo, while dynamic row/
/// claim/bag/scalar banks are cheaply re-derived after every move/chance result.
pub(super) fn non_board_indices(
    state: &RustGameState,
    perspective: u8,
) -> Result<Vec<i32>, String> {
    if perspective >= 2 {
        return Err(format!("perspective must be 0 or 1, got {perspective}"));
    }
    let opponent = 1 - perspective;
    let mut active = Vec::with_capacity(64);
    for &did in &state.current_row {
        active.push((ROW_OFF + checked_domino_id(did, "current_row")?) as i32);
    }
    for &(owner, did) in state.pending_claims.iter().skip(state.actor_index) {
        let role = usize::from(owner != perspective);
        active
            .push((PENDING_OFF + role * NUM_DOMINOES + checked_domino_id(did, "pending")?) as i32);
    }
    for &(owner, did) in &state.next_claims {
        let role = usize::from(owner != perspective);
        active.push((NEXT_OFF + role * NUM_DOMINOES + checked_domino_id(did, "next")?) as i32);
    }
    for &did in &state.deck {
        active.push((BAG_OFF + checked_domino_id(did, "deck")?) as i32);
    }

    active.push((PHASE_OFF + state.phase as usize) as i32);
    if state.phase != GAME_OVER {
        let actor_role = usize::from(current_actor(state)? != perspective);
        active.push((ACTOR_OFF + actor_role) as i32);
    }
    active.push((SLOT_OFF + turn_slot(state)) as i32);
    for (role, player) in [(0usize, perspective), (1usize, opponent)] {
        if state.discards[player as usize] > 0 {
            active.push((DISC_OFF + role) as i32);
        }
    }
    if state.harmony {
        active.push(RULES_OFF as i32);
    }
    if state.middle_kingdom {
        active.push((RULES_OFF + 1) as i32);
    }

    active.sort_unstable();
    Ok(active)
}

pub(super) fn board_feature_index(
    perspective: u8,
    owner: u8,
    x: i8,
    y: i8,
    terrain: u8,
    crowns: u8,
) -> Result<i32, String> {
    if perspective >= 2 || owner >= 2 {
        return Err("invalid perspective/owner for board feature".to_owned());
    }
    let role = usize::from(owner != perspective);
    let cell = cell_index(x, y)?;
    let half = half_type_index(terrain, crowns)
        .ok_or_else(|| format!("invalid half type ({terrain},{crowns})"))?;
    Ok((BOARD_OFF + (role * NUM_CELLS + cell) * NUM_HALF + half) as i32)
}

#[derive(Default)]
struct BoardFacts {
    cell_count: [i32; NUM_TERRAINS],
    crown_count: [i32; NUM_TERRAINS],
    largest_size: [i32; NUM_TERRAINS],
    largest_crowns: [i32; NUM_TERRAINS],
    largest_crownless: [i32; NUM_TERRAINS],
    crownless_region_count: i32,
    stranded_crowns: i32,
    global_largest: i32,
    open_frontier: [i32; NUM_TERRAINS],
    holes: i32,
    gaps: i32,
    castle_extent: [i32; 4],
}

fn board_facts(board: &RustBoard) -> BoardFacts {
    let mut facts = BoardFacts::default();
    let mut visited = [false; CELLS];

    for sy in board.min_y..=board.max_y {
        for sx in board.min_x..=board.max_x {
            let si = idx(sx, sy);
            let terrain = board.terrain[si];
            if visited[si] || terrain < 2 {
                continue;
            }
            let ti = (terrain - 2) as usize;
            let mut stack = Vec::with_capacity(49);
            stack.push((sx, sy));
            visited[si] = true;
            let mut size = 0i32;
            let mut crowns = 0i32;
            while let Some((x, y)) = stack.pop() {
                size += 1;
                crowns += board.crowns[idx(x, y)] as i32;
                for (dx, dy) in DIRS {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= N as i8 || ny >= N as i8 {
                        continue;
                    }
                    let ni = idx(nx, ny);
                    if !visited[ni] && board.terrain[ni] == terrain {
                        visited[ni] = true;
                        stack.push((nx, ny));
                    }
                }
            }
            facts.cell_count[ti] += size;
            facts.crown_count[ti] += crowns;
            facts.global_largest = facts.global_largest.max(size);
            if size > facts.largest_size[ti]
                || (size == facts.largest_size[ti] && crowns > facts.largest_crowns[ti])
            {
                facts.largest_size[ti] = size;
                facts.largest_crowns[ti] = crowns;
            }
            if crowns == 0 {
                facts.crownless_region_count += 1;
                facts.largest_crownless[ti] = facts.largest_crownless[ti].max(size);
            }
            if size == 1 {
                facts.stranded_crowns += crowns;
            }
        }
    }

    let mut frontier: [HashSet<usize>; NUM_TERRAINS] = std::array::from_fn(|_| HashSet::new());
    for y in board.min_y..=board.max_y {
        for x in board.min_x..=board.max_x {
            let terrain = board.terrain[idx(x, y)];
            if terrain < 2 {
                continue;
            }
            for (dx, dy) in DIRS {
                let nx = x + dx;
                let ny = y + dy;
                if !(MIN_CELL..=MAX_CELL).contains(&nx)
                    || !(MIN_CELL..=MAX_CELL).contains(&ny)
                    || board.terrain[idx(nx, ny)] != EMPTY
                {
                    continue;
                }
                let width = board.max_x.max(nx) - board.min_x.min(nx) + 1;
                let height = board.max_y.max(ny) - board.min_y.min(ny) + 1;
                if width <= 7 && height <= 7 {
                    frontier[(terrain - 2) as usize].insert(idx(nx, ny));
                }
            }
        }
    }
    for (ti, cells) in frontier.iter().enumerate() {
        facts.open_frontier[ti] = cells.len() as i32;
    }

    for y in board.min_y..=board.max_y {
        for x in board.min_x..=board.max_x {
            if board.terrain[idx(x, y)] != EMPTY {
                continue;
            }
            if DIRS
                .iter()
                .all(|&(dx, dy)| board.terrain[idx(x + dx, y + dy)] != EMPTY)
            {
                facts.holes += 1;
            }
        }
    }
    let area = (board.max_x - board.min_x + 1) as i32 * (board.max_y - board.min_y + 1) as i32;
    facts.gaps = area - board.occupied as i32;
    facts.castle_extent = [
        (board.castle_x - board.min_x) as i32,
        (board.max_x - board.castle_x) as i32,
        (board.castle_y - board.min_y) as i32,
        (board.max_y - board.castle_y) as i32,
    ];
    facts
}

#[inline]
fn normalized(value: i32, denominator: f64) -> f32 {
    (value as f64 / denominator) as f32
}

fn append_extension(out: &mut Vec<f32>, board: &RustBoard) {
    let f = board_facts(board);
    out.extend(f.cell_count.iter().map(|&v| normalized(v, 48.0)));
    out.extend(
        f.crown_count
            .iter()
            .enumerate()
            .map(|(t, &v)| normalized(v, MAX_CROWNS_PER_TERRAIN[t])),
    );
    out.extend(
        f.largest_crowns
            .iter()
            .enumerate()
            .map(|(t, &v)| normalized(v, MAX_CROWNS_PER_TERRAIN[t])),
    );
    out.push(normalized(f.global_largest, 48.0));
    out.push(normalized(f.crownless_region_count, 48.0));
    out.push(normalized(f.stranded_crowns, TOTAL_CATALOG_CROWNS));
    out.extend(f.open_frontier.iter().map(|&v| normalized(v, 48.0)));
    out.push(normalized(f.holes, 48.0));
    out.push(normalized(f.gaps, 48.0));
    out.extend(f.castle_extent.iter().map(|&v| normalized(v, 6.0)));
    out.extend(f.largest_crownless.iter().map(|&v| normalized(v, 48.0)));
}

fn append_bag(out: &mut Vec<f32>, state: &RustGameState) {
    let mut half_count = [0i32; NUM_TERRAINS];
    let mut crowns = [0i32; NUM_TERRAINS];
    for &did in &state.deck {
        let (ta, ca, tb, cb) = dom(did);
        for (terrain, crown) in [(ta, ca), (tb, cb)] {
            let ti = (terrain - 2) as usize;
            half_count[ti] += 1;
            crowns[ti] += crown as i32;
        }
    }
    out.extend(
        half_count
            .iter()
            .enumerate()
            .map(|(t, &v)| normalized(v, MAX_HALVES_PER_TERRAIN[t])),
    );
    out.extend(
        crowns
            .iter()
            .enumerate()
            .map(|(t, &v)| normalized(v, MAX_CROWNS_PER_TERRAIN[t])),
    );
}

fn append_claims(out: &mut Vec<f32>, state: &RustGameState, perspective: u8) {
    let start = out.len();
    out.resize(start + 24, 0.0);
    if !matches!(state.phase, PLACE_AND_SELECT | FINAL_PLACEMENT) {
        return;
    }
    for (k, &(owner, did)) in state
        .pending_claims
        .iter()
        .skip(state.actor_index)
        .take(4)
        .enumerate()
    {
        let (ta, ca, tb, cb) = dom(did);
        let legal = state.boards[owner as usize]
            .legal_placements(ta, ca, tb, cb)
            .len();
        let base = start + k * 6;
        out[base] = 1.0;
        out[base + 1] =
            (legal.min(MAX_LEGAL_PLACEMENTS as usize) as f64 / MAX_LEGAL_PLACEMENTS as f64) as f32;
        out[base + 2] = if legal == 0 { 1.0 } else { 0.0 };
        out[base + 3] = if owner == perspective { 1.0 } else { -1.0 };
        out[base + 4] = ((did - 1) as f64 / 47.0) as f32;
        out[base + 5] = (k.min(3) as f64 / 3.0) as f32;
    }
}

fn append_pick_positions(out: &mut Vec<f32>, state: &RustGameState, perspective: u8) {
    let mut claims = state.next_claims.clone();
    claims.sort_by_key(|&(_, did)| did);
    for k in 0..4 {
        out.push(match claims.get(k) {
            Some(&(owner, _)) if owner == perspective => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        });
    }
}

fn append_progress_and_fill(out: &mut Vec<f32>, state: &RustGameState, perspective: u8) {
    let placed = [
        state.boards[0].occupied.saturating_sub(1) as i32,
        state.boards[1].occupied.saturating_sub(1) as i32,
    ];
    let placed_dominoes = (placed[0] + placed[1]) as f64 / 2.0;
    let discarded = (state.discards[0] + state.discards[1]) as f64;
    out.push(((placed_dominoes + discarded) / 48.0) as f32);

    for player in [perspective, 1 - perspective] {
        let board = &state.boards[player as usize];
        let area = (board.max_x - board.min_x + 1) as i32 * (board.max_y - board.min_y + 1) as i32;
        out.push(if area == 0 {
            0.0
        } else {
            (placed[player as usize] as f64 / area as f64) as f32
        });
    }
}

pub(super) fn summary(state: &RustGameState, perspective: u8) -> Result<Vec<f32>, String> {
    validate_state(state, perspective)?;
    let opponent = 1 - perspective;
    let mut out = vec![0.0f32; BASE_SIZE];
    write_board_summary(&mut out, 0, state, perspective);
    write_board_summary(&mut out, BOARD_SUMMARY, state, opponent);
    append_extension(&mut out, &state.boards[perspective as usize]);
    append_extension(&mut out, &state.boards[opponent as usize]);
    append_bag(&mut out, state);
    append_claims(&mut out, state, perspective);
    append_pick_positions(&mut out, state, perspective);
    append_progress_and_fill(&mut out, state, perspective);
    if out.len() != SUMMARY_SIZE {
        return Err(format!(
            "Rust NNUE summary size {} != frozen {SUMMARY_SIZE}",
            out.len()
        ));
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err("Rust NNUE summary contains a non-finite value".to_owned());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::new_game;

    #[test]
    fn frozen_layout_sizes_are_consistent() {
        assert_eq!(RULES_OFF + 2, CORE_SIZE);
        assert_eq!(BASE_SIZE + 2 * EXT_PER + GLOBAL_SIZE, SUMMARY_SIZE);
    }

    #[test]
    fn catalog_half_types_are_all_supported() {
        for did in 1..=48 {
            let (ta, ca, tb, cb) = dom(did);
            assert!(half_type_index(ta, ca).is_some());
            assert!(half_type_index(tb, cb).is_some());
        }
    }

    #[test]
    fn fresh_state_encodes_both_halves() {
        let state = new_game(7, true, true);
        let sparse = sparse_indices(&state, 0).unwrap();
        let summary = summary(&state, 0).unwrap();
        assert_eq!(summary.len(), SUMMARY_SIZE);
        assert!(sparse.windows(2).all(|w| w[0] < w[1]));
        assert!(sparse.iter().all(|&i| i >= 0 && (i as usize) < CORE_SIZE));
    }
}
