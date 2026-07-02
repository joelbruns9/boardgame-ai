//! Experimental deck=0 leaf-solving benchmark support.
//!
//! This module is intentionally not part of the training path. It exists to
//! preserve the K-sweep prototype for exact deck=0 draft leaf solving.

use super::*;

pub(crate) fn board_total_score(board: &RustBoard, harmony: bool, middle_kingdom: bool) -> i32 {
    let (territory, harmony_bonus, middle_bonus) = board.score(harmony, middle_kingdom);
    territory + harmony_bonus + middle_bonus
}

pub(crate) fn max_board_score_after_dominoes(
    board: &RustBoard,
    dominoes: &[u16],
    harmony: bool,
    middle_kingdom: bool,
) -> PyResult<i32> {
    if dominoes.is_empty() {
        return Ok(board_total_score(board, harmony, middle_kingdom));
    }

    let (ta, ca, tb, cb) = dom(dominoes[0]);
    let placements = board.legal_placements(ta, ca, tb, cb);
    if placements.is_empty() {
        return max_board_score_after_dominoes(board, &dominoes[1..], harmony, middle_kingdom);
    }

    let mut best = i32::MIN;
    for (x1, y1, x2, y2, flipped) in placements {
        let mut next = board.copy();
        next.place(ta, ca, tb, cb, x1, y1, x2, y2, flipped)?;
        let score = max_board_score_after_dominoes(&next, &dominoes[1..], harmony, middle_kingdom)?;
        if score > best {
            best = score;
        }
    }
    Ok(best)
}

pub(crate) fn solve_deck0_final_placement_separable_raw_margin(
    state: &RustGameState,
) -> PyResult<Option<i32>> {
    if state.phase == GAME_OVER {
        let (s0, s1) = state.scores();
        return Ok(Some(s0 - s1));
    }
    if state.phase != FINAL_PLACEMENT || !state.deck.is_empty() {
        return Ok(None);
    }

    let mut remaining: [Vec<u16>; 2] = [Vec::new(), Vec::new()];
    for &(player, domino_id) in state.pending_claims.iter().skip(state.actor_index) {
        remaining[player as usize].push(domino_id);
    }
    let p0 = max_board_score_after_dominoes(
        &state.boards[0],
        &remaining[0],
        state.harmony,
        state.middle_kingdom,
    )?;
    let p1 = max_board_score_after_dominoes(
        &state.boards[1],
        &remaining[1],
        state.harmony,
        state.middle_kingdom,
    )?;
    Ok(Some(p0 - p1))
}

#[derive(Default, Clone)]
pub(crate) struct Deck0DraftDpStats {
    pub(crate) nodes: u64,
    pub(crate) cache_hits: u64,
    pub(crate) final_cutoffs: u64,
    pub(crate) deadline_hits: u64,
}

pub(crate) fn solve_deck0_draft_dp_raw_margin(
    state: &RustGameState,
    max_current_row_len: usize,
    deadline: std::time::Instant,
    cache: &mut HashMap<EndgameKey, Option<f64>>,
    stats: &mut Deck0DraftDpStats,
) -> PyResult<Option<f64>> {
    if std::time::Instant::now() >= deadline {
        stats.deadline_hits += 1;
        return Ok(None);
    }

    if state.phase == GAME_OVER {
        let (s0, s1) = state.scores();
        return Ok(Some((s0 - s1) as f64));
    }
    if state.phase == FINAL_PLACEMENT && state.deck.is_empty() {
        let key = endgame_key(state);
        if let Some(&cached) = cache.get(&key) {
            stats.cache_hits += 1;
            return Ok(cached);
        }
        stats.final_cutoffs += 1;
        let raw =
            solve_deck0_final_placement_separable_raw_margin(state)?.map(|margin| margin as f64);
        cache.insert(key, raw);
        return Ok(raw);
    }
    if state.phase != PLACE_AND_SELECT
        || !state.deck.is_empty()
        || state.current_row.len() > max_current_row_len
    {
        return Ok(None);
    }

    let key = endgame_key(state);
    if let Some(&cached) = cache.get(&key) {
        stats.cache_hits += 1;
        return Ok(cached);
    }

    stats.nodes += 1;
    let actor = state.actor()?;
    let legal = state.legal_actions_indexed();
    if legal.is_empty() {
        cache.insert(key, None);
        return Ok(None);
    }

    let mut best = if actor == 0 {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let mut any = false;
    for &(_idx, placement, pick) in &legal {
        let child = state.step(placement, pick)?;
        let Some(v) =
            solve_deck0_draft_dp_raw_margin(&child, max_current_row_len, deadline, cache, stats)?
        else {
            cache.insert(key, None);
            return Ok(None);
        };
        any = true;
        if actor == 0 {
            if v > best {
                best = v;
            }
        } else if v < best {
            best = v;
        }
    }

    let result = if any { Some(best) } else { None };
    cache.insert(key, result);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first_deck0_draft_state(seed: u64) -> PyResult<RustGameState> {
        let mut rng = StdRng::seed_from_u64(seed ^ 0xDADADA);
        let mut state = new_game(seed, true, true);
        for _ in 0..200 {
            if state.phase == PLACE_AND_SELECT && state.deck.is_empty() {
                return Ok(state);
            }
            if state.phase == GAME_OVER {
                break;
            }
            let legal = state.legal_actions_indexed();
            let Some(&(_idx, placement, pick)) = legal.choose(&mut rng) else {
                break;
            };
            state = state.step(placement, pick)?;
        }
        Err(PyValueError::new_err(
            "did not reach deck0 PLACE_AND_SELECT",
        ))
    }

    #[test]
    fn deck0_draft_dp_matches_joint_solver_on_real_states() -> PyResult<()> {
        for seed in 0..5 {
            let state = first_deck0_draft_state(seed)?;
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let mut cache: HashMap<EndgameKey, Option<f64>> = HashMap::new();
            let mut stats = Deck0DraftDpStats::default();
            let dp = solve_deck0_draft_dp_raw_margin(&state, 4, deadline, &mut cache, &mut stats)?
                .expect("DP should solve deck0 draft state");
            let joint_deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let joint = solve_endgame_ab(
                &state,
                joint_deadline,
                MARGIN_LO,
                MARGIN_HI,
                SolverOrderMode::Lookahead2Clustered,
                0,
            )?
            .expect("joint solver should solve deck0 draft state");
            assert!(
                (dp - joint).abs() < 1e-9,
                "seed {seed}: DP {dp} != joint {joint}, nodes={}, cutoffs={}",
                stats.nodes,
                stats.final_cutoffs
            );
        }
        Ok(())
    }
}
