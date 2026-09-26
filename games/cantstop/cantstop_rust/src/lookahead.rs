//! Selective 2-turn lookahead (mirrors `games/cantstop/lookahead.py`).
//!
//! The turn solver's leaves are end-of-turn boards scored by the net. Here
//! the most important of them are scored instead by solving the NEXT
//! player's whole turn from them, from before their opening roll: the exact
//! expectation over their dice and their best play, with the net at the end
//! of *that* turn. That is one turn of search across the turn boundary,
//! applied where it matters:
//!
//! 1. solve the turn as usual (net at every leaf);
//! 2. `choose`: the bust board plus the top K-1 stop leaves by
//!    `TurnSolver::leaf_reach` (ties by key; only reachable leaves);
//! 3. solve the next turn from each chosen board (`TurnSolver` at AWAIT_ROLL,
//!    root roll value = the refined value);
//! 4. `combine`: chosen leaves take the refined value; with `offset`, every
//!    other leaf is shifted by the reach-weighted mean refinement, so a
//!    refined option is not favoured or penalised just for having been
//!    refined; then `rebackup`.
//!
//! Every sum runs in a fixed order so the Python reference matches bit for
//! bit.

use crate::solver::{SolveError, TurnSolver};

/// Leaves to refine, in refinement order, with their reach weights.
pub fn choose(main: &TurnSolver, k: usize) -> Result<(Vec<usize>, Vec<f64>), SolveError> {
    if k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let reach = main.leaf_reach()?;
    let mut stops: Vec<usize> = (1..main.num_leaves()).filter(|&i| reach[i] > 0.0).collect();
    stops.sort_by(|&x, &y| {
        reach[y]
            .partial_cmp(&reach[x])
            .expect("reach is finite")
            .then_with(|| main.leaf_key(x).cmp(&main.leaf_key(y)))
    });
    let mut leaves = vec![0usize];
    leaves.extend(stops.into_iter().take(k - 1));
    let weights = leaves.iter().map(|&i| reach[i]).collect();
    Ok((leaves, weights))
}

/// Clip negatives to zero and renormalise, in seat order (Python mirrors
/// the order). Review finding P1: a common shift keeps each row's SUM at 1
/// but not its entries >= 0 -- a hashed-mock 3p position gave a root value
/// of -0.005 for one seat and training targets down to -0.14, which the
/// soft cross-entropy rewards without bound. A shifted row always keeps a
/// positive entry (the shift sums to zero), so the total is positive.
fn project_to_simplex(row: &mut [f64]) {
    let mut total = 0.0f64;
    for x in row.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
        total += *x;
    }
    for x in row.iter_mut() {
        *x /= total;
    }
}

/// New leaf values: `v1` (row-major, `n` per leaf) with the chosen leaves
/// replaced by `refined` and, if `offset`, the rest shifted by the
/// reach-weighted mean of (refined - v1) over the chosen leaves.
pub fn combine(
    v1: &[f64],
    n: usize,
    leaves: &[usize],
    weights: &[f64],
    refined: &[Vec<f64>],
    offset: bool,
) -> Vec<f64> {
    let mut out = v1.to_vec();
    if offset {
        let mut d = [0.0f64; 4];
        let mut wsum = 0.0f64;
        for ((&leaf, &w), r) in leaves.iter().zip(weights).zip(refined) {
            for s in 0..n {
                d[s] += w * (r[s] - v1[leaf * n + s]);
            }
            wsum += w;
        }
        if wsum > 0.0 {
            for x in d.iter_mut().take(n) {
                *x /= wsum;
            }
            for (leaf, row) in out.chunks_exact_mut(n).enumerate() {
                if !leaves.contains(&leaf) {
                    for s in 0..n {
                        row[s] += d[s];
                    }
                    project_to_simplex(&mut row[..n]);
                }
            }
        }
    }
    for (&leaf, r) in leaves.iter().zip(refined) {
        out[leaf * n..(leaf + 1) * n].copy_from_slice(&r[..n]);
    }
    out
}
