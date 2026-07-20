//! Deterministic level-synchronous forced tree for opponent-reply labels.
//!
//! Network evaluation remains a batched Python/PyTorch callback.  State
//! stepping, action grouping, chance expansion, transposition merging, backup,
//! and optional same-level Rayon planning run in Rust.  Parallel plans are
//! collected in input order and merged serially, so thread count cannot change
//! node IDs, edge order, reductions, ties, or serialized outputs.

use super::*;
use std::collections::BTreeMap;

type Placement = Option<(i8, i8, i8, i8, bool)>;
type Action = (Placement, Option<u16>);

#[derive(Clone)]
struct DEdge {
    pick: i32,
    action_idx: u16,
    action: Action,
    children: Vec<(usize, f64)>,
    sampled: bool,
    value: f64,
    stderr: f64,
}

struct DNode {
    state: RustGameState,
    depth: usize,
    crossings: u8,
    edges: Vec<DEdge>,
    value: Option<f64>,
    stderr: f64,
}

#[derive(Clone)]
struct EvalRow {
    legal: Vec<(u16, Placement, Option<u16>)>,
    priors: Vec<f64>,
}

struct PlannedEdge {
    pick: i32,
    action_idx: u16,
    action: Action,
    children: Vec<(RustGameState, u8, f64)>,
    sampled: bool,
}

struct NodePlan {
    node_id: usize,
    edges: Vec<PlannedEdge>,
    pre_reveal_leaves: usize,
    chance_events: usize,
}

fn evaluate_rows_chunk(
    arena: &[DNode],
    node_ids: &[usize],
    evaluator: &Py<PyAny>,
) -> PyResult<Vec<EvalRow>> {
    let k = node_ids.len();
    if k == 0 {
        return Ok(Vec::new());
    }
    let board_size = N_BOARD_CH * OUT_N * OUT_N;
    let mut mb = vec![0.0f32; k * board_size];
    let mut ob = vec![0.0f32; k * board_size];
    let mut flat = vec![0.0f32; k * FLAT_SIZE];
    let mut legals = Vec::with_capacity(k);
    let mut indices = Vec::with_capacity(k);
    for (row, &node_id) in node_ids.iter().enumerate() {
        let state = &arena[node_id].state;
        let actor = state.actor()?;
        state.encode_arrays_into(actor, &mut mb, &mut ob, &mut flat, row)?;
        let legal = state.legal_actions_indexed();
        indices.push(legal.iter().map(|item| item.0 as i64).collect::<Vec<_>>());
        legals.push(legal);
    }

    let gathered = Python::attach(|py| -> PyResult<Vec<Vec<f64>>> {
        let mb_py = Array4::from_shape_vec((k, N_BOARD_CH, OUT_N, OUT_N), mb)
            .expect("denial mb shape")
            .into_pyarray(py);
        let ob_py = Array4::from_shape_vec((k, N_BOARD_CH, OUT_N, OUT_N), ob)
            .expect("denial ob shape")
            .into_pyarray(py);
        let flat_py = Array2::from_shape_vec((k, FLAT_SIZE), flat)
            .expect("denial flat shape")
            .into_pyarray(py);
        let idx_arrays: Vec<_> = indices
            .into_iter()
            .map(|values| values.into_pyarray(py))
            .collect();
        let idx_list = PyList::new(py, idx_arrays)?;
        let result = evaluator
            .bind(py)
            .call1((mb_py, ob_py, flat_py, idx_list))?;
        let tuple = result.downcast::<PyTuple>()?;
        let list_item = tuple.get_item(1)?;
        let list = list_item.downcast::<PyList>()?;
        let mut rows = Vec::with_capacity(k);
        for row in 0..k {
            let array_item = list.get_item(row)?;
            let array = array_item.downcast::<PyArray1<f32>>()?;
            rows.push(
                array
                    .readonly()
                    .as_slice()?
                    .iter()
                    .map(|&value| value as f64)
                    .collect(),
            );
        }
        Ok(rows)
    })?;

    Ok(legals
        .into_iter()
        .zip(gathered)
        .map(|(legal, logits)| EvalRow {
            legal,
            priors: softmax_f64(&logits),
        })
        .collect())
}

fn evaluate_rows(
    arena: &[DNode],
    node_ids: &[usize],
    evaluator: &Py<PyAny>,
    batch_size: usize,
) -> PyResult<Vec<EvalRow>> {
    let mut rows = Vec::with_capacity(node_ids.len());
    for chunk in node_ids.chunks(batch_size.max(1)) {
        rows.extend(evaluate_rows_chunk(arena, chunk, evaluator)?);
    }
    Ok(rows)
}

fn evaluate_leaf_values_chunk(
    arena: &[DNode],
    node_ids: &[usize],
    evaluator: &Py<PyAny>,
) -> PyResult<Vec<f64>> {
    let k = node_ids.len();
    if k == 0 {
        return Ok(Vec::new());
    }
    let board_size = N_BOARD_CH * OUT_N * OUT_N;
    let mut mb = vec![0.0f32; k * board_size];
    let mut ob = vec![0.0f32; k * board_size];
    let mut flat = vec![0.0f32; k * FLAT_SIZE];
    let mut actors = Vec::with_capacity(k);
    let mut indices = Vec::with_capacity(k);
    for (row, &node_id) in node_ids.iter().enumerate() {
        let state = &arena[node_id].state;
        let actor = state.actor()?;
        actors.push(actor);
        state.encode_arrays_into(actor, &mut mb, &mut ob, &mut flat, row)?;
        indices.push(
            state
                .legal_action_indices()
                .into_iter()
                .map(|value| value as i64)
                .collect::<Vec<_>>(),
        );
    }
    let actor_values = Python::attach(|py| -> PyResult<Vec<f64>> {
        let mb_py = Array4::from_shape_vec((k, N_BOARD_CH, OUT_N, OUT_N), mb)
            .expect("denial leaf mb shape")
            .into_pyarray(py);
        let ob_py = Array4::from_shape_vec((k, N_BOARD_CH, OUT_N, OUT_N), ob)
            .expect("denial leaf ob shape")
            .into_pyarray(py);
        let flat_py = Array2::from_shape_vec((k, FLAT_SIZE), flat)
            .expect("denial leaf flat shape")
            .into_pyarray(py);
        let idx_arrays: Vec<_> = indices
            .into_iter()
            .map(|values| values.into_pyarray(py))
            .collect();
        let idx_list = PyList::new(py, idx_arrays)?;
        let result = evaluator
            .bind(py)
            .call1((mb_py, ob_py, flat_py, idx_list))?;
        let tuple = result.downcast::<PyTuple>()?;
        let array_item = tuple.get_item(0)?;
        let array = array_item.downcast::<PyArray1<f32>>()?;
        Ok(array
            .readonly()
            .as_slice()?
            .iter()
            .map(|&value| value as f64)
            .collect())
    })?;
    Ok(actor_values
        .into_iter()
        .zip(actors)
        .map(|(value, actor)| if actor == 0 { value } else { -value })
        .collect())
}

fn evaluate_leaf_values(
    arena: &[DNode],
    node_ids: &[usize],
    evaluator: &Py<PyAny>,
    batch_size: usize,
) -> PyResult<Vec<f64>> {
    let mut values = Vec::with_capacity(node_ids.len());
    for chunk in node_ids.chunks(batch_size.max(1)) {
        values.extend(evaluate_leaf_values_chunk(arena, chunk, evaluator)?);
    }
    Ok(values)
}

fn replace_draw(
    child: &RustGameState,
    pre_deck: &[u16],
    row: &[u16],
) -> Result<RustGameState, String> {
    if row.len() != 4 {
        return Err(format!(
            "chance row must have four tiles, got {}",
            row.len()
        ));
    }
    let mut out = child.cloned();
    let mut remaining = pre_deck.to_vec();
    let mut selected = row.to_vec();
    selected.sort_unstable();
    for value in &selected {
        let Some(index) = remaining.iter().position(|item| item == value) else {
            return Err(format!(
                "chance tile {value} is absent from the pre-deal bag"
            ));
        };
        remaining.remove(index);
    }
    remaining.sort_unstable();
    out.current_row = selected;
    out.deck = remaining;
    Ok(out)
}

fn pre_reveal(child: &RustGameState, pre_deck: &[u16]) -> RustGameState {
    let mut out = child.cloned();
    out.current_row.clear();
    out.deck = pre_deck.to_vec();
    out.deck.sort_unstable();
    out
}

fn plan_node(
    node: &DNode,
    node_id: usize,
    evaluation: &EvalRow,
    root_actions: &[(i32, u16)],
    pick_plies: usize,
    placement_top_k: usize,
    root_actor: u8,
    chance_rows: &[Vec<u16>],
    chance_sampled: bool,
) -> Result<NodePlan, String> {
    let mut selected: Vec<(i32, u16, Placement, Option<u16>)> = Vec::new();
    if node.depth == 0 && !root_actions.is_empty() {
        for &(pick, requested_idx) in root_actions {
            let Some((idx, placement, picked)) =
                evaluation.legal.iter().find(|item| item.0 == requested_idx)
            else {
                return Err(format!("root representative for pick {pick} is not legal"));
            };
            selected.push((pick, *idx, *placement, *picked));
        }
        selected.sort_by_key(|item| (item.0, item.1));
    } else {
        let mut groups: BTreeMap<i32, Vec<(u16, Placement, Option<u16>, f64)>> = BTreeMap::new();
        for (offset, &(idx, placement, picked)) in evaluation.legal.iter().enumerate() {
            groups
                .entry(picked.map_or(-1, |value| value as i32))
                .or_default()
                .push((idx, placement, picked, evaluation.priors[offset]));
        }
        let actor = node.state.actor().map_err(|error| error.to_string())?;
        let top_k = if node.depth == 0 || actor != root_actor {
            placement_top_k.max(1)
        } else {
            1
        };
        for (pick, mut actions) in groups {
            actions.sort_by(|left, right| {
                right
                    .3
                    .total_cmp(&left.3)
                    .then_with(|| left.0.cmp(&right.0))
            });
            for (idx, placement, picked, _prior) in actions.into_iter().take(top_k) {
                selected.push((pick, idx, placement, picked));
            }
        }
    }

    let mut edges = Vec::with_capacity(selected.len());
    let mut pre_reveal_leaves = 0;
    let mut chance_events = 0;
    for (pick, action_idx, placement, picked) in selected {
        let pre_deck = node.state.deck.clone();
        let child = node
            .state
            .step(placement, picked)
            .map_err(|error| error.to_string())?;
        let next_depth = node.depth + 1;
        let dealt = pre_deck.len().saturating_sub(child.deck.len()) == 4;
        let mut children = Vec::new();
        let sampled;
        if dealt && next_depth >= pick_plies {
            children.push((pre_reveal(&child, &pre_deck), node.crossings, 1.0));
            pre_reveal_leaves += 1;
            sampled = false;
        } else if dealt {
            if node.crossings >= 1 {
                return Err("forced denial tree attempted a second interior chance node".into());
            }
            if chance_rows.is_empty() {
                return Err("interior chance expansion requires at least one supplied row".into());
            }
            let weight = 1.0 / chance_rows.len() as f64;
            for row in chance_rows {
                children.push((
                    replace_draw(&child, &pre_deck, row)?,
                    node.crossings + 1,
                    weight,
                ));
            }
            chance_events += 1;
            sampled = chance_sampled;
        } else {
            children.push((child, node.crossings, 1.0));
            sampled = false;
        }
        edges.push(PlannedEdge {
            pick,
            action_idx,
            action: (placement, picked),
            children,
            sampled,
        });
    }
    Ok(NodePlan {
        node_id,
        edges,
        pre_reveal_leaves,
        chance_events,
    })
}

fn better(candidate: f64, current: f64, actor: u8) -> bool {
    if actor == 0 {
        candidate > current
    } else {
        candidate < current
    }
}

fn best_edge_indices(node: &DNode, actor: u8) -> Vec<usize> {
    let mut groups: Vec<(i32, usize)> = Vec::new();
    for (index, edge) in node.edges.iter().enumerate() {
        if let Some((_, current)) = groups.iter_mut().find(|item| item.0 == edge.pick) {
            if better(edge.value, node.edges[*current].value, actor) {
                *current = index;
            }
        } else {
            groups.push((edge.pick, index));
        }
    }
    groups.into_iter().map(|item| item.1).collect()
}

fn backup(
    arena: &mut [DNode],
    pick_plies: usize,
    evaluator: &Py<PyAny>,
    eval_batch_size: usize,
) -> PyResult<()> {
    let terminal_ids: Vec<usize> = arena
        .iter()
        .enumerate()
        .filter_map(|(id, node)| {
            (node.edges.is_empty() && node.state.phase == GAME_OVER).then_some(id)
        })
        .collect();
    for id in terminal_ids {
        arena[id].value = Some(arena[id].state.official_outcome_i8() as f64);
        arena[id].stderr = 0.0;
    }
    let leaf_ids: Vec<usize> = arena
        .iter()
        .enumerate()
        .filter_map(|(id, node)| {
            (node.edges.is_empty() && node.state.phase != GAME_OVER).then_some(id)
        })
        .collect();
    let values = evaluate_leaf_values(arena, &leaf_ids, evaluator, eval_batch_size)?;
    for (id, value) in leaf_ids.into_iter().zip(values) {
        arena[id].value = Some(value);
        arena[id].stderr = 0.0;
    }

    for depth in (0..pick_plies).rev() {
        let node_ids: Vec<usize> = arena
            .iter()
            .enumerate()
            .filter_map(|(id, node)| (node.depth == depth && !node.edges.is_empty()).then_some(id))
            .collect();
        for node_id in node_ids {
            for edge_index in 0..arena[node_id].edges.len() {
                let children = arena[node_id].edges[edge_index].children.clone();
                let values: Vec<f64> = children
                    .iter()
                    .map(|(child, _)| arena[*child].value.expect("child must be backed"))
                    .collect();
                let value = children
                    .iter()
                    .zip(values.iter())
                    .map(|((_, weight), child_value)| weight * child_value)
                    .sum::<f64>();
                let propagated = children
                    .iter()
                    .map(|(child, weight)| (weight * arena[*child].stderr).powi(2))
                    .sum::<f64>();
                let sampled = arena[node_id].edges[edge_index].sampled;
                let sample_variance = if sampled && values.len() > 1 {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let variance = values
                        .iter()
                        .map(|value| (value - mean).powi(2))
                        .sum::<f64>()
                        / (values.len() - 1) as f64;
                    variance / values.len() as f64
                } else {
                    0.0
                };
                arena[node_id].edges[edge_index].value = value;
                arena[node_id].edges[edge_index].stderr =
                    (propagated + sample_variance).max(0.0).sqrt();
            }
            let actor = arena[node_id].state.actor()?;
            let best = best_edge_indices(&arena[node_id], actor);
            let mut selected = best[0];
            for &candidate in &best[1..] {
                if better(
                    arena[node_id].edges[candidate].value,
                    arena[node_id].edges[selected].value,
                    actor,
                ) {
                    selected = candidate;
                }
            }
            arena[node_id].value = Some(arena[node_id].edges[selected].value);
            arena[node_id].stderr = arena[node_id].edges[selected].stderr;
        }
    }
    Ok(())
}

type RootRow = (i32, f64, f64, u16);
type ReplyPickRow = (i32, f64, f64, u16);
type ReplyRow = (i32, RustGameState, Vec<ReplyPickRow>);
type Structure = (usize, usize, usize, usize, usize, usize);

fn run_tree(
    root_state: &RustGameState,
    evaluator: &Py<PyAny>,
    root_actions: &[(i32, u16)],
    chance_rows: &[Vec<u16>],
    chance_sampled: bool,
    pick_plies: usize,
    placement_top_k: usize,
    rayon_threads: usize,
    eval_batch_size: usize,
    placement_root_actor: Option<u8>,
) -> PyResult<(Vec<RootRow>, Vec<ReplyRow>, Structure)> {
    if pick_plies < 1 || placement_top_k < 1 {
        return Err(PyValueError::new_err(
            "pick_plies and placement_top_k must be positive",
        ));
    }
    let root_actor = root_state.actor()?;
    let placement_root_actor = placement_root_actor.unwrap_or(root_actor);
    if placement_root_actor > 1 {
        return Err(PyValueError::new_err("placement_root_actor must be 0 or 1"));
    }
    let mut arena = vec![DNode {
        state: root_state.cloned(),
        depth: 0,
        crossings: 0,
        edges: Vec::new(),
        value: None,
        stderr: 0.0,
    }];
    let mut levels: Vec<Vec<usize>> = vec![vec![0]];
    let mut transpositions: Vec<HashMap<(u128, u8), usize>> = vec![HashMap::new()];
    transpositions[0].insert((advisor_public_signature(root_state), 0), 0);
    let mut chance_events = 0;
    let mut pre_reveal_leaves = 0;
    let pool = if rayon_threads > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(rayon_threads)
                .build()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        )
    } else {
        None
    };

    for depth in 0..pick_plies {
        let node_ids = levels.get(depth).cloned().unwrap_or_default();
        let live: Vec<usize> = node_ids
            .into_iter()
            .filter(|&id| arena[id].state.phase != GAME_OVER && arena[id].edges.is_empty())
            .collect();
        let evaluations = evaluate_rows(&arena, &live, evaluator, eval_batch_size)?;
        let build_plans = || {
            live.par_iter()
                .zip(evaluations.par_iter())
                .map(|(&node_id, evaluation)| {
                    plan_node(
                        &arena[node_id],
                        node_id,
                        evaluation,
                        root_actions,
                        pick_plies,
                        placement_top_k,
                        placement_root_actor,
                        chance_rows,
                        chance_sampled,
                    )
                })
                .collect::<Vec<_>>()
        };
        let plans = if let Some(pool) = &pool {
            pool.install(build_plans)
        } else {
            live.iter()
                .zip(evaluations.iter())
                .map(|(&node_id, evaluation)| {
                    plan_node(
                        &arena[node_id],
                        node_id,
                        evaluation,
                        root_actions,
                        pick_plies,
                        placement_top_k,
                        placement_root_actor,
                        chance_rows,
                        chance_sampled,
                    )
                })
                .collect()
        };
        if levels.len() <= depth + 1 {
            levels.push(Vec::new());
            transpositions.push(HashMap::new());
        }
        for planned in plans {
            let planned = planned.map_err(PyValueError::new_err)?;
            chance_events += planned.chance_events;
            pre_reveal_leaves += planned.pre_reveal_leaves;
            let mut edges = Vec::with_capacity(planned.edges.len());
            for planned_edge in planned.edges {
                let mut children = Vec::with_capacity(planned_edge.children.len());
                for (child_state, crossings, weight) in planned_edge.children {
                    let key = (advisor_public_signature(&child_state), crossings);
                    let child_id = if let Some(&existing) = transpositions[depth + 1].get(&key) {
                        existing
                    } else {
                        let id = arena.len();
                        arena.push(DNode {
                            state: child_state,
                            depth: depth + 1,
                            crossings,
                            edges: Vec::new(),
                            value: None,
                            stderr: 0.0,
                        });
                        transpositions[depth + 1].insert(key, id);
                        levels[depth + 1].push(id);
                        id
                    };
                    children.push((child_id, weight));
                }
                edges.push(DEdge {
                    pick: planned_edge.pick,
                    action_idx: planned_edge.action_idx,
                    action: planned_edge.action,
                    children,
                    sampled: planned_edge.sampled,
                    value: 0.0,
                    stderr: 0.0,
                });
            }
            arena[planned.node_id].edges = edges;
        }
    }
    backup(&mut arena, pick_plies, evaluator, eval_batch_size)?;

    let root_best = best_edge_indices(&arena[0], root_actor);
    let mut root_rows: Vec<RootRow> = root_best
        .iter()
        .map(|&index| {
            let edge = &arena[0].edges[index];
            (edge.pick, edge.value, edge.stderr, edge.action_idx)
        })
        .collect();
    root_rows.sort_by_key(|row| row.0);

    let mut replies = Vec::new();
    for &root_edge_index in &root_best {
        let root_edge = &arena[0].edges[root_edge_index];
        for &(child_id, _weight) in &root_edge.children {
            let child = &arena[child_id];
            if child.state.phase == GAME_OVER || child.edges.is_empty() {
                continue;
            }
            let actor = child.state.actor()?;
            if actor == root_actor {
                continue;
            }
            let mut rows: Vec<ReplyPickRow> = best_edge_indices(child, actor)
                .iter()
                .map(|&index| {
                    let edge = &child.edges[index];
                    (edge.pick, edge.value, edge.stderr, edge.action_idx)
                })
                .collect();
            rows.sort_by_key(|row| row.0);
            replies.push((root_edge.pick, child.state.cloned(), rows));
        }
    }
    replies.sort_by_key(|row| row.0);
    let structure = (
        arena.len(),
        arena.iter().map(|node| node.edges.len()).sum(),
        arena.iter().filter(|node| node.edges.is_empty()).count(),
        arena.iter().map(|node| node.depth).max().unwrap_or(0),
        chance_events,
        pre_reveal_leaves,
    );
    Ok((root_rows, replies, structure))
}

/// Run the forced denial tree in Rust.
///
/// ``root_actions`` entries are ``(pick_id_or_-1, canonical_action_index)``.
/// ``chance_rows`` are computed by the Python oracle once from the public bag;
/// passing them in preserves the registered Python-``random`` CRN exactly.
/// Returns ``(root_rows, reply_rows, structure)`` where values remain in the
/// player-0 frame and each reply row carries its encoded-capable RustGameState.
#[pyfunction]
#[pyo3(signature = (
    state,
    evaluator,
    root_actions,
    chance_rows,
    chance_sampled,
    pick_plies=8,
    placement_top_k=2,
    rayon_threads=1,
    eval_batch_size=512,
    placement_root_actor=None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn denial_forced_tree(
    py: Python<'_>,
    state: &RustGameState,
    evaluator: Bound<'_, PyAny>,
    root_actions: Vec<(i32, u16)>,
    chance_rows: Vec<Vec<u16>>,
    chance_sampled: bool,
    pick_plies: usize,
    placement_top_k: usize,
    rayon_threads: usize,
    eval_batch_size: usize,
    placement_root_actor: Option<u8>,
) -> PyResult<(Vec<RootRow>, Vec<ReplyRow>, Structure)> {
    if state.phase == GAME_OVER {
        return Err(PyValueError::new_err("cannot search a terminal state"));
    }
    let state = state.cloned();
    let evaluator: Py<PyAny> = evaluator.unbind();
    py.detach(move || {
        run_tree(
            &state,
            &evaluator,
            &root_actions,
            &chance_rows,
            chance_sampled,
            pick_plies,
            placement_top_k,
            rayon_threads,
            eval_batch_size,
            placement_root_actor,
        )
    })
}
