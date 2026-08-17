//! Replay-buffer derivation: validated game replay plus packed encoder output.
//!
//! The durable buffer remains JSONL and Python remains the reference
//! implementation.  This module is the production hot path: independent games
//! are replayed and encoded on the explicitly sized Rayon pack pool, then
//! returned as compact byte arrays for NumPy to view without per-token Python
//! objects.

use crate::codec::{decode_action, legal_action_indices};
use crate::data::{card, progress};
use crate::encoder::{encode_into, TokenBuf, FEATURE_COUNTS};
use crate::eval;
use crate::self_play;
use crate::state::{GameState, Phase};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rayon::prelude::*;

const FEATURE_WIDTH: usize = crate::encoder::MAX_FEATURES;

pub(crate) struct DeriveSpec {
    pub state: GameState,
    pub actions: Vec<usize>,
    pub actors: Vec<usize>,
    pub include: Vec<bool>,
    pub chance_log: Vec<(u8, Vec<usize>)>,
    pub winner: Option<usize>,
    pub victory_type: Option<u8>,
    pub scores: Option<(i32, i32)>,
    pub final_digest: Option<String>,
    pub trajectory_digest: Option<String>,
}

#[derive(Default)]
pub(crate) struct DerivedGame {
    token_offsets: Vec<u8>,
    type_ids: Vec<u8>,
    entity_ids: Vec<u8>,
    aux_ids: Vec<u8>,
    features: Vec<u8>,
    row_move_indices: Vec<u8>,
    move_legal_offsets: Vec<u8>,
    move_legal_actions: Vec<u8>,
    move_actors: Vec<u8>,
    ending_age: u8,
    max_absolute_track: i32,
    sixth_science_symbol: bool,
    progress_tokens: usize,
    science_pairs: usize,
    military_tokens_triggered: usize,
    military_gold_pillaged: i32,
    wonders_built: usize,
    wonders_discarded: usize,
    final_conflict_position: i32,
    science_count_0: usize,
    science_count_1: usize,
}

#[inline]
fn push_u32(out: &mut Vec<u8>, value: usize) -> Result<(), String> {
    let value = u32::try_from(value).map_err(|_| "derive offset exceeds u32".to_owned())?;
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn actor(state: &GameState) -> usize {
    state
        .pending_choice
        .as_ref()
        .map_or(state.active_player, |pending| pending.player)
}

fn science_count(state: &GameState, seat: usize) -> usize {
    let mut symbols = [false; 7];
    for &card_id in &state.cities[seat].buildings {
        if let Some(symbol) = card(card_id).science {
            symbols[symbol as usize] = true;
        }
    }
    for &progress_id in &state.cities[seat].progress_tokens {
        if let Some(symbol) = progress(progress_id).science {
            symbols[symbol as usize] = true;
        }
    }
    symbols.into_iter().filter(|present| *present).count()
}

fn derive_one(mut spec: DeriveSpec) -> Result<DerivedGame, String> {
    if spec.actions.len() != spec.actors.len() || spec.actions.len() != spec.include.len() {
        return Err("derive move metadata is not aligned".to_owned());
    }
    let mut out = DerivedGame::default();
    push_u32(&mut out.token_offsets, 0)?;
    push_u32(&mut out.move_legal_offsets, 0)?;
    let mut token_count = 0usize;
    let mut legal_count = 0usize;
    let mut actual_chance = Vec::new();
    let mut token_buf = TokenBuf::new();
    let mut trajectory = crate::digest::TrajectoryDigest::new();
    out.max_absolute_track = spec.state.conflict_position.abs();

    for move_index in 0..spec.actions.len() {
        if spec.state.phase == Phase::Complete {
            return Err(format!("move {move_index}: game already complete"));
        }
        out.max_absolute_track = out
            .max_absolute_track
            .max(spec.state.conflict_position.abs());
        trajectory.update(&spec.state);
        let current_actor = actor(&spec.state);
        if current_actor != spec.actors[move_index] {
            return Err(format!(
                "move {move_index}: actor {current_actor} != recorded {}",
                spec.actors[move_index]
            ));
        }
        out.move_actors.push(current_actor as u8);
        let legal = legal_action_indices(&spec.state);
        for &action in &legal {
            let action = i16::try_from(action)
                .map_err(|_| format!("move {move_index}: action {action} exceeds int16"))?;
            out.move_legal_actions
                .extend_from_slice(&action.to_le_bytes());
        }
        legal_count += legal.len();
        push_u32(&mut out.move_legal_offsets, legal_count)?;

        let action = spec.actions[move_index];
        if !legal.contains(&action) {
            return Err(format!("move {move_index}: illegal action index {action}"));
        }
        let events = self_play::actual_chance_outcomes(&spec.state, action, move_index)
            .map_err(|error| format!("move {move_index}: {error}"))?;
        actual_chance.extend(
            events
                .into_iter()
                .map(|event| (event.kind as u8, event.outcome)),
        );

        if spec.include[move_index] {
            encode_into(&spec.state, &mut token_buf);
            let tokens = token_buf.tokens();
            for token in tokens {
                if token.features.len() != FEATURE_COUNTS[token.type_id] {
                    return Err(format!("move {move_index}: token feature width mismatch"));
                }
                out.type_ids.push(token.type_id as u8);
                let entity = i16::try_from(token.entity_id)
                    .map_err(|_| format!("move {move_index}: entity id overflow"))?;
                let aux = i16::try_from(token.aux_id + 1)
                    .map_err(|_| format!("move {move_index}: aux id overflow"))?;
                out.entity_ids.extend_from_slice(&entity.to_le_bytes());
                out.aux_ids.extend_from_slice(&aux.to_le_bytes());
                for column in 0..FEATURE_WIDTH {
                    let value = token.features.get(column).copied().unwrap_or(0.0);
                    out.features.extend_from_slice(&value.to_le_bytes());
                }
            }
            token_count += tokens.len();
            push_u32(&mut out.token_offsets, token_count)?;
            push_u32(&mut out.row_move_indices, move_index)?;
        }

        let decoded = decode_action(&spec.state, action);
        spec.state.apply_action(&decoded);
    }

    if actual_chance != spec.chance_log {
        return Err(format!(
            "chance log differs: generated {} events, recorded {}",
            actual_chance.len(),
            spec.chance_log.len()
        ));
    }
    if spec.state.phase != Phase::Complete {
        return Err("replayed game did not complete".to_owned());
    }
    let final_digest = crate::digest::state_digest(&spec.state);
    trajectory.update(&spec.state);
    let trajectory_digest = trajectory.finish();
    if spec
        .final_digest
        .as_ref()
        .is_some_and(|expected| expected != &final_digest)
    {
        return Err(format!(
            "final digest {final_digest} differs from recorded {}",
            spec.final_digest.as_deref().unwrap_or_default()
        ));
    }
    if spec
        .trajectory_digest
        .as_ref()
        .is_some_and(|expected| expected != &trajectory_digest)
    {
        return Err(format!(
            "trajectory digest {trajectory_digest} differs from recorded {}",
            spec.trajectory_digest.as_deref().unwrap_or_default()
        ));
    }
    if spec.state.winner != spec.winner
        || spec.state.victory_type.map(|victory| victory as u8) != spec.victory_type
        || spec.state.final_scores != spec.scores
    {
        return Err("Rust replay final result differs from recorded result".to_owned());
    }

    out.max_absolute_track = out
        .max_absolute_track
        .max(spec.state.conflict_position.abs());
    out.ending_age = spec.state.age;
    out.sixth_science_symbol = (0..2).any(|seat| science_count(&spec.state, seat) >= 6);
    out.progress_tokens = spec
        .state
        .cities
        .iter()
        .map(|city| city.progress_tokens.len())
        .sum();
    out.science_pairs = spec
        .state
        .cities
        .iter()
        .map(|city| city.claimed_science_pairs.len())
        .sum();
    out.military_tokens_triggered =
        4usize.saturating_sub(spec.state.military_tokens_remaining.len());
    out.military_gold_pillaged = 14
        - spec
            .state
            .military_tokens_remaining
            .iter()
            .map(|(_, value)| *value)
            .sum::<i32>();
    out.wonders_built = spec
        .state
        .cities
        .iter()
        .map(|city| city.built_wonders.len())
        .sum();
    out.wonders_discarded = spec.state.retired_wonders.len();
    out.final_conflict_position = spec.state.conflict_position;
    out.science_count_0 = science_count(&spec.state, 0);
    out.science_count_1 = science_count(&spec.state, 1);
    Ok(out)
}

pub(crate) fn derive_batch(specs: Vec<DeriveSpec>) -> Result<Vec<DerivedGame>, String> {
    if eval::pack_threads() <= 1 {
        return specs.into_iter().map(derive_one).collect();
    }
    eval::with_pack_pool(|| {
        specs
            .into_par_iter()
            .enumerate()
            .map(|(game_index, spec)| {
                derive_one(spec).map_err(|error| format!("game {game_index}: {error}"))
            })
            .collect()
    })
}

fn packed_bytes<'py>(py: Python<'py>, source: &[u8]) -> Bound<'py, PyBytes> {
    PyBytes::new(py, source)
}

pub(crate) fn to_python(py: Python<'_>, games: Vec<DerivedGame>) -> PyResult<Vec<Py<PyDict>>> {
    games
        .into_iter()
        .map(|game| {
            let payload = PyDict::new(py);
            payload.set_item("feature_width", FEATURE_WIDTH)?;
            payload.set_item("token_offsets", packed_bytes(py, &game.token_offsets))?;
            payload.set_item("type_ids", packed_bytes(py, &game.type_ids))?;
            payload.set_item("entity_ids", packed_bytes(py, &game.entity_ids))?;
            payload.set_item("aux_ids", packed_bytes(py, &game.aux_ids))?;
            payload.set_item("features_f64", packed_bytes(py, &game.features))?;
            payload.set_item("row_move_indices", packed_bytes(py, &game.row_move_indices))?;
            payload.set_item(
                "move_legal_offsets",
                packed_bytes(py, &game.move_legal_offsets),
            )?;
            payload.set_item(
                "move_legal_actions",
                packed_bytes(py, &game.move_legal_actions),
            )?;
            payload.set_item("move_actors", packed_bytes(py, &game.move_actors))?;
            let stats = PyDict::new(py);
            stats.set_item("ending_age", game.ending_age)?;
            stats.set_item("max_absolute_track", game.max_absolute_track)?;
            stats.set_item("sixth_science_symbol", game.sixth_science_symbol)?;
            stats.set_item("progress_tokens", game.progress_tokens)?;
            stats.set_item("science_pairs", game.science_pairs)?;
            stats.set_item("military_tokens_triggered", game.military_tokens_triggered)?;
            stats.set_item("military_gold_pillaged", game.military_gold_pillaged)?;
            stats.set_item("wonders_built", game.wonders_built)?;
            stats.set_item("wonders_discarded", game.wonders_discarded)?;
            stats.set_item("final_conflict_position", game.final_conflict_position)?;
            stats.set_item("science_count_0", game.science_count_0)?;
            stats.set_item("science_count_1", game.science_count_1)?;
            payload.set_item("stats", stats)?;
            Ok(payload.unbind())
        })
        .collect::<PyResult<Vec<_>>>()
        .map_err(|error| PyValueError::new_err(error.to_string()))
}
