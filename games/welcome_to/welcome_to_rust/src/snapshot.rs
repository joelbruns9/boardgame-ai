//! The M0-C state snapshot, emitted in exactly the shape
//! `games/welcome_to/snapshot.py::to_snapshot` produces.
//!
//! ⚠ **Container types are part of the shape.** `sorted(dict.items())` in
//! Python yields *tuples*, and `[1, 2] != (1, 2)` there even element for
//! element, so `plan_turns` and `reshuffle_votes` are emitted as tuples while
//! everything else is a list. The same trap as `Vec<u8>` mapping to `bytes`
//! (RUST_PORT_PLAN.md §5): a container mismatch reads as a data mismatch.
//!
//! ⚠ **Rows are ragged.** Python's sheet rows are 10/11/12 long, not padded to
//! `MAX_STREET_LEN`; the tail of the fixed Rust array is not state and is not
//! emitted.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::constants::{
    Effect, EMPTY, FENCE_SIZES, MAX_ESTATE_SIZE, MAX_STREET_LEN, NUM_STREETS, STREET_SIZES,
};
use crate::game::{Config, Game, Phase, TurnCtx, NO_CARD};
use crate::rng::Rng;
use crate::sheet::Sheet;

/// Must equal `snapshot.SNAPSHOT_VERSION`. A Rust build against an older schema
/// then fails loudly instead of two differently-shaped dictionaries being
/// compared key by key and agreeing.
pub const SNAPSHOT_VERSION: i64 = 1;

fn opt_card(card: i32) -> Option<i32> {
    if card == NO_CARD {
        None
    } else {
        Some(card)
    }
}

fn sheet_dict<'py>(py: Python<'py>, sheet: &Sheet) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);

    let numbers = PyList::empty(py);
    let is_bis = PyList::empty(py);
    let written_turn = PyList::empty(py);
    let fences = PyList::empty(py);
    let top_fences = PyList::empty(py);
    for x in 0..NUM_STREETS {
        let size = STREET_SIZES[x];
        numbers.append(
            (0..size)
                .map(|y| {
                    if sheet.numbers[x][y] == EMPTY {
                        None
                    } else {
                        Some(sheet.numbers[x][y])
                    }
                })
                .collect::<Vec<Option<i32>>>(),
        )?;
        is_bis.append((0..size).map(|y| sheet.is_bis[x][y]).collect::<Vec<bool>>())?;
        written_turn.append(
            (0..size)
                .map(|y| sheet.written_turn[x][y])
                .collect::<Vec<i32>>(),
        )?;
        fences.append(
            (0..FENCE_SIZES[x])
                .map(|j| sheet.fences[x][j])
                .collect::<Vec<bool>>(),
        )?;
        top_fences.append(
            (0..size)
                .map(|y| sheet.top_fences[x][y])
                .collect::<Vec<bool>>(),
        )?;
    }

    out.set_item("numbers", numbers)?;
    out.set_item("is_bis", is_bis)?;
    out.set_item("written_turn", written_turn)?;
    out.set_item("fences", fences)?;
    out.set_item("top_fences", top_fences)?;
    out.set_item("parks", sheet.parks.to_vec())?;
    out.set_item("pools", sheet.pools.to_vec())?;
    out.set_item("estate_marks", sheet.estate_marks.to_vec())?;
    out.set_item("temps", sheet.temps)?;
    out.set_item("bis_marks", sheet.bis_marks)?;
    out.set_item("permits", sheet.permits)?;
    out.set_item("roundabouts", sheet.roundabouts)?;
    Ok(out)
}

fn ctx_dict<'py>(py: Python<'py>, game: &Game) -> PyResult<Bound<'py, PyDict>> {
    let ctx = &game.ctx;
    let out = PyDict::new(py);
    out.set_item("slot", ctx.slot.map(|v| v as i64))?;
    out.set_item("number", ctx.number)?;
    out.set_item("effect", ctx.effect.map(|e| e.as_i64()))?;
    out.set_item(
        "last_house",
        ctx.last_house.map(|(x, y)| vec![x as i64, y as i64]),
    )?;
    out.set_item("built_roundabout", ctx.built_roundabout)?;
    out.set_item("roundabout_declined", ctx.roundabout_declined)?;
    out.set_item("refused", ctx.refused)?;
    out.set_item("plan_slot", ctx.plan_slot.map(|v| v as i64))?;
    out.set_item(
        "pending_sizes",
        ctx.pending_sizes.iter().map(|&s| s as i64).collect::<Vec<i64>>(),
    )?;
    let chosen = PyList::empty(py);
    for &(x, start, size) in ctx.chosen_estates.iter() {
        chosen.append(vec![x as i64, start as i64, size as i64])?;
    }
    out.set_item("chosen_estates", chosen)?;
    Ok(out)
}

/// Pairs sorted by key, matching Python's `sorted(d.items())`.
///
/// A `Vec<(A, B)>` converts to a list of **tuples**, which is what makes this
/// comparable with the Python side element for element.
fn sorted_pairs<T: Copy>(pairs: &[(i64, T)]) -> Vec<(i64, T)> {
    let mut out = pairs.to_vec();
    out.sort_by_key(|(key, _)| *key);
    out
}

/// The whole state, in one order-preserving dictionary.
pub fn snapshot<'py>(py: Python<'py>, game: &Game) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("version", SNAPSHOT_VERSION)?;

    let config = PyDict::new(py);
    config.set_item("players", game.config.players)?;
    config.set_item("advanced", game.config.advanced)?;
    config.set_item("expert", game.config.expert)?;
    config.set_item("solo_rules", game.config.solo_rules)?;
    out.set_item("config", config)?;

    let sheets = PyList::empty(py);
    for sheet in game.sheets.iter() {
        sheets.append(sheet_dict(py, sheet)?)?;
    }
    out.set_item("sheets", sheets)?;
    let public = PyList::empty(py);
    for sheet in game.public_sheets.iter() {
        public.append(sheet_dict(py, sheet)?)?;
    }
    out.set_item("public_sheets", public)?;

    out.set_item("deck", game.deck.clone())?;
    out.set_item("deck_pos", game.deck_pos)?;
    out.set_item("discard", game.discard.clone())?;

    let stack_new = PyList::empty(py);
    for group in game.stack_new.iter() {
        stack_new.append(group.iter().map(|&c| opt_card(c)).collect::<Vec<_>>())?;
    }
    out.set_item("stack_new", stack_new)?;
    let stack_old = PyList::empty(py);
    for group in game.stack_old.iter() {
        stack_old.append(group.iter().map(|&c| opt_card(c)).collect::<Vec<_>>())?;
    }
    out.set_item("stack_old", stack_old)?;

    out.set_item(
        "expert_pending",
        game.expert_pending
            .iter()
            .map(|&c| opt_card(c))
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "plan_ids",
        game.plan_ids.iter().map(|&p| p as i64).collect::<Vec<i64>>(),
    )?;

    let plan_turns: Vec<Vec<(i64, i64)>> = game
        .plan_turns
        .iter()
        .map(|slot| {
            let pairs: Vec<(i64, i64)> =
                slot.iter().map(|&(p, t)| (p as i64, t as i64)).collect();
            sorted_pairs(&pairs)
        })
        .collect();
    out.set_item("plan_turns", plan_turns)?;

    out.set_item("turn", game.turn)?;
    out.set_item("actor", game.actor)?;
    out.set_item("phase", game.phase.as_i64())?;
    out.set_item("ctx", ctx_dict(py, game)?)?;
    out.set_item(
        "turn_choice",
        game.turn_choice
            .iter()
            .map(|&c| opt_card(c))
            .collect::<Vec<_>>(),
    )?;
    out.set_item("reshuffle_next_turn", game.reshuffle_next_turn)?;

    let votes: Vec<(i64, bool)> = game
        .reshuffle_votes
        .iter()
        .map(|&(seat, vote)| (seat as i64, vote))
        .collect();
    out.set_item("reshuffle_votes", sorted_pairs(&votes))?;

    let rng = PyDict::new(py);
    rng.set_item("kind", "portable")?;
    rng.set_item("state", game.rng.state())?;
    out.set_item("rng", rng)?;

    out.set_item("solo_card_drawn", game.solo_card_drawn)?;
    out.set_item("boundary_prepared", game.boundary_prepared)?;
    out.set_item("deck_remaining", game.deck_remaining())?;
    out.set_item("is_terminal", game.is_terminal())?;

    debug_assert_eq!(MAX_ESTATE_SIZE, 6);
    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────────
// Back in — the other direction M0-C asks for
// ──────────────────────────────────────────────────────────────────────────
//
// This is what lets the M1 gate compare *constructed* positions, not only the
// ones random and greedy play reach. Two of the three `isEndOfGame` clauses and
// several boundary cases are rare enough in played games to be effectively
// untested otherwise, and "rare" is where a rules divergence survives.

fn field<'py, T>(raw: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
{
    match raw.get_item(key)? {
        Some(value) => value.extract(),
        None => Err(PyValueError::new_err(format!(
            "snapshot is missing the field {key:?}"
        ))),
    }
}

/// A nested dictionary. `field` cannot fetch one: a `Bound<PyDict>` extraction
/// fails with a cast error rather than a `PyErr`, so it does not satisfy the
/// generic bound.
fn sub_dict<'py>(raw: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyDict>> {
    let value = raw
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("snapshot is missing the field {key:?}")))?;
    value
        .cast_into::<PyDict>()
        .map_err(|_| PyValueError::new_err(format!("snapshot field {key:?} is not a dict")))
}

/// A list of nested dictionaries, for the same reason.
fn sub_dicts<'py>(raw: &Bound<'py, PyDict>, key: &str) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let value = raw
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("snapshot is missing the field {key:?}")))?;
    let mut out = Vec::new();
    for item in value.try_iter()? {
        out.push(item?.cast_into::<PyDict>().map_err(|_| {
            PyValueError::new_err(format!("snapshot field {key:?} holds a non-dict"))
        })?);
    }
    Ok(out)
}

fn rows_into<T: Copy>(
    rows: Vec<Vec<T>>,
    sizes: [usize; NUM_STREETS],
    fill: T,
    what: &str,
) -> PyResult<[[T; MAX_STREET_LEN]; NUM_STREETS]> {
    if rows.len() != NUM_STREETS {
        return Err(PyValueError::new_err(format!(
            "{what} has {} streets, expected {NUM_STREETS}",
            rows.len()
        )));
    }
    let mut out = [[fill; MAX_STREET_LEN]; NUM_STREETS];
    for (x, row) in rows.into_iter().enumerate() {
        if row.len() != sizes[x] {
            return Err(PyValueError::new_err(format!(
                "{what} street {x} has {} boxes, expected {}",
                row.len(),
                sizes[x]
            )));
        }
        for (y, value) in row.into_iter().enumerate() {
            out[x][y] = value;
        }
    }
    Ok(out)
}

fn sheet_from<'py>(raw: &Bound<'py, PyDict>) -> PyResult<Sheet> {
    let numbers: Vec<Vec<Option<i32>>> = field(raw, "numbers")?;
    let numbers: Vec<Vec<i32>> = numbers
        .into_iter()
        .map(|row| row.into_iter().map(|n| n.unwrap_or(EMPTY)).collect())
        .collect();

    let mut sheet = Sheet::new();
    sheet.numbers = rows_into(numbers, STREET_SIZES, EMPTY, "numbers")?;
    sheet.is_bis = rows_into(field(raw, "is_bis")?, STREET_SIZES, false, "is_bis")?;
    sheet.written_turn = rows_into(field(raw, "written_turn")?, STREET_SIZES, -1, "written_turn")?;
    sheet.fences = rows_into(field(raw, "fences")?, FENCE_SIZES, false, "fences")?;
    sheet.top_fences = rows_into(field(raw, "top_fences")?, STREET_SIZES, false, "top_fences")?;

    let parks: Vec<i32> = field(raw, "parks")?;
    let pools: Vec<i32> = field(raw, "pools")?;
    let estate_marks: Vec<i32> = field(raw, "estate_marks")?;
    if parks.len() != NUM_STREETS || pools.len() != NUM_STREETS {
        return Err(PyValueError::new_err("parks/pools must have one entry per street"));
    }
    if estate_marks.len() != MAX_ESTATE_SIZE {
        return Err(PyValueError::new_err("estate_marks must have six rows"));
    }
    sheet.parks.copy_from_slice(&parks);
    sheet.pools.copy_from_slice(&pools);
    sheet.estate_marks.copy_from_slice(&estate_marks);
    sheet.temps = field(raw, "temps")?;
    sheet.bis_marks = field(raw, "bis_marks")?;
    sheet.permits = field(raw, "permits")?;
    sheet.roundabouts = field(raw, "roundabouts")?;
    Ok(sheet)
}

fn ctx_from<'py>(raw: &Bound<'py, PyDict>) -> PyResult<TurnCtx> {
    let effect: Option<i64> = field(raw, "effect")?;
    let effect = match effect {
        None => None,
        Some(value) => Some(Effect::from_i64(value).ok_or_else(|| {
            PyValueError::new_err(format!("{value} is not an Effect"))
        })?),
    };
    let last_house: Option<Vec<usize>> = field(raw, "last_house")?;
    let last_house = match last_house {
        None => None,
        Some(pair) if pair.len() == 2 => Some((pair[0], pair[1])),
        Some(_) => return Err(PyValueError::new_err("last_house must be (street, box)")),
    };
    let chosen: Vec<Vec<usize>> = field(raw, "chosen_estates")?;
    let mut chosen_estates = Vec::with_capacity(chosen.len());
    for estate in chosen {
        if estate.len() != 3 {
            return Err(PyValueError::new_err(
                "an estate is (street, first box, size)",
            ));
        }
        chosen_estates.push((estate[0], estate[1], estate[2]));
    }
    Ok(TurnCtx {
        slot: field(raw, "slot")?,
        number: field(raw, "number")?,
        effect,
        last_house,
        built_roundabout: field(raw, "built_roundabout")?,
        roundabout_declined: field(raw, "roundabout_declined")?,
        refused: field(raw, "refused")?,
        plan_slot: field(raw, "plan_slot")?,
        pending_sizes: field(raw, "pending_sizes")?,
        chosen_estates,
    })
}

fn cards_from(values: Vec<Option<i32>>) -> Vec<i32> {
    values.into_iter().map(|c| c.unwrap_or(NO_CARD)).collect()
}

fn groups_from(values: Vec<Vec<Option<i32>>>, what: &str) -> PyResult<Vec<[i32; 3]>> {
    let mut out = Vec::with_capacity(values.len());
    for group in values {
        if group.len() != 3 {
            return Err(PyValueError::new_err(format!("{what} groups hold three cards")));
        }
        let cards = cards_from(group);
        out.push([cards[0], cards[1], cards[2]]);
    }
    Ok(out)
}

/// Rebuild a state from a snapshot. The inverse of [`snapshot`].
///
/// ⚠ **The RNG must be `portable`.** A `cpython` snapshot carries no state Rust
/// could restore, and silently substituting a fresh generator would make the
/// hand-off look successful while the two engines drew different cards from
/// there on.
pub fn from_snapshot<'py>(raw: &Bound<'py, PyDict>) -> PyResult<Game> {
    let version: i64 = field(raw, "version")?;
    if version != SNAPSHOT_VERSION {
        return Err(PyValueError::new_err(format!(
            "snapshot version {version} != {SNAPSHOT_VERSION}; the schema changed \
             and this state was written by a different build"
        )));
    }

    let config_raw = sub_dict(raw, "config")?;
    let config = Config {
        players: field(&config_raw, "players")?,
        advanced: field(&config_raw, "advanced")?,
        expert: field(&config_raw, "expert")?,
        solo_rules: field(&config_raw, "solo_rules")?,
    };

    let sheets_raw = sub_dicts(raw, "sheets")?;
    let public_raw = sub_dicts(raw, "public_sheets")?;
    let sheets = sheets_raw
        .iter()
        .map(sheet_from)
        .collect::<PyResult<Vec<Sheet>>>()?;
    let public_sheets = public_raw
        .iter()
        .map(sheet_from)
        .collect::<PyResult<Vec<Sheet>>>()?;
    if sheets.len() != config.players || public_sheets.len() != config.players {
        return Err(PyValueError::new_err(
            "the snapshot has a different number of sheets than seats",
        ));
    }

    let plan_ids: Vec<usize> = field(raw, "plan_ids")?;
    if plan_ids.len() != 3 {
        return Err(PyValueError::new_err("three plans are in play, always"));
    }

    let plan_turns_raw: Vec<Vec<(i32, i32)>> = field(raw, "plan_turns")?;
    if plan_turns_raw.len() != 3 {
        return Err(PyValueError::new_err("plan_turns has one entry per plan slot"));
    }
    let plan_turns = [
        plan_turns_raw[0].clone(),
        plan_turns_raw[1].clone(),
        plan_turns_raw[2].clone(),
    ];

    let phase_raw: i64 = field(raw, "phase")?;
    let phase = Phase::from_i64(phase_raw)
        .ok_or_else(|| PyValueError::new_err(format!("{phase_raw} is not a Phase")))?;

    let ctx_raw = sub_dict(raw, "ctx")?;
    let rng_raw = sub_dict(raw, "rng")?;
    let kind: String = field(&rng_raw, "kind")?;
    if kind != "portable" {
        return Err(PyValueError::new_err(format!(
            "the Rust engine cannot adopt a {kind:?} generator; rebuild the game \
             with rng_kind=\"portable\" (RUST_PORT_PLAN.md M0-B)"
        )));
    }
    let rng_state: u64 = field(&rng_raw, "state")?;

    Ok(Game {
        config,
        sheets,
        public_sheets,
        deck: field(raw, "deck")?,
        deck_pos: field(raw, "deck_pos")?,
        discard: field(raw, "discard")?,
        stack_new: groups_from(field(raw, "stack_new")?, "stack_new")?,
        stack_old: groups_from(field(raw, "stack_old")?, "stack_old")?,
        expert_pending: cards_from(field(raw, "expert_pending")?),
        plan_ids: [plan_ids[0], plan_ids[1], plan_ids[2]],
        plan_turns,
        turn: field(raw, "turn")?,
        actor: field(raw, "actor")?,
        phase,
        ctx: ctx_from(&ctx_raw)?,
        turn_choice: cards_from(field(raw, "turn_choice")?),
        reshuffle_next_turn: field(raw, "reshuffle_next_turn")?,
        reshuffle_votes: field(raw, "reshuffle_votes")?,
        rng: Rng::new(rng_state),
        solo_card_drawn: field(raw, "solo_card_drawn")?,
        boundary_prepared: field(raw, "boundary_prepared")?,
    })
}
