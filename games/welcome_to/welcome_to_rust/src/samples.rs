//! Direct S2 training-row capture and asynchronous durable shard writing.
//!
//! A searched root is encoded while its real Rust state is live. Terminal
//! targets are attached once the game ends; no Python game replay is required
//! on the production path. Each shard also retains the compact JSON trajectory
//! as the long-lived audit/re-derivation source.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::constants::{EMPTY, STREET_SIZES};
use crate::encoder::{self, EncodedState};
use crate::game::{EngineError, Game};
use crate::macro_codec;
use crate::tables;
use crate::{to_py, RustGameState};

pub const TRAINING_SHARD_VERSION: u16 = 1;
pub const GLOBAL_TARGET_NAMES: [&str; 9] = [
    "turns_left",
    "rank_p_0",
    "rank_mask_0",
    "rank_p_1",
    "rank_mask_1",
    "rank_p_2",
    "rank_mask_2",
    "rank_p_3",
    "rank_mask_3",
];
pub const PER_SEAT_TARGET_NAMES: [&str; 20] = [
    "score",
    "permits",
    "houses",
    "capacity_left",
    "plans_completed",
    "score_parks",
    "score_pools",
    "score_estates",
    "score_plans",
    "score_temp",
    "score_bis",
    "score_permits",
    "score_roundabouts",
    "turns_to_plan_0",
    "turns_to_plan_0_mask",
    "turns_to_plan_1",
    "turns_to_plan_1_mask",
    "turns_to_plan_2",
    "turns_to_plan_2_mask",
    "seat_valid",
];
pub const GLOBAL_TARGET_COUNT: usize = GLOBAL_TARGET_NAMES.len();
pub const PER_SEAT_TARGET_COUNT: usize = PER_SEAT_TARGET_NAMES.len();
pub const TARGET_FLOAT_COUNT: usize =
    GLOBAL_TARGET_COUNT + encoder::MAX_SEATS * PER_SEAT_TARGET_COUNT;
const MAGIC: &[u8; 8] = b"WTSHRD01";
const SCORE_SCALE: f64 = 80.0;
const PERMIT_SCALE: f64 = 3.0;
const TURN_SCALE: f64 = 25.0;
const BOX_SCALE: f64 = 33.0;
const PLAN_SCALE: f64 = 3.0;

struct RootSample {
    encoded: EncodedState,
    legal: Vec<u8>,
    action: u16,
    actor: u8,
    turn: i32,
    seat_order: Vec<usize>,
    policy: Vec<(u16, u32)>,
}

struct FinalSample {
    root: RootSample,
    targets: [f32; TARGET_FLOAT_COUNT],
}

struct CompletedGame {
    seed: u64,
    trajectory_json: String,
    samples: Vec<FinalSample>,
}

struct Outcome {
    score: i32,
    components: crate::sheet::SheetScore,
    permits: i32,
    houses: i32,
    capacity_left: i32,
    plan_turns: [Option<i32>; 3],
    plans_completed: i32,
    final_turn: i32,
    num_seats: usize,
    rank_distribution: [f32; encoder::MAX_SEATS],
}

fn ratio(value: i32, scale: f64) -> f32 {
    (value as f64 / scale) as f32
}

fn outcomes(game: &Game) -> Result<Vec<Outcome>, EngineError> {
    if !game.is_terminal() {
        return Err(EngineError::Invalid(
            "training outcomes require a terminal game".into(),
        ));
    }
    let seats = game.config.players;
    let scores = game.scores(None);
    let keys: Vec<(i32, [i32; 7])> = (0..seats)
        .map(|seat| (scores[seat], game.sheets[seat].tiebreak_key()))
        .collect();
    let mut order: Vec<usize> = (0..seats).collect();
    order.sort_by(|&left, &right| keys[right].cmp(&keys[left]).then(left.cmp(&right)));
    let mut ranks = vec![[0.0f32; encoder::MAX_SEATS]; seats];
    let mut lo = 0usize;
    while lo < seats {
        let mut hi = lo + 1;
        while hi < seats && keys[order[hi]] == keys[order[lo]] {
            hi += 1;
        }
        let share = (1.0f64 / (hi - lo) as f64) as f32;
        for &seat in &order[lo..hi] {
            ranks[seat][lo..hi].fill(share);
        }
        lo = hi;
    }

    let mut result = Vec::with_capacity(seats);
    for player in 0..seats {
        let sheet = &game.sheets[player];
        let plan_turns = std::array::from_fn(|slot| {
            game.plan_turns[slot]
                .iter()
                .find(|(seat, _)| *seat == player as i32)
                .map(|(_, turn)| *turn)
        });
        let houses = STREET_SIZES
            .iter()
            .enumerate()
            .map(|(street, &size)| {
                game.sheets[player].numbers[street][..size]
                    .iter()
                    .filter(|&&number| number != EMPTY)
                    .count() as i32
            })
            .sum();
        result.push(Outcome {
            score: scores[player],
            components: game.score_breakdown(player, None),
            permits: sheet.permits,
            houses,
            capacity_left: sheet.placement_capacity().iter().sum(),
            plans_completed: plan_turns.iter().flatten().count() as i32,
            plan_turns,
            final_turn: game.turn,
            num_seats: seats,
            rank_distribution: ranks[player],
        });
    }
    Ok(result)
}

fn seat_targets(outcome: &Outcome, turn: i32) -> [f32; PER_SEAT_TARGET_COUNT] {
    let score = outcome.components;
    let mut out = [0.0f32; PER_SEAT_TARGET_COUNT];
    out[0] = ratio(outcome.score, SCORE_SCALE);
    out[1] = ratio(outcome.permits, PERMIT_SCALE);
    out[2] = ratio(outcome.houses, BOX_SCALE);
    out[3] = ratio(outcome.capacity_left, BOX_SCALE);
    out[4] = ratio(outcome.plans_completed, PLAN_SCALE);
    out[5] = ratio(score.parks, SCORE_SCALE);
    out[6] = ratio(score.pools, SCORE_SCALE);
    out[7] = ratio(score.estates, SCORE_SCALE);
    out[8] = ratio(score.plans, SCORE_SCALE);
    out[9] = ratio(score.temp, SCORE_SCALE);
    out[10] = ratio(score.bis, SCORE_SCALE);
    out[11] = ratio(score.permits, SCORE_SCALE);
    out[12] = ratio(score.roundabouts, SCORE_SCALE);
    for slot in 0..3 {
        let value = 13 + slot * 2;
        match outcome.plan_turns[slot] {
            Some(completed) => {
                out[value] = ratio((completed - turn).max(0), TURN_SCALE);
                out[value + 1] = 1.0;
            }
            None => {
                out[value] = -1.0;
                out[value + 1] = 0.0;
            }
        }
    }
    out[19] = 1.0;
    out
}

fn sample_targets(
    outcomes: &[Outcome],
    seat_order: &[usize],
    turn: i32,
) -> Result<[f32; TARGET_FLOAT_COUNT], EngineError> {
    if seat_order.is_empty() || seat_order.len() > encoder::MAX_SEATS {
        return Err(EngineError::Invalid("invalid training seat axis".into()));
    }
    let viewer = &outcomes[seat_order[0]];
    let mut out = [0.0f32; TARGET_FLOAT_COUNT];
    out[0] = ratio((viewer.final_turn - turn).max(0), TURN_SCALE);
    for rank in 0..encoder::MAX_SEATS {
        out[1 + rank * 2] = viewer.rank_distribution[rank];
        out[2 + rank * 2] = (rank < viewer.num_seats) as u8 as f32;
    }
    for encoded_seat in 0..encoder::MAX_SEATS {
        let start = GLOBAL_TARGET_COUNT + encoded_seat * PER_SEAT_TARGET_COUNT;
        if encoded_seat < seat_order.len() {
            out[start..start + PER_SEAT_TARGET_COUNT]
                .copy_from_slice(&seat_targets(&outcomes[seat_order[encoded_seat]], turn));
        } else {
            for slot in 0..3 {
                out[start + 13 + slot * 2] = -1.0;
            }
        }
    }
    Ok(out)
}

/// Per-live-game Rust owner of root encodings until terminal targets are known.
#[pyclass(module = "welcome_to_rust")]
pub struct RustTrainingCapture {
    seed: u64,
    roots: Vec<RootSample>,
    finished: bool,
}

#[pymethods]
impl RustTrainingCapture {
    #[new]
    fn new(seed: u64) -> Self {
        Self {
            seed,
            roots: Vec::new(),
            finished: false,
        }
    }

    fn capture(
        &mut self,
        state: &RustGameState,
        actions: Vec<usize>,
        visits: Vec<f64>,
        action: usize,
        prune_roundabout_pass: bool,
    ) -> PyResult<()> {
        if self.finished {
            return Err(PyRuntimeError::new_err(
                "training capture is already finished",
            ));
        }
        if actions.is_empty() || actions.len() != visits.len() {
            return Err(PyValueError::new_err(
                "training actions and visits must be non-empty and aligned",
            ));
        }
        let search_actions =
            macro_codec::search_legal_macros(&state.inner, prune_roundabout_pass).map_err(to_py)?;
        if actions != search_actions {
            return Err(PyValueError::new_err(
                "training policy actions disagree with search legality",
            ));
        }
        let legal_actions = macro_codec::legal_macros(&state.inner).map_err(to_py)?;
        let legal_set: HashSet<usize> = legal_actions.iter().copied().collect();
        if !legal_set.contains(&action) {
            return Err(PyValueError::new_err("training action is not legal"));
        }
        let mut seen = HashSet::with_capacity(actions.len());
        let mut policy = Vec::with_capacity(actions.len());
        let mut mass = 0u64;
        for (macro_action, visit) in actions.into_iter().zip(visits) {
            if macro_action >= macro_codec::NUM_MACRO_ACTIONS
                || !legal_set.contains(&macro_action)
                || !seen.insert(macro_action)
                || !visit.is_finite()
                || visit < 0.0
                || visit.fract() != 0.0
                || visit > u32::MAX as f64
            {
                return Err(PyValueError::new_err(
                    "training policy contains an invalid action or visit count",
                ));
            }
            let count = visit as u32;
            mass += u64::from(count);
            policy.push((macro_action as u16, count));
        }
        if mass == 0 {
            return Err(PyValueError::new_err(
                "training policy needs positive visit mass",
            ));
        }
        let encoded = encoder::encode_state(&state.inner, state.inner.actor).map_err(to_py)?;
        let mut legal = vec![0u8; macro_codec::NUM_MACRO_ACTIONS];
        for macro_action in legal_actions {
            legal[macro_action] = 1;
        }
        let actor = state.inner.actor;
        let seat_order = (0..state.inner.config.players)
            .map(|offset| (actor + offset) % state.inner.config.players)
            .collect();
        self.roots.push(RootSample {
            encoded,
            legal,
            action: action as u16,
            actor: actor as u8,
            turn: state.inner.turn,
            seat_order,
            policy,
        });
        Ok(())
    }

    fn finish(
        &mut self,
        state: &RustGameState,
        trajectory_json: String,
    ) -> PyResult<RustTrainingGame> {
        if self.finished {
            return Err(PyRuntimeError::new_err(
                "training capture is already finished",
            ));
        }
        if trajectory_json.is_empty() {
            return Err(PyValueError::new_err("trajectory JSON cannot be empty"));
        }
        let terminal = outcomes(&state.inner).map_err(to_py)?;
        let roots = std::mem::take(&mut self.roots);
        let samples = roots
            .into_iter()
            .map(|root| {
                let targets = sample_targets(&terminal, &root.seat_order, root.turn)?;
                Ok(FinalSample { root, targets })
            })
            .collect::<Result<Vec<_>, EngineError>>()
            .map_err(to_py)?;
        self.finished = true;
        Ok(RustTrainingGame {
            inner: Some(CompletedGame {
                seed: self.seed,
                trajectory_json,
                samples,
            }),
        })
    }

    #[getter]
    fn samples(&self) -> usize {
        self.roots.len()
    }
}

#[pyclass(module = "welcome_to_rust")]
pub struct RustTrainingGame {
    inner: Option<CompletedGame>,
}

#[pymethods]
impl RustTrainingGame {
    #[getter]
    fn seed(&self) -> PyResult<u64> {
        self.inner
            .as_ref()
            .map(|game| game.seed)
            .ok_or_else(|| PyRuntimeError::new_err("training game was already submitted"))
    }

    #[getter]
    fn samples(&self) -> PyResult<usize> {
        self.inner
            .as_ref()
            .map(|game| game.samples.len())
            .ok_or_else(|| PyRuntimeError::new_err("training game was already submitted"))
    }
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_f32s(out: &mut Vec<u8>, values: &[f32]) {
    out.reserve(std::mem::size_of_val(values));
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn serialize_game(game: &CompletedGame) -> Result<Vec<u8>, String> {
    let json = game.trajectory_json.as_bytes();
    let mut out = Vec::new();
    push_u64(&mut out, game.seed);
    push_u32(
        &mut out,
        u32::try_from(json.len()).map_err(|_| "trajectory JSON exceeds u32".to_string())?,
    );
    out.extend_from_slice(json);
    push_u32(
        &mut out,
        u32::try_from(game.samples.len()).map_err(|_| "sample count exceeds u32".to_string())?,
    );
    for sample in &game.samples {
        push_f32s(&mut out, &sample.root.encoded.sheet_planes);
        push_f32s(&mut out, &sample.root.encoded.sheet_scalars);
        push_f32s(&mut out, &sample.root.encoded.viewer_plane);
        push_f32s(&mut out, &sample.root.encoded.global_scalars);
        out.extend_from_slice(&sample.root.legal);
        push_u16(&mut out, sample.root.action);
        out.push(sample.root.actor);
        push_i32(&mut out, sample.root.turn);
        push_u16(
            &mut out,
            u16::try_from(sample.root.policy.len())
                .map_err(|_| "policy support exceeds u16".to_string())?,
        );
        for &(action, visits) in &sample.root.policy {
            push_u16(&mut out, action);
            push_u32(&mut out, visits);
        }
        push_f32s(&mut out, &sample.targets);
    }
    Ok(out)
}

fn write_shard(path: &Path, games: &[CompletedGame]) -> Result<(), String> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "training shard path has no UTF-8 filename".to_string())?;
    let temporary = path.with_file_name(format!("{file_name}.{}.tmp", std::process::id()));
    let result = (|| {
        let file = File::create(&temporary).map_err(|error| error.to_string())?;
        let mut writer = BufWriter::new(file);
        writer.write_all(MAGIC).map_err(|error| error.to_string())?;
        writer
            .write_all(&TRAINING_SHARD_VERSION.to_le_bytes())
            .map_err(|error| error.to_string())?;
        writer
            .write_all(&(GLOBAL_TARGET_COUNT as u16).to_le_bytes())
            .map_err(|error| error.to_string())?;
        writer
            .write_all(&(PER_SEAT_TARGET_COUNT as u16).to_le_bytes())
            .map_err(|error| error.to_string())?;
        writer
            .write_all(&(encoder::MAX_SEATS as u16).to_le_bytes())
            .map_err(|error| error.to_string())?;
        writer
            .write_all(&tables::table_signature().to_le_bytes())
            .map_err(|error| error.to_string())?;
        writer
            .write_all(
                &u32::try_from(games.len())
                    .map_err(|_| "game count exceeds u32".to_string())?
                    .to_le_bytes(),
            )
            .map_err(|error| error.to_string())?;
        for game in games {
            let payload = serialize_game(game)?;
            writer
                .write_all(&(payload.len() as u64).to_le_bytes())
                .and_then(|_| writer.write_all(&payload))
                .map_err(|error| error.to_string())?;
        }
        writer.flush().map_err(|error| error.to_string())?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| error.to_string())?;
        drop(writer);
        fs::rename(&temporary, path).map_err(|error| error.to_string())?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn read_u16(reader: &mut impl Read) -> Result<u16, String> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(reader: &mut impl Read) -> Result<u32, String> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64, String> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_shard_seeds(path: &Path) -> Result<Vec<u64>, String> {
    let mut reader = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
    let file_length = reader
        .get_ref()
        .metadata()
        .map_err(|e| e.to_string())?
        .len();
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != MAGIC
        || read_u16(&mut reader)? != TRAINING_SHARD_VERSION
        || read_u16(&mut reader)? as usize != GLOBAL_TARGET_COUNT
        || read_u16(&mut reader)? as usize != PER_SEAT_TARGET_COUNT
        || read_u16(&mut reader)? as usize != encoder::MAX_SEATS
        || read_u64(&mut reader)? != tables::table_signature()
    {
        return Err("training shard ABI or table signature mismatch".into());
    }
    let count = read_u32(&mut reader)? as usize;
    let mut seeds = Vec::with_capacity(count);
    for _ in 0..count {
        let length = read_u64(&mut reader)?;
        let start = reader.stream_position().map_err(|e| e.to_string())?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| "training shard record length overflowed".to_string())?;
        if length < 8 || end > file_length {
            return Err("training shard contains a truncated game record".into());
        }
        seeds.push(read_u64(&mut reader)?);
        reader
            .seek(SeekFrom::Start(end))
            .map_err(|e| e.to_string())?;
    }
    if reader.stream_position().map_err(|e| e.to_string())? != file_length {
        return Err("training shard has trailing bytes".into());
    }
    Ok(seeds)
}

enum WriterCommand {
    Add(CompletedGame),
    Flush(mpsc::Sender<Result<(), String>>),
    Close(mpsc::Sender<Result<(), String>>),
}

fn shard_path(prefix: &Path, index: usize) -> Result<PathBuf, String> {
    let stem = prefix
        .file_stem()
        .or_else(|| prefix.file_name())
        .and_then(|value| value.to_str())
        .ok_or_else(|| "training shard prefix has no UTF-8 stem".to_string())?;
    Ok(prefix.with_file_name(format!("{stem}.part-{index:06}.wts")))
}

fn existing_shards(prefix: &Path) -> Result<Vec<(usize, PathBuf)>, String> {
    let stem = prefix
        .file_stem()
        .or_else(|| prefix.file_name())
        .and_then(|value| value.to_str())
        .ok_or_else(|| "training shard prefix has no UTF-8 stem".to_string())?;
    let start = format!("{stem}.part-");
    let parent = prefix.parent().unwrap_or_else(|| Path::new("."));
    let mut shards = Vec::new();
    for entry in fs::read_dir(parent).map_err(|error| error.to_string())? {
        let path = entry.map_err(|error| error.to_string())?.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        let Some(index) = name
            .strip_prefix(&start)
            .and_then(|value| value.strip_suffix(".wts"))
            .and_then(|value| value.parse::<usize>().ok())
        else {
            continue;
        };
        shards.push((index, path));
    }
    shards.sort_by_key(|(index, _)| *index);
    if shards.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err("training shard indices are not unique".into());
    }
    Ok(shards)
}

fn writer_loop(
    prefix: PathBuf,
    shard_games: usize,
    mut next_shard: usize,
    receiver: mpsc::Receiver<WriterCommand>,
    failure: Arc<Mutex<Option<String>>>,
) {
    let mut buffer = Vec::with_capacity(shard_games);
    let flush = |buffer: &mut Vec<CompletedGame>, next_shard: &mut usize| {
        if buffer.is_empty() {
            return Ok(());
        }
        let destination = shard_path(&prefix, *next_shard)?;
        write_shard(&destination, buffer)?;
        buffer.clear();
        *next_shard += 1;
        Ok(())
    };
    while let Ok(command) = receiver.recv() {
        match command {
            WriterCommand::Add(game) => {
                if failure
                    .lock()
                    .expect("writer failure lock poisoned")
                    .is_some()
                {
                    continue;
                }
                buffer.push(game);
                if buffer.len() >= shard_games {
                    if let Err(error) = flush(&mut buffer, &mut next_shard) {
                        *failure.lock().expect("writer failure lock poisoned") = Some(error);
                    }
                }
            }
            WriterCommand::Flush(reply) => {
                let result = failure
                    .lock()
                    .expect("writer failure lock poisoned")
                    .clone()
                    .map_or_else(|| flush(&mut buffer, &mut next_shard), |error| Err(error));
                if let Err(error) = &result {
                    *failure.lock().expect("writer failure lock poisoned") = Some(error.clone());
                }
                let _ = reply.send(result);
            }
            WriterCommand::Close(reply) => {
                let result = failure
                    .lock()
                    .expect("writer failure lock poisoned")
                    .clone()
                    .map_or_else(|| flush(&mut buffer, &mut next_shard), |error| Err(error));
                let _ = reply.send(result);
                return;
            }
        }
    }
}

/// Bounded producer queue and background atomic shard writer.
#[pyclass(module = "welcome_to_rust")]
pub struct RustSampleShardWriter {
    sender: Option<mpsc::SyncSender<WriterCommand>>,
    thread: Option<std::thread::JoinHandle<()>>,
    failure: Arc<Mutex<Option<String>>>,
    admitted: Arc<Mutex<HashSet<u64>>>,
}

#[pymethods]
impl RustSampleShardWriter {
    #[new]
    #[pyo3(signature = (path, shard_games=25, queue_games=100))]
    fn new(path: PathBuf, shard_games: usize, queue_games: usize) -> PyResult<Self> {
        if shard_games == 0 || queue_games == 0 {
            return Err(PyValueError::new_err(
                "shard_games and queue_games must be positive",
            ));
        }
        let prefix = if path.is_dir() {
            path.join("trajectories.jsonl")
        } else {
            path
        };
        if let Some(parent) = prefix.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        }
        let shards = existing_shards(&prefix).map_err(PyRuntimeError::new_err)?;
        let mut existing = HashSet::new();
        for (_, candidate) in &shards {
            for seed in read_shard_seeds(candidate).map_err(PyRuntimeError::new_err)? {
                if !existing.insert(seed) {
                    return Err(PyValueError::new_err(format!(
                        "training shards repeat seed {seed}"
                    )));
                }
            }
        }
        let next_shard = shards
            .last()
            .map_or(0, |(index, _)| index.saturating_add(1));
        let failure = Arc::new(Mutex::new(None));
        let (sender, receiver) = mpsc::sync_channel(queue_games);
        let thread_failure = Arc::clone(&failure);
        let thread = std::thread::spawn(move || {
            writer_loop(prefix, shard_games, next_shard, receiver, thread_failure)
        });
        Ok(Self {
            sender: Some(sender),
            thread: Some(thread),
            failure,
            admitted: Arc::new(Mutex::new(existing)),
        })
    }

    fn add(&self, py: Python<'_>, game: Py<RustTrainingGame>) -> PyResult<()> {
        self.check_failure()?;
        let mut game = game.bind(py).borrow_mut();
        let seed = game.seed()?;
        {
            let mut admitted = self.admitted.lock().expect("admitted lock poisoned");
            if !admitted.insert(seed) {
                return Err(PyValueError::new_err(format!(
                    "training seed {seed} was already submitted"
                )));
            }
        }
        let completed = game
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("training game was already submitted"))?;
        drop(game);
        let sender = self
            .sender
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("sample writer is closed"))?
            .clone();
        py.detach(move || sender.send(WriterCommand::Add(completed)))
            .map_err(|_| PyRuntimeError::new_err("sample writer thread stopped"))
    }

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        self.check_failure()?;
        let (reply, response) = mpsc::channel();
        let sender = self
            .sender
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("sample writer is closed"))?
            .clone();
        py.detach(move || {
            sender
                .send(WriterCommand::Flush(reply))
                .map_err(|_| "sample writer thread stopped".to_string())?;
            response
                .recv()
                .map_err(|_| "sample writer dropped flush response".to_string())?
        })
        .map_err(PyRuntimeError::new_err)
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(sender) = self.sender.take() else {
            return self.check_failure();
        };
        let (reply, response) = mpsc::channel();
        let result = py.detach(move || {
            sender
                .send(WriterCommand::Close(reply))
                .map_err(|_| "sample writer thread stopped".to_string())?;
            drop(sender);
            response
                .recv()
                .map_err(|_| "sample writer dropped close response".to_string())?
        });
        if let Some(thread) = self.thread.take() {
            py.detach(move || thread.join())
                .map_err(|_| PyRuntimeError::new_err("sample writer thread panicked"))?;
        }
        result.map_err(PyRuntimeError::new_err)
    }

    #[getter]
    fn completed_seeds(&self) -> Vec<u64> {
        let mut seeds: Vec<u64> = self
            .admitted
            .lock()
            .expect("admitted lock poisoned")
            .iter()
            .copied()
            .collect();
        seeds.sort_unstable();
        seeds
    }
}

impl RustSampleShardWriter {
    fn check_failure(&self) -> PyResult<()> {
        match self
            .failure
            .lock()
            .expect("writer failure lock poisoned")
            .clone()
        {
            Some(error) => Err(PyRuntimeError::new_err(error)),
            None => Ok(()),
        }
    }
}

impl Drop for RustSampleShardWriter {
    fn drop(&mut self) {
        if let Some(sender) = self.sender.take() {
            let (reply, _response) = mpsc::channel();
            let _ = sender.send(WriterCommand::Close(reply));
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
