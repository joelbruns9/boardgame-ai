//! Phase F1 pyo3 bindings: a Rust 7 Wonders Duel engine exposed to Python for
//! the byte-exact replay gate (F1a) and the make/unmake round-trip gate (F1b).
//!
//! The engine is constructed from a fully-locked setup (extracted from a Python
//! `GameState.new`) plus the recorded Great Library draws, and replays action
//! indices. See `state.rs` for why no Python RNG is modelled and why the
//! fingerprint is the equivalence surface.

mod bots;
mod chance;
mod codec;
mod data;
mod encoder;
mod engine;
mod eval;
mod pool;
mod rng;
mod rules;
mod self_play;
mod state;
mod tree;
mod tree_resumable;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::VecDeque;

use eval::Eval;
use state::{GameState, Setup};

fn card_ids(names: &[String]) -> Vec<usize> {
    names.iter().map(|n| data::card_id(n)).collect()
}
fn wonder_ids(names: &[String]) -> Vec<usize> {
    names.iter().map(|n| data::wonder_id(n)).collect()
}
fn progress_ids(names: &[String]) -> Vec<usize> {
    names.iter().map(|n| data::progress_id(n)).collect()
}

fn self_play_record_to_py(py: Python<'_>, record: self_play::GameRecord) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("schema", 1)?;
    out.set_item("spec_version", "codec-1")?;
    out.set_item("seed", record.seed)?;
    out.set_item("first_player", record.first_player)?;
    out.set_item("iteration", record.iteration)?;
    out.set_item("winner", record.winner)?;
    out.set_item(
        "victory_type",
        record.victory_type.map(|kind| match kind {
            state::VictoryType::Military => "military",
            state::VictoryType::Scientific => "scientific",
            state::VictoryType::Civilian => "civilian",
            state::VictoryType::SharedCivilian => "shared_civilian",
        }),
    )?;
    out.set_item("scores", record.scores)?;
    let kind = if record.agent_names.iter().any(|name| name != "network") {
        "mixed"
    } else {
        "self_play"
    };
    out.set_item(
        "agents",
        [
            ("p0", record.agent_names[0].as_str()),
            ("p1", record.agent_names[1].as_str()),
            ("kind", kind),
        ]
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>(),
    )?;

    let moves = PyList::empty(py);
    for row in record.moves {
        let item = PyDict::new(py);
        item.set_item("i", row.i)?;
        item.set_item("actor", row.actor)?;
        item.set_item("action", row.action)?;
        item.set_item("legal", row.legal)?;
        item.set_item("visits", row.visits)?;
        item.set_item(
            "policy_target",
            if row.is_bot {
                None
            } else {
                Some(row.policy_target)
            },
        )?;
        item.set_item(
            "root_value",
            if row.is_bot {
                None
            } else {
                Some(row.root_value)
            },
        )?;
        item.set_item("sims", row.sims)?;
        item.set_item("mode", if row.is_bot { "bot" } else { "closed" })?;
        item.set_item(
            "gumbel_topk",
            if row.is_bot {
                None
            } else {
                Some(row.gumbel_topk)
            },
        )?;
        item.set_item("policy_excluded", row.policy_excluded)?;
        item.set_item("full_search", row.full_search)?;
        item.set_item("search_seed", row.search_seed)?;
        moves.append(item)?;
    }
    out.set_item("moves", moves)?;

    let chance_log = PyList::empty(py);
    for event in record.chance_log {
        let item = PyDict::new(py);
        item.set_item("move_index", event.move_index)?;
        item.set_item("kind_id", event.kind as u8)?;
        item.set_item("outcome_ids", event.outcome.clone())?;
        item.set_item(
            "outcome",
            event
                .outcome
                .into_iter()
                .map(|id| self_play::component_name(event.kind, id))
                .collect::<Vec<_>>(),
        )?;
        chance_log.append(item)?;
    }
    out.set_item("chance_log", chance_log)?;
    out.set_item("final_fingerprint", record.final_fingerprint)?;
    Ok(out.unbind())
}

#[allow(clippy::too_many_arguments)]
fn make_self_play_config(
    game_seed: u64,
    iteration: Option<i64>,
    leaf_batch: usize,
    cheap_sims_min: usize,
    cheap_sims_max: usize,
    full_sims_min: usize,
    full_sims_max: usize,
    full_search_fraction: f64,
    top_k: usize,
    draft_prior: f64,
    c_puct: f64,
    c_visit: f64,
    c_scale: f64,
    force: bool,
    puct_root: bool,
    age_deal_samples: usize,
    cheap_double_reveal_offsets: usize,
    max_moves: usize,
    conflict_free_waves: bool,
    round_robin_candidates: bool,
) -> self_play::SelfPlayConfig {
    self_play::SelfPlayConfig {
        game_seed,
        iteration,
        leaf_batch,
        leaf_batch_by_player: None,
        deterministic_actions: false,
        cheap_sims_min,
        cheap_sims_max,
        full_sims_min,
        full_sims_max,
        full_search_fraction,
        top_k,
        draft_prior,
        c_puct,
        c_visit,
        c_scale,
        force_expand_root_chance: force,
        puct_root,
        age_deal_samples,
        age_deal_samples_by_player: None,
        cheap_double_reveal_offsets_by_player: None,
        cheap_double_reveal_offsets,
        bot_by_player: [None, None],
        net_by_player: [0, 0],
        bot_exploration: 0.0,
        bot_policy_iterations: 10,
        max_moves,
        conflict_free_waves,
        round_robin_candidates,
    }
}

/// Enum-valued fields cross the boundary as **declaration indices**. Python and
/// Rust declare `Phase`, `PendingChoiceKind`, `VictoryType` and `ScienceSymbol`
/// in the same order, so the index is the contract; an out-of-range value is a
/// serializer bug and must fail loudly rather than silently pick a variant.
fn science_symbol_from_index(i: u8) -> PyResult<data::ScienceSymbol> {
    use data::ScienceSymbol::*;
    Ok(match i {
        0 => ArmillarySphere,
        1 => Wheel,
        2 => Sundial,
        3 => MortarAndPestle,
        4 => SetSquare,
        5 => QuillAndInk,
        6 => Law,
        other => return Err(PyValueError::new_err(format!("bad science symbol {other}"))),
    })
}

fn pending_kind_from_index(i: u8) -> PyResult<state::PendingChoiceKind> {
    use state::PendingChoiceKind::*;
    Ok(match i {
        0 => DestroyOpponentBrown,
        1 => DestroyOpponentGrey,
        2 => BuildFromDiscardFree,
        3 => ChooseUnusedProgress,
        4 => ChooseAvailableProgress,
        other => return Err(PyValueError::new_err(format!("bad pending kind {other}"))),
    })
}

fn victory_from_index(i: u8) -> PyResult<state::VictoryType> {
    use state::VictoryType::*;
    Ok(match i {
        0 => Military,
        1 => Scientific,
        2 => Civilian,
        3 => SharedCivilian,
        other => return Err(PyValueError::new_err(format!("bad victory type {other}"))),
    })
}

/// A resumable PUCT search over a persistent tree — the advisor's searcher.
///
/// Every other Rust entry point is one-shot: give it `sims`, get a result. The
/// advisor streams instead, deepening ONE tree across many `advance` calls so
/// the panel's numbers refine while the user thinks. Calling a one-shot search
/// per chunk would rebuild the tree each time and destroy that.
///
/// Fixed at `puct_root = true` and `leaf_batch = 1` on purpose:
/// `check_puct_root` rejects PUCT-root together with leaf batching, because the
/// root would then select under WU virtual loss. The advisor keeps PUCT-root
/// semantics (matching the Python `_ClosedHandle` it replaces) and gives up
/// root-level batching. See ADVISOR_RUST_UNIFICATION.md §5 step 4.
///
/// `force_expand_root_chance` is likewise unavailable here (it needs the F4.5
/// forced-child cache), so root chance edges use visit-weighted Q rather than an
/// exact expectation — a real difference from the Python advisor's default.
#[pyclass]
struct RustPuctSearch {
    session: tree_resumable::SearchSession,
    /// The leaf-evaluation boundary.
    ///
    /// `Scalar` calls Python once per leaf. `Batched` calls it once per wave --
    /// the whole point of `leaf_batch > 1`, since batching the tree buys nothing
    /// while evaluation is serial. `Mock` drives the deterministic Rust
    /// evaluator, which the equivalence gate needs because the one-shot
    /// `closed_search` it is compared against uses that same mock.
    evaluator: HandleEval,
}

enum HandleEval {
    Scalar(eval::PyEval),
    Batched(eval::PyBatchEval),
    Mock,
}

#[pymethods]
impl RustPuctSearch {
    /// Open a search over `game`'s position. Evaluates the root immediately
    /// (one call through `adapter`) and expands it; runs no simulations.
    #[staticmethod]
    #[pyo3(signature = (game, adapter, max_sims, seed=0, c_puct=1.5, c_visit=50.0, c_scale=0.1, top_k=16, leaf_batch=1))]
    fn open(
        game: &RustGame,
        adapter: Py<PyAny>,
        max_sims: usize,
        seed: u64,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        top_k: usize,
        leaf_batch: usize,
    ) -> PyResult<Self> {
        let cfg = tree::SearchConfig {
            sims: max_sims.max(1),
            top_k,
            c_puct,
            c_visit,
            c_scale,
            seed,
            force_expand_root_chance: false,
            puct_root: true,
            age_deal_samples: 0,
            double_reveal_offsets: 0,
            conflict_free_waves: false,
            round_robin_candidates: false,
        };
        // A batched wave needs a batched boundary, or every leaf still crosses
        // into Python on its own and leaf_batch changes nothing (measured: 1.00x
        // through 1.07x across leaf_batch 1..16 on the scalar bridge).
        let evaluator = if leaf_batch > 1 {
            HandleEval::Batched(eval::PyBatchEval::new(adapter))
        } else {
            HandleEval::Scalar(eval::PyEval::new(adapter))
        };
        let root_evaluation = match &evaluator {
            HandleEval::Batched(e) => e.evaluate(&game.state)?,
            HandleEval::Scalar(e) => e.evaluate(&game.state)?,
            HandleEval::Mock => eval::MockEval.evaluate(&game.state)?,
        };
        // leaf_batch > 1 opts the root into virtual-loss selection; see
        // begin_search_from_root_virtual_loss for what that trades away.
        let session = if leaf_batch > 1 {
            tree_resumable::begin_search_from_root_virtual_loss(
                &game.state,
                &cfg,
                leaf_batch,
                root_evaluation,
            )?
        } else {
            tree_resumable::begin_search_from_root(&game.state, &cfg, 1, root_evaluation)?
        };
        Ok(RustPuctSearch { session, evaluator })
    }

    /// Mock-evaluator twin of `open`, for the equivalence gate only.
    #[staticmethod]
    #[pyo3(signature = (game, max_sims, seed=0, c_puct=1.5, c_visit=50.0, c_scale=0.1, top_k=16))]
    fn open_mock(
        game: &RustGame,
        max_sims: usize,
        seed: u64,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        top_k: usize,
    ) -> PyResult<Self> {
        let cfg = tree::SearchConfig {
            sims: max_sims.max(1),
            top_k,
            c_puct,
            c_visit,
            c_scale,
            seed,
            force_expand_root_chance: false,
            puct_root: true,
            age_deal_samples: 0,
            double_reveal_offsets: 0,
            conflict_free_waves: false,
            round_robin_candidates: false,
        };
        let root_evaluation = eval::MockEval.evaluate(&game.state)?;
        let session = tree_resumable::begin_search_from_root(&game.state, &cfg, 1, root_evaluation)?;
        Ok(RustPuctSearch {
            session,
            evaluator: HandleEval::Mock,
        })
    }

    /// Run up to `chunk` more simulations; return the total completed. Stops on
    /// the chunk boundary, so a caller can publish a snapshot and come back.
    fn advance(&mut self, chunk: usize) -> PyResult<usize> {
        let target = self.session.sims_done().saturating_add(chunk);
        while self.session.sims_done() < target {
            match self.session.next_event()? {
                tree_resumable::SearchEvent::Complete => break,
                tree_resumable::SearchEvent::Evaluation(request) => {
                    let evaluations = {
                        let states = self.session.evaluation_states(&request)?;
                        let actors: Vec<_> =
                            request.leaves.iter().map(|leaf| leaf.actor).collect();
                        let legals: Vec<_> = request
                            .leaves
                            .iter()
                            .map(|leaf| leaf.legal.clone())
                            .collect();
                        match &self.evaluator {
                            HandleEval::Scalar(e) => {
                                e.evaluate_batch_prepared(&states, &actors, &legals)
                            }
                            HandleEval::Batched(e) => {
                                e.evaluate_batch_prepared(&states, &actors, &legals)
                            }
                            HandleEval::Mock => eval::MockEval
                                .evaluate_batch_prepared(&states, &actors, &legals),
                        }
                    };
                    match evaluations {
                        Ok(rows) => self.session.apply_evaluations(request.request_id, rows)?,
                        Err(err) => {
                            self.session.cancel_pending();
                            return Err(err);
                        }
                    }
                }
            }
        }
        Ok(self.session.sims_done())
    }

    /// `(sims_done, root_visits, root_value_sum_p0, root_actor,
    /// [(action_index, visits, value_sum_p0, prior)])`. Raw sums: the caller
    /// divides and applies the p0->actor sign, which keeps seat knowledge in the
    /// Python adapter where it already lives.
    fn snapshot(&self) -> (usize, u32, f64, usize, Vec<(usize, u32, f64, f64)>) {
        let (visits, value_sum, actor, edges) = self.session.root_stats();
        (self.session.sims_done(), visits, value_sum, actor, edges)
    }

    /// `[(root_action_index, ranked_follow_up_indices, contingent)]` for the
    /// root actions whose move is not over -- see `follow_ups` in
    /// `tree_resumable.rs`. `contingent` marks an option set that is itself
    /// random (the Great Library's draw), where the caller must render a
    /// preference order rather than one forced move. Absent from `snapshot` on
    /// purpose: it is a separate walk, and keeping it out leaves that tuple's
    /// shape (and its tests) alone.
    fn follow_ups(&self) -> Vec<(usize, Vec<usize>, bool)> {
        self.session.follow_ups()
    }

    fn arena_nodes(&self) -> usize {
        self.session.arena_nodes()
    }

    /// Approximate resident bytes of this search's arena. The advisor stops
    /// deepening when this passes its budget: on a wide root the tree grows a
    /// node per simulation and each node owns a cloned `GameState`, so an
    /// unbounded "think until the board changes" search is unbounded in memory
    /// too (measured: 400k sims -> 1.7 GB).
    fn arena_deep_bytes(&self) -> usize {
        self.session.arena_deep_bytes()
    }
}

/// A 7WD game state driven from Python by codec action index.
#[pyclass]
struct RustGame {
    state: GameState,
}

#[pymethods]
impl RustGame {
    /// Construct from a fully-locked setup. Lists carry component *names*; Rust
    /// maps them to the same ids Python's `CARD_IDS`/`WONDER_IDS`/`PROGRESS_IDS`
    /// assign (both index into the identical `data.py` tables).
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        first_player, available_progress, unused_progress, wonder_group0,
        wonder_group1, unused_wonders, age1, age2, age3, removed1, removed2,
        removed3, selected_guilds, unused_guilds, library_draws
    ))]
    fn new(
        first_player: usize,
        available_progress: Vec<String>,
        unused_progress: Vec<String>,
        wonder_group0: Vec<String>,
        wonder_group1: Vec<String>,
        unused_wonders: Vec<String>,
        age1: Vec<String>,
        age2: Vec<String>,
        age3: Vec<String>,
        removed1: Vec<String>,
        removed2: Vec<String>,
        removed3: Vec<String>,
        selected_guilds: Vec<String>,
        unused_guilds: Vec<String>,
        library_draws: Vec<Vec<String>>,
    ) -> PyResult<Self> {
        if first_player > 1 {
            return Err(PyValueError::new_err("first_player must be 0 or 1"));
        }
        let setup = Setup {
            first_player,
            available_progress_tokens: progress_ids(&available_progress),
            unused_progress_tokens: progress_ids(&unused_progress),
            wonder_groups: [wonder_ids(&wonder_group0), wonder_ids(&wonder_group1)],
            unused_wonders: wonder_ids(&unused_wonders),
            age_decks: [
                Vec::new(),
                card_ids(&age1),
                card_ids(&age2),
                card_ids(&age3),
            ],
            removed_age_cards: [
                Vec::new(),
                card_ids(&removed1),
                card_ids(&removed2),
                card_ids(&removed3),
            ],
            selected_guilds: card_ids(&selected_guilds),
            unused_guilds: card_ids(&unused_guilds),
        };
        let draws: VecDeque<Vec<usize>> = library_draws.iter().map(|d| progress_ids(d)).collect();
        Ok(RustGame {
            state: GameState::from_setup(setup, draws),
        })
    }

    /// Construct from a **complete mid-game state** rather than a setup.
    ///
    /// `new` reaches a position by replaying actions from a locked deal, which
    /// is all self-play needs. The advisor cannot: it rebuilds a position from a
    /// public BGA observation, with hidden information supplied by a
    /// determinizer and **no action history at all**, so there is no seed and no
    /// prefix to replay. Without this the Rust engine, encoder and searcher are
    /// all unreachable from the advisor.
    ///
    /// Ids are integers in the same spaces Python's `CARD_IDS` / `WONDER_IDS` /
    /// `PROGRESS_IDS` use. Enum-valued fields are **declaration indices**, which
    /// agree across the two languages by construction (`Phase`,
    /// `PendingChoiceKind`, `VictoryType` and `ScienceSymbol` are declared in the
    /// same order on both sides); `rust_bridge.rust_state()` is the only
    /// supported producer of these arguments.
    ///
    /// This is a *wide* boundary and its whole value is exactness, so it is
    /// gated by `test_rust_state_injection.py`, which asserts that an injected
    /// state fingerprints identically to the same position reached by replay.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        first_player, phase, active_player, age, cities, available_progress,
        unused_progress, wonder_group0, wonder_group1, unused_wonders,
        wonder_offer, wonder_round, wonder_pick_index, age_decks,
        removed_age_cards, selected_guilds, unused_guilds, tableau_age,
        tableau_slots, discard_pile, buried_cards, retired_wonders,
        wonder_burials, pending_choice, pending_extra_turn, pending_shields,
        conflict_position, military_tokens_remaining, winner, victory_type,
        final_scores, library_draws
    ))]
    fn from_state(
        first_player: usize,
        phase: u8,
        active_player: usize,
        age: u8,
        // (coins, wonders, built_wonders, buildings, progress_tokens, science_pairs)
        cities: Vec<(i32, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<u8>)>,
        available_progress: Vec<usize>,
        unused_progress: Vec<usize>,
        wonder_group0: Vec<usize>,
        wonder_group1: Vec<usize>,
        unused_wonders: Vec<usize>,
        wonder_offer: Vec<usize>,
        wonder_round: u8,
        wonder_pick_index: u8,
        age_decks: Vec<Vec<usize>>,
        removed_age_cards: Vec<Vec<usize>>,
        selected_guilds: Vec<usize>,
        unused_guilds: Vec<usize>,
        tableau_age: u8,
        tableau_slots: Vec<(usize, bool, bool)>,
        discard_pile: Vec<usize>,
        buried_cards: Vec<usize>,
        retired_wonders: Vec<usize>,
        wonder_burials: Vec<(usize, usize)>,
        pending_choice: Option<(u8, usize, Vec<usize>, bool)>,
        pending_extra_turn: bool,
        pending_shields: i32,
        conflict_position: i32,
        military_tokens_remaining: Vec<(i32, i32)>,
        winner: Option<usize>,
        victory_type: Option<u8>,
        final_scores: Option<(i32, i32)>,
        library_draws: Vec<Vec<usize>>,
    ) -> PyResult<Self> {
        fn four(v: Vec<Vec<usize>>, what: &str) -> PyResult<[Vec<usize>; 4]> {
            let n = v.len();
            <[Vec<usize>; 4]>::try_from(v).map_err(|_| {
                PyValueError::new_err(format!(
                    "{what} must have 4 entries (index 0 unused, then ages 1..3), got {n}"
                ))
            })
        }
        if cities.len() != 2 {
            return Err(PyValueError::new_err("cities must have exactly 2 entries"));
        }
        let phase = match phase {
            0 => state::Phase::WonderDraft,
            1 => state::Phase::PlayAge,
            2 => state::Phase::ChooseNextStartPlayer,
            3 => state::Phase::Complete,
            other => return Err(PyValueError::new_err(format!("bad phase {other}"))),
        };
        let mut built_cities: Vec<state::CityState> = Vec::with_capacity(2);
        for (coins, wonders, built_wonders, buildings, progress_tokens, pairs) in cities {
            let mut claimed = Vec::with_capacity(pairs.len());
            for p in pairs {
                claimed.push(science_symbol_from_index(p)?);
            }
            built_cities.push(state::CityState {
                coins,
                wonders,
                built_wonders,
                buildings,
                progress_tokens,
                claimed_science_pairs: claimed,
            });
        }
        let pending = match pending_choice {
            None => None,
            Some((kind, player, options, consume_all_options)) => Some(state::PendingChoice {
                kind: pending_kind_from_index(kind)?,
                player,
                options,
                consume_all_options,
            }),
        };
        let victory = match victory_type {
            None => None,
            Some(v) => Some(victory_from_index(v)?),
        };
        let slots = tableau_slots
            .into_iter()
            .map(|(card_id, revealed, present)| state::TableauCard {
                card_id,
                revealed,
                present,
            })
            .collect();
        let state = GameState {
            first_player,
            phase,
            active_player,
            age,
            cities: [built_cities.remove(0), built_cities.remove(0)],
            available_progress_tokens: available_progress,
            unused_progress_tokens: unused_progress,
            wonder_groups: [wonder_group0, wonder_group1],
            unused_wonders,
            wonder_offer,
            wonder_round,
            wonder_pick_index,
            age_decks: four(age_decks, "age_decks")?,
            removed_age_cards: four(removed_age_cards, "removed_age_cards")?,
            selected_guilds,
            unused_guilds,
            tableau: state::TableauState {
                age: tableau_age,
                slots,
            },
            discard_pile,
            buried_cards,
            retired_wonders,
            wonder_burials,
            pending_choice: pending,
            pending_extra_turn,
            pending_shields,
            conflict_position,
            military_tokens_remaining,
            winner,
            victory_type: victory,
            final_scores,
            library_draws: library_draws.into_iter().collect(),
        };
        Ok(RustGame { state })
    }

    /// Sorted codec indices of exactly the engine's legal actions.
    fn legal_action_indices(&self) -> Vec<usize> {
        codec::legal_action_indices(&self.state)
    }

    /// Canonical integer fingerprint of all game-logic state (RNG excluded).
    fn fingerprint(&self) -> Vec<i32> {
        self.state.fingerprint()
    }

    /// Decode `index` in the current state and apply it (advances the game).
    /// Rejects any index that is not a currently-legal action: this is the
    /// public boundary, and the decoder alone does not verify wonder
    /// ownership/retirement or affordability, so an unchecked index could
    /// otherwise mutate state illegally.
    fn apply_index(&mut self, index: usize) -> PyResult<()> {
        if !codec::legal_action_indices(&self.state).contains(&index) {
            return Err(PyValueError::new_err(format!(
                "illegal action index {index} for the current state"
            )));
        }
        let action = codec::decode_action(&self.state, index);
        self.state.apply_action(&action);
        Ok(())
    }

    #[pyo3(signature = (kind, seed=0, exploration=0.0))]
    fn bot_action(&self, kind: &str, seed: u64, exploration: f64) -> PyResult<usize> {
        let kind = bots::BotKind::parse(kind)
            .ok_or_else(|| PyValueError::new_err(format!("unknown Rust bot: {kind}")))?;
        if !(0.0..=1.0).contains(&exploration) {
            return Err(PyValueError::new_err("exploration must be in [0, 1]"));
        }
        Ok(bots::select_action(
            &self.state,
            kind,
            &mut rng::Rng::new(seed),
            exploration,
        ))
    }

    /// F1b: apply `index` then unmake, returning whether the *complete* state is
    /// restored (`GameState: PartialEq`, not just the cross-language
    /// fingerprint). Leaves the game unchanged (snapshot-based undo).
    fn roundtrip_ok(&mut self, index: usize) -> PyResult<bool> {
        let before = self.state.clone();
        let undo = self.state.snapshot();
        let action = codec::decode_action(&self.state, index);
        self.state.apply_action(&action);
        self.state.restore(undo);
        Ok(self.state == before)
    }

    /// F1b (strengthened): exhaustive make/unmake audit from the current state —
    /// every legal action to `depth` plies (nested LIFO), full-state undo, and
    /// apply determinism. Non-destructive (operates on clones). Run on sampled
    /// states in the gate; `depth=2` proves nesting without an O(branch^3) cost.
    #[pyo3(signature = (depth=2))]
    fn roundtrip_all_ok(&self, depth: usize) -> bool {
        engine::make_unmake_audit(&self.state, depth).is_ok()
    }

    /// F2.1 foundation: the unseen-card pool read from the public projection.
    /// Returns `(age1, age2, age3, guild, wonders, offboard_progress)`, each a
    /// sorted id list — the encoder's hidden-structure inputs. Viewer-independent
    /// (hidden info is symmetric).
    fn unseen_pool(
        &self,
    ) -> (
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
    ) {
        let p = pool::unseen_pool(&self.state);
        let [age1, age2, age3, guild] = p.cards;
        (age1, age2, age3, guild, p.wonders, p.offboard_progress)
    }

    /// F2.2: the actor-relative encoder token sequence. Each token is
    /// `(type_id, entity_id, aux_id, features)` with `type_id` in `TokenType`
    /// declaration order (GLOBAL=0 … POOL_WONDER=8). Actor-relative: derived
    /// from the pending choice's player, else the active player.
    fn encode(&self) -> Vec<(usize, i32, i32, Vec<f64>)> {
        encoder::encode(&self.state)
            .into_iter()
            .map(|t| (t.type_id, t.entity_id, t.aux_id, t.features))
            .collect()
    }

    /// F3.1a: predicted chance events for action `index`, as
    /// `(kind_id, context)` — kind_id in `ChanceKind` order (CardReveal=0 …
    /// AgeDeal=3), context flattened (CardReveal `[row, x, back]`, AgeDeal
    /// `[age]`, else `[]`).
    fn chance_signature(&self, index: usize) -> Vec<(u8, Vec<i32>)> {
        let action = codec::decode_action(&self.state, index);
        chance::chance_signature(&self.state, &action)
            .into_iter()
            .map(|s| (s.kind as u8, s.context))
            .collect()
    }

    /// F3.1a: all `(outcomes, probability, observable_key)` chains for action
    /// `index`'s enumerable chance specs. Each chain's `outcomes` is one id list
    /// per spec (CardReveal `[card_id]`, GreatLibraryDraw `[p,p,p]`,
    /// WonderGroupReveal `[w,w,w,w]`); `key` equals the outcomes off AGE_DEAL.
    /// Errors on AgeDeal (sample-only).
    fn enumerate_chains(
        &self,
        index: usize,
    ) -> PyResult<Vec<(Vec<Vec<usize>>, f64, Vec<Vec<i32>>)>> {
        let action = codec::decode_action(&self.state, index);
        let specs = chance::chance_signature(&self.state, &action);
        if specs.iter().any(|s| s.kind == chance::ChanceKind::AgeDeal) {
            return Err(PyValueError::new_err("cannot enumerate AGE_DEAL chains"));
        }
        Ok(chance::enumerate_chains(&self.state, &specs))
    }

    /// F3.1b: sample one chance chain for action `index` from a fresh
    /// `Rng(seed)` — the standalone-seed form the gate compares against Python's
    /// `sample_outcomes(..., PortableRng(seed))`. Returns `(outcomes, prob)` with
    /// `prob` absent when a spec is sample-only (AGE_DEAL).
    fn sample_outcomes(
        &self,
        index: usize,
        seed: u64,
    ) -> (Vec<Vec<usize>>, Option<f64>, Vec<Vec<i32>>) {
        let action = codec::decode_action(&self.state, index);
        let specs = chance::chance_signature(&self.state, &action);
        let mut rng = rng::Rng::new(seed);
        chance::sample_outcomes(&self.state, &specs, &mut rng)
    }

    /// F3.1b: fingerprint of the state after applying action `index` with
    /// supplied chance `outcomes` (one id list per spec). Non-destructive
    /// (snapshot/restore) so the gate can probe many outcomes from one state.
    fn fingerprint_after_chance(
        &mut self,
        index: usize,
        outcomes: Vec<Vec<usize>>,
    ) -> PyResult<Vec<i32>> {
        let undo = self.state.snapshot();
        let action = codec::decode_action(&self.state, index);
        let result = self.state.apply_with_chance(&action, &outcomes);
        match result {
            Ok(()) => {
                let fp = self.state.fingerprint();
                self.state.restore(undo);
                Ok(fp)
            }
            Err(e) => {
                self.state.restore(undo);
                Err(PyValueError::new_err(e))
            }
        }
    }

    /// F3.2: deterministic mock leaf evaluation of the current state
    /// `(value_p0, priors aligned to legal_action_indices)` — the shared oracle
    /// for the tree-equivalence gate.
    fn mock_eval(&self) -> (f64, Vec<f64>) {
        eval::MockEval::eval_state(&self.state)
    }

    /// F3.2: build the closed tree from the current state with `MockEval`, a
    /// fixed round-robin root schedule, `sims` simulations and RNG `seed`, and
    /// return its canonical digest for the 1e-6 equivalence gate.
    #[pyo3(signature = (sims, seed, c_puct=1.5))]
    fn closed_tree_digest(&self, sims: usize, seed: u64, c_puct: f64) -> PyResult<Vec<f64>> {
        let root = tree::closed_tree_fixed(&self.state, sims, &eval::MockEval, seed, c_puct)?;
        let mut out = Vec::new();
        tree::digest(&root, &mut out);
        Ok(out)
    }

    /// F3.3: full closed search (Gumbel root + sequential halving) from the
    /// current state under `MockEval`. Returns `(action_index, action_value,
    /// root_value, visits, policy_target, gumbel_topk, sims, tree_digest)` with
    /// `visits`/`policy_target` aligned to `legal_action_indices`.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (sims, top_k, seed, c_puct=1.5, c_visit=50.0, c_scale=0.1, force=false, puct_root=false, double_reveal_offsets=0, conflict_free_waves=false, round_robin_candidates=false))]
    fn closed_search(
        &self,
        sims: usize,
        top_k: usize,
        seed: u64,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
        puct_root: bool,
        double_reveal_offsets: usize,
        conflict_free_waves: bool,
        round_robin_candidates: bool,
    ) -> PyResult<(
        usize,
        f64,
        f64,
        Vec<u32>,
        Vec<f64>,
        Vec<usize>,
        usize,
        Vec<f64>,
    )> {
        let cfg = tree::SearchConfig {
            sims,
            top_k,
            c_puct,
            c_visit,
            c_scale,
            seed,
            force_expand_root_chance: force,
            puct_root,
            age_deal_samples: 0,
            double_reveal_offsets,
            conflict_free_waves,
            round_robin_candidates,
        };
        let (res, root) = tree::search_closed(&self.state, &eval::MockEval, &cfg)?;
        let mut dig = Vec::new();
        tree::digest(&root, &mut dig);
        Ok((
            res.action_index,
            res.action_value,
            res.root_value,
            res.visits,
            res.policy_target,
            res.gumbel_topk,
            res.sims,
            dig,
        ))
    }

    /// F4.1: arena-backed, phase-split closed search.  `leaf_batch=1` is the
    /// exact refactor gate: selection/materialization yields an evaluation
    /// request, evaluation is applied separately, and stable arena paths are
    /// backed up before the next simulation is selected.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (sims, top_k, seed, c_puct=1.5, c_visit=50.0, c_scale=0.1, force=false, double_reveal_offsets=0, conflict_free_waves=false, round_robin_candidates=false))]
    fn closed_search_resumable(
        &self,
        sims: usize,
        top_k: usize,
        seed: u64,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
        double_reveal_offsets: usize,
        conflict_free_waves: bool,
        round_robin_candidates: bool,
    ) -> PyResult<(
        usize,
        f64,
        f64,
        Vec<u32>,
        Vec<f64>,
        Vec<usize>,
        usize,
        Vec<f64>,
    )> {
        let cfg = tree::SearchConfig {
            sims,
            top_k,
            c_puct,
            c_visit,
            c_scale,
            seed,
            force_expand_root_chance: force,
            puct_root: false,
            age_deal_samples: 0,
            double_reveal_offsets,
            conflict_free_waves,
            round_robin_candidates,
        };
        let (res, arena) = tree_resumable::search_closed(&self.state, &eval::MockEval, &cfg)?;
        let mut dig = Vec::new();
        tree_resumable::digest(&arena, &mut dig);
        Ok((
            res.action_index,
            res.action_value,
            res.root_value,
            res.visits,
            res.policy_target,
            res.gumbel_topk,
            res.sims,
            dig,
        ))
    }

    /// F4.1 real-evaluator counterpart to `closed_search_resumable`. This is
    /// still the scalar F3.4 Python adapter; F4.4/F4.5 replace the boundary,
    /// while this method keeps the phase split independently gateable today.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (adapter, sims, top_k, seed, c_puct=1.5, c_visit=50.0, c_scale=0.1, force=false, conflict_free_waves=false, round_robin_candidates=false))]
    fn closed_search_resumable_net(
        &self,
        adapter: Py<PyAny>,
        sims: usize,
        top_k: usize,
        seed: u64,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
        conflict_free_waves: bool,
        round_robin_candidates: bool,
    ) -> PyResult<(
        usize,
        f64,
        f64,
        Vec<u32>,
        Vec<f64>,
        Vec<usize>,
        usize,
        Vec<f64>,
    )> {
        let cfg = tree::SearchConfig {
            sims,
            top_k,
            c_puct,
            c_visit,
            c_scale,
            seed,
            force_expand_root_chance: force,
            puct_root: false,
            age_deal_samples: 0,
            double_reveal_offsets: 0,
            conflict_free_waves,
            round_robin_candidates,
        };
        let evaluator = eval::PyEval::new(adapter);
        let (res, arena) = tree_resumable::search_closed(&self.state, &evaluator, &cfg)?;
        let mut dig = Vec::new();
        tree_resumable::digest(&arena, &mut dig);
        Ok((
            res.action_index,
            res.action_value,
            res.root_value,
            res.visits,
            res.policy_target,
            res.gumbel_topk,
            res.sims,
            dig,
        ))
    }

    /// F4.2 WU-PUCT leaf waves under the deterministic mock evaluator. Returns
    /// the normal search tuple plus `(scheduled, requested, unique, terminal,
    /// collisions, waves, max_wave_paths, max_wave_unique)`, completed-Q aligned
    /// to legal actions, and the tree digest.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (leaf_batch, sims, top_k, seed, c_puct=1.5, c_visit=50.0, c_scale=0.1, force=false, puct_root=false, conflict_free_waves=false, round_robin_candidates=false))]
    fn closed_search_batched(
        &self,
        leaf_batch: usize,
        sims: usize,
        top_k: usize,
        seed: u64,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
        puct_root: bool,
        conflict_free_waves: bool,
        round_robin_candidates: bool,
    ) -> PyResult<(
        usize,
        f64,
        f64,
        Vec<u32>,
        Vec<f64>,
        Vec<usize>,
        usize,
        (usize, usize, usize, usize, usize, usize, usize, usize),
        Vec<f64>,
        Vec<f64>,
    )> {
        let cfg = tree::SearchConfig {
            sims,
            top_k,
            c_puct,
            c_visit,
            c_scale,
            seed,
            force_expand_root_chance: force,
            puct_root,
            age_deal_samples: 0,
            double_reveal_offsets: 0,
            conflict_free_waves,
            round_robin_candidates,
        };
        let (res, arena, metrics) =
            tree_resumable::search_closed_batched(&self.state, &eval::MockEval, &cfg, leaf_batch)?;
        let mut dig = Vec::new();
        tree_resumable::digest(&arena, &mut dig);
        Ok((
            res.action_index,
            res.action_value,
            res.root_value,
            res.visits,
            res.policy_target,
            res.gumbel_topk,
            res.sims,
            (
                metrics.scheduled_simulations,
                metrics.requested_nn_leaves,
                metrics.unique_nn_leaves,
                metrics.terminal_leaves,
                metrics.collisions,
                metrics.leaf_waves,
                metrics.max_wave_paths,
                metrics.max_wave_unique,
            ),
            metrics.root_completed_q,
            dig,
        ))
    }

    /// F4.2 WU leaf-wave path through the scalar correctness adapter. The
    /// adapter remains one Python call per unique leaf until the F4.4 global
    /// coalescer; this surface gates batched search semantics with a real net.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (adapter, leaf_batch, sims, top_k, seed, c_puct=1.5, c_visit=50.0, c_scale=0.1, force=false, conflict_free_waves=false, round_robin_candidates=false))]
    fn closed_search_batched_net(
        &self,
        adapter: Py<PyAny>,
        leaf_batch: usize,
        sims: usize,
        top_k: usize,
        seed: u64,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
        conflict_free_waves: bool,
        round_robin_candidates: bool,
    ) -> PyResult<(
        usize,
        f64,
        f64,
        Vec<u32>,
        Vec<f64>,
        Vec<usize>,
        usize,
        (usize, usize, usize, usize, usize, usize, usize, usize),
        Vec<f64>,
        Vec<f64>,
    )> {
        let cfg = tree::SearchConfig {
            sims,
            top_k,
            c_puct,
            c_visit,
            c_scale,
            seed,
            force_expand_root_chance: force,
            puct_root: false,
            age_deal_samples: 0,
            double_reveal_offsets: 0,
            conflict_free_waves,
            round_robin_candidates,
        };
        let evaluator = eval::PyEval::new(adapter);
        let (res, arena, metrics) =
            tree_resumable::search_closed_batched(&self.state, &evaluator, &cfg, leaf_batch)?;
        let mut dig = Vec::new();
        tree_resumable::digest(&arena, &mut dig);
        Ok((
            res.action_index,
            res.action_value,
            res.root_value,
            res.visits,
            res.policy_target,
            res.gumbel_topk,
            res.sims,
            (
                metrics.scheduled_simulations,
                metrics.requested_nn_leaves,
                metrics.unique_nn_leaves,
                metrics.terminal_leaves,
                metrics.collisions,
                metrics.leaf_waves,
                metrics.max_wave_paths,
                metrics.max_wave_unique,
            ),
            metrics.root_completed_q,
            dig,
        ))
    }

    /// F3.4: like `closed_search` but with the real net. `adapter` is a Python
    /// callable `(tokens, actor, legal) -> (value_actor, priors)`; the Rust
    /// encoder (F2) feeds it, so results match Python's searcher on the same net.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (adapter, sims, top_k, seed, c_puct=1.5, c_visit=50.0, c_scale=0.1, force=false, puct_root=false, conflict_free_waves=false, round_robin_candidates=false))]
    fn closed_search_net(
        &self,
        adapter: Py<PyAny>,
        sims: usize,
        top_k: usize,
        seed: u64,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
        puct_root: bool,
        conflict_free_waves: bool,
        round_robin_candidates: bool,
    ) -> PyResult<(
        usize,
        f64,
        f64,
        Vec<u32>,
        Vec<f64>,
        Vec<usize>,
        usize,
        Vec<f64>,
    )> {
        let cfg = tree::SearchConfig {
            sims,
            top_k,
            c_puct,
            c_visit,
            c_scale,
            seed,
            force_expand_root_chance: force,
            puct_root,
            age_deal_samples: 0,
            double_reveal_offsets: 0,
            conflict_free_waves,
            round_robin_candidates,
        };
        let evaluator = eval::PyEval::new(adapter);
        let (res, root) = tree::search_closed(&self.state, &evaluator, &cfg)?;
        let mut dig = Vec::new();
        tree::digest(&root, &mut dig);
        Ok((
            res.action_index,
            res.action_value,
            res.root_value,
            res.visits,
            res.policy_target,
            res.gumbel_topk,
            res.sims,
            dig,
        ))
    }

    /// F4.3: run one complete network-vs-network self-play game in Rust under
    /// the deterministic Phase-D schedule.  Python is called only for neural
    /// leaf evaluations and receives one completed raw record at the end.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        adapter, game_seed, leaf_batch, cheap_sims_min, cheap_sims_max,
        full_sims_min, full_sims_max, full_search_fraction, top_k, draft_prior,
        iteration=None, c_puct=1.5, c_visit=50.0, c_scale=0.1, force=false,
        age_deal_samples=0, cheap_double_reveal_offsets=0, max_moves=256,
        conflict_free_waves=false, round_robin_candidates=false
    ))]
    fn self_play_net(
        &self,
        adapter: Py<PyAny>,
        game_seed: u64,
        leaf_batch: usize,
        cheap_sims_min: usize,
        cheap_sims_max: usize,
        full_sims_min: usize,
        full_sims_max: usize,
        full_search_fraction: f64,
        top_k: usize,
        draft_prior: f64,
        iteration: Option<i64>,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
        age_deal_samples: usize,
        cheap_double_reveal_offsets: usize,
        max_moves: usize,
        conflict_free_waves: bool,
        round_robin_candidates: bool,
    ) -> PyResult<Py<PyDict>> {
        let cfg = make_self_play_config(
            game_seed,
            iteration,
            leaf_batch,
            cheap_sims_min,
            cheap_sims_max,
            full_sims_min,
            full_sims_max,
            full_search_fraction,
            top_k,
            draft_prior,
            c_puct,
            c_visit,
            c_scale,
            force,
            false, // self-play always uses the Gumbel root
            age_deal_samples,
            cheap_double_reveal_offsets,
            max_moves,
            conflict_free_waves,
            round_robin_candidates,
        );
        let evaluator = eval::PyEval::new(adapter);
        let record = self_play::run(&self.state, &evaluator, &cfg)?;
        Python::attach(|py| self_play_record_to_py(py, record))
    }

    /// Deterministic mock-evaluator counterpart used by the F4.3 full-game
    /// oracle and replay/schema gates.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        game_seed, leaf_batch, cheap_sims_min, cheap_sims_max, full_sims_min,
        full_sims_max, full_search_fraction, top_k, draft_prior,
        iteration=None, c_puct=1.5, c_visit=50.0, c_scale=0.1, force=false,
        age_deal_samples=0, cheap_double_reveal_offsets=0, max_moves=256,
        conflict_free_waves=false, round_robin_candidates=false
    ))]
    fn self_play_mock(
        &self,
        game_seed: u64,
        leaf_batch: usize,
        cheap_sims_min: usize,
        cheap_sims_max: usize,
        full_sims_min: usize,
        full_sims_max: usize,
        full_search_fraction: f64,
        top_k: usize,
        draft_prior: f64,
        iteration: Option<i64>,
        c_puct: f64,
        c_visit: f64,
        c_scale: f64,
        force: bool,
        age_deal_samples: usize,
        cheap_double_reveal_offsets: usize,
        max_moves: usize,
        conflict_free_waves: bool,
        round_robin_candidates: bool,
    ) -> PyResult<Py<PyDict>> {
        let cfg = make_self_play_config(
            game_seed,
            iteration,
            leaf_batch,
            cheap_sims_min,
            cheap_sims_max,
            full_sims_min,
            full_sims_max,
            full_search_fraction,
            top_k,
            draft_prior,
            c_puct,
            c_visit,
            c_scale,
            force,
            false, // self-play always uses the Gumbel root
            age_deal_samples,
            cheap_double_reveal_offsets,
            max_moves,
            conflict_free_waves,
            round_robin_candidates,
        );
        let record = self_play::run(&self.state, &eval::MockEval, &cfg)?;
        Python::attach(|py| self_play_record_to_py(py, record))
    }

    fn is_complete(&self) -> bool {
        self.state.phase == state::Phase::Complete
    }

    #[getter]
    fn active_player(&self) -> usize {
        self.state.active_player
    }

    /// Seat that owns the next decision: the pending chooser when one exists,
    /// otherwise the active player. This is the seat whose network must drive a
    /// search from this position, so a two-network arena can pick the right one.
    #[getter]
    fn actor(&self) -> usize {
        tree::state_actor(&self.state)
    }

    #[getter]
    fn winner(&self) -> Option<usize> {
        self.state.winner
    }

    #[getter]
    fn victory_type(&self) -> Option<&'static str> {
        self.state.victory_type.map(|kind| match kind {
            state::VictoryType::Military => "military",
            state::VictoryType::Scientific => "scientific",
            state::VictoryType::Civilian => "civilian",
            state::VictoryType::SharedCivilian => "shared_civilian",
        })
    }

    #[getter]
    fn final_scores(&self) -> Option<(i32, i32)> {
        self.state.final_scores
    }
}

/// Total size of the fixed action space (1202).
#[pyfunction]
fn num_actions() -> usize {
    codec::NUM_ACTIONS
}

/// The encoder schema signature this build produces (must equal Python's
/// `ENCODER_SIGNATURE`). F4 uses it to reject checkpoints trained on a different
/// feature schema.
#[pyfunction]
fn encoder_signature() -> &'static str {
    encoder::ENCODER_SIGNATURE
}

/// `n` consecutive `gumbel()` draws from `Rng(seed)` — lets the gate check
/// cross-runtime `ln` parity in bulk (F3.3 needs bit-identical Gumbel keys).
#[pyfunction]
fn gumbel_stream(seed: u64, n: usize) -> Vec<f64> {
    let mut r = rng::Rng::new(seed);
    (0..n).map(|_| r.gumbel()).collect()
}

/// `x.ln()` for each input — the gate uses it to confirm cross-runtime `ln`
/// parity over the range `log_prior = ln(max(prior, 1e-12))` covers.
#[pyfunction]
fn ln_values(xs: Vec<f64>) -> Vec<f64> {
    xs.iter().map(|&x| x.ln()).collect()
}

fn scheduler_result_to_py(
    py: Python<'_>,
    result: self_play::SchedulerResult,
) -> PyResult<(Vec<Py<PyDict>>, Py<PyDict>)> {
    let records = result
        .records
        .into_iter()
        .map(|record| self_play_record_to_py(py, record))
        .collect::<PyResult<_>>()?;
    let metrics = PyDict::new(py);
    let m = result.metrics;
    metrics.set_item("games", m.games)?;
    metrics.set_item("moves", m.moves)?;
    metrics.set_item("simulations", m.simulations)?;
    metrics.set_item("requested_nn_leaves", m.requested_nn_leaves)?;
    metrics.set_item("unique_nn_leaves", m.unique_nn_leaves)?;
    metrics.set_item("terminal_leaves", m.terminal_leaves)?;
    metrics.set_item("collisions", m.collisions)?;
    metrics.set_item("global_batches", m.global_batches)?;
    metrics.set_item("global_rows", m.global_rows)?;
    metrics.set_item("root_rows", m.root_rows)?;
    metrics.set_item("leaf_rows", m.leaf_rows)?;
    metrics.set_item("forced_rows", m.forced_rows)?;
    metrics.set_item("fixed_support_edges", m.fixed_support_edges)?;
    metrics.set_item("forced_card_reveal_rows", m.forced_rows_by_kind[0])?;
    metrics.set_item("forced_great_library_rows", m.forced_rows_by_kind[1])?;
    metrics.set_item("forced_wonder_group_rows", m.forced_rows_by_kind[2])?;
    metrics.set_item("forced_age_deal_rows", m.forced_rows_by_kind[3])?;
    metrics.set_item("ordinary_leaf_rows", m.ordinary_leaf_rows)?;
    metrics.set_item("forced_cache_hits", m.forced_cache_hits)?;
    metrics.set_item("forced_rows_per_search", m.forced_rows_per_search.clone())?;
    metrics.set_item("max_batch_rows", m.max_batch_rows)?;
    metrics.set_item("scheduler_cycles", m.scheduler_cycles)?;
    metrics.set_item("scheduler_workers", m.scheduler_workers)?;
    metrics.set_item("max_inflight_batches", m.max_inflight_batches)?;
    metrics.set_item("boundary_tokens", m.boundary_tokens)?;
    metrics.set_item("boundary_padded_tokens", m.boundary_padded_tokens)?;
    metrics.set_item("boundary_max_tokens", m.boundary_max_tokens)?;
    metrics.set_item("encode_pack_ns", m.encode_pack_ns)?;
    metrics.set_item("queue_wait_ns", m.queue_wait_ns)?;
    metrics.set_item("py_call_ns", m.py_call_ns)?;
    metrics.set_item("extract_ns", m.extract_ns)?;
    metrics.set_item("rust_tree_ns", m.rust_tree_ns)?;
    metrics.set_item("rust_chance_ns", m.rust_chance_ns)?;
    metrics.set_item("rust_record_ns", m.rust_record_ns)?;
    metrics.set_item("scatter_ns", m.scatter_ns)?;
    // Phase 1a: scheduler-thread partition. Together with `scatter_ns` these
    // tile the loop, so `scheduler_wall_ns` minus their sum is loop bookkeeping.
    // `sched_collect_ns` is inclusive of rust_tree/rust_chance/encode_pack.
    metrics.set_item("sched_refill_ns", m.sched_refill_ns)?;
    metrics.set_item("sched_collect_ns", m.sched_collect_ns)?;
    metrics.set_item("sched_retire_ns", m.sched_retire_ns)?;
    metrics.set_item("sched_assemble_ns", m.sched_assemble_ns)?;
    metrics.set_item("sched_submit_ns", m.sched_submit_ns)?;
    metrics.set_item("sched_wait_ns", m.sched_wait_ns)?;
    metrics.set_item("scheduler_ready_slot_cycles", m.scheduler_ready_slot_cycles)?;
    metrics.set_item(
        "scheduler_waiting_slot_cycles",
        m.scheduler_waiting_slot_cycles,
    )?;
    metrics.set_item("scheduler_idle_slot_cycles", m.scheduler_idle_slot_cycles)?;
    // Phase 0: time-weighted occupancy. The `*_slot_cycles` counters above are
    // loop-iteration counts and must not be read as time shares.
    metrics.set_item("scheduler_wall_ns", m.scheduler_wall_ns)?;
    metrics.set_item("live_slot_ns", m.live_slot_ns)?;
    metrics.set_item("ready_slot_ns", m.ready_slot_ns)?;
    metrics.set_item("waiting_slot_ns", m.waiting_slot_ns)?;
    metrics.set_item("idle_slot_ns", m.idle_slot_ns)?;
    metrics.set_item("max_live_slots", m.max_live_slots)?;
    metrics.set_item("max_active_slots", m.max_active_slots)?;
    metrics.set_item(
        "time_weighted_live_slots",
        if m.scheduler_wall_ns == 0 {
            0.0
        } else {
            m.live_slot_ns as f64 / m.scheduler_wall_ns as f64
        },
    )?;
    metrics.set_item("batch_live_slots", m.batch_live_slots.clone())?;
    metrics.set_item("batch_submit_ns", m.batch_submit_ns.clone())?;
    metrics.set_item("arena_nodes_live_peak", m.arena_nodes_live_peak)?;
    metrics.set_item("arena_nodes_slot_peak", m.arena_nodes_slot_peak)?;
    metrics.set_item("arena_deep_bytes_slot_peak", m.arena_deep_bytes_slot_peak)?;
    metrics.set_item("arena_node_struct_bytes", m.arena_node_struct_bytes)?;
    metrics.set_item("wave_width_histogram", m.wave_width_histogram.to_vec())?;
    metrics.set_item("conflict_cuts", m.conflict_cuts)?;
    metrics.set_item(
        "mean_wave_width",
        {
            let waves: usize = m.wave_width_histogram.iter().sum();
            let paths: usize = m
                .wave_width_histogram
                .iter()
                .enumerate()
                .map(|(bucket, count)| (bucket + 1) * count)
                .sum();
            if waves == 0 {
                0.0
            } else {
                paths as f64 / waves as f64
            }
        },
    )?;
    metrics.set_item(
        "padding_ratio",
        if m.boundary_padded_tokens == 0 {
            0.0
        } else {
            1.0 - m.boundary_tokens as f64 / m.boundary_padded_tokens as f64
        },
    )?;
    metrics.set_item("batch_rows", m.batch_rows.clone())?;
    metrics.set_item(
        "mean_batch_rows",
        if m.batch_rows.is_empty() {
            0.0
        } else {
            m.batch_rows.iter().sum::<usize>() as f64 / m.batch_rows.len() as f64
        },
    )?;
    Ok((records, metrics.unbind()))
}

fn search_result_to_py(
    py: Python<'_>,
    result: tree::SearchResult,
    metrics: tree_resumable::SearchMetrics,
    digest: Vec<f64>,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("action", result.action_index)?;
    out.set_item("action_value", result.action_value)?;
    out.set_item("root_value", result.root_value)?;
    out.set_item("visits", result.visits)?;
    out.set_item("policy", result.policy_target)?;
    out.set_item("topk", result.gumbel_topk)?;
    out.set_item("sims", result.sims)?;
    out.set_item("completed_q", metrics.root_completed_q)?;
    out.set_item("survivors", metrics.halving_survivors.clone())?;
    out.set_item("digest", digest)?;
    let counters = PyDict::new(py);
    counters.set_item("scheduled", metrics.scheduled_simulations)?;
    counters.set_item("requested", metrics.requested_nn_leaves)?;
    counters.set_item("unique", metrics.unique_nn_leaves)?;
    counters.set_item("terminal", metrics.terminal_leaves)?;
    counters.set_item("collisions", metrics.collisions)?;
    counters.set_item("waves", metrics.leaf_waves)?;
    counters.set_item("max_wave_paths", metrics.max_wave_paths)?;
    counters.set_item("max_wave_unique", metrics.max_wave_unique)?;
    out.set_item("metrics", counters)?;
    let nn_work = PyDict::new(py);
    nn_work.set_item("forced_rows", metrics.forced_outcome_rows)?;
    nn_work.set_item("forced_cache_hits", metrics.cached_forced_leaves)?;
    // Evaluation and gate runs require exact chance; they can assert this is
    // zero rather than inspecting configuration plumbing by eye.
    nn_work.set_item("fixed_support_edges", metrics.fixed_support_edges)?;
    out.set_item("nn_work", nn_work)?;
    Ok(out.unbind())
}

/// F4.6 position-calibration primitive: run many independent root searches
/// through one flat global evaluator instead of paying a scalar Python hop for
/// every leaf. Results remain in input order and use the same resumable search.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    adapter, games, search_seeds, global_batch_cap, leaf_batch, sims, top_k,
    c_puct=1.5, c_visit=50.0, c_scale=0.1, force=false,
    age_deal_samples=0, inference_timeout_ms=0.0, puct_root=false,
    double_reveal_offsets=0, conflict_free_waves=false, round_robin_candidates=false
))]
fn search_many_flat_net(
    py: Python<'_>,
    adapter: Py<PyAny>,
    games: Vec<Py<RustGame>>,
    search_seeds: Vec<u64>,
    global_batch_cap: usize,
    leaf_batch: usize,
    sims: usize,
    top_k: usize,
    c_puct: f64,
    c_visit: f64,
    c_scale: f64,
    force: bool,
    age_deal_samples: usize,
    inference_timeout_ms: f64,
    puct_root: bool,
    double_reveal_offsets: usize,
    conflict_free_waves: bool,
    round_robin_candidates: bool,
) -> PyResult<Vec<Py<PyDict>>> {
    if games.is_empty() || games.len() != search_seeds.len() {
        return Err(PyValueError::new_err(
            "games and search_seeds must be non-empty and aligned",
        ));
    }
    if global_batch_cap == 0 || leaf_batch == 0 || sims == 0 || top_k == 0 {
        return Err(PyValueError::new_err(
            "global_batch_cap, leaf_batch, sims, and top_k must be positive",
        ));
    }
    if leaf_batch > global_batch_cap {
        return Err(PyValueError::new_err(format!(
            "leaf_batch={leaf_batch} exceeds global_batch_cap={global_batch_cap}"
        )));
    }
    let states: Vec<GameState> = games
        .iter()
        .map(|game| game.borrow(py).state.clone())
        .collect();
    let (worker, timed_out, _boundary_metrics, worker_handle) =
        eval::spawn_py_flat_worker(adapter, inference_timeout_ms, global_batch_cap)?;
    let outputs = py.detach(move || {
        let state_refs: Vec<&GameState> = states.iter().collect();
        let actors: Vec<_> = states.iter().map(tree::state_actor).collect();
        let legals: Vec<_> = states.iter().map(codec::legal_action_indices).collect();
        let roots = worker.evaluate_batch_prepared(&state_refs, &actors, &legals)?;
        let mut sessions = Vec::with_capacity(states.len());
        for ((state, seed), evaluation) in states.iter().zip(search_seeds).zip(roots) {
            let cfg = tree::SearchConfig {
                sims,
                top_k,
                c_puct,
                c_visit,
                c_scale,
                seed,
                force_expand_root_chance: force,
                puct_root,
                age_deal_samples,
                double_reveal_offsets,
                conflict_free_waves,
                round_robin_candidates,
            };
            let session = if force {
                tree_resumable::begin_search_from_root_forced(state, &cfg, leaf_batch, evaluation)?
            } else {
                tree_resumable::begin_search_from_root(state, &cfg, leaf_batch, evaluation)?
            };
            sessions.push(Some(session));
        }

        struct Group {
            slot: usize,
            request: tree_resumable::EvalBatchRequest,
            states: Vec<GameState>,
            actors: Vec<usize>,
            legals: Vec<Vec<usize>>,
        }

        let mut completed: Vec<
            Option<(tree::SearchResult, tree_resumable::SearchMetrics, Vec<f64>)>,
        > = (0..sessions.len()).map(|_| None).collect();
        while completed.iter().any(Option::is_none) {
            let mut groups = VecDeque::new();
            let live = sessions
                .iter()
                .filter(|session| session.is_some())
                .count()
                .max(1);
            let forced_row_limit = (global_batch_cap / live).max(1);
            for slot in 0..sessions.len() {
                let Some(session) = sessions[slot].as_mut() else {
                    continue;
                };
                match session.next_event_with_limit(forced_row_limit) {
                    Ok(tree_resumable::SearchEvent::Evaluation(request)) => {
                        let states = match session.evaluation_states(&request) {
                            Ok(rows) => rows.into_iter().cloned().collect(),
                            Err(err) => {
                                session.cancel_pending();
                                return Err(err);
                            }
                        };
                        groups.push_back(Group {
                            slot,
                            actors: request.leaves.iter().map(|leaf| leaf.actor).collect(),
                            legals: request
                                .leaves
                                .iter()
                                .map(|leaf| leaf.legal.clone())
                                .collect(),
                            request,
                            states,
                        });
                    }
                    Ok(tree_resumable::SearchEvent::Complete) => {
                        let session = sessions[slot].take().expect("session must exist");
                        let (result, arena, metrics) = session.into_result()?;
                        let mut digest = Vec::new();
                        tree_resumable::digest(&arena, &mut digest);
                        completed[slot] = Some((result, metrics, digest));
                    }
                    Err(err) => {
                        session.cancel_pending();
                        return Err(err);
                    }
                }
            }
            while !groups.is_empty() {
                let mut batch = Vec::new();
                let mut rows = 0;
                while let Some(group) = groups.front() {
                    let count = group.states.len();
                    if count > global_batch_cap {
                        for session in sessions.iter_mut().flatten() {
                            session.cancel_pending();
                        }
                        return Err(PyValueError::new_err(format!(
                            "search leaf wave has {count} rows above global cap {global_batch_cap}"
                        )));
                    }
                    if !batch.is_empty() && rows + count > global_batch_cap {
                        break;
                    }
                    rows += count;
                    batch.push(groups.pop_front().expect("front group must exist"));
                }
                let owned: Vec<_> = batch
                    .iter()
                    .flat_map(|group| group.states.iter().cloned())
                    .collect();
                let actors: Vec<_> = batch
                    .iter()
                    .flat_map(|group| group.actors.iter().copied())
                    .collect();
                let legals: Vec<_> = batch
                    .iter()
                    .flat_map(|group| group.legals.iter().cloned())
                    .collect();
                let evaluations = match worker
                    .submit_prepared(owned, actors, legals)
                    .and_then(|ticket| ticket.wait())
                {
                    Ok(rows) => rows,
                    Err(err) => {
                        for session in sessions.iter_mut().flatten() {
                            session.cancel_pending();
                        }
                        return Err(err);
                    }
                };
                let mut cursor = 0;
                for group in batch {
                    let count = group.states.len();
                    let result = sessions[group.slot]
                        .as_mut()
                        .expect("search session must exist")
                        .apply_evaluations(
                            group.request.request_id,
                            evaluations[cursor..cursor + count].to_vec(),
                        );
                    cursor += count;
                    if let Err(err) = result {
                        for session in sessions.iter_mut().flatten() {
                            session.cancel_pending();
                        }
                        return Err(err);
                    }
                }
            }
        }
        drop(worker);
        if timed_out.load(std::sync::atomic::Ordering::Acquire) {
            drop(worker_handle);
        } else if worker_handle.join().is_err() {
            return Err(PyRuntimeError::new_err(
                "flat search inference worker panicked during shutdown",
            ));
        }
        Ok(completed
            .into_iter()
            .map(|row| row.expect("all searches must complete"))
            .collect::<Vec<_>>())
    })?;
    outputs
        .into_iter()
        .map(|(result, metrics, digest)| search_result_to_py(py, result, metrics, digest))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn cooperative_jobs(
    py: Python<'_>,
    games: &[Py<RustGame>],
    game_seeds: &[u64],
    iteration: Option<i64>,
    leaf_batch: usize,
    cheap_sims_min: usize,
    cheap_sims_max: usize,
    full_sims_min: usize,
    full_sims_max: usize,
    full_search_fraction: f64,
    top_k: usize,
    draft_prior: f64,
    c_puct: f64,
    c_visit: f64,
    c_scale: f64,
    force: bool,
    puct_root: bool,
    age_deal_samples: usize,
    cheap_double_reveal_offsets: usize,
    max_moves: usize,
    conflict_free_waves: bool,
    round_robin_candidates: bool,
) -> PyResult<Vec<(GameState, self_play::SelfPlayConfig)>> {
    if games.len() != game_seeds.len() {
        return Err(PyValueError::new_err(format!(
            "received {} games but {} game seeds",
            games.len(),
            game_seeds.len()
        )));
    }
    games
        .iter()
        .zip(game_seeds)
        .map(|(game, &game_seed)| {
            let state = game.borrow(py).state.clone();
            let cfg = make_self_play_config(
                game_seed,
                iteration,
                leaf_batch,
                cheap_sims_min,
                cheap_sims_max,
                full_sims_min,
                full_sims_max,
                full_search_fraction,
                top_k,
                draft_prior,
                c_puct,
                c_visit,
                c_scale,
                force,
                puct_root,
                age_deal_samples,
                cheap_double_reveal_offsets,
                max_moves,
                conflict_free_waves,
                round_robin_candidates,
            );
            Ok((state, cfg))
        })
        .collect()
}

/// F4.4 deterministic cooperative scheduler under the mock evaluator.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    games, game_seeds, global_batch_cap, leaf_batch, cheap_sims_min,
    cheap_sims_max, full_sims_min, full_sims_max, full_search_fraction, top_k,
    draft_prior, iteration=None, c_puct=1.5, c_visit=50.0, c_scale=0.1,
    force=false, age_deal_samples=0, cheap_double_reveal_offsets=0, max_moves=256,
    cheap_double_reveal_offsets_p0=None, cheap_double_reveal_offsets_p1=None,
    max_active_slots=0, conflict_free_waves=false, round_robin_candidates=false
))]
fn self_play_many_mock(
    py: Python<'_>,
    games: Vec<Py<RustGame>>,
    game_seeds: Vec<u64>,
    global_batch_cap: usize,
    leaf_batch: usize,
    cheap_sims_min: usize,
    cheap_sims_max: usize,
    full_sims_min: usize,
    full_sims_max: usize,
    full_search_fraction: f64,
    top_k: usize,
    draft_prior: f64,
    iteration: Option<i64>,
    c_puct: f64,
    c_visit: f64,
    c_scale: f64,
    force: bool,
    age_deal_samples: usize,
    cheap_double_reveal_offsets: usize,
    max_moves: usize,
    cheap_double_reveal_offsets_p0: Option<usize>,
    cheap_double_reveal_offsets_p1: Option<usize>,
    max_active_slots: usize,
    conflict_free_waves: bool,
    round_robin_candidates: bool,
) -> PyResult<(Vec<Py<PyDict>>, Py<PyDict>)> {
    let jobs = cooperative_jobs(
        py,
        &games,
        &game_seeds,
        iteration,
        leaf_batch,
        cheap_sims_min,
        cheap_sims_max,
        full_sims_min,
        full_sims_max,
        full_search_fraction,
        top_k,
        draft_prior,
        c_puct,
        c_visit,
        c_scale,
        force,
        false, // self-play always uses the Gumbel root
        age_deal_samples,
        cheap_double_reveal_offsets,
        max_moves,
        conflict_free_waves,
        round_robin_candidates,
    )?;
    let mut jobs = jobs;
    match (cheap_double_reveal_offsets_p0, cheap_double_reveal_offsets_p1) {
        (None, None) => {}
        (Some(p0), Some(p1)) => {
            for (_, cfg) in &mut jobs {
                cfg.cheap_double_reveal_offsets_by_player = Some([p0, p1]);
            }
        }
        _ => {
            return Err(PyValueError::new_err(
                "cheap_double_reveal_offsets_p0 and _p1 must be supplied together",
            ));
        }
    }
    let result = py.detach(move || {
        self_play::run_many(jobs, &eval::MockEval, global_batch_cap, max_active_slots)
    })?;
    scheduler_result_to_py(py, result)
}

/// F4.4 global Python evaluator boundary. `adapter(rows)` is called once per
/// global batch; each row is `(tokens, actor, legal)` and results stay aligned.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    adapter, games, game_seeds, global_batch_cap, leaf_batch, cheap_sims_min,
    cheap_sims_max, full_sims_min, full_sims_max, full_search_fraction, top_k,
    draft_prior, iteration=None, c_puct=1.5, c_visit=50.0, c_scale=0.1,
    force=false, age_deal_samples=0, cheap_double_reveal_offsets=0, max_moves=256, inference_timeout_ms=0.0,
    max_inflight_batches=2, scheduler_workers=1, leaf_batch_p0=None, leaf_batch_p1=None,
    age_deal_samples_p0=None, age_deal_samples_p1=None, deterministic_actions=false,
    cheap_double_reveal_offsets_p0=None, cheap_double_reveal_offsets_p1=None,
    max_active_slots=0, conflict_free_waves=false, round_robin_candidates=false
))]
fn self_play_many_net(
    py: Python<'_>,
    adapter: Py<PyAny>,
    games: Vec<Py<RustGame>>,
    game_seeds: Vec<u64>,
    global_batch_cap: usize,
    leaf_batch: usize,
    cheap_sims_min: usize,
    cheap_sims_max: usize,
    full_sims_min: usize,
    full_sims_max: usize,
    full_search_fraction: f64,
    top_k: usize,
    draft_prior: f64,
    iteration: Option<i64>,
    c_puct: f64,
    c_visit: f64,
    c_scale: f64,
    force: bool,
    age_deal_samples: usize,
    cheap_double_reveal_offsets: usize,
    max_moves: usize,
    inference_timeout_ms: f64,
    max_inflight_batches: usize,
    scheduler_workers: usize,
    leaf_batch_p0: Option<usize>,
    leaf_batch_p1: Option<usize>,
    age_deal_samples_p0: Option<usize>,
    age_deal_samples_p1: Option<usize>,
    deterministic_actions: bool,
    cheap_double_reveal_offsets_p0: Option<usize>,
    cheap_double_reveal_offsets_p1: Option<usize>,
    max_active_slots: usize,
    conflict_free_waves: bool,
    round_robin_candidates: bool,
) -> PyResult<(Vec<Py<PyDict>>, Py<PyDict>)> {
    let mut jobs = cooperative_jobs(
        py,
        &games,
        &game_seeds,
        iteration,
        leaf_batch,
        cheap_sims_min,
        cheap_sims_max,
        full_sims_min,
        full_sims_max,
        full_search_fraction,
        top_k,
        draft_prior,
        c_puct,
        c_visit,
        c_scale,
        force,
        false, // self-play always uses the Gumbel root
        age_deal_samples,
        cheap_double_reveal_offsets,
        max_moves,
        conflict_free_waves,
        round_robin_candidates,
    )?;
    match (leaf_batch_p0, leaf_batch_p1) {
        (None, None) => {}
        (Some(p0), Some(p1)) if p0 > 0 && p1 > 0 => {
            for (_, cfg) in &mut jobs {
                cfg.leaf_batch_by_player = Some([p0, p1]);
                cfg.deterministic_actions = deterministic_actions;
            }
        }
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "leaf_batch_p0 and leaf_batch_p1 must be positive",
            ));
        }
        _ => {
            return Err(PyValueError::new_err(
                "leaf_batch_p0 and leaf_batch_p1 must be supplied together",
            ));
        }
    }
    if leaf_batch_p0.is_none() {
        for (_, cfg) in &mut jobs {
            cfg.deterministic_actions = deterministic_actions;
        }
    }
    match (cheap_double_reveal_offsets_p0, cheap_double_reveal_offsets_p1) {
        (None, None) => {}
        (Some(p0), Some(p1)) => {
            // Seat-mirrored search-strength arena: capped versus exhaustive on
            // one shared net.
            for (_, cfg) in &mut jobs {
                cfg.cheap_double_reveal_offsets_by_player = Some([p0, p1]);
            }
        }
        _ => {
            return Err(PyValueError::new_err(
                "cheap_double_reveal_offsets_p0 and _p1 must be supplied together",
            ));
        }
    }
    match (age_deal_samples_p0, age_deal_samples_p1) {
        (None, None) => {}
        (Some(p0), Some(p1)) if p0 <= 32 && p1 <= 32 => {
            for (_, cfg) in &mut jobs {
                cfg.age_deal_samples_by_player = Some([p0, p1]);
            }
        }
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "age_deal_samples_p0 and age_deal_samples_p1 cannot exceed 32",
            ));
        }
        _ => {
            return Err(PyValueError::new_err(
                "age_deal_samples_p0 and age_deal_samples_p1 must be supplied together",
            ));
        }
    }
    let (worker, timed_out, worker_handle) =
        eval::spawn_py_batch_worker(adapter, inference_timeout_ms, global_batch_cap)?;
    let result = py.detach(move || {
        let result = self_play::run_many_pipelined_sharded(
            jobs,
            &worker,
            global_batch_cap,
            max_inflight_batches,
            scheduler_workers,
            max_active_slots,
        );
        drop(worker);
        if timed_out.load(std::sync::atomic::Ordering::Acquire) {
            // Rust cannot safely kill a Python/Torch call. Detach the timed-out
            // worker so every scheduler slot wakes immediately; the worker owns
            // no slot/search state and exits when the adapter call returns.
            drop(worker_handle);
            return result;
        }
        if worker_handle.join().is_err() {
            return Err(PyRuntimeError::new_err(
                "global inference worker panicked during shutdown",
            ));
        }
        result
    })?;
    scheduler_result_to_py(py, result)
}

/// F4.5 production-shaped flat transformer boundary. The adapter receives one
/// dictionary of packed byte buffers and returns only actor value plus priors
/// aligned to the packed legal-action rows.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    adapter, games, game_seeds, global_batch_cap, leaf_batch, cheap_sims_min,
    cheap_sims_max, full_sims_min, full_sims_max, full_search_fraction, top_k,
    draft_prior, iteration=None, c_puct=1.5, c_visit=50.0, c_scale=0.1,
    force=false, age_deal_samples=0, cheap_double_reveal_offsets=0, max_moves=256, inference_timeout_ms=0.0,
    max_inflight_batches=2, scheduler_workers=1, leaf_batch_p0=None, leaf_batch_p1=None,
    age_deal_samples_p0=None, age_deal_samples_p1=None, deterministic_actions=false,
    bot_p0=None, bot_p1=None, bots_p0=None, bots_p1=None,
    nets_p0=None, nets_p1=None,
    bot_exploration=0.0, bot_policy_iterations=10
, puct_root=false, cheap_double_reveal_offsets_p0=None,
    cheap_double_reveal_offsets_p1=None, max_active_slots=0,
    conflict_free_waves=false, round_robin_candidates=false))]
fn self_play_many_flat_net(
    py: Python<'_>,
    adapter: Py<PyAny>,
    games: Vec<Py<RustGame>>,
    game_seeds: Vec<u64>,
    global_batch_cap: usize,
    leaf_batch: usize,
    cheap_sims_min: usize,
    cheap_sims_max: usize,
    full_sims_min: usize,
    full_sims_max: usize,
    full_search_fraction: f64,
    top_k: usize,
    draft_prior: f64,
    iteration: Option<i64>,
    c_puct: f64,
    c_visit: f64,
    c_scale: f64,
    force: bool,
    age_deal_samples: usize,
    cheap_double_reveal_offsets: usize,
    max_moves: usize,
    inference_timeout_ms: f64,
    max_inflight_batches: usize,
    scheduler_workers: usize,
    leaf_batch_p0: Option<usize>,
    leaf_batch_p1: Option<usize>,
    age_deal_samples_p0: Option<usize>,
    age_deal_samples_p1: Option<usize>,
    deterministic_actions: bool,
    bot_p0: Option<String>,
    bot_p1: Option<String>,
    bots_p0: Option<Vec<Option<String>>>,
    bots_p1: Option<Vec<Option<String>>>,
    // W1.3 league play: per-game network id for each seat. `None` (the
    // default) means one network for every game, which packs no ids at all.
    nets_p0: Option<Vec<u8>>,
    nets_p1: Option<Vec<u8>>,
    bot_exploration: f64,
    bot_policy_iterations: i64,
    puct_root: bool,
    cheap_double_reveal_offsets_p0: Option<usize>,
    cheap_double_reveal_offsets_p1: Option<usize>,
    max_active_slots: usize,
    conflict_free_waves: bool,
    round_robin_candidates: bool,
) -> PyResult<(Vec<Py<PyDict>>, Py<PyDict>)> {
    let mut jobs = cooperative_jobs(
        py,
        &games,
        &game_seeds,
        iteration,
        leaf_batch,
        cheap_sims_min,
        cheap_sims_max,
        full_sims_min,
        full_sims_max,
        full_search_fraction,
        top_k,
        draft_prior,
        c_puct,
        c_visit,
        c_scale,
        force,
        puct_root,
        age_deal_samples,
        cheap_double_reveal_offsets,
        max_moves,
        conflict_free_waves,
        round_robin_candidates,
    )?;
    match (leaf_batch_p0, leaf_batch_p1) {
        (None, None) => {}
        (Some(p0), Some(p1)) if p0 > 0 && p1 > 0 => {
            for (_, cfg) in &mut jobs {
                cfg.leaf_batch_by_player = Some([p0, p1]);
            }
        }
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "leaf_batch_p0 and leaf_batch_p1 must be positive",
            ));
        }
        _ => {
            return Err(PyValueError::new_err(
                "leaf_batch_p0 and leaf_batch_p1 must be supplied together",
            ));
        }
    }
    let parse_bot = |name: Option<&str>| -> PyResult<Option<bots::BotKind>> {
        name.map(|value| {
            bots::BotKind::parse(value)
                .ok_or_else(|| PyValueError::new_err(format!("unknown Rust bot: {value}")))
        })
        .transpose()
    };
    // Bots are a per-game property (`SelfPlayConfig::bot_by_player`), but the
    // scalar form broadcasts one assignment over the whole call. That forces a
    // caller with a mix -- Phase D self-play is ~15% curriculum-bot games split
    // across (bot type, seat) -- to issue one call per combination, and a call
    // of three games drains a slot pool of forty-eight and sends three-row
    // batches into a boundary whose cost is almost entirely fixed per batch.
    //
    // `bots_p0`/`bots_p1` take one entry per game instead, so such a caller can
    // put every game in one call and let the scheduler interleave them. The
    // scalar form stays for callers that genuinely have one configuration for
    // the call (the curriculum seed buffer, the arena and evaluation paths).
    let per_game_bots = match (&bots_p0, &bots_p1) {
        (None, None) => None,
        (Some(p0), Some(p1)) => {
            if bot_p0.is_some() || bot_p1.is_some() {
                return Err(PyValueError::new_err(
                    "bots_p0/bots_p1 replace bot_p0/bot_p1; supply one form, not both",
                ));
            }
            if p0.len() != jobs.len() || p1.len() != jobs.len() {
                return Err(PyValueError::new_err(format!(
                    "bots_p0 ({}) and bots_p1 ({}) must have one entry per game ({})",
                    p0.len(),
                    p1.len(),
                    jobs.len()
                )));
            }
            let mut parsed = Vec::with_capacity(jobs.len());
            for (left, right) in p0.iter().zip(p1.iter()) {
                parsed.push([
                    parse_bot(left.as_deref())?,
                    parse_bot(right.as_deref())?,
                ]);
            }
            Some(parsed)
        }
        _ => {
            return Err(PyValueError::new_err(
                "bots_p0 and bots_p1 must be supplied together",
            ));
        }
    };
    // Per-game network assignment, same shape as `bots_p0`/`bots_p1` so league
    // games and ordinary self-play share one scheduler call. Validated here
    // rather than trusted: an out-of-range id would silently index the wrong
    // model in the Python adapter.
    let per_game_nets = match (&nets_p0, &nets_p1) {
        (None, None) => None,
        (Some(p0), Some(p1)) => {
            if p0.len() != jobs.len() || p1.len() != jobs.len() {
                return Err(PyValueError::new_err(format!(
                    "nets_p0 ({}) and nets_p1 ({}) must have one entry per game ({})",
                    p0.len(),
                    p1.len(),
                    jobs.len()
                )));
            }
            if p0.iter().chain(p1.iter()).any(|&id| id > 1) {
                return Err(PyValueError::new_err(
                    "network ids must be 0 or 1",
                ));
            }
            Some(
                p0.iter()
                    .zip(p1.iter())
                    .map(|(&left, &right)| [left, right])
                    .collect::<Vec<_>>(),
            )
        }
        _ => {
            return Err(PyValueError::new_err(
                "nets_p0 and nets_p1 must be supplied together",
            ));
        }
    };
    let bot_by_player = [parse_bot(bot_p0.as_deref())?, parse_bot(bot_p1.as_deref())?];
    for (index, (_, cfg)) in jobs.iter_mut().enumerate() {
        cfg.deterministic_actions = deterministic_actions;
        cfg.bot_by_player = match &per_game_bots {
            Some(bots) => bots[index],
            None => bot_by_player,
        };
        cfg.net_by_player = match &per_game_nets {
            Some(nets) => nets[index],
            None => [0, 0],
        };
        cfg.bot_exploration = bot_exploration;
        cfg.bot_policy_iterations = bot_policy_iterations;
    }
    match (cheap_double_reveal_offsets_p0, cheap_double_reveal_offsets_p1) {
        (None, None) => {}
        (Some(p0), Some(p1)) => {
            // Seat-mirrored search-strength arena: capped versus exhaustive on
            // one shared net.
            for (_, cfg) in &mut jobs {
                cfg.cheap_double_reveal_offsets_by_player = Some([p0, p1]);
            }
        }
        _ => {
            return Err(PyValueError::new_err(
                "cheap_double_reveal_offsets_p0 and _p1 must be supplied together",
            ));
        }
    }
    match (age_deal_samples_p0, age_deal_samples_p1) {
        (None, None) => {}
        (Some(p0), Some(p1)) if p0 <= 32 && p1 <= 32 => {
            for (_, cfg) in &mut jobs {
                cfg.age_deal_samples_by_player = Some([p0, p1]);
            }
        }
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "age_deal_samples_p0 and age_deal_samples_p1 cannot exceed 32",
            ));
        }
        _ => {
            return Err(PyValueError::new_err(
                "age_deal_samples_p0 and age_deal_samples_p1 must be supplied together",
            ));
        }
    }
    let (worker, timed_out, boundary_metrics, worker_handle) =
        eval::spawn_py_flat_worker(adapter, inference_timeout_ms, global_batch_cap)?;
    let result = py.detach(move || {
        let mut result = self_play::run_many_pipelined_sharded(
            jobs,
            &worker,
            global_batch_cap,
            max_inflight_batches,
            scheduler_workers,
            max_active_slots,
        );
        drop(worker);
        if timed_out.load(std::sync::atomic::Ordering::Acquire) {
            drop(worker_handle);
            return result;
        }
        if worker_handle.join().is_err() {
            return Err(PyRuntimeError::new_err(
                "flat inference worker panicked during shutdown",
            ));
        }
        if let Ok(output) = &mut result {
            let counters = boundary_metrics
                .lock()
                .map_err(|_| PyRuntimeError::new_err("boundary metrics lock poisoned"))?
                .clone();
            output.metrics.boundary_tokens = counters.tokens;
            output.metrics.boundary_padded_tokens = counters.padded_tokens;
            output.metrics.boundary_max_tokens = counters.max_tokens;
            output.metrics.encode_pack_ns = counters.encode_pack_ns;
            output.metrics.queue_wait_ns = counters.queue_wait_ns;
            output.metrics.py_call_ns = counters.py_call_ns;
            output.metrics.extract_ns = counters.extract_ns;
        }
        result
    })?;
    scheduler_result_to_py(py, result)
}

#[pymodule]
mod seven_wonders_rust {
    #[pymodule_export]
    use super::RustGame;

    #[pymodule_export]
    use super::RustPuctSearch;

    #[pymodule_export]
    use super::num_actions;

    #[pymodule_export]
    use super::encoder_signature;

    #[pymodule_export]
    use super::gumbel_stream;

    #[pymodule_export]
    use super::ln_values;

    #[pymodule_export]
    use super::self_play_many_mock;

    #[pymodule_export]
    use super::self_play_many_net;

    #[pymodule_export]
    use super::self_play_many_flat_net;

    #[pymodule_export]
    use super::search_many_flat_net;
}

#[cfg(test)]
mod tests {
    //! Rust-side smoke coverage so `cargo test` is load-bearing independent of
    //! the Python gate. The exhaustive cross-language replay lives in
    //! `test_rust_engine_equiv.py`; these lock crate invariants and the F1b
    //! make/unmake audit on a self-contained setup (valid-sized, not shuffled;
    //! enough to drive the draft and Age I, which is where undo is exercised).

    use crate::codec;
    use crate::engine::make_unmake_audit;
    use crate::state::{GameState, Phase, Setup};
    use std::collections::VecDeque;

    fn sample_setup() -> Setup {
        Setup {
            first_player: 0,
            available_progress_tokens: vec![0, 1, 2, 3, 4],
            unused_progress_tokens: vec![5, 6, 7, 8, 9],
            wonder_groups: [vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
            unused_wonders: vec![8, 9, 10, 11],
            age_decks: [
                Vec::new(),
                (0..20).collect(),
                (0..20).collect(),
                (0..20).collect(),
            ],
            removed_age_cards: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            selected_guilds: Vec::new(),
            unused_guilds: Vec::new(),
        }
    }

    #[test]
    fn action_space_is_1202() {
        assert_eq!(codec::NUM_ACTIONS, 1202);
    }

    #[test]
    fn the_chance_witness_catches_a_wrong_prediction() {
        //! `apply_with_chance` cannot detect a mispredicted signature by
        //! counting outcomes — the caller derived them from that same
        //! signature. The witness compares the prediction against what the
        //! engine actually left behind, so it has to fire in BOTH directions
        //! or it is decoration.

        use crate::chance::{chance_signature, ChanceKind, ChanceSpec};
        use crate::engine::ChanceWitness;

        let mut g = GameState::from_setup(sample_setup(), VecDeque::new());
        while g.phase != Phase::PlayAge {
            let legal = codec::legal_action_indices(&g);
            g.apply_action(&codec::decode_action(&g, legal[0]));
        }
        // Play on until some take exposes a face-down card, so there is a real
        // event to under-predict. At the start of Age I nothing is exposed by a
        // single take: both coverers of a face-down slot are still present.
        let index = loop {
            let legal = codec::legal_action_indices(&g);
            if let Some(&i) = legal
                .iter()
                .find(|&&i| !chance_signature(&g, &codec::decode_action(&g, i)).is_empty())
            {
                break i;
            }
            assert_eq!(g.phase, Phase::PlayAge, "left Age I without exposing a card");
            g.apply_action(&codec::decode_action(&g, legal[0]));
        };
        let action = codec::decode_action(&g, index);
        let specs = chance_signature(&g, &action);

        let witness = ChanceWitness::of(&g);
        g.apply_action(&action);

        witness
            .check(&g, &specs)
            .expect("the true signature must agree with the engine");

        let mut over = specs.clone();
        over.push(ChanceSpec {
            kind: ChanceKind::AgeDeal,
            context: vec![2],
        });
        assert!(
            witness.check(&g, &over).is_err(),
            "an AGE_DEAL that never fired must be caught -- it has already \
             rewritten age_decks by this point"
        );

        assert!(
            witness.check(&g, &[]).is_err(),
            "a CARD_REVEAL the engine fired unpredicted must be caught"
        );

        // The draw-queue arm, which the first version of this test missed and
        // the Python equivalence gate caught: the length is only an invariant
        // when the witness is taken BEFORE the outcome is pushed on. Reproduce
        // that shape -- an outcome pre-installed for an event that never fires.
        let queued = ChanceWitness::of(&g);
        g.library_draws.push_front(vec![0, 1, 2]);
        assert!(
            queued
                .check(
                    &g,
                    &[ChanceSpec {
                        kind: ChanceKind::GreatLibraryDraw,
                        context: vec![],
                    }],
                )
                .is_err(),
            "a GREAT_LIBRARY_DRAW that never fired must be caught -- its \
             pre-installed outcome is still sitting on the queue"
        );
    }

    #[test]
    fn fingerprint_deterministic_and_clone_equal() {
        let g = GameState::from_setup(sample_setup(), VecDeque::new());
        assert_eq!(g.fingerprint(), g.fingerprint());
        assert!(g.clone() == g);
    }

    #[test]
    fn encoder_feature_counts_match_schema() {
        use crate::encoder::{encode, FEATURE_COUNTS};
        let mut g = GameState::from_setup(sample_setup(), VecDeque::new());
        let mut steps = 0;
        while g.phase != Phase::Complete && steps < 14 {
            for t in encode(&g) {
                assert_eq!(
                    t.features.len(),
                    FEATURE_COUNTS[t.type_id],
                    "token type {} feature count",
                    t.type_id
                );
            }
            let legal = codec::legal_action_indices(&g);
            g.apply_action(&codec::decode_action(&g, legal[0]));
            steps += 1;
        }
        assert!(steps > 8);
    }

    #[test]
    fn make_unmake_audit_holds_through_age_one() {
        let mut g = GameState::from_setup(sample_setup(), VecDeque::new());
        // Draft branch is <= 4 wide, so a depth-2 (nested) audit is cheap here.
        make_unmake_audit(&g, 2).expect("draft make/unmake");
        // Drive the 8 draft picks and a few Age I decisions, auditing the full
        // legal fan-out (depth 1) at each live state.
        let mut steps = 0;
        while g.phase != Phase::Complete && steps < 14 {
            make_unmake_audit(&g, 1).expect("live-state make/unmake");
            let legal = codec::legal_action_indices(&g);
            g.apply_action(&codec::decode_action(&g, legal[0]));
            steps += 1;
        }
        assert!(steps > 8, "test should reach Age I play, got {steps} steps");
    }
}
