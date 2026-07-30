from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from games.seven_wonders_duel.buffer import read_records, replay
from games.seven_wonders_duel.bots import GreedyBot
from games.seven_wonders_duel.codec import legal_action_indices
from games.seven_wonders_duel.data import WONDERS_BY_NAME
from games.seven_wonders_duel.game import new_game
from games.seven_wonders_duel.inference import Evaluation
from games.seven_wonders_duel.loop_inference import CoalescingEvaluator
from games.seven_wonders_duel.phase_d import (
    CURRICULUM_BOT_TYPES,
    BotAgentSpec,
    GateResult,
    PhaseDConfig,
    PhaseDLoop,
    WONDER_DRAFT_TIERS,
    _self_play_game,
    ResolvedSchedules,
    blend_draft_priors,
    curriculum_fraction,
    filter_warm_records_by_staleness,
    generate_seed_buffer,
    resolve_anneal_iterations,
    UnderpoweredGateWarning,
    should_run_anchor_gate,
    temperature_for_move,
)
from games.seven_wonders_duel import phase_d as phase_d_module
from games.seven_wonders_duel.phase_d import _write_records
from games.seven_wonders_duel.train import stable_game_split, stable_is_validation
from games.az_loop import GameJob, expected_games_to_decide

# Most tests here run deliberately degenerate gates (2-4 games) to keep
# them fast; the underpowered-gate warning is correct there and only
# noise.  The test that asserts the warning fires opts back in locally.
pytestmark = pytest.mark.filterwarnings(
    "ignore::games.seven_wonders_duel.phase_d.UnderpoweredGateWarning"
)


class UniformEvaluator:
    def __init__(self):
        self.batch_sizes = []

    def evaluate(self, encodings, legal_lists):
        self.batch_sizes.append(len(encodings))
        return [
            Evaluation(
                policy=np.full(len(legal), 1.0 / len(legal), dtype=np.float32),
                wdl=np.asarray([0.4, 0.2, 0.4], dtype=np.float32),
                joint7=np.full(7, 1.0 / 7.0, dtype=np.float32),
                margin=0.0,
                military=0.0,
                science=np.zeros(2, dtype=np.float32),
            )
            for legal in legal_lists
        ]


def test_curriculum_and_temperature_schedules_anneal_to_zero_and_quarter():
    assert curriculum_fraction(0.2, 0, 10) == 0.2
    assert curriculum_fraction(0.2, 5, 10) == 0.1
    assert curriculum_fraction(0.2, 10, 10) == 0.0
    assert temperature_for_move(0) == 1.0
    assert temperature_for_move(20) == 0.25
    assert temperature_for_move(100) == 0.25


def test_draft_prior_is_normalized_and_favors_best_offered_wonder():
    game = new_game(4)
    legal = legal_action_indices(game)
    neural = {index: 1.0 / len(legal) for index in legal}
    blended = blend_draft_priors(game, neural, 1.0)
    assert abs(sum(blended.values()) - 1.0) < 1e-9
    assert set(blended) == set(legal)
    assert max(blended.values()) > min(blended.values())


def test_draft_prior_exactly_matches_locked_zeusai_tiers():
    assert set(WONDER_DRAFT_TIERS) == set(WONDERS_BY_NAME)
    extra_turn = {
        "The Temple of Artemis",
        "Piraeus",
        "The Hanging Gardens",
        "The Appian Way",
        "The Sphinx",
    }
    assert {WONDER_DRAFT_TIERS[name] for name in extra_turn} == {1.0}
    assert WONDER_DRAFT_TIERS["The Statue of Zeus"] == 0.8
    assert WONDER_DRAFT_TIERS["The Great Library"] == 0.8
    assert WONDER_DRAFT_TIERS["The Mausoleum"] == 0.6
    assert WONDER_DRAFT_TIERS["Circus Maximus"] == 0.6
    assert WONDER_DRAFT_TIERS["The Colossus"] == 0.6
    assert WONDER_DRAFT_TIERS["The Great Lighthouse"] == 0.4
    assert WONDER_DRAFT_TIERS["The Pyramids"] == 0.0


@dataclass(frozen=True)
class SplitExample:
    iteration: int | None
    game_key: int
    move: int


def _split_population(iterations=(0, 1), games=40):
    return [
        SplitExample(iteration, iteration * 1000 + game, move)
        for iteration in iterations
        for game in range(games)
        for move in range(2)
    ]


def test_stable_split_trains_on_fresh_games_without_game_leakage():
    examples = _split_population()
    curriculum = [SplitExample(None, 10_000 + game, 0) for game in range(3)]
    train, validation = stable_game_split(examples + curriculum, 0.2, "salt")
    train_keys = {(e.iteration, e.game_key) for e in train}
    val_keys = {(e.iteration, e.game_key) for e in validation}
    assert not (train_keys & val_keys)
    assert {e.iteration for e in validation} == {0, 1}
    assert {0, 1} <= {e.iteration for e in train}
    assert all(example in train for example in curriculum)


def test_stable_split_keeps_a_game_on_the_same_side_across_iterations():
    """The property run 02's split lacked.

    ``phase_d_game_honest_split`` reseeded from ``seed + iteration`` every
    training iteration, so a game could validate at iteration 3, train at 4 and
    validate again at 5 -- contaminating the holdout.  Assignment must depend
    only on the game's identity, not on when the split is taken.
    """

    early = _split_population(iterations=(0, 1))
    late = _split_population(iterations=(0, 1, 2, 3))
    _, early_val = stable_game_split(early, 0.2, "salt")
    _, late_val = stable_game_split(late, 0.2, "salt")
    early_keys = {(e.iteration, e.game_key) for e in early_val}
    late_keys = {(e.iteration, e.game_key) for e in late_val}
    assert early_keys
    # Every game held out early is still held out once the buffer has grown.
    assert early_keys <= late_keys


def test_stable_split_is_process_independent():
    """blake2b, not the builtin hash, so a resume reproduces the split."""

    examples = _split_population()
    first = {
        (e.iteration, e.game_key) for e in stable_game_split(examples, 0.2, "s")[1]
    }
    second = {
        (e.iteration, e.game_key) for e in stable_game_split(examples, 0.2, "s")[1]
    }
    assert first == second
    assert stable_is_validation(0, 7, 0.2, "s") == stable_is_validation(0, 7, 0.2, "s")
    # A different salt must be able to produce a different assignment.
    salts = {stable_is_validation(0, key, 0.2, "other") for key in range(200)}
    assert salts == {True, False}


def test_stable_split_holds_out_roughly_the_requested_fraction():
    examples = _split_population(iterations=range(6), games=200)
    train, validation = stable_game_split(examples, 0.05, "salt")
    held = len({e.game_key for e in validation})
    total = held + len({e.game_key for e in train})
    assert 0.03 < held / total < 0.08


def test_curriculum_examples_are_never_validation():
    curriculum = [SplitExample(None, key, 0) for key in range(500)]
    train, validation = stable_game_split(curriculum, 0.5, "salt")
    assert not validation
    assert len(train) == 500


def test_curriculum_anneal_auto_fits_inside_the_run():
    """Run 02 configured 20 anneal iterations for a 12-iteration run.

    The seed buffer was still at 45% of its original size at the last
    iteration.  ``-1`` resolves to half the run so the anneal always finishes.
    """

    assert resolve_anneal_iterations(-1, 12) == 6
    assert resolve_anneal_iterations(-1, 40) == 20
    assert resolve_anneal_iterations(-1, 1) == 1
    # An explicit duration is still honoured verbatim.
    assert resolve_anneal_iterations(20, 12) == 20
    assert resolve_anneal_iterations(0, 12) == 0
    # Auto-fitted, the curriculum is fully gone by the end of the run.
    assert curriculum_fraction(1.0, 12, resolve_anneal_iterations(-1, 12)) == 0.0


def test_gate_power_flags_the_configuration_that_never_decided():
    """Run 02: 100 games against a 3% indifference region, 11/11 probation."""

    run02 = PhaseDConfig(gate_max_games=100, gate_indifference=0.03)
    power = run02.gate_power()
    assert power["underpowered"]
    assert 350 < power["expected_games_to_accept_at_h1"] < 400
    generous = PhaseDConfig(gate_max_games=1000, gate_indifference=0.03)
    assert generous.gate_power()["resolvable"]


def test_evenly_matched_candidate_never_resolves():
    """The reason a small gate returns probation rather than a verdict.

    At the midpoint of the indifference region the log-likelihood ratio has no
    drift, so no finite game budget reaches a boundary.
    """

    assert expected_games_to_decide(0.47, 0.53, true_rate=0.50) == float("inf")
    assert expected_games_to_decide(0.47, 0.53, true_rate=0.53) < float("inf")


def test_cheap_double_reveal_offsets_default_off_and_reach_generation_only():
    """Chance capping ships as an off-by-default generation flag.

    Defaulting it on would change generation before Steps 4-5 measured its
    approximation quality and throughput; passing it to the gate would confound
    an arena result with the approximation being evaluated."""

    assert PhaseDConfig().cheap_double_reveal_offsets == 0
    PhaseDConfig(cheap_double_reveal_offsets=2).validate()
    with pytest.raises(ValueError, match="cheap_double_reveal_offsets"):
        PhaseDConfig(cheap_double_reveal_offsets=-1).validate()

    source = inspect.getsource(PhaseDLoop)
    generation, gate = (
        source.index("cheap_sims_min=self.config.cheap_sims_min"),
        source.index("cheap_sims_min=self.config.gate_sims"),
    )
    assert generation < source.index("cheap_double_reveal_offsets") < gate


def test_wilson_gate_config_does_not_reuse_the_obsolete_sprt_power_warning():
    # W5 chooses a cap from measured cloud cost. Unlike the old SPRT
    # indifference region, a smaller Wilson cap is still a valid (if stricter)
    # three-way decision and must not warn that it can decide nothing.
    PhaseDConfig(gate_max_games=100, promotion_every=1).validate()
    with pytest.raises(ValueError, match="promotion_min_lcb"):
        PhaseDConfig(promotion_min_lcb=1.1).validate()


def test_wilson_gate_uses_seat_pair_as_observation():
    from .phase_d import wilson_pair_decision

    # Twenty individual games would be the wrong n; these are ten independent
    # seed/seat pairs and the interval must be computed at n=10.
    pairs = [1.0, 0.0] * 4 + [1.0, 1.0]
    decision, used, rate, lower, upper, reason = wilson_pair_decision(pairs)
    assert used == 10
    assert rate == pytest.approx(0.6)
    assert lower < 0.5 < upper
    assert decision == "continue"
    assert reason == "probation"


def test_wilson_gate_promotes_reverts_and_otherwise_probates():
    from .phase_d import wilson_pair_decision

    promoted = wilson_pair_decision([1.0] * 20)
    assert promoted[0] == "accept"
    assert promoted[-1] == "promotion_lcb"

    # W5.5: revert is a confidence bound, so a *tied* candidate is probation no
    # matter how many pairs it is measured over -- the old point-estimate rule
    # reverted a third of these.
    tied = wilson_pair_decision([0.5] * 400)
    assert tied[0] == "continue"
    assert tied[-1] == "probation"

    reverted = wilson_pair_decision([0.0] * 20)
    assert reverted[0] == "reject"
    assert reverted[-1] == "revert_ucb"


def test_wilson_gate_never_stops_early_on_a_prefix():
    from .phase_d import wilson_pair_decision

    # A run of wins that crosses the promote boundary mid-match, then a
    # regression to dead even. Optional stopping promotes on the prefix; the
    # fixed-N rule reads the whole sample and returns probation.
    prefix = [1.0] * 12
    assert wilson_pair_decision(prefix)[0] == "accept", "prefix must be promotable"
    scores = prefix + [0.0] * 12 + [0.5] * 16
    decision, pairs, rate, _lcb, _ucb, reason = wilson_pair_decision(scores)
    assert (decision, pairs, reason) == ("continue", 40, "probation")
    assert rate == pytest.approx(0.5)


def test_revert_is_suppressed_for_one_gate_after_a_schedule_knot():
    from .phase_d import wilson_pair_decision

    losing = [0.0] * 20
    assert wilson_pair_decision(losing)[0] == "reject"
    suppressed = wilson_pair_decision(losing, revert_suppressed=True)
    assert suppressed[0] == "continue"
    assert suppressed[-1] == "revert_suppressed_knot"
    # Promotion is *not* suppressed: a knot is no reason to withhold a win.
    assert wilson_pair_decision([1.0] * 20, revert_suppressed=True)[0] == "accept"


def test_anchor_wilson_measurement_reports_against_its_threshold():
    from .phase_d import wilson_pair_decision

    result = wilson_pair_decision(
        [1.0] * 20,
        measurement=True,
        measurement_threshold=0.6,
    )
    assert result[0] == "accept"
    assert result[1] == 20
    assert result[-1] == "fixed_n"


def test_anchor_gate_cadence_counts_promotions_not_iterations():
    assert not should_run_anchor_gate(
        promoted=False, previous_promotions=2, cadence=3
    )
    assert not should_run_anchor_gate(
        promoted=True, previous_promotions=1, cadence=3
    )
    assert should_run_anchor_gate(
        promoted=True, previous_promotions=2, cadence=3
    )
    assert not should_run_anchor_gate(
        promoted=True, previous_promotions=2, cadence=0
    )


def test_coalescing_evaluator_aligns_concurrent_requests():
    backend = UniformEvaluator()
    service = CoalescingEvaluator(backend, max_batch=8, max_wait_ms=50).start()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(service.evaluate, [object()], [(index, index + 1)])
                for index in range(4)
            ]
            results = [future.result() for future in futures]
    finally:
        service.close()
    assert [len(result[0].policy) for result in results] == [2, 2, 2, 2]
    assert service.positions == 4
    assert any(size > 1 for size in backend.batch_sizes)


@pytest.mark.parametrize("backend", ("python", "rust"))
def test_seed_buffer_cycles_all_four_bots_through_both_seats(
    tmp_path: Path, backend: str
):
    records = generate_seed_buffer(
        tmp_path / f"seed_{backend}.jsonl",
        games=8,
        seed=100,
        workers=2,
        backend=backend,
    )
    assert len(records) == 8
    names = {bot_type().name for bot_type in CURRICULUM_BOT_TYPES}
    assert {record.agents["p0"] for record in records} & names == names
    assert {record.agents["p1"] for record in records} & names == names
    assert all(record.agents["opponent_type"] == "bot" for record in records)
    for record in records:
        replay(record)


def test_rust_seed_buffer_matches_python_bot_trajectories(tmp_path: Path):
    kwargs = {"games": 8, "seed": 321, "workers": 1}
    python = generate_seed_buffer(
        tmp_path / "python.jsonl", backend="python", **kwargs
    )
    rust = generate_seed_buffer(tmp_path / "rust.jsonl", backend="rust", **kwargs)
    assert [tuple(move.action for move in record.moves) for record in rust] == [
        tuple(move.action for move in record.moves) for record in python
    ]
    assert [(record.winner, record.victory_type) for record in rust] == [
        (record.winner, record.victory_type) for record in python
    ]


def test_final_buffer_export_can_warm_start_and_ages_through_replay_window(
    tmp_path: Path,
):
    source = tmp_path / "source.jsonl"
    saved = tmp_path / "buffer_final.jsonl"
    source_records = generate_seed_buffer(
        source,
        games=4,
        seed=123,
        workers=1,
        backend="python",
    )
    loop = PhaseDLoop(
        PhaseDConfig(
            run_dir=str(tmp_path / "run"),
            seed_games=0,
            warm_buffer=str(source),
            save_buffer=str(saved),
            # Fixed iteration window on purpose: this test pins warm-buffer
            # aging against an explicit `replay_window`, which is the legacy
            # basis. The games-basis window is covered in test_games_clock.py
            # and test_schedules_in_games.py.
            schedule_basis="iterations",
            replay_window=2,
            d_model=32,
            layers=1,
            device="cpu",
        )
    )
    loop.initialize()

    assert len(loop.training_records(0)) == 4
    assert len(loop.training_records(1)) == 4
    assert loop.training_records(2) == []

    loop._save_replay_buffer()
    exported = read_records(saved)
    assert [record.trajectory_digest for record in exported] == [
        record.trajectory_digest for record in source_records
    ]


def test_playout_cap_randomization_marks_cheap_policy_targets_excluded():
    config = PhaseDConfig(
        run_dir="unused",
        workers=1,
        games_per_iteration=1,
        seed_games=0,
        opponent_fraction=0.0,
        cheap_sims_min=1,
        cheap_sims_max=1,
        full_sims_min=1,
        full_sims_max=1,
        full_search_fraction=0.0,
        top_k=2,
        device="cpu",
    )
    record = _self_play_game(
        GameJob(0, 222),
        UniformEvaluator(),
        config,
        0,
        ResolvedSchedules(curriculum_mix_fraction=0.0, draft_prior=1.0),
    )
    assert record.moves
    assert all(move.sims == 1 for move in record.moves)
    assert all(move.policy_target is not None for move in record.moves)
    assert all(move.policy_excluded for move in record.moves)
    replay(record)


def test_initialize_train_checkpoint_and_promote_plumbing(tmp_path: Path):
    config = PhaseDConfig(
        run_dir=str(tmp_path / "run"),
        workers=2,
        games_per_iteration=1,
        seed_games=2,
        d_model=32,
        layers=1,
        train_steps=2,
        train_warmup_steps=0,
        validate_every=1,
        train_batch_size=64,
        min_games_to_train=2,
        device="cpu",
    )
    loop = PhaseDLoop(config)
    loop.initialize()
    records = loop.training_records(0)
    candidate = loop.train_candidate(records, 0)
    assert candidate.exists()
    loop.load_model(candidate)
    loop.promote(candidate, 0)
    assert loop.current_best.exists()
    assert len(loop.hof.entries()) == 1


def test_real_model_rust_self_play_writes_replayable_iteration(tmp_path: Path):
    config = PhaseDConfig(
        run_dir=str(tmp_path / "run"),
        workers=2,
        inference_batch=8,
        inference_wait_ms=10,
        games_per_iteration=2,
        seed_games=0,
        opponent_fraction=0.0,
        d_model=32,
        layers=1,
        cheap_sims_min=1,
        cheap_sims_max=1,
        full_sims_min=1,
        full_sims_max=1,
        full_search_fraction=1.0,
        top_k=2,
        device="cpu",
    )
    loop = PhaseDLoop(config)
    loop.initialize()
    records = loop.generate_iteration(loop.load_model(loop.current_best), 0)
    assert len(records) == 2
    assert loop.last_generation_stats["mode"] == "rust"
    assert loop.last_generation_stats["rust_games"] == 2
    assert loop.last_generation_stats["python_bot_games"] == 0
    assert all(
        record.agents["opponent_type"] == "current_best"
        for record in records
    )
    assert all(not move.policy_excluded for record in records for move in record.moves)
    for record in records:
        replay(record)


def test_rust_self_play_keeps_curriculum_bot_seats_inside_rust(tmp_path: Path):
    config = PhaseDConfig(
        run_dir=str(tmp_path / "run"),
        workers=1,
        games_per_iteration=2,
        seed_games=0,
        opponent_fraction=1.0,
        bot_exploration=0.0,
        d_model=32,
        layers=1,
        cheap_sims_min=1,
        cheap_sims_max=1,
        full_sims_min=1,
        full_sims_max=1,
        full_search_fraction=1.0,
        top_k=2,
        device="cpu",
    )
    loop = PhaseDLoop(config)
    loop.initialize()
    records = loop.generate_iteration(loop.load_model(loop.current_best), 0)
    assert loop.last_generation_stats["mode"] == "rust"
    assert loop.last_generation_stats["rust_bot_games"] == 2
    assert loop.last_generation_stats["python_bot_games"] == 0
    assert all(record.agents["kind"] == "mixed" for record in records)
    assert all(record.agents["opponent_type"] == "bot" for record in records)
    assert all(any(move.mode == "bot" for move in record.moves) for record in records)
    assert all(
        all(move.policy_target is None for move in record.moves if move.mode == "bot")
        for record in records
    )
    for record in records:
        replay(record)


def test_process_generation_is_deterministic_and_replayable(tmp_path: Path):
    def build(name: str) -> PhaseDLoop:
        config = PhaseDConfig(
            run_dir=str(tmp_path / name),
            workers=1,
            process_workers=2,
            generation_backend="python",
            inference_batch=8,
            games_per_iteration=2,
            seed_games=0,
            opponent_fraction=0.0,
            d_model=32,
            layers=1,
            cheap_sims_min=1,
            cheap_sims_max=1,
            full_sims_min=1,
            full_sims_max=1,
            full_search_fraction=1.0,
            top_k=2,
            device="cpu",
        )
        loop = PhaseDLoop(config)
        loop.initialize()
        return loop

    first = build("run_a")
    records_a = first.generate_iteration(first.load_model(first.current_best), 0)
    assert first.last_generation_stats["mode"] == "process"
    assert first.last_generation_stats["process_workers"] == 2
    second = build("run_b")
    records_b = second.generate_iteration(second.load_model(second.current_best), 0)
    assert records_a == records_b
    assert len(records_a) == 2
    assert all(
        record.agents["opponent_type"] == "current_best"
        for record in records_a
    )
    for record in records_a:
        replay(record)


def test_process_gate_is_bit_identical_to_sequential_gate(tmp_path: Path):
    def build(name: str, process_workers: int) -> PhaseDLoop:
        config = PhaseDConfig(
            run_dir=str(tmp_path / name),
            workers=1,
            process_workers=process_workers,
            inference_batch=8,
            seed_games=0,
            d_model=32,
            layers=1,
            top_k=2,
            gate_sims=1,
            gate_max_games=4,
            device="cpu",
        )
        loop = PhaseDLoop(config)
        loop.initialize()
        return loop

    sequential = build("sequential", 0)
    parallel = build("parallel", 2)
    # Identical seeds build identical initial weights, and gate games on CPU
    # are deterministic per seed, so the speculative wave path must reproduce
    # the sequential decision, game count, and score exactly.
    result_seq = sequential.promotion_gate(sequential.current_best)
    result_par = parallel.promotion_gate(parallel.current_best)
    assert result_par == result_seq
    # W5.6: boundary-stopped promotion evidence is not fed into Elo. Fixed-N
    # anchors are the only unbiased ladder input.
    assert not (sequential.run_dir / "elo" / "elo_games.jsonl").exists()
    assert not (parallel.run_dir / "elo" / "elo_games.jsonl").exists()


def test_rust_model_vs_bot_anchor_uses_native_bot_seat(tmp_path: Path):
    config = PhaseDConfig(
        run_dir=str(tmp_path / "run"),
        workers=1,
        seed_games=0,
        d_model=32,
        layers=1,
        top_k=2,
        gate_sims=1,
        gate_max_games=2,
        rust_slots=1,
        device="cpu",
    )
    loop = PhaseDLoop(config)
    loop.initialize()
    report, outcomes = loop._sprt_match(
        loop._model_agent_spec(loop.current_best, "candidate"),
        BotAgentSpec(GreedyBot()),
        threshold=0.5,
        seed_offset=70_000_000,
    )
    assert report.games == 2
    assert len(outcomes) == 2
    assert {outcome.agents for outcome in outcomes} == {
        ("candidate_iter_-1", "greedy"),
        ("greedy", "candidate_iter_-1"),
    }


def test_anchor_failure_does_not_block_current_best_promotion(
    tmp_path: Path, monkeypatch
):
    config = PhaseDConfig(
        run_dir=str(tmp_path / "run"),
        workers=1,
        games_per_iteration=1,
        seed_games=0,
        d_model=32,
        layers=1,
        anchor_gate_every_promotions=1,
        device="cpu",
    )
    loop = PhaseDLoop(config)
    loop.initialize()
    candidate = loop.checkpoint_dir / "candidate_test.pt"
    candidate.write_bytes(b"candidate")
    promoted = []
    monkeypatch.setattr(loop, "generate_iteration", lambda model, iteration: [])
    monkeypatch.setattr(loop, "training_records", lambda iteration: [])
    monkeypatch.setattr(
        loop, "train_candidate", lambda records, iteration: candidate
    )
    monkeypatch.setattr(
        loop,
        "promotion_gate",
        lambda path: GateResult("best", 0.5, "accept", 20, 0.7),
    )
    monkeypatch.setattr(
        loop,
        "anchor_gates",
        lambda path: [GateResult("greedy", 0.65, "reject", 20, 0.55)],
    )
    monkeypatch.setattr(
        loop, "promote", lambda path, iteration: promoted.append((path, iteration))
    )
    row = loop.run_iteration(0)
    assert row["promoted"]
    assert not row["phase_gate_passed"]
    assert promoted == [(candidate, 0)]
    assert row["promotion_gate"]["decision"] == "accept"
    assert row["anchor_gates"][0]["decision"] == "reject"

    monkeypatch.setattr(
        loop,
        "promotion_gate",
        lambda path: GateResult("best", 0.5, "reject", 20, 0.3),
    )

    def unexpected_anchor_gate(_path):
        raise AssertionError("anchors must not run for a rejected candidate")

    monkeypatch.setattr(loop, "anchor_gates", unexpected_anchor_gate)
    rejected = loop.run_iteration(1)
    assert not rejected["promoted"]
    assert rejected["anchor_gates"] == []
    assert promoted == [(candidate, 0)]
    log_rows = [
        json.loads(line)
        for line in (loop.run_dir / "training_log.jsonl").read_text().splitlines()
    ]
    assert [row["iteration"] for row in log_rows] == [0, 1]
    assert log_rows[0]["promotion_gate"]["decision"] == "accept"
    assert log_rows[1]["promotion_gate"]["decision"] == "reject"


# -- Milestone 2: soft-gate lifecycle (controller path) ---------------------


def _soft_gate_config(tmp_path: Path, **overrides) -> PhaseDConfig:
    base = dict(
        run_dir=str(tmp_path / "run"),
        selfplay_generator_mode="soft_gate",
        bootstrap_policy="auto_first_trained",
        promotion_every=1,
        iterations=2,
        games_per_iteration=1,
        seed_games=2,
        workers=1,
        d_model=32,
        layers=1,
        cheap_sims_min=1,
        cheap_sims_max=1,
        full_sims_min=1,
        full_sims_max=1,
        full_search_fraction=1.0,
        top_k=2,
        train_steps=2,
        train_warmup_steps=0,
        validate_every=1,
        train_batch_size=64,
        gate_sims=1,
        gate_max_games=2,
        anchor_gate_every_promotions=0,
        device="cpu",
    )
    base.update(overrides)
    return PhaseDConfig(**base)


def _scripted_gate(decision: str, sizes: list[int] | None = None):
    def gate(candidate, *, opponent=None, games=0, iteration=None):
        if sizes is not None:
            sizes.append(games)
        return GateResult("best", 0.5, decision, 2, 0.5)

    return gate


def test_strict_gate_is_the_backward_compatible_default():
    assert PhaseDConfig(run_dir="unused").selfplay_generator_mode == "strict_gate"
    assert PhaseDConfig(run_dir="unused").bootstrap_policy == "gate"


def test_soft_gate_bootstrap_ratchets_learner_forward(tmp_path, monkeypatch):
    loop = PhaseDLoop(_soft_gate_config(tmp_path))
    monkeypatch.setattr(loop, "promotion_gate", _scripted_gate("continue"))
    rows = loop.run()

    # First trained learner escapes iteration -1 without a strength gate.
    assert rows[0]["promotion_action"] == "bootstrap_promote"
    assert rows[0]["promotion_scheduled"] is False
    assert rows[0]["current_best_iteration"] == 0

    # Iteration 1 is a probation continuation: latest advances, best is frozen.
    assert rows[1]["promotion_action"] == "probation"
    assert rows[1]["current_best_iteration"] == 0
    assert rows[1]["current_best_sha256"] == rows[0]["latest_sha256"]
    assert rows[1]["latest_sha256"] != rows[1]["current_best_sha256"]

    checkpoints = loop.checkpoint_dir
    assert (checkpoints / "latest.pt").exists()
    assert (checkpoints / "current_best.pt").exists()


def test_soft_gate_reject_switches_generation_to_best(tmp_path, monkeypatch):
    loop = PhaseDLoop(_soft_gate_config(tmp_path, iterations=3))
    monkeypatch.setattr(loop, "promotion_gate", _scripted_gate("reject"))
    rows = loop.run()

    # iter0 bootstrap -> latest; iter1 generates with latest then gate rejects;
    # iter2 must switch generation to the protected best for recovery data.
    assert rows[0]["generator_source"] == "latest"
    assert rows[1]["generator_source"] == "latest"
    assert rows[1]["promotion_action"] == "revert"
    assert rows[2]["generator_source"] == "current_best"
    # A reject never overwrites the protected best.
    assert rows[2]["current_best_sha256"] == rows[0]["latest_sha256"]


def test_soft_gate_revert_reset_restores_best_into_latest(tmp_path, monkeypatch):
    loop = PhaseDLoop(
        _soft_gate_config(tmp_path, iterations=3, revert_reset_after=2)
    )
    monkeypatch.setattr(loop, "promotion_gate", _scripted_gate("reject"))
    rows = loop.run()

    assert rows[1]["promotion_action"] == "revert"
    assert rows[2]["promotion_action"] == "revert_reset"
    # After the reset the learner weights equal the protected best.
    assert rows[2]["latest_sha256"] == rows[2]["current_best_sha256"]


def test_soft_gate_accept_promotes_and_archives_outgoing_best(tmp_path, monkeypatch):
    loop = PhaseDLoop(_soft_gate_config(tmp_path, iterations=2))
    monkeypatch.setattr(loop, "promotion_gate", _scripted_gate("accept"))
    rows = loop.run()

    assert rows[0]["promotion_action"] == "bootstrap_promote"
    assert rows[1]["promotion_action"] == "promote"
    # The promoted learner becomes the new protected best.
    assert rows[1]["current_best_iteration"] == 1
    assert rows[1]["current_best_sha256"] == rows[1]["latest_sha256"]
    # The outgoing (bootstrap) best is archived to HOF exactly once.
    assert len(loop.hof.entries()) == 1


def test_soft_gate_reuses_paired_sprt_decision_unchanged(tmp_path, monkeypatch):
    """The soft gate must consume the paired-SPRT decision directly, mapping
    accept/continue/reject to promote/probation/revert with no second decision
    system."""

    seen: list[str] = []

    def gate(candidate, *, opponent=None, games=0, iteration=None):
        # Prove the adapter passes latest vs current_best into the real gate.
        assert Path(candidate).name == "latest.pt"
        assert Path(opponent).name == "current_best.pt"
        decision = ["continue", "reject"][len(seen)]
        seen.append(decision)
        return GateResult("best", 0.5, decision, 2, 0.5)

    loop = PhaseDLoop(_soft_gate_config(tmp_path, iterations=3))
    monkeypatch.setattr(loop, "promotion_gate", gate)
    rows = loop.run()

    assert seen == ["continue", "reject"]
    assert [row["promotion_action"] for row in rows] == [
        "bootstrap_promote",
        "probation",
        "revert",
    ]


def test_soft_gate_resume_continues_without_cold_start(tmp_path, monkeypatch):
    first = PhaseDLoop(_soft_gate_config(tmp_path, iterations=1))
    monkeypatch.setattr(first, "promotion_gate", _scripted_gate("continue"))
    first_rows = first.run()
    assert [row["iteration"] for row in first_rows] == [0]

    second = PhaseDLoop(_soft_gate_config(tmp_path, iterations=1))
    monkeypatch.setattr(second, "promotion_gate", _scripted_gate("continue"))
    second_rows = second.run()

    # The resumed run picks up at iteration 1 from the on-disk latest/best,
    # never restarting from iteration -1.
    assert [row["iteration"] for row in second_rows] == [1]
    assert second_rows[0]["promotion_action"] == "probation"
    assert second_rows[0]["current_best_iteration"] == 0


# -- Milestone 4: replay operations -----------------------------------------


def test_warm_staleness_filter_ages_out_old_games_and_keeps_curriculum():
    records = [SimpleNamespace(iteration=i) for i in range(6)]
    records.append(SimpleNamespace(iteration=None))  # curriculum, never aged
    retained, stats = filter_warm_records_by_staleness(records, max_staleness=2)

    # newest numbered iteration is 5; keep age < 2 -> iterations 4 and 5.
    assert sorted(r.iteration for r in retained if r.iteration is not None) == [4, 5]
    assert any(r.iteration is None for r in retained)
    assert stats == {
        "loaded": 7,
        "retained": 3,
        "dropped": 4,
        "newest_iteration": 5,
        "max_staleness": 2,
    }


def test_warm_buffer_import_drops_stale_numbered_games(tmp_path):
    base = generate_seed_buffer(
        tmp_path / "seed.jsonl", games=4, seed=11, workers=1, backend="python"
    )
    numbered = [replace(record, iteration=i) for i, record in enumerate(base)]
    warm = tmp_path / "warm.jsonl"
    _write_records(warm, numbered)

    loop = PhaseDLoop(
        PhaseDConfig(
            run_dir=str(tmp_path / "run"),
            seed_games=0,
            warm_buffer=str(warm),
            warm_buffer_max_staleness=2,
            replay_window=5,
            d_model=32,
            layers=1,
            device="cpu",
        )
    )
    loop.initialize()

    # newest=3; age < 2 keeps iterations 2 and 3, drops 0 and 1.
    assert sorted(record.iteration for record in loop.warm_records) == [2, 3]
    assert loop.last_warm_stats["loaded"] == 4
    assert loop.last_warm_stats["retained"] == 2
    assert loop.last_warm_stats["dropped"] == 2


def test_atomic_save_leaves_previous_export_readable_on_interrupted_write(
    tmp_path, monkeypatch
):
    records = generate_seed_buffer(
        tmp_path / "src.jsonl", games=2, seed=7, workers=1, backend="python"
    )
    dest = tmp_path / "export.jsonl"
    _write_records(dest, records)
    original = dest.read_bytes()

    real = phase_d_module.to_json_line
    state = {"n": 0}

    def flaky(record):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("simulated crash mid-write")
        return real(record)

    monkeypatch.setattr(phase_d_module, "to_json_line", flaky)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _write_records(dest, records)

    # A partial temp write must never replace the last valid export.
    assert dest.read_bytes() == original
    assert [record.trajectory_digest for record in read_records(dest)] == [
        record.trajectory_digest for record in records
    ]


def test_soft_gate_autosave_exports_reloadable_buffer(tmp_path, monkeypatch):
    saved = tmp_path / "auto.jsonl"
    loop = PhaseDLoop(
        _soft_gate_config(
            tmp_path,
            iterations=2,
            save_buffer=str(saved),
            buffer_autosave_every=1,
        )
    )
    monkeypatch.setattr(loop, "promotion_gate", _scripted_gate("continue"))
    loop.run()

    assert saved.exists()
    reloaded = read_records(saved)
    assert reloaded
    for record in reloaded:
        replay(record)
    # A resumed/re-exported snapshot must not accumulate duplicate games.
    digests = [record.trajectory_digest for record in reloaded]
    assert len(digests) == len(set(digests))


# -- Milestone 5: logging parity --------------------------------------------


def _read_log_rows(loop: PhaseDLoop) -> list[dict]:
    text = (loop.run_dir / "training_log.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


LIFECYCLE_ROW_FIELDS = {
    "iteration",
    "log_schema_version",
    "control_state",
    "generator_mode",
    "generator_source",
    "generator_checkpoint",
    "generator_sha256",
    "learner_source",
    "latest_checkpoint",
    "latest_sha256",
    "current_best_checkpoint",
    "current_best_sha256",
    "current_best_iteration",
    "bootstrap_state",
    "promotion_scheduled",
    "promotion_action",
    "consecutive_reverts",
}


def test_training_log_golden_rows_cover_all_lifecycle_actions(tmp_path, monkeypatch):
    loop = PhaseDLoop(_soft_gate_config(tmp_path, iterations=4))
    decisions = iter(["continue", "reject", "accept"])

    def gate(candidate, *, opponent=None, games=0, iteration=None):
        return GateResult("best", 0.5, next(decisions), 2, 0.5)

    monkeypatch.setattr(loop, "promotion_gate", gate)
    loop.run()

    rows = _read_log_rows(loop)
    # Exactly one row per completed iteration, covering all four actions.
    assert [row["iteration"] for row in rows] == [0, 1, 2, 3]
    assert [row["promotion_action"] for row in rows] == [
        "bootstrap_promote",
        "probation",
        "revert",
        "promote",
    ]
    for row in rows:
        missing = LIFECYCLE_ROW_FIELDS - set(row)
        assert not missing, missing
        assert row["log_schema_version"] == 2
        assert row["stats"]["schema_version"] == 2
        assert row["learner_source"] == "latest"


def test_training_log_has_exactly_one_row_per_iteration_across_resume(
    tmp_path, monkeypatch
):
    first = PhaseDLoop(_soft_gate_config(tmp_path, iterations=1))
    monkeypatch.setattr(first, "promotion_gate", _scripted_gate("continue"))
    first.run()

    second = PhaseDLoop(_soft_gate_config(tmp_path, iterations=1))
    monkeypatch.setattr(second, "promotion_gate", _scripted_gate("continue"))
    second.run()

    rows = _read_log_rows(second)
    # Resume must not duplicate the already-logged iteration 0.
    assert [row["iteration"] for row in rows] == [0, 1]


def test_pending_iteration_rollback_removes_restart_blockers(tmp_path):
    from .training_adapter import SevenWondersDuelLifecycleAdapter

    loop = PhaseDLoop(_soft_gate_config(tmp_path, iterations=1))
    loop.buffer_dir.mkdir(parents=True, exist_ok=True)
    loop.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    incomplete_buffer = loop.buffer_dir / "iter_0070.jsonl"
    incomplete_candidate = loop.checkpoint_dir / "candidate_0070.pt"
    prior_buffer = loop.buffer_dir / "iter_0069.jsonl"
    for path in (incomplete_buffer, incomplete_candidate, prior_buffer):
        path.write_bytes(b"partial")
    loop.optimizer_state_path.write_bytes(b"stale moments")

    SevenWondersDuelLifecycleAdapter(loop).rollback_iteration(70)

    assert not incomplete_buffer.exists()
    assert not incomplete_candidate.exists()
    assert not loop.optimizer_state_path.exists()
    assert prior_buffer.exists(), "committed iterations are immutable"


def test_disabled_run_log_still_writes_structured_log(tmp_path, monkeypatch):
    from games.az_loop import RunLog

    loop = PhaseDLoop(_soft_gate_config(tmp_path, iterations=1))
    monkeypatch.setattr(loop, "promotion_gate", _scripted_gate("continue"))
    with RunLog(loop.run_dir / "run.log", enabled=False):
        loop.run()

    # Disabling the human transcript must not affect JSONL/manifest persistence.
    assert (loop.run_dir / "training_log.jsonl").exists()
    assert (loop.run_dir / "run_manifest.json").exists()
    assert not (loop.run_dir / "run.log").exists()


# -- W5.8/W5.9: scheduled gate size and knot amnesty ------------------------


def test_ladder_sizes_reach_the_gate_and_step_up_on_probation(tmp_path, monkeypatch):
    sizes: list[int] = []
    loop = PhaseDLoop(
        _soft_gate_config(
            tmp_path,
            iterations=4,
            gate_ladder_games=(2, 4, 6),
            gate_ladder_step_up_after=2,
        )
    )
    monkeypatch.setattr(loop, "promotion_gate", _scripted_gate("continue", sizes))
    rows = loop.run()

    # Iteration 0 bootstraps (no gate), then three probations: the third runs at
    # the second rung because the first two stepped the ladder up.
    assert sizes == [2, 2, 4]
    assert rows[-1]["control_state"]["gate_rung"] == 1


def test_a_promotion_steps_the_ladder_back_down(tmp_path, monkeypatch):
    sizes: list[int] = []
    loop = PhaseDLoop(
        _soft_gate_config(
            tmp_path,
            iterations=5,
            gate_ladder_games=(2, 4, 6),
            gate_ladder_step_up_after=1,
        )
    )
    decisions = iter(["continue", "continue", "accept"])

    def gate(candidate, *, opponent=None, games=0, iteration=None):
        sizes.append(games)
        return GateResult("best", 0.5, next(decisions, "continue"), 2, 0.5)

    monkeypatch.setattr(loop, "promotion_gate", gate)
    loop.run()
    # Up on each probation, then back one rung after the promotion.
    assert sizes == [2, 4, 6, 4]


def test_gate_size_falls_back_to_gate_max_games_without_a_ladder(tmp_path, monkeypatch):
    sizes: list[int] = []
    loop = PhaseDLoop(_soft_gate_config(tmp_path, iterations=3, gate_max_games=2))
    monkeypatch.setattr(loop, "promotion_gate", _scripted_gate("continue", sizes))
    loop.run()
    assert sizes == [2, 2]


def test_revert_suppression_covers_the_gate_that_follows_a_schedule_knot(tmp_path):
    loop = PhaseDLoop(
        _soft_gate_config(
            tmp_path,
            schedule_basis="games",
            curriculum_anneal_games=4,
            draft_prior_games=4,
            hof_opponent_fraction=0.0,
            promotion_every=1,
        )
    )
    knots = loop.schedule_knots()
    assert 4 in knots

    class _Ledger:
        def __init__(self, per_iteration: int):
            self.per_iteration = per_iteration

        def total_before(self, iteration: int) -> int:
            return max(0, iteration) * self.per_iteration

        def total_through(self, iteration: int) -> int:
            return (max(0, iteration) + 1) * self.per_iteration

    loop.games_ledger = _Ledger(2)
    # Iteration 1 spans games 2..4, so the knot at 4 lands inside it.
    assert loop.revert_suppressed(1) is True
    assert loop.revert_suppressed(2) is False


def test_hof_start_is_only_a_knot_when_the_league_is_switched_on(tmp_path):
    off = PhaseDLoop(
        _soft_gate_config(
            tmp_path / "off",
            schedule_basis="games",
            hof_opponent_fraction=0.0,
            hof_start_games=1234,
        )
    )
    assert 1234 not in off.schedule_knots()
    on = PhaseDLoop(
        _soft_gate_config(
            tmp_path / "on",
            schedule_basis="games",
            hof_opponent_fraction=0.15,
            hof_start_games=1234,
        )
    )
    assert 1234 in on.schedule_knots()


def test_revert_thresholds_must_partition_the_decision_space():
    PhaseDConfig(promotion_min_lcb=0.50, revert_max_ucb=0.50).validate()
    with pytest.raises(ValueError, match="revert_max_ucb must not exceed"):
        PhaseDConfig(promotion_min_lcb=0.50, revert_max_ucb=0.55).validate()


def test_confirmed_revert_resets_the_learner_but_a_single_one_does_not():
    """W5.5's two-stage revert, driven by the real rule rather than a script.

    No historical run reverts under this rule (A7 r10 restatement), so the path
    from losing pair scores to REVERT_RESET has to be covered synthetically.
    """

    from games.az_loop import GateLadder
    from games.az_loop.training_control import (
        GeneratorMode,
        PromotionAction,
        gate_transition,
        initial_state,
    )
    from .phase_d import wilson_pair_decision

    losing = wilson_pair_decision([0.0] * 20)[0]
    even = wilson_pair_decision([0.5] * 20)[0]
    assert (losing, even) == ("reject", "continue")

    ladder = GateLadder.fixed(40)
    state = initial_state(GeneratorMode.SOFT_GATE)

    first = gate_transition(
        state, losing, revert_reset_after=2, iteration=1, ladder=ladder
    )
    assert first.action is PromotionAction.REVERT
    assert first.reset_learner is False
    assert first.next_state.consecutive_reverts == 1

    # An even gate in between clears the count: only *consecutive* evidence of
    # a regression is allowed to discard the learner.
    interrupted = gate_transition(
        first.next_state, even, revert_reset_after=2, iteration=2, ladder=ladder
    )
    assert interrupted.next_state.consecutive_reverts == 0

    confirmed = gate_transition(
        first.next_state, losing, revert_reset_after=2, iteration=2, ladder=ladder
    )
    assert confirmed.action is PromotionAction.REVERT_RESET
    assert confirmed.reset_learner is True
