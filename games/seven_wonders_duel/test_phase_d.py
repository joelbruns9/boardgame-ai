from __future__ import annotations

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
    blend_draft_priors,
    curriculum_fraction,
    filter_warm_records_by_staleness,
    generate_seed_buffer,
    phase_d_game_honest_split,
    should_run_anchor_gate,
    temperature_for_move,
)
from games.seven_wonders_duel import phase_d as phase_d_module
from games.seven_wonders_duel.phase_d import _write_records
from games.az_loop import GameJob


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


def test_phase_d_split_trains_on_fresh_games_without_game_leakage():
    examples = [
        SplitExample(iteration, iteration * 100 + game, move)
        for iteration in (0, 1)
        for game in range(10)
        for move in range(2)
    ]
    curriculum = [SplitExample(None, 10_000 + game, 0) for game in range(3)]
    train, validation = phase_d_game_honest_split(
        examples + curriculum, val_frac=0.2, seed=9
    )
    train_keys = {(example.iteration, example.game_key) for example in train}
    val_keys = {(example.iteration, example.game_key) for example in validation}
    assert not (train_keys & val_keys)
    assert {example.iteration for example in validation} == {0, 1}
    assert {0, 1} <= {example.iteration for example in train}
    assert all(example in train for example in curriculum)


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
    record = _self_play_game(GameJob(0, 222), UniformEvaluator(), config, 0)
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
        train_epochs=1,
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
    ledger_seq = (sequential.run_dir / "elo" / "elo_games.jsonl").read_text()
    ledger_par = (parallel.run_dir / "elo" / "elo_games.jsonl").read_text()
    assert ledger_par == ledger_seq


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
        train_epochs=1,
        train_batch_size=64,
        gate_sims=1,
        gate_max_games=2,
        anchor_gate_every_promotions=0,
        device="cpu",
    )
    base.update(overrides)
    return PhaseDConfig(**base)


def _scripted_gate(decision: str):
    def gate(candidate, *, opponent=None):
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

    def gate(candidate, *, opponent=None):
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

    def gate(candidate, *, opponent=None):
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
        assert row["log_schema_version"] == 1
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
