from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from games.seven_wonders_duel.buffer import replay
from games.seven_wonders_duel.codec import legal_action_indices
from games.seven_wonders_duel.data import WONDERS_BY_NAME
from games.seven_wonders_duel.game import new_game
from games.seven_wonders_duel.inference import Evaluation
from games.seven_wonders_duel.loop_inference import CoalescingEvaluator
from games.seven_wonders_duel.phase_d import (
    CURRICULUM_BOT_TYPES,
    GateResult,
    PhaseDConfig,
    PhaseDLoop,
    WONDER_DRAFT_TIERS,
    _self_play_game,
    blend_draft_priors,
    curriculum_fraction,
    generate_seed_buffer,
    phase_d_game_honest_split,
    should_run_anchor_gate,
    temperature_for_move,
)
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


def test_seed_buffer_cycles_all_four_bots_through_both_seats(tmp_path: Path):
    records = generate_seed_buffer(
        tmp_path / "seed.jsonl", games=8, seed=100, workers=2
    )
    assert len(records) == 8
    names = {bot_type().name for bot_type in CURRICULUM_BOT_TYPES}
    assert {record.agents["p0"] for record in records} & names == names
    assert {record.agents["p1"] for record in records} & names == names
    for record in records:
        replay(record)


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


def test_real_model_coalesced_self_play_writes_replayable_iteration(tmp_path: Path):
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
    assert loop.last_generation_stats["inference_positions"] > 0
    assert loop.last_generation_stats["inference_batches"] > 0
    assert all(not move.policy_excluded for record in records for move in record.moves)
    for record in records:
        replay(record)


def test_process_generation_is_deterministic_and_replayable(tmp_path: Path):
    def build(name: str) -> PhaseDLoop:
        config = PhaseDConfig(
            run_dir=str(tmp_path / name),
            workers=1,
            process_workers=2,
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
