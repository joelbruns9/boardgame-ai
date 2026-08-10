from __future__ import annotations

import random
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from games.kingdomino.chance_correct_match import (
    AdvisorSearchBot,
    SearchSpec,
    _parse_args,
    main,
    run_paired_search_match,
)
from games.kingdomino.game import GameState
from games.kingdomino.network import KingdominoNet
from games.kingdomino.round_robin_eval import GameResult
from games.kingdomino.self_play import make_rust_evaluator


def test_search_spec_is_immutable_and_validates_chance_modes():
    spec = SearchSpec(sims=400, chance_exposure=1)
    assert spec.topology == "one_reveal"
    with pytest.raises(FrozenInstanceError):
        spec.sims = 800

    with pytest.raises(ValueError, match="sims"):
        SearchSpec(sims=0)
    with pytest.raises(ValueError, match="chance_enum_max_rows"):
        SearchSpec(sims=1, chance_exposure=1, chance_enum_max_rows=0)
    with pytest.raises(ValueError, match="chance_backup"):
        SearchSpec(sims=1, chance_backup="closed_mean")
    with pytest.raises(ValueError, match="chance_traversal"):
        SearchSpec(sims=1, chance_traversal="best_q")


def test_advisor_search_bot_forwards_its_own_spec_and_counts_nn_rows():
    calls = []

    def evaluator(my_board, opp_board, flat, legal_indices):
        return np.zeros(3, dtype=np.float32), [np.zeros(1)] * 3

    def search_fn(rust_state, counted_evaluator, sims, **kwargs):
        counted_evaluator(
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            [[0], [1], [2]],
        )
        calls.append((rust_state, sims, kwargs))
        return [
            (9, 4, 0.0, 0.10),
            (3, 4, 0.0, 0.20),
            (2, 3, 0.0, 0.90),
        ], 0.0

    spec = SearchSpec(
        sims=321,
        chance_exposure=1,
        chance_enum_max_rows=12,
        chance_backup="hajek",
        chance_traversal="balanced",
        leaf_batch=7,
        cpuct=1.7,
        fpu=-0.15,
    )
    bot = AdvisorSearchBot(
        evaluator,
        spec,
        search_fn=search_fn,
        rust_state_factory=lambda state: "rust-state",
        action_decoder=lambda index, state: ("decoded", index),
    )

    chosen = bot.choose_action(object(), ["a", "b"], rng=random.Random(5))

    assert chosen == ("decoded", 3)
    assert len(calls) == 1
    rust_state, sims, kwargs = calls[0]
    assert rust_state == "rust-state"
    assert sims == 321
    assert kwargs["chance_exposure"] == 1
    assert kwargs["chance_enum_max_rows"] == 12
    assert kwargs["chance_backup"] == "hajek"
    assert kwargs["chance_traversal"] == "balanced"
    assert kwargs["leaf_batch"] == 7
    assert kwargs["cpuct"] == 1.7
    assert kwargs["fpu"] == -0.15
    assert kwargs["dirichlet_eps"] == 0.0
    assert bot.counters.search_calls == 1
    assert bot.counters.nn_evaluations == 3
    assert bot.counters.elapsed_seconds >= 0.0


def test_advisor_search_bot_skips_search_for_forced_action():
    def fail(*args, **kwargs):
        raise AssertionError("forced actions must not invoke search")

    bot = AdvisorSearchBot(
        fail,
        SearchSpec(sims=10),
        search_fn=fail,
        rust_state_factory=fail,
    )
    assert bot.choose_action(object(), ["only"]) == "only"
    assert bot.counters.search_calls == 0
    assert bot.counters.nn_evaluations == 0


def test_advisor_search_bot_real_rust_boundary_cpu_smoke():
    state = GameState.new(seed=71)
    net = KingdominoNet(channels=8, blocks=1, bilinear_dim=8)
    evaluator = make_rust_evaluator(net, device="cpu")
    bot = AdvisorSearchBot(
        evaluator,
        SearchSpec(
            sims=2,
            chance_exposure=1,
            chance_enum_max_rows=12,
            chance_backup="hajek",
            chance_traversal="balanced",
            leaf_batch=2,
        ),
    )

    legal = state.legal_actions()
    chosen = bot.choose_action(state, legal, rng=random.Random(99))

    assert chosen in legal
    assert bot.counters.search_calls == 1
    assert bot.counters.nn_evaluations >= 1


def test_paired_search_match_swaps_seats_on_identical_deck_seeds():
    calls = []

    def fake_play(p0_name, p0_bot, p1_name, p1_bot, *, seed):
        calls.append((p0_name, p1_name, seed))
        treatment_is_p0 = p0_name == "treatment"
        return GameResult(
            seed=seed,
            p0=p0_name,
            p1=p1_name,
            score0=10 if treatment_is_p0 else 0,
            score1=0 if treatment_is_p0 else 10,
            winner="treatment",
            steps=1,
        )

    stats, pair, games = run_paired_search_match(
        object(),
        object(),
        paired_seeds=3,
        seed_start=100,
        play_game_fn=fake_play,
    )

    assert calls == [
        ("treatment", "control", 100),
        ("control", "treatment", 100),
        ("treatment", "control", 101),
        ("control", "treatment", 101),
        ("treatment", "control", 102),
        ("control", "treatment", 102),
    ]
    assert len(games) == 6
    assert pair.games == 6
    assert pair.a_wins == 6
    assert stats.pairs == 3
    assert stats.pair_wins == 3
    assert stats.pair_score_rate == 1.0


def test_paired_search_match_rejects_invalid_design():
    with pytest.raises(ValueError, match="paired_seeds"):
        run_paired_search_match(object(), object(), paired_seeds=0, seed_start=1)
    with pytest.raises(ValueError, match="names"):
        run_paired_search_match(
            object(),
            object(),
            paired_seeds=1,
            seed_start=1,
            treatment_name="same",
            control_name="same",
        )


def test_cli_accepts_independent_sim_budgets_and_requires_provenance(tmp_path):
    args = _parse_args([
        "--output", str(tmp_path / "match.json"),
        "--selection-reason", "preregistered A2a pulse",
        "--paired-seeds", "4",
        "--sims", "400",
        "--treatment-sims", "800",
        "--control-sims", "600",
    ])
    assert args.treatment_sims == 800
    assert args.control_sims == 600
    assert args.selection_reason == "preregistered A2a pulse"


def test_cli_refuses_to_overwrite_completed_artifact(tmp_path):
    output = tmp_path / "match.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        main([
            "--output", str(output),
            "--selection-reason", "overwrite guard",
            "--paired-seeds", "1",
        ])
