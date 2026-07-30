"""Per-game curriculum-bot routing.

`bot_p0`/`bot_p1` are per-*call*, so a mix of bot configurations forced one
scheduler call per `(bot type, seat)` group. With ~15% of Phase D games split
over up to eight groups, each call drained its own slot pool two or three games
at a time and sent tiny batches into a boundary whose cost is nearly fixed per
batch: bot games ran at ~0.52 games/s at every slot count while the neural group
scaled 0.97 -> 1.40.

`bots_p0`/`bots_p1` take one entry per game, so every game goes in one call.
What must not change is the *assignment* -- which game gets which bot, on which
seat -- and the discrete trajectory. What legitimately does change is batch
composition, and `self_play.rs` documents that batch shape moves recorded
targets by float reduction order. Both halves are gated here separately.
"""

from __future__ import annotations

import pytest
import torch

from .inference import Evaluator
from .rust_bridge import rust_flat_batch_adapter, rust_games_for_self_play
from .test_f4_phase3b_fused import assert_records_identical
from .train import build_model

BOT_A = "military_aggressive/v1"
BOT_B = "science_economy/v1"

# Two bot types on two seats plus pure self-play: four groups over eight games,
# the shape the old path had to split into four calls.
ASSIGNMENT: dict[int, tuple[str, int]] = {
    1: (BOT_A, 0),
    2: (BOT_B, 1),
    4: (BOT_A, 0),
    5: (BOT_A, 1),
    6: (BOT_B, 0),
}
GAMES = 8
SEEDS = [20260727_00 + index for index in range(GAMES)]
FIRST_PLAYERS = [index % 2 for index in range(GAMES)]


def _kwargs():
    return dict(
        global_batch_cap=32,
        leaf_batch=1,
        cheap_sims_min=1,
        cheap_sims_max=2,
        full_sims_min=2,
        full_sims_max=3,
        full_search_fraction=0.3,
        top_k=3,
        draft_prior=0.5,
        iteration=3,
        max_active_slots=4,
        bot_exploration=0.05,
        bot_policy_iterations=10,
    )


@pytest.fixture(scope="module")
def evaluator():
    torch.manual_seed(20260727)
    return Evaluator(build_model("transformer", 32, 2), "cpu", 64)


def _call(evaluator, indices, **extra):
    import seven_wonders_rust as swr

    seeds = [SEEDS[index] for index in indices]
    records, metrics = swr.self_play_many_flat_net(
        adapter=rust_flat_batch_adapter(evaluator),
        games=rust_games_for_self_play(
            seeds, [FIRST_PLAYERS[index] for index in indices]
        ),
        game_seeds=seeds,
        **_kwargs(),
        **extra,
    )
    return records, metrics


def _grouped(evaluator):
    """What the per-call API forced: one call per (bot type, seat) group."""

    keys = [None] + sorted({entry for entry in ASSIGNMENT.values()})
    out: dict[int, dict] = {}
    for key in keys:
        indices = [
            index
            for index in range(GAMES)
            if ASSIGNMENT.get(index) == key
        ]
        if not indices:
            continue
        extra = (
            {}
            if key is None
            else {
                "bot_p0": key[0] if key[1] == 0 else None,
                "bot_p1": key[0] if key[1] == 1 else None,
            }
        )
        records, _ = _call(evaluator, indices, **extra)
        out.update(dict(zip(indices, records)))
    return [out[index] for index in range(GAMES)]


def _single(evaluator):
    """One call for every game, bots assigned per job."""

    records, metrics = _call(
        evaluator,
        list(range(GAMES)),
        bots_p0=[
            ASSIGNMENT[index][0]
            if index in ASSIGNMENT and ASSIGNMENT[index][1] == 0
            else None
            for index in range(GAMES)
        ],
        bots_p1=[
            ASSIGNMENT[index][0]
            if index in ASSIGNMENT and ASSIGNMENT[index][1] == 1
            else None
            for index in range(GAMES)
        ],
    )
    return records, metrics


def test_per_game_routing_assigns_exactly_the_bots_the_groups_did(evaluator):
    """The gate that matters: same bot, same seat, same game.

    Asserted on the recorded `agents`, which is what the buffer and every
    downstream consumer actually see, and exactly (no float involvement).
    """

    single, _ = _single(evaluator)
    grouped = _grouped(evaluator)

    assert [record["seed"] for record in single] == SEEDS, "input order"
    for index, (left, right) in enumerate(zip(grouped, single)):
        assert left["agents"] == right["agents"], f"game {index}"

    # And the assignment is the one the table asks for, not merely a matching
    # pair of wrong answers.
    for index, record in enumerate(single):
        expected = ASSIGNMENT.get(index)
        if expected is None:
            assert record["agents"]["p0"] == "network"
            assert record["agents"]["p1"] == "network"
            assert record["agents"]["kind"] == "self_play"
        else:
            name, seat = expected
            assert record["agents"][f"p{seat}"] == name
            assert record["agents"][f"p{1 - seat}"] == "network"
            assert record["agents"]["kind"] == "mixed"


def test_per_game_routing_changes_no_decision(evaluator):
    """Discrete trajectories exact; recorded targets within float tolerance.

    Batching games together changes reduction order, so targets move -- the same
    class of difference the pool's own gates measure rather than assume away.
    """

    single, _ = _single(evaluator)
    grouped = _grouped(evaluator)
    drift = assert_records_identical(
        grouped, single, target_tolerance=1e-4, context="bot routing: "
    )
    assert drift["max_policy_target_drift"] < 1e-4
    assert drift["max_root_value_drift"] < 1e-4


def test_every_game_shares_one_scheduler_call(evaluator):
    """The point of the change: one drain, not one per group.

    Eight games over four groups used to mean four calls and four pools; the
    metrics of a single call must now cover every game.
    """

    _, metrics = _single(evaluator)
    assert metrics["games"] == GAMES
    assert metrics["max_live_slots"] == min(4, GAMES)


def test_phase_d_generation_issues_one_call_whatever_the_bot_mix(monkeypatch):
    """`_generate_iteration_rust` must not reintroduce grouping."""

    import seven_wonders_rust as swr

    from . import phase_d as pd

    calls = []
    real = swr.self_play_many_flat_net

    def counting(*args, **kwargs):
        calls.append(len(kwargs["game_seeds"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(swr, "self_play_many_flat_net", counting)

    config = pd.PhaseDConfig(
        run_dir="",
        device="cpu",
        games_per_iteration=8,
        seed_games=0,
        d_model=32,
        layers=2,
        cheap_sims_min=1,
        cheap_sims_max=1,
        full_sims_min=1,
        full_sims_max=1,
        full_search_fraction=0.0,
        top_k=2,
        # Force a bot mix: every game rolls against this fraction.
        opponent_fraction=1.0,
    )
    loop = pd.PhaseDLoop.__new__(pd.PhaseDLoop)
    loop.config = config
    loop.last_generation_stats = {}
    torch.manual_seed(11)
    model = build_model("transformer", 32, 2)
    jobs = [pd.GameJob(index=index, seed=900 + index) for index in range(8)]

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        records = loop._generate_iteration_rust(
            model,
            0,
            Path(directory) / "iter.jsonl",
            jobs,
            # Resolved by the caller now, so this bypassed-`__init__` loop needs
            # no games ledger. 1.0 keeps the "every game is a bot game" setup;
            # draft_prior 1.0 is what iteration 0 resolved to before.
            pd.ResolvedSchedules(curriculum_mix_fraction=1.0, draft_prior=1.0),
        )

    assert len(calls) == 1, f"expected one scheduler call, got {len(calls)}"
    assert calls == [8]
    assert len(records) == 8
    assert loop.last_generation_stats["rust_chunks"] == 1
    # opponent_fraction=1.0 means every game is a bot game, spread over the
    # (type, seat) groups the old path would have split into separate calls.
    assert loop.last_generation_stats["rust_bot_games"] == 8
    assert loop.last_generation_stats["rust_games"] == 0
    kinds = {record.agents.get("kind") for record in records}
    assert kinds == {"mixed"}


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"bots_p0": [None] * GAMES}, "must be supplied together"),
        ({"bots_p1": [None] * GAMES}, "must be supplied together"),
        (
            {"bots_p0": [None] * 3, "bots_p1": [None] * GAMES},
            "one entry per game",
        ),
        (
            {
                "bots_p0": [None] * GAMES,
                "bots_p1": [None] * GAMES,
                "bot_p0": BOT_A,
            },
            "supply one form, not both",
        ),
        (
            {"bots_p0": ["no_such_bot"] * GAMES, "bots_p1": [None] * GAMES},
            "unknown Rust bot",
        ),
    ],
)
def test_per_game_bot_arguments_are_validated(evaluator, extra, message):
    with pytest.raises(ValueError, match=message):
        _call(evaluator, list(range(GAMES)), **extra)
