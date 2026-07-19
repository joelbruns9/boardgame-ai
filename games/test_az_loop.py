from __future__ import annotations

import json
import operator
from pathlib import Path
import random

import pytest

from games.az_loop import (
    EloLedger,
    GameJob,
    HallOfFame,
    LinearSchedule,
    RunManifest,
    SPRT,
    play_match,
    run_jobs,
    run_jobs_in_processes,
)
from games.kingdomino.loop_adapter import KingdominoLoopAdapter


class FirstLegalAgent:
    def __init__(self, name: str):
        self.name = name

    def select_action(self, state, legal_actions, rng: random.Random) -> int:
        return legal_actions[0]


def test_run_jobs_is_ordered_independently_of_submission_order():
    jobs = [GameJob(index=i, seed=100 + i) for i in reversed(range(8))]
    assert run_jobs(jobs, lambda job: job.seed, workers=3) == list(range(100, 108))


def _seed_or_raise(job: GameJob) -> int:
    if job.kind == "poison":
        raise ValueError(f"job {job.index} poisoned")
    return job.seed


def test_run_jobs_in_processes_matches_thread_ordering_semantics():
    jobs = [GameJob(index=i, seed=100 + i) for i in reversed(range(6))]
    getter = operator.attrgetter("seed")
    assert run_jobs_in_processes(jobs, getter, workers=2) == list(range(100, 106))


def test_run_jobs_in_processes_propagates_worker_failure():
    jobs = [
        GameJob(index=0, seed=1),
        GameJob(index=1, seed=2, kind="poison"),
    ]
    with pytest.raises(ValueError, match="poisoned"):
        run_jobs_in_processes(jobs, _seed_or_raise, workers=2)


def test_linear_schedule_and_sprt_reach_expected_decisions():
    schedule = LinearSchedule(1.0, 0.0, 10)
    assert schedule.value(0) == 1.0
    assert schedule.value(5) == 0.5
    assert schedule.value(20) == 0.0

    strong = SPRT(0.45, 0.55)
    while strong.result().decision == "continue":
        strong.update(1.0)
    weak = SPRT(0.45, 0.55)
    while weak.result().decision == "continue":
        weak.update(0.0)
    assert strong.result().decision == "accept"
    assert weak.result().decision == "reject"


def test_hof_elo_and_manifest_are_checkpoint_format_agnostic(tmp_path: Path):
    checkpoint = tmp_path / "candidate.bin"
    checkpoint.write_bytes(b"checkpoint payload")
    hof = HallOfFame(tmp_path / "hof")
    first = hof.add(checkpoint, iteration=3)
    assert hof.add(checkpoint, iteration=4) == first
    assert hof.sample(random.Random(1), "latest") == first

    adapter = KingdominoLoopAdapter()
    outcome = play_match(
        adapter,
        (FirstLegalAgent("a"), FirstLegalAgent("b")),
        seed=7,
        first_player=0,
    )
    ratings = EloLedger(
        tmp_path / "elo", fixed_ratings={"b": 1000.0}
    ).record([outcome])
    assert set(ratings) == {"a", "b"}
    assert ratings["b"] == 1000.0
    assert ratings["a"] != 1000.0

    manifest = RunManifest(tmp_path / "run", Path(__file__).resolve().parents[1])
    manifest.initialize(
        config={"seed": 7},
        adapter_contract=adapter.contract(),
        model_contract={"name": "test"},
    )
    manifest.add_checkpoint(checkpoint, 3, promoted=False)
    manifest.append_iteration({"iteration": 3, "promoted": False})
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    assert payload["adapter_contract"]["adapter"] == "kingdomino"
    assert payload["checkpoints"][0]["iteration"] == 3
    assert payload["iterations"] == [{"iteration": 3, "promoted": False}]


def test_kingdomino_adapter_completes_through_shared_match_runner():
    outcome = play_match(
        KingdominoLoopAdapter(),
        (FirstLegalAgent("left"), FirstLegalAgent("right")),
        seed=19,
    )
    assert outcome.actions > 0
    assert outcome.scores is not None
    assert outcome.winner in (None, 0, 1)
