"""Deterministic worker and match boundaries for game-specific AZ loops."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import multiprocessing
import random
from typing import Any, Callable, Protocol, Sequence, TypeVar


class Agent(Protocol):
    name: str

    def select_action(
        self, state: Any, legal_actions: Sequence[int], rng: random.Random
    ) -> int: ...


class GameAdapter(Protocol):
    """The deliberately small engine boundary used by orchestration/evaluation."""

    name: str

    def new_game(self, seed: int, first_player: int = 0) -> Any: ...

    def actor(self, state: Any) -> int: ...

    def legal_actions(self, state: Any) -> Sequence[int]: ...

    def step(self, state: Any, action: int) -> Any: ...

    def terminal(self, state: Any) -> bool: ...

    def outcome(self, state: Any) -> tuple[int | None, tuple[int, int] | None, str]: ...

    def contract(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class GameJob:
    index: int
    seed: int
    kind: str = "self_play"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    seed: int
    first_player: int
    agents: tuple[str, str]
    winner: int | None
    scores: tuple[int, int] | None
    victory_type: str
    actions: int

    def score_for(self, seat: int) -> float:
        if self.winner is None:
            return 0.5
        return 1.0 if self.winner == seat else 0.0


T = TypeVar("T")


def run_jobs(
    jobs: Sequence[GameJob],
    worker: Callable[[GameJob], T],
    *,
    workers: int = 1,
) -> list[T]:
    """Run independent jobs and return results in job-index order.

    Exceptions are never converted into partial training data.  The first
    observed failure cancels work that has not started and is re-raised.
    """

    if workers <= 0:
        raise ValueError("workers must be positive")
    if len({job.index for job in jobs}) != len(jobs):
        raise ValueError("job indices must be unique")
    if workers == 1:
        return [worker(job) for job in sorted(jobs, key=lambda item: item.index)]

    results: dict[int, T] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="az-game") as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        try:
            for future in as_completed(futures):
                job = futures[future]
                results[job.index] = future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return [results[job.index] for job in sorted(jobs, key=lambda item: item.index)]


def run_jobs_in_processes(
    jobs: Sequence[GameJob],
    worker: Callable[[GameJob], T],
    *,
    workers: int = 1,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
) -> list[T]:
    """Process-pool variant of :func:`run_jobs` for GIL-bound workloads.

    ``worker`` and ``initializer`` must be module-level callables and every
    argument/result must pickle. The spawn start method is used on every
    platform so a CUDA context in the parent process is never forked into a
    child. Ordering and failure semantics match :func:`run_jobs`.
    """

    if workers <= 0:
        raise ValueError("workers must be positive")
    if len({job.index for job in jobs}) != len(jobs):
        raise ValueError("job indices must be unique")
    results: dict[int, T] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    ) as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        try:
            for future in as_completed(futures):
                results[futures[future].index] = future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return [results[job.index] for job in sorted(jobs, key=lambda item: item.index)]


def play_match(
    adapter: GameAdapter,
    agents: tuple[Agent, Agent],
    *,
    seed: int,
    first_player: int = 0,
    max_actions: int = 512,
) -> MatchOutcome:
    state = adapter.new_game(seed, first_player)
    rngs = (
        random.Random(seed ^ 0x9E3779B97F4A7C15),
        random.Random(seed ^ 0xD1B54A32D192ED03),
    )
    actions = 0
    while not adapter.terminal(state):
        if actions >= max_actions:
            raise RuntimeError(f"{adapter.name} game exceeded {max_actions} actions")
        actor = adapter.actor(state)
        legal = tuple(adapter.legal_actions(state))
        if not legal:
            raise RuntimeError("non-terminal state has no legal actions")
        action = agents[actor].select_action(state, legal, rngs[actor])
        if action not in legal:
            raise ValueError(f"{agents[actor].name} returned illegal action {action}")
        state = adapter.step(state, action)
        actions += 1
    winner, scores, victory_type = adapter.outcome(state)
    return MatchOutcome(
        seed=seed,
        first_player=first_player,
        agents=(agents[0].name, agents[1].name),
        winner=winner,
        scores=scores,
        victory_type=victory_type,
        actions=actions,
    )
