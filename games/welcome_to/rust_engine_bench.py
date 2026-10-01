"""M1 instrumentation — ``RUST_PORT_PLAN.md`` §7.

⚠ **This is a microbenchmark and says so.** M1 is not wired into the search, so
the only honest measurement available now is the engine on its own: transitions
per second and state clones per second, Python against Rust. The number that
decides whether the port paid for itself is end-to-end leaves/s and games/hour,
and that cannot be measured until M6 — §7 says to say which is which, and 7WD's
experience says the microbenchmark will read high (1.99× became 1.89× on the
real path there, and a +48% concurrency step became +21%).

    python -m games.welcome_to.rust_engine_bench --games 200
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

from games.welcome_to.game import GameConfig, GameState
from games.welcome_to.portable_rng import PortableRng

try:  # pragma: no cover
    import welcome_to_rust as wr
except ImportError:  # pragma: no cover
    wr = None  # type: ignore[assignment]


@dataclass
class Result:
    engine: str
    games: int
    steps: int
    seconds: float

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.seconds

    @property
    def games_per_second(self) -> float:
        return self.games / self.seconds


def _python_game(seed: int, config: GameConfig) -> int:
    state = GameState.new(seed=seed, config=config)
    picker = PortableRng(seed)
    steps = 0
    while not state.is_terminal:
        legal = state.legal_actions()
        state.apply(legal[picker.randrange(len(legal))])
        steps += 1
    return steps


def _rust_game(seed: int, config: GameConfig) -> int:
    state = wr.RustGameState(
        seed,
        players=config.players,
        advanced=config.advanced,
        expert=config.expert,
        solo_rules=config.solo_rules,
    )
    picker = PortableRng(seed)
    steps = 0
    while not state.is_terminal:
        legal = state.legal_actions()
        state.apply(legal[picker.randrange(len(legal))])
        steps += 1
    return steps


def run(engine: str, games: int, config: GameConfig, seed0: int = 0) -> Result:
    play = _python_game if engine == "python" else _rust_game
    started = time.perf_counter()
    steps = sum(play(seed0 + i, config) for i in range(games))
    return Result(engine, games, steps, time.perf_counter() - started)


def clone_rate(engine: str, config: GameConfig, copies: int = 20000) -> float:
    """State clones per second — the operation a search does most, and the one
    a `Vec`/`HashMap` transliteration of the Python representation would have
    made expensive (§6)."""
    if engine == "python":
        state = GameState.new(seed=1, config=config)
        for _ in range(40):
            state.apply(state.legal_actions()[0])
        started = time.perf_counter()
        for _ in range(copies):
            state.copy()
        return copies / (time.perf_counter() - started)
    state = wr.RustGameState(1, players=config.players, advanced=config.advanced)
    for _ in range(40):
        state.apply(state.legal_actions()[0])
    started = time.perf_counter()
    for _ in range(copies):
        state.copy()
    return copies / (time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description="M1 engine microbenchmark")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--advanced", action="store_true", default=True)
    args = parser.parse_args()

    if wr is None:  # pragma: no cover
        raise SystemExit("welcome_to_rust is not built; run `maturin develop --release`")

    config = GameConfig(players=args.players, advanced=args.advanced, solo_rules=False)
    results = [run(engine, args.games, config) for engine in ("python", "rust")]

    print(f"config: {config}, {args.games} uniform-random games")
    for r in results:
        print(
            f"  {r.engine:6s}  {r.steps:7d} actions  {r.seconds:6.2f}s  "
            f"{r.steps_per_second:10.0f} actions/s  {r.games_per_second:8.1f} games/s"
        )
    speedup = results[1].steps_per_second / results[0].steps_per_second
    print(f"  engine-only speed-up: {speedup:.1f}x  (microbenchmark; see the docstring)")

    for engine in ("python", "rust"):
        print(f"  {engine:6s} state clones/s: {clone_rate(engine, config):10.0f}")


if __name__ == "__main__":
    main()
