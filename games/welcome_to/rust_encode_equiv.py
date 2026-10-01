"""M3 bit-exact Python/Rust encoder equivalence gate.

Python's :func:`games.welcome_to.encoder.encode_state` is the oracle.  Every
valid viewer is compared at every primitive state, including the terminal
state, with ``np.array_equal``.  The release gate is::

    python -m games.welcome_to.rust_encode_equiv --encodings 400000
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from games.welcome_to.bots import GreedyBot
from games.welcome_to.encoder import encode_state
from games.welcome_to.game import GameConfig, GameState
from games.welcome_to.portable_rng import PortableRng
from games.welcome_to.rust_encoder import encode_state as rust_encode_state
from games.welcome_to.rust_equiv import DRIVERS, GATE_CONFIGS

try:  # pragma: no cover - optional until the crate is built
    import welcome_to_rust as wr
except ImportError:  # pragma: no cover
    wr = None  # type: ignore[assignment]


OUTPUT_NAMES = ("sheet_planes", "sheet_scalars", "viewer_plane", "global_scalars")


class EncoderDivergence(AssertionError):
    """One Rust feature differs from the Python oracle."""


@dataclass(frozen=True)
class GateReport:
    games: int
    states: int
    encodings: int
    actions: int
    coverage: Counter[tuple[int, bool, str]]


def _choose(
    driver: str,
    state: GameState,
    legal: Sequence[int],
    picker: PortableRng,
    bot: GreedyBot,
) -> int:
    if driver == "greedy":
        return bot.act(state)
    pool = list(legal)
    if driver == "no-refusal":
        from games.welcome_to import action_codec as codec

        pool = [action for action in pool if action != codec.A_PERMIT_REFUSAL] or pool
    return pool[picker.randrange(len(pool))]


def _first_difference(left: np.ndarray, right: np.ndarray) -> str:
    if left.shape != right.shape or left.dtype != right.dtype:
        return f"shape/dtype {left.shape} {left.dtype} != {right.shape} {right.dtype}"
    unequal = np.flatnonzero(left.ravel() != right.ravel())
    if unequal.size == 0:
        # np.array_equal also treats matching NaNs as unequal by default.
        unequal = np.flatnonzero(np.isnan(left.ravel()) | np.isnan(right.ravel()))
    flat = int(unequal[0])
    index = np.unravel_index(flat, left.shape)
    left_value = left[index]
    right_value = right[index]
    left_bits = int(np.asarray(left_value, dtype=np.float32).view(np.uint32))
    right_bits = int(np.asarray(right_value, dtype=np.float32).view(np.uint32))
    return (
        f"index {index}: Python {left_value!r} (0x{left_bits:08x}) != "
        f"Rust {right_value!r} (0x{right_bits:08x})"
    )


def compare_state(py: GameState, rs, *, where: str) -> int:
    """Compare every valid viewer; return the number of encodings checked."""

    for viewer in range(py.config.players):
        left = encode_state(py, viewer)
        right = rust_encode_state(rs, viewer)
        for name, left_array, right_array in zip(
            OUTPUT_NAMES, left, right, strict=True
        ):
            if not np.array_equal(left_array, right_array):
                detail = _first_difference(left_array, right_array)
                raise EncoderDivergence(
                    f"{name} diverged for viewer {viewer} at {where}: {detail}"
                )
    return py.config.players


def check_game(seed: int, config: GameConfig, driver: str) -> tuple[int, int, int]:
    """Return ``(states, encodings, actions)`` for one complete game."""

    if wr is None:  # pragma: no cover
        raise RuntimeError("welcome_to_rust is not built; run maturin develop --release")
    py = GameState.new(seed=seed, config=config)
    rs = wr.RustGameState(
        seed,
        players=config.players,
        advanced=config.advanced,
        expert=config.expert,
        solo_rules=config.solo_rules,
    )
    picker = PortableRng(seed ^ 0x454E_434F_4445_5221)  # "ENCODER!"
    bot = GreedyBot(random.Random(seed))
    states = encodings = actions = 0
    while True:
        where = (
            f"seed {seed}, {config.players}p advanced={config.advanced}, "
            f"{driver}, action {actions}, turn {py.turn}, actor {py.actor}, "
            f"phase {py.phase.name}"
        )
        encodings += compare_state(py, rs, where=where)
        states += 1
        if py.is_terminal:
            break
        py_legal = py.legal_actions()
        rs_legal = rs.legal_actions()
        if py_legal != rs_legal:
            raise EncoderDivergence(f"legal actions diverged at {where}")
        action = _choose(driver, py, py_legal, picker, bot)
        py.apply(action)
        rs.apply(action)
        actions += 1
    return states, encodings, actions


def gate(
    target_encodings: int,
    *,
    seed0: int = 0,
    configs: Sequence[GameConfig] = GATE_CONFIGS,
    drivers: Sequence[str] = DRIVERS,
    progress: int = 25000,
) -> GateReport:
    """Run complete games until at least ``target_encodings`` are compared."""

    if target_encodings < 1:
        raise ValueError("target_encodings must be positive")
    games = states = encodings = actions = 0
    coverage: Counter[tuple[int, bool, str]] = Counter()
    next_progress = progress
    started = time.perf_counter()
    while encodings < target_encodings:
        config = configs[games % len(configs)]
        driver = drivers[(games // len(configs)) % len(drivers)]
        game_states, game_encodings, game_actions = check_game(
            seed0 + games, config, driver
        )
        states += game_states
        encodings += game_encodings
        actions += game_actions
        coverage[(config.players, config.advanced, driver)] += 1
        games += 1
        if progress and encodings >= next_progress:
            elapsed = time.perf_counter() - started
            print(
                f"{encodings:,}/{target_encodings:,} encodings  {games:,} games  "
                f"{encodings / elapsed:,.0f} encodings/s",
                flush=True,
            )
            next_progress = ((encodings // progress) + 1) * progress
    return GateReport(games, states, encodings, actions, coverage)


def main() -> None:
    parser = argparse.ArgumentParser(description="M3 bit-exact Rust encoder gate")
    parser.add_argument("--encodings", type=int, default=400000)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--progress", type=int, default=25000)
    args = parser.parse_args()

    started = time.perf_counter()
    report = gate(args.encodings, seed0=args.seed0, progress=args.progress)
    elapsed = time.perf_counter() - started
    print(
        f"OK: {report.encodings:,} encodings in {report.states:,} states across "
        f"{report.games:,} games and {report.actions:,} actions; zero "
        f"divergences; {elapsed:.1f}s ({report.encodings / elapsed:,.0f}/s)."
    )
    for (players, advanced, driver), count in sorted(report.coverage.items()):
        print(f"  {players}p advanced={int(advanced)} {driver:11s} {count:5d} games")


if __name__ == "__main__":
    main()
