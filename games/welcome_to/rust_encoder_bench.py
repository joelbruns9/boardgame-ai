"""M3 encoder microbenchmark, including the diagnostic FFI conversion.

M3 is not wired into search yet, so this reports encoder rows/s only.  The Rust
number includes allocating four immutable byte buffers and exposing NumPy views;
M6's batch-major reusable buffers should be cheaper, but are not credited here.

    python -m games.welcome_to.rust_encoder_bench --rows 10000
"""

from __future__ import annotations

import argparse
import random
import time

from games.welcome_to.bots import GreedyBot
from games.welcome_to.encoder import encode_state
from games.welcome_to.game import GameConfig, GameState
from games.welcome_to.rust_encoder import encode_state as rust_encode_state
from games.welcome_to.snapshot import to_snapshot

try:  # pragma: no cover
    import welcome_to_rust as wr
except ImportError:  # pragma: no cover
    wr = None  # type: ignore[assignment]


def _position(players: int, advanced: bool, turn: int) -> GameState:
    config = GameConfig(players=players, advanced=advanced, solo_rules=False)
    state = GameState.new(seed=71, config=config)
    bots = [GreedyBot(random.Random(7100 + seat)) for seat in range(players)]
    while not state.is_terminal and state.turn < turn:
        state.apply(bots[state.actor].act(state))
    if state.is_terminal:
        raise RuntimeError(f"benchmark position ended before turn {turn}")
    return state


def _rate(function, state, viewers: list[int], rows: int) -> float:
    for viewer in viewers:
        function(state, viewer)
    started = time.perf_counter()
    for row in range(rows):
        function(state, viewers[row % len(viewers)])
    return rows / (time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description="M3 encoder microbenchmark")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--turn", type=int, default=10)
    parser.add_argument("--no-advanced", action="store_true")
    args = parser.parse_args()
    if wr is None:  # pragma: no cover
        raise SystemExit("welcome_to_rust is not built; run maturin develop --release")

    py = _position(args.players, not args.no_advanced, args.turn)
    rs = wr.RustGameState.from_snapshot(to_snapshot(py))
    viewers = list(range(args.players))
    python_rate = _rate(encode_state, py, viewers, args.rows)
    rust_rate = _rate(rust_encode_state, rs, viewers, args.rows)
    print(
        f"{args.players}p advanced={int(not args.no_advanced)} turn={py.turn}; "
        f"{args.rows:,} rows/backend"
    )
    print(f"  Python: {python_rate:,.0f} rows/s")
    print(f"  Rust:   {rust_rate:,.0f} rows/s (packed diagnostic FFI)")
    print(f"  speed-up: {rust_rate / python_rate:.1f}x (microbenchmark)")


if __name__ == "__main__":
    main()
