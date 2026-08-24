"""M4 information-key microbenchmark (not end-to-end search throughput).

    python -m games.welcome_to.rust_key_bench --rows 100000
"""

from __future__ import annotations

import argparse
import time

from games.welcome_to import mcts, snapshot
from games.welcome_to.rust_key_equiv import _played_midturn

try:  # pragma: no cover
    import welcome_to_rust as wr
except ImportError:  # pragma: no cover
    wr = None  # type: ignore[assignment]


def _rate(function, rows: int) -> float:
    for _ in range(16):
        function()
    started = time.perf_counter()
    for _ in range(rows):
        function()
    return rows / (time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description="M4 information-key microbenchmark")
    parser.add_argument("--rows", type=int, default=100000)
    args = parser.parse_args()
    if wr is None:  # pragma: no cover
        raise SystemExit("welcome_to_rust is not built; run maturin develop --release")

    py = _played_midturn(players=4)
    viewer = py.actor
    rs = wr.RustGameState.from_snapshot(snapshot.to_snapshot(py))
    python_rate = _rate(lambda: mcts.information_key(py, viewer), args.rows)
    rust_rate = _rate(lambda: rs.information_key(viewer), args.rows)
    key_size = len(rs.information_key(viewer))
    print(f"4p advanced=1 turn={py.turn}, viewer={viewer}; {args.rows:,} rows/backend")
    print(f"  Python: {python_rate:,.0f} keys/s")
    print(f"  Rust:   {rust_rate:,.0f} keys/s ({key_size:,} bytes/key across FFI)")
    print(f"  speed-up: {rust_rate / python_rate:.1f}x (microbenchmark)")


if __name__ == "__main__":
    main()
