"""Blocking M5 throughput A/B (M6 batching is intentionally absent)."""

from __future__ import annotations

import argparse
import time

import numpy as np

from games.welcome_to import encoder as enc
from games.welcome_to import macro_codec as mc
from games.welcome_to import mcts
from games.welcome_to import snapshot
from games.welcome_to.portable_rng import PortableRng, derive_search_seed
from games.welcome_to.rust_search_equiv import collect_positions

import welcome_to_rust as wr


class BenchEvaluator:
    def __init__(self) -> None:
        self.leaves = 0
        self.policies = 0

    @staticmethod
    def _policy(legal) -> np.ndarray:
        result = np.zeros(mc.NUM_MACRO_ACTIONS, dtype=np.float32)
        result[np.asarray(legal, dtype=np.intp)] = np.float32(1.0 / len(legal))
        return result

    def evaluate(self, state, viewer):
        enc.encode_state(state, viewer)
        self.leaves += 1
        return self._policy(mc.legal_macros(state)), 0.25

    def policy(self, state, viewer):
        enc.encode_state(state, viewer)
        self.policies += 1
        return self._policy(mc.legal_macros(state))

    def evaluate_request(self, kind, buffers, legal, viewer, seats, request_id):
        del buffers, viewer, seats, request_id
        if kind == 0:
            self.leaves += 1
        else:
            self.policies += 1
        return self._policy(legal).astype("<f4", copy=False).tobytes(), (
            0.25 if kind == 0 else None
        )


def run(*, positions: int = 48, simulations: int = 32) -> dict[str, float]:
    corpus = collect_positions(positions)
    rust_states = [
        wr.RustGameState.from_snapshot(snapshot.to_snapshot(state)) for state in corpus
    ]
    config = mcts.SearchConfig(simulations=simulations)

    python_eval = BenchEvaluator()
    started = time.perf_counter()
    for index, state in enumerate(corpus):
        seed = derive_search_seed(0xB3EC4, index)
        mcts.MCTS(python_eval, config).search(state, rng=PortableRng(seed))
    python_seconds = time.perf_counter() - started

    rust_eval = BenchEvaluator()
    started = time.perf_counter()
    for index, state in enumerate(rust_states):
        seed = derive_search_seed(0xB3EC4, index)
        wr.RustMcts(simulations=simulations).search(state, rust_eval, seed)
    rust_seconds = time.perf_counter() - started

    total_simulations = positions * simulations
    return {
        "simulations": float(total_simulations),
        "python_seconds": python_seconds,
        "rust_seconds": rust_seconds,
        "python_simulations_s": total_simulations / python_seconds,
        "rust_simulations_s": total_simulations / rust_seconds,
        "speedup": python_seconds / rust_seconds,
        "python_leaves_s": python_eval.leaves / python_seconds,
        "rust_leaves_s": rust_eval.leaves / rust_seconds,
        "python_requests": float(python_eval.leaves + python_eval.policies),
        "rust_requests": float(rust_eval.leaves + rust_eval.policies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=48)
    parser.add_argument("--simulations", type=int, default=32)
    args = parser.parse_args()
    result = run(positions=args.positions, simulations=args.simulations)
    print(
        f"Python {result['python_simulations_s']:.1f} simulations/s, "
        f"{result['python_leaves_s']:.1f} LEAF rows/s, "
        f"{result['python_seconds']:.2f}s"
    )
    print(
        f"Rust   {result['rust_simulations_s']:.1f} simulations/s, "
        f"{result['rust_leaves_s']:.1f} LEAF rows/s, "
        f"{result['rust_seconds']:.2f}s"
    )
    print(
        f"M5 blocking speedup {result['speedup']:.2f}x; "
        f"requests Python/Rust={int(result['python_requests'])}/"
        f"{int(result['rust_requests'])}"
    )


if __name__ == "__main__":
    main()
