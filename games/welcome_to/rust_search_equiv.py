"""M5 fixed-tape equivalence gate for the Rust search descent.

The quick pytest sample calls :func:`run_gate` at small scale. The milestone
gate is ``python -m games.welcome_to.rust_search_equiv``: at least 256 played-in
positions, each searched under several independent M0-F seeds.
"""

from __future__ import annotations

import argparse
import collections
import random
import time
import zlib
from dataclasses import dataclass

import numpy as np

from games.welcome_to import encoder as enc
from games.welcome_to import macro_codec as mc
from games.welcome_to import mcts
from games.welcome_to import snapshot
from games.welcome_to.bots import GreedyBot
from games.welcome_to.game import GameConfig, GameState, Phase
from games.welcome_to.portable_rng import PortableRng, derive_search_seed

try:
    import welcome_to_rust as wr
except ImportError as exc:  # pragma: no cover - CLI diagnostic
    raise SystemExit(
        "build the extension with `maturin develop --release` in "
        "games/welcome_to/welcome_to_rust"
    ) from exc


@dataclass(frozen=True, slots=True)
class Request:
    kind: int
    viewer: int
    seats: int
    request_id: int
    legal: tuple[int, ...]
    encoding: tuple[bytes, bytes, bytes, bytes]


class FixedTapeEvaluator:
    """A deterministic evaluator shared by both engines.

    Uniform priors make first-max ordering and opponent weighted sampling do
    real work without introducing Torch reductions into Gate 1. The value is a
    binary fraction, so exact f64 totals are a meaningful requirement.
    """

    def __init__(self, root: int) -> None:
        self.root = root
        self.requests: list[Request] = []

    @staticmethod
    def policy_for(legal) -> np.ndarray:
        result = np.zeros(mc.NUM_MACRO_ACTIONS, dtype=np.float32)
        result[np.asarray(legal, dtype=np.intp)] = np.float32(1.0 / len(legal))
        return result

    def _python_request(self, kind: int, state: GameState, viewer: int) -> None:
        if kind == 0:
            assert viewer == self.root, "a LEAF was not evaluated as the root player"
        else:
            assert viewer == state.actor, "a POLICY was not evaluated as the actor"
        arrays = enc.encode_state(state, viewer)
        self.requests.append(
            Request(
                kind,
                viewer,
                state.config.players,
                len(self.requests),
                tuple(mc.legal_macros(state)),
                tuple(array.astype("<f4", copy=False).tobytes() for array in arrays),  # type: ignore[arg-type]
            )
        )

    def evaluate(self, state: GameState, viewer: int):
        self._python_request(0, state, viewer)
        return self.policy_for(mc.legal_macros(state)), 0.25

    def policy(self, state: GameState, viewer: int):
        self._python_request(1, state, viewer)
        return self.policy_for(mc.legal_macros(state))

    def evaluate_request(
        self, kind, buffers, legal, viewer, seats, request_id
    ) -> tuple[bytes, float | None]:
        if kind == 0:
            assert viewer == self.root, "a Rust LEAF was not attributed to root"
        self.requests.append(
            Request(
                kind,
                viewer,
                seats,
                request_id,
                tuple(legal),
                tuple(bytes(raw) for raw in buffers),  # type: ignore[arg-type]
            )
        )
        priors = self.policy_for(legal)
        return priors.astype("<f4", copy=False).tobytes(), 0.25 if kind == 0 else None


class CountingMcts(mcts.MCTS):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.terminal_leaves = 0

    def _leaf_gen(self, state, root):
        if state.is_terminal:
            self.terminal_leaves += 1
        return (yield from super()._leaf_gen(state, root))


def collect_positions(count: int, seed: int = 0x5EED) -> list[GameState]:
    """Played-in, matrix-balanced decision roots; no constructed legality."""
    configs = [
        GameConfig(players=players, advanced=advanced)
        for players in (2, 3, 4)
        for advanced in (False, True)
    ]
    quotas = [count // len(configs)] * len(configs)
    for index in range(count % len(configs)):
        quotas[index] += 1
    positions: list[GameState] = []
    for cell, (config, quota) in enumerate(zip(configs, quotas)):
        captured = 0
        game_index = 0
        while captured < quota:
            game_seed = seed + cell * 10_000 + game_index
            state = GameState.new(seed=game_seed, config=config, rng_kind="portable")
            driver = PortableRng(game_seed ^ 0xD1A6_005E)
            guard = 0
            while not state.is_terminal and captured < quota:
                legal = mc.search_legal_macros(state, True)
                if len(legal) > 1:
                    positions.append(state.copy())
                    captured += 1
                mc.apply_macro(state, driver.choice(legal))
                guard += 1
                if guard > 10_000:
                    raise RuntimeError("position collector did not terminate")
            game_index += 1
    return positions


def collect_terminal_positions() -> list[GameState]:
    """Last non-forced root-player decision in each supported matrix cell."""
    out: list[GameState] = []
    for cell, (players, advanced) in enumerate(
        (p, a) for p in (2, 3, 4) for a in (False, True)
    ):
        seed = 70_000 + cell
        state = GameState.new(
            seed=seed,
            config=GameConfig(players=players, advanced=advanced),
            rng_kind="portable",
        )
        bots = [GreedyBot(random.Random(seed * 10 + seat)) for seat in range(players)]
        previous = None
        while not state.is_terminal:
            if (
                state.actor == 0
                and state.phase is not Phase.WRITE_NUMBER
                and len(mc.search_legal_macros(state, True)) > 1
            ):
                previous = state.copy()
            state.apply(bots[state.actor].act(state))
        if previous is None:
            raise AssertionError(f"no terminal predecessor for {players}p/{advanced=}")
        out.append(previous)
    return out


def _python_tree(node: mcts.Node, *, widening: bool = False):
    children = []
    outcome_stats = []
    if widening:
        for index, action in enumerate(node.actions):
            for ordinal, (observation, outcome) in enumerate(
                node.outcomes.get(index, {}).items()
            ):
                child = node.children.get((int(action), observation))
                outcome_stats.append(
                    (
                        int(action),
                        ordinal,
                        outcome.count,
                        len(outcome.particles),
                        outcome.terminal_value,
                        child is not None,
                    )
                )
                if child is not None:
                    children.append(
                        (int(action), ordinal, _python_tree(child, widening=True))
                    )
    else:
        ordinal: collections.Counter[int] = collections.Counter()
        for (action, _observation), child in node.children.items():
            children.append((action, ordinal[action], _python_tree(child)))
            ordinal[action] += 1
    children.sort(key=lambda item: (item[0], item[1]))
    outcome_stats.sort(key=lambda item: (item[0], item[1]))
    base = (
        tuple(int(x) for x in node.actions),
        tuple(float(x) for x in node.visits),
        tuple(float(x) for x in node.total),
    )
    if not widening:
        return base + (tuple(children),)
    return base + (
        tuple(node.edge_visits.get(index, 0) for index in range(len(node.actions))),
        tuple(index in node.edge_exact for index in range(len(node.actions))),
        tuple(outcome_stats),
        tuple(children),
    )


def _rust_tree(native, root_node: int, *, widening: bool = False):
    nodes = {node["id"]: node for node in native.debug_tree(root_node)}

    def visit(node_id: int):
        node = nodes[node_id]
        children = tuple(
            (item["action"], item["ordinal"], visit(item["child"]))
            for item in node["outcomes"]
            if item["child"] is not None
        )
        base = (
            tuple(node["actions"]),
            tuple(node["visits"]),
            tuple(node["total"]),
        )
        if not widening:
            return base + (children,)
        outcome_stats = tuple(
            (
                item["action"],
                item["ordinal"],
                item["count"],
                item["particle_count"],
                item["terminal_value"],
                item["child"] is not None,
            )
            for item in node["outcomes"]
        )
        return base + (
            tuple(node["edge_visits"]),
            tuple(node["edge_exact"]),
            outcome_stats,
            children,
        )

    return visit(root_node)


def compare_one(
    state: GameState,
    simulations: int,
    search_seed: int,
    *,
    chance_widening: float | None = 1.0,
    max_particles: int = 4,
) -> dict[str, int]:
    root = state.actor
    config = mcts.SearchConfig(
        simulations=simulations,
        chance_widening=chance_widening,
        max_particles=max_particles,
    )
    python_eval = FixedTapeEvaluator(root)
    rust_eval = FixedTapeEvaluator(root)
    python = CountingMcts(python_eval, config)
    native = wr.RustMcts(
        simulations=simulations,
        chance_widening=chance_widening,
        max_particles=max_particles,
    )

    actions, visits, node = python.search(state, root, PortableRng(search_seed))
    rust_state = wr.RustGameState.from_snapshot(snapshot.to_snapshot(state))
    result = native.search(rust_state, rust_eval, search_seed, root)

    assert result["actions"] == actions.tolist()
    assert np.array_equal(result["visits"], visits)
    assert np.array_equal(result["total"], node.total)
    assert int(actions[int(np.argmax(visits))]) == result["actions"][
        int(np.argmax(result["visits"]))
    ]
    if rust_eval.requests != python_eval.requests:
        shared = min(len(rust_eval.requests), len(python_eval.requests))
        first = next(
            (
                index
                for index in range(shared)
                if rust_eval.requests[index] != python_eval.requests[index]
            ),
            shared,
        )
        rust_request = rust_eval.requests[first] if first < len(rust_eval.requests) else None
        python_request = (
            python_eval.requests[first] if first < len(python_eval.requests) else None
        )
        def summary(request):
            if request is None:
                return None
            return (
                request.kind,
                request.viewer,
                request.seats,
                request.request_id,
                request.legal,
                tuple((len(raw), zlib.crc32(raw)) for raw in request.encoding),
            )
        differences = []
        if rust_request is not None and python_request is not None:
            for column, (rust_raw, python_raw) in enumerate(
                zip(rust_request.encoding, python_request.encoding)
            ):
                rust_values = np.frombuffer(rust_raw, dtype="<f4")
                python_values = np.frombuffer(python_raw, dtype="<f4")
                indices = np.flatnonzero(rust_values != python_values)[:12]
                if len(indices):
                    differences.append(
                        (
                            column,
                            len(np.flatnonzero(rust_values != python_values)),
                            tuple(
                                (int(i), float(rust_values[i]), float(python_values[i]))
                                for i in indices
                            ),
                        )
                    )
        raise AssertionError(
            f"request tape differs at {first}; lengths "
            f"rust={len(rust_eval.requests)} python={len(python_eval.requests)}; "
            f"rust={summary(rust_request)!r}; python={summary(python_request)!r}; "
            f"diffs={differences!r}"
        )
    widening = chance_widening is not None
    rust_tree = _rust_tree(native, result["root_node"], widening=widening)
    python_tree = _python_tree(node, widening=widening)
    if rust_tree != python_tree:
        labels = (
            ("actions", "visits", "total", "edge_visits", "edge_exact", "outcomes", "children")
            if widening
            else ("actions", "visits", "total", "children")
        )
        mismatch = next(
            label
            for label, rust_part, python_part in zip(labels, rust_tree, python_tree)
            if rust_part != python_part
        )
        raise AssertionError(
            f"tree mismatch in root {mismatch}: "
            f"rust={rust_tree[labels.index(mismatch)]!r} "
            f"python={python_tree[labels.index(mismatch)]!r}"
        )
    if widening:
        python_particles = sum(
            len(outcome.particles)
            for _node, _index, outcomes in _all_python_edges(node)
            for outcome in outcomes.values()
        )
        assert native.particle_states_allocated == python_particles
    else:
        assert native.particle_slots_allocated == 0
    assert native.terminal_leaves == python.terminal_leaves
    return {
        "requests": len(python_eval.requests),
        "policy": sum(request.kind == 1 for request in python_eval.requests),
        "terminal": python.terminal_leaves,
    }


def _all_python_edges(root: mcts.Node):
    edges = []

    def walk(node):
        for index, outcomes in node.outcomes.items():
            edges.append((node, index, outcomes))
        for child in node.children.values():
            walk(child)

    walk(root)
    return edges


def run_gate(
    *,
    positions: int = 256,
    seeds_per_position: int = 3,
    simulations: int = 12,
    chance_widening: float | None = 1.0,
) -> dict[str, float]:
    if positions < 1 or seeds_per_position < 1:
        raise ValueError("positions and seeds_per_position must be positive")
    corpus = collect_positions(positions)
    terminal_cases = collect_terminal_positions()
    tail = min(len(corpus), len(terminal_cases))
    corpus[-tail:] = terminal_cases[:tail]
    started = time.perf_counter()
    requests = policy = terminal = 0
    for position_index, state in enumerate(corpus):
        for tape in range(seeds_per_position):
            seed = derive_search_seed(0x4D35 + position_index, tape)
            try:
                counts = compare_one(
                    state,
                    simulations,
                    seed,
                    chance_widening=chance_widening,
                )
            except AssertionError as exc:
                raise AssertionError(
                    f"position={position_index}, tape={tape}, "
                    f"players={state.config.players}, advanced={state.config.advanced}, "
                    f"turn={state.turn}, actor={state.actor}, phase={state.phase.name}: {exc}"
                ) from exc
            requests += counts["requests"]
            policy += counts["policy"]
            terminal += counts["terminal"]
    elapsed = time.perf_counter() - started
    return {
        "positions": float(len(corpus)),
        "searches": float(len(corpus) * seeds_per_position),
        "simulations": float(len(corpus) * seeds_per_position * simulations),
        "requests": float(requests),
        "policy_requests": float(policy),
        "terminal_leaves": float(terminal),
        "seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=12)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--chance-widening", type=float, default=1.0)
    mode.add_argument("--control", action="store_true")
    args = parser.parse_args()
    result = run_gate(
        positions=args.positions,
        seeds_per_position=args.seeds,
        simulations=args.simulations,
        chance_widening=None if args.control else args.chance_widening,
    )
    print(
        "M5 fixed-tape gate green"
        f" ({'control' if args.control else f'PW C={args.chance_widening:g}'}): "
        f"{int(result['positions'])} positions × {args.seeds} seeds, "
        f"{int(result['simulations']):,} simulations, "
        f"{int(result['requests']):,} requests "
        f"({int(result['policy_requests']):,} POLICY), "
        f"{int(result['terminal_leaves']):,} terminal leaves, "
        f"{result['seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
