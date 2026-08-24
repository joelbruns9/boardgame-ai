"""Fast M5 checks. The large fixed-tape gate lives in ``rust_search_equiv``."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch

from games.welcome_to import encoder as enc
from games.welcome_to import macro_codec as mc
from games.welcome_to import mcts
from games.welcome_to import network as nw
from games.welcome_to import rust_search
from games.welcome_to import snapshot
from games.welcome_to.bots import GreedyBot
from games.welcome_to.game import GameConfig, GameState, Phase
from games.welcome_to.portable_rng import PortableRng, derive_search_seed

wr = pytest.importorskip("welcome_to_rust")


def test_the_evaluator_abi_is_the_f64_value_revision():
    assert wr.EVALUATOR_ABI_VERSION == rust_search.EVALUATOR_ABI_VERSION == 2


class FixedEvaluator:
    """One exact binary-fraction value and a uniform full-legal policy."""

    def __init__(self) -> None:
        self.requests: list[tuple] = []

    @staticmethod
    def _policy(legal) -> np.ndarray:
        out = np.zeros(mc.NUM_MACRO_ACTIONS, dtype=np.float32)
        out[np.asarray(legal, dtype=np.intp)] = np.float32(1.0 / len(legal))
        return out

    def _record(self, kind: int, state: GameState, viewer: int) -> None:
        arrays = enc.encode_state(state, viewer)
        self.requests.append(
            (
                kind,
                viewer,
                state.config.players,
                tuple(mc.legal_macros(state)),
                tuple(array.astype("<f4", copy=False).tobytes() for array in arrays),
            )
        )

    def evaluate(self, state, viewer):
        self._record(0, state, viewer)
        return self._policy(mc.legal_macros(state)), 0.25

    def policy(self, state, viewer):
        self._record(1, state, viewer)
        return self._policy(mc.legal_macros(state))

    def evaluate_request(self, kind, buffers, legal, viewer, seats, request_id):
        del request_id
        self.requests.append(
            (kind, viewer, seats, tuple(legal), tuple(bytes(raw) for raw in buffers))
        )
        return self._policy(legal).astype("<f4", copy=False).tobytes(), (
            0.25 if kind == 0 else None
        )


class FixedBatchEvaluator:
    """Shape-invariant implementation of the frozen M0-E V2 batch ABI."""

    def __init__(self, *, reverse: bool = False) -> None:
        self.reverse = reverse
        self.batch_sizes: list[int] = []
        self.ids: list[int] = []
        self.kind_batches: list[tuple[int, ...]] = []
        self.payloads: list[dict[str, bytes]] = []

    def forward(self, batch):
        assert batch["version"] == 2
        rows = batch["rows"]
        ids = np.frombuffer(batch["request_id"], dtype="<u4").copy()
        kinds = np.frombuffer(batch["kind"], dtype=np.uint8).copy()
        seats = np.frombuffer(batch["seats"], dtype=np.uint8)
        offsets = np.frombuffer(batch["legal_offsets"], dtype="<u4")
        legal = np.frombuffer(batch["legal_indices"], dtype="<u2")
        assert len(ids) == len(kinds) == len(seats) == rows
        assert len(offsets) == rows + 1 and offsets[0] == 0
        assert offsets[-1] == len(legal)
        self.batch_sizes.append(rows)
        self.ids.extend(ids.tolist())
        self.kind_batches.append(tuple(kinds.tolist()))
        self.payloads.append(
            {
                name: bytes(batch[name])
                for name in (
                    "sheet_planes",
                    "sheet_scalars",
                    "viewer_plane",
                    "global_scalars",
                )
            }
        )

        priors = np.zeros((rows, mc.NUM_MACRO_ACTIONS), dtype="<f4")
        for row in range(rows):
            actions = legal[offsets[row] : offsets[row + 1]].astype(np.intp)
            priors[row, actions] = np.float32(1.0 / len(actions))
        values = np.where(kinds == 0, 0.25, 0.0).astype("<f8")
        order = np.arange(rows - 1, -1, -1) if self.reverse else np.arange(rows)
        return {
            "version": 2,
            "rows": rows,
            "request_id": ids[order].astype("<u4", copy=False).tobytes(),
            "priors": priors[order].astype("<f4", copy=False).tobytes(),
            "values": values[order].astype("<f8", copy=False).tobytes(),
        }


def _pair(state: GameState, simulations: int = 16, seed: int = 31, **kwargs):
    config = mcts.SearchConfig(simulations=simulations, **kwargs)
    python_eval = FixedEvaluator()
    rust_eval = FixedEvaluator()
    python = mcts.MCTS(python_eval, config)
    rust = rust_search.native_search(config)
    search_seed = derive_search_seed(seed, 0)
    py_actions, py_visits, py_node = python.search(
        state, rng=PortableRng(search_seed)
    )
    rust_state = wr.RustGameState.from_snapshot(snapshot.to_snapshot(state))
    native = rust.search(rust_state, rust_eval, search_seed)
    return python_eval, rust_eval, py_actions, py_visits, py_node, native, rust


def test_fixed_tape_search_matches_at_a_turn_boundary():
    state = GameState.new(seed=7, rng_kind="portable")
    py_eval, rust_eval, actions, visits, node, native, rust = _pair(
        state, chance_widening=None
    )
    assert native["actions"] == actions.tolist()
    assert np.array_equal(native["visits"], visits)
    assert np.array_equal(native["total"], node.total)
    assert rust_eval.requests == py_eval.requests
    assert rust.particle_slots_allocated == 0
    assert sum(native["visits"]) == 16


def test_search_seed_derivation_is_shared():
    assert wr.derive_search_seed(123, 7) == derive_search_seed(123, 7)


def test_wide_float32_weight_accumulation_is_shared():
    weights = np.full(331, np.float32(1.0 / 331), dtype=np.float32)
    rng = PortableRng(0xBAD5EED)
    python = [rng.choices(range(331), weights=weights, k=1)[0] for _ in range(256)]
    rust = wr.portable_rng_weighted_indices(
        0xBAD5EED, [float(weight) for weight in weights], 256
    )
    assert rust == python


def test_widening_configuration_is_accepted_and_invalid_values_fail_loudly():
    assert mcts.SearchConfig().chance_widening == 1.0
    wr.RustMcts(chance_widening=1.0, chance_widening_alpha=0.5, max_particles=4)
    for value in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="chance_widening"):
            wr.RustMcts(chance_widening=value)
    with pytest.raises(ValueError, match="chance_widening_alpha"):
        wr.RustMcts(chance_widening=1.0, chance_widening_alpha=1.1)
    with pytest.raises(ValueError, match="max_particles"):
        wr.RustMcts(chance_widening=1.0, max_particles=0)


def _played_in_position(turn: int = 12) -> GameState:
    seed = 7
    state = GameState.new(
        seed=seed,
        config=GameConfig(players=2, advanced=True),
        rng_kind="portable",
    )
    bots = [GreedyBot(random.Random(seed * 10 + seat)) for seat in range(2)]
    while not state.is_terminal and state.turn < turn:
        state.apply(bots[state.actor].act(state))
    while not state.is_terminal and (
        state.actor != 0 or state.phase is Phase.WRITE_NUMBER
    ):
        state.apply(bots[state.actor].act(state))
    return state


def test_progressive_widening_matches_python_and_reuses_retained_particles():
    state = _played_in_position()
    py_eval, rust_eval, actions, visits, node, native, rust = _pair(
        state,
        simulations=128,
        seed=123,
        max_particles=4,
    )
    assert native["actions"] == actions.tolist()
    assert np.array_equal(native["visits"], visits)
    assert np.array_equal(native["total"], node.total)
    assert rust_eval.requests == py_eval.requests

    traversals = fresh_draws = exact_edges = 0
    for debug_node in rust.debug_tree(native["root_node"]):
        by_action: dict[int, list[dict]] = {}
        for outcome in debug_node["outcomes"]:
            by_action.setdefault(outcome["action"], []).append(outcome)
            assert outcome["count"] >= 1
            assert outcome["particle_count"] <= 4
        for index, action in enumerate(debug_node["actions"]):
            edge_visits = debug_node["edge_visits"][index]
            outcomes = by_action.get(action, [])
            if not outcomes:
                continue
            allowed = max(math.ceil(edge_visits**0.5), 1)
            assert len(outcomes) <= allowed
            traversals += edge_visits
            fresh_draws += sum(outcome["count"] for outcome in outcomes)
            if debug_node["edge_exact"][index]:
                exact_edges += 1
                assert len(outcomes) == 1
                assert outcomes[0]["count"] == 1
                assert outcomes[0]["particle_count"] == 0
            else:
                assert all(
                    outcome["particle_count"]
                    or outcome["terminal_value"] is not None
                    for outcome in outcomes
                )

    assert exact_edges > 0
    assert traversals > fresh_draws, "no widened edge reused a retained outcome"
    assert rust.particle_slots_allocated > 0
    assert 0 < rust.particle_states_allocated <= 4 * rust.particle_slots_allocated


def test_malformed_evaluator_responses_fail_loudly():
    state = wr.RustGameState.from_snapshot(
        snapshot.to_snapshot(GameState.new(seed=3, rng_kind="portable"))
    )

    class Bad:
        def __init__(self, fill):
            self.fill = fill

        def evaluate_request(self, *_args):
            priors = np.full(mc.NUM_MACRO_ACTIONS, self.fill, dtype="<f4")
            return priors.tobytes(), 0.0

    with pytest.raises(ValueError, match="illegal macro"):
        wr.RustMcts(simulations=1).search(state, Bad(1.0), 1)
    with pytest.raises(RuntimeError, match="positive finite mass"):
        wr.RustMcts(simulations=1).search(state, Bad(0.0), 1)


def test_one_row_real_network_search_has_an_exact_fingerprint():
    torch.manual_seed(19)
    net = nw.WelcomeToNet(
        nw.NetConfig(
            sheet_hidden=32,
            sheet_out=16,
            trunk_hidden=48,
            trunk_blocks=1,
            head_hidden=32,
        )
    ).eval()
    config = mcts.SearchConfig(simulations=8)
    state = GameState.new(seed=23, rng_kind="portable")
    seed = derive_search_seed(23, 0)

    python = mcts.MCTS(mcts.NetEvaluator(net, torch.device("cpu"), config), config)
    actions, visits, node = python.search(state, rng=PortableRng(seed))
    packed = rust_search.PackedNetEvaluator(net, torch.device("cpu"), config)
    native = rust_search.native_search(config)
    rust_state = wr.RustGameState.from_snapshot(snapshot.to_snapshot(state))
    result = native.search(rust_state, packed, seed)

    assert result["actions"] == actions.tolist()
    assert np.array_equal(result["visits"], visits)
    assert np.array_equal(result["total"], node.total)


def test_forced_nodes_are_collapsed_without_an_evaluation():
    # Seed 1's first macro enters ACTION_PARK, whose dominated pass is pruned;
    # the build is applied but must not become a one-action node/LEAF request.
    state = GameState.new(seed=1, rng_kind="portable")
    first = mc.search_legal_macros(state, True)[0]
    forced = mc.step_macro(state, first)
    assert len(mc.search_legal_macros(forced, True)) == 1
    forced_encoding = tuple(
        array.astype("<f4", copy=False).tobytes()
        for array in enc.encode_state(forced, state.actor)
    )
    py_eval, rust_eval, *_ = _pair(state, simulations=8)
    assert py_eval.requests == rust_eval.requests
    assert all(
        not (kind == 0 and encoding == forced_encoding)
        for kind, _viewer, _seats, _legal, encoding in py_eval.requests
    )


def test_play_reuses_the_exact_within_turn_subtree():
    # Seed 0's first macro enters a six-way estate decision. A budget above the
    # opening width revisits that child, making reuse observable rather than
    # merely incrementing the reroot counter on a zero-visit node.
    state = GameState.new(seed=0, rng_kind="portable")
    rust_state = wr.RustGameState.from_snapshot(snapshot.to_snapshot(state))
    config = mcts.SearchConfig(simulations=240)
    py_eval = FixedEvaluator()
    rust_eval = FixedEvaluator()
    python = mcts.MCTS(py_eval, config)
    native = wr.RustMcts(simulations=240)

    first_seed = derive_search_seed(0, 0)
    py_choice = python.play(state, rng=PortableRng(first_seed))
    rust_first = native.play(rust_state, rust_eval, first_seed)
    assert rust_first["choice"] == py_choice
    mc.apply_macro(state, py_choice)
    rust_state.apply_macro(py_choice)
    assert state.turn == 1 and len(mc.search_legal_macros(state, True)) > 1

    second_seed = derive_search_seed(0, 1)
    py_second = python.play(state, rng=PortableRng(second_seed))
    rust_second = native.play(rust_state, rust_eval, second_seed)
    assert rust_second["choice"] == py_second
    assert python.reroots == native.reroots == 1
    assert python.simulations_reused == native.simulations_reused > 0
    assert py_eval.requests == rust_eval.requests


def test_python_generated_root_noise_preserves_the_fixed_tape():
    state = GameState.new(seed=7, rng_kind="portable")
    config = mcts.SearchConfig(
        simulations=24, dirichlet_concentration=10.0, dirichlet_weight=0.25
    )
    seed = derive_search_seed(7, 0)
    py_eval = FixedEvaluator()
    rust_eval = FixedEvaluator()
    python = mcts.MCTS(py_eval, config)
    actions, visits, node = python.search(state, rng=PortableRng(seed))

    tape = PortableRng(seed)
    noise, advanced_seed = rust_search.root_noise(config, len(actions), tape)
    assert noise is not None
    native = rust_search.native_search(config)
    rust_state = wr.RustGameState.from_snapshot(snapshot.to_snapshot(state))
    result = native.search(
        rust_state, rust_eval, advanced_seed, noise=noise.tolist()
    )
    assert result["actions"] == actions.tolist()
    assert np.array_equal(result["visits"], visits)
    assert np.array_equal(result["total"], node.total)
    assert py_eval.requests == rust_eval.requests
    assert native.debug_tree(result["root_node"])[0]["noised"] is True


def test_m6_scheduler_coalesces_searches_and_routes_reordered_rows():
    python_states = [GameState.new(seed=seed, rng_kind="portable") for seed in (11, 12, 13, 14)]
    states = [
        wr.RustGameState.from_snapshot(
            snapshot.to_snapshot(state)
        )
        for state in python_states
    ]
    seeds = [derive_search_seed(seed, 0) for seed in (11, 12, 13, 14)]
    evaluator = FixedBatchEvaluator(reverse=True)
    scheduler = wr.RustScheduler(capacity=4, simulations=16)
    waved = scheduler.search(states, evaluator, seeds, max_batch=4)

    independent = []
    for state, seed in zip(states, seeds):
        independent.append(wr.RustMcts(simulations=16).search(state, FixedEvaluator(), seed))
    for left, right in zip(waved, independent):
        assert left["actions"] == right["actions"]
        assert np.array_equal(left["visits"], right["visits"])
        assert np.array_equal(left["total"], right["total"])
    assert max(evaluator.batch_sizes) == 4
    assert len(evaluator.ids) == len(set(evaluator.ids))
    assert set(kind for batch in evaluator.kind_batches for kind in batch) == {0, 1}
    first = evaluator.payloads[0]
    encoded = [enc.encode_state(state, state.actor) for state in python_states]
    for name, column in zip(
        ("sheet_planes", "sheet_scalars", "viewer_plane", "global_scalars"),
        zip(*encoded),
    ):
        expected = b"".join(
            array.astype("<f4", copy=False).tobytes() for array in column
        )
        assert first[name] == expected


def test_m6_real_network_has_the_same_discrete_search_fingerprint_at_batch_four():
    torch.manual_seed(29)
    net = nw.WelcomeToNet(
        nw.NetConfig(
            sheet_hidden=32,
            sheet_out=16,
            trunk_hidden=48,
            trunk_blocks=1,
            head_hidden=32,
        )
    ).eval()
    config = mcts.SearchConfig(simulations=12)
    states = [
        wr.RustGameState.from_snapshot(
            snapshot.to_snapshot(GameState.new(seed=seed, rng_kind="portable"))
        )
        for seed in (31, 32, 33, 34)
    ]
    seeds = [derive_search_seed(seed, 0) for seed in (31, 32, 33, 34)]
    scalar = rust_search.native_scheduler(config, capacity=4).search(
        states,
        rust_search.PackedNetEvaluator(net, torch.device("cpu"), config),
        seeds,
        max_batch=1,
    )
    evaluator = rust_search.PackedNetEvaluator(net, torch.device("cpu"), config)
    waved = rust_search.native_scheduler(config, capacity=4).search(
        states, evaluator, seeds, max_batch=4
    )
    assert evaluator.calls < evaluator.rows
    for left, right in zip(waved, scalar):
        assert left["actions"] == right["actions"]
        assert np.array_equal(left["visits"], right["visits"])


def test_m6_scheduler_preserves_slots_for_within_turn_rerooting():
    state = wr.RustGameState.from_snapshot(
        snapshot.to_snapshot(GameState.new(seed=0, rng_kind="portable"))
    )
    scheduler = wr.RustScheduler(capacity=1, simulations=240)
    evaluator = FixedBatchEvaluator()
    first = scheduler.play([state], evaluator, [derive_search_seed(0, 0)])[0]
    state.apply_macro(first["choice"])
    scheduler.play([state], evaluator, [derive_search_seed(0, 1)])
    stats = scheduler.stats(0)
    assert stats["reroots"] == 1
    assert stats["simulations_reused"] > 0


def test_m6_evaluator_failure_wakes_every_search_and_preserves_the_python_error():
    states = [
        wr.RustGameState.from_snapshot(
            snapshot.to_snapshot(GameState.new(seed=seed, rng_kind="portable"))
        )
        for seed in (41, 42, 43)
    ]

    class Failing:
        def forward(self, _batch):
            raise RuntimeError("M6 batch failed at the source")

    scheduler = wr.RustScheduler(capacity=3, simulations=8)
    with pytest.raises(RuntimeError, match="M6 batch failed at the source"):
        scheduler.search(states, Failing(), [1, 2, 3])
    # The worker-owned searches were returned to their slots even on failure.
    scheduler.reset()
    assert len(scheduler.search(states, FixedBatchEvaluator(), [1, 2, 3])) == 3


def test_m6_rejects_a_misaligned_response_before_routing_any_row():
    states = [
        wr.RustGameState.from_snapshot(
            snapshot.to_snapshot(GameState.new(seed=seed, rng_kind="portable"))
        )
        for seed in (44, 45)
    ]

    class DuplicateId(FixedBatchEvaluator):
        def forward(self, batch):
            response = super().forward(batch)
            ids = np.frombuffer(response["request_id"], dtype="<u4").copy()
            ids[1] = ids[0]
            response["request_id"] = ids.tobytes()
            return response

    scheduler = wr.RustScheduler(capacity=2, simulations=4)
    with pytest.raises(ValueError, match="duplicate request_id"):
        scheduler.search(states, DuplicateId(), [1, 2])
    scheduler.reset()


def test_m6_scheduler_rejects_duplicate_slots_before_starting_workers():
    states = [
        wr.RustGameState.from_snapshot(
            snapshot.to_snapshot(GameState.new(seed=seed, rng_kind="portable"))
        )
        for seed in (51, 52)
    ]
    with pytest.raises(ValueError, match="slots must be unique"):
        wr.RustScheduler(capacity=2, simulations=2).search(
            states, FixedBatchEvaluator(), [1, 2], slots=[0, 0]
        )
