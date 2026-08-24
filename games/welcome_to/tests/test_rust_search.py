"""Fast M5 checks. The large fixed-tape gate lives in ``rust_search_equiv``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from games.welcome_to import encoder as enc
from games.welcome_to import macro_codec as mc
from games.welcome_to import mcts
from games.welcome_to import network as nw
from games.welcome_to import rust_search
from games.welcome_to import snapshot
from games.welcome_to.game import GameState
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


def _pair(state: GameState, simulations: int = 16, seed: int = 31):
    config = mcts.SearchConfig(simulations=simulations)
    python_eval = FixedEvaluator()
    rust_eval = FixedEvaluator()
    python = mcts.MCTS(python_eval, config)
    rust = wr.RustMcts(simulations=simulations)
    search_seed = derive_search_seed(seed, 0)
    py_actions, py_visits, py_node = python.search(
        state, rng=PortableRng(search_seed)
    )
    rust_state = wr.RustGameState.from_snapshot(snapshot.to_snapshot(state))
    native = rust.search(rust_state, rust_eval, search_seed)
    return python_eval, rust_eval, py_actions, py_visits, py_node, native, rust


def test_fixed_tape_search_matches_at_a_turn_boundary():
    state = GameState.new(seed=7, rng_kind="portable")
    py_eval, rust_eval, actions, visits, node, native, rust = _pair(state)
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


def test_widening_is_refused_until_it_is_implemented():
    with pytest.raises(ValueError, match="chance_widening"):
        wr.RustMcts(chance_widening=1.0)


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
