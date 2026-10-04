"""Fast playouts (2026-10-03): batched parallel encoding and the top-move picker
must be exact replacements for the per-state path."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from games.welcome_to import mcts, network as nw, rust_search

wr = pytest.importorskip("welcome_to_rust")


def _states(count=300, seed=1):
    rng = random.Random(seed)
    out = []
    for game_seed in range(40):
        players = 2 + game_seed % 3
        state = wr.RustGameState(game_seed, players=players, advanced=True, expert=False, solo_rules=False)
        while not state.is_terminal and len(out) < count:
            if rng.random() < 0.3:
                out.append((state.step_macro(rng.choice(state.legal_macros())), players))
            state.apply_macro(rng.choice(state.legal_macros()))
        if len(out) >= count:
            break
    return [(s, n) for s, n in out if not s.is_terminal]


def test_encode_batch_is_byte_identical_to_encode_state_and_legal_macros():
    states = [s for s, _ in _states()]
    out = wr.encode_batch(states)
    serial = [s.encode_state(None) for s in states]
    for k in range(4):
        assert bytes(out[k]) == b"".join(bytes(row[k]) for row in serial), k
    legal = np.frombuffer(out[4], dtype="<u2")
    offsets = np.frombuffer(out[5], dtype="<u4")
    assert len(offsets) == len(states) + 1
    for i, state in enumerate(states):
        assert list(legal[offsets[i] : offsets[i + 1]]) == state.legal_macros()


def test_encode_batch_does_not_depend_on_the_worker_count(monkeypatch):
    states = [s for s, _ in _states(120, seed=2)]
    monkeypatch.setenv("WTO_ENCODE_THREADS", "1")
    one = wr.encode_batch(states)
    monkeypatch.setenv("WTO_ENCODE_THREADS", "7")
    seven = wr.encode_batch(states)
    assert all(bytes(a) == bytes(b) for a, b in zip(one, seven))
    assert wr.encode_batch([])[0] == b""


def test_best_moves_matches_the_policy_states_argmax():
    torch.manual_seed(4)
    net = nw.WelcomeToNet(nw.NetConfig(sheet_hidden=16, sheet_out=8, trunk_hidden=24, trunk_blocks=1, head_hidden=16, plan_hidden=16, plan_out=8)).eval()
    packed = rust_search.PackedNetEvaluator(net, torch.device("cpu"), mcts.SearchConfig(simulations=2))
    pairs = _states(250, seed=3)
    states, seats = [s for s, _ in pairs], [n for _, n in pairs]
    fast = packed.best_moves(states, seats)
    policies, legals = packed.policy_states(states, seats)
    slow = [max(legal, key=lambda m: (float(policy[m]), -m)) for policy, legal in zip(policies, legals)]
    assert fast == slow
