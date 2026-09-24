"""Tests for the value net and the solver-leaf adapter.

Run: python -m pytest games/cantstop/tests/test_model.py -q
"""

import random

import numpy as np
import pytest
import torch

from games.cantstop.engine import (
    ALL_RULESETS, GameState, Phase, RuleSet, legal_moves, random_dice, roll,
)
from games.cantstop.encoder import (
    FEATURE_SIZE, MAX_SEATS, encode_batch, seat_order, seat_to_slot,
)
from games.cantstop.model import (
    CantStopNet, NetEvaluator, load_net, masked_cross_entropy,
    masked_softmax, save_net, seat_mask_tensor,
)
from games.cantstop.solver import ProgressHeuristic, TurnSolver


def fresh(num_players=3, extended=True, blocking=True):
    return GameState(RuleSet.make(num_players, extended, blocking))


def small_net(seed=0):
    torch.manual_seed(seed)
    return CantStopNet(hidden=(32, 32))


def rolled(state, rng):
    """Advance to a state with dice on the table (phase AWAIT_MOVE)."""
    while True:
        if roll(state, random_dice(rng)):
            return state


# ---- masking ----

def test_masked_softmax_zeroes_masked_slots_and_normalizes():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, True, False, False]])
    out = masked_softmax(logits, mask)
    assert out[0, 2] == 0.0 and out[0, 3] == 0.0
    assert out.sum().item() == pytest.approx(1.0, abs=1e-6)
    # The live slots keep their relative odds.
    assert (out[0, 1] / out[0, 0]).item() == pytest.approx(np.e, abs=1e-5)


def test_seat_mask_comes_from_the_features_themselves():
    for n in (2, 3, 4):
        x = torch.from_numpy(encode_batch([fresh(num_players=n,
                                                 extended=False)]))
        mask = seat_mask_tensor(x)
        assert mask.tolist() == [[i < n for i in range(MAX_SEATS)]]


def test_win_probs_never_feeds_an_absent_seat():
    net = small_net()
    x = torch.from_numpy(encode_batch([fresh(num_players=2, extended=False)]))
    p = net.win_probs(x)
    assert p[0, 2].item() == 0.0 and p[0, 3].item() == 0.0
    assert p.sum().item() == pytest.approx(1.0, abs=1e-6)


# ---- net basics ----

def test_forward_shape():
    net = small_net()
    x = torch.zeros(5, FEATURE_SIZE)
    assert net(x).shape == (5, MAX_SEATS)


def test_save_and_load_round_trip(tmp_path):
    net = small_net(seed=3)
    path = tmp_path / "net.pt"
    save_net(net, path)
    back = load_net(path)
    assert back.config() == net.config()
    x = torch.from_numpy(encode_batch([fresh()]))
    assert torch.allclose(net(x), back(x))


# ---- evaluator contract ----

def test_evaluator_matches_the_heuristic_contract():
    """Same shape and normalization as ProgressHeuristic, for every rule set."""
    net = small_net()
    rng = random.Random(5)
    for rules in ALL_RULESETS:
        states = [GameState(rules) for _ in range(3)]
        for s in states:
            s.active_player = rng.randrange(rules.num_players)
        ours = NetEvaluator(net, device="cpu")(states)
        theirs = ProgressHeuristic()(states)
        assert ours.shape == theirs.shape == (3, rules.num_players)
        assert np.allclose(ours.sum(axis=1), 1.0)
        assert np.all(ours >= 0.0)


def test_evaluator_returns_absolute_seat_order():
    """Slot 0 of the net is the seat to move; the solver indexes by absolute
    seat. Check the rotation directly against the net's own raw output.

    Note the outputs for different movers are NOT rotations of each other:
    the encoding is seat-relative, so changing who moves changes the board
    the net sees. The invariant is only that slot k lands at seat_order[k].
    """
    net = small_net(seed=1)
    ev = NetEvaluator(net, device="cpu")
    base = fresh(num_players=3, extended=False, blocking=False)
    base.progress[0][7] = 6

    for seat in range(3):
        s = base.clone()
        s.active_player = seat
        raw = net.win_probs(
            torch.from_numpy(encode_batch([s]))).detach().numpy()[0]
        out = ev([s])[0]
        for slot, absolute in enumerate(seat_order(s)):
            assert out[absolute] == pytest.approx(raw[slot], abs=1e-6)


def test_evaluator_returns_float64():
    """The solver accumulates leaf values thousands of times in backward
    induction, and the Rust port is specified to match in f64."""
    out = NetEvaluator(small_net(), device="cpu")([fresh()])
    assert out.dtype == np.float64


def test_evaluator_rejects_an_empty_batch():
    with pytest.raises(ValueError, match="no boards"):
        NetEvaluator(small_net(), device="cpu")([])


def test_evaluator_chunking_matches_one_shot():
    net = small_net(seed=2)
    rng = random.Random(9)
    states = []
    for _ in range(7):
        s = fresh(num_players=4, extended=False)
        s.progress[0][7] = rng.randrange(13)
        states.append(s)
    one = NetEvaluator(net, device="cpu")(states)
    many = NetEvaluator(net, device="cpu", batch_size=2)(states)
    assert np.allclose(one, many)


# ---- the integration that matters ----

def test_solver_runs_on_the_net_and_picks_legal_moves():
    """The whole point: the net is a drop-in leaf for TurnSolver."""
    net = small_net(seed=4)
    ev = NetEvaluator(net, device="cpu")
    rng = random.Random(13)
    for rules in ALL_RULESETS:
        s = rolled(GameState(rules), rng)
        solver = TurnSolver(s, ev)
        move = solver.choose_move(s)
        assert move in legal_moves(s, s.dice)
        value = solver.value(s)
        assert value.shape == (rules.num_players,)
        assert value.sum() == pytest.approx(1.0, abs=1e-6)


def test_one_solve_costs_exactly_one_evaluator_call():
    """The batching claim the Rust-port plan rests on: every leaf in a turn,
    plus the shared bust board, goes over in a single call."""
    net = small_net(seed=6)
    ev = NetEvaluator(net, device="cpu")
    s = rolled(fresh(num_players=3, extended=True, blocking=False),
               random.Random(21))
    solver = TurnSolver(s, ev)
    assert ev.calls == 1
    assert ev.rows == solver.evaluator_calls > 1


def test_solver_stops_to_take_a_win_whatever_the_net_says():
    """Solver decisions track leaf values, and a winning stop is scored
    exactly rather than by the net -- so this holds for any evaluator."""
    s = fresh(num_players=2, extended=False, blocking=False)
    s.active_player = 0
    s.claimed_by[2] = 0
    s.claimed_by[3] = 0          # two of the three columns already won
    s.runners = {12: 3}          # column 12 is 3 tall: this runner is at the top
    s.phase = Phase.AWAIT_DECISION

    def net_hates_the_mover(states):
        out = np.zeros((len(states), 2))
        out[:, 0] = 0.01         # tell the solver seat 0 is losing badly
        out[:, 1] = 0.99
        return out

    solver = TurnSolver(s, net_hates_the_mover)
    assert solver.should_stop(s)
    assert solver.value(s)[0] == pytest.approx(1.0)


def test_solver_rolls_on_when_there_is_nothing_banked_to_lose():
    """The mirror case: two runners barely on the board and a free third
    runner, so busting costs almost nothing and rolling is preferred."""
    s = fresh(num_players=2, extended=False, blocking=False)
    s.active_player = 0
    s.runners = {4: 1, 11: 1}
    s.phase = Phase.AWAIT_DECISION

    solver = TurnSolver(s, ProgressHeuristic())
    assert not solver.should_stop(s)


# ---- training step ----

def test_masked_cross_entropy_ignores_absent_seats():
    logits = torch.tensor([[0.0, 0.0, 50.0, 50.0]])
    mask = torch.tensor([[True, True, False, False]])
    target = torch.tensor([0])
    loss = masked_cross_entropy(logits, target, mask)
    # Two live slots with equal logits: ln 2, regardless of the dead ones.
    assert loss.item() == pytest.approx(np.log(2), abs=1e-5)


def test_the_net_can_fit_a_tiny_labelled_set():
    """Smoke test that the training path works end to end: encode boards,
    label with a winner slot, minimize, and watch the loss fall."""
    rng = random.Random(41)
    states, targets = [], []
    for _ in range(64):
        s = fresh(num_players=3, extended=False, blocking=False)
        s.active_player = rng.randrange(3)
        leader = rng.randrange(3)
        s.progress[leader][7] = 12
        states.append(s)
        targets.append(seat_to_slot(s, leader))

    x = torch.from_numpy(encode_batch(states))
    y = torch.tensor(targets)
    mask = seat_mask_tensor(x)

    net = small_net(seed=7)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    first = masked_cross_entropy(net(x), y, mask).item()
    for _ in range(200):
        opt.zero_grad()
        loss = masked_cross_entropy(net(x), y, mask)
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5

    # And it learned the intended thing: the leader is the predicted winner.
    probs = net.win_probs(x)
    assert (probs.argmax(dim=1) == y).float().mean().item() > 0.9
