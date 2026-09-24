"""Tests for the arena, the replay buffer and the MVP training loop.

Real games cost ~1 s per turn, so anything here that plays games plays as few
as possible. The pure logic (seating rotation, Wilson interval, buffer
trimming) is tested directly and cheaply.

Run: python -m pytest games/cantstop/tests/test_train.py -q
"""

import json
import random
from collections import Counter

import numpy as np
import pytest
import torch

from games.cantstop.arena import (
    compare, play_match, player_of_seat, seating_for_game, verdict,
    wilson_interval, win_rate,
)
from games.cantstop.encoder import FEATURE_SIZE
from games.cantstop.engine import RuleSet
from games.cantstop.model import CantStopNet, NetEvaluator
from games.cantstop.solver import ProgressHeuristic
from games.cantstop.train import (
    MVP_RULES, ReplayBuffer, generate, run, train_steps,
)

CHEAP = MVP_RULES


def tiny_net(seed=0):
    torch.manual_seed(seed)
    return CantStopNet(hidden=(16, 16))


# ---- seating ----

def test_every_player_sits_in_every_seat_equally_often():
    """Can't Stop has a real first-player advantage; an unrotated match would
    measure that instead of the players."""
    for n in (2, 3, 4):
        seats = Counter()
        for game in range(n * 3):
            for player, seat in enumerate(seating_for_game(n, game)):
                seats[(player, seat)] += 1
        assert len(seats) == n * n
        assert set(seats.values()) == {3}


def test_seating_is_a_permutation():
    for n in (2, 3, 4):
        for game in range(5):
            assert sorted(seating_for_game(n, game)) == list(range(n))


def test_a_win_is_credited_to_the_player_not_the_seat():
    """With seats rotating, crediting the winning seat instead of the player
    who sat there inverts the whole match -- and looks fine in any one game."""
    for n in (2, 3, 4):
        for game in range(n * 2):
            seat_of = seating_for_game(n, game)
            for player in range(n):
                assert player_of_seat(seat_of, seat_of[player]) == player
    # The rotated case where seat and player genuinely differ.
    seat_of = seating_for_game(3, 1)          # player 0 sits at seat 1
    assert seat_of[0] == 1
    assert player_of_seat(seat_of, 1) == 0
    assert player_of_seat(seat_of, 0) == 2


def test_play_match_rejects_the_wrong_number_of_players():
    with pytest.raises(ValueError, match="need 2 players"):
        play_match(CHEAP, [ProgressHeuristic()], games=1)


# ---- reporting ----

def test_win_rate():
    assert win_rate([3, 1], 0) == 0.75
    assert win_rate([0, 0]) == 0.0


def test_wilson_interval_brackets_the_rate_and_narrows_with_data():
    lo_small, hi_small = wilson_interval(30, 60)
    lo_big, hi_big = wilson_interval(300, 600)
    assert lo_small < 0.5 < hi_small
    assert lo_big < 0.5 < hi_big
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_wilson_interval_is_honest_about_no_data():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_a_short_match_does_not_claim_significance():
    """60 games at 55% must not read as 'better' -- the MVP gate depends on
    this not overclaiming."""
    lo, _ = wilson_interval(33, 60)
    assert 33 / 60 > 0.5
    assert lo < 0.5


def test_better_means_the_interval_clears_even_not_just_the_rate():
    """The whole point of reporting an interval. A 55% result over 60 games
    is above even and still not evidence of anything."""
    noisy = verdict([33, 27], num_players=2)
    assert noisy["win_rate"] > noisy["even_match"]
    assert noisy["better"] is False

    solid = verdict([400, 200], num_players=2)
    assert solid["better"] is True

    # And the even-match bar moves with the seat count.
    assert verdict([40, 30, 30], num_players=3)["even_match"] == 1 / 3
    assert verdict([120, 90, 90], num_players=3)["better"] is True


# ---- replay buffer ----

def test_buffer_trims_by_rows_and_keeps_the_newest():
    buf = ReplayBuffer(max_rows=10)
    for chunk in range(4):
        x = np.full((4, FEATURE_SIZE), chunk, dtype=np.float32)
        buf.add(x, np.full(4, chunk, dtype=np.int64))
    x, y = buf.arrays()
    assert len(buf) == len(x) == len(y)
    assert len(buf) <= 12          # whole chunks only
    assert y.max() == 3            # newest chunk survived
    assert y.min() > 0             # oldest chunk was dropped


def test_buffer_ignores_empty_additions_and_checks_lengths():
    buf = ReplayBuffer(max_rows=10)
    buf.add(np.zeros((0, FEATURE_SIZE), np.float32), np.zeros(0, np.int64))
    assert len(buf) == 0
    x, y = buf.arrays()
    assert x.shape == (0, FEATURE_SIZE) and y.shape == (0,)
    with pytest.raises(ValueError, match="disagree"):
        buf.add(np.zeros((2, FEATURE_SIZE), np.float32),
                np.zeros(3, np.int64))


def test_buffer_never_empties_itself():
    """A single chunk larger than the cap must still be trainable on."""
    buf = ReplayBuffer(max_rows=2)
    buf.add(np.zeros((9, FEATURE_SIZE), np.float32), np.zeros(9, np.int64))
    assert len(buf) == 9


# ---- training step ----

def test_train_steps_reduces_loss_on_a_fixed_buffer():
    rng = random.Random(0)
    buf = ReplayBuffer(max_rows=1000)
    x = np.zeros((128, FEATURE_SIZE), dtype=np.float32)
    # Two live seats, and a feature that perfectly predicts the winner slot.
    from games.cantstop.encoder import seat_present_index
    x[:, seat_present_index(0)] = 1.0
    x[:, seat_present_index(1)] = 1.0
    y = np.array([i % 2 for i in range(128)], dtype=np.int64)
    x[:, 0] = y.astype(np.float32)
    buf.add(x, y)

    net = tiny_net(1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    device = torch.device("cpu")
    first = train_steps(net, buf, opt, 1, 64, device, rng)
    later = train_steps(net, buf, opt, 300, 64, device, rng)
    assert later < first


def test_train_steps_on_an_empty_buffer_is_not_an_error():
    net = tiny_net()
    opt = torch.optim.Adam(net.parameters())
    loss = train_steps(net, ReplayBuffer(10), opt, 5, 8,
                       torch.device("cpu"), random.Random(0))
    assert np.isnan(loss)


# ---- generation schedule ----

def test_generate_plays_the_scheduled_games():
    results = generate([CHEAP], 1, ProgressHeuristic(), random.Random(2))
    assert len(results) == 1
    assert results[0].rules == CHEAP
    assert len(results[0]) == results[0].turns - 1


# ---- end to end ----

def test_one_iteration_writes_a_checkpoint_and_a_log_line(tmp_path):
    net = run(out_dir=tmp_path, iterations=1, games=1, hidden=(16, 16),
              steps=5, batch_size=16, arena_games=0, seed=4, device="cpu")
    assert (tmp_path / "iter_0000.pt").exists()
    assert (tmp_path / "iter_0001.pt").exists()

    lines = (tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["iteration"] == 1
    assert record["rows"] > 0
    assert record["solves"] > 0
    assert not np.isnan(record["loss"])
    # The collapse watch the plan asks for is in every log line.
    assert record["mean_turn_length"] > 0


def test_compare_reports_an_interval_not_just_a_rate():
    result = compare(CHEAP, NetEvaluator(tiny_net(), device="cpu"),
                     ProgressHeuristic(), games=2, rng=random.Random(6))
    assert result["games"] == 2
    assert sum(result["wins"]) == 2
    assert result["even_match"] == 0.5
    lo, hi = result["ci95"]
    assert lo <= result["win_rate"] <= hi
    # Two games can never be significant.
    assert result["better"] is False
