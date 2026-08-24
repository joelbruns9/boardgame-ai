"""S2 searched-trajectory, replay, and scheduler integration gates."""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from games.welcome_to import datagen
from games.welcome_to import network as nw
from games.welcome_to import self_play

pytest.importorskip("welcome_to_rust")


_SMALL = nw.NetConfig(
    sheet_hidden=16,
    sheet_out=8,
    trunk_hidden=24,
    trunk_blocks=1,
    head_hidden=16,
)


def _net(seed: int = 17) -> nw.WelcomeToNet:
    torch.manual_seed(seed)
    return nw.WelcomeToNet(_SMALL).eval()


@pytest.fixture(scope="module")
def generated():
    return self_play.generate(
        _net(),
        config=self_play.SelfPlayConfig(
            games=4,
            inflight=4,
            max_batch=4,
            seed=8_100,
            opening_temperature_turns=10,
        ),
        search_config=self_play.default_search_config(simulations=3),
        device="cpu",
    )


def test_s2_small_mix_keeps_every_encoder_seat_count_live():
    counts = self_play.seat_counts(4)
    assert len(counts) == 4
    assert {2, 3, 4}.issubset(counts)
    counts = self_play.seat_counts(10)
    assert (counts.count(2), counts.count(3), counts.count(4)) == (6, 3, 1)


def test_s2_generation_replays_every_root_and_emits_visit_targets(generated):
    trajectories, metrics = generated
    assert len(trajectories) == 4
    assert {trajectory.players for trajectory in trajectories} == {2, 3, 4}
    assert metrics["searched_roots"] == sum(len(t.searches) for t in trajectories)
    assert metrics["evaluator_calls"] < metrics["evaluator_rows"]
    assert metrics["mean_batch"] > 1.0

    for trajectory in trajectories:
        assert trajectory.opponents[0] == "learner"
        assert all(name == "incumbent" for name in trajectory.opponents[1:])
        samples = list(self_play.replay(trajectory))
        assert len(samples) == len(trajectory.searches) > 0
        by_decision = {target.decision: target for target in trajectory.searches}
        for sample, target in zip(samples, trajectory.searches):
            assert sample.actor == 0
            assert sample.policy is not None
            assert float(sample.policy.sum()) == pytest.approx(1.0)
            expected = np.zeros_like(sample.policy)
            expected[np.asarray(target.actions)] = np.asarray(target.visits)
            expected /= expected.sum()
            assert np.array_equal(sample.policy, expected)
            assert sample.action in target.actions
            assert target.decision in by_decision


def test_s2_json_round_trip_and_batch_are_training_ready(generated, tmp_path):
    trajectories, _ = generated
    path = self_play.write_trajectories(tmp_path / "s2.jsonl", trajectories)
    loaded = self_play.read_trajectories(path)
    assert loaded == trajectories
    samples = [sample for trajectory in loaded for sample in self_play.replay(trajectory)]
    raw = datagen.batch(samples[:8])
    assert raw["policy"].shape == (8, 684)
    assert np.allclose(raw["policy"].sum(axis=1), 1.0)
    assert np.all(raw["policy"][~raw["legal"]] == 0.0)
    streamed = next(
        self_play.iter_batches(loaded, 4, random.Random(0), shuffle_buffer=8)
    )
    assert streamed["policy"].shape == (4, 684)

    # With a visit target present, the sampled action is metadata rather than
    # the loss target. Changing it must not change soft policy cross-entropy.
    batch = nw.to_tensors(raw)
    net = _net(18)
    out = net(
        batch["sheet_planes"],
        batch["sheet_scalars"],
        batch["viewer_plane"],
        batch["global_scalars"],
    )
    _, before = nw.losses(out, batch)
    batch["action"] = torch.zeros_like(batch["action"])
    _, after = nw.losses(out, batch)
    assert torch.equal(before["policy"], after["policy"])


def test_s2_replay_rejects_a_root_whose_search_legality_changed(generated):
    trajectory = generated[0][0]
    first = trajectory.searches[0]
    bad = replace(first, actions=tuple(reversed(first.actions)))
    corrupted = replace(trajectory, searches=(bad, *trajectory.searches[1:]))
    with pytest.raises(ValueError, match="actions changed"):
        list(self_play.replay(corrupted))


def test_s2_discrete_games_do_not_depend_on_scheduler_width():
    kwargs = dict(
        search_config=self_play.default_search_config(simulations=2),
        device="cpu",
    )
    serial, _ = self_play.generate(
        _net(23),
        config=self_play.SelfPlayConfig(games=3, inflight=1, max_batch=1, seed=9_200),
        **kwargs,
    )
    waved, _ = self_play.generate(
        _net(23),
        config=self_play.SelfPlayConfig(games=3, inflight=3, max_batch=3, seed=9_200),
        **kwargs,
    )
    serial_by_seed = {trajectory.seed: trajectory for trajectory in serial}
    waved_by_seed = {trajectory.seed: trajectory for trajectory in waved}
    assert waved_by_seed == serial_by_seed


def test_s2_real_opponents_have_reproducible_independent_seat_streams():
    opponent = self_play.Opponent("frozen", _net(31))
    first = self_play._new_live(0, 1234, 4, [opponent])
    second = self_play._new_live(0, 1234, 4, [opponent])
    first_draws = [rng.next_u64() for rng in first.policy_rngs[1:]]
    second_draws = [rng.next_u64() for rng in second.policy_rngs[1:]]
    assert first_draws == second_draws
    assert len(set(first_draws)) == 3
