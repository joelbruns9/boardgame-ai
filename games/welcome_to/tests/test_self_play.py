"""S2 searched-trajectory, replay, and scheduler integration gates."""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from games.welcome_to import datagen
from games.welcome_to import network as nw
from games.welcome_to import s2_throughput
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
def captured_generation(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("rust-samples") / "trajectories.jsonl"
    writer = self_play.wr.RustSampleShardWriter(
        prefix, shard_games=2, queue_games=2
    )
    try:
        generated = self_play.generate(
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
            on_captured=lambda _trajectory, captured: writer.add(captured),
        )
    finally:
        writer.close()
    return generated, prefix


@pytest.fixture(scope="module")
def generated(captured_generation):
    return captured_generation[0]


def test_s2_small_mix_keeps_every_encoder_seat_count_live():
    counts = self_play.seat_counts(4)
    assert len(counts) == 4
    assert {2, 3, 4}.issubset(counts)
    counts = self_play.seat_counts(10)
    assert (counts.count(2), counts.count(3), counts.count(4)) == (6, 3, 1)
    with pytest.raises(ValueError, match="CUDA-only"):
        s2_throughput.sweep("unused.pt", inflight=(1,), games=1, device="cpu")


def test_s2_learner_and_opponents_share_one_temperature_schedule():
    config = self_play.SelfPlayConfig(
        games=4,
        opening_temperature_turns=10,
        opening_temperature=1.0,
        late_temperature=0.0,
    )
    assert self_play.temperature_for_turn(config, 10) == 1.0
    assert self_play.temperature_for_turn(config, 11) == 0.0

    policy = np.zeros(684, dtype=np.float32)
    policy[[10, 20, 30]] = [0.2, 0.7, 0.1]
    assert self_play.sample_policy(
        (10, 20, 30), policy, self_play.PortableRng(1), 0.0
    ) == 20
    first = self_play.sample_policy(
        (10, 20, 30), policy, self_play.PortableRng(2), 1.0
    )
    second = self_play.sample_policy(
        (10, 20, 30), policy, self_play.PortableRng(2), 1.0
    )
    assert first == second


def test_s2_generation_replays_every_root_and_emits_visit_targets(generated):
    trajectories, metrics = generated
    assert len(trajectories) == 4
    assert {trajectory.players for trajectory in trajectories} == {2, 3, 4}
    assert metrics["searched_roots"] == sum(len(t.searches) for t in trajectories)
    assert metrics["evaluator_calls"] < metrics["evaluator_rows"]
    assert metrics["mean_batch"] > 1.0
    scheduler_profile = metrics["scheduler_profile"]
    assert scheduler_profile["workers"] == 4
    assert scheduler_profile["requests"] > 0
    assert scheduler_profile["search_ms"] > 0
    assert scheduler_profile["encode_ms"] > 0
    assert metrics["evaluator_profiles"][0]["postprocess_sync_ms"] > 0

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


def test_s2_rust_sample_shards_are_exactly_the_python_oracle(captured_generation):
    (trajectories, _), prefix = captured_generation
    cached = self_play.read_trajectories(prefix)
    assert cached == trajectories
    assert len(self_play.training_shard_paths(prefix)) == 2
    assert not list(prefix.parent.glob("*.tmp"))

    for trajectory, cached_trajectory in zip(trajectories, cached):
        # Round-tripping the JSON deliberately drops the Rust row source and
        # forces the independent Python replay oracle.
        oracle = self_play.SelfPlayTrajectory.from_json(trajectory.to_json())
        oracle_samples = list(self_play.replay(oracle))
        cached_samples = list(self_play.replay(cached_trajectory))
        assert len(cached_samples) == len(oracle_samples) == len(trajectory.searches)
        for actual, expected in zip(cached_samples, oracle_samples):
            assert np.array_equal(actual.sheet_planes, expected.sheet_planes)
            assert np.array_equal(actual.sheet_scalars, expected.sheet_scalars)
            assert np.array_equal(actual.viewer_plane, expected.viewer_plane)
            assert np.array_equal(actual.global_scalars, expected.global_scalars)
            assert np.array_equal(actual.legal, expected.legal)
            assert np.array_equal(actual.policy, expected.policy)
            assert (actual.action, actual.actor, actual.turn) == (
                expected.action,
                expected.actor,
                expected.turn,
            )
            assert actual.targets.keys() == expected.targets.keys()
            for name in actual.targets:
                assert np.array_equal(
                    np.asarray(actual.targets[name], dtype=np.float32),
                    np.asarray(expected.targets[name], dtype=np.float32),
                ), name


def test_s2_rust_sample_writer_restarts_and_rejects_truncation(
    captured_generation, tmp_path
):
    (trajectories, _), prefix = captured_generation
    writer = self_play.wr.RustSampleShardWriter(prefix, shard_games=2, queue_games=1)
    assert writer.completed_seeds == sorted(t.seed for t in trajectories)
    writer.close()

    shard = self_play.training_shard_paths(prefix)[0]
    truncated_prefix = tmp_path / "broken.jsonl"
    truncated = tmp_path / "broken.part-000000.wts"
    truncated.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises((RuntimeError, ValueError), match="truncated"):
        self_play.wr.RustSampleShardWriter(truncated_prefix)


def test_s2_atomic_shards_resume_by_seed(generated, tmp_path):
    trajectories, _ = generated
    prefix = tmp_path / "trajectories.jsonl"
    writer = self_play.TrajectoryShardWriter(prefix, shard_games=2)
    for trajectory in trajectories[:3]:
        writer.add(trajectory)
    writer.close()
    shards = self_play.trajectory_sources(prefix)
    assert len(shards) == 2
    assert self_play.read_trajectories(prefix) == trajectories[:3]
    assert not list(tmp_path.glob("*.tmp"))

    resumed = self_play.TrajectoryShardWriter(prefix, shard_games=2)
    assert resumed.completed_seeds == frozenset(t.seed for t in trajectories[:3])
    with pytest.raises(ValueError, match="already written"):
        resumed.add(trajectories[0])
    resumed.add(trajectories[3])
    resumed.close()
    assert self_play.read_trajectories(prefix) == trajectories

    config = self_play.SelfPlayConfig(games=4, seed=8_100)
    assert self_play.validate_resume(trajectories, config) == frozenset(
        trajectory.seed for trajectory in trajectories
    )
    original = trajectories[0]
    changed_players = 3 if original.players == 2 else 2
    wrong_players = replace(
        original,
        players=changed_players,
        scores=(original.scores + (0,))[:changed_players],
        opponents=(original.opponents + ("extra",))[:changed_players],
    )
    with pytest.raises(ValueError, match="players"):
        self_play.validate_resume((wrong_players, *trajectories[1:]), config)


def test_s2_generation_manifest_refuses_mixed_runs(tmp_path):
    checkpoint = tmp_path / "learner.pt"
    checkpoint.write_bytes(b"frozen learner")
    config = self_play.SelfPlayConfig(games=4, seed=12)
    search = self_play.default_search_config(simulations=3)
    manifest = self_play.run_manifest(config, search, checkpoint, [checkpoint])
    prefix = tmp_path / "corpus.jsonl"
    path = self_play.ensure_run_manifest(
        prefix, manifest, has_existing_games=False
    )
    assert path.exists()
    assert self_play.ensure_run_manifest(
        prefix, manifest, has_existing_games=True
    ) == path
    changed = dict(manifest)
    changed["table_signature"] = int(manifest["table_signature"]) + 1
    with pytest.raises(ValueError, match="does not match"):
        self_play.ensure_run_manifest(prefix, changed, has_existing_games=True)

    with pytest.raises(ValueError, match="unverifiable resume"):
        self_play.ensure_run_manifest(
            tmp_path / "legacy.jsonl", manifest, has_existing_games=True
        )


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


def test_s2_resumed_generation_only_replays_missing_seed_jobs(generated):
    complete, _ = generated
    completed = frozenset(trajectory.seed for trajectory in complete[:2])
    admitted = []
    resumed, metrics = self_play.generate(
        _net(),
        config=self_play.SelfPlayConfig(
            games=4,
            inflight=4,
            max_batch=4,
            seed=8_100,
        ),
        search_config=self_play.default_search_config(simulations=3),
        device="cpu",
        skip_seeds=completed,
        on_trajectory=admitted.append,
    )
    assert admitted == resumed
    assert len(resumed) == 2
    combined = {trajectory.seed: trajectory for trajectory in (*complete[:2], *resumed)}
    assert combined == {trajectory.seed: trajectory for trajectory in complete}
    assert metrics["games"] == 2.0


def test_s2_real_opponents_have_reproducible_independent_seat_streams():
    opponent = self_play.Opponent("frozen", _net(31))
    first = self_play._new_live(0, 1234, 4, [opponent])
    second = self_play._new_live(0, 1234, 4, [opponent])
    first_draws = [rng.next_u64() for rng in first.policy_rngs[1:]]
    second_draws = [rng.next_u64() for rng in second.policy_rngs[1:]]
    assert first_draws == second_draws
    assert len(set(first_draws)) == 3
