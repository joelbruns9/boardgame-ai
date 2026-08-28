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
from games.welcome_to import training

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


class _FakeNvml:
    """Enough NVML to exercise device resolution without a driver."""

    NVMLError_NotFound = LookupError

    def __init__(self, uuids):
        self.uuids = uuids

    def nvmlDeviceGetCount(self):
        return len(self.uuids)

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetHandleByUUID(self, uuid):
        wanted = uuid.decode() if isinstance(uuid, bytes) else uuid
        for index, value in enumerate(self.uuids):
            if value.lower() == wanted.lower():
                return index
        raise LookupError("Not Found")

    def nvmlDeviceGetUUID(self, handle):
        return self.uuids[handle].encode()


def test_nvml_resolves_the_cuda_device_by_uuid_not_by_index(monkeypatch):
    """The sweep must never attribute another card's power draw to this run.

    NVML enumerates by PCI bus id while CUDA does not, and
    ``CUDA_VISIBLE_DEVICES`` remaps CUDA's indices without touching NVML's, so
    an index handle names a different physical GPU on a multi-GPU host. The
    previous spelling passed torch's bare UUID where NVML wants a ``GPU-``
    prefix, so the lookup always raised and the index fallback ran every time.
    """

    class _Properties:
        uuid = "aaaaaaaa-0000-0000-0000-000000000001"

    fake = _FakeNvml(
        [
            "GPU-bbbbbbbb-0000-0000-0000-000000000000",
            "GPU-aaaaaaaa-0000-0000-0000-000000000001",
        ]
    )
    monkeypatch.setattr(
        s2_throughput.torch.cuda, "get_device_properties", lambda index: _Properties()
    )
    # CUDA index 0, but the matching card is NVML index 1: an index handle would
    # have sampled the wrong GPU while still reporting nvml_available.
    assert s2_throughput._GpuSampler._resolve_handle(fake, 0) == 1


def test_nvml_refuses_to_guess_an_index_on_a_multi_gpu_host(monkeypatch):
    """No UUID and more than one card is a reportable error, not a guess."""

    class _Properties:
        uuid = ""

    fake = _FakeNvml(["GPU-a", "GPU-b"])
    monkeypatch.setattr(
        s2_throughput.torch.cuda, "get_device_properties", lambda index: _Properties()
    )
    with pytest.raises(RuntimeError, match="more than one"):
        s2_throughput._GpuSampler._resolve_handle(fake, 0)

    single = _FakeNvml(["GPU-a"])
    assert s2_throughput._GpuSampler._resolve_handle(single, 0) == 0


def test_s2_frozen_seat_streams_are_unique_across_a_contiguous_seed_block():
    """The regression the XOR seat derivation had, stated as the run that hits it.

    Seeds arrive as ``range(seed, seed + games)`` and seats are 1..3, so
    ``seed ^ DOMAIN ^ seat`` put the seat index in the same low bits as the
    seed: game ``s`` seat 1 shared a stream with game ``s ^ 3`` seat 2. Over a
    64-seed block that duplicated 128 of 192 frozen-opponent streams, giving
    correlated tie-breaks and temperature samples at the same decision index --
    worst in the opening, which is where divergence has to happen.
    """
    streams = [
        self_play.derive_seat_stream(seed, seat)
        for seed in range(1_000, 1_064)
        for seat in (1, 2, 3)
    ]
    assert len(set(streams)) == len(streams)
    assert self_play.derive_seat_stream(1_000, 1) != self_play.derive_seat_stream(
        1_003, 2
    )


def test_s2_playout_caps_are_stable_for_complete_turns_and_exact_for_full_games():
    config = self_play.SelfPlayConfig(
        games=100,
        seed=41_000,
        playout_cap_randomization=True,
        full_search_fraction=0.25,
        fast_search_simulations=64,
        full_search_game_fraction=0.05,
    )
    full_games = self_play.full_search_game_seeds(config)
    assert len(full_games) == 5
    assert full_games == self_play.full_search_game_seeds(config)
    assert all(
        self_play.full_search_for_turn(
            config, seed, turn, all_full_search=True
        )
        for seed in full_games
        for turn in range(1, 30)
    )

    ordinary = next(
        seed
        for seed in range(config.seed, config.seed + config.games)
        if seed not in full_games
    )
    first = [
        self_play.full_search_for_turn(config, ordinary, turn) for turn in range(40)
    ]
    second = [
        self_play.full_search_for_turn(config, ordinary, turn) for turn in range(40)
    ]
    assert first == second

    draws = [
        self_play.full_search_for_turn(config, seed, turn)
        for seed in range(config.seed, config.seed + config.games)
        if seed not in full_games
        for turn in range(1, 41)
    ]
    assert 0.22 < sum(draws) / len(draws) < 0.28


def test_s2_fast_turns_are_noiseless_and_not_recorded(monkeypatch, tmp_path):
    def unexpected_noise(*_args, **_kwargs):
        raise AssertionError("fast search requested root noise")

    monkeypatch.setattr(self_play.rust_search, "root_noise", unexpected_noise)
    prefix = tmp_path / "fast.jsonl"
    writer = self_play.wr.RustSampleShardWriter(
        prefix, shard_games=2, queue_games=2
    )
    try:
        trajectories, metrics = self_play.generate(
            _net(19),
            config=self_play.SelfPlayConfig(
                games=2,
                inflight=2,
                max_batch=2,
                seed=8_500,
                playout_cap_randomization=True,
                full_search_fraction=0.0,
                fast_search_simulations=1,
                full_search_game_fraction=0.0,
            ),
            search_config=self_play.default_search_config(simulations=2),
            device="cpu",
            on_captured=lambda _trajectory, captured: writer.add(captured),
        )
    finally:
        writer.close()
    assert all(not trajectory.searches for trajectory in trajectories)
    assert self_play.read_trajectories(prefix) == trajectories
    assert metrics["searched_roots"] == 0.0
    assert metrics["full_search_turns"] == 0.0
    assert metrics["fast_search_turns"] > 0.0
    assert metrics["full_search_roots"] == 0.0
    assert metrics["fast_search_roots"] > 0.0


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


def test_s2_rust_training_loader_batches_match_the_row_oracle(captured_generation):
    (_, _), prefix = captured_generation
    cached = self_play.read_trajectories(prefix)
    samples = [sample for trajectory in cached for sample in self_play.replay(trajectory)]
    expected = datagen.batch(samples)
    loader = self_play.rust_training_loader(
        cached, len(samples) + 1, shuffle_seed=None
    )
    assert loader is not None
    batches = list(self_play.iter_rust_training_batches(loader))
    assert len(batches) == 1
    actual = batches[0]
    assert actual.keys() == expected.keys()
    for name in expected:
        assert np.array_equal(actual[name], expected[name]), name

    loader.reset(12345)
    first = np.concatenate(
        [batch["action"] for batch in self_play.iter_rust_training_batches(loader)]
    )
    loader.reset(12345)
    second = np.concatenate(
        [batch["action"] for batch in self_play.iter_rust_training_batches(loader)]
    )
    assert np.array_equal(first, second)

    loader.reset_random(9876, 3)
    random_first = np.concatenate(
        [batch["action"] for batch in self_play.iter_rust_training_batches(loader)]
    )
    loader.reset_random(9876, 3)
    random_second = np.concatenate(
        [batch["action"] for batch in self_play.iter_rust_training_batches(loader)]
    )
    assert len(random_first) == 3 * (len(samples) + 1)
    assert np.array_equal(random_first, random_second)


def test_legacy_wts_targets_upgrade_without_inventing_plan_race_order():
    assert tuple(self_play.wr.LEGACY_TRAINING_PER_SEAT_TARGET_NAMES) == (
        training.LEGACY_PER_SEAT_TARGETS
    )
    global_count = len(training.GLOBAL_TARGETS)
    per_count = len(training.LEGACY_PER_SEAT_TARGETS)
    flat = np.zeros(global_count + training.MAX_SEATS * per_count, dtype=np.float32)
    seats = flat[global_count:].reshape(training.MAX_SEATS, per_count)
    old = {name: index for index, name in enumerate(training.LEGACY_PER_SEAT_TARGETS)}
    seats[0, old["seat_valid"]] = 1.0
    seats[0, old["turns_to_plan_1_mask"]] = 1.0
    seats[0, old["houses"]] = 1.0
    seats[0, old["plans_completed"]] = 1.0
    seats[0, old["permits"]] = 1.0
    targets = self_play._decode_wts_targets(
        flat, training.GLOBAL_TARGETS, training.LEGACY_PER_SEAT_TARGETS
    )
    assert targets["will_complete_plan_1"][0] == 1.0
    assert targets["plan_1_first"][0] == float(training.NEVER)
    assert targets["plan_1_first_mask"][0] == 0.0
    assert targets["end_trigger_full_sheet"][0] == 1.0
    assert targets["end_trigger_all_plans"][0] == 1.0
    assert targets["end_trigger_max_permit"][0] == 1.0


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
    league_manifest = self_play.run_manifest(
        config,
        search,
        checkpoint,
        [checkpoint],
        opponent_pool=[{"name": "best", "weight": 1.0}],
    )
    assert league_manifest["opponent_pool"] == [{"name": "best", "weight": 1.0}]
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


def test_s2_playout_cap_games_do_not_depend_on_scheduler_width():
    common = dict(
        games=4,
        seed=9_300,
        playout_cap_randomization=True,
        full_search_fraction=0.25,
        fast_search_simulations=1,
        full_search_game_fraction=0.25,
    )
    kwargs = dict(
        search_config=self_play.default_search_config(simulations=2),
        device="cpu",
    )
    serial, serial_metrics = self_play.generate(
        _net(24),
        config=self_play.SelfPlayConfig(inflight=1, max_batch=1, **common),
        **kwargs,
    )
    waved, waved_metrics = self_play.generate(
        _net(24),
        config=self_play.SelfPlayConfig(inflight=4, max_batch=4, **common),
        **kwargs,
    )
    assert {trajectory.seed: trajectory for trajectory in waved} == {
        trajectory.seed: trajectory for trajectory in serial
    }
    for name in (
        "full_search_games",
        "full_search_turns",
        "full_game_turns",
        "ordinary_full_search_turns",
        "fast_search_turns",
        "full_search_roots",
        "fast_search_roots",
    ):
        assert waved_metrics[name] == serial_metrics[name]
    assert serial_metrics["full_search_games"] == 1.0
    assert serial_metrics["full_search_roots"] > 0.0
    assert serial_metrics["fast_search_roots"] > 0.0


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


def test_resumed_metrics_count_positions_from_existing_and_new_shards(generated):
    complete, _ = generated
    existing = complete[:2]
    new = complete[2:]
    new_roots = sum(len(game.searches) for game in new)
    metrics = self_play.cumulative_generation_metrics(
        {"searched_roots": float(new_roots)},
        existing_games=len(existing),
        existing_searched_roots=sum(len(game.searches) for game in existing),
        new=new,
    )
    assert metrics["total_games"] == len(complete)
    assert metrics["searched_roots"] == sum(
        len(game.searches) for game in complete
    )
    assert metrics["new_searched_roots"] == new_roots


def test_s2_real_opponents_have_reproducible_independent_seat_streams():
    opponent = self_play.Opponent("frozen", _net(31))
    first = self_play._new_live(0, 1234, 4, [opponent])
    second = self_play._new_live(0, 1234, 4, [opponent])
    first_draws = [rng.next_u64() for rng in first.policy_rngs[1:]]
    second_draws = [rng.next_u64() for rng in second.policy_rngs[1:]]
    assert first_draws == second_draws
    assert len(set(first_draws)) == 3


def test_s2_weighted_opponent_pool_is_deterministic_and_respects_mass():
    current = self_play.Opponent("current", _net(32), weight=0.9)
    archive = self_play.Opponent("archive", _net(33), weight=0.1)
    names = []
    repeated = []
    for seed in range(500, 700):
        names.extend(self_play._new_live(0, seed, 4, [current, archive]).opponent_names[1:])
        repeated.extend(
            self_play._new_live(1, seed, 4, [current, archive]).opponent_names[1:]
        )
    assert names == repeated
    assert names.count("current") > 5 * names.count("archive")
