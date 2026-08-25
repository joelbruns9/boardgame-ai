"""S2 learner-only self-play trajectories on the M6 Rust scheduler.

One searched learner occupies seat 0. Every other real-game seat samples a
frozen checkpoint policy from an independently seeded portable stream. Search
simulations remain the learner's information-set MCTS; only learner roots emit
policy targets, and those targets are visit distributions rather than the action
eventually sampled from them.

The stored format is deliberately different from S0's primitive teacher
trajectory. S2 records complete *macro* action games plus sparse root visit rows.
Production roots are encoded directly from the live Rust state, terminal targets
are finalized in Rust, and a bounded background writer commits atomic binary
shards. Each shard retains the raw portable trajectory so Python replay remains
an independent offline oracle and targets can be re-derived after a schema
change; it is no longer a serial admission stage in the self-play hot path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Collection, Iterator, Optional, Sequence

import numpy as np
import torch

from games.welcome_to import datagen
from games.welcome_to import encoder as enc
from games.welcome_to import macro_codec as mc
from games.welcome_to import mcts
from games.welcome_to import network as nw
from games.welcome_to import rust_search
from games.welcome_to import snapshot
from games.welcome_to import training
from games.welcome_to.game import GameConfig, GameState
from games.welcome_to.portable_rng import PortableRng, derive_search_seed

try:
    import welcome_to_rust as wr
except ImportError:  # pragma: no cover - source checkouts need no Rust toolchain
    wr = None


FORMAT_VERSION = 1
LEARNER_SEAT = 0
SEAT_MIX: tuple[tuple[int, float], ...] = ((2, 0.60), (3, 0.30), (4, 0.10))
_JOB_ORDER_DOMAIN = 0x5332_4A4F_424F_5244  # "S2JOBORD"
_POOL_DOMAIN = 0x5332_504F_4F4C_4153  # "S2POOLAS"
_POLICY_DOMAIN = 0x5332_504F_4C49_4359  # "S2POLICY"
_MASK64 = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class SearchTarget:
    """One learner root, sparse in the 684-action vocabulary."""

    decision: int
    actions: tuple[int, ...]
    visits: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.decision < 0:
            raise ValueError("search target decision must be non-negative")
        if not self.actions or len(self.actions) != len(self.visits):
            raise ValueError("search target actions/visits are empty or misaligned")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("search target repeats a macro action")
        if any(not 0 <= action < mc.NUM_MACRO_ACTIONS for action in self.actions):
            raise ValueError("search target contains an out-of-range macro")
        if any(visit < 0 for visit in self.visits) or sum(self.visits) <= 0:
            raise ValueError("search target visits need positive non-negative mass")


@dataclass(frozen=True, slots=True)
class SelfPlayTrajectory:
    """One complete S2 game and the learner's searched roots."""

    seed: int
    players: int
    actions: tuple[int, ...]
    searches: tuple[SearchTarget, ...]
    scores: tuple[int, ...]
    opponents: tuple[str, ...]
    prune_roundabout_pass: bool = True
    advanced: bool = True
    learner: int = LEARNER_SEAT
    rng: str = "portable"
    version: int = FORMAT_VERSION
    _sample_source: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.version != FORMAT_VERSION:
            raise ValueError(
                f"self-play trajectory version {self.version} != {FORMAT_VERSION}"
            )
        if self.players not in (2, 3, 4):
            raise ValueError("S2 supports exactly 2-4 seats")
        if not self.advanced or self.learner != LEARNER_SEAT or self.rng != "portable":
            raise ValueError("S2 requires advanced, seat-0 learner, portable games")
        if len(self.scores) != self.players or len(self.opponents) != self.players:
            raise ValueError("scores/opponents must have one entry per seat")
        decisions = [target.decision for target in self.searches]
        if decisions != sorted(set(decisions)):
            raise ValueError("search targets must be unique and decision-ordered")
        if decisions and decisions[-1] >= len(self.actions):
            raise ValueError("search target points beyond the macro trajectory")

    @property
    def config(self) -> GameConfig:
        return GameConfig(players=self.players, advanced=True, solo_rules=False)

    def to_json(self) -> str:
        raw = {
            "seed": self.seed,
            "players": self.players,
            "actions": self.actions,
            "searches": [asdict(target) for target in self.searches],
            "scores": self.scores,
            "opponents": self.opponents,
            "prune_roundabout_pass": self.prune_roundabout_pass,
            "advanced": self.advanced,
            "learner": self.learner,
            "rng": self.rng,
            "version": self.version,
        }
        return json.dumps(raw, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "SelfPlayTrajectory":
        raw = json.loads(line)
        return cls(
            seed=int(raw["seed"]),
            players=int(raw["players"]),
            actions=tuple(int(action) for action in raw["actions"]),
            searches=tuple(
                SearchTarget(
                    decision=int(target["decision"]),
                    actions=tuple(int(action) for action in target["actions"]),
                    visits=tuple(int(visit) for visit in target["visits"]),
                )
                for target in raw["searches"]
            ),
            scores=tuple(int(score) for score in raw["scores"]),
            opponents=tuple(str(name) for name in raw["opponents"]),
            prune_roundabout_pass=bool(raw.get("prune_roundabout_pass", True)),
            advanced=bool(raw.get("advanced", True)),
            learner=int(raw.get("learner", LEARNER_SEAT)),
            rng=str(raw.get("rng", "portable")),
            version=int(raw.get("version", -1)),
        )


@dataclass(frozen=True, slots=True)
class Opponent:
    """A frozen real-game opponent policy and its sampling weight."""

    name: str
    net: nw.WelcomeToNet
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("opponent name cannot be empty")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("opponent weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    games: int = 64
    inflight: int = 256
    max_batch: int = 256
    scheduler_workers: int = 8
    seed: int = 0
    opening_temperature_turns: int = 10
    opening_temperature: float = 1.0
    late_temperature: float = 0.0
    max_decisions: int = 2_000

    def __post_init__(self) -> None:
        if (
            self.games <= 0
            or self.inflight <= 0
            or self.max_batch <= 0
            or self.scheduler_workers <= 0
        ):
            raise ValueError(
                "games, inflight, max_batch, and scheduler_workers must be positive"
            )
        if self.opening_temperature_turns < 0 or self.max_decisions <= 0:
            raise ValueError("temperature turns must be non-negative and max_decisions positive")
        for name, value in (
            ("opening_temperature", self.opening_temperature),
            ("late_temperature", self.late_temperature),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(slots=True)
class _LiveGame:
    slot: int
    seed: int
    players: int
    state: object
    opponent_indices: tuple[int, ...]
    opponent_names: tuple[str, ...]
    policy_rngs: tuple[Optional[PortableRng], ...]
    actions: list[int]
    searches: list[SearchTarget]
    capture: object
    learner_decisions: int = 0


def default_search_config(simulations: int = 200) -> mcts.SearchConfig:
    """S2's starting search geometry, with root exploration enabled."""
    return mcts.SearchConfig(
        simulations=simulations,
        chance_widening=1.0,
        chance_widening_alpha=0.5,
        max_particles=4,
        dirichlet_concentration=10.0,
        dirichlet_weight=0.25,
        noise_fresh_fraction=0.25,
        temperature=0.0,  # overridden per root by SelfPlayConfig
    )


def temperature_for_turn(config: SelfPlayConfig, turn: int) -> float:
    """The one opening/late schedule used by learner and real opponents."""
    return (
        config.opening_temperature
        if turn <= config.opening_temperature_turns
        else config.late_temperature
    )


def sample_policy(
    legal: Sequence[int],
    policy: np.ndarray,
    rng: PortableRng,
    temperature: float,
) -> int:
    """Sample a legal network policy at ``temperature`` without underflow.

    Zero temperature chooses uniformly among exact maxima.  The tie draw is
    deliberate: frozen seats retain independent streams even when an early
    checkpoint emits identical logits for several moves.
    """
    if not legal:
        raise ValueError("cannot sample an empty legal policy")
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("policy temperature must be finite and non-negative")
    priors = [float(policy[action]) for action in legal]
    if any(not math.isfinite(prior) or prior < 0.0 for prior in priors):
        raise ValueError("policy priors must be finite and non-negative")
    maximum = max(priors)
    if maximum <= 0.0:
        raise ValueError("legal policy has no positive mass")
    if temperature <= 0.0:
        best = [action for action, prior in zip(legal, priors) if prior == maximum]
        return int(rng.choice(best))
    inverse = 1.0 / temperature
    weights = [(prior / maximum) ** inverse for prior in priors]
    return int(rng.choices(legal, weights=weights, k=1)[0])


def seat_counts(games: int) -> list[int]:
    """Largest-remainder 60/30/10 mix, then deterministically shuffled."""
    exact = [(players, games * share) for players, share in SEAT_MIX]
    counts = {players: int(math.floor(value)) for players, value in exact}
    remainder = games - sum(counts.values())
    for players, value in sorted(exact, key=lambda item: -(item[1] % 1.0))[:remainder]:
        counts[players] += 1
    if games >= len(SEAT_MIX):
        for players, _ in SEAT_MIX:
            if counts[players] != 0:
                continue
            donor = max(
                (other for other, _ in SEAT_MIX if counts[other] > 1),
                key=counts.get,
            )
            counts[donor] -= 1
            counts[players] += 1
    return [players for players, _ in SEAT_MIX for _ in range(counts[players])]


def replay(trajectory: SelfPlayTrajectory) -> Iterator[datagen.Sample]:
    """Read captured rows, or rebuild a legacy trajectory with the Python oracle."""
    if trajectory._sample_source is not None:
        yield from trajectory._sample_source.samples()
        return
    state = GameState.new(
        seed=trajectory.seed,
        config=trajectory.config,
        rng_kind=trajectory.rng,
    )
    targets = {target.decision: target for target in trajectory.searches}
    visits = []
    for decision, action in enumerate(trajectory.actions):
        target = targets.get(decision)
        if target is not None:
            if state.actor != trajectory.learner:
                raise ValueError(
                    f"search target {decision} belongs to actor {state.actor}, "
                    f"not learner {trajectory.learner}"
                )
            expected = tuple(
                mc.search_legal_macros(state, trajectory.prune_roundabout_pass)
            )
            if target.actions != expected:
                raise ValueError(
                    f"search root {decision} actions changed: "
                    f"recorded {target.actions[:8]}, replay {expected[:8]}"
                )
            policy = np.zeros(mc.NUM_MACRO_ACTIONS, dtype=np.float32)
            policy[np.asarray(target.actions, dtype=np.intp)] = np.asarray(
                target.visits, dtype=np.float32
            )
            policy /= policy.sum()
            visits.append(
                (
                    enc.encode_state(state, trajectory.learner),
                    mc.legal_mask(state),
                    action,
                    state.actor,
                    state.turn,
                    enc.seat_order(state, trajectory.learner),
                    policy,
                )
            )
        if action not in mc.legal_macros(state):
            raise ValueError(f"recorded macro {action} is illegal at decision {decision}")
        mc.apply_macro(state, action)

    if not state.is_terminal:
        raise ValueError("self-play trajectory does not reach a terminal state")
    if tuple(state.scores()) != trajectory.scores:
        raise ValueError(
            f"self-play replay scores {state.scores()} != recorded {trajectory.scores}"
        )
    if len(visits) != len(trajectory.searches):
        raise ValueError("one or more search targets were not replayed")

    outcomes = training.final_outcomes(state)
    for encoded, legal, action, actor, turn, order, policy in visits:
        sheet_planes, sheet_scalars, viewer_plane, global_scalars = encoded
        yield datagen.Sample(
            sheet_planes=sheet_planes,
            sheet_scalars=sheet_scalars,
            viewer_plane=viewer_plane,
            global_scalars=global_scalars,
            legal=legal,
            action=action,
            actor=actor,
            turn=turn,
            targets=training.sample_targets(outcomes, order, turn),
            policy=policy,
        )


def write_trajectories(
    path: str | Path, trajectories: Sequence[SelfPlayTrajectory]
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(trajectory.to_json() + "\n")
    return path


def _shard_paths(path: Path) -> list[Path]:
    suffix = path.suffix or ".jsonl"
    stem = path.stem if path.suffix else path.name
    return sorted(path.parent.glob(f"{stem}.part-*{suffix}"))


def training_shard_paths(path: str | Path) -> list[Path]:
    """Atomic Rust sample shards associated with a trajectory prefix."""
    path = Path(path)
    if path.is_dir():
        return sorted(path.glob("*.wts"))
    stem = path.stem if path.suffix else path.name
    return sorted(path.parent.glob(f"{stem}.part-*.wts"))


_WTS_HEADER = struct.Struct("<8sHHHHQI")
_WTS_RECORD = struct.Struct("<Q")
_WTS_FIXED_ENCODER_FLOATS = (
    int(np.prod(enc.SHEET_PLANES_SHAPE))
    + enc.MAX_SEATS * enc.NUM_SHEET_SCALAR
    + int(np.prod(enc.VIEWER_PLANE_SHAPE))
    + enc.NUM_GLOBAL_SCALAR
)
_WTS_SAMPLE_PREFIX = (
    _WTS_FIXED_ENCODER_FLOATS * 4
    + mc.NUM_MACRO_ACTIONS
    + 2  # selected action
    + 1  # actor
    + 4  # turn
    + 2  # sparse policy support
)


@dataclass(frozen=True, slots=True)
class _CachedGameSamples:
    path: Path
    offsets: tuple[int, ...]

    @staticmethod
    def _f32(handle, shape: tuple[int, ...]) -> np.ndarray:
        count = int(np.prod(shape))
        raw = handle.read(count * 4)
        if len(raw) != count * 4:
            raise ValueError("training shard ended inside an encoder array")
        return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()

    def samples(self) -> Iterator[datagen.Sample]:
        with self.path.open("rb") as handle:
            for offset in self.offsets:
                handle.seek(offset)
                sheet_planes = self._f32(handle, enc.SHEET_PLANES_SHAPE)
                sheet_scalars = self._f32(
                    handle, (enc.MAX_SEATS, enc.NUM_SHEET_SCALAR)
                )
                viewer_plane = self._f32(handle, enc.VIEWER_PLANE_SHAPE)
                global_scalars = self._f32(handle, (enc.NUM_GLOBAL_SCALAR,))
                legal_raw = handle.read(mc.NUM_MACRO_ACTIONS)
                if len(legal_raw) != mc.NUM_MACRO_ACTIONS:
                    raise ValueError("training shard ended inside a legal mask")
                legal = np.frombuffer(legal_raw, dtype=np.uint8).astype(np.bool_)
                fixed = handle.read(9)
                if len(fixed) != 9:
                    raise ValueError("training shard ended inside sample metadata")
                action, actor, turn, support = struct.unpack("<HBiH", fixed)
                sparse = handle.read(support * 6)
                if len(sparse) != support * 6:
                    raise ValueError("training shard ended inside a sparse policy")
                policy = np.zeros(mc.NUM_MACRO_ACTIONS, dtype=np.float32)
                for index in range(support):
                    macro_action, visits = struct.unpack_from("<HI", sparse, index * 6)
                    policy[macro_action] = np.float32(visits)
                mass = policy.sum(dtype=np.float32)
                if not mass > 0:
                    raise ValueError("training shard policy has no visit mass")
                policy /= mass
                target_count = len(training.GLOBAL_TARGETS) + (
                    training.MAX_SEATS * len(training.PER_SEAT_TARGETS)
                )
                raw_targets = handle.read(target_count * 4)
                if len(raw_targets) != target_count * 4:
                    raise ValueError("training shard ended inside target values")
                flat = np.frombuffer(raw_targets, dtype="<f4")
                targets: dict[str, float | tuple[float, ...]] = {
                    name: float(flat[index])
                    for index, name in enumerate(training.GLOBAL_TARGETS)
                }
                seats = flat[len(training.GLOBAL_TARGETS) :].reshape(
                    training.MAX_SEATS, len(training.PER_SEAT_TARGETS)
                )
                for index, name in enumerate(training.PER_SEAT_TARGETS):
                    targets[name] = tuple(float(value) for value in seats[:, index])
                yield datagen.Sample(
                    sheet_planes=sheet_planes,
                    sheet_scalars=sheet_scalars,
                    viewer_plane=viewer_plane,
                    global_scalars=global_scalars,
                    legal=legal,
                    action=int(action),
                    actor=int(actor),
                    turn=int(turn),
                    targets=targets,
                    policy=policy,
                )


def _read_training_shard(path: Path) -> list[SelfPlayTrajectory]:
    if wr is None:
        raise RuntimeError("welcome_to_rust is required to validate training shards")
    out: list[SelfPlayTrajectory] = []
    with path.open("rb") as handle:
        file_size = path.stat().st_size
        raw = handle.read(_WTS_HEADER.size)
        if len(raw) != _WTS_HEADER.size:
            raise ValueError(f"training shard {path} has a truncated header")
        magic, version, globals_, per_seat, max_seats, signature, games = (
            _WTS_HEADER.unpack(raw)
        )
        if (
            magic != b"WTSHRD01"
            or version != int(wr.TRAINING_SHARD_VERSION)
            or globals_ != len(training.GLOBAL_TARGETS)
            or per_seat != len(training.PER_SEAT_TARGETS)
            or max_seats != training.MAX_SEATS
            or signature != int(wr.table_signature())
            or tuple(wr.TRAINING_GLOBAL_TARGET_NAMES) != training.GLOBAL_TARGETS
            or tuple(wr.TRAINING_PER_SEAT_TARGET_NAMES)
            != training.PER_SEAT_TARGETS
        ):
            raise ValueError(f"training shard {path} has an incompatible ABI")
        target_bytes = (
            len(training.GLOBAL_TARGETS)
            + training.MAX_SEATS * len(training.PER_SEAT_TARGETS)
        ) * 4
        for _ in range(games):
            raw_length = handle.read(_WTS_RECORD.size)
            if len(raw_length) != _WTS_RECORD.size:
                raise ValueError(f"training shard {path} ended before a game record")
            (record_length,) = _WTS_RECORD.unpack(raw_length)
            record_start = handle.tell()
            record_end = record_start + record_length
            if record_length < 16 or record_end > file_size:
                raise ValueError(f"training shard {path} has a truncated game record")
            fixed = handle.read(12)
            if len(fixed) != 12:
                raise ValueError(f"training shard {path} has a truncated game record")
            seed, json_length = struct.unpack("<QI", fixed)
            raw_json = handle.read(json_length)
            if len(raw_json) != json_length:
                raise ValueError(f"training shard {path} has truncated trajectory JSON")
            trajectory = SelfPlayTrajectory.from_json(raw_json.decode("utf-8"))
            if trajectory.seed != seed:
                raise ValueError(
                    f"training shard {path} seed {seed} disagrees with its trajectory"
                )
            raw_count = handle.read(4)
            if len(raw_count) != 4:
                raise ValueError(f"training shard {path} omitted its sample count")
            (sample_count,) = struct.unpack("<I", raw_count)
            offsets = []
            for _sample in range(sample_count):
                offsets.append(handle.tell())
                handle.seek(_WTS_SAMPLE_PREFIX - 2, os.SEEK_CUR)
                if handle.tell() + 2 > record_end:
                    raise ValueError(f"training shard {path} ended inside a sample")
                raw_support = handle.read(2)
                if len(raw_support) != 2:
                    raise ValueError(f"training shard {path} ended inside a sample")
                (support,) = struct.unpack("<H", raw_support)
                handle.seek(support * 6 + target_bytes, os.SEEK_CUR)
                if handle.tell() > record_end:
                    raise ValueError(f"training shard {path} ended inside a sample")
            if handle.tell() != record_end:
                raise ValueError(f"training shard {path} game record length is inconsistent")
            if sample_count != len(trajectory.searches):
                raise ValueError(
                    f"training shard {path} has {sample_count} rows for "
                    f"{len(trajectory.searches)} search targets"
                )
            out.append(
                replace(
                    trajectory,
                    _sample_source=_CachedGameSamples(path, tuple(offsets)),
                )
            )
        if handle.read(1):
            raise ValueError(f"training shard {path} has trailing bytes")
    return out


def trajectory_sources(path: str | Path) -> list[Path]:
    """Resolve a legacy monolith, a shard prefix, or a shard directory."""
    path = Path(path)
    if path.is_dir():
        sources = sorted(path.glob("*.jsonl"))
    else:
        sources = ([path] if path.is_file() else []) + _shard_paths(path)
    return sources


def read_trajectories(path: str | Path) -> list[SelfPlayTrajectory]:
    sources = trajectory_sources(path)
    sample_sources = training_shard_paths(path)
    if not sources and not sample_sources:
        raise FileNotFoundError(f"no trajectory file or shards found for {path}")
    trajectories: list[SelfPlayTrajectory] = []
    for source in sources:
        with source.open(encoding="utf-8") as handle:
            trajectories.extend(
                SelfPlayTrajectory.from_json(line) for line in handle if line.strip()
            )
    for source in sample_sources:
        trajectories.extend(_read_training_shard(source))
    seeds = [trajectory.seed for trajectory in trajectories]
    if len(seeds) != len(set(seeds)):
        raise ValueError("trajectory corpus contains duplicate game seeds")
    return trajectories


class TrajectoryShardWriter:
    """Atomically admit replay-validated games in bounded-loss shards.

    A hard process failure can lose only the current in-memory shard. Completed
    shard files are immutable and discovered on restart; their seeds are then
    removed from the deterministic job schedule.
    """

    def __init__(self, path: str | Path, shard_games: int = 25) -> None:
        if shard_games <= 0:
            raise ValueError("shard_games must be positive")
        self.path = Path(path)
        if self.path.is_dir():
            self.path = self.path / "trajectories.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.shard_games = shard_games
        self.buffer: list[SelfPlayTrajectory] = []
        sources = trajectory_sources(self.path)
        self.existing = read_trajectories(self.path) if sources else []
        seeds = [trajectory.seed for trajectory in self.existing]
        if len(seeds) != len(set(seeds)):
            raise ValueError("trajectory shards contain duplicate game seeds")
        self.completed_seeds = frozenset(seeds)
        self.next_shard = 0
        while self._shard_path(self.next_shard).exists():
            self.next_shard += 1

    def _shard_path(self, index: int) -> Path:
        suffix = self.path.suffix or ".jsonl"
        stem = self.path.stem if self.path.suffix else self.path.name
        return self.path.with_name(f"{stem}.part-{index:06d}{suffix}")

    def add(self, trajectory: SelfPlayTrajectory) -> None:
        if trajectory.seed in self.completed_seeds or any(
            item.seed == trajectory.seed for item in self.buffer
        ):
            raise ValueError(f"trajectory seed {trajectory.seed} was already written")
        self.buffer.append(trajectory)
        if len(self.buffer) >= self.shard_games:
            self.flush()

    def flush(self) -> Optional[Path]:
        if not self.buffer:
            return None
        destination = self._shard_path(self.next_shard)
        temporary: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=destination.name + ".",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                for trajectory in self.buffer:
                    handle.write(trajectory.to_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            self.completed_seeds = self.completed_seeds | frozenset(
                item.seed for item in self.buffer
            )
            self.buffer.clear()
            self.next_shard += 1
            return destination
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def close(self) -> None:
        self.flush()


def validate_resume(
    trajectories: Sequence[SelfPlayTrajectory], config: SelfPlayConfig
) -> frozenset[int]:
    """Prove persisted games belong to this exact seed/seat-count schedule."""
    planned = dict(
        zip(
            range(config.seed, config.seed + config.games),
            seat_counts(config.games),
        )
    )
    seen: set[int] = set()
    for trajectory in trajectories:
        if trajectory.seed in seen:
            raise ValueError(f"resume corpus repeats seed {trajectory.seed}")
        seen.add(trajectory.seed)
        expected_players = planned.get(trajectory.seed)
        if expected_players is None:
            raise ValueError(
                f"resume seed {trajectory.seed} is outside "
                f"[{config.seed}, {config.seed + config.games})"
            )
        if trajectory.players != expected_players:
            raise ValueError(
                f"resume seed {trajectory.seed} has {trajectory.players} players, "
                f"expected {expected_players}"
            )
    return frozenset(seen)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_manifest(
    config: SelfPlayConfig,
    search_config: mcts.SearchConfig,
    checkpoint: str | Path,
    opponent_checkpoints: Sequence[str | Path],
) -> dict:
    """Frozen semantic and scheduler identity for safe shard resume."""
    return {
        "format": "welcome_to_s2_generation",
        "version": 1,
        "trajectory_version": FORMAT_VERSION,
        "table_signature": int(wr.table_signature()) if wr is not None else None,
        "self_play_config": asdict(config),
        "search_config": asdict(search_config),
        "learner": {
            "path": str(Path(checkpoint).resolve()),
            "sha256": _file_sha256(checkpoint),
        },
        "opponents": [
            {
                "path": str(Path(path).resolve()),
                "sha256": _file_sha256(path),
            }
            for path in opponent_checkpoints
        ],
    }


def ensure_run_manifest(
    trajectory_path: str | Path,
    manifest: dict,
    *,
    has_existing_games: bool,
) -> Path:
    """Create or exactly validate the sidecar governing resumable shards."""
    path = Path(trajectory_path)
    if path.is_dir():
        path = path / "trajectories.jsonl"
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if recorded != manifest:
            raise ValueError(
                f"generation manifest {manifest_path} does not match this run; "
                "use a new output path rather than mixing corpora"
            )
        return manifest_path
    if has_existing_games:
        raise ValueError(
            f"trajectory shards exist without {manifest_path}; refusing an "
            "unverifiable resume"
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=manifest_path.name + ".",
            suffix=".tmp",
            dir=manifest_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return manifest_path


def iter_batches(
    trajectories: Sequence[SelfPlayTrajectory],
    batch_size: int,
    rng,
    shuffle_buffer: int = 8192,
) -> Iterator[dict[str, np.ndarray]]:
    """Stream captured or legacy S2 rows through the shuffle-buffer loader."""
    from games.welcome_to import train

    yield from train.iter_batches(
        trajectories,
        batch_size,
        rng,
        shuffle_buffer=shuffle_buffer,
        replay_fn=replay,
    )


def _new_live(
    slot: int,
    seed: int,
    players: int,
    opponents: Sequence[Opponent],
) -> _LiveGame:
    assignment = PortableRng((seed ^ _POOL_DOMAIN) & _MASK64)
    population = list(range(len(opponents)))
    weights = [opponent.weight for opponent in opponents]
    indices = [-1]
    names = ["learner"]
    rngs: list[Optional[PortableRng]] = [None]
    for seat in range(1, players):
        index = assignment.choices(population, weights=weights, k=1)[0]
        indices.append(index)
        names.append(opponents[index].name)
        rngs.append(PortableRng((seed ^ _POLICY_DOMAIN ^ seat) & _MASK64))
    return _LiveGame(
        slot=slot,
        seed=seed,
        players=players,
        state=wr.RustGameState(
            seed,
            players=players,
            advanced=True,
            expert=False,
            solo_rules=False,
        ),
        opponent_indices=tuple(indices),
        opponent_names=tuple(names),
        policy_rngs=tuple(rngs),
        actions=[],
        searches=[],
        capture=wr.RustTrainingCapture(seed),
    )


def _summary(
    trajectories: Sequence[SelfPlayTrajectory],
    final_states: Sequence[GameState],
    evaluator_rows: int,
    evaluator_calls: int,
    batch_widths: dict[int, int],
    max_batch: int,
    seconds: float,
) -> dict[str, float]:
    def percentile(fraction: float) -> float:
        target = max(1, math.ceil(evaluator_calls * fraction))
        cumulative = 0
        for width, calls in sorted(batch_widths.items()):
            cumulative += calls
            if cumulative >= target:
                return float(width)
        return 0.0

    seats = sum(state.config.players for state in final_states)
    sheets = [sheet for state in final_states for sheet in state.sheets]
    searched = [target for trajectory in trajectories for target in trajectory.searches]
    learner_margins = [
        trajectory.scores[0] - max(trajectory.scores[1:]) for trajectory in trajectories
    ]
    metrics = {
        "games": float(len(trajectories)),
        "games_per_hour": len(trajectories) * 3600.0 / max(seconds, 1e-9),
        "evaluator_rows_per_second": evaluator_rows / max(seconds, 1e-9),
        "evaluator_rows": float(evaluator_rows),
        "evaluator_calls": float(evaluator_calls),
        "mean_batch": evaluator_rows / max(evaluator_calls, 1),
        "batch_p50": percentile(0.50),
        "batch_p90": percentile(0.90),
        "batch_p99": percentile(0.99),
        "batch_max": float(max(batch_widths, default=0)),
        "full_batch_fraction": batch_widths.get(max_batch, 0)
        / max(evaluator_calls, 1),
        "searched_roots": float(len(searched)),
        "mean_search_branching": (
            sum(len(target.actions) for target in searched) / len(searched)
            if searched
            else 0.0
        ),
        "mean_decisions_per_game": sum(len(t.actions) for t in trajectories)
        / len(trajectories),
        "learner_score": sum(t.scores[0] for t in trajectories) / len(trajectories),
        "learner_margin_vs_best": sum(learner_margins) / len(learner_margins),
        "permits_per_seat_game": sum(sheet.permits for sheet in sheets) / seats,
        "plans_per_seat_game": sum(
            sum(1 for slot in state.plan_turns if player in slot)
            for state in final_states
            for player in range(state.config.players)
        )
        / seats,
        "plan_ending_fraction": sum(
            "completed all three plans" in (state.end_of_game_reason() or "")
            for state in final_states
        )
        / len(final_states),
        "roundabouts_per_seat_game": sum(sheet.roundabouts for sheet in sheets) / seats,
        "bis_writes_per_seat_game": sum(
            sum(value for row in sheet.is_bis for value in row) for sheet in sheets
        )
        / seats,
        "estates_size_7plus_per_seat_game": sum(
            sum(size >= 7 for _, _, size in sheet.estates()) for sheet in sheets
        )
        / seats,
        "temps_per_seat_game": sum(sheet.temps for sheet in sheets) / seats,
    }
    metrics.update(training.diversity_report(list(final_states)))
    for players in (2, 3, 4):
        metrics[f"games_{players}p"] = float(sum(t.players == players for t in trajectories))
    return metrics


def generate(
    learner: nw.WelcomeToNet,
    *,
    config: Optional[SelfPlayConfig] = None,
    search_config: Optional[mcts.SearchConfig] = None,
    opponents: Optional[Sequence[Opponent]] = None,
    device: Optional[torch.device | str] = None,
    skip_seeds: Collection[int] = (),
    on_trajectory: Optional[Callable[[SelfPlayTrajectory], None]] = None,
    on_captured: Optional[Callable[[SelfPlayTrajectory, object], None]] = None,
) -> tuple[list[SelfPlayTrajectory], dict[str, Any]]:
    """Generate a continuously replenished batch of learner-only S2 games.

    ``skip_seeds`` is the resume seam. The full seed/seat schedule is built
    first and only persisted seeds are removed, so scheduler width and restart
    timing cannot change any remaining game's identity.
    """
    if wr is None:
        raise RuntimeError(
            "welcome_to_rust is not installed; run maturin develop --release in "
            "games/welcome_to/welcome_to_rust"
        )
    config = config or SelfPlayConfig()
    search_config = search_config or default_search_config()
    torch_device = torch.device(device or next(learner.parameters()).device)
    learner = learner.to(torch_device).eval()
    opponents = tuple(opponents or (Opponent("incumbent", learner),))
    if not opponents:
        raise ValueError("S2 needs at least one frozen opponent policy")
    for opponent in opponents:
        opponent.net.to(torch_device).eval()

    learner_eval = rust_search.PackedNetEvaluator(learner, torch_device, search_config)
    evaluator_by_net: dict[int, rust_search.PackedNetEvaluator] = {id(learner): learner_eval}
    opponent_evals = []
    for opponent in opponents:
        evaluator = evaluator_by_net.get(id(opponent.net))
        if evaluator is None:
            evaluator = rust_search.PackedNetEvaluator(
                opponent.net, torch_device, search_config
            )
            evaluator_by_net[id(opponent.net)] = evaluator
        opponent_evals.append(evaluator)

    jobs = list(zip(range(config.seed, config.seed + config.games), seat_counts(config.games)))
    PortableRng((config.seed ^ _JOB_ORDER_DOMAIN) & _MASK64).shuffle(jobs)
    skip_seeds = frozenset(int(seed) for seed in skip_seeds)
    requested_seeds = {seed for seed, _ in jobs}
    unknown = skip_seeds - requested_seeds
    if unknown:
        raise ValueError(f"skip_seeds contains seeds outside this run: {sorted(unknown)[:8]}")
    jobs = [job for job in jobs if job[0] not in skip_seeds]
    if not jobs:
        raise ValueError("no pending S2 games remain after applying skip_seeds")

    target_games = len(jobs)
    width = min(config.inflight, target_games)
    scheduler = rust_search.native_cloud_scheduler(
        search_config,
        capacity=width,
        workers=min(config.scheduler_workers, width),
    )
    next_job = 0
    live: list[Optional[_LiveGame]] = [None] * width
    for slot in range(width):
        seed, players = jobs[next_job]
        live[slot] = _new_live(slot, seed, players, opponents)
        next_job += 1

    trajectories: list[SelfPlayTrajectory] = []
    final_states: list[GameState] = []
    completed = 0
    started = time.perf_counter()
    while completed < target_games:
        learner_games = [game for game in live if game is not None and game.state.actor == 0]
        if learner_games:
            seeds = []
            noises = []
            temperatures = []
            for game in learner_games:
                search_seed = derive_search_seed(game.seed, game.learner_decisions)
                tape = PortableRng(search_seed)
                width_at_root = len(
                    game.state.search_legal_macros(search_config.prune_roundabout_pass)
                )
                noise, advanced_seed = rust_search.root_noise(
                    search_config, width_at_root, tape
                )
                seeds.append(advanced_seed)
                noises.append(None if noise is None else noise.tolist())
                temperatures.append(temperature_for_turn(config, game.state.turn))
            results = scheduler.play(
                [game.state for game in learner_games],
                learner_eval,
                seeds,
                roots=[0] * len(learner_games),
                noises=noises,
                temperatures=temperatures,
                slots=[game.slot for game in learner_games],
                max_batch=config.max_batch,
            )
            for game, result in zip(learner_games, results):
                choice = int(result["choice"])
                visits = tuple(int(round(value)) for value in result["visits"])
                if sum(visits) > 0:
                    if any(float(value) != visit for value, visit in zip(result["visits"], visits)):
                        raise RuntimeError("Rust search returned a non-integral visit count")
                    game.capture.capture(
                        game.state,
                        [int(action) for action in result["actions"]],
                        [float(value) for value in result["visits"]],
                        choice,
                        search_config.prune_roundabout_pass,
                    )
                    game.searches.append(
                        SearchTarget(
                            decision=len(game.actions),
                            actions=tuple(int(action) for action in result["actions"]),
                            visits=visits,
                        )
                    )
                game.actions.append(choice)
                game.learner_decisions += 1
                game.state.apply_macro(choice)

        by_opponent: dict[int, list[_LiveGame]] = {}
        for game in live:
            if game is None or game.state.is_terminal or game.state.actor == 0:
                continue
            by_opponent.setdefault(game.opponent_indices[game.state.actor], []).append(game)
        for opponent_index, games in by_opponent.items():
            evaluator = opponent_evals[opponent_index]
            policies, legals = evaluator.policy_states(
                [game.state for game in games], [game.players for game in games]
            )
            for game, policy, legal in zip(games, policies, legals):
                actor = game.state.actor
                rng = game.policy_rngs[actor]
                assert rng is not None
                choice = sample_policy(
                    legal,
                    policy,
                    rng,
                    temperature_for_turn(config, game.state.turn),
                )
                game.actions.append(int(choice))
                game.state.apply_macro(int(choice))

        for slot, game in enumerate(live):
            if game is None:
                continue
            if len(game.actions) > config.max_decisions:
                raise RuntimeError(
                    f"game seed {game.seed} exceeded {config.max_decisions} macro decisions"
                )
            if not game.state.is_terminal:
                continue
            final_state = snapshot.from_snapshot(game.state.snapshot())
            trajectory = SelfPlayTrajectory(
                seed=game.seed,
                players=game.players,
                actions=tuple(game.actions),
                searches=tuple(game.searches),
                scores=tuple(final_state.scores()),
                opponents=game.opponent_names,
                prune_roundabout_pass=search_config.prune_roundabout_pass,
            )
            captured = game.capture.finish(game.state, trajectory.to_json())
            if on_captured is not None:
                on_captured(trajectory, captured)
            if on_trajectory is not None:
                on_trajectory(trajectory)
            trajectories.append(trajectory)
            final_states.append(final_state)
            completed += 1
            scheduler.reset(slot)
            if next_job < len(jobs):
                seed, players = jobs[next_job]
                live[slot] = _new_live(slot, seed, players, opponents)
                next_job += 1
            else:
                live[slot] = None

    seconds = time.perf_counter() - started
    unique_evaluators = {id(evaluator): evaluator for evaluator in evaluator_by_net.values()}
    rows = sum(evaluator.rows for evaluator in unique_evaluators.values())
    calls = sum(evaluator.calls for evaluator in unique_evaluators.values())
    batch_widths: dict[int, int] = {}
    for evaluator in unique_evaluators.values():
        for width_at_call, count in evaluator.batch_widths.items():
            batch_widths[width_at_call] = batch_widths.get(width_at_call, 0) + count
    metrics = _summary(
        trajectories,
        final_states,
        rows,
        calls,
        batch_widths,
        config.max_batch,
        seconds,
    )
    metrics["batch_width_calls"] = {
        str(width_at_call): float(count)
        for width_at_call, count in sorted(batch_widths.items())
    }
    metrics["generation_seconds"] = seconds
    metrics["scheduler_profile"] = dict(scheduler.profile())
    metrics["evaluator_profiles"] = [
        evaluator.profile() for evaluator in unique_evaluators.values()
    ]
    return trajectories, metrics


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse

    from games.welcome_to import train

    parser = argparse.ArgumentParser(description="Generate S2 searched trajectories.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent-checkpoint", action="append", default=[])
    parser.add_argument("--games", type=int, default=SelfPlayConfig().games)
    parser.add_argument("--inflight", type=int, default=SelfPlayConfig().inflight)
    parser.add_argument("--max-batch", type=int, default=SelfPlayConfig().max_batch)
    parser.add_argument(
        "--scheduler-workers", type=int, default=SelfPlayConfig().scheduler_workers
    )
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opening-temperature-turns", type=int, default=10)
    parser.add_argument("--opening-temperature", type=float, default=1.0)
    parser.add_argument("--late-temperature", type=float, default=0.0)
    parser.add_argument("--shard-games", type=int, default=25)
    parser.add_argument("--writer-queue-games", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="runs/welcome_to_s2/trajectories.jsonl")
    args = parser.parse_args(argv)

    config = SelfPlayConfig(
        games=args.games,
        inflight=args.inflight,
        max_batch=args.max_batch,
        scheduler_workers=args.scheduler_workers,
        seed=args.seed,
        opening_temperature_turns=args.opening_temperature_turns,
        opening_temperature=args.opening_temperature,
        late_temperature=args.late_temperature,
    )
    search_config = default_search_config(args.simulations)
    learner = train.load(args.checkpoint, args.device)
    opponent_paths = args.opponent_checkpoint or [args.checkpoint]
    opponents = [
        Opponent(Path(path).stem, learner if path == args.checkpoint else train.load(path, args.device))
        for path in opponent_paths
    ]
    has_existing = bool(trajectory_sources(args.out) or training_shard_paths(args.out))
    existing = read_trajectories(args.out) if has_existing else []
    completed_seeds = validate_resume(existing, config)
    existing_games = len(existing)
    ensure_run_manifest(
        args.out,
        run_manifest(config, search_config, args.checkpoint, opponent_paths),
        has_existing_games=bool(existing_games),
    )
    # Resume validation needs full trajectories, generation does not. Do not
    # retain an old overnight corpus alongside the new one for the whole run.
    existing.clear()
    pending = config.games - len(completed_seeds)
    if pending == 0:
        print(f"S2 corpus already complete: {config.games} games at {args.out}")
        return 0

    store = wr.RustSampleShardWriter(
        args.out,
        shard_games=args.shard_games,
        queue_games=args.writer_queue_games,
    )
    try:
        trajectories, metrics = generate(
            learner,
            config=config,
            search_config=search_config,
            opponents=opponents,
            device=args.device,
            skip_seeds=completed_seeds,
            on_captured=lambda _trajectory, captured: store.add(captured),
        )
    finally:
        # Also drain the bounded queue and flush a short final shard on clean
        # completion, Ctrl-C, or a Python exception.
        store.close()

    path = Path(args.out)
    metrics["existing_games"] = float(existing_games)
    metrics["new_games"] = float(len(trajectories))
    metrics["total_games"] = float(len(completed_seeds) + len(trajectories))
    metrics_path = path.with_suffix(path.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"wrote {len(trajectories)} new S2 games / "
        f"{int(metrics['searched_roots'])} searched roots as shards of {path} "
        f"({int(metrics['total_games'])}/{config.games} complete)"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
