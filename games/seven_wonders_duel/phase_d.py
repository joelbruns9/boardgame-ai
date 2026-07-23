"""Phase D toy-scale AlphaZero loop infrastructure.

This module assembles deterministic self-play workers, request-coalesced neural
inference, replay windows, curriculum seeding/mixing, candidate training, SPRT
gates, promotion, HOF, Elo, and run manifests.  Defaults describe the intended
toy run; tests use deliberately tiny configurations and do not launch training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Sequence

import torch

from games.az_loop import (
    BootstrapPolicy,
    ControllerConfig,
    EloLedger,
    GameJob,
    GeneratorMode,
    HallOfFame,
    LinearSchedule,
    MatchOutcome,
    ReplayWindow,
    RunController,
    RunLog,
    RunManifest,
    SPRT,
    play_match,
    run_jobs,
    run_jobs_in_processes,
)

from .bots import (
    GreedyBot,
    MilitaryAggressiveBot,
    MilitaryEconomyBot,
    ScienceAggressiveBot,
    ScienceEconomyBot,
)
from .buffer import GameRecord, GameRecorder, read_records, to_json_line
from .codec import decode_action, encode_action
from .dataset import Example, examples_from_records
from .game import Phase
from .inference import Evaluator
from .loop_adapter import SevenWondersDuelLoopAdapter
from .loop_inference import CoalescingEvaluator
from .rust_bridge import (
    phase_d_records_from_rust,
    rust_flat_batch_adapter,
    rust_games_for_self_play,
    rust_seat_routed_flat_batch_adapter,
)
from .search import GumbelMCTS, SearchConfig, SearchResult, state_actor
from .train import (
    baselines,
    build_model,
    evaluate as evaluate_model,
    load_checkpoint,
    make_checkpoint,
    train_loop,
)


CURRICULUM_BOT_TYPES = (
    ScienceAggressiveBot,
    ScienceEconomyBot,
    MilitaryAggressiveBot,
    MilitaryEconomyBot,
)

# Locked ZeusAI context-free tier list (AZ_PROJECT_PLAN.md §2). Route bias
# belongs in the curriculum games; this prior supplies broad draft competence
# and anneals away. Values encode tiers, not calibrated win-rate differences.
WONDER_DRAFT_TIERS = {
    "The Temple of Artemis": 1.0,
    "Piraeus": 1.0,
    "The Hanging Gardens": 1.0,
    "The Appian Way": 1.0,
    "The Sphinx": 1.0,
    "The Statue of Zeus": 0.8,
    "The Great Library": 0.8,
    "The Mausoleum": 0.6,
    "Circus Maximus": 0.6,
    "The Colossus": 0.6,
    "The Great Lighthouse": 0.4,
    "The Pyramids": 0.0,
}


@dataclass(slots=True)
class PhaseDConfig:
    run_dir: str = "runs/seven_wonders_duel/phase_d"
    seed: int = 20260718
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    workers: int = 8
    process_workers: int = 0
    inference_batch: int = 64
    inference_wait_ms: float = 2.0
    iterations: int = 1
    games_per_iteration: int = 500
    seed_games: int = 5_000
    replay_window: int = 20
    save_buffer: str = ""
    warm_buffer: str = ""
    seed_retain_fraction: float = 1.0
    curriculum_anneal_iterations: int = 10
    opponent_fraction: float = 0.15
    bot_policy_iterations: int = 10
    bot_exploration: float = 0.05
    draft_prior_iterations: int = 20
    cheap_sims_min: int = 16
    cheap_sims_max: int = 24
    full_sims_min: int = 64
    full_sims_max: int = 128
    full_search_fraction: float = 0.25
    search_mode: str = "closed"
    top_k: int = 16
    d_model: int = 128
    layers: int = 4
    train_epochs: int = 8
    train_batch_size: int = 512
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    aux_weight: float = 0.2
    train_patience: int = 8
    val_fraction: float = 0.1
    min_games_to_train: int = 2
    gate_sims: int = 64
    gate_max_games: int = 400
    gate_alpha: float = 0.05
    gate_beta: float = 0.05
    gate_indifference: float = 0.03
    anchor_gate_every_promotions: int = 3
    selfplay_generator_mode: str = "strict_gate"
    bootstrap_policy: str = "gate"
    promotion_every: int = 1
    revert_reset_after: int = 0
    buffer_autosave_every: int = 0
    warm_buffer_max_staleness: int = 0
    generation_backend: str = "rust"
    gate_backend: str = "rust"
    rust_slots: int = 16
    rust_global_batch_cap: int = 256
    rust_max_inflight_batches: int = 1
    rust_scheduler_workers: int = 1
    leaf_batch: int = 1
    force_root_chance: bool = True
    age_deal_samples: int = 32

    def validate(self) -> None:
        if self.workers <= 0 or self.games_per_iteration <= 0:
            raise ValueError("workers and games_per_iteration must be positive")
        if self.process_workers < 0:
            raise ValueError("process_workers must be non-negative")
        if self.seed_games < 0 or self.replay_window <= 0:
            raise ValueError(
                "seed_games must be non-negative and replay_window positive"
            )
        for name in (
            "seed_retain_fraction",
            "opponent_fraction",
            "full_search_fraction",
            "bot_exploration",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 1 <= self.cheap_sims_min <= self.cheap_sims_max:
            raise ValueError("invalid cheap simulation range")
        if not 1 <= self.full_sims_min <= self.full_sims_max:
            raise ValueError("invalid full simulation range")
        if self.search_mode not in ("closed", "open"):
            raise ValueError("search_mode must be closed or open")
        if self.gate_max_games <= 0 or self.gate_max_games % 2:
            raise ValueError("gate_max_games must be a positive even number")
        if self.anchor_gate_every_promotions < 0:
            raise ValueError("anchor_gate_every_promotions must be non-negative")
        valid_modes = {mode.value for mode in GeneratorMode}
        if self.selfplay_generator_mode not in valid_modes:
            raise ValueError(
                f"selfplay_generator_mode must be one of {sorted(valid_modes)}"
            )
        valid_policies = {policy.value for policy in BootstrapPolicy}
        if self.bootstrap_policy not in valid_policies:
            raise ValueError(
                f"bootstrap_policy must be one of {sorted(valid_policies)}"
            )
        if self.promotion_every < 0:
            raise ValueError("promotion_every must be non-negative")
        if self.revert_reset_after < 0:
            raise ValueError("revert_reset_after must be non-negative")
        if self.buffer_autosave_every < 0:
            raise ValueError("buffer_autosave_every must be non-negative")
        if self.warm_buffer_max_staleness < 0:
            raise ValueError("warm_buffer_max_staleness must be non-negative")
        if self.gate_backend not in ("rust", "python"):
            raise ValueError("gate_backend must be rust or python")
        if self.generation_backend not in ("rust", "python"):
            raise ValueError("generation_backend must be rust or python")
        if min(
            self.rust_slots,
            self.rust_global_batch_cap,
            self.rust_max_inflight_batches,
            self.rust_scheduler_workers,
            self.leaf_batch,
        ) <= 0:
            raise ValueError("Rust scheduler geometry must be positive")
        if self.leaf_batch > self.rust_global_batch_cap:
            raise ValueError("leaf_batch cannot exceed rust_global_batch_cap")
        if not 0 <= self.age_deal_samples <= 32:
            raise ValueError("age_deal_samples must be in [0, 32]")
        if self.d_model <= 0 or self.d_model % 4 or self.layers <= 0:
            raise ValueError("d_model must be positive/divisible by 4 and layers positive")
        if self.train_epochs <= 0 or self.train_batch_size <= 0:
            raise ValueError("training epochs and batch size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.train_patience <= 0:
            raise ValueError("train_patience must be positive")


def curriculum_fraction(initial: float, iteration: int, duration: int) -> float:
    return LinearSchedule(initial, 0.0, duration).value(iteration)


def temperature_for_move(move_index: int) -> float:
    return LinearSchedule(1.0, 0.25, 20).value(move_index)


def phase_d_game_honest_split(
    examples: list[Example], val_frac: float, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """Online split: hold games out within each self-play generation.

    Every labeled iteration contributes fresh games to both train and
    validation without sharing a game between them. Curriculum examples
    (iteration=None) are training-only. The offline trainer retains its
    whole-iteration split for temporal/generalization analysis.
    """

    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must lie in [0, 1)")
    validation_keys: set[tuple[int, int]] = set()
    iterations = sorted(
        {example.iteration for example in examples if example.iteration is not None}
    )
    for iteration in iterations:
        game_keys = sorted(
            {
                example.game_key
                for example in examples
                if example.iteration == iteration
            }
        )
        if val_frac <= 0.0 or len(game_keys) < 2:
            continue
        rng = random.Random(seed ^ (iteration * 0x9E3779B1))
        rng.shuffle(game_keys)
        held = min(len(game_keys) - 1, max(1, round(len(game_keys) * val_frac)))
        validation_keys.update((iteration, key) for key in game_keys[:held])
    train = [
        example
        for example in examples
        if example.iteration is None
        or (example.iteration, example.game_key) not in validation_keys
    ]
    validation = [
        example
        for example in examples
        if example.iteration is not None
        and (example.iteration, example.game_key) in validation_keys
    ]
    return train, validation


def should_run_anchor_gate(
    *, promoted: bool, previous_promotions: int, cadence: int
) -> bool:
    if not promoted or cadence <= 0:
        return False
    return (previous_promotions + 1) % cadence == 0


def _normalize(weights: dict[int, float]) -> dict[int, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0.0:
        uniform = 1.0 / len(weights)
        return {key: uniform for key in weights}
    return {key: max(0.0, value) / total for key, value in weights.items()}


def blend_draft_priors(
    state, priors: dict[int, float], amount: float
) -> dict[int, float]:
    """Blend neural priors with a public Wonder tier prior at draft nodes."""

    if state.phase is not Phase.WONDER_DRAFT or amount <= 0.0:
        return _normalize(priors)
    amount = min(1.0, amount)
    logits = {}
    for index in priors:
        wonder = decode_action(state, index).wonder_name
        if wonder is None:
            raise AssertionError("draft action is missing a Wonder")
        logits[index] = WONDER_DRAFT_TIERS[wonder]
    peak = max(logits.values())
    tier = _normalize({key: math.exp(value - peak) for key, value in logits.items()})
    neural = _normalize(priors)
    return _normalize(
        {
            key: (1.0 - amount) * neural[key] + amount * tier[key]
            for key in neural
        }
    )


class CurriculumMCTS(GumbelMCTS):
    def __init__(self, evaluator, config: SearchConfig, draft_prior: float = 0.0):
        super().__init__(evaluator, config)
        self.draft_prior = draft_prior

    def _evaluate(self, state):
        value, priors = super()._evaluate(state)
        return value, blend_draft_priors(state, priors, self.draft_prior)


def _sample_policy(
    policy: dict[int, float], temperature: float, rng: random.Random
) -> int:
    actions = sorted(policy)
    if temperature <= 0.0:
        return max(actions, key=policy.__getitem__)
    power = 1.0 / temperature
    weights = [max(policy[action], 1e-12) ** power for action in actions]
    return rng.choices(actions, weights=weights, k=1)[0]


class BotAgent:
    def __init__(self, bot):
        self.bot = bot
        self.name = bot.name

    def select_action(self, state, legal_actions, rng) -> int:
        action = self.bot.select_action(state)
        return encode_action(state, action)


class SearchAgent:
    def __init__(
        self,
        name: str,
        evaluator,
        *,
        sims: int,
        mode: str,
        top_k: int,
        draft_prior: float = 0.0,
    ):
        self.name = name
        self.evaluator = evaluator
        self.sims = sims
        self.mode = mode
        self.top_k = top_k
        self.draft_prior = draft_prior

    def select_action(self, state, legal_actions, rng) -> int:
        search = CurriculumMCTS(
            self.evaluator,
            SearchConfig(
                sims=self.sims,
                top_k=self.top_k,
                mode=self.mode,
                seed=rng.getrandbits(63),
            ),
            self.draft_prior,
        )
        return search.search(state).action_index


def _write_records(path: Path, records: Sequence[GameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(to_json_line(record) + "\n")
    temporary.replace(path)


def filter_warm_records_by_staleness(
    records: Sequence[GameRecord], max_staleness: int
) -> tuple[list[GameRecord], dict[str, int]]:
    """Drop imported games older than ``max_staleness`` iterations.

    Age is measured against the newest numbered iteration present in the import.
    Curriculum records (``iteration is None``) are never aged out.  Source
    iteration metadata is preserved exactly -- records are filtered, never
    renumbered.  Returns the retained records and actual loaded/retained/dropped
    counts.
    """

    numbered = [
        record.iteration for record in records if record.iteration is not None
    ]
    newest = max(numbered, default=0)
    retained = [
        record
        for record in records
        if record.iteration is None or (newest - record.iteration) < max_staleness
    ]
    stats = {
        "loaded": len(records),
        "retained": len(retained),
        "dropped": len(records) - len(retained),
        "newest_iteration": newest,
        "max_staleness": max_staleness,
    }
    return retained, stats


def summarize_records(records: Sequence[GameRecord]) -> dict[str, Any]:
    moves = [move for record in records for move in record.moves]
    searched = [move for move in moves if move.sims > 0]
    eligible = [move for move in moves if not move.policy_excluded]
    kinds = Counter(record.agents.get("kind", "unknown") for record in records)
    victories = Counter(record.victory_type or "draw" for record in records)
    return {
        "games": len(records),
        "moves": len(moves),
        "game_kinds": dict(sorted(kinds.items())),
        "victory_types": dict(sorted(victories.items())),
        "policy_eligible_moves": len(eligible),
        "policy_eligible_fraction": len(eligible) / len(moves) if moves else 0.0,
        "searched_moves": len(searched),
        "average_sims": (
            sum(move.sims for move in searched) / len(searched) if searched else 0.0
        ),
    }


def _bot_seed_game(job: GameJob) -> GameRecord:
    bot_type = CURRICULUM_BOT_TYPES[
        (job.index // 2) % len(CURRICULUM_BOT_TYPES)
    ]
    rush = bot_type(seed=job.seed ^ 0xA5A5)
    greedy = GreedyBot()
    rush_is_zero = job.index % 2 == 0
    bots = (rush, greedy) if rush_is_zero else (greedy, rush)
    recorder = GameRecorder(
        job.seed,
        first_player=(job.index // 2) % 2,
        agents={"p0": bots[0].name, "p1": bots[1].name, "kind": "curriculum_seed"},
        iteration=None,
    )
    while recorder.game.phase is not Phase.COMPLETE:
        actor = state_actor(recorder.game)
        action = bots[actor].select_action(recorder.game)
        recorder.play(encode_action(recorder.game, action))
    return recorder.finish()


def generate_seed_buffer(
    path: str | Path,
    *,
    games: int,
    seed: int,
    workers: int,
    process_workers: int = 0,
    backend: str = "python",
    rust_slots: int = 16,
    rust_global_batch_cap: int = 256,
) -> list[GameRecord]:
    destination = Path(path)
    if destination.exists():
        existing = read_records(destination)
        if len(existing) != games:
            raise ValueError(
                f"seed buffer has {len(existing)} games, expected {games}; "
                "use a new run directory to change seed_games"
            )
        return existing
    jobs = [
        GameJob(index=index, seed=seed + 10_000_000 + index, kind="curriculum_seed")
        for index in range(games)
    ]
    if backend == "rust":
        import seven_wonders_rust as swr

        grouped: dict[tuple[str, int], list[GameJob]] = {}
        for job in jobs:
            bot_type = CURRICULUM_BOT_TYPES[
                (job.index // 2) % len(CURRICULUM_BOT_TYPES)
            ]
            grouped.setdefault((bot_type().name, job.index % 2), []).append(job)
        indexed: dict[int, GameRecord] = {}
        for (rush_name, rush_seat), group_jobs in grouped.items():
            for start in range(0, len(group_jobs), rust_slots):
                chunk = group_jobs[start : start + rust_slots]
                seeds = [job.seed for job in chunk]
                first_players = [(job.index // 2) % 2 for job in chunk]
                raw_records, _ = swr.self_play_many_flat_net(
                    adapter=lambda _payload: [],
                    games=rust_games_for_self_play(seeds, first_players),
                    game_seeds=seeds,
                    global_batch_cap=rust_global_batch_cap,
                    leaf_batch=1,
                    cheap_sims_min=1,
                    cheap_sims_max=1,
                    full_sims_min=1,
                    full_sims_max=1,
                    full_search_fraction=0.0,
                    top_k=1,
                    draft_prior=0.0,
                    iteration=None,
                    bot_p0=rush_name if rush_seat == 0 else "greedy",
                    bot_p1=rush_name if rush_seat == 1 else "greedy",
                    bot_exploration=0.0,
                    bot_policy_iterations=0,
                )
                for raw in raw_records:
                    raw["agents"]["kind"] = "curriculum_seed"
                converted = phase_d_records_from_rust(raw_records)
                indexed.update(
                    {job.index: record for job, record in zip(chunk, converted)}
                )
        records = [indexed[job.index] for job in jobs]
    elif process_workers:
        records = run_jobs_in_processes(
            jobs, _bot_seed_game, workers=process_workers
        )
    else:
        records = run_jobs(jobs, _bot_seed_game, workers=workers)
    _write_records(destination, records)
    return records


# Per-process state for run_jobs_in_processes generation. The initializer runs
# once per spawned worker; the dict never leaks between processes.
_PROCESS_STATE: dict[str, Any] = {}


def _process_generation_init(
    model_state: dict[str, torch.Tensor], config: PhaseDConfig, iteration: int
) -> None:
    # One BLAS thread per process: generation scales by process count, and
    # oversubscribing cores with intra-op threads slows every worker down.
    torch.set_num_threads(1)
    model = build_model("transformer", config.d_model, config.layers)
    model.load_state_dict(model_state)
    # CPU inference per process: at generation batch sizes the tiny network is
    # a few ms on a core, while fanning every process into one GPU serializes
    # on the CUDA context. The GPU stays free for training and gates.
    _PROCESS_STATE["evaluator"] = Evaluator(model, "cpu", config.inference_batch)
    _PROCESS_STATE["config"] = config
    _PROCESS_STATE["iteration"] = iteration


def _process_self_play_game(job: GameJob) -> GameRecord:
    return _self_play_game(
        job,
        _PROCESS_STATE["evaluator"],
        _PROCESS_STATE["config"],
        _PROCESS_STATE["iteration"],
    )


@dataclass(frozen=True, slots=True)
class ModelAgentSpec:
    """Picklable recipe for a SearchAgent; built parent- or child-side."""

    name: str
    model_state: dict[str, torch.Tensor]
    d_model: int
    layers: int
    sims: int
    mode: str
    top_k: int


@dataclass(frozen=True, slots=True)
class BotAgentSpec:
    bot: Any


GateAgentSpec = ModelAgentSpec | BotAgentSpec


def _spec_name(spec: GateAgentSpec) -> str:
    return spec.bot.name if isinstance(spec, BotAgentSpec) else spec.name


def _build_gate_agent(spec: GateAgentSpec, device: str, inference_batch: int):
    if isinstance(spec, BotAgentSpec):
        return BotAgent(spec.bot)
    model = build_model("transformer", spec.d_model, spec.layers)
    model.load_state_dict(spec.model_state)
    return SearchAgent(
        spec.name,
        Evaluator(model, device, inference_batch),
        sims=spec.sims,
        mode=spec.mode,
        top_k=spec.top_k,
    )


def _process_gate_init(
    candidate_spec: GateAgentSpec,
    opponent_spec: GateAgentSpec,
    inference_batch: int,
) -> None:
    torch.set_num_threads(1)
    _PROCESS_STATE["gate_adapter"] = SevenWondersDuelLoopAdapter()
    _PROCESS_STATE["gate_candidate"] = _build_gate_agent(
        candidate_spec, "cpu", inference_batch
    )
    _PROCESS_STATE["gate_opponent"] = _build_gate_agent(
        opponent_spec, "cpu", inference_batch
    )


def _process_gate_game(job: GameJob):
    candidate = _PROCESS_STATE["gate_candidate"]
    opponent = _PROCESS_STATE["gate_opponent"]
    agents = (
        (candidate, opponent)
        if job.payload["candidate_is_zero"]
        else (opponent, candidate)
    )
    return play_match(
        _PROCESS_STATE["gate_adapter"],
        agents,
        seed=job.seed,
        first_player=job.payload["first_player"],
    )


def _search_move(
    game,
    evaluator,
    config: PhaseDConfig,
    iteration: int,
    move_index: int,
    rng: random.Random,
) -> tuple[int, SearchResult, bool]:
    full = rng.random() < config.full_search_fraction
    sims = rng.randint(
        config.full_sims_min if full else config.cheap_sims_min,
        config.full_sims_max if full else config.cheap_sims_max,
    )
    draft_amount = LinearSchedule(1.0, 0.0, config.draft_prior_iterations).value(
        iteration
    )
    search = CurriculumMCTS(
        evaluator,
        SearchConfig(
            sims=sims,
            top_k=config.top_k,
            mode=config.search_mode,
            seed=rng.getrandbits(63),
        ),
        draft_amount,
    )
    result = search.search(game)
    action = _sample_policy(result.policy_target, temperature_for_move(move_index), rng)
    return action, result, full


def _self_play_game(
    job: GameJob,
    evaluator,
    config: PhaseDConfig,
    iteration: int,
) -> GameRecord:
    rng = random.Random(job.seed ^ 0xC6BC279692B5CC83)
    mix_fraction = curriculum_fraction(
        config.opponent_fraction, iteration, config.curriculum_anneal_iterations
    )
    mixed = rng.random() < mix_fraction
    bot = None
    bot_seat = None
    if mixed:
        bot_type = CURRICULUM_BOT_TYPES[
            (job.index // 2) % len(CURRICULUM_BOT_TYPES)
        ]
        bot = bot_type(seed=job.seed ^ 0x51ED, exploration=0.05)
        bot_seat = job.index % 2
    agents = {
        "p0": bot.name if bot_seat == 0 else "network",
        "p1": bot.name if bot_seat == 1 else "network",
        "kind": "mixed" if mixed else "self_play",
    }
    recorder = GameRecorder(
        job.seed,
        first_player=(job.index // 2) % 2,
        agents=agents,
        iteration=iteration,
    )
    move_index = 0
    while recorder.game.phase is not Phase.COMPLETE:
        actor = state_actor(recorder.game)
        if actor == bot_seat:
            action = encode_action(recorder.game, bot.select_action(recorder.game))
            recorder.play(
                action,
                mode="bot",
                policy_excluded=iteration >= config.bot_policy_iterations,
            )
        else:
            action, result, full = _search_move(
                recorder.game,
                evaluator,
                config,
                iteration,
                move_index,
                rng,
            )
            recorder.play(
                action,
                visits=result.visits,
                policy_target=result.policy_target,
                root_value=result.root_value,
                sims=result.sims,
                mode=result.mode,
                gumbel_topk=result.gumbel_topk,
                policy_excluded=not full,
            )
        move_index += 1
    return recorder.finish()


@dataclass(frozen=True, slots=True)
class GateResult:
    opponent: str
    threshold: float
    decision: str
    games: int
    score_rate: float


class PhaseDLoop:
    def __init__(self, config: PhaseDConfig):
        config.validate()
        self.config = config
        self.run_dir = Path(config.run_dir)
        self.buffer_dir = self.run_dir / "buffers"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.current_best = self.checkpoint_dir / "current_best.pt"
        self.adapter = SevenWondersDuelLoopAdapter()
        self.hof = HallOfFame(self.run_dir / "hof")
        self.elo = EloLedger(
            self.run_dir / "elo", fixed_ratings={GreedyBot.name: 1000.0}
        )
        self.manifest = RunManifest(self.run_dir, Path(__file__).resolve().parents[2])
        self.training_log = self.run_dir / "training_log.jsonl"
        self.warm_records: list[GameRecord] = []
        self.last_generation_stats: dict[str, Any] = {}
        self.last_training_stats: dict[str, Any] = {}
        self.last_warm_stats: dict[str, int] = {}

    def _append_training_log(self, row: dict[str, Any]) -> None:
        """Append one completed iteration using the manifest's existing metrics."""

        with self.training_log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def _sync_training_log(self, manifest: dict[str, Any]) -> None:
        """Backfill manifest rows missing from an interrupted or older run's log."""

        logged_iterations: set[int] = set()
        if self.training_log.exists():
            with self.training_log.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        logged_iterations.add(int(json.loads(line)["iteration"]))
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            f"invalid training log row {line_number}: "
                            f"{self.training_log}"
                        ) from exc
        for row in manifest.get("iterations", []):
            if int(row["iteration"]) not in logged_iterations:
                self._append_training_log(row)

    def _new_model(self):
        return build_model("transformer", self.config.d_model, self.config.layers)

    def _warm_records_for_iteration(self, iteration: int) -> list[GameRecord]:
        """Age imported records through the same iteration replay window."""

        if not self.warm_records:
            return []
        numbered = [
            record.iteration
            for record in self.warm_records
            if record.iteration is not None
        ]
        newest = max(numbered, default=0)
        return [
            record
            for record in self.warm_records
            if newest - (record.iteration if record.iteration is not None else newest)
            + iteration
            < self.config.replay_window
        ]

    def _save_replay_buffer(self) -> None:
        """Atomically export the replay set available at the latest generation."""

        if not self.config.save_buffer:
            return
        iteration_paths = sorted(self.buffer_dir.glob("iter_[0-9][0-9][0-9][0-9].jsonl"))
        if iteration_paths:
            latest = int(iteration_paths[-1].stem.removeprefix("iter_"))
            records = self.training_records(latest)
        else:
            records = self._warm_records_for_iteration(0)
            seed_path = self.buffer_dir / "curriculum_seed.jsonl"
            if seed_path.exists():
                records += read_records(seed_path)
        destination = Path(self.config.save_buffer)
        _write_records(destination, records)
        print(f"Buffer saved: {len(records)} games -> {destination}")

    def _load_warm_buffer(self, warm_path: Path) -> None:
        """Import a saved buffer, applying the staleness age filter."""

        if not warm_path.exists():
            raise FileNotFoundError(f"warm buffer does not exist: {warm_path}")
        max_staleness = (
            self.config.warm_buffer_max_staleness or self.config.replay_window
        )
        records = read_records(warm_path)
        self.warm_records, self.last_warm_stats = filter_warm_records_by_staleness(
            records, max_staleness
        )
        stats = self.last_warm_stats
        print(
            f"Buffer loaded: {stats['loaded']} games from {warm_path} "
            f"(retained {stats['retained']}, dropped {stats['dropped']} "
            f"older than staleness {max_staleness})"
        )

    def _autosave_replay_buffer(self, completed_iterations: int) -> None:
        """Atomically autosave every N completed iterations; never fatal.

        Scheduling and failure policy live here so a failed export warns and the
        run continues.  The write itself is atomic (temp + os.replace), so a hard
        kill mid-save can only leave a stale ``.tmp`` beside the last valid
        export, never a truncated replacement.
        """

        every = self.config.buffer_autosave_every
        if every <= 0 or not self.config.save_buffer:
            return
        if completed_iterations % every != 0:
            return
        try:
            self._save_replay_buffer()
        except Exception as exc:  # never terminate training on an autosave failure
            print(
                f"WARNING: buffer autosave failed after "
                f"{completed_iterations} iterations: {exc}"
            )

    def initialize(self, *, bootstrap_checkpoint: bool = True) -> None:
        had_manifest = self.manifest.path.exists()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if not had_manifest:
            model = self._new_model()
            self.manifest.initialize(
                config=self.config,
                adapter_contract=self.adapter.contract(),
                model_contract={
                    "model": "transformer",
                    "d_model": self.config.d_model,
                    "layers": self.config.layers,
                    "parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                },
            )
        # The soft-gate controller owns latest.pt/current_best.pt creation via the
        # lifecycle adapter; the legacy strict-gate path seeds current_best here.
        if bootstrap_checkpoint and not self.current_best.exists():
            if had_manifest:
                payload = json.loads(self.manifest.path.read_text(encoding="utf-8"))
                if payload.get("iterations") or payload.get("checkpoints"):
                    raise FileNotFoundError(
                        "current_best.pt is missing from an established run; "
                        "refusing to resume from random weights"
                    )
            torch.manual_seed(self.config.seed)
            checkpoint = make_checkpoint(
                self._new_model(),
                {
                    "model": "transformer",
                    "d_model": self.config.d_model,
                    "layers": self.config.layers,
                    "iteration": -1,
                },
            )
            torch.save(checkpoint, self.current_best)
        manifest_payload = json.loads(self.manifest.path.read_text(encoding="utf-8"))
        self._sync_training_log(manifest_payload)
        if self.config.warm_buffer:
            self._load_warm_buffer(Path(self.config.warm_buffer))
        if self.config.seed_games:
            generate_seed_buffer(
                self.buffer_dir / "curriculum_seed.jsonl",
                games=self.config.seed_games,
                seed=self.config.seed,
                workers=self.config.workers,
                process_workers=self.config.process_workers,
                backend=self.config.generation_backend,
                rust_slots=self.config.rust_slots,
                rust_global_batch_cap=self.config.rust_global_batch_cap,
            )

    def load_model(self, path: str | Path):
        model = self._new_model()
        load_checkpoint(path, model)
        return model

    @staticmethod
    def checkpoint_agent_name(path: str | Path, role: str) -> str:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        iteration = checkpoint.get("config", {}).get("iteration", "unknown")
        return f"{role}_iter_{iteration}"

    def generate_iteration(self, model, iteration: int) -> list[GameRecord]:
        destination = self.buffer_dir / f"iter_{iteration:04d}.jsonl"
        if destination.exists():
            raise FileExistsError(f"iteration buffer already exists: {destination}")
        jobs = [
            GameJob(
                index=index,
                seed=self.config.seed + iteration * 1_000_000 + index,
            )
            for index in range(self.config.games_per_iteration)
        ]
        if self.config.generation_backend == "rust":
            return self._generate_iteration_rust(model, iteration, destination, jobs)
        if self.config.process_workers:
            source = getattr(model, "_orig_mod", model)
            model_state = {
                key: value.cpu() for key, value in source.state_dict().items()
            }
            started = time.monotonic()
            records = run_jobs_in_processes(
                jobs,
                _process_self_play_game,
                workers=self.config.process_workers,
                initializer=_process_generation_init,
                initargs=(model_state, self.config, iteration),
            )
            elapsed = time.monotonic() - started
            self.last_generation_stats = {
                "seconds": elapsed,
                "games_per_second": len(records) / elapsed if elapsed else 0.0,
                "mode": "process",
                "process_workers": self.config.process_workers,
            }
            _write_records(destination, records)
            return records
        base = Evaluator(model, self.config.device, self.config.inference_batch)
        started = time.monotonic()
        with CoalescingEvaluator(
            base,
            max_batch=self.config.inference_batch,
            max_wait_ms=self.config.inference_wait_ms,
        ) as service:
            records = run_jobs(
                jobs,
                lambda job: _self_play_game(job, service, self.config, iteration),
                workers=self.config.workers,
            )
            elapsed = time.monotonic() - started
            self.last_generation_stats = {
                "seconds": elapsed,
                "games_per_second": len(records) / elapsed if elapsed else 0.0,
                "mode": "thread",
                "inference_batches": service.batches,
                "inference_positions": service.positions,
                "mean_inference_batch": (
                    service.positions / service.batches if service.batches else 0.0
                ),
            }
        _write_records(destination, records)
        return records

    def _generate_iteration_rust(
        self,
        model,
        iteration: int,
        destination: Path,
        jobs: list[GameJob],
    ) -> list[GameRecord]:
        """Generate neural and curriculum-bot games in the Rust scheduler."""

        import seven_wonders_rust as swr

        mix_fraction = curriculum_fraction(
            self.config.opponent_fraction,
            iteration,
            self.config.curriculum_anneal_iterations,
        )
        grouped: dict[tuple[str | None, int | None], list[GameJob]] = {}
        for job in jobs:
            rng = random.Random(job.seed ^ 0xC6BC279692B5CC83)
            if rng.random() < mix_fraction:
                bot_type = CURRICULUM_BOT_TYPES[
                    (job.index // 2) % len(CURRICULUM_BOT_TYPES)
                ]
                key = (bot_type().name, job.index % 2)
            else:
                key = (None, None)
            grouped.setdefault(key, []).append(job)

        evaluator = Evaluator(
            model,
            self.config.device,
            self.config.rust_global_batch_cap,
        )
        adapter = rust_flat_batch_adapter(evaluator)
        started = time.monotonic()
        indexed: dict[int, GameRecord] = {}
        rust_metrics = []
        draft_prior = LinearSchedule(
            1.0, 0.0, self.config.draft_prior_iterations
        ).value(iteration)
        neural_games = 0
        bot_games = 0
        for (bot_name, bot_seat), group_jobs in grouped.items():
            if bot_name is None:
                neural_games += len(group_jobs)
            else:
                bot_games += len(group_jobs)
            for start in range(0, len(group_jobs), self.config.rust_slots):
                chunk = group_jobs[start : start + self.config.rust_slots]
                seeds = [job.seed for job in chunk]
                first_players = [(job.index // 2) % 2 for job in chunk]
                raw_records, metrics = swr.self_play_many_flat_net(
                    adapter=adapter,
                    games=rust_games_for_self_play(seeds, first_players),
                    game_seeds=seeds,
                    global_batch_cap=self.config.rust_global_batch_cap,
                    leaf_batch=self.config.leaf_batch,
                    cheap_sims_min=self.config.cheap_sims_min,
                    cheap_sims_max=self.config.cheap_sims_max,
                    full_sims_min=self.config.full_sims_min,
                    full_sims_max=self.config.full_sims_max,
                    full_search_fraction=self.config.full_search_fraction,
                    top_k=self.config.top_k,
                    draft_prior=draft_prior,
                    iteration=iteration,
                    force=self.config.force_root_chance,
                    age_deal_samples=self.config.age_deal_samples,
                    max_inflight_batches=self.config.rust_max_inflight_batches,
                    scheduler_workers=self.config.rust_scheduler_workers,
                    bot_p0=bot_name if bot_seat == 0 else None,
                    bot_p1=bot_name if bot_seat == 1 else None,
                    bot_exploration=self.config.bot_exploration,
                    bot_policy_iterations=self.config.bot_policy_iterations,
                )
                converted = phase_d_records_from_rust(raw_records)
                indexed.update(
                    {job.index: record for job, record in zip(chunk, converted)}
                )
                rust_metrics.append(metrics)

        records = [indexed[job.index] for job in jobs]
        elapsed = time.monotonic() - started
        self.last_generation_stats = {
            "seconds": elapsed,
            "games_per_second": len(records) / elapsed if elapsed else 0.0,
            "mode": "rust",
            "rust_games": neural_games,
            "rust_bot_games": bot_games,
            "python_bot_games": 0,
            "rust_chunks": len(rust_metrics),
            "python_inference_batches": 0,
            "python_inference_positions": 0,
        }
        _write_records(destination, records)
        return records

    def training_records(self, iteration: int) -> list[GameRecord]:
        window = ReplayWindow(self.config.replay_window)
        paths = window.paths(self.buffer_dir, iteration)
        live = self._warm_records_for_iteration(iteration)
        live.extend(record for path in paths for record in read_records(path))
        seed_fraction = curriculum_fraction(
            self.config.seed_retain_fraction,
            iteration,
            self.config.curriculum_anneal_iterations,
        )
        seed_path = self.buffer_dir / "curriculum_seed.jsonl"
        if seed_fraction <= 0.0 or not seed_path.exists():
            return live
        seed_records = read_records(seed_path)
        desired = round(len(seed_records) * seed_fraction)
        rng = random.Random(self.config.seed + iteration)
        rng.shuffle(seed_records)
        return live + seed_records[: min(desired, len(seed_records))]

    def train_candidate(
        self,
        records: list[GameRecord],
        iteration: int,
        *,
        source_checkpoint: str | Path | None = None,
    ) -> Path:
        if len(records) < self.config.min_games_to_train:
            raise ValueError(
                f"need {self.config.min_games_to_train} games to train, "
                f"got {len(records)}"
            )
        examples = examples_from_records(records)
        target_baselines = baselines(examples)
        train_examples, val_examples = phase_d_game_honest_split(
            examples, self.config.val_fraction, self.config.seed + iteration
        )
        if not train_examples:
            raise ValueError("game-honest split produced no training examples")
        torch.manual_seed(self.config.seed + iteration)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed + iteration)
        # Soft-gate runs continue the rolling learner from latest.pt; strict-gate
        # and direct callers default to current_best (the historical behavior).
        model = self.load_model(
            source_checkpoint if source_checkpoint is not None else self.current_best
        )
        newest_iteration = max(
            (
                example.iteration
                for example in examples
                if example.iteration is not None
            ),
            default=None,
        )
        temporal_examples = (
            [example for example in examples if example.iteration == newest_iteration]
            if newest_iteration is not None
            else []
        )
        pretrain_metrics = None
        if temporal_examples:
            model.to(self.config.device)
            pretrain_metrics = evaluate_model(
                model,
                temporal_examples,
                self.config.device,
                self.config.train_batch_size,
                self.config.aux_weight,
            )
        history = train_loop(
            model,
            train_examples,
            val_examples,
            device=self.config.device,
            epochs=self.config.train_epochs,
            batch_size=self.config.train_batch_size,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            aux_weight=self.config.aux_weight,
            patience=self.config.train_patience,
        )
        self.last_training_stats = {
            "examples": len(examples),
            "train_examples": len(train_examples),
            "validation_examples": len(val_examples),
            "train_games": len({example.game_key for example in train_examples}),
            "validation_games": len(
                {example.game_key for example in val_examples}
            ),
            "newest_iteration": newest_iteration,
            "pretrain_newest_metrics": pretrain_metrics,
            "epochs": history,
        }
        candidate = self.checkpoint_dir / f"candidate_{iteration:04d}.pt"
        checkpoint = make_checkpoint(
            model,
            {
                "model": "transformer",
                "d_model": self.config.d_model,
                "layers": self.config.layers,
                "iteration": iteration,
                "history": history,
                "baselines": target_baselines,
                "phase_d_split": self.last_training_stats,
            },
        )
        torch.save(checkpoint, candidate)
        return candidate

    def _model_agent_spec(self, path: str | Path, role: str) -> ModelAgentSpec:
        model = self.load_model(path)
        source = getattr(model, "_orig_mod", model)
        return ModelAgentSpec(
            name=self.checkpoint_agent_name(path, role),
            model_state={
                key: value.cpu() for key, value in source.state_dict().items()
            },
            d_model=self.config.d_model,
            layers=self.config.layers,
            sims=self.config.gate_sims,
            mode=self.config.search_mode,
            top_k=self.config.top_k,
        )

    def _gate_job(self, index: int, seed_offset: int) -> GameJob:
        return GameJob(
            index=index,
            seed=self.config.seed + seed_offset + index // 2,
            kind="gate",
            payload={
                "first_player": (index // 2) % 2,
                "candidate_is_zero": index % 2 == 0,
            },
        )

    def _play_gate_waves(
        self,
        candidate_spec: GateAgentSpec,
        opponent_spec: GateAgentSpec,
        test: SPRT,
        seed_offset: int,
    ) -> list:
        """Speculative parallel SPRT: identical decision, ledger, and game
        count to sequential play.

        Game outcomes depend only on their seeds, never on the SPRT state, so
        waves of whole seed-pairs run in parallel and their outcomes feed the
        test in index order. The first paired boundary crossing truncates the
        record exactly where the sequential loop would have stopped; games
        already played past it are discarded, costing at most one wave of
        wasted compute and zero statistical difference.
        """

        outcomes = []
        workers = self.config.process_workers
        wave_games = 2 * workers
        index = 0
        while index < self.config.gate_max_games:
            count = min(wave_games, self.config.gate_max_games - index)
            jobs = [
                self._gate_job(index + offset, seed_offset)
                for offset in range(count)
            ]
            wave = run_jobs_in_processes(
                jobs,
                _process_gate_game,
                workers=min(workers, count),
                initializer=_process_gate_init,
                initargs=(candidate_spec, opponent_spec, self.config.inference_batch),
            )
            for offset, outcome in enumerate(wave):
                game_index = index + offset
                outcomes.append(outcome)
                result = test.update(
                    outcome.score_for(0 if game_index % 2 == 0 else 1)
                )
                if game_index % 2 == 1 and result.decision != "continue":
                    return outcomes
            index += count
        return outcomes

    def _rust_model_gate_waves(
        self,
        candidate_spec: ModelAgentSpec,
        opponent_spec: ModelAgentSpec,
        test: SPRT,
        seed_offset: int,
    ) -> list[MatchOutcome]:
        """Run paired model-vs-model SPRT games in the Rust F4 scheduler."""

        import seven_wonders_rust as swr

        def evaluator(spec: ModelAgentSpec) -> Evaluator:
            model = build_model("transformer", spec.d_model, spec.layers)
            model.load_state_dict(spec.model_state)
            return Evaluator(
                model,
                self.config.device,
                self.config.rust_global_batch_cap,
            )

        candidate_eval = evaluator(candidate_spec)
        opponent_eval = evaluator(opponent_spec)
        adapters = (
            rust_seat_routed_flat_batch_adapter(
                (candidate_eval, opponent_eval)
            ),
            rust_seat_routed_flat_batch_adapter(
                (opponent_eval, candidate_eval)
            ),
        )
        outcomes: list[MatchOutcome] = []
        maximum_pairs = self.config.gate_max_games // 2
        for start in range(0, maximum_pairs, self.config.rust_slots):
            pair_indices = list(
                range(start, min(start + self.config.rust_slots, maximum_pairs))
            )
            seeds = [self.config.seed + seed_offset + pair for pair in pair_indices]
            first_players = [pair % 2 for pair in pair_indices]
            leg_records = []
            for adapter in adapters:
                records, _ = swr.self_play_many_flat_net(
                    adapter=adapter,
                    games=rust_games_for_self_play(seeds, first_players),
                    game_seeds=seeds,
                    global_batch_cap=self.config.rust_global_batch_cap,
                    leaf_batch=1,
                    cheap_sims_min=self.config.gate_sims,
                    cheap_sims_max=self.config.gate_sims,
                    full_sims_min=self.config.gate_sims,
                    full_sims_max=self.config.gate_sims,
                    full_search_fraction=0.0,
                    top_k=self.config.top_k,
                    draft_prior=0.0,
                    iteration=-1,
                    force=self.config.force_root_chance,
                    age_deal_samples=self.config.age_deal_samples,
                    max_inflight_batches=self.config.rust_max_inflight_batches,
                    scheduler_workers=self.config.rust_scheduler_workers,
                    deterministic_actions=True,
                )
                leg_records.append(records)
            for offset, (pair, seed, first_player) in enumerate(
                zip(pair_indices, seeds, first_players)
            ):
                for leg, candidate_seat in enumerate((0, 1)):
                    record = leg_records[leg][offset]
                    agents = (
                        (candidate_spec.name, opponent_spec.name)
                        if candidate_seat == 0
                        else (opponent_spec.name, candidate_spec.name)
                    )
                    scores = record["scores"]
                    outcome = MatchOutcome(
                        seed=seed,
                        first_player=first_player,
                        agents=agents,
                        winner=record["winner"],
                        scores=tuple(scores) if scores is not None else None,
                        victory_type=record["victory_type"] or "unknown",
                        actions=len(record["moves"]),
                    )
                    outcomes.append(outcome)
                    result = test.update(outcome.score_for(candidate_seat))
                    game_index = pair * 2 + leg
                    if game_index % 2 == 1 and result.decision != "continue":
                        return outcomes
        return outcomes

    def _rust_bot_gate_waves(
        self,
        candidate_spec: ModelAgentSpec,
        opponent_spec: BotAgentSpec,
        test: SPRT,
        seed_offset: int,
    ) -> list[MatchOutcome]:
        """Run model-vs-bot anchor games wholly in the Rust game loop."""

        import seven_wonders_rust as swr

        model = build_model(
            "transformer", candidate_spec.d_model, candidate_spec.layers
        )
        model.load_state_dict(candidate_spec.model_state)
        evaluator = Evaluator(
            model, self.config.device, self.config.rust_global_batch_cap
        )
        adapter = rust_flat_batch_adapter(evaluator)
        outcomes: list[MatchOutcome] = []
        maximum_pairs = self.config.gate_max_games // 2
        bot_name = opponent_spec.bot.name
        for start in range(0, maximum_pairs, self.config.rust_slots):
            pair_indices = list(
                range(start, min(start + self.config.rust_slots, maximum_pairs))
            )
            seeds = [self.config.seed + seed_offset + pair for pair in pair_indices]
            first_players = [pair % 2 for pair in pair_indices]
            leg_records = []
            for candidate_seat in (0, 1):
                records, _ = swr.self_play_many_flat_net(
                    adapter=adapter,
                    games=rust_games_for_self_play(seeds, first_players),
                    game_seeds=seeds,
                    global_batch_cap=self.config.rust_global_batch_cap,
                    leaf_batch=1,
                    cheap_sims_min=self.config.gate_sims,
                    cheap_sims_max=self.config.gate_sims,
                    full_sims_min=self.config.gate_sims,
                    full_sims_max=self.config.gate_sims,
                    full_search_fraction=0.0,
                    top_k=self.config.top_k,
                    draft_prior=0.0,
                    iteration=-1,
                    force=self.config.force_root_chance,
                    age_deal_samples=self.config.age_deal_samples,
                    max_inflight_batches=self.config.rust_max_inflight_batches,
                    scheduler_workers=self.config.rust_scheduler_workers,
                    deterministic_actions=True,
                    bot_p0=bot_name if candidate_seat == 1 else None,
                    bot_p1=bot_name if candidate_seat == 0 else None,
                    bot_exploration=0.0,
                    bot_policy_iterations=0,
                )
                leg_records.append(records)
            for offset, (pair, seed, first_player) in enumerate(
                zip(pair_indices, seeds, first_players)
            ):
                for leg, candidate_seat in enumerate((0, 1)):
                    record = leg_records[leg][offset]
                    agents = (
                        (candidate_spec.name, bot_name)
                        if candidate_seat == 0
                        else (bot_name, candidate_spec.name)
                    )
                    scores = record["scores"]
                    outcome = MatchOutcome(
                        seed=seed,
                        first_player=first_player,
                        agents=agents,
                        winner=record["winner"],
                        scores=tuple(scores) if scores is not None else None,
                        victory_type=record["victory_type"] or "unknown",
                        actions=len(record["moves"]),
                    )
                    outcomes.append(outcome)
                    result = test.update(outcome.score_for(candidate_seat))
                    game_index = pair * 2 + leg
                    if game_index % 2 == 1 and result.decision != "continue":
                        return outcomes
        return outcomes

    def _sprt_match(
        self,
        candidate_spec: GateAgentSpec,
        opponent_spec: GateAgentSpec,
        *,
        threshold: float,
        seed_offset: int,
    ) -> tuple[GateResult, list]:
        delta = self.config.gate_indifference
        test = SPRT(
            max(0.001, threshold - delta),
            min(0.999, threshold + delta),
            alpha=self.config.gate_alpha,
            beta=self.config.gate_beta,
        )
        if (
            self.config.gate_backend == "rust"
            and isinstance(candidate_spec, ModelAgentSpec)
            and isinstance(opponent_spec, ModelAgentSpec)
        ):
            outcomes = self._rust_model_gate_waves(
                candidate_spec, opponent_spec, test, seed_offset
            )
        elif (
            self.config.gate_backend == "rust"
            and isinstance(candidate_spec, ModelAgentSpec)
            and isinstance(opponent_spec, BotAgentSpec)
        ):
            outcomes = self._rust_bot_gate_waves(
                candidate_spec, opponent_spec, test, seed_offset
            )
        elif self.config.process_workers:
            outcomes = self._play_gate_waves(
                candidate_spec, opponent_spec, test, seed_offset
            )
        else:
            candidate_agent = _build_gate_agent(
                candidate_spec, self.config.device, self.config.inference_batch
            )
            opponent_agent = _build_gate_agent(
                opponent_spec, self.config.device, self.config.inference_batch
            )
            outcomes = []
            for index in range(self.config.gate_max_games):
                candidate_is_zero = index % 2 == 0
                agents = (
                    (candidate_agent, opponent_agent)
                    if candidate_is_zero
                    else (opponent_agent, candidate_agent)
                )
                outcome = play_match(
                    self.adapter,
                    agents,
                    seed=self.config.seed + seed_offset + index // 2,
                    first_player=(index // 2) % 2,
                )
                outcomes.append(outcome)
                result = test.update(
                    outcome.score_for(0 if candidate_is_zero else 1)
                )
                # Stop only after the paired seed has put the candidate in both
                # seats.  A one-orientation boundary crossing is seat noise.
                if index % 2 == 1 and result.decision != "continue":
                    break
        result = test.result()
        return (
            GateResult(
                opponent=_spec_name(opponent_spec),
                threshold=threshold,
                decision=result.decision,
                games=result.games,
                score_rate=result.score_rate,
            ),
            outcomes,
        )

    def promotion_gate(
        self, candidate: str | Path, *, opponent: str | Path | None = None
    ) -> GateResult:
        opponent = Path(opponent) if opponent is not None else self.current_best
        report, outcomes = self._sprt_match(
            self._model_agent_spec(candidate, "candidate"),
            self._model_agent_spec(opponent, "best"),
            threshold=0.50,
            seed_offset=50_000_000,
        )
        self.elo.record(outcomes)
        return report

    def anchor_gates(self, checkpoint: str | Path) -> list[GateResult]:
        checkpoint_agent = self._model_agent_spec(checkpoint, "anchor_subject")
        targets = [
            (BotAgentSpec(GreedyBot()), 0.65),
            *[(BotAgentSpec(bot_type()), 0.60) for bot_type in CURRICULUM_BOT_TYPES],
        ]
        reports = []
        all_outcomes = []
        for offset, (opponent, threshold) in enumerate(targets):
            report, outcomes = self._sprt_match(
                checkpoint_agent,
                opponent,
                threshold=threshold,
                seed_offset=51_000_000 + offset * 1_000_000,
            )
            reports.append(report)
            all_outcomes.extend(outcomes)
        self.elo.record(all_outcomes)
        return reports

    def gate(
        self, candidate: str | Path, *, include_anchors: bool = True
    ) -> list[GateResult]:
        """Compatibility/convenience entry point for an explicit full gate.

        The training loop calls the promotion and anchor gates separately so
        anchor failures cannot block the strength ratchet.
        """

        promotion = self.promotion_gate(candidate)
        anchors = self.anchor_gates(candidate) if include_anchors else []
        return [promotion, *anchors]

    def phase_gate(self, checkpoint: str | Path | None = None) -> list[GateResult]:
        """Run the Phase D exit criteria explicitly, independent of promotion."""

        return self.anchor_gates(checkpoint or self.current_best)

    def promote(self, candidate: str | Path, iteration: int) -> None:
        candidate = Path(candidate)
        temporary = self.current_best.with_suffix(".pt.tmp")
        shutil.copy2(candidate, temporary)
        temporary.replace(self.current_best)
        self.hof.add(self.current_best, iteration=iteration, tag="promoted")

    def run_iteration(self, iteration: int) -> dict[str, Any]:
        model = self.load_model(self.current_best)
        generated = self.generate_iteration(model, iteration)
        records = self.training_records(iteration)
        candidate = self.train_candidate(records, iteration)
        promotion_gate = self.promotion_gate(candidate)
        promoted = promotion_gate.decision == "accept"
        payload = json.loads(self.manifest.path.read_text(encoding="utf-8"))
        previous_promotions = sum(
            bool(row.get("promoted")) for row in payload.get("iterations", [])
        )
        if promoted:
            self.promote(candidate, iteration)
        run_anchors = should_run_anchor_gate(
            promoted=promoted,
            previous_promotions=previous_promotions,
            cadence=self.config.anchor_gate_every_promotions,
        )
        anchor_gates = self.anchor_gates(candidate) if run_anchors else []
        phase_gate_passed = bool(anchor_gates) and all(
            gate.decision == "accept" for gate in anchor_gates
        )
        gates = [promotion_gate, *anchor_gates]
        self.manifest.add_checkpoint(candidate, iteration, promoted)
        row = {
            "iteration": iteration,
            "generated_games": len(generated),
            "training_games": len(records),
            "candidate": str(candidate.resolve()),
            "promoted": promoted,
            "promotion_gate": asdict(promotion_gate),
            "anchor_gates": [asdict(gate) for gate in anchor_gates],
            "phase_gate_passed": phase_gate_passed,
            "gates": [asdict(gate) for gate in gates],
            "generated_summary": summarize_records(generated),
            "training_summary": summarize_records(records),
            "generation_performance": self.last_generation_stats,
            "training_performance": self.last_training_stats,
        }
        self.manifest.append_iteration(row)
        self._append_training_log(row)
        return row

    def run(self) -> list[dict[str, Any]]:
        mode = GeneratorMode(self.config.selfplay_generator_mode)
        if mode == GeneratorMode.STRICT_GATE:
            return self._run_strict_gate()
        return self._run_controller(mode)

    def _run_strict_gate(self) -> list[dict[str, Any]]:
        """Legacy Phase D lifecycle: gate every candidate against current_best."""

        self.initialize()
        payload = json.loads(self.manifest.path.read_text(encoding="utf-8"))
        completed = [row["iteration"] for row in payload.get("iterations", [])]
        start = max(completed, default=-1) + 1
        rows: list[dict[str, Any]] = []
        try:
            for iteration in range(start, start + self.config.iterations):
                rows.append(self.run_iteration(iteration))
                self._autosave_replay_buffer(iteration + 1)
            return rows
        finally:
            if self.config.save_buffer:
                try:
                    self._save_replay_buffer()
                except Exception as exc:
                    print(f"WARNING: buffer save failed: {exc}")

    def _run_controller(self, mode: GeneratorMode) -> list[dict[str, Any]]:
        """Soft-gate lifecycle: the shared controller owns latest/best roles."""

        from .training_adapter import SevenWondersDuelLifecycleAdapter

        self.initialize(bootstrap_checkpoint=False)
        controller = RunController(
            adapter=SevenWondersDuelLifecycleAdapter(self),
            store=_PhaseDRunStore(self),
            checkpoint_dir=self.checkpoint_dir,
            config=ControllerConfig(
                mode=mode,
                bootstrap_policy=BootstrapPolicy(self.config.bootstrap_policy),
                promotion_every=self.config.promotion_every,
                revert_reset_after=self.config.revert_reset_after,
                anchor_gate_every_promotions=self.config.anchor_gate_every_promotions,
                buffer_autosave_every=self.config.buffer_autosave_every,
                seed=self.config.seed,
                iterations=self.config.iterations,
            ),
        )
        try:
            return controller.run()
        finally:
            if self.config.save_buffer:
                try:
                    self._save_replay_buffer()
                except Exception as exc:
                    print(f"WARNING: buffer save failed: {exc}")


class _PhaseDRunStore:
    """Adapts the run manifest + training log to the controller's RunStore."""

    def __init__(self, loop: "PhaseDLoop"):
        self.loop = loop

    def append_iteration(self, row: dict[str, Any]) -> None:
        self.loop.manifest.append_iteration(row)
        self.loop._append_training_log(row)

    def iterations(self) -> list[dict[str, Any]]:
        payload = json.loads(self.loop.manifest.path.read_text(encoding="utf-8"))
        return payload.get("iterations", [])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games-per-iteration", type=int, default=500)
    parser.add_argument("--seed-games", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--process-workers",
        type=int,
        default=0,
        help="legacy Python-backend generation/gate processes (0 = threads); "
        "the Rust generation and Rust gate backends do not use this setting",
    )
    parser.add_argument("--inference-batch", type=int, default=64)
    parser.add_argument("--inference-wait-ms", type=float, default=2.0)
    parser.add_argument("--replay-window", type=int, default=20)
    parser.add_argument(
        "--save-buffer",
        default="",
        help="atomically save the final replay games to this JSONL path",
    )
    parser.add_argument(
        "--warm-buffer",
        default="",
        help="load replay games from a prior --save-buffer JSONL export",
    )
    parser.add_argument("--seed-retain-fraction", type=float, default=1.0)
    parser.add_argument("--curriculum-anneal-iterations", type=int, default=10)
    parser.add_argument("--opponent-fraction", type=float, default=0.15)
    parser.add_argument("--bot-policy-iterations", type=int, default=10)
    parser.add_argument("--bot-exploration", type=float, default=0.05)
    parser.add_argument("--draft-prior-iterations", type=int, default=20)
    parser.add_argument("--cheap-sims-min", type=int, default=16)
    parser.add_argument("--cheap-sims-max", type=int, default=24)
    parser.add_argument("--full-sims-min", type=int, default=64)
    parser.add_argument("--full-sims-max", type=int, default=128)
    parser.add_argument("--full-search-fraction", type=float, default=0.25)
    parser.add_argument("--search-mode", choices=("closed", "open"), default="closed")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--train-epochs", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--train-patience", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-games-to-train", type=int, default=2)
    parser.add_argument(
        "--generation-backend", choices=("rust", "python"), default="rust"
    )
    parser.add_argument("--gate-backend", choices=("rust", "python"), default="rust")
    parser.add_argument("--gate-sims", type=int, default=64)
    parser.add_argument("--gate-max-games", type=int, default=400)
    parser.add_argument("--gate-alpha", type=float, default=0.05)
    parser.add_argument("--gate-beta", type=float, default=0.05)
    parser.add_argument("--gate-indifference", type=float, default=0.03)
    parser.add_argument("--rust-slots", type=int, default=16)
    parser.add_argument("--rust-global-batch-cap", type=int, default=256)
    parser.add_argument("--rust-max-inflight-batches", type=int, default=1)
    parser.add_argument("--rust-scheduler-workers", type=int, default=1)
    parser.add_argument("--leaf-batch", type=int, default=1)
    parser.add_argument(
        "--force-root-chance", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--age-deal-samples", type=int, choices=(0, 4, 8, 16, 32), default=32)
    parser.add_argument("--anchor-gate-every-promotions", type=int, default=3)
    parser.add_argument(
        "--selfplay-generator-mode",
        choices=tuple(mode.value for mode in GeneratorMode),
        default="strict_gate",
        help="strict_gate = legacy gate-every-candidate lifecycle; soft_gate = "
        "cumulative rolling learner with promotion protection",
    )
    parser.add_argument(
        "--bootstrap-policy",
        choices=tuple(policy.value for policy in BootstrapPolicy),
        default="gate",
        help="auto_first_trained installs the first trained learner as best "
        "without a strength gate; gate preserves the old behavior",
    )
    parser.add_argument("--promotion-every", type=int, default=1)
    parser.add_argument("--revert-reset-after", type=int, default=0)
    parser.add_argument(
        "--buffer-autosave-every",
        type=int,
        default=0,
        help="atomically re-export --save-buffer every N iterations (0 = on exit "
        "only); a failed autosave warns but never terminates training",
    )
    parser.add_argument(
        "--warm-buffer-max-staleness",
        type=int,
        default=0,
        help="drop warm-buffer games older than N iterations at import "
        "(0 = default to --replay-window)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--run-log",
        default="",
        help="human-readable transcript path (default <run-dir>/run.log)",
    )
    parser.add_argument(
        "--no-run-log",
        action="store_true",
        help="disable the human-readable transcript; JSONL/manifest are unaffected",
    )
    parser.add_argument(
        "--plumbing-smoke",
        action="store_true",
        help="use tiny generation/training/gate budgets; verifies plumbing only",
    )
    args = parser.parse_args(argv)
    config = PhaseDConfig(
        run_dir=args.run_dir,
        seed=args.seed,
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        seed_games=args.seed_games,
        workers=args.workers,
        process_workers=args.process_workers,
        inference_batch=args.inference_batch,
        inference_wait_ms=args.inference_wait_ms,
        replay_window=args.replay_window,
        save_buffer=args.save_buffer,
        warm_buffer=args.warm_buffer,
        seed_retain_fraction=args.seed_retain_fraction,
        curriculum_anneal_iterations=args.curriculum_anneal_iterations,
        opponent_fraction=args.opponent_fraction,
        bot_policy_iterations=args.bot_policy_iterations,
        bot_exploration=args.bot_exploration,
        draft_prior_iterations=args.draft_prior_iterations,
        cheap_sims_min=args.cheap_sims_min,
        cheap_sims_max=args.cheap_sims_max,
        full_sims_min=args.full_sims_min,
        full_sims_max=args.full_sims_max,
        full_search_fraction=args.full_search_fraction,
        search_mode=args.search_mode,
        top_k=args.top_k,
        d_model=args.d_model,
        layers=args.layers,
        train_epochs=args.train_epochs,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        aux_weight=args.aux_weight,
        train_patience=args.train_patience,
        val_fraction=args.val_fraction,
        min_games_to_train=args.min_games_to_train,
        generation_backend=args.generation_backend,
        gate_backend=args.gate_backend,
        gate_sims=args.gate_sims,
        gate_max_games=args.gate_max_games,
        gate_alpha=args.gate_alpha,
        gate_beta=args.gate_beta,
        gate_indifference=args.gate_indifference,
        rust_slots=args.rust_slots,
        rust_global_batch_cap=args.rust_global_batch_cap,
        rust_max_inflight_batches=args.rust_max_inflight_batches,
        rust_scheduler_workers=args.rust_scheduler_workers,
        leaf_batch=args.leaf_batch,
        force_root_chance=args.force_root_chance,
        age_deal_samples=args.age_deal_samples,
        anchor_gate_every_promotions=args.anchor_gate_every_promotions,
        selfplay_generator_mode=args.selfplay_generator_mode,
        bootstrap_policy=args.bootstrap_policy,
        promotion_every=args.promotion_every,
        revert_reset_after=args.revert_reset_after,
        buffer_autosave_every=args.buffer_autosave_every,
        warm_buffer_max_staleness=args.warm_buffer_max_staleness,
        device=args.device,
    )
    if args.plumbing_smoke:
        config = replace(
            config,
            games_per_iteration=2,
            seed_games=8,
            workers=2,
            d_model=32,
            layers=1,
            cheap_sims_min=1,
            cheap_sims_max=1,
            full_sims_min=1,
            full_sims_max=1,
            full_search_fraction=1.0,
            train_epochs=1,
            train_batch_size=64,
            gate_sims=1,
            gate_max_games=2,
            anchor_gate_every_promotions=1,
        )
    run_log_path = args.run_log or str(Path(config.run_dir) / "run.log")
    header = {
        "Run directory": Path(config.run_dir).resolve(),
        "Command": " ".join(sys.argv),
        "Resume iteration": _resume_iteration_label(config.run_dir),
        "Generator mode": config.selfplay_generator_mode,
        "Structured log": (Path(config.run_dir) / "training_log.jsonl").resolve(),
        "Manifest": (Path(config.run_dir) / "run_manifest.json").resolve(),
    }
    with RunLog(run_log_path, enabled=not args.no_run_log, header=header) as run_log:
        loop = PhaseDLoop(config)
        rows = loop.run()
        latest_ckpt = loop.checkpoint_dir / "latest.pt"
        run_log.completion_fields = {
            "Completed iterations": len(rows),
            "Latest checkpoint": (
                latest_ckpt if latest_ckpt.exists() else loop.current_best
            ),
            "Current best": loop.current_best,
            "Final buffer": config.save_buffer or "disabled",
        }
        output: Any = rows
        if args.plumbing_smoke:
            output = {
                "iterations": rows,
                "explicit_phase_gate": [
                    asdict(result) for result in loop.phase_gate()
                ],
            }
    print(json.dumps(output, indent=2))
    return 0


def _resume_iteration_label(run_dir: str | Path) -> str:
    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.exists():
        return "new run"
    try:
        prior = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "iterations", []
        )
    except (json.JSONDecodeError, OSError):
        return "new run"
    if not prior:
        return "new run"
    return str(max(int(row["iteration"]) for row in prior) + 1)


if __name__ == "__main__":
    raise SystemExit(main())
