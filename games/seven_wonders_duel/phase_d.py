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
import time
from typing import Any, Sequence

import torch

from games.az_loop import (
    EloLedger,
    GameJob,
    HallOfFame,
    LinearSchedule,
    ReplayWindow,
    RunManifest,
    SPRT,
    play_match,
    run_jobs,
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
    inference_batch: int = 64
    inference_wait_ms: float = 2.0
    iterations: int = 1
    games_per_iteration: int = 500
    seed_games: int = 5_000
    replay_window: int = 20
    seed_retain_fraction: float = 1.0
    curriculum_anneal_iterations: int = 10
    opponent_fraction: float = 0.15
    bot_policy_iterations: int = 10
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
    aux_weight: float = 0.2
    val_fraction: float = 0.1
    min_games_to_train: int = 2
    gate_sims: int = 64
    gate_max_games: int = 400
    gate_alpha: float = 0.05
    gate_beta: float = 0.05
    gate_indifference: float = 0.03
    anchor_gate_every_promotions: int = 3

    def validate(self) -> None:
        if self.workers <= 0 or self.games_per_iteration <= 0:
            raise ValueError("workers and games_per_iteration must be positive")
        if self.seed_games < 0 or self.replay_window <= 0:
            raise ValueError(
                "seed_games must be non-negative and replay_window positive"
            )
        for name in (
            "seed_retain_fraction",
            "opponent_fraction",
            "full_search_fraction",
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
    records = run_jobs(jobs, _bot_seed_game, workers=workers)
    _write_records(destination, records)
    return records


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
        self.last_generation_stats: dict[str, Any] = {}
        self.last_training_stats: dict[str, Any] = {}

    def _new_model(self):
        return build_model("transformer", self.config.d_model, self.config.layers)

    def initialize(self) -> None:
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
        if not self.current_best.exists():
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
        if self.config.seed_games:
            generate_seed_buffer(
                self.buffer_dir / "curriculum_seed.jsonl",
                games=self.config.seed_games,
                seed=self.config.seed,
                workers=self.config.workers,
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
                "inference_batches": service.batches,
                "inference_positions": service.positions,
                "mean_inference_batch": (
                    service.positions / service.batches if service.batches else 0.0
                ),
            }
        _write_records(destination, records)
        return records

    def training_records(self, iteration: int) -> list[GameRecord]:
        window = ReplayWindow(self.config.replay_window)
        paths = window.paths(self.buffer_dir, iteration)
        live = [record for path in paths for record in read_records(path)]
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

    def train_candidate(self, records: list[GameRecord], iteration: int) -> Path:
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
        model = self.load_model(self.current_best)
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
            aux_weight=self.config.aux_weight,
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

    def _sprt_match(
        self,
        candidate_agent,
        opponent_agent,
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
                opponent=opponent_agent.name,
                threshold=threshold,
                decision=result.decision,
                games=result.games,
                score_rate=result.score_rate,
            ),
            outcomes,
        )

    def promotion_gate(self, candidate: str | Path) -> GateResult:
        candidate_eval = Evaluator(
            self.load_model(candidate), self.config.device, self.config.inference_batch
        )
        best_eval = Evaluator(
            self.load_model(self.current_best),
            self.config.device,
            self.config.inference_batch,
        )
        candidate_agent = SearchAgent(
            self.checkpoint_agent_name(candidate, "candidate"),
            candidate_eval,
            sims=self.config.gate_sims,
            mode=self.config.search_mode,
            top_k=self.config.top_k,
        )
        opponent = SearchAgent(
            self.checkpoint_agent_name(self.current_best, "best"),
            best_eval,
            sims=self.config.gate_sims,
            mode=self.config.search_mode,
            top_k=self.config.top_k,
        )
        report, outcomes = self._sprt_match(
            candidate_agent,
            opponent,
            threshold=0.50,
            seed_offset=50_000_000,
        )
        self.elo.record(outcomes)
        return report

    def anchor_gates(self, checkpoint: str | Path) -> list[GateResult]:
        checkpoint_eval = Evaluator(
            self.load_model(checkpoint),
            self.config.device,
            self.config.inference_batch,
        )
        checkpoint_agent = SearchAgent(
            self.checkpoint_agent_name(checkpoint, "anchor_subject"),
            checkpoint_eval,
            sims=self.config.gate_sims,
            mode=self.config.search_mode,
            top_k=self.config.top_k,
        )
        targets = [
            (BotAgent(GreedyBot()), 0.65),
            *[(BotAgent(bot_type()), 0.60) for bot_type in CURRICULUM_BOT_TYPES],
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
        return row

    def run(self) -> list[dict[str, Any]]:
        self.initialize()
        payload = json.loads(self.manifest.path.read_text(encoding="utf-8"))
        completed = [row["iteration"] for row in payload.get("iterations", [])]
        start = max(completed, default=-1) + 1
        return [
            self.run_iteration(iteration)
            for iteration in range(start, start + self.config.iterations)
        ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games-per-iteration", type=int, default=500)
    parser.add_argument("--seed-games", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--anchor-gate-every-promotions", type=int, default=3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--plumbing-smoke",
        action="store_true",
        help="use tiny generation/training/gate budgets; verifies plumbing only",
    )
    args = parser.parse_args(argv)
    config = PhaseDConfig(
        run_dir=args.run_dir,
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        seed_games=args.seed_games,
        workers=args.workers,
        anchor_gate_every_promotions=args.anchor_gate_every_promotions,
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
    loop = PhaseDLoop(config)
    rows = loop.run()
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


if __name__ == "__main__":
    raise SystemExit(main())
