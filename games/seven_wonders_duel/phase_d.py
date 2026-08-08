"""Phase D toy-scale AlphaZero loop infrastructure.

This module assembles deterministic self-play workers, request-coalesced neural
inference, replay windows, curriculum seeding/mixing, candidate training, SPRT
gates, promotion, HOF, Elo, and run manifests.  Defaults describe the intended
toy run; tests use deliberately tiny configurations and do not launch training.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import gc
import hashlib
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Callable, Sequence
import warnings

import psutil
import torch

from games.az_loop import (
    BootstrapPolicy,
    ControllerConfig,
    EloLedger,
    GameJob,
    GameSchedule,
    AnchorMeasurement,
    GateLadder,
    InterventionLadder,
    LadderState,
    StagnationDetector,
    GamesLedger,
    GeneratorMode,
    GrowingReplayWindow,
    HallOfFame,
    LinearSchedule,
    MatchOutcome,
    ReplayWindow,
    ResourceMonitor,
    RunController,
    RunLog,
    RunManifest,
    SPRT,
    WindowSelection,
    atomic_copy,
    expected_games_to_decide,
    play_match,
    run_jobs,
    run_jobs_in_processes,
    wilson_interval,
)

from .bots import (
    GreedyBot,
    MilitaryAggressiveBot,
    MilitaryEconomyBot,
    ScienceAggressiveBot,
    ScienceEconomyBot,
)
from .buffer import (
    LEGACY_DIGEST_VERSION,
    GameRecord,
    GameRecorder,
    check_target_versions,
    read_records,
    resolve_opponent_type,
    to_json_line,
)
from .codec import decode_action, encode_action
from .dataset import (
    Example,
    GameDerivationStats,
    derive_records_rust,
    examples_from_record,
    examples_from_records,
    is_fast_search_move,
)
from .game import Phase
from .inference import Evaluator
from .loop_adapter import SevenWondersDuelLoopAdapter
from .loop_inference import CoalescingEvaluator
from .rust_bridge import (
    phase_d_records_from_rust,
    rust_flat_batch_adapter,
    rust_games_for_self_play,
    rust_searcher_routed_flat_batch_adapter,
    rust_seat_routed_flat_batch_adapter,
)
from .search import GumbelMCTS, SearchConfig, SearchResult, state_actor
from .train import (
    baselines,
    build_model,
    evaluate as evaluate_model,
    heads_from_config,
    load_checkpoint,
    make_checkpoint,
    stable_game_split,
    train_steps,
)
from .net import LEGACY_HEADS


LEGACY_EXAMPLE_BYTES = 17_800
MEASURED_ARRAY_BYTES = 13_100
DEFAULT_CACHE_CALIBRATION_FACTOR = LEGACY_EXAMPLE_BYTES / MEASURED_ARRAY_BYTES
EXAMPLE_ARRAY_FIELDS = (
    "type_ids",
    "entity_ids",
    "aux_ids",
    "features",
    "legal",
    "policy_target",
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
    # -1 = fit the anneal to the run (half its length).  A fixed duration that
    # outlives the run never completes: run 02 used 20 over 12 iterations and
    # still carried 450 of the 1000 bot seed games at the end.  Under a fixed
    # step budget that residue is a permanent share of every minibatch, so the
    # anneal has to finish inside the run it is configured for.
    curriculum_anneal_iterations: int = -1
    opponent_fraction: float = 0.15
    bot_policy_iterations: int = 10
    bot_exploration: float = 0.05
    draft_prior_iterations: int = 20

    schedule_basis: str = "games"
    """Clock every training schedule reads: ``games`` or ``iterations``.

    ``games`` is the default and the one the cloud plan locks (W1.2).  An
    iteration is not a unit of anything: run 03 used 400 games/iteration, and a
    cloud run at 500-800 would silently rescale every iteration-keyed schedule --
    the curriculum would anneal out in half the games, the draft prior likewise.
    Keying on cumulative games makes a configuration mean the same thing at any
    ``games_per_iteration``.

    ``iterations`` preserves the pre-2026-07-29 behaviour exactly, for resuming
    runs whose manifests are expressed in it.  Which basis a run uses is part of
    its identity: changing it across a resume is refused, because it silently
    moves every schedule position.
    """

    curriculum_anneal_games: int = 10_000
    """Games over which the curriculum-bot mix anneals to zero (``games`` basis).

    10,000 is measured, not chosen: the net's win rate against the curriculum
    bots passed ~95% by iteration 20 of run 03 at 400 games/iteration (A6), so
    beyond that point the bots are supplying decided games rather than
    instruction.  Counted in loop-generated games, excluding the seed corpus --
    see ``GamesLedger``.
    """

    draft_prior_games: int = 10_000
    """Games over which the wonder-draft tier prior anneals out (``games`` basis).

    Held at the curriculum's duration deliberately: the prior exists to supply
    broad draft competence while the bots are still teaching, and both are
    scaffolding that should leave together.
    """

    replay_window_coefficient: float = 16.0
    replay_window_exponent: float = 0.6
    """Growing replay window: ``window = c * total_games ** alpha`` (W1.1).

    Defaults put the window at ~75% of all games early (on-policy, fast
    adaptation), ~4,000 games at 10k total, and ~16,000 at 100k -- sublinear, so
    the newest iteration never becomes a vanishing fraction of each batch.
    ``alpha`` in 0.5-0.8 per the plan; 0.6 sits mid-range.

    These are a starting shape fitted to run 03's staleness-vs-value-accuracy
    curve, which is suggestive rather than conclusive: the window grew
    1,400 -> 8,000 games over iterations 0-20, froze at its iteration cap, and
    value accuracy peaked at iteration 35 and decayed after.
    """

    replay_window_cap_games: int = 20_000
    """Hard ceiling on the growing window, in games.

    **Not an independent choice.** The window is what the example cache holds, so
    this ceiling and ``example_cache_examples`` describe the same memory. At the
    measured 17.8 KB per example and ~20 examples per game, 20,000 games is
    roughly 7 GB of host RSS. W2.3 derives both from one budget; until then, a
    change here that is not matched there is how a run dies at iteration 70.
    """

    hof_opponent_fraction: float = 0.0
    """Share of generation games played against an archived HOF checkpoint.

    Zero remains the compatibility default. The cloud launch value is explicitly
    pinned to 0.15; its realized share is recorded in schema-v2 stats and
    re-validated on the cloud training host.
    """

    hof_sampling_mode: str = "recency"
    """How a HOF opponent is drawn: ``recency``, ``uniform``, or ``latest``.

    ``recency`` weights recent checkpoints more heavily, which keeps the
    opponent pool near the current frontier while still reaching back. Sampling
    is keyed on the run seed and the iteration, so it is deterministic and
    reproduces exactly on resume.
    """

    hof_start_games: int = 10_000
    """Games before league play begins.

    Early self-play checkpoints are weak and nearly identical to each other, so
    a HOF drawn from them costs generation time without adding diversity.
    Kingdomino gates this the same way (`hof_start_iter=50`); expressed in games
    here, per W1.2. Defaults to the curriculum's duration, so the archive starts
    supplying opponents exactly as the bots stop.
    """
    cheap_sims_min: int = 16
    cheap_sims_max: int = 24
    full_sims_min: int = 64
    full_sims_max: int = 128
    full_search_fraction: float = 0.25
    search_mode: str = "closed"
    top_k: int = 16
    d_model: int = 128
    layers: int = 4
    heads: int | None = None
    """Attention heads; ``None`` derives 64 dimensions per head from d_model.

    Explicit only to hold a head count fixed while sweeping width. Whatever it
    resolves to is written into every checkpoint and is what rebuilds use.
    """

    precision: str = "fp32"
    """Model-call precision. ``bf16`` is opt-in; ``fp32`` preserves defaults."""

    train_steps: int = 300
    train_warmup_steps: int = 100
    train_batch_size: int = 512
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    aux_weight: float = 0.2
    value_weight: float = 1.0
    value_bootstrap: float = 0.0
    validate_every: int = 100
    restore_best_val: bool = False
    val_fraction: float = 0.05
    val_split_salt: str = "swd-v1"
    min_games_to_train: int = 2
    eval_search_mode: str = "gumbel"
    """Root selection for EVALUATION games (gate, arena, bot anchors).

    ``puct`` matches the advisor, which runs plain PUCT via ``descend()``.  The
    Gumbel root exists to make a small fixed budget yield an unbiased
    policy-improvement TARGET; evaluation is not building a target, and the
    Gumbel keys perturb which candidates get searched at all.  Self-play is
    unaffected and always uses the Gumbel root.

    Default stays ``gumbel`` because switching changes what every gate number
    means -- results are not comparable across the two.
    """

    record_fast_moves: bool = False
    """Emit training examples for cheap-search moves.

    Off by default, matching KataGo (Wu 2020 §3.1: "Only turns with a full
    search are recorded for training") and Kingdomino. The value target is
    limited by one noisy result per *game*, so extra positions from the same
    game share that label while inflating the buffer and diluting the share
    of each batch that carries a policy target.

    Turning this on restores the pre-2026-07-25 behaviour; it roughly
    quadruples buffer size, so ``--train-steps`` must rise with it to keep
    ``samples_per_new_position`` in range.
    """

    derive_backend: str = "rust"
    """Replay/encoding implementation: production Rust or Python reference."""

    example_cache_examples: int = 250_000
    """Vectorized examples held in memory across iterations (0 disables).

    `examples_from_records` replays every game through the verified engine path
    -- mask hashes, actors, chance log, trajectory and final digests -- which is
    what makes a stale or tampered buffer raise instead of silently training on
    regenerated states.  Buffer files are immutable once written, so re-deriving
    the whole replay window every iteration re-verifies data that cannot have
    changed: measured at 223 s over 1,633 games and 404 s over 4,800, against
    11 s of actual training.

    Each game is therefore replayed once per process and its examples kept.  The
    compatibility count converts at the measured 17.8 KB per example (~4.45 GB
    at the default), then a startup-calibrated retained-byte estimate enforces
    the bound while evicting least-recently-used games. Size it above
    ``replay_window * games_per_iteration * 20 + seed_games * 60`` or the window
    itself will thrash.
    """

    example_cache_bytes: int = 0
    """Preferred cache ceiling in bytes; 0 converts the legacy count above."""

    memory_budget_gb: float = 0.0
    """Host-RSS budget; 0 resolves to 85% of physical RAM at startup."""

    vram_budget_gb: float = 0.0
    """Device-memory budget; 0 resolves to 90% of detected physical VRAM."""

    memory_headroom_gb: float = 2.0
    """Host memory deliberately reserved outside the process budget."""

    example_cache_floor_bytes: int = 0
    """Minimum retained cache after pressure eviction (0 permits a full drop)."""

    min_buffer_positions: int = 0
    """Skip training until the replay buffer holds this many positions.

    A fixed step budget hammers whatever is in the buffer, so a thin early
    buffer gets trained on hard before any diversity exists: at
    ``--train-steps 300 --train-batch-size 512`` the first iteration presents
    153,600 samples, which against one iteration of self-play is ~7x over a
    single-policy dataset.  The bot seed curriculum currently masks this by
    prefilling the buffer; with ``--seed-games 0`` it does not.

    Counted in positions, not games, so it self-adjusts to
    ``--games-per-iteration``.  ``0`` disables the warmup.
    """
    gate_sims: int = 64
    gate_max_games: int = 400
    gate_alpha: float = 0.05
    gate_beta: float = 0.05
    gate_indifference: float = 0.03
    promotion_min_lcb: float = 0.50
    """Promote when the pair-level Wilson **lower** bound clears this (W5.5)."""

    revert_max_ucb: float = 0.48
    """Revert when the pair-level Wilson **upper** bound falls below this.

    A confidence bound, not a point estimate: `rate < 0.48` reverts an evenly
    matched candidate 32% of the time at 200 games and 35% at 100, which is
    noise, not evidence. The threshold sits below `promotion_min_lcb` on
    purpose -- a fixed threshold gets *more* sensitive as the ladder raises the
    game count, and mild regression while training is normal, so the recoverable
    direction is deliberately the less trigger-happy one.
    """

    gate_confidence_z: float = 1.96
    gate_slots: int = 48
    gate_ladder_games: tuple[int, ...] = ()
    """Gate-size rungs (W5.8); empty keeps one rung at ``gate_max_games``."""

    gate_ladder_step_up_after: int = 2
    gate_ladder_floor_games: int = 0
    self_anchor_games: int = 0
    """Fixed-N games for the W7a games-indexed self-anchor; 0 disables it."""

    self_anchor_lag_games: int = 20_000
    """How far back the anchor sits. The anchor catches up when learning stops,
    which is what makes 0.500 the null."""

    self_anchor_every_games: int = 10_000
    """Games between anchor measurements."""

    intervention_ladder: bool = False
    """W7b. Present but disabled: a running process cannot pick up a mid-run
    change, and restarting to get one is what W6.5 refuses."""

    intervention_window_games: int = 20_000
    """Games a rung is held before its effect is judged."""

    allow_resume_code_drift: bool = False
    allow_hof_change: bool = False
    """Permit a recorded HOF regime change on a resume (W1.5).

    All three HOF fields create a forward regime boundary, and
    ``hof_start_games`` is positional. Recording the change does not make the
    iterations across it comparable; consumers must segment metrics at the
    boundary. The change and its games clock are persisted so that segmentation
    is possible, and the clock becomes a revert-suppress knot.
    """
    """Permit a resume on a different commit or dirty tree (W6.5)."""

    gate_revert_suppress_knots: tuple[int, ...] = ()
    """Extra games-clock points after which one gate may not revert (W5.9).

    Schedule-driven knots are derived from the schedules themselves; this is for
    a disruption the config cannot see, such as an LR change made on resume.
    """

    anchor_gate_every_promotions: int = 3
    anchor_games: int = 200
    # Iteration-cadence out-of-distribution anchor.  The promotion-keyed
    # cadence above never fired in run 02 because nothing was ever promoted,
    # so the bot suite -- the only opponent set outside the self-play
    # distribution -- went unmeasured for the whole run.
    anchor_every_iterations: int = 0
    selfplay_generator_mode: str = "strict_gate"
    bootstrap_policy: str = "gate"
    init_checkpoint: str = ""
    promotion_every: int = 4
    revert_reset_after: int = 0
    probation_reset_after: int = 0
    buffer_autosave_every: int = 0
    warm_buffer_max_staleness: int = 0
    allow_stale_targets: bool = False
    """Load a warm buffer whose targets predate the current definition.

    Off by default: the mix is silent corruption, so it has to be chosen."""
    generation_backend: str = "rust"
    gate_backend: str = "rust"
    rust_slots: int = 16
    rust_global_batch_cap: int = 256
    gate_global_batch_cap: int = 0
    """Gate-path batch cap; 0 follows ``rust_global_batch_cap``.

    The cap's *sign* depends on the slot count it runs at, so generation and the
    gate cannot share one value once their slot counts differ.  Measured on the
    laptop 3070 at 64 sims (``w5_gate_slots_sweep``): at 48 slots raising the cap
    from 256 to 1024 costs ~4%, because the scheduler waits on batches that will
    never fill; at 144 slots the same change gains ~12%.  Generation is pinned at
    48 slots by the throughput programme and is ~85% of an iteration, so a cap
    chosen for a wide gate must not reach it.
    """

    rust_max_inflight_batches: int = 1
    rust_scheduler_workers: int = 1
    leaf_batch: int = 1
    force_root_chance: bool = True
    age_deal_samples: int = 32
    cheap_double_reveal_offsets: int = 0
    """Balanced double-reveal support on CHEAP generation moves only.

    Requires ``generation_backend='rust'``: the Python generator does not force
    expand root chance, so a positive value there would silently do nothing.

    Zero keeps forced expansion exhaustive, which is what every run so far
    produced. A positive X trades exact root chance on a pure double card-reveal
    edge for the balanced ``n * X`` subset -- those edges are 54.5% of all forced
    children (CHANCE_ENUMERATION_PLAN.md). Full-search moves and the arena/gate
    always stay exhaustive, so training targets and gate results are unaffected
    by the setting."""

    def anneal_iterations(self) -> int:
        """Curriculum anneal duration with the ``-1`` auto sentinel resolved."""

        return resolve_anneal_iterations(
            self.curriculum_anneal_iterations, self.iterations
        )

    def uses_games_basis(self) -> bool:
        return self.schedule_basis == "games"

    def curriculum_schedule(self) -> GameSchedule:
        """Curriculum-bot mix against the games clock."""

        return GameSchedule(self.opponent_fraction, 0.0, self.curriculum_anneal_games)

    def seed_retain_schedule(self) -> GameSchedule:
        """Share of the bot seed corpus retained, against the games clock.

        Shares the curriculum's duration: the seed corpus and the live bot mix
        are the same scaffolding, and retaining seed games after the bots have
        annealed out would leave a permanent share of every minibatch coming
        from opponents the net beats 95%+ of the time.
        """

        return GameSchedule(
            self.seed_retain_fraction, 0.0, self.curriculum_anneal_games
        )

    def draft_prior_schedule(self) -> GameSchedule:
        """Wonder-draft tier prior against the games clock."""

        return GameSchedule(1.0, 0.0, self.draft_prior_games)

    def growing_window(self) -> GrowingReplayWindow:
        """The games-keyed replay window.

        ``floor_games`` is one iteration's worth: below that the window would ask
        for less than the run just generated, which whole-iteration selection
        cannot express anyway.
        """

        return GrowingReplayWindow(
            coefficient=self.replay_window_coefficient,
            exponent=self.replay_window_exponent,
            cap_games=self.replay_window_cap_games,
            floor_games=max(1, self.games_per_iteration),
        )

    def schedule_identity(self) -> dict[str, Any]:
        """The schedule fields a resume must not silently change (W1.4).

        Only the fields for the *active* basis are included, so switching an
        unused legacy knob is not treated as a change to a running schedule --
        but ``schedule_basis`` itself is always here, because changing it moves
        every position at once.
        """

        identity: dict[str, Any] = {"schedule_basis": self.schedule_basis}
        if self.uses_games_basis():
            identity.update(
                curriculum_anneal_games=self.curriculum_anneal_games,
                draft_prior_games=self.draft_prior_games,
                opponent_fraction=self.opponent_fraction,
                seed_retain_fraction=self.seed_retain_fraction,
                replay_window_coefficient=self.replay_window_coefficient,
                replay_window_exponent=self.replay_window_exponent,
                replay_window_cap_games=self.replay_window_cap_games,
            )
            # `games_per_iteration` is deliberately absent. Under the games basis
            # it no longer determines any schedule position, so a resume is free
            # to change it -- that freedom is the entire point of W1.2, and
            # pinning it here would reintroduce the coupling by the back door.
            # (It still nudges the window's floor, which self-corrects within one
            # iteration.) Under the iterations basis it is load-bearing, and
            # `replay_window` below is what pins the resulting window.
        else:
            identity.update(
                curriculum_anneal_iterations=self.curriculum_anneal_iterations,
                draft_prior_iterations=self.draft_prior_iterations,
                opponent_fraction=self.opponent_fraction,
                seed_retain_fraction=self.seed_retain_fraction,
                replay_window=self.replay_window,
            )
        identity.update(
            hof_opponent_fraction=self.hof_opponent_fraction,
            hof_sampling_mode=self.hof_sampling_mode,
            hof_start_games=self.hof_start_games,
        )
        return identity

    def gate_batch_cap(self) -> int:
        """The batch cap every evaluation path runs under.

        One accessor rather than the fallback spelled out at each call site, so
        a new gate path cannot silently pick up generation's cap.
        """

        return self.gate_global_batch_cap or self.rust_global_batch_cap

    def gate_ladder(self) -> GateLadder:
        """Scheduled gate sizes (W5.8), or one rung at ``gate_max_games``."""

        if not self.gate_ladder_games:
            return GateLadder.fixed(self.gate_max_games)
        return GateLadder(
            rungs=tuple(self.gate_ladder_games),
            step_up_after=self.gate_ladder_step_up_after,
            floor_games=self.gate_ladder_floor_games,
        )

    def gate_power(self) -> dict[str, Any]:
        """How many games this gate needs before it can decide anything.

        Run 02 configured 100 games against a 3% indifference region and
        returned ``probation`` on 11 of 11 candidates while spending roughly as
        much wall clock on gate games as on training.  That was not bad luck:
        a candidate at the H1 boundary needs ~368 games here, and one that is
        genuinely equal to the best needs unboundedly many.  Surfacing the
        number at config time makes an underpowered gate a visible choice.
        """

        delta = self.gate_indifference
        needed = expected_games_to_decide(
            max(0.001, 0.50 - delta),
            min(0.999, 0.50 + delta),
            alpha=self.gate_alpha,
            beta=self.gate_beta,
        )
        return {
            "expected_games_to_accept_at_h1": needed,
            "gate_max_games": self.gate_max_games,
            "underpowered": self.gate_max_games < needed,
            "resolvable": self.gate_max_games >= needed,
        }

    def validate(self) -> None:
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if self.schedule_basis not in {"games", "iterations"}:
            raise ValueError("schedule_basis must be games or iterations")
        if self.curriculum_anneal_games < 0 or self.draft_prior_games < 0:
            raise ValueError(
                "curriculum_anneal_games and draft_prior_games must be non-negative"
            )
        if not 0.0 <= self.hof_opponent_fraction <= 1.0:
            raise ValueError("hof_opponent_fraction must lie in [0, 1]")
        if self.hof_start_games < 0:
            raise ValueError("hof_start_games must be non-negative")
        if self.hof_sampling_mode not in {"recency", "uniform", "latest"}:
            raise ValueError(
                "hof_sampling_mode must be recency, uniform, or latest"
            )
        if self.schedule_basis == "games":
            # Constructing it here turns an incoherent window into a config-time
            # error rather than an iteration-30 surprise.
            self.growing_window()
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
        if not 0.0 <= self.promotion_min_lcb <= 1.0:
            raise ValueError("promotion_min_lcb must lie in [0, 1]")
        if not 0.0 <= self.revert_max_ucb <= 1.0:
            raise ValueError("revert_max_ucb must lie in [0, 1]")
        if self.revert_max_ucb > self.promotion_min_lcb:
            # Otherwise a single result could satisfy both bounds and the
            # three-way rule would stop being a partition.
            raise ValueError("revert_max_ucb must not exceed promotion_min_lcb")
        self.gate_ladder().validate()
        if self.gate_confidence_z <= 0 or self.gate_slots <= 0:
            raise ValueError("gate_confidence_z and gate_slots must be positive")
        if self.anchor_gate_every_promotions < 0:
            raise ValueError("anchor_gate_every_promotions must be non-negative")
        if self.anchor_games <= 0 or self.anchor_games % 2:
            raise ValueError("anchor_games must be a positive even number")
        if self.self_anchor_games < 0 or self.self_anchor_games % 2:
            raise ValueError("self_anchor_games must be a non-negative even number")
        if self.self_anchor_lag_games <= 0 or self.self_anchor_every_games <= 0:
            raise ValueError("self-anchor lag and cadence must be positive")
        if self.intervention_window_games <= 0:
            raise ValueError("intervention_window_games must be positive")
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
        if self.probation_reset_after < 0:
            raise ValueError("probation_reset_after must be non-negative")
        if self.buffer_autosave_every < 0:
            raise ValueError("buffer_autosave_every must be non-negative")
        if self.example_cache_examples < 0:
            raise ValueError("example_cache_examples must be non-negative")
        if self.example_cache_bytes < 0 or self.example_cache_floor_bytes < 0:
            raise ValueError("example cache byte bounds must be non-negative")
        if self.memory_budget_gb < 0 or self.vram_budget_gb < 0:
            raise ValueError("memory budgets must be non-negative")
        if self.memory_headroom_gb < 0:
            raise ValueError("memory_headroom_gb must be non-negative")
        if self.warm_buffer_max_staleness < 0:
            raise ValueError("warm_buffer_max_staleness must be non-negative")
        if self.gate_backend not in ("rust", "python"):
            raise ValueError("gate_backend must be rust or python")
        if self.generation_backend not in ("rust", "python"):
            raise ValueError("generation_backend must be rust or python")
        if self.derive_backend not in ("rust", "python"):
            raise ValueError("derive_backend must be rust or python")
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
        if self.gate_global_batch_cap < 0:
            raise ValueError("gate_global_batch_cap must be non-negative")
        if not 0 <= self.age_deal_samples <= 32:
            raise ValueError("age_deal_samples must be in [0, 32]")
        if self.cheap_double_reveal_offsets < 0:
            raise ValueError("cheap_double_reveal_offsets must be non-negative")
        if self.cheap_double_reveal_offsets and self.generation_backend != "rust":
            # The Python generation path builds its SearchConfig in `_search_move`
            # without force expansion at all, so capping there is not merely
            # unimplemented -- it cannot apply. Failing is the only honest
            # option: silently generating uncapped games while the run manifest
            # records a cap is exactly the kind of corruption that is invisible
            # until someone compares two runs months later.
            raise ValueError(
                "cheap_double_reveal_offsets requires generation_backend='rust' "
                "(the Python generator does not force-expand root chance)"
            )
        if self.d_model <= 0 or self.d_model % 4 or self.layers <= 0:
            raise ValueError("d_model must be positive/divisible by 4 and layers positive")
        if self.train_steps <= 0 or self.train_batch_size <= 0:
            raise ValueError("train_steps and batch size must be positive")
        if self.train_warmup_steps < 0:
            raise ValueError("train_warmup_steps must be non-negative")
        if self.validate_every <= 0:
            raise ValueError("validate_every must be positive")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must lie in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")


class UnderpoweredGateWarning(UserWarning):
    """The promotion gate cannot resolve its own indifference region."""


def curriculum_fraction(initial: float, iteration: int, duration: int) -> float:
    return LinearSchedule(initial, 0.0, duration).value(iteration)


def resolve_anneal_iterations(configured: int, iterations: int) -> int:
    """Resolve the ``-1`` auto sentinel to half the planned run length.

    Anything explicitly configured is honoured as-is; the caller warns when an
    explicit duration outlives the run and so leaves seed data in the buffer at
    the final iteration.
    """

    if configured >= 0:
        return configured
    return max(1, iterations // 2)


#: Self-play move-selection temperature schedule: anneals 1.0 -> floor over
#: `_TEMPERATURE_ANNEAL_MOVES` moves, then holds. Set by `--temperature-floor`
#: and `--temperature-anneal-moves`; the defaults reproduce every run before
#: 2026-08-05, when these were hard-coded here and in `self_play.rs`.
#:
#: This is the main diversity lever in self-play. Sampling is proportional to
#: visits ** (1 / temperature), so the historical floor of 0.25 is visits**4 --
#: a converged policy plays its favourite near-deterministically, and with a
#: 20-move anneal that covers roughly 70% of a ~70-move game.
_TEMPERATURE_FLOOR = 0.25
_TEMPERATURE_ANNEAL_MOVES = 20.0


def _configure_cheap_top_k(width: int) -> None:
    """Apply the cheap-move root width to the Rust generator.

    A no-op for the Python backend, which has no separate cheap width; runs that
    want the split must use --generation-backend rust.
    """

    if width < 0:
        raise ValueError("--cheap-top-k must be non-negative")
    try:
        import seven_wonders_rust as swr
    except ImportError:  # pragma: no cover - Python backend needs no bridge
        return
    swr.set_cheap_top_k(int(width))


def set_temperature_schedule(floor: float, anneal_moves: float) -> None:
    """Set the schedule for both generators, which must agree.

    The Rust backend keeps its own copy of these, so setting one without the
    other would make the two paths diverge on move selection while still
    passing every structural equivalence check.
    """

    global _TEMPERATURE_FLOOR, _TEMPERATURE_ANNEAL_MOVES
    if not 0.0 < floor <= 1.0:
        raise ValueError("temperature floor must be in (0, 1]")
    if anneal_moves < 1:
        raise ValueError("temperature anneal moves must be >= 1")
    _TEMPERATURE_FLOOR = float(floor)
    _TEMPERATURE_ANNEAL_MOVES = float(anneal_moves)
    try:
        import seven_wonders_rust as swr
    except ImportError:  # pragma: no cover - Python backend needs no bridge
        return
    swr.set_temperature_schedule(float(floor), float(anneal_moves))


def temperature_for_move(move_index: int) -> float:
    return LinearSchedule(1.0, _TEMPERATURE_FLOOR, _TEMPERATURE_ANNEAL_MOVES).value(
        move_index
    )


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
        result = search.search(state)
        # Not `result.action_index`: that is the Gumbel-perturbed selection,
        # which belongs to self-play exploration.  A gate or arena game plays
        # the improved policy's argmax, built from the same logits without the
        # Gumbel keys.
        return max(result.policy_target, key=result.policy_target.get)


def _tag_league_opponents(
    records: Sequence[GameRecord], league: LeagueAssignment
) -> list[GameRecord]:
    """Name the archived opponent in each league game's ``agents`` block.

    W1.3 requires the opponent identity recorded *per game* so W3 can split
    outcomes by opponent -- "did we beat the archive" is a different question
    from "did we beat ourselves", and pooling them hides both.

    ``agents`` is pure metadata: it feeds neither ``final_digest`` nor
    ``trajectory_digest``, so relabelling here cannot invalidate a record or
    change what replay verifies. Records come back in job order (the Rust
    scheduler guarantees it regardless of completion order), which is what makes
    indexing into the assignment sound.
    """

    if len(records) != len(league.nets_p0):
        raise ValueError(
            f"league assignment covers {len(league.nets_p0)} games but "
            f"{len(records)} records came back"
        )
    tagged: list[GameRecord] = []
    for index, record in enumerate(records):
        seat = league.archive_seat(index)
        if seat is None:
            tagged.append(record)
            continue
        agents = dict(record.agents)
        prior_type = resolve_opponent_type(agents)
        agents["league_assignment"] = league.name
        # Bot routing takes precedence over network routing inside Rust. When
        # the archive was assigned to the bot-controlled seat, network 1 never
        # evaluated a move; preserve the nominal assignment, but report the
        # actual game as current-vs-bot.
        if prior_type == "bot" and agents.get(f"p{seat}") != "network":
            agents["league_assignment_used"] = "false"
            tagged.append(replace(record, agents=agents))
            continue
        agents[f"p{seat}"] = league.name
        agents["kind"] = "league"
        agents["opponent_type"] = (
            "hof_bot" if prior_type == "bot" else "hof"
        )
        agents["opponent_source"] = league.checkpoint
        agents["league_assignment_used"] = "true"
        tagged.append(replace(record, agents=agents))
    return tagged


@lru_cache(maxsize=64)
def _digest_of(path: str, size: int, mtime_ns: int) -> str:
    """sha256 of a checkpoint, memoised on (path, size, mtime).

    Checkpoints are ~4 MB and the anchor compares two of them per measurement.
    Keying on the stat fields rather than the path alone means a file rewritten
    in place -- `latest.pt` every iteration -- re-hashes rather than returning a
    stale digest.
    """

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_digest(path: str | Path) -> str:
    stat = Path(path).stat()
    return _digest_of(str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)


HOF_SCHEDULE_KEYS = frozenset(
    {"hof_opponent_fraction", "hof_sampling_mode", "hof_start_games"}
)
"""Schedule-identity keys a resume may change under ``--allow-hof-change``."""


def _sanitize_non_finite(value: Any, path: str = "") -> tuple[Any, list[str]]:
    """Replace inf/nan with None anywhere in a row, reporting where they were.

    Returns ``(clean, paths)``. Containers are rebuilt only when something below
    them changed, so a clean row comes back unchanged rather than deep-copied --
    these rows carry the whole per-step history and are not small.
    """

    if isinstance(value, float):
        if math.isfinite(value):
            return value, []
        return None, [path or "<root>"]
    if isinstance(value, dict):
        found: list[str] = []
        clean: dict[Any, Any] = {}
        for key, item in value.items():
            cleaned, paths = _sanitize_non_finite(item, f"{path}/{key}")
            clean[key] = cleaned
            found.extend(paths)
        return (clean if found else value), found
    if isinstance(value, (list, tuple)):
        found = []
        items = []
        for index, item in enumerate(value):
            cleaned, paths = _sanitize_non_finite(item, f"{path}[{index}]")
            items.append(cleaned)
            found.extend(paths)
        if not found:
            return value, []
        return (tuple(items) if isinstance(value, tuple) else items), found
    return value, []


def _write_records(path: Path, records: Sequence[GameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            line = to_json_line(record)
            handle.write(line + "\n")
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
        agents={
            "p0": bots[0].name,
            "p1": bots[1].name,
            "kind": "curriculum_seed",
            "opponent_type": "bot",
        },
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
            # One call per group; the Rust pool keeps `rust_slots` games active.
            for chunk in [group_jobs]:
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
                    max_active_slots=rust_slots,
                )
                for raw in raw_records:
                    raw["agents"]["kind"] = "curriculum_seed"
                converted = phase_d_records_from_rust(raw_records, validate=False)
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
    model_state: dict[str, torch.Tensor],
    config: PhaseDConfig,
    iteration: int,
    schedules: ResolvedSchedules,
) -> None:
    # One BLAS thread per process: generation scales by process count, and
    # oversubscribing cores with intra-op threads slows every worker down.
    torch.set_num_threads(1)
    model = build_model("transformer", config.d_model, config.layers, config.heads)
    model.load_state_dict(model_state)
    # CPU inference per process: at generation batch sizes the tiny network is
    # a few ms on a core, while fanning every process into one GPU serializes
    # on the CUDA context. The GPU stays free for training and gates.
    _PROCESS_STATE["evaluator"] = Evaluator(
        model,
        "cpu",
        config.inference_batch,
        precision=config.precision,
    )
    _PROCESS_STATE["config"] = config
    _PROCESS_STATE["iteration"] = iteration
    _PROCESS_STATE["schedules"] = schedules


def _process_self_play_game(job: GameJob) -> GameRecord:
    return _self_play_game(
        job,
        _PROCESS_STATE["evaluator"],
        _PROCESS_STATE["config"],
        _PROCESS_STATE["iteration"],
        _PROCESS_STATE["schedules"],
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
    heads: int = LEGACY_HEADS
    """Head count the weights were trained under; never re-derived.

    Defaulted rather than required so older pickled specs still load, and set to
    the historical value for the same reason a checkpoint without a ``heads``
    key resolves to it.
    """


@dataclass(frozen=True, slots=True)
class BotAgentSpec:
    bot: Any


GateAgentSpec = ModelAgentSpec | BotAgentSpec


def _spec_name(spec: GateAgentSpec) -> str:
    return spec.bot.name if isinstance(spec, BotAgentSpec) else spec.name


def _build_gate_agent(
    spec: GateAgentSpec,
    device: str,
    inference_batch: int,
    precision: str = "fp32",
):
    if isinstance(spec, BotAgentSpec):
        return BotAgent(spec.bot)
    model = build_model("transformer", spec.d_model, spec.layers, spec.heads)
    model.load_state_dict(spec.model_state)
    return SearchAgent(
        spec.name,
        Evaluator(model, device, inference_batch, precision=precision),
        sims=spec.sims,
        mode=spec.mode,
        top_k=spec.top_k,
    )


def _process_gate_init(
    candidate_spec: GateAgentSpec,
    opponent_spec: GateAgentSpec,
    inference_batch: int,
    precision: str,
) -> None:
    torch.set_num_threads(1)
    _PROCESS_STATE["gate_adapter"] = SevenWondersDuelLoopAdapter()
    _PROCESS_STATE["gate_candidate"] = _build_gate_agent(
        candidate_spec, "cpu", inference_batch, precision
    )
    _PROCESS_STATE["gate_opponent"] = _build_gate_agent(
        opponent_spec, "cpu", inference_batch, precision
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


@dataclass(frozen=True, slots=True)
class LeagueAssignment:
    """Which archived opponent this iteration plays, and in which games.

    One opponent per iteration, following Kingdomino: sampling per game would
    make the batch composition depend on how many games happen to draw each
    checkpoint, and it would defeat the point of caching one opponent model on
    the device for the whole iteration.

    ``nets_p0``/``nets_p1`` are the per-game seat assignments handed to Rust:
    network 0 is always the learner, network 1 the archive. A pure self-play
    game is ``(0, 0)``. Seats alternate across the league games so the archive
    plays both sides within a single iteration.
    """

    checkpoint: str
    sha256: str
    iteration_added: int
    nets_p0: tuple[int, ...]
    nets_p1: tuple[int, ...]

    @property
    def games(self) -> int:
        return sum(
            1
            for p0, p1 in zip(self.nets_p0, self.nets_p1)
            if p0 or p1
        )

    @property
    def name(self) -> str:
        """Stable opponent identity for stats and the ``agents`` block.

        Carries the archive's iteration and checkpoint hash, so W3 can group
        outcomes by opponent and two different archives can never collide under
        one name.
        """

        return f"hof_iter_{self.iteration_added:04d}_{self.sha256[:12]}"

    def opponent_for(self, index: int) -> str | None:
        """The opponent in game ``index``, or ``None`` when it is pure self-play."""

        if not (self.nets_p0[index] or self.nets_p1[index]):
            return None
        return self.name

    def archive_seat(self, index: int) -> int | None:
        """Seat the archive occupies in game ``index``, or ``None`` if absent."""

        if self.nets_p0[index]:
            return 0
        if self.nets_p1[index]:
            return 1
        return None


@dataclass(frozen=True, slots=True)
class ResolvedSchedules:
    """Schedule values for one iteration, resolved before generation starts.

    Passed in rather than recomputed per game.  Under the games basis a schedule
    reads the games ledger, which lives on the loop and is neither available in a
    generation worker process nor safe to consult per game -- the clock must be
    the same for every game in an iteration, or games within one iteration would
    see different curriculum mixes.  Resolving once and passing the values makes
    that invariant structural instead of hoped for.
    """

    curriculum_mix_fraction: float
    draft_prior: float


def _search_move(
    game,
    evaluator,
    config: PhaseDConfig,
    schedules: ResolvedSchedules,
    move_index: int,
    rng: random.Random,
) -> tuple[int, SearchResult, bool]:
    full = rng.random() < config.full_search_fraction
    sims = rng.randint(
        config.full_sims_min if full else config.cheap_sims_min,
        config.full_sims_max if full else config.cheap_sims_max,
    )
    draft_amount = schedules.draft_prior
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
    schedules: ResolvedSchedules,
) -> GameRecord:
    rng = random.Random(job.seed ^ 0xC6BC279692B5CC83)
    mixed = rng.random() < schedules.curriculum_mix_fraction
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
        "opponent_type": "bot" if mixed else "current_best",
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
                schedules,
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
    pairs: int = 0
    wilson_lcb: float = 0.0
    wilson_ucb: float = 1.0
    stop_reason: str = ""
    evaluated_games: int = 0
    seconds: float = field(default=0.0, compare=False)
    moves_per_game: float = 0.0
    fixed_n: bool = True
    pair_scores: tuple[float, ...] = ()
    revert_suppressed: bool = False


def wilson_pair_decision(
    pair_scores: Sequence[float],
    *,
    promotion_min_lcb: float = 0.50,
    revert_max_ucb: float = 0.48,
    z: float = 1.96,
    measurement: bool = False,
    measurement_threshold: float = 0.50,
    revert_suppressed: bool = False,
) -> tuple[str, int, float, float, float, str]:
    """Three-way gate decision using independent seat-pairs as observations.

    The whole sample is read **once**, at its fixed size.  Evaluating the bounds
    after every pair and stopping on the first crossing -- which this function
    used to do -- is optional stopping: it promotes an evenly matched candidate
    15-19% of the time instead of ~2%, and no promotion is ever undone anywhere
    in the system (W5.5).

    ``measurement=True`` is the anchor form: report the rate against a fixed
    threshold with no lifecycle meaning.
    """

    if not pair_scores:
        return "continue", 0, 0.0, 0.0, 1.0, "no_pairs"
    points = 0.0
    for score in pair_scores:
        if score not in (0.0, 0.5, 1.0):
            raise ValueError("pair outcomes must be one of 0, 0.5, or 1")
        points += score
    pairs = len(pair_scores)
    rate = points / pairs
    lcb, ucb = wilson_interval(points, pairs, z=z)
    if measurement:
        decision = "accept" if rate >= measurement_threshold else "reject"
        return decision, pairs, rate, lcb, ucb, "fixed_n"
    if lcb > promotion_min_lcb:
        return "accept", pairs, rate, lcb, ucb, "promotion_lcb"
    if ucb < revert_max_ucb:
        if revert_suppressed:
            # W5.9: a schedule knot just moved the ground under the learner.
            # One gate of amnesty, and the evidence is still on the row.
            return "continue", pairs, rate, lcb, ucb, "revert_suppressed_knot"
        return "reject", pairs, rate, lcb, ucb, "revert_ucb"
    return "continue", pairs, rate, lcb, ucb, "probation"


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
        # W7. `base_config` stays exactly as launched: the resume guards compare
        # against it, and an intervention must never look like a config change
        # the operator did not make.
        self.base_config = config
        self.stagnation_path = self.run_dir / "stagnation.json"
        self.detector = StagnationDetector()
        self.ladder = InterventionLadder(
            enabled=config.intervention_ladder,
            measurement_window_games=config.intervention_window_games,
        )
        # The schedule clock. Derived from the buffer files, which are immutable
        # once written, so a resume recomputes it instead of restoring it.
        self.games_ledger = GamesLedger(self.buffer_dir)
        self.elo = EloLedger(
            self.run_dir / "elo", fixed_ratings={GreedyBot.name: 1000.0}
        )
        self.manifest = RunManifest(self.run_dir, Path(__file__).resolve().parents[2])
        self.training_log = self.run_dir / "training_log.jsonl"
        self.warm_records: list[GameRecord] = []
        # Rust cannot reproduce the CPython-RNG component of legacy durable
        # digests. Keep production on Rust, but make that deliberate gap loud
        # once when legacy persisted data first enters this process.
        self._legacy_digest_warning_emitted = False
        # Per-game vectorized examples, keyed by the game's own content digest.
        # Ordered so eviction can be least-recently-used: the games that fall out
        # are the iterations that have left the replay window.
        self._example_cache: OrderedDict[tuple, list[Example]] = OrderedDict()
        self._example_cache_raw_bytes: dict[tuple, int] = {}
        self._example_cache_game_stats: dict[tuple, GameDerivationStats] = {}
        self.cache_calibration_factor = DEFAULT_CACHE_CALIBRATION_FACTOR
        self._cache_calibrated = False
        # Games-clock points where a resume was allowed to move the HOF
        # share; loaded from the manifest so they survive later resumes.
        self._schedule_change_knots: tuple[int, ...] = ()
        self.last_example_cache_stats: dict[str, Any] = {}
        self.last_generation_stats: dict[str, Any] = {}
        self.last_training_stats: dict[str, Any] = {}
        self.last_gate_stats: dict[str, Any] = {}
        # W7c: the anchor reuses this when it resolves to the same match.
        self.last_promotion_gate: GateResult | None = None
        self.last_promotion_gate_iteration: int | None = None
        self.last_promotion_gate_subject_sha256: str | None = None
        self.last_promotion_gate_opponent_sha256: str | None = None
        self.last_warm_stats: dict[str, int] = {}
        self.resource_monitor = ResourceMonitor()
        self.phase_seconds: dict[str, float] = {}
        gib = 1024**3
        total_ram = int(psutil.virtual_memory().total)
        configured_host = (
            int(config.memory_budget_gb * gib)
            if config.memory_budget_gb > 0
            else int(total_ram * 0.85)
        )
        self.memory_budget_bytes = max(
            0, configured_host - int(config.memory_headroom_gb * gib)
        )
        if torch.cuda.is_available() and config.device.startswith("cuda"):
            _free, total_vram = torch.cuda.mem_get_info()
            self.vram_budget_bytes = (
                int(config.vram_budget_gb * gib)
                if config.vram_budget_gb > 0
                else int(total_vram * 0.90)
            )
        else:
            self.vram_budget_bytes = 0

    # -- schedule clock ----------------------------------------------------

    def generation_clock(self, iteration: int) -> int:
        """Games that existed when ``iteration`` began generating.

        The value every generation-time schedule reads.  Deliberately *before*,
        not *through*: a schedule keyed on games through the current iteration
        would depend on the games it is itself deciding how to generate, and
        would differ between a fresh run and a resume of the same iteration.
        """

        return self.games_ledger.total_before(iteration)

    def training_clock(self, iteration: int) -> int:
        """Games available once ``iteration`` has generated.

        The value the replay window reads, because the window is applied after
        generation, when selecting what to train on -- the games just written are
        part of what is available.
        """

        return self.games_ledger.total_through(iteration)

    def curriculum_mix_fraction(self, iteration: int) -> float:
        """Share of generation games that pair the net against a curriculum bot."""

        if self.config.uses_games_basis():
            return self.config.curriculum_schedule().value(
                self.generation_clock(iteration)
            )
        return curriculum_fraction(
            self.config.opponent_fraction, iteration, self.config.anneal_iterations()
        )

    def draft_prior_amount(self, iteration: int) -> float:
        """Weight on the wonder-draft tier prior at this point in the run."""

        if self.config.uses_games_basis():
            return self.config.draft_prior_schedule().value(
                self.generation_clock(iteration)
            )
        return LinearSchedule(1.0, 0.0, self.config.draft_prior_iterations).value(
            iteration
        )

    def seed_retain_fraction(self, iteration: int) -> float:
        """Share of the bot seed corpus still mixed into training."""

        if self.config.uses_games_basis():
            return self.config.seed_retain_schedule().value(
                self.training_clock(iteration)
            )
        return curriculum_fraction(
            self.config.seed_retain_fraction,
            iteration,
            self.config.anneal_iterations(),
        )

    # -- W7 stagnation ------------------------------------------------------

    def stagnation_state(self) -> dict[str, Any]:
        """Anchor history and ladder position, across resumes."""

        if not self.stagnation_path.exists():
            return {"measurements": [], "ladder": LadderState().as_dict()}
        return json.loads(self.stagnation_path.read_text(encoding="utf-8"))

    def _write_stagnation_state(self, state: dict[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.stagnation_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.stagnation_path)

    def apply_intervention(self, intervention) -> None:
        """Rebuild the working config from the pristine one plus one rung (W7b).

        Always from ``base_config``, never from the current config: rungs are
        exclusive, so stacking them would make the second rung's effect
        unattributable, which is the whole reason for the measurement window.
        """

        base = self.base_config
        if intervention is None:
            self.config = base
            return
        sims = intervention.sims_multiplier
        self.config = replace(
            base,
            cheap_sims_min=max(1, int(round(base.cheap_sims_min * sims))),
            cheap_sims_max=max(1, int(round(base.cheap_sims_max * sims))),
            full_sims_min=max(1, int(round(base.full_sims_min * sims))),
            full_sims_max=max(1, int(round(base.full_sims_max * sims))),
            full_search_fraction=min(
                1.0,
                base.full_search_fraction
                * intervention.full_search_fraction_multiplier,
            ),
            replay_window_coefficient=(
                base.replay_window_coefficient * intervention.window_multiplier
            ),
            replay_window_cap_games=int(
                round(base.replay_window_cap_games * intervention.window_multiplier)
            ),
            hof_opponent_fraction=(
                base.hof_opponent_fraction
                if intervention.hof_fraction is None
                else intervention.hof_fraction
            ),
            learning_rate=base.learning_rate * intervention.learning_rate_multiplier,
        )

    def measure_stagnation(self, iteration: int) -> dict[str, Any] | None:
        """Run the anchor if it is due, then update detection and the ladder.

        Returns the W3 ``game_specific`` block for this iteration, or ``None``
        when no measurement was due.
        """

        if self.config.self_anchor_games <= 0:
            return None
        state = self.stagnation_state()
        measurements = [
            AnchorMeasurement.from_dict(row) for row in state["measurements"]
        ]
        now = self.training_clock(iteration)
        last = measurements[-1].games if measurements else None
        if last is not None and now - last < self.config.self_anchor_every_games:
            return None
        result = self.self_anchor_gate(iteration)
        if result is None:
            return None
        report, anchor_games = result
        measurements.append(
            AnchorMeasurement(
                games=now,
                score_rate=report.score_rate,
                lower=report.wilson_lcb,
                upper=report.wilson_ucb,
                anchor_games=anchor_games,
                iteration=iteration,
            )
        )
        verdict = self.detector.verdict(measurements)
        ladder_state = self.ladder.advance(
            LadderState.from_dict(state.get("ladder")), verdict, now
        )
        self.apply_intervention(self.ladder.active(ladder_state))
        self._write_stagnation_state(
            {
                "measurements": [item.as_dict() for item in measurements],
                "ladder": ladder_state.as_dict(),
            }
        )
        active = self.ladder.active(ladder_state)
        return {
            "anchor": {
                "score_rate": report.score_rate,
                "wilson_lcb": report.wilson_lcb,
                "wilson_ucb": report.wilson_ucb,
                "games": report.games,
                "anchor_games": anchor_games,
                "lag_games": now - anchor_games,
                "opponent": report.opponent,
            },
            "verdict": {
                "stagnant": verdict.stagnant,
                "reasons": list(verdict.reasons),
                "slope_per_10k_games": verdict.slope_per_10k_games,
                "measurements": verdict.measurements_considered,
            },
            "ladder": {
                **ladder_state.as_dict(),
                "enabled": self.ladder.enabled,
                "active": None if active is None else active.name,
            },
        }

    def restore_intervention(self) -> None:
        """Re-apply the persisted rung after a resume."""

        if not self.config.intervention_ladder:
            return
        ladder_state = LadderState.from_dict(self.stagnation_state().get("ladder"))
        self.apply_intervention(self.ladder.active(ladder_state))

    def anchor_reference(self, iteration: int) -> tuple[Path, int] | None:
        """The learner's own checkpoint from ``self_anchor_lag_games`` ago.

        W7c. The subject and the reference are both points on the *candidate*
        trajectory -- the thing that trains every iteration -- not on the
        promotion lineage.

        W7a indexed the promotion lineage instead, and run 03 showed what that
        costs.  ``current_best`` froze at iteration 60; once the clock passed
        ``promotion + lag`` the reference resolved to ``current_best`` itself, so
        the measurement was a net against itself and the code returned a
        synthetic 0.500 rather than play it.  That is not "no improvement", it is
        *no measurement*: the anchor went dark for 35 iterations spanning both a
        collapse and a full recovery, and reported the same number throughout.
        An anchor that is undefined exactly when promotions stop cannot answer
        the question it exists for, because promotions stopping is the question.

        ``learner_NNNN.pt`` is copied from ``latest.pt`` after promotion and reset
        effects have been applied. Raw candidates cannot be used here: on a reset
        iteration they contain the rejected weights that were immediately
        replaced and were never the learner in force.

        Returns ``(path, anchor_games)``, or ``None`` when the run is younger
        than the lag -- the documented "skip rather than compare against
        nothing" case -- or when the checkpoint that far back is gone.
        """

        lag = self.config.self_anchor_lag_games
        now = self.training_clock(iteration)
        target = now - lag
        if target <= 0:
            return None
        history: list[tuple[int, Path]] = []
        for known in self.games_ledger.known_iterations():
            if known >= iteration:
                continue
            path = self.learner_checkpoint(known)
            if path.is_file():
                history.append((self.games_ledger.total_through(known), path))
        # Each snapshot was the learner from its own iteration until the next,
        # so the one in force at `target` is the newest at or before it.
        in_force = [item for item in history if item[0] <= target]
        if not in_force:
            return None
        return max(in_force)[1], max(in_force)[0]

    def learner_checkpoint(self, iteration: int) -> Path:
        """Immutable post-transition learner snapshot for one iteration."""

        return self.checkpoint_dir / f"learner_{iteration:04d}.pt"

    def current_best_iteration(self) -> int | None:
        """Iteration ``current_best`` was promoted at, from its own metadata."""

        if not Path(self.current_best).is_file():
            return None
        stored = torch.load(
            self.current_best, map_location="cpu", weights_only=False
        ).get("config", {})
        iteration = stored.get("iteration")
        return None if iteration is None else int(iteration)

    def anchor_subject(self) -> Path:
        """The net the anchor measures: the learner, not the promoted best."""

        return self.checkpoint_dir / "latest.pt"

    def anchor_caught_up(self, path: Path) -> bool:
        """True when the anchor resolved to the subject itself.

        Then the match is a net against itself: with paired seeds, both seats
        swapped, and deterministic gate play, the two games of a pair are the
        same game, so every pair scores exactly 0.5 and the result is 0.500 with
        no variance. Playing it would burn a full gate to rediscover that, so
        the null is reported directly. ``test_a_caught_up_anchor_is_exactly_the
        _null_when_played`` plays it for real and checks this holds.

        Kept after W7c retargeted the anchor to the learner trajectory, where
        it should now be unreachable: the reference is strictly older than the
        current iteration, so it cannot be the subject.  It stays as the cheap
        guard that keeps the degenerate case from ever being *played*, since a
        pruned or relinked checkpoint could still alias the two.
        """

        subject = self.anchor_subject()
        if not (Path(path).is_file() and subject.is_file()):
            return False
        if Path(path).resolve() == subject.resolve():
            return True
        # Distinct files can still hold identical weights -- `current_best.pt` is
        # a copy of some candidate, and a revert-reset makes `latest.pt` a copy
        # of `current_best.pt`. Compare contents, not paths.
        return _file_digest(path) == _file_digest(subject)

    def anchor_duplicates_gate(self, path: Path, iteration: int) -> bool:
        """True when this iteration's promotion gate already played this match.

        The gate plays ``latest`` against ``current_best``; the anchor plays
        ``latest`` against a lagged candidate. When the lagged candidate is the
        net that ``current_best`` was copied from, the two matches are the same
        match, and under W7a's promotion-indexed reference that collision was
        guaranteed rather than rare -- both ran on the same 2,000-game cadence.
        Reusing the gate's result is exact, not an approximation: same subject,
        same opponent weights, same fixed-N rule.
        """

        report = self.last_promotion_gate
        subject = self.anchor_subject()
        if (
            report is None
            or self.last_promotion_gate_iteration != iteration
            or report.games != self.config.self_anchor_games
            or self.last_promotion_gate_subject_sha256 is None
            or self.last_promotion_gate_opponent_sha256 is None
            or not subject.is_file()
            or not path.is_file()
        ):
            return False
        return (
            self.last_promotion_gate_subject_sha256 == _file_digest(subject)
            and self.last_promotion_gate_opponent_sha256 == _file_digest(path)
        )

    def self_anchor_gate(self, iteration: int) -> tuple[GateResult, int] | None:
        """Fixed-N ``latest`` vs its games-lagged self (W7c).

        A measurement, not a decision: no early stopping (W5.6), no lifecycle
        effect, and it never touches the promotion ladder.

        Two ways it avoids playing games it does not need to: the degenerate
        self-match guard (``anchor_caught_up``), and reuse of this iteration's
        promotion gate when that gate already played this exact pairing
        (``anchor_duplicates_gate``).
        """

        if self.config.self_anchor_games <= 0:
            return None
        # `latest.pt` is installed by the soft-gate controller. A strict-gate run
        # has no rolling learner file, so there is no subject to measure and the
        # anchor skips rather than failing the iteration around it.
        if not self.anchor_subject().is_file():
            return None
        reference = self.anchor_reference(iteration)
        if reference is None:
            return None
        path, anchor_games = reference
        games = self.config.self_anchor_games
        if self.anchor_duplicates_gate(path, iteration):
            report = self.last_promotion_gate
            assert report is not None  # anchor_duplicates_gate checked it
            return (
                replace(
                    report,
                    opponent=f"self_lag_{anchor_games}_from_gate",
                    threshold=0.50,
                    decision="measurement",
                ),
                anchor_games,
            )
        if self.anchor_caught_up(path):
            # The reference holds the subject's own weights. The measurement is
            # a tie by construction, so record the null rather than spending a
            # gate to reproduce it.
            pairs = games // 2
            lcb, ucb = wilson_interval(
                pairs * 0.5, pairs, z=self.config.gate_confidence_z
            )
            return (
                GateResult(
                    opponent=f"self_lag_{anchor_games}_caught_up",
                    threshold=0.50,
                    decision="measurement",
                    games=games,
                    score_rate=0.5,
                    pairs=pairs,
                    wilson_lcb=lcb,
                    wilson_ucb=ucb,
                    stop_reason="fixed_n",
                    evaluated_games=0,
                    fixed_n=True,
                    pair_scores=tuple([0.5] * pairs),
                ),
                anchor_games,
            )
        subject_path = self.anchor_subject()
        self._admit_gate((subject_path, path))
        started = time.monotonic()
        try:
            subject = self._model_agent_spec(subject_path, "anchor_subject")
            past = self._model_agent_spec(path, "anchor_past_self")
            report, _outcomes = self._wilson_model_match(
                subject,
                past,
                seed_offset=53_000_000 + iteration,
                games=self.config.self_anchor_games,
            )
            # Rebuild as a measurement: the promote/revert wording of a gate
            # decision has no meaning against a past self.
            return (
                replace(
                    report,
                    opponent=f"self_lag_{anchor_games}",
                    threshold=0.50,
                    decision="measurement",
                    stop_reason="fixed_n",
                ),
                anchor_games,
            )
        finally:
            self.phase_seconds["gate"] = (
                self.phase_seconds.get("gate", 0.0) + time.monotonic() - started
            )
            self._cleanup_gate_resources()

    def league_assignment(self, iteration: int, games: int) -> LeagueAssignment | None:
        """Draw this iteration's archived opponent, or ``None`` for pure self-play.

        Deterministic and resume-stable: the RNG is keyed on the run seed and the
        iteration only, so re-running an iteration draws the same opponent and
        assigns it to the same games. Nothing about the draw depends on wall
        clock, HOF directory ordering beyond its append order, or how far the run
        got before it was interrupted.
        """

        config = self.config
        if config.hof_opponent_fraction <= 0.0:
            return None
        if self.generation_clock(iteration) < config.hof_start_games:
            return None
        entries = self.hof.entries()
        if not entries:
            return None
        league_games = int(round(games * config.hof_opponent_fraction))
        if league_games <= 0:
            return None
        rng = random.Random(config.seed + iteration * 100_003)
        entry = self.hof.sample(rng, mode=config.hof_sampling_mode)
        if entry is None:
            return None

        # Which games are league games, and on which seat the archive sits.
        # Spread by stride rather than taking a prefix: the first N games of an
        # iteration are not interchangeable with the rest -- `first_player` is
        # `(index // 2) % 2` -- so a prefix would correlate league play with
        # seating.
        nets_p0 = [0] * games
        nets_p1 = [0] * games
        if league_games >= games:
            chosen = list(range(games))
        else:
            stride = games / league_games
            chosen = sorted({int(i * stride) for i in range(league_games)})
        for order, index in enumerate(chosen):
            # Alternate the archive's seat so it plays both sides this iteration.
            if order % 2:
                nets_p0[index] = 1
            else:
                nets_p1[index] = 1
        return LeagueAssignment(
            checkpoint=entry.path,
            sha256=entry.sha256,
            iteration_added=entry.iteration,
            nets_p0=tuple(nets_p0),
            nets_p1=tuple(nets_p1),
        )

    def resolved_schedules(self, iteration: int) -> ResolvedSchedules:
        """Freeze this iteration's generation-time schedule values."""

        return ResolvedSchedules(
            curriculum_mix_fraction=self.curriculum_mix_fraction(iteration),
            draft_prior=self.draft_prior_amount(iteration),
        )

    def schedule_knots(self) -> tuple[int, ...]:
        """Games-clock points where a schedule stops or starts moving (W5.9).

        These are the disruptions the config knows about: the curriculum finishes
        annealing, the draft prior finishes annealing, and the HOF opponent share
        switches on. Each changes the distribution the learner is training
        against, so the candidate immediately afterwards can dip for reasons that
        have nothing to do with the learner being worse.
        """

        # A HOF share changed on resume is a knot on either basis: the opponent
        # mix moved under the learner at a known clock, and the candidate right
        # after it can dip for reasons that are not the learner getting worse.
        recorded = getattr(self, "_schedule_change_knots", ())
        if not self.config.uses_games_basis():
            return tuple(
                sorted(set(self.config.gate_revert_suppress_knots) | set(recorded))
            )
        knots = {
            self.config.curriculum_anneal_games,
            self.config.draft_prior_games,
            *self.config.gate_revert_suppress_knots,
            *recorded,
        }
        if self.config.hof_opponent_fraction > 0:
            knots.add(self.config.hof_start_games)
        return tuple(sorted(knot for knot in knots if knot > 0))

    def revert_suppressed(self, iteration: int) -> bool:
        """True when a knot crossed within the gate cadence ending here.

        Promotion stays live across a knot -- a candidate that clears the LCB
        right after an LR change is genuinely better -- so only the revert
        branch is suppressed, and only for the one gate that follows.
        """

        cadence = max(1, self.config.promotion_every)
        opened = self.generation_clock(max(0, iteration - cadence + 1))
        closed = self.training_clock(iteration)
        return any(opened < knot <= closed for knot in self.schedule_knots())

    def window_selection(self, iteration: int) -> WindowSelection | None:
        """Which iterations the growing window covers, or ``None`` on the legacy basis."""

        if not self.config.uses_games_basis():
            return None
        ledger = self.games_ledger
        return self.config.growing_window().select(
            self.training_clock(iteration),
            iteration,
            ledger.games_for,
            ledger.known_iterations(),
        )

    def window_iterations(self, iteration: int) -> int:
        """Window length in iterations, for the warm-buffer aging filter.

        The warm buffer predates the games clock and ages records by iteration
        distance.  Rather than leave it on a config value the games basis no
        longer uses -- which would silently diverge from the real window -- derive
        an equivalent length from the window actually in force.
        """

        selection = self.window_selection(iteration)
        if selection is None:
            return self.config.replay_window
        target = self.config.growing_window().games(self.training_clock(iteration))
        target_iterations = max(1, target // max(1, self.config.games_per_iteration))
        # The TARGET, not the realised count. Early in a run the ledger holds
        # only a handful of the run's own iterations, so `len(selection)` is
        # tiny -- and using it here throttles the warm buffer to match a history
        # that does not exist yet, which is the one thing a warm buffer is for.
        # cloud3 loaded 21,000 warm games and trained iteration 0 on 2,000;
        # cloud4 loaded 40,000 and trained on 2,000, at 3.16 passes. Taking the
        # larger of the two keeps the steady-state behaviour identical, since
        # the realised count reaches the target once the run has generated
        # enough of its own iterations.
        if selection.iterations:
            return max(len(selection.iterations), target_iterations)
        return target_iterations

    def schedule_state(self, iteration: int) -> dict[str, Any]:
        """Every schedule's value and realised effect, for the stats row (W1.5).

        Both the schedule value and the realised window, per the plan: a run
        whose realised window sits far below its target has a window schedule
        that is not doing what the config says, and that is invisible unless both
        are logged.
        """

        selection = self.window_selection(iteration)
        state: dict[str, Any] = {
            "basis": self.config.schedule_basis,
            "games_before_iteration": self.generation_clock(iteration),
            "games_through_iteration": self.training_clock(iteration),
            "curriculum_mix_fraction": self.curriculum_mix_fraction(iteration),
            "draft_prior": self.draft_prior_amount(iteration),
            "seed_retain_fraction": self.seed_retain_fraction(iteration),
            "hof_opponent_fraction": self.config.hof_opponent_fraction,
            "learning_rate": self.config.learning_rate,
        }
        league = self.league_assignment(iteration, self.config.games_per_iteration)
        if league is None:
            state["league_games"] = 0
            state["league_opponent"] = None
        else:
            state.update(
                league_games=league.games,
                league_opponent=league.name,
                league_opponent_sha256=league.sha256,
                league_opponent_iteration=league.iteration_added,
            )
        if selection is None:
            state["replay_window_iterations"] = self.config.replay_window
        else:
            state.update(
                replay_window_target_games=selection.target_games,
                replay_window_realised_games=selection.realised_games,
                replay_window_iterations=len(selection.iterations),
                replay_window_oldest_iteration=selection.oldest_iteration,
            )
        return state

    def _append_training_log(self, row: dict[str, Any]) -> None:
        """Append one completed iteration using the manifest's existing metrics.

        Non-finite metrics are replaced with ``null`` and the paths that held
        them are recorded on the row. ``allow_nan=False`` stays -- an
        ``Infinity`` literal is not JSON and every reader would have to cope --
        but telemetry must never be able to destroy a run: by the time this is
        called, generation, training and the gate are all done, and the row is
        pure reporting. Run ``rss_check`` died exactly here, at iteration 2, on
        an ``inf`` gradient norm from a routine GradScaler overflow, discarding
        nine minutes of completed work to report a number nothing consumes.

        Sanitising is not silencing: the paths go on the row, a warning is
        printed, and a reader can find them. A genuinely diverged loss still
        shows up -- as a null beside a named field.
        """

        row, non_finite = _sanitize_non_finite(row)
        if non_finite:
            row = {**row, "non_finite_fields": non_finite}
            print(
                f"WARNING: iteration {row.get('iteration')} produced "
                f"{len(non_finite)} non-finite metric(s), logged as null: "
                + ", ".join(non_finite[:8])
                + (" ..." if len(non_finite) > 8 else "")
            )
        with self.training_log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def training_log_rows(self) -> list[dict[str, Any]]:
        """Every completed iteration, from the append-only log.

        The log is the source of truth for rows; the manifest holds provenance
        and a count (see ``RunManifest.note_iteration``).  Safe to call after
        ``initialize``, which runs ``_sync_training_log`` and so guarantees any
        rows an older run left in the manifest are present here.
        """

        if not self.training_log.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.training_log.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid training log row {line_number}: "
                        f"{self.training_log}"
                    ) from exc
        return rows

    def _sync_learner_checkpoints(self, rows: Sequence[dict[str, Any]]) -> None:
        """Backfill immutable post-transition learner snapshots for older runs.

        Every trained champion originated as a retained candidate, so reset rows
        can normally be reconstructed by matching the row's ``latest_sha256``
        against the candidate archive. New iterations write these snapshots
        directly through ``LifecycleAdapter.record_learner``.
        """

        if not rows:
            return
        sources: dict[str, Path] = {}
        for pattern in ("learner_*.pt", "candidate_*.pt", "_bootstrap_init.pt"):
            for path in self.checkpoint_dir.glob(pattern):
                sources.setdefault(_file_digest(path), path)
        for rolling in (self.current_best, self.checkpoint_dir / "latest.pt"):
            if rolling.is_file():
                sources.setdefault(_file_digest(rolling), rolling)

        for row in sorted(rows, key=lambda item: int(item["iteration"])):
            iteration = int(row["iteration"])
            expected = row.get("latest_sha256")
            if not expected:
                continue
            target = self.learner_checkpoint(iteration)
            if target.is_file():
                actual = _file_digest(target)
                if actual != expected:
                    raise ValueError(
                        f"learner snapshot digest mismatch at iteration {iteration}: "
                        f"row has {expected}, file has {actual}"
                    )
                sources.setdefault(actual, target)
                continue
            source = sources.get(str(expected))
            if source is None:
                warnings.warn(
                    f"cannot reconstruct learner snapshot for iteration {iteration} "
                    f"with digest {expected}; the self-anchor will skip that point",
                    stacklevel=2,
                )
                continue
            atomic_copy(source, target)
            sources.setdefault(str(expected), target)

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
        return build_model(
            "transformer",
            self.config.d_model,
            self.config.layers,
            self.config.heads,
        )

    @staticmethod
    def _built_heads(model) -> int:
        """The head count a model was actually built with.

        Read off the attention module rather than re-derived from config, so a
        checkpoint can never record a head count the weights were not trained
        under.  Note `model.heads` is the output-head bundle, not this.
        """

        source = getattr(model, "_orig_mod", model)
        encoder = getattr(source, "encoder", None)
        if encoder is not None and len(encoder.layers):
            return int(encoder.layers[0].self_attn.num_heads)
        return int(getattr(source, "attention_heads", LEGACY_HEADS))

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
        window_iterations = self.window_iterations(iteration)
        return [
            record
            for record in self.warm_records
            if newest - (record.iteration if record.iteration is not None else newest)
            + iteration
            < window_iterations
        ]

    def _warn_unverifiable_legacy_digests(
        self,
        records: Sequence[GameRecord],
        *,
        source: str,
    ) -> None:
        """Warn once when Rust will skip legacy RNG-inclusive digest fields."""

        if self.config.derive_backend != "rust" or self._legacy_digest_warning_emitted:
            return
        legacy = sum(
            record.digest_version == LEGACY_DIGEST_VERSION for record in records
        )
        if not legacy:
            return
        self._legacy_digest_warning_emitted = True
        warnings.warn(
            f"{legacy} of {len(records)} records in {source} carry legacy "
            "RNG-inclusive digests; the Rust derive path cannot verify their "
            "stored trajectory and final digests. Structural runtime checks "
            "still apply. Run --derive-backend python once for a full preflight "
            "if provenance is uncertain.",
            UserWarning,
            stacklevel=2,
        )

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
        # Before the staleness filter: an age window says nothing about whether
        # the labels still mean the same thing, and a recent record computed
        # under an old target rule is exactly as unusable as an ancient one.
        check_target_versions(
            records,
            source=str(warm_path),
            allow_stale=self.config.allow_stale_targets,
        )
        self.warm_records, self.last_warm_stats = filter_warm_records_by_staleness(
            records, max_staleness
        )
        self._warn_unverifiable_legacy_digests(
            self.warm_records,
            source=f"warm buffer {warm_path}",
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
                    "heads": self._built_heads(model),
                    "precision": self.config.precision,
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
            bootstrap_model = self._new_model()
            checkpoint = make_checkpoint(
                bootstrap_model,
                {
                    "model": "transformer",
                    "d_model": self.config.d_model,
                    "layers": self.config.layers,
                    "heads": self._built_heads(bootstrap_model),
                    "precision": self.config.precision,
                    "iteration": -1,
                },
            )
            torch.save(checkpoint, self.current_best)
        manifest_payload = json.loads(self.manifest.path.read_text(encoding="utf-8"))
        stored_precision = manifest_payload.get("config", {}).get(
            "precision", "fp32"
        )
        if stored_precision != self.config.precision:
            raise ValueError(
                "cannot resume with changed precision: run started with "
                f"{stored_precision!r} but resumed with "
                f"{self.config.precision!r}"
            )
        self._refuse_changed_code(manifest_payload)
        self._reload_schedule_change_knots(manifest_payload)
        self._refuse_changed_schedules(manifest_payload)
        # After the guards, never before: they compare against `base_config`,
        # and an intervention is not a config change the operator made.
        self.restore_intervention()
        self._sync_training_log(manifest_payload)
        self._sync_learner_checkpoints(self.training_log_rows())
        if self.config.warm_buffer:
            self._load_warm_buffer(Path(self.config.warm_buffer))
        if self.config.seed_games:
            seed_path = self.buffer_dir / "curriculum_seed.jsonl"
            generate_seed_buffer(
                seed_path,
                games=self.config.seed_games,
                seed=self.config.seed,
                workers=self.config.workers,
                process_workers=self.config.process_workers,
                backend=self.config.generation_backend,
                rust_slots=self.config.rust_slots,
                rust_global_batch_cap=self.config.rust_global_batch_cap,
            )

    def _refuse_changed_code(self, manifest_payload: dict[str, Any]) -> None:
        """Refuse a resume that runs different code than the run started with (W6.5).

        A multi-day run is resumed by re-running the setup script, which pulls
        first. Without this, a pull that lands between two iterations silently
        splits the run across two engines, and every measurement afterwards is
        attributed to whichever commit the manifest happened to record. The
        commit alone is not enough: a dirty tree at the same SHA is different
        code, so the launch-time diff digest is compared too.
        """

        stored = manifest_payload.get("git") or {}
        stored_commit = str(stored.get("commit", "unknown"))
        if stored_commit in ("", "unknown"):
            # A manifest from a checkout without git provenance; nothing to
            # compare against, and refusing would strand the run.
            return
        current = self.manifest.code_identity()
        current.pop("_diff", None)
        drift: list[str] = []
        if current["commit"] not in ("", "unknown") and current["commit"] != stored_commit:
            drift.append(
                f"commit: run started on {stored_commit[:12]} but resumed on "
                f"{current['commit'][:12]}"
            )
        elif "diff_sha256" in stored and stored["diff_sha256"] != current["diff_sha256"]:
            drift.append(
                "uncommitted changes differ from the ones the run started with"
            )
        if not drift:
            return
        detail = "; ".join(drift)
        if self.config.allow_resume_code_drift:
            warnings.warn(
                f"resuming on different code ({detail}); "
                "--allow-resume-code-drift was passed, so the run continues and "
                "its rows now span more than one engine. Confirm the Rust/Python "
                "derive parity gate was rerun for this code before continuing",
                stacklevel=2,
            )
            return
        raise ValueError(
            f"cannot resume on different code ({detail}); check out the "
            f"recorded commit, or pass --allow-resume-code-drift to accept a "
            "run whose iterations span more than one engine"
        )

    def _refuse_changed_schedules(self, manifest_payload: dict[str, Any]) -> None:
        """Refuse a resume that moves any schedule position (W1.4).

        Same mechanism and same reasoning as the precision guard: a schedule
        changed mid-run makes every iteration before and after it incomparable,
        and the run has no way to record that it happened.

        The default for a missing ``schedule_basis`` is ``iterations``, not the
        config default -- a manifest written before the games clock existed
        describes a run that really did anneal on iteration counts, so resuming
        it under the new default is exactly the silent rescaling this guard is
        for.  Such a run resumes with ``--schedule-basis iterations``.
        """

        stored_config = manifest_payload.get("config") or {}
        if not stored_config:
            return
        stored_basis = stored_config.get("schedule_basis", "iterations")
        stored = dict(self.config.schedule_identity())
        for key in stored:
            stored[key] = stored_config.get(key, None)
        stored["schedule_basis"] = stored_basis
        # The manifest config is immutable launch provenance. Recorded changes
        # form the effective regime on later resumes, so compare with the last
        # accepted value rather than repeatedly comparing with the launch value.
        for entry in self._recorded_schedule_changes(manifest_payload):
            for key, delta in entry.get("changes", {}).items():
                if key in stored and isinstance(delta, dict) and "to" in delta:
                    stored[key] = delta["to"]
        current = self.config.schedule_identity()

        changed = {
            key: (stored[key], value)
            for key, value in current.items()
            # A key absent from an older manifest cannot be compared, so it is
            # not treated as a change -- only a value that is present and
            # different is.  Basis is exempt: its default is known above.
            if stored[key] is not None and stored[key] != value
        }
        if stored_basis != current["schedule_basis"]:
            changed["schedule_basis"] = (stored_basis, current["schedule_basis"])
        if not changed:
            return

        # HOF changes are an explicit, recorded exception rather than comparable
        # schedules. Any non-HOF schedule change in the same resume still makes
        # the launch invalid under this narrow override.
        hof_changed = {k: v for k, v in changed.items() if k in HOF_SCHEDULE_KEYS}
        positional = {k: v for k, v in changed.items() if k not in HOF_SCHEDULE_KEYS}
        if hof_changed and not positional and self.config.allow_hof_change:
            self._record_hof_change(hof_changed)
            return

        detail = "; ".join(
            f"{key}: {was!r} -> {now!r}" for key, (was, now) in sorted(changed.items())
        )
        hint = ""
        if "schedule_basis" in changed:
            hint = (
                f" To continue this run unchanged, pass "
                f"--schedule-basis {stored_basis}."
            )
        elif hof_changed and not positional:
            hint = (
                " --allow-hof-change accepts this HOF-only regime boundary, "
                "records it against the games clock, and suppresses the next "
                "gate's revert; metrics across the boundary remain incomparable."
            )
        raise ValueError(
            f"cannot resume with changed training schedules ({detail})." + hint
        )

    def _record_hof_change(self, changed: dict[str, tuple[Any, Any]]) -> None:
        """Write the change to the manifest and make it a revert-suppress knot.

        Durability is the point: without it a reader of the finished run cannot
        tell which iterations trained against the archive, which is exactly the
        objection the schedule guard raises.
        """

        clock = self.games_ledger.total_through(
            max(self.games_ledger.known_iterations(), default=-1)
        )
        self.manifest.record_schedule_change(
            {
                "at_games": clock,
                "changes": {k: {"from": w, "to": n} for k, (w, n) in changed.items()},
            }
        )
        self._reload_schedule_change_knots()
        warnings.warn(
            "HOF opponent share changed on resume ("
            + "; ".join(f"{k}: {w!r} -> {n!r}" for k, (w, n) in sorted(changed.items()))
            + f") at games clock {clock}; recorded in the manifest, and the next "
            "gate's revert is suppressed while the opponent mix shifts",
            stacklevel=2,
        )

    def _recorded_schedule_changes(
        self, manifest_payload: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Recorded regime changes in games-clock order.

        This is the single parser used both by the schedule guard and by knot
        restoration, so they cannot disagree about which entries are effective.
        Stable sorting preserves append order for multiple changes at one clock.
        """

        if manifest_payload is None:
            try:
                manifest_payload = json.loads(
                    self.manifest.path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                return []
        entries = [
            entry
            for entry in manifest_payload.get("schedule_changes", [])
            if isinstance(entry, dict) and int(entry.get("at_games", 0)) >= 0
        ]
        return sorted(entries, key=lambda entry: int(entry.get("at_games", 0)))

    def _reload_schedule_change_knots(
        self, manifest_payload: dict[str, Any] | None = None
    ) -> None:
        """Recorded change points, read back so a later resume still sees them."""

        self._schedule_change_knots = tuple(
            sorted(
                int(entry["at_games"])
                for entry in self._recorded_schedule_changes(manifest_payload)
                if int(entry.get("at_games", 0)) > 0
            )
        )

    def _load_model_checkpoint(self, path: str | Path):
        """Rebuild a saved model under the architecture it was trained with.

        The head count comes from the checkpoint, not the run config, and a
        disagreement raises.  Attention parameter shapes do not encode the head
        count, so `load_state_dict` accepts a mismatch and the model silently
        computes something the weights were never trained for -- unlike a
        d_model or layer-count mismatch, which fails loudly on shape.
        """

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        stored = checkpoint.get("config", {})
        heads = heads_from_config(stored)
        expected = self._built_heads(self._new_model())
        if heads != expected:
            raise ValueError(
                f"checkpoint {path} was trained with heads={heads} but this run "
                f"builds heads={expected}; resume with the original --heads or "
                "start a new run directory"
            )
        model = build_model(
            "transformer", self.config.d_model, self.config.layers, heads
        )
        load_checkpoint(path, model, checkpoint=checkpoint)
        return model, checkpoint

    def load_model(self, path: str | Path):
        return self._load_model_checkpoint(path)[0]

    @staticmethod
    def checkpoint_agent_name(
        path: str | Path, role: str, checkpoint: dict[str, Any] | None = None
    ) -> str:
        if checkpoint is None:
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
        # Resolved once, before any game runs: every game in an iteration must
        # see the same schedule position, and the games clock advances as this
        # iteration writes its buffer.
        schedules = self.resolved_schedules(iteration)
        if self.config.generation_backend == "rust":
            return self._generate_iteration_rust(
                model, iteration, destination, jobs, schedules
            )
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
                initargs=(model_state, self.config, iteration, schedules),
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
        base = Evaluator(
            model,
            self.config.device,
            self.config.inference_batch,
            precision=self.config.precision,
        )
        started = time.monotonic()
        with CoalescingEvaluator(
            base,
            max_batch=self.config.inference_batch,
            max_wait_ms=self.config.inference_wait_ms,
        ) as service:
            records = run_jobs(
                jobs,
                lambda job: _self_play_game(
                    job, service, self.config, iteration, schedules
                ),
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
        schedules: ResolvedSchedules,
    ) -> list[GameRecord]:
        """Generate neural and curriculum-bot games in the Rust scheduler."""

        import seven_wonders_rust as swr

        mix_fraction = schedules.curriculum_mix_fraction
        # One entry per job, in job order: the bot's name and seat, or None for
        # a pure self-play game. Assignment is unchanged -- same RNG stream, same
        # bot type, same seat -- only the *scheduling* differs: every game now
        # goes into a single scheduler call.
        #
        # Grouping by (bot type, seat) and calling once per group was forced by
        # the old per-call bot API, and it cost ~22% of generation: ~15% of games
        # split over up to eight groups, each draining its own slot pool a few
        # games at a time, so bot games ran at ~0.52 games/s against the neural
        # group's 1.40 regardless of how many slots the pool had.
        assignments: list[tuple[str, int] | None] = []
        for job in jobs:
            rng = random.Random(job.seed ^ 0xC6BC279692B5CC83)
            if rng.random() < mix_fraction:
                bot_type = CURRICULUM_BOT_TYPES[
                    (job.index // 2) % len(CURRICULUM_BOT_TYPES)
                ]
                assignments.append((bot_type().name, job.index % 2))
            else:
                assignments.append(None)

        evaluator = Evaluator(
            model,
            self.config.device,
            self.config.rust_global_batch_cap,
            precision=self.config.precision,
        )
        league = self.league_assignment(iteration, len(jobs))
        if league is None:
            adapter = rust_flat_batch_adapter(evaluator)
        else:
            # Network 0 is the learner, network 1 the archive. Routed on the
            # searcher inside Rust, so the archive's network drives the whole of
            # the archive's search and nothing else -- see
            # `rust_searcher_routed_flat_batch_adapter`.
            opponent_model = self.load_model(league.checkpoint)
            adapter = rust_searcher_routed_flat_batch_adapter(
                (
                    evaluator,
                    Evaluator(
                        opponent_model,
                        self.config.device,
                        self.config.rust_global_batch_cap,
                        precision=self.config.precision,
                    ),
                )
            )
            print(
                f"iteration {iteration}: league play -- {league.games} of "
                f"{len(jobs)} games vs {league.name}"
            )
        started = time.monotonic()
        rust_metrics = []
        draft_prior = schedules.draft_prior
        bot_games = sum(1 for entry in assignments if entry is not None)
        neural_games = len(jobs) - bot_games
        # Phase 1 put a whole group in one call so the scheduler could hold
        # `rust_slots` games active and activate a queued game whenever one
        # finished. Per-game bot assignment extends that to the whole iteration:
        # neural and curriculum-bot games share one pool.
        seeds = [job.seed for job in jobs]
        first_players = [(job.index // 2) % 2 for job in jobs]
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
            cheap_double_reveal_offsets=(
                self.config.cheap_double_reveal_offsets
            ),
            max_inflight_batches=self.config.rust_max_inflight_batches,
            scheduler_workers=self.config.rust_scheduler_workers,
            max_active_slots=self.config.rust_slots,
            bots_p0=[
                entry[0] if entry is not None and entry[1] == 0 else None
                for entry in assignments
            ],
            bots_p1=[
                entry[0] if entry is not None and entry[1] == 1 else None
                for entry in assignments
            ],
            nets_p0=list(league.nets_p0) if league is not None else None,
            nets_p1=list(league.nets_p1) if league is not None else None,
            bot_exploration=self.config.bot_exploration,
            bot_policy_iterations=self.config.bot_policy_iterations,
        )
        records = phase_d_records_from_rust(raw_records, validate=False)
        if league is not None:
            records = _tag_league_opponents(records, league)
        rust_metrics.append(metrics)
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
            # W3: retain the scheduler evidence instead of reducing it to the
            # number of chunks. These counters explain batch-width and forced
            # expansion throughput changes across a long run.
            "rust_scheduler": dict(metrics),
        }
        if not hasattr(self, "phase_seconds"):
            self.phase_seconds = {}
        self.phase_seconds["generation"] = elapsed
        _write_records(destination, records)
        return records

    def training_records(self, iteration: int) -> list[GameRecord]:
        selection = self.window_selection(iteration)
        if selection is None:
            paths = ReplayWindow(self.config.replay_window).paths(
                self.buffer_dir, iteration
            )
        else:
            paths = [
                self.games_ledger.path_for(known) for known in selection.iterations
            ]
        live = self._warm_records_for_iteration(iteration)
        iteration_records = [
            record for path in paths for record in read_records(path)
        ]
        live.extend(iteration_records)
        seed_fraction = self.seed_retain_fraction(iteration)
        seed_path = self.buffer_dir / "curriculum_seed.jsonl"
        if seed_fraction <= 0.0 or not seed_path.exists():
            self._warn_unverifiable_legacy_digests(
                live,
                source=f"iteration {iteration} replay window",
            )
            return live
        seed_records = read_records(seed_path)
        desired = round(len(seed_records) * seed_fraction)
        rng = random.Random(self.config.seed + iteration)
        rng.shuffle(seed_records)
        selected = live + seed_records[: min(desired, len(seed_records))]
        self._warn_unverifiable_legacy_digests(
            selected,
            source=f"iteration {iteration} replay window",
        )
        return selected

    @property
    def optimizer_state_path(self) -> Path:
        return self.checkpoint_dir / "optimizer_state.pt"

    def _load_optimizer_state(self) -> dict | None:
        """AdamW moments carried across self-play iterations.

        Rebuilding the optimizer every iteration threw away the moment
        estimates and restarted the LR schedule, so each iteration re-descended
        ground the previous one had already covered.  Absent or unreadable
        state is not an error -- a cold start just warms up again.
        """

        path = self.optimizer_state_path
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # noqa: BLE001 - corrupt state is recoverable
            print(f"optimizer state unreadable ({error}); starting cold")
            return None
        return payload.get("state")

    def _save_optimizer_state(self, state: dict, iteration: int) -> None:
        payload = {"state": state, "iteration": iteration}
        temporary = self.optimizer_state_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(self.optimizer_state_path)

    def clear_optimizer_state(self) -> None:
        """Drop the carried moments when the learner is reset to current_best.

        After a revert the weights jump backwards; moments accumulated against
        the rejected trajectory no longer describe the loss surface under them.
        """

        self.optimizer_state_path.unlink(missing_ok=True)

    @staticmethod
    def _examples_raw_array_bytes(examples: Sequence[Example]) -> int:
        return sum(
            int(getattr(example, field).nbytes)
            for example in examples
            for field in EXAMPLE_ARRAY_FIELDS
        )

    def _cache_capacity_bytes(self) -> int:
        if self.config.example_cache_bytes > 0:
            return int(self.config.example_cache_bytes)
        return int(self.config.example_cache_examples * LEGACY_EXAMPLE_BYTES)

    def _cache_totals(self) -> tuple[int, int]:
        live = set(self._example_cache)
        for key in list(self._example_cache_raw_bytes):
            if key not in live:
                del self._example_cache_raw_bytes[key]
        for key in list(self._example_cache_game_stats):
            if key not in live:
                del self._example_cache_game_stats[key]
        raw = sum(self._example_cache_raw_bytes.get(key, 0) for key in live)
        estimated = int(math.ceil(raw * self.cache_calibration_factor))
        return raw, estimated

    def _evict_cache_to(self, target_estimated_bytes: int) -> int:
        target = max(
            int(self.config.example_cache_floor_bytes),
            int(target_estimated_bytes),
        )
        evicted = 0
        _raw, estimated = self._cache_totals()
        while self._example_cache and estimated > target:
            key, _dropped = self._example_cache.popitem(last=False)
            self._example_cache_raw_bytes.pop(key, None)
            self._example_cache_game_stats.pop(key, None)
            evicted += 1
            _raw, estimated = self._cache_totals()
        return evicted

    def sample_resources(self, phase: str) -> dict[str, int | float | None]:
        sample = self.resource_monitor.sample(phase)
        if self.memory_budget_bytes and int(sample["rss_bytes"] or 0) > self.memory_budget_bytes:
            before = int(sample["rss_bytes"] or 0)
            raw, estimated = self._cache_totals()
            over = before - self.memory_budget_bytes
            target = max(0, estimated - int(over * 1.25))
            evicted = self._evict_cache_to(target)
            gc.collect()
            after = self.resource_monitor.sample(f"{phase}_after_pressure")
            self.resource_monitor.pressure(
                phase,
                budget_bytes=self.memory_budget_bytes,
                rss_before_bytes=before,
                rss_after_bytes=int(after["rss_bytes"] or 0),
                cache_raw_bytes=raw,
                cache_estimated_bytes=estimated,
                evicted_games=evicted,
            )
        return sample

    def resource_stats(self):
        raw, estimated = self._cache_totals()
        return self.resource_monitor.as_stats(
            cache_estimated_bytes=estimated,
            cache_raw_array_bytes=raw,
            cache_calibration_factor=self.cache_calibration_factor,
            phase_seconds=self.phase_seconds,
        )

    def _cached_examples(
        self,
        records: list[GameRecord],
        *,
        on_record_derived: (
            Callable[[GameRecord, GameDerivationStats], None] | None
        ) = None,
    ) -> list[Example]:
        """Vectorized examples for `records`, replaying each game at most once.

        Equivalent to `examples_from_records(records, ...)` -- same examples, same
        order -- but a game already replayed in this process is served from the
        cache instead of being replayed again. Its W3 game-stat summary travels
        with the cached examples, so reporting does not add another traversal.
        Kingdomino's loop keeps encoded examples in a live ring buffer and never
        re-derives them; 7WD stores compact, verifiable game *records* and
        rebuilds, which costs the whole replay window every iteration. This
        keeps 7WD's records as the durable source of truth and pays their
        replay/encoding once per process.

        Keyed on a sha256 of the record's serialized payload, computed once while
        parsing JSONL. In-memory/generated records fall back to hashing the same
        canonical `to_json_line` that writes them. The key therefore covers every
        field the examples depend on without reserializing the entire 40k-game
        replay window on every iteration.

        `trajectory_digest` is not sufficient and was the first version of this
        key. It chains the *replayed states*, so it says nothing about
        `policy_target`, `visits`, `sims` or `policy_excluded` -- which decide
        both whether a move becomes an example at all (`is_fast_search_move`) and
        what its policy label is. Two records with the same played trajectory and
        different search targets, which is exactly what reanalysis and warm-buffer
        imports produce, would have shared an entry. It is also a *stored* field,
        so keying on it let a record whose actions were altered without updating
        it hit the cache and skip the replay that would have caught the
        alteration. The old implementation paid that full serialization and hash
        on every lookup: ~0.74 ms per game is ~30 s at the cloud run's 40k-game
        window even when every game is a cache hit.
        """

        cache = self._example_cache
        fast = self.config.record_fast_moves
        rss_before = int(psutil.Process().memory_info().rss)
        raw_before, _estimated_before = self._cache_totals()
        derived_games = 0
        derived_examples = 0
        python_derived_games = 0
        rust_derived_games = 0
        used: set[tuple] = set()
        out: list[Example] = []
        keyed_records: list[tuple[tuple[str, bool], GameRecord]] = []
        missing: OrderedDict[tuple[str, bool], GameRecord] = OrderedDict()
        for record in records:
            digest = record.source_digest
            if digest is None:
                digest = hashlib.sha256(
                    to_json_line(record).encode("utf-8")
                ).hexdigest()
            key = (digest, fast)
            keyed_records.append((key, record))
            if key not in cache or key not in self._example_cache_game_stats:
                missing.setdefault(key, record)

        missing_items = list(missing.items())
        python_items = [
            (key, record)
            for key, record in missing_items
            if self.config.derive_backend == "python"
        ]
        rust_items = [
            (key, record)
            for key, record in missing_items
            if self.config.derive_backend == "rust"
        ]
        derived_by_key: dict[
            tuple[str, bool], tuple[list[Example], GameDerivationStats]
        ] = {}
        for key, record in python_items:
            summaries: list[GameDerivationStats] = []
            examples = examples_from_record(
                record,
                record_fast_moves=fast,
                on_derived=summaries.append,
            )
            if len(summaries) != 1:
                raise AssertionError(
                    "example derivation must produce one game-stat summary"
                )
            derived_by_key[key] = (examples, summaries[0])
            python_derived_games += 1
        if rust_items:
            rust_rows = derive_records_rust(
                [record for _key, record in rust_items],
                record_fast_moves=fast,
            )
            if len(rust_rows) != len(rust_items):
                raise AssertionError("Rust example derivation lost record alignment")
            derived_by_key.update(
                (key, row) for (key, _record), row in zip(rust_items, rust_rows)
            )
            rust_derived_games += len(rust_items)
        derived_rows = [derived_by_key[key] for key, _record in missing_items]

        if len(derived_rows) != len(missing_items):
            raise AssertionError("example derivation lost record alignment")
        for ((key, _record), (examples, game_stats)) in zip(
            missing_items, derived_rows
        ):
            cache[key] = examples
            self._example_cache_game_stats[key] = game_stats
            self._example_cache_raw_bytes[key] = self._examples_raw_array_bytes(
                examples
            )
            derived_games += 1
            derived_examples += len(examples)

        for key, record in keyed_records:
            cached = cache[key]
            game_stats = self._example_cache_game_stats[key]
            cache.move_to_end(key)
            if on_record_derived is not None:
                on_record_derived(record, game_stats)
            used.add(key)
            out.extend(cached)

        if derived_games and not self._cache_calibrated:
            rss_after = int(psutil.Process().memory_info().rss)
            raw_after, _estimated_after = self._cache_totals()
            raw_delta = max(0, raw_after - raw_before)
            rss_delta = max(0, rss_after - rss_before)
            if raw_delta:
                measured = rss_delta / raw_delta
                # RSS is noisy because allocators reserve arenas. Never use a
                # factor below the A1 measurement, and cap one-off noise so a
                # concurrent allocation cannot disable the cache permanently.
                self.cache_calibration_factor = min(
                    3.0,
                    max(DEFAULT_CACHE_CALIBRATION_FACTOR, measured),
                )
            self._cache_calibrated = True
        capacity = self._cache_capacity_bytes()
        evicted = 0
        if capacity == 0:
            cache.clear()
            self._example_cache_raw_bytes.clear()
            self._example_cache_game_stats.clear()
        else:
            # Evict strictly to the cap, including games in the current window.
            # `out` already holds every reference this training call needs, so
            # dropping a current-window entry costs a replay next iteration and
            # nothing else. An earlier version refused to evict below the current
            # window, which meant a window larger than the cap was retained in
            # full -- the cap silently did nothing, which is the opposite of what
            # a memory bound is for. Entries touched above sit at the end, so
            # eviction still takes the least recently used first.
            evicted = self._evict_cache_to(capacity)
        raw_held, estimated_held = self._cache_totals()
        held_examples = sum(len(value) for value in cache.values())
        self.last_example_cache_stats = {
            "games": len(records),
            "examples": len(out),
            "replayed_games": derived_games,
            "replayed_examples": derived_examples,
            "python_derived_games": python_derived_games,
            "rust_derived_games": rust_derived_games,
            "cached_games": len(cache),
            "cached_examples": held_examples,
            "evicted_games": evicted,
            "capacity_bytes": capacity,
            "raw_array_bytes": raw_held,
            "estimated_bytes": estimated_held,
            "calibration_factor": self.cache_calibration_factor,
            # Retained for old dashboards; it is explicitly an estimate.
            "capacity_examples": capacity // LEGACY_EXAMPLE_BYTES,
        }
        # The condition that matters is not "did we evict" but "is the cap below
        # one window": that is the state where every iteration re-replays
        # everything, and it must be loud on the *first* such call rather than
        # inferred later from a slow run.
        window_estimated = int(
            self._examples_raw_array_bytes(out) * self.cache_calibration_factor
        )
        if capacity and window_estimated > capacity:
            print(
                f"example cache cap ({capacity / 1e9:.2f} GB) is below this "
                f"window ({len(out):,} examples, ~{window_estimated / 1e9:.2f} "
                "GB calibrated); every iteration will re-replay the whole "
                "window. Raise --example-cache-bytes/--example-cache-examples "
                "or shrink --replay-window-cap-games."
            )
        self.sample_resources("post_replay_derivation")
        return out

    def train_candidate(
        self,
        records: list[GameRecord],
        iteration: int,
        *,
        source_checkpoint: str | Path | None = None,
        on_record_derived: (
            Callable[[GameRecord, GameDerivationStats], None] | None
        ) = None,
    ) -> Path:
        if len(records) < self.config.min_games_to_train:
            raise ValueError(
                f"need {self.config.min_games_to_train} games to train, "
                f"got {len(records)}"
            )
        replay_started = time.monotonic()
        examples = self._cached_examples(
            records,
            on_record_derived=on_record_derived,
        )
        replay_seconds = time.monotonic() - replay_started
        self.phase_seconds["replay_derivation"] = replay_seconds
        target_baselines = baselines(examples)
        train_examples, val_examples = stable_game_split(
            examples, self.config.val_fraction, self.config.val_split_salt
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
                precision=self.config.precision,
            )
        training_started = time.monotonic()
        history, optimizer_state = train_steps(
            model,
            train_examples,
            val_examples,
            device=self.config.device,
            steps=self.config.train_steps,
            batch_size=self.config.train_batch_size,
            lr=self.config.learning_rate,
            warmup_steps=self.config.train_warmup_steps,
            weight_decay=self.config.weight_decay,
            aux_weight=self.config.aux_weight,
            value_weight=self.config.value_weight,
            value_bootstrap=self.config.value_bootstrap,
            validate_every=self.config.validate_every,
            optimizer_state=self._load_optimizer_state(),
            restore_best_val=self.config.restore_best_val,
            seed=self.config.seed + iteration,
            precision=self.config.precision,
        )
        training_seconds = time.monotonic() - training_started
        self.phase_seconds["training"] = training_seconds
        self._save_optimizer_state(optimizer_state, iteration)
        samples = self.config.train_steps * self.config.train_batch_size
        new_examples = len(temporal_examples)
        self.last_training_stats = {
            "examples": len(examples),
            # KD's `trainable_examples`: how many of these carry a policy
            # target. Below `examples` once league or bot games are in the
            # window, because a seat the learner did not play contributes its
            # value labels but not its policy (`learner_policy_seats`).
            "policy_examples": sum(1 for example in examples if example.has_policy),
            "train_examples": len(train_examples),
            "validation_examples": len(val_examples),
            "train_games": len({example.game_key for example in train_examples}),
            "validation_games": len(
                {example.game_key for example in val_examples}
            ),
            "newest_iteration": newest_iteration,
            "pretrain_newest_metrics": pretrain_metrics,
            # The KataGo quantity: how hard each newly generated position is
            # trained on.  Run 02 ran at ~113x here; AlphaGo Zero sat near 1-2x.
            "new_examples": new_examples,
            "samples_consumed": samples,
            "samples_per_new_position": (
                samples / new_examples if new_examples else None
            ),
            "buffer_passes": samples / len(train_examples),
            "curriculum_fraction": self.seed_retain_fraction(iteration),
            # Every schedule's value and realised effect at this iteration.
            "schedules": self.schedule_state(iteration),
            # How much of the replay window had to be replayed rather than
            # served from the cache. `replayed_games` well above one iteration's
            # worth means the cache is thrashing against its cap.
            "example_cache": dict(self.last_example_cache_stats),
            "replay_derivation_seconds": replay_seconds,
            "seconds": training_seconds,
            "precision": self.config.precision,
            "steps": history,
        }
        candidate = self.checkpoint_dir / f"candidate_{iteration:04d}.pt"
        checkpoint = make_checkpoint(
            model,
            {
                "model": "transformer",
                "d_model": self.config.d_model,
                "layers": self.config.layers,
                "heads": self._built_heads(model),
                "precision": self.config.precision,
                "iteration": iteration,
                "history": history,
                "baselines": target_baselines,
                "phase_d_split": self.last_training_stats,
            },
        )
        torch.save(checkpoint, candidate)
        self.sample_resources("post_training")
        return candidate

    def _model_agent_spec(self, path: str | Path, role: str) -> ModelAgentSpec:
        model, checkpoint = self._load_model_checkpoint(path)
        source = getattr(model, "_orig_mod", model)
        return ModelAgentSpec(
            name=self.checkpoint_agent_name(path, role, checkpoint),
            model_state={
                key: value.cpu() for key, value in source.state_dict().items()
            },
            d_model=self.config.d_model,
            layers=self.config.layers,
            heads=self._built_heads(model),
            sims=self.config.gate_sims,
            mode=self.config.search_mode,
            top_k=self.config.top_k,
        )

    def _admit_gate(self, checkpoints: Sequence[str | Path]) -> None:
        """Reserve host/device headroom before loading gate models.

        Checkpoint file size is a conservative proxy for parameter bytes. Host
        construction briefly holds the deserialized state and model tensors, so
        count it twice; device construction holds one copy per evaluator.
        """

        if torch.cuda.is_available() and self.config.device.startswith("cuda"):
            torch.cuda.empty_cache()
        sample = self.sample_resources("pre_gate")
        checkpoint_bytes = sum(Path(path).stat().st_size for path in checkpoints)
        packed_bytes = self.config.gate_batch_cap() * 512 * 1024
        projected_host = int(sample["rss_bytes"] or 0) + 2 * checkpoint_bytes + packed_bytes
        if self.memory_budget_bytes and projected_host > self.memory_budget_bytes:
            need = projected_host - self.memory_budget_bytes
            _raw, estimated = self._cache_totals()
            evicted = self._evict_cache_to(max(0, estimated - int(need * 1.25)))
            gc.collect()
            after = self.resource_monitor.sample("pre_gate_after_eviction")
            projected_host = (
                int(after["rss_bytes"] or 0) + 2 * checkpoint_bytes + packed_bytes
            )
            self.resource_monitor.pressure(
                "pre_gate_admission",
                projected_rss_bytes=projected_host,
                budget_bytes=self.memory_budget_bytes,
                evicted_games=evicted,
            )
            if projected_host > self.memory_budget_bytes:
                raise MemoryError(
                    "promotion gate admission refused: projected host RSS "
                    f"{projected_host / 1e9:.2f} GB exceeds budget "
                    f"{self.memory_budget_bytes / 1e9:.2f} GB after cache eviction"
                )
        if self.vram_budget_bytes:
            projected_vram = int(sample["vram_physical_bytes"] or 0) + checkpoint_bytes + packed_bytes
            if projected_vram > self.vram_budget_bytes:
                raise MemoryError(
                    "promotion gate admission refused: projected physical VRAM "
                    f"{projected_vram / 1e9:.2f} GB exceeds budget "
                    f"{self.vram_budget_bytes / 1e9:.2f} GB"
                )

    def _cleanup_gate_resources(self) -> None:
        self.resource_monitor.sample("gate_peak")
        gc.collect()
        if torch.cuda.is_available() and self.config.device.startswith("cuda"):
            torch.cuda.empty_cache()
        self.sample_resources("post_gate")

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
                initargs=(
                    candidate_spec,
                    opponent_spec,
                    self.config.inference_batch,
                    self.config.precision,
                ),
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
            model = build_model("transformer", spec.d_model, spec.layers, spec.heads)
            model.load_state_dict(spec.model_state)
            return Evaluator(
                model,
                self.config.device,
                self.config.gate_batch_cap(),
                precision=self.config.precision,
            )

        candidate_eval = evaluator(candidate_spec)
        opponent_eval = evaluator(opponent_spec)
        # One SINGLE-net adapter per checkpoint.  The former seat-routed adapter
        # dispatched each packed row to the net of that row's *acting* player,
        # so inside one search tree the opponent's network evaluated every
        # opponent-to-move leaf and neither side ever searched with its own net.
        # A strong net facing a weak one then read garbage for every reply while
        # handing the weak net good values for its own replies, which inverted
        # the result: iter0 scored 0.225 against a random-init net at 2 sims
        # under that routing versus 0.925 with clean evaluation, and the gap
        # widened with depth because each extra simulation added another
        # wrongly-evaluated leaf.  Games are now stepped from Python so a whole
        # search always runs under the mover's own network; games sharing a
        # mover still batch together.
        adapters = (
            rust_flat_batch_adapter(candidate_eval),
            rust_flat_batch_adapter(opponent_eval),
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
            for candidate_seat in (0, 1):
                # adapters[seat] is the net sitting in that seat this leg.
                seat_adapters = (
                    adapters if candidate_seat == 0 else (adapters[1], adapters[0])
                )
                leg_records.append(
                    self._play_two_net_games(
                        swr,
                        rust_games_for_self_play(seeds, first_players),
                        seeds,
                        seat_adapters,
                    )
                )
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
                        actions=record["actions"],
                    )
                    outcomes.append(outcome)
                    result = test.update(outcome.score_for(candidate_seat))
                    game_index = pair * 2 + leg
                    if game_index % 2 == 1 and result.decision != "continue":
                        return outcomes
        return outcomes

    def _rust_model_gate_rolling(
        self,
        candidate_spec: ModelAgentSpec,
        opponent_spec: ModelAgentSpec,
        seed_offset: int,
        games: int,
        precisions: tuple[str, str] | None = None,
    ) -> list[MatchOutcome]:
        """One rolling Rust scheduler call for both seat legs and all pairs.

        Network ids are attached to the *searcher seat* (W1 routing), so every
        leaf in one search uses the mover's network. Both legs share the active
        pool, restoring batch width without the old per-leaf actor-routing bug.

        The size is fixed by the caller and every game is played: there is no
        mid-match stopping rule, so the scheduler is free to shard.
        """

        import seven_wonders_rust as swr

        # Per-side precision exists for W6.2b's precision arena, where both
        # sides are the *same* checkpoint and the only difference under test is
        # the dtype the forward pass runs in. A normal gate passes None and both
        # sides use the run's precision.
        candidate_precision, opponent_precision = (
            precisions
            if precisions is not None
            else (self.config.precision, self.config.precision)
        )

        def evaluator(spec: ModelAgentSpec, precision: str) -> Evaluator:
            model = build_model("transformer", spec.d_model, spec.layers, spec.heads)
            model.load_state_dict(spec.model_state)
            return Evaluator(
                model,
                self.config.device,
                self.config.gate_batch_cap(),
                precision=precision,
            )

        candidate_eval = evaluator(candidate_spec, candidate_precision)
        opponent_eval = evaluator(opponent_spec, opponent_precision)
        adapter = rust_searcher_routed_flat_batch_adapter(
            (candidate_eval, opponent_eval)
        )
        pairs = games // 2
        seeds: list[int] = []
        first_players: list[int] = []
        candidate_seats: list[int] = []
        nets_p0: list[int] = []
        nets_p1: list[int] = []
        for pair in range(pairs):
            seed = self.config.seed + seed_offset + pair
            first_player = pair % 2
            for candidate_seat in (0, 1):
                seeds.append(seed)
                first_players.append(first_player)
                candidate_seats.append(candidate_seat)
                nets_p0.append(0 if candidate_seat == 0 else 1)
                nets_p1.append(1 if candidate_seat == 0 else 0)
        started = time.monotonic()
        records, metrics = swr.self_play_many_flat_net(
            adapter=adapter,
            games=rust_games_for_self_play(seeds, first_players),
            game_seeds=seeds,
            global_batch_cap=self.config.gate_batch_cap(),
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
            max_active_slots=self.config.gate_slots,
            deterministic_actions=True,
            puct_root=self.config.eval_search_mode == "puct",
            nets_p0=nets_p0,
            nets_p1=nets_p1,
        )
        elapsed = time.monotonic() - started
        self.last_gate_stats = {
            "seconds": elapsed,
            "scheduler": dict(metrics),
            "games_evaluated": len(records),
            "moves": sum(len(record["moves"]) for record in records),
            "worker_starts": 1,
            "gate_slots": self.config.gate_slots,
            "gate_games": games,
        }
        outcomes: list[MatchOutcome] = []
        for record, seed, first_player, candidate_seat in zip(
            records, seeds, first_players, candidate_seats
        ):
            agents = (
                (candidate_spec.name, opponent_spec.name)
                if candidate_seat == 0
                else (opponent_spec.name, candidate_spec.name)
            )
            scores = record["scores"]
            outcomes.append(
                MatchOutcome(
                    seed=seed,
                    first_player=first_player,
                    agents=agents,
                    winner=record["winner"],
                    scores=tuple(scores) if scores is not None else None,
                    victory_type=record["victory_type"] or "unknown",
                    actions=len(record["moves"]),
                )
            )
        return outcomes

    def _play_two_net_games(
        self,
        swr,
        games: list,
        seeds: list[int],
        seat_adapters: tuple,
    ) -> list[dict]:
        """Step a batch of games, searching each move under the mover's net.

        Each ply partitions the live games by whose turn it is and issues one
        batched search per seat, so a search never mixes networks.  Splitting by
        seat roughly halves the rows per batch versus the old shared-tree call;
        that is the cost of correctness here, and arena batches were already
        small relative to the scheduler's capacity.
        """

        live = list(range(len(games)))
        actions_played = [0] * len(games)
        move_index = 0
        while live:
            by_seat: dict[int, list[int]] = {0: [], 1: []}
            for slot in live:
                by_seat[games[slot].actor].append(slot)
            for seat, slots in by_seat.items():
                if not slots:
                    continue
                results = swr.search_many_flat_net(
                    seat_adapters[seat],
                    [games[slot] for slot in slots],
                    # Distinct per game, per ply, and per seat.
                    [
                        seeds[slot] + move_index * 1_000_003 + seat * 7_919
                        for slot in slots
                    ],
                    self.config.gate_batch_cap(),
                    1,
                    self.config.gate_sims,
                    self.config.top_k,
                    force=self.config.force_root_chance,
                    age_deal_samples=self.config.age_deal_samples,
                    puct_root=self.config.eval_search_mode == "puct",
                )
                for slot, result in zip(slots, results):
                    legal = games[slot].legal_action_indices()
                    policy = result["policy"]
                    # argmax of the improved policy, not the Gumbel-perturbed
                    # `action`: exploration noise has no place in an arena game.
                    best = max(range(len(legal)), key=lambda i: policy[i])
                    games[slot].apply_index(legal[best])
                    actions_played[slot] += 1
            move_index += 1
            live = [slot for slot in live if not games[slot].is_complete()]
        return [
            {
                "winner": game.winner,
                "victory_type": game.victory_type,
                "scores": game.final_scores,
                "actions": actions,
            }
            for game, actions in zip(games, actions_played)
        ]

    def _rust_bot_gate_waves(
        self,
        candidate_spec: ModelAgentSpec,
        opponent_spec: BotAgentSpec,
        test: SPRT,
        seed_offset: int,
        *,
        max_games: int | None = None,
    ) -> list[MatchOutcome]:
        """Run model-vs-bot anchor games wholly in the Rust game loop."""

        import seven_wonders_rust as swr

        model = build_model(
            "transformer",
            candidate_spec.d_model,
            candidate_spec.layers,
            candidate_spec.heads,
        )
        model.load_state_dict(candidate_spec.model_state)
        evaluator = Evaluator(
            model,
            self.config.device,
            self.config.gate_batch_cap(),
            precision=self.config.precision,
        )
        adapter = rust_flat_batch_adapter(evaluator)
        outcomes: list[MatchOutcome] = []
        maximum_pairs = (max_games or self.config.gate_max_games) // 2
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
                    global_batch_cap=self.config.gate_batch_cap(),
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
                    puct_root=self.config.eval_search_mode == "puct",
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
                candidate_spec,
                self.config.device,
                self.config.inference_batch,
                self.config.precision,
            )
            opponent_agent = _build_gate_agent(
                opponent_spec,
                self.config.device,
                self.config.inference_batch,
                self.config.precision,
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

    @staticmethod
    def _pair_scores(outcomes: Sequence[MatchOutcome]) -> list[float]:
        if len(outcomes) % 2:
            raise ValueError("gate outcomes must contain complete seat pairs")
        paired = []
        for index in range(0, len(outcomes), 2):
            points = (
                outcomes[index].score_for(0)
                + outcomes[index + 1].score_for(1)
            )
            paired.append(1.0 if points > 1.0 else (0.5 if points == 1.0 else 0.0))
        return paired

    def _wilson_model_match(
        self,
        candidate_spec: ModelAgentSpec,
        opponent_spec: ModelAgentSpec,
        *,
        seed_offset: int,
        games: int,
        revert_suppressed: bool = False,
        precisions: tuple[str, str] | None = None,
    ) -> tuple[GateResult, list[MatchOutcome]]:
        """Play exactly ``games`` games, then decide once (W5.5)."""

        if games <= 0 or games % 2:
            raise ValueError("gate games must be a positive even number")
        if self.config.gate_backend != "rust":
            candidate_agent = _build_gate_agent(
                candidate_spec,
                self.config.device,
                self.config.inference_batch,
                self.config.precision,
            )
            opponent_agent = _build_gate_agent(
                opponent_spec,
                self.config.device,
                self.config.inference_batch,
                self.config.precision,
            )
            started = time.monotonic()
            outcomes = []
            for index in range(games):
                candidate_seat = index % 2
                agents = (
                    (candidate_agent, opponent_agent)
                    if candidate_seat == 0
                    else (opponent_agent, candidate_agent)
                )
                outcomes.append(
                    play_match(
                        self.adapter,
                        agents,
                        seed=self.config.seed + seed_offset + index // 2,
                        first_player=(index // 2) % 2,
                    )
                )
            self.last_gate_stats = {
                "seconds": time.monotonic() - started,
                "games_evaluated": len(outcomes),
                "worker_starts": 0,
                "gate_games": games,
            }
        else:
            outcomes = self._rust_model_gate_rolling(
                candidate_spec, opponent_spec, seed_offset, games, precisions
            )
        pair_scores = self._pair_scores(outcomes)
        decision, pairs, rate, lcb, ucb, stop_reason = wilson_pair_decision(
            pair_scores,
            promotion_min_lcb=self.config.promotion_min_lcb,
            revert_max_ucb=self.config.revert_max_ucb,
            z=self.config.gate_confidence_z,
            revert_suppressed=revert_suppressed,
        )
        performance = self.last_gate_stats
        evaluated_games = len(outcomes)
        moves = sum(outcome.actions for outcome in outcomes)
        return (
            GateResult(
                opponent=opponent_spec.name,
                threshold=self.config.promotion_min_lcb,
                decision=decision,
                games=pairs * 2,
                score_rate=rate,
                pairs=pairs,
                wilson_lcb=lcb,
                wilson_ucb=ucb,
                stop_reason=stop_reason,
                evaluated_games=evaluated_games,
                seconds=float(performance.get("seconds", 0.0)),
                moves_per_game=moves / evaluated_games if evaluated_games else 0.0,
                fixed_n=True,
                pair_scores=tuple(pair_scores),
                revert_suppressed=revert_suppressed,
            ),
            outcomes,
        )

    def _fixed_anchor_match(
        self,
        candidate_spec: ModelAgentSpec,
        opponent_spec: BotAgentSpec,
        *,
        threshold: float,
        seed_offset: int,
    ) -> tuple[GateResult, list[MatchOutcome]]:
        class _Continue:
            decision = "continue"

        class _NeverStop:
            @staticmethod
            def update(_score):
                return _Continue()

        started = time.monotonic()
        if self.config.gate_backend == "rust":
            outcomes = self._rust_bot_gate_waves(
                candidate_spec,
                opponent_spec,
                _NeverStop(),
                seed_offset,
                max_games=self.config.anchor_games,
            )
        else:
            candidate_agent = _build_gate_agent(
                candidate_spec,
                self.config.device,
                self.config.inference_batch,
                self.config.precision,
            )
            opponent_agent = _build_gate_agent(
                opponent_spec,
                self.config.device,
                self.config.inference_batch,
                self.config.precision,
            )
            outcomes = []
            for index in range(self.config.anchor_games):
                candidate_seat = index % 2
                agents = (
                    (candidate_agent, opponent_agent)
                    if candidate_seat == 0
                    else (opponent_agent, candidate_agent)
                )
                outcomes.append(
                    play_match(
                        self.adapter,
                        agents,
                        seed=self.config.seed + seed_offset + index // 2,
                        first_player=(index // 2) % 2,
                    )
                )
        elapsed = time.monotonic() - started
        pair_scores = self._pair_scores(outcomes)
        decision, pairs, rate, lcb, ucb, stop_reason = wilson_pair_decision(
            pair_scores,
            z=self.config.gate_confidence_z,
            measurement=True,
            measurement_threshold=threshold,
        )
        moves = sum(outcome.actions for outcome in outcomes)
        return (
            GateResult(
                opponent=_spec_name(opponent_spec),
                threshold=threshold,
                decision=decision,
                games=len(outcomes),
                score_rate=rate,
                pairs=pairs,
                wilson_lcb=lcb,
                wilson_ucb=ucb,
                stop_reason=stop_reason,
                evaluated_games=len(outcomes),
                seconds=elapsed,
                moves_per_game=moves / len(outcomes) if outcomes else 0.0,
                fixed_n=True,
                pair_scores=tuple(pair_scores),
            ),
            outcomes,
        )

    def promotion_gate(
        self,
        candidate: str | Path,
        *,
        opponent: str | Path | None = None,
        games: int = 0,
        iteration: int | None = None,
    ) -> GateResult:
        """Fixed-N promotion gate.

        ``games`` comes from the controller's W5.8 ladder; zero falls back to
        ``gate_max_games`` for callers that size the gate themselves (the W5.7
        cost bench, and any direct use in tests).
        """

        opponent = Path(opponent) if opponent is not None else self.current_best
        self._admit_gate((candidate, opponent))
        started = time.monotonic()
        try:
            candidate_spec = self._model_agent_spec(candidate, "candidate")
            opponent_spec = self._model_agent_spec(opponent, "best")
            report, outcomes = self._wilson_model_match(
                candidate_spec,
                opponent_spec,
                seed_offset=50_000_000,
                games=games or self.config.gate_max_games,
                revert_suppressed=(
                    False if iteration is None else self.revert_suppressed(iteration)
                ),
            )
            # Promotion evidence is allowed to stop on a boundary and is
            # therefore not an Elo sample. Only fixed-N anchors enter the
            # ladder; recording this prefix would bias ratings toward whichever
            # boundary happened to stop the match.
            self.last_promotion_gate = report
            self.last_promotion_gate_iteration = iteration
            self.last_promotion_gate_subject_sha256 = _file_digest(candidate)
            self.last_promotion_gate_opponent_sha256 = _file_digest(opponent)
            return report
        finally:
            self.phase_seconds["gate"] = (
                self.phase_seconds.get("gate", 0.0) + time.monotonic() - started
            )
            # Specs own CPU state dictionaries; evaluator/model objects are
            # local to the gate helpers. Dropping both before empty_cache makes
            # cleanup deterministic at the phase boundary.
            if "candidate_spec" in locals():
                del candidate_spec
            if "opponent_spec" in locals():
                del opponent_spec
            self._cleanup_gate_resources()

    def anchor_gates(self, checkpoint: str | Path) -> list[GateResult]:
        self._admit_gate((checkpoint,))
        started = time.monotonic()
        try:
            checkpoint_agent = self._model_agent_spec(checkpoint, "anchor_subject")
            targets = [
                (BotAgentSpec(GreedyBot()), 0.65),
                *[
                    (BotAgentSpec(bot_type()), 0.60)
                    for bot_type in CURRICULUM_BOT_TYPES
                ],
            ]
            reports = []
            all_outcomes = []
            for offset, (opponent, threshold) in enumerate(targets):
                report, outcomes = self._fixed_anchor_match(
                    checkpoint_agent,
                    opponent,
                    threshold=threshold,
                    seed_offset=51_000_000 + offset * 1_000_000,
                )
                reports.append(report)
                all_outcomes.extend(outcomes)
            self.elo.record(all_outcomes)
            return reports
        finally:
            self.phase_seconds["gate"] = (
                self.phase_seconds.get("gate", 0.0) + time.monotonic() - started
            )
            if "checkpoint_agent" in locals():
                del checkpoint_agent
            self._cleanup_gate_resources()

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

    def buffer_warmup_shortfall(self, records: Sequence[GameRecord]) -> str:
        """Reason to skip training this iteration, or ``""`` to proceed.

        Counts the moves that will actually become training examples, so the
        threshold means the same thing regardless of ``--record-fast-moves``.
        """

        minimum = self.config.min_buffer_positions
        if minimum <= 0:
            return ""
        keep_fast = self.config.record_fast_moves
        positions = sum(
            1
            for record in records
            for move in record.moves
            if keep_fast or not is_fast_search_move(move)
        )
        if positions >= minimum:
            return ""
        return (
            f"buffer holds {positions:,} positions, below the "
            f"--min-buffer-positions warmup of {minimum:,}"
        )

    def run_iteration(self, iteration: int) -> dict[str, Any]:
        model = self.load_model(self.current_best)
        generated = self.generate_iteration(model, iteration)
        records = self.training_records(iteration)
        shortfall = self.buffer_warmup_shortfall(records)
        if shortfall:
            print(f"iteration {iteration}: training skipped -- {shortfall}")
            row = {
                "iteration": iteration,
                "generated_games": len(generated),
                "training_games": len(records),
                "training_skipped": True,
                "training_skip_reason": shortfall,
                "promoted": False,
                "generated_summary": summarize_records(generated),
                "training_summary": summarize_records(records),
                "generation_performance": self.last_generation_stats,
            }
            self._append_training_log(row)
            self.manifest.note_iteration(iteration)
            return row
        candidate = self.train_candidate(records, iteration)
        promotion_gate = self.promotion_gate(candidate)
        promoted = promotion_gate.decision == "accept"
        previous_promotions = sum(
            bool(row.get("promoted")) for row in self.training_log_rows()
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
        self._append_training_log(row)
        self.manifest.note_iteration(iteration)
        return row

    def run(self) -> list[dict[str, Any]]:
        mode = GeneratorMode(self.config.selfplay_generator_mode)
        if mode == GeneratorMode.STRICT_GATE:
            return self._run_strict_gate()
        return self._run_controller(mode)

    def _run_strict_gate(self) -> list[dict[str, Any]]:
        """Legacy Phase D lifecycle: gate every candidate against current_best."""

        self.initialize()
        completed = [row["iteration"] for row in self.training_log_rows()]
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
                probation_reset_after=self.config.probation_reset_after,
                anchor_gate_every_promotions=self.config.anchor_gate_every_promotions,
                anchor_every_iterations=self.config.anchor_every_iterations,
                buffer_autosave_every=self.config.buffer_autosave_every,
                seed=self.config.seed,
                iterations=self.config.iterations,
                gate_ladder=self.config.gate_ladder(),
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
        # Log first: it is the source of truth, and a crash between the two
        # writes should leave the row present rather than counted-but-lost.
        self.loop._append_training_log(row)
        self.loop.manifest.note_iteration(int(row["iteration"]))

    def iterations(self) -> list[dict[str, Any]]:
        return self.loop.training_log_rows()


def build_parser() -> argparse.ArgumentParser:
    """Every Phase D flag, in one place a tool can inspect without running.

    W6.3 needs to assert that the sweep's translated flags are accepted and
    land where they should; that check must not launch a training run to
    find out.
    """

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
    parser.add_argument(
        "--curriculum-anneal-iterations",
        type=int,
        default=-1,
        help="iterations over which the bot seed buffer anneals to zero; "
        "-1 fits it to half the run so it always completes",
    )
    parser.add_argument("--opponent-fraction", type=float, default=0.15)
    parser.add_argument("--bot-policy-iterations", type=int, default=10)
    parser.add_argument("--bot-exploration", type=float, default=0.05)
    parser.add_argument("--draft-prior-iterations", type=int, default=20)
    parser.add_argument(
        "--schedule-basis",
        choices=("games", "iterations"),
        default="games",
        help="clock every training schedule reads (default: games). "
        "'iterations' preserves pre-2026-07-29 behaviour for resuming old runs; "
        "changing it across a resume is refused",
    )
    parser.add_argument(
        "--curriculum-anneal-games",
        type=int,
        default=10_000,
        help="games over which the curriculum-bot mix and seed corpus anneal to "
        "zero (games basis); 10k is where the net passed ~95%% against the bots",
    )
    parser.add_argument(
        "--draft-prior-games",
        type=int,
        default=10_000,
        help="games over which the wonder-draft tier prior anneals out "
        "(games basis)",
    )
    parser.add_argument(
        "--replay-window-coefficient",
        type=float,
        default=16.0,
        help="c in the growing window games = c * total_games ** alpha",
    )
    parser.add_argument(
        "--replay-window-exponent",
        type=float,
        default=0.6,
        help="alpha in the growing window; 0.5-0.8 keeps growth sublinear",
    )
    parser.add_argument(
        "--replay-window-cap-games",
        type=int,
        default=20_000,
        help="ceiling on the growing window, in games; must be sized against "
        "the same memory budget as --example-cache-examples",
    )
    parser.add_argument(
        "--hof-opponent-fraction",
        type=float,
        default=0.0,
        help="share of generation games played against an archived HOF "
        "checkpoint instead of current_best (compatibility default: 0; "
        "cloud launch value: 0.15)",
    )
    parser.add_argument(
        "--hof-sampling-mode",
        choices=("recency", "uniform", "latest"),
        default="recency",
        help="how a HOF opponent is drawn when --hof-opponent-fraction > 0",
    )
    parser.add_argument(
        "--hof-start-games",
        type=int,
        default=10_000,
        help="games before league play begins; early checkpoints are weak and "
        "near-identical, so an archive drawn from them adds cost, not diversity",
    )
    parser.add_argument("--cheap-sims-min", type=int, default=16)
    parser.add_argument("--cheap-sims-max", type=int, default=24)
    parser.add_argument("--full-sims-min", type=int, default=64)
    parser.add_argument("--full-sims-max", type=int, default=128)
    parser.add_argument("--full-search-fraction", type=float, default=0.25)
    parser.add_argument("--search-mode", choices=("closed", "open"), default="closed")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument(
        "--heads",
        type=int,
        default=None,
        help="attention heads (default: 64 dims per head, floor 4)",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="model-call precision; bf16 is opt-in",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=300,
        help="optimizer updates per iteration on uniform random minibatches",
    )
    parser.add_argument("--train-warmup-steps", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument(
        "--restore-best-val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="restore the best-validation weights at the end of an iteration; "
        "diagnostic-only by default until arena games show validation loss "
        "predicts strength",
    )
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--val-split-salt", default="swd-v1")
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
    parser.add_argument(
        "--promotion-min-lcb",
        type=float,
        default=0.50,
        help="promote when the pair-level Wilson lower bound clears this",
    )
    parser.add_argument(
        "--revert-max-ucb",
        type=float,
        default=0.48,
        help="revert when the pair-level Wilson UPPER bound falls below this "
        "(a confidence bound, not the old point-estimate --revert-win-rate)",
    )
    parser.add_argument(
        "--allow-resume-code-drift",
        action="store_true",
        help="resume even though the commit or working tree differs from the "
        "one the run started on; its rows will span more than one engine",
    )
    parser.add_argument(
        "--allow-hof-change",
        action="store_true",
        help="permit changing --hof-opponent-fraction / --hof-sampling-mode / "
        "--hof-start-games on a resume. The HOF-only regime boundary is "
        "recorded against the games clock and suppresses the next gate's "
        "revert; metrics across it remain incomparable. Any other schedule "
        "change is still refused.",
    )
    parser.add_argument(
        "--self-anchor-games",
        type=int,
        default=0,
        help="fixed-N games for the W7a games-indexed self-anchor (0 = off); "
        "the anchor is whatever current_best was --self-anchor-lag-games ago",
    )
    parser.add_argument("--self-anchor-lag-games", type=int, default=20_000)
    parser.add_argument("--self-anchor-every-games", type=int, default=10_000)
    parser.add_argument(
        "--intervention-ladder",
        action="store_true",
        help="enable W7b's response to detected stagnation; detection reports "
        "either way",
    )
    parser.add_argument("--intervention-window-games", type=int, default=20_000)
    parser.add_argument("--gate-confidence-z", type=float, default=1.96)
    parser.add_argument(
        "--gate-ladder-games",
        type=int,
        nargs="+",
        default=[],
        help="ascending even gate sizes (W5.8), e.g. 100 200 400 800; empty "
        "keeps one rung at --gate-max-games",
    )
    parser.add_argument(
        "--gate-ladder-step-up-after",
        type=int,
        default=2,
        help="consecutive probations before the gate steps up one rung",
    )
    parser.add_argument(
        "--gate-ladder-floor-games",
        type=int,
        default=0,
        help="games that must exist before the ladder may step up at all",
    )
    parser.add_argument(
        "--gate-revert-suppress-knots",
        type=int,
        nargs="+",
        default=[],
        help="extra games-clock points after which one gate may not revert "
        "(W5.9); schedule-driven knots are derived automatically",
    )
    parser.add_argument(
        "--gate-slots",
        type=int,
        default=48,
        help="rolling active-game slots used only by model promotion gates",
    )
    parser.add_argument("--rust-slots", type=int, default=16)
    parser.add_argument("--rust-global-batch-cap", type=int, default=256)
    parser.add_argument(
        "--gate-global-batch-cap",
        type=int,
        default=0,
        help=(
            "batch cap for evaluation paths only (0 follows "
            "--rust-global-batch-cap). The cap helps at high slot counts and "
            "hurts at low ones, so a wide gate and a narrow generator need "
            "different values. Size it with w5_gate_slots_sweep."
        ),
    )
    parser.add_argument("--rust-max-inflight-batches", type=int, default=1)
    parser.add_argument("--rust-scheduler-workers", type=int, default=1)
    parser.add_argument("--leaf-batch", type=int, default=1)
    parser.add_argument(
        "--force-root-chance", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--age-deal-samples", type=int, choices=(0, 4, 8, 16, 32), default=32)
    parser.add_argument(
        "--cheap-double-reveal-offsets",
        type=int,
        default=0,
        help="offsets per first-reveal stratum on pure double card-reveal edges, "
        "CHEAP generation moves only (0 = exhaustive, the shipped behaviour; "
        "3 is the value CHANCE_ENUMERATION_PLAN.md now favours -- its tables "
        "still show the X=2 it originally guessed, corrected at the top of that "
        "document). Still ungated: the arena and training A/B remain outstanding, "
        "which is why this ships off.",
    )
    parser.add_argument(
        "--value-bootstrap",
        type=float,
        default=0.0,
        help="blend the search's root value into the value target: "
        "(1-lambda)*outcome + lambda*[(1+v)/2, 0, (1-v)/2]. 0 is the historical "
        "hard label. The outcome is a per-GAME label shared by all ~16 rows of "
        "a game, which the network fits by recognising the game and emitting its "
        "result; root_value differs at every position, so blending it makes the "
        "target position-specific and removes that shortcut. Measured at 0.5 "
        "against cloud3 iteration 30: holdout value loss rose 8.6%% over 15 "
        "training rounds instead of 174%%, with value_acc HIGHER throughout. "
        "1.0 would be pure self-distillation, with the game result no longer "
        "constraining the head at all.",
    )
    parser.add_argument(
        "--value-weight",
        type=float,
        default=1.0,
        help="multiplier on every head that fits a per-game label (value, "
        "joint7, margin, military, science). Those five share one label across "
        "all ~16 rows of a game, so an iteration yielding ~16,500 policy labels "
        "yields only ~1,000 independent outcome labels -- while carrying 1.8 of "
        "the loss weight against the policy head's 1.0. 1.0 is historical.",
    )
    parser.add_argument(
        "--cheap-top-k",
        type=int,
        default=0,
        help="root candidate width on CHEAP self-play moves; 0 reuses --top-k "
        "(the historical behaviour). Sequential halving gives every candidate "
        "the same first-round allocation, so width and budget are not "
        "independent: 7WD's ~10 legal actions at 16-24 sims leave each candidate "
        "one simulation, and a one-simulation Q is the value head's static "
        "opinion with no opponent reply. Halving the width doubles that floor at "
        "no throughput cost. Full-search moves always use --top-k, so their "
        "targets stay comparable across runs; gates and anchors are unaffected. "
        "Measure the effect with gumbel_target_kl.py before choosing a value.",
    )
    parser.add_argument(
        "--temperature-floor",
        type=float,
        default=0.25,
        help="self-play move-selection temperature after annealing (0.25 = the "
        "historical hard-coded value). Sampling is proportional to "
        "visits ** (1/T), so 0.25 is visits**4 -- near-deterministic once the "
        "policy converges. Raising it widens the late-game distribution and is "
        "the main diversity lever in self-play. Evaluation paths are unaffected: "
        "they take the argmax.",
    )
    parser.add_argument(
        "--temperature-anneal-moves",
        type=float,
        default=20.0,
        help="moves over which temperature falls from 1.0 to --temperature-floor "
        "(20 = historical). Games run ~70 moves, so 20 leaves ~70%% of each game "
        "at the floor.",
    )
    parser.add_argument(
        "--pack-threads",
        type=int,
        default=0,
        help="threads for the Rust feature-packing pool. 0 derives it from the "
        "cgroup quota, cpuset and process affinity. Packing otherwise uses "
        "rayon's global pool, which takes every CPU the process can SEE -- on a "
        "container-limited slice that is the host's count, not the quota sold, "
        "and oversubscribing turns the measured win into a loss.",
    )
    parser.add_argument("--anchor-gate-every-promotions", type=int, default=3)
    parser.add_argument(
        "--anchor-games",
        type=int,
        default=200,
        help="fixed-N games per anchor opponent; anchors never early-stop",
    )
    parser.add_argument(
        "--anchor-every-iterations",
        type=int,
        default=0,
        help="also run the bot anchor suite every N iterations regardless of "
        "promotions, so out-of-distribution strength is tracked even when the "
        "promotion gate never fires (0 = promotion-keyed only)",
    )
    parser.add_argument(
        "--selfplay-generator-mode",
        choices=tuple(mode.value for mode in GeneratorMode),
        default="strict_gate",
        help="strict_gate = legacy gate-every-candidate lifecycle; soft_gate = "
        "cumulative rolling learner with promotion protection",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="start a NEW run from these weights instead of a random "
        "initialisation. Required to seed a run: a fresh soft-gate run installs "
        "its freshly initialised learner over both latest.pt and "
        "current_best.pt, so copying checkpoints into the run directory "
        "beforehand does not work -- they are overwritten before iteration 0. "
        "Model dimensions must match --d-model/--layers/--heads. Ignored on "
        "resume, which reads the run's own checkpoints.",
    )
    parser.add_argument(
        "--bootstrap-policy",
        choices=tuple(policy.value for policy in BootstrapPolicy),
        default="gate",
        help="auto_first_trained installs the first trained learner as best "
        "without a strength gate; gate preserves the old behavior",
    )
    parser.add_argument(
        "--promotion-every",
        type=int,
        default=4,
        help="run the promotion gate every N iterations (0 = never). A gate's "
        "cost scales with --gate-max-games, which must be large enough to "
        "resolve --gate-indifference or the gate decides nothing",
    )
    parser.add_argument("--revert-reset-after", type=int, default=0)
    parser.add_argument(
        "--probation-reset-after",
        type=int,
        default=0,
        help=(
            "reset the learner to current_best after this many consecutive "
            "probations (0 disables). Sustained probation is the state where "
            "nothing moves; pair with a gate ladder so resolution is bought "
            "before progress is discarded."
        ),
    )
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
        "--eval-search-mode",
        choices=("gumbel", "puct"),
        default="gumbel",
        help="root selection for evaluation games (gate, arena, anchors). "
        "'puct' matches the advisor's search; self-play always uses gumbel. "
        "Switching changes what gate numbers mean -- they are not comparable "
        "across the two modes",
    )
    parser.add_argument(
        "--record-fast-moves",
        action="store_true",
        help="emit training examples for cheap-search moves; default off, "
        "matching KataGo playout cap randomization. Raises buffer size ~4x, "
        "so --train-steps must rise with it",
    )
    parser.add_argument(
        "--derive-backend",
        choices=("rust", "python"),
        default="rust",
        help="buffer replay/encoding backend. Rust is the production default; "
        "Python is the independently implemented reference/fallback",
    )
    parser.add_argument(
        "--min-buffer-positions",
        type=int,
        default=0,
        help="skip training (generate only) until the replay buffer holds this "
        "many positions; no promotion or anchor gate runs on a skipped "
        "iteration. 0 disables the warmup",
    )
    parser.add_argument(
        "--example-cache-examples",
        type=int,
        default=250_000,
        help="legacy cache count, converted at the measured 17.8 KB/example "
        "(250k is ~4.45 GB). Games already replayed are served from "
        "the cache instead of being re-replayed; 0 disables it and restores "
        "the previous behaviour of rebuilding the whole replay window every "
        "iteration",
    )
    parser.add_argument(
        "--example-cache-gb",
        type=float,
        default=0.0,
        help="preferred calibrated cache ceiling in GB; 0 converts "
        "--example-cache-examples for backward compatibility",
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=0.0,
        help="host RSS budget in GB (0 = 85%% of detected RAM); configured "
        "--memory-headroom-gb is reserved outside this process limit",
    )
    parser.add_argument(
        "--vram-budget-gb",
        type=float,
        default=0.0,
        help="physical device-memory budget in GB (0 = 90%% of detected VRAM)",
    )
    parser.add_argument(
        "--memory-headroom-gb",
        type=float,
        default=2.0,
        help="host RAM reserved outside the process RSS budget",
    )
    parser.add_argument(
        "--allow-stale-targets",
        action="store_true",
        help="import a warm buffer whose policy targets were computed under a "
        "superseded definition; trains the policy head on inconsistent labels",
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
    return parser


def smoke_config(config: PhaseDConfig) -> PhaseDConfig:
    """Shrink a configuration to plumbing-smoke budgets (`--plumbing-smoke`).

    The point of the smoke is to run the *launch* flag set cheaply, so it must
    survive every flag a launch passes. That includes ``--heads``: the shipped
    configuration is 384x8x6 and 6 does not divide the smoke's 32-wide model, so
    an explicit head count has to be dropped along with the width it belonged to
    -- otherwise the check dies in ``build_model`` before any plumbing runs.
    """

    return replace(
        config,
        games_per_iteration=2,
        seed_games=8,
        workers=2,
        d_model=32,
        layers=1,
        heads=None,
        cheap_sims_min=1,
        cheap_sims_max=1,
        full_sims_min=1,
        full_sims_max=1,
        full_search_fraction=1.0,
        train_steps=2,
        train_warmup_steps=0,
        validate_every=1,
        train_batch_size=64,
        gate_sims=1,
        gate_max_games=2,
        anchor_gate_every_promotions=1,
    )


def _configure_pack_pool(requested: int) -> int:
    """Size the Rust packing pool before any generation runs.

    Row-parallel packing measured 1.2078x on the laptop, but only if the pool is
    sized to the cores that are actually usable. Rayon's global pool defaults to
    every CPU the process can see, which on a rented slice is the *host's* count
    -- so a 192-core host selling 12 would spawn 192 packing threads. The
    detection combines cgroup quota, cpuset and affinity and takes the minimum.

    Non-fatal: a run that cannot size its pool should still run, on the global
    pool, with the reason on stdout.
    """

    try:
        import seven_wonders_rust as swr
    except ImportError:  # pragma: no cover - the Python backend needs no pool
        return 0
    if not hasattr(swr, "set_pack_threads"):  # pragma: no cover - older build
        print("pack pool: extension predates --pack-threads; using rayon default")
        return 0
    limits: dict = {}
    if requested <= 0:
        try:
            from .cloud_preflight import container_limits

            limits = container_limits()
            requested = int(limits.get("effective_cpus") or 0)
        except Exception as error:  # pragma: no cover - platform dependent
            print(f"pack pool: CPU limit detection failed ({error!r})")
            requested = 0
    if requested <= 0:
        requested = os.cpu_count() or 1
    try:
        actual = swr.set_pack_threads(requested)
    except Exception as error:  # pragma: no cover
        print(f"pack pool: could not size ({error!r}); using rayon default")
        return 0
    visible = os.cpu_count() or 1
    note = f" ({visible} visible)" if actual != visible else ""
    print(f"pack pool: {actual} threads{note} limits={limits or 'none detected'}")
    return actual


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_pack_pool(args.pack_threads)
    set_temperature_schedule(args.temperature_floor, args.temperature_anneal_moves)
    _configure_cheap_top_k(args.cheap_top_k)
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
        schedule_basis=args.schedule_basis,
        curriculum_anneal_games=args.curriculum_anneal_games,
        draft_prior_games=args.draft_prior_games,
        replay_window_coefficient=args.replay_window_coefficient,
        replay_window_exponent=args.replay_window_exponent,
        replay_window_cap_games=args.replay_window_cap_games,
        hof_opponent_fraction=args.hof_opponent_fraction,
        hof_sampling_mode=args.hof_sampling_mode,
        hof_start_games=args.hof_start_games,
        cheap_sims_min=args.cheap_sims_min,
        cheap_sims_max=args.cheap_sims_max,
        full_sims_min=args.full_sims_min,
        full_sims_max=args.full_sims_max,
        full_search_fraction=args.full_search_fraction,
        search_mode=args.search_mode,
        top_k=args.top_k,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        precision=args.precision,
        train_steps=args.train_steps,
        train_warmup_steps=args.train_warmup_steps,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        aux_weight=args.aux_weight,
        value_weight=args.value_weight,
        value_bootstrap=args.value_bootstrap,
        validate_every=args.validate_every,
        restore_best_val=args.restore_best_val,
        val_fraction=args.val_fraction,
        val_split_salt=args.val_split_salt,
        min_games_to_train=args.min_games_to_train,
        min_buffer_positions=args.min_buffer_positions,
        example_cache_examples=args.example_cache_examples,
        example_cache_bytes=int(args.example_cache_gb * 1024**3),
        memory_budget_gb=args.memory_budget_gb,
        vram_budget_gb=args.vram_budget_gb,
        memory_headroom_gb=args.memory_headroom_gb,
        record_fast_moves=args.record_fast_moves,
        derive_backend=args.derive_backend,
        eval_search_mode=args.eval_search_mode,
        generation_backend=args.generation_backend,
        gate_backend=args.gate_backend,
        gate_sims=args.gate_sims,
        gate_max_games=args.gate_max_games,
        gate_alpha=args.gate_alpha,
        gate_beta=args.gate_beta,
        gate_indifference=args.gate_indifference,
        promotion_min_lcb=args.promotion_min_lcb,
        revert_max_ucb=args.revert_max_ucb,
        self_anchor_games=args.self_anchor_games,
        self_anchor_lag_games=args.self_anchor_lag_games,
        self_anchor_every_games=args.self_anchor_every_games,
        intervention_ladder=args.intervention_ladder,
        intervention_window_games=args.intervention_window_games,
        allow_resume_code_drift=args.allow_resume_code_drift,
        allow_hof_change=args.allow_hof_change,
        gate_ladder_games=tuple(args.gate_ladder_games),
        gate_ladder_step_up_after=args.gate_ladder_step_up_after,
        gate_ladder_floor_games=args.gate_ladder_floor_games,
        gate_revert_suppress_knots=tuple(args.gate_revert_suppress_knots),
        gate_confidence_z=args.gate_confidence_z,
        gate_slots=args.gate_slots,
        rust_slots=args.rust_slots,
        rust_global_batch_cap=args.rust_global_batch_cap,
        gate_global_batch_cap=args.gate_global_batch_cap,
        rust_max_inflight_batches=args.rust_max_inflight_batches,
        rust_scheduler_workers=args.rust_scheduler_workers,
        leaf_batch=args.leaf_batch,
        force_root_chance=args.force_root_chance,
        age_deal_samples=args.age_deal_samples,
        cheap_double_reveal_offsets=args.cheap_double_reveal_offsets,
        anchor_gate_every_promotions=args.anchor_gate_every_promotions,
        anchor_games=args.anchor_games,
        anchor_every_iterations=args.anchor_every_iterations,
        selfplay_generator_mode=args.selfplay_generator_mode,
        bootstrap_policy=args.bootstrap_policy,
        init_checkpoint=args.init_checkpoint,
        promotion_every=args.promotion_every,
        revert_reset_after=args.revert_reset_after,
        probation_reset_after=args.probation_reset_after,
        buffer_autosave_every=args.buffer_autosave_every,
        warm_buffer_max_staleness=args.warm_buffer_max_staleness,
        allow_stale_targets=args.allow_stale_targets,
        device=args.device,
    )
    if args.plumbing_smoke:
        config = smoke_config(config)
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
