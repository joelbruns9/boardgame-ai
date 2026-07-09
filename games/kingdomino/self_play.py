"""
self_play.py — serial AlphaZero self-play training for 2-player Kingdomino.

This is the correctness-first build (Session 1 of the plan): a single-process
loop that generates self-play games, trains both heads, and benchmarks
progress.  It is deliberately serial — slow but easy to verify.  The throughput
layer (batched inference server + multi-process workers) is added later and
must reproduce this version's data given the same seeds.

THE LOOP (one iteration):
  1. Self-play: the current network drives MCTS to play games.  Engine
     'python' uses AlphaZeroMCTS+PIMC (closed-loop); 'open_loop' uses
     OpenLoopMCTS (resamples deck order per simulation, no outer PIMC loop).
     For each move we store (encoded public state, MCTS visit policy, legal
     mask, current actor); at game end every stored position is labelled with
     the terminal outcome z from its actor's perspective.
  2. Train: sample batches from the replay buffer (with on-the-fly D4
     augmentation) and minimise value MSE + masked policy cross-entropy.
  3. Benchmark + checkpoint: play the new network against a baseline to track
     progress, and save the weights.

KEY CORRECTNESS POINTS:
  - Imperfect information: run_pimc redeterminizes the hidden deck INSIDE the
    search; the stored training state is the PUBLIC state, encoded info-set
    safe (encoder never reads deck order).  No determinized world leaks out.
  - Perspective: value target and the encoder share the current-actor frame;
    the policy target indices come from encode_action in the same frame as the
    policy head's output, so target and prediction align.
  - Masked policy loss: the softmax denominator covers LEGAL actions only,
    matching how priors are extracted at inference.  The legal mask is stored
    and augmented by the SAME D4 element as the policy, so they stay consistent
    under rotation/flip.

DOES NOT IMPORT evaluation.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from games.kingdomino.game import GameState, Phase, determine_winner
from games.kingdomino.encoder import (
    encode_state, compute_target_z, FLAT_LAYOUT, FLAT_SIZE,
    compute_target_own_score, compute_target_opponent_score, compute_target_win,
)
from games.kingdomino.action_codec import encode_action, decode_action, NUM_JOINT_ACTIONS
from games.kingdomino.augmentation import (
    augment, augment_mask, NUM_D4_TRANSFORMS,
    _D4_ELEMENTS, _transform_spatial, _transform_policy,
)
from games.kingdomino.network import KingdominoNet, masked_log_softmax
from games.kingdomino.mcts_az import (
    AlphaZeroMCTS, OpenLoopMCTS,
    make_serial_evaluator, make_batched_evaluator,
    run_pimc, run_pimc_open_loop,
    visit_counts_to_policy, select_move,
)
# Module alias so make_rust_evaluator reads MARGIN_GAIN/ALPHA at call time —
# the run_self_play_training / worker-init override (mcts_az.MARGIN_GAIN =
# cfg.margin_gain) mutates this same module object, so the override propagates.
import games.kingdomino.mcts_az as _mcts_az
from games.kingdomino.bots import GreedyBot
from games.kingdomino.diagnostics import (
    compute_all_diagnostics, check_alpha_transition,
)
from games.kingdomino.run_manifest import (
    initialize_run_manifest, record_checkpoint,
)
from games.kingdomino.promotion import (
    DEFAULT_CURRENT_BEST,
    DEFAULT_FIXED_SUITE,
    compare_fixed_suite,
    decide_promotion,
    evaluate_network_match,
    fixed_suite_summary_for_net,
    promote_current_best,
    promotion_payload,
    sha256_file,
)
from games.kingdomino.hof import (
    DEFAULT_HOF_DIR,
    HOFEntry,
    add_hof_entry,
    load_hof_net,
    read_hof_index,
    sample_hof_entry,
)


# ─── 1. Configuration ─────────────────────────────────────────────────────
GENERATOR_MODES = ("latest", "current_best", "strict_gate", "soft_gate")


@dataclass
class SelfPlayConfig:
    """Self-play training config.

    CLOUD RUN (Phase 5) rationale — the settled launch settings and WHY:
      - sims=800: target quality regime (visit targets sharpen meaningfully
        between 200 and 800; this is where open-loop's averaging over futures
        matters).
      - batch_slots=32, leaf_batch=6: confirmed optimal by a sweep on the 3070
        (32ch/4b, sims=100, 64 games): games/s peaked at bs=32 (4.12) and FELL
        on either side (16:3.42, 24:3.46, 48:3.93, 64:3.83) — at bs=32 the GPU
        forward is already ~88% filled and compute-bound, so larger batches add
        per-tick latency + CPU tree work with no gain.  (games_per_iter=50 also
        caps usable slots at ~50.)  inference_amp left OFF: fp16 autocast was a
        0.74x SLOWDOWN here (1.22→0.90 games/s) — the net is small enough to be
        latency-bound, so the fp16 cast overhead exceeds any compute saving.
      - leaf-eval D2H readback is f32 (make_rust_evaluator .float(), Rust widens
        to f64 on entry): halves the K×3390 logit transfer; tree math stays f64.
      - alpha=0.5: reserved margin band B in the win-gated leaf value.
      - lambda_score=0.5, lambda_w=0.25: start low; policy is the core signal.
      - buffer_capacity=100_000: 50 iters × 50 games × ~80 positions ≈ 200k
        total; the 100k cap holds roughly the most recent ~25 iterations.
      - min_buffer_to_train=5_000: at ~4000 positions/iter, training starts
        after iter 2.
      - engine='batched_open_loop' with workers=1: a single Rust BatchedMCTS;
        parallelism comes from batch_slots, NOT workers (a hard constraint).
    """
    # network
    channels: int = 96
    blocks: int = 8
    bilinear_dim: int = 64
    # search
    n_simulations: int = 100
    n_determinizations: int = 1   # PIMC worlds sampled per move
    leaf_batch: int = 1           # leaf-parallel batch for self-play search
                                  # (1 = serial; >1 validated via policy_compare)
    batch_slots: int = 32         # concurrent slots for --engine batched
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    fpu: float = 0.0              # first-play-urgency value for unvisited children
    virtual_loss: int = 1        # leaf-parallel / batched virtual loss magnitude
    allow_tf32: bool = True       # speed up CUDA float32 conv/linear inference
    inference_amp: bool = False   # optional autocast for self-play inference
    eval_pad_to_batch: int = 0     # pad inference batches to fixed size (CUDA graphs)
    pin_transfer: bool = False     # stage evaluator inputs in pinned host memory
    profile_eval_timing: bool = False
    temp_moves: int = 20          # sample ∝ visits for this many plies, then greedy
    # replay buffer
    buffer_capacity: int = 50_000
    # training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    # Threads for ReplayBuffer.sample_batch densify+augment.  Default 1 (serial):
    # the per-example work is dominated by small GIL-held numpy ops, and the
    # GIL-free Rust augment is too short to overlap — measured threading REGRESSES
    # this workload ~2x (see ReplayBuffer.sample_batch).  The real speedup came
    # from moving augment_mask into Rust (~1.8x serial).  Kept configurable for
    # platforms/workloads where threading may help.
    sample_workers: int = 1
    value_weight: float = 1.0
    policy_weight: float = 1.0
    lambda_score: float = 0.5     # weight on own + opp score MSE losses
    lambda_w: float = 0.25        # weight on win BCE loss
    score_scale: float = 160.0    # normalization divisor for score heads
    grad_clip: float = 1.0        # max global grad norm; <=0 disables clipping
    augment: bool = True
    # Win-gated leaf value (open-loop / batched_open_loop): with w = 2*win-1,
    #   win_gate  = w**4
    #   leaf_value = (1-alpha)*w + alpha*win_gate*tanh((own_norm-opp_norm)*margin_gain).
    # alpha is the reserved margin band B: margin is suppressed in close positions
    # and fully active in DECIDED ones (symmetric wins/losses).  These override the
    # module-level mcts_az.MARGIN_GAIN / mcts_az.ALPHA at the top of
    # run_self_play_training, and are saved in the checkpoint config so an old
    # checkpoint's leaf-value formula is recoverable.  alpha=0.5 matches mcts_az.
    margin_gain: float = 2.0
    alpha: float = 0.5
    # Exact endgame solver budget for the batched engines. When a self-play root
    # reaches a terminal-adjacent position (deck∈{0,4}), it is solved exactly
    # (minimax) instead of MCTS — perfect value/policy targets, no GPU forwards.
    # Per-position wall-clock limit in seconds; <= 0.0 disables it (ablation).
    # Higher budgets reduce fallbacks but spend more CPU on the hardest endgames.
    exact_endgame_max_secs: float = 3.0
    # How exact roots price dominated children for the POLICY label (the root
    # value and chosen move are exact in every mode):
    #   "exact"       — every child solved full-window (historical; ~11x the
    #                   work of a value-only solve on the real fallback tail);
    #   "soft_clamp"  — children within exact_clamp_delta raw points of the
    #                   best are exact, the rest only PROVEN >= delta worse and
    #                   recorded at the clamp value (label error one-sided and
    #                   bounded);
    #   "argmax_ties" — uniform over proven-tied-best children (DEFAULT: won
    #                   the 2026-07-05 label-shape ablation 231-162-7 vs
    #                   soft_clamp head-to-head AND is the cheapest mode; see
    #                   exact_endgame_solver.md).
    exact_policy_mode: str = "argmax_ties"
    exact_clamp_delta: float = 10.0
    # Optional JSONL sidecar for exact-solver fallback root states. Kept outside
    # the replay buffer so normal training examples remain compact and leak-safe.
    exact_fallback_positions: str = ""
    # Training-batch sampling weight for endgame positions (game_progress >= 0.75)
    # relative to others. Endgames carry exact minimax labels, so concentrating
    # gradient there is high-value. 1.0 = uniform; 2.0 = endgames drawn 2x.
    endgame_oversample: float = 2.0
    # loop
    n_iterations: int = 40
    games_per_iteration: int = 50
    train_steps_per_iteration: int = 200
    min_buffer_to_train: int = 2_000   # don't train until the buffer has warmed up
    # Phase-sliced calibration diagnostics (Milestone 2): run compute_all_diagnostics
    # every N iterations (amortises the ~1-2s cost without losing curve resolution).
    # 0 disables.  Logged keys: win_brier_by_phase, policy_kl_by_phase, calibration
    # curve, value bias, mcts_lift_rate, and the alpha_trigger stub.
    diag_every: int = 5
    # benchmark
    benchmark_every: int = 1
    benchmark_seeds: int = 10
    benchmark_sims: int = 50
    benchmark_determinizations: Optional[int] = None  # None → reuse n_determinizations
    # Elo rating integration (Phase 2).  Periodic rating of saved checkpoints
    # against the anchor pool via elo_rating.py.  Optional — a missing
    # elo_rating.py or anchors file degrades to a no-op, never crashing training.
    elo_every: int = 0              # rate checkpoint every N iters (0 = disabled)
    elo_anchors: str = ""           # path to elo_anchors.csv (default: auto-find)
    elo_db: str = ""                # path to elo_db.json (default: auto-find)
    elo_games_log: str = ""         # path to elo_games.jsonl (default: auto-find)
    elo_sims: int = 400             # sims for rating games
    elo_games_per_anchor: int = 32  # paired seeds per anchor (64 games each)
    smart_elo: bool = False
    smart_elo_on_promote: bool = False
    smart_elo_games_per_anchor: int = 32
    smart_elo_sims: int = 400
    # Milestone 6 promotion / gated self-play. Default off: normal runs still use
    # the latest learner for self-play. When enabled, self-play is generated by
    # current_best while the learner trains; only a promoted learner replaces the
    # generator, preventing transient regressions from poisoning new self-play.
    current_best_path: str = str(DEFAULT_CURRENT_BEST)
    gated_selfplay: bool = False
    selfplay_generator_mode: str = "latest"
    promotion_every: int = 0
    promotion_games: int = 384
    promotion_sims: int = 100
    soft_gate_revert_win_rate: float = 0.48
    promotion_min_win_rate: float = 0.55
    promotion_min_lcb: float = 0.50
    promotion_confidence_z: float = 1.96
    promotion_seed: int = 20260630
    promotion_fixed_suite: str = str(DEFAULT_FIXED_SUITE)
    promotion_fixed_suite_tolerance: float = 0.05
    promotion_skip_fixed_suite: bool = False
    promotion_update_best: bool = False
    # Run8: gate a rolling average of the last K iteration checkpoints instead
    # of the raw learner snapshot. Snapshots are noisy samples off the
    # trajectory mean (run7: individual checkpoints measured 43-44% vs the
    # peak while their 10-iter average measured 48.9%); gating the average
    # removes the winner's-curse on promotions and banks reproducible
    # strength. 0/1 = gate the raw learner (old behavior).
    promotion_average_k: int = 0
    # Run8: after this many CONSECUTIVE failed-below-revert gate checks, reset
    # the LEARNER's weights (and optimizer moments) to current_best. One
    # revert only swaps the generator and gives the learner 5+ iterations of
    # baseline-quality data to recover (protects a mid-breakthrough learner);
    # run7 showed a diverged learner never recovers unaided (9 straight
    # reverts). 0 = never reset (old behavior).
    revert_reset_after: int = 0
    # Milestone 7 Hall-of-Fame opponent mixing. Default off. The first
    # implementation samples one HOF opponent per iteration and runs HOF games
    # as a separate mixed-model open-loop block to keep the high-throughput
    # self-play path unchanged.
    hof_dir: str = str(DEFAULT_HOF_DIR)
    hof_fraction: float = 0.0
    hof_fraction_schedule: str = ""
    hof_start_iter: int = 50
    hof_sample_weights: str = "recency"
    hof_sims: int = 200
    hof_current_sims: int = 0  # learner-side sims in HOF games; 0 = full n_simulations
    hof_temp_moves: int = 0
    hof_dirichlet_epsilon: float = 0.0
    hof_add_every: int = 0
    hof_add_tag: str = "current_best"
    # io / misc
    device: str = "cpu"
    seed: int = 0
    warm_start_path: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    # Replay-buffer persistence (see ReplayBuffer.save / .load).
    save_buffer: str = ""
    # Path to save the final replay buffer after training completes (also saved
    # on KeyboardInterrupt).  Empty string = don't save.  Not auto-derived from
    # checkpoint_dir — must be requested explicitly so a multi-GB pickle is never
    # written by surprise.  Example: checkpoints_ol_local_cont/buffer_final.pkl
    warm_buffer: str = ""
    # Path to a previously saved buffer to load before iteration 1.  Empty string
    # = start with an empty buffer.  Only meaningful alongside --warm_start (the
    # network weights should match the policy that produced the buffer).
    warm_buffer_max_staleness: int = 200
    # Discard examples older than this many iterations when loading warm_buffer.
    # Default 200 is permissive — keeps everything in a typical run.  Set lower
    # (e.g. 50) to drop very old examples from a much weaker policy.
    # Per-iteration structured log (JSON Lines).  If None, auto-derive from
    # checkpoint_dir ({checkpoint_dir}/training_log.jsonl) or, if that is also
    # None, ./training_log_{timestamp}.jsonl.  Always appended, so resuming a
    # run continues the same log.
    log_path: Optional[str] = None
    # engine: "python"    = AlphaZeroMCTS + PIMC (closed-loop, oracle);
    #         "open_loop" = OpenLoopMCTS (open-loop, averages over deck
    #                       orders internally; n_determinizations ignored);
    #         "rust"      = RustMCTS (in-process leaf-eval callback, no IPC server);
    #         "batched"   = one Rust BatchedMCTS driving batch_slots games.
    engine: str = "python"
    # Item 20: wrap the INFERENCE net (make_rust_evaluator) with torch.compile on
    # CUDA.  Inference-only — never the training net.  Off by default (A/B flag).
    compile_net: bool = False
    # compile_dynamic: forwarded as torch.compile(dynamic=...).  None = torch's
    # default (auto).  True compiles ONE shape-generic graph, avoiding the
    # per-shape recompilation storm the variable leaf-eval batch (mean ~45,
    # range 6-192) causes with static graphs — set it if bench_compile flags a
    # dynamic=False storm.  False forces static per-shape graphs.
    compile_dynamic: bool | None = None
    # Item 19: prefetch the next training batch on a background thread while the
    # GPU runs train_step on the current one.  Off by default (A/B flag).
    prefetch_batches: bool = False
    # Item 17: run two BatchedMCTS instances over disjoint halves of the game
    # batch, overlapping one's CPU tree work with the other's GPU forward.  Only
    # for engine=batched / batched_open_loop.  Off by default (A/B flag).
    double_buffer: bool = False
    # Step 1.5: solve endgames on a background thread (overlaps the GPU eval).
    # Pair with a larger batch_slots (overbooking) so slots out solving don't
    # collapse the GPU batch.  Off by default.
    async_solve: bool = False
    # Preferred CPU split for async_solve: reserve this many logical CPUs for
    # game generation (MCTS descent/backup + Python orchestration), and give the
    # rest to the dedicated exact-solver pool. Scales naturally on cloud boxes.
    game_cpus: int = 2
    # Explicit override for the dedicated solver pool. 0 => derive from
    # total_cpus - game_cpus. Keep for compatibility and special experiments.
    solver_cpus: int = 0
    # Milestone 5 dynamic schedules. Format: "0:value,50:value,..."; iteration
    # uses the greatest key <= current iteration (0-based in config, 1-based run
    # iteration maps to schedule step iteration-1).
    lr_schedule: str = ""
    alpha_schedule: str = ""
    sims_schedule: str = ""
    exact_endgame_max_secs_schedule: str = ""
    games_per_iter_schedule: str = ""
    c_puct_schedule: str = ""
    dirichlet_epsilon_schedule: str = ""
    temp_moves_schedule: str = ""
    train_steps_schedule: str = ""
    buffer_capacity_schedule: str = ""
    # KataGo-inspired playout-cap randomization: a fraction of self-play games
    # use a smaller sim cap to diversify value targets while the rest keep the
    # full scheduled cap for sharper policy targets.
    fast_game_fraction: float = 0.0
    fast_game_fraction_schedule: str = ""
    fast_game_sims: int = 100
    # KataGo-style move-level playout-cap randomization. When enabled, each
    # move independently chooses a full or fast search. Fast moves default to
    # strong cheap play: no root noise, greedy selection, and no policy record.
    playout_cap_randomization: bool = False
    full_search_fraction: float = 0.25
    fast_move_sims: int = 100
    record_fast_moves: bool = False
    fast_move_dirichlet_epsilon: float = 0.0
    fast_move_temp_moves: int = 0
    # KataGo-inspired target cleanup. Conservative pruning removes one-visit
    # exploration noise. Forced-playout subtraction is opt-in and consumes root
    # priors/visit counts when the self-play path exports them.
    policy_target_pruning: bool = False
    forced_playout_subtraction: bool = False
    forced_playout_k: float = 2.0


# ─── 2. Replay buffer ─────────────────────────────────────────────────────
@dataclass
class Example:
    """One training example.  Boards/flat stored as float16 to halve memory
    (their values — one-hot, small fractions — lose nothing meaningful); the
    policy target and legal mask are stored sparse since both are far smaller
    than their dense 3390-wide form."""
    my_board: np.ndarray    # float16 (9, 13, 13)
    opp_board: np.ndarray   # float16 (9, 13, 13)
    flat: np.ndarray        # float16 (FLAT_SIZE,)
    policy_idx: np.ndarray  # int32 — non-zero policy indices
    policy_val: np.ndarray  # float32 — corresponding visit-proportional mass
    legal_idx: np.ndarray   # int32 — all legal joint indices (⊇ policy_idx)
    z: float                # value target in [-1, 1], current-actor frame
    own_score: float        # raw own final score (un-normalized), this actor's view
    opp_score: float        # raw opponent final score (un-normalized), this actor's view
    win_target: float       # 1.0 win / 0.5 draw / 0.0 loss, this actor's view
    # Optional root-search metadata used only by KataGo-style target cleanup.
    # Training batches ignore these fields; old replay buffers remain compatible
    # because they simply won't have useful values here.
    root_prior_idx: Optional[np.ndarray] = None   # int32 action indices
    root_prior_val: Optional[np.ndarray] = None   # float32 root priors after noise
    root_visit_count: Optional[np.ndarray] = None # int32 visits aligned to prior_idx
    # Sample ownership/provenance for HOF mixing. The training sampler only
    # consumes trainable examples; default values keep old buffers compatible.
    owner: str = "self"       # self | current | hof
    trainable: bool = True
    game_type: str = "self"
    opponent_source: str = ""
    # Training iteration that generated this example.  Used by
    # ReplayBuffer.mean_age to track buffer staleness.  Defaults to 0 so any
    # construction site that misses the kwarg yields a (stale) age reading
    # rather than crashing.
    iteration: int = 0


def _validate_example_schema(ex: Example, *, context: str) -> None:
    """Fail loudly when replay data was produced by an older encoder schema."""
    flat_shape = tuple(getattr(ex.flat, "shape", ()))
    if flat_shape != (FLAT_SIZE,):
        raise ValueError(
            f"{context}: replay example flat shape {flat_shape} does not match "
            f"current FLAT_SIZE ({FLAT_SIZE},). Start a fresh replay buffer after "
            "encoder/checkpoint_version migrations."
        )


def configure_torch_performance(cfg: SelfPlayConfig) -> None:
    """Set CUDA matmul/convolution knobs used by self-play inference."""
    if not str(cfg.device).startswith("cuda"):
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)
    if cfg.allow_tf32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _apply_augment(mb: np.ndarray, ob: np.ndarray, flat: np.ndarray,
                   policy: np.ndarray, t_id: int):
    """Apply a D4 transform to the array components only (the scalar targets are
    rotation-invariant and handled by the caller).  Uses the Rust d4_augment when
    available — it releases the GIL, so this is the part that actually runs in
    parallel across sample_batch worker threads — and falls back to the numpy
    path otherwise.  Bit-identical to augmentation.augment() for the arrays."""
    try:
        from kingdomino_rust import d4_augment as _rd4
        return _rd4(mb, ob, flat, policy, int(t_id))
    except ImportError:
        k, flip, dp = _D4_ELEMENTS[t_id]
        return (_transform_spatial(mb, k, flip),
                _transform_spatial(ob, k, flip),
                flat.copy(),
                _transform_policy(policy, k, flip, dp))


class ReplayBuffer:
    """Fixed-capacity ring buffer with O(1) random access for sampling."""
    def __init__(self, capacity: int, n_sample_workers: int = 1):
        self.capacity = capacity
        self.data: List[Example] = []
        self._pos = 0
        # Persistent thread pool reused across every sample_batch call (creating
        # one per call would swamp the per-batch work with pool-startup cost at
        # ~200 batches/iteration).  None => serial (n_sample_workers <= 1).
        self._n_workers = int(n_sample_workers)
        self._pool = (ThreadPoolExecutor(max_workers=self._n_workers)
                      if self._n_workers > 1 else None)
        # Endgame-oversampling weight cache (Change 4). Recomputing per-example
        # weights every batch is O(buffer); cache and invalidate only on add().
        self._weight_cache: Optional[np.ndarray] = None
        self._weight_cache_oversample: float = 1.0

    def close(self) -> None:
        """Shut down the sample-batch thread pool.  Call at training end."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def save(self, path: Union[str, Path]) -> None:
        """Pickle the buffer contents to disk.

        Only the ring-buffer state ('data', 'pos', 'capacity') is written —
        the ThreadPoolExecutor (_pool) is intentionally NOT pickled; it is
        reconstructed fresh in __init__ on load.  The write goes to a temp
        file that is atomically renamed into place, so a crash mid-write
        cannot leave a half-written buffer behind.
        """
        import pickle
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'data': self.data,
            'pos': self._pos,
            'capacity': self.capacity,
        }
        tmp = out.with_suffix('.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(out)  # atomic replace — no half-written file on crash
        print(f"Buffer saved: {len(self.data)} examples -> {out}")

    def load(self, path: Union[str, Path], *,
             current_iteration: int = 0,
             max_staleness: Optional[int] = None) -> int:
        """Load buffer contents from disk.  Returns number of examples loaded.

        max_staleness: if set, discard examples where
            current_iteration - example.iteration > max_staleness.
        This prevents loading very old examples from a much weaker policy.
        """
        import pickle
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        examples = payload['data']
        if max_staleness is not None:
            before = len(examples)
            examples = [
                e for e in examples
                if current_iteration - e.iteration <= max_staleness
            ]
            dropped = before - len(examples)
            if dropped:
                print(f"Buffer load: dropped {dropped} stale examples "
                      f"(staleness > {max_staleness})")
        # Respect current buffer capacity — truncate if saved buffer was larger
        if len(examples) > self.capacity:
            examples = examples[-self.capacity:]
        for i, ex in enumerate(examples):
            _validate_example_schema(
                ex,
                context=f"ReplayBuffer.load({path}) example {i}",
            )
        self.data = list(examples)
        self._pos = payload.get('pos', 0) % max(1, len(self.data))
        print(f"Buffer loaded: {len(self.data)} examples from {path}")
        return len(self.data)

    def add(self, examples: List[Example]) -> None:
        for ex in examples:
            if not getattr(ex, "trainable", True):
                continue
            _validate_example_schema(ex, context="ReplayBuffer.add")
            if len(self.data) < self.capacity:
                self.data.append(ex)
            else:
                self.data[self._pos] = ex
                self._pos = (self._pos + 1) % self.capacity
        # Any change to the buffer contents invalidates the oversampling weights.
        if examples:
            self._weight_cache = None

    def __len__(self) -> int:
        return len(self.data)

    def mean_age(self, current_iteration: int) -> float:
        """Mean iteration age of examples in the buffer.

        Each Example stores the iteration it was generated in;
        mean_age = current_iteration - mean(example.iteration).  Rising age
        means the buffer holds increasingly stale self-play data.  Returns 0.0
        if the buffer is empty.
        """
        if not self.data:
            return 0.0
        return current_iteration - float(
            np.mean([ex.iteration for ex in self.data])
        )

    def _endgame_weights(self, oversample: float) -> Optional[np.ndarray]:
        """Cached per-example sampling probabilities for endgame oversampling.

        Returns None when `oversample == 1.0` (uniform — caller uses fast
        integers()). Otherwise returns a normalized probability vector that gives
        endgame examples (game_progress >= 0.75) `oversample`x the weight of
        others. Cached and invalidated on add() since it is O(buffer) to build.
        """
        if oversample == 1.0:
            return None
        if (self._weight_cache is None
                or self._weight_cache_oversample != oversample):
            prog_idx = FLAT_LAYOUT['game_progress'].start
            weights = np.array(
                [oversample if float(ex.flat[prog_idx]) >= 0.75 else 1.0
                 for ex in self.data],
                dtype=np.float32,
            )
            total = weights.sum()
            if total > 0:
                weights /= total
            self._weight_cache = weights
            self._weight_cache_oversample = oversample
        return self._weight_cache

    def _draw_idxs(self, batch_size: int, rng: np.random.Generator,
                   oversample: float = 1.0) -> np.ndarray:
        """Draw `batch_size` buffer indices, optionally oversampling endgames."""
        weights = self._endgame_weights(oversample)
        if weights is not None:
            return rng.choice(len(self.data), size=batch_size, p=weights,
                              replace=True)
        return rng.integers(0, len(self.data), size=batch_size)

    def sample_batch(self, batch_size: int, rng: np.random.Generator,
                     device: str = "cpu", augment_d4: bool = True,
                     endgame_oversample_weight: float = 1.0):
        """Return a training batch as tensors:
        (my_board, opp_board, flat, policy, legal_mask, z,
         own_score, opp_score, win_target).
        Each example is densified and (optionally) given a random D4 transform.
        own_score/opp_score are raw (un-normalized); win_target is 1/0.5/0.

        The per-example densify + cast + augment work is independent across
        examples and (via Rust d4_augment, which drops the GIL) parallelisable, so
        it is dispatched over a persistent thread pool when n_sample_workers > 1.
        ALL rng draws happen up front in this (main) thread — np.random.Generator
        is not thread-safe — and the only torch / .to(device) calls happen here in
        the main thread AFTER the pool returns (CUDA contexts are per-thread; the
        workers touch CPU numpy only).
        """
        # Pre-draw every random choice in the main thread (rng is not thread-safe).
        idxs = self._draw_idxs(batch_size, rng, endgame_oversample_weight)
        t_ids = (rng.integers(0, NUM_D4_TRANSFORMS, size=batch_size)
                 if augment_d4 else None)

        def _process_one(args):
            buf_idx, t_id = args
            ex = self.data[int(buf_idx)]
            policy = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)
            policy[ex.policy_idx] = ex.policy_val
            mask = np.zeros(NUM_JOINT_ACTIONS, dtype=bool)
            mask[ex.legal_idx] = True
            mb = ex.my_board.astype(np.float32)
            ob = ex.opp_board.astype(np.float32)
            flat = ex.flat.astype(np.float32)
            z = float(ex.z)
            own_score = float(ex.own_score)
            opp_score = float(ex.opp_score)
            win_tgt = float(ex.win_target)
            if t_id is not None:
                # d4_augment (Rust) releases the GIL — true parallelism here.  The
                # three scalar targets are rotation-invariant (augmentation
                # contract 7), so they pass through unchanged.
                mb, ob, flat, policy = _apply_augment(mb, ob, flat, policy, int(t_id))
                mask = augment_mask(mask, int(t_id))
            return mb, ob, flat, policy, mask, z, own_score, opp_score, win_tgt

        args = [(idxs[i], int(t_ids[i]) if t_ids is not None else None)
                for i in range(batch_size)]
        if self._pool is not None and batch_size > 1:
            # chunksize matters: 256 individually-submitted tasks would drown the
            # ~40µs/example work in per-future dispatch overhead.  Hand each worker
            # a contiguous block (a few chunks per worker for light load-balancing)
            # so the GIL-free Rust augment work actually overlaps across threads.
            chunk = max(1, batch_size // (self._n_workers * 4))
            results = list(self._pool.map(_process_one, args, chunksize=chunk))
        else:
            results = [_process_one(a) for a in args]

        mbs, obs, flats, pols, masks = [], [], [], [], []
        zs, own_ss, opp_ss, win_ts = [], [], [], []
        for mb, ob, flat, pol, mask, z, own_s, opp_s, win_t in results:
            mbs.append(mb); obs.append(ob); flats.append(flat)
            pols.append(pol); masks.append(mask); zs.append(z)
            own_ss.append(own_s); opp_ss.append(opp_s); win_ts.append(win_t)

        # torch / device transfers only here, in the main thread.
        to = lambda arr: torch.from_numpy(np.stack(arr)).to(device)
        f32 = lambda vals: torch.tensor(vals, dtype=torch.float32, device=device)
        return (
            to(mbs).float(), to(obs).float(), to(flats).float(),
            to(pols).float(),
            torch.from_numpy(np.stack(masks)).to(device),       # bool
            f32(zs),
            f32(own_ss), f32(opp_ss), f32(win_ts),
        )


# ─── Milestone 5 schedules / target cleanup ─────────────────────────────────
def _parse_schedule(text: str, *, cast):
    """Parse "0:foo,50:bar" into sorted (iteration, value) pairs."""
    text = (text or "").strip()
    if not text:
        return []
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid schedule entry {part!r}; expected iter:value")
        k, v = part.split(":", 1)
        out.append((int(k.strip()), cast(v.strip())))
    out.sort(key=lambda x: x[0])
    if out and out[0][0] < 0:
        raise ValueError(f"Schedule iterations must be >= 0: {text!r}")
    return out


def _schedule_value(schedule, step: int, default):
    value = default
    for at, candidate in schedule:
        if step >= at:
            value = candidate
        else:
            break
    return value


def _compiled_schedules(cfg: SelfPlayConfig) -> dict:
    return {
        "lr": _parse_schedule(cfg.lr_schedule, cast=float),
        "alpha": _parse_schedule(cfg.alpha_schedule, cast=float),
        "n_simulations": _parse_schedule(cfg.sims_schedule, cast=int),
        "exact_endgame_max_secs": _parse_schedule(
            cfg.exact_endgame_max_secs_schedule, cast=float),
        "games_per_iteration": _parse_schedule(cfg.games_per_iter_schedule, cast=int),
        "c_puct": _parse_schedule(cfg.c_puct_schedule, cast=float),
        "dirichlet_epsilon": _parse_schedule(cfg.dirichlet_epsilon_schedule, cast=float),
        "temp_moves": _parse_schedule(cfg.temp_moves_schedule, cast=int),
        "train_steps_per_iteration": _parse_schedule(cfg.train_steps_schedule, cast=int),
        "buffer_capacity": _parse_schedule(cfg.buffer_capacity_schedule, cast=int),
        "fast_game_fraction": _parse_schedule(cfg.fast_game_fraction_schedule, cast=float),
        "hof_fraction": _parse_schedule(cfg.hof_fraction_schedule, cast=float),
    }


def _active_config_for_iteration(cfg: SelfPlayConfig, schedules: dict, it: int) -> SelfPlayConfig:
    """Return a shallow config copy with iteration-local schedule values."""
    step = it - 1
    values = {}
    for field, schedule in schedules.items():
        if schedule:
            values[field] = _schedule_value(schedule, step, getattr(cfg, field))
    if not values:
        return cfg
    return replace(cfg, **values)


def _apply_optimizer_schedule(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _apply_buffer_capacity(buffer: ReplayBuffer, capacity: int) -> None:
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError(f"buffer capacity must be positive, got {capacity}")
    if capacity == buffer.capacity:
        return
    if len(buffer.data) > capacity:
        buffer.data = buffer.data[-capacity:]
        buffer._pos = 0
    else:
        buffer._pos %= max(1, len(buffer.data))
    buffer.capacity = capacity
    buffer._weight_cache = None


def _is_exact_endgame_example(ex: Example) -> bool:
    """Best-effort detector for roots exact-solved by the current M1 engine."""
    bag_len = int(np.asarray(ex.flat, dtype=np.float32)[FLAT_LAYOUT["bag"]].sum())
    return bag_len in (0, 4)


def _prune_policy_target(
    pidx: np.ndarray,
    pval: np.ndarray,
    *,
    total_visits: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Prune one-visit visit-count noise and renormalize.

    KataGo's full policy target pruning subtracts forced playouts using root
    priors. The current training record only carries visits, so this conservative
    M5 pass applies the unambiguous second half: remove children whose target
    mass is consistent with <=1 visit. It keeps at least one child and preserves
    exact targets by caller policy.
    """
    if len(pidx) <= 1 or total_visits <= 1:
        return pidx, pval, 0, 0.0
    pval64 = np.asarray(pval, dtype=np.float64)
    threshold = (1.0 / float(total_visits)) + 1e-8
    keep = pval64 > threshold
    if not np.any(keep):
        keep[int(np.argmax(pval64))] = True
    removed = int(len(pidx) - int(np.count_nonzero(keep)))
    removed_mass = float(pval64[~keep].sum())
    if removed == 0:
        return pidx, pval, 0, 0.0
    new_idx = np.asarray(pidx, dtype=np.int32)[keep]
    new_val = pval64[keep]
    new_val /= new_val.sum()
    return new_idx.astype(np.int32), new_val.astype(np.float32), removed, removed_mass


def _has_root_search_stats(ex: Example) -> bool:
    return (
        getattr(ex, "root_prior_idx", None) is not None
        and getattr(ex, "root_prior_val", None) is not None
        and getattr(ex, "root_visit_count", None) is not None
        and len(ex.root_prior_idx) == len(ex.root_prior_val) == len(ex.root_visit_count)
        and len(ex.root_prior_idx) > 0
    )


def _forced_playout_subtract_policy_target(
    prior_idx: np.ndarray,
    prior_val: np.ndarray,
    visit_count: np.ndarray,
    *,
    k: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, int, float, float, int]:
    """Apply KataGo-style forced-playout subtraction to root visits.

    Returns (policy_idx, policy_val, removed_actions, removed_mass,
    subtracted_visits, effective_total_visits). If subtraction would collapse
    every action to zero, the original visit target is returned. The priors are
    expected to be the actual root priors used by search after Dirichlet noise.
    """
    idx = np.asarray(prior_idx, dtype=np.int32)
    priors = np.asarray(prior_val, dtype=np.float64)
    visits = np.asarray(visit_count, dtype=np.float64)
    if len(idx) == 0 or len(idx) != len(priors) or len(idx) != len(visits):
        raise ValueError("root prior idx/val/visit arrays must be non-empty and aligned")
    visits = np.maximum(visits, 0.0)
    total = float(visits.sum())
    if total <= 0.0:
        raise ValueError("Cannot subtract forced playouts from zero root visits")
    original = visits / total
    if k <= 0.0:
        keep = visits > 0.0
        return (
            idx[keep],
            original[keep].astype(np.float32),
            0,
            0.0,
            0.0,
            int(round(total)),
        )

    priors = np.maximum(priors, 0.0)
    prior_sum = float(priors.sum())
    if prior_sum <= 0.0:
        keep = visits > 0.0
        return (
            idx[keep],
            original[keep].astype(np.float32),
            0,
            0.0,
            0.0,
            int(round(total)),
        )
    priors = priors / prior_sum
    forced = np.sqrt(float(k) * priors * total)
    adjusted = np.maximum(0.0, visits - forced)
    adjusted_total = float(adjusted.sum())
    if adjusted_total <= 1e-12:
        keep = visits > 0.0
        return (
            idx[keep],
            original[keep].astype(np.float32),
            0,
            0.0,
            0.0,
            int(round(total)),
        )

    keep = adjusted > 1e-12
    removed_actions = int(np.count_nonzero((visits > 0.0) & ~keep))
    removed_mass = float(original[(visits > 0.0) & ~keep].sum())
    policy_val = adjusted[keep] / adjusted_total
    subtracted_visits = float((visits - adjusted).sum())
    return (
        idx[keep].astype(np.int32),
        policy_val.astype(np.float32),
        removed_actions,
        removed_mass,
        subtracted_visits,
        max(1, int(round(adjusted_total))),
    )


def _prune_examples_policy_targets(
    examples_by_game: List[List[Example]],
    *,
    total_visits: int,
    skip_exact: bool,
    one_visit_pruning: bool = True,
    forced_playout_subtraction: bool = False,
    forced_playout_k: float = 2.0,
) -> dict:
    pruned_actions = 0
    pruned_mass = 0.0
    changed_examples = 0
    forced_actions = 0
    forced_mass = 0.0
    forced_examples = 0
    forced_subtracted_visits = 0.0
    forced_missing_stats = 0
    for game in examples_by_game:
        for ex in game:
            if skip_exact and _is_exact_endgame_example(ex):
                continue
            target_total_visits = total_visits
            if forced_playout_subtraction:
                if _has_root_search_stats(ex):
                    old_idx = np.asarray(ex.policy_idx, dtype=np.int32)
                    old_val = np.asarray(ex.policy_val, dtype=np.float32)
                    (new_idx, new_val, removed, mass, sub_visits,
                     effective_visits) = _forced_playout_subtract_policy_target(
                        ex.root_prior_idx,
                        ex.root_prior_val,
                        ex.root_visit_count,
                        k=forced_playout_k,
                    )
                    same_target = (
                        np.array_equal(old_idx, new_idx)
                        and old_val.shape == new_val.shape
                        and np.allclose(old_val, new_val, atol=1e-7)
                    )
                    if not same_target:
                        ex.policy_idx = new_idx
                        ex.policy_val = new_val
                        forced_actions += removed
                        forced_mass += mass
                        forced_subtracted_visits += sub_visits
                        forced_examples += 1
                    target_total_visits = effective_visits
                else:
                    forced_missing_stats += 1
            if one_visit_pruning:
                new_idx, new_val, removed, mass = _prune_policy_target(
                    ex.policy_idx, ex.policy_val, total_visits=target_total_visits)
                if removed:
                    ex.policy_idx = new_idx
                    ex.policy_val = new_val
                    pruned_actions += removed
                    pruned_mass += mass
                    changed_examples += 1
    return {
        "policy_pruned_actions": pruned_actions,
        "policy_pruned_mass": pruned_mass,
        "policy_pruned_examples": changed_examples,
        "forced_pruned_actions": forced_actions,
        "forced_pruned_mass": forced_mass,
        "forced_pruned_examples": forced_examples,
        "forced_subtracted_visits": forced_subtracted_visits,
        "forced_missing_stats_examples": forced_missing_stats,
    }


def _merge_batched_stats(stats_list: list[dict]) -> dict | None:
    stats_list = [s for s in stats_list if s]
    if not stats_list:
        return None
    elapsed = sum(float(s.get("elapsed", 0.0)) for s in stats_list)
    total_evals = sum(int(s.get("total_evals", 0)) for s in stats_list)
    ticks = sum(int(s.get("ticks", 0)) for s in stats_list)
    out = {
        "ticks": ticks,
        "elapsed": elapsed,
        "total_evals": total_evals,
        "mean_batch": (
            sum(float(s.get("mean_batch", 0.0)) * int(s.get("ticks", 0))
                for s in stats_list)
            / max(1, ticks)
        ),
        "max_batch_cap": max(int(s.get("max_batch_cap", 0)) for s in stats_list),
        "max_batch_seen": max(int(s.get("max_batch_seen", 0)) for s in stats_list),
        "requests_per_sec": total_evals / elapsed if elapsed > 0 else 0.0,
        "step_sec": sum(float(s.get("step_sec", 0.0)) for s in stats_list),
        "eval_sec": sum(float(s.get("eval_sec", 0.0)) for s in stats_list),
        "update_sec": sum(float(s.get("update_sec", 0.0)) for s in stats_list),
        "exact_solve_count": sum(int(s.get("exact_solve_count", 0)) for s in stats_list),
        "exact_tree_solve_count": sum(int(s.get("exact_tree_solve_count", 0)) for s in stats_list),
        "exact_cache_hit_count": sum(int(s.get("exact_cache_hit_count", 0)) for s in stats_list),
        "exact_fallback_count": sum(int(s.get("exact_fallback_count", 0)) for s in stats_list),
        "exact_attempt_deck4_initial_count": sum(
            int(s.get("exact_attempt_deck4_initial_count", 0)) for s in stats_list),
        "exact_attempt_deck4_retry_count": sum(
            int(s.get("exact_attempt_deck4_retry_count", 0)) for s in stats_list),
        "exact_attempt_deck0_count": sum(
            int(s.get("exact_attempt_deck0_count", 0)) for s in stats_list),
        "exact_fallback_deck4_initial_count": sum(
            int(s.get("exact_fallback_deck4_initial_count", 0)) for s in stats_list),
        "exact_fallback_deck4_retry_count": sum(
            int(s.get("exact_fallback_deck4_retry_count", 0)) for s in stats_list),
        "exact_fallback_deck0_count": sum(
            int(s.get("exact_fallback_deck0_count", 0)) for s in stats_list),
        "exact_solver_secs": sum(float(s.get("exact_solver_secs", 0.0)) for s in stats_list),
        "exact_fallback_positions_saved": sum(
            int(s.get("exact_fallback_positions_saved", 0)) for s in stats_list),
        "total_cpus": max(int(s.get("total_cpus", 0)) for s in stats_list),
        "game_cpus": max(int(s.get("game_cpus", 0)) for s in stats_list),
        "solver_cpus": max(int(s.get("solver_cpus", 0)) for s in stats_list),
        "solver_cpus_override": any(bool(s.get("solver_cpus_override", False))
                                    for s in stats_list),
        "fast_move_count": sum(int(s.get("fast_move_count", 0)) for s in stats_list),
        "full_move_count": sum(int(s.get("full_move_count", 0)) for s in stats_list),
        "recorded_fast_move_count": sum(
            int(s.get("recorded_fast_move_count", 0)) for s in stats_list),
        "recorded_full_move_count": sum(
            int(s.get("recorded_full_move_count", 0)) for s in stats_list),
        "exact_recorded_move_count": sum(
            int(s.get("exact_recorded_move_count", 0)) for s in stats_list),
    }
    cap = out["max_batch_cap"]
    out["fill_ratio"] = out["mean_batch"] / cap if cap > 0 else 0.0
    for key in ("eval_h2d_sec", "eval_forward_sec", "eval_readback_sec", "eval_calls"):
        if any(key in s for s in stats_list):
            out[key] = sum(s.get(key, 0) for s in stats_list)
    return out


def _choose_playout_profile(
    cfg: SelfPlayConfig,
    rng: np.random.Generator,
) -> tuple[bool, int, float, int, bool]:
    """Return (is_full, sims, root_noise_eps, temp_moves, record_example)."""
    if not cfg.playout_cap_randomization:
        return True, int(cfg.n_simulations), float(cfg.dirichlet_epsilon), int(cfg.temp_moves), True
    fraction = max(0.0, min(1.0, float(cfg.full_search_fraction)))
    is_full = bool(rng.random() < fraction)
    if is_full:
        return True, int(cfg.n_simulations), float(cfg.dirichlet_epsilon), int(cfg.temp_moves), True
    return (
        False,
        max(1, int(cfg.fast_move_sims)),
        float(cfg.fast_move_dirichlet_epsilon),
        int(cfg.fast_move_temp_moves),
        bool(cfg.record_fast_moves),
    )


# ─── 3. Self-play game generation ─────────────────────────────────────────
def _game_rngs(game_seed: int) -> Tuple[random.Random, np.random.Generator]:
    """Derive the (py_rng, np_rng) for one self-play game purely from its seed.

    A game's entire output (determinizations, root Dirichlet noise, stochastic
    move selection) must be a deterministic function of game_seed ALONE — not of
    shared mutable RNG state or which worker/iteration ran it.  That property is
    what lets the parallel loop reproduce the serial loop exactly (see
    correctness_oracle).  It also makes self-play data independent of execution
    order, which is good for reproducibility and debugging regardless.

    The two streams are seeded from independently-mixed derivations of game_seed
    (via SeedSequence) so the Python and NumPy generators don't share low-bit
    structure.
    """
    ss = np.random.SeedSequence(game_seed)
    py_seed, np_seed = (int(x) for x in ss.generate_state(2, dtype=np.uint64))
    return random.Random(py_seed), np.random.default_rng(np_seed)


def _temperature(move_num: int, temp_moves: int) -> float:
    """τ=1 (sample ∝ visits) for the first `temp_moves` plies, then greedy."""
    return 1.0 if move_num < temp_moves else 0.0


def play_selfplay_game(
    mcts: Union[AlphaZeroMCTS, OpenLoopMCTS],
    *,
    n_determinizations: int,
    temp_moves: int,
    seed: int,
    py_rng: random.Random,
    np_rng: np.random.Generator,
    leaf_batch: int = 1,
    open_loop: bool = False,
    iteration: int = 0,
    playout_cfg: Optional[SelfPlayConfig] = None,
) -> Tuple[List[Example], Tuple[int, int]]:
    """Play one self-play game; return (training examples, final scores).

    `leaf_batch` is forwarded to each search (see AlphaZeroMCTS.search):
    leaf_batch=1 is the serial reference; >1 batches leaf evaluations with
    virtual loss for throughput.  The MCTS must carry a batched_evaluator for
    leaf_batch>1 to give a GPU-feed win (otherwise it loops the single one).

    When `open_loop` is True, `mcts` is an OpenLoopMCTS and the search routes
    through run_pimc_open_loop instead: it resamples the deck per simulation and
    averages internally, so n_determinizations and leaf_batch are not used (the
    outer PIMC loop is redundant).  The Example format produced is identical, so
    all downstream plumbing (buffer, train_step, augmentation) is unchanged.
    """
    state = GameState.new(seed=seed)
    records = []  # (mb, ob, flat, policy_idx, policy_val, legal_idx, actor)
    move_num = 0

    while state.phase != Phase.GAME_OVER:
        if playout_cfg is not None:
            _is_full, move_sims, noise_eps, move_temp_moves, record_example = (
                _choose_playout_profile(playout_cfg, np_rng)
            )
            old_sims = getattr(mcts, "n_simulations", None)
            old_eps = getattr(mcts, "dirichlet_epsilon", None)
            mcts.n_simulations = move_sims
            if old_eps is not None:
                mcts.dirichlet_epsilon = noise_eps
            add_noise = noise_eps > 0.0
        else:
            move_temp_moves = temp_moves
            record_example = True
            old_sims = old_eps = None
            add_noise = True
        # Root Dirichlet noise on every self-play move (standard AlphaZero).
        if open_loop:
            # Open-loop averages over deck orders internally (one fresh
            # determinization per simulation); no outer PIMC loop.  It derives
            # its own Python RNG from np_rng, so py_rng / n_determinizations /
            # leaf_batch are intentionally not passed here.
            visit_counts, _, _ = run_pimc_open_loop(
                mcts, state, add_noise=add_noise, rng=np_rng,
            )
        else:
            visit_counts, _ = run_pimc(
                mcts, state, py_rng,
                n_determinizations=n_determinizations,
                add_noise=add_noise, np_rng=np_rng,
                leaf_batch=leaf_batch,
            )
        if playout_cfg is not None:
            if old_sims is not None:
                mcts.n_simulations = old_sims
            if old_eps is not None:
                mcts.dirichlet_epsilon = old_eps
        # Training target is the visit distribution itself (τ=1), independent
        # of the selection temperature below.
        policy = visit_counts_to_policy(visit_counts, state, temperature=1.0)

        actor = state.current_actor
        if record_example:
            mb, ob, flat = encode_state(state, actor)        # PUBLIC, info-set safe
            legal = state.legal_actions()
            legal_idx = np.fromiter((encode_action(a, state) for a in legal),
                                    dtype=np.int32, count=len(legal))
            pidx = np.nonzero(policy)[0].astype(np.int32)
            pval = policy[pidx].astype(np.float32)
            records.append((
                mb.astype(np.float16), ob.astype(np.float16), flat.astype(np.float16),
                pidx, pval, legal_idx, actor,
            ))

        temp = _temperature(move_num, move_temp_moves)
        action = select_move(visit_counts, temp, np_rng)
        state = state.step(action)
        move_num += 1

    # Terminal targets, all from player-0's frame; flipped per record's actor.
    z0 = compute_target_z(state, player=0)              # kept for buffer compat
    own0 = compute_target_own_score(state, player=0)    # raw P0 final score
    opp0 = compute_target_opponent_score(state, player=0)  # raw P1 final score
    win0 = compute_target_win(state, player=0)          # 1.0/0.5/0.0 via cascade
    examples = []
    for (mb, ob, flat, pidx, pval, lidx, actor) in records:
        if actor == 0:
            z = z0
            own_s, opp_s, win_t = own0, opp0, win0
        else:
            z = -z0
            # Flip to actor-1's frame: scores swap; win complements (draw stays
            # 0.5).  Hand-check: P0 won (win0=1.0) → P1 win_target = 0.0. ✓
            own_s, opp_s = opp0, own0
            win_t = (1.0 - win0) if win0 != 0.5 else 0.5
        examples.append(
            Example(mb, ob, flat, pidx, pval, lidx, z, own_s, opp_s, win_t,
                    iteration=iteration)
        )
    scores = (state.boards[0].score().total, state.boards[1].score().total)
    return examples, scores


def play_current_vs_hof_game(
    current_mcts: OpenLoopMCTS,
    hof_mcts: OpenLoopMCTS,
    *,
    current_player: int,
    current_cfg: SelfPlayConfig,
    seed: int,
    np_rng: np.random.Generator,
    iteration: int = 0,
    opponent_source: str = "",
) -> Tuple[List[Example], Tuple[int, int], dict]:
    """Play one mixed current-vs-HOF game and keep only current-owned labels."""
    state = GameState.new(seed=seed)
    records = []
    move_num = 0
    current_records = 0
    hof_moves = 0

    while state.phase != Phase.GAME_OVER:
        actor = int(state.current_actor)
        is_current = actor == int(current_player)
        mcts = current_mcts if is_current else hof_mcts
        add_noise = is_current and float(current_cfg.dirichlet_epsilon) > 0.0
        visit_counts, _, _ = run_pimc_open_loop(
            mcts, state, add_noise=add_noise, rng=np_rng)
        policy = visit_counts_to_policy(visit_counts, state, temperature=1.0)

        if is_current:
            mb, ob, flat = encode_state(state, actor)
            legal = state.legal_actions()
            legal_idx = np.fromiter((encode_action(a, state) for a in legal),
                                    dtype=np.int32, count=len(legal))
            pidx = np.nonzero(policy)[0].astype(np.int32)
            pval = policy[pidx].astype(np.float32)
            records.append((
                mb.astype(np.float16), ob.astype(np.float16), flat.astype(np.float16),
                pidx, pval, legal_idx, actor,
            ))
            current_records += 1
            temp = _temperature(move_num, int(current_cfg.temp_moves))
        else:
            hof_moves += 1
            temp = 0.0

        action = select_move(visit_counts, temp, np_rng)
        state = state.step(action)
        move_num += 1

    z0 = compute_target_z(state, player=0)
    own0 = compute_target_own_score(state, player=0)
    opp0 = compute_target_opponent_score(state, player=0)
    win0 = compute_target_win(state, player=0)
    examples = []
    game_type = "current_vs_hof" if current_player == 0 else "hof_vs_current"
    for (mb, ob, flat, pidx, pval, lidx, actor) in records:
        if actor == 0:
            z = z0
            own_s, opp_s, win_t = own0, opp0, win0
        else:
            z = -z0
            own_s, opp_s = opp0, own0
            win_t = (1.0 - win0) if win0 != 0.5 else 0.5
        examples.append(
            Example(
                mb, ob, flat, pidx, pval, lidx, z, own_s, opp_s, win_t,
                owner="current",
                trainable=True,
                game_type=game_type,
                opponent_source=opponent_source,
                iteration=iteration,
            )
        )
    scores = (state.boards[0].score().total, state.boards[1].score().total)
    return examples, scores, {
        "current_records": current_records,
        "hof_moves": hof_moves,
        "current_player": int(current_player),
    }


def _run_hof_orientation(batched, eval_seat0, eval_seat1):
    """Drive one learner-vs-HOF BatchedMCTS to completion, routing each leaf to
    the net of the seat whose search produced it (searcher-owns-network, via
    row_search_actors — the same mechanism elo_rating uses for two-net matches).
    Returns [(seed, raw_example_tuples, (s0, s1))] carrying ALL moves' examples;
    the caller keeps only the learner's seat (each tuple's trailing actor field)."""
    results: List[Tuple[int, list, Tuple[int, int]]] = []
    ticks = 0
    while not batched.done():
        mb, ob, flat, idxs_list = batched.step()
        mb = np.asarray(mb); ob = np.asarray(ob); flat = np.asarray(flat)
        search_actors = np.asarray(batched.row_search_actors(), dtype=np.int64)
        n = mb.shape[0]
        values = np.zeros(n, dtype=np.float32)
        gathered: List[Optional[np.ndarray]] = [None] * n
        for actor_id, evaluator in ((0, eval_seat0), (1, eval_seat1)):
            rows = np.flatnonzero(search_actors == actor_id)
            if rows.size == 0:
                continue
            sub_idxs = [idxs_list[int(r)] for r in rows]
            v, g = evaluator(mb[rows], ob[rows], flat[rows], sub_idxs)
            values[rows] = np.asarray(v, dtype=np.float32)
            for i, r in enumerate(rows):
                gathered[int(r)] = np.asarray(g[i], dtype=np.float32)
        for r in range(n):
            if gathered[r] is None:
                gathered[r] = np.zeros(len(idxs_list[r]), dtype=np.float32)
        for seed, examples, scores in batched.update(values, gathered):
            results.append((int(seed), examples, (int(scores[0]), int(scores[1]))))
        ticks += 1
        if ticks > 2_000_000:
            raise RuntimeError("HOF BatchedMCTS exceeded tick guard")
    return results


def play_hof_games_batched(
    learner_net: KingdominoNet,
    hof_net: KingdominoNet,
    cfg: SelfPlayConfig,
    *,
    n_games: int,
    game_seed_start: int,
    iteration: int = 0,
    opponent_source: str = "",
) -> Tuple[List[List[Example]], List[Tuple[int, int]], dict]:
    """Rust-batched learner-vs-HOF self-play — the fast replacement for the serial
    play_current_vs_hof_game loop under the batched engines.

    Both seats are searched by their own net inside ONE open-loop BatchedMCTS
    (leaves routed by row_search_actors), so HOF games run at the same
    GPU-batched throughput as normal self-play instead of the ~28x-slower serial
    Python MCTS.  Only the LEARNER's moves become training examples — the HOF
    net's search policy is never a training target — using each example tuple's
    trailing actor field to keep the learner's seat.  The learner plays half the
    games in seat 0 and half in seat 1 for seat balance.

    ASYMMETRIC DEEP TARGETS (run7): the learner seat searches EXACTLY like
    normal self-play — sims=--sims with playout-cap randomization and
    full_search_fraction, recording only full-search moves — so the policy
    targets harvested from HOF games are full-strength.  The frozen HOF
    opponent seat is pinned via the engine's hof_opponent_seat override to a
    shallow no-record profile: sims=--hof_sims, temp_moves=--hof_temp_moves,
    dirichlet_eps=--hof_dirichlet_epsilon.  Its role is steering the learner
    into diverse positions, not labelling them.  (--hof_current_sims applies
    only to the legacy serial path and is ignored here.)

    Returns (per_game_learner_examples, per_game_scores_seat0_frame, stats) where
    stats has trainable_examples and mean_diff (learner-frame score margin)."""
    import kingdomino_rust

    effective_solver_cpus, _game_cpus, _total = _resolve_async_solver_cpus(cfg)

    def _mk_eval(net: KingdominoNet):
        return make_rust_evaluator(
            net, device=cfg.device, amp=cfg.inference_amp,
            margin_gain=cfg.margin_gain, alpha=cfg.alpha)

    eval_learner = _mk_eval(learner_net)
    eval_hof = _mk_eval(hof_net)
    hof_sims = max(1, int(cfg.hof_sims))

    def _make(n: int, seed0: int, hof_seat: int):
        return kingdomino_rust.BatchedMCTS(
            max(1, int(cfg.batch_slots)), int(n), int(seed0),
            max(1, int(cfg.n_simulations)),
            leaf_batch=max(1, int(cfg.leaf_batch)),
            virtual_loss=int(cfg.virtual_loss),
            cpuct=float(cfg.c_puct), fpu=float(cfg.fpu),
            dirichlet_alpha=float(cfg.dirichlet_alpha),
            dirichlet_eps=float(cfg.dirichlet_epsilon),
            temp_moves=int(cfg.temp_moves),
            open_loop=(cfg.engine == "batched_open_loop"),
            score_scale=float(cfg.score_scale),
            margin_gain=float(cfg.margin_gain), alpha=float(cfg.alpha),
            exact_endgame_max_secs=float(cfg.exact_endgame_max_secs),
            exact_policy_mode=str(cfg.exact_policy_mode),
            exact_clamp_delta=float(cfg.exact_clamp_delta),
            async_solve=bool(cfg.async_solve),
            solver_cpus=int(effective_solver_cpus),
            playout_cap_randomization=bool(cfg.playout_cap_randomization),
            full_search_fraction=float(cfg.full_search_fraction),
            fast_move_sims=max(1, int(cfg.fast_move_sims)),
            record_fast_moves=False,
            fast_move_dirichlet_eps=float(cfg.fast_move_dirichlet_epsilon),
            fast_move_temp_moves=int(cfg.fast_move_temp_moves),
            hof_opponent_seat=int(hof_seat),
            hof_opponent_sims=hof_sims,
            hof_opponent_dirichlet_eps=float(cfg.hof_dirichlet_epsilon),
            hof_opponent_temp_moves=int(cfg.hof_temp_moves),
        )

    all_examples: List[List[Example]] = []
    all_scores: List[Tuple[int, int]] = []
    learner_diffs: List[int] = []
    n0 = int(n_games) // 2                    # orientation 0: learner in seat 0
    n1 = int(n_games) - n0                    # orientation 1: learner in seat 1
    for n_or, learner_seat, seed0 in ((n0, 0, int(game_seed_start)),
                                      (n1, 1, int(game_seed_start) + n0)):
        if n_or <= 0:
            continue
        eval0 = eval_learner if learner_seat == 0 else eval_hof
        eval1 = eval_hof if learner_seat == 0 else eval_learner
        game_type = "current_vs_hof" if learner_seat == 0 else "hof_vs_current"
        for seed, raw_examples, (s0, s1) in _run_hof_orientation(
                _make(n_or, seed0, 1 - learner_seat), eval0, eval1):
            keep: List[Example] = []
            for tup in raw_examples:
                if int(tup[-1]) != learner_seat:  # keep only learner-searched moves
                    continue
                ex = _example_from_rust_tuple(tup, iteration=iteration)
                ex.owner = "current"
                ex.trainable = True
                ex.game_type = game_type
                ex.opponent_source = opponent_source
                keep.append(ex)
            all_examples.append(keep)
            all_scores.append((s0, s1))
            learner_diffs.append((s0 - s1) if learner_seat == 0 else (s1 - s0))

    stats = {
        "trainable_examples": int(sum(len(e) for e in all_examples)),
        "mean_diff": float(np.mean(learner_diffs)) if learner_diffs else 0.0,
    }
    return all_examples, all_scores, stats


# ─── 3b. Rust-engine self-play generation ──────────────────────────────────
# The Rust engine (--engine rust) plays the ENTIRE game inside a RustGameState
# and runs RustMCTS.search in place of AlphaZeroMCTS+PIMC.  It produces the same
# (examples, scores) contract as play_selfplay_game, so the loop / buffer /
# training / checkpointing below are completely unchanged.  Leaf evaluation goes
# through an in-process Python callable (make_rust_evaluator) — NOT the IPC
# inference server.  This is a SEPARATE engine, not a bit-for-bit reproduction of
# the Python path (its Dirichlet noise and leaf FP differ); the Python path
# remains the oracle.
# Module-level latch so the "Triton missing" heads-up (Item 20) prints at most
# once per process — make_rust_evaluator is called fresh every iteration.
_COMPILE_TRITON_WARNED = False


def _triton_available() -> bool:
    """True if the Triton package is importable.  torch.compile's inductor
    backend needs Triton to generate GPU kernels; without it, compilation of a
    CUDA graph raises TritonMissing and dynamo falls back to eager."""
    import importlib.util
    return importlib.util.find_spec("triton") is not None


def make_rust_evaluator(
    net: KingdominoNet,
    device: str = "cpu",
    amp: bool = False,
    pad_to_batch: int = 0,
    pin_transfer: bool = False,
    profile_timing: bool = False,
    *,
    margin_gain: float = _mcts_az.MARGIN_GAIN,
    alpha: float = _mcts_az.ALPHA,
    compile_net: bool = False,
    compile_dynamic: bool | None = None,
):
    """In-process batched leaf evaluator for RustMCTS.search / BatchedMCTS.update.
    Contract:
    (mb (K,9,13,13) f32, ob (K,9,13,13) f32, flat (K,FLAT_SIZE) f32, idxs_list)
      -> (values (K,) f32, [gathered_logits_i f32]).
    Returns f32 to HALVE the D2H transfer volume (logits are K×3390); the Rust
    tree casts to f64 on entry and keeps its internal accumulation in f64.

    compile_net (Item 20): when True and on CUDA, wrap the INFERENCE copy of the
    net with torch.compile.  This is the leaf-eval net only — never the training
    net (which calls .backward()).  net.eval() is already set above (BatchNorm
    requires eval before compile).  Batch size varies tick to tick; torch.compile
    recompiles on each new shape then caches, so steady-state ticks hit the cache.
    compile_dynamic is forwarded to torch.compile(dynamic=...): None (torch
    default/auto), True (one shape-generic graph — avoids per-shape recompile
    storms under the variable batch), or False (static per-shape graphs).
    """
    net = net.to(device).eval()
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if compile_net and use_cuda and hasattr(torch, "compile"):
        # Heads-up (once per process): without Triton, inductor can't codegen GPU
        # kernels — torch.compile silently falls back to EAGER (no speedup), with
        # the real cause buried in dynamo's WON'T CONVERT logs.  Surface it plainly.
        global _COMPILE_TRITON_WARNED
        if not _triton_available() and not _COMPILE_TRITON_WARNED:
            print("WARNING: --compile requested but Triton is not installed; "
                  "torch.compile's inductor backend will fall back to EAGER "
                  "(no speedup). Install triton to enable GPU compilation.")
            _COMPILE_TRITON_WARNED = True
        # Fall back to eager on any dynamo capture error rather than crashing the
        # run (e.g. an op the backend can't trace) — perf feature, not correctness.
        torch._dynamo.config.suppress_errors = True
        net = torch.compile(net, dynamic=compile_dynamic)
    use_pinned = bool(pin_transfer and use_cuda)
    # Fix 2: bind the leaf-value blend params at construction (no module-global
    # reads at call time).  Callers forward cfg.margin_gain / cfg.alpha.
    mg = float(margin_gain)
    al = float(alpha)
    timing = {"h2d": 0.0, "forward": 0.0, "readback": 0.0, "calls": 0}

    def _sync() -> None:
        if profile_timing and use_cuda:
            torch.cuda.synchronize()

    def _to_device(arr: np.ndarray) -> torch.Tensor:
        src = torch.from_numpy(arr)
        if use_pinned:
            pinned = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
            pinned.copy_(src)
            return pinned.to(device, non_blocking=True)
        return src.to(device)

    def evaluator(mbs, obs, flats, idxs_list):
        n = len(mbs)
        mb_np = np.ascontiguousarray(mbs)
        ob_np = np.ascontiguousarray(obs)
        flat_np = np.ascontiguousarray(flats)
        pad_to = int(pad_to_batch)
        if pad_to > n:
            mb_pad = np.zeros((pad_to, *mb_np.shape[1:]), dtype=mb_np.dtype)
            ob_pad = np.zeros((pad_to, *ob_np.shape[1:]), dtype=ob_np.dtype)
            flat_pad = np.zeros((pad_to, flat_np.shape[1]), dtype=flat_np.dtype)
            mb_pad[:n] = mb_np
            ob_pad[:n] = ob_np
            flat_pad[:n] = flat_np
            mb_np, ob_np, flat_np = mb_pad, ob_pad, flat_pad
        t0 = time.perf_counter() if profile_timing else 0.0
        mb_t = _to_device(mb_np)
        ob_t = _to_device(ob_np)
        flat_t = _to_device(flat_np)
        if profile_timing:
            _sync()
            timing["h2d"] += time.perf_counter() - t0
        use_amp = bool(amp and str(device).startswith("cuda"))
        t1 = time.perf_counter() if profile_timing else 0.0
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                own, opp, win_prob, logits = net(mb_t, ob_t, flat_t)
        if profile_timing:
            _sync()
            timing["forward"] += time.perf_counter() - t1
        t2 = time.perf_counter() if profile_timing else 0.0
        # Win-gated leaf value.  This evaluator is IN-PROCESS and is the one the
        # batched / batched_open_loop engines use; mg/al are bound at construction
        # (Fix 2) from cfg.margin_gain / cfg.alpha — no module-global reads.
        margin_val = torch.tanh((own - opp) * mg)
        win_val = 2.0 * win_prob - 1.0
        win_gate = win_val * win_val
        win_gate = win_gate * win_gate                 # win_val**4 (n=4 certainty gate)
        # f32 readback (was .double()): halves D2H bytes for values and the
        # K×3390 logits.  .float() is a no-op for an already-f32 forward and
        # promotes f16 (AMP) up to f32; the Rust tree casts to f64 on entry.
        values = ((1.0 - al) * win_val + al * win_gate * margin_val).reshape(-1)[:n].float().cpu().numpy()
        full = logits[:n].float().cpu().numpy()
        if profile_timing:
            timing["readback"] += time.perf_counter() - t2
            timing["calls"] += 1
        gathered = [full[i][idxs_list[i]] for i in range(len(idxs_list))]
        return values, gathered

    if profile_timing:
        evaluator.timing = timing
    return evaluator


def make_rust_coalescing_evaluator(client):
    """Adapt an InferenceClient (e.g. LocalInferenceService.make_client()) into the
    Rust batched-evaluator contract, COALESCING leaves across concurrent game
    threads into one forward.  Each thread's RustMCTS, while blocked in the
    client's future (event.wait releases the GIL), lets the others submit — so the
    batcher assembles a batch spanning many games.  Returns f32 values + f32
    gathered logits (the Rust tree casts to f64 on entry; f32 halves D2H volume).
    Use this (not make_rust_evaluator) whenever multiple games run concurrently.
    """
    from games.kingdomino.inference_service import make_ipc_batched_evaluator
    base = make_ipc_batched_evaluator(client)

    def evaluator(mbs, obs, flats, idxs_list):
        values, gathered = base(mbs, obs, flats, idxs_list)
        return (np.asarray(values, dtype=np.float32),
                [np.asarray(g, dtype=np.float32) for g in gathered])

    return evaluator


def _select_idx(counts: dict, temperature: float, rng: np.random.Generator) -> int:
    """Pick a joint index from {joint_idx: visit_count}; temperature=0 → greedy.
    Rust-engine counterpart of select_move (which keys on action objects)."""
    idxs = list(counts.keys())
    c = np.array([counts[i] for i in idxs], dtype=np.float64)
    if c.sum() <= 0:
        raise ValueError("All root visit counts are zero; increase n_simulations.")
    if temperature <= 1e-6:
        return idxs[int(c.argmax())]
    w = c ** (1.0 / temperature)
    w /= w.sum()
    return idxs[int(rng.choice(len(idxs), p=w))]


def _rust_idx_to_action(rs_state, joint_idx: int):
    """Map a chosen joint index to the (placement, pick) step args for rs_state,
    via the state's own legal actions (ascending-index order, parallel to
    legal_action_indices)."""
    acts = rs_state.legal_actions()
    idxs = rs_state.legal_action_indices()
    for a, i in zip(acts, idxs):
        if int(i) == joint_idx:
            return a
    raise ValueError(f"joint index {joint_idx} is not legal in the current state.")


def play_selfplay_game_rust(
    rust_mcts, evaluator, *,
    n_simulations: int, n_determinizations: int, temp_moves: int,
    c_puct: float, dirichlet_alpha: float, dirichlet_epsilon: float,
    leaf_batch: int, virtual_loss: int, seed: int,
    py_rng: random.Random, np_rng: np.random.Generator,
    score_scale: float = 160.0, margin_gain: float = 2.0, alpha: float = 0.5,
    iteration: int = 0,
    playout_cfg: Optional[SelfPlayConfig] = None,
) -> Tuple[List[Example], Tuple[int, int]]:
    """Rust-engine self-play game.  Mirrors play_selfplay_game's outputs.

    The game runs in a RustGameState seeded to match GameState.new(seed)'s
    opening position; RustMCTS.search drives play.  PIMC redeterminizes the Rust
    deck per determinization and aggregates root visit counts; root Dirichlet
    noise is applied inside the search.  Training states are the PUBLIC encoding
    (info-set safe — the encoder reads no deck order)."""
    import kingdomino_rust
    game_over = int(Phase.GAME_OVER)

    py_init = GameState.new(seed=seed)
    rs = kingdomino_rust.RustGameState(
        py_init.start_player, list(py_init.deck), list(py_init.current_row))

    records = []  # (mb, ob, flat, pidx, pval, legal_idx, actor)
    move_num = 0
    while rs.phase != game_over:
        if playout_cfg is not None:
            _is_full, move_sims, noise_eps, move_temp_moves, record_example = (
                _choose_playout_profile(playout_cfg, np_rng)
            )
        else:
            move_sims = n_simulations
            noise_eps = dirichlet_epsilon
            move_temp_moves = temp_moves
            record_example = True
        agg: dict = {}
        for _ in range(n_determinizations):
            det = rs.redeterminize(seed=int(np_rng.integers(0, 2**63 - 1)))
            pairs = rust_mcts.search(
                det, evaluator, int(move_sims),
                dirichlet_alpha=dirichlet_alpha, dirichlet_eps=float(noise_eps),
                fpu=0.0, cpuct=c_puct, seed=int(np_rng.integers(0, 2**63 - 1)),
                leaf_batch=leaf_batch, virtual_loss=virtual_loss,
                score_scale=score_scale, margin_gain=margin_gain, alpha=alpha,
            )
            for idx, cnt in pairs:
                agg[int(idx)] = agg.get(int(idx), 0) + int(cnt)

        total = sum(agg.values())
        policy = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)
        for idx, cnt in agg.items():
            policy[idx] = cnt / total
        actor = rs.current_actor()
        if record_example:
            mb, ob, flat = rs.encode(actor)                   # PUBLIC, info-set safe
            mb = np.asarray(mb); ob = np.asarray(ob); flat = np.asarray(flat)
            legal_idx = np.asarray(rs.legal_action_indices(), dtype=np.int32)
            pidx = np.nonzero(policy)[0].astype(np.int32)
            pval = policy[pidx].astype(np.float32)
            records.append((
                mb.astype(np.float16), ob.astype(np.float16), flat.astype(np.float16),
                pidx, pval, legal_idx, actor,
            ))

        temp = _temperature(move_num, move_temp_moves)
        chosen = _select_idx(agg, temp, np_rng)
        placement, pick = _rust_idx_to_action(rs, chosen)
        rs = rs.step(placement, pick)
        move_num += 1

    s0, s1 = rs.scores()
    z0 = math.tanh((s0 - s1) / 30.0)   # = compute_target_z(player=0), sigma=30
    own0, opp0 = float(s0), float(s1)
    # LIMITATION: RustGameState does not expose tiebreaker data (largest
    # territory, total crowns). Score ties fall through to draw (win_target=0.5).
    # The Python path (play_selfplay_game) correctly routes through determine_winner.
    if s0 > s1:
        win0 = 1.0
    elif s1 > s0:
        win0 = 0.0
    else:
        win0 = 0.5  # score tie → call it a draw (cascade unavailable)
    examples = []
    for (mb, ob, flat, pidx, pval, lidx, actor) in records:
        if actor == 0:
            z = z0
            own_s, opp_s, win_t = own0, opp0, win0
        else:
            z = -z0
            own_s, opp_s = opp0, own0
            win_t = (1.0 - win0) if win0 != 0.5 else 0.5
        examples.append(
            Example(mb, ob, flat, pidx, pval, lidx, z, own_s, opp_s, win_t,
                    iteration=iteration)
        )
    return examples, (s0, s1)


def _example_from_rust_tuple(tup, iteration: int = 0) -> Example:
    """Convert BatchedMCTS's sparse training tuple to the existing buffer type.

    Phase 3R: the tuple is now 10 elements — the Rust engine fills own_score,
    opp_score, win_target at game end (score-only win; no tiebreaker cascade in
    Rust, matching play_selfplay_game_rust's documented limitation).  The Rust
    tuple carries no iteration field, so the caller passes it in.
    """
    if len(tup) == 12:
        # Current format: root_stats + trailing actor (0/1). The actor is only
        # needed by the two-net HOF path (_hof_actor_from_rust_tuple); ignored here.
        (mb, ob, flat, pidx, pval, lidx, root_stats,
         z, own_score, opp_score, win_target, _actor) = tup
        root_prior_idx, root_prior_val, root_visit_count = root_stats
    elif len(tup) == 11:
        mb, ob, flat, pidx, pval, lidx, root_stats, z, own_score, opp_score, win_target = tup
        root_prior_idx, root_prior_val, root_visit_count = root_stats
    elif len(tup) == 10:
        mb, ob, flat, pidx, pval, lidx, z, own_score, opp_score, win_target = tup
        root_prior_idx = root_prior_val = root_visit_count = None
    else:
        raise ValueError(f"Unexpected Rust example tuple length: {len(tup)}")
    return Example(
        np.asarray(mb, dtype=np.float16),
        np.asarray(ob, dtype=np.float16),
        np.asarray(flat, dtype=np.float16),
        np.asarray(pidx, dtype=np.int32),
        np.asarray(pval, dtype=np.float32),
        np.asarray(lidx, dtype=np.int32),
        float(z),
        own_score=float(own_score),
        opp_score=float(opp_score),
        win_target=float(win_target),
        root_prior_idx=(None if root_prior_idx is None
                        else np.asarray(root_prior_idx, dtype=np.int32)),
        root_prior_val=(None if root_prior_val is None
                        else np.asarray(root_prior_val, dtype=np.float32)),
        root_visit_count=(None if root_visit_count is None
                          else np.asarray(root_visit_count, dtype=np.int32)),
        iteration=iteration,
    )


def _write_exact_fallback_records(path: str, records: list, *, iteration: int,
                                  engine: str, n_simulations: int,
                                  checkpoint_dir: Optional[str]) -> int:
    """Append exact-solver fallback root states to a JSONL sidecar."""
    if not path or not records:
        return 0
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_id = Path(checkpoint_dir).name if checkpoint_dir else None
    with out.open("a", encoding="utf-8") as f:
        for rec in records:
            row = dict(rec)
            # PyO3 returns Vec<u8> fields (board terrain/crowns) as Python `bytes`,
            # which json.dumps cannot serialize; coerce to a list of ints (matches
            # what endgame_solver_harness.py / RustGameState.from_parts expect).
            for k, v in row.items():
                if isinstance(v, (bytes, bytearray)):
                    row[k] = list(v)
            row.update({
                "iteration": int(iteration),
                "engine": engine,
                "n_simulations": int(n_simulations),
                "checkpoint_dir": checkpoint_dir,
                "run_id": run_id,
            })
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(records)


def _resolve_async_solver_cpus(cfg: SelfPlayConfig) -> tuple[int, int, int]:
    """Return (effective_solver_cpus, effective_game_cpus, total_cpus)."""
    total = max(1, int(os.cpu_count() or 1))
    if int(cfg.solver_cpus) > 0:
        solver_cpus = max(1, min(total, int(cfg.solver_cpus)))
        game_cpus = max(1, total - solver_cpus)
    else:
        max_game_cpus = max(1, total - 1)
        game_cpus = max(1, min(max_game_cpus, int(cfg.game_cpus)))
        solver_cpus = max(1, total - game_cpus)
    return solver_cpus, game_cpus, total


def _double_buffer_loop(make_batched, evaluator, n_games, seed_start):
    """Drive two BatchedMCTS instances (A, B) with one background CPU worker so
    one instance's Rust tree work overlaps the other's GPU forward (Item 17).

    Schedule per tick (the overlap comes from a single-worker thread pool; the
    evaluator releases the GIL during the CUDA forward and step/update release
    it during the Rust work, so they run concurrently):
      1. evaluator(A's leaves)              ← GPU, main; B's CPU (from last tick)
                                              runs in the background concurrently
      2. collect B's background leaves
      3. submit A's CPU work (update+step) to the background
      4. evaluator(B's leaves)              ← GPU, main; A's CPU runs concurrently
      5. collect A's next leaves
      6. submit B's CPU work; it overlaps step (1) of the next tick

    Correctness invariants:
      - update(vals, gath) is always called with the results of the step() that
        produced those leaves — A's results never reach B's update and vice versa.
      - each instance is touched by exactly one thread at a time (main does
        priming + the evaluator over its leaf arrays; the per-instance future
        does that instance's update()+step(); futures never overlap per instance).
      - the evaluator (torch/CUDA) is called from the MAIN thread only.

    Seeds match the single-buffer case exactly: A covers games
    [seed_start, seed_start+games_a), B covers [seed_start+games_a, ...), so
    game i still uses seed seed_start + i.  Returns
    (finished, batch_sizes, ticks, exact_solve_count, exact_tree_solve_count,
     exact_cache_hit_count, exact_fallback_count, exact_solver_secs,
     fast_move_count, full_move_count, recorded_fast_move_count,
     recorded_full_move_count, exact_recorded_move_count, step_sec, eval_sec,
     update_sec, elapsed);
    step_sec/update_sec are summed across both instances and OVERLAP eval_sec
    (their sum can exceed elapsed — that overlap is the speedup).
    """
    games_a = n_games // 2
    games_b = n_games - games_a
    A = make_batched(games_a, seed_start)
    B = make_batched(games_b, seed_start + games_a)

    finished_a: list = []
    finished_b: list = []
    batch_sizes: List[int] = []
    prof = {"step": 0.0, "update": 0.0}   # written only by the active CPU thread
    ticks = 0

    def _timed_step(inst):
        ts = time.perf_counter()
        leaves = inst.step()
        prof["step"] += time.perf_counter() - ts
        return leaves

    def _cpu_work(inst, finished_list, vals, gath):
        """Background-thread work for one instance: scatter the forward results
        and advance, then produce the next leaves (None if now done)."""
        ts = time.perf_counter()
        finished_list.extend(inst.update(vals, gath))
        prof["update"] += time.perf_counter() - ts
        if inst.done():
            return None
        return _timed_step(inst)

    t0 = time.time()
    eval_sec = 0.0

    def _eval(leaves):
        nonlocal eval_sec, ticks
        mb, ob, flat, idxs = leaves
        batch_sizes.append(int(mb.shape[0]))
        ts = time.perf_counter()
        vals, gath = evaluator(mb, ob, flat, idxs)
        eval_sec += time.perf_counter() - ts
        ticks += 1
        return vals, gath

    # Prime both instances on the main thread (n_games >= 2, so each has >=1 game).
    leaves_a = _timed_step(A)
    leaves_b = _timed_step(B)
    a_future = None
    b_future = None

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        while not (A.done() and B.done()):
            # (1) GPU forward for A; B's CPU work (submitted last tick) overlaps.
            vals_a = gath_a = None
            if leaves_a is not None:
                vals_a, gath_a = _eval(leaves_a)

            # (2) Collect B's leaves produced in the background during step (1).
            if b_future is not None:
                leaves_b = b_future.result()
                b_future = None

            # (3) Submit A's CPU work so it overlaps B's GPU forward below.
            if leaves_a is not None:
                a_future = executor.submit(_cpu_work, A, finished_a, vals_a, gath_a)

            # (4) GPU forward for B; A's CPU work runs concurrently.
            vals_b = gath_b = None
            if leaves_b is not None:
                vals_b, gath_b = _eval(leaves_b)

            # (5) Collect A's next leaves (its CPU work ran during step (4)).
            if a_future is not None:
                leaves_a = a_future.result()
                a_future = None

            # (6) Submit B's CPU work; it overlaps step (1) of the next tick.
            if leaves_b is not None:
                b_future = executor.submit(_cpu_work, B, finished_b, vals_b, gath_b)

            if ticks > 2_000_000:
                raise RuntimeError("BatchedMCTS exceeded tick guard")
    finally:
        executor.shutdown(wait=True)

    elapsed = time.time() - t0
    finished = finished_a + finished_b
    exact_solve_count = int(A.exact_solve_count) + int(B.exact_solve_count)
    exact_tree_solve_count = int(A.exact_tree_solve_count) + int(B.exact_tree_solve_count)
    exact_cache_hit_count = int(A.exact_cache_hit_count) + int(B.exact_cache_hit_count)
    exact_fallback_count = int(A.exact_fallback_count) + int(B.exact_fallback_count)
    exact_attempt_deck4_initial_count = (
        int(A.exact_attempt_deck4_initial_count)
        + int(B.exact_attempt_deck4_initial_count))
    exact_attempt_deck4_retry_count = (
        int(A.exact_attempt_deck4_retry_count)
        + int(B.exact_attempt_deck4_retry_count))
    exact_attempt_deck0_count = (
        int(A.exact_attempt_deck0_count) + int(B.exact_attempt_deck0_count))
    exact_fallback_deck4_initial_count = (
        int(A.exact_fallback_deck4_initial_count)
        + int(B.exact_fallback_deck4_initial_count))
    exact_fallback_deck4_retry_count = (
        int(A.exact_fallback_deck4_retry_count)
        + int(B.exact_fallback_deck4_retry_count))
    exact_fallback_deck0_count = (
        int(A.exact_fallback_deck0_count) + int(B.exact_fallback_deck0_count))
    exact_solver_secs = float(A.exact_solver_secs) + float(B.exact_solver_secs)
    fast_move_count = int(A.fast_move_count) + int(B.fast_move_count)
    full_move_count = int(A.full_move_count) + int(B.full_move_count)
    recorded_fast_move_count = (
        int(A.recorded_fast_move_count) + int(B.recorded_fast_move_count))
    recorded_full_move_count = (
        int(A.recorded_full_move_count) + int(B.recorded_full_move_count))
    exact_recorded_move_count = (
        int(A.exact_recorded_move_count) + int(B.exact_recorded_move_count))
    exact_fallback_records = (
        list(A.drain_exact_fallback_records())
        + list(B.drain_exact_fallback_records()))
    return (finished, batch_sizes, ticks,
            exact_solve_count, exact_tree_solve_count, exact_cache_hit_count,
            exact_fallback_count, exact_solver_secs,
            {
                "exact_attempt_deck4_initial_count": exact_attempt_deck4_initial_count,
                "exact_attempt_deck4_retry_count": exact_attempt_deck4_retry_count,
                "exact_attempt_deck0_count": exact_attempt_deck0_count,
                "exact_fallback_deck4_initial_count": exact_fallback_deck4_initial_count,
                "exact_fallback_deck4_retry_count": exact_fallback_deck4_retry_count,
                "exact_fallback_deck0_count": exact_fallback_deck0_count,
            },
            fast_move_count, full_move_count,
            recorded_fast_move_count, recorded_full_move_count,
            exact_recorded_move_count,
            exact_fallback_records,
            prof["step"], eval_sec, prof["update"], elapsed)


def play_selfplay_games_batched(
    net: KingdominoNet,
    cfg: SelfPlayConfig,
    *,
    n_games: int,
    game_seed_start: int,
    iteration: int = 0,
    double_buffer: bool = False,
) -> Tuple[List[List[Example]], List[Tuple[int, int]], dict]:
    """Generate self-play games with a Rust BatchedMCTS.

    double_buffer (Item 17): run TWO independent BatchedMCTS instances (A and B)
    over disjoint halves of the game batch, alternating so one instance's CPU
    tree work (Rust step/update) overlaps the other's GPU forward.  A single
    background thread (one at a time) runs the off-instance's update()+step()
    while the main thread runs the evaluator; both release the GIL during their
    heavy work, so they overlap.  Falls back to the single-buffer loop when
    n_games < 2.  Produces the SAME games (seeding unchanged: game i uses seed
    game_seed_start + i) and the same stats keys as the single-buffer path."""
    if cfg.n_determinizations != 1:
        raise ValueError("--engine batched currently requires --determinizations 1")

    import kingdomino_rust

    evaluator = make_rust_evaluator(
        net, device=cfg.device, amp=cfg.inference_amp,
        pad_to_batch=cfg.eval_pad_to_batch,
        pin_transfer=cfg.pin_transfer,
        profile_timing=cfg.profile_eval_timing,
        margin_gain=cfg.margin_gain, alpha=cfg.alpha,
        compile_net=cfg.compile_net, compile_dynamic=cfg.compile_dynamic,
    )
    n_slots = max(1, int(cfg.batch_slots))
    lb = max(1, int(cfg.leaf_batch))
    effective_solver_cpus, effective_game_cpus, total_cpus = _resolve_async_solver_cpus(cfg)

    def _make_batched(games: int, seed_start: int):
        """Build one BatchedMCTS.  Each instance gets the FULL n_slots (not
        halved) — slots are concurrent positions within an instance, so the
        per-forward GPU batch stays up to n_slots*leaf_batch per instance."""
        return kingdomino_rust.BatchedMCTS(
            n_slots,
            int(games),
            int(seed_start),
            int(cfg.n_simulations),
            leaf_batch=lb,
            virtual_loss=int(cfg.virtual_loss),
            cpuct=float(cfg.c_puct),
            fpu=float(cfg.fpu),
            dirichlet_alpha=float(cfg.dirichlet_alpha),
            dirichlet_eps=float(cfg.dirichlet_epsilon),
            temp_moves=int(cfg.temp_moves),
            open_loop=(cfg.engine == "batched_open_loop"),
            score_scale=float(cfg.score_scale),
            margin_gain=float(cfg.margin_gain),
            alpha=float(cfg.alpha),
            exact_endgame_max_secs=float(cfg.exact_endgame_max_secs),
            exact_policy_mode=str(cfg.exact_policy_mode),
            exact_clamp_delta=float(cfg.exact_clamp_delta),
            async_solve=bool(cfg.async_solve),
            solver_cpus=int(effective_solver_cpus),
            playout_cap_randomization=bool(cfg.playout_cap_randomization),
            full_search_fraction=float(cfg.full_search_fraction),
            fast_move_sims=int(cfg.fast_move_sims),
            record_fast_moves=bool(cfg.record_fast_moves),
            fast_move_dirichlet_eps=float(cfg.fast_move_dirichlet_epsilon),
            fast_move_temp_moves=int(cfg.fast_move_temp_moves),
        )

    use_db = bool(double_buffer)
    if use_db and n_games < 2:
        print("WARNING: --double_buffer requested but n_games < 2; "
              "falling back to single-buffer.")
        use_db = False

    exact_solve_count = 0
    exact_tree_solve_count = 0
    exact_cache_hit_count = 0
    exact_fallback_count = 0
    exact_split_stats = {
        "exact_attempt_deck4_initial_count": 0,
        "exact_attempt_deck4_retry_count": 0,
        "exact_attempt_deck0_count": 0,
        "exact_fallback_deck4_initial_count": 0,
        "exact_fallback_deck4_retry_count": 0,
        "exact_fallback_deck0_count": 0,
    }
    exact_solver_secs = 0.0
    fast_move_count = 0
    full_move_count = 0
    recorded_fast_move_count = 0
    recorded_full_move_count = 0
    exact_recorded_move_count = 0
    exact_fallback_records = []
    if not use_db:
        # ── Single-buffer path (unchanged) ──
        batched = _make_batched(n_games, game_seed_start)
        finished = []
        batch_sizes: List[int] = []
        t0 = time.time()
        step_sec = 0.0
        eval_sec = 0.0
        update_sec = 0.0
        ticks = 0
        while not batched.done():
            ts = time.perf_counter()
            mb, ob, flat, idxs_list = batched.step()
            step_sec += time.perf_counter() - ts
            b = int(mb.shape[0])
            batch_sizes.append(b)
            ts = time.perf_counter()
            values, gathered = evaluator(mb, ob, flat, idxs_list)
            eval_sec += time.perf_counter() - ts
            ts = time.perf_counter()
            finished.extend(batched.update(values, gathered))
            update_sec += time.perf_counter() - ts
            ticks += 1
            if ticks > 2_000_000:
                raise RuntimeError("BatchedMCTS exceeded tick guard")
        elapsed = time.time() - t0
        exact_solve_count = int(batched.exact_solve_count)
        exact_tree_solve_count = int(batched.exact_tree_solve_count)
        exact_cache_hit_count = int(batched.exact_cache_hit_count)
        exact_fallback_count = int(batched.exact_fallback_count)
        exact_split_stats = {
            "exact_attempt_deck4_initial_count": int(
                batched.exact_attempt_deck4_initial_count),
            "exact_attempt_deck4_retry_count": int(
                batched.exact_attempt_deck4_retry_count),
            "exact_attempt_deck0_count": int(batched.exact_attempt_deck0_count),
            "exact_fallback_deck4_initial_count": int(
                batched.exact_fallback_deck4_initial_count),
            "exact_fallback_deck4_retry_count": int(
                batched.exact_fallback_deck4_retry_count),
            "exact_fallback_deck0_count": int(batched.exact_fallback_deck0_count),
        }
        exact_solver_secs = float(batched.exact_solver_secs)
        fast_move_count = int(batched.fast_move_count)
        full_move_count = int(batched.full_move_count)
        recorded_fast_move_count = int(batched.recorded_fast_move_count)
        recorded_full_move_count = int(batched.recorded_full_move_count)
        exact_recorded_move_count = int(batched.exact_recorded_move_count)
        exact_fallback_records = list(batched.drain_exact_fallback_records())
    else:
        # ── Double-buffer path (Item 17) ──
        (finished, batch_sizes, ticks,
         exact_solve_count, exact_tree_solve_count, exact_cache_hit_count,
         exact_fallback_count, exact_solver_secs, exact_split_stats,
         fast_move_count, full_move_count,
         recorded_fast_move_count, recorded_full_move_count,
         exact_recorded_move_count,
         exact_fallback_records,
         step_sec, eval_sec, update_sec, elapsed) = _double_buffer_loop(
            _make_batched, evaluator, n_games, game_seed_start)

    exact_fallback_saved = _write_exact_fallback_records(
        cfg.exact_fallback_positions,
        exact_fallback_records,
        iteration=iteration,
        engine=cfg.engine,
        n_simulations=cfg.n_simulations,
        checkpoint_dir=cfg.checkpoint_dir,
    )

    finished.sort(key=lambda r: int(r[0]))
    all_examples: List[List[Example]] = []
    all_scores: List[Tuple[int, int]] = []
    for _seed, examples, scores in finished:
        all_examples.append(
            [_example_from_rust_tuple(ex, iteration=iteration) for ex in examples])
        all_scores.append((int(scores[0]), int(scores[1])))

    nonzero_batches = [b for b in batch_sizes if b > 0]
    total_evals = int(sum(batch_sizes))
    cap = n_slots * lb
    mean_batch = float(np.mean(nonzero_batches)) if nonzero_batches else 0.0
    stats = {
        "ticks": ticks,
        "elapsed": elapsed,
        "total_evals": total_evals,
        "mean_batch": mean_batch,
        "max_batch_cap": cap,
        "fill_ratio": mean_batch / cap if cap > 0 else 0.0,
        "max_batch_seen": max(nonzero_batches) if nonzero_batches else 0,
        "requests_per_sec": total_evals / elapsed if elapsed > 0 else 0.0,
        "step_sec": step_sec,
        "eval_sec": eval_sec,
        "update_sec": update_sec,
        "exact_solve_count": exact_solve_count,
        "exact_tree_solve_count": exact_tree_solve_count,
        "exact_cache_hit_count": exact_cache_hit_count,
        "exact_fallback_count": exact_fallback_count,
        **exact_split_stats,
        "exact_solver_secs": exact_solver_secs,
        "exact_fallback_positions_saved": exact_fallback_saved,
        "total_cpus": total_cpus,
        "game_cpus": effective_game_cpus,
        "solver_cpus": effective_solver_cpus,
        "solver_cpus_override": int(cfg.solver_cpus) > 0,
        "fast_move_count": fast_move_count,
        "full_move_count": full_move_count,
        "recorded_fast_move_count": recorded_fast_move_count,
        "recorded_full_move_count": recorded_full_move_count,
        "exact_recorded_move_count": exact_recorded_move_count,
    }
    ev_timing = getattr(evaluator, "timing", None)
    if ev_timing:
        stats.update({
            "eval_h2d_sec": float(ev_timing.get("h2d", 0.0)),
            "eval_forward_sec": float(ev_timing.get("forward", 0.0)),
            "eval_readback_sec": float(ev_timing.get("readback", 0.0)),
            "eval_calls": int(ev_timing.get("calls", 0)),
        })
    return all_examples, all_scores, stats


# ─── 3c. Per-iteration logging helpers ─────────────────────────────────────
def _grad_norm(params) -> float:
    """L2 norm of the gradients across a parameter group.

    Cheap — iterates over the existing param.grad tensors after a backward
    pass.  Call it AFTER optimizer.step() but BEFORE the next zero_grad (the
    grads are still populated; train_step zeros at the START of each step, not
    the end).  Params with no .grad (e.g. unused this step) are skipped.
    """
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def _policy_params(net: KingdominoNet) -> list:
    """The policy-head parameters of KingdominoNet.

    The network has no single ``policy_head`` module — the joint policy logits
    are produced by the bilinear stack: flat_policy_mlp + placement_conv +
    special_placement + pick_mlp + no_pick + W (see network.py forward()).  The
    shared trunk and the three scalar heads are excluded.
    """
    params: list = []
    params += list(net.flat_policy_mlp.parameters())
    params += list(net.placement_conv.parameters())
    params.append(net.special_placement)
    params += list(net.pick_mlp.parameters())
    params.append(net.no_pick)
    params.append(net.W)
    return params


def _diag_metrics(net: KingdominoNet, diag_batch) -> Tuple[float, float]:
    """(policy_entropy, win_brier) on a FIXED diagnostic batch.

    Both metrics share the same fixed sample so their trends reflect the net
    changing on identical positions, not batch-to-batch sampling noise.
    policy_entropy is the mean masked-policy entropy (decreasing ⇒ the policy is
    sharpening); win_brier is MSE(win_prob, win_target) — the win head's
    calibration on that consistent set.  Pure diagnostic (no_grad); the caller
    is responsible for restoring train/eval mode as needed.
    """
    mb_d, ob_d, flat_d, _pol_d, legal_mask_d, _z_d, _own_d, _opp_d, win_t_d = diag_batch
    net.eval()
    with torch.no_grad():
        _, _, win_prob_d, logits_d = net(mb_d, ob_d, flat_d)
        logp_d = masked_log_softmax(logits_d, legal_mask_d)
        logp_d = torch.where(legal_mask_d, logp_d, torch.zeros_like(logp_d))
        p_d = torch.exp(logp_d)
        entropy = -(p_d * logp_d).sum(dim=1).mean().item()
        win_brier = F.mse_loss(win_prob_d, win_t_d).item()
    return float(entropy), float(win_brier)


def _derive_log_path(cfg: "SelfPlayConfig") -> str:
    """Resolve the per-iteration JSONL log path (see SelfPlayConfig.log_path)."""
    if cfg.log_path is not None:
        path = cfg.log_path
    elif cfg.checkpoint_dir is not None:
        path = os.path.join(cfg.checkpoint_dir, "training_log.jsonl")
    else:
        path = f"./training_log_{int(time.time())}.jsonl"
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    return path


def _log_row(log_path: str, row: dict) -> None:
    """Append one iteration's metrics as a single JSON line."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compact_summary(it: int, *, sp_games: int, row: dict, trained: bool,
                     buf_n: int, min_buf: int) -> str:
    """One-line, greppable per-iteration summary built from the log row.

    Sections (joined by ' | '): always sp; train/grads/diag only when training
    ran (else a 'skipped (buf n/min)' note); bench only when a benchmark ran.
    """
    parts = [f"iter {it:03d}"]
    parts.append(
        f"sp: {sp_games} games {row['games_per_sec']:.2f}/s "
        f"diff={row['sp_score_diff_mean']:.1f}±{row['sp_score_diff_std']:.1f} "
        f"buf={row['buffer_size']}(age={row['buffer_mean_age']:.1f})")
    if trained:
        parts.append(
            f"train: pol={row['policy_loss']:.3f} own={row['own_loss']:.3f} "
            f"opp={row['opp_loss']:.3f} win={row['win_loss']:.3f} "
            f"brier={row['win_brier']:.3f}/{row['baseline_brier']:.3f}")
        parts.append(
            f"grads: pol={row['grad_norm_policy']:.3f} "
            f"win={row['grad_norm_win']:.3f} own={row['grad_norm_own']:.3f} "
            f"opp={row['grad_norm_opp']:.3f}")
        ent = row['policy_entropy']
        wbd = row['win_brier_diag']
        parts.append(
            f"diag: entropy={ent:.2f} brier_diag="
            + (f"{wbd:.3f}" if wbd is not None else "n/a"))
    else:
        parts.append(f"train: skipped (buf {buf_n}/{min_buf})")
    if row.get("win_brier_endgame") is not None:
        parts.append(
            f"diag_phase: brier=["
            f"{row.get('win_brier_opening', 0) or 0:.3f}/"
            f"{row.get('win_brier_midgame', 0) or 0:.3f}/"
            f"{row.get('win_brier_endgame', 0) or 0:.3f}] "
            f"kl=[{row.get('policy_kl_opening', 0) or 0:.2f}/"
            f"{row.get('policy_kl_midgame', 0) or 0:.2f}/"
            f"{row.get('policy_kl_endgame', 0) or 0:.2f}]")
    if row.get("bench_win_rate") is not None:
        bwb = row.get("bench_win_brier")
        bench = f"bench: win={row['bench_win_rate']:.1%} margin={row['bench_score_margin']:+.1f}"
        if bwb is not None:
            bench += f" brier={bwb:.3f}"
        parts.append(bench)
    if row.get("exact_tree_solve_count"):
        exact_part = (
            f"exact: trees={row['exact_tree_solve_count']} "
            f"hits={row['exact_cache_hit_count']} "
            f"fallback={row['exact_fallback_count']}")
        if row.get("exact_solver_secs") is not None:
            exact_part += f" solver={row['exact_solver_secs']:.1f}s"
        parts.append(exact_part)
    return " | ".join(parts)


def _new_history() -> dict:
    """Fresh per-iteration history dict (shared by the serial + parallel loops).

    Removed vs older builds: ``value_loss`` (the single-head loss, gone since
    Phase 1b) and ``selfplay_score_diff`` (replaced by sp_score_diff_mean).
    Benchmark keys are sparse — only appended on iterations a benchmark runs.
    """
    return {
        # Training losses (per iteration, mean over train_steps)
        "policy_loss":        [],
        "own_loss":           [],
        "opp_loss":           [],
        "win_loss":           [],
        "win_brier":          [],
        "baseline_brier":     [],

        # Gradient norms (per iteration, mean over train_steps)
        "grad_norm_policy":   [],
        "grad_norm_win":      [],
        "grad_norm_own":      [],
        "grad_norm_opp":      [],

        # Self-play stats (per iteration)
        "sp_score_diff_mean": [],
        "sp_score_diff_std":  [],
        "games_per_sec":      [],
        "buffer_size":        [],
        "buffer_mean_age":    [],

        # Diagnostic batch (per iteration, fixed sample)
        "policy_entropy":     [],
        "win_brier_diag":     [],

        # Benchmark (sparse — only when a benchmark runs)
        "benchmark":          [],   # [(iter, win_rate), ...]
        "score_margin":       [],   # [margin, ...] aligned with benchmark
    }


# ─── 4. Training step ─────────────────────────────────────────────────────
def train_step(
    net: KingdominoNet, batch, optimizer, *,
    policy_weight: float = 1.0, lambda_score: float = 0.5,
    lambda_w: float = 0.25, score_scale: float = 160.0,
    grad_clip: float = 1.0,
) -> Tuple[float, float, float, float, float, float]:
    """One optimiser step on a batch.  Returns (policy_loss, own_loss, opp_loss,
    win_loss, win_brier, baseline_brier).

    Four-head loss (replaces the old single value-MSE head):
      - own_loss / opp_loss: MSE of the score heads against the raw final
        scores normalized by score_scale.
      - win_loss: binary cross-entropy of win_prob against the 1/0.5/0 target.
      - policy_loss: masked cross-entropy — the target visit distribution π is
        zero on illegal actions, the masked log-softmax denominator covers
        legal actions only, and illegal entries get a large-but-finite log-prob
        so the 0·(−big) products are 0 rather than NaN.

    Diagnostics (no_grad — do NOT affect the gradient):
      - win_brier: mean((win_prob - win_target)^2) — the proper scoring rule
        for the win head (MSE on the probability).
      - baseline_brier: the trivial constant-predictor Brier at the batch base
        rate = base_rate*(1-base_rate) (Bernoulli variance).  The win head
        beats the trivial baseline when win_brier < baseline_brier; a Brier
        that rises over training signals a frame-conversion / calibration bug.

    Hardening (cheap relative to MCTS, left on permanently):
      - Batch invariants checked before the forward pass: every row has at
        least one legal action, and every policy-target row sums to 1 (D4
        augmentation is a permutation, so this holds to float precision).
      - Loss checked finite before backward, failing loudly on the first bad
        batch instead of silently poisoning the weights.
      - Global grad-norm clipping before the step guards against the occasional
        high-variance batch (noisy early self-play targets).
    """
    mb, ob, flat, policy, legal_mask, z, own_t, opp_t, win_t = batch
    # z is unpacked but unused — kept for buffer format compatibility.
    # Will be removed in a future cleanup pass once tests are updated.

    # ── batch invariants (fast-fail before the forward pass) ──
    if not legal_mask.any(dim=1).all():
        raise ValueError("Batch contains a row with no legal actions.")
    if not torch.allclose(policy.sum(dim=1), torch.ones(policy.shape[0],
                                                        device=policy.device),
                          atol=1e-4):
        raise ValueError("Policy target row does not sum to 1.")

    own_pred, opp_pred, win_prob, logits = net(mb, ob, flat)

    own_norm = own_t / score_scale
    opp_norm = opp_t / score_scale
    own_loss = F.mse_loss(own_pred, own_norm)
    opp_loss = F.mse_loss(opp_pred, opp_norm)
    win_loss = F.binary_cross_entropy(win_prob, win_t)

    # Brier diagnostics (no_grad — purely for logging, not the gradient).
    with torch.no_grad():
        win_brier = F.mse_loss(win_prob, win_t).item()
        base_rate = win_t.mean()
        baseline_brier = (base_rate * (1.0 - base_rate)).item()

    logp = masked_log_softmax(logits, legal_mask)        # (B, 3390)
    # Zero the (very negative) illegal log-probs BEFORE multiplying.  The target
    # π is already zero there, so this changes nothing on-support while making
    # the off-support products exactly 0 rather than 0 * (−huge) — guaranteeing
    # a finite loss regardless of how saturated the illegal logits become.
    logp = torch.where(legal_mask, logp, torch.zeros_like(logp))
    policy_loss = -(policy * logp).sum(dim=1).mean()

    loss = (policy_weight * policy_loss
            + lambda_score * (own_loss + opp_loss)
            + lambda_w * win_loss)

    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"non-finite loss: policy={policy_loss.item()}, "
            f"own={own_loss.item()}, opp={opp_loss.item()}, "
            f"win={win_loss.item()}"
        )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    optimizer.step()
    return (float(policy_loss.item()), float(own_loss.item()),
            float(opp_loss.item()), float(win_loss.item()),
            float(win_brier), float(baseline_brier))


# ─── 5. Benchmark ─────────────────────────────────────────────────────────
class AZPlayer:
    """Adapts AlphaZeroMCTS+PIMC to the bot interface for evaluation matches.

    Greedy play: no root noise, pick the most-visited root action (τ=0).
    """
    def __init__(self, mcts: AlphaZeroMCTS, n_determinizations: int = 1,
                 np_rng: Optional[np.random.Generator] = None):
        self.mcts = mcts
        self.n_determinizations = n_determinizations
        self._np_rng = np_rng or np.random.default_rng()

    def choose_action(self, state, actions, rng=None):
        py_rng = rng or random.Random()
        vc, _ = run_pimc(self.mcts, state, py_rng,
                         n_determinizations=self.n_determinizations,
                         add_noise=False, np_rng=self._np_rng)
        return select_move(vc, 0.0, self._np_rng)


class OpenLoopAZPlayer:
    """Adapts OpenLoopMCTS to the bot interface for evaluation matches.

    Greedy play (no root noise, τ=0).  Open-loop averages over deck orders
    internally, so there is no n_determinizations argument — run_pimc_open_loop
    is called directly.
    """
    def __init__(self, mcts: OpenLoopMCTS,
                 np_rng: Optional[np.random.Generator] = None):
        self.mcts = mcts
        self._np_rng = np_rng or np.random.default_rng()

    def choose_action(self, state, actions, rng=None):
        vc, _, _ = run_pimc_open_loop(
            self.mcts, state,
            add_noise=False, rng=self._np_rng,
        )
        return select_move(vc, 0.0, self._np_rng)


def benchmark_vs(
    az_player, opponent, n_seeds: int, *, seed: int = 0, verbose: bool = False,
) -> dict:
    """Paired-seed match (each deck played twice, sides swapped).  Returns AZ
    win counts.  Total games = 2 * n_seeds."""
    az_wins = opp_wins = draws = 0
    games = 0
    margin_sum = 0.0   # Σ (az_score - opp_score) over benchmark games
    for i in range(n_seeds):
        for az_is_p0 in (True, False):
            p0, p1 = (az_player, opponent) if az_is_p0 else (opponent, az_player)
            state = GameState.new(seed=seed + i)
            rng = random.Random(seed * 7919 + i * 2 + int(az_is_p0))
            while state.phase != Phase.GAME_OVER:
                bot = p0 if state.current_actor == 0 else p1
                state = state.step(bot.choose_action(state, state.legal_actions(), rng=rng))
            # Route through the authoritative cascade (score → largest
            # territory → total crowns → draw). A raw score comparison would
            # mis-credit genuine score ties that the tiebreaker resolves.
            winner = determine_winner(state)
            az_idx = 0 if az_is_p0 else 1
            if winner is None:
                draws += 1
            elif winner == az_idx:
                az_wins += 1
            else:
                opp_wins += 1
            scores = state.scores()
            margin_sum += scores[az_idx] - scores[1 - az_idx]
            games += 1
        if verbose and (i + 1) % max(1, n_seeds // 5) == 0:
            print(f"    bench seed {i+1}/{n_seeds}: az={az_wins} draw={draws} opp={opp_wins}")
    return {"az_wins": az_wins, "opp_wins": opp_wins, "draws": draws,
            "n_games": games, "az_win_rate": az_wins / max(1, games),
            "mean_margin": margin_sum / max(1, games)}


def benchmark_vs_rust(
    net: KingdominoNet, cfg: "SelfPlayConfig", n_seeds: int, *,
    seed: int = 0, opponent=None, opponent_net: Optional[KingdominoNet] = None,
    opp_rng_seed: int = 12345, verbose: bool = False,
) -> dict:
    """Fast Rust-backed benchmark for the ``batched_open_loop`` engine.

    Drop-in replacement for ``benchmark_vs`` when the engine is batched_open_loop:
    same paired-seed match (sides swapped), same return dict.  The Python
    ``OpenLoopMCTS`` benchmark player is ~50x too slow because its per-simulation
    tree descent is pure Python; this routes the AZ player's moves through
    ``kingdomino_rust.RustMCTS`` (Rust tree descent + the GPU forward via
    ``make_rust_evaluator``) instead.

    WHY A DEDICATED FUNCTION, NOT A ``choose_action`` PLAYER:
      Neither Rust search API can search an arbitrary mid-game position handed in
      from Python.  ``RustGameState`` has only a fresh-game constructor (no
      mid-game injection), and ``BatchedMCTS`` plays *complete* self-play games
      from a seed (it exposes ``step``/``update``/``done`` — no per-move
      interface).  So a Rust search must be driven from a ``RustGameState`` built
      from the seed and stepped from the start.  This plays each benchmark game in
      LOCKSTEP: a Python ``GameState`` (drives the opponent bot + the
      ``determine_winner`` cascade) and a mirrored ``RustGameState`` stepped
      move-for-move via the shared joint action index — the same lockstep used by
      ``test_rust_augment``.  A ``choose_action(state)`` player cannot do this:
      it never sees the opponent's moves, so it cannot keep a Rust mirror in sync.

    SEARCH TYPE: ``RustMCTS`` is CLOSED-loop (the only fast per-move Rust search).
    The root is redeterminized before each search for info-set fairness
    (single-determinization PIMC — the exact search the ``rust`` engine
    self-plays with).  This differs from the training OPEN-loop search (which
    resamples the deck per simulation); an open-loop per-move Rust search would
    require a Rust-side addition.  As a consistent, fast relative-strength metric
    tracked across iterations, the closed-loop proxy is appropriate.

    ``opponent_net`` (optional): if given, the opponent also plays via RustMCTS
    with that frozen net (eval-vs-checkpoint); otherwise ``opponent`` (default
    ``GreedyBot``) chooses on the Python state.
    """
    import kingdomino_rust as kr
    az_eval = make_rust_evaluator(net, device=cfg.device,
                                  margin_gain=cfg.margin_gain, alpha=cfg.alpha)
    opp_eval = None
    if opponent_net is not None:
        opp_eval = make_rust_evaluator(opponent_net, device=cfg.device,
                                       margin_gain=cfg.margin_gain, alpha=cfg.alpha)
    if opponent is None:
        opponent = GreedyBot()
    rust_mcts = kr.RustMCTS()

    def _search_idx(rs, evaluator, np_rng) -> int:
        """Redeterminize the root, run one RustMCTS search, return the greedy
        (most-visited, tau=0) joint index."""
        det = rs.redeterminize(seed=int(np_rng.integers(0, 2**63 - 1)))
        pairs = rust_mcts.search(
            det, evaluator, cfg.benchmark_sims,
            dirichlet_alpha=cfg.dirichlet_alpha, dirichlet_eps=0.0,   # greedy: no root noise
            fpu=cfg.fpu, cpuct=cfg.c_puct,
            seed=int(np_rng.integers(0, 2**63 - 1)),
            leaf_batch=cfg.leaf_batch, virtual_loss=cfg.virtual_loss,
            score_scale=cfg.score_scale, margin_gain=cfg.margin_gain, alpha=cfg.alpha,
        )
        return int(max(pairs, key=lambda kv: kv[1])[0])

    az_wins = opp_wins = draws = 0
    games = 0
    margin_sum = 0.0
    for i in range(n_seeds):
        for az_is_p0 in (True, False):
            py = GameState.new(seed=seed + i)
            rs = kr.RustGameState(py.start_player, list(py.deck), list(py.current_row),
                                  py.config.harmony, py.config.middle_kingdom)
            az_idx = 0 if az_is_p0 else 1
            az_rng = np.random.default_rng(seed * 7919 + i * 2 + int(az_is_p0))
            opp_np_rng = np.random.default_rng(opp_rng_seed + i * 2 + int(az_is_p0))
            opp_bot_rng = random.Random(seed * 104729 + i * 2 + int(az_is_p0))
            while py.phase != Phase.GAME_OVER:
                if py.current_actor == az_idx:
                    joint = _search_idx(rs, az_eval, az_rng)
                elif opp_eval is not None:
                    joint = _search_idx(rs, opp_eval, opp_np_rng)
                else:
                    action = opponent.choose_action(py, py.legal_actions(), rng=opp_bot_rng)
                    joint = int(encode_action(action, py))
                # Step BOTH states with the SAME joint index, keeping the mirror
                # exact: _rust_idx_to_action / decode_action both resolve `joint`
                # against the (identical, public) current state.
                placement, pick = _rust_idx_to_action(rs, joint)
                rs = rs.step(placement, pick)
                py = py.step(decode_action(joint, py))
            winner = determine_winner(py)
            if winner is None:
                draws += 1
            elif winner == az_idx:
                az_wins += 1
            else:
                opp_wins += 1
            scores = py.scores()
            margin_sum += scores[az_idx] - scores[1 - az_idx]
            games += 1
        if verbose and (i + 1) % max(1, n_seeds // 5) == 0:
            print(f"    bench seed {i+1}/{n_seeds}: az={az_wins} draw={draws} opp={opp_wins}")
    return {"az_wins": az_wins, "opp_wins": opp_wins, "draws": draws,
            "n_games": games, "az_win_rate": az_wins / max(1, games),
            "mean_margin": margin_sum / max(1, games)}


# ─── 6. The self-play training loop ───────────────────────────────────────
def make_mcts(net: KingdominoNet, cfg: SelfPlayConfig, n_simulations: int) -> AlphaZeroMCTS:
    # The batched evaluator backs leaf_batch>1 (one forward over N leaves); it is
    # never called when leaf_batch=1 (that path uses the single evaluator), so
    # building it unconditionally is harmless and keeps the seam ready.
    return AlphaZeroMCTS(
        make_serial_evaluator(net, device=cfg.device,
                              margin_gain=cfg.margin_gain, alpha=cfg.alpha),
        batched_evaluator=make_batched_evaluator(net, device=cfg.device,
                              margin_gain=cfg.margin_gain, alpha=cfg.alpha),
        c_puct=cfg.c_puct,
        n_simulations=n_simulations,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_epsilon=cfg.dirichlet_epsilon,
        fpu=cfg.fpu,
        virtual_loss=cfg.virtual_loss,
        score_scale=cfg.score_scale,
        margin_gain=cfg.margin_gain,
        alpha=cfg.alpha,
    )


def make_open_loop_mcts(
    net: KingdominoNet, cfg: SelfPlayConfig, n_simulations: int,
) -> OpenLoopMCTS:
    """Build an OpenLoopMCTS instance from config.

    No batched_evaluator — open-loop simulation is inherently serial
    (each sim follows a different concrete state path).  Throughput comes
    from the Rust port (Phase 3R), not Python leaf-batching.

    n_determinizations is intentionally not forwarded: OpenLoopMCTS
    averages over deck orders internally (one fresh determinization per
    simulation), so the outer PIMC loop is redundant and should not be
    applied on top.  Callers must use run_pimc_open_loop, not run_pimc.
    """
    return OpenLoopMCTS(
        make_serial_evaluator(net, device=cfg.device,
                              margin_gain=cfg.margin_gain, alpha=cfg.alpha),
        c_puct=cfg.c_puct,
        n_simulations=n_simulations,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_epsilon=cfg.dirichlet_epsilon,
        fpu=cfg.fpu,
        virtual_loss=cfg.virtual_loss,
        score_scale=cfg.score_scale,
        margin_gain=cfg.margin_gain,
        alpha=cfg.alpha,
        exact_endgame_max_secs=cfg.exact_endgame_max_secs,
    )


def save_checkpoint(path: str, net: KingdominoNet, cfg: SelfPlayConfig,
                    iteration: int, history: dict,
                    run_manifest: Optional[dict] = None) -> None:
    torch.save({
        "model_state": net.state_dict(),
        "kind": "alphazero_selfplay",
        "policy_head_trained": True,
        "checkpoint_version": KingdominoNet.checkpoint_version,
        "iteration": iteration,
        # vars(cfg) records every SelfPlayConfig field — incl. margin_gain/alpha,
        # so the leaf-value formula is recoverable from the checkpoint alone.
        "config": vars(cfg),
        "history": history,
        "run_manifest": run_manifest or {},
    }, path)


def validate_checkpoint_config(ckpt: dict, cfg: SelfPlayConfig) -> None:
    """Warn if the loaded checkpoint's config differs from the current
    SelfPlayConfig on fields that affect training correctness.

    Called after loading a checkpoint to resume training. Mismatches on
    architecture fields (channels, blocks) are errors; mismatches on
    hyperparameter fields are warnings.
    """
    saved = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not saved:
        print("WARNING: checkpoint has no config dict — cannot validate.")
        return

    # Hard errors: architecture must match exactly.  checkpoint_version lives on
    # the network class, not the config; compare against KingdominoNet's value.
    for field in ("channels", "blocks"):
        if field in saved and saved[field] != getattr(cfg, field, None):
            raise ValueError(
                f"Checkpoint config mismatch on '{field}': "
                f"saved={saved[field]}, current={getattr(cfg, field)}"
            )
    saved_ver = ckpt.get("checkpoint_version", saved.get("checkpoint_version"))
    if saved_ver is not None and saved_ver != KingdominoNet.checkpoint_version:
        raise ValueError(
            f"Checkpoint version mismatch: saved={saved_ver}, "
            f"current={KingdominoNet.checkpoint_version}"
        )

    # Warnings: hyperparameter drift is allowed but should be visible.
    warn_fields = ("n_simulations", "lr", "lambda_score", "lambda_w",
                   "margin_gain", "alpha", "batch_slots", "leaf_batch")
    for field in warn_fields:
        if field in saved and saved[field] != getattr(cfg, field, None):
            print(f"WARNING: checkpoint config '{field}' differs: "
                  f"saved={saved[field]}, current={getattr(cfg, field)}")


def load_generator_net(path: str, cfg: SelfPlayConfig) -> KingdominoNet:
    """Load the promoted self-play generator with the learner architecture."""
    ckpt = torch.load(path, map_location=cfg.device)
    validate_checkpoint_config(ckpt, cfg)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    gen = KingdominoNet(
        channels=cfg.channels,
        blocks=cfg.blocks,
        bilinear_dim=cfg.bilinear_dim,
        score_scale=cfg.score_scale,
    ).to(cfg.device)
    gen.load_state_dict(state)
    gen.eval()
    return gen


@dataclass
class GeneratorState:
    """Current self-play generator selection.

    Phase 1 keeps this as bookkeeping around existing behavior. Later phases
    will update it after soft-gate promotion/revert/probation decisions.
    """

    mode: str
    net: KingdominoNet
    source: str
    checkpoint_path: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    action: str = "initial"
    baseline_net: Optional[KingdominoNet] = None
    baseline_source: Optional[str] = None
    baseline_checkpoint_path: Optional[str] = None
    baseline_sha256: Optional[str] = None


def _effective_generator_mode(cfg: SelfPlayConfig) -> str:
    mode = str(cfg.selfplay_generator_mode or "latest").strip().lower()
    if mode not in GENERATOR_MODES:
        raise ValueError(
            f"unknown self-play generator mode {cfg.selfplay_generator_mode!r}; "
            f"expected one of {', '.join(GENERATOR_MODES)}"
        )
    if cfg.gated_selfplay:
        if mode not in ("latest", "strict_gate"):
            raise ValueError(
                "--gated_selfplay is compatible only with the legacy latest/"
                "strict_gate generator mode; use --selfplay_generator_mode "
                "directly for newer modes"
            )
        return "strict_gate"
    return mode


def _init_generator_state(
    cfg: SelfPlayConfig,
    learner_net: KingdominoNet,
    *,
    verbose: bool,
) -> GeneratorState:
    mode = _effective_generator_mode(cfg)
    if mode in ("current_best", "strict_gate", "soft_gate"):
        if not cfg.current_best_path or not os.path.exists(cfg.current_best_path):
            flag = "--gated_selfplay" if mode == "strict_gate" else f"--selfplay_generator_mode {mode}"
            raise FileNotFoundError(
                f"{flag} requires --current_best_path to exist: "
                f"{cfg.current_best_path!r}"
            )
        checkpoint_path = str(cfg.current_best_path)
        baseline_net = load_generator_net(checkpoint_path, cfg)
        checkpoint_sha = sha256_file(checkpoint_path)
        if mode == "soft_gate":
            state = GeneratorState(
                mode=mode,
                net=learner_net,
                source="learner_latest",
                baseline_net=baseline_net,
                baseline_source=checkpoint_path,
                baseline_checkpoint_path=checkpoint_path,
                baseline_sha256=checkpoint_sha,
            )
            if verbose:
                print(f"Self-play generator (soft_gate): learner_latest "
                      f"(baseline={checkpoint_path})")
            return state
        state = GeneratorState(
            mode=mode,
            net=baseline_net,
            source=checkpoint_path,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            baseline_net=baseline_net,
            baseline_source=checkpoint_path,
            baseline_checkpoint_path=checkpoint_path,
            baseline_sha256=checkpoint_sha,
        )
        if verbose:
            print(f"Self-play generator ({mode}): {checkpoint_path}")
        return state

    state = GeneratorState(
        mode=mode,
        net=learner_net,
        source="learner_latest",
    )
    if verbose and mode == "soft_gate":
        print("Self-play generator (soft_gate): learner_latest")
    return state


def _generator_action_after_promotion_check(
    *,
    mode: str,
    match,
    promotion_passed: bool,
    revert_win_rate: float,
) -> str:
    if promotion_passed:
        return "promote"
    if mode == "soft_gate":
        if match is not None and float(match.win_rate) < float(revert_win_rate):
            return "revert"
        return "probation"
    return "reject"


def _run_elo_rating(
    checkpoint_path: str,
    checkpoint_name: str,
    cfg: "SelfPlayConfig",
    verbose: bool = True,
) -> Optional[dict]:
    """Rate a saved checkpoint against the active anchor pool via elo_rating.py.

    Returns {"elo_rating", "elo_stderr", "elo_n_games"} or None if rating is
    skipped (missing module/anchors/active anchors) or fails.  ALL errors are
    caught and printed as warnings — an Elo failure must never crash training.
    The import is deferred so elo_rating.py is an optional dependency.
    """
    try:
        from games.kingdomino.elo_rating import (
            EloConfig, load_anchors, rate_checkpoint,
        )
    except ImportError as e:
        if verbose:
            print(f"  [elo] skipped: {e}")
        return None

    try:
        anchors_path = cfg.elo_anchors or str(
            Path(__file__).parent / "elo_anchors.csv")
        db_path = cfg.elo_db or "elo_db.json"
        games_path = cfg.elo_games_log or "elo_games.jsonl"

        if not os.path.exists(anchors_path):
            if verbose:
                print(f"  [elo] skipped: anchors not found at {anchors_path}")
            return None

        elo_cfg = EloConfig(
            anchors_csv=anchors_path,
            db_path=db_path,
            games_path=games_path,
            games_per_anchor=cfg.elo_games_per_anchor,
            sims=cfg.elo_sims,
            device=cfg.device,
            n_slots=cfg.batch_slots,
            leaf_batch=cfg.leaf_batch,
            c_puct=cfg.c_puct,
            margin_gain=cfg.margin_gain,
            alpha=cfg.alpha,
        )

        anchors = load_anchors(anchors_path)
        active_anchors = [a for a in anchors if a.is_active]
        if not active_anchors:
            if verbose:
                print("  [elo] skipped: no active anchors")
            return None

        if verbose:
            print(f"  [elo] rating {checkpoint_name} vs "
                  f"{len(active_anchors)} anchors at sims={cfg.elo_sims}...",
                  flush=True)

        t0 = time.time()
        rating, stderr, n_games = rate_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
            cfg=elo_cfg,
            anchors=active_anchors,
        )
        elapsed = time.time() - t0
        if verbose:
            print(f"  [elo] {checkpoint_name}: Elo {rating:.0f} +/- {stderr:.0f} "
                  f"({n_games} games, {elapsed:.0f}s)", flush=True)
        return {"elo_rating": rating, "elo_stderr": stderr,
                "elo_n_games": n_games}

    except Exception as e:
        if verbose:
            print(f"  [elo] WARNING: rating failed: {e}")
        return None


def _run_smart_elo_rating(
    *,
    checkpoint_path: str,
    checkpoint_name: str,
    cfg: "SelfPlayConfig",
    reason: str,
    verbose: bool = True,
) -> Optional[dict]:
    db_path = cfg.elo_db or "elo_db.json"
    try:
        from games.kingdomino.elo_rating import load_db
        db = load_db(db_path)
    except Exception:
        db = {"checkpoints": {}}
    existing = db.get("checkpoints", {}).get(checkpoint_name)
    if existing is not None:
        if verbose:
            print(f"  [elo] smart Elo skipped: {checkpoint_name} already rated")
        return {
            "elo_rating": existing.get("rating"),
            "elo_stderr": existing.get("rating_stderr"),
            "elo_n_games": existing.get("n_games"),
            "smart_elo_reason": reason,
            "smart_elo_name": checkpoint_name,
            "smart_elo_skipped": "already_rated",
        }
    smart_cfg = replace(
        cfg,
        elo_games_per_anchor=int(cfg.smart_elo_games_per_anchor),
        elo_sims=int(cfg.smart_elo_sims),
    )
    result = _run_elo_rating(
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_name,
        cfg=smart_cfg,
        verbose=verbose,
    )
    if result is not None:
        result = dict(result)
        result["smart_elo_reason"] = reason
        result["smart_elo_name"] = checkpoint_name
    return result


def _rolling_average_state(checkpoint_dir: str, it: int, k: int):
    """Average the model_state of the last k iteration checkpoints
    (iter_{it-k+1:04d}.pt .. iter_{it:04d}.pt; missing files skipped).

    Returns (state_dict_f32, n_averaged). n_averaged < 2 means there was
    nothing to average — caller falls back to the raw learner. Same running
    float64 mean as average_checkpoints.py; GroupNorm nets average cleanly
    (no BatchNorm running stats). CPU-only and light (~6MB per checkpoint)."""
    mean_state: dict = {}
    n = 0
    for j in range(max(1, it - k + 1), it + 1):
        cp = Path(checkpoint_dir) / f"iter_{j:04d}.pt"
        if not cp.exists():
            continue
        # Same-run checkpoints (save_checkpoint) always carry "model_state".
        sd = torch.load(cp, map_location="cpu")["model_state"]
        n += 1
        if n == 1:
            for key, v in sd.items():
                mean_state[key] = v.double() if v.is_floating_point() else v.clone()
        else:
            for key, v in sd.items():
                if v.is_floating_point():
                    mean_state[key] += (v.double() - mean_state[key]) / n
                else:
                    mean_state[key] = v.clone()
    out = {key: (v.float() if v.is_floating_point() else v)
           for key, v in mean_state.items()}
    return out, n


def run_self_play_training(cfg: SelfPlayConfig, verbose: bool = True) -> dict:
    """Run the serial self-play loop.  Returns the trained net and history."""
    configure_torch_performance(cfg)
    # Fix 2: the leaf-value blend params (cfg.margin_gain / cfg.alpha) and the
    # terminal-value params (cfg.score_scale) are now bound at MCTS / evaluator /
    # BatchedMCTS construction time (make_mcts / make_open_loop_mcts /
    # make_rust_evaluator / play_selfplay_games_batched), so the old
    # `mcts_az.MARGIN_GAIN = cfg.margin_gain` global override is gone — that
    # eliminates the multiprocessing fragility (spawned workers re-importing the
    # module fresh).
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    net = KingdominoNet(channels=cfg.channels, blocks=cfg.blocks,
                        bilinear_dim=cfg.bilinear_dim,
                        score_scale=cfg.score_scale).to(cfg.device)
    # Iteration the warm-start checkpoint reached (0 = fresh run).  Used only as
    # the reference point for warm_buffer staleness filtering — the training loop
    # below always counts iterations from 1.
    start_iteration = 0
    if cfg.warm_start_path:
        ckpt = torch.load(cfg.warm_start_path, map_location=cfg.device)
        validate_checkpoint_config(ckpt, cfg)   # error on arch drift, warn on hparams
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(state)
        if isinstance(ckpt, dict):
            start_iteration = int(ckpt.get("iteration", 0))
        if verbose:
            print(f"Warm-started from {cfg.warm_start_path} "
                  f"(iteration={start_iteration})")

    generator_state = _init_generator_state(cfg, net, verbose=verbose)
    selfplay_net = generator_state.net
    selfplay_source = generator_state.source
    # Run8: count CONSECUTIVE gate reverts for --revert_reset_after.
    consecutive_reverts = 0
    # Skip promotion checks until the first real training pass: during buffer
    # warmup the learner is byte-identical to its warm start, so a gate match
    # is a guaranteed ~50% self-match (~25-30 min of GPU each at run8 power).
    has_trained_ever = False

    buffer = ReplayBuffer(cfg.buffer_capacity, n_sample_workers=cfg.sample_workers)
    # Pre-load a previously saved replay buffer (warm start of the DATA, not just
    # the weights).  Done after the buffer is built and before the first
    # iteration, so iteration 1 already has examples to train on.
    if cfg.warm_buffer:
        if not cfg.warm_start_path:
            print("WARNING: --warm_buffer provided without --warm_start; "
                  "buffer will be loaded but network weights are random.")
        n = buffer.load(
            cfg.warm_buffer,
            current_iteration=start_iteration,
            max_staleness=cfg.warm_buffer_max_staleness,
        )
        print(f"Warm buffer: {n} examples pre-loaded "
              f"(start_iteration={start_iteration})")
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    np_rng = np.random.default_rng(cfg.seed)   # used by training/benchmark only
    history = _new_history()
    log_path = _derive_log_path(cfg)
    run_manifest = None
    if cfg.checkpoint_dir:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        run_manifest = initialize_run_manifest(
            cfg, cfg.checkpoint_dir, log_path=log_path, net=net)
    if verbose:
        print(f"Per-iteration log: {log_path}")
        if run_manifest is not None:
            print(f"Run manifest: {run_manifest['run_manifest_path']}")
    game_seed = cfg.seed * 1_000_003  # disjoint from benchmark seeds

    # EARLY STOPPING CRITERIA (defined pre-launch per plan §5.3):
    # - Investigate if win Brier NOT improving after 20 iters
    #   (check lambda_w, frame conversion in mcts_az.py MARGIN_GAIN/ALPHA).
    # - Investigate if policy entropy NOT decreasing after 20 iters
    #   (check open-loop correctness, sims count).
    # - Plateau: win rate vs GreedyBot within ±5% for 10 consecutive iters
    #   AND all losses still decreasing → continue.
    # - Hard stop: win rate vs GreedyBot within ±5% for 10 consecutive iters
    #   AND losses flat → stop.
    # These are monitoring criteria, not automated — check the history dict.

    # Fixed probe batch for the policy-entropy metric: sampled once (the first
    # iteration that trains) and reused every iteration, so the entropy trend
    # measures the policy sharpening on identical positions rather than batch-to-
    # batch sampling noise.  Its own deterministic RNG keeps np_rng (and thus the
    # training/benchmark stream) unperturbed.
    diag_entropy_batch = None
    diag_rng = np.random.default_rng(cfg.seed + 7919)

    # Accumulated structured log rows for the empirical alpha-transition trigger
    # (check_alpha_transition needs the recent win_brier_endgame / baseline ratios).
    _diag_rows: List[dict] = []
    schedules = _compiled_schedules(cfg)
    has_active_schedules = any(bool(schedule) for schedule in schedules.values())

    it = 0   # defined before the loop so the finally block's final-rating is safe
    try:
        for it in range(1, cfg.n_iterations + 1):
            iter_cfg = _active_config_for_iteration(cfg, schedules, it)
            _apply_optimizer_schedule(optimizer, iter_cfg.lr)
            _apply_buffer_capacity(buffer, iter_cfg.buffer_capacity)
            if verbose:
                print(f"\n{'='*60}\nIteration {it}/{cfg.n_iterations}\n{'='*60}")
                if has_active_schedules:
                    print("  schedule: "
                          f"lr={iter_cfg.lr:g} sims={iter_cfg.n_simulations} "
                          f"alpha={iter_cfg.alpha:g} games={iter_cfg.games_per_iteration} "
                          f"train_steps={iter_cfg.train_steps_per_iteration} "
                          f"fast_frac={iter_cfg.fast_game_fraction:g}")

            # Per-iteration metrics for the structured log; None = not computed
            # this iteration (training skipped / no benchmark).  Filled below.
            trained = False
            n_endgame_in_batch = None  # set on diag iterations (oversample check)
            pol_m = own_m = opp_m = win_m = None
            win_brier_m = baseline_brier_m = None
            gn_pol = gn_win = gn_own = gn_opp = None
            entropy = win_brier_diag = None
            bench_win_rate = bench_score_margin = bench_win_brier = None
            elo_result = None   # set by the Elo block if elo_every triggers
            smart_elo_result = None
            smart_elo_triggered = False
            smart_elo_reason = None
            smart_elo_name = None
            promotion_result = None
            hof_added_entry = None

            # ── 1. Self-play ──
            generation_net = generator_state.net
            generation_net.eval()
            if iter_cfg.engine == "rust":
                import kingdomino_rust
                rust_mcts = kingdomino_rust.RustMCTS()
                rust_eval = make_rust_evaluator(generation_net, device=iter_cfg.device, amp=iter_cfg.inference_amp,
                                                margin_gain=iter_cfg.margin_gain, alpha=iter_cfg.alpha,
                                                compile_net=iter_cfg.compile_net,
                                                compile_dynamic=iter_cfg.compile_dynamic)
            elif iter_cfg.engine in ("batched", "batched_open_loop"):
                pass
            elif iter_cfg.engine == "open_loop":
                ol_mcts = make_open_loop_mcts(generation_net, iter_cfg, iter_cfg.n_simulations)
            else:  # "python"
                mcts = make_mcts(generation_net, iter_cfg, iter_cfg.n_simulations)
            t0 = time.time()
            diffs = []
            batched_stats = None
            policy_prune_stats = {
                "policy_pruned_actions": 0,
                "policy_pruned_mass": 0.0,
                "policy_pruned_examples": 0,
                "forced_pruned_actions": 0,
                "forced_pruned_mass": 0.0,
                "forced_pruned_examples": 0,
                "forced_subtracted_visits": 0.0,
                "forced_missing_stats_examples": 0,
            }
            hof_stats = {
                "enabled": False,
                "games": 0,
                "trainable_examples": 0,
                "opponent": None,
                "opponent_sha256": None,
                "mean_diff": None,
            }
            total_games = int(iter_cfg.games_per_iteration)
            hof_games = 0
            hof_entry: Optional[HOFEntry] = None
            if (float(iter_cfg.hof_fraction) > 0.0
                    and it >= int(iter_cfg.hof_start_iter)):
                requested = int(round(
                    total_games * max(0.0, min(1.0, float(iter_cfg.hof_fraction)))))
                if requested > 0:
                    entries = read_hof_index(iter_cfg.hof_dir)
                    hof_rng = random.Random(iter_cfg.seed + it * 100_003)
                    hof_entry = sample_hof_entry(
                        entries, rng=hof_rng,
                        weights=iter_cfg.hof_sample_weights)
                    if hof_entry is None:
                        if verbose:
                            print(f"  HOF: no entries in {iter_cfg.hof_dir}; "
                                  "running normal self-play only")
                    else:
                        hof_games = max(1, min(total_games, requested))
                        hof_stats["enabled"] = True
                        hof_stats["games"] = hof_games
                        hof_stats["opponent"] = hof_entry.path
                        hof_stats["opponent_sha256"] = hof_entry.sha256
            normal_games = total_games - hof_games
            if iter_cfg.playout_cap_randomization:
                fast_games = 0
            else:
                fast_games = int(round(
                    normal_games
                    * max(0.0, min(1.0, iter_cfg.fast_game_fraction))))
                fast_games = max(0, min(normal_games, fast_games))
            full_games = normal_games - fast_games
            if iter_cfg.engine in ("batched", "batched_open_loop"):
                all_examples, all_scores = [], []
                stat_parts = []
                if fast_games:
                    fast_cfg = replace(iter_cfg, n_simulations=max(1, int(iter_cfg.fast_game_sims)))
                    ex_fast, sc_fast, st_fast = play_selfplay_games_batched(
                        generation_net, fast_cfg, n_games=fast_games,
                        game_seed_start=game_seed, iteration=it,
                        double_buffer=fast_cfg.double_buffer,
                    )
                    all_examples.extend(ex_fast); all_scores.extend(sc_fast)
                    stat_parts.append(st_fast)
                    if fast_cfg.policy_target_pruning or fast_cfg.forced_playout_subtraction:
                        ps = _prune_examples_policy_targets(
                            ex_fast, total_visits=max(1, int(fast_cfg.n_simulations)),
                            skip_exact=fast_cfg.exact_endgame_max_secs > 0.0,
                            one_visit_pruning=fast_cfg.policy_target_pruning,
                            forced_playout_subtraction=fast_cfg.forced_playout_subtraction,
                            forced_playout_k=fast_cfg.forced_playout_k)
                        for k, v in ps.items():
                            policy_prune_stats[k] += v
                    game_seed += fast_games
                if full_games:
                    ex_full, sc_full, st_full = play_selfplay_games_batched(
                        generation_net, iter_cfg, n_games=full_games,
                        game_seed_start=game_seed, iteration=it,
                        double_buffer=iter_cfg.double_buffer,
                    )
                    all_examples.extend(ex_full); all_scores.extend(sc_full)
                    stat_parts.append(st_full)
                    if iter_cfg.policy_target_pruning or iter_cfg.forced_playout_subtraction:
                        ps = _prune_examples_policy_targets(
                            ex_full, total_visits=max(1, int(iter_cfg.n_simulations)),
                            skip_exact=iter_cfg.exact_endgame_max_secs > 0.0,
                            one_visit_pruning=iter_cfg.policy_target_pruning,
                            forced_playout_subtraction=iter_cfg.forced_playout_subtraction,
                            forced_playout_k=iter_cfg.forced_playout_k)
                        for k, v in ps.items():
                            policy_prune_stats[k] += v
                    game_seed += full_games
                batched_stats = _merge_batched_stats(stat_parts)
                for examples, scores in zip(all_examples, all_scores):
                    buffer.add(examples)
                    diffs.append(scores[0] - scores[1])
            else:
                for g in range(normal_games):
                    # Per-game RNGs derived from game_seed alone (see _game_rngs): keeps
                    # self-play data independent of execution order and lets the
                    # parallel loop reproduce this exactly.
                    g_py_rng, g_np_rng = _game_rngs(game_seed)
                    game_sims = (
                        int(iter_cfg.n_simulations)
                        if iter_cfg.playout_cap_randomization
                        else (max(1, int(iter_cfg.fast_game_sims))
                              if g < fast_games else int(iter_cfg.n_simulations))
                    )
                    if iter_cfg.engine == "rust":
                        examples, scores = play_selfplay_game_rust(
                            rust_mcts, rust_eval,
                            n_simulations=game_sims,
                            n_determinizations=iter_cfg.n_determinizations,
                            temp_moves=iter_cfg.temp_moves, c_puct=iter_cfg.c_puct,
                            dirichlet_alpha=iter_cfg.dirichlet_alpha,
                            dirichlet_epsilon=iter_cfg.dirichlet_epsilon,
                            leaf_batch=iter_cfg.leaf_batch, virtual_loss=1,
                            seed=game_seed, py_rng=g_py_rng, np_rng=g_np_rng,
                            score_scale=iter_cfg.score_scale,
                            margin_gain=iter_cfg.margin_gain, alpha=iter_cfg.alpha,
                            iteration=it,
                            playout_cfg=(iter_cfg if iter_cfg.playout_cap_randomization else None),
                        )
                    elif iter_cfg.engine == "open_loop":
                        if game_sims != getattr(ol_mcts, "n_simulations", None):
                            ol_mcts = make_open_loop_mcts(generation_net, iter_cfg, game_sims)
                        examples, scores = play_selfplay_game(
                            ol_mcts,
                            n_determinizations=1,   # ignored internally; required by signature
                            temp_moves=iter_cfg.temp_moves,
                            seed=game_seed,
                            py_rng=g_py_rng,
                            np_rng=g_np_rng,
                            leaf_batch=1,            # not used by open-loop
                            open_loop=True,
                            iteration=it,
                            playout_cfg=(iter_cfg if iter_cfg.playout_cap_randomization else None),
                        )
                    else:  # "python"
                        if game_sims != getattr(mcts, "n_simulations", None):
                            mcts = make_mcts(generation_net, iter_cfg, game_sims)
                        examples, scores = play_selfplay_game(
                            mcts, n_determinizations=iter_cfg.n_determinizations,
                            temp_moves=iter_cfg.temp_moves, seed=game_seed,
                            py_rng=g_py_rng,
                            np_rng=g_np_rng,
                            leaf_batch=iter_cfg.leaf_batch,
                            iteration=it,
                            playout_cfg=(iter_cfg if iter_cfg.playout_cap_randomization else None),
                        )
                    if iter_cfg.policy_target_pruning or iter_cfg.forced_playout_subtraction:
                        wrapped = [examples]
                        ps = _prune_examples_policy_targets(
                            wrapped, total_visits=max(1, int(game_sims)),
                            skip_exact=iter_cfg.exact_endgame_max_secs > 0.0,
                            one_visit_pruning=iter_cfg.policy_target_pruning,
                            forced_playout_subtraction=iter_cfg.forced_playout_subtraction,
                            forced_playout_k=iter_cfg.forced_playout_k)
                        for k, v in ps.items():
                            policy_prune_stats[k] += v
                    buffer.add(examples)
                    diffs.append(scores[0] - scores[1])
                    game_seed += 1
            if hof_games and hof_entry is not None:
                hof_net = load_hof_net(hof_entry.path, device=iter_cfg.device)
                if iter_cfg.engine in ("batched", "batched_open_loop"):
                    # Fast path: learner-vs-HOF games on the Rust BatchedMCTS
                    # (two-net, searcher-owns-network). Only learner moves become
                    # examples; both seats searched via GPU-batched leaves.
                    hof_ex, hof_sc, hstats = play_hof_games_batched(
                        generation_net, hof_net, iter_cfg,
                        n_games=hof_games, game_seed_start=game_seed,
                        iteration=it, opponent_source=hof_entry.path)
                    if (iter_cfg.policy_target_pruning
                            or iter_cfg.forced_playout_subtraction):
                        # Recorded HOF-game moves are LEARNER full searches at
                        # n_simulations (the shallow hof_sims profile belongs to
                        # the never-recorded opponent seat), so prune against
                        # the frontier budget, not hof_sims.
                        ps = _prune_examples_policy_targets(
                            hof_ex, total_visits=max(1, int(iter_cfg.n_simulations)),
                            skip_exact=iter_cfg.exact_endgame_max_secs > 0.0,
                            one_visit_pruning=iter_cfg.policy_target_pruning,
                            forced_playout_subtraction=iter_cfg.forced_playout_subtraction,
                            forced_playout_k=iter_cfg.forced_playout_k)
                        for k, v in ps.items():
                            policy_prune_stats[k] += v
                    for examples, (s0, s1) in zip(hof_ex, hof_sc):
                        buffer.add(examples)
                        diffs.append(s0 - s1)
                    hof_stats["trainable_examples"] += int(hstats["trainable_examples"])
                    hof_stats["mean_diff"] = float(hstats["mean_diff"])
                    game_seed += hof_games
                else:
                    # Legacy serial path (Python OpenLoopMCTS). --hof_current_sims
                    # caps the learner side (the full n_simulations is far too slow
                    # serially); 0 keeps the old full-sims behaviour.
                    current_hof_sims = (int(iter_cfg.hof_current_sims)
                                        if int(iter_cfg.hof_current_sims) > 0
                                        else int(iter_cfg.n_simulations))
                    current_hof_mcts = make_open_loop_mcts(
                        generation_net, iter_cfg, max(1, current_hof_sims))
                    hof_cfg = replace(
                        iter_cfg,
                        n_simulations=max(1, int(iter_cfg.hof_sims)),
                        temp_moves=int(iter_cfg.hof_temp_moves),
                        dirichlet_epsilon=float(iter_cfg.hof_dirichlet_epsilon),
                        playout_cap_randomization=False,
                    )
                    hof_mcts = make_open_loop_mcts(
                        hof_net, hof_cfg, int(hof_cfg.n_simulations))
                    hof_diffs = []
                    for h in range(hof_games):
                        _, g_np_rng = _game_rngs(game_seed)
                        current_player = h % 2
                        examples, scores, mixed_stats = play_current_vs_hof_game(
                            current_hof_mcts,
                            hof_mcts,
                            current_player=current_player,
                            current_cfg=iter_cfg,
                            seed=game_seed,
                            np_rng=g_np_rng,
                            iteration=it,
                            opponent_source=hof_entry.path,
                        )
                        if (iter_cfg.policy_target_pruning
                                or iter_cfg.forced_playout_subtraction):
                            wrapped = [examples]
                            ps = _prune_examples_policy_targets(
                                wrapped, total_visits=max(1, int(iter_cfg.n_simulations)),
                                skip_exact=iter_cfg.exact_endgame_max_secs > 0.0,
                                one_visit_pruning=iter_cfg.policy_target_pruning,
                                forced_playout_subtraction=iter_cfg.forced_playout_subtraction,
                                forced_playout_k=iter_cfg.forced_playout_k)
                            for k, v in ps.items():
                                policy_prune_stats[k] += v
                        buffer.add(examples)
                        diff = scores[0] - scores[1]
                        diffs.append(diff)
                        hof_diffs.append(diff if current_player == 0 else -diff)
                        hof_stats["trainable_examples"] += int(
                            mixed_stats["current_records"])
                        game_seed += 1
                    if hof_diffs:
                        hof_stats["mean_diff"] = float(
                            np.asarray(hof_diffs, dtype=np.float32).mean())
                hof_net.to("cpu")
                if str(iter_cfg.device).startswith("cuda"):
                    torch.cuda.empty_cache()
            sp_elapsed = time.time() - t0
            played_games = normal_games + hof_games
            sp_rate = played_games / sp_elapsed if sp_elapsed > 0 else 0.0
            diffs_arr = np.array(diffs, dtype=np.float32)
            sp_score_diff_mean = float(diffs_arr.mean()) if len(diffs_arr) else 0.0
            sp_score_diff_std = float(diffs_arr.std()) if len(diffs_arr) else 0.0
            buffer_size = len(buffer)
            buffer_mean_age = buffer.mean_age(it)
            history["sp_score_diff_mean"].append(sp_score_diff_mean)
            history["sp_score_diff_std"].append(sp_score_diff_std)
            history["games_per_sec"].append(sp_rate)
            history["buffer_size"].append(buffer_size)
            history["buffer_mean_age"].append(buffer_mean_age)
            if verbose:
                print(f"  self-play: {played_games} games "
                      f"({sp_rate:.2f} games/sec), buffer={buffer_size}")
                if hof_games:
                    print(f"  HOF: {hof_games}/{played_games} games, "
                          f"opponent={Path(str(hof_stats['opponent'])).name}, "
                          f"trainable_examples={hof_stats['trainable_examples']}, "
                          f"current_mean_diff={hof_stats['mean_diff']:+.1f}")
                if fast_games:
                    print(f"  fast games: {fast_games}/{normal_games} "
                          f"at {iter_cfg.fast_game_sims} sims "
                          f"(full={full_games} at {iter_cfg.n_simulations})")
                if iter_cfg.policy_target_pruning:
                    print(f"  policy pruning: examples={policy_prune_stats['policy_pruned_examples']} "
                          f"actions={policy_prune_stats['policy_pruned_actions']} "
                          f"mass={policy_prune_stats['policy_pruned_mass']:.4f}")
                if iter_cfg.forced_playout_subtraction:
                    print(f"  forced playout subtraction: examples={policy_prune_stats['forced_pruned_examples']} "
                          f"actions={policy_prune_stats['forced_pruned_actions']} "
                          f"mass={policy_prune_stats['forced_pruned_mass']:.4f} "
                          f"visits={policy_prune_stats['forced_subtracted_visits']:.1f} "
                          f"missing_stats={policy_prune_stats['forced_missing_stats_examples']}")
                if batched_stats is not None:
                    print(f"  batched: mean_batch={batched_stats['mean_batch']:.1f}/"
                          f"{batched_stats['max_batch_cap']} "
                          f"(fill {batched_stats['fill_ratio']:.0%}), "
                          f"{batched_stats['requests_per_sec']:.0f} evals/sec, "
                          f"max_batch_seen={batched_stats['max_batch_seen']}, "
                          f"ticks={batched_stats['ticks']}")
                    total = max(1e-9, batched_stats["elapsed"])
                    print(f"  batched timing: step={batched_stats['step_sec']:.1f}s "
                          f"({batched_stats['step_sec']/total:.0%}), "
                          f"eval={batched_stats['eval_sec']:.1f}s "
                          f"({batched_stats['eval_sec']/total:.0%}), "
                          f"update={batched_stats['update_sec']:.1f}s "
                          f"({batched_stats['update_sec']/total:.0%})")
                    if "eval_forward_sec" in batched_stats:
                        print(f"  eval timing: h2d={batched_stats['eval_h2d_sec']:.1f}s, "
                              f"forward={batched_stats['eval_forward_sec']:.1f}s, "
                              f"readback={batched_stats['eval_readback_sec']:.1f}s, "
                              f"calls={batched_stats['eval_calls']}")
                    print(f"  exact endgame: solved={batched_stats.get('exact_solve_count', 0)} "
                          f"trees={batched_stats.get('exact_tree_solve_count', 0)} "
                          f"cache_hits={batched_stats.get('exact_cache_hit_count', 0)} "
                          f"fallback={batched_stats.get('exact_fallback_count', 0)}")
                    print(
                        f"  exact attempts: deck4_initial="
                        f"{batched_stats.get('exact_attempt_deck4_initial_count', 0)} "
                        f"deck4_retry={batched_stats.get('exact_attempt_deck4_retry_count', 0)} "
                        f"deck0={batched_stats.get('exact_attempt_deck0_count', 0)}; "
                        f"fallbacks: deck4_initial="
                        f"{batched_stats.get('exact_fallback_deck4_initial_count', 0)} "
                        f"deck4_retry={batched_stats.get('exact_fallback_deck4_retry_count', 0)} "
                        f"deck0={batched_stats.get('exact_fallback_deck0_count', 0)}"
                    )
                    if iter_cfg.playout_cap_randomization:
                        print(
                            f"  playout cap: full={batched_stats.get('full_move_count', 0)} "
                            f"fast={batched_stats.get('fast_move_count', 0)} "
                            f"recorded_full={batched_stats.get('recorded_full_move_count', 0)} "
                            f"recorded_fast={batched_stats.get('recorded_fast_move_count', 0)} "
                            f"exact_recorded={batched_stats.get('exact_recorded_move_count', 0)}"
                        )

            # ── 2. Train ──
            if len(buffer) < iter_cfg.min_buffer_to_train:
                if verbose:
                    print(f"  buffer below warmup ({len(buffer)}/{iter_cfg.min_buffer_to_train}); "
                          f"skipping training this iteration")
            elif iter_cfg.train_steps_per_iteration <= 0:
                if verbose:
                    print("  train: train_steps_per_iteration=0; skipping training")
            else:
                trained = True
                has_trained_ever = True
                net.train()
                p_sum = o_sum = q_sum = w_sum = 0.0
                brier_sum = baseline_sum = 0.0
                gnp_sum = gnw_sum = gno_sum = gnq_sum = 0.0

                def _run_train_step(batch):
                    """Run one optimiser step on `batch` and accumulate metrics."""
                    nonlocal p_sum, o_sum, q_sum, w_sum, brier_sum, baseline_sum
                    nonlocal gnp_sum, gnw_sum, gno_sum, gnq_sum
                    (policy_loss, own_loss, opp_loss, win_loss,
                     win_brier, baseline_brier) = train_step(
                        net, batch, optimizer,
                        policy_weight=iter_cfg.policy_weight,
                        lambda_score=iter_cfg.lambda_score,
                        lambda_w=iter_cfg.lambda_w,
                        score_scale=iter_cfg.score_scale,
                        grad_clip=iter_cfg.grad_clip,
                    )
                    # Grad norms: train_step zeros grads at the START of each step
                    # (not the end), so the grads are still populated here, after
                    # optimizer.step() (post-clip values).
                    gnp_sum += _grad_norm(_policy_params(net))
                    gnw_sum += _grad_norm(net.win_mlp.parameters())
                    gno_sum += _grad_norm(net.own_score_mlp.parameters())
                    gnq_sum += _grad_norm(net.opponent_score_mlp.parameters())
                    p_sum += policy_loss; o_sum += own_loss
                    q_sum += opp_loss; w_sum += win_loss
                    brier_sum += win_brier; baseline_sum += baseline_brier

                # Item 19: prefetch the next batch on a background thread while the
                # GPU runs train_step on the current one.  Only worthwhile with >1
                # step.  The prefetch thread uses its OWN rng (np_rng is not
                # thread-safe) and does only CPU numpy/Rust augment + a blocking
                # H2D copy; the main thread's train_step (the sole CUDA user) never
                # overlaps it because we .result() before stepping.
                use_prefetch = (iter_cfg.prefetch_batches
                                and iter_cfg.train_steps_per_iteration > 1)
                if use_prefetch:
                    prefetch_rng = np.random.default_rng(int(np_rng.integers(2**63)))
                    sample_fn = lambda: buffer.sample_batch(
                        iter_cfg.batch_size, prefetch_rng,
                        device=iter_cfg.device, augment_d4=iter_cfg.augment,
                        endgame_oversample_weight=iter_cfg.endgame_oversample)
                    executor = ThreadPoolExecutor(max_workers=1)
                    next_batch_future = executor.submit(sample_fn)   # prime
                    try:
                        for step in range(iter_cfg.train_steps_per_iteration):
                            batch = next_batch_future.result()
                            if step + 1 < iter_cfg.train_steps_per_iteration:
                                next_batch_future = executor.submit(sample_fn)
                            _run_train_step(batch)
                    finally:
                        executor.shutdown(wait=False)
                else:
                    for step in range(iter_cfg.train_steps_per_iteration):
                        batch = buffer.sample_batch(
                            iter_cfg.batch_size, np_rng,
                            device=iter_cfg.device, augment_d4=iter_cfg.augment,
                            endgame_oversample_weight=iter_cfg.endgame_oversample)
                        _run_train_step(batch)
                n = iter_cfg.train_steps_per_iteration
                pol_m, own_m, opp_m, win_m = p_sum/n, o_sum/n, q_sum/n, w_sum/n
                win_brier_m, baseline_brier_m = brier_sum/n, baseline_sum/n
                gn_pol, gn_win = gnp_sum/n, gnw_sum/n
                gn_own, gn_opp = gno_sum/n, gnq_sum/n
                history["policy_loss"].append(pol_m)
                history["own_loss"].append(own_m)
                history["opp_loss"].append(opp_m)
                history["win_loss"].append(win_m)
                history["win_brier"].append(win_brier_m)
                history["baseline_brier"].append(baseline_brier_m)
                history["grad_norm_policy"].append(gn_pol)
                history["grad_norm_win"].append(gn_win)
                history["grad_norm_own"].append(gn_own)
                history["grad_norm_opp"].append(gn_opp)
                if verbose:
                    print(f"  train: policy={pol_m:.4f}  own={own_m:.4f}  "
                          f"opp={opp_m:.4f} win={win_m:.4f}  "
                          f"brier={win_brier_m:.4f}  base={baseline_brier_m:.4f}")

                # ── Diagnostic batch (once per iteration, not per step) ──
                # policy_entropy + win_brier_diag share a single FIXED probe batch
                # (sampled once on the first training iteration), so both trends
                # reflect the net changing on identical positions rather than
                # batch sampling noise.  Decreasing entropy ⇒ the policy is
                # sharpening; win_brier_diag tracks win-head calibration.  The
                # probe's own RNG keeps np_rng (training/benchmark stream)
                # unperturbed.
                if diag_entropy_batch is None:
                    # Sample once (positions are densified into independent tensors,
                    # so later ring-buffer eviction doesn't touch them).
                    diag_entropy_batch = buffer.sample_batch(
                        min(256, len(buffer)), diag_rng,
                        device=iter_cfg.device, augment_d4=False,
                    )
                entropy, win_brier_diag = _diag_metrics(net, diag_entropy_batch)
                history["policy_entropy"].append(entropy)
                history["win_brier_diag"].append(win_brier_diag)
                net.train()  # restore train mode (_diag_metrics set eval)
                if verbose:
                    print(f"  diag: entropy={entropy:.4f}  "
                          f"brier_diag={win_brier_diag:.4f}")

            # ── 3. Benchmark + checkpoint ──
            if iter_cfg.benchmark_every and it % iter_cfg.benchmark_every == 0:
                net.eval()
                bench_dets = (iter_cfg.benchmark_determinizations
                              if iter_cfg.benchmark_determinizations is not None
                              else iter_cfg.n_determinizations)
                if iter_cfg.engine == "batched_open_loop":
                    # Rust-backed lockstep benchmark (~50x faster than the Python
                    # OpenLoopMCTS player; see benchmark_vs_rust docstring).
                    stats = benchmark_vs_rust(net, iter_cfg, iter_cfg.benchmark_seeds,
                                              seed=iter_cfg.seed + 99, verbose=False)
                else:
                    if iter_cfg.engine == "open_loop":
                        az = OpenLoopAZPlayer(
                            make_open_loop_mcts(net, iter_cfg, iter_cfg.benchmark_sims),
                            np_rng=np_rng,
                        )
                    else:
                        az = AZPlayer(make_mcts(net, iter_cfg, iter_cfg.benchmark_sims),
                                      n_determinizations=bench_dets, np_rng=np_rng)
                    stats = benchmark_vs(az, GreedyBot(), iter_cfg.benchmark_seeds,
                                         seed=iter_cfg.seed + 99, verbose=False)
                bench_win_rate = stats["az_win_rate"]
                bench_score_margin = stats["mean_margin"]
                # Brier on the fixed diagnostic batch at benchmark time (a
                # benchmark-time calibration reading on the same consistent set).
                if diag_entropy_batch is not None:
                    _, bench_win_brier = _diag_metrics(net, diag_entropy_batch)
                    net.eval()  # benchmark already runs in eval mode
                history["benchmark"].append((it, bench_win_rate))
                history["score_margin"].append(bench_score_margin)
                if verbose:
                    print(f"  benchmark vs Greedy: {stats['az_win_rate']:.1%} "
                          f"({stats['az_wins']}-{stats['draws']}-{stats['opp_wins']} "
                          f"over {stats['n_games']} games), "
                          f"mean_margin={stats['mean_margin']:+.1f}")

            checkpoint_path = None
            if cfg.checkpoint_dir:
                os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(cfg.checkpoint_dir, f"iter_{it:04d}.pt")
                save_checkpoint(checkpoint_path, net, iter_cfg, it, history,
                                run_manifest=run_manifest)
                record_checkpoint(cfg.checkpoint_dir, checkpoint_path, it)

            # Skip gate checks while training hasn't started YET (buffer still
            # warming): the learner is byte-identical to its warm start, so
            # the match is a guaranteed ~50% self-match. Does NOT apply when
            # training is explicitly disabled (train_steps=0) — there the
            # operator is gating a fixed learner on purpose.
            if (generator_state.mode in ("strict_gate", "soft_gate")
                    and iter_cfg.promotion_every
                    and it % int(iter_cfg.promotion_every) == 0
                    and not has_trained_ever
                    and iter_cfg.train_steps_per_iteration > 0):
                if verbose:
                    print("  promotion: skipped (no training has run yet - "
                          "learner is still the warm start)")
            elif (generator_state.mode in ("strict_gate", "soft_gate")
                    and iter_cfg.promotion_every
                    and it % int(iter_cfg.promotion_every) == 0):
                # Run8: the gate CANDIDATE is a rolling average of the last K
                # iteration checkpoints when promotion_average_k > 1 — the raw
                # learner keeps training/generating either way; only what gets
                # measured, promoted, and banked changes.
                candidate_net = net
                candidate_ckpt_path = checkpoint_path
                avg_k_used = 0
                if int(iter_cfg.promotion_average_k) > 1 and cfg.checkpoint_dir:
                    avg_state, avg_k_used = _rolling_average_state(
                        cfg.checkpoint_dir, it, int(iter_cfg.promotion_average_k))
                    if avg_k_used >= 2:
                        candidate_net = KingdominoNet(
                            channels=iter_cfg.channels,
                            blocks=iter_cfg.blocks,
                            bilinear_dim=iter_cfg.bilinear_dim,
                            score_scale=iter_cfg.score_scale,
                        )
                        candidate_net.load_state_dict(avg_state)
                        candidate_net.to(iter_cfg.device)
                        candidate_ckpt_path = os.path.join(
                            cfg.checkpoint_dir, f"avg_iter_{it:04d}_k{avg_k_used}.pt")
                        save_checkpoint(candidate_ckpt_path, candidate_net,
                                        iter_cfg, it, history,
                                        run_manifest=run_manifest)
                    else:
                        candidate_net = net
                if verbose:
                    cand_desc = (f"avg of last {avg_k_used} checkpoints"
                                 if avg_k_used >= 2 else "learner")
                    print(f"  promotion: evaluating {cand_desc} vs current best "
                          f"({iter_cfg.promotion_games} games, "
                          f"{iter_cfg.promotion_sims} sims)")
                candidate_net.eval()
                baseline_net = generator_state.baseline_net or generator_state.net
                baseline_net.eval()
                match = evaluate_network_match(
                    candidate_net, baseline_net,
                    games=iter_cfg.promotion_games,
                    sims=iter_cfg.promotion_sims,
                    device=iter_cfg.device,
                    batch_slots=iter_cfg.batch_slots,
                    leaf_batch=iter_cfg.leaf_batch,
                    seed=iter_cfg.promotion_seed + it * 100_000,
                    c_puct=iter_cfg.c_puct,
                    fpu=iter_cfg.fpu,
                    margin_gain=iter_cfg.margin_gain,
                    alpha=iter_cfg.alpha,
                    z=iter_cfg.promotion_confidence_z,
                )
                cand_fixed = base_fixed = None
                if iter_cfg.promotion_skip_fixed_suite:
                    fixed_cmp = compare_fixed_suite(
                        None, None,
                        tolerance=iter_cfg.promotion_fixed_suite_tolerance)
                else:
                    cand_fixed = fixed_suite_summary_for_net(
                        candidate_net,
                        suite=iter_cfg.promotion_fixed_suite,
                        device=iter_cfg.device,
                        checkpoint_label=f"learner_iter_{it:04d}",
                    )
                    base_fixed = fixed_suite_summary_for_net(
                        baseline_net,
                        suite=iter_cfg.promotion_fixed_suite,
                        device=iter_cfg.device,
                        checkpoint_label=f"current_best_iter_{it:04d}",
                    )
                    fixed_cmp = compare_fixed_suite(
                        cand_fixed, base_fixed,
                        tolerance=iter_cfg.promotion_fixed_suite_tolerance)
                decision = decide_promotion(
                    match, fixed_cmp,
                    min_win_rate=iter_cfg.promotion_min_win_rate,
                    min_lcb=iter_cfg.promotion_min_lcb,
                )
                promotion_result = {
                    "passed": decision.passed,
                    "action": _generator_action_after_promotion_check(
                        mode=generator_state.mode,
                        match=match,
                        promotion_passed=decision.passed,
                        revert_win_rate=iter_cfg.soft_gate_revert_win_rate,
                    ),
                    "win_rate": match.win_rate,
                    "lcb": match.lower_confidence_bound,
                    "wins": match.wins,
                    "losses": match.losses,
                    "draws": match.draws,
                    "mean_margin": match.mean_margin,
                    "reasons": list(decision.reasons),
                    "fixed_suite_checked": fixed_cmp.checked,
                    "fixed_suite_passed": fixed_cmp.passed,
                    "fixed_suite_delta_mae": fixed_cmp.delta_mean_abs_exact_value_error,
                }
                if verbose:
                    print(f"  promotion: {match.wins}-{match.losses}-{match.draws} "
                          f"win_rate={match.win_rate:.1%} "
                          f"LCB={match.lower_confidence_bound:.1%} "
                          f"passed={decision.passed} "
                          f"action={promotion_result['action']}")
                    for reason in decision.reasons:
                        print(f"    - {reason}")
                if promotion_result["action"] == "revert":
                    generator_state.net = baseline_net
                    generator_state.source = generator_state.baseline_source or str(iter_cfg.current_best_path)
                    generator_state.checkpoint_path = generator_state.baseline_checkpoint_path
                    generator_state.checkpoint_sha256 = generator_state.baseline_sha256
                    generator_state.action = "revert"
                    selfplay_net = generator_state.net
                    selfplay_source = generator_state.source
                    consecutive_reverts += 1
                    # Run8: a diverged learner never recovers unaided (run7:
                    # nine straight reverts). After N CONSECUTIVE reverts,
                    # reset the learner's weights to the baseline and clear
                    # the optimizer moments (stale Adam state would drag it
                    # straight back into the diverged basin). One revert alone
                    # never resets — a mid-breakthrough learner gets 5+
                    # iterations of baseline-generated data to recover first.
                    if (int(iter_cfg.revert_reset_after) > 0
                            and consecutive_reverts >= int(iter_cfg.revert_reset_after)):
                        net.load_state_dict(baseline_net.state_dict())
                        net.to(iter_cfg.device)
                        net.train()
                        optimizer.state.clear()
                        generator_state.action = "revert_reset"
                        if verbose:
                            print(f"  promotion: LEARNER RESET to baseline after "
                                  f"{consecutive_reverts} consecutive reverts "
                                  f"(optimizer moments cleared)")
                        consecutive_reverts = 0
                elif promotion_result["action"] == "probation":
                    consecutive_reverts = 0
                    generator_state.net = net
                    generator_state.source = f"learner_iter_{it:04d}"
                    generator_state.checkpoint_path = checkpoint_path
                    generator_state.checkpoint_sha256 = (
                        sha256_file(checkpoint_path) if checkpoint_path else None)
                    generator_state.action = "probation"
                    selfplay_net = generator_state.net
                    selfplay_source = generator_state.source
                elif promotion_result["action"] == "promote":
                    consecutive_reverts = 0
                    if generator_state.mode == "strict_gate":
                        # The PROMOTED artifact (possibly the rolling average)
                        # becomes the generator in strict_gate.
                        selfplay_net.load_state_dict(candidate_net.state_dict())
                        selfplay_net.eval()
                    selfplay_source = f"learner_iter_{it:04d}"
                    generator_state.net = net if generator_state.mode == "soft_gate" else selfplay_net
                    generator_state.source = selfplay_source
                    generator_state.checkpoint_path = checkpoint_path
                    generator_state.checkpoint_sha256 = (
                        sha256_file(checkpoint_path) if checkpoint_path else None)
                    generator_state.action = "promote_in_memory"
                    should_update_best = (
                        generator_state.mode == "soft_gate"
                        or iter_cfg.promotion_update_best
                    )
                    if should_update_best:
                        if candidate_ckpt_path is None:
                            print("WARNING: promotion update requested but no "
                                  "checkpoint_path exists; current_best.pt unchanged")
                        else:
                            hof_entry_before_promote = None
                            if (generator_state.mode == "soft_gate"
                                    and iter_cfg.current_best_path
                                    and os.path.exists(iter_cfg.current_best_path)):
                                hof_entry_before_promote = add_hof_entry(
                                    iter_cfg.current_best_path,
                                    hof_dir=iter_cfg.hof_dir,
                                    tag="pre_promote_current_best",
                                    iteration=it,
                                    metadata={
                                        "source": "self_play.soft_gate_pre_promote",
                                        "checkpoint_dir": cfg.checkpoint_dir,
                                        "candidate": candidate_ckpt_path,
                                    },
                                )
                            payload = promotion_payload(
                                candidate=candidate_ckpt_path,
                                current_best=iter_cfg.current_best_path,
                                decision=decision,
                                candidate_fixed_summary=cand_fixed,
                                baseline_fixed_summary=base_fixed,
                                extra={
                                    "source": f"self_play.{generator_state.mode}",
                                    "iteration": it,
                                    "run_checkpoint_dir": cfg.checkpoint_dir,
                                    "promotion_average_k": int(avg_k_used),
                                    "hof_previous_current_best": (
                                        hof_entry_before_promote.path
                                        if hof_entry_before_promote else None),
                                },
                            )
                            promote_current_best(
                                candidate_ckpt_path,
                                best_dir=Path(iter_cfg.current_best_path).parent,
                                current_best=iter_cfg.current_best_path,
                                payload=payload,
                            )
                            promoted_path = str(iter_cfg.current_best_path)
                            if generator_state.mode == "soft_gate":
                                selfplay_source = f"learner_iter_{it:04d}"
                                generator_state.source = selfplay_source
                                generator_state.checkpoint_path = checkpoint_path
                                generator_state.checkpoint_sha256 = (
                                    sha256_file(checkpoint_path)
                                    if checkpoint_path else None)
                            else:
                                selfplay_source = promoted_path
                                generator_state.source = promoted_path
                                generator_state.checkpoint_path = promoted_path
                                generator_state.checkpoint_sha256 = sha256_file(promoted_path)
                            generator_state.action = "promote_current_best"
                            generator_state.baseline_net = load_generator_net(
                                promoted_path, iter_cfg)
                            generator_state.baseline_source = promoted_path
                            generator_state.baseline_checkpoint_path = promoted_path
                            generator_state.baseline_sha256 = sha256_file(promoted_path)
                            selfplay_net = generator_state.net
                    if (iter_cfg.smart_elo and iter_cfg.smart_elo_on_promote
                            and candidate_ckpt_path is not None):
                        run_id = Path(cfg.checkpoint_dir).name if cfg.checkpoint_dir else "run"
                        smart_elo_triggered = True
                        smart_elo_reason = "promote"
                        smart_elo_name = f"{run_id}_iter_{it:04d}_promoted"
                        smart_elo_result = _run_smart_elo_rating(
                            checkpoint_path=candidate_ckpt_path,
                            checkpoint_name=smart_elo_name,
                            cfg=iter_cfg,
                            reason=smart_elo_reason,
                            verbose=verbose,
                        )
                    elif (iter_cfg.smart_elo and iter_cfg.smart_elo_on_promote
                          and checkpoint_path is None and verbose):
                        print("  [elo] smart Elo skipped: promoted checkpoint "
                              "was not saved to disk")

            if (iter_cfg.hof_add_every and it % int(iter_cfg.hof_add_every) == 0):
                if iter_cfg.current_best_path and os.path.exists(iter_cfg.current_best_path):
                    existing_hof_hashes = {
                        e.sha256 for e in read_hof_index(iter_cfg.hof_dir)
                    }
                    maybe_added_entry = add_hof_entry(
                        iter_cfg.current_best_path,
                        hof_dir=iter_cfg.hof_dir,
                        tag=iter_cfg.hof_add_tag,
                        iteration=it,
                        metadata={
                            "source": "self_play.hof_add_every",
                            "checkpoint_dir": cfg.checkpoint_dir,
                        },
                    )
                    if maybe_added_entry.sha256 not in existing_hof_hashes:
                        hof_added_entry = maybe_added_entry
                    if verbose:
                        if hof_added_entry is not None:
                            print(f"  HOF: added {Path(hof_added_entry.path).name}")
                        else:
                            print("  HOF: current_best already present; skipped add")
                elif verbose:
                    print(f"  HOF: skipped add; current_best_path missing: "
                          f"{iter_cfg.current_best_path}")

            # ── 3b. Elo rating (periodic) ──  AFTER the checkpoint is on disk so
            # the rater can load it; BEFORE the log row so its result can be logged.
            if cfg.elo_every and it % cfg.elo_every == 0 and cfg.checkpoint_dir:
                run_id = Path(cfg.checkpoint_dir).name
                elo_result = _run_elo_rating(
                    checkpoint_path=os.path.join(cfg.checkpoint_dir, f"iter_{it:04d}.pt"),
                    checkpoint_name=f"{run_id}_iter_{it:04d}",
                    cfg=iter_cfg,
                    verbose=verbose,
                )

            # ── 4. Structured log row + compact summary (END of iteration) ──
            # Oversampling check (diag iterations only): draw one batch's worth of
            # indices with the configured endgame weight and count how many land in
            # the endgame (game_progress >= 0.75). Uses diag_rng so the training
            # stream is unperturbed; an independent draw is statistically equivalent
            # to inspecting the real training batches.
            if (iter_cfg.diag_every and it % iter_cfg.diag_every == 0
                    and len(buffer) >= iter_cfg.min_buffer_to_train):
                prog_idx = FLAT_LAYOUT['game_progress'].start
                diag_idxs = buffer._draw_idxs(
                    iter_cfg.batch_size, diag_rng, iter_cfg.endgame_oversample)
                n_endgame_in_batch = int(sum(
                    1 for i in diag_idxs
                    if float(buffer.data[int(i)].flat[prog_idx]) >= 0.75))

            row = {
                "iter": it,
                "timestamp": time.time(),
                "policy_loss": pol_m,
                "own_loss": own_m,
                "opp_loss": opp_m,
                "win_loss": win_m,
                "win_brier": win_brier_m,
                "baseline_brier": baseline_brier_m,
                "grad_norm_policy": gn_pol,
                "grad_norm_win": gn_win,
                "grad_norm_own": gn_own,
                "grad_norm_opp": gn_opp,
                "sp_score_diff_mean": sp_score_diff_mean,
                "sp_score_diff_std": sp_score_diff_std,
                "games_per_sec": sp_rate,
                "buffer_size": buffer_size,
                "buffer_mean_age": buffer_mean_age,
                "policy_entropy": entropy,
                "win_brier_diag": win_brier_diag,
                "bench_win_rate": bench_win_rate,
                "bench_score_margin": bench_score_margin,
                "bench_win_brier": bench_win_brier,
                "elo_rating": elo_result["elo_rating"] if elo_result else None,
                "elo_stderr": elo_result["elo_stderr"] if elo_result else None,
                "elo_n_games": elo_result["elo_n_games"] if elo_result else None,
                "smart_elo_triggered": smart_elo_triggered,
                "smart_elo_reason": smart_elo_reason,
                "smart_elo_name": smart_elo_name,
                "smart_elo_rating": (
                    smart_elo_result["elo_rating"] if smart_elo_result else None),
                "smart_elo_stderr": (
                    smart_elo_result["elo_stderr"] if smart_elo_result else None),
                "smart_elo_n_games": (
                    smart_elo_result["elo_n_games"] if smart_elo_result else None),
                "smart_elo_skipped": (
                    smart_elo_result.get("smart_elo_skipped")
                    if smart_elo_result else None),
                "gated_selfplay": iter_cfg.gated_selfplay,
                "selfplay_source": selfplay_source,
                "generator_mode": generator_state.mode,
                "generator_source": generator_state.source,
                "generator_checkpoint_path": generator_state.checkpoint_path,
                "generator_sha256": generator_state.checkpoint_sha256,
                "generator_action": generator_state.action,
                "generator_baseline_source": generator_state.baseline_source,
                "generator_baseline_sha256": generator_state.baseline_sha256,
                "hof_fraction": iter_cfg.hof_fraction,
                "hof_enabled": hof_stats["enabled"],
                "hof_games": hof_stats["games"],
                "hof_trainable_examples": hof_stats["trainable_examples"],
                "hof_opponent": hof_stats["opponent"],
                "hof_opponent_sha256": hof_stats["opponent_sha256"],
                "hof_mean_diff": hof_stats["mean_diff"],
                "hof_added": hof_added_entry is not None,
                "hof_added_path": hof_added_entry.path if hof_added_entry else None,
                "promotion_checked": promotion_result is not None,
                "promotion_action": (
                    promotion_result["action"] if promotion_result else None),
                "promotion_passed": (
                    promotion_result["passed"] if promotion_result else None),
                "promotion_revert_win_rate": iter_cfg.soft_gate_revert_win_rate,
                "promotion_win_rate": (
                    promotion_result["win_rate"] if promotion_result else None),
                "promotion_lcb": (
                    promotion_result["lcb"] if promotion_result else None),
                "promotion_reasons": (
                    promotion_result["reasons"] if promotion_result else None),
                "promotion_fixed_suite_delta_mae": (
                    promotion_result["fixed_suite_delta_mae"]
                    if promotion_result else None),
                # Exact endgame solver stats (batched engines only; None otherwise).
                "exact_solve_count": (batched_stats.get("exact_solve_count", 0)
                                      if batched_stats else None),
                "exact_tree_solve_count": (batched_stats.get("exact_tree_solve_count", 0)
                                           if batched_stats else None),
                "exact_cache_hit_count": (batched_stats.get("exact_cache_hit_count", 0)
                                          if batched_stats else None),
                "exact_fallback_count": (batched_stats.get("exact_fallback_count", 0)
                                         if batched_stats else None),
                "exact_attempt_deck4_initial_count": (
                    batched_stats.get("exact_attempt_deck4_initial_count", 0)
                    if batched_stats else None),
                "exact_attempt_deck4_retry_count": (
                    batched_stats.get("exact_attempt_deck4_retry_count", 0)
                    if batched_stats else None),
                "exact_attempt_deck0_count": (
                    batched_stats.get("exact_attempt_deck0_count", 0)
                    if batched_stats else None),
                "exact_fallback_deck4_initial_count": (
                    batched_stats.get("exact_fallback_deck4_initial_count", 0)
                    if batched_stats else None),
                "exact_fallback_deck4_retry_count": (
                    batched_stats.get("exact_fallback_deck4_retry_count", 0)
                    if batched_stats else None),
                "exact_fallback_deck0_count": (
                    batched_stats.get("exact_fallback_deck0_count", 0)
                    if batched_stats else None),
                "exact_solver_secs": (batched_stats.get("exact_solver_secs", 0.0)
                                      if batched_stats else None),
                "exact_fallback_positions_saved": (
                    batched_stats.get("exact_fallback_positions_saved", 0)
                    if batched_stats else None),
                "total_cpus": (
                    batched_stats.get("total_cpus", None) if batched_stats else None),
                "game_cpus": (
                    batched_stats.get("game_cpus", None) if batched_stats else None),
                "solver_cpus": (
                    batched_stats.get("solver_cpus", None) if batched_stats else None),
                "solver_cpus_override": (
                    batched_stats.get("solver_cpus_override", None)
                    if batched_stats else None),
                "playout_cap_randomization": iter_cfg.playout_cap_randomization,
                "full_search_fraction": iter_cfg.full_search_fraction,
                "fast_move_sims": iter_cfg.fast_move_sims,
                "record_fast_moves": iter_cfg.record_fast_moves,
                "fast_move_dirichlet_epsilon": iter_cfg.fast_move_dirichlet_epsilon,
                "fast_move_temp_moves": iter_cfg.fast_move_temp_moves,
                "fast_move_count": (batched_stats.get("fast_move_count", 0)
                                    if batched_stats else None),
                "full_move_count": (batched_stats.get("full_move_count", 0)
                                    if batched_stats else None),
                "recorded_fast_move_count": (
                    batched_stats.get("recorded_fast_move_count", 0)
                    if batched_stats else None),
                "recorded_full_move_count": (
                    batched_stats.get("recorded_full_move_count", 0)
                    if batched_stats else None),
                "exact_recorded_move_count": (
                    batched_stats.get("exact_recorded_move_count", 0)
                    if batched_stats else None),
                "endgame_oversample": iter_cfg.endgame_oversample,
                "exact_policy_mode": iter_cfg.exact_policy_mode,
                "exact_clamp_delta": iter_cfg.exact_clamp_delta,
                "n_endgame_in_batch": n_endgame_in_batch,
                "lr": iter_cfg.lr,
                "alpha": iter_cfg.alpha,
                "n_simulations": iter_cfg.n_simulations,
                "games_per_iteration": iter_cfg.games_per_iteration,
                "train_steps_per_iteration": iter_cfg.train_steps_per_iteration,
                "buffer_capacity": iter_cfg.buffer_capacity,
                "c_puct": iter_cfg.c_puct,
                "dirichlet_epsilon": iter_cfg.dirichlet_epsilon,
                "temp_moves": iter_cfg.temp_moves,
                "fast_game_fraction": iter_cfg.fast_game_fraction,
                "fast_game_sims": iter_cfg.fast_game_sims,
                "fast_games": fast_games,
                "policy_target_pruning": iter_cfg.policy_target_pruning,
                "forced_playout_subtraction": iter_cfg.forced_playout_subtraction,
                "forced_playout_k": iter_cfg.forced_playout_k,
                **policy_prune_stats,
            }

            # ── Phase-sliced calibration diagnostics (Milestone 2) ──  Run every
            # cfg.diag_every iterations on a buffer snapshot, after training.
            # Guarded so a diagnostics bug can never crash a long training run.
            if (iter_cfg.diag_every and it % iter_cfg.diag_every == 0
                    and len(buffer) >= iter_cfg.min_buffer_to_train):
                try:
                    phase_diag = compute_all_diagnostics(
                        list(buffer.data), net, device=str(iter_cfg.device),
                        score_scale=iter_cfg.score_scale,
                        margin_gain=iter_cfg.margin_gain,
                        alpha=iter_cfg.alpha,
                    )
                    row.update(phase_diag)
                except Exception as e:
                    if verbose:
                        print(f"  diagnostics failed: {e}")
                finally:
                    net.train()  # compute_all_diagnostics leaves the net in eval
            # Alpha-transition trigger (stub — logged, not yet acted on; M5).
            _diag_rows.append(row)
            row["alpha_trigger"] = check_alpha_transition(_diag_rows)

            _log_row(log_path, row)
            if verbose:
                print(_compact_summary(
                    it, sp_games=played_games, row=row,
                    trained=trained, buf_n=buffer_size,
                    min_buf=iter_cfg.min_buffer_to_train))

    finally:
        # Save the buffer on EXIT — clean completion AND KeyboardInterrupt — so a
        # long run that the user Ctrl+C's still yields its replay buffer.  Guarded
        # so a save failure can't mask the original exception or skip buffer.close().
        if cfg.save_buffer:
            try:
                buffer.save(cfg.save_buffer)
                print(f"Buffer saved on exit: {cfg.save_buffer}")
            except Exception as e:
                print(f"WARNING: buffer save failed: {e}")

        # Final-checkpoint Elo rating (runs on clean completion AND on
        # Ctrl+C/exception, so the last checkpoint is always placed on the
        # ladder).  Guarded so it can never mask the original exception.
        if cfg.elo_every and cfg.checkpoint_dir and it:
            try:
                run_id = Path(cfg.checkpoint_dir).name
                final_path = os.path.join(cfg.checkpoint_dir, f"iter_{it:04d}.pt")
                if os.path.exists(final_path):
                    _run_elo_rating(
                        checkpoint_path=final_path,
                        checkpoint_name=f"{run_id}_iter_{it:04d}_final",
                        cfg=cfg,
                        verbose=verbose,
                    )
            except Exception as e:
                print(f"WARNING: final Elo rating failed: {e}")

        # Global ladder re-solve: one joint Bradley-Terry fit over the FULL game
        # log, so every rated checkpoint is placed on a mutually-consistent scale
        # (not just its own per-checkpoint MLE).  Cheap, in-memory; guarded so it
        # can never crash the run.
        if cfg.elo_every or cfg.smart_elo:
            try:
                from games.kingdomino.elo_rating import resolve_ladder
                games_path = cfg.elo_games_log or "elo_games.jsonl"
                db_path = cfg.elo_db or "elo_db.json"
                if os.path.exists(games_path):
                    if verbose:
                        print("  [elo] resolving global ladder from full game log...")
                    resolve_ladder(
                        games_path=games_path,
                        db_path=db_path,
                        verbose=verbose,
                    )
                    if verbose:
                        print("  [elo] ladder updated in elo_db.json")
            except Exception as e:
                if verbose:
                    print(f"  [elo] WARNING: resolve failed: {e}")

        buffer.close()

    return {"net": net, "history": history, "buffer": buffer}


# ─── 7. CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Serial AlphaZero self-play for Kingdomino")
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--games_per_iter", type=int, default=50)
    p.add_argument("--train_steps", type=int, default=200)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--determinizations", type=int, default=1)
    p.add_argument("--leaf_batch", type=int, default=1,
                   help="leaf-parallel batch for self-play search (1=serial). "
                        "Validate divergence with policy_compare before using >1; "
                        "applies to the serial in-process generation path.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--sample_workers", type=int, default=1,
                   help="threads for ReplayBuffer.sample_batch densify+augment "
                        "(1 = serial; >1 measured to REGRESS this GIL-bound "
                        "workload ~2x — kept for experimentation)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_schedule", default="",
                   help="piecewise schedule, e.g. '0:1e-3,50:3e-4'")
    p.add_argument("--alpha_schedule", default="",
                   help="piecewise schedule for value blend alpha")
    p.add_argument("--sims_schedule", default="",
                   help="piecewise schedule for full self-play MCTS simulations")
    p.add_argument("--exact_endgame_max_secs_schedule", default="",
                   help="piecewise schedule for the exact-endgame-solver wall-clock "
                        "budget, e.g. '0:1.0,10:2.0,25:3.0' (light early while the "
                        "net/game distribution is still weak, ramped up once "
                        "self-play reaches more representative endgames)")
    p.add_argument("--games_per_iter_schedule", default="",
                   help="piecewise schedule for self-play games per iteration")
    p.add_argument("--c_puct_schedule", default="",
                   help="piecewise schedule for MCTS c_puct")
    p.add_argument("--dirichlet_epsilon_schedule", default="",
                   help="piecewise schedule for root Dirichlet noise epsilon")
    p.add_argument("--temp_moves_schedule", default="",
                   help="piecewise schedule for temperature sampling plies")
    p.add_argument("--train_steps_schedule", default="",
                   help="piecewise schedule for training steps per iteration")
    p.add_argument("--buffer_capacity_schedule", default="",
                   help="piecewise schedule for replay buffer capacity")
    p.add_argument("--buffer", type=int, default=50_000)
    p.add_argument("--channels", type=int, default=96)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--bilinear_dim", type=int, default=64)
    p.add_argument("--benchmark_seeds", type=int, default=10)
    p.add_argument("--benchmark_determinizations", type=int, default=None,
                   help="PIMC worlds per benchmark move (default: reuse --determinizations)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--policy_weight", type=float, default=1.0,
                   help="weight on the masked policy cross-entropy loss")
    p.add_argument("--lambda_score", type=float, default=0.5,
                   help="weight on own+opp score MSE losses")
    p.add_argument("--lambda_w", type=float, default=0.25,
                   help="weight on win BCE loss")
    p.add_argument("--score_scale", type=float, default=160.0,
                   help="normalization divisor for the score heads")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="max global grad norm; <=0 disables clipping")
    p.add_argument("--c_puct", type=float, default=1.5)
    p.add_argument("--temp_moves", type=int, default=20,
                   help="sample ∝ visits for this many plies, then greedy")
    p.add_argument("--fpu", type=float, default=0.0,
                   help="first-play-urgency value for unvisited children")
    p.add_argument("--virtual_loss", type=int, default=1,
                   help="virtual loss magnitude (leaf-parallel / batched paths)")
    p.add_argument("--margin_gain", type=float, default=2.0,
                   help="leaf value: scales (own_norm-opp_norm) before tanh "
                        "(overrides mcts_az.MARGIN_GAIN)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="win-gated leaf value: reserved margin band B "
                        "(overrides mcts_az.ALPHA; (1-B)·win + B·win⁴·margin)")
    p.add_argument("--benchmark_every", type=int, default=1,
                   help="benchmark vs GreedyBot every N iterations")
    p.add_argument("--benchmark_sims", type=int, default=50,
                   help="MCTS sims per move during benchmarking")
    p.add_argument("--elo_every", type=int, default=0,
                   help="Rate checkpoint against anchor pool every N iterations "
                        "(0 = disabled). Uses batched open-loop engine at elo_sims.")
    p.add_argument("--elo_anchors", default="",
                   help="Path to elo_anchors.csv (default: auto-find in package dir)")
    p.add_argument("--elo_db", default="",
                   help="Path to elo_db.json (default: elo_db.json in cwd)")
    p.add_argument("--elo_games_log", default="",
                   help="Path to elo_games.jsonl (default: elo_games.jsonl in cwd)")
    p.add_argument("--elo_sims", type=int, default=400,
                   help="MCTS sims per move for Elo rating games")
    p.add_argument("--elo_games_per_anchor", type=int, default=32,
                   help="Paired seeds per anchor for rating (64 games total per anchor)")
    p.add_argument("--smart_elo", action="store_true",
                   help="Enable event-driven Elo triggers independent of --elo_every")
    p.add_argument("--smart_elo_on_promote", action="store_true",
                   help="Run Elo when a promotion gate promotes a new best model")
    p.add_argument("--smart_elo_games_per_anchor", type=int, default=32,
                   help="Paired seeds per anchor for smart Elo triggers")
    p.add_argument("--smart_elo_sims", type=int, default=400,
                   help="MCTS sims per move for smart Elo triggers")
    p.add_argument("--current_best_path", default=str(DEFAULT_CURRENT_BEST),
                   help="canonical promoted checkpoint used by --warm_start_current_best "
                        "and --gated_selfplay")
    p.add_argument("--warm_start_current_best", action="store_true",
                   help="warm-start learner weights from --current_best_path")
    p.add_argument("--gated_selfplay", action="store_true",
                   help="generate self-play from --current_best_path instead of the "
                        "latest learner; update the generator only after promotion")
    p.add_argument("--selfplay_generator_mode", default="latest",
                   choices=GENERATOR_MODES,
                   help="self-play generator control mode. 'latest' preserves "
                        "normal learner self-play; 'strict_gate' is the legacy "
                        "--gated_selfplay behavior; 'soft_gate' lets the latest "
                        "learner generate unless a promotion check reverts it.")
    p.add_argument("--promotion_every", type=int, default=0,
                   help="with --gated_selfplay, evaluate learner vs generator every N "
                        "iterations (0 = disabled)")
    p.add_argument("--promotion_games", type=int, default=384,
                   help="total head-to-head games for in-run promotion checks")
    p.add_argument("--promotion_sims", type=int, default=100,
                   help="MCTS sims per move for in-run promotion checks")
    p.add_argument("--soft_gate_revert_win_rate", type=float, default=0.48,
                   help="with soft_gate, revert generator to current_best when "
                        "latest scores below this raw win rate")
    p.add_argument("--promotion_min_win_rate", type=float, default=0.55)
    p.add_argument("--promotion_min_lcb", type=float, default=0.50)
    p.add_argument("--promotion_confidence_z", type=float, default=1.96)
    p.add_argument("--promotion_seed", type=int, default=20260630)
    p.add_argument("--promotion_fixed_suite", default=str(DEFAULT_FIXED_SUITE))
    p.add_argument("--promotion_fixed_suite_tolerance", type=float, default=0.05)
    p.add_argument("--promotion_skip_fixed_suite", action="store_true")
    p.add_argument("--promotion_average_k", type=int, default=0,
                   help="gate a rolling average of the last K iteration "
                        "checkpoints instead of the raw learner (0/1 = off); "
                        "removes snapshot noise / winner's curse from the gate")
    p.add_argument("--revert_reset_after", type=int, default=0,
                   help="reset the LEARNER weights (and optimizer moments) to "
                        "current_best after this many CONSECUTIVE gate reverts "
                        "(0 = never)")
    p.add_argument("--promotion_update_best", action="store_true",
                   help="if an in-run promotion passes, also copy the saved "
                        "checkpoint to --current_best_path")
    p.add_argument("--hof_dir", default=str(DEFAULT_HOF_DIR),
                   help="Hall-of-Fame checkpoint pool directory")
    p.add_argument("--hof_fraction", type=float, default=0.0,
                   help="fraction of games per iteration to play vs one HOF opponent")
    p.add_argument("--hof_fraction_schedule", default="",
                   help='schedule like "0:0.0,50:0.05,100:0.1"')
    p.add_argument("--hof_start_iter", type=int, default=50)
    p.add_argument("--hof_sample_weights", default="recency",
                   choices=("recency", "uniform", "mixed", "latest"))
    p.add_argument("--hof_sims", type=int, default=200,
                   help="MCTS sims for deterministic HOF opponent moves")
    p.add_argument("--hof_current_sims", type=int, default=0,
                   help="LEGACY serial HOF path only (engine not in batched/"
                        "batched_open_loop): sims for the learner side, since the "
                        "full --sims is unusably slow on the serial Python MCTS; "
                        "set ~100-400, 0 = use --sims. Ignored by the batched HOF "
                        "path, which searches both seats at --hof_sims.")
    p.add_argument("--hof_temp_moves", type=int, default=0,
                   help="HOF move sampling temperature window; default 0 = best play")
    p.add_argument("--hof_dirichlet_epsilon", type=float, default=0.0,
                   help="HOF root noise; default 0.0 = no forced exploration")
    p.add_argument("--hof_add_every", type=int, default=0,
                   help="copy current_best_path into hof_dir every N iterations")
    p.add_argument("--hof_add_tag", default="current_best")
    p.add_argument("--no_augment", action="store_true",
                   help="disable D4 augmentation (useful when debugging policy/mask)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no_tf32", action="store_true",
                   help="disable TF32 CUDA matmul/convolution for inference/training")
    p.add_argument("--amp_inference", action="store_true",
                   help="use CUDA float16 autocast for self-play inference")
    p.add_argument("--engine",
                   choices=["python", "open_loop", "rust", "batched", "batched_open_loop"],
                   default="python",
                   help="Search engine: 'python' (AlphaZeroMCTS+PIMC, closed-loop "
                        "oracle); 'open_loop' (OpenLoopMCTS, resamples deck order "
                        "per simulation, averages internally — n_determinizations/"
                        "leaf_batch ignored); 'rust' (RustMCTS, in-process leaf "
                        "eval, no IPC server); 'batched' (one synchronized Rust "
                        "BatchedMCTS); 'batched_open_loop' (Rust BatchedMCTS with "
                        "per-simulation deck resampling — the fast open-loop path).")
    p.add_argument("--batch_slots", type=int, default=32,
                   help="concurrent slots for --engine batched; separate from "
                        "--games_per_iter.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warm_start", default=None)
    p.add_argument("--min_buffer", type=int, default=None,
                   help="don't train until this many positions are in the buffer "
                        "(default: one iteration's worth, so training starts on iter 2)")
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--save_buffer", default="",
                   help="path to save the final replay buffer when training ends "
                        "(also saved on Ctrl+C); empty = don't save")
    p.add_argument("--warm_buffer", default="",
                   help="path to a previously saved buffer to load before iter 1 "
                        "(use with --warm_start); empty = start empty")
    p.add_argument("--warm_buffer_max_staleness", type=int, default=200,
                   help="discard warm_buffer examples older than this many "
                        "iterations (default 200 = permissive)")
    p.add_argument("--log_path", default=None,
                   help="per-iteration JSONL log path (default: auto-derive "
                        "{checkpoint_dir}/training_log.jsonl, else "
                        "./training_log_{timestamp}.jsonl)")
    p.add_argument("--compile", action="store_true",
                   help="(Item 20) torch.compile the inference net on CUDA "
                        "(leaf-eval only; training net stays uncompiled)")
    p.add_argument("--compile_dynamic", choices=["auto", "on", "off"],
                   default="auto",
                   help="dynamic= passed to torch.compile when --compile is set: "
                        "auto=torch default; on=dynamic=True (one shape-generic "
                        "graph — avoids the per-shape recompile storm the variable "
                        "leaf-eval batch causes; recommended if bench_compile flags "
                        "one); off=dynamic=False (static per-shape graphs).")
    p.add_argument("--prefetch_batches", action="store_true",
                   help="(Item 19) prefetch the next training batch on a "
                        "background thread while the GPU runs train_step")
    p.add_argument("--double_buffer", action="store_true",
                   help="(Item 17) Run two BatchedMCTS instances in parallel, "
                        "overlapping CPU tree work with GPU forward. Only for "
                        "engine=batched or batched_open_loop.")
    p.add_argument("--async_solve", action="store_true",
                   help="(Step 1.5) Solve endgames on a background thread so they "
                        "overlap the GPU eval. Pair with a larger --batch_slots "
                        "(overbooking) so solving slots don't collapse the batch.")
    p.add_argument("--game_cpus", type=int, default=2,
                   help="preferred async-solver CPU split: reserve this many "
                        "logical CPUs for game generation and give all remaining "
                        "CPUs to the exact-solver pool (default 2)")
    p.add_argument("--solver_cpus", type=int, default=0,
                   help="explicit override for threads in the dedicated "
                        "endgame-solver pool. 0 = derive from total CPUs minus "
                        "--game_cpus. Prefer --game_cpus for cloud runs.")
    p.add_argument("--fast_game_fraction", type=float, default=0.0,
                   help="KataGo-style playout-cap randomization: fraction of "
                        "self-play games using --fast_game_sims instead of the "
                        "scheduled full sim cap.")
    p.add_argument("--fast_game_fraction_schedule", default="",
                   help="piecewise schedule for --fast_game_fraction")
    p.add_argument("--fast_game_sims", type=int, default=100,
                   help="simulations for fast exploration games")
    p.add_argument("--playout_cap_randomization", action="store_true",
                   help="KataGo-style move-level playout-cap randomization: "
                        "sample full/fast search per move instead of per game.")
    p.add_argument("--full_search_fraction", type=float, default=0.25,
                   help="fraction of non-exact moves searched at --sims when "
                        "--playout_cap_randomization is enabled")
    p.add_argument("--fast_move_sims", type=int, default=100,
                   help="simulations for fast moves under move-level playout caps")
    p.add_argument("--record_fast_moves", action="store_true",
                   help="store fast-move policy targets; default off so fast "
                        "moves only advance self-play")
    p.add_argument("--fast_move_dirichlet_epsilon", type=float, default=0.0,
                   help="root Dirichlet epsilon for fast moves; default 0.0")
    p.add_argument("--fast_move_temp_moves", type=int, default=0,
                   help="temperature plies for fast moves; default 0 (greedy)")
    p.add_argument("--policy_target_pruning", action="store_true",
                   help="KataGo-inspired target cleanup: prune <=1-visit MCTS "
                        "policy noise before storing replay examples.")
    p.add_argument("--forced_playout_subtraction", action="store_true",
                   help="KataGo-style target cleanup: subtract forced playouts "
                        "using root priors before storing policy targets.")
    p.add_argument("--forced_playout_k", type=float, default=2.0,
                   help="KataGo forced-playout subtraction constant k")
    p.add_argument("--profile_eval_timing", action="store_true",
                   help="Split eval_sec into h2d / forward / readback to see "
                        "whether the evaluator is forward-bound or transfer/"
                        "packaging-bound.")
    p.add_argument("--exact_endgame_max_secs", type=float, default=3.0,
                   help="Per-position wall-clock time limit for the exact endgame "
                        "solver (seconds). 0.0 disables exact endgame solving; "
                        "higher values reduce fallbacks on the hardest endgames.")
    p.add_argument("--exact_policy_mode", default="argmax_ties",
                   choices=["exact", "soft_clamp", "argmax_ties"],
                   help="How exact roots price dominated children for the policy "
                        "label. argmax_ties = uniform over proven-tied-best "
                        "(default; won the label-shape ablation and is the "
                        "cheapest mode); soft_clamp = exact within "
                        "--exact_clamp_delta of best, clamp the rest; exact = "
                        "historical full-window per child. Root value and chosen "
                        "move are exact in every mode.")
    p.add_argument("--exact_clamp_delta", type=float, default=10.0,
                   help="soft_clamp threshold in raw margin points: children "
                        "proven at least this far below the best child are "
                        "recorded at the clamp value instead of solved exactly.")
    p.add_argument("--exact_fallback_positions", default="",
                   help="Optional JSONL sidecar path for exact-solver fallback "
                        "root states. Empty disables saving.")
    p.add_argument("--endgame_oversample", type=float, default=2.0,
                   help="Sampling weight for endgame positions (game_progress >= "
                        "0.75) relative to other positions. 1.0 = uniform. 2.0 = "
                        "endgame positions drawn 2x as often as their buffer "
                        "frequency.")
    a = p.parse_args()

    # Default warmup = one full iteration's worth of positions, so training
    # starts on the second iteration rather than never (when smoke-testing with
    # few games) or immediately (before the buffer has any diversity).
    min_buf = a.min_buffer if a.min_buffer is not None else a.games_per_iter * 52
    warm_start_path = a.warm_start
    if a.warm_start_current_best:
        if warm_start_path:
            raise SystemExit("--warm_start and --warm_start_current_best are mutually exclusive")
        warm_start_path = a.current_best_path

    cfg = SelfPlayConfig(
        n_iterations=a.iterations, games_per_iteration=a.games_per_iter,
        train_steps_per_iteration=a.train_steps, n_simulations=a.sims,
        n_determinizations=a.determinizations, leaf_batch=a.leaf_batch,
        batch_slots=a.batch_slots,
        batch_size=a.batch_size, sample_workers=a.sample_workers, lr=a.lr,
        lr_schedule=a.lr_schedule, alpha_schedule=a.alpha_schedule,
        sims_schedule=a.sims_schedule,
        exact_endgame_max_secs_schedule=a.exact_endgame_max_secs_schedule,
        games_per_iter_schedule=a.games_per_iter_schedule,
        c_puct_schedule=a.c_puct_schedule,
        dirichlet_epsilon_schedule=a.dirichlet_epsilon_schedule,
        temp_moves_schedule=a.temp_moves_schedule,
        train_steps_schedule=a.train_steps_schedule,
        buffer_capacity_schedule=a.buffer_capacity_schedule,
        buffer_capacity=a.buffer, channels=a.channels, blocks=a.blocks,
        bilinear_dim=a.bilinear_dim, benchmark_seeds=a.benchmark_seeds,
        benchmark_determinizations=a.benchmark_determinizations,
        grad_clip=a.grad_clip, augment=not a.no_augment,
        weight_decay=a.weight_decay, policy_weight=a.policy_weight,
        lambda_score=a.lambda_score, lambda_w=a.lambda_w,
        score_scale=a.score_scale,
        c_puct=a.c_puct, temp_moves=a.temp_moves,
        fpu=a.fpu, virtual_loss=a.virtual_loss,
        margin_gain=a.margin_gain, alpha=a.alpha,
        benchmark_every=a.benchmark_every, benchmark_sims=a.benchmark_sims,
        elo_every=a.elo_every, elo_anchors=a.elo_anchors, elo_db=a.elo_db,
        elo_games_log=a.elo_games_log, elo_sims=a.elo_sims,
        elo_games_per_anchor=a.elo_games_per_anchor,
        smart_elo=a.smart_elo,
        smart_elo_on_promote=a.smart_elo_on_promote,
        smart_elo_games_per_anchor=a.smart_elo_games_per_anchor,
        smart_elo_sims=a.smart_elo_sims,
        current_best_path=a.current_best_path,
        gated_selfplay=a.gated_selfplay,
        selfplay_generator_mode=a.selfplay_generator_mode,
        promotion_every=a.promotion_every,
        promotion_games=a.promotion_games,
        promotion_sims=a.promotion_sims,
        soft_gate_revert_win_rate=a.soft_gate_revert_win_rate,
        promotion_min_win_rate=a.promotion_min_win_rate,
        promotion_min_lcb=a.promotion_min_lcb,
        promotion_confidence_z=a.promotion_confidence_z,
        promotion_seed=a.promotion_seed,
        promotion_fixed_suite=a.promotion_fixed_suite,
        promotion_fixed_suite_tolerance=a.promotion_fixed_suite_tolerance,
        promotion_skip_fixed_suite=a.promotion_skip_fixed_suite,
        promotion_average_k=a.promotion_average_k,
        revert_reset_after=a.revert_reset_after,
        promotion_update_best=a.promotion_update_best,
        hof_dir=a.hof_dir,
        hof_fraction=a.hof_fraction,
        hof_fraction_schedule=a.hof_fraction_schedule,
        hof_start_iter=a.hof_start_iter,
        hof_sample_weights=a.hof_sample_weights,
        hof_sims=a.hof_sims,
        hof_current_sims=a.hof_current_sims,
        hof_temp_moves=a.hof_temp_moves,
        hof_dirichlet_epsilon=a.hof_dirichlet_epsilon,
        hof_add_every=a.hof_add_every,
        hof_add_tag=a.hof_add_tag,
        device=a.device, seed=a.seed, warm_start_path=warm_start_path,
        checkpoint_dir=a.checkpoint_dir, log_path=a.log_path,
        save_buffer=a.save_buffer, warm_buffer=a.warm_buffer,
        warm_buffer_max_staleness=a.warm_buffer_max_staleness,
        min_buffer_to_train=min_buf,
        engine=a.engine, allow_tf32=not a.no_tf32,
        inference_amp=a.amp_inference,
        compile_net=a.compile,
        compile_dynamic={"auto": None, "on": True, "off": False}[a.compile_dynamic],
        prefetch_batches=a.prefetch_batches,
        double_buffer=a.double_buffer,
        async_solve=a.async_solve,
        game_cpus=a.game_cpus,
        solver_cpus=a.solver_cpus,
        fast_game_fraction=a.fast_game_fraction,
        fast_game_fraction_schedule=a.fast_game_fraction_schedule,
        fast_game_sims=a.fast_game_sims,
        playout_cap_randomization=a.playout_cap_randomization,
        full_search_fraction=a.full_search_fraction,
        fast_move_sims=a.fast_move_sims,
        record_fast_moves=a.record_fast_moves,
        fast_move_dirichlet_epsilon=a.fast_move_dirichlet_epsilon,
        fast_move_temp_moves=a.fast_move_temp_moves,
        policy_target_pruning=a.policy_target_pruning,
        forced_playout_subtraction=a.forced_playout_subtraction,
        forced_playout_k=a.forced_playout_k,
        profile_eval_timing=a.profile_eval_timing,
        exact_endgame_max_secs=a.exact_endgame_max_secs,
        exact_policy_mode=a.exact_policy_mode,
        exact_clamp_delta=a.exact_clamp_delta,
        exact_fallback_positions=a.exact_fallback_positions,
        endgame_oversample=a.endgame_oversample,
    )
    run_self_play_training(cfg, verbose=True)
