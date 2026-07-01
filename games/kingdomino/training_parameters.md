# Kingdomino AlphaZero — Training Parameter Reference

## Example Command

```powershell
python -m games.kingdomino.self_play `
  --engine batched_open_loop `
  --device cuda `
  --async_solve `
  --solver_cpus 6 `
  --warm_start runs\kingdomino\local_48x6_run6\iter_0050.pt `
  --warm_buffer runs\kingdomino\local_48x6_run6\buffer_final.pkl `
  --warm_buffer_max_staleness 200 `
  --iterations 55 `
  --games_per_iter 150 `
  --train_steps 600 `
  --sims 1600 `
  --channels 48 `
  --blocks 6 `
  --batch_slots 32 `
  --leaf_batch 6 `
  --batch_size 256 `
  --lr 3e-4 `
  --weight_decay 1e-4 `
  --buffer 300000 `
  --lambda_score 0.5 `
  --lambda_w 0.25 `
  --score_scale 160.0 `
  --policy_weight 1.0 `
  --grad_clip 1.0 `
  --margin_gain 2.0 `
  --alpha 0.8 `
  --c_puct 1.5 `
  --exact_endgame_max_secs 3.0 `
  --endgame_oversample 2.0 `
  --temp_moves 20 `
  --fpu -0.2 `
  --virtual_loss 1 `
  --benchmark_every 10 `
  --benchmark_sims 50 `
  --benchmark_seeds 20 `
  --checkpoint_dir runs\kingdomino\local_48x6_run7 `
  --save_buffer runs\kingdomino\local_48x6_run7\buffer_final.pkl `
  --elo_every 10 `
  --elo_sims 400 `
  --elo_games_per_anchor 40 `
  --elo_db elo_db.json `
  --elo_games_log elo_games.jsonl `
  --seed 0
```

> Note: PowerShell requires single-line commands (no backslash continuation).
> The above uses backtick line continuation for readability only.

> **Throughput-optimal solver settings (`--async_solve --solver_cpus N`):** these
> make the exact endgame solver effectively *free* on throughput. Without them the
> solver runs on the synchronous critical path and costs ~13% games/s; with them it
> overlaps the GPU eval on a dedicated thread pool and recovers to the solver-off
> rate (~0.14 vs ~0.12 games/s on the 8-core laptop) while still emitting exact
> endgame targets. Set `--solver_cpus` to roughly `physical_cores − 2` (6 here),
> re-tuning per machine. See the two parameters below for details. `--channels 48
> --blocks 6` reflects the current production architecture (32ch/4b is saturated).

---

## Parameters by Category

---

### Engine & Hardware

#### `--engine`
**Values:** `python` | `open_loop` | `rust` | `batched` | `batched_open_loop`
**Current:** `batched_open_loop`

Which search engine drives self-play game generation.

| Value | Description | Use case |
|-------|-------------|----------|
| `python` | Python AlphaZeroMCTS + PIMC closed-loop | Debugging, correctness reference |
| `open_loop` | Python OpenLoopMCTS, resamples deck per simulation | Correctness testing |
| `rust` | Rust RustMCTS, in-process leaf eval | Intermediate speed |
| `batched` | Rust BatchedMCTS, closed-loop | Not recommended |
| `batched_open_loop` | Rust BatchedMCTS + open-loop deck resampling | **Production — use this** |

`batched_open_loop` is 48× faster than Python open-loop and correctly
handles Kingdomino's hidden deck order by resampling per simulation rather
than committing to a single future (det=1). All other engines are slower
or strategically inferior.

#### `--device`
**Values:** `cpu` | `cuda`
**Current:** `cuda`

Which device runs network inference and training. Always `cuda` when a GPU
is available — CPU training is ~20-50× slower and not viable for sims=1600.

#### `--batch_slots`
**Values:** 8–64, integer
**Current:** 32

Number of concurrent game slots inside `BatchedMCTS`. Each slot runs one
game; all slots are advanced in lockstep and their leaf positions are
batched into a single GPU forward call each tick.

- **Too low:** GPU underutilized (small batches, lots of kernel launch overhead)
- **Too high:** CPU tree work per tick increases, gains diminish
- **32 confirmed optimal** on RTX 3070 at 32ch/4b via sweep. Re-sweep on
  new hardware before assuming 32 is still optimal.
- Independent of `--games_per_iter` — slots are reused across games.

#### `--leaf_batch`
**Values:** 1–8, integer
**Current:** 6

Number of leaf positions gathered per slot per tick before calling the GPU.
Total batch size per forward ≈ `batch_slots × leaf_batch` (max 192 at 32×6).

- Higher values improve GPU utilization but increase memory per tick
- 6 confirmed optimal alongside batch_slots=32 on RTX 3070

---

### Network Architecture

#### `--channels`
**Values:** 32, 48, 64, 96, 128
**Current:** 32

Number of convolutional filters in each residual block. Controls model
capacity — how much the network can represent.

- **Too small:** policy loss plateaus early, can't learn complex strategies
- **Too large:** slower inference, more VRAM, diminishing returns until
  the smaller architecture is genuinely saturated
- **32ch/4b is approaching saturation** — policy loss plateaued at ~1.94
  before lr+FPU changes broke through to 1.869. Scale up once policy loss
  genuinely flatlines for 10+ iterations despite parameter adjustments.
- **Next step:** 48ch/6b — adds full 13×13 receptive field coverage
  (4 blocks covers 9×9; 6 blocks covers the full canvas)

#### `--blocks`
**Values:** 4, 6, 8, 10
**Current:** 4

Number of residual blocks in the shared trunk. Deeper networks can represent
more complex positional patterns but are slower to evaluate.

- More blocks = slower GPU forward = fewer sims achievable per second
- Scales together with `--channels`: go to 48ch/6b before 32ch/8b
- At 6 blocks the receptive field covers the full 13×13 board canvas —
  the primary architectural reason to move from 4 to 6 blocks

#### `--bilinear_dim`
**Values:** 32, 64, 128
**Current:** 64 (default)

Internal dimension of the bilinear interaction layer between the board
representation and the flat features. Rarely needs tuning — leave at 64.

---

### Self-Play Loop

#### `--iterations`
**Values:** Any positive integer
**Current:** 55

Total number of train/self-play cycles. Each iteration:
1. Generates `games_per_iter` games using the current network
2. Adds positions to the replay buffer
3. Runs `train_steps` gradient updates
4. (Optionally) benchmarks vs GreedyBot
5. (Optionally, every `elo_every` iters) rates checkpoint vs anchor pool

More iterations = more training, longer wall time. The run is not
complete until policy loss has plateaued for 10+ iterations.

#### `--games_per_iter`
**Values:** 20–500
**Current:** 150

Games generated per iteration before training. Key trade-offs:

- **More games:** richer, more diverse data per training cycle; slower
  feedback loop (network updates less frequently relative to games played)
- **Fewer games:** faster feedback loop; buffer diversity suffers;
  risk of training on highly correlated positions

At 0.29 games/s (laptop idle, sims=1600), wall time per iter ≈
`games_per_iter / 0.29` seconds of self-play.

| games/iter | self-play time | recommended train_steps |
|------------|---------------|------------------------|
| 50 | ~3 min | 200 |
| 100 | ~6 min | 400 |
| 150 | ~9 min | 600 |
| 200 | ~12 min | 800 |

#### `--train_steps`
**Values:** 50–2000
**Current:** 600

Gradient update steps per iteration. Should scale proportionally with
`games_per_iter` to keep the self-play/training ratio roughly constant.

- **Too few:** network doesn't fully absorb new data before next iteration
- **Too many:** overfits to the current buffer contents; wastes time
- **Rule of thumb:** `train_steps ≈ 4 × games_per_iter`

#### `--sims`
**Values:** 200–3200
**Current:** 1600

MCTS simulations per move during self-play. The single most important
quality parameter — controls how good the policy targets are.

- **Too low (< 800):** visit counts are too noisy; policy targets are nearly
  uniform; network can't learn from them. 50 and 200 sims produce unlearnable
  targets (empirically confirmed — the policy-loss plateau at ~1.91 broke
  when sims increased to 1600).
- **800–1600:** confirmed learning range. 1600 produces sharper targets
  and was used for the cloud run breakthrough at iter 42.
- **Higher (3200):** better targets but directly costs throughput.
  At 0.29 games/s for sims=1600, 3200 would halve throughput.
- **Does not affect benchmark or Elo rating** — those use separate sim counts.

---

#### Dynamic schedule flags
**Values:** comma-separated `iteration:value` pairs, zero-based schedule steps
**Current:** optional

Milestone 5 added single-run curriculum schedules. The run loop applies the
active per-iteration config to self-play, training, benchmark construction,
checkpoints, diagnostics, and JSONL logging.

Available schedule flags:

| Flag | Controls |
|------|----------|
| `--lr_schedule` | optimizer learning rate |
| `--alpha_schedule` | search leaf-value blend |
| `--sims_schedule` | full self-play MCTS simulations |
| `--games_per_iter_schedule` | games generated per iteration |
| `--c_puct_schedule` | PUCT exploration constant |
| `--dirichlet_epsilon_schedule` | root Dirichlet noise strength |
| `--temp_moves_schedule` | opening plies sampled from visits |
| `--train_steps_schedule` | optimizer steps per iteration |
| `--buffer_capacity_schedule` | replay buffer capacity |
| `--fast_game_fraction_schedule` | fraction of fast-sim games |

Example:

```powershell
--lr_schedule "0:1e-3,50:3e-4,150:1e-4" `
--alpha_schedule "0:0.8,50:0.0" `
--sims_schedule "0:800,20:1600,100:3200"
```

Schedule keys are zero-based curriculum steps: key `0` applies on iteration 1,
key `50` applies on iteration 51, etc. The active value is the greatest key less
than or equal to the current step.

Implementation method during a training run:

1. At startup, the run loop parses each schedule string into sorted
   `(step, value)` pairs.
2. At the start of iteration `it`, it computes `step = it - 1`.
3. For each scheduled field, it selects the greatest schedule key `<= step`.
   If no schedule entry applies, the base CLI/config value remains active.
4. The loop builds an iteration-local config from those active values.
5. That iteration-local config is used for self-play generation, optimizer LR,
   replay buffer capacity, training steps/batches, benchmark construction,
   checkpoints, diagnostics, and JSONL logging.

The schedule update happens once per iteration, before self-play. It does not
change settings mid-iteration. For example, `--sims_schedule "0:800,20:1600"`
means iterations 1-20 use 800 full-search sims, and iteration 21 onward uses
1600 until another schedule key overrides it.

Avoid changing `lr`, `alpha`, and `sims` all at one hard cliff unless the run is
explicitly testing that regime change. A sudden target/search change can make
loss curves look like a regression even when the final checkpoint is stronger.
For production learning runs, prefer small stair steps every ~8-12 iterations:

```powershell
--sims_schedule "0:800,8:1000,16:1200,28:1400,40:1600" `
--alpha_schedule "0:0.8,8:0.65,16:0.5,28:0.35,40:0.25" `
--lr_schedule "0:3e-4,16:2e-4,32:1.25e-4,44:1e-4"
```

This keeps early training cheap and exploratory, then gradually increases target
quality while reducing the optimizer step size.

---

#### `--fast_game_fraction`, `--fast_game_sims`
**Values:** fraction `0.0-1.0`, integer sims
**Current:** legacy/off unless specified

Legacy game-level playout-cap mix. Each iteration splits self-play into:

- **Fast games:** `fast_game_fraction * games_per_iter`, using `fast_game_sims`
- **Full games:** the remainder, using the active `--sims` / `--sims_schedule`

This decouples two competing goals:

- policy targets want high-sim searches so visit distributions are sharp
- value targets want more diverse states, which cheaper searches produce faster

Recommended starting point: `--fast_game_fraction 0.15 --fast_game_sims 100`
paired with a normal full sim cap such as 1600. The log records `fast_games`,
`fast_game_fraction`, and the active sim cap for each iteration.

This is now mostly kept for compatibility. It is coarser than KataGo's actual
playout-cap randomization because every move in a fast game is fast and every
move in a full game is full.

#### `--playout_cap_randomization`
**Values:** flag (off by default)
**Current:** preferred KataGo-style mode for new experiments

Move-level playout-cap randomization. When enabled, every non-exact self-play
move independently chooses either:

- **Full search:** `--sims` / `--sims_schedule`, normal root Dirichlet noise,
  normal `--temp_moves`, and the position is recorded.
- **Fast search:** `--fast_move_sims`, `--fast_move_dirichlet_epsilon`,
  `--fast_move_temp_moves`, and recorded only if `--record_fast_moves` is set.

Agreed defaults:

```text
--record_fast_moves            false
--fast_move_temp_moves         0
--fast_move_dirichlet_epsilon  0.0
```

These defaults make fast moves strong cheap play rather than noisy training
targets: no root noise, greedy selection, and no replay example. Exact-solved
endgame moves are still recorded because their targets come from the solver, not
from a collapsed low-sim search.

Additional knobs:

| Flag | Default | Meaning |
|------|---------|---------|
| `--full_search_fraction` | `0.25` | probability that a non-exact move uses the full sim cap |
| `--fast_move_sims` | `100` | simulations for fast moves |
| `--record_fast_moves` | off | store fast-search policy targets |
| `--fast_move_dirichlet_epsilon` | `0.0` | root noise epsilon for fast moves |
| `--fast_move_temp_moves` | `0` | sampled plies for fast moves |

When `--playout_cap_randomization` is enabled, the training loop bypasses the
legacy `--fast_game_fraction` split so the two modes do not stack.

`--full_search_fraction` is a throughput/learning optimum, not a
"higher is always better" knob. The useful rate is approximately:

```text
useful_training_positions_per_hour
  = games/sec
    * moves_per_game
    * full_search_fraction
    * target_quality_factor
    * diversity_factor
```

Low values generate more unique games and trajectories per hour, but record few
training positions per game and can leave the replay buffer underfilled. High
values record more full-search positions per game, but reduce games/sec and add
more correlated positions from the same games. Run9 used `0.25`; it finished 50
iterations with only `165941 / 300000` buffer examples, so that setting was
probably too low for the 48x6 overnight regime.

Recommended sweep values:

```text
0.25 baseline
0.33
0.40
0.50
```

Compare `recorded_full_move_count / wall_time`, `buffer_size` growth,
`games_per_sec`, `exact_solver_secs`, policy KL, win Brier, and final Elo. For
the current 48x6 laptop setup, start around `0.35`; move toward `0.40` only if
throughput remains acceptable. Leave `--record_fast_moves` off unless explicitly
testing lower-quality fast-search policy targets.

---

#### `--policy_target_pruning`
**Values:** flag (off by default)
**Current:** optional Milestone 5 feature

KataGo-inspired cleanup for policy targets before examples enter the replay
buffer. With only `--policy_target_pruning`, the implementation removes children
whose stored policy mass is consistent with one visit or less, then
renormalizes the remaining target. Exact endgame examples are left untouched
because their targets already come from exact child values rather than MCTS
exploration noise.

#### `--forced_playout_subtraction`, `--forced_playout_k`
**Values:** flag plus float (`k=2.0` default)
**Current:** optional Milestone 5 feature

Fuller KataGo-style policy cleanup. When enabled, normal Rust batched MCTS
records export root priors and raw visit counts alongside the stored sparse
policy target. Before the example enters replay, the loop subtracts the expected
forced-exploration visits:

KataGo's formula needs the root prior `P(c)` for every root child:

```text
n_forced(c) = sqrt(k * P(c) * sum_N(c'))
```

Then it clamps at zero and renormalizes the remaining visit counts into the
stored policy target. If subtraction would collapse the entire target to zero,
the original target is kept. Exact endgame examples are skipped.

Recommended first use:

```powershell
--forced_playout_subtraction --forced_playout_k 2.0
```

Training-quality guardrails:

- The feature is off by default.
- Training batches ignore the auxiliary root-prior fields.
- Disabled pruning is tested to leave policy targets unchanged even when root
  metadata exists.
- Replay sampling is tested to produce the same dense policy/legal-mask contract
  with root metadata present.
- Exact endgame examples are tested to remain untouched.
- The Rust tuple reader accepts both old 10-field and new 11-field examples
  with nested root stats.

Log fields include `forced_pruned_examples`, `forced_pruned_actions`,
`forced_pruned_mass`, `forced_subtracted_visits`, and
`forced_missing_stats_examples`.

---

#### `--exact_endgame_max_secs`
**Values:** `0.0` (disabled), `3.0`, larger for quality runs
**Current:** `3.0`

Per-position **wall-clock** budget (seconds) for the exact solver on
terminal-adjacent self-play roots in the Rust `BatchedMCTS` path. `0.0` disables
exact endgame solving for ablation. This replaced the old node-count budget
(`--exact_endgame_max_nodes`): wall-clock is what bounds throughput, and a node
budget timed out at a roughly fixed wall-clock cost regardless of complexity.

When a deck=4 solve exceeds the budget, the slot sets a per-game
`exact_unsolvable` sentinel and falls back to MCTS rather than retrying the same
full-row root every move. The retry policy is intentionally narrow: always allow
cheap `deck=0` exact solving later, and allow one additional `deck=4` attempt
once the current round has progressed to two or fewer remaining claims. A small
sample of real deck=4 positions showed this "after two moves" state was ~15x
faster at the median, so it often recovers exact labels without a retry storm.

The exact solver applies when the game has no remaining chance branching:

- `deck=0`: final placements only.
- `deck=4`: the last hidden row is deterministic because those four tiles are
  sorted into the next `current_row`.

When a batched self-play slot reaches one of these roots, `BatchedMCTS` attempts
an exact minimax solve before spending MCTS simulations or GPU inference. If the
solve fits within budget, the move is selected from exact child values, an exact
policy target is emitted, and the deterministic continuation can reuse the
precomputed exact plan. If the budget is exceeded, the slot falls back to normal
MCTS for that root.

**Historical 1600-sim smoke** (old node budget, 32 games, 48x6 CUDA,
`batch_slots=4`, `leaf_batch=6`) — kept for context; `500000` nodes was the best
throughput point before the switch to a wall-clock budget:

| Old node setting | games/s | exact moves | trees | cache hits | fallback |
|------------------|---------|-------------|-------|------------|----------|
| `0` | 0.0742 | 0 | 0 | 0 | 0 |
| `500000` | **0.0816** | 369 | 32 | 337 | 15 |
| `2000000` | 0.0769 | 379 | 32 | 347 | 5 |
| `5000000` | 0.0760 | 382 | 32 | 350 | 2 |

Interpretation:

- `3.0s` is the routine-training default: enough to solve essentially all eligible
  deck=4 roots (p90 ~1.3s at alpha=0.8) while capping the rare hard tail.
- Larger budgets reduce fallbacks but spend more CPU time in exactly the tail
  positions where the solver is most expensive.
- `0.0` is useful for measuring whether exact solving is helping a particular run.
- Advisor/reanalysis use cases should prefer larger budgets and more aggressive
  solver work; training should optimize the throughput/label-quality tradeoff.

Run9 used `10.0s` and averaged roughly 637 solver-seconds per iteration while
GPU fill was low. For learning-speed runs, return to `3.0s` or at most `5.0s`
before increasing search parallelism. Watch `exact_fallback_count` plus the
split attempt/fallback counters for `deck4_initial`, `deck4_retry`, and `deck0`;
a small fallback increase is acceptable if examples/hour and games/hour improve.
All three solve entry points share the single `exact_endgame_max_secs` cap.

Exact policy targets are soft expert targets, not one-hot best-move labels. The
solver converts exact child values into an advantage-weighted softmax with
`temperature = value_range / 3`; if all legal moves are equal value, the target is
uniform across legal moves. This keeps ambiguous endgames soft and decisive
endgames sharp without a fixed temperature hyperparameter.

---

#### `--endgame_oversample`
**Values:** `1.0` (uniform), `2.0`
**Current:** `2.0`

Training-batch sampling weight for endgame positions (`game_progress >= 0.75`)
relative to other positions. Endgame examples carry the best labels in the buffer
(exact minimax values + exact-derived policy targets), so weighting them `2×`
concentrates gradient where the labels are most reliable. `1.0` recovers exact
uniform sampling. The realised fraction is logged as `n_endgame_in_batch` on
diagnostic iterations (≈33% at weight 2.0 on a ~20%-endgame buffer).

---

#### `--async_solve`
**Values:** flag (off by default)
**Recommended:** on (paired with `--solver_cpus`)

Run the exact endgame solver on a dedicated **background thread** so it overlaps
the GPU eval instead of blocking the synchronous `step()` critical path. When a
slot reaches a terminal-adjacent endgame it is snapshotted and dispatched to the
background solver; the slot keeps its game and resumes MCTS in place if the solve
times out, while the main loop keeps feeding the GPU from the other slots. Solved
games rejoin on the next harvest.

- **Default (off):** the solver runs synchronously inside `step()` and costs ~13%
  games/s (it is otherwise free in CPU-scheduling terms — the cost is just that it
  sits on the critical path).
- **On (with `--solver_cpus N`):** the solver becomes throughput-free. On the
  8-core laptop it recovers to ~0.14 games/s (== solver-off) vs ~0.12 sync, while
  still emitting exact endgame targets.
- **Correctness is identical** — deterministic, bit-for-bit the same per-seed
  games and finished results as the sync path (verified).

Off by default so the simpler sync path stays the baseline. Turn it on for routine
solver-on training. Its real payoff grows on the cloud 5090 (more spare cores).

#### `--solver_cpus`
**Values:** 0 (auto) – physical core count, integer
**Default:** 0 (auto = half of available threads)
**Recommended:** `physical_cores − 2` (e.g. 6 on an 8-core box)

Size of the dedicated Rayon thread pool that runs the within-solve (YBW) endgame
parallelism when `--async_solve` is on. Game generation (MCTS descent + backup)
gets the remaining cores via the global pool. Confining the solver to its own pool
is what makes the overlap pay off — a single shared pool lets the solver's long,
non-preemptible subtree-solves head-of-line-block the latency-critical work that
feeds the GPU, which *inflates* eval/update and makes async slower than sync.

**Sweep (8-core laptop, `--async_solve`, batch_slots=32):**

| solver_cpus | games/s | endgames solved / 100 |
|-------------|---------|-----------------------|
| 5 | 0.140 | 54 |
| **6** | **0.139** | **59** |
| 7 | 0.141 | 59 |

Throughput is flat across 5–7 (GPU-bound, solver fully hidden); solve-success
peaks at 6 then plateaus. So `physical_cores − 2` is the knee — give the solver
everything except ~2 cores reserved for generation. Re-tune per machine; the cloud
5090 (16–32 vCPUs) has far more headroom. Only has effect with `--async_solve`.

---

### Replay Buffer

#### `--buffer`
**Values:** 50000–500000
**Current:** 300000

Maximum number of training positions held in the replay buffer. Older
positions are evicted when full (ring buffer).

At ~80 positions per game and 150 games/iter:

| buffer size | iterations of history | RAM usage |
|-------------|----------------------|-----------|
| 100,000 | ~8 iters | ~743 MB |
| 150,000 | ~12 iters | ~1.1 GB |
| 300,000 | ~25 iters | ~2.2 GB |
| 500,000 | ~42 iters | ~3.7 GB |

- **Too small:** network trains on a narrow, repetitive slice of its own
  recent play; less stable learning signal
- **Too large:** examples from much weaker earlier policies dilute the
  training signal; also uses more RAM
- **25 iterations of history is a good target** — enough diversity
  without excessive staleness
- System RAM: 16 GB total; ~8-10 GB available after Windows overhead.
  300k (2.2 GB) is comfortable; 500k (3.7 GB) is feasible but watch RAM usage.

#### `--warm_buffer`
**Values:** path to `.pkl` file, or omit

Load a previously saved buffer before iteration 1, skipping the cold-start
period where the buffer is nearly empty and training signal is poor.

- Saves ~20-25 iterations of warm-up time on resumed runs
- Typically used alongside `--warm_start` (same checkpoint that generated
  the buffer)
- File size: ~7.6 KB per example → ~2.2 GB at 300k capacity
- Staleness filtering controlled by `--warm_buffer_max_staleness`

#### `--warm_buffer_max_staleness`
**Values:** 10–500, integer
**Default:** 200

When loading a warm buffer, discard examples older than this many
iterations relative to the warm-start checkpoint's iteration number.
Default 200 is permissive — keeps almost everything for a typical run.

#### `--save_buffer`
**Values:** path to `.pkl` file, or omit

Save the replay buffer to disk when the run ends (or on Ctrl+C).
Uses atomic write (tmp → replace) so a crash during save leaves no
corrupt file. Always specify this — losing the buffer means cold-start
on the next run.

---

### Training Hyperparameters

#### `--lr`
**Values:** 1e-4 – 1e-2
**Current:** 3e-4 *(reduced from 1e-3 in run 4)*

Learning rate for the Adam optimizer.

- 1e-3: standard AlphaZero starting value — use early in training
- 3e-4: fine-tuning regime — use when policy loss plateaus at 1e-3.
  Run 4 confirmed this broke a policy-loss plateau (1.94 → 1.869 over
  55 iterations) that 1e-3 couldn't escape.
- 1e-4: deep fine-tuning — try if 3e-4 also plateaus

#### `--weight_decay`
**Values:** 0 – 1e-3
**Current:** 1e-4

L2 regularization on network weights. 1e-4 is standard; rarely needs changing.

#### `--batch_size`
**Values:** 128–1024
**Current:** 256

Number of examples per gradient update step. 256 fits comfortably in
8 GB VRAM at 32ch/4b. Increasing to 512 gives marginally more stable
gradients with minimal throughput impact.

#### `--grad_clip`
**Values:** 0.5 – 5.0, or ≤0 to disable
**Current:** 1.0

Maximum global gradient norm. 1.0 is standard AlphaZero. Reduce to 0.5
if loss spikes occur. Rarely needs changing.

#### `--policy_weight`
**Values:** 0.5 – 2.0
**Current:** 1.0

Multiplier on the policy cross-entropy loss term. Rarely changed from 1.0.

#### `--lambda_score`
**Values:** 0.0 – 1.0
**Current:** 0.5

Weight on the own_score + opponent_score MSE loss terms.

- `own_score` head carries tempo and first-pick signal
- `opp_score` head carries blocking signal
- These two signals are not recoverable from margin alone — separate
  heads give richer supervision even though search uses only the margin
- 0.5 lets policy dominate while training score heads meaningfully

#### `--lambda_w`
**Values:** 0.0 – 1.0
**Current:** 0.25

Weight on the win probability BCE loss term. Lower than lambda_score
because win targets are noisier. Once the model is mature (win_brier
stabilized), consider increasing to 0.5 to sharpen win head calibration
— a well-calibrated win head is essential since α=0.0 search relies
entirely on the win probability signal.

#### `--score_scale`
**Values:** 50.0 – 200.0
**Current:** 160.0 *(corrected from 100.0)*

Normalization divisor for the score heads. Mighty Duel Kingdomino scores
typically run 100–160:

| score_scale | target range | mean target | verdict |
|-------------|-------------|-------------|---------|
| 100 | 1.0 – 1.6 | ~1.3 | too high — inflates MSE loss |
| 160 | 0.625 – 1.0 | ~0.81 | **recommended** |
| 200 | 0.5 – 0.8 | ~0.65 | slightly low but fine |

Values above 1.0 are acceptable (~20% of games at score_scale=160).
Safe to change between warm-start runs — score heads adapt within a few
iterations. Margin computation `(own - opp)` cancels the scale factor
so search behavior is unaffected.

---

### Search Parameters

#### `--c_puct`
**Values:** 0.5 – 3.0
**Current:** 1.5 *(confirmed optimal by sweep)*

PUCT exploration constant. Balances exploiting high-value moves vs
exploring less-visited moves.

**Sweep results (α=0.0 fixed, sims=400, 240 games per config):**

| c_puct | Elo | vs local_cont_iter100 |
|--------|-----|-----------------------|
| 1.5 | **1020** | 71.2% |
| 1.0 | 1007 | 72.5% |
| 2.0 | 989 | 68.8% |
| 2.5 | 985 | 60.0% |

1.5 is confirmed optimal. Higher values spread visits too thin at
sims=1600 with Kingdomino's small branching factor (~30-50 actions).
Lower values are slightly under-exploratory. Do not change without
re-sweeping on a new architecture.

#### `--temp_moves`
**Values:** 0 – 40
**Current:** 20

Number of plies at the start of each game where moves are sampled
proportionally to visit counts (temperature=1). After this, moves are
selected greedily. Not related to simulation depth — only controls
move selection after MCTS completes.

- Ensures opening diversity so training data covers diverse positions
- 20 plies covers roughly the first half of a Kingdomino game
- Set to 0 during evaluation (already handled automatically by Elo
  rating and benchmark paths)

#### `--fpu`
**Values:** -1.0 – 1.0
**Current:** -0.2 *(changed from 0.0 in run 4)*

First-play urgency. The Q value assigned to unvisited child nodes in
PUCT selection. The formula assigns `fpu` directly as Q for unvisited
nodes — it is NOT subtracted, it IS the Q value:

```
PUCT(unvisited) = fpu + c_puct · P(a) · √N(parent) / 1
```

- **0.0:** neutral — unvisited nodes start from Q=0
- **-0.2:** pessimistic (FPU reduction) — unvisited nodes look worse
  than average, concentrating visits on moves the policy already rates
  highly. Confirmed by Leela Chess Zero research to improve play quality.
  Run 4 confirmed this helped break the policy-loss plateau alongside
  lr reduction.
- **Positive:** optimistic — encourages exploration aggressively

Used during both training self-play and Elo evaluation (EloConfig
default=-0.2). Set to 0.0 during benchmark_vs_rust (closed-loop path).
Temperature and Dirichlet noise are set to 0 during evaluation; FPU
is kept at -0.2 as it reflects the agent's search behavior, not
exploration noise.

#### `--virtual_loss`
**Values:** 1 – 5
**Current:** 1

Virtual loss magnitude for the batched engine. Discourages concurrent
simulations from choosing the same path. 1 is appropriate for
leaf_batch=6 — higher values distort Q values and are counterproductive.

#### `--margin_gain`
**Values:** 1.0 – 4.0
**Current:** 2.0

Scales the score margin before the tanh in the leaf value formula:
`leaf_value = α·tanh((own_norm - opp_norm) · margin_gain) + (1-α)·(2·win_prob-1)`

2.0 produces a good range: a 20-point margin (normalized ~0.2) gives
tanh(0.4) ≈ 0.38, meaningful but not saturated.

#### `--alpha`
**Values:** 0.0 – 1.0
**Current:** 0.8 for early training → 0.0 once win head calibrates

Weight on the margin term vs win probability term in the leaf value:
- `α=1.0`: pure margin (ignores win head during search)
- `α=0.0`: pure win probability (AlphaZero style)
- `α=0.8`: margin-dominant

Controls MCTS search behavior only — training loss is unaffected.
With α=0.8, a move with margin +10 and win prob 70% beats a move with
margin +8 and win prob 80% — the margin signal dominates.

**Sweep results (c_puct=1.5, sims=400, 240 games per config):**

| α | Elo | vs local_cont_iter100 |
|----|-----|-----------------------|
| 0.0 | **1020** | 71.2% |
| 0.2 | 1007 | 70.6% |
| 0.8 | 1004 | 66.9% |
| 0.5 | 987 | 60.6% |

**α=0.0 (pure win probability) is confirmed best for evaluation and
mature training.** α=0.5 underperforms both extremes — avoid it.

**Training schedule recommendation:**

The optimal α depends on training stage because it governs search
quality during self-play:

- **Early training (iterations 1–50):** use α=0.8. The win head is
  uncalibrated at the start of a new run (especially after a cold start
  or architecture change). Pure win probability search (α=0.0) with an
  uncalibrated win head produces poor self-play quality — MCTS follows
  a misleading signal. Margin is more reliable until the win head has
  seen enough games to calibrate (win_brier stabilizing, typically
  ~30-50 iterations).

- **Mature training (iterations 50+):** switch to α=0.0. Once the win
  head is calibrated, pure win probability search correctly prefers the
  higher-win-prob move even when it has lower score margin — the
  strategically correct behavior. The sweep confirmed a ~16 Elo gain
  from this switch on a mature model.

**How to switch mid-run:** warm-start from the iter-50 checkpoint with
`--alpha 0.0`. No other changes needed.

**For Elo evaluation:** always use α=0.0 regardless of training stage.
The canonical rating agent is α=0.0, c_puct=1.5, sims=400.

---

### Benchmarking

#### `--benchmark_every`
**Values:** 1 – 20, or 0 to disable
**Current:** 10

Run a benchmark vs GreedyBot every N iterations.

- Set to 1 for frequent feedback early in training
- Set to 10+ once GreedyBot is saturated (model wins 100% from iter 20
  onward) — the margin signal still grows but win rate is uninformative
- Each benchmark takes ~35s at benchmark_seeds=20, benchmark_sims=50
- GreedyBot is saturated at 100% win rate — benchmark is now a secondary
  signal; Elo rating is the primary strength metric

#### `--benchmark_sims`
**Values:** 50 – 400
**Current:** 50

MCTS simulations per move during benchmark games. Lower than training
sims for speed.

- 50 sims: fast (~35s for 20 seeds), adequate for margin tracking
- 200 sims: slower (~2 min), more reliable

#### `--benchmark_seeds`
**Values:** 10 – 100
**Current:** 20

Number of deck seeds for benchmark games. Total games = 2 × seeds
(paired design, each seed plays both sides).

- 20 seeds (40 games): ±14% CI — adequate for tracking
- 50 seeds (100 games): ±9% — for fine-grained comparisons

---

### Elo Rating

The Elo system automatically rates checkpoints against a fixed anchor
pool during training using open-loop BatchedMCTS with two-network
routing (`row_search_actors()` — searcher-owns-network). Ratings use
MLE per checkpoint inline, then a global Bradley-Terry re-solve
(Ordo-style) runs automatically at run end.

**Architecture files:**
- `elo_anchors.csv` — anchor pool definition
- `elo_db.json` — current Elo ladder (all rated checkpoints)
- `elo_games.jsonl` — append-only game log (accumulates across all runs)
- `elo_rating.py` — rating driver, standalone CLI

**Current anchor pool (sims=400, fpu=-0.2, α=0.0):**

| Anchor | Elo | Active |
|--------|-----|--------|
| greedy_bot | 0 | No (reference floor only) |
| cloud_iter100 | ~454 | Yes |
| local_cont_iter070 | ~781 | No (checkpoint deleted) |
| local_cont_iter100 | ~854 | Yes |
| local_cont5_lr2e4_iter055 | ~1009 | Yes |

#### `--elo_every`
**Values:** 0 (disabled) | 5 | 10 | 20
**Current:** 10

Rate the checkpoint against the anchor pool every N iterations. Also
always rates the final checkpoint in the finally block (on clean exit
or Ctrl+C). After the final rating, automatically runs `--resolve` to
re-solve the global ladder from the full game log.

- 0: Elo disabled entirely
- 5: more granular training curve, ~5.6 min overhead per 10 iters
- 10: recommended — ~2.8 min overhead per 10 iters, good granularity
- 20: minimal overhead but coarse training curve

At 0.72 games/s (two-network batched open-loop, sims=400):
3 anchors × 80 games = 240 games ≈ **5.5 minutes per rating session**
at elo_games_per_anchor=40.

#### `--elo_sims`
**Values:** 50 – 800
**Current:** 400

MCTS simulations per move during Elo rating games. Part of the agent
definition — all checkpoints must be rated at the same sims count for
the ladder to be comparable. Canonical value: 400.

- Lower (50-100): fast but noisy ratings — only for smoke tests
- 400: canonical — good precision at reasonable cost
- 800: sharper but doubles rating time

#### `--elo_games_per_anchor`
**Values:** 10 – 80
**Current:** 40

Paired seeds per anchor (total games = 2 × this per anchor). With 3
active anchors and MLE fit across all games:

| games_per_anchor | total games | ±Elo (95% CI) | time at 0.72 games/s |
|-----------------|-------------|----------------|----------------------|
| 10 | 60 | ±92 | ~1.4 min |
| 20 | 120 | ±65 | ~2.8 min |
| 40 | 240 | ±46 | ~5.5 min |
| 60 | 360 | ±37 | ~8.3 min |
| 80 | 480 | ±32 | ~11 min |

40 is the recommended default — ±46 Elo is sufficient to detect
meaningful improvement between runs while keeping rating sessions under
6 minutes. The `--resolve` post-run re-solve tightens this further
by combining all sessions' game data jointly.

#### `--elo_db`
**Values:** path to `elo_db.json`
**Current:** `elo_db.json` (in cwd)

Path to the Elo ladder database. Contains all rated checkpoints with
their ratings, standard errors, game counts, and anchor entries.
Defaults to `elo_db.json` in the current working directory.

Always point to the same production file across runs so the ladder
accumulates. Specify a temp path during smoke tests to protect
production data.

#### `--elo_games_log`
**Values:** path to `elo_games.jsonl`
**Current:** `elo_games.jsonl` (in cwd)

Path to the append-only game log. One JSON line per game, accumulates
across all training runs. Used by `--resolve` to re-solve the global
ladder. Never overwrite or delete this file — it is the primary asset
for long-term rating quality.

Fields per line: checkpoint, opponent, seed, orientation, score_checkpoint,
score_opponent, winner, sims, engine, routing, timestamp.

---

### Elo CLI (standalone)

**Rate a checkpoint:**
```
python -m games.kingdomino.elo_rating --checkpoint <path> --name <name> --alpha 0.0 --sims 400 --games_per_anchor 40 --device cuda --verbose
```

**Re-solve global ladder from game log:**
```
python -m games.kingdomino.elo_rating --resolve --verbose
```

**Re-bootstrap anchor pool (e.g. after adding a new anchor):**
```
python -m games.kingdomino.elo_rating --reanchor --sims 400 --games_per_anchor 40 --device cuda --verbose
```

**View current leaderboard:**
```
python -m games.kingdomino.elo_rating --leaderboard
```

**Search parameter sweep (α or c_puct):**
```
python -m games.kingdomino.elo_rating --checkpoint <path> --name <name>_alpha00 --alpha 0.0 --c_puct 1.5 --sims 400 --games_per_anchor 40 --device cuda --verbose
python -m games.kingdomino.elo_rating --checkpoint <path> --name <name>_alpha08 --alpha 0.8 --c_puct 1.5 --sims 400 --games_per_anchor 40 --device cuda --verbose
```
Then run `--resolve` to get globally consistent ratings across all configurations.

---

### Promotion Gate CLI

Promote a checkpoint only if it beats `current_best.pt` with statistical
confidence and does not regress on the fixed eval suite:

```
.\.venv\Scripts\python.exe .\scripts\promote_checkpoint.py `
  --candidate runs\kingdomino\<run>\iter_XXXX.pt `
  --current_best runs\kingdomino\best_checkpoint\current_best.pt `
  --best_dir runs\kingdomino\best_checkpoint `
  --games 400 `
  --sims 400 `
  --device cuda `
  --confirm
```

Without `--confirm`, the script writes the decision JSON but does not overwrite
`current_best.pt`. For the first-ever best checkpoint, use `--bootstrap` plus
`--confirm`.

Promotion requires:

```
win_rate >= 55%
AND Wilson lower confidence bound > 50%
AND fixed-suite exact-value MAE does not regress beyond tolerance
```

`current_best.pt` is also the advisor default. There is no separate
`advisor_default.pt`.

---

### Checkpoint & IO

#### `--warm_start`
**Values:** path to `.pt` checkpoint file, or omit

Load network weights from a previous checkpoint before training begins.
The checkpoint must have matching architecture (channels, blocks).
When resuming a run, the iteration counter continues from where it left
off (stored in the checkpoint's metadata).

For normal continued development, prefer:

```
--warm_start_current_best
```

This loads `--current_best_path` (default:
`runs/kingdomino/best_checkpoint/current_best.pt`) and avoids accidentally
starting from a regressed final checkpoint.

#### `--gated_selfplay`
**Values:** flag, off by default

Decouple the learner from the data-generating model. With this flag:

1. self-play is generated by `--current_best_path`;
2. the learner still trains every iteration;
3. every `--promotion_every` iterations, the learner is evaluated against the
   self-play generator;
4. only a passed promotion gate updates the in-memory self-play generator.

This allows the model to keep learning without letting a transient regression
poison future self-play data. Add `--promotion_update_best` only when you want
an in-run passed promotion to also overwrite `current_best.pt` on disk.

Validation status: targeted tests cover Wilson LCB behavior, fixed-suite
regression blocking, audit-file/backup writes, and a one-iteration gated
self-play smoke. Before using gated self-play for a long run, do one short CUDA
smoke with `--gated_selfplay --promotion_every 1 --promotion_games 20
--promotion_sims 50 --benchmark_every 0 --elo_every 0` and confirm
`promotion_checked`, `promotion_passed`, and `selfplay_source` appear as expected
in `training_log.jsonl`. For production, use larger promotion games
(`400-800`) and keep `--promotion_update_best` off until the dry-run decision
JSONs look sane.

#### Hall-of-Fame Opponent Mixing
**Values:** off by default

Seed the HOF pool from a promoted/manual checkpoint:

```
.\.venv\Scripts\python.exe .\scripts\seed_hof.py `
  --source runs\kingdomino\best_checkpoint\current_best.pt `
  --hof_dir runs\kingdomino\best_checkpoint\hof `
  --tag current_best_seed
```

Enable a small HOF block during training:

```
--hof_dir runs\kingdomino\best_checkpoint\hof
--hof_fraction_schedule "0:0.0,50:0.05,100:0.1"
--hof_start_iter 50
--hof_sample_weights recency
--hof_sims 200
--hof_temp_moves 0
--hof_dirichlet_epsilon 0.0
```

The first-pass M7 implementation samples one HOF opponent per iteration and
runs HOF games as a separate mixed-model open-loop block. This keeps the main
batched self-play path unchanged while we measure the training value and
throughput cost.

HOF opponents are deterministic by default: no Dirichlet noise and no
temperature exploration. Only the current model's turns are stored as trainable
examples; HOF-owned decisions are not used as policy targets.

Use `--hof_add_every N` to copy `--current_best_path` into the HOF pool every
N iterations. This should usually be paired with promotion gating so HOF entries
come only from promoted checkpoints.

Validation status: targeted tests cover HOF index metadata, loading each HOF
checkpoint with its own architecture, latest/recency sampling, trainable-example
filtering, and a one-iteration mixed self-play smoke. The remaining validation is
operational: run a short CUDA smoke with a seeded HOF pool and `hof_fraction`
around `0.05-0.10`, then check `hof_games`, `hof_trainable_examples`,
`hof_opponent_sha256`, and loss/Elo stability. Do not jump straight to the
original 30% HOF target; ramp only after the small-fraction run shows no loss
spikes or throughput surprises.

#### `--checkpoint_dir`
**Values:** directory path

Directory where checkpoint `.pt` files are saved after each iteration,
plus `training_log.jsonl`. Checkpoints are saved as `iter_XXXX.pt`.
All iterations are saved — disk usage is modest (~5-10 MB per checkpoint
at 32ch/4b).

When `--checkpoint_dir` is set, the run also writes the Milestone 5.5
provenance bundle into the same directory:
`run_manifest.json`, `git_commit.txt`, `dirty_diff.patch`,
`model_contract.json`, `ruleset_hash.json`, `schedule_config.json`, and
`hardware_benchmark.json`. The `.pt` checkpoint embeds a compact
`run_manifest` block with the manifest path, git commit/dirty flag, ruleset
hash, model-contract path, schedule-config path, and hardware-context path.
`run_manifest.json` is updated after each checkpoint with `last_checkpoint`
and the full checkpoint list.

The checkpoint_dir basename is used as the run-id prefix for Elo
checkpoint names in the game log (e.g. `checkpoints_ol_local_cont4_iter_0010`),
ensuring cross-run uniqueness in `elo_games.jsonl`.

#### `--seed`
**Values:** 0 – 2^128
**Current:** 0

Random seed for the numpy RNG used in `sample_batch`. Does NOT affect
self-play game seeds, network weights, or Rust MCTS. Two runs with the
same seed will be very similar but not bit-identical due to CUDA
non-determinism in backprop. Primarily useful for reproducibility.

---

### Flags (no value)

#### `--no_augment`
Disable D4 (8-fold board rotation/reflection) augmentation during
`sample_batch`. Only disable for debugging. Always leave on for training.

#### `--no_tf32`
Disable TF32 precision for CUDA matrix multiplications. Leave enabled
unless you suspect precision issues.

#### `--amp_inference`
Enable float16 autocast during self-play network inference.

**Do not use.** Measured 0.74× — a 26% slowdown on 32ch/4b. The network
is latency-bound at this size; fp16 cast overhead exceeds any compute
saving. Documented here to prevent re-testing.

#### `--double_buffer`
Run two `BatchedMCTS` instances in parallel, overlapping CPU tree work
with GPU forward calls.

Measured −8.5% on RTX 3070 (GPU still dominates at 84% of wall time;
GIL contention erases the overlap benefit). Re-test on the RTX 5090
cloud machine where the faster GPU may flip the bottleneck to CPU.

#### `--prefetch_batches`
Prefetch the next training batch in a background thread while the GPU
runs the current `train_step`.

Measured regression locally — `sample_batch` is too fast relative to
`train_step` for the overlap to pay off. Leave off.

#### `--profile_eval_timing`
Split `eval_sec` into `eval_h2d_sec` / `eval_forward_sec` / `eval_readback_sec`
(host→device transfer, GPU forward pass, device→host readback) in the training log,
to see whether the evaluator is forward-bound or transfer/packaging-bound.

Diagnostic only — adds CUDA syncs around each stage, so use it for a short profiling
run, not routine training. Measured (48ch/6b, async@32, batch ~137): forward ≈90% of
eval, transfers ≈8%, so the evaluator is **GPU-forward-compute-bound** — the lever
is the forward itself (channels-last / smaller net / faster GPU), not transfer
slimming or double-buffering.

#### `--compile`
Wrap the inference network with `torch.compile` (requires Triton,
Linux only). Estimated 1.2–1.5× speedup on eval_sec.

Not testable on Windows (Triton unavailable). Test on cloud Linux box
before any cloud run:
`python -m games.kingdomino.bench_compile --device cuda --sims 200 --games 20`

---

## Key Relationships

```
total_wall_time ≈ iterations × (games_per_iter/throughput + train_steps×0.1s)
                + (iterations/elo_every) × elo_rating_time

throughput (games/s) driven by: sims, batch_slots, leaf_batch, channels, blocks,
exact_endgame_max_secs (cost hidden by --async_solve --solver_cpus)

elo_rating_time ≈ (anchors × games_per_anchor × 2) / 0.72 games/s
  e.g. 3 anchors × 40 games = 240 games ÷ 0.72 = ~5.5 min per rating session

buffer_coverage (iters) = buffer / (games_per_iter × ~80 positions/game)

with playout-cap randomization and fast moves not recorded:
recorded_positions_per_iter ~= games_per_iter * ~80 * full_search_fraction
useful_positions/hour depends on both games/sec and full_search_fraction;
run short sweeps at 0.25/0.33/0.40/0.50 rather than assuming max is best

target_quality driven by: sims (dominant), batch_slots×leaf_batch (fill ratio)

train_steps should scale with games_per_iter: ~4× games_per_iter

Elo ±CI ≈ 400 / (log(10) × √(total_elo_games × p × (1-p)))
  at p=0.6, 240 games: ±46 Elo (95% CI)
```

## Recommended Configs by Use Case

| Use case | sims | games/iter | train_steps | buffer | elo_every | notes |
|----------|------|------------|-------------|--------|-----------|-------|
| Smoke test | 200 | 10 | 50 | 10000 | 0 | verify no errors; exact off (`--exact_endgame_max_secs 0`) |
| Fast iteration | 800 | 50 | 200 | 100000 | 0 | explore hyperparams; exact 3.0s |
| Laptop overnight | 1600 | 150 | 600 | 300000 | 10 | current config; exact 3.0s + `--async_solve --solver_cpus 6` |
| 48x6 replay-width run | scheduled 800-1600 | 224 | 600 | 300000 | 0-10 | `--playout_cap_randomization --full_search_fraction 0.35`, exact 3.0s, smooth lr/alpha/sims schedules |
| Cloud overnight | 1600 | 300 | 1200 | 500000 | 10 | ~3× throughput; `--async_solve --solver_cpus <vcpus−2>` |
| Scale up (48ch/6b) | 1600 | 200 | 800 | 400000 | 10 | after 32ch/4b saturates; exact 3.0s + `--async_solve` |

## Elo Anchor Pool Maintenance

The anchor pool should bracket the current model's strength with ~150-250
Elo gaps between anchors. As training improves the model:

**When to add a new top anchor:** when the top-rated non-anchor checkpoint
consistently rates >150 Elo above the current top anchor over 3+ consecutive
rating sessions. Promote it via `--reanchor`.

**When to add a mid anchor:** when a gap between adjacent anchors exceeds
~350 Elo (new checkpoints in that range get imprecise ratings). Find a
checkpoint from training history that falls in the gap and add it.

**Current spacing:**
- cloud_iter100 (~454) → local_cont_iter100 (~854): gap 400 — large,
  no mid anchor (local_cont_iter070 checkpoint was deleted)
- local_cont_iter100 (~854) → local_cont5_lr2e4_iter055 (~1009): gap 155 ✓

The 400 Elo gap between the bottom two anchors means checkpoints in
the 600-800 range will have wider confidence intervals. Add a mid anchor
when a suitable checkpoint from that range is available.

**After adding an anchor:** run `--reanchor --sims 400 --games_per_anchor 40`
to re-bootstrap the pool with the new anchor included.
