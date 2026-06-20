# Kingdomino AlphaZero — Training Parameter Reference

## Example Command

```powershell
python -m games.kingdomino.self_play `
  --engine batched_open_loop `
  --device cuda `
  --warm_start games\kingdomino\best_checkpoint\best_32x4.pt `
  --warm_buffer runs\kingdomino\local_cont3\buffer_final.pkl `
  --iterations 55 `
  --games_per_iter 150 `
  --train_steps 600 `
  --sims 1600 `
  --channels 32 `
  --blocks 4 `
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
  --temp_moves 20 `
  --fpu -0.2 `
  --virtual_loss 1 `
  --benchmark_every 10 `
  --benchmark_sims 50 `
  --benchmark_seeds 20 `
  --checkpoint_dir runs\kingdomino\local_cont4 `
  --save_buffer runs\kingdomino\local_cont4\buffer_final.pkl `
  --elo_every 10 `
  --elo_sims 400 `
  --elo_games_per_anchor 40 `
  --elo_db elo_db.json `
  --elo_games_log elo_games.jsonl `
  --seed 0
```

> Note: PowerShell requires single-line commands (no backslash continuation).
> The above uses backtick line continuation for readability only.

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
because win targets are noisier. Increase once the model is mature and
you want to improve win head calibration for the α sweep experiment.

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
**Current:** 1.5

PUCT exploration constant. Balances exploiting high-value moves vs
exploring less-visited moves.

- Kingdomino's small branching factor (~30-50 actions) supports higher
  values than Go/Chess. Post-convergence sweep: try 2.0 and 2.5.
- 1.5 is a safe default for active training.

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
**Current:** 0.8

Weight on the margin term vs win probability term in the leaf value:
- `α=1.0`: pure margin (ignores win head during search)
- `α=0.0`: pure win probability (AlphaZero style)
- `α=0.8`: margin-dominant (current)

Controls MCTS search behavior only — training loss is unaffected.
With α=0.8, a move with margin +10 and win prob 70% beats a move with
margin +8 and win prob 80% — the margin signal dominates.

Post-convergence sweep planned: α ∈ {0.8, 0.5, 0.2, 0.0} via round-robin
on a fixed checkpoint to find optimal search weighting without retraining.

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

**Current anchor pool (sims=400, fpu=-0.2):**

| Anchor | Elo | Active |
|--------|-----|--------|
| greedy_bot | 0 | No (reference floor only) |
| cloud_iter100 | ~510 | Yes |
| local_cont_iter070 | ~781 | Yes |
| local_cont_iter100 | ~876 | Yes |

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
python -m games.kingdomino.elo_rating --checkpoint <path> --name <name> --sims 400 --games_per_anchor 40 --device cuda --verbose
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

---

### Checkpoint & IO

#### `--warm_start`
**Values:** path to `.pt` checkpoint file, or omit

Load network weights from a previous checkpoint before training begins.
The checkpoint must have matching architecture (channels, blocks).
When resuming a run, the iteration counter continues from where it left
off (stored in the checkpoint's metadata).

#### `--checkpoint_dir`
**Values:** directory path

Directory where checkpoint `.pt` files are saved after each iteration,
plus `training_log.jsonl`. Checkpoints are saved as `iter_XXXX.pt`.
All iterations are saved — disk usage is modest (~5-10 MB per checkpoint
at 32ch/4b).

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

throughput (games/s) driven by: sims, batch_slots, leaf_batch, channels, blocks

elo_rating_time ≈ (anchors × games_per_anchor × 2) / 0.72 games/s
  e.g. 3 anchors × 40 games = 240 games ÷ 0.72 = ~5.5 min per rating session

buffer_coverage (iters) = buffer / (games_per_iter × ~80 positions/game)

target_quality driven by: sims (dominant), batch_slots×leaf_batch (fill ratio)

train_steps should scale with games_per_iter: ~4× games_per_iter

Elo ±CI ≈ 400 / (log(10) × √(total_elo_games × p × (1-p)))
  at p=0.6, 240 games: ±46 Elo (95% CI)
```

## Recommended Configs by Use Case

| Use case | sims | games/iter | train_steps | buffer | elo_every | notes |
|----------|------|------------|-------------|--------|-----------|-------|
| Smoke test | 200 | 10 | 50 | 10000 | 0 | verify no errors |
| Fast iteration | 800 | 50 | 200 | 100000 | 0 | explore hyperparams |
| Laptop overnight | 1600 | 150 | 600 | 300000 | 10 | current config |
| Cloud overnight | 1600 | 300 | 1200 | 500000 | 10 | ~3× throughput |
| Scale up (48ch/6b) | 1600 | 200 | 800 | 400000 | 10 | after 32ch/4b saturates |

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
- cloud_iter100 (~510) → local_cont_iter070 (~781): gap 271 ✓
- local_cont_iter070 (~781) → local_cont_iter100 (~876): gap 95 — close
  but acceptable since new checkpoints trend above iter100 anyway

**After adding an anchor:** run `--reanchor --sims 400 --games_per_anchor 40`
to re-bootstrap the pool with the new anchor included.