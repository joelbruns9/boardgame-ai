# Seven Wonders Duel — Training Parameter Reference

This document describes every command-line argument exposed by the Phase D
training pipeline in `games.seven_wonders_duel.phase_d`.

**Reconciled against the live parser on 2026-07-30** (W0-W7). If you add a flag,
add it here: `games/seven_wonders_duel/test_setup_cloud.py` checks the cloud
launch command against the parser, but nothing checks this document, so it goes
stale silently. The quickest audit is to diff `build_parser()`'s option strings
against the `###` headings below.

Where a figure is quoted as measured, it comes from `runs/laptop_training_03`
and its continuations on an RTX 3070 laptop at `d_model=128, layers=4`. The
cloud configuration is `384x8x6` at bf16 and its costs are roughly 1.85x per
generated game and 5.12x per training step; see `CLOUD_TRAINING_PLAN.md`.

## Recommended Laptop Command

```powershell
./.venv/Scripts/python.exe -m games.seven_wonders_duel.phase_d `
  --run-dir games/seven_wonders_duel/runs/<run_name> `
  --seed 20260728 `
  --device cuda `
  --generation-backend rust `
  --gate-backend rust `
  --schedule-basis games `
  --selfplay-generator-mode soft_gate `
  --bootstrap-policy auto_first_trained `
  --promotion-every 5 `
  --revert-reset-after 2 `
  --promotion-min-lcb 0.50 `
  --revert-max-ucb 0.48 `
  --gate-ladder-games 100 200 400 800 `
  --gate-ladder-step-up-after 2 `
  --gate-ladder-floor-games 10000 `
  --gate-sims 64 `
  --anchor-every-iterations 5 `
  --anchor-games 200 `
  --self-anchor-games 200 `
  --self-anchor-lag-games 20000 `
  --self-anchor-every-games 10000 `
  --iterations 30 `
  --games-per-iteration 400 `
  --seed-games 2000 `
  --curriculum-anneal-games 10000 `
  --draft-prior-games 10000 `
  --replay-window-coefficient 16 `
  --replay-window-exponent 0.6 `
  --replay-window-cap-games 20000 `
  --save-buffer games/seven_wonders_duel/runs/<run_name>/buffer_final.jsonl `
  --buffer-autosave-every 1 `
  --d-model 128 `
  --layers 4 `
  --precision fp32 `
  --train-steps 76 `
  --train-warmup-steps 25 `
  --train-batch-size 512 `
  --validate-every 100 `
  --learning-rate 2e-4 `
  --cheap-sims-min 16 `
  --cheap-sims-max 24 `
  --full-sims-min 64 `
  --full-sims-max 128 `
  --full-search-fraction 0.25 `
  --rust-slots 48 `
  --rust-scheduler-workers 1 `
  --rust-global-batch-cap 256 `
  --rust-max-inflight-batches 1 `
  --leaf-batch 1 `
  --age-deal-samples 32
```

PowerShell uses the backtick at the end of a line as its continuation
character. There must be no spaces after a continuation backtick.

**The gate is on, and that is a change from this page's previous advice.** The
earlier version set `--promotion-every 0` and argued that gates were
unaffordable on a laptop: "a properly powered gate is ~800 games, which at
`gate-sims 64` and the measured 0.145 games/s is about 90 minutes per gate."
Both halves of that are now wrong.

- *The cost.* Measured over iterations 70-89 of `laptop_training_03_w2_resume`
  on this laptop, an iteration without a gate takes **8.4 min** and an iteration
  with a 200-game/64-sim gate takes **16.1 min**, so the gate costs about
  **7.7 min** -- not 90. The difference is W5's throughput work: one persistent
  inference worker for the whole gate, models built once, both seat legs
  concurrent in one rolling pool.
- *The statistics.* The 800-game figure came from an SPRT indifference region.
  The rule is now a fixed-N Wilson interval over **seat pairs**, and the gate
  size is scheduled by a ladder rather than fixed. A small gate is safe, not
  merely cheap: with confidence bounds on both sides, too little evidence
  produces probation, never a coin flip.

Set `--schedule-basis games` for any new run. The iterations basis is retained
only so runs that started under it can be resumed; every schedule below reads
the games clock when the basis is `games`.

## What One Iteration Does

Each iteration performs these steps:

1. Load `current_best.pt`.
2. Generate `games-per-iteration` new games.
3. Combine recent iteration files with the retained bot seed curriculum.
4. Split whole games into training and validation sets.
5. Train a candidate for `train-steps` optimizer updates on uniform random
   minibatches, carrying AdamW moments over from the previous iteration.
6. Every `promotion-every` iterations, compare the candidate with
   `current_best.pt` using a paired SPRT gate.
7. Promote an accepted candidate and periodically run fixed bot-anchor gates.
8. Save the candidate, metrics, records, Elo results, and manifest updates.

The run directory is resumable. An established run must still contain its
`current_best.pt`; the pipeline deliberately refuses to silently restart an
established run from random weights.

## Run, Reproducibility, and Hardware

### `--run-dir`

**Required. Value:** directory path

Stores checkpoints, `current_best.pt`, replay JSONL files, Elo data, and the run
manifest. It also contains an append-only `training_log.jsonl` with one JSON
object per completed iteration. Each row records the existing generation and
training performance, epoch losses and validation metrics, record summaries,
promotion result, and any anchor-gate results. Use a new directory for a new
experiment. Reusing an existing run directory resumes that run and continues
after its last recorded iteration. If an older or interrupted run has manifest
iterations missing from the JSONL log, startup backfills them from the manifest.

### `--seed`

**Default:** `20260718`

Base seed used to derive game, curriculum, split, model initialization, and gate
seeds. CPU-side behavior is designed to be reproducible. CUDA training can
still differ slightly because GPU kernels are not guaranteed to be bit-exact.

### `--device`

**Default:** `cuda` when available, otherwise `cpu`

Torch device for neural inference and gradient training. Use `cuda` on the
laptop. Rust owns game/search execution, while Torch still performs network
forward passes and SGD on this device.

### `--iterations`

**Default:** `1`. **Value:** positive integer

Number of additional self-play/train/gate cycles to run. On resume this is the
number of new iterations, not the final absolute iteration number.

### `--plumbing-smoke`

**Default:** off. **Value:** flag

Overrides the major budgets with tiny values: two generated games, eight seed
games, a 32-wide one-layer model, one simulation, one training epoch, and a
two-game gate. It verifies plumbing only and does not produce a useful model.

## Game Generation and Curriculum

### `--games-per-iteration`

**Default:** `500`. **Value:** positive integer

New games generated before each training phase. More games improve diversity
but lengthen the delay between model updates. Approximate neural self-play time
is `games / games_per_second`, before training and evaluation.

### `--seed-games`

**Default:** `5000`. **Value:** non-negative integer

Number of bot-vs-bot curriculum games placed in `curriculum_seed.jsonl` during
run initialization. The Rust backend generates these without NN calls. Set to
`0` to disable seed-buffer generation. If the seed file already exists, normal
initialization does not regenerate it.

### `--replay-window`

**Default:** `20`. **Value:** positive integer. **Iterations basis only.**

Number of recent iteration record files included in training, measured in
iterations rather than games or positions. A larger window improves historical
diversity but makes the dataset less focused on the newest model.

Under `--schedule-basis games` this value is ignored and the window comes from
`--replay-window-coefficient/-exponent/-cap-games` below. It is still validated
and still pinned on resume, because a run that started on the iterations basis
must keep the window it was trained with.

### `--save-buffer`

**Default:** disabled. **Value:** output `.jsonl` path

Atomically exports the replay games available at training exit. Saving happens
after clean completion and is also attempted on Ctrl+C or an exception, without
masking the original failure. The temporary-file-plus-rename write prevents a
partial export from replacing a previous good buffer. A typical setting is:

```powershell
--save-buffer games/seven_wonders_duel/runs/laptop_training_01/buffer_final.jsonl
```

The export contains the current live replay window, the still-retained portion
of any imported warm buffer, and the seed curriculum portion selected for the
latest generated iteration. It stores complete portable game records rather
than Python pickles.

### `--buffer-autosave-every`

**Default:** `0`. **Value:** non-negative integer

Re-exports `--save-buffer` every N completed iterations in addition to the exit
save. `0` saves only on exit. The write is atomic (temporary file plus rename),
so a hard kill mid-save can only leave a stale `.tmp` beside the last valid
export, never a truncated replacement. A failed autosave prints a warning and
training continues -- it never terminates the run. Bounds buffer loss on an
abrupt kill to at most N iterations. Requires `--save-buffer` to be set.

### `--warm-buffer`

**Default:** disabled. **Value:** input `.jsonl` path

Loads a buffer produced by `--save-buffer` before training starts. Imported
records participate immediately and age out according to `--replay-window`,
using their existing iteration metadata. This prevents old-policy data from
remaining in replay forever. Usually pair a warm buffer with a compatible model
checkpoint/run lineage; replay produced by a much stronger model can be a poor
fit for randomly initialized weights. Set `--seed-games 0` if the warm buffer
already contains enough military/science curriculum data and you do not want an
additional fresh bot seed buffer.

### `--warm-buffer-max-staleness`

**Default:** `0` (falls back to `--replay-window`). **Value:** non-negative
integer

Applies an explicit iteration-age filter when importing a warm buffer: games
older than N iterations, measured against the newest numbered iteration in the
import, are dropped at load time. Curriculum records (no iteration number) are
never aged out. Source iteration metadata is preserved exactly -- records are
filtered, never renumbered -- and the actual loaded/retained/dropped counts are
reported. `0` uses the active `--replay-window` as the staleness bound.

### `--allow-stale-targets`

**Default:** off. **Value:** flag

Imports a warm buffer whose `policy_target` values were computed under a
superseded target definition.

`buffer.TARGET_VERSION` versions what the **labels** mean, separately from
`schema`/`spec_version`, which version the codec. The 2026-07-25 sigma change
altered every `policy_target` without touching the codec, so old records stayed
loadable and would have silently mixed two definitions of the objective the
policy head regresses onto. Loading a warm buffer now checks this and refuses,
naming the counts and versions it found.

Age and target version are independent: the check runs *before*
`--warm-buffer-max-staleness`, because a recent record computed under an old
rule is exactly as unusable as an ancient one. One stale record among thousands
still refuses -- the mix is the hazard, not the proportion.

This flag overrides the refusal and warns. Reach for it only when you have
decided the inconsistency is acceptable. The alternative is a fresh buffer, or
re-deriving targets by replay, which stays possible because reading a stale
record is still permitted.

Every buffer produced before 2026-07-25 -- including run 02's
`buffer_final.jsonl` -- is `target_version=1` and will refuse to load.

### `--seed-retain-fraction`

**Default:** `1.0`. **Range:** `0.0–1.0`

Initial fraction of the bot seed buffer mixed into training. It decays linearly
to zero over `curriculum-anneal-iterations`. The selected seed records are
shuffled deterministically each iteration.

### `--curriculum-anneal-iterations`

**Default:** `-1` (auto). **Value:** integer duration, or `-1`

Number of iterations over which both seed-buffer retention and mixed bot
opponents decay from their initial fractions to zero. This gives the early
model structured examples, then transitions toward pure neural self-play.

`-1` fits the duration to **half the planned run** (`max(1, iterations // 2)`),
so the anneal always completes. Any explicit non-negative value is honoured
as-is, and the loop warns when an explicit duration outlives the run -- which
would leave bot curriculum data still mixed into the buffer at the final
iteration.

### `--opponent-fraction`

**Default:** `0.15`. **Range:** `0.0–1.0`

Initial probability that a generated self-play game replaces one neural seat
with a curriculum bot. This probability decays to zero over
`curriculum-anneal-iterations`. Bot types and seats are balanced by job index.

### `--bot-policy-iterations`

**Default:** `10`. **Value:** integer

Iteration cutoff for retaining bot-owned moves as policy examples. At and after
this iteration, bot decisions remain part of mixed-game trajectories but their
moves are marked `policy_excluded`. Network-owned moves remain trainable.

### `--bot-exploration`

**Default:** `0.05`. **Range:** `0.0–1.0`

Probability that a Rust curriculum bot explores among its top candidate moves
instead of selecting its deterministic best move. It increases trajectory
diversity. Rust uses a portable seeded RNG, so runs are deterministic for the
same configuration, but exploratory paths are not bit-identical to the former
Python bots' Mersenne Twister paths.

### `--draft-prior-iterations`

**Default:** `20`. **Value:** integer duration

Linearly anneals the handcrafted draft prior from full strength to zero. It
helps early search before the policy network is useful, then gets out of the
way as the learned prior improves.

### `--workers`

**Default:** `8`. **Value:** positive integer

Worker count for seed-buffer generation and the legacy Python threaded path.
It is not the Rust scheduler shard count; use `rust-scheduler-workers` for that.

### `--process-workers`

**Default:** `0`. **Value:** non-negative integer

Process count for legacy Python generation and gate execution. `0` uses Python
threads. This option has no effect on normal Rust generation or Rust gates.

## Schedules and the Games Clock

Every schedule is expressed in **games**, not iterations, so that
`--games-per-iteration` can change on a resume without moving any schedule
position. The basis is pinned per run and a resume that changes it is refused.

### `--schedule-basis`

**Default:** `games`. **Choices:** `games`, `iterations`

Which clock the schedules read. `games` uses the cumulative game count from the
games ledger; `iterations` uses the iteration index and is retained only so
pre-2026-07-29 runs remain resumable.

The clock is deliberately *games before this iteration generated*, not games
through it: a schedule keyed on the games it is itself deciding how to generate
would differ between a fresh run and a resume of the same iteration.

### `--curriculum-anneal-games`

**Default:** `10000`. **Value:** non-negative integer. **Games basis.**

Games over which the curriculum-bot mix anneals to zero. The measured basis for
10k: the net beat the scripted bots 58.7%/78.0% at iterations 0-9 and
98.6%/91.4% by 30-39, so the bots were exhausted as opponents at roughly 10k
games. The iterations-basis equivalent is `--curriculum-anneal-iterations`.

### `--draft-prior-games`

**Default:** `10000`. **Value:** non-negative integer. **Games basis.**

Games over which the wonder-draft tier prior anneals away. The iterations-basis
equivalent is `--draft-prior-iterations`.

### `--replay-window-coefficient`, `--replay-window-exponent`, `--replay-window-cap-games`

**Defaults:** `16.0`, `0.6`, `20000`. **Games basis.**

The growing replay window: `window_games = coefficient * total_games ** exponent`,
capped. The defaults hold the window near 75% of all games early, when data is
scarce and on-policy, and let it fall toward a fixed span later.

The cap is what the W6.4 preflight sizes host memory against -- the run's
*maximum* window, not its current one -- at a measured 122 KB per retained
`GameRecord`.

### `--hof-opponent-fraction`

**Default:** `0.0`. **Value:** fraction in [0, 1]

Share of generation games played against an archived Hall-of-Fame checkpoint
instead of `current_best`. Zero is the compatibility default and keeps the HOF
write-only. The cloud launch value is **0.15**, chosen explicitly rather than
made a default so old resumes and unrelated tests keep their opponent mix.

Archived opponents are routed by the **searcher** seat, so every leaf of one
search uses the mover's network. Routing on the leaf actor instead would let the
opponent's network evaluate the interior of your own search tree.

### `--hof-sampling-mode`

**Default:** `recency`. **Choices:** `recency`, `uniform`, `latest`

How an archived opponent is drawn. `recency` weights newer archives linearly.

### `--hof-start-games`

**Default:** `10000`. **Value:** non-negative integer

Games before league play begins. Early archives are weak enough that playing
them is closer to curriculum than to opponent diversity.

## Search and Training-Target Quality

### `--cheap-sims-min`, `--cheap-sims-max`

**Defaults:** `16`, `24`. **Constraint:** `1 <= min <= max`

Inclusive random simulation range for cheap-search moves. Cheap moves create
game diversity at lower cost. They are used to choose actions but their policy
targets are excluded from training.

### `--full-sims-min`, `--full-sims-max`

**Defaults:** `64`, `128`. **Constraint:** `1 <= min <= max`

Inclusive random simulation range for full-search moves. Their visit policies
are retained as training targets. Increasing these values usually improves
target quality while reducing games/s roughly in proportion to added search.

### `--full-search-fraction`

**Default:** `0.25`. **Range:** `0.0–1.0`

Probability that a neural move uses the full simulation range. Other moves use
the cheap range. This is a major quality/throughput balance:

```text
recorded policy positions per game
  ~= neural moves per game * full-search-fraction
```

Too low yields many games but few policy targets. Too high yields more
correlated targets per game and substantially lowers trajectory throughput.

### `--search-mode`

**Default:** `closed`. **Values:** `closed`, `open`

Selects the MCTS information model. `closed` is the current production mode and
uses the realized game state. `open` is available for experiments with the
open-loop search implementation; do not switch modes casually because it
changes the algorithm and target distribution.

### `--top-k`

**Default:** `16`. **Value:** positive integer

Maximum Gumbel/search candidate width retained at a root. Larger values admit
more candidate actions but increase root work. It also controls the top-k
metadata stored with search records.

### `--force-root-chance`, `--no-force-root-chance`

**Default:** enabled

Controls forced materialization of root chance outcomes. The enabled path
avoids redundant extra-ply evaluations and passed the exact forced-cache gate.
Leave enabled for normal training; the negative form is useful for ablation.

### `--cheap-double-reveal-offsets`

**Default:** `0` (exhaustive -- the shipped behaviour). **Value:** non-negative
integer

Offsets per first-reveal stratum on pure double card-reveal chance edges, on
**cheap generation moves only**. Zero enumerates the full support; a small value
caps it, trading a fixed-support approximation for search width elsewhere.

Off by default and deliberately so: the fixed-support machinery is built and
reviewed but its arena and training A/B have not been run. `2` is the value
`CHANCE_ENUMERATION_PLAN.md` recommends sweeping first. It reaches generation
only -- gates and evaluation always enumerate exhaustively, so a change here
cannot silently alter how candidates are compared.

### `--age-deal-samples`

**Default:** `32`. **Allowed:** `0`, `4`, `8`, `16`, `32`

Number of paired AgeDeal chance samples used at the real transitions from Age I
to II and Age II to III. It is not applied to initial Age I setup, where player
zero always makes the first wonder pick. More samples reduce chance-outcome
noise but add CPU/search work at those two transitions. `0` disables the paired
AgeDeal sampling treatment. The current default is 32 because lower exploratory
calibrations did not meet the action-agreement target.

### `--leaf-batch`

**Default:** `1`. **Value:** positive integer no larger than the global cap

Number of unique leaves gathered from one game before submitting evaluation.
`1` is the exact, approved production algorithm and matches the sequential
oracle. Values greater than one use WU-UCT batching and must be treated as a
separately approved algorithm, not merely a throughput setting. Keep this at
`1` for training unless that algorithm receives its own quality approval.

## Rust Scheduler and Inference Geometry

### `--generation-backend`

**Default:** `rust`. **Values:** `rust`, `python`

Backend for seed games and self-play. `rust` runs all per-game logic—including
curriculum bots—in Rust and calls Python only at the flat Torch inference
boundary. `python` preserves the slower reference/legacy path.

### `--gate-backend`

**Default:** `rust`. **Values:** `rust`, `python`

Backend for candidate-vs-best and model-vs-bot evaluation games. Evaluation
game logic and bots run in Rust with the default. Torch still evaluates neural
positions.

### `--rust-slots`

**Default:** `16`. **Value:** positive integer

Maximum concurrent game slots in each Rust scheduler call. More slots expose
more leaves for global batching, but increase active tree state and CPU work.
The current laptop sweep selected 16.

### `--rust-global-batch-cap`

**Default:** `256`. **Value:** positive integer

Maximum number of neural rows packed into one flat Torch forward call. It must
be at least `leaf-batch`. Larger caps allow better coalescing when enough work
is ready, but do not force every batch to reach the cap.

### `--rust-max-inflight-batches`

**Default:** `1`. **Value:** positive integer

Maximum Torch batches submitted but not yet completed. One is the verified
laptop setting. Additional inflight batches can overlap work on other hardware,
but also increase queueing, memory use, and scheduling variability.

### `--rust-scheduler-workers`

**Default:** `1`. **Value:** positive integer

Number of persistent Rust scheduler shards. These are scoped standard-library
threads, not Rayon workers and not reserved or pinned CPU cores. Each shard owns
game slots while sharing the evaluator. The laptop sweep selected one worker.

### `--inference-batch`

**Default:** `64`. **Value:** positive integer

Maximum inference batch for the legacy Python generator and Python gate path.
Rust generation instead uses `rust-global-batch-cap`. It is also used when a
legacy Python agent evaluator is constructed.

### `--inference-wait-ms`

**Default:** `2.0`. **Value:** non-negative milliseconds

Maximum coalescing wait for the legacy Python threaded evaluator. It does not
control the Rust flat-batch scheduler.

## Model and Optimizer

### `--d-model`

**Default:** `128`. **Constraint:** positive and divisible by four

Transformer embedding width. It is one of the two main model-capacity knobs.
Wider models use more parameters, VRAM, inference compute, and training compute.
Together with four layers, width 128 produces the current approximately
1.03-million-parameter laptop model.

### `--layers`

**Default:** `4`. **Value:** positive integer

Number of transformer layers. More layers increase representation depth and
roughly linearly increase most trunk parameters and forward-pass work.
Checkpoints require the same architecture when loaded.

### `--heads`

**Default:** unset (read from the checkpoint). **Value:** positive integer

Attention-head count. Left unset it follows the checkpoint being loaded, which
is the safe behaviour: three of W0's harness failures came from a checkpoint's
geometry being assumed rather than read, and one wasted run rebuilt a 256-wide
checkpoint as a 128-wide model.

### `--precision`

**Default:** `fp32`. **Choices:** `fp32`, `bf16`

Precision for model calls on the generation, gate, and validation paths. Pinned
per run: a resume that changes it is refused.

**The benefit is width-dependent, which is easy to get wrong.** W0 measured bf16
at **0.97x** on S's production path (a slowdown), **1.26x** on M, and **1.69x**
on L. "bf16 everywhere" was locked in an earlier revision on a measurement taken
at one width, and that generalisation was wrong. If the run falls back to
S = 128x4, fp32 goes with it.

bf16 is also not merely a rounding of the same policy: it changed 43 of 64 L
trajectories in W0's arena. `precision_arena.py` plays one checkpoint against
itself, bf16 versus fp32, against a known null of 0.500 -- run it before
trusting a precision you have not played a scored game in.

## Host and Device Memory

The loop bounds host memory explicitly because run 03 died at iteration 70 with
a host `MemoryError` inside a gate, and reported it as a channel-disconnect
error from the inference worker rather than as the allocation failure it was.

### `--example-cache-gb`

**Default:** `0` (converts the legacy count). **Value:** GiB

Preferred ceiling for the vectorised-example cache, in calibrated **retained**
bytes. Games already replayed are served from the cache instead of being
re-derived; the cache is why an iteration spends ~11 s training rather than
~23 s re-replaying its window.

### `--example-cache-examples`

**Default:** `250000`. **Value:** non-negative integer

Legacy count-based ceiling, converted at the measured **17.8 KB per example**
(so 250k is about 4.45 GB). `0` disables the cache entirely.

Summing the six numpy arrays in an `Example` gives ~13.1 KB, which understates
the real cost by about 26% -- it excludes the ndarray objects, the dataclass,
cache keys, list overhead, and allocator fragmentation. The bound is therefore
applied to `nbytes x calibration_factor`, with the factor measured against RSS
at startup and never allowed below the measured 17.8/13.1 ratio.

### `--memory-budget-gb`

**Default:** `0` (85% of detected RAM). **Value:** GiB

Host RSS budget. On breach the cache is evicted toward a floor and a
`memory_pressure` event is recorded in the stats row. Slower re-replay beats
losing a run.

### `--memory-headroom-gb`

**Default:** `2.0`. **Value:** GiB

Host memory reserved outside the process budget.

### `--vram-budget-gb`

**Default:** `0` (90% of detected VRAM). **Value:** GiB

Physical device-memory budget, checked before a gate loads its two models.
Reported separately from allocator-*reserved* memory, which exceeded physical
VRAM under Windows/WDDM in W0 and is not a capacity number.

### `--train-steps`

**Default:** `300`. **Value:** positive integer

Optimizer updates per iteration, drawn as uniform random minibatches from the
replay buffer. This replaces the former `--train-epochs`.

An epoch visits every buffered position once, so as the buffer grows old
positions are re-presented on every subsequent iteration while the amount of
*new* data per iteration stays flat. Run 02 reached roughly 113 training
presentations per newly generated position by iteration 11 (2.0M presentations
against ~17.8k new examples); AlphaGo Zero sat near 1-2. A fixed step budget
decouples training cost from buffer size and makes training pressure per unit
of new data an explicit, logged quantity -- see `samples_per_new_position` in
`training_log.jsonl`.

At `--train-batch-size 512`, 300 steps consumes 153,600 samples. Against a
typical ~18k new positions per iteration that is about 8.5x, which is the
KataGo `max-train-bucket-per-new-data` regime.

#### The relationship that must be maintained

`--train-steps` and `--games-per-iteration` are **coupled**. Raising games per
iteration without raising steps drives training pressure toward 1x; raising
steps without raising games repeats run 02's 113x. Watch
`samples_per_new_position` in `training_log.jsonl` and keep it near 4-8:

```text
samples_per_new_position
  = train-steps * train-batch-size
    / (games-per-iteration * positions-per-game * recorded-fraction)
```

Measured on run 02: **71.9 positions per game**. For the recommended run at 300
games and 300 steps this gives 7.1x, and lifetime presentations work out to
~8.5x for a position generated early in the run, ~7.1x on average.

#### `recorded-fraction`: we deviate from KataGo here, deliberately

That term is currently **1.0** -- every position is recorded. Cheap-search
moves are stored with `policy_excluded=True`, so they carry no policy target
but still supply value and auxiliary targets.

KataGo does **not** do this. From Wu (2020) §3.1, Playout Cap Randomization:
"Only turns with a full search are recorded for training." Kingdomino follows
KataGo (`self_play.py:1196`, `record_fast_moves` defaults off and the example
is never appended). Seven Wonders Duel is the outlier.

The paper's reasoning is that the value target is data-limited at *one noisy
binary result per game*, so the way to help value training is to play **more
games**, not to record more positions from each game -- extra positions from
one game share that single label and are highly correlated.

Two consequences of our deviation, both worth knowing before changing it:

* **The policy head sees ~25% of each batch.** With `--full-search-fraction
  0.25`, only a quarter of sampled rows have `has_policy`. The policy loss is
  averaged over those rows so its scale is right, but policy sample throughput
  is a quarter of value sample throughput.
* **Switching to KataGo's rule would shrink the buffer ~4x**, taking
  `samples_per_new_position` from 7.1 to ~28 at unchanged settings. Aligning
  would require `--train-steps` around 75, or ~4x more games per iteration --
  which is what KataGo does, and what a laptop cannot.

So the current setting is a reasonable adaptation to a small games budget, not
an oversight. Revisit it on cloud hardware where more games per iteration is
affordable.

### `--train-warmup-steps`

**Default:** `100`. **Value:** non-negative integer

Linear learning-rate warmup, applied **only** on a cold optimizer. Once
optimizer state is being carried across iterations the learning rate is flat.
Re-warming every iteration would reproduce the sawtooth the old per-iteration
cosine restart already caused.

Optimizer state (AdamW moments) persists in `checkpoints/optimizer_state.pt`
between iterations and is cleared on a revert, when the weights jump backwards
and the accumulated moments no longer describe the loss surface under them.

### `--train-batch-size`

**Default:** `512`. **Value:** positive integer

Examples per gradient update. Larger batches improve GPU utilization but use
more VRAM. With a fixed step budget, a larger batch consumes proportionally
more samples per iteration rather than performing fewer updates.

### `--learning-rate`

**Default:** `2e-4`. **Value:** positive float

Optimizer step size. Raising it accelerates change but increases instability;
lowering it makes updates more conservative. Do not interpret losses across
runs without also checking this value and replay composition.

### `--weight-decay`

**Default:** `1e-4`. **Value:** non-negative float

Optimizer weight regularization. It discourages excessively large parameters
and can reduce overfitting.

### `--aux-weight`

**Default:** `0.2`. **Value:** float

Relative weight applied to auxiliary training objectives alongside the main
policy/value losses. Changing it changes both optimization and the combined
validation score used for early stopping.

### `--validate-every`

**Default:** `100`. **Value:** positive integer

Evaluate the held-out set every N steps (and always on the final step). The
validation snapshot is fixed for the duration of an iteration, so the
checkpoints at step 100, 200 and 300 are compared against identical examples.

### `--restore-best-val` / `--no-restore-best-val`

**Default:** off.

Restore the lowest-validation-loss weights at the end of an iteration. The
`--no-` form is the explicit negation, for a launch script that wants the
setting stated rather than inherited.

This is **off by default and should stay off** until arena games establish that
validation loss predicts playing strength. It was unconditionally *on* in run
02 (inside the old `train_loop`), and from iteration 3 onward the best epoch
was always epoch 0 -- so every candidate shipped was the epoch-0 weights and
the other seven epochs were computed and discarded, about 4,500s of an 8.4h
run.

Validation loss is not a strength measure. A stronger network can score worse
against older, weaker MCTS targets precisely because it disagrees with them.
Use it to detect when further updates stop generalizing; use arena games to
decide what is stronger.

### `--val-fraction`

**Default:** `0.05`. **Typical range:** `0.0–1.0`

Fraction of whole games held out for validation. The split is game-honest: all
positions from one game stay together, preventing near-duplicate positions from
the same trajectory from leaking into both sets.

### `--val-split-salt`

**Default:** `swd-v1`.

Salt for the train/validation assignment hash. A game's side is decided by
`blake2b(salt | iteration | game_key)`, so it stays fixed for as long as the
game lives in the replay window. The previous split reseeded from
`seed + iteration` on every training iteration, which let a game validate at
one iteration, train at the next and validate again at the third --
contaminating the holdout and understating validation loss on older data.
Changing the salt re-draws the whole split; do not change it mid-run.

### `--record-fast-moves`

**Default:** off. **Value:** flag

Emits training examples for cheap-search moves. Off by default, matching KataGo
and Kingdomino.

Wu (2020) §3.1: *"Only turns with a full search are recorded for training."* The
value target is limited by one noisy binary result per **game**, so extra
positions drawn from the same game share that label -- they inflate the buffer
and dilute the share of each batch carrying a policy target, without adding
proportional information.

Fast-search moves are still **recorded in the buffer** and still replayed. They
have to be: the buffer's defining invariant is that `replay(record)` reproduces
every state from `(seed, actions)`, so removing a move would break replay for
every later move in that game. The exclusion happens at the example boundary in
`dataset.examples_from_record`.

Bot moves are *not* affected. `policy_excluded` conflates two things -- a cheap
search and a curriculum bot's move -- distinguished by `sims`, which a bot move
records as zero. Bot moves keep contributing value targets.

Measured on run 02's buffer, excluding fast searches takes recorded positions
from **69.6 to 22.0 per game**; fast searches are 68.3% of all moves, bot moves
7.3%, full searches 24.4%.

**Changing this requires re-sizing `--train-steps`.** Turning it on roughly
triples buffer size; at 300 games/iteration, `--train-steps 75` gives 5.8x with
it off and about 1.8x with it on.

### `--min-buffer-positions`

**Default:** `0` (disabled). **Value:** non-negative integer

Skips training -- generation still runs -- until the replay buffer holds this
many positions. **No promotion or anchor gate runs on a skipped iteration.**

`--train-steps` is a fixed budget, so it presents the same 153,600 samples
(at 300 steps x batch 512) whether the buffer holds 20,000 positions or 400,000.
Against a first iteration of self-play alone that is roughly 7x over a small,
single-policy dataset before any diversity exists.

The bot seed curriculum currently masks this by prefilling the buffer -- with
`--seed-games 2000` the first iteration already trains against ~140,000
curriculum positions. Set this whenever you reduce or remove the curriculum;
`--seed-games 0` without it repeats the failure mode.

Counted in positions rather than games so it self-adjusts to
`--games-per-iteration`; a position is one recorded move, matching what the
dataset emits one example per. Skipped iterations are logged with
`training_skipped` and `training_skip_reason`, and are distinct from a training
*failure*, which still raises.

The gate suppression matters independently: comparing an untrained learner
against the protected best spends evaluation games re-measuring a checkpoint
that has not moved.

### `--min-games-to-train`

**Default:** `2`. **Value:** integer

Minimum number of available replay games required before candidate training.
This is principally a safety check for smoke tests and damaged/empty buffers.

## Search Constants

### `c_visit` / `c_scale`

**Defaults:** `50.0` / `0.1`.

The Gumbel-AlphaZero sigma factor, `(c_visit + max_visits) * c_scale`, applied
to the completed Q values of the root's legal actions.

`c_scale` was `1.0` until 2026-07-25, and sigma was applied to a **raw**
actor-relative Q in [-1, 1]. That made sigma span +/-50 while log-prior
differences are ~1-3, so the completed-Q term swamped the prior by more than an
order of magnitude and the improved policy was effectively independent of what
the network had learned about move preferences.

Sigma now min-max rescales completed Q to [0, 1] across the root's legal
actions first, matching the paper and mctx's
`qtransform_completed_by_mix_value` (whose `value_scale` default is the 0.1
adopted here). Two properties follow, both covered by tests: sigma is invariant
to shifting or scaling the Q window, so a decision scores the same in a winning
position as in a losing one; and when no search information exists yet -- every
action sharing the root value -- sigma collapses to zero and the improved
policy is exactly the prior.

**This changes `policy_target`, which is a training target.** Targets recorded
before this change are not comparable with new ones; start a run from a fresh
buffer rather than mixing them.

This is now enforced rather than advisory: records carry
`buffer.TARGET_VERSION`, and importing a warm buffer that predates the change
refuses unless `--allow-stale-targets` is passed. Bump `TARGET_VERSION`
whenever the meaning of `policy_target`, `root_value` or the value label
changes.

### Action selection in evaluation

Self-play plays a temperature-annealed sample from `policy_target`. Competitive
play (gate, arena, anchors) plays `argmax(policy_target)`.

It must not play `SearchResult.action_index`: that is the Gumbel-perturbed
selection, and Gumbel keys are exploration noise. At low simulation counts the
sigma term is near-constant across unvisited actions and the selection reduces
to `argmax(gumbel + log_prior)`, which by the Gumbel-max trick is an exact
*sample* from the prior rather than its argmax. `deterministic_actions=true`
only suppresses the additional temperature sampling; it does not remove this.

## Promotion and Anchor Evaluation

### `--gate-sims`

**Default:** `64`. **Value:** positive integer

Search simulations per neural move in candidate-vs-best and bot-anchor games.
This is independent of the self-play cheap/full simulation ranges. Higher
values reduce evaluation search noise but make gates slower.

**Deeper gates measure a smaller margin.** Search is a policy-improvement
operator, and it extracts more from a weak prior than a strong one, so a
head-to-head compresses as both sides get more of it. Measured on run 02's
iteration 0 vs iteration 11, matched conditions, 400 games per point, only
`sims` varying:

| gate-sims | iter11 score | iter11 Elo edge |
|----------:|-------------:|----------------:|
| 2         | 0.805        | +246            |
| 64        | 0.650        | +107            |

Non-overlapping CIs. Gating at 64 while self-play generates at `cheap_sims`
16-24 therefore judges a regime the training data never comes from, and sees
roughly **43%** of the improvement generation-regime play would show. That
systematically under-credits policy-head gains, which is what a bootstrapping
run mostly produces.

This is a genuine trade-off, not a bug to fix. Gating at generation sims makes
the ratchet responsive to the improvements actually being made; gating high is
the more honest measure of deployed strength if the end product searches
deeply. Note also that **widening `--gate-indifference` does not compensate** --
compression shrinks the measured margin while a wider band demands a larger
one, so the two push in the same direction.

**Caveat: one knob drives both.** `--gate-sims` sets the simulation count for
the promotion gate *and* the bot-anchor suite, so the natural split -- ratchet
at generation sims, anchors deep for true-strength tracking -- is not currently
expressible. Splitting it would need a separate `--anchor-sims`.

### `--eval-search-mode`

**Default:** `gumbel`. **Values:** `gumbel`, `puct`

Root selection for **evaluation** games -- promotion gate, arena, and bot
anchors. Self-play is unaffected and always uses the Gumbel root.

The Gumbel root (top-k + sequential halving) exists to make a small fixed
simulation budget yield an unbiased policy-improvement **target**. Evaluation is
not building a target, and the Gumbel keys are exploration noise that perturb
which candidates get searched at all. `puct` selects at the root by PUCT, like
every node below it, and plays argmax visits.

**This is the search the advisor runs** (`advisor_adapter` calls `descend()` at
`c_puct` 1.5, never `_gumbel_root`). If the advisor is the product, `puct` is
what a gate must measure for its numbers to mean advisor strength.

Default stays `gumbel` because switching changes what every gate number means;
results are not comparable across the two modes. `eval_suite` includes the mode
in each match fingerprint, so changing it re-runs rather than silently reusing a
cached result.

Requires `leaf-batch 1` (which evaluation already uses). Above that the root
would select under WU virtual loss -- a different algorithm, not a throughput
setting, per the `--leaf-batch` policy. The Rust side raises rather than
silently approximating.

Measured cost: none. 32 games at 64 sims took 217s under `puct` against 219s
under `gumbel`.

### The promotion rule

A gate plays a **fixed** number of games, chosen before the match, and then
decides **once**:

| outcome | condition |
|---|---|
| promote | pair-level Wilson **LCB** > `--promotion-min-lcb` |
| revert | pair-level Wilson **UCB** < `--revert-max-ucb` |
| probation | otherwise -- the interval still spans the band |

The observation unit is the **seat pair**, not the game: paired games share a
seed, and Wilson assumes independence, so pairing makes the bound exact rather
than mildly anti-conservative. A pair scores in {0, 0.5, 1}.

**There is no mid-match stopping.** An earlier version evaluated both boundaries
after every pair and stopped on the first crossing. That is optional stopping:
simulated at 40k trials with a 10% draw rate, it promoted an *evenly matched*
candidate 14.9% of the time at a 200-game cap and 19.1% at 800, against ~2% for
the same rule read once at a fixed size. The larger cap made it worse, which
inverts the entire argument for a larger cap. Nothing in the system ever undoes
a promotion, so this matters.

**The cap buys power, not safety.** Probability of promoting, by the candidate's
true per-game strength (simulated, fixed-N, z=1.96):

| games | q=0.52 | q=0.55 | q=0.60 | q=0.65 |
|---:|---:|---:|---:|---:|
| 100 | -- | 8.9% | 25.1% | 50.6% |
| 200 | 4.6% | 13.4% | 43.9% | 79.7% |
| 400 | 6.5% | 24.0% | 74.0% | 97.8% |
| 800 | 9.5% | 42.2% | 95.9% | ~100% |

The false-positive column is flat at ~2% at every size. That is what makes a
small early gate safe.

### `--promotion-min-lcb`

**Default:** `0.50`. **Value:** fraction in [0, 1]

Promote when the pair-level Wilson lower bound clears this. At z=1.96 that is
~97.5% one-sided confidence against an equal candidate.

Note the requirement in *pair* units: 0.598 at 200 games, 0.569 at 400, 0.549 at
800. Expected pair points equal the per-game win rate, so these are directly
comparable to a win rate -- but they are one doubling worse than the same table
computed per game, which is a correction to earlier planning figures.

### `--revert-max-ucb`

**Default:** `0.48`. **Value:** fraction in [0, 1]. Must not exceed
`--promotion-min-lcb`.

Revert when the pair-level Wilson **upper** bound falls below this -- a
confidence bound, not a point estimate. The previous `--revert-win-rate` compared
the raw rate against 0.48 and reverted an evenly matched candidate **32%** of the
time at 200 games and **35%** at 100. The UCB form leaves an equal candidate in
probation ~97% of the time at every size.

The threshold sits *below* `--promotion-min-lcb` deliberately. A fixed threshold
gets more sensitive as the ladder raises the game count, and mild regression
during training is normal -- particularly across an LR or curriculum-mix knot --
so the recoverable direction is the less trigger-happy one.

### `--gate-confidence-z`

**Default:** `1.96`. **Value:** positive

Wilson z for both bounds. Leave it at 1.96 and tune the gate **size** instead;
two interacting dials make the promote and revert behaviour hard to reason about.

### `--gate-ladder-games`

**Default:** empty (one rung at `--gate-max-games`). **Value:** ascending even
integers, e.g. `100 200 400 800`

Scheduled gate sizes. The ladder steps **up** one rung after
`--gate-ladder-step-up-after` consecutive probations and **down** one rung after
a promotion. Early in a run the learner improves fast and a small gate resolves
it; late in a run the candidate is +2-3% and the evidence has to be bought.

Choosing the size from *prior* gates and the clock is sound in a way mid-match
stopping is not: the size is fixed before the games are played, and the candidate
is a different net every gate.

### `--gate-ladder-step-up-after`

**Default:** `2`. **Value:** positive integer

Consecutive probations before the gate steps up a rung. A revert holds the rung
rather than dropping it, so the gate that could confirm it runs at the resolution
that produced it.

### `--gate-ladder-floor-games`

**Default:** `0`. **Value:** non-negative integer

Games that must exist before the ladder may step up at all. Bootstrap gates are
noisy and their probations are not evidence of stagnation. Below the floor the
probation counter is held at zero rather than accumulating, so clearing the floor
starts the count fresh instead of cashing in a backlog.

### `--gate-revert-suppress-knots`

**Default:** empty. **Value:** games-clock points

Extra points after which one gate may not revert. Schedule-driven knots (the
curriculum finishing, the draft prior finishing, HOF starting) are derived
automatically; this is for a disruption the config cannot see, such as an LR
change made on resume. Promotion is never suppressed -- a candidate that clears
the LCB right after an LR change is genuinely better.

### `--gate-slots`

**Default:** `48`. **Value:** positive integer

Rolling active-game slots used only by promotion gates. Both seat legs share one
pool, so occupancy moves in units of two games; keep it even.

What matters is `gate_slots x leaves-in-flight` filling
`--rust-global-batch-cap`, not any relationship to the gate size. Keep the gate
size well above the slot count -- at 48 slots a 100-game gate is only ~2
pool-fills deep, which is where the end-of-gate drain starts to cost.

### `--gate-max-games`

**Default:** `400`. **Constraint:** positive even integer

Gate size when `--gate-ladder-games` is empty, and the size the W5.7 cost bench
uses. With a ladder configured this is unused.

### `--gate-alpha`, `--gate-beta`, `--gate-indifference`

**Defaults:** `0.05`, `0.05`, `0.03`. **Vestigial.**

SPRT error targets and indifference half-width. They are still accepted and
still recorded in the manifest, but **no promotion gate reads them** -- the rule
above replaced SPRT entirely. They remain only so old run configurations parse.

### `--anchor-games`

**Default:** `200`. **Constraint:** positive even integer

Fixed games per bot-anchor opponent. Anchors never early-stop: they are
measurements, not decisions, and early stopping biases a score toward whichever
boundary stopped it. With five curriculum bots plus the greedy bot this is
`6 x --anchor-games` per anchor gate, which is a real cost -- and it is not
covered by the gate-cost fit, which measures promotion gates only.

Anchors are also the **only** input to the Elo ladder. Promotion gates no longer
feed it: a run whose `elo/elo_games.jsonl` contains model-vs-model rows has
regressed.

### `--anchor-gate-every-promotions`

**Default:** `3`. **Value:** non-negative integer

Runs the fixed bot-anchor suite after every N successful promotions. `3` means
after the third, sixth, ninth, and subsequent promotions. `0` disables periodic
anchor gates. Anchor results characterize progress and Phase D exit criteria;
they do not block the candidate-vs-best strength ratchet.

Note this cadence is keyed to *promotions*, so it never fires in a run where
nothing is promoted. Run 02 promoted nothing across 12 iterations and therefore
never measured the bot suite at all. Prefer `--anchor-every-iterations` when
promotions are rare or the gate is disabled.

### `--anchor-every-iterations`

**Default:** `0` (promotion-keyed only). **Value:** non-negative integer

Also runs the bot-anchor suite every N iterations regardless of promotions,
measuring `latest.pt` (the rolling learner) rather than `current_best.pt`.

The bot suite is the only opponent set outside the self-play distribution, and
the two can move in opposite directions: run 02's iteration 11 beat iteration 0
head-to-head while *losing* ground against two of the five scripted bots
(science_aggressive 90% -> 74%, military_aggressive 80% -> 70%). A head-to-head
promotion number on its own does not establish that a checkpoint got better.

Both figures above were recorded under the pre-2026-07-25 evaluation defects.
The head-to-head has since been re-measured on the same 400 seeds and is
**0.650**, not the 0.595 originally recorded. The bot rows were never affected
by the seat-routing bug -- they use a single-net adapter -- but they were still
played with Gumbel-perturbed actions under unnormalised sigma, so treat them as
indicative and re-measure before drawing conclusions.

## Stagnation Detection and Response

Elo cannot detect stagnation in this loop: every candidate plays only its own
`current_best`, so the ladder has no fixed reference. A promotion-lagged anchor
fails too, and fails exactly when it matters -- when promotions stop, both
pointers freeze, and the score becomes a constant no threshold crosses.

### `--self-anchor-games`

**Default:** `0` (off). **Constraint:** non-negative even integer

Fixed games for the self-anchor measurement. The opponent is whatever
`current_best` was `--self-anchor-lag-games` ago.

**Why games and not promotions.** The anchor is a time-shifted pointer into the
history of bests, re-evaluated every measurement. The target (`now - lag`)
advances every iteration; the history only grows on a promotion. So when
promotions stop, the pointer keeps walking until it reaches `current_best`
itself -- and then the measurement is a net against itself, which scores exactly
**0.500**. The null calibrates itself and cannot go stale.

A caught-up anchor is reported rather than played: with paired seeds, both seats
swapped, and deterministic gate play, the two games of a pair are the same game,
so the subject wins one and loses one. Playing it would spend a whole gate
rediscovering a tie.

### `--self-anchor-lag-games`

**Default:** `20000`. **Value:** positive integer

How far back the anchor sits. Also sets how long a frozen best takes to be
caught: with no promotions, the anchor reaches `current_best` one lag after it
was promoted.

### `--self-anchor-every-games`

**Default:** `10000`. **Value:** positive integer

Games between measurements. Detection needs at least three, so this sets how
long a fresh run takes before stagnation can be reported at all.

### Detection

Two independent triggers, both requiring several measurements -- never a single
point, because one anchor score straddling 0.500 is what an evenly matched pair
of nets looks like on any given afternoon:

- **interval**: the recent intervals all include 0.500, so no measurement
  distinguishes the current net from its own past self;
- **slope**: the OLS slope of score against the games clock is at or below
  0.005 per 10k games.

The slope is per *games*, not per measurement: an intervention that lengthens
the window changes the cadence, and a per-measurement slope would then mean two
different things inside one series.

The anchor is also W0's falsifier. The width decision rests on 14.9M parameters
having more headroom under continued self-play than 1.03M -- a claim no
fixed-corpus study can test. A flat anchor slope at L's cost (1.85x generation,
5.12x training step) is the documented case for the S/fp32 fallback.

### `--intervention-ladder`

**Default:** off. **Flag.**

Enables the response to detected stagnation. Detection reports either way; this
only controls whether anything acts on it. Four rungs, escalated one at a time:

1. **raise the search budget** (sims x1.5, full-search fraction x1.25) -- the
   most principled first response, since stagnation usually means the
   search-improved policy is no longer better than the raw policy at that budget;
2. **jump the replay window** (x1.5);
3. **raise the HOF fraction** (to 0.30);
4. **LR warm restart** (x3).

Model growth is deliberately not on this ladder; size is manual.

Rungs are **exclusive** and always applied to the configuration as launched,
never stacked -- stacking would make the second rung's effect unattributable,
which is the whole reason for a measurement window between rungs. A recovery
drops the rung; an exhausted ladder keeps the last one rather than making a
fifth uncontrolled change on top of the four that did not work.

Leave it off for a run whose purpose is to measure something. A rung that fires
mid-series changes the regime the anchor is measuring, and it re-prices the run
(rung 1 raises generation cost by ~1.5x, which invalidates any gate-share
budget computed beforehand).

### `--intervention-window-games`

**Default:** `20000`. **Value:** positive integer

Games a rung is held before its effect is judged.

## Training Lifecycle

These arguments select how the loop treats the learner, self-play generator, and
protected best checkpoint. The default (`strict_gate`) reproduces the historical
Phase D lifecycle exactly; the soft-gate modes route the run through the shared
`games.az_loop` controller so a candidate becomes the next learner even when a
short promotion check is inconclusive.

### `--selfplay-generator-mode`

**Default:** `strict_gate`. **Choices:** `latest`, `current_best`, `strict_gate`,
`soft_gate`

Chooses the generator/learner policy. `strict_gate` preserves the legacy
lifecycle: self-play uses `current_best.pt` and a candidate affects generation
only after an SPRT `accept`. `soft_gate` (recommended for new bootstrap training)
keeps a rolling learner in `latest.pt`, generates with `latest` while
probationary, promotes to `current_best.pt` on `accept`, and reverts generation
to `current_best.pt` after a `reject`. `latest` always generates with the rolling
learner; `current_best` always generates with the protected best (useful for
controlled ablations).

### `--bootstrap-policy`

**Default:** `gate`. **Choices:** `auto_first_trained`, `gate`

`auto_first_trained` installs the first successfully trained learner as both
`latest.pt` and `current_best.pt` without a strength gate, so a fresh run escapes
the untrained iteration `-1` checkpoint immediately. `gate` preserves the old
behavior of gating the first candidate against the untrained baseline. Only
consulted by the soft-gate modes.

### `--promotion-every`

**Default:** `4`. **Value:** non-negative integer

Runs the paired-SPRT promotion check after every N completed training
iterations. `1` gates every iteration; `0` disables automatic promotion checks
(the learner still advances). Non-gated iterations log
`promotion_action: "not_scheduled"` and never touch the consecutive-revert
counter.

**Size the gate before you schedule it.** A gate that cannot resolve its own
indifference region costs games and returns nothing. Run 02 ran
`--promotion-every 1` with `--gate-max-games 100` against a 3% indifference
region: 11 of 11 candidates returned `probation`, `current_best` never moved
off iteration 0, and the gate games consumed roughly as much wall clock as all
training combined.

That was arithmetic, not luck. Under Wald's approximation a candidate sitting
at the H1 boundary (53%) needs about **368** games to be accepted at
alpha = beta = 0.05, and a candidate that is genuinely *equal* to the best sits
at the midpoint of the indifference region where the log-likelihood ratio has
no drift -- no finite budget ever reaches a boundary. The configuration warns
at startup (`UnderpoweredGateWarning`) with the number it would need.

Three workable configurations:

| Intent | Settings |
|---|---|
| Real gate | `--promotion-every 4 --gate-max-games 800` |
| Coarse gate | `--promotion-every 4 --gate-indifference 0.10 --gate-max-games 200` |
| No gate | `--promotion-every 0` (soft-gate rolling learner; spend the games on self-play) |

### `--revert-reset-after`

**Default:** `0`. **Value:** non-negative integer

In `soft_gate`, after this many **consecutive** `reject` gate checks the learner
weights are reset to `current_best.pt` before the next training phase. Earlier
rejects only switch generation to the protected best while the learner keeps
training on recovery data. `0` disables automatic learner reset. The counter is
measured in gate checks, not iterations, and any `probation` or `promote` resets
it.

### `--allow-resume-code-drift`

**Default:** off. **Flag.**

Permits a resume whose commit -- or whose uncommitted diff -- differs from the
one the run started on.

Resuming is normally done by re-running the setup script, which pulls first, so
without this guard an update landing between two iterations would silently split
a run across two engines and attribute every later measurement to whichever
commit the manifest happened to record. The commit alone is not enough: a dirty
tree at the same SHA is different code, so a digest of the launch-time diff is
compared too.

Pass it when the drift is intentional and acknowledged -- continuing an older
run under a new rule, for instance. The run's rows then genuinely span more than
one engine, and comparisons across the boundary need that in mind.

Precision and schedule positions have their own guards and are **not** covered
by this flag; those refuse unconditionally, because changing them mid-run
invalidates the data rather than merely complicating its provenance.

### `--run-log`

**Default:** `<run-dir>/run.log`. **Value:** transcript path

Path for the human-readable run transcript. Everything printed during a run is
mirrored to both the console and this file, so a live run can be followed and
warnings, gates, checkpoints, stalls, and crashes diagnosed without
shell-specific redirection (`Tee-Object`/`tee`/`nohup`). The file is appended on
resume (a new delimited invocation header is written, prior output is never
truncated), uses UTF-8 with normalized `\n` newlines, and is flushed per line.
On a crash the transcript records a termination block with the traceback and the
original error still propagates. This is separate from `training_log.jsonl`; the
structured log and manifest are written independently.

### `--no-run-log`

**Default:** off. Disables the human-readable transcript (console only).
Intended for tests and embedding. Does not affect `training_log.jsonl` or
`run_manifest.json` persistence.

## Important Relationships

```text
neural self-play wall time
  ~= games-per-iteration / measured games-per-second

full-search policy targets per iteration
  ~= games-per-iteration
     * average neural moves per game
     * full-search-fraction

maximum theoretical scheduler leaf rows
  ~= rust-slots * leaf-batch

actual Torch batch rows
  <= rust-global-batch-cap

training history represented by live replay
  ~= replay-window * games-per-iteration          (iterations basis)
  ~= min(cap, coefficient * total_games ** exponent)   (games basis)

initial curriculum contribution
  = bot seed records * seed-retain-fraction
  + new games * opponent-fraction

promotion gate wall time
  ~= fixed overhead + per-game search cost
  measured at 128x4: ~7.7 min for 200 games at gate-sims 64
  (an iteration without a gate is ~8.4 min on the same laptop)

anchor gate wall time
  ~= 6 opponents * anchor-games * per-game search cost

peak host memory
  ~= 122 KB * max_window_games
   + example cache ceiling
   + ~2 GB process
```

Do not tune `leaf-batch` as ordinary geometry: values above one change the
search algorithm. Safe laptop throughput tuning should first vary
`rust-slots`, `rust-scheduler-workers`, `rust-global-batch-cap`, and
`rust-max-inflight-batches` while holding `leaf-batch=1`.

## Suggested Configurations

| Use case | Iterations | Games/iteration | Seed games | Gate size | Model | Notes |
|---|---:|---:|---:|---|---|---|
| Plumbing only | overridden | 2 | 8 | 2 | 32 × 1 | Add `--plumbing-smoke`; not training |
| Short CUDA validation | 2 | 50 | 250 | 20 | 128 × 4 | Checks records, losses, resume, and gates |
| Laptop pilot | 5 | 250 | 500 | 100 | 128 × 4 | Recommended first meaningful run |
| Longer laptop run | 30 | 400 | 2,000 | ladder 100–800 | 128 × 4 | ~8.4 min/iteration measured; add ~7.7 min per gate |
| Cloud launch | 60+ | re-sweep | 5,000+ | ladder, ceiling from the per-rung fit | 384 × 8 × 6, bf16 | See `CLOUD_TRAINING_PLAN.md`; scheduler geometry from that box's sweep |

Here `128 × 4` means `d-model=128` and `layers=4`, not channels and residual
blocks. Before a cloud run, re-sweep scheduler geometry on that hardware and
choose model size based on measured end-to-end throughput rather than GPU
utilization alone.

### Recommended soft-gate command (new bootstrap runs)

The default lifecycle is `strict_gate` for backward compatibility. New training
runs should use the soft-gate lifecycle, which keeps a cumulative rolling learner
so an inconclusive short promotion check no longer discards it:

```powershell
./.venv/Scripts/python.exe -m games.seven_wonders_duel.phase_d `
  --run-dir games/seven_wonders_duel/runs/<run_name> `
  --schedule-basis games `
  --selfplay-generator-mode soft_gate `
  --bootstrap-policy auto_first_trained `
  --promotion-every 5 `
  --revert-reset-after 2 `
  --promotion-min-lcb 0.50 `
  --revert-max-ucb 0.48 `
  --gate-ladder-games 100 200 400 800 `
  --gate-ladder-floor-games 10000 `
  --self-anchor-games 200 `
  --generation-backend rust `
  --gate-backend rust `
  --device cuda `
  --leaf-batch 1 `
  --save-buffer games/seven_wonders_duel/runs/<run_name>/buffer_final.jsonl `
  --buffer-autosave-every 1
```

Fill in the game, search, model, and scheduler budgets from the sized laptop
configuration (see the table above); the flags shown are the lifecycle and gate
controls that differ from a legacy strict-gate run. `run.log` is written under
the run directory automatically.

`--revert-reset-after 2` is the two-stage revert: the first reverting gate
switches the generator back to the protected best while the learner keeps its
weights and keeps training, and only a **second consecutive** reverting gate
rolls the learner back. Requiring persistence rather than lowering the threshold
is what separates a transient dip -- normal across a schedule knot -- from a real
regression.
