# Seven Wonders Duel — Training Parameter Reference

This document describes every command-line argument exposed by the Phase D
training pipeline in `games.seven_wonders_duel.phase_d`.

## Recommended Laptop Command

```powershell
.\.venv\Scripts\python.exe -m games.seven_wonders_duel.phase_d `
  --run-dir games\seven_wonders_duel\runs\laptop_training_01 `
  --device cuda `
  --generation-backend rust `
  --gate-backend rust `
  --iterations 5 `
  --games-per-iteration 250 `
  --seed-games 500 `
  --save-buffer games\seven_wonders_duel\runs\laptop_training_01\buffer_final.jsonl `
  --d-model 128 `
  --layers 4 `
  --train-epochs 8 `
  --train-batch-size 512 `
  --learning-rate 2e-4 `
  --cheap-sims-min 16 `
  --cheap-sims-max 24 `
  --full-sims-min 64 `
  --full-sims-max 128 `
  --full-search-fraction 0.25 `
  --rust-slots 16 `
  --rust-scheduler-workers 1 `
  --rust-global-batch-cap 256 `
  --rust-max-inflight-batches 1 `
  --leaf-batch 1 `
  --age-deal-samples 32 `
  --gate-sims 64 `
  --gate-max-games 100 `
  --anchor-gate-every-promotions 3
```

PowerShell uses the backtick at the end of a line as its continuation
character. There must be no spaces after a continuation backtick.

The command above is a first meaningful laptop run, not a final-strength
configuration. At the measured neural self-play rate of about 0.11 games/s,
the 1,250 neural self-play games alone may take roughly three hours. Training
and promotion games add to that time. Rust bot-only seed games are much faster
because they do not invoke the neural network.

## What One Iteration Does

Each iteration performs these steps:

1. Load `current_best.pt`.
2. Generate `games-per-iteration` new games.
3. Combine recent iteration files with the retained bot seed curriculum.
4. Split whole games into training and validation sets.
5. Train a candidate for up to `train-epochs` epochs.
6. Compare the candidate with `current_best.pt` using a paired SPRT gate.
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

**Default:** `20`. **Value:** positive integer

Number of recent iteration record files included in training. This is measured
in iterations, not games or positions. A larger window improves historical
diversity but makes the dataset less focused on the newest model.

### `--save-buffer`

**Default:** disabled. **Value:** output `.jsonl` path

Atomically exports the replay games available at training exit. Saving happens
after clean completion and is also attempted on Ctrl+C or an exception, without
masking the original failure. The temporary-file-plus-rename write prevents a
partial export from replacing a previous good buffer. A typical setting is:

```powershell
--save-buffer games\seven_wonders_duel\runs\laptop_training_01\buffer_final.jsonl
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

### `--seed-retain-fraction`

**Default:** `1.0`. **Range:** `0.0–1.0`

Initial fraction of the bot seed buffer mixed into training. It decays linearly
to zero over `curriculum-anneal-iterations`. The selected seed records are
shuffled deterministically each iteration.

### `--curriculum-anneal-iterations`

**Default:** `10`. **Value:** integer duration

Number of iterations over which both seed-buffer retention and mixed bot
opponents decay from their initial fractions to zero. A value of `10` gives the
early model structured examples, then transitions toward pure neural self-play.

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

### `--train-epochs`

**Default:** `8`. **Value:** positive integer

Maximum full passes over the iteration's combined training dataset. Early
stopping can end training sooner based on validation performance.

### `--train-batch-size`

**Default:** `512`. **Value:** positive integer

Examples per gradient update. Larger batches improve GPU utilization but use
more VRAM and perform fewer optimizer updates per epoch.

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

### `--train-patience`

**Default:** `8`. **Value:** positive integer

Number of epochs without validation improvement tolerated before early
stopping. With the default eight training epochs, a patience of eight normally
allows the entire requested training phase.

### `--val-fraction`

**Default:** `0.1`. **Typical range:** `0.0–1.0`

Fraction of whole games held out for validation. The split is game-honest: all
positions from one game stay together, preventing near-duplicate positions from
the same trajectory from leaking into both sets.

### `--min-games-to-train`

**Default:** `2`. **Value:** integer

Minimum number of available replay games required before candidate training.
This is principally a safety check for smoke tests and damaged/empty buffers.

## Promotion and Anchor Evaluation

### `--gate-sims`

**Default:** `64`. **Value:** positive integer

Search simulations per neural move in candidate-vs-best and bot-anchor games.
This is independent of the self-play cheap/full simulation ranges. Higher
values reduce evaluation search noise but make gates slower.

### `--gate-max-games`

**Default:** `400`. **Constraint:** positive even integer

Maximum games for each SPRT match. Games are evaluated in paired seat-swapped
legs, so stopping is checked only after a complete pair. The test often stops
before the cap when evidence is decisive.

### `--gate-alpha`

**Default:** `0.05`. **Value:** probability

SPRT false-accept error target: the tolerated probability of accepting the
candidate under the lower-strength hypothesis. Reducing it demands more
evidence and usually more games.

### `--gate-beta`

**Default:** `0.05`. **Value:** probability

SPRT false-reject error target: the tolerated probability of rejecting the
candidate under the higher-strength hypothesis. Reducing it also makes the
test more conservative and usually longer.

### `--gate-indifference`

**Default:** `0.03`. **Value:** positive fraction

Half-width around each target score rate used to form the two SPRT hypotheses.
For promotion at a target of 0.50, the default compares approximately 0.47
against 0.53. A narrower band distinguishes smaller improvements but needs more
games; a wider band resolves faster but ignores small changes.

### `--anchor-gate-every-promotions`

**Default:** `3`. **Value:** non-negative integer

Runs the fixed bot-anchor suite after every N successful promotions. `3` means
after the third, sixth, ninth, and subsequent promotions. `0` disables periodic
anchor gates. Anchor results characterize progress and Phase D exit criteria;
they do not block the candidate-vs-best strength ratchet.

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

**Default:** `1`. **Value:** non-negative integer

Runs the paired-SPRT promotion check after every N completed training
iterations. `1` gates every iteration; `0` disables automatic promotion checks
(the learner still advances). Non-gated iterations log
`promotion_action: "not_scheduled"` and never touch the consecutive-revert
counter.

### `--revert-reset-after`

**Default:** `0`. **Value:** non-negative integer

In `soft_gate`, after this many **consecutive** `reject` gate checks the learner
weights are reset to `current_best.pt` before the next training phase. Earlier
rejects only switch generation to the protected best while the learner keeps
training on recovery data. `0` disables automatic learner reset. The counter is
measured in gate checks, not iterations, and any `probation` or `promote` resets
it.

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
  ~= replay-window * games-per-iteration

initial curriculum contribution
  = bot seed records * seed-retain-fraction
  + new games * opponent-fraction
```

Do not tune `leaf-batch` as ordinary geometry: values above one change the
search algorithm. Safe laptop throughput tuning should first vary
`rust-slots`, `rust-scheduler-workers`, `rust-global-batch-cap`, and
`rust-max-inflight-batches` while holding `leaf-batch=1`.

## Suggested Configurations

| Use case | Iterations | Games/iteration | Seed games | Gate cap | Model | Notes |
|---|---:|---:|---:|---:|---|---|
| Plumbing only | overridden | 2 | 8 | 2 | 32 × 1 | Add `--plumbing-smoke`; not training |
| Short CUDA validation | 2 | 50 | 250 | 20 | 128 × 4 | Checks records, losses, resume, and gates |
| Laptop pilot | 5 | 250 | 500 | 100 | 128 × 4 | Recommended first meaningful run |
| Longer laptop run | 10 | 500 | 5,000 | 200–400 | 128 × 4 | Multi-hour/overnight; inspect pilot first |
| Cloud starting point | re-sweep | re-sweep | 5,000+ | 400 | choose after sizing | Re-sweep scheduler and model geometry |

Here `128 × 4` means `d-model=128` and `layers=4`, not channels and residual
blocks. Before a cloud run, re-sweep scheduler geometry on that hardware and
choose model size based on measured end-to-end throughput rather than GPU
utilization alone.

### Recommended soft-gate command (new bootstrap runs)

The default lifecycle is `strict_gate` for backward compatibility. New training
runs should use the soft-gate lifecycle, which keeps a cumulative rolling learner
so an inconclusive short promotion check no longer discards it:

```powershell
.\.venv\Scripts\python.exe -m games.seven_wonders_duel.phase_d `
  --run-dir games\seven_wonders_duel\runs\<run_name> `
  --selfplay-generator-mode soft_gate `
  --bootstrap-policy auto_first_trained `
  --promotion-every 1 `
  --revert-reset-after 3 `
  --generation-backend rust `
  --gate-backend rust `
  --device cuda `
  --leaf-batch 1 `
  --save-buffer games\seven_wonders_duel\runs\<run_name>\buffer_final.jsonl `
  --buffer-autosave-every 1
```

Fill in the game, search, model, and scheduler budgets from the sized laptop
configuration (see the table above); the flags shown are the lifecycle controls
that differ from a legacy strict-gate run. `run.log` is written under the run
directory automatically.
