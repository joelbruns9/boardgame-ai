# Chance-progressive full training: cloud runbook

This is the executable handoff for the approved full-cycle experiment. It
implements the frozen decisions in
`CHANCE_AWARE_FULL_TRAINING_REVIEW_RESPONSE.md`; it does not rent an instance or
start a run by itself.

The original G3 gate against open-loop `x0` remains a recorded failure. The
owner explicitly approved the 2026-08-11 amendment that treats G3 as the
incremental progressive-vs-pilot gate. Under that comparator G3 passed. This
does not retroactively claim that the chance-split family beat `x0` on the
eight-position boundary screen.

## 1. Frozen experiment

- Source: exact reviewed commit on `codex/kingdomino-chance-correct`; never
  `main` and never an uncommitted working tree.
- Baseline: `runs/kingdomino/best_checkpoint/current_best.pt`, SHA-256
  `4bf07b0ca14e5452e6533a9232967e89bb0ab0df88c99e9928a65f402b1f04b3`.
- Search treatment: decks 8 and 12, `N_init=2`, `W0=4`, `D_min=4`, schedule
  `4,8,16,32,64,70`, cap 16 at both decks, initialization guardrail 25%.
- Replay weighting: 1.0x (neutral).
- Revised Phase-B shape: first generate 23 × 400 chance-aware games with the
  frozen baseline and **zero optimizer steps**, then train for 24 × 400 games
  at 192 optimizer steps per iteration (about 5.6× steady-state replay reuse).
  Search uses 4,800 full-search sims, 200 fast sims, 25% full-search moves;
  80x6 network; fresh 200,000-example buffer; atomic save every iteration.
- Machine-readable authority:
  `games/kingdomino/configs/chance_progressive_cloud_v1.json`.

The `70` entries above the cap remain in the schedule so the implementation
and manifest preserve the reviewed ladder. The cap is terminal, so neither
deck can widen past 16.

## 2. Rent only after source and artifact upload are ready

Target an RTX 5090 with at least 16 logical CPU cores, driver R570 or newer,
and a CUDA 12.8 / `cu128`-capable image. Keep the provider spending limit and
maximum rental duration enabled. Use G4 throughput to estimate the two Phase-B
stages; on the 2026-08-12 RTX 5090 run, G4 measured 0.805 games/s.

Push the approved feature branch, record its full 40-character commit, and
transfer the Git-ignored baseline without renaming it. Example from the local
machine (replace host and remote checkout path):

```bash
scp runs/kingdomino/best_checkpoint/current_best.pt \
  root@<host>:/workspace/boardgame-ai/runs/kingdomino/best_checkpoint/current_best.pt
```

Do not copy a replay buffer. The Phase-B prefill deliberately starts fresh.

## 3. Setup and calibration

From the repository root on the cloud box:

```bash
export EXPECTED_COMMIT=<approved-40-character-sha>
export REPO_REF=codex/kingdomino-chance-correct
bash setup_chance_progressive_cloud.sh
```

Setup checks out the exact detached commit, installs the CUDA/Rust stack,
builds the release extension, runs the focused suites, verifies the checkpoint
hash, runs the existing bootstrap calibration, and prints launch commands. It
does not start training.

Inspect the calibration output under
`runs/kingdomino/chance_progressive_cloud_v1/calibration` and choose the best
measured `batch_slots`; do not inherit the dry-run placeholder. `game_cpus` is
frozen at 2: game driving stays light while all remaining logical CPUs are
assigned to the Rayon/async exact-solver pool.

```bash
export BATCH_SLOTS=<calibrated-value>
export RUN_ROOT=runs/kingdomino/chance_progressive_cloud_v1
```

## 4. G4 feasibility probe

G4 is a real 32-game, one-iteration progressive run with one optimizer step.
It exercises both treated deck boundaries, admission, widening, the exact
solver, replay insertion, training, checkpointing, and atomic buffer save.

```bash
bash run_chance_progressive_cloud.sh g4
tail -f "$RUN_ROOT/g4/cloud_process.log"
```

When the process exits:

```bash
python -m games.kingdomino.chance_progressive_cloud validate \
  --phase g4 --run-root "$RUN_ROOT"
```

Proceed at `>=0.30 games/s`. At `0.20-0.30`, proceeding is allowed but the
longer runtime must be accepted explicitly. Below `0.20`, destroy the instance
and reassess. The validator also fails if either deck has no treated examples,
or if search, admission, widening, or exact solving did not actually execute.

## 5. Phase A: recorded stop; no rerun

The original Phase A trained immediately once its 5,000-example minimum was
crossed. On 2026-08-12 it stopped at iteration 5: pair score 42.2%, Wilson
interval `[35.4%,49.3%]`. This demonstrates regression for the young-buffer,
300-step recipe. It does not test a prefilled, lower-reuse recipe. The owner
explicitly chose not to rerun Phase A; its artifacts remain immutable, and
neither its regressed checkpoint nor its partial open-loop buffer feeds Phase B.

```bash
python -m games.kingdomino.chance_progressive_cloud validate \
  --phase phase_a --run-root "$RUN_ROOT"
```

The expected status is `stopped_by_measurement_ucb`. This is a valid recorded
stop, not a Phase-B prerequisite.

## 6. Phase B1: frozen-best chance-aware prefill

The prefill runs 23 iterations × 400 games with the original `current_best`
controlling both seats, progressive search enabled at every eligible deck-8
and deck-12 position, and `train_steps=0`. It starts a fresh buffer and cannot
alter the network. At the observed ~8,700 examples per iteration it should
fill or nearly fill the 200,000-example capacity.

```bash
bash run_chance_progressive_cloud.sh phase_b_prefill
tail -f "$RUN_ROOT/phase_b_prefill/cloud_process.log"
```

Validate before training:

```bash
python -m games.kingdomino.chance_progressive_cloud validate \
  --phase phase_b_prefill --run-root "$RUN_ROOT"
```

The validator requires all 23 rows, zero training steps, frozen-current-best
generation, active progressive mechanisms at both decks, the 25% guardrail,
and at least 175,000 saved examples. Phase B refuses to launch without these
artifacts even if the operator skips validation.

## 7. Phase B2: chance-aware training

Training resets the learner to the original pinned `current_best`, loads the
completed chance-aware prefill buffer, and runs 24 iterations at 192 optimizer
steps per 400 games. With the observed 8,600–8,800 new positions per iteration,
that is approximately 5.6–5.7× replay reuse rather than the stopped recipe's
8.8×. The learner then generates symmetric chance-aware self-play, subject to
the run-local soft gate. The regressed Phase-A checkpoint is never loaded.

Before launch, the runner copies the original baseline to
`phase_b/run_local_best/current_best.pt`. Soft-gate promotions may update only
that run-local copy; the repository baseline remains unchanged.

```bash
bash run_chance_progressive_cloud.sh phase_b
tail -f "$RUN_ROOT/phase_b/cloud_process.log"
```

Every fifth **training** iteration, the learner is compared with the run-local
best over 384 shared-open-loop games at 400 sims. Promotion requires raw score
`>=55%`, seat-pair Wilson LCB `>50%`, and fixed-suite tolerance `0.05`. Raw
score below 48% reverts the generator. Two consecutive reverts stop cleanly;
the circuit breaker does not reset learner weights.

```bash
python -m games.kingdomino.chance_progressive_cloud validate \
  --phase phase_b --run-root "$RUN_ROOT"
```

The final strength gates use the separately frozen seeds `20330000`,
`20340000`, and `20350000`. Do not run or reinterpret those until the Phase-B
artifacts have been synced home and the training trajectory reviewed.

## 8. Symmetric progressive checkpoint evaluation

`games.kingdomino.symmetric_checkpoint_eval` is the explicit opt-in evaluator
for comparing two networks under the deployed chance-aware search. It preserves
the incumbent evaluator's common seeds and seat swaps, but applies the same
progressive configuration to both networks in both orientations. It does not
change the shared-open-loop promotion gates.

The hybrid search is deliberately deck-dependent:

- deck counts above 12 use ordinary open-loop search;
- deck counts exactly 12 and 8 use the frozen progressive treatment;
- the existing exact endgame routing remains in force at its tail boundary.

The progressive initialization rows count against each search's `--sims`
budget. Thus a 400-sim progressive match and a 400-sim open-loop match have the
same NN-row allowance, although progressive search intentionally spends some of
its allowance resolving chance outcomes and can consequently be shallower.

After the current open-loop evaluation has exited and the cloud checkout has
been updated to the reviewed implementation commit, run the progressive match
over the same 1,024 paired seeds:

```bash
export CANDIDATE="$RUN_ROOT/phase_b/iter_0020.pt"
export BASELINE="$RUN_ROOT/phase_b/run_local_best/current_best.pt"
export PROGRESSIVE_EVAL="$RUN_ROOT/eval_iter20_vs_baseline_progressive_2048"

test -f "$CANDIDATE"
test -f "$BASELINE"
test ! -e "$PROGRESSIVE_EVAL"

mkdir -p "$PROGRESSIVE_EVAL"
nohup python3 -m games.kingdomino.symmetric_checkpoint_eval \
  --candidate "$CANDIDATE" \
  --baseline "$BASELINE" \
  --output-dir "$PROGRESSIVE_EVAL" \
  --games 2048 \
  --sims 400 \
  --device cuda \
  --batch-slots 48 \
  --leaf-batch 6 \
  --seed 20330000 \
  --c-puct 1.5 \
  --fpu=-0.2 \
  --margin-gain 2.0 \
  --alpha 0.5 \
  --search-mode chance_progressive \
  --chance-progressive-decks 8,12 \
  --chance-width-schedule 4,8,16,32,64,70 \
  --chance-n-init 2 \
  --chance-d-min 4 \
  --chance-deck8-cap 16 \
  --chance-deck12-cap 16 \
  --chance-max-init-fraction 0.25 \
  > "$PROGRESSIVE_EVAL/process.log" 2>&1 &
echo $! | tee "$PROGRESSIVE_EVAL/process.pid"
```

The command writes a manifest before loading the networks and atomically writes
`games.jsonl` and `result.json` on completion. A progressive result is marked
invalid and the process exits nonzero if either requested deck was not searched
and crossed, or if no admissions, bootstrap rows, or width samples were
recorded. Inspect it with:

```bash
tail -f "$PROGRESSIVE_EVAL/process.log"
cat "$PROGRESSIVE_EVAL/result.json"
```

For a matched open-loop control, reuse the same command and seeds with a fresh
output directory and `--search-mode open_loop`. The chance flags are then
ignored and the mechanism counters are expected to be zero.

## 9. Safe stop, restart policy, and data recovery

To stop without losing the replay buffer, create the phase-local `STOP` file:

```bash
touch "$RUN_ROOT/phase_b_prefill/STOP"   # or phase_b/STOP
```

The loop consumes it at a boundary, saves the buffer, and exits. Avoid `kill
-9`; if provider shutdown is unavoidable, the last successful iteration still
has an atomic buffer autosave and checkpoint. A normal `Ctrl+C` also reaches
the final buffer save, but `STOP` has more predictable latency around Rust
search and gate matches.

The frozen launcher refuses to append to a directory that already contains a
training log or iteration checkpoint. Resume/recovery must therefore be an
explicitly reviewed continuation with a new directory, warm checkpoint, and
buffer; never delete or overwrite the audit trail to make the original launch
command run again.

Continuously sync at least these artifacts off the rental:

- `cloud_launch_manifest.json`
- `run_manifest.json` and `training_log.jsonl`
- `iter_*.pt` and `buffer_final.pkl`
- `measurement_iter_*.json` and promotion logs
- `run_local_best/`, exact fallback logs, and `cloud_process.log`

Before destroying the box, hash the synced files and compare counts and sizes
at both ends.
