# Promotion Gating and Smart Elo Plan

## Goal

Prevent weak latest checkpoints from becoming the official best model, while
still allowing the learner to generate frontier self-play data when it is not
clearly worse than `current_best.pt`.

The desired policy is a soft gate:

```text
latest loses clearly:
  revert self-play generator to current_best

latest is roughly equal:
  keep latest as self-play generator, do not promote

latest wins convincingly:
  promote latest to current_best, keep latest as generator, run Elo
```

This keeps `current_best.pt` conservative and trustworthy without freezing the
self-play distribution around one old model.

## Current Behavior

The code already has strict gated self-play:

- `--gated_selfplay` makes self-play use `--current_best_path`.
- `--promotion_every` periodically evaluates learner vs current best.
- Passed promotions can update the in-memory generator.
- `--promotion_update_best` can overwrite `current_best.pt`.

This is safe but may stagnate because the latest learner does not normally
generate its own frontier states.

The code also has standalone Elo and in-training Elo:

- `elo_rating.py` rates checkpoints against the fixed anchor pool.
- `--elo_every` rates periodic checkpoints.
- Routine default is now `games_per_anchor=32`, matching 32 concurrent slots.

## Proposed Training Control Modes

Add an explicit generator-control mode. Soft gate should become the recommended
and eventually default training-control policy; strict gate remains only as a
legacy/debugging mode.

```text
--selfplay_generator_mode latest
--selfplay_generator_mode current_best
--selfplay_generator_mode strict_gate
--selfplay_generator_mode soft_gate
```

Recommended semantics:

- `latest`: current behavior when `--gated_selfplay` is off.
- `current_best`: always generate from `current_best.pt`; no learner promotion
  affects generation.
- `strict_gate`: existing `--gated_selfplay` behavior.
- `soft_gate`: latest learner generates unless a gate decides it is clearly
  worse than current best.

For backward compatibility:

- Omit the new flag -> current behavior.
- `--gated_selfplay` maps to `strict_gate`.
- After smoke and medium validation, make `soft_gate` the documented default
  command pattern for serious training runs.

## Soft Gate Decision Policy

Add these parameters:

```text
--soft_gate_revert_win_rate 0.48
--soft_gate_promote_win_rate 0.55
--soft_gate_promote_min_lcb 0.50
--soft_gate_use_lcb_for_revert
--soft_gate_revert_max_ucb 0.50
```

Initial simple policy:

```text
if win_rate < soft_gate_revert_win_rate:
  action = revert
  generator = current_best
  current_best unchanged

elif win_rate >= soft_gate_promote_win_rate and lcb > soft_gate_promote_min_lcb
     and fixed suite passes:
  action = promote
  generator = latest
  copy old current_best to HOF
  update current_best.pt

else:
  action = probation
  generator = latest
  current_best unchanged
```

Reasoning:

- `< 48%` avoids reverting on small match noise around 50%.
- `48-55%` keeps learning on frontier data without declaring a new official best.
- `>=55%` plus LCB and fixed-suite checks is the true promotion gate.

Future refinement:

- Use Wilson upper confidence bound for reverts: revert only if the model is
  statistically likely to be below 50%.
- Add hysteresis: require two consecutive revert decisions before changing the
  generator back to current best.

## Smart Elo Policy

Elo should not run at every promotion check. Promotion matches are the cheap
control loop; Elo is the more expensive ladder measurement.

Promotion matches should be deterministic evaluator tests, not noisy self-play
tests:

```text
dirichlet_epsilon = 0.0
temp_moves = 0
```

Use lower sims than full Elo because this gate is primarily checking whether
the neural net value/policy has improved enough to control generation. Initial
recommendation:

```text
--promotion_sims 100
```

Choose `--promotion_games` as a multiple of the concurrent slots, same as Elo.
With 32 slots, use `--promotion_games 384` or `448` for production rather than
400. For smoke tests, use `32` or `64`.

Add these parameters:

```text
--smart_elo
--smart_elo_on_promote
--smart_elo_on_run_end
--smart_elo_on_probation_streak 0
--smart_elo_games_per_anchor 32
--smart_elo_sims 400
```

Initial policy:

```text
on promote:
  run Elo for the promoted checkpoint using games_per_anchor=32

on run end:
  if final generator checkpoint was not Elo-rated, optionally rate it

on probation streak:
  optional; if latest remains probationary for N gates, run Elo for diagnosis
```

Default recommendation:

- Enable `--smart_elo_on_promote`.
- Keep `--smart_elo_on_probation_streak 0` initially.
- Set `--elo_every 0` for normal soft-gated training if you only want Elo when
  a new best model is promoted. Periodic Elo and smart Elo are independent:
  `--elo_every 0` disables scheduled Elo, but `--smart_elo_on_promote` still
  runs Elo after a successful promotion.

## Implementation Plan

### Phase 1: Internal State Model

Add a small generator state object in `self_play.py`.

Fields:

```text
generator_mode
generator_source
generator_checkpoint_path
generator_sha256
soft_gate_action
probation_streak
revert_streak
last_promotion_iteration
last_elo_iteration
```

Responsibilities:

- Decide which net generates self-play for each iteration.
- Load `current_best.pt` only when needed.
- Switch to latest learner after probation/promote decisions.
- Switch back to current best after revert decisions.
- On promotion, add the old `current_best.pt` to the HOF pool before overwriting
  it.

Acceptance checks:

- Existing ungated runs behave unchanged.
- Existing `--gated_selfplay` runs behave unchanged.
- Logs include stable `generator_source` and `selfplay_source`.

Implementation status: complete.

Programs changed:

- `games/kingdomino/self_play.py`
  - Added `GENERATOR_MODES`.
  - Added `SelfPlayConfig.selfplay_generator_mode`.
  - Added `GeneratorState`.
  - Added `_effective_generator_mode`.
  - Added `_init_generator_state`.
  - Wired the training loop to read the active generator from
    `GeneratorState`.
  - Preserved legacy `--gated_selfplay` by mapping it to `strict_gate`.
  - Added CLI flag `--selfplay_generator_mode`.
  - Added structured log fields: `generator_mode`, `generator_source`,
    `generator_checkpoint_path`, `generator_sha256`, and `generator_action`.

- `games/kingdomino/test_milestone6_promotion.py`
  - Extended the strict gated smoke test to assert generator state fields.
  - Added a soft-gate Phase 1 smoke test confirming soft gate currently uses
    `learner_latest` and logs generator state without requiring current best.

Test results:

```text
.\.venv\Scripts\python.exe -m py_compile games\kingdomino\self_play.py games\kingdomino\test_milestone6_promotion.py
PASS

.\.venv\Scripts\python.exe -m pytest games\kingdomino\test_milestone6_promotion.py -q
6 passed in 7.08s
```

### Phase 2: Soft Gate Decision

Refactor existing promotion decision handling so it returns one of:

```text
revert
probation
promote
no_check
```

Use existing `evaluate_network_match`, fixed-suite comparison, and promotion
payload helpers where possible.

Add log fields:

```text
promotion_action
promotion_checked
promotion_passed
promotion_win_rate
promotion_lcb
promotion_reasons
soft_gate_revert_threshold
soft_gate_promote_threshold
generator_before
generator_after
```

Acceptance checks:

- Synthetic match stats below 48% produce `revert`.
- Synthetic match stats from 48-55% produce `probation`.
- Synthetic match stats above 55% with LCB > 50% and fixed-suite pass produce
  `promote`.
- Fixed-suite failure blocks `promote` and falls back to `probation` or `revert`
  based on win rate.

Implementation status: complete.

Programs changed:

- `games/kingdomino/self_play.py`
  - Changed promotion defaults to `promotion_games=384` and
    `promotion_sims=100`.
  - Added `soft_gate_revert_win_rate=0.48`.
  - Extended `GeneratorState` with current-best baseline fields:
    `baseline_net`, `baseline_source`, `baseline_checkpoint_path`, and
    `baseline_sha256`.
  - Made `soft_gate` require `current_best_path` and load current best as the
    promotion baseline while still using `learner_latest` as the generator.
  - Added `_generator_action_after_promotion_check`, returning `revert`,
    `probation`, `promote`, or `reject`.
  - Updated the promotion block so `strict_gate` and `soft_gate` both evaluate
    learner vs current best.
  - Implemented soft-gate transitions:
    - `revert`: generator switches to current best.
    - `probation`: generator stays latest learner; current best unchanged.
    - `promote`: current best is overwritten, old current best is copied to HOF
      first, generator stays latest learner, and the baseline reloads from the
      new current best.
  - Added log fields: `promotion_action`, `promotion_revert_win_rate`,
    `generator_baseline_source`, and `generator_baseline_sha256`.
  - Added CLI flag `--soft_gate_revert_win_rate`.

- `games/kingdomino/test_milestone6_promotion.py`
  - Updated the soft-gate smoke test to seed and assert current-best baseline
    fields.
  - Added pure threshold tests for `revert`, `probation`, `promote`, and legacy
    strict-gate `reject`.

Test results:

```text
.\.venv\Scripts\python.exe -m py_compile games\kingdomino\self_play.py games\kingdomino\test_milestone6_promotion.py
PASS

.\.venv\Scripts\python.exe -m pytest games\kingdomino\test_milestone6_promotion.py -q
7 passed in 3.90s
```

Note: after those tests passed, a small safety cleanup was made so soft-gate
promotion does not mutate a baseline-net copy when the generator had previously
reverted. A rerun of the same venv test commands was attempted, but the Codex
escalation reviewer rejected further Python/Torch commands due to the usage
limit. The final cleanup was source-reviewed but not re-executed in the venv.

### Phase 3: Smart Elo Trigger

Add a helper that rates a checkpoint only when a trigger fires.

Use existing in-training Elo path with:

```text
games_per_anchor = smart_elo_games_per_anchor or elo_games_per_anchor
sims = smart_elo_sims or elo_sims
name = <run_id>_iter_XXXX_promoted
```

Avoid duplicate ratings:

- Track names already present in `elo_db.json`.
- Skip if the same checkpoint/name/config was already rated.

Add log fields:

```text
smart_elo_triggered
smart_elo_reason
smart_elo_rating
smart_elo_stderr
smart_elo_n_games
smart_elo_name
```

Acceptance checks:

- Promotion triggers Elo when enabled.
- Promotion-triggered Elo still runs when `--elo_every 0`.
- Probation does not trigger Elo by default.
- Revert does not trigger Elo.
- Elo uses `games_per_anchor=32` unless overridden.
- Existing `--elo_every` still works independently.

Implementation status: complete.

Programs changed:

- `games/kingdomino/self_play.py`
  - Added `SelfPlayConfig.smart_elo`, `smart_elo_on_promote`,
    `smart_elo_games_per_anchor`, and `smart_elo_sims`.
  - Added `_run_smart_elo_rating`, which reuses the existing in-training Elo
    path with smart-Elo-specific games-per-anchor and sim settings.
  - Added duplicate protection for smart Elo by checking `elo_db.json` for an
    existing checkpoint name before launching rating games.
  - Wired successful `promote` actions to trigger smart Elo when both
    `--smart_elo` and `--smart_elo_on_promote` are enabled.
  - Kept scheduled Elo independent: `--elo_every 0` disables periodic Elo, but
    promotion-triggered smart Elo still runs.
  - Updated final ladder resolution so smart-Elo-only runs still resolve the
    global ladder from `elo_games.jsonl`.
  - Added structured log fields: `smart_elo_triggered`, `smart_elo_reason`,
    `smart_elo_name`, `smart_elo_rating`, `smart_elo_stderr`, and
    `smart_elo_n_games`, plus `smart_elo_skipped` for duplicate-rating skips.
  - Added CLI flags: `--smart_elo`, `--smart_elo_on_promote`,
    `--smart_elo_games_per_anchor`, and `--smart_elo_sims`.

- `games/kingdomino/test_milestone6_promotion.py`
  - Added a smart-Elo helper test that monkeypatches the expensive Elo run,
    verifies smart defaults override scheduled Elo defaults, and verifies an
    existing Elo DB entry skips duplicate rating games.

Test results:

```text
.\.venv\Scripts\python.exe -m py_compile games\kingdomino\self_play.py games\kingdomino\test_milestone6_promotion.py
PASS

.\.venv\Scripts\python.exe -m pytest games\kingdomino\test_milestone6_promotion.py -q
8 passed in 4.24s
```

### Phase 4: CLI and Documentation

Add command examples:

Soft gate dry run:

```powershell
.\.venv\Scripts\python.exe -m games.kingdomino.self_play `
  ...training args... `
  --selfplay_generator_mode soft_gate `
  --warm_start_current_best `
  --promotion_every 5 `
  --promotion_games 384 `
  --promotion_sims 100 `
  --smart_elo `
  --smart_elo_on_promote `
  --smart_elo_games_per_anchor 32
```

First smoke test:

```powershell
.\.venv\Scripts\python.exe -m games.kingdomino.self_play `
  ...tiny run args... `
  --selfplay_generator_mode soft_gate `
  --warm_start_current_best `
  --promotion_every 1 `
  --promotion_games 32 `
  --promotion_sims 50 `
  --smart_elo `
  --smart_elo_on_promote `
  --smart_elo_games_per_anchor 32 `
  --benchmark_every 0 `
  --elo_every 0
```

Documentation updates:

- `training_parameters.md`: explain `strict_gate` vs `soft_gate`.
- `README.md`: update canonical training command once smoke passes.
- `kingdomino_project_plan.md`: record the soft-gate rationale and run10 lesson.

Implementation status: complete.

Programs changed:

- `games/kingdomino/self_play.py`
  - Updated `--selfplay_generator_mode` CLI help so `soft_gate` is described as
    implemented behavior rather than upcoming bookkeeping.

- `games/kingdomino/training_parameters.md`
  - Updated the canonical command to use `--warm_start_current_best`,
    `--selfplay_generator_mode soft_gate`, promotion checks, smart Elo on
    promote, and `--elo_every 0`.
  - Documented smart Elo and clarified that scheduled Elo can be disabled while
    promotion-triggered Elo remains active.
  - Replaced the old gated-self-play guidance with generator-mode guidance,
    soft-gate action semantics, and a short CUDA smoke command.
  - Updated recommended-use-case rows so production runs prefer soft gate plus
    smart Elo.

- `games/kingdomino/README.md`
  - Replaced the old laptop command with a soft-gated 48x6 continuation command.

- `games/kingdomino/CLOUD_RUN.md`
  - Added a same-architecture continuation note for using soft gate and smart
    Elo safely on cloud runs.

- `games/kingdomino/kingdomino_project_plan.md`
  - Recorded the run10 lesson and updated Milestone 6/checkpoint strategy to
    prefer soft-gated self-play, HOF backup before overwrite, and smart Elo on
    promotion.

Test results:

```text
.\.venv\Scripts\python.exe -m py_compile games\kingdomino\self_play.py games\kingdomino\test_milestone6_promotion.py
PASS

.\.venv\Scripts\python.exe -m pytest games\kingdomino\test_milestone6_promotion.py -q
8 passed in 3.96s
```

### Phase 5: Tests

Add or extend tests around:

- Decision thresholds.
- Generator state transitions.
- Promotion payload fields.
- Smart Elo trigger conditions.
- Backward compatibility for `--gated_selfplay`.

Suggested test cases:

```text
latest 47% -> revert, no promote, no Elo
latest 52% -> probation, no promote, no Elo
latest 56%, LCB 51%, suite pass -> promote, smart Elo
latest 56%, LCB 49% -> probation, no smart Elo
latest 56%, suite fail -> no promote
promotion match uses dirichlet_epsilon=0.0 and temp_moves=0
promotion games are rounded/rejected to a multiple of batch_slots
promotion backs up old current_best into HOF before overwriting
smart Elo on promote still runs with elo_every=0
--gated_selfplay -> strict behavior unchanged
--elo_every and --smart_elo can both log without duplicate names
```

Implementation status: complete.

Programs changed:

- `games/kingdomino/test_milestone6_promotion.py`
  - Added shared helpers for tiny CPU soft-gate smoke configs and synthetic
    `MatchStats`.
  - Added tests confirming LCB failure and fixed-suite failure stay
    probationary rather than promoting.
  - Added a parameterized training-loop test for soft-gate `revert`,
    `probation`, and `promote` actions using monkeypatched match stats.
  - Verified promotion-match wiring uses `promotion_games` and
    `promotion_sims`.
  - Verified smart Elo triggers only on `promote`, not on `revert` or
    `probation`.
  - Verified promote writes promotion audit metadata and copies the previous
    current best into HOF before overwrite.

Test results:

```text
.\.venv\Scripts\python.exe -m py_compile games\kingdomino\self_play.py games\kingdomino\test_milestone6_promotion.py
PASS

.\.venv\Scripts\python.exe -m pytest games\kingdomino\test_milestone6_promotion.py -q
12 passed in 6.95s
```

### Phase 6: Smoke and Rollout

Run a short CUDA smoke:

```text
iterations=2-3
promotion_every=1
promotion_games=32
promotion_sims=50
elo_every=0
smart_elo_on_promote enabled
```

Verify:

```text
generator_source changes only according to action
promotion_action is logged
smart_elo_triggered appears only on promote
training continues after revert/probation/promote
current_best.pt is overwritten only after promote
old current_best is copied to HOF before overwrite
smart Elo still runs with elo_every=0 when promotion occurs
```

Then run one medium validation:

```text
promotion_games=96
promotion_sims=100
smart_elo_games_per_anchor=32
```

Only after the medium validation, enable production values:

```text
promotion_games=384
promotion_sims=100
smart_elo_games_per_anchor=32
```

Implementation status: short smoke complete; medium validation deferred.

Programs changed:

- No source changes.
- Throwaway smoke artifacts were written only under
  `runs/kingdomino/phase6_smoke/`.

Smoke results:

```text
Seeded throwaway current best:
runs\kingdomino\phase6_smoke\best_checkpoint\current_best.pt

Open-loop smoke:
.\.venv\Scripts\python.exe -m games.kingdomino.self_play
  --engine open_loop --device cpu
  --iterations 2 --games_per_iter 2 --train_steps 0 --sims 2
  --channels 8 --blocks 1 --bilinear_dim 8
  --warm_start_current_best
  --current_best_path runs\kingdomino\phase6_smoke\best_checkpoint\current_best.pt
  --selfplay_generator_mode soft_gate
  --promotion_every 1 --promotion_games 32 --promotion_sims 50
  --promotion_min_lcb 0.99 --promotion_skip_fixed_suite
  --smart_elo --smart_elo_on_promote
  --elo_every 0

Result:
- Completed 2/2 iterations.
- Actual self-play generated 208 examples.
- Promotion checked on both iterations.
- Both checks produced `promotion_action=probation`.
- `smart_elo_triggered=false` on both iterations.
- Buffer and checkpoints were written under the throwaway train dir.

Batched/CUDA-shaped smoke:
.\.venv\Scripts\python.exe -m games.kingdomino.self_play
  --engine batched_open_loop --device cuda
  --iterations 1 --games_per_iter 2 --train_steps 0 --sims 2
  --channels 8 --blocks 1 --bilinear_dim 8
  --warm_start_current_best
  --current_best_path runs\kingdomino\phase6_smoke\best_checkpoint\current_best.pt
  --selfplay_generator_mode soft_gate
  --promotion_every 1 --promotion_games 32 --promotion_sims 20
  --promotion_min_lcb 0.99 --promotion_skip_fixed_suite
  --smart_elo --smart_elo_on_promote
  --elo_every 0

Result:
- Completed 1/1 iteration.
- Actual batched self-play generated 104 examples.
- Batched stats logged: mean_batch=3.0/4, max_batch_seen=4.
- Promotion checked and produced `promotion_action=probation`.
- `smart_elo_triggered=false`.
- Exact-solver stats logged with exact solving disabled.
```

Artifact verification:

```text
runs\kingdomino\phase6_smoke\best_checkpoint\current_best.pt exists
runs\kingdomino\phase6_smoke\best_checkpoint\promotion_log.jsonl absent
runs\kingdomino\phase6_smoke\best_checkpoint\hof absent/empty
runs\kingdomino\phase6_smoke\elo_db.json absent
runs\kingdomino\phase6_smoke\elo_games.jsonl absent
```

The heavier medium validation (`promotion_games=96`, `promotion_sims=100`) was
not run in this pass. The short smoke covered real self-play, real promotion
matches, checkpoint writes, buffer writes, manifest/log creation, soft-gate
probation logging, and isolation from production model/Elo/HOF paths.

## Decisions

1. Should soft gate replace `--gated_selfplay`, or live as a separate
   `--selfplay_generator_mode soft_gate`?

   Decision: soft gate should replace strict gate as the recommended training
   policy because it preserves frontier exploration while still protecting
   `current_best.pt`. Keep strict gate only for backward compatibility and
   debugging.

2. Should a promotion automatically write `current_best.pt`?

   Decision: yes. A passed soft-gate promotion should automatically rewrite
   `current_best.pt`, but it must first copy the old current best into the HOF
   pool with audit metadata.

3. Should probation checkpoints be Elo-rated?

   Decision: no by default. Add `--smart_elo_on_probation_streak N` for
   diagnosis if promotions stall.

4. Should revert use raw win rate or confidence bound?

   Decision: raw win rate is fine. Start with `win_rate < 0.48`; later add
   Wilson upper confidence bound only if reverts are noisy.

## Success Criteria

This work is successful when:

- A run can continue using latest-learner self-play while preventing clear
  regressions from controlling future data.
- `current_best.pt` only changes after a statistically supported promotion, and
  the prior current best is preserved in HOF first.
- Promoted checkpoints are automatically Elo-rated with the 32-game-per-anchor
  routine default, even when `--elo_every 0`.
- The training log makes generator decisions auditable.
- Existing ungated and strict-gated workflows remain compatible.
