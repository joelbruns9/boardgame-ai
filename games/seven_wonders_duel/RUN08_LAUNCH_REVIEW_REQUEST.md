# Review request: run08 launch wiring (fixed league cycle, latest mode, train steps, G2b census)

Branch `sevenwd-w9-prototype` (worktree `boardgame-ai-7wd`), commit `08d465a` plus
the wording fix committed with this file, on top of `525bdb1` (the fixes for the
review of `50e02c7`; response at the end of `GROWTH_2B_G10A_G12_REVIEW_REQUEST.md`).
The run's decisions are in `MODEL_GROWTH_PLAN.md`, **"Final run preparation"**
(owner decisions 2026-10-06). This brief is the entry point.

**Why this matters now.** This is the configuration of the final cloud run. The
previous reviews covered what self-play and training DO. This one covers what the
run is TOLD to do. run07 lost hours to five launcher defects found only on the
box. The laptop dry run below already caught a sixth, before renting anything.

## What run08 is

`launch_7wd_run.sh` is now run08's settings file. run07's version is in git
history. Against run07:

| | run07 | run08 | why |
|---|---|---|---|
| start | candidate_0085 + cloud2 warm buffer | `final_41_100_g10a` (encoder-8 pretrain), **empty buffer** | pretrain absorbed run07 41-100 |
| generator | `soft_gate`, gate every 5 at 64 sims | **`latest`**, `PROMOTION_EVERY=0`, no learner resets | gate dropped; self-anchor is the stopping rule |
| search | 100 cheap / 1,600 full, 25% full | **1,200 on every move** (`FULL_SEARCH_FRACTION=1.0`) | decision 11 + budget |
| HOF | 15% from 10k games | **off** | no promotions means nothing would fill it |
| specialists | random class draw from 10k games | **fixed cycle `S M S M S` from iteration 0** | see below |
| G12 restarts | — | **0.25** | built in `50e02c7` |
| W4 outlook target | ramped in over 10k games | **no ramp** | W4 already trained by the pretrain |
| train steps | 190 / 1,000 games | **550** | measured: ~51 policy rows/game, not ~15.5 |

Exact tactics (G4 incl. 2b) and G2b tactic labels are phase_d defaults (on) and
are not repeated in the settings file.

## Scope (`08d465a`)

| item | files |
|---|---|
| **Fixed league cycle**: `specialist.cycle_opponent_class`, `--league-schedule random|cycle`. Under `cycle` a scheduled specialist with no checkpoint is announced, and with the HOF share at 0 it is never replaced by an old HOF archive | `specialist.py`, `phase_d.py` (`PhaseDConfig.league_schedule`, validation, schedule identity, `league_assignment`, CLI), `test_specialist_league.py`, `training_parameters.md` |
| **Launcher knobs**: `GENERATOR_MODE`, `LEAGUE_SCHEDULE`, `RESTART_FRACTION`. The launcher refuses `latest` with a scheduled gate or with soft-gate learner resets | `setup_cloud_7wd.sh`, `test_setup_cloud.py` |
| **run08 settings** | `launch_7wd_run.sh` |
| **G2b phase-out census**: offline, derives a buffer with tactic labels off and on, and reports labelled / policy-changed / value-changed shares | `g2b_census.py`, `test_tactic_labels.py` |

## Already gated

- **Cycle.** The order for science 0.15 / military 0.10 is pinned (`S M S M S`
  repeating). Every 40-iteration window holds all three shares (HOF included)
  to within two. A zero share is never scheduled. A loop-level test plays
  S M S M S from iteration 0 with 125 of 500 games per iteration. The unready
  case returns None and prints. Numerically, with HOF at 0 / 0.07 / 0.15, the
  largest PREFIX deviation over 300 iterations is 0.4 / 0.625 / 0.625, under
  one. Resume identity is unchanged for `random`, so pre-existing runs still
  resume.
- **Laptop dry run** (run08's exact flags, shrunk: 4 iterations x 48 games, 64
  sims, tiny anchors). Script: `runs/seven_wonders_duel/prep/dryrun_run08.ps1`;
  log: `dryrun_run08/training_log.jsonl`.
  - The first attempt died at startup: the controller refuses
    `revert_reset_after` / `probation_reset_after` > 0 outside `soft_gate`.
    Fixed in the settings file (both 0) and in the launcher (refused up front).
  - The second attempt ran clean:
    - The generator was `latest` every iteration. Iteration 0 was the bootstrap
      promote, then `not_scheduled`, so no gate ever ran.
    - Specialists were seeded at iteration 0 and played science, military,
      science, military (12 of 48 games each). Each trained on its own league
      seat (~400 rows per turn).
    - Restarts were 12/48 from iteration 1, disjoint from league games:
      `kinds` = league 12 + self_play 36.
    - W5 alpha carried over from the pretrain (0.751) and refits each iteration.
    - The self-anchor fired on cadence.
- **Train steps.** General policy inflow was 3,292 rows in iteration 0, then
  ~2,450 per iteration once restarts began (restart games start mid-game). So
  ~51 rows per game, against run07's ~15.5 (15,079-16,256 per 1,000 games,
  iterations 20-29). 550 x 512 / ~51k gives ~5.5 samples per new row, against
  ~6.3 in run07.
- **G2b census baseline.** On run07 `iter_0100` (first 300 games): labelled
  5.76%, policy target changed in 1.26% of policy rows (mean TV 0.228 when
  changed), value changed in 0.60%. The test checks that bot games with
  uniform search targets register changes, and that the same games with
  G2b-clean targets register none.
- **Tests.** `test_setup_cloud.py`, `test_training_parameters_doc.py`,
  `test_league_generation.py`, `test_specialist_league.py`,
  `test_tactic_labels.py`: all pass. The full suite was NOT rerun for this
  commit (last full run: `50e02c7`, 1938 passed, only the known
  `test_async_solver` load flake).

## Focus areas

1. **`latest` mode never moves `current_best`.** It stays the starting network
   for the whole run. I traced its readers:
   - training and S2b reanalysis read the learner (`source_checkpoint` =
     `latest.pt` in the controller path);
   - the self-anchor measures the learner (`anchor_subject`);
   - specialist seeding reads `current_best` once, at iteration 0 (intended:
     the pretrain);
   - the frozen general anchor is the start (intended);
   - `promotion_gate`'s default opponent is never called with
     `PROMOTION_EVERY=0`.

   Did I miss a reader? Is there a path under `latest` that copies
   `current_best` over `latest.pt` that the controller's validation does not
   cover?
2. **The cycle is indexed by ITERATION, not by league ordinal.** With
   `hof_start_games=0` these coincide. With a later start the phase is offset,
   and I claim the share bound still holds because every window is within two.
   Is iteration the right index on a resume, and after a resume that changes
   `hof_start_games` under `--allow-hof-change`?
3. **Unready specialist with the HOF off returns None,** so those games become
   plain self-play for that iteration, announced. The alternative is to refuse.
   Under run08 both specialists seed at iteration 0, so this should never fire.
   Is "announce and continue" right for a final run?
4. **Train steps from a 64-sim dry run.** Rows per game should not depend on the
   simulation budget: every move is searched and recorded regardless. Restart
   share and game length could. Is ~51 rows/game a sound basis at 1,200 sims,
   and is ~5.5 samples per new row the right target for an empty-buffer start?
   With `MIN_BUFFER_POSITIONS=100000`, training first runs at iteration 1 on
   roughly two iterations of data.
5. **Specialist policy inflow.** `training_performance.policy_inflow` reports
   `specialist:N: 0` every iteration, while each specialist's own row reports
   ~400 banked rows and trains. I read the former as the general-route census
   (the specialist seat is not general inflow) and the latter as the
   specialist's real input, consistent with run07's logs. Confirm, or point at
   the misreading.
6. **Settings-file drift.** Exact tactics and G2b ride on phase_d defaults
   rather than being pinned in the settings file. A default flipped before
   launch would silently change the run. Should they be pinned?

## Known limitations (not asking you to find these)

- **Generation cost is estimated, not measured:** ~2.3x run07's sims per move,
  so ~2 h per 1,000-game iteration on run07's box. The box sweep measures it.
- **No live fixed anchor.** It is scored offline from `learner_NNNN.pt` against
  the start (owner decision). Learner snapshots are written every iteration.
- **The G2b census is offline,** run on downloaded buffers. It is not logged
  per iteration.
- **`build_my_symbol_lost` (G10a)** is identically zero by construction. Left
  in to keep the encoder signature.

## How to run the gates

```
$env:PYTHONPATH="."
..\boardgame-ai\.venv\Scripts\python.exe -m pytest -q games/seven_wonders_duel/test_specialist_league.py games/seven_wonders_duel/test_league_generation.py games/seven_wonders_duel/test_setup_cloud.py games/seven_wonders_duel/test_training_parameters_doc.py games/seven_wonders_duel/test_tactic_labels.py
..\boardgame-ai\.venv\Scripts\python.exe -m games.seven_wonders_duel.g2b_census ..\boardgame-ai\runs\seven_wonders_duel\run07_bundle\buffers\iter_0100.jsonl --max-games 300
```

The dry run is `runs/seven_wonders_duel/prep/dryrun_run08.ps1`: about 25
minutes on a laptop 3070, and it needs `prep/final_41_100_g10a/pretrained.pt`.

## Sign-offs requested

1. `latest` + `PROMOTION_EVERY=0` + zero resets is a gate-free run with no path
   back to the starting weights.
2. The cycle's shares and resume behaviour.
3. `TRAIN_STEPS=550` for the measured inflow.
4. Whether to pin the default-on features (focus 6) before launch.

## Response to the review of ebc70c0 (2026-10-07)

Review: `reviews/sevenwd-run08-ebc70c0-review.md`. All three findings are
valid and fixed.

| # | finding | verdict | fix |
|---|---|---|---|
| 1 | `league_schedule` omitted from the identity for `random`, so cycle -> random resumed silently, and a legacy manifest accepted cycle | **Valid** | Always in `schedule_identity()`. A missing historical value is read as `random` in `_refuse_changed_schedules`. Tests for cycle -> random (refused), random -> cycle (refused), legacy -> random (accepted), legacy -> cycle (refused), cycle -> cycle (accepted). |
| 2 | Census counts the general route only; specialist-owned corrections were invisible | **Valid** | Each record is projected onto every trained route (`project_examples`), with per-route denominators and changes plus `policy_changed_any_route`. On the dry-run buffers it now reports the reviewer's specialist corrections exactly: 1 / 4 / 8 / 0. Regression: bot games owned by `specialist:1` give 0 general policy rows and >0 specialist changes. |
| 3 | An expectimax +-1 turning exact is a supervision change the census missed | **Valid** | `value_changed` counts a scalar OR exactness change. `value_exactness_only` and `value_changed_effective` (which drops exactness-only rows already overridden by the certain-win rule) are reported separately. Dry run: 26 exactness-only rows, 13 effective, matching the review. Regression covers both cases. |

**Sign-off items acted on**

- **Pinned.** `EXACT_TACTICS=1` and `TACTIC_LABELS=1` are in the settings file.
  The launcher passes `--exact-tactics/--no-exact-tactics` and
  `--tactic-labels/--no-tactic-labels` explicitly, so emitted config and sweep
  validation see them. The documented cloud command is updated.
- **Train steps are derived, not fixed.** `TRAIN_STEPS = 0.55 x
  GAMES_PER_ITERATION` (550 at 1,000). Warmup is no longer set in the settings
  file, so setup derives it from whatever `TRAIN_STEPS` is.
- **Startup exposure** (~12 presentations for the first cohort vs ~5 at steady
  state): taken to the owner as a decision. It is not a defect.

**Corrections to the brief** (the reviewer is right on all three):

- League and restart games are **not** disjoint: 0 / 0 / 2 / 2 league games
  were restarts. A `kind` partition cannot show disjointness. Nothing requires
  it.
- W5 alpha was **not** refitted at iteration 0. There were 220 held-out policy
  positions, below the minimum of 256. It refitted from iteration 1.
- `current_best` does **not** stay the pretrain all run. `auto_first_trained`
  copies the first trained learner to it once (iteration 1 at run08 scale),
  then it is fixed. Specialist seeding and the frozen general anchor run before
  generation at iteration 0, so they use the pretrain as intended. Offline
  start comparisons should use the uploaded pretrain, not `current_best.pt`.
