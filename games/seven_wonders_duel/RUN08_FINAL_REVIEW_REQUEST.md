# Review request: run08 final pre-launch changes (`19da15a..fee623e`)

Branch `sevenwd-w9-prototype` (worktree `boardgame-ai-7wd`). The previous review
(`reviews/sevenwd-run08-ebc70c0-review.md`) and its fixes (`19da15a`) are
recorded at the end of `RUN08_LAUNCH_REVIEW_REQUEST.md`. This is the last
review before the branch is pushed and a box is rented. Two commits:

| commit | what |
|---|---|
| `a36c607` | value target: `VALUE_BOOTSTRAP` 0.5 -> 0.75; `MIN_BUFFER_POSITIONS` 100k -> 350k; a distance-scaled outcome-share schedule (built, tested, **shipped off**); the offline A/B harness |
| `fee623e` | launcher: stage 0 box vetting; detached CUDA-graph check of the real run; run07's scheduler geometry pinned with the sweep skipped; solver timeout 320M -> 1280M nodes; lambda probe recorded |

## 1. Value target (`a36c607`)

**Why.** Every move is now searched (run07: 25% full), so ~51-59 rows share
each game's outcome against run07's ~15.5. At ~5.5 samples per row, each
outcome label is presented ~3x as often. The outcome is 37.5% of the flat
value target at bootstrap 0.5 (`train.value_targets`: outcome
`(1-b)(1-s)`, own search `b(1-s)`, short-term `s`, with s = 0.25).

**Offline A/B** (`value_target_ab.py`; run07 iterations 91-100, warm start
`candidate_0080`, G0-sealed games withheld, 10% game-split holdout = 855 games,
5,000 steps = ~358 presentations per training game, same seed):

| | flat 0.5 | floored (decay 0.97, floor 0.2) | flat 0.75 |
|---|---|---|---|
| held-out outcome log loss (start 0.458) | 0.475 | 0.469 | 0.468 |
| train - held-out gap (start 0.019) | 0.113 | 0.082 | 0.083 |
| proof abs error, held-out | 0.168 | 0.166 | 0.166 |
| held-out policy / W4 loss | 0.936 / 0.931 | 0.935 / 0.924 | 0.935 / 0.924 |

Gap growth by moves-to-end (flat 0.5): 0-9 +0.113, 10-29 +0.116, 30-49 +0.092,
50+ +0.050. Memorisation is worst LATE (late positions fingerprint a game),
which reversed the premise the floored schedule was built on. Floored and
flat 0.75 tie on every measure, so the existing flag ships and the schedule
stays off.

**G0** (`tactical_suite evaluate --split sealed`, 300 per class, network /
64 / 800 sims): tactical classes unchanged. Its realised-result classes
(`ordinary`, `quiet`, `predecessor`, `reveal_trap`) report `abs_error`, which
favours flat 0.5 by ~0.005-0.011. I rescored the saved readings with Brier and
log loss (game-level bootstrap): level, leaning to 0.75 on games 81-100
(network `ordinary` log loss -0.011, CI [-0.022, -0.001]); games 61-80 were
in the starting checkpoint's training.

**Build.** `Example.plies_to_end` (moves left after the row's move, set in
`_with_short_term`, so both derivation backends), collated as `plies_to_end`
(-1 unknown). `value_targets(outcome_share_decay, outcome_share_floor)`: when
decay > 0 the per-row outcome share in the bootstrap blend is
`max(floor, (1 - value_bootstrap) * decay**plies)`, unknown rows take the
floor. `PhaseDConfig` validation, CLI flags, `setup_cloud_7wd.sh`
`OUTCOME_SHARE_DECAY/FLOOR` (default 0 = off).

**Focus:**

1. **Is `abs_error` against a realised result the wrong instrument for a
   target-weight comparison?** I claim it rewards sharpness (its minimiser is
   the median, ±1 here, not the probability), so it structurally favours the
   outcome-heavy arm. If you agree, should `tactical_suite compare` report a
   proper score beside it for those classes?
2. **Is the A/B's holdout scoring honest?** Held-out log loss is computed
   against the realised result on unseen games (a proper rule). The
   "memorisation" reading is the train-minus-held-out gap on a same-size
   sample of training games. Rows were reused ~17x (run08: ~5.5x), so the
   absolute overfit is exaggerated; is the ARM comparison still valid?
3. **The schedule ships off.** Confirm decay 0 reproduces the flat blend
   bit-for-bit (`test_off_is_the_flat_blend_exactly`), and that adding the
   `plies_to_end` key to every batch changes nothing else (W4, restart rows,
   specialist projections, the example cache).
4. **`plies_to_end` counts records' moves,** including cheap (unrecorded)
   moves and bot moves. Restart records: is `len(record.moves)` the remaining
   game or only from the restart point? (Restart rows are outcome-free, so it
   cannot matter for the shipped config; it matters if the schedule is ever
   turned on.)

## 2. Startup exposure (`a36c607`)

`MIN_BUFFER_POSITIONS=350000`: training first runs after ~6 iterations of
rows (~59 all-example rows/game). Owner choice over ramping train steps. The
review of `ebc70c0` modelled ~12 presentations for the first cohort at 100k.
**Focus 5:** what does the first cohort get at 350k, and does anything else
key off the first trained iteration (`auto_first_trained` copies it to
`current_best`; W5 alpha refit; specialists' own min-buffer)?

## 3. Launcher (`fee623e`)

**Stage 0, box vetting** (before the toolchain, defined in-file for the
stage-2 reason): cgroup quota (`cpu.max` v2, cfs v1) vs `nproc` (share
>= 0.75) and >= 16 effective CPUs; one awk spinner per PHYSICAL core
(`lscpu -p=CORE,SOCKET`, fallback `nproc`) vs one alone, slowdown <= 3.0;
mean `/proc/cpuinfo` MHz sampled during the load >= 2000. Laptop check: 8
spinners x2.05 (laptop turbo), 16 hyperthread spinners x3.41 -- why the
physical-core count and the 3.0 threshold.

**Focus 6:** false refusals on a healthy box (container `/proc/cpuinfo`
reporting static or host MHz; `lscpu` missing; cgroup v1/v2 mixes; a box
where `nproc` already reflects the quota). The run07 bad boxes: 3.84-CPU
quota on a 24-CPU listing (x7.3), and 400-800 MHz under load.

**Stage 10, CUDA-graph watcher.** After `launch_detached`, a nohup'd Python
heredoc polls `training_log.jsonl` every 60 s (24 h cap, exits if the run
dies) and writes a verdict for each of the first 3 rows to
`graph_check.log`: NOT REPLAYING on capture failures, zero replays, or eager
share > 5%. `GRAPH_GUARD=stop` SIGTERMs the run.

**Focus 7:** the counters (`graph_replays` etc. in
`generation_performance.rust_boundary`) are per-adapter absolutes -- per
iteration or cumulative across the process? Does a 5% eager allowance hide a
regime run07 would have hit (eager on uncaptured shapes)? Is SIGTERM safe for
Phase D mid-iteration? Does the heredoc-under-nohup-with-`&` survive the
setup shell exiting?

**Geometry pinned, sweep skipped (owner).** `RUST_SLOTS=512`,
`RUST_GLOBAL_BATCH_CAP=2048`, `RUST_MAX_INFLIGHT_BATCHES=2`,
`RUST_SCHEDULER_WORKERS=4`, `SOLVER_THREADS=2` (per shard),
`RUST_INFERENCE_WAIT_MS=0.0`, `GATE_SLOTS=144`, `GATE_GLOBAL_BATCH_CAP=1024`,
`SKIP_SWEEPS=1` -- run07's `measured_env.sh` (5090 + 7945HX). Stage 10 reports
them as not measured on this box. `test_the_run_file_sets_no_scheduler_geometry`
became `test_scheduler_geometry_is_pinned_only_with_the_sweep_skipped`.

**Focus 8:** with the sweep skipped, what else does stage 8b do that the run
now silently loses (the full-sims re-measure; anything stage 10 or the solver
caps read from `sweeps/`)? Is `SOLVER_THREADS` pinned per shard still right if
the rented box has more cores than run07's?

**Solver timeout 4x.** `ENDGAME_SOLVER_MAX_NODES` 320M -> 1280M, attempt bar
unchanged at 40M. Priced from the setup table (cloud2 corpus, 12 threads):
+12 proofs per ~7.8k, nodes/iter 28.5B -> 32.1B, wasted 14.6% -> 4.0%, stall
373 s -> 1,494 s. Stage 6b derives `max_secs` from it (run07: 725 s at 320M).

**Focus 9:** run08 has 8 solver threads in total (4 shards x 2), not the
table's 12. Does a 1,280M solve pin one of a shard's two solver threads for
~25 min, and what queues behind it? The comment says the solver runs
synchronously inside a shard; does a long solve stall that shard's leaf
production, not just the parked game?

## 4. Specialist lambda (recorded in `fee623e`)

`specialist_probe` on the G10a pretrain (run07 iter_0100, 200 positions,
1,200 sims, puct, asymmetric, lambda 3):

| | moved | credible | q_cost | own-type outlook |
|---|---|---|---|---|
| science | 0.14 | 0.57 | 0.056 | 0.098 -> 0.104 (+6.1%) |
| military | 0.08 | 0.69 | 0.045 | 0.066 -> 0.065 (-2%) |

Science reproduces the cloud2 calibration (0.155 / 0.516 / 0.076 / +6.2%).
Military has no single-ply pursuit, as on cloud2 at every lambda. run07's
replay window: military-class games ended military 21% (HOF games 16%),
science-class games ended scientific 25% (HOF 18.6%). Kept at 3 (owner).

## Known limitations (not asking you to find these)

- Iteration time at 1,200 sims is estimated (~2 h / 1,000 games), not
  measured; the first heartbeat measures it.
- `OUTLOOK_BOOTSTRAP` stays 0.5, untested at 0.75 (owner).
- Re-dealt restarts are the parked remedy if run08's late-game gap stays open.

## Gates run

- `test_outcome_share.py` (new, 7), `test_short_term_value.py`,
  `test_g2_value_contract.py`, `test_phase_d.py`, `test_restart_archive.py`,
  `test_hierarchical_value.py`, `test_setup_cloud.py`,
  `test_training_parameters_doc.py`: all pass.
- Stage 0 run on the laptop under Git Bash (harness: functions extracted,
  helpers stubbed). `bash -n` on both scripts.
- **Full 7WD suite** (sequential, 42 min): 1965 passed, 6 skipped, 1 failed
  -- `test_w0_sizing::test_packed_batch_is_value_identical_to_collate`: the
  W0 sizing tool's packed batch did not carry the new `plies_to_end` key.
  Fixed (mirrored in `w0_sizing._empty_storage`/packing); W0 tests pass.
  Sizing tooling only; production training collates through `dataset.collate`.
- **Rust** (`cargo test --release`): 67 passed, 2 ignored, after two
  test-only fixes, neither from this range's code:
  - `encoder_feature_counts_match_schema` panicked: W3 control channels
    default ON and a pure-Rust test has no control table. The test now turns
    them off (counts are unchanged; the channels stay as zeros).
  - A doc comment's formula in `tree_resumable::Strata` (G4 2b) compiled as
    a doctest; now a `text` block.

  Neither changes the built extension's behaviour.

## Sign-offs requested

1. Value bootstrap 0.75 on this evidence, schedule off.
2. Stage 0 thresholds will not refuse a healthy box.
3. The graph watcher is safe to leave running beside the run.
4. Skipping the sweep loses nothing the run reads.
5. The 4x solver timeout at 8 solver threads.
