# 7WD cloud training: readiness plan

**Status:** **W0/W1/W2 complete. W3 and W5 have implementations under review;
their remaining corrections and production parameter closure are still open.**
Revision 7.
**Revision history:** r7 closes W2 from the successful 70-to-90 recovery,
bounded host-memory plateau, checked allocation/error path, and the target
RTX 5090's 32 GB VRAM margin. Target-device VRAM measurement remains a short
launch preflight, not an open memory-engineering workstream. r6 grounds
production acceptance on the RTX 5090 host,
locks the explicit HOF launch fraction at 0.15, and leaves the gate-cap default
open until that host's W5.7 fit. r4 folded in W0 from
`runs/w0_sizing_v2/report_v2.md` and
propagated its cost/sizing consequences into W2, W5 and W6. r3 (2026-07-28)
rewrote r2 against an external review (10 findings, 9 confirmed against code, 1
partially accepted).
**Date:** 2026-07-30.
**Trigger:** `runs/laptop_training_03` died at iteration 70 of 90 with a host
`MemoryError` after 11 h. The loop itself is healthy -- 9 promotions in 13 gates
-- so the blocker is operational, not algorithmic.

Every number here was measured on `runs/laptop_training_03`, on this laptop
(RTX 3070, 8 GB), or in `runs/w0_sizing_v2`; the appendix records how.

**The one number that changed everything downstream:** the chosen model is
**14.9 M parameters, 14.5x the 1.03 M net every cost figure in revisions 1-3 was
measured against**. Generation costs **1.85x** per game and a training step
**5.12x**. Estimates that were derived at 128x4 are now flagged as such
throughout, and W5's gate-budget arithmetic in particular needs re-deriving
before its cap is set.

**Scoreboard for the cloud run:**

| source | civilian | scientific | military |
|---|---|---|---|
| ZeusAI (converged) | 61.7% | 21.4% | 16.9% |
| BGA humans | 58.0% | 25.6% | 16.4% |
| **run 03, iters 60-69** | **70.9%** | **13.0%** | **16.1%** |

Military is at reference level; the whole gap is science. ZeusAI needed ~100k
games to discover science organically and ran 1,000 sims/move; run 03 played 28k
games at 39 average sims. The gap may be volume, not architecture. This plan
does not close it -- it makes it **measured every iteration**.

---

## Decisions locked

| question | decision | implemented by |
|---|---|---|
| model size | **`d_model=384`, `layers=8`, `heads=6`, 14.9 M params** -- measured, W0 accepted. No mid-run growth -- size is **manual** | W0 **done** |
| precision | **bf16 on every path**, explicit and persisted -- but note the benefit is **width-dependent**, not universal (0.97x on 128x4, 1.69x on 384x8) | W0 **done** |
| LR | **starting point 5e-5**, from W0's replicated fits -- and it sits on the **edge of the searched grid**, so treat it as provisional | W1 **done** (logged per iteration) |
| fallback | **S = 128x4x4 heads / fp32** if cost becomes the primary objective, or if the run underperforms | W7a triggers |
| replay window | **growing window**, `16 * games**0.6`, cap 20k games | W1 **done** |
| curriculum bots | early accelerant only, annealed **in games** (10k) | W1 **done** |
| opponent diversity | **HOF league sampling**, searcher-routed, learner-only labels; compatibility default off, cloud launch explicitly **0.15** | W1 **done**, launch value locked for W2/W5 |
| schedules | every schedule in **games**; `schedule_basis` pinned per run | W1 **done** |
| gate statistic | **pair-level Wilson-LCB three-way rule** (promote / probation / revert), with UCB futility and fixed-N anchors | W5 **implemented** |
| gate budget | configurable; **default remains provisional until the RTX 5090 200/400/800 fit** | W5.7 cloud acceptance |
| gate cost | persistent rolling worker, concurrent seat legs, cloud-measured before selecting the cap | W5 **implemented**, W5.7 pending target host |
| stagnation | **games-indexed** anchor + metric-triggered intervention ladder | W7 |
| cloud target | vast.ai, single box | W6 |

---

## Workstreams

| # | Workstream | Size | Blocks launch? |
|---|---|---|---|
| W0 | Model size + precision, decided by measurement | M | ~~Yes~~ **DONE 2026-07-29** |
| W1 | Training schedules, growing window, HOF league | M | ~~Yes~~ **DONE 2026-07-29** |
| W2 | Memory: fix the crash, bound the footprint, measure it | M | ~~Yes~~ **DONE 2026-07-30** |
| W3 | Shared run-stats contract + reporting | M | **IMPLEMENTED; attribution and replay-overhead corrections pending** |
| W4 | BGA advisor end-to-end with the iter-60 model | M | No -- parallel |
| W5 | Gate efficiency + Wilson-LCB decision rule | M | **LOCAL GATES PASSED; RTX 5090 cap fit pending** |
| W6 | Cloud setup script | M | **Yes** |
| W7 | Stagnation detection + intervention ladder | M | **Yes** (both parts) |

**W1 preceded W2 and W5 validation** and is now done. W2 is closed: the original
host-memory/restart failure is fixed, and the 32 GB RTX 5090 has ample margin
over the L-model paths already exercised on the 8 GB laptop. League play still
adds a second model and changes batch composition, so the cloud launch performs
one exact-geometry VRAM preflight and continues logging physical occupancy. That
measurement is operational telemetry, not a new acceptance gate. W5's gate-cost
fit must still use the launch HOF value rather than the 0.0 compatibility
default.

---

## W0 -- Size and precision: DONE (2026-07-29)

Full record: **`runs/w0_sizing_v2/report_v2.md`**. Commits `6384f0e` (W0.2),
`8fde971` (W0.3), `774a82b` (W0.1), `d279308`..`d669588` (W0 V2 repairs).
Verification: 573 tests pass in `games/seven_wonders_duel/`.

**Outcome:** ship **L = `d_model=384`, `layers=8`, `heads=6`, bf16**.

| arm | params | val total | policy top-1 | vs S in arena | train ms/step | production cost/game |
|---|---:|---:|---:|---:|---:|---:|
| S = 128x4x4 | 1.03 M | **1.9311** | **0.6235** | -- | 67.0 | 1.00x (fp32) |
| M = 256x6x4 | 5.21 M | 1.9443 | 0.6166 | 0.506 | 149.8 | 1.19x (bf16) |
| **L = 384x8x6** | **14.90 M** | 1.9455 | 0.6145 | **0.558** | 343.0 | **1.85x (bf16)** |

Three sub-results matter downstream more than the size choice itself:

1. **Precision benefit is width-dependent.** bf16 is **0.97x** (a slowdown) on
   S's production path, **1.26x** on M, **1.69x** on L. Revisions 1-3 locked
   "bf16 everywhere" on a measurement taken at one width; that generalisation was
   wrong, and it happens to be right for the arm we ship. If the fallback to S is
   ever taken, **fp32 goes with it**.
2. **Search still pays at every width.** L's search-vs-raw score is 0.875 with
   39.2% action disagreement -- the wide policy has *not* absorbed the
   64-simulation improvement gap. **Do not cut the cloud search budget.**
3. **L does not fit an 8 GB GPU comfortably.** Peak allocated 4.95 GiB; the
   physical sampler hit **7,978 of 8,192 MiB** during L training. **16 GB is the
   cloud floor** (W6.4).

**What W0 did not establish, carried forward as accepted risk.** The report's
"Statistical scope" section is the authority; in brief:

- **The width claim is not established at the level of training seeds.** L's
  0.558 clears 0.5 on a game-level Wilson interval ([0.509, 0.606], 792 games)
  but not on a seed-level interval ([0.482, 0.634]) -- checkpoint-seed variance
  exceeded game-level noise, and per-seed scores were 0.549 / 0.502 / 0.623, so
  the headline is concentrated in one of three L checkpoints. Settling it would
  need 6-8 seeds per arm; that work is closed, not pending.
- **The evidence that would have argued for S is confounded.** S is exactly the
  architecture of the corpus generator (`laptop_training_03` iter 60, 128x4),
  whose own val `policy_top1` was 0.6255 against S's 0.6235 -- so the supervised
  metrics measure imitation fidelity, not strength, and cannot rank the arms.
  Removing them as evidence *against* L is what carries the decision, together
  with the expectation that 14.9 M parameters have more headroom under continued
  self-play than 1.03 M. **That expectation is the bet this plan is making**, and
  W7a's anchor is what will falsify it.
- **The shipped configuration has never played a scored game.** All arenas ran
  fp32; bf16 changed 43 of 64 L trajectories in the A/B. Closed by the pre-launch
  check added to W6.2.
- **L's LR of 5e-5 is the bottom edge of the searched grid.** Nothing lower was
  tried, and all three L arena checkpoints inherit it. See W1.5.

**Scope note, so the W0 statistical lesson is not over-applied.** "Fixed-N games
cannot overcome checkpoint-seed variance" applies to claims that *generalise over
training seeds* -- W0's width question. It does **not** weaken W5's gates or
W7a's anchor: those compare two specific named checkpoints, which is exactly the
estimand a game-level Wilson interval addresses. W5.5's pair-level Wilson rule
stands unchanged.

### Superseded reference data

The sizing table below was measured pre-decision, in isolation, and is kept only
because W6.3's sweep and W2's preflight cite it. **Production A/B figures
supersede it for any cost decision** -- the isolated-forward numbers overstate
bf16's benefit (2.41x isolated vs 1.69x in production for L) because they exclude
encoding, transfer, search and scheduling.

| config | params | fp32 rows/s @b256 | bf16 rows/s | advisor 800-sim move |
|---|---|---|---|---|
| 128x4 (was current) | 1.03 M | 14,118 | 32,182 | 0.3 s |
| 256x6 | 5.2 M | 3,645 | 9,835 | 0.3 s |
| **384x8 (chosen)** | 14.9 M | 1,430 | 4,356 | 0.55 s |
| 512x8 | 26.2 M | 824 | 2,781 | 0.93 s |
| 768x12 (ZeusAI) | 86.5 M | 239 | -- | 3.1 s |

**The NN utilisation picture inverted.** Generation demands ~4,960 rows/s (A3).
The 1.03 M net supplied 14,118 fp32 -- ~35% utilisation, GPU to spare. L supplies
**5,873 bf16** rows/s on the isolated b256 bench, i.e. generation now runs at
roughly **85% of NN capacity**. 7WD self-play has moved from CPU/scheduler-bound
to **NN-bound**. Consequences: W5's fixed-overhead fixes still pay in full, but
its *marginal* per-game cost will not shrink; and advisor latency (0.55 s at 800
sims) remains comfortable.

---

## W1 -- Training schedules, growing window, HOF league: DONE (2026-07-29)

**Acceptance met on a real 6-iteration Rust generation run**, not mocks:
curriculum mix annealed 0.150 -> 0.000 exactly at the configured game count;
draft prior 1.000 -> 0.000 alongside it; window targets grew 26 -> 78 sublinearly
with realised windows trailing by whole iterations; league play began exactly at
the `hof_start_games` threshold and took 6 of 24 games (the configured 25%); the
archive played 12 games on each seat; **0 of 839 archive moves kept a policy
target while 463 learner moves did**; and a resume with the ledger deleted
reproduced every schedule value, including with `games_per_iteration` doubled.
707 tests pass.

What shipped, and the three things worth knowing:

1. **The clock is derived, never stored** (`games/az_loop/games_ledger.py`).
   Cumulative games come from counting lines in the per-iteration buffer files,
   which are immutable once written; a cache keyed on file size makes that cheap
   and self-invalidating. There is deliberately no stored schedule position,
   because a stored one drifts the moment `games_per_iteration` changes -- which
   is the exact failure this workstream exists to prevent. Deleting the cache
   changes no number.
2. **`schedule_basis` is part of a run's identity.** `games` is the default;
   `iterations` reproduces pre-2026-07-29 behaviour exactly. Changing the basis,
   or any schedule constant, is refused on resume by the same mechanism as
   `precision`. A manifest with no recorded basis defaults to `iterations`,
   because a run written before the clock existed really did anneal on iteration
   counts -- defaulting it to `games` would have silently rescaled every schedule
   on the first resume after this shipped. `games_per_iteration` is deliberately
   *absent* from the identity: making it free to change is the point.
3. **League play required a Rust change, and the reason matters.** See below.

### W1.3's correction: the searcher owns the network

The first design for league play was Python-only, reusing
`rust_seat_routed_flat_batch_adapter`. That was wrong twice over, and both errors
are worth recording because they are easy to repeat.

- **The routing key was wrong.** That adapter dispatches on the packed `actors`
  byte, which is the **leaf** actor -- the player to move at the evaluated
  position, which alternates with tree depth. Two-network play must dispatch on
  the **searcher**: when it is seat 0's turn, seat 0's network drives the entire
  search and evaluates *every* leaf, including the seat-1 nodes. Routing on the
  leaf actor instead produces a player that is neither checkpoint, and nothing in
  a run would report it -- the games complete, the records validate, and the
  strength numbers quietly describe a chimera. Kingdomino documents exactly this
  distinction on `row_search_actors` (`kingdomino_rust/src/lib.rs:9135`); 7WD had
  the wrong-shaped primitive and no comment saying so. That adapter is now marked
  as diagnostics-only.
- **Splitting the games would not have helped.** `run_many` packs eval groups
  "in job-index order up to `global_batch_cap`" across *all* active slots, so one
  batch mixes games sitting at different plies with different searchers.
  Partitioning games by opponent does not make a single adapter sufficient.

So the packed payload now carries a per-row `net_ids`, resolved in Rust from the
slot's real-state actor, and `self_play_many_flat_net` takes per-game
`nets_p0`/`nets_p1` exactly as it already took `bots_p0`/`bots_p1`. One scheduler
call still covers the whole iteration -- no split slot pool, no drain-tail cost.

Two implementation traps found by the gates rather than by reasoning:

- The trait's default `evaluate_batch_prepared_routed` initially rejected the
  empty `net_ids` that ordinary self-play sends, which the F4 boundary tests
  caught immediately.
- Routing worked in `run_many` but silently did nothing in
  `run_many_pipelined` -- the path production uses whenever
  `max_inflight_batches > 1` -- because `WorkerRequest` dropped the field at the
  worker boundary. **A league test that only exercised the direct scheduler would
  have passed while production ran unrouted.**

**Checkpoints are unaffected.** `ENCODER_SIGNATURE` is a hash of the feature
schema, and a routing array is neither a token nor a feature, so it does not
move; the F4 checkpoint boundary still accepts every existing net. The model
reads its batch by key (`net.py:262`, `:236`), so an added key is ignored by
construction. W0's L checkpoints remain valid.

### Learner-only policy targets

Following Kingdomino's `play_current_vs_hof_game` ("keep only current-owned
labels"), the archive's moves are excluded from the policy loss -- enforced in
Rust at the record site, where curriculum-bot moves are already excluded the same
way. Network 0 is by definition the learner. The moves still exist and the game
still supplies a value target for both seats; only the policy label is withheld,
because a target produced by an older net trains the learner to imitate it.

### Defaults, and what is still off

`hof_opponent_fraction` defaults to **0.0**, so league play is inert until chosen
-- it changes both the memory profile and the batch composition that W2's and
W5's acceptances measure, so enabling it is a deliberate, recorded decision.
`hof_start_games` defaults to 10,000, matching the curriculum's duration: the
archive starts supplying opponents exactly as the bots stop.

**Unmeasured:** the throughput cost of league play at production width. The
one-call design should avoid the drain-tail penalty entirely, but the second
model adds device memory and its own forward passes, and that has not been
benchmarked. Worth folding into W5.7's measurement rather than asserting now.

## W1 -- original specification (retained for the reasoning)

Revision 2 declared these mandatory and gave them no implementing task; W3 only
*recorded* their values. Current code contradicts all three: `replay_window:
int = 20` is a fixed scalar (`phase_d.py:128`), `curriculum_anneal_iterations`
and `draft_prior_iterations` are iteration-counted (`phase_d.py:137`, `:143`),
and HOF is write-only -- `self.hof.add(...)` at `phase_d.py:1968` is its only
use; nothing ever samples it.

**W1.1 -- Growing replay window.** *(M)*
`window_games(total_games) = min(cap, c * total_games**alpha)` with alpha in
0.5-0.8, fitted to your own staleness-vs-value-accuracy data rather than
imported. Small early (on-policy, fast adaptation), growing late (variance
reduction and the diversity that mattered in Kingdomino). Requirements: the
schedule value and the realised window in every stats row; **resume recomputes
from total games, never from a stored window**; the cap is derived from the
memory budget (W2.3), not chosen independently.

Motivating evidence: run 03's window grew 1,400 → 8,000 games over iterations
0-20 and then froze; value accuracy peaked at iteration 35 and decayed after.
Suggestive, not proof.

**W1.2 -- Every schedule in games.** *(M)*
Curriculum bot mix, draft prior, and window all keyed on cumulative games.
Bots anneal out over the first ~10k games -- measured exhaustion point, where
the net's win rate against them passes ~95% (A6). Run 03 used 400
games/iteration; a cloud run at 500-800 would silently rescale every
iteration-keyed schedule.

**W1.3 -- HOF league sampling.** *(M)*
Sample a configurable fraction of generation opponents from archived HOF
checkpoints instead of always `current_best`. Requirements: sampling policy
(uniform over last N, or recency-weighted) in config; the opponent identity
recorded per game so W3 can split outcomes by opponent; a deterministic,
resume-stable sampler keyed on the run seed.

**W1.4 -- Tests and resume semantics.** *(S)*
Schedule functions unit-tested at boundaries; a resume mid-schedule
reproduces identical values; changing a schedule across resume is refused by
the same mechanism as `lifecycle_config`.

**W1.5 -- Decide the cloud LR schedule at L's width.** *(S, new in r4)*
W0 selected **5e-5** for L, but it did so on a *fixed corpus with 4,000 steps*,
and it is the **lowest LR W0 ever ran for L** -- the tie-band rule walked the
selection to the grid edge and nothing below was tested. The cloud loop is a
different regime: incremental fits per iteration, a growing window, and warm
starts from `current_best`. So 5e-5 is a starting point, not a measured optimum.

Requirements: the LR (and warmup) recorded per iteration in the W3 stats block
alongside grad norm, so a too-high or too-low setting is *visible from the log*
rather than inferred from a stalled gate ladder; the value pinned in the manifest
and refused on resume (W6.5). No new sweep -- W0 is closed. This is a
"make it observable and reversible" task, not a tuning task.

**Acceptance:** a short run shows window growth tracking the formula, bot mix
reaching zero at the configured game count, HOF opponents appearing in
generation at the configured rate, and a resume reproducing all three exactly.

---

## W2 -- Memory: DONE (2026-07-30)

### What happened

1. A Python-level allocation failed inside the gate's inference boundary at
   iteration 70 (`run.log:151`; the bare `MemoryError: ` with its trailing space
   is pyo3's `Display for PyErr`, so the failure was on the Python side of the
   adapter call).
2. The worker is fail-fast: `if request.reply.send(result).is_err() || failed {
   break; }` (`eval.rs:670`). The thread exited.
3. The next `submit_prepared(...).wait()` found the channel disconnected and
   raised `ValueError: global inference worker dropped its response`
   (`eval.rs:549`) -- the exception that ended the run, and a red herring.

### Tasks

- **W2.1** *(S)* Latch the worker's terminal `PyErr` and re-raise it from the
  next `wait()`. A run must never again die reporting a symptom.
- **W2.2** *(S)* Replace `PyByteArray::new` with `PyByteArray::new_with`
  (`eval.rs:426-443`). The former is pyo3's **unchecked** constructor -- it
  wraps a possibly-NULL pointer instead of returning `Err`, which is exactly the
  call that cannot fail cleanly under memory pressure.
- **W2.3** *(M)* **RSS-calibrated** cache bound, not an `nbytes` sum.
  `Example` holds **six** numpy arrays; summing their `nbytes` gives ~13.1 KB
  per example, but the measured cost is **17.8 KB** (A1) -- `nbytes` understates
  by ~26% because it excludes ndarray objects, the dataclass, cache keys, list
  overhead, and allocator fragmentation. Bound on `nbytes x calibration_factor`
  with the factor measured at startup, and hold explicit headroom. Accept the
  old `--example-cache-examples` flag, converting at 17.8 KB; fix the 12.5 KB
  constant at `phase_d.py:1412`.
- **W2.4** *(S)* Per-iteration RSS telemetry (psutil, already a dependency) at
  post-generation, post-training, post-gate, plus peak and cache bytes, into the
  W3 stats block.
- **W2.5** *(S)* **Pre-gate admission check**, not only post-hoc pressure
  detection. The iteration-70 failure happened *inside* the gate, so a
  post-gate sample would have observed it after the crash. Before a gate runs:
  estimate its peak (two models + evaluators + `global_batch_cap` rows of packed
  buffers), and if projected RSS exceeds the budget, evict the cache **first**.
- **W2.6** *(S)* `--memory-budget-gb`; on breach evict to a floor and log a
  `memory_pressure` event. Slower re-replay beats losing a run.
- **W2.7** *(S)* Free the gate's models explicitly at gate exit -- `del` +
  `gc.collect()` + `torch.cuda.empty_cache()`. **W0 raises the stakes here**: a
  gate holds two 14.9 M models, not two 1.03 M ones, and W0 measured L at 4.95 GiB
  peak allocated for a *single* model in training.

**Scope after W0.** The A1 footprint numbers (17.8 KB per `Example`, 122 KB per
`GameRecord`) are properties of the data, so W2.3's calibration is unaffected by
model size. Host RSS and device VRAM remain separate telemetry. A precise
predictive VRAM model is not required for launch: L paths already ran on the
8 GB laptop, while the production RTX 5090 has 32 GB. The cloud host runs one
short L/bf16 preflight with the exact batch, slot, HOF, and two-model gate
geometry, then records actual physical and allocated peaks during training.

**Acceptance result:** checked allocation failures preserve the original
Python/CUDA exception; the cache converges to its calibrated byte ceiling;
gate cleanup and restart rollback are implemented; and a copy of
`laptop_training_03` resumed iteration 70 and completed through iteration 89
with zero stderr. Once saturated, the cache held approximately 4.45 GB and
post-training RSS varied by 1.55% across iterations 86-89, peaking at 6.56 GB.
W2 is complete. Continued RSS/VRAM logging on the cloud host is operational
monitoring, not deferred W2 acceptance.

---

## W3 -- Shared run-stats contract

`games/az_loop/contract.py` types every result's metrics as `dict[str, Any]` and
`run_controller._build_row` copies them verbatim. Three consequences:

- answering "what were the victory types" took four throwaway scripts;
- the Rust generation metrics -- NN rows, batch sizes, forced-row share, wave
  stats -- are **collected and discarded**: `phase_d.py:1251` appends them and
  logs only `len(rust_metrics)` as `rust_chunks`;
- generation decayed **27%** over run 03 (1.03 → 0.81 games/s) and the metric
  that would explain it is the discarded one. On a 24-hour cloud run that is
  ~6 hours of generation, unexplained.

New `games/az_loop/stats.py`: a versioned, typed `IterationStats` filled by the
adapter and validated by the controller. `LOG_SCHEMA_VERSION` 1 → 2, readers
tolerant of v1 so run 03 stays analyzable.

**Core (every game fills):**

| group | fields |
|---|---|
| `generation` | games, moves, moves/game (mean, median, p10, p90), decisions searched, mean sims, seconds, games/s, **NN rows, rows/s, forced-row share, mean batch size**, **opponent mix (HOF vs current_best vs bot)** |
| `outcomes` | **`terminal_reason` histogram**, winner distribution, first-player win rate, mean margin, **split by opponent type** |
| `replay` | window games, examples, new examples, reuse factor, staleness (max, mean, p90), **schedule value and realised window** |
| `training` | per-head train/val losses, accuracies, lr, grad norm, steps, seconds, **replay-derivation seconds**, **precision in effect** |
| `gates` | opponent, **win rate, Wilson LCB/UCB, games**, decision, stop reason |
| `resources` | RSS post-gen/post-train/post-gate, peak RSS, cache bytes, GPU util, **VRAM peak allocated and peak physical, reported separately**, seconds per phase |
| `model` | **`d_model`, `layers`, `heads`, parameter count** -- one row-level record of the configuration every other number in the row was produced by (new in r4) |

The `model` group exists because of a W0 lesson: three of that study's harness
failures came from a checkpoint's width being assumed rather than read, and its
one wasted run rebuilt a 256-wide checkpoint as a 128-wide model. Any row whose
numbers cannot be attributed to a width is a row that cannot be compared. Note
also that PyTorch's allocator-**reserved** counter exceeded physical VRAM under
Windows/WDDM in W0 and is not a capacity number -- log physical occupancy
alongside it, or the field is actively misleading.

`terminal_reason` is the game-agnostic form of victory type: Kingdomino emits
`{"score": n}`, 7WD the civilian/military/scientific histogram.

**Game extension** (`game_specific`, adapter-owned) for 7WD:
- **victory type x game length**, and the **age and move index at which the game
  ended**. Measured: military/science wins land at move ~64-65 vs ~73 civilian
  -- deep Age III conversions, not rushes (A6). Logging the age turns "did a
  rush plan ever emerge" into a query.
- science: sixth-symbol races, tokens taken, pairs completed.
- military: max track position, tokens triggered, gold pillaged.
- draft/wonder: wonders built vs discarded, Age III completion rate.

**W3.4 -- `tools/az_report.py`.** *(M)* One command, any run directory: the
block-aggregated outcome mix with binomial error bars, the gate ladder with
Wilson bounds, the learning curve, throughput, RSS, and the victory mix against
the ZeusAI/BGA reference. Without it the schema is just tidier JSON.

**Acceptance:** a 7WD run and a synthetic second-game fixture both emit valid
schema-v2 rows; `az_report.py` reproduces the planning tables; the 27%
generation decay is diagnosable from the log alone.

---

## W4 -- BGA advisor with the iter-60 model (parallel)

Already built: the shared host (`games/advisor/`), the 7WD adapter with a scrape
branch (`advisor_adapter.py:147`), the determinizer, the endgame solver, the
`gamedatas` → wire extractor (`bga_extract.py`, live-verified, freshness guard),
the raw-dump snippet (`bga_snippet.js`), and a local UI (`web_app.py` +
`web_static/index.html`).

- **W4.1** *(S)* Add the `{"bga": <raw gamedatas>}` branch to `state_from_wire`;
  `wire_from_bga` already emits exactly what the observation branch accepts.
  ~15 lines plus tests against `testdata/bga_*.json`.
- **W4.2** *(M)* Package `bga_snippet.js` as an MV3 extension: grab
  `window.gameui.gamedatas`, POST to the local host, render ranked moves. All
  game knowledge stays server-side.
- **W4.3** *(M)* Play 10+ games with iter-60. Measure advisor latency at
  production sims, `UnsupportedBgaState` rate and triggers, stale-payload
  catches, advisor-vs-played-move agreement. **The model is the weak part and
  that is fine** -- this gate tests plumbing.

**Acceptance:** 10 games with zero silently-wrong positions; every unsupported
state raises rather than guesses; documented latency budget.

---

## W5 -- Gate efficiency, then the decision rule

### What the gate costs

Measured (A5): **1.08 h of the 11.06 h run, 9.8%**, for 1,240 games. Per-game
cost decomposes as **~193 s fixed per gate + ~1.74 s per game**. The marginal
1.74/1.09 = **1.60x** over a self-play game matches the 64/39.1 = **1.64x** sims
ratio -- the marginal gate game costs exactly what the extra search costs.
**All the waste is the fixed 193 s.** A 20-game gate spends 227 s, the
equivalent of 85 self-play games.

| | self-play | gate |
|---|---|---|
| scheduler calls | **1 per 400 games** | **~840 per gate** (per ply, per seat, per wave, per leg) |
| inference worker | one, for the iteration | spawned and joined **every call** |
| games in flight | 48-slot rolling pool, refills | <=48 pairs per wave, split by seat (~24 rows), drains to empty |
| models built | 1 per iteration | 2 per gate, plus ~4 redundant `torch.load` |
| seat legs | n/a | **sequential** |

This is trap #3 from the throughput programme ("fixed chunk submission decays
concurrency to zero as each chunk drains") still live in the gate path.

### Throughput first

- **W5.1** *(M)* One persistent inference worker across all plies of a gate.
- **W5.2** *(S)* Build models/evaluators **once per gate**; remove the redundant
  `torch.load` in `checkpoint_agent_name`.
- **W5.3** *(M)* Run the two seat-legs **concurrently** -- independent, and
  interleaving restores batch width without reintroducing the seat-routing bug
  that once inverted gate results.
- **W5.4** *(S)* Rolling refill across pairs and a gate-specific slot count.

Target 193 s → ~50 s, which is what pays for the larger cap.

### Decision rule -- port Kingdomino's

- **W5.5** *(M)* Port `games/kingdomino/promotion.py`'s Wilson rule and
  `_generator_action_after_promotion_check` (`self_play.py:2908`):
  - `wilson_lower_bound(points, games, z=1.96)`, **draws = 0.5 points**;
  - **promote** if LCB > `promotion_min_lcb` (0.50);
  - **revert** if win rate < `revert_win_rate` (0.48);
  - **probation** otherwise;
  - **observation unit = the seat pair**, outcome in {0, 0.5, 1}. Wilson assumes
    independence, and paired games share a seed; pairing the unit makes the
    bound exact rather than mildly anti-conservative;
  - cap **800**; futility stop when the Wilson **upper** bound falls below 0.50
    (promotion arithmetically unreachable).

Why this and not the Bayesian spec from revision 2: it is already implemented
and tested in this repo, Wilson handles draws natively (the Beta-Binomial
fudged them), and the three-way output maps directly onto `az_loop`'s existing
ACCEPT / CONTINUE / REJECT consumed by `gate_transition`.

Replaying run 03 under this rule agrees on **9 of 13** gates (A7). The three
promote → probation flips (gates 15, 20, 60) are gates SPRT truncated at 74-154
games, where the interval is simply too wide to clear 0.50.

**This settles the statistics of the cap.** Win rate required for LCB > 0.50:
**0.598** at 100 games, **0.569** at 200, 0.549 at 400, **0.535** at 800. At
today's 200-game cap a candidate needs a **+7%** edge to promote; at 800 it needs
**+3.5%**. **It does not settle the cost of the cap -- see below.**

### The cap's cost must be re-derived at L's width (new in r4)

A5's decomposition -- **193 s fixed + 1.74 s per game** -- was measured with the
**1.03 M** net. The shipped net is **14.9 M**, and W0 measured production
generation at **1.85x** per game. The fixed 193 s is scheduler and
worker-lifecycle overhead and should be roughly width-independent; the marginal
1.74 s is search, which is now NN-bound.

**Extrapolated, not measured** -- flagged deliberately, because scaling a
two-point fit by a ratio measured on a different code path is exactly the kind of
reasoning that has failed review here before:

| gate size | at 128x4 (measured) | at 384x8, W5.1-5.4 landed (**estimate**) |
|---|---:|---:|
| 200 games | 541 s | ~690 s |
| 800 games | 1,585 s | ~2,610 s (~44 min) |

If that estimate holds, an 800-game gate every fifth iteration is a materially
larger share of the run than the 9.8% A5 measured, and the trade against the
statistical benefit (+7% edge needed → +3.5%) becomes a real decision rather than
an obvious one.

**W5.7** *(S, new in r4)* **Re-measure the gate cost decomposition at L's width
before setting the cap.** Same method as A5, three or more gate sizes rather than
two so the fit is not a two-point extrapolation. This is the last input to the
cap; do not hard-code 800 until it lands. The cap is a config value regardless, so
this task gates the *default*, not the launch.

**Decided 2026-07-29:** W5.7 runs against **W0's `sweep_L_lr5e-05_seed*.pt`
checkpoints**, not a future cloud checkpoint. Gate cost is driven by tensor shapes
and the sims budget, not by checkpoint quality, so a 4,000-step fixed-corpus fit
is an adequate timing proxy -- and it means W5.7 is unblocked today rather than
waiting on the run it is supposed to configure. The one thing checkpoint quality
does affect is **game length** (W0 measured 68-70 moves/game across all three
arms, so the effect is small); record moves/game with the fit so the assumption is
checkable rather than assumed.

Negative result worth recording: a statistically valid futility stop **never
fires at n <= 200** -- even a 0.422 win rate at 96 games still has UCB ~0.52. It
begins paying at n ~ 800, cutting clear losers around 300-400 games. Futility is
worth having *because* of the larger cap, not instead of it.

- **W5.6** *(S)* **Gates are decisions; anchors are measurements.** Early
  stopping biases the score rate toward whichever boundary stopped it, so W7's
  anchor check runs **fixed-N with no early stopping**. Note run 03 feeds
  truncated matches straight into `self.elo.record()`, biasing the ladder the
  same way.

### Known residual risk: no promotion rollback

Verified: `ACCEPT` sets `replace_best=True` (`training_control.py:195`);
`REJECT` sets `replace_best=False` (`:225`), discarding the *candidate* and
keeping the possibly-bad best; `reset_learner` copies `current_best → latest`
(`run_controller.py:450-458`), propagating it; anchor gates return metrics with
**no lifecycle effect** (`run_controller.py:460`). **A bad promotion is never
undone anywhere in the system.**

The mitigation is the strict promote bound: LCB > 0.50 at z=1.96 is ~97.5%
one-sided confidence, and the asymmetry against the 0.48 point-estimate revert
threshold is deliberate -- promotion is irreversible and must be confident;
reverting only discards a candidate. **Periodic HOF revalidation of
`current_best` against archived checkpoints, with demotion on loss, is the real
fix and is deferred**; it is recorded here so the gap is known rather than
assumed away.

**Rejected during planning, recorded so they are not re-proposed:**

- *Tree reuse across plies* -- deferred pending measurement. The usual 1.5-2x is
  deterministic-game folklore; 7WD resolves a reveal after each move, so only
  the subtree under the *realized* outcome survives and the reuse fraction may
  be 10-20%. Measure visits-inherited / visits-total for one arena game first;
  below ~15% it is not worth the plumbing. If built, inherited visits are
  **deducted from the budget** so "64 sims" keeps its meaning.
- *Control variates* -- paired seeds and seats already capture the variance;
  victory type is an **outcome**, not an exogenous covariate, and adjusting for
  it would bias rather than sharpen.
- *Accumulating evidence across gates* -- the candidate is a different net every
  gate, so the pair never repeats.
- *Harvesting gate games as training data* -- 1,240 games is **4.4%** of the
  28,000 self-play games, argmax-only, no search distributions, off-policy for
  both nets, and `_play_two_net_games` builds no `GameRecord` at all.

---

## W6 -- Cloud setup

`setup_cloud_7wd.sh` (250 lines) **predates the Rust engine**: pure Python, no
`rustup`, no `maturin`, header explicitly says the Rust build is
Kingdomino-only. Every number in this plan assumes `--generation-backend rust
--gate-backend rust`. `setup_cloud.sh` (Kingdomino, 301 lines) already has the
missing stages.

- **W6.1** *(M)* Factor `setup_cloud_common.sh`: clone/update, Python + cu128
  torch ordering, GPU hard-fail gate, rustup >= 1.85, crate build, smoke,
  detached `nohup` launch, idempotent resume. Per-game files supply crate path,
  smoke command, launch command. Third instance of the standardization pattern
  after az_loop and the advisor.
- **W6.2** *(M)* **A smoke that cannot pass vacuously.** As specified in
  revision 2 it would have: `pytest` is **absent from `requirements.txt`**;
  `test_buffer_games_equivalent` calls `pytest.skip` when the corpus is missing
  (`test_rust_engine_equiv.py:1424` -- its docstring says the buffers live "on
  the gate box, not a fresh checkout"); and `runs/` is **gitignored**
  (`.gitignore:37`). Required: add `pytest` to requirements; **commit a small
  fixed corpus** (~50 games) outside `runs/`; run the equivalence suite with
  `-p no:randomly -W error` and **fail on any unexpected skip** (assert the
  collected/skipped counts against an expected manifest).
- **W6.2b** *(S, new in r4)* **Play the shipped configuration before trusting it.**
  W0's arenas all ran fp32, and bf16 changed 43 of 64 L trajectories, so
  **L/bf16 has never played a scored game**. Add a fixed-N **L/bf16 vs L/fp32**
  arena to the cloud smoke -- same checkpoint on both sides, so this is a pure
  precision-fidelity check with a known null (0.500) and no seed-variance problem
  of the kind that limited W0's width claim. A few hundred games on the rented GPU
  costs minutes and converts the largest residual W0 caveat into a measurement.
  If it comes back off 0.500 by more than its interval, ship L/fp32 and accept the
  1.69x cost.
- **W6.3** *(M)* Sweep → launch-flag pipeline (`run_f4_cloud_sweep.sh` →
  `f4_cloud_select.py` → `f4_cloud_finalize.py`) with an **explicit flag
  translation layer**: the bench takes `--slots`, `--global-batch-cap`,
  `--max-inflight-batches`, `--scheduler-workers`
  (`f4_throughput_bench.py:1055-1068`); Phase D takes the `--rust-` prefixed
  forms (`phase_d.py:2208-2211`). Unit-test the mapping. It fails loudly today
  (argparse rejects unknown flags) but silently produces a hand-copy step.
- **W6.4** *(S)* Preflight memory sizing against the **maximum scheduled
  window** from W1.1, not the nominal current one, using W2.3's calibrated
  factor plus headroom. Fail at setup, not at 3 a.m. **Now also a VRAM gate:**
  W0 measured L at 7,978 of 8,192 MiB physical on this laptop, so **refuse to
  launch L on less than 16 GB** and check it in the same preflight. The
  instance-selection consequence for vast.ai is a hard filter, not a preference.
- **W6.5** *(S)* **Pin the commit and refuse incompatible resume.** The manifest
  records `git.commit` (`manifest.py:73`) and resume never compares it;
  `_validate_lifecycle_config` (`run_controller.py:238`) checks only lifecycle
  fields and checkpoint hashes. Record commit + `--precision` + schedule
  identifiers, and refuse a resume whose code or config differs unless
  explicitly overridden.
- **W6.6** *(S)* Run-health heartbeat: one line per iteration (iteration,
  games/s, RSS, best_iter, last gate LCB, anchor score).
- **W6.7** *(S)* `snapshot` helper for manual downloads. Copying mid-iteration
  can produce an inconsistent set -- `pending_iteration.json`, `_recovery/`,
  `run_manifest.json`, and the newest buffer file are written at different
  moments. Wait for an iteration boundary, write the manifest last, report the
  minimal resumable set. Per-iteration buffers are immutable, so incremental
  pulls are ~12 MB/iteration.

**Acceptance:** fresh box → training launched unattended in one command; the
equivalence suite **runs** (zero unexpected skips) on the rented GPU before
training starts; the L/bf16-vs-L/fp32 arena lands within its interval of 0.500;
the VRAM preflight refuses a sub-16 GB box; launch flags come from that box's
measured sweep; re-running resumes; a resume on a different commit is refused.

---

## W7 -- Stagnation

Both halves ship **before** launch. Revision 2 had the intervention ladder
landing mid-run; that does not work -- a running process will not pick it up,
and restarting after a pull changes controller semantics mid-run, which W6.5
will (correctly) then refuse. **W7b ships present-but-disabled**, enabled by
config at launch or at a deliberate restart.

### W7a -- Detection

Run 03's Elo cannot detect stagnation by construction: every candidate played
only its own `current_best`, so the ladder has no fixed reference.

A **promotion-lagged** anchor fails too, and fails exactly when it matters. With
anchor = `best[-K promotions]`: while promotions land, both pointers advance and
the score is meaningful. When promotions stop, **both pointers freeze** -- you
are comparing two frozen networks, the score is a constant (say 0.60), and a
`< 0.55` threshold never fires. It answers "were the last K promotions real?",
and when promotions stop, the numerator and denominator stop with them.

**The anchor is indexed by games.** Anchor = whatever `current_best` was N games
ago (e.g. 20k). While learning, the anchor trails a genuinely weaker net. When
learning stops, the anchor **advances until it catches up to the frozen
`current_best`**, and the score converges to exactly **0.50** -- a net against
itself. Unambiguous, self-calibrating, cannot go stale.

Specification: fixed N games, **no early stopping** (measurement, not decision --
W5.6); report the score with its Wilson interval; trigger on the **interval or a
slope across several measurements**, never a single point; before N games have
elapsed, use the bootstrap net or skip rather than comparing against nothing.

Also in the heartbeat: promotions in the last M games; val `value_acc` trend
(run 03's turned at iteration 35, ~25 iterations before the run died); victory
mix drift against the ZeusAI/BGA reference. The one automatic stop stays
`--revert-reset-after`.

**The anchor is also W0's falsifier (new in r4).** W0 could not establish that
384x8 beats 128x4 at the level of training seeds; the decision rests on the
expectation that 14.9 M parameters have more headroom *under continued self-play*
than 1.03 M -- a claim no fixed-corpus study can test. The games-indexed anchor is
the first instrument in this plan that can. Concretely: run 03 reached 70.9/13.0/
16.1 civilian/scientific/military and 9 promotions in 13 gates at 1.03 M. If the
L run's anchor slope is flat while its cost is 1.85x generation and 5.12x
training-step, **the width bet has failed and S/fp32 is the documented fallback**
(decisions table). Record the comparison explicitly in the heartbeat rather than
leaving it to be reconstructed later -- that reconstruction is what cost four
throwaway scripts in W3's motivating case.

### W7b -- Intervention ladder (present, disabled by default)

States NORMAL → STAGNANT → INTERVENTION(k) → RE-MEASURE, escalating one rung at
a time with a fixed measurement window between rungs so each effect is
attributable:

1. **Raise the search budget** (`full_sims_max`, `full_search_fraction`) -- the
   most principled first response: stagnation usually means the search-improved
   policy is no longer better than the raw policy at that budget.
2. **Jump the replay window** up its schedule.
3. **Raise the HOF opponent fraction.**
4. **LR warm restart or decay.**

**Model growth is not on this ladder.** Size is manual.

**Schedule vs metric:**

| lever | today | plan |
|---|---|---|
| curriculum bot mix | schedule (iterations) | **schedule, in games** (~10k) |
| draft prior anneal | schedule (iterations) | **schedule, in games** |
| replay window | fixed | **schedule (power law in games)**, metric-triggered jump |
| sims budget | fixed ranges | schedule default, **metric-triggered raise** |
| HOF opponent fraction | absent | config, **metric-triggered raise** |
| LR | schedule (warmup) | schedule, **metric-triggered restart** |
| model size | fixed | **manual** |

Principle: schedules for what should change regardless; metrics for what should
change only on evidence. All schedules in **games**.

---

## Sequencing

```
W0 (size + precision) -- DONE 2026-07-29: L = 384x8x6, bf16
  -> W1 (schedules, window, HOF) -- DONE 2026-07-29
    -> W2 (memory) -- DONE 2026-07-30
      -> W3 (stats corrections) -> W5 (gates) -> W6 (cloud) -> W7 -> LAUNCH
                                     ^ W5.7 measures complete gate cost at L's width

W4 (BGA advisor, iter-60) -- parallel throughout
```

W0 first: size and precision change every downstream sizing number. **They did** --
see the flagged extrapolations in W2, W5 and W6. Any figure in this plan measured
before 2026-07-29 was measured at 1.03 M parameters.
W1 before W2/W5: HOF league sampling changes the memory profile and batch
composition their acceptances measure.
W2 before W3: memory telemetry is a stats-schema field.
W3 before W5/W6: an unattended multi-day run is only worth launching if it can
be read afterwards -- the 27% generation decay is already unexplained.
W5 throughput before the cap increase: the fixes pay for it.
W7 in full before launch: W6.5 will refuse a mid-run pull.

---

## Out of scope

- **The science gap.** W3 makes it visible per iteration; closing it is a
  decision for after the cloud run produces data.
- **Model growth mid-run.** Manual.
- **Promotion rollback / HOF revalidation.** Deferred, recorded as a known
  residual risk in W5.
- **Re-opening the width question.** W0 is closed. Establishing 384x8 > 128x4 at
  seed level would take 6-8 training seeds per arm; the decision instead rests on
  headroom under self-play, which W7a's anchor measures for free during the run
  itself. Recorded in W0 as accepted risk, not as pending work.
- **Tuning L's LR below 5e-5.** No new sweep. W1.5 makes the setting observable
  and reversible instead.
- **Multi-box sharded generation.** `phase_e/launch_shards.sh` is the precedent
  if throughput becomes the limit.
- **Kingdomino migration onto the lifecycle controller.** W3 makes it possible
  later, not required now.
- **A games budget.** Replaced by strength-per-GPU-hour against the games-indexed
  anchor.

---

## Appendix -- measurements (2026-07-28 unless noted)

**Read A3, A4 and A5 with the W0 result in hand.** All three were measured with
the **1.03 M** net, which is no longer the shipped configuration. A3's utilisation
conclusion inverts at L's width (~85%, not ~35%); A4's speedup range overstates
what production delivers (1.69x, not 2.3-3.0x); A5's cost fit is the input W5.7
re-measures. A1, A2, A6 and A7 are unaffected -- they concern data footprint, a
crash chain, game outcomes and gate statistics, none of which depend on width.

**A1. Memory footprint.** 100 games from `buffer_final.jsonl` through the real
derivation path, RSS-delta: **17.8 KB per `Example`**, **122 KB per
`GameRecord`**. `Example` holds **six** numpy arrays summing to ~13.1 KB
(features dominate: 49 tokens x 130 float16 ≈ 12.7 KB), so `nbytes`
**understates the real cost by ~26%**. At run 03's settings: 250k-example cache
= 4.45 GB, 8,000-game window = 0.98 GB, ~5.4 GB live since iteration 25 on a
16 GB laptop. `phase_d.py:1412` assumes 12.5 KB -- 1.4x optimistic.

**A2. Crash chain.** `run.log:151` bare `MemoryError: ` (pyo3 `Display for
PyErr`, verified against pyo3 0.28.3 `src/err/mod.rs:663`) → worker breaks at
`eval.rs:670` → `ValueError` at `eval.rs:549`. Unchecked allocation site:
`eval.rs:426-443`.

**A3. Generation NN demand.** 28,233 searched moves x 39.08 avg sims = 1.10 M
search evals/iteration; forced root-chance rows ~55% of all NN rows (throughput
programme), so ~2.45 M rows over 494 s ≈ **4,960 rows/s**, against 14,118 rows/s
capacity at 1M params fp32 → **~35% NN utilisation**.

**A4. Precision fidelity and speed.** 512 real positions through
`current_best.pt` (iter 60):

| | mean policy KL | max dP | argmax agreement | mean dP(win) | max dP(win) |
|---|---|---|---|---|---|
| TF32 | 7.9e-08 | 6.5e-04 | **100.00%** | 9.0e-05 | 4.2e-04 |
| bf16 | 8.3e-06 | 5.3e-03 | 99.61% | 1.1e-03 | 5.3e-03 |

The 2 bf16 flips had top1-top2 prior gaps of 0.0016 (overall mean 0.426).
Speedups @b256: TF32 **1.6-2.0x**, bf16 **2.3-3.0x**. Path audit in W0.3:
generation/gates/validation/advisor are fp32 today; **training is already AMP
fp16** (`train.py:488`, no dtype → fp16 on CUDA).

**A5. Gate cost.** Per-iteration wall clock from buffer mtimes minus generation
seconds. Non-gate baseline **57 s** (training ~10 s + replay derivation ~45 s).
Gates: **1.08 h / 11.06 h = 9.8%** for 1,240 games = **3.14 s/game** vs
**1.09 s/game** self-play (**2.9x**). Two-point fit: **193 s fixed + 1.74 s per
game**; 1.74/1.09 = 1.60x matches the 64/39.1 = 1.64x sims ratio.

**A6. Victory mix, game length, bots.** Self-play only:

| iterations | n | civilian | military | scientific |
|---|---|---|---|---|
| 00-09 | 3,475 | 75.0% | 17.1% | 7.9% |
| 30-39 | 3,857 | 69.4% | 16.9% | 13.7% |
| 60-69 | 4,000 | 70.9% | 16.1% | 13.0% |

2 sd at n=4,000 is ±1.1%. Gate games (64 sims, argmax, n=1,240): 69.0/15.7/15.2%
-- non-civilian endings are as common in argmax play as in self-play. Length,
iters 60-69: military 64.0 mean moves, scientific 65.1, civilian 73.0 (p10
55/56/71) -- Age III conversions, not rushes. Net vs curriculum bots: 58.7% (mil)
/ 78.0% (sci) at iters 00-09 → 98.6% / 91.4% by 30-39; **exhausted as opponents
by ~iteration 20** (~10k games), which sets W1.2's anneal.

**A7. Gate rule replay (Kingdomino Wilson LCB, z=1.96, min_lcb 0.50, revert
0.48).** Agreement with run 03's SPRT decisions: **9/13**.

| gate | n | win rate | LCB | KD rule | run 03 |
|---|---|---|---|---|---|
| 5 | 20 | 0.900 | 0.699 | promote | promote |
| 15 | 110 | 0.573 | 0.479 | probation | promote |
| 20 | 74 | 0.608 | 0.494 | probation | promote |
| 25 | 200 | 0.530 | 0.461 | probation | probation |
| 35 | 56 | 0.643 | 0.512 | promote | promote |
| 50 | 200 | 0.465 | 0.397 | revert | probation |
| 60 | 154 | 0.552 | 0.473 | probation | promote |
| 65 | 96 | 0.422 | 0.328 | revert | revert |

Win rate needed for LCB > 0.50: **0.598** @100, **0.569** @200, 0.549 @400,
**0.535** @800. A Wilson-UCB futility stop **never fires at n <= 200** (a 0.422
win rate at 96 games still has UCB ~0.52); it begins paying near n = 800.

**Caveat on A7 and the superseded revision-2 replay:** both treated seat-paired
games as independent. W5.5 makes the *pair* the observation unit, which is
exact; the absolute bounds above are therefore mildly anti-conservative and
should be recomputed at implementation. Relative comparisons between rules are
unaffected, since the same approximation applies on both sides.

**A8. W0 size and precision (2026-07-29).** Full record in
`runs/w0_sizing_v2/report_v2.md`; that document is the authority and this is a
pointer. Corpus: 12,000 games from `iter_0041`-`iter_0070`, 210,779 examples,
game-honest split, cache SHA `149dd2ec…`, pipeline audited on 4,096 real rows.
Three arms x three seeds; LR chosen by a bracketed sweep then replicated;
792-game fixed-N arenas over all 3x3 checkpoint-seed cells; an untouched
2,000-game holdout from iterations 36-40 consulted only after LR selection;
573 tests green.

Headline figures are in the W0 section above. The two results most likely to be
misread if taken second-hand:

- **The arena's 0.558 for L vs S is a game-level result, not a width result.**
  Game-level Wilson [0.509, 0.606]; seed-level [0.482, 0.634]. Checkpoint-pair
  variance (0.00378) exceeded within-cell game noise (0.00244), and per-L-seed
  scores were 0.549 / 0.502 / 0.623. The design spent 792 games on the smallest of
  three variance components. **The generalisable lesson for this repo: when the
  claim is about a design choice rather than about two named checkpoints, seeds buy
  power and games do not.**
- **Supervised loss could not arbitrate.** S shares the corpus generator's exact
  architecture (128x4) and converges on the teacher's own `policy_top1` (0.6235 vs
  0.6255), so the metric measured imitation fidelity. Any future fixed-corpus
  sizing study on self-play data has this confound built in and needs a design
  that does not rank arms by fit to a teacher of one arm's shape.
