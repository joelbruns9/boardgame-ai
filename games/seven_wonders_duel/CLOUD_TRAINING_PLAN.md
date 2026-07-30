# 7WD cloud training: readiness plan

**Status:** **W0/W1/W2/W3/W5/W6 complete. W7 (stagnation) is the last launch
blocker; W5.7's per-rung gate-cost fit runs on the target host at first
launch.**
Revision 11.
**Revision history:** r11 closes W6: a shared `setup_cloud_common.sh` with the
Rust stages the 7WD script never had, an equivalence smoke that cannot pass
vacuously, a committed corpus, the precision arena (which required per-net
autocast in the routed adapter), the sweep-to-launch flag translation, the
memory/VRAM preflight, the resume commit pin, the heartbeat, and the snapshot
helper. r10 rewrites W5's decision rule after review: sequential
stopping is removed (it ran at 15-19% false promotion), revert becomes a Wilson
**UCB < 0.48** test with `revert_reset_after = 2`, and the single cap is replaced
by a scheduled gate-size ladder. r9 closes W3 with explicit current-best/HOF/bot/HOF+bot
provenance, actual-use handling for bot-shadowed HOF assignments, and realized
opponent shares in `az_report`. r8 removes W3's stats-only replay by collecting and
caching game observations during required trainable-position derivation. r7
closes W2 from the successful 70-to-90 recovery,
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
| gate statistic | **fixed-N pair-level Wilson three-way rule**: promote LCB > 0.50, revert UCB < 0.48, else probation. No mid-match stopping. Fixed-N anchors | W5.5 **rewritten in r10** |
| revert cost | **two-stage**: first revert switches the generator only; `revert_reset_after = 2` before any learner reset | W5.5 **r10** |
| gate budget | **scheduled ladder** 100/200/400/800, +1 rung after 2 probations, -1 after a promotion; `promotion_every` unchanged. Ceiling provisional until the RTX 5090 per-rung fit | W5.8 / W5.7 cloud acceptance |
| gate cost | persistent rolling worker, concurrent seat legs, cloud-measured per rung | W5.1-5.4 **implemented**, W5.7 pending target host |
| stagnation | **games-indexed** anchor + metric-triggered intervention ladder | W7 |
| cloud target | vast.ai, single box | W6 |

---

## Workstreams

| # | Workstream | Size | Blocks launch? |
|---|---|---|---|
| W0 | Model size + precision, decided by measurement | M | ~~Yes~~ **DONE 2026-07-29** |
| W1 | Training schedules, growing window, HOF league | M | ~~Yes~~ **DONE 2026-07-29** |
| W2 | Memory: fix the crash, bound the footprint, measure it | M | ~~Yes~~ **DONE 2026-07-30** |
| W3 | Shared run-stats contract + reporting | M | ~~Yes~~ **DONE 2026-07-30** |
| W4 | BGA advisor end-to-end with the iter-60 model | M | No -- parallel |
| W5 | Gate efficiency + fixed-N Wilson decision rule + size ladder | M | **THROUGHPUT DONE; W5.5 REWRITE (r10), W5.8-5.10 AND THE RTX 5090 PER-RUNG FIT OPEN** |
| W6 | Cloud setup script | M | ~~Yes~~ **DONE 2026-07-30** (box-side acceptance runs on first launch) |
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

**Recovery semantics worth knowing before someone re-derives them from the
code:** rolling back an uncommitted iteration deletes its buffer file, its
candidate, *and* the persisted Adam moments. The journal never snapshotted the
moments, so the alternative was applying iteration-N moments to restored
iteration-(N-1) weights. The restarted iteration therefore warms up cold, which
is slower and sound; a post-crash iteration that looks slow is expected.

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
| `generation` | games, moves, moves/game (mean, median, p10, p90), decisions searched, mean sims, seconds, games/s, **NN rows, rows/s, forced-row share, mean batch size**, **exclusive opponent mix (`current_best`, `hof`, `bot`, `hof_bot`)** |
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

Game-extension observations are collected during the verified replay that
creates trainable positions. The compact per-game summary is cached with those
positions, so a warm cache supplies both without another replay. Warmup
iterations also derive and cache positions once before writing their row. This
removes the former unmeasured full replay from the adapter; its cost now belongs
to the existing `replay_derivation` phase.

Opponent provenance is explicit in every new buffer record. HOF assignment and
curriculum assignment can overlap: a real HOF-vs-bot game is `hof_bot`, while a
nominal HOF assignment shadowed by the bot on that seat remains `bot` and records
that the league assignment was unused. `az_report` reports exclusive counts and
computes realized HOF share as `(hof + hof_bot) / games` and realized bot share
as `(bot + hof_bot) / games`. Legacy `kind`-only buffers remain readable.

**Implementation status:** W3 is complete. Both review corrections—the
stats-only replay and lossy opponent attribution—are closed.

**W3.4 -- `tools/az_report.py`.** *(M)* One command, any run directory: the
block-aggregated outcome mix with binomial error bars, the gate ladder with
Wilson bounds, the learning curve, throughput, RSS, and the victory mix against
the ZeusAI/BGA reference. Without it the schema is just tidier JSON.

**Acceptance result:** 7WD and the synthetic second-game fixture emit valid
schema-v2 rows; opponent routing/serialization/reporting tests cover all four
categories and legacy fallback; `az_report.py` reproduces the planning tables,
realized opponent shares, and the throughput diagnostics needed to explain the
27% generation decay.

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

### Decision rule -- fixed-N, pair-level Wilson, three-way (rewritten in r10)

- **W5.5** *(M)* Port `games/kingdomino/promotion.py`'s Wilson rule and
  `_generator_action_after_promotion_check` (`self_play.py:2908`). The gate
  plays a **fixed** number of games, chosen *before* the match, and then decides
  **once**:
  - **observation unit = the seat pair**, outcome in {0, 0.5, 1}. Wilson assumes
    independence, and paired games share a seed; pairing the unit makes the
    bound exact rather than mildly anti-conservative;
  - Wilson interval on pairs at `z = 1.96`, **draws = 0.5 points**;
  - **promote** if LCB > `promotion_min_lcb` (**0.50**);
  - **revert** if UCB < `revert_max_ucb` (**0.48**);
  - **probation** otherwise -- i.e. whenever the interval still spans the
    threshold band. Probation is the default and the safe state in every
    direction: the learner keeps training, the next gate is another look;
  - **`revert_reset_after = 2`** (see the two-stage revert below).

Why this and not the Bayesian spec from revision 2: it is already implemented
and tested in this repo, Wilson handles draws natively (the Beta-Binomial
fudged them), and the three-way output maps directly onto `az_loop`'s existing
ACCEPT / CONTINUE / REJECT consumed by `gate_transition`.

Replaying run 03 under the LCB half of this rule agrees on **9 of 13** gates
(A7). The three promote → probation flips (gates 15, 20, 60) are gates SPRT
truncated at 74-154 games, where the interval is simply too wide to clear 0.50.
Recomputed at the pair unit with the r10 revert test, run 03's two reverts also
become probation -- so **no historical gate exercises the revert branch** and it
has to be covered by a synthetic test rather than a replay (A7, r10
restatement).

#### Sequential stopping is removed (r10)

Revisions 6-9 shipped a rule that evaluated both boundaries **after every pair**
and stopped on the first crossing. That is optional stopping, and it destroys the
calibration the whole rule rests on. Simulated (40k trials per cell, 10% draw
rate, z=1.96), the probability of **promoting an evenly-matched candidate**:

| cap | stop-every-pair | decide once at the cap |
|---|---:|---:|
| 200 games | **14.9%** | 1.9% |
| 800 games | **19.1%** | 1.8% |

The larger cap made it *worse*, which inverts the entire cap argument below. It
also mapped the futility crossing onto `REJECT`, so ~30% of evenly-matched
candidates took the revert path, and it forced `scheduler_workers=1` in the gate
path. Against that, the measured saving from futility stopping was **~8% of gate
games** against an even candidate -- under 1% of run wall time, less than the
sharding it prevented. Delete the sequential machinery on both sides of the
boundary (`gate_stop_pairs`, `abort_unfinished_gate`, `into_gate_prefix`, the two
pyo3 parameters, and the in-loop scan in `wilson_pair_decision`). The UCB
computation stays -- as a **terminal** decision on the fixed sample.

#### Correction: the pair unit costs one doubling (r10)

The r4 sensitivity table below was computed with **games** as `n`. The rule uses
**pairs**. Expected pair points equal the per-game win rate, so the numbers are
directly comparable, but every row shifts one doubling:

| games | pairs | pair win rate needed for LCB > 0.50 |
|---:|---:|---:|
| 200 | 100 | **0.598** |
| 400 | 200 | 0.569 |
| 800 | 400 | **0.549** |
| 1,600 | 800 | 0.535 |

So a 200-game gate needs a **+10%** edge, not +7%, and the +3.5% target needs
1,600 games, not 800. Wilson's binomial variance is conservative when splits are
common, so the true requirement sits somewhat below these; treat them as an
upper bound.

**The cap buys power, not safety.** Probability of promoting, by the candidate's
true per-game strength `q` (simulated, fixed-N, z=1.96):

| games | q=0.52 | q=0.55 | q=0.60 | q=0.65 | q=0.70 |
|---:|---:|---:|---:|---:|---:|
| 100 | -- | 8.9% | 25.1% | 50.6% | 76.7% |
| 200 | 4.6% | 13.4% | 43.9% | 79.7% | 96.8% |
| 400 | 6.5% | 24.0% | 74.0% | 97.8% | ~100% |
| 800 | 9.5% | **42.2%** | 95.9% | ~100% | ~100% |

The false-promotion column is flat at ~2% at **every** size. That is what makes a
small early gate safe, and it is why the size is a cost/latency choice rather
than a statistical one. It also explains run 02: at 13% power against a genuine
+5% net, a 200-game gate needs ~7 attempts to catch it.

#### Why revert is a UCB, and why 0.48

`revert_win_rate < 0.48` as a **point estimate** is unguarded, and it degrades
exactly where the ladder wants to operate. Probability of reverting an
evenly-matched candidate:

| games | point estimate < 0.48 | UCB < 0.48 |
|---:|---:|---:|
| 50 | **37.5%** | 3.4% |
| 100 | **35.3%** | 1.0% |
| 200 | **31.7%** | 0.7% |
| 800 | 19.0% | 0.1% |

Under a soft gate that is a one-in-three chance of acting against a candidate
that is simply equal. The UCB form fixes it: an equal candidate lands in
probation ~97% of the time at every size, and a small gate degrades gracefully
into "no opinion" rather than into noise.

The threshold is **0.48, not 0.50**, and the asymmetry is deliberate in the
opposite direction from r4's. A fixed threshold gets *more* sensitive as `n`
grows, and the ladder grows `n` precisely when true differences are smallest, so
a symmetric 0.50 rule becomes trigger-happy late in the run -- 43% revert against
a q=0.45 candidate at the 800 rung. Mild regression during training is normal and
expected, particularly across an LR or curriculum-mix knot, and the learner often
needs a few iterations to work through it. Detection probability by true strength
(simulated):

| games | q=0.35 | q=0.40 | q=0.45 | q=0.50 |
|---:|---:|---:|---:|---:|
| 100 | 38.1% | 16.5% | 4.9% | 1.0% |
| 200 | 65.0% | 27.9% | 6.3% | 0.7% |
| 400 | 92.3% | 52.0% | 9.5% | 0.4% |
| 800 | 99.8% | 81.2% | 14.9% | 0.1% |

#### Two-stage revert: `revert_reset_after = 2`

`gate_transition` already separates the cheap action from the expensive one, and
the cloud run uses both:

- **REVERT** -- the generator source switches to `current_best`. The learner
  keeps its weights and keeps training. During a genuine hump this is the right
  thing to do anyway: generate from the known-good net while the learner works
  through the disruption. Nothing is lost.
- **REVERT_RESET** -- `latest ← current_best`. This is the action that discards
  learning, and it fires only after **2 consecutive** reverting gates.

Requiring persistence rather than lowering the threshold is what separates a
transient dip from a real regression. Probability of reaching REVERT_RESET
(per-gate rates squared; conservative for a transient dip, since the second
gate's true strength has already recovered):

| rung | persistent q=0.40 | transient q=0.45 | equal q=0.50 |
|---:|---:|---:|---:|
| 400 | 27% | 0.9% | ~0% |
| 800 | **66%** | **2.2%** | ~0% |

A diverging learner (NaN, bad LR) sits at q ≈ 0.2-0.3 and is caught at the bottom
rung immediately. A mild dip essentially never costs the learner. That is the
separation the rule is for.

- **W5.9** *(S, new in r10)* **Suppress revert across schedule knots.** The LR,
  curriculum-mix, and window schedules are on a games clock, so their knots are
  known config points rather than surprises. Force probation for the first gate
  after a knot crosses. Promotion stays live -- a candidate that clears LCB > 0.50
  right after an LR change is genuinely better and there is no reason to withhold
  it. This targets the named mechanism directly instead of de-tuning the rule for
  the other 95% of the run.

- **W5.10** *(S, new in r10)* **Log the per-pair score vector** in the W3 gate
  block, not just the aggregate. A few hundred bytes per gate, and it makes any
  past gate re-decidable under a different threshold or size without replaying
  games -- the property that made the A7 replay possible in the first place.

### The gate-size ladder (new in r10)

- **W5.8** *(M, new in r10)* Gate size is **scheduled**, not a single cap. Rungs
  **100 → 200 → 400 → 800** games. Step **up** one rung after **2 consecutive
  probations**; step **down** one rung after a promotion; hold a games-clock floor
  before laddering at all so bootstrap noise does not ladder up. Early in a run
  the learner is improving fast (q ≈ 0.65-0.70), where a 100-game gate already
  promotes 51-77% of the time; late in a run the candidate is +2-3% and the
  evidence has to be bought. Choosing `n` from *prior* gates and the clock is
  statistically clean in a way sequential stopping is not: `n` is fixed before
  the games are played, and the candidate is a different net every gate anyway.

**`promotion_every` stays fixed.** Stretching the cadence as the rung rises would
hold gate share constant at the cost of promotion latency; spending more of a
mature run on the decision is the better trade, and it is made deliberately here
rather than by accident. The consequence is that gate share **grows with the
rung**, and the top rung is what W5.7 has to price.

**Sizing for the pool, not for the worker count.** The quantisation that matters
is not `n mod slots`:

- `n` must be **even** -- both legs of a pair sit in the pool simultaneously, so
  occupancy moves in units of two games. `gate_slots` should be even too.
- `n` should be **>> `gate_slots`**, not congruent to it. Games have variable
  length so they never finish in lockstep; W5.4's rolling refill removes the
  drain, leaving a single tail at the end of the gate bounded by roughly one
  pool's worth of games. At 48 slots the 100-game bottom rung is only ~2
  pool-fills deep, which is where that tail starts to bite -- either keep the
  bottom rung at >= 4x `gate_slots` or lower `gate_slots` on the low rungs.
- If the gate is sharded (possible again now that the sequential stop is gone),
  `n` divisible by **2 x shards** keeps shards balanced. Record order is
  preserved globally, so pairing stays correct either way.
- What actually drives GPU efficiency is `gate_slots` x leaves-in-flight ≈
  `global_batch_cap`. Tune that; let `n` be any even number above the floor.

### The rungs' cost must be re-derived at L's width (r4, rescoped in r10)

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

Those two estimates imply **~50 s fixed + ~3.2 s per game** once W5.1-5.4 have
landed -- i.e. the fixed cost that motivated the whole throughput package is
mostly gone and gate cost is now **near-linear in games**. That is what makes the
W5.8 ladder worth having: a low early rung genuinely costs proportionally less,
where under the old 193 s fixed cost it would not have.

It also sets the bill for the top rung. Iteration wall time required to keep
gates at 10% of the run, at `promotion_every = 4` (share = G / (4I + G), so
I = 2.25 G):

| rung | gate cost (**estimate**) | iteration time needed for a 10% share |
|---:|---:|---:|
| 100 games | ~370 s | ~14 min |
| 200 games | ~690 s | ~26 min |
| 400 games | ~1,330 s | ~50 min |
| 800 games | ~2,610 s | **~98 min** |

If iterations land nearer 30-40 minutes, the 800 rung is 30-45% of wall time.
That is not automatically wrong -- late in a run, where self-play is returning
+2-3% and the promote decision is the hard part, it may well be the better trade
-- but it is a policy choice to make against a measured number, not one the
ladder should make silently.

**W5.7** *(S, r4; rescoped in r10)* **Re-measure the gate cost at L's width, per
rung.** Same method as A5, at **100/200/400/800** so the fit is not a two-point
extrapolation, plus a **representative ungated iteration wall time** from the
same host. The deliverable is no longer "the cap" -- the ladder replaces it --
but the **cost of each rung and the share each implies**, which is what sets the
ladder's ceiling. Do not hard-code the top rung until it lands.

**Decided 2026-07-29:** W5.7 runs against **W0's `sweep_L_lr5e-05_seed*.pt`
checkpoints**, not a future cloud checkpoint. Gate cost is driven by tensor shapes
and the sims budget, not by checkpoint quality, so a 4,000-step fixed-corpus fit
is an adequate timing proxy -- and it means W5.7 is unblocked today rather than
waiting on the run it is supposed to configure. The one thing checkpoint quality
does affect is **game length** (W0 measured 68-70 moves/game across all three
arms, so the effect is small); record moves/game with the fit so the assumption is
checkable rather than assumed.

Negative result worth recording, and the reason r10 stopped treating futility as
a stopping rule: a statistically valid futility stop **never fires at n <= 200**
-- even a 0.422 win rate at 96 games still has UCB ~0.52 -- and even at n ~ 800 it
saves only ~8% of gate games. The UCB is worth computing; it is not worth
computing *mid-match*. It earns its place as the terminal revert test instead.

- **W5.6** *(S)* **Gates are decisions; anchors are measurements.** W7's anchor
  check runs **fixed-N with no early stopping**, at `--anchor-games` per
  opponent. Run 03 fed SPRT-truncated matches straight into `self.elo.record()`,
  which biases the ladder toward whichever boundary stopped each match.
  **Implemented:** promotion gates no longer feed Elo at all -- fixed-N anchors
  are the only ladder input, and `promotion_gate` records nothing. A run whose
  `elo/elo_games.jsonl` contains model-vs-model rows has regressed.

### Known residual risk: no promotion rollback

Verified: `ACCEPT` sets `replace_best=True` (`training_control.py:195`);
`REJECT` sets `replace_best=False` (`:225`), discarding the *candidate* and
keeping the possibly-bad best; `reset_learner` copies `current_best → latest`
(`run_controller.py:450-458`), propagating it; anchor gates return metrics with
**no lifecycle effect** (`run_controller.py:460`). **A bad promotion is never
undone anywhere in the system.**

The mitigation is the strict promote bound: LCB > 0.50 at z=1.96 is ~97.5%
one-sided confidence, held at ~2% false promotion at **every** rung by deciding
once at a fixed `n` (r10; the sequential rule it replaces ran at 15-19%). Note
that this is the error rate against an *equal* candidate, which is the case where
a wrong promotion costs nothing; the tail that matters -- promoting a materially
worse net -- is far smaller (0.5% at q=0.48, ~0.02% at q=0.45, at the 400 rung).
Reverting is the recoverable direction and is deliberately the less trigger-happy
one: UCB < 0.48, and two consecutive before any learning is discarded.
**Periodic HOF revalidation of
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
  gate, so the pair never repeats. Note this cuts both ways: it is also why a
  stagnating learner genuinely needs a bigger sample rather than more gates, which
  is the argument for the W5.8 ladder.
- *Sequential stopping inside a gate* (r10, after shipping it in r6-r9) -- 15-19%
  false promotion against an even candidate, futility mapped onto REVERT, and
  `scheduler_workers=1` forced, in exchange for under 1% of run wall time. Do not
  re-propose it. Adapting the **next** gate's size from **past** gates is a
  different thing and is what W5.8 does.
- *Relaxing `z` to buy power* -- considered in r10 as an alternative to the
  ladder. It works (z=1.28 at 400 games has more power than z=1.96 at 800, with
  the q<=0.45 tail still under 0.4%), but it adds a second dial that interacts
  with the first. `n` alone is the dial; `z` stays at 1.96.
- *Harvesting gate games as training data* -- 1,240 games is **4.4%** of the
  28,000 self-play games, argmax-only, no search distributions, off-policy for
  both nets, and `_play_two_net_games` builds no `GameRecord` at all.

---

## W6 -- Cloud setup: DONE (2026-07-30)

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

**Implementation status (2026-07-30): built and tested; the box-side half of the
acceptance runs on first launch.**

| task | where |
|---|---|
| W6.1 | `setup_cloud_common.sh` (sourced, never executed) + a rewritten `setup_cloud_7wd.sh` with the Rust stages it never had |
| W6.2 | `cloud_equivalence_smoke.py` + `cloud_equivalence_manifest.json`; a 50-game corpus committed under `testdata/equiv_corpus/`; `pytest` added to `requirements.txt` |
| W6.2b | `precision_arena.py` |
| W6.3 | `f4_launch_flags.py`, and `phase_d.build_parser()` extracted so the translation can be checked without launching |
| W6.4 | `cloud_preflight.py` |
| W6.5 | `RunManifest.code_identity()` + `PhaseDLoop._refuse_changed_code` |
| W6.6 | `RunController.heartbeat_line` → stdout and `heartbeat.log` |
| W6.7 | `games/az_loop/snapshot.py` |

Three things the build changed relative to this specification:

1. **The corpus tests no longer skip -- they fail.** With a committed corpus,
   absence is a repository fault rather than an environment one, and the whole
   point of W6.2 is that this suite must never be quietly absent. The runner
   additionally checks that every test in the manifest **ran**: `-k` deselection
   happens after pytest's collection hook, so a filtered-out test still looks
   collected, and the first version of this check passed while two tests sat out.
2. **Mixed-precision routing had to be built for W6.2b to mean anything.**
   `rust_searcher_routed_flat_batch_adapter` wrapped the whole batch in one
   autocast and refused two evaluators of different precisions. Lifting the
   refusal without per-net autocast would have produced an arena that compared
   bf16 with bf16 and reported a null. Each net now applies its own; the
   single-precision path is untouched, so W1's bit-exact routing is unaffected.
3. **The launch command is tested against the real parser.** A renamed flag in
   `TRAIN_CMD` would otherwise be found on a rented box, after the toolchain
   build, the equivalence suite, and the smoke have all passed.

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
      -> W3 (stats) -- DONE 2026-07-30
        -> W5 (gates) -> W6 (cloud) -> W7 -> LAUNCH
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
W5 throughput before the ladder's upper rungs: the fixes pay for them.
W5.5's rewrite before any cloud gate runs: the r6-r9 sequential rule promotes an
even candidate 15-19% of the time, and no promotion is ever undone.
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

**r10 restatement of A7 at the pair unit.** The two lines above are per *game*;
at `n = games / 2` the requirement is 0.598 @200, 0.569 @400, 0.549 @800 (see
W5.5). The "KD rule" column also used the old point-estimate revert. Re-deciding
the two revert rows under `UCB < 0.48`: gate 50 (0.465 over 100 pairs) gives
[0.370, 0.562] and gate 65 (0.422 over 48 pairs) gives [0.293, 0.562] -- both
**probation**. So run 03 contains **no** evidence that would have reverted under
the shipped rule, which is the intended conservatism, and it means the revert
branch is unexercised by any historical run. It needs a synthetic test, not a
replay, to be trusted.

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
