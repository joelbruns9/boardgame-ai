# Design Document — CPU Parallelism for the Self-Play Training Loop

> **Status:** rewritten after code audit + measurement (2026-06-29). The
> diagnosis evolved twice — record kept so we don't relitigate:
> 1. **Original doc:** "`PyResult` return types force GIL serialization in
>    `step()`/`update()`." Wrong — those loops already run parallel.
> 2. **First fix attempt (mine):** "`resolve_exact_slots` is serial *across
>    slots* → parallelize across slots." Implemented, then **reverted** — it made
>    fallback *worse* (80%→95%), because the training solver is internally serial
>    and running many such solves at once oversubscribes the cores.
> 3. **Actual root cause (validated):** the training path used the **single-core
>    serial** alpha-beta solver; the **parallel YBW** solver existed but was never
>    wired into training. Fix = make each solve use the whole machine
>    (within-solve parallelism). Fallback **95%→30%**, exact-solve success
>    **5%→70%**. See "Step 1".

---

## Goal

Use all available CPU cores during self-play + training, with the split between
**game generation** and **endgame solving** allocated **dynamically** by the
scheduler — never a fixed budget. The endgame solver is **essential to training
quality** and must stay in the pipeline; the task is to make it feasible via good
CPU management. Target deployment is a cloud box with a 5090 and 16–32 vCPUs, not
just the development laptop.

---

## Corrected Diagnosis (the `step`/`update` GIL myth)

The original doc claimed `step()`/`update()` `par_iter_mut` is "GIL-serialized
because `PyResult<T>` requires the GIL." **Not how PyO3/Rayon interact.**
`PyResult<T>` is just `Result<T, PyErr>` — a plain Rust type; a return type
cannot acquire the GIL. Rayon workers are OS threads that never touch the GIL
unless code calls into CPython. Audit confirms those closures are pure Rust on
the hot path (`actor()` lib.rs:733 pure; `legal_actions_indexed()` lib.rs:923
returns plain `Vec`; `SlotStepOutput`/`MoveRecord` plain Rust). **So `step()`'s
`par_iter_mut` already runs in parallel.** That diagnosis is dead.

The "10% CPU" figure was a whole-iteration average dominated by (a) the
**single-core** endgame solver and (b) GPU-eval idle time — not by `step`
serialization.

### Measured bottleneck evolution (historical)

- **Runs 1–6 (no solver):** `step≈120s(15%) eval≈700s(85%) update≈10s(1%)` —
  GPU-bound; one core kept the batch full.
- **Run 8 (solver on, 3s budget):** `step=334s(40%) eval=479s(58%) update=13s` —
  the solver tripled step time and starved the GPU (fill 85%→58%), at ~80%
  fallback.

---

## Step 1 — Make each endgame solve use the whole machine (DONE, validated)

### Root cause

The training solve path is **single-threaded**:
`solve_exact_plan` → `solve_root_exact_cached` → `solve_endgame_ab_value_cached`
→ **`solve_endgame_ab`** (lib.rs:3499), the serial alpha-beta. Each training
solve ran on **one core**.

A **parallel YBW solver** already exists — `solve_endgame_ab_parallel`
(lib.rs:2132): solves the first root child serially to seed a bound, then the
remaining children via `remaining.par_iter()` (lib.rs:2182), using all cores. But
it was wired **only into the `measure`/bench pymethods, never the training
path**. The benchmark in `docs/heuristic_testing.md` (best ordering
`lookahead2_clustered`: p50 0.64s, **p90 3.35s**, ~88% solvable within 3s) is
*that* solver on the full machine — the feasibility proof.

So the original design note "solve_endgame_ab is single-threaded and BatchedMCTS
already parallelises across slots" was the bug: across-slot parallelism gives
*throughput* but never speeds up an *individual* solve, and the budget is
**per-solve wall-clock**. A position needing ~15s on one core still times out at
3s no matter how many slots run beside it → ~80–95% fallback.

### Why the across-slots `par_iter` (first attempt) was wrong

Parallelizing across slots ran many internally-serial solves concurrently. It
did **not** speed up any individual solve, and it **oversubscribed** the cores,
inflating each solve's wall time → fallback rose 80%→95%. The apparent "9× faster
solver" was illusory: 95% of solves were *failing faster in aggregate*, not
solving. Reverted.

### The fix (implemented)

1. **Within-solve parallelism:** `solve_root_exact_cached` now solves its root
   children with `legal.par_iter()` — each child's exact (full-window) value on
   its own core (lib.rs:~3532). Mirrors YBW; gives the per-child policy values the
   training target needs. Value-correct: the 6 move-ordering value-invariance
   tests + full suite pass (38/38).
2. **Serial across slots:** `resolve_exact_slots` reverted to a serial loop, so
   only one (internally-parallel) solve runs at a time — no nesting, no
   oversubscription.
3. **Diagnostic:** `exact_solver_secs` getter (lib.rs) + `self_play.py` logging.

### Validated results (laptop, 8 physical cores; 100 games, 1600 sims, 3s budget, warm run6/iter50)

| Metric | Run 8 serial (orig, ≈) | Across-slots par_iter (wrong) | **Within-solve YBW (correct)** |
|---|---|---|---|
| **exact-solve success (endgames)** | ~20% | **5%** (5/100) | **70%** (70/100) |
| **fallback** | ~80 | 95 | **30** |
| exact moves played (solved) | — | 60 | **840** |
| `solver_secs` | ~280s | 30.3s (illusory) | 187.7s |
| step phase | 334s (40%) | 91.7s (13%) | 244.6s (30%) |
| GPU eval | 479s (58%) | 614.7s (85%) | 565.7s (69%) |
| batch fill | 58% | 73% | 78% |
| games/s | — | 0.14 | 0.12 |

**Verdict:** the solver now delivers exact endgame values for **70% of games**,
up from ~5–20%. This is the win that matters — exact targets are what make the
endgame solver valuable to training. The user's premise was right: the solver is
feasible with proper CPU management; the management is *parallelize within the
solve, serialize across slots*.

### The remaining cost → Step 1.5

The solver is now back **on the critical path** (187s, 30% of step), so GPU fill
dipped to 69% and games/s to 0.12. That's the cost of the solver actually
running. It is **not** a regression in value — it's latency that should be
**hidden behind GPU eval**, not removed. See Step 1.5.

### Levers to push solve success past 70% (separate quality experiment)

- **Decouple exact value from exact policy (highest-potential lever).** The
  training solve currently does **full-window per child** — an exact value for
  *every* root child, no cross-child alpha-beta cutoffs, because the policy target
  consumes all child values. That is the expensive, deadline-sensitive mode: you
  pay full-window cost even on children that will get ~0 policy mass, and one
  child missing the deadline reverts the *whole* endgame to MCTS. A best-move
  solve (alpha-beta *with* cutoffs, like the bench's YBW path) gets the exact game
  value + principal move far more cheaply — potentially pushing success well past
  70% at the same 3s budget, a bigger win than parallelism.
  - **Prerequisite check:** confirm whether `exact_policy_target` builds a *soft*
    distribution over all child values or is effectively one-hot on the optimal
    move. If one-hot, the all-child solve is pure waste and decoupling is free; if
    soft, decoupling trades target richness for solve speed and needs a
    **training-quality A/B**, not just a throughput measurement.
  - **Partial > binary fallback:** exact mass on proven-optimal children + prior/
    visit mass on the unproven rest is likely better supervision than full
    fallback. Keep it simple — do *not* blend exact values with MCTS visit counts
    in one target (different scales, injects noise); bounded-interval targets are
    over-engineering for now.
- **Bigger budget** (3s→8–10s) becomes affordable once solves overlap the GPU —
  but test it only *after* the decoupling above; if the tail needs minutes, more
  budget won't reach it.
- Move ordering already tuned (`lookahead2_clustered`, see
  `docs/heuristic_testing.md`).

This is **orthogonal to the overlap work** and gated on a training-quality
comparison — defer until after Step 1.5 is measured.

---

## Step 1.5 — Overlap the solve phase with GPU eval (laptop CPU-management win)

Run the serial-across-slots solve phase **concurrently with GPU eval** instead of
blocking on it. During a GPU forward pass the CPU is ~idle, so a background solver
(concurrency 1, each solve using full-machine YBW) hides most of the 187s behind
the 565s of eval — recovering GPU fill toward 85% and games/s, with no loss of
solve success. This is the small-scale cousin of Step 2 and may be skipped if we
go straight to Step 2.

Key constraint: **one solve at a time** (full-core YBW), overlapped — *not* many
concurrent solves (which re-creates the oversubscription that broke the first
attempt).

### Attempt 1 — GIL release + existing double-buffer path (measured, insufficient)

Implemented: `resolve_exact_slots` now runs its (pure-Rust) body under `py.detach`
(GIL released), so a second thread can drive the GPU while this instance solves.
Tested with the existing `--double_buffer` loop (two `BatchedMCTS` A/B instances,
one's GPU eval overlapping the other's CPU work on a 1-worker executor).

| | single-buffer | double-buffer |
|---|---|---|
| games/s | 0.1216 | **0.1185** (worse) |
| step / eval (% of wall) | 30% / 69% (sum 100%) | 30% / 74% (**sum 105%**) |
| batch fill | 78% | 74% |
| solver_secs | 187.7 | 201.8 |

The `>100%` timing sum **proves the overlap is real** — but only ~45s of it, and
it's **net-negative** because:

1. **A/B phase synchronization.** Both instances start their 50 games at tick 0
   with near-constant game length, so their endgame bursts (and eval phases)
   **align** — when A solves, B is *also* solving, not evaluating. Little
   solve-vs-eval overlap. (This is the *same* clustering that motivates staggering
   — here it defeats double-buffering across the two instances.)
2. **Batch-split penalty.** Splitting 100 games into two 50-game instances drops
   fill 78%→74% and raises total eval time, eating the small overlap gain.

**Kept:** the `py.detach` GIL release — it's correct, harmless in single-buffer,
and a prerequisite for *any* overlap. **Rejected:** double-buffer as the vehicle
(phase-sync + batch-split cap it).

### Solver cost/benefit — controlled A/B (solver off vs on, same config)

| | solver OFF | solver ON (within-solve) | Δ |
|---|---|---|---|
| eval_sec | 630.3s | 565.7s | **−64.6s (−10%)** |
| ticks | 55,744 | 46,096 | −17% |
| total_evals (≈) | 7.8M | 6.9M | −12% |
| batch fill | 73% | 78% | +5pp |
| step_sec | 61.2s | 244.6s | +183s (solver) |
| **games/s** | **0.140** | **0.122** | **−13%** |
| endgames solved | 0 | 70/100 | — |

Two findings:

1. **The solver genuinely offloads the GPU** — ~12% fewer positions, ~10% less
   eval time — and it *raises* batch fill (73%→78%), because endgame positions
   have few legal moves and contribute small, inefficient batches; removing them
   leaves the fuller mid-game positions. (Earlier guess that fill would drop was
   wrong.)
2. **But on the critical path the solver makes self-play *slower*** (0.122 vs
   0.140 games/s): it saves 64s of GPU but adds 183s of CPU. Today the solver
   buys training quality at ~13% throughput cost.

**This is the motivation for the overlap.** Hide the 183s behind the 630s of eval
and — because the solver *also* cuts eval by 64s — overlapped solver-on could in
the **best case** approach wall ≈ 600s → games/s ≈ 0.15–0.16, faster than
solver-off AND with exact targets, flipping the solver from a 13% cost to a net
gain. Treat that as an optimistic *ceiling*, not a projection: it assumes most CPU
solve time is hideable and ignores reduced active slots during clusters,
solver/eval CPU interference, encoding/update work, and cache contention. The
realistic gate is below.

### Attempt 2 — async solve jobs + overbooked search slots (recommended next)

Overlap **within one instance**, no batch split, phase-independent. Two coupled
ideas:

1. **Detached solve jobs.** When a position enters `ExactSolving`, snapshot it
   (`real_state` clone — already cheap) and dispatch the solve to a **background
   solver thread**, carrying the *game identity + records* so the completed solve
   re-attaches to its game, not its old slot. The main loop never blocks on a
   solve. While the GPU evaluates non-endgame positions, the background thread
   (GIL already released) solves the endgame ones — overlap is intrinsic and
   phase-independent.

2. **Overbooked search slots (the key throughput idea).** Do *not* simply
   "exclude the solving slot and continue with ~30" — under clustering, 20 slots
   entering `ExactSolving` together collapses the GPU-fed batch exactly when solve
   load peaks. Instead **separate "active search slots" from "pending solve
   jobs"**: maintain a `target_active_search_slots` population by **backfilling a
   replacement game** into any slot whose game is out on a pending solve. Solved
   endgames rejoin later; the GPU-facing search population stays full regardless
   of how many games are mid-solve. This *tolerates* clustering instead of trying
   to prevent it, and without splitting batches (what sank double-buffering). It
   strictly dominates staggering for batch fill; staggering only smooths the
   solve-arrival rate and is secondary.

Cost: more concurrent game state in flight (≈ `target_active_search_slots`
searching + the pending-solve backlog + snapshots) and bookkeeping to attach a
completed solve to its game. This is the same machinery as Step 2's "solver inline
in worker," so it doubles as a Step-2 stepping stone — **design the slot/job
separation in from the start; retrofitting it is expensive.**

### Attempt 2 — measured (built, correct, and a net win once the pools are split)

Built: standalone `play_out_exact_endgame` (owned data) + a background solver
thread + `SolvingInBackground` slot state + dispatch/harvest + in-place fallback
resume + overbooking via `--batch_slots`, behind `--async_solve` (default off).
A **dedicated solver thread pool** (`--solver_cpus N`) confines the within-solve
YBW `par_iter`; game generation gets the rest of the cores via the global pool.
Correctness: deterministic, **identical per-seed games vs sync**; sync path 38/38.

| | games/s | step | eval | update | wall | solved |
|---|---|---|---|---|---|---|
| sync@32 (baseline) | 0.122 | 244.6s | 565.7s | 11.5s | 822s | 70 |
| async@32 (shared pool) | 0.118 | 168.7s | 631.4s | 46.3s | 846s | 64 |
| async@48 (shared, overbook) | 0.128 | 143.8s | 587.7s | 48.0s | 779s | 69 |
| **async@32, solver_cpus=6** | **0.139** | 62.7s | 643.7s | 13.4s | 720s | 59 |
| solver-off@32 | 0.140 | 61.2s | 630.3s | 12.8s | 704s | 0 |

**The first "contention" diagnosis was wrong; the real cause was a shared pool.**
It is *not* raw core starvation — ~1 core feeds the GPU (pre-solver fact), so 8
cores have room for solving + feeding. The problem was that the solver and the
GPU-feeding **shared the global Rayon pool**: the within-solve YBW submits long,
non-preemptible subtree-solves, and `step()`/`update()`'s descent/backup
`par_iter` queued *behind* them (head-of-line blocking). That inflated `step`,
`eval`, **and** `update` together (all contend for the shared pool / cores).

**The fix — a dedicated solver pool — works.** With `solver_cpus=6` (6 cores
solving, ~2 reserved for generation), all three phases snap back to the
solver-off levels (`step` 62.7s ≈ 61.2s; `update` 13.4s ≈ 12.8s — confirming the
`update` blow-up was contention, **not** a bug), and **games/s = 0.139 ≈
solver-off 0.140**. The solver is now **throughput-free on the laptop** while
delivering exact targets for 59/100 games. The earlier "needs cloud cores"
conclusion was premature.

**The knob trades throughput vs solve-success.** Fewer solver cores → faster
generation but slower per-solve → fewer endgames solved in budget (solver_cpus=6:
59 solved; the shared-pool/full-machine runs solved ~64–70 but contended). That
is exactly the per-machine `--solver_cpus` dial. The **cheaper-solver lever**
(value/policy decoupling) compounds: more solve-success per core means the same
quality at fewer cores, freeing more for generation. Next steps: sweep
`solver_cpus` (e.g. 5/6/7) for the throughput↔solve-success knee, and carry the
dedicated-pool design into Step 2 (where it *is* the admission control).

### Step 1.5 gate (stricter than games/s alone)

Whole-iteration games/s is too coarse. Gate on:

- **evaluator busy fraction** (the real overlap proof — should approach the
  no-solver eval-bound regime) and **mean active non-solving slots / batch fill**.
- **solve-queue depth** over time (does the backlog drain, or grow unboundedly
  during clusters?).
- **p50 / p90 / p99 solve latency** (the p99 tail is what creates fallbacks).
- **fallback rate** (must not regress vs the 30% single-buffer baseline).
- **end-to-end games/s** vs both solver-off (0.140) and single-buffer solver-on
  (0.122).

Pass = games/s recovers toward/above solver-off **and** fallback holds, with the
evaluator measurably busier. Unbounded solve-queue growth during clusters (even
with overbooking) signals the admission-control problem (below) has arrived early.

---

## Why the laptop assumption breaks on the cloud 5090

On the laptop one core keeps the GPU fed, so descent never needed to be parallel.
**Laptop-specific.** On a 5090: faster inference ⇒ higher leaves/sec demand ⇒
single-core descent starves the GPU; and cloud boxes give 16–32 vCPUs to exploit.
So the real target is a model where **descent and solving both overlap the GPU**
and the scheduler allocates cores dynamically — Step 2.

### The dynamic-allocation principle (with a caveat)

> Never statically *partition* threads into fixed role quotas. Put descent tasks
> and within-solve YBW child tasks into **one work-stealing pool** so no core sits
> idle while there is work.

But work-stealing is **load balancing, not admission control.** Rayon has no
notion that descent is latency-critical (it feeds the GPU) while solves are
throughput-work — so if many games enter `ExactSolving` together, a flood of YBW
child tasks can crowd out descent and starve the GPU, oscillating between great
solve latency and poor GPU feed. "No manual budget ever needed" is too strong. At
cloud scale Step 2 likely needs an explicit policy — at most *K* active root
solves, or *M* workers reserved for search/eval submission, or solve admission
driven by evaluator-queue depth / GPU-busy fraction. On the laptop (GPU-bound, CPU
slack) pure-pool contention is unlikely to bind, so this is a **Step-2 concern,
deferred until measurement shows it actually binds** — don't build the controller
speculatively.

---

## Step 2 — Thread-per-worker + coalescing evaluator (the cloud target)

Not greenfield — the hard part exists and is benchmarked (milestone 6):
`RustMCTS::search` (lib.rs:2976) drives one game with `py.detach` (GIL released
for all tree work); `evaluate_batch` (lib.rs:2424) + the in-process **coalescing
evaluator** batch NN requests across concurrent game threads (~6151 evals/s).

### The model

Run **N OS worker threads**, each driving a subset of games end-to-end with the
GIL released; a shared coalescing evaluator gathers leaf requests into GPU
batches. **The endgame solver runs inline in the worker** that owns the game
(pure Rust, GIL already released), via the same detached-job + overbooking design
as Step 1.5.

- **Descent and solving both overlap the GPU** — while a worker waits on the
  evaluator, others do CPU work.
- **Cores stay busy without fixed role quotas** — workers in endgames solve,
  workers in normal play descend; one work-stealing pool keeps cores fed. But see
  the admission-control caveat above: load balancing is not prioritization.
- **Scales to the 5090 + many vCPUs.**

### Step 2 scope

1. Adopt the multi-thread `RustMCTS` + coalescing-evaluator path for the training
   driver (today: synchronous `BatchedMCTS` `step`/`update`).
2. Run the (within-solve-parallel) endgame solver as detached jobs in the worker
   loop, with overbooked search workers (carry the Step-1.5 design forward).
3. Wire self-play data collection / finished-game handling into the worker model.
4. **Tune the coalescing evaluator explicitly.** Worker-per-game async submission
   can fragment batches vs. synchronized `BatchedMCTS`, so CPU parallelism can
   rise while GPU efficiency falls — a net wash. Specify and gate on `max_batch`,
   `max_wait_ms`, and worker count, with a **mean-batch / fill acceptance
   criterion on the 5090**. (The evaluator exists and batches across threads,
   ~6151 evals/s; what's unmeasured is its fill vs. the synchronized model.)
5. **Admission / adaptive solve control — only if measurement requires it.** Start
   with the shared pool. If Step-2 metrics show solve work starving the GPU feed
   (solve-queue growth + dropping evaluator-busy fraction), add a knob: cap active
   root solves at *K*, and/or drive solve admission from live evaluator-queue
   depth and GPU-busy fraction (fewer active solves when the GPU is starved, more
   when it is saturated and the solve queue grows). Do not build this speculatively.

### Step 2 gate

- **Evaluator busy fraction** stays high (overlap holds) and **mean batch / fill**
  meets the acceptance criterion (coalescing not fragmenting batches).
- Endgame fallback rate stays low (per-solve deadline still met under cloud
  oversubscription — re-check with the Step-1.5 instrumentation).
- **solve-queue depth bounded** (admission control engaged if not).
- Training-quality curves unchanged; games/s scales with cores.

---

## Dropped from the previous plan (and why)

- **"GIL release in `step()`/`update()`" (old Phase 2/3)** — premise false; those
  loops are already parallel. No-op.
- **`actor_internal()` / `legal_actions_indexed_internal()`** — unnecessary;
  already pure Rust / plain types.
- **Across-slots `par_iter` over the solver** — actively harmful (oversubscribes
  the internally-parallel solver). Reverted.

---

## Files in scope

| File | Step | Change | Status |
|---|---|---|---|
| `lib.rs` | 1 | `solve_root_exact_cached`: `par_iter` over root children (within-solve YBW); `resolve_exact_slots` serial across slots | **done** |
| `lib.rs` | 1 | `exact_solver_secs` getter | **done** |
| `self_play.py` | 1 | log `exact_solver_secs` | **done** |
| `lib.rs` / `self_play.py` | 1.5 | overlap serial solve phase with GPU eval | todo |
| `lib.rs` / `self_play.py` | 2 | worker-pool + coalescing evaluator; solver inline; shared Rayon pool | todo |

---

## Recommendation

1. **Step 1 is done and validated** — solver success 5%→70%, fallback 95%→30%.
   The solver now earns its place.
2. **Step 1.5 (async solve jobs + overbooked search slots)** is the next build.
   Design the slot/job separation in from the start; gate on the stricter metrics
   above, not just games/s.
3. **Step 2** is the cloud-5090 answer for dynamic all-core use; reuse the
   milestone-6 coalescing path, carry the detached-job + overbooking design
   forward, and tune the coalescing evaluator explicitly.

### Build simple first; defer the rest until measurement demands it

The design can grow into a sophisticated system (adaptive controllers,
interval-blended targets, admission policies). That is the right *eventual* shape
but the wrong *first* build — tuning machinery for contention that may not bind at
this scale wastes effort. Build the simple version, instrument it, and add
complexity only where the numbers show a real problem.

**Deferred until after Step-1.5 testing (pending measurement):**
- Admission / adaptive solve-concurrency control (Step 2; only if the shared pool
  starves the GPU feed).
- Hybrid / partial exact-target mode and value/policy decoupling (a separate
  *training-quality* experiment; verify the target is soft first).
- Raising the solve budget (3s→8–10s) — test only after the decoupling lever.
- Coalescing-evaluator tuning targets for the 5090 (Step 2).
