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

### Levers to push solve success past 70% (later, once overlapped)

- The training solve does **full-window per child** (no cross-child alpha-beta
  cutoffs, because every child's exact value feeds the policy target) — heavier
  than the bench's single best-move YBW solve, so its in-budget rate is below the
  bench's ~88%. Revisit only if the policy target can tolerate pruned child
  values.
- **Bigger budget** (3s→8–10s) becomes affordable once solves overlap the GPU.
- Move ordering already tuned (`lookahead2_clustered`, see
  `docs/heuristic_testing.md`).

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
and — because the solver *also* cuts eval by 64s — overlapped solver-on should
land near wall ≈ 600s → games/s ≈ 0.15–0.16, i.e. **faster than solver-off AND
with exact targets.** The async solve queue flips the solver from a 13% cost to a
net gain.

### Attempt 2 — single-buffer async solve queue (recommended next)

Overlap **within one instance**, no batch split, phase-independent: when a slot
enters `ExactSolving`, dispatch its solve to a **background solver thread**; the
main loop excludes that slot from the batch and keeps `step→eval→update` on the
other ~30 slots; harvest completed solves and rejoin. While the GPU evaluates the
non-endgame slots, the background thread (GIL already released) solves the endgame
ones — different slots of the same instance, so overlap is intrinsic and does not
depend on phase. This is the same machinery as Step 2's "solver inline in worker,"
so it doubles as a Step-2 stepping stone. Staggering (see below) further smooths
the background queue's load but is secondary.

---

## Why the laptop assumption breaks on the cloud 5090

On the laptop one core keeps the GPU fed, so descent never needed to be parallel.
**Laptop-specific.** On a 5090: faster inference ⇒ higher leaves/sec demand ⇒
single-core descent starves the GPU; and cloud boxes give 16–32 vCPUs to exploit.
So the real target is a model where **descent and solving both overlap the GPU**
and the scheduler allocates cores dynamically — Step 2.

### The dynamic-allocation principle

> Never statically assign threads to roles. Put all CPU work (descent tasks +
> within-solve YBW child tasks) into **one work-stealing pool** and let it
> balance. Rayon's global pool already does this — the within-solve `par_iter`
> tasks simply become more items in the shared pool, so no manual solve-vs-gen
> budget is ever needed.

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
(pure Rust, GIL already released).

- **Descent and solving both overlap the GPU** — while a worker waits on the
  evaluator, others do CPU work.
- **Dynamic allocation for free** — workers in endgames solve, workers in normal
  play descend; the OS scheduler + the shared Rayon pool balance. No budget.
- **Scales to the 5090 + many vCPUs.**

### Step 2 scope

1. Adopt the multi-thread `RustMCTS` + coalescing-evaluator path for the training
   driver (today: synchronous `BatchedMCTS` `step`/`update`).
2. Run the (within-solve-parallel) endgame solver inline in the worker loop.
3. Wire self-play data collection / finished-game handling into the worker model.
4. **Bound total solve parallelism via the single shared Rayon pool** — do *not*
   let every worker launch an independent full-machine YBW solve (that is the
   oversubscription failure at cloud scale). Let YBW child-tasks and descent
   tasks contend in one pool so work-stealing arbitrates.

### Step 2 gate

- CPU utilization **during the GPU-eval window** rises toward saturating spare
  cores (proves overlap).
- Endgame fallback rate stays low (the per-solve deadline still met under cloud
  oversubscription — re-check with the same instrumentation).
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
2. **Step 1.5 (overlap)** recovers the GPU fill the working solver now costs on
   the laptop; optional if going straight to Step 2.
3. **Step 2** is the cloud-5090 answer for dynamic all-core use; reuse the
   milestone-6 coalescing path and keep all CPU work in one Rayon pool.
