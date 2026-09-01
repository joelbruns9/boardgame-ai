# Throughput levers

A companion to `RENTING_A_BOX.md`. That one is about not wasting money on a
launch. This one is about not wasting it on tuning.

Every rule here was paid for. One run lost roughly a day of rented box to a
throughput sweep that measured a configuration the run never ran, and then to a
"tuned" relaunch that came out 16% **slower** than the placeholder it replaced.
None of it was a hard bug. Every step looked reasonable at the time.

---

## 1. The one distinction that matters

Throughput levers fall into four classes. **Almost every expensive mistake is
treating a lever as though it belongs to a different class than it does.**

| class | changes what the net learns? | how to decide it |
|---|---|---|
| **A. Geometry** | no | wall clock alone |
| **B. Boundary** | no | wall clock alone, but it is a code change |
| **C. Target-changing** | **yes** | strength per wall-clock hour, never games/s |
| **D. Workload** | **yes, indirectly** | how much work exists, before any timing |

Write down which class a lever is in *before* you measure it. A class C lever
evaluated on games/s will always look either free or catastrophic, and both
readings are meaningless.

### A. Geometry — free, sweep first

Concurrent games in flight, batch caps, in-flight batch depth, scheduler shards,
the evaluation-side equivalents, packing threads.

These change *when* work happens, never *what* work happens. Two configurations
must produce identical games from identical seeds. **Verify that** — fingerprint
the discrete trajectory (actions, digests, visit counts), never the float
targets, which legitimately drift by ~1e-5 when batch shape changes the order of
float reductions.

This is the only class you can sweep on wall clock with no strength question
attached, and it should be the first thing you touch on the box.

### B. Boundary — where the real wins have actually been

Encoding, packing, tensor construction, the language boundary, result scatter;
per-row work that should be vectorised, per-call work that should be per-batch.

**Historically this is where all the speed came from, and it is never where
anyone looks first.** One project's generation throughput went 1,278 → 4,760
games/hour. The plan had been built around scheduler concurrency and *none* of
the gain came from there — it was a fused embedder, a vectorised gather, and one
API-shape fix. A per-row `float()` loop inside the gather was worth 1.34× by
itself, and no timer category contained it.

Boundary work is code, so it cannot be swept. A/B it on the real generation
path, on a laptop, before renting anything.

### C. Target-changing — price in strength, not speed

Leaf batch and virtual loss at the root, simulation counts, the cheap/full
search split, exact-solver budgets, chance sampling width.

These change the policy or value target. Two settings at *equal games/s are not
equal*, and a setting that costs 20% throughput can still be the right one.

The honest comparison is **equal wall clock, different setting, compare
strength** — a head-to-head match or a gate win rate after a fixed number of
hours. Anything else measures the wrong quantity precisely.

Prefer a paired design: put both arms in the same game where the domain allows
it (per-seat settings, identical deals, seat-swapped). That resolves the effect
with far fewer games than holding a fixed reference opponent and paying for its
noise twice.

### D. Workload — the class nobody remembers exists

A lever that changes **how much work the system decides to do**. These are the
most dangerous, because they are usually spelled like a safety cap.

The canonical example, and the one that cost the most: an exact-solver node
budget with a fitted cost model installed. It reads as "spend at most N nodes
per solve." It is actually the **admission threshold**:

```
attempt iff predicted_cost + margin <= log10(budget)
```

Doubling it does three things at once. It doubles the cost of every solve that
declines; it admits a band of new positions that are by construction the most
expensive ones the model knows about; and it lands that band exactly where the
fit is least reliable. Cost is convex in the knob while yield is concave. There
is no safe way to "creep it up while watching throughput" — you find the cliff,
not the optimum.

**Test for class D:** does the knob appear anywhere in an eligibility,
affordability, or admission decision? If yes it sets workload, not just a
ceiling.

---

## 2. Vocabulary, and why it drifts between projects

**None of these words are standard, and several of them changed meaning between
two games in this same repository.** That drift is not cosmetic: it is directly
how a per-shard setting got typed as though it were a total.

Read this section before quoting any number at anyone.

### 2.1 Units of search work

- **simulation** — one root-to-leaf descent. The budget knob (`--full-sims`,
  `--cheap-sims`) counts these.
- **leaf** — a search node that needs a *network evaluation*. **Not every
  simulation produces one.** A descent that ends on a terminal node is scored by
  the rules, and a descent through an already-expanded subtree produces nothing
  new. So `simulations > leaves > GPU rows`, sometimes by a lot — which is why a
  late-game position with a 1,600-simulation budget can cost the GPU almost
  nothing.
- **leaf batch** — leaves evaluated per search step **within one game**. In 7WD
  it is *per seat*; the full and cheap paths carry separate values.
- **wave width** — leaves actually in flight for **one game**. Bounded above by
  leaf batch, and collapses toward 1 when a collision-avoidance scheme is on
  under a concentrating root.
- **batch width / rows** — leaves summed **across games** in one evaluator call.
  This is the only one of the four the GPU ever sees.

Confusing the last two is how a diagnostic improvement gets reported as a
throughput win.

### 2.2 Units of parallelism

- **slot** — one active game the scheduler is holding, including its tree arena.
  Slots cost memory, so they are not a free dial.
- **shard**, also called a **scheduler worker** — one OS thread running its own
  scheduler loop over a chunk of the games. Each shard assembles its **own**
  batches; unless something merges them, shards fragment batch width rather than
  pooling it (§4.4).
- **inflight batches** — how many evaluator calls a shard may have outstanding
  before it blocks.
- **global batch cap** — the maximum rows in one evaluator call. A ceiling, not
  a target; observed batch width is usually far below it.
- **rayon** — a work-stealing thread-pool library with a process-wide **global**
  pool. Anything using `par_iter` without a dedicated pool is silently taking
  every visible CPU. `RAYON_NUM_THREADS` sweeps the global pool without a
  rebuild, which makes it a cheap axis to test.

### 2.3 The same word, different scope

This table is the actual lesson. Both columns are correct *for their project*.

| term | Kingdomino | 7 Wonders Duel |
|---|---|---|
| **solver threads** | a **total**; one dedicated rayon pool; `0` = auto = half of available threads | **per shard**; a hand-rolled pool per shard, so the real count is `value × shards` |
| **generation threading** | rayon's **global** pool | hand-spawned `std::thread` shards; rayon is used only for packing and derive |
| **`--workers` / `--process-workers`** | — | serve the **Python** generation path and the seed buffer. On a `--generation-backend rust` run they never touch generation at all, because that branch returns first. `--process-workers` additionally sizes the Python gate's wave |
| **slots** | per scheduler | a **global** budget divided across shards |

So "twelve solver threads" means twelve OS threads in Kingdomino and
`12 × shards` in 7WD. Copying a value across projects is copying a number whose
denominator changed.

Three habits that make this survivable:

- **Whenever a value is "per" something, print the product** — and print it
  against *physical cores*, not visible CPUs (§3.2, §4.6).
- **Print every setting the operator did not pass.** A knob that looks like the
  parallelism control but is inert on the active backend is worse than an absent
  one, because it reads as configured.
- **Name the scope in the flag itself** where you can. `--solver-threads-total`
  alongside `--solver-threads` costs nothing and removes the whole failure mode.

---

## 3. Before you sweep anything

This is the section that would have saved the money.

### 3.1 The sweep must run the thing the run runs

A sweep harness constructs the loop itself, which means it can silently omit
whatever the production entry point configures. In the case that cost the most,
the sweep called the thread-pool sizer for the exact solver but never the
function that installs its node budget — and the budget defaults to zero, which
disables solving entirely. **Every point was measured with the solver switched
off**, on a run where the solver takes 22-37% of generation wall.

The sweep dutifully printed `solver threads: 6 per shard x 2 shards = 12 total`.
Twelve threads were spawned. They sat blocked on an empty queue.

Requirements for any sweep harness:

- **Build its config from the run's manifest**, not from dataclass defaults.
  Sweeping one search algorithm at 24/128 sims against a run that plays another
  at 100/1600 finds an optimum belonging to a machine nobody is running.
- **Assert every subsystem is live**, not merely configured. Print a count of
  work each one actually did — solves attempted, bot games played, league games
  routed — and refuse to report a result when any of them is zero.
- **Give each point more jobs than slots.** A benchmark that chunks seeds into
  exactly `slots` jobs per call cannot refill, and measures nothing.
- **Sweep the axes jointly.** They interact: a batch cap binds as concurrency
  rises, so sweeping concurrency alone finds a ceiling that belongs to the cap.
- **Hold the other classes constant across an axis.** If solver load moves with
  the worker count, then the worker axis is measuring the solver.
- **Run more than one point.** A grid that resolves to a single configuration
  reports `1.00x` against itself. That is not a measurement, and it is
  surprisingly easy not to notice.

### 3.2 Know which resource binds, and use the right number for it

Only a lever that relieves the binding resource can help. Everything else
rearranges work.

- **Count physical cores, not logical threads.** `os.cpu_count()`, `nproc`, and
  most oversubscription warnings report SMT threads. For branchy,
  memory-latency-bound work — tree search, alpha-beta, encoding — a second
  thread on a busy core buys 10-30%, not 100%, so sizing to the logical count is
  a silent 2× oversubscription. Settle it once:
  `lscpu | grep -E 'Core|Socket|Thread'`; cores per socket × sockets is the real
  number.
- **Distrust every GPU utilisation figure until you have read how it is
  sampled.** NVML's `utilization.gpu` means "was any kernel resident," not
  occupancy — one tiny kernel reads 100%. If the sampler fires at phase
  boundaries and averages a handful of point samples, the number describes the
  least representative instants in the run. And a timer wrapped around an
  asynchronous enqueue measures the enqueue, not the device.
- **The honest measure is the recorded time breakdown**, not a utilisation
  gauge: time inside the forward against total scheduler wall, averaged over a
  whole iteration.

### 3.3 Read what the run already writes

Before building instrumentation, check what is already recorded and never
surfaced. Per-shard time breakdowns, batch widths, wave widths, blocking time
and slot occupancy had all been written to the structured log for weeks while
the same questions were being answered by inference.

And when you do lean on a recorded metric, **check that it survives
aggregation.** One blocking-time counter was incremented correctly, exported
correctly, and omitted from the merge that combines per-shard metrics — so it
read zero in exactly the multi-shard configuration where it mattered.

---

### 3.4 Rules for reading a number once you have one

- **Discard the first iteration after any restart.** It is systematically slow —
  cold caches, cold allocator, cold page cache. Measured here: the first
  iteration after one relaunch came in 15% below that run's own steady state,
  and after another, 8% below. Comparing a cold first iteration against a
  warm mean is how a working change gets reverted.
- **Reverse the order of points on alternate repetitions.** Otherwise thermal
  drift and noisy neighbours load entirely onto whichever point runs last. On
  rented hardware you are sharing a host with someone you cannot see.
- **One point is never a measurement** (§3.1). Neither is one iteration.
- **Report the quantity the lever moves *and* the quantity you care about.**
  Batch width and games per hour are different numbers, and only one of them
  pays for the box.

---

## 4. The lever catalogue

Ordered by historical return, not by how interesting they are.

### 4.1 Port the hot path to a compiled language (class B, largest of all)

The single biggest throughput change in this repository's history was not a
knob. Kingdomino's Python search was rewritten in Rust and produced roughly
**28× the leaves per second**. Nothing in the sweepable classes is within an
order of magnitude of that.

It is also the least reversible thing on this list, so treat it as a port with a
correctness obligation rather than an optimisation:

- **Keep the reference implementation alive and diffable.** Both games retain
  `--generation-backend`, `--gate-backend` and `--derive-backend` switches
  between `rust` and `python` precisely so the two can be compared on demand.
  The Python side stops being fast and becomes the oracle.
- **Gate on parity, and re-run the gate when the code moves.** A resume across a
  code change warns to re-run the Rust/Python derive parity gate for exactly
  this reason: a buffer whose rows span two engines is only safe if you have
  checked the engines agree.
- **Fingerprint discrete outputs, never floats.** Assert on actions, digests and
  visit counts. Targets legitimately drift ~1e-5 across implementations, so
  "byte-identical records" is a claim that will fail for the wrong reason.

Four gotchas that transferred verbatim between the two ports and will transfer
again:

- **Match the float width exactly.** `f64` where Python uses `f64`. An `f32`
  port produces plausible, subtly different targets and passes every test that
  is not a parity test.
- **Cross the boundary in bytes, not objects.** Per-row Python object
  construction at the boundary eats the gain you just bought.
- **Release the GIL around the compiled work**, and re-acquire only where you
  must touch Python. Scheduling runs GIL-detached and only the evaluator worker
  attaches; a port that holds the GIL across the hot path gets none of the
  parallelism it was written for. The attach/detach API has also changed
  across pyo3 versions — pin the version and read its migration notes.
- **A compiled hot path has a minimum useful batch.** Below a certain leaf
  batch, per-call overhead dominates and the port can measure *slower* than what
  it replaced.

**And expect the old knobs to go inert without going away.** Once generation is
compiled, the Python-side parallelism flags stop touching it, but they remain in
the CLI, in the launcher, and in everyone's mental model — see §2.3. Print the
settings the operator did not pass, and say which backend is actually live.

### 4.2 Vectorise the boundary (class B)

Any per-row Python or per-row allocation in the encode/pack/gather path scales
with rows while per-batch cost stays flat, so it dominates precisely when
batching starts working. Look for `float()`, `softmax`, dict construction or
tensor allocation inside a row loop.

### 4.3 Make heterogeneous work share one call (class B)

When a fraction of games use a different opponent, bot, or network, check
whether that fraction forces its own dispatch. The recurring shape is:

> **15% of games, 32% of time.**

That ratio has now appeared twice on the same project for two unrelated reasons.
First because per-*call* bot assignment forced one scheduler call per
`(bot type, seat)` — up to eight pools of two or three games each. Later because
a routed evaluator splits every batch by network id and runs one forward per
network, so league play at 15% cost 31% of generation throughput.

Fix the first shape by making the assignment per-game. The second needs a
per-row network embedding or a separate stream; splitting the batch is the
expensive part, not the extra rows.

### 4.4 Coalesce across schedulers before adding schedulers (class A/B)

If the evaluator does not merge requests from concurrent scheduler shards, then
adding shards **fragments batches instead of pooling them**. Each shard
assembles only from its own share of a global slot budget, and the shards
serialise through the evaluator one small batch at a time. Four shards at fixed
slots is a quarter of the batch width, not four times the throughput.

Check this before sweeping the shard axis, because it changes what the axis
means.

### 4.5 Understand what concurrency can and cannot buy (class A)

More games in flight **rebundles** work; it does not reduce work. CPU cost per
leaf is constant, so doubling concurrency doubles both the batch width and the
CPU needed to fill it — the same leaves per second, in fewer larger calls.

The only thing recovered is the **fixed per-call** component. So the value of
every batching lever is set entirely by how large fixed per-call cost is
relative to per-row work. Measure that ratio once: it bounds this whole family
of levers before you spend a day on any of them.

Corollary: **filling the GPU is not the goal.** Leaves per second is. A wider
batch that takes proportionally longer to assemble is better telemetry and
identical throughput.

### 4.6 Split fixed CPU deliberately (class A, zero-sum)

When two consumers compete for a fixed core count — game generation and a
background solver, say — there is no "spare" capacity to discover. The scheduler
will absorb every core you give it. The split is a **trade you choose**, not a
remainder you measure.

Two rules:

- **A per-unit value multiplied across a boundary is the single most common
  defect class.** Threads *per shard* times shards. Leaf batch *per seat*. Wave
  width *per game* against batch width *per shard*. Whenever a number is "per"
  something, print the product and compare it to physical cores.
- **More worker threads against a fixed queue of work buys latency, not
  volume.** If the pool already drains as fast as work is dispatched, extra
  threads block on an empty queue while the cores they took are gone from
  generation. The signal that a pool is keeping up is the absence of
  deadline-driven timeouts, not its utilisation.

### 4.7 Batching inside the search (class C, needs a strength A/B)

Evaluating several leaves per search step widens batches, but it changes the
search: under a root whose visit distribution *is* the policy target, virtual
loss changes that target. This is not a throughput knob in disguise; it is a
different algorithm at the root.

Two separate cautions:

- **Batch width is not throughput.** A change measuring 29 → 39 rows per call is
  +34% on a diagnostic and can convert to nothing at all on a box where per-call
  cost is not what binds. Report both numbers or you will bank a win that does
  not exist.
- **The mechanisms can be mutually exclusive.** A collision-avoidance scheme
  that *forbids* collisions and virtual loss that merely *discourages* them
  cannot both be on: under a concentrating root, forbidding collapses the wave
  to width ~1 and silently disables the batching you are trying to measure. A
  null from a configuration with half the mechanism switched off is not a null.

---

### 4.8 Model and inference cost (mostly class C)

Precision, `torch.compile`, width, depth, head count, readout shape. These are
the levers people reach for first and they are usually the wrong ones, because
**they only pay in the regime where inference actually binds.**

On a generation loop that is CPU-bound — the common case once the hot path is
compiled and the boundary is vectorised — making the network faster buys almost
nothing, because the device was already idle waiting for leaves. Making it
smaller buys nothing *and* costs capacity. Check §4.5's ratio before spending
any time here.

Two notes on classification:

- **Capacity (width, depth, heads) is class C, and not subtly.** It changes what
  the network can represent. It is a run-design decision, not a throughput one.
- **Precision is class C too, despite looking like a free speedup.** Reduced
  precision changes numerics, and whether that costs strength is an empirical
  question — so it deserves a paired strength arena rather than a wall-clock
  comparison. Treating it as free is the same category error as treating leaf
  batch as free.

`torch.compile` is the exception: genuinely class B, no target change, and worth
having on. Just remember it wraps the module, so anything reading the model's
`state_dict` has to unwrap it first.

---

## 5. The rest of the iteration

Generation is not the iteration. An iteration is generation **plus training,
derive, and evaluation**, and optimising only the first one caps how much you
can possibly win.

### 5.1 Evaluation is a second budget, and it grows

Once training stops being the bottleneck, evaluation usually becomes it. Decide
the acceptable evaluation share of wall clock **before** the run, not when the
gate ladder surprises you: a laddered gate at 200/600/1000/1500 games implies
roughly **10/25/35/45%** of a twenty-minute iteration as the rungs step up. The
share is not constant — it rises as the run matures and the gate needs more
games to resolve smaller differences.

Three things follow:

- **Evaluation has its own geometry knobs, and they are not the generation
  ones.** They are also worth sweeping separately: measured here, gate
  concurrency of 96 / 144 / 256 gave 1.181 / 1.093 / 1.748 games per second —
  **+60% from a pure class A change**, larger than anything the generation sweep
  found, with the batch cap making no difference at all (1.746 against 1.748).
- **Evaluation is usually a different cost regime from generation.** Gate games
  here run at a fraction of the search budget with the expensive extras off,
  which is why they are ~5× faster per game. Do not assume a setting that helps
  generation helps the gate, or the reverse.
- **A bigger gate is not waste; it is resolution.** Two consecutive inconclusive
  200-game gates here stepped the ladder to 600 and immediately resolved to a
  promotion. The point estimate had been above the threshold the whole time and
  the interval simply could not prove it. Budgeting evaluation too tightly buys
  wall clock with decisions you can no longer make.

### 5.2 Do not recompute what cannot have changed

Replay buffers are usually immutable once written. Re-deriving the whole window
every iteration therefore re-verifies data that could not have changed —
measured at 223s over 1,633 games and 404s over 4,800, against 11s of actual
training.

Cache the derived artifacts, and then:

- **Size the cache to the working set, from a formula, not a default.** The
  window here is `replay_window_games × examples_per_game`; the cache default
  covered 62% of it, so 38% was re-derived every iteration forever.
- **Make the shortfall loud.** A one-line warning naming both numbers is what
  turned this from invisible into a two-minute diagnosis.
- **Know the cache's accounting unit.** If the ceiling is denominated in a
  nominal per-item size that differs from the real one, the configured number
  and the memory it costs are not the same quantity.
- **Then check whether it matters.** This one was worth ~24 seconds against a
  ~3,100-second iteration. Correct, and not a speedup. Fix it because it is the
  right value, not because it buys time.

---

## 6. Order of operations

**On a laptop, before renting:**

1. Read the existing time breakdown. Decide which resource binds, and note
   the share going to generation against training, derive and evaluation (§5).
2. Measure the fixed-per-call against per-row cost ratio. It bounds every
   batching lever you might consider.
3. Do the class B work — the port first if the hot path is still interpreted
   (§4.1), then the boundary. It has historically returned more than everything
   else combined, and it costs no rental.
4. Build the sweep harness and **prove it runs the real configuration** — every
   subsystem live, config from the manifest, more jobs than slots, at least two
   points.
5. Classify every remaining lever A/B/C/D, and write the classification down.

**On the box:**

6. Sweep class A jointly, with fingerprints confirming the games did not change.
   **Sweep evaluation's geometry as its own grid** (§5.1) — it is a different
   cost regime and has historically returned more than the generation grid.
7. Re-measure any class B result that was A/B'd elsewhere. **Benchmark figures
   only partly transfer**: 1.99× on a microbenchmark became 1.89× on the real
   path, and +48% for a concurrency step became +21%. Same cause each time — the
   earlier fixes removed the fixed cost that concurrency had been hiding.
8. Leave class C alone unless you are prepared to pay for a strength
   measurement. If you are, run it as equal-wall-clock arms.
9. Never change class D to "use up headroom." Headroom measured at one workload
   says nothing about the cost at another.

---

## 7. Cost discipline

- **Convert every proposed optimisation into days saved before building it.** A
  35% gain on a 7.6-day run is 2.6 days. An unmeasured strength risk on the same
  run may be worth more or less than that, and only saying both numbers out loud
  makes the trade visible.
- **A tuning change is not free even when it is correct.** Restarting to apply
  it costs the in-flight iteration plus the chance of a launch failure. Batch
  tuning changes into a single restart, and validate the assembled command
  before detaching.
- **Late in a run, throughput usually stops being the binding constraint on the
  outcome.** Once improvement per unit of data is decaying, generating games
  faster reaches the same ceiling sooner — it does not raise it. Decide whether
  you are buying a better model or the same model earlier, and price it
  accordingly.
- **Do not tune a lever you cannot change on a resume.** Find the run-identity
  list first (`RENTING_A_BOX.md` §1.7 and §8). A frozen knob is a next-run
  decision, and measuring it now is work banked for later at best.

---

## 8. The one-line version

**Classify the lever, prove the harness runs the real thing, and measure the
quantity the lever actually moves** — which for half of them is not throughput
at all.
