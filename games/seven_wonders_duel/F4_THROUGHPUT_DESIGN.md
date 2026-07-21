# Phase F4 — batched inference bridge + throughput (design & plan)

Working design doc for F4, the Phase F exit. `PHASE_F.md` keeps the milestone
summary + running log; this holds the detailed plan. Charter: `AZ_PROJECT_PLAN.md`
§8.4.

## 1. Goal / exit gate

**≥20× self-play throughput vs the Python loop at equal settings** (same net,
sims, top_k; Kingdomino achieved ~28×). "Throughput" = games/s (primary) and
leaves/s (diagnostic). This is the whole point of the Rust port: make self-play
fast enough to actually train.

## 2. Why F4 is different from F1–F3 (the central tension)

F1–F3 were **bit-exact ports** — every step gated identical to Python. F4 is a
**performance** step, and its fast path is **deliberately not bit-identical** to
the sequential reference:

- The F3.3 searcher is a *sequential* Gumbel root: sequential halving simulates
  candidates in rounds, and each simulation's backprop changes the tree the next
  simulation sees (PUCT selection deeper down, and completed-Q candidate
  reduction between rounds).
- Batching leaves for the NN means running several descents **before** their
  backprops land, so a batched searcher explores a slightly different tree.
  (Standard parallel-MCTS coalescing, typically with *virtual loss* to diversify
  descents within a batch.)

So F4 cannot be validated by an equivalence gate. It needs a different strategy
(§4). The **sequential F3.3 searcher stays the correctness oracle.**

### 2a. Virtual loss works with Gumbel (resolved 2026-07-21)

Earlier drafts overcomplicated this. **Virtual-loss coalescing works with the
Gumbel root**, the same as KD's PUCT MCTS. The Gumbel root only decides the
*schedule* (which candidate gets how many sims, when to halve); every simulation
still descends **PUCT below the root**, which is exactly where virtual loss
operates. The schedule is orthogonal to how the sims are executed.

"Intra-round backprops": a sequential-halving *round* runs `per_action` sims per
surviving candidate. Sequentially, each sim backprops (updates visits/values)
before the next, so later sims see the update. Coalescing collects several leaves
before evaluating any, so those backprops are **deferred** — the only source of
non-identity, and virtual loss is the standard fix (deferred descents diversify
instead of piling on one path).

**The one Gumbel-specific rule:** flush all pending backprops at **round
boundaries** (the halving reads the round's completed-Q). Coalescing runs freely
within a round; flush before each reduction. Round batch sizes are healthy
(candidates × per_action). So the design is simply **KD's virtual-loss +
`leaf_batch` coalescing, flushed at round boundaries** — no novel scheme needed.
Tune `leaf_batch` by the KD method (sweep, find where quality declines — KD's
answer was 6; 7WD's may differ, esp. on a big GPU).

### 2b. Both batching levers are required (KD empirics, 2026-07-21)

Kingdomino ran the same class of GPU (RTX 5090) and its measured recipe to keep it
**relatively busy** was **~32 concurrent games AND virtual loss (`leaf_batch≈6`)**
— *both together*. Cross-game concurrency alone did **not** fill the GPU. So for
7WD (same GPU, same box):

- **Virtual-loss intra-search coalescing is required, not optional** (`leaf_batch`
  ≈ 6, re-tune by the KD sweep). It multiplies each game's per-eval batch.
- **Cross-game batching is via IN-PROCESS coalescing**, not a separate server
  process: ~32 game searches multiplexed **in one process on 1–2 cores**, an
  in-process coalescer accumulating their eval requests, GIL released
  (`py.detach`) during the GPU forward so Rust tree work overlaps inference. (KD
  M6: in-process coalescing + `py.detach`, `lb=6` → mean_batch ~45, ~6.1k
  evals/s.)
- **Cores are scarce:** most of the box is reserved for the **exact endgame
  solver** (7WD's Phase H analog — exact relabeling / tail solve), so self-play
  must be **core-frugal by design** — hence game-multiplexing on 1–2 cores, not a
  process per core.
- **No rayon** for game generation (its work-stealing overhead was too high at
  game-step granularity). Use threads + channels or a single-threaded cooperative
  coalescing loop.

**F3.4's `PyEval` still fits** — its adapter becomes "submit to the in-process
coalescer and wait." But the searcher **does** need the §3a phase split to collect
`leaf_batch` leaves via virtual loss and yield to the coalescer, so that work is
core F4, not deferred.

## 3. System architecture (in-process coalescing — KD M6 model)

```
 one self-play process, 1-2 cores:
   game 0 search --lb=6 leaves--> \
   game 1 search --lb=6 leaves-->  in-process coalescer --> GPU net (py.detach):
   ... (~32 games multiplexed)      accumulate across games    big batch, forward,
   game 31 search --lb=6 leaves--> /  ~45 leaves/batch          scatter replies

 remaining cores: exact endgame solver (separate workload)
```

- **~32 concurrent game searches** multiplexed on **1–2 cores** (cooperative loop
  or threads + channels — **not rayon**, not a process per game). Each runs the
  §3a phase-split searcher with virtual loss (`leaf_batch≈6`).
- **In-process coalescer** accumulates the games' eval requests, releases the GIL
  (`py.detach`) and runs one GPU forward, then scatters replies back to the games.
- **F3.4 `PyEval` adapter** points at "submit to the coalescer and wait" instead
  of a local net.
- **Exact endgame solver** runs on the remaining cores — self-play stays within
  its 1–2-core budget.

## 3a. Phase-split searcher (core — enables virtual-loss coalescing + the oracle)

Refactor the searcher into three phases so a game can collect `leaf_batch` leaves
per eval (virtual loss) and yield to the coalescer, and so the `leaf_batch=1`
oracle gate (§4b) is well defined:

1. **Selection / materialization** — descend from the root (forced edge at the
   root per the Gumbel schedule; PUCT below), sampling chance and materializing
   children, until reaching a leaf that needs evaluation. Apply virtual loss on
   the path if using (B). Record the path + the pending leaf.
2. **Pending-leaf evaluation** — one batched `Eval` call over all collected
   leaves (GIL acquired only here).
3. **Backpropagation** — expand each leaf with its (value, priors) and backprop
   along its recorded path (removing virtual loss if applied).

Supporting perf work (also from review):
- **Batched `Eval` interface** — `evaluate_batch(&[&GameState]) -> Vec<…>`; the
  scalar `evaluate` becomes `evaluate_batch` of length 1.
- **GIL release** — `py.detach` (KD M6) during tree work; reacquire only for the
  batched eval. The F3.4 `Python::attach` already reattaches correctly after a
  detach.
- **Flat encode buffers** — encode into reusable numeric buffers, not per-leaf
  Python `Token`/`Encoding` objects (kills allocation + the F2 `PaymentContext`
  cost matters here — see F2 deferral).
- **Precomputed actor/legal** — pass node-cached actor/legal to the evaluator
  instead of recomputing in `PyEval`.
- **Batch forced children + keep priors** — force-expansion currently evaluates
  children then discards priors, so the first visit re-evaluates (F3.3 note);
  materialize all forced children, batch-evaluate once, and store priors.

### 3a. Python-hop vs in-process NN

Default: **batched Python hop** (Rust encodes → one `evaluate_batch` Python call →
torch forward on a batch tensor). KD hit ~28× this way. Only if profiles show the
Python/torch hop still dominates after coalescing do we consider **tch/ONNX
in-process** (no Python hop) — a bigger lift, deferred unless needed.

## 4. Validation strategy

### 4a. Phase-split `leaf_batch=1` == the sequential oracle, exactly
The phase-split searcher at batch size 1 (no coalescing / virtual loss) MUST
reproduce the F3.3 `search_closed` result + tree digest to 1e-9. This proves the
refactor is correct before virtual loss changes behavior — the one hard gate F4
keeps, and the reason the phase split must support batch=1.

### 4b. `leaf_batch>1` (virtual loss) validated statistically + on quality
Not exact (deferred backprops). Validate:
- chosen-action agreement rate + root-value/policy agreement vs the sequential
  searcher over many positions;
- **trap-suite blunder rate** on the E-Tier-1 consequential fixtures (`35:63`,
  `107:60`) — must not blunder more than sequential;
- **sweep `leaf_batch`** to find where quality declines (KD's method → 6; re-tune
  for 7WD). This is the required-lever knob, not optional.

### 4c. In-process coalescer batch invariance
A batched GPU forward returns the same per-request values (to fp tolerance) as
evaluating each request alone — batching concurrent games together doesn't change
any single game's evaluations. (Quality only comes from virtual loss, §4b, not
from coalescing.)

### 4d. Harness correctness
The ~32 concurrent games produce complete, valid, replayable games; buffer schema
intact; and self-play stays within its 1–2-core budget (the exact solver keeps the
rest).

### 4e. Throughput benchmark (§5).

## 5. Benchmark methodology (the ≥20× gate)

- **Baseline:** the Python self-play loop (`phase_d`/`self_play` path) driving the
  Python `GumbelMCTS`, in its best config on the same box (it is GIL-limited —
  KD's threaded generation saturated ~4×).
- **Candidate:** the Rust in-process coalescing self-play (~32 concurrent games,
  virtual loss, 1–2 cores) on the same net/settings/box.
- **Equal settings:** identical sims, top_k, net checkpoint, closed mode, same
  hardware (the target cloud box: multi-core + the training GPU).
- **Metric:** **aggregate games/s** across all workers (primary); leaves/s and
  **GPU utilization %** (diagnostics — low GPU util despite max workers is the
  trigger for F4.4 intra-worker coalescing). Report all three + the component
  breakdown (encode / IPC / GPU forward / tree).
- **Gate:** ≥20× aggregate games/s vs the Python loop at equal settings. Record
  the number + box spec in `PHASE_F.md` (KD's ~28× is the yardstick).

## 6. Sub-sequence (revised 2026-07-21 — KD in-process model)

- **F4.0** — **phase-split searcher** (§3a: selection → collect leaf → eval →
  backprop, path recording; likely an arena to record paths safely). Gate:
  `leaf_batch=1` reproduces the F3.3 digest exactly (§4a).
- **F4.1** — **virtual loss + in-process coalescer** across ~32 concurrent game
  searches on 1–2 cores (threads/channels or a cooperative loop — no rayon),
  `py.detach` around the GPU forward, `PyEval` adapter → coalescer-submit. Gates:
  coalescer batch invariance (§4c), statistical + trap-blunder validation and the
  `leaf_batch` sweep (§4b), harness correctness (§4d).
- **F4.2** — **throughput benchmark** (§5): aggregate games/s + GPU utilization +
  component breakdown (encode / coalesce / GPU forward / tree) vs the Python loop.
  Measure first, then decide (agreed).
- **F4.3** — **perf tuning** as profiles dictate: flat encode buffers, precomputed
  actor/legal, batched forced children (keep priors, kill the first-visit
  re-eval), concurrency count, coalescer window, 1-vs-2-core split. Iterate to
  ≥20× with headroom.
- **F4.4** *(conditional)* — **tch/ONNX in-process** only if the Python/torch hop
  still dominates after coalescing.

Throughout: self-play stays within a **1–2-core budget**; the rest of the box is
the exact endgame solver's.

## 7. Open decisions / constraints (log here)

1. **In-process coalescing on 1–2 cores, ~32 concurrent games, virtual loss
   `lb≈6`** (2026-07-21, user — KD's measured recipe to keep an RTX 5090 busy on
   the same box). Both levers required; virtual loss is not optional. **No rayon**
   (work-stealing overhead too high at game-step granularity).
2. **Core budget:** most cores reserved for the exact endgame solver → self-play
   is core-frugal by design.
3. **`leaf_batch`:** start ≈6 (KD), re-tune by the quality-decline sweep (§4b).
4. **Python-hop vs tch/ONNX in-process:** measure full-loop throughput +
   components first (agreed), decide at F4.4.

## 8. Reuse from Kingdomino
KD M5/M6: `allow_threads`/`py.detach`, in-process coalescing, `leaf_batch>1`
essential (single-descent per NN call caps at ~1.7k leaves/s; batching → ~46k).
Port the coalescing scaffolding; the Gumbel schedule (§2a) is the new part.
