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

### 2b. Two batching levers — cross-worker is the primary one (2026-07-21)

Training runs on a rented multi-core box + a powerful GPU, **multi-process from
the start**. That makes the dominant batching lever **cross-worker**, not
intra-worker:

- **Cross-worker batching (a central inference server) is quality-free.** Each
  self-play worker's search is fully independent, so batching their eval requests
  together has **zero** effect on any single search — no virtual-loss
  cross-contamination. A powerful GPU is filled by *many workers*, not by
  degrading each search.
- **Intra-worker virtual-loss coalescing trades a little quality for batch size.**
  It is the *secondary* lever, needed only if worker count can't fill the GPU.

So on the target box: **N worker processes at `leaf_batch=1`** (max quality) +
**one inference server** coalescing across them into big GPU batches. Intra-worker
virtual-loss coalescing is an optional add-on.

**F3.4's `PyEval` already fits this**: its Python adapter can point at
"submit-to-server-and-wait" instead of "run local net." A worker = the F3.4 Rust
searcher + a server-submit adapter — no new searcher code for the multi-process
path.

## 3. System architecture (multi-process + inference server — primary)

```
 worker 0 (proc): Rust searcher --submit(leaf)--> \
 worker 1 (proc): Rust searcher --submit(leaf)-->  inference server (GPU net):
 ...                                              coalesce across workers over a
 worker N (proc): Rust searcher --submit(leaf)--> /  short window -> big batch ->
                                                     forward -> dispatch replies
```

- **Workers** are processes (sidestep the GIL). Each runs the F3.4 Rust searcher;
  its `PyEval` adapter is a **server-submit** call (`submit(encoded_leaf) ->
  (value, priors)`) instead of a local net. `leaf_batch=1` (max quality).
- **Inference server** owns the GPU net, collects requests from all workers over
  a small time/size window, batches them, runs one forward, and returns per-worker
  replies. This is the quality-free batch lever (§2b).
- Reuse the KD **inference-service design** (§2b of `AZ_PROJECT_PLAN.md` flagged it
  as an extract-now item).

## 3a. Intra-worker phase split (secondary — for optional coalescing + the oracle)

Refactor the searcher into three phases so a *single worker* can also batch its
own leaves (virtual-loss coalescing, the secondary lever) and — critically — so
the `leaf_batch=1` oracle gate (§4.1) is well defined:

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

### 4a. Multi-process path — already searcher-equivalent
Each worker runs the **existing F3.4 `closed_search_net`** at `leaf_batch=1`, which
is already gated equal-to-Python. So the primary path needs **no new searcher
equivalence work**. New checks:
- **Inference-server batch invariance:** a batched forward returns the same
  per-request values (to fp tolerance) as evaluating each request alone — i.e.
  batching workers together doesn't change any worker's evaluations.
- **Harness correctness:** workers produce complete, valid, replayable games; the
  buffer schema is intact.

### 4b. Intra-worker coalescing (optional, only if built)
- **`leaf_batch=1` == the sequential oracle, exactly.** The phase-split searcher
  at batch size 1 (no coalescing/virtual loss) MUST reproduce the F3.3
  `search_closed` result + tree digest to 1e-9 — proves the refactor before any
  batching changes behavior.
- **`leaf_batch>1` validated statistically**, not exactly: chosen-action agreement
  rate + root-value/policy agreement vs sequential, and the **trap-suite blunder
  rate** on the E-Tier-1 consequential fixtures (`35:63`, `107:60`) — must not
  blunder more than sequential. Sweep `leaf_batch` to find where quality declines
  (KD's method; KD's answer was 6).

### 4c. Throughput benchmark (§5).

## 5. Benchmark methodology (the ≥20× gate)

- **Baseline:** the Python self-play loop (`phase_d`/`self_play` path) driving the
  Python `GumbelMCTS`, in its best config on the same box (it is GIL-limited —
  KD's threaded generation saturated ~4×).
- **Candidate:** the Rust multi-process workers + inference server on the same
  net/settings/box.
- **Equal settings:** identical sims, top_k, net checkpoint, closed mode, same
  hardware (the target cloud box: multi-core + the training GPU).
- **Metric:** **aggregate games/s** across all workers (primary); leaves/s and
  **GPU utilization %** (diagnostics — low GPU util despite max workers is the
  trigger for F4.4 intra-worker coalescing). Report all three + the component
  breakdown (encode / IPC / GPU forward / tree).
- **Gate:** ≥20× aggregate games/s vs the Python loop at equal settings. Record
  the number + box spec in `PHASE_F.md` (KD's ~28× is the yardstick).

## 6. Sub-sequence (reordered 2026-07-21 — multi-process first)

The multi-process + inference-server path reuses the F3.4 searcher, so the big
searcher refactor is deferred and made conditional.

- **F4.1** — **inference server + multi-process self-play harness.** Server owns
  the GPU net, coalesces requests across workers into batches (batch-invariance
  check, §4a). Workers are processes running `closed_search_net` with a
  server-submit adapter (`leaf_batch=1`). Reuse the KD inference-service design.
- **F4.2** — **throughput benchmark** (§5): full-loop games/s + component
  breakdown (encode / IPC / GPU forward / tree) vs the Python loop; measure GPU
  utilization. This informs everything below (agreed: measure first, then decide).
- **F4.3** — **perf tuning** as profiles dictate: `py.detach` in the worker,
  server window/batch tuning, flat encode buffers, precomputed actor/legal,
  batched forced children (keep priors, kill the first-visit re-eval), worker
  count. Iterate to ≥20× (aim for headroom).
- **F4.4** *(conditional)* — **intra-worker virtual-loss coalescing.** The phase
  split (§3a) + virtual loss + `leaf_batch` sweep (§4b). Build **only if** F4.2
  shows the GPU underutilized despite max workers (i.e. cross-worker batching
  alone can't fill it).
- **F4.5** *(conditional)* — **tch/ONNX in-process** only if the Python/torch hop
  still dominates after coalescing (§3a of `AZ_PROJECT_PLAN`).

## 7. Open decisions (resolve as they arise, log here)

1. **Multi-process: yes, from the start** (2026-07-21, user — training on a rented
   multi-core box + powerful GPU). Cross-worker batching via the inference server
   is the primary, quality-free lever (§2b); more workers, not virtual loss, fills
   the GPU.
2. **Intra-worker virtual-loss coalescing:** works with Gumbel (§2a); deferred to
   F4.4 and only if needed. Tune `leaf_batch` by the KD sweep.
3. **Python-hop vs tch/ONNX in-process:** measure full-loop throughput +
   components first (agreed), decide at F4.5.

## 8. Reuse from Kingdomino
KD M5/M6: `allow_threads`/`py.detach`, in-process coalescing, `leaf_batch>1`
essential (single-descent per NN call caps at ~1.7k leaves/s; batching → ~46k).
Port the coalescing scaffolding; the Gumbel schedule (§2a) is the new part.
