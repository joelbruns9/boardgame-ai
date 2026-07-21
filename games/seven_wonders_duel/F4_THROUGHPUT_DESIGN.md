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

### 2a. Gumbel vs KD's PUCT (the key design question)

KD's coalescing was for a plain PUCT MCTS — every simulation is an independent
descent, trivially batchable. The 7WD root is **Gumbel + sequential halving**,
which is *structured*: a fixed candidate set gets a fixed sim budget per round,
and rounds depend on prior rounds' Q. Batching options:

- **(A) Batch within a round.** A round runs `candidates × per_action` forced-edge
  simulations; collect their leaves across the round, one NN batch, backprop, then
  do the candidate reduction. Simple; batch size ≈ round budget. Non-identical
  because intra-round backprops are deferred.
- **(B) Batch the whole search with virtual loss.** Treat each forced-edge sim as
  a parallel descent, apply virtual loss on the path, coalesce `leaf_batch`
  leaves regardless of round boundaries. Larger batches, more staleness.
- **(C) Sub-root batching only.** Keep the root schedule sequential; batch the
  leaves reached *below* the root per simulation. Small batches (tree depth),
  least deviation, least speedup.

**Recommendation: start with (A)** — it respects the halving structure, gives
useful batch sizes (round budget can be tens–hundreds of leaves), and its
deviation from sequential is bounded to one round. Revisit (B) with virtual loss
if profiles show batch sizes too small. Decide with F4.1 profiling, not up front.

## 3. Architecture (the reviewer's F4 blueprint)

Refactor the searcher into three phases so evaluation is batchable:

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

## 4. Validation strategy (no equivalence gate)

1. **`leaf_batch=1` == the sequential oracle, exactly.** The phase-split searcher
   with batch size 1 (no coalescing, no virtual loss) MUST reproduce the F3.3
   `search_closed` result + tree digest bit-for-bit (to 1e-9). This proves the
   refactor is correct before any batching changes behavior. This is the one hard
   gate F4 keeps.
2. **`leaf_batch>1` validated statistically**, not exactly:
   - chosen-action agreement rate vs the sequential searcher over many positions;
   - root-value / policy agreement within a tolerance;
   - **trap-suite blunder rate** on the E-Tier-1 consequential fixtures (esp.
     `35:63`, `107:60`) — the batched searcher must not blunder more than
     sequential. This ties F4 back to the real quality bar.
3. **Throughput benchmark** (§5).

## 5. Benchmark methodology (the ≥20× gate)

- **Baseline:** the Python self-play loop (`phase_d`/`self_play` path) driving the
  Python `GumbelMCTS` per move, single process.
- **Candidate:** the Rust batched searcher on the same net/settings.
- **Equal settings:** identical sims, top_k, net checkpoint, closed mode, CPU
  (and separately a GPU run if the net is GPU-resident — batching helps most
  there).
- **Metric:** games/s over a fixed number of self-play games (primary);
  leaves/s (diagnostic). Report both.
- **Gate:** ≥20× games/s at equal settings. Record the number + machine in
  `PHASE_F.md` (KD's ~28× is the yardstick).

## 6. Sub-sequence (each with its own check)

- **F4.0** — batched `Eval` interface (`evaluate_batch`); refactor `search_closed`
  into selection/eval/backprop phases; `leaf_batch=1` reproduces the F3.3 digest
  exactly (the §4.1 oracle gate).
- **F4.1** — coalescing loop (scheme A), `py.detach` GIL release + in-process
  batched eval; profile batch sizes; statistical agreement + trap blunder-rate
  vs sequential.
- **F4.2** — perf: flat encode buffers, precomputed actor/legal, batched forced
  children (keep priors); re-profile.
- **F4.3** — throughput benchmark harness; measure games/s vs the Python loop;
  iterate to ≥20% headroom over 20×.
- **F4.4** *(conditional)* — tch/ONNX in-process only if the Python hop still
  dominates.

## 7. Open decisions (resolve as they arise, log here)

1. Coalescing scheme A vs B (virtual loss) — F4.1 profiling.
2. Python-hop vs in-process NN — F4.4, profile-driven.
3. Multi-process self-play (KD saturated ~4× under the GIL; the coalescing path,
   not more threads, is the lever — M6) vs single-process batched. Likely
   single-process batched first, multi-process later if needed.

## 8. Reuse from Kingdomino
KD M5/M6: `allow_threads`/`py.detach`, in-process coalescing, `leaf_batch>1`
essential (single-descent per NN call caps at ~1.7k leaves/s; batching → ~46k).
Port the coalescing scaffolding; the Gumbel schedule (§2a) is the new part.
