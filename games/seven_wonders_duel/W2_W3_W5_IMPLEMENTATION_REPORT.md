# W2/W3/W5 implementation report

Date: 2026-07-30
Branch: `7wd-w2-w3-w5-memory-stats-gates`

## Grounding

This work prepares the Seven Wonders Duel loop for production L-model training
on an RTX 5090 host with more CPU and RAM than the development laptop. No L
model is trained locally. Laptop measurements are correctness evidence only and
do not select production cache, batching, slot, or gate-cap values.

The cloud launch explicitly uses:

- L = `d_model=384`, `layers=8`, `heads=6`
- bf16
- LR `5e-5`
- HOF fraction `0.15`

The code compatibility default for HOF remains `0.0`, preventing old resumes and
unrelated tests from silently changing their opponent distribution.

## W2 — memory: DONE

Implemented:

1. The Rust inference worker latches its terminal `PyErr`. A disconnected
   ticket or later submission re-raises the original exception instead of
   reporting that the worker dropped its response.
2. Every packed buffer uses pyo3's checked `PyByteArray::new_with`, so Python
   allocation failure returns `MemoryError` normally.
3. The example cache is bounded in calibrated retained bytes. The old
   `--example-cache-examples` option remains accepted and converts at the
   measured 17.8 KB/example; `--example-cache-gb` is the preferred interface.
4. Startup calibration compares RSS growth with the six arrays' raw `nbytes`,
   never allowing a factor below the measured 17.8/13.1 ratio.
5. `--memory-budget-gb`, `--vram-budget-gb`, and
   `--memory-headroom-gb` resolve against the runtime host. Pressure evicts the
   LRU cache first and records a `memory_pressure` event.
6. Gate admission estimates host RSS and physical VRAM before loading models.
   It raises a clean `MemoryError` if eviction cannot create safe headroom.
7. Generation, replay derivation, training, pre-gate, gate peak, and post-gate
   resource samples are retained. Gate exit performs GC and empties the CUDA
   allocator cache.
8. Pending-iteration recovery now removes the uncommitted buffer/candidate and
   stale optimizer moments after restoring `latest` and `current_best`, making
   the iteration that crashed inside a gate genuinely restartable.

## W3 — shared statistics: DONE

Implemented `games/az_loop/stats.py` and schema version 2:

- typed generation, outcomes, replay, training, gates, resources, model, and
  game-specific groups;
- controller-side validation before a row is committed;
- schema-v1 readers remain supported;
- Rust scheduler counters previously discarded are now retained;
- allocated and physical VRAM are separate fields;
- every v2 row attributes results to width, layers, heads, parameter count, and
  precision;
- the 7WD extension records victory length/age, science, military, and
  draft/wonder signals;
- those game-specific observations are now collected during the verified replay
  that creates trainable positions and cached with the examples. The former
  stats-only full replay and its unattributed wall time are removed;
- every new record carries an explicit `opponent_type`: `current_best`, `hof`,
  `bot`, or `hof_bot`. A HOF assignment shadowed by a bot is recorded as unused
  and does not inflate realized HOF traffic;
- `az_report.py` emits exclusive opponent counts plus realized HOF and bot
  shares, with legacy `kind` values mapped for historical rows.

`tools/az_report.py` reads either schema. Against `laptop_training_03`, it
reproduced the ten-iteration outcome blocks, the gate ladder, and an
early-to-late throughput change of -24.5%. V1 correctly reports that batch,
forced-row, and RSS diagnostics are unavailable; v2 rows contain those fields.

## W5 — gates

Implemented:

1. Model checkpoints are read once when constructing a gate spec; the redundant
   metadata `torch.load` is gone.
2. A model gate puts both seat legs and all queued pairs into one rolling Rust
   scheduler. A single inference worker persists for the gate.
3. Network routing belongs to the searcher seat, so every leaf in one search
   uses the mover's checkpoint. Both seat legs run concurrently in one pool.
4. Gate concurrency has a separate `--gate-slots` setting.
5. Promotion evidence uses independent seat pairs. A pair result is exactly
   `{0, 0.5, 1}`; draws count as half.
6. Promote when Wilson LCB exceeds 0.50, stop futile candidates when UCB falls
   below 0.50, revert at the cap below 0.48, and otherwise continue probation.
7. The Rust scheduler stops queue admission only after a complete pair
   boundary. It keeps the same inference worker through that decision.
8. Promotion prefixes are excluded from Elo. Anchors run fixed-N, never
   early-stop, and are the unbiased Elo input.
9. `w5_gate_bench.py` requires at least three sizes, disables early stopping for
   timing, records moves/game and resources, fits fixed plus marginal cost, and
   rejects a non-L checkpoint unless explicitly used for a functional smoke.

The production gate cap is deliberately not selected on the laptop. The RTX
5090 acceptance runs 200/400/800 games and selects the largest cap satisfying
the configured wall-time share.

## Local acceptance results

The copied historical S run
`runs/laptop_training_03_w2_resume` reconciled the interrupted iteration 70
and completed iterations 70 through 89. The original `laptop_training_03`
directory was not modified.

- 20/20 resumed iterations committed, all with valid schema-v2 stats.
- The process exited normally, wrote the final 8,000-game replay buffer, and
  produced zero bytes on stderr.
- The explicit historical HOF fraction was `0.0`, and all 8,000 generated games
  reported `hof=0`.
- Four 200-game Wilson gates completed. All four ended in cap probation; their
  optional-stopping prefixes were not added to Elo.
- The calibrated cache reached its byte ceiling by iteration 86 and then held
  between 4,449,755,487 and 4,449,955,727 bytes.
- Across the cache-saturated iterations 86-89, post-training RSS ranged from
  6,462,849,024 to 6,563,774,464 bytes, a 1.55% range around the mean. Peak RSS
  was 6,563,774,464 bytes and no memory-pressure event fired.
- `az_report.py` read the mixed v1/v2 90-iteration run, reproduced all outcome
  blocks and gates, and exposed batch, forced-row, RSS, learning, and reference
  mix fields for the v2 continuation.

This establishes the failure mode W2 was created to address: the interrupted
run is restartable, allocation errors remain intelligible, and retained host
memory converges to a bound. L paths have already run on the laptop's 8 GB GPU;
the production RTX 5090 has 32 GB. A short exact-geometry L/bf16 preflight will
record target-device headroom, but no adaptive VRAM estimator or separate
60-iteration L soak is required to close W2.

Final verification:

- opponent-attribution focused Python tests: 92 passed;
- full `games/seven_wonders_duel` plus `games/az_loop` Python suites: 660
  passed, one existing tensor-conversion warning;
- Rust unit tests: 16 passed, with one non-fatal test-helper dead-code warning;
- `git diff --check`: clean.

## Judgment calls

1. **HOF 0.15 is explicit, not a global default.** This keeps old resumes
   stable while ensuring cloud acceptance measures the launch distribution.
2. **Resource budgets resolve at runtime.** Hard-coding laptop RAM/VRAM would
   make the safety mechanism wrong on the target host.
3. **No laptop L training.** The historical S resume closes the W2 regression.
   The first exact-geometry cloud iteration is an operational smoke check with
   telemetry, not a deferred L memory soak.
4. **Gate timing disables evidence stopping.** Otherwise a lucky early boundary
   would make a nominal 800-game timing row incomparable with a 200-game row.
5. **Promotion games do not update Elo.** Optional stopping biases their score;
   fixed-N anchors provide the valid ladder.
6. **Fixed-N gate cap remains provisional.** Statistical appeal alone does not
   settle its share of RTX 5090 training wall time.
7. **Removed a duplicate W1 Rust test initializer field.** `cargo test` exposed
   two `net_by_player` initializers in one fixture. Removing the duplicate was
   necessary to make the pre-existing W1 routing tests compile; runtime code was
   unchanged by that cleanup.
8. **Cold optimizer restart after interrupted training.** The recovery journal
   did not snapshot Adam moments. Deleting them avoids applying iteration-70
   moments to restored iteration-69 weights; a cold warmup is slower but sound.
9. **Stopped a disposable CPU plumbing smoke.** It began competing with the
   required historical resume before producing a row, so it was terminated.
   The completed resume subsequently supplied 20 real schema-v2 rows; the
   disposable small-model timing was neither needed nor retained as evidence.

## Acceptance boundary

W2 is complete. It proves error propagation, calibrated cache enforcement,
bounded host-memory behavior, deterministic gate cleanup, and restart recovery.
The cloud launch still runs one exact-geometry L/bf16 preflight and retains
RSS/physical-VRAM telemetry, but those are operational checks rather than an
open W2 engineering project.

W3 is complete: schema, reporting, opponent attribution, and replay-cost
accounting are closed. W5's decision rule and full promotion-plus-anchor cost
budget remain under review;
RTX 5090 gate measurements and production cap selection belong to that W5 work.

Commands and expected artifacts are in
`W2_W3_W5_CLOUD_ACCEPTANCE.md`.
