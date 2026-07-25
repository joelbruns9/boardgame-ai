# Review Request — Game-Agnostic Soft-Gate Training Controller

**Commit:** `f5bb92b` on branch `codex/reply-pilot`
**Plan of record:** `games/seven_wonders_duel/KINGDOMINO_TRAINING_CONVERSION_PLAN.md`
**Status:** Milestones 1–5 + M7 end-to-end validation complete; M6 skipped (optional).
70 conversion tests; full `seven_wonders_duel/` + `az_loop` suite **364 passed**.

---

## 1. What this changes and why

The Phase D loop treated `current_best.pt` as all three of: the protected best,
the self-play generator, and the initialization point for the next candidate.
An SPRT `continue`/`reject` left `current_best` unchanged, so the next iteration
reloaded the same model and **discarded the candidate's learning**. In
`runs/laptop_training_10h_01`, all eight candidates independently restarted from
the random iteration `-1` checkpoint.

This change introduces three explicit checkpoint identities — a rolling learner
(`latest.pt`), a promotion-protected best (`current_best.pt`), and immutable
candidate snapshots — driven by a **game-agnostic soft-gate controller** in
`games.az_loop`. Seven Wonders Duel becomes its first client via a thin adapter.
`strict_gate` (the default) reproduces the legacy lifecycle exactly.

**The fix, verified:** a real `soft_gate` CUDA run produces `bootstrap_promote`
→ `probation`, with `latest.pt` advancing past `current_best.pt` (cumulative
learning) while `current_best` stays frozen.

---

## 2. How to review (suggested order)

Review the **new shared package first** — it is self-contained and pure — then
the thin 7WD adapter, then the `phase_d.py` wiring.

| Order | File | ~Lines | What to check |
|---|---|---|---|
| 1 | `games/az_loop/training_control.py` | 281 | The pure transition state machine. No I/O; exhaustively testable. |
| 2 | `games/az_loop/checkpoint_lifecycle.py` | 135 | Atomic file ops on opaque checkpoints. |
| 3 | `games/az_loop/contract.py` | 139 | `LifecycleAdapter` Protocol + typed request/result objects. |
| 4 | `games/az_loop/run_controller.py` | 440 | Orchestration + resume. The heart of the change. |
| 5 | `games/az_loop/run_log.py` | 171 | The `run.log` tee. |
| 6 | `games/seven_wonders_duel/training_adapter.py` | 140 | Delegates to existing `PhaseDLoop` methods. |
| 7 | `games/seven_wonders_duel/phase_d.py` (diff) | — | Mode dispatch + adapter hooks (see §4). |
| 8 | Tests | — | `test_az_loop_controller.py`, `test_run_log.py`, new `test_phase_d.py` cases. |

---

## 3. ⚠️ Bundled pre-existing code — do not review as new

`HEAD` (`bfcd92c`) is old. The working tree carried a large amount of
**previously-uncommitted Phase F work** that interleaves with the conversion in
the same files, so it is unavoidably part of this commit:

- `phase_d.py`: the Rust generation path (`_generate_iteration_rust`), Rust gate
  waves (`_rust_model_gate_waves`, `_rust_bot_gate_waves`), streaming training
  log (`_append_training_log`, `_sync_training_log`), warm-buffer plumbing, etc.
- `test_phase_d.py`: the pre-existing Rust/process/seed-buffer tests.
- `rust_bridge.py`: **entirely** pre-existing (added only so `phase_d.py`'s
  imports resolve at this commit).

**Please review only the conversion diff.** The conversion-owned surface is:
the five new `az_loop` modules, `training_adapter.py`, the two new test files,
and — within `phase_d.py`/`test_phase_d.py` — the items listed in §4/§6.

---

## 4. Conversion-owned changes in `phase_d.py`

1. **Mode dispatch.** `run()` → `_run_strict_gate()` (the original loop,
   untouched) for `strict_gate`, else `_run_controller(mode)`.
2. **Learner source (the actual fix).** `train_candidate(..., source_checkpoint=None)`
   loads from the controller-selected checkpoint (`latest.pt`) instead of always
   `current_best`. Default preserves legacy behavior.
3. **Gate opponent.** `promotion_gate(candidate, *, opponent=None)` — the
   controller passes `latest.pt` vs `current_best.pt` explicitly.
4. **Bootstrap split.** `initialize(*, bootstrap_checkpoint=True)` — the
   controller path skips legacy `current_best` creation (the adapter does it).
5. **Replay ops (M4).** `filter_warm_records_by_staleness` (pure), `_load_warm_buffer`,
   `_autosave_replay_buffer`.
6. **CLI + `RunLog` wrap (M5).** New flags; `main()` wraps the run in `RunLog`.
7. **`_PhaseDRunStore`** adapts the manifest + training log to the controller's
   `RunStore` protocol.

---

## 5. Design decisions worth a look

- **`strict_gate` stays on the legacy path**, not routed through the controller.
  Rationale: `test_anchor_failure_does_not_block_current_best_promotion` pins the
  exact legacy row schema and calls `run_iteration` directly. Routing strict
  through the controller would change that schema. This keeps "strict reproduces
  the old lifecycle" literally true.
- **Controller owns the atomic checkpoint lifecycle and autosave *scheduling*;
  the adapter owns the game-specific *writes*.** (Per the plan's ownership table.)
- **`run.log` is parent-process-only.** Self-play/gate workers run in spawned
  processes (`core.run_jobs_in_processes`); a parent `threading.Lock` cannot
  serialize them, so workers never write `run.log` — they return results and the
  orchestrator logs. This was review fix #1 folded into the plan.
- **HOF archives the *outgoing* best, and never an `untrained` one** (so
  bootstrap's random weights don't enter the protected HOF). This differs from
  the old `promote()`, which archived the *new* best.
- **The revert counter is per gate-check, not per iteration.** `not_scheduled`
  iterations (when `--promotion-every > 1`) don't touch it.
- **Atomicity:** all rolling-file writes are temp-beside-dest + `os.replace` +
  `fsync` (same-volume atomic rename on Windows too).
- **Resume** reconstructs `GeneratorState` from the last row's `control_state`,
  verifies on-disk `latest`/`best` SHA-256, and refuses a missing/tampered
  checkpoint or a mode switch.

---

## 6. Please scrutinize these specifically

1. **Promotion cadence counting** (`RunController._promotion_scheduled`): the
   "gate every Nth eligible iteration" ordinal excludes the bootstrap row and is
   derived from persisted rows so it is resume-stable. Is the ordinal logic
   correct across a resume that lands mid-cadence?
2. **Generator-source aliasing** (`GeneratorState.as_row`/`from_row`): the row
   stores `next_generator_source` (what the *next* iteration generates with),
   deliberately distinct from the top-level `generator_source` (what generated
   *this* iteration). Confirm resume reads the right one.
3. **`run.log` under process-pool workers**: confirm no worker path writes to the
   tee, and that `capsys`-style stream swapping restores cleanly on exception.
4. **Autosave crash-safety claim**: `test_atomic_save_leaves_previous_export_readable_on_interrupted_write`
   simulates a mid-write crash. Is the guarantee (partial `.tmp` never replaces
   the last valid export) airtight?
5. **Open design question (unchanged code):** the soft gate maps SPRT `reject` →
   `revert` directly. The 7WD gate is `SPRT(0.5-δ, 0.5+δ)`, so `reject` ≈ "score
   confidently ≤ 0.47." This couples the promote and revert thresholds to one δ,
   unlike Kingdomino's independent `soft_gate_revert_win_rate=0.48`. At
   `gate_max_games=50` a `reject` is rare, so the revert path is exercised mainly
   by synthetic tests. Is the coupling acceptable, or do we want an independent
   revert threshold? (Flagged in the plan review; not changed here.)

---

## 7. Test coverage

70 conversion tests (full suite `364 passed`):

- **`games/test_az_loop_controller.py`** (26) — fake in-memory byte-file adapter;
  checkpoint bytes chain `init|t0|t1…` so cumulative lineage, exact hashes, and
  resume are asserted directly. Includes the full synthetic transition matrix
  (`bootstrap → accept → continue → reject → continue` with exact SHA + generator
  identity), resume hash-mismatch/missing/mode-switch refusals, autosave cadence
  and non-fatal failure.
- **`games/test_run_log.py`** (6) — mirror/header/footer, append-on-resume,
  exception + KeyboardInterrupt capture + re-raise, disabled no-op, unopenable
  warn-and-continue.
- **`games/seven_wonders_duel/test_phase_d.py`** (new cases) — real-pipeline
  soft-gate: bootstrap ratchet, accept/promote + HOF archive, reject/revert
  generation switch, revert-reset, paired-SPRT-unchanged, golden training-log
  rows over all four actions, resume-no-dup, warm staleness, atomic save.

### How to verify locally

```bash
# Conversion tests
.venv/Scripts/python.exe -m pytest games/test_az_loop_controller.py \
  games/test_run_log.py games/test_az_loop.py \
  games/seven_wonders_duel/test_phase_d.py -q

# Full regression (≈4 min)
.venv/Scripts/python.exe -m pytest games/seven_wonders_duel/ \
  games/test_az_loop.py games/test_az_loop_controller.py games/test_run_log.py -q

# Real soft-gate smoke (CPU, ~40s): bootstrap escapes iteration -1
.venv/Scripts/python.exe -m games.seven_wonders_duel.phase_d \
  --run-dir /tmp/soft_smoke --plumbing-smoke \
  --selfplay-generator-mode soft_gate --bootstrap-policy auto_first_trained \
  --device cpu --generation-backend python --gate-backend python \
  --seed-games 6 --iterations 2
# then inspect /tmp/soft_smoke/run.log and training_log.jsonl
```

---

## 8. Out of scope / intentionally not done

- **M6** (persistent optimizer + fixed-step training) — optional per plan §9/§19;
  no evidence epoch-based training is a limitation. `on_learner_reset` is a 7WD
  no-op placeholder for it.
- **Search/engine/codec/encoder/Rust** — untouched. No `leaf_batch > 1`.
- **Kingdomino** — unchanged; used only as a behavioral reference.
- **The multi-hour laptop run** (M7 step 6) — operator's to launch; command in
  `training_parameters.md` → *Recommended soft-gate command*.
- The other ~43 dirty files in the tree (kingdomino, rust crate source, `f4_*.py`
  scratch) are **not** part of this commit.

---

*This review-request doc is itself uncommitted; commit or discard as you like.*
