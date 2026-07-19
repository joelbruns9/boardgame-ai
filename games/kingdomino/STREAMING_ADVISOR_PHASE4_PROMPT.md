# Implementation prompt — Streaming Advisor Phase 4: live validation + review followups

Context: `games/kingdomino/STREAMING_ADVISOR_PLAN.md`. Phases 0–3 shipped in commit
`e7b993f` ("Ship streaming advisor Phases 0-3"): job API + single-active-job registry,
Rust `AdvisorSearchHandle` (cumulative tree + public-state TT), milestone fragility,
exact-as-job with `ExactTimeout` → in-job NN continuation, one extension controller for
nn/exact/auto. All code-review findings from Phases 0–3 were fixed and verified; every
automated suite is green (Python 17, extension 9, Rust 48).

This phase closes what review could NOT verify from code: live end-to-end behavior, plus
three residuals accepted pending live evidence.

**Division of labor — this phase has three stages with a hard boundary in the middle:**
- **Stage A (agent): instrumentation.** Make the user's play session measure itself.
  No live play, no BGA interaction of any kind.
- **Stage B (USER ONLY): live play.** The user plays a real BGA game against the
  checklist below. The agent NEVER opens, joins, starts, or acts in a game. If Stage B
  hasn't happened yet, Stage A is the whole task — stop there and hand over.
- **Stage C (agent): analysis + conditional fixes.** Read the collected data, decide the
  triggers, implement only what fires.

## Stage A — instrumentation (agent)

Add lightweight, permanent-or-flag-gated diagnostics so Stage B produces numbers instead
of impressions. Keep each one trivial; no behavior changes.

1. **Scrape counter (extension):** count `readPageState()` calls per minute, split by
   trigger (mutation-debounce vs interval backstop vs settle loop). Surface via a
   `console.log` summary every 60 s and/or a debug field on the overlay. Feeds item 3.
2. **Premature-settle counter (extension):** log every streaming job stopped for
   `position-change`, with its age at stop. Jobs stopped < ~1.5 s after start are the
   premature-settle signature. Feeds item 4.
3. **Job lifecycle timing (server):** log per job: engine path taken, `solving_exact`
   duration, per-child-solve count/duration on the exact path, time-to-first-snapshot,
   and fallback events. Feeds item 2 (cold-solve duration) and the checklist.
4. **Checklist artifact:** print/write the Stage B checklist (below) somewhere the user
   can tick through during play (a markdown file is fine).

Validate Stage A with the existing suites (counters must not break
`tests/test_auto_settle.mjs` contexts) and a brief local server run. Then STOP and hand
over to the user.

## Stage B — live play checklist (USER, at the keyboard)

Play at least one real game with streaming on. Tick each item; jot timings where asked:

- [ ] NN turn: overlay refreshes ~1 s; sims count is cumulative; stability hint only
      appears after real progress.
- [ ] Fragility: `testing_fragility` shows at the milestone; ⚠ rows update afterwards;
      note whether "robust stale" ever fires.
- [ ] Deck 5 → 4: exact job starts with no gap and no legacy fallback;
      `Solving exact endgame...` pending UI; finished exact snapshot has margins/swindle
      chips and NO sims line.
- [ ] Timeout fallback: with `exact_max_secs` set tiny (e.g. 0.5) on a hard endgame, the
      SAME job continues as an NN stream with the timeout banner.
- [ ] Stop / Good enough: once during `running`, once during `solving_exact`; note stop
      latency.
- [ ] Subjective: did you ever want to act before the first refresh converged? (This is
      the Item-3 reopen criterion — note it either way.)
- [ ] Save/copy the browser console + server log output for Stage C.

## Stage C — analysis + conditional fixes (agent)

From Stage B's logs and checklist, decide each item explicitly and record the numbers:

2. **Partial exact snapshots** (Phase 3's skipped stretch). Trigger: cold-cache exact
   solves left the pending UI empty > ~3 s. If triggered: worker publishes an
   intermediate snapshot after root + each child solve (`partial: true` until done),
   strictly additive to the response shape, same version/poll mechanics — the per-child
   margin cache means this is assembly cadence only. Otherwise record durations and skip.
3. **MutationObserver scope.** Trigger: scrape counts are heavy (order thousands/min
   during animations, or visible jank). If triggered: scope the observer to the game
   container(s), keeping the 2.5 s interval backstop unchanged so a scoping mistake
   degrades to the old cadence, not silence. Otherwise record the counts and close
   permanently.
4. **Premature-settle churn.** Trigger: more than ~1 premature-settle stop per game. If
   triggered: one additional confirming read before job START only — do not slow
   re-renders or the game-log path (already on the outer cadence). Otherwise close.
- **Game-log audit:** verify appended records contain only settled decision states.
- **Checklist failures:** fix at the mechanism level, not with UI patches.

Any Stage C code change gets a regression test in the matching suite
(`test_web_app_streaming.py` / `tests/test_auto_settle.mjs`) and a rerun of Python +
extension suites; Rust changes also rebuild (`maturin develop --release`) + rerun Rust
tests. Record the trigger measurements and decisions in this file or the plan doc.

## Out of scope — accepted/closed decisions, do not relitigate

- **Single global job policy** (documented at `web_app.py` near the cancel loop) —
  multi-table is out of scope.
- **Chunk-restart fallback's quadratic work** — deliberate degraded mode for a missing
  Rust handle only.
- **Legacy one-shot path behind `streaming: false`** — kept for backward compat.
- **Item 3 cross-move subtree reuse** — measured and closed-deferred
  (`STREAMING_ADVISOR_PLAN.md`, end of Item 3): 54% recurrence is structural
  (2 decisions/reveal-window in 2p); 27.5 s median same-window gap vs
  seconds-to-convergence means a warm start never binds. Reopen only if Stage B's
  subjective check says you routinely act before the first refresh converges.
- `/api/recommend` request/response contract stays frozen; draft-matrix mini-searches
  stay freshly rooted (never the handle TT).
